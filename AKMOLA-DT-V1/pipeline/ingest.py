"""
AKMOLA-DT-V1 — Data Ingestion Pipeline
========================================
RasterProcessor: handles DEM reprojection, cropping, hole-filling,
and tiling with metadata sidecar generation.

Region: Akmola Oblast / Esil River / Astana
Ref ТЗ §2 (pipeline.ingest)
"""

import json
import logging
from pathlib import Path
from typing import Generator, Tuple, Optional, Dict, Any

import numpy as np
from scipy.ndimage import generic_filter

try:
    import rasterio
    from rasterio.warp import reproject, Resampling, calculate_default_transform
    from rasterio.transform import from_bounds, Affine
    from rasterio.merge import merge
    from rasterio.fill import fillnodata
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    # Minimal Affine fallback so generate_synthetic_dem / tile_generator
    # work without rasterio installed.
    from collections import namedtuple
    _AffineBase = namedtuple("Affine", ["a", "b", "c", "d", "e", "f"])
    class Affine(_AffineBase):  # type: ignore[no-redef]
        """Lightweight stand-in for rasterio.transform.Affine."""
        def __new__(cls, a, b, c, d, e, f):
            return super().__new__(cls, a, b, c, d, e, f)

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    TARGET_CRS, PIXEL_SIZE, CHIP_SIZE, STRIDE,
    PAD_MODE, NODATA_VALUE, DTYPE, TILES_DIR,
)

logger = logging.getLogger(__name__)


class RasterProcessor:
    """
    Handles DEM ingestion: reprojection, cropping, NaN-filling, and tiling.

    Strict geo-referencing: 1-pixel shift can corrupt physical interpretation,
    especially in steppe terrain where micro-depressions drive flow accumulation.
    """

    def __init__(
        self,
        target_crs: str = TARGET_CRS,
        pixel_size: float = PIXEL_SIZE,
        chip_size: int = CHIP_SIZE,
        stride: int = STRIDE,
    ):
        self.target_crs = target_crs
        self.pixel_size = pixel_size
        self.chip_size = chip_size
        self.stride = stride

    # ─── Core Methods ────────────────────────────────────────────

    def reproject_and_crop(
        self,
        src_path: str,
        bbox_utm: Tuple[float, float, float, float],
        local_dem_path: Optional[str] = None,
    ) -> Tuple[np.ndarray, Affine]:
        """
        Reproject a source DEM (e.g., SRTM) to the target CRS, crop to bbox,
        fill nodata holes via interpolation.

        Parameters
        ----------
        src_path : str
            Path to the source DEM raster (GeoTIFF).
        bbox_utm : tuple
            Bounding box in target CRS coords: (left, bottom, right, top).
        local_dem_path : str, optional
            Path to a higher-resolution local DEM to merge over SRTM.
            Local DEM takes priority (steppe adaptation: micro-depressions).

        Returns
        -------
        elevation : np.ndarray
            2D elevation array, float32, nodata filled.
        transform : Affine
            Affine transform for the output array.
        """
        if not HAS_RASTERIO:
            raise ImportError(
                "rasterio is required for raster processing. "
                "Install with: pip install rasterio"
            )

        left, bottom, right, top = bbox_utm

        # Calculate exact output dimensions at 30m resolution
        width = int(round((right - left) / self.pixel_size))
        height = int(round((top - bottom) / self.pixel_size))

        # Build precise affine transform (critical: exact 30.0m pixel size)
        dst_transform = Affine(
            self.pixel_size, 0.0, left,
            0.0, -self.pixel_size, top
        )

        dst_array = np.full((height, width), NODATA_VALUE, dtype=DTYPE)

        with rasterio.open(src_path) as src:
            reproject(
                source=rasterio.band(src, 1),
                destination=dst_array,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs=self.target_crs,
                resampling=Resampling.bilinear,
                dst_nodata=NODATA_VALUE,
            )

        # Merge local high-res DEM if available (steppe adaptation)
        if local_dem_path is not None:
            dst_local = np.full((height, width), NODATA_VALUE, dtype=DTYPE)
            with rasterio.open(local_dem_path) as local_src:
                reproject(
                    source=rasterio.band(local_src, 1),
                    destination=dst_local,
                    src_transform=local_src.transform,
                    src_crs=local_src.crs,
                    dst_transform=dst_transform,
                    dst_crs=self.target_crs,
                    resampling=Resampling.bilinear,
                    dst_nodata=NODATA_VALUE,
                )
            # Local DEM takes priority where valid
            valid_local = dst_local != NODATA_VALUE
            dst_array[valid_local] = dst_local[valid_local]
            logger.info("Merged local DEM (%d valid pixels)", valid_local.sum())

        # Fill nodata holes via interpolation (critical before derivatives)
        nodata_mask = dst_array == NODATA_VALUE
        if nodata_mask.any():
            mask = (~nodata_mask).astype(np.uint8)
            dst_array = fillnodata(dst_array, mask, max_search_distance=100)
            logger.info("Filled %d nodata pixels", nodata_mask.sum())

        # Replace any remaining NaN
        dst_array = np.nan_to_num(dst_array, nan=0.0)

        logger.info(
            "Reprojected to %s: %dx%d @ %.1fm",
            self.target_crs, width, height, self.pixel_size
        )
        return dst_array, dst_transform

    def tile_generator(
        self,
        array: np.ndarray,
        transform: Affine,
        stride: Optional[int] = None,
        chip_size: Optional[int] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        """
        Sliding-window tile generator with reflect padding for FFT stability.

        Yields dicts with:
        - 'data': np.ndarray of shape [chip_size, chip_size]
        - 'tile_id': str like 'x03_y07'
        - 'bounds': [left, bottom, right, top] in UTM
        - 'affine_transform': [a, b, c, d, e, f]
        - 'row': int, 'col': int (grid indices)

        Parameters
        ----------
        array : np.ndarray
            2D elevation array.
        transform : Affine
            Affine transform for the input array.
        stride : int, optional
            Step size (default from config: 128).
        chip_size : int, optional
            Tile size (default from config: 256).
        """
        stride = stride or self.stride
        chip_size = chip_size or self.chip_size

        h, w = array.shape

        # Pad array with reflect for FFT stability (FNO requirement)
        pad_h = max(0, chip_size - h % stride) if h % stride != 0 else 0
        pad_w = max(0, chip_size - w % stride) if w % stride != 0 else 0

        # Ensure we have enough padding for at least one full chip
        if h < chip_size:
            pad_h = chip_size - h
        if w < chip_size:
            pad_w = chip_size - w

        padded = np.pad(
            array,
            ((0, pad_h), (0, pad_w)),
            mode=PAD_MODE,
        )

        padded_h, padded_w = padded.shape
        tile_row = 0

        for y_start in range(0, padded_h - chip_size + 1, stride):
            tile_col = 0
            for x_start in range(0, padded_w - chip_size + 1, stride):
                chip = padded[y_start:y_start + chip_size, x_start:x_start + chip_size]

                # Compute geo-bounds for this tile
                # transform * (col, row) → (x, y)
                left = transform.c + x_start * transform.a
                top = transform.f + y_start * transform.e
                right = left + chip_size * transform.a
                bottom = top + chip_size * transform.e

                tile_transform = Affine(
                    transform.a, 0.0, left,
                    0.0, transform.e, top
                )

                tile_id = f"x{tile_col:02d}_y{tile_row:02d}"

                yield {
                    "data": chip.astype(DTYPE),
                    "tile_id": tile_id,
                    "bounds": [left, bottom, right, top],
                    "affine_transform": [
                        tile_transform.a, tile_transform.b, tile_transform.c,
                        tile_transform.d, tile_transform.e, tile_transform.f,
                    ],
                    "row": tile_row,
                    "col": tile_col,
                }

                tile_col += 1
            tile_row += 1

        logger.info(
            "Generated %dx%d tile grid (chip=%d, stride=%d)",
            tile_col, tile_row, chip_size, stride,
        )

    # ─── Convenience: process from file to tiles ─────────────────

    def process_dem(
        self,
        src_path: str,
        bbox_utm: Tuple[float, float, float, float],
        local_dem_path: Optional[str] = None,
    ) -> Generator[Dict[str, Any], None, None]:
        """Full pipeline: reproject → crop → fill → tile."""
        elevation, transform = self.reproject_and_crop(
            src_path, bbox_utm, local_dem_path
        )
        return self.tile_generator(elevation, transform)


# ─── Synthetic Data Support ──────────────────────────────────────

def generate_synthetic_dem(
    height: int = 512,
    width: int = 512,
    pixel_size: float = PIXEL_SIZE,
    origin: Tuple[float, float] = (405000.0, 4780512.0),
    seed: int = 42,
) -> Tuple[np.ndarray, Affine]:
    """
    Generate a synthetic DEM for testing.
    Simulates steppe terrain with micro-depressions and gentle slopes.

    Returns
    -------
    elevation : np.ndarray [H, W], float32
    transform : Affine
    """
    rng = np.random.RandomState(seed)

    # Base: gentle regional slope (steppe, ~0.5° tilt)
    y_coords = np.linspace(0, 1, height).reshape(-1, 1)
    x_coords = np.linspace(0, 1, width).reshape(1, -1)
    base = 300.0 + 15.0 * y_coords + 5.0 * x_coords  # gentle NW→SE gradient

    # Add broad undulations (large-scale terrain features)
    for _ in range(3):
        freq_x = rng.uniform(1, 4)
        freq_y = rng.uniform(1, 4)
        phase = rng.uniform(0, 2 * np.pi)
        amp = rng.uniform(2, 8)
        base += amp * np.sin(freq_x * x_coords * 2 * np.pi + phase) * \
                np.cos(freq_y * y_coords * 2 * np.pi + phase * 0.5)

    # Add micro-depressions (key steppe feature)
    n_depressions = rng.randint(5, 15)
    for _ in range(n_depressions):
        cx = rng.randint(20, width - 20)
        cy = rng.randint(20, height - 20)
        radius = rng.uniform(10, 40)
        depth = rng.uniform(0.5, 3.0)
        yy, xx = np.ogrid[:height, :width]
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        base -= depth * np.exp(-dist ** 2 / (2 * radius ** 2))

    # Add some noise (micro-topography)
    base += rng.normal(0, 0.3, (height, width))

    elevation = base.astype(DTYPE)

    transform = Affine(pixel_size, 0.0, origin[0], 0.0, -pixel_size, origin[1])

    return elevation, transform


# ─── CLI Entry Point ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AKMOLA-DT-V1 Data Ingestion")
    parser.add_argument("--input", type=str, help="Path to source DEM GeoTIFF")
    parser.add_argument("--bbox", type=float, nargs=4,
                        metavar=("LEFT", "BOTTOM", "RIGHT", "TOP"),
                        help="Bounding box in UTM coords")
    parser.add_argument("--local-dem", type=str, default=None,
                        help="Optional local high-res DEM")
    parser.add_argument("--output", type=str, default=str(TILES_DIR),
                        help="Output directory for tiles")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic DEM instead of real data")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.synthetic:
        logger.info("Generating synthetic DEM for testing...")
        elevation, transform = generate_synthetic_dem()
        processor = RasterProcessor()
        tiles = processor.tile_generator(elevation, transform)
    else:
        if not args.input or not args.bbox:
            parser.error("--input and --bbox are required (or use --synthetic)")
        processor = RasterProcessor()
        tiles = processor.process_dem(args.input, tuple(args.bbox), args.local_dem)

    count = 0
    for tile in tiles:
        tile_path = output_dir / f"tile_{tile['tile_id']}.npy"
        np.save(tile_path, tile["data"])

        meta_path = output_dir / f"tile_{tile['tile_id']}_meta.json"
        meta = {
            "tile_id": tile["tile_id"],
            "bounds": tile["bounds"],
            "affine_transform": tile["affine_transform"],
            "crs": TARGET_CRS,
            "shape": list(tile["data"].shape),
        }
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        count += 1

    logger.info("Saved %d tiles to %s", count, output_dir)
