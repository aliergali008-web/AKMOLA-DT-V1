"""
AKMOLA-DT-V1 — Feature Engineering Pipeline
=============================================
GeoFeatureCalculator: computes terrain derivatives and stacks them
into the [C=6, H, W] tensor required by EL-FNO.

Channels: [Elevation, Slope_Mag, Aspect_X, Aspect_Y, Ruggedness, Log_Flow_Accum]

Region adaptation: steppe terrain with micro-depressions.
Ref ТЗ §3 (pipeline.features)
"""

import logging
from typing import Tuple, Optional

import numpy as np
from scipy.ndimage import convolve, generic_filter, uniform_filter

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import (
    TERRAIN_CHANNELS, EPSILON, TRI_WINDOW_SIZE,
    PIXEL_SIZE, DTYPE, CHANNEL_NAMES,
)

logger = logging.getLogger(__name__)


class GeoFeatureCalculator:
    """
    Vectorized geo-feature calculators (NumPy/SciPy).
    Produces the 6-channel terrain tensor for Lagrangian Guide input.

    In steppe/plain conditions, emphasis is on:
    - Fine gradients (not steep slopes)
    - Micro-depressions and local water accumulation
    - Terrain ruggedness from field ditches, gullies, and embankments
    """

    def __init__(self, pixel_size: float = PIXEL_SIZE, epsilon: float = EPSILON):
        self.pixel_size = pixel_size
        self.eps = epsilon

    # ─── 1. Slope Vector ─────────────────────────────────────────

    def compute_slope(
        self, elevation: np.ndarray, method: str = "sobel"
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute slope magnitude, aspect_cos (∂z/∂x), aspect_sin (∂z/∂y).

        Parameters
        ----------
        elevation : np.ndarray [H, W]
        method : str
            'sobel' for sharper edge detection,
            'zevenbergen' for smoother 3×3 estimation.

        Returns
        -------
        slope_mag : np.ndarray [H, W]
        aspect_cos : np.ndarray [H, W]
        aspect_sin : np.ndarray [H, W]
        """
        if method == "sobel":
            # Sobel kernels
            kx = np.array([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]], dtype=DTYPE) / (8.0 * self.pixel_size)
            ky = np.array([[-1, -2, -1],
                           [ 0,  0,  0],
                           [ 1,  2,  1]], dtype=DTYPE) / (8.0 * self.pixel_size)
        elif method == "zevenbergen":
            # Zevenbergen-Thorne: simpler central differences
            kx = np.array([[ 0, 0, 0],
                           [-1, 0, 1],
                           [ 0, 0, 0]], dtype=DTYPE) / (2.0 * self.pixel_size)
            ky = np.array([[ 0, -1, 0],
                           [ 0,  0, 0],
                           [ 0,  1, 0]], dtype=DTYPE) / (2.0 * self.pixel_size)
        else:
            raise ValueError(f"Unknown method: {method}. Use 'sobel' or 'zevenbergen'.")

        p = convolve(elevation, kx, mode="reflect")  # ∂z/∂x
        q = convolve(elevation, ky, mode="reflect")  # ∂z/∂y

        slope_mag = np.sqrt(p ** 2 + q ** 2).astype(DTYPE)
        denom = slope_mag + self.eps
        aspect_cos = (p / denom).astype(DTYPE)
        aspect_sin = (q / denom).astype(DTYPE)

        return slope_mag, aspect_cos, aspect_sin

    # ─── 2. Terrain Ruggedness Index (TRI) ───────────────────────

    def compute_tri(
        self, elevation: np.ndarray, window_size: int = TRI_WINDOW_SIZE
    ) -> np.ndarray:
        """
        Terrain Ruggedness Index: standard deviation of elevation
        in a window_size × window_size neighborhood.

        In steppe context, TRI highlights micro-topographic roughness
        (ditches, field edges, embankments) that control deformation
        magnitude in the Lagrangian Guide.

        Parameters
        ----------
        elevation : np.ndarray [H, W]
        window_size : int (default: 5)

        Returns
        -------
        tri : np.ndarray [H, W]
        """
        # mean of z
        mean_z = uniform_filter(elevation.astype(np.float64), size=window_size, mode="reflect")
        # mean of z^2
        mean_z2 = uniform_filter((elevation.astype(np.float64)) ** 2, size=window_size, mode="reflect")
        # variance = E[z^2] - (E[z])^2
        variance = np.maximum(mean_z2 - mean_z ** 2, 0.0)
        tri = np.sqrt(variance).astype(DTYPE)

        return tri

    # ─── 3. Curvature (Laplacian) ────────────────────────────────

    def compute_curvature(self, elevation: np.ndarray) -> np.ndarray:
        """
        Laplacian curvature: ∂²z/∂x² + ∂²z/∂y².
        Identifies convergence/divergence zones at small scale.

        Positive curvature → concave (flow converges)
        Negative curvature → convex (flow diverges)

        Parameters
        ----------
        elevation : np.ndarray [H, W]

        Returns
        -------
        curvature : np.ndarray [H, W]
        """
        # Second derivative kernels
        d2x = np.array([[1, -2, 1]], dtype=DTYPE) / (self.pixel_size ** 2)
        d2y = np.array([[1], [-2], [1]], dtype=DTYPE) / (self.pixel_size ** 2)

        laplacian = (
            convolve(elevation, d2x.reshape(1, 3), mode="reflect") +
            convolve(elevation, d2y.reshape(3, 1), mode="reflect")
        )

        return laplacian.astype(DTYPE)

    # ─── 4. Flow Accumulation (Hydro-conditioning) ───────────────

    def compute_flow_accumulation(self, elevation: np.ndarray) -> np.ndarray:
        """
        Simplified D8 flow accumulation.
        Full pipeline: fill sinks → D8 direction → log(accum + 1).

        In steppe catchments, small depressions and ditches define
        local accumulation paths — this feature is CRITICAL for
        flood prediction accuracy.

        Parameters
        ----------
        elevation : np.ndarray [H, W]

        Returns
        -------
        log_flow_accum : np.ndarray [H, W]
        """
        # Try using richdem if available
        try:
            import richdem as rd
            dem_rd = rd.rdarray(elevation.astype(np.float64), no_data=-9999.0)
            rd.FillDepressions(dem_rd, in_place=True)
            accum = rd.FlowAccumulation(dem_rd, method="D8")
            log_flow = np.log1p(np.array(accum, dtype=np.float64))
            return log_flow.astype(DTYPE)
        except ImportError:
            pass

        # Fallback: pure NumPy D8 implementation
        return self._d8_flow_accumulation_numpy(elevation)

    def _d8_flow_accumulation_numpy(self, elevation: np.ndarray) -> np.ndarray:
        """
        Pure NumPy D8 flow accumulation (fallback when richdem unavailable).

        Steps:
        1. Fill sinks (iterative)
        2. Compute D8 flow direction
        3. Accumulate flow
        """
        h, w = elevation.shape
        dem = elevation.copy().astype(np.float64)

        # Step 1: Simple sink filling (iterative raise)
        dem = self._fill_sinks(dem)

        # Step 2: D8 flow direction
        # Directions: 0=E, 1=NE, 2=N, 3=NW, 4=W, 5=SW, 6=S, 7=SE
        dy = [-1, -1,  0,  1, 1, 1, 0, -1]
        dx = [ 0,  1,  1,  1, 0, -1, -1, -1]
        dist = [1.0, np.sqrt(2), 1.0, np.sqrt(2), 1.0, np.sqrt(2), 1.0, np.sqrt(2)]

        flow_dir = np.full((h, w), -1, dtype=np.int8)

        for r in range(h):
            for c in range(w):
                max_drop = 0.0
                max_d = -1
                for d in range(8):
                    nr, nc = r + dy[d], c + dx[d]
                    if 0 <= nr < h and 0 <= nc < w:
                        drop = (dem[r, c] - dem[nr, nc]) / dist[d]
                        if drop > max_drop:
                            max_drop = drop
                            max_d = d
                flow_dir[r, c] = max_d

        # Step 3: Flow accumulation via topological sort
        accum = np.ones((h, w), dtype=np.float64)

        # Sort cells by elevation (highest first)
        flat_indices = np.argsort(dem.ravel())[::-1]

        for idx in flat_indices:
            r, c = divmod(idx, w)
            d = flow_dir[r, c]
            if d >= 0:
                nr, nc = r + dy[d], c + dx[d]
                if 0 <= nr < h and 0 <= nc < w:
                    accum[nr, nc] += accum[r, c]

        log_flow = np.log1p(accum).astype(DTYPE)
        return log_flow

    def _fill_sinks(self, dem: np.ndarray, max_iter: int = 1000) -> np.ndarray:
        """Simple iterative sink-filling algorithm."""
        h, w = dem.shape
        filled = dem.copy()

        for iteration in range(max_iter):
            changed = False
            for r in range(1, h - 1):
                for c in range(1, w - 1):
                    neighbors = []
                    for dr in [-1, 0, 1]:
                        for dc in [-1, 0, 1]:
                            if dr == 0 and dc == 0:
                                continue
                            neighbors.append(filled[r + dr, c + dc])
                    min_neighbor = min(neighbors)
                    if filled[r, c] < min_neighbor:
                        filled[r, c] = min_neighbor + 1e-5
                        changed = True
            if not changed:
                break

        return filled

    # ─── Build Full Tensor ───────────────────────────────────────

    def build_tensor(
        self,
        elevation: np.ndarray,
        slope_method: str = "sobel",
        normalize: bool = True,
    ) -> np.ndarray:
        """
        Build the 6-channel terrain tensor from raw elevation.

        Parameters
        ----------
        elevation : np.ndarray [H, W]
        slope_method : str
            'sobel' or 'zevenbergen'
        normalize : bool
            If True, normalize each channel to [0, 1] range.

        Returns
        -------
        tensor : np.ndarray [C=6, H, W]
            Channels: [Elevation, Slope_Mag, Aspect_X, Aspect_Y, Ruggedness, Log_Flow_Accum]
        """
        logger.info("Computing terrain features (%s)...", slope_method)

        # Compute all features
        slope_mag, aspect_cos, aspect_sin = self.compute_slope(elevation, slope_method)
        tri = self.compute_tri(elevation)
        log_flow = self.compute_flow_accumulation(elevation)

        # Stack into tensor
        channels = [
            elevation,
            slope_mag,
            aspect_cos,
            aspect_sin,
            tri,
            log_flow,
        ]

        tensor = np.stack(channels, axis=0).astype(DTYPE)

        if normalize:
            tensor = self._normalize_channels(tensor)

        logger.info(
            "Terrain tensor built: shape=%s, channels=%s",
            tensor.shape, CHANNEL_NAMES,
        )
        return tensor

    def _normalize_channels(self, tensor: np.ndarray) -> np.ndarray:
        """Normalize each channel to [0, 1] independently."""
        result = tensor.copy()
        for c in range(tensor.shape[0]):
            ch = result[c]
            vmin, vmax = ch.min(), ch.max()
            if vmax - vmin > self.eps:
                result[c] = (ch - vmin) / (vmax - vmin)
            else:
                result[c] = np.zeros_like(ch)
        return result


# ─── CLI Entry Point ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    from pipeline.ingest import generate_synthetic_dem

    parser = argparse.ArgumentParser(description="AKMOLA-DT-V1 Feature Engineering")
    parser.add_argument("--input", type=str, help="Path to elevation .npy tile")
    parser.add_argument("--synthetic", action="store_true",
                        help="Use synthetic DEM for testing")
    parser.add_argument("--method", choices=["sobel", "zevenbergen"], default="sobel")
    parser.add_argument("--size", type=int, default=256,
                        help="Synthetic DEM size (only with --synthetic)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    calc = GeoFeatureCalculator()

    if args.synthetic:
        logger.info("Using synthetic DEM (%dx%d)...", args.size, args.size)
        elevation, _ = generate_synthetic_dem(height=args.size, width=args.size)
    elif args.input:
        elevation = np.load(args.input)
    else:
        parser.error("Provide --input or use --synthetic")

    tensor = calc.build_tensor(elevation, slope_method=args.method)
    print(f"\nTensor shape: {tensor.shape}")
    print(f"Channels: {CHANNEL_NAMES}")
    for i, name in enumerate(CHANNEL_NAMES):
        ch = tensor[i]
        print(f"  [{i}] {name:20s}  min={ch.min():.4f}  max={ch.max():.4f}  mean={ch.mean():.4f}")
