"""
AKMOLA-DT-V1 — Output Artifacts & Metadata
============================================
TileWriter: saves processed tiles (.npy) with sidecar JSON metadata.

Output structure:
    /data/processed/utm43n_akmola/tiles/
        tile_x15_y42.npy           # Tensor stack [C, H, W]
        tile_x15_y42_meta.json     # Metadata

Ref ТЗ §5
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import TARGET_CRS, TILES_DIR, DTYPE, CHANNEL_NAMES

logger = logging.getLogger(__name__)


class TileWriter:
    """
    Saves processed terrain tiles with metadata sidecar JSON.
    Ensures reproducible geo-referencing and statistics tracking.
    """

    def __init__(self, output_dir: Optional[str] = None, crs: str = TARGET_CRS):
        self.output_dir = Path(output_dir) if output_dir else TILES_DIR
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.crs = crs

    def save_tile(
        self,
        tile_data: np.ndarray,
        tile_id: str,
        bounds: List[float],
        affine_transform: List[float],
        stats: Optional[Dict[str, float]] = None,
        notes: str = "",
    ) -> Dict[str, str]:
        """
        Save a tile as .npy + sidecar JSON.

        Parameters
        ----------
        tile_data : np.ndarray [C, H, W]
            The 6-channel terrain tensor.
        tile_id : str
            Tile identifier, e.g., 'x15_y42'.
        bounds : list
            [left, bottom, right, top] in UTM coords.
        affine_transform : list
            [a, b, c, d, e, f] affine parameters.
        stats : dict, optional
            Pre-computed statistics (or auto-computed).
        notes : str
            Additional notes for the metadata.

        Returns
        -------
        paths : dict
            {'npy': str, 'json': str} paths to saved files.
        """
        # Auto-compute stats if not provided
        if stats is None:
            stats = self._compute_stats(tile_data)

        # Save .npy
        npy_path = self.output_dir / f"tile_{tile_id}.npy"
        np.save(npy_path, tile_data.astype(DTYPE))

        # Build metadata
        metadata = {
            "tile_id": tile_id,
            "bounds": bounds,
            "affine_transform": affine_transform,
            "crs": self.crs,
            "shape": list(tile_data.shape),
            "channels": CHANNEL_NAMES,
            "dtype": str(DTYPE),
            "stats": stats,
            "notes": notes,
        }

        # Save JSON
        json_path = self.output_dir / f"tile_{tile_id}_meta.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

        logger.info("Saved tile %s → %s", tile_id, npy_path)
        return {"npy": str(npy_path), "json": str(json_path)}

    def _compute_stats(self, tile_data: np.ndarray) -> Dict[str, float]:
        """
        Auto-compute per-channel statistics for QA metadata.
        """
        stats = {}

        if tile_data.ndim == 3:
            for i, name in enumerate(CHANNEL_NAMES):
                if i < tile_data.shape[0]:
                    ch = tile_data[i]
                    stats[f"mean_{name.lower()}"] = float(np.mean(ch))
                    stats[f"std_{name.lower()}"] = float(np.std(ch))
                    stats[f"min_{name.lower()}"] = float(np.min(ch))
                    stats[f"max_{name.lower()}"] = float(np.max(ch))
        else:
            stats["mean"] = float(np.mean(tile_data))
            stats["std"] = float(np.std(tile_data))

        # Special stats from ТЗ example
        if tile_data.ndim == 3 and tile_data.shape[0] >= 5:
            stats["mean_slope"] = float(np.mean(tile_data[1]))       # Slope_Mag
            stats["ruggedness_index"] = float(np.mean(tile_data[4]))  # Ruggedness

        return stats

    def load_tile(self, tile_id: str) -> Dict[str, Any]:
        """
        Load a tile and its metadata.

        Returns
        -------
        result : dict with 'data' (np.ndarray) and 'metadata' (dict)
        """
        npy_path = self.output_dir / f"tile_{tile_id}.npy"
        json_path = self.output_dir / f"tile_{tile_id}_meta.json"

        data = np.load(npy_path)

        with open(json_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        return {"data": data, "metadata": metadata}

    def list_tiles(self) -> List[str]:
        """List all tile IDs in the output directory."""
        tiles = []
        for f in sorted(self.output_dir.glob("tile_*_meta.json")):
            tile_id = f.stem.replace("tile_", "").replace("_meta", "")
            tiles.append(tile_id)
        return tiles


# ─── CLI Entry Point ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AKMOLA-DT-V1 Tile Manager")
    parser.add_argument("--list", action="store_true", help="List all saved tiles")
    parser.add_argument("--inspect", type=str, help="Inspect a tile by ID")
    parser.add_argument("--dir", type=str, default=str(TILES_DIR),
                        help="Tiles directory")
    args = parser.parse_args()

    writer = TileWriter(output_dir=args.dir)

    if args.list:
        tiles = writer.list_tiles()
        print(f"Found {len(tiles)} tiles:")
        for t in tiles:
            print(f"  {t}")

    elif args.inspect:
        result = writer.load_tile(args.inspect)
        print(f"Tile: {args.inspect}")
        print(f"Shape: {result['data'].shape}")
        print(f"Metadata:")
        print(json.dumps(result["metadata"], indent=2))
