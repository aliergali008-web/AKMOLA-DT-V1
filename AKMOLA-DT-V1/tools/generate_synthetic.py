"""
AKMOLA-DT-V1 — Synthetic Data Generator & Full Pipeline Runner
================================================================
Generates synthetic steppe DEM data, runs feature engineering,
and optionally runs model inference — all without real satellite data.

Usage:
    python -m tools.generate_synthetic
    python -m tools.generate_synthetic --run-model
"""

import json
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    TILES_DIR, CHIP_SIZE, STRIDE, PIXEL_SIZE,
    DTYPE, CHANNEL_NAMES, TARGET_CRS,
)
from pipeline.ingest import RasterProcessor, generate_synthetic_dem
from pipeline.features import GeoFeatureCalculator
from pipeline.output import TileWriter

logger = logging.getLogger(__name__)


def run_full_pipeline(
    dem_size: int = 512,
    chip_size: int = CHIP_SIZE,
    stride: int = STRIDE,
    output_dir: str = None,
    run_model: bool = False,
    seed: int = 42,
) -> dict:
    """
    Run the full AKMOLA-DT-V1 pipeline on synthetic data.

    Steps:
    1. Generate synthetic steppe DEM
    2. Tile the DEM
    3. Compute terrain features for each tile
    4. Save tiles with metadata
    5. (Optional) Run model inference

    Returns
    -------
    results : dict with summary statistics
    """
    output_dir = output_dir or str(TILES_DIR)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  AKMOLA-DT-V1 | Full Pipeline — Synthetic Data")
    print("=" * 60)

    # ─── Step 1: Generate Synthetic DEM ──────────────────────────
    print("\n[1/4] Generating synthetic steppe DEM...")
    elevation, transform = generate_synthetic_dem(
        height=dem_size, width=dem_size, seed=seed
    )
    print(f"  DEM shape: {elevation.shape}")
    print(f"  Elevation range: {elevation.min():.1f} — {elevation.max():.1f} m")
    print(f"  Transform: {transform}")

    # ─── Step 2: Tile the DEM ────────────────────────────────────
    print(f"\n[2/4] Tiling (chip={chip_size}, stride={stride})...")
    processor = RasterProcessor(chip_size=chip_size, stride=stride)
    tiles_raw = list(processor.tile_generator(elevation, transform))
    print(f"  Generated {len(tiles_raw)} tiles")

    # ─── Step 3: Feature Engineering ─────────────────────────────
    print("\n[3/4] Computing terrain features...")
    calc = GeoFeatureCalculator()
    writer = TileWriter(output_dir=output_dir)

    tile_results = []
    for i, tile_info in enumerate(tiles_raw):
        tile_elev = tile_info["data"]
        tensor = calc.build_tensor(tile_elev, normalize=True)

        # Save with metadata
        paths = writer.save_tile(
            tile_data=tensor,
            tile_id=tile_info["tile_id"],
            bounds=tile_info["bounds"],
            affine_transform=tile_info["affine_transform"],
            notes="Synthetic steppe DEM — micro-depressions and flow accumulation.",
        )

        tile_results.append({
            "tile_id": tile_info["tile_id"],
            "shape": list(tensor.shape),
            "paths": paths,
        })
        print(f"  [{i+1}/{len(tiles_raw)}] {tile_info['tile_id']} → {tensor.shape}")

    # ─── Step 4: Model Inference (optional) ──────────────────────
    model_results = None
    if run_model:
        print("\n[4/4] Running EL-FNO model inference...")
        try:
            import torch
            from model.core import EL_FNO_Model, model_summary

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = EL_FNO_Model().to(device)
            info = model_summary(model)
            print(f"  Model: {info['total_params_M']} parameters")

            # Run inference on first tile
            first_tile = np.load(tile_results[0]["paths"]["npy"])
            terrain_t = torch.from_numpy(first_tile).unsqueeze(0).to(device)
            state_t = torch.zeros(1, 2, chip_size, chip_size, device=device)

            with torch.no_grad():
                prediction = model(state_t, terrain_t)

            model_results = {
                "output_shape": list(prediction.shape),
                "prediction_range": [
                    float(prediction.min()),
                    float(prediction.max()),
                ],
                "device": str(device),
                "params": info,
            }
            print(f"  Output shape: {prediction.shape}")
            print(f"  Prediction range: [{prediction.min():.4f}, {prediction.max():.4f}]")

            # Save prediction
            pred_path = output_path / "prediction_sample.npy"
            np.save(pred_path, prediction.cpu().numpy())
            print(f"  Saved prediction → {pred_path}")

        except ImportError:
            print("  ⚠ PyTorch not available — skipping model inference")
    else:
        print("\n[4/4] Skipping model inference (use --run-model to enable)")

    # ─── Summary ─────────────────────────────────────────────────
    summary = {
        "dem_size": [dem_size, dem_size],
        "pixel_size_m": PIXEL_SIZE,
        "n_tiles": len(tile_results),
        "tile_shape": tile_results[0]["shape"] if tile_results else None,
        "channels": CHANNEL_NAMES,
        "crs": TARGET_CRS,
        "output_dir": str(output_path),
        "model_inference": model_results,
    }

    summary_path = output_path / "pipeline_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print(f"  ✓ Pipeline complete! {len(tile_results)} tiles saved to:")
    print(f"    {output_path}")
    print("=" * 60)

    return summary


# ─── CLI Entry Point ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="AKMOLA-DT-V1 Synthetic Pipeline Runner"
    )
    parser.add_argument("--size", type=int, default=512,
                        help="Synthetic DEM size (default: 512)")
    parser.add_argument("--chip", type=int, default=CHIP_SIZE,
                        help=f"Tile size (default: {CHIP_SIZE})")
    parser.add_argument("--stride", type=int, default=STRIDE,
                        help=f"Stride (default: {STRIDE})")
    parser.add_argument("--output", type=str, default=str(TILES_DIR),
                        help="Output directory")
    parser.add_argument("--run-model", action="store_true",
                        help="Also run model inference")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    run_full_pipeline(
        dem_size=args.size,
        chip_size=args.chip,
        stride=args.stride,
        output_dir=args.output,
        run_model=args.run_model,
        seed=args.seed,
    )
