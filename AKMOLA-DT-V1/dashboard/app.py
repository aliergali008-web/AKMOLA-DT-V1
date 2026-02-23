"""
AKMOLA-DT-V1 — Web Dashboard (Flask Backend)
==============================================
Visual evaluation interface for the Digital Twin pipeline.
Provides tile visualization, feature heatmaps, model inference,
and acceptance criteria checking.

Usage:
    python -m dashboard.app
    → Open http://localhost:5000 in browser
"""

import json
import logging
import sys
from pathlib import Path
from io import BytesIO
import base64

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import (
    TILES_DIR, CHANNEL_NAMES, TARGET_CRS, CHIP_SIZE,
    TERRAIN_CHANNELS, DTYPE,
)

try:
    from flask import Flask, render_template, jsonify, request, send_file
    from flask_cors import CORS
    HAS_FLASK = True
except ImportError:
    HAS_FLASK = False

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

logger = logging.getLogger(__name__)

# ─── Flask App ───────────────────────────────────────────────────

def create_app(tiles_dir: str = None) -> "Flask":
    if not HAS_FLASK:
        raise ImportError("Flask is required: pip install flask flask-cors")

    tiles_path = Path(tiles_dir) if tiles_dir else TILES_DIR

    app = Flask(
        __name__,
        template_folder=str(Path(__file__).parent / "templates"),
        static_folder=str(Path(__file__).parent / "static"),
    )
    CORS(app)

    # ─── API Routes ──────────────────────────────────────────────

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/api/tiles")
    def list_tiles():
        """List all available tiles with metadata."""
        tiles = []
        for meta_file in sorted(tiles_path.glob("tile_*_meta.json")):
            with open(meta_file, "r", encoding="utf-8") as f:
                meta = json.load(f)
            tiles.append(meta)
        return jsonify({"tiles": tiles, "count": len(tiles), "crs": TARGET_CRS})

    @app.route("/api/tile/<tile_id>")
    def get_tile(tile_id: str):
        """Get tile metadata and channel statistics."""
        meta_path = tiles_path / f"tile_{tile_id}_meta.json"
        if not meta_path.exists():
            return jsonify({"error": f"Tile {tile_id} not found"}), 404

        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        npy_path = tiles_path / f"tile_{tile_id}.npy"
        if npy_path.exists():
            data = np.load(npy_path)
            meta["live_stats"] = {}
            for i, name in enumerate(CHANNEL_NAMES):
                if i < data.shape[0]:
                    ch = data[i]
                    meta["live_stats"][name] = {
                        "min": float(ch.min()),
                        "max": float(ch.max()),
                        "mean": float(ch.mean()),
                        "std": float(ch.std()),
                    }

        return jsonify(meta)

    @app.route("/api/tile/<tile_id>/channel/<int:channel_idx>")
    def get_channel_image(tile_id: str, channel_idx: int):
        """Render a single channel as a heatmap PNG."""
        if not HAS_MATPLOTLIB:
            return jsonify({"error": "matplotlib required"}), 500

        npy_path = tiles_path / f"tile_{tile_id}.npy"
        if not npy_path.exists():
            return jsonify({"error": f"Tile {tile_id} not found"}), 404

        data = np.load(npy_path)
        if channel_idx >= data.shape[0]:
            return jsonify({"error": f"Channel {channel_idx} out of range"}), 400

        channel = data[channel_idx]
        cmap = request.args.get("cmap", "terrain" if channel_idx == 0 else "viridis")

        fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=100)
        im = ax.imshow(channel, cmap=cmap, interpolation="nearest")
        ax.set_title(CHANNEL_NAMES[channel_idx] if channel_idx < len(CHANNEL_NAMES) else f"Channel {channel_idx}")
        plt.colorbar(im, ax=ax, shrink=0.8)
        ax.set_xlabel("X (pixels)")
        ax.set_ylabel("Y (pixels)")

        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor="#1a1a2e")
        plt.close(fig)
        buf.seek(0)

        return send_file(buf, mimetype="image/png")

    @app.route("/api/tile/<tile_id>/all_channels")
    def get_all_channels(tile_id: str):
        """Render all 6 channels as a grid image."""
        if not HAS_MATPLOTLIB:
            return jsonify({"error": "matplotlib required"}), 500

        npy_path = tiles_path / f"tile_{tile_id}.npy"
        if not npy_path.exists():
            return jsonify({"error": f"Tile {tile_id} not found"}), 404

        data = np.load(npy_path)
        n_channels = min(data.shape[0], len(CHANNEL_NAMES))

        fig, axes = plt.subplots(2, 3, figsize=(15, 10), dpi=100)
        fig.suptitle(f"Tile {tile_id} — Terrain Features", fontsize=16, color="white")
        fig.patch.set_facecolor("#1a1a2e")

        cmaps = ["terrain", "hot", "RdBu_r", "RdBu_r", "YlOrRd", "Blues"]

        for i, ax in enumerate(axes.flat):
            if i < n_channels:
                im = ax.imshow(data[i], cmap=cmaps[i], interpolation="nearest")
                ax.set_title(CHANNEL_NAMES[i], color="white", fontsize=12)
                plt.colorbar(im, ax=ax, shrink=0.8)
            else:
                ax.axis("off")
            ax.tick_params(colors="white")

        plt.tight_layout()
        buf = BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", facecolor="#1a1a2e")
        plt.close(fig)
        buf.seek(0)

        return send_file(buf, mimetype="image/png")

    @app.route("/api/model/test")
    def test_model():
        """Run a quick model forward pass and return results."""
        try:
            import torch
            from model.core import EL_FNO_Model, model_summary

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = EL_FNO_Model(width=32, modes=8, n_layers=2).to(device)
            info = model_summary(model)

            # Try loading a real tile
            tile_files = list(tiles_path.glob("tile_*.npy"))
            if tile_files:
                tile_data = np.load(tile_files[0])
                h, w = tile_data.shape[1], tile_data.shape[2]
                terrain_t = torch.from_numpy(tile_data).unsqueeze(0).float().to(device)
                state_t = torch.zeros(1, 2, h, w, device=device)
            else:
                h, w = 64, 64
                terrain_t = torch.randn(1, 6, h, w, device=device)
                state_t = torch.zeros(1, 2, h, w, device=device)

            with torch.no_grad():
                output = model(state_t, terrain_t)

            return jsonify({
                "status": "success",
                "model_params": info,
                "input_shape": {
                    "state": list(state_t.shape),
                    "terrain": list(terrain_t.shape),
                },
                "output_shape": list(output.shape),
                "prediction_stats": {
                    "min": float(output.min()),
                    "max": float(output.max()),
                    "mean": float(output.mean()),
                },
                "device": str(device),
            })
        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500

    @app.route("/api/model/predict/<tile_id>")
    def predict_tile(tile_id: str):
        """Run model inference on a specific tile and return heatmap."""
        try:
            import torch
            from model.core import EL_FNO_Model

            npy_path = tiles_path / f"tile_{tile_id}.npy"
            if not npy_path.exists():
                return jsonify({"error": f"Tile {tile_id} not found"}), 404

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = EL_FNO_Model(width=32, modes=8, n_layers=2).to(device)
            model.eval()

            tile_data = np.load(npy_path)
            terrain_t = torch.from_numpy(tile_data).unsqueeze(0).float().to(device)
            h, w = tile_data.shape[1], tile_data.shape[2]
            state_t = torch.zeros(1, 2, h, w, device=device)

            with torch.no_grad():
                prediction = model(state_t, terrain_t)

            pred_np = prediction.cpu().numpy()[0, 0]  # [H, W]

            if not HAS_MATPLOTLIB:
                return jsonify({
                    "tile_id": tile_id,
                    "prediction_stats": {
                        "min": float(pred_np.min()),
                        "max": float(pred_np.max()),
                        "mean": float(pred_np.mean()),
                    }
                })

            fig, axes = plt.subplots(1, 3, figsize=(18, 6), dpi=100)
            fig.suptitle(f"Tile {tile_id} — Model Prediction", fontsize=16, color="white")
            fig.patch.set_facecolor("#1a1a2e")

            # Elevation
            axes[0].imshow(tile_data[0], cmap="terrain", interpolation="nearest")
            axes[0].set_title("Elevation", color="white")

            # Flow Accumulation
            axes[1].imshow(tile_data[5], cmap="Blues", interpolation="nearest")
            axes[1].set_title("Flow Accumulation", color="white")

            # Prediction (Water Depth)
            im = axes[2].imshow(pred_np, cmap="YlGnBu", interpolation="nearest")
            axes[2].set_title("Predicted Water Depth", color="white")
            plt.colorbar(im, ax=axes[2], shrink=0.8)

            for ax in axes:
                ax.tick_params(colors="white")

            plt.tight_layout()
            buf = BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight", facecolor="#1a1a2e")
            plt.close(fig)
            buf.seek(0)

            return send_file(buf, mimetype="image/png")

        except Exception as e:
            return jsonify({"status": "error", "message": str(e)}), 500

    @app.route("/api/qa/check")
    def qa_check():
        """Run acceptance criteria checks on all tiles."""
        results = {
            "tiles_checked": 0,
            "dimension_pass": True,
            "physics_warnings": [],
            "details": [],
        }

        for npy_file in sorted(tiles_path.glob("tile_*.npy")):
            if "_meta" in npy_file.stem:
                continue
            tile_id = npy_file.stem.replace("tile_", "")
            data = np.load(npy_file)
            results["tiles_checked"] += 1

            detail = {"tile_id": tile_id, "shape": list(data.shape), "checks": {}}

            # Dimension check: should be [6, 256, 256] or [6, H, W]
            if data.ndim != 3 or data.shape[0] != TERRAIN_CHANNELS:
                detail["checks"]["dimension"] = "FAIL"
                results["dimension_pass"] = False
            else:
                detail["checks"]["dimension"] = "PASS"

            # NaN check
            nan_count = int(np.isnan(data).sum())
            detail["checks"]["nan_free"] = "PASS" if nan_count == 0 else f"FAIL ({nan_count} NaN)"

            # Flow accumulation sanity (channel 5)
            if data.shape[0] >= 6:
                flow = data[5]
                if np.any(flow < 0):
                    warn = f"Tile {tile_id}: negative flow accumulation values"
                    results["physics_warnings"].append(warn)
                    detail["checks"]["flow_positive"] = "WARN"
                else:
                    detail["checks"]["flow_positive"] = "PASS"

            results["details"].append(detail)

        results["overall"] = "PASS" if results["dimension_pass"] and len(results["physics_warnings"]) == 0 else "WARN"
        return jsonify(results)

    @app.route("/api/pipeline/summary")
    def pipeline_summary():
        """Get pipeline run summary if available."""
        summary_path = tiles_path / "pipeline_summary.json"
        if summary_path.exists():
            with open(summary_path, "r", encoding="utf-8") as f:
                return jsonify(json.load(f))
        return jsonify({"status": "no pipeline run found"})

    return app


# ─── CLI Entry Point ─────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="AKMOLA-DT-V1 Dashboard")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--tiles-dir", type=str, default=str(TILES_DIR))
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    app = create_app(tiles_dir=args.tiles_dir)

    print("=" * 60)
    print("  AKMOLA-DT-V1 | Evaluation Dashboard")
    print(f"  → http://{args.host}:{args.port}")
    print("=" * 60)

    app.run(host=args.host, port=args.port, debug=args.debug)
