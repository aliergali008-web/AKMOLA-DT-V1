"""
AKMOLA-DT-V1 — Central Configuration
Project: AI Digital Twin for Flood Prediction
Region: Akmola Oblast / Esil River / Astana
"""

import numpy as np
from pathlib import Path

# ─── Project Paths ───────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed" / "utm43n_akmola"
TILES_DIR = PROCESSED_DIR / "tiles"
RESULTS_DIR = PROJECT_ROOT / "results"

# ─── Coordinate Reference System ────────────────────────────────
TARGET_CRS = "EPSG:32643"  # WGS84 / UTM zone 43N

# ─── Raster Processing ──────────────────────────────────────────
PIXEL_SIZE = 30.0           # meters (SRTM 1-arcsecond ≈ 30 m)
CHIP_SIZE = 256             # pixels per tile side
STRIDE = 128                # sliding window stride (50% overlap)
PAD_MODE = "reflect"        # padding mode for FFT stability
NODATA_VALUE = -9999.0
RESAMPLING_ELEVATION = "bilinear"
RESAMPLING_MASK = "nearest"

# ─── Feature Engineering ────────────────────────────────────────
TERRAIN_CHANNELS = 6        # [Elevation, Slope_Mag, Aspect_X, Aspect_Y, Ruggedness, Log_Flow_Accum]
EPSILON = 1e-8              # numerical stability for aspect computation
TRI_WINDOW_SIZE = 5         # Terrain Ruggedness Index window (5×5)

# ─── Model Architecture ─────────────────────────────────────────
STATE_CHANNELS = 2          # hydrodynamic state variables (e.g., water depth, velocity)
MODEL_WIDTH = 64            # hidden dimension of FNO layers
FNO_MODES = 16              # number of Fourier modes to keep
FNO_LAYERS = 4              # number of spectral blocks
LAGRANGIAN_HIDDEN = 32      # hidden dim for terrain guide network

# ─── Input/Output Tensor Shapes ─────────────────────────────────
# Input:  [Batch, STATE_CHANNELS + TERRAIN_CHANNELS + 2(offsets), CHIP_SIZE, CHIP_SIZE]
# i.e.    [B, 10, 256, 256]
# Output: [Batch, 1, CHIP_SIZE, CHIP_SIZE]  → predicted water depth
INPUT_CHANNELS = STATE_CHANNELS + TERRAIN_CHANNELS + 2  # = 10
OUTPUT_CHANNELS = 1

# ─── Data Types ──────────────────────────────────────────────────
DTYPE = np.float32

# ─── Channel Order ───────────────────────────────────────────────
CHANNEL_NAMES = [
    "Elevation",
    "Slope_Mag",
    "Aspect_X",
    "Aspect_Y",
    "Ruggedness",
    "Log_Flow_Accum",
]
