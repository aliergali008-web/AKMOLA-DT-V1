# AKMOLA-DT-V1 — AI Digital Twin for Flood Prediction

**Region:** Akmola Oblast / Esil River / Astana  
**Architecture:** EL-FNO (Eulerian-Lagrangian Adaptive Fourier Neural Operator)

---

## Quick Start

### 1. Install Dependencies

```bash
cd AKMOLA-DT-V1
pip install -r requirements.txt
```

### 2. Run the Pipeline (Synthetic Data)

```bash
python -m tools.generate_synthetic
```

This generates a synthetic steppe DEM with micro-depressions, computes all 6 terrain features, and saves tiled tensors.

### 3. Open the Dashboard

```bash
python -m dashboard.app
```

Open **http://localhost:5000** in your browser.

### 4. Run Tests

```bash
python -m pytest tests/ -v
```

### 5. Test the Model

```bash
python -m model.core
```

---

## Project Structure

```
AKMOLA-DT-V1/
├── config.py                  # Central configuration
├── pipeline/
│   ├── ingest.py              # RasterProcessor (DEM → tiles)
│   ├── features.py            # GeoFeatureCalculator (6-channel tensor)
│   └── output.py              # TileWriter (save .npy + JSON metadata)
├── model/
│   └── core.py                # LagrangianGating + SpectralBlock2d + EL_FNO_Model
├── tools/
│   └── generate_synthetic.py  # Full pipeline on synthetic data
├── dashboard/
│   ├── app.py                 # Flask web server
│   └── templates/index.html   # Dashboard UI
├── tests/
│   ├── test_ingest.py
│   ├── test_features.py
│   └── test_model.py
├── data/processed/            # Generated tiles
└── requirements.txt
```

## Terrain Tensor Channels

| # | Channel | Description |
|---|---------|-------------|
| 0 | Elevation | Normalized DEM height |
| 1 | Slope_Mag | Gradient magnitude (Sobel) |
| 2 | Aspect_X | cos(aspect) direction |
| 3 | Aspect_Y | sin(aspect) direction |
| 4 | Ruggedness | TRI (5×5 window std dev) |
| 5 | Log_Flow_Accum | log(D8 flow accumulation + 1) |

## Dashboard Features

- **Overview** — Pipeline stats, CRS, tile counts
- **Tiles** — Visual heatmaps for all 6 feature channels
- **Model** — Run EL-FNO inference, view predicted water depth
- **QA** — Automated acceptance criteria checks

## Architecture: EL-FNO

```
Terrain Features → [LagrangianGating] → Grid Offsets
                                              ↓
State + Terrain + Offsets → [Feature Lifter] → [FNO Spectral Blocks ×4] → [Decoder] → Water Depth
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
