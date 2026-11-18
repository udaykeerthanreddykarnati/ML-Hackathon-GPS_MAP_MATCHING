# PathSnap: Distributed Map-Matching & Route Reconstruction

**PES University (RR Campus) — UE24CS352A: Machine Learning Hackathon | Section D | Team 11**  
Full technical report & live results: [`REPORT.md`](REPORT.md) / [`11.pdf`](11.pdf)

End-to-end machine learning and HMM/Viterbi route reconstruction pipeline for snapping noisy GPS trajectories (`.track`) onto directed road networks (`.nodes`, `.arcs`), deployed across a two-laptop architecture (FastAPI backend on Laptop 1, Evaluator UI on Laptop 2).

## Repository Structure

```text
├── api/
│   └── app.py                  # FastAPI backend (Laptop 1): inference + server-side evaluation
├── ui/
│   ├── index.html              # Evaluator web console (Laptop 2)
│   ├── styles.css
│   └── app.js
├── src/pathsnap/
│   ├── config.py               # Paths, constants, PATHSNAP_EVAL_DIR config
│   ├── geo.py                  # Per-trip WGS84 local metric projection & track kinematics
│   ├── data_loader.py          # Raw trip loaders (.track, .nodes, .arcs, optional .route)
│   ├── candidates.py           # Segment-densified cKDTree candidate generator
│   ├── alignment.py            # Offline monotonic DP alignment (.track -> .route)
│   ├── features.py             # 31 coordinate-invariant geometric, heading & graph features
│   ├── hmm.py                  # HMM emission/transition + bounded-Dijkstra Viterbi decoder
│   └── metrics.py              # Classification, Top-1, continuity & sequence LCS F1 metrics
├── scripts/
│   ├── 01_eda.py               # Inventory, EDA summary plots & Folium maps
│   ├── 01b_eda_extras.py       # Integrity, weak components, alignment & 0-30s dt diagnostics
│   ├── 02_train_baseline.py    # Feature caching, 5-fold GroupKFold Model 1 (LightGBM) + HMM eval
│   ├── 03_train_xgboost.py     # Model 2 (XGBoost Hist) CV + head-to-head comparison
│   ├── 04_train_remaining_models.py # Models 3-6 (LogReg, ExtraTrees, PyTorch MLP, Nystroem+SVC)
│   ├── 04b_train_random_forest.py   # Model 7 (Random Forest) CV + HMM eval & comparison update
│   ├── 05_bayesian_tuning.py   # 25-trial Optuna TPE + HMM grid tuning & bundle export
│   └── 06_finalize_deliverables.py  # Final tables, comparison plots & 5-layer Folium maps
├── artifacts/                  # Exported model bundle (scaler, classifier, HMM config)
├── outputs/                    # Generated plots, CSVs, Folium HTML maps & logs
└── requirements.txt
```

## Setup

```bash
uv venv --python /opt/homebrew/bin/python3.11 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Run EDA & Full Training Pipeline

```bash
.venv/bin/python scripts/01_eda.py
.venv/bin/python scripts/01b_eda_extras.py
.venv/bin/python scripts/02_train_baseline.py
.venv/bin/python scripts/03_train_xgboost.py
.venv/bin/python scripts/04_train_remaining_models.py
.venv/bin/python scripts/04b_train_random_forest.py
.venv/bin/python scripts/05_bayesian_tuning.py
.venv/bin/python scripts/06_finalize_deliverables.py
```

## Two-Laptop Execution

### Laptop 1 — FastAPI Model & Evaluation Server
Run on all network interfaces (`0.0.0.0:8000`) so Laptop 2 can reach it over LAN/hotspot. Point `PATHSNAP_EVAL_DIR` to the reserved evaluation folder when demonstrating with evaluator trips (defaults to `Student folder/student_test` during development):

```bash
export PATHSNAP_EVAL_DIR="/path/to/evaluator_test"
export PATHSNAP_EVAL_SPLIT_NAME="evaluator_test"
.venv/bin/uvicorn api.app:app --host 0.0.0.0 --port 8000
```

Live API request logs and timing/evaluation metrics print directly in the Laptop 1 terminal.

### Laptop 2 — Evaluator Web UI
Serve the static UI folder from Laptop 2 (no Python ML dependencies needed on Laptop 2):

```bash
python3 -m http.server 3000 --directory ui
```

Open `http://localhost:3000` in a browser, enter Laptop 1's IP in **Laptop 1 API URL** (e.g., `http://192.168.1.25:8000`), click **Connect**, and select any of the 10 evaluation trips.
