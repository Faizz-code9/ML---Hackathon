# PathSnap — Map Matching with Machine Learning

> PES University · UE24CS352A Machine Learning Hackathon

## Overview

PathSnap reconstructs realistic vehicle trajectories by map-matching noisy GPS measurements to an underlying road network graph using a combination of supervised ML classifiers and an HMM/Viterbi sequential decoder.

---

## Two-Laptop Setup

### Laptop 1 — Backend / API Server

```powershell
# 1. Install dependencies
pip install fastapi uvicorn scikit-learn lightgbm xgboost optuna joblib networkx pandas numpy folium

# 2. Download trained models from Kaggle output → place in models/

# 3. Find your local IP
ipconfig   # Look for IPv4 Address (e.g. 192.168.1.5)

# 4. Start the server
cd api
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

### Laptop 2 — Frontend / Evaluator UI

```powershell
# 1. Install dependencies
pip install streamlit streamlit-folium requests folium

# 2. Set Laptop 1's IP (replace with actual IP from above)
$env:LAPTOP1_IP = "192.168.1.5"

# 3. Start the UI
cd ui
streamlit run app.py
```

The Streamlit app will open at `http://localhost:8501`. Select a trip and model, then click **Run Prediction**.

---

## Project Structure

```
PathSnap/
├── data/
│   ├── training/        ← 80 trips (track + nodes + arcs + route)
│   ├── student_test/    ← 10 trips (no route — offline testing)
│   └── eval_test/       ← 10 trips (evaluator reserved — DO NOT TOUCH)
├── src/
│   ├── features.py      ← Feature engineering pipeline
│   ├── hmm.py           ← HMM + Viterbi decoder
│   └── metrics.py       ← Evaluation metrics
├── api/
│   └── main.py          ← FastAPI server (Laptop 1)
├── ui/
│   └── app.py           ← Streamlit UI (Laptop 2)
├── models/              ← Saved model .joblib files
├── notebooks/           ← Kaggle training notebook
├── maps/                ← Exported Folium maps
├── report/              ← Technical report
├── .gitignore
└── README.md
```

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check + loaded models |
| `GET` | `/trips` | List 10 evaluator test trip IDs |
| `GET` | `/models` | List loaded model names |
| `POST` | `/predict/{trip_id}?model_name=lgbm&use_hmm=true` | Run map-matching, return predicted arcs + GeoJSON |
| `GET` | `/metrics/{trip_id}?model_name=lgbm&use_hmm=true` | Server-side metrics vs. ground truth |

---

## Training on Kaggle

1. Upload `PathSnap_student_materials.zip` as a Kaggle Dataset
2. Upload `notebooks/pathsnap_training.ipynb` as a Kaggle Notebook
3. Enable **GPU Accelerator** (T4 x2)
4. Run all cells
5. Download output `models/` directory and place in this repo's `models/`

---

## Models Trained

| # | Model | CV Macro F1 | Notes |
|---|-------|------------|-------|
| 1 | Logistic Regression | — | Baseline |
| 2 | k-NN (k=5) | — | Instance-based |
| 3 | Random Forest | — | Ensemble trees |
| 4 | LightGBM (Optuna-tuned) | — | **Best performer** |
| 5 | MLP Neural Network | — | 3 hidden layers |

---

## Team

PES University — UE24CS352A
