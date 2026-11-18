"""
PathSnap — FastAPI Backend (Laptop 1)
api/main.py

Run with:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload

Evaluator's laptop connects via: http://<LAPTOP1_IP>:8000
"""

import os
import sys
import time
import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Allow imports from src/
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from features import (
    load_track, load_nodes, load_arcs, load_route,
    build_arc_geometry, get_candidate_arcs,
    compute_features_for_trip, FEATURE_COLS
)
from hmm import (
    build_road_graph, arc_to_nodes,
    viterbi_decode, detect_gaps, route_continuity
)
from metrics import evaluate_model

# ─────────────────────────────────────────────
#  Paths — adjust if your folder structure differs
# ─────────────────────────────────────────────

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
EVAL_TRIPS_DIR = DATA_DIR / "eval_test"      # Ground truth stays HERE
STUDENT_TRIPS_DIR = DATA_DIR / "student_test"
MODELS_DIR = BASE_DIR / "models"

# ─────────────────────────────────────────────
#  App setup
# ─────────────────────────────────────────────

app = FastAPI(
    title="PathSnap API",
    description="Map-matching server for PathSnap hackathon",
    version="1.0.0"
)

# Allow Laptop 2 (any origin on local network) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
#  Load all models at startup
# ─────────────────────────────────────────────

MODELS = {}

@app.on_event("startup")
def load_models():
    model_files = list(MODELS_DIR.glob("*.joblib"))
    if not model_files:
        print("⚠️  No models found in models/ — predictions will fail.")
        return
    for mf in model_files:
        name = mf.stem
        MODELS[name] = joblib.load(mf)
        print(f"  ✓ Loaded model: {name}")
    print(f"\n🚀 PathSnap API ready. {len(MODELS)} model(s) loaded.")


def get_model(model_name: str = None):
    if not MODELS:
        raise HTTPException(status_code=503, detail="No models loaded.")
    if model_name and model_name in MODELS:
        return MODELS[model_name], model_name
    # Default: use the first loaded model
    name = list(MODELS.keys())[0]
    return MODELS[name], name


# ─────────────────────────────────────────────
#  Helper: load a trip's files
# ─────────────────────────────────────────────

def load_trip_files(trip_dir: Path):
    track_files = list(trip_dir.glob("*.track"))
    node_files = list(trip_dir.glob("*.nodes"))
    arc_files = list(trip_dir.glob("*.arcs"))
    route_files = list(trip_dir.glob("*.route"))

    if not track_files:
        raise FileNotFoundError(f"No .track file in {trip_dir}")

    track_df = load_track(str(track_files[0]))
    nodes_df = load_nodes(str(node_files[0]))
    arcs_df = load_arcs(str(arc_files[0]))
    route_arcs = load_route(str(route_files[0])) if route_files else None
    return track_df, nodes_df, arcs_df, route_arcs


# ─────────────────────────────────────────────
#  Endpoints
# ─────────────────────────────────────────────

@app.get("/")
def root():
    return {"status": "ok", "models_loaded": list(MODELS.keys())}


@app.get("/trips")
def list_trips():
    """List all 10 evaluator test trip IDs."""
    trips = sorted([d.name for d in EVAL_TRIPS_DIR.iterdir() if d.is_dir()])
    return {"trips": trips, "count": len(trips)}


@app.get("/models")
def list_models():
    """List all loaded models."""
    return {"models": list(MODELS.keys())}


@app.post("/predict/{trip_id}")
def predict(trip_id: str, model_name: str = None, use_hmm: bool = True):
    """
    Run map-matching for a given evaluator test trip.

    Returns:
        - predicted arc sequence
        - GeoJSON LineString of predicted route
        - GPS track GeoJSON
        - inference time
    """
    trip_dir = EVAL_TRIPS_DIR / trip_id
    if not trip_dir.exists():
        raise HTTPException(status_code=404, detail=f"Trip '{trip_id}' not found.")

    print(f"\n[REQUEST] /predict/{trip_id}  model={model_name}  use_hmm={use_hmm}")
    t_start = time.time()

    try:
        track_df, nodes_df, arcs_df, ground_truth = load_trip_files(trip_dir)
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))

    arc_geom = build_arc_geometry(nodes_df, arcs_df)
    arc_node_map = arc_to_nodes(arcs_df)
    graph = build_road_graph(arcs_df)

    # Build features (no route — inference mode)
    feat_df = compute_features_for_trip(track_df, arc_geom, route_arc_ids=None)
    model, used_model = get_model(model_name)

    # --- Per-fix emission probabilities ---
    emission_per_fix = []
    candidate_arcs_per_fix = []

    for fix_idx in sorted(feat_df['fix_index'].unique()):
        fix_rows = feat_df[feat_df['fix_index'] == fix_idx]
        if fix_rows.empty:
            emission_per_fix.append({})
            candidate_arcs_per_fix.append([])
            continue

        X_fix = fix_rows[FEATURE_COLS].values
        arc_ids = fix_rows['arc_id'].tolist()

        try:
            proba = model.predict_proba(X_fix)[:, 1]  # P(correct arc)
        except Exception:
            proba = model.predict(X_fix).astype(float)

        emission_per_fix.append(dict(zip(arc_ids, proba.tolist())))
        candidate_arcs_per_fix.append(arc_ids)

    # --- Decode ---
    if use_hmm:
        gap_set = detect_gaps(track_df)
        predicted_arcs = viterbi_decode(
            emission_per_fix, candidate_arcs_per_fix,
            arc_node_map, graph, gap_indices=gap_set
        )
    else:
        # Classifier-only: pick argmax emission per fix
        predicted_arcs = []
        for emits, cands in zip(emission_per_fix, candidate_arcs_per_fix):
            if not emits:
                predicted_arcs.append(None)
            else:
                predicted_arcs.append(max(emits, key=emits.get))

    t_elapsed = time.time() - t_start

    # --- Build GeoJSON for predicted route ---
    arc_geom_lookup = {a['arc_id']: a for a in arc_geom}
    predicted_coords = []
    for arc_id in predicted_arcs:
        if arc_id is not None and arc_id in arc_geom_lookup:
            a = arc_geom_lookup[arc_id]
            predicted_coords.append([a['lon_a'], a['lat_a']])
            predicted_coords.append([a['lon_b'], a['lat_b']])

    gps_coords = [[row['longitude'], row['latitude']]
                  for _, row in track_df.iterrows()]

    print(f"[DONE] Trip={trip_id}  Model={used_model}  HMM={use_hmm}  "
          f"Time={t_elapsed:.2f}s  Fixes={len(predicted_arcs)}")

    # NOTE: ground_truth is intentionally NOT returned to the client.
    # Metrics are computed server-side in the /metrics endpoint.

    return {
        "trip_id": trip_id,
        "model_used": used_model,
        "use_hmm": use_hmm,
        "inference_time_sec": round(t_elapsed, 3),
        "n_fixes": len(predicted_arcs),
        "predicted_arcs": [int(a) if a is not None else -1 for a in predicted_arcs],
        "predicted_route_geojson": {
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": predicted_coords},
            "properties": {"trip_id": trip_id, "model": used_model}
        },
        "gps_track_geojson": {
            "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": gps_coords},
            "properties": {"trip_id": trip_id}
        },
    }


@app.get("/metrics/{trip_id}")
def compute_metrics(trip_id: str, model_name: str = None, use_hmm: bool = True):
    """
    Compute evaluation metrics for a trip.
    Ground truth is loaded server-side — NEVER sent to client.
    """
    trip_dir = EVAL_TRIPS_DIR / trip_id
    if not trip_dir.exists():
        raise HTTPException(status_code=404, detail=f"Trip '{trip_id}' not found.")

    print(f"\n[REQUEST] /metrics/{trip_id}  model={model_name}  use_hmm={use_hmm}")

    # Rerun prediction (or could cache — for simplicity, rerun)
    pred_response = predict(trip_id, model_name=model_name, use_hmm=use_hmm)
    predicted_arcs = pred_response["predicted_arcs"]

    # Load ground truth SERVER-SIDE only
    track_df, nodes_df, arcs_df, ground_truth = load_trip_files(trip_dir)
    arc_node_map = arc_to_nodes(arcs_df)

    if ground_truth is None:
        return {"error": "No ground truth available for this trip."}

    # Align lengths
    n = min(len(predicted_arcs), len(ground_truth))
    y_pred = [p if p != -1 else -999 for p in predicted_arcs[:n]]
    y_true = ground_truth[:n]

    correct = sum(p == t for p, t in zip(y_pred, y_true))
    top1_acc = correct / n if n > 0 else 0.0

    # Route continuity of prediction
    rc = route_continuity([p if p != -999 else None for p in y_pred], arc_node_map)

    # Macro F1 (binary: correct arc vs not)
    from sklearn.metrics import f1_score
    y_true_bin = [1] * n
    y_pred_bin = [1 if p == t else 0 for p, t in zip(y_pred, y_true)]
    mf1 = f1_score(y_true_bin, y_pred_bin, average='macro', zero_division=0)

    metrics = {
        "trip_id": trip_id,
        "model_used": pred_response["model_used"],
        "use_hmm": use_hmm,
        "n_fixes": n,
        "top1_accuracy": round(top1_acc, 4),
        "macro_f1": round(mf1, 4),
        "route_continuity": round(rc, 4),
        "inference_time_sec": pred_response["inference_time_sec"],
    }

    print(f"[METRICS] {metrics}")
    return metrics
