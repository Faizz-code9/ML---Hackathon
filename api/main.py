"""
PathSnap — FastAPI Backend (Laptop 1)
api/main.py

Run with:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload

Evaluator's laptop connects via: http://<LAPTOP1_IP>:8000
"""

import sys
import time
import warnings
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

warnings.filterwarnings("ignore", category=UserWarning)

# Allow imports from src/
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from features import (
    load_track, load_nodes, load_arcs, load_route,
    build_arc_geometry, compute_features_for_trip, FEATURE_COLS
)
from hmm import (
    build_road_graph, arc_to_nodes,
    viterbi_decode, detect_gaps, route_continuity
)

# ─────────────────────────────────────────────
#  Paths
# ─────────────────────────────────────────────

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
EVAL_TRIPS_DIR = DATA_DIR / "eval_test"      # Ground truth stays HERE
STUDENT_TRIPS_DIR = DATA_DIR / "student_test"
MODELS_DIR = BASE_DIR / "models"


def get_active_trips_dir() -> Path:
    """Use eval_test/ if it contains trip folders; otherwise fall back to student_test/."""
    if EVAL_TRIPS_DIR.exists() and any(d.is_dir() for d in EVAL_TRIPS_DIR.iterdir()):
        return EVAL_TRIPS_DIR
    return STUDENT_TRIPS_DIR


# ─────────────────────────────────────────────
#  App setup
# ─────────────────────────────────────────────

app = FastAPI(
    title="PathSnap API",
    description="Map-matching server for PathSnap hackathon",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
#  In-memory stores (models + prediction cache)
# ─────────────────────────────────────────────

MODELS = {}
PRED_CACHE = {}   # (trip_id, model_name, use_hmm) -> pred_response dict


@app.on_event("startup")
def load_models():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_files = sorted(MODELS_DIR.glob("*.joblib"))
    if not model_files:
        print("⚠️  No models found in models/ — predictions will fail until .joblib files are added.")
        return
    for mf in model_files:
        name = mf.stem
        MODELS[name] = joblib.load(mf)
        print(f"  ✓ Loaded model: {name}")
    active_dir = get_active_trips_dir()
    print(f"\n🚀 PathSnap API ready. {len(MODELS)} model(s) loaded. Active trips dir: {active_dir.name}/")


def get_model(model_name: str = None):
    if not MODELS:
        raise HTTPException(status_code=503, detail="No models loaded in models/ directory.")
    if model_name and model_name in MODELS:
        return MODELS[model_name], model_name
    # Prefer tuned lgbm if present
    for preferred in ("lgbm_tuned", "lgbm", "random_forest"):
        if preferred in MODELS:
            return MODELS[preferred], preferred
    name = list(MODELS.keys())[0]
    return MODELS[name], name


def load_trip_files(trip_dir: Path):
    track_files = list(trip_dir.glob("*.track"))
    node_files  = list(trip_dir.glob("*.nodes"))
    arc_files   = list(trip_dir.glob("*.arcs"))
    route_files = list(trip_dir.glob("*.route"))

    if not (track_files and node_files and arc_files):
        raise FileNotFoundError(f"Missing required trip files in {trip_dir}")

    track_df   = load_track(str(track_files[0]))
    nodes_df   = load_nodes(str(node_files[0]))
    arcs_df    = load_arcs(str(arc_files[0]))
    route_arcs = load_route(str(route_files[0])) if route_files else None
    return track_df, nodes_df, arcs_df, route_arcs


def deduplicate_consecutive(arcs):
    """Collapse consecutive duplicate arc IDs: [A, A, B, B, B, C] -> [A, B, C]."""
    out = []
    for a in arcs:
        if a is None or a == -1:
            continue
        if not out or out[-1] != a:
            out.append(a)
    return out


# ─────────────────────────────────────────────
#  Endpoints
# ─────────────────────────────────────────────

@app.get("/")
def root():
    active_dir = get_active_trips_dir()
    return {
        "status": "ok",
        "models_loaded": list(MODELS.keys()),
        "active_split": active_dir.name,
    }


@app.get("/trips")
def list_trips():
    """List available test trip IDs (eval_test if populated, else student_test)."""
    active_dir = get_active_trips_dir()
    trips = sorted([d.name for d in active_dir.iterdir() if d.is_dir()]) if active_dir.exists() else []
    return {"trips": trips, "count": len(trips), "split": active_dir.name}


@app.get("/models")
def list_models():
    """List all loaded models."""
    return {"models": list(MODELS.keys())}


@app.post("/predict/{trip_id}")
def predict(trip_id: str, model_name: str = None, use_hmm: bool = True):
    """
    Run map-matching for a given test trip.
    Ground truth (.route) is NEVER returned to the client.
    """
    trips_dir = get_active_trips_dir()
    trip_dir = trips_dir / trip_id
    if not trip_dir.exists():
        raise HTTPException(status_code=404, detail=f"Trip '{trip_id}' not found in {trips_dir.name}/.")

    model, used_model = get_model(model_name)
    print(f"\n[REQUEST] /predict/{trip_id}  model={used_model}  use_hmm={use_hmm}")
    t_start = time.time()

    try:
        track_df, nodes_df, arcs_df, _ = load_trip_files(trip_dir)
    except FileNotFoundError as e:
        raise HTTPException(status_code=500, detail=str(e))

    arc_geom = build_arc_geometry(nodes_df, arcs_df)
    feat_df = compute_features_for_trip(track_df, arc_geom, route_arc_ids=None)

    # Single batched predict_proba call for the whole trip (100x faster than per-fix loop)
    X_all = feat_df[FEATURE_COLS]
    try:
        proba_all = model.predict_proba(X_all)[:, 1]
    except Exception:
        proba_all = model.predict(X_all).astype(float)

    feat_df = feat_df[['fix_index', 'arc_id']].copy()
    feat_df['prob'] = proba_all

    # Group emissions by fix_index
    n_fixes = len(track_df)
    emission_per_fix = [{} for _ in range(n_fixes)]
    candidate_arcs_per_fix = [[] for _ in range(n_fixes)]

    for fix_idx, aid, prob in zip(
        feat_df['fix_index'].to_numpy(),
        feat_df['arc_id'].to_numpy(),
        feat_df['prob'].to_numpy(),
    ):
        fi = int(fix_idx)
        ai = int(aid)
        emission_per_fix[fi][ai] = float(prob)
        candidate_arcs_per_fix[fi].append(ai)

    # --- Decode ---
    if use_hmm:
        arc_node_map = arc_to_nodes(arcs_df)
        graph = build_road_graph(arcs_df)
        gap_set = detect_gaps(track_df)
        predicted_arcs = viterbi_decode(
            emission_per_fix, candidate_arcs_per_fix,
            arc_node_map, graph, gap_indices=gap_set
        )
    else:
        predicted_arcs = [
            max(emits, key=emits.get) if emits else None
            for emits in emission_per_fix
        ]

    t_elapsed = time.time() - t_start

    # --- Build GeoJSON for predicted route (deduplicated consecutive arcs) ---
    dedup_arcs = deduplicate_consecutive(predicted_arcs)
    lon_a, lat_a = arc_geom['lon_a'], arc_geom['lat_a']
    lon_b, lat_b = arc_geom['lon_b'], arc_geom['lat_b']
    n_arcs = len(lon_a)

    predicted_coords = []
    for aid in dedup_arcs:
        if 0 <= aid < n_arcs:
            predicted_coords.append([float(lon_a[aid]), float(lat_a[aid])])
            predicted_coords.append([float(lon_b[aid]), float(lat_b[aid])])

    gps_coords = [
        [float(lon), float(lat)]
        for lon, lat in zip(track_df['longitude'].to_numpy(), track_df['latitude'].to_numpy())
    ]

    print(f"[DONE] Trip={trip_id}  Model={used_model}  HMM={use_hmm}  "
          f"Time={t_elapsed:.2f}s  Fixes={len(predicted_arcs)}  UniqueRouteArcs={len(dedup_arcs)}")

    response = {
        "trip_id": trip_id,
        "model_used": used_model,
        "use_hmm": use_hmm,
        "inference_time_sec": round(t_elapsed, 3),
        "n_fixes": len(predicted_arcs),
        "n_predicted_route_arcs": len(dedup_arcs),
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

    PRED_CACHE[(trip_id, used_model, use_hmm)] = response
    return response


@app.get("/metrics/{trip_id}")
def compute_metrics(trip_id: str, model_name: str = None, use_hmm: bool = True):
    """
    Compute evaluation metrics for a trip.
    Ground truth (.route) is loaded SERVER-SIDE ONLY and never returned.
    """
    trips_dir = get_active_trips_dir()
    trip_dir = trips_dir / trip_id
    if not trip_dir.exists():
        raise HTTPException(status_code=404, detail=f"Trip '{trip_id}' not found.")

    _, used_model = get_model(model_name)
    print(f"\n[REQUEST] /metrics/{trip_id}  model={used_model}  use_hmm={use_hmm}")

    cache_key = (trip_id, used_model, use_hmm)
    if cache_key in PRED_CACHE:
        pred_response = PRED_CACHE[cache_key]
    else:
        pred_response = predict(trip_id, model_name=used_model, use_hmm=use_hmm)

    predicted_arcs = pred_response["predicted_arcs"]

    # Load ground truth SERVER-SIDE ONLY
    _, _, arcs_df, ground_truth = load_trip_files(trip_dir)
    arc_node_map = arc_to_nodes(arcs_df)

    if not ground_truth:
        return {
            "trip_id": trip_id,
            "model_used": used_model,
            "use_hmm": use_hmm,
            "n_fixes": len(predicted_arcs),
            "route_continuity": round(
                route_continuity([a if a != -1 else None for a in predicted_arcs], arc_node_map), 4
            ),
            "inference_time_sec": pred_response["inference_time_sec"],
            "note": "No ground truth .route file present for this trip.",
        }

    gt_set = set(ground_truth)
    valid_pred = [a for a in predicted_arcs if a != -1]
    dedup_pred = deduplicate_consecutive(predicted_arcs)
    pred_set = set(dedup_pred)

    # 1. Fix-level route hit accuracy (% of GPS fixes snapped onto a true route arc)
    fix_hits = sum(1 for a in valid_pred if a in gt_set)
    fix_acc = fix_hits / len(valid_pred) if valid_pred else 0.0

    # 2. Route-level Precision, Recall, F1 (comparing matched arc set vs ground-truth .route)
    tp = len(pred_set & gt_set)
    precision = tp / len(pred_set) if pred_set else 0.0
    recall    = tp / len(gt_set)   if gt_set   else 0.0
    route_f1  = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    # 3. Route continuity (same arc or directly connected next arc)
    rc = route_continuity([a if a != -1 else None for a in predicted_arcs], arc_node_map)

    metrics = {
        "trip_id": trip_id,
        "model_used": used_model,
        "use_hmm": use_hmm,
        "n_fixes": len(predicted_arcs),
        "top1_accuracy": round(fix_acc, 4),
        "macro_f1": round(route_f1, 4),
        "route_precision": round(precision, 4),
        "route_recall": round(recall, 4),
        "route_continuity": round(rc, 4),
        "inference_time_sec": pred_response["inference_time_sec"],
    }

    print(f"[METRICS] {metrics}")
    return metrics
