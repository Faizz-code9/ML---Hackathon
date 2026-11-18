"""
PathSnap — Model Comparison Script
scripts/compare_models.py

Loads every saved .joblib model from models/, runs prediction on a
supplied data directory (student_test or a local subset of training),
and prints a ranked comparison table using the existing
src/metrics.py::evaluate_model function.

This script wires together src/features.py, src/metrics.py, and
src/model_card.py — none of which were connected to each other
outside of the Kaggle notebook.

Usage
-----
    # Compare on student test trips (no route files — skips per-fix accuracy)
    python scripts/compare_models.py --data_dir data/student_test

    # Compare on training trips with ground truth (full metrics)
    python scripts/compare_models.py --data_dir data/training --max_trips 10

    # Save results to CSV
    python scripts/compare_models.py --data_dir data/training --output results.csv

Requirements
------------
    pip install scikit-learn lightgbm joblib pandas numpy
"""

import sys
import os
import argparse
import time
from pathlib import Path
from glob import glob

import numpy as np
import pandas as pd
import joblib

# Allow imports from src/
_SRC = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(_SRC))

from features import (
    load_track, load_nodes, load_arcs, load_route,
    build_arc_geometry, compute_features_for_trip, FEATURE_COLS,
)
from metrics import evaluate_model
from model_card import MODEL_REGISTRY, get_model_card


# ─────────────────────────────────────────────
#  Helpers
# ─────────────────────────────────────────────

def load_models(models_dir: Path) -> dict:
    """Load all .joblib files from models_dir. Return {stem: pipeline}."""
    models = {}
    for path in sorted(models_dir.glob("*.joblib")):
        try:
            models[path.stem] = joblib.load(path)
            print(f"  ✓ Loaded {path.name}")
        except Exception as e:
            print(f"  ✗ Could not load {path.name}: {e}")
    return models


def collect_trips(data_dir: Path, max_trips: int | None = None) -> list[Path]:
    """Return sorted list of trip directories inside data_dir."""
    dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    if max_trips:
        dirs = dirs[:max_trips]
    return dirs


def evaluate_one_trip(trip_dir: Path, model, model_name: str) -> dict | None:
    """
    Run inference on one trip and compute metrics.

    Returns a dict of metrics, or None if the trip cannot be evaluated
    (e.g. missing files or no ground truth).
    """
    track_files = list(trip_dir.glob("*.track"))
    node_files  = list(trip_dir.glob("*.nodes"))
    arc_files   = list(trip_dir.glob("*.arcs"))
    route_files = list(trip_dir.glob("*.route"))

    if not (track_files and node_files and arc_files):
        return None  # Incomplete trip

    track_df = load_track(str(track_files[0]))
    nodes_df = load_nodes(str(node_files[0]))
    arcs_df  = load_arcs(str(arc_files[0]))
    route_arcs = load_route(str(route_files[0])) if route_files else None

    arc_geom = build_arc_geometry(nodes_df, arcs_df)

    # Build features (inference mode — no route labels)
    feat_df = compute_features_for_trip(track_df, arc_geom, route_arc_ids=None)

    if feat_df.empty:
        return None

    t0 = time.perf_counter()
    try:
        y_pred = model.predict(feat_df[FEATURE_COLS].values)
        try:
            y_proba = model.predict_proba(feat_df[FEATURE_COLS].values)[:, 1]
        except AttributeError:
            y_proba = None
    except Exception as e:
        print(f"    ✗ Prediction failed for {trip_dir.name}: {e}")
        return None
    elapsed = time.perf_counter() - t0

    result = {
        "trip": trip_dir.name,
        "model": model_name,
        "n_rows": len(feat_df),
        "n_fixes": feat_df["fix_index"].nunique(),
        "inference_time_sec": round(elapsed, 4),
    }

    if route_arcs is not None:
        # Build per-row ground truth labels for classification metrics
        y_true = feat_df["label"].values if "label" in feat_df.columns else None

        # Re-compute labels from route arcs if present
        arc_geom_lookup = {a["arc_id"]: a for a in arc_geom}
        route_set = set(route_arcs)

        # For classifier metrics: label each candidate row
        # (1 if the arc_id appears in the route, 0 otherwise)
        # This mirrors how training labels were created.
        label_col = feat_df.apply(
            lambda row: 1 if row["arc_id"] in route_set else 0, axis=1
        ).values

        metrics = evaluate_model(label_col, y_pred, y_proba)
        result.update(metrics)

    return result


# ─────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Compare PathSnap models on a local trip directory."
    )
    parser.add_argument(
        "--data_dir", required=True,
        help="Path to trip directory (e.g. data/student_test or data/training).",
    )
    parser.add_argument(
        "--models_dir", default="models",
        help="Path to models directory (default: models/).",
    )
    parser.add_argument(
        "--max_trips", type=int, default=None,
        help="Limit number of trips evaluated (useful for quick checks).",
    )
    parser.add_argument(
        "--output", default=None,
        help="Optional path to save results as CSV (e.g. results.csv).",
    )
    args = parser.parse_args()

    repo_root = Path(__file__).parent.parent
    data_dir   = repo_root / args.data_dir
    models_dir = repo_root / args.models_dir

    if not data_dir.exists():
        print(f"✗ Data directory not found: {data_dir}")
        sys.exit(1)
    if not models_dir.exists():
        print(f"✗ Models directory not found: {models_dir}")
        sys.exit(1)

    # ── Load models ──────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("  Loading models from:", models_dir)
    print(f"{'─'*60}")
    models = load_models(models_dir)
    if not models:
        print("No models loaded. Exiting.")
        sys.exit(1)

    # ── Collect trips ────────────────────────────────────────────────
    trip_dirs = collect_trips(data_dir, args.max_trips)
    print(f"\n  Evaluating {len(trip_dirs)} trip(s) from: {data_dir}\n")

    # ── Evaluate each model on each trip ─────────────────────────────
    all_results = []

    for model_name, model in models.items():
        print(f"\n  ── Model: {model_name} {'─'*(40-len(model_name))}")
        trip_results = []
        for trip_dir in trip_dirs:
            res = evaluate_one_trip(trip_dir, model, model_name)
            if res:
                trip_results.append(res)
                suffix = ""
                if "macro_f1" in res:
                    suffix = f"  F1={res['macro_f1']:.4f}  Acc={res['top1_accuracy']:.4f}"
                print(f"    {trip_dir.name}  {res['n_fixes']:3d} fixes{suffix}")
        all_results.extend(trip_results)

    if not all_results:
        print("\nNo results produced. Check that trip files are present.")
        sys.exit(0)

    results_df = pd.DataFrame(all_results)

    # ── Aggregate per model ───────────────────────────────────────────
    metric_cols = [c for c in ["macro_f1", "top1_accuracy", "log_loss"] if c in results_df.columns]

    agg = results_df.groupby("model").agg(
        n_trips=("trip", "count"),
        mean_inference_sec=("inference_time_sec", "mean"),
        **{f"mean_{m}": (m, "mean") for m in metric_cols},
        **{f"std_{m}":  (m, "std")  for m in metric_cols},
    ).reset_index()

    if metric_cols:
        agg = agg.sort_values(f"mean_{metric_cols[0]}", ascending=False)

    # ── Print comparison table ────────────────────────────────────────
    print(f"\n\n{'='*65}")
    print("  PathSnap — Model Comparison Summary")
    print(f"{'='*65}")
    print(f"  Data   : {data_dir}")
    print(f"  Trips  : {agg['n_trips'].iloc[0] if len(agg) > 0 else 0}")
    print(f"{'─'*65}")

    # Build display table
    display_rows = []
    for _, row in agg.iterrows():
        model_key = row["model"]
        display_name = (
            MODEL_REGISTRY.get(model_key, {}).get("display_name", model_key)
        )
        entry = {
            "Model": display_name,
            "Key": model_key,
            "Trips": int(row["n_trips"]),
        }
        for m in metric_cols:
            mean_val = row.get(f"mean_{m}", float("nan"))
            std_val  = row.get(f"std_{m}",  float("nan"))
            entry[m] = f"{mean_val:.4f} ± {std_val:.4f}"
        entry["Avg Inf. (s)"] = f"{row['mean_inference_sec']:.4f}"
        display_rows.append(entry)

    display_df = pd.DataFrame(display_rows)

    # Pretty-print
    col_widths = {c: max(len(c), display_df[c].astype(str).str.len().max())
                  for c in display_df.columns}
    header = "  " + "  ".join(c.ljust(col_widths[c]) for c in display_df.columns)
    print(header)
    print("  " + "  ".join("─" * col_widths[c] for c in display_df.columns))
    for _, row in display_df.iterrows():
        print("  " + "  ".join(str(row[c]).ljust(col_widths[c]) for c in display_df.columns))
    print(f"{'─'*65}")

    if metric_cols:
        best_model_key = agg.iloc[0]["model"]
        best_model_name = MODEL_REGISTRY.get(best_model_key, {}).get("display_name", best_model_key)
        print(f"\n  🏆  Best model: {best_model_name}  ({best_model_key})")
        print(f"       mean_{metric_cols[0]} = {agg.iloc[0][f'mean_{metric_cols[0]}']:.4f}")

    # ── Save CSV ──────────────────────────────────────────────────────
    if args.output:
        out_path = Path(args.output)
        agg.to_csv(out_path, index=False)
        print(f"\n  Saved aggregated results → {out_path}")

    print()


if __name__ == "__main__":
    main()
