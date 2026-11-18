"""
PathSnap — Offline Evaluation & HMM Ablation Script
src/evaluate_student_test.py

Run once all .joblib models are placed in models/:
    python src/evaluate_student_test.py

Evaluates every saved model across all 10 student_test trips,
plus runs the Classifier-Only vs Classifier + HMM/Viterbi ablation.
"""

import sys
import time
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import log_loss

sys.path.insert(0, str(Path(__file__).parent))

from features import (
    load_track, load_nodes, load_arcs, load_route,
    build_arc_geometry, compute_features_for_trip, FEATURE_COLS
)
from hmm import (
    build_road_graph, arc_to_nodes,
    viterbi_decode, detect_gaps, route_continuity
)

BASE_DIR = Path(__file__).parent.parent
STUDENT_TEST_DIR = BASE_DIR / "data" / "student_test"
MODELS_DIR = BASE_DIR / "models"


def deduplicate_consecutive(arcs):
    out = []
    for a in arcs:
        if a is None or a == -1:
            continue
        if not out or out[-1] != a:
            out.append(a)
    return out


def eval_single_model_on_trips(model, trips_cache, use_hmm=True):
    fix_accs, route_f1s, route_precs, route_recs, conts, lls, times = [], [], [], [], [], [], []

    for item in trips_cache:
        t0 = time.time()
        feat_df = item["feat_df"]
        X = feat_df[FEATURE_COLS]
        y_true = feat_df["label"].to_numpy()

        try:
            proba = model.predict_proba(X)[:, 1]
        except Exception:
            proba = model.predict(X).astype(float)

        ll = log_loss(y_true, np.clip(proba, 1e-7, 1 - 1e-7), labels=[0, 1])

        n_fixes = item["n_fixes"]
        emission_per_fix = [{} for _ in range(n_fixes)]
        cand_per_fix = [[] for _ in range(n_fixes)]
        for fi, ai, pr in zip(feat_df["fix_index"].to_numpy(), feat_df["arc_id"].to_numpy(), proba):
            emission_per_fix[int(fi)][int(ai)] = float(pr)
            cand_per_fix[int(fi)].append(int(ai))

        if use_hmm:
            pred_arcs = viterbi_decode(
                emission_per_fix, cand_per_fix,
                item["arc_node_map"], item["graph"],
                gap_indices=item["gap_set"]
            )
        else:
            pred_arcs = [
                max(em, key=em.get) if em else None
                for em in emission_per_fix
            ]

        elapsed = time.time() - t0

        gt_set = item["gt_set"]
        valid_pred = [a for a in pred_arcs if a is not None]
        dedup_pred = deduplicate_consecutive(pred_arcs)
        pred_set = set(dedup_pred)

        fix_acc = sum(1 for a in valid_pred if a in gt_set) / len(valid_pred) if valid_pred else 0.0
        tp = len(pred_set & gt_set)
        prec = tp / len(pred_set) if pred_set else 0.0
        rec  = tp / len(gt_set)   if gt_set   else 0.0
        rf1  = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
        rc   = route_continuity(pred_arcs, item["arc_node_map"])

        fix_accs.append(fix_acc)
        route_f1s.append(rf1)
        route_precs.append(prec)
        route_recs.append(rec)
        conts.append(rc)
        lls.append(ll)
        times.append(elapsed)

    return {
        "route_f1": float(np.mean(route_f1s)),
        "fix_acc": float(np.mean(fix_accs)),
        "precision": float(np.mean(route_precs)),
        "recall": float(np.mean(route_recs)),
        "continuity": float(np.mean(conts)),
        "log_loss": float(np.mean(lls)),
        "avg_time_s": float(np.mean(times)),
    }


def main():
    model_files = sorted(MODELS_DIR.glob("*.joblib"))
    if not model_files:
        print(f"❌ No .joblib files found in {MODELS_DIR}. Download them from Kaggle first.")
        return

    trip_dirs = sorted([d for d in STUDENT_TEST_DIR.iterdir() if d.is_dir()])
    print(f"🔄 Pre-loading {len(trip_dirs)} student_test trips...")
    trips_cache = []
    for td in trip_dirs:
        tr = load_track(next(td.glob("*.track")))
        nd = load_nodes(next(td.glob("*.nodes")))
        ar = load_arcs(next(td.glob("*.arcs")))
        rt = load_route(next(td.glob("*.route")))
        geom = build_arc_geometry(nd, ar)
        feat_df = compute_features_for_trip(tr, geom, route_arc_ids=rt)
        trips_cache.append({
            "trip_id": td.name,
            "n_fixes": len(tr),
            "feat_df": feat_df,
            "gt_set": set(rt),
            "arc_node_map": arc_to_nodes(ar),
            "graph": build_road_graph(ar),
            "gap_set": detect_gaps(tr),
        })
    print(f"✅ Loaded {len(trips_cache)} trips.\n")

    print("════════════ STUDENT-TEST EVALUATION (WITH HMM) ════════════")
    rows = []
    loaded_models = {}
    for mf in model_files:
        m = joblib.load(mf)
        loaded_models[mf.stem] = m
        res = eval_single_model_on_trips(m, trips_cache, use_hmm=True)
        rows.append({"Model": mf.stem, **res})
        print(
            f"  {mf.stem:22s} | Route F1: {res['route_f1']:.4f} | "
            f"Fix Acc: {res['fix_acc']*100:5.1f}% | Cont: {res['continuity']*100:5.1f}% | "
            f"LogLoss: {res['log_loss']:.4f} | Time: {res['avg_time_s']:.2f}s"
        )

    best_name = "lgbm_tuned" if "lgbm_tuned" in loaded_models else max(rows, key=lambda r: r["route_f1"])["Model"]
    best_model = loaded_models[best_name]

    print(f"\n════════════ HMM ABLATION ({best_name}) ════════════")
    no_hmm = eval_single_model_on_trips(best_model, trips_cache, use_hmm=False)
    with_hmm = eval_single_model_on_trips(best_model, trips_cache, use_hmm=True)
    print(f"  Classifier ONLY : Route F1 = {no_hmm['route_f1']:.4f} | Fix Acc = {no_hmm['fix_acc']*100:.1f}% | Continuity = {no_hmm['continuity']*100:.1f}%")
    print(f"  Classifier + HMM: Route F1 = {with_hmm['route_f1']:.4f} | Fix Acc = {with_hmm['fix_acc']*100:.1f}% | Continuity = {with_hmm['continuity']*100:.1f}%")
    print(f"  Delta           : Route F1 = {with_hmm['route_f1']-no_hmm['route_f1']:+.4f} | Continuity = {(with_hmm['continuity']-no_hmm['continuity'])*100:+.1f}%")


if __name__ == "__main__":
    main()
