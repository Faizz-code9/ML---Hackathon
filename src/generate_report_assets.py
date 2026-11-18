"""
PathSnap — Reproducible Report Asset Generator
src/generate_report_assets.py

Generates ALL verifiable experimental metrics, EDA statistics, matplotlib plots,
and the 4-layer interactive Folium HTML map used in the Technical Report.

Run from repo root:
    python src/generate_report_assets.py
"""

import sys
import time
import json
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import folium
from pathlib import Path
from sklearn.metrics import (
    f1_score, accuracy_score, log_loss, confusion_matrix
)

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).parent.parent
SRC_DIR = BASE_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from features import (
    load_track, load_nodes, load_arcs, load_route,
    build_arc_geometry, compute_features_for_trip, FEATURE_COLS
)
from hmm import (
    build_road_graph, arc_to_nodes,
    viterbi_decode, detect_gaps, route_continuity
)

TRAIN_DIR = BASE_DIR / "data" / "training"
STUDENT_DIR = BASE_DIR / "data" / "student_test"
MODELS_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "report"
MAPS_DIR = REPORT_DIR / "maps"
MAPS_DIR.mkdir(parents=True, exist_ok=True)

# Verified 5-fold GroupKFold CV results from Kaggle training notebook
CV_RESULTS = {
    "logistic_regression": {"mean": 0.7303, "std": 0.0288, "family": "Linear", "train_s": 12.5},
    "knn":                 {"mean": 0.7097, "std": 0.0200, "family": "Instance-Based", "train_s": 23.0},
    "random_forest":       {"mean": 0.7352, "std": 0.0134, "family": "Bagging Ensemble", "train_s": 175.6},
    "lgbm":                {"mean": 0.7315, "std": 0.0283, "family": "Gradient Boosting", "train_s": 64.5},
    "mlp":                 {"mean": 0.7331, "std": 0.0204, "family": "Neural Network", "train_s": 278.2},
    "lgbm_tuned":          {"mean": 0.7383, "std": 0.0275, "family": "Gradient Boosting (Tuned)", "train_s": 123.8},
}


def dedup_consecutive(arcs):
    out = []
    for a in arcs:
        if a is None or a == -1:
            continue
        if not out or out[-1] != a:
            out.append(a)
    return out


# ─────────────────────────────────────────────────────────────
#  1. EDA & Missing-Value Analysis (training + student_test)
# ─────────────────────────────────────────────────────────────
def run_eda():
    print("🔄 [1/5] Running EDA & missing-value audit on training (80) & student_test (10)...")
    summary = {}
    all_intervals = []
    train_arcs_counts = []
    train_fixes_counts = []
    train_nodes_counts = []

    for split_name, split_dir in [("training", TRAIN_DIR), ("student_test", STUDENT_DIR)]:
        trip_dirs = sorted([d for d in split_dir.iterdir() if d.is_dir()])
        nan_track = nan_nodes = nan_arcs = 0
        dup_ts = 0
        gaps_gt5 = 0
        total_fixes = 0
        total_intervals = 0
        one_sec_intervals = 0
        n_nodes_list, n_arcs_list, n_fixes_list, n_route_list = [], [], [], []

        for td in trip_dirs:
            tr = load_track(next(td.glob("*.track")))
            nd = load_nodes(next(td.glob("*.nodes")))
            ar = load_arcs(next(td.glob("*.arcs")))
            rt = load_route(next(td.glob("*.route")))

            nan_track += int(tr.isna().sum().sum())
            nan_nodes += int(nd.isna().sum().sum())
            nan_arcs  += int(ar.isna().sum().sum())

            diffs = tr["timestamp_sec"].diff().dropna()
            dup_ts += int((diffs == 0).sum())
            gaps_gt5 += int((diffs > 5.0).sum())
            total_intervals += len(diffs)
            one_sec_intervals += int(((diffs >= 0.9) & (diffs <= 1.1)).sum())
            total_fixes += len(tr)

            n_fixes_list.append(len(tr))
            n_nodes_list.append(len(nd))
            n_arcs_list.append(len(ar))
            n_route_list.append(len(rt))

            if split_name == "training":
                all_intervals.extend(diffs.to_numpy().tolist())
                train_arcs_counts.append(len(ar))
                train_fixes_counts.append(len(tr))
                train_nodes_counts.append(len(nd))

        summary[split_name] = {
            "n_trips": len(trip_dirs),
            "total_fixes": total_fixes,
            "mean_fixes": round(float(np.mean(n_fixes_list)), 1),
            "min_fixes": int(np.min(n_fixes_list)),
            "max_fixes": int(np.max(n_fixes_list)),
            "min_nodes": int(np.min(n_nodes_list)),
            "max_nodes": int(np.max(n_nodes_list)),
            "min_arcs": int(np.min(n_arcs_list)),
            "max_arcs": int(np.max(n_arcs_list)),
            "mean_route_arcs": round(float(np.mean(n_route_list)), 1),
            "nan_track": nan_track,
            "nan_nodes": nan_nodes,
            "nan_arcs": nan_arcs,
            "duplicate_timestamps": dup_ts,
            "gaps_gt_5s": gaps_gt5,
            "pct_1hz": round(100.0 * one_sec_intervals / max(1, total_intervals), 2),
        }

    # --- Plot EDA distributions (using training partition) ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    clipped_int = [x for x in all_intervals if 0 <= x <= 10]
    axes[0].hist(clipped_int, bins=30, color="#2b6cb0", edgecolor="white")
    axes[0].axvline(1.0, color="#e53e3e", linestyle="--", linewidth=1.5, label="1.0s (1 Hz)")
    axes[0].set_title("GPS Sampling Intervals (Training, 80 Trips)")
    axes[0].set_xlabel("Interval Δt (seconds, clipped ≤ 10s)")
    axes[0].set_ylabel("Count")
    axes[0].legend()

    axes[1].hist(np.array(train_arcs_counts) / 1000.0, bins=20, color="#dd6b20", edgecolor="white")
    axes[1].set_title("Road Network Size per Trip (Training)")
    axes[1].set_xlabel("Number of Road Arcs (thousands)")
    axes[1].set_ylabel("Number of Trips")

    axes[2].hist(train_fixes_counts, bins=20, color="#2f855a", edgecolor="white")
    axes[2].set_title("GPS Fixes per Trip (Training)")
    axes[2].set_xlabel("Number of GPS Fixes")
    axes[2].set_ylabel("Number of Trips")

    plt.tight_layout()
    plt.savefig(REPORT_DIR / "eda_distributions.png", dpi=160)
    plt.close()
    print("  ✓ Saved report/eda_distributions.png")
    return summary


# ─────────────────────────────────────────────────────────────
#  2. Evaluate All 6 Models on student_test (Obs + Route level)
# ─────────────────────────────────────────────────────────────
def evaluate_student_test():
    print("🔄 [2/5] Evaluating all 6 models on student_test (10 unseen trips)...")
    trip_dirs = sorted([d for d in STUDENT_DIR.iterdir() if d.is_dir()])
    trips_cache = []
    for td in trip_dirs:
        tr = load_track(next(td.glob("*.track")))
        nd = load_nodes(next(td.glob("*.nodes")))
        ar = load_arcs(next(td.glob("*.arcs")))
        rt = load_route(next(td.glob("*.route")))
        geom = build_arc_geometry(nd, ar)
        feat = compute_features_for_trip(tr, geom, route_arc_ids=rt)
        trips_cache.append({
            "trip_id": td.name, "track": tr, "nodes": nd, "arcs": ar,
            "route": rt, "geom": geom, "feat_df": feat, "n_fixes": len(tr),
            "gt_set": set(rt), "arc_node_map": arc_to_nodes(ar),
            "graph": build_road_graph(ar), "gap_set": detect_gaps(tr),
        })

    model_order = ["logistic_regression", "knn", "random_forest", "lgbm", "mlp", "lgbm_tuned"]
    loaded = {name: joblib.load(MODELS_DIR / f"{name}.joblib") for name in model_order}

    def eval_model(model, use_hmm=True):
        all_y_true, all_y_pred, all_y_prob = [], [], []
        fix_accs, route_f1s, precs, recs, conts, times = [], [], [], [], [], []
        per_trip_preds = {}

        for item in trips_cache:
            t0 = time.time()
            df = item["feat_df"]
            X = df[FEATURE_COLS]
            y_true = df["label"].to_numpy()
            proba = model.predict_proba(X)[:, 1]
            y_pred = model.predict(X)

            all_y_true.append(y_true)
            all_y_pred.append(y_pred)
            all_y_prob.append(proba)

            n = item["n_fixes"]
            em = [{} for _ in range(n)]
            ca = [[] for _ in range(n)]
            for fi, ai, pr in zip(df["fix_index"].to_numpy(), df["arc_id"].to_numpy(), proba):
                em[int(fi)][int(ai)] = float(pr)
                ca[int(fi)].append(int(ai))

            if use_hmm:
                pred_arcs = viterbi_decode(em, ca, item["arc_node_map"], item["graph"], gap_indices=item["gap_set"])
            else:
                pred_arcs = [max(e, key=e.get) if e else None for e in em]

            dt = time.time() - t0
            per_trip_preds[item["trip_id"]] = pred_arcs

            gt = item["gt_set"]
            vp = [a for a in pred_arcs if a is not None]
            ps = set(dedup_consecutive(pred_arcs))
            fix_acc = sum(1 for a in vp if a in gt) / len(vp) if vp else 0.0
            tp = len(ps & gt)
            prec = tp / len(ps) if ps else 0.0
            rec  = tp / len(gt) if gt else 0.0
            rf1  = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0
            rc   = route_continuity(pred_arcs, item["arc_node_map"])

            fix_accs.append(fix_acc)
            route_f1s.append(rf1)
            precs.append(prec)
            recs.append(rec)
            conts.append(rc)
            times.append(dt)

        y_t = np.concatenate(all_y_true)
        y_p = np.concatenate(all_y_pred)
        y_pr = np.clip(np.concatenate(all_y_prob), 1e-7, 1 - 1e-7)

        return {
            "obs_macro_f1": round(float(f1_score(y_t, y_p, average="macro", zero_division=0)), 4),
            "obs_accuracy": round(float(accuracy_score(y_t, y_p)), 4),
            "log_loss": round(float(log_loss(y_t, y_pr, labels=[0, 1])), 4),
            "fix_top1_acc": round(float(np.mean(fix_accs)), 4),
            "route_f1": round(float(np.mean(route_f1s)), 4),
            "route_prec": round(float(np.mean(precs)), 4),
            "route_rec": round(float(np.mean(recs)), 4),
            "route_continuity": round(float(np.mean(conts)), 4),
            "avg_inf_time_s": round(float(np.mean(times)), 3),
            "cm": confusion_matrix(y_t, y_p).tolist(),
            "per_trip_preds": per_trip_preds,
        }

    eval_with_hmm = {}
    eval_no_hmm = {}
    for name in model_order:
        eval_with_hmm[name] = eval_model(loaded[name], use_hmm=True)
        eval_no_hmm[name] = eval_model(loaded[name], use_hmm=False)
        r = eval_with_hmm[name]
        print(
            f"  ✓ {name:20s} | ObsF1={r['obs_macro_f1']:.4f} | Top1={r['fix_top1_acc']*100:5.1f}% | "
            f"RouteF1={r['route_f1']:.4f} | Cont={r['route_continuity']*100:5.1f}% | LL={r['log_loss']:.4f} | t={r['avg_inf_time_s']:.2f}s"
        )

    return trips_cache, loaded, eval_with_hmm, eval_no_hmm


# ─────────────────────────────────────────────────────────────
#  3. Compulsory Model Comparison & Diagnostic Plots
# ─────────────────────────────────────────────────────────────
def generate_model_plots(loaded, eval_with_hmm, eval_no_hmm):
    print("🔄 [3/5] Generating model comparison, Optuna, HMM ablation & MLP curve plots...")
    names = ["logistic_regression", "knn", "random_forest", "lgbm", "mlp", "lgbm_tuned"]
    labels = ["Logistic\nReg.", "k-NN\n(k=7)", "Random\nForest", "LightGBM\n(Default)", "MLP\n(128→64)", "LightGBM\n(Tuned)"]
    colors = ["#4299e1", "#ed8936", "#48bb78", "#9f7aea", "#ed64a6", "#2b6cb0"]

    # --- Plot A: 5-Fold GroupKFold CV Macro F1 Comparison ---
    means = [CV_RESULTS[n]["mean"] for n in names]
    stds  = [CV_RESULTS[n]["std"] for n in names]
    plt.figure(figsize=(9.5, 4.5))
    bars = plt.bar(labels, means, yerr=stds, capsize=5, color=colors, edgecolor="#2d3748", alpha=0.9)
    plt.ylim(0.65, 0.79)
    plt.ylabel("5-Fold GroupKFold CV Macro F1")
    plt.title("Cross-Validation Macro F1-Score Across All Models (80 Training Trips)")
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    for bar, m in zip(bars, means):
        plt.text(bar.get_x() + bar.get_width()/2, m + 0.004, f"{m:.4f}", ha="center", va="bottom", fontweight="bold", fontsize=9)
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "cv_f1_comparison.png", dpi=160)
    plt.close()

    # --- Plot B: Student-Test Top-1 Accuracy, Route F1, Obs Macro F1 & Continuity ---
    x = np.arange(len(names))
    w = 0.25
    top1 = [eval_with_hmm[n]["fix_top1_acc"] for n in names]
    rf1  = [eval_with_hmm[n]["route_f1"] for n in names]
    of1  = [eval_with_hmm[n]["obs_macro_f1"] for n in names]

    plt.figure(figsize=(10.5, 4.8))
    b1 = plt.bar(x - w, of1,  w, label="Obs-Level Macro F1", color="#a0aec0", edgecolor="#2d3748")
    b2 = plt.bar(x,     top1, w, label="Fix-Level Top-1 Acc (w/ HMM)", color="#3182ce", edgecolor="#2d3748")
    b3 = plt.bar(x + w, rf1,  w, label="Route F1-Score (w/ HMM)", color="#38a169", edgecolor="#2d3748")
    plt.xticks(x, labels)
    plt.ylim(0.65, 1.02)
    plt.ylabel("Score")
    plt.title("Student-Test Evaluation: Observation Macro F1, Top-1 Accuracy & Route F1 (10 Trips)")
    plt.legend(loc="lower right")
    plt.grid(axis="y", linestyle="--", alpha=0.4)
    for bars_grp in (b1, b2, b3):
        for b in bars_grp:
            h = b.get_height()
            plt.text(b.get_x() + b.get_width()/2, h + 0.006, f"{h:.3f}", ha="center", va="bottom", fontsize=7.5)
    plt.tight_layout()
    plt.savefig(REPORT_DIR / "student_test_comparison.png", dpi=160)
    plt.close()

    # --- Plot C: Log Loss & Inference Time Comparison ---
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.4))
    lls = [eval_with_hmm[n]["log_loss"] for n in names]
    ts  = [eval_with_hmm[n]["avg_inf_time_s"] for n in names]

    bars_ll = axes[0].bar(labels, lls, color=colors, edgecolor="#2d3748", alpha=0.9)
    axes[0].set_title("Student-Test Log Loss Comparison (Lower = Better Calibration)")
    axes[0].set_ylabel("Binary Log Loss")
    axes[0].grid(axis="y", linestyle="--", alpha=0.4)
    for b, v in zip(bars_ll, lls):
        axes[0].text(b.get_x() + b.get_width()/2, v + 0.015, f"{v:.4f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    bars_t = axes[1].bar(labels, ts, color=colors, edgecolor="#2d3748", alpha=0.9)
    axes[1].set_title("Mean Inference Time per Trip on Student-Test (Lower = Faster)")
    axes[1].set_ylabel("Seconds per Trip")
    axes[1].grid(axis="y", linestyle="--", alpha=0.4)
    for b, v in zip(bars_t, ts):
        axes[1].text(b.get_x() + b.get_width()/2, v + 0.025, f"{v:.2f}s", ha="center", va="bottom", fontsize=8.5, fontweight="bold")

    plt.tight_layout()
    plt.savefig(REPORT_DIR / "logloss_and_time.png", dpi=160)
    plt.close()

    # --- Plot D: Real Epoch-Wise MLP Training History (ONLY for MLP — authentic curve!) ---
    mlp_est = loaded["mlp"].named_steps["model"]
    if hasattr(mlp_est, "loss_curve_"):
        fig, ax1 = plt.subplots(figsize=(8.5, 4.2))
        epochs = np.arange(1, len(mlp_est.loss_curve_) + 1)
        ax1.plot(epochs, mlp_est.loss_curve_, marker="o", color="#e53e3e", linewidth=2, label="Training Log Loss")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Training Binary Cross-Entropy Loss", color="#e53e3e")
        ax1.tick_params(axis="y", labelcolor="#e53e3e")
        ax1.grid(True, linestyle="--", alpha=0.35)

        if hasattr(mlp_est, "validation_scores_") and mlp_est.validation_scores_:
            ax2 = ax1.twinx()
            ax2.plot(epochs, mlp_est.validation_scores_, marker="s", color="#2b6cb0", linewidth=2, label="Internal Validation Accuracy")
            ax2.set_ylabel("Internal Validation Accuracy", color="#2b6cb0")
            ax2.tick_params(axis="y", labelcolor="#2b6cb0")
            lines1, labs1 = ax1.get_legend_handles_labels()
            lines2, labs2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labs1 + labs2, loc="center right")
        else:
            ax1.legend()

        plt.title("MLP Neural Network — Real Epoch-Wise Training & Early-Stopping History")
        plt.tight_layout()
        plt.savefig(REPORT_DIR / "mlp_training_curve.png", dpi=160)
        plt.close()
        print("  ✓ Saved report/mlp_training_curve.png (real epochs:", len(mlp_est.loss_curve_), ")")

    # --- Plot E: HMM / Viterbi Ablation Chart (All 6 Models) ---
    no_rf1 = [eval_no_hmm[n]["route_f1"] for n in names]
    yes_rf1 = [eval_with_hmm[n]["route_f1"] for n in names]
    no_cnt = [eval_no_hmm[n]["route_continuity"] for n in names]
    yes_cnt = [eval_with_hmm[n]["route_continuity"] for n in names]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    w2 = 0.35
    axes[0].bar(x - w2/2, no_rf1, w2, label="Classifier Only", color="#fc8181", edgecolor="#2d3748")
    axes[0].bar(x + w2/2, yes_rf1, w2, label="Classifier + HMM/Viterbi", color="#38a169", edgecolor="#2d3748")
    axes[0].set_xticks(x); axes[0].set_xticklabels(labels, fontsize=8.5)
    axes[0].set_ylim(0.60, 1.02)
    axes[0].set_ylabel("Route F1-Score")
    axes[0].set_title("HMM Ablation: Route F1 (Student-Test)")
    axes[0].legend(loc="lower right")
    axes[0].grid(axis="y", linestyle="--", alpha=0.4)

    axes[1].bar(x - w2/2, no_cnt, w2, label="Classifier Only", color="#f6ad55", edgecolor="#2d3748")
    axes[1].bar(x + w2/2, yes_cnt, w2, label="Classifier + HMM/Viterbi", color="#3182ce", edgecolor="#2d3748")
    axes[1].set_xticks(x); axes[1].set_xticklabels(labels, fontsize=8.5)
    axes[1].set_ylim(0.65, 1.03)
    axes[1].set_ylabel("Route Continuity")
    axes[1].set_title("HMM Ablation: Route Continuity (Student-Test)")
    axes[1].legend(loc="lower right")
    axes[1].grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.savefig(REPORT_DIR / "hmm_ablation.png", dpi=160)
    plt.close()
    print("  ✓ Saved cv_f1_comparison.png, student_test_comparison.png, logloss_and_time.png, hmm_ablation.png")


# ─────────────────────────────────────────────────────────────
#  4. Interactive Folium Map + High-Res Road/GPS/GT/Pred Figure
# ─────────────────────────────────────────────────────────────
def generate_geographical_visuals(trips_cache, eval_with_hmm):
    print("🔄 [4/5] Generating interactive 4-layer Folium HTML map & static geographical comparison...")
    # Pick a crisp, clean student_test trip (Trip index 0 = 00000000)
    item = trips_cache[0]
    tid = item["trip_id"]
    tr = item["track"]
    geom = item["geom"]
    gt_route = item["route"]
    pred_arcs = dedup_consecutive(eval_with_hmm["lgbm_tuned"]["per_trip_preds"][tid])

    gps_lats = tr["latitude"].to_numpy()
    gps_lons = tr["longitude"].to_numpy()
    lat_min, lat_max = float(gps_lats.min()), float(gps_lats.max())
    lon_min, lon_max = float(gps_lons.min()), float(gps_lons.max())
    pad_lat = max(0.002, (lat_max - lat_min) * 0.08)
    pad_lon = max(0.002, (lon_max - lon_min) * 0.08)

    # Filter local relevant road-network arcs inside the trip bounding box
    lon_a, lat_a = geom["lon_a"], geom["lat_a"]
    lon_b, lat_b = geom["lon_b"], geom["lat_b"]
    in_box = (
        (lon_a >= lon_min - pad_lon) & (lon_a <= lon_max + pad_lon) &
        (lat_a >= lat_min - pad_lat) & (lat_a <= lat_max + pad_lat)
    )
    local_indices = np.where(in_box)[0]
    # Cap Folium grey road lines to 2500 nearby segments so HTML stays snappy
    if len(local_indices) > 2500:
        rng = np.random.default_rng(42)
        local_indices_folium = rng.choice(local_indices, size=2500, replace=False)
    else:
        local_indices_folium = local_indices

    # --- A. Interactive Folium HTML Map (committed to GitHub!) ---
    m = folium.Map(
        location=[float(np.mean(gps_lats)), float(np.mean(gps_lons))],
        zoom_start=14,
        tiles="OpenStreetMap",
    )

    # Layer 1: Relevant road-network geometry (grey)
    net_layer = folium.FeatureGroup(name="Road Network (Local Subgraph)", show=True)
    for idx in local_indices_folium:
        folium.PolyLine(
            [[float(lat_a[idx]), float(lon_a[idx])], [float(lat_b[idx]), float(lon_b[idx])]],
            color="#718096", weight=1, opacity=0.35
        ).add_to(net_layer)
    net_layer.add_to(m)

    # Layer 2: Original GPS observations (blue points + polyline)
    gps_layer = folium.FeatureGroup(name="Original GPS Observations", show=True)
    gps_latlon = [[float(la), float(lo)] for la, lo in zip(gps_lats, gps_lons)]
    folium.PolyLine(gps_latlon, color="#0077ff", weight=6, opacity=0.45, tooltip="Raw GPS Track").add_to(gps_layer)
    for la, lo in gps_latlon[::15]:
        folium.CircleMarker([la, lo], radius=2.5, color="#0055cc", fill=True, fill_opacity=0.8).add_to(gps_layer)
    folium.Marker(gps_latlon[0], popup=f"Trip {tid} Start", icon=folium.Icon(color="green", icon="play")).add_to(gps_layer)
    folium.Marker(gps_latlon[-1], popup=f"Trip {tid} End", icon=folium.Icon(color="darkred", icon="stop")).add_to(gps_layer)
    gps_layer.add_to(m)

    # Layer 3: Reference Ground-Truth Route (green)
    gt_layer = folium.FeatureGroup(name="Reference Route (Ground Truth)", show=True)
    gt_coords = []
    for aid in gt_route:
        if 0 <= aid < len(lon_a):
            gt_coords.append([float(lat_a[aid]), float(lon_a[aid])])
            gt_coords.append([float(lat_b[aid]), float(lon_b[aid])])
    if gt_coords:
        folium.PolyLine(gt_coords, color="#2f855a", weight=5, opacity=0.75, tooltip="Reference Route (.route)").add_to(gt_layer)
    gt_layer.add_to(m)

    # Layer 4: Predicted Route (red dashed/solid on top)
    pred_layer = folium.FeatureGroup(name="Predicted Route (LightGBM Tuned + HMM)", show=True)
    pred_coords = []
    for aid in pred_arcs:
        if 0 <= aid < len(lon_a):
            pred_coords.append([float(lat_a[aid]), float(lon_a[aid])])
            pred_coords.append([float(lat_b[aid]), float(lon_b[aid])])
    if pred_coords:
        folium.PolyLine(pred_coords, color="#e53e3e", weight=3, opacity=0.95, dash_array="6 3", tooltip="Predicted Route (HMM)").add_to(pred_layer)
    pred_layer.add_to(m)

    # Floating HTML Legend
    legend_html = f"""
    <div style="position: fixed; bottom: 25px; left: 25px; z-index: 9999;
                background: white; padding: 10px 14px; border: 2px solid #2d3748;
                border-radius: 6px; font-family: Arial; font-size: 12px; box-shadow: 0 2px 6px rgba(0,0,0,0.2);">
      <b>PathSnap Map — Student-Test Trip {tid}</b><br>
      <span style="color:#718096;">━━</span> Local Road Network<br>
      <span style="color:#0077ff;">━━</span> Original GPS Observations<br>
      <span style="color:#2f855a;">━━</span> Reference Route (.route)<br>
      <span style="color:#e53e3e;">┅┅</span> Predicted Route (LightGBM Tuned + HMM)
    </div>
    """
    m.get_root().html.add_child(folium.Element(legend_html))
    m.fit_bounds([[lat_min, lon_min], [lat_max, lon_max]])
    folium.LayerControl(collapsed=False).add_to(m)

    map_path = MAPS_DIR / f"student_trip_{tid}_map.html"
    m.save(str(map_path))
    print(f"  ✓ Saved interactive Folium map: {map_path}")

    # --- B. Static High-Res 2-Panel Plot (Full Trip + Zoomed Intersection) ---
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5))

    def draw_panel(ax, xlim, ylim, title):
        # Road network segments inside view
        mask = (
            (lon_a >= xlim[0]) & (lon_a <= xlim[1]) &
            (lat_a >= ylim[0]) & (lat_a <= ylim[1])
        )
        idxs = np.where(mask)[0]
        if len(idxs) > 8000:
            idxs = idxs[:: max(1, len(idxs) // 8000)]
        for i in idxs:
            ax.plot([lon_a[i], lon_b[i]], [lat_a[i], lat_b[i]], color="#cbd5e0", linewidth=0.5, zorder=1)

        # Reference route (green)
        for aid in gt_route:
            if 0 <= aid < len(lon_a):
                ax.plot([lon_a[aid], lon_b[aid]], [lat_a[aid], lat_b[aid]], color="#2f855a", linewidth=3.5, alpha=0.75, zorder=2)

        # GPS observations (blue scatter)
        ax.scatter(gps_lons, gps_lats, s=8, color="#3182ce", alpha=0.6, zorder=3, label="Original GPS Fixes")

        # Predicted route (red)
        for aid in pred_arcs:
            if 0 <= aid < len(lon_a):
                ax.plot([lon_a[aid], lon_b[aid]], [lat_a[aid], lat_b[aid]], color="#e53e3e", linewidth=1.8, linestyle="--", alpha=0.95, zorder=4)

        # Proxy legend entries
        ax.plot([], [], color="#cbd5e0", linewidth=1.5, label="Road Network Geometry")
        ax.plot([], [], color="#2f855a", linewidth=3.5, label="Reference Route (.route)")
        ax.plot([], [], color="#e53e3e", linewidth=2.0, linestyle="--", label="Predicted Route (LGBM Tuned + HMM)")

        ax.set_xlim(xlim); ax.set_ylim(ylim)
        ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
        ax.set_title(title)
        ax.legend(loc="best", fontsize=8, framealpha=0.92)
        ax.grid(True, linestyle=":", alpha=0.4)

    draw_panel(
        axes[0],
        (lon_min - pad_lon, lon_max + pad_lon),
        (lat_min - pad_lat, lat_max + pad_lat),
        f"Student-Test Trip {tid}: Road Network, GPS, Reference & Predicted Route",
    )

    # Zoomed-in region around midpoint of GPS trajectory
    mid_idx = len(gps_lons) // 2
    mlo, mla = float(gps_lons[mid_idx]), float(gps_lats[mid_idx])
    z_span = 0.006
    draw_panel(
        axes[1],
        (mlo - z_span, mlo + z_span),
        (mla - z_span * 0.7, mla + z_span * 0.7),
        f"Zoomed View (Trip {tid}): GPS Noise Snapped onto Road Arcs",
    )

    plt.tight_layout()
    plt.savefig(REPORT_DIR / "geographical_road_network.png", dpi=170)
    plt.close()
    print("  ✓ Saved report/geographical_road_network.png")


# ─────────────────────────────────────────────────────────────
#  5. Save Verified JSON Summary for Exact Report Numbers
# ─────────────────────────────────────────────────────────────
def main():
    eda = run_eda()
    trips_cache, loaded, eval_hmm, eval_no_hmm = evaluate_student_test()
    generate_model_plots(loaded, eval_hmm, eval_no_hmm)
    generate_geographical_visuals(trips_cache, eval_hmm)

    summary = {
        "eda": eda,
        "cv": CV_RESULTS,
        "student_test_with_hmm": {
            k: {mk: mv for mk, mv in v.items() if mk != "per_trip_preds"}
            for k, v in eval_hmm.items()
        },
        "student_test_no_hmm": {
            k: {mk: mv for mk, mv in v.items() if mk != "per_trip_preds"}
            for k, v in eval_no_hmm.items()
        },
    }
    out_json = REPORT_DIR / "metrics_summary.json"
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n✅ [5/5] All report assets & verified metrics saved to {out_json}!")


if __name__ == "__main__":
    main()
