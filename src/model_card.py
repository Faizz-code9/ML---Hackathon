"""
PathSnap — Model Card & Hyperparameter Registry
src/model_card.py

Records every model's configuration, training methodology, and
Optuna search space in one place.  This is the single source of
truth for "what exactly was trained and how", solving the problem
that the notebook never persisted its output cells.

Usage
-----
    from src.model_card import MODEL_REGISTRY, print_model_card, get_model_card

    # Print a human-readable summary to stdout
    print_model_card()

    # Get the card for a single model
    card = get_model_card("lgbm_tuned")

    # Iterate over all models
    for name, card in MODEL_REGISTRY.items():
        print(name, card["cv_methodology"])
"""

from __future__ import annotations

# ─────────────────────────────────────────────
#  Training methodology (shared across models)
# ─────────────────────────────────────────────

_CV_METHODOLOGY = {
    "strategy": "GroupKFold",
    "n_splits": 5,
    "group_column": "trip_id",
    "rationale": (
        "Groups ensure that all candidate rows from the same trip "
        "stay in the same fold, preventing data leakage across trips."
    ),
    "scoring_metric": "macro_f1",
    "task": "binary_classification",
    "positive_class": "correct candidate arc (label=1)",
    "negative_class": "wrong candidate arc (label=0)",
    "class_imbalance_note": (
        "Dataset is imbalanced: typically ~10–15 % positive rows "
        "(one correct arc per GPS fix among up to 10 candidates). "
        "No explicit resampling applied; macro F1 handles imbalance."
    ),
}

_FEATURE_COLS = [
    "dist_to_arc_m",
    "bearing_diff",
    "cos_bearing_diff",
    "sin_bearing_diff",
    "speed_mps",
    "time_delta_sec",
    "arc_length_m",
    "candidate_rank",
]

_CANDIDATE_GENERATION = {
    "radius_m": 50,
    "top_k": 10,
    "distance_metric": "perpendicular distance from GPS point to arc segment (metres)",
}

# ─────────────────────────────────────────────
#  Model Registry
# ─────────────────────────────────────────────

MODEL_REGISTRY: dict[str, dict] = {

    "logistic_regression": {
        "display_name": "Logistic Regression (Baseline)",
        "library": "sklearn.linear_model.LogisticRegression",
        "role": "baseline",
        "saved_as": "logistic_regression.joblib",
        "needs_scaling": True,
        "scaler": "sklearn.preprocessing.StandardScaler",
        "hyperparameters": {
            "C": 1.0,
            "max_iter": 1000,
            "solver": "lbfgs",
            "multi_class": "auto",
        },
        "hyperparameter_search": None,
        "cv_methodology": _CV_METHODOLOGY,
        "feature_cols": _FEATURE_COLS,
        "candidate_generation": _CANDIDATE_GENERATION,
        "notes": (
            "No hyperparameter search was performed for this model; "
            "default sklearn values are used as a sanity-check baseline."
        ),
    },

    "knn": {
        "display_name": "k-Nearest Neighbours",
        "library": "sklearn.neighbors.KNeighborsClassifier",
        "role": "baseline",
        "saved_as": "knn.joblib",
        "needs_scaling": True,
        "scaler": "sklearn.preprocessing.StandardScaler",
        "hyperparameters": {
            "n_neighbors": 5,
            "metric": "euclidean",
            "n_jobs": -1,
            "algorithm": "auto",
        },
        "hyperparameter_search": None,
        "cv_methodology": _CV_METHODOLOGY,
        "feature_cols": _FEATURE_COLS,
        "candidate_generation": _CANDIDATE_GENERATION,
        "notes": (
            "k=5 was chosen as a standard starting point. "
            "Euclidean distance on StandardScaler-normalised features."
        ),
    },

    "random_forest": {
        "display_name": "Random Forest",
        "library": "sklearn.ensemble.RandomForestClassifier",
        "role": "ensemble_baseline",
        "saved_as": "random_forest.joblib",
        "needs_scaling": False,
        "scaler": None,
        "hyperparameters": {
            "n_estimators": 200,
            "max_depth": 15,
            "min_samples_leaf": 5,
            "n_jobs": -1,
            "random_state": 42,
        },
        "hyperparameter_search": None,
        "cv_methodology": _CV_METHODOLOGY,
        "feature_cols": _FEATURE_COLS,
        "candidate_generation": _CANDIDATE_GENERATION,
        "notes": (
            "Tree-based model; no scaling required. "
            "max_depth=15 and min_samples_leaf=5 add mild regularisation "
            "to avoid overfitting on individual trip road networks."
        ),
    },

    "lgbm": {
        "display_name": "LightGBM (default params)",
        "library": "lightgbm.LGBMClassifier",
        "role": "strong_baseline",
        "saved_as": "lgbm.joblib",
        "needs_scaling": False,
        "scaler": None,
        "hyperparameters": {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "num_leaves": 63,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "n_jobs": -1,
            "random_state": 42,
            "verbose": -1,
        },
        "hyperparameter_search": None,
        "cv_methodology": _CV_METHODOLOGY,
        "feature_cols": _FEATURE_COLS,
        "candidate_generation": _CANDIDATE_GENERATION,
        "notes": (
            "Hand-tuned defaults chosen from LightGBM best-practice guidelines. "
            "Gradient boosted trees are scale-invariant so no StandardScaler is used. "
            "This model is then further tuned by Optuna to produce lgbm_tuned."
        ),
    },

    "lgbm_tuned": {
        "display_name": "LightGBM (Optuna-tuned)",
        "library": "lightgbm.LGBMClassifier",
        "role": "best_model",
        "saved_as": "lgbm_tuned.joblib",
        "needs_scaling": False,
        "scaler": None,
        "hyperparameters": {
            # Actual best params depend on the Optuna run output.
            # These are the search space bounds, not the final values.
            # Replace with study.best_trial.params after running the notebook.
            "__note__": (
                "Actual best params are determined by the Optuna study. "
                "Run the Kaggle notebook and record study.best_trial.params here."
            ),
        },
        "hyperparameter_search": {
            "library": "optuna",
            "direction": "maximize",
            "metric": "macro_f1",
            "n_trials": 50,
            "inner_cv_folds": 3,
            "search_space": {
                "n_estimators": {"type": "int", "low": 100, "high": 1000},
                "learning_rate": {"type": "float_log", "low": 0.01, "high": 0.3},
                "num_leaves": {"type": "int", "low": 15, "high": 255},
                "max_depth": {"type": "int", "low": 3, "high": 15},
                "min_child_samples": {"type": "int", "low": 5, "high": 100},
                "subsample": {"type": "float", "low": 0.5, "high": 1.0},
                "colsample_bytree": {"type": "float", "low": 0.5, "high": 1.0},
                "reg_alpha": {"type": "float_log", "low": 1e-8, "high": 10.0},
                "reg_lambda": {"type": "float_log", "low": 1e-8, "high": 10.0},
            },
            "sampler": "TPE (default Optuna sampler)",
            "pruner": "none (all 50 trials run to completion)",
        },
        "cv_methodology": _CV_METHODOLOGY,
        "feature_cols": _FEATURE_COLS,
        "candidate_generation": _CANDIDATE_GENERATION,
        "notes": (
            "Best-performing model selected for the final HMM pipeline. "
            "Optuna search runs 50 trials with 3-fold inner CV for speed; "
            "the winning params are then re-trained on all 80 trips."
        ),
    },

    "mlp": {
        "display_name": "MLP Neural Network",
        "library": "sklearn.neural_network.MLPClassifier",
        "role": "deep_baseline",
        "saved_as": "mlp.joblib",
        "needs_scaling": True,
        "scaler": "sklearn.preprocessing.StandardScaler",
        "hyperparameters": {
            "hidden_layer_sizes": (256, 128, 64),
            "activation": "relu",
            "solver": "adam",
            "max_iter": 200,
            "early_stopping": True,
            "random_state": 42,
        },
        "hyperparameter_search": None,
        "cv_methodology": _CV_METHODOLOGY,
        "feature_cols": _FEATURE_COLS,
        "candidate_generation": _CANDIDATE_GENERATION,
        "notes": (
            "Three hidden layers (256→128→64 neurons). "
            "StandardScaler required because MLP is sensitive to feature scale. "
            "Early stopping uses a 10 % internal validation split."
        ),
    },
}


# ─────────────────────────────────────────────
#  Accessor helpers
# ─────────────────────────────────────────────

def get_model_card(model_name: str) -> dict:
    """
    Return the model card for *model_name*.

    Raises KeyError if the model is not registered.
    """
    if model_name not in MODEL_REGISTRY:
        available = list(MODEL_REGISTRY.keys())
        raise KeyError(
            f"Model '{model_name}' not in registry. "
            f"Available: {available}"
        )
    return MODEL_REGISTRY[model_name]


def print_model_card(model_name: str | None = None) -> None:
    """
    Print a human-readable model card to stdout.

    If *model_name* is None, print all registered models.
    """
    targets = (
        {model_name: MODEL_REGISTRY[model_name]}
        if model_name
        else MODEL_REGISTRY
    )

    sep = "─" * 65

    print(f"\n{'=' * 65}")
    print("  PathSnap — Model Card & Hyperparameter Registry")
    print(f"{'=' * 65}")

    for name, card in targets.items():
        print(f"\n{sep}")
        print(f"  Model : {card['display_name']}")
        print(f"  Key   : {name}")
        print(f"  Role  : {card['role']}")
        print(f"  File  : {card['saved_as']}")
        print(f"  Lib   : {card['library']}")

        print(f"\n  Hyperparameters:")
        for k, v in card["hyperparameters"].items():
            print(f"    {k:30s} = {v}")

        if card["needs_scaling"]:
            print(f"\n  Preprocessing : {card['scaler']}")

        if card["hyperparameter_search"]:
            hs = card["hyperparameter_search"]
            print(f"\n  Hyperparameter Search ({hs['library']}):")
            print(f"    n_trials      = {hs['n_trials']}")
            print(f"    inner_cv_folds= {hs['inner_cv_folds']}")
            print(f"    metric        = {hs['metric']}")
            print(f"    sampler       = {hs['sampler']}")
            print(f"    Search space  :")
            for param, spec in hs["search_space"].items():
                print(f"      {param:25s} {spec}")
        else:
            print(f"\n  Hyperparameter Search : None (fixed configuration)")

        cv = card["cv_methodology"]
        print(f"\n  CV Strategy : {cv['strategy']} (n_splits={cv['n_splits']})")
        print(f"  Grouped by  : {cv['group_column']}")
        print(f"  Scoring     : {cv['scoring_metric']}")

        if card["notes"]:
            print(f"\n  Notes : {card['notes']}")

    print(f"\n{sep}\n")


def list_models() -> list[str]:
    """Return a sorted list of all registered model names."""
    return sorted(MODEL_REGISTRY.keys())


def models_requiring_scaling() -> list[str]:
    """Return names of models that need StandardScaler preprocessing."""
    return [n for n, c in MODEL_REGISTRY.items() if c["needs_scaling"]]


if __name__ == "__main__":
    print_model_card()
