"""
PathSnap — Feature Input Validation
src/validate.py

Validates feature rows before they are passed to a model for inference.
Called by the API (api/main.py) and can be reused in any script that
builds features with src/features.py.

Design principles
-----------------
- Raise ValueError with a descriptive message so the caller can surface
  the issue clearly rather than producing a silent NaN prediction.
- Use warnings for soft issues that do not prevent inference (e.g. a
  speed value that is unusually high but not impossible).
- Never mutate the input DataFrame; validation is read-only.
"""

import math
import warnings
import numpy as np
import pandas as pd

from features import FEATURE_COLS


# ─────────────────────────────────────────────
#  Per-column valid ranges
#  (derived from the data description and domain knowledge)
# ─────────────────────────────────────────────

#  col_name  → (min_inclusive, max_inclusive, allow_zero, description)
_COLUMN_RANGES = {
    "dist_to_arc_m": (0.0, 200.0, True,
                      "Perpendicular distance from GPS fix to arc (metres)"),
    "bearing_diff": (0.0, 180.0, True,
                     "Absolute bearing difference (degrees, folded to [0,180])"),
    "cos_bearing_diff": (-1.0, 1.0, True,
                         "Cosine of bearing difference"),
    "sin_bearing_diff": (-1.0, 1.0, True,
                         "Sine of bearing difference"),
    "speed_mps": (0.0, 60.0, True,
                  "Vehicle speed in m/s (≈ 216 km/h upper bound)"),
    "time_delta_sec": (0.0, 300.0, True,
                       "Time since previous GPS fix (seconds)"),
    "arc_length_m": (0.0, 5000.0, False,
                     "Length of candidate arc in metres"),
    "candidate_rank": (1, 10, False,
                       "1-based rank of candidate arc by proximity"),
}

# Soft warning thresholds (values that are technically legal but suspicious)
_SOFT_THRESHOLDS = {
    "speed_mps": 30.0,          # > 108 km/h — possible but rare in urban areas
    "dist_to_arc_m": 100.0,     # > 100 m — candidate is very far from GPS fix
    "time_delta_sec": 30.0,     # > 30 s gap — might indicate a GPS outage
}


# ─────────────────────────────────────────────
#  Public API
# ─────────────────────────────────────────────

def validate_feature_dataframe(df: pd.DataFrame, raise_on_error: bool = True) -> list[str]:
    """
    Validate a feature DataFrame produced by compute_features_for_trip().

    Parameters
    ----------
    df : pd.DataFrame
        Must contain all columns in FEATURE_COLS.
    raise_on_error : bool
        If True (default), raise ValueError on any hard violation.
        If False, return all error strings instead.

    Returns
    -------
    list[str]
        List of error/warning messages (empty when everything is clean).

    Raises
    ------
    ValueError
        If raise_on_error=True and any hard validation rule fails.
    """
    errors: list[str] = []
    warnings_list: list[str] = []

    # ── 1. Required columns present ──────────────────────────────────
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        msg = f"Missing required feature columns: {missing}"
        errors.append(msg)

    if errors:
        # Can't proceed with further checks if columns are absent
        if raise_on_error:
            raise ValueError("\n".join(errors))
        return errors

    # ── 2. Empty DataFrame ───────────────────────────────────────────
    if df.empty:
        msg = "Feature DataFrame is empty (0 rows)."
        errors.append(msg)
        if raise_on_error:
            raise ValueError(msg)
        return errors

    # ── 3. NaN / Inf check ───────────────────────────────────────────
    for col in FEATURE_COLS:
        series = df[col]
        n_nan = series.isna().sum()
        if n_nan > 0:
            errors.append(
                f"Column '{col}' contains {n_nan} NaN value(s). "
                "Check upstream feature computation."
            )
        n_inf = np.isinf(series.replace([None], np.nan).astype(float)).sum()
        if n_inf > 0:
            errors.append(
                f"Column '{col}' contains {n_inf} infinite value(s). "
                "Common cause: division by zero in speed or distance calculation."
            )

    # ── 4. Range checks ──────────────────────────────────────────────
    for col, (lo, hi, allow_zero, desc) in _COLUMN_RANGES.items():
        if col not in df.columns:
            continue
        series = df[col].dropna()
        if not allow_zero:
            zeros = (series == 0).sum()
            if zeros > 0:
                errors.append(
                    f"Column '{col}' ({desc}) has {zeros} zero value(s), "
                    "which is physically invalid."
                )
        out_lo = (series < lo).sum()
        out_hi = (series > hi).sum()
        if out_lo > 0:
            errors.append(
                f"Column '{col}' has {out_lo} value(s) below minimum {lo}. "
                f"({desc})"
            )
        if out_hi > 0:
            errors.append(
                f"Column '{col}' has {out_hi} value(s) above maximum {hi}. "
                f"({desc})"
            )

    # ── 5. Soft warnings ─────────────────────────────────────────────
    for col, threshold in _SOFT_THRESHOLDS.items():
        if col not in df.columns:
            continue
        n_above = (df[col] > threshold).sum()
        if n_above > 0:
            warnings_list.append(
                f"[WARNING] Column '{col}' has {n_above} value(s) above "
                f"soft threshold {threshold}. Verify these rows."
            )

    # ── 6. candidate_rank must be integer-valued ─────────────────────
    if "candidate_rank" in df.columns:
        non_int = df["candidate_rank"].dropna()
        if not np.array_equal(non_int, non_int.astype(int)):
            errors.append(
                "Column 'candidate_rank' contains non-integer values. "
                "Expected integers in [1, 10]."
            )

    # Emit soft warnings via the warnings module (never blocks inference)
    for w in warnings_list:
        warnings.warn(w, UserWarning, stacklevel=2)

    if errors and raise_on_error:
        raise ValueError(
            f"Feature validation failed with {len(errors)} error(s):\n"
            + "\n".join(f"  • {e}" for e in errors)
        )

    return errors + warnings_list


def validate_single_fix(fix_features: dict, raise_on_error: bool = True) -> list[str]:
    """
    Validate a single GPS-fix feature dict (as produced per row in
    compute_features_for_trip before DataFrame assembly).

    Parameters
    ----------
    fix_features : dict
        Keys should include all FEATURE_COLS.
    raise_on_error : bool
        Same semantics as validate_feature_dataframe.

    Returns
    -------
    list[str]  — error/warning strings (empty = clean)
    """
    df = pd.DataFrame([fix_features])
    return validate_feature_dataframe(df, raise_on_error=raise_on_error)


def summarise_validation(df: pd.DataFrame) -> dict:
    """
    Return a summary dict suitable for logging or the API health endpoint.

    Example return value::

        {
            "n_rows": 1200,
            "n_features": 8,
            "has_nan": False,
            "has_inf": False,
            "range_violations": 0,
            "warnings": ["[WARNING] speed_mps has 3 values above 30.0"],
            "valid": True
        }
    """
    issues = validate_feature_dataframe(df, raise_on_error=False)
    errors = [i for i in issues if not i.startswith("[WARNING]")]
    warns = [i for i in issues if i.startswith("[WARNING]")]

    nan_count = df[FEATURE_COLS].isna().sum().sum() if all(c in df.columns for c in FEATURE_COLS) else -1
    inf_count = (
        np.isinf(df[FEATURE_COLS].astype(float)).sum().sum()
        if all(c in df.columns for c in FEATURE_COLS) else -1
    )

    return {
        "n_rows": len(df),
        "n_features": len(FEATURE_COLS),
        "has_nan": bool(nan_count > 0),
        "has_inf": bool(inf_count > 0),
        "range_violations": len(errors),
        "warnings": warns,
        "valid": len(errors) == 0,
    }
