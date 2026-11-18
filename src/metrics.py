"""
PathSnap — Evaluation Metrics
src/metrics.py
"""

import numpy as np
from sklearn.metrics import f1_score, log_loss, accuracy_score


def macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average='macro', zero_division=0)


def top1_accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)


def compute_log_loss(y_true, y_proba):
    try:
        return log_loss(y_true, y_proba)
    except Exception:
        return float('nan')


def evaluate_model(y_true, y_pred, y_proba=None):
    """Return dict of all required metrics."""
    results = {
        'macro_f1': macro_f1(y_true, y_pred),
        'top1_accuracy': top1_accuracy(y_true, y_pred),
    }
    if y_proba is not None:
        results['log_loss'] = compute_log_loss(y_true, y_proba)
    return results
