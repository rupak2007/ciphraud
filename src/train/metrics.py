"""Evaluation metrics for the Phase 3 baselines.

CLAUDE.md Sec.6 mandates PR-AUC as the primary metric and explicitly
forbids accuracy as a headline result given the ~3.5% fraud rate
(docs/eda.md Sec.1 measured 3.499%). Every function here is a pure
function of (y_true, y_prob) or (y_true, y_pred) -- no fitting, no I/O --
so it's identically testable on synthetic and real data.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def threshold_independent_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float]:
    """PR-AUC (primary) and ROC-AUC (secondary) -- CLAUDE.md Sec.6.

    PR-AUC here is `average_precision_score`, the standard, exact area
    under the precision-recall curve (not a trapezoidal approximation),
    which is what "PR-AUC" means throughout this project's docs.
    """
    return {
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
    }


def metrics_at_threshold(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict[str, Any]:
    """Precision, recall, F1, F2, and confusion matrix at a fixed threshold.

    F2 (beta=2) weights recall over precision -- appropriate for fraud
    detection, where a missed fraud (false negative) is typically costlier
    than a false alarm (CLAUDE.md Sec.6's F2 requirement).
    """
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(fbeta_score(y_true, y_pred, beta=1.0, zero_division=0)),
        "f2": float(fbeta_score(y_true, y_pred, beta=2.0, zero_division=0)),
        "confusion_matrix": {
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        },
    }


def select_threshold(
    y_true: np.ndarray, y_prob: np.ndarray, metric: str = "f1"
) -> dict[str, Any]:
    """Pick the threshold maximizing `metric` (f1 or f2) via `precision_recall_curve`.

    The caller must pass VALIDATION data only -- this function has no way
    to enforce that itself, so `src/train/pipeline.py` is the one place
    threshold selection is ever invoked, always with (X_val, y_val), never
    (X_test, y_test). See docs/baselines.md Sec.3 for why this is a
    leakage-relevant boundary, not just a convention.
    """
    if metric not in ("f1", "f2"):
        raise ValueError(f"metric must be 'f1' or 'f2', got {metric!r}")
    beta = 1.0 if metric == "f1" else 2.0

    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    # precision_recall_curve returns len(thresholds) == len(precision) - 1;
    # drop the last (threshold=+inf, undefined-recall) precision/recall pair.
    precision, recall = precision[:-1], recall[:-1]

    with np.errstate(divide="ignore", invalid="ignore"):
        f_beta = (
            (1 + beta**2)
            * precision
            * recall
            / np.where((beta**2 * precision + recall) == 0, np.nan, beta**2 * precision + recall)
        )
    f_beta = np.nan_to_num(f_beta, nan=0.0)

    best_idx = int(np.argmax(f_beta))
    best_threshold = float(thresholds[best_idx])
    return {
        "metric_optimized": metric,
        "selected_threshold": best_threshold,
        **metrics_at_threshold(y_true, y_prob, best_threshold),
    }


def full_evaluation(
    y_true: np.ndarray, y_prob: np.ndarray, fixed_thresholds: tuple[float, ...] = (0.1, 0.3, 0.5)
) -> dict[str, Any]:
    """Everything docs/plan.md Phase 3 asks for, computed once: PR-AUC,
    ROC-AUC, an F1-selected threshold, an F2-selected threshold, and a
    fixed-threshold sweep for direct comparability across models/runs."""
    return {
        **threshold_independent_metrics(y_true, y_prob),
        "f1_selected": select_threshold(y_true, y_prob, metric="f1"),
        "f2_selected": select_threshold(y_true, y_prob, metric="f2"),
        "fixed_thresholds": [metrics_at_threshold(y_true, y_prob, t) for t in fixed_thresholds],
    }


def error_analysis(
    df_index: pd.Index, y_true: np.ndarray, y_prob: np.ndarray, threshold: float
) -> dict[str, Any]:
    """False-positive / false-negative characterization at a fixed threshold
    (CLAUDE.md Sec.6: "error analysis... for at least the strongest
    plaintext model"). Returns TransactionIDs and predicted probabilities
    for both error classes, capped, for inclusion in a report without
    dumping every row.

    `y_true`/`y_prob` must be plain numpy arrays (as every other function
    in this module expects) whose row order matches `df_index`.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    false_positive_positions = np.flatnonzero((y_pred == 1) & (y_true == 0))
    false_negative_positions = np.flatnonzero((y_pred == 0) & (y_true == 1))

    def _sample(positions: np.ndarray, cap: int = 20) -> list[dict[str, Any]]:
        capped = positions[:cap]
        return [
            {"TransactionID": int(df_index[p]), "predicted_probability": float(y_prob[p])}
            for p in capped
        ]

    n_negative = int((y_true == 0).sum())
    n_positive = int((y_true == 1).sum())
    return {
        "threshold": float(threshold),
        "n_false_positive": int(len(false_positive_positions)),
        "n_false_negative": int(len(false_negative_positions)),
        "false_positive_rate_of_negatives": len(false_positive_positions) / max(n_negative, 1),
        "false_negative_rate_of_positives": len(false_negative_positions) / max(n_positive, 1),
        "sample_false_positives": _sample(false_positive_positions),
        "sample_false_negatives": _sample(false_negative_positions),
    }
