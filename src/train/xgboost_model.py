"""XGBoost baseline (CLAUDE.md Sec.6 required baseline #2).

Hyperparameter selection (max_depth x learning_rate grid) uses Phase 2's
expanding-window CV folds (`src/train/cv.py`) -- never random k-fold
(CLAUDE.md Sec.6) -- scored by mean best-iteration PR-AUC across folds,
where "PR-AUC" is XGBoost's own `aucpr` eval metric (the same quantity
`sklearn.metrics.average_precision_score` reports, used natively as the
early-stopping criterion). CV selection is deliberately given
`(X_cv, y_cv, cv_dt)` -- Phase 2's CV folds span the primary train+val
region by design (docs/pipeline.md Sec.1), so this must be
`features.X_train_plus_val` etc. (`src/train/data.py`), NOT
`features.X_train` alone: the latter's DT range stops at
`train_val_boundary` and cannot cover the last fold's eval window, which
falls inside the primary "val" partition's own DT range (this was a real
bug, caught by the real-data test producing a zero-row fold -- see
`docs/baselines.md` Sec.6/Sec.8). The final model is then refit on
`(X_train, y_train)` ONLY, with early stopping against `X_val`/`y_val` --
this is what "use validation data for model selection" concretely means
for XGBoost: it decides `n_estimators` (via `best_iteration`), not any
other hyperparameter.

`subsample=0.8, colsample_bytree=0.8` are set deliberately (not left at
XGBoost's `1.0` default): besides being standard regularization practice,
they make the seed-stability check in `tests/test_train_pipeline_real_data.py`
meaningful -- with no row/feature subsampling, XGBoost's histogram-based
tree construction is close to deterministic regardless of `random_state`,
which would make a seed-stability check trivially pass without actually
exercising anything.

A plain `xgboost.XGBClassifier` is used, matching
`concrete.ml.sklearn.XGBClassifier`'s constructor/fit surface for Phase 5+
FHE compilation, per the same reasoning as `src/train/logistic_regression.py`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

from src.train.cv import build_positional_cv_folds
from src.train.imbalance import compute_scale_pos_weight
from src.data import split as data_split
from src.logging_setup import get_logger

logger = get_logger(__name__)


@dataclass
class XGBoostResult:
    model: xgb.XGBClassifier
    selected_max_depth: int
    selected_learning_rate: float
    selected_n_estimators: int
    cv_results: list[dict[str, Any]]


def build_xgb_classifier(
    max_depth: int, learning_rate: float, scale_pos_weight: float, seed: int, n_estimators: int = 500
) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        early_stopping_rounds=20,
        random_state=seed,
        n_jobs=-1,
    )


def select_hyperparameters_via_cv(
    X_cv: pd.DataFrame,
    y_cv: pd.Series,
    cv_dt: pd.Series,
    boundaries: data_split.SplitBoundaries,
    max_depth_grid: list[int],
    learning_rate_grid: list[float],
    seed: int,
) -> list[dict[str, Any]]:
    """Mean CV best-iteration PR-AUC for each (max_depth, learning_rate) combo.

    `X_cv`/`y_cv`/`cv_dt` must span the primary train+val region (e.g.
    `features.X_train_plus_val` from `src/train/data.py`) -- see module
    docstring for why train-only data breaks the last fold.
    """
    folds = build_positional_cv_folds(cv_dt, boundaries)
    X_arr, y_arr = X_cv.to_numpy(), y_cv.to_numpy()

    results = []
    for max_depth in max_depth_grid:
        for learning_rate in learning_rate_grid:
            fold_scores = []
            fold_best_iterations = []
            for fold_index, train_pos, eval_pos in folds:
                fold_scale_pos_weight = compute_scale_pos_weight(y_arr[train_pos])
                model = build_xgb_classifier(max_depth, learning_rate, fold_scale_pos_weight, seed)
                model.fit(
                    X_arr[train_pos],
                    y_arr[train_pos],
                    eval_set=[(X_arr[eval_pos], y_arr[eval_pos])],
                    verbose=False,
                )
                fold_scores.append(float(model.best_score))
                fold_best_iterations.append(int(model.best_iteration))
                logger.info(
                    "XGBoost CV fold scored",
                    extra={
                        "extra_fields": {
                            "max_depth": max_depth,
                            "learning_rate": learning_rate,
                            "fold_index": fold_index,
                            "pr_auc": model.best_score,
                            "best_iteration": model.best_iteration,
                        }
                    },
                )
            results.append(
                {
                    "max_depth": max_depth,
                    "learning_rate": learning_rate,
                    "fold_pr_auc": fold_scores,
                    "fold_best_iterations": fold_best_iterations,
                    "mean_pr_auc": float(np.mean(fold_scores)),
                    "std_pr_auc": float(np.std(fold_scores)),
                }
            )
    return results


def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    X_cv: pd.DataFrame,
    y_cv: pd.Series,
    cv_dt: pd.Series,
    boundaries: data_split.SplitBoundaries,
    max_depth_grid: list[int],
    learning_rate_grid: list[float],
    seed: int,
) -> XGBoostResult:
    """`(X_train, y_train)`: train-only, used for the final refit (with
    `X_val`/`y_val` for early stopping). `(X_cv, y_cv, cv_dt)`: train+val,
    used only for CV-based hyperparameter selection. See module docstring
    for why these must NOT be the same data."""
    cv_results = select_hyperparameters_via_cv(
        X_cv, y_cv, cv_dt, boundaries, max_depth_grid, learning_rate_grid, seed
    )
    best = max(cv_results, key=lambda r: r["mean_pr_auc"])
    logger.info(
        "Selected XGBoost hyperparameters via CV",
        extra={
            "extra_fields": {
                "selected_max_depth": best["max_depth"],
                "selected_learning_rate": best["learning_rate"],
                "mean_cv_pr_auc": best["mean_pr_auc"],
            }
        },
    )

    # Final refit on the FULL train partition; X_val/y_val is used ONLY to
    # pick n_estimators via early stopping here -- never for the
    # hyperparameter grid above, and never for the headline metrics
    # reported by src/train/pipeline.py (those are computed separately,
    # after this model is already fit).
    final_scale_pos_weight = compute_scale_pos_weight(y_train.to_numpy())
    final_model = build_xgb_classifier(best["max_depth"], best["learning_rate"], final_scale_pos_weight, seed)
    final_model.fit(
        X_train.to_numpy(),
        y_train.to_numpy(),
        eval_set=[(X_val.to_numpy(), y_val.to_numpy())],
        verbose=False,
    )
    logger.info(
        "Final XGBoost model fit",
        extra={"extra_fields": {"best_iteration": final_model.best_iteration, "best_score": final_model.best_score}},
    )

    return XGBoostResult(
        model=final_model,
        selected_max_depth=best["max_depth"],
        selected_learning_rate=best["learning_rate"],
        selected_n_estimators=int(final_model.best_iteration) + 1,
        cv_results=cv_results,
    )
