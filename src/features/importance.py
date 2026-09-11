"""XGBoost total-gain feature-importance ranking (`docs/plan.md` Phase 4).

`docs/plan.md` Phase 4's first task is "compute feature importances from
the Phase 3 XGBoost model (or permutation importance)" -- this project
uses the XGBoost model's own `total_gain` (see module-level rationale
below), not permutation importance, per this session's explicit decision.

The Phase 3 XGBoost model (`src/train/xgboost_model.py`) was fit on
`.to_numpy()`, never a DataFrame, so its booster only knows features by
position (`f0`, `f1`, ...). Every function here takes the caller's
`feature_names` list explicitly rather than trusting `booster.feature_names`
(which is `None` for a numpy-fit booster) -- mapping `f{i}` back to a real
column name is this module's job, not something to assume the booster
already knows.
"""

from __future__ import annotations

import pandas as pd
import xgboost as xgb


class ImportanceError(Exception):
    """Raised when the booster and the caller's feature names disagree."""


def xgboost_total_gain_ranking(booster: xgb.Booster, feature_names: list[str]) -> pd.DataFrame:
    """Rank `feature_names` by the booster's `total_gain` importance.

    `total_gain` (the sum of gain across every split that used a feature),
    not average `gain`, is used as the ranking criterion: average gain
    overweights a feature that was used in a single lucky, high-gain split
    but contributed to the model far less overall than a feature used
    moderately across many splits. `gain` and `weight` (split count) are
    kept as columns for transparency, not used for the sort.

    `booster.get_score()` omits any feature the booster never split on --
    these are explicitly zero-filled here rather than silently dropped, so
    every one of Phase 2's `len(feature_names)` columns gets a rank.

    Ties (including the common case of many zero-gain features) are
    broken by feature name, ascending -- so the ranking is a pure,
    deterministic function of `(booster, feature_names)`, never dependent
    on dict iteration order.
    """
    if booster.num_features() != len(feature_names):
        raise ImportanceError(
            f"booster.num_features()={booster.num_features()} does not match "
            f"len(feature_names)={len(feature_names)} -- the booster was not "
            f"fit on exactly this feature matrix"
        )

    total_gain = booster.get_score(importance_type="total_gain")
    gain = booster.get_score(importance_type="gain")
    weight = booster.get_score(importance_type="weight")

    rows = [
        {
            "feature": name,
            "total_gain": float(total_gain.get(f"f{i}", 0.0)),
            "gain": float(gain.get(f"f{i}", 0.0)),
            "weight": float(weight.get(f"f{i}", 0.0)),
        }
        for i, name in enumerate(feature_names)
    ]
    ranking = (
        pd.DataFrame(rows)
        .sort_values(["total_gain", "feature"], ascending=[False, True])
        .reset_index(drop=True)
    )
    ranking["rank"] = ranking.index + 1
    return ranking


def topk_jaccard(ranking_a: pd.DataFrame, ranking_b: pd.DataFrame, k: int) -> float:
    """Jaccard similarity of the top-`k` feature sets from two rankings.

    Used to measure ranking stability across seeds (`docs/plan.md` Phase 4
    exit criteria imply tiers should be a *reproducible* artifact -- a
    ranking that reshuffles wildly across seeds would undermine that even
    if each individual ranking is internally deterministic).
    """
    set_a = set(ranking_a.sort_values("rank")["feature"].head(k))
    set_b = set(ranking_b.sort_values("rank")["feature"].head(k))
    if not set_a and not set_b:
        return 1.0
    return len(set_a & set_b) / len(set_a | set_b)
