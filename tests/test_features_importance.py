import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from src.features.importance import ImportanceError, topk_jaccard, xgboost_total_gain_ranking


def _fit_booster(n_features=5, n_estimators=1, max_depth=1, seed=0) -> xgb.Booster:
    """A tiny, deliberately shallow booster where feature 0 is the only
    informative column -- with n_estimators=1, max_depth=1 it can split on
    at most one feature, so every other feature is guaranteed zero-gain."""
    rng = np.random.RandomState(seed)
    n = 500
    X = rng.rand(n, n_features)
    y = (X[:, 0] > 0.5).astype(int)
    model = xgb.XGBClassifier(
        n_estimators=n_estimators, max_depth=max_depth, random_state=seed, eval_metric="aucpr"
    )
    model.fit(X, y)
    return model.get_booster()


def test_xgboost_total_gain_ranking_ranks_informative_feature_first():
    booster = _fit_booster(n_features=5, n_estimators=1, max_depth=1)
    names = [f"feat_{i}" for i in range(5)]
    ranking = xgboost_total_gain_ranking(booster, names)
    assert ranking.iloc[0]["feature"] == "feat_0"
    assert ranking.iloc[0]["rank"] == 1


def test_xgboost_total_gain_ranking_zero_fills_unused_features():
    booster = _fit_booster(n_features=5, n_estimators=1, max_depth=1)
    names = [f"feat_{i}" for i in range(5)]
    ranking = xgboost_total_gain_ranking(booster, names)
    assert len(ranking) == 5
    unused = ranking[ranking["feature"] != "feat_0"]
    assert (unused["total_gain"] == 0.0).all()


def test_xgboost_total_gain_ranking_raises_on_feature_count_mismatch():
    booster = _fit_booster(n_features=5)
    with pytest.raises(ImportanceError):
        xgboost_total_gain_ranking(booster, [f"feat_{i}" for i in range(4)])


def test_xgboost_total_gain_ranking_ties_broken_by_feature_name():
    booster = _fit_booster(n_features=5, n_estimators=1, max_depth=1)
    # Position 0 (informative) named "z_feat" so it doesn't win the
    # alphabetical tie-break by coincidence -- it must rank first purely
    # because of nonzero total_gain.
    names = ["z_feat", "a_feat", "feat_2", "feat_3", "feat_4"]
    ranking = xgboost_total_gain_ranking(booster, names)
    assert ranking.iloc[0]["feature"] == "z_feat"
    zero_gain_order = ranking[ranking["total_gain"] == 0.0]["feature"].tolist()
    assert zero_gain_order == sorted(zero_gain_order)


def test_xgboost_total_gain_ranking_ranks_are_1_to_n():
    booster = _fit_booster(n_features=5, n_estimators=1, max_depth=1)
    ranking = xgboost_total_gain_ranking(booster, [f"feat_{i}" for i in range(5)])
    assert sorted(ranking["rank"].tolist()) == [1, 2, 3, 4, 5]


def test_topk_jaccard_identical_rankings_is_one():
    booster = _fit_booster(n_features=10, n_estimators=3, max_depth=2)
    names = [f"feat_{i}" for i in range(10)]
    ranking = xgboost_total_gain_ranking(booster, names)
    assert topk_jaccard(ranking, ranking, k=5) == 1.0


def test_topk_jaccard_disjoint_sets_is_zero():
    df_a = pd.DataFrame({"feature": ["a", "b"], "rank": [1, 2]})
    df_b = pd.DataFrame({"feature": ["c", "d"], "rank": [1, 2]})
    assert topk_jaccard(df_a, df_b, k=2) == 0.0


def test_topk_jaccard_partial_overlap():
    df_a = pd.DataFrame({"feature": ["a", "b", "c"], "rank": [1, 2, 3]})
    df_b = pd.DataFrame({"feature": ["a", "b", "d"], "rank": [1, 2, 3]})
    assert topk_jaccard(df_a, df_b, k=3) == pytest.approx(2 / 4)
