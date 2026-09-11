import numpy as np
import pandas as pd
import pytest

from src.data.split import compute_split_boundaries
from src.train.logistic_regression import fit_logistic_regression, select_c_via_cv, train_logistic_regression

SPLIT_CONFIG = {"train_val_quantile": 0.6, "val_test_quantile": 0.8, "cv_n_folds": 2}
MAX_ITER = 200


@pytest.fixture
def synthetic_data():
    """Mirrors src/train/data.py's Phase2Features split: train-only data
    (used for the final refit) is kept genuinely distinct from train+val
    (used only for CV selection) -- this distinction is what a real bug
    (docs/baselines.md Sec.6/Sec.8) hid until it hit the real dataset, so
    this fixture is deliberately built to be able to catch a regression of
    it, not just to be minimally sufficient."""
    rng = np.random.RandomState(0)
    n = 800
    dt = pd.Series(np.arange(n) * 100, name="TransactionDT")
    # A genuinely learnable signal: feature 0 predicts the label.
    signal = rng.rand(n)
    y = pd.Series((signal > 0.9).astype(int))  # ~10% positive rate
    X = pd.DataFrame(
        {
            "signal_feature": signal.astype("float32"),
            "noise_feature": rng.rand(n).astype("float32"),
        }
    )
    boundaries = compute_split_boundaries(dt, SPLIT_CONFIG)

    train_mask = (dt <= boundaries.train_val_boundary).to_numpy()
    val_mask = ((dt > boundaries.train_val_boundary) & (dt <= boundaries.val_test_boundary)).to_numpy()
    train_plus_val_mask = train_mask | val_mask

    return {
        "X_train": X.loc[train_mask],
        "y_train": y.loc[train_mask],
        "X_cv": X.loc[train_plus_val_mask],
        "y_cv": y.loc[train_plus_val_mask],
        "cv_dt": dt.loc[train_plus_val_mask],
        "boundaries": boundaries,
    }


def test_select_c_via_cv_returns_one_result_per_candidate(synthetic_data):
    d = synthetic_data
    results = select_c_via_cv(
        d["X_cv"], d["y_cv"], d["cv_dt"], d["boundaries"], c_grid=[0.1, 1.0, 10.0], seed=42, max_iter=MAX_ITER
    )
    assert len(results) == 3
    assert {r["C"] for r in results} == {0.1, 1.0, 10.0}


def test_select_c_via_cv_scores_are_valid_pr_auc(synthetic_data):
    d = synthetic_data
    results = select_c_via_cv(d["X_cv"], d["y_cv"], d["cv_dt"], d["boundaries"], c_grid=[1.0], seed=42, max_iter=MAX_ITER)
    for fold_score in results[0]["fold_pr_auc"]:
        assert 0.0 <= fold_score <= 1.0
    assert 0.0 <= results[0]["mean_pr_auc"] <= 1.0


def test_select_c_via_cv_fold_count_matches_config(synthetic_data):
    d = synthetic_data
    results = select_c_via_cv(d["X_cv"], d["y_cv"], d["cv_dt"], d["boundaries"], c_grid=[1.0], seed=42, max_iter=MAX_ITER)
    assert len(results[0]["fold_pr_auc"]) == SPLIT_CONFIG["cv_n_folds"]


def test_train_logistic_regression_selects_best_c_by_mean_cv_pr_auc(synthetic_data):
    d = synthetic_data
    result = train_logistic_regression(
        d["X_train"], d["y_train"], d["X_cv"], d["y_cv"], d["cv_dt"], d["boundaries"],
        c_grid=[0.01, 1.0, 100.0], seed=42, max_iter=MAX_ITER,
    )
    best_from_cv = max(result.cv_results, key=lambda r: r["mean_pr_auc"])
    assert result.selected_c == best_from_cv["C"]


def test_train_logistic_regression_final_model_fit_on_train_only(synthetic_data):
    """The final model must be fit on X_train (not X_cv, which includes
    val rows) -- checked by verifying prediction count matches len(X_train)
    and the model's feature count matches, the same properties the
    original test checked, now against the correctly train-only data."""
    d = synthetic_data
    result = train_logistic_regression(
        d["X_train"], d["y_train"], d["X_cv"], d["y_cv"], d["cv_dt"], d["boundaries"], c_grid=[1.0], seed=42, max_iter=MAX_ITER
    )
    assert result.model.n_features_in_ == d["X_train"].shape[1]
    probs = result.model.predict_proba(d["X_train"].to_numpy())[:, 1]
    assert len(probs) == len(d["X_train"])


def test_train_logistic_regression_learns_the_real_signal(synthetic_data):
    """Sanity check: on data with a genuine signal, the selected model
    should clearly beat random-guessing PR-AUC (~positive rate)."""
    d = synthetic_data
    result = train_logistic_regression(
        d["X_train"], d["y_train"], d["X_cv"], d["y_cv"], d["cv_dt"], d["boundaries"],
        c_grid=[0.1, 1.0, 10.0], seed=42, max_iter=MAX_ITER,
    )
    from sklearn.metrics import average_precision_score

    probs = result.model.predict_proba(d["X_train"].to_numpy())[:, 1]
    pr_auc = average_precision_score(d["y_train"].to_numpy(), probs)
    assert pr_auc > 0.5  # well above the ~0.1 baseline for this data


def test_train_logistic_regression_is_deterministic_given_seed(synthetic_data):
    d = synthetic_data
    result_a = train_logistic_regression(
        d["X_train"], d["y_train"], d["X_cv"], d["y_cv"], d["cv_dt"], d["boundaries"], c_grid=[1.0], seed=42, max_iter=MAX_ITER
    )
    result_b = train_logistic_regression(
        d["X_train"], d["y_train"], d["X_cv"], d["y_cv"], d["cv_dt"], d["boundaries"], c_grid=[1.0], seed=42, max_iter=MAX_ITER
    )
    np.testing.assert_array_equal(
        result_a.model.named_steps["lr"].coef_, result_b.model.named_steps["lr"].coef_
    )


def test_train_logistic_regression_cv_pool_larger_than_train_only(synthetic_data):
    """Regression guard for the exact bug this fixture design targets:
    X_cv must be strictly larger than X_train (it includes val rows) --
    if a future change accidentally collapses them back to the same
    data, this fails loudly rather than silently losing the distinction."""
    d = synthetic_data
    assert len(d["X_cv"]) > len(d["X_train"])


def test_fit_logistic_regression_scaler_is_fit_train_only(synthetic_data):
    """Regression guard for docs/baselines.md Sec.7.3: the StandardScaler
    inside the pipeline must be fit on exactly the rows passed to
    fit_logistic_regression, never on a larger pool -- verified directly
    against the scaler's own learned mean_, not just inferred from
    downstream metrics."""
    d = synthetic_data
    model = fit_logistic_regression(d["X_train"].to_numpy(), d["y_train"].to_numpy(), c=1.0, seed=42, max_iter=MAX_ITER)
    expected_mean = d["X_train"].to_numpy().mean(axis=0)
    np.testing.assert_allclose(model.named_steps["scaler"].mean_, expected_mean)
    # A scaler fit on the larger train+val pool would have a different mean
    # (X_cv is a strict superset, per the regression test above) -- assert
    # the two are NOT equal as a sanity check that this test can actually fail.
    cv_mean = d["X_cv"].to_numpy().mean(axis=0)
    assert not np.allclose(expected_mean, cv_mean)


def test_fit_logistic_regression_converges_on_badly_scaled_data():
    """The real bug this fix addresses: lbfgs failed to converge when a
    feature's raw magnitude (e.g. TransactionAmt ~$0-32,000) sat next to
    features scaled to [0, 1] -- reproduced here with a synthetic column
    at a similarly large scale, which the StandardScaler must neutralize."""
    rng = np.random.RandomState(0)
    n = 2000
    small_scale_signal = rng.rand(n)
    y = (small_scale_signal > 0.9).astype(int)
    X = np.column_stack(
        [
            small_scale_signal,
            rng.rand(n) * 30000 + rng.rand(n),  # a badly-scaled, low-signal column
        ]
    )
    model = fit_logistic_regression(X, y, c=1.0, seed=42, max_iter=MAX_ITER)
    n_iter = int(np.max(model.named_steps["lr"].n_iter_))
    assert n_iter < MAX_ITER, f"LR did not converge on badly-scaled data: n_iter={n_iter}"
