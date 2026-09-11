import pandas as pd
import pytest

from src.data.split import (
    SplitError,
    assign_cv_eval_fold,
    assign_primary_split,
    build_split_assignments,
    compute_split_boundaries,
    iter_expanding_window_folds,
)

SPLIT_CONFIG = {"train_val_quantile": 0.6, "val_test_quantile": 0.8, "cv_n_folds": 3}


@pytest.fixture
def dt_series() -> pd.Series:
    # 100 evenly spaced DT values, 0..9900 (step 100) -- easy to reason
    # about quantile boundaries by hand.
    return pd.Series(range(0, 10000, 100))


def test_compute_split_boundaries_are_within_range(dt_series):
    boundaries = compute_split_boundaries(dt_series, SPLIT_CONFIG)
    assert boundaries.dt_min == 0
    assert boundaries.dt_max == 9900
    assert boundaries.dt_min < boundaries.train_val_boundary < boundaries.val_test_boundary < boundaries.dt_max
    assert boundaries.train_val_boundary == round(0.6 * 9900)
    assert boundaries.val_test_boundary == round(0.8 * 9900)


def test_compute_split_boundaries_cv_boundaries_strictly_increasing(dt_series):
    boundaries = compute_split_boundaries(dt_series, SPLIT_CONFIG)
    cv = boundaries.cv_boundaries
    assert len(cv) == SPLIT_CONFIG["cv_n_folds"] + 1
    assert list(cv) == sorted(cv)
    assert len(set(cv)) == len(cv)


def test_compute_split_boundaries_last_cv_boundary_equals_val_test_boundary(dt_series):
    boundaries = compute_split_boundaries(dt_series, SPLIT_CONFIG)
    assert boundaries.cv_boundaries[-1] == boundaries.val_test_boundary


def test_compute_split_boundaries_rejects_out_of_order_quantiles(dt_series):
    with pytest.raises(SplitError, match="train_val_quantile"):
        compute_split_boundaries(dt_series, {"train_val_quantile": 0.8, "val_test_quantile": 0.6})


def test_compute_split_boundaries_rejects_quantiles_outside_open_unit_interval(dt_series):
    with pytest.raises(SplitError):
        compute_split_boundaries(dt_series, {"train_val_quantile": 0.0, "val_test_quantile": 0.8})
    with pytest.raises(SplitError):
        compute_split_boundaries(dt_series, {"train_val_quantile": 0.6, "val_test_quantile": 1.0})


def test_compute_split_boundaries_rejects_zero_span():
    constant_dt = pd.Series([100] * 10)
    with pytest.raises(SplitError, match="zero or negative span"):
        compute_split_boundaries(constant_dt, SPLIT_CONFIG)


def test_assign_primary_split_produces_only_expected_labels(dt_series):
    boundaries = compute_split_boundaries(dt_series, SPLIT_CONFIG)
    labels = assign_primary_split(dt_series, boundaries)
    assert set(labels.unique()) <= {"train", "val", "test"}
    assert len(labels) == len(dt_series)


def test_assign_primary_split_is_temporally_ordered(dt_series):
    boundaries = compute_split_boundaries(dt_series, SPLIT_CONFIG)
    labels = assign_primary_split(dt_series, boundaries)
    train_max = dt_series[labels == "train"].max()
    val_min = dt_series[labels == "val"].min()
    val_max = dt_series[labels == "val"].max()
    test_min = dt_series[labels == "test"].min()
    assert train_max <= val_min
    assert val_max <= test_min


def test_assign_primary_split_no_random_kfold_shuffling(dt_series):
    """The split must be a deterministic function of DT, not a shuffle.

    Re-running assignment on the same input must be byte-identical -- a
    random k-fold implementation would not guarantee this without a fixed
    seed, and even then would not preserve temporal contiguity, which this
    test checks directly.
    """
    boundaries = compute_split_boundaries(dt_series, SPLIT_CONFIG)
    labels_a = assign_primary_split(dt_series, boundaries)
    labels_b = assign_primary_split(dt_series, boundaries)
    pd.testing.assert_series_equal(labels_a, labels_b)
    # Temporal contiguity: once a row belongs to 'val' or 'test', no LATER
    # DT value can revert to an earlier-stage label.
    sorted_labels = labels_a.loc[dt_series.sort_values().index].tolist()
    stage = {"train": 0, "val": 1, "test": 2}
    stage_sequence = [stage[label] for label in sorted_labels]
    assert stage_sequence == sorted(stage_sequence)


def test_duplicate_dt_never_straddles_boundary():
    """Regression test: a boundary landing exactly on a heavily-duplicated
    DT value must not split that value's rows across two partitions.

    docs/eda.md Sec.2 measured 17,191 adjacent duplicate TransactionDT
    values in the real dataset; this constructs a worst-case synthetic
    analogue where the computed boundary is expected to fall inside a
    large tied block.
    """
    # 50 rows at DT=1000 (a big tied block straddling where a 0.5 quantile
    # boundary would naturally land), flanked by distinct values.
    dt = pd.Series([0] * 10 + [1000] * 50 + [2000] * 10)
    boundaries = compute_split_boundaries(
        dt, {"train_val_quantile": 0.4, "val_test_quantile": 0.6, "cv_n_folds": 0}
    )
    labels = assign_primary_split(dt, boundaries)
    tied_block_labels = labels[dt == 1000]
    assert tied_block_labels.nunique() == 1, (
        f"DT=1000 block was split across labels: {tied_block_labels.unique()}"
    )


def test_assign_cv_eval_fold_never_assigns_beyond_last_boundary(dt_series):
    boundaries = compute_split_boundaries(dt_series, SPLIT_CONFIG)
    fold = assign_cv_eval_fold(dt_series, boundaries)
    beyond = dt_series > boundaries.cv_boundaries[-1]
    assert fold[beyond].isna().all()


def test_assign_cv_eval_fold_values_are_valid_fold_indices(dt_series):
    boundaries = compute_split_boundaries(dt_series, SPLIT_CONFIG)
    fold = assign_cv_eval_fold(dt_series, boundaries)
    observed = set(fold.dropna().unique())
    assert observed <= set(range(len(boundaries.cv_boundaries) - 1))


def test_iter_expanding_window_folds_train_strictly_precedes_eval(dt_series):
    boundaries = compute_split_boundaries(dt_series, SPLIT_CONFIG)
    folds = iter_expanding_window_folds(dt_series, boundaries)
    assert len(folds) == SPLIT_CONFIG["cv_n_folds"]
    for fold_index, train_mask, eval_mask in folds:
        assert not (train_mask & eval_mask).any(), f"fold {fold_index}: train/eval overlap"
        if train_mask.any() and eval_mask.any():
            assert dt_series[train_mask].max() <= dt_series[eval_mask].min()


def test_iter_expanding_window_folds_train_window_strictly_expands(dt_series):
    boundaries = compute_split_boundaries(dt_series, SPLIT_CONFIG)
    folds = iter_expanding_window_folds(dt_series, boundaries)
    train_sizes = [int(train_mask.sum()) for _, train_mask, _ in folds]
    assert train_sizes == sorted(train_sizes)
    assert len(set(train_sizes)) == len(train_sizes), f"train sizes did not strictly grow: {train_sizes}"


def test_iter_expanding_window_folds_never_reach_val_test_boundary(dt_series):
    boundaries = compute_split_boundaries(dt_series, SPLIT_CONFIG)
    folds = iter_expanding_window_folds(dt_series, boundaries)
    for _, train_mask, eval_mask in folds:
        combined = train_mask | eval_mask
        assert dt_series[combined].max() <= boundaries.val_test_boundary


def test_build_split_assignments_row_count_conserved(dt_series):
    boundaries = compute_split_boundaries(dt_series, SPLIT_CONFIG)
    transaction_id = pd.Series(range(len(dt_series)))
    assignments = build_split_assignments(transaction_id, dt_series, boundaries)
    assert len(assignments) == len(dt_series)
    assert set(assignments.columns) == {"TransactionID", "TransactionDT", "split", "cv_eval_fold"}


def test_build_split_assignments_is_deterministic(dt_series):
    boundaries = compute_split_boundaries(dt_series, SPLIT_CONFIG)
    transaction_id = pd.Series(range(len(dt_series)))
    a = build_split_assignments(transaction_id, dt_series, boundaries)
    b = build_split_assignments(transaction_id, dt_series, boundaries)
    pd.testing.assert_frame_equal(a, b)
