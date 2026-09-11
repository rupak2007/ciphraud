import pandas as pd

from src.data.split import compute_split_boundaries
from src.train.cv import build_positional_cv_folds

SPLIT_CONFIG = {"train_val_quantile": 0.6, "val_test_quantile": 0.8, "cv_n_folds": 3}


def test_build_positional_cv_folds_positions_are_valid_row_indices():
    dt = pd.Series(range(0, 10000, 100))  # 100 rows, positions 0..99
    boundaries = compute_split_boundaries(dt, SPLIT_CONFIG)
    folds = build_positional_cv_folds(dt, boundaries)

    assert len(folds) == SPLIT_CONFIG["cv_n_folds"]
    for fold_index, train_pos, eval_pos in folds:
        assert train_pos.min() >= 0
        assert train_pos.max() < len(dt)
        assert eval_pos.min() >= 0
        assert eval_pos.max() < len(dt)


def test_build_positional_cv_folds_positions_match_boolean_masks():
    dt = pd.Series(range(0, 10000, 100))
    boundaries = compute_split_boundaries(dt, SPLIT_CONFIG)
    folds = build_positional_cv_folds(dt, boundaries)

    from src.data.split import iter_expanding_window_folds

    mask_folds = iter_expanding_window_folds(dt, boundaries)
    for (idx_a, train_pos, eval_pos), (idx_b, train_mask, eval_mask) in zip(folds, mask_folds):
        assert idx_a == idx_b
        assert list(train_pos) == list(train_mask.to_numpy().nonzero()[0])
        assert list(eval_pos) == list(eval_mask.to_numpy().nonzero()[0])


def test_build_positional_cv_folds_train_and_eval_disjoint():
    dt = pd.Series(range(0, 10000, 100))
    boundaries = compute_split_boundaries(dt, SPLIT_CONFIG)
    folds = build_positional_cv_folds(dt, boundaries)
    for _, train_pos, eval_pos in folds:
        assert set(train_pos).isdisjoint(set(eval_pos))


def test_build_positional_cv_folds_works_with_non_default_index():
    """train_dt in real usage is indexed by TransactionID, not a RangeIndex
    -- positions must still be correct 0-based row positions."""
    dt = pd.Series(range(0, 10000, 100), index=range(5000, 5100))
    boundaries = compute_split_boundaries(dt, SPLIT_CONFIG)
    folds = build_positional_cv_folds(dt, boundaries)
    for _, train_pos, eval_pos in folds:
        assert train_pos.max() < 100  # positions, not the 5000+ index labels
        assert eval_pos.max() < 100
