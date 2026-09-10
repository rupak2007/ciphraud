from pathlib import Path

import pandas as pd
import pytest

from src.data.eda import (
    _ExpandingCategoricalTracker,
    _fold_edges,
    compute_candidate_boundaries,
    compute_class_balance_by_day,
    compute_positives_per_fold,
    analyze_transactiondt_semantics,
    run_full_pass,
)
from src.data.load import infer_dtype_map

LOAD_CONFIG = {"chunksize": 3, "downcast_floats": True, "categorical_as_category": True}


# ---------------------------------------------------------------------------
# TransactionDT semantics
# ---------------------------------------------------------------------------


def _core_frame(dt_values, id_values=None, fraud_values=None):
    n = len(dt_values)
    return pd.DataFrame(
        {
            "TransactionID": id_values if id_values is not None else list(range(1, n + 1)),
            "TransactionDT": dt_values,
            "isFraud": fraud_values if fraud_values is not None else [0] * n,
            "TransactionAmt": [10.0] * n,
        }
    )


def test_transactiondt_semantics_min_max_span():
    core = _core_frame([0, 3600, 7200, 10800])
    result = analyze_transactiondt_semantics(core)
    assert result["dt_min"] == 0
    assert result["dt_max"] == 10800
    assert result["span_seconds"] == 10800
    assert result["epoch_reading_plausible"] is False  # 0 is nowhere near a Unix epoch


def test_transactiondt_semantics_hour_of_day_exact_counts():
    # 0s -> hour 0; 3600s -> hour 1; 7200s -> hour 2; 90000s -> hour 1 (25h -> wraps mod 24)
    core = _core_frame([0, 3600, 7200, 90000])
    result = analyze_transactiondt_semantics(core)
    assert result["hour_of_day_counts"] == {0: 1, 1: 2, 2: 1}


def test_transactiondt_semantics_monotonic_true_when_sorted():
    core = _core_frame([0, 100, 200, 300])
    assert analyze_transactiondt_semantics(core)["is_monotonic_non_decreasing"] is True


def test_transactiondt_semantics_monotonic_false_when_not_sorted():
    core = _core_frame([0, 300, 100, 200])
    assert analyze_transactiondt_semantics(core)["is_monotonic_non_decreasing"] is False


def test_transactiondt_semantics_duplicate_dt_count():
    core = _core_frame([0, 100, 100, 100, 200])
    # diffs: 100,0,0,100 -> two zero-diffs -> 2 adjacent duplicates
    assert analyze_transactiondt_semantics(core)["adjacent_duplicate_dt_count"] == 2


def test_transactiondt_semantics_spearman_perfect_when_id_tracks_dt():
    core = _core_frame(dt_values=[0, 100, 200, 300], id_values=[10, 20, 30, 40])
    result = analyze_transactiondt_semantics(core)
    assert result["spearman_transactionid_vs_transactiondt"] == pytest.approx(1.0)


def test_transactiondt_semantics_spearman_low_when_uncorrelated():
    core = _core_frame(dt_values=[0, 100, 200, 300], id_values=[40, 10, 30, 20])
    result = analyze_transactiondt_semantics(core)
    assert result["spearman_transactionid_vs_transactiondt"] < 1.0


# ---------------------------------------------------------------------------
# Candidate boundaries and per-fold positives
# ---------------------------------------------------------------------------


def test_compute_candidate_boundaries_time_range_quantile():
    core = _core_frame([0, 25, 50, 75, 100])
    boundaries = compute_candidate_boundaries(core, [0.5])
    assert boundaries == [{"quantile": 0.5, "transactiondt_boundary": 50}]


def test_compute_candidate_boundaries_multiple_quantiles():
    core = _core_frame([0, 100])
    boundaries = compute_candidate_boundaries(core, [0.25, 0.75])
    assert boundaries[0]["transactiondt_boundary"] == 25
    assert boundaries[1]["transactiondt_boundary"] == 75


def test_compute_positives_per_fold_splits_correctly_and_sums_to_total():
    core = _core_frame(
        dt_values=[0, 10, 20, 30, 40, 50, 60, 70, 80, 90],
        fraud_values=[1, 0, 1, 0, 1, 0, 1, 0, 1, 0],
    )
    boundaries = compute_candidate_boundaries(core, [0.5])  # boundary at dt=45
    folds = compute_positives_per_fold(core, boundaries)
    assert len(folds) == 2
    assert sum(f["n_rows"] for f in folds) == 10
    assert sum(f["n_positive"] for f in folds) == int(core["isFraud"].sum())
    # fold 0: dt <= 45 -> rows at 0,10,20,30,40 (5 rows); fold 1: dt in (45,90] -> 5 rows
    assert folds[0]["n_rows"] == 5
    assert folds[1]["n_rows"] == 5


def test_compute_class_balance_by_day():
    core = _core_frame(
        dt_values=[0, 3600, 86400, 86400 + 3600, 86400 + 7200],
        fraud_values=[1, 0, 1, 1, 0],
    )
    result = compute_class_balance_by_day(core, time_bucket_seconds=86400)
    day0 = result[result["day_index"] == 0].iloc[0]
    day1 = result[result["day_index"] == 1].iloc[0]
    assert day0["n_rows"] == 2
    assert day0["n_positive"] == 1
    assert day1["n_rows"] == 3
    assert day1["n_positive"] == 2


# ---------------------------------------------------------------------------
# Expanding-window categorical tracker (OOV + cardinality)
# ---------------------------------------------------------------------------


def test_expanding_tracker_fold0_never_flagged_oov():
    tracker = _ExpandingCategoricalTracker(["cat"], n_folds=2)
    fold0 = pd.DataFrame({"cat": ["a", "b", "a"]})
    tracker.observe_fold(0, fold0)
    oov = tracker.oov_table()
    # fold 0 produces no OOV rows at all (base fold)
    assert (oov["fold_index"] == 0).sum() == 0


def test_expanding_tracker_flags_unseen_value_in_later_fold():
    tracker = _ExpandingCategoricalTracker(["cat"], n_folds=2)
    tracker.observe_fold(0, pd.DataFrame({"cat": ["a", "b"]}))
    tracker.observe_fold(1, pd.DataFrame({"cat": ["a", "c", "c"]}))
    oov = tracker.oov_table()
    row = oov[(oov["column"] == "cat") & (oov["fold_index"] == 1)].iloc[0]
    assert row["n_values"] == 3
    assert row["n_oov"] == 2  # the two "c"s are unseen; "a" is not
    assert row["oov_rate"] == pytest.approx(2 / 3)


def test_expanding_tracker_updates_seen_set_after_each_fold():
    tracker = _ExpandingCategoricalTracker(["cat"], n_folds=3)
    tracker.observe_fold(0, pd.DataFrame({"cat": ["a"]}))
    tracker.observe_fold(1, pd.DataFrame({"cat": ["b"]}))  # "b" unseen -> oov, then merged into seen
    tracker.observe_fold(2, pd.DataFrame({"cat": ["b"]}))  # "b" now seen from fold 1 -> not oov
    oov = tracker.oov_table()
    fold2_row = oov[(oov["column"] == "cat") & (oov["fold_index"] == 2)].iloc[0]
    assert fold2_row["n_oov"] == 0


def test_expanding_tracker_cardinality_counts_distinct_values():
    tracker = _ExpandingCategoricalTracker(["cat"], n_folds=1)
    tracker.observe_fold(0, pd.DataFrame({"cat": ["a", "a", "b", "c"]}))
    cardinality = tracker.cardinality_table()
    row = cardinality[cardinality["column"] == "cat"].iloc[0]
    assert row["n_distinct"] == 3


def test_expanding_tracker_ignores_nulls():
    tracker = _ExpandingCategoricalTracker(["cat"], n_folds=1)
    tracker.observe_fold(0, pd.DataFrame({"cat": ["a", None, "b"]}))
    cardinality = tracker.cardinality_table()
    row = cardinality[cardinality["column"] == "cat"].iloc[0]
    assert row["n_distinct"] == 2


# ---------------------------------------------------------------------------
# _fold_edges
# ---------------------------------------------------------------------------


def test_fold_edges_bracket_full_range():
    core = _core_frame([10, 20, 30, 40])
    boundaries = [{"quantile": 0.5, "transactiondt_boundary": 25}]
    edges = _fold_edges(core, boundaries)
    assert edges == [9, 25, 41]


# ---------------------------------------------------------------------------
# run_full_pass: V-column null-mask grouping (the trickiest piece), via a
# real small synthetic CSV on disk (exercises the actual chunked-read path)
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_transaction_csv(tmp_path: Path) -> Path:
    # V1 and V2 share an IDENTICAL null mask; V3 has a different one.
    df = pd.DataFrame(
        {
            "TransactionID": [1, 2, 3, 4, 5, 6],
            "TransactionDT": [0, 10, 20, 30, 40, 50],
            "isFraud": [0, 1, 0, 0, 1, 0],
            "TransactionAmt": [10.0, 20.0, 5.0, 100.0, 7.0, 12.0],
            "ProductCD": ["W", "C", "W", "R", "W", "C"],
            "V1": [1.0, None, 1.0, None, 1.0, None],
            "V2": [2.0, None, 2.0, None, 2.0, None],
            "V3": [None, None, 3.0, 3.0, 3.0, 3.0],
        }
    )
    path = tmp_path / "synthetic_transaction.csv"
    df.to_csv(path, index=False)
    return path


def test_run_full_pass_groups_identical_null_masks(synthetic_transaction_csv):
    dtype_map = infer_dtype_map(synthetic_transaction_csv, LOAD_CONFIG)
    fold_edges = [-1, 51]  # single fold covering everything
    result = run_full_pass(
        synthetic_transaction_csv,
        LOAD_CONFIG,
        dtype_map,
        fold_edges,
        time_bucket_seconds=86400,
        total_rows=6,
    )
    blocks = result["v_null_mask_blocks"]
    v1_block = blocks[blocks["columns"].str.contains("V1")].iloc[0]
    assert "V2" in v1_block["columns"]
    assert "V3" not in v1_block["columns"]
    assert v1_block["n_columns"] == 2

    v3_block = blocks[blocks["columns"] == "V3"]
    assert len(v3_block) == 1


def test_run_full_pass_missingness_global_counts(synthetic_transaction_csv):
    dtype_map = infer_dtype_map(synthetic_transaction_csv, LOAD_CONFIG)
    fold_edges = [-1, 51]
    result = run_full_pass(
        synthetic_transaction_csv, LOAD_CONFIG, dtype_map, fold_edges, 86400, total_rows=6
    )
    missingness = result["missingness_global"].set_index("column")
    assert missingness.loc["V1", "null_count"] == 3
    assert missingness.loc["V1", "non_null_count"] == 3
    assert missingness.loc["TransactionID", "null_count"] == 0


def test_run_full_pass_cardinality_for_categorical_column(synthetic_transaction_csv):
    dtype_map = infer_dtype_map(synthetic_transaction_csv, LOAD_CONFIG)
    fold_edges = [-1, 51]
    result = run_full_pass(
        synthetic_transaction_csv, LOAD_CONFIG, dtype_map, fold_edges, 86400, total_rows=6
    )
    cardinality = result["cardinality"].set_index("column")
    assert cardinality.loc["ProductCD", "n_distinct"] == 3  # W, C, R
