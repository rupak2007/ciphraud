import numpy as np
import pandas as pd
import pytest

from src.data.preprocess import (
    DEFAULT_EXCLUDED_COLUMNS,
    FittedPreprocessor,
    PreprocessError,
    TrainScopedFrequencyEncoder,
    TrainScopedMedianImputer,
    build_feature_matrix,
    fit_preprocessor,
    infer_column_roles,
    join_transaction_identity,
)


# ---------------------------------------------------------------------------
# join_transaction_identity
# ---------------------------------------------------------------------------


@pytest.fixture
def transaction_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "TransactionID": [1, 2, 3, 4],
            "TransactionDT": [0, 100, 200, 300],
            "isFraud": [0, 1, 0, 0],
            "TransactionAmt": [10.0, 20.0, 30.0, 40.0],
            "ProductCD": ["W", "C", "W", "R"],
        }
    )


@pytest.fixture
def identity_df() -> pd.DataFrame:
    # Only TransactionID 1 and 3 have identity rows.
    return pd.DataFrame({"TransactionID": [1, 3], "DeviceType": ["mobile", "desktop"]})


def test_join_adds_has_identity_column(transaction_df, identity_df):
    joined = join_transaction_identity(transaction_df, identity_df)
    assert "has_identity" in joined.columns
    assert joined.set_index("TransactionID")["has_identity"].to_dict() == {1: 1, 2: 0, 3: 1, 4: 0}


def test_join_preserves_row_count(transaction_df, identity_df):
    joined = join_transaction_identity(transaction_df, identity_df)
    assert len(joined) == len(transaction_df)


def test_join_fills_nan_for_unmatched_rows(transaction_df, identity_df):
    joined = join_transaction_identity(transaction_df, identity_df)
    unmatched = joined.set_index("TransactionID").loc[2]
    assert pd.isna(unmatched["DeviceType"])


def test_join_matched_rows_get_correct_identity_data(transaction_df, identity_df):
    joined = join_transaction_identity(transaction_df, identity_df)
    matched = joined.set_index("TransactionID").loc[1]
    assert matched["DeviceType"] == "mobile"


# ---------------------------------------------------------------------------
# infer_column_roles
# ---------------------------------------------------------------------------


def test_infer_column_roles_splits_numeric_and_categorical():
    df = pd.DataFrame(
        {
            "TransactionID": [1, 2],
            "TransactionDT": [0, 100],
            "isFraud": [0, 1],
            "amt": [1.5, 2.5],
            "cat": pd.Categorical(["a", "b"]),
        }
    )
    roles = infer_column_roles(df)
    assert roles["numeric"] == ["amt"]
    assert roles["categorical"] == ["cat"]


def test_infer_column_roles_excludes_configured_columns():
    df = pd.DataFrame({"TransactionID": [1], "keep_me": [1.0], "drop_me": [2.0]})
    roles = infer_column_roles(df, exclude_columns=("TransactionID", "drop_me"))
    assert roles["numeric"] == ["keep_me"]


# ---------------------------------------------------------------------------
# TrainScopedFrequencyEncoder
# ---------------------------------------------------------------------------


def test_frequency_encoder_fit_records_correct_frequencies():
    train_df = pd.DataFrame({"cat": ["a", "a", "a", "b"]})
    encoder = TrainScopedFrequencyEncoder(columns=["cat"]).fit(train_df)
    assert encoder.frequency_tables_["cat"]["a"] == pytest.approx(0.75)
    assert encoder.frequency_tables_["cat"]["b"] == pytest.approx(0.25)


def test_frequency_encoder_fit_records_fit_index():
    train_df = pd.DataFrame({"cat": ["a", "b"]}, index=[10, 11])
    encoder = TrainScopedFrequencyEncoder(columns=["cat"]).fit(train_df)
    assert list(encoder.fit_index_) == [10, 11]


def test_frequency_encoder_transform_before_fit_raises():
    encoder = TrainScopedFrequencyEncoder(columns=["cat"])
    with pytest.raises(PreprocessError, match="before fit"):
        encoder.transform(pd.DataFrame({"cat": ["a"]}))


def test_frequency_encoder_unseen_category_maps_to_zero():
    train_df = pd.DataFrame({"cat": ["a", "a", "b"]})
    encoder = TrainScopedFrequencyEncoder(columns=["cat"]).fit(train_df)
    transformed = encoder.transform(pd.DataFrame({"cat": ["a", "never_seen"]}))
    assert transformed["cat_freq"].iloc[0] == pytest.approx(2 / 3)
    assert transformed["cat_freq"].iloc[1] == 0.0


def test_frequency_encoder_null_value_maps_to_zero():
    train_df = pd.DataFrame({"cat": ["a", "b"]})
    encoder = TrainScopedFrequencyEncoder(columns=["cat"]).fit(train_df)
    transformed = encoder.transform(pd.DataFrame({"cat": [None]}))
    assert transformed["cat_freq"].iloc[0] == 0.0


def test_frequency_encoder_does_not_use_val_data_even_if_present_in_transform_call():
    """The encoder must never re-derive frequencies from whatever it's
    asked to transform -- only from what it was fit on."""
    train_df = pd.DataFrame({"cat": ["a", "a", "a", "a"]})  # freq(a) = 1.0 in train
    encoder = TrainScopedFrequencyEncoder(columns=["cat"]).fit(train_df)
    # val has a totally different distribution; if the encoder recomputed
    # frequencies here, freq(a) would come out as 0.5, not 1.0.
    val_df = pd.DataFrame({"cat": ["a", "b"]})
    transformed = encoder.transform(val_df)
    assert transformed["cat_freq"].iloc[0] == pytest.approx(1.0)


def test_frequency_encoder_handles_category_dtype_column():
    """Regression test for a real bug found against the actual IEEE-CIS data:
    .map() on a `category`-dtype Series returns a Categorical container
    whose categories are the mapped frequency values, and .fillna(0.0)
    then raises `TypeError: Cannot setitem on a Categorical with a new
    category` unless 0.0 already happens to be one of them. Every other
    encoder test in this file uses plain object/str columns, which never
    exercised this path -- this is why the bug wasn't caught until the
    real-data pipeline run (see docs/pipeline.md Sec.4 deviation 7)."""
    train_df = pd.DataFrame({"cat": pd.Categorical(["a", "a", "b"])})
    encoder = TrainScopedFrequencyEncoder(columns=["cat"]).fit(train_df)

    val_df = pd.DataFrame({"cat": pd.Categorical(["a", "b", "never_seen_in_train"])})
    transformed = encoder.transform(val_df)

    assert transformed["cat_freq"].iloc[0] == pytest.approx(2 / 3)
    assert transformed["cat_freq"].iloc[1] == pytest.approx(1 / 3)
    assert transformed["cat_freq"].iloc[2] == 0.0


def test_frequency_encoder_oov_rate_measures_unseen_fraction():
    train_df = pd.DataFrame({"cat": ["a", "b"]})
    encoder = TrainScopedFrequencyEncoder(columns=["cat"]).fit(train_df)
    val_df = pd.DataFrame({"cat": ["a", "c", "d"]})  # 2/3 unseen
    rates = encoder.oov_rate(val_df)
    assert rates["cat"] == pytest.approx(2 / 3)


# ---------------------------------------------------------------------------
# TrainScopedMedianImputer
# ---------------------------------------------------------------------------


def test_median_imputer_fit_records_correct_median():
    train_df = pd.DataFrame({"amt": [1.0, 2.0, 3.0, 100.0]})
    imputer = TrainScopedMedianImputer(columns=["amt"]).fit(train_df)
    assert imputer.medians_["amt"] == pytest.approx(2.5)


def test_median_imputer_transform_fills_nan_with_train_median():
    train_df = pd.DataFrame({"amt": [1.0, 2.0, 3.0]})
    imputer = TrainScopedMedianImputer(columns=["amt"]).fit(train_df)
    transformed = imputer.transform(pd.DataFrame({"amt": [np.nan]}))
    assert transformed["amt"].iloc[0] == pytest.approx(2.0)


def test_median_imputer_adds_was_missing_indicator():
    train_df = pd.DataFrame({"amt": [1.0, 2.0]})
    imputer = TrainScopedMedianImputer(columns=["amt"]).fit(train_df)
    transformed = imputer.transform(pd.DataFrame({"amt": [np.nan, 5.0]}))
    assert transformed["amt_was_missing"].tolist() == [1, 0]


def test_median_imputer_does_not_use_val_data_for_fill_value():
    """The critical leakage-safety property: the fill value used on val
    must come from train's median, not be re-derived from val."""
    train_df = pd.DataFrame({"amt": [10.0, 10.0, 10.0]})  # median = 10
    imputer = TrainScopedMedianImputer(columns=["amt"]).fit(train_df)
    val_df = pd.DataFrame({"amt": [np.nan, 999.0, 999.0]})  # val median would be 999 if recomputed
    transformed = imputer.transform(val_df)
    assert transformed["amt"].iloc[0] == pytest.approx(10.0)


def test_median_imputer_transform_before_fit_raises():
    imputer = TrainScopedMedianImputer(columns=["amt"])
    with pytest.raises(PreprocessError, match="before fit"):
        imputer.transform(pd.DataFrame({"amt": [1.0]}))


def test_median_imputer_handles_all_null_fit_column_without_nan_leakage():
    train_df = pd.DataFrame({"amt": [np.nan, np.nan]})
    imputer = TrainScopedMedianImputer(columns=["amt"]).fit(train_df)
    assert imputer.medians_["amt"] == 0.0  # documented fallback, not NaN propagation


# ---------------------------------------------------------------------------
# fit_preprocessor / build_feature_matrix -- full integration
# ---------------------------------------------------------------------------


@pytest.fixture
def joined_df(transaction_df, identity_df) -> pd.DataFrame:
    return join_transaction_identity(transaction_df, identity_df)


def test_fit_preprocessor_never_includes_banned_columns_in_output(joined_df):
    preprocessor = fit_preprocessor(joined_df)
    X, y = build_feature_matrix(joined_df, preprocessor)
    for banned in DEFAULT_EXCLUDED_COLUMNS:
        assert banned not in X.columns, f"{banned} leaked into the feature matrix"


def test_build_feature_matrix_indexes_by_transaction_id(joined_df):
    preprocessor = fit_preprocessor(joined_df)
    X, y = build_feature_matrix(joined_df, preprocessor)
    assert list(X.index) == list(joined_df["TransactionID"])
    assert list(y.index) == list(joined_df["TransactionID"])


def test_build_feature_matrix_y_matches_isfraud(joined_df):
    preprocessor = fit_preprocessor(joined_df)
    X, y = build_feature_matrix(joined_df, preprocessor)
    assert y.tolist() == joined_df["isFraud"].tolist()


def test_fit_preprocessor_passthrough_column_unchanged(joined_df):
    preprocessor = fit_preprocessor(joined_df, passthrough_columns=("has_identity",))
    X, y = build_feature_matrix(joined_df, preprocessor)
    assert "has_identity" in X.columns
    assert X["has_identity"].tolist() == joined_df.set_index("TransactionID")["has_identity"].tolist()


def test_fit_preprocessor_encoder_and_imputer_share_train_fit_index(joined_df):
    preprocessor = fit_preprocessor(joined_df)
    assert list(preprocessor.frequency_encoder.fit_index_) == list(preprocessor.median_imputer.fit_index_)
    assert list(preprocessor.frequency_encoder.fit_index_) == list(joined_df.index)


def test_fit_preprocessor_train_val_leakage_scenario_is_measurable():
    """Simulates the realistic Phase-2 usage pattern: fit on train only,
    transform val with a distribution shift, and confirm the val-only
    statistic never contaminates the transform (mirrors
    tests/test_leakage.py but through the full fit_preprocessor/
    build_feature_matrix path)."""
    train_df = pd.DataFrame(
        {
            "TransactionID": [1, 2, 3],
            "TransactionDT": [0, 1, 2],
            "isFraud": [0, 0, 1],
            "amt": [10.0, 10.0, 10.0],
            "cat": ["x", "x", "x"],
        }
    ).set_index("TransactionID", drop=False)
    val_df = pd.DataFrame(
        {
            "TransactionID": [4, 5],
            "TransactionDT": [10, 11],
            "isFraud": [0, 1],
            "amt": [np.nan, 500.0],
            "cat": ["y", "y"],
        }
    ).set_index("TransactionID", drop=False)

    preprocessor = fit_preprocessor(train_df, passthrough_columns=())
    X_val, _ = build_feature_matrix(val_df, preprocessor)

    # amt fill value must be train's median (10.0), not influenced by val's 500.0
    assert X_val["amt"].iloc[0] == pytest.approx(10.0)
    # 'y' was never seen in train -> frequency 0.0, not derived from val
    assert X_val["cat_freq"].iloc[0] == 0.0
