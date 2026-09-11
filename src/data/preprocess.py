"""Leakage-safe preprocessing: transaction<->identity join, missing-value
imputation, and categorical frequency encoding.

Every statistic used at transform time (a per-column median, a per-category
frequency) is computed by `.fit()` on exactly one DataFrame, and that
object records the exact row index it was fit on (`fit_index_`) so
`src/data/leakage.py::check_fitted_statistic_uses_only_allowed_index` can
verify -- not just trust -- that no validation/test-period row ever
contributed to a training-time statistic (CLAUDE.md Sec.6; instructions.md
Data-Leakage Rules).

Column roles (numeric vs categorical) are inferred empirically from the
actual DataFrame dtypes (`infer_column_roles`), not hardcoded from a
guessed IEEE-CIS column list -- consistent with how `src/data/load.py`
and `src/data/schema.py` already do this (CLAUDE.md Sec.17).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

# Columns that must never appear in a model-ready feature matrix.
#
# TransactionID: banned per explicit project mandate. docs/eda.md Sec.2
# measured Spearman(TransactionID, TransactionDT) ~= 0.99999999999974 --
# it is a near-perfect proxy for row/time order.
#
# TransactionDT: excluded from the DEFAULT feature set as a documented
# judgment call, not a literal project mandate -- for the same underlying
# reason as the TransactionID ban (a raw absolute time offset lets a model
# implicitly key off "which period is this row from", which cannot
# generalize the way a model evaluated on strictly-later data needs it to).
# This is configurable via configs/phase2/pipeline.yaml's
# preprocess.exclude_columns if a future phase wants to derive genuine
# cyclical time features (hour-of-day, day-of-week) from it instead --
# that is feature engineering and belongs to Phase 4, not Phase 2.
#
# isFraud: the label, never a feature.
DEFAULT_EXCLUDED_COLUMNS: tuple[str, ...] = ("TransactionID", "TransactionDT", "isFraud")


class PreprocessError(Exception):
    """Raised when preprocessing is misused (e.g. transform before fit)."""


def join_transaction_identity(
    transaction_df: pd.DataFrame, identity_df: pd.DataFrame
) -> pd.DataFrame:
    """Left join on TransactionID; adds `has_identity` before merging.

    A pure structural join carries no leakage risk on its own (no
    statistic is fit here) -- see docs/eda.md Sec.7 for why `has_identity`
    is a recommended feature: fraud rows are >2x as likely to have
    identity data (54.8% vs 23.3%), and that fact is genuinely available
    at prediction time (unlike a label-derived statistic).
    """
    identity_ids = set(identity_df["TransactionID"])
    has_identity = transaction_df["TransactionID"].isin(identity_ids).astype("int8")
    merged = transaction_df.merge(identity_df, on="TransactionID", how="left")
    # merged.insert(...) on the ~600-column merge result triggers pandas'
    # "highly fragmented" performance warning even for a single insert;
    # .copy() after inserting defragments it (same fix as
    # src/data/eda.py's identical issue in Phase 1).
    merged.insert(
        loc=merged.columns.get_loc("TransactionID") + 1,
        column="has_identity",
        value=has_identity.to_numpy(),
    )
    return merged.copy()


def infer_column_roles(
    df: pd.DataFrame, exclude_columns: tuple[str, ...] = DEFAULT_EXCLUDED_COLUMNS
) -> dict[str, list[str]]:
    """Empirically classify columns as 'numeric' or 'categorical' by actual dtype.

    Deliberately dtype-driven, not a hardcoded IEEE-CIS column-name table
    (see module docstring). `has_identity` is int8 and would classify as
    numeric here, which is correct: it is already a clean 0/1 indicator,
    not something needing imputation or frequency encoding.
    """
    numeric, categorical = [], []
    for col in df.columns:
        if col in exclude_columns:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            numeric.append(col)
        else:
            categorical.append(col)
    return {"numeric": numeric, "categorical": categorical}


@dataclass
class TrainScopedFrequencyEncoder:
    """Per-category frequency, fit on exactly one DataFrame's rows.

    Unseen categories at transform time (including all-NaN rows, e.g. a
    transaction with no matching identity row) map to 0.0 -- a legitimate
    numeric encoding of "never observed in training", not a special case.
    docs/eda.md Sec.6 measured 0.0% OOV at every candidate boundary for the
    14 profiled low-cardinality categoricals, so in practice this fallback
    should rarely trigger for those columns; `oov_rate()` below measures it
    directly for every run rather than assuming that finding still holds
    for whatever columns are actually passed in (e.g. card1-5, which
    docs/eda.md Sec.6 explicitly flagged as NOT profiled in Phase 1).
    """

    columns: list[str]
    frequency_tables_: dict[str, dict[Any, float]] = field(default_factory=dict, init=False)
    fit_index_: pd.Index | None = field(default=None, init=False)

    def fit(self, df: pd.DataFrame) -> TrainScopedFrequencyEncoder:
        self.fit_index_ = df.index
        n = len(df)
        for col in self.columns:
            counts = df[col].value_counts(dropna=True)
            self.frequency_tables_[col] = (counts / n).to_dict()
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.fit_index_ is None:
            raise PreprocessError("TrainScopedFrequencyEncoder.transform called before fit")
        # Built in a dict then constructed once -- see the identical note in
        # TrainScopedMedianImputer.transform (avoids the pandas
        # "highly fragmented" warning at real-dataset column counts).
        columns: dict[str, pd.Series] = {}
        for col in self.columns:
            table = self.frequency_tables_[col]
            # .map() on a `category`-dtype column returns a Categorical
            # whose categories are the mapped frequency values -- .fillna(0.0)
            # then fails with "Cannot setitem on a Categorical with a new
            # category" unless 0.0 already happens to be one of them.
            # Casting to float64 first drops the Categorical container
            # before fillna ever runs.
            mapped = df[col].map(table).astype("float64")
            columns[f"{col}_freq"] = mapped.fillna(0.0).astype("float32")
        return pd.DataFrame(columns, index=df.index)

    def oov_rate(self, df: pd.DataFrame) -> dict[str, float]:
        """Diagnostic: fraction of non-null values in `df` unseen during fit."""
        rates: dict[str, float] = {}
        for col in self.columns:
            seen = set(self.frequency_tables_[col].keys())
            values = df[col].dropna()
            rates[col] = float((~values.isin(seen)).mean()) if len(values) else 0.0
        return rates


@dataclass
class TrainScopedMedianImputer:
    """Per-column median fill, fit on exactly one DataFrame's rows.

    Also emits a `{col}_was_missing` indicator at transform time, computed
    row-wise from whatever DataFrame is passed to `.transform()` -- this
    needs no fitted statistic and is therefore leakage-safe by
    construction on any partition, including val/test. docs/eda.md Sec.3
    found a specific column block (M7-M9, V1-V11, D11) whose missingness
    rate drifts sharply over time; a global-mean/median fill would be
    unsafe for it (it would implicitly encode "this row is from a later,
    more-complete period" into the fill value), which is exactly why this
    class only ever fits on the caller-supplied DataFrame and never on
    anything the caller doesn't explicitly scope to train.
    """

    columns: list[str]
    medians_: dict[str, float] = field(default_factory=dict, init=False)
    fit_index_: pd.Index | None = field(default=None, init=False)

    def fit(self, df: pd.DataFrame) -> TrainScopedMedianImputer:
        self.fit_index_ = df.index
        for col in self.columns:
            median = df[col].median()
            # A column could be entirely null within a small/synthetic fit
            # window; fall back to 0.0 rather than propagate NaN into every
            # transformed value, and this is rare enough in the real
            # dataset (docs/eda.md Sec.3: no column is 100% missing) to be
            # a defensive fallback, not the expected path.
            self.medians_[col] = float(median) if pd.notna(median) else 0.0
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.fit_index_ is None:
            raise PreprocessError("TrainScopedMedianImputer.transform called before fit")
        # Build every column in a dict first, then construct the DataFrame
        # once -- assigning ~2 columns per input column one at a time into
        # a growing frame triggers pandas' "highly fragmented" performance
        # warning at real-dataset column counts (~200 numeric columns).
        columns: dict[str, pd.Series] = {}
        for col in self.columns:
            columns[f"{col}_was_missing"] = df[col].isna().astype("int8")
            columns[col] = df[col].fillna(self.medians_[col]).astype("float32")
        return pd.DataFrame(columns, index=df.index)


@dataclass
class FittedPreprocessor:
    """Bundles the two fitted transformers plus the passthrough numeric columns.

    `has_identity` (and any other already-clean, purely-numeric,
    already-0/1 columns the caller doesn't want re-encoded) passes through
    untouched -- it needs neither imputation (no missingness by
    construction) nor frequency encoding (it's already a clean indicator).
    """

    frequency_encoder: TrainScopedFrequencyEncoder
    median_imputer: TrainScopedMedianImputer
    passthrough_columns: list[str]

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        parts = [
            df[self.passthrough_columns].astype("float32"),
            self.median_imputer.transform(df),
            self.frequency_encoder.transform(df),
        ]
        return pd.concat(parts, axis=1)


def fit_preprocessor(
    train_df: pd.DataFrame,
    exclude_columns: tuple[str, ...] = DEFAULT_EXCLUDED_COLUMNS,
    passthrough_columns: tuple[str, ...] = ("has_identity",),
) -> FittedPreprocessor:
    """Fit a frequency encoder + median imputer on `train_df` only.

    `train_df` must already be the caller's train-only partition -- this
    function does no splitting or filtering itself, by design: the split
    boundary decision belongs to `src/data/split.py`, and mixing that
    concern into the fit function would make it too easy to accidentally
    pass an unfiltered frame here. `src/data/leakage.py` audits the result
    (`fit_index_`) rather than trusting this contract silently.
    """
    roles = infer_column_roles(train_df, exclude_columns=exclude_columns)
    numeric_to_impute = [c for c in roles["numeric"] if c not in passthrough_columns]

    frequency_encoder = TrainScopedFrequencyEncoder(columns=roles["categorical"]).fit(train_df)
    median_imputer = TrainScopedMedianImputer(columns=numeric_to_impute).fit(train_df)

    return FittedPreprocessor(
        frequency_encoder=frequency_encoder,
        median_imputer=median_imputer,
        passthrough_columns=list(passthrough_columns),
    )


def build_feature_matrix(
    df: pd.DataFrame, preprocessor: FittedPreprocessor
) -> tuple[pd.DataFrame, pd.Series]:
    """Transform `df` into (X, y) using an already-fit preprocessor.

    X is indexed by TransactionID (removed as a column so it cannot be
    accidentally selected downstream -- defense in depth on top of it
    already being excluded from `infer_column_roles`; see
    DEFAULT_EXCLUDED_COLUMNS). y is `isFraud`, same index.
    """
    indexed = df.set_index("TransactionID", drop=True)
    y = indexed["isFraud"].astype("int8")
    X = preprocessor.transform(indexed)
    return X, y
