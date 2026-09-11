"""Phase 1 EDA driver for the IEEE-CIS Fraud Detection dataset.

Run as:
    python -m src.data.eda --config configs/phase1/eda.yaml

Writes machine-readable aggregate tables to `output.dir` and a handful of
plots. Phase 1 persists no derived training data and performs no
persisted join -- only EDA aggregates. The split/leakage-audit pipeline
and the materialized transaction<->identity join are Phase 2's job
(docs/plan.md Phase 1 vs Phase 2; architecture.md Sec.15).

Design notes (see docs/eda.md for the findings themselves):
  - Reads happen in a small, deliberate number of passes over the ~650MB
    train_transaction.csv, not one-pass-does-everything cleverness:
      Step A: a tiny usecols=[TransactionID, TransactionDT, isFraud,
        TransactionAmt] load (~19 MB) -- cheap enough to fully materialize,
        and sufficient for schema validation, TransactionDT semantics,
        class balance over time, and candidate split boundaries.
      Step B: one full chunked forward pass (all columns) for
        missingness, V-column null-mask grouping, categorical
        cardinality, and OOV rate / positives-per-fold (both computed in
        an expanding-window fashion consistent with the project's actual
        split philosophy -- see `_ExpandingCategoricalTracker`).
      Step C: one full load restricted to numeric (float) columns, for
        exact min/max/mean/std/skew/quantiles via pandas built-ins.
  - Chunked-exact aggregation is used in preference to the sampling
    fallback docs/plan.md mentions, per the Phase 1 plan's documented
    deviation: sampling cannot establish exact tail-bucket counts, exact
    null-mask identity between columns, or exact OOV rates.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # headless: this is a script, never an interactive session
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT, load_config
from src.data import load as data_load
from src.data import provenance as data_provenance
from src.data import schema
from src.logging_setup import get_logger

logger = get_logger(__name__)

LIBRARY_NAMES = ["numpy", "pandas", "scikit-learn", "xgboost", "matplotlib", "PyYAML"]


class DataAcquisitionError(Exception):
    """Raised when the raw dataset files are missing or fail integrity checks."""


# ---------------------------------------------------------------------------
# File resolution and integrity
# ---------------------------------------------------------------------------


def resolve_raw_paths(config: dict[str, Any]) -> tuple[Path, Path]:
    raw_dir = PROJECT_ROOT / config["data"]["raw_dir"]
    transaction_path = raw_dir / config["data"]["transaction_file"]
    identity_path = raw_dir / config["data"]["identity_file"]
    missing = [p for p in (transaction_path, identity_path) if not p.exists()]
    if missing:
        names = ", ".join(str(p) for p in missing)
        raise DataAcquisitionError(
            f"Raw data file(s) not found: {names}. "
            "See docs/data_acquisition.md for the manual download procedure."
        )
    return transaction_path, identity_path


def verify_or_report_digest(path: Path, expected: str, file_key: str) -> str:
    digest = data_load.compute_file_digest(path)
    if not expected:
        logger.info(
            "No expected SHA256 recorded yet for %s; computed %s. "
            "Record this in configs/phase1/eda.yaml under data.expected_sha256.%s "
            "so future runs verify against it.",
            file_key,
            digest,
            file_key,
            extra={"extra_fields": {"file": file_key, "sha256": digest, "verified": False}},
        )
    elif digest != expected:
        raise DataAcquisitionError(
            f"{file_key}: SHA256 mismatch. Expected {expected}, got {digest}. "
            "The download may be truncated or corrupted -- re-download per "
            "docs/data_acquisition.md."
        )
    else:
        logger.info(
            "%s SHA256 verified.",
            file_key,
            extra={"extra_fields": {"file": file_key, "sha256": digest, "verified": True}},
        )
    return digest


# ---------------------------------------------------------------------------
# Step A -- small paired-column load: schema validation, DT semantics,
# class balance over time, candidate split boundaries.
# ---------------------------------------------------------------------------

_CORE_COLUMNS = ["TransactionID", "TransactionDT", "isFraud", "TransactionAmt"]


def load_core_frame(transaction_path: Path) -> pd.DataFrame:
    return data_load.load_full(
        transaction_path,
        dtype_map={
            "TransactionID": "int64",
            "TransactionDT": "int64",
            "isFraud": "int64",
            "TransactionAmt": "float64",
        },
        usecols=_CORE_COLUMNS,
    )


def analyze_transactiondt_semantics(core: pd.DataFrame) -> dict[str, Any]:
    dt = core["TransactionDT"]
    dt_min, dt_max = int(dt.min()), int(dt.max())
    span_seconds = dt_max - dt_min

    hour_of_day = (dt // 3600) % 24
    hour_counts = hour_of_day.value_counts().sort_index()
    trough_hour = int(hour_counts.idxmin())
    diurnal_amplitude = float((hour_counts.max() - hour_counts.min()) / hour_counts.mean())

    is_monotonic = bool(dt.is_monotonic_increasing)
    diffs = dt.diff().dropna()
    duplicate_dt_count = int((diffs == 0).sum())
    max_gap_seconds = int(diffs.max()) if len(diffs) else 0

    spearman_id_dt = float(core["TransactionID"].corr(core["TransactionDT"], method="spearman"))

    return {
        "dt_min": dt_min,
        "dt_max": dt_max,
        "span_seconds": span_seconds,
        "span_days_if_seconds": span_seconds / 86400,
        "span_years_if_seconds_since_epoch": (dt_max - dt_min) / (365.25 * 86400)
        if dt_min > 1_000_000_000
        else None,
        "epoch_reading_plausible": dt_min > 1_000_000_000,
        "hour_of_day_counts": {int(h): int(c) for h, c in hour_counts.items()},
        "diurnal_trough_hour": trough_hour,
        "diurnal_relative_amplitude": diurnal_amplitude,
        "is_monotonic_non_decreasing": is_monotonic,
        "adjacent_duplicate_dt_count": duplicate_dt_count,
        "max_consecutive_gap_seconds": max_gap_seconds,
        "spearman_transactionid_vs_transactiondt": spearman_id_dt,
    }


def compute_candidate_boundaries(core: pd.DataFrame, quantiles: list[float]) -> list[dict[str, Any]]:
    """Time-RANGE quantile boundaries (linear interpolation of DT min/max).

    Deliberately time-range quantiles, not row-count quantiles: reports
    the actual row/positive counts each produces (see
    `compute_positives_per_fold`) so Phase 2 can see directly whether
    transaction volume is even enough over time for this to be a
    reasonable boundary choice, or whether row-count quantiles would be
    better -- that comparison is the point of this analysis.
    """
    dt_min, dt_max = int(core["TransactionDT"].min()), int(core["TransactionDT"].max())
    boundaries = []
    for q in quantiles:
        dt_boundary = int(round(dt_min + q * (dt_max - dt_min)))
        boundaries.append({"quantile": q, "transactiondt_boundary": dt_boundary})
    return boundaries


def compute_positives_per_fold(core: pd.DataFrame, boundaries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    edges = [core["TransactionDT"].min() - 1] + [b["transactiondt_boundary"] for b in boundaries] + [
        core["TransactionDT"].max() + 1
    ]
    fold_stats = []
    for i in range(len(edges) - 1):
        mask = (core["TransactionDT"] > edges[i]) & (core["TransactionDT"] <= edges[i + 1])
        fold = core.loc[mask]
        fold_stats.append(
            {
                "fold_index": i,
                "dt_lower_exclusive": int(edges[i]),
                "dt_upper_inclusive": int(edges[i + 1]),
                "n_rows": int(len(fold)),
                "n_positive": int(fold["isFraud"].sum()),
                "positive_rate": float(fold["isFraud"].mean()) if len(fold) else None,
            }
        )
    return fold_stats


def compute_class_balance_by_day(core: pd.DataFrame, time_bucket_seconds: int) -> pd.DataFrame:
    day_index = core["TransactionDT"] // time_bucket_seconds
    grouped = core.groupby(day_index)["isFraud"].agg(n_rows="count", n_positive="sum")
    grouped["positive_rate"] = grouped["n_positive"] / grouped["n_rows"]
    grouped.index.name = "day_index"
    return grouped.reset_index()


# ---------------------------------------------------------------------------
# Step B -- single full chunked pass: missingness, V-column null-mask
# grouping, categorical cardinality, expanding-window OOV / positives.
# ---------------------------------------------------------------------------


class _ExpandingCategoricalTracker:
    """Per-column value Counters, used for both cardinality and OOV.

    A column's Counter keys double as its "seen so far" set. OOV for fold
    i is checked against the Counter as it stood at the *start* of fold i
    (i.e. all prior folds) -- consistent with the project's expanding-
    window split philosophy (docs/plan.md Sec.6), not an arbitrary
    train/test split.
    """

    def __init__(self, categorical_columns: list[str], n_folds: int) -> None:
        self.counters: dict[str, Counter[str]] = {c: Counter() for c in categorical_columns}
        self.oov_counts = np.zeros((len(categorical_columns), n_folds), dtype=np.int64)
        self.total_counts = np.zeros((len(categorical_columns), n_folds), dtype=np.int64)
        self.columns = categorical_columns

    def observe_fold(self, fold_index: int, sub_chunk: pd.DataFrame) -> None:
        for col_idx, col in enumerate(self.columns):
            values = sub_chunk[col].dropna()
            if fold_index > 0:
                seen = self.counters[col]
                is_oov = ~values.isin(seen.keys())
                self.oov_counts[col_idx, fold_index] += int(is_oov.sum())
                self.total_counts[col_idx, fold_index] += int(len(values))
            self.counters[col].update(values.astype(str))

    def oov_table(self) -> pd.DataFrame:
        rows = []
        for col_idx, col in enumerate(self.columns):
            for fold_index in range(1, self.oov_counts.shape[1]):
                total = int(self.total_counts[col_idx, fold_index])
                oov = int(self.oov_counts[col_idx, fold_index])
                rows.append(
                    {
                        "column": col,
                        "fold_index": fold_index,
                        "n_values": total,
                        "n_oov": oov,
                        "oov_rate": oov / total if total else None,
                    }
                )
        return pd.DataFrame(rows)

    def cardinality_table(self) -> pd.DataFrame:
        return pd.DataFrame(
            [{"column": c, "n_distinct": len(counter)} for c, counter in self.counters.items()]
        )


def _fold_edges(core: pd.DataFrame, boundaries: list[dict[str, Any]]) -> list[int]:
    return (
        [int(core["TransactionDT"].min()) - 1]
        + [b["transactiondt_boundary"] for b in boundaries]
        + [int(core["TransactionDT"].max()) + 1]
    )


def run_full_pass(
    transaction_path: Path,
    load_config_dict: dict[str, Any],
    dtype_map: dict[str, str],
    fold_edges: list[int],
    time_bucket_seconds: int,
    total_rows: int,
) -> dict[str, Any]:
    categorical_columns = sorted(c for c, dt in dtype_map.items() if dt == "category")
    v_columns = sorted(
        c for c in dtype_map if schema.classify_column(c) == "V"
    )

    tracker = _ExpandingCategoricalTracker(categorical_columns, n_folds=len(fold_edges) - 1)
    v_hashers: dict[str, "hashlib._Hash"] = {c: hashlib.sha256() for c in v_columns}
    missingness_by_day = pd.DataFrame()
    rows_seen = 0

    for chunk_idx, chunk in enumerate(
        data_load.iter_chunks(transaction_path, load_config_dict, dtype_map)
    ):
        rows_seen += len(chunk)
        day_index = chunk["TransactionDT"] // time_bucket_seconds
        chunk_missing_by_day = chunk.groupby(day_index).count()
        missingness_by_day = (
            chunk_missing_by_day
            if missingness_by_day.empty
            else missingness_by_day.add(chunk_missing_by_day, fill_value=0)
        )

        for col in v_columns:
            mask_bytes = np.packbits(chunk[col].isna().to_numpy()).tobytes()
            v_hashers[col].update(mask_bytes)

        # Fold assignment for this chunk, respecting the expanding-window
        # definition; chunks are processed in file order and the file's
        # sortedness by TransactionDT was verified in Step A.
        dt = chunk["TransactionDT"]
        for fold_index in range(len(fold_edges) - 1):
            mask = (dt > fold_edges[fold_index]) & (dt <= fold_edges[fold_index + 1])
            if mask.any():
                tracker.observe_fold(fold_index, chunk.loc[mask])

        if chunk_idx % 5 == 0:
            logger.info(
                "Full pass progress",
                extra={"extra_fields": {"rows_seen": rows_seen, "total_rows": total_rows}},
            )

    # Repeated .add() across ~12 chunks leaves the frame fragmented (pandas
    # PerformanceWarning); defragment once, here, rather than per chunk.
    missingness_by_day = missingness_by_day.copy()
    global_missing = (missingness_by_day.sum(axis=0)).astype(int)
    global_non_null = global_missing  # count() sums are non-null counts
    missingness_global = pd.DataFrame(
        {
            "column": global_non_null.index,
            "non_null_count": global_non_null.values,
            "null_count": total_rows - global_non_null.values,
            "null_rate": (total_rows - global_non_null.values) / total_rows,
            "family": [schema.classify_column(c) for c in global_non_null.index],
        }
    ).sort_values("null_rate", ascending=False)

    v_hashes = {c: h.hexdigest() for c, h in v_hashers.items()}
    hash_to_cols: dict[str, list[str]] = defaultdict(list)
    for col, digest in v_hashes.items():
        hash_to_cols[digest].append(col)
    v_null_mask_blocks = pd.DataFrame(
        [
            {"block_id": i, "null_mask_sha256": digest, "n_columns": len(cols), "columns": ",".join(cols)}
            for i, (digest, cols) in enumerate(sorted(hash_to_cols.items(), key=lambda kv: -len(kv[1])))
        ]
    )

    return {
        "missingness_global": missingness_global,
        "missingness_by_time_bucket": missingness_by_day.reset_index(names="day_index"),
        "v_null_mask_blocks": v_null_mask_blocks,
        "oov_rates": tracker.oov_table(),
        "cardinality": tracker.cardinality_table(),
        "rows_seen": rows_seen,
    }


# ---------------------------------------------------------------------------
# Step C -- full numeric-only load: exact min/max/mean/std/skew/quantiles.
# ---------------------------------------------------------------------------


def compute_numeric_ranges(transaction_path: Path, dtype_map: dict[str, str]) -> tuple[pd.DataFrame, int]:
    numeric_cols = [c for c, dt in dtype_map.items() if dt in ("float32", "float64")]
    numeric_df = data_load.load_full(transaction_path, dtype_map=dtype_map, usecols=numeric_cols)
    footprint_bytes = int(numeric_df.memory_usage(deep=True).sum())

    stats = numeric_df.agg(["min", "max", "mean", "std", "skew"]).T
    quantiles = numeric_df.quantile([0.25, 0.5, 0.75]).T
    quantiles.columns = ["p25", "p50", "p75"]
    combined = stats.join(quantiles)
    combined.index.name = "column"
    combined["family"] = [schema.classify_column(c) for c in combined.index]
    return combined.reset_index(), footprint_bytes


# ---------------------------------------------------------------------------
# Identity table
# ---------------------------------------------------------------------------


def analyze_identity_coverage(identity_path: Path, core: pd.DataFrame) -> dict[str, Any]:
    identity_ids = data_load.load_full(
        identity_path, dtype_map={"TransactionID": "int64"}, usecols=["TransactionID"]
    )["TransactionID"]
    schema.validate_identity_subset(core["TransactionID"], identity_ids)

    has_identity = core["TransactionID"].isin(set(identity_ids))
    coverage_overall = float(has_identity.mean())
    coverage_by_fraud = core.groupby("isFraud").apply(
        lambda g: float(g["TransactionID"].isin(set(identity_ids)).mean()), include_groups=False
    )
    return {
        "n_identity_rows": int(len(identity_ids)),
        "coverage_overall": coverage_overall,
        "coverage_by_isfraud": {int(k): float(v) for k, v in coverage_by_fraud.items()},
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def make_plots(
    class_balance_by_day: pd.DataFrame,
    transactiondt_semantics: dict[str, Any],
    numeric_ranges: pd.DataFrame,
    core: pd.DataFrame,
    output_dir: Path,
) -> list[str]:
    written = []

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(class_balance_by_day["day_index"], class_balance_by_day["positive_rate"])
    ax.set_xlabel("Day index (TransactionDT // 86400)")
    ax.set_ylabel("Fraud rate")
    ax.set_title("Fraud rate over time")
    fig.tight_layout()
    path = output_dir / "fraud_rate_over_time.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    written.append(path.name)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(class_balance_by_day["day_index"], class_balance_by_day["n_rows"])
    ax.set_xlabel("Day index (TransactionDT // 86400)")
    ax.set_ylabel("Transaction count")
    ax.set_title("Transaction volume over time")
    fig.tight_layout()
    path = output_dir / "volume_over_time.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    written.append(path.name)

    hour_counts = transactiondt_semantics["hour_of_day_counts"]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(list(hour_counts.keys()), list(hour_counts.values()))
    ax.set_xlabel("Hour of day (TransactionDT // 3600 mod 24)")
    ax.set_ylabel("Transaction count")
    ax.set_title("Hour-of-day histogram (evidence TransactionDT is in seconds)")
    fig.tight_layout()
    path = output_dir / "hour_of_day_histogram.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    written.append(path.name)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(core["TransactionAmt"].clip(upper=core["TransactionAmt"].quantile(0.99)), bins=60)
    ax.set_xlabel("TransactionAmt (clipped at p99)")
    ax.set_ylabel("Count")
    ax.set_title("TransactionAmt distribution")
    fig.tight_layout()
    path = output_dir / "transaction_amt_distribution.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    written.append(path.name)

    return written


# ---------------------------------------------------------------------------
# Output writers
# ---------------------------------------------------------------------------


def _write_json(obj: Any, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=2, sort_keys=True, default=str), encoding="utf-8")


def _write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, lineterminator="\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run(config_path: str) -> None:
    config = load_config(config_path)
    transaction_path, identity_path = resolve_raw_paths(config)
    output_dir = PROJECT_ROOT / config["output"]["dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Verifying raw file integrity")
    expected = config["data"]["expected_sha256"]
    tx_digest = verify_or_report_digest(transaction_path, expected.get("train_transaction.csv", ""), "train_transaction.csv")
    id_digest = verify_or_report_digest(identity_path, expected.get("train_identity.csv", ""), "train_identity.csv")

    logger.info("Reading headers and validating schema")
    transaction_header = data_load.read_header(transaction_path)
    identity_header = data_load.read_header(identity_path)
    schema.validate_required_columns(transaction_header, schema.REQUIRED_TRANSACTION_COLUMNS, "train_transaction")
    schema.validate_required_columns(identity_header, schema.REQUIRED_IDENTITY_COLUMNS, "train_identity")
    column_diff = schema.diff_columns(
        transaction_header, schema.REQUIRED_TRANSACTION_COLUMNS
    )

    logger.info("Step A: loading core columns")
    core = load_core_frame(transaction_path)
    schema.validate_unique_transaction_id(core, "train_transaction")
    schema.validate_is_fraud(core["isFraud"])
    schema.validate_transaction_amt(core["TransactionAmt"])
    total_rows = len(core)

    dt_semantics = analyze_transactiondt_semantics(core)
    boundaries = compute_candidate_boundaries(core, config["eda"]["candidate_split_quantiles"])
    positives_per_fold = compute_positives_per_fold(core, boundaries)
    class_balance_by_day = compute_class_balance_by_day(core, config["eda"]["time_bucket_seconds"])
    identity_coverage = analyze_identity_coverage(identity_path, core)

    logger.info("Inferring dtype map from a sample")
    dtype_map = data_load.infer_dtype_map(transaction_path, config["load"])

    logger.info("Step B: full chunked pass (missingness, V-masks, cardinality, OOV)")
    fold_edges = _fold_edges(core, boundaries)
    full_pass = run_full_pass(
        transaction_path,
        config["load"],
        dtype_map,
        fold_edges,
        config["eda"]["time_bucket_seconds"],
        total_rows,
    )

    logger.info("Step C: full numeric-only load for exact ranges/quantiles/skew")
    numeric_ranges, numeric_footprint_bytes = compute_numeric_ranges(transaction_path, dtype_map)

    logger.info("Writing plots")
    plot_files = make_plots(class_balance_by_day, dt_semantics, numeric_ranges, core, output_dir)

    logger.info("Writing output tables")
    column_inventory = pd.DataFrame(
        {"column": transaction_header, "family": [schema.classify_column(c) for c in transaction_header], "dtype": [dtype_map.get(c, "n/a") for c in transaction_header]}
    )
    _write_csv(column_inventory, output_dir / "column_inventory.csv")
    _write_csv(full_pass["missingness_global"], output_dir / "missingness_global.csv")
    _write_csv(full_pass["missingness_by_time_bucket"], output_dir / "missingness_by_time_bucket.csv")
    _write_csv(full_pass["v_null_mask_blocks"], output_dir / "v_null_mask_blocks.csv")
    _write_csv(class_balance_by_day, output_dir / "class_balance_by_time.csv")
    _write_csv(numeric_ranges, output_dir / "numeric_ranges.csv")
    _write_csv(full_pass["oov_rates"], output_dir / "oov_rates.csv")
    _write_csv(full_pass["cardinality"], output_dir / "cardinality.csv")
    _write_csv(pd.DataFrame(positives_per_fold), output_dir / "candidate_split_boundaries.csv")
    _write_json(dt_semantics, output_dir / "transactiondt_semantics.json")

    dataset_summary = {
        "n_transaction_rows": total_rows,
        "n_transaction_columns": len(transaction_header),
        "n_identity_rows": identity_coverage["n_identity_rows"],
        "n_identity_columns": len(identity_header),
        "global_fraud_rate": float(core["isFraud"].mean()),
        "global_n_positive": int(core["isFraud"].sum()),
        "column_diff_vs_required": column_diff,
        "identity_coverage": identity_coverage,
        "numeric_full_load_footprint_bytes": numeric_footprint_bytes,
        "plot_files": plot_files,
    }
    _write_json(dataset_summary, output_dir / "dataset_summary.json")

    provenance = data_provenance.build_provenance(
        config=config,
        config_path=Path(config_path),
        seed=config["seed"],
        raw_file_digests={"train_transaction.csv": tx_digest, "train_identity.csv": id_digest},
        library_names=LIBRARY_NAMES,
    )
    _write_json(provenance, output_dir / "provenance.json")

    logger.info(
        "Phase 1 EDA complete",
        extra={"extra_fields": {"output_dir": str(output_dir), "n_rows": total_rows}},
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/phase1/eda.yaml", help="Path to the EDA config YAML"
    )
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
