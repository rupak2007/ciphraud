"""Phase 2 split + leakage-safe preprocessing pipeline for IEEE-CIS Fraud Detection.

Run as:
    python -m src.data.pipeline --config configs/phase2/pipeline.yaml

Orchestrates, in order:
  1. Load + validate the raw transaction/identity CSVs (reusing
     src/data/load.py + src/data/schema.py from Phase 1).
  2. Join transaction<->identity (src/data/preprocess.py::join_transaction_identity)
     -- a structural join, no statistic fit, no leakage risk on its own.
  3. Compute time-based expanding-window split boundaries and assign every
     row to train/val/test + an expanding-window CV eval fold
     (src/data/split.py). Persist this as the split artifact.
  4. Fit the preprocessor (frequency encoder + median imputer) on the
     TRAIN partition only (src/data/preprocess.py::fit_preprocessor).
  5. Run the full leakage audit (src/data/leakage.py::run_leakage_audit)
     -- this is what actually verifies step 4 only touched train, not
     merely trusts it. A failing check raises and crashes this run.
  6. Transform train/val/test into feature matrices; optionally cache them
     under data/processed/ (gitignored) for Phase 3's convenience.
  7. Write the split artifact, leakage audit report, preprocessing
     summary, and provenance to results/phase2_pipeline/.

See docs/pipeline.md for the full methodology and rationale.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from src.config import PROJECT_ROOT, load_config
from src.data import leakage as data_leakage
from src.data import load as data_load
from src.data import preprocess as data_preprocess
from src.data import provenance as data_provenance
from src.data import schema
from src.data import split as data_split
from src.logging_setup import get_logger

logger = get_logger(__name__)

LIBRARY_NAMES = ["numpy", "pandas", "scikit-learn", "xgboost", "PyYAML"]


def load_and_validate_raw(config: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, str, str]:
    transaction_path, identity_path = data_load.resolve_raw_paths(config, PROJECT_ROOT)

    logger.info("Verifying raw file integrity")
    expected = config["data"]["expected_sha256"]
    tx_digest = data_load.verify_or_report_digest(
        transaction_path,
        expected.get("train_transaction.csv", ""),
        "train_transaction.csv",
        logger,
        "configs/phase2/pipeline.yaml under data.expected_sha256",
    )
    id_digest = data_load.verify_or_report_digest(
        identity_path,
        expected.get("train_identity.csv", ""),
        "train_identity.csv",
        logger,
        "configs/phase2/pipeline.yaml under data.expected_sha256",
    )

    logger.info("Loading transaction and identity tables (single-shot; measured safe -- see docs/pipeline.md)")
    tx_dtype_map = data_load.infer_dtype_map(transaction_path, config["load"])
    id_dtype_map = data_load.infer_dtype_map(identity_path, config["load"])
    transaction_df = data_load.load_full(transaction_path, dtype_map=tx_dtype_map)
    identity_df = data_load.load_full(identity_path, dtype_map=id_dtype_map)

    logger.info("Validating schema")
    schema.validate_transaction_dataframe(transaction_df)
    schema.validate_identity_dataframe(identity_df)
    schema.validate_identity_against_transaction(transaction_df, identity_df)

    return transaction_df, identity_df, tx_digest, id_digest


def run(config_path: str) -> dict[str, Any]:
    config = load_config(config_path)
    output_dir = PROJECT_ROOT / config["output"]["dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    transaction_df, identity_df, tx_digest, id_digest = load_and_validate_raw(config)

    logger.info("Joining transaction <-> identity")
    joined = data_preprocess.join_transaction_identity(transaction_df, identity_df)

    logger.info("Computing split boundaries and assignments")
    boundaries = data_split.compute_split_boundaries(joined["TransactionDT"], config["split"])
    split_labels = data_split.assign_primary_split(joined["TransactionDT"], boundaries)
    assignments = data_split.build_split_assignments(
        joined["TransactionID"], joined["TransactionDT"], boundaries
    )

    train_mask = split_labels == "train"
    val_mask = split_labels == "val"
    test_mask = split_labels == "test"
    train_df = joined.loc[train_mask].set_index("TransactionID", drop=False)
    val_df = joined.loc[val_mask].set_index("TransactionID", drop=False)
    test_df = joined.loc[test_mask].set_index("TransactionID", drop=False)

    logger.info(
        "Split sizes",
        extra={
            "extra_fields": {
                "n_train": int(train_mask.sum()),
                "n_val": int(val_mask.sum()),
                "n_test": int(test_mask.sum()),
            }
        },
    )

    logger.info("Fitting preprocessor on TRAIN partition only")
    exclude_columns = tuple(config["preprocess"]["exclude_columns"])
    passthrough_columns = tuple(config["preprocess"]["passthrough_columns"])
    preprocessor = data_preprocess.fit_preprocessor(
        train_df, exclude_columns=exclude_columns, passthrough_columns=passthrough_columns
    )

    logger.info("Transforming train/val/test")
    X_train, y_train = data_preprocess.build_feature_matrix(train_df, preprocessor)
    X_val, y_val = data_preprocess.build_feature_matrix(val_df, preprocessor)
    X_test, y_test = data_preprocess.build_feature_matrix(test_df, preprocessor)

    logger.info("Running leakage audit")
    audit_report = data_leakage.run_leakage_audit(
        dt=joined["TransactionDT"],
        split_labels=split_labels,
        cv_boundaries=boundaries.cv_boundaries,
        feature_columns=list(X_train.columns),
        banned_columns=exclude_columns,
        fitted_statistics={
            "frequency_encoder": preprocessor.frequency_encoder,
            "median_imputer": preprocessor.median_imputer,
        },
        train_index=train_df.index,
    )
    logger.info(
        "Leakage audit passed",
        extra={"extra_fields": {"n_checks": audit_report["n_checks"]}},
    )

    # OOV rates for the categorical columns actually used, measured on val
    # and test against the train-fit vocabulary -- turns docs/eda.md
    # Sec.6's "card1/2/3/5 and identity categoricals not profiled in
    # Phase 1" gap into a measured Phase 2 finding rather than leaving it
    # unmeasured (CLAUDE.md Sec.18: never silently drop a documented gap).
    oov_on_val = preprocessor.frequency_encoder.oov_rate(val_df)
    oov_on_test = preprocessor.frequency_encoder.oov_rate(test_df)

    if config["output"].get("cache_processed_data", False):
        processed_dir = PROJECT_ROOT / config["output"]["processed_dir"]
        processed_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Caching transformed feature matrices", extra={"extra_fields": {"dir": str(processed_dir)}})
        for name, X, y in (("train", X_train, y_train), ("val", X_val, y_val), ("test", X_test, y_test)):
            # Pickle, not Parquet: pyarrow isn't a project dependency (see
            # docs/environment.md), and this is a private, gitignored,
            # same-machine cache -- pickle needs no extra library and
            # preserves the float32/int8 dtypes exactly, unlike CSV.
            X.assign(isFraud=y).to_pickle(processed_dir / f"{name}.pkl")

    logger.info("Writing output tables")
    assignments.to_csv(output_dir / "split_assignments.csv", index=False, lineterminator="\n")
    (output_dir / "split_boundaries.json").write_text(
        json.dumps(
            {**boundaries.to_dict(), "quantiles": {k: config["split"][k] for k in ("train_val_quantile", "val_test_quantile", "cv_n_folds")}},
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    (output_dir / "leakage_audit_report.json").write_text(
        json.dumps(audit_report, indent=2, sort_keys=True), encoding="utf-8"
    )

    preprocessing_summary = {
        "n_features": X_train.shape[1],
        "feature_columns": list(X_train.columns),
        "excluded_columns": list(exclude_columns),
        "passthrough_columns": list(passthrough_columns),
        "numeric_imputed_columns": preprocessor.median_imputer.columns,
        "categorical_encoded_columns": preprocessor.frequency_encoder.columns,
        "imputation_medians": preprocessor.median_imputer.medians_,
        "oov_rate_on_val": oov_on_val,
        "oov_rate_on_test": oov_on_test,
        "split_sizes": {
            "train": int(train_mask.sum()),
            "val": int(val_mask.sum()),
            "test": int(test_mask.sum()),
        },
        "positive_rate": {
            "train": float(y_train.mean()),
            "val": float(y_val.mean()),
            "test": float(y_test.mean()),
        },
    }
    (output_dir / "preprocessing_summary.json").write_text(
        json.dumps(preprocessing_summary, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )

    provenance = data_provenance.build_provenance(
        config=config,
        config_path=Path(config_path),
        seed=config["seed"],
        raw_file_digests={"train_transaction.csv": tx_digest, "train_identity.csv": id_digest},
        library_names=LIBRARY_NAMES,
    )
    (output_dir / "provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )

    logger.info(
        "Phase 2 pipeline complete",
        extra={"extra_fields": {"output_dir": str(output_dir), "n_features": X_train.shape[1]}},
    )

    return {
        "audit_report": audit_report,
        "preprocessing_summary": preprocessing_summary,
        "boundaries": boundaries,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="configs/phase2/pipeline.yaml", help="Path to the Phase 2 pipeline config YAML"
    )
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
