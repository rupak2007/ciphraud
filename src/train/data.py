"""The Phase 2 pipeline as the SOLE source of model-ready data for Phase 3.

Nothing in `src/train/` re-derives loading, joining, splitting, or
preprocessing logic. Every feature matrix Phase 3 ever trains or evaluates
on comes from exactly one place: `src.data.pipeline.run()` -- either a
verified-fresh on-disk cache it already wrote (`data/processed/*.pkl`,
`results/phase2_pipeline/`), or a freshly (re-)executed run when that cache
doesn't match the current `configs/phase2/pipeline.yaml`. This is what
`load_phase2_features` below enforces: it never reads the cache blindly.

This also recovers the per-row `TransactionDT` and the Phase 2 expanding-
window CV fold boundaries for the train partition, needed for Phase 3's
own leakage-safe hyperparameter selection (`src/train/*_model.py`) --
reconstructed from Phase 2's own persisted `split_assignments.csv` /
`split_boundaries.json`, never recomputed from the raw CSVs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from src.config import PROJECT_ROOT, load_config
from src.data import pipeline as data_pipeline
from src.data import provenance as data_provenance
from src.data import split as data_split
from src.logging_setup import get_logger

logger = get_logger(__name__)


@dataclass
class Phase2Features:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    train_dt: pd.Series  # TransactionDT per train row, aligned to X_train's index/order
    boundaries: data_split.SplitBoundaries
    # Train+val CONCATENATED, for CV-fold-based hyperparameter selection
    # ONLY (src/train/logistic_regression.py, src/train/xgboost_model.py).
    # Phase 2's expanding-window CV folds (docs/pipeline.md Sec.1) span the
    # primary train+val region by design -- the last fold's eval window
    # falls inside the primary "val" partition's own DT range, which
    # X_train/train_dt alone cannot cover (they only span up to
    # train_val_boundary). Passing X_train/train_dt to CV selection
    # instead of these fields produces an empty eval slice for that fold
    # and a hard crash -- this is exactly the bug this comment exists to
    # prevent regressing. The FINAL model refit still uses X_train/y_train
    # only (never val), per docs/baselines.md Sec.2.
    X_train_plus_val: pd.DataFrame
    y_train_plus_val: pd.Series
    train_plus_val_dt: pd.Series


def _cache_is_fresh(config: dict, processed_dir: Path, phase2_output_dir: Path) -> bool:
    """A cache is fresh iff every expected file exists AND the recorded
    config_hash in results/phase2_pipeline/provenance.json matches the
    CURRENT configs/phase2/pipeline.yaml content -- not just file
    presence/mtime, which wouldn't catch an edited-then-untouched config."""
    required = [
        processed_dir / "train.pkl",
        processed_dir / "val.pkl",
        processed_dir / "test.pkl",
        phase2_output_dir / "provenance.json",
        phase2_output_dir / "split_assignments.csv",
        phase2_output_dir / "split_boundaries.json",
    ]
    if not all(p.exists() for p in required):
        return False

    import json

    recorded_hash = json.loads((phase2_output_dir / "provenance.json").read_text())["config_hash"]
    current_hash = data_provenance.compute_config_hash(config)
    return recorded_hash == current_hash


def load_phase2_features(pipeline_config_path: str = "configs/phase2/pipeline.yaml") -> Phase2Features:
    """The ONLY entry point Phase 3 code uses to obtain model-ready data.

    Fast path: config unchanged since the last Phase 2 run -> load the
    already-written, already-leakage-audited cache. Slow path: config
    changed (or cache missing) -> re-run `src.data.pipeline.run()` in
    full, which re-verifies raw-file SHA256s, re-runs the leakage audit,
    and rewrites the cache -- Phase 3 never trusts a stale or hand-edited
    cache.
    """
    config = load_config(pipeline_config_path)
    processed_dir = PROJECT_ROOT / config["output"]["processed_dir"]
    phase2_output_dir = PROJECT_ROOT / config["output"]["dir"]

    if _cache_is_fresh(config, processed_dir, phase2_output_dir):
        logger.info("Phase 2 cache is fresh (config_hash matches); loading directly")
    else:
        logger.info("Phase 2 cache missing or stale; re-running src.data.pipeline")
        data_pipeline.run(pipeline_config_path)

    train = pd.read_pickle(processed_dir / "train.pkl")
    val = pd.read_pickle(processed_dir / "val.pkl")
    test = pd.read_pickle(processed_dir / "test.pkl")

    X_train, y_train = train.drop(columns="isFraud"), train["isFraud"]
    X_val, y_val = val.drop(columns="isFraud"), val["isFraud"]
    X_test, y_test = test.drop(columns="isFraud"), test["isFraud"]

    assignments = pd.read_csv(phase2_output_dir / "split_assignments.csv").set_index("TransactionID")
    train_dt = assignments.loc[X_train.index, "TransactionDT"]

    X_train_plus_val = pd.concat([X_train, X_val])
    y_train_plus_val = pd.concat([y_train, y_val])
    train_plus_val_dt = assignments.loc[X_train_plus_val.index, "TransactionDT"]

    import json

    boundaries = data_split.SplitBoundaries.from_dict(
        json.loads((phase2_output_dir / "split_boundaries.json").read_text())
    )

    logger.info(
        "Loaded Phase 2 features",
        extra={
            "extra_fields": {
                "n_train": len(X_train),
                "n_val": len(X_val),
                "n_test": len(X_test),
                "n_features": X_train.shape[1],
            }
        },
    )

    return Phase2Features(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        train_dt=train_dt,
        boundaries=boundaries,
        X_train_plus_val=X_train_plus_val,
        y_train_plus_val=y_train_plus_val,
        train_plus_val_dt=train_plus_val_dt,
    )
