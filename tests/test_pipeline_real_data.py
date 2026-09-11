"""Phase 2 pipeline integration test against the REAL IEEE-CIS dataset.

Skipped when data/raw/{train_transaction,train_identity}.csv are absent,
mirroring tests/test_eda_real_data.py and tests/test_fhe_environment_smoke.py:
the real ~650MB dataset must never be required for `pytest` to pass, but
when present, this proves the full split/leakage/preprocessing pipeline
actually runs end-to-end on the real data (not just the synthetic wiring
test in test_pipeline_integration.py).

The pipeline runs ONCE per test session (module-scoped fixture) -- it
takes real time against the real dataset, and each test below asserts a
different property of that single run's output rather than re-running it.
"""

import pandas as pd
import pytest

from src.config import PROJECT_ROOT, load_config
from src.data.pipeline import run

_CONFIG_PATH = "configs/phase2/pipeline.yaml"


def _real_data_available() -> bool:
    config = load_config(_CONFIG_PATH)
    raw_dir = PROJECT_ROOT / config["data"]["raw_dir"]
    transaction_path = raw_dir / config["data"]["transaction_file"]
    identity_path = raw_dir / config["data"]["identity_file"]
    return transaction_path.exists() and identity_path.exists()


pytestmark = pytest.mark.skipif(
    not _real_data_available(),
    reason=(
        "Real IEEE-CIS data not found under data/raw/. See "
        "docs/data_acquisition.md for the manual download procedure."
    ),
)


@pytest.fixture(scope="module")
def pipeline_result():
    return run(_CONFIG_PATH)


@pytest.fixture(scope="module")
def output_dir():
    return PROJECT_ROOT / load_config(_CONFIG_PATH)["output"]["dir"]


def test_pipeline_runs_end_to_end_on_real_data(pipeline_result):
    assert pipeline_result["audit_report"]["all_passed"] is True


def test_pipeline_real_data_outputs_exist(pipeline_result, output_dir):
    for filename in (
        "split_assignments.csv",
        "split_boundaries.json",
        "leakage_audit_report.json",
        "preprocessing_summary.json",
        "provenance.json",
    ):
        assert (output_dir / filename).exists(), f"missing {filename}"


def test_pipeline_real_data_split_sizes_are_plausible(pipeline_result, output_dir):
    assignments = pd.read_csv(output_dir / "split_assignments.csv")
    # 590,540 rows total per docs/eda.md.
    assert len(assignments) == 590_540
    counts = assignments["split"].value_counts()
    assert counts["train"] > counts["val"] > 0
    assert counts["test"] > 0
    assert counts.sum() == 590_540


def test_pipeline_real_data_no_row_dropped_or_duplicated(pipeline_result, output_dir):
    assignments = pd.read_csv(output_dir / "split_assignments.csv")
    assert assignments["TransactionID"].is_unique


def test_pipeline_real_data_feature_matrix_has_no_banned_columns(pipeline_result):
    summary = pipeline_result["preprocessing_summary"]
    for banned in ("TransactionID", "TransactionDT", "isFraud"):
        assert banned not in summary["feature_columns"]


def test_pipeline_real_data_positive_rates_are_plausible(pipeline_result):
    """Sanity check against docs/eda.md Sec.1's measured global rate (3.499%)."""
    rates = pipeline_result["preprocessing_summary"]["positive_rate"]
    for split_name, rate in rates.items():
        assert 0.0 < rate < 0.15, f"{split_name} positive rate {rate} is implausible"
