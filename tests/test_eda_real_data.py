"""Phase 1 EDA integration test against the REAL IEEE-CIS dataset.

Skipped when data/raw/{train_transaction,train_identity}.csv are absent,
mirroring the pattern in tests/test_fhe_environment_smoke.py: the real
~650 MB dataset must never be required for `pytest` to pass, but when
present, this proves the full pipeline actually runs end-to-end (not
just the synthetic-data wiring test in test_eda_integration.py).
"""

from pathlib import Path

import pytest

from src.config import PROJECT_ROOT, load_config
from src.data.eda import run

_CONFIG_PATH = "configs/phase1/eda.yaml"


def _real_data_available() -> bool:
    config = load_config(_CONFIG_PATH)
    raw_dir = PROJECT_ROOT / config["data"]["raw_dir"]
    transaction_path = raw_dir / config["data"]["transaction_file"]
    identity_path = raw_dir / config["data"]["identity_file"]
    return transaction_path.exists() and identity_path.exists()


@pytest.mark.skipif(
    not _real_data_available(),
    reason=(
        "Real IEEE-CIS data not found under data/raw/. See "
        "docs/data_acquisition.md for the manual download procedure."
    ),
)
def test_run_end_to_end_on_real_data():
    run(_CONFIG_PATH)
    output_dir = PROJECT_ROOT / load_config(_CONFIG_PATH)["output"]["dir"]
    assert (output_dir / "dataset_summary.json").exists()
    assert (output_dir / "transactiondt_semantics.json").exists()
    assert (output_dir / "provenance.json").exists()
