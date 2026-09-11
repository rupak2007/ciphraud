"""Phase 4 pipeline integration test against the REAL IEEE-CIS dataset.

Skipped when Phase 2's model-ready data isn't available (mirroring every
other *_real_data.py test in this repo). Requires a committed Phase 3 run
whose config_hash matches the current configs/phase3/baselines.yaml (the
Phase 3 integrity gate, src/features/pipeline.py::_verify_phase3_integrity)
-- if Phase 3 hasn't been (re-)run since a config change, this test fails
with that gate's own clear error rather than a confusing downstream one.

The full pipeline runs ONCE per test session (module-scoped fixture) --
ranking + stability + the dense curve + per-tier CV re-tuning at this
scale takes real time.
"""

import json

import pytest

from src.config import PROJECT_ROOT, load_config
from src.features.pipeline import run
from src.features.tiers import load_tiers

_PHASE4_CONFIG_PATH = "configs/phase4/features.yaml"


def _real_data_available() -> bool:
    config = load_config(_PHASE4_CONFIG_PATH)
    phase2_config = load_config(config["phase2_config"])
    raw_dir = PROJECT_ROOT / phase2_config["data"]["raw_dir"]
    return (raw_dir / phase2_config["data"]["transaction_file"]).exists() and (
        raw_dir / phase2_config["data"]["identity_file"]
    ).exists()


pytestmark = pytest.mark.skipif(
    not _real_data_available(),
    reason=(
        "Real IEEE-CIS data not found under data/raw/. See "
        "docs/data_acquisition.md for the manual download procedure."
    ),
)


@pytest.fixture(scope="module")
def pipeline_result():
    return run(_PHASE4_CONFIG_PATH)


def test_pipeline_runs_end_to_end_on_real_data(pipeline_result):
    assert pipeline_result["test_partition_touched"] is False
    assert set(pipeline_result["tiers"].keys()) == {"top_20", "top_50", "top_100"}


def test_pipeline_tiers_are_nested(pipeline_result):
    config = load_config(_PHASE4_CONFIG_PATH)
    output_dir = PROJECT_ROOT / config["output"]["dir"]
    tiers = load_tiers(output_dir / "tiers.json")["tiers"]
    assert tiers["top_20"] == tiers["top_50"][:20]
    assert tiers["top_50"] == tiers["top_100"][:50]


def test_pipeline_every_tier_beats_random_baseline(pipeline_result):
    """Every tier's PR-AUC must clear the ~3.5% positive-rate baseline
    (docs/eda.md Sec.1) by a wide margin, for both models -- the same
    sanity floor tests/test_train_pipeline_real_data.py enforces for the
    full feature set, applied per tier."""
    for tier_name, tier in pipeline_result["tiers"].items():
        lr_pr_auc = tier["logistic_regression"]["val_evaluation"]["pr_auc"]
        xgb_pr_auc = tier["xgboost"]["val_evaluation"]["pr_auc"]
        assert lr_pr_auc > 0.15, f"{tier_name} LR PR-AUC {lr_pr_auc} too close to baseline"
        assert xgb_pr_auc > 0.15, f"{tier_name} XGBoost PR-AUC {xgb_pr_auc} too close to baseline"


def test_pipeline_reproduces_phase3_full_feature_metrics(pipeline_result):
    diff = pipeline_result["reproduction_check"]
    assert diff["lr_diff"] <= diff["tolerance"]
    assert diff["xgb_diff"] <= diff["tolerance"]


def test_pipeline_test_partition_not_in_results(pipeline_result):
    config = load_config(_PHASE4_CONFIG_PATH)
    output_dir = PROJECT_ROOT / config["output"]["dir"]
    raw_text = (output_dir / "metrics.json").read_text()
    parsed = json.loads(raw_text)
    assert parsed["test_partition_touched"] is False
    for tier in parsed["tiers"].values():
        assert "test_evaluation" not in tier.get("logistic_regression", {})
        assert "test_evaluation" not in tier.get("xgboost", {})


# Tier-membership determinism on a full REPEATED real-data run
# (docs/plan.md Phase 4's required test) is deliberately NOT an automated
# test here: re-running this pipeline a second time roughly doubles an
# already ~1hr+ real-data test run. It is instead a one-time manual
# verification step (see docs/features.md's Verification section), while
# tests/test_features_tiers.py::test_build_tiers_is_deterministic_regardless_of_row_order
# and the tampered-hash test cover determinism at the unit level on every run.
