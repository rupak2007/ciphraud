"""Phase 6 XGBoost FHE tests against the REAL per-tier results (`docs/plan.md` Phase 6).

Unlike Phase 5's real-data test, this does not re-run the pipeline in a
fixture: compiling, simulating, and really executing a several-hundred-tree
ensemble takes hours per tier (docs/fhe_xgboost.md). It verifies the
recorded evidence written by `python -m src.fhe.xgb_poc` instead, and
independently recomputes the cheap part (T0 and the handoff hashes).

Requires concrete-ml (WSL2 venv only) and a completed run for the tier;
each tier skips with a clear reason otherwise.
"""

import importlib.util
import json

import numpy as np
import pytest

from src.config import PROJECT_ROOT, load_config
from src.data import provenance as data_provenance

_CONFIG_PATH = "configs/phase6/xgb_poc.yaml"
_CONFIG = load_config(_CONFIG_PATH)
_RESULTS = PROJECT_ROOT / _CONFIG["output"]["dir"]


def _is_concrete_ml_available() -> bool:
    try:
        return importlib.util.find_spec("concrete.ml") is not None
    except ModuleNotFoundError:
        return False


pytestmark = pytest.mark.skipif(
    not _is_concrete_ml_available(),
    reason="Requires concrete-ml (WSL2 venv only, see docs/environment.md).",
)


def _tier_results(tier: str) -> dict:
    metrics_path = _RESULTS / tier / "metrics.json"
    if not metrics_path.exists():
        pytest.skip(f"No Phase 6 results for {tier}: run `python -m src.fhe.xgb_poc --config {_CONFIG_PATH} --tier {tier}`")
    return {
        "metrics": json.loads(metrics_path.read_text()),
        "report": json.loads((_RESULTS / tier / "correctness_report.json").read_text()),
        "provenance": json.loads((_RESULTS / tier / "provenance.json").read_text()),
        "manifest": json.loads((_RESULTS / tier / "handoff_manifest.json").read_text()),
    }


tiers = pytest.mark.parametrize("tier", _CONFIG["tiers"])


@tiers
def test_results_were_produced_by_the_current_config(tier):
    r = _tier_results(tier)
    assert r["provenance"]["config_hash"] == data_provenance.compute_config_hash(_CONFIG)
    assert r["metrics"]["n_bits"] == _CONFIG["n_bits"]
    assert r["metrics"]["test_partition_touched"] is False
    assert r["metrics"]["calibration"]["source"] == "train"


@tiers
def test_handoff_integrity_and_t0_recomputed_independently(tier):
    from src.fhe.compile.tree import load_inference_classifier
    from src.fhe.handoff import load_handoff, sha256_file

    r = _tier_results(tier)
    handoff_dir = PROJECT_ROOT / _CONFIG["output"]["handoff_dir"]
    manifest, arrays = load_handoff(handoff_dir / f"xgb_{tier}.npz", _RESULTS / tier / "handoff_manifest.json")
    booster_path = PROJECT_ROOT / manifest["booster_path"]
    assert sha256_file(booster_path) == manifest["booster_sha256"] == r["metrics"]["booster_sha256"]

    clf = load_inference_classifier(booster_path)
    assert clf.get_booster().num_boosted_rounds() == manifest["n_trees_inference"] == r["metrics"]["n_trees"]
    diff = np.abs(clf.predict_proba(arrays["X_val"])[:, 1] - arrays["reference_val_prob"]).max()
    assert diff <= _CONFIG["tolerances"]["t0_max_abs_diff"]


@tiers
def test_circuit_compiled_with_programmable_bootstraps(tier):
    stats = _tier_results(tier)["metrics"].get("circuit_stats")
    assert stats is not None, "tier did not compile -- see metrics.json 'error'"
    assert stats["programmable_bootstrap_count"] > 0
    assert stats["n_bits_inputs"] == [_CONFIG["n_bits"]]


@tiers
def test_t3_quantization_gate(tier):
    t3 = _tier_results(tier)["report"]["t3"]
    assert t3["decision_agreement"] >= _CONFIG["tolerances"]["t3_min_decision_agreement"]
    assert t3["pr_auc_drop"] <= _CONFIG["tolerances"]["t3_max_pr_auc_drop"]
    assert t3["passed"] is True


@tiers
def test_t2_simulation_integer_outputs_exact(tier):
    t2 = _tier_results(tier)["report"]["t2"]
    assert t2["exact_integer_match"] is True
    assert t2["passed"] is True


@tiers
def test_t1_real_execution_integer_outputs_exact(tier):
    r = _tier_results(tier)
    t1 = r["report"]["t1"]
    assert t1["n_rows"] >= 1
    assert t1["exact_integer_match"] is True
    assert t1["decrypted_vs_disable_exact_integer_match"] is True
    assert t1["passed"] is True


@tiers
def test_tier_passed_every_gate(tier):
    m = _tier_results(tier)["metrics"]
    assert m["status"] == "passed"
    assert m["all_gates_passed"] is True
