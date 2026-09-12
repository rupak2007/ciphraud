"""Phase 5 FHE PoC test against the REAL handoff (`docs/plan.md` Phase 5).

Requires BOTH: Concrete-ML importable (WSL2 FHE venv only --
`docs/environment.md`) AND the Windows-side export having already been run
(`python -m src.fhe.export --config configs/phase5/lr_poc.yaml`), which
produces `data/fhe_handoff/lr_top20.npz` + `results/phase5_fhe_poc/
{handoff_manifest,lr_top20_params}.json`. Skipped with a clear reason if
either is missing, mirroring every other `*_real_data.py` test in this repo.

The full PoC (compile + full-val T3/T2 + the real encrypt/run/decrypt
round trip) runs ONCE per test session (module-scoped fixture).
"""

import importlib.util

import pytest

from src.config import PROJECT_ROOT, load_config

_PHASE5_CONFIG_PATH = "configs/phase5/lr_poc.yaml"


def _is_concrete_ml_available() -> bool:
    try:
        return importlib.util.find_spec("concrete.ml") is not None
    except ModuleNotFoundError:
        return False


def _handoff_available() -> bool:
    config = load_config(_PHASE5_CONFIG_PATH)
    npz_path = PROJECT_ROOT / config["output"]["handoff_npz"]
    manifest_path = PROJECT_ROOT / config["output"]["handoff_manifest"]
    return npz_path.exists() and manifest_path.exists()


concrete_ml_available = _is_concrete_ml_available()
handoff_available = _handoff_available() if concrete_ml_available else False

pytestmark = pytest.mark.skipif(
    not (concrete_ml_available and handoff_available),
    reason=(
        "Requires concrete-ml (WSL2 venv only, see docs/environment.md) AND "
        "a completed Windows-side export (`python -m src.fhe.export --config "
        f"{_PHASE5_CONFIG_PATH}`, producing data/fhe_handoff/ + "
        "results/phase5_fhe_poc/handoff_manifest.json)."
    ),
)


@pytest.fixture(scope="module")
def poc_result():
    # Imported lazily (not at module level): src.fhe.poc imports
    # src.fhe.compile.linear, which imports concrete.ml unconditionally --
    # a module-level import here would crash test COLLECTION on Windows
    # (pytest.mark.skipif only skips execution, not collection/import).
    from src.fhe.poc import run

    return run(_PHASE5_CONFIG_PATH)


def test_poc_runs_end_to_end_on_real_handoff(poc_result):
    assert poc_result["tier"] == "top_20"
    assert poc_result["n_features"] == 20
    assert poc_result["test_partition_touched"] is False


def test_t0_transfer_passes(poc_result):
    assert poc_result["correctness_report"]["t0"]["passed"] is True


def test_t1_execution_round_trip_passes_exactly(poc_result):
    """docs/plan.md Phase 5's literal exit criterion: at least one full
    encrypt->infer->decrypt round trip completes correctly."""
    t1 = poc_result["correctness_report"]["t1"]
    assert t1["passed"] is True
    assert t1["exact_match"] is True
    assert t1["n_rows_executed"] >= 1


def test_t2_simulation_passes_exactly(poc_result):
    assert poc_result["correctness_report"]["t2"]["passed"] is True


def test_t3_quantization_passes(poc_result):
    t3 = poc_result["correctness_report"]["t3"]
    assert t3["passed"] is True
    assert t3["decision_agreement"] >= 0.99
    assert t3["pr_auc_drop"] <= 0.01


def test_all_gates_passed(poc_result):
    assert poc_result["all_gates_passed"] is True


def test_circuit_has_no_programmable_bootstraps(poc_result):
    """A linear model's affine arithmetic needs no PBS -- asserted here as
    a real, measured property of the actual compiled top_20 circuit, not
    just the toy model in tests/test_fhe_lr_poc.py."""
    assert poc_result["circuit_stats"]["programmable_bootstrap_count"] == 0
