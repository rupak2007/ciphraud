"""Phase 10 deployment-build tests (`src/fhe/deploy.py`): the pure helpers run everywhere; persistence, the reproduction proof and the
fresh-load verification use tiny synthetic deployments (`tests/conftest.py`) and need Concrete-ML (WSL2 only).
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.config import PROJECT_ROOT
from src.fhe import deploy
from tests.conftest import needs_concrete

# ---- pure helpers -----------------------------------------------------------------------------------------------------


def test_request_positions_put_the_fixed_row_first_then_other_t1_rows_then_repeats():
    assert deploy.request_positions(32148, [32148, 13560], True, 5) == [32148, 13560] + [32148] * 5
    assert deploy.request_positions(32148, [32148, 13560], False, 0) == [32148]  # XGBoost: exactly one request
    assert deploy.request_positions(7, [3, 7, 9], True, 0) == [7, 3, 9]  # the fixed row is never duplicated by the T1 list
    assert deploy.request_positions(7, [7], True, 2) == [7, 7, 7]


def test_config_lookups_fail_loudly_with_the_available_choices():
    config = {"models": [{"id": "a"}, {"id": "b"}]}
    assert deploy.model_config(config, "b")["id"] == "b"
    with pytest.raises(deploy.DeployError, match=r"no model 'c'.*\['a', 'b'\]"):
        deploy.model_config(config, "c")
    with pytest.raises(deploy.DeployError, match="no configuration labeled"):
        deploy.grid_entry({"configurations": [{"label": "x"}]}, "y")


def test_paths_keep_server_and_client_artifacts_in_separate_directories():
    config = {"output": {"deploy_dir": "data/phase10/deploy", "results_dir": "results/phase10_client_server"}}
    p = deploy.paths_for(config, "m")
    assert p["server_dir"].parent == p["client_dir"].parent == p["deploy_root"] and p["server_dir"] != p["client_dir"]
    assert p["manifest"].parent == p["reference"].parent and "data" not in p["manifest"].relative_to(PROJECT_ROOT).parts[:1]  # identity is committed under results/, artifacts stay under gitignored data/


def test_sha256_helpers_agree_with_hashlib(tmp_path):
    data = b"ciphraud" * 200_000
    path = tmp_path / "blob"
    path.write_bytes(data)
    assert deploy.sha256_path(path) == deploy.sha256_bytes(data) == hashlib.sha256(data).hexdigest()


def test_code_file_hashes_cover_the_phase_10_sources_and_are_stable():
    hashes = deploy.code_file_hashes()
    for expected in ("src/fhe/deploy.py", "src/benchmark/memwatch.py", "src/benchmark/phase10_roundtrip.py", "src/server/app.py", "src/client/client.py", "configs/phase10/client_server.yaml"):
        assert expected in hashes, expected
    assert all(re.fullmatch(r"[0-9a-f]{64}", h["sha256"]) and (h["git_blob_sha1"] is None or re.fullmatch(r"[0-9a-f]{40}", h["git_blob_sha1"])) for h in hashes.values())
    assert hashes == deploy.code_file_hashes()


def test_recorded_ints_reads_the_source_phases_checkpoints_and_rejects_a_contradiction(tmp_path):
    (tmp_path / "latency").mkdir()
    (tmp_path / "t1_correctness").mkdir()
    np.save(tmp_path / "latency" / "trial_000.npy", np.array([[5, -6]]))
    np.save(tmp_path / "t1_correctness" / "row_0000.npy", np.array([5, -6]))
    np.save(tmp_path / "t1_correctness" / "row_0001.npy", np.array([[[1, 2]]]))
    source = {"id": "m", "recorded_dir": tmp_path}
    got = deploy.recorded_ints(source, 100, [100, 200])
    assert set(got) == {100, 200} and got[100].tolist() == [5, -6] and got[200].tolist() == [1, 2]
    np.save(tmp_path / "t1_correctness" / "row_0000.npy", np.array([9, 9]))
    with pytest.raises(deploy.DeployError, match="disagree"):
        deploy.recorded_ints(source, 100, [100, 200])
    assert deploy.recorded_ints({"id": "m", "recorded_dir": tmp_path / "missing"}, 1, [1]) == {}  # LR recorded only its latency row; absence is not an error


def test_security_level_is_read_from_the_circuit_configuration_when_available():
    circuit = SimpleNamespace(configuration=SimpleNamespace(security_level="SecurityLevel.SECURITY_128_BITS"))
    assert deploy.security_level(circuit) == "SecurityLevel.SECURITY_128_BITS"
    assert deploy.security_level(SimpleNamespace()) is None


def test_deploy_never_retrains_and_imports_no_concrete_at_import_time():
    text = (PROJECT_ROOT / "src" / "fhe" / "deploy.py").read_text(encoding="utf-8")
    for forbidden in ("train_qat_checkpointed(", "train_qat_mlp(", "train_float_twin", ".fit("):
        assert forbidden not in text, forbidden  # the MLP is reloaded from its Phase 9 checkpoint; nothing here can retrain
    code = "import sys; import src.fhe.deploy, src.fhe.deploy as d; assert not [m for m in sys.modules if m.startswith('concrete')], 'concrete imported eagerly'"
    done = subprocess.run([sys.executable, "-c", code], cwd=PROJECT_ROOT, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr[-400:]


# ---- persistence / reproduction proof / fresh-load verification (Concrete-ML) ------------------------------------------------


@needs_concrete
def test_persist_circuit_splits_server_and_client_artifacts_and_reports_their_hashes(toy_env):
    d = toy_env.deployments["lr"]
    assert not (d.paths["deploy_root"] / "_stage").exists()
    assert sorted(p.name for p in d.paths["server_dir"].iterdir()) == ["server.zip"]
    assert sorted(p.name for p in d.paths["client_dir"].iterdir()) == ["client.zip", "preprocessing.json"]
    assert d.manifest["artifacts"]["server_zip"]["sha256"] == deploy.sha256_path(d.paths["server_dir"] / "server.zip")
    assert d.manifest["artifacts"]["client_zip"]["sha256"] == deploy.sha256_path(d.paths["client_dir"] / "client.zip")
    assert d.manifest["artifacts"]["via_mlir"] is False


def _toy_source(toy_env, tmp_path, *, stats_delta=0, pr_auc_delta=0.0, recorded=None):
    """A `source` dict for the toy LR, plus the fake 'source phase' metrics file the proof is checked against."""
    from src.fhe.compile.linear import circuit_stats
    from src.train.metrics import threshold_independent_metrics

    d = toy_env.deployments["lr"]
    toy = d.toy
    from src.client.preprocess import apply_preprocessing

    X_model = apply_preprocessing(toy.spec, toy.raw)
    stats = circuit_stats(toy.circuit)
    prob = toy.model.predict_proba(X_model, fhe="disable")[:, 1]
    pr_auc = float(threshold_independent_metrics(toy.y, prob)["pr_auc"])
    expected_stats = {**stats, "size_of_secret_keys_bytes": stats["size_of_secret_keys_bytes"] + stats_delta}
    metrics_path = tmp_path / "metrics.json"
    metrics_path.write_text(json.dumps({"circuit_stats": expected_stats, "accuracy": {"quantized_full_metrics": {"pr_auc": pr_auc + pr_auc_delta}}}), encoding="utf-8")
    recorded_dir = tmp_path / "recorded"
    if recorded is not None:
        (recorded_dir / "latency").mkdir(parents=True)
        np.save(recorded_dir / "latency" / "trial_000.npy", np.asarray(recorded))
    return {
        "id": "toy_lr", "kind": "lr", "cml_model": toy.model, "circuit": toy.circuit, "X_val_model": X_model, "y_val": toy.y,
        "circuit_stats": stats, "metrics_path": metrics_path, "recorded_dir": recorded_dir,
    }


@needs_concrete
def test_the_reproduction_proof_passes_for_a_faithful_rebuild_and_names_what_it_checked(toy_env, tmp_path):
    d = toy_env.deployments["lr"]
    source = _toy_source(toy_env, tmp_path, recorded=d.rows[0]["clear"].reshape(1, 1))
    proof, rows = deploy.reproduction_proof(source, [0, 1], 0, [0, 1])
    assert proof["passed"] is True and proof["circuit_stats_match"] is True and proof["validation_pr_auc"]["match"] is True
    assert rows["0"]["clear_equals_simulate"] is True and rows["0"]["clear_equals_recorded"] is True and rows["1"]["recorded_ints"] is None
    assert rows["0"]["clear_ints"] == d.rows[0]["clear"].tolist() and rows["0"]["probability"] == d.rows[0]["probability"]


@needs_concrete
@pytest.mark.parametrize("kwargs", [{"stats_delta": 8}, {"pr_auc_delta": 1e-6}, {"recorded": [[123456789]]}])
def test_the_reproduction_proof_fails_on_any_deviation_from_the_recorded_source_result(toy_env, tmp_path, kwargs):
    source = _toy_source(toy_env, tmp_path, **kwargs)
    proof, _ = deploy.reproduction_proof(source, [0], 0, [0])
    assert proof["passed"] is False


@needs_concrete
def test_verify_persisted_loads_only_the_artifacts_and_round_trips_in_this_process(toy_env):
    result = deploy.verify_persisted(toy_env.config, "toy_lr", run_fhe=True)
    assert result["passed"] is True and result["decrypted_equals_reference"] is True and result["loaded_server_sizes_match_manifest"] is True
    assert json.loads(toy_env.deployments["lr"].paths["verify"].read_text())["passed"] is True


@needs_concrete
def test_verify_persisted_fails_when_the_manifest_hash_no_longer_matches_the_artifact(toy_env):
    d = toy_env.deployments["lr"]
    original = d.paths["manifest"].read_text()
    try:
        manifest = json.loads(original)
        manifest["artifacts"]["server_zip"]["sha256"] = "0" * 64
        d.paths["manifest"].write_text(json.dumps(manifest), encoding="utf-8")
        result = deploy.verify_persisted(toy_env.config, "toy_lr", run_fhe=False)
        assert result["server_zip_sha256_matches"] is False and result["passed"] is False
    finally:
        d.paths["manifest"].write_text(original, encoding="utf-8")
