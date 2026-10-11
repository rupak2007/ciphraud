"""Phase 10 integration tests: the REAL runner (`src/benchmark/phase10_roundtrip.py`) driving REAL `python -m src.server` uvicorn
processes over loopback HTTP, on tiny synthetic deployments (`tests/conftest.py`) -- plus real-artifact runs for the deployed LR
and MLP when those artifacts exist locally. WSL2 only (Concrete-ML + FastAPI). XGBoost's real request (~44 min) is the recorded
experiment, not a pytest.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.config import PROJECT_ROOT
from tests.conftest import needs_concrete

pytest.importorskip("fastapi")
pytest.importorskip("httpx")


def _config_with(toy_env, tmp_path: Path, **guard_overrides) -> tuple[dict, str]:
    """A copy of the toy config (children re-read the config from a FILE, so overrides must be written to one)."""
    config = json.loads(json.dumps(toy_env.config))
    config["memory_guards"].update(guard_overrides)
    path = tmp_path / "client_server.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config, str(path)


@pytest.mark.parametrize("kind", ["lr", "mlp"])
@needs_concrete
def test_inprocess_runner_round_trips_over_a_real_server_process(toy_env, kind):
    from src.benchmark.phase10_roundtrip import run_inprocess

    d = toy_env.deployments[kind]
    metrics = run_inprocess(toy_env.config, str(toy_env.config_path), d.id)
    assert metrics["status"] == "passed", metrics.get("failure")
    requests = metrics["requests"]
    assert len(requests) == 4 and [r["position"] for r in requests] == [0, 1, 0, 0]
    assert all(r["correctness"]["exact_match"] for r in requests) and metrics["all_decisions_match_reference"]
    assert all(r["probability_abs_diff"] == 0 for r in requests)

    # the decomposition is present and never blended into one number
    assert set(requests[0]["client_timings"]) == {"preprocess_seconds", "quantize_seconds", "encrypt_seconds", "serialize_seconds", "deserialize_seconds", "decrypt_seconds", "postprocess_seconds"}
    assert set(requests[0]["server_timings"]) == {"input_deserialize_s", "fhe_run_s", "output_serialize_s", "handler_total_s"}
    assert all(r["transport_and_framework_seconds"] >= 0 and r["server_timings"]["fhe_run_s"] > 0 for r in requests)
    assert {"client_keygen_seconds", "evaluation_keys_bytes", "key_upload_http_seconds", "server_key_deserialize_seconds", "key_delete_seconds"} <= set(metrics["one_time"])
    assert metrics["one_time"]["secret_key_sent"] is False
    assert metrics["summary"]["n_requests"] == 3 and "request_0" in metrics["summary"]  # requests on the fixed row, request 0 reported separately

    # what crossed the wire: only evaluation keys and input ciphertexts
    audit = metrics["wire_audit"]
    assert audit["all_bodies_are_evaluation_keys_or_ciphertexts"] and audit["classes"].count("evaluation_keys") == 1 and audit["classes"].count("input_ciphertext") == 4

    assert metrics["keys_resident_after_delete"] is False and metrics["artifact_compatibility"]["match"] is True
    assert metrics["memory"]["per_process"]["server"]["peak_rss_mb"] > 0 and metrics["memory"]["aborted"] is False
    out = Path(toy_env.config["output"]["results_dir"]) / d.id
    assert json.loads((out / "metrics.json").read_text())["status"] == "passed"
    provenance = json.loads((out / "provenance.json").read_text())
    assert provenance["code_files"]["src/benchmark/phase10_roundtrip.py"]["sha256"] and provenance["deployment_manifest_sha256"]


@needs_concrete
def test_no_request_body_contains_the_secret_keyset_or_plaintext_features(toy_env, tmp_path):
    """Dynamic secret-key/plaintext isolation on the wire (docs/client_server.md): every byte the client sends is scanned."""
    from src.benchmark.phase10_roundtrip import ServerProcess, free_port
    from src.client.client import CiphraudClient
    from src.client.http import ApiClient
    from src.client.preprocess import apply_preprocessing

    d = toy_env.deployments["lr"]
    server = ServerProcess(str(toy_env.config_path), d.id, free_port(), tmp_path / "server.log")
    bodies: list[bytes] = []
    api = None
    try:
        server.wait_ready(120)
        api = ApiClient(server.url, request_hook=lambda request: bodies.append(bytes(request.content)))
        client = CiphraudClient(d.paths["client_dir"])
        client.generate_keys()
        keyset_path = tmp_path / "keyset.bin"
        client.save_keys(keyset_path)
        keyset = keyset_path.read_bytes()
        ek = client.evaluation_keys()
        key_id = api.register_keys(d.id, ek["bytes"], client.client_zip_sha256)["json"]["key_id"]
        raw = d.toy.raw[:1]
        prepared = client.prepare_request(raw)
        api.infer(d.id, key_id, prepared["ciphertext"])
        api.delete_keys(key_id)
    finally:
        if api:
            api.close()
        server.stop()
    plaintext_forms = [raw.tobytes(), apply_preprocessing(d.toy.spec, raw).tobytes(), np.asarray(prepared["q_in"]).tobytes()]
    assert len([b for b in bodies if b]) == 2  # the evaluation keys and one ciphertext, nothing else
    for body in bodies:
        assert keyset not in body
        assert all(form not in body for form in plaintext_forms)


@needs_concrete
def test_staged_flow_round_trips_across_three_processes_and_deletes_the_client_keyset(toy_env):
    from src.benchmark.phase10_roundtrip import run_staged

    d = toy_env.deployments["mlp"]
    metrics = run_staged(toy_env.config, str(toy_env.config_path), d.id)
    assert metrics["status"] == "passed", metrics.get("failure")
    assert set(metrics["stages"]) == {"keys", "serve", "decrypt"} and all(s["status"] == "ok" for s in metrics["stages"].values())
    assert metrics["requests"][0]["correctness"]["exact_match"] and metrics["all_requests_exact_match"]
    assert metrics["wire_audit"]["all_bodies_are_evaluation_keys_or_ciphertexts"]
    assert metrics["client_keyset_file_deleted"] is True
    assert not (Path(toy_env.config["output"]["client_keys_dir"]) / d.id / "keyset.bin").exists()
    assert all(s["parent_watch"]["n_samples"] > 0 and s["parent_watch"]["watchdog_error"] is None for s in metrics["stages"].values())  # every stage was watched from OUTSIDE
    assert metrics["stages"]["keys"]["evaluation_keys_bytes"] > 0 and metrics["stages"]["serve"]["go_no_go"]["go"] is True
    assert json.loads((Path(toy_env.config["output"]["results_dir"]) / d.id / "metrics.json").read_text())["status"] == "passed"


@needs_concrete
def test_preflight_only_registers_keys_and_never_starts_the_fhe_run(toy_env):
    from src.benchmark.phase10_roundtrip import run_staged

    d = toy_env.deployments["mlp"]
    metrics = run_staged(toy_env.config, str(toy_env.config_path), d.id, preflight_only=True)
    assert metrics["status"] == "preflight_passed" and metrics["preflight_only"] is True
    assert set(metrics["stages"]) == {"keys", "serve"} and metrics["stages"]["serve"]["preflight_only"] is True
    assert metrics["stages"]["serve"]["fhe_execution_started"] is False and "sent" not in metrics["stages"]["serve"]
    assert metrics["client_keyset_file_deleted"] is True
    assert (Path(toy_env.config["output"]["results_dir"]) / d.id / "preflight.json").exists()


@needs_concrete
def test_a_failed_go_no_go_stops_before_the_fhe_run_and_is_recorded(toy_env, tmp_path):
    from src.benchmark.phase10_roundtrip import run_staged

    config, path = _config_with(toy_env, tmp_path, go_min_mem_available_mb=1e9)  # impossible: memory can never satisfy it
    metrics = run_staged(config, path, toy_env.deployments["mlp"].id)
    assert metrics["status"] == "preflight_failed_go_no_go" and "decrypt" not in metrics["stages"]
    assert metrics["stages"]["serve"]["go_no_go"]["go"] is False
    failure = metrics["failure"]
    assert failure["fhe_execution_started"] is False and failure["server_started"] is True
    assert failure["secret_key_stayed_local"] is True and failure["secret_key_sent"] is False
    assert metrics["client_keyset_file_deleted"] is True
    assert "requests" not in metrics  # no result is claimed for a run that never happened


@needs_concrete
def test_a_tripped_memory_guard_is_recorded_as_aborted_memory_and_the_key_stays_local(toy_env, tmp_path):
    from src.benchmark.phase10_roundtrip import run_staged

    config, path = _config_with(toy_env, tmp_path, abort_min_mem_available_mb=1e9)  # trips on the very first sample
    metrics = run_staged(config, path, toy_env.deployments["mlp"].id)
    assert metrics["status"] == "aborted_memory"
    assert metrics["failure"]["secret_key_stayed_local"] is True and metrics["failure"]["secret_key_sent"] is False
    assert metrics["stages"]["keys"]["parent_watch"]["aborted"] is True and "MemAvailable" in metrics["stages"]["keys"]["reason"]
    assert "requests" not in metrics


# ---- the real deployed artifacts (skipped when absent, e.g. on a fresh checkout: data/ is gitignored) ----------------------


def _real_setup(model_id: str, tmp_path: Path):
    manifest = PROJECT_ROOT / "results" / "phase10_client_server" / "deploy" / model_id / "manifest.json"
    if not manifest.exists() or not (PROJECT_ROOT / "data" / "phase10" / "deploy" / model_id / "server" / "server.zip").exists():
        pytest.skip(f"{model_id} is not deployed locally (run src.fhe.deploy)")
    config = yaml.safe_load((PROJECT_ROOT / "configs" / "phase10" / "client_server.yaml").read_text())
    tmp_results = tmp_path / "results"
    shutil.copytree(manifest.parent, tmp_results / "deploy" / model_id)
    config["output"].update({"results_dir": str(tmp_results), "logs_dir": str(tmp_path / "logs"), "client_keys_dir": str(tmp_path / "keys")})
    path = tmp_path / "client_server.yaml"
    path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config, str(path)


@pytest.mark.parametrize("model_id", ["lr_top20_bits16", "mlp_top20_bits3"])
@needs_concrete
def test_real_deployed_model_round_trips_and_matches_the_recorded_phase_8_9_integers(model_id, tmp_path):
    from src.benchmark.phase10_roundtrip import run_inprocess

    config, path = _real_setup(model_id, tmp_path)
    metrics = run_inprocess(config, path, model_id)  # writes only under tmp_path
    assert metrics["status"] == "passed", metrics.get("failure")
    assert len(metrics["requests"]) == 7
    fixed = [r for r in metrics["requests"] if r["role"] in ("fixed", "repeat")]
    assert all(r["correctness"]["equals_recorded_source_phase"] is True for r in fixed)  # identical to what Phase 8/9 recorded for row 32,148
    assert metrics["wire_audit"]["all_bodies_are_evaluation_keys_or_ciphertexts"]
    if model_id.startswith("mlp"):
        assert "failed the Phase 9 quantization-accuracy gate" in metrics["t3_disclosure"]


# ---- the serve-stage sender must stay light: it only makes HTTP calls (measured: importing xgb_poc cost ~590 MB, memory the server needs) ----

def test_the_serve_stage_sender_does_not_import_concrete_or_torch_for_memory_measurement():
    import inspect
    import subprocess
    import sys

    pytest.importorskip("resource")  # Linux-only stdlib module, exactly what the helper uses
    from src.benchmark.phase10_roundtrip import stage_serve

    source = inspect.getsource(stage_serve)
    assert "xgb_poc" not in source and "concrete" not in source and "torch" not in source
    code = (
        "import sys\n"
        "import src.benchmark.phase10_roundtrip as m\n"
        "from src.client.http import ApiClient\n"
        "assert m._sender_peak_rss_mb() > 0\n"
        "bad = [k for k in sys.modules if k == 'torch' or k.startswith('torch.') or k == 'concrete' or k.startswith('concrete.')]\n"
        "assert not bad, bad[:5]\n"
    )
    done = subprocess.run([sys.executable, "-c", code], cwd=PROJECT_ROOT, capture_output=True, text=True)
    assert done.returncode == 0, done.stderr[-800:]
