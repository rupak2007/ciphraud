"""Phase 10 server tests: the FastAPI service and the model registry, against tiny synthetic deployments (`tests/conftest.py`).

Uses FastAPI's in-process TestClient with the REAL registry, the REAL Concrete-ML server and real ciphertexts -- no mocks of
the cryptography. WSL2 only (Concrete-ML + FastAPI); skipped elsewhere.
"""

from __future__ import annotations

import json
import re
import threading
from pathlib import Path

import numpy as np
import pytest

from tests.conftest import needs_concrete

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

OCTET = {"Content-Type": "application/octet-stream"}
CLIENT_HDR = "X-Ciphraud-Client-Zip-Sha256"
SERVER_SRC = Path(__file__).resolve().parents[1] / "src" / "server"


def hdr(toy_env, kind="lr"):
    """Headers for POST /v1/keys: octet-stream plus the sha256 of the client artifact the keys were generated from."""
    return {**OCTET, CLIENT_HDR: toy_env.deployments[kind].manifest["artifacts"]["client_zip"]["sha256"]}


@pytest.fixture
def make_api(toy_env):
    from src.server.app import create_app
    from src.server.registry import ModelRegistry

    def build(models=("toy_lr", "toy_mlp"), **server_overrides):
        registry = ModelRegistry(toy_env.config, list(models))
        app = create_app(registry, {**toy_env.config["server"], **server_overrides})
        return TestClient(app), app.state.ciphraud, registry

    return build


def _client_material(toy_env, kind):
    from src.client.client import CiphraudClient

    d = toy_env.deployments[kind]
    client = CiphraudClient(d.paths["client_dir"])
    client.generate_keys()
    return d, client, client.evaluation_keys(), client.prepare_request(d.toy.raw[:1])


# ---- discovery --------------------------------------------------------------------------------------------------------


@needs_concrete
def test_models_endpoint_lists_loaded_models_hashes_and_no_resident_key(make_api, toy_env):
    api, _, _ = make_api()
    body = api.get("/v1/models").json()
    assert {m["id"] for m in body["models"]} == {"toy_lr", "toy_mlp"}
    lr = next(m for m in body["models"] if m["id"] == "toy_lr")
    assert lr["server_zip_sha256"] == toy_env.deployments["lr"].manifest["artifacts"]["server_zip"]["sha256"]
    assert lr["client_zip_sha256"] == toy_env.deployments["lr"].manifest["artifacts"]["client_zip"]["sha256"]
    assert {"model_type", "n_bits", "n_features", "versions", "circuit_stats", "load_seconds", "t3_disclosure"} <= set(lr)
    assert body["key_resident"] is None and body["busy"] is False and isinstance(body["pid"], int)


# ---- happy path and key lifecycle -------------------------------------------------------------------------------------


@needs_concrete
def test_register_infer_delete_round_trip_reproduces_the_clear_integers(make_api, toy_env):
    api, state, _ = make_api()
    d, client, ek, prepared = _client_material(toy_env, "lr")
    registered = api.post("/v1/keys", params={"model": "toy_lr"}, content=ek["bytes"], headers=hdr(toy_env))
    assert registered.status_code == 200
    info = registered.json()
    assert info["key_id"] == ek["sha256"] and info["n_bytes"] == ek["n_bytes"] and info["deserialize_seconds"] >= 0
    assert api.get("/v1/models").json()["key_resident"]["key_id"] == ek["sha256"]

    response = api.post("/v1/infer", params={"model": "toy_lr", "key_id": ek["sha256"]}, content=prepared["ciphertext"], headers=OCTET)
    assert response.status_code == 200 and response.headers["content-type"] == "application/octet-stream"
    finished = client.finish(response.content)
    assert np.array_equal(finished["integers"].reshape(-1), d.rows[0]["clear"])

    timing = json.loads(response.headers["x-ciphraud-timing"])
    assert {"input_deserialize_s", "fhe_run_s", "output_serialize_s", "handler_total_s", "server_memory"} <= set(timing)
    assert 0 < timing["fhe_run_s"] <= timing["handler_total_s"]
    assert response.headers["x-ciphraud-server-zip-sha256"] == d.manifest["artifacts"]["server_zip"]["sha256"]

    assert api.delete("/v1/keys", params={"key_id": ek["sha256"]}).status_code == 204
    assert state.resident is None and api.get("/v1/models").json()["key_resident"] is None
    gone = api.post("/v1/infer", params={"model": "toy_lr", "key_id": ek["sha256"]}, content=prepared["ciphertext"], headers=OCTET)
    assert gone.status_code == 404


@needs_concrete
def test_second_key_registration_is_409_until_the_first_is_deleted(make_api, toy_env):
    api, _, _ = make_api()
    _, _, ek, _ = _client_material(toy_env, "lr")
    assert api.post("/v1/keys", params={"model": "toy_lr"}, content=ek["bytes"], headers=hdr(toy_env)).status_code == 200
    second = api.post("/v1/keys", params={"model": "toy_lr"}, content=ek["bytes"], headers=hdr(toy_env))
    assert second.status_code == 409 and second.json()["detail"]["error"] == "key_set_resident"
    assert api.delete("/v1/keys", params={"key_id": ek["sha256"]}).status_code == 204
    assert api.post("/v1/keys", params={"model": "toy_lr"}, content=ek["bytes"], headers=hdr(toy_env)).status_code == 200


# ---- errors: the server answers with a 4xx and keeps serving ----------------------------------------------------------


@needs_concrete
def test_unknown_model_is_404_on_both_endpoints(make_api):
    api, _, _ = make_api()
    assert api.post("/v1/keys", params={"model": "nope"}, content=b"x", headers=OCTET).status_code == 404
    assert api.post("/v1/infer", params={"model": "nope", "key_id": "k"}, content=b"x", headers=OCTET).status_code == 404


@needs_concrete
def test_infer_and_delete_with_an_unknown_key_id_are_404(make_api, toy_env):
    api, _, _ = make_api()
    _, _, ek, prepared = _client_material(toy_env, "lr")
    assert api.post("/v1/infer", params={"model": "toy_lr", "key_id": "0" * 64}, content=prepared["ciphertext"], headers=OCTET).status_code == 404
    api.post("/v1/keys", params={"model": "toy_lr"}, content=ek["bytes"], headers=hdr(toy_env))
    assert api.delete("/v1/keys", params={"key_id": "0" * 64}).status_code == 404
    assert api.delete("/v1/keys", params={"key_id": ek["sha256"]}).status_code == 204


@needs_concrete
def test_plaintext_or_json_payloads_are_rejected_with_415_on_both_endpoints(make_api, toy_env):
    api, _, _ = make_api()
    _, _, ek, _ = _client_material(toy_env, "lr")
    api.post("/v1/keys", params={"model": "toy_lr"}, content=ek["bytes"], headers=hdr(toy_env))
    features = [0.1] * 8
    as_json = api.post("/v1/infer", params={"model": "toy_lr", "key_id": ek["sha256"]}, json={"features": features})
    as_text = api.post("/v1/infer", params={"model": "toy_lr", "key_id": ek["sha256"]}, content=json.dumps(features), headers={"Content-Type": "text/plain"})
    assert as_json.status_code == 415 and as_text.status_code == 415
    assert api.post("/v1/keys", params={"model": "toy_lr"}, json={"anything": 1}, headers={CLIENT_HDR: hdr(toy_env)[CLIENT_HDR]}).status_code in (409, 415)  # 409: a key set is resident; either way, not accepted


@needs_concrete
def test_malformed_ciphertext_is_400_and_the_server_keeps_serving(make_api, toy_env):
    api, _, _ = make_api()
    d, client, ek, prepared = _client_material(toy_env, "lr")
    api.post("/v1/keys", params={"model": "toy_lr"}, content=ek["bytes"], headers=hdr(toy_env))
    params = {"model": "toy_lr", "key_id": ek["sha256"]}
    for bad in (b"not a ciphertext", b"", prepared["ciphertext"][: len(prepared["ciphertext"]) // 2]):
        response = api.post("/v1/infer", params=params, content=bad, headers=OCTET)
        assert response.status_code == 400 and response.json()["detail"]["error"] == "bad_ciphertext"
    ok = api.post("/v1/infer", params=params, content=prepared["ciphertext"], headers=OCTET)
    assert ok.status_code == 200 and np.array_equal(client.finish(ok.content)["integers"].reshape(-1), d.rows[0]["clear"])


@needs_concrete
def test_malformed_evaluation_keys_are_400_and_nothing_becomes_resident(make_api, toy_env):
    api, state, _ = make_api()
    response = api.post("/v1/keys", params={"model": "toy_lr"}, content=b"definitely not keys", headers=hdr(toy_env))
    assert response.status_code == 400 and response.json()["detail"]["error"] == "bad_evaluation_keys"
    assert state.resident is None


@needs_concrete
def test_oversize_bodies_are_413(make_api, toy_env):
    api, state, _ = make_api(max_ciphertext_bytes=16, max_key_upload_bytes=16)
    assert api.post("/v1/keys", params={"model": "toy_lr"}, content=b"x" * 64, headers=hdr(toy_env)).status_code == 413
    assert state.resident is None


@needs_concrete
def test_a_key_set_registered_for_another_model_is_422(make_api, toy_env):
    api, _, _ = make_api()
    _, _, ek, prepared = _client_material(toy_env, "lr")
    api.post("/v1/keys", params={"model": "toy_lr"}, content=ek["bytes"], headers=hdr(toy_env))
    response = api.post("/v1/infer", params={"model": "toy_mlp", "key_id": ek["sha256"]}, content=prepared["ciphertext"], headers=OCTET)
    assert response.status_code == 422 and response.json()["detail"]["error"] == "key_model_mismatch"


@needs_concrete
def test_a_concurrent_request_gets_503_and_get_models_stays_responsive(make_api, toy_env):
    api, state, registry = make_api()
    d, client, ek, prepared = _client_material(toy_env, "lr")
    api.post("/v1/keys", params={"model": "toy_lr"}, content=ek["bytes"], headers=hdr(toy_env))
    entry = registry.get("toy_lr")
    real_run, started, release = entry.server.run, threading.Event(), threading.Event()

    def slow_run(*args, **kwargs):
        started.set()
        release.wait(timeout=60)
        return real_run(*args, **kwargs)

    entry.server.run = slow_run
    params = {"model": "toy_lr", "key_id": ek["sha256"]}
    holder: dict = {}
    worker = threading.Thread(target=lambda: holder.update(response=api.post("/v1/infer", params=params, content=prepared["ciphertext"], headers=OCTET)))
    worker.start()
    try:
        assert started.wait(timeout=30)
        assert api.get("/v1/models").json()["busy"] is True  # responsive while an inference runs
        assert api.post("/v1/infer", params=params, content=prepared["ciphertext"], headers=OCTET).status_code == 503
        assert api.delete("/v1/keys", params={"key_id": ek["sha256"]}).status_code == 409  # cannot drop keys mid-run
    finally:
        release.set()
        worker.join(timeout=60)
    assert holder["response"].status_code == 200
    assert np.array_equal(client.finish(holder["response"].content)["integers"].reshape(-1), d.rows[0]["clear"])


# ---- keys from a different circuit ------------------------------------------------------------------------------------


@needs_concrete
def test_an_lr_circuit_needs_no_evaluation_keys_so_key_registration_is_a_formality_for_it(make_api, toy_env):
    """Pins a measured fact (`docs/client_server.md`): the LR circuit has 0 bootstraps, its 'evaluation keys' are 64 bytes, and
    the server runs it with ANY registered key set; the result still decrypts correctly with the client's own key."""
    api, _, _ = make_api()
    _, _, mlp_keys, _ = _client_material(toy_env, "mlp")
    d, lr_client, lr_keys, lr_prepared = _client_material(toy_env, "lr")
    assert lr_keys["n_bytes"] < 1024 < mlp_keys["n_bytes"]
    assert api.post("/v1/keys", params={"model": "toy_lr"}, content=mlp_keys["bytes"], headers=hdr(toy_env, "lr")).status_code == 200
    response = api.post("/v1/infer", params={"model": "toy_lr", "key_id": mlp_keys["sha256"]}, content=lr_prepared["ciphertext"], headers=OCTET)
    assert response.status_code == 200
    assert np.array_equal(lr_client.finish(response.content)["integers"].reshape(-1), d.rows[0]["clear"])


@needs_concrete
def test_key_registration_is_bound_to_the_client_artifact_of_the_served_circuit(make_api, toy_env):
    """Another circuit's keys (declared with their own artifact hash), a wrong hash, or no hash are all refused with 422 --
    before the body is read and without touching the resident-key state."""
    api, state, _ = make_api(models=("toy_lr", "toy_mlp", "toy_xgboost"))
    _, _, xgb_keys, _ = _client_material(toy_env, "xgboost")
    for headers in (hdr(toy_env, "xgboost"), {**OCTET, CLIENT_HDR: "0" * 64}, OCTET):
        response = api.post("/v1/keys", params={"model": "toy_mlp"}, content=xgb_keys["bytes"], headers=headers)
        assert response.status_code == 422 and response.json()["detail"]["error"] == "client_artifact_mismatch"
        assert state.resident is None
    assert api.post("/v1/keys", params={"model": "toy_xgboost"}, content=xgb_keys["bytes"], headers=hdr(toy_env, "xgboost")).status_code == 200


@needs_concrete
def test_pin_running_a_circuit_with_another_circuits_keys_never_yields_a_correct_result(toy_env, tmp_path):
    """Why the binding above exists. Measured (`docs/client_server.md`): the MLP circuit run with the XGBoost circuit's evaluation
    keys crashes Concrete's native runtime with a SEGFAULT, taking the whole process down. This runs it in a CHILD process and
    pins only what must always hold -- no correct result -- whichever way a future Concrete version fails (crash, error, garbage)."""
    import subprocess
    import sys

    from src.config import PROJECT_ROOT

    mlp, xgb = toy_env.deployments["mlp"], toy_env.deployments["xgboost"]
    np.save(tmp_path / "raw.npy", mlp.toy.raw[:1])
    np.save(tmp_path / "clear.npy", mlp.rows[0]["clear"])
    code = f"""
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, {str(PROJECT_ROOT)!r})
import numpy as np
from concrete import fhe
from concrete.ml.deployment import FHEModelServer
from src.client.client import CiphraudClient

foreign = CiphraudClient({str(xgb.paths["client_dir"])!r}); foreign.generate_keys()
keys = fhe.EvaluationKeys.deserialize(foreign.evaluation_keys()["bytes"])
owner = CiphraudClient({str(mlp.paths["client_dir"])!r}); owner.generate_keys()
request = owner.prepare_request(np.load({str(tmp_path / "raw.npy")!r}))
server = FHEModelServer({str(mlp.paths["server_dir"])!r})
result = server.run(fhe.Value.deserialize(request["ciphertext"]), keys)
ints = owner.finish(result.serialize())["integers"].reshape(-1)
print("CORRECT_RESULT" if np.array_equal(ints, np.load({str(tmp_path / "clear.npy")!r})) else "GARBAGE_RESULT", flush=True)
"""
    # `-u` + an explicit STARTED marker: a process that has run a Concrete circuit can drop buffered stdout at exit, which would make
    # the negative assertion below pass vacuously; the marker proves the child's output is really being captured.
    code = code.replace("import numpy as np\n", "import numpy as np\nprint('STARTED', flush=True)\n", 1)
    done = subprocess.run([sys.executable, "-u", "-c", code], capture_output=True, text=True, timeout=300)
    assert "STARTED" in done.stdout, f"child output not captured (returncode {done.returncode}): {done.stderr[-300:]}"
    assert "CORRECT_RESULT" not in done.stdout, f"foreign keys produced a correct result (returncode {done.returncode})"


# ---- registry ---------------------------------------------------------------------------------------------------------


@needs_concrete
def test_registry_rejects_unknown_ids_missing_deployments_and_a_tampered_server_zip(toy_env, tmp_path):
    from src.server.registry import ModelRegistry, RegistryError

    with pytest.raises(RegistryError, match="unknown model ids"):
        ModelRegistry(toy_env.config, ["nope"])
    empty = {**toy_env.config, "output": {**toy_env.config["output"], "deploy_dir": str(tmp_path / "none")}}
    with pytest.raises(RegistryError, match="no deployment"):
        ModelRegistry(empty, ["toy_lr"])

    import shutil

    d = toy_env.deployments["lr"]
    copy_root = tmp_path / "tampered"
    shutil.copytree(d.paths["server_dir"], copy_root / "toy_lr" / "server")
    with open(copy_root / "toy_lr" / "server" / "server.zip", "ab") as handle:
        handle.write(b"tamper")
    tampered = {**toy_env.config, "output": {**toy_env.config["output"], "deploy_dir": str(copy_root)}}
    with pytest.raises(RegistryError, match="does not match the manifest"):
        ModelRegistry(tampered, ["toy_lr"])


# ---- the server has no decrypt/encrypt/keygen capability (static) -----------------------------------------------------

FORBIDDEN = [
    r"FHEModelClient", r"\bfhe\.Client\b", r"\.decrypt\(", r"deserialize_decrypt", r"\.encrypt\(", r"\bkeygen\b", r"client\.zip",
    r"preprocessing\.json", r"\bKeys\b", r"quantize_input", r"dequantize", r"post_processing",
]


def test_server_sources_contain_no_client_side_cryptography():
    """`docs/client_server.md`: nothing under src/server can generate keys, encrypt, decrypt, quantize or read the client artifacts."""
    sources = sorted(SERVER_SRC.glob("*.py"))
    assert {p.name for p in sources} >= {"app.py", "registry.py", "__main__.py"}
    for path in sources:
        text = path.read_text(encoding="utf-8")
        for pattern in FORBIDDEN:
            assert not re.search(pattern, text), f"{path.name} matches forbidden pattern {pattern!r}"
