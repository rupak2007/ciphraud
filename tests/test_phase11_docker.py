"""Phase 11: container packaging of the inference server.

Static tests always run (no Docker needed). The live test starts the real container and sends one real encrypted LR request
through the existing client; it is skipped unless CIPHRAUD_DOCKER_TEST=1, the Docker daemon is reachable, the image
`ciphraud-server:phase11` exists, and the LR deployment + Concrete-ML are available locally (see docs/docker.md).
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
from importlib.util import find_spec
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
IMAGE = "ciphraud-server:phase11"
MODEL = "lr_top20_bits16"
# From WSL the usable CLI is the Windows docker.exe (interop); on a plain Linux/Windows host it is `docker`.
DOCKER = "docker.exe" if shutil.which("docker.exe") else "docker"


def _text(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8")


# ---- static: the image never contains keys, results, data or client code -----------------------------------------

def test_dockerfile_copies_only_server_code_and_the_phase10_config():
    copies = re.findall(r"^COPY\s+(.+)$", _text("Dockerfile"), flags=re.M)
    sources = {line.split()[0] for line in copies}
    assert sources == {"requirements-server-cpu.txt", "src", "configs/phase10"}, sources
    assert not any(s.startswith(("data", "results", "models")) for s in sources)


def test_dockerignore_excludes_everything_but_the_whitelisted_paths_and_any_key_material():
    lines = [l.strip() for l in _text(".dockerignore").splitlines() if l.strip() and not l.startswith("#")]
    assert lines[0] == "*"  # default-deny
    allowed = {l[1:] for l in lines if l.startswith("!")}
    assert allowed == {"Dockerfile", "requirements-server-cpu.txt", "src/", "configs/phase10/"}
    assert any("keyset" in l for l in lines) and any(l.endswith(".key") for l in lines)


def test_dockerfile_runs_as_non_root_pins_python_and_installs_the_pinned_requirements():
    d = _text("Dockerfile")
    assert re.search(r"^FROM python:3\.12-", d, flags=re.M)  # Concrete-ML 1.9.0 is validated on Python 3.12.3
    assert "pip install -r requirements-server-cpu.txt" in d and "--upgrade" not in d
    assert re.search(r"^USER ciphraud$", d, flags=re.M)
    assert "HEALTHCHECK" in d and "/v1/models" in d
    assert '"--host", "0.0.0.0"' in d  # reachable through the published port; the host side must publish on loopback only
    assert "client.zip" not in d and "secret" not in d.lower().replace("no secret keys", "")


def _pins(name: str) -> list[str]:
    return [l.strip() for l in _text(name).splitlines() if l.strip() and not l.startswith(("#", "--"))]


def test_requirements_pin_every_package_exactly():
    for name in ("requirements-fhe.txt", "requirements-server-cpu.txt"):
        reqs = _pins(name)
        assert reqs and all("==" in r for r in reqs), (name, [r for r in reqs if "==" not in r])
        assert any(r.startswith("concrete-ml==1.9.0") for r in reqs) and any(r.startswith("concrete-python==2.10.0") for r in reqs)


def test_server_cpu_requirements_are_the_validated_freeze_minus_cuda_wheels_with_cpu_torch_only():
    full, cpu = _pins("requirements-fhe.txt"), _pins("requirements-server-cpu.txt")
    removed = [r for r in full if r not in cpu]
    added = [r for r in cpu if r not in full]
    assert all(r.startswith("nvidia-") for r in removed if r != "torch==2.3.1"), removed
    assert set(r for r in removed if not r.startswith("nvidia-")) == {"torch==2.3.1"}
    assert added == ["torch==2.3.1+cpu"], added  # nothing else differs: every other version is the recorded one
    assert not any(r.startswith("nvidia-") for r in cpu)
    assert "--extra-index-url https://download.pytorch.org/whl/cpu" in _text("requirements-server-cpu.txt")


def test_server_package_has_no_client_or_decrypt_code_path_for_the_container():
    for path in (ROOT / "src" / "server").glob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "FHEModelClient" not in text and "client.zip" not in text.replace("client_zip_sha256", ""), path


# ---- live: real container, one real encrypted request ------------------------------------------------------------

def _docker_ready() -> str | None:
    import os

    if os.environ.get("CIPHRAUD_DOCKER_TEST") != "1":
        return "set CIPHRAUD_DOCKER_TEST=1 to run the live container test"
    if shutil.which(DOCKER) is None:
        return "docker CLI not found"
    if subprocess.run([DOCKER, "info"], capture_output=True).returncode != 0:
        return "docker daemon not reachable"
    if subprocess.run([DOCKER, "image", "inspect", IMAGE], capture_output=True).returncode != 0:
        return f"image {IMAGE} not built (docker build -t {IMAGE} .)"
    if find_spec("concrete.ml") is None:
        return "Concrete-ML not importable (run this test inside the WSL FHE venv)"
    if not (ROOT / "data" / "phase10" / "deploy" / MODEL / "server" / "server.zip").exists():
        return f"{MODEL} is not deployed locally"
    return None


@pytest.fixture()
def container():
    reason = _docker_ready()
    if reason:
        pytest.skip(reason)
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    name = f"ciphraud-test-{port}"
    server_dir = ROOT / "data" / "phase10" / "deploy" / MODEL / "server"
    manifest_dir = ROOT / "results" / "phase10_client_server" / "deploy" / MODEL
    def host_path(path: Path) -> str:
        # when the docker CLI is the Windows binary reached from WSL (path under /mnt/), the engine needs Windows-style paths
        if (shutil.which(DOCKER) or "").startswith("/mnt/") and shutil.which("wslpath"):
            return subprocess.run(["wslpath", "-w", str(path)], capture_output=True, text=True, check=True).stdout.strip()
        return str(path)

    cmd = [
        DOCKER, "run", "-d", "--name", name, "--read-only", "--tmpfs", "/tmp:rw,exec,nosuid,size=512m", "--memory", "3g",  # exec: Concrete unpacks the circuit's sharedlib.so into /tmp and dlopens it (a default noexec tmpfs fails)
       
        "-p", f"127.0.0.1:{port}:8710",  # loopback only
        "-v", f"{host_path(server_dir)}:/app/data/phase10/deploy/{MODEL}/server:ro",  # server.zip only: the client artifacts are never mounted
        "-v", f"{host_path(manifest_dir)}:/app/results/phase10_client_server/deploy/{MODEL}:ro",
        IMAGE, "--models", MODEL,
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    try:
        yield name, port
    finally:
        logs = subprocess.run([DOCKER, "logs", name], capture_output=True, text=True)  # kept for diagnosis; server logs hold no key material
        log_dir = ROOT / "data" / "phase11" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        (log_dir / f"container_{name}.log").write_text((logs.stdout or "") + "\n--stderr--\n" + (logs.stderr or ""), encoding="utf-8")
        subprocess.run([DOCKER, "rm", "-f", name], capture_output=True)


def _wait_healthy(name: str, timeout: float = 240.0) -> None:
    end = time.time() + timeout
    while time.time() < end:
        out = subprocess.run([DOCKER, "inspect", "-f", "{{.State.Health.Status}}|{{.State.Running}}", name], capture_output=True, text=True).stdout.strip()
        if out.startswith("healthy"):
            return
        if out.endswith("false"):
            raise AssertionError("container exited: " + subprocess.run([DOCKER, "logs", name], capture_output=True, text=True).stderr[-2000:])
        time.sleep(3)
    raise AssertionError("container never became healthy")


def test_container_serves_models_rejects_bad_requests_and_round_trips_one_encrypted_lr_request(container):
    import httpx

    from src.benchmark.phase10_roundtrip import compare_integers, read_raw_rows
    from src.client.client import CiphraudClient
    from src.client.http import ApiClient

    name, port = container
    _wait_healthy(name)
    base = f"http://127.0.0.1:{port}"
    manifest = json.loads((ROOT / "results" / "phase10_client_server" / "deploy" / MODEL / "manifest.json").read_text())
    reference = json.loads((ROOT / "results" / "phase10_client_server" / "deploy" / MODEL / "reference.json").read_text())

    # readiness / discovery and API error handling
    models = httpx.get(f"{base}/v1/models", timeout=30).json()
    assert MODEL in json.dumps(models)
    import hashlib

    hdr = {"X-Ciphraud-Client-Zip-Sha256": hashlib.sha256((ROOT / "data" / "phase10" / "deploy" / MODEL / "client" / "client.zip").read_bytes()).hexdigest()}
    octet = {"content-type": "application/octet-stream", **hdr}
    # with the client-artifact header present, a JSON body is refused as 415 (the server never accepts plaintext/JSON payloads)
    assert httpx.post(f"{base}/v1/keys", params={"model": MODEL}, json={"x": 1}, headers=hdr, timeout=30).status_code == 415
    # documented validation order (src/server/app.py): unknown_model 404 -> client_artifact_mismatch 422 (header) -> key_set_resident 409 -> body checks
    r = httpx.post(f"{base}/v1/keys", params={"model": MODEL}, json={"x": 1}, timeout=30)  # no header: refused at the header check, before the media type
    assert (r.status_code, r.json()["detail"]["error"]) == (422, "client_artifact_mismatch")
    r = httpx.post(f"{base}/v1/keys", params={"model": "no_such_model"}, content=b"x", headers=octet, timeout=30)  # model lookup comes first
    assert (r.status_code, r.json()["detail"]["error"]) == (404, "unknown_model")
    r = httpx.post(f"{base}/v1/infer", params={"model": MODEL, "key_id": "0" * 64}, content=b"x", headers={"content-type": "application/octet-stream"}, timeout=30)  # no key resident
    assert (r.status_code, r.json()["detail"]["error"]) == (404, "unknown_key_id")

    # one real encrypted request through the same client code as Phase 10
    client = CiphraudClient(ROOT / "data" / "phase10" / "deploy" / MODEL / "client")
    client.generate_keys()
    api = ApiClient(base)
    try:
        eval_keys = client.evaluation_keys()
        reg = api.register_keys(MODEL, eval_keys["bytes"], client.client_zip_sha256)
        key_id = reg["json"]["key_id"]
        position = int(reference["fixed_position"])
        raw = read_raw_rows(manifest, [position])[position]
        request = client.prepare_request(raw)
        reply = api.infer(MODEL, key_id, request["ciphertext"])
        result = client.finish(reply["content"])
        check = compare_integers(result["integers"], reference["rows"][str(position)])
        assert check["exact_match"] is True, check
        api.delete_keys(key_id)
    finally:
        api.close()

    # nothing secret reached the server side: no keyset file in the container, logs free of key material
    logs = subprocess.run([DOCKER, "logs", name], capture_output=True, text=True)
    assert "secret" not in (logs.stdout + logs.stderr).lower()
    assert np.isfinite(result["probability"])


def test_dockerfile_builds_the_source_only_onnxoptimizer_pin_with_a_toolchain_that_is_purged_in_the_same_layer():
    d = _text("Dockerfile")
    step = d[d.index("RUN apt-get update && apt-get install -y --no-install-recommends build-essential cmake"):]
    step = step[: step.index("rm -rf /var/lib/apt/lists/*") + len("rm -rf /var/lib/apt/lists/*")]
    assert "pip install -r requirements-server-cpu.txt" in step and "apt-get purge -y --auto-remove build-essential cmake" in step
    assert "onnxoptimizer==0.3.13" in _text("requirements-server-cpu.txt")  # exact version required by concrete-ml 1.9.0; not bumped
    assert "libgomp1" in d.split("build-essential")[0]  # the OpenMP runtime is installed (and kept) before the purge


# ======================================================================================================================
# Added 2026-10-12: MLP container validation + cheap failure-handling tests (docs/docker.md "Container validation 2").
# One module-scoped container serves BOTH models (lr_top20_bits16 and mlp_top20_bits3). The LR test above is unchanged.
# Status codes asserted below are the documented contract of src/server/app.py (exact, never "one of several").
# ======================================================================================================================

MLP = "mlp_top20_bits3"
CLIENT_HDR = "X-Ciphraud-Client-Zip-Sha256"
OCTET = "application/octet-stream"
RES_DIR = ROOT / "data" / "phase11" / "logs"


def _host_path(path: Path) -> str:
    # when the docker CLI is the Windows binary reached from WSL (path under /mnt/), the engine needs Windows-style paths
    if (shutil.which(DOCKER) or "").startswith("/mnt/") and shutil.which("wslpath"):
        return subprocess.run(["wslpath", "-w", str(path)], capture_output=True, text=True, check=True, timeout=30).stdout.strip()
    return str(path)


def _two_model_ready() -> str | None:
    reason = _docker_ready()
    if reason:
        return reason
    if not (ROOT / "data" / "phase10" / "deploy" / MLP / "server" / "server.zip").exists():
        return f"{MLP} is not deployed locally"
    return None


@pytest.fixture(scope="module")
def two_model_container():
    reason = _two_model_ready()
    if reason:
        pytest.skip(reason)
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    name = f"ciphraud-test-multi-{port}"
    cmd = [DOCKER, "run", "-d", "--name", name, "--read-only", "--tmpfs", "/tmp:rw,exec,nosuid,size=512m", "--memory", "3g", "-p", f"127.0.0.1:{port}:8710"]
    for model in (MODEL, MLP):
        cmd += ["-v", f"{_host_path(ROOT / 'data' / 'phase10' / 'deploy' / model / 'server')}:/app/data/phase10/deploy/{model}/server:ro"]  # server.zip only
        cmd += ["-v", f"{_host_path(ROOT / 'results' / 'phase10_client_server' / 'deploy' / model)}:/app/results/phase10_client_server/deploy/{model}:ro"]
    cmd += [IMAGE, "--models", f"{MODEL},{MLP}"]
    started = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    assert started.returncode == 0, f"docker run failed: {started.stderr[-600:]}"
    try:
        _wait_healthy(name, timeout=240)
        yield name, port
    finally:
        logs = subprocess.run([DOCKER, "logs", name], capture_output=True, text=True, timeout=120)
        RES_DIR.mkdir(parents=True, exist_ok=True)
        (RES_DIR / f"container_{name}.log").write_text((logs.stdout or "") + "\n--stderr--\n" + (logs.stderr or ""), encoding="utf-8")
        subprocess.run([DOCKER, "rm", "-f", name], capture_output=True, timeout=120)


def _manifest(model: str) -> dict:
    return json.loads((ROOT / "results" / "phase10_client_server" / "deploy" / model / "manifest.json").read_text())


def _reference(model: str) -> dict:
    return json.loads((ROOT / "results" / "phase10_client_server" / "deploy" / model / "reference.json").read_text())


def _client(model: str):
    from src.client.client import CiphraudClient

    return CiphraudClient(ROOT / "data" / "phase10" / "deploy" / model / "client")


def _err(response, status: int, code: str) -> None:
    assert response.status_code == status, (response.status_code, response.text[:300])
    assert response.json()["detail"]["error"] == code, response.text[:300]


def _round_trip(base: str, model: str, client, key_id: str, position: int, infer_timeout: float, api=None):
    """One encrypted request through the real client code; returns (probability, exact-match record)."""
    import httpx

    from src.benchmark.phase10_roundtrip import compare_integers, read_raw_rows
    from src.client.http import ApiClient

    own = api is None
    if own:
        api = ApiClient(base)
        api._http.timeout = httpx.Timeout(infer_timeout)
    try:
        request = client.prepare_request(read_raw_rows(_manifest(model), [position])[position])
        reply = api.infer(model, key_id, request["ciphertext"])
        result = client.finish(reply["content"])
    finally:
        if own:
            api.close()
    check = compare_integers(result["integers"], _reference(model)["rows"][str(position)])
    return result, check, request, reply


def test_container_lists_both_models_and_carries_the_mlp_t3_disclosure(two_model_container):
    import httpx

    _, port = two_model_container
    models = {m["id"]: m for m in httpx.get(f"http://127.0.0.1:{port}/v1/models", timeout=30).json()["models"]}
    assert set(models) == {MODEL, MLP}
    assert models[MODEL]["model_type"] == "lr" and models[MLP]["model_type"] == "mlp"
    assert "failed the Phase 9 quantization-accuracy gate" in models[MLP]["t3_disclosure"]  # the MLP is never presented as a validated fraud model


def test_malformed_wrong_and_incompatible_requests_get_the_documented_errors_and_the_server_survives(two_model_container):
    import httpx

    name, port = two_model_container
    base = f"http://127.0.0.1:{port}"
    lr_hdr = {CLIENT_HDR: _manifest(MODEL)["artifacts"]["client_zip"]["sha256"]}
    post = lambda path, **kw: httpx.post(base + path, timeout=30, **kw)  # noqa: E731

    # ---- /v1/keys: lookup and header protections (no encrypted inference involved)
    _err(post("/v1/keys", params={"model": "no_such_model"}, content=b"x", headers={"content-type": OCTET, **lr_hdr}), 404, "unknown_model")
    _err(post("/v1/keys", params={"model": MODEL}, content=b"x", headers={"content-type": OCTET}), 422, "client_artifact_mismatch")  # header missing
    _err(post("/v1/keys", params={"model": MODEL}, content=b"x", headers={"content-type": OCTET, CLIENT_HDR: "0" * 64}), 422, "client_artifact_mismatch")  # wrong hash
    lr = _client(MODEL)
    lr.generate_keys()
    eval_keys = lr.evaluation_keys()["bytes"]
    _err(post("/v1/keys", params={"model": MLP}, content=eval_keys, headers={"content-type": OCTET, **lr_hdr}), 422, "client_artifact_mismatch")  # LR keys offered for the MLP circuit
    # ---- /v1/keys: body protections (header correct)
    _err(post("/v1/keys", params={"model": MODEL}, json={"x": 1}, headers=lr_hdr), 415, "unsupported_media_type")
    _err(post("/v1/keys", params={"model": MODEL}, content=b"this is not an evaluation key set", headers={"content-type": OCTET, **lr_hdr}), 400, "bad_evaluation_keys")
    _err(post("/v1/keys", params={"model": MODEL}, content=eval_keys[: len(eval_keys) // 2], headers={"content-type": OCTET, **lr_hdr}), 400, "bad_evaluation_keys")  # truncated real keys
    # ---- register the real keys; a second set must be refused
    registered = post("/v1/keys", params={"model": MODEL}, content=eval_keys, headers={"content-type": OCTET, **lr_hdr})
    assert registered.status_code == 200, registered.text[:300]
    key_id = registered.json()["key_id"]
    _err(post("/v1/keys", params={"model": MODEL}, content=eval_keys, headers={"content-type": OCTET, **lr_hdr}), 409, "key_set_resident")
    # ---- /v1/infer protections
    _err(post("/v1/infer", params={"model": "no_such_model", "key_id": key_id}, content=b"x", headers={"content-type": OCTET}), 404, "unknown_model")
    _err(post("/v1/infer", params={"model": MODEL, "key_id": "0" * 16}, content=b"x", headers={"content-type": OCTET}), 404, "unknown_key_id")
    _err(post("/v1/infer", params={"model": MLP, "key_id": key_id}, content=b"x", headers={"content-type": OCTET}), 422, "key_model_mismatch")  # LR keys used on the MLP
    _err(post("/v1/infer", params={"model": MODEL, "key_id": key_id}, json={"features": [1.0]}), 415, "unsupported_media_type")  # plaintext-like JSON is never accepted
    _err(post("/v1/infer", params={"model": MODEL, "key_id": key_id}, content=b"not a ciphertext", headers={"content-type": OCTET}), 400, "bad_ciphertext")
    _err(httpx.post(base + "/v1/infer", params={"model": MODEL, "key_id": key_id}, content=b"\0" * (64 * 1024 * 1024 + 1), headers={"content-type": OCTET}, timeout=120), 413, "payload_too_large")
    # ---- the server survived all of that: a valid encrypted request still works and is exact
    result, check, _, _ = _round_trip(base, MODEL, lr, key_id, int(_reference(MODEL)["fixed_position"]), 120)
    assert check["exact_match"] is True, check
    # ---- key deletion protections
    _err(httpx.delete(base + "/v1/keys", params={"key_id": "0" * 16}, timeout=30), 404, "unknown_key_id")
    assert httpx.delete(base + "/v1/keys", params={"key_id": key_id}, timeout=30).status_code == 204
    _err(httpx.delete(base + "/v1/keys", params={"key_id": key_id}, timeout=30), 404, "unknown_key_id")  # already gone
    _err(post("/v1/infer", params={"model": MODEL, "key_id": key_id}, content=b"x", headers={"content-type": OCTET}), 404, "unknown_key_id")  # a deleted key set cannot be used
    assert subprocess.run([DOCKER, "inspect", "-f", "{{.State.Running}}|{{.State.OOMKilled}}", name], capture_output=True, text=True, timeout=60).stdout.strip() == "true|false"


def test_mlp_encrypted_round_trip_in_the_container_matches_the_recorded_clear_quantized_integers(two_model_container):
    import hashlib
    import httpx

    from src.benchmark.phase10_roundtrip import WireAudit
    from src.client.http import ApiClient

    name, port = two_model_container
    base = f"http://127.0.0.1:{port}"
    reference, manifest = _reference(MLP), _manifest(MLP)
    client = _client(MLP)
    keygen = client.generate_keys()
    keys = client.evaluation_keys()
    audit = WireAudit()
    api = ApiClient(base, request_hook=audit.hook)
    api._http.timeout = httpx.Timeout(900)
    positions = [int(p) for p in reference["t1_positions"]]  # the two recorded rows (one fraud, one legitimate)
    ciphertext_hashes: set[str] = set()
    checks = []
    try:
        reg = api.register_keys(MLP, keys["bytes"], client.client_zip_sha256)
        key_id = reg["json"]["key_id"]
        assert reg["json"]["n_bytes"] == keys["n_bytes"]  # the server holds exactly the serialized evaluation keys
        for position in positions:
            result, check, request, reply = _round_trip(base, MLP, client, key_id, position, 900, api=api)
            ciphertext_hashes.add(hashlib.sha256(request["ciphertext"]).hexdigest())
            checks.append({"position": position, "exact_match": check["exact_match"], **{k: v for k, v in check.items() if k != "exact_match"}, "probability": result["probability"], "server_fhe_run_s": reply["server_timing"].get("fhe_run_s")})
        stats = subprocess.run([DOCKER, "stats", "--no-stream", "--format", "{{.MemUsage}}|{{.MemPerc}}|{{.CPUPerc}}", name], capture_output=True, text=True, timeout=60).stdout.strip()
        deleted = api.delete_keys(key_id)
        assert deleted["seconds"] >= 0
    finally:
        api.close()
    assert all(c["exact_match"] for c in checks), checks  # decrypted == clear-quantized == simulated (== recorded Phase 9 ints where recorded)
    # ---- key ownership: the only request bodies the container ever received are the evaluation keys and input ciphertexts
    classified = audit.classify(keys["sha256"], ciphertext_hashes)
    assert classified["all_bodies_are_evaluation_keys_or_ciphertexts"] is True, classified
    assert classified["classes"].count("evaluation_keys") == 1 and classified["classes"].count("input_ciphertext") == len(positions)
    # ---- deleted key set is unusable; container healthy, not OOM-killed, logs free of key material
    _err(httpx.post(base + "/v1/infer", params={"model": MLP, "key_id": key_id}, content=b"x", headers={"content-type": OCTET}, timeout=30), 404, "unknown_key_id")
    state = subprocess.run([DOCKER, "inspect", "-f", "{{.State.Running}}|{{.State.OOMKilled}}", name], capture_output=True, text=True, timeout=60).stdout.strip()
    assert state == "true|false", state
    logs = subprocess.run([DOCKER, "logs", name], capture_output=True, text=True, timeout=120)
    assert "secret" not in (logs.stdout + logs.stderr).lower()
    RES_DIR.mkdir(parents=True, exist_ok=True)
    (RES_DIR / "mlp_container_result.json").write_text(json.dumps({
        "model": MLP, "t3_disclosure": manifest["t3_disclosure"], "keygen_seconds": keygen["keygen_seconds"], "evaluation_key_bytes": keys["n_bytes"], "rows": checks,
        "wire_classes": classified["classes"], "docker_stats_after_requests": stats, "container_state": state,
    }, indent=2, default=str), encoding="utf-8")


def test_container_restart_recovers_and_forgets_resident_keys(two_model_container):
    """Runs last: restarts the container (keys live only in server memory) and checks a clean recovery."""
    import httpx

    name, port = two_model_container
    base = f"http://127.0.0.1:{port}"
    lr_hdr = {CLIENT_HDR: _manifest(MODEL)["artifacts"]["client_zip"]["sha256"]}
    lr = _client(MODEL)
    lr.generate_keys()
    keys = lr.evaluation_keys()["bytes"]
    first = httpx.post(base + "/v1/keys", params={"model": MODEL}, content=keys, headers={"content-type": OCTET, **lr_hdr}, timeout=30)
    assert first.status_code == 200
    old_key_id = first.json()["key_id"]
    assert subprocess.run([DOCKER, "restart", "-t", "10", name], capture_output=True, text=True, timeout=180).returncode == 0
    _wait_healthy(name, timeout=240)
    assert {m["id"] for m in httpx.get(base + "/v1/models", timeout=30).json()["models"]} == {MODEL, MLP}  # models reload from the read-only mounts
    _err(httpx.post(base + "/v1/infer", params={"model": MODEL, "key_id": old_key_id}, content=b"x", headers={"content-type": OCTET}, timeout=30), 404, "unknown_key_id")  # the resident key set did not survive
    second = httpx.post(base + "/v1/keys", params={"model": MODEL}, content=keys, headers={"content-type": OCTET, **lr_hdr}, timeout=30)
    assert second.status_code == 200, second.text[:300]  # no stale "key_set_resident" after a restart
    _, check, _, _ = _round_trip(base, MODEL, lr, second.json()["key_id"], int(_reference(MODEL)["fixed_position"]), 120)
    assert check["exact_match"] is True, check
    assert httpx.delete(base + "/v1/keys", params={"key_id": second.json()["key_id"]}, timeout=30).status_code == 204
