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


def _text(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8")


# ---- static: the image never contains keys, results, data or client code -----------------------------------------

def test_dockerfile_copies_only_server_code_and_the_phase10_config():
    copies = re.findall(r"^COPY\s+(.+)$", _text("Dockerfile"), flags=re.M)
    sources = {line.split()[0] for line in copies}
    assert sources == {"requirements-fhe.txt", "src", "configs/phase10"}, sources
    assert not any(s.startswith(("data", "results", "models")) for s in sources)


def test_dockerignore_excludes_everything_but_the_whitelisted_paths_and_any_key_material():
    lines = [l.strip() for l in _text(".dockerignore").splitlines() if l.strip() and not l.startswith("#")]
    assert lines[0] == "*"  # default-deny
    allowed = {l[1:] for l in lines if l.startswith("!")}
    assert allowed == {"Dockerfile", "requirements-fhe.txt", "src/", "configs/phase10/"}
    assert any("keyset" in l for l in lines) and any(l.endswith(".key") for l in lines)


def test_dockerfile_runs_as_non_root_pins_python_and_installs_the_pinned_requirements():
    d = _text("Dockerfile")
    assert re.search(r"^FROM python:3\.12-", d, flags=re.M)  # Concrete-ML 1.9.0 is validated on Python 3.12.3
    assert "pip install -r requirements-fhe.txt" in d and "--upgrade" not in d
    assert re.search(r"^USER ciphraud$", d, flags=re.M)
    assert "HEALTHCHECK" in d and "/v1/models" in d
    assert '"--host", "0.0.0.0"' in d  # reachable through the published port; the host side must publish on loopback only
    assert "client.zip" not in d and "secret" not in d.lower().replace("no secret keys", "")


def test_requirements_pin_every_package_exactly():
    reqs = [l.strip() for l in _text("requirements-fhe.txt").splitlines() if l.strip() and not l.startswith("#")]
    assert reqs and all("==" in r for r in reqs), [r for r in reqs if "==" not in r]
    assert any(r.startswith("concrete-ml==1.9.0") for r in reqs) and any(r.startswith("concrete-python==2.10.0") for r in reqs)


def test_server_package_has_no_client_or_decrypt_code_path_for_the_container():
    for path in (ROOT / "src" / "server").glob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "FHEModelClient" not in text and "client.zip" not in text.replace("client_zip_sha256", ""), path


# ---- live: real container, one real encrypted request ------------------------------------------------------------

def _docker_ready() -> str | None:
    import os

    if os.environ.get("CIPHRAUD_DOCKER_TEST") != "1":
        return "set CIPHRAUD_DOCKER_TEST=1 to run the live container test"
    if shutil.which("docker") is None:
        return "docker CLI not found"
    if subprocess.run(["docker", "info"], capture_output=True).returncode != 0:
        return "docker daemon not reachable"
    if subprocess.run(["docker", "image", "inspect", IMAGE], capture_output=True).returncode != 0:
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
    cmd = [
        "docker", "run", "-d", "--name", name, "--read-only", "--tmpfs", "/tmp", "--memory", "3g",
        "-p", f"127.0.0.1:{port}:8710",  # loopback only
        "-v", f"{server_dir}:/app/data/phase10/deploy/{MODEL}/server:ro",  # server.zip only: the client artifacts are never mounted
        "-v", f"{manifest_dir}:/app/results/phase10_client_server/deploy/{MODEL}:ro",
        IMAGE, "--models", MODEL,
    ]
    subprocess.run(cmd, check=True, capture_output=True)
    try:
        yield name, port
    finally:
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)


def _wait_healthy(name: str, timeout: float = 240.0) -> None:
    end = time.time() + timeout
    while time.time() < end:
        out = subprocess.run(["docker", "inspect", "-f", "{{.State.Health.Status}}|{{.State.Running}}", name], capture_output=True, text=True).stdout.strip()
        if out.startswith("healthy"):
            return
        if out.endswith("false"):
            raise AssertionError("container exited: " + subprocess.run(["docker", "logs", name], capture_output=True, text=True).stderr[-2000:])
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
    assert httpx.post(f"{base}/v1/keys", params={"model": MODEL}, json={"x": 1}, timeout=30).status_code == 415  # no JSON, no plaintext route
    assert httpx.post(f"{base}/v1/keys", params={"model": "no_such_model"}, content=b"x", headers={"content-type": "application/octet-stream"}, timeout=30).status_code == 404
    assert httpx.post(f"{base}/v1/infer", params={"model": MODEL, "key_id": "0" * 64}, content=b"x", headers={"content-type": "application/octet-stream"}, timeout=30).status_code in (400, 404, 422)

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
    logs = subprocess.run(["docker", "logs", name], capture_output=True, text=True)
    assert "secret" not in (logs.stdout + logs.stderr).lower()
    assert np.isfinite(result["probability"])
