"""Phase 10 client/server round-trip runner (`docs/plan.md` Phase 10; `docs/client_server.md`).

WSL2/Linux only. From the project root, after `src.fhe.deploy` has built the model:

    ~/.venvs/fhe-fraud-detection/bin/python -m src.benchmark.phase10_roundtrip --config configs/phase10/client_server.yaml --model lr_top20_bits16
    ~/.venvs/fhe-fraud-detection/bin/python -m src.benchmark.phase10_roundtrip --config configs/phase10/client_server.yaml --model xgboost_top50_bits14 --staged

This proves INTEGRATION, not another benchmark. It spawns the real server process (`python -m src.server`) and drives it over
loopback HTTP: fresh client keys -> upload evaluation keys -> encrypted requests -> decrypt -> compare with the recorded
reference integers -> delete the keys. Every request's timing is decomposed (client preprocess/quantize/encrypt/serialize,
HTTP round trip, server deserialize / FHE run / serialize, client deserialize/decrypt/post-process) and never blended into
one number; `fhe_run_s` is the only figure comparable to the Phase 8/9 FHE latency. Loopback only -- not WAN.

  in-process  (LR, MLP)   one client process holds the keys for all requests.
  --staged    (XGBoost)   keys are ~1.9 GB, which cannot be resident in a client AND a server on the 3.8 GiB VM at once:
                            stage `keys`    keygen, export evaluation keys, encrypt the request, save the keyset; EXIT
                            stage `serve`   a key-less sender uploads the keys to a fresh server; memory go/no-go; ONE FHE run
                            stage `decrypt` a fresh process reloads the keyset and decrypts.
                          The pre-registered memory guards (`memory_guards` in the config) kill only our own processes; a
                          tripped guard, a failed go/no-go or a timeout is recorded as evidence (`status`), never retried
                          with looser limits and never dropped.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np

from src.benchmark.harness import trial_stats
from src.benchmark.memwatch import MemWatch, system_memory_mb
from src.config import PROJECT_ROOT, load_config
from src.data import provenance as data_provenance
from src.fhe.deploy import LIBRARY_NAMES, code_file_hashes, model_config, paths_for, sha256_bytes, sha256_path
from src.logging_setup import get_logger

logger = get_logger(__name__)

CLIENT_TIMING_KEYS = ("preprocess_seconds", "quantize_seconds", "encrypt_seconds", "serialize_seconds", "deserialize_seconds", "decrypt_seconds", "postprocess_seconds")


# ------------------------------------------------------------------------------------------------ helpers


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def _die_with_parent() -> None:
    """Linux PR_SET_PDEATHSIG: the server child is SIGKILLed if this runner dies, so a crash can never leave a multi-GB server orphaned."""
    ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGKILL)


def free_port() -> int:
    import socket

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class ServerProcess:
    """`python -m src.server` as a child process on loopback; its stdout/stderr go to a log file, not a pipe."""

    def __init__(self, config_path: str, model_id: str, port: int, log_path: Path) -> None:
        self.port, self.log_path = port, log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = open(log_path, "wb")
        self.started = time.perf_counter()
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "src.server", "--config", config_path, "--models", model_id, "--port", str(port)],
            cwd=PROJECT_ROOT, stdout=self._log, stderr=subprocess.STDOUT, preexec_fn=_die_with_parent,
        )

    @property
    def pid(self) -> int:
        return self.proc.pid

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def wait_ready(self, timeout: float) -> dict[str, Any]:
        import httpx

        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"the server exited during startup (code {self.proc.returncode}); see {self.log_path}")
            try:
                response = httpx.get(f"{self.url}/v1/models", timeout=5)
                if response.status_code == 200:
                    return {"ready_seconds": time.perf_counter() - self.started, "models": response.json()["models"]}
            except httpx.HTTPError:
                pass
            time.sleep(0.25)
        raise TimeoutError(f"the server was not ready within {timeout:.0f} s; see {self.log_path}")

    def stop(self) -> int | None:
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        self._log.close()
        return self.proc.returncode


def read_raw_rows(manifest: dict[str, Any], positions: list[int]) -> dict[int, np.ndarray]:
    """Raw validation rows from the source handoff (the client's own plaintext input), hash-verified. Never the test partition."""
    npz = PROJECT_ROOT / manifest["source"]["handoff_npz"]
    if sha256_path(npz) != manifest["source"]["handoff_npz_sha256"]:
        raise RuntimeError(f"{npz}: sha256 does not match the deployment manifest")
    X_val = np.load(npz, allow_pickle=False)["X_val"]
    return {int(p): np.asarray(X_val[p : p + 1]) for p in sorted(set(positions))}


def compare_integers(decrypted: np.ndarray, ref_row: dict[str, Any]) -> dict[str, Any]:
    flat = np.asarray(decrypted).reshape(-1).astype(np.int64)
    clear, simulated, recorded = np.asarray(ref_row["clear_ints"]), np.asarray(ref_row["simulate_ints"]), ref_row["recorded_ints"]
    out = {
        "equals_clear_quantized": bool(np.array_equal(flat, clear)), "equals_simulate": bool(np.array_equal(flat, simulated)),
        "equals_recorded_source_phase": None if recorded is None else bool(np.array_equal(flat, np.asarray(recorded))),
    }
    out["exact_match"] = bool(out["equals_clear_quantized"] and out["equals_simulate"] and out["equals_recorded_source_phase"] is not False)
    return out


def request_record(index: int, position: int, role: str, prepared: dict[str, Any], sent: dict[str, Any], finished: dict[str, Any], ref_row: dict[str, Any], threshold: float) -> dict[str, Any]:
    client_timings = {**prepared["timings"], **finished["timings"]}
    server = {k: sent["server_timing"][k] for k in ("input_deserialize_s", "fhe_run_s", "output_serialize_s", "handler_total_s")}
    reference_probability = ref_row["probability"]
    return {
        "index": index, "position": position, "role": role, "y_true": ref_row["y_true"],
        "probability": finished["probability"], "decision": finished["decision"], "threshold": threshold,
        "reference_probability": reference_probability, "probability_abs_diff": abs(finished["probability"] - reference_probability),
        "reference_decision": bool(reference_probability >= threshold), "decrypted_ints": np.asarray(finished["integers"]).reshape(-1).tolist(),
        "correctness": compare_integers(finished["integers"], ref_row), "client_timings": client_timings, "http_roundtrip_seconds": sent["seconds"],
        "server_timings": server, "transport_and_framework_seconds": sent["seconds"] - server["handler_total_s"],
        "bytes": {"request_body": sent["request_bytes"], "response_body": sent["response_bytes"]},
        "sum_of_measured_components_seconds": sum(client_timings.values()) + sent["seconds"],
        "server_memory_after_request": sent["server_timing"].get("server_memory"),
    }


def summarize_requests(records: list[dict[str, Any]], fixed_position: int) -> dict[str, Any]:
    """Per-component mean/std over the requests on the fixed row; request 0 (which pays one-time warm-up costs) is also reported separately."""
    fixed = [r for r in records if r["position"] == fixed_position]
    if not fixed:
        return {}
    series: dict[str, list[float]] = {"http_roundtrip_seconds": [r["http_roundtrip_seconds"] for r in fixed], "transport_and_framework_seconds": [r["transport_and_framework_seconds"] for r in fixed]}
    for key in CLIENT_TIMING_KEYS:
        series[f"client_{key}"] = [r["client_timings"][key] for r in fixed if key in r["client_timings"]]
    for key in ("input_deserialize_s", "fhe_run_s", "output_serialize_s", "handler_total_s"):
        series[f"server_{key}"] = [r["server_timings"][key] for r in fixed]
    return {
        "position": fixed_position, "n_requests": len(fixed),
        "note": "repeated requests on ONE fixed row (an integration check, not independent samples); request 0 is reported separately because it pays one-time warm-up costs",
        "stats": {name: trial_stats(vals) for name, vals in series.items() if vals},
        "request_0": {name: vals[0] for name, vals in series.items() if vals},
    }


def build_provenance(config: dict[str, Any], config_path: str, manifest_path: Path) -> dict[str, Any]:
    provenance = data_provenance.build_provenance(config=config, config_path=Path(config_path), seed=config["seed"], raw_file_digests={}, library_names=LIBRARY_NAMES)
    provenance["code_files"] = code_file_hashes()
    provenance["deployment_manifest_sha256"] = sha256_path(manifest_path)
    return provenance


class WireAudit:
    """Records what the client actually put on the wire (method, path, size, sha256) so the secret-key/plaintext claim can be
    checked from evidence: every request body must be exactly the evaluation keys or an input ciphertext."""

    def __init__(self) -> None:
        self.entries: list[dict[str, Any]] = []

    def hook(self, request: Any) -> None:
        body = request.content if request.method in ("POST", "PUT") else b""
        self.entries.append({"method": request.method, "path": request.url.path, "n_bytes": len(body), "sha256": sha256_bytes(body) if body else None})

    def classify(self, evaluation_keys_sha256: str, ciphertext_sha256: set[str]) -> dict[str, Any]:
        classes = []
        for entry in self.entries:
            if entry["sha256"] is None:
                classes.append("no_body")
            elif entry["sha256"] == evaluation_keys_sha256:
                classes.append("evaluation_keys")
            elif entry["sha256"] in ciphertext_sha256:
                classes.append("input_ciphertext")
            else:
                classes.append("UNRECOGNIZED")
        return {"requests": self.entries, "classes": classes, "all_bodies_are_evaluation_keys_or_ciphertexts": "UNRECOGNIZED" not in classes}


def _guards(config: dict[str, Any]) -> dict[str, Any]:
    g = config["memory_guards"]
    return {"interval_seconds": float(g["sample_interval_seconds"]), "abort_min_mem_available_mb": float(g["abort_min_mem_available_mb"]), "abort_max_swap_used_mb": float(g["abort_max_swap_used_mb"])}


# ------------------------------------------------------------------------------------------------ in-process flow (LR, MLP)


def run_inprocess(config: dict[str, Any], config_path: str, model_id: str) -> dict[str, Any]:
    from src.client.client import CiphraudClient
    from src.client.http import ApiClient
    from src.fhe.deploy import request_positions
    from src.fhe.xgb_poc import peak_rss_mb

    paths = paths_for(config, model_id)
    manifest, reference = json.loads(paths["manifest"].read_text()), json.loads(paths["reference"].read_text())
    model_cfg = model_config(config, model_id)
    positions = request_positions(reference["fixed_position"], reference["t1_positions"], bool(model_cfg["t1_rows"]), int(model_cfg["repeat_fixed_row"]))
    rows = read_raw_rows(manifest, positions)
    logs = PROJECT_ROOT / config["output"]["logs_dir"] / model_id
    port = free_port()
    metrics: dict[str, Any] = {"id": model_id, "model_type": manifest["model_type"], "mode": "in_process_client", "request_positions": positions, "t3_disclosure": manifest["t3_disclosure"], "test_partition_touched": False}
    watch = MemWatch(log_path=logs / "memwatch.jsonl", **_guards(config))
    server = ServerProcess(config_path, model_id, port, logs / "server.log")
    failure_stage = "server_start"
    api = None
    fhe_started = False
    audit = WireAudit()
    try:
        watch.watch("server", server.pid, kill=True)
        watch.watch("client", os.getpid(), kill=False)
        watch.start()
        ready = server.wait_ready(float(config["server"]["startup_timeout_seconds"]))
        metrics["server_startup"] = {"ready_seconds": ready["ready_seconds"], "model_load_seconds": ready["models"][0]["load_seconds"], "server_pid_matches_child": True}
        api = ApiClient(server.url, request_hook=audit.hook)
        failure_stage = "client_load"
        t0 = time.perf_counter()
        client = CiphraudClient(paths["client_dir"])
        metrics["client_load_seconds"] = time.perf_counter() - t0
        served = ready["models"][0]
        metrics["artifact_compatibility"] = {
            "client_zip_sha256": client.client_zip_sha256, "server_reports_client_zip_sha256": served["client_zip_sha256"], "server_zip_sha256": served["server_zip_sha256"],
            "versions": served["versions"], "match": client.client_zip_sha256 == served["client_zip_sha256"],
        }
        if not metrics["artifact_compatibility"]["match"]:
            raise RuntimeError("the client artifact does not match the one the served circuit was deployed with")
        failure_stage = "keygen"
        keygen = client.generate_keys()
        ek = client.evaluation_keys()
        failure_stage = "key_upload"
        uploaded = api.register_keys(model_id, ek["bytes"], client.client_zip_sha256)
        if uploaded["json"]["key_id"] != ek["sha256"]:
            raise RuntimeError("the server's key_id differs from the sha256 of the evaluation-key bytes the client sent")
        key_id = uploaded["json"]["key_id"]
        metrics["one_time"] = {
            "client_keygen_seconds": keygen["keygen_seconds"], "evaluation_keys_serialize_seconds": ek["serialize_seconds"], "evaluation_keys_bytes": ek["n_bytes"],
            "key_upload_http_seconds": uploaded["seconds"], "server_key_deserialize_seconds": uploaded["json"]["deserialize_seconds"], "secret_key_sent": False,
        }
        records, ct_hashes = [], set()
        failure_stage = "request"
        fhe_started = True
        for i, position in enumerate(positions):
            role = "fixed" if i == 0 else ("t1" if position != reference["fixed_position"] else "repeat")
            prepared = client.prepare_request(rows[position])
            ct_hashes.add(sha256_bytes(prepared["ciphertext"]))
            sent = api.infer(model_id, key_id, prepared["ciphertext"])
            finished = client.finish(sent["content"])
            records.append(request_record(i, position, role, prepared, sent, finished, reference["rows"][str(position)], reference["threshold"]))
            logger.info("Request complete", extra={"extra_fields": {"model": model_id, "i": i, "exact": records[-1]["correctness"]["exact_match"], "fhe_run_s": records[-1]["server_timings"]["fhe_run_s"]}})
        failure_stage = "key_delete"
        metrics["one_time"]["key_delete_seconds"] = api.delete_keys(key_id)["seconds"]
        after = api.models()
        metrics["keys_resident_after_delete"] = after["key_resident"] is not None
        metrics["requests"] = records
        metrics["summary"] = summarize_requests(records, reference["fixed_position"])
        metrics["wire_audit"] = audit.classify(ek["sha256"], ct_hashes)
        metrics["all_requests_exact_match"] = all(r["correctness"]["exact_match"] for r in records)
        metrics["all_decisions_match_reference"] = all(r["decision"] == r["reference_decision"] for r in records)
        metrics["status"] = "passed" if metrics["all_requests_exact_match"] and not metrics["keys_resident_after_delete"] and metrics["wire_audit"]["all_bodies_are_evaluation_keys_or_ciphertexts"] else "failed_exact_match"
    except Exception as exc:  # recorded, never swallowed silently
        metrics["status"] = "aborted_memory" if watch.aborted else "error"
        metrics["failure"] = {
            "stage": failure_stage, "type": type(exc).__name__, "message": str(exc)[:500], "server_started": "server_startup" in metrics,
            "fhe_execution_started": fhe_started, "secret_key_stayed_local": True, "secret_key_sent": False,
        }
    finally:
        if api:
            api.close()
        summary = watch.stop()
        server_rc = server.stop()
        metrics["memory"] = {**summary, "client_process_peak_rss_mb": peak_rss_mb()}
        metrics["server_exit_code"] = server_rc
    manifest_path = paths["manifest"]
    out_dir = PROJECT_ROOT / config["output"]["results_dir"] / model_id
    _write_json(out_dir / "metrics.json", metrics)
    _write_json(out_dir / "provenance.json", build_provenance(config, config_path, manifest_path))
    return metrics


# ------------------------------------------------------------------------------------------------ staged flow (XGBoost)


def _stage_dir(config: dict[str, Any], model_id: str) -> Path:
    return PROJECT_ROOT / config["output"]["client_keys_dir"] / model_id


def stage_keys(config: dict[str, Any], config_path: str, model_id: str) -> dict[str, Any]:
    from src.client.client import CiphraudClient
    from src.fhe.xgb_poc import peak_rss_mb

    paths = paths_for(config, model_id)
    manifest, reference = json.loads(paths["manifest"].read_text()), json.loads(paths["reference"].read_text())
    stage = _stage_dir(config, model_id)
    stage.mkdir(parents=True, exist_ok=True)
    position = int(reference["fixed_position"])
    rows = read_raw_rows(manifest, [position])
    state: dict[str, Any] = {"stage": "keys", "id": model_id, "position": position, "status": "started"}
    marker = stage / "state_keys.json"
    # No in-process watchdog here: Concrete's native keygen/serialize calls hold the GIL, which starves a watchdog THREAD in this very
    # process exactly when memory spikes (measured: 23 samples in an ~18 s stage at 10 Hz). The parent orchestrator (`run_staged`),
    # idle in `wait()`, watches this process from outside and kills it if a guard trips.
    try:
        t0 = time.perf_counter()
        client = CiphraudClient(paths["client_dir"])
        state["client_load_seconds"] = time.perf_counter() - t0
        state.update(client.generate_keys())
        ek = client.evaluation_keys()
        (stage / "eval_keys.bin").write_bytes(ek["bytes"])
        state.update({"evaluation_keys_bytes": ek["n_bytes"], "evaluation_keys_serialize_seconds": ek["serialize_seconds"], "evaluation_keys_sha256": ek["sha256"], "client_zip_sha256": client.client_zip_sha256})
        del ek
        prepared = client.prepare_request(rows[position])
        (stage / "request_000.ct").write_bytes(prepared["ciphertext"])
        state["request"] = {"ciphertext_bytes": prepared["n_bytes"], "ciphertext_sha256": sha256_bytes(prepared["ciphertext"]), "timings": prepared["timings"]}
        keyset = stage / "keyset.bin"
        keyset.unlink(missing_ok=True)
        state.update(client.save_keys(keyset))
        state.update({"status": "ok", "memory": {"process_peak_rss_mb": peak_rss_mb()}})
    except Exception as exc:  # noqa: BLE001 -- recorded as evidence, then re-raised
        state.update({"status": "error", "error": f"{type(exc).__name__}: {str(exc)[:300]}", "memory": {"process_peak_rss_mb": peak_rss_mb()}})
        _write_json(marker, state)
        raise
    _write_json(marker, state)
    return state


def _sender_peak_rss_mb() -> float:
    """This process's peak RSS in MB (same formula as `xgb_poc.peak_rss_mb`, stdlib only). The serve stage is a key-less HTTP sender: importing
    `xgb_poc` just for this one number pulled Concrete-ML and torch into it (about 590 MB, measured), memory the server needs."""
    import resource

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def stage_serve(config: dict[str, Any], config_path: str, model_id: str, preflight_only: bool = False) -> dict[str, Any]:
    from src.client.http import ApiClient

    peak_rss_mb = _sender_peak_rss_mb

    guards = config["memory_guards"]
    paths = paths_for(config, model_id)
    stage = _stage_dir(config, model_id)
    keys_state = json.loads((stage / "state_keys.json").read_text())
    state: dict[str, Any] = {"stage": "serve", "id": model_id, "status": "started", "fhe_execution_started": False, "secret_key_sent": False}
    logs = PROJECT_ROOT / config["output"]["logs_dir"] / model_id
    watch = MemWatch(log_path=logs / "memwatch_serve.jsonl", **_guards(config))
    server = ServerProcess(config_path, model_id, free_port(), logs / "server.log")
    api = None
    audit = WireAudit()
    try:
        watch.watch("server", server.pid, kill=True)  # the memory hog is the server; the sender is never the guard's target
        watch.watch("sender", os.getpid(), kill=False)
        watch.start()
        ready = server.wait_ready(float(config["server"]["startup_timeout_seconds"]))
        state["server_startup"] = {"ready_seconds": ready["ready_seconds"], "model_load_seconds": ready["models"][0]["load_seconds"]}
        state["memory_after_server_load"] = system_memory_mb() | {"server": _proc(server.pid)}
        api = ApiClient(server.url, request_hook=audit.hook)
        eval_bytes = (stage / "eval_keys.bin").read_bytes()
        uploaded = api.register_keys(model_id, eval_bytes, keys_state["client_zip_sha256"])
        state["key_upload"] = {"http_seconds": uploaded["seconds"], "server_deserialize_seconds": uploaded["json"]["deserialize_seconds"], "bytes": len(eval_bytes)}
        del eval_bytes
        time.sleep(2 * float(guards["sample_interval_seconds"]))
        after = system_memory_mb()
        state["memory_after_registration"] = after | {"server": _proc(server.pid)}
        go = after["mem_available_mb"] >= float(guards["go_min_mem_available_mb"]) and after["swap_used_mb"] <= float(guards["go_max_swap_used_mb"])
        state["go_no_go"] = {"go": bool(go), "mem_available_mb": after["mem_available_mb"], "swap_used_mb": after["swap_used_mb"], "required_mem_available_mb": float(guards["go_min_mem_available_mb"]), "allowed_swap_used_mb": float(guards["go_max_swap_used_mb"])}
        if not go:
            state["status"] = "preflight_failed_go_no_go"
            api.delete_keys(uploaded["json"]["key_id"])
        elif preflight_only:
            # Stages A-C only: the keys are registered and the memory verdict recorded, but NO FHE run is started.
            state.update({"status": "ok", "preflight_only": True})
            state["key_delete_seconds"] = api.delete_keys(uploaded["json"]["key_id"])["seconds"]
        else:
            ciphertext = (stage / "request_000.ct").read_bytes()
            box: dict[str, Any] = {}

            def call() -> None:
                try:
                    box["sent"] = api.infer(model_id, uploaded["json"]["key_id"], ciphertext)
                except Exception as exc:  # noqa: BLE001 -- recorded below
                    box["error"] = exc

            state["fhe_execution_started"] = True
            thread = threading.Thread(target=call, daemon=True)
            started = time.perf_counter()
            thread.start()
            thread.join(timeout=float(guards["run_timeout_seconds"]))
            state["infer_wall_seconds"] = time.perf_counter() - started
            if thread.is_alive():
                state["status"] = "run_timeout"
            elif "error" in box:
                state["status"] = "aborted_memory" if watch.aborted else "error"
                state["error"] = f"{type(box['error']).__name__}: {str(box['error'])[:300]}"
            else:
                sent = box["sent"]
                (stage / "response_000.bin").write_bytes(sent["content"])
                state.update({"status": "ok", "sent": {k: sent[k] for k in ("seconds", "server_timing", "request_bytes", "response_bytes", "server_zip_sha256")}})
                state["key_delete_seconds"] = api.delete_keys(uploaded["json"]["key_id"])["seconds"]
        classes = audit.classify(keys_state["evaluation_keys_sha256"], {keys_state["request"]["ciphertext_sha256"]})
        state["wire_audit"] = classes
    except Exception as exc:  # noqa: BLE001
        state["status"] = "aborted_memory" if watch.aborted else "error"
        state["error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
    finally:
        if api:
            api.close()
        state["memory"] = {**watch.stop(), "sender_process_peak_rss_mb": peak_rss_mb()}
        state["server_exit_code"] = server.stop()
    _write_json(stage / "state_serve.json", state)
    return state


def _proc(pid: int) -> dict[str, float] | None:
    from src.benchmark.memwatch import process_memory_mb

    return process_memory_mb(pid)


def stage_decrypt(config: dict[str, Any], config_path: str, model_id: str) -> dict[str, Any]:
    from src.client.client import CiphraudClient
    from src.fhe.xgb_poc import peak_rss_mb

    paths = paths_for(config, model_id)
    manifest, reference = json.loads(paths["manifest"].read_text()), json.loads(paths["reference"].read_text())
    stage = _stage_dir(config, model_id)
    serve = json.loads((stage / "state_serve.json").read_text())
    state: dict[str, Any] = {"stage": "decrypt", "id": model_id, "status": "started"}
    marker = stage / "state_decrypt.json"
    try:  # watched from outside by the parent orchestrator (see `stage_keys`)
        client = CiphraudClient(paths["client_dir"])
        state.update(client.load_keys(stage / "keyset.bin"))
        finished = client.finish((stage / "response_000.bin").read_bytes())
        ref_row = reference["rows"][str(reference["fixed_position"])]
        state.update({
            "probability": finished["probability"], "decision": finished["decision"], "decrypted_ints": finished["integers"].reshape(-1).tolist(),
            "correctness": compare_integers(finished["integers"], ref_row), "timings": finished["timings"], "status": "ok",
            "memory": {"process_peak_rss_mb": peak_rss_mb()},
        })
    except Exception as exc:  # noqa: BLE001 -- recorded as evidence, then re-raised
        state.update({"status": "error", "error": f"{type(exc).__name__}: {str(exc)[:300]}", "memory": {"process_peak_rss_mb": peak_rss_mb()}})
        _write_json(marker, state)
        raise
    _write_json(marker, state)
    return state


def run_staged(config: dict[str, Any], config_path: str, model_id: str, preflight_only: bool = False) -> dict[str, Any]:
    """Orchestrate the stages as SEPARATE processes, then merge them into `metrics.json` (or `preflight.json` when
    `preflight_only`: keygen -> upload -> go/no-go with NO FHE run, to measure memory feasibility cheaply)."""
    paths = paths_for(config, model_id)
    manifest, reference = json.loads(paths["manifest"].read_text()), json.loads(paths["reference"].read_text())
    stage = _stage_dir(config, model_id)
    if stage.exists():
        for leftover in stage.iterdir():
            leftover.unlink()
    metrics: dict[str, Any] = {
        "id": model_id, "model_type": manifest["model_type"], "mode": "staged_client", "request_positions": [reference["fixed_position"]],
        "t3_disclosure": manifest["t3_disclosure"], "test_partition_touched": False, "memory_guards": config["memory_guards"],
        "memory_at_start": system_memory_mb(), "stages": {},
    }
    status = "error"
    stage_names = ("keys", "serve") if preflight_only else ("keys", "serve", "decrypt")
    metrics["preflight_only"] = preflight_only
    for name in stage_names:
        t0 = time.perf_counter()
        extra = ["--preflight-only"] if (preflight_only and name == "serve") else []
        # The PARENT watches every stage from outside: it sits in `wait()` (GIL free), so its sampling thread is never starved by the
        # child's native calls. If a guard trips it SIGKILLs the child (a server child then dies with it: PR_SET_PDEATHSIG).
        parent_watch = MemWatch(log_path=PROJECT_ROOT / config["output"]["logs_dir"] / model_id / f"memwatch_parent_{name}.jsonl", **_guards(config))
        parent_watch.start()
        proc = subprocess.Popen([sys.executable, "-m", "src.benchmark.phase10_roundtrip", "--config", config_path, "--model", model_id, "--stage", name, *extra], cwd=PROJECT_ROOT)
        parent_watch.watch(f"stage_{name}", proc.pid, kill=True)
        proc.wait()
        parent_summary = parent_watch.stop()
        state_path = stage / f"state_{name}.json"
        stage_state = json.loads(state_path.read_text()) if state_path.exists() else {"status": "no_state_written"}
        if parent_summary["aborted"]:  # the guard, not the stage, is the authority on why it stopped
            stage_state = {**stage_state, "status": "aborted_memory", "reason": parent_summary["abort_reason"]}
        metrics["stages"][name] = {"exit_code": proc.returncode, "wall_seconds": time.perf_counter() - t0, "parent_watch": parent_summary, **stage_state}
        metrics["memory_after_" + name] = system_memory_mb()
        if stage_state.get("status") != "ok":
            # A stage that died without writing state (e.g. a kernel OOM kill) is recorded as exactly that.
            stage_status = stage_state.get("status", "error")
            status = stage_status if stage_status in ("aborted_memory", "preflight_failed_go_no_go", "run_timeout") else ("preflight_failed_" + name if name != "decrypt" else "error")
            metrics["failure"] = {
                "stage": name, "stage_status": stage_status, "exit_code": proc.returncode, "reason": stage_state.get("reason") or stage_state.get("error"),
                "server_started": "server_startup" in metrics["stages"].get("serve", {}), "fhe_execution_started": bool(metrics["stages"].get("serve", {}).get("fhe_execution_started")),
                "secret_key_stayed_local": True, "secret_key_sent": False,
            }
            break
    else:
        if preflight_only:
            status = "preflight_passed"
            metrics["status"] = status
            _finalize_staged(config, config_path, model_id, paths, stage, metrics, name="preflight.json")
            return metrics
        d = metrics["stages"]["decrypt"]
        sent = metrics["stages"]["serve"]["sent"]
        metrics["requests"] = [{
            "position": reference["fixed_position"], "role": "fixed", "probability": d["probability"], "decision": d["decision"], "decrypted_ints": d["decrypted_ints"], "correctness": d["correctness"],
            "server_timings": {k: sent["server_timing"][k] for k in ("input_deserialize_s", "fhe_run_s", "output_serialize_s", "handler_total_s")},
            "http_roundtrip_seconds": sent["seconds"], "transport_and_framework_seconds": sent["seconds"] - sent["server_timing"]["handler_total_s"],
            "client_timings": {**metrics["stages"]["keys"]["request"]["timings"], **d["timings"]}, "bytes": {"request_body": sent["request_bytes"], "response_body": sent["response_bytes"]},
        }]
        metrics["wire_audit"] = metrics["stages"]["serve"]["wire_audit"]
        metrics["all_requests_exact_match"] = d["correctness"]["exact_match"]
        status = "passed" if d["correctness"]["exact_match"] and metrics["wire_audit"]["all_bodies_are_evaluation_keys_or_ciphertexts"] else "failed_exact_match"
    metrics["status"] = status
    _finalize_staged(config, config_path, model_id, paths, stage, metrics, name="preflight.json" if preflight_only else "metrics.json")
    return metrics


def _finalize_staged(config: dict[str, Any], config_path: str, model_id: str, paths: dict[str, Path], stage: Path, metrics: dict[str, Any], name: str) -> None:
    """Delete the client-local secret keyset file and write the result. Runs on EVERY exit path of the staged flow."""
    keyset = stage / "keyset.bin"
    metrics["client_keyset_file_deleted"] = False
    if keyset.exists():
        keyset.unlink()
        metrics["client_keyset_file_deleted"] = True
    out_dir = PROJECT_ROOT / config["output"]["results_dir"] / model_id
    _write_json(out_dir / name, metrics)
    _write_json(out_dir / ("preflight_provenance.json" if name == "preflight.json" else "provenance.json"), build_provenance(config, config_path, paths["manifest"]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/phase10/client_server.yaml")
    parser.add_argument("--model", required=True)
    parser.add_argument("--staged", action="store_true", help="XGBoost: run the three-process staged flow")
    parser.add_argument("--stage", choices=("keys", "serve", "decrypt"), help="internal: one stage of the staged flow")
    parser.add_argument("--preflight-only", action="store_true", help="staged flow: keygen -> upload -> memory go/no-go, NO FHE run (writes preflight.json)")
    args = parser.parse_args()
    import warnings

    warnings.filterwarnings("ignore")
    config = load_config(args.config)
    if args.stage == "serve":
        stage_serve(config, args.config, args.model, preflight_only=args.preflight_only)
        _exit(0)
    if args.stage:
        {"keys": stage_keys, "decrypt": stage_decrypt}[args.stage](config, args.config, args.model)
        _exit(0)
    if args.staged:
        result = run_staged(config, args.config, args.model, preflight_only=args.preflight_only)
    else:
        result = run_inprocess(config, args.config, args.model)
    print(json.dumps({"id": result["id"], "status": result["status"], "n_requests": len(result.get("requests", []))}), flush=True)
    _exit(0 if result["status"] in ("passed", "preflight_passed") else 1)


def _exit(code: int) -> None:
    """Flush and exit with the REAL status: a process that has run a Concrete circuit in-process can otherwise exit 0 regardless of
    `sys.exit` and drop unflushed output (measured; see `src/fhe/deploy.py::_finish`). `metrics.json` is the authoritative record."""
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


if __name__ == "__main__":
    main()
