"""The Phase 10 FastAPI inference service (`docs/client_server.md`; deviation from `docs/architecture.md` Sec.18 recorded there).

Four label-less endpoints; every body is Concrete's own serialized bytes as `application/octet-stream`:

    GET    /v1/models                           what is loaded (readiness probe, hashes, versions, T3 disclosure)
    POST   /v1/keys?model=<id>                  register the client's EVALUATION keys (body = serialized bytes; header
                                                X-Ciphraud-Client-Zip-Sha256 = the client artifact the keys were generated from)
    POST   /v1/infer?model=<id>&key_id=<id>     run the circuit on one input ciphertext (body = serialized ciphertext)
    DELETE /v1/keys?key_id=<id>                 drop the key set from memory

The server holds only the compiled circuit and the client's evaluation keys. No endpoint accepts feature values or returns
a decrypted result; nothing here can decrypt. Memory policy for the 3.8 GiB WSL VM: at most ONE key set is resident (a
second registration is a 409), keys live only in memory, and one lock keeps FHE runs from overlapping (a concurrent
request is a 503) while `GET /v1/models` stays responsive during a long run. Malformed input is a 4xx, never a crashed process.

Server-side timing is returned in the `X-Ciphraud-Timing` header: input deserialize, `fhe_run_s` (the only figure comparable
to the Phase 8/9 FHE latency), output serialize, `handler_total_s` (their sum, excluding time spent receiving the body), and
the server process's own memory.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import threading
import time
from dataclasses import dataclass
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.concurrency import run_in_threadpool

from src.benchmark.memwatch import process_memory_mb
from src.logging_setup import get_logger

logger = get_logger(__name__)

OCTET = "application/octet-stream"
CLIENT_ARTIFACT_HEADER = "X-Ciphraud-Client-Zip-Sha256"


@dataclass
class ResidentKeys:
    key_id: str
    model: str
    evaluation_keys: Any
    n_bytes: int
    deserialize_seconds: float


class _State:
    def __init__(self, registry: Any, server_cfg: dict[str, Any]) -> None:
        self.registry = registry
        self.cfg = server_cfg
        self.run_lock = threading.Lock()
        self.key_lock = threading.Lock()
        self.resident: ResidentKeys | None = None


def _error(status: int, code: str, message: str, **extra: Any) -> HTTPException:
    return HTTPException(status_code=status, detail={"error": code, "message": message, **extra})


def _fhe() -> Any:
    from concrete import fhe

    return fhe


async def _read_octet_body(request: Request, cap: int) -> bytes:
    content_type = request.headers.get("content-type", "").split(";")[0].strip().lower()
    if content_type != OCTET:
        raise _error(415, "unsupported_media_type", f"expected content-type {OCTET}, got {content_type or 'none'}; this server never accepts plaintext or JSON payloads")
    declared = request.headers.get("content-length")
    if declared is not None and declared.isdigit() and int(declared) > cap:
        raise _error(413, "payload_too_large", f"body of {declared} bytes exceeds the {cap}-byte limit")
    buffer = bytearray()
    async for chunk in request.stream():
        if len(buffer) + len(chunk) > cap:
            raise _error(413, "payload_too_large", f"body exceeds the {cap}-byte limit")
        buffer += chunk
    return bytes(buffer)


def _register(state: _State, model_id: str, body: bytes, key_id: str) -> dict[str, Any]:
    fhe = _fhe()
    t0 = time.perf_counter()
    try:
        keys = fhe.EvaluationKeys.deserialize(body)
    except Exception as exc:  # malformed bytes surface as a Concrete runtime error
        raise _error(400, "bad_evaluation_keys", f"the body is not a serialized EvaluationKeys object: {type(exc).__name__}") from exc
    seconds = time.perf_counter() - t0
    with state.key_lock:
        if state.resident is not None:
            raise _error(409, "key_set_resident", "a key set is already resident; DELETE /v1/keys first", key_id=state.resident.key_id)
        state.resident = ResidentKeys(key_id, model_id, keys, len(body), seconds)
    return {"key_id": key_id, "model": model_id, "n_bytes": len(body), "deserialize_seconds": seconds, "server_memory": process_memory_mb("self")}


def _run_locked(state: _State, model_id: str, resident: ResidentKeys, body: bytes) -> tuple[bytes, dict[str, Any]]:
    fhe = _fhe()
    if not state.run_lock.acquire(blocking=False):
        raise _error(503, "busy", "an inference is already running")
    try:
        entry = state.registry.get(model_id)
        t_start = time.perf_counter()
        try:
            value = fhe.Value.deserialize(body)
        except Exception as exc:
            raise _error(400, "bad_ciphertext", f"the body is not a serialized ciphertext: {type(exc).__name__}") from exc
        input_deserialize = time.perf_counter() - t_start
        t0 = time.perf_counter()
        try:
            result = entry.server.run(value, resident.evaluation_keys)
        except Exception as exc:  # e.g. a ciphertext or key set that does not belong to this circuit
            raise _error(422, "inference_failed", f"the circuit rejected this ciphertext/key set: {type(exc).__name__}: {str(exc)[:200]}") from exc
        fhe_run = time.perf_counter() - t0
        t0 = time.perf_counter()
        payload = result.serialize()
        output_serialize = time.perf_counter() - t0
        timing = {
            "input_deserialize_s": input_deserialize, "fhe_run_s": fhe_run, "output_serialize_s": output_serialize,
            "handler_total_s": time.perf_counter() - t_start, "server_memory": process_memory_mb("self"),
        }
        return payload, timing
    finally:
        state.run_lock.release()


def create_app(registry: Any, server_cfg: dict[str, Any]) -> FastAPI:
    """`registry` needs `.get(id)` (an entry with `.server.run(...)`) and `.describe()`; see `src/server/registry.py`."""
    app = FastAPI(title="Ciphraud encrypted inference", version="1")
    state = _State(registry, server_cfg)
    app.state.ciphraud = state

    @app.get("/v1/models")
    def list_models() -> dict[str, Any]:
        resident = state.resident
        return {
            "models": registry.describe(),
            "key_resident": None if resident is None else {"key_id": resident.key_id, "model": resident.model, "n_bytes": resident.n_bytes},
            "busy": state.run_lock.locked(), "pid": os.getpid(),
        }

    @app.post("/v1/keys")
    async def register_keys(request: Request, model: str) -> dict[str, Any]:
        entry = registry.get(model)
        if entry is None:
            raise _error(404, "unknown_model", f"no model {model!r} is loaded")
        # Concrete's native runtime SEGFAULTS (killing this whole process) when it runs a circuit with evaluation keys generated
        # for a different circuit (measured, docs/client_server.md). The client therefore declares the sha256 of the client
        # artifact its keys were generated from, and a mismatch is rejected before the body is even read. This protects against an
        # honest mistake; a client that deliberately lies can still crash the process (availability against a malicious client is out
        # of scope: loopback, no authentication).
        declared = request.headers.get(CLIENT_ARTIFACT_HEADER)
        expected = entry.manifest["artifacts"]["client_zip"]["sha256"]
        if declared != expected:
            raise _error(422, "client_artifact_mismatch", f"the {CLIENT_ARTIFACT_HEADER} header must equal this model's client-artifact sha256 ({expected[:16]}...); got {str(declared)[:16]}")
        if state.resident is not None:
            raise _error(409, "key_set_resident", "a key set is already resident; DELETE /v1/keys first", key_id=state.resident.key_id)
        body = await _read_octet_body(request, int(state.cfg["max_key_upload_bytes"]))
        key_id = hashlib.sha256(body).hexdigest()
        result = await run_in_threadpool(_register, state, model, body, key_id)
        del body
        gc.collect()
        logger.info("Evaluation keys registered", extra={"extra_fields": {"model": model, "key_id": key_id[:16], "n_bytes": result["n_bytes"]}})
        return result

    @app.post("/v1/infer")
    async def infer(request: Request, model: str, key_id: str) -> Response:
        if registry.get(model) is None:
            raise _error(404, "unknown_model", f"no model {model!r} is loaded")
        resident = state.resident
        if resident is None or resident.key_id != key_id:
            raise _error(404, "unknown_key_id", "no resident key set with this key_id; POST /v1/keys first")
        if resident.model != model:
            raise _error(422, "key_model_mismatch", f"this key set was registered for {resident.model!r}, not {model!r}")
        body = await _read_octet_body(request, int(state.cfg["max_ciphertext_bytes"]))
        payload, timing = await run_in_threadpool(_run_locked, state, model, resident, body)
        logger.info("Inference served", extra={"extra_fields": {"model": model, "fhe_run_s": timing["fhe_run_s"], "response_bytes": len(payload)}})
        headers = {"X-Ciphraud-Timing": json.dumps(timing), "X-Ciphraud-Server-Zip-Sha256": registry.get(model).server_zip_sha256}
        return Response(content=payload, media_type=OCTET, headers=headers)

    @app.delete("/v1/keys", status_code=204)
    def delete_keys(key_id: str) -> Response:
        if state.run_lock.locked():
            raise _error(409, "busy", "an inference is running; retry after it finishes")
        with state.key_lock:
            if state.resident is None or state.resident.key_id != key_id:
                raise _error(404, "unknown_key_id", "no resident key set with this key_id")
            state.resident = None
        gc.collect()
        logger.info("Evaluation keys deleted", extra={"extra_fields": {"key_id": key_id[:16]}})
        return Response(status_code=204)

    return app
