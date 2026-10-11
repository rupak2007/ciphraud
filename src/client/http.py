"""HTTP transport for the Phase 10 client (`httpx`, already a transitive dependency of the Concrete-ML environment).

Every body is raw `application/octet-stream` bytes produced by Concrete's own `serialize()` (evaluation keys, ciphertexts);
metadata travels in query parameters and response headers. No timeout is set: one XGBoost server run is ~44 minutes
(Phase 8) and a default 5-second client timeout would kill it. Each call reports its own wall time so the caller can keep
transport separate from server-side FHE time.
"""

from __future__ import annotations

import json
import time
from typing import Any, Callable

import httpx

OCTET = {"Content-Type": "application/octet-stream"}


class ApiError(Exception):
    def __init__(self, status: int, detail: Any) -> None:
        super().__init__(f"HTTP {status}: {detail}")
        self.status = status
        self.detail = detail


def _raise_for(response: httpx.Response) -> None:
    if response.status_code >= 400:
        try:
            detail = response.json()
        except ValueError:
            detail = response.text[:500]
        raise ApiError(response.status_code, detail)


class ApiClient:
    def __init__(self, base_url: str, request_hook: Callable[[httpx.Request], None] | None = None) -> None:
        hooks = {"request": [request_hook]} if request_hook else {}
        self._http = httpx.Client(base_url=base_url, timeout=None, event_hooks=hooks)

    def close(self) -> None:
        self._http.close()

    def models(self) -> dict[str, Any]:
        response = self._http.get("/v1/models")
        _raise_for(response)
        return response.json()

    def register_keys(self, model: str, evaluation_keys: bytes, client_zip_sha256: str) -> dict[str, Any]:
        """`client_zip_sha256`: the client artifact the keys were generated from -- the server refuses a mismatch, because
        Concrete's native runtime crashes when a circuit is run with another circuit's keys (`docs/client_server.md`)."""
        t0 = time.perf_counter()
        response = self._http.post("/v1/keys", params={"model": model}, content=evaluation_keys, headers={**OCTET, "X-Ciphraud-Client-Zip-Sha256": client_zip_sha256})
        seconds = time.perf_counter() - t0
        _raise_for(response)
        return {"json": response.json(), "seconds": seconds, "request_bytes": len(evaluation_keys)}

    def infer(self, model: str, key_id: str, ciphertext: bytes) -> dict[str, Any]:
        t0 = time.perf_counter()
        response = self._http.post("/v1/infer", params={"model": model, "key_id": key_id}, content=ciphertext, headers=OCTET)
        seconds = time.perf_counter() - t0
        _raise_for(response)
        timing = json.loads(response.headers.get("x-ciphraud-timing", "{}"))
        return {
            "content": response.content, "seconds": seconds, "server_timing": timing, "request_bytes": len(ciphertext),
            "response_bytes": len(response.content), "server_zip_sha256": response.headers.get("x-ciphraud-server-zip-sha256"),
        }

    def delete_keys(self, key_id: str) -> dict[str, Any]:
        t0 = time.perf_counter()
        response = self._http.delete("/v1/keys", params={"key_id": key_id})
        seconds = time.perf_counter() - t0
        _raise_for(response)
        return {"seconds": seconds}
