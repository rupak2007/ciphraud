"""Server-side model registry (`docs/client_server.md`).

Loads, for each configured model id, ONLY the persisted `server.zip` written by `src/fhe/deploy.py`, after checking its
sha256 against the deployment manifest. The server never opens the client-side artifacts. WSL2/Linux only (Concrete-ML).
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.config import PROJECT_ROOT


class RegistryError(Exception):
    """A configured model cannot be served (missing artifact, hash mismatch, version mismatch)."""


@dataclass
class LoadedModel:
    id: str
    manifest: dict[str, Any]
    server: Any
    server_zip_sha256: str
    load_seconds: float

    def describe(self) -> dict[str, Any]:
        m = self.manifest
        return {
            "id": self.id, "model_type": m["model_type"], "tier": m["tier"], "n_bits": m["n_bits"], "n_features": m["n_features"],
            "server_zip_sha256": self.server_zip_sha256, "client_zip_sha256": m["artifacts"]["client_zip"]["sha256"],
            "versions": m["versions"], "security_level": m["security_level"], "circuit_stats": m["circuit_stats"],
            "source": m["source"], "t3_disclosure": m["t3_disclosure"], "load_seconds": self.load_seconds,
        }


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ModelRegistry:
    def __init__(self, config: dict[str, Any], model_ids: list[str] | None = None) -> None:
        from concrete.ml.deployment import FHEModelServer

        configured = [m["id"] for m in config["models"]]
        wanted = model_ids or configured
        unknown = [i for i in wanted if i not in configured]
        if unknown:
            raise RegistryError(f"unknown model ids {unknown}; configured: {configured}")
        self._models: dict[str, LoadedModel] = {}
        for model_id in wanted:
            manifest_path = PROJECT_ROOT / config["output"]["results_dir"] / "deploy" / model_id / "manifest.json"
            server_dir = PROJECT_ROOT / config["output"]["deploy_dir"] / model_id / "server"
            if not manifest_path.exists() or not (server_dir / "server.zip").exists():
                raise RegistryError(f"{model_id}: no deployment (expected {manifest_path} and {server_dir / 'server.zip'}); run src.fhe.deploy first")
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            actual = _sha256_path(server_dir / "server.zip")
            if actual != manifest["artifacts"]["server_zip"]["sha256"]:
                raise RegistryError(f"{model_id}: server.zip sha256 {actual} does not match the manifest {manifest['artifacts']['server_zip']['sha256']}")
            t0 = time.perf_counter()
            try:
                server = FHEModelServer(str(server_dir))
            except ValueError as exc:  # Concrete's own client/server version check
                raise RegistryError(f"{model_id}: {exc}") from exc
            self._models[model_id] = LoadedModel(model_id, manifest, server, actual, time.perf_counter() - t0)

    def get(self, model_id: str) -> LoadedModel | None:
        return self._models.get(model_id)

    def describe(self) -> list[dict[str, Any]]:
        return [m.describe() for m in self._models.values()]
