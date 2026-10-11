"""The Phase 10 client: key generation, quantize -> encrypt -> serialize, deserialize -> decrypt -> probability/decision.

WSL2/Linux only (Concrete-ML). The client owns the secret key: it exists only in this process's memory (or, for the staged
XGBoost run, in one client-local keyset file the runner deletes afterwards) and this module has no code path that sends it
anywhere -- the only key material it ever exposes is the serialized EVALUATION keys (`evaluation_keys`).

Built on Concrete-ML's own deployment client (`FHEModelClient`, loaded from `client.zip`), split into separately timed steps
using its public `.model` / `.client` attributes. Two measured facts shape it (`docs/client_server.md`):

  * Concrete's `Client.encrypt` silently generates fresh keys if none are loaded, so a fresh decrypt process that forgot to
    load its keyset would quietly use a DIFFERENT one. This client therefore refuses to encrypt or decrypt until keys were
    explicitly generated or loaded here (`ClientKeyError`).
  * Default keygen is randomized across processes (two default runs gave different keysets; identical explicit seeds gave
    identical ones), so `generate_keys` uses the default and never handles a seed.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any

import numpy as np

from src.client.preprocess import apply_preprocessing, load_preprocessing


class ClientKeyError(Exception):
    """An encrypt/decrypt was attempted before keys were explicitly generated or loaded."""


class CiphraudClient:
    def __init__(self, client_dir: Path | str) -> None:
        from concrete.ml.deployment import FHEModelClient

        self.client_dir = Path(client_dir)
        self.spec = load_preprocessing(self.client_dir / "preprocessing.json")
        self._fhe = FHEModelClient(str(self.client_dir))  # no key_dir: Concrete's keyset cache is never used
        self._keys: str | None = None
        self.client_zip_sha256 = hashlib.sha256((self.client_dir / "client.zip").read_bytes()).hexdigest()

    # ---- keys -------------------------------------------------------------------------------------------------------

    def generate_keys(self) -> dict[str, float]:
        t0 = time.perf_counter()
        self._fhe.client.keygen(force=True)
        self._keys = "generated"
        return {"keygen_seconds": time.perf_counter() - t0}

    def evaluation_keys(self) -> dict[str, Any]:
        """The serialized evaluation keys: the ONLY key material that may leave this process."""
        self._require_keys()
        t0 = time.perf_counter()
        blob = self._fhe.client.evaluation_keys.serialize()
        return {"bytes": blob, "n_bytes": len(blob), "serialize_seconds": time.perf_counter() - t0, "sha256": hashlib.sha256(blob).hexdigest()}

    def save_keys(self, path: Path | str) -> dict[str, float]:
        """Persist the FULL keyset (secret key included) to a client-local file, for the staged XGBoost run only."""
        self._require_keys()
        t0 = time.perf_counter()
        self._fhe.client.keys.save(str(path))
        return {"save_keys_seconds": time.perf_counter() - t0, "keyset_bytes": Path(path).stat().st_size}

    def load_keys(self, path: Path | str) -> dict[str, float]:
        t0 = time.perf_counter()
        self._fhe.client.keys.load(str(path))
        self._keys = "loaded"
        return {"load_keys_seconds": time.perf_counter() - t0}

    def _require_keys(self) -> None:
        if self._keys is None:
            raise ClientKeyError("no keys: call generate_keys() or load_keys() first (Concrete would otherwise generate fresh keys silently)")

    # ---- request / response -----------------------------------------------------------------------------------------

    def prepare_request(self, X_raw: np.ndarray) -> dict[str, Any]:
        """Raw feature row -> preprocess -> quantize -> encrypt -> serialize. Each step timed separately."""
        self._require_keys()
        timings: dict[str, float] = {}
        t0 = time.perf_counter()
        x = apply_preprocessing(self.spec, X_raw)
        timings["preprocess_seconds"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        q = np.asarray(self._fhe.model.quantize_input(x))
        timings["quantize_seconds"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        encrypted = self._fhe.client.encrypt(q)
        timings["encrypt_seconds"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        value = encrypted[0] if isinstance(encrypted, tuple) else encrypted
        ciphertext = value.serialize()
        timings["serialize_seconds"] = time.perf_counter() - t0
        return {"ciphertext": ciphertext, "q_in": q, "n_bytes": len(ciphertext), "timings": timings}

    def finish(self, response: bytes) -> dict[str, Any]:
        """Serialized encrypted result -> deserialize -> decrypt -> dequantize -> probability -> decision."""
        from concrete import fhe

        self._require_keys()
        timings: dict[str, float] = {}
        t0 = time.perf_counter()
        value = fhe.Value.deserialize(response)
        timings["deserialize_seconds"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        decrypted = self._fhe.client.decrypt(value)
        timings["decrypt_seconds"] = time.perf_counter() - t0
        integers = np.asarray(decrypted[0] if isinstance(decrypted, tuple) else decrypted)
        t0 = time.perf_counter()
        model = self._fhe.model
        if self.spec["probability"] == "mlp_softmax":
            from src.fhe.compile import mlp

            probability = float(mlp.fraud_probability(model.dequantize_output(integers))[0])
        else:
            probability = float(model.post_processing(model.dequantize_output(integers))[:, 1][0])
        decision = bool(probability >= float(self.spec["threshold"]))
        timings["postprocess_seconds"] = time.perf_counter() - t0
        return {"integers": integers, "probability": probability, "decision": decision, "timings": timings}
