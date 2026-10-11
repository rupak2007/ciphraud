"""Phase 10 API/behavior probe: the measured facts `docs/client_server.md` and the client/server design rely on.

WSL2/Linux only. Needs the LR and MLP deployments (`src.fhe.deploy`). From the project root:

    ~/.venvs/fhe-fraud-detection/bin/python -m src.fhe.probes.phase10_probe

Writes `results/phase10_client_server/api_probe.json`. Sections:

  keygen_determinism   default / fixed-seed / random-seed keygen in FRESH processes (is default keygen randomized?)
  encrypt_without_keys does Concrete's `Client.encrypt` silently generate keys when none were loaded?
  preserialized_run    does `FHEModelServer.run` accept pre-deserialized `fhe.Value` / `fhe.EvaluationKeys` (to time them separately)?
  wire_sizes           serialized evaluation-key / ciphertext / result bytes for the deployed LR and MLP
  exit_behavior        which step (load / keygen / encrypt / RUN) makes a process lose its exit status and unflushed stdout
  run_time_ab          MLP `fhe_run` time: client's native key object vs deserialized keys vs the original in-process `circuit.run`
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from src.config import PROJECT_ROOT, load_config

CONFIG = "configs/phase10/client_server.yaml"


def _deploy_paths(model_id: str) -> dict[str, Path]:
    from src.fhe.deploy import paths_for

    return paths_for(load_config(CONFIG), model_id)


def keygen_determinism(model_id: str = "lr_top20_bits16") -> dict[str, Any]:
    """Six FRESH processes; a keyset's sha256 identifies it."""
    client_dir = _deploy_paths(model_id)["client_dir"]
    code = (
        "import hashlib, os, sys, warnings; warnings.filterwarnings('ignore')\n"
        "from concrete.ml.deployment import FHEModelClient\n"
        "c = FHEModelClient(sys.argv[2]); m = sys.argv[1]\n"
        "if m == 'default': c.client.keygen(force=True)\n"
        "elif m == 'fixed': c.client.keygen(force=True, secret_seed=12345, encryption_seed=678)\n"
        "else: c.client.keygen(force=True, secret_seed=int.from_bytes(os.urandom(16), 'big'), encryption_seed=int.from_bytes(os.urandom(16), 'big'))\n"
        "print(hashlib.sha256(c.client.keys.serialize()).hexdigest(), flush=True)\n"
    )
    hashes: dict[str, list[str]] = {}
    for mode in ("default", "default", "fixed", "fixed", "random", "random"):
        done = subprocess.run([sys.executable, "-u", "-c", code, mode, str(client_dir)], capture_output=True, text=True, cwd=PROJECT_ROOT)
        hashes.setdefault(mode, []).append(done.stdout.strip())
    return {
        "keyset_sha256_prefix_by_mode": {k: [h[:16] for h in v] for k, v in hashes.items()},
        "default_keygen_differs_across_processes": hashes["default"][0] != hashes["default"][1],
        "fixed_seed_identical_across_processes": hashes["fixed"][0] == hashes["fixed"][1],
        "random_seed_differs_across_processes": hashes["random"][0] != hashes["random"][1],
    }


def encrypt_without_keys(model_id: str = "lr_top20_bits16") -> dict[str, Any]:
    from concrete.ml.deployment import FHEModelClient

    client_dir = _deploy_paths(model_id)["client_dir"]
    n_features = int(json.loads((client_dir / "preprocessing.json").read_text())["n_features"])
    client = FHEModelClient(str(client_dir))
    q = np.asarray(client.model.quantize_input(np.zeros((1, n_features), dtype=np.float64)))
    try:
        client.client.encrypt(q)
        return {"encrypt_succeeded_without_keygen": True, "keys_generated_afterwards": bool(client.client.keys.are_generated)}
    except Exception as exc:  # noqa: BLE001
        return {"encrypt_succeeded_without_keygen": False, "error": f"{type(exc).__name__}: {str(exc)[:200]}"}


def _real_flow(model_id: str) -> dict[str, Any]:
    """One direct (non-HTTP) client->server round trip on the deployed model, returning sizes and both `run` input styles."""
    from concrete import fhe
    from concrete.ml.deployment import FHEModelServer

    from src.client.client import CiphraudClient

    paths = _deploy_paths(model_id)
    manifest = json.loads(paths["manifest"].read_text())
    reference = json.loads(paths["reference"].read_text())
    server_load = time.perf_counter()
    server = FHEModelServer(str(paths["server_dir"]))
    server_load = time.perf_counter() - server_load
    client = CiphraudClient(paths["client_dir"])
    keygen = client.generate_keys()
    ek = client.evaluation_keys()
    position = int(reference["fixed_position"])
    X_val = np.load(PROJECT_ROOT / manifest["source"]["handoff_npz"])["X_val"]
    request = client.prepare_request(X_val[position : position + 1])
    t0 = time.perf_counter()
    response_bytes = server.run(request["ciphertext"], ek["bytes"])
    bytes_run = time.perf_counter() - t0
    value, keys_obj = fhe.Value.deserialize(request["ciphertext"]), fhe.EvaluationKeys.deserialize(ek["bytes"])
    t0 = time.perf_counter()
    response_obj = server.run(value, keys_obj)
    obj_run = time.perf_counter() - t0
    expected = np.asarray(reference["rows"][str(position)]["clear_ints"])
    keyset_path = PROJECT_ROOT / "data" / "phase10" / f"_probe_keyset_{model_id}.bin"
    keyset_path.parent.mkdir(parents=True, exist_ok=True)
    keyset_path.unlink(missing_ok=True)
    client.save_keys(keyset_path)
    keyset_bytes = keyset_path.stat().st_size
    keyset_path.unlink()
    return {
        "server_load_seconds": server_load, "client_keygen_seconds": keygen["keygen_seconds"], "evaluation_keys_bytes": ek["n_bytes"], "saved_keyset_file_bytes": keyset_bytes,
        "input_ciphertext_bytes": request["n_bytes"], "response_bytes": len(response_bytes),
        "circuit_stats_sizes": {k: manifest["circuit_stats"][k] for k in ("size_of_inputs_bytes", "size_of_outputs_bytes", "size_of_secret_keys_bytes", "size_of_bootstrap_keys_bytes", "size_of_keyswitch_keys_bytes")},
        "run_with_bytes_seconds": bytes_run, "run_with_deserialized_objects_seconds": obj_run,
        "preserialized_run_accepts_value_and_evaluation_keys": isinstance(response_obj, fhe.Value),
        "bytes_and_object_results_both_decrypt_to_the_reference": bool(
            np.array_equal(client.finish(response_bytes)["integers"].reshape(-1), expected) and np.array_equal(client.finish(response_obj.serialize())["integers"].reshape(-1), expected)
        ),
    }


def exit_behavior(model_id: str = "lr_top20_bits16") -> dict[str, Any]:
    """Which step of a client/server flow makes a process lose its exit status and unflushed stdout? Each step runs in a FRESH child
    process that ends with `print('HELLO'); sys.exit(3)`. Measured: only RUNNING the circuit (`FHEModelServer.run`) does it."""
    paths = _deploy_paths(model_id)
    d = repr(str(paths["deploy_root"]))
    steps = {
        "control_no_concrete": "pass",
        "load_server": f"from concrete.ml.deployment import FHEModelServer\nS = FHEModelServer({d} + '/server')",
        "load_client_and_keygen": f"from concrete.ml.deployment import FHEModelClient\nC = FHEModelClient({d} + '/client'); C.client.keygen(force=True)",
        "encrypt": f"from concrete.ml.deployment import FHEModelClient\nimport numpy as np\nC = FHEModelClient({d} + '/client'); C.client.keygen(force=True)\ne = C.client.encrypt(np.asarray(C.model.quantize_input(np.zeros((1, 20)))))",
        "server_run": (
            f"from concrete.ml.deployment import FHEModelClient, FHEModelServer\nimport numpy as np\nS = FHEModelServer({d} + '/server'); C = FHEModelClient({d} + '/client'); C.client.keygen(force=True)\n"
            "e = C.client.encrypt(np.asarray(C.model.quantize_input(np.zeros((1, 20)))))\nv = e[0] if isinstance(e, tuple) else e\nS.run(v.serialize(), C.client.evaluation_keys.serialize())"
        ),
    }
    out: dict[str, Any] = {}
    for name, body in steps.items():
        code = f"import sys, warnings; warnings.filterwarnings('ignore')\n{body}\nprint('HELLO')\nsys.exit(3)\n"
        done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, cwd=PROJECT_ROOT)
        out[name] = {"exit_status_requested": 3, "exit_status_observed": done.returncode, "unflushed_stdout_kept": "HELLO" in done.stdout}
    out["status_and_stdout_lost_only_after_running_a_circuit"] = bool(
        all(out[n]["exit_status_observed"] == 3 and out[n]["unflushed_stdout_kept"] for n in ("control_no_concrete", "load_server", "load_client_and_keygen", "encrypt"))
        and out["server_run"]["exit_status_observed"] != 3 and not out["server_run"]["unflushed_stdout_kept"]
    )
    return out


def run_time_ab(model_id: str = "mlp_top20_bits3", n: int = 4) -> dict[str, Any]:
    from concrete import fhe
    from concrete.ml.deployment import FHEModelServer

    from src.client.client import CiphraudClient
    from src.fhe.deploy import load_source

    paths = _deploy_paths(model_id)
    manifest = json.loads(paths["manifest"].read_text())
    reference = json.loads(paths["reference"].read_text())
    position = int(reference["fixed_position"])
    server = FHEModelServer(str(paths["server_dir"]))
    client = CiphraudClient(paths["client_dir"])
    client.generate_keys()
    native = client._fhe.client.evaluation_keys
    deserialized = fhe.EvaluationKeys.deserialize(client._fhe.client.evaluation_keys.serialize())
    value = fhe.Value.deserialize(client.prepare_request(np.load(PROJECT_ROOT / manifest["source"]["handoff_npz"])["X_val"][position : position + 1])["ciphertext"])

    def timed(keys: Any) -> list[float]:
        out = []
        for _ in range(n):
            t0 = time.perf_counter()
            server.run(value, keys)
            out.append(time.perf_counter() - t0)
        return out

    native_s, deserialized_s = timed(native), timed(deserialized)
    source = load_source(load_config(CONFIG), model_id)  # the ORIGINAL in-process path: circuit.keygen / encrypt / run
    circuit = source["circuit"]
    circuit.keygen()
    q = np.asarray(source["cml_model"].quantize_input(source["X_val_model"][position : position + 1]))
    inproc_s = []
    for _ in range(n):
        encrypted = circuit.encrypt(q)
        t0 = time.perf_counter()
        circuit.run(encrypted)
        inproc_s.append(time.perf_counter() - t0)
    steady = lambda xs: float(np.mean(xs[1:]))  # noqa: E731 -- the first call after key preparation is a warm-up
    return {
        "model": model_id, "n_runs_each": n, "native_evaluation_keys_seconds": native_s, "deserialized_evaluation_keys_seconds": deserialized_s, "in_process_circuit_run_seconds": inproc_s,
        "steady_state_mean_seconds": {"native_keys": steady(native_s), "deserialized_keys": steady(deserialized_s), "in_process_circuit_run": steady(inproc_s)},
        "phase9_recorded_fhe_latency_total_mean_seconds": json.loads((PROJECT_ROOT / manifest["source"]["metrics_path"]).read_text())["fhe_latency"]["total"]["mean_seconds"],
        "cpu_count": os.cpu_count(),
    }


def main() -> None:
    warnings.filterwarnings("ignore")
    out_dir = PROJECT_ROOT / "results" / "phase10_client_server"
    out_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {
        "note": "measured facts behind docs/client_server.md; regenerate with `python -m src.fhe.probes.phase10_probe`",
        "keygen_determinism": keygen_determinism(), "encrypt_without_keys": encrypt_without_keys(),
        "wire_and_run": {mid: _real_flow(mid) for mid in ("lr_top20_bits16", "mlp_top20_bits3")},
        "exit_behavior": exit_behavior(), "run_time_ab": run_time_ab(),
    }
    (out_dir / "api_probe.json").write_text(json.dumps(result, indent=2, sort_keys=True, default=str), encoding="utf-8")
    print(json.dumps({k: (v if k != "wire_and_run" else "see file") for k, v in result.items() if k != "run_time_ab"}, indent=2, default=str), flush=True)
    print("run_time_ab steady-state means:", result["run_time_ab"]["steady_state_mean_seconds"], flush=True)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
