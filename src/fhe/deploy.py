"""Phase 10 deployment build (`docs/plan.md` Phase 10; `docs/client_server.md`).

WSL2/Linux only for anything that touches Concrete-ML (`docs/environment.md`); the Concrete imports are lazy so the pure
helpers (`request_positions`, `code_file_hashes`, ...) stay importable and testable everywhere. Run, from the project root:

    ~/.venvs/fhe-fraud-detection/bin/python -m src.fhe.deploy --config configs/phase10/client_server.yaml --model lr_top20_bits16
    ~/.venvs/fhe-fraud-detection/bin/python -m src.fhe.deploy --config configs/phase10/client_server.yaml --model lr_top20_bits16 --verify

This is a COMPILATION-pipeline step, not part of the server: it rebuilds the EXACT circuit Phase 8/9 validated (the same
loaders, handoff, calibration and bit-width -- read from the Phase 8/9 grid config by label, never duplicated), PROVES the
rebuild reproduces the recorded result, and only then persists it with Concrete-ML's own deployment API:

  data/phase10/deploy/<id>/server/server.zip    what the server process loads (and nothing else)
  data/phase10/deploy/<id>/client/client.zip    quantizers + FHE specs, what the client loads
  data/phase10/deploy/<id>/client/preprocessing.json   client-side preprocessing, decision threshold
  results/phase10_client_server/deploy/<id>/manifest.json    identity: hashes, versions, circuit stats, provenance
  results/phase10_client_server/deploy/<id>/reference.json   expected integer outputs + the reproduction proof (runner only)

Nothing is retrained: the MLP is reloaded from its Phase 9 checkpoint after `_fresh` confirms its fingerprint (this module
never calls `train_qat_checkpointed`, which could retrain). A reproduction mismatch stops that model -- an unverified circuit
is never deployed. `--verify` runs in a FRESH process, loads only the persisted artifacts, and (for cheap models) checks a
direct client->server round trip; it exists to prove persistence works without the original model object.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from src.config import PROJECT_ROOT, load_config
from src.data import provenance as data_provenance
from src.logging_setup import get_logger

logger = get_logger(__name__)

LIBRARY_NAMES = ["numpy", "pandas", "scikit-learn", "xgboost", "concrete-ml", "concrete-python", "torch", "brevitas", "fastapi", "uvicorn", "httpx", "PyYAML"]
CIRCUIT_STAT_KEYS = (
    "programmable_bootstrap_count", "size_of_inputs_bytes", "size_of_outputs_bytes",
    "size_of_secret_keys_bytes", "size_of_bootstrap_keys_bytes", "size_of_keyswitch_keys_bytes",
)
CODE_FILES = ["configs/phase10/client_server.yaml", "src/fhe/deploy.py", "src/benchmark/memwatch.py", "src/benchmark/phase10_roundtrip.py"]
CODE_GLOBS = ["src/server/*.py", "src/client/*.py"]


class DeployError(Exception):
    """A model cannot be deployed: its rebuild does not reproduce the validated Phase 8/9 record."""


# ------------------------------------------------------------------------------------------------ pure helpers


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def code_file_paths(root: Path = PROJECT_ROOT) -> list[Path]:
    paths = [root / rel for rel in CODE_FILES]
    for pattern in CODE_GLOBS:
        paths.extend(sorted(root.glob(pattern)))
    return [p for p in paths if p.is_file()]


def _git_blob_sha1(path: Path, root: Path) -> str | None:
    """`git hash-object` of the file as git would store it (after any autocrlf clean filter): equals the blob id the file
    will have once committed, so a later `git rev-parse <commit>:<path>` can confirm the run used the committed code."""
    try:
        out = subprocess.run(["git", "hash-object", str(path)], cwd=root, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 and out.stdout.strip() else None


def code_file_hashes(root: Path = PROJECT_ROOT) -> dict[str, dict[str, str | None]]:
    """Per Phase 10 source file: raw sha256 and the git blob sha1 -- recorded in every Phase 10 provenance.json so the
    results identify the exact code without a pre-run commit (and can be checked against the eventual commit)."""
    return {
        p.relative_to(root).as_posix(): {"sha256": sha256_path(p), "git_blob_sha1": _git_blob_sha1(p, root)}
        for p in code_file_paths(root)
    }


def request_positions(fixed_position: int, t1_positions: list[int], t1_rows: bool, repeat_fixed_row: int) -> list[int]:
    """Validation-row positions of the requests, in order: the fixed row first, then the source phase's other T1 rows
    (if `t1_rows`), then `repeat_fixed_row` cheap repeats of the fixed row."""
    positions = [int(fixed_position)]
    if t1_rows:
        positions += [int(p) for p in t1_positions if int(p) != int(fixed_position)]
    positions += [int(fixed_position)] * int(repeat_fixed_row)
    return positions


def model_config(config: dict[str, Any], model_id: str) -> dict[str, Any]:
    for entry in config["models"]:
        if entry["id"] == model_id:
            return entry
    raise DeployError(f"no model {model_id!r} in the Phase 10 config (have: {[m['id'] for m in config['models']]})")


def grid_entry(grid_config: dict[str, Any], label: str) -> dict[str, Any]:
    for entry in grid_config["configurations"]:
        if entry["label"] == label:
            return entry
    raise DeployError(f"no configuration labeled {label!r} in the grid config")


def paths_for(config: dict[str, Any], model_id: str) -> dict[str, Path]:
    deploy_root = PROJECT_ROOT / config["output"]["deploy_dir"] / model_id
    results_root = PROJECT_ROOT / config["output"]["results_dir"] / "deploy" / model_id
    return {
        "deploy_root": deploy_root, "server_dir": deploy_root / "server", "client_dir": deploy_root / "client",
        "results_root": results_root, "manifest": results_root / "manifest.json", "reference": results_root / "reference.json",
        "verify": results_root / "verify.json",
    }


def _flat(values: Any) -> np.ndarray:
    return np.asarray(values).reshape(-1).astype(np.int64)


# ------------------------------------------------------------------------------------------------ rebuild the validated circuit


def _lr_or_xgb_source(model_cfg: dict[str, Any], grid_config: dict[str, Any], entry: dict[str, Any]) -> dict[str, Any]:
    """LR / XGBoost: the Phase 8 loader (itself `run.py::_load_lr`/`_load_xgboost`) with this label's own n_bits."""
    from src.benchmark.phase8_grid import _load_configuration
    from src.fhe.compile.linear import circuit_stats

    loaded = _load_configuration(entry)
    source_config = loaded["source_config"]
    kind = model_cfg["model_type"]
    if kind == "lr":
        manifest_path = PROJECT_ROOT / source_config["output"]["handoff_manifest"]
        manifest = json.loads(manifest_path.read_text())
        params = json.loads((PROJECT_ROOT / manifest["params_path"]).read_text())
        npz = PROJECT_ROOT / source_config["output"]["handoff_npz"]
        preprocessing = {"kind": "standardize", "scaler_mean": params["scaler_mean"], "scaler_scale": params["scaler_scale"], "dtype": "float64"}
        probability = "post_processing"
    else:
        manifest_path = PROJECT_ROOT / source_config["output"]["dir"] / entry["tier"] / "handoff_manifest.json"
        manifest = json.loads(manifest_path.read_text())
        npz = PROJECT_ROOT / source_config["output"]["handoff_dir"] / f"xgb_{entry['tier']}.npz"
        preprocessing = {"kind": "none"}
        probability = "post_processing"
        # Concrete-ML's tree post-processing sums the trees in the clear only when `_fhe_ensembling` is False (the default);
        # that flag is NOT stored in client.zip, so the deployment silently depends on it. Assert it.
        if getattr(loaded["cml_model"], "_fhe_ensembling", False):
            raise DeployError("XGBoost model has _fhe_ensembling=True; the client-side post-processing would be wrong")
    return {
        "cml_model": loaded["cml_model"], "circuit": loaded["circuit"], "compile_seconds": loaded["compile_seconds"],
        "X_val_model": loaded["X_val_model"], "y_val": loaded["y_val"], "threshold": loaded["threshold"],
        "n_features": int(loaded["n_features"]), "n_bits": int(loaded["n_bits"]), "tier": entry.get("tier"),
        "preprocessing": preprocessing, "probability": probability, "columns": manifest.get("columns"),
        "handoff_npz": npz, "handoff_npz_sha256": manifest["npz_sha256"], "handoff_manifest": manifest_path,
        "circuit_stats": circuit_stats(loaded["circuit"]), "grid_output": grid_config["output"],
        "recorded_dir": PROJECT_ROOT / grid_config["output"]["handoff_dir"] / "checkpoints" / model_cfg["id"],
        "metrics_path": PROJECT_ROOT / grid_config["output"]["dir"] / model_cfg["id"] / "metrics.json",
    }


def _mlp_source(model_cfg: dict[str, Any], grid_config: dict[str, Any], entry: dict[str, Any]) -> dict[str, Any]:
    """MLP: reload the Phase 9 QAT checkpoint and recompile with the same seeded calibration as `run_configuration`. The
    checkpoint's fingerprint must be fresh under the CURRENT code/config -- if it is not, stop; never retrain."""
    from src.benchmark import phase9_grid as p9
    from src.fhe.compile import mlp
    from src.fhe.compile.linear import circuit_stats
    from src.fhe.xgb_poc import calibration_positions

    tier, n_bits, seed = entry["tier"], int(entry["n_bits"]), grid_config["seed"]
    data = p9.load_tier_data(grid_config, tier)
    directory = p9._checkpoint_root(grid_config) / "qat" / f"{tier}_bits{n_bits}_seed{seed}"
    model_path, info_path, logits_path, meta_path = directory / "model.json", directory / "training.json", directory / "val_logits.npy", directory / "meta.json"
    fingerprint = p9._fingerprint(grid_config, data, kind="qat", tier=tier, n_bits=n_bits, seed=seed, calibration=grid_config["compile_calibration"])
    if not p9._fresh(meta_path, fingerprint, [model_path, info_path, logits_path]):
        raise DeployError(f"{model_cfg['id']}: the Phase 9 QAT checkpoint at {directory} is not fresh under the current code/config; refusing to retrain")
    model = mlp.load_model(model_path.read_text())
    reloaded = mlp.clear_quantized_logits(model, data["X_val"])
    t0_ok = bool(np.array_equal(reloaded, np.load(logits_path)))
    cal = data["X_train"][calibration_positions(data["X_train"], grid_config["compile_calibration"]["rows"], grid_config["compile_calibration"]["include_extremes"], seed)]
    circuit, compile_seconds = mlp.compile_mlp(model, cal)
    metrics_path = PROJECT_ROOT / grid_config["output"]["dir"] / model_cfg["id"] / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    scaler = data["scaler"]
    return {
        "cml_model": model, "circuit": circuit, "compile_seconds": compile_seconds, "X_val_model": data["X_val"], "y_val": data["y_val"],
        "threshold": metrics["accuracy"]["gates"]["t1_correctness"]["threshold"], "n_features": data["n_features"], "n_bits": n_bits, "tier": tier,
        "preprocessing": {"kind": "standardize", "scaler_mean": scaler["scaler_mean"], "scaler_scale": scaler["scaler_scale"], "dtype": "float32"},
        "probability": "mlp_softmax", "columns": data["manifest"].get("columns"),
        "handoff_npz": PROJECT_ROOT / grid_config["output"]["handoff_dir"] / f"mlp_{tier}.npz", "handoff_npz_sha256": data["manifest"]["npz_sha256"],
        "handoff_manifest": PROJECT_ROOT / grid_config["output"]["dir"] / "export" / tier / "handoff_manifest.json",
        "circuit_stats": circuit_stats(circuit), "grid_output": grid_config["output"], "t0_reload_equals_saved_logits": t0_ok,
        "recorded_dir": PROJECT_ROOT / grid_config["output"]["handoff_dir"] / "checkpoints" / "fhe" / model_cfg["id"],
        "metrics_path": metrics_path, "reloaded_logits": reloaded,
    }


def load_source(config: dict[str, Any], model_id: str) -> dict[str, Any]:
    model_cfg = model_config(config, model_id)
    grid_config = load_config(model_cfg["grid_config"])
    entry = grid_entry(grid_config, model_id)
    if model_cfg["model_type"] == "mlp":
        source = _mlp_source(model_cfg, grid_config, entry)
    elif model_cfg["model_type"] in ("lr", "xgboost"):
        source = _lr_or_xgb_source(model_cfg, grid_config, entry)
    else:
        raise DeployError(f"unknown model_type {model_cfg['model_type']!r}")
    source.update({"model_cfg": model_cfg, "grid_entry": entry, "kind": model_cfg["model_type"], "id": model_id})
    return source


# ------------------------------------------------------------------------------------------------ reference integers and the reproduction proof


def clear_ints(source: dict[str, Any], q: np.ndarray) -> np.ndarray:
    """Clear-quantized integer outputs (the computation behind `fhe="disable"`) for already-quantized inputs."""
    model = source["cml_model"]
    if source["kind"] == "mlp":
        return np.asarray(model.quantized_module_.quantized_forward(q, fhe="disable"))
    return np.asarray(model._inference(q))


def probabilities(source: dict[str, Any], integers: np.ndarray) -> np.ndarray:
    """P(fraud) from decrypted/clear integers, exactly as the source phase computed it."""
    model = source["cml_model"]
    if source["kind"] == "mlp":
        from src.fhe.compile import mlp

        return mlp.fraud_probability(model.dequantize_output(integers))
    return model.post_processing(model.dequantize_output(integers))[:, 1]


def recorded_ints(source: dict[str, Any], fixed_position: int, t1_positions: list[int]) -> dict[int, np.ndarray]:
    """Decrypted integers the SOURCE phase recorded on disk, keyed by validation-row position (may be empty/partial: LR
    recorded only its latency row)."""
    base: Path = source["recorded_dir"]
    out: dict[int, np.ndarray] = {}
    latency = base / "latency" / "trial_000.npy"
    if latency.exists():
        out[int(fixed_position)] = _flat(np.load(latency))
    for i, position in enumerate(t1_positions):
        path = base / "t1_correctness" / f"row_{i:04d}.npy"
        if path.exists():
            arr = _flat(np.load(path))
            if int(position) in out and not np.array_equal(out[int(position)], arr):
                raise DeployError(f"{source['id']}: recorded T1 row {i} and the latency row disagree for position {position}")
            out[int(position)] = arr
    return out


def reproduction_proof(source: dict[str, Any], positions: list[int], fixed_position: int, t1_positions: list[int]) -> tuple[dict[str, Any], dict[str, Any]]:
    """(proof, per-row expected outputs). Raises nothing itself: `passed` is False if anything fails to reproduce."""
    metrics = json.loads(Path(source["metrics_path"]).read_text())
    stats_expected = {k: metrics["circuit_stats"][k] for k in CIRCUIT_STAT_KEYS}
    stats_actual = {k: source["circuit_stats"][k] for k in CIRCUIT_STAT_KEYS}
    proof: dict[str, Any] = {
        "circuit_stats_expected": stats_expected, "circuit_stats_rebuilt": stats_actual, "circuit_stats_match": stats_expected == stats_actual,
    }
    recorded = recorded_ints(source, fixed_position, t1_positions)
    model, X_model, y_val = source["cml_model"], source["X_val_model"], source["y_val"]
    rows: dict[str, Any] = {}
    all_ok = proof["circuit_stats_match"]
    for position in sorted(set(positions)):
        x = X_model[position : position + 1]
        q = np.asarray(model.quantize_input(x))
        clear_raw = np.asarray(clear_ints(source, q))
        clear = _flat(clear_raw)
        simulated = _flat(source["circuit"].simulate(q))
        row = {
            "y_true": int(y_val[position]), "q_in": q.reshape(-1).tolist(), "clear_ints": clear.tolist(), "simulate_ints": simulated.tolist(),
            "recorded_ints": None if position not in recorded else recorded[position].tolist(),
            "clear_equals_simulate": bool(np.array_equal(clear, simulated)),
            "clear_equals_recorded": None if position not in recorded else bool(np.array_equal(clear, recorded[position])),
            "probability": float(probabilities(source, clear_raw)[0]),
        }
        rows[str(position)] = row
        all_ok = all_ok and row["clear_equals_simulate"] and row["clear_equals_recorded"] is not False
    if source["kind"] == "mlp":
        proof["t0_reload_equals_saved_logits"] = source["t0_reload_equals_saved_logits"]
        all_ok = all_ok and source["t0_reload_equals_saved_logits"]
    if source["kind"] in ("lr", "mlp"):
        from src.train.metrics import threshold_independent_metrics

        if source["kind"] == "mlp":
            from src.fhe.compile import mlp

            prob = mlp.fraud_probability(source["reloaded_logits"])
        else:
            prob = model.predict_proba(X_model, fhe="disable")[:, 1]
        pr_auc = float(threshold_independent_metrics(y_val, prob)["pr_auc"])
        expected = float(metrics["accuracy"]["quantized_full_metrics"]["pr_auc"])
        proof["validation_pr_auc"] = {"rebuilt": pr_auc, "recorded": expected, "match": bool(abs(pr_auc - expected) <= 1e-12)}
        all_ok = all_ok and proof["validation_pr_auc"]["match"]
    else:
        proof["validation_pr_auc"] = {"skipped": "full-validation clear evaluation of the tree ensemble is memory/time-costly; reproduction rests on identical circuit statistics and exact recorded integers"}
    proof["passed"] = bool(all_ok)
    return proof, rows


# ------------------------------------------------------------------------------------------------ persist


def security_level(circuit: Any) -> str | None:
    for owner in (circuit, getattr(circuit, "server", None)):
        configuration = getattr(owner, "configuration", None)
        level = getattr(configuration, "security_level", None)
        if level is not None:
            return str(level)
    return None


def persist_circuit(cml_model: Any, paths: dict[str, Path], via_mlir: bool = False) -> dict[str, Any]:
    """`FHEModelDev.save` into a staging directory, then split into separate server/ and client/ directories so the server
    process reads only server.zip. `via_mlir=False`: the server loads the compiled library directly (no MLIR recompilation
    that could re-derive different crypto parameters); it needs the same Python/Concrete versions, which the version check
    inside the zips enforces."""
    from concrete.ml.deployment import FHEModelDev

    deploy_root = paths["deploy_root"]
    stage = deploy_root / "_stage"
    for directory in (paths["server_dir"], paths["client_dir"]):
        directory.mkdir(parents=True, exist_ok=True)
    FHEModelDev(str(stage), cml_model).save(via_mlir=via_mlir)
    shutil.move(str(stage / "server.zip"), paths["server_dir"] / "server.zip")
    shutil.move(str(stage / "client.zip"), paths["client_dir"] / "client.zip")
    shutil.rmtree(stage)
    return {
        "server_zip": {"sha256": sha256_path(paths["server_dir"] / "server.zip"), "bytes": (paths["server_dir"] / "server.zip").stat().st_size},
        "client_zip": {"sha256": sha256_path(paths["client_dir"] / "client.zip"), "bytes": (paths["client_dir"] / "client.zip").stat().st_size},
        "via_mlir": via_mlir,
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def deploy(config: dict[str, Any], config_path: str, model_id: str, force: bool = False) -> dict[str, Any]:
    from importlib.metadata import version

    paths = paths_for(config, model_id)
    if paths["deploy_root"].exists() and any(paths["deploy_root"].iterdir()):
        if not force:
            raise DeployError(f"{paths['deploy_root']} already holds a deployment; pass --force to rebuild it")
        shutil.rmtree(paths["deploy_root"])
    model_cfg = model_config(config, model_id)
    t0 = time.perf_counter()
    source = load_source(config, model_id)
    metrics = json.loads(Path(source["metrics_path"]).read_text())
    fixed_position = int(metrics["row_selection"]["position"])
    t1_positions = [int(p) for p in metrics["accuracy"]["correctness_sample_positions"]]
    positions = request_positions(fixed_position, t1_positions, bool(model_cfg["t1_rows"]), int(model_cfg["repeat_fixed_row"]))
    logger.info("Reproducing the validated circuit", extra={"extra_fields": {"model": model_id, "positions": sorted(set(positions))}})
    proof, rows = reproduction_proof(source, positions, fixed_position, t1_positions)
    if not proof["passed"]:
        _write_json(paths["results_root"] / "reproduction_failed.json", {"model": model_id, "proof": proof, "rows": rows})
        raise DeployError(f"{model_id}: the rebuilt circuit does not reproduce the recorded Phase 8/9 result (see reproduction_failed.json); nothing deployed")

    artifacts = persist_circuit(source["cml_model"], paths)
    preprocessing = {
        **source["preprocessing"], "model_id": model_id, "model_type": source["kind"], "n_features": source["n_features"],
        "columns": source["columns"], "threshold": float(source["threshold"]), "probability": source["probability"],
    }
    _write_json(paths["client_dir"] / "preprocessing.json", preprocessing)
    artifacts["preprocessing_json"] = {"sha256": sha256_path(paths["client_dir"] / "preprocessing.json")}

    provenance = data_provenance.build_provenance(config=config, config_path=Path(config_path), seed=config["seed"], raw_file_digests={}, library_names=LIBRARY_NAMES)
    provenance["code_files"] = code_file_hashes()
    manifest = {
        "id": model_id, "model_type": source["kind"], "tier": source["tier"], "n_bits": source["n_bits"], "n_features": source["n_features"],
        "artifacts": artifacts, "artifact_paths": {
            "server_zip": (paths["server_dir"] / "server.zip").relative_to(PROJECT_ROOT).as_posix(),
            "client_zip": (paths["client_dir"] / "client.zip").relative_to(PROJECT_ROOT).as_posix(),
            "preprocessing_json": (paths["client_dir"] / "preprocessing.json").relative_to(PROJECT_ROOT).as_posix(),
        },
        "versions": {name: version(name) for name in ("concrete-python", "concrete-ml", "numpy")} | {"python": sys.version.split()[0]},
        "security_level": security_level(source["circuit"]),
        "compile_seconds": source["compile_seconds"], "circuit_stats": source["circuit_stats"],
        "source": {
            "grid_config": model_cfg["grid_config"], "metrics_path": Path(source["metrics_path"]).relative_to(PROJECT_ROOT).as_posix(),
            "metrics_sha256": sha256_path(Path(source["metrics_path"])), "source_status": metrics["status"],
            "handoff_npz": Path(source["handoff_npz"]).relative_to(PROJECT_ROOT).as_posix(), "handoff_npz_sha256": source["handoff_npz_sha256"],
        },
        "t3_disclosure": (model_cfg.get("t3_disclosure") or "").strip() or None,
        "build_seconds": time.perf_counter() - t0, "provenance": provenance, "test_partition_touched": False,
    }
    reference = {
        "id": model_id, "fixed_position": fixed_position, "t1_positions": t1_positions, "request_positions": positions,
        "reproduction_proof": proof, "rows": rows, "threshold": float(source["threshold"]),
    }
    _write_json(paths["manifest"], manifest)
    _write_json(paths["reference"], reference)
    logger.info("Deployment persisted", extra={"extra_fields": {"model": model_id, "server_zip_bytes": artifacts["server_zip"]["bytes"]}})
    return manifest


# ------------------------------------------------------------------------------------------------ fresh-process verification


def verify_persisted(config: dict[str, Any], model_id: str, run_fhe: bool) -> dict[str, Any]:
    """Load ONLY the persisted artifacts (no source model object) in this fresh process and check them: hashes match the
    manifest; server and client load; the loaded server's key/ciphertext sizes equal the rebuilt circuit's; and, for cheap
    models (`run_fhe`), a direct client->server round trip on the fixed row equals the reference integers exactly."""
    from concrete.ml.deployment import FHEModelClient, FHEModelServer

    paths = paths_for(config, model_id)
    manifest = json.loads(paths["manifest"].read_text())
    reference = json.loads(paths["reference"].read_text())
    result: dict[str, Any] = {"id": model_id, "run_fhe": run_fhe}
    result["server_zip_sha256_matches"] = sha256_path(paths["server_dir"] / "server.zip") == manifest["artifacts"]["server_zip"]["sha256"]
    result["client_zip_sha256_matches"] = sha256_path(paths["client_dir"] / "client.zip") == manifest["artifacts"]["client_zip"]["sha256"]
    t0 = time.perf_counter()
    server = FHEModelServer(str(paths["server_dir"]))
    result["server_load_seconds"] = time.perf_counter() - t0
    client = FHEModelClient(str(paths["client_dir"]))
    stats = manifest["circuit_stats"]
    loaded = {
        "size_of_secret_keys_bytes": int(server.server.size_of_secret_keys), "size_of_bootstrap_keys_bytes": int(server.server.size_of_bootstrap_keys),
        "size_of_keyswitch_keys_bytes": int(server.server.size_of_keyswitch_keys),
    }
    result["loaded_server_sizes"] = loaded
    result["loaded_server_sizes_match_manifest"] = all(loaded[k] == stats[k] for k in loaded)
    if run_fhe:
        from src.client.client import CiphraudClient

        cli = CiphraudClient(paths["client_dir"])
        keygen = cli.generate_keys()
        eval_keys = cli.evaluation_keys()
        npz = np.load(PROJECT_ROOT / manifest["source"]["handoff_npz"])
        position = int(reference["fixed_position"])
        request = cli.prepare_request(npz["X_val"][position : position + 1])
        response = server.run(request["ciphertext"], eval_keys["bytes"])
        finished = cli.finish(response)
        expected = np.asarray(reference["rows"][str(position)]["clear_ints"]).reshape(-1)
        result.update({
            "keygen_seconds": keygen["keygen_seconds"], "decrypted_ints": finished["integers"].reshape(-1).tolist(),
            "decrypted_equals_reference": bool(np.array_equal(finished["integers"].reshape(-1), expected)),
        })
    result["passed"] = bool(
        result["server_zip_sha256_matches"] and result["client_zip_sha256_matches"] and result["loaded_server_sizes_match_manifest"]
        and (not run_fhe or result["decrypted_equals_reference"])
    )
    _write_json(paths["verify"], result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/phase10/client_server.yaml")
    parser.add_argument("--model", required=True)
    parser.add_argument("--force", action="store_true", help="rebuild even if a deployment already exists")
    parser.add_argument("--verify", action="store_true", help="fresh-process check of the persisted artifacts (run in a NEW process after building)")
    parser.add_argument("--no-fhe", action="store_true", help="with --verify: skip the direct FHE round trip (XGBoost: 1.9 GB keys, ~44 min)")
    args = parser.parse_args()
    import warnings

    warnings.filterwarnings("ignore")
    config = load_config(args.config)
    if args.verify:
        result = verify_persisted(config, args.model, run_fhe=not args.no_fhe)
        _finish(json.dumps(result, indent=2, default=str), 0 if result["passed"] else 1)
    manifest = deploy(config, args.config, args.model, force=args.force)
    _finish(json.dumps({k: manifest[k] for k in ("id", "artifacts", "circuit_stats", "security_level", "compile_seconds", "build_seconds")}, indent=2, default=str), 0)


def _finish(text: str, code: int) -> None:
    """Print, flush, and exit with the REAL status. Measured (`docs/client_server.md`, `src/fhe/probes/phase10_probe.py`): a process
    that has RUN a circuit in-process (`FHEModelServer.run`) exits with status 0 and drops unflushed stdout no matter what
    `sys.exit(n)` was given -- loading a server, keygen and encrypt do not trigger it -- so a plain `print` + `sys.exit(1)` silently
    reports success. Flushing and `os._exit` skip that shutdown; the JSON written to disk (`verify.json`) is the authoritative record."""
    print(text, flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


if __name__ == "__main__":
    main()
