"""Phase 9 quantized-MLP grid runner (`docs/plan.md` Phase 9; `docs/fhe_mlp.md`).

WSL2/Linux only. Run as, from the project root:

    ~/.venvs/fhe-fraud-detection/bin/python -m src.benchmark.phase9_grid --config configs/phase9/mlp_grid.yaml --label mlp_top20_bits4
    ~/.venvs/fhe-fraud-detection/bin/python -m src.benchmark.phase9_grid --config configs/phase9/mlp_grid.yaml --plaintext   # FR5 + seed stability
    ~/.venvs/fhe-fraud-detection/bin/python -m src.benchmark.phase9_grid --config configs/phase9/mlp_grid.yaml --summary

Per configuration (one tier x one bit-width; one process per configuration, like Phase 8):

  TRAIN   QAT MLP (`src/fhe/compile/mlp.py`) + the float twin (shared by every bit-width of a tier/seed); both are
          checkpointed, so a re-run resumes instead of retraining. The QAT model is written to disk, then RELOADED.
  T0      the reloaded model's clear-quantized outputs equal the trainer's, EXACTLY (decision D3).
  COMPILE + circuit statistics (TLU/bootstrap count, bit-widths, key and ciphertext sizes, compile time).
  T3      clear-quantized QAT model vs the float twin, full validation, Phase 8's bars (decision D2).
  T2      `circuit.simulate` vs the clear integer outputs, exact, on seeded validation rows.
  T1      real encrypt -> run -> decrypt on a 2-row stratified sample, exact.
  LATENCY 5 repeated executions of the same fixed validation row Phase 8 used (position 32,148).

A configuration whose compiled circuit's key material would not fit in memory is NOT attempted under FHE: it is recorded
as `infeasible_key_memory`, with its plaintext gates (T0, T3) and the circuit statistics that show why. Any exception is
recorded in `metrics.json` (`*_failed`); nothing is dropped. Never touches the test partition or any Phase 1-8 file.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from src.benchmark.harness import fhe_round_trip_trials, plaintext_latency_trials
from src.config import PROJECT_ROOT, load_config
from src.data import provenance as data_provenance
from src.fhe.compile import mlp
from src.fhe.handoff import load_handoff, sha256_file, standardize_features
from src.fhe.poc import _stratified_sample_positions
from src.fhe.validate.correctness import integer_output_check, t3_quantization_check
from src.fhe.xgb_poc import calibration_positions, explicit_round_trip, peak_rss_mb, simulate_integers, t2_positions
from src.logging_setup import get_logger
from src.train.metrics import metrics_at_threshold, select_threshold, threshold_independent_metrics

logger = get_logger(__name__)

LIBRARY_NAMES = ["numpy", "pandas", "scikit-learn", "concrete-ml", "concrete-python", "torch", "brevitas", "PyYAML"]
GATES = ("t0", "t3", "t2", "t1_correctness")
TRAINER_VERSION = "phase9-v1"  # bump to invalidate every saved checkpoint if the training protocol changes
EXPORT_CONFIG = "configs/phase9/mlp_export.yaml"


class BenchmarkConfigError(Exception):
    pass


# ------------------------------------------------------------------------------------------------ data + checkpoints


def load_tier_data(config: dict[str, Any], tier: str) -> dict[str, Any]:
    handoff_dir = PROJECT_ROOT / config["output"]["handoff_dir"]
    manifest, arrays = load_handoff(handoff_dir / f"mlp_{tier}.npz", PROJECT_ROOT / config["output"]["dir"] / "export" / tier / "handoff_manifest.json")
    scaler = json.loads((PROJECT_ROOT / manifest["scaler_path"]).read_text())
    if sha256_file(PROJECT_ROOT / manifest["scaler_path"]) != manifest["scaler_sha256"]:
        raise BenchmarkConfigError(f"{tier}: scaler JSON does not match the hash in the handoff manifest")
    return {
        "manifest": manifest, "scaler": scaler,
        "X_train": standardize_features(scaler, arrays["X_train"]).astype(np.float32),
        "X_val": standardize_features(scaler, arrays["X_val"]).astype(np.float32),
        "y_train": arrays["y_train"], "y_val": arrays["y_val"], "val_transaction_ids": arrays["val_transaction_ids"],
        "n_features": int(manifest["n_features"]),
    }


def _fingerprint(config: dict[str, Any], data: dict[str, Any], **parts: Any) -> str:
    payload = {"trainer": TRAINER_VERSION, "mlp": config["mlp"], "npz_sha256": data["manifest"]["npz_sha256"], "scaler_sha256": data["manifest"]["scaler_sha256"], **parts}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()


def _checkpoint_root(config: dict[str, Any]) -> Path:
    return PROJECT_ROOT / config["output"]["handoff_dir"] / "checkpoints"


def _fresh(meta_path: Path, fingerprint: str, files: list[Path]) -> bool:
    return meta_path.exists() and json.loads(meta_path.read_text()).get("fingerprint") == fingerprint and all(f.exists() for f in files)


def train_qat_checkpointed(config: dict[str, Any], data: dict[str, Any], tier: str, n_bits: int, seed: int) -> dict[str, Any]:
    """Train (or reuse) the QAT model for (tier, n_bits, seed). Returns paths + the training record; the caller RELOADS the model."""
    directory = _checkpoint_root(config) / "qat" / f"{tier}_bits{n_bits}_seed{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    model_path, info_path, logits_path, meta_path = directory / "model.json", directory / "training.json", directory / "val_logits.npy", directory / "meta.json"
    fingerprint = _fingerprint(config, data, kind="qat", tier=tier, n_bits=n_bits, seed=seed)
    if _fresh(meta_path, fingerprint, [model_path, info_path, logits_path]):
        logger.info("QAT checkpoint reused", extra={"extra_fields": {"tier": tier, "n_bits": n_bits, "seed": seed}})
        return {"model_path": model_path, "logits_path": logits_path, "info": json.loads(info_path.read_text()), "reused": True}
    meta_path.unlink(missing_ok=True)
    model, info = mlp.train_qat_mlp(config["mlp"], n_bits, data["X_train"], data["y_train"], data["X_val"], data["y_val"], seed)
    model_path.write_text(mlp.dump_model(model), encoding="utf-8")
    np.save(logits_path, mlp.clear_quantized_logits(model, data["X_val"]))
    info_path.write_text(json.dumps(info, indent=2, sort_keys=True, default=str), encoding="utf-8")
    meta_path.write_text(json.dumps({"fingerprint": fingerprint}))
    logger.info("QAT model trained", extra={"extra_fields": {"tier": tier, "n_bits": n_bits, "seed": seed, "epochs": info["epochs_run"], "best_epoch": info["best_epoch"]}})
    return {"model_path": model_path, "logits_path": logits_path, "info": info, "reused": False}


def train_float_twin_checkpointed(config: dict[str, Any], data: dict[str, Any], tier: str, seed: int) -> dict[str, Any]:
    """The float twin depends on (tier, seed) only, so it is shared by every bit-width of that tier."""
    directory = _checkpoint_root(config) / "float_twin" / f"{tier}_seed{seed}"
    directory.mkdir(parents=True, exist_ok=True)
    info_path, logits_path, meta_path = directory / "training.json", directory / "val_logits.npy", directory / "meta.json"
    fingerprint = _fingerprint(config, data, kind="float_twin", tier=tier, seed=seed)
    if _fresh(meta_path, fingerprint, [info_path, logits_path]):
        return {"logits": np.load(logits_path), "info": json.loads(info_path.read_text()), "reused": True}
    meta_path.unlink(missing_ok=True)
    net, info = mlp.train_float_twin(config["mlp"], data["X_train"], data["y_train"], data["X_val"], data["y_val"], seed)
    logits = mlp.float_logits(net, data["X_val"])
    np.save(logits_path, logits)
    info_path.write_text(json.dumps(info, indent=2, sort_keys=True, default=str), encoding="utf-8")
    meta_path.write_text(json.dumps({"fingerprint": fingerprint}))
    logger.info("Float twin trained", extra={"extra_fields": {"tier": tier, "seed": seed, "epochs": info["epochs_run"], "best_epoch": info["best_epoch"]}})
    return {"logits": logits, "info": info, "reused": False}


def _clear_integers(model: Any, q: np.ndarray, chunk: int) -> np.ndarray:
    return np.ascontiguousarray(np.concatenate([np.asarray(model.quantized_module_.quantized_forward(q[s : s + chunk], fhe="disable")) for s in range(0, q.shape[0], chunk)]))


def _probabilities(model: Any, integers: np.ndarray) -> np.ndarray:
    return mlp.fraud_probability(model.dequantize_output(integers))


def _full_metrics(y_val: np.ndarray, prob: np.ndarray, threshold: float) -> dict[str, Any]:
    return {**threshold_independent_metrics(y_val, prob), **metrics_at_threshold(y_val, prob, threshold)}


# ------------------------------------------------------------------------------------------------ plaintext evaluation (FR5)


def plaintext_evaluation(config: dict[str, Any], tier: str, n_bits: int, seed: int) -> dict[str, Any]:
    """FR5: the quantized MLP as a plaintext baseline, plus its float twin, on validation (no FHE)."""
    data = load_tier_data(config, tier)
    qat = train_qat_checkpointed(config, data, tier, n_bits, seed)
    twin = train_float_twin_checkpointed(config, data, tier, seed)
    qat_prob = mlp.fraud_probability(np.load(qat["logits_path"]))
    float_prob = mlp.fraud_probability(twin["logits"])
    y_val = data["y_val"]
    qat_threshold = select_threshold(y_val, qat_prob)["selected_threshold"]
    float_threshold = select_threshold(y_val, float_prob)["selected_threshold"]
    return {
        "tier": tier, "n_bits": n_bits, "seed": seed, "n_features": data["n_features"], "test_partition_touched": False,
        "qat_clear_quantized": {**_full_metrics(y_val, qat_prob, qat_threshold), "epochs_run": qat["info"]["epochs_run"], "best_epoch": qat["info"]["best_epoch"]},
        "float_twin": {**_full_metrics(y_val, float_prob, float_threshold), "epochs_run": twin["info"]["epochs_run"], "best_epoch": twin["info"]["best_epoch"]},
        "qat_minus_float_pr_auc": float(threshold_independent_metrics(y_val, qat_prob)["pr_auc"] - threshold_independent_metrics(y_val, float_prob)["pr_auc"]),
    }


def run_plaintext(config: dict[str, Any], n_bits: int) -> dict[str, Any]:
    """Per tier: QAT at `n_bits` and its float twin for every configured seed, with the spread across seeds."""
    out_dir = PROJECT_ROOT / config["output"]["dir"] / "plaintext"
    summary: dict[str, Any] = {"n_bits": n_bits, "seeds": config["seed_stability"]["seeds"], "tiers": {}}
    for tier in ("top_20", "top_50", "top_100"):
        runs = [plaintext_evaluation(config, tier, n_bits, seed) for seed in config["seed_stability"]["seeds"]]
        qat, flt = np.array([r["qat_clear_quantized"]["pr_auc"] for r in runs]), np.array([r["float_twin"]["pr_auc"] for r in runs])
        summary["tiers"][tier] = {
            "runs": runs,
            "qat_pr_auc": {"mean": float(qat.mean()), "std": float(qat.std(ddof=1)), "values": qat.tolist()},
            "float_pr_auc": {"mean": float(flt.mean()), "std": float(flt.std(ddof=1)), "values": flt.tolist()},
        }
        (out_dir / tier).mkdir(parents=True, exist_ok=True)
        (out_dir / tier / "metrics.json").write_text(json.dumps(summary["tiers"][tier], indent=2, sort_keys=True), encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


# ------------------------------------------------------------------------------------------------ one grid configuration


def run_configuration(config: dict[str, Any], config_path: str, label: str) -> dict[str, Any]:
    entry = next((e for e in config["configurations"] if e["label"] == label), None)
    if entry is None:
        raise BenchmarkConfigError(f"No configuration labeled {label!r} in {config_path}")
    tier, n_bits, seed = entry["tier"], int(entry["n_bits"]), config["seed"]
    output_dir = PROJECT_ROOT / config["output"]["dir"]
    metrics: dict[str, Any] = {"label": label, "model_type": "mlp", "tier": tier, "n_bits": n_bits, "seed": seed}

    def finish(status: str, error: dict[str, Any] | None = None, **extra: Any) -> dict[str, Any]:
        metrics["status"] = status
        metrics.update(extra)
        if error:
            metrics["error"] = error
        config_dir = output_dir / label
        config_dir.mkdir(parents=True, exist_ok=True)
        (config_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True, default=str), encoding="utf-8")
        provenance = data_provenance.build_provenance(config=config, config_path=Path(config_path), seed=seed, raw_file_digests={}, library_names=LIBRARY_NAMES)
        (config_dir / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True, default=str), encoding="utf-8")
        logger.info("Phase 9 configuration finished", extra={"extra_fields": {"label": label, "status": status}})
        return metrics

    def failed(stage: str, exc: Exception) -> dict[str, Any]:
        return finish(f"{stage}_failed", {"stage": stage, "type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()})

    # ---- train / reload / T0
    stage = "train"
    try:
        data = load_tier_data(config, tier)
        qat = train_qat_checkpointed(config, data, tier, n_bits, seed)
        twin = train_float_twin_checkpointed(config, data, tier, seed)
        model = mlp.load_model(qat["model_path"].read_text())
        reloaded_logits = mlp.clear_quantized_logits(model, data["X_val"])
        saved_logits = np.load(qat["logits_path"])
        diff = np.abs(reloaded_logits - saved_logits)
        report: dict[str, Any] = {"t0": {"reload_reproduces_clear_quantized_logits_exactly": bool(np.array_equal(reloaded_logits, saved_logits)), "max_abs_diff": float(diff.max()), "n_rows": int(diff.shape[0]), "passed": bool(np.array_equal(reloaded_logits, saved_logits))}}
    except Exception as exc:
        return failed(stage, exc)

    X_val, y_val, X_train = data["X_val"], data["y_val"], data["X_train"]
    metrics.update({
        "n_features": data["n_features"], "test_partition_touched": False,
        "source_config_hash": data_provenance.compute_config_hash(load_config(EXPORT_CONFIG)),
        "training": {"qat": qat["info"], "float_twin": twin["info"], "qat_checkpoint_reused": qat["reused"], "float_twin_checkpoint_reused": twin["reused"]},
    })

    # ---- compile
    stage = "compile"
    try:
        cal = X_train[calibration_positions(X_train, config["compile_calibration"]["rows"], config["compile_calibration"]["include_extremes"], seed)]
        circuit, compile_seconds = mlp.compile_mlp(model, cal)
        stats = mlp.mlp_stats(model, circuit, config["mlp"], n_bits)
    except Exception as exc:
        return failed(stage, exc)
    key_gb = (stats["size_of_bootstrap_keys_bytes"] + stats["size_of_keyswitch_keys_bytes"] + stats["size_of_secret_keys_bytes"]) / 1e9
    # The training matrix is not needed again (calibration already taken); freeing it keeps peak RSS lower while the keys
    # (up to 1.6 GB at 4 bits) are held -- the top_100 probe peaked at 3.4 GB of the 3.8 GiB WSL memory.
    del X_train
    data.pop("X_train", None)
    gc.collect()
    metrics.update({
        "compile_seconds": compile_seconds, "circuit_stats": stats, "key_material_gb": key_gb,
        "ciphertext_size_bytes": {"input": stats["size_of_inputs_bytes"], "output": stats["size_of_outputs_bytes"]},
        "key_size_bytes": {"secret": stats["size_of_secret_keys_bytes"], "bootstrap": stats["size_of_bootstrap_keys_bytes"], "keyswitch": stats["size_of_keyswitch_keys_bytes"]},
        "peak_rss_mb_after_compile": peak_rss_mb(),
    })

    # ---- T3 (plaintext; needs no keys)
    stage = "accuracy"
    try:
        qat_prob, float_prob = mlp.fraud_probability(reloaded_logits), mlp.fraud_probability(twin["logits"])
        threshold = select_threshold(y_val, float_prob)["selected_threshold"]
        tol = config["tolerances"]
        report["t3"] = t3_quantization_check(qat_prob, float_prob, y_val, threshold, tol["t3_min_decision_agreement"], tol["t3_max_pr_auc_drop"])
        quantized_metrics, float_metrics = _full_metrics(y_val, qat_prob, threshold), _full_metrics(y_val, float_prob, threshold)
        own = select_threshold(y_val, qat_prob)
        metrics["accuracy"] = {"gates": report, "quantized_full_metrics": quantized_metrics, "float_full_metrics": float_metrics, "qat_own_f1_threshold": own}
    except Exception as exc:
        return failed(stage, exc)

    max_key_gb = float(config.get("max_key_material_gb", 2.4))
    if key_gb > max_key_gb:
        # a real finding, not a crash: the compiled circuit's keys cannot be held in this machine's memory
        return finish("infeasible_key_memory", None, key_memory_limit_gb=max_key_gb, infeasible_reason=f"key material {key_gb:.2f} GB exceeds the {max_key_gb} GB limit for the 3.8 GiB WSL memory; T2/T1/latency not run", peak_rss_mb=peak_rss_mb(),
                      gates_passed={"t0": report["t0"]["passed"], "t3": report["t3"]["passed"], "t2": None, "t1_correctness": None}, all_gates_passed=False)

    # ---- T2 + T1 (FHE)
    try:
        stage = "accuracy_fhe"
        directory = _checkpoint_root(config) / "fhe" / label
        base = _fingerprint(config, data, kind="fhe", tier=tier, n_bits=n_bits, seed=seed)
        chunk = int(config["disable_chunk_size"])
        t2_cfg = config["t2"]
        t2_pos = t2_positions(X_val.shape[0], t2_cfg["rows"], seed)
        q_t2 = model.quantize_input(X_val[t2_pos])  # only the rows that are simulated are quantized
        disable_q = _clear_integers(model, q_t2, chunk)
        sim_q, seconds_per_row = simulate_integers(circuit, q_t2, directory / "t2_simulate", hashlib.sha256(f"{base}|t2|{t2_cfg['rows']}|{t2_cfg['chunk_size']}".encode()).hexdigest(), int(t2_cfg["chunk_size"]))
        report["t2"] = integer_output_check(sim_q, disable_q, candidate_prob=_probabilities(model, sim_q), reference_prob=_probabilities(model, disable_q), threshold=threshold)
        report["t2"]["simulate_seconds_per_row"] = seconds_per_row

        sample_pos = _stratified_sample_positions(y_val, config["correctness_sample"]["n"], config["correctness_sample"]["n_positive"], seed)
        q_sample = model.quantize_input(X_val[sample_pos])
        sample_sim_q = np.array([circuit.simulate(q_sample[i : i + 1])[0] for i in range(q_sample.shape[0])])
        sample_clear_q = _clear_integers(model, q_sample, chunk)
        decrypted_q, round_trip_timing = explicit_round_trip(circuit, q_sample, directory / "t1_correctness", hashlib.sha256(f"{base}|t1".encode()).hexdigest())
        report["t1_correctness"] = integer_output_check(decrypted_q, sample_sim_q, candidate_prob=_probabilities(model, decrypted_q), reference_prob=_probabilities(model, sample_sim_q), threshold=threshold)
        report["t1_correctness"]["decrypted_vs_disable_exact_integer_match"] = bool(np.array_equal(decrypted_q, sample_clear_q))
        report["t1_correctness"]["round_trip_timing"] = round_trip_timing
        metrics["accuracy"]["correctness_sample_positions"] = [int(p) for p in sample_pos]
    except Exception as exc:
        return failed(stage, exc)

    # ---- latency: 5 repeated executions of one fixed row (the row Phase 8 used)
    stage = "latency"
    try:
        row_cfg = config.get("row_selection", {})
        row_position = int(_stratified_sample_positions(y_val, n=1, n_positive=row_cfg.get("n_positive", 0), seed=seed)[0])
        x_row = X_val[row_position : row_position + 1]
        q_row = model.quantize_input(x_row)
        plaintext_latency = plaintext_latency_trials(lambda: mlp.fraud_probability(mlp.clear_quantized_logits(model, x_row)), config["latency_trials"])
        latency_fp = hashlib.sha256(f"{base}|latency|{row_position}".encode()).hexdigest()
        fhe_latency = fhe_round_trip_trials(circuit, q_row, config["latency_trials"], directory / "latency", latency_fp)
    except Exception as exc:
        return failed(stage, exc)
    metrics.update({
        "row_selection": {"position": row_position, "y_true": int(y_val[row_position]), "n_positive": row_cfg.get("n_positive", 0), "seed": seed},
        "plaintext_latency": plaintext_latency, "fhe_latency": fhe_latency, "peak_rss_mb": peak_rss_mb(),
    })
    all_passed = all(report[g]["passed"] for g in GATES)
    return finish("passed" if all_passed else "failed_accuracy_gates", None, gates_passed={g: report[g]["passed"] for g in GATES}, all_gates_passed=all_passed)


def write_summary(config: dict[str, Any]) -> dict[str, Any]:
    base = PROJECT_ROOT / config["output"]["dir"]
    rows = []
    for entry in config["configurations"]:
        path = base / entry["label"] / "metrics.json"
        if not path.exists():
            rows.append({"label": entry["label"], "status": "not_run"})
            continue
        m = json.loads(path.read_text())
        row = {k: m.get(k) for k in ("label", "model_type", "tier", "n_bits", "status", "all_gates_passed", "compile_seconds", "peak_rss_mb", "key_material_gb")}
        if m.get("accuracy"):
            row["quantized_pr_auc"] = m["accuracy"]["quantized_full_metrics"]["pr_auc"]
            row["float_pr_auc"] = m["accuracy"]["float_full_metrics"]["pr_auc"]
            row["t3_decision_agreement"] = m["accuracy"]["gates"]["t3"]["decision_agreement"]
        if m.get("circuit_stats"):
            row["programmable_bootstraps"] = m["circuit_stats"]["programmable_bootstrap_count"]
        if m.get("fhe_latency"):
            row["fhe_mean_seconds"] = m["fhe_latency"]["total"]["mean_seconds"]
            row["fhe_std_seconds"] = m["fhe_latency"]["total"]["std_seconds"]
        rows.append(row)
    summary = {"configurations": rows, "all_run": all(r["status"] != "not_run" for r in rows), "all_passed": all(bool(r.get("all_gates_passed")) for r in rows)}
    base.mkdir(parents=True, exist_ok=True)
    (base / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/phase9/mlp_grid.yaml")
    parser.add_argument("--label", action="append", help="configuration label to run (repeatable); defaults to every configured label")
    parser.add_argument("--summary", action="store_true")
    parser.add_argument("--plaintext", action="store_true", help="FR5 plaintext evaluation + seed stability (no FHE)")
    parser.add_argument("--plaintext-bits", type=int, help="bit-width for --plaintext (default: the smallest configured bit-width)")
    args = parser.parse_args()
    import warnings

    warnings.filterwarnings("ignore")
    config = load_config(args.config)
    if args.summary:
        write_summary(config)
        return
    if args.plaintext:
        bits = args.plaintext_bits or int(config["seed_stability"]["n_bits"])
        run_plaintext(config, bits)
        return
    failures = []
    for label in args.label or [e["label"] for e in config["configurations"]]:
        result = run_configuration(config, args.config, label)
        if result["status"] not in ("passed",):
            failures.append((label, result["status"]))
    write_summary(config)
    if failures:
        logger.info("Phase 9 configurations not passing (recorded, not hidden)", extra={"extra_fields": {"failures": failures}})


if __name__ == "__main__":
    main()
