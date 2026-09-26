"""Phase 8 research-grid runner (`docs/plan.md` Phase 8; see `docs/research.md`
for the full methodology once written).

WSL2/Linux only. Run as, from the project root:

    ~/.venvs/fhe-fraud-detection/bin/python -m src.benchmark.phase8_grid --config configs/phase8/research_grid.yaml
    ~/.venvs/fhe-fraud-detection/bin/python -m src.benchmark.phase8_grid --config configs/phase8/research_grid.yaml --label lr_top20_bits8
    ~/.venvs/fhe-fraud-detection/bin/python -m src.benchmark.phase8_grid --config configs/phase8/research_grid.yaml --summary

This module composes -- never modifies -- three already-existing pieces:
  1. `src/benchmark/run.py::_load_lr`/`_load_xgboost` (Phase 7, extended with
     an optional `n_bits` override) to rebuild + compile an already-
     committed Phase 5/6 configuration at either bit-width.
  2. `src/benchmark/harness.py::fhe_round_trip_trials`/`plaintext_latency_trials`
     (Phase 7) for the "5 REPEATED EXECUTIONS OF A FIXED REPRESENTATIVE
     INPUT" latency measurement -- explicitly not described as 5
     independent samples (approved plan, clarification 1).
  3. Each model's OWN already-validated Phase 5/6 correctness methodology
     (`src/fhe/poc.py`, `src/fhe/xgb_poc.py`, `src/fhe/validate/correctness.py`)
     for the SEPARATE 2-row stratified correctness sample and the full-val
     T0/T2/T3 accuracy gates (clarification 2: kept structurally distinct
     from the latency trial count throughout this file).

Every real encrypt->run->decrypt call (both the latency trials and the
correctness sample) is checkpointed to disk, because a single XGBoost trial
can cost 30-50 minutes at these bit-widths (docs/research.md) -- a run
interrupted mid-grid resumes at the next un-executed trial/row/tier rather
than restarting, and re-invoking this module for an already-completed
configuration is a fast no-op (its `metrics.json` already exists).

Never touches the test partition and never modifies any Phase 1-7 file.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import traceback
from pathlib import Path
from typing import Any

import numpy as np

from src.benchmark.harness import fhe_round_trip_trials, plaintext_latency_trials
from src.benchmark.run import BenchmarkConfigError, _load_lr, _load_xgboost
from src.config import PROJECT_ROOT, load_config
from src.data import provenance as data_provenance
from src.fhe.compile.linear import circuit_stats, compile_model  # noqa: F401  (compile_model used inside _load_*)
from src.fhe.poc import _explicit_round_trip, _stratified_sample_positions
from src.fhe.validate.correctness import (
    integer_output_check,
    t0_transfer_check,
    t1_execution_check,
    t2_simulation_check,
    t3_quantization_check,
)
from src.fhe.xgb_poc import _probabilities, calibration_positions, disable_integers, explicit_round_trip, peak_rss_mb, simulate_integers, t2_positions
from src.logging_setup import get_logger
from src.train.metrics import metrics_at_threshold, threshold_independent_metrics

logger = get_logger(__name__)

LIBRARY_NAMES = ["numpy", "pandas", "scikit-learn", "xgboost", "concrete-ml", "concrete-python", "PyYAML"]
GATES = ("t0", "t3", "t2", "t1_correctness")


class Phase8GateFailure(Exception):
    """Raised after all measurements for a configuration are written, if
    a required accuracy gate failed (mirrors `src.fhe.xgb_poc.XGBGateFailure`)."""


def _load_configuration(entry: dict[str, Any]) -> dict[str, Any]:
    """Dispatch to `_load_lr`/`_load_xgboost` with this entry's `n_bits`
    override -- the ONLY place a Phase 8 entry's bit-width takes effect."""
    source_config = load_config(entry["source_config"])
    model_type = entry["model_type"]
    n_bits = entry["n_bits"]

    if model_type == "lr":
        loaded = _load_lr(source_config, n_bits=n_bits)
    elif model_type == "xgboost":
        if "tier" not in entry:
            raise BenchmarkConfigError(f"{entry['label']}: model_type 'xgboost' requires a 'tier'")
        loaded = _load_xgboost(source_config, entry["tier"], n_bits=n_bits)
    else:
        raise BenchmarkConfigError(f"{entry['label']}: unknown model_type {model_type!r}")
    loaded["source_config"] = source_config
    return loaded


def _accuracy_lr(loaded: dict[str, Any], config: dict[str, Any], checkpoint_dir: Path) -> dict[str, Any]:
    """T0/T2/T3/T1-correctness for LR, reusing `src/fhe/poc.py`'s exact
    methodology (full-val simulate for T2 -- cheap, zero PBS; probability-
    level exact-match for T1, `src/fhe/validate/correctness.py::t1_execution_check`)."""
    cml_model, X_val, y_val = loaded["cml_model"], loaded["X_val"], loaded["y_val"]
    X_val_model = loaded["X_val_model"]  # standardized inputs: what the compiled LR actually consumes
    reference_val_prob, threshold = loaded["reference_val_prob"], loaded["threshold"]
    tol = config["tolerances"]
    report: dict[str, Any] = {}

    report["t0"] = t0_transfer_check(loaded["plaintext_predict"](X_val), reference_val_prob, tol["t0_max_abs_diff"])

    disable_prob = cml_model.predict_proba(X_val_model, fhe="disable")[:, 1]
    report["t3"] = t3_quantization_check(
        quantized_prob=disable_prob, float_prob=reference_val_prob, y_true=y_val, threshold=threshold,
        min_decision_agreement=tol["t3_min_decision_agreement"], max_pr_auc_drop=tol["t3_max_pr_auc_drop"],
    )

    simulate_prob_full = cml_model.predict_proba(X_val_model, fhe="simulate")[:, 1]
    report["t2"] = t2_simulation_check(simulate_prob_full, disable_prob)

    sample_pos = _stratified_sample_positions(y_val, config["correctness_sample"]["n"], config["correctness_sample"]["n_positive"], config["seed"])
    X_sample = X_val_model[sample_pos]
    simulate_prob_sample = cml_model.predict_proba(X_sample, fhe="simulate")[:, 1]
    decrypted_prob_sample, round_trip_timing = _explicit_round_trip(cml_model, loaded["circuit"], X_sample)
    report["t1_correctness"] = t1_execution_check(decrypted_prob_sample, simulate_prob_sample)
    report["t1_correctness"]["round_trip_timing"] = round_trip_timing

    quantized_metrics = {**threshold_independent_metrics(y_val, disable_prob), **metrics_at_threshold(y_val, disable_prob, threshold)}
    float_metrics = {**threshold_independent_metrics(y_val, reference_val_prob), **metrics_at_threshold(y_val, reference_val_prob, threshold)}
    return report, quantized_metrics, float_metrics, sample_pos


def _accuracy_xgboost(loaded: dict[str, Any], config: dict[str, Any], checkpoint_dir: Path, fingerprint_base: str) -> dict[str, Any]:
    """T0/T2/T3/T1-correctness for XGBoost, reusing `src/fhe/xgb_poc.py`'s
    exact methodology (5000-row seeded T2 sample, integer-level exact-match,
    checkpointed simulate/execute -- both real per-row FHE calls, so both
    are checkpointed)."""
    cml_model, X_val, y_val = loaded["cml_model"], loaded["X_val"], loaded["y_val"]
    reference_val_prob, threshold = loaded["reference_val_prob"], loaded["threshold"]
    tol = config["tolerances"]
    report: dict[str, Any] = {}

    report["t0"] = t0_transfer_check(loaded["plaintext_predict"](X_val), reference_val_prob, tol["t0_max_abs_diff"])

    q_val = cml_model.quantize_input(X_val)
    disable_q = disable_integers(cml_model, q_val, config["disable_chunk_size"])
    disable_prob = _probabilities(cml_model, disable_q)
    report["t3"] = t3_quantization_check(
        quantized_prob=disable_prob, float_prob=reference_val_prob, y_true=y_val, threshold=threshold,
        min_decision_agreement=tol["t3_min_decision_agreement"], max_pr_auc_drop=tol["t3_max_pr_auc_drop"],
    )

    t2_cfg = config["t2"]
    t2_pos = t2_positions(X_val.shape[0], t2_cfg["rows"], config["seed"])
    t2_fingerprint = hashlib.sha256(f"{fingerprint_base}|t2|{t2_cfg['rows']}|{t2_cfg['chunk_size']}".encode()).hexdigest()
    sim_q, seconds_per_row = simulate_integers(loaded["circuit"], q_val[t2_pos], checkpoint_dir / "t2_simulate", t2_fingerprint, t2_cfg["chunk_size"])
    report["t2"] = integer_output_check(sim_q, disable_q[t2_pos], candidate_prob=_probabilities(cml_model, sim_q), reference_prob=disable_prob[t2_pos], threshold=threshold)
    report["t2"]["simulate_seconds_per_row"] = seconds_per_row

    sample_pos = _stratified_sample_positions(y_val, config["correctness_sample"]["n"], config["correctness_sample"]["n_positive"], config["seed"])
    q_sample = q_val[sample_pos]
    sample_sim_q = np.array([loaded["circuit"].simulate(q_sample[i : i + 1])[0] for i in range(q_sample.shape[0])])
    correctness_fingerprint = hashlib.sha256(f"{fingerprint_base}|t1_correctness".encode()).hexdigest()
    decrypted_q, round_trip_timing = explicit_round_trip(loaded["circuit"], q_sample, checkpoint_dir / "t1_correctness", correctness_fingerprint)
    decrypted_prob = _probabilities(cml_model, decrypted_q)
    report["t1_correctness"] = integer_output_check(decrypted_q, sample_sim_q, candidate_prob=decrypted_prob, reference_prob=_probabilities(cml_model, sample_sim_q), threshold=threshold)
    report["t1_correctness"]["decrypted_vs_disable_exact_integer_match"] = bool(np.array_equal(decrypted_q, disable_q[sample_pos]))
    report["t1_correctness"]["round_trip_timing"] = round_trip_timing

    quantized_metrics = {**threshold_independent_metrics(y_val, disable_prob), **metrics_at_threshold(y_val, disable_prob, threshold)}
    float_metrics = {**threshold_independent_metrics(y_val, reference_val_prob), **metrics_at_threshold(y_val, reference_val_prob, threshold)}
    return report, quantized_metrics, float_metrics, sample_pos


def run_configuration(config: dict[str, Any], config_path: str, label: str) -> dict[str, Any]:
    entry = next((e for e in config["configurations"] if e["label"] == label), None)
    if entry is None:
        raise BenchmarkConfigError(f"No configuration labeled {label!r} in {config_path}")

    output_dir = PROJECT_ROOT / config["output"]["dir"]
    handoff_dir = PROJECT_ROOT / config["output"]["handoff_dir"]
    seed = config["seed"]
    metrics: dict[str, Any] = {"label": label, "model_type": entry["model_type"], "tier": entry.get("tier"), "n_bits": entry["n_bits"], "seed": seed}

    def finish(status: str, error: dict[str, Any] | None = None, **extra) -> dict[str, Any]:
        metrics["status"] = status
        metrics.update(extra)
        if error:
            metrics["error"] = error
        config_dir = output_dir / label
        config_dir.mkdir(parents=True, exist_ok=True)
        (config_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True, default=str), encoding="utf-8")
        provenance = data_provenance.build_provenance(config=config, config_path=Path(config_path), seed=seed, raw_file_digests={}, library_names=LIBRARY_NAMES)
        (config_dir / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True, default=str), encoding="utf-8")
        logger.info("Phase 8 configuration finished", extra={"extra_fields": {"label": label, "status": status}})
        return metrics

    stage = "load_and_compile"
    try:
        loaded = _load_configuration(entry)
    except Exception as exc:  # an infeasible configuration -- recorded, never silently dropped
        return finish(f"{stage}_failed", {"stage": stage, "type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()})

    stats = {**circuit_stats(loaded["circuit"]), **loaded["extra_stats"]}
    metrics.update({
        "n_features": loaded["n_features"], "compile_seconds": loaded["compile_seconds"], "circuit_stats": stats,
        "ciphertext_size_bytes": {"input": stats["size_of_inputs_bytes"], "output": stats["size_of_outputs_bytes"]},
        "key_size_bytes": {"secret": stats["size_of_secret_keys_bytes"], "bootstrap": stats["size_of_bootstrap_keys_bytes"], "keyswitch": stats["size_of_keyswitch_keys_bytes"]},
        "peak_rss_mb_after_compile": peak_rss_mb(), "test_partition_touched": False,
        "source_config_hash": data_provenance.compute_config_hash(loaded["source_config"]),
    })

    fingerprint_base = f"{label}|{entry['n_bits']}|{seed}"
    if entry["model_type"] == "lr":
        # LR checkpoints written before the scaler fix belong to a different (wrong)
        # circuit; a distinct fingerprint makes them stale instead of silently reused.
        # XGBoost's fingerprint is deliberately unchanged, so its checkpoints stay valid.
        fingerprint_base += "|standardized-input-v2"
    checkpoint_dir = handoff_dir / "checkpoints" / label

    stage = "accuracy"
    try:
        if entry["model_type"] == "lr":
            report, quantized_metrics, float_metrics, sample_pos = _accuracy_lr(loaded, config, checkpoint_dir)
        else:
            report, quantized_metrics, float_metrics, sample_pos = _accuracy_xgboost(loaded, config, checkpoint_dir, fingerprint_base)
    except Exception as exc:
        return finish(f"{stage}_failed", {"stage": stage, "type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()})
    metrics["accuracy"] = {"gates": report, "quantized_full_metrics": quantized_metrics, "float_full_metrics": float_metrics, "correctness_sample_positions": [int(p) for p in sample_pos]}

    stage = "latency"
    row_pos_cfg = config.get("row_selection", {})
    row_position = int(_stratified_sample_positions(loaded["y_val"], n=1, n_positive=row_pos_cfg.get("n_positive", 0), seed=seed)[0])
    X_row = loaded["X_val"][row_position : row_position + 1]
    q_row = loaded["cml_model"].quantize_input(loaded["X_val_model"][row_position : row_position + 1])
    try:
        plaintext_latency = plaintext_latency_trials(lambda: loaded["plaintext_predict"](X_row), config["latency_trials"])
        latency_fingerprint = hashlib.sha256(f"{fingerprint_base}|latency|{row_position}".encode()).hexdigest()
        fhe_latency = fhe_round_trip_trials(loaded["circuit"], q_row, config["latency_trials"], checkpoint_dir / "latency", latency_fingerprint)
    except Exception as exc:
        return finish(f"{stage}_failed", {"stage": stage, "type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()})
    metrics["row_selection"] = {"position": row_position, "y_true": int(loaded["y_val"][row_position]), "n_positive": row_pos_cfg.get("n_positive", 0), "seed": seed}
    metrics["plaintext_latency"] = plaintext_latency
    metrics["fhe_latency"] = fhe_latency
    metrics["peak_rss_mb"] = peak_rss_mb()

    all_gates_passed = all(report[g]["passed"] for g in GATES)
    return finish("passed" if all_gates_passed else "failed_accuracy_gates", gates_passed={g: report[g]["passed"] for g in GATES}, all_gates_passed=all_gates_passed)


def write_summary(config: dict[str, Any]) -> dict[str, Any]:
    base = PROJECT_ROOT / config["output"]["dir"]
    rows = []
    for entry in config["configurations"]:
        path = base / entry["label"] / "metrics.json"
        if not path.exists():
            rows.append({"label": entry["label"], "status": "not_run"})
            continue
        m = json.loads(path.read_text())
        row = {k: m.get(k) for k in ("label", "model_type", "tier", "n_bits", "status", "all_gates_passed", "compile_seconds", "peak_rss_mb")}
        if m.get("accuracy"):
            row["quantized_pr_auc"] = m["accuracy"]["quantized_full_metrics"]["pr_auc"]
            row["float_pr_auc"] = m["accuracy"]["float_full_metrics"]["pr_auc"]
            row["t3_decision_agreement"] = m["accuracy"]["gates"]["t3"]["decision_agreement"]
        if m.get("fhe_latency"):
            row["fhe_mean_seconds"] = m["fhe_latency"]["total"]["mean_seconds"]
            row["fhe_std_seconds"] = m["fhe_latency"]["total"]["std_seconds"]
            row["ciphertext_input_bytes"] = m["ciphertext_size_bytes"]["input"]
            row["ciphertext_output_bytes"] = m["ciphertext_size_bytes"]["output"]
        rows.append(row)
    summary = {"configurations": rows, "all_run": all(r["status"] != "not_run" for r in rows), "all_passed": all(r.get("all_gates_passed") for r in rows)}
    (base / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/phase8/research_grid.yaml")
    parser.add_argument("--label", action="append", help="Configuration label to run (repeatable); defaults to every configured label")
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.summary:
        write_summary(config)
        return

    failures = []
    for label in args.label or [e["label"] for e in config["configurations"]]:
        result = run_configuration(config, args.config, label)
        if result["status"] != "passed":
            failures.append((label, result["status"]))
    write_summary(config)
    if failures:
        raise Phase8GateFailure(f"Phase 8 configurations not passing: {failures} -- see {PROJECT_ROOT / config['output']['dir']}")


if __name__ == "__main__":
    main()
