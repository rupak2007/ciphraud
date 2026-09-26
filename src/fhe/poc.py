"""Phase 5 FHE proof-of-concept orchestrator (`docs/plan.md` Phase 5).

WSL2/Linux only -- Concrete-ML has no Windows wheels (`docs/environment.md`).
Run as (from the WSL FHE venv, from the project root):
    ~/.venvs/fhe-fraud-detection/bin/python -m src.fhe.poc --config configs/phase5/lr_poc.yaml

Orchestrates, in order (see `docs/fhe_poc.md` for the full methodology):
  1. Load + verify the Windows-side handoff (`src/fhe/handoff.py`) -- npz
     SHA-256, manifest shape checks. Never reads anything from
     `src/train/data.py` or any Phase 1-4 artifact directly.
  2. T0: the handoff's `params` reconstructed via pure numpy
     (`rebuild_pipeline_predict_proba`) vs. the Windows-computed reference
     probabilities baked into the handoff -- isolates cross-environment
     transfer error.
  3. Build + compile the Concrete-ML LR (`src/fhe/compile/linear.py`),
     calibrated on the SCALED TRAIN partition from the handoff (never val,
     never test).
  4. T3: `predict_proba(X_val, fhe="disable")` (clear-quantized) vs. the
     float reference, on the FULL val partition -- isolates quantization
     error.
  5. T2: `predict_proba(X_val, fhe="simulate")` vs. `fhe="disable"`, on the
     FULL val partition -- isolates any circuit/compile-time effect.
  6. E5/T1: the EXPLICIT round trip (quantize_input -> keygen -> encrypt ->
     run -> decrypt -> dequantize_output -> post_processing) on a seeded,
     class-stratified sample of val rows -- the real encrypt/infer/decrypt
     Phase 5's exit criterion requires -- compared against `fhe="simulate"`
     on that same sample.
  7. Write `results/phase5_fhe_poc/{correctness_report,circuit_stats,metrics,provenance}.json`
     and `execute_sample.csv`.

**No use of the test partition anywhere**: this module never even receives
`X_test`/`y_test` -- the handoff npz (`src/fhe/export.py`) only ever
contains train/val arrays.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT, load_config
from src.data import provenance as data_provenance
from src.fhe.compile.linear import build_concrete_lr, circuit_stats, compile_model
from src.fhe.handoff import load_handoff, rebuild_pipeline_predict_proba, standardize_features
from src.fhe.validate.correctness import (
    t0_transfer_check,
    t1_execution_check,
    t2_simulation_check,
    t3_quantization_check,
)
from src.logging_setup import get_logger

logger = get_logger(__name__)

LIBRARY_NAMES = ["numpy", "pandas", "scikit-learn", "concrete-ml", "concrete-python", "PyYAML"]


class PoCGateFailure(Exception):
    """Raised when a required correctness gate (T0/T1/T2/T3) fails -- the
    caller must stop and report, never silently continue past a failed gate."""


def _stratified_sample_positions(y_val: np.ndarray, n: int, n_positive: int, seed: int) -> np.ndarray:
    """Positions (not TransactionIDs) of a seeded, class-stratified sample
    of val rows: exactly `n_positive` fraud rows and `n - n_positive`
    legitimate rows, so the small execute sample isn't accidentally
    all-one-class."""
    rng = np.random.RandomState(seed)
    positive_positions = np.flatnonzero(y_val == 1)
    negative_positions = np.flatnonzero(y_val == 0)
    if len(positive_positions) < n_positive or len(negative_positions) < (n - n_positive):
        raise ValueError(
            f"val partition doesn't have enough rows for a stratified sample of "
            f"{n_positive} positive / {n - n_positive} negative (has {len(positive_positions)} "
            f"positive, {len(negative_positions)} negative)"
        )
    chosen_positive = rng.choice(positive_positions, size=n_positive, replace=False)
    chosen_negative = rng.choice(negative_positions, size=n - n_positive, replace=False)
    positions = np.concatenate([chosen_positive, chosen_negative])
    rng.shuffle(positions)
    return positions


def _explicit_round_trip(cml_model: Any, circuit: Any, X_sample: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    """The EXPLICIT client/server round trip, one row at a time:
    quantize_input -> encrypt -> run -> decrypt -> dequantize_output ->
    post_processing. Returns the P(fraud) probability per row and timing
    observations (keygen once, then per-row encrypt/run/decrypt) --
    single-run wall-clock observations, NOT a benchmark (Phase 7 owns
    repeated-trial measurement)."""
    q = cml_model.quantize_input(X_sample)  # shape (n, n_features), int

    t0 = time.perf_counter()
    circuit.keygen()
    keygen_seconds = time.perf_counter() - t0

    probs = np.empty(X_sample.shape[0], dtype=np.float64)
    row_seconds = []
    for i in range(X_sample.shape[0]):
        t_row = time.perf_counter()
        q_row = q[i : i + 1]  # keep the batch dimension -- circuit.encrypt requires it
        encrypted = circuit.encrypt(q_row)
        ran = circuit.run(encrypted)
        decrypted = circuit.decrypt(ran)
        row_seconds.append(time.perf_counter() - t_row)

        dequantized = cml_model.dequantize_output(np.asarray(decrypted).reshape(1, -1))
        post = cml_model.post_processing(dequantized)
        probs[i] = float(np.asarray(post).reshape(1, -1)[0, 1])

    timing = {
        "keygen_seconds": keygen_seconds,
        "mean_row_seconds": float(np.mean(row_seconds)),
        "max_row_seconds": float(np.max(row_seconds)),
        "min_row_seconds": float(np.min(row_seconds)),
    }
    return probs, timing


def run(config_path: str) -> dict[str, Any]:
    config = load_config(config_path)
    output_dir = PROJECT_ROOT / config["output"]["dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    tolerances = config["tolerances"]

    logger.info("Loading + verifying the Windows-side handoff")
    npz_path = PROJECT_ROOT / config["output"]["handoff_npz"]
    manifest_path = PROJECT_ROOT / config["output"]["handoff_manifest"]
    manifest, arrays = load_handoff(npz_path, manifest_path)
    params_path = PROJECT_ROOT / manifest["params_path"]
    params = json.loads(params_path.read_text())

    X_train, X_val, y_val = arrays["X_train"], arrays["X_val"], arrays["y_val"]
    # The handoff holds RAW features, but `lr_coef` belongs to the model trained on
    # STANDARDIZED features: the compiled model must only ever see standardized
    # inputs (client-side scaler, docs/architecture.md Sec.6). T0 below stays on the
    # raw matrix because it rebuilds the whole pipeline (scaler included) from params.
    X_train_model = standardize_features(params, X_train)
    X_val_model = standardize_features(params, X_val)
    val_transaction_ids = arrays["val_transaction_ids"]
    reference_val_prob = arrays["reference_val_prob"]
    threshold = manifest["threshold"]

    logger.info("T0: transfer check (rebuilt-from-params vs. Windows reference)")
    rebuilt_prob = rebuild_pipeline_predict_proba(params, X_val)[:, 1]
    t0 = t0_transfer_check(rebuilt_prob, reference_val_prob, tolerances["t0_max_abs_diff"])
    if not t0["passed"]:
        raise PoCGateFailure(f"T0 (transfer) failed: {t0}")
    logger.info("T0 passed", extra={"extra_fields": t0})

    logger.info("Building + compiling Concrete-ML LR", extra={"extra_fields": {"n_bits": config["n_bits"]}})
    cml_model = build_concrete_lr(params, X_train_model, config["n_bits"])
    circuit, compile_seconds = compile_model(cml_model, X_train_model)
    stats = circuit_stats(circuit)
    stats["compile_seconds"] = compile_seconds
    logger.info("Compiled", extra={"extra_fields": stats})

    # T3/T2/T1 are computed unconditionally, even if one of them fails --
    # each isolates a different error source (module docstring), so a
    # failure in one must not suppress measuring the others. All results
    # are written to disk BEFORE any failure is raised (below), so a
    # failing run still leaves a complete, inspectable report rather than
    # losing whatever was already measured.
    logger.info("T3: quantization check (fhe=disable vs. float reference, full val)")
    disable_prob = cml_model.predict_proba(X_val_model, fhe="disable")[:, 1]
    t3 = t3_quantization_check(
        quantized_prob=disable_prob, float_prob=reference_val_prob, y_true=y_val, threshold=threshold,
        min_decision_agreement=tolerances["t3_min_decision_agreement"],
        max_pr_auc_drop=tolerances["t3_max_pr_auc_drop"],
    )
    logger.info("T3 result", extra={"extra_fields": {k: v for k, v in t3.items() if not isinstance(v, dict)}})

    logger.info("T2: simulation check (fhe=simulate vs. fhe=disable, full val)")
    simulate_prob_full = cml_model.predict_proba(X_val_model, fhe="simulate")[:, 1]
    t2 = t2_simulation_check(simulate_prob_full, disable_prob)
    logger.info("T2 result", extra={"extra_fields": t2})

    logger.info("E5/T1: explicit encrypt->run->decrypt round trip on a stratified execute sample")
    exec_cfg = config["execute_sample"]
    sample_positions = _stratified_sample_positions(y_val, exec_cfg["n"], exec_cfg["n_positive"], config["seed"])
    X_sample = X_val_model[sample_positions]
    y_sample = y_val[sample_positions]
    sample_transaction_ids = val_transaction_ids[sample_positions]

    simulate_prob_sample = cml_model.predict_proba(X_sample, fhe="simulate")[:, 1]
    decrypted_prob_sample, round_trip_timing = _explicit_round_trip(cml_model, circuit, X_sample)
    t1 = t1_execution_check(decrypted_prob_sample, simulate_prob_sample)
    logger.info("T1 result", extra={"extra_fields": {**t1, **round_trip_timing}})

    pd.DataFrame({
        "TransactionID": sample_transaction_ids,
        "y_true": y_sample,
        "simulate_prob": simulate_prob_sample,
        "decrypted_prob": decrypted_prob_sample,
    }).to_csv(output_dir / "execute_sample.csv", index=False)

    correctness_report = {"t0": t0, "t1": t1, "t2": t2, "t3": t3}
    (output_dir / "correctness_report.json").write_text(
        json.dumps(correctness_report, indent=2, sort_keys=True, default=str), encoding="utf-8"
    )
    (output_dir / "circuit_stats.json").write_text(json.dumps(stats, indent=2, sort_keys=True), encoding="utf-8")

    results = {
        "tier": manifest["tier"],
        "n_bits": config["n_bits"],
        "n_features": manifest["n_features"],
        "n_train_calibration": int(X_train.shape[0]),
        "n_val": int(X_val.shape[0]),
        "n_execute_sample": int(X_sample.shape[0]),
        "correctness_report": correctness_report,
        "circuit_stats": stats,
        "round_trip_timing": round_trip_timing,
        "all_gates_passed": bool(t0["passed"] and t1["passed"] and t2["passed"] and t3["passed"]),
        "test_partition_touched": False,  # this module never even receives X_test/y_test
    }
    (output_dir / "metrics.json").write_text(json.dumps(results, indent=2, sort_keys=True, default=str), encoding="utf-8")

    provenance = data_provenance.build_provenance(
        config=config, config_path=Path(config_path), seed=config["seed"],
        raw_file_digests={}, library_names=LIBRARY_NAMES,
    )
    (output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True, default=str), encoding="utf-8")

    logger.info(
        "Phase 5 PoC complete",
        extra={"extra_fields": {"all_gates_passed": results["all_gates_passed"]}},
    )

    failed_gates = [name for name, gate in (("T1", t1), ("T2", t2), ("T3", t3)) if not gate["passed"]]
    if failed_gates:
        # Every measurement is already written to
        # results/phase5_fhe_poc/{correctness_report,metrics}.json above --
        # this is reported, not silently absorbed or worked around by
        # retraining/re-quantizing without approval (this session's
        # explicit instruction).
        raise PoCGateFailure(
            f"Required gate(s) {failed_gates} failed -- see "
            f"{output_dir / 'correctness_report.json'} for full detail: "
            f"{json.dumps({g: correctness_report[g.lower()] for g in failed_gates}, default=str)}"
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/phase5/lr_poc.yaml", help="Path to the Phase 5 PoC config YAML")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
