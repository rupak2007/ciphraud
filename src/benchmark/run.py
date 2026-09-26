"""Phase 7 benchmark harness CLI (`docs/plan.md` Phase 7).

WSL2/Linux only (Concrete-ML has no Windows wheels, `docs/environment.md`).
Run as, from the project root:

    ~/.venvs/fhe-fraud-detection/bin/python -m src.benchmark.run --config configs/phase7/smoke.yaml

For each configured entry, in order:

  1. Load the REFERENCED Phase 5/6 source config and its already-committed,
     already-correctness-validated handoff (`src/fhe/handoff.py::load_handoff`,
     unchanged) -- this module never re-exports and never re-validates
     correctness; that is `src/fhe/poc.py`/`src/fhe/xgb_poc.py`'s job, already
     done. `model_type`, `tier`, and `n_bits` all come from the source config,
     never from the benchmark config itself, so a benchmark run cannot
     silently diverge from the model/tier/quantization Phase 5/6 validated.
  2. Rebuild + compile the SAME model with the SAME calibration inputs Phase
     5/6 already used (`build_concrete_lr`/`build_concrete_xgb`,
     `compile_model`) -- an independent recompilation of an already-validated
     configuration, not a new one.
  3. Time `trials` repeated runs of (a) the plaintext (non-FHE) prediction and
     (b) the real `encrypt -> run -> decrypt` round trip, both on ONE fixed,
     seeded request -- isolating measurement noise from row-to-row variation
     (`docs/plan.md` Phase 7 risk: "measurement noise from other system
     load"). `keygen` is timed once, amortized, never folded into per-request
     latency (matches Phase 5/6's own `_explicit_round_trip`/
     `explicit_round_trip`).
  4. Record circuit/ciphertext/key-size statistics once per configuration
     (`circuit_stats`, `tree_stats` for XGBoost) and peak process RSS.
  5. Write `results/phase7_benchmark/{config_hash}/{metrics,provenance}.json`,
     keyed by THIS benchmark entry's own config hash -- never overloading or
     mutating Phase 5/6's own config hashes or result files -- plus an
     aggregate `results/phase7_benchmark/summary.json`.

Never touches the test partition (only ever reads `X_val`/`y_val` from the
existing handoff npz, exactly as Phase 5/6 do) and never modifies any
Phase 1-6 file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.benchmark.harness import fhe_round_trip_trials, plaintext_latency_trials
from src.config import PROJECT_ROOT, load_config
from src.data import provenance as data_provenance
from src.fhe.compile.linear import circuit_stats, compile_model
from src.fhe.compile.tree import build_concrete_xgb, load_inference_classifier, tree_stats
from src.fhe.handoff import load_handoff, rebuild_pipeline_predict_proba, standardize_features
from src.fhe.poc import _stratified_sample_positions
from src.fhe.xgb_poc import calibration_positions, peak_rss_mb
from src.logging_setup import get_logger

logger = get_logger(__name__)

LIBRARY_NAMES = ["numpy", "pandas", "scikit-learn", "xgboost", "concrete-ml", "concrete-python", "PyYAML"]


class BenchmarkConfigError(Exception):
    """Raised when a benchmark entry's referenced source config/handoff is inconsistent."""


def _load_lr(source_config: dict[str, Any], n_bits: int | None = None) -> dict[str, Any]:
    """Rebuild + compile the committed Phase 5 LR exactly as `src/fhe/poc.py`
    does. `n_bits`, when given, overrides the source config's own value --
    used ONLY by Phase 8 to test a second bit-width against the SAME
    already-validated tier/calibration; omitted (the default), this
    behaves identically to Phase 7's own call sites."""
    npz_path = PROJECT_ROOT / source_config["output"]["handoff_npz"]
    manifest_path = PROJECT_ROOT / source_config["output"]["handoff_manifest"]
    manifest, arrays = load_handoff(npz_path, manifest_path)
    params = json.loads((PROJECT_ROOT / manifest["params_path"]).read_text())

    from src.fhe.compile.linear import build_concrete_lr

    effective_n_bits = n_bits if n_bits is not None else source_config["n_bits"]
    X_val, y_val = arrays["X_val"], arrays["y_val"]
    # The handoff holds RAW features but `lr_coef` belongs to the model trained on
    # STANDARDIZED features, so the compiled model is built, calibrated and evaluated
    # on standardized inputs only (client-side scaler; docs/architecture.md Sec.6).
    # `X_val` stays raw -- `plaintext_predict` rebuilds the whole pipeline (scaler
    # included) from it -- and `X_val_model` is what the circuit/quantizer consume.
    # The raw train matrix is dropped once standardized to bound memory (top_100).
    X_train_model = standardize_features(params, arrays.pop("X_train"))
    X_val_model = standardize_features(params, X_val)
    cml_model = build_concrete_lr(params, X_train_model, effective_n_bits)
    circuit, compile_seconds = compile_model(cml_model, X_train_model)
    return {
        "cml_model": cml_model,
        "circuit": circuit,
        "compile_seconds": compile_seconds,
        "extra_stats": {},
        "plaintext_predict": lambda X: rebuild_pipeline_predict_proba(params, X)[:, 1],
        "X_val": X_val,
        "X_val_model": X_val_model,
        "y_val": y_val,
        "n_features": manifest["n_features"],
        "n_bits": effective_n_bits,
        "reference_val_prob": arrays["reference_val_prob"],
        "threshold": manifest.get("threshold"),
        "val_transaction_ids": arrays["val_transaction_ids"],
    }


def _load_xgboost(source_config: dict[str, Any], tier: str, n_bits: int | None = None) -> dict[str, Any]:
    """Rebuild + compile a committed Phase 6 XGBoost tier exactly as
    `src/fhe/xgb_poc.py::run_tier` does -- SAME `calibration_positions` call,
    SAME seed, all read from `source_config`. `n_bits`, when given,
    overrides the source config's own value (Phase 8 only; see `_load_lr`)."""
    handoff_dir = PROJECT_ROOT / source_config["output"]["handoff_dir"]
    manifest_path = PROJECT_ROOT / source_config["output"]["dir"] / tier / "handoff_manifest.json"
    manifest, arrays = load_handoff(handoff_dir / f"xgb_{tier}.npz", manifest_path)
    booster_path = PROJECT_ROOT / manifest["booster_path"]
    clf = load_inference_classifier(booster_path)

    effective_n_bits = n_bits if n_bits is not None else source_config["n_bits"]
    X_train, X_val, y_val = arrays["X_train"], arrays["X_val"], arrays["y_val"]
    cal_cfg = source_config["calibration"]
    cal_pos = calibration_positions(X_train, cal_cfg["rows"], cal_cfg.get("include_extremes", False), source_config["seed"])
    X_cal = X_train[cal_pos]
    cml_model = build_concrete_xgb(clf, X_cal, effective_n_bits)
    circuit, compile_seconds = compile_model(cml_model, X_cal)
    return {
        "cml_model": cml_model,
        "circuit": circuit,
        "compile_seconds": compile_seconds,
        "extra_stats": tree_stats(clf, cml_model, circuit),
        "plaintext_predict": lambda X: clf.predict_proba(X)[:, 1],
        "X_val": X_val,
        "X_val_model": X_val,  # trees consume the raw features directly; alias kept so callers are model-agnostic
        "y_val": y_val,
        "n_features": manifest["n_features"],
        "n_bits": effective_n_bits,
        "reference_val_prob": arrays["reference_val_prob"],
        "threshold": manifest.get("threshold"),
        "val_transaction_ids": arrays["val_transaction_ids"],
        "clf": clf,
    }


def benchmark_configuration(entry: dict[str, Any], seed: int, trials: int, n_positive: int) -> dict[str, Any]:
    """Benchmark one already-committed, already-correctness-validated
    configuration. `entry` supplies only `label`, `model_type`,
    `source_config`, and (for XGBoost) `tier` -- never `n_bits`, which
    always comes from the referenced source config, so a benchmark
    configuration cannot silently diverge from what Phase 5/6 validated."""
    source_config_path = entry["source_config"]
    source_config = load_config(source_config_path)
    model_type = entry["model_type"]

    if model_type == "lr":
        loaded = _load_lr(source_config)
    elif model_type == "xgboost":
        if "tier" not in entry:
            raise BenchmarkConfigError(f"{entry['label']}: model_type 'xgboost' requires a 'tier'")
        loaded = _load_xgboost(source_config, entry["tier"])
    else:
        raise BenchmarkConfigError(f"{entry['label']}: unknown model_type {model_type!r} (expected 'lr' or 'xgboost')")

    cml_model, circuit = loaded["cml_model"], loaded["circuit"]
    X_val, y_val = loaded["X_val"], loaded["y_val"]

    # ONE fixed, seeded request, reused for every trial -- FHE latency is
    # input-independent (a compiled circuit performs the same fixed sequence
    # of operations regardless of which values it is evaluating), so the
    # choice of row affects only the plaintext prediction's VALUE, never
    # either path's LATENCY; documented, not assumed.
    row_position = _stratified_sample_positions(y_val, n=1, n_positive=n_positive, seed=seed)[0]
    X_row = X_val[row_position : row_position + 1]
    q_row = cml_model.quantize_input(loaded["X_val_model"][row_position : row_position + 1])

    logger.info("Benchmarking configuration", extra={"extra_fields": {"label": entry["label"], "trials": trials}})
    plaintext_latency = plaintext_latency_trials(lambda: loaded["plaintext_predict"](X_row), trials)
    fhe_latency = fhe_round_trip_trials(circuit, q_row, trials)

    stats = {**circuit_stats(circuit), **loaded["extra_stats"]}

    source_config_hash = data_provenance.compute_config_hash(source_config)
    benchmark_entry_effective = {
        "label": entry["label"], "model_type": model_type, "tier": entry.get("tier"),
        "source_config": source_config_path, "source_config_hash": source_config_hash,
        "n_bits": loaded["n_bits"], "seed": seed, "trials": trials, "n_positive": n_positive,
    }
    config_hash = data_provenance.compute_config_hash(benchmark_entry_effective)

    result = {
        **benchmark_entry_effective,
        "config_hash": config_hash,
        "n_features": loaded["n_features"],
        "compile_seconds": loaded["compile_seconds"],
        "circuit_stats": stats,
        "ciphertext_size_bytes": {"input": stats["size_of_inputs_bytes"], "output": stats["size_of_outputs_bytes"]},
        "key_size_bytes": {
            "secret": stats["size_of_secret_keys_bytes"],
            "bootstrap": stats["size_of_bootstrap_keys_bytes"],
            "keyswitch": stats["size_of_keyswitch_keys_bytes"],
        },
        "row_selection": {"position": int(row_position), "y_true": int(y_val[row_position]), "n_positive": n_positive},
        "plaintext_latency": plaintext_latency,
        "fhe_latency": fhe_latency,
        "peak_rss_mb": peak_rss_mb(),
        "test_partition_touched": False,
    }
    return result


def run(config_path: str) -> dict[str, Any]:
    config = load_config(config_path)
    output_dir = PROJECT_ROOT / config["output"]["dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    seed, trials = config["seed"], config["trials"]
    n_positive = config.get("row_selection", {}).get("n_positive", 0)

    summaries = []
    for entry in config["configurations"]:
        result = benchmark_configuration(entry, seed, trials, n_positive)
        config_dir = output_dir / result["config_hash"]
        config_dir.mkdir(parents=True, exist_ok=True)
        (config_dir / "metrics.json").write_text(json.dumps(result, indent=2, sort_keys=True, default=str), encoding="utf-8")
        provenance = data_provenance.build_provenance(
            config=config, config_path=Path(config_path), seed=seed, raw_file_digests={}, library_names=LIBRARY_NAMES,
        )
        (config_dir / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True, default=str), encoding="utf-8")
        logger.info("Benchmark configuration complete", extra={"extra_fields": {"label": result["label"], "config_hash": result["config_hash"]}})
        summaries.append({
            "label": result["label"], "model_type": result["model_type"], "tier": result["tier"],
            "n_bits": result["n_bits"], "config_hash": result["config_hash"],
            "compile_seconds": result["compile_seconds"],
            "plaintext_mean_seconds": result["plaintext_latency"]["mean_seconds"],
            "plaintext_std_seconds": result["plaintext_latency"]["std_seconds"],
            "fhe_mean_seconds": result["fhe_latency"]["total"]["mean_seconds"],
            "fhe_std_seconds": result["fhe_latency"]["total"]["std_seconds"],
            "fhe_outputs_reproducible": result["fhe_latency"]["outputs_reproducible"],
            "ciphertext_input_bytes": result["ciphertext_size_bytes"]["input"],
            "ciphertext_output_bytes": result["ciphertext_size_bytes"]["output"],
            "programmable_bootstrap_count": result["circuit_stats"]["programmable_bootstrap_count"],
        })

    summary = {"n_bits_by_config": {s["label"]: s["n_bits"] for s in summaries}, "configurations": summaries}
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True, default=str), encoding="utf-8")
    logger.info("Phase 7 benchmark run complete", extra={"extra_fields": {"n_configurations": len(summaries)}})
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/phase7/smoke.yaml", help="Path to the Phase 7 benchmark config YAML")
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
