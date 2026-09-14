"""Phase 6 FHE XGBoost orchestrator (`docs/plan.md` Phase 6).

WSL2/Linux only. Run one tier per process, so a memory failure on one tier
can't take the others' results with it, then aggregate:

    ~/.venvs/fhe-fraud-detection/bin/python -m src.fhe.xgb_poc --config configs/phase6/xgb_poc.yaml --tier top_20
    ~/.venvs/fhe-fraud-detection/bin/python -m src.fhe.xgb_poc --config configs/phase6/xgb_poc.yaml --summary

Per tier, in order (docs/fhe_xgboost.md has the methodology):
  1. Load + verify the Windows-side handoff (`src/fhe/export_xgboost.py`):
     npz SHA-256, exported booster SHA-256, tree count.
  2. T0: WSL xgboost predicting the exported booster vs. the Windows reference.
  3. Build + compile the Concrete-ML XGBClassifier on train-only calibration rows.
  4. T3: `fhe="disable"` vs. the float reference on the FULL val partition.
  5. T2: official `simulate` vs. `disable`, compared as exact integer circuit
     outputs, on the configured val rows. Simulation runs one row at a time
     and is checkpointed, because it is slow for large tree ensembles.
  6. T1: real keygen -> encrypt -> run -> decrypt on a seeded stratified val
     sample, compared as exact integer outputs against `simulate`.
  7. Write results/phase6_fhe_xgboost/{tier}/{correctness_report,circuit_stats,metrics,provenance}.json
     and execute_sample.csv. Everything measured is written before any
     gate failure is raised.

The test partition is never received: the handoff npz only holds train/val arrays.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import resource
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.config import PROJECT_ROOT, load_config
from src.data import provenance as data_provenance
from src.fhe.compile.linear import circuit_stats, compile_model
from src.fhe.compile.tree import build_concrete_xgb, load_inference_classifier, tree_stats
from src.fhe.handoff import HandoffError, load_handoff, sha256_file
from src.fhe.poc import _stratified_sample_positions
from src.fhe.validate.correctness import integer_output_check, t0_transfer_check, t3_quantization_check
from src.logging_setup import get_logger

logger = get_logger(__name__)

LIBRARY_NAMES = ["numpy", "pandas", "scikit-learn", "xgboost", "concrete-ml", "concrete-python", "PyYAML"]
GATES = ("t0", "t3", "t2", "t1")


class XGBGateFailure(Exception):
    """Raised after all measurements are written, when a tier fails a required gate."""


def peak_rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def calibration_positions(X_train: np.ndarray, n_rows: int | str, include_extremes: bool, seed: int) -> np.ndarray:
    """Train-row positions used for quantization calibration and the compile inputset.

    With `include_extremes`, every feature's train min and max row is added, so
    each per-feature input quantizer gets exactly the full-train range."""
    if n_rows == "all":
        return np.arange(X_train.shape[0])
    rng = np.random.RandomState(seed)
    chosen = rng.choice(X_train.shape[0], size=int(n_rows), replace=False)
    if include_extremes:
        chosen = np.concatenate([chosen, np.argmin(X_train, axis=0), np.argmax(X_train, axis=0)])
    return np.unique(chosen)


def t2_positions(n_val: int, t2_rows: int | str, seed: int) -> np.ndarray:
    if t2_rows == "all":
        return np.arange(n_val)
    return np.sort(np.random.RandomState(seed).choice(n_val, size=int(t2_rows), replace=False))


def disable_integers(cml_model: Any, q: np.ndarray, chunk_size: int) -> np.ndarray:
    """Clear-quantized integer outputs (the computation behind `fhe="disable"`), in row
    chunks: Concrete-ML's tree forward materializes (trees x rows x nodes) tensors, which
    would not fit in memory for the whole val partition at once. Rows are independent."""
    return np.ascontiguousarray(
        np.concatenate([cml_model._inference(q[s : s + chunk_size]) for s in range(0, q.shape[0], chunk_size)])
    )


def simulate_integers(circuit: Any, q: np.ndarray, checkpoint_dir: Path, fingerprint: str, chunk_size: int) -> tuple[np.ndarray, float]:
    """Official per-row `circuit.simulate`, stacked exactly the way Concrete-ML's own
    `predict(fhe="simulate")` does. Completed chunks are saved and reused on restart,
    but only when the fingerprint matches and a re-simulated spot-check row agrees."""
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    meta_path = checkpoint_dir / "meta.json"
    if meta_path.exists() and json.loads(meta_path.read_text()).get("fingerprint") != fingerprint:
        for stale in checkpoint_dir.glob("chunk_*.npy"):
            stale.unlink()
    meta_path.write_text(json.dumps({"fingerprint": fingerprint, "n_rows": int(q.shape[0]), "chunk_size": chunk_size}))

    chunks = []
    simulated_seconds = 0.0
    simulated_rows = 0
    for start in range(0, q.shape[0], chunk_size):
        stop = min(start + chunk_size, q.shape[0])
        path = checkpoint_dir / f"chunk_{start:08d}_{stop:08d}.npy"
        if path.exists():
            chunk = np.load(path, allow_pickle=False)
            if not np.array_equal(circuit.simulate(q[start : start + 1])[0], chunk[0]):
                raise RuntimeError(f"checkpoint {path} disagrees with a fresh simulation of its first row")
        else:
            t0 = time.perf_counter()
            chunk = np.array([circuit.simulate(q[i : i + 1])[0] for i in range(start, stop)])
            simulated_seconds += time.perf_counter() - t0
            simulated_rows += stop - start
            np.save(path, chunk)
            logger.info("T2 simulate chunk done", extra={"extra_fields": {"stop": stop, "of": int(q.shape[0])}})
        chunks.append(chunk)
    seconds_per_row = simulated_seconds / simulated_rows if simulated_rows else float("nan")
    return np.concatenate(chunks), seconds_per_row


def explicit_round_trip(
    circuit: Any, q: np.ndarray, checkpoint_dir: Path | None = None, fingerprint: str | None = None,
) -> tuple[np.ndarray, dict[str, float]]:
    """Client/server split of the real encrypted path, one row at a time:
    client keygen + encrypt, server `run`, client decrypt. Returns the decrypted
    integer circuit outputs, stacked the same way Concrete-ML's `predict` does.

    Measured on the real committed tiers (docs/fhe_xgboost.md): a single row's
    `run` alone takes ~23.5 minutes for `top_50` (219 trees, 151,986 PBS) and
    did not complete within 7+ minutes across three attempts for `top_20` (358
    trees, 248,452 PBS) before this interactive environment's WSL2 VM or
    session was interrupted. `checkpoint_dir`/`fingerprint`, when given, persist
    each row's result to disk immediately after it completes (mirroring
    `simulate_integers`'s chunk checkpoints) so a re-run resumes at the next
    un-executed row instead of re-running every prior row -- keygen is cheap
    (~2s) and is always redone, since the secret key itself is never persisted."""
    if checkpoint_dir is not None:
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        meta_path = checkpoint_dir / "meta.json"
        if meta_path.exists() and json.loads(meta_path.read_text()).get("fingerprint") != fingerprint:
            for stale in checkpoint_dir.glob("row_*"):
                stale.unlink()
        meta_path.write_text(json.dumps({"fingerprint": fingerprint, "n_rows": int(q.shape[0])}))

    t0 = time.perf_counter()
    circuit.keygen()
    keygen_seconds = time.perf_counter() - t0

    decrypted, encrypt_s, run_s, decrypt_s = [], [], [], []
    for i in range(q.shape[0]):
        row_path = checkpoint_dir / f"row_{i:04d}.npy" if checkpoint_dir else None
        timing_path = checkpoint_dir / f"row_{i:04d}_timing.json" if checkpoint_dir else None
        if row_path is not None and row_path.exists():
            row_decrypted = np.load(row_path, allow_pickle=False)
            if not np.array_equal(row_decrypted, circuit.simulate(q[i : i + 1])[0]):
                raise RuntimeError(f"checkpoint {row_path} disagrees with a fresh simulation of that row")
            timing = json.loads(timing_path.read_text())
            decrypted.append(row_decrypted)
            encrypt_s.append(timing["encrypt_seconds"])
            run_s.append(timing["run_seconds"])
            decrypt_s.append(timing["decrypt_seconds"])
            logger.info("T1 row reused from checkpoint", extra={"extra_fields": {"row": i + 1, "of": int(q.shape[0])}})
            continue

        t0 = time.perf_counter()
        encrypted = circuit.encrypt(q[i : i + 1])
        e_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        result = circuit.run(encrypted)
        r_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        row_decrypted = np.asarray(circuit.decrypt(result))[0]
        d_s = time.perf_counter() - t0
        decrypted.append(row_decrypted)
        encrypt_s.append(e_s)
        run_s.append(r_s)
        decrypt_s.append(d_s)
        if row_path is not None:
            np.save(row_path, row_decrypted)
            timing_path.write_text(json.dumps({"encrypt_seconds": e_s, "run_seconds": r_s, "decrypt_seconds": d_s}))
        logger.info("T1 row executed", extra={"extra_fields": {"row": i + 1, "of": int(q.shape[0]), "run_seconds": r_s}})

    timing = {
        "keygen_seconds": keygen_seconds,
        "mean_encrypt_seconds": float(np.mean(encrypt_s)),
        "mean_run_seconds": float(np.mean(run_s)),
        "min_run_seconds": float(np.min(run_s)),
        "max_run_seconds": float(np.max(run_s)),
        "mean_decrypt_seconds": float(np.mean(decrypt_s)),
    }
    return np.array(decrypted), timing


def _probabilities(cml_model: Any, q_out: np.ndarray) -> np.ndarray:
    return cml_model.post_processing(cml_model.dequantize_output(q_out))[:, 1]


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def run_tier(config: dict[str, Any], config_path: str, tier: str) -> dict[str, Any]:
    output_dir = PROJECT_ROOT / config["output"]["dir"] / tier
    handoff_dir = PROJECT_ROOT / config["output"]["handoff_dir"]
    tolerances = config["tolerances"]
    seed = config["seed"]
    report: dict[str, Any] = {}
    metrics: dict[str, Any] = {"tier": tier, "n_bits": config["n_bits"], "test_partition_touched": False}

    def finish(status: str, error: dict[str, Any] | None = None) -> dict[str, Any]:
        metrics["status"] = status
        metrics["gates_passed"] = {g: report[g]["passed"] for g in GATES if g in report}
        metrics["all_gates_passed"] = all(g in report and report[g]["passed"] for g in GATES)
        metrics["peak_rss_mb"] = peak_rss_mb()
        if error:
            metrics["error"] = error
        _write_json(output_dir / "correctness_report.json", report)
        _write_json(output_dir / "metrics.json", metrics)
        _write_json(output_dir / "provenance.json", data_provenance.build_provenance(
            config=config, config_path=Path(config_path), seed=seed, raw_file_digests={}, library_names=LIBRARY_NAMES,
        ))
        logger.info("Phase 6 tier finished", extra={"extra_fields": {"tier": tier, "status": status}})
        return metrics

    logger.info("Loading + verifying the Windows-side handoff", extra={"extra_fields": {"tier": tier}})
    manifest, arrays = load_handoff(handoff_dir / f"xgb_{tier}.npz", output_dir / "handoff_manifest.json")
    booster_path = PROJECT_ROOT / manifest["booster_path"]
    if manifest["tier"] != tier:
        raise HandoffError(f"manifest is for tier {manifest['tier']!r}, expected {tier!r}")
    if sha256_file(booster_path) != manifest["booster_sha256"]:
        raise HandoffError(f"{booster_path} SHA-256 does not match the manifest")

    X_train, X_val, y_val = arrays["X_train"], arrays["X_val"], arrays["y_val"]
    reference_val_prob = arrays["reference_val_prob"]
    threshold = manifest["threshold"]
    clf = load_inference_classifier(booster_path)
    if clf.get_booster().num_boosted_rounds() != manifest["n_trees_inference"]:
        raise HandoffError(f"{booster_path} tree count does not match the manifest")
    metrics.update({
        "n_features": manifest["n_features"], "n_trees": manifest["n_trees_inference"],
        "n_trees_stored_in_phase4_booster": manifest["n_trees_stored_in_phase4_booster"],
        "threshold": threshold, "n_val": int(X_val.shape[0]), "booster_sha256": manifest["booster_sha256"],
        "handoff_npz_sha256": manifest["npz_sha256"], "windows_git_commit": manifest["windows_git_commit"],
    })

    logger.info("T0: transfer check")
    report["t0"] = t0_transfer_check(clf.predict_proba(X_val)[:, 1], reference_val_prob, tolerances["t0_max_abs_diff"])
    if not report["t0"]["passed"]:
        return finish("stopped_at_t0")

    cal_cfg = config["calibration"]
    cal_pos = calibration_positions(X_train, cal_cfg["rows"], cal_cfg.get("include_extremes", False), seed)
    X_cal = X_train[cal_pos]
    metrics["calibration"] = {
        "source": "train", "requested_rows": cal_cfg["rows"], "include_extremes": cal_cfg.get("include_extremes", False),
        "n_rows": int(X_cal.shape[0]), "positions_sha256": hashlib.sha256(cal_pos.tobytes()).hexdigest(),
    }

    stage = "build"
    try:
        logger.info("Building + compiling Concrete-ML XGBClassifier", extra={"extra_fields": {"n_calibration_rows": int(X_cal.shape[0])}})
        t0 = time.perf_counter()
        cml_model = build_concrete_xgb(clf, X_cal, config["n_bits"])
        metrics["from_sklearn_model_seconds"] = time.perf_counter() - t0
        stage = "compile"
        circuit, metrics["compile_seconds"] = compile_model(cml_model, X_cal)
        stats = {**circuit_stats(circuit), **tree_stats(clf, cml_model, circuit)}
        _write_json(output_dir / "circuit_stats.json", stats)
        metrics["circuit_stats"] = stats
        metrics["peak_rss_after_compile_mb"] = peak_rss_mb()
    except Exception as exc:  # recorded as an infeasible configuration, never silently dropped
        return finish(f"{stage}_failed", {"stage": stage, "type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()})

    logger.info("T3: quantization check (full val)")
    t0 = time.perf_counter()
    q_val = cml_model.quantize_input(X_val)
    disable_q = disable_integers(cml_model, q_val, config["disable_chunk_size"])
    disable_prob = _probabilities(cml_model, disable_q)
    metrics["disable_full_val_seconds"] = time.perf_counter() - t0
    n_check = min(1000, X_val.shape[0])
    metrics["disable_prob_vs_public_predict_proba_max_abs_diff"] = float(
        np.abs(cml_model.predict_proba(X_val[:n_check], fhe="disable")[:, 1] - disable_prob[:n_check]).max()
    )
    report["t3"] = t3_quantization_check(
        quantized_prob=disable_prob, float_prob=reference_val_prob, y_true=y_val, threshold=threshold,
        min_decision_agreement=tolerances["t3_min_decision_agreement"], max_pr_auc_drop=tolerances["t3_max_pr_auc_drop"],
    )

    t2_pos = t2_positions(X_val.shape[0], config["t2"]["rows"], seed)
    fingerprint = hashlib.sha256("|".join([
        manifest["booster_sha256"], manifest["npz_sha256"], metrics["calibration"]["positions_sha256"],
        hashlib.sha256(t2_pos.tobytes()).hexdigest(), str(config["n_bits"]), str(config["t2"]["chunk_size"]),
    ]).encode()).hexdigest()
    stage = "simulate"
    try:
        logger.info("T2: simulate vs disable (integer outputs)", extra={"extra_fields": {"n_rows": int(t2_pos.shape[0])}})
        sim_q, metrics["simulate_seconds_per_row"] = simulate_integers(
            circuit, q_val[t2_pos], handoff_dir / "checkpoints" / tier / "t2_simulate", fingerprint, config["t2"]["chunk_size"],
        )
    except Exception as exc:
        return finish("simulate_failed", {"stage": stage, "type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()})
    report["t2"] = integer_output_check(
        sim_q, disable_q[t2_pos], candidate_prob=_probabilities(cml_model, sim_q),
        reference_prob=disable_prob[t2_pos], threshold=threshold,
    )
    report["t2"]["val_rows"] = "all" if config["t2"]["rows"] == "all" else int(t2_pos.shape[0])

    exec_cfg = config["execute_sample"]
    sample_pos = _stratified_sample_positions(y_val, exec_cfg["n"], exec_cfg["n_positive"], seed)
    q_sample = q_val[sample_pos]
    exec_fingerprint = hashlib.sha256("|".join([
        manifest["booster_sha256"], manifest["npz_sha256"], str(config["n_bits"]),
        hashlib.sha256(sample_pos.tobytes()).hexdigest(),
    ]).encode()).hexdigest()
    stage = "execute"
    try:
        logger.info("T1: real encrypt -> run -> decrypt", extra={"extra_fields": {"n_rows": int(q_sample.shape[0])}})
        sample_sim_q = np.array([circuit.simulate(q_sample[i : i + 1])[0] for i in range(q_sample.shape[0])])
        decrypted_q, metrics["round_trip_timing"] = explicit_round_trip(
            circuit, q_sample, handoff_dir / "checkpoints" / tier / "t1_execute", exec_fingerprint,
        )
    except Exception as exc:
        return finish("execute_failed", {"stage": stage, "type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()})
    decrypted_prob = _probabilities(cml_model, decrypted_q)
    report["t1"] = integer_output_check(
        decrypted_q, sample_sim_q, candidate_prob=decrypted_prob,
        reference_prob=_probabilities(cml_model, sample_sim_q), threshold=threshold,
    )
    report["t1"]["decrypted_vs_disable_exact_integer_match"] = bool(np.array_equal(decrypted_q, disable_q[sample_pos]))

    pd.DataFrame({
        "TransactionID": arrays["val_transaction_ids"][sample_pos],
        "y_true": y_val[sample_pos],
        "float_reference_prob": reference_val_prob[sample_pos],
        "disable_prob": disable_prob[sample_pos],
        "decrypted_prob": decrypted_prob,
    }).to_csv(output_dir / "execute_sample.csv", index=False)
    metrics["n_execute_sample"] = int(q_sample.shape[0])

    return finish("passed" if all(report[g]["passed"] for g in GATES) else "failed_gates")


def write_summary(config: dict[str, Any]) -> dict[str, Any]:
    base = PROJECT_ROOT / config["output"]["dir"]
    tiers = {}
    for tier in config["tiers"]:
        path = base / tier / "metrics.json"
        if not path.exists():
            tiers[tier] = {"status": "not_run"}
            continue
        m = json.loads(path.read_text())
        tiers[tier] = {k: m.get(k) for k in ("status", "gates_passed", "all_gates_passed", "n_features", "n_trees", "compile_seconds", "peak_rss_mb")}
    summary = {"n_bits": config["n_bits"], "tiers": tiers, "all_tiers_passed": all(t.get("all_gates_passed") for t in tiers.values())}
    _write_json(base / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/phase6/xgb_poc.yaml")
    parser.add_argument("--tier", action="append", help="Tier to run (repeatable); defaults to every configured tier")
    parser.add_argument("--summary", action="store_true", help="Only aggregate existing per-tier metrics")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.summary:
        write_summary(config)
        return

    failures = []
    for tier in args.tier or config["tiers"]:
        result = run_tier(config, args.config, tier)
        if result["status"] != "passed":
            failures.append((tier, result["status"], result.get("gates_passed")))
    write_summary(config)
    if failures:
        raise XGBGateFailure(f"Phase 6 tiers not passing: {failures} -- see {PROJECT_ROOT / config['output']['dir']}")


if __name__ == "__main__":
    main()
