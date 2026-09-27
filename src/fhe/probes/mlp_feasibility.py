"""Phase 9 R5 -- which (tier x bit-width) MLP configurations are feasible on this machine? (`docs/plan.md` Phase 9)

For every candidate the probe trains the pinned MLP briefly on a seeded TRAIN subsample of the real tier data
(standardized with the train-only scaler), compiles it, and records what the compiler and the machine say:
success or the exact error, bootstrap (TLU) count, maximum integer bit-width, key and ciphertext sizes, compile time,
peak RSS -- and, when the predicted key material fits in memory, a real key generation plus ONE timed encrypted run
so the grid's latency and memory are estimated from a measurement.

Each candidate runs in its OWN process (a child invoked with `--one`): a candidate that exhausts the 3.8 GiB WSL memory
is killed by the OS and is recorded as such instead of taking the whole probe down, and peak RSS is per candidate.
Nothing is dropped: infeasible candidates are results (`docs/instructions.md`, "Failed or infeasible configurations").

A short-trained network is only an INDICATION of the final circuit (bit-widths and key sizes follow the trained weight
ranges). The grid run re-verifies every chosen configuration for real, and a configuration that turns out infeasible
there is recorded as such.

Run under the WSL2 FHE venv:
    ~/.venvs/fhe-fraud-detection/bin/python -m src.fhe.probes.mlp_feasibility --config configs/phase9/mlp_grid.yaml
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from src.config import PROJECT_ROOT, load_config
from src.logging_setup import get_logger

logger = get_logger(__name__)

TIERS = ("top_20", "top_50", "top_100")
RESULT_PREFIX = "PROBE_RESULT "
OUT_PATH = PROJECT_ROOT / "results" / "phase9_mlp" / "feasibility.json"


def _load_tier(tier: str, seed: int, n_rows: int) -> tuple[np.ndarray, ...]:
    from src.fhe.handoff import load_handoff, standardize_features

    manifest, arrays = load_handoff(
        PROJECT_ROOT / "data" / "fhe_handoff" / "phase9" / f"mlp_{tier}.npz",
        PROJECT_ROOT / "results" / "phase9_mlp" / "export" / tier / "handoff_manifest.json",
    )
    params = json.loads((PROJECT_ROOT / manifest["scaler_path"]).read_text())
    X_train = standardize_features(params, arrays["X_train"]).astype(np.float32)
    X_val = standardize_features(params, arrays["X_val"]).astype(np.float32)
    rng = np.random.RandomState(seed)
    idx = np.sort(rng.choice(X_train.shape[0], size=min(n_rows, X_train.shape[0]), replace=False))
    return X_train[idx], arrays["y_train"][idx], X_val, arrays["y_val"], X_train


def probe_one(config: dict[str, Any], tier: str, n_bits: int, max_key_gb: float) -> dict[str, Any]:
    """Runs inside a child process. Returns one candidate's record."""
    from src.fhe.compile import mlp
    from src.fhe.xgb_poc import calibration_positions, peak_rss_mb

    probe_cfg = config["feasibility_probe"]
    cfg = config["mlp"]
    record: dict[str, Any] = {"tier": tier, "n_bits": n_bits, "n_accum_bits": cfg["n_accum_bits"], "train_rows": probe_cfg["train_rows"], "epochs": probe_cfg["epochs"]}
    try:
        Xs, ys, Xv, yv, X_full = _load_tier(tier, config["seed"], probe_cfg["train_rows"])
        t0 = time.perf_counter()
        model, info = mlp.train_qat_mlp({**cfg, "max_epochs": probe_cfg["epochs"], "early_stopping_patience": 1000}, n_bits, Xs, ys, Xv[: probe_cfg["val_rows"]], yv[: probe_cfg["val_rows"]], config["seed"])
        record["train_seconds"] = time.perf_counter() - t0
        record["val_pr_auc_clear_quantized_subsample"] = info["final_val_pr_auc_clear_quantized"]
        record["peak_rss_mb_after_training"] = peak_rss_mb()
    except Exception as exc:  # training itself failing is a result too
        record.update({"stage": "train", "compiled": False, "error_type": type(exc).__name__, "error": str(exc)[:500]})
        return record

    cal = X_full[calibration_positions(X_full, config["compile_calibration"]["rows"], config["compile_calibration"]["include_extremes"], config["seed"])]
    try:
        circuit, compile_seconds = mlp.compile_mlp(model, cal)
        stats = mlp.mlp_stats(model, circuit, cfg, n_bits)
        record.update({"compiled": True, "compile_seconds": compile_seconds, "circuit": stats, "peak_rss_mb_after_compile": peak_rss_mb()})
    except Exception as exc:
        record.update({"stage": "compile", "compiled": False, "error_type": type(exc).__name__, "error": str(exc)[:500]})
        return record

    key_gb = (stats["size_of_bootstrap_keys_bytes"] + stats["size_of_keyswitch_keys_bytes"] + stats["size_of_secret_keys_bytes"]) / 1e9
    record["predicted_key_material_gb"] = key_gb
    if key_gb > max_key_gb:
        record.update({"keygen_attempted": False, "keygen_skipped_reason": f"predicted key material {key_gb:.2f} GB exceeds the {max_key_gb} GB safety limit for the 3.8 GiB WSL memory"})
        return record

    try:
        q = model.quantize_input(Xv[:1])
        t0 = time.perf_counter(); circuit.keygen(); keygen = time.perf_counter() - t0
        t0 = time.perf_counter(); encrypted = circuit.encrypt(q); encrypt = time.perf_counter() - t0
        t0 = time.perf_counter(); result = circuit.run(encrypted); run = time.perf_counter() - t0
        t0 = time.perf_counter(); decrypted = np.asarray(circuit.decrypt(result)); decrypt = time.perf_counter() - t0
        record.update({
            "keygen_attempted": True, "keygen_seconds": keygen, "encrypt_seconds": encrypt, "run_seconds": run, "decrypt_seconds": decrypt,
            "decrypted_equals_simulate": bool(np.array_equal(decrypted[0], circuit.simulate(q)[0])), "peak_rss_mb_after_run": peak_rss_mb(),
        })
        t0 = time.perf_counter()
        for i in range(20):
            circuit.simulate(model.quantize_input(Xv[i : i + 1]))
        record["simulate_seconds_per_row"] = (time.perf_counter() - t0) / 20
    except Exception as exc:
        record.update({"keygen_attempted": True, "stage": "keygen_or_run", "run_ok": False, "error_type": type(exc).__name__, "error": str(exc)[:500], "peak_rss_mb": peak_rss_mb()})
    return record


def run_probe(config_path: str, max_key_gb: float, tiers: tuple[str, ...], bits: list[int]) -> dict[str, Any]:
    config = load_config(config_path)
    rows = []
    for tier in tiers:
        for n_bits in bits:
            logger.info("Probing candidate", extra={"extra_fields": {"tier": tier, "n_bits": n_bits}})
            t0 = time.perf_counter()
            proc = subprocess.run(
                [sys.executable, "-m", "src.fhe.probes.mlp_feasibility", "--config", config_path, "--one", tier, str(n_bits), "--max-key-gb", str(max_key_gb)],
                capture_output=True, text=True, cwd=PROJECT_ROOT,
            )
            line = next((l for l in reversed(proc.stdout.splitlines()) if l.startswith(RESULT_PREFIX)), None)
            if line is not None:
                row = json.loads(line[len(RESULT_PREFIX):])
            else:  # the child died without reporting: almost always the OS killing it for memory
                row = {"tier": tier, "n_bits": n_bits, "compiled": None, "process_returncode": proc.returncode, "error": f"child produced no result (return code {proc.returncode}; -9 = killed, likely out of memory)", "stderr_tail": proc.stderr[-400:]}
            row["wall_seconds"] = time.perf_counter() - t0
            logger.info("Candidate finished", extra={"extra_fields": {k: v for k, v in row.items() if k in ("tier", "n_bits", "compiled", "keygen_attempted", "run_seconds", "error")}})
            rows.append(row)
    payload = {
        "note": "indicative feasibility from a short-trained network on a train subsample; the grid run re-verifies every chosen configuration",
        "machine": {"wsl_ram_gib": 3.8, "cpu_cores": 6}, "max_predicted_key_material_gb_for_keygen": max_key_gb, "candidates": rows,
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/phase9/mlp_grid.yaml")
    parser.add_argument("--one", nargs=2, metavar=("TIER", "BITS"), help="child mode: probe a single candidate and print its record")
    parser.add_argument("--max-key-gb", type=float, default=2.4)
    parser.add_argument("--bits", type=int, nargs="+", help="override the candidate bit-widths from the config")
    args = parser.parse_args()
    warnings.filterwarnings("ignore")
    if args.one:
        config = load_config(args.config)
        record = probe_one(config, args.one[0], int(args.one[1]), args.max_key_gb)
        # flushed explicitly: the Concrete runtime can end the process without flushing Python's buffers (measured: children
        # that had just run real encryption exited with code 0 and an empty stdout)
        sys.stdout.write("\n" + RESULT_PREFIX + json.dumps(record, default=str) + "\n")
        sys.stdout.flush()
        return
    config = load_config(args.config)
    run_probe(args.config, args.max_key_gb, TIERS, args.bits or config["feasibility_probe"]["candidate_bits"])


if __name__ == "__main__":
    main()
