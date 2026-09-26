"""Load, validate and cross-check the completed Phase 8 result files.

Everything here only READS `results/phase8_research/` and `configs/phase8/research_grid.yaml`.
It is the single place that defines what a well-formed Phase 8 result is (`docs/plan.md` Phase 8 tests:
"results-schema validation; spot-check that reported means match raw per-trial data"), so the Pareto
analysis and the tests share one definition:

    python -m src.analysis.phase8_results          # prints a validation report, exit 1 on any problem
"""

from __future__ import annotations

import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import numpy as np

from src.config import PROJECT_ROOT, load_config
from src.data import provenance as data_provenance

RESULTS_DIR = PROJECT_ROOT / "results" / "phase8_research"
GRID_CONFIG = "configs/phase8/research_grid.yaml"
GATES = ("t0", "t1_correctness", "t2", "t3")
STATUSES = {"passed", "failed_accuracy_gates"}  # every grid configuration ran to completion; nothing else is legal here
LATENCY_BLOCKS = ("encrypt", "run", "decrypt", "total")
REL_TOL = 1e-9

REQUIRED_METRICS_KEYS = (
    "accuracy", "all_gates_passed", "ciphertext_size_bytes", "circuit_stats", "compile_seconds", "fhe_latency", "gates_passed",
    "key_size_bytes", "label", "model_type", "n_bits", "n_features", "peak_rss_mb", "peak_rss_mb_after_compile",
    "plaintext_latency", "row_selection", "seed", "source_config_hash", "status", "test_partition_touched", "tier",
)
REQUIRED_PROVENANCE_KEYS = ("config_hash", "config_path", "git_commit", "library_versions", "seed", "timestamp_utc")


def load_grid_config() -> dict[str, Any]:
    return load_config(GRID_CONFIG)


def entry_config_hash(config: dict[str, Any], entry: dict[str, Any]) -> str:
    """Per-entry config hash, derived after the fact.

    The Phase 8 runner recorded only the whole-grid hash in `provenance.json`. This hash covers exactly
    what determines one configuration's result -- its own grid entry plus every grid-wide setting
    (seed, trial counts, sample sizes, tolerances) -- and is computed with the project's own
    `compute_config_hash`. It is NOT stored in the runner's output files.
    """
    shared = {k: v for k, v in config.items() if k not in ("configurations", "output")}
    return data_provenance.compute_config_hash({"entry": entry, "grid_settings": shared})


def load_results(results_dir: Path = RESULTS_DIR) -> dict[str, dict[str, Any]]:
    """`{label: {"metrics": ..., "provenance": ...}}` for every grid configuration that has a metrics.json."""
    out: dict[str, dict[str, Any]] = {}
    for entry in load_grid_config()["configurations"]:
        d = results_dir / entry["label"]
        if (d / "metrics.json").exists():
            out[entry["label"]] = {
                "metrics": json.loads((d / "metrics.json").read_text(encoding="utf-8")),
                "provenance": json.loads((d / "provenance.json").read_text(encoding="utf-8")) if (d / "provenance.json").exists() else None,
            }
    return out


def _close(a: float, b: float, rel: float = REL_TOL, abs_: float = 1e-12) -> bool:
    return math.isclose(a, b, rel_tol=rel, abs_tol=abs_)


def check_latency_block(name: str, block: dict[str, Any], expected_trials: int) -> list[str]:
    """Reported n/mean/std/min/max must equal what the raw per-trial values give (sample std, ddof=1)."""
    problems: list[str] = []
    trials = block.get("trials_seconds")
    if not isinstance(trials, list) or not trials:
        return [f"{name}: missing trials_seconds"]
    if block.get("n_trials") != len(trials):
        problems.append(f"{name}: n_trials={block.get('n_trials')} but {len(trials)} raw trials")
    if len(trials) != expected_trials:
        problems.append(f"{name}: {len(trials)} raw trials, grid config requires {expected_trials}")
    if any((not isinstance(t, (int, float))) or (not math.isfinite(t)) or t < 0 for t in trials):
        problems.append(f"{name}: non-finite or negative raw trial value")
        return problems
    arr = np.asarray(trials, dtype=np.float64)
    expected = {"mean_seconds": float(arr.mean()), "std_seconds": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
                "min_seconds": float(arr.min()), "max_seconds": float(arr.max())}
    for key, want in expected.items():
        if key not in block:
            problems.append(f"{name}: missing {key}")
        elif not _close(block[key], want):
            problems.append(f"{name}: reported {key}={block[key]!r} but raw trials give {want!r}")
    tp = block.get("throughput_requests_per_second")
    if tp is not None and expected["mean_seconds"] > 0 and not _close(tp, 1.0 / expected["mean_seconds"]):
        problems.append(f"{name}: throughput {tp!r} != 1/mean ({1.0 / expected['mean_seconds']!r})")
    return problems


def verify_means_against_raw(metrics: dict[str, Any], expected_trials: int) -> list[str]:
    """Recompute every reported latency statistic from its raw trial list, and check the blocks agree."""
    problems: list[str] = []
    lat = metrics["fhe_latency"]
    for name in LATENCY_BLOCKS:
        problems += check_latency_block(f"fhe_latency.{name}", lat[name], expected_trials)
    problems += check_latency_block("plaintext_latency", metrics["plaintext_latency"], expected_trials)
    try:
        parts = [np.asarray(lat[n]["trials_seconds"], dtype=np.float64) for n in ("encrypt", "run", "decrypt")]
        total = np.asarray(lat["total"]["trials_seconds"], dtype=np.float64)
        # `total` is timed as its own interval around the whole round trip, so it is never below the sum of
        # the three stages; measured gap in all 12 configurations is <= 12 ms (<= 0.05% of total).
        gap = total - (parts[0] + parts[1] + parts[2])
        if np.any(gap < -1e-6) or np.any(gap > 1e-3 * total):
            problems.append("fhe_latency: total must be >= encrypt + run + decrypt per trial, within 0.1%")
    except (KeyError, ValueError) as exc:
        problems.append(f"fhe_latency: cannot combine per-stage trials ({exc})")
    return problems


def validate_result(metrics: dict[str, Any], provenance: dict[str, Any] | None, config: dict[str, Any], entry: dict[str, Any]) -> list[str]:
    """Schema, internal-consistency and identity checks for one configuration. Empty list == valid."""
    label = entry["label"]
    problems: list[str] = []
    missing = [k for k in REQUIRED_METRICS_KEYS if k not in metrics]
    if missing:
        return [f"{label}: metrics.json missing keys {missing}"]

    m = re.fullmatch(r"(lr|xgboost)_top(\d+)_bits(\d+)", label)
    if not m:
        problems.append(f"{label}: label does not match the grid naming scheme")
    else:
        model, top, bits = m.group(1), int(m.group(2)), int(m.group(3))
        if metrics["model_type"] != model or entry["model_type"] != model:
            problems.append(f"{label}: model_type {metrics['model_type']!r} != {model!r}")
        if metrics["n_features"] != top:
            problems.append(f"{label}: n_features {metrics['n_features']} != {top}")
        if metrics["n_bits"] != bits or entry["n_bits"] != bits:
            problems.append(f"{label}: n_bits {metrics['n_bits']} != {bits}")
        want_tier = f"top_{top}" if model == "xgboost" else None
        if metrics["tier"] != want_tier:
            problems.append(f"{label}: tier {metrics['tier']!r} != {want_tier!r}")
    if metrics["label"] != label:
        problems.append(f"{label}: metrics label is {metrics['label']!r}")
    if metrics["status"] not in STATUSES:
        problems.append(f"{label}: status {metrics['status']!r} is not a completed-run status")
    if metrics["seed"] != config["seed"]:
        problems.append(f"{label}: seed {metrics['seed']} != grid seed {config['seed']}")
    if metrics["test_partition_touched"] is not False:
        problems.append(f"{label}: test_partition_touched is {metrics['test_partition_touched']!r}, must be False")

    gp = metrics["gates_passed"]
    if set(gp) != set(GATES) or not all(isinstance(v, bool) for v in gp.values()):
        problems.append(f"{label}: gates_passed must hold exactly {GATES} as booleans, got {gp!r}")
    else:
        if metrics["all_gates_passed"] != all(gp.values()):
            problems.append(f"{label}: all_gates_passed disagrees with the four gates")
        if (metrics["status"] == "passed") != all(gp.values()):
            problems.append(f"{label}: status {metrics['status']!r} disagrees with the four gates")
        gates = metrics["accuracy"]["gates"]
        for g in GATES:
            if gates[g].get("passed") != gp[g]:
                problems.append(f"{label}: gates_passed.{g} disagrees with accuracy.gates.{g}.passed")
        t3 = gates["t3"]
        tol = config["tolerances"]
        want_t3 = t3["decision_agreement"] >= tol["t3_min_decision_agreement"] and t3["pr_auc_drop"] <= tol["t3_max_pr_auc_drop"]
        if bool(t3["passed"]) != want_t3:
            problems.append(f"{label}: T3 recorded passed={t3['passed']} but the configured bars give {want_t3}")
        if not _close(t3["pr_auc_drop"], t3["float_pr_auc"] - t3["quantized_pr_auc"], rel=1e-6, abs_=1e-9):
            problems.append(f"{label}: T3 pr_auc_drop != float_pr_auc - quantized_pr_auc")
        full = metrics["accuracy"]
        if not _close(full["quantized_full_metrics"]["pr_auc"], t3["quantized_pr_auc"], rel=1e-9, abs_=1e-12):
            problems.append(f"{label}: quantized PR-AUC differs between T3 gate and full metrics")
        if not _close(full["float_full_metrics"]["pr_auc"], t3["float_pr_auc"], rel=1e-9, abs_=1e-12):
            problems.append(f"{label}: float PR-AUC differs between T3 gate and full metrics")
        for name in ("quantized_pr_auc", "float_pr_auc", "decision_agreement"):
            if not 0.0 <= t3[name] <= 1.0:
                problems.append(f"{label}: T3 {name}={t3[name]} outside [0, 1]")

    stats = metrics["circuit_stats"]
    pbs = stats.get("programmable_bootstrap_count")
    boot = metrics["key_size_bytes"]["bootstrap"]
    if entry["model_type"] == "lr" and (pbs != 0 or boot != 0):
        problems.append(f"{label}: an LR circuit should have 0 bootstraps and no bootstrap key (got {pbs}, {boot})")
    if entry["model_type"] == "xgboost" and not (isinstance(pbs, int) and pbs > 0 and boot > 0):
        problems.append(f"{label}: an XGBoost circuit must have bootstraps and a bootstrap key (got {pbs}, {boot})")
    for k in ("input", "output"):
        if not metrics["ciphertext_size_bytes"][k] > 0:
            problems.append(f"{label}: ciphertext_size_bytes.{k} not positive")
    if not (metrics["compile_seconds"] > 0 and metrics["peak_rss_mb"] > 0):
        problems.append(f"{label}: compile_seconds / peak_rss_mb not positive")
    if metrics["fhe_latency"].get("outputs_reproducible") is not True:
        problems.append(f"{label}: fhe_latency.outputs_reproducible is not True")

    rs = metrics["row_selection"]
    if rs.get("seed") != config["seed"] or rs.get("n_positive") != config["row_selection"]["n_positive"]:
        problems.append(f"{label}: row_selection {rs!r} does not follow the grid config")
    problems += verify_means_against_raw(metrics, config["latency_trials"])

    if provenance is None:
        problems.append(f"{label}: provenance.json missing")
    else:
        pmissing = [k for k in REQUIRED_PROVENANCE_KEYS if k not in provenance]
        if pmissing:
            problems.append(f"{label}: provenance.json missing keys {pmissing}")
        else:
            if provenance["config_path"] != GRID_CONFIG:
                problems.append(f"{label}: provenance config_path {provenance['config_path']!r}")
            if provenance["config_hash"] != data_provenance.compute_config_hash(config):
                problems.append(f"{label}: provenance config_hash does not match the current grid config (config edited since the run?)")
            if not re.fullmatch(r"[0-9a-f]{40}", provenance["git_commit"]):
                problems.append(f"{label}: provenance git_commit is not a 40-hex commit id")
    return problems


def validate_summary(summary: dict[str, Any], results: dict[str, dict[str, Any]], config: dict[str, Any]) -> list[str]:
    """`summary.json` must list every grid configuration in grid order and agree with each metrics.json."""
    problems: list[str] = []
    want = [e["label"] for e in config["configurations"]]
    got = [r["label"] for r in summary["configurations"]]
    if got != want:
        problems.append(f"summary: labels {got} != grid {want}")
    if summary["all_run"] != all(r["status"] != "not_run" for r in summary["configurations"]):
        problems.append("summary: all_run inconsistent")
    if summary["all_passed"] != all(bool(r.get("all_gates_passed")) for r in summary["configurations"]):
        problems.append("summary: all_passed inconsistent")
    for row in summary["configurations"]:
        res = results.get(row["label"])
        if res is None:
            problems.append(f"summary: {row['label']} has no metrics.json")
            continue
        m = res["metrics"]
        pairs = {
            "status": (row["status"], m["status"]), "all_gates_passed": (row.get("all_gates_passed"), m["all_gates_passed"]),
            "compile_seconds": (row.get("compile_seconds"), m["compile_seconds"]),
            "fhe_mean_seconds": (row.get("fhe_mean_seconds"), m["fhe_latency"]["total"]["mean_seconds"]),
            "fhe_std_seconds": (row.get("fhe_std_seconds"), m["fhe_latency"]["total"]["std_seconds"]),
            "quantized_pr_auc": (row.get("quantized_pr_auc"), m["accuracy"]["quantized_full_metrics"]["pr_auc"]),
            "t3_decision_agreement": (row.get("t3_decision_agreement"), m["accuracy"]["gates"]["t3"]["decision_agreement"]),
        }
        for key, (a, b) in pairs.items():
            same = a == b if not isinstance(a, float) else _close(a, b)
            if not same:
                problems.append(f"summary: {row['label']} {key} {a!r} != metrics.json {b!r}")
    return problems


def validate_all(results_dir: Path = RESULTS_DIR) -> dict[str, Any]:
    config = load_grid_config()
    results = load_results(results_dir)
    problems: list[str] = []
    for entry in config["configurations"]:
        res = results.get(entry["label"])
        if res is None:
            problems.append(f"{entry['label']}: no metrics.json")
            continue
        problems += validate_result(res["metrics"], res["provenance"], config, entry)
    summary_path = results_dir / "summary.json"
    if summary_path.exists():
        problems += validate_summary(json.loads(summary_path.read_text(encoding="utf-8")), results, config)
    else:
        problems.append("summary.json missing")
    hashes = {e["label"]: entry_config_hash(config, e) for e in config["configurations"]}
    if len(set(hashes.values())) != len(hashes):
        problems.append("per-entry config hashes are not unique")
    return {"n_configurations": len(config["configurations"]), "n_results": len(results), "problems": problems, "entry_config_hashes": hashes}


def main() -> None:
    report = validate_all()
    print(f"Phase 8 result validation: {report['n_results']}/{report['n_configurations']} configurations found")
    for p in report["problems"]:
        print("  PROBLEM:", p)
    print("OK -- no problems" if not report["problems"] else f"{len(report['problems'])} problem(s)")
    sys.exit(1 if report["problems"] else 0)


if __name__ == "__main__":
    main()
