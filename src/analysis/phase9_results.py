"""Load, validate and cross-check the Phase 9 (quantized MLP) result files.

Read-only over `results/phase9_mlp/` and `configs/phase9/mlp_grid.yaml`. It reuses Phase 8's raw-trial checks
(`phase8_results.verify_means_against_raw`) so "reported means equal the raw per-trial data" is one definition for all
three models, and adds the MLP-specific rules (`docs/plan.md` Phase 9 tests: correctness validation per MLP
configuration; infeasible combinations explicitly logged):

    python -m src.analysis.phase9_results          # prints a validation report, exit 1 on any problem

A configuration is in exactly one of these states, and each state has its own required content:
  * `passed` / `failed_accuracy_gates` -- ran to completion: all four gates, circuit statistics, 5 latency trials;
  * `infeasible_key_memory`            -- compiled, plaintext gates recorded, FHE not attempted because the keys do not fit;
  * `<stage>_failed`                   -- an exception, recorded with its type, message and traceback.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

from src.analysis.phase8_results import GATES, _close, verify_means_against_raw
from src.config import PROJECT_ROOT, load_config
from src.data import provenance as data_provenance

RESULTS_DIR = PROJECT_ROOT / "results" / "phase9_mlp"
GRID_CONFIG = "configs/phase9/mlp_grid.yaml"
COMPLETED = {"passed", "failed_accuracy_gates"}
FAILURE_STAGES = ("train", "compile", "accuracy", "accuracy_fhe", "latency")
FAILED_STATUSES = {f"{s}_failed" for s in FAILURE_STAGES}
REQUIRED_KEYS = ("label", "model_type", "tier", "n_bits", "n_features", "seed", "status", "test_partition_touched")
REQUIRED_PROVENANCE_KEYS = ("config_hash", "config_path", "git_commit", "library_versions", "seed", "timestamp_utc")


def load_grid_config() -> dict[str, Any]:
    return load_config(GRID_CONFIG)


def load_results(results_dir: Path = RESULTS_DIR) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for entry in load_grid_config()["configurations"]:
        d = results_dir / entry["label"]
        if (d / "metrics.json").exists():
            out[entry["label"]] = {
                "metrics": json.loads((d / "metrics.json").read_text(encoding="utf-8")),
                "provenance": json.loads((d / "provenance.json").read_text(encoding="utf-8")) if (d / "provenance.json").exists() else None,
            }
    return out


def _validate_completed(label: str, m: dict[str, Any], config: dict[str, Any]) -> list[str]:
    problems: list[str] = []
    gp = m.get("gates_passed", {})
    if set(gp) != set(GATES) or not all(isinstance(v, bool) for v in gp.values()):
        return [f"{label}: gates_passed must hold exactly {GATES} as booleans, got {gp!r}"]
    gates = m["accuracy"]["gates"]
    for g in GATES:
        if gates[g].get("passed") != gp[g]:
            problems.append(f"{label}: gates_passed.{g} disagrees with accuracy.gates.{g}.passed")
    if m["all_gates_passed"] != all(gp.values()) or (m["status"] == "passed") != all(gp.values()):
        problems.append(f"{label}: status / all_gates_passed disagree with the four gates")

    t3, tol = gates["t3"], config["tolerances"]
    want = t3["decision_agreement"] >= tol["t3_min_decision_agreement"] and t3["pr_auc_drop"] <= tol["t3_max_pr_auc_drop"]
    if bool(t3["passed"]) != want:
        problems.append(f"{label}: T3 recorded passed={t3['passed']} but the configured bars give {want}")
    if not _close(t3["pr_auc_drop"], t3["float_pr_auc"] - t3["quantized_pr_auc"], rel=1e-6, abs_=1e-9):
        problems.append(f"{label}: T3 pr_auc_drop != float_pr_auc - quantized_pr_auc")
    for src, ref in (("quantized_full_metrics", "quantized_pr_auc"), ("float_full_metrics", "float_pr_auc")):
        if not _close(m["accuracy"][src]["pr_auc"], t3[ref], rel=1e-9, abs_=1e-12):
            problems.append(f"{label}: {src}.pr_auc differs from the T3 gate's {ref}")
    if not (gates["t0"].get("reload_reproduces_clear_quantized_logits_exactly") is True and gates["t0"]["max_abs_diff"] == 0.0) and gp["t0"]:
        problems.append(f"{label}: T0 marked passed without an exact reload")
    for g in ("t2", "t1_correctness"):
        if gp[g] != (gates[g]["exact_integer_match"] is True):
            problems.append(f"{label}: {g} passed flag does not equal its exact_integer_match")

    cs = m["circuit_stats"]
    if not (isinstance(cs["programmable_bootstrap_count"], int) and cs["programmable_bootstrap_count"] > 0):
        problems.append(f"{label}: an MLP circuit must contain table lookups (programmable_bootstrap_count > 0)")
    if not (0 < cs["max_integer_bit_width"] <= 16):
        problems.append(f"{label}: max_integer_bit_width {cs['max_integer_bit_width']} outside 1..16")
    if cs["n_w_bits"] != m["n_bits"] or cs["n_a_bits"] != m["n_bits"] or cs["n_accum_bits"] != config["mlp"]["n_accum_bits"]:
        problems.append(f"{label}: circuit bit-widths disagree with the configuration")
    if not (m["compile_seconds"] > 0 and m["peak_rss_mb"] > 0 and m["ciphertext_size_bytes"]["input"] > 0 and m["key_size_bytes"]["bootstrap"] > 0):
        problems.append(f"{label}: compile time / memory / ciphertext / key sizes must be positive")
    if m["fhe_latency"].get("outputs_reproducible") is not True:
        problems.append(f"{label}: fhe_latency.outputs_reproducible is not True")
    rs = m["row_selection"]
    if rs.get("seed") != config["seed"] or rs.get("n_positive") != config["row_selection"]["n_positive"]:
        problems.append(f"{label}: row_selection {rs!r} does not follow the grid config")
    if m["training"]["qat"]["epochs_run"] < 1 or m["training"]["float_twin"]["epochs_run"] < 1:
        problems.append(f"{label}: training record is missing epochs")
    problems += [f"{label}: {p}" for p in verify_means_against_raw(m, config["latency_trials"])]
    return problems


def validate_result(metrics: dict[str, Any], provenance: dict[str, Any] | None, config: dict[str, Any], entry: dict[str, Any]) -> list[str]:
    label = entry["label"]
    missing = [k for k in REQUIRED_KEYS if k not in metrics]
    if missing:
        return [f"{label}: metrics.json missing keys {missing}"]
    problems: list[str] = []
    match = re.fullmatch(r"mlp_top(\d+)_bits(\d+)", label)
    if not match:
        problems.append(f"{label}: label does not match the Phase 9 naming scheme")
    else:
        top, bits = int(match.group(1)), int(match.group(2))
        if metrics["model_type"] != "mlp" or metrics["tier"] != f"top_{top}" or metrics["n_features"] != top or metrics["n_bits"] != bits:
            problems.append(f"{label}: identity fields disagree with the label")
        if entry["tier"] != f"top_{top}" or int(entry["n_bits"]) != bits:
            problems.append(f"{label}: grid entry disagrees with the label")
    if metrics["seed"] != config["seed"]:
        problems.append(f"{label}: seed {metrics['seed']} != grid seed {config['seed']}")
    if metrics["test_partition_touched"] is not False:
        problems.append(f"{label}: test_partition_touched must be False")

    status = metrics["status"]
    if status in COMPLETED:
        problems += _validate_completed(label, metrics, config)
    elif status == "infeasible_key_memory":
        limit = float(config.get("max_key_material_gb", 2.4))
        if not metrics.get("key_material_gb", 0) > limit or "infeasible_reason" not in metrics:
            problems.append(f"{label}: infeasible_key_memory requires key_material_gb > {limit} and a reason")
        if "fhe_latency" in metrics:
            problems.append(f"{label}: an infeasible configuration must not carry FHE latency")
        gp = metrics.get("gates_passed", {})
        if gp.get("t2") is not None or gp.get("t1_correctness") is not None or not isinstance(gp.get("t0"), bool) or not isinstance(gp.get("t3"), bool):
            problems.append(f"{label}: infeasible record must hold T0/T3 booleans and null T2/T1")
        if metrics.get("circuit_stats", {}).get("programmable_bootstrap_count", 0) <= 0:
            problems.append(f"{label}: infeasible record must still carry the circuit statistics that explain it")
    elif status in FAILED_STATUSES:
        err = metrics.get("error", {})
        if not all(k in err for k in ("stage", "type", "message", "traceback")):
            problems.append(f"{label}: a failed configuration must record stage, type, message and traceback")
    else:
        problems.append(f"{label}: unknown status {status!r}")

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
    problems: list[str] = []
    want = [e["label"] for e in config["configurations"]]
    got = [r["label"] for r in summary["configurations"]]
    if got != want:
        problems.append(f"summary: labels {got} != grid {want}")
    for row in summary["configurations"]:
        res = results.get(row["label"])
        if res is None:
            if row["status"] != "not_run":
                problems.append(f"summary: {row['label']} claims a result that does not exist")
            continue
        m = res["metrics"]
        if row["status"] != m["status"]:
            problems.append(f"summary: {row['label']} status {row['status']!r} != metrics.json {m['status']!r}")
        if m.get("fhe_latency") and not _close(row["fhe_mean_seconds"], m["fhe_latency"]["total"]["mean_seconds"]):
            problems.append(f"summary: {row['label']} fhe_mean_seconds disagrees with metrics.json")
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
    return {"n_configurations": len(config["configurations"]), "n_results": len(results), "problems": problems}


def main() -> None:
    report = validate_all()
    print(f"Phase 9 result validation: {report['n_results']}/{report['n_configurations']} configurations found")
    for p in report["problems"]:
        print("  PROBLEM:", p)
    print("OK -- no problems" if not report["problems"] else f"{len(report['problems'])} problem(s)")
    sys.exit(1 if report["problems"] else 0)


if __name__ == "__main__":
    main()
