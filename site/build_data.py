"""Collect the portfolio page's numbers from COMMITTED result files only (read-only; nothing is recomputed or estimated).

    python site/build_data.py        # writes site/data/results.js (window.CIPHRAUD_DATA = {...})

Sources: results/phase8_research/summary.json, results/phase9_mlp/summary.json, results/phase9_mlp/pareto/combined_table.csv,
results/phase12_test_eval/test_results.json (+ freeze.json and run_complete.json hashes, shown for traceability).
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"


def load(path: str):
    return json.loads((R / path).read_text(encoding="utf-8"))


def sha(path: str) -> str:
    return hashlib.sha256((R / path).read_bytes()).hexdigest()


def main() -> None:
    test = load("phase12_test_eval/test_results.json")["configurations"]
    table = {row["label"]: row for row in csv.DictReader((R / "phase9_mlp/pareto/combined_table.csv").open(encoding="utf-8"))}
    phase9 = {c["label"]: c for c in load("phase9_mlp/summary.json")["configurations"]}
    phase8 = {c["label"]: c for c in load("phase8_research/summary.json")["configurations"]}
    labels = list(phase8) + list(phase9)
    rows = []
    for label in labels:
        t = test[label]
        src = phase8.get(label) or phase9[label]
        row = {
            "label": label, "model": src["model_type"], "n_bits": src["n_bits"], "status": src["status"],
            "val_pr_auc": src["quantized_pr_auc"], "float_pr_auc_val": src["float_pr_auc"],
            "test_pr_auc": t["quantized"]["pr_auc"], "test_pr_auc_ci": t["quantized"]["pr_auc_ci"],
            "test_roc_auc": t["quantized"]["roc_auc"], "test_roc_auc_ci": t["quantized"]["roc_auc_ci"],
            "t3_agreement_val": src.get("t3_decision_agreement"), "compile_s": src.get("compile_seconds"),
        }
        if label in table:
            tr = table[label]
            row.update(latency_mean_s=float(tr["fhe_latency_mean_s"]), latency_std_s=float(tr["fhe_latency_std_s"]), peak_rss_mb=float(tr["peak_rss_mb"]),
                       ciphertext_total_bytes=int(float(tr["ciphertext_total_bytes"])), key_total_bytes=int(float(tr["key_total_bytes"])),
                       bootstraps=int(float(tr["programmable_bootstraps"])))
        else:
            row.update(latency_mean_s=None, key_material_gb=src.get("key_material_gb"), bootstraps=src.get("programmable_bootstraps"), peak_rss_mb=src.get("peak_rss_mb"))
        rows.append(row)
    data = {
        "rows": rows,
        "test_set": {"rows": 104637, "positives": 3562, "positive_rate": 0.0340414958379923},
        "hashes": {
            "freeze.json": sha("phase12_test_eval/freeze.json"), "test_results.json": sha("phase12_test_eval/test_results.json"),
            "summary.md": sha("phase12_test_eval/summary.md"), "run_complete.json": sha("phase12_test_eval/run_complete.json"),
        },
    }
    out = ROOT / "site" / "data" / "results.js"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("window.CIPHRAUD_DATA = " + json.dumps(data, indent=1) + ";\n", encoding="utf-8", newline="\n")
    print(f"wrote {out.relative_to(ROOT)}: {len(rows)} configurations")


if __name__ == "__main__":
    main()
