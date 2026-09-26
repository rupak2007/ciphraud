"""Old-vs-corrected comparison for the six Phase 8 Logistic-Regression configurations.

The pre-fix results (compiled LR built on RAW features with standardized-space coefficients) are archived,
unmodified, in `results/phase8_research/prefix_lr_scaler_bug/`; the corrected ones (compiled LR consuming
standardized inputs) are in `results/phase8_research/lr_*`. This script only READS those files and writes
`results/phase8_research/lr_scaler_fix_comparison.{json,md}` -- every number in `docs/research.md`'s LR
erratum table regenerates from here:

    python -m src.analysis.lr_scaler_fix_comparison
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.config import PROJECT_ROOT

RESULTS = PROJECT_ROOT / "results" / "phase8_research"
ARCHIVE = RESULTS / "prefix_lr_scaler_bug"
LABELS = ["lr_top20_bits8", "lr_top20_bits16", "lr_top50_bits8", "lr_top50_bits16", "lr_top100_bits8", "lr_top100_bits16"]


def _summarize(metrics: dict[str, Any]) -> dict[str, Any]:
    t3 = metrics["accuracy"]["gates"]["t3"]
    lat = metrics["fhe_latency"]["total"]
    return {
        "status": metrics["status"],
        "gates_passed": metrics["gates_passed"],
        "t3_decision_agreement": t3["decision_agreement"],
        "t3_pr_auc_drop": t3["pr_auc_drop"],
        "quantized_pr_auc": t3["quantized_pr_auc"],
        "float_pr_auc": t3["float_pr_auc"],
        "quantized_roc_auc": t3["quantized_roc_auc"],
        "float_roc_auc": t3["float_roc_auc"],
        "mean_abs_prob_diff": t3["mean_abs_prob_diff"],
        "compile_seconds": metrics["compile_seconds"],
        "fhe_latency_mean_seconds": lat["mean_seconds"],
        "fhe_latency_std_seconds": lat["std_seconds"],
        "ciphertext_size_bytes": metrics["ciphertext_size_bytes"],
    }


def build() -> dict[str, Any]:
    rows = {}
    for label in LABELS:
        pre = json.loads((ARCHIVE / label / "metrics.json").read_text())
        post = json.loads((RESULTS / label / "metrics.json").read_text())
        rows[label] = {"pre_fix": _summarize(pre), "corrected": _summarize(post)}
    return rows


def to_markdown(rows: dict[str, Any]) -> str:
    head = (
        "| Config | Pre-fix agreement | Pre-fix PR-AUC drop | Corrected agreement | Corrected PR-AUC drop | "
        "Corrected quantized / float PR-AUC | Corrected T3 |\n|---|---|---|---|---|---|---|\n"
    )
    body = ""
    for label, r in rows.items():
        a, b = r["pre_fix"], r["corrected"]
        body += (
            f"| `{label}` | {a['t3_decision_agreement']:.4f} | {a['t3_pr_auc_drop']:.4f} | {b['t3_decision_agreement']:.5f} | "
            f"{b['t3_pr_auc_drop']:.5f} | {b['quantized_pr_auc']:.4f} / {b['float_pr_auc']:.4f} | "
            f"{'PASS' if b['gates_passed']['t3'] else 'FAIL'} |\n"
        )
    return head + body


def main() -> None:
    rows = build()
    (RESULTS / "lr_scaler_fix_comparison.json").write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
    (RESULTS / "lr_scaler_fix_comparison.md").write_text(to_markdown(rows), encoding="utf-8")
    print(to_markdown(rows))


if __name__ == "__main__":
    main()
