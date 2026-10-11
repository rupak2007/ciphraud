"""Cross-check headline numbers quoted in report.md against the committed result records.

Run: python -m src.analysis.report_check   (exit 1 on any mismatch). Reads only; changes nothing.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load(rel: str) -> dict:
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def main() -> int:
    report = (ROOT / "report.md").read_text(encoding="utf-8")
    problems: list[str] = []

    def need(label: str, text: str) -> None:
        if text not in report:
            problems.append(f"{label}: '{text}' not found in report.md")

    for label in ("lr_top20_bits16", "lr_top50_bits16", "lr_top100_bits16", "lr_top20_bits8", "lr_top50_bits8", "lr_top100_bits8"):
        m = _load(f"results/phase8_research/{label}/metrics.json")
        lat = m["fhe_latency"]["total"] if "total" in m["fhe_latency"] else None
        acc = m["accuracy"]
        need(label + " quantized PR-AUC", f"{acc['quantized_full_metrics']['pr_auc']:.4f}")
        if lat:
            need(label + " latency", f"{lat['mean_seconds'] * 1000:.2f} ms")
    for label in ("xgboost_top20_bits8", "xgboost_top20_bits14", "xgboost_top50_bits8", "xgboost_top50_bits14", "xgboost_top100_bits8", "xgboost_top100_bits14"):
        m = _load(f"results/phase8_research/{label}/metrics.json")
        need(label + " quantized PR-AUC", f"{m['accuracy']['quantized_full_metrics']['pr_auc']:.4f}")
        tot = m["fhe_latency"].get("total")
        if tot:
            need(label + " latency", f"{tot['mean_seconds']:,.1f} s")
    for label in ("mlp_top20_bits3", "mlp_top20_bits4", "mlp_top50_bits3", "mlp_top50_bits4", "mlp_top100_bits3", "mlp_top100_bits4"):
        m = _load(f"results/phase9_mlp/{label}/metrics.json")
        need(label + " quantized PR-AUC", f"{m['accuracy']['quantized_full_metrics']['pr_auc']:.4f}")
        need(label + " float PR-AUC", f"{m['accuracy']['float_full_metrics']['pr_auc']:.4f}")
        if m["status"] == "failed_accuracy_gates":
            need(label + " status", "failed gate")
    s = _load("results/phase10_client_server/summary.json")
    for mod in s["models"]:
        assert mod["all_requests_exact_match"] and mod["status"] == "passed", mod["id"]
    x = next(m for m in s["models"] if m["id"] == "xgboost_top50_bits14")
    need("xgboost phase10 fhe run", f"{x['fhe_run_mean_s']:,.2f} s")
    need("xgboost phase10 probability", "0.0999311")
    l = next(m for m in s["models"] if m["id"] == "lr_top20_bits16")
    need("lr phase10 mean", f"{l['fhe_run_mean_s'] * 1000:.0f} ms")
    ml = next(m for m in s["models"] if m["id"] == "mlp_top20_bits3")
    need("mlp phase10 mean", f"{ml['fhe_run_mean_s']:.2f} s")
    need("MLP T3 disclosure", "failed the Phase 9\nquantization-accuracy gate".replace("\n", " "))
    need("all six MLP T3 failures", "All six MLP configurations failed T3")
    for p in problems:
        print("MISMATCH", p)
    print("OK -- all quoted numbers found" if not problems else f"{len(problems)} problem(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
