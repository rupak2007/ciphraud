"""Combined Pareto and model-effect analysis: LR + XGBoost (Phase 8) + quantized MLP (Phase 9).

Reads ONLY completed result files (`results/phase8_research/*/metrics.json` via `phase8_pareto.build_rows`, and
`results/phase9_mlp/*/metrics.json`), validated first; runs nothing and never touches the test partition. Writes to
`results/phase9_mlp/pareto/` (the committed Phase 8 outputs in `results/phase8_research/pareto/` are not rewritten):

    combined_table.csv / .json     one row per configuration that has an FHE latency (every number traces to a metrics.json)
    combined_frontiers.json        frontiers per cost axis, effect tables, and the MLP configurations without an FHE latency
    combined_tables.md             the same, rendered for docs/fhe_mlp.md
    fig_combined_accuracy_vs_latency.png, fig_combined_accuracy_vs_memory_ciphertext.png

    python -m src.analysis.phase9_pareto

Definitions are Phase 8's, unchanged (`phase8_pareto` docstring): accuracy = validation PR-AUC of the clear-quantized model;
latency = mean of 5 repeated executions of ONE fixed row (position 32,148), VM monotonic clock; memory = whole-process peak
RSS; ciphertext = input + output bytes; dominance and the "within one standard deviation" noise check as before. For the
MLP, "float" means its float twin (decision D2). An MLP configuration that is infeasible or failed has no latency, so it
cannot sit on a latency/memory/ciphertext frontier; it is listed in its own table with its status and reason -- never dropped.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from src.analysis import phase8_pareto as p8
from src.analysis import phase8_results as r8
from src.analysis import phase9_results as r9
from src.data import provenance as data_provenance

OUT_DIR = r9.RESULTS_DIR / "pareto"
PRIOR_ART_NEURAL_NETWORK_SECONDS = 0.296  # docs/prd.md Sec.11: "296ms for a neural network" (thesis figure, setup unverified)


def _row9(metrics: dict[str, Any], config: dict[str, Any], entry: dict[str, Any]) -> dict[str, Any]:
    t3 = metrics["accuracy"]["gates"]["t3"]
    lat = metrics["fhe_latency"]["total"]
    keys = metrics["key_size_bytes"]
    cs = metrics["circuit_stats"]
    return {
        "label": entry["label"], "model": "mlp", "n_features": metrics["n_features"], "n_bits": metrics["n_bits"],
        "status": metrics["status"], "t3_passed": bool(metrics["gates_passed"]["t3"]), "all_gates_passed": bool(metrics["all_gates_passed"]),
        "quantized_pr_auc": metrics["accuracy"]["quantized_full_metrics"]["pr_auc"], "float_pr_auc": metrics["accuracy"]["float_full_metrics"]["pr_auc"],
        "pr_auc_drop": t3["pr_auc_drop"], "t3_decision_agreement": t3["decision_agreement"],
        "quantized_f1": metrics["accuracy"]["quantized_full_metrics"]["f1"], "float_f1": metrics["accuracy"]["float_full_metrics"]["f1"],
        "fhe_latency_mean_s": lat["mean_seconds"], "fhe_latency_std_s": lat["std_seconds"], "fhe_latency_min_s": lat["min_seconds"],
        "fhe_latency_max_s": lat["max_seconds"], "n_latency_trials": lat["n_trials"],
        "plaintext_latency_mean_s": metrics["plaintext_latency"]["mean_seconds"],
        "compile_s": metrics["compile_seconds"], "peak_rss_mb": metrics["peak_rss_mb"],
        "programmable_bootstraps": cs["programmable_bootstrap_count"], "n_trees": None,
        "ms_per_bootstrap": 1000.0 * lat["mean_seconds"] / cs["programmable_bootstrap_count"],
        "ciphertext_input_bytes": metrics["ciphertext_size_bytes"]["input"], "ciphertext_output_bytes": metrics["ciphertext_size_bytes"]["output"],
        "ciphertext_total_bytes": metrics["ciphertext_size_bytes"]["input"] + metrics["ciphertext_size_bytes"]["output"],
        "key_bootstrap_bytes": keys["bootstrap"], "key_keyswitch_bytes": keys["keyswitch"], "key_secret_bytes": keys["secret"],
        "key_total_bytes": keys["bootstrap"] + keys["keyswitch"] + keys["secret"],
        "max_integer_bit_width": cs["max_integer_bit_width"], "qat_best_epoch": metrics["training"]["qat"]["best_epoch"],
        "qat_budget_limited": not metrics["training"]["qat"]["stopped_early"],
        "entry_config_hash": data_provenance.compute_config_hash({"entry": entry, "grid_settings": {k: v for k, v in config.items() if k not in ("configurations", "output")}}),
    }


def build_mlp_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """(rows with an FHE latency, records of MLP configurations without one)."""
    config = r9.load_grid_config()
    results = r9.load_results()
    rows, without = [], []
    for entry in config["configurations"]:
        res = results.get(entry["label"])
        if res is None:
            without.append({"label": entry["label"], "status": "not_run"})
            continue
        m = res["metrics"]
        if m["status"] in r9.COMPLETED:
            rows.append(_row9(m, config, entry))
        else:
            gates = (m.get("accuracy") or {}).get("gates", {})
            without.append({
                "label": entry["label"], "status": m["status"], "n_bits": m.get("n_bits"), "n_features": m.get("n_features"),
                "reason": m.get("infeasible_reason") or (m.get("error") or {}).get("message"),
                "key_material_gb": m.get("key_material_gb"), "programmable_bootstraps": (m.get("circuit_stats") or {}).get("programmable_bootstrap_count"),
                "max_integer_bit_width": (m.get("circuit_stats") or {}).get("max_integer_bit_width"),
                "t3_passed": m.get("gates_passed", {}).get("t3"), "t3_decision_agreement": (gates.get("t3") or {}).get("decision_agreement"),
                "quantized_pr_auc": ((m.get("accuracy") or {}).get("quantized_full_metrics") or {}).get("pr_auc"),
                "float_pr_auc": ((m.get("accuracy") or {}).get("float_full_metrics") or {}).get("pr_auc"),
            })
    return rows, without


def combined_effects(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by = {(r["model"], r["n_features"], r["n_bits"]): r for r in rows}
    best_bits = {"lr": 16, "xgboost": 14}
    model_effect = []
    for n in (20, 50, 100):
        mlp_rows = [r for r in rows if r["model"] == "mlp" and r["n_features"] == n]
        if not mlp_rows or ("lr", n, 16) not in by or ("xgboost", n, 14) not in by:
            continue
        lr, xgb = by[("lr", n, 16)], by[("xgboost", n, 14)]
        for mlp in sorted(mlp_rows, key=lambda r: r["n_bits"]):
            model_effect.append({
                "n_features": n, "mlp_bits": mlp["n_bits"], "mlp_t3_passed": mlp["t3_passed"],
                "mlp_pr_auc": mlp["quantized_pr_auc"], "lr16_pr_auc": lr["quantized_pr_auc"], "xgboost14_pr_auc": xgb["quantized_pr_auc"],
                "mlp_latency_s": mlp["fhe_latency_mean_s"], "lr16_latency_s": lr["fhe_latency_mean_s"], "xgboost14_latency_s": xgb["fhe_latency_mean_s"],
                "mlp_latency_x_lr16": mlp["fhe_latency_mean_s"] / lr["fhe_latency_mean_s"], "xgboost14_latency_x_mlp": xgb["fhe_latency_mean_s"] / mlp["fhe_latency_mean_s"],
                "mlp_ciphertext_bytes": mlp["ciphertext_total_bytes"], "mlp_peak_rss_mb": mlp["peak_rss_mb"], "mlp_key_gb": mlp["key_total_bytes"] / 1e9,
                "mlp_bootstraps": mlp["programmable_bootstraps"],
            })
    mlp_bits = sorted({r["n_bits"] for r in rows if r["model"] == "mlp"})
    bit_effect = []
    if len(mlp_bits) >= 2:
        lo, hi = mlp_bits[0], mlp_bits[-1]
        for n in (20, 50, 100):
            if ("mlp", n, lo) in by and ("mlp", n, hi) in by:
                a, b = by[("mlp", n, lo)], by[("mlp", n, hi)]
                bit_effect.append({
                    "n_features": n, "low_bits": lo, "high_bits": hi, "pr_auc_low": a["quantized_pr_auc"], "pr_auc_high": b["quantized_pr_auc"],
                    "t3_low": a["t3_passed"], "t3_high": b["t3_passed"], "latency_x": b["fhe_latency_mean_s"] / a["fhe_latency_mean_s"],
                    "bootstraps_x": b["programmable_bootstraps"] / a["programmable_bootstraps"], "key_x": b["key_total_bytes"] / a["key_total_bytes"],
                    "peak_rss_x": b["peak_rss_mb"] / a["peak_rss_mb"],
                })
    prior = [{"label": r["label"], "latency_s": r["fhe_latency_mean_s"], "x_vs_296ms": r["fhe_latency_mean_s"] / PRIOR_ART_NEURAL_NETWORK_SECONDS} for r in rows if r["model"] == "mlp"]
    return {"model": model_effect, "mlp_bit_width": bit_effect, "prior_art_296ms": prior}


def analyze() -> dict[str, Any]:
    rows8 = p8.build_rows()
    rows9, without = build_mlp_rows()
    rows = rows8 + rows9
    passing = [r for r in rows if r["t3_passed"]]
    frontiers = {axis: {"all_configurations": p8.frontier(rows, key), "t3_passing_only": p8.frontier(passing, key)} for axis, (key, _n, _u) in p8.AXES.items()}
    return {"rows": rows, "mlp_without_latency": without, "frontiers": frontiers, "latency_dominance_within_noise": p8.noise_only_dominance(rows), "effects": combined_effects(rows)}


def _fmt(x: float) -> str:
    return p8._fmt_s(x)


def to_markdown(a: dict[str, Any]) -> str:
    rows = a["rows"]
    fr = a["frontiers"]["latency"]
    md = ["### Combined table (validation; latency = mean ± std of 5 repeated executions of one input)\n",
          "| Config | T3 | Quantized / float(-twin) PR-AUC | Agreement | FHE latency | Compile | Peak RSS | Bootstraps | Ciphertext in / out | Key material | Latency frontier |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        tag = "both" if r["label"] in fr["all_configurations"] and r["label"] in fr["t3_passing_only"] else "all-config only" if r["label"] in fr["all_configurations"] else "T3-passing only" if r["label"] in fr["t3_passing_only"] else "—"
        md.append(f"| `{r['label']}` | {'pass' if r['t3_passed'] else '**fail**'} | {r['quantized_pr_auc']:.4f} / {r['float_pr_auc']:.4f} | {r['t3_decision_agreement']:.4f} | {_fmt(r['fhe_latency_mean_s'])} ± {_fmt(r['fhe_latency_std_s'])} | {r['compile_s']:.1f} s | {r['peak_rss_mb']:,.0f} MB | {r['programmable_bootstraps']:,} | {r['ciphertext_input_bytes']:,} / {r['ciphertext_output_bytes']:,} B | {r['key_total_bytes'] / 1e9:.2f} GB | {tag} |")
    if a["mlp_without_latency"]:
        md.append("\n### MLP configurations without an FHE latency (recorded, not dropped)\n")
        md.append("| Config | Status | Key material | Bootstraps | T3 | Quantized / float-twin PR-AUC | Reason |\n|---|---|---|---|---|---|---|")
        for w in a["mlp_without_latency"]:
            q = f"{w['quantized_pr_auc']:.4f} / {w['float_pr_auc']:.4f}" if w.get("quantized_pr_auc") is not None else "—"
            t3 = "—" if w.get("t3_passed") is None else ("pass" if w["t3_passed"] else "**fail**")
            kg = f"{w['key_material_gb']:.2f} GB" if w.get("key_material_gb") else "—"
            md.append(f"| `{w['label']}` | {w['status']} | {kg} | {w.get('programmable_bootstraps') or '—'} | {t3} | {q} | {w.get('reason') or '—'} |")
    md.append("\n### Pareto frontiers over LR + XGBoost + MLP (accuracy = quantized PR-AUC ↑; cost ↓)\n")
    md.append("| Cost axis | Frontier over all configurations | Frontier over T3-passing configurations |\n|---|---|---|")
    for axis, (_k, name, unit) in p8.AXES.items():
        f = a["frontiers"][axis]
        md.append(f"| {name} ({unit}) | {' → '.join(f'`{x}`' for x in f['all_configurations'])} | {' → '.join(f'`{x}`' for x in f['t3_passing_only']) or '—'} |")
    if a["latency_dominance_within_noise"]:
        md.append("\nLatency dominances that hold on the means but not within one standard deviation: " + "; ".join(f"`{d['dominator']}` over `{d['dominated']}`" for d in a["latency_dominance_within_noise"]) + ".")
    e = a["effects"]
    if e["model"]:
        md.append("\n### Tree vs neural network under FHE (each MLP bit-width against LR 16-bit and XGBoost 14-bit at the same tier)\n")
        md.append("| Features | MLP bits | MLP T3 | MLP PR-AUC | LR-16 PR-AUC | XGB-14 PR-AUC | MLP latency | LR-16 latency | XGB-14 latency | MLP ÷ LR-16 | XGB-14 ÷ MLP | MLP ciphertext | MLP keys | MLP peak RSS |\n|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for r in e["model"]:
            md.append(f"| {r['n_features']} | {r['mlp_bits']} | {'pass' if r['mlp_t3_passed'] else '**fail**'} | {r['mlp_pr_auc']:.4f} | {r['lr16_pr_auc']:.4f} | {r['xgboost14_pr_auc']:.4f} | {_fmt(r['mlp_latency_s'])} | {_fmt(r['lr16_latency_s'])} | {_fmt(r['xgboost14_latency_s'])} | {r['mlp_latency_x_lr16']:,.0f}× | {r['xgboost14_latency_x_mlp']:,.0f}× | {r['mlp_ciphertext_bytes']:,} B | {r['mlp_key_gb']:.2f} GB | {r['mlp_peak_rss_mb']:,.0f} MB |")
    if e["mlp_bit_width"]:
        md.append("\n### MLP bit-width effect (high vs low)\n")
        md.append("| Features | Bits | PR-AUC low → high | T3 low → high | Latency × | Bootstraps × | Key material × | Peak RSS × |\n|---|---|---|---|---|---|---|---|")
        for r in e["mlp_bit_width"]:
            md.append(f"| {r['n_features']} | {r['low_bits']} → {r['high_bits']} | {r['pr_auc_low']:.4f} → {r['pr_auc_high']:.4f} | {'pass' if r['t3_low'] else 'fail'} → {'pass' if r['t3_high'] else 'fail'} | {r['latency_x']:.2f}× | {r['bootstraps_x']:.2f}× | {r['key_x']:.2f}× | {r['peak_rss_x']:.2f}× |")
    md.append("\n### Prior-art sanity check: MLP latency vs the ~296 ms neural network in `docs/prd.md` §11\n")
    md.append("The cited figure is compared only as a number; the cited work's network, bit-widths, hardware and definition of latency were not verified here.\n")
    md.append("| Config | Measured latency | ÷ 296 ms |\n|---|---|---|")
    for r in e["prior_art_296ms"]:
        md.append(f"| `{r['label']}` | {_fmt(r['latency_s'])} | {r['x_vs_296ms']:,.0f}× |")
    return "\n".join(md) + "\n"


def write_tables(a: dict[str, Any], out_dir: Path) -> None:
    rows = a["rows"]
    keys = sorted({k for r in rows for k in r})
    with (out_dir / "combined_table.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    (out_dir / "combined_table.json").write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
    payload = {"frontiers": a["frontiers"], "mlp_without_latency": a["mlp_without_latency"], "latency_dominance_within_noise": a["latency_dominance_within_noise"], "effects": a["effects"]}
    (out_dir / "combined_frontiers.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    (out_dir / "combined_tables.md").write_text(to_markdown(a), encoding="utf-8")


MODEL_COLOR = {**p8.MODEL_COLOR, "mlp": "#1baf7a"}  # categorical slots 1-3 of the reference palette (all-pairs safe)
LOW_BITS = {"lr": 8, "xgboost": 8, "mlp": 3}
NAME = {**p8.MODEL_NAME, "mlp": "Quantized MLP"}


def _scatter(ax: Any, rows: list[dict[str, Any]], cost_key: str, front_pass: list[str], front_all: list[str], xerr_key: str | None = None) -> None:
    for r in rows:
        color = MODEL_COLOR[r["model"]]
        if xerr_key:
            ax.errorbar(r[cost_key], r["quantized_pr_auc"], xerr=r[xerr_key], fmt="none", ecolor=color, elinewidth=1, capsize=2, alpha=0.55, zorder=2)
        ax.scatter(r[cost_key], r["quantized_pr_auc"], s=64, marker="o" if r["n_bits"] == LOW_BITS[r["model"]] else "s", facecolor=color if r["t3_passed"] else p8.SURFACE, edgecolor=color, linewidth=1.8, zorder=3)
        ax.annotate(f"{r['n_features']}", (r[cost_key], r["quantized_pr_auc"]), textcoords="offset points", xytext=(7, 5), fontsize=8, color=p8.INK_2)
    pts = sorted((r for r in rows if r["label"] in front_pass), key=lambda r: r[cost_key])
    if len(pts) > 1:
        xs, ys = [], []
        for i, r in enumerate(pts):
            if i:
                xs.append(r[cost_key]); ys.append(pts[i - 1]["quantized_pr_auc"])
            xs.append(r[cost_key]); ys.append(r["quantized_pr_auc"])
        ax.step(xs, ys, where="post", color=p8.INK, linewidth=1.2, linestyle=(0, (4, 3)), zorder=1)
    for r in rows:
        if r["label"] in front_all:
            ax.scatter(r[cost_key], r["quantized_pr_auc"], s=190, marker="o", facecolor="none", edgecolor=p8.INK, linewidth=0.9, zorder=2)


def _legend(ax: Any, loc: str, anchor: tuple[float, float] | None = None) -> None:
    from matplotlib.lines import Line2D

    handles = [Line2D([], [], marker="o", linestyle="", markerfacecolor=MODEL_COLOR[m], markeredgecolor=MODEL_COLOR[m], label=NAME[m]) for m in ("lr", "xgboost", "mlp")] + [
        Line2D([], [], marker="o", linestyle="", markerfacecolor=p8.INK_2, markeredgecolor=p8.INK_2, label="lower bit-width (circle)"),
        Line2D([], [], marker="s", linestyle="", markerfacecolor=p8.INK_2, markeredgecolor=p8.INK_2, label="higher bit-width (square)"),
        Line2D([], [], marker="o", linestyle="", markerfacecolor=p8.INK_2, markeredgecolor=p8.INK_2, label="filled: passes T3"),
        Line2D([], [], marker="o", linestyle="", markerfacecolor=p8.SURFACE, markeredgecolor=p8.INK_2, label="hollow: fails T3"),
        Line2D([], [], marker="o", linestyle="", markerfacecolor="none", markeredgecolor=p8.INK, markersize=11, label="ring: on the frontier (all)"),
        Line2D([], [], color=p8.INK, linestyle=(0, (4, 3)), label="staircase: frontier of T3-passing"),
    ]
    leg = ax.legend(handles=handles, loc=loc, bbox_to_anchor=anchor, fontsize=8, frameon=True, facecolor=p8.SURFACE, edgecolor=p8.GRID, framealpha=1.0)
    for t in leg.get_texts():
        t.set_color(p8.INK_2)


def render_figures(a: dict[str, Any], out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows, fr = a["rows"], a["frontiers"]
    fig, ax = plt.subplots(figsize=(9.8, 6.4), facecolor=p8.SURFACE)
    p8._style_axes(ax)
    _scatter(ax, rows, "fhe_latency_mean_s", fr["latency"]["t3_passing_only"], fr["latency"]["all_configurations"], xerr_key="fhe_latency_std_s")
    ax.set_xscale("log")
    ax.set_xlabel("FHE round-trip latency, mean ± 1 std of 5 repeated executions of one input (seconds, log scale)")
    ax.set_ylabel("Quantized PR-AUC (validation)")
    ax.set_title("Accuracy vs latency: LR, XGBoost and quantized MLP (labels = feature count)", loc="left", color=p8.INK, fontsize=12)
    _legend(ax, loc="lower left", anchor=(0.30, 0.02))
    fig.tight_layout()
    fig.savefig(out_dir / "fig_combined_accuracy_vs_latency.png", dpi=160, facecolor=p8.SURFACE)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), facecolor=p8.SURFACE, sharey=True)
    for ax, axis, label in ((axes[0], "memory", "Whole-process peak RSS (MB)"), (axes[1], "ciphertext", "Input + output ciphertext per request (bytes, log scale)")):
        p8._style_axes(ax)
        _scatter(ax, rows, p8.AXES[axis][0], fr[axis]["t3_passing_only"], fr[axis]["all_configurations"])
        ax.set_xlabel(label)
        if axis == "ciphertext":
            ax.set_xscale("log")
    axes[0].set_ylabel("Quantized PR-AUC (validation)")
    axes[0].set_title("Accuracy vs memory", loc="left", color=p8.INK, fontsize=12)
    axes[1].set_title("Accuracy vs ciphertext size", loc="left", color=p8.INK, fontsize=12)
    _legend(axes[0], loc="upper left")
    fig.tight_layout()
    fig.savefig(out_dir / "fig_combined_accuracy_vs_memory_ciphertext.png", dpi=160, facecolor=p8.SURFACE)
    plt.close(fig)


def main() -> None:
    for name, report in (("Phase 8", r8.validate_all()), ("Phase 9", r9.validate_all())):
        if report["problems"]:
            raise SystemExit(f"Refusing to analyse: {name} results failed validation:\n  " + "\n  ".join(report["problems"]))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    a = analyze()
    write_tables(a, OUT_DIR)
    render_figures(a, OUT_DIR)
    print(to_markdown(a))


if __name__ == "__main__":
    main()
