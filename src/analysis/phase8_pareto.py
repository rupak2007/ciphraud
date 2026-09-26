"""Pareto-frontier and effect analysis over the 12 completed Phase 8 configurations.

Reads ONLY `results/phase8_research/*/metrics.json` (validated first by `phase8_results`); runs nothing,
touches no model, and never reads the test partition. Writes to `results/phase8_research/pareto/`:

    phase8_table.csv / .json     one row per configuration (every number traces to its metrics.json)
    pareto_frontiers.json        frontier membership for each cost axis, plus effect tables
    pareto_tables.md             the same tables, rendered for docs/research.md
    fig_accuracy_vs_latency.png, fig_accuracy_vs_memory_ciphertext.png, fig_feature_count_effects.png

    python -m src.analysis.phase8_pareto

Definitions (fixed, used for every figure and table)
  * accuracy  = PR-AUC of the *clear-quantized* model on the validation partition. T2 shows the compiled
                circuit reproduces it exactly, so it is the accuracy the encrypted model delivers. Validation
                only: the test partition has not been evaluated.
  * latency   = mean of 5 repeated executions of ONE fixed validation row (position 32,148), whole round trip
                (encrypt + run + decrypt), VM monotonic clock -- see docs/research.md Sec.7 for what that means.
  * memory    = whole-process peak RSS (load + compile + full-validation gates + trials), not inference-only.
  * ciphertext= input + output ciphertext bytes for one request.
  * A configuration A dominates B on an axis when A is at least as accurate and no costlier, and strictly better
    on at least one. The frontier is the set that nothing dominates. It is computed twice: over all 12
    configurations, and over only the configurations that pass every gate (the T3 bars are the project's own
    definition of "quantization preserved the model"). Failing configurations are always shown, never dropped.
  * Latency noise: a dominance that holds on the means but not when each mean is moved one standard deviation
    against the dominator is reported as "within noise".
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from src.analysis.phase8_results import RESULTS_DIR, entry_config_hash, load_grid_config, load_results, validate_all

OUT_DIR = RESULTS_DIR / "pareto"

# Cost axes: (key in the row, human name, unit label)
AXES = {
    "latency": ("fhe_latency_mean_s", "FHE round-trip latency (mean of 5 repeated executions)", "seconds"),
    "memory": ("peak_rss_mb", "whole-process peak RSS", "MB"),
    "ciphertext": ("ciphertext_total_bytes", "input + output ciphertext per request", "bytes"),
}


def build_rows() -> list[dict[str, Any]]:
    config = load_grid_config()
    results = load_results()
    rows = []
    for entry in config["configurations"]:
        m = results[entry["label"]]["metrics"]
        t3 = m["accuracy"]["gates"]["t3"]
        lat = m["fhe_latency"]["total"]
        keys = m["key_size_bytes"]
        rows.append({
            "label": entry["label"], "model": m["model_type"], "n_features": m["n_features"], "n_bits": m["n_bits"],
            "status": m["status"], "t3_passed": bool(m["gates_passed"]["t3"]), "all_gates_passed": bool(m["all_gates_passed"]),
            "quantized_pr_auc": m["accuracy"]["quantized_full_metrics"]["pr_auc"],
            "float_pr_auc": m["accuracy"]["float_full_metrics"]["pr_auc"],
            "pr_auc_drop": t3["pr_auc_drop"], "t3_decision_agreement": t3["decision_agreement"],
            "quantized_f1": m["accuracy"]["quantized_full_metrics"]["f1"], "float_f1": m["accuracy"]["float_full_metrics"]["f1"],
            "fhe_latency_mean_s": lat["mean_seconds"], "fhe_latency_std_s": lat["std_seconds"],
            "fhe_latency_min_s": lat["min_seconds"], "fhe_latency_max_s": lat["max_seconds"], "n_latency_trials": lat["n_trials"],
            "plaintext_latency_mean_s": m["plaintext_latency"]["mean_seconds"],
            "compile_s": m["compile_seconds"], "peak_rss_mb": m["peak_rss_mb"],
            "programmable_bootstraps": m["circuit_stats"]["programmable_bootstrap_count"], "n_trees": m["circuit_stats"].get("n_trees"),
            "ms_per_bootstrap": (1000.0 * lat["mean_seconds"] / m["circuit_stats"]["programmable_bootstrap_count"]) if m["circuit_stats"]["programmable_bootstrap_count"] else None,
            "ciphertext_input_bytes": m["ciphertext_size_bytes"]["input"], "ciphertext_output_bytes": m["ciphertext_size_bytes"]["output"],
            "ciphertext_total_bytes": m["ciphertext_size_bytes"]["input"] + m["ciphertext_size_bytes"]["output"],
            "key_bootstrap_bytes": keys["bootstrap"], "key_keyswitch_bytes": keys["keyswitch"], "key_secret_bytes": keys["secret"],
            "key_total_bytes": keys["bootstrap"] + keys["keyswitch"] + keys["secret"],
            "entry_config_hash": entry_config_hash(config, entry),
        })
    return rows


def dominates(a: dict[str, Any], b: dict[str, Any], cost_key: str) -> bool:
    ge_acc, le_cost = a["quantized_pr_auc"] >= b["quantized_pr_auc"], a[cost_key] <= b[cost_key]
    strict = a["quantized_pr_auc"] > b["quantized_pr_auc"] or a[cost_key] < b[cost_key]
    return ge_acc and le_cost and strict


def frontier(rows: list[dict[str, Any]], cost_key: str) -> list[str]:
    """Labels not dominated by any other row in `rows`, ordered by increasing cost."""
    keep = [r for r in rows if not any(dominates(o, r, cost_key) for o in rows if o is not r)]
    return [r["label"] for r in sorted(keep, key=lambda r: (r[cost_key], -r["quantized_pr_auc"]))]


def noise_only_dominance(rows: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Latency dominances that hold on the means but disappear when the dominator's mean is raised by 1 std
    and the dominated one's lowered by 1 std."""
    out = []
    for a in rows:
        for b in rows:
            if a is b or not dominates(a, b, "fhe_latency_mean_s"):
                continue
            robust = a["quantized_pr_auc"] >= b["quantized_pr_auc"] and a["fhe_latency_mean_s"] + a["fhe_latency_std_s"] < b["fhe_latency_mean_s"] - b["fhe_latency_std_s"]
            if not robust:
                out.append({"dominator": a["label"], "dominated": b["label"]})
    return out


def effect_tables(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by = {(r["model"], r["n_features"], r["n_bits"]): r for r in rows}
    bits = {"lr": (8, 16), "xgboost": (8, 14)}
    feature_effect, bit_effect, model_effect = [], [], []
    for model, (lo, hi) in bits.items():
        for n_bits in (lo, hi):
            base = by[(model, 20, n_bits)]
            for n in (20, 50, 100):
                r = by[(model, n, n_bits)]
                feature_effect.append({
                    "model": model, "n_bits": n_bits, "n_features": n, "quantized_pr_auc": r["quantized_pr_auc"], "t3_passed": r["t3_passed"],
                    "latency_mean_s": r["fhe_latency_mean_s"], "latency_x_vs_top20": r["fhe_latency_mean_s"] / base["fhe_latency_mean_s"],
                    "peak_rss_mb": r["peak_rss_mb"], "ciphertext_total_bytes": r["ciphertext_total_bytes"],
                    "programmable_bootstraps": r["programmable_bootstraps"], "n_trees": r["n_trees"],
                })
        for n in (20, 50, 100):
            a, b = by[(model, n, lo)], by[(model, n, hi)]
            bit_effect.append({
                "model": model, "n_features": n, "low_bits": lo, "high_bits": hi,
                "quantized_pr_auc_low": a["quantized_pr_auc"], "quantized_pr_auc_high": b["quantized_pr_auc"],
                "pr_auc_gain": b["quantized_pr_auc"] - a["quantized_pr_auc"], "t3_low": a["t3_passed"], "t3_high": b["t3_passed"],
                "latency_x": b["fhe_latency_mean_s"] / a["fhe_latency_mean_s"],
                "bootstraps_x": (b["programmable_bootstraps"] / a["programmable_bootstraps"]) if a["programmable_bootstraps"] else None,
                "bootstrap_key_x": (b["key_bootstrap_bytes"] / a["key_bootstrap_bytes"]) if a["key_bootstrap_bytes"] else None,
                "ciphertext_total_x": b["ciphertext_total_bytes"] / a["ciphertext_total_bytes"],
                "peak_rss_x": b["peak_rss_mb"] / a["peak_rss_mb"],
            })
    for n in (20, 50, 100):
        lr, xgb = by[("lr", n, 16)], by[("xgboost", n, 14)]
        model_effect.append({
            "n_features": n, "lr_bits": 16, "xgboost_bits": 14, "lr_pr_auc": lr["quantized_pr_auc"], "xgboost_pr_auc": xgb["quantized_pr_auc"],
            "pr_auc_gain_xgboost": xgb["quantized_pr_auc"] - lr["quantized_pr_auc"],
            "lr_latency_s": lr["fhe_latency_mean_s"], "xgboost_latency_s": xgb["fhe_latency_mean_s"],
            "latency_x_xgboost_over_lr": xgb["fhe_latency_mean_s"] / lr["fhe_latency_mean_s"],
            "lr_ciphertext_total_bytes": lr["ciphertext_total_bytes"], "xgboost_ciphertext_total_bytes": xgb["ciphertext_total_bytes"],
            "lr_peak_rss_mb": lr["peak_rss_mb"], "xgboost_peak_rss_mb": xgb["peak_rss_mb"],
        })
    return {"feature_count": feature_effect, "bit_width": bit_effect, "model": model_effect}


def analyze() -> dict[str, Any]:
    rows = build_rows()
    passing = [r for r in rows if r["t3_passed"]]
    frontiers = {}
    for axis, (key, _name, _unit) in AXES.items():
        frontiers[axis] = {"all_configurations": frontier(rows, key), "t3_passing_only": frontier(passing, key)}
    return {"rows": rows, "frontiers": frontiers, "latency_dominance_within_noise": noise_only_dominance(rows), "effects": effect_tables(rows)}


# ---------------------------------------------------------------------------------------------- rendering


def _fmt_s(x: float) -> str:
    if x < 1:
        return f"{x * 1000:.2f} ms"
    return f"{x:,.1f} s"


def to_markdown(a: dict[str, Any]) -> str:
    rows = a["rows"]
    front_all, front_pass = a["frontiers"]["latency"]["all_configurations"], a["frontiers"]["latency"]["t3_passing_only"]
    md = ["### Full table (validation; latency = mean ± std of 5 repeated executions of one input)\n",
          "| Config | T3 | Quantized / float PR-AUC | Agreement | FHE latency | Compile | Peak RSS | Bootstraps | Ciphertext in / out | Bootstrap key | Latency frontier |",
          "|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        tag = ("both" if r["label"] in front_all and r["label"] in front_pass else "all-config only" if r["label"] in front_all else "T3-passing only" if r["label"] in front_pass else "—")
        md.append(
            f"| `{r['label']}` | {'pass' if r['t3_passed'] else '**fail**'} | {r['quantized_pr_auc']:.4f} / {r['float_pr_auc']:.4f} | {r['t3_decision_agreement']:.4f} | "
            f"{_fmt_s(r['fhe_latency_mean_s'])} ± {_fmt_s(r['fhe_latency_std_s'])} | {r['compile_s']:.1f} s | {r['peak_rss_mb']:,.0f} MB | "
            f"{r['programmable_bootstraps']:,} | {r['ciphertext_input_bytes']:,} / {r['ciphertext_output_bytes']:,} B | {r['key_bootstrap_bytes'] / 1e6:,.0f} MB | {tag} |")
    md.append("\n### Pareto frontiers (accuracy = quantized PR-AUC, higher is better; cost lower is better)\n")
    md.append("| Cost axis | Frontier over all 12 configurations | Frontier over T3-passing configurations |\n|---|---|---|")
    for axis, (_k, name, unit) in AXES.items():
        f = a["frontiers"][axis]
        md.append(f"| {name} ({unit}) | {' → '.join(f'`{x}`' for x in f['all_configurations'])} | {' → '.join(f'`{x}`' for x in f['t3_passing_only'])} |")
    if a["latency_dominance_within_noise"]:
        md.append("\nLatency dominances that hold on the means but not within one standard deviation: " + "; ".join(f"`{d['dominator']}` over `{d['dominated']}`" for d in a["latency_dominance_within_noise"]) + ".")
    else:
        md.append("\nEvery latency dominance holds even with each mean moved one standard deviation against the dominator.")
    e = a["effects"]
    md.append("\n### Feature-count effect (ratio vs `top_20` at the same model and bit-width)\n")
    md.append("XGBoost cost follows the number of trees the tier's model was trained with (early stopping chose 358 / 219 / 271 for `top_20` / `top_50` / `top_100`), not the feature count directly, so XGBoost cost is not monotonic in features.\n")
    md.append("| Model, bits | Features | Trees | Quantized PR-AUC | T3 | Latency | × vs top_20 | Peak RSS | Ciphertext total | Bootstraps |\n|---|---|---|---|---|---|---|---|---|---|")
    for r in e["feature_count"]:
        trees = f"{r['n_trees']}" if r["n_trees"] else "n/a"
        md.append(f"| {r['model']}, {r['n_bits']} | {r['n_features']} | {trees} | {r['quantized_pr_auc']:.4f} | {'pass' if r['t3_passed'] else '**fail**'} | {_fmt_s(r['latency_mean_s'])} | {r['latency_x_vs_top20']:.2f}× | {r['peak_rss_mb']:,.0f} MB | {r['ciphertext_total_bytes']:,} B | {r['programmable_bootstraps']:,} |")
    md.append("\n### Bit-width effect (high vs low, same model and tier)\n")
    md.append("| Model | Features | Bits low → high | Quantized PR-AUC low → high | PR-AUC gain | T3 low → high | Latency × | Bootstraps × | Bootstrap key × | Peak RSS × |\n|---|---|---|---|---|---|---|---|---|---|")
    for r in e["bit_width"]:
        bx = f"{r['bootstraps_x']:.2f}×" if r["bootstraps_x"] else "n/a (0)"
        kx = f"{r['bootstrap_key_x']:.2f}×" if r["bootstrap_key_x"] else "n/a (0)"
        md.append(f"| {r['model']} | {r['n_features']} | {r['low_bits']} → {r['high_bits']} | {r['quantized_pr_auc_low']:.4f} → {r['quantized_pr_auc_high']:.4f} | {r['pr_auc_gain']:+.4f} | {'pass' if r['t3_low'] else 'fail'} → {'pass' if r['t3_high'] else 'fail'} | {r['latency_x']:.2f}× | {bx} | {kx} | {r['peak_rss_x']:.2f}× |")
    md.append("\n### Prior-art sanity check: measured XGBoost latency vs the ~6 ms in `docs/prd.md` §11\n")
    md.append("Measured cost per programmable bootstrap (mean round-trip latency ÷ bootstrap count; this machine, whole 6-core CPU in use). "
              "The cited figure is compared only as a number: the cited work's model size, tree count, bit-width, hardware, threading and whether the figure is per request were not verified here.\n")
    md.append("| Config | Trees | Bootstraps | Latency | Latency per bootstrap | Latency ÷ 6 ms |\n|---|---|---|---|---|---|")
    for r in rows:
        if r["model"] == "xgboost":
            md.append(f"| `{r['label']}` | {r['n_trees']} | {r['programmable_bootstraps']:,} | {_fmt_s(r['fhe_latency_mean_s'])} | {r['ms_per_bootstrap']:.2f} ms | {r['fhe_latency_mean_s'] / 0.006:,.0f}× |")
    md.append("\n### Model effect (each model at its highest feasible bit-width: LR 16, XGBoost 14)\n")
    md.append("| Features | LR PR-AUC | XGBoost PR-AUC | XGBoost − LR | LR latency | XGBoost latency | XGBoost ÷ LR latency | LR / XGBoost ciphertext | LR / XGBoost peak RSS |\n|---|---|---|---|---|---|---|---|---|")
    for r in e["model"]:
        md.append(f"| {r['n_features']} | {r['lr_pr_auc']:.4f} | {r['xgboost_pr_auc']:.4f} | {r['pr_auc_gain_xgboost']:+.4f} | {_fmt_s(r['lr_latency_s'])} | {_fmt_s(r['xgboost_latency_s'])} | {r['latency_x_xgboost_over_lr']:,.0f}× | {r['lr_ciphertext_total_bytes']:,} / {r['xgboost_ciphertext_total_bytes']:,} B | {r['lr_peak_rss_mb']:,.0f} / {r['xgboost_peak_rss_mb']:,.0f} MB |")
    return "\n".join(md) + "\n"


def write_tables(a: dict[str, Any], out_dir: Path) -> None:
    rows = a["rows"]
    with (out_dir / "phase8_table.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    (out_dir / "phase8_table.json").write_text(json.dumps(rows, indent=2, sort_keys=True), encoding="utf-8")
    payload = {"frontiers": a["frontiers"], "latency_dominance_within_noise": a["latency_dominance_within_noise"], "effects": a["effects"]}
    (out_dir / "pareto_frontiers.json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    (out_dir / "pareto_tables.md").write_text(to_markdown(a), encoding="utf-8")


# Palette: categorical slots 1-2 of the reference palette (blue, orange); text tokens; light surface.
SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
MODEL_COLOR = {"lr": "#2a78d6", "xgboost": "#eb6834"}
MODEL_NAME = {"lr": "Logistic Regression", "xgboost": "XGBoost"}


def _marker(r: dict[str, Any]) -> str:
    return "o" if r["n_bits"] in (8,) else "s"  # circle = 8 bits, square = the model's high bit-width (LR 16, XGBoost 14)


def _style_axes(ax: Any) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_2)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.xaxis.label.set_color(INK)
    ax.yaxis.label.set_color(INK)


def _scatter(ax: Any, rows: list[dict[str, Any]], cost_key: str, front_pass: list[str], front_all: list[str], xerr_key: str | None = None) -> None:
    for r in rows:
        color = MODEL_COLOR[r["model"]]
        filled = r["t3_passed"]
        if xerr_key:
            ax.errorbar(r[cost_key], r["quantized_pr_auc"], xerr=r[xerr_key], fmt="none", ecolor=color, elinewidth=1, capsize=2, alpha=0.55, zorder=2)
        ax.scatter(r[cost_key], r["quantized_pr_auc"], s=64, marker=_marker(r), facecolor=color if filled else SURFACE, edgecolor=color, linewidth=1.8, zorder=3)
        ax.annotate(f"{r['n_features']}", (r[cost_key], r["quantized_pr_auc"]), textcoords="offset points", xytext=(7, 5), fontsize=8, color=INK_2)
    pts = sorted((r for r in rows if r["label"] in front_pass), key=lambda r: r[cost_key])
    if len(pts) > 1:
        # staircase through the frontier: after each point accuracy is held flat until the next, better point
        xs, ys = [], []
        for i, r in enumerate(pts):
            if i:
                xs.append(r[cost_key]); ys.append(pts[i - 1]["quantized_pr_auc"])
            xs.append(r[cost_key]); ys.append(r["quantized_pr_auc"])
        ax.step(xs, ys, where="post", color=INK, linewidth=1.2, linestyle=(0, (4, 3)), zorder=1)
    for r in rows:
        if r["label"] in front_all:
            ax.scatter(r[cost_key], r["quantized_pr_auc"], s=190, marker="o", facecolor="none", edgecolor=INK, linewidth=0.9, zorder=2)


def _legend(ax: Any, loc: str = "lower right", anchor: tuple[float, float] | None = None) -> None:
    from matplotlib.lines import Line2D
    handles = [
        Line2D([], [], marker="o", linestyle="", markerfacecolor=MODEL_COLOR["lr"], markeredgecolor=MODEL_COLOR["lr"], label="Logistic Regression"),
        Line2D([], [], marker="o", linestyle="", markerfacecolor=MODEL_COLOR["xgboost"], markeredgecolor=MODEL_COLOR["xgboost"], label="XGBoost"),
        Line2D([], [], marker="o", linestyle="", markerfacecolor=INK_2, markeredgecolor=INK_2, label="8 bits (circle)"),
        Line2D([], [], marker="s", linestyle="", markerfacecolor=INK_2, markeredgecolor=INK_2, label="LR 16 / XGBoost 14 bits (square)"),
        Line2D([], [], marker="o", linestyle="", markerfacecolor=INK_2, markeredgecolor=INK_2, label="filled: passes T3"),
        Line2D([], [], marker="o", linestyle="", markerfacecolor=SURFACE, markeredgecolor=INK_2, label="hollow: fails T3"),
        Line2D([], [], marker="o", linestyle="", markerfacecolor="none", markeredgecolor=INK, markersize=11, label="ring: on the frontier (all 12)"),
        Line2D([], [], color=INK, linestyle=(0, (4, 3)), label="staircase: frontier of T3-passing"),
    ]
    leg = ax.legend(handles=handles, loc=loc, bbox_to_anchor=anchor, fontsize=8, frameon=True, facecolor=SURFACE, edgecolor=GRID, framealpha=1.0)
    for t in leg.get_texts():
        t.set_color(INK_2)


def render_figures(a: dict[str, Any], out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = a["rows"]
    fr = a["frontiers"]

    fig, ax = plt.subplots(figsize=(9.5, 6.2), facecolor=SURFACE)
    _style_axes(ax)
    _scatter(ax, rows, "fhe_latency_mean_s", fr["latency"]["t3_passing_only"], fr["latency"]["all_configurations"], xerr_key="fhe_latency_std_s")
    ax.set_xscale("log")
    ax.set_xlabel("FHE round-trip latency, mean ± 1 std of 5 repeated executions of one input (seconds, log scale)")
    ax.set_ylabel("Quantized PR-AUC (validation)")
    ax.set_title("Accuracy vs latency, all 12 Phase 8 configurations (labels = feature count)", loc="left", color=INK, fontsize=12)
    _legend(ax, loc="lower left", anchor=(0.30, 0.02))  # the empty middle of the plot, clear of both model clusters
    fig.tight_layout()
    fig.savefig(out_dir / "fig_accuracy_vs_latency.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), facecolor=SURFACE, sharey=True)
    for ax, axis, label in ((axes[0], "memory", "Whole-process peak RSS (MB)"), (axes[1], "ciphertext", "Input + output ciphertext per request (bytes, log scale)")):
        _style_axes(ax)
        key = AXES[axis][0]
        _scatter(ax, rows, key, fr[axis]["t3_passing_only"], fr[axis]["all_configurations"])
        ax.set_xlabel(label)
        if axis == "ciphertext":
            ax.set_xscale("log")
    axes[0].set_ylabel("Quantized PR-AUC (validation)")
    axes[0].set_title("Accuracy vs memory", loc="left", color=INK, fontsize=12)
    axes[1].set_title("Accuracy vs ciphertext size", loc="left", color=INK, fontsize=12)
    _legend(axes[0], loc="upper left")  # empty upper-left of the memory panel
    fig.tight_layout()
    fig.savefig(out_dir / "fig_accuracy_vs_memory_ciphertext.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4), facecolor=SURFACE)
    series = [("lr", 8), ("lr", 16), ("xgboost", 8), ("xgboost", 14)]
    dash = {8: (0, (2, 2)), 16: "-", 14: "-"}
    for model, n_bits in series:
        pts = sorted((r for r in rows if r["model"] == model and r["n_bits"] == n_bits), key=lambda r: r["n_features"])
        xs = [r["n_features"] for r in pts]
        for ax, key in ((axes[0], "quantized_pr_auc"), (axes[1], "fhe_latency_mean_s")):
            ax.plot(xs, [r[key] for r in pts], color=MODEL_COLOR[model], linestyle=dash[n_bits], linewidth=2, marker="o" if n_bits == 8 else "s", markersize=7,
                    markerfacecolor=MODEL_COLOR[model], label=f"{MODEL_NAME[model]}, {n_bits} bits")
    for ax in axes:
        _style_axes(ax)
        ax.set_xticks([20, 50, 100])
        ax.set_xlabel("Feature tier (number of features)")
    axes[0].set_ylabel("Quantized PR-AUC (validation)")
    axes[0].set_title("Accuracy vs feature count", loc="left", color=INK, fontsize=12)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("FHE round-trip latency (seconds, log scale)")
    axes[1].set_title("Latency vs feature count", loc="left", color=INK, fontsize=12)
    leg = axes[0].legend(fontsize=8.5, frameon=True, facecolor=SURFACE, edgecolor=GRID, framealpha=1.0, loc="center", bbox_to_anchor=(0.5, 0.64))
    for t in leg.get_texts():
        t.set_color(INK_2)
    fig.tight_layout()
    fig.savefig(out_dir / "fig_feature_count_effects.png", dpi=160, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    report = validate_all()
    if report["problems"]:
        raise SystemExit("Refusing to analyse: Phase 8 results failed validation:\n  " + "\n  ".join(report["problems"]))
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    a = analyze()
    write_tables(a, OUT_DIR)
    render_figures(a, OUT_DIR)
    print(to_markdown(a))


if __name__ == "__main__":
    main()
