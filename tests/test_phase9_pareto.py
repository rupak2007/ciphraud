"""Tests for the combined LR + XGBoost + MLP Pareto analysis (`src/analysis/phase9_pareto.py`).

Pure-Python (no Concrete-ML): synthetic rows exercise the effect tables, the frontier over three model families, the
records of MLP configurations that have no latency, and the rendered markdown; the final group regenerates the committed
combined outputs from the real results and is skipped until they exist.
"""

import json

import pytest

from src.analysis import phase8_pareto as p8
from src.analysis import phase9_pareto as p9
from src.analysis import phase9_results as r9
from tests.test_phase9_results import valid_completed


def _row(model, n, bits, pr, lat, passed=True, rss=1000.0, ct=10_000, pbs=0, keys=1e9):
    return {
        "label": f"{model}_top{n}_bits{bits}", "model": model, "n_features": n, "n_bits": bits, "t3_passed": passed, "quantized_pr_auc": pr,
        "float_pr_auc": pr, "fhe_latency_mean_s": lat, "fhe_latency_std_s": lat * 0.05, "peak_rss_mb": rss, "ciphertext_total_bytes": ct,
        "programmable_bootstraps": pbs, "key_total_bytes": keys, "t3_decision_agreement": 0.99, "compile_s": 1.0, "ciphertext_input_bytes": 1, "ciphertext_output_bytes": ct - 1,
        "status": "passed" if passed else "failed_accuracy_gates",
    }


def _grid():
    rows = []
    for n, lr_pr, xgb_pr, mlp_pr in ((20, 0.35, 0.52, 0.40), (50, 0.39, 0.54, 0.44), (100, 0.42, 0.57, 0.47)):
        rows += [_row("lr", n, 16, lr_pr, 0.01 * n / 20), _row("xgboost", n, 14, xgb_pr, 3000.0, rss=3400.0, ct=4_000_000, pbs=300_000, keys=2e9)]
        rows += [_row("mlp", n, 3, mlp_pr - 0.05, 20.0 * n / 20, passed=False, rss=2000.0, ct=8_000, pbs=1000, keys=1e9),
                 _row("mlp", n, 4, mlp_pr, 70.0 * n / 20, passed=False, rss=3500.0, ct=9_000, pbs=2000, keys=2e9)]
    return rows


def test_model_effect_compares_each_mlp_width_with_lr16_and_xgboost14():
    effects = p9.combined_effects(_grid())
    assert len(effects["model"]) == 6  # 3 tiers x 2 MLP widths
    r = next(r for r in effects["model"] if r["n_features"] == 50 and r["mlp_bits"] == 4)
    assert r["lr16_pr_auc"] == 0.39 and r["xgboost14_pr_auc"] == 0.54 and r["mlp_pr_auc"] == 0.44
    assert r["mlp_latency_x_lr16"] == pytest.approx(70.0 * 2.5 / (0.01 * 2.5)) and r["xgboost14_latency_x_mlp"] == pytest.approx(3000.0 / (70.0 * 2.5))


def test_mlp_bit_width_effect_and_prior_art_ratio():
    effects = p9.combined_effects(_grid())
    assert [b["n_features"] for b in effects["mlp_bit_width"]] == [20, 50, 100]
    first = effects["mlp_bit_width"][0]
    assert first["low_bits"] == 3 and first["high_bits"] == 4 and first["latency_x"] == pytest.approx(70.0 / 20.0) and first["key_x"] == pytest.approx(2.0)
    prior = {p["label"]: p for p in effects["prior_art_296ms"]}
    assert prior["mlp_top20_bits4"]["x_vs_296ms"] == pytest.approx(70.0 / 0.296)
    assert len(prior) == 6 and all(k.startswith("mlp") for k in prior)  # only MLP rows are compared with the neural-network figure


def test_effects_are_empty_rather_than_wrong_when_a_family_is_missing():
    only_mlp = [r for r in _grid() if r["model"] == "mlp"]
    effects = p9.combined_effects(only_mlp)
    assert effects["model"] == [] and len(effects["mlp_bit_width"]) == 3  # no LR/XGB rows -> no cross-model comparison


def test_frontier_over_three_families_uses_phase8_definitions():
    rows = _grid()
    # Hand-derived: LR 16-bit (0.01-0.05 s) dominates every MLP 3-bit row and mlp_top20_bits4 (equal-or-better PR-AUC, far cheaper);
    # mlp_top50_bits4 (0.44 at 175 s) and mlp_top100_bits4 (0.47 at 350 s) beat every LR PR-AUC, so they are genuine trade-off points;
    # XGBoost's three tiers share one latency, so only the most accurate (top_100) survives.
    assert p8.frontier(rows, "fhe_latency_mean_s") == ["lr_top20_bits16", "lr_top50_bits16", "lr_top100_bits16", "mlp_top50_bits4", "mlp_top100_bits4", "xgboost_top100_bits14"]
    # the synthetic MLP rows all fail T3, so the T3-passing frontier is LR then the best XGBoost
    assert p8.frontier([r for r in rows if r["t3_passed"]], "fhe_latency_mean_s") == ["lr_top20_bits16", "lr_top50_bits16", "lr_top100_bits16", "xgboost_top100_bits14"]


def _metrics(status="passed"):
    m = valid_completed("mlp_top20_bits4")
    m["key_size_bytes"] = {"secret": 60_000, "bootstrap": 888_000_000, "keyswitch": 126_000_000}
    m["accuracy"]["quantized_full_metrics"].update({"f1": 0.4})
    m["accuracy"]["float_full_metrics"].update({"f1": 0.45})
    m["training"]["qat"].update({"best_epoch": 27, "stopped_early": False})
    m["circuit_stats"].update({"programmable_bootstrap_count": 640})
    m["status"] = status
    return m


def test_row9_maps_a_completed_record_to_the_common_row_schema():
    config = r9.load_grid_config()
    row = p9._row9(_metrics(), config, next(e for e in config["configurations"] if e["label"] == "mlp_top20_bits4"))
    assert row["model"] == "mlp" and row["n_features"] == 20 and row["n_bits"] == 4 and row["programmable_bootstraps"] == 640
    assert row["key_total_bytes"] == 60_000 + 888_000_000 + 126_000_000 and row["ciphertext_total_bytes"] == 480 + 3000
    assert row["ms_per_bootstrap"] == pytest.approx(1000 * row["fhe_latency_mean_s"] / 640) and row["qat_budget_limited"] is True
    # every column the Pareto/frontier code reads from a Phase 8 row must exist on an MLP row too
    assert set(row) >= {"quantized_pr_auc", "float_pr_auc", "fhe_latency_mean_s", "fhe_latency_std_s", "peak_rss_mb", "ciphertext_total_bytes", "key_bootstrap_bytes", "key_total_bytes", "t3_passed", "programmable_bootstraps", "label", "model"}


def test_configurations_without_a_latency_are_listed_not_dropped(monkeypatch):
    config = r9.load_grid_config()
    infeasible = valid_completed("mlp_top100_bits4")
    for k in ("fhe_latency", "plaintext_latency", "row_selection"):
        infeasible.pop(k)
    infeasible.update({"status": "infeasible_key_memory", "key_material_gb": 3.9, "infeasible_reason": "keys too large", "gates_passed": {"t0": True, "t3": False, "t2": None, "t1_correctness": None}, "all_gates_passed": False})
    failed = {"label": "mlp_top100_bits3", "status": "compile_failed", "n_bits": 3, "n_features": 100, "error": {"stage": "compile", "type": "NoParametersFound", "message": "no parameters", "traceback": "..."}}
    results = {"mlp_top20_bits4": {"metrics": _metrics(), "provenance": None}, "mlp_top100_bits4": {"metrics": infeasible, "provenance": None}, "mlp_top100_bits3": {"metrics": failed, "provenance": None}}
    monkeypatch.setattr(r9, "load_results", lambda *a, **k: results)
    rows, without = p9.build_mlp_rows()
    assert [r["label"] for r in rows] == ["mlp_top20_bits4"]
    by = {w["label"]: w for w in without}
    assert by["mlp_top100_bits4"]["status"] == "infeasible_key_memory" and by["mlp_top100_bits4"]["reason"] == "keys too large" and by["mlp_top100_bits4"]["key_material_gb"] == 3.9
    assert by["mlp_top100_bits3"]["status"] == "compile_failed" and by["mlp_top100_bits3"]["reason"] == "no parameters"
    assert by["mlp_top50_bits3"]["status"] == "not_run"  # configured but absent


def test_markdown_renders_every_section_including_without_latency_and_prior_art():
    rows = _grid()
    a = {
        "rows": rows, "mlp_without_latency": [{"label": "mlp_x", "status": "infeasible_key_memory", "reason": "r", "key_material_gb": 3.0, "programmable_bootstraps": 5, "t3_passed": False, "quantized_pr_auc": 0.3, "float_pr_auc": 0.4}],
        "frontiers": {axis: {"all_configurations": p8.frontier(rows, key), "t3_passing_only": p8.frontier([r for r in rows if r["t3_passed"]], key)} for axis, (key, _n, _u) in p8.AXES.items()},
        "latency_dominance_within_noise": [], "effects": p9.combined_effects(rows),
    }
    md = p9.to_markdown(a)
    for heading in ("Combined table", "without an FHE latency", "Pareto frontiers over LR + XGBoost + MLP", "Tree vs neural network", "MLP bit-width effect", "296 ms"):
        assert heading in md
    assert "`mlp_x`" in md and "infeasible_key_memory" in md


@pytest.mark.skipif(not (p9.OUT_DIR / "combined_table.json").exists(), reason="combined Phase 9 outputs not generated yet")
def test_committed_combined_outputs_regenerate_from_the_results():
    rows, _ = p9.build_mlp_rows()
    committed = json.loads((p9.OUT_DIR / "combined_table.json").read_text())
    assert p8.build_rows() + rows == committed
    payload = json.loads((p9.OUT_DIR / "combined_frontiers.json").read_text())
    for axis, (key, _n, _u) in p8.AXES.items():
        assert payload["frontiers"][axis]["all_configurations"] == p8.frontier(committed, key)
