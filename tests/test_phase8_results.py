"""Phase 8 tests required by docs/plan.md: results-schema validation, and spot-checks that the reported means
match the raw per-trial data -- plus tests of the Pareto-frontier logic.

The real-result tests read the 12 committed Phase 8 result files (small JSON). They run everywhere (no
Concrete-ML needed) and are skipped only if those files are absent. Negative controls corrupt an in-memory copy
of a real result and require the validators to notice, so a validator that accepts everything cannot pass.
"""

from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from src.analysis import phase8_pareto as pareto
from src.analysis import phase8_results as res
from src.config import PROJECT_ROOT

pytestmark = pytest.mark.skipif(not (res.RESULTS_DIR / "summary.json").exists(), reason="Phase 8 result files not present")


@pytest.fixture(scope="module")
def config():
    return res.load_grid_config()


@pytest.fixture(scope="module")
def results():
    return res.load_results()


def _entry(config, label):
    return next(e for e in config["configurations"] if e["label"] == label)


def test_all_twelve_configurations_have_results(config, results):
    assert [e["label"] for e in config["configurations"]] == list(results)
    assert len(results) == 12


def test_every_result_is_schema_valid(config, results):
    problems = []
    for label, r in results.items():
        problems += res.validate_result(r["metrics"], r["provenance"], config, _entry(config, label))
    assert problems == []


def test_summary_json_agrees_with_every_metrics_file(config, results):
    summary = json.loads((res.RESULTS_DIR / "summary.json").read_text(encoding="utf-8"))
    assert res.validate_summary(summary, results, config) == []


def test_reported_latency_statistics_equal_the_raw_trials(config, results):
    """Independent recomputation (not via the validator): mean, sample std (ddof=1), min and max of the raw trials."""
    for label, r in results.items():
        lat = r["metrics"]["fhe_latency"]
        for block in res.LATENCY_BLOCKS:
            raw = np.asarray(lat[block]["trials_seconds"], dtype=float)
            assert len(raw) == config["latency_trials"] == lat[block]["n_trials"], (label, block)
            assert lat[block]["mean_seconds"] == pytest.approx(raw.mean(), rel=1e-12), (label, block)
            assert lat[block]["std_seconds"] == pytest.approx(raw.std(ddof=1), rel=1e-12), (label, block)
            assert lat[block]["min_seconds"] == raw.min() and lat[block]["max_seconds"] == raw.max(), (label, block)


def test_no_result_touched_the_test_partition_and_all_use_one_row(results):
    assert all(r["metrics"]["test_partition_touched"] is False for r in results.values())
    assert {r["metrics"]["row_selection"]["position"] for r in results.values()} == {32148}


def test_grid_config_was_not_edited_after_the_runs(results):
    """Every provenance file recorded the same whole-grid hash, and it still matches the config on disk."""
    hashes = {r["provenance"]["config_hash"] for r in results.values()}
    assert len(hashes) == 1
    assert res.validate_all()["problems"] == []


def test_per_entry_config_hashes_are_unique_and_deterministic(config):
    h1 = {e["label"]: res.entry_config_hash(config, e) for e in config["configurations"]}
    h2 = {e["label"]: res.entry_config_hash(config, e) for e in config["configurations"]}
    assert h1 == h2 and len(set(h1.values())) == 12


def test_published_pareto_table_regenerates_from_the_results():
    committed = json.loads((pareto.OUT_DIR / "phase8_table.json").read_text(encoding="utf-8"))
    assert pareto.build_rows() == committed


# ---- negative controls: the validators must reject corrupted results -------------------------------------------


@pytest.mark.parametrize(
    "mutate, expect",
    [
        (lambda m: m["fhe_latency"]["run"].__setitem__("mean_seconds", m["fhe_latency"]["run"]["mean_seconds"] * 1.01), "mean_seconds"),
        (lambda m: m["fhe_latency"]["run"].__setitem__("std_seconds", 0.0), "std_seconds"),
        (lambda m: m["fhe_latency"]["run"]["trials_seconds"].pop(), "raw trials"),
        (lambda m: m["fhe_latency"]["total"].__setitem__("throughput_requests_per_second", 1.0), "throughput"),
        (lambda m: m.__setitem__("test_partition_touched", True), "test_partition_touched"),
        (lambda m: m.__setitem__("n_features", 21), "n_features"),
        (lambda m: m["gates_passed"].__setitem__("t3", not m["gates_passed"]["t3"]), "gates"),
        (lambda m: m["accuracy"]["gates"]["t3"].__setitem__("decision_agreement", 0.5), "T3"),
        (lambda m: m.pop("compile_seconds"), "missing keys"),
        (lambda m: m["fhe_latency"].__setitem__("outputs_reproducible", False), "outputs_reproducible"),
    ],
)
def test_validator_rejects_corrupted_result(config, results, mutate, expect):
    label = "xgboost_top50_bits14"
    r = results[label]
    bad = copy.deepcopy(r["metrics"])
    mutate(bad)
    problems = res.validate_result(bad, r["provenance"], config, _entry(config, label))
    assert any(expect in p for p in problems), problems


def test_validator_rejects_stale_provenance_hash(config, results):
    label = "lr_top20_bits8"
    prov = copy.deepcopy(results[label]["provenance"])
    prov["config_hash"] = "0" * 16
    assert any("config_hash" in p for p in res.validate_result(results[label]["metrics"], prov, config, _entry(config, label)))


def test_summary_validator_rejects_a_disagreeing_row(config, results):
    summary = json.loads((res.RESULTS_DIR / "summary.json").read_text(encoding="utf-8"))
    summary["configurations"][0]["fhe_mean_seconds"] *= 2
    assert any("fhe_mean_seconds" in p for p in res.validate_summary(summary, results, config))


# ---- Pareto logic on toy rows ----------------------------------------------------------------------------------


def _row(label, acc, cost, std=0.0, passed=True):
    return {"label": label, "quantized_pr_auc": acc, "fhe_latency_mean_s": cost, "fhe_latency_std_s": std, "t3_passed": passed}


def test_dominance_requires_a_strict_improvement():
    a, b = _row("a", 0.5, 1.0), _row("b", 0.5, 1.0)
    assert not pareto.dominates(a, b, "fhe_latency_mean_s") and not pareto.dominates(b, a, "fhe_latency_mean_s")
    assert pareto.dominates(_row("c", 0.6, 1.0), a, "fhe_latency_mean_s")
    assert pareto.dominates(_row("d", 0.5, 0.5), a, "fhe_latency_mean_s")
    assert not pareto.dominates(_row("e", 0.6, 2.0), a, "fhe_latency_mean_s")  # better but costlier: a trade-off, not dominance


def test_frontier_keeps_tradeoffs_and_drops_dominated_points():
    rows = [_row("cheap", 0.3, 1.0), _row("mid", 0.5, 2.0), _row("dominated", 0.4, 3.0), _row("best", 0.7, 5.0)]
    assert pareto.frontier(rows, "fhe_latency_mean_s") == ["cheap", "mid", "best"]


def test_frontier_of_a_single_point_is_that_point():
    assert pareto.frontier([_row("only", 0.1, 1.0)], "fhe_latency_mean_s") == ["only"]


def test_noise_only_dominance_is_flagged():
    a, b = _row("a", 0.6, 100.0, std=40.0), _row("b", 0.5, 110.0, std=40.0)  # a beats b on the means, but 1 std overlaps
    far_a, far_b = _row("fa", 0.6, 10.0, std=1.0), _row("fb", 0.5, 110.0, std=1.0)
    assert pareto.noise_only_dominance([a, b]) == [{"dominator": "a", "dominated": "b"}]
    assert pareto.noise_only_dominance([far_a, far_b]) == []


def test_committed_frontiers_are_consistent_with_the_rows():
    payload = json.loads((pareto.OUT_DIR / "pareto_frontiers.json").read_text(encoding="utf-8"))
    rows = pareto.build_rows()
    for axis, (key, _n, _u) in pareto.AXES.items():
        assert payload["frontiers"][axis]["all_configurations"] == pareto.frontier(rows, key)
        assert payload["frontiers"][axis]["t3_passing_only"] == pareto.frontier([r for r in rows if r["t3_passed"]], key)
        # no frontier member may be dominated by any configuration in its own population
        for r in rows:
            if r["label"] in payload["frontiers"][axis]["all_configurations"]:
                assert not any(pareto.dominates(o, r, key) for o in rows if o is not r)


def test_project_root_is_the_repo():
    assert (PROJECT_ROOT / "configs" / "phase8" / "research_grid.yaml").exists()
