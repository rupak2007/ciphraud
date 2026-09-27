"""Phase 9 result-file validation tests: schema, means-vs-raw, the three record states, and negative controls.

Runs in every environment (no Concrete-ML). Valid records are built by hand from the schema `src/benchmark/phase9_grid.py`
writes, so the validator is tested even before any real result exists; a final group validates the real committed results
and is skipped until they are present.
"""

import copy

import numpy as np
import pytest

from src.analysis import phase9_results as res
from src.benchmark.harness import trial_stats
from src.data import provenance as data_provenance

CONFIG = res.load_grid_config()


def _entry(label="mlp_top20_bits4"):
    return next(e for e in CONFIG["configurations"] if e["label"] == label)


def _block(base, n=5, spread=0.01):
    return trial_stats([base * (1 + spread * i) for i in range(n)]) | {"throughput_requests_per_second": 1.0 / trial_stats([base * (1 + spread * i) for i in range(n)])["mean_seconds"]}


def _fhe_latency(n=5):
    enc, run, dec = [0.01 * (1 + 0.01 * i) for i in range(n)], [30.0 * (1 + 0.02 * i) for i in range(n)], [0.005 * (1 + 0.01 * i) for i in range(n)]
    total = [a + b + c + 0.001 for a, b, c in zip(enc, run, dec)]
    return {"encrypt": trial_stats(enc), "run": trial_stats(run), "decrypt": trial_stats(dec), "total": trial_stats(total) | {"throughput_requests_per_second": 1.0 / trial_stats(total)["mean_seconds"]},
            "keygen_seconds": 10.0, "outputs_reproducible": True, "throughput_requests_per_second": 0.03}


def valid_completed(label="mlp_top20_bits4"):
    tier_n, bits = int(label.split("top")[1].split("_")[0]), int(label.split("bits")[1])
    t3 = {"decision_agreement": 0.995, "pr_auc_drop": 0.004, "quantized_pr_auc": 0.40, "float_pr_auc": 0.404, "passed": True}
    return {
        "label": label, "model_type": "mlp", "tier": f"top_{tier_n}", "n_bits": bits, "n_features": tier_n, "seed": CONFIG["seed"],
        "status": "passed", "test_partition_touched": False, "all_gates_passed": True,
        "gates_passed": {"t0": True, "t3": True, "t2": True, "t1_correctness": True},
        "accuracy": {
            "gates": {"t0": {"reload_reproduces_clear_quantized_logits_exactly": True, "max_abs_diff": 0.0, "passed": True}, "t3": t3,
                      "t2": {"exact_integer_match": True, "passed": True}, "t1_correctness": {"exact_integer_match": True, "passed": True}},
            "quantized_full_metrics": {"pr_auc": 0.40}, "float_full_metrics": {"pr_auc": 0.404},
        },
        "circuit_stats": {"programmable_bootstrap_count": 640, "max_integer_bit_width": 10, "n_w_bits": bits, "n_a_bits": bits, "n_accum_bits": CONFIG["mlp"]["n_accum_bits"]},
        "compile_seconds": 9.5, "peak_rss_mb": 2500.0, "ciphertext_size_bytes": {"input": 480, "output": 3000}, "key_size_bytes": {"bootstrap": 888_000_000},
        "fhe_latency": _fhe_latency(), "plaintext_latency": _block(0.001) | {"outputs_reproducible": True},
        "row_selection": {"position": 32148, "y_true": 0, "n_positive": CONFIG["row_selection"]["n_positive"], "seed": CONFIG["seed"]},
        "training": {"qat": {"epochs_run": 8}, "float_twin": {"epochs_run": 9}},
    }


def provenance():
    return {"config_hash": data_provenance.compute_config_hash(CONFIG), "config_path": res.GRID_CONFIG, "git_commit": "a" * 40, "library_versions": {}, "seed": 42, "timestamp_utc": "2026-09-27T00:00:00+00:00"}


def problems(metrics, prov="default"):
    return res.validate_result(metrics, provenance() if prov == "default" else prov, CONFIG, _entry(metrics["label"]))


def test_a_well_formed_completed_record_validates():
    assert problems(valid_completed()) == []


def test_a_completed_record_that_failed_t3_is_still_valid_when_consistent():
    m = valid_completed()
    m["accuracy"]["gates"]["t3"].update({"decision_agreement": 0.9, "passed": False})
    m["gates_passed"]["t3"], m["all_gates_passed"], m["status"] = False, False, "failed_accuracy_gates"
    assert problems(m) == []


@pytest.mark.parametrize(
    "mutate, expect",
    [
        (lambda m: m["fhe_latency"]["run"].__setitem__("mean_seconds", 1.0), "mean_seconds"),
        (lambda m: m["fhe_latency"]["total"]["trials_seconds"].pop(), "raw trials"),
        (lambda m: m["accuracy"]["gates"]["t3"].__setitem__("decision_agreement", 0.5), "T3"),
        (lambda m: m["accuracy"]["gates"]["t3"].__setitem__("float_pr_auc", 0.9), "pr_auc_drop"),
        (lambda m: m["accuracy"]["gates"]["t0"].__setitem__("max_abs_diff", 1e-6), "T0"),
        (lambda m: m["accuracy"]["gates"]["t2"].__setitem__("exact_integer_match", False), "exact_integer_match"),
        (lambda m: m["circuit_stats"].__setitem__("programmable_bootstrap_count", 0), "table lookups"),
        (lambda m: m["circuit_stats"].__setitem__("n_w_bits", 5), "bit-widths"),
        (lambda m: m["circuit_stats"].__setitem__("max_integer_bit_width", 17), "max_integer_bit_width"),
        (lambda m: m.__setitem__("test_partition_touched", True), "test_partition_touched"),
        (lambda m: m.__setitem__("n_features", 21), "identity"),
        (lambda m: m["gates_passed"].__setitem__("t3", False), "disagrees"),
        (lambda m: m["fhe_latency"].__setitem__("outputs_reproducible", False), "outputs_reproducible"),
        (lambda m: m["row_selection"].__setitem__("seed", 1), "row_selection"),
        (lambda m: m["training"]["qat"].__setitem__("epochs_run", 0), "training record"),
        (lambda m: m.pop("seed"), "missing keys"),
        (lambda m: m.__setitem__("status", "weird"), "unknown status"),
    ],
)
def test_validator_rejects_corrupted_completed_record(mutate, expect):
    bad = copy.deepcopy(valid_completed())
    mutate(bad)
    assert any(expect in p for p in problems(bad)), problems(bad)


def test_stale_provenance_hash_and_bad_commit_are_rejected():
    stale = provenance() | {"config_hash": "0" * 16}
    assert any("config_hash" in p for p in problems(valid_completed(), stale))
    assert any("git_commit" in p for p in problems(valid_completed(), provenance() | {"git_commit": "e85d16a"}))
    assert any("provenance.json missing" in p for p in problems(valid_completed(), None))


def _infeasible():
    m = valid_completed("mlp_top100_bits4")
    for k in ("fhe_latency", "plaintext_latency", "row_selection"):
        m.pop(k)
    m.update({"status": "infeasible_key_memory", "key_material_gb": 3.9, "infeasible_reason": "key material 3.90 GB exceeds the 2.4 GB limit",
              "gates_passed": {"t0": True, "t3": True, "t2": None, "t1_correctness": None}, "all_gates_passed": False})
    return m


def test_infeasible_configuration_is_a_valid_recorded_state():
    assert problems(_infeasible()) == []


@pytest.mark.parametrize(
    "mutate, expect",
    [
        (lambda m: m.__setitem__("key_material_gb", 1.0), "requires key_material_gb"),
        (lambda m: m.__setitem__("fhe_latency", _fhe_latency()), "must not carry FHE latency"),
        (lambda m: m["gates_passed"].__setitem__("t2", True), "null T2/T1"),
        (lambda m: m["circuit_stats"].__setitem__("programmable_bootstrap_count", 0), "circuit statistics"),
    ],
)
def test_validator_rejects_malformed_infeasible_record(mutate, expect):
    bad = _infeasible()
    mutate(bad)
    assert any(expect in p for p in problems(bad)), problems(bad)


def test_failed_configuration_must_carry_its_error():
    m = {"label": "mlp_top20_bits4", "model_type": "mlp", "tier": "top_20", "n_bits": 4, "n_features": 20, "seed": 42, "test_partition_touched": False, "status": "compile_failed"}
    assert any("stage, type, message and traceback" in p for p in problems(m))
    m["error"] = {"stage": "compile", "type": "NoParametersFound", "message": "x", "traceback": "Traceback ..."}
    assert problems(m) == []


def test_summary_validator_checks_labels_status_and_means():
    rows = [{"label": e["label"], "status": "not_run"} for e in CONFIG["configurations"]]
    assert res.validate_summary({"configurations": rows}, {}, CONFIG) == []
    assert any("labels" in p for p in res.validate_summary({"configurations": rows[:-1]}, {}, CONFIG))
    m = valid_completed()
    rows[1] = {"label": "mlp_top20_bits4", "status": "passed", "fhe_mean_seconds": m["fhe_latency"]["total"]["mean_seconds"]}
    good = res.validate_summary({"configurations": rows}, {"mlp_top20_bits4": {"metrics": m}}, CONFIG)
    assert good == []
    rows[1]["fhe_mean_seconds"] *= 2
    assert any("fhe_mean_seconds" in p for p in res.validate_summary({"configurations": rows}, {"mlp_top20_bits4": {"metrics": m}}, CONFIG))
    rows[1]["status"] = "failed_accuracy_gates"
    assert any("status" in p for p in res.validate_summary({"configurations": rows}, {"mlp_top20_bits4": {"metrics": m}}, CONFIG))


def test_grid_config_matches_the_approved_decisions():
    """D1: two bit-widths (4 = the largest feasible for all tiers, and the next lower, 3) over the three committed tiers."""
    entries = CONFIG["configurations"]
    assert {(e["tier"], e["n_bits"]) for e in entries} == {(t, b) for t in ("top_20", "top_50", "top_100") for b in (3, 4)} and len(entries) == 6
    assert CONFIG["mlp"]["n_accum_bits"] == 16 and CONFIG["latency_trials"] == 5 and CONFIG["seed"] == 42
    assert CONFIG["tolerances"] == {"t3_min_decision_agreement": 0.99, "t3_max_pr_auc_drop": 0.01}  # Phase 8's bars (D2)
    assert CONFIG["seed_stability"]["n_bits"] == 4 and len(CONFIG["seed_stability"]["seeds"]) == 3  # D5


# ---- the real results (skipped until they exist) ---------------------------------------------------------------------


@pytest.mark.skipif(not (res.RESULTS_DIR / "summary.json").exists(), reason="Phase 9 result files not present yet")
def test_real_results_validate_and_latency_means_match_raw_trials():
    report = res.validate_all()
    assert report["problems"] == []
    for label, r in res.load_results().items():
        m = r["metrics"]
        if m.get("fhe_latency"):
            for block in ("encrypt", "run", "decrypt", "total"):
                raw = np.asarray(m["fhe_latency"][block]["trials_seconds"], dtype=float)
                assert m["fhe_latency"][block]["mean_seconds"] == pytest.approx(raw.mean(), rel=1e-12), (label, block)
                assert m["fhe_latency"][block]["std_seconds"] == pytest.approx(raw.std(ddof=1), rel=1e-12), (label, block)
