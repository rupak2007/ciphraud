"""Phase 10 results-validator tests: synthetic result dicts (pure, every environment) that the validator must accept when honest and
reject when a claim is not supported by the raw per-request records -- plus the recorded real results when they exist."""

from __future__ import annotations

import copy
import json

import pytest

from src.analysis.phase10_results import (
    MLP_T3_DISCLOSURE, build_summary, q_in_digest, validate_deployment, validate_metrics, validate_preflight, validate_provenance,
)
from src.benchmark.phase10_roundtrip import summarize_requests
from src.config import PROJECT_ROOT, load_config
from src.fhe.deploy import request_positions

H = "a" * 64


def _case(kind: str = "lr"):
    rows = {
        "5": {"y_true": 0, "q_in_sha256": q_in_digest([1]), "q_in_length": 1, "clear_ints": [10, -3], "simulate_ints": [10, -3], "recorded_ints": [10, -3], "clear_equals_simulate": True, "clear_equals_recorded": True, "probability": 0.2},
        "9": {"y_true": 1, "q_in_sha256": q_in_digest([2]), "q_in_length": 1, "clear_ints": [7, 8], "simulate_ints": [7, 8], "recorded_ints": None, "clear_equals_simulate": True, "clear_equals_recorded": None, "probability": 0.9},
    }
    reference = {"id": "m", "fixed_position": 5, "t1_positions": [5, 9], "threshold": 0.5, "rows": rows, "reproduction_proof": {"passed": True, "circuit_stats_match": True}}
    manifest = {
        "id": "m", "model_type": kind, "security_level": "SecurityLevel.SECURITY_128_BITS", "t3_disclosure": MLP_T3_DISCLOSURE if kind == "mlp" else None,
        "artifacts": {"server_zip": {"sha256": H}, "client_zip": {"sha256": H}, "preprocessing_json": {"sha256": H}},
        "source": {"handoff_npz_sha256": H, "metrics_sha256": H},
        "circuit_stats": {"size_of_bootstrap_keys_bytes": 0, "size_of_keyswitch_keys_bytes": 0, "size_of_secret_keys_bytes": 1000},
    }
    config_model = {"id": "m", "t1_rows": True, "repeat_fixed_row": 2}
    plan = request_positions(5, [5, 9], True, 2)
    requests = []
    for i, position in enumerate(plan):
        ref = rows[str(position)]
        requests.append({
            "index": i, "position": position, "role": "fixed" if i == 0 else "repeat", "y_true": ref["y_true"], "probability": ref["probability"], "decision": ref["probability"] >= 0.5,
            "threshold": 0.5, "reference_probability": ref["probability"], "probability_abs_diff": 0.0, "reference_decision": ref["probability"] >= 0.5, "decrypted_ints": ref["clear_ints"],
            "correctness": {"equals_clear_quantized": True, "equals_simulate": True, "equals_recorded_source_phase": None if ref["recorded_ints"] is None else True, "exact_match": True},
            "client_timings": {k: 0.001 * (i + 1) for k in ("preprocess_seconds", "quantize_seconds", "encrypt_seconds", "serialize_seconds", "deserialize_seconds", "decrypt_seconds", "postprocess_seconds")},
            "http_roundtrip_seconds": 0.5 + 0.01 * i, "server_timings": {"input_deserialize_s": 0.01, "fhe_run_s": 0.2 + 0.01 * i, "output_serialize_s": 0.01, "handler_total_s": 0.3},
            "transport_and_framework_seconds": 0.5 + 0.01 * i - 0.3, "bytes": {"request_body": 456, "response_body": 10064},
        })
    metrics = {
        "id": "m", "model_type": kind, "mode": "in_process_client", "request_positions": plan, "t3_disclosure": manifest["t3_disclosure"], "test_partition_touched": False,
        "status": "passed", "requests": requests, "summary": summarize_requests(requests, 5), "keys_resident_after_delete": False,
        "one_time": {"secret_key_sent": False}, "artifact_compatibility": {"match": True}, "memory": {"aborted": False},
        "wire_audit": {"classes": ["no_body", "evaluation_keys"] + ["input_ciphertext"] * 4 + ["no_body", "no_body"], "all_bodies_are_evaluation_keys_or_ciphertexts": True},
    }
    return manifest, reference, config_model, metrics


def _problems(metrics, manifest, reference, config_model):
    return validate_metrics("m", metrics, manifest, reference, config_model)


def test_an_honest_result_validates_clean_for_every_model_type():
    for kind in ("lr", "xgboost", "mlp"):
        manifest, reference, config_model, metrics = _case(kind)
        assert _problems(metrics, manifest, reference, config_model) == []
        assert validate_deployment("m", manifest, reference) == []


@pytest.mark.parametrize(
    "mutate, fragment",
    [
        (lambda m: m["requests"][0].update(decrypted_ints=[10, -4]), "recomputed"),
        (lambda m: m["requests"][1]["correctness"].update(exact_match=False), "not exact"),
        (lambda m: m["requests"][0]["correctness"].update(equals_simulate=False), "recomputed"),
        (lambda m: m["wire_audit"]["classes"].__setitem__(2, "UNRECOGNIZED"), "neither the evaluation keys nor"),
        (lambda m: m["wire_audit"].update(all_bodies_are_evaluation_keys_or_ciphertexts=False), "neither the evaluation keys nor"),
        (lambda m: m.update(keys_resident_after_delete=True), "still resident"),
        (lambda m: m["one_time"].update(secret_key_sent=True), "secret_key_sent"),
        (lambda m: m["artifact_compatibility"].update(match=False), "compatibility"),
        (lambda m: m["requests"][2].update(transport_and_framework_seconds=9.0), "transport_and_framework_seconds is not"),
        (lambda m: m["requests"][0]["server_timings"].pop("fhe_run_s"), "server timing keys"),
        (lambda m: m["summary"]["stats"]["server_fhe_run_s"].update(mean_seconds=99.0), "mean/std"),
        (lambda m: m["requests"].pop(), "not the plan"),
        (lambda m: m.update(request_positions=[5]), "differ from the configured plan"),
        (lambda m: m.update(test_partition_touched=True), "test_partition_touched"),
        (lambda m: m.update(status="mystery"), "unknown status"),
        (lambda m: m["memory"].update(aborted=True), "memory guard"),
    ],
)
def test_the_validator_rejects_each_unsupported_claim(mutate, fragment):
    manifest, reference, config_model, metrics = _case("lr")
    mutate(metrics)
    problems = _problems(metrics, manifest, reference, config_model)
    assert any(fragment in p for p in problems), problems


def test_a_negative_transport_time_is_rejected():
    manifest, reference, config_model, metrics = _case("lr")
    r = metrics["requests"][0]
    r["http_roundtrip_seconds"] = 0.1
    r["transport_and_framework_seconds"] = r["http_roundtrip_seconds"] - r["server_timings"]["handler_total_s"]
    assert any("negative" in p for p in _problems(metrics, manifest, reference, config_model))


def test_the_mlp_t3_disclosure_is_required_verbatim_in_the_manifest_and_the_result():
    manifest, reference, config_model, metrics = _case("mlp")
    metrics["t3_disclosure"] = "the MLP failed T3"
    assert any("verbatim MLP T3 disclosure" in p for p in _problems(metrics, manifest, reference, config_model))
    manifest["t3_disclosure"] = None
    assert any("verbatim MLP T3 disclosure" in p for p in validate_deployment("m", manifest, reference))


def test_deployment_requires_a_passed_reproduction_proof_and_a_recorded_security_level():
    manifest, reference, _, _ = _case("lr")
    reference["reproduction_proof"]["passed"] = False
    assert any("reproduction proof" in p for p in validate_deployment("m", manifest, reference))
    manifest, reference, _, _ = _case("lr")
    manifest["security_level"] = None
    assert any("security level" in p for p in validate_deployment("m", manifest, reference))
    manifest, reference, _, _ = _case("lr")
    reference["rows"]["5"]["clear_equals_recorded"] = False
    assert any("does not reproduce" in p for p in validate_deployment("m", manifest, reference))


def _failed(status: str, **failure):
    manifest, reference, config_model, metrics = _case("mlp")
    metrics.pop("requests"), metrics.pop("summary"), metrics.pop("wire_audit")
    metrics["status"] = status
    metrics["failure"] = {"stage": "serve", "secret_key_stayed_local": True, "secret_key_sent": False, "fhe_execution_started": False, **failure}
    return manifest, reference, config_model, metrics


@pytest.mark.parametrize("status", ["aborted_memory", "preflight_failed_go_no_go", "run_timeout", "error", "preflight_failed_keys"])
def test_evidence_backed_failures_validate_when_they_affirm_the_key_stayed_local(status):
    manifest, reference, config_model, metrics = _failed(status)
    assert _problems(metrics, manifest, reference, config_model) == []


def test_failures_without_evidence_or_with_an_unaffirmed_key_are_rejected():
    manifest, reference, config_model, metrics = _failed("aborted_memory")
    metrics.pop("failure")
    assert any("without a failure record" in p for p in _problems(metrics, manifest, reference, config_model))
    manifest, reference, config_model, metrics = _failed("aborted_memory", secret_key_stayed_local=False)
    assert any("stayed local" in p for p in _problems(metrics, manifest, reference, config_model))
    manifest, reference, config_model, metrics = _failed("preflight_failed_go_no_go", fhe_execution_started=True)
    assert any("must not have started" in p for p in _problems(metrics, manifest, reference, config_model))


def test_preflight_requires_every_stage_to_have_been_watched_from_outside_by_a_healthy_watchdog():
    watch = {"n_samples": 10, "watchdog_error": None}
    preflight = {"status": "preflight_passed", "preflight_only": True, "client_keyset_file_deleted": True, "stages": {"keys": {"parent_watch": watch}, "serve": {"parent_watch": watch, "fhe_execution_started": False}}}
    assert validate_preflight("m", preflight) == []
    bad = copy.deepcopy(preflight)
    bad["stages"]["keys"]["parent_watch"] = {"n_samples": 0, "watchdog_error": None}
    assert any("watched from outside" in p for p in validate_preflight("m", bad))
    bad = copy.deepcopy(preflight)
    bad["stages"]["serve"]["fhe_execution_started"] = True
    assert any("started the FHE run" in p for p in validate_preflight("m", bad))
    bad = copy.deepcopy(preflight)
    bad["client_keyset_file_deleted"] = False
    assert any("not deleted" in p for p in validate_preflight("m", bad))


def test_provenance_must_carry_the_code_hashes_and_can_be_checked_against_the_files_on_disk():
    real = "src/analysis/phase10_results.py"
    from src.fhe.deploy import sha256_path

    good = {"config_hash": "x", "git_commit": "y", "library_versions": {}, "timestamp_utc": "t", "deployment_manifest_sha256": H, "code_files": {real: {"sha256": sha256_path(PROJECT_ROOT / real), "git_blob_sha1": None}}}
    assert validate_provenance("m", good, check_code_hashes=True) == []
    changed = copy.deepcopy(good)
    changed["code_files"][real]["sha256"] = H
    assert any("has changed since" in p for p in validate_provenance("m", changed, check_code_hashes=True))
    assert validate_provenance("m", changed, check_code_hashes=False) == []
    assert any("lacks code_files" in p for p in validate_provenance("m", {k: v for k, v in good.items() if k != "code_files"}, check_code_hashes=False))


def test_summary_reports_one_row_per_model_with_the_decomposed_times():
    manifest, reference, config_model, metrics = _case("lr")
    metrics["memory"] = {"per_process": {"server": {"peak_rss_mb": 123.0}}, "aborted": False}
    summary = build_summary({"m": {"manifest": manifest, "metrics": metrics}})
    row = summary["models"][0]
    assert row["status"] == "passed" and row["n_requests"] == 4 and row["all_requests_exact_match"] is True
    assert row["fhe_run_mean_s"] > 0 and row["http_roundtrip_mean_s"] > row["fhe_run_mean_s"] and row["server_peak_rss_mb"] == 123.0


# ---- the recorded real results (present once the Phase 10 runs have been made) --------------------------------------------


def test_the_recorded_phase_10_results_validate_clean_when_present():
    from src.analysis.phase10_results import current_record_dir, validate_all

    config = load_config("configs/phase10/client_server.yaml")
    results = PROJECT_ROOT / config["output"]["results_dir"]
    records = {m["id"]: current_record_dir(results / m["id"]) for m in config["models"]}
    if any(record is None for record in records.values()):
        pytest.skip("the Phase 10 runs have not all been recorded yet")
    problems, found = validate_all(config)
    assert problems == [] and set(found) == {m["id"] for m in config["models"]}
    statuses = {model: json.loads((record / "metrics.json").read_text())["status"] for model, record in records.items()}
    assert statuses["lr_top20_bits16"] == "passed" and statuses["mlp_top20_bits3"] == "passed"
    # XGBoost is whatever was really recorded, but never a pass without completed, exact requests
    xgb = json.loads((records["xgboost_top50_bits14"] / "metrics.json").read_text())
    assert xgb["status"] in {"passed", "aborted_memory", "run_timeout", "error"}
    assert xgb["status"] != "passed" or (xgb["requests"] and all(r["correctness"]["exact_match"] for r in xgb["requests"]))


# ---- committed artifacts keep an integrity digest of the quantized input, never the vector -------------------------------------------------

def test_reference_rows_must_carry_a_digest_and_not_the_quantized_input_vector():
    manifest, reference, _, _ = _case("lr")
    assert validate_deployment("m", manifest, reference) == []
    reference["rows"]["5"]["q_in"] = [1]
    assert any("stores the quantized input vector" in p for p in validate_deployment("m", manifest, reference))
    manifest, reference, _, _ = _case("lr")
    del reference["rows"]["9"]["q_in_sha256"]
    assert any("q_in_sha256" in p for p in validate_deployment("m", manifest, reference))


def test_the_input_digest_is_over_little_endian_int64_and_is_deterministic():
    import hashlib

    import numpy as np

    assert q_in_digest([1, -2, 3]) == hashlib.sha256(np.array([1, -2, 3], dtype="<i8").tobytes()).hexdigest()
    assert q_in_digest(np.array([[1, -2, 3]], dtype=np.int32)) == q_in_digest([1, -2, 3])
    assert q_in_digest([1, 2]) != q_in_digest([2, 1])


# ---- derived failure flags: a stage killed before writing its state must not silently report False ------------------------------------------

def _killed_serve_stage(**failure):
    """The recorded XGBoost shape: staged client, the serve stage killed by the outside watchdog (no go_no_go / server_startup state)."""
    manifest, reference, config_model, metrics = _failed("aborted_memory", server_started=False, **failure)
    metrics["mode"] = "staged_client"
    metrics["request_positions"] = [reference["fixed_position"]]
    metrics["stages"] = {"serve": {"exit_code": -9, "status": "aborted_memory"}}
    return manifest, reference, config_model, metrics


def _corrections(metrics, server_started=True, fhe=True):
    failure = metrics["failure"]
    failure["server_started_corrected"], failure["fhe_execution_started_corrected"] = server_started, fhe
    ev = {"server_ready_event": True, "evaluation_keys_registered_bytes": 123}
    metrics["post_run_corrections"] = [
        {"field": "failure.server_started", "recorded_value": False, "corrected_value": server_started, "basis": "observed", "reason": "r", "evidence": ev},
        {"field": "failure.fhe_execution_started", "recorded_value": False, "corrected_value": fhe, "basis": "inferred", "reason": "r", "evidence": {}},
    ]
    return metrics


def test_a_killed_serve_stage_with_default_false_flags_is_rejected_until_it_carries_corrections():
    manifest, reference, config_model, metrics = _killed_serve_stage()
    problems = _problems(metrics, manifest, reference, config_model)
    assert any("failure.server_started=False is a default, not an observation" in p for p in problems)
    assert any("failure.fhe_execution_started=False is a default, not an observation" in p for p in problems)
    assert _problems(_corrections(metrics), manifest, reference, config_model) == []


def test_a_completed_serve_stage_record_needs_no_correction():
    manifest, reference, config_model, metrics = _killed_serve_stage()
    metrics["stages"]["serve"]["go_no_go"] = {"go": True}
    assert _problems(metrics, manifest, reference, config_model) == []


def test_a_correction_must_preserve_the_raw_value_and_carry_its_evidence():
    manifest, reference, config_model, metrics = _killed_serve_stage()
    metrics = _corrections(metrics)
    metrics["failure"]["server_started"] = True  # the raw field was silently rewritten instead of corrected
    assert any("raw values must be preserved" in p for p in _problems(metrics, manifest, reference, config_model))
    manifest, reference, config_model, metrics = _killed_serve_stage()
    metrics = _corrections(metrics)
    metrics["post_run_corrections"][0]["evidence"] = {}
    assert any("server_ready event" in p for p in _problems(metrics, manifest, reference, config_model))
    manifest, reference, config_model, metrics = _killed_serve_stage()
    metrics = _corrections(metrics)
    metrics["post_run_corrections"][1]["basis"] = "guess"
    assert any("basis must be" in p for p in _problems(metrics, manifest, reference, config_model))
    manifest, reference, config_model, metrics = _killed_serve_stage()
    metrics = _corrections(metrics)
    metrics["failure"]["fhe_execution_started_corrected"] = False
    assert any("_corrected does not equal" in p for p in _problems(metrics, manifest, reference, config_model))


def test_the_raw_server_log_is_cross_checked_when_it_is_available(tmp_path):
    manifest, reference, config_model, metrics = _killed_serve_stage()
    metrics = _corrections(metrics)
    log = tmp_path / "server.log"
    log.write_text(json.dumps({"event": "server_ready", "pid": 1}) + chr(10), encoding="utf-8")
    assert validate_metrics("m", metrics, manifest, reference, config_model, log) == []
    uncorrected = copy.deepcopy(metrics)
    uncorrected.pop("post_run_corrections")
    uncorrected["failure"].pop("server_started_corrected")
    problems = validate_metrics("m", uncorrected, manifest, reference, config_model, log)
    assert any("shows server_ready" in p for p in problems)


# ---- archived as-run records (attempt_* / superseded_*) stay verifiable and consistent ------------------------------------------------------

def _archive(tmp_path, metrics):
    from src.analysis.phase10_results import validate_archive  # noqa: F401
    import hashlib

    d = tmp_path / "attempt_1_aborted_memory"
    d.mkdir()
    (d / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    (d / "SHA256SUMS.json").write_text(json.dumps({"metrics.json": hashlib.sha256((d / "metrics.json").read_bytes()).hexdigest()}), encoding="utf-8")
    return d


def test_an_archive_must_match_its_recorded_hashes_and_stay_internally_consistent(tmp_path):
    from src.analysis.phase10_results import validate_archive

    manifest, reference, config_model, metrics = _killed_serve_stage()
    metrics = _corrections(metrics)
    d = _archive(tmp_path, metrics)
    assert validate_archive("m", d, manifest, reference, config_model) == []
    (d / "metrics.json").write_text(json.dumps({**metrics, "status": "passed"}), encoding="utf-8")  # history silently rewritten
    assert any("modified since it was archived" in p for p in validate_archive("m", d, manifest, reference, config_model))
    (d / "SHA256SUMS.json").unlink()
    assert any("no SHA256SUMS.json" in p for p in validate_archive("m", d, manifest, reference, config_model))


def test_an_archived_record_still_needs_its_derived_flag_corrections(tmp_path):
    from src.analysis.phase10_results import validate_archive

    manifest, reference, config_model, metrics = _killed_serve_stage()  # default-False flags, no corrections
    d = _archive(tmp_path, metrics)
    assert any("is a default, not an observation" in p for p in validate_archive("m", d, manifest, reference, config_model))


def test_the_current_record_is_the_top_level_one_or_else_the_newest_numbered_attempt(tmp_path):
    from src.analysis.phase10_results import current_record_dir

    assert current_record_dir(tmp_path) is None
    for name in ("attempt_1_aborted_memory", "attempt_2_aborted_memory", "attempt_10_aborted_memory", "superseded_pre_sender_fix"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "metrics.json").write_text("{}", encoding="utf-8")
    assert current_record_dir(tmp_path).name == "attempt_10_aborted_memory"  # numeric, not lexicographic; `superseded_*` is never current
    (tmp_path / "metrics.json").write_text("{}", encoding="utf-8")
    assert current_record_dir(tmp_path) == tmp_path
