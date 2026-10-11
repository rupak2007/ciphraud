"""Validate the Phase 10 client/server results (`docs/client_server.md`); pure numpy/json, runs on any platform.

    python -m src.analysis.phase10_results                     # validate; exit code 1 if any problem
    python -m src.analysis.phase10_results --check-code-hashes  # also require the recorded source hashes to equal the files on disk NOW
    python -m src.analysis.phase10_results --summary            # also write results/phase10_client_server/summary.json

Checks, per configured model: the deployment manifest/reference (reproduction proof passed, hashes well-formed); `metrics.json`
(status vocabulary, per-request exact-match / decision / wire-audit / key-deletion claims RECOMPUTED from the raw per-request
records, the transport arithmetic, summary statistics against the raw trials, the verbatim MLP T3 disclosure); failure records
(stage, whether the server started and FHE began, secret key stayed local); and the provenance code hashes. A status of
`passed` is only accepted if every request in the plan really was exact.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from src.benchmark.harness import trial_stats
from src.config import PROJECT_ROOT, load_config

MLP_T3_DISCLOSURE = (
    "The client/server FHE deployment was demonstrated using mlp_top20_bits3, which passed the relevant execution/integrity checks "
    "but failed the Phase 9 quantization-accuracy gate."
)
FINAL_STATUSES = {"passed", "failed_exact_match", "preflight_failed_keys", "preflight_failed_serve", "preflight_failed_go_no_go", "aborted_memory", "run_timeout", "error"}
PREFLIGHT_STATUSES = FINAL_STATUSES | {"preflight_passed"}
CLIENT_KEYS = {"preprocess_seconds", "quantize_seconds", "encrypt_seconds", "serialize_seconds", "deserialize_seconds", "decrypt_seconds", "postprocess_seconds"}
SERVER_KEYS = {"input_deserialize_s", "fhe_run_s", "output_serialize_s", "handler_total_s"}


def q_in_digest(q: Any) -> str:
    """sha256 over the quantized input vector as little-endian int64 -- the integrity reference committed INSTEAD of the vector itself."""
    return hashlib.sha256(np.asarray(q, dtype="<i8").reshape(-1).tobytes()).hexdigest()


def _hex64(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def validate_deployment(model_id: str, manifest: dict[str, Any], reference: dict[str, Any]) -> list[str]:
    problems = []
    if manifest.get("id") != model_id or reference.get("id") != model_id:
        problems.append(f"{model_id}: manifest/reference id mismatch")
    for name in ("server_zip", "client_zip"):
        if not _hex64(manifest["artifacts"][name]["sha256"]):
            problems.append(f"{model_id}: manifest {name} sha256 is malformed")
    if not _hex64(manifest["artifacts"]["preprocessing_json"]["sha256"]) or not _hex64(manifest["source"]["handoff_npz_sha256"]) or not _hex64(manifest["source"]["metrics_sha256"]):
        problems.append(f"{model_id}: a manifest source/preprocessing hash is malformed")
    if not reference["reproduction_proof"].get("passed"):
        problems.append(f"{model_id}: the reproduction proof did not pass")
    if reference["reproduction_proof"].get("circuit_stats_match") is not True:
        problems.append(f"{model_id}: the rebuilt circuit statistics do not equal the source phase's")
    for position, row in reference["rows"].items():
        if row["clear_equals_simulate"] is not True or row["clear_equals_recorded"] is False:
            problems.append(f"{model_id}: reference row {position} does not reproduce (clear vs simulate/recorded)")
        if "q_in" in row:
            problems.append(f"{model_id}: reference row {position} stores the quantized input vector (q_in); committed artifacts keep only q_in_sha256")
        if not _hex64(row.get("q_in_sha256")) or not isinstance(row.get("q_in_length"), int) or row["q_in_length"] <= 0:
            problems.append(f"{model_id}: reference row {position} lacks a well-formed q_in_sha256 / q_in_length integrity reference")
    if manifest["security_level"] is None:
        problems.append(f"{model_id}: the TFHE security level was not recorded")
    if manifest["model_type"] == "mlp" and manifest["t3_disclosure"] != MLP_T3_DISCLOSURE:
        problems.append(f"{model_id}: the manifest lacks the verbatim MLP T3 disclosure")
    return problems


def _check_request(model_id: str, r: dict[str, Any], reference: dict[str, Any], staged: bool) -> list[str]:
    problems = []
    tag = f"{model_id} request {r.get('index', 0)}"
    ref = reference["rows"].get(str(r["position"]))
    if ref is None:
        return [f"{tag}: position {r['position']} is not in the reference"]
    flat = np.asarray(r["decrypted_ints"]).reshape(-1)
    c = r["correctness"]
    recomputed = {
        "equals_clear_quantized": bool(np.array_equal(flat, np.asarray(ref["clear_ints"]))),
        "equals_simulate": bool(np.array_equal(flat, np.asarray(ref["simulate_ints"]))),
        "equals_recorded_source_phase": None if ref["recorded_ints"] is None else bool(np.array_equal(flat, np.asarray(ref["recorded_ints"]))),
    }
    for key, value in recomputed.items():
        if c[key] != value:
            problems.append(f"{tag}: recorded {key}={c[key]} but recomputed {value} from the decrypted integers")
    if c["exact_match"] != (recomputed["equals_clear_quantized"] and recomputed["equals_simulate"] and recomputed["equals_recorded_source_phase"] is not False):
        problems.append(f"{tag}: exact_match is inconsistent with its components")
    if "probability_abs_diff" in r and abs(r["probability"] - ref["probability"]) != r["probability_abs_diff"]:
        problems.append(f"{tag}: probability_abs_diff is not |probability - reference|")
    if set(r["server_timings"]) != SERVER_KEYS:
        problems.append(f"{tag}: server timing keys are {sorted(r['server_timings'])}")
    if not staged and set(r["client_timings"]) != CLIENT_KEYS:
        problems.append(f"{tag}: client timing keys are {sorted(r['client_timings'])}")
    if staged and not CLIENT_KEYS <= set(r["client_timings"]):
        problems.append(f"{tag}: client timing keys are {sorted(r['client_timings'])}")
    timings = list(r["client_timings"].values()) + list(r["server_timings"].values()) + [r["http_roundtrip_seconds"]]
    if any((not isinstance(t, (int, float))) or t < 0 for t in timings if not isinstance(t, dict)):
        problems.append(f"{tag}: a timing is negative or non-numeric")
    if abs(r["transport_and_framework_seconds"] - (r["http_roundtrip_seconds"] - r["server_timings"]["handler_total_s"])) > 1e-9:
        problems.append(f"{tag}: transport_and_framework_seconds is not http_roundtrip - handler_total")
    if r["transport_and_framework_seconds"] < 0:
        problems.append(f"{tag}: transport time is negative (the server handler cannot exceed the client's round trip)")
    return problems


def validate_metrics(model_id: str, metrics: dict[str, Any], manifest: dict[str, Any], reference: dict[str, Any], config_model: dict[str, Any], server_log: Path | None = None) -> list[str]:
    from src.fhe.deploy import request_positions

    problems: list[str] = []
    status = metrics.get("status")
    if status not in FINAL_STATUSES:
        return [f"{model_id}: unknown status {status!r}"]
    if metrics.get("test_partition_touched") is not False:
        problems.append(f"{model_id}: test_partition_touched is not False")
    if manifest["model_type"] == "mlp" and metrics.get("t3_disclosure") != MLP_T3_DISCLOSURE:
        problems.append(f"{model_id}: the result lacks the verbatim MLP T3 disclosure")
    staged = metrics.get("mode") == "staged_client"
    plan = [reference["fixed_position"]] if staged else request_positions(reference["fixed_position"], reference["t1_positions"], bool(config_model["t1_rows"]), int(config_model["repeat_fixed_row"]))
    if metrics.get("request_positions") != plan:
        problems.append(f"{model_id}: request_positions {metrics.get('request_positions')} differ from the configured plan {plan}")

    if status == "passed":
        requests = metrics.get("requests", [])
        if [r["position"] for r in requests] != plan:
            problems.append(f"{model_id}: passed, but the executed requests {[r['position'] for r in requests]} are not the plan {plan}")
        for r in requests:
            problems.extend(_check_request(model_id, r, reference, staged))
            if not r["correctness"]["exact_match"]:
                problems.append(f"{model_id}: status is passed but request at position {r['position']} was not exact")
        audit = metrics.get("wire_audit", {})
        if audit.get("all_bodies_are_evaluation_keys_or_ciphertexts") is not True or "UNRECOGNIZED" in audit.get("classes", []):
            problems.append(f"{model_id}: the wire audit found a request body that is neither the evaluation keys nor an input ciphertext")
        if audit.get("classes", []).count("evaluation_keys") != 1 or audit.get("classes", []).count("input_ciphertext") != len(requests):
            problems.append(f"{model_id}: the wire audit does not show exactly one key upload and one ciphertext per request")
        if staged:
            if metrics.get("client_keyset_file_deleted") is not True:
                problems.append(f"{model_id}: the staged client's secret keyset file was not deleted")
        else:
            if metrics.get("keys_resident_after_delete") is not False:
                problems.append(f"{model_id}: server-side key material was still resident after DELETE")
            if metrics["one_time"].get("secret_key_sent") is not False:
                problems.append(f"{model_id}: secret_key_sent is not False")
            if metrics.get("artifact_compatibility", {}).get("match") is not True:
                problems.append(f"{model_id}: the client/server artifact compatibility check did not match")
            fixed = [r for r in requests if r["position"] == reference["fixed_position"]]
            summary = metrics.get("summary", {})
            if summary.get("n_requests") != len(fixed):
                problems.append(f"{model_id}: summary n_requests {summary.get('n_requests')} != {len(fixed)} fixed-row requests")
            else:
                for name, stats in summary.get("stats", {}).items():
                    values = _series(fixed, name)
                    if values is None or not np.allclose(values, stats["trials_seconds"], rtol=0, atol=0):
                        problems.append(f"{model_id}: summary series {name} does not equal the raw per-request values")
                    elif not np.isclose(stats["mean_seconds"], trial_stats(values)["mean_seconds"], rtol=1e-12, atol=0) or not np.isclose(stats["std_seconds"], trial_stats(values)["std_seconds"], rtol=1e-12, atol=0):
                        problems.append(f"{model_id}: summary {name} mean/std do not match the raw values")
        if metrics.get("memory", {}).get("aborted") is True:
            problems.append(f"{model_id}: passed, yet the memory guard reports an abort")
    else:
        failure = metrics.get("failure")
        if not failure:
            problems.append(f"{model_id}: status {status} without a failure record")
        else:
            for key in ("secret_key_stayed_local", "secret_key_sent"):
                if key not in failure:
                    problems.append(f"{model_id}: failure record lacks {key}")
            if failure.get("secret_key_stayed_local") is not True or failure.get("secret_key_sent") is not False:
                problems.append(f"{model_id}: a failure record does not affirm that the secret key stayed local")
            if "fhe_execution_started" not in failure:
                problems.append(f"{model_id}: failure record lacks fhe_execution_started")
            if status == "preflight_failed_go_no_go" and failure.get("fhe_execution_started") is not False:
                problems.append(f"{model_id}: a go/no-go refusal must not have started the FHE run")
            problems.extend(validate_failure_state(model_id, metrics, server_log))
    return problems


DERIVED_FLAGS = ("server_started", "fhe_execution_started")


def _server_ready_in_log(server_log: Path | None) -> bool | None:
    """True/False from the raw server log, None when the log is not available (it is local and gitignored)."""
    if server_log is None or not server_log.exists():
        return None
    for line in server_log.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            if json.loads(line).get("event") == "server_ready":
                return True
        except ValueError:
            continue
    return False


def validate_failure_state(model_id: str, metrics: dict[str, Any], server_log: Path | None = None) -> list[str]:
    """`failure.server_started` / `failure.fhe_execution_started` are DERIVED from the serve stage's own state file. When the stage is killed
    before it writes that file (the outside watchdog's abort), the derivation silently yields False, whatever really happened. Such a
    record is only accepted if each affected flag carries an explicit `post_run_corrections` entry (raw value preserved, evidence attached)."""
    problems: list[str] = []
    failure = metrics.get("failure") or {}
    corrections = {c.get("field"): c for c in metrics.get("post_run_corrections", []) if isinstance(c, dict)}
    for name, c in corrections.items():
        tag = f"{model_id}: correction of {name}"
        for key in ("field", "recorded_value", "corrected_value", "basis", "reason", "evidence"):
            if key not in c:
                problems.append(f"{tag} lacks {key}")
        if not (isinstance(name, str) and name.startswith("failure.")) or name[len("failure."):] not in DERIVED_FLAGS:
            problems.append(f"{tag}: only the derived flags {DERIVED_FLAGS} may be corrected")
            continue
        flag = name[len("failure."):]
        if failure.get(flag) != c.get("recorded_value"):
            problems.append(f"{tag}: the raw field now reads {failure.get(flag)!r}, not the recorded {c.get('recorded_value')!r} (raw values must be preserved)")
        if failure.get(f"{flag}_corrected") != c.get("corrected_value"):
            problems.append(f"{tag}: failure.{flag}_corrected does not equal the correction's corrected_value")
        if c.get("basis") not in ("observed", "inferred"):
            problems.append(f"{tag}: basis must be 'observed' or 'inferred'")
        if flag == "server_started" and c.get("corrected_value") is True and not (isinstance(c.get("evidence"), dict) and c["evidence"].get("server_ready_event") and isinstance(c["evidence"].get("evaluation_keys_registered_bytes"), int)):
            problems.append(f"{tag}: claiming the server started needs its server_ready event and registered key bytes as evidence")
    stage_state_written = "go_no_go" in metrics.get("stages", {}).get("serve", {})
    if failure.get("stage") == "serve" and metrics.get("mode") == "staged_client" and not stage_state_written:
        for flag in DERIVED_FLAGS:
            if failure.get(flag) is False and f"failure.{flag}" not in corrections:
                problems.append(f"{model_id}: the serve stage was killed before writing its state, so failure.{flag}=False is a default, not an observation; it needs a post_run_corrections entry")
    ready = _server_ready_in_log(server_log)
    if ready is True and failure.get("server_started") is False and failure.get("server_started_corrected") is not True:
        problems.append(f"{model_id}: failure.server_started is False but the raw server log shows server_ready")
    return problems


def _series(requests: list[dict[str, Any]], name: str) -> list[float] | None:
    """The raw per-request values behind a summary statistic, or None if they cannot be recomputed (a missing key is a problem to
    report, never a crash of the validator)."""
    try:
        if name.startswith("client_"):
            key = name[len("client_") :]
            return [r["client_timings"][key] for r in requests if key in r["client_timings"]]
        if name.startswith("server_"):
            key = name[len("server_") :]
            return [r["server_timings"][key] for r in requests]
        if name in ("http_roundtrip_seconds", "transport_and_framework_seconds"):
            return [r[name] for r in requests]
    except KeyError:
        return None
    return None


def validate_preflight(model_id: str, preflight: dict[str, Any]) -> list[str]:
    problems = []
    if preflight.get("status") not in PREFLIGHT_STATUSES:
        return [f"{model_id}: preflight has an unknown status {preflight.get('status')!r}"]
    if preflight.get("preflight_only") is not True:
        problems.append(f"{model_id}: preflight.json is not marked preflight_only")
    if preflight["status"] == "preflight_passed":
        if preflight["stages"]["serve"].get("fhe_execution_started") not in (False, None):
            problems.append(f"{model_id}: a preflight-only run started the FHE run")
        if preflight.get("client_keyset_file_deleted") is not True:
            problems.append(f"{model_id}: the staged client keyset file was not deleted after the preflight")
    for name, stage in preflight["stages"].items():
        watch = stage.get("parent_watch")
        if watch is None or watch.get("n_samples", 0) <= 0 or watch.get("watchdog_error") is not None:
            problems.append(f"{model_id}: preflight stage {name} was not watched from outside by a healthy watchdog")
    return problems


def validate_provenance(model_id: str, provenance: dict[str, Any], check_code_hashes: bool) -> list[str]:
    from src.fhe.deploy import sha256_path

    problems = []
    for key in ("config_hash", "git_commit", "library_versions", "timestamp_utc", "code_files", "deployment_manifest_sha256"):
        if key not in provenance:
            problems.append(f"{model_id}: provenance lacks {key}")
    if check_code_hashes:
        for rel, hashes in provenance.get("code_files", {}).items():
            path = PROJECT_ROOT / rel
            if not path.exists():
                problems.append(f"{model_id}: recorded source file {rel} no longer exists")
            elif sha256_path(path) != hashes["sha256"]:
                problems.append(f"{model_id}: {rel} has changed since this result was produced (sha256 differs from the recorded one)")
    return problems


def current_record_dir(model_dir: Path) -> Path | None:
    """The directory holding a model's CURRENT official record: `<model>/` itself, or -- for a model whose every attempt is archived
    (XGBoost, aborted twice) -- its newest `attempt_N_*` directory. Older attempts stay archived history."""
    if (model_dir / "metrics.json").exists():
        return model_dir
    attempts = [d for d in model_dir.glob("attempt_*") if d.is_dir() and (d / "metrics.json").exists()]
    return max(attempts, key=lambda d: int(d.name.split("_")[1])) if attempts else None


ARCHIVE_PREFIXES = ("attempt_", "superseded_")


def validate_archive(model_id: str, directory: Path, manifest: dict[str, Any], reference: dict[str, Any], config_model: dict[str, Any]) -> list[str]:
    """An archived as-run record (`attempt_*` for an XGBoost attempt, `superseded_*` for an official result replaced after a code change). It
    is history: its files must still equal the sha256 recorded when it was archived, its metrics must still be internally consistent, and its
    provenance keeps the code hashes of the code that produced it, so the LIVE code-hash comparison does not apply to it."""
    from src.fhe.deploy import sha256_path

    tag = f"{model_id}/{directory.name}"
    problems: list[str] = []
    sums = directory / "SHA256SUMS.json"
    if not sums.exists():
        return [f"{tag}: no SHA256SUMS.json (an archive must be verifiable against the hashes recorded when it was archived)"]
    for rel, expected in json.loads(sums.read_text(encoding="utf-8")).items():
        path = directory / rel
        if not path.exists():
            problems.append(f"{tag}: archived file {rel} is missing")
        elif sha256_path(path) != expected:
            problems.append(f"{tag}: archived file {rel} has been modified since it was archived")
    if (directory / "metrics.json").exists():
        metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
        problems.extend(f"{tag}: {p}" for p in validate_metrics(model_id, metrics, manifest, reference, config_model, directory / "raw_logs" / "server.log"))
    else:
        problems.append(f"{tag}: no metrics.json")
    for name in ("provenance.json", "preflight_provenance.json"):
        if (directory / name).exists():
            problems.extend(f"{tag}: {p}" for p in validate_provenance(model_id, json.loads((directory / name).read_text(encoding="utf-8")), False))
    if (directory / "preflight.json").exists():
        problems.extend(f"{tag}: {p}" for p in validate_preflight(model_id, json.loads((directory / "preflight.json").read_text(encoding="utf-8"))))
    return problems


def validate_all(config: dict[str, Any], check_code_hashes: bool = False) -> tuple[list[str], dict[str, Any]]:
    results_dir = PROJECT_ROOT / config["output"]["results_dir"]
    problems: list[str] = []
    found: dict[str, Any] = {}
    for model_cfg in config["models"]:
        model_id = model_cfg["id"]
        deploy_dir = results_dir / "deploy" / model_id
        if not (deploy_dir / "manifest.json").exists():
            problems.append(f"{model_id}: no deployment manifest")
            continue
        manifest = json.loads((deploy_dir / "manifest.json").read_text())
        reference = json.loads((deploy_dir / "reference.json").read_text())
        problems.extend(validate_deployment(model_id, manifest, reference))
        entry: dict[str, Any] = {"manifest": manifest}
        record = current_record_dir(results_dir / model_id)
        if record is not None:
            metrics = json.loads((record / "metrics.json").read_text(encoding="utf-8"))
            server_log = (record / "raw_logs" / "server.log") if record.name.startswith("attempt_") else PROJECT_ROOT / config["output"]["logs_dir"] / model_id / "server.log"
            problems.extend(validate_metrics(model_id, metrics, manifest, reference, model_cfg, server_log))
            problems.extend(validate_provenance(model_id, json.loads((record / "provenance.json").read_text(encoding="utf-8")), check_code_hashes))
            entry["metrics"] = metrics
            entry["record_dir"] = record.name if record.name.startswith("attempt_") else "."
        else:
            problems.append(f"{model_id}: no metrics.json (an attempted configuration must always leave a result, never be dropped)")
        if record is not None and (record / "preflight.json").exists():
            preflight = json.loads((record / "preflight.json").read_text(encoding="utf-8"))
            problems.extend(validate_preflight(model_id, preflight))
            entry["preflight"] = preflight
        for archive in sorted((results_dir / model_id).glob("*")):
            if archive.is_dir() and archive.name.startswith(ARCHIVE_PREFIXES):
                problems.extend(validate_archive(model_id, archive, manifest, reference, model_cfg))
        found[model_id] = entry
    return problems, found


def build_summary(found: dict[str, Any]) -> dict[str, Any]:
    rows = []
    for model_id, entry in found.items():
        m = entry.get("metrics", {})
        row: dict[str, Any] = {"id": model_id, "model_type": entry["manifest"]["model_type"], "status": m.get("status"), "mode": m.get("mode"), "n_requests": len(m.get("requests", []))}
        row["current_record"] = entry.get("record_dir", ".")
        if m.get("requests"):
            row["all_requests_exact_match"] = all(r["correctness"]["exact_match"] for r in m["requests"])
            if "summary" in m and m["summary"].get("stats"):
                stats = m["summary"]["stats"]
                row.update({"fhe_run_mean_s": stats["server_fhe_run_s"]["mean_seconds"], "http_roundtrip_mean_s": stats["http_roundtrip_seconds"]["mean_seconds"], "transport_mean_s": stats["transport_and_framework_seconds"]["mean_seconds"]})
            else:
                r0 = m["requests"][0]
                row.update({"fhe_run_mean_s": r0["server_timings"]["fhe_run_s"], "http_roundtrip_mean_s": r0["http_roundtrip_seconds"], "transport_mean_s": r0["transport_and_framework_seconds"]})
        if m.get("memory"):
            row["server_peak_rss_mb"] = m["memory"].get("per_process", {}).get("server", {}).get("peak_rss_mb")
        row["circuit_key_material_gb"] = (entry["manifest"]["circuit_stats"]["size_of_bootstrap_keys_bytes"] + entry["manifest"]["circuit_stats"]["size_of_keyswitch_keys_bytes"] + entry["manifest"]["circuit_stats"]["size_of_secret_keys_bytes"]) / 1e9
        rows.append(row)
    return {"models": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/phase10/client_server.yaml")
    parser.add_argument("--check-code-hashes", action="store_true")
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    problems, found = validate_all(config, check_code_hashes=args.check_code_hashes)
    print(f"Phase 10 result validation: {len(found)}/{len(config['models'])} models found")
    if args.summary:
        out = PROJECT_ROOT / config["output"]["results_dir"] / "summary.json"
        out.write_text(json.dumps(build_summary(found), indent=2, sort_keys=True), encoding="utf-8")
        print(f"wrote {out}")
    if problems:
        print("PROBLEMS:")
        for p in problems:
            print(" -", p)
        sys.exit(1)
    print("OK -- no problems")


if __name__ == "__main__":
    main()
