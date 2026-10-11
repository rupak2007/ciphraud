"""Post-run corrections to the recorded Phase 10 results (no experiment is re-run; the raw evidence is never rewritten).

    python -m src.analysis.phase10_correct_record --minimize-reference        # q_in vectors -> sha256 digests in the committed reference.json files
    python -m src.analysis.phase10_correct_record --correct-serve-abort xgboost_top50_bits14

`--correct-serve-abort`: the staged serve stage was killed by the outside memory watchdog BEFORE it wrote its own state file, so the
booleans `failure.server_started` and `failure.fhe_execution_started` (derived from that file) fell back to False. This adds, next to the
UNCHANGED raw fields, `failure.<flag>_corrected` plus a `post_run_corrections` entry carrying the evidence, computed from the raw
server log and memory samples (local, gitignored; their sha256 is recorded). Refuses to run twice or when a raw field is not the
recorded False.

`--minimize-reference`: artifact minimisation. Committed results keep the validation-row identifiers and `q_in_sha256` (sha256 of the
quantized input as little-endian int64) but not the quantized input vector itself.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any


from src.analysis.phase10_results import q_in_digest
from src.config import PROJECT_ROOT, load_config

MINIMIZATION_NOTE = (
    "Artifact minimisation: the quantized input vector (q_in) of each validation row was replaced by q_in_sha256 (sha256 over the vector as "
    "little-endian int64) and q_in_length. The row identifiers, clear/simulated outputs and probabilities are unchanged."
)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str), encoding="utf-8")


def minimize_reference(config: dict[str, Any]) -> list[str]:
    changed = []
    for model in config["models"]:
        path = PROJECT_ROOT / config["output"]["results_dir"] / "deploy" / model["id"] / "reference.json"
        reference = json.loads(path.read_text(encoding="utf-8"))
        touched = False
        for row in reference["rows"].values():
            if "q_in" in row:
                row["q_in_sha256"] = q_in_digest(row["q_in"])
                row["q_in_length"] = len(row["q_in"])
                del row["q_in"]
                touched = True
        if touched:
            reference["artifact_minimization"] = MINIMIZATION_NOTE
            _write_json(path, reference)
            changed.append(str(path.relative_to(PROJECT_ROOT)))
    return changed


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _last_run_samples(path: Path) -> list[dict[str, Any]]:
    """memwatch_serve.jsonl is appended by every serve stage (preflight, then the full run); the last run is the one after the last >30 s gap."""
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    start = 0
    for i in range(1, len(rows)):
        if rows[i]["t"] - rows[i - 1]["t"] > 30:
            start = i
    return rows[start:]


def serve_abort_evidence(logs: Path) -> dict[str, Any]:
    server_log, memwatch = logs / "server.log", logs / "memwatch_serve.jsonl"
    lines = [json.loads(line) for line in server_log.read_text(encoding="utf-8").splitlines() if line.strip()]
    ready = next(line for line in lines if line.get("event") == "server_ready")
    registered = next(line for line in lines if line.get("message") == "Evaluation keys registered")
    t_reg = dt.datetime.fromisoformat(registered["timestamp"]).timestamp()
    rows = _last_run_samples(memwatch)
    server = [r["procs"]["server"]["rss_mb"] for r in rows if "server" in r["procs"]]
    sender = [r["procs"]["sender"]["rss_mb"] for r in rows if "sender" in r["procs"]]
    after = [r for r in rows if r["t"] >= t_reg + 1.0 and "server" in r["procs"]]  # past the request-body buffer release that follows registration
    before = [r for r in rows if r["t"] < t_reg and "server" in r["procs"]]
    last = rows[-1]
    before_end = [r for r in rows if r["t"] <= last["t"] - 5 and "server" in r["procs"]][-1]
    return {
        "raw_evidence_files": {"server_log_sha256": _sha256(server_log), "memwatch_serve_sha256": _sha256(memwatch), "location": "data/phase10/logs/<model>/ (local, gitignored)"},
        "server_ready_event": True,
        "server_ready_line": ready,
        "evaluation_keys_registered_bytes": registered["n_bytes"],
        "evaluation_keys_registered_line": registered,
        "seconds_from_key_registration_to_last_sample": round(last["t"] - t_reg, 3),
        "last_sample_utc": dt.datetime.fromtimestamp(last["t"], dt.timezone.utc).isoformat(),
        "samples_in_run": len(rows),
        "server_rss_mb": {
            "peak_reported_by_watch": max(r["procs"]["server"]["peak_rss_mb"] for r in rows if "server" in r["procs"]),
            "max_sampled": max(server),
            "at_last_sample": last["procs"]["server"]["rss_mb"],
            "five_seconds_before_last_sample": before_end["procs"]["server"]["rss_mb"],
            "max_before_registration_line": max(r["procs"]["server"]["rss_mb"] for r in before),
            "min_after_registration_plus_1s": min(r["procs"]["server"]["rss_mb"] for r in after),
            "at_last_sample_minus_min_after_registration": round(last["procs"]["server"]["rss_mb"] - min(r["procs"]["server"]["rss_mb"] for r in after), 1),
        },
        "sender_rss_mb": {"peak_reported_by_watch": max(r["procs"]["sender"]["peak_rss_mb"] for r in rows if "sender" in r["procs"]), "max_sampled": max(sender), "at_last_sample": last["procs"]["sender"]["rss_mb"]},
        "mem_available_mb": {"min_sampled": min(r["mem_available_mb"] for r in rows), "at_last_sample": last["mem_available_mb"], "five_seconds_before_last_sample": before_end["mem_available_mb"]},
        "swap_used_mb": {"max_sampled": max(r["swap_used_mb"] for r in rows)},
        "scope_note": (
            "In the recorded stages.serve.parent_watch, per_process.stage_serve is the sender/stage process (its 923 MB peak equals the sender's peak here), "
            "not the server; the server was watched by the stage's own memory watch (memwatch_serve.jsonl)."
        ),
    }


def correct_serve_abort(config: dict[str, Any], model_id: str) -> dict[str, Any]:
    path = PROJECT_ROOT / config["output"]["results_dir"] / model_id / "metrics.json"
    metrics = json.loads(path.read_text(encoding="utf-8"))
    failure = metrics.get("failure") or {}
    if metrics.get("post_run_corrections"):
        raise SystemExit(f"{model_id}: already corrected; refusing to add a second correction")
    if failure.get("stage") != "serve" or failure.get("server_started") is not False or failure.get("fhe_execution_started") is not False:
        raise SystemExit(f"{model_id}: the raw failure record is not the expected serve-stage False/False; nothing to correct")
    ev = serve_abort_evidence(PROJECT_ROOT / config["output"]["logs_dir"] / model_id)
    failure["server_started_corrected"] = True
    failure["fhe_execution_started_corrected"] = True
    seconds = ev["seconds_from_key_registration_to_last_sample"]
    growth = ev["server_rss_mb"]["at_last_sample_minus_min_after_registration"]
    metrics["post_run_corrections"] = [
        {
            "field": "failure.server_started",
            "recorded_value": False,
            "corrected_value": True,
            "basis": "observed",
            "reason": (
                "The recorded boolean was derived from the presence of a completed serve-stage state record (`server_startup`). The stage was killed by the "
                "outside memory watchdog before writing that record, so the derivation defaulted to False. The raw server log shows the server reached "
                "server_ready and registered all evaluation-key bytes."
            ),
            "evidence": {k: ev[k] for k in ("server_ready_event", "server_ready_line", "evaluation_keys_registered_bytes", "evaluation_keys_registered_line", "raw_evidence_files")},
        },
        {
            "field": "failure.fhe_execution_started",
            "recorded_value": False,
            "corrected_value": True,
            "basis": "inferred",
            "reason": (
                "Same derivation defect (the flag is read from the state record that was never written), so False is a default, not an observation. It is INFERRED, not "
                f"directly observed, that the encrypted request had been sent: the last watchdog sample, at which the stage was killed, is {seconds} s after keys were registered, whereas a go/no-go "
                "refusal or a preflight-only run ends the stage within a few seconds of registration (by the stage_serve code path: a 0.2 s pause, then key deletion or the request), and the server's RSS at the last sample was "
                f"{growth} MB above its lowest value from one second after registration onward, with no other client action recorded. No response, decryption or result exists, so nothing was evaluated to completion."
            ),
            "evidence": {k: ev[k] for k in ("seconds_from_key_registration_to_last_sample", "last_sample_utc", "samples_in_run", "server_rss_mb", "sender_rss_mb", "mem_available_mb", "swap_used_mb", "scope_note", "raw_evidence_files")},
        },
    ]
    metrics["post_run_corrections_note"] = (
        "All fields outside `post_run_corrections` and the `*_corrected` flags are the raw record exactly as written by the run. Corrections were added after the run from "
        "the raw server log and memory samples; the experiment was not re-run."
    )
    _write_json(path, metrics)
    return ev


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/phase10/client_server.yaml")
    parser.add_argument("--minimize-reference", action="store_true")
    parser.add_argument("--correct-serve-abort", metavar="MODEL_ID")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.minimize_reference:
        print("minimised:", minimize_reference(config) or "nothing to do")
    if args.correct_serve_abort:
        print(json.dumps(correct_serve_abort(config, args.correct_serve_abort)["server_rss_mb"], indent=2))


if __name__ == "__main__":
    main()
