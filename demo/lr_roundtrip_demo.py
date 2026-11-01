"""Narrated LR encrypted round trip for the portfolio recording (presentation only; not part of the research pipeline).

Drives the EXISTING implementation unchanged: `src.client.client.CiphraudClient` (keys, quantize, encrypt, decrypt),
`src.client.http.ApiClient` (HTTP) and the real `python -m src.server` FastAPI process, using the already-deployed
`lr_top20_bits16` artifacts and the recorded Phase 10 reference integers. Everything printed is measured from this run.
Nothing is written under results/ or src/; the server log goes to the directory given by --scratch.

--pace SECONDS inserts a pause before each section header and between the "sent" and "server evaluation" lines, so a
viewer can follow each step (the encrypted steps otherwise finish within ~0.1 s and print together). The pauses sit
outside every timed operation, so the timings printed are unaffected; the run says so in its first lines.

Run (WSL, FHE venv, repository root):
    python -m demo.lr_roundtrip_demo --scratch /tmp/ciphraud_demo [--pace 1.5]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

from src.benchmark.phase10_roundtrip import ServerProcess, WireAudit, compare_integers, free_port, read_raw_rows
from src.config import PROJECT_ROOT, load_config
from src.fhe.deploy import paths_for, sha256_bytes

CONFIG = "configs/phase10/client_server.yaml"
MODEL = "lr_top20_bits16"
PACE = 0.0  # seconds; set from --pace


def say(msg: str = "") -> None:
    print(msg, flush=True)


def pause() -> None:
    if PACE > 0:
        time.sleep(PACE)


def step(title: str) -> None:
    pause()
    say()
    say(f"== {title}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scratch", required=True, help="directory for the server log (outside the repository results)")
    parser.add_argument("--pace", type=float, default=0.0, help="presentation pause in seconds between sections (default 0: none)")
    args = parser.parse_args()
    global PACE
    PACE = max(0.0, args.pace)
    scratch = Path(args.scratch)
    scratch.mkdir(parents=True, exist_ok=True)

    from src.client.client import CiphraudClient
    from src.client.http import ApiClient

    config = load_config(CONFIG)
    paths = paths_for(config, MODEL)
    manifest = json.loads(paths["manifest"].read_text())
    reference = json.loads(paths["reference"].read_text())
    position = int(reference["fixed_position"])

    say("Ciphraud - encrypted fraud scoring, logistic regression (20 features, 16-bit), one transaction")
    say("Client and server are two separate processes talking over loopback HTTP.")
    if PACE > 0:
        say(f"Paced for viewing: {PACE:g} s pauses are added between sections, outside every timed operation.")

    step("1. Start the inference server (holds the compiled model, never a secret key)")
    port = free_port()
    server = ServerProcess(CONFIG, MODEL, port, scratch / "server.log")
    try:
        ready = server.wait_ready(float(config["server"]["startup_timeout_seconds"]))
        say(f"server pid {server.pid} listening on {server.url}; model {ready['models'][0]['id'] if 'id' in ready['models'][0] else MODEL} loaded")

        step("2. Client side: load client artifact, generate keys")
        client = CiphraudClient(paths["client_dir"])
        served = ready["models"][0]
        say(f"client artifact sha256 matches the served circuit: {client.client_zip_sha256 == served['client_zip_sha256']}")
        keygen = client.generate_keys()
        say(f"secret + evaluation keys generated on the CLIENT in {keygen['keygen_seconds']:.2f} s (secret key stays in client memory)")
        ek = client.evaluation_keys()
        say(f"evaluation (public) keys: {ek['n_bytes']:,} bytes, sha256 {ek['sha256'][:16]}...")

        audit = WireAudit()
        api = ApiClient(server.url, request_hook=audit.hook)
        uploaded = api.register_keys(MODEL, ek["bytes"], client.client_zip_sha256)
        say(f"uploaded evaluation keys only; server key_id {uploaded['json']['key_id'][:16]}... (server never receives the secret key)")

        step("3. Client side: preprocess, quantize, encrypt one transaction")
        raw = read_raw_rows(manifest, [position])[position]
        say(f"plaintext transaction: validation row {position}, {raw.shape[1]} raw features (stays on the client)")
        prepared = client.prepare_request(raw)
        say(f"quantized input integers (client-only): {np.asarray(prepared['q_in']).reshape(-1).tolist()}")
        say(f"encrypted in {prepared['timings']['encrypt_seconds'] * 1000:.1f} ms -> ciphertext {prepared['n_bytes']:,} bytes, sha256 {sha256_bytes(prepared['ciphertext'])[:16]}...")

        step("4. Send ciphertext; the server computes on it without decrypting")
        sent = api.infer(MODEL, uploaded["json"]["key_id"], prepared["ciphertext"])
        st = sent["server_timing"]
        say(f"request body {sent['request_bytes']:,} bytes (ciphertext only), response {sent['response_bytes']:,} bytes (still encrypted)")
        pause()
        say(f"server FHE evaluation: {st['fhe_run_s'] * 1000:.1f} ms (this is the first, cold request on this server)")

        step("5. Client side: decrypt and decide")
        finished = client.finish(sent["content"])
        ints = np.asarray(finished["integers"]).reshape(-1).astype(np.int64).tolist()
        say(f"decrypted integer(s): {ints}")
        say(f"fraud probability {finished['probability']:.6f}, threshold {reference['threshold']:.6f} -> decision: {'FRAUD' if finished['decision'] else 'not fraud'}")

        step("6. Check against the plaintext-quantized model and the recorded Phase 10 reference")
        ref_row = reference["rows"][str(position)]
        cmp = compare_integers(finished["integers"], ref_row)
        say(f"reference integers (clear quantized model): {ref_row['clear_ints']}")
        say(f"equals clear-quantized model:      {cmp['equals_clear_quantized']}")
        say(f"equals Concrete simulation:        {cmp['equals_simulate']}")
        say(f"equals recorded source-phase value: {cmp['equals_recorded_source_phase']}")
        say(f"EXACT INTEGER AGREEMENT: {cmp['exact_match']}")

        step("7. What crossed the wire (client-side audit of every request body)")
        classes = audit.classify(ek["sha256"], {sha256_bytes(prepared["ciphertext"])})
        for entry, cls in zip(audit.entries, classes["classes"]):
            say(f"{entry['method']:6s} {entry['path']:<28s} {entry['n_bytes']:>12,} bytes  {cls}")
        say(f"every body is evaluation keys or an input ciphertext: {classes['all_bodies_are_evaluation_keys_or_ciphertexts']}")

        step("8. Cleanup")
        api.delete_keys(uploaded["json"]["key_id"])
        say(f"server keys deleted; key still resident on server: {api.models()['key_resident'] is not None}")
        api.close()
        say()
        say("One request, one fixed validation row, loopback. A functional demonstration, not a benchmark.")
        return 0 if cmp["exact_match"] else 1
    finally:
        rc = server.stop()
        say(f"server stopped (exit code {rc})")


if __name__ == "__main__":
    sys.exit(main())
