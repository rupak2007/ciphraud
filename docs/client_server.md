# Phase 10 — Client/Server Encrypted Inference

Status: **complete: the formal exit criterion is satisfied, with the caveats below.** Real two-process HTTP
encrypted inference is demonstrated for all three model types: LR (7/7 exact-match requests), MLP (7/7) and
XGBoost (`xgboost_top50_bits14`, 1/1: all 219 decrypted output integers equal the clear-quantized, simulated and
Phase 8 recorded values). The XGBoost request needed three supervised local attempts: attempts 1 and 2 ended
`aborted_memory` under the pre-registered guard on the default ~3.9 GiB WSL memory limit, and attempt 3 **passed**
after the WSL memory limit was raised to 5 GB on the same computer (§7, §8). Caveats: the MLP configuration
failed the Phase 9 quantization-accuracy gate (§4); the XGBoost result is a single cold request (n=1), is not
a benchmark latency, and its durations come from a VM clock that ran about 8% slower than real time (§5); and the XGBoost pass depends on an environment setting (the WSL memory limit), not on
any change to the model, circuit, protocol, guards or gates. Phase 10 is an *integration* phase: nothing was
retrained, and no Phase 8/9 result, gate, bit-width, calibration, memory guard or the 2.4 GB key-material gate was
changed.

## 1. Architecture

```
CLIENT process (src/client)                     SERVER process (src/server, FastAPI/uvicorn)
 raw row -> preprocess -> quantize -> encrypt     holds server.zip only (sha256-verified)
 secret key stays in client memory                holds evaluation keys in memory only
        |  POST /v1/keys   (evaluation keys)  --->  deserialize, keep one key set
        |  POST /v1/infer  (input ciphertext) --->  FHEModelServer.run
        |  <--- encrypted result + X-Ciphraud-Timing header
 decrypt -> dequantize -> probability -> decision
        |  DELETE /v1/keys ---> key set dropped
```

Deployment persistence: `src/fhe/deploy.py` rebuilds each already-validated circuit through the
existing Phase 8/9 loaders, proves it reproduces the source phase (circuit stats == source
metrics; clear == simulate == recorded ints; PR-AUC identical), then saves with
`FHEModelDev.save(via_mlir=False)` and splits it into `server/server.zip` and `client/client.zip`
under `data/phase10/deploy/` (gitignored). The server reads only `server.zip`. The three deployments were
re-verified in fresh processes (`deploy --verify`) before the final runs.

## 2. What stays client-side / what is sent

Client-side only: plaintext feature row, preprocessing and quantization, the secret key,
decryption, probability and decision. Sent to the server: (a) the serialized **evaluation keys**,
(b) the input **ciphertext**, (c) a client-artifact sha256 header used to bind the keys to the
circuit. Returned: the encrypted result plus a timing header. Tests enforce this statically (no
client-side crypto identifiers in `src/server/**`) and dynamically (every captured request body is
evaluation-key bytes or an input ciphertext; `wire_audit.all_bodies_are_evaluation_keys_or_ciphertexts`
is true for both recorded runs).

## 3. Endpoints (deviation from `architecture.md` §18 recorded there)

| Endpoint | Purpose | Errors |
|---|---|---|
| `GET /v1/models` | model discovery and readiness | none |
| `POST /v1/keys?model=` | register evaluation keys (octet-stream) | 404, 409 (key resident), 413, 415, 400, 422 (client-artifact mismatch) |
| `POST /v1/infer?model=&key_id=` | one encrypted inference | 404, 415, 400, 422, 503 |
| `DELETE /v1/keys?key_id=` | drop the key set | 204, 404, 409 (busy) |

No endpoint accepts plaintext features. One resident key set at a time (memory safety).

## 4. Models

| Model | Config | Basis |
|---|---|---|
| LR | `lr_top20_bits16` | passes Phase 8 T0-T3 |
| MLP | `mlp_top20_bits3` | see disclosure |
| XGBoost | `xgboost_top50_bits14` | passes Phase 8 T0-T3; keys 1.93 GB |

**MLP disclosure (verbatim):** The client/server FHE deployment was demonstrated using
mlp_top20_bits3, which passed the relevant execution/integrity checks but failed the Phase 9
quantization-accuracy gate. It is not presented as a validated or production fraud model.

## 5. End-to-end results (loopback HTTP, one machine, WSL2, 6 vCPU; WSL limit ~3.9 GiB for LR/MLP and XGBoost attempts 1-2, 5GB for XGBoost attempt 3)

Current official records (`results/phase10_client_server/<model>/metrics.json`, produced by the final code):

| | LR | MLP | XGBoost |
|---|---|---|---|
| Status | passed | passed | **passed** (attempt 3; attempts 1-2 were `aborted_memory`) |
| Requests | 7 (row 32,148 x6 + row 13,560) | 7 (same) | 1 (row 32,148, staged client) |
| Decrypted == clear-quantized == recorded ints | 7/7 | 7/7 | 1/1 (219 integers equal clear, simulated and Phase 8 recorded) |
| Decisions match reference | yes | yes | yes (probability 0.0999311, abs diff 0.0, decision False; the row is legitimate, y=0) |
| Evaluation keys uploaded | 64 B | 115,553,032 B | 308,396,952 B |
| Request / response body | 744 B / 17,920 B | 744 B / 65,816 B | 1,464 B / 3,590,120 B |
| Server FHE run, fixed-row mean (6 requests) | 0.075 s (std 0.109; request 0: 0.297 s) | 16.17 s (std 2.24; request 0: 20.31 s) | 2,445.06 s on the guest's monotonic clock (about 2,650 s of real time; one cold request) |
| HTTP round trip, mean | 0.088 s | 16.19 s | 2,445.18 s |
| Transport + framework, mean | 0.013 s | 0.016 s | 0.109 s |
| Peak client RSS | 644 MB | 1,106 MB | 1,871 MB (keys stage), 1,207 MB (decrypt stage) |
| Peak server RSS | 644 MB | 1,468 MB | 3,112 MB (sender process 351 MB) |
| Min MemAvailable / max swap | 2,667 MB / 0.0 MB | 1,622 MB / 0.1 MB | 1,263 MB / 0.3 MB (serve stage; guard not tripped) |

LR/MLP one-time costs: MLP keygen 3.80 s, key serialize 1.90 s, upload 4.55 s, server deserialize 2.21 s,
key delete 0.78 s (LR: all under 0.21 s). Request 0 pays warm-up costs and is reported separately in each
`metrics.json`. Repeats on the fixed row are an integration check, not independent samples. The per-request
decomposition (client preprocess / quantize / encrypt / serialize; HTTP round trip; server input-deserialize /
**fhe_run** / output-serialize / handler total; client deserialize / decrypt / postprocess) is in each
`metrics.json`; it is deliberately not merged into one "FHE latency", and only `server_fhe_run_s` is comparable to
Phase 8/9 latency. Timings are loopback only, not WAN.

**XGBoost (attempt 3) decomposition.** Keys stage (client process, 22.5 s wall): keygen 5.21 s, evaluation-key
serialize 2.34 s, keyset save 2.09 s. Serve stage: server ready in 5.9 s, key upload 6.10 s over HTTP, server key
deserialize 3.84 s, go/no-go GO (3,344 MB available, swap 0), **server FHE run 2,445.06 s**, output serialize
0.014 s, key delete 2.05 s. Decrypt stage (separate client process, 11.1 s wall): keyset load 3.35 s, decrypt
0.0215 s, post-process 0.0009 s. The served key bytes and input ciphertext hashes in the record equal those of the
staged files (checked before they were removed).
**Clock caveat (measured).** Every recorded duration comes from the VM's monotonic clock (`perf_counter`). During this run
that clock ran about **8% slower than real time**: the raw 0.1 s memory samples contain 78 forward steps of the real-time
clock of about 3 s each (217 s in total), and the real time from key registration to the end of the serve stage was
2,649.8 s against 2,447.2 s measured monotonically (a 202.5 s difference). The guest's real-time clock agrees with the host
(launched about 18:05 IST, finished about 18:50 IST, about 45 minutes in all). So the 2,445.06 s evaluation figure is
correct as a record of the guest's monotonic clock, and the real elapsed time was about 2,650 s. The same effect shows up in
every recorded run (steps were 10% of the MLP run's span, 22% of XGBoost attempt 2's); all durations in this phase,
including LR and MLP, are on the same lagging clock. Its cause was not investigated, and the Phase 8/9 timings were taken in the same
kind of WSL VM. (Correction added after Phase 12: Phase 8 had already measured the same effect with a sleep probe, about 10-11% slow, and a 5-11% gap in checkpoint timestamps; see `docs/research.md` section 7.5 and `report.md` section 12. Phase 9 was not separately probed.) Details: `environment.json` -> `clock_note`.
**Comparability.** Phase 8's recorded in-process evaluation for this circuit was 2,650.6 s (std 771.9 s, n=5) on the same CPU.
This run is a single cold request through the client/server stack with the WSL limit at 5GB and a Windows background download
running for a few minutes of it. The two numbers sit in the same range (and the real-time figure, about 2,650 s, happens to
equal the Phase 8 mean), but this is not a benchmark and not a speed-up claim; Phase 8 holds the latency figures. Hardware and
settings are recorded in `xgboost_top50_bits14/environment.json`.

**Timing is not stable on this VM.** LR and MLP were recorded twice on the same code path (the first recording
predates the sender fix and is archived as `superseded_pre_sender_fix/`; the sender stage is not used by the
in-process LR/MLP runs). The integers, probabilities and decisions are identical. The timings are not: MLP server
FHE mean 7.20 s (first) vs 16.17 s (final), LR 0.029 s vs 0.075 s, MLP keygen 2.31 s vs 3.80 s. The final
record is the official one and the first is kept, neither is selected for being faster. The cause was not
investigated; Phase 9's recorded MLP latency is 5.17 s and `api_probe.json` (`run_time_ab`) already showed
run-to-run variation. Do not use these numbers as benchmark latencies; Phase 8/9 hold those.

## 6. Findings from the API probe (`results/phase10_client_server/api_probe.json`)

- Default `keygen` is randomized across processes; fixed seeds are deterministic. The client uses explicit random seeds.
- Concrete's `encrypt` silently generates keys if none were loaded; the client raises `ClientKeyError` instead.
- Evaluation keys from a different circuit crash the native runtime; keys are therefore bound to `client.zip` by sha256 (422 on mismatch).
- Running a circuit in-process makes the Python process exit with status 0 and lose unflushed stdout; loading, keygen and encrypt do not. Scripts flush and `os._exit`.
- Native calls hold the GIL, so the memory watchdog runs in the parent process.

## 7. Memory supervision

External fail-closed `MemWatch` (0.1 s sampling) with pre-registered guards: abort below 250 MB
MemAvailable or above 700 MB swap; the go/no-go for the FHE run requires at least 900 MB available
and at most 256 MB swap; run timeout 10,800 s. The guards were fixed in `configs/phase10/client_server.yaml` before
the first XGBoost attempt and were not changed for the second.

**Sender-memory fix (the only source change between the attempts).** The XGBoost serve stage is a key-less HTTP
sender, but it imported `src.fhe.xgb_poc` only to call `peak_rss_mb`, which loads Concrete-ML and torch. Measured:
that one import raises the process from 38 MB to 618 MB. `stage_serve` now uses a stdlib `resource.getrusage`
helper with the same formula (`_sender_peak_rss_mb`), and a test asserts that the sender stage imports neither
torch nor concrete. Model, bit-width, circuit, protocol, API, guards, acceptance criteria and all Phase 8/9
material are unchanged.

**WSL memory limit (an environment setting, not a methodology change).** Attempts 1 and 2 ran under WSL's default
cap (50% of the 7.95 GB host; `MemTotal` read 3,979,384-3,982,148 kB across boots, about 3.9 GiB). Attempt 3 ran after `C:\Users\Rupak\.wslconfig`
was created with `memory=5GB` and `swap=1GB` (`MemTotal` 5,038,916 kB, about 4.8 GiB, swap 1 GiB) and after
unneeded Windows applications were closed. The guards, thresholds and the swap size were not changed. To revert:
delete `C:\Users\Rupak\.wslconfig` and run `wsl --shutdown`.

## 8. XGBoost memory preflight and attempts

`xgboost_top50_bits14` key material: bootstrap keys 1,795,547,136 B + keyswitch keys 135,073,792 B (1.93 GB
full size; 308,396,952 B serialized). Phase 8 recorded a single-process peak of 3,356 MB for this configuration.

**Preflight-only, before and after the sender fix** (`--staged --preflight-only`; the keys are registered and
then deleted, no encrypted request is sent):

| | before (attempt 1's preflight) | after (attempt 2's preflight) |
|---|---|---|
| Sender baseline RSS | about 620 MB | 43 MB |
| Sender peak RSS | 922 MB | 346 MB |
| Server peak RSS | 2,103 MB | 2,108 MB |
| Minimum MemAvailable | 917 MB | 1,273 MB |
| Max swap | 3.8 MB | 0.06 MB |
| Go/no-go (needs >= 900 MB, <= 256 MB swap) | GO (2,037 MB) | GO (2,398 MB) |
| Key upload / server deserialize | 12.4 s / 6.9 s | 20.2 s / 11.5 s |

**Attempt 1 (`attempt_1_aborted_memory/`).** Keys stage succeeded (client peak 1,872 MB, keygen 5.61 s,
serialize 11.74 s). The server started and registered all 308,396,952 B of keys; 15.6 s later the watchdog killed
the stage at MemAvailable 246 MB < 250 MB (exit -9, peak swap 61.6 MB). Server RSS peaked at 2,733 MB and the
sender at 923 MB (620 MB of that baseline was the avoidable import). No result was returned.

**Attempt 2 (`attempt_2_aborted_memory/`, after the sender fix).** Keys stage succeeded (client peak 1,871 MB,
keygen 8.16 s, serialize 12.02 s). The server started (`server_ready`) and registered all 308,396,952 B. The
stage then ran for **1,549 s (25.8 min)** after registration. Server RSS reached 2,846-2,891 MB within 30-60 s,
then stayed between 2,759 and 2,891 MB for the rest of the run, with MemAvailable between 457 and 519 MB and swap
about 5 MB. About 0.3 s before the end, server RSS rose from 2,759 MB (five seconds earlier) to 2,975 MB and
MemAvailable fell from 469 MB to **249 MB, 1 MB below the 250 MB floor**; the watchdog killed the stage (exit -9,
wall 1,507.6 s). Watch-reported server peak 3,009 MB, sender peak 347 MB (338 MB at the end), max swap 5.2 MB.
No response, decryption or result exists and the server logged no completed inference. What the final +216 MB
step was is **not established** (the evaluation may have been close to finishing or entering a later phase);
this document does not explain it.

**Attempt 3 (current record, `xgboost_top50_bits14/metrics.json`; passed).** Launched 2026-10-11 12:35:37 UTC
(18:05 IST) with WSL at 5GB, after a preflight in which every check passed. Keys stage succeeded (client peak
1,871 MB). The server started and registered all 308,396,952 B; the go/no-go said GO with 3,344 MB available. The
encrypted evaluation then ran **2,445 s** (guest monotonic clock; about 2,650 s of real time, §5) and returned a 3,590,120 B encrypted result, which the separate decrypt
stage decrypted to 219 integers equal to the clear-quantized, simulated and Phase 8 recorded values; the
probability equals the reference exactly and the decision is correct. Server RSS rose to 2,896 MB within the first
minute, stayed at about 2.9-2.95 GB, and peaked at 3,112 MB; the sender stayed at 345-351 MB; WSL MemAvailable never
fell below **1,263 MB** (floor 250 MB) and swap peaked at 0.3 MB; the watchdog never tripped. The late
+216 MB step seen in attempt 2 was not a problem with the extra headroom (its cause is still not explained here).
The staged secret keyset was deleted by the decrypt stage, and the staged public evaluation keys, input
ciphertext and encrypted response were removed afterwards (their hashes are in `environment.json`). A Windows background download
(`BackgroundDownload.exe` under `svchost`; its purpose was not verified, likely an update) started at about 18:38 IST and made the host page heavily for about 3 minutes (VM footprint on the host
dropped from 3.8 to 1.8 GB); the run's own memory and result were unaffected. The record is fully observed
(`fhe_execution_started: true` came from a completed serve stage; no correction was needed).

**Server vs sender memory.** In attempt 1's recorded `stages.serve.parent_watch.per_process.stage_serve` the
923 MB is the sender/stage process, not the server; the server was watched by the stage's own memory watch
(`memwatch_serve.jsonl`). An earlier draft of this document wrongly reported the attempt-1 server as "at most
923 MB". The preflight's server peak (2.1 GB) is the upload/deserialization transient; the run's ~2.8-3.0 GB
appears after registration, consistent with (not proof of) the serialized keys being expanded to their full
1.93 GB when the first evaluation starts. The native code was not inspected.

**Derived flags and corrections (§12).** In both attempts `failure.server_started: false` and
`failure.fhe_execution_started: false` were defaults from a state file the killed stage never wrote. The server
did start (observed). That the encrypted request had been sent is **inferred, not observed**: the stage was
alive 15.6 s (attempt 1) and 1,549 s (attempt 2) after registration, whereas a go/no-go refusal or a
preflight-only run ends the stage within seconds. Nothing was evaluated to completion.

**Interpretation.** Attempts 1 and 2 ended `aborted_memory`: the configured guard terminated the run before an
encrypted result was returned, on the default ~3.9 GiB WSL limit. Attempt 3 on the same computer with the WSL limit at
5GB completed the same request with the same model, circuit, guards and code, and its result is exact. So the
earlier aborts were a memory limit of that environment, not a property of the model or of the protocol, and the
failed attempts stay recorded (archived) next to the pass. The guards, model, bit-width and the 2.4 GB gate were not
changed in any attempt, the configuration was not deleted or skipped, and no OOM was forced. This is one request
(n=1); it shows the stack works for XGBoost, it does not measure its latency or its variance.

## 9. Security claim boundary

Only this is claimed: **The application protocol does not transmit the client's secret key or
plaintext feature vector to the inference server.** Not claimed: OS-level isolation (same host and
user), TLS or authentication (**not implemented**; plain HTTP on loopback), integrity against a
malicious server, timing or size side channels, model confidentiality. Metadata the server sees:
request timing and sizes, a persistent `key_id` linking requests, and key/ciphertext sizes that
fingerprint the model. Evidence: wire audit true for LR and MLP (every request body is evaluation keys or an
input ciphertext); `secret_key_sent: false` in every record; the staged client keyset file was deleted after
all three XGBoost attempts and no key material remains under `data/phase10` (the XGBoost wire audit is also clean:
one evaluation-key upload and one 1,464 B ciphertext); `src/server/**` contains no client-side
crypto identifiers (static test).

## 10. What Phase 10 proves / does not prove

Proves: the existing validated LR and MLP circuits execute correctly across a process and HTTP
boundary with only evaluation keys and ciphertexts on the wire, with results exactly equal to the
clear-quantized and recorded integers; serialization and transport overhead is small relative to FHE
time; the deployment artifacts reproduce their source phases; for XGBoost, one full client/server encrypted
inference completes with all 219 decrypted output integers exactly equal to the clear-quantized, simulated and
Phase 8 recorded values.

Does not prove: XGBoost latency or its variance (one cold request); WAN behavior or throughput; TLS or
authentication; a validated MLP fraud model (T3 fails); security beyond §9; stable latencies (§5); that XGBoost
fits in less memory than the 5GB WSL limit used for attempt 3 (it failed twice under about 3.9 GiB).

**Exit criterion: SATISFIED.** `docs/plan.md` Phase 10 requires "at least one successful end-to-end
encrypted request per model type"; `docs/prd.md` §13 requires "at least one working encrypted inference path per
model type". LR: 7/7 exact. MLP: 7/7 exact, as an execution/integrity result, with the Phase 9 quantization-accuracy
failure disclosed (§4). XGBoost: 1/1 exact in attempt 3. The criterion was not redefined: it is met by actual
requests that completed and matched the validated references, and the two earlier aborted attempts remain in the
record. What is NOT claimed: that the XGBoost run is reproducible on less memory; any latency conclusion; any
security beyond §9.

## 11. Reproduce

All commands run inside the WSL venv (`~/.venvs/fhe-fraud-detection`), from the repository root. They need
`requirements-fhe.txt` (includes the pinned `fastapi`/`uvicorn` stack) and the Phase 6/8/9 handoff artifacts under
`data/fhe_handoff/`.

```
# 1. Deploy (rebuild the validated circuit, check it against the recorded Phase 8/9 integers, save server.zip/client.zip + manifest)
python -m src.fhe.deploy --config configs/phase10/client_server.yaml --model lr_top20_bits16 --force
python -m src.fhe.deploy --config configs/phase10/client_server.yaml --model mlp_top20_bits3 --force
python -m src.fhe.deploy --config configs/phase10/client_server.yaml --model xgboost_top50_bits14 --force
python -m src.analysis.phase10_correct_record --minimize-reference     # before committing regenerated deploy artifacts

# 2. Round trips (client and server as separate processes over loopback HTTP)
python -m src.benchmark.phase10_roundtrip --config configs/phase10/client_server.yaml --model lr_top20_bits16
python -m src.benchmark.phase10_roundtrip --config configs/phase10/client_server.yaml --model mlp_top20_bits3

# 3. XGBoost: staged three-process flow (keys -> serve -> decrypt); about 45 minutes of real time for the single request
python -m src.benchmark.phase10_roundtrip --config configs/phase10/client_server.yaml --model xgboost_top50_bits14 --staged --preflight-only   # optional: stop at the memory go/no-go
python -m src.benchmark.phase10_roundtrip --config configs/phase10/client_server.yaml --model xgboost_top50_bits14 --staged

# 4. Validate the evidence against the live code and the archives
python -m src.analysis.phase10_results --check-code-hashes --summary
```

**XGBoost memory requirement.** The staged run needs a WSL VM of about 5 GB (`%UserProfile%\.wslconfig` with `[wsl2]`,
`memory=5GB`, `swap=1GB`, then `wsl --shutdown`). With the default cap of about 3.9 GiB two attempts aborted on the
pre-registered memory guards (section 8). The guards in `configs/phase10/client_server.yaml` must not be loosened to force a
run. Close other memory-heavy programs first and check that `MemAvailable` is at least about 4 GB before starting; the run
stops itself if a guard trips. The pre-run state of the passing attempt is in
`results/phase10_client_server/xgboost_top50_bits14/preflight_snapshot.json` (copied unchanged, with its original
modification time, from the local `data/phase10/logs/`).

**Tests.** `python -m pytest tests/test_phase10_*.py` (100 tests; the real-model round trips skip when the artifacts are
absent). In WSL, run the full suite with `--ignore-glob="*real_data*"`.

Provenance records per-file source hashes plus the base commit `afb09da` (no pre-run commit was made), so
`--check-code-hashes` matches only while the Phase 10 source files are unchanged.

## 12. Post-run corrections, archives and artifact minimisation

Applied after the runs, without re-running any experiment for the correction itself, by
`src/analysis/phase10_correct_record.py`.

- **Derived-flag correction.** In both XGBoost attempt records the raw fields are unchanged. Beside them:
  `failure.server_started_corrected: true`, `failure.fhe_execution_started_corrected: true` and a
  `post_run_corrections` list (per flag: recorded value, corrected value, `basis` = `observed` / `inferred`,
  reason, evidence with the raw server-log lines, memory statistics and the sha256 of the raw logs). The validator
  (`src/analysis/phase10_results.py`) rejects a staged record whose serve stage was killed before writing its
  state, if either derived flag reads `False` without such a correction, rejects a correction that changes the raw
  value or lacks evidence, and cross-checks `server_ready` in the raw server log when it is available. The
  run-time code that derives the flags (`phase10_roundtrip.py`) still yields `False` for a killed stage; the
  validator is what catches it.
- **Archive layout.** `xgboost_top50_bits14/attempt_1_aborted_memory/` and `attempt_2_aborted_memory/` hold each
  attempt verbatim (`metrics.json`, `provenance.json`, `preflight.json`, `preflight_provenance.json`) with its raw
  logs under `raw_logs/` and a `SHA256SUMS.json`; the raw-log hashes equal the ones recorded in the corrections.
  `lr_top20_bits16/superseded_pre_sender_fix/` and `mlp_top20_bits3/superseded_pre_sender_fix/` hold the first LR/MLP
  recordings (recorded against the previous `phase10_roundtrip.py`). The validator checks every archive against
  its `SHA256SUMS.json` and for internal consistency, but not against the live source hashes. A model's current
  record is its top-level `metrics.json`, or else its newest `attempt_N_*`; that record is checked against the live
  code hashes (LR, MLP and XGBoost attempt 2 were all produced by the final code).
- **Attempt 3 evidence.** The current XGBoost record is the top-level `xgboost_top50_bits14/metrics.json`
  (+ `provenance.json`, recorded code and config hashes validated by `phase10_results --check-code-hashes`),
  with `environment.json` (hardware, WSL settings, command, preflight summary, hashes of the staged artifacts and of
  the raw logs) and `serve_memory_timeline_10s.csv` (the serve stage in 10-second windows: minimum MemAvailable, maximum swap and maximum server/sender RSS per window, from the raw 0.1 s log).
  The raw logs (about 11 MB) stay local under the gitignored `data/phase10/logs/xgboost_top50_bits14/`, with their
  sha256 in `environment.json`. Attempts 1 and 2 stay archived under `attempt_1_aborted_memory/` and
  `attempt_2_aborted_memory/`.
- **`q_in` minimisation.** The committed `deploy/*/reference.json` files no longer store the quantized input
  vector of the validation rows (no earlier phase commits such vectors). They keep the row identifiers,
  `q_in_sha256` (sha256 of the vector as little-endian int64) and `q_in_length`. The digest is reproducible from
  the client's own quantized request (test `test_client_quantization_equals_the_compiled_models_exactly`).
  `deploy.py` still writes `q_in` when a deployment is regenerated; run
  `python -m src.analysis.phase10_correct_record --minimize-reference` before committing regenerated artifacts.
  The deployment manifests embed the source hashes as they were at deployment time; `phase10_roundtrip.py`'s hash
  there predates the sender fix, which is factually correct for the deployment step.
