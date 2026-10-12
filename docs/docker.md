# Phase 11 — Container packaging of the inference server

**Status: PARTIALLY VALIDATED (updated 2026-10-12).** The CPU-only server image was built, and one live container round trip with the cheap LR model
passed. Not tested in a container: MLP and XGBoost inference, restart/failure scenarios, TLS or authentication (none exist), network exposure beyond loopback.
Phase 11 is therefore not "complete"; what is established is listed under "Results" below.

## What exists

- `Dockerfile` — server-only image: `python:3.12-slim-bookworm`, `libgomp1`, the pins of `requirements-server-cpu.txt` (the `requirements-fhe.txt` freeze minus the 12 `nvidia-*` wheels, with `torch==2.3.1+cpu`; see the review below), then `src/` and
  `configs/phase10/` only. Runs as non-root user `ciphraud` (uid 10001). `HEALTHCHECK` calls `GET /v1/models`.
  Entrypoint: `python -m src.server --config configs/phase10/client_server.yaml --host 0.0.0.0`; default command `--models lr_top20_bits16`.
- `.dockerignore` — default-deny (`*`), whitelist `Dockerfile`, `requirements-fhe.txt`, `src/`, `configs/phase10/`; key-like files excluded.
  Nothing from `data/`, `results/`, `models/` or the client artifacts can enter a layer.
- `tests/test_phase11_docker.py` — five static tests (what is copied, ignore rules, non-root/pinned Python/healthcheck, every requirement
  pinned, no client code path in `src/server`) pass; one live test needs Docker (see Results).

## Security boundary

Unchanged from Phase 10: the server never receives the secret key or plaintext features. In the container: no keys or client artifacts are baked in;
deployment artifacts are mounted at run time and read-only, **only the `server/` directory (`server.zip`) is mounted, never `client/`**; the root
filesystem can be read-only (`--read-only --tmpfs /tmp`); the port must be published on loopback (`-p 127.0.0.1:8710:8710`) because the server inside listens on
`0.0.0.0` and there is still **no TLS and no authentication**. Do not publish it on a public interface. The security claim is still only the narrow one in `docs/architecture.md` section 10.

## Run recipe

*Audit note 2026-10-12: this recipe has since been executed through the live test (see Results); the tmpfs option below is the corrected `exec` form.*

```
docker build -t ciphraud-server:phase11 .
docker run --rm --read-only --tmpfs /tmp:rw,exec,nosuid,size=512m --memory 3g -p 127.0.0.1:8710:8710 \
  -v "$PWD/data/phase10/deploy/lr_top20_bits16/server:/app/data/phase10/deploy/lr_top20_bits16/server:ro" \
  -v "$PWD/results/phase10_client_server/deploy/lr_top20_bits16:/app/results/phase10_client_server/deploy/lr_top20_bits16:ro" \
  ciphraud-server:phase11 --models lr_top20_bits16
```

Then use the unchanged client from the WSL venv with base URL `http://127.0.0.1:8710`. The live test does exactly this for the cheap LR model only:

```
CIPHRAUD_DOCKER_TEST=1 python -m pytest tests/test_phase11_docker.py -q     # inside the WSL FHE venv, Docker running, image built
```

It checks readiness, error handling (JSON body 415, unknown model 404, bad key id rejected), one real encrypted request whose decrypted integer
equals the recorded clear value, key deletion, and that the container logs contain no secret material. XGBoost is not part of this test and was not containerized.

## Native dependencies and requirements

Concrete-ML 1.9.0 has Linux (and macOS) wheels only; Python 3.12; x86-64 Linux image. The pinned freeze installs torch and the NVIDIA CUDA wheels, so the
image is several GB (size not measured). The server needs `libgomp1` for OpenMP; no other apt package is installed. Memory: an XGBoost deployment needs about 3-5 GB inside the container (Phase 10),
so Docker's VM (WSL2 cap 5 GB on this PC) leaves little margin; LR (about 0.65 GB server RSS) and MLP are the realistic container tests.

## Blocker and storage estimate (measured 2026-10-11; nothing was pulled or built)

*HISTORICAL: written before the build. The Docker engine blocker no longer applies; the actual build and storage numbers are in Results at the end.*

- **Blocker:** the Docker Desktop engine is not running and was deliberately not started; building needs it plus a multi-GB download. Decision recorded: defer until disk space is judged safe.
- **Free space on C:** 16 GB at first check, 47 GB at a later check (93% used). Docker Desktop's own data disk already occupies about 18 GB on C: (`AppData\Local\Docker\wsl\disk`); what it holds cannot be listed while the engine is off.
- **Installed size of the pinned stack**, measured in the existing WSL venv (same `requirements-fhe.txt` versions): `site-packages` 5.9 GB, of which `nvidia` (CUDA wheels pulled in by the pinned torch) 2.8 GB and `torch` 1.6 GB; `xgboost` 0.33 GB, `concrete_python.libs` 0.18 GB.
- **Estimate for the image** (arithmetic, not a build): about 6.1-6.3 GB (that 5.9 GB layer plus a roughly 0.15 GB base image and code). Peak disk during `docker build` is higher because wheels are downloaded (about 3-4 GB, not cached with `PIP_NO_CACHE_DIR`) before the layer is committed; plan for about 10-13 GB of free space in Docker's disk, which only grows (it does not shrink without compaction).
- **Not changed to save space:** the pins (a CPU-only torch would change the binary under the validated circuits, so it is not substituted) and the model configuration.
- **Disposable-candidate inventory (sizes only, nothing deleted):** Windows pip cache about 4.6 GB (`AppData\Local\pip\cache`, re-downloadable); WSL pip cache about 3.1 GB (inside the WSL disk image, so deleting it does not return space to C: without compaction); Windows `AppData\Local\Temp` about 9.2 GB (mixed contents, including this tool's session scratch; not inspected file by file). Project evidence, datasets, model files, logs and `graphify-out` are not candidates.
- **Not yet verified:** that the image builds, that the pins install on `python:3.12-slim-bookworm`, the real image size, healthcheck timing, and that Concrete's runtime works under `--read-only`.

**Status: Phase 11 is partially validated.** It becomes complete only when the live container test passes
(`CIPHRAUD_DOCKER_TEST=1 python -m pytest tests/test_phase11_docker.py`). Decision needed: free space and approve a build, or leave Phase 11 as is.

## Deviations

`docs/architecture.md` section 21 previously listed containerization as optional; it now points here. No Phase 1-10 model, bit-width, gate, memory guard or result was changed.

## Smaller-image feasibility review (2026-10-12; read-only, nothing built, pulled, started or deleted)

*HISTORICAL: estimates only; measured size was 3.33 GB (the 2.5 GB estimate below was too low). The equivalence question was later tested once, for LR only (see Results).*

**Where Docker stores data.** Docker Desktop (WSL2 backend) keeps images and layers in `C:\Users\<user>\AppData\Local\Docker\wsl\disk\docker_data.vhdx`: 19.1 GB file, last modified 2026-10-09,
plus a 0.1 GB `main\ext4.vhdx`. A VHDX grows with use and does not shrink on deletion without compaction. Free space on C: was 47 GB of 636 GB. What is inside the 19 GB cannot be listed while the engine is off.

**Are the NVIDIA packages needed for this server?** Measured in the WSL venv by loading the real LR `server.zip` through `ModelRegistry`:
`concrete.ml` imports `torch` (plus `brevitas`, `onnx`, `skorch`, `xgboost`, `sklearn`, `pandas`, `scipy`), and the installed `torch==2.3.1+cu121` wheel then maps `libcudart`, `libcublas`, `libcudnn`, `libnccl` and the other
`nvidia-*` libraries into the process. They are mapped at import, not used for the computation: the encrypted evaluation runs in Concrete's native runtime, not in torch, and the server never calls torch. The NVIDIA wheels are a
packaging dependency of the CUDA build of torch (`Requires: nvidia-cublas-cu12 ...`), not something the server needs for arithmetic.

**Is a smaller compatible image possible?** Probably yes, in principle, with one substitution: install `torch==2.3.1+cpu` (same version, built without CUDA) from `https://download.pytorch.org/whl/cpu` instead of the `+cu121` wheel, and drop the 12 `nvidia-*` pins from a
server-only requirements file (leave `requirements-fhe.txt` unchanged). Facts checked: the wheel `torch-2.3.1+cpu-cp312-cp312-linux_x86_64.whl` exists on that index (HEAD request: 190,367,618 bytes). Sizes in the current venv:
`nvidia/` 2.8 GB, `torch/` 1.6 GB (`libtorch_cuda.so` 841 MB, `libtorch_cpu.so` 460 MB). Estimate: removing `nvidia/` and the CUDA part of torch brings the installed stack from about 5.9 GB to about 2.2 GB, and the image to about 2.5 GB; peak build disk to
roughly 4-6 GB. These are arithmetic estimates, not measurements.

**What this does and does not preserve.** Model files, bit-widths, memory guards, gates and all recorded results are not touched; the FHE circuit and keys come from Concrete's native code and the saved `server.zip`. But it is a different torch binary from the
one that produced the accepted results, in a different environment (Debian image rather than the WSL venv), so "identical behavior" is **not asserted**; it can only be shown by building the image and passing the live container test against the recorded integers
(`CIPHRAUD_DOCKER_TEST=1`), which has not been done. The Phase 11 status stays partial. If approved, the change would be a new `requirements-server-cpu.txt` and a Dockerfile edit, validated by that test; it is not implemented.

### Update 2026-10-12: server dependency file (approved to prepare; image still NOT built)

`requirements-server-cpu.txt` (new) is generated from `requirements-fhe.txt`: 81 pins -> 69. Exact diff of pins:
`-nvidia-cublas-cu12==12.1.3.1`, `-nvidia-cuda-cupti-cu12==12.1.105`, `-nvidia-cuda-nvrtc-cu12==12.1.105`, `-nvidia-cuda-runtime-cu12==12.1.105`, `-nvidia-cudnn-cu12==8.9.2.26`, `-nvidia-cufft-cu12==11.0.2.54`,
`-nvidia-curand-cu12==10.3.2.106`, `-nvidia-cusolver-cu12==11.4.5.107`, `-nvidia-cusparse-cu12==12.1.0.106`, `-nvidia-nccl-cu12==2.20.5`, `-nvidia-nvjitlink-cu12==12.9.86`, `-nvidia-nvtx-cu12==12.1.105`;
`-torch==2.3.1` / `+torch==2.3.1+cpu`; `+--extra-index-url https://download.pytorch.org/whl/cpu`. Every other pin is identical (a static test enforces exactly this). `Dockerfile` and `.dockerignore` now use the new file;
`requirements-fhe.txt` (the validated environment) is unchanged. Torch's own dependencies (filelock, sympy, networkx, jinja2, fsspec, typing-extensions, mpmath) are already in the freeze.

Expected storage (estimates, not measurements): CPU torch wheel download 190 MB (measured by HEAD request); installed Python layer about 2.2 GB (5.9 GB minus `nvidia/` 2.8 GB minus the CUDA part of torch about 1.1 GB, plus the CPU build); image about 2.5 GB; peak build disk about 4-6 GB inside Docker's data disk (`docker_data.vhdx`, currently 19.1 GB, grows but does not shrink). Free space on C: 47 GB.
Not verified until a build: that pip resolves the set cleanly (not tested; a `pip install --dry-run` would download wheels), the real image size, and equivalence of behavior, which only the live container test can show. Phase 11 remains PARTIAL.

## Build failure and fix: `onnxoptimizer==0.3.13` has no Python 3.12 wheel (2026-10-12)

**Evidence (first real build, gate v4, `data/phase11/logs/docker_build.log`, 11:21-11:22, exit 1 after 75 s).** In `RUN pip install -r requirements-server-cpu.txt`, pip downloaded the source tarball
`onnxoptimizer-0.3.13.tar.gz` (18.5 MB) instead of a wheel and failed in "Getting requirements to build wheel" with `AssertionError: Could not find "cmake" executable!`. This was **not** a memory failure
(free virtual memory stayed at about 6.4 GB during the build). Cause: `concrete-ml==1.9.0` requires `onnxoptimizer==0.3.13` exactly; that release publishes wheels for CPython 3.7-3.11 only, and the base image
`python:3.12-slim-bookworm` has no C++ toolchain or cmake.

**Fix (packaging only).** In the `Dockerfile`, one `RUN` layer installs `build-essential cmake`, runs `pip install -r requirements-server-cpu.txt`, then `apt-get purge -y --auto-remove build-essential cmake` and
removes the apt lists, so the toolchain is used for the build and is not present in the resulting image (a single layer is required; purging in a later layer would not shrink the image). `libgomp1` (the OpenMP runtime)
stays installed before the purge. Static test: `tests/test_phase11_docker.py::test_dockerfile_builds_the_source_only_onnxoptimizer_pin_with_a_toolchain_that_is_purged_in_the_same_layer`.
A multi-stage build was considered; it would also keep the toolchain out of the final image but needs the installed site-packages copied between stages. The single-layer form was chosen as the smaller change.

**What did not change.** Every dependency pin (`requirements-server-cpu.txt` is identical to `requirements-fhe.txt` except the documented nvidia/`+cpu` differences), the Python version, the server code, the models, the
Phase 10 integers, and all Phase 12 freeze artifacts (`results/phase12_test_eval/freeze.json`, SHA-256 prefix `3A192A94...`, unchanged before and after).

**Result of gate v5 (2026-10-12, `data/phase11/logs/docker_gate_v5_state.log`).**
- **Build: SUCCEEDED.** `docker build` exit 0 in 560 s; image `ciphraud-server:phase11`, id `sha256:2419f5db94d773664f422f96fa7749667410d4514cd6d6c70c5dc4f6bdedbcbd`, 3,326,746,012 bytes (3.33 GB; the 2.5 GB estimate above was too low).
  Inside the image: `torch 2.3.1+cpu` and **no** `nvidia-*` packages (probe output `[] ['2.3.1+cpu']`).
- **Static tests: PASSED** (`tests/test_phase11_docker.py`, run from the Windows venv with Docker stopped): 7 passed, 1 skipped, including the new toolchain-purge test.
- **Live container test: NOT RUN at this point (SUPERSEDED later the same day: it was run and passed 8/8, see Results below).** The gate logged `SKIP live test: headroom not ok after build [3.1 21314]` (free virtual memory 3.1 GB, 21,314 pages/s with Docker Desktop running) and then stopped Docker Desktop.
  So the encrypted LR round trip through the container has **not** been demonstrated, and the image's behavioral equivalence to the validated WSL environment is **not** asserted. A skipped test is not a pass.
- The gate therefore ended on a SKIP, not on its DONE line. Phase 11 remains PARTIAL until the live test passes.


## Results (2026-10-12; logs in `data/phase11/logs/`, gitignored)

**Build** (gate v5, `docker_gate_v5_state.log`, `docker_build.log`): exit 0 in 560 s; image `ciphraud-server:phase11`, 3,326,746,012 bytes (3.33 GB), CPU-only (`torch 2.3.1+cpu`, `torch.cuda.is_available() == False`, no `nvidia-*` package),
runs as non-root uid 10001, no compiler left in the image, `/app` contains only `src`, `configs`, the requirements file and empty mount points (no results, data, keys or client artifacts). Lowest free Windows commit during the build: 2.9 GB (abort line was 2.0 GB).
Docker storage afterwards: images 9.75 GB (7.82 GB reclaimable), build cache 13.86 GB, C: free about 129 GB. Nothing was deleted.
**Findings that required changes (all fixed in the committed-to-be files):**
1. `onnxoptimizer==0.3.13` (required exactly by concrete-ml 1.9.0) has no Python 3.12 wheel, so it is built from source in the image; the Dockerfile installs `build-essential` and `cmake` in the same layer and purges them (first build attempt failed on the missing `cmake`).
2. Concrete unpacks the circuit's `sharedlib.so` into `/tmp` and loads it, so the container's tmpfs must allow execution: `--tmpfs /tmp:rw,exec,nosuid,size=512m` (a default `noexec` tmpfs failed with "failed to map segment from shared object").
3. My live test first asserted the wrong HTTP status (415 without the required client-artifact header; the server correctly answers 422 there) and, once, hit a transient `docker run` failure right after the engine started (Windows file sharing not ready yet).

**Live integration test** (`CIPHRAUD_DOCKER_TEST=1 python -m pytest tests/test_phase11_docker.py -v`, run from WSL against Docker Desktop through `docker.exe`; `pytest_live_manual2.log`): **8 passed in 32.6 s**.
The container (loopback-only port, read-only root filesystem, only `server.zip` and the manifest mounted read-only, 3 GB limit) served `GET /v1/models`; refused JSON and header-less key uploads; accepted the evaluation keys of a client that kept its secret key; served one encrypted `lr_top20_bits16` request whose decrypted integers equalled the recorded clear-quantized integers exactly (the same `compare_integers` check as Phase 10); deleted the key set;
its logs contain no key material; no test container was left. Server-side evaluation time for that single request was 0.38 s (n = 1, a cold first request on this machine; not a benchmark, not comparable to Phase 8/10 LR timings).
**Not claimed:** XGBoost or MLP in a container (XGBoost needs about 5 GB of VM memory and about 45 minutes; not run, separately gated), behaviour on another host, TLS/authentication (still none), or equivalence with the WSL-venv environment beyond this one request.
**Run caveats on this PC:** Docker Desktop's engine shares the 5 GB WSL VM; starting it costs about 3.2 GB of Windows commit, and the live test needs `docker.exe` reachable from WSL. The engine must have been up for a minute or so before `docker run` with Windows-path mounts.


## Container validation 2 (2026-10-12 15:00-15:09): MLP in a container and failure handling

**Scope:** MLP (`mlp_top20_bits3`) encrypted inference through the container, plus cheap failure-handling tests. XGBoost in a container was **not** run (it needs about 5 GB of VM memory and about 45 minutes; separately gated). No TLS or authentication was added.
**Pre-flight:** OpenSky was paused (no training or queue process; only a watcher), free Windows commit 12.2 GB before Docker and 9.8 GB with the engine up and warmed, paging about 1,000 pages/s, no image rebuild, `.wslconfig` unchanged (`data/phase11/logs/task12_resources.log`).
**Tests added** (`tests/test_phase11_docker.py`; the LR live test and all static tests are unchanged; one module-scoped container loads both models read-only, loopback only, `--read-only`, 3 GB limit):
1. `test_container_lists_both_models_and_carries_the_mlp_t3_disclosure` - both models listed; the MLP entry carries "failed the Phase 9 quantization-accuracy gate".
2. `test_malformed_wrong_and_incompatible_requests_get_the_documented_errors_and_the_server_survives` - exact documented statuses: unknown model on keys/infer 404 `unknown_model`; missing or wrong client-artifact header 422 `client_artifact_mismatch`; LR keys offered for the MLP 422 `client_artifact_mismatch`; JSON body (header correct) 415 `unsupported_media_type`; garbage keys and truncated real keys 400 `bad_evaluation_keys`; second key upload 409 `key_set_resident`; unknown key id on infer or delete 404 `unknown_key_id`; LR keys used on the MLP 422 `key_model_mismatch`; JSON on infer 415; garbage ciphertext 400 `bad_ciphertext`; 64 MiB + 1 byte body 413 `payload_too_large`; delete 204 then 404; a deleted key set cannot be used (404); after all of that a valid LR encrypted request still returns exact integers; the container is running and not OOM-killed.
3. `test_mlp_encrypted_round_trip_in_the_container_matches_the_recorded_clear_quantized_integers` - client keygen 4.07 s, evaluation keys 115,553,032 bytes uploaded once; two encrypted requests (rows 13560 and 32148, the recorded T1 rows); decrypted integers equal the clear-quantized, simulated and recorded Phase 9 integers exactly for both (probabilities 0.851953 and 0.092688; server evaluation 12.15 s and 15.54 s, n=1 each, not a benchmark); the wire audit shows exactly one evaluation-key body and two input-ciphertext bodies (no other payload; the secret key never left the client); keys deleted, then unusable; container logs contain no key material; `docker stats` after the requests: 984 MiB of the 3 GiB limit (`data/phase11/logs/mlp_container_result.json`).
4. `test_container_restart_recovers_and_forgets_resident_keys` - `docker restart`; healthy again; both models reload from the mounts; the key set registered before the restart is gone (404 `unknown_key_id`); a new upload is accepted (no stale 409); a fresh LR round trip is exact; delete 204.
**Results:** first and only attempt of the new live tests: **4 passed in 122.5 s** (exit 0, `pytest_live_mlp_failure_tests.log`); then the whole file once: **12 passed in 106.0 s** (exit 0, `pytest_live_full_file_after_extension.log`), including the unchanged LR live test. Static-only run on Windows afterwards: 24 passed, 6 skipped (live tests and the Concrete-ML-only test) across the Phase 11 and 12 files. Docker Desktop was stopped afterwards; no container left; free commit 8.7 GB.
**Not shown / limits:** this is local loopback evidence for the LR and MLP circuits, two requests for the MLP and a few for the LR (not a benchmark, not load or concurrency testing); XGBoost in a container is untested; the client-artifact header protects against honest mistakes (foreign keys crash Concrete's native runtime) and is **not** an authentication mechanism: a client that declares the wrong hash for foreign keys was deliberately not tested because it could crash the runtime; the single resident key slot means one client at a time; any local process can reach the loopback port.

### TLS and authentication: recommendation (no scope change made)
- **Loopback-only research demo (current scope): not necessary.** The project's single claim (the server never receives the secret key or plaintext features) is enforced by the protocol and was checked by a wire audit; the port is published on 127.0.0.1 only. Residual local risk: any process on the machine can use the unauthenticated endpoints (e.g. occupy the single key slot). Acceptable for a demo; say so wherever it is described.
- **Anything reachable from a network or shared host: both are required before use.** TLS for integrity and metadata protection of requests and responses in transit (FHE ciphertexts are not authenticated, so a network attacker could tamper with or replay them; this is outside the stated "honest server" threat model but real), and authentication/authorization per client plus rate limits. The one-resident-key-set design also makes it single-tenant; a multi-client service needs per-client key isolation and a redesign, not just a proxy.
- Suggested minimum if ever needed: keep the container on loopback, put a TLS-terminating reverse proxy with per-client tokens (or mutual TLS) in front, and add a regression test for the unauthenticated-access refusal. Not implemented.

## Container validation 3 (2026-10-12 16:00-16:25): contract-exact assertions
- The three LR-test probes that accepted several status codes (`in (415, 422)`, `in (404, 422)`, `in (400, 404, 422)`) now assert the documented validation order exactly (`src/server/app.py`): no client-artifact header on a valid model -> **422 `client_artifact_mismatch`** (the header check precedes the media-type check); unknown model with a valid header -> **404 `unknown_model`**; infer with no resident key -> **404 `unknown_key_id`**. The `detail.error` code is asserted as well as the status.
- Image `ciphraud-server:phase11` (id `2419f5db94d7`, 3.33 GB) was present and not rebuilt. Docker Desktop was started, the whole live file was run **once**: **12 passed in 216.04 s, exit 0** (`data/phase11/logs/pytest_live_final_tightened_assertions.log`, WSL FHE venv, `CIPHRAUD_DOCKER_TEST=1`); Docker Desktop was then stopped and no container remained. (The earlier 106 s runs had a warmer engine; the difference is engine start-up, not the test.)
- Comment-only edits after the build (the `Dockerfile` header and the first line of `requirements-server-cpu.txt` no longer say "LR only" / "DRAFT"): executable instructions and pinned versions are unchanged, so no rebuild was needed. A future rebuild will invalidate the layer cache for the `COPY requirements-server-cpu.txt` layer only.
- Limits unchanged: XGBoost is untested in a container; there is no TLS or authentication (loopback demo only); MLP evidence is two requests; one resident key set (single-tenant).
