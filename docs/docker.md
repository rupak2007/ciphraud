# Phase 11 — Container packaging of the inference server

**Status: partially complete.** The container definition, ignore rules, run recipe and tests are written and statically validated.
**The image has not been built and the container has not been run**, so there is no containerized encrypted request on record.
Do not read this page as evidence that the server works in Docker.

## What exists

- `Dockerfile` — server-only image: `python:3.12-slim-bookworm`, `libgomp1`, the exact pins of `requirements-fhe.txt`, then `src/` and
  `configs/phase10/` only. Runs as non-root user `ciphraud` (uid 10001). `HEALTHCHECK` calls `GET /v1/models`.
  Entrypoint: `python -m src.server --config configs/phase10/client_server.yaml --host 0.0.0.0`; default command `--models lr_top20_bits16`.
- `.dockerignore` — default-deny (`*`), whitelist `Dockerfile`, `requirements-fhe.txt`, `src/`, `configs/phase10/`; key-like files excluded.
  Nothing from `data/`, `results/`, `models/` or the client artifacts can enter a layer.
- `tests/test_phase11_docker.py` — five static tests (what is copied, ignore rules, non-root/pinned Python/healthcheck, every requirement
  pinned, no client code path in `src/server`) pass; one live test is skipped unless Docker is ready (see below).

## Security boundary

Unchanged from Phase 10: the server never receives the secret key or plaintext features. In the container: no keys or client artifacts are baked in;
deployment artifacts are mounted at run time and read-only, **only the `server/` directory (`server.zip`) is mounted, never `client/`**; the root
filesystem can be read-only (`--read-only --tmpfs /tmp`); the port must be published on loopback (`-p 127.0.0.1:8710:8710`) because the server inside listens on
`0.0.0.0` and there is still **no TLS and no authentication**. Do not publish it on a public interface. The security claim is still only the narrow one in `docs/architecture.md` section 10.

## Run recipe (not yet executed)

```
docker build -t ciphraud-server:phase11 .
docker run --rm --read-only --tmpfs /tmp --memory 3g -p 127.0.0.1:8710:8710 \
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
