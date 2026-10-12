# Ciphraud inference SERVER image (Phase 11). Server only: no client code path, no secret keys, no model artifacts baked in.
# Build:  docker build -t ciphraud-server:phase11 .
# Run:    see docs/docker.md (artifacts are mounted read-only at run time; publish the port on loopback only).
# Concrete-ML 1.9.0 needs Python 3.12 on Linux x86-64 (no Windows wheels). Dependencies come from requirements-server-cpu.txt: the
# requirements-fhe.txt freeze minus the 12 nvidia-* CUDA wheels, with torch==2.3.1+cpu (see docs/docker.md). Built 2026-10-12; live-tested over loopback with LR and MLP only (XGBoost untested; no TLS/authentication), see docs/docker.md.
FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1

# libgomp1: OpenMP runtime used by the native FHE/ML libraries. Nothing else is installed from apt.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 ciphraud

WORKDIR /app
COPY requirements-server-cpu.txt ./
# onnxoptimizer==0.3.13 (required EXACTLY by concrete-ml 1.9.0) has no Python 3.12 wheel (cp37-cp311 only), so pip must build it from source
# (found by the first build attempt, 2026-10-12). The toolchain is installed, used and purged in this one layer so it is not in the image.
RUN apt-get update && apt-get install -y --no-install-recommends build-essential cmake     && pip install -r requirements-server-cpu.txt     && apt-get purge -y --auto-remove build-essential cmake     && rm -rf /var/lib/apt/lists/*

# Code and the server config only. results/ and data/ are NOT copied (no keys, no artifacts, no datasets in any layer).
COPY src ./src
COPY configs/phase10 ./configs/phase10
RUN mkdir -p /app/data/phase10/deploy /app/results/phase10_client_server/deploy && chown -R ciphraud:ciphraud /app
USER ciphraud

EXPOSE 8710
# Healthy once the server answers GET /v1/models (the model list doubles as the readiness probe).
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8710/v1/models', timeout=5).status == 200 else 1)"

# Inside the container the server must listen on all interfaces; the host side must publish it on 127.0.0.1 only (docs/docker.md).
ENTRYPOINT ["python", "-m", "src.server", "--config", "configs/phase10/client_server.yaml", "--host", "0.0.0.0"]
CMD ["--models", "lr_top20_bits16"]
