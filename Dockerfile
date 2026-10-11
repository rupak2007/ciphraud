# Ciphraud inference SERVER image (Phase 11). Server only: no client code path, no secret keys, no model artifacts baked in.
# Build:  docker build -t ciphraud-server:phase11 .
# Run:    see docs/docker.md (artifacts are mounted read-only at run time; publish the port on loopback only).
# Concrete-ML 1.9.0 needs Python 3.12 on Linux x86-64 (no Windows wheels); the pinned freeze includes torch and CUDA wheels,
# so the image is several GB. Versions are exactly those in requirements-fhe.txt.
FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1

# libgomp1: OpenMP runtime used by the native FHE/ML libraries. Nothing else is installed from apt.
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 10001 ciphraud

WORKDIR /app
COPY requirements-fhe.txt ./
RUN pip install -r requirements-fhe.txt

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
