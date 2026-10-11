"""Launch the Phase 10 inference server (WSL2/Linux), from the project root:

    ~/.venvs/fhe-fraud-detection/bin/python -m src.server --config configs/phase10/client_server.yaml --models lr_top20_bits16

Loads only the named models (default: all configured), prints one JSON readiness line with the process id and load times,
then serves on `server.host:server.port` (loopback by default; no TLS or authentication -- see `docs/client_server.md`).
"""

from __future__ import annotations

import argparse
import json
import warnings

import uvicorn

from src.config import load_config
from src.server.app import create_app
from src.server.registry import ModelRegistry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/phase10/client_server.yaml")
    parser.add_argument("--models", help="comma-separated model ids to serve (default: every configured model)")
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    args = parser.parse_args()
    warnings.filterwarnings("ignore")
    config = load_config(args.config)
    ids = args.models.split(",") if args.models else None
    registry = ModelRegistry(config, ids)
    server_cfg = config["server"]
    host, port = args.host or server_cfg["host"], args.port or int(server_cfg["port"])
    import os

    print(json.dumps({"event": "server_ready", "pid": os.getpid(), "host": host, "port": port, "models": {m["id"]: m["load_seconds"] for m in registry.describe()}}), flush=True)
    uvicorn.run(create_app(registry, server_cfg), host=host, port=port, log_level="warning", access_log=False)


if __name__ == "__main__":
    main()
