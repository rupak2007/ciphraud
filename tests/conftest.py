"""Shared fixtures for the Phase 10 client/server tests.

Everything here builds TINY synthetic models (8 features) with the project's own builders (`build_concrete_lr`,
`build_concrete_xgb`, `mlp.train_qat_mlp`) and persists them with the SAME `src/fhe/deploy.py::persist_circuit` the real
deployment uses, into a session temp directory, together with a synthetic config whose paths are absolute temp paths. So
the real registry, server, client and runner code paths are exercised without any real data. Concrete-ML is imported lazily
inside the fixtures: on Windows (no Concrete-ML) the tests that need them skip via `needs_concrete`.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

NF = 8


def concrete_available() -> bool:
    try:
        return importlib.util.find_spec("concrete.ml") is not None
    except ModuleNotFoundError:
        return False


needs_concrete = pytest.mark.skipif(
    not concrete_available(),
    reason="concrete-ml is not installed in this environment. It has no Windows wheels; run under the WSL2 venv. See docs/environment.md.",
)


def _toy_arrays():
    rng = np.random.default_rng(0)
    Xs = rng.normal(size=(600, NF)).astype(np.float32)
    w = rng.normal(size=NF)
    y = ((Xs @ w + rng.normal(scale=1.0, size=600)) > 0.5).astype(np.int64)
    mean, scale = np.linspace(-1.0, 1.0, NF), np.linspace(0.5, 2.0, NF)
    raw = (Xs.astype(np.float64) * scale + mean).astype(np.float32)
    return rng, Xs, y, mean, scale, raw


def _build(kind: str) -> SimpleNamespace:
    rng, Xs, y, mean, scale, raw = _toy_arrays()
    scaler = {"kind": "standardize", "scaler_mean": mean.tolist(), "scaler_scale": scale.tolist()}
    if kind == "lr":
        from src.fhe.compile.linear import build_concrete_lr, compile_model

        params = {"lr_coef": [rng.normal(size=NF).tolist()], "lr_intercept": [0.1], "lr_classes": [0, 1], "lr_C": 1.0}
        model = build_concrete_lr(params, Xs.astype(np.float64), 8)
        circuit, _ = compile_model(model, Xs.astype(np.float64))
        spec = {**scaler, "dtype": "float64", "probability": "post_processing"}
    elif kind == "xgboost":
        import xgboost as xgb

        from src.fhe.compile.linear import compile_model
        from src.fhe.compile.tree import build_concrete_xgb

        clf = xgb.XGBClassifier(n_estimators=6, max_depth=3, n_jobs=1).fit(raw, y)
        model = build_concrete_xgb(clf, raw, 6)
        circuit, _ = compile_model(model, raw)
        spec = {"kind": "none", "probability": "post_processing"}
    else:
        from src.fhe.compile import mlp

        cfg = {"n_layers": 3, "n_hidden_neurons_multiplier": 2, "n_accum_bits": 16, "lr": 0.01, "batch_size": 128, "max_epochs": 2, "early_stopping_patience": 2, "min_delta": 0.0}
        model, _ = mlp.train_qat_mlp(cfg, 3, Xs[:400], y[:400], Xs[400:], y[400:], seed=3)
        circuit, _ = mlp.compile_mlp(model, Xs[:400])
        spec = {**scaler, "dtype": "float32", "probability": "mlp_softmax"}
    spec = {**spec, "model_id": f"toy_{kind}", "model_type": kind, "n_features": NF, "columns": [f"f{i}" for i in range(NF)], "threshold": 0.5}
    return SimpleNamespace(kind=kind, model=model, circuit=circuit, raw=raw, y=y, spec=spec, id=f"toy_{kind}")


@pytest.fixture(scope="session")
def toy_models():
    """Compiled tiny models keyed by 'lr' / 'xgboost' / 'mlp' (built once per test session)."""
    return {kind: _build(kind) for kind in ("lr", "xgboost", "mlp")}


@pytest.fixture(scope="session")
def toy_env(tmp_path_factory, toy_models):
    """The three toy models persisted like real deployments, plus a synthetic config (absolute temp paths) and a YAML copy of it."""
    import yaml

    from src.client.preprocess import apply_preprocessing
    from src.fhe.compile.linear import circuit_stats
    from src.fhe.deploy import clear_ints, paths_for, persist_circuit, probabilities, request_positions, sha256_bytes, sha256_path

    root = tmp_path_factory.mktemp("phase10_toy")
    config = {
        "phase": 10, "seed": 42,
        "server": {"host": "127.0.0.1", "port": 0, "max_key_upload_bytes": 400_000_000, "max_ciphertext_bytes": 1_000_000, "startup_timeout_seconds": 120},
        "output": {"results_dir": str(root / "results"), "deploy_dir": str(root / "deploy"), "client_keys_dir": str(root / "client_keys"), "logs_dir": str(root / "logs")},
        # request plan per toy model: [fixed row 0] + [T1 row 1] + 2 repeats of row 0 = 4 requests
        "models": [{"id": m.id, "model_type": m.kind, "grid_config": "n/a", "t1_rows": True, "repeat_fixed_row": 2} for m in toy_models.values()],
        "memory_guards": {"sample_interval_seconds": 0.1, "abort_min_mem_available_mb": 50, "abort_max_swap_used_mb": 900, "go_min_mem_available_mb": 100, "go_max_swap_used_mb": 900, "run_timeout_seconds": 600},
    }
    config_path = root / "client_server.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    from importlib.metadata import version

    deployments = {}
    for kind, m in toy_models.items():
        paths = paths_for(config, m.id)
        artifacts = persist_circuit(m.model, paths)
        (paths["client_dir"] / "preprocessing.json").write_text(json.dumps(m.spec), encoding="utf-8")
        npz = root / "handoff" / f"{m.id}.npz"
        npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez(npz, X_val=m.raw, y_val=m.y)
        manifest = {
            "id": m.id, "model_type": kind, "tier": "toy", "n_bits": 8, "n_features": NF, "artifacts": artifacts,
            "versions": {"concrete-python": version("concrete-python"), "concrete-ml": version("concrete-ml")}, "security_level": None,
            "circuit_stats": circuit_stats(m.circuit), "source": {"handoff_npz": str(npz), "handoff_npz_sha256": sha256_path(npz)}, "t3_disclosure": None,
        }
        paths["manifest"].parent.mkdir(parents=True, exist_ok=True)
        paths["manifest"].write_text(json.dumps(manifest), encoding="utf-8")
        source = {"kind": kind, "cml_model": m.model, "circuit": m.circuit}
        from src.analysis.phase10_results import q_in_digest

        rows, ref_rows = {}, {}
        for position in range(3):
            x = apply_preprocessing(m.spec, m.raw[position : position + 1])
            q = np.asarray(m.model.quantize_input(x))
            raw_clear = np.asarray(clear_ints(source, q))
            probability = float(probabilities(source, raw_clear)[0])
            rows[position] = {"q_in": q, "clear": raw_clear.reshape(-1).astype(np.int64), "probability": probability}
            ref_rows[str(position)] = {
                "y_true": int(m.y[position]), "q_in_sha256": q_in_digest(q), "q_in_length": int(q.size), "clear_ints": raw_clear.reshape(-1).tolist(),
                "simulate_ints": np.asarray(m.circuit.simulate(q)).reshape(-1).tolist(), "recorded_ints": None, "probability": probability,
            }
        plan = request_positions(0, [0, 1], True, 2)
        reference = {"id": m.id, "fixed_position": 0, "t1_positions": [0, 1], "request_positions": plan, "rows": ref_rows, "threshold": 0.5, "reproduction_proof": {"passed": True}}
        paths["reference"].write_text(json.dumps(reference), encoding="utf-8")
        deployments[kind] = SimpleNamespace(id=m.id, paths=paths, manifest=manifest, rows=rows, toy=m, sha256=sha256_bytes, reference=reference)
    return SimpleNamespace(root=root, config=config, config_path=config_path, deployments=deployments)
