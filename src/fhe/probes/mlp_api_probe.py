"""Phase 9 R1 -- Concrete-ML `NeuralNetClassifier` API and behaviour discovery (E0).

Small, seeded, SYNTHETIC-data experiments that pin down, by observation, how Concrete-ML 1.9.0 behaves
before the Phase 9 code relies on it (`CLAUDE.md` Sec.17: identify uncertainty, run a small controlled test,
record the observed behaviour). No project data is read and nothing outside `results/phase9_mlp/` is written.
Run under the WSL2 FHE venv:

    ~/.venvs/fhe-fraud-detection/bin/python -m src.fhe.probes.mlp_api_probe [--fhe-run]

Facts recorded in `results/phase9_mlp/api_probe.json`:
  1. kwargs that are forbidden / set automatically, and the architecture that `module__n_layers` and
     `module__n_hidden_neurons_multiplier` produce;
  2. how the accumulator ceiling behaves (`module__n_accum_bits`) across weight/activation widths and input
     dimensions: compile success, PBS count, maximum integer bit-width, bootstrap/keyswitch key sizes;
  3. integer-level outputs: does `quantized_forward(fhe="disable")` equal `circuit.simulate`, and does
     `post_processing(dequantize_output(ints))` reproduce `predict_proba`;
  4. serialization: what works and what does not before/after compile;
  5. (optional, `--fhe-run`) one real encrypt -> run -> decrypt round trip on the smallest circuit.
"""

from __future__ import annotations

import argparse
import json
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from src.config import PROJECT_ROOT
from src.logging_setup import get_logger

logger = get_logger(__name__)

OUT_PATH = PROJECT_ROOT / "results" / "phase9_mlp" / "api_probe.json"
SEED = 0


def _synthetic(dim: int, n: int = 3000, seed: int = SEED) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, dim)).astype(np.float32)
    w = rng.normal(size=dim)
    y = ((X @ w + rng.normal(scale=2.0, size=n)) > 1.5).astype(np.int64)
    return X, y


def _params(n_bits: int, n_accum_bits: int = 16, n_layers: int = 3, epochs: int = 4) -> dict[str, Any]:
    from torch import nn

    return dict(
        module__n_layers=n_layers, module__n_w_bits=n_bits, module__n_a_bits=n_bits, module__n_accum_bits=n_accum_bits,
        module__activation_function=nn.ReLU, max_epochs=epochs, lr=0.01, batch_size=256, verbose=0,
        callbacks="disable",
    )


def _seed(seed: int = SEED) -> None:
    """`random_state` is NOT an accepted kwarg (see `probe_kwargs`), so weight init and batch order are seeded globally."""
    import torch

    np.random.seed(seed)
    torch.manual_seed(seed)


def probe_kwargs() -> dict[str, Any]:
    from concrete.ml.sklearn import NeuralNetClassifier

    out: dict[str, Any] = {}
    try:
        NeuralNetClassifier(module__n_outputs=2, module__input_dim=20, module__n_layers=3)
        out["manual_n_outputs_input_dim"] = "accepted"
    except ValueError as exc:
        out["manual_n_outputs_input_dim"] = f"ValueError: {exc}"

    X, y = _synthetic(20)
    try:
        NeuralNetClassifier(**_params(4), random_state=SEED).fit(X, y)
        out["random_state_kwarg"] = "accepted"
    except TypeError as exc:
        out["random_state_kwarg"] = f"TypeError: {str(exc)[:160]}"
    _seed()
    model = NeuralNetClassifier(**_params(4))
    model.fit(X, y)
    seeded_twice = NeuralNetClassifier(**_params(4))
    _seed()
    seeded_twice.fit(X, y)
    out["torch_manual_seed_makes_training_reproducible"] = bool(
        np.array_equal(model.predict_proba(X[:200]), seeded_twice.predict_proba(X[:200]))
    )
    out["n_layers_3_weight_shapes_dim20"] = [list(p.shape) for n, p in model.module_.named_parameters() if "weight" in n]
    out["defaults_seen"] = {"n_hidden_neurons_multiplier": 4, "power_of_two_scaling": True, "activation": "ReLU (as passed)"}
    return out


def probe_accumulator_sweep(dims: tuple[int, ...] = (20, 100), bits: tuple[int, ...] = (2, 4, 6, 8)) -> list[dict[str, Any]]:
    """Compile-only (no key generation): what does the compiler report for each width and input dimension?"""
    from concrete.ml.sklearn import NeuralNetClassifier

    rows = []
    for dim in dims:
        X, y = _synthetic(dim)
        for b in bits:
            row: dict[str, Any] = {"input_dim": dim, "n_w_bits": b, "n_a_bits": b, "n_accum_bits": 16}
            try:
                model = NeuralNetClassifier(**_params(b))
                model.fit(X, y)
                t0 = time.perf_counter()
                circuit = model.compile(X)
                st = circuit.statistics
                row.update({
                    "compiled": True, "compile_seconds": time.perf_counter() - t0,
                    "programmable_bootstrap_count": int(st["programmable_bootstrap_count"]),
                    "max_integer_bit_width": int(circuit.graph.maximum_integer_bit_width()),
                    "bootstrap_key_bytes": int(st["size_of_bootstrap_keys"]), "keyswitch_key_bytes": int(st["size_of_keyswitch_keys"]),
                    "hidden_width": int(model.module_.state_dict()[next(k for k in model.module_.state_dict() if "weight" in k)].shape[0]),
                })
            except Exception as exc:  # recorded, never dropped (docs/instructions.md FHE rules)
                row.update({"compiled": False, "error_type": type(exc).__name__, "error": str(exc)[:400]})
            logger.info("accumulator sweep row", extra={"extra_fields": row})
            rows.append(row)
    return rows


def probe_integer_outputs_and_serialization() -> dict[str, Any]:
    from concrete.ml.common.serialization.dumpers import dumps
    from concrete.ml.common.serialization.loaders import loads
    from concrete.ml.sklearn import NeuralNetClassifier

    X, y = _synthetic(20)
    model = NeuralNetClassifier(**_params(4))
    model.fit(X, y)
    circuit = model.compile(X)
    q = model.quantize_input(X[:50])
    quantized_module = model.quantized_module_

    clear_ints = np.asarray(quantized_module.quantized_forward(q, fhe="disable"))
    sim_ints = np.array([circuit.simulate(q[i : i + 1])[0] for i in range(q.shape[0])])
    proba_manual = model.post_processing(model.dequantize_output(sim_ints))
    out: dict[str, Any] = {
        "input_quantizer_n_bits": int(np.asarray(quantized_module.input_quantizers[0].n_bits)),
        "output_quantizer_n_bits": int(quantized_module.output_quantizers[0].n_bits),
        "quantize_input_dtype": str(q.dtype), "circuit_output_dtype": str(sim_ints.dtype), "circuit_output_shape_per_row": list(sim_ints.shape[1:]),
        "clear_integer_outputs_equal_simulate": bool(np.array_equal(clear_ints, sim_ints)),
        "post_processing_of_dequantized_integers_equals_predict_proba_disable": bool(np.array_equal(proba_manual, model.predict_proba(X[:50], fhe="disable"))),
        "predict_proba_disable_equals_simulate": bool(np.array_equal(model.predict_proba(X[:50], fhe="disable"), model.predict_proba(X[:50], fhe="simulate"))),
    }

    reloaded = loads(dumps(model))
    out["dumps_loads_with_callbacks_disable"] = "ok"
    out["dumps_loads_reproduces_clear_quantized_proba_exactly"] = bool(
        np.array_equal(reloaded.predict_proba(X[:500], fhe="disable"), model.predict_proba(X[:500], fhe="disable"))
    )
    out["reloaded_model_is_compiled_flag"] = bool(getattr(reloaded.quantized_module_, "is_compiled", False))

    from torch import nn
    no_disable = NeuralNetClassifier(**{**_params(4), "callbacks": None})
    no_disable.fit(X, y)
    try:
        dumps(no_disable)
        out["dumps_with_default_callbacks"] = "ok"
    except NotImplementedError as exc:
        out["dumps_with_default_callbacks"] = f"NotImplementedError: {exc}"
    try:
        import pickle
        pickle.dumps(model)
        out["pickle_after_compile"] = "ok"
    except Exception as exc:
        out["pickle_after_compile"] = f"{type(exc).__name__}: {exc}"
    del nn
    return out


def probe_real_fhe_round_trip() -> dict[str, Any]:
    from concrete.ml.sklearn import NeuralNetClassifier

    X, y = _synthetic(20)
    model = NeuralNetClassifier(**_params(4))
    model.fit(X, y)
    circuit = model.compile(X)
    q = model.quantize_input(X[:1])
    t0 = time.perf_counter(); circuit.keygen(); keygen = time.perf_counter() - t0
    t0 = time.perf_counter(); enc = circuit.encrypt(q); encrypt = time.perf_counter() - t0
    t0 = time.perf_counter(); res = circuit.run(enc); run = time.perf_counter() - t0
    t0 = time.perf_counter(); dec = np.asarray(circuit.decrypt(res)); decrypt = time.perf_counter() - t0
    return {
        "config": "synthetic, 20 features, n_bits=4, n_accum_bits=16", "keygen_seconds": keygen, "encrypt_seconds": encrypt, "run_seconds": run,
        "decrypt_seconds": decrypt, "decrypted_equals_simulate": bool(np.array_equal(dec[0], circuit.simulate(q)[0])),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fhe-run", action="store_true", help="also time one real encrypt->run->decrypt round trip (~20 s)")
    args = parser.parse_args()
    warnings.filterwarnings("ignore")

    import concrete.ml
    import torch

    result: dict[str, Any] = {
        "note": "synthetic data only; facts about Concrete-ML behaviour, not about the fraud models",
        "versions": {"concrete-ml": concrete.ml.version.__version__, "torch": torch.__version__},
        "kwargs_and_architecture": probe_kwargs(),
        "accumulator_sweep_n_accum_bits_16": probe_accumulator_sweep(),
        "integer_outputs_and_serialization": probe_integer_outputs_and_serialization(),
    }
    if args.fhe_run:
        result["real_fhe_round_trip"] = probe_real_fhe_round_trip()
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(result, indent=2, sort_keys=True, default=str), encoding="utf-8")
    logger.info("MLP API probe written", extra={"extra_fields": {"path": str(OUT_PATH)}})


if __name__ == "__main__":
    main()
