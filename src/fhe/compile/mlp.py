"""Phase 9 -- quantized MLP (Brevitas QAT via Concrete-ML's `NeuralNetClassifier`): build, train, compile, evaluate.

WSL2/Linux only for anything that touches Concrete-ML (`docs/environment.md`); `torch`/`concrete` are imported lazily
inside the functions that need them so the pure-numpy helpers stay importable and testable everywhere. Like
`src/fhe/compile/linear.py`, nothing here reads a config file or a path: every function takes plain arrays and a
plain hyperparameter dict, so it is unit-testable on a tiny synthetic problem (`tests/test_fhe_compile_mlp.py`).

Decisions this module implements (`docs/fhe_mlp.md` records each one; the API facts are measured by
`src/fhe/probes/mlp_api_probe.py`, `results/phase9_mlp/api_probe.json`):

  * **One bit-width knob.** `n_bits` sets weight AND activation bits (`module__n_w_bits = module__n_a_bits`);
    the accumulator is capped at `n_accum_bits` (16, the width Concrete's table lookups support).
  * **Class imbalance** via `criterion__weight` -- balanced weights from the TRAIN labels only, as for LR/XGBoost.
  * **Validation early stopping.** One epoch per `fit` call (`warm_start`), the *clear-quantized* validation PR-AUC
    after each epoch, best weights restored (`Concrete-ML re-exports and re-quantizes the module on every fit`, so a
    zero-epoch refit after restoring rebuilds the quantized module from the best weights).
  * **Probability = sigmoid(logit_1 - logit_0)** computed here from the dequantized logits. Concrete-ML's own
    `predict_proba` applies an element-wise sigmoid to both logits for two classes (its rows do not sum to 1), which
    is monotone for classification but is not a probability of the fraud class; every probability in Phase 9 comes
    from `fraud_probability` instead, for the QAT model and the float twin alike.
  * **Seeds.** `random_state` is not an accepted kwarg; training is seeded with `torch.manual_seed`.
"""

from __future__ import annotations

import copy
import time
from typing import Any

import numpy as np
from sklearn.metrics import average_precision_score

DEFAULT_CHUNK = 20000


# ------------------------------------------------------------------------------------------------ pure numpy


def fraud_probability(logits: np.ndarray) -> np.ndarray:
    """P(fraud) = softmax(logits)[:, 1] = sigmoid(logit_1 - logit_0), evaluated stably (no overflow for large |diff|)."""
    logits = np.asarray(logits, dtype=np.float64)
    diff = logits[:, 1] - logits[:, 0]
    return 0.5 * (1.0 + np.tanh(0.5 * diff))


def balanced_class_weights(y_train: np.ndarray) -> list[float]:
    """`[w_0, w_1]` for `criterion__weight`: sklearn's 'balanced' formula, from the TRAIN labels only."""
    y = np.asarray(y_train)
    counts = np.bincount(y, minlength=2).astype(np.float64)
    if (counts == 0).any():
        raise ValueError("balanced_class_weights: both classes must be present in y_train")
    return (len(y) / (2.0 * counts)).tolist()


def client_quantize(quantizer: Any, values: np.ndarray) -> np.ndarray:
    """The client-side input quantization, re-implemented from the quantizer's OWN parameters (scale, zero point,
    bit-width, offset, narrow-range flag) -- the function a client that holds only these parameters would run before
    encrypting. `tests/test_fhe_compile_mlp.py` requires it to equal `model.quantize_input` exactly."""
    q = np.rint(np.asarray(values, dtype=np.float64) / np.asarray(quantizer.scale) + np.asarray(quantizer.zero_point))
    if not quantizer.no_clipping:
        low = -int(quantizer.offset) + (1 if quantizer.is_narrow else 0)
        q = np.clip(q, low, 2 ** int(np.asarray(quantizer.n_bits)) - 1 - int(quantizer.offset))
    return q.astype(np.int64)


# ------------------------------------------------------------------------------------------------ QAT model


def seed_everything(seed: int) -> None:
    import torch

    np.random.seed(seed)
    torch.manual_seed(seed)


def qat_kwargs(mlp_cfg: dict[str, Any], n_bits: int, class_weights: list[float]) -> dict[str, Any]:
    from torch import nn

    return dict(
        module__n_layers=mlp_cfg["n_layers"], module__n_w_bits=n_bits, module__n_a_bits=n_bits,
        module__n_accum_bits=mlp_cfg["n_accum_bits"], module__n_hidden_neurons_multiplier=mlp_cfg["n_hidden_neurons_multiplier"],
        module__activation_function=nn.ReLU, criterion__weight=list(class_weights),
        lr=mlp_cfg["lr"], batch_size=mlp_cfg["batch_size"], max_epochs=1, warm_start=True, callbacks="disable", verbose=0,
    )


def build_qat_mlp(mlp_cfg: dict[str, Any], n_bits: int, class_weights: list[float]) -> Any:
    from concrete.ml.sklearn import NeuralNetClassifier

    return NeuralNetClassifier(**qat_kwargs(mlp_cfg, n_bits, class_weights))


def clear_quantized_logits(model: Any, X: np.ndarray, chunk: int = DEFAULT_CHUNK) -> np.ndarray:
    """Dequantized logits of the CLEAR quantized model (the computation behind `fhe="disable"`), in row chunks."""
    quantized_module = model.quantized_module_
    integers = [np.asarray(quantized_module.quantized_forward(model.quantize_input(X[s : s + chunk]), fhe="disable")) for s in range(0, X.shape[0], chunk)]
    return np.asarray(model.dequantize_output(np.concatenate(integers)))


def restore_weights(model: Any, state: dict[str, Any], X_calibration: np.ndarray, y_calibration: np.ndarray) -> None:
    """Put `state` back into the module and rebuild Concrete-ML's ONNX/quantized module from it: a zero-epoch `fit`
    (every `fit` re-exports and re-quantizes; with `max_epochs=0` and `warm_start` no training step runs).

    `X_calibration`/`y_calibration` need not be the full training set (decision D7, `docs/fhe_mlp.md`): a calibration
    diagnostic on the completed `top_20_bits3`/`bits4` checkpoints found this re-export calibration-set-size
    INVARIANT for Brevitas QAT -- the quantizer scale/zero-point are fixed, learned power-of-two values, not
    statistics derived from the calibration data's range, so a ~3,000-row subset (the same one already used to
    compile the FHE circuit) reproduces every quantizer parameter and all validation-row integer outputs exactly
    (`tests/test_fhe_compile_mlp.py::test_restore_weights_with_a_calibration_subset_matches_full_data_exactly`).
    This is a memory fix for the WSL VM's ~3.8 GiB cap (the earlier full-380,815-row calibration was killed by the
    kernel OOM killer for `top_50`/`top_100`), not a methodology change.

    Measured (docs/fhe_mlp.md): `model.set_params(max_epochs=...)` changes only the Concrete-ML wrapper and does NOT
    reach the inner skorch net that actually trains, so the epoch count must be set on `model.sklearn_model`."""
    model.base_module.load_state_dict(state)
    model.sklearn_model.set_params(max_epochs=0)
    try:
        model.fit(X_calibration, y_calibration)
    finally:
        model.sklearn_model.set_params(max_epochs=1)


def qat_forward_logits(model: Any, X: np.ndarray, chunk: int = DEFAULT_CHUNK) -> np.ndarray:
    """Logits of the torch QAT module (Brevitas fake-quantized forward). Fast, and it needs no ONNX export, so it is what
    picks the stopping epoch; the model that is reported and compiled is always scored with `clear_quantized_logits`."""
    import torch

    module = model.base_module
    module.eval()
    with torch.no_grad():
        return np.concatenate([module(torch.from_numpy(np.asarray(X[s : s + chunk], dtype=np.float32))).numpy() for s in range(0, X.shape[0], chunk)])


def _init_sklearn_model(model: Any, X_train: np.ndarray, y_train: np.ndarray) -> None:
    """Construct `model.sklearn_model` the same way Concrete-ML's OWN `NeuralNetClassifier.fit()` preamble does,
    WITHOUT calling that `.fit()` -- so its unconditional full-data ONNX export + `PostTrainingQATImporter(...)
    .quantize_module(X)` (decision D7 already removed it from the FINAL restore step; a memory probe measured it, not
    the actual gradient training, as the sole driver of the WSL VM's OOM) never runs for epoch 1 either (decision D8,
    `docs/fhe_mlp.md`).

    *** PRIVATE CONCRETE-ML/SKORCH API DEPENDENCY -- READ BEFORE UPGRADING concrete-ml OR skorch ***
    Traced from the installed library source (`concrete/ml/sklearn/qnn.py::NeuralNetClassifier.fit`,
    `concrete/ml/sklearn/base.py::QuantizedTorchEstimatorMixin.fit`/`BaseEstimator._fit_sklearn_model`,
    `skorch/net.py::NeuralNet.fit`), epoch 1's ORIGINAL `model.fit(X_train, y_train)` does exactly:
      1. sets `module__n_outputs = len(numpy.unique(y))` and `module__input_dim = X.shape[1]` on the wrapper;
      2. converts `criterion__weight` from a list to a `torch.Tensor` if needed;
      3. constructs `model.sklearn_model = model.sklearn_model_class(**model.get_sklearn_params())` (first call only);
      4. calls `model.sklearn_model.fit(X, y)` -- skorch's OWN `fit()`, whose entire body is
         `if not initialized_: self.initialize()` (builds the torch module/criterion/optimizer; consumes the torch
         RNG for weight init) `; self.partial_fit(X, y)` (runs ONE real mini-batched training epoch) -- NOTHING else;
      5. THEN, back in Concrete-ML's own `.fit()` (one layer above skorch, never entered by this function): exports
         the trained module to ONNX using a single probe row (cheap) and calls `quantize_module(X)` on the FULL `X`
         (the expensive, row-count-scaling step this function avoids).
    This function performs ONLY steps 1-3 above, replicating the exact formulas Concrete-ML's preamble uses. The
    caller must then invoke step 4 itself, via `model.sklearn_model.fit(X_train, y_train)` (skorch-level, NOT
    `model.fit`), to get the identical real training epoch without ever entering step 5.

    It relies on three Concrete-ML/skorch internals that have no leading underscore (reachable) but are NOT
    documented public API -- exactly what a `concrete-ml`/`skorch` version upgrade could change or remove:
      - `model.sklearn_model_class`  (currently `skorch.classifier.NeuralNetClassifier`)
      - `model.get_sklearn_params()` (Concrete-ML method; relies on skorch's own convention that EVERY plain
        instance attribute not ending in "_" is a constructor parameter -- `skorch.net.NeuralNet._get_param_names`)
      - the construction line itself, copied from `BaseEstimator._fit_sklearn_model`
    Verified equivalent on the real `top_20` data before this was implemented (bit-identical `base_module.state_dict()`
    and `qat_forward_logits` after one subsequent `model.sklearn_model.fit()` call vs. the original `model.fit()`
    path) and covered by `tests/test_fhe_compile_mlp.py::test_d8_*`, which will fail loudly -- not silently diverge --
    if a future upgrade changes any of the three items above."""
    n_classes = int(len(np.unique(y_train)))
    model.module__n_outputs = n_classes
    model.module__input_dim = int(np.asarray(X_train).shape[1])
    if isinstance(model.criterion__weight, list):
        import torch

        model.criterion__weight = torch.from_numpy(np.asarray(model.criterion__weight, dtype=np.float32)).float()
    model.sklearn_model = model.sklearn_model_class(**model.get_sklearn_params())


def train_qat_mlp(
    mlp_cfg: dict[str, Any], n_bits: int, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, seed: int,
    X_calibration: np.ndarray | None = None, y_calibration: np.ndarray | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Train with class-weighted loss, early stopping on validation PR-AUC, best weights restored.

    Speed (measured, `docs/fhe_mlp.md`): every Concrete-ML `fit` call re-exports the network to ONNX and re-quantizes it on
    whatever `X` it is given, a fixed ~60-70 s that dwarfs a ~20 s epoch. Epoch 1 needs no such export -- only the FINAL
    restore does (decision D7) -- so epoch 1 is built via `_init_sklearn_model` + skorch's own `.fit()` (decision D8),
    never entering Concrete-ML's `.fit()` at all; later epochs run on the inner skorch net (`partial_fit`, as before, no
    export), the stopping epoch is chosen from the validation PR-AUC of the QAT forward, and ONE final zero-epoch `fit`
    exports the best weights -- the only place `X_calibration`/`y_calibration` apply (decision D7, `docs/fhe_mlp.md`;
    see `restore_weights`). Every epoch of actual gradient training always uses the full `X_train`/`y_train`, in the
    same order, with the same seed, hyperparameters and early stopping as before D7/D8; omitting `X_calibration`
    reproduces the exact original full-data-everywhere behavior. The reported clear-quantized validation PR-AUC of the
    final model is recorded next to the epoch-selection curve.

    Returns the fitted (NOT compiled) model and a history dict. The validation partition is used for the stopping epoch
    (the same caveat as XGBoost's validation-based early stopping, `docs/research.md` Sec.7.7)."""
    weights = balanced_class_weights(y_train)
    seed_everything(seed)
    model = build_qat_mlp(mlp_cfg, n_bits, weights)
    X_train = np.asarray(X_train, dtype=np.float32)
    X_val = np.asarray(X_val, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int64)
    X_cal = X_train if X_calibration is None else np.asarray(X_calibration, dtype=np.float32)
    y_cal = y_train if y_calibration is None else np.asarray(y_calibration, dtype=np.int64)
    _init_sklearn_model(model, X_train, y_train)  # decision D8: build model.sklearn_model without the full-data export

    history: list[dict[str, float]] = []
    best_pr, best_epoch, best_state, bad = -1.0, 0, None, 0
    for epoch in range(1, int(mlp_cfg["max_epochs"]) + 1):
        t0 = time.perf_counter()
        if epoch == 1:
            model.sklearn_model.fit(X_train, y_train)  # skorch's OWN fit: initialize() + exactly one real training epoch
        else:
            model.sklearn_model.partial_fit(X_train, y_train)  # one more epoch, no ONNX export
        pr = float(average_precision_score(y_val, fraud_probability(qat_forward_logits(model, X_val))))
        history.append({"epoch": epoch, "val_pr_auc_qat_forward": pr, "seconds": time.perf_counter() - t0})
        if pr > best_pr + float(mlp_cfg.get("min_delta", 0.0)):
            best_pr, best_epoch, best_state, bad = pr, epoch, copy.deepcopy(model.base_module.state_dict()), 0
        else:
            bad += 1
            if bad >= int(mlp_cfg["early_stopping_patience"]):
                break

    t0 = time.perf_counter()
    restore_weights(model, best_state, X_cal, y_cal)  # exports the BEST epoch's weights (calibration set: see above)
    export_seconds = time.perf_counter() - t0
    final_pr = float(average_precision_score(y_val, fraud_probability(clear_quantized_logits(model, X_val))))
    return model, {
        "epochs_run": len(history), "best_epoch": best_epoch, "best_val_pr_auc_qat_forward": best_pr,
        "final_val_pr_auc_clear_quantized": final_pr, "stopped_early": len(history) < int(mlp_cfg["max_epochs"]),
        "final_export_seconds": export_seconds, "epoch_selection_metric": "validation PR-AUC of the QAT (fake-quantized) forward",
        "class_weights": weights, "history": history,
        "calibration_rows_used": int(X_cal.shape[0]), "calibration_is_full_train_set": bool(X_cal.shape[0] == X_train.shape[0]),
    }


# ------------------------------------------------------------------------------------------------ float twin


def _float_net(n_features: int, mlp_cfg: dict[str, Any]) -> Any:
    from torch import nn

    hidden = int(mlp_cfg["n_hidden_neurons_multiplier"]) * n_features
    layers: list[Any] = [nn.Linear(n_features, hidden), nn.ReLU()]
    for _ in range(int(mlp_cfg["n_layers"]) - 2):
        layers += [nn.Linear(hidden, hidden), nn.ReLU()]
    layers.append(nn.Linear(hidden, 2))
    return nn.Sequential(*layers)


def float_logits(net: Any, X: np.ndarray, chunk: int = DEFAULT_CHUNK) -> np.ndarray:
    import torch

    net.eval()
    with torch.no_grad():
        return np.concatenate([net(torch.from_numpy(np.asarray(X[s : s + chunk], dtype=np.float32))).numpy() for s in range(0, X.shape[0], chunk)])


def train_float_twin(
    mlp_cfg: dict[str, Any], X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray, seed: int,
) -> tuple[Any, dict[str, Any]]:
    """The float reference for T3 (decision D2): the SAME architecture (input -> multiplier*input -> ... -> 2, ReLU),
    data, seed, class weights, optimizer, learning rate, batch size, epoch budget and validation early stopping as the
    QAT model, with no quantization. A plain PyTorch loop; best-validation weights restored."""
    import torch
    from torch import nn

    weights = balanced_class_weights(y_train)
    seed_everything(seed)
    net = _float_net(X_train.shape[1], mlp_cfg)
    optimizer = torch.optim.Adam(net.parameters(), lr=float(mlp_cfg["lr"]))
    loss_fn = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32))
    Xt = torch.from_numpy(np.asarray(X_train, dtype=np.float32))
    yt = torch.from_numpy(np.asarray(y_train, dtype=np.int64))
    generator = torch.Generator().manual_seed(seed)

    history, best_pr, best_epoch, best_state, bad = [], -1.0, 0, None, 0
    for epoch in range(1, int(mlp_cfg["max_epochs"]) + 1):
        t0 = time.perf_counter()
        net.train()
        order = torch.randperm(Xt.shape[0], generator=generator)
        for s in range(0, Xt.shape[0], int(mlp_cfg["batch_size"])):
            idx = order[s : s + int(mlp_cfg["batch_size"])]
            optimizer.zero_grad()
            loss_fn(net(Xt[idx]), yt[idx]).backward()
            optimizer.step()
        pr = float(average_precision_score(y_val, fraud_probability(float_logits(net, X_val))))
        history.append({"epoch": epoch, "val_pr_auc_float": pr, "seconds": time.perf_counter() - t0})
        if pr > best_pr + float(mlp_cfg.get("min_delta", 0.0)):
            best_pr, best_epoch, best_state, bad = pr, epoch, copy.deepcopy(net.state_dict()), 0
        else:
            bad += 1
            if bad >= int(mlp_cfg["early_stopping_patience"]):
                break
    net.load_state_dict(best_state)
    return net, {"epochs_run": len(history), "best_epoch": best_epoch, "best_val_pr_auc_float": best_pr, "class_weights": weights, "history": history}


# ------------------------------------------------------------------------------------------------ serialization + compile


def dump_model(model: Any) -> str:
    """JSON string via Concrete-ML's own serializer (works only because `callbacks="disable"`; `pickle` does not work).

    Measured (docs/fhe_mlp.md): a model trained with `criterion__weight` serializes, but `loads` then fails with
    `Unexpected key(s) in state_dict: "weight"` because the freshly built loss has no weight buffer. The loss is not
    used at inference, so its class-weight buffer is dropped for the duration of the dump only and put back after."""
    from concrete.ml.common.serialization.dumpers import dumps

    criterion = model.sklearn_model.criterion_
    class_weight = criterion.weight
    criterion.weight = None
    try:
        return dumps(model)
    finally:
        criterion.weight = class_weight


def load_model(serialized: str) -> Any:
    from concrete.ml.common.serialization.loaders import loads

    return loads(serialized)


def compile_mlp(model: Any, X_calibration: np.ndarray) -> tuple[Any, float]:
    """Compile against a standardized TRAIN calibration inputset; returns the circuit and the compile time (seconds)."""
    t0 = time.perf_counter()
    circuit = model.compile(np.asarray(X_calibration, dtype=np.float32))
    return circuit, time.perf_counter() - t0


def mlp_stats(model: Any, circuit: Any, mlp_cfg: dict[str, Any], n_bits: int) -> dict[str, Any]:
    """Everything `docs/instructions.md` asks to be logged per MLP configuration, plus the bootstrap (TLU) count.

    `programmable_bootstrap_count` is the number of table lookups the compiled circuit performs (every non-linear step
    and every re-quantization is one); `hidden_neurons` is the structural ReLU count (one activation per hidden neuron)
    for comparison. `p_error`, key sizes and ciphertext sizes come from the circuit itself."""
    weights = [tuple(p.shape) for name, p in model.base_module.named_parameters() if name.endswith("weight") and p.ndim == 2]
    statistics = circuit.statistics
    return {
        "n_w_bits": n_bits, "n_a_bits": n_bits, "n_accum_bits": int(mlp_cfg["n_accum_bits"]), "n_layers": int(mlp_cfg["n_layers"]),
        "layer_shapes": [list(s) for s in weights], "hidden_neurons": int(sum(s[0] for s in weights[:-1])),
        "n_bits_inputs": [int(np.asarray(q.n_bits)) for q in model.quantized_module_.input_quantizers],
        "n_bits_output": int(model.quantized_module_.output_quantizers[0].n_bits),
        "max_integer_bit_width": int(circuit.graph.maximum_integer_bit_width()),
        "programmable_bootstrap_count": int(statistics["programmable_bootstrap_count"]),
        "key_switch_count": int(statistics["key_switch_count"]),
        "p_error": float(circuit.p_error), "global_p_error": float(circuit.global_p_error), "complexity": float(circuit.complexity),
        "size_of_secret_keys_bytes": int(circuit.size_of_secret_keys), "size_of_bootstrap_keys_bytes": int(circuit.size_of_bootstrap_keys),
        "size_of_keyswitch_keys_bytes": int(circuit.size_of_keyswitch_keys),
        "size_of_inputs_bytes": int(circuit.size_of_inputs), "size_of_outputs_bytes": int(circuit.size_of_outputs),
    }
