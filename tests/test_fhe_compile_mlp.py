"""Phase 9 MLP build/train/compile tests.

Two groups. The pure-numpy helpers (`fraud_probability`, `balanced_class_weights`, `client_quantize`) run in every
environment. Everything that trains or compiles a network is `skipif`-gated on Concrete-ML (WSL2 FHE venv only,
`docs/environment.md`) and uses a tiny synthetic problem -- never the real data.
"""

from __future__ import annotations

import importlib.util
from types import SimpleNamespace

import numpy as np
import pytest
from sklearn.metrics import average_precision_score
from sklearn.utils.class_weight import compute_class_weight

from src.fhe.compile import mlp


def _concrete_available() -> bool:
    try:
        return importlib.util.find_spec("concrete.ml") is not None
    except ModuleNotFoundError:
        return False


needs_concrete = pytest.mark.skipif(
    not _concrete_available(),
    reason="concrete-ml is not installed in this environment. It has no Windows wheels; run under the WSL2 venv. See docs/environment.md.",
)

MLP_CFG = {"n_layers": 3, "n_hidden_neurons_multiplier": 2, "n_accum_bits": 16, "lr": 0.01, "batch_size": 256, "max_epochs": 4, "early_stopping_patience": 2, "min_delta": 0.0}


# ---- pure numpy -------------------------------------------------------------------------------------------------


def test_fraud_probability_is_softmax_of_the_second_class_and_stable():
    logits = np.array([[0.0, 0.0], [1.0, 3.0], [3.0, 1.0], [-2000.0, 2000.0], [2000.0, -2000.0]])
    p = mlp.fraud_probability(logits)
    expected = np.exp(logits[:3, 1]) / np.exp(logits[:3]).sum(axis=1)
    np.testing.assert_allclose(p[:3], expected, rtol=1e-12)
    assert p[3] == 1.0 and p[4] == 0.0 and np.isfinite(p).all()  # saturates without overflow warnings


def test_fraud_probability_is_not_the_library_elementwise_sigmoid():
    """Concrete-ML applies sigmoid to each logit for two classes (rows do not sum to 1); ours is a proper probability."""
    logits = np.array([[1.0, 2.0]])
    elementwise_second = 1.0 / (1.0 + np.exp(-logits[:, 1]))
    assert mlp.fraud_probability(logits)[0] != pytest.approx(elementwise_second[0])


def test_balanced_class_weights_equal_sklearns_and_use_train_labels_only():
    y = np.array([0] * 90 + [1] * 10)
    np.testing.assert_allclose(mlp.balanced_class_weights(y), compute_class_weight("balanced", classes=np.array([0, 1]), y=y))
    with pytest.raises(ValueError, match="both classes"):
        mlp.balanced_class_weights(np.zeros(5, dtype=int))


def test_client_quantize_clips_to_the_quantizer_range():
    quantizer = SimpleNamespace(scale=np.array(0.5), zero_point=np.array(0), n_bits=np.array(4.0), offset=8, is_narrow=False, no_clipping=False)
    q = mlp.client_quantize(quantizer, np.array([0.0, 1.0, -1.26, 3.4, 100.0, -100.0]))
    np.testing.assert_array_equal(q, [0, 2, -3, 7, 7, -8])
    narrow = SimpleNamespace(**{**quantizer.__dict__, "is_narrow": True})
    assert mlp.client_quantize(narrow, np.array([-100.0]))[0] == -7


# ---- Concrete-ML (WSL2) -----------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(3000, 12)).astype(np.float32)
    w = rng.normal(size=12)
    y = ((X @ w + rng.normal(scale=1.5, size=3000)) > 1.6).astype(np.int64)
    return {"X_train": X[:2000], "y_train": y[:2000], "X_val": X[2000:], "y_val": y[2000:]}


@pytest.fixture(scope="module")
def trained(data):
    model, info = mlp.train_qat_mlp(MLP_CFG, 4, data["X_train"], data["y_train"], data["X_val"], data["y_val"], seed=7)
    return model, info


@needs_concrete
def test_training_learns_and_records_history(trained, data):
    model, info = trained
    p = mlp.fraud_probability(mlp.clear_quantized_logits(model, data["X_val"]))
    assert average_precision_score(data["y_val"], p) > data["y_val"].mean() + 0.1  # well above the chance level
    assert 1 <= info["epochs_run"] <= MLP_CFG["max_epochs"] and len(info["history"]) == info["epochs_run"]
    assert info["class_weights"] == pytest.approx(mlp.balanced_class_weights(data["y_train"]))


@needs_concrete
def test_seeded_training_is_reproducible(data):
    a, _ = mlp.train_qat_mlp({**MLP_CFG, "max_epochs": 2}, 4, data["X_train"], data["y_train"], data["X_val"], data["y_val"], seed=3)
    b, _ = mlp.train_qat_mlp({**MLP_CFG, "max_epochs": 2}, 4, data["X_train"], data["y_train"], data["X_val"], data["y_val"], seed=3)
    np.testing.assert_array_equal(mlp.clear_quantized_logits(a, data["X_val"]), mlp.clear_quantized_logits(b, data["X_val"]))


@needs_concrete
def test_restore_weights_rebuilds_the_quantized_module_from_the_saved_state(data):
    """The early-stopping mechanism: weights saved after epoch 1, training continued, then restored -> the clear-quantized
    outputs equal those right after epoch 1 exactly."""
    import copy

    mlp.seed_everything(1)
    model = mlp.build_qat_mlp(MLP_CFG, 4, mlp.balanced_class_weights(data["y_train"]))
    model.fit(data["X_train"], data["y_train"])
    state = copy.deepcopy(model.base_module.state_dict())
    after_epoch_1 = mlp.clear_quantized_logits(model, data["X_val"])
    model.fit(data["X_train"], data["y_train"])
    model.fit(data["X_train"], data["y_train"])
    assert not np.array_equal(mlp.clear_quantized_logits(model, data["X_val"]), after_epoch_1)  # training really moved the weights
    mlp.restore_weights(model, state, data["X_train"], data["y_train"])
    np.testing.assert_array_equal(mlp.clear_quantized_logits(model, data["X_val"]), after_epoch_1)


@needs_concrete
def test_the_returned_model_scores_its_recorded_best_validation_pr_auc(data):
    """Whatever epoch was best, the model handed back must BE that epoch (restored when it was not the last)."""
    cfg = {**MLP_CFG, "max_epochs": 6, "early_stopping_patience": 1, "lr": 0.05}
    model, info = mlp.train_qat_mlp(cfg, 4, data["X_train"], data["y_train"], data["X_val"], data["y_val"], seed=5)
    p = mlp.fraud_probability(mlp.clear_quantized_logits(model, data["X_val"]))
    assert average_precision_score(data["y_val"], p) == pytest.approx(info["best_val_pr_auc_clear_quantized"], abs=1e-12)


@needs_concrete
def test_float_twin_has_the_same_architecture_and_is_trained_the_same_way(trained, data):
    model, _ = trained
    net, info = mlp.train_float_twin(MLP_CFG, data["X_train"], data["y_train"], data["X_val"], data["y_val"], seed=7)
    qat_shapes = [tuple(p.shape) for n, p in model.base_module.named_parameters() if n.endswith("weight") and p.ndim == 2]
    float_shapes = [tuple(p.shape) for n, p in net.named_parameters() if n.endswith("weight")]
    assert qat_shapes == float_shapes == [(24, 12), (24, 24), (2, 24)]
    assert info["class_weights"] == pytest.approx(mlp.balanced_class_weights(data["y_train"]))
    p = mlp.fraud_probability(mlp.float_logits(net, data["X_val"]))
    assert average_precision_score(data["y_val"], p) == pytest.approx(info["best_val_pr_auc_float"], abs=1e-7)


@needs_concrete
def test_compile_logs_tlu_and_key_statistics(trained, data):
    model, _ = trained
    circuit, seconds = mlp.compile_mlp(model, data["X_train"][:500])
    stats = mlp.mlp_stats(model, circuit, MLP_CFG, 4)
    assert seconds > 0 and stats["programmable_bootstrap_count"] > 0  # an MLP needs table lookups (LR needs none)
    assert stats["hidden_neurons"] == 48 and stats["layer_shapes"] == [[24, 12], [24, 24], [2, 24]]
    assert stats["n_bits_inputs"] == [4] and stats["max_integer_bit_width"] <= 16
    assert stats["size_of_bootstrap_keys_bytes"] > 0 and 0.0 < stats["p_error"] < 1e-6


@needs_concrete
def test_client_side_quantization_equals_the_compiled_models_quantize_input(trained, data):
    """CLAUDE.md Sec.11: quantization parameters must be consistent between the client and the compiled model."""
    model, _ = trained
    mlp.compile_mlp(model, data["X_train"][:500])
    quantizer = model.quantized_module_.input_quantizers[0]
    X = np.concatenate([data["X_val"], data["X_val"] * 25.0])  # includes values far outside the calibration range
    np.testing.assert_array_equal(mlp.client_quantize(quantizer, X), model.quantize_input(X))


@needs_concrete
def test_clear_integers_equal_simulate_and_probabilities_reproduce_from_them(trained, data):
    model, _ = trained
    circuit, _ = mlp.compile_mlp(model, data["X_train"][:500])
    q = model.quantize_input(data["X_val"][:40])
    clear = np.asarray(model.quantized_module_.quantized_forward(q, fhe="disable"))
    simulated = np.array([circuit.simulate(q[i : i + 1])[0] for i in range(q.shape[0])])
    np.testing.assert_array_equal(clear, simulated)
    np.testing.assert_array_equal(
        mlp.fraud_probability(model.dequantize_output(simulated)), mlp.fraud_probability(mlp.clear_quantized_logits(model, data["X_val"][:40]))
    )


@needs_concrete
def test_dump_load_before_compile_reproduces_clear_outputs_exactly_and_still_compiles(data):
    """T0 for the MLP: the checkpoint written after training reloads to exactly the same clear-quantized function."""
    model, _ = mlp.train_qat_mlp({**MLP_CFG, "max_epochs": 2}, 4, data["X_train"], data["y_train"], data["X_val"], data["y_val"], seed=2)
    reloaded = mlp.load_model(mlp.dump_model(model))
    np.testing.assert_array_equal(mlp.clear_quantized_logits(reloaded, data["X_val"]), mlp.clear_quantized_logits(model, data["X_val"]))
    circuit_a, _ = mlp.compile_mlp(model, data["X_train"][:500])
    circuit_b, _ = mlp.compile_mlp(reloaded, data["X_train"][:500])
    q = model.quantize_input(data["X_val"][:10])
    np.testing.assert_array_equal(np.array([circuit_a.simulate(q[i : i + 1])[0] for i in range(10)]), np.array([circuit_b.simulate(q[i : i + 1])[0] for i in range(10)]))
