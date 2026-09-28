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


def _quantizer_snapshot(quantized_module) -> list[tuple]:
    def qz(q):
        return (np.asarray(q.scale).tolist(), np.asarray(q.zero_point).tolist(), int(np.asarray(q.n_bits)), int(q.offset), bool(q.is_narrow), bool(q.no_clipping))

    return [qz(q) for q in list(quantized_module.input_quantizers) + list(quantized_module.output_quantizers)]


@needs_concrete
def test_restore_weights_with_a_calibration_subset_matches_full_data_exactly(data):
    """Decision D7 (docs/fhe_mlp.md): a calibration diagnostic on the real top_20_bits3/bits4 checkpoints found this
    re-export calibration-set-size INVARIANT -- Brevitas QAT's quantizer scale/zero-point are fixed, learned
    power-of-two values, not statistics derived from the calibration data's range. Reproduced here on the synthetic
    problem: a ~19% calibration subset gives BIT-IDENTICAL quantizer parameters and clear-quantized outputs to the
    full 2,000-row training set, from the SAME trained weights."""
    import copy

    mlp.seed_everything(1)
    model = mlp.build_qat_mlp(MLP_CFG, 4, mlp.balanced_class_weights(data["y_train"]))
    model.fit(data["X_train"], data["y_train"])
    state = copy.deepcopy(model.base_module.state_dict())

    mlp.restore_weights(model, state, data["X_train"], data["y_train"])
    full_logits = mlp.clear_quantized_logits(model, data["X_val"])
    full_snapshot = _quantizer_snapshot(model.quantized_module_)

    pos = np.where(data["y_train"] == 1)[0][:150]
    neg = np.where(data["y_train"] == 0)[0][:230]
    subset_idx = np.concatenate([pos, neg])
    assert subset_idx.shape[0] < data["X_train"].shape[0]  # a genuine subset, not the full set in disguise
    mlp.restore_weights(model, state, data["X_train"][subset_idx], data["y_train"][subset_idx])
    subset_logits = mlp.clear_quantized_logits(model, data["X_val"])
    subset_snapshot = _quantizer_snapshot(model.quantized_module_)

    assert full_snapshot == subset_snapshot
    np.testing.assert_array_equal(full_logits, subset_logits)


@needs_concrete
def test_train_qat_mlp_calibration_subset_matches_full_data_and_default_is_unchanged(data):
    """The optional X_calibration/y_calibration args (D7) change ONLY the final re-export's data; every epoch of
    actual gradient training (weights, seeds, history) is untouched, and omitting them reproduces the exact original
    full-data-everywhere behavior."""
    cfg = {**MLP_CFG, "max_epochs": 3}
    model_full, info_full = mlp.train_qat_mlp(cfg, 4, data["X_train"], data["y_train"], data["X_val"], data["y_val"], seed=11)

    pos = np.where(data["y_train"] == 1)[0][:150]
    neg = np.where(data["y_train"] == 0)[0][:400]
    cal_idx = np.concatenate([pos, neg])
    model_subset, info_subset = mlp.train_qat_mlp(
        cfg, 4, data["X_train"], data["y_train"], data["X_val"], data["y_val"], seed=11,
        X_calibration=data["X_train"][cal_idx], y_calibration=data["y_train"][cal_idx],
    )

    # identical training trajectory (same seed): X_calibration cannot affect any epoch of actual training (wall-clock
    # "seconds" is excluded -- it is timing noise, not a training outcome)
    history_full = [{k: v for k, v in h.items() if k != "seconds"} for h in info_full["history"]]
    history_subset = [{k: v for k, v in h.items() if k != "seconds"} for h in info_subset["history"]]
    assert history_full == history_subset and info_full["best_epoch"] == info_subset["best_epoch"]
    # the only place a calibration subset can matter -- the final clear-quantized function -- is bit-identical too
    np.testing.assert_array_equal(mlp.clear_quantized_logits(model_full, data["X_val"]), mlp.clear_quantized_logits(model_subset, data["X_val"]))
    assert info_full["calibration_rows_used"] == data["X_train"].shape[0] and info_full["calibration_is_full_train_set"] is True
    assert info_subset["calibration_rows_used"] == cal_idx.shape[0] and info_subset["calibration_is_full_train_set"] is False


# ---- decision D8: epoch 1 built via a manual skorch initialize()+fit(), bypassing Concrete-ML's own .fit() --------
#
# `_init_sklearn_model` depends on private Concrete-ML/skorch internals (documented in its own docstring in
# src/fhe/compile/mlp.py); these tests are also the tripwire for a future concrete-ml/skorch upgrade silently
# changing that plumbing -- a divergence here means "PRIVATE_API_DEPENDENCY_CHANGED", not a training-logic bug.


def _train_via_pre_d8_reference(mlp_cfg, n_bits, X_train, y_train, X_val, y_val, seed):
    """Reference re-implementation of the PRE-D8 trainer (epoch 1 via Concrete-ML's own `model.fit()`, which also
    performs the full-data export D8 removes), kept ONLY to prove `train_qat_mlp` (now using D8) produces an
    identical training trajectory. Deliberately does NOT reuse the (already D8-changed) production code, so a bug
    that broke both paths identically could not hide behind shared code."""
    import copy

    weights = mlp.balanced_class_weights(y_train)
    mlp.seed_everything(seed)
    model = mlp.build_qat_mlp(mlp_cfg, n_bits, weights)
    X_train = np.asarray(X_train, dtype=np.float32)
    X_val = np.asarray(X_val, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int64)

    history: list[float] = []
    best_pr, best_epoch, best_state, bad = -1.0, 0, None, 0
    for epoch in range(1, int(mlp_cfg["max_epochs"]) + 1):
        if epoch == 1:
            model.fit(X_train, y_train)  # the ORIGINAL, pre-D8 epoch-1 path (Concrete-ML's full wrapper)
        else:
            model.sklearn_model.partial_fit(X_train, y_train)
        pr = float(average_precision_score(y_val, mlp.fraud_probability(mlp.qat_forward_logits(model, X_val))))
        history.append(pr)
        if pr > best_pr + float(mlp_cfg.get("min_delta", 0.0)):
            best_pr, best_epoch, best_state, bad = pr, epoch, copy.deepcopy(model.base_module.state_dict()), 0
        else:
            bad += 1
            if bad >= int(mlp_cfg["early_stopping_patience"]):
                break
    return model, history, best_epoch, best_state


@needs_concrete
def test_d8_init_sklearn_model_is_deterministic_given_the_same_seed(data):
    """Prerequisite for everything below: `_init_sklearn_model` + skorch's own `.initialize()` (no training at all)
    must draw the SAME initial weights every time for the same seed."""
    import torch

    weights = mlp.balanced_class_weights(data["y_train"])

    def build():
        mlp.seed_everything(21)
        model = mlp.build_qat_mlp(MLP_CFG, 4, weights)
        mlp._init_sklearn_model(model, data["X_train"], data["y_train"])
        model.sklearn_model.initialize()  # builds module_/criterion_/optimizer_; consumes the RNG for weight init
        return model

    model_a, model_b = build(), build()
    sa, sb = model_a.base_module.state_dict(), model_b.base_module.state_dict()
    assert sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)


@needs_concrete
def test_d8_epoch1_state_dict_and_logits_match_the_pre_d8_reference_exactly(data):
    """The core D8 equivalence claim: with the SAME seed, epoch 1 built via `_init_sklearn_model` +
    `model.sklearn_model.fit()` (D8) ends at BIT-IDENTICAL weights and validation logits to the ORIGINAL epoch-1
    path (`model.fit()`, Concrete-ML's full wrapper, which also runs the expensive full-data export D8 bypasses)."""
    import torch

    weights = mlp.balanced_class_weights(data["y_train"])

    mlp.seed_everything(9)
    ref_model = mlp.build_qat_mlp(MLP_CFG, 4, weights)
    ref_model.fit(data["X_train"], data["y_train"])  # ORIGINAL, pre-D8 epoch 1

    mlp.seed_everything(9)
    new_model = mlp.build_qat_mlp(MLP_CFG, 4, weights)
    mlp._init_sklearn_model(new_model, data["X_train"], data["y_train"])
    new_model.sklearn_model.fit(data["X_train"], data["y_train"])  # D8 epoch 1

    sr, sn = ref_model.base_module.state_dict(), new_model.base_module.state_dict()
    assert sr.keys() == sn.keys() and all(torch.equal(sr[k], sn[k]) for k in sr)
    np.testing.assert_array_equal(mlp.qat_forward_logits(ref_model, data["X_val"]), mlp.qat_forward_logits(new_model, data["X_val"]))


@needs_concrete
def test_d8_epoch1_does_not_trigger_concrete_mls_full_data_export(data):
    """The whole point of D8: after `_init_sklearn_model` + `model.sklearn_model.fit()`, Concrete-ML's OWN `.fit()`
    (and therefore its unconditional full-data `quantize_module`, the operation a memory probe measured as the sole
    driver of the WSL VM's OOM) must never have run."""
    weights = mlp.balanced_class_weights(data["y_train"])
    mlp.seed_everything(4)
    model = mlp.build_qat_mlp(MLP_CFG, 4, weights)
    mlp._init_sklearn_model(model, data["X_train"], data["y_train"])
    model.sklearn_model.fit(data["X_train"], data["y_train"])

    assert model._is_fitted is False  # Concrete-ML's own .fit() sets this True; it never ran here
    assert len(model.quantized_module_.input_quantizers) == 0  # never quantized -- quantize_module(X) never called


@needs_concrete
def test_d8_full_multi_epoch_trajectory_and_restored_model_match_the_pre_d8_reference_exactly(data):
    """End-to-end equivalence: the full trainer (`train_qat_mlp`, now using D8's epoch-1 path) must reproduce the
    SAME epoch-by-epoch training trajectory, the SAME best epoch, and -- after `restore_weights` -- BIT-IDENTICAL
    quantizer parameters and clear-quantized outputs as the original pre-D8 trainer. `early_stopping_patience` is
    set to never trigger, for the strictest possible trajectory comparison (the full epoch budget, every epoch)."""
    cfg = {**MLP_CFG, "max_epochs": 6, "early_stopping_patience": 6}
    seed = 23

    ref_model, ref_history, ref_best_epoch, ref_best_state = _train_via_pre_d8_reference(
        cfg, 4, data["X_train"], data["y_train"], data["X_val"], data["y_val"], seed
    )
    mlp.restore_weights(ref_model, ref_best_state, data["X_train"], data["y_train"])  # pre-D7/D8 style: full-data restore

    new_model, info = mlp.train_qat_mlp(cfg, 4, data["X_train"], data["y_train"], data["X_val"], data["y_val"], seed)
    # X_calibration omitted -> defaults to the full training set (D7's default), matching ref_model's restore above
    # exactly, so this test isolates D8's effect only.

    new_history = [h["val_pr_auc_qat_forward"] for h in info["history"]]
    assert ref_history == new_history and ref_best_epoch == info["best_epoch"]  # identical trajectory, identical best epoch

    assert _quantizer_snapshot(ref_model.quantized_module_) == _quantizer_snapshot(new_model.quantized_module_)
    np.testing.assert_array_equal(
        mlp.clear_quantized_logits(ref_model, data["X_val"]), mlp.clear_quantized_logits(new_model, data["X_val"])
    )


@needs_concrete
def test_the_returned_model_is_the_best_epoch_and_reports_its_clear_quantized_score(data):
    """Whatever epoch was best, the model handed back must BE that epoch: its clear-quantized validation PR-AUC equals the
    recorded final score exactly and tracks the best QAT-forward score that chose the epoch (a proxy, not bit-identical)."""
    cfg = {**MLP_CFG, "max_epochs": 6, "early_stopping_patience": 1, "lr": 0.05}
    model, info = mlp.train_qat_mlp(cfg, 4, data["X_train"], data["y_train"], data["X_val"], data["y_val"], seed=5)
    p = mlp.fraud_probability(mlp.clear_quantized_logits(model, data["X_val"]))
    assert average_precision_score(data["y_val"], p) == pytest.approx(info["final_val_pr_auc_clear_quantized"], abs=1e-12)
    assert abs(info["final_val_pr_auc_clear_quantized"] - info["best_val_pr_auc_qat_forward"]) < 0.05
    assert info["best_epoch"] == int(np.argmax([h["val_pr_auc_qat_forward"] for h in info["history"]])) + 1  # earliest maximum

    # the QAT forward at the returned weights equals the QAT forward that was recorded at the best epoch
    fwd = mlp.fraud_probability(mlp.qat_forward_logits(model, data["X_val"]))
    assert average_precision_score(data["y_val"], fwd) == pytest.approx(info["best_val_pr_auc_qat_forward"], abs=1e-9)


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
