"""Phase 6 FHE XGBoost tests against a tiny SYNTHETIC tree model.

Concrete-ML has no Windows wheels (`docs/environment.md`), so every test is
`skipif`-gated and only runs under the WSL2 FHE venv. The toy booster is
early-stopped so it stores trees beyond `best_iteration`, which is what the
tree-count tests depend on. `tests/test_fhe_xgb_poc_real_data.py` is the
real-handoff counterpart.
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


def _is_concrete_ml_available() -> bool:
    try:
        return importlib.util.find_spec("concrete.ml") is not None
    except ModuleNotFoundError:
        return False


pytestmark = pytest.mark.skipif(
    not _is_concrete_ml_available(),
    reason=(
        "concrete-ml is not installed in this environment. It has no Windows "
        "wheels; run this test under the WSL2 venv (~/.venvs/fhe-fraud-detection). "
        "See docs/environment.md."
    ),
)

N_BITS = 6


@pytest.fixture(scope="module")
def toy(tmp_path_factory):
    import xgboost as xgb

    rng = np.random.RandomState(0)
    n = 600
    X = np.column_stack([rng.rand(n) * 100, rng.rand(n), rng.randn(n), rng.rand(n) * 5]).astype(np.float64)
    y = ((X[:, 0] / 100 + X[:, 1] + 0.3 * rng.rand(n)) > 1.1).astype(int)
    X_train, y_train, X_val, y_val = X[:400], y[:400], X[400:], y[400:]

    model = xgb.XGBClassifier(
        n_estimators=60, max_depth=2, learning_rate=0.5, early_stopping_rounds=5,
        eval_metric="aucpr", random_state=0, n_jobs=1,
    )
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    booster = model.get_booster()
    best = int(booster.attr("best_iteration"))
    assert booster.num_boosted_rounds() > best + 1, "toy booster must store trees beyond best_iteration"

    base = tmp_path_factory.mktemp("toy_xgb")
    full_path, sliced_path = base / "full.json", base / "sliced.json"
    booster.save_model(full_path)
    booster[0 : best + 1].save_model(sliced_path)
    return {
        "X_train": X_train, "X_val": X_val, "y_val": y_val, "model": model,
        "full_path": full_path, "sliced_path": sliced_path,
        "n_stored": booster.num_boosted_rounds(), "n_inference": best + 1,
    }


@pytest.fixture(scope="module")
def compiled(toy):
    from src.fhe.compile.linear import compile_model
    from src.fhe.compile.tree import build_concrete_xgb, load_inference_classifier

    clf = load_inference_classifier(toy["sliced_path"])
    cml_model = build_concrete_xgb(clf, toy["X_train"], N_BITS)
    circuit, compile_seconds = compile_model(cml_model, toy["X_train"])
    return clf, cml_model, circuit, compile_seconds


def test_concrete_compiles_every_stored_tree_ignoring_best_iteration(toy):
    """Regression lock for the Phase 6 finding: Concrete-ML converts ALL trees stored in
    a booster, so an unsliced early-stopped booster would compile a different model
    than the one xgboost's predict_proba uses."""
    from src.fhe.compile.tree import build_concrete_xgb, load_inference_classifier

    for path, expected in ((toy["full_path"], toy["n_stored"]), (toy["sliced_path"], toy["n_inference"])):
        cml_model = build_concrete_xgb(load_inference_classifier(path), toy["X_train"], N_BITS)
        out = cml_model._inference(cml_model.quantize_input(toy["X_val"][:3]))
        assert out.shape[-1] == expected


def test_sliced_booster_reproduces_early_stopped_model_exactly(toy):
    from src.fhe.compile.tree import load_inference_classifier

    clf = load_inference_classifier(toy["sliced_path"])
    np.testing.assert_array_equal(clf.predict_proba(toy["X_val"])[:, 1], toy["model"].predict_proba(toy["X_val"])[:, 1])


def test_load_inference_classifier_rejects_non_binary_objective(tmp_path):
    import xgboost as xgb

    from src.fhe.compile.tree import TreeCompileError, load_inference_classifier

    rng = np.random.RandomState(1)
    X, y = rng.rand(90, 3), np.repeat([0, 1, 2], 30)
    multi = xgb.XGBClassifier(n_estimators=3, max_depth=2, objective="multi:softprob").fit(X, y)
    path = tmp_path / "multi.json"
    multi.save_model(path)
    with pytest.raises(TreeCompileError, match="objective"):
        load_inference_classifier(path)


def test_compile_produces_tree_circuit_with_stats(toy, compiled):
    from src.fhe.compile.linear import circuit_stats
    from src.fhe.compile.tree import tree_stats

    clf, cml_model, circuit, compile_seconds = compiled
    stats = {**circuit_stats(circuit), **tree_stats(clf, cml_model, circuit)}
    assert compile_seconds > 0
    # Unlike Phase 5's LR circuit, tree comparisons and leaf lookups need programmable bootstraps.
    assert stats["programmable_bootstrap_count"] > 0
    assert stats["n_trees"] == toy["n_inference"]
    assert stats["n_bits_inputs"] == [N_BITS]
    assert stats["max_integer_bit_width"] > 0


def test_quantize_input_parameters_are_unchanged_by_compile(toy):
    """CLAUDE.md Sec.11: client-side quantization must match the compiled model."""
    from src.fhe.compile.linear import compile_model
    from src.fhe.compile.tree import build_concrete_xgb, load_inference_classifier

    cml_model = build_concrete_xgb(load_inference_classifier(toy["sliced_path"]), toy["X_train"], N_BITS)
    q_before = cml_model.quantize_input(toy["X_val"][:20])
    compile_model(cml_model, toy["X_train"])
    np.testing.assert_array_equal(q_before, cml_model.quantize_input(toy["X_val"][:20]))


def test_chunked_disable_integers_match_single_batch(toy, compiled):
    from src.fhe.xgb_poc import disable_integers

    _, cml_model, _, _ = compiled
    q = cml_model.quantize_input(toy["X_val"])
    np.testing.assert_array_equal(disable_integers(cml_model, q, chunk_size=17), cml_model._inference(q))


def test_simulate_integers_match_disable_exactly(toy, compiled, tmp_path):
    from src.fhe.validate.correctness import integer_output_check
    from src.fhe.xgb_poc import disable_integers, simulate_integers

    _, cml_model, circuit, _ = compiled
    q = cml_model.quantize_input(toy["X_val"][:30])
    sim_q, seconds_per_row = simulate_integers(circuit, q, tmp_path / "ckpt", "fp", chunk_size=8)
    assert integer_output_check(sim_q, disable_integers(cml_model, q, 1000))["passed"] is True
    assert seconds_per_row > 0


def test_simulate_checkpoints_are_reused_and_invalidated_by_fingerprint(toy, compiled, tmp_path):
    from src.fhe.xgb_poc import simulate_integers

    _, cml_model, circuit, _ = compiled
    q = cml_model.quantize_input(toy["X_val"][:10])
    ckpt = tmp_path / "ckpt"
    first, _ = simulate_integers(circuit, q, ckpt, "fp-a", chunk_size=4)
    assert len(list(ckpt.glob("chunk_*.npy"))) == 3

    resumed, seconds_per_row = simulate_integers(circuit, q, ckpt, "fp-a", chunk_size=4)
    np.testing.assert_array_equal(resumed, first)
    assert np.isnan(seconds_per_row)  # nothing re-simulated

    tampered = np.load(sorted(ckpt.glob("chunk_*.npy"))[0])
    tampered[0] += 1
    np.save(sorted(ckpt.glob("chunk_*.npy"))[0], tampered)
    with pytest.raises(RuntimeError, match="disagrees"):
        simulate_integers(circuit, q, ckpt, "fp-a", chunk_size=4)

    fresh, _ = simulate_integers(circuit, q, ckpt, "fp-b", chunk_size=4)  # new fingerprint discards stale chunks
    np.testing.assert_array_equal(fresh, first)
    assert json.loads((ckpt / "meta.json").read_text())["fingerprint"] == "fp-b"


def test_explicit_round_trip_integers_match_simulate(toy, compiled):
    """T1 on the toy circuit: real keygen -> encrypt -> run -> decrypt."""
    from src.fhe.validate.correctness import integer_output_check
    from src.fhe.xgb_poc import explicit_round_trip

    _, cml_model, circuit, _ = compiled
    q = cml_model.quantize_input(toy["X_val"][:2])
    sim_q = np.array([circuit.simulate(q[i : i + 1])[0] for i in range(q.shape[0])])
    decrypted_q, timing = explicit_round_trip(circuit, q)
    assert integer_output_check(decrypted_q, sim_q)["passed"] is True
    assert timing["mean_run_seconds"] > 0


def test_explicit_round_trip_checkpoints_are_reused_and_invalidated_by_fingerprint(toy, compiled, tmp_path):
    """Same real motivation as the T2 simulate-checkpoint test: a real row's `run`
    step is minutes long on the committed tiers (docs/fhe_xgboost.md), so a resumed
    T1 run must not redo an already-executed row."""
    from src.fhe.xgb_poc import explicit_round_trip

    _, cml_model, circuit, _ = compiled
    q = cml_model.quantize_input(toy["X_val"][:2])
    ckpt = tmp_path / "ckpt"

    first, timing_a = explicit_round_trip(circuit, q, ckpt, "fp-a")
    assert timing_a["mean_run_seconds"] > 0
    assert sorted(p.name for p in ckpt.glob("row_*.npy")) == ["row_0000.npy", "row_0001.npy"]

    resumed, timing_b = explicit_round_trip(circuit, q, ckpt, "fp-a")
    np.testing.assert_array_equal(resumed, first)
    assert timing_b["keygen_seconds"] >= 0  # keygen is always redone; rows are reused

    tampered = np.load(ckpt / "row_0000.npy")
    tampered[0] += 1
    np.save(ckpt / "row_0000.npy", tampered)
    with pytest.raises(RuntimeError, match="disagrees"):
        explicit_round_trip(circuit, q, ckpt, "fp-a")

    fresh, _ = explicit_round_trip(circuit, q, ckpt, "fp-b")  # new fingerprint discards stale rows
    np.testing.assert_array_equal(fresh, first)
    assert json.loads((ckpt / "meta.json").read_text())["fingerprint"] == "fp-b"


def test_calibration_positions_include_every_feature_extreme():
    from src.fhe.xgb_poc import calibration_positions

    rng = np.random.RandomState(3)
    X = rng.randn(500, 4)
    pos = calibration_positions(X, n_rows=20, include_extremes=True, seed=42)
    np.testing.assert_array_equal(X[pos].min(axis=0), X.min(axis=0))
    np.testing.assert_array_equal(X[pos].max(axis=0), X.max(axis=0))
    assert len(pos) <= 20 + 2 * X.shape[1]
    np.testing.assert_array_equal(pos, calibration_positions(X, n_rows=20, include_extremes=True, seed=42))
    assert len(calibration_positions(X, n_rows="all", include_extremes=True, seed=42)) == 500
