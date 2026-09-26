"""Phase 5 FHE PoC tests against a tiny SYNTHETIC model -- Concrete-ML has
no Windows wheels (`docs/environment.md`), so every test here is
`skipif`-gated on it being importable, and only runs for real under the
WSL2 FHE venv (`~/.venvs/fhe-fraud-detection`). On Windows, this whole file
skips, matching `tests/test_fhe_environment_smoke.py`'s existing pattern.

`tests/test_fhe_lr_poc_real_data.py` is the real-handoff counterpart.
"""

import importlib.util

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def _is_concrete_ml_available() -> bool:
    try:
        return importlib.util.find_spec("concrete.ml") is not None
    except ModuleNotFoundError:
        return False


concrete_ml_available = _is_concrete_ml_available()
pytestmark = pytest.mark.skipif(
    not concrete_ml_available,
    reason=(
        "concrete-ml is not installed in this environment. It has no Windows "
        "wheels; run this test under the WSL2 venv (~/.venvs/fhe-fraud-detection). "
        "See docs/environment.md."
    ),
)


@pytest.fixture
def toy_pipeline_and_params():
    from src.fhe.handoff import extract_lr_pipeline_params

    rng = np.random.RandomState(0)
    n, d = 400, 5
    X = rng.randn(n, d) * np.array([1.0, 50.0, 0.01, 200.0, 3.0])
    y = (X[:, 0] + X[:, 1] / 50.0 > 0).astype(int)
    pipeline = Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression(C=1.0, max_iter=500))])
    pipeline.fit(X, y)

    X_scaled = pipeline.named_steps["scaler"].transform(X).astype(np.float64)
    params = extract_lr_pipeline_params(pipeline)
    return params, X_scaled, y


def test_build_concrete_lr_from_exported_params_predicts_reasonably(toy_pipeline_and_params):
    """from_sklearn_model on a param-RECONSTRUCTED estimator (never the
    original fitted Pipeline object) -- the whole point of the parameter
    handoff (src/fhe/handoff.py)."""
    from src.fhe.compile.linear import build_concrete_lr

    params, X_scaled, y = toy_pipeline_and_params
    cml_model = build_concrete_lr(params, X_scaled, n_bits=8)
    prob = cml_model.predict_proba(X_scaled, fhe="disable")[:, 1]
    # Sanity: the quantized model should still separate the classes well
    # on data this clearly separable -- not a tight numeric check (that's
    # T3's job on real data), just a build/predict smoke test.
    from sklearn.metrics import roc_auc_score

    assert roc_auc_score(y, prob) > 0.9


def test_compile_produces_a_circuit_with_stats(toy_pipeline_and_params):
    from src.fhe.compile.linear import build_concrete_lr, circuit_stats, compile_model

    params, X_scaled, _ = toy_pipeline_and_params
    cml_model = build_concrete_lr(params, X_scaled, n_bits=8)
    circuit, compile_seconds = compile_model(cml_model, X_scaled)
    stats = circuit_stats(circuit)

    assert compile_seconds > 0
    # A linear model's affine arithmetic needs no programmable bootstrap --
    # the sigmoid/threshold run client-side, post-decryption
    # (src/fhe/compile/linear.py module docstring). Asserted, not assumed:
    # if this ever becomes nonzero, T1 (exact-match requirement) would
    # start failing for a real cryptographic reason (p_error), which is
    # exactly the kind of environment surprise Phase 5 exists to surface.
    assert stats["programmable_bootstrap_count"] == 0
    assert stats["size_of_secret_keys_bytes"] > 0


def test_quantize_input_parameters_are_unchanged_by_compile(toy_pipeline_and_params):
    """Quantization-parameter consistency between the client path and the
    compiled model (CLAUDE.md Sec.11's required test): the client-side
    `quantize_input` scale/zero-point is fixed at `from_sklearn_model`
    time, and `compile()` must not silently change it -- verified by
    quantizing the SAME input before and after compiling and requiring
    bit-for-bit identical integers."""
    from src.fhe.compile.linear import build_concrete_lr, compile_model

    params, X_scaled, _ = toy_pipeline_and_params
    cml_model = build_concrete_lr(params, X_scaled, n_bits=8)
    q_before = cml_model.quantize_input(X_scaled[:20])
    compile_model(cml_model, X_scaled)
    q_after = cml_model.quantize_input(X_scaled[:20])
    np.testing.assert_array_equal(q_before, q_after)


def test_simulate_matches_disable_exactly(toy_pipeline_and_params):
    """T2 on a synthetic model: fhe='simulate' vs. fhe='disable' must be
    bit-for-bit identical for a circuit with zero programmable bootstraps."""
    from src.fhe.compile.linear import build_concrete_lr, compile_model
    from src.fhe.validate.correctness import t2_simulation_check

    params, X_scaled, _ = toy_pipeline_and_params
    cml_model = build_concrete_lr(params, X_scaled, n_bits=8)
    compile_model(cml_model, X_scaled)

    disable_prob = cml_model.predict_proba(X_scaled, fhe="disable")[:, 1]
    simulate_prob = cml_model.predict_proba(X_scaled, fhe="simulate")[:, 1]
    result = t2_simulation_check(simulate_prob, disable_prob)
    assert result["passed"] is True


def test_explicit_round_trip_matches_simulate_exactly(toy_pipeline_and_params):
    """T1, the PoC's central exit criterion: a REAL keygen->encrypt->run->
    decrypt round trip, done explicitly (not via predict_proba's
    fhe='execute' convenience path), must match fhe='simulate' exactly."""
    from src.fhe.compile.linear import build_concrete_lr, compile_model
    from src.fhe.poc import _explicit_round_trip
    from src.fhe.validate.correctness import t1_execution_check

    params, X_scaled, _ = toy_pipeline_and_params
    cml_model = build_concrete_lr(params, X_scaled, n_bits=8)
    circuit, _ = compile_model(cml_model, X_scaled)

    X_sample = X_scaled[:5]
    simulate_prob = cml_model.predict_proba(X_sample, fhe="simulate")[:, 1]
    decrypted_prob, timing = _explicit_round_trip(cml_model, circuit, X_sample)

    result = t1_execution_check(decrypted_prob, simulate_prob)
    assert result["passed"] is True
    assert timing["keygen_seconds"] > 0
    assert timing["mean_row_seconds"] > 0


@pytest.fixture
def toy_raw_features_pipeline():
    """The REAL handoff contract: RAW (unscaled) features plus the exported
    pipeline parameters -- unlike `toy_pipeline_and_params`, which pre-scales."""
    from src.fhe.handoff import extract_lr_pipeline_params

    rng = np.random.RandomState(3)
    n, d = 600, 5
    X = rng.randn(n, d) * np.array([1.0, 50.0, 0.01, 200.0, 3.0]) + np.array([0.0, 20.0, 0.5, -300.0, 1.0])
    y = (X[:, 0] + (X[:, 1] - 20.0) / 50.0 > 0).astype(int)
    pipeline = Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression(C=1.0, max_iter=500))])
    pipeline.fit(X, y)
    return pipeline, extract_lr_pipeline_params(pipeline), X


def test_compiled_lr_built_on_standardized_inputs_matches_the_pipeline(toy_raw_features_pipeline):
    """Regression lock for the Phase 8 audit finding (docs/research.md, LR scaler erratum):
    starting from RAW features, the compiled model -- built, calibrated and
    evaluated on `standardize_features` output, as src/fhe/poc.py and
    src/benchmark/run.py now do -- must agree with the plaintext pipeline up to
    quantization noise. Before the fix this comparison was never made against
    the real raw-feature handoff contract."""
    from src.fhe.compile.linear import build_concrete_lr
    from src.fhe.handoff import standardize_features

    pipeline, params, X = toy_raw_features_pipeline
    X_model = standardize_features(params, X)
    cml_model = build_concrete_lr(params, X_model, n_bits=8)
    prob = cml_model.predict_proba(X_model, fhe="disable")[:, 1]
    reference = pipeline.predict_proba(X)[:, 1]
    assert np.mean((prob >= 0.5) == (reference >= 0.5)) >= 0.9
    assert np.mean(np.abs(prob - reference)) < 0.1


def test_compiled_lr_built_on_raw_inputs_does_not_match_the_pipeline(toy_raw_features_pipeline):
    """Negative control: the pre-fix behaviour (raw features straight into a
    model holding standardized-space coefficients) must NOT pass the check
    above, so that check demonstrably has teeth."""
    from src.fhe.compile.linear import build_concrete_lr

    pipeline, params, X = toy_raw_features_pipeline
    cml_model = build_concrete_lr(params, X, n_bits=8)
    prob = cml_model.predict_proba(X, fhe="disable")[:, 1]
    reference = pipeline.predict_proba(X)[:, 1]
    assert np.mean(np.abs(prob - reference)) > 0.1
