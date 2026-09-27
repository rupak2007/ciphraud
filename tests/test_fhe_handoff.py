"""Tests for src/fhe/handoff.py -- runs in both the Windows and WSL FHE
venvs (numpy + stdlib only), since the handoff format itself must be
readable/writable identically in both."""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.fhe.handoff import (
    HandoffError,
    extract_lr_pipeline_params,
    load_handoff,
    rebuild_pipeline_predict_proba,
    save_handoff,
    standardize_features,
)


@pytest.fixture
def fitted_pipeline():
    rng = np.random.RandomState(0)
    X = rng.randn(300, 5) * np.array([1.0, 50.0, 0.01, 200.0, 3.0])  # deliberately unscaled
    y = (X[:, 0] + X[:, 1] / 50.0 > 0).astype(int)
    pipeline = Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression(C=2.0, max_iter=500))])
    pipeline.fit(X, y)
    return pipeline, X, y


def test_extract_lr_pipeline_params_shapes(fitted_pipeline):
    pipeline, X, _ = fitted_pipeline
    params = extract_lr_pipeline_params(pipeline)
    assert params["n_features"] == X.shape[1]
    assert len(params["scaler_mean"]) == X.shape[1]
    assert len(params["scaler_scale"]) == X.shape[1]
    assert len(params["lr_coef"][0]) == X.shape[1]
    assert params["lr_C"] == 2.0


def test_rebuild_pipeline_predict_proba_matches_original_exactly(fitted_pipeline):
    pipeline, X, _ = fitted_pipeline
    params = extract_lr_pipeline_params(pipeline)
    original = pipeline.predict_proba(X)
    rebuilt = rebuild_pipeline_predict_proba(params, X)
    np.testing.assert_allclose(original, rebuilt, atol=1e-12)


def test_rebuild_pipeline_predict_proba_matches_on_unseen_rows(fitted_pipeline):
    """Not just the training rows -- the val-shaped use case."""
    pipeline, X, _ = fitted_pipeline
    rng = np.random.RandomState(1)
    X_new = rng.randn(50, X.shape[1]) * np.array([1.0, 50.0, 0.01, 200.0, 3.0])
    params = extract_lr_pipeline_params(pipeline)
    np.testing.assert_allclose(pipeline.predict_proba(X_new), rebuild_pipeline_predict_proba(params, X_new), atol=1e-12)


def test_save_and_load_handoff_round_trip(fitted_pipeline, tmp_path):
    pipeline, X, y = fitted_pipeline
    columns = [f"f{i}" for i in range(X.shape[1])]
    ref_prob = pipeline.predict_proba(X[:50])[:, 1]

    npz_path = tmp_path / "handoff.npz"
    manifest_path = tmp_path / "manifest.json"
    save_handoff(
        npz_path, manifest_path,
        X_train=X[50:], X_val=X[:50], y_val=y[:50],
        val_transaction_ids=np.arange(50), reference_val_prob=ref_prob,
        columns=columns, manifest_extra={"tier": "top_20"},
    )

    manifest, arrays = load_handoff(npz_path, manifest_path)
    assert manifest["columns"] == columns
    assert manifest["n_features"] == X.shape[1]
    assert manifest["tier"] == "top_20"
    np.testing.assert_array_equal(arrays["X_val"], X[:50])
    np.testing.assert_array_equal(arrays["y_val"], y[:50])
    np.testing.assert_allclose(arrays["reference_val_prob"], ref_prob)


def test_load_handoff_rejects_missing_files(tmp_path):
    with pytest.raises(HandoffError, match="not found"):
        load_handoff(tmp_path / "missing.npz", tmp_path / "missing.json")


def test_load_handoff_rejects_tampered_npz(fitted_pipeline, tmp_path):
    pipeline, X, y = fitted_pipeline
    columns = [f"f{i}" for i in range(X.shape[1])]
    npz_path = tmp_path / "handoff.npz"
    manifest_path = tmp_path / "manifest.json"
    save_handoff(
        npz_path, manifest_path,
        X_train=X[50:], X_val=X[:50], y_val=y[:50],
        val_transaction_ids=np.arange(50), reference_val_prob=pipeline.predict_proba(X[:50])[:, 1],
        columns=columns, manifest_extra={},
    )
    # Tamper: re-save the npz with different content but leave the manifest's
    # recorded hash pointing at the original.
    with open(npz_path, "wb") as f:
        np.savez(f, X_train=X[50:] + 1.0, X_val=X[:50], y_val=y[:50],
                  val_transaction_ids=np.arange(50), reference_val_prob=np.zeros(50))

    with pytest.raises(HandoffError, match="SHA-256 mismatch"):
        load_handoff(npz_path, manifest_path)


def test_load_handoff_rejects_manifest_column_count_mismatch(fitted_pipeline, tmp_path):
    pipeline, X, y = fitted_pipeline
    columns = [f"f{i}" for i in range(X.shape[1])]
    npz_path = tmp_path / "handoff.npz"
    manifest_path = tmp_path / "manifest.json"
    save_handoff(
        npz_path, manifest_path,
        X_train=X[50:], X_val=X[:50], y_val=y[:50],
        val_transaction_ids=np.arange(50), reference_val_prob=pipeline.predict_proba(X[:50])[:, 1],
        columns=columns, manifest_extra={},
    )
    import json

    manifest = json.loads(manifest_path.read_text())
    manifest["columns"] = columns[:-1]  # drop one column name, now mismatched
    manifest_path.write_text(json.dumps(manifest))

    with pytest.raises(HandoffError, match="columns"):
        load_handoff(npz_path, manifest_path)


def test_save_handoff_rejects_non_npz_suffix(fitted_pipeline, tmp_path):
    pipeline, X, y = fitted_pipeline
    with pytest.raises(HandoffError, match="npz_path"):
        save_handoff(
            tmp_path / "handoff.bin", tmp_path / "manifest.json",
            X_train=X, X_val=X[:10], y_val=y[:10],
            val_transaction_ids=np.arange(10), reference_val_prob=np.zeros(10),
            columns=["a"] * X.shape[1], manifest_extra={},
        )


def _tiny_handoff_kwargs(n_train=6, n_val=4):
    rng = np.random.RandomState(0)
    return dict(
        X_train=rng.rand(n_train, 3), X_val=rng.rand(n_val, 3), y_val=np.array([0, 1] * (n_val // 2)),
        val_transaction_ids=np.arange(n_val), columns=["a", "b", "c"], manifest_extra={"tier": "t"},
    )


def test_handoff_without_new_optional_arrays_has_the_pre_phase9_layout(tmp_path):
    """Backward compatibility: existing callers (Phases 5-8) pass a reference and no y_train, and must get exactly the
    array set they always got."""
    kwargs = _tiny_handoff_kwargs()
    save_handoff(tmp_path / "h.npz", tmp_path / "m.json", reference_val_prob=np.linspace(0, 1, 4), **kwargs)
    _, arrays = load_handoff(tmp_path / "h.npz", tmp_path / "m.json")
    assert set(arrays) == {"X_train", "X_val", "y_val", "val_transaction_ids", "reference_val_prob"}


def test_handoff_accepts_labels_and_no_reference_for_the_mlp(tmp_path):
    kwargs = _tiny_handoff_kwargs()
    y_train = np.array([0, 1, 0, 0, 1, 0])
    save_handoff(tmp_path / "h.npz", tmp_path / "m.json", reference_val_prob=None, y_train=y_train, **kwargs)
    _, arrays = load_handoff(tmp_path / "h.npz", tmp_path / "m.json")
    assert set(arrays) == {"X_train", "X_val", "y_val", "val_transaction_ids", "y_train"}
    np.testing.assert_array_equal(arrays["y_train"], y_train)


def test_handoff_rejects_mismatched_label_length(tmp_path):
    with pytest.raises(HandoffError, match="y_train"):
        save_handoff(tmp_path / "h.npz", tmp_path / "m.json", reference_val_prob=None, y_train=np.zeros(5, dtype=int), **_tiny_handoff_kwargs())


def _sigmoid_score(params, X):
    coef = np.asarray(params["lr_coef"], dtype=np.float64).reshape(-1)
    intercept = float(np.asarray(params["lr_intercept"], dtype=np.float64).reshape(-1)[0])
    with np.errstate(over="ignore"):  # raw-feature scores saturate on purpose in the control test
        return 1.0 / (1.0 + np.exp(-(X @ coef + intercept)))


def test_standardize_features_makes_exported_coefficients_reproduce_the_pipeline(fitted_pipeline):
    """`lr_coef` are the coefficients of the model trained on STANDARDIZED
    features: applying them to `standardize_features(params, X)` must give
    exactly the pipeline's probabilities. This is the contract every
    Concrete-ML LR build depends on (src/fhe/compile/linear.py)."""
    pipeline, X, _ = fitted_pipeline
    params = extract_lr_pipeline_params(pipeline)
    np.testing.assert_allclose(_sigmoid_score(params, standardize_features(params, X)), pipeline.predict_proba(X)[:, 1], atol=1e-12)


def test_applying_exported_coefficients_to_raw_features_is_not_the_pipeline(fitted_pipeline):
    """Regression lock for the Phase 8 audit finding: the handoff holds RAW
    features, so using them directly with `lr_coef` (no scaler) evaluates a
    different function. Asserted to be materially different so the mistake
    can't hide behind a loose tolerance."""
    pipeline, X, _ = fitted_pipeline
    params = extract_lr_pipeline_params(pipeline)
    raw_prob = _sigmoid_score(params, X)
    true_prob = pipeline.predict_proba(X)[:, 1]
    assert np.max(np.abs(raw_prob - true_prob)) > 0.1

