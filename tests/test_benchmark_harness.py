"""Phase 7 benchmark harness tests (`docs/plan.md` Phase 7).

`trial_stats`/`time_repeated`/`plaintext_latency_trials` are pure functions
of plain callables/arrays and are tested on every platform. Everything that
touches a real Concrete-ML circuit (`fhe_round_trip_trials`, and the full
`src.benchmark.run` pipeline) is `skipif`-gated on concrete-ml, same as
every other Phase 5/6 FHE test, and built against tiny SYNTHETIC LR/XGBoost
models saved to `tmp_path` -- never the real committed Phase 5/6 data.
"""

import importlib.util
import json

import numpy as np
import pytest


def _is_concrete_ml_available() -> bool:
    try:
        return importlib.util.find_spec("concrete.ml") is not None
    except ModuleNotFoundError:
        return False


concrete_ml_available = _is_concrete_ml_available()
skip_without_concrete_ml = pytest.mark.skipif(
    not concrete_ml_available,
    reason="concrete-ml is not installed in this environment. It has no Windows wheels; run under the WSL2 venv. See docs/environment.md.",
)


# --------------------------------------------------------------------------
# Pure statistics tests -- no Concrete-ML required, run on every platform.
# --------------------------------------------------------------------------

def test_trial_stats_computes_mean_std_min_max():
    from src.benchmark.harness import trial_stats

    stats = trial_stats([1.0, 2.0, 3.0])
    assert stats["n_trials"] == 3
    assert stats["mean_seconds"] == pytest.approx(2.0)
    assert stats["std_seconds"] == pytest.approx(1.0)  # sample std, ddof=1
    assert stats["min_seconds"] == 1.0
    assert stats["max_seconds"] == 3.0
    assert stats["trials_seconds"] == [1.0, 2.0, 3.0]


def test_trial_stats_single_trial_reports_zero_std_not_nan():
    from src.benchmark.harness import trial_stats

    stats = trial_stats([5.0])
    assert stats["n_trials"] == 1
    assert stats["std_seconds"] == 0.0


def test_trial_stats_rejects_empty_input():
    from src.benchmark.harness import trial_stats

    with pytest.raises(ValueError):
        trial_stats([])


def test_time_repeated_calls_fn_exactly_n_times_and_collects_outputs():
    from src.benchmark.harness import time_repeated

    calls = []
    seconds, outputs = time_repeated(lambda: calls.append(1) or len(calls), n_trials=4)
    assert len(calls) == 4
    assert outputs == [1, 2, 3, 4]
    assert len(seconds) == 4
    assert all(s >= 0 for s in seconds)


def test_plaintext_latency_trials_reports_reproducibility():
    from src.benchmark.harness import plaintext_latency_trials

    stable = plaintext_latency_trials(lambda: np.array([0.42]), n_trials=3)
    assert stable["outputs_reproducible"] is True
    assert stable["n_trials"] == 3
    assert stable["throughput_requests_per_second"] > 0

    counter = {"i": 0}

    def unstable():
        counter["i"] += 1
        return np.array([counter["i"]])  # different every call

    flaky = plaintext_latency_trials(unstable, n_trials=3)
    assert flaky["outputs_reproducible"] is False


# --------------------------------------------------------------------------
# Concrete-ML-dependent tests -- WSL2 venv only, tiny synthetic models.
# --------------------------------------------------------------------------

@pytest.fixture
def toy_lr_circuit():
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    from src.fhe.compile.linear import build_concrete_lr, compile_model
    from src.fhe.handoff import extract_lr_pipeline_params

    rng = np.random.RandomState(0)
    n, d = 300, 4
    X = rng.randn(n, d) * np.array([1.0, 50.0, 0.01, 200.0])
    y = (X[:, 0] + X[:, 1] / 50.0 > 0).astype(int)
    pipeline = Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression(C=1.0, max_iter=500))])
    pipeline.fit(X, y)
    X_scaled = pipeline.named_steps["scaler"].transform(X).astype(np.float64)
    params = extract_lr_pipeline_params(pipeline)

    cml_model = build_concrete_lr(params, X_scaled, n_bits=8)
    circuit, _ = compile_model(cml_model, X_scaled)
    return cml_model, circuit, X_scaled


@skip_without_concrete_ml
def test_fhe_round_trip_trials_matches_simulate_and_reports_reproducibility(toy_lr_circuit):
    from src.benchmark.harness import fhe_round_trip_trials

    cml_model, circuit, X_scaled = toy_lr_circuit
    q_row = cml_model.quantize_input(X_scaled[:1])

    result = fhe_round_trip_trials(circuit, q_row, n_trials=3)
    assert result["total"]["n_trials"] == 3
    assert result["keygen_seconds"] > 0
    assert result["outputs_reproducible"] is True  # same row, same circuit -> deterministic
    assert result["throughput_requests_per_second"] > 0
    for step in ("encrypt", "run", "decrypt"):
        assert result[step]["n_trials"] == 3
        assert result[step]["mean_seconds"] >= 0


@skip_without_concrete_ml
def test_fhe_round_trip_trials_checkpoints_are_reused_and_invalidated_by_fingerprint(toy_lr_circuit, tmp_path):
    """Phase 8 relies on this: a single trial can cost 30-50 minutes at
    n_bits=14/16 on real tiers, so a resumed grid run must not redo an
    already-completed trial. Mirrors `src/fhe/xgb_poc.py`'s existing
    checkpoint test pattern exactly."""
    from src.benchmark.harness import fhe_round_trip_trials

    cml_model, circuit, X_scaled = toy_lr_circuit
    q_row = cml_model.quantize_input(X_scaled[:1])
    ckpt = tmp_path / "ckpt"

    first = fhe_round_trip_trials(circuit, q_row, n_trials=3, checkpoint_dir=ckpt, fingerprint="fp-a")
    assert sorted(p.name for p in ckpt.glob("trial_*.npy")) == ["trial_000.npy", "trial_001.npy", "trial_002.npy"]

    resumed = fhe_round_trip_trials(circuit, q_row, n_trials=3, checkpoint_dir=ckpt, fingerprint="fp-a")
    assert resumed["total"]["trials_seconds"] == first["total"]["trials_seconds"]  # reused, not re-timed

    tampered = np.load(ckpt / "trial_000.npy")
    tampered[0] += 1
    np.save(ckpt / "trial_000.npy", tampered)
    with pytest.raises(RuntimeError, match="disagrees"):
        fhe_round_trip_trials(circuit, q_row, n_trials=3, checkpoint_dir=ckpt, fingerprint="fp-a")

    fresh = fhe_round_trip_trials(circuit, q_row, n_trials=3, checkpoint_dir=ckpt, fingerprint="fp-b")  # new fingerprint discards stale trials
    assert fresh["outputs_reproducible"] is True
    assert json.loads((ckpt / "meta.json").read_text())["fingerprint"] == "fp-b"


@skip_without_concrete_ml
def test_benchmark_configuration_end_to_end_on_synthetic_lr(tmp_path):
    """Exercises the full `src.benchmark.run.benchmark_configuration` path
    -- config-hash-keyed result, both latency paths, ciphertext/key sizes
    -- against a synthetic LR handoff saved to `tmp_path`, never the real
    committed Phase 5 data. Every path written here is absolute, so
    `PROJECT_ROOT / <absolute path>` (used throughout `src/benchmark/run.py`
    and `src/fhe/*`) resolves to it regardless of the real project root --
    no monkeypatching needed."""
    import yaml
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    from src.data.provenance import compute_config_hash
    from src.fhe.handoff import extract_lr_pipeline_params, save_handoff

    rng = np.random.RandomState(1)
    n, d = 300, 4
    X = rng.randn(n, d) * np.array([1.0, 50.0, 0.01, 200.0])
    y = (X[:, 0] + X[:, 1] / 50.0 > 0).astype(int)
    pipeline = Pipeline([("scaler", StandardScaler()), ("lr", LogisticRegression(C=1.0, max_iter=500))])
    pipeline.fit(X, y)
    X_scaled = pipeline.named_steps["scaler"].transform(X).astype(np.float64)
    params = extract_lr_pipeline_params(pipeline)
    reference_prob = pipeline.predict_proba(X)[:, 1]

    params_path = tmp_path / "lr_params.json"
    params_path.write_text(json.dumps(params))
    npz_path = tmp_path / "handoff.npz"
    manifest_path = tmp_path / "handoff_manifest.json"
    save_handoff(
        npz_path, manifest_path,
        X_train=X, X_val=X, y_val=y, val_transaction_ids=np.arange(n),  # RAW X, like src/fhe/export.py (the loader standardizes)
        reference_val_prob=reference_prob, columns=[f"f{i}" for i in range(d)],
        manifest_extra={"params_path": str(params_path)},
    )

    source_config = {"n_bits": 8, "output": {"handoff_npz": str(npz_path), "handoff_manifest": str(manifest_path)}}
    source_config_path = tmp_path / "fake_lr_poc.yaml"
    source_config_path.write_text(yaml.safe_dump(source_config))

    from src.benchmark.run import benchmark_configuration

    entry = {"label": "lr_smoke_test", "model_type": "lr", "source_config": str(source_config_path)}
    result = benchmark_configuration(entry, seed=42, trials=2, n_positive=0)

    assert result["label"] == "lr_smoke_test"
    assert result["model_type"] == "lr"
    assert result["n_bits"] == 8
    assert result["compile_seconds"] > 0
    assert result["plaintext_latency"]["n_trials"] == 2
    assert result["fhe_latency"]["total"]["n_trials"] == 2
    assert result["fhe_latency"]["outputs_reproducible"] is True
    assert result["ciphertext_size_bytes"]["input"] > 0
    assert result["ciphertext_size_bytes"]["output"] > 0
    assert result["test_partition_touched"] is False
    assert len(result["config_hash"]) == 16  # src.data.provenance.compute_config_hash
    assert result["source_config_hash"] == compute_config_hash(source_config)
