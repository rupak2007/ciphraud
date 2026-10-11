"""Phase 10 client tests: client-side preprocessing (pure numpy, every environment) and the encrypt/decrypt client
(Concrete-ML, WSL2 only, on tiny synthetic LR / XGBoost / MLP deployments from `tests/conftest.py`).

Includes the `docs/architecture.md` Sec.19 unit test: the client's quantization must equal the compiled model's exactly.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.client.preprocess import apply_preprocessing
from tests.conftest import NF, needs_concrete

# ---- preprocessing (pure) ---------------------------------------------------------------------------------------------


def test_standardize_matches_the_scaler_formula_and_casts_to_the_recorded_dtype():
    spec = {"kind": "standardize", "scaler_mean": [1.0, 2.0], "scaler_scale": [2.0, 4.0], "dtype": "float32", "n_features": 2}
    out = apply_preprocessing(spec, np.array([[3.0, 10.0], [1.0, 2.0]]))
    assert out.dtype == np.float32
    np.testing.assert_allclose(out, [[1.0, 2.0], [0.0, 0.0]])
    assert apply_preprocessing({**spec, "dtype": "float64"}, np.array([[3.0, 10.0]])).dtype == np.float64


def test_none_preprocessing_returns_the_raw_rows_unchanged():
    raw = np.arange(6, dtype=np.float32).reshape(2, 3)
    out = apply_preprocessing({"kind": "none", "n_features": 3}, raw)
    assert out.dtype == np.float32 and np.array_equal(out, raw)


def test_preprocessing_rejects_wrong_shape_and_unknown_kind():
    with pytest.raises(ValueError, match="2-D"):
        apply_preprocessing({"kind": "none"}, np.zeros(3))
    with pytest.raises(ValueError, match="expected 3 features"):
        apply_preprocessing({"kind": "none", "n_features": 3}, np.zeros((1, 2)))
    with pytest.raises(ValueError, match="unknown preprocessing"):
        apply_preprocessing({"kind": "mystery"}, np.zeros((1, 2)))


# ---- the client (Concrete-ML) -----------------------------------------------------------------------------------------


@pytest.fixture
def server_for(toy_env):
    """A direct (non-HTTP) Concrete-ML server for a toy deployment."""
    from concrete.ml.deployment import FHEModelServer

    return lambda kind: FHEModelServer(str(toy_env.deployments[kind].paths["server_dir"]))


@needs_concrete
def test_client_refuses_to_encrypt_or_decrypt_before_keys_exist(toy_env):
    from src.client.client import CiphraudClient, ClientKeyError

    d = toy_env.deployments["lr"]
    client = CiphraudClient(d.paths["client_dir"])
    with pytest.raises(ClientKeyError):
        client.prepare_request(d.toy.raw[:1])
    with pytest.raises(ClientKeyError):
        client.finish(b"anything")
    with pytest.raises(ClientKeyError):
        client.evaluation_keys()


@needs_concrete
def test_concretes_encrypt_silently_generates_keys_which_is_why_the_client_guards(toy_env):
    """Pins the measured behavior (docs/client_server.md): the library encrypts with freshly generated keys if none were loaded."""
    from concrete.ml.deployment import FHEModelClient

    d = toy_env.deployments["lr"]
    library = FHEModelClient(str(d.paths["client_dir"]))
    q = np.asarray(library.model.quantize_input(np.zeros((1, NF))))
    library.client.encrypt(q)  # no keygen call, and no error
    assert library.client.keys.are_generated


@pytest.mark.parametrize("kind", ["lr", "xgboost", "mlp"])
@needs_concrete
def test_client_quantization_equals_the_compiled_models_exactly(toy_env, kind):
    """`docs/architecture.md` Sec.19: quantization-parameter consistency between the client and the compiled model."""
    from src.client.client import CiphraudClient

    d = toy_env.deployments[kind]
    client = CiphraudClient(d.paths["client_dir"])
    client.generate_keys()
    prepared = client.prepare_request(d.toy.raw[:1])
    assert np.array_equal(prepared["q_in"], d.rows[0]["q_in"])
    from src.analysis.phase10_results import q_in_digest

    assert q_in_digest(prepared["q_in"]) == d.reference["rows"]["0"]["q_in_sha256"]  # the committed integrity reference is reproducible
    assert "q_in" not in d.reference["rows"]["0"]


@pytest.mark.parametrize("kind", ["lr", "xgboost", "mlp"])
@needs_concrete
def test_client_round_trip_through_the_server_reproduces_the_clear_integers_and_probability(toy_env, server_for, kind):
    from concrete import fhe

    from src.client.client import CiphraudClient

    d = toy_env.deployments[kind]
    client = CiphraudClient(d.paths["client_dir"])
    client.generate_keys()
    ek = client.evaluation_keys()
    prepared = client.prepare_request(d.toy.raw[:1])
    response = server_for(kind).run(prepared["ciphertext"], ek["bytes"])
    finished = client.finish(response)
    assert np.array_equal(finished["integers"].reshape(-1), d.rows[0]["clear"])
    assert finished["probability"] == pytest.approx(d.rows[0]["probability"], abs=0)
    assert finished["decision"] == (d.rows[0]["probability"] >= 0.5)
    assert isinstance(fhe.Value.deserialize(response), fhe.Value)
    assert set(prepared["timings"]) == {"preprocess_seconds", "quantize_seconds", "encrypt_seconds", "serialize_seconds"}
    assert set(finished["timings"]) == {"deserialize_seconds", "decrypt_seconds", "postprocess_seconds"}


@needs_concrete
def test_evaluation_keys_are_not_the_keyset_and_do_not_contain_it(toy_env, tmp_path):
    from src.client.client import CiphraudClient

    d = toy_env.deployments["lr"]
    client = CiphraudClient(d.paths["client_dir"])
    client.generate_keys()
    ek = client.evaluation_keys()
    keyset_path = tmp_path / "keyset.bin"
    client.save_keys(keyset_path)
    keyset = keyset_path.read_bytes()
    assert ek["bytes"] != keyset and keyset not in ek["bytes"]
    assert ek["sha256"] == d.sha256(ek["bytes"])


@needs_concrete
def test_saved_keys_decrypt_in_a_fresh_client_and_a_different_keyset_does_not(toy_env, server_for, tmp_path):
    """The staged XGBoost flow: keygen+encrypt in one process, decrypt in another with the reloaded keyset."""
    from src.client.client import CiphraudClient

    d = toy_env.deployments["lr"]
    first = CiphraudClient(d.paths["client_dir"])
    first.generate_keys()
    ek = first.evaluation_keys()
    prepared = first.prepare_request(d.toy.raw[:1])
    response = server_for("lr").run(prepared["ciphertext"], ek["bytes"])
    keyset = tmp_path / "keyset.bin"
    first.save_keys(keyset)

    reloaded = CiphraudClient(d.paths["client_dir"])
    reloaded.load_keys(keyset)
    assert np.array_equal(reloaded.finish(response)["integers"].reshape(-1), d.rows[0]["clear"])

    stranger = CiphraudClient(d.paths["client_dir"])
    stranger.generate_keys()  # a different keyset (default keygen is randomized)
    try:
        wrong = stranger.finish(response)["integers"].reshape(-1)
    except Exception:  # noqa: BLE001 -- failing loudly is also an acceptable outcome
        return
    assert not np.array_equal(wrong, d.rows[0]["clear"])
