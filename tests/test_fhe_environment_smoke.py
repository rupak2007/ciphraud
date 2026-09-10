"""Phase 0 FHE-toolchain smoke test.

Concrete-ML only installs under WSL2/Linux (see docs/environment.md for
the confirmed Windows platform-compatibility finding). This test is
skipped when Concrete-ML is not importable -- e.g. when run against the
Windows `.venv` -- and only asserts real behavior when run against the
WSL2 `.venv-wsl` where Concrete-ML is actually installed.
"""

import importlib.util

import pytest

def _is_concrete_ml_available() -> bool:
    try:
        return importlib.util.find_spec("concrete.ml") is not None
    except ModuleNotFoundError:
        return False


concrete_ml_available = _is_concrete_ml_available()


@pytest.mark.skipif(
    not concrete_ml_available,
    reason=(
        "concrete-ml is not installed in this environment. It has no "
        "Windows wheels; run this test under the WSL2 venv "
        "(~/.venvs/fhe-fraud-detection). See docs/environment.md."
    ),
)
def test_concrete_ml_imports():
    from concrete.ml.sklearn import LogisticRegression  # noqa: F401
