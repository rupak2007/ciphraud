"""Phase 0 environment smoke test.

Per plan.md Phase 0 exit criteria: `pytest` must run, core dependency
imports must succeed, and installed versions must match the pinned lock
file for whichever environment is actually running.

Two lock files exist because Concrete-ML has no Windows wheels (see
docs/environment.md): `requirements.txt` governs the Windows `.venv` and
the non-FHE pipeline stages; `requirements-fhe.txt` governs the WSL2 venv
and is a self-contained superset (it pins its own numpy/pandas/
scikit-learn/xgboost versions, constrained by concrete-ml, which differ
from requirements.txt's). This test detects which environment it's
running in via whether concrete-ml is importable, and checks installed
versions against the matching lock file -- checking against the wrong
file would either fail spuriously or silently pass against the wrong
constraints.
"""

import importlib.util
import re
from importlib import metadata
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REQUIREMENTS_FILE = PROJECT_ROOT / "requirements.txt"
REQUIREMENTS_FHE_FILE = PROJECT_ROOT / "requirements-fhe.txt"

CORE_IMPORTS = ["numpy", "pandas", "sklearn", "xgboost", "yaml", "pytest"]


def _concrete_ml_available() -> bool:
    try:
        return importlib.util.find_spec("concrete.ml") is not None
    except ModuleNotFoundError:
        return False


def _parse_pinned_requirements(requirements_file: Path) -> dict[str, str]:
    pinned = {}
    for line in requirements_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.match(r"^([A-Za-z0-9_.\-]+)==([A-Za-z0-9_.\-]+)$", line)
        if match:
            pinned[match.group(1)] = match.group(2)
    return pinned


@pytest.mark.parametrize("module_name", CORE_IMPORTS)
def test_core_dependency_imports(module_name):
    __import__(module_name)


def test_requirements_file_exists_and_nonempty():
    assert REQUIREMENTS_FILE.exists()
    pinned = _parse_pinned_requirements(REQUIREMENTS_FILE)
    assert len(pinned) > 0


def test_installed_versions_match_pinned_requirements():
    # Under WSL2 with concrete-ml installed, requirements.txt's pins
    # (resolved without concrete-ml's numpy/scikit-learn/xgboost
    # constraints) are expected to mismatch -- requirements-fhe.txt is
    # the authoritative lock file for that environment instead.
    active_lock_file = (
        REQUIREMENTS_FHE_FILE if _concrete_ml_available() else REQUIREMENTS_FILE
    )
    pinned = _parse_pinned_requirements(active_lock_file)
    mismatches = []
    for dist_name, expected_version in pinned.items():
        try:
            installed_version = metadata.version(dist_name)
        except metadata.PackageNotFoundError:
            mismatches.append(f"{dist_name}: not installed")
            continue
        if installed_version != expected_version:
            mismatches.append(
                f"{dist_name}: pinned={expected_version} installed={installed_version}"
            )
    assert not mismatches, (
        f"Version mismatches against {active_lock_file.name}: {mismatches}"
    )
