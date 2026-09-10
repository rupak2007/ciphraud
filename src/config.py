"""Central config loading for the FHE fraud detection pipelines.

Every experiment, pipeline stage, and benchmark run is parameterized by a
versioned YAML config file under `configs/` (per architecture.md §17) --
never by editing constants in code. This module is the single place that
resolves a config path to a validated dict.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = PROJECT_ROOT / "configs"


class ConfigError(Exception):
    """Raised when a config file is missing or malformed."""


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load a YAML experiment config.

    Args:
        config_path: Path to a config file. Relative paths are resolved
            against `configs/` first, then against the current working
            directory, so callers can pass either `"phase0/smoke.yaml"`
            or a full path.

    Returns:
        The parsed config as a dict.

    Raises:
        ConfigError: if the file does not exist or is not valid YAML
            mapping at the top level.
    """
    candidate = Path(config_path)
    if not candidate.is_absolute():
        in_configs = CONFIGS_DIR / candidate
        candidate = in_configs if in_configs.exists() else candidate

    if not candidate.exists():
        raise ConfigError(f"Config file not found: {candidate}")

    with open(candidate, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ConfigError(
            f"Config file must contain a top-level mapping, got "
            f"{type(data).__name__}: {candidate}"
        )

    return data


def get_env(name: str, default: str | None = None, *, required: bool = False) -> str | None:
    """Read an environment variable, centralizing env access.

    Used instead of hardcoding machine-specific paths or scattering
    `os.environ` calls across pipeline modules.
    """
    value = os.environ.get(name, default)
    if required and value is None:
        raise ConfigError(f"Required environment variable not set: {name}")
    return value
