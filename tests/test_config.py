from pathlib import Path

import pytest

from src.config import ConfigError, load_config


def test_load_config_resolves_relative_to_configs_dir():
    config = load_config("phase0/smoke.yaml")
    assert config["phase"] == 0
    assert config["seed"] == 42


def test_load_config_accepts_absolute_path():
    from src.config import CONFIGS_DIR

    absolute = CONFIGS_DIR / "phase0" / "smoke.yaml"
    config = load_config(absolute)
    assert config["phase"] == 0


def test_load_config_missing_file_raises():
    with pytest.raises(ConfigError):
        load_config("phase0/does_not_exist.yaml")


def test_load_config_rejects_non_mapping(tmp_path: Path):
    bad_config = tmp_path / "list_config.yaml"
    bad_config.write_text("- 1\n- 2\n", encoding="utf-8")
    with pytest.raises(ConfigError):
        load_config(bad_config)


def test_load_config_empty_file_returns_empty_dict(tmp_path: Path):
    empty_config = tmp_path / "empty.yaml"
    empty_config.write_text("", encoding="utf-8")
    assert load_config(empty_config) == {}
