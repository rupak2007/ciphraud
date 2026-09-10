import json
import logging

from src.logging_setup import get_logger


def test_get_logger_emits_json(capsys):
    logger = get_logger("test.logging_setup")
    logger.info("hello world")

    captured = capsys.readouterr()
    line = captured.out.strip().splitlines()[-1]
    payload = json.loads(line)

    assert payload["message"] == "hello world"
    assert payload["level"] == "INFO"
    assert payload["logger"] == "test.logging_setup"
    assert "timestamp" in payload


def test_get_logger_does_not_duplicate_handlers():
    logger_a = get_logger("test.logging_setup.dup")
    logger_b = get_logger("test.logging_setup.dup")
    assert logger_a is logger_b
    assert len(logger_a.handlers) == 1


def test_get_logger_respects_level():
    logger = get_logger("test.logging_setup.level", level=logging.WARNING)
    assert logger.level == logging.WARNING
