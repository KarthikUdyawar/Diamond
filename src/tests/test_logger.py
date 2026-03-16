"""
src/tests/test_logger.py
─────────────────────────
Unit tests for src/logger.py.
"""

from __future__ import annotations

import logging
from collections.abc import Generator

import pytest

import src.logger as logger_module
from src.config import get_settings
from src.logger import bind_context, clear_context, configure_logging, get_logger


@pytest.fixture(autouse=True)
def _reset_logger_state() -> Generator[None, None, None]:
    """Reset logger + settings cache between tests."""
    logger_module._configured = False
    get_settings.cache_clear()
    yield
    logger_module._configured = False
    get_settings.cache_clear()


class TestGetLogger:
    def test_returns_non_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("APP_ENV", "local")
        assert get_logger("test.module") is not None

    def test_accepts_arbitrary_name(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("APP_ENV", "local")
        assert get_logger("src.features") is not None

    def test_multiple_loggers_no_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("APP_ENV", "local")
        l1 = get_logger("a")
        l2 = get_logger("b")
        assert l1 is not None
        assert l2 is not None


class TestConfigureLogging:
    def test_idempotent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("APP_ENV", "local")
        configure_logging()
        configure_logging()
        configure_logging()  # should not raise

    def test_sets_root_level_debug(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("APP_ENV", "local")
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")
        configure_logging()
        assert logging.getLogger().level == logging.DEBUG

    def test_sets_root_level_warning(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("APP_ENV", "testing")
        monkeypatch.setenv("LOG_LEVEL", "WARNING")
        configure_logging()
        assert logging.getLogger().level == logging.WARNING

    def test_sets_root_level_info(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("APP_ENV", "staging")
        monkeypatch.setenv("LOG_LEVEL", "INFO")
        configure_logging()
        assert logging.getLogger().level == logging.INFO

    def test_noisy_loggers_silenced(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("APP_ENV", "local")
        configure_logging()
        for name in ("urllib3", "httpx", "mlflow", "optuna", "catboost"):
            assert logging.getLogger(name).level == logging.WARNING

    def test_configured_flag_set(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("APP_ENV", "local")
        assert logger_module._configured is False
        configure_logging()
        assert logger_module._configured is True


class TestContextBinding:
    def test_bind_does_not_raise(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("APP_ENV", "local")
        bind_context(run_id="abc123", model="catboost")

    def test_clear_does_not_raise(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("APP_ENV", "local")
        bind_context(run_id="abc123")
        clear_context()

    def test_bind_clear_cycle(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("APP_ENV", "local")
        for _ in range(5):
            bind_context(x=1)
            clear_context()


class TestLoggerEmission:
    def test_info_does_not_raise(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("APP_ENV", "local")
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")
        logger = get_logger("test.info")
        logger.info("test_event", key="value", number=42)

    def test_debug_does_not_raise(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("APP_ENV", "local")
        get_logger("test.debug").debug("debug_event", detail="x")

    def test_warning_does_not_raise(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("APP_ENV", "testing")
        get_logger("test.warning").warning("warn_event", reason="test")

    def test_error_does_not_raise(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("APP_ENV", "local")
        get_logger("test.error").error("error_event", code=500)

    def test_staging_json_renderer_no_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("APP_ENV", "staging")
        monkeypatch.setenv("LOG_LEVEL", "INFO")
        configure_logging()
        get_logger("test.staging").info("staging_event", env="staging")

    def test_testing_no_color_renderer(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("APP_ENV", "testing")
        monkeypatch.setenv("LOG_LEVEL", "WARNING")
        configure_logging()
        get_logger("test.testing").warning("test_event")
