"""
src/logger.py
─────────────
Structured logging for the Diamond project, built on structlog.

Behaviour by environment (driven by get_settings().app_env):
    local   → ConsoleRenderer (coloured, human-readable, timestamped)
    testing → ConsoleRenderer (no colour, minimal noise during pytest)
    staging → JSONRenderer    (machine-readable, one JSON object per line)

Log level is driven by get_settings().log_level:
    local   : DEBUG
    testing : WARNING
    staging : INFO

Usage
-----
    from src.logger import get_logger

    logger = get_logger(__name__)
    logger.info("pipeline_started", rows=len(df), env=cfg.app_env)
    logger.debug("split_done", train=len(X_train), test=len(X_test))
    logger.warning("empty_secret", field="kaggle_key")
    logger.error("model_load_failed", exc_info=True)

Every log call accepts arbitrary keyword arguments which are included as
structured fields — no string formatting required.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog
from structlog.types import EventDict, Processor

# ── Internal state ─────────────────────────────────────────────────────────────
_configured = False


def _add_app_env(
    logger: Any,  # noqa: ANN401
    method: str,
    event_dict: EventDict,
) -> EventDict:
    """Inject app_env into every log record without importing at module level."""
    try:
        from src.config import get_settings  # local import avoids circular deps

        event_dict["app_env"] = get_settings().app_env
    except Exception:  # noqa: BLE001 — never crash the logger
        event_dict["app_env"] = "unknown"
    return event_dict


def _get_log_level() -> int:
    """Resolve the numeric log level from settings, defaulting to DEBUG."""
    try:
        from src.config import get_settings

        return int(logging.getLevelName(get_settings().log_level))
    except Exception:  # noqa: BLE001
        return logging.DEBUG


def _is_staging() -> bool:
    try:
        from src.config import get_settings

        return get_settings().is_staging
    except Exception:  # noqa: BLE001
        return False


def _is_testing() -> bool:
    try:
        from src.config import get_settings

        return get_settings().is_testing
    except Exception:  # noqa: BLE001
        return False


def configure_logging() -> None:
    """
    Configure structlog and the stdlib root logger.

    Safe to call multiple times — subsequent calls are no-ops.
    Called automatically on the first get_logger() call.
    """
    global _configured  # noqa: PLW0603
    if _configured:
        return

    level = _get_log_level()
    staging = _is_staging()
    testing = _is_testing()

    # ── Shared processors (applied to every log event) ─────────────────────────
    shared_processors: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        _add_app_env,
    ]

    # ── Renderer ───────────────────────────────────────────────────────────────
    if staging:
        renderer: Processor = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(
            colors=not testing,  # no ANSI codes in pytest output
            exception_formatter=structlog.dev.plain_traceback,
        )

    # ── Configure structlog ────────────────────────────────────────────────────
    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # ── Configure stdlib root logger ───────────────────────────────────────────
    formatter = structlog.stdlib.ProcessorFormatter(
        processor=renderer,
        foreign_pre_chain=shared_processors,
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.addHandler(handler)
    root_logger.setLevel(level)

    # Silence noisy third-party loggers
    for noisy in (
        "urllib3",
        "httpx",
        "httpcore",
        "mlflow",
        "optuna",
        "catboost",
        "LightGBM",
    ):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _configured = True


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """
    Return a bound structlog logger for the given module name.

    Configures logging on first call. Subsequent calls are cheap.

    Parameters
    ----------
    name : str
        Typically ``__name__`` of the calling module.

    Returns
    -------
    structlog.stdlib.BoundLogger

    Examples
    --------
    >>> logger = get_logger(__name__)
    >>> logger.info("training_started", model="catboost", trials=50)
    >>> logger.error("unexpected_error", exc_info=True)
    """
    configure_logging()
    return structlog.get_logger(name)


def bind_context(**kwargs: Any) -> None:  # noqa: ANN401
    """
    Bind key-value pairs to the current context (thread/async-local).

    All subsequent log calls in this context will include these fields
    without needing to pass them explicitly.

    Example
    -------
    >>> bind_context(run_id="abc123", model="catboost-tuned")
    >>> logger.info("epoch_done", epoch=1)
    # output includes run_id and model automatically
    """
    structlog.contextvars.bind_contextvars(**kwargs)


def clear_context() -> None:
    """
    Clear all context-local bindings.

    Call at the start of each request or training job to avoid leakage
    between runs.
    """
    structlog.contextvars.clear_contextvars()
