"""Reusable logging helpers for console and file output."""

from __future__ import annotations

import logging
from pathlib import Path


LOG_FORMAT = "%(asctime)s | %(name)s | %(levelname)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
SUPPORTED_LEVELS = {"INFO": logging.INFO, "WARNING": logging.WARNING, "ERROR": logging.ERROR}


def _normalize_level(level: str | int) -> int:
    """Convert a user-provided level into a logging level constant."""
    if isinstance(level, int):
        return level

    normalized = level.upper()
    if normalized not in SUPPORTED_LEVELS:
        raise ValueError(f"Unsupported log level: {level}")

    return SUPPORTED_LEVELS[normalized]


def setup_logger(
    name: str = "strawberry",
    level: str | int = "INFO",
    log_file: str | Path = "logs/app.log",
    enable_console: bool = True,
    enable_file: bool = True,
) -> logging.Logger:
    """Create or reuse a configured logger for the application.

    The logger is configured only once per name to avoid duplicated output
    during repeated imports or GUI refresh cycles.
    """

    logger = logging.getLogger(name)
    logger.setLevel(_normalize_level(level))
    logger.propagate = False

    formatter = logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT)

    if enable_console and not _has_handler(logger, logging.StreamHandler):
        console_handler = logging.StreamHandler()
        console_handler.setLevel(_normalize_level(level))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    if enable_file:
        file_path = Path(log_file)
        file_path.parent.mkdir(parents=True, exist_ok=True)

        if not _has_file_handler(logger, file_path):
            file_handler = logging.FileHandler(file_path, encoding="utf-8")
            file_handler.setLevel(_normalize_level(level))
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "strawberry") -> logging.Logger:
    """Return an existing logger instance without altering its configuration."""
    return logging.getLogger(name)


def _has_handler(logger: logging.Logger, handler_type: type[logging.Handler]) -> bool:
    """Check whether the logger already has a handler of the given type."""
    return any(type(handler) is handler_type for handler in logger.handlers)


def _has_file_handler(logger: logging.Logger, file_path: Path) -> bool:
    """Check whether the logger already writes to the target file."""
    resolved_path = file_path.resolve()
    return any(
        isinstance(handler, logging.FileHandler) and Path(handler.baseFilename).resolve() == resolved_path
        for handler in logger.handlers
    )
