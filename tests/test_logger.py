"""Tests for the reusable logging helpers."""

from pathlib import Path

from utils.logger import setup_logger


def test_setup_logger_creates_log_file_and_avoids_duplicate_handlers(tmp_path: Path) -> None:
    """Logger setup should be idempotent for the same logger name and file."""
    log_file = tmp_path / "app.log"

    logger = setup_logger(name="strawberry.test", log_file=log_file)
    logger_again = setup_logger(name="strawberry.test", log_file=log_file)

    logger.info("logger smoke test")

    assert logger is logger_again
    assert log_file.exists()
    assert len(logger.handlers) == 2
