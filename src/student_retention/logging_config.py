"""Logging configuration for student_retention package."""
import logging
import sys
from pathlib import Path


def get_logger(name: str, log_file: Path | None = None, level: int = logging.INFO) -> logging.Logger:
    """Configure and return a structured logger.

    Args:
        name: Logger name (usually __name__)
        log_file: Optional path to log file. If None, logs to console only.
        level: Logging level (default: INFO)

    Returns:
        logging.Logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Avoid duplicate handlers if logger was already configured
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
