"""Logging configuration for the inference server."""

import logging
import sys
from typing import Optional


def setup_logging(level: str = "INFO") -> None:
    """Setup basic logging to console."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        stream=sys.stdout
    )


def get_logger(name: str) -> logging.Logger:
    """Get logger for module."""
    if not name.startswith("deterministic_inference"):
        name = f"deterministic_inference.{name}"
    return logging.getLogger(name)
