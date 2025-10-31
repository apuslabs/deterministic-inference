"""Logging configuration for the inference server."""

import logging
import sys
from datetime import datetime
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler
from pathlib import Path
from typing import Optional


def get_default_log_dir() -> Path:
    """Get default log directory: ~/.cache/deterministic-inference/logs/"""
    log_dir = Path.home() / ".cache" / "deterministic-inference" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    log_to_console: bool = True,
    log_dir: Optional[str] = None,
    enable_rotation: bool = True,
    max_bytes: int = 10 * 1024 * 1024,
    backup_count: int = 5,
) -> logging.Logger:
    """Setup logging with auto file output and rotation."""
    logger = logging.getLogger("deterministic_inference")
    logger.setLevel(getattr(logging, level.upper()))
    logger.handlers.clear()
    
    detailed_formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)-30s | %(funcName)-20s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    
    simple_formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S"
    )
    
    # Console handler with simple format
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))
        console_handler.setFormatter(simple_formatter)
        logger.addHandler(console_handler)
    
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        final_log_file = str(log_path)
    else:
        if log_dir:
            log_directory = Path(log_dir)
        else:
            log_directory = get_default_log_dir()
        
        log_directory.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_log_file = str(log_directory / f"inference_server_{timestamp}.log")
        
        latest_link = log_directory / "latest.log"
        if latest_link.exists() or latest_link.is_symlink():
            latest_link.unlink()
        try:
            latest_link.symlink_to(Path(final_log_file).name)
        except (OSError, NotImplementedError):
            pass
    if enable_rotation:
        file_handler = RotatingFileHandler(
            final_log_file,
            mode='a',
            maxBytes=max_bytes,
            backupCount=backup_count,
            encoding='utf-8'
        )
    else:
        file_handler = logging.FileHandler(
            final_log_file,
            mode='a',
            encoding='utf-8'
        )
    
    file_handler.setLevel(getattr(logging, level.upper()))
    file_handler.setFormatter(detailed_formatter)
    logger.addHandler(file_handler)
    
    logger.info(f"Log file: {final_log_file}")
    logger.info(f"Level: {level}, rotation: {enable_rotation}, console: {log_to_console}")
    
    logger.propagate = False
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get logger for module."""
    if not name.startswith("deterministic_inference"):
        name = f"deterministic_inference.{name}"
    return logging.getLogger(name)
