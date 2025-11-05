"""Configuration management for the inference server."""

import os
from types import SimpleNamespace
from typing import Optional


def load_config(
    model_path: Optional[str] = None,
    port: Optional[int] = None,
    backend_port: Optional[int] = None,
    timeout: Optional[int] = None,
    debug: bool = False
) -> SimpleNamespace:
    """Load configuration from parameters and environment variables."""
    config = SimpleNamespace()

    # Load from environment first, then override with parameters
    config.model_path = model_path or os.getenv("INFERENCE_MODEL_PATH")
    config.proxy_host = "127.0.0.1"
    config.proxy_port = port or int(os.getenv("INFERENCE_PORT", "8080"))
    config.backend_host = "127.0.0.1"
    config.backend_port = backend_port or int(os.getenv("INFERENCE_BACKEND_PORT", "30000"))
    config.backend_startup_timeout = timeout or int(os.getenv("INFERENCE_STARTUP_TIMEOUT", "300"))
    config.log_level = "DEBUG" if debug else "INFO"

    # Validate
    if not config.model_path:
        raise ValueError("model_path is required. Use --model-path or set INFERENCE_MODEL_PATH")

    if not (1024 <= config.backend_port <= 65535):
        raise ValueError(f"Invalid backend port: {config.backend_port}")

    if not (1024 <= config.proxy_port <= 65535):
        raise ValueError(f"Invalid proxy port: {config.proxy_port}")

    return config
