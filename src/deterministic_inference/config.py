"""Configuration management for the inference server."""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Config:
    """Configuration for the inference server.
    
    Configuration precedence: CLI args > Environment variables > Defaults
    """
    
    # Model configuration
    model_path: Optional[str] = None
    
    # Backend configuration
    backend: str = "sglang"
    backend_host: str = "127.0.0.1"
    backend_port: int = 30000
    
    # Proxy server configuration
    proxy_host: str = "127.0.0.1"
    proxy_port: int = 8080
    
    # Logging configuration
    log_level: str = "INFO"
    log_file: Optional[str] = None
    log_to_console: bool = True
    
    # Backend startup configuration
    backend_startup_timeout: int = 300  # seconds
    
    @classmethod
    def from_env(cls) -> "Config":
        """Create configuration from environment variables.
        
        Environment variables use the prefix INFERENCE_:
        - INFERENCE_MODEL_PATH
        - INFERENCE_BACKEND
        - INFERENCE_BACKEND_HOST
        - INFERENCE_BACKEND_PORT
        - INFERENCE_PROXY_HOST
        - INFERENCE_PROXY_PORT
        - INFERENCE_LOG_LEVEL
        - INFERENCE_LOG_FILE
        - INFERENCE_LOG_TO_CONSOLE
        - INFERENCE_BACKEND_STARTUP_TIMEOUT
        """
        return cls(
            model_path=os.getenv("INFERENCE_MODEL_PATH"),
            backend=os.getenv("INFERENCE_BACKEND", "sglang"),
            backend_host=os.getenv("INFERENCE_BACKEND_HOST", "127.0.0.1"),
            backend_port=int(os.getenv("INFERENCE_BACKEND_PORT", "30000")),
            proxy_host=os.getenv("INFERENCE_PROXY_HOST", "127.0.0.1"),
            proxy_port=int(os.getenv("INFERENCE_PROXY_PORT", "8080")),
            log_level=os.getenv("INFERENCE_LOG_LEVEL", "INFO").upper(),
            log_file=os.getenv("INFERENCE_LOG_FILE"),
            log_to_console=os.getenv("INFERENCE_LOG_TO_CONSOLE", "true").lower() == "true",
            backend_startup_timeout=int(os.getenv("INFERENCE_BACKEND_STARTUP_TIMEOUT", "300")),
        )
    
    def merge_with_args(self, **kwargs) -> "Config":
        """Merge configuration with CLI arguments.
        
        CLI arguments override environment variables and defaults.
        Only non-None values are used for override.
        """
        merged_values = {}
        for key, value in kwargs.items():
            if value is not None:
                merged_values[key] = value
            else:
                merged_values[key] = getattr(self, key)
        
        return Config(**merged_values)
    
    def validate(self) -> None:
        """Validate configuration."""
        if not self.model_path:
            raise ValueError("model_path is required")
        
        if self.backend not in ["sglang"]:
            raise ValueError(f"Unsupported backend: {self.backend}. Supported: sglang")
        
        if self.log_level not in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            raise ValueError(f"Invalid log level: {self.log_level}")
        
        if self.backend_port < 1 or self.backend_port > 65535:
            raise ValueError(f"Invalid backend port: {self.backend_port}")
        
        if self.proxy_port < 1 or self.proxy_port > 65535:
            raise ValueError(f"Invalid proxy port: {self.proxy_port}")
    
    def __repr__(self) -> str:
        """String representation (masks sensitive data)."""
        return (
            f"Config(model_path={self.model_path!r}, "
            f"backend={self.backend!r}, "
            f"backend_host={self.backend_host!r}, "
            f"backend_port={self.backend_port}, "
            f"proxy_host={self.proxy_host!r}, "
            f"proxy_port={self.proxy_port}, "
            f"log_level={self.log_level!r})"
        )
