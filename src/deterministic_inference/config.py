"""Configuration management for the inference server."""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Config:
    """Server configuration. Priority: CLI args > Env vars > Defaults"""
    
    model_path: Optional[str] = None
    
    backend: str = "sglang"
    backend_host: str = "127.0.0.1"
    backend_port: int = 30000
    
    proxy_host: str = "127.0.0.1"
    proxy_port: int = 8080
    
    log_level: str = "INFO"
    log_file: Optional[str] = None
    log_dir: Optional[str] = None
    log_to_console: bool = True
    log_rotation: bool = True
    log_max_bytes: int = 10 * 1024 * 1024
    log_backup_count: int = 5
    
    backend_startup_timeout: int = 300
    
    @classmethod
    def from_env(cls) -> "Config":
        """Load config from environment variables."""
        return cls(
            model_path=os.getenv("INFERENCE_MODEL_PATH"),
            backend_port=int(os.getenv("INFERENCE_BACKEND_PORT", "30000")),
            proxy_port=int(os.getenv("INFERENCE_PORT", "8080")),
            backend_startup_timeout=int(os.getenv("INFERENCE_STARTUP_TIMEOUT", "300")),
        )
    
    def merge_with_args(self, **kwargs) -> "Config":
        """Merge with CLI args (non-None values override)."""
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
            raise ValueError("model_path is required. Use --model-path or set INFERENCE_MODEL_PATH")
        
        if self.backend_port < 1024 or self.backend_port > 65535:
            raise ValueError(f"Invalid backend port: {self.backend_port}")
        
        if self.proxy_port < 1024 or self.proxy_port > 65535:
            raise ValueError(f"Invalid proxy port: {self.proxy_port}")
    
    def __repr__(self) -> str:
        return (
            f"Config(model={self.model_path!r}, "
            f"port={self.proxy_port}, "
            f"backend_port={self.backend_port}, "
            f"timeout={self.backend_startup_timeout}s)"
        )
