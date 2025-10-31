"""Abstract base class for inference backends."""

from abc import ABC, abstractmethod
from typing import Optional


class Backend(ABC):
    """Abstract base for inference backends."""
    
    def __init__(self, model_path: str, host: str = "127.0.0.1", port: int = 30000):
        self.model_path = model_path
        self.host = host
        self.port = port
        self.base_url = f"http://{self.host}:{self.port}"
    
    @abstractmethod
    def start_server(self) -> bool:
        """Start backend server."""
        pass
    
    @abstractmethod
    def stop_server(self) -> None:
        """Stop backend server gracefully."""
        pass
    
    @abstractmethod
    def health_check(self) -> bool:
        """Check if backend is healthy."""
        pass
    
    @abstractmethod
    def is_running(self) -> bool:
        """Check if backend process is running."""
        pass
    
    def get_base_url(self) -> str:
        """Get backend base URL."""
        return self.base_url
    
    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(model={self.model_path!r}, "
            f"host={self.host!r}, port={self.port})"
        )
