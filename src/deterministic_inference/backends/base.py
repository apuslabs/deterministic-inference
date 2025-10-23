"""Abstract base class for inference backends."""

from abc import ABC, abstractmethod
from typing import Optional


class Backend(ABC):
    """Abstract base class for inference backend implementations.
    
    This defines the interface that all backend implementations must follow,
    enabling extensibility for multiple inference frameworks (SGLang, vLLM, TGI, etc.).
    """
    
    def __init__(self, model_path: str, host: str = "127.0.0.1", port: int = 30000):
        """Initialize the backend.
        
        Args:
            model_path: Path to the model directory
            host: Host address to bind the backend server
            port: Port number for the backend server
        """
        self.model_path = model_path
        self.host = host
        self.port = port
        self.base_url = f"http://{self.host}:{self.port}"
    
    @abstractmethod
    def start_server(self) -> bool:
        """Start the inference backend server.
        
        Returns:
            True if server started successfully, False otherwise
        """
        pass
    
    @abstractmethod
    def stop_server(self) -> None:
        """Stop the inference backend server.
        
        Should perform graceful shutdown and cleanup of resources.
        """
        pass
    
    @abstractmethod
    def health_check(self) -> bool:
        """Check if the backend server is healthy and ready.
        
        Returns:
            True if server is healthy and ready, False otherwise
        """
        pass
    
    @abstractmethod
    def is_running(self) -> bool:
        """Check if the backend process is currently running.
        
        Returns:
            True if process is running, False otherwise
        """
        pass
    
    def get_base_url(self) -> str:
        """Get the base URL of the backend server.
        
        Returns:
            Base URL (e.g., "http://127.0.0.1:30000")
        """
        return self.base_url
    
    def __repr__(self) -> str:
        """String representation of the backend."""
        return (
            f"{self.__class__.__name__}(model_path={self.model_path!r}, "
            f"host={self.host!r}, port={self.port})"
        )
