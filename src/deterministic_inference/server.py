"""Main inference server orchestration."""

import signal
import sys
import threading
import time
from http.server import HTTPServer
from typing import Optional

from deterministic_inference.backends.base import Backend
from deterministic_inference.backends.sglang import SGLangBackend
from deterministic_inference.config import Config
from deterministic_inference.environment import (
    EnvironmentCollectionError,
    collect_gpu_environment_json,
)
from deterministic_inference.logging_config import get_logger
from deterministic_inference.proxy.handler import OpenAIProxyHandler

logger = get_logger(__name__)


class InferenceServer:
    """Main inference server that orchestrates backend and proxy."""
    
    def __init__(self, config: Config):
        """Initialize the inference server.
        
        Args:
            config: Server configuration
        """
        self.config = config
        self.backend: Optional[Backend] = None
        self.http_server: Optional[HTTPServer] = None
        self.server_thread: Optional[threading.Thread] = None
        self._shutdown_event = threading.Event()
        self.environment_json: Optional[str] = None
        
        # Initialize backend based on configuration
        self._init_backend()
        self._collect_environment_info()
    
    def _init_backend(self):
        """Initialize the inference backend based on configuration."""
        if self.config.backend == "sglang":
            logger.info("Initializing SGLang backend")
            self.backend = SGLangBackend(
                model_path=self.config.model_path,
                host=self.config.backend_host,
                port=self.config.backend_port,
                startup_timeout=self.config.backend_startup_timeout
            )
        else:
            raise ValueError(f"Unsupported backend: {self.config.backend}")
        
        # Set backend for the proxy handler
        OpenAIProxyHandler.backend = self.backend
    
    def _collect_environment_info(self) -> None:
        """Collect GPU environment information once at startup."""
        logger.info("Collecting GPU environment information")
        try:
            self.environment_json = collect_gpu_environment_json()
        except EnvironmentCollectionError as exc:
            logger.error("Failed to collect GPU environment information", exc_info=True)
            raise

        OpenAIProxyHandler.environment_info = self.environment_json
        logger.debug("GPU environment payload: %s", self.environment_json)

    def start(self) -> bool:
        """Start the inference server.
        
        Returns:
            True if server started successfully, False otherwise
        """
        logger.info("Starting Deterministic Inference Server")
        logger.info(f"Configuration: {self.config}")
        
        # Start backend server
        logger.info("Starting backend server...")
        if not self.backend.start_server():
            logger.error("Failed to start backend server")
            return False
        
        # Start HTTP proxy server
        try:
            logger.info(f"Starting HTTP proxy on {self.config.proxy_host}:{self.config.proxy_port}")
            self.http_server = HTTPServer(
                (self.config.proxy_host, self.config.proxy_port),
                OpenAIProxyHandler
            )
            
            # Start HTTP server in background thread
            self.server_thread = threading.Thread(
                target=self._serve_forever,
                daemon=True
            )
            self.server_thread.start()
            
            logger.info("=" * 70)
            logger.info("Deterministic Inference Server started successfully!")
            logger.info(f"OpenAI-compatible API: http://{self.config.proxy_host}:{self.config.proxy_port}")
            logger.info(f"Backend server: {self.backend.get_base_url()}")
            logger.info(f"Health check: http://{self.config.proxy_host}:{self.config.proxy_port}/health")
            logger.info("=" * 70)
            
            return True
        
        except Exception as e:
            logger.error(f"Error starting HTTP proxy server: {e}", exc_info=True)
            self.backend.stop_server()
            return False
    
    def _serve_forever(self):
        """Run HTTP server until shutdown is requested."""
        try:
            if not self.http_server:
                logger.error("HTTP server not initialised before serve_forever call")
                return
            self.http_server.serve_forever()
        except Exception as e:
            if not self._shutdown_event.is_set():
                logger.error(f"HTTP server error: {e}", exc_info=True)
    
    def stop(self):
        """Stop the inference server gracefully."""
        logger.info("Stopping Deterministic Inference Server...")
        self._shutdown_event.set()
        
        # Stop HTTP server
        if self.http_server:
            logger.info("Stopping HTTP proxy server...")
            self.http_server.shutdown()
            self.http_server.server_close()
            logger.info("HTTP proxy server stopped")
        
        # Stop backend server
        if self.backend:
            self.backend.stop_server()
        
        logger.info("Deterministic Inference Server stopped")
    
    def wait_forever(self):
        """Keep the server running until interrupted."""
        try:
            while not self._shutdown_event.is_set():
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Received keyboard interrupt")
            self.stop()


def setup_signal_handlers(server: InferenceServer):
    """Setup signal handlers for graceful shutdown.
    
    Args:
        server: InferenceServer instance to shutdown on signal
    """
    def signal_handler(signum, frame):
        signal_name = signal.Signals(signum).name
        logger.info(f"Received signal {signal_name}")
        server.stop()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
