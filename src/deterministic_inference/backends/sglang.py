"""SGLang backend implementation."""

import os
import subprocess
import time
import urllib.error
import urllib.request
from typing import Optional

from deterministic_inference.backends.base import Backend
from deterministic_inference.logging_config import get_logger

logger = get_logger(__name__)


class SGLangBackend(Backend):
    """SGLang inference backend implementation."""
    
    def __init__(
        self,
        model_path: str,
        host: str = "127.0.0.1",
        port: int = 30000,
        startup_timeout: int = 300
    ):
        """Initialize SGLang backend.
        
        Args:
            model_path: Path to the model directory
            host: Host address to bind the SGLang server
            port: Port number for the SGLang server
            startup_timeout: Maximum time to wait for server startup (seconds)
        """
        super().__init__(model_path, host, port)
        self.startup_timeout = startup_timeout
        self.process: Optional[subprocess.Popen] = None
    
    def start_server(self) -> bool:
        """Start the SGLang inference server.
        
        Returns:
            True if server started successfully, False otherwise
        """
        if not self.model_path:
            logger.error("No model path specified")
            return False
        
        if self.process is not None:
            logger.warning("SGLang server is already running")
            return True
        
        try:
            cmd = [
                "python3", "-m", "sglang.launch_server",
                "--model-path", self.model_path,
                "--host", self.host,
                "--port", str(self.port),
                "--attention-backend", "fa3",
                "--enable-deterministic-inference"
            ]
            
            logger.info(f"Starting SGLang server with model: {self.model_path}")
            logger.debug(f"Command: {' '.join(cmd)}")
            
            # Start SGLang server process
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=os.environ.copy()
            )
            
            # Wait for server to be ready
            if self._wait_for_ready():
                logger.info(f"SGLang server started successfully at {self.base_url}")
                return True
            else:
                logger.error("Failed to start SGLang server")
                self.stop_server()
                return False
        
        except Exception as e:
            logger.error(f"Error starting SGLang server: {e}", exc_info=True)
            return False
    
    def _wait_for_ready(self) -> bool:
        """Wait for SGLang server to be ready.
        
        Returns:
            True if server became ready, False on timeout or error
        """
        start_time = time.time()
        logger.info(f"Waiting for SGLang server to be ready (timeout: {self.startup_timeout}s)...")
        
        while time.time() - start_time < self.startup_timeout:
            try:
                # Check if process is still running
                if self.process and self.process.poll() is not None:
                    stdout, stderr = self.process.communicate()
                    logger.error("SGLang process exited unexpectedly")
                    logger.error(f"STDOUT: {stdout.decode()}")
                    logger.error(f"STDERR: {stderr.decode()}")
                    return False
                
                # Check if server is responding
                if self.health_check():
                    logger.info("SGLang server is ready")
                    return True
                
                time.sleep(2)
            
            except Exception as e:
                logger.debug(f"Health check failed (retrying): {e}")
                time.sleep(2)
        
        logger.error(f"Timeout waiting for SGLang server (waited {self.startup_timeout}s)")
        return False
    
    def health_check(self) -> bool:
        """Check if SGLang server is healthy.
        
        Returns:
            True if server is healthy, False otherwise
        """
        if not self.is_running():
            return False
        
        try:
            response = urllib.request.urlopen(f"{self.base_url}/health", timeout=5)
            return response.getcode() == 200
        except (urllib.error.URLError, urllib.error.HTTPError):
            return False
        except Exception as e:
            logger.debug(f"Health check error: {e}")
            return False
    
    def is_running(self) -> bool:
        """Check if SGLang process is running.
        
        Returns:
            True if process is running, False otherwise
        """
        return self.process is not None and self.process.poll() is None
    
    def stop_server(self) -> None:
        """Stop the SGLang server gracefully."""
        if self.process is None:
            logger.debug("No SGLang process to stop")
            return
        
        logger.info("Stopping SGLang server...")
        
        try:
            # Try graceful termination first
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
                logger.info("SGLang server stopped gracefully")
            except subprocess.TimeoutExpired:
                # Force kill if graceful shutdown fails
                logger.warning("SGLang server did not stop gracefully, forcing kill")
                self.process.kill()
                self.process.wait()
                logger.info("SGLang server killed")
        except Exception as e:
            logger.error(f"Error stopping SGLang server: {e}", exc_info=True)
        finally:
            self.process = None
