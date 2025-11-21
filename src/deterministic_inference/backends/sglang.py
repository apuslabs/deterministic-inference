"""SGLang backend implementation."""

import os
import signal
import subprocess
import time
import urllib.error
import urllib.request
from typing import Optional

from deterministic_inference.backends.base import Backend
from deterministic_inference.logging_config import get_logger

logger = get_logger(__name__)


class SGLangBackend(Backend):
    """SGLang backend for inference server process."""

    def __init__(
        self,
        model_path: str,
        host: str = "127.0.0.1",
        port: int = 30000,
        startup_timeout: int = 300
    ):
        super().__init__(model_path, host, port)
        self.startup_timeout = startup_timeout
        self.process: Optional[subprocess.Popen] = None
    
    def start_server(self) -> bool:
        """Start SGLang server."""
        if not self.model_path:
            logger.error("No model path specified")
            return False
        
        if self.process is not None:
            if self.is_running():
                logger.warning("SGLang server already running")
                return True
            else:
                logger.warning("Process object exists but dead, cleaning up")
                self.process = None
        
        if self._is_port_in_use():
            logger.error(f"Port {self.port} already in use")
            return False
        
        try:
            cmd = [
                "python3", "-m", "sglang.launch_server",
                "--model-path", self.model_path,
                "--host", self.host,
                "--port", str(self.port),
                "--attention-backend", "fa3",
                "--enable-deterministic-inference",
                "--context-length", "32768",
                "--stream-output"
            ]
            
            logger.info(f"Starting SGLang: {self.model_path} on {self.host}:{self.port}")
            
            self.process = subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
                env=os.environ.copy()
            )
            
            logger.info(f"Process started (PID: {self.process.pid})")
            
            if self._wait_for_ready():
                logger.info(f"Server ready at {self.base_url}")
                return True
            else:
                logger.error("Server failed to start")
                self.stop_server()
                return False
        
        except Exception as e:
            logger.error(f"Error starting server: {e}", exc_info=True)
            if self.process is not None:
                self.stop_server()
            return False
    
    def _is_port_in_use(self) -> bool:
        """Check if port is in use."""
        try:
            response = urllib.request.urlopen(
                f"http://{self.host}:{self.port}/health",
                timeout=1
            )
            return True
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
            return False
        except Exception:
            return False
    
    def _wait_for_ready(self) -> bool:
        """Wait for server ready."""
        start_time = time.time()
        logger.info(f"Waiting for server (timeout: {self.startup_timeout}s)")
        
        last_log_time = start_time
        check_interval = 2
        
        while time.time() - start_time < self.startup_timeout:
            try:
                if self.process and self.process.poll() is not None:
                    return_code = self.process.returncode
                    logger.error(
                        f"Process exited unexpectedly (code {return_code}). "
                        f"Check the console output above for error details."
                    )
                    return False
                
                if self.health_check():
                    elapsed = int(time.time() - start_time)
                    logger.info(f"Server ready ({elapsed}s)")
                    return True
                
                current_time = time.time()
                if current_time - last_log_time >= 10:
                    elapsed = int(current_time - start_time)
                    logger.info(f"Still waiting ({elapsed}s)")
                    last_log_time = current_time
                
                time.sleep(check_interval)
            except Exception as e:
                logger.debug(f"Health check error: {e}")
                time.sleep(check_interval)
        
        logger.error(f"Timeout after {self.startup_timeout}s")
        return False
    
    def health_check(self) -> bool:
        """Check if server is healthy."""
        if not self.is_running():
            return False
        
        try:
            response = urllib.request.urlopen(f"{self.base_url}/health", timeout=5)
            return response.getcode() == 200
        except Exception:
            return False
    
    def is_running(self) -> bool:
        """Check if process is running."""
        if self.process is None:
            return False
        return self.process.poll() is None
    
    def stop_server(self, timeout: int = 30) -> None:
        """Stop server gracefully."""
        if self.process is None:
            return

        try:
            pid = self.process.pid
            logger.info(f"Stopping server (PID: {pid})")

            if self.process.poll() is not None:
                logger.info("Process already terminated")
                return

            # Send SIGTERM first
            self.process.terminate()
            logger.debug("Sent SIGTERM")

            try:
                return_code = self.process.wait(timeout=timeout)
                logger.info(f"Server stopped (code: {return_code})")
                return
            except subprocess.TimeoutExpired:
                logger.warning(f"Timeout after {timeout}s, forcing kill")

            # Force kill if timeout
            self.process.kill()
            logger.debug("Sent SIGKILL")

            try:
                self.process.wait(timeout=5)
                logger.info("Server forcefully terminated")
            except subprocess.TimeoutExpired:
                logger.error("Process still alive after kill")

        except Exception as e:
            logger.error(f"Error stopping server: {e}", exc_info=True)
        finally:
            self.process = None

    def __del__(self) -> None:
        """Cleanup on GC."""
        if self.process is not None and self.is_running():
            try:
                self.stop_server(timeout=5)
            except Exception:
                pass  # Ignore errors during cleanup
    
    def __enter__(self):
        """Start server."""
        if not self.is_running():
            if not self.start_server():
                raise RuntimeError("Failed to start SGLang server")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop server."""
        self.stop_server()
        return False
