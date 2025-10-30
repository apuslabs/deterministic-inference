"""SGLang backend implementation."""

import os
import signal
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
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
        self.pid_file = Path(f"/tmp/sglang_backend_{self.port}.pid")
    
    def start_server(self) -> bool:
        """Start the SGLang inference server.
        
        Returns:
            True if server started successfully, False otherwise
        """
        if not self.model_path:
            logger.error("No model path specified")
            return False
        
        # Check for existing PID file
        if self.pid_file.exists():
            try:
                existing_pid = int(self.pid_file.read_text().strip())
                if self._is_process_running(existing_pid):
                    logger.error(
                        f"Another SGLang instance is already running (PID: {existing_pid}). "
                        f"Cannot start new instance on port {self.port}."
                    )
                    return False
                else:
                    logger.warning(f"Stale PID file found (PID: {existing_pid}), removing")
                    self.pid_file.unlink()
            except (ValueError, OSError) as e:
                logger.warning(f"Error reading PID file: {e}, removing")
                self.pid_file.unlink()
        
        if self.process is not None:
            logger.warning("SGLang server is already running in this instance")
            return True
        
        try:
            cmd = [
                "python3", "-m", "sglang.launch_server",
                "--model-path", self.model_path,
                "--host", self.host,
                "--port", str(self.port),
                "--attention-backend", "fa3",
                "--enable-deterministic-inference",
                "--context-length", "32768"
            ]
            
            logger.info(f"Starting SGLang server: {self.model_path}")
            
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=os.environ.copy(),
                preexec_fn=os.setsid
            )
            
            try:
                self.pid_file.write_text(str(self.process.pid))
            except OSError as e:
                logger.error(f"Failed to write PID file: {e}")
                self.stop_server()
                return False
            
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
        """Wait for SGLang server to be ready."""
        start_time = time.time()
        logger.info(f"Waiting for server (timeout: {self.startup_timeout}s)...")
        
        while time.time() - start_time < self.startup_timeout:
            try:
                if self.process and self.process.poll() is not None:
                    stdout, stderr = self.process.communicate()
                    logger.error(f"Process exited unexpectedly: {stderr.decode()}")
                    return False
                
                if self.health_check():
                    logger.info("Server ready")
                    return True
                
                time.sleep(2)
            except Exception as e:
                logger.debug(f"Health check retry: {e}")
                time.sleep(2)
        
        logger.error(f"Timeout after {self.startup_timeout}s")
        return False
    
    def health_check(self) -> bool:
        """Check if SGLang server is healthy."""
        if not self.is_running():
            return False
        
        try:
            response = urllib.request.urlopen(f"{self.base_url}/health", timeout=5)
            return response.getcode() == 200
        except Exception:
            return False
    
    def is_running(self) -> bool:
        """Check if SGLang process is running."""
        if self.process is None:
            return False
        
        if self.process.poll() is not None:
            return False
        
        if self.pid_file.exists():
            try:
                pid = int(self.pid_file.read_text().strip())
                return self._is_process_running(pid)
            except (ValueError, OSError):
                return False
        
        return True
    
    def _is_process_running(self, pid: int) -> bool:
        """Check if a process with given PID is running."""
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False
    
    def stop_server(self) -> None:
        """Stop the SGLang server gracefully."""
        if self.process is None:
            self._cleanup_pid_file()
            return
        
        logger.info("Stopping server...")
        
        try:
            try:
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
            except ProcessLookupError:
                self.process.terminate()
            
            try:
                self.process.wait(timeout=15)
                logger.info("Server stopped")
            except subprocess.TimeoutExpired:
                logger.warning("Forcing kill")
                try:
                    os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                except ProcessLookupError:
                    self.process.kill()
                self.process.wait()
        except Exception as e:
            logger.error(f"Error stopping server: {e}")
        finally:
            self.process = None
            self._cleanup_pid_file()
    
    def _cleanup_pid_file(self) -> None:
        """Remove PID file if it exists."""
        if self.pid_file.exists():
            try:
                self.pid_file.unlink()
            except OSError as e:
                logger.warning(f"Failed to remove PID file: {e}")
