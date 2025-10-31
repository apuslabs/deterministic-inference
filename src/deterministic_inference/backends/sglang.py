"""SGLang backend implementation."""

import atexit
import os
import signal
import subprocess
import time
import urllib.error
import urllib.request
import weakref
from typing import Optional

from deterministic_inference.backends.base import Backend
from deterministic_inference.logging_config import get_logger

logger = get_logger(__name__)

_process_registry = weakref.WeakValueDictionary()


def _cleanup_all_processes():
    """Global cleanup on exit."""
    if not _process_registry:
        return
    
    logger.info(f"Cleanup {len(_process_registry)} SGLang process(es)")
    
    for instance in list(_process_registry.values()):
        try:
            if instance.is_running():
                instance.stop_server(timeout=10)
        except Exception as e:
            logger.error(f"Cleanup error: {e}")


atexit.register(_cleanup_all_processes)


class SGLangBackend(Backend):
    """SGLang backend for heavy inference server process."""
    
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
        self._shutdown_requested = False
        
        atexit.register(self._atexit_cleanup)
        _process_registry[id(self)] = self
    
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
                "--context-length", "32768"
            ]
            
            logger.info(f"Starting SGLang: {self.model_path} on {self.host}:{self.port}")
            
            self.process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                stdin=subprocess.DEVNULL,
                env=os.environ.copy(),
                preexec_fn=os.setsid,
                start_new_session=True
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
                    stderr_output = ""
                    
                    if self.process.stderr:
                        try:
                            stderr_data = self.process.stderr.read()
                            stderr_output = stderr_data.decode('utf-8', errors='replace')[:1000]
                        except Exception as e:
                            logger.debug(f"Couldn't read stderr: {e}")
                    
                    logger.error(
                        f"Process exited unexpectedly (code {return_code})"
                        + (f":\n{stderr_output}" if stderr_output else "")
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
        
        if self._shutdown_requested:
            logger.debug("Shutdown already in progress, skipping")
            return
        
        self._shutdown_requested = True
        
        try:
            pid = self.process.pid
            logger.info(f"Stopping server (PID: {pid})")
            
            if self.process.poll() is not None:
                logger.info("Process already terminated")
                return
            
            try:
                os.killpg(os.getpgid(pid), signal.SIGTERM)
                logger.debug("Sent SIGTERM to process group")
            except (ProcessLookupError, PermissionError) as e:
                logger.warning(f"Process group termination failed: {e}, using direct terminate")
                self.process.terminate()
            
            try:
                return_code = self.process.wait(timeout=timeout)
                logger.info(f"Server stopped (code: {return_code})")
                return
            except subprocess.TimeoutExpired:
                logger.warning(f"Timeout after {timeout}s, forcing kill")
            
            try:
                os.killpg(os.getpgid(pid), signal.SIGKILL)
                logger.debug("Sent SIGKILL to process group")
            except (ProcessLookupError, PermissionError):
                self.process.kill()
            
            try:
                self.process.wait(timeout=5)
                logger.info("Server forcefully terminated")
            except subprocess.TimeoutExpired:
                logger.error("Process still alive after SIGKILL")
                
        except Exception as e:
            logger.error(f"Error stopping server: {e}", exc_info=True)
        finally:
            self.process = None
            self._shutdown_requested = False
    
    def _atexit_cleanup(self) -> None:
        """Cleanup on exit."""
        if self.process is not None and self.is_running():
            logger.warning("Cleaning up on exit")
            self.stop_server(timeout=10)
    
    def __del__(self) -> None:
        """Cleanup on GC."""
        if self.process is not None and self.is_running():
            logger.warning("Cleaning up on destruction")
            self.stop_server(timeout=10)
    
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
