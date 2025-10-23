"""OpenAI API proxy request handler."""

import json
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler
from typing import Optional

from deterministic_inference.backends.base import Backend
from deterministic_inference.logging_config import get_logger

logger = get_logger(__name__)


class OpenAIProxyHandler(BaseHTTPRequestHandler):
    """HTTP request handler that proxies OpenAI API calls to inference backend."""
    
    # Class variable to hold the backend instance
    backend: Optional[Backend] = None
    
    def do_POST(self):
        """Handle POST requests - proxy to backend OpenAI-compatible endpoint."""
        logger.debug(f"Received POST request to: {self.path}")
        
        if self.path.startswith("/v1/completions") or self.path.startswith("/v1/chat/completions"):
            self._proxy_to_backend()
        else:
            logger.warning(f"Unknown endpoint: {self.path}")
            self._send_error_response(404, "Not Found", f"Endpoint {self.path} not supported")
    
    def do_GET(self):
        """Handle GET requests."""
        if self.path == "/health":
            self._send_health_response()
        else:
            logger.warning(f"Unknown GET endpoint: {self.path}")
            self._send_error_response(404, "Not Found", f"Endpoint {self.path} not found")
    
    def _proxy_to_backend(self):
        """Proxy request to backend server."""
        if not self.backend or not self.backend.is_running():
            logger.error("Backend server not available")
            self._send_error_response(
                503,
                "Service Unavailable",
                "Backend inference server is not running"
            )
            return
        
        try:
            # Read request body
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length)
            
            logger.debug(f"Proxying to backend: {self.path}")
            logger.debug(f"Request body: {post_data.decode('utf-8', errors='replace')[:500]}")
            
            # Prepare backend URL
            backend_url = f"{self.backend.get_base_url()}{self.path}"
            
            # Create request to backend
            req = urllib.request.Request(
                backend_url,
                data=post_data,
                headers=dict(self.headers)
            )
            
            # Forward request to backend
            with urllib.request.urlopen(req, timeout=300) as response:
                # Send response back to client
                self.send_response(response.getcode())
                
                # Copy headers from backend response
                for header, value in response.headers.items():
                    if header.lower() not in ['connection', 'transfer-encoding']:
                        self.send_header(header, value)
                self.end_headers()
                
                # Stream response body
                while True:
                    chunk = response.read(8192)
                    if not chunk:
                        break
                    self.wfile.write(chunk)
                
                logger.info(f"Successfully proxied request to {self.path}")
        
        except urllib.error.HTTPError as e:
            # Forward backend's HTTP error response transparently
            logger.warning(f"Backend returned HTTP error: {e.code} - {e.reason}")
            
            self.send_response(e.code)
            
            # Copy headers from backend error response
            for header, value in e.headers.items():
                if header.lower() not in ['connection', 'transfer-encoding']:
                    self.send_header(header, value)
            self.end_headers()
            
            # Copy error response body
            response_body = e.read()
            self.wfile.write(response_body)
            
            logger.debug(f"Backend error response: {response_body.decode('utf-8', errors='replace')[:500]}")
        
        except urllib.error.URLError as e:
            logger.error(f"Network error connecting to backend: {e}")
            self._send_error_response(
                502,
                "Bad Gateway",
                f"Cannot connect to backend server: {str(e)}"
            )
        
        except Exception as e:
            logger.error(f"Unexpected error proxying request: {e}", exc_info=True)
            self._send_error_response(500, "Internal Server Error", str(e))
    
    def _send_health_response(self):
        """Send health check response."""
        backend_status = "ready" if (self.backend and self.backend.is_running()) else "not_ready"
        backend_healthy = self.backend.health_check() if self.backend else False
        
        response = {
            "status": "healthy",
            "backend": {
                "status": backend_status,
                "healthy": backend_healthy,
                "url": self.backend.get_base_url() if self.backend else None
            }
        }
        
        logger.debug(f"Health check response: {response}")
        
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(response).encode())
    
    def _send_error_response(self, code: int, message: str, detail: str = ""):
        """Send error response in OpenAI API format.
        
        Args:
            code: HTTP status code
            message: Error message
            detail: Additional error details
        """
        error_response = {
            "error": {
                "code": code,
                "message": message,
                "type": "proxy_error"
            }
        }
        
        if detail:
            error_response["error"]["detail"] = detail
        
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps(error_response).encode())
    
    def log_message(self, format, *args):
        """Override to suppress default HTTP request logging."""
        # We handle logging via our own logger
        pass
