"""Command-line interface for the inference server."""

import argparse
import sys
from typing import Optional

from deterministic_inference.config import Config
from deterministic_inference.logging_config import setup_logging, get_logger
from deterministic_inference.server import InferenceServer, setup_signal_handlers


def parse_args(args: Optional[list] = None) -> argparse.Namespace:
    """Parse command-line arguments.
    
    Args:
        args: List of arguments to parse (defaults to sys.argv[1:])
    
    Returns:
        Parsed arguments namespace
    """
    parser = argparse.ArgumentParser(
        prog="deterministic-inference-server",
        description="OpenAI-compatible proxy server for inference backends",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Start server with SGLang backend
  deterministic-inference-server --model-path /path/to/model
  
  # Start with custom ports
  deterministic-inference-server --model-path /path/to/model --proxy-port 8000
  
  # Enable debug logging with file output
  deterministic-inference-server --model-path /path/to/model --log-level DEBUG --log-file server.log
  
Environment Variables:
  INFERENCE_MODEL_PATH          Model directory path
  INFERENCE_BACKEND             Backend type (default: sglang)
  INFERENCE_BACKEND_HOST        Backend server host (default: 127.0.0.1)
  INFERENCE_BACKEND_PORT        Backend server port (default: 30000)
  INFERENCE_PROXY_HOST          Proxy server host (default: 127.0.0.1)
  INFERENCE_PROXY_PORT          Proxy server port (default: 8080)
  INFERENCE_LOG_LEVEL           Log level (default: INFO)
  INFERENCE_LOG_FILE            Log file path (optional)
  INFERENCE_LOG_TO_CONSOLE      Log to console (default: true)
  
Configuration Precedence: CLI args > Environment variables > Defaults
        """
    )
    
    # Model configuration
    parser.add_argument(
        "--model-path",
        type=str,
        help="Path to the model directory (required)"
    )
    
    # Backend configuration
    parser.add_argument(
        "--backend",
        type=str,
        choices=["sglang"],
        help="Inference backend type (default: sglang)"
    )
    parser.add_argument(
        "--backend-host",
        type=str,
        help="Backend server host (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--backend-port",
        type=int,
        help="Backend server port (default: 30000)"
    )
    parser.add_argument(
        "--backend-startup-timeout",
        type=int,
        help="Backend startup timeout in seconds (default: 300)"
    )
    
    # Proxy server configuration
    parser.add_argument(
        "--proxy-host",
        type=str,
        help="Proxy server host (default: 127.0.0.1)"
    )
    parser.add_argument(
        "--proxy-port",
        type=int,
        help="Proxy server port (default: 8080)"
    )
    
    # Logging configuration
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO)"
    )
    parser.add_argument(
        "--log-file",
        type=str,
        help="Path to log file (optional, logs to file if specified)"
    )
    parser.add_argument(
        "--log-to-console",
        type=str,
        choices=["true", "false"],
        help="Whether to log to console (default: true)"
    )
    
    return parser.parse_args(args)


def main(args: Optional[list] = None) -> int:
    """Main entry point for the CLI.
    
    Args:
        args: List of arguments to parse (defaults to sys.argv[1:])
    
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    try:
        # Parse command-line arguments
        parsed_args = parse_args(args)
        
        # Build configuration: CLI args > Env vars > Defaults
        base_config = Config.from_env()
        
        # Convert parsed args to dict, handling special cases
        cli_kwargs = {
            "model_path": parsed_args.model_path,
            "backend": parsed_args.backend,
            "backend_host": parsed_args.backend_host,
            "backend_port": parsed_args.backend_port,
            "backend_startup_timeout": parsed_args.backend_startup_timeout,
            "proxy_host": parsed_args.proxy_host,
            "proxy_port": parsed_args.proxy_port,
            "log_level": parsed_args.log_level,
            "log_file": parsed_args.log_file,
        }
        
        # Handle log_to_console conversion
        if parsed_args.log_to_console is not None:
            cli_kwargs["log_to_console"] = parsed_args.log_to_console == "true"
        else:
            cli_kwargs["log_to_console"] = None
        
        # Merge CLI args with base config (CLI takes precedence)
        config = base_config.merge_with_args(**cli_kwargs)
        
        # Setup logging before anything else
        setup_logging(
            level=config.log_level,
            log_file=config.log_file,
            log_to_console=config.log_to_console
        )
        
        logger = get_logger(__name__)
        logger.info("Starting Deterministic Inference Server CLI")
        
        # Validate configuration
        try:
            config.validate()
        except ValueError as e:
            logger.error(f"Configuration error: {e}")
            print(f"Error: {e}", file=sys.stderr)
            print("\nUse --help for usage information", file=sys.stderr)
            return 1
        
        # Create and start server
        server = InferenceServer(config)
        
        # Setup signal handlers for graceful shutdown
        setup_signal_handlers(server)
        
        # Start server
        if not server.start():
            logger.error("Failed to start server")
            return 1
        
        # Wait for shutdown signal
        server.wait_forever()
        
        return 0
    
    except KeyboardInterrupt:
        print("\nInterrupted by user", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"Fatal error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
