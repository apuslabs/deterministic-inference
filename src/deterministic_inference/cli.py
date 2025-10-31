"""Command-line interface for the inference server."""

import argparse
import sys
from typing import Optional

from deterministic_inference.config import Config
from deterministic_inference.logging_config import setup_logging, get_logger
from deterministic_inference.server import InferenceServer, setup_signal_handlers


def parse_args(args: Optional[list] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="deterministic-inference-server",
        description="OpenAI-compatible inference server with SGLang backend",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  deterministic-inference-server --model-path /path/to/model
  deterministic-inference-server --model-path /path/to/model --port 8000
  deterministic-inference-server --model-path /path/to/model --debug

Logs:
  Auto-saved to ~/.cache/deterministic-inference/logs/ with rotation.
  
Env Variables:
  INFERENCE_MODEL_PATH       Model path
  INFERENCE_PORT             Server port (default: 8080)
  INFERENCE_BACKEND_PORT     Backend port (default: 30000)
  INFERENCE_STARTUP_TIMEOUT  Startup timeout seconds (default: 300)
        """
    )
    parser.add_argument(
        "--model-path",
        type=str,
        help="Model directory path (required)"
    )
    parser.add_argument(
        "--port",
        type=int,
        help="Server port (default: 8080)"
    )
    parser.add_argument(
        "--backend-port",
        type=int,
        help="SGLang backend port (default: 30000)"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        help="Startup timeout seconds (default: 300)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    return parser.parse_args(args)


def main(args: Optional[list] = None) -> int:
    """Main entry point."""
    try:
        parsed_args = parse_args(args)
        
        # Config priority: CLI args > Env vars > Defaults
        base_config = Config.from_env()
        
        cli_kwargs = {
            "model_path": parsed_args.model_path,
            "backend_port": parsed_args.backend_port,
            "proxy_port": parsed_args.port,
            "backend_startup_timeout": parsed_args.timeout,
            "log_level": "DEBUG" if parsed_args.debug else None,
        }
        
        config = base_config.merge_with_args(**cli_kwargs)
        
        try:
            config.validate()
        except ValueError as e:
            print(f"Error: {e}", file=sys.stderr)
            print("Use --help for usage information", file=sys.stderr)
            return 1
        
        # Auto file logging with rotation
        setup_logging(
            level=config.log_level,
            log_to_console=True,
            enable_rotation=True,
        )
        
        logger = get_logger(__name__)
        logger.info("=" * 70)
        logger.info("Deterministic Inference Server")
        logger.info("=" * 70)
        logger.info(f"Model: {config.model_path}")
        logger.info(f"Server: http://127.0.0.1:{config.proxy_port}")
        logger.info(f"Backend: SGLang on port {config.backend_port}")
        logger.info("=" * 70)
        
        server = InferenceServer(config)
        setup_signal_handlers(server)
        
        if not server.start():
            logger.error("Failed to start server")
            return 1
        
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
