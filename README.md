# Deterministic Inference Server

An OpenAI-compatible proxy server for inference backends, starting with SGLang support and designed for extensibility.

## Overview

This project provides a production-ready inference server that:
- Exposes OpenAI-compatible API endpoints (`/v1/completions`, `/v1/chat/completions`)
- Proxies requests to backend inference frameworks (currently SGLang)
- Supports structured logging with configurable outputs
- Provides graceful shutdown and health monitoring
- Designed for extensibility to support multiple backends (vLLM, TGI, etc.)

## Features

- ✅ **OpenAI API Compatibility**: Drop-in replacement for OpenAI API endpoints
- ✅ **SGLang Backend**: Full support for SGLang inference framework
- ✅ **Extensible Architecture**: Abstract backend interface for easy additions
- ✅ **Structured Logging**: Configurable logging to console and/or file
- ✅ **Configuration Flexibility**: CLI args, environment variables, or defaults
- ✅ **Health Monitoring**: Built-in health check endpoint
- ✅ **Graceful Shutdown**: Proper cleanup on SIGTERM/SIGINT
- ✅ **Integration Tests**: Comprehensive tests using OpenAI SDK

## Installation

### Using uv (Recommended)

```bash
# Clone the repository
git clone https://github.com/apuslabs/deterministic-inference.git
cd deterministic-inference

# Install with uv
uv sync

# Or install in development mode
uv pip install -e .

# Install with test dependencies
uv pip install -e ".[test]"
```

### Using pip

```bash
pip install -e .

# With test dependencies
pip install -e ".[test]"
```

## Quick Start

### 1. Start the Server

Using the installed command:

```bash
deterministic-inference-server --model-path /path/to/model
```

Or using Python module:

```bash
python -m deterministic_inference --model-path /path/to/model
```

### 2. Make Requests

Using the OpenAI Python SDK:

```python
from openai import OpenAI

# Point to your local server
client = OpenAI(
    base_url="http://localhost:8080/v1",
    api_key="dummy-key"  # Not validated, but required by SDK
)

# Text completion
response = client.completions.create(
    model="your-model",
    prompt="Once upon a time",
    max_tokens=50
)

# Chat completion
response = client.chat.completions.create(
    model="your-model",
    messages=[
        {"role": "user", "content": "Hello!"}
    ]
)
```

Using curl:

```bash
# Health check
curl http://localhost:8080/health

# Chat completion
curl http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "your-model",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 50
  }'
```

## Configuration

### Command-Line Arguments

```bash
deterministic-inference-server \
  --model-path /path/to/model \           # Required
  --backend sglang \                      # Backend type (default: sglang)
  --backend-host 127.0.0.1 \              # Backend server host
  --backend-port 30000 \                  # Backend server port
  --proxy-host 0.0.0.0 \                  # Proxy server host
  --proxy-port 8080 \                     # Proxy server port
  --log-level INFO \                      # Log level
  --log-file /var/log/inference.log       # Optional log file
```

### Environment Variables

```bash
export INFERENCE_MODEL_PATH=/path/to/model
export INFERENCE_BACKEND=sglang
export INFERENCE_BACKEND_HOST=127.0.0.1
export INFERENCE_BACKEND_PORT=30000
export INFERENCE_PROXY_HOST=0.0.0.0
export INFERENCE_PROXY_PORT=8080
export INFERENCE_LOG_LEVEL=INFO
export INFERENCE_LOG_FILE=/var/log/inference.log

deterministic-inference-server
```

### Configuration Precedence

Configuration is loaded in the following order (later sources override earlier ones):
1. **Defaults** - Sensible defaults built into the application
2. **Environment Variables** - Prefixed with `INFERENCE_`
3. **CLI Arguments** - Highest priority, override everything else

## Usage Examples

### Basic Usage

```bash
# Start with minimal configuration
deterministic-inference-server --model-path /models/llama-2-7b
```

### Production Usage

```bash
# Start with production settings
deterministic-inference-server \
  --model-path /models/llama-2-7b \
  --proxy-host 0.0.0.0 \
  --proxy-port 8080 \
  --log-level INFO \
  --log-file /var/log/inference.log
```

### Debug Mode

```bash
# Enable debug logging
deterministic-inference-server \
  --model-path /models/llama-2-7b \
  --log-level DEBUG
```

## Integration from Erlang

This server is designed to be called from Erlang applications:

```erlang
% Start server process
ServerPort = open_port(
    {spawn, "deterministic-inference-server --model-path /path/to/model --proxy-port 8080"},
    [stream, exit_status, {line, 1024}]
),

% Wait for server to be ready
timer:sleep(5000),

% Check health
{ok, {{_, 200, _}, _, Body}} = httpc:request("http://localhost:8080/health"),

% Use the server...

% Stop server gracefully
port_close(ServerPort).
```

## API Endpoints

### Health Check

```
GET /health
```

Returns server and backend status:

```json
{
  "status": "healthy",
  "backend": {
    "status": "ready",
    "healthy": true,
    "url": "http://127.0.0.1:30000"
  }
}
```

### Completions

```
POST /v1/completions
```

Compatible with OpenAI's `/v1/completions` endpoint.

### Chat Completions

```
POST /v1/chat/completions
```

Compatible with OpenAI's `/v1/chat/completions` endpoint.

## Testing

Run integration tests (requires server to be running):

```bash
# Install test dependencies
uv pip install -e ".[test]"

# Start the server in one terminal
deterministic-inference-server --model-path /path/to/model

# Run tests in another terminal
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=deterministic_inference
```

## Project Structure

```
deterministic-inference/
├── src/
│   └── deterministic_inference/
│       ├── __init__.py           # Package initialization
│       ├── __main__.py           # Module entry point
│       ├── cli.py                # CLI argument parsing
│       ├── config.py             # Configuration management
│       ├── logging_config.py     # Logging setup
│       ├── server.py             # Main server orchestration
│       ├── backends/
│       │   ├── base.py           # Abstract backend interface
│       │   └── sglang.py         # SGLang implementation
│       └── proxy/
│           └── handler.py        # OpenAI API proxy handler
├── tests/
│   └── test_integration.py       # Integration tests
├── docs/
│   └── REQUIREMENTS.md           # Detailed requirements
├── pyproject.toml                # Package configuration
└── README.md                     # This file
```

## Architecture

The server consists of three main components:

1. **Backend Manager** (`backends/`): Manages the lifecycle of inference backend processes (SGLang, etc.)
2. **Proxy Server** (`proxy/`): HTTP server that implements OpenAI-compatible API
3. **Server Orchestrator** (`server.py`): Coordinates backend and proxy lifecycle

### Extensibility

To add a new backend:

1. Create a new class inheriting from `Backend` in `backends/`
2. Implement required methods: `start_server()`, `stop_server()`, `health_check()`, `is_running()`
3. Update `server.py` to recognize the new backend type
4. Add configuration options in `config.py` and `cli.py`

## Logging

Logs include structured information:
- Timestamp
- Log level
- Module name
- Message

Example log output:
```
2025-10-23 10:30:45 | INFO     | deterministic_inference.server | Starting Deterministic Inference Server
2025-10-23 10:30:45 | INFO     | deterministic_inference.backends.sglang | Starting SGLang server with model: /models/llama-2-7b
2025-10-23 10:30:50 | INFO     | deterministic_inference.backends.sglang | SGLang server is ready
```

## Troubleshooting

### Server won't start

1. Check if model path exists and is accessible
2. Verify ports are not already in use
3. Check logs for detailed error messages
4. Ensure SGLang is installed: `pip install sglang`

### Backend not responding

1. Check backend logs in server output
2. Verify backend health: `curl http://localhost:8080/health`
3. Increase startup timeout: `--backend-startup-timeout 600`

### Connection errors

1. Verify firewall settings
2. Check if proxy host/port are correct
3. Ensure backend is running: check health endpoint

## Contributing

Contributions are welcome! Please ensure:
- Code follows existing style
- New backends implement the `Backend` interface
- Integration tests pass
- Documentation is updated

## License

MIT License - see LICENSE file for details

## Contact

ApusLabs - contact@apuslabs.com
