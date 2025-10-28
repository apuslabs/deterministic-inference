"""Integration tests for the inference server using OpenAI SDK."""

import json
import os
import time

import pytest
from openai import OpenAI


# Default base URL, can be overridden with SERVER_BASE_URL environment variable
DEFAULT_BASE_URL = "http://127.0.0.1:8080"
# DEFAULT_BASE_URL = "http://127.0.0.1:8734/~inference@1.0"


@pytest.fixture(scope="module")
def base_url():
    """Get the base URL for the inference server from environment or use default."""
    return os.getenv("SERVER_BASE_URL", DEFAULT_BASE_URL)


@pytest.fixture(scope="module")
def openai_client(base_url):
    """Create OpenAI client pointing to the local inference server.
    
    Note: This assumes the inference server is already running.
    You can start it manually or use a pytest fixture to manage the server lifecycle.
    """
    client = OpenAI(
        base_url=base_url,
        api_key="test-api-key"  # Dummy API key for testing
    )
    return client


@pytest.fixture(scope="module")
def server_health_check(base_url):
    """Ensure server is healthy before running tests."""
    import urllib.request
    import urllib.error
    
    health_url = f"{base_url}/health"
    max_retries = 10
    retry_delay = 2
    
    for i in range(max_retries):
        try:
            response = urllib.request.urlopen(health_url, timeout=5)
            if response.getcode() == 200:
                print(f"Server is healthy and ready at {base_url}")
                return True
        except urllib.error.URLError:
            if i < max_retries - 1:
                print(f"Server not ready at {base_url}, retrying in {retry_delay}s...")
                time.sleep(retry_delay)
            else:
                pytest.skip(f"Server is not available at {base_url}")
    
    pytest.skip(f"Server health check failed at {base_url}")


class TestHealthEndpoint:
    """Test the health check endpoint."""
    
    def test_health_endpoint(self, base_url, server_health_check):
        """Test that health endpoint returns 200."""
        import urllib.request
        
        health_url = f"{base_url}/health"
        response = urllib.request.urlopen(health_url, timeout=5)
        assert response.getcode() == 200
        import json
        data = json.loads(response.read())
        assert "status" in data
        assert data["status"] == "healthy"


class TestCompletionsAPI:
    """Test /v1/completions endpoint."""
    
    def test_basic_completion(self, openai_client, server_health_check):
        """Test basic text completion."""
        response = openai_client.completions.create(
            model="test-model",  # Model name doesn't matter for proxy
            prompt="Once upon a time",
            max_tokens=50,
            temperature=0.7
        )
        
        assert response.id is not None
        assert len(response.choices) > 0
        assert response.choices[0].text is not None
        assert response.usage is not None
        _assert_environment_metadata(response)
    
    def test_completion_with_temperature_zero(self, openai_client, server_health_check):
        """Test completion with temperature 0 for deterministic output."""
        response = openai_client.completions.create(
            model="test-model",
            prompt="The capital of France is",
            max_tokens=10,
            temperature=0.0
        )
        
        assert response.id is not None
        assert len(response.choices) > 0
        assert response.choices[0].text is not None
        _assert_environment_metadata(response)


class TestChatCompletionsAPI:
    """Test /v1/chat/completions endpoint."""
    
    def test_basic_chat_completion(self, openai_client, server_health_check):
        """Test basic chat completion."""
        response = openai_client.chat.completions.create(
            model="test-model",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello, how are you?"}
            ],
            max_tokens=50,
            temperature=0.7
        )
        
        assert response.id is not None
        assert len(response.choices) > 0
        assert response.choices[0].message.content is not None
        assert response.choices[0].message.role == "assistant"
        assert response.usage is not None
        _assert_environment_metadata(response)
    
    def test_chat_completion_deterministic(self, openai_client, server_health_check):
        """Test chat completion with temperature 0 for deterministic output."""
        response = openai_client.chat.completions.create(
            model="test-model",
            messages=[
                {"role": "user", "content": "What is 2+2?"}
            ],
            max_tokens=20,
            temperature=0.0
        )
        
        assert response.id is not None
        assert len(response.choices) > 0
        assert response.choices[0].message.content is not None
        _assert_environment_metadata(response)
    
    def test_chat_completion_multiple_messages(self, openai_client, server_health_check):
        """Test chat completion with conversation history."""
        response = openai_client.chat.completions.create(
            model="test-model",
            messages=[
                {"role": "system", "content": "You are a math tutor."},
                {"role": "user", "content": "What is 5 + 3?"},
                {"role": "assistant", "content": "5 + 3 = 8"},
                {"role": "user", "content": "What about 10 + 7?"}
            ],
            max_tokens=30,
            temperature=0.5
        )
        
        assert response.id is not None
        assert len(response.choices) > 0
        assert response.choices[0].message.content is not None
        _assert_environment_metadata(response)


class TestErrorHandling:
    """Test error handling scenarios."""
    
    def test_invalid_endpoint(self, base_url):
        """Test that invalid endpoints return 404."""
        import urllib.request
        import urllib.error
        
        try:
            invalid_url = f"{base_url}//invalid"
            urllib.request.urlopen(invalid_url, timeout=5)
            assert False, "Should have raised HTTPError"
        except urllib.error.HTTPError as e:
            assert e.code == 404


# Pytest configuration
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )


# Mark all tests in this module as integration tests
pytestmark = pytest.mark.integration


def _assert_environment_metadata(response) -> None:
    """Assert that environment metadata is present and valid."""
    payload = response.model_dump()
    environment_str = payload.get("environment")
    assert environment_str is not None, "environment metadata missing from response"
    assert isinstance(environment_str, str), "environment metadata must be a JSON string"

    environment = json.loads(environment_str)
    for key in ("driver_version", "cuda_version", "gpu_count", "gpus"):
        assert key in environment, f"environment metadata missing '{key}'"

    assert isinstance(environment["gpu_count"], int)
    assert isinstance(environment["gpus"], list)
