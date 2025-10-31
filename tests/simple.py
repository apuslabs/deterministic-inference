#!/usr/bin/env python3
import json
from openai import OpenAI

def print_separator(title):
    """Print a nice separator with title"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)

# Initialize OpenAI client
client = OpenAI(
    base_url="https://hb.apus.network/~inference@1.0",
    api_key="test-api-key"
)

# Test 1: Normal inference (first call)
print_separator("Test 1: Normal Inference (First Call)")
response1 = client.completions.create(
    model="test-model",
    prompt="Once upon a time",
    max_tokens=200,
    temperature=0.7,
)
print(json.dumps(response1.model_dump(), indent=2))

# Test 2: Normal inference (second call - should be identical)
print_separator("Test 2: Normal Inference (Second Call)")
response2 = client.completions.create(
    model="test-model",
    prompt="Once upon a time",
    max_tokens=200,
    temperature=0.7,
)
print(json.dumps(response2.model_dump(), indent=2))

# Test 3: Verify deterministic output
print_separator("Test 3: Deterministic Verification")
is_deterministic = response1.choices[0].text == response2.choices[0].text
print(f"Outputs are identical (deterministic): {is_deterministic}")
if is_deterministic:
    print("✓ Deterministic inference confirmed!")
else:
    print("✗ Warning: Outputs differ!")

# Test 4: Inference with TEE attestation
print_separator("Test 4: Inference with TEE Attestation")
response3 = client.completions.create(
    model="test-model",
    prompt="Once upon a time",
    max_tokens=200,
    temperature=0.7,
    extra_query={"tee": "true"}
)
print(json.dumps(response3.model_dump(), indent=2))

# Test 5: Chat completion (normal)
print_separator("Test 5: Chat Completion (Normal)")
response4 = client.chat.completions.create(
    model="test-model",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ],
    max_tokens=100,
    temperature=0.7,
)
print(json.dumps(response4.model_dump(), indent=2))


print_separator("All Tests Completed")

