"""Deterministic Inference Server - OpenAI-compatible proxy for inference backends."""

__version__ = "0.1.0"
__author__ = "ApusLabs"

from deterministic_inference.server import InferenceServer

__all__ = ["InferenceServer", "__version__"]
