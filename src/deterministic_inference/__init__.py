"""Deterministic Inference Server - OpenAI-compatible proxy for inference backends."""

__version__ = "0.1.0"
__author__ = "ApusLabs"

from deterministic_inference.config import Config
from deterministic_inference.server import InferenceServer

__all__ = ["Config", "InferenceServer", "__version__"]
