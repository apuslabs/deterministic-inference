"""Environment information collection utilities."""

import json
from typing import Dict, Any, List


class EnvironmentCollectionError(RuntimeError):
    """Raised when environment information cannot be collected."""


def _format_cuda_version(raw_version: int) -> str:
    """Convert raw CUDA driver version integer into human-readable string."""
    major = raw_version // 1000
    minor = (raw_version % 1000) // 10
    patch = raw_version % 10
    if patch:
        return f"{major}.{minor}.{patch}"
    return f"{major}.{minor}"


def collect_gpu_environment() -> Dict[str, Any]:
    """Collect GPU environment information and return as dictionary."""
    try:
        import pynvml as nvml
    except ImportError as exc:
        raise EnvironmentCollectionError(
            "nvidia-ml-py is required to collect GPU information."
        ) from exc

    try:
        nvml.nvmlInit()
    except Exception as exc:
        raise EnvironmentCollectionError(f"Failed to initialise NVML: {exc}") from exc

    try:
        # Get driver version
        try:
            driver_version = nvml.nvmlSystemGetDriverVersion()
            if isinstance(driver_version, bytes):
                driver_version = driver_version.decode("utf-8")
        except Exception:
            driver_version = "unknown"

        # Get CUDA version
        try:
            try:
                raw_cuda_version = nvml.nvmlSystemGetCudaDriverVersion_v2()
            except AttributeError:
                raw_cuda_version = nvml.nvmlSystemGetCudaDriverVersion()
            cuda_version = _format_cuda_version(int(raw_cuda_version))
        except Exception:
            cuda_version = "unknown"

        # Get GPU count and info
        gpu_count = nvml.nvmlDeviceGetCount()
        gpus: List[Dict[str, Any]] = []

        for index in range(gpu_count):
            try:
                handle = nvml.nvmlDeviceGetHandleByIndex(index)
                name = nvml.nvmlDeviceGetName(handle)
                if isinstance(name, bytes):
                    name = name.decode("utf-8")

                memory_info = nvml.nvmlDeviceGetMemoryInfo(handle)

                gpus.append({
                    "index": index,
                    "name": name,
                    "memory_total_bytes": int(memory_info.total),
                    "memory_total_mib": round(int(memory_info.total) / (1024 ** 2), 2)
                })
            except Exception:
                # Skip GPU if there's an error querying it
                continue

        return {
            "driver_version": driver_version,
            "cuda_version": cuda_version,
            "gpu_count": gpu_count,
            "gpus": gpus
        }

    except Exception as exc:
        raise EnvironmentCollectionError(f"Failed to query GPU information: {exc}") from exc
    finally:
        try:
            nvml.nvmlShutdown()
        except Exception:
            pass  # Ignore shutdown errors


def collect_gpu_environment_json() -> str:
    """Collect GPU environment information as a JSON string."""
    environment = collect_gpu_environment()
    return json.dumps(environment, separators=(",", ":"))
