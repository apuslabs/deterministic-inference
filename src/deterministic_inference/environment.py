"""Environment information collection utilities."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Dict, List


class EnvironmentCollectionError(RuntimeError):
    """Raised when environment information cannot be collected."""


@dataclass(frozen=True)
class GPUInfo:
    """Details about a single GPU device."""

    index: int
    name: str
    memory_total_bytes: int

    @property
    def memory_total_mebibytes(self) -> float:
        """Total memory in MiB for convenience."""
        return round(self.memory_total_bytes / (1024 ** 2), 2)


@dataclass(frozen=True)
class GPUEnvironment:
    """GPU environment snapshot collected at server start."""

    driver_version: str
    cuda_version: str
    gpu_count: int
    gpus: List[GPUInfo]

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the GPU environment to a dict."""
        return {
            "driver_version": self.driver_version,
            "cuda_version": self.cuda_version,
            "gpu_count": self.gpu_count,
            "gpus": [
                {
                    "index": gpu.index,
                    "name": gpu.name,
                    "memory_total_bytes": gpu.memory_total_bytes,
                    "memory_total_mib": gpu.memory_total_mebibytes,
                }
                for gpu in self.gpus
            ],
        }

    def to_json(self) -> str:
        """Serialise the GPU environment as a JSON string."""
        return json.dumps(self.to_dict(), separators=(",", ":"))


def _load_nvml_module():
    """Load the NVML module from nvidia-ml-py.

    Returns:
        The NVML module with required symbols.

    Raises:
        EnvironmentCollectionError: If the module cannot be imported.
    """
    try:
        import pynvml as nvml
    except ImportError as exc:  # pragma: no cover - import failure path
        raise EnvironmentCollectionError(
            "nvidia-ml-py is required to collect GPU information."
        ) from exc
    return nvml


def _format_cuda_version(raw_version: int) -> str:
    """Convert raw CUDA driver version integer into human-readable string."""
    major = raw_version // 1000
    minor = (raw_version % 1000) // 10
    patch = raw_version % 10
    if patch:
        return f"{major}.{minor}.{patch}"
    return f"{major}.{minor}"


def collect_gpu_environment() -> GPUEnvironment:
    """Collect the GPU environment information using NVML.

    Returns:
        GPUEnvironment snapshot.

    Raises:
        EnvironmentCollectionError: If NVML initialisation or querying fails.
    """
    nvml = _load_nvml_module()

    try:
        nvml.nvmlInit()
    except nvml.NVMLError as exc:  # type: ignore[attr-defined]
        raise EnvironmentCollectionError(f"Failed to initialise NVML: {exc}") from exc

    try:
        try:
            driver_version = nvml.nvmlSystemGetDriverVersion().decode("utf-8")
        except AttributeError:
            driver_version = str(nvml.nvmlSystemGetDriverVersion())

        try:
            raw_cuda_version = nvml.nvmlSystemGetCudaDriverVersion_v2()
        except AttributeError:
            raw_cuda_version = nvml.nvmlSystemGetCudaDriverVersion()
        cuda_version = _format_cuda_version(int(raw_cuda_version))

        gpu_count = nvml.nvmlDeviceGetCount()
        gpus: List[GPUInfo] = []

        for index in range(gpu_count):
            handle = nvml.nvmlDeviceGetHandleByIndex(index)
            name = nvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode("utf-8")
            memory_info = nvml.nvmlDeviceGetMemoryInfo(handle)
            gpus.append(
                GPUInfo(
                    index=index,
                    name=name,
                    memory_total_bytes=int(memory_info.total),
                )
            )

        return GPUEnvironment(
            driver_version=driver_version,
            cuda_version=cuda_version,
            gpu_count=gpu_count,
            gpus=gpus,
        )
    except nvml.NVMLError as exc:  # type: ignore[attr-defined]
        raise EnvironmentCollectionError(f"Failed to query GPU information: {exc}") from exc
    finally:
        try:
            nvml.nvmlShutdown()
        except Exception:
            # Do not mask earlier exceptions with shutdown errors
            pass


def collect_gpu_environment_json() -> str:
    """Collect GPU environment information as a JSON string."""
    environment = collect_gpu_environment()
    return environment.to_json()
