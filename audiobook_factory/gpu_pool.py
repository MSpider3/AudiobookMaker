"""
audiobook_factory/gpu_pool.py
==============================
Generic, provider-agnostic GPU device pool and manager.

This module is the single source of truth for GPU detection, VRAM filtering,
and multi-device worker dispatch. It contains ZERO dependencies on any specific
TTS provider.
"""
from __future__ import annotations

from contextlib import contextmanager
import logging
import queue
import threading
import time
from typing import Any, Callable, Generator, TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from audiobook_factory.tts_providers.base_tts_provider import BaseTTSProvider

logger = logging.getLogger(__name__)

__all__ = ["GPUDetector", "ProviderPool", "GPUPoolManager"]

_DEFAULT_MIN_VRAM_GB: float = 5.0
_ACQUIRE_POLL_TIMEOUT_SEC: float = 0.5
_BYTES_PER_GB: float = 1024.0 * 1024.0 * 1024.0


class GPUDetector:
    """Utility class for querying CUDA devices and memory availability.

    Thread-safe static helper with no instance state.
    """

    @staticmethod
    def detect_devices() -> list[str]:
        """Detect available CUDA devices or fall back to CPU.

        Returns:
            List of device strings, e.g. ["cuda:0", "cuda:1"] or ["cpu"].
        """
        try:
            if not torch.cuda.is_available():
                return ["cpu"]
            count = torch.cuda.device_count()
            if count <= 0:
                return ["cpu"]
            return [f"cuda:{i}" for i in range(count)]
        except Exception as exc:
            logger.warning("CUDA device detection failed: %s. Falling back to CPU.", exc)
            return ["cpu"]

    @staticmethod
    def get_device_info(device: str) -> dict[str, Any]:
        """Retrieve details (name, free VRAM, total VRAM) for a device.

        Args:
            device: Device string such as "cuda:0" or "cpu".

        Returns:
            Dictionary containing device details.
        """
        if not device.startswith("cuda"):
            return {
                "device": device,
                "name": "cpu",
                "free_vram_gb": 0.0,
                "total_vram_gb": 0.0,
            }

        try:
            idx = int(device.split(":")[1]) if ":" in device else 0
            name = torch.cuda.get_device_name(idx)
            free_b, total_b = torch.cuda.mem_get_info(idx)
            return {
                "device": device,
                "name": name,
                "free_vram_gb": round(free_b / _BYTES_PER_GB, 2),
                "total_vram_gb": round(total_b / _BYTES_PER_GB, 2),
            }
        except Exception as exc:
            logger.warning("Failed to query device info for %s: %s", device, exc)
            return {
                "device": device,
                "name": "unknown",
                "free_vram_gb": 0.0,
                "total_vram_gb": 0.0,
            }

    @staticmethod
    def filter_by_min_vram(devices: list[str], min_gb: float) -> list[str]:
        """Filter a list of CUDA devices to those with sufficient free VRAM.

        Args:
            devices: List of device strings.
            min_gb: Minimum required free VRAM in gigabytes.

        Returns:
            Filtered list of device strings. Guarantees at least one device is returned.
        """
        valid_devices: list[str] = []
        for dev in devices:
            if not dev.startswith("cuda"):
                valid_devices.append(dev)
                continue

            info = GPUDetector.get_device_info(dev)
            if info["free_vram_gb"] >= min_gb:
                valid_devices.append(dev)
            else:
                logger.warning(
                    "Device %s has %.2f GB free VRAM (< required %.2f GB); skipping.",
                    dev,
                    info["free_vram_gb"],
                    min_gb,
                )

        if not valid_devices:
            logger.warning(
                "No devices met the %.2f GB free VRAM threshold. Falling back to initial device list.",
                min_gb,
            )
            return list(devices)

        return valid_devices

    @staticmethod
    def log_summary() -> None:
        """Log a formatted summary of all detected compute devices."""
        devices = GPUDetector.detect_devices()
        if devices == ["cpu"]:
            logger.info("[GPU] No CUDA device found — running on CPU")
            return

        for dev in devices:
            info = GPUDetector.get_device_info(dev)
            logger.info(
                "[GPU] %s — %s — %.1f GB free / %.1f GB total",
                info["device"],
                info["name"],
                info["free_vram_gb"],
                info["total_vram_gb"],
            )


def _check_cancellation(cancel_token: Any | None) -> bool:
    """Helper to evaluate various cancel token interface styles safely."""
    if cancel_token is None:
        return False
    if hasattr(cancel_token, "is_cancelled"):
        val = getattr(cancel_token, "is_cancelled")
        return val() if callable(val) else bool(val)
    if hasattr(cancel_token, "cancelled"):
        val = getattr(cancel_token, "cancelled")
        return val() if callable(val) else bool(val)
    return False


class ProviderPool:
    """Generic thread-safe pool for managing TTS provider instances across devices.

    Thread-safe. Uses queue.Queue for provider dispatch.
    """

    def __init__(
        self,
        provider_factory: Callable[[str], BaseTTSProvider],
        devices: list[str],
        provider_name: str,
    ) -> None:
        """Initialize the pool by instantiating one provider per device.

        Args:
            provider_factory: Callable creating a BaseTTSProvider for a device string.
            devices: List of device target strings (e.g. ["cuda:0", "cuda:1"]).
            provider_name: Display/logging name for the provider type (e.g. "qwen").
        """
        self._provider_factory = provider_factory
        self._devices = list(devices)
        self._provider_name = provider_name
        self._queue: queue.Queue[BaseTTSProvider] = queue.Queue()

        logger.info(
            "Initializing ProviderPool[%s] with %d device(s): %s",
            self._provider_name,
            len(self._devices),
            ", ".join(self._devices),
        )

        for dev in self._devices:
            provider_instance = self._provider_factory(dev)
            self._queue.put(provider_instance)

    @property
    def device_count(self) -> int:
        """Number of active device provider instances in this pool."""
        return len(self._devices)

    @property
    def devices(self) -> list[str]:
        """List of device strings managed by this pool."""
        return list(self._devices)

    @property
    def provider_name(self) -> str:
        """Provider identifier string."""
        return self._provider_name

    def is_healthy(self) -> bool:
        """Check if the pool contains ready provider instances.

        Returns:
            True if the pool has initialized device instances, False otherwise.
        """
        return self.device_count > 0

    def acquire(self, cancel_token: Any | None = None) -> BaseTTSProvider:
        """Acquire an available TTS provider from the pool.

        Blocks until a provider becomes available or cancellation is requested.

        Args:
            cancel_token: Optional token exposing an is_cancelled property/method.

        Returns:
            An available BaseTTSProvider instance.

        Raises:
            InterruptedError: If operation is cancelled while waiting for a provider.
        """
        while True:
            if _check_cancellation(cancel_token):
                raise InterruptedError("Synthesis acquisition cancelled by user.")

            try:
                return self._queue.get(timeout=_ACQUIRE_POLL_TIMEOUT_SEC)
            except queue.Empty:
                continue

    def release(self, provider: BaseTTSProvider) -> None:
        """Return a TTS provider instance to the pool queue.

        Args:
            provider: The BaseTTSProvider instance to return.
        """
        self._queue.put(provider)

    @contextmanager
    def acquire_context(
        self, cancel_token: Any | None = None
    ) -> Generator[BaseTTSProvider, None, None]:
        """Context manager to safely acquire and automatically release a provider.

        Args:
            cancel_token: Optional cancellation token.

        Yields:
            An acquired BaseTTSProvider instance.
        """
        provider = self.acquire(cancel_token=cancel_token)
        try:
            yield provider
        finally:
            self.release(provider)


class GPUPoolManager:
    """Singleton manager maintaining active provider pools by name.

    Thread-safe. Access via GPUPoolManager.instance().
    """

    _instance: GPUPoolManager | None = None
    _lock: threading.Lock = threading.Lock()

    def __init__(self) -> None:
        """Private initializer. Use GPUPoolManager.instance() instead."""
        self._pools: dict[str, ProviderPool] = {}
        self._manager_lock = threading.Lock()

    @classmethod
    def instance(cls) -> GPUPoolManager:
        """Retrieve the global GPUPoolManager singleton instance.

        Returns:
            The GPUPoolManager singleton.
        """
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = cls()
        return cls._instance

    def get_pool(
        self,
        provider_name: str,
        provider_factory: Callable[[str], BaseTTSProvider],
        min_vram_gb: float = _DEFAULT_MIN_VRAM_GB,
        gpu_count_override: int = 0,
    ) -> ProviderPool:
        """Retrieve or construct the ProviderPool for a given provider name.

        Args:
            provider_name: Unique provider string identifier (e.g. "qwen").
            provider_factory: Callable constructing a provider instance for a device.
            min_vram_gb: Minimum free VRAM threshold in GB for CUDA devices.
            gpu_count_override: Optional cap on the number of GPUs to allocate (0 = auto).

        Returns:
            The active ProviderPool instance for the specified provider.
        """
        with self._manager_lock:
            if provider_name in self._pools:
                return self._pools[provider_name]

            detected_devices = GPUDetector.detect_devices()
            if gpu_count_override > 0 and detected_devices != ["cpu"]:
                detected_devices = detected_devices[:gpu_count_override]

            filtered_devices = GPUDetector.filter_by_min_vram(
                detected_devices, min_vram_gb
            )
            GPUDetector.log_summary()

            pool = ProviderPool(
                provider_factory=provider_factory,
                devices=filtered_devices,
                provider_name=provider_name,
            )
            self._pools[provider_name] = pool
            return pool

    def all_pools(self) -> dict[str, ProviderPool]:
        """Return a snapshot dictionary of all currently active provider pools.

        Returns:
            Dictionary mapping provider names to ProviderPool objects.
        """
        with self._manager_lock:
            return dict(self._pools)

    def shutdown(self) -> None:
        """Clean up and clear all active provider pools."""
        with self._manager_lock:
            for name, pool in self._pools.items():
                logger.info("Shutting down ProviderPool[%s]", name)
            self._pools.clear()
