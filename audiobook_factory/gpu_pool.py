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
_AFFINITY_PREFER_TIMEOUT_SEC: float = 0.1
_BYTES_PER_GB: float = 1024.0 * 1024.0 * 1024.0
_DEFAULT_CHUNK_CHARS: int = 399
_VRAM_PER_CHUNK_GB: float = 0.5
_MIN_BATCH_SIZE: int = 4
_MAX_BATCH_SIZE: int = 32


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
    def suggest_batch_size(
        device: str,
        base_chunk_chars: int = _DEFAULT_CHUNK_CHARS,
    ) -> int:
        """Suggests a TTS batch size based on available VRAM for the device.

        Uses an empirical estimate of activation memory per chunk. Clamped
        between _MIN_BATCH_SIZE and _MAX_BATCH_SIZE.

        Args:
            device: CUDA device string (e.g. "cuda:0") or "cpu".
            base_chunk_chars: Character count per TTS chunk (default: 399).

        Returns:
            Integer batch size suggestion for this device.
        """
        if device == "cpu":
            return _MIN_BATCH_SIZE
        info = GPUDetector.get_device_info(device)
        free_gb = info["free_vram_gb"]
        chars_scaling = base_chunk_chars / _DEFAULT_CHUNK_CHARS if _DEFAULT_CHUNK_CHARS > 0 else 1.0
        denom = _VRAM_PER_CHUNK_GB * chars_scaling
        estimated_batches = int(free_gb / denom) if denom > 0 else _MIN_BATCH_SIZE
        return max(_MIN_BATCH_SIZE, min(_MAX_BATCH_SIZE, estimated_batches))

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


from asyncio import CancelledError

def _check_cancellation(cancel_token: Any | None) -> None:
    """Helper to evaluate cancellation and raise CancelledError if set.

    Args:
        cancel_token: Optional cancellation token object.

    Raises:
        CancelledError: If cancellation is requested.
    """
    if cancel_token is None:
        return
    is_canc = False
    if hasattr(cancel_token, "is_cancelled"):
        val = getattr(cancel_token, "is_cancelled")
        is_canc = val() if callable(val) else bool(val)
    elif hasattr(cancel_token, "cancelled"):
        val = getattr(cancel_token, "cancelled")
        is_canc = val() if callable(val) else bool(val)
    if is_canc:
        raise CancelledError("Acquisition cancelled by CancelToken")



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
        self._device_queues: dict[str, queue.Queue[BaseTTSProvider]] = {
            dev: queue.Queue(maxsize=1) for dev in self._devices
        }
        self._device_map: dict[str, BaseTTSProvider] = {}

        logger.info(
            "Initializing ProviderPool[%s] with %d device(s): %s",
            self._provider_name,
            len(self._devices),
            ", ".join(self._devices),
        )

        for dev in self._devices:
            provider_instance = self._provider_factory(dev)
            self._device_queues[dev].put(provider_instance)
            self._device_map[dev] = provider_instance

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

    def get_provider_for_device(self, device: str) -> BaseTTSProvider | None:
        """Returns the provider instance bound to a specific device, without acquiring.

        Deprecated: As of Phase 3, direct device_map access is used. This method is
        retained for backward compatibility.

        Args:
            device: CUDA device string e.g. "cuda:0".

        Returns:
            The provider instance for that device, or None if not found.
        """
        return self._device_map.get(device)

    def preferred_device_for_index(self, chapter_index: int) -> str:
        """Returns the preferred device for a chapter index.

        Deprecated: As of Phase 3, chunk dispatch uses static pre-assignment
        in chapter_pipeline.py. This method is retained for backward compatibility.
        Use pool.devices[chapter_index % pool.device_count] directly.
        """
        if not self._devices:
            return "cpu"
        return self._devices[chapter_index % len(self._devices)]

    def acquire(
        self,
        cancel_token: Any | None = None,
        preferred_device: str | None = None,
    ) -> BaseTTSProvider:
        """Acquire an available TTS provider from the pool with device affinity.

        Args:
            cancel_token: Optional cancellation token.
            preferred_device: Target device string (e.g. "cuda:0") to prefer.

        Returns:
            An available BaseTTSProvider instance.

        Raises:
            CancelledError: If cancellation is requested.
        """
        if preferred_device and preferred_device in self._device_queues:
            try:
                return self._device_queues[preferred_device].get(timeout=_AFFINITY_PREFER_TIMEOUT_SEC)
            except queue.Empty:
                pass

        while True:
            _check_cancellation(cancel_token)
            for dev_queue in self._device_queues.values():
                try:
                    return dev_queue.get(block=False)
                except queue.Empty:
                    continue
            time.sleep(_ACQUIRE_POLL_TIMEOUT_SEC)

    def release(self, provider: BaseTTSProvider) -> None:
        """Return a TTS provider instance to its device queue.

        Args:
            provider: The BaseTTSProvider instance to return.
        """
        dev = getattr(provider, "device", None)
        if dev and dev in self._device_queues:
            self._device_queues[dev].put(provider)
        else:
            for q in self._device_queues.values():
                try:
                    q.put_nowait(provider)
                    break
                except queue.Full:
                    continue

    @contextmanager
    def acquire_context(
        self,
        cancel_token: Any | None = None,
        preferred_device: str | None = None,
    ) -> Generator[BaseTTSProvider, None, None]:
        """Context manager to safely acquire and automatically release a provider."""
        provider = self.acquire(cancel_token=cancel_token, preferred_device=preferred_device)
        try:
            yield provider
        finally:
            self.release(provider)


def _warmup_provider(provider: BaseTTSProvider) -> None:
    """Forces model initialization on the provider's device.

    Called once per provider after pool construction. Ensures
    provider is ready before any synthesis request arrives,
    eliminating lazy-initialization race conditions in concurrent
    multi-chapter synthesis.

    Args:
        provider: Provider instance to warm up.
    """
    try:
        provider.ensure_ready()
        logger.info("Provider on %s warmed up successfully.", provider.device)
    except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
        logger.error(
            "Provider warmup failed on %s (%s: %s). "
            "This device will be excluded from the pool.",
            provider.device,
            type(exc).__name__,
            exc,
        )
        raise


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

            import concurrent.futures as _cf

            futures: dict[str, _cf.Future] = {}
            with _cf.ThreadPoolExecutor(
                max_workers=max(1, len(filtered_devices)),
                thread_name_prefix="provider_warmup",
            ) as warmup_executor:
                futures = {
                    device: warmup_executor.submit(_warmup_provider, provider)
                    for device, provider in pool._device_map.items()
                }
                for future in _cf.as_completed(futures.values()):
                    pass

            failed_devices = [
                device for device, future in futures.items()
                if future.exception() is not None
            ]

            if failed_devices:
                logger.warning(
                    "Warmup failed for %d device(s): %s. Removing from pool.",
                    len(failed_devices),
                    ", ".join(
                        f"{d} ({futures[d].exception()})"
                        for d in failed_devices
                    ),
                )
                for device in failed_devices:
                    pool._device_queues.pop(device, None)
                    pool._device_map.pop(device, None)
                pool._devices = [d for d in pool._devices if d not in failed_devices]

                if not pool._devices:
                    details = ", ".join(f"{d}: {futures[d].exception()}" for d in failed_devices)
                    raise RuntimeError(
                        f"All provider warmups failed ({details}). Cannot synthesize audio. "
                        "Check GPU memory and model path."
                    )

                logger.info(
                    "Continuing with %d of %d devices: %s",
                    len(pool._devices),
                    len(pool._devices) + len(failed_devices),
                    ", ".join(pool._devices),
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
