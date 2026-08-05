"""
audiobook_factory/tts_providers/base_tts_provider.py
======================================================
Abstract base class for all TTS backends.

Adding a new TTS provider in the future:
1. Create a new file  tts_providers/my_provider.py
2. Subclass BaseTTSProvider and implement the three abstract methods.
3. Register the name in get_tts_provider() below.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from audiobook_factory.pipeline import AudiobookConfig


class BaseTTSProvider(ABC):
    """
    Minimal interface every TTS provider must implement.

    Methods
    -------
    synthesize(text, voice_ref, out_path)
        Convert *text* to speech, using *voice_ref* (path to a WAV clone
        reference), and write the result to *out_path* (WAV).

    estimate_cost(total_chars) -> float
        Return an approximate cost in USD for *total_chars* characters.
        Return 0.0 for local/free providers.

    get_name() -> str
        Return the short human-readable provider name (e.g. "Qwen3-TTS").

    cleanup()
        Release GPU / model resources.  Called after all chapters are done.
    """

    def __init__(self, config: "AudiobookConfig") -> None:
        self.config = config

    @property
    @abstractmethod
    def device(self) -> str:
        """The torch device string this provider is bound to (e.g. 'cuda:0')."""
        ...

    @abstractmethod
    def synthesize(
        self,
        text: str,
        voice_ref: str | bytes,
        out_path: str | None = None,
        *,
        return_bytes: bool = False,
    ) -> tuple[str | bytes, float]:
        """Generate speech for *text* and save WAV to *out_path* or return PCM bytes."""

    @abstractmethod
    def synthesize_batch(
        self,
        texts: list[str],
        voice_ref: bytes,
        *,
        return_bytes: bool = True,
    ) -> list[tuple[bytes | str, float]]:
        """Synthesizes multiple text chunks in a single GPU forward pass.

        More efficient than calling synthesize() N times because the fixed
        per-call overhead (attention mask setup, x-vector lookup, tensor
        allocation) is paid once for the entire batch.

        Args:
            texts: List of text strings to synthesize. Each must be <= max_len chars.
            voice_ref: Preprocessed voice reference WAV bytes. Same for all items.
            return_bytes: If True, returns raw WAV bytes. If False, behavior is
                          implementation-defined (may raise NotImplementedError).
                          Batch synthesis always defaults to return_bytes=True.

        Returns:
            List of (audio, duration) tuples, one per input text, in the same
            order as the input list. audio is bytes when return_bytes=True.

        Raises:
            RuntimeError: If the batch forward pass fails. Implementations should
                          fall back to per-item synthesis on batch failure.
        """
        ...

    @abstractmethod
    def estimate_cost(self, total_chars: int) -> float:
        """Estimated USD cost for *total_chars* characters. 0.0 = free."""

    @abstractmethod
    def get_name(self) -> str:
        """Short display name, e.g. 'Qwen3-TTS'."""

    @property
    def is_ready(self) -> bool:
        """Returns True if the provider is loaded and ready for synthesis.

        Default checks for a non-None _model attribute. Subclasses may
        override for different readiness semantics.
        """
        return getattr(self, "_model", None) is not None

    def ensure_ready(self) -> None:
        """Ensures the provider is fully initialized and ready for synthesis.

        The default implementation calls _ensure_initialised() if it exists.
        Subclasses may override to perform additional readiness checks.

        This method is called by GPUPoolManager during pool warmup to
        guarantee initialization before any synthesis request arrives.

        Thread-safe: implementations must be safe to call from any thread.
        """
        if hasattr(self, "_ensure_initialised"):
            self._ensure_initialised()

    def cleanup(self) -> None:
        """Release resources. Override if the provider holds GPU models."""

    @staticmethod
    def _validate_voice_ref(voice_ref: object) -> None:
        """Validates voice_ref is bytes or str before synthesis.

        Raises:
            TypeError: If voice_ref is neither bytes nor str.
            ValueError: If voice_ref is empty or non-existent file.
        """
        if not isinstance(voice_ref, (bytes, str)):
            raise TypeError(
                f"voice_ref must be bytes or str, got {type(voice_ref).__name__}. "
                "Pass either WAV file bytes or a path string."
            )
        if isinstance(voice_ref, bytes) and len(voice_ref) < 100:
            raise ValueError(
                f"voice_ref bytes too short ({len(voice_ref)}). "
                "The WAV data appears to be empty or corrupted."
            )
        if isinstance(voice_ref, str) and not __import__("os").path.exists(voice_ref):
            raise ValueError(
                f"voice_ref path does not exist: {voice_ref}"
            )

    @classmethod
    def create_for_device(cls, device: str, config: "AudiobookConfig", dtype_override: str | None = None) -> "BaseTTSProvider":
        """Factory classmethod: constructs a provider instance pinned to `device`.

        Args:
            device: Target torch device string, e.g. "cuda:0", "cuda:1", "cpu".
            config: AudiobookConfig options.
            dtype_override: If set, overrides torch_dtype selection.

        Returns:
            An instantiated BaseTTSProvider pinned to the device.
        """
        raise NotImplementedError("Subclasses must implement create_for_device")


# ── Factory ───────────────────────────────────────────────────────────────────

def get_tts_provider(
    name: str,
    config: "AudiobookConfig",
    device: str | None = None,
    dtype_override: str | None = None,
) -> BaseTTSProvider:
    """
    Return an instantiated provider for *name*.

    Currently supported
    -------------------
    "qwen"  — Qwen3-TTS-1.7B local voice-cloning model (default)

    More providers will be registered here in the future.
    """
    name = name.lower().strip()

    if name in ("qwen", "qwen3", "qwen3-tts", ""):
        from audiobook_factory.tts_providers.qwen_provider import QwenTTSProvider
        if device is not None:
            return QwenTTSProvider.create_for_device(device, config, dtype_override=dtype_override)
        return QwenTTSProvider(config, device=device, dtype_override=dtype_override)

    raise ValueError(
        f"Unknown TTS provider: '{name}'. "
        f"Currently supported: 'qwen'. More providers coming in a future release."
    )


