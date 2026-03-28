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

    @abstractmethod
    def synthesize(self, text: str, voice_ref: str, out_path: str) -> None:
        """Generate speech for *text* and save WAV to *out_path*."""

    @abstractmethod
    def estimate_cost(self, total_chars: int) -> float:
        """Estimated USD cost for *total_chars* characters. 0.0 = free."""

    @abstractmethod
    def get_name(self) -> str:
        """Short display name, e.g. 'Qwen3-TTS'."""

    def cleanup(self) -> None:
        """Release resources. Override if the provider holds GPU models."""


# ── Factory ───────────────────────────────────────────────────────────────────

def get_tts_provider(name: str, config: "AudiobookConfig") -> BaseTTSProvider:
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
        return QwenTTSProvider(config)

    raise ValueError(
        f"Unknown TTS provider: '{name}'. "
        f"Currently supported: 'qwen'. More providers coming in a future release."
    )
