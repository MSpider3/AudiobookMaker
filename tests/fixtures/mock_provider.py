"""
mock_provider.py
================
A deterministic Mock TTS Provider conforming to the BaseTTSProvider interface.
Used for local unit and integration testing without requiring GPU or large weights.
"""

from __future__ import annotations

import io
import os
import sys
import threading
import time
import numpy as np
import soundfile as sf

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from audiobook_factory.pipeline import AudiobookConfig
from audiobook_factory.tts_providers.base_tts_provider import BaseTTSProvider


class MockTTSProvider(BaseTTSProvider):
    """
    Deterministic mock TTS provider for local integration tests.
    Generates authentic non-silent audio with speech-like harmonic frequencies.
    """

    def __init__(self, config: AudiobookConfig, device: str = "cpu") -> None:
        super().__init__(config)
        self._device = device
        self._is_ready = True
        self._lock = threading.Lock()

        # Injected failure controls
        self.inject_error: Exception | None = None
        self.inject_oom: bool = False
        self.inject_silence: bool = False
        self.inject_nan: bool = False
        self.simulated_delay: float = 0.0
        self.synthesis_call_count: int = 0
        self.batch_call_count: int = 0

    @property
    def device(self) -> str:
        return self._device

    @property
    def is_ready(self) -> bool:
        return self._is_ready

    def ensure_ready(self) -> None:
        self._is_ready = True

    def cleanup(self) -> None:
        self._is_ready = False

    def estimate_cost(self, total_chars: int) -> float:
        return 0.0

    def get_name(self) -> str:
        return "mock"

    @classmethod
    def create_for_device(
        cls, device: str, config: AudiobookConfig, dtype_override: str | None = None
    ) -> MockTTSProvider:
        return cls(config, device=device)

    def _generate_synthetic_waveform(self, text: str, sample_rate: int = 24000) -> tuple[np.ndarray, float]:
        """Generate deterministic multi-harmonic audio signal with speech envelope."""
        # Calculate duration based on text length (~15 chars per second, minimum 0.5s)
        duration = max(0.5, len(text.strip()) * 0.065)
        num_samples = int(sample_rate * duration)

        if self.inject_silence:
            return np.zeros(num_samples, dtype=np.float32), duration

        if self.inject_nan:
            arr = np.ones(num_samples, dtype=np.float32)
            arr[100:200] = np.nan
            return arr, duration

        t = np.linspace(0, duration, num_samples, endpoint=False, dtype=np.float32)
        f0 = 150.0 + (hash(text) % 40)  # slightly varied pitch per text

        # 3 harmonic components
        signal = (
            0.50 * np.sin(2 * np.pi * f0 * t) +
            0.30 * np.sin(2 * np.pi * (f0 * 2) * t) +
            0.15 * np.sin(2 * np.pi * (f0 * 3) * t)
        )

        # Amplitude envelope (syllabic modulation)
        mod = 0.5 * (1.0 + np.sin(2 * np.pi * 4.0 * t))
        signal = signal * mod

        # Fades
        fade_len = min(num_samples // 4, int(0.04 * sample_rate))
        if fade_len > 0:
            signal[:fade_len] *= np.linspace(0.0, 1.0, fade_len)
            signal[-fade_len:] *= np.linspace(1.0, 0.0, fade_len)

        # Scale to -18 dBFS RMS
        rms = np.sqrt(np.mean(signal**2))
        if rms > 0:
            signal = signal * (0.12 / rms)
        signal = np.clip(signal, -0.95, 0.95).astype(np.float32)

        return signal, duration

    def _waveform_to_wav_bytes(self, signal: np.ndarray, sample_rate: int = 24000) -> bytes:
        with io.BytesIO() as buf:
            sf.write(buf, signal, sample_rate, format="WAV", subtype="PCM_16")
            return buf.getvalue()

    def synthesize(
        self,
        text: str,
        voice_ref: str | bytes,
        out_path: str | None = None,
        *,
        return_bytes: bool = False,
    ) -> tuple[str | bytes, float]:
        with self._lock:
            self.synthesis_call_count += 1
            if self.simulated_delay > 0:
                time.sleep(self.simulated_delay)

            if self.inject_oom:
                raise RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")

            if self.inject_error is not None:
                raise self.inject_error

            sr = getattr(self.config, "sample_rate", 24000)
            signal, duration = self._generate_synthetic_waveform(text, sample_rate=sr)

            if out_path:
                os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
                sf.write(out_path, signal, sr, subtype="PCM_16")
                if return_bytes:
                    wav_bytes = self._waveform_to_wav_bytes(signal, sample_rate=sr)
                    return wav_bytes, duration
                return out_path, duration
            else:
                wav_bytes = self._waveform_to_wav_bytes(signal, sample_rate=sr)
                return wav_bytes, duration

    def synthesize_batch(
        self,
        texts: list[str],
        voice_ref: bytes,
        *,
        return_bytes: bool = True,
    ) -> list[tuple[bytes | str, float]]:
        with self._lock:
            self.batch_call_count += 1
            if self.simulated_delay > 0:
                time.sleep(self.simulated_delay)

            if self.inject_oom:
                raise RuntimeError("CUDA out of memory. Tried to allocate 4.00 GiB in batch forward pass")

            if self.inject_error is not None:
                raise self.inject_error

            results = []
            sr = getattr(self.config, "sample_rate", 24000)
            for text in texts:
                signal, duration = self._generate_synthetic_waveform(text, sample_rate=sr)
                wav_bytes = self._waveform_to_wav_bytes(signal, sample_rate=sr)
                results.append((wav_bytes, duration))
            return results
