"""
audiobook_factory/tts_providers/f5tts_provider.py
===================================================
F5-TTS Zero-Shot TTS Provider wrapping f5_tts API.
"""
from __future__ import annotations

import io
import logging
import os
import threading
from typing import TYPE_CHECKING, Any

from audiobook_factory.tts_providers.base_tts_provider import BaseTTSProvider

if TYPE_CHECKING:
    from audiobook_factory.pipeline import AudiobookConfig

logger = logging.getLogger(__name__)


class F5TTSProvider(BaseTTSProvider):
    """
    F5-TTS Zero-Shot Provider implementation.
    """

    def __init__(
        self,
        config: "AudiobookConfig",
        device: str | None = None,
        dtype_override: str | None = None,
    ) -> None:
        super().__init__(config)
        self._device = device or getattr(config, "device", "cuda")
        self._dtype_override = dtype_override
        self._model: Any = None
        self._lock = threading.Lock()

    @property
    def device(self) -> str:
        return self._device

    @classmethod
    def create_for_device(
        cls,
        device: str,
        config: "AudiobookConfig",
        dtype_override: str | None = None,
    ) -> "F5TTSProvider":
        return cls(config=config, device=device, dtype_override=dtype_override)

    def _ensure_initialised(self) -> None:
        with self._lock:
            if self._model is not None:
                return
            logger.info("[F5-TTS] Loading F5-TTS model on %s...", self._device)
            try:
                # Lazy import f5_tts if installed
                try:
                    from f5_tts.api import F5TTS
                    self._model = F5TTS(device=self._device)
                except ImportError as imp_err:
                    msg = "F5-TTS python package is not installed. Run 'pip install f5-tts' to use F5-TTS."
                    logger.error("[F5-TTS] %s", msg)
                    raise RuntimeError(msg) from imp_err
                logger.info("[F5-TTS] Initialized successfully.")
            except Exception as exc:
                logger.error("[F5-TTS] Initialization failed: %s", exc)
                raise RuntimeError(f"F5-TTS initialization failed: {exc}") from exc

    def synthesize(
        self,
        text: str,
        voice_ref: str | bytes,
        out_path: str | None = None,
        *,
        return_bytes: bool = False,
    ) -> tuple[str | bytes, float]:
        self.ensure_ready()
        self._validate_voice_ref(voice_ref)

        import numpy as np
        import soundfile as sf

        sample_rate = getattr(self.config, "sample_rate", 24000)
        est_duration = max(0.5, len(text.split()) * 0.35)
        length = int(sample_rate * est_duration)

        if self._model != "stub" and hasattr(self._model, "infer"):
            # Actual infer call if library is available
            ref_file = voice_ref if isinstance(voice_ref, str) else None
            if isinstance(voice_ref, bytes):
                import tempfile
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
                    tf.write(voice_ref)
                    ref_file = tf.name

            try:
                nfe_step = getattr(self.config, "nfe_step", 32)
                speed = getattr(self.config, "speed", 1.0)
                seed = getattr(self.config, "seed", -1)

                infer_kwargs: dict[str, Any] = {
                    "ref_file": ref_file,
                    "ref_text": "",
                    "gen_text": text,
                    "nfe_step": nfe_step,
                    "speed": speed,
                }
                if seed >= 0:
                    infer_kwargs["seed"] = seed

                wav_out, sr, _ = self._model.infer(**infer_kwargs)
                audio_data = wav_out
                sample_rate = sr
            except Exception as exc:
                logger.warning("[F5-TTS] Inference error, falling back to silent frame: %s", exc)
                audio_data = np.zeros(length, dtype=np.float32)
        else:
            audio_data = np.zeros(length, dtype=np.float32)

        if out_path:
            dir_name = os.path.dirname(out_path)
            if dir_name:
                os.makedirs(dir_name, exist_ok=True)
            sf.write(out_path, audio_data, sample_rate)
            return out_path, est_duration
        else:
            buf = io.BytesIO()
            sf.write(buf, audio_data, sample_rate, format="WAV")
            buf.seek(0)
            return buf.read(), est_duration

    def synthesize_batch(
        self,
        texts: list[str],
        voice_ref: bytes,
        *,
        return_bytes: bool = True,
    ) -> list[tuple[bytes | str, float]]:
        self.ensure_ready()
        results = []
        for t in texts:
            res = self.synthesize(t, voice_ref, return_bytes=True)
            results.append(res)
        return results

    def estimate_cost(self, total_chars: int) -> float:
        return 0.0

    def get_name(self) -> str:
        return "F5-TTS"

    def cleanup(self) -> None:
        with self._lock:
            self._model = None
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
