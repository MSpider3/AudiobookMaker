"""
audiobook_factory/tts_providers/qwen_provider.py
==================================================
Qwen3-TTS voice-cloning provider.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Any

from audiobook_factory.tts_providers.base_tts_provider import BaseTTSProvider

if TYPE_CHECKING:
    from audiobook_factory.pipeline import AudiobookConfig

logger = logging.getLogger(__name__)

# Trim very quiet audio edges before saving (amplitude threshold).
_TRIM_THRESHOLD: float = 0.04
_TORCH_COMPILE_MODE: str = "max-autotune"


import hashlib

class QwenTTSProvider(BaseTTSProvider):
    """Local Qwen3-TTS provider supporting all model variants (Base, CustomVoice, VoiceDesign)."""

    def __init__(self, config: AudiobookConfig, device: str | None = None) -> None:
        """Initialize the Qwen3-TTS provider instance.

        Args:
            config: AudiobookConfig settings.
            device: Target torch device string (e.g. "cuda:0", "cuda:1", "cpu").
        """
        super().__init__(config)
        self._init_with_device(device or getattr(config, "device", "cuda"), config)

    def _init_with_device(self, device: str, config: AudiobookConfig) -> None:
        """Initialize instance variables pinned to target device."""
        import torch
        self.config = config
        self._device: str = device
        self._model: Any = None
        self._loaded_model_name: str | None = None
        self._x_vector_cache: dict[str, torch.Tensor] = {}
        self._lock: threading.Lock = threading.Lock()

    @classmethod
    def create_for_device(cls, device: str, config: AudiobookConfig) -> QwenTTSProvider:
        """Factory classmethod: constructs a QwenTTSProvider instance pinned to `device`.

        Args:
            device: Target torch device string (e.g. "cuda:0", "cuda:1", "cpu").
            config: AudiobookConfig settings.

        Returns:
            An instantiated QwenTTSProvider pinned to the device.
        """
        return cls(config, device=device)

    @property
    def device(self) -> str:
        """The torch device string this provider is bound to."""
        return self._device

    def get_name(self) -> str:
        """Return display name of the provider."""
        return f"Qwen3-TTS ({self.config.tts_model_name}) [{self._device}]"

    def estimate_cost(self, total_chars: int) -> float:
        """Return USD cost estimate for synthesis (0.0 for local model)."""
        return 0.0

    def _resolve_voice_ref(self, voice_ref: str | bytes | None) -> str | None:
        """Resolves a voice reference (file path string or raw WAV bytes) into a valid file path string.

        If voice_ref is raw bytes, writes it to a cached temporary .wav file (keyed by SHA256)
        and returns the file path string. This ensures extract_x_vector and generate_voice_clone
        receive a valid file path string expected by qwen_tts.
        """
        if not voice_ref:
            return None
        if isinstance(voice_ref, str):
            return voice_ref
        if isinstance(voice_ref, bytes):
            import hashlib
            import os
            import tempfile
            key = hashlib.sha256(voice_ref).hexdigest()[:16]
            temp_path = os.path.join(tempfile.gettempdir(), f"qwen_voiceref_{key}.wav")
            if not os.path.exists(temp_path) or os.path.getsize(temp_path) == 0:
                with open(temp_path, "wb") as f:
                    f.write(voice_ref)
            return temp_path
        return str(voice_ref)

    def _ensure_x_vector_cached(self, voice_ref: str | bytes) -> str | None:
        """Pre-compute and cache the speaker x-vector for the reference voice.

        Args:
            voice_ref: Path to speaker reference WAV audio file or raw bytes.

        Returns:
            Cache key string or None.
        """
        if not voice_ref:
            return None

        ref_path = self._resolve_voice_ref(voice_ref)
        if not ref_path:
            return None

        if isinstance(voice_ref, bytes):
            key = hashlib.sha256(voice_ref).hexdigest()[:16]
        else:
            key = hashlib.sha256(str(voice_ref).encode("utf-8", errors="replace")).hexdigest()[:16]

        if key in self._x_vector_cache:
            return key

        if self._model is None or not hasattr(self._model, "model") or self._model.model is None:
            return None

        model_type = getattr(self._model.model, "tts_model_type", "base")
        if model_type != "base":
            return None

        try:
            vec = None
            if hasattr(self._model, "extract_x_vector"):
                vec = self._model.extract_x_vector(ref_path)
            elif hasattr(self._model, "get_speaker_embedding"):
                vec = self._model.get_speaker_embedding(ref_path)

            if vec is not None:
                self._x_vector_cache[key] = vec
                logger.info("    [QwenTTS] ⚡ X-vector cached under key %s", key)
                return key
        except Exception as e:
            logger.warning("    [QwenTTS] X-vector caching failed (%s) — will use voice_ref per-call.", e)
        return None

    def synthesize(
        self,
        text: str,
        voice_ref: str | bytes,
        out_path: str | None = None,
        *,
        return_bytes: bool = False,
    ) -> tuple[str | bytes, float]:
        """Synthesize speech for input text and write output WAV file or return PCM bytes."""
        import io
        import soundfile as sf
        import torch

        max_retries = 1
        for attempt in range(max_retries + 1):
            try:
                with self._lock:
                    self._ensure_initialised()
                    ref_path = self._resolve_voice_ref(voice_ref or self.config.voice_file)
                    x_key = self._ensure_x_vector_cached(ref_path) if ref_path else None

                    model_type = getattr(self._model.model, "tts_model_type", "base")

                    if model_type == "base":
                        gen_kwargs = dict(
                            text=text,
                            language=getattr(self.config, "language", "English"),
                            x_vector_only_mode=True,
                            temperature=self.config.temperature,
                            top_p=self.config.top_p,
                        )
                        if x_key is not None and x_key in self._x_vector_cache and hasattr(self._model, "generate_voice_clone"):
                            gen_kwargs["x_vector"] = self._x_vector_cache[x_key]
                        else:
                            gen_kwargs["ref_audio"] = ref_path
                        wav_data, sr = self._model.generate_voice_clone(**gen_kwargs)
                    elif model_type == "custom_voice":
                        wav_data, sr = self._model.generate_custom_voice(
                            text=text,
                            speaker=self.config.tts_timbre or "serena",
                            language=getattr(self.config, "language", "English"),
                            instruct=self.config.tts_instruct,
                            temperature=self.config.temperature,
                            top_p=self.config.top_p,
                        )
                    elif model_type == "voice_design":
                        wav_data, sr = self._model.generate_voice_design(
                            text=text,
                            instruct=self.config.tts_instruct,
                            language=getattr(self.config, "language", "English"),
                            temperature=self.config.temperature,
                            top_p=self.config.top_p,
                        )
                    else:
                        raise ValueError(f"Unknown model type: {model_type}")

                    audio = wav_data[0] if isinstance(wav_data, (list, tuple)) else wav_data
                    if hasattr(audio, "ndim") and audio.ndim > 1:
                        audio = audio[0]
                    if isinstance(audio, torch.Tensor):
                        audio = audio.cpu().float().numpy()

                    duration = len(audio) / float(sr) if sr > 0 else 0.0

                if return_bytes or out_path is None:
                    buf = io.BytesIO()
                    sf.write(buf, audio, sr, format="WAV")
                    return (buf.getvalue(), duration)
                else:
                    sf.write(out_path, audio, sr)
                    return (out_path, duration)

            except torch.cuda.OutOfMemoryError as e:
                logger.error("    [QwenTTS] CUDA OOM encountered on %s. Attempting recovery...", self._device)
                self.cleanup()
                raise RuntimeError(
                    f"CUDA Out of Memory on {self._device}. Try reducing worker_count or lowering max_len."
                ) from e
            except Exception as e:
                if attempt < max_retries:
                    logger.warning(
                        "    [QwenTTS] Synthesis failed (%s). Retrying (%d/%d)...",
                        e,
                        attempt + 1,
                        max_retries,
                    )
                    import time
                    time.sleep(1)
                    continue
                raise

    def synthesize_batch(
        self,
        texts: list[str],
        voice_ref: bytes,
        *,
        return_bytes: bool = True,
    ) -> list[tuple[bytes | str, float]]:
        """Single-GPU batched synthesis using QwenTTS forward pass.

        Acquires self._lock for the duration of the forward pass. Releases
        lock before any WAV encoding. Falls back to per-item synthesis if the
        batched call raises RuntimeError or CUDA OutOfMemoryError.

        Thread-safe via self._lock. Caller must hold exclusive ownership of
        this provider — do not call from two threads simultaneously.
        """
        # Guard: ensure model is initialized before lock acquisition
        if self._model is None:
            logger.warning(
                "Model not initialized on %s at synthesize_batch() entry. "
                "Calling ensure_ready().",
                self._device,
            )
            self.ensure_ready()

        if self._model is None:
            raise RuntimeError(
                f"Model is None on {self._device} after ensure_ready(). "
                "Cannot synthesize."
            )

        import io
        import soundfile as sf
        import torch

        if not texts:
            return []

        try:
            with self._lock:
                self._ensure_initialised()
                ref_path = self._resolve_voice_ref(voice_ref or self.config.voice_file)
                x_key = self._ensure_x_vector_cached(ref_path) if ref_path else None
                if self._model is None or not hasattr(self._model, "model") or self._model.model is None:
                    raise RuntimeError(f"QwenTTS model instance is not properly loaded on {self._device}.")
                model_type = getattr(self._model.model, "tts_model_type", "base")
                languages = [getattr(self.config, "language", "English")] * len(texts)

                if model_type == "base":
                    gen_kwargs = dict(
                        text=texts,
                        language=languages,
                        x_vector_only_mode=True,
                        temperature=self.config.temperature,
                        top_p=self.config.top_p,
                    )
                    if x_key is not None and x_key in self._x_vector_cache and hasattr(self._model, "generate_voice_clone"):
                        gen_kwargs["x_vector"] = self._x_vector_cache[x_key]
                    else:
                        gen_kwargs["ref_audio"] = [ref_path] * len(texts)
                    wav_data_list, sr = self._model.generate_voice_clone(**gen_kwargs)
                elif model_type == "custom_voice":
                    speakers = [self.config.tts_timbre or "serena"] * len(texts)
                    instructs = [self.config.tts_instruct] * len(texts)
                    wav_data_list, sr = self._model.generate_custom_voice(
                        text=texts,
                        speaker=speakers,
                        language=languages,
                        instruct=instructs,
                        temperature=self.config.temperature,
                        top_p=self.config.top_p,
                    )
                elif model_type == "voice_design":
                    instructs = [self.config.tts_instruct] * len(texts)
                    wav_data_list, sr = self._model.generate_voice_design(
                        text=texts,
                        instruct=instructs,
                        language=languages,
                        temperature=self.config.temperature,
                        top_p=self.config.top_p,
                    )
                else:
                    raise ValueError(f"Unknown model type: {model_type}")

                processed_wavs = []
                for wav_data in wav_data_list:
                    audio = wav_data[0] if isinstance(wav_data, (list, tuple)) else wav_data
                    if hasattr(audio, "ndim") and audio.ndim > 1:
                        audio = audio[0]
                    if isinstance(audio, torch.Tensor):
                        audio = audio.cpu().float().numpy()
                    processed_wavs.append(audio)

        except (RuntimeError, torch.cuda.OutOfMemoryError) as exc:
            logger.warning(
                "    [QwenTTS] Batch synthesis failed on %s (%d items): %s, falling back to per-item synthesis",
                self._device, len(texts), exc,
            )
            import gc
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

            results: list[tuple[bytes | str, float]] = []
            for text in texts:
                audio_res, duration = self.synthesize(text, voice_ref, return_bytes=True)
                results.append((audio_res, duration))
            return results

        output: list[tuple[bytes | str, float]] = []
        for audio in processed_wavs:
            buf = io.BytesIO()
            sf.write(buf, audio, sr, format="WAV")
            duration = len(audio) / float(sr) if sr > 0 else 0.0
            output.append((buf.getvalue(), duration))

        return output

    def cleanup(self) -> None:
        """Clean up loaded PyTorch model and free GPU memory."""
        import gc
        import torch
        if self._model is not None:
            del self._model
            self._model = None
            self._loaded_model_name = None
            self._x_vector_cache.clear()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def ensure_ready(self) -> None:
        """Forces model loading and verifies model is non-None.

        Raises:
            RuntimeError: If model fails to load or is None after initialization.
        """
        self._ensure_initialised()
        if self._model is None:
            raise RuntimeError(
                f"Model is None after initialization on {self._device}. "
                "Check GPU memory and model path."
            )
        logger.debug("Provider ready on %s (model loaded).", self._device)

    def _ensure_initialised(self) -> None:
        """Ensure the underlying Qwen model is loaded on self._device."""
        if self._model is not None and self._loaded_model_name == self.config.tts_model_name:
            return

        if self._model is not None:
            self.cleanup()

        self._load_model()

    def _build_model_load_kwargs(self, config: AudiobookConfig) -> dict[str, Any]:
        """Build model loading keyword arguments based on configuration.

        Handles lazy loading check for bitsandbytes when INT8 quantization is requested.
        Preserves bfloat16 for non-quantized loading.

        Args:
            config: AudiobookConfig options.

        Returns:
            Dict of keyword arguments for Qwen3TTSModel.from_pretrained.
        """
        quant = getattr(config, "quantization", "none")
        if quant == "int8":
            try:
                import bitsandbytes  # noqa: F401
                from transformers import BitsAndBytesConfig
            except ImportError:
                raise ImportError(
                    "bitsandbytes is required for INT8 quantization. "
                    "Install it with: pip install bitsandbytes>=0.41.0"
                )
            return {
                "device_map": self._device,
                "quantization_config": BitsAndBytesConfig(load_in_8bit=True),
            }
        import torch
        return {
            "device_map": self._device,
            "torch_dtype": torch.bfloat16,
        }

    def _load_model(self) -> None:
        """Load the Qwen3TTSModel on the assigned target device.

        Note: torch.compile is applied at instance-level to self._model.model
        so that each device instance maintains its own compiled PyTorch graph.
        """
        import os
        import sys
        import torch

        class DevNull:
            def write(self, msg: str) -> None: pass
            def flush(self) -> None: pass
            def isatty(self) -> bool: return False
            def close(self) -> None: pass

        orig_stdout = sys.stdout
        orig_stderr = sys.stderr
        devnull = DevNull()
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            from qwen_tts import Qwen3TTSModel
        finally:
            sys.stdout = orig_stdout
            sys.stderr = orig_stderr

        logger.info("    [QwenTTS] Loading model on %s: %s…", self._device, self.config.tts_model_name)

        try:
            import flash_attn  # noqa: F401
            attn_impl = "flash_attention_2"
            logger.info("    [QwenTTS] flash_attn detected → using FlashAttention 2.")
        except ImportError:
            attn_impl = "sdpa"
            logger.info("    [QwenTTS] flash_attn not found → falling back to SDPA attention.")

        self._model = Qwen3TTSModel.from_pretrained(
            self.config.tts_model_name,
            attn_implementation=attn_impl,
            **self._build_model_load_kwargs(self.config),
        )
        self._loaded_model_name = self.config.tts_model_name

        gen_cfg = self._model.model.generation_config
        if gen_cfg.pad_token_id is None:
            gen_cfg.pad_token_id = gen_cfg.eos_token_id

        if getattr(self.config, "torch_compile", False):
            if getattr(self.config, "quantization", "none") == "int8":
                logger.warning(
                    "torch_compile=True is ignored when quantization='int8'. "
                    "bitsandbytes INT8 kernels are incompatible with torch.compile()."
                )
            else:
                try:
                    logger.info("    [QwenTTS] ⚡ Compiling underlying transformer graphs (mode=%s)...", _TORCH_COMPILE_MODE)
                    self._model.model = torch.compile(
                        self._model.model,
                        mode=_TORCH_COMPILE_MODE,
                        fullgraph=False,
                    )
                    logger.info(
                        "torch.compile(mode='max-autotune') applied on %s. "
                        "First chapter will incur ~10–30s autotuning overhead. "
                        "All subsequent chapters will be 15–25%% faster.",
                        self._device,
                    )
                except Exception as exc:
                    logger.warning("    [QwenTTS] ⚠️ torch.compile not supported or failed: %s", exc)

        logger.info("    [QwenTTS] %s ready on %s.", self.config.tts_model_name, self._device)
