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
import os

_MAX_VOICE_REF_CACHE: int = 8
_VOICE_REF_CACHE: dict[str, str] = {}
_VOICE_REF_LOCK = threading.Lock()


def _voice_ref_cache_get(key: str) -> str | None:
    with _VOICE_REF_LOCK:
        if key in _VOICE_REF_CACHE:
            path = _VOICE_REF_CACHE.pop(key)
            _VOICE_REF_CACHE[key] = path
            if os.path.exists(path) and os.path.getsize(path) > 0:
                return path
    return None


def _voice_ref_cache_put(key: str, path: str) -> None:
    with _VOICE_REF_LOCK:
        if key in _VOICE_REF_CACHE:
            _VOICE_REF_CACHE.pop(key)
        elif len(_VOICE_REF_CACHE) >= _MAX_VOICE_REF_CACHE:
            oldest_key, oldest_path = next(iter(_VOICE_REF_CACHE.items()))
            _VOICE_REF_CACHE.pop(oldest_key)
            try:
                if os.path.exists(oldest_path):
                    os.unlink(oldest_path)
            except OSError:
                pass
        _VOICE_REF_CACHE[key] = path


def _sanitize_dict_keys(obj: Any) -> None:
    """Converts dict view objects to lists in model config attributes.

    Python 3.12 introduced stricter pickling that rejects dict_keys,
    dict_values, and dict_items view objects. HuggingFace transformers
    may store these internally in GenerationConfig. This method runs
    once after model loading to make all config attributes picklable.

    Covers: obj, obj.config, obj.generation_config.
    """
    if obj is None:
        return
    _VIEW_TYPES = (type({}.keys()), type({}.values()), type({}.items()))

    def _sanitize_obj(target: Any) -> None:
        if target is None or not hasattr(target, "__dict__"):
            return
        for attr_name, value in list(vars(target).items()):
            if isinstance(value, _VIEW_TYPES):
                setattr(target, attr_name, list(value))
            elif isinstance(value, dict):
                for k, v in list(value.items()):
                    if isinstance(v, _VIEW_TYPES):
                        value[k] = list(v)

    _sanitize_obj(obj)
    _sanitize_obj(getattr(obj, "config", None))
    _sanitize_obj(getattr(obj, "generation_config", None))
    if hasattr(obj, "model"):
        _sanitize_obj(obj.model)
        _sanitize_obj(getattr(obj.model, "config", None))
        _sanitize_obj(getattr(obj.model, "generation_config", None))


class QwenTTSProvider(BaseTTSProvider):
    """Local Qwen3-TTS provider supporting all model variants (Base, CustomVoice, VoiceDesign)."""

    def __init__(
        self,
        config: AudiobookConfig,
        device: str | None = None,
        dtype_override: str | None = None,
    ) -> None:
        """Initialize the Qwen3-TTS provider instance.

        Args:
            config: AudiobookConfig settings.
            device: Target torch device string (e.g. "cuda:0", "cuda:1", "cpu").
            dtype_override: Optional torch_dtype string ("float16", "bfloat16", "float32").
        """
        super().__init__(config)
        self._dtype_override: str | None = dtype_override
        self._init_with_device(device or getattr(config, "device", "cuda"), config)

    def _init_with_device(self, device: str, config: AudiobookConfig) -> None:
        """Initialize instance variables pinned to target device."""
        from audiobook_factory.preflight import _apply_python312_pickle_patch
        _apply_python312_pickle_patch()

        import torch
        self.config = config
        self._device: str = device
        self._model: Any = None
        self._loaded_model_name: str | None = None
        self._x_vector_cache: dict[str, torch.Tensor] = {}
        self._voice_prompt_cache: dict[str, Any] = {}
        self._transcript_cache: dict[str, str] = {}
        self._asr_pipe: Any = None
        self._lock: threading.Lock = threading.Lock()

    @classmethod
    def create_for_device(
        cls,
        device: str,
        config: AudiobookConfig,
        dtype_override: str | None = None,
    ) -> QwenTTSProvider:
        """Factory classmethod: constructs a QwenTTSProvider instance pinned to `device`.

        Args:
            device: Target torch device string (e.g. "cuda:0", "cuda:1", "cpu").
            config: AudiobookConfig settings.
            dtype_override: Optional torch_dtype string ("float16", "bfloat16", "float32").

        Returns:
            An instantiated QwenTTSProvider pinned to the device.
        """
        return cls(config, device=device, dtype_override=dtype_override)

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

    def _bind_cuda_device(self) -> None:
        """Sets the current active CUDA device to self._device to ensure thread safety on multi-GPU systems."""
        if self._device and self._device.startswith("cuda"):
            try:
                import torch
                idx = int(self._device.split(":")[1]) if ":" in self._device else 0
                torch.cuda.set_device(idx)
            except Exception as e:
                logger.warning("Failed to set CUDA device to %s: %s", self._device, e)

    def _resolve_voice_ref(self, voice_ref: str | bytes | None) -> str | None:
        """Resolves a voice reference (file path string or raw WAV bytes) into a valid file path string.

        If voice_ref is raw bytes, writes it to a cached temporary .wav file (keyed by SHA256)
        and returns the file path string. Subsequent calls with the same bytes return the cached path.
        """
        if not voice_ref:
            return None
        if isinstance(voice_ref, str):
            return voice_ref
        if isinstance(voice_ref, bytes):
            import tempfile
            key = hashlib.sha256(voice_ref).hexdigest()[:16]
            cached = _voice_ref_cache_get(key)
            if cached is not None:
                return cached
            temp_path = os.path.join(tempfile.gettempdir(), f"qwen_voiceref_{key}.wav")
            if not os.path.exists(temp_path) or os.path.getsize(temp_path) == 0:
                with open(temp_path, "wb") as f:
                    f.write(voice_ref)
            _voice_ref_cache_put(key, temp_path)
            logger.debug("Voice ref cached to %s", temp_path)
            return temp_path
        raise ValueError(f"voice_ref must be bytes or str, got {type(voice_ref).__name__}")

    def _get_voice_transcript(self, ref_path: str) -> str | None:
        """Retrieves or transcribes reference audio text.

        Tries in order:
        1. config.voice_transcript or config.ref_text if configured.
        2. In-memory transcript cache.
        3. Sidecar .txt file if exists alongside ref_path.
        4. Automatic speech recognition via lightweight Whisper (whisper-tiny).
        """
        # 1. Direct configuration
        cfg_transcript = getattr(self.config, "voice_transcript", None) or getattr(self.config, "ref_text", None)
        if cfg_transcript and str(cfg_transcript).strip():
            return str(cfg_transcript).strip()

        # 2. In-memory cache
        if ref_path in self._transcript_cache:
            return self._transcript_cache[ref_path]

        # 3. Sidecar transcript file (e.g. voice.wav.txt or voice.txt)
        import os
        sidecars = [ref_path + ".txt", os.path.splitext(ref_path)[0] + ".txt"]
        for sidecar in sidecars:
            if os.path.exists(sidecar) and os.path.isfile(sidecar):
                try:
                    with open(sidecar, "r", encoding="utf-8", errors="replace") as f:
                        text = f.read().strip()
                    if text:
                        self._transcript_cache[ref_path] = text
                        logger.info("    [QwenTTS] 🎙️ Loaded reference transcript from sidecar %s: '%s'", os.path.basename(sidecar), text[:60])
                        return text
                except Exception as exc:
                    logger.debug("Failed reading sidecar %s: %s", sidecar, exc)

        # 4. Automatic speech recognition with Whisper
        try:
            import torch
            from transformers import pipeline
            if getattr(self, "_asr_pipe", None) is None:
                device_idx = int(self._device.split(":")[1]) if (self._device.startswith("cuda") and ":" in self._device) else (0 if self._device == "cuda" else -1)
                self._asr_pipe = pipeline(
                    "automatic-speech-recognition",
                    model="openai/whisper-tiny",
                    device=device_idx if (torch.cuda.is_available() and device_idx >= 0) else -1,
                    torch_dtype=torch.float16 if (torch.cuda.is_available() and device_idx >= 0) else torch.float32,
                )
            res = self._asr_pipe(ref_path)
            transcript = (res.get("text") or "").strip() if isinstance(res, dict) else ""
            if transcript:
                self._transcript_cache[ref_path] = transcript
                logger.info("    [QwenTTS] 🎙️ Auto-transcribed reference audio '%s': \"%s\"", os.path.basename(ref_path), transcript)
                return transcript
        except Exception as asr_err:
            logger.debug("    [QwenTTS] Whisper auto-transcription skipped/unavailable: %s", asr_err)

        return None

    def _ensure_voice_prompt_cached(self, voice_ref: str | bytes) -> tuple[Any, str | None, str | None]:
        """Pre-computes and caches voice clone prompt and speaker embeddings.

        Returns:
            Tuple of (voice_clone_prompt_obj_or_None, x_vector_key_or_None, ref_text_or_None).
        """
        self._bind_cuda_device()
        if not voice_ref:
            return None, None, None

        ref_path = self._resolve_voice_ref(voice_ref)
        if not ref_path:
            return None, None, None

        if isinstance(voice_ref, bytes):
            key = hashlib.sha256(voice_ref).hexdigest()[:16]
        else:
            key = hashlib.sha256(str(voice_ref).encode("utf-8", errors="replace")).hexdigest()[:16]

        ref_text = self._get_voice_transcript(ref_path)

        if self._model is None or not hasattr(self._model, "model") or self._model.model is None:
            return None, None, ref_text

        # Return cached voice clone prompt if valid
        if key in self._voice_prompt_cache:
            return self._voice_prompt_cache[key], key, ref_text

        x_key = self._ensure_x_vector_cached(voice_ref)

        if hasattr(self._model, "create_voice_clone_prompt"):
            try:
                # Use ICL (In-Context Learning) mode if ref_text is present, preserving gender, timbre, and acoustics
                use_xvec_only = (ref_text is None or len(ref_text.strip()) == 0)
                prompt = self._model.create_voice_clone_prompt(
                    ref_audio=ref_path,
                    ref_text=ref_text,
                    x_vector_only_mode=use_xvec_only,
                )
                self._voice_prompt_cache[key] = prompt
                logger.info("    [QwenTTS] ⚡ Voice clone prompt cached under key %s (ICL mode=%s)", key, not use_xvec_only)
                return prompt, x_key, ref_text
            except Exception as e:
                logger.warning("    [QwenTTS] create_voice_clone_prompt failed (%s) — falling back to per-call voice_ref.", e)

        return None, x_key, ref_text

    def _ensure_x_vector_cached(self, voice_ref: str | bytes) -> str | None:
        """Pre-compute and cache the speaker x-vector for the reference voice.

        Args:
            voice_ref: Path to speaker reference WAV audio file or raw bytes.

        Returns:
            Cache key string or None.
        """
        self._bind_cuda_device()
        if not voice_ref:
            return None

        ref_path = self._resolve_voice_ref(voice_ref)
        if not ref_path:
            return None

        if isinstance(voice_ref, bytes):
            key = hashlib.sha256(voice_ref).hexdigest()[:16]
        else:
            key = hashlib.sha256(str(voice_ref).encode("utf-8", errors="replace")).hexdigest()[:16]

        import torch
        if key in self._x_vector_cache:
            cached = self._x_vector_cache[key]
            if hasattr(cached, "device") and (
                cached.device.type == "cpu" or
                str(cached.device) != self._device or
                cached.numel() == 0
            ):
                logger.warning(
                    "    [QwenTTS] Cached x-vector for key %s is on wrong device "
                    "or empty (%s vs expected %s). Recomputing.",
                    key, getattr(cached, "device", "unknown"), self._device,
                )
                del self._x_vector_cache[key]
            else:
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
                if isinstance(vec, torch.Tensor) and str(vec.device) != self._device:
                    try:
                        vec = vec.to(self._device)
                    except Exception as e:
                        logger.warning("Failed to move x-vector to device %s: %s", self._device, e)
                self._x_vector_cache[key] = vec
                logger.info("    [QwenTTS] ⚡ X-vector cached under key %s (device %s)", key, getattr(vec, 'device', 'unknown'))
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
        self._bind_cuda_device()
        self._validate_voice_ref(voice_ref or self.config.voice_file)
        import io
        import soundfile as sf
        import torch

        max_retries = 1
        for attempt in range(max_retries + 1):
            try:
                with self._lock:
                    self._ensure_initialised()
                    ref_path = self._resolve_voice_ref(voice_ref or self.config.voice_file)
                    model_type = getattr(self._model.model, "tts_model_type", "base")

                    if model_type == "base":
                        prompt, x_key, ref_text = self._ensure_voice_prompt_cached(ref_path) if ref_path else (None, None, None)
                        gen_kwargs = dict(
                            text=text,
                            language=getattr(self.config, "language", "English"),
                            temperature=self.config.temperature,
                            top_p=self.config.top_p,
                        )
                        if prompt is not None:
                            gen_kwargs["voice_clone_prompt"] = prompt
                        elif ref_text:
                            gen_kwargs["ref_audio"] = ref_path
                            gen_kwargs["ref_text"] = ref_text
                            gen_kwargs["x_vector_only_mode"] = False
                        elif x_key is not None and x_key in self._x_vector_cache and hasattr(self._model, "generate_voice_clone"):
                            gen_kwargs["x_vector"] = self._x_vector_cache[x_key]
                            gen_kwargs["x_vector_only_mode"] = True
                        else:
                            gen_kwargs["ref_audio"] = ref_path
                            gen_kwargs["x_vector_only_mode"] = True
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
        self._bind_cuda_device()
        self._validate_voice_ref(voice_ref or self.config.voice_file)
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
                if self._model is None or not hasattr(self._model, "model") or self._model.model is None:
                    raise RuntimeError(f"QwenTTS model instance is not properly loaded on {self._device}.")
                model_type = getattr(self._model.model, "tts_model_type", "base")
                languages = [getattr(self.config, "language", "English")] * len(texts)

                if model_type == "base":
                    prompt, x_key, ref_text = self._ensure_voice_prompt_cached(ref_path) if ref_path else (None, None, None)
                    gen_kwargs = dict(
                        text=texts,
                        language=languages,
                        temperature=self.config.temperature,
                        top_p=self.config.top_p,
                    )
                    if prompt is not None:
                        gen_kwargs["voice_clone_prompt"] = prompt
                    elif ref_text:
                        gen_kwargs["ref_audio"] = [ref_path] * len(texts)
                        gen_kwargs["ref_text"] = [ref_text] * len(texts)
                        gen_kwargs["x_vector_only_mode"] = False
                    elif x_key is not None and x_key in self._x_vector_cache and hasattr(self._model, "generate_voice_clone"):
                        gen_kwargs["x_vector"] = self._x_vector_cache[x_key]
                        gen_kwargs["x_vector_only_mode"] = True
                    else:
                        gen_kwargs["ref_audio"] = [ref_path] * len(texts)
                        gen_kwargs["x_vector_only_mode"] = True
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
        for i, audio in enumerate(processed_wavs):
            if audio is None or (hasattr(audio, "__len__") and len(audio) == 0):
                logger.error(
                    "    [QwenTTS] Batch synthesis returned empty audio for chunk %d "
                    "on %s. Model may be producing silence.",
                    i, self._device,
                )
                raise RuntimeError(
                    f"Empty audio from batch synthesis on {self._device} "
                    f"for chunk {i}. Check voice reference and model state."
                )

            buf = io.BytesIO()
            sf.write(buf, audio, sr, format="WAV")
            duration = len(audio) / float(sr) if sr > 0 else 0.0
            audio_bytes = buf.getvalue()

            if len(audio_bytes) < 100:
                logger.error(
                    "    [QwenTTS] Batch synthesis produced WAV output under 100 bytes (%d bytes) for chunk %d on %s.",
                    len(audio_bytes), i, self._device,
                )
                raise RuntimeError(
                    f"Empty or corrupted WAV audio ({len(audio_bytes)} bytes) from batch synthesis on {self._device} "
                    f"for chunk {i}. Check voice reference and model state."
                )

            logger.debug(
                "[synth_batch] chunk %d: audio_type=%s audio_len=%d duration=%.2f",
                i, type(audio_bytes).__name__, len(audio_bytes), duration,
            )
            output.append((audio_bytes, duration))

        return output

    def cleanup(self) -> None:
        """Clean up loaded PyTorch model and free GPU memory."""
        self._bind_cuda_device()
        import gc
        import torch
        if self._model is not None:
            del self._model
            self._model = None
            self._loaded_model_name = None
            self._x_vector_cache.clear()
            self._voice_prompt_cache.clear()
            self._transcript_cache.clear()
            self._asr_pipe = None
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
        if getattr(self, "_dtype_override", None) == "float16":
            dtype = torch.float16
        elif getattr(self, "_dtype_override", None) == "bfloat16":
            dtype = torch.bfloat16
        elif getattr(self, "_dtype_override", None) == "float32":
            dtype = torch.float32
        else:
            supports_bf16 = (
                torch.cuda.is_available()
                and hasattr(torch.cuda, "is_bf16_supported")
                and torch.cuda.is_bf16_supported()
            )
            dtype = torch.bfloat16 if supports_bf16 else torch.float16
        return {
            "device_map": self._device,
            "torch_dtype": dtype,
        }

    def _load_model(self) -> None:
        """Load the Qwen3TTSModel on the assigned target device.

        Note: torch.compile is applied at instance-level to self._model.model
        so that each device instance maintains its own compiled PyTorch graph.
        """
        self._bind_cuda_device()
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

        _sanitize_dict_keys(self._model)

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
