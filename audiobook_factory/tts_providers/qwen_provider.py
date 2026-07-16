"""
audiobook_factory/tts_providers/qwen_provider.py
==================================================
Qwen3-TTS voice-cloning provider.
"""

from typing import TYPE_CHECKING
from audiobook_factory.tts_providers.base_tts_provider import BaseTTSProvider

if TYPE_CHECKING:
    from audiobook_factory.pipeline import AudiobookConfig

# Trim very quiet audio edges before saving (amplitude threshold).
_TRIM_THRESHOLD = 0.04


class QwenTTSProvider(BaseTTSProvider):
    """
    Local Qwen3-TTS provider supporting all model variants (Base, CustomVoice, VoiceDesign).
    """

    def __init__(self, config: "AudiobookConfig") -> None:
        super().__init__(config)
        self._model = None
        self._loaded_model_name = None
        self._cached_x_vector = None       # cached speaker embedding
        self._cached_voice_path = None     # path the cached x-vector was built from
        import threading
        self._lock = threading.Lock()

    # ── BaseTTSProvider interface ─────────────────────────────────────────────

    def get_name(self) -> str:
        return f"Qwen3-TTS ({self.config.tts_model_name})"

    def estimate_cost(self, total_chars: int) -> float:
        return 0.0

    def _ensure_x_vector_cached(self, voice_ref: str) -> None:
        """Pre-compute and cache the speaker x-vector for the reference voice.
        This avoids re-reading and encoding the entire voice WAV on every chunk."""
        if self._cached_x_vector is not None and self._cached_voice_path == voice_ref:
            return  # already cached for this voice file

        import torch
        model_type = getattr(self._model.model, "tts_model_type", "base")
        if model_type != "base" or not voice_ref:
            return  # x-vector caching only applies to Base voice clone

        try:
            # Extract the x-vector embedding from the reference audio
            if hasattr(self._model, 'extract_x_vector'):
                self._cached_x_vector = self._model.extract_x_vector(voice_ref)
                self._cached_voice_path = voice_ref
                print(f"    [QwenTTS] ⚡ X-vector cached for: {voice_ref}")
            elif hasattr(self._model, 'get_speaker_embedding'):
                self._cached_x_vector = self._model.get_speaker_embedding(voice_ref)
                self._cached_voice_path = voice_ref
                print(f"    [QwenTTS] ⚡ Speaker embedding cached for: {voice_ref}")
            else:
                # Model does not expose x-vector extraction — skip caching
                pass
        except Exception as e:
            print(f"    [QwenTTS] X-vector caching failed ({e}) — will use voice_ref per-call.")
            self._cached_x_vector = None

    def synthesize(self, text: str, voice_ref: str, out_path: str) -> None:
        import soundfile as sf
        import torch

        max_retries = 1
        for attempt in range(max_retries + 1):
            try:
                with self._lock:
                    self._ensure_initialised()
                    self._ensure_x_vector_cached(voice_ref or self.config.voice_file)

                    model_type = getattr(self._model.model, "tts_model_type", "base")

                    if model_type == "base":
                        # Use cached x-vector if available to avoid redundant voice encoding
                        gen_kwargs = dict(
                            text=text,
                            language=getattr(self.config, "language", "English"),
                            x_vector_only_mode=True,
                            temperature=self.config.temperature,
                            top_p=self.config.top_p,
                        )
                        if self._cached_x_vector is not None and hasattr(self._model, 'generate_voice_clone'):
                            gen_kwargs['x_vector'] = self._cached_x_vector
                        else:
                            gen_kwargs['ref_audio'] = voice_ref or self.config.voice_file
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

                    # Detach audio from GPU while still holding lock
                    audio = wav_data[0] if isinstance(wav_data, (list, tuple)) else wav_data
                    if hasattr(audio, "ndim") and audio.ndim > 1:
                        audio = audio[0]
                    if isinstance(audio, torch.Tensor):
                        audio = audio.cpu().float().numpy()

                # ── GPU is now free — file I/O outside the lock ───────────────
                sf.write(out_path, audio, sr)
                break

            except torch.cuda.OutOfMemoryError as e:
                print("    [QwenTTS] CUDA OOM encountered. Attempting recovery...")
                self.cleanup()
                raise RuntimeError("CUDA Out of Memory. Try reducing worker_count to 1 or lowering max_len.") from e
            except Exception as e:
                if attempt < max_retries:
                    print(f"    [QwenTTS] Synthesis failed ({e}). Retrying ({attempt+1}/{max_retries})...")
                    import time
                    time.sleep(1)
                    continue
                raise

    def synthesize_batch(self, texts: list[str], voice_refs: list[str], out_paths: list[str]) -> list[float]:
        import soundfile as sf
        import torch

        durations = [0.0] * len(texts)
        if not texts:
            return durations

        max_retries = 1
        for attempt in range(max_retries + 1):
            try:
                with self._lock:
                    self._ensure_initialised()
                    # Cache x-vector for the first voice ref (usually all the same)
                    primary_voice = voice_refs[0] if voice_refs else self.config.voice_file
                    self._ensure_x_vector_cached(primary_voice or self.config.voice_file)

                    model_type = getattr(self._model.model, "tts_model_type", "base")
                    languages = [getattr(self.config, "language", "English")] * len(texts)

                    if model_type == "base":
                        # Try using cached x-vector for batch calls too
                        gen_kwargs = dict(
                            text=texts,
                            language=languages,
                            x_vector_only_mode=True,
                            temperature=self.config.temperature,
                            top_p=self.config.top_p,
                        )
                        if self._cached_x_vector is not None and hasattr(self._model, 'generate_voice_clone'):
                            gen_kwargs['x_vector'] = self._cached_x_vector
                        else:
                            voice_files = [vr or self.config.voice_file for vr in voice_refs]
                            gen_kwargs['ref_audio'] = voice_files
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

                    # Detach all audio from GPU while holding lock
                    processed_wavs = []
                    for wav_data in wav_data_list:
                        audio = wav_data[0] if isinstance(wav_data, (list, tuple)) else wav_data
                        if hasattr(audio, "ndim") and audio.ndim > 1:
                            audio = audio[0]
                        if isinstance(audio, torch.Tensor):
                            audio = audio.cpu().float().numpy()
                        processed_wavs.append(audio)

                # ── Save WAV files outside the lock ──
                for i, (audio, out_path) in enumerate(zip(processed_wavs, out_paths)):
                    sf.write(out_path, audio, sr)
                    durations[i] = len(audio) / sr
                
                return durations

            except torch.cuda.OutOfMemoryError as e:
                print("    [QwenTTS] CUDA OOM encountered during batch synthesis. Attempting recovery...")
                self.cleanup()
                raise RuntimeError("CUDA Out of Memory in batch synthesis. Try reducing worker_count (batch size).") from e
            except Exception as e:
                if attempt < max_retries:
                    print(f"    [QwenTTS] Batch synthesis failed ({e}). Retrying ({attempt+1}/{max_retries})...")
                    import time
                    time.sleep(1)
                    continue
                raise

    def cleanup(self) -> None:
        import torch
        import gc
        if self._model is not None:
            del self._model
            self._model = None
            self._loaded_model_name = None
            self._cached_x_vector = None
            self._cached_voice_path = None
            gc.collect()
            torch.cuda.empty_cache()

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _ensure_initialised(self) -> None:
        if self._model is not None and self._loaded_model_name == self.config.tts_model_name:
            return

        # If name changed, clean up first
        if self._model is not None:
            self.cleanup()

        self._load_model()

    def _load_model(self) -> None:
        import torch
        import os, sys

        class DevNull:
            def write(self, msg): pass
            def flush(self): pass
            def isatty(self): return False
            def close(self): pass

        # Suppress annoying SoX not found and TF printout noise during import
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

        print(f"    [QwenTTS] Loading model: {self.config.tts_model_name}…")

        # Auto-detect whether flash_attn is installed.
        # flash_attention_2 requires the separate `flash_attn` package which is not
        # available on every GPU (e.g. T4 in free Colab). Fall back to PyTorch's
        # built-in SDPA which is fast and always available.
        try:
            import flash_attn  # noqa: F401
            attn_impl = "flash_attention_2"
            print("    [QwenTTS] flash_attn detected → using FlashAttention 2.")
        except ImportError:
            attn_impl = "sdpa"
            print("    [QwenTTS] flash_attn not found → falling back to SDPA attention.")

        self._model = Qwen3TTSModel.from_pretrained(
            self.config.tts_model_name,
            device_map=self.config.device,
            torch_dtype=torch.bfloat16,
            attn_implementation=attn_impl,
        )
        self._loaded_model_name = self.config.tts_model_name

        # Suppress pad_token_id warning.
        gen_cfg = self._model.model.generation_config
        if gen_cfg.pad_token_id is None:
            gen_cfg.pad_token_id = gen_cfg.eos_token_id

        # Compile model for raw acceleration if requested
        if getattr(self.config, "torch_compile", False):
            try:
                print("    [QwenTTS] ⚡ Compiling underlying transformer graphs with torch.compile...")
                self._model.model = torch.compile(self._model.model, mode="reduce-overhead")
                print("    [QwenTTS] ⚡ Model compile registered.")
            except Exception as e:
                print(f"    [QwenTTS] ⚠️ torch.compile not supported or failed: {e}")

        print(f"    [QwenTTS] {self.config.tts_model_name} ready.")

