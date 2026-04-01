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
        import threading
        self._lock = threading.Lock()

    # ── BaseTTSProvider interface ─────────────────────────────────────────────

    def get_name(self) -> str:
        return f"Qwen3-TTS ({self.config.tts_model_name})"

    def estimate_cost(self, total_chars: int) -> float:
        return 0.0

    def synthesize(self, text: str, voice_ref: str, out_path: str) -> None:
        import soundfile as sf
        import torch

        max_retries = 1
        for attempt in range(max_retries + 1):
            try:
                with self._lock:
                    self._ensure_initialised()

                    model_type = getattr(self._model.model, "tts_model_type", "base")

                    if model_type == "base":
                        wav_data, sr = self._model.generate_voice_clone(
                            text=text,
                            language=getattr(self.config, "language", "English"),
                            ref_audio=voice_ref or self.config.voice_file,
                            x_vector_only_mode=True,
                            temperature=self.config.temperature,
                            top_p=self.config.top_p,
                        )
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

                    # ── Release the lock immediately after generation ─────────────────
                    audio = wav_data[0] if isinstance(wav_data, (list, tuple)) else wav_data
                    if hasattr(audio, "ndim") and audio.ndim > 1:
                        audio = audio[0]
                    if isinstance(audio, torch.Tensor):
                        audio = audio.cpu().float().numpy()

                # ── GPU is now free for other threads/chapters ────────────────
                # File I/O happens outside the lock to keep GPU utilization high
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

    def cleanup(self) -> None:
        import torch
        import gc
        if self._model is not None:
            del self._model
            self._model = None
            self._loaded_model_name = None
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
        
        # We explicitly supply torch_dtype to fix the Flash Attention warning
        self._model = Qwen3TTSModel.from_pretrained(
            self.config.tts_model_name,
            device_map=self.config.device,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
        self._loaded_model_name = self.config.tts_model_name

        # Suppress pad_token_id warning.
        gen_cfg = self._model.model.generation_config
        if gen_cfg.pad_token_id is None:
            gen_cfg.pad_token_id = gen_cfg.eos_token_id

        print(f"    [QwenTTS] {self.config.tts_model_name} ready.")

