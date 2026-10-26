"""
audiobook_factory/tts_providers/ming_provider.py
==================================================
Ming-3.0-TTS 0.5B (inclusionAI): zero-shot voice cloning in Chinese,
English, Japanese, Korean, German, French and Spanish. MIT licence.

Upstream: https://github.com/inclusionAI/Ming-3.0-tts. Not a pip package:
clone it and set ``ABM_MING_REPO`` (see ``requirements/tts-ming.txt``). The
"tiny" MoE variant (18 GB) does not fit a 16 GB GPU and is not offered.

Upstream's ``MingTTS`` generates one text at a time ("assumes a batch size
of 1"), so a batch is a loop. It loads the weights in bfloat16.
"""
from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Any

from audiobook_factory.tts_providers.base_tts_provider import (
    BaseTTSProvider,
    ProviderInfo,
    add_clone_to_path,
)

if TYPE_CHECKING:
    from audiobook_factory.pipeline import AudiobookConfig

logger = logging.getLogger(__name__)

_MODEL_ID: str = "inclusionAI/Ming-3.0-tts-0.5B"
_REPO_ENV_VAR: str = "ABM_MING_REPO"
_SAMPLE_RATE: int = 44100
_SPEAKER_RATE: int = 16000
# config.language (lower case) -> upstream language id.
_LANGUAGE_IDS: dict[str, str] = {
    "english": "en", "chinese": "zh", "japanese": "ja", "korean": "ko",
    "german": "de", "french": "fr", "spanish": "es",
}
# Added in Ming-3.0; upstream's MULTILINGUAL task needs the language id.
_MULTILINGUAL: frozenset[str] = frozenset({"ja", "ko", "de", "fr", "es"})


class MingTTSProvider(BaseTTSProvider):
    """Ming-3.0-TTS 0.5B bound to one device."""

    INFO = ProviderInfo(
        name="ming",
        display_name="Ming-3.0-TTS 0.5B",
        description="Small (1.5B parameters) seven-language voice cloning from a clip and its transcript. MIT licence.",
        license="MIT",
        commercial_use=True,
        homepage="https://github.com/inclusionAI/Ming-3.0-tts",
        default_model=_MODEL_ID,
        models=(_MODEL_ID,),
        native_sample_rate=_SAMPLE_RATE,
        min_vram_gb=5.0,
        languages=("Chinese", "English", "Japanese", "Korean", "German", "French", "Spanish"),
        supports_voice_clone=True,
        transcript="required",
        supports_batch=False,
        supports_seed=True,
        pip_requirements=("-r requirements/tts-ming.txt",),
        install_notes=(
            "Clone https://github.com/inclusionAI/Ming-3.0-tts and set ABM_MING_REPO to it. "
            "Weights (3 GB) download on first use."
        ),
    )

    def __init__(
        self,
        config: "AudiobookConfig",
        device: str | None = None,
        dtype_override: str | None = None,
    ) -> None:
        super().__init__(config)
        self._device = device or getattr(config, "device", "cuda")
        self._model: Any = None
        self._lock = threading.RLock()

    @property
    def device(self) -> str:
        return self._device

    def _ensure_initialised(self) -> None:
        with self._lock:
            if self._model is not None:
                return
            add_clone_to_path(_REPO_ENV_VAR, "ming_tts.py", "Ming-3.0-TTS")
            try:
                from ming_tts import MingTTS  # type: ignore
            except ImportError as exc:
                raise RuntimeError(
                    f"Ming-3.0-TTS could not import its code ({exc}). "
                    "Install requirements/tts-ming.txt in this environment."
                ) from exc
            self.bind_device()
            logger.info("[Ming] Loading %s on %s...", _MODEL_ID, self._device)
            # Our pipeline already rewrites numbers and symbols; upstream's
            # normalizer also needs pynini.
            model = MingTTS(_MODEL_ID, device=self._device, text_normalization=False)
            model._load_prompt = lambda paths, _m=model: _load_prompt(_m, paths)
            self._model = model
            logger.info("[Ming] Ready on %s.", self._device)

    def cleanup(self) -> None:
        with self._lock:
            self._model = None
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as exc:
            logger.debug("Could not empty the CUDA cache: %s", exc)

    def _task(self) -> tuple[str, str | None]:
        """Upstream task name and language id for ``config.language``."""
        language = _LANGUAGE_IDS.get(str(getattr(self.config, "language", "") or "").strip().lower())
        return ("MULTILINGUAL" if language in _MULTILINGUAL else "TTS"), language

    def synthesize(
        self,
        text: str,
        voice_ref: str | bytes,
        out_path: str | None = None,
        *,
        return_bytes: bool = False,
    ) -> tuple[str | bytes, float]:
        self.ensure_ready()
        voice_path = self.resolve_voice_path(voice_ref)
        if not voice_path:
            raise RuntimeError("Ming-3.0-TTS clones a voice: a reference clip is required.")
        transcript = self.reference_transcript(voice_path)
        if not transcript:
            # Without it upstream silently drops the acoustic prompt and keeps
            # only the speaker embedding: a much weaker clone.
            raise RuntimeError("Ming-3.0-TTS needs the transcript of the reference clip (voice transcript setting).")
        task, language = self._task()
        with self._lock:
            self.bind_device()
            self.seed_everything()
            waveform = self._model.synthesize(
                task=task, text=text, language_id=language,
                prompt_wav_path=voice_path, prompt_text=transcript, use_spk_emb=True,
            )
        return self.finish(waveform, int(self._model.sample_rate), out_path=out_path,
                           return_bytes=return_bytes or not out_path)


def _load_prompt(model: Any, paths: list[str]) -> tuple[Any, list[Any]]:
    """Upstream ``MingTTS._load_prompt`` reading with soundfile.

    Upstream calls ``torchaudio.load(path, backend="soundfile")``; torchaudio
    2.9 and later route ``load`` through torchcodec instead.
    """
    import soundfile as sf
    import torch
    import torchaudio.functional as F

    waveforms, embeddings = [], []
    for path in paths:
        data, rate = sf.read(path, dtype="float32", always_2d=True)
        if data.size == 0:
            raise ValueError(f"prompt audio is empty: {path}")
        waveform = torch.from_numpy(data.mean(axis=1)).unsqueeze(0)
        speaker = waveform if rate == _SPEAKER_RATE else F.resample(waveform, rate, _SPEAKER_RATE)
        embeddings.append(model.spkemb_extractor(speaker))
        if rate != model.sample_rate:
            waveform = F.resample(waveform, rate, model.sample_rate)
        waveforms.append(waveform)
    return model._pad_prompt_waveform(torch.cat(waveforms, dim=-1)), embeddings
