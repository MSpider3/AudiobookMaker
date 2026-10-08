"""
audiobook_factory/tts_providers/omnivoice_provider.py
======================================================
OmniVoice (k2-fsa / Xiaomi) zero-shot TTS provider.

OmniVoice is a non-autoregressive "diffusion language model" TTS: a Qwen3-0.6B
backbone iteratively unmasks 8 codebooks of Higgs-Audio-v2 tokens (25 Hz,
24 kHz audio). Because nothing is generated token by token, a whole batch of
chunks costs one set of ``num_step`` forward passes, so ``synthesize_batch``
hands the whole batch to upstream in a single call.

Written against ``omnivoice`` 0.2.1 (PyPI) / GitHub commit ``08be0b4``. The
upstream calls used here are:

* ``OmniVoice.from_pretrained(model_id, device_map=<device>, dtype=<dtype>)``
* ``OmniVoice.generate(text=[...], language=..., voice_clone_prompt=...,
  instruct=..., speed=..., duration=..., normalize_text=..., **generation)``
  which returns one 1-D float32 ``numpy`` array per text at
  ``model.sampling_rate``
* ``OmniVoice.create_voice_clone_prompt(ref_audio=path, ref_text=...,
  preprocess_prompt=...)`` which returns a reusable ``VoiceClonePrompt``
* ``OmniVoice.load_asr_model(model_name=..., device=...)`` (Whisper)
* ``VoiceClonePrompt.save(path)`` and the ``VoiceClonePrompt(...)`` constructor

Saved prompts (``config.voice_preset`` files and the voice cache) are read
back with ``torch.load(weights_only=True)`` and rebuilt field by field here
rather than through ``VoiceClonePrompt.load``. Upstream 0.2.1 loads the same
way, but a preset is a file the user picks, so the provider guarantees it is
never unpickled whatever upstream version is installed, and it checks the
tokens fit the loaded model before they reach ``generate``.

Voice Studio entry points: ``design_voice`` returns the designed narrator clip
for auditioning, ``save_voice_preset`` writes the conditioned voice with
``VoiceClonePrompt.save`` and ``load_voice_preset`` validates such a file.

How the narrator voice is resolved (first match wins)
-----------------------------------------------------
1. ``config.voice_preset`` — a ``.pt`` file written by ``VoiceClonePrompt.save``
   is loaded as is; any other file is treated as a reference clip.
2. A reference clip (the ``voice_ref`` argument, else ``config.voice_file``).
   With a transcript (``config.voice_transcript`` or a sidecar ``.txt``) the
   clip is encoded directly. Without one, Whisper transcribes it once, the
   resulting prompt is written to the voice cache on disk and the Whisper
   model is dropped again, so no instance transcribes twice and no ASR model
   stays resident during synthesis.
3. No clip at all — *voice design*. OmniVoice has no "design a voice, then
   reuse it" call: every ``generate(instruct=...)`` invents a new speaker. So
   the provider generates one short utterance, turns that audio into a
   ``VoiceClonePrompt`` and narrates the whole book by cloning it. The prompt
   is cached on disk under a key of (instruct, language, voice seed, model id),
   guarded by a file lock, so every GPU instance speaks with the same voice
   and a resumed book keeps its narrator.

   The key uses the ``voice_seed`` option, not ``config.seed``: the pipeline
   re-synthesizes a rejected chunk with ``config.seed + attempt``, and a key
   on ``config.seed`` would hand that one chunk a different narrator.
   ``config.seed`` still seeds the first design when ``voice_seed`` is unset.

Not mapped, because OmniVoice has no equivalent: ``top_p``, ``top_k``,
``repetition_penalty``, ``tts_timbre`` (no preset speakers) and
``quantization``. ``temperature`` is not mapped either: the model's sampling
temperatures (``class_temperature``, ``position_temperature``) work on a
different scale from an autoregressive model's, so they are separate options
that default to upstream's values.
"""
from __future__ import annotations

import contextlib
import gc
import hashlib
import json
import logging
import math
import os
import re
import tempfile
import threading
import time
from collections import OrderedDict
from typing import TYPE_CHECKING, Any, Callable, Iterator

from audiobook_factory.tts_providers.base_tts_provider import (
    BaseTTSProvider,
    ProviderInfo,
    ProviderOption,
)

if TYPE_CHECKING:
    from audiobook_factory.pipeline import AudiobookConfig

logger = logging.getLogger(__name__)

_DEFAULT_MODEL: str = "k2-fsa/OmniVoice"
# Research checkpoint from the paper (Chinese + English Emilia only). Its model
# card requires denoise=False and no language id.
_EMILIA_MODEL: str = "k2-fsa/OmniVoice-Emilia"
_INSTALL_COMMAND: str = 'pip install "omnivoice>=0.2.1"'
_NATIVE_SAMPLE_RATE: int = 24000
_MIN_VRAM_GB: float = 4.0

_DTYPE_NAMES: tuple[str, ...] = ("float16", "bfloat16", "float32")

_DEFAULT_ASR_MODEL: str = "openai/whisper-large-v3-turbo"
_ASR_MODELS: tuple[str, ...] = (
    "openai/whisper-large-v3-turbo",
    "openai/whisper-large-v3",
    "openai/whisper-medium",
    "openai/whisper-small",
    "openai/whisper-base",
    "openai/whisper-tiny",
)

# Forwarded verbatim to OmniVoice.generate(); upstream routes them into its
# OmniVoiceGenerationConfig.
_GENERATION_OPTION_KEYS: tuple[str, ...] = (
    "num_step",
    "guidance_scale",
    "t_shift",
    "denoise",
    "class_temperature",
    "position_temperature",
    "layer_penalty_factor",
    "preprocess_prompt",
    "postprocess_output",
    "pad_duration",
    "fade_duration",
    "audio_chunk_duration",
    "audio_chunk_threshold",
)

_VOICE_CACHE_DIR_NAME: str = ".omnivoice_voices"
_VOICE_CACHE_FALLBACK_DIR_NAME: str = "abm_omnivoice_voices"
_PROMPT_FILE_SUFFIXES: tuple[str, ...] = (".pt", ".pth")
_PROMPT_CACHE_MAX: int = 4
# Layout written by upstream's VoiceClonePrompt.save():
# {"format_version": 1, "ref_audio_tokens": LongTensor(C, T), "ref_text": str, "ref_rms": float}
_PROMPT_FORMAT_VERSION: int = 1
_FILE_LOCK_TIMEOUT_S: float = 900.0
_MIN_VOICE_BYTES: int = 100

# Budget for the utterance spoken while designing a voice. One unit is roughly
# one Latin letter; upstream's duration heuristic works out to about 14 units
# per second, so this aims for a 3-10 s reference, the range upstream recommends.
_DESIGN_TEXT_MAX_WEIGHT: float = 130.0
_DESIGN_TEXT_MIN_WEIGHT: float = 45.0
_DESIGN_MIN_SECONDS: float = 1.0
_WIDE_SCRIPT_START: int = 0x2E80
_WIDE_CHAR_WEIGHT: float = 3.0
_SPACE_WEIGHT: float = 0.2

# A transcript is checked against the clip it describes, in the same units per
# second. Speech runs at roughly 8-25; far outside that the text does not match
# the audio (a Whisper hallucination, or a transcript of some other clip), and
# since upstream derives every chunk's duration from this ratio the whole book
# would come out rushed or dragging.
_FRAME_RATE_HZ: float = 25.0
_MIN_TRANSCRIPT_RATE: float = 2.0
_MAX_TRANSCRIPT_RATE: float = 40.0

_DESIGN_TEXTS: dict[str, str] = {
    "en": (
        "The old lighthouse keeper climbed the winding stairs every evening, "
        "carrying a small lamp and a book of stories to read by the sea."
    ),
    "zh": "每天傍晚，老灯塔看守人都会提着一盏小灯，沿着盘旋的楼梯慢慢走上去，在海边读一个故事。",
}

# Names this application may hand over that upstream's language table does not
# know. Everything else is passed through: upstream accepts its own language
# names ("English") and ids ("en"), and falls back to language-agnostic mode
# with a warning for anything it cannot resolve.
_LANGUAGE_ALIASES: dict[str, str | None] = {
    "": None,
    "auto": None,
    "none": None,
    "automatic": None,
    "arabic": "arb",
    "mandarin": "zh",
    "mandarin chinese": "zh",
    "simplified chinese": "zh",
    "traditional chinese": "zh",
    "tagalog": "fil",
}

# A representative subset of the 646 supported languages; every name resolves
# through upstream's table or _LANGUAGE_ALIASES.
_LANGUAGES: tuple[str, ...] = (
    "English", "Chinese", "Cantonese", "Japanese", "Korean", "German", "French",
    "Spanish", "Portuguese", "Italian", "Russian", "Dutch", "Polish", "Turkish",
    "Arabic", "Hindi", "Bengali", "Tamil", "Telugu", "Marathi", "Gujarati",
    "Kannada", "Malayalam", "Urdu", "Persian", "Hebrew", "Indonesian", "Malay",
    "Vietnamese", "Thai", "Filipino", "Swahili", "Ukrainian", "Czech", "Slovak",
    "Romanian", "Hungarian", "Bulgarian", "Croatian", "Serbian", "Greek",
    "Catalan", "Swedish", "Danish", "Finnish", "Norwegian",
)

_VOICE_DESIGN_HELP: str = (
    "Voice design takes comma-separated attribute tags, not free-form prose: "
    "gender (male, female); age (child, teenager, young adult, middle-aged, "
    "elderly); pitch (very low pitch, low pitch, moderate pitch, high pitch, "
    "very high pitch); style (whisper); English accent (american, british, "
    "australian, canadian, indian, chinese, korean, japanese, portuguese or "
    "russian accent); or a Chinese dialect tag. At most one tag per category, "
    "for example 'female, middle-aged, low pitch, british accent'."
)

_SENTENCE_END_CHARS: str = ".!?。！？…"
_SENTENCE_RE = re.compile(r".+?(?:[.!?。！？…]+[\"'”’）)\]]*|$)\s*")
_INSTRUCT_SPLIT_RE = re.compile(r"\s*[,，]\s*")
_REGIONAL_CODE_RE = re.compile(r"^([a-z]{2,3})[-_][a-z0-9]{2,4}$")

# Serialises building a shared voice (design / auto-transcription) between the
# provider instances of this process; a file lock does the same across processes.
_VOICE_CACHE_LOCK = threading.Lock()
# Upstream's FlashInfer path keeps its attention plan in a module-level
# dictionary, so two models must never decode at the same time when it is on.
_FLASHINFER_LOCK = threading.Lock()


def _stable_hash(*parts: Any) -> str:
    """Returns a short, stable hex digest of *parts*."""
    payload = json.dumps(parts, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:24]


def _canonical_instruct(instruct: str | None) -> str:
    """Normalises an instruct string so equivalent spellings share a cache key."""
    if not instruct:
        return ""
    items = [item.strip().lower() for item in _INSTRUCT_SPLIT_RE.split(instruct)]
    return ", ".join(item for item in items if item)


def _text_weight(text: str) -> float:
    """Rough speaking length of *text*, in Latin-letter units."""
    total = 0.0
    for char in text:
        if char.isspace():
            total += _SPACE_WEIGHT
        elif ord(char) >= _WIDE_SCRIPT_START:
            total += _WIDE_CHAR_WEIGHT
        else:
            total += 1.0
    return total


def _leading_excerpt(texts: list[str], max_weight: float = _DESIGN_TEXT_MAX_WEIGHT) -> str:
    """Returns the opening sentences of *texts* that fit in *max_weight*.

    Whole sentences are kept where possible; a first sentence that is too
    long on its own is cut at a word boundary.
    """
    joined = re.sub(r"\s+", " ", " ".join(t for t in texts if t)).strip()
    if not joined:
        return ""
    excerpt = ""
    for sentence in _SENTENCE_RE.findall(joined):
        if _text_weight(excerpt + sentence) > max_weight:
            break
        excerpt += sentence
    if excerpt.strip():
        return excerpt.strip()

    weight = 0.0
    end = 0
    for index, char in enumerate(joined):
        weight += _text_weight(char)
        if weight > max_weight:
            break
        end = index + 1
    cut = joined[:end]
    if end < len(joined) and " " in cut:
        cut = cut[: cut.rindex(" ")]
    return cut.strip()


class OmniVoiceProvider(BaseTTSProvider):
    """OmniVoice zero-shot TTS: voice cloning, voice design, 646 languages.

    One instance owns one model on one device. All model access is serialised
    by ``self._lock``; settings are read from ``self.config`` on every call.

    Parameters
    ----------
    config : AudiobookConfig
        Run settings. Re-read on every call, so the pool may replace it.
    device : str | None
        Torch device this instance is bound to (``"cuda:0"``, ``"cpu"``).
        Defaults to ``config.device``.
    dtype_override : str | None
        ``"float16"``, ``"bfloat16"`` or ``"float32"``. Defaults to float16 on
        CUDA (what upstream itself loads, and the fast path on a T4, which has
        no native bfloat16) and float32 elsewhere.
    """

    INFO = ProviderInfo(
        name="omnivoice",
        display_name="OmniVoice",
        description=(
            "Fast non-autoregressive zero-shot TTS (0.6B parameters, 24 kHz) that "
            "clones the narrator from a 3-10 s clip, or designs one from attribute "
            "tags in the style prompt such as 'female, low pitch, british accent'. "
            "646 languages are supported; the language list here is a representative "
            "subset, and the 'Language code' option selects any other. Inline tags "
            "such as [laughter] and [sigh] are spoken as sounds. The weights are "
            "CC-BY-NC: non-commercial use only."
        ),
        license="CC-BY-NC",
        commercial_use=False,
        homepage="https://github.com/k2-fsa/OmniVoice",
        default_model=_DEFAULT_MODEL,
        models=(_DEFAULT_MODEL, _EMILIA_MODEL),
        native_sample_rate=_NATIVE_SAMPLE_RATE,
        min_vram_gb=_MIN_VRAM_GB,
        languages=_LANGUAGES,
        supports_voice_clone=True,
        transcript="optional",
        supports_instruct=True,
        supports_batch=True,
        supports_speed=True,
        supports_seed=True,
        supports_voice_preset=True,
        preset_voices=(),
        options=(
            ProviderOption(
                key="num_step", label="Decoding steps", kind="int", default=32,
                minimum=4, maximum=64, step=1,
                help="Iterative unmasking steps per batch. 32 is the reference quality; 16 is about twice as fast.",
            ),
            ProviderOption(
                key="guidance_scale", label="Guidance scale (CFG)", kind="float", default=2.0,
                minimum=0.0, maximum=5.0, step=0.1,
                help="Classifier-free guidance strength; 0 disables guidance.",
            ),
            ProviderOption(
                key="t_shift", label="Time-step shift", kind="float", default=0.1,
                minimum=0.01, maximum=1.0, step=0.01,
                help="Noise-schedule shift; smaller values spend more steps on the early, low-SNR part of decoding.",
            ),
            ProviderOption(
                key="denoise", label="Denoise reference", kind="bool", default=True,
                help="Adds the denoise token so a noisy reference clip still yields clean speech.",
            ),
            ProviderOption(
                key="class_temperature", label="Token temperature", kind="float", default=0.0,
                minimum=0.0, maximum=2.0, step=0.05,
                help="Sampling temperature for audio tokens; 0 is greedy and the most stable for narration.",
            ),
            ProviderOption(
                key="position_temperature", label="Position temperature", kind="float", default=5.0,
                minimum=0.0, maximum=10.0, step=0.5,
                help="Randomness in the order tokens are unmasked; 0 always unmasks the most confident positions first.",
            ),
            ProviderOption(
                key="layer_penalty_factor", label="Layer penalty", kind="float", default=5.0,
                minimum=0.0, maximum=10.0, step=0.5,
                help="Makes the coarse codebook layers unmask before the fine ones.",
            ),
            ProviderOption(
                key="preprocess_prompt", label="Clean reference clip", kind="bool", default=True,
                help="Removes long silences from the reference clip and ends its transcript with punctuation.",
            ),
            ProviderOption(
                key="postprocess_output", label="Trim long silences", kind="bool", default=True,
                help="Removes silences longer than half a second from each generated chunk.",
            ),
            ProviderOption(
                key="pad_duration", label="Edge padding (s)", kind="float", default=0.1,
                minimum=0.0, maximum=1.0, step=0.05,
                help="Silence added to both ends of every chunk.",
            ),
            ProviderOption(
                key="fade_duration", label="Edge fade (s)", kind="float", default=0.1,
                minimum=0.0, maximum=1.0, step=0.05,
                help="Fade-in and fade-out applied to every chunk to prevent clicks.",
            ),
            ProviderOption(
                key="audio_chunk_duration", label="Long-form piece length (s)", kind="float", default=15.0,
                minimum=5.0, maximum=30.0, step=1.0,
                help="When a chunk is longer than the threshold below, the model splits it into pieces of about this length.",
            ),
            ProviderOption(
                key="audio_chunk_threshold", label="Long-form threshold (s)", kind="float", default=30.0,
                minimum=10.0, maximum=120.0, step=1.0,
                help="Estimated chunk duration above which the model splits the text internally.",
            ),
            ProviderOption(
                key="duration", label="Fixed chunk duration (s)", kind="float", default=0.0,
                minimum=0.0, maximum=120.0, step=0.5,
                help="Forces every chunk to this many seconds and overrides speed. Leave at 0 for narration.",
            ),
            ProviderOption(
                key="normalize_text", label="Spell out numbers", kind="bool", default=False,
                help="Lets OmniVoice itself expand numbers, dates and currency (needs 'omnivoice[tn]'); "
                     "rarely needed, the app already rewrites them.",
            ),
            ProviderOption(
                key="language_id", label="Language code", kind="str", default="",
                help="OmniVoice language id such as 'en', 'yue' or 'arb'; overrides the book language. Use 'auto' for none.",
            ),
            ProviderOption(
                key="asr_model", label="Transcription model", kind="choice",
                default=_DEFAULT_ASR_MODEL, choices=_ASR_MODELS,
                help="Whisper model that transcribes the reference clip when no transcript is given.",
            ),
            ProviderOption(
                key="asr_device", label="Transcription device", kind="choice",
                default="gpu", choices=("gpu", "cpu"),
                help="Where Whisper runs for that one transcription; it is unloaded straight afterwards.",
            ),
            ProviderOption(
                key="voice_seed", label="Designed-voice seed", kind="int", default=-1,
                minimum=-1, maximum=2147483647, step=1,
                help="Picks which designed voice narrates. Change it for a different narrator; "
                     "the run seed alone never changes a voice that already exists.",
            ),
            ProviderOption(
                key="design_text", label="Voice design sentence", kind="str", default="",
                help="Sentence spoken once to create a designed voice; empty uses a built-in sentence or the book's opening.",
            ),
            ProviderOption(
                key="voice_cache_dir", label="Voice cache folder", kind="str", default="",
                help="Where designed and auto-transcribed voices are kept; empty uses '.omnivoice_voices' in the output folder.",
            ),
            ProviderOption(
                key="flashinfer", label="FlashInfer acceleration", kind="bool", default=False,
                help="Experimental 2x decoding speed-up; needs OmniVoice from GitHub plus flashinfer-python, and runs one GPU at a time.",
            ),
        ),
        pip_requirements=("omnivoice>=0.2.1",),
        install_notes=(
            "omnivoice requires transformers>=5.3.0, which conflicts with qwen-tts "
            "(transformers 4.57.x): install it in its own environment. Python 3.13+ "
            "also needs 'audioop-lts' for pydub. The model (about 3.3 GB) is not gated "
            "and needs no token. Without a reference transcript Whisper "
            "(openai/whisper-large-v3-turbo, about 1.6 GB) is downloaded and run once. "
            + _VOICE_DESIGN_HELP
            + " The weights are CC-BY-NC and bundle the Higgs Audio v2 tokenizer under "
            "the Boson Higgs Audio 2 Community License."
        ),
    )

    def __init__(
        self,
        config: "AudiobookConfig",
        device: str | None = None,
        dtype_override: str | None = None,
    ) -> None:
        super().__init__(config)
        self._device: str = device or getattr(config, "device", None) or "cuda"
        self._dtype_override: str | None = dtype_override
        self._model: Any = None
        self._loaded_key: tuple[str, str, bool] | None = None
        self._lock = threading.RLock()
        self._prompt_cache: OrderedDict[tuple, Any] = OrderedDict()
        self._digest_memo: dict[str, tuple[int, int, str]] = {}
        self._batch_limit: int = 0
        self._warned: set[str] = set()

    @property
    def device(self) -> str:
        """The torch device string this instance is bound to."""
        return self._device

    # ── Loading ──────────────────────────────────────────────────────────────

    def ensure_ready(self) -> None:
        """Loads the model if needed. Safe to call from any thread."""
        self._ensure_initialised()

    def _dtype_name(self) -> str:
        """Name of the torch dtype the model is loaded in."""
        override = (self._dtype_override or "").strip().lower()
        if override in _DTYPE_NAMES:
            return override
        if override:
            self._warn_once(
                f"dtype:{override}",
                "[OmniVoice] Unknown dtype_override '%s'; using the default.", override,
            )
        return "float16" if self._device.startswith("cuda") else "float32"

    def _flashinfer_enabled(self) -> bool:
        """True when the FlashInfer option is on and this device can use it."""
        if not self.option("flashinfer", False):
            return False
        if not self._device.startswith("cuda"):
            self._warn_once(
                "flashinfer-device",
                "[OmniVoice] FlashInfer needs a CUDA device; ignoring it on %s.", self._device,
            )
            return False
        return True

    def _ensure_initialised(self) -> None:
        """Loads the model, or reloads it when the configured model changed.

        Raises
        ------
        RuntimeError
            If ``omnivoice`` is missing or the model cannot be loaded.
        """
        with self._lock:
            wanted = (self.resolve_model_id(), self._dtype_name(), self._flashinfer_enabled())
            if self._model is not None and self._loaded_key == wanted:
                return
            if self._model is not None:
                logger.info(
                    "[OmniVoice] Settings changed (%s -> %s); reloading on %s.",
                    self._loaded_key, wanted, self._device,
                )
                self._release()

            model_id, dtype_name, use_flashinfer = wanted
            try:
                import torch
                from omnivoice import OmniVoice, VoiceClonePrompt
            except ImportError as exc:
                # ModuleNotFoundError: the package itself is absent. Any other
                # ImportError comes from inside it, typically an old transformers.
                missing = isinstance(exc, ModuleNotFoundError) and exc.name in ("omnivoice", "torch")
                reason = (
                    "is not installed" if missing
                    else f"could not be imported ({exc}); it needs transformers>=5.3.0"
                )
                message = f"OmniVoice {reason}. Install it with: {_INSTALL_COMMAND}"
                logger.error("[OmniVoice] %s", message)
                raise RuntimeError(message) from exc
            if not hasattr(VoiceClonePrompt, "save"):
                raise RuntimeError(
                    "The installed OmniVoice is too old (it cannot save voice prompts). "
                    f"Upgrade it with: {_INSTALL_COMMAND}"
                )

            if getattr(self.config, "quantization", "none") not in ("", "none", None):
                self._warn_once(
                    "quantization",
                    "[OmniVoice] quantization='%s' is not supported by OmniVoice and is ignored.",
                    getattr(self.config, "quantization", ""),
                )

            self.bind_device()
            logger.info("[OmniVoice] Loading %s on %s (%s)...", model_id, self._device, dtype_name)
            started = time.monotonic()
            try:
                # A device string pins every weight to this one device.
                model = OmniVoice.from_pretrained(
                    model_id,
                    device_map=self._device,
                    dtype=getattr(torch, dtype_name),
                )
            except Exception as exc:
                self._free_cuda_memory()
                raise RuntimeError(
                    f"OmniVoice could not load '{model_id}' on {self._device}: {exc}"
                ) from exc
            if use_flashinfer:
                try:
                    self._apply_flashinfer(model)
                except Exception:
                    # A half-patched model must not be kept.
                    del model
                    self._free_cuda_memory()
                    raise

            self._model = model
            self._loaded_key = wanted
            self._prompt_cache.clear()
            self._batch_limit = 0
            logger.info(
                "[OmniVoice] Ready on %s in %.1fs.", self._device, time.monotonic() - started,
            )

    @staticmethod
    def _apply_flashinfer(model: Any) -> None:
        """Patches *model* with upstream's FlashInfer decoding path."""
        try:
            from omnivoice.models.omnivoice_flashinfer import apply_flashinfer
        except ImportError as exc:
            raise RuntimeError(
                "OmniVoice FlashInfer acceleration is enabled but unavailable "
                f"({exc}). It needs OmniVoice installed from GitHub "
                "(pip install git+https://github.com/k2-fsa/OmniVoice.git) and "
                "flashinfer-python; otherwise turn the 'flashinfer' option off."
            ) from exc
        try:
            apply_flashinfer(model)
        except Exception as exc:
            raise RuntimeError(
                f"OmniVoice FlashInfer acceleration failed to initialise: {exc}. "
                "Turn the 'flashinfer' option off."
            ) from exc
        logger.info("[OmniVoice] FlashInfer acceleration enabled.")

    # ── Settings read at call time ───────────────────────────────────────────

    def _warn_once(self, key: str, message: str, *args: Any) -> None:
        """Logs a warning the first time *key* is seen on this instance."""
        if key not in self._warned:
            self._warned.add(key)
            logger.warning(message, *args)

    def _bounded(self, key: str) -> Any:
        """Reads a numeric option clamped to its declared range."""
        value = self.option(key)
        declared = next((o for o in self.info().options if o.key == key), None)
        if declared is None or isinstance(value, bool) or not isinstance(value, (int, float)):
            return value
        if declared.minimum is not None and value < declared.minimum:
            value = type(value)(declared.minimum)
        if declared.maximum is not None and value > declared.maximum:
            value = type(value)(declared.maximum)
        return value

    def _choice(self, key: str) -> str:
        """Reads a choice option, falling back to its default when invalid."""
        declared = next(o for o in self.info().options if o.key == key)
        value = str(self.option(key) or "").strip()
        if value in declared.choices:
            return value
        if value:
            self._warn_once(
                f"choice:{key}:{value}",
                "[OmniVoice] '%s' is not a valid value for '%s'; using '%s'.",
                value, key, declared.default,
            )
        return str(declared.default)

    def _language(self, model_id: str, override: str | None = None) -> str | None:
        """Language name or id handed to upstream; ``None`` is language-agnostic.

        *override* replaces the configured language (``language_id`` option,
        then ``config.language``).
        """
        if model_id == _EMILIA_MODEL:
            return None
        if override is not None:
            raw = str(override).strip()
        else:
            raw = str(self.option("language_id", "") or "").strip()
            if not raw:
                raw = str(getattr(self.config, "language", "") or "").strip()
        key = raw.lower()
        if key in _LANGUAGE_ALIASES:
            return _LANGUAGE_ALIASES[key]
        regional = _REGIONAL_CODE_RE.match(key)
        if regional:
            return regional.group(1)  # "zh-CN" -> "zh"; upstream has no regional ids
        # Upstream matches ids case-sensitively and names case-insensitively.
        return key if len(raw) <= 3 else raw

    def _instruct(self) -> str | None:
        """The voice-design attribute string, or ``None`` when empty."""
        return (getattr(self.config, "tts_instruct", "") or "").strip() or None

    def _generation_options(self, model_id: str) -> dict[str, Any]:
        """Decoding settings forwarded to ``OmniVoice.generate``."""
        options = {key: self._bounded(key) for key in _GENERATION_OPTION_KEYS}
        if model_id == _EMILIA_MODEL:
            # Trained without prompt denoising (see its model card).
            options["denoise"] = False
        return options

    def _generate_kwargs(self, texts: list[str], prompt: Any) -> dict[str, Any]:
        """Builds the keyword arguments of one ``OmniVoice.generate`` call."""
        model_id = self.resolve_model_id()
        kwargs: dict[str, Any] = {
            "text": list(texts),
            "language": self._language(model_id),
            "voice_clone_prompt": prompt,
            "instruct": self._instruct(),
        }
        try:
            speed = float(getattr(self.config, "speed", 1.0) or 1.0)
        except (TypeError, ValueError):
            speed = 1.0
        if speed > 0 and abs(speed - 1.0) > 1e-6:
            kwargs["speed"] = speed
        duration = float(self._bounded("duration") or 0.0)
        if duration > 0:
            kwargs["duration"] = duration
        if self.option("normalize_text", False):
            kwargs["normalize_text"] = True
        kwargs.update(self._generation_options(model_id))
        return kwargs

    # ── Synthesis ────────────────────────────────────────────────────────────

    def synthesize(
        self,
        text: str,
        voice_ref: str | bytes,
        out_path: str | None = None,
        *,
        return_bytes: bool = False,
    ) -> tuple[str | bytes, float]:
        """Synthesizes one text chunk.

        Parameters
        ----------
        text : str
            Text to speak.
        voice_ref : str | bytes
            Reference clip (path or audio bytes). Empty selects
            ``config.voice_file``, and with no clip at all the voice is designed
            from ``config.tts_instruct``.
        out_path : str | None
            Where to write the WAV when ``return_bytes`` is False.
        return_bytes : bool
            Return WAV bytes instead of writing ``out_path``.

        Returns
        -------
        tuple[str | bytes, float]
            ``(out_path or wav_bytes, duration_seconds)``.

        Raises
        ------
        RuntimeError
            If the chunk cannot be synthesized.
        """
        audios, sample_rate = self._run([text], voice_ref)
        return self.finish(audios[0], sample_rate, out_path, return_bytes)

    def synthesize_batch(
        self,
        texts: list[str],
        voice_ref: bytes,
        *,
        return_bytes: bool = True,
    ) -> list[tuple[bytes | str, float]]:
        """Synthesizes several chunks in one batched forward pass, in input order.

        Parameters
        ----------
        texts : list[str]
            Text chunks, each at most ``config.max_len`` characters.
        voice_ref : bytes
            Reference clip shared by every chunk; empty selects voice design.
        return_bytes : bool
            Kept for interface compatibility; WAV bytes are always returned.

        Returns
        -------
        list[tuple[bytes | str, float]]
            One ``(wav_bytes, duration_seconds)`` per input text.

        Raises
        ------
        RuntimeError
            If any chunk cannot be synthesized.
        """
        if not texts:
            return []
        audios, sample_rate = self._run(list(texts), voice_ref)
        return [self.finish(audio, sample_rate, None, True) for audio in audios]

    def _run(self, texts: list[str], voice_ref: str | bytes | None) -> tuple[list[Any], int]:
        """Generates one waveform per text and returns them with the sample rate."""
        for index, text in enumerate(texts):
            if not isinstance(text, str) or not text.strip():
                raise RuntimeError(
                    f"OmniVoice cannot synthesize an empty text (item {index} of {len(texts)})."
                )
        with self._lock:
            self._ensure_initialised()
            model = self._model
            self.bind_device()
            started = time.monotonic()
            prompt = self._resolve_prompt(model, voice_ref, texts)
            audios = self._generate_adaptive(model, texts, prompt)
            sample_rate = int(getattr(model, "sampling_rate", 0) or _NATIVE_SAMPLE_RATE)
            logger.debug(
                "[OmniVoice] %d chunk(s) on %s in %.2fs.",
                len(texts), self._device, time.monotonic() - started,
            )
        return audios, sample_rate

    @staticmethod
    def _is_oom(exc: BaseException) -> bool:
        """True when *exc* is a CUDA out-of-memory error."""
        try:
            import torch
        except ImportError:
            return False
        return isinstance(exc, torch.cuda.OutOfMemoryError)

    def _free_cuda_memory(self) -> None:
        """Collects garbage and returns cached CUDA blocks to the driver."""
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as exc:
            logger.debug("[OmniVoice] Could not empty the CUDA cache: %s", exc)

    def _generate_adaptive(self, model: Any, texts: list[str], prompt: Any) -> list[Any]:
        """Runs batched generation, falling back to one text at a time on OOM.

        The pipeline chooses the batch (``config.batch_size`` or free VRAM) and
        it is decoded in one upstream call. After an out-of-memory error the
        batch that failed is redone item by item and later calls are capped at
        half its size until the model is reloaded.
        """
        limit = self._batch_limit or len(texts)

        audios: list[Any] = []
        for start in range(0, len(texts), limit):
            group = texts[start:start + limit]
            out_of_memory = False
            try:
                audios.extend(self._generate(model, group, prompt))
            except Exception as exc:
                if len(group) == 1 or not self._is_oom(exc):
                    raise
                out_of_memory = True
            if out_of_memory:
                # Retried outside the except block so the failed batch's
                # tensors are no longer referenced by the traceback.
                self._free_cuda_memory()
                self._batch_limit = max(1, len(group) // 2)
                logger.warning(
                    "[OmniVoice] Out of memory on %s with a batch of %d; retrying "
                    "one chunk at a time and capping later batches at %d.",
                    self._device, len(group), self._batch_limit,
                )
                for text in group:
                    audios.extend(self._generate(model, [text], prompt))
        return audios

    def _seed_torch(self, seed: int) -> None:
        """Seeds torch's RNGs for this device; a negative seed leaves them random.

        Like ``seed_everything`` it reseeds only this instance's GPU, so the
        other GPU's worker is not disturbed mid-generation.
        """
        if seed < 0:
            return
        import torch
        torch.manual_seed(seed)
        if self._device.startswith("cuda") and torch.cuda.is_available():
            try:
                index = int(self._device.split(":")[1]) if ":" in self._device else torch.cuda.current_device()
                with torch.cuda.device(index):
                    torch.cuda.manual_seed(seed)
            except Exception as exc:
                logger.debug("[OmniVoice] Could not seed %s: %s", self._device, exc)

    @contextlib.contextmanager
    def _decoding_guard(self) -> Iterator[None]:
        """Serialises decoding across instances while FlashInfer is active."""
        if self._loaded_key is not None and self._loaded_key[2]:
            with _FLASHINFER_LOCK:
                yield
        else:
            yield

    def _generate(self, model: Any, texts: list[str], prompt: Any) -> list[Any]:
        """One upstream ``generate`` call for *texts*, all with the same voice."""
        self.seed_everything()
        return self._call_generate(model, self._generate_kwargs(texts, prompt), len(texts))

    def _call_generate(self, model: Any, kwargs: dict[str, Any], expected: int) -> list[Any]:
        """Calls ``model.generate`` and checks it returned *expected* waveforms."""
        try:
            with self._decoding_guard():
                audios = model.generate(**kwargs)
        except Exception as exc:
            if self._is_oom(exc):
                if expected == 1:
                    self._free_cuda_memory()
                    raise RuntimeError(
                        f"OmniVoice ran out of GPU memory on {self._device} for a single chunk."
                    ) from exc
                raise
            if isinstance(exc, ValueError) and kwargs.get("instruct"):
                raise RuntimeError(
                    f"OmniVoice rejected the style prompt {kwargs['instruct']!r}: {exc}\n"
                    + _VOICE_DESIGN_HELP
                ) from exc
            raise RuntimeError(f"OmniVoice synthesis failed on {self._device}: {exc}") from exc
        if audios is None or len(audios) != expected:
            count = 0 if audios is None else len(audios)
            raise RuntimeError(
                f"OmniVoice returned {count} waveform(s) for {expected} text(s) on {self._device}."
            )
        return list(audios)

    # ── Narrator voice ───────────────────────────────────────────────────────

    def _cache_get(self, key: tuple) -> Any:
        """Returns the cached prompt for *key*, or ``None``."""
        prompt = self._prompt_cache.get(key)
        if prompt is not None:
            self._prompt_cache.move_to_end(key)
        return prompt

    def _cache_put(self, key: tuple, prompt: Any) -> Any:
        """Stores *prompt* in the in-memory cache, evicting the oldest entry."""
        self._prompt_cache[key] = prompt
        self._prompt_cache.move_to_end(key)
        while len(self._prompt_cache) > _PROMPT_CACHE_MAX:
            self._prompt_cache.popitem(last=False)
        return prompt

    def _file_digest(self, path: str) -> str:
        """SHA-256 of a file's content, memoised on its size and mtime."""
        stat = os.stat(path)
        memo = self._digest_memo.get(path)
        if memo is not None and memo[0] == stat.st_size and memo[1] == stat.st_mtime_ns:
            return memo[2]
        sha = hashlib.sha256()
        with open(path, "rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                sha.update(block)
        digest = sha.hexdigest()
        if len(self._digest_memo) > 32:
            self._digest_memo.clear()
        self._digest_memo[path] = (stat.st_size, stat.st_mtime_ns, digest)
        return digest

    def _voice_cache_dir(self) -> str:
        """Directory holding designed and auto-transcribed voices."""
        configured = str(self.option("voice_cache_dir", "") or "").strip()
        primary = configured or os.path.join(
            getattr(self.config, "output_dir", "") or ".", _VOICE_CACHE_DIR_NAME,
        )
        try:
            os.makedirs(primary, exist_ok=True)
            return primary
        except OSError as exc:
            fallback = os.path.join(tempfile.gettempdir(), _VOICE_CACHE_FALLBACK_DIR_NAME)
            self._warn_once(
                f"cache-dir:{primary}",
                "[OmniVoice] Cannot use voice cache folder %s (%s); using %s.",
                primary, exc, fallback,
            )
            os.makedirs(fallback, exist_ok=True)
            return fallback

    @staticmethod
    def _file_lock(path: str) -> Any:
        """Cross-process lock for *path*; a no-op when ``filelock`` is missing."""
        try:
            from filelock import FileLock
        except ImportError:
            logger.debug("[OmniVoice] filelock is not installed; voice cache is locked per process only.")
            return contextlib.nullcontext()
        return FileLock(path, timeout=_FILE_LOCK_TIMEOUT_S)

    @staticmethod
    def _read_prompt_file(path: str) -> Any:
        """Reads a file written by ``VoiceClonePrompt.save`` without unpickling.

        The file is loaded with ``weights_only=True`` (tensors and plain values
        only), checked field by field and rebuilt into a ``VoiceClonePrompt``.

        Raises
        ------
        ValueError
            If the file is missing or is not a valid saved voice prompt.
        """
        import torch
        from omnivoice import VoiceClonePrompt

        if not os.path.isfile(path):
            raise ValueError(f"OmniVoice voice preset not found: {path}")
        try:
            data = torch.load(path, map_location="cpu", weights_only=True)
        except Exception as exc:
            raise ValueError(f"{path} is not an OmniVoice voice preset: {exc}") from exc
        if not isinstance(data, dict):
            raise ValueError(f"{path} is not an OmniVoice voice preset (unexpected content).")
        version = data.get("format_version")
        if version != _PROMPT_FORMAT_VERSION:
            raise ValueError(
                f"{path} uses voice preset format {version!r}; this provider reads "
                f"format {_PROMPT_FORMAT_VERSION}."
            )
        tokens = data.get("ref_audio_tokens")
        if (
            not isinstance(tokens, torch.Tensor) or tokens.dim() != 2 or tokens.numel() == 0
            or tokens.is_floating_point() or tokens.is_complex() or tokens.dtype == torch.bool
        ):
            raise ValueError(f"{path} holds no usable reference audio tokens.")
        ref_text = data.get("ref_text")
        if not isinstance(ref_text, str) or not ref_text.strip():
            raise ValueError(f"{path} holds no reference transcript.")
        try:
            ref_rms = float(data.get("ref_rms"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{path} holds no reference loudness.") from exc
        if not math.isfinite(ref_rms) or ref_rms <= 0:
            raise ValueError(f"{path} holds an invalid reference loudness ({ref_rms}).")
        return VoiceClonePrompt(ref_audio_tokens=tokens.long(), ref_text=ref_text, ref_rms=ref_rms)

    def _fit_prompt(self, model: Any, prompt: Any, path: str) -> Any:
        """Checks a loaded prompt against the model and moves it to its device.

        Raises
        ------
        ValueError
            If the tokens were made for a different codec layout.
        """
        tokens = prompt.ref_audio_tokens
        model_config = getattr(model, "config", None)
        codebooks = getattr(model_config, "num_audio_codebook", None)
        if codebooks is not None and int(tokens.shape[0]) != int(codebooks):
            raise ValueError(
                f"{path} has {int(tokens.shape[0])} audio codebooks but "
                f"{self.resolve_model_id()} uses {int(codebooks)}."
            )
        vocab = getattr(model_config, "audio_vocab_size", None)
        if vocab is not None and (int(tokens.min()) < 0 or int(tokens.max()) >= int(vocab)):
            raise ValueError(f"{path} contains audio tokens outside the model's vocabulary.")
        device = getattr(model, "device", None)
        if device is not None:
            try:
                prompt.ref_audio_tokens = tokens.to(device)
            except Exception as exc:  # generate() moves them per call anyway
                logger.debug("[OmniVoice] Could not move voice prompt to %s: %s", device, exc)
        return prompt

    def _load_prompt_file(self, model: Any, path: str) -> Any:
        """Loads a cached prompt; ``None`` when it is absent or unusable."""
        if not os.path.isfile(path):
            return None
        try:
            return self._fit_prompt(model, self._read_prompt_file(path), path)
        except ValueError as exc:
            logger.warning("[OmniVoice] Ignoring unusable voice file: %s", exc)
            return None

    @staticmethod
    def _save_prompt_file(prompt: Any, path: str) -> None:
        """Atomically writes *prompt* to *path*."""
        tmp_path = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"
        try:
            prompt.save(tmp_path)
            os.replace(tmp_path, path)
        finally:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass

    @staticmethod
    def _disk_key(kind: str, path: str) -> tuple:
        """In-memory cache key of a disk-backed prompt, tied to the file's state.

        Another instance re-rolling the voice, or the user deleting the file,
        changes the key, so a stale in-memory copy is never used.
        """
        try:
            stat = os.stat(path)
        except OSError:
            return (kind, path, 0, 0)
        return (kind, path, stat.st_size, stat.st_mtime_ns)

    def _disk_prompt(
        self,
        model: Any,
        kind: str,
        cache_path: str,
        build: Callable[[], Any],
        must_persist: bool,
        force: bool = False,
    ) -> Any:
        """Returns the voice stored at *cache_path*, building it once if absent.

        The check and the build run under a process-wide lock and a file lock,
        so of several instances asking for the same voice exactly one creates
        it and the others load the identical result. *force* rebuilds and
        overwrites it.
        """
        if not force:
            cached = self._cache_get(self._disk_key(kind, cache_path))
            if cached is not None:
                return cached
        with _VOICE_CACHE_LOCK, self._file_lock(cache_path + ".lock"):
            prompt = None if force else self._load_prompt_file(model, cache_path)
            if prompt is not None:
                logger.info("[OmniVoice] Reusing saved voice %s on %s.", cache_path, self._device)
            else:
                prompt = build()
                try:
                    self._save_prompt_file(prompt, cache_path)
                except Exception as exc:
                    if must_persist:
                        raise RuntimeError(
                            f"OmniVoice could not save the designed voice to {cache_path}: {exc}. "
                            "Without it every GPU would narrate in a different voice; set the "
                            "'voice_cache_dir' option to a writable folder."
                        ) from exc
                    logger.warning("[OmniVoice] Could not save voice file %s: %s", cache_path, exc)
            return self._cache_put(self._disk_key(kind, cache_path), prompt)

    def _create_prompt(self, model: Any, path: str, transcript: str | None, preprocess: bool) -> Any:
        """Encodes a reference clip into a reusable ``VoiceClonePrompt``."""
        import torch
        try:
            with torch.no_grad():
                return model.create_voice_clone_prompt(
                    ref_audio=path,
                    ref_text=transcript,
                    preprocess_prompt=preprocess,
                )
        except Exception as exc:
            raise RuntimeError(
                f"OmniVoice could not prepare the reference clip {path} on {self._device}: {exc}"
            ) from exc

    @staticmethod
    def _prompt_seconds(model: Any, prompt: Any) -> float:
        """Length of a prompt's reference audio in seconds; 0.0 when unknown."""
        tokens = getattr(prompt, "ref_audio_tokens", None)
        frames = int(tokens.shape[-1]) if hasattr(tokens, "shape") and len(tokens.shape) else 0
        tokenizer_config = getattr(getattr(model, "audio_tokenizer", None), "config", None)
        frame_rate = float(getattr(tokenizer_config, "frame_rate", 0) or _FRAME_RATE_HZ)
        return frames / frame_rate

    def _transcript_is_plausible(self, model: Any, prompt: Any) -> bool:
        """False when the transcript cannot be what the reference clip says."""
        seconds = self._prompt_seconds(model, prompt)
        if seconds <= 0:
            return True
        rate = _text_weight(getattr(prompt, "ref_text", "") or "") / seconds
        return _MIN_TRANSCRIPT_RATE <= rate <= _MAX_TRANSCRIPT_RATE

    def _resolve_prompt(self, model: Any, voice_ref: str | bytes | None, texts: list[str]) -> Any:
        """Returns the ``VoiceClonePrompt`` every chunk of this call is spoken with."""
        preset = (getattr(self.config, "voice_preset", "") or "").strip()
        if preset:
            if not os.path.isfile(preset):
                raise RuntimeError(f"OmniVoice voice preset not found: {preset}")
            if preset.lower().endswith(_PROMPT_FILE_SUFFIXES):
                try:
                    return self._preset_prompt(model, preset)
                except ValueError as exc:
                    raise RuntimeError(str(exc)) from exc
            voice_path: str | None = preset
        else:
            voice_path = self._voice_path(voice_ref)
        if voice_path:
            return self._clone_prompt(model, voice_path)
        return self._designed_prompt(model, texts)

    def _voice_path(self, voice_ref: str | bytes | None) -> str | None:
        """Path of the reference clip, or ``None`` when there is none."""
        if isinstance(voice_ref, (bytes, bytearray)) and 0 < len(voice_ref) < _MIN_VOICE_BYTES:
            raise RuntimeError(
                f"OmniVoice voice reference is only {len(voice_ref)} bytes; the audio is empty or corrupted."
            )
        path = self.resolve_voice_path(voice_ref if voice_ref else None)
        if path and not os.path.isfile(path):
            raise RuntimeError(f"OmniVoice voice reference not found: {path}")
        return path

    def _preset_prompt(self, model: Any, preset: str) -> Any:
        """Loads and validates a saved ``VoiceClonePrompt`` preset.

        Raises
        ------
        ValueError
            If the file is not a preset this model can use.
        """
        key = self._disk_key("preset", preset)
        cached = self._cache_get(key)
        if cached is not None:
            return cached
        prompt = self._fit_prompt(model, self._read_prompt_file(preset), preset)
        logger.info("[OmniVoice] Loaded voice preset %s on %s.", preset, self._device)
        return self._cache_put(key, prompt)

    def _transcript_for(self, path: str, digest: str) -> str:
        """Transcript of the clip at *path*, or ``""`` when none is known."""
        transcript = self.reference_transcript(path)
        if transcript:
            return transcript
        # The pipeline hands the clip over as bytes, which lose the sidecar
        # transcript next to the original file; use it when it is the same clip.
        voice_file = getattr(self.config, "voice_file", "") or ""
        if voice_file and voice_file != path and os.path.isfile(voice_file):
            try:
                if self._file_digest(voice_file) == digest:
                    return self.reference_transcript(voice_file)
            except OSError as exc:
                logger.debug("[OmniVoice] Could not read %s: %s", voice_file, exc)
        return ""

    def _clone_prompt(self, model: Any, path: str, transcript: str | None = None) -> Any:
        """Voice prompt for a reference clip, encoded once per clip and transcript.

        *transcript* overrides the configured transcript and auto-transcription.
        """
        digest = self._file_digest(path)
        transcript = (transcript or "").strip() or self._transcript_for(path, digest)
        preprocess = bool(self.option("preprocess_prompt", True))

        if transcript:
            key: tuple = ("clone", digest, transcript, preprocess)
            cached = self._cache_get(key)
            if cached is not None:
                return cached
            logger.info("[OmniVoice] Encoding reference clip on %s (transcript supplied).", self._device)
            prompt = self._create_prompt(model, path, transcript, preprocess)
            if not self._transcript_is_plausible(model, prompt):
                self._warn_once(
                    f"transcript:{digest}",
                    "[OmniVoice] The reference transcript does not seem to match the reference "
                    "clip (%d characters for about %.1f s of speech). Pacing is derived from "
                    "it, so check that it is the exact text spoken in the clip.",
                    len(transcript), self._prompt_seconds(model, prompt),
                )
            return self._cache_put(key, prompt)

        asr_model = self._choice("asr_model")
        cache_path = os.path.join(
            self._voice_cache_dir(),
            "clone_" + _stable_hash("clone-asr", digest, self.resolve_model_id(), asr_model, preprocess) + ".pt",
        )
        return self._disk_prompt(
            model, "clone-asr", cache_path,
            lambda: self._transcribe_and_create(model, path, asr_model, preprocess),
            must_persist=False,
        )

    def _transcribe_and_create(self, model: Any, path: str, asr_model: str, preprocess: bool) -> Any:
        """Transcribes the clip with Whisper once, then drops the ASR model."""
        asr_device = "cpu" if self._choice("asr_device") == "cpu" else self._device
        logger.info(
            "[OmniVoice] No reference transcript; transcribing once with %s on %s. "
            "Set the voice transcript to skip this.", asr_model, asr_device,
        )
        try:
            try:
                model.load_asr_model(model_name=asr_model, device=asr_device)
            except Exception as exc:
                raise RuntimeError(
                    f"OmniVoice could not load the transcription model {asr_model}: {exc}. "
                    "Provide the reference clip's transcript instead."
                ) from exc
            prompt = self._create_prompt(model, path, None, preprocess)
        finally:
            self._release_asr(model)
        transcript = (getattr(prompt, "ref_text", "") or "").strip()
        if not transcript:
            raise RuntimeError(
                "OmniVoice could not transcribe the reference clip (no speech recognised). "
                "Provide its transcript, or use a clip with clear speech."
            )
        if not self._transcript_is_plausible(model, prompt):
            raise RuntimeError(
                f"OmniVoice could not transcribe the reference clip reliably: {asr_model} "
                f"returned {len(transcript)} characters for about "
                f"{self._prompt_seconds(model, prompt):.1f} s of speech ({transcript[:80]!r}). "
                "Provide the clip's transcript, or choose a larger transcription model."
            )
        logger.info("[OmniVoice] Reference transcript: %s", transcript)
        return prompt

    def _release_asr(self, model: Any) -> None:
        """Drops the Whisper pipeline upstream keeps on the model after transcribing."""
        if getattr(model, "_asr_pipe", None) is not None:
            model._asr_pipe = None
            self._free_cuda_memory()

    @staticmethod
    def _design_text(language: str | None, texts: list[str], explicit: str) -> str:
        """Sentence spoken to create a designed voice; ``""`` when none is available."""
        if explicit:
            return explicit
        key = (language or "").lower()
        if key in ("en", "english"):
            return _DESIGN_TEXTS["en"]
        if key in ("zh", "chinese"):
            return _DESIGN_TEXTS["zh"]
        # Any other language: the book's own opening words are in the right language.
        excerpt = _leading_excerpt(texts)
        if not excerpt or _text_weight(excerpt) >= _DESIGN_TEXT_MIN_WEIGHT:
            return excerpt
        # The model is unreliable on 1-2 s utterances without a reference, so a
        # very short opening ("Chapter one") is repeated up to about 3 s.
        if excerpt.rstrip("\"'”’）)]")[-1:] not in tuple(_SENTENCE_END_CHARS):
            excerpt += "。" if ord(excerpt[-1]) >= _WIDE_SCRIPT_START else "."
        design_text = excerpt
        while _text_weight(design_text) < _DESIGN_TEXT_MIN_WEIGHT:
            design_text = f"{design_text} {excerpt}"
        return design_text

    def _design_spec(
        self,
        instruct: str | None = None,
        language: str | None = None,
        text: str | None = None,
    ) -> dict[str, Any]:
        """Identity of a designed voice and where it is cached.

        Arguments override ``config.tts_instruct``, the configured language
        and the ``design_text`` option. ``config.seed`` is deliberately not
        part of the identity: the pipeline changes it to re-take a rejected
        chunk, which must not change the narrator. The ``voice_seed`` option is.
        """
        model_id = self.resolve_model_id()
        if instruct is None:
            instruct = getattr(self.config, "tts_instruct", "") or ""
        instruct = instruct.strip() or None
        language = self._language(model_id, override=language)
        seed = max(-1, int(self._bounded("voice_seed")))
        explicit_text = (text if text is not None else str(self.option("design_text", "") or "")).strip()
        cache_path = os.path.join(
            self._voice_cache_dir(),
            "design_" + _stable_hash(
                "design", _canonical_instruct(instruct), (language or "").lower(),
                seed, model_id, explicit_text,
            ) + ".pt",
        )
        return {
            "model_id": model_id, "instruct": instruct, "language": language, "seed": seed,
            "text": explicit_text, "cache_path": cache_path,
        }

    def _designed_prompt(
        self,
        model: Any,
        texts: list[str],
        spec: dict[str, Any] | None = None,
        force: bool = False,
    ) -> Any:
        """Voice prompt for a book with no reference clip.

        The voice is designed once per (instruct, language, voice seed, model)
        and then cloned for every chunk, by every instance.
        """
        spec = spec or self._design_spec()

        def build() -> Any:
            design_text = self._design_text(spec["language"], texts, spec["text"])
            if not design_text:
                raise ValueError(
                    f"OmniVoice has no built-in voice design sentence for language "
                    f"{spec['language']!r}: pass a sentence in that language (the 'text' "
                    "argument or the 'design_text' option)."
                )
            return self._design_voice(
                model, spec["cache_path"], design_text, spec["instruct"], spec["language"], spec["seed"],
            )

        return self._disk_prompt(model, "design", spec["cache_path"], build, must_persist=True, force=force)

    def _design_voice(
        self,
        model: Any,
        cache_path: str,
        design_text: str,
        instruct: str | None,
        language: str | None,
        seed: int,
    ) -> Any:
        """Generates the designed voice's reference utterance and encodes it."""
        import numpy as np
        import soundfile as sf

        if seed < 0:
            try:
                seed = max(-1, int(getattr(self.config, "seed", -1)))
            except (TypeError, ValueError):
                seed = -1
        logger.info(
            "[OmniVoice] Designing the narrator voice on %s (style=%r, language=%r, seed=%d).",
            self._device, instruct or "model's choice", language, seed,
        )
        model_id = self.resolve_model_id()
        kwargs: dict[str, Any] = {
            "text": [design_text],
            "language": language,
            "voice_clone_prompt": None,
            "instruct": instruct,
        }
        kwargs.update(self._generation_options(model_id))
        self._seed_torch(seed)
        audio = self.to_mono_float32(self._call_generate(model, kwargs, 1)[0])
        sample_rate = int(getattr(model, "sampling_rate", 0) or _NATIVE_SAMPLE_RATE)
        if audio.size < int(_DESIGN_MIN_SECONDS * sample_rate) or not np.isfinite(audio).all():
            raise RuntimeError(
                f"OmniVoice voice design produced an unusable reference "
                f"({audio.size / float(sample_rate):.2f}s) on {self._device}; "
                "try another 'voice_seed' or a longer 'design_text'."
            )

        stem = cache_path[: -len(".pt")]
        wav_path = stem + ".wav"
        tmp_wav = f"{wav_path}.{os.getpid()}.tmp"
        sf.write(tmp_wav, audio, sample_rate, format="WAV")
        os.replace(tmp_wav, wav_path)
        prompt = self._create_prompt(
            model, wav_path, design_text, bool(self.option("preprocess_prompt", True)),
        )
        try:
            with open(stem + ".json", "w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "model": model_id, "instruct": instruct or "", "language": language or "",
                        "seed": seed, "text": design_text, "reference_audio": os.path.basename(wav_path),
                    },
                    fh, ensure_ascii=False, indent=2,
                )
        except OSError as exc:
            logger.debug("[OmniVoice] Could not write voice description: %s", exc)
        logger.info(
            "[OmniVoice] Designed voice saved to %s (listen to %s). It is reused for "
            "this style and language until you change the 'voice_seed' option or "
            "delete the file.", cache_path, wav_path,
        )
        return prompt

    # ── Voice Studio: audition, save and load a voice ────────────────────────

    def design_voice(
        self,
        instruct: str | None = None,
        text: str | None = None,
        language: str | None = None,
        *,
        force: bool = False,
    ) -> tuple[bytes, int, str]:
        """Creates (or fetches) the designed narrator clip for auditioning.

        The clip is the one a book run with the same style, language,
        ``voice_seed`` and design sentence clones, taken from the same on-disk
        voice cache, so what the user hears is what the audiobook gets.

        Parameters
        ----------
        instruct : str | None
            Voice attribute tags; defaults to ``config.tts_instruct``. Empty
            lets the model choose a voice.
        text : str | None
            Sentence to speak; defaults to the ``design_text`` option, then a
            built-in sentence (English and Chinese only).
        language : str | None
            Target language name or id; defaults to the configured language.
        force : bool
            Design the voice again and replace the cached one (a re-roll when
            the seed is random).

        Returns
        -------
        tuple[bytes, int, str]
            ``(wav_bytes, sample_rate, text_spoken)``. Pass the bytes and the
            text to :meth:`save_voice_preset` to keep the voice.

        Raises
        ------
        ValueError
            If no sentence is available for the language.
        RuntimeError
            If the voice cannot be designed.
        """
        import soundfile as sf

        with self._lock:
            self._ensure_initialised()
            model = self._model
            self.bind_device()
            spec = self._design_spec(instruct, language, text)
            prompt = self._designed_prompt(model, [], spec, force=force)
            stem = spec["cache_path"][: -len(".pt")]
            wav_path = stem + ".wav"
            if not os.path.isfile(wav_path):
                raise RuntimeError(
                    f"The audition clip of the designed voice is missing ({wav_path}); "
                    "call design_voice(force=True) to design the voice again."
                )
            with open(wav_path, "rb") as fh:
                wav_bytes = fh.read()
            sample_rate = int(sf.info(wav_path).samplerate)
            spoken = ""
            try:
                with open(stem + ".json", encoding="utf-8") as fh:
                    spoken = str(json.load(fh).get("text") or "")
            except (OSError, ValueError, AttributeError) as exc:
                logger.debug("[OmniVoice] No voice description next to %s: %s", wav_path, exc)
            spoken = spoken or str(getattr(prompt, "ref_text", "") or "")
        return wav_bytes, sample_rate, spoken

    def _describe_prompt(self, model: Any, prompt: Any, path: str, mode: str) -> dict[str, Any]:
        """JSON-safe description of a voice prompt."""
        tokens = prompt.ref_audio_tokens
        return {
            "path": path,
            "provider": self.info().name,
            "format": "omnivoice.VoiceClonePrompt",
            "format_version": _PROMPT_FORMAT_VERSION,
            "mode": mode,
            "model_id": self.resolve_model_id(),
            "ref_text": str(prompt.ref_text),
            "ref_seconds": round(self._prompt_seconds(model, prompt), 3),
            "ref_frames": int(tokens.shape[-1]),
            "num_codebooks": int(tokens.shape[0]),
            "ref_rms": float(prompt.ref_rms),
            "sample_rate": int(getattr(model, "sampling_rate", 0) or _NATIVE_SAMPLE_RATE),
        }

    def save_voice_preset(
        self,
        path: str,
        voice_ref: str | bytes | None = None,
        *,
        transcript: str | None = None,
    ) -> dict[str, Any]:
        """Saves the conditioned narrator voice as a reusable preset file.

        The file is written by upstream's ``VoiceClonePrompt.save``: the
        reference audio tokens, the reference transcript and its loudness.
        Set ``config.voice_preset`` to it and later runs need neither the
        clip nor a transcription.

        Parameters
        ----------
        path : str
            Destination file; use a ``.pt`` name so ``config.voice_preset``
            recognises it.
        voice_ref : str | bytes | None
            Reference clip (path or audio bytes); defaults to
            ``config.voice_file``. With no clip at all the designed voice
            (``config.tts_instruct``) is saved.
        transcript : str | None
            Transcript of the clip; defaults to ``config.voice_transcript``,
            a sidecar ``.txt``, then a one-off Whisper transcription.

        Returns
        -------
        dict[str, Any]
            ``path``, ``mode`` (``"clone"`` or ``"design"``), ``ref_text``,
            ``model_id`` and the prompt's size.

        Raises
        ------
        ValueError
            If *path* is empty, or a voice has to be designed and no sentence
            is available for the language.
        RuntimeError
            If the voice cannot be prepared or the file cannot be written.
        """
        if not path or not str(path).strip():
            raise ValueError("save_voice_preset needs a destination path.")
        path = str(path)
        with self._lock:
            self._ensure_initialised()
            model = self._model
            self.bind_device()
            voice_path = self._voice_path(voice_ref)
            if voice_path:
                mode = "clone"
                prompt = self._clone_prompt(model, voice_path, transcript=transcript)
            else:
                mode = "design"
                prompt = self._designed_prompt(model, [])
            try:
                directory = os.path.dirname(path)
                if directory:
                    os.makedirs(directory, exist_ok=True)
                self._save_prompt_file(prompt, path)
            except Exception as exc:
                raise RuntimeError(f"OmniVoice could not write the voice preset {path}: {exc}") from exc
            info = self._describe_prompt(model, prompt, path, mode)
        logger.info("[OmniVoice] Saved %s voice preset to %s.", mode, path)
        return info

    def load_voice_preset(self, path: str) -> dict[str, Any]:
        """Loads a preset onto this instance's device and checks it fits the model.

        Synthesis does this by itself when ``config.voice_preset`` is set;
        call it directly to validate a file the user picked. The file is read
        with ``torch.load(weights_only=True)``, never unpickled.

        Returns
        -------
        dict[str, Any]
            The same description :meth:`save_voice_preset` returns. ``mode``
            is ``"unknown"``: upstream's file format does not record whether
            the voice was cloned or designed.

        Raises
        ------
        ValueError
            If the file is missing, is not a saved voice prompt, or was made
            for a different codec layout.
        """
        with self._lock:
            self._ensure_initialised()
            model = self._model
            self.bind_device()
            prompt = self._preset_prompt(model, str(path))
            return self._describe_prompt(model, prompt, str(path), "unknown")

    # ── Teardown ─────────────────────────────────────────────────────────────

    def _release(self) -> None:
        """Drops the model, the ASR pipeline and every cached voice prompt."""
        model = self._model
        self._model = None
        self._loaded_key = None
        self._prompt_cache.clear()
        if model is not None and getattr(model, "_asr_pipe", None) is not None:
            model._asr_pipe = None
        del model
        self._free_cuda_memory()

    def cleanup(self) -> None:
        """Releases the model and frees its GPU memory."""
        with self._lock:
            self._release()
            self._batch_limit = 0
