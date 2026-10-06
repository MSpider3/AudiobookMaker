"""
audiobook_factory/tts_providers/qwen_provider.py
==================================================
Qwen3-TTS provider, written against qwen-tts 0.1.1 (identical to git main of
https://github.com/QwenLM/Qwen3-TTS at the time of writing).

One narrator, four ways to get it
---------------------------------
Base checkpoints (voice clone)
    ``create_voice_clone_prompt`` is run once per reference and the resulting
    prompt is reused for every chunk. With a transcript the prompt is built in
    ICL mode (reference codes + text, the faithful mode); without one it falls
    back to speaker-embedding-only mode.
Voice preset (``config.voice_preset``)
    The clone prompt saved to disk by :meth:`QwenTTSProvider.save_voice_preset`.
    Needs no reference clip and gives every GPU the identical prompt.
CustomVoice checkpoints
    One of the built-in speakers (``config.tts_timbre``), optionally steered by
    ``config.tts_instruct`` on the 1.7B checkpoint.
VoiceDesign checkpoint
    By default "design then clone": the voice described by
    ``config.tts_instruct`` is generated once on a calibration sentence, the
    clip is stored in a shared on-disk cache, and the matching Base checkpoint
    clones it for the whole book. Calling ``generate_voice_design`` per chunk
    (option ``design_then_clone=False``) gives a different voice per chunk.

Upstream facts this module relies on
------------------------------------
* ``Qwen3TTSModel._merge_generate_kwargs`` names exactly ten generation
  arguments; anything else is forwarded to ``model.generate`` and silently
  dropped there, so only those ten are ever sent (``_UPSTREAM_GENERATE_KWARGS``).
* Every published checkpoint ships ``generation_config.json`` with
  ``max_new_tokens=8192`` (about eleven minutes of audio), which is what a
  runaway chunk costs unless a budget is passed. See ``_token_budget``.
* The speech tokenizer runs at 24 kHz with 1920 samples per code frame, i.e.
  12.5 frames per second.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import logging
import math
import os
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Iterator

from audiobook_factory.tts_providers.base_tts_provider import (
    BaseTTSProvider,
    ProviderInfo,
    ProviderOption,
)

if TYPE_CHECKING:
    from audiobook_factory.pipeline import AudiobookConfig

logger = logging.getLogger(__name__)

# ── Checkpoints ──────────────────────────────────────────────────────────────

_MODEL_BASE_1B7: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
_MODEL_BASE_0B6: str = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
_MODEL_CUSTOM_1B7: str = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
_MODEL_CUSTOM_0B6: str = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
_MODEL_DESIGN_1B7: str = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"
_BASE_MODELS: tuple[str, ...] = (_MODEL_BASE_1B7, _MODEL_BASE_0B6)

# Built-in CustomVoice speakers (talker_config.spk_id of both CustomVoice checkpoints).
_PRESET_SPEAKERS: tuple[str, ...] = (
    "Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric", "Ryan", "Aiden", "Ono_Anna", "Sohee",
)
_SPEAKER_DISPLAY: dict[str, str] = {name.lower(): name for name in _PRESET_SPEAKERS}
# Upstream recommends a speaker's native language; used when no timbre is set.
_DEFAULT_SPEAKER_BY_LANGUAGE: dict[str, str] = {
    "chinese": "serena", "english": "aiden", "japanese": "ono_anna", "korean": "sohee",
}
_FALLBACK_SPEAKER: str = "serena"

# Languages every published checkpoint lists in talker_config.codec_language_id.
_LANGUAGES: tuple[str, ...] = (
    "Chinese", "English", "Japanese", "Korean", "German",
    "French", "Russian", "Portuguese", "Spanish", "Italian",
)
_AUTO_LANGUAGE: str = "Auto"
_LANGUAGE_ALIASES: dict[str, str] = {
    "": "auto", "auto": "auto", "automatic": "auto", "detect": "auto",
    "zh": "chinese", "zho": "chinese", "chi": "chinese", "cmn": "chinese",
    "mandarin": "chinese", "中文": "chinese", "汉语": "chinese", "漢語": "chinese",
    "en": "english", "eng": "english",
    "ja": "japanese", "jp": "japanese", "jpn": "japanese", "日本語": "japanese",
    "ko": "korean", "kor": "korean", "kr": "korean", "한국어": "korean",
    "de": "german", "deu": "german", "ger": "german", "deutsch": "german",
    "fr": "french", "fra": "french", "fre": "french", "français": "french", "francais": "french",
    "ru": "russian", "rus": "russian", "русский": "russian",
    "pt": "portuguese", "por": "portuguese", "português": "portuguese", "portugues": "portuguese",
    "es": "spanish", "spa": "spanish", "español": "spanish", "espanol": "spanish",
    "it": "italian", "ita": "italian", "italiano": "italian",
}

# ── Generation ───────────────────────────────────────────────────────────────

# The only names Qwen3TTSModel._merge_generate_kwargs knows. Its **kwargs
# catch-all hands anything else to model.generate(), which ignores it.
_UPSTREAM_GENERATE_KWARGS: frozenset[str] = frozenset({
    "do_sample", "top_k", "top_p", "temperature", "repetition_penalty",
    "subtalker_dosample", "subtalker_top_k", "subtalker_top_p",
    "subtalker_temperature", "max_new_tokens",
})

# Code frames per second: 24000 Hz output / 1920 samples per frame
# (speech_tokenizer/config.json: output_sample_rate, decode_upsample_rate).
_CODEC_FRAME_RATE_HZ: float = 12.5

# Speech-duration estimate per character, deliberately on the slow side of
# narration pace. Whitespace is free, so the Latin figure is per letter:
# 150 words/min at ~4.7 letters per word is ~0.085 s per letter. Mandarin
# narration runs 4-5 characters per second, Japanese ~7 morae per second and
# Korean 5-6 syllables per second. A digit usually expands to a whole word.
_SECONDS_PER_LETTER: float = 0.09
_SECONDS_PER_HAN: float = 0.28
_SECONDS_PER_KANA: float = 0.14
_SECONDS_PER_HANGUL: float = 0.20
_SECONDS_PER_DIGIT: float = 0.40
_SECONDS_PER_PUNCTUATION: float = 0.20

# Budget = expected frames x safety factor + padding, never below the floor
# and never above what the checkpoints' own generation_config.json allows.
_TOKEN_BUDGET_SAFETY: float = 2.5
_TOKEN_BUDGET_PADDING: int = 25        # 2 s for leading / trailing silence
_MIN_NEW_TOKENS: int = 150             # 12 s floor for very short chunks
_MAX_NEW_TOKENS_CAP: int = 8192
# A chunk whose audio is within this many frames of its budget never emitted
# end-of-speech: it is either a runaway or was cut off. (Upstream returns
# max_new_tokens - 1 frames for a row that never stops.)
_BUDGET_HIT_TOLERANCE: int = 2

_TORCH_COMPILE_MODE: str = "max-autotune"

# ── Reference transcript (ASR) ───────────────────────────────────────────────

# ICL cloning conditions on the transcript word for word, so a wrong word is
# audible in every chunk. large-v3-turbo is within a fraction of a WER point of
# large-v3, is ~1.6 GB in fp16 (it sits beside the 4.5 GB TTS model on a 16 GB
# T4 with room to spare) and is unloaded again as soon as the text is known.
_DEFAULT_ASR_MODEL: str = "openai/whisper-large-v3-turbo"
_ASR_MODELS: tuple[str, ...] = (
    "openai/whisper-large-v3-turbo",
    "openai/whisper-large-v3",
    "openai/whisper-medium",
    "openai/whisper-small",
    "openai/whisper-base",
    "openai/whisper-tiny",
)
_ASR_SAMPLE_RATE: int = 16000
# A failed transcription written by another process is trusted for this long,
# so concurrent workers agree on the cloning mode without blocking a later run.
_ASR_NEGATIVE_TTL_S: float = 900.0

# ── Voice design ─────────────────────────────────────────────────────────────

_DESIGN_CACHE_VERSION: int = 1
_MIN_DESIGN_SECONDS: float = 1.0
# One or two neutral narration sentences (10-15 s of speech) per language.
_CALIBRATION_TEXTS: dict[str, str] = {
    "english": (
        "The morning light spread slowly across the quiet valley, and somewhere in the "
        "distance a church bell began to ring. She closed the old book, smiled to herself, "
        "and wondered what the day would bring."
    ),
    "chinese": (
        "清晨的阳光慢慢洒满了安静的山谷，远处传来一阵悠扬的钟声。"
        "她合上那本旧书，轻轻一笑，心里想着今天会发生什么样的故事。"
    ),
    "japanese": (
        "朝の光が静かな谷にゆっくりと広がり、遠くで鐘の音が鳴りはじめました。"
        "彼女は古い本を閉じて、そっと微笑み、今日はどんな一日になるのだろうと考えました。"
    ),
    "korean": (
        "아침 햇살이 고요한 골짜기에 천천히 퍼지고, 멀리서 종소리가 울리기 시작했습니다. "
        "그녀는 낡은 책을 덮고 조용히 미소 지으며 오늘은 어떤 하루가 될지 생각했습니다."
    ),
    "german": (
        "Das Morgenlicht breitete sich langsam über das stille Tal aus, und irgendwo in der "
        "Ferne begann eine Glocke zu läuten. Sie schloss das alte Buch, lächelte leise und "
        "fragte sich, was der Tag wohl bringen würde."
    ),
    "french": (
        "La lumière du matin s'étendait lentement sur la vallée silencieuse, et quelque part "
        "au loin une cloche se mit à sonner. Elle referma le vieux livre, sourit doucement et "
        "se demanda ce que la journée lui réservait."
    ),
    "russian": (
        "Утренний свет медленно разливался по тихой долине, и где-то вдалеке зазвонил колокол. "
        "Она закрыла старую книгу, тихо улыбнулась и подумала о том, что принесёт ей этот день."
    ),
    "portuguese": (
        "A luz da manhã espalhava-se lentamente pelo vale silencioso, e ao longe um sino "
        "começou a tocar. Ela fechou o velho livro, sorriu baixinho e perguntou-se o que o dia "
        "lhe traria."
    ),
    "spanish": (
        "La luz de la mañana se extendía lentamente por el valle silencioso, y en algún lugar "
        "a lo lejos una campana empezó a sonar. Ella cerró el viejo libro, sonrió en silencio "
        "y se preguntó qué le traería el día."
    ),
    "italian": (
        "La luce del mattino si diffondeva lentamente sulla valle silenziosa, e da qualche "
        "parte in lontananza una campana cominciò a suonare. Lei chiuse il vecchio libro, "
        "sorrise piano e si chiese che cosa le avrebbe portato la giornata."
    ),
}

# ── Voice presets ────────────────────────────────────────────────────────────

_PRESET_FORMAT: str = "abm-qwen3-tts-voice-preset"
_PRESET_FORMAT_VERSION: int = 1
_MAX_PRESET_BYTES: int = 64 * 1024 * 1024
_MAX_PRESET_REF_FRAMES: int = 20000       # far beyond any sane reference clip
_MAX_PRESET_EMBEDDING_DIM: int = 16384
_MAX_PRESET_TEXT_CHARS: int = 20000
_PRESET_META_KEYS: tuple[str, ...] = (
    "format", "format_version", "model_id", "tts_model_size", "tokenizer_type",
    "sample_rate", "speaker_embedding_dim", "created_at", "source", "language", "instruct",
)
# Reference clips longer than this slow every chunk down in ICL mode.
_LONG_REFERENCE_SECONDS: float = 30.0

# ── Shared on-disk cache ─────────────────────────────────────────────────────

_CACHE_DIR_ENV: str = "ABM_QWEN_CACHE_DIR"
_LOCK_TIMEOUT_S: float = 3600.0           # a design may include a 4.5 GB download
_LOCK_STALE_S: float = 3600.0             # marker-file fallback only: holder presumed dead
_LOCK_POLL_S: float = 0.2
_MAX_JSON_BYTES: int = 4 * 1024 * 1024

_MAX_PROMPT_CACHE: int = 4
_MAX_VOICE_REF_CACHE: int = 8
_VOICE_REF_CACHE: dict[str, str] = {}
_VOICE_REF_LOCK = threading.Lock()

# Transcripts agreed on by every provider instance in this process.
_SHARED_TRANSCRIPTS: dict[str, str] = {}
_SHARED_TRANSCRIPT_LOCK = threading.Lock()

_IMPORT_LOCK = threading.Lock()

# ASR pipeline factory. None means "use transformers.pipeline"; transformers is
# imported on first use so this module loads without it.
pipeline: Any = None


class _RunawayGenerationError(RuntimeError):
    """A chunk still filled its whole token budget after the retry."""


# Errors that a retry or a smaller batch cannot fix.
_NON_RETRYABLE: tuple[type[BaseException], ...] = (
    ValueError, TypeError, NotImplementedError, _RunawayGenerationError,
)


@dataclass
class _PromptItem:
    """Structural stand-in for ``qwen_tts.VoiceClonePromptItem``.

    Upstream only reads these five attributes, so this is used when the real
    class cannot be imported.
    """

    ref_code: Any
    ref_spk_embedding: Any
    x_vector_only_mode: bool
    icl_mode: bool
    ref_text: str | None = None


@dataclass(frozen=True)
class _Reference:
    """A clone reference: the clip, what identifies it, and its transcript."""

    path: str
    identity: str
    text: str | None
    origin: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class _DesignSpec:
    """Everything that determines a designed voice."""

    model_id: str
    instruct: str
    language: str
    seed: int
    text: str

    @property
    def key(self) -> str:
        """Cache key shared by every process designing this voice."""
        blob = json.dumps(
            [_DESIGN_CACHE_VERSION, self.model_id, self.instruct, self.language, self.seed, self.text],
            ensure_ascii=False,
        )
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:32]


@dataclass
class _GenerationPlan:
    """One upstream ``generate_*`` call, minus the texts and sampling arguments."""

    call: Callable[..., Any]
    per_item: dict[str, Any]
    shared: dict[str, Any]

    def kwargs_for(self, count: int) -> dict[str, Any]:
        """Keyword arguments for a batch of *count* texts."""
        kwargs: dict[str, Any] = {name: [value] * count for name, value in self.per_item.items()}
        kwargs.update(self.shared)
        return kwargs


# ── Module helpers ───────────────────────────────────────────────────────────

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


def _cache_root() -> str:
    """Directory shared by every provider instance and process on this machine."""
    candidates: list[str] = []
    override = (os.environ.get(_CACHE_DIR_ENV) or "").strip()
    if override:
        candidates.append(override)
    candidates.append(os.path.join(os.path.expanduser("~"), ".cache", "audiobook_maker", "qwen"))
    candidates.append(os.path.join(tempfile.gettempdir(), "abm_qwen_cache"))
    for candidate in candidates:
        try:
            os.makedirs(candidate, exist_ok=True)
        except OSError:
            continue
        if os.access(candidate, os.W_OK):
            return candidate
    raise RuntimeError(f"No writable cache directory for Qwen3-TTS (tried: {', '.join(candidates)}).")


def _try_os_lock(handle: Any) -> bool | None:
    """Tries a kernel file lock on *handle* without blocking.

    Returns True when acquired, False when another holder has it, and None
    when the platform or filesystem offers no kernel lock.
    """
    try:
        import fcntl
    except ImportError:
        fcntl = None  # type: ignore[assignment]
    if fcntl is not None:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except BlockingIOError:
            return False
        except OSError:
            return None
    try:
        import msvcrt
    except ImportError:
        return None
    try:
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)  # type: ignore[attr-defined]
        return True
    except OSError:
        return False


def _release_os_lock(handle: Any) -> None:
    try:
        import fcntl
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        return
    except ImportError:
        pass
    except OSError:
        return
    try:
        import msvcrt
        handle.seek(0)
        msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)  # type: ignore[attr-defined]
    except (ImportError, OSError):
        pass


@contextlib.contextmanager
def _file_lock(lock_path: str, timeout: float = _LOCK_TIMEOUT_S) -> Iterator[None]:
    """Cross-process and cross-thread mutual exclusion on *lock_path*.

    Uses ``fcntl.flock`` on POSIX and ``msvcrt.locking`` on Windows, both of
    which the kernel releases if the holder dies. Where neither works (some
    network filesystems) it falls back to an exclusively created marker file
    that is considered abandoned after ``_LOCK_STALE_S`` seconds.

    Raises
    ------
    TimeoutError
        If the lock is not obtained within *timeout* seconds.
    """
    os.makedirs(os.path.dirname(lock_path) or ".", exist_ok=True)
    deadline = time.monotonic() + timeout
    handle = open(lock_path, "a+b")
    try:
        state = _try_os_lock(handle)
        while state is False:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for lock {lock_path}")
            time.sleep(_LOCK_POLL_S)
            state = _try_os_lock(handle)
        if state:
            try:
                yield
            finally:
                _release_os_lock(handle)
            return
    finally:
        handle.close()

    marker = lock_path + ".excl"
    while True:
        try:
            os.close(os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY))
            break
        except FileExistsError:
            try:
                if time.time() - os.path.getmtime(marker) > _LOCK_STALE_S:
                    os.unlink(marker)
                    continue
            except OSError:
                continue
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Timed out waiting for lock {marker}")
            time.sleep(_LOCK_POLL_S)
    try:
        yield
    finally:
        try:
            os.unlink(marker)
        except OSError:
            pass


def _atomic_write_bytes(path: str, data: bytes) -> None:
    tmp_path = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"
    with open(tmp_path, "wb") as fh:
        fh.write(data)
    os.replace(tmp_path, path)


def _read_json(path: str) -> dict[str, Any] | None:
    """Reads a small JSON object, or None when missing, oversized or malformed."""
    try:
        if not os.path.isfile(path) or os.path.getsize(path) > _MAX_JSON_BYTES:
            return None
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _read_local_checkpoint_config(path: str) -> dict[str, Any] | None:
    """Returns config.json of a local Qwen3-TTS checkpoint directory, else None."""
    if not path or not os.path.isdir(path):
        return None
    data = _read_json(os.path.join(path, "config.json"))
    if data is None or data.get("model_type") != "qwen3_tts":
        return None
    return data


def _canonical_language(name: str | None) -> str | None:
    """Maps a language name or code to upstream's lowercase name.

    Returns ``"auto"`` for an explicit auto request and None when the name is
    not one this module knows.
    """
    key = (name or "").strip().lower().replace("_", "-")
    known = {language.lower() for language in _LANGUAGES}
    for candidate in (key, key.split("-")[0], key.split("(")[0].strip(), key.split(" ")[0]):
        if candidate in known:
            return candidate
        if candidate in _LANGUAGE_ALIASES:
            return _LANGUAGE_ALIASES[candidate]
    return None


def _static_language(name: str | None) -> str:
    """Language name for a call made before any model is loaded."""
    canonical = _canonical_language(name)
    if canonical is None or canonical == "auto":
        return _AUTO_LANGUAGE
    return canonical.title()


def _estimate_speech_seconds(text: str) -> float:
    """Estimates how long *text* takes to speak, by script.

    A Han character is a whole syllable (often a whole word) while a Latin
    letter is a fraction of one, so the same character count is several times
    longer in Chinese than in English.
    """
    total = 0.0
    for ch in text:
        if ch.isspace():
            continue
        cp = ord(ch)
        if (0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF
                or 0xF900 <= cp <= 0xFAFF or 0x20000 <= cp <= 0x2FA1F):
            total += _SECONDS_PER_HAN
        elif 0x3040 <= cp <= 0x30FF or 0x31F0 <= cp <= 0x31FF or 0xFF66 <= cp <= 0xFF9F:
            total += _SECONDS_PER_KANA
        elif 0xAC00 <= cp <= 0xD7A3 or 0x1100 <= cp <= 0x11FF or 0x3130 <= cp <= 0x318F:
            total += _SECONDS_PER_HANGUL
        elif ch.isdigit():
            total += _SECONDS_PER_DIGIT
        elif ch.isalpha():
            total += _SECONDS_PER_LETTER
        else:
            total += _SECONDS_PER_PUNCTUATION
    return total


def _token_budget(texts: list[str]) -> int:
    """``max_new_tokens`` for a batch: sized for its longest text.

    A batch generates until every row has emitted end-of-speech, so one
    runaway row costs the whole budget. The budget is 2.5x the expected
    length of the longest text plus padding, floored at 12 s and capped at
    the checkpoints' own limit.
    """
    longest = max((_estimate_speech_seconds(text) for text in texts), default=0.0)
    frames = math.ceil(longest * _CODEC_FRAME_RATE_HZ * _TOKEN_BUDGET_SAFETY) + _TOKEN_BUDGET_PADDING
    return int(min(_MAX_NEW_TOKENS_CAP, max(_MIN_NEW_TOKENS, frames)))


def _is_out_of_memory(exc: BaseException) -> bool:
    try:
        import torch
        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return True
    except Exception:
        pass
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


def _patch_check_model_inputs() -> None:
    """Lets qwen-tts import on transformers builds whose ``check_model_inputs`` is a bare decorator.

    qwen-tts writes ``@check_model_inputs()``. That call form fails with a
    TypeError on transformers versions other than the one it pins (seen with
    4.57.1 and 5.x), so a call without arguments is made to return the
    decorator itself.
    """
    try:
        import transformers.utils.generic as generic
    except Exception:
        return
    original = getattr(generic, "check_model_inputs", None)
    if original is None or getattr(original, "_is_patched_for_qwen", False):
        return

    def _patched_check(*args: Any, **kwargs: Any) -> Any:
        if not args and not kwargs:
            return original
        return original(*args, **kwargs)

    _patched_check._is_patched_for_qwen = True  # type: ignore[attr-defined]
    generic.check_model_inputs = _patched_check


def _import_qwen_tts() -> Any:
    """Imports ``qwen_tts`` once and returns ``Qwen3TTSModel``.

    The import prints banners to stdout/stderr. They are silenced only for the
    duration of the import and only under ``_IMPORT_LOCK``, so two providers
    loading in different threads cannot leave ``sys.stdout`` swapped.
    """
    with _IMPORT_LOCK:
        module = sys.modules.get("qwen_tts")
        if module is not None and hasattr(module, "Qwen3TTSModel"):
            return module.Qwen3TTSModel
        _patch_check_model_inputs()
        try:
            with open(os.devnull, "w") as sink, \
                    contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
                from qwen_tts import Qwen3TTSModel  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "The qwen-tts package is not installed. Run: "
                "pip install 'qwen-tts @ git+https://github.com/QwenLM/Qwen3-TTS.git'"
            ) from exc
        return Qwen3TTSModel


def _prompt_item_class() -> Any:
    module = sys.modules.get("qwen_tts")
    return getattr(module, "VoiceClonePromptItem", None) or _PromptItem


def _read_preset_file(path: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Loads and validates a voice preset without trusting its contents.

    The file is read with ``torch.load(weights_only=True)``, which refuses to
    unpickle anything but tensors and plain containers, and every field is
    type- and range-checked before use.

    Returns
    -------
    tuple[list[dict], dict]
        Validated prompt items (tensors on CPU) and the metadata.

    Raises
    ------
    ValueError
        If the file is missing, too large, or not a valid preset.
    """
    import torch

    if not path or not os.path.isfile(path):
        raise ValueError(f"Voice preset not found: {path}")
    size = os.path.getsize(path)
    if size == 0 or size > _MAX_PRESET_BYTES:
        raise ValueError(f"Voice preset {path} has an implausible size ({size} bytes).")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise ValueError(f"{path} is not a valid Qwen3-TTS voice preset: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("items"), list) or not payload["items"]:
        raise ValueError(f"{path} is not a valid Qwen3-TTS voice preset: no prompt items.")

    declared_format = payload.get("format")
    if declared_format is not None and declared_format != _PRESET_FORMAT:
        raise ValueError(f"{path} is a '{declared_format}' file, not a Qwen3-TTS voice preset.")
    version = payload.get("format_version", 0)
    if not isinstance(version, int) or isinstance(version, bool) or version > _PRESET_FORMAT_VERSION:
        raise ValueError(
            f"Voice preset {path} uses format version {version!r}; this build reads up to "
            f"version {_PRESET_FORMAT_VERSION}. Update AudiobookMaker or re-save the preset."
        )

    items: list[dict[str, Any]] = []
    for index, raw in enumerate(payload["items"]):
        if not isinstance(raw, dict):
            raise ValueError(f"Voice preset {path}: item {index} is not a mapping.")
        embedding = raw.get("ref_spk_embedding")
        if not torch.is_tensor(embedding) or not embedding.is_floating_point():
            raise ValueError(f"Voice preset {path}: item {index} has no speaker embedding.")
        embedding = embedding.detach().reshape(-1).to(torch.float32)
        if not 0 < embedding.numel() <= _MAX_PRESET_EMBEDDING_DIM or not bool(torch.isfinite(embedding).all()):
            raise ValueError(f"Voice preset {path}: item {index} has an invalid speaker embedding.")

        x_vector_only = bool(raw.get("x_vector_only_mode", False))
        icl_mode = bool(raw.get("icl_mode", not x_vector_only))
        ref_text = raw.get("ref_text")
        if ref_text is not None and (not isinstance(ref_text, str) or len(ref_text) > _MAX_PRESET_TEXT_CHARS):
            raise ValueError(f"Voice preset {path}: item {index} has an invalid reference text.")

        ref_code = raw.get("ref_code")
        if x_vector_only or not icl_mode:
            ref_code = None
        if ref_code is not None:
            if (not torch.is_tensor(ref_code) or ref_code.is_floating_point() or ref_code.is_complex()
                    or ref_code.dtype == torch.bool or ref_code.ndim not in (1, 2)
                    or not 0 < ref_code.shape[0] <= _MAX_PRESET_REF_FRAMES
                    or ref_code.numel() == 0 or int(ref_code.min()) < 0):
                raise ValueError(f"Voice preset {path}: item {index} has invalid reference codes.")
            ref_code = ref_code.detach().to(torch.long)
        if icl_mode and (ref_code is None or not (ref_text or "").strip()):
            raise ValueError(
                f"Voice preset {path}: item {index} is an ICL prompt without reference codes or text."
            )
        items.append({
            "ref_code": ref_code,
            "ref_spk_embedding": embedding,
            "x_vector_only_mode": x_vector_only,
            "icl_mode": icl_mode,
            "ref_text": ref_text,
        })

    meta: dict[str, Any] = {}
    for key in _PRESET_META_KEYS:
        value = payload.get(key)
        if isinstance(value, (str, int, float)) and not isinstance(value, bool):
            meta[key] = value
    return items, meta


def _preset_info(path: str, items: list[dict[str, Any]], meta: dict[str, Any]) -> dict[str, Any]:
    """Tensor-free description of a preset, for the UI."""
    first = items[0]
    code = first["ref_code"]
    info = dict(meta)
    info.update(
        path=path,
        mode="x_vector" if first["x_vector_only_mode"] else "icl",
        ref_text=first["ref_text"] or "",
        ref_frames=int(code.shape[0]) if code is not None else 0,
        speaker_embedding_dim=int(first["ref_spk_embedding"].shape[-1]),
    )
    return info


class QwenTTSProvider(BaseTTSProvider):
    """Local Qwen3-TTS provider for the Base, CustomVoice and VoiceDesign checkpoints.

    Besides the :class:`BaseTTSProvider` interface it offers, for the UI:
    :meth:`save_voice_preset`, :meth:`load_voice_preset`,
    :meth:`read_voice_preset_info`, :meth:`design_voice`,
    :meth:`list_speakers` and :meth:`list_languages`.
    """

    INFO = ProviderInfo(
        name="qwen",
        display_name="Qwen3-TTS",
        description=(
            "Alibaba's multilingual TTS: clone a narrator from a short clip (Base), pick a "
            "built-in speaker (CustomVoice) or describe a voice in words (VoiceDesign)."
        ),
        license="Apache-2.0",
        commercial_use=True,
        homepage="https://github.com/QwenLM/Qwen3-TTS",
        default_model=_MODEL_BASE_1B7,
        models=(
            _MODEL_BASE_1B7,
            _MODEL_BASE_0B6,
            _MODEL_CUSTOM_1B7,
            _MODEL_CUSTOM_0B6,
            _MODEL_DESIGN_1B7,
        ),
        # speech_tokenizer/config.json: output_sample_rate (also the default in
        # qwen_tts/core/tokenizer_12hz/configuration_qwen3_tts_tokenizer_v2.py).
        native_sample_rate=24000,
        # 1.7B weights are 3.9 GB in bf16 plus a 0.7 GB speech tokenizer; the
        # rest is the batch's KV cache and the transient Whisper model.
        min_vram_gb=6.0,
        languages=(_AUTO_LANGUAGE,) + _LANGUAGES,
        supports_voice_clone=True,
        transcript="optional",
        supports_instruct=True,
        supports_batch=True,
        supports_speed=False,
        supports_seed=True,
        preset_voices=_PRESET_SPEAKERS,
        options=(
            ProviderOption(
                key="x_vector_only_mode", label="Speaker embedding only", kind="bool", default=False,
                help="Clone from the speaker embedding alone and ignore the reference transcript. "
                     "Less faithful than the default ICL mode.",
            ),
            ProviderOption(
                key="auto_transcribe", label="Transcribe reference automatically", kind="bool", default=True,
                help="Run Whisper on the reference clip when no transcript is supplied.",
            ),
            ProviderOption(
                key="asr_model", label="Reference transcription model", kind="choice",
                default=_DEFAULT_ASR_MODEL, choices=_ASR_MODELS,
                help="Whisper model used once to transcribe the reference clip, then unloaded.",
            ),
            ProviderOption(
                key="design_then_clone", label="Design the voice once, then clone it", kind="bool",
                default=True,
                help="VoiceDesign only. Keeps one narrator for the whole book; turning it off "
                     "designs a new voice for every chunk.",
            ),
            ProviderOption(
                key="design_clone_model", label="Clone model for designed voices", kind="choice",
                default=_MODEL_BASE_1B7, choices=_BASE_MODELS,
                help="Base checkpoint that reads the book in the designed voice.",
            ),
            ProviderOption(
                key="design_text", label="Voice design sample text", kind="str", default="",
                help="Sentence spoken to create the designed voice. Empty uses a built-in "
                     "sentence in the book's language.",
            ),
            ProviderOption(
                key="do_sample", label="Sampling", kind="bool", default=True,
                help="Sample the main codebook; off means greedy decoding.",
            ),
            ProviderOption(
                key="subtalker_dosample", label="Sub-talker sampling", kind="bool", default=True,
                help="Sample the residual codebooks; off means greedy decoding.",
            ),
            ProviderOption(
                key="subtalker_temperature", label="Sub-talker temperature", kind="float", default=0.9,
                minimum=0.05, maximum=1.5, step=0.05,
                help="Sampling temperature of the residual codebooks.",
            ),
            ProviderOption(
                key="subtalker_top_k", label="Sub-talker top-k", kind="int", default=50,
                minimum=1, maximum=2048, step=1,
                help="Top-k of the residual codebooks.",
            ),
            ProviderOption(
                key="subtalker_top_p", label="Sub-talker top-p", kind="float", default=1.0,
                minimum=0.05, maximum=1.0, step=0.05,
                help="Nucleus threshold of the residual codebooks.",
            ),
            ProviderOption(
                key="max_new_tokens", label="Max codec tokens per chunk", kind="int", default=0,
                minimum=0, maximum=_MAX_NEW_TOKENS_CAP, step=1,
                help="0 sizes the limit from the chunk text (12.5 tokens are one second of audio).",
            ),
            ProviderOption(
                key="non_streaming_mode", label="Text input mode", kind="choice", default="auto",
                choices=("auto", "on", "off"),
                help="Feed the whole text up front (on) or simulate streaming input (off). "
                     "auto keeps each mode's upstream default.",
            ),
        ),
        pip_requirements=("qwen-tts @ git+https://github.com/QwenLM/Qwen3-TTS.git",),
        install_notes=(
            "Optional: 'pip install flash-attn --no-build-isolation' lowers VRAM use. "
            "A reference clip without a transcript is transcribed once with Whisper "
            f"({_DEFAULT_ASR_MODEL}, about 1.6 GB download)."
        ),
    )

    # Defaults that let methods run on an instance built without __init__.
    _model: Any = None
    _asr_pipe: Any = None
    _device: str = "cpu"
    _dtype_override: str | None = None
    _loaded_model_name: str | None = None
    _loaded_quantization: str | None = None
    _loaded_signature: tuple[Any, ...] | None = None

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
        self._dtype_override = dtype_override
        self._init_with_device(device or getattr(config, "device", "cuda"), config)

    def _init_with_device(self, device: str, config: AudiobookConfig) -> None:
        """Initialize instance variables pinned to target device."""
        from audiobook_factory.preflight import _apply_python312_pickle_patch
        _apply_python312_pickle_patch()

        self.config = config
        self._device = device
        self._model = None
        self._asr_pipe = None
        self._loaded_model_name = None
        self._loaded_quantization = None
        self._loaded_signature = None
        self._ensure_state()

    def _ensure_state(self) -> None:
        """Creates per-instance containers that ``__init__`` normally sets up."""
        state = self.__dict__
        if "_lock" not in state:
            state["_lock"] = threading.RLock()
        for name in ("_voice_prompt_cache", "_transcript_cache", "_digest_cache", "_design_memo"):
            state.setdefault(name, {})
        state.setdefault("_warned", set())

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
        return f"Qwen3-TTS ({self._configured_model_id()}) [{self._device}]"

    def _warn_once(self, key: str, message: str, *args: Any) -> None:
        self._ensure_state()
        if key in self._warned:
            return
        self._warned.add(key)
        logger.warning(message, *args)

    # ── Which checkpoint, which mode ─────────────────────────────────────────

    def _configured_model_id(self) -> str:
        """The checkpoint the user selected.

        A hub id must be one of ``INFO.models``. A local directory is accepted
        when its ``config.json`` declares a Qwen3-TTS model, which is how
        checkpoints produced by upstream's fine-tuning script are loaded.
        """
        requested = (getattr(self.config, "tts_model_name", "") or "").strip()
        if requested and requested not in self.info().models \
                and _read_local_checkpoint_config(requested) is not None:
            return requested
        return self.resolve_model_id()

    @staticmethod
    def _declared_type(model_id: str) -> str:
        """``tts_model_type`` of a checkpoint, known before it is loaded."""
        local = _read_local_checkpoint_config(model_id)
        if local is not None:
            declared = str(local.get("tts_model_type") or "base")
            return declared if declared in ("base", "custom_voice", "voice_design") else "base"
        name = model_id.lower()
        if "customvoice" in name:
            return "custom_voice"
        if "voicedesign" in name:
            return "voice_design"
        return "base"

    def _mode(self) -> str:
        """``clone``, ``custom_voice``, ``voice_design`` or ``design_clone``."""
        declared = self._declared_type(self._configured_model_id())
        if declared == "voice_design":
            return "design_clone" if self.option("design_then_clone", True) else "voice_design"
        if declared == "custom_voice":
            return "custom_voice"
        return "clone"

    def _target_model_id(self) -> str:
        """The checkpoint that must be loaded to synthesize the book."""
        if self._mode() == "design_clone":
            choice = str(self.option("design_clone_model", _MODEL_BASE_1B7) or "")
            return choice if choice in _BASE_MODELS else _MODEL_BASE_1B7
        return self._configured_model_id()

    def _preset_path(self) -> str:
        """``config.voice_preset`` when the current mode clones a voice."""
        path = (getattr(self.config, "voice_preset", "") or "").strip()
        if path and self._mode() in ("clone", "design_clone"):
            return path
        return ""

    def _needs_voice_ref(self) -> bool:
        """Whether synthesis consumes a reference clip.

        Only a Base checkpoint without a voice preset does; CustomVoice speaks
        from a built-in timbre and VoiceDesign from a text description.
        """
        return self._mode() == "clone" and not self._preset_path()

    def _check_voice_source(self, voice_ref: str | bytes | None) -> None:
        if not self._needs_voice_ref():
            return
        ref = voice_ref or getattr(self.config, "voice_file", "") or None
        if not ref:
            raise ValueError(
                "Qwen3-TTS Base checkpoints clone a voice: set a reference clip (voice_file) "
                "or a saved voice preset (voice_preset)."
            )
        self._validate_voice_ref(ref)

    def _loaded_model_type(self) -> str:
        return str(getattr(getattr(self._model, "model", None), "tts_model_type", None) or "base")

    # ── Language and speaker ─────────────────────────────────────────────────

    def _supported_languages(self) -> set[str]:
        getter = getattr(self._model, "get_supported_languages", None)
        if callable(getter):
            try:
                listed = getter()
                if listed:
                    return {str(name).lower() for name in listed}
            except Exception as exc:
                logger.debug("get_supported_languages failed: %s", exc)
        return {name.lower() for name in self.info().languages}

    def _resolve_language(self) -> str:
        """Maps ``config.language`` onto a name the loaded model accepts.

        Unknown or unsupported languages become ``"Auto"`` (the model detects
        the language from the text) instead of failing mid-book.
        """
        requested = getattr(self.config, "language", "") or ""
        supported = self._supported_languages()
        canonical = _canonical_language(requested)
        if canonical == "auto":
            return _AUTO_LANGUAGE
        if canonical is None and requested.strip().lower() in supported:
            canonical = requested.strip().lower()
        if canonical is not None and canonical in supported:
            return canonical.title()
        self._warn_once(
            f"language:{requested}",
            "[QwenTTS] Language '%s' is not supported by %s (supported: %s); "
            "using automatic language detection.",
            requested, self._loaded_model_name or self._configured_model_id(),
            ", ".join(sorted(supported)),
        )
        return _AUTO_LANGUAGE

    def _resolve_speaker(self, language: str) -> str:
        """Validates ``config.tts_timbre`` against the loaded CustomVoice model.

        Raises
        ------
        ValueError
            If the speaker is not one the checkpoint knows; the message lists
            the valid names.
        """
        getter = getattr(self._model, "get_supported_speakers", None)
        supported = [str(name).lower() for name in (getter() or [])] if callable(getter) else []
        requested = (getattr(self.config, "tts_timbre", "") or "").strip()
        if not requested:
            default = _DEFAULT_SPEAKER_BY_LANGUAGE.get(language.lower(), _FALLBACK_SPEAKER)
            if supported and default not in supported:
                default = supported[0]
            self._warn_once(
                f"speaker-default:{default}",
                "[QwenTTS] No speaker selected (tts_timbre); using '%s'.",
                _SPEAKER_DISPLAY.get(default, default),
            )
            return default
        key = requested.lower()
        if supported and key not in supported:
            key = key.replace(" ", "_").replace("-", "_")
        if supported and key not in supported:
            valid = ", ".join(_SPEAKER_DISPLAY.get(name, name) for name in supported)
            raise ValueError(
                f"Unknown Qwen3-TTS speaker '{requested}' for {self._loaded_model_name}. "
                f"Valid speakers: {valid}."
            )
        return key

    def list_speakers(self) -> list[str]:
        """Built-in speakers of the configured (or loaded) CustomVoice checkpoint.

        Returns
        -------
        list[str]
            Speaker names usable as ``config.tts_timbre``; empty for Base and
            VoiceDesign checkpoints.
        """
        if self._model is not None and self._loaded_model_type() == "custom_voice":
            names = list(self._model.get_supported_speakers() or [])
        else:
            model_id = self._configured_model_id()
            if self._declared_type(model_id) != "custom_voice":
                return []
            local = _read_local_checkpoint_config(model_id)
            if local is None:
                return list(_PRESET_SPEAKERS)
            talker = local.get("talker_config")
            names = list((talker.get("spk_id") or {}) if isinstance(talker, dict) else [])
        return [_SPEAKER_DISPLAY.get(str(name).lower(), str(name)) for name in names]

    def list_languages(self) -> list[str]:
        """Languages accepted as ``config.language``, ``"Auto"`` first."""
        if self._model is None:
            return list(self.info().languages)
        names = sorted(self._supported_languages() - {"auto"})
        return [_AUTO_LANGUAGE] + [name.title() for name in names]

    # ── Reference clip and transcript ────────────────────────────────────────

    @staticmethod
    def _voice_ref_signature(ref_path: str) -> str:
        """Identifies a reference file by path *and* current contents.

        A path alone goes stale when the file is re-recorded or re-processed
        in place, which would keep serving the previous voice from cache.
        """
        try:
            stat = os.stat(ref_path)
            return f"{ref_path}|{stat.st_size}|{stat.st_mtime_ns}"
        except OSError:
            return ref_path

    def _voice_ref_digest(self, ref_path: str) -> str | None:
        """SHA-256 of the reference file, or None when it cannot be read.

        The digest (not the path) keys every cache, so the per-GPU instances
        agree even when each wrote the reference bytes to its own temp file.
        """
        self._ensure_state()
        signature = self._voice_ref_signature(ref_path)
        if signature in self._digest_cache:
            return self._digest_cache[signature]
        digest: str | None
        try:
            hasher = hashlib.sha256()
            with open(ref_path, "rb") as fh:
                for block in iter(lambda: fh.read(1 << 20), b""):
                    hasher.update(block)
            digest = hasher.hexdigest()
        except OSError:
            digest = None
        if len(self._digest_cache) >= 32:
            self._digest_cache.clear()
        self._digest_cache[signature] = digest
        return digest

    def _resolve_voice_ref(self, voice_ref: str | bytes | None) -> str | None:
        """Resolves a voice reference (file path string or raw WAV bytes) into a valid file path string.

        If voice_ref is raw bytes, writes it to a cached temporary .wav file (keyed by SHA256)
        and returns the file path string. Subsequent calls with the same bytes return the cached path.
        """
        if not voice_ref:
            return None
        if isinstance(voice_ref, str):
            return voice_ref
        if isinstance(voice_ref, (bytes, bytearray)):
            data = bytes(voice_ref)
            key = hashlib.sha256(data).hexdigest()[:16]
            cached = _voice_ref_cache_get(key)
            if cached is not None:
                return cached
            temp_path = os.path.join(tempfile.gettempdir(), f"qwen_voiceref_{key}.wav")
            if not os.path.exists(temp_path) or os.path.getsize(temp_path) != len(data):
                _atomic_write_bytes(temp_path, data)
            _voice_ref_cache_put(key, temp_path)
            logger.debug("Voice ref cached to %s", temp_path)
            return temp_path
        raise ValueError(f"voice_ref must be bytes or str, got {type(voice_ref).__name__}")

    def _asr_model_id(self) -> str:
        choice = str(self.option("asr_model", _DEFAULT_ASR_MODEL) or "")
        if choice in _ASR_MODELS:
            return choice
        self._warn_once(
            f"asr-model:{choice}",
            "[QwenTTS] Unknown asr_model '%s'; using %s.", choice, _DEFAULT_ASR_MODEL,
        )
        return _DEFAULT_ASR_MODEL

    def _get_voice_transcript(self, ref_path: str) -> str | None:
        """Retrieves or transcribes reference audio text.

        Tries in order:
        1. ``config.voice_transcript`` (or legacy ``config.ref_text``).
        2. A sidecar ``.txt`` file next to *ref_path*.
        3. This instance's memo, including remembered failures.
        4. The transcript shared by every instance and process (memory, then
           the on-disk cache), running Whisper once if nobody has it yet.

        Returns
        -------
        str | None
            The transcript, or None when there is none (cloning then uses
            speaker-embedding-only mode).
        """
        self._ensure_state()
        configured = self.reference_transcript(ref_path)
        if not configured:
            configured = str(getattr(self.config, "ref_text", "") or "").strip()
        if configured:
            return configured

        auto = bool(self.option("auto_transcribe", True))
        asr_model = self._asr_model_id() if auto else "off"
        memo_key = f"{self._voice_ref_signature(ref_path)}|{asr_model}"
        if memo_key in self._transcript_cache:
            return self._transcript_cache[memo_key] or None

        transcript = self._shared_transcript(ref_path, asr_model) if auto else ""
        # Remember misses too, so a failure is not retried (and re-logged) per batch.
        if len(self._transcript_cache) >= 32:
            self._transcript_cache.clear()
        self._transcript_cache[memo_key] = transcript
        return transcript or None

    def _shared_transcript(self, ref_path: str, asr_model: str) -> str:
        """Returns the one transcript every instance uses for this clip.

        The first caller transcribes while holding a file lock; everyone else
        (the other GPU's instance, another process) reads its result, so two
        GPUs can never clone the same clip in different modes.
        """
        digest = self._voice_ref_digest(ref_path)
        identity = digest or f"path:{ref_path}"
        key = hashlib.sha256(f"{identity}|{asr_model}".encode("utf-8", errors="replace")).hexdigest()[:32]
        with _SHARED_TRANSCRIPT_LOCK:
            if key in _SHARED_TRANSCRIPTS:
                return _SHARED_TRANSCRIPTS[key]

        if digest is None:
            # Unreadable file: nothing stable to key a disk entry on.
            text = self._run_asr(ref_path, asr_model)
        else:
            try:
                entry_path = os.path.join(_cache_root(), "transcripts", f"{key}.json")
                with _file_lock(entry_path + ".lock"):
                    with _SHARED_TRANSCRIPT_LOCK:
                        if key in _SHARED_TRANSCRIPTS:
                            return _SHARED_TRANSCRIPTS[key]
                    entry = _read_json(entry_path) or {}
                    cached = entry.get("text")
                    try:
                        age = time.time() - float(entry.get("created_at") or 0.0)
                    except (TypeError, ValueError):
                        age = -1.0   # unreadable timestamp: treat the entry as expired
                    if isinstance(cached, str) and cached.strip():
                        text = cached.strip()
                        logger.info("[QwenTTS] Reference transcript loaded from cache: \"%s\"", text[:80])
                    elif entry and 0.0 <= age < _ASR_NEGATIVE_TTL_S:
                        text = ""
                    else:
                        text = self._run_asr(ref_path, asr_model)
                        _atomic_write_bytes(entry_path, json.dumps({
                            "text": text, "asr_model": asr_model, "created_at": time.time(),
                        }, ensure_ascii=False).encode("utf-8"))
            except (OSError, RuntimeError, TimeoutError) as exc:
                logger.warning("[QwenTTS] Transcript cache unavailable (%s); transcribing locally.", exc)
                text = self._run_asr(ref_path, asr_model)

        with _SHARED_TRANSCRIPT_LOCK:
            return _SHARED_TRANSCRIPTS.setdefault(key, text)

    @staticmethod
    def _asr_input(ref_path: str) -> Any:
        """Decodes the clip to 16 kHz mono so the ASR pipeline needs no ffmpeg."""
        try:
            import numpy as np
            import soundfile as sf

            audio, sample_rate = sf.read(ref_path, dtype="float32", always_2d=True)
            mono = audio.mean(axis=1)
            if sample_rate != _ASR_SAMPLE_RATE:
                try:
                    import librosa
                    mono = librosa.resample(mono, orig_sr=sample_rate, target_sr=_ASR_SAMPLE_RATE)
                    sample_rate = _ASR_SAMPLE_RATE
                except Exception:
                    pass  # the pipeline resamples itself
            return {"raw": np.ascontiguousarray(mono, dtype=np.float32), "sampling_rate": int(sample_rate)}
        except Exception:
            return ref_path

    def _run_asr(self, ref_path: str, asr_model: str) -> str:
        """Transcribes the clip with Whisper and frees the model again.

        Returns ``""`` on any failure; the caller records that as a miss.
        """
        text = ""
        try:
            import torch

            factory = pipeline
            if factory is None:
                from transformers import pipeline as factory  # type: ignore[no-redef]
            if self._asr_pipe is None:
                index = -1
                if self._device.startswith("cuda") and torch.cuda.is_available():
                    index = int(self._device.split(":")[1]) if ":" in self._device else 0
                dtype = torch.float16 if index >= 0 else torch.float32
                logger.info("[QwenTTS] Transcribing the reference clip with %s...", asr_model)
                try:
                    self._asr_pipe = factory(
                        "automatic-speech-recognition", model=asr_model, device=index, dtype=dtype,
                    )
                except TypeError:
                    # transformers < 4.56 spells the argument torch_dtype.
                    self._asr_pipe = factory(
                        "automatic-speech-recognition", model=asr_model, device=index, torch_dtype=dtype,
                    )
            # No extra kwargs: the ASR pipeline forwards unknown ones to
            # Whisper's generate(), which rejects them and aborts the call.
            # The language is left to Whisper too: the reference may be in a
            # different language than the book.
            result = self._asr_pipe(self._asr_input(ref_path), chunk_length_s=30)
            text = (result.get("text") or "").strip() if isinstance(result, dict) else ""
            if text:
                logger.info(
                    "[QwenTTS] Auto-transcribed reference audio '%s': \"%s\"",
                    os.path.basename(ref_path), text,
                )
            else:
                logger.warning("[QwenTTS] Whisper returned no text for the reference voice.")
        except Exception as asr_err:
            logger.warning(
                "[QwenTTS] Could not auto-transcribe the reference voice (%s). "
                "Cloning will fall back to speaker-embedding-only mode, which is less "
                "faithful — provide the reference transcript to avoid this.",
                asr_err,
            )
        finally:
            self._release_asr()
        return text

    def _release_asr(self) -> None:
        """Drops the Whisper pipeline so its VRAM goes back to the TTS model."""
        if self._asr_pipe is None:
            return
        self._asr_pipe = None
        self._free_memory()

    # ── Voice clone prompt ───────────────────────────────────────────────────

    def _reference_for(self, voice_ref: str | bytes | None, explicit: bool = False) -> _Reference:
        """Picks the clip to clone: the designed voice or the reference file.

        Args:
            voice_ref: Reference passed by the caller (path or WAV bytes).
            explicit: True when the caller named *voice_ref* on purpose, which
                then wins over the designed voice.
        """
        if self._mode() == "design_clone" and not (explicit and voice_ref):
            spec = self._design_spec()
            path, text, _ = self._designed_reference(spec)
            return _Reference(
                path=path,
                identity=f"design:{spec.key}:{self._voice_ref_signature(path)}",
                text=text,
                origin={"source": "voice_design", "instruct": spec.instruct, "language": spec.language},
            )
        path = self._resolve_voice_ref(voice_ref or getattr(self.config, "voice_file", "") or None)
        if not path or not os.path.isfile(path):
            raise ValueError(
                f"Qwen3-TTS has no reference voice to clone (got '{path or ''}'). "
                "Set voice_file or voice_preset."
            )
        identity = self._voice_ref_digest(path) or self._voice_ref_signature(path)
        return _Reference(path=path, identity=identity, text=self._get_voice_transcript(path),
                          origin={"source": "clone"})

    def _build_prompt(self, reference: _Reference, ref_text: str | None) -> list[Any]:
        """Builds (once) the reusable clone prompt for a reference.

        The prompt bakes in the audio, its transcript and the mode, so the
        cache key covers all three. ICL mode is used whenever there is a
        transcript; without one only the speaker embedding is used.
        """
        self._ensure_state()
        ref_text = (ref_text or "").strip() or None
        if self.option("x_vector_only_mode", False):
            ref_text = None
        x_vector_only = ref_text is None
        key = hashlib.sha256(
            f"{reference.identity}|{'xvec' if x_vector_only else 'icl'}|{ref_text or ''}"
            .encode("utf-8", errors="replace")
        ).hexdigest()[:24]
        if key in self._voice_prompt_cache:
            return self._voice_prompt_cache[key]

        items = self._model.create_voice_clone_prompt(
            ref_audio=reference.path,
            ref_text=ref_text,
            x_vector_only_mode=x_vector_only,
        )
        items = list(items or [])[:1]
        if not items:
            raise RuntimeError(f"Qwen3-TTS returned no voice clone prompt for {reference.path}.")
        self._remember_prompt(key, items)

        ref_code = getattr(items[0], "ref_code", None)
        frames = int(ref_code.shape[0]) if ref_code is not None and hasattr(ref_code, "shape") else 0
        if frames / _CODEC_FRAME_RATE_HZ > _LONG_REFERENCE_SECONDS:
            logger.warning(
                "[QwenTTS] The reference clip is %.0f s long; every chunk is conditioned on all "
                "of it. 5-15 s is enough and is faster.", frames / _CODEC_FRAME_RATE_HZ,
            )
        logger.info(
            "[QwenTTS] Voice clone prompt built on %s (%s mode).",
            self._device, "x-vector-only" if x_vector_only else "ICL",
        )
        return items

    def _remember_prompt(self, key: str, items: list[Any]) -> None:
        while len(self._voice_prompt_cache) >= _MAX_PROMPT_CACHE:
            self._voice_prompt_cache.pop(next(iter(self._voice_prompt_cache)))
        self._voice_prompt_cache[key] = items

    def _clone_prompt(self, voice_ref: str | bytes | None) -> list[Any]:
        """The prompt for this run: a saved preset, else one built from the reference."""
        preset = self._preset_path()
        if preset:
            return self._preset_prompt(preset)
        reference = self._reference_for(voice_ref)
        return self._build_prompt(reference, reference.text)

    # ── Voice presets ────────────────────────────────────────────────────────

    def _require_base_model(self, purpose: str) -> None:
        if self._model is None:
            raise RuntimeError(f"Qwen3-TTS model is not loaded on {self._device}.")
        model_type = self._loaded_model_type()
        if model_type != "base":
            raise ValueError(
                f"Cannot {purpose} with {self._loaded_model_name} ({model_type}): voice presets "
                "are voice-clone prompts and need a Base checkpoint, or the VoiceDesign "
                "checkpoint with design_then_clone enabled."
            )

    def _model_facts(self) -> dict[str, Any]:
        """Properties of the loaded model that a preset must match."""
        inner = getattr(self._model, "model", None)
        config = getattr(inner, "config", None)
        talker = getattr(config, "talker_config", None)

        def _int(value: Any) -> int | None:
            return value if isinstance(value, int) and not isinstance(value, bool) and value > 0 else None

        sample_rate = None
        getter = getattr(getattr(inner, "speech_tokenizer", None), "get_output_sample_rate", None)
        if callable(getter):
            try:
                sample_rate = _int(int(getter()))
            except Exception:
                sample_rate = None
        return {
            "tokenizer_type": getattr(inner, "tokenizer_type", None),
            "tts_model_size": getattr(inner, "tts_model_size", None),
            "embedding_dim": _int(getattr(getattr(config, "speaker_encoder_config", None), "enc_dim", None))
                             or _int(getattr(talker, "hidden_size", None)),
            "num_code_groups": _int(getattr(talker, "num_code_groups", None)),
            "first_codebook": _int(getattr(talker, "vocab_size", None)),
            "other_codebooks": _int(getattr(getattr(talker, "code_predictor_config", None), "vocab_size", None)),
            "sample_rate": sample_rate or self.info().native_sample_rate,
        }

    def save_voice_preset(
        self,
        path: str,
        voice_ref: str | bytes | None = None,
        *,
        transcript: str | None = None,
    ) -> dict[str, Any]:
        """Saves the reusable voice-clone prompt as a preset file.

        The file holds the ``VoiceClonePromptItem`` fields (reference codes,
        speaker embedding, mode flags, reference text) as CPU tensors plus
        metadata, under an ``items`` key laid out like upstream's own demo
        saves it. It contains only tensors and plain values, so it loads with
        ``torch.load(weights_only=True)``.

        Parameters
        ----------
        path : str
            Destination file (conventionally ``*.pt``).
        voice_ref : str | bytes | None
            Reference clip (path or WAV bytes). Defaults to
            ``config.voice_file``; with a VoiceDesign checkpoint and no
            reference, the designed voice is saved.
        transcript : str | None
            Transcript of *voice_ref*; overrides ``config.voice_transcript``
            and auto-transcription.

        Returns
        -------
        dict[str, Any]
            Tensor-free description of the saved preset.

        Raises
        ------
        ValueError
            If the configured checkpoint cannot clone or there is no reference.
        """
        import torch

        self._ensure_state()
        self.bind_device()
        with self._lock:
            mode = self._mode()
            if mode not in ("clone", "design_clone"):
                raise ValueError(
                    f"Cannot save a voice preset with {self._configured_model_id()}: presets are "
                    "voice-clone prompts and need a Base checkpoint, or the VoiceDesign "
                    "checkpoint with design_then_clone enabled."
                )
            if mode == "design_clone" and not voice_ref:
                self._designed_reference()
            self._ensure_initialised()
            self._require_base_model("save a voice preset")
            reference = self._reference_for(voice_ref, explicit=True)
            items = self._build_prompt(reference, transcript if transcript is not None else reference.text)
            facts = self._model_facts()

            raw_items: list[dict[str, Any]] = []
            for item in items:
                ref_code = getattr(item, "ref_code", None)
                raw_items.append({
                    "ref_code": None if ref_code is None else ref_code.detach().to("cpu", torch.long),
                    "ref_spk_embedding": item.ref_spk_embedding.detach().to("cpu", torch.float32),
                    "x_vector_only_mode": bool(item.x_vector_only_mode),
                    "icl_mode": bool(item.icl_mode),
                    "ref_text": getattr(item, "ref_text", None),
                })
            payload: dict[str, Any] = {
                "format": _PRESET_FORMAT,
                "format_version": _PRESET_FORMAT_VERSION,
                "model_id": str(self._loaded_model_name or ""),
                "tts_model_size": str(facts["tts_model_size"] or ""),
                "tokenizer_type": str(facts["tokenizer_type"] or ""),
                "sample_rate": int(facts["sample_rate"]),
                "speaker_embedding_dim": int(raw_items[0]["ref_spk_embedding"].shape[-1]),
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "source": str(reference.origin.get("source", "clone")),
                "language": str(reference.origin.get("language", "")),
                "instruct": str(reference.origin.get("instruct", "")),
                "items": raw_items,
            }

        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, exist_ok=True)
        tmp_path = f"{path}.{os.getpid()}.tmp"
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)
        logger.info("[QwenTTS] Voice preset saved to %s.", path)
        meta = {key: payload[key] for key in _PRESET_META_KEYS if key in payload}
        return _preset_info(path, raw_items, meta)

    @staticmethod
    def read_voice_preset_info(path: str) -> dict[str, Any]:
        """Describes a preset file without loading a TTS model.

        Returns
        -------
        dict[str, Any]
            ``model_id``, ``tokenizer_type``, ``created_at``, ``mode``
            (``"icl"`` or ``"x_vector"``), ``ref_text`` and the other metadata.

        Raises
        ------
        ValueError
            If the file is not a valid preset.
        """
        items, meta = _read_preset_file(path)
        return _preset_info(path, items, meta)

    def load_voice_preset(self, path: str) -> dict[str, Any]:
        """Loads a preset onto this instance's device and checks it fits the model.

        Synthesis does this by itself when ``config.voice_preset`` is set;
        call it directly to validate a file the user picked.

        Returns
        -------
        dict[str, Any]
            Tensor-free description of the preset.

        Raises
        ------
        ValueError
            If the file is invalid or was made for an incompatible
            model or tokenizer.
        """
        self._ensure_state()
        self.bind_device()
        with self._lock:
            self._ensure_initialised()
            _, info = self._load_preset(path)
        return info

    def _preset_prompt(self, path: str) -> list[Any]:
        return self._load_preset(path)[0]

    def _load_preset(self, path: str) -> tuple[list[Any], dict[str, Any]]:
        self._require_base_model("use a voice preset")
        key = f"preset:{self._voice_ref_signature(path)}"
        cached = self._voice_prompt_cache.get(key)
        if cached is not None:
            return cached
        raw_items, meta = _read_preset_file(path)
        if len(raw_items) > 1:
            logger.warning("[QwenTTS] Voice preset %s holds %d voices; using the first.", path, len(raw_items))
        raw = raw_items[0]
        self._check_preset_compatible(path, raw, meta)
        item_class = _prompt_item_class()
        ref_code = raw["ref_code"]
        prompt = [item_class(
            ref_code=None if ref_code is None else ref_code.to(self._device),
            ref_spk_embedding=raw["ref_spk_embedding"].to(self._device),
            x_vector_only_mode=raw["x_vector_only_mode"],
            icl_mode=raw["icl_mode"],
            ref_text=raw["ref_text"],
        )]
        result = (prompt, _preset_info(path, raw_items[:1], meta))
        self._remember_prompt(key, result)  # type: ignore[arg-type]
        logger.info(
            "[QwenTTS] Voice preset %s loaded on %s (%s mode).",
            os.path.basename(path), self._device, result[1]["mode"],
        )
        return result

    def _check_preset_compatible(self, path: str, raw: dict[str, Any], meta: dict[str, Any]) -> None:
        """Rejects a preset whose tensors do not fit the loaded checkpoint."""
        facts = self._model_facts()
        origin = meta.get("model_id") or "an unknown model"
        loaded = self._loaded_model_name
        if not meta.get("format"):
            logger.warning(
                "[QwenTTS] %s has no AudiobookMaker metadata (upstream demo format); "
                "checking tensor shapes only.", path,
            )

        saved_tokenizer = meta.get("tokenizer_type")
        if saved_tokenizer and facts["tokenizer_type"] and saved_tokenizer != facts["tokenizer_type"]:
            raise ValueError(
                f"Voice preset {path} was made with tokenizer '{saved_tokenizer}' ({origin}) but "
                f"{loaded} uses '{facts['tokenizer_type']}'. Re-save the preset with this model."
            )
        dim = int(raw["ref_spk_embedding"].shape[-1])
        if facts["embedding_dim"] and dim != facts["embedding_dim"]:
            raise ValueError(
                f"Voice preset {path} was made for {origin} (speaker embedding size {dim}) but "
                f"{loaded} expects size {facts['embedding_dim']}. Select the checkpoint the "
                "preset was saved with, or re-save the preset."
            )
        saved_size = meta.get("tts_model_size")
        if saved_size and facts["tts_model_size"] and saved_size != facts["tts_model_size"]:
            raise ValueError(
                f"Voice preset {path} was made for a '{saved_size}' model ({origin}) but {loaded} "
                f"is '{facts['tts_model_size']}'. Select the matching checkpoint or re-save the preset."
            )
        ref_code = raw["ref_code"]
        if ref_code is None:
            return
        groups = facts["num_code_groups"]
        if ref_code.ndim == 2 and groups and int(ref_code.shape[1]) != groups:
            raise ValueError(
                f"Voice preset {path} has {int(ref_code.shape[1])} code groups per frame but "
                f"{loaded} uses {groups}. It was made with a different speech tokenizer."
            )
        # Codes index embedding tables; an out-of-range id would abort the CUDA context.
        first = ref_code[:, 0] if ref_code.ndim == 2 else ref_code
        if facts["first_codebook"] and int(first.max()) >= facts["first_codebook"]:
            raise ValueError(f"Voice preset {path} contains reference codes outside the model's codebook.")
        if ref_code.ndim == 2 and ref_code.shape[1] > 1 and facts["other_codebooks"] \
                and int(ref_code[:, 1:].max()) >= facts["other_codebooks"]:
            raise ValueError(f"Voice preset {path} contains reference codes outside the model's codebook.")

    # ── Voice design ─────────────────────────────────────────────────────────

    def _design_spec(
        self,
        instruct: str | None = None,
        text: str | None = None,
        language: str | None = None,
    ) -> _DesignSpec:
        """Resolves what the designed voice is made from; None reads the config."""
        configured = self._configured_model_id()
        model_id = configured if self._declared_type(configured) == "voice_design" else _MODEL_DESIGN_1B7
        resolved_language = _static_language(
            language if language is not None else getattr(self.config, "language", "")
        )
        resolved_text = (text if text is not None else str(self.option("design_text", "") or "")).strip()
        if not resolved_text:
            resolved_text = _CALIBRATION_TEXTS.get(resolved_language.lower(), _CALIBRATION_TEXTS["english"])
        raw_seed = getattr(self.config, "seed", -1)
        seed = int(raw_seed) if raw_seed is not None and int(raw_seed) >= 0 else -1
        resolved_instruct = (
            instruct if instruct is not None else getattr(self.config, "tts_instruct", "") or ""
        ).strip()
        return _DesignSpec(
            model_id=model_id, instruct=resolved_instruct, language=resolved_language,
            seed=seed, text=resolved_text,
        )

    def _designed_reference(self, spec: _DesignSpec | None = None, force: bool = False) -> tuple[str, str, int]:
        """Returns the designed voice clip, creating it if nobody has yet.

        The clip lives in the shared cache directory under a key derived from
        (instruct, language, seed, text, model). Creation happens under a
        file lock: the first instance designs, every other instance and
        process waits and then clones that same clip.

        Returns
        -------
        tuple[str, str, int]
            ``(wav_path, text_spoken, sample_rate)``.
        """
        self._ensure_state()
        # Lock order is always instance lock, then file lock.
        with self._lock:
            spec = spec or self._design_spec()
            memo = self._design_memo.get(spec.key)
            if memo is not None and not force and os.path.isfile(memo[0]):
                return memo

            directory = os.path.join(_cache_root(), "designed_voices")
            wav_path = os.path.join(directory, f"{spec.key}.wav")
            meta_path = os.path.join(directory, f"{spec.key}.json")
            with _file_lock(os.path.join(directory, f"{spec.key}.lock")):
                meta = None if force else self._read_design_meta(wav_path, meta_path, spec)
                if meta is None:
                    import soundfile as sf

                    if not spec.instruct:
                        logger.warning(
                            "[QwenTTS] Designing a voice without a description (tts_instruct is "
                            "empty); the model will pick an arbitrary voice."
                        )
                    logger.info(
                        "[QwenTTS] Designing the narrator voice on %s with %s (language=%s, seed=%d).",
                        self._device, spec.model_id, spec.language, spec.seed,
                    )
                    audio, sample_rate = self._run_voice_design(spec)
                    # The JSON marks the clip as complete: drop it before
                    # replacing the audio and rewrite it last.
                    with contextlib.suppress(OSError):
                        os.unlink(meta_path)
                    tmp_path = f"{wav_path}.{os.getpid()}.tmp"
                    sf.write(tmp_path, audio, sample_rate, format="WAV", subtype="PCM_16")
                    os.replace(tmp_path, wav_path)
                    meta = {
                        "version": _DESIGN_CACHE_VERSION,
                        "model_id": spec.model_id,
                        "instruct": spec.instruct,
                        "language": spec.language,
                        "seed": spec.seed,
                        "text": spec.text,
                        "sample_rate": int(sample_rate),
                        "duration": round(len(audio) / float(sample_rate), 3),
                        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    }
                    _atomic_write_bytes(meta_path, json.dumps(meta, ensure_ascii=False, indent=2).encode("utf-8"))
                    logger.info("[QwenTTS] Designed voice saved to %s (%.1f s).", wav_path, meta["duration"])
                else:
                    logger.info("[QwenTTS] Reusing the designed voice %s on %s.", wav_path, self._device)

            memo = (wav_path, spec.text, int(meta["sample_rate"]))
            if len(self._design_memo) >= 16:
                self._design_memo.clear()
            self._design_memo[spec.key] = memo
            return memo

    @staticmethod
    def _read_design_meta(wav_path: str, meta_path: str, spec: _DesignSpec) -> dict[str, Any] | None:
        """Metadata of a complete cached design, or None when it must be made."""
        meta = _read_json(meta_path)
        if meta is None or meta.get("text") != spec.text or meta.get("version") != _DESIGN_CACHE_VERSION:
            return None
        sample_rate = meta.get("sample_rate")
        if not isinstance(sample_rate, int) or isinstance(sample_rate, bool) or sample_rate <= 0:
            return None
        try:
            if os.path.getsize(wav_path) <= 44:
                return None
        except OSError:
            return None
        return meta

    def _run_voice_design(self, spec: _DesignSpec) -> tuple[Any, int]:
        """Generates the design clip, swapping the VoiceDesign checkpoint in and out."""
        import numpy as np

        with self._lock:
            self.bind_device()
            wanted = self._signature_for(spec.model_id)
            if self._model is None or self._loaded_signature != wanted:
                self._unload_model()
                self._load(spec.model_id)
            if self._model is None:
                raise RuntimeError(f"Could not load {spec.model_id} on {self._device}.")
            try:
                if self._loaded_model_type() != "voice_design":
                    raise ValueError(f"{spec.model_id} is not a VoiceDesign checkpoint.")
                plan = _GenerationPlan(
                    call=self._model.generate_voice_design,
                    per_item={"instruct": spec.instruct, "language": spec.language},
                    shared=self._shared_call_kwargs(),
                )
                wavs, sample_rate = self._generate_checked(plan, [spec.text])
            finally:
                # Make room for the checkpoint that reads the book.
                if self._loaded_signature != self._signature_for(self._target_model_id()):
                    self._unload_model()
        audio = self.to_mono_float32(wavs[0])
        if audio.size < sample_rate * _MIN_DESIGN_SECONDS or not np.isfinite(audio).all():
            raise RuntimeError(
                f"Qwen3-TTS VoiceDesign produced an unusable sample ({audio.size / float(sample_rate):.2f} s)."
            )
        return audio, sample_rate

    def design_voice(
        self,
        instruct: str | None = None,
        text: str | None = None,
        language: str | None = None,
        *,
        force: bool = False,
    ) -> tuple[bytes, int, str]:
        """Creates (or fetches) the designed narrator sample for auditioning.

        The sample is the exact clip a book run with the same instruct,
        language, seed and text will clone, so what the user hears is what
        the audiobook gets. Loads the VoiceDesign checkpoint on this
        instance's device for the duration of the call.

        Parameters
        ----------
        instruct : str | None
            Voice description; defaults to ``config.tts_instruct``.
        text : str | None
            Sentence to speak; defaults to option ``design_text``, then a
            built-in sentence in the target language.
        language : str | None
            Target language; defaults to ``config.language``.
        force : bool
            Regenerate even if this voice is cached (a "re-roll" when
            ``config.seed`` is random).

        Returns
        -------
        tuple[bytes, int, str]
            ``(wav_bytes, sample_rate, text_spoken)``. Pass the bytes and the
            text to :meth:`save_voice_preset` to keep the voice.
        """
        spec = self._design_spec(instruct, text, language)
        wav_path, spoken, sample_rate = self._designed_reference(spec, force=force)
        with open(wav_path, "rb") as fh:
            return fh.read(), sample_rate, spoken

    # ── Generation ───────────────────────────────────────────────────────────

    def _generation_kwargs(self, max_new_tokens: int) -> dict[str, Any]:
        """Sampling arguments for a ``generate_*`` call.

        Covers all ten arguments of upstream's ``_merge_generate_kwargs`` and
        nothing else. A value that is unset or meaningless (``top_k=0``,
        temperature with sampling off) is left out so the checkpoint's own
        ``generation_config.json`` default applies.
        """
        config = self.config
        kwargs: dict[str, Any] = {"max_new_tokens": int(max_new_tokens)}

        temperature = float(getattr(config, "temperature", 0.0) or 0.0)
        do_sample = bool(self.option("do_sample", True)) and temperature > 0.0
        kwargs["do_sample"] = do_sample
        if do_sample:
            kwargs["temperature"] = temperature
            top_p = float(getattr(config, "top_p", 0.0) or 0.0)
            if 0.0 < top_p <= 1.0:
                kwargs["top_p"] = top_p
            top_k = int(getattr(config, "top_k", 0) or 0)
            if top_k > 0:
                kwargs["top_k"] = top_k
        repetition_penalty = float(getattr(config, "repetition_penalty", 0.0) or 0.0)
        if repetition_penalty > 0.0:
            kwargs["repetition_penalty"] = repetition_penalty

        sub_temperature = float(self.option("subtalker_temperature", 0.9) or 0.0)
        sub_sample = bool(self.option("subtalker_dosample", True)) and sub_temperature > 0.0
        kwargs["subtalker_dosample"] = sub_sample
        if sub_sample:
            kwargs["subtalker_temperature"] = sub_temperature
            sub_top_p = float(self.option("subtalker_top_p", 1.0) or 0.0)
            if 0.0 < sub_top_p <= 1.0:
                kwargs["subtalker_top_p"] = sub_top_p
            sub_top_k = int(self.option("subtalker_top_k", 50) or 0)
            if sub_top_k > 0:
                kwargs["subtalker_top_k"] = sub_top_k
        return {name: value for name, value in kwargs.items() if name in _UPSTREAM_GENERATE_KWARGS}

    def _shared_call_kwargs(self) -> dict[str, Any]:
        """Explicit ``non_streaming_mode`` when the user overrode the default."""
        choice = str(self.option("non_streaming_mode", "auto") or "auto").strip().lower()
        if choice in ("on", "true", "1", "yes"):
            return {"non_streaming_mode": True}
        if choice in ("off", "false", "0", "no"):
            return {"non_streaming_mode": False}
        return {}

    def _batch_budget(self, texts: list[str]) -> int:
        """Token budget for a batch; option ``max_new_tokens`` overrides the estimate."""
        override = int(self.option("max_new_tokens", 0) or 0)
        if override > 0:
            return min(override, _MAX_NEW_TOKENS_CAP)
        return _token_budget(texts)

    def _seed(self, offset: int = 0) -> None:
        """Seeds this instance's RNG when ``config.seed >= 0``.

        On CUDA only this device's generator is seeded. ``seed_everything``
        reseeds every GPU, which would disturb a generation running on the
        other GPU in another thread.
        """
        seed = getattr(self.config, "seed", -1)
        if seed is None or int(seed) < 0:
            return
        value = int(seed) + offset
        import torch
        if self._device.startswith("cuda") and torch.cuda.is_available():
            index = int(self._device.split(":")[1]) if ":" in self._device else torch.cuda.current_device()
            with torch.cuda.device(index):
                torch.cuda.manual_seed(value)
        elif offset == 0:
            self.seed_everything()
        else:
            torch.manual_seed(value)

    def _samples_per_frame(self, sample_rate: int) -> float:
        tokenizer = getattr(getattr(self._model, "model", None), "speech_tokenizer", None)
        getter = getattr(tokenizer, "get_decode_upsample_rate", None)
        if callable(getter):
            try:
                value = float(getter())
                if value > 0:
                    return value
            except Exception:
                pass
        return sample_rate / _CODEC_FRAME_RATE_HZ

    def _call(
        self,
        plan: _GenerationPlan,
        texts: list[str],
        max_new_tokens: int,
        seed_offset: int = 0,
    ) -> tuple[list[Any], int]:
        """Runs one upstream forward pass and returns mono float32 waveforms."""
        self._seed(seed_offset)
        wavs, sample_rate = plan.call(
            text=list(texts),
            **plan.kwargs_for(len(texts)),
            **self._generation_kwargs(max_new_tokens),
        )
        if len(wavs) != len(texts):
            raise RuntimeError(
                f"Qwen3-TTS returned {len(wavs)} waveforms for {len(texts)} texts on {self._device}."
            )
        return [self.to_mono_float32(wav) for wav in wavs], int(sample_rate)

    def _generate_checked(self, plan: _GenerationPlan, texts: list[str]) -> tuple[list[Any], int]:
        """Generates a batch under a token budget and catches runaways.

        A row that fills its whole budget never emitted end-of-speech. It is
        regenerated alone with twice the budget (which also rescues a text
        that really was that long); if it fills that too, the chunk fails
        rather than returning babble.
        """
        budget = self._batch_budget(texts)
        wavs, sample_rate = self._call(plan, texts, budget)
        samples_per_frame = self._samples_per_frame(sample_rate)
        for index, wav in enumerate(wavs):
            if len(wav) / samples_per_frame < budget - _BUDGET_HIT_TOLERANCE:
                continue
            retry_budget = min(_MAX_NEW_TOKENS_CAP, budget * 2)
            logger.warning(
                "[QwenTTS] Chunk of %d chars hit its %d-token budget on %s; retrying alone with %d.",
                len(texts[index]), budget, self._device, retry_budget,
            )
            retried, _ = self._call(plan, [texts[index]], retry_budget, seed_offset=1)
            if len(retried[0]) / samples_per_frame >= retry_budget - _BUDGET_HIT_TOLERANCE:
                raise _RunawayGenerationError(
                    f"Qwen3-TTS did not finish a {len(texts[index])}-char chunk within "
                    f"{retry_budget} codec tokens ({retry_budget / _CODEC_FRAME_RATE_HZ:.0f} s) on "
                    f"{self._device}: runaway generation. Text starts: {texts[index][:60]!r}"
                )
            wavs[index] = retried[0]
        return wavs, sample_rate

    def _generate_resilient(self, plan: _GenerationPlan, texts: list[str]) -> tuple[list[Any], int]:
        """Generates a batch, shrinking it on OOM and isolating bad items.

        Out of memory halves the batch and retries; the model stays loaded.
        Any other failure of a multi-item batch retries the items one by one,
        and a single item gets one more attempt before the error is raised.
        """
        failure: BaseException | None = None
        try:
            return self._generate_checked(plan, texts)
        except _NON_RETRYABLE:
            raise
        except Exception as exc:
            logger.debug("[QwenTTS] Generation failed on %s", self._device, exc_info=True)
            # Drop the traceback: its frames pin the tensors of the failed pass.
            failure = exc.with_traceback(None)
        out_of_memory = _is_out_of_memory(failure)
        self._free_memory()

        if out_of_memory and len(texts) == 1:
            raise RuntimeError(
                f"CUDA out of memory on {self._device} for a single chunk of {len(texts[0])} chars. "
                "Lower max_len or worker_count, or use the 0.6B checkpoint."
            ) from failure
        if len(texts) > 1:
            if out_of_memory:
                middle = (len(texts) + 1) // 2
                logger.warning(
                    "[QwenTTS] CUDA out of memory on %s for a batch of %d; retrying as %d + %d.",
                    self._device, len(texts), middle, len(texts) - middle,
                )
                parts = [texts[:middle], texts[middle:]]
            else:
                logger.warning(
                    "[QwenTTS] Batch synthesis failed on %s (%d items): %s; retrying per item.",
                    self._device, len(texts), failure,
                )
                parts = [[text] for text in texts]
            wavs: list[Any] = []
            sample_rate = 0
            for part in parts:
                part_wavs, sample_rate = self._generate_resilient(plan, part)
                wavs.extend(part_wavs)
            return wavs, sample_rate

        logger.warning("[QwenTTS] Synthesis failed on %s (%s); retrying once.", self._device, failure)
        try:
            return self._generate_checked(plan, texts)
        except _NON_RETRYABLE:
            raise
        except Exception as exc:
            if _is_out_of_memory(exc):
                self._free_memory()
            raise RuntimeError(f"Qwen3-TTS synthesis failed on {self._device}: {exc}") from exc

    def _generate(self, texts: list[str], voice_ref: str | bytes | None) -> tuple[list[Any], int]:
        """Routes a batch to the ``generate_*`` call of the loaded checkpoint."""
        if self._model is None or getattr(self._model, "model", None) is None:
            raise RuntimeError(f"QwenTTS model instance is not properly loaded on {self._device}.")
        model_type = self._loaded_model_type()
        language = self._resolve_language()
        instruct = (getattr(self.config, "tts_instruct", "") or "").strip()
        shared = self._shared_call_kwargs()

        if model_type == "base":
            if instruct and self._mode() == "clone":
                self._warn_once(
                    "instruct-base",
                    "[QwenTTS] tts_instruct is ignored by Base checkpoints; use CustomVoice (1.7B) "
                    "or VoiceDesign for instruction control.",
                )
            shared["voice_clone_prompt"] = self._clone_prompt(voice_ref)
            plan = _GenerationPlan(self._model.generate_voice_clone, {"language": language}, shared)
        elif model_type == "custom_voice":
            per_item: dict[str, Any] = {"speaker": self._resolve_speaker(language), "language": language}
            if instruct:
                # Upstream discards instruct for the 0.6B CustomVoice model.
                if str(getattr(self._model.model, "tts_model_size", "") or "") in "0b6":
                    self._warn_once(
                        "instruct-0b6",
                        "[QwenTTS] %s does not support instruction control; tts_instruct is ignored. "
                        "Use the 1.7B CustomVoice checkpoint for it.", self._loaded_model_name,
                    )
                else:
                    per_item["instruct"] = instruct
            if (getattr(self.config, "voice_preset", "") or "").strip():
                self._warn_once(
                    "preset-custom",
                    "[QwenTTS] voice_preset is ignored by CustomVoice checkpoints (speaker: %s).",
                    per_item["speaker"],
                )
            plan = _GenerationPlan(self._model.generate_custom_voice, per_item, shared)
        elif model_type == "voice_design":
            self._warn_once(
                "design-per-chunk",
                "[QwenTTS] VoiceDesign is generating every chunk independently "
                "(design_then_clone is off): the narrator's voice will vary between chunks.",
            )
            plan = _GenerationPlan(
                self._model.generate_voice_design, {"instruct": instruct, "language": language}, shared,
            )
        else:
            raise ValueError(f"Unsupported Qwen3-TTS model type '{model_type}' ({self._loaded_model_name}).")
        return self._generate_resilient(plan, texts)

    def synthesize(
        self,
        text: str,
        voice_ref: str | bytes,
        out_path: str | None = None,
        *,
        return_bytes: bool = False,
    ) -> tuple[str | bytes, float]:
        """Synthesize speech for input text and write output WAV file or return WAV bytes."""
        self._ensure_state()
        self.bind_device()
        self._check_voice_source(voice_ref)
        with self._lock:
            self._ensure_initialised()
            wavs, sample_rate = self._generate([text], voice_ref)
        return self.finish(wavs[0], sample_rate, out_path, return_bytes or out_path is None)

    def synthesize_batch(
        self,
        texts: list[str],
        voice_ref: bytes,
        *,
        return_bytes: bool = True,
    ) -> list[tuple[bytes | str, float]]:
        """Single-GPU batched synthesis in one Qwen3-TTS forward pass.

        Holds ``self._lock`` for the forward pass and releases it before WAV
        encoding. On CUDA out-of-memory the batch is halved and retried; other
        failures fall back to per-item synthesis.

        Thread-safe via ``self._lock``. Caller must hold exclusive ownership of
        this provider — do not call from two threads simultaneously.
        """
        if not texts:
            return []
        self._ensure_state()
        self.bind_device()
        self._check_voice_source(voice_ref)
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

        with self._lock:
            self._ensure_initialised()
            wavs, sample_rate = self._generate(list(texts), voice_ref)

        output: list[tuple[bytes | str, float]] = []
        for index, wav in enumerate(wavs):
            try:
                output.append(self.finish(wav, sample_rate, None, True))
            except RuntimeError as exc:
                raise RuntimeError(
                    f"{exc} (batch chunk {index}; check voice reference and model state)"
                ) from exc
        return output

    # ── Model lifecycle ──────────────────────────────────────────────────────

    def _free_memory(self) -> None:
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as exc:
            logger.debug("Could not empty the CUDA cache: %s", exc)

    def _unload_model(self) -> None:
        """Drops the TTS model and everything computed with it."""
        self._ensure_state()
        had_model = self._model is not None
        self._model = None
        self._loaded_model_name = None
        self._loaded_quantization = None
        self._loaded_signature = None
        self._voice_prompt_cache.clear()
        if had_model:
            self._free_memory()

    def cleanup(self) -> None:
        """Frees the TTS model, the ASR pipeline and every per-instance cache."""
        self._ensure_state()
        with self._lock:
            self.bind_device()
            self._release_asr()
            self._unload_model()
            self._transcript_cache.clear()
            self._digest_cache.clear()
            self._design_memo.clear()
            self._warned.clear()

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

    def _signature_for(self, model_id: str) -> tuple[Any, ...]:
        """Everything that, when changed, requires reloading the weights."""
        quantization = getattr(self.config, "quantization", "none")
        return (
            model_id,
            quantization,
            self._dtype_override,
            bool(getattr(self.config, "torch_compile", False)) and quantization != "int8",
        )

    def _load(self, model_id: str) -> None:
        self._load_model(model_id)
        if self._model is not None:
            self._loaded_signature = self._signature_for(model_id)

    def _ensure_initialised(self) -> None:
        """Ensure the checkpoint this run needs is loaded on ``self._device``.

        Reloads when the model id, quantization, dtype override or
        ``torch_compile`` changed since the last load. For a VoiceDesign
        checkpoint in design-then-clone mode the designed voice is created
        first (loading and unloading the VoiceDesign checkpoint if no other
        instance has made it yet) and the Base checkpoint is what stays loaded.
        """
        self._ensure_state()
        with self._lock:
            if self._mode() == "design_clone" and not self._preset_path():
                self._designed_reference()
            target = self._target_model_id()
            if self._model is not None and self._loaded_signature == self._signature_for(target):
                return
            if self._model is not None:
                logger.info("[QwenTTS] Configuration changed; reloading the model on %s.", self._device)
                self._unload_model()
            self._load(target)

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
                import bitsandbytes  # type: ignore # noqa: F401
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
            "dtype": dtype,
        }

    def _attention_implementation(self, load_kwargs: dict[str, Any]) -> str:
        """FlashAttention 2 when installed and usable, else SDPA.

        Upstream: FlashAttention 2 only works with a model loaded in float16
        or bfloat16, and it needs a CUDA device.
        """
        import torch
        # A bitsandbytes INT8 load keeps its activations in half precision.
        half_precision = (
            load_kwargs.get("dtype") in (torch.float16, torch.bfloat16)
            or "quantization_config" in load_kwargs
        )
        if not (self._device.startswith("cuda") and half_precision):
            return "sdpa"
        try:
            import flash_attn  # type: ignore # noqa: F401
        except ImportError:
            logger.info("[QwenTTS] flash_attn not found; using SDPA attention.")
            return "sdpa"
        logger.info("[QwenTTS] flash_attn detected; using FlashAttention 2.")
        return "flash_attention_2"

    def _load_model(self, model_id: str | None = None) -> None:
        """Load a Qwen3TTSModel on the assigned target device.

        Args:
            model_id: Checkpoint to load; defaults to the one this run needs.

        Note: torch.compile is applied at instance-level to self._model.model
        so that each device instance maintains its own compiled PyTorch graph.
        """
        self.bind_device()
        import torch

        model_id = model_id or self._target_model_id()
        qwen_model_class = _import_qwen_tts()
        logger.info("[QwenTTS] Loading model on %s: %s...", self._device, model_id)

        load_kwargs = self._build_model_load_kwargs(self.config)
        model = qwen_model_class.from_pretrained(
            model_id,
            attn_implementation=self._attention_implementation(load_kwargs),
            **load_kwargs,
        )
        self._model = model
        self._loaded_model_name = model_id
        self._loaded_quantization = getattr(self.config, "quantization", "none")

        _sanitize_dict_keys(model)

        inner = getattr(model, "model", None)
        gen_cfg = getattr(inner, "generation_config", None)
        if gen_cfg is not None and getattr(gen_cfg, "pad_token_id", None) is None:
            gen_cfg.pad_token_id = getattr(gen_cfg, "eos_token_id", None)
        # Sampling runs in the talker, with eos_token_id=codec_eos_token_id. With
        # no pad token transformers falls back to that id and logs a warning on
        # every batch; setting the same id up front keeps the result and drops the log.
        talker_cfg = getattr(getattr(inner, "talker", None), "generation_config", None)
        codec_eos = getattr(
            getattr(getattr(inner, "config", None), "talker_config", None), "codec_eos_token_id", None,
        )
        if (talker_cfg is not None and getattr(talker_cfg, "pad_token_id", None) is None
                and isinstance(codec_eos, int) and not isinstance(codec_eos, bool)):
            talker_cfg.pad_token_id = codec_eos

        if getattr(self.config, "torch_compile", False) and inner is not None:
            if getattr(self.config, "quantization", "none") == "int8":
                logger.warning(
                    "torch_compile=True is ignored when quantization='int8'. "
                    "bitsandbytes INT8 kernels are incompatible with torch.compile()."
                )
            else:
                try:
                    logger.info(
                        "[QwenTTS] Compiling underlying transformer graphs (mode=%s)...",
                        _TORCH_COMPILE_MODE,
                    )
                    model.model = torch.compile(inner, mode=_TORCH_COMPILE_MODE, fullgraph=False)
                    logger.info(
                        "torch.compile(mode='max-autotune') applied on %s. "
                        "First chapter will incur ~10–30s autotuning overhead. "
                        "All subsequent chapters will be 15–25%% faster.",
                        self._device,
                    )
                except Exception as exc:
                    logger.warning("[QwenTTS] torch.compile not supported or failed: %s", exc)

        logger.info("[QwenTTS] %s ready on %s.", model_id, self._device)
