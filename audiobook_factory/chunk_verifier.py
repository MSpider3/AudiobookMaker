"""
audiobook_factory/chunk_verifier.py
====================================
Checks that a synthesized chunk actually says its text.

Neural TTS occasionally truncates a chunk, runs on past the end of the text,
or returns near-silence. Over a ten-hour book that is dozens of audible
defects, so each chunk is checked before it is accepted:

``"duration"``
    Free. Compares the audio length with what the text should take to speak
    and rejects silence. Catches truncation, runaway generation and dead
    chunks, which are the common failures.
``"asr"``
    Also transcribes the chunk with Whisper and compares the transcript with
    the text. Catches skipped or substituted words, at the cost of an ASR pass
    (a few percent of total run time on a GPU) and roughly 2 GB of VRAM.
``"off"``
    No checks.

A chunk that fails is re-synthesized by the caller; the verifier only judges.
"""
from __future__ import annotations

import io
import logging
import re
import threading
import unicodedata
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

VERIFY_MODES: tuple[str, ...] = ("off", "duration", "asr")

_LATIN_CHARS_PER_SECOND: float = 15.0
# Narration runs at roughly 150 words a minute ≈ 15 characters a second.
_CJK_CHARS_PER_SECOND: float = 4.5
# Han / kana / hangul carry a syllable or more each.

_MIN_RATIO: float = 0.4
# Audio shorter than this fraction of the expected length is truncated.
_MIN_RATIO_APPLIES_ABOVE_SEC: float = 1.5
# Very short texts are dominated by the model's own padding; don't judge them.
_MAX_RATIO: float = 3.0
_MAX_SLACK_SEC: float = 3.0
# Audio longer than expected * _MAX_RATIO + _MAX_SLACK_SEC is runaway generation.
_SILENCE_RMS: float = 1e-3
# About -60 dBFS: below this the chunk is effectively silent.
_ASR_SAMPLE_RATE: int = 16000

_CJK_PATTERN = re.compile(
    "[぀-ヿ㐀-䶿一-鿿가-힯豈-﫿]"
)
_NON_WORD = re.compile(r"[^\w\s]", re.UNICODE)

_WHISPER_LANGUAGE_CODES: dict[str, str] = {
    "english": "en", "chinese": "zh", "japanese": "ja", "korean": "ko",
    "german": "de", "french": "fr", "russian": "ru", "portuguese": "pt",
    "spanish": "es", "italian": "it", "arabic": "ar", "hindi": "hi",
    "dutch": "nl", "polish": "pl", "turkish": "tr", "vietnamese": "vi",
}


@dataclass(frozen=True)
class ChunkVerdict:
    """Result of checking one chunk.

    Attributes
    ----------
    ok : bool
        True when the chunk is accepted.
    reason : str
        Empty when ok; otherwise a short description of what is wrong.
    score : float
        Badness, 0.0 = perfect. Used to keep the best of several attempts.
    """

    ok: bool
    reason: str = ""
    score: float = 0.0


def expected_seconds(text: str, speed: float = 1.0) -> float:
    """Estimates how long *text* takes to narrate, in seconds."""
    cjk = len(_CJK_PATTERN.findall(text))
    other = sum(1 for ch in text if not ch.isspace()) - cjk
    seconds = cjk / _CJK_CHARS_PER_SECOND + max(0, other) / (_LATIN_CHARS_PER_SECOND * 0.85)
    return seconds / max(0.25, float(speed or 1.0))


def _normalize_for_compare(text: str) -> list[str]:
    """Lower-cases, strips punctuation and tokenizes for transcript comparison.

    Space-delimited scripts are compared word by word; CJK text has no word
    spacing, so each character is a token.
    """
    text = unicodedata.normalize("NFKC", text).lower()
    text = _NON_WORD.sub(" ", text).replace("_", " ")
    tokens: list[str] = []
    for word in text.split():
        if _CJK_PATTERN.search(word):
            tokens.extend(ch for ch in word if not ch.isspace())
        else:
            tokens.append(word)
    return tokens


def error_rate(reference: str, hypothesis: str) -> float:
    """Token error rate between two texts (word level, character level for CJK).

    Returns
    -------
    float
        Edit distance divided by the reference length; 0.0 for identical
        texts, and 1.0 or more when little of the reference was spoken.
    """
    ref = _normalize_for_compare(reference)
    hyp = _normalize_for_compare(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    previous = list(range(len(hyp) + 1))
    for i, ref_token in enumerate(ref, 1):
        current = [i]
        for j, hyp_token in enumerate(hyp, 1):
            cost = 0 if ref_token == hyp_token else 1
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost))
        previous = current
    return previous[-1] / len(ref)


class ChunkVerifier:
    """Judges synthesized chunks. One instance is shared by all GPU workers.

    Parameters
    ----------
    mode : str
        One of :data:`VERIFY_MODES`.
    language : str
        Book language name (``config.language``), used as the ASR language hint.
    speed : float
        Speaking speed the TTS model was asked for (1.0 = normal).
    asr_model : str
        Whisper model id for ``mode="asr"``.
    max_error_rate : float
        Transcript error rate above which a chunk is rejected.
    """

    def __init__(
        self,
        mode: str = "duration",
        language: str = "English",
        speed: float = 1.0,
        asr_model: str = "openai/whisper-large-v3-turbo",
        max_error_rate: float = 0.3,
    ) -> None:
        mode = (mode or "off").lower().strip()
        self.mode: str = mode if mode in VERIFY_MODES else "duration"
        self.language: str = language or "English"
        self.speed: float = float(speed or 1.0)
        self.asr_model: str = asr_model
        self.max_error_rate: float = float(max_error_rate)
        self._asr: Any = None
        self._asr_backend: str = ""
        self._asr_failed: bool = False
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        """False when no check would run."""
        return self.mode != "off"

    # ── Duration / silence ───────────────────────────────────────────────────

    def _check_signal(self, text: str, wav_bytes: bytes, duration: float) -> tuple[ChunkVerdict, Any, int]:
        """Runs the free checks and returns (verdict, samples, sample_rate)."""
        import numpy as np
        import soundfile as sf

        try:
            samples, rate = sf.read(io.BytesIO(wav_bytes), dtype="float32")
        except Exception as exc:
            return ChunkVerdict(False, f"unreadable audio ({exc})", 10.0), None, 0
        if samples.ndim > 1:
            samples = samples.mean(axis=1)
        if samples.size == 0 or not np.isfinite(samples).all():
            return ChunkVerdict(False, "empty or non-finite audio", 10.0), samples, rate

        rms = float(np.sqrt(np.mean(np.square(samples))))
        if rms < _SILENCE_RMS:
            return ChunkVerdict(False, f"silent (rms {rms:.5f})", 10.0), samples, rate

        actual = float(duration) if duration and duration > 0 else samples.size / float(rate)
        expected = expected_seconds(text, self.speed)
        if expected > _MIN_RATIO_APPLIES_ABOVE_SEC and actual < expected * _MIN_RATIO:
            return (
                ChunkVerdict(False, f"too short ({actual:.1f}s for ~{expected:.1f}s of text)",
                             expected / max(actual, 0.05)),
                samples, rate,
            )
        if actual > expected * _MAX_RATIO + _MAX_SLACK_SEC:
            return (
                ChunkVerdict(False, f"too long ({actual:.1f}s for ~{expected:.1f}s of text)",
                             actual / max(expected, 0.05)),
                samples, rate,
            )
        return ChunkVerdict(True), samples, rate

    # ── ASR ──────────────────────────────────────────────────────────────────

    def _load_asr(self) -> None:
        """Loads Whisper once, preferring faster-whisper, on the freest GPU."""
        device, compute_type = "cpu", "int8"
        try:
            import torch
            if torch.cuda.is_available():
                free = [torch.cuda.mem_get_info(i)[0] for i in range(torch.cuda.device_count())]
                index = max(range(len(free)), key=free.__getitem__)
                device, compute_type = f"cuda:{index}", "float16"
        except Exception as exc:
            logger.debug("Could not probe CUDA for the ASR verifier: %s", exc)

        try:
            from faster_whisper import WhisperModel  # type: ignore

            name = self.asr_model.split("/")[-1].replace("whisper-", "")
            kwargs: dict[str, Any] = {"compute_type": compute_type}
            if device.startswith("cuda"):
                kwargs.update(device="cuda", device_index=int(device.split(":")[1]))
            else:
                kwargs.update(device="cpu")
            self._asr = WhisperModel(name, **kwargs)
            self._asr_backend = "faster-whisper"
        except Exception as exc:
            # Not installed, or its CUDA/cuDNN build does not match this
            # machine: the transformers implementation needs nothing extra.
            if not isinstance(exc, ImportError):
                logger.info("[verify] faster-whisper unavailable (%s); using transformers Whisper.", exc)
            import torch
            from transformers import pipeline  # type: ignore

            on_gpu = device.startswith("cuda")
            self._asr = pipeline(
                "automatic-speech-recognition",
                model=self.asr_model,
                device=int(device.split(":")[1]) if on_gpu else -1,
                dtype=torch.float16 if on_gpu else torch.float32,
            )
            self._asr_backend = "transformers"
        logger.info("[verify] ASR %s (%s) loaded on %s.", self.asr_model, self._asr_backend, device)

    def _transcribe(self, samples: Any, rate: int) -> str:
        """Transcribes one chunk. Returns "" when ASR is unavailable."""
        import numpy as np

        if rate != _ASR_SAMPLE_RATE:
            try:
                from math import gcd
                from scipy.signal import resample_poly  # type: ignore
                divisor = gcd(int(rate), _ASR_SAMPLE_RATE)
                samples = resample_poly(samples, _ASR_SAMPLE_RATE // divisor, int(rate) // divisor)
            except ImportError:
                positions = np.linspace(0, len(samples) - 1, int(len(samples) * _ASR_SAMPLE_RATE / rate))
                samples = np.interp(positions, np.arange(len(samples)), samples)
        samples = np.ascontiguousarray(samples, dtype=np.float32)
        language = _WHISPER_LANGUAGE_CODES.get(self.language.lower().strip())

        with self._lock:
            if self._asr is None and not self._asr_failed:
                try:
                    self._load_asr()
                except Exception as exc:
                    self._asr_failed = True
                    logger.warning(
                        "[verify] Could not load ASR model %s (%s). "
                        "Falling back to duration-only verification.", self.asr_model, exc,
                    )
            if self._asr is None:
                return ""
            if self._asr_backend == "faster-whisper":
                segments, _ = self._asr.transcribe(samples, language=language, beam_size=1)
                return " ".join(segment.text for segment in segments).strip()
            generate_kwargs = {"language": language} if language else {}
            result = self._asr(
                {"array": samples, "sampling_rate": _ASR_SAMPLE_RATE},
                chunk_length_s=30,
                generate_kwargs=generate_kwargs,
            )
            return (result.get("text") or "").strip() if isinstance(result, dict) else ""

    # ── Public API ───────────────────────────────────────────────────────────

    def check(self, text: str, wav_bytes: bytes, duration: float) -> ChunkVerdict:
        """Judges one synthesized chunk.

        Never raises: a verifier failure accepts the chunk rather than
        failing the chapter.
        """
        if not self.enabled:
            return ChunkVerdict(True)
        try:
            verdict, samples, rate = self._check_signal(text, wav_bytes, duration)
            if not verdict.ok or self.mode != "asr" or samples is None:
                return verdict
            transcript = self._transcribe(samples, rate)
            if not transcript and self._asr is None:
                return verdict
            rate_of_error = error_rate(text, transcript)
            if rate_of_error > self.max_error_rate:
                return ChunkVerdict(
                    False,
                    f"transcript mismatch ({rate_of_error:.0%} of words differ)",
                    rate_of_error,
                )
            return ChunkVerdict(True, "", rate_of_error)
        except Exception as exc:
            logger.warning("[verify] Chunk check failed (%s); accepting the chunk.", exc)
            return ChunkVerdict(True)

    def close(self) -> None:
        """Releases the ASR model."""
        with self._lock:
            self._asr = None
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
