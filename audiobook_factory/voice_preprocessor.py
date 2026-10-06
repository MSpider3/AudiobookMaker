"""
audiobook_factory/voice_preprocessor.py
========================================
Cleaning pipeline for the narrator's reference clip before zero-shot cloning.

The clip decides how the whole book sounds: a cloning model copies whatever
it hears, including clipped word onsets, gated consonants, level jumps and
edits inside a sentence. Every step here is therefore written to change the
voice as little as possible, and anything that could not run is reported
instead of being silently skipped.

Processing order (each step individually toggleable through
``PreprocessConfig``; the numbers in brackets are the historical step numbers
still shown by the UI):

  1. Decode, repair NaN/inf, fold to mono, remove DC offset
  2. High-pass filter [3]     - zero-phase Butterworth, removes rumble
  3. Noise reduction [1]      - spectral gate with a noise profile learned
                                from the clip's own pauses
  4. Noise gate [2]           - downward expander, threshold relative to the
                                clip's peak, attack / hold / release smoothing
  5. Best-window selection    - optional; pick the densest N seconds
  6. Silence handling [4]     - trim leading / trailing silence (default) and
                                optionally shorten long internal pauses
  7. Resample [7]             - down to the cloning rate, never up by default
  8. Level [5]                - integrated-loudness normalisation with a
                                true-peak ceiling, always last

Formant shifting [6] is accepted in the config for compatibility but is no
longer applied: it rewrote the timbre the clone is supposed to copy.

Public API
----------
``preprocess``              -> WAV bytes (unchanged contract)
``preprocess_with_report``  -> ``(WAV bytes, VoiceReport)``
``analyze_voice``           -> ``VoiceReport`` for any clip, no processing
``clear_cache``             -> remove cached results
"""
from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import logging
import math
import os
import tempfile
import threading
import time
import warnings
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

_LogFn = Callable[[str], None]

# ── Cache ─────────────────────────────────────────────────────────────────────
_CACHE_DIR_NAME: str = ".voice_cache"
_CACHE_DIR_ENV: str = "ABM_VOICE_CACHE_DIR"
_CACHE_ENTRY_SUFFIX: str = "_preprocessed.wav"
_CACHE_TMP_SUFFIX: str = ".tmp"
_CACHE_HASH_LENGTH: int = 16
_CACHE_MAX_ENTRIES: int = 64
_CACHE_MAX_BYTES: int = 256 * 1024 * 1024
_CACHE_STALE_TMP_SECONDS: float = 3600.0
# Bump whenever the DSP changes so results of an older pipeline are never served.
_PIPELINE_VERSION: int = 2
_CACHE_LOCK: threading.Lock = threading.Lock()

# ── Input limits ──────────────────────────────────────────────────────────────
# A reference clip is seconds long. Ten minutes leaves room for best-window
# selection on a long recording while keeping a mistaken upload (a whole
# audiobook) from exhausting memory.
_MAX_INPUT_SECONDS: float = 600.0

# ── Level analysis ────────────────────────────────────────────────────────────
_FRAME_MS: float = 10.0
_DEFAULT_SILENCE_THRESHOLD_DB: float = -40.0
_DEFAULT_MIN_SEGMENT_MS: int = 100
_SILENCE_DB: float = -120.0
_MAX_SNR_DB: float = 99.0
_ROBUST_PEAK_PERCENTILE: float = 99.0
_NOISE_FLOOR_FRACTION: float = 0.05
_NOISE_MARGIN_DB: float = 8.0
_MAX_ADAPTIVE_THRESHOLD_DB: float = -25.0
_MERGE_GAP_MS: float = 200.0
_CLIP_RELATIVE_TOLERANCE: float = 2.0 ** -14
_CLIP_ABSOLUTE_TOLERANCE: float = 1.0 / 32768.0
_CLIP_MIN_RAIL_RATIO: float = 0.25
_CLIP_MIN_RUN: int = 3

# ── High-pass filter ──────────────────────────────────────────────────────────
_HIGHPASS_ORDER: int = 5
_HIGHPASS_PAD_SECONDS: float = 0.1

# ── Noise reduction ───────────────────────────────────────────────────────────
_NOISE_REDUCE_SKIP_SNR_DB: float = 45.0
_NOISE_PROFILE_MIN_SECONDS: float = 0.3
_NOISE_PROFILE_GUARD_FRAMES: int = 5
_NOISE_PROFILE_MIN_RUN_FRAMES: int = 10
_NOISE_FFT_SECONDS: float = 0.02
_NOISE_FREQ_SMOOTH_HZ: float = 200.0

# ── Noise gate ────────────────────────────────────────────────────────────────
_GATE_CONTROL_MS: float = 1.0
_GATE_DETECTOR_MS: float = 10.0
_GATE_LOOKAHEAD_MS: float = 10.0
_GATE_HOLD_MS: float = 80.0
_GATE_ATTACK_MS: float = 3.0
_GATE_RELEASE_MS: float = 150.0

# ── Silence handling ──────────────────────────────────────────────────────────
_MIN_PAUSE_KEPT_MS: float = 80.0
_CROSSFADE_MS: float = 10.0
_WINDOW_CUT_SEARCH_SECONDS: float = 2.0
_WINDOW_CLIP_PENALTY: float = 50.0
_WINDOW_LEVEL_WEIGHT: float = 0.005

# ── Resampling (scipy fallback filter) ────────────────────────────────────────
_RESAMPLE_HALF_TAPS: int = 64
_RESAMPLE_PASSBAND: float = 0.95
_RESAMPLE_KAISER_BETA: float = 12.0

# ── Loudness ──────────────────────────────────────────────────────────────────
_LOUDNESS_BLOCK_SECONDS: float = 0.4
_LOUDNESS_BLOCK_OVERLAP: float = 0.75
_LOUDNESS_ABSOLUTE_GATE_LUFS: float = -70.0
_LOUDNESS_RELATIVE_GATE_LU: float = -10.0
_LOUDNESS_OFFSET: float = -0.691
_TRUE_PEAK_CHUNK: int = 1 << 20
_TRUE_PEAK_OVERLAP: int = 256
_SAFE_PEAK: float = 10.0 ** (-0.1 / 20.0)
_NORMALIZE_MODES: tuple[str, ...] = ("loudness", "peak")

# ── Report thresholds ─────────────────────────────────────────────────────────
_WARN_MIN_DURATION_S: float = 5.0
_WARN_MAX_DURATION_S: float = 30.0
_WARN_CLIPPED_RATIO: float = 1e-4
_WARN_LOW_SNR_DB: float = 20.0
_WARN_MIN_SPEECH_RATIO: float = 0.5
_WARN_QUIET_LUFS: float = -45.0


# ══════════════════════════════════════════════════════════════════════════════
# Config and report dataclasses
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PreprocessConfig:
    """Settings for the reference-clip cleaning pipeline.

    The defaults are the recommended settings for cloning: they repair what
    is safe to repair (rumble, a little hiss, leading / trailing silence,
    level, sample rate) and leave the performance itself untouched.

    Attributes
    ----------
    noise_reduce : bool
        Spectral noise reduction using a profile learned from the clip's own
        pauses. Skipped automatically when the clip is already clean or has no
        pause to learn from.
    noise_reduce_strength : float
        0.0 - 1.0; fraction of the detected noise removed. 0.25 is about
        2.5 dB, 0.5 about 6 dB. Higher values start to dull fricatives.
    noise_gate : bool
        Downward expander for the gaps between words. Off by default: with a
        clean recording it does nothing useful, with a noisy one it makes the
        noise pump.
    noise_gate_threshold_db : float
        Gate threshold in dB **relative to the clip's peak**, so the result
        does not depend on the recording level.
    highpass_filter : bool
        Zero-phase high-pass that removes rumble and plosive thumps.
    highpass_cutoff_hz : int
        Cutoff in Hz (-6 dB point of the zero-phase response).
    silence_removal : bool
        Opt-in: shorten internal pauses longer than ``max_silence_kept_ms``.
        Speech is never cut; only the middle of a pause is removed.
    silence_threshold_db : float
        Speech / silence threshold in dB relative to the clip's peak.
    min_segment_ms : int
        Sound bursts shorter than this, isolated by silence on both sides
        (clicks, key presses), are treated as silence.
    max_silence_kept_ms : int
        Length a long pause is shortened to when ``silence_removal`` is on.
        Never less than 80 ms so word onsets and tails keep their margin.
    normalize_volume : bool
        Normalise the level as the last step.
    normalize_target_dbfs : float
        Peak target in dBFS; only used when ``normalize_mode == "peak"``.
    formant_shift, formant_quefrency, formant_timbre
        Accepted for compatibility and ignored (see module docstring).
    resample : bool
        Bring the clip to ``target_sample_rate``.
    target_sample_rate : int
        Rate handed to the cloning model. 24 kHz covers the conditioning
        encoders of the supported models.
    noise_gate_range_db : float
        Maximum attenuation of the gate in dB. The gate attenuates; it never
        mutes to digital silence.
    trim_silence : bool
        Trim leading and trailing silence, keeping ``edge_silence_ms``.
    edge_silence_ms : int
        Silence kept before the first and after the last word.
    edge_fade_ms : int
        Raised-cosine fade applied at both ends of the clip; 0 disables it.
    normalize_mode : str
        ``"loudness"`` (integrated LUFS, default) or ``"peak"``.
    loudness_target_lufs : float
        Integrated-loudness target for ``normalize_mode == "loudness"``.
    true_peak_ceiling_dbfs : float
        True-peak ceiling. When the target would exceed it the gain is
        reduced; the clip is never limited or compressed.
    allow_upsample : bool
        Allow ``resample`` to raise the sample rate. Off by default because
        upsampling adds no information.
    select_best_window : bool
        Opt-in: cut a long recording down to its best
        ``best_window_seconds`` (most speech, no clipping).
    best_window_seconds : float
        Length of the window chosen by ``select_best_window``.
    """

    # Step 1: Noise reduction (gentle; profile learned from the clip's pauses)
    noise_reduce:           bool  = True
    noise_reduce_strength:  float = 0.25   # 0.0 – 1.0

    # Step 2: Noise gate (off by default: a gate cannot improve a clean take)
    noise_gate:             bool  = False
    noise_gate_threshold_db: float = -45.0  # dB relative to the clip's peak

    # Step 3: High-pass filter (removes sub-bass rumble / plosive thumps)
    highpass_filter:        bool  = True
    highpass_cutoff_hz:     int   = 80      # Hz

    # Step 4: Pause shortening (off by default: internal edits hurt prosody)
    silence_removal:        bool  = False
    silence_threshold_db:   float = _DEFAULT_SILENCE_THRESHOLD_DB   # dB re peak
    min_segment_ms:         int   = _DEFAULT_MIN_SEGMENT_MS   # ms — shorter bursts are clicks
    max_silence_kept_ms:    int   = 500     # ms a long pause is shortened to

    # Step 5: Level (always applied last)
    normalize_volume:       bool  = True
    normalize_target_dbfs:  float = -3.0    # dBFS, normalize_mode="peak" only

    # Step 6: Formant shifting (accepted, ignored)
    formant_shift:          bool  = False
    formant_quefrency:      float = 1.0
    formant_timbre:         float = 1.0

    # Step 7: Resample (downsample only unless allow_upsample)
    resample:               bool  = True
    target_sample_rate:     int   = 24000

    # ── Fields added after the original seven steps ──────────────────────────
    noise_gate_range_db:    float = 12.0    # dB of attenuation when closed
    trim_silence:           bool  = True
    edge_silence_ms:        int   = 150     # ms kept at each end
    edge_fade_ms:           int   = 10      # ms raised-cosine fade at each end
    normalize_mode:         str   = "loudness"
    loudness_target_lufs:   float = -20.0   # LUFS
    true_peak_ceiling_dbfs: float = -1.0    # dBTP
    allow_upsample:         bool  = False
    select_best_window:     bool  = False
    best_window_seconds:    float = 15.0


@dataclass
class VoiceReport:
    """Measurements of one clip plus human-readable warnings.

    Attributes
    ----------
    duration_s : float
        Length in seconds.
    sample_rate : int
        Sample rate in Hz.
    channels : int
        Channel count of the file that was measured.
    peak_dbfs : float
        Sample peak.
    true_peak_dbfs : float
        Inter-sample (4x oversampled) peak.
    loudness_lufs : float
        Integrated loudness (ITU-R BS.1770); -120 for silence.
    noise_floor_dbfs : float
        RMS level of the quietest 5 % of 10 ms frames.
    snr_db : float
        Speech level minus noise floor. An estimate: a clip with no pause at
        all reads low.
    clipped_ratio : float
        Fraction of samples sitting in flat-topped runs at the clip's peak.
    speech_ratio : float
        Fraction of the clip that is speech (pauses under 200 ms count as
        speech).
    warnings : list[str]
        Problems the user should act on.
    steps_applied : list[str]
        Pipeline steps that ran. Empty for ``analyze_voice`` and cache hits.
    steps_skipped : list[str]
        ``"step: reason"`` for every requested step that did not run.
    from_cache : bool
        The audio came from the disk cache, so the step lists are unknown.
    source : VoiceReport | None
        Measurements of the clip as uploaded, when this report describes a
        processed clip.
    """

    duration_s:       float
    sample_rate:      int
    channels:         int
    peak_dbfs:        float
    true_peak_dbfs:   float
    loudness_lufs:    float
    noise_floor_dbfs: float
    snr_db:           float
    clipped_ratio:    float
    speech_ratio:     float
    warnings:         list[str] = field(default_factory=list)
    steps_applied:    list[str] = field(default_factory=list)
    steps_skipped:    list[str] = field(default_factory=list)
    from_cache:       bool = False
    source:           VoiceReport | None = None

    def to_dict(self) -> dict:
        """Return a JSON-serialisable dict (nested ``source`` included)."""
        return dataclasses.asdict(self)


@dataclass
class _LevelStats:
    """Frame-level measurements shared by the detection steps."""

    hop:            int
    frame_rms:      np.ndarray
    frame_db:       np.ndarray
    noise_floor_db: float
    speech_db:      float
    snr_db:         float
    voiced:         np.ndarray


@dataclass
class _PipelineResult:
    """Outcome of one pipeline run."""

    wav_bytes: bytes
    report:    VoiceReport
    degraded:  bool   # a requested step could not run; must not be cached


# ══════════════════════════════════════════════════════════════════════════════
# Cache
# ══════════════════════════════════════════════════════════════════════════════

def _get_cache_dir() -> str:
    """Return the cache directory, creating it if needed.

    ``ABM_VOICE_CACHE_DIR`` overrides the default location next to this
    module (useful when the package directory is read-only).
    """
    override = os.environ.get(_CACHE_DIR_ENV, "").strip()
    if override:
        cache_dir = os.path.abspath(os.path.expanduser(override))
    else:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        cache_dir = os.path.join(base_dir, _CACHE_DIR_NAME)
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def _read_input_bytes(input_audio: bytes | str | os.PathLike) -> bytes:
    """Return the raw file bytes for a path or pass bytes through."""
    if isinstance(input_audio, (bytes, bytearray, memoryview)):
        return bytes(input_audio)
    if isinstance(input_audio, (str, os.PathLike)):
        path = os.fspath(input_audio)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Voice sample not found: {path}")
        with open(path, "rb") as f:
            return f.read()
    raise TypeError(f"input_audio must be bytes or str/path, got {type(input_audio)}")


def _config_fingerprint(config: PreprocessConfig) -> str:
    """Canonical JSON of the config; ``80`` and ``80.0`` hash the same."""
    canonical: dict = {"_pipeline_version": _PIPELINE_VERSION}
    for key, value in dataclasses.asdict(config).items():
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            value = float(value)
        canonical[key] = value
    return json.dumps(canonical, sort_keys=True)


def _get_cache_path(input_audio: bytes | str | os.PathLike, config: PreprocessConfig) -> str:
    """Cache filename from SHA-256 of the audio, the config and the pipeline version."""
    if isinstance(input_audio, (str, os.PathLike)) and not os.path.isfile(os.fspath(input_audio)):
        input_bytes = os.fspath(input_audio).encode("utf-8")
    else:
        input_bytes = _read_input_bytes(input_audio)

    audio_hash = hashlib.sha256(input_bytes).hexdigest()[:_CACHE_HASH_LENGTH]
    config_hash = hashlib.sha256(
        _config_fingerprint(config).encode("utf-8")
    ).hexdigest()[:_CACHE_HASH_LENGTH]
    filename = f"voice_{audio_hash}_{config_hash}{_CACHE_ENTRY_SUFFIX}"
    return os.path.join(_get_cache_dir(), filename)


def _read_cache(cache_path: str) -> bytes | None:
    """Return cached WAV bytes, or None on a miss / unreadable / corrupt entry."""
    try:
        with open(cache_path, "rb") as f:
            data = f.read()
    except FileNotFoundError:
        return None
    except OSError as exc:
        logger.warning("[Preprocess Cache] Failed to read cache entry: %s", exc)
        return None

    if len(data) < 44 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        logger.warning(
            "[Preprocess Cache] Ignoring corrupt entry: %s", os.path.basename(cache_path)
        )
        try:
            os.remove(cache_path)
        except OSError:
            pass
        return None

    try:
        os.utime(cache_path, None)   # mark as recently used for eviction
    except OSError:
        pass
    logger.info("[Preprocess Cache] Cache HIT: %s", os.path.basename(cache_path))
    return data


def _write_cache(cache_path: str, data: bytes) -> None:
    """Write a cache entry atomically, then evict old entries.

    Each writer uses its own temp file, so the UI and the API can store the
    same key at the same time without corrupting it.
    """
    cache_dir = os.path.dirname(cache_path)
    tmp_path = ""
    try:
        with _CACHE_LOCK:
            fd, tmp_path = tempfile.mkstemp(dir=cache_dir, suffix=_CACHE_TMP_SUFFIX)
            with os.fdopen(fd, "wb") as f:
                f.write(data)
            os.replace(tmp_path, cache_path)
            tmp_path = ""
            _evict_cache(cache_dir, keep=cache_path)
        logger.info("[Preprocess Cache] Saved cache entry: %s", os.path.basename(cache_path))
    except OSError as exc:
        logger.warning("[Preprocess Cache] Failed to write cache entry: %s", exc)
        if tmp_path:
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def _evict_cache(cache_dir: str, keep: str = "") -> None:
    """Drop least-recently-used entries beyond the size limits and stale temp files."""
    now = time.time()
    entries: list[tuple[float, int, str]] = []
    try:
        names = os.listdir(cache_dir)
    except OSError:
        return
    for name in names:
        path = os.path.join(cache_dir, name)
        try:
            st = os.stat(path)
        except OSError:
            continue
        if name.endswith(_CACHE_TMP_SUFFIX):
            if now - st.st_mtime > _CACHE_STALE_TMP_SECONDS:
                try:
                    os.remove(path)
                except OSError:
                    pass
        elif name.endswith(_CACHE_ENTRY_SUFFIX):
            entries.append((st.st_mtime, st.st_size, path))

    entries.sort(reverse=True)   # newest first
    total = 0
    for index, (_, size, path) in enumerate(entries):
        total += size
        over_limit = index >= _CACHE_MAX_ENTRIES or total > _CACHE_MAX_BYTES
        if over_limit and path != keep:
            try:
                os.remove(path)
            except OSError:
                pass


def clear_cache() -> int:
    """Delete every cached result and leftover temp file.

    Returns
    -------
    int
        Number of files removed.
    """
    try:
        cache_dir = _get_cache_dir()
        names = os.listdir(cache_dir)
    except OSError:
        return 0
    removed = 0
    with _CACHE_LOCK:
        for name in names:
            if name.endswith((_CACHE_ENTRY_SUFFIX, _CACHE_TMP_SUFFIX)):
                try:
                    os.remove(os.path.join(cache_dir, name))
                    removed += 1
                except OSError:
                    pass
    return removed


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════

def preprocess(
    input_audio: bytes | str | os.PathLike,
    config: PreprocessConfig | None = None,
    log_fn: _LogFn | None = None,
    use_cache: bool = True,
) -> bytes:
    """Clean a reference clip and return it as 16-bit mono WAV bytes.

    Parameters
    ----------
    input_audio : bytes | str | os.PathLike
        Audio file bytes (WAV, FLAC, OGG or MP3) or a path to such a file.
    config : PreprocessConfig, optional
        Pipeline settings; the cloning defaults are used when None.
    log_fn : callable, optional
        ``callable(str)`` receiving progress lines.
    use_cache : bool, default True
        Reuse / store results in the disk cache. A run in which a requested
        step could not be applied (missing library, error) is never cached.

    Returns
    -------
    bytes
        Processed WAV audio.

    Raises
    ------
    FileNotFoundError
        ``input_audio`` is a path that does not exist.
    ValueError
        The audio cannot be decoded or contains no samples.
    """
    input_bytes = _read_input_bytes(input_audio)
    if config is None:
        config = PreprocessConfig()
    cache_path, cached = _cache_lookup(input_bytes, config, use_cache, log_fn)
    if cached is not None:
        return cached
    result = _run_pipeline(input_bytes, config, log_fn)
    _cache_store(cache_path, result)
    return result.wav_bytes


def preprocess_with_report(
    input_audio: bytes | str | os.PathLike,
    config: PreprocessConfig | None = None,
    log_fn: _LogFn | None = None,
    use_cache: bool = True,
) -> tuple[bytes, VoiceReport]:
    """Clean a reference clip and describe the result.

    Same processing, cache and errors as :func:`preprocess`. The report
    measures the processed clip (what the cloning model will hear),
    ``report.source`` measures the upload, and ``report.warnings`` covers
    both.

    Parameters
    ----------
    input_audio, config, log_fn, use_cache
        See :func:`preprocess`.

    Returns
    -------
    tuple[bytes, VoiceReport]
        Processed WAV bytes and the analysis report.
    """
    input_bytes = _read_input_bytes(input_audio)
    if config is None:
        config = PreprocessConfig()
    cache_path, cached = _cache_lookup(input_bytes, config, use_cache, log_fn)
    if cached is not None:
        return cached, _report_for_cached(input_bytes, cached)
    result = _run_pipeline(input_bytes, config, log_fn)
    _cache_store(cache_path, result)
    return result.wav_bytes, result.report


def analyze_voice(input_audio: bytes | str | os.PathLike) -> VoiceReport:
    """Measure a clip without processing it.

    Parameters
    ----------
    input_audio : bytes | str | os.PathLike
        Audio file bytes or a path to an audio file.

    Returns
    -------
    VoiceReport
        Duration, levels, noise floor, clipping and speech ratio, with
        warnings for anything that will hurt cloning.
    """
    report, notes = _analyze(_read_input_bytes(input_audio))
    report.warnings = notes + _source_warnings(report) + _quality_warnings(report)
    return report


def _analyze(input_bytes: bytes) -> tuple[VoiceReport, list[str]]:
    """Measure file bytes; also return decode-time notes (repaired samples)."""
    data, sr = _decode(input_bytes)
    data, n_bad = _sanitize(data)
    audio = _remove_dc(_to_mono(data)[0])
    report = _measure(audio, sr, data.shape[1], float(_clipped_mask(data).mean()))
    return report, ([_nonfinite_warning(n_bad)] if n_bad else [])


def _report_for_cached(input_bytes: bytes, cached: bytes) -> VoiceReport:
    """Rebuild the report for a cache hit by measuring both clips again."""
    source, notes = _analyze(input_bytes)
    report, _ = _analyze(cached)
    report.warnings = notes + _source_warnings(source) + _quality_warnings(report)
    report.source = source
    report.from_cache = True
    return report


def _cache_lookup(
    input_bytes: bytes,
    config: PreprocessConfig,
    use_cache: bool,
    log_fn: _LogFn | None,
) -> tuple[str | None, bytes | None]:
    """Return ``(cache path or None, cached bytes or None)``.

    The cache is an optimisation: an unwritable cache directory disables it
    for this call instead of failing the preprocessing.
    """
    if not use_cache:
        return None, None
    try:
        cache_path = _get_cache_path(input_bytes, config)
    except OSError as exc:
        logger.warning("[Preprocess Cache] Cache unavailable: %s", exc)
        return None, None
    cached = _read_cache(cache_path)
    if cached is not None and log_fn:
        log_fn(f"[Preprocess] Loaded from cache ({len(cached)} bytes)")
    return cache_path, cached


def _cache_store(cache_path: str | None, result: _PipelineResult) -> None:
    """Cache a result unless a requested step failed to run.

    A result produced without a requested step is not what the config hash
    promises; caching it would keep serving the unprocessed audio after the
    missing library has been installed.
    """
    if cache_path is None:
        return
    if result.degraded:
        logger.warning(
            "[Preprocess Cache] Result not cached, a step did not run: %s",
            "; ".join(result.report.steps_skipped),
        )
        return
    _write_cache(cache_path, result.wav_bytes)


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline
# ══════════════════════════════════════════════════════════════════════════════

def _run_preprocessing_pipeline(
    input_bytes: bytes,
    config: PreprocessConfig,
    log_fn: _LogFn | None = None,
) -> bytes:
    """Run file bytes through the pipeline and return WAV bytes (no cache)."""
    return _run_pipeline(input_bytes, config, log_fn).wav_bytes


def _run_pipeline(
    input_bytes: bytes,
    config: PreprocessConfig,
    log_fn: _LogFn | None = None,
) -> _PipelineResult:
    """Decode, clean, measure and encode one clip."""
    if config.normalize_mode not in _NORMALIZE_MODES:
        raise ValueError(
            f"normalize_mode must be one of {_NORMALIZE_MODES}, got {config.normalize_mode!r}"
        )

    applied: list[str] = []
    skipped: list[str] = []
    notes: list[str] = []          # warnings raised by the pipeline itself
    degraded = False

    def log(msg: str) -> None:
        if log_fn:
            log_fn(msg)
        else:
            logger.info(msg)

    def skip(step: str, reason: str, *, failure: bool = False) -> None:
        """Record a step that did not run; ``failure`` also blocks caching."""
        nonlocal degraded
        skipped.append(f"{step}: {reason}")
        if failure:
            degraded = True
            notes.append(f"{step} was requested but could not run: {reason}.")
        else:
            log(f"[Preprocess] {step} skipped: {reason}.")

    # ── Decode, repair, fold to mono ─────────────────────────────────────────
    data, sr = _decode(input_bytes)
    data, n_bad = _sanitize(data)
    if n_bad:
        notes.append(_nonfinite_warning(n_bad))
    channels = data.shape[1]
    clipped_mask = _clipped_mask(data)
    audio, mono_note = _to_mono(data)
    if mono_note:
        notes.append(mono_note)
    audio = _remove_dc(audio)
    source = _measure(audio, sr, channels, float(clipped_mask.mean()))
    log(f"[Preprocess] Loaded audio: {source.duration_s:.1f}s @ {sr}Hz, {channels} channel(s)")

    # ── High-pass filter ─────────────────────────────────────────────────────
    if config.highpass_filter:
        cutoff = float(config.highpass_cutoff_hz)
        if not 0.0 < cutoff < sr / 2.0:
            skip("highpass_filter", f"cutoff {cutoff:g} Hz is outside 0..{sr / 2:g} Hz")
        elif len(audio) < 16:
            skip("highpass_filter", "clip is too short")
        else:
            try:
                log(f"[Preprocess] High-pass filter @ {cutoff:g}Hz...")
                audio = _highpass(audio, sr, cutoff)
                applied.append("highpass_filter")
            except ImportError:
                skip("highpass_filter", "scipy is not installed", failure=True)

    # ── Noise reduction ──────────────────────────────────────────────────────
    if config.noise_reduce:
        strength = float(np.clip(config.noise_reduce_strength, 0.0, 1.0))
        # Always the built-in threshold here: a user threshold set high for
        # trimming would put quiet speech into the noise profile.
        stats = _level_stats(audio, sr, _DEFAULT_SILENCE_THRESHOLD_DB)
        noise = _noise_profile(audio, sr, stats)
        n_fft = _noise_fft_size(sr)
        if strength <= 0.0:
            skip("noise_reduce", "strength is 0")
        elif stats.snr_db >= _NOISE_REDUCE_SKIP_SNR_DB:
            skip("noise_reduce", f"clip is already clean (SNR {stats.snr_db:.0f} dB)")
        elif noise is None or len(noise) < 2 * n_fft or len(audio) < 4 * n_fft:
            skip("noise_reduce", "no pause long enough to learn the noise from")
        else:
            try:
                log(f"[Preprocess] Noise reduction (strength {strength:.2f})...")
                audio = _reduce_noise(audio, sr, noise, strength, n_fft)
                applied.append("noise_reduce")
            except ImportError:
                skip("noise_reduce", "noisereduce is not installed", failure=True)
            except Exception as exc:   # third-party DSP: report it, keep the audio
                skip("noise_reduce", f"failed ({exc})", failure=True)

    # ── Noise gate ───────────────────────────────────────────────────────────
    if config.noise_gate:
        if config.noise_gate_range_db <= 0.0:
            skip("noise_gate", "range is 0 dB")
        else:
            log(f"[Preprocess] Noise gate @ {config.noise_gate_threshold_db:g} dB re peak...")
            audio = _noise_gate(
                audio, sr, config.noise_gate_threshold_db, config.noise_gate_range_db
            )
            applied.append("noise_gate")

    # ── Best window ──────────────────────────────────────────────────────────
    if config.select_best_window:
        window = _best_window(
            _level_stats(audio, sr, config.silence_threshold_db),
            n_samples=len(audio),
            window=int(sr * max(0.0, config.best_window_seconds)),
            pad=int(sr * max(0, config.edge_silence_ms) / 1000.0),
            min_frames=_min_segment_frames(config.min_segment_ms),
            clipped_mask=clipped_mask,
        )
        if window is None:
            skip("select_best_window", "clip is not longer than the window")
        else:
            start, end = window
            log(f"[Preprocess] Best window: {start / sr:.1f}s - {end / sr:.1f}s")
            audio = audio[start:end]
            applied.append("select_best_window")

    # ── Silence: trim the ends, optionally shorten internal pauses ───────────
    silence_steps = [
        name for name, enabled in (
            ("trim_silence", config.trim_silence),
            ("silence_removal", config.silence_removal),
        ) if enabled
    ]
    if silence_steps:
        segments = _find_speech(audio, sr, config.silence_threshold_db, config.min_segment_ms)
        if not segments:
            for name in silence_steps:
                skip(name, "no speech detected")
        else:
            before = len(audio)
            audio = _remove_silence(
                audio, sr,
                threshold_db=config.silence_threshold_db,
                min_segment_ms=config.min_segment_ms,
                max_silence_kept_ms=config.max_silence_kept_ms,
                trim_edges=config.trim_silence,
                shorten_pauses=config.silence_removal,
                edge_silence_ms=config.edge_silence_ms,
                segments=segments,
            )
            applied.extend(silence_steps)
            log(f"[Preprocess] Silence: {before / sr:.2f}s -> {len(audio) / sr:.2f}s")

    # ── Formant shift: deliberately not applied ──────────────────────────────
    if config.formant_shift:
        skipped.append("formant_shift: no longer applied")
        notes.append(
            "Formant shift is ignored: it changes the timbre the clone is meant "
            "to copy and used to produce silent or clipped audio."
        )

    # Fade before resampling so the resampler never sees a hard edge.
    audio = _fade_edges(audio, int(sr * max(0, config.edge_fade_ms) / 1000.0))

    # ── Resample ─────────────────────────────────────────────────────────────
    target_sr = int(config.target_sample_rate)
    if config.resample and target_sr != sr:
        if target_sr <= 0:
            skip("resample", f"invalid target rate {target_sr}")
        elif target_sr > sr and not config.allow_upsample:
            skip("resample", f"source is {sr} Hz; upsampling to {target_sr} Hz adds nothing")
        elif len(audio) < 2:
            skip("resample", "clip is too short")
        else:
            try:
                log(f"[Preprocess] Resample {sr}->{target_sr}Hz...")
                audio = _resample(audio, sr, target_sr)
                sr = target_sr
                applied.append("resample")
            except ImportError:
                skip("resample", "neither soxr nor scipy is installed", failure=True)

    # ── Level: always last ───────────────────────────────────────────────────
    if config.normalize_volume:
        if config.normalize_mode == "peak":
            log(f"[Preprocess] Normalize peak to {config.normalize_target_dbfs:g} dBFS...")
            audio = _normalize_peak(audio, config.normalize_target_dbfs)
            applied.append("normalize_volume")
        else:
            log(
                f"[Preprocess] Normalize loudness to {config.loudness_target_lufs:g} LUFS "
                f"(ceiling {config.true_peak_ceiling_dbfs:g} dBTP)..."
            )
            audio, gain_db, limited = _normalize_loudness(
                audio, sr, config.loudness_target_lufs, config.true_peak_ceiling_dbfs
            )
            if gain_db is None:
                skip("normalize_volume", "clip is silent")
            else:
                applied.append("normalize_volume")
                if limited:
                    log(
                        "[Preprocess] Peak ceiling reached; loudness stays below the "
                        f"target (gain {gain_db:+.1f} dB)."
                    )

    # Whatever the settings, never hand libsndfile samples it would clip.
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > _SAFE_PEAK:
        audio = audio * (_SAFE_PEAK / peak)
        log("[Preprocess] Level reduced to avoid clipping in the 16-bit output.")

    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")

    report = _measure(audio, sr, 1, 0.0)
    report.source = source
    report.steps_applied = applied
    report.steps_skipped = skipped
    report.warnings = notes + _source_warnings(source) + _quality_warnings(report)
    for message in report.warnings:
        logger.warning("[Preprocess] %s", message)
        if log_fn:
            log_fn(f"[Preprocess] WARNING: {message}")
    log("[Preprocess] Done.")
    return _PipelineResult(wav_bytes=buf.getvalue(), report=report, degraded=degraded)


# ══════════════════════════════════════════════════════════════════════════════
# Decoding and measurement
# ══════════════════════════════════════════════════════════════════════════════

def _decode(input_bytes: bytes) -> tuple[np.ndarray, int]:
    """Decode file bytes to a ``(frames, channels)`` float64 array and its rate.

    Raises
    ------
    ValueError
        The bytes are not a decodable audio file, the file is empty, or it
        is longer than ``_MAX_INPUT_SECONDS``.
    """
    try:
        info = sf.info(io.BytesIO(input_bytes))
        too_long = info.samplerate > 0 and info.frames / info.samplerate > _MAX_INPUT_SECONDS
        if not too_long:
            data, sr = sf.read(io.BytesIO(input_bytes), dtype="float64", always_2d=True)
    except (sf.SoundFileError, RuntimeError, OSError) as exc:
        reason = str(exc).rsplit(": ", 1)[-1].rstrip(".")   # drop the BytesIO repr
        raise ValueError(
            f"Could not decode the audio ({reason}). Supported: WAV, FLAC, OGG, MP3."
        ) from exc
    if too_long:
        raise ValueError(
            f"The recording is {info.frames / info.samplerate / 60.0:.0f} minutes long; "
            f"upload at most {_MAX_INPUT_SECONDS / 60.0:.0f} minutes "
            "(5-30 seconds of clean speech is ideal)."
        )
    if data.shape[0] == 0 or data.shape[1] == 0:
        raise ValueError("The audio file contains no samples.")
    if sr <= 0:
        raise ValueError(f"The audio file reports an invalid sample rate ({sr}).")
    return data, int(sr)


def _sanitize(data: np.ndarray) -> tuple[np.ndarray, int]:
    """Replace NaN / inf samples with silence; return the count replaced."""
    bad = ~np.isfinite(data)
    n_bad = int(bad.sum())
    if n_bad:
        data = np.where(bad, 0.0, data)
    return data, n_bad


def _nonfinite_warning(n_bad: int) -> str:
    """Warning text for samples replaced by ``_sanitize``."""
    return f"Replaced {n_bad} corrupt (NaN/inf) sample(s) with silence."


def _remove_dc(audio: np.ndarray) -> np.ndarray:
    """Subtract the mean so a DC offset does not pose as signal or noise."""
    return audio - float(np.mean(audio)) if audio.size else audio


def _to_mono(data: np.ndarray) -> tuple[np.ndarray, str]:
    """Fold channels to mono without losing the voice.

    A plain average halves the voice when one channel is dead and cancels it
    when the channels are out of phase, so channels more than 20 dB below the
    loudest are left out, and if the average still loses more than 6 dB the
    loudest channel is used alone.

    Returns
    -------
    tuple[np.ndarray, str]
        The mono signal and a note describing any non-trivial choice.
    """
    if data.shape[1] == 1:
        return data[:, 0].copy(), ""
    rms = np.sqrt(np.mean(data ** 2, axis=0))
    loudest = int(np.argmax(rms))
    if rms[loudest] <= 0.0:
        return data[:, 0].copy(), ""
    active = rms >= rms[loudest] * 10.0 ** (-20.0 / 20.0)
    if int(active.sum()) == 1:
        return data[:, loudest].copy(), (
            f"Only channel {loudest + 1} carries the voice; the other channel(s) were dropped."
        )
    mix = data[:, active].mean(axis=1)
    if float(np.sqrt(np.mean(mix ** 2))) < 0.5 * rms[loudest]:
        return data[:, loudest].copy(), (
            "The channels are out of phase; "
            f"channel {loudest + 1} was used alone instead of a mono mix."
        )
    return mix, ""


def _db(value: float) -> float:
    """Amplitude to dB, floored at ``_SILENCE_DB``."""
    if not value > 0.0:
        return _SILENCE_DB
    return max(_SILENCE_DB, 20.0 * math.log10(value))


def _frame_levels(audio: np.ndarray, hop: int) -> tuple[np.ndarray, np.ndarray]:
    """RMS and peak of consecutive ``hop``-sample frames (last one may be short)."""
    n = len(audio)
    n_frames = max(1, -(-n // hop))
    padded = np.zeros(n_frames * hop, dtype=np.float64)
    padded[:n] = audio
    blocks = padded.reshape(n_frames, hop)
    counts = np.full(n_frames, hop, dtype=np.float64)
    counts[-1] = max(1, n - (n_frames - 1) * hop)
    rms = np.sqrt((blocks ** 2).sum(axis=1) / counts)
    return rms, np.abs(blocks).max(axis=1)


def _frame_hop(sr: int) -> int:
    """Samples per analysis frame."""
    return max(1, int(round(sr * _FRAME_MS / 1000.0)))


def _robust_peak(frame_peaks: np.ndarray) -> float:
    """Peak level that ignores a stray click (99th percentile of frame peaks)."""
    ref = float(np.percentile(frame_peaks, _ROBUST_PEAK_PERCENTILE))
    return ref if ref > 0.0 else float(frame_peaks.max())


def _level_stats(audio: np.ndarray, sr: int, threshold_db: float) -> _LevelStats:
    """Frame levels, noise floor and a speech / silence decision per frame.

    Levels are judged relative to a robust peak (99th percentile of the
    per-frame peaks) so a single click cannot shift every threshold. The
    speech threshold is ``threshold_db`` below that peak, raised to sit above
    the noise floor when the recording is noisy, but never higher than
    25 dB below the peak.
    """
    hop = _frame_hop(sr)
    rms, peaks = _frame_levels(audio, hop)
    ref = _robust_peak(peaks)
    frame_db = 20.0 * np.log10(np.maximum(rms, 10.0 ** (_SILENCE_DB / 20.0)))

    n_quiet = min(len(rms), max(3, int(len(rms) * _NOISE_FLOOR_FRACTION)))
    quiet = np.sort(rms)[:n_quiet]
    noise_floor_db = _db(float(np.sqrt(np.mean(quiet ** 2))))

    ref_db = _db(ref)
    threshold = max(
        ref_db + threshold_db,
        min(noise_floor_db + _NOISE_MARGIN_DB, ref_db + _MAX_ADAPTIVE_THRESHOLD_DB),
    )
    voiced = (frame_db > threshold) if ref > 0.0 else np.zeros(len(rms), dtype=bool)

    if voiced.any():
        speech_db = _db(float(np.sqrt(np.mean(rms[voiced] ** 2))))
        snr_db = float(np.clip(speech_db - noise_floor_db, 0.0, _MAX_SNR_DB))
    else:
        speech_db, snr_db = _SILENCE_DB, 0.0

    return _LevelStats(
        hop=hop, frame_rms=rms, frame_db=frame_db, noise_floor_db=noise_floor_db,
        speech_db=speech_db, snr_db=snr_db, voiced=voiced,
    )


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """``[start, end)`` index ranges of consecutive True values."""
    if mask.size == 0:
        return []
    edges = np.diff(np.concatenate(([0], mask.astype(np.int8), [0])))
    starts = np.flatnonzero(edges == 1)
    ends = np.flatnonzero(edges == -1)
    return list(zip(starts.tolist(), ends.tolist()))


def _min_segment_frames(min_segment_ms: float) -> int:
    """``min_segment_ms`` in analysis frames, at least one."""
    return max(1, int(math.ceil(float(min_segment_ms) / _FRAME_MS)))


def _speech_segments(voiced: np.ndarray, min_frames: int) -> list[tuple[int, int]]:
    """Speech segments as ``[start, end)`` frame ranges.

    Voiced runs separated by less than 200 ms belong to one phrase (stop
    closures, short breaths). A phrase whose own span is shorter than
    ``min_frames`` is a click, not speech, and is dropped. The span is
    measured on the sound itself, never on the silence around it.
    """
    merge_gap = max(1, int(round(_MERGE_GAP_MS / _FRAME_MS)))
    merged: list[list[int]] = []
    for start, end in _runs(voiced):
        if merged and start - merged[-1][1] < merge_gap:
            merged[-1][1] = end
        else:
            merged.append([start, end])
    return [(s, e) for s, e in merged if e - s >= min_frames]


def _find_speech(
    audio: np.ndarray, sr: int, threshold_db: float, min_segment_ms: float
) -> list[tuple[int, int]]:
    """Speech segments as ``[start, end)`` sample ranges."""
    stats = _level_stats(audio, sr, threshold_db)
    frames = _speech_segments(stats.voiced, _min_segment_frames(min_segment_ms))
    return [(s * stats.hop, min(e * stats.hop, len(audio))) for s, e in frames]


def _clipped_mask(data: np.ndarray) -> np.ndarray:
    """Per-sample mask of flat-topped runs at either rail, any channel.

    A run of three or more consecutive samples within one 16-bit step of the
    channel's extreme is a clipped crest; a clean waveform touches its
    extreme for a single sample. The two rails are judged separately because
    converters, and DC removal after the fact, leave them at different
    levels. A rail far below the other is not a rail at all (a one-sided
    signal resting at zero) and is skipped.
    """
    if data.ndim == 1:
        data = data[:, None]
    mask = np.zeros(data.shape[0], dtype=bool)
    for ch in range(data.shape[1]):
        x = data[:, ch]
        if x.size == 0:
            continue
        top, bottom = float(x.max()), float(x.min())
        peak = max(abs(top), abs(bottom))
        if peak <= 0.0:
            continue
        tolerance = max(peak * _CLIP_RELATIVE_TOLERANCE, _CLIP_ABSOLUTE_TOLERANCE)
        rails = []
        if top >= _CLIP_MIN_RAIL_RATIO * peak:
            rails.append(x >= top - tolerance)
        if -bottom >= _CLIP_MIN_RAIL_RATIO * peak:
            rails.append(x <= bottom + tolerance)
        for near in rails:
            for start, end in _runs(near):
                if end - start >= _CLIP_MIN_RUN:
                    mask[start:end] = True
    return mask


def _measure(audio: np.ndarray, sr: int, channels: int, clipped_ratio: float) -> VoiceReport:
    """Measure a mono signal. Warnings are added by the caller."""
    stats = _level_stats(audio, sr, _DEFAULT_SILENCE_THRESHOLD_DB)
    segments = _speech_segments(stats.voiced, _min_segment_frames(_DEFAULT_MIN_SEGMENT_MS))
    speech_frames = sum(e - s for s, e in segments)
    loudness = _integrated_loudness(audio, sr)
    return VoiceReport(
        duration_s=len(audio) / float(sr),
        sample_rate=int(sr),
        channels=int(channels),
        peak_dbfs=_db(float(np.max(np.abs(audio))) if audio.size else 0.0),
        true_peak_dbfs=_db(_true_peak(audio, sr)),
        loudness_lufs=max(_SILENCE_DB, loudness) if math.isfinite(loudness) else _SILENCE_DB,
        noise_floor_dbfs=stats.noise_floor_db,
        snr_db=stats.snr_db,
        clipped_ratio=float(clipped_ratio),
        speech_ratio=speech_frames / float(len(stats.voiced)) if len(stats.voiced) else 0.0,
    )


def _source_warnings(report: VoiceReport) -> list[str]:
    """Problems of the recording itself that processing cannot undo."""
    out: list[str] = []
    if report.clipped_ratio >= _WARN_CLIPPED_RATIO:
        out.append(
            f"The recording is clipped ({report.clipped_ratio * 100:.2f}% of samples). "
            "Clipping distortion is copied by the clone; re-record at a lower input level."
        )
    if report.speech_ratio > 0.0 and report.loudness_lufs < _WARN_QUIET_LUFS:
        out.append(
            f"The recording is very quiet ({report.loudness_lufs:.0f} LUFS); "
            "raising it also raises its noise. Record closer to the microphone."
        )
    return out


def _quality_warnings(report: VoiceReport) -> list[str]:
    """Warnings about the length, noise and pauses of the clip to be cloned."""
    if report.speech_ratio <= 0.0:
        return ["No speech was detected in the clip."]
    out: list[str] = []
    if report.duration_s < _WARN_MIN_DURATION_S:
        out.append(
            f"The clip is only {report.duration_s:.1f} s long; "
            f"use at least {_WARN_MIN_DURATION_S:.0f} s of speech for a stable clone."
        )
    elif report.duration_s > _WARN_MAX_DURATION_S:
        out.append(
            f"The clip is {report.duration_s:.0f} s long; cloning works best with "
            f"{_WARN_MIN_DURATION_S:.0f}-{_WARN_MAX_DURATION_S:.0f} s. "
            "Enable best-window selection or upload a shorter clip."
        )
    if report.snr_db < _WARN_LOW_SNR_DB:
        out.append(
            f"High background noise (estimated SNR {report.snr_db:.0f} dB); "
            "the clone will carry it. Record in a quieter place or closer to the microphone."
        )
    if report.speech_ratio < _WARN_MIN_SPEECH_RATIO:
        out.append(
            f"Only {report.speech_ratio * 100:.0f}% of the clip is speech; "
            "long pauses make the clone slow and hesitant."
        )
    return out


# ══════════════════════════════════════════════════════════════════════════════
# DSP steps
# ══════════════════════════════════════════════════════════════════════════════

def _highpass(audio: np.ndarray, sr: int, cutoff_hz: float) -> np.ndarray:
    """Zero-phase Butterworth high-pass.

    Second-order sections keep the filter stable at high sample rates, where
    the transfer-function form loses precision for an 80 Hz cutoff, and the
    padding is sized to the filter's settling time rather than its tap count
    so the first and last milliseconds are not disturbed.
    """
    from scipy.signal import butter, sosfiltfilt

    sos = butter(_HIGHPASS_ORDER, cutoff_hz / (sr / 2.0), btype="high", output="sos")
    padlen = min(len(audio) - 1, int(sr * _HIGHPASS_PAD_SECONDS))
    return sosfiltfilt(sos, audio, padlen=padlen)


def _noise_fft_size(sr: int) -> int:
    """Power-of-two FFT size of about 20-40 ms at any sample rate."""
    return 1 << max(8, int(math.ceil(math.log2(sr * _NOISE_FFT_SECONDS))))


def _noise_profile(audio: np.ndarray, sr: int, stats: _LevelStats) -> np.ndarray | None:
    """Collect the clip's noise-only samples, or None if there are too few.

    Only stretches of at least 100 ms that sit at the noise floor count, and
    50 ms next to speech is left out, so breaths, word tails and reverb do
    not end up in the profile and get treated as noise.
    """
    near_speech = _dilate(stats.voiced, _NOISE_PROFILE_GUARD_FRAMES, _NOISE_PROFILE_GUARD_FRAMES)
    at_floor = stats.frame_db <= stats.noise_floor_db + _NOISE_MARGIN_DB
    hop = stats.hop
    parts = [
        audio[start * hop:min(end * hop, len(audio))]
        for start, end in _runs(at_floor & ~near_speech)
        if end - start >= _NOISE_PROFILE_MIN_RUN_FRAMES
    ]
    if not parts:
        return None
    noise = np.concatenate(parts)
    if len(noise) < _NOISE_PROFILE_MIN_SECONDS * sr or not np.any(noise):
        return None
    return noise


def _reduce_noise(
    audio: np.ndarray, sr: int, noise: np.ndarray, strength: float, n_fft: int
) -> np.ndarray:
    """Stationary spectral gating against a measured noise profile.

    Letting noisereduce estimate the noise from the whole clip (its default)
    sets the threshold from the speech itself and attenuates quiet speech;
    with a profile taken from the pauses only sound at the noise level is
    touched. The mask is smoothed over 200 Hz rather than the library's
    500 Hz: the wider smoothing averages each harmonic with the noise bins
    next to it and turns the voice itself down by 0.5-1 dB, while 200 Hz
    is still wide enough to keep the residue free of musical noise at the
    strengths this pipeline uses.
    """
    import noisereduce as nr

    # noisereduce rejects a smoothing width below one of its frequency steps.
    smooth_hz = max(_NOISE_FREQ_SMOOTH_HZ, 2.0 * sr / n_fft)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cleaned = nr.reduce_noise(
            y=audio, sr=sr, y_noise=noise,
            stationary=True, prop_decrease=strength, n_fft=n_fft,
            freq_mask_smooth_hz=smooth_hz,
        )
    cleaned = np.asarray(cleaned, dtype=np.float64)
    if cleaned.shape != audio.shape or not np.all(np.isfinite(cleaned)):
        raise RuntimeError("noise reduction returned invalid audio")
    return cleaned


def _dilate(mask: np.ndarray, before: int, after: int) -> np.ndarray:
    """Extend every True by ``before`` positions earlier and ``after`` later."""
    n = len(mask)
    cum = np.concatenate(([0], np.cumsum(mask.astype(np.int64))))
    idx = np.arange(n)
    lo = np.clip(idx - after, 0, n)
    hi = np.clip(idx + before + 1, 0, n)
    return (cum[hi] - cum[lo]) > 0


def _noise_gate(
    audio: np.ndarray, sr: int, threshold_db: float, range_db: float
) -> np.ndarray:
    """Downward expander with look-ahead, hold and attack / release smoothing.

    The level detector is a 10 ms RMS measured relative to the clip's robust
    peak, so a quiet take and a loud take of the same performance are gated
    identically. Below the threshold the signal is turned down by at most
    ``range_db``; it is never muted, so the room tone between words stays
    continuous and soft consonants keep their place.
    """
    n = len(audio)
    if n == 0 or range_db <= 0.0:
        return audio
    ref = _robust_peak(_frame_levels(audio, _frame_hop(sr))[1])
    if ref <= 0.0:
        return audio

    hop = max(1, int(round(sr * _GATE_CONTROL_MS / 1000.0)))
    win = max(hop, int(round(sr * _GATE_DETECTOR_MS / 1000.0)))
    cum = np.concatenate(([0.0], np.cumsum(audio ** 2)))
    centres = np.arange(0, n, hop)
    lo = np.clip(centres - win // 2, 0, n)
    hi = np.clip(centres + win // 2 + 1, 0, n)
    power = (cum[hi] - cum[lo]) / np.maximum(hi - lo, 1)
    level_db = 10.0 * np.log10(np.maximum(power, 1e-30) / (ref * ref))

    per_ms = 1000.0 / sr * hop   # milliseconds per control step
    is_open = _dilate(
        level_db > threshold_db,
        before=int(round(_GATE_LOOKAHEAD_MS / per_ms)),
        after=int(round(_GATE_HOLD_MS / per_ms)),
    )
    target = np.where(is_open, 0.0, -abs(range_db))

    attack = math.exp(-per_ms / _GATE_ATTACK_MS)
    release = math.exp(-per_ms / _GATE_RELEASE_MS)
    gain_db = np.empty(len(target))
    current = float(target[0])
    for i, goal in enumerate(target.tolist()):
        coeff = attack if goal > current else release
        current = goal + (current - goal) * coeff
        gain_db[i] = current

    gain = 10.0 ** (np.interp(np.arange(n), centres, gain_db) / 20.0)
    return audio * gain


def _fade_edges(audio: np.ndarray, fade: int) -> np.ndarray:
    """Raised-cosine fade in and out so the clip starts and ends at zero."""
    fade = min(fade, len(audio) // 2)
    if fade < 1:
        return audio
    ramp = 0.5 - 0.5 * np.cos(np.pi * np.arange(fade) / fade)
    out = audio.copy()
    out[:fade] *= ramp
    out[-fade:] *= ramp[::-1]
    return out


def _remove_silence(
    audio: np.ndarray,
    sr: int,
    threshold_db: float,
    min_segment_ms: int,
    max_silence_kept_ms: int,
    *,
    trim_edges: bool = True,
    shorten_pauses: bool = True,
    edge_silence_ms: float | None = None,
    segments: list[tuple[int, int]] | None = None,
) -> np.ndarray:
    """Trim the ends and / or shorten long pauses without touching speech.

    Parameters
    ----------
    audio : np.ndarray
        Mono signal.
    sr : int
        Sample rate in Hz.
    threshold_db : float
        Speech threshold in dB relative to the clip's peak.
    min_segment_ms : int
        Isolated bursts shorter than this are clicks and count as silence.
    max_silence_kept_ms : int
        Pauses longer than this are shortened to it (floor: 80 ms). The
        first half is kept after the preceding word and the second half
        before the next one, joined by a 10 ms crossfade inside the silence.
    trim_edges : bool
        Cut leading and trailing silence down to ``edge_silence_ms``.
    shorten_pauses : bool
        Shorten internal pauses.
    edge_silence_ms : float, optional
        Silence kept at each end; half of the kept pause when None.
    segments : list[tuple[int, int]], optional
        Speech sample ranges from ``_find_speech`` when already computed.

    Returns
    -------
    np.ndarray
        The edited signal; the input itself when no speech was found.
    """
    if segments is None:
        segments = _find_speech(audio, sr, threshold_db, min_segment_ms)
    if not segments:
        return audio

    n = len(audio)
    keep = int(sr * max(float(max_silence_kept_ms), _MIN_PAUSE_KEPT_MS) / 1000.0)
    if edge_silence_ms is None:
        edge = keep // 2
    else:
        edge = int(sr * max(0.0, float(edge_silence_ms)) / 1000.0)
    start = max(0, segments[0][0] - edge) if trim_edges else 0
    end = min(n, segments[-1][1] + edge) if trim_edges else n

    if not shorten_pauses or len(segments) < 2:
        return audio[start:end]

    xfade = min(int(sr * _CROSSFADE_MS / 1000.0), keep // 2)
    pieces: list[np.ndarray] = []
    overlaps: list[int] = []
    cursor = start
    for (_, prev_end), (next_start, _) in zip(segments[:-1], segments[1:]):
        gap = next_start - prev_end
        if gap <= keep:
            continue
        # Keep ``keep`` samples of the pause in total: half after the word
        # that ends, half before the word that starts. The crossfade lies in
        # the middle of what is kept, so it never reaches speech.
        overlap = max(0, min(xfade, gap - keep))
        half = (keep + overlap) // 2
        pieces.append(audio[cursor:prev_end + half])
        overlaps.append(overlap)
        cursor = next_start - half
    pieces.append(audio[cursor:end])
    return _crossfade_concat(pieces, overlaps)


def _crossfade_concat(pieces: list[np.ndarray], overlaps: list[int]) -> np.ndarray:
    """Join ``pieces``, overlapping each joint by ``overlaps[i]`` samples."""
    out = pieces[0].astype(np.float64, copy=True)
    for piece, overlap in zip(pieces[1:], overlaps):
        overlap = min(overlap, len(out), len(piece))
        if overlap > 0:
            ramp = (np.arange(overlap) + 0.5) / overlap
            mixed = out[-overlap:] * (1.0 - ramp) + piece[:overlap] * ramp
            out = np.concatenate((out[:-overlap], mixed, piece[overlap:]))
        else:
            out = np.concatenate((out, piece))
    return out


def _best_window(
    stats: _LevelStats,
    n_samples: int,
    window: int,
    pad: int,
    min_frames: int,
    clipped_mask: np.ndarray | None,
) -> tuple[int, int] | None:
    """Pick the ``window``-sample stretch with the most clean speech.

    Candidate windows start just before a phrase begins and end after the
    last phrase that fits, so neither edge lands inside a word. Each is
    scored by how much of it is speech, with a heavy penalty for clipped
    samples and a small bonus for louder (closer, cleaner) speech.

    Returns
    -------
    tuple[int, int] or None
        ``(start, end)`` in samples, or None when the clip is not longer than
        the window or holds no speech.
    """
    if window <= 0 or n_samples <= window:
        return None
    segments = _speech_segments(stats.voiced, min_frames)
    if not segments:
        return None

    hop = stats.hop
    n_frames = len(stats.voiced)
    voiced_cum = np.concatenate(([0], np.cumsum(stats.voiced.astype(np.int64))))
    power_cum = np.concatenate(([0.0], np.cumsum(np.where(stats.voiced, stats.frame_rms ** 2, 0.0))))
    clip_cum = None
    if clipped_mask is not None and len(clipped_mask) == n_samples and clipped_mask.any():
        clip_cum = np.concatenate(([0], np.cumsum(clipped_mask.astype(np.int64))))

    ends = [min(n_samples, e * hop + pad) for _, e in segments]
    best: tuple[int, int] | None = None
    best_score = -math.inf
    for index, (seg_start, _) in enumerate(segments):
        start = max(0, seg_start * hop - pad)
        limit = min(n_samples, start + window)
        fitting = [e for e in ends[index:] if e <= limit]
        if fitting:
            end = fitting[-1]
        else:
            # One phrase is longer than the window: cut at its quietest
            # moment near the end rather than at an arbitrary sample.
            f_hi = min(n_frames, limit // hop)
            f_lo = max(seg_start + 1, f_hi - int(_WINDOW_CUT_SEARCH_SECONDS * 1000.0 / _FRAME_MS))
            if f_hi > f_lo:
                end = (f_lo + int(np.argmin(stats.frame_db[f_lo:f_hi])) + 1) * hop
            else:
                end = limit
            end = min(end, n_samples)
        if end <= start:
            continue

        f0, f1 = start // hop, min(n_frames, -(-end // hop))
        speech_frames = int(voiced_cum[f1] - voiced_cum[f0])
        if speech_frames == 0:
            continue
        score = speech_frames * hop / float(window)
        level_db = 10.0 * math.log10(
            max(float(power_cum[f1] - power_cum[f0]) / speech_frames, 1e-30)
        )
        score += _WINDOW_LEVEL_WEIGHT * float(np.clip(level_db - stats.speech_db, -12.0, 12.0))
        if clip_cum is not None:
            score -= _WINDOW_CLIP_PENALTY * float(clip_cum[end] - clip_cum[start]) / (end - start)
        if score > best_score + 1e-9:
            best_score, best = score, (start, end)
    return best


def _resample(audio: np.ndarray, sr: int, target_sr: int) -> np.ndarray:
    """High-quality band-limited resampling (soxr VHQ, else scipy polyphase).

    The scipy path designs its own anti-alias filter: ``resample_poly``'s
    default is only 20 taps per phase and lets content just above the new
    Nyquist fold back at about -10 dB.
    """
    try:
        import soxr
        return np.asarray(soxr.resample(audio, sr, target_sr, quality="VHQ"), dtype=np.float64)
    except ImportError:
        from scipy.signal import firwin, resample_poly

        divisor = math.gcd(int(sr), int(target_sr))
        up, down = int(target_sr) // divisor, int(sr) // divisor
        rate = max(up, down)
        taps = firwin(
            2 * _RESAMPLE_HALF_TAPS * rate + 1,
            _RESAMPLE_PASSBAND / rate,
            window=("kaiser", _RESAMPLE_KAISER_BETA),
        )
        return resample_poly(audio, up, down, window=taps)


# ══════════════════════════════════════════════════════════════════════════════
# Level
# ══════════════════════════════════════════════════════════════════════════════

def _k_weighting_response(freqs: np.ndarray, sr: int) -> np.ndarray:
    """Frequency response of the BS.1770 K-weighting pre-filter at ``freqs``."""
    z = np.exp(-2j * np.pi * freqs / sr)

    def biquad(b: tuple[float, float, float], a: tuple[float, float, float]) -> np.ndarray:
        return (b[0] + b[1] * z + b[2] * z * z) / (a[0] + a[1] * z + a[2] * z * z)

    # High shelf: +4 dB above 1.5 kHz, Q = 1/sqrt(2)
    amp = 10.0 ** (4.0 / 40.0)
    w0 = 2.0 * math.pi * 1500.0 / sr
    alpha = math.sin(w0) / (2.0 / math.sqrt(2.0))
    cos_w0, root = math.cos(w0), 2.0 * math.sqrt(amp) * alpha
    shelf = biquad(
        (amp * ((amp + 1) + (amp - 1) * cos_w0 + root),
         -2 * amp * ((amp - 1) + (amp + 1) * cos_w0),
         amp * ((amp + 1) + (amp - 1) * cos_w0 - root)),
        ((amp + 1) - (amp - 1) * cos_w0 + root,
         2 * ((amp - 1) - (amp + 1) * cos_w0),
         (amp + 1) - (amp - 1) * cos_w0 - root),
    )
    # High-pass: 38 Hz, Q = 0.5
    w0 = 2.0 * math.pi * 38.0 / sr
    alpha = math.sin(w0) / (2.0 * 0.5)
    cos_w0 = math.cos(w0)
    highpass = biquad(
        ((1 + cos_w0) / 2, -(1 + cos_w0), (1 + cos_w0) / 2),
        (1 + alpha, -2 * cos_w0, 1 - alpha),
    )
    return shelf * highpass


def _loudness_numpy(audio: np.ndarray, sr: int) -> float:
    """ITU-R BS.1770 integrated loudness using numpy only.

    The K-weighting is applied in the frequency domain, then 400 ms blocks
    with 75 % overlap are gated at -70 LUFS absolute and -10 LU relative.
    Clips shorter than one block are measured ungated.
    """
    n = len(audio)
    if n == 0:
        return -math.inf
    size = 1 << int(math.ceil(math.log2(max(n, 2))))
    spectrum = np.fft.rfft(audio, size)
    spectrum *= _k_weighting_response(np.fft.rfftfreq(size, 1.0 / sr), sr)
    weighted = np.fft.irfft(spectrum, size)[:n]

    block = int(round(_LOUDNESS_BLOCK_SECONDS * sr))
    if n < block:
        mean_square = np.array([float(np.mean(weighted ** 2))])
    else:
        step = max(1, int(round(block * (1.0 - _LOUDNESS_BLOCK_OVERLAP))))
        cum = np.concatenate(([0.0], np.cumsum(weighted ** 2)))
        starts = np.arange(0, n - block + 1, step)
        mean_square = (cum[starts + block] - cum[starts]) / block

    with np.errstate(divide="ignore"):
        block_lufs = _LOUDNESS_OFFSET + 10.0 * np.log10(mean_square)
    gated = mean_square[block_lufs > _LOUDNESS_ABSOLUTE_GATE_LUFS]
    if gated.size == 0:
        return -math.inf
    relative = _LOUDNESS_OFFSET + 10.0 * math.log10(float(gated.mean())) + _LOUDNESS_RELATIVE_GATE_LU
    gated = mean_square[(block_lufs > _LOUDNESS_ABSOLUTE_GATE_LUFS) & (block_lufs > relative)]
    if gated.size == 0:
        return -math.inf
    return _LOUDNESS_OFFSET + 10.0 * math.log10(float(gated.mean()))


def _integrated_loudness(audio: np.ndarray, sr: int) -> float:
    """Integrated loudness in LUFS; ``-inf`` for silence.

    Uses pyloudnorm when it is installed and the clip is at least one
    400 ms block long, otherwise the numpy implementation above.
    """
    if audio.size == 0 or not np.any(audio):
        return -math.inf
    if len(audio) >= int(round(_LOUDNESS_BLOCK_SECONDS * sr)) + 1:
        try:
            import pyloudnorm

            with warnings.catch_warnings(), np.errstate(divide="ignore", invalid="ignore"):
                warnings.simplefilter("ignore")
                value = float(pyloudnorm.Meter(sr).integrated_loudness(audio))
            return value if not math.isnan(value) else -math.inf
        except ImportError:
            pass
        except Exception as exc:   # fall back rather than lose the level step
            logger.debug("[Preprocess] pyloudnorm failed (%s); using numpy loudness.", exc)
    return _loudness_numpy(audio, sr)


def _true_peak(audio: np.ndarray, sr: int) -> float:
    """Inter-sample peak from 4x oversampling (sample peak without scipy)."""
    if audio.size == 0:
        return 0.0
    peak = float(np.max(np.abs(audio)))
    if peak <= 0.0 or audio.size < 8:
        return peak
    try:
        from scipy.signal import resample_poly
    except ImportError:
        return peak
    factor = 4 if sr < 88200 else 2
    for start in range(0, len(audio), _TRUE_PEAK_CHUNK):
        chunk = audio[max(0, start - _TRUE_PEAK_OVERLAP):start + _TRUE_PEAK_CHUNK + _TRUE_PEAK_OVERLAP]
        peak = max(peak, float(np.max(np.abs(resample_poly(chunk, factor, 1)))))
    return peak


def _normalize_loudness(
    audio: np.ndarray, sr: int, target_lufs: float, ceiling_dbfs: float
) -> tuple[np.ndarray, float | None, bool]:
    """Apply one static gain to reach ``target_lufs`` under a true-peak ceiling.

    The dynamics are never altered: when the target would push the true peak
    above ``ceiling_dbfs`` the gain is lowered instead of limiting.

    Returns
    -------
    tuple[np.ndarray, float | None, bool]
        The scaled signal, the gain applied in dB (None for silence, signal
        unchanged) and whether the ceiling reduced the gain.
    """
    loudness = _integrated_loudness(audio, sr)
    if not math.isfinite(loudness):
        return audio, None, False
    gain_db = float(target_lufs) - loudness
    headroom_db = float(ceiling_dbfs) - _db(_true_peak(audio, sr))
    limited = gain_db > headroom_db
    if limited:
        gain_db = headroom_db
    return audio * 10.0 ** (gain_db / 20.0), gain_db, limited


def _normalize_peak(audio: np.ndarray, target_dbfs: float) -> np.ndarray:
    """Scale so the sample peak sits at ``target_dbfs`` (legacy "peak" mode)."""
    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak == 0.0:
        return audio
    return audio * (10.0 ** (target_dbfs / 20.0) / peak)
