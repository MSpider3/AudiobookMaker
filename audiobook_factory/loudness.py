"""
loudness.py
===========
Loudness normalisation for finished chapters, with a look-ahead peak limiter.

Narration from a TTS model has a wide gap between its loudest peaks and its
average level, so one static gain usually cannot reach the loudness target
without pushing peaks over the true-peak ceiling. The old behaviour was to
lower the gain until the peaks fitted, which left chapters several LU quieter
than asked for. Here the full gain is applied and the few peaks that would
cross the ceiling are turned down by a limiter instead.

The limiter works on a gain curve sampled once per millisecond:

1. the gain each block needs to stay under the ceiling,
2. a centred sliding minimum (the "hold"), so the gain is already down before
   a peak arrives and stays down across a whole pitch period,
3. two box filters, so the gain never steps (no clicks),
4. linear interpolation back to one gain per sample.

Because the hold is wider than the smoothing, the gain at every sample is at
or below what that sample needs: the output cannot exceed the ceiling.

``audiobook_rust/src/audio/master.rs`` implements the same algorithm with the
same constants; this module is the pure-Python path used when the extension is
not built.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)

# ── Limiter shape (keep in step with master.rs) ──────────────────────────────

_LIMITER_BLOCK_SECONDS: float = 0.001   # one gain value per millisecond
_LIMITER_HOLD_BLOCKS: int = 8           # sliding minimum over +/- 8 ms
_LIMITER_SMOOTH_BLOCKS: int = 3         # box filter over +/- 3 ms, applied twice
_LIMITER_SMOOTH_PASSES: int = 2
# The limiter controls sample peaks; inter-sample peaks sit a little higher.
_LIMITER_MARGIN_DB: float = 0.3
# Below this shortfall the gain is simply capped, as before: not worth limiting.
_LIMITER_MIN_SHORTFALL_DB: float = 0.5
# Never push peaks further than this into the limiter; beyond it the audio
# would sound squashed, so the chapter is left a little under target instead.
_MAX_LIMITER_REDUCTION_DB: float = 9.0
# Limiting removes a little loudness; correct once if it is more than this.
_LOUDNESS_RETRY_THRESHOLD_LU: float = 0.1

_TRUE_PEAK_CHUNK: int = 1 << 20
_TRUE_PEAK_OVERLAP: int = 64


@dataclass
class LoudnessResult:
    """What :func:`normalize_loudness` did to a signal.

    Attributes
    ----------
    input_lufs : float
        Integrated loudness before normalisation (``-inf`` for silence).
    output_lufs : float
        Integrated loudness of the returned signal.
    gain_db : float
        Static gain applied before limiting.
    limited : bool
        Whether the limiter was needed.
    reached_target : bool
        False when the peaks were too far above the average level to reach the
        target without audible limiting; the signal is then quieter than asked.
    """

    input_lufs: float
    output_lufs: float
    gain_db: float = 0.0
    limited: bool = False
    reached_target: bool = True


def _db(linear: float) -> float:
    return 20.0 * math.log10(max(float(linear), 1e-10))


def integrated_loudness(audio: np.ndarray, sr: int) -> float:
    """Integrated loudness in LUFS (EBU R128), ``-inf`` for silence.

    Raises
    ------
    ImportError
        If ``pyloudnorm`` is not installed.
    ValueError
        If the signal is shorter than one 400 ms measurement block.
    """
    import pyloudnorm

    value = float(pyloudnorm.Meter(int(sr)).integrated_loudness(audio))
    return -math.inf if math.isnan(value) else value


def true_peak(audio: np.ndarray, sr: int) -> float:
    """Inter-sample peak (linear) from 4x oversampling; sample peak without scipy."""
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


def limiter_gain(audio: np.ndarray, sr: int, ceiling: float) -> np.ndarray | None:
    """Per-sample gain that keeps *audio* at or below *ceiling*.

    Parameters
    ----------
    audio : np.ndarray
        Mono signal.
    sr : int
        Sample rate in Hz.
    ceiling : float
        Highest allowed absolute sample value (linear).

    Returns
    -------
    np.ndarray | None
        One gain per sample in ``(0, 1]``, or None when no sample is above the
        ceiling.
    """
    count = int(audio.size)
    if count == 0 or ceiling <= 0.0:
        return None
    block = max(1, int(round(sr * _LIMITER_BLOCK_SECONDS)))
    blocks = -(-count // block)
    magnitude = np.zeros(blocks * block, dtype=np.float32)
    magnitude[:count] = np.abs(audio)
    block_peak = magnitude.reshape(blocks, block).max(axis=1)
    if float(block_peak.max()) <= ceiling:
        return None

    required = np.minimum(1.0, ceiling / np.maximum(block_peak, 1e-12)).astype(np.float64)
    hold = _LIMITER_HOLD_BLOCKS
    windows = np.lib.stride_tricks.sliding_window_view(np.pad(required, hold, mode="edge"), 2 * hold + 1)
    curve = windows.min(axis=1)
    smooth = _LIMITER_SMOOTH_BLOCKS
    kernel = np.full(2 * smooth + 1, 1.0 / (2 * smooth + 1))
    for _ in range(_LIMITER_SMOOTH_PASSES):
        curve = np.convolve(np.pad(curve, smooth, mode="edge"), kernel, mode="valid")

    centres = (np.arange(blocks, dtype=np.float64) + 0.5) * block - 0.5
    return np.interp(np.arange(count, dtype=np.float64), centres, curve)


def limit_peaks(audio: np.ndarray, sr: int, ceiling: float) -> np.ndarray:
    """Returns *audio* with every sample at or below *ceiling* (linear)."""
    gain = limiter_gain(audio, sr, ceiling)
    if gain is None:
        return audio
    return (audio * gain).astype(audio.dtype, copy=False)


def normalize_loudness(
    audio: np.ndarray, sr: int, target_lufs: float, true_peak_db: float
) -> tuple[np.ndarray, LoudnessResult]:
    """Brings *audio* to *target_lufs* without exceeding *true_peak_db*.

    Parameters
    ----------
    audio : np.ndarray
        Mono signal.
    sr : int
        Sample rate in Hz.
    target_lufs : float
        Integrated loudness to reach.
    true_peak_db : float
        True-peak ceiling in dBTP.

    Returns
    -------
    tuple[np.ndarray, LoudnessResult]
        The normalised signal and a description of what was done. Silence is
        returned unchanged.

    Raises
    ------
    ImportError
        If ``pyloudnorm`` is not installed.
    ValueError
        If the signal is too short to measure.
    """
    loudness = integrated_loudness(audio, sr)
    if not math.isfinite(loudness):
        return audio, LoudnessResult(input_lufs=loudness, output_lufs=loudness)

    gain_db = float(target_lufs) - loudness
    headroom_db = float(true_peak_db) - _db(true_peak(audio, sr))
    shortfall_db = gain_db - headroom_db
    if shortfall_db <= _LIMITER_MIN_SHORTFALL_DB:
        gain_db = min(gain_db, headroom_db)
        scaled = (audio * 10.0 ** (gain_db / 20.0)).astype(audio.dtype, copy=False)
        return scaled, LoudnessResult(loudness, loudness + gain_db, gain_db)

    max_gain_db = headroom_db + _MAX_LIMITER_REDUCTION_DB
    reached = gain_db <= max_gain_db
    gain_db = min(gain_db, max_gain_db)
    ceiling = 10.0 ** ((float(true_peak_db) - _LIMITER_MARGIN_DB) / 20.0)

    limited = limit_peaks(audio * 10.0 ** (gain_db / 20.0), sr, ceiling)
    output_lufs = integrated_loudness(limited, sr)
    missing = float(target_lufs) - output_lufs
    if reached and math.isfinite(output_lufs) and missing > _LOUDNESS_RETRY_THRESHOLD_LU:
        # The limiter took a little loudness away; ask for that much more.
        gain_db = min(gain_db + missing, max_gain_db)
        limited = limit_peaks(audio * 10.0 ** (gain_db / 20.0), sr, ceiling)
        output_lufs = integrated_loudness(limited, sr)

    over_db = _db(true_peak(limited, sr)) - float(true_peak_db)
    if over_db > 0.0:
        limited = limited * 10.0 ** (-over_db / 20.0)
        output_lufs -= over_db
    limited = limited.astype(audio.dtype, copy=False)
    if not reached:
        logger.warning(
            "Chapter peaks are %.1f dB above what a %g LUFS target allows; it was "
            "mastered to %.1f LUFS to avoid audible limiting.",
            shortfall_db, target_lufs, output_lufs,
        )
    return limited, LoudnessResult(loudness, output_lufs, gain_db, True, reached)
