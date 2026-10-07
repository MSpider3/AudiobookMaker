"""
test_loudness.py
================
Loudness normalisation with the look-ahead peak limiter
(``audiobook_factory/loudness.py`` and its Rust twin in ``master.rs``).
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import pytest
import soundfile as sf

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from audiobook_factory import loudness
from audiobook_factory.chapter_pipeline import _master_final
from audiobook_factory.pipeline import AudiobookConfig, _check_rust

_RATE: int = 24000
_TARGET_LUFS: float = -18.0
_CEILING_DB: float = -1.5


def _peaky_speech(seconds: float = 20.0, seed: int = 3) -> np.ndarray:
    """Speech-like noise bursts with sharp peaks: quiet on average, loud at the peaks.

    This is what TTS narration looks like to a loudness meter, and the case
    one static gain cannot bring to target.
    """
    rng = np.random.default_rng(seed)
    count = int(seconds * _RATE)
    t = np.arange(count) / _RATE
    syllables = np.clip(np.sin(2 * np.pi * 3.0 * t), 0.0, None) ** 2
    voiced = np.sin(2 * np.pi * 140.0 * t) + 0.5 * np.sin(2 * np.pi * 280.0 * t)
    signal = 0.06 * syllables * (voiced + 0.3 * rng.standard_normal(count))
    # A plosive-like peak about once a second, well above the syllable level
    # but too short to carry the loudness.
    for position in rng.integers(_RATE, count - _RATE, size=int(seconds)):
        burst = np.hanning(48) * rng.choice([-1.0, 1.0]) * 0.3
        signal[position:position + 48] += burst
    return signal.astype(np.float32)


def _steady_tone(seconds: float = 5.0, amplitude: float = 0.05) -> np.ndarray:
    t = np.arange(int(seconds * _RATE)) / _RATE
    return (amplitude * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)


class TestLimiter:

    def test_output_never_exceeds_the_ceiling(self):
        audio = _peaky_speech() * 4.0
        ceiling = 0.5
        limited = loudness.limit_peaks(audio, _RATE, ceiling)
        assert float(np.abs(audio).max()) > ceiling
        assert float(np.abs(limited).max()) <= ceiling + 1e-6

    def test_audio_below_the_ceiling_is_returned_untouched(self):
        audio = _steady_tone(amplitude=0.2)
        assert loudness.limiter_gain(audio, _RATE, 0.5) is None
        assert loudness.limit_peaks(audio, _RATE, 0.5) is audio

    def test_gain_moves_smoothly_and_only_near_peaks(self):
        audio = np.zeros(_RATE, dtype=np.float32)
        audio[:] = 0.1
        audio[_RATE // 2] = 1.0  # one click in the middle
        gain = loudness.limiter_gain(audio, _RATE, 0.5)
        assert gain is not None
        assert gain[_RATE // 2] <= 0.5 + 1e-9
        # No step between neighbouring samples (a step is an audible click).
        assert float(np.abs(np.diff(gain)).max()) < 0.01
        # Fully released well before and after the peak.
        quiet = np.r_[gain[: _RATE // 2 - _RATE // 20], gain[_RATE // 2 + _RATE // 20:]]
        assert float(quiet.min()) == pytest.approx(1.0)

    def test_empty_signal(self):
        empty = np.zeros(0, dtype=np.float32)
        assert loudness.limiter_gain(empty, _RATE, 0.5) is None


class TestNormalizeLoudness:

    def test_peaky_narration_reaches_the_target(self):
        audio = _peaky_speech()
        before = loudness.integrated_loudness(audio, _RATE)
        headroom = _CEILING_DB - 20 * math.log10(loudness.true_peak(audio, _RATE))
        # The premise: a static gain alone would stop well short of the target.
        assert _TARGET_LUFS - before > headroom + 2.0

        out, result = loudness.normalize_loudness(audio, _RATE, _TARGET_LUFS, _CEILING_DB)
        assert result.limited and result.reached_target
        assert loudness.integrated_loudness(out, _RATE) == pytest.approx(_TARGET_LUFS, abs=0.5)
        assert result.output_lufs == pytest.approx(loudness.integrated_loudness(out, _RATE), abs=0.2)
        assert 20 * math.log10(loudness.true_peak(out, _RATE)) <= _CEILING_DB + 0.05
        assert out.dtype == audio.dtype and out.shape == audio.shape

    def test_signal_with_headroom_gets_one_static_gain(self):
        audio = _steady_tone()
        out, result = loudness.normalize_loudness(audio, _RATE, _TARGET_LUFS, _CEILING_DB)
        assert not result.limited
        assert loudness.integrated_loudness(out, _RATE) == pytest.approx(_TARGET_LUFS, abs=0.2)
        ratio = out[1000:2000] / audio[1000:2000]
        assert float(ratio.max() - ratio.min()) < 1e-4  # same gain everywhere

    def test_loud_signal_is_turned_down(self):
        audio = _steady_tone(amplitude=0.8)
        out, result = loudness.normalize_loudness(audio, _RATE, _TARGET_LUFS, _CEILING_DB)
        assert result.gain_db < 0 and not result.limited
        assert loudness.integrated_loudness(out, _RATE) == pytest.approx(_TARGET_LUFS, abs=0.2)

    def test_silence_is_left_alone(self):
        audio = np.zeros(_RATE * 2, dtype=np.float32)
        out, result = loudness.normalize_loudness(audio, _RATE, _TARGET_LUFS, _CEILING_DB)
        assert out is audio and not result.limited
        assert math.isinf(result.input_lufs)

    def test_extreme_peaks_are_not_squashed_to_reach_the_target(self):
        # A near-silent recording with one full-scale click: reaching the
        # target would need more than _MAX_LIMITER_REDUCTION_DB of limiting.
        audio = _steady_tone(amplitude=0.002, seconds=6.0)
        audio[_RATE] = 0.9
        out, result = loudness.normalize_loudness(audio, _RATE, _TARGET_LUFS, _CEILING_DB)
        assert result.limited and not result.reached_target
        assert loudness.integrated_loudness(out, _RATE) < _TARGET_LUFS - 3.0
        assert 20 * math.log10(loudness.true_peak(out, _RATE)) <= _CEILING_DB + 0.05

    def test_a_second_pass_changes_nothing(self):
        # The MP3 fast path masters an already mastered chapter again.
        once, _ = loudness.normalize_loudness(_peaky_speech(), _RATE, _TARGET_LUFS, _CEILING_DB)
        twice, result = loudness.normalize_loudness(once, _RATE, _TARGET_LUFS, _CEILING_DB)
        assert not result.limited
        assert float(np.abs(twice - once).max()) < 0.02


class TestMasterFinal:
    """``_master_final`` on whichever backend this environment has, and on the fallback."""

    def _master(self, tmp_path, audio: np.ndarray) -> np.ndarray:
        source = tmp_path / "chapter_partial.wav"
        sf.write(source, audio, _RATE, subtype="FLOAT")
        out = tmp_path / "chapter_master.wav"
        config = AudiobookConfig(lufs=_TARGET_LUFS, true_peak=_CEILING_DB, sample_rate=_RATE)
        assert _master_final([str(source)], str(out), config) is True
        mastered, rate = sf.read(out, dtype="float32")
        assert rate == _RATE
        return mastered

    def _assert_on_target(self, mastered: np.ndarray) -> None:
        assert loudness.integrated_loudness(mastered, _RATE) == pytest.approx(_TARGET_LUFS, abs=0.6)
        assert 20 * math.log10(loudness.true_peak(mastered, _RATE)) <= _CEILING_DB + 0.3

    def test_peaky_chapter_reaches_the_target(self, tmp_path):
        self._assert_on_target(self._master(tmp_path, _peaky_speech()))

    def test_python_fallback_reaches_the_target(self, tmp_path, monkeypatch):
        import audiobook_factory.chapter_pipeline as chapter_pipeline

        monkeypatch.setattr(chapter_pipeline, "_check_rust", lambda: False)
        self._assert_on_target(self._master(tmp_path, _peaky_speech()))

    @pytest.mark.skipif(not _check_rust(), reason="Rust extension not built")
    def test_rust_and_python_agree(self, tmp_path, monkeypatch):
        import audiobook_factory.chapter_pipeline as chapter_pipeline

        audio = _peaky_speech()
        (tmp_path / "rust").mkdir()
        (tmp_path / "python").mkdir()
        from_rust = self._master(tmp_path / "rust", audio)
        monkeypatch.setattr(chapter_pipeline, "_check_rust", lambda: False)
        from_python = self._master(tmp_path / "python", audio)
        count = min(len(from_rust), len(from_python))
        # Two loudness meters and 16-bit output: close, not bit-identical.
        assert float(np.abs(from_rust[:count] - from_python[:count]).max()) < 0.03
