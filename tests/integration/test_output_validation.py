"""
test_output_validation.py
=========================
Integration tests for AudioValidator and SubtitleValidator.
"""

from __future__ import annotations

import io
import os
import sys
import tempfile
import numpy as np
import pytest
import soundfile as sf

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tests.audio_validator import AudioValidator, SubtitleValidator


class TestOutputValidation:

    def test_validate_valid_audio_file(self):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            t = np.linspace(0, 1.0, 24000, dtype=np.float32)
            sig = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
            sf.write(tf.name, sig, 24000, subtype="PCM_16")

        try:
            res = AudioValidator.validate_audio_file(tf.name, expected_sample_rate=24000)
            assert res.is_valid
            assert not res.is_silent
            assert not res.has_nan
            assert abs(res.duration_sec - 1.0) < 0.05
        finally:
            os.remove(tf.name)

    def test_validate_detects_silence(self):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            zeros = np.zeros(24000, dtype=np.float32)
            sf.write(tf.name, zeros, 24000, subtype="PCM_16")

        try:
            res = AudioValidator.validate_audio_file(tf.name)
            assert not res.is_valid
            assert res.is_silent
            assert "silent" in res.error_message.lower()
        finally:
            os.remove(tf.name)

    def test_validate_detects_nan(self):
        nan_arr = np.ones(1000, dtype=np.float32)
        nan_arr[50] = np.nan
        with io.BytesIO() as buf:
            sf.write(buf, nan_arr, 24000, format="WAV", subtype="FLOAT")
            wav_bytes = buf.getvalue()

        res = AudioValidator.validate_audio_bytes(wav_bytes)
        assert not res.is_valid
        assert res.has_nan

    def test_validate_lrc_monotonic(self):
        with tempfile.NamedTemporaryFile(suffix=".lrc", mode="w", delete=False) as tf:
            tf.write("[00:00.00] Line one\n[00:02.50] Line two\n[00:05.10] Line three\n")

        try:
            ok, err = SubtitleValidator.validate_lrc(tf.name)
            assert ok, err
        finally:
            os.remove(tf.name)

    def test_validate_lrc_non_monotonic_fails(self):
        with tempfile.NamedTemporaryFile(suffix=".lrc", mode="w", delete=False) as tf:
            tf.write("[00:05.00] Line one\n[00:02.00] Out of order line\n")

        try:
            ok, err = SubtitleValidator.validate_lrc(tf.name)
            assert not ok
            assert "Non-monotonic" in err
        finally:
            os.remove(tf.name)
