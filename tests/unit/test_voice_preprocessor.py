"""
test_voice_preprocessor.py
==========================
Unit tests for the 7-step DSP voice cleaning pipeline and cache persistence.
"""

from __future__ import annotations

import os
import sys
import pytest
import soundfile as sf

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from audiobook_factory.voice_preprocessor import (
    PreprocessConfig,
    preprocess,
    _get_cache_path,
    _read_cache,
    _write_cache,
)
from tests.audio_validator import AudioValidator


class TestVoicePreprocessor:

    @pytest.fixture
    def sample_wav_bytes(self):
        wav_path = os.path.join(_ROOT, "tests", "fixtures", "audio", "synthetic_voice_reference.wav")
        if not os.path.exists(wav_path):
            from tests.fixture_generation.generate_test_audio import generate_all_audio_fixtures
            generate_all_audio_fixtures()
        with open(wav_path, "rb") as f:
            return f.read()

    def test_preprocess_produces_valid_audio(self, sample_wav_bytes):
        cfg = PreprocessConfig(
            noise_reduce=True,
            noise_gate=True,
            highpass_filter=True,
            silence_removal=True,
            normalize_volume=True,
        )
        cleaned_bytes = preprocess(sample_wav_bytes, cfg)
        assert len(cleaned_bytes) > 1000

        res = AudioValidator.validate_audio_bytes(cleaned_bytes, min_duration=0.5)
        assert res.is_valid, f"Validation failed: {res.error_message}"
        assert res.rms > 0.01
        assert not res.is_silent
        assert not res.has_nan

    def test_preprocess_caching_behavior(self, sample_wav_bytes):
        cfg = PreprocessConfig(noise_reduce=False, normalize_volume=True)
        cache_path = _get_cache_path(sample_wav_bytes, cfg)

        # First run (creates or reads cache)
        out1 = preprocess(sample_wav_bytes, cfg)
        assert os.path.exists(cache_path)

        # Second run should return identical cached bytes
        out2 = preprocess(sample_wav_bytes, cfg)
        assert out1 == out2

    def test_cache_path_changes_with_config(self, sample_wav_bytes):
        cfg1 = PreprocessConfig(noise_reduce=True, noise_reduce_strength=0.3)
        cfg2 = PreprocessConfig(noise_reduce=True, noise_reduce_strength=0.8)

        path1 = _get_cache_path(sample_wav_bytes, cfg1)
        path2 = _get_cache_path(sample_wav_bytes, cfg2)
        assert path1 != path2
