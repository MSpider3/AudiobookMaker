"""
test_voice_preprocessor.py
==========================
Unit tests for the voice cleaning pipeline's public contract and cache
persistence, run on the repository's reference fixture.

Signal-level behaviour (onsets, gating, loudness, resampling, the analysis
report) is covered in ``test_voice_preprocessor_dsp.py``.
"""

from __future__ import annotations

import io
import os
import sys
import pytest
import soundfile as sf

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import audiobook_factory.voice_preprocessor as voice_preprocessor
from audiobook_factory.voice_preprocessor import (
    PreprocessConfig,
    VoiceReport,
    analyze_voice,
    preprocess,
    preprocess_with_report,
    _get_cache_path,
    _read_cache,
    _write_cache,
)
from tests.audio_validator import AudioValidator


@pytest.fixture(autouse=True)
def isolated_cache(tmp_path, monkeypatch):
    """Point the cache at a temp directory so tests never touch the real one."""
    cache_dir = tmp_path / "voice_cache"
    monkeypatch.setenv("ABM_VOICE_CACHE_DIR", str(cache_dir))
    return cache_dir


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

    def test_cache_lives_in_the_configured_directory(self, sample_wav_bytes, isolated_cache):
        cfg = PreprocessConfig()

        preprocess(sample_wav_bytes, cfg)

        assert os.path.dirname(_get_cache_path(sample_wav_bytes, cfg)) == str(isolated_cache)
        assert len(os.listdir(isolated_cache)) == 1

    def test_cache_key_changes_with_pipeline_version(self, sample_wav_bytes, monkeypatch):
        cfg = PreprocessConfig()
        before = _get_cache_path(sample_wav_bytes, cfg)

        monkeypatch.setattr(
            voice_preprocessor, "_PIPELINE_VERSION", voice_preprocessor._PIPELINE_VERSION + 1
        )

        assert _get_cache_path(sample_wav_bytes, cfg) != before

    def test_write_then_read_cache_round_trips(self, sample_wav_bytes, isolated_cache):
        path = _get_cache_path(sample_wav_bytes, PreprocessConfig())

        assert _read_cache(path) is None
        _write_cache(path, sample_wav_bytes)

        assert _read_cache(path) == sample_wav_bytes
        assert os.listdir(isolated_cache) == [os.path.basename(path)], "temp file left behind"

    def test_output_is_16_bit_mono_wav(self, sample_wav_bytes):
        info = sf.info(io.BytesIO(preprocess(sample_wav_bytes, use_cache=False)))

        assert info.format == "WAV" and info.subtype == "PCM_16"
        assert info.channels == 1
        assert info.samplerate == 24000

    def test_report_api_matches_plain_preprocess(self, sample_wav_bytes):
        cfg = PreprocessConfig()

        wav_bytes, report = preprocess_with_report(sample_wav_bytes, cfg, use_cache=False)

        assert wav_bytes == preprocess(sample_wav_bytes, cfg, use_cache=False)
        assert isinstance(report, VoiceReport)
        assert isinstance(report.source, VoiceReport)
        assert report.source.duration_s == pytest.approx(4.0, abs=0.01)
        assert report.loudness_lufs == pytest.approx(cfg.loudness_target_lufs, abs=0.5)
        # The fixture is four seconds of unbroken tone: short, and with no
        # pause to tell signal from noise.
        assert any("only" in w for w in report.warnings)

    def test_analyze_voice_accepts_bytes_and_paths(self, sample_wav_bytes, tmp_path):
        path = tmp_path / "voice.wav"
        path.write_bytes(sample_wav_bytes)

        from_bytes = analyze_voice(sample_wav_bytes)
        from_path = analyze_voice(str(path))

        assert from_bytes == from_path
        assert from_bytes.sample_rate == 24000
        assert from_bytes.duration_s == pytest.approx(4.0, abs=0.01)
        assert from_bytes.source is None and from_bytes.steps_applied == []
