"""
test_tts_provider_contract.py
==============================
Unit tests for TTS Provider contract conformance, factory resolution,
and input parameter validation.
"""

from __future__ import annotations

import os
import sys
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from audiobook_factory.pipeline import AudiobookConfig
from audiobook_factory.tts_providers.base_tts_provider import BaseTTSProvider, get_tts_provider
from tests.fixtures.mock_provider import MockTTSProvider
from tests.audio_validator import AudioValidator


class TestTTSProviderContract:

    @pytest.fixture
    def config(self):
        return AudiobookConfig(sample_rate=24000)

    @pytest.fixture
    def sample_voice_bytes(self):
        wav_path = os.path.join(_ROOT, "tests", "fixtures", "audio", "synthetic_voice_reference.wav")
        with open(wav_path, "rb") as f:
            return f.read()

    def test_mock_provider_implements_contract(self, config, sample_voice_bytes):
        provider = MockTTSProvider(config)
        assert isinstance(provider, BaseTTSProvider)
        assert provider.device == "cpu"
        assert provider.is_ready is True
        assert provider.get_name() == "mock"
        assert provider.estimate_cost(1000) == 0.0

        # Single synthesis
        audio, dur = provider.synthesize("Hello world", sample_voice_bytes, return_bytes=True)
        assert isinstance(audio, bytes)
        assert dur > 0.0
        val = AudioValidator.validate_audio_bytes(audio)
        assert val.is_valid

        # Batch synthesis
        texts = ["Sentence one.", "Sentence two.", "Sentence three."]
        batch_out = provider.synthesize_batch(texts, sample_voice_bytes, return_bytes=True)
        assert len(batch_out) == 3
        for b_audio, b_dur in batch_out:
            assert isinstance(b_audio, bytes)
            assert b_dur > 0.0
            assert AudioValidator.validate_audio_bytes(b_audio).is_valid

    def test_validate_voice_ref_guards(self, config):
        provider = MockTTSProvider(config)

        # None type
        with pytest.raises(TypeError):
            provider._validate_voice_ref(None)

        # Non-existent string path
        with pytest.raises(ValueError, match="does not exist"):
            provider._validate_voice_ref("/tmp/non_existent_voice_12345.wav")

        # Bytes too short
        with pytest.raises(ValueError, match="too short"):
            provider._validate_voice_ref(b"RIFFshort")

    def test_get_tts_provider_unknown_raises(self, config):
        with pytest.raises(ValueError, match="Unknown TTS provider"):
            get_tts_provider("non_existent_provider_xyz", config)

    def test_get_tts_provider_all_valid_names(self, config):
        from audiobook_factory.tts_providers.qwen_provider import QwenTTSProvider
        from audiobook_factory.tts_providers.vibevoice_provider import VibeVoiceTTSProvider
        from audiobook_factory.tts_providers.f5tts_provider import F5TTSProvider

        for name in ("qwen", "qwen3", "qwen3-tts", ""):
            p = get_tts_provider(name, config)
            assert isinstance(p, QwenTTSProvider)

        for name in ("vibevoice", "vibe-voice", "vibevoice-1.5b"):
            p = get_tts_provider(name, config)
            assert isinstance(p, VibeVoiceTTSProvider)

        for name in ("f5tts", "f5-tts", "f5_tts"):
            p = get_tts_provider(name, config)
            assert isinstance(p, F5TTSProvider)

    def test_f5tts_missing_dependency_raises_clean_message(self, config):
        p = get_tts_provider("f5tts", config)
        try:
            import f5_tts  # noqa: F401
            pytest.skip("f5-tts is installed in environment")
        except ImportError:
            with pytest.raises(RuntimeError, match="pip install f5-tts"):
                p.ensure_ready()

    def test_vibevoice_unapproved_model_raises(self):
        bad_cfg = AudiobookConfig(
            tts_provider_name="vibevoice",
            tts_model_name="evil-repo/unapproved-vibevoice",
        )
        p = get_tts_provider("vibevoice", bad_cfg)
        with pytest.raises(ValueError, match="Refusing to load unapproved VibeVoice model"):
            p.ensure_ready()
