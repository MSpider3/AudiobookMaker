"""
test_silent_provider_output.py
==============================
Regression tests for BUG-01 (VibeVoice silent output) and BUG-02 (F5-TTS error masking).
Ensures uninitialized or failing providers fail explicitly rather than producing zero-filled audio.
"""

from __future__ import annotations

import os
import sys
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from audiobook_factory.pipeline import AudiobookConfig
from audiobook_factory.tts_providers.vibevoice_provider import VibeVoiceTTSProvider
from audiobook_factory.tts_providers.f5tts_provider import F5TTSProvider


class TestSilentProviderOutputRegression:

    @pytest.fixture
    def sample_voice_bytes(self):
        wav_path = os.path.join(_ROOT, "tests", "fixtures", "audio", "synthetic_voice_reference.wav")
        with open(wav_path, "rb") as f:
            return f.read()

    def test_vibevoice_raises_when_uninitialized(self, sample_voice_bytes):
        cfg = AudiobookConfig()
        provider = VibeVoiceTTSProvider(cfg)
        with pytest.raises(RuntimeError, match="VibeVoice"):
            provider.synthesize("Test sentence for vibevoice.", sample_voice_bytes)

    def test_f5tts_raises_when_uninitialized(self, sample_voice_bytes):
        cfg = AudiobookConfig()
        provider = F5TTSProvider(cfg)
        with pytest.raises(RuntimeError, match="F5-TTS"):
            provider.synthesize("Test sentence for f5tts.", sample_voice_bytes)
