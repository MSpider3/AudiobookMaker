"""Test VibeVoice TTS Provider model allowlist enforcement (BUG-R2-C1-A2-H8).

VibeVoice uses trust_remote_code=True, so only the curated upstream model id
'bezzam/VibeVoice-1.5B-hf' must be permitted. Any unapproved model repository
must be rejected with ValueError.
"""
import pytest
from unittest.mock import patch, MagicMock
from audiobook_factory.pipeline import AudiobookConfig
from audiobook_factory.tts_providers.vibevoice_provider import VibeVoiceTTSProvider


def test_vibevoice_rejects_unapproved_model():
    """Ensure an untrusted model repo name raises ValueError before loading."""
    cfg = AudiobookConfig(
        book_title="Test",
        tts_provider_name="vibevoice",
        tts_model_name="evil-actor/malicious-VibeVoice-exploit",
    )
    provider = VibeVoiceTTSProvider(config=cfg, device="cpu")
    
    with pytest.raises(ValueError, match="Refusing to load unapproved VibeVoice model"):
        provider._ensure_initialised()


def test_vibevoice_accepts_blessed_model():
    """Ensure the approved model id proceeds to initialization."""
    cfg = AudiobookConfig(
        book_title="Test",
        tts_provider_name="vibevoice",
        tts_model_name="bezzam/VibeVoice-1.5B-hf",
    )
    provider = VibeVoiceTTSProvider(config=cfg, device="cpu")
    
    # Mock transformers calls to verify it accepts the blessed model
    with patch("transformers.AutoProcessor.from_pretrained") as mock_proc, \
         patch("transformers.AutoModelForCausalLM.from_pretrained") as mock_model:
        mock_proc.return_value = MagicMock()
        mock_model.return_value = MagicMock()
        
        provider._ensure_initialised()
        assert provider._model is not None
