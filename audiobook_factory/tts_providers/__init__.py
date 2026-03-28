"""
audiobook_factory/tts_providers/__init__.py
"""
from audiobook_factory.tts_providers.base_tts_provider import BaseTTSProvider, get_tts_provider

__all__ = ["BaseTTSProvider", "get_tts_provider"]
