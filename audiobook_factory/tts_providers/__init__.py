"""
audiobook_factory/tts_providers
================================
Pluggable TTS backends. Provider classes are imported on first use so that
importing this package never pulls in a model library.
"""
from __future__ import annotations

from audiobook_factory.tts_providers.base_tts_provider import (
    BaseTTSProvider,
    ProviderInfo,
    ProviderOption,
    get_tts_provider,
)
from audiobook_factory.tts_providers.registry import (
    apply_recommended_settings,
    canonical_name,
    is_known_provider,
    list_providers,
    provider_class,
    provider_info,
    provider_names,
)

__all__ = [
    "BaseTTSProvider",
    "ProviderInfo",
    "ProviderOption",
    "get_tts_provider",
    "apply_recommended_settings",
    "canonical_name",
    "is_known_provider",
    "list_providers",
    "provider_class",
    "provider_info",
    "provider_names",
    "QwenTTSProvider",
    "F5TTSProvider",
]

_LAZY_CLASSES: dict[str, str] = {
    "QwenTTSProvider": "qwen",
    "F5TTSProvider": "f5tts",
}


def __getattr__(name: str):
    if name in _LAZY_CLASSES:
        return provider_class(_LAZY_CLASSES[name])
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
