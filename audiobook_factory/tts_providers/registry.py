"""
audiobook_factory/tts_providers/registry.py
============================================
Single source of truth for which TTS providers exist.

Provider modules are imported lazily, so listing providers (for the UI
dropdown, ``--help`` text or settings validation) never loads torch or any
model library.
"""
from __future__ import annotations

import importlib
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from audiobook_factory.tts_providers.base_tts_provider import BaseTTSProvider, ProviderInfo

logger = logging.getLogger(__name__)

# canonical name → (module, class name)
_PROVIDERS: dict[str, tuple[str, str]] = {
    "qwen":      ("audiobook_factory.tts_providers.qwen_provider", "QwenTTSProvider"),
    "indextts":  ("audiobook_factory.tts_providers.indextts_provider", "IndexTTSProvider"),
    "moss":      ("audiobook_factory.tts_providers.moss_provider", "MossTTSProvider"),
    "omnivoice": ("audiobook_factory.tts_providers.omnivoice_provider", "OmniVoiceProvider"),
    "fish":      ("audiobook_factory.tts_providers.fish_provider", "FishSpeechProvider"),
    "higgs":     ("audiobook_factory.tts_providers.higgs_provider", "HiggsAudioProvider"),
    "f5tts":     ("audiobook_factory.tts_providers.f5tts_provider", "F5TTSProvider"),
    "mock":      ("tests.fixtures.mock_provider", "MockTTSProvider"),
}

_ALIASES: dict[str, str] = {
    "": "qwen", "qwen3": "qwen", "qwen3-tts": "qwen",
    "index-tts": "indextts", "indextts2": "indextts", "indextts-2": "indextts",
    "indextts-2.5": "indextts", "index-tts-2.5": "indextts",
    "moss-tts": "moss", "mosstts": "moss",
    "omni-voice": "omnivoice",
    "fish-speech": "fish", "fishspeech": "fish", "s2-pro": "fish", "fish-s2-pro": "fish",
    "higgs-audio": "higgs", "higgs-audio-v3": "higgs", "higgs-v3": "higgs", "higgs-tts": "higgs",
    "f5-tts": "f5tts", "f5_tts": "f5tts",
    "dummy": "mock", "test": "mock",
}

# Not offered in the UI or CLI help.
_HIDDEN: frozenset[str] = frozenset({"mock"})


def canonical_name(name: str | None) -> str:
    """Maps a provider name or alias to its registry key.

    Raises
    ------
    ValueError
        If the name is not a known provider or alias.
    """
    key = (name or "").lower().strip()
    key = _ALIASES.get(key, key)
    if key not in _PROVIDERS:
        raise ValueError(
            f"Unknown TTS provider: '{name}'. "
            f"Currently supported: {', '.join(repr(n) for n in provider_names(include_hidden=True))}."
        )
    return key


def is_known_provider(name: str | None) -> bool:
    """Returns True if *name* is a registered provider name or alias."""
    try:
        canonical_name(name)
        return True
    except ValueError:
        return False


def provider_names(include_hidden: bool = False) -> list[str]:
    """Returns registered provider keys in display order."""
    return [n for n in _PROVIDERS if include_hidden or n not in _HIDDEN]


def provider_class(name: str | None) -> type["BaseTTSProvider"]:
    """Imports and returns the provider class for *name*."""
    module_name, class_name = _PROVIDERS[canonical_name(name)]
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def provider_info(name: str | None) -> "ProviderInfo":
    """Returns the capability description of the provider *name*."""
    return provider_class(name).info()


def list_providers(include_hidden: bool = False) -> list["ProviderInfo"]:
    """Returns the capability descriptions of every importable provider.

    A provider whose module fails to import is skipped with a warning rather
    than breaking the UI for every other engine.
    """
    infos = []
    for name in provider_names(include_hidden=include_hidden):
        try:
            infos.append(provider_info(name))
        except Exception as exc:
            logger.warning("TTS provider '%s' is unavailable: %s", name, exc)
    return infos


def apply_recommended_settings(name: str | None, settings: dict, explicit: set[str] | None = None) -> dict:
    """Fills a settings dict with the provider's recommended sampling values.

    Args:
        name: Provider name or alias.
        settings: ``AudiobookConfig`` field values (mutated and returned).
        explicit: Keys the user set deliberately; these are never overwritten.

    Returns:
        ``settings``, with every recommended key not in ``explicit`` set.
    """
    explicit = explicit or set()
    try:
        recommended = provider_info(name).recommended_settings
    except Exception as exc:
        logger.debug("No recommended settings for provider %r: %s", name, exc)
        return settings
    for key, value in recommended.items():
        if key not in explicit:
            settings[key] = value
    return settings
