"""
test_gpu_pool_multi_provider.py
================================
Regression tests for switching between multiple TTS providers within one
process (GPUPoolManager / ProviderPool).

Before this fix, GPUPoolManager kept every provider pool it ever created
resident forever: switching from one TTS engine to another (e.g. Qwen ->
VibeVoice) loaded a second full model onto the same device(s) without ever
freeing the first, and GPUPoolManager.shutdown() cleared its bookkeeping
dict without calling provider.cleanup() at all. Repeated or alternating use
of more than one TTS provider in a single session therefore accumulated GPU
memory until synthesis failed (commonly surfacing as a CUDA OOM).
"""

from __future__ import annotations

import os
import sys
import threading
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from audiobook_factory.pipeline import AudiobookConfig
from audiobook_factory.gpu_pool import GPUPoolManager
from tests.fixtures.mock_provider import MockTTSProvider


@pytest.fixture(autouse=True)
def _isolated_manager():
    """Give every test a fresh GPUPoolManager singleton so pools from one
    test can't leak state into the next."""
    GPUPoolManager._instance = None
    yield
    # Best-effort teardown: free whatever the test left loaded.
    try:
        GPUPoolManager.instance().shutdown()
    except Exception:
        pass
    GPUPoolManager._instance = None


def _mock_factory(config):
    return lambda dev: MockTTSProvider.create_for_device(dev, config)


class TestMultiProviderSwitching:

    def test_switching_provider_evicts_and_cleans_up_previous_pool(self):
        config = AudiobookConfig()
        manager = GPUPoolManager.instance()

        pool_a = manager.get_pool("mock-engine-a", _mock_factory(config), min_vram_gb=0.0)
        provider_a = pool_a.get_provider_for_device(pool_a.devices[0])
        assert provider_a.is_ready is True

        # Switching to a different provider must free the first one's model.
        pool_b = manager.get_pool("mock-engine-b", _mock_factory(config), min_vram_gb=0.0)
        assert provider_a.is_ready is False, (
            "Previous provider's cleanup() was not called when switching TTS engines"
        )
        assert manager.all_pools().keys() == {"mock-engine-b"}, (
            "Old provider pool should be evicted, not left resident alongside the new one"
        )
        assert pool_b.get_provider_for_device(pool_b.devices[0]).is_ready is True

    def test_reselecting_same_provider_reuses_pool_without_reload(self):
        config = AudiobookConfig()
        manager = GPUPoolManager.instance()

        pool_1 = manager.get_pool("mock-engine-a", _mock_factory(config), min_vram_gb=0.0)
        pool_2 = manager.get_pool("mock-engine-a", _mock_factory(config), min_vram_gb=0.0)
        assert pool_1 is pool_2

    def test_switch_raises_instead_of_evicting_a_busy_provider(self):
        config = AudiobookConfig()
        manager = GPUPoolManager.instance()

        pool_a = manager.get_pool("mock-engine-a", _mock_factory(config), min_vram_gb=0.0)
        # Simulate a chapter mid-synthesis: the provider is checked out and
        # not yet returned to the pool.
        held_provider = pool_a.acquire()

        with pytest.raises(RuntimeError, match="mid-synthesis"):
            manager.get_pool("mock-engine-b", _mock_factory(config), min_vram_gb=0.0)

        # The busy provider must still be usable — it was not torn down.
        assert held_provider.is_ready is True
        pool_a.release(held_provider)

        # Once released (idle), switching succeeds and frees engine A.
        manager.get_pool("mock-engine-b", _mock_factory(config), min_vram_gb=0.0)
        assert held_provider.is_ready is False

    def test_keep_other_providers_opts_out_of_eviction(self):
        config = AudiobookConfig()
        manager = GPUPoolManager.instance()

        pool_a = manager.get_pool("mock-engine-a", _mock_factory(config), min_vram_gb=0.0)
        provider_a = pool_a.get_provider_for_device(pool_a.devices[0])

        manager.get_pool(
            "mock-engine-b", _mock_factory(config), min_vram_gb=0.0,
            keep_other_providers=True,
        )
        assert provider_a.is_ready is True
        assert set(manager.all_pools().keys()) == {"mock-engine-a", "mock-engine-b"}

    def test_shutdown_cleans_up_every_loaded_provider(self):
        config = AudiobookConfig()
        manager = GPUPoolManager.instance()

        pool_a = manager.get_pool("mock-engine-a", _mock_factory(config), min_vram_gb=0.0)
        provider_a = pool_a.get_provider_for_device(pool_a.devices[0])

        manager.shutdown()

        assert provider_a.is_ready is False
        assert manager.all_pools() == {}

    def test_evict_explicit_method(self):
        config = AudiobookConfig()
        manager = GPUPoolManager.instance()

        pool_a = manager.get_pool("mock-engine-a", _mock_factory(config), min_vram_gb=0.0)
        provider_a = pool_a.get_provider_for_device(pool_a.devices[0])

        assert manager.evict("does-not-exist") is False
        assert manager.evict("mock-engine-a") is True
        assert provider_a.is_ready is False
        assert manager.all_pools() == {}


class TestPreviewProviderCache:
    """Voice Studio's preview_tts() used to leak a fresh model on every call."""

    def test_preview_reuses_provider_and_frees_on_switch(self, monkeypatch):
        from audiobook_factory import pipeline as pipeline_mod

        pipeline_mod._preview_provider_cache["provider"] = None
        pipeline_mod._preview_provider_cache["name"] = None

        created: list[MockTTSProvider] = []

        def _fake_get_tts_provider(name, config, device=None, dtype_override=None):
            p = MockTTSProvider(config, device=device or "cpu")
            created.append(p)
            return p

        monkeypatch.setattr(
            "audiobook_factory.tts_providers.get_tts_provider", _fake_get_tts_provider
        )

        cfg = AudiobookConfig(tts_provider_name="mock")
        pipeline_mod.preview_tts("Hello there", cfg)
        pipeline_mod.preview_tts("Hello again", cfg)
        assert len(created) == 1, "Same provider should be reused across previews"
        assert created[0].is_ready is True

        cfg2 = AudiobookConfig(tts_provider_name="mock-other")
        pipeline_mod.preview_tts("Switch engines", cfg2)
        assert len(created) == 2, "A new provider should be created for a different engine"
        assert created[0].is_ready is False, "Previous preview provider must be cleaned up"
        assert created[1].is_ready is True

        pipeline_mod._cleanup_preview_provider()
        assert created[1].is_ready is False

    def test_preview_switches_on_model_variant_change(self, monkeypatch):
        from audiobook_factory import pipeline as pipeline_mod

        pipeline_mod._cleanup_preview_provider()
        created: list[MockTTSProvider] = []

        def _fake_get_tts_provider(name, config, device=None, dtype_override=None):
            p = MockTTSProvider(config, device=device or "cpu")
            created.append(p)
            return p

        monkeypatch.setattr(
            "audiobook_factory.tts_providers.get_tts_provider", _fake_get_tts_provider
        )

        cfg1 = AudiobookConfig(tts_provider_name="mock", tts_model_name="variant-a")
        pipeline_mod.preview_tts("Text 1", cfg1)
        assert len(created) == 1

        cfg2 = AudiobookConfig(tts_provider_name="mock", tts_model_name="variant-b")
        pipeline_mod.preview_tts("Text 2", cfg2)
        assert len(created) == 2, "Changing model variant must trigger reload"
        assert created[0].is_ready is False, "Previous model variant must be cleaned up"
        assert created[1].is_ready is True

        pipeline_mod._cleanup_preview_provider()


class TestPoolIdleState:
    def test_pool_is_idle_after_cleanup(self):
        from audiobook_factory.gpu_pool import ProviderPool

        config = AudiobookConfig()
        pool = ProviderPool(_mock_factory(config), ["cpu"], "test-idle")
        assert pool.is_idle is True
        pool.cleanup()
        assert pool.is_idle is True


@pytest.mark.asyncio
async def test_wait_for_other_providers_cancels_promptly():
    import asyncio
    from api.worker import _wait_for_other_providers_idle, Task
    from audiobook_factory.pipeline import AudiobookConfig, CancelToken
    from audiobook_factory.gpu_pool import GPUPoolManager

    manager = GPUPoolManager.instance()
    cfg_a = AudiobookConfig(tts_provider_name="engine-a")
    pool_a = manager.get_pool("engine-a", _mock_factory(cfg_a), min_vram_gb=0.0)
    provider_a = pool_a.acquire()  # Keep pool_a busy

    cfg_b = AudiobookConfig(tts_provider_name="engine-b")
    cancel_token = CancelToken()
    task = Task(task_id="test-cancel-wait", config_dict={}, chapters=[], cancel_token=cancel_token)

    async def cancel_soon():
        await asyncio.sleep(0.1)
        cancel_token.cancel()

    asyncio.create_task(cancel_soon())
    # Should not block for 600 seconds — should return promptly on cancel
    await asyncio.wait_for(_wait_for_other_providers_idle(cfg_b, task, timeout=10.0), timeout=2.0)
    assert cancel_token.is_cancelled is True
    pool_a.release(provider_a)

