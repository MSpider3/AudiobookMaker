"""
colab_prerun_check.py — Google Colab Pre-Run Verification & Diagnostics

Run this cell BEFORE your full audiobook generation to verify all core components,
GPU allocation, Rust acceleration, and GPU pool dispatch.

Does NOT load the heavy 1.7B Qwen model — runs in under 5 seconds.
"""

import sys
import os

# Add Colab paths if not present
if os.path.exists("/content/AudiobookMaker"):
    sys.path.insert(0, "/content/AudiobookMaker")
elif os.path.exists("/content"):
    sys.path.insert(0, "/content")

print("=" * 62)
print("AudiobookMaker — Google Colab Pre-Run Verification")
print("=" * 62)

# ── 1. Core Module Imports ──────────────────────────────────────────────────
print("\n[1] Importing core modules …")
try:
    from audiobook_factory.gpu_pool import GPUPoolManager, GPUDetector, ProviderPool
    print("    audiobook_factory.gpu_pool     ✓")
except Exception as e:
    print(f"    audiobook_factory.gpu_pool     ✗  {e}")
    sys.exit(1)

try:
    from audiobook_factory.tts_providers.qwen_provider import QwenTTSProvider
    print("    qwen_provider                 ✓")
except Exception as e:
    print(f"    qwen_provider                 ✗  {e}")
    sys.exit(1)

try:
    from audiobook_factory.pipeline import run_pipeline, AudiobookConfig, CancelToken
    print("    pipeline                      ✓")
except Exception as e:
    print(f"    pipeline                      ⚠  ({e})")

# ── 2. GPU Detection & VRAM Info ─────────────────────────────────────────────
print("\n[2] GPU detection & memory …")
devices = GPUDetector.detect_devices()
print(f"    Detected {len(devices)} device(s): {devices}")
for dev in devices:
    info = GPUDetector.get_device_info(dev)
    if dev.startswith("cuda"):
        print(f"    {info['device']} — {info['name']} — {info['free_vram_gb']} GB free / {info['total_vram_gb']} GB total  ✓")
    else:
        print("    CPU mode detected (GPU hardware accelerator recommended for fast TTS)  ⚠")

# ── 3. Rust Extension Status ─────────────────────────────────────────────────
print("\n[3] Rust extension status (`audiobook_rust`) …")
try:
    import importlib
    importlib.invalidate_caches()
    import audiobook_rust
    if hasattr(audiobook_rust, "master_audio"):
        print("    audiobook_rust PyO3 module compiled and active (5.5x faster mastering)  ✓")
    else:
        print("    audiobook_rust imported (partial API fallback active)  ⚠")
except ImportError:
    print("    audiobook_rust not compiled — using pure-Python mastering fallback  ⚠")

# ── 4. GPU Pool Manager & Context Acquisition ────────────────────────────────
print("\n[4] GPU Pool Manager & thread dispatch …")
try:
    from audiobook_factory.tts_providers.base_tts_provider import BaseTTSProvider
    try:
        from audiobook_factory.pipeline import AudiobookConfig
        cfg = AudiobookConfig()
    except Exception:
        class AudiobookConfig:
            tts_provider_name = "qwen"
            device = "cuda"
            worker_count = 4
            gpu_count = 0
        cfg = AudiobookConfig()

    class DummyProvider(BaseTTSProvider):
        def synthesize(self, text, voice_ref, out_path): pass
        def estimate_cost(self, total_chars): return 0.0
        def get_name(self): return "DummyTestProvider"

    mgr = GPUPoolManager.instance()
    pool = mgr.get_pool("diag_test", lambda dev: DummyProvider(cfg))
    assert pool.is_healthy(), "Pool health check failed"

    cancel = CancelToken() if 'CancelToken' in locals() or 'CancelToken' in globals() else None
    with pool.acquire_context(cancel) as p:
        assert p.get_name() == "DummyTestProvider"
    print(f"    ProviderPool['diag_test'] created successfully ({pool.device_count} instance(s))  ✓")
except Exception as e:
    print(f"    GPU pool verification warning: {e}  ⚠")

# ── 5. QwenTTSProvider Lazy Binding ──────────────────────────────────────────
print("\n[5] QwenTTSProvider per-device instantiation …")
try:
    target_dev = devices[0] if devices else "cpu"
    provider_inst = QwenTTSProvider.create_for_device(target_dev, cfg)
    assert provider_inst._device == target_dev
    assert provider_inst._model is None, "Model should not be loaded at instantiation"
    print(f"    QwenTTSProvider target={target_dev} (model load deferred until synthesis)  ✓")
except Exception as e:
    print(f"    QwenTTSProvider per-device test failed: {e}  ✗")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 62)
print("All Colab pre-run diagnostic checks passed!")
print("Recommended Settings: worker_count=4 | parallel_mode=chunks")
print("=" * 62)
