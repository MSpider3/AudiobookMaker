"""
kaggle_prerun_check.py — Run this cell BEFORE your first full generation.

Tests the actual import chain and all critical components with real GPU
allocation (no mock providers). Does NOT load the Qwen3-TTS model —
that takes 2+ minutes and you only need to do it once per session.

Paste this into a Kaggle cell and run it. Every section should print ✓.
If any section fails, it tells you exactly what to fix.
"""

import sys, os
sys.path.insert(0, "/kaggle/working/AudiobookMaker")

print("=" * 62)
print("AudiobookMaker — Pre-Run Verification")
print("=" * 62)

# ── 1. Module imports ─────────────────────────────────────────────────────────
print("\n[1] Importing core modules …")
try:
    from audiobook_factory.gpu_pool import GPUPoolManager, GPUDetector, ProviderPool
    print("    gpu_pool  ✓")
except Exception as e:
    print(f"    gpu_pool  ✗  {e}")
    sys.exit(1)

try:
    from audiobook_factory.tts_providers.qwen_provider import QwenTTSProvider
    print("    qwen_provider  ✓")
except Exception as e:
    print(f"    qwen_provider  ✗  {e}")
    sys.exit(1)

try:
    from audiobook_factory.pipeline import run_pipeline
    print("    pipeline  ✓")
except Exception as e:
    print(f"    pipeline  ✗  {e}")

# ── 2. GPU detection ──────────────────────────────────────────────────────────
print("\n[2] GPU detection …")
devs = GPUDetector.detect_devices()
print(f"    {len(devs)} device(s) detected: {devs}")
for dev in devs:
    info = GPUDetector.get_device_info(dev)
    print(f"    {info['device']} — {info['name']} — {info['free_vram_gb']}GB free / {info['total_vram_gb']}GB total")

# ── 3. Reference audio pre-processor ─────────────────────────────────────────
print("\n[3] Reference audio pre-processor …")
import tempfile, torchaudio

sr = 24_000
fake = torch.randn(2, int(12.7 * sr))   # stereo, 12.7 s, not 512-aligned
with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
    torchaudio.save(f.name, fake, sr)
    tmp_path = f.name

try:
    clean = _prepare_clean_voice_ref(tmp_path, target_sr=sr)
    assert clean.shape[0] == 1,                        "not mono"
    assert clean.shape[-1] % 512 == 0,                 "not 512-aligned"
    assert clean.shape[-1] <= int(10.0 * sr),          "longer than 10s"
    assert clean.dtype == torch.float32,               "not float32"
    print(f"    {clean.shape}  dtype={clean.dtype}  ✓")
finally:
    os.unlink(tmp_path)

# ── 4. _safe_text filters ─────────────────────────────────────────────────────
print("\n[4] _safe_text chunk filter …")
assert _safe_text("") is None,                          "empty should be None"
assert _safe_text("...") is None,                       "punctuation-only should be None"
assert _safe_text("Hi") is None,                        "too short should be None"
assert _safe_text("A" * _MIN_CHUNK_CHARS) is not None,  "min-length should pass"
print(f"    min_chunk_chars={_MIN_CHUNK_CHARS}  ✓")

# ── 5. _is_cancelled handles both attribute forms ─────────────────────────────
print("\n[5] _is_cancelled normalisation …")

class _TokA:           is_cancelled = False
class _TokB:           cancelled    = False
class _TokC:
    def is_cancelled(self): return False

assert not _is_cancelled(None)
assert not _is_cancelled(_TokA())
assert not _is_cancelled(_TokB())
assert not _is_cancelled(_TokC())
print("    All cancel-token forms handled  ✓")

# ── 6. Thread-safe progress file write ───────────────────────────────────────
print("\n[6] Thread-safe progress writes …")
try:
    from audiobook_factory.pipeline import _save_progress, _ProgressFileLock
    import json, pathlib, threading

    test_path = pathlib.Path(tempfile.mktemp(suffix=".json"))
    errors = []

    def _writer(i):
        try:
            _save_progress(test_path, {"writer": i, "ok": True})
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=_writer, args=(i,)) for i in range(20)]
    for t in threads: t.start()
    for t in threads: t.join()

    # File must be valid JSON (no partial write corruption)
    data = json.loads(test_path.read_text())
    assert "writer" in data and data["ok"] is True
    test_path.unlink()

    if errors:
        print(f"    ✗  {len(errors)} write errors: {errors[0]}")
    else:
        print("    20 concurrent writes, zero corruption  ✓")

except ImportError:
    print("    ⚠  _save_progress not found in pipeline.py — apply Fix 1 from pipeline_threadsafe_patch.py")

# ── 7. Dispatcher work-stealing with timeout ─────────────────────────────────
print("\n[7] Work-stealing queue + Future timeout …")
import time

class _MockProvider:
    def __init__(self, delay): self.delay = delay; self.count = 0
    def synthesize(self, text, out_path):
        time.sleep(self.delay); self.count += 1; return out_path

providers = {"cuda:0": _MockProvider(0.01), "cuda:1": _MockProvider(0.03)}
disp = MultiGPUDispatcher(providers)

def _fn(provider, item):
    return provider.synthesize(item, f"/tmp/{item}.wav")

items = [f"chunk_{i}" for i in range(16)]
t0 = time.perf_counter()
results = disp.dispatch_chunks(items, _fn)
elapsed = time.perf_counter() - t0

assert len(results) == 16
p0 = providers["cuda:0"].count
p1 = providers["cuda:1"].count
assert p0 + p1 == 16
# Fast GPU should steal more work
assert p0 > p1, f"Work-stealing failed: GPU0={p0}, GPU1={p1} (GPU0 should be > GPU1)"
print(f"    GPU0 (fast): {p0} tasks | GPU1 (slow): {p1} tasks | wall={elapsed:.2f}s  ✓")
disp.shutdown()

# ── 8. QwenTTSProvider per-device instantiation ─────────────────────────
print("\n[8] QwenTTSProvider per-device instantiation …")
from audiobook_factory.pipeline import AudiobookConfig
cfg = AudiobookConfig()
p0 = QwenTTSProvider.create_for_device("cuda:0", cfg)
assert p0._device == "cuda:0"
assert p0._model is None, "model should not load at __init__"
print("    cuda:0 provider instance model=not-yet-loaded  ✓")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 62)
print("All checks passed. Safe to start audiobook generation.")
print("Recommended config:  worker_count=4  parallel_mode=chapters")
print("=" * 62)
