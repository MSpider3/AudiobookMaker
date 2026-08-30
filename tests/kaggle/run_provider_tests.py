"""
tests/kaggle/run_provider_tests.py
==================================
Kaggle TTS provider test runner validating synthesis quality and hardware acceleration.
"""

from __future__ import annotations

import json
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from audiobook_factory.pipeline import AudiobookConfig
from audiobook_factory.tts_providers import get_tts_provider
from tests.audio_validator import AudioValidator


def run_provider_tests() -> dict:
    print("=" * 70)
    print("  AUDIOBOOKMAKER — KAGGLE TTS PROVIDER TEST SUITE")
    print("=" * 70)

    voice_path = os.path.join(_ROOT, "tests", "fixtures", "audio", "synthetic_voice_reference.wav")
    if not os.path.exists(voice_path):
        from tests.fixture_generation.generate_test_audio import generate_all_audio_fixtures
        generate_all_audio_fixtures()

    with open(voice_path, "rb") as f:
        voice_bytes = f.read()

    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "providers_tested": {},
        "status": "passed"
    }

    # Test providers: "mock", "qwen"
    test_text = "In the ancient library, dusty tomes held the secrets of forgotten epochs."
    
    providers_to_test = ["mock"]
    try:
        import torch
        if torch.cuda.is_available():
            providers_to_test.append("qwen")
    except Exception:
        pass

    for p_name in providers_to_test:
        print(f"\n[Testing Provider] '{p_name}'...")
        cfg = AudiobookConfig(tts_provider_name=p_name)
        try:
            p_inst = get_tts_provider(p_name, cfg)
            p_inst.ensure_ready()

            t0 = time.time()
            audio_bytes, dur = p_inst.synthesize(test_text, voice_bytes, return_bytes=True)
            elapsed = time.time() - t0

            val = AudioValidator.validate_audio_bytes(audio_bytes, min_duration=0.5)
            rtf = elapsed / dur if dur > 0 else 0.0

            provider_res = {
                "status": "passed" if val.is_valid else "failed",
                "duration_sec": round(dur, 2),
                "latency_sec": round(elapsed, 2),
                "rtf": round(rtf, 2),
                "rms": round(val.rms, 4),
                "is_silent": val.is_silent,
                "has_nan": val.has_nan,
                "error": val.error_message if not val.is_valid else None
            }
            results["providers_tested"][p_name] = provider_res
            print(f"  Result: {provider_res['status'].upper()} | Dur: {dur:.2f}s | RTF: {rtf:.2f}x | RMS: {val.rms:.4f}")
        except Exception as exc:
            results["providers_tested"][p_name] = {
                "status": "failed",
                "error": str(exc)
            }
            print(f"  Result: FAILED ({exc})")

    out_dir = os.path.join(_ROOT, "results")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "provider_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    run_provider_tests()
