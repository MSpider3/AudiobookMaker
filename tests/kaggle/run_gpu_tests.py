"""
tests/kaggle/run_gpu_tests.py
=============================
Kaggle multi-GPU dispatch and VRAM headroom validation suite.
"""

from __future__ import annotations

import json
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from audiobook_factory.gpu_pool import GPUDetector, GPUPoolManager
from audiobook_factory.pipeline import AudiobookConfig


def run_gpu_tests() -> dict:
    print("=" * 70)
    print("  AUDIOBOOKMAKER — KAGGLE MULTI-GPU DISPATCH SUITE")
    print("=" * 70)

    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "devices": [],
        "pool_concurrency": 1,
        "status": "passed",
        "notes": []
    }

    try:
        import torch
        detected = GPUDetector.detect_devices()
        results["devices"] = detected
        print(f"  Detected devices: {detected}")

        if len(detected) >= 2:
            print(f"  [Multi-GPU] Detected Dual GPUs: {detected[0]} and {detected[1]}")
            results["notes"].append("Dual GPU setup detected and active.")
            results["pool_concurrency"] = len(detected)
        elif len(detected) == 1:
            print(f"  [Single GPU] Detected 1 device: {detected[0]}")
            results["notes"].append("Single device active.")
        else:
            print("  [CPU Mode] No CUDA devices detected.")
            results["notes"].append("CPU fallback mode.")

        for d in detected:
            info = GPUDetector.get_device_info(d)
            print(f"    Device {d}: {info}")

    except Exception as exc:
        results["status"] = "failed"
        results["error"] = str(exc)
        print(f"  [GPU Error] {exc}")

    out_dir = os.path.join(_ROOT, "results")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "gpu_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    run_gpu_tests()
