"""
tests/kaggle/run_smoke_tests.py
===============================
Kaggle smoke test script verifying CUDA GPU detection, VRAM capacity,
PyTorch / HuggingFace imports, and native Rust acceleration.
"""

from __future__ import annotations

import json
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def run_smoke_tests() -> dict:
    print("=" * 70)
    print("  AUDIOBOOKMAKER — KAGGLE SMOKE TEST SUITE")
    print("=" * 70)

    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "python_version": sys.version,
        "cuda_available": False,
        "gpu_count": 0,
        "gpus": [],
        "rust_extension": False,
        "status": "passed",
        "errors": []
    }

    # 1. Check PyTorch and CUDA
    try:
        import torch
        results["torch_version"] = torch.__version__
        results["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            results["gpu_count"] = torch.cuda.device_count()
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                total_gb = props.total_memory / (1024 ** 3)
                free_gb = torch.cuda.mem_get_info(i)[0] / (1024 ** 3)
                gpu_info = {
                    "index": i,
                    "name": props.name,
                    "total_vram_gb": round(total_gb, 2),
                    "free_vram_gb": round(free_gb, 2),
                    "compute_capability": f"{props.major}.{props.minor}",
                }
                results["gpus"].append(gpu_info)
                print(f"  [GPU {i}] {props.name} | VRAM: {free_gb:.2f}/{total_gb:.2f} GB | CC: {props.major}.{props.minor}")
        else:
            print("  [WARNING] CUDA is not available! Running on CPU.")
    except Exception as exc:
        results["errors"].append(f"Torch/CUDA error: {exc}")
        print(f"  [ERROR] Torch/CUDA error: {exc}")

    # 2. Check Rust Native Extension
    try:
        import audiobook_rust
        has_norm = hasattr(audiobook_rust, "normalize_text")
        has_split = hasattr(audiobook_rust, "split_sentences")
        has_master = hasattr(audiobook_rust, "master_audio")
        results["rust_extension"] = has_norm and has_split and has_master
        print(f"  [Native Rust] audiobook_rust available: {results['rust_extension']}")
    except ImportError:
        results["rust_extension"] = False
        print("  [Native Rust] audiobook_rust NOT compiled (using pure-Python fallbacks).")

    # 3. Check Core Imports
    required_modules = [
        "transformers", "accelerate", "soundfile", "scipy",
        "mutagen", "pyloudnorm", "fastapi", "uvicorn", "websockets"
    ]
    for mod in required_modules:
        try:
            __import__(mod)
            print(f"  [Import OK] {mod}")
        except ImportError as exc:
            results["errors"].append(f"Missing module {mod}: {exc}")
            print(f"  [Import FAIL] {mod}: {exc}")

    if results["errors"]:
        results["status"] = "failed"

    out_dir = os.path.join(_ROOT, "results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "smoke_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("=" * 70)
    print(f"  SMOKE TEST STATUS: {results['status'].upper()} (Saved to results/smoke_results.json)")
    print("=" * 70)
    return results


if __name__ == "__main__":
    res = run_smoke_tests()
    if res["status"] != "passed":
        sys.exit(1)
