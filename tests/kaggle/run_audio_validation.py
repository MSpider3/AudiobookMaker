"""
tests/kaggle/run_audio_validation.py
===================================
Kaggle audio and subtitle validation scanner across all generated artifacts.
"""

from __future__ import annotations

import json
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tests.audio_validator import AudioValidator, SubtitleValidator


def run_audio_validation() -> dict:
    print("=" * 70)
    print("  AUDIOBOOKMAKER — KAGGLE ARTIFACT VALIDATION SUITE")
    print("=" * 70)

    results_dir = os.path.join(_ROOT, "results")
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "audio_files_checked": 0,
        "audio_files_valid": 0,
        "subtitles_checked": 0,
        "subtitles_valid": 0,
        "failures": [],
        "status": "passed"
    }

    if not os.path.exists(results_dir):
        print("  No results directory found to scan.")
        return report

    for root, _, files in os.walk(results_dir):
        for f in files:
            full_path = os.path.join(root, f)
            rel_path = os.path.relpath(full_path, results_dir)

            if f.endswith((".mp3", ".wav", ".flac", ".m4b")):
                report["audio_files_checked"] += 1
                val = AudioValidator.validate_audio_file(full_path)
                if val.is_valid:
                    report["audio_files_valid"] += 1
                    print(f"  [AUDIO VALID] {rel_path} ({val.duration_sec:.2f}s, RMS: {val.rms:.4f})")
                else:
                    report["failures"].append({"file": rel_path, "type": "audio", "error": val.error_message})
                    print(f"  [AUDIO INVALID] {rel_path}: {val.error_message}")

            elif f.endswith(".lrc"):
                report["subtitles_checked"] += 1
                ok, err = SubtitleValidator.validate_lrc(full_path)
                if ok:
                    report["subtitles_valid"] += 1
                    print(f"  [LRC VALID] {rel_path}")
                else:
                    report["failures"].append({"file": rel_path, "type": "lrc", "error": err})
                    print(f"  [LRC INVALID] {rel_path}: {err}")

            elif f.endswith(".srt"):
                report["subtitles_checked"] += 1
                ok, err = SubtitleValidator.validate_srt(full_path)
                if ok:
                    report["subtitles_valid"] += 1
                    print(f"  [SRT VALID] {rel_path}")
                else:
                    report["failures"].append({"file": rel_path, "type": "srt", "error": err})
                    print(f"  [SRT INVALID] {rel_path}: {err}")

            elif f.endswith(".vtt"):
                report["subtitles_checked"] += 1
                ok, err = SubtitleValidator.validate_vtt(full_path)
                if ok:
                    report["subtitles_valid"] += 1
                    print(f"  [VTT VALID] {rel_path}")
                else:
                    report["failures"].append({"file": rel_path, "type": "vtt", "error": err})
                    print(f"  [VTT INVALID] {rel_path}: {err}")

    if report["failures"]:
        report["status"] = "failed"

    with open(os.path.join(results_dir, "validation_results.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("=" * 70)
    print(f"  VALIDATION SUMMARY: Audio: {report['audio_files_valid']}/{report['audio_files_checked']} | "
          f"Subtitles: {report['subtitles_valid']}/{report['subtitles_checked']}")
    print("=" * 70)
    return report


if __name__ == "__main__":
    run_audio_validation()
