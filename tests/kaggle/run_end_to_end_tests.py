"""
tests/kaggle/run_end_to_end_tests.py
====================================
Kaggle End-to-End Pipeline test runner generating full audiobooks from document fixtures.
"""

from __future__ import annotations

import json
import os
import queue
import sys
import tempfile
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from audiobook_factory.pipeline import AudiobookConfig, CancelToken, run_pipeline
from audiobook_factory.text_extractor import extract
from tests.audio_validator import AudioValidator, SubtitleValidator


def run_end_to_end_tests() -> dict:
    print("=" * 70)
    print("  AUDIOBOOKMAKER — KAGGLE END-TO-END PIPELINE SUITE")
    print("=" * 70)

    voice_path = os.path.join(_ROOT, "tests", "fixtures", "audio", "synthetic_voice_reference.wav")
    fixtures_dir = os.path.join(_ROOT, "tests", "fixtures", "source_documents")

    if not os.path.exists(voice_path) or not os.path.exists(fixtures_dir):
        from tests.fixture_generation.generate_test_audio import generate_all_audio_fixtures
        from tests.fixture_generation.generate_test_documents import generate_all_fixtures
        generate_all_audio_fixtures()
        generate_all_fixtures()

    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "runs": {},
        "status": "passed"
    }

    formats = [("epub", "dummy_book.epub"), ("txt", "dummy_book.txt"), ("docx", "dummy_book.docx")]

    for fmt_name, filename in formats:
        doc_path = os.path.join(fixtures_dir, filename)
        if not os.path.exists(doc_path):
            continue

        print(f"\n[E2E Test] Processing fixture: {filename}...")
        chapters, cover = extract(doc_path)
        print(f"  Extracted {len(chapters)} chapter(s).")

        out_dir = os.path.join(_ROOT, "results", f"e2e_{fmt_name}")
        os.makedirs(out_dir, exist_ok=True)

        provider_name = "mock"
        try:
            import torch
            if torch.cuda.is_available():
                provider_name = "qwen"
        except Exception:
            pass

        cfg = AudiobookConfig(
            book_title=f"E2E_{fmt_name.upper()}",
            voice_file=voice_path,
            output_dir=out_dir,
            output_format="mp3",
            tts_provider_name=provider_name,
            export_lrc=True,
            export_srt=True,
            export_vtt=True,
            export_text=True,
        )

        log_q = queue.Queue()
        prog_q = queue.Queue()
        cancel = CancelToken()

        t0 = time.time()
        try:
            out_files = run_pipeline(
                config=cfg,
                chapters=chapters,
                log_queue=log_q,
                prog_queue=prog_q,
                cancel=cancel,
            )
            elapsed = time.time() - t0

            validated_files = []
            for f in out_files:
                val = AudioValidator.validate_audio_file(f, min_duration=0.5)
                lrc = os.path.splitext(f)[0] + ".lrc"
                srt = os.path.splitext(f)[0] + ".srt"
                vtt = os.path.splitext(f)[0] + ".vtt"

                lrc_ok, _ = SubtitleValidator.validate_lrc(lrc) if os.path.exists(lrc) else (False, "missing")
                srt_ok, _ = SubtitleValidator.validate_srt(srt) if os.path.exists(srt) else (False, "missing")
                vtt_ok, _ = SubtitleValidator.validate_vtt(vtt) if os.path.exists(vtt) else (False, "missing")

                validated_files.append({
                    "path": os.path.basename(f),
                    "is_valid_audio": val.is_valid,
                    "duration_sec": round(val.duration_sec, 2),
                    "lrc_valid": lrc_ok,
                    "srt_valid": srt_ok,
                    "vtt_valid": vtt_ok
                })

            results["runs"][fmt_name] = {
                "status": "passed" if all(v["is_valid_audio"] for v in validated_files) else "failed",
                "chapter_count": len(chapters),
                "generated_files": len(out_files),
                "elapsed_sec": round(elapsed, 2),
                "details": validated_files
            }
            print(f"  Result: {results['runs'][fmt_name]['status'].upper()} in {elapsed:.2f}s ({len(out_files)} files)")

        except Exception as exc:
            results["runs"][fmt_name] = {
                "status": "failed",
                "error": str(exc)
            }
            print(f"  Result: FAILED ({exc})")

    out_dir = os.path.join(_ROOT, "results")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "e2e_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    run_end_to_end_tests()
