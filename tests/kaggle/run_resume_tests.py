"""
tests/kaggle/run_resume_tests.py
================================
Kaggle resume and interruption tolerance test runner.
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
from audiobook_factory.text_extractor import ExtractedChapter
from audiobook_factory.progress_io import read_progress_file, write_progress_file
from tests.audio_validator import AudioValidator


def run_resume_tests() -> dict:
    print("=" * 70)
    print("  AUDIOBOOKMAKER — KAGGLE RESUME AND RECOVERY SUITE")
    print("=" * 70)

    voice_path = os.path.join(_ROOT, "tests", "fixtures", "audio", "synthetic_voice_reference.wav")
    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "status": "passed",
        "tests": {}
    }

    with tempfile.TemporaryDirectory() as td:
        provider_name = "mock"
        try:
            import torch
            if torch.cuda.is_available():
                provider_name = "qwen"
        except Exception:
            pass

        cfg = AudiobookConfig(
            book_title="KaggleResume",
            voice_file=voice_path,
            output_dir=td,
            output_format="mp3",
            tts_provider_name=provider_name,
            resume_incomplete_chunks=True,
        )

        sentences = [f"This is sentence {i} of the crash and resume simulation test." for i in range(4)]
        chapters = [ExtractedChapter(num=1, title="Chapter 1", text=" ".join(sentences), sentences=sentences)]

        # Simulate partial progress JSON
        prog_path = os.path.join(td, "generation_progress.json")
        prog_data = {
            "book_title": "KaggleResume",
            "settings": {"config_version": 6, "output_format": "mp3", "resume_incomplete_chunks": True},
            "chapters": [
                {"num": 1, "title": "Chapter 1", "status": "pending", "completed_chunks": [0], "retry_count": 0}
            ],
            "generation_summary": {"total_chapters": 1, "completed_count": 0, "all_complete": False}
        }
        write_progress_file(prog_path, prog_data)

        # Pre-create chunk 0 in .temp_chunks
        temp_dir = os.path.join(td, ".temp_chunks", "abm_ch001")
        os.makedirs(temp_dir, exist_ok=True)
        chunk0_path = os.path.join(temp_dir, "chunk_ch_1_0.wav")
        from tests.fixtures.mock_provider import MockTTSProvider
        MockTTSProvider(cfg).synthesize(sentences[0], open(voice_path, "rb").read(), out_path=chunk0_path)

        log_q = queue.Queue()
        prog_q = queue.Queue()
        cancel = CancelToken()

        try:
            out_files = run_pipeline(
                config=cfg,
                chapters=chapters,
                log_queue=log_q,
                prog_queue=prog_q,
                cancel=cancel,
            )

            assert len(out_files) == 1
            mp3_path = out_files[0]
            val = AudioValidator.validate_audio_file(mp3_path, min_duration=0.5)

            results["tests"]["chunk_resume"] = {
                "status": "passed" if val.is_valid else "failed",
                "output_file": os.path.basename(mp3_path),
                "is_valid": val.is_valid,
                "duration_sec": round(val.duration_sec, 2),
            }
            print(f"  Chunk Resume Test: {results['tests']['chunk_resume']['status'].upper()}")

        except Exception as exc:
            results["tests"]["chunk_resume"] = {"status": "failed", "error": str(exc)}
            results["status"] = "failed"
            print(f"  Chunk Resume Test: FAILED ({exc})")

    out_dir = os.path.join(_ROOT, "results")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "resume_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    run_resume_tests()
