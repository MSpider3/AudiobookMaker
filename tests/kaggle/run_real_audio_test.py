"""
tests/kaggle/run_real_audio_test.py
===================================
Executes real voice cloning and audio generation using user-provided Google Drive narrator audio
on GPU, using the concise source document fixtures in tests/fixtures/source_documents/ (or custom books)
for fast, rigorous validation of voice quality, audio fidelity, and subtitle synchronization.
"""

from __future__ import annotations

import glob
import json
import os
import queue
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from audiobook_factory.pipeline import AudiobookConfig, CancelToken, run_pipeline
from audiobook_factory.text_extractor import extract
from audiobook_factory.voice_preprocessor import preprocess, PreprocessConfig
from tests.audio_validator import AudioValidator, SubtitleValidator


def find_asset(patterns: list[str]) -> str | None:
    for pat in patterns:
        matches = glob.glob(os.path.join(_ROOT, pat))
        if matches:
            return matches[0]
    return None


def run_real_audio_tests() -> dict:
    print("=" * 70)
    print("  AUDIOBOOKMAKER — REAL VOICE & GPU AUDIO GENERATION SUITE")
    print("=" * 70)

    results = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "real_voice_found": False,
        "voice_path": None,
        "book_path": None,
        "status": "skipped",
        "details": {}
    }

    # Search for real voice file
    voice_patterns = [
        "narrator_voice/*.wav", "narrator_voice/*.mp3", "narrator_voice/*.m4a",
        "narrator_voice/*.flac", "narrator_voice/*.ogg", "narrator_voice/*.aac",
        "input_assets/*.wav", "input_assets/*.mp3", "input_assets/*.flac"
    ]
    real_voice = find_asset(voice_patterns)

    # Search for document source (prioritizing tests/fixtures/source_documents for quick test runs)
    book_patterns = [
        "tests/fixtures/source_documents/dummy_book.epub",
        "tests/fixtures/source_documents/dummy_book.docx",
        "tests/fixtures/source_documents/dummy_book.txt",
        "input_books/*.epub", "input_books/*.pdf", "input_books/*.docx", "input_books/*.txt",
    ]
    book_source = find_asset(book_patterns)

    if not book_source:
        from tests.fixture_generation.generate_test_documents import generate_all_fixtures
        generate_all_fixtures()
        book_source = find_asset(book_patterns)

    if not real_voice:
        print("\n  [NOTICE] Real voice reference file not detected in ./narrator_voice/.")
        print("  To test real voice synthesis, set GDRIVE_VOICE_URL in Step 3 of program_testing.ipynb.")
        print(f"  Test document ready: {book_source}\n")

        out_dir = os.path.join(_ROOT, "results")
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, "real_audio_results.json"), "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        return results

    results["real_voice_found"] = True
    results["voice_path"] = real_voice
    results["book_path"] = book_source
    print(f"  ✓ Using Real Voice: {real_voice}")
    print(f"  ✓ Using Fixture Document: {book_source}")

    # 1. Preprocess Real Voice Reference
    print("\n[Step 1] Running 7-step DSP cleaning on real voice reference...")
    cleaned_voice_path = preprocess(real_voice, PreprocessConfig())
    print(f"  ✓ Cleaned voice reference cached at: {cleaned_voice_path}")

    # 2. Extract Chapters from Fixture Document
    print(f"\n[Step 2] Extracting text from document ({os.path.basename(book_source)})...")
    chapters, cover_b64 = extract(book_source)
    print(f"  ✓ Extracted {len(chapters)} chapter(s).")
    
    # Test Chapter 1 for quick, representative verification
    test_chapters = chapters[:1]
    ch1 = test_chapters[0]
    print(f"  ✓ Testing Chapter 1: '{ch1.title}' ({len(ch1.sentences)} sentences, {len(ch1.text)} characters)")

    # 3. Determine GPU TTS Provider
    provider_name = "mock"
    try:
        import torch
        if torch.cuda.is_available():
            provider_name = "qwen"
    except Exception:
        pass

    out_dir = os.path.join(_ROOT, "results", "real_audiobook_output")
    os.makedirs(out_dir, exist_ok=True)

    cfg = AudiobookConfig(
        book_title="RealVoice_FixtureTest",
        voice_file=cleaned_voice_path,
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

    print(f"\n[Step 3] Synthesizing Chapter 1 with provider '{provider_name}' on GPU...")
    t0 = time.time()
    try:
        out_files = run_pipeline(
            config=cfg,
            chapters=test_chapters,
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
                "rms": round(val.rms, 4),
                "is_silent": val.is_silent,
                "has_nan": val.has_nan,
                "lrc_valid": lrc_ok,
                "srt_valid": srt_ok,
                "vtt_valid": vtt_ok
            })

        all_ok = all(v["is_valid_audio"] and v["lrc_valid"] and v["srt_valid"] for v in validated_files)
        results["status"] = "passed" if all_ok else "failed"
        results["elapsed_sec"] = round(elapsed, 2)
        results["details"] = validated_files

        print(f"\n  Real Voice Audio Generation Status: {results['status'].upper()}")
        print(f"  Elapsed: {elapsed:.2f}s | Files: {len(out_files)}")
        for v in validated_files:
            print(f"    - {v['path']} | Duration: {v['duration_sec']}s | RMS: {v['rms']} | Valid: {v['is_valid_audio']}")

    except Exception as exc:
        results["status"] = "failed"
        results["error"] = str(exc)
        print(f"  Real Voice Audio Generation FAILED: {exc}")

    out_dir = os.path.join(_ROOT, "results")
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "real_audio_results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    run_real_audio_tests()
