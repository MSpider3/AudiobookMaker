"""
verify_fixtures.py
==================
Verifies the integrity of generated test fixtures.
"""

from __future__ import annotations

import json
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import soundfile as sf
from audiobook_factory.text_extractor import extract


def get_fixtures_dir() -> str:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, "fixtures")


def verify_documents() -> bool:
    fixtures_dir = get_fixtures_dir()
    src_docs_dir = os.path.join(fixtures_dir, "source_documents")
    expected_json_path = os.path.join(fixtures_dir, "text", "expected_extraction.json")

    if not os.path.exists(expected_json_path):
        print(f"FAIL: Missing expected JSON: {expected_json_path}")
        return False

    with open(expected_json_path, "r", encoding="utf-8") as f:
        expected = json.load(f)

    all_passed = True
    formats = ["txt", "docx", "pdf", "epub", "odt"]

    for fmt in formats:
        doc_path = os.path.join(src_docs_dir, f"dummy_book.{fmt}")
        if not os.path.exists(doc_path):
            print(f"FAIL: Fixture missing: {doc_path}")
            all_passed = False
            continue

        size = os.path.getsize(doc_path)
        if size < 50:
            print(f"FAIL: Fixture file too small ({size} bytes): {doc_path}")
            all_passed = False
            continue

        print(f"Testing extraction for {fmt.upper()} ({doc_path}, {size} bytes)...")
        try:
            chapters, cover = extract(doc_path)
            if not chapters:
                print(f"FAIL: Extraction returned 0 chapters for {fmt}")
                all_passed = False
            else:
                print(f"  OK: Extracted {len(chapters)} chapters from {fmt.upper()}")
                total_text_len = sum(len(ch.text) for ch in chapters)
                print(f"  OK: Total extracted characters: {total_text_len}")
        except Exception as e:
            print(f"FAIL: Extraction threw exception for {fmt}: {e}")
            all_passed = False

    return all_passed


def verify_audio() -> bool:
    fixtures_dir = get_fixtures_dir()
    wav_path = os.path.join(fixtures_dir, "audio", "synthetic_voice_reference.wav")
    expected_path = os.path.join(fixtures_dir, "audio", "expected_audio_properties.json")

    if not os.path.exists(wav_path):
        print(f"FAIL: Missing audio fixture: {wav_path}")
        return False

    if not os.path.exists(expected_path):
        print(f"FAIL: Missing audio properties JSON: {expected_path}")
        return False

    with open(expected_path, "r", encoding="utf-8") as f:
        expected = json.load(f)

    data, sr = sf.read(wav_path)
    dur = len(data) / sr
    rms = float(sf.np.sqrt(sf.np.mean(data**2))) if hasattr(sf, "np") else float((data**2).mean()**0.5)

    print(f"Audio check: sr={sr}, dur={dur:.2f}s, rms={rms:.4f}")
    if sr != expected["sample_rate"]:
        print(f"FAIL: Sample rate mismatch: got {sr}, expected {expected['sample_rate']}")
        return False
    if abs(dur - expected["duration_sec"]) > 0.05:
        print(f"FAIL: Duration mismatch: got {dur}, expected {expected['duration_sec']}")
        return False
    if rms < 0.01:
        print(f"FAIL: Audio RMS is too low (silence detected): {rms}")
        return False

    print("OK: Audio fixture verification passed.")
    return True


def run_verification() -> bool:
    print("=" * 60)
    print("Verifying Test Fixtures")
    print("=" * 60)
    docs_ok = verify_documents()
    audio_ok = verify_audio()
    if docs_ok and audio_ok:
        print("\nALL FIXTURES VERIFIED SUCCESSFULLY!")
        return True
    else:
        print("\nSOME FIXTURES FAILED VERIFICATION.")
        return False


if __name__ == "__main__":
    success = run_verification()
    exit(0 if success else 1)
