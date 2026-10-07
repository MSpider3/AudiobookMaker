"""
test_kaggle_assets.py
=====================
The Kaggle test notebook reads its narrator voice and books from
``tests/kaggle/assets``. These tests keep that folder complete and in step
with the fixtures it is copied from.
"""

from __future__ import annotations

import filecmp
import os
import sys

import soundfile as sf

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tests.kaggle import sync_assets


class TestKaggleAssets:

    def test_books_match_the_fixtures(self):
        wanted = sync_assets.book_files()
        assert wanted, "no fixture books found"
        assert sorted(os.listdir(sync_assets.BOOKS_DIR)) == wanted, (
            "tests/kaggle/assets/books is out of date — run: python tests/kaggle/sync_assets.py"
        )
        for name in wanted:
            assert filecmp.cmp(
                os.path.join(sync_assets.FIXTURES_DIR, name),
                os.path.join(sync_assets.BOOKS_DIR, name),
                shallow=False,
            ), f"{name} differs from its fixture — run: python tests/kaggle/sync_assets.py"

    def test_every_book_format_is_present(self):
        suffixes = {os.path.splitext(name)[1] for name in os.listdir(sync_assets.BOOKS_DIR)}
        assert {".epub", ".pdf", ".docx", ".odt", ".txt", ".mobi"} <= suffixes
        assert os.path.exists(os.path.join(sync_assets.BOOKS_DIR, "expected_chapters.json"))

    def test_narrator_voice_is_a_usable_reference_with_a_transcript(self):
        voice_dir = os.path.join(sync_assets.ASSETS_DIR, "voice")
        clips = [name for name in os.listdir(voice_dir) if name.endswith(".wav")]
        assert len(clips) == 1, "exactly one narrator clip is expected"
        clip = os.path.join(voice_dir, clips[0])
        info = sf.info(clip)
        assert info.channels == 1 and info.samplerate == 24000
        assert 5.0 <= info.duration <= 30.0
        with open(os.path.splitext(clip)[0] + ".txt", encoding="utf-8") as fh:
            transcript = fh.read().strip()
        # Roughly two to four words a second for narration.
        assert info.duration * 1.5 <= len(transcript.split()) <= info.duration * 5
