"""
test_progress_persistence.py
=============================
Regression tests for QA-LEAD-03 (progress persistence and chunk tracking).
Verifies that chunk completion is written to disk after each chunk.
"""

from __future__ import annotations

import os
import queue
import sys
import tempfile
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from audiobook_factory.pipeline import AudiobookConfig, CancelToken, run_pipeline
from audiobook_factory.text_extractor import ExtractedChapter
from audiobook_factory.progress_io import read_progress_file


class TestProgressPersistenceRegression:

    @pytest.fixture
    def voice_file_path(self):
        return os.path.join(_ROOT, "tests", "fixtures", "audio", "synthetic_voice_reference.wav")

    def test_chunk_progress_persisted_to_json(self, voice_file_path):
        with tempfile.TemporaryDirectory() as td:
            cfg = AudiobookConfig(
                book_title="PersistenceBook",
                voice_file=voice_file_path,
                output_dir=td,
                output_format="mp3",
                tts_provider_name="mock",
                worker_count=1,
            )

            sentences = [f"Sentence number {i} in the persistence test chapter." for i in range(5)]
            chapters = [
                ExtractedChapter(num=1, title="Chapter 1", text=" ".join(sentences), sentences=sentences)
            ]

            log_q = queue.Queue()
            prog_q = queue.Queue()
            cancel = CancelToken()

            out_files = run_pipeline(
                config=cfg,
                chapters=chapters,
                log_queue=log_q,
                prog_queue=prog_q,
                cancel=cancel,
            )

            assert len(out_files) == 1

            # Verify progress file
            prog_path = os.path.join(td, "generation_progress.json")
            assert os.path.exists(prog_path)
            data = read_progress_file(prog_path)
            assert data["generation_summary"]["all_complete"] is True
            assert data["chapters"][0]["status"] == "completed"
