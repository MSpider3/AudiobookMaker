"""
test_progress_io.py
===================
Unit tests for atomic progress file reading, writing, self-healing,
and concurrent thread-safe state updates.
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import sys
import tempfile
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from audiobook_factory.progress_io import (
    read_progress_file,
    write_progress_file,
    update_chapter_chunk,
    update_chapter_status,
    update_chapter_retry,
)


class TestProgressIO:

    @pytest.fixture
    def sample_progress_data(self):
        return {
            "book_title": "Test Book",
            "book_path": "/tmp/book.epub",
            "voice_file": "/tmp/voice.wav",
            "settings": {"config_version": 6, "output_format": "mp3"},
            "chapters": [
                {
                    "num": 1,
                    "title": "Chapter 1",
                    "status": "pending",
                    "completed_chunks": [],
                    "retry_count": 0,
                    "last_error": None
                }
            ],
            "generation_summary": {
                "total_chapters": 1,
                "completed_count": 0,
                "failed_count": 0,
                "failed_chapters": [],
                "all_complete": False
            }
        }

    def test_write_and_read_atomic_progress_file(self, sample_progress_data):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "generation_progress.json")
            write_progress_file(path, sample_progress_data)
            assert os.path.exists(path)

            loaded = read_progress_file(path)
            assert loaded["book_title"] == "Test Book"
            assert len(loaded["chapters"]) == 1

    def test_read_utf8_bom(self, sample_progress_data):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "progress_bom.json")
            with open(path, "w", encoding="utf-8-sig") as f:
                json.dump(sample_progress_data, f)

            loaded = read_progress_file(path)
            assert loaded["book_title"] == "Test Book"

    def test_read_leading_garbage_healing(self, sample_progress_data):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "progress_corrupt.json")
            raw_json = json.dumps(sample_progress_data)
            with open(path, "w", encoding="utf-8") as f:
                f.write("junk_data_prefix \n\r " + raw_json)

            loaded = read_progress_file(path)
            assert loaded["book_title"] == "Test Book"

    def test_read_empty_file_raises(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "empty.json")
            with open(path, "w", encoding="utf-8") as f:
                f.write("")
            with pytest.raises(ValueError, match="empty or too small"):
                read_progress_file(path)

    def test_read_html_content_raises(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "error.html")
            with open(path, "w", encoding="utf-8") as f:
                f.write("<!DOCTYPE html><html><body>502 Bad Gateway</body></html>")
            with pytest.raises(ValueError, match="HTML"):
                read_progress_file(path)

    def test_update_chapter_chunk_no_duplicates(self, sample_progress_data):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "generation_progress.json")
            write_progress_file(path, sample_progress_data)

            update_chapter_chunk(path, chapter_num=1, chunk_index=0)
            update_chapter_chunk(path, chapter_num=1, chunk_index=1)
            update_chapter_chunk(path, chapter_num=1, chunk_index=0)  # duplicate

            loaded = read_progress_file(path)
            chunks = loaded["chapters"][0]["completed_chunks"]
            assert chunks == [0, 1]

    def test_update_chapter_status_resets_chunks(self, sample_progress_data):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "generation_progress.json")
            sample_progress_data["chapters"][0]["completed_chunks"] = [0, 1, 2]
            write_progress_file(path, sample_progress_data)

            update_chapter_status(path, chapter_num=1, status="completed", reset_chunks=True)
            loaded = read_progress_file(path)
            assert loaded["chapters"][0]["status"] == "completed"
            assert loaded["chapters"][0]["completed_chunks"] == []

    def test_concurrent_chunk_updates_no_lost_writes(self, sample_progress_data):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "generation_progress.json")
            write_progress_file(path, sample_progress_data)

            num_chunks = 30
            with concurrent.futures.ThreadPoolExecutor(max_workers=8) as ex:
                futures = [
                    ex.submit(update_chapter_chunk, path, 1, i)
                    for i in range(num_chunks)
                ]
                concurrent.futures.wait(futures)

            loaded = read_progress_file(path)
            chunks = loaded["chapters"][0]["completed_chunks"]
            assert sorted(chunks) == list(range(num_chunks))
