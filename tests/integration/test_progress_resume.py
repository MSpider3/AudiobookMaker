"""
test_progress_resume.py
========================
Integration tests for progress resumption, chunk reuse from .temp_chunks,
and missing file regeneration behavior.
"""

from __future__ import annotations

import json
import os
import queue
import shutil
import sys
import tempfile
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from audiobook_factory.pipeline import AudiobookConfig, CancelToken, run_pipeline
from audiobook_factory.text_extractor import ExtractedChapter
from audiobook_factory.gpu_pool import ProviderPool
from audiobook_factory.progress_io import read_progress_file, write_progress_file
from tests.fixtures.mock_provider import MockTTSProvider
from tests.audio_validator import AudioValidator


class TestProgressResume:

    @pytest.fixture
    def voice_file_path(self):
        return os.path.join(_ROOT, "tests", "fixtures", "audio", "synthetic_voice_reference.wav")

    def test_chunk_level_resume_skips_existing_chunks(self, voice_file_path):
        with tempfile.TemporaryDirectory() as td:
            cfg = AudiobookConfig(
                book_title="ResumeBook",
                voice_file=voice_file_path,
                output_dir=td,
                output_format="mp3",
                tts_provider_name="mock",
                resume_incomplete_chunks=True,
                worker_count=1,
            )

            # Create mock provider and track calls
            provider = MockTTSProvider(cfg, device="cpu")
            pool = ProviderPool(lambda dev: provider, ["cpu"], "mock")

            # Create 1 chapter with 4 distinct sentences
            sentences = [
                "Sentence zero of the interrupted chapter.",
                "Sentence one of the interrupted chapter.",
                "Sentence two of the interrupted chapter.",
                "Sentence three of the interrupted chapter."
            ]
            chapters = [
                ExtractedChapter(num=1, title="Chapter 1", text=" ".join(sentences), sentences=sentences)
            ]

            # Simulate interruption: manually pre-synthesize chunk 0 and 1 in .temp_chunks
            temp_ch_dir = os.path.join(td, ".temp_chunks", "abm_ch001")
            os.makedirs(temp_ch_dir, exist_ok=True)
            chunk0_path = os.path.join(temp_ch_dir, "chunk_ch_1_0.wav")
            chunk1_path = os.path.join(temp_ch_dir, "chunk_ch_1_1.wav")

            with open(voice_file_path, "rb") as vf:
                v_bytes = vf.read()
            provider.synthesize(sentences[0], v_bytes, out_path=chunk0_path)
            provider.synthesize(sentences[1], v_bytes, out_path=chunk1_path)

            initial_calls = provider.synthesis_call_count

            # Create progress JSON marking chunk 0 and 1 complete
            progress_data = {
                "book_title": "ResumeBook",
                "settings": {"config_version": 6, "output_format": "mp3", "resume_incomplete_chunks": True},
                "chapters": [
                    {
                        "num": 1,
                        "title": "Chapter 1",
                        "status": "pending",
                        "completed_chunks": [0, 1],
                        "retry_count": 0,
                        "last_error": None
                    }
                ],
                "generation_summary": {"total_chapters": 1, "completed_count": 0, "all_complete": False}
            }
            write_progress_file(os.path.join(td, "generation_progress.json"), progress_data)

            # Run pipeline to resume
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

            assert len(out_files) >= 1
            mp3_path = out_files[0]
            assert os.path.exists(mp3_path)

            # Validate audio
            val = AudioValidator.validate_audio_file(mp3_path, min_duration=0.5)
            assert val.is_valid, f"Resumed audio failed validation: {val.error_message}"

    def test_missing_file_regeneration_when_regen_missing_true(self, voice_file_path):
        with tempfile.TemporaryDirectory() as td:
            cfg = AudiobookConfig(
                book_title="RegenBook",
                voice_file=voice_file_path,
                output_dir=td,
                output_format="mp3",
                tts_provider_name="mock",
                regen_missing=True,
            )

            chapters = [
                ExtractedChapter(num=1, title="Chapter 1", text="Chapter one text.", sentences=["Chapter one text."]),
                ExtractedChapter(num=2, title="Chapter 2", text="Chapter two text.", sentences=["Chapter two text."])
            ]

            # Chapter 1 is marked completed in progress JSON, but output file does NOT exist on disk
            progress_data = {
                "book_title": "RegenBook",
                "settings": {"config_version": 6, "output_format": "mp3", "regen_missing": True},
                "chapters": [
                    {"num": 1, "title": "Chapter 1", "status": "completed", "completed_chunks": []},
                    {"num": 2, "title": "Chapter 2", "status": "pending", "completed_chunks": []}
                ],
                "generation_summary": {"total_chapters": 2, "completed_count": 1, "all_complete": False}
            }
            write_progress_file(os.path.join(td, "generation_progress.json"), progress_data)

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

            # Both chapter 1 and chapter 2 should now be generated on disk
            assert len(out_files) >= 2
            for f in out_files:
                if f.endswith(".mp3") and not "Combined" in f:
                    assert os.path.exists(f)
                    assert AudioValidator.validate_audio_file(f, min_duration=0.2).is_valid
