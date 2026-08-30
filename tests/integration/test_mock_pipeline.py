"""
test_mock_pipeline.py
=====================
Integration tests for top-level pipeline execution (pipeline.py:run_pipeline)
using MockTTSProvider across multiple chapters, subtitle formats, and packaging modes.
"""

from __future__ import annotations

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
from tests.fixtures.mock_provider import MockTTSProvider
from tests.audio_validator import AudioValidator, SubtitleValidator


class TestMockPipeline:

    @pytest.fixture
    def sample_chapters(self):
        return [
            ExtractedChapter(
                num=1,
                title="The Crimson Tower",
                text="Arthur stood before the tower. The clock struck noon.",
                sentences=["Arthur stood before the tower.", "The clock struck noon."]
            ),
            ExtractedChapter(
                num=2,
                title="Whispers in the Archive",
                text="Inside the archive, old vellum smelled of cedar. Eleanor read the manuscript.",
                sentences=["Inside the archive, old vellum smelled of cedar.", "Eleanor read the manuscript."]
            )
        ]

    @pytest.fixture
    def voice_file_path(self):
        return os.path.join(_ROOT, "tests", "fixtures", "audio", "synthetic_voice_reference.wav")

    def test_full_pipeline_multi_chapter_generation(self, sample_chapters, voice_file_path):
        with tempfile.TemporaryDirectory() as td:
            cfg = AudiobookConfig(
                book_title="Antiquity",
                author="Arthur Pendelton",
                voice_file=voice_file_path,
                output_dir=td,
                output_format="mp3",
                tts_provider_name="mock",
                worker_count=1,
                export_lrc=True,
                export_srt=True,
                export_vtt=True,
                export_text=True,
                single_file_mode=False,
            )

            # Create mock pool
            mock_pool = ProviderPool(lambda dev: MockTTSProvider(cfg, device=dev), ["cpu"], "mock")

            log_q = queue.Queue()
            prog_q = queue.Queue()
            cancel = CancelToken()

            out_files = run_pipeline(
                config=cfg,
                chapters=sample_chapters,
                log_queue=log_q,
                prog_queue=prog_q,
                cancel=cancel,
            )

            assert len(out_files) == 2, f"Expected 2 output files, got {out_files}"

            # Validate generated individual chapter MP3 and subtitle files in output directory
            for ch_idx in (1, 2):
                chapter_files = [
                    f for f in out_files
                    if f"Chapter {ch_idx}" in f and f.endswith(".mp3")
                ]
                assert len(chapter_files) == 1, f"Missing MP3 for chapter {ch_idx} in {out_files}"
                ch_mp3 = chapter_files[0]
                assert os.path.exists(ch_mp3)

                val = AudioValidator.validate_audio_file(ch_mp3, min_duration=0.5)
                assert val.is_valid, f"Chapter {ch_idx} audio invalid: {val.error_message}"
                assert not val.is_silent

                # Validate subtitles
                lrc_path = os.path.splitext(ch_mp3)[0] + ".lrc"
                srt_path = os.path.splitext(ch_mp3)[0] + ".srt"
                vtt_path = os.path.splitext(ch_mp3)[0] + ".vtt"
                txt_path = os.path.splitext(ch_mp3)[0] + ".txt"

                assert os.path.exists(lrc_path), f"LRC missing: {lrc_path}"
                assert os.path.exists(srt_path), f"SRT missing: {srt_path}"
                assert os.path.exists(vtt_path), f"VTT missing: {vtt_path}"
                assert os.path.exists(txt_path), f"TXT missing: {txt_path}"

                lrc_ok, lrc_err = SubtitleValidator.validate_lrc(lrc_path)
                assert lrc_ok, f"LRC validation failed: {lrc_err}"

                srt_ok, srt_err = SubtitleValidator.validate_srt(srt_path)
                assert srt_ok, f"SRT validation failed: {srt_err}"

                vtt_ok, vtt_err = SubtitleValidator.validate_vtt(vtt_path)
                assert vtt_ok, f"VTT validation failed: {vtt_err}"
