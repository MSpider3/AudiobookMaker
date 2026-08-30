"""
test_chapter_pipeline.py
========================
Integration test for the 3-stage overlapped chapter pipeline
(audiobook_factory/chapter_pipeline.py:run_chapter_pipeline).
"""

from __future__ import annotations

import os
import sys
import tempfile
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from audiobook_factory.pipeline import AudiobookConfig, CancelToken
from audiobook_factory.chapter_pipeline import run_chapter_pipeline
from audiobook_factory.gpu_pool import ProviderPool
from tests.fixtures.mock_provider import MockTTSProvider
from tests.audio_validator import AudioValidator


class TestChapterPipeline:

    @pytest.fixture
    def voice_bytes(self):
        wav_path = os.path.join(_ROOT, "tests", "fixtures", "audio", "synthetic_voice_reference.wav")
        with open(wav_path, "rb") as f:
            return f.read()

    def test_run_chapter_pipeline_single_device(self, voice_bytes):
        with tempfile.TemporaryDirectory() as td:
            cfg = AudiobookConfig(
                output_dir=td,
                sample_rate=24000,
                max_len=100,
                pause=0.3,
            )
            pool = ProviderPool(lambda dev: MockTTSProvider(cfg, device=dev), ["cpu"], "mock")

            sentences = [
                "Sentence number one in chapter one.",
                "Sentence number two in chapter one.",
                "Sentence number three in chapter one.",
                "Sentence number four in chapter one."
            ]

            out_wav = os.path.join(td, "chapter_1_mastered.wav")
            cancel = CancelToken()
            completed_chunk_indices = []

            def _on_chunk(idx: int):
                completed_chunk_indices.append(idx)

            durations = run_chapter_pipeline(
                sentences=sentences,
                voice_ref=voice_bytes,
                out_wav_path=out_wav,
                out_dir=td,
                chapter_index=1,
                config=cfg,
                pool=pool,
                cancel_token=cancel,
                log_callback=lambda m: None,
                chunk_completed_cb=_on_chunk,
            )

            assert os.path.exists(out_wav)
            assert len(durations) == len(sentences)
            assert len(completed_chunk_indices) == len(sentences)

            val = AudioValidator.validate_audio_file(out_wav, min_duration=1.0)
            assert val.is_valid, f"Mastered audio failed validation: {val.error_message}"
            assert not val.is_silent
            assert val.sample_rate == 24000
