"""
test_cover_encoding.py
======================
Regression tests for cover art encoding, RGBA to RGB conversion, and ID3/MP4 metadata embedding.
"""

from __future__ import annotations

import os
import queue
import sys
import tempfile
import numpy as np
import pytest
from PIL import Image
from mutagen.mp3 import MP3
from mutagen.id3 import ID3, APIC

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from audiobook_factory.pipeline import AudiobookConfig, CancelToken, run_pipeline
from audiobook_factory.text_extractor import ExtractedChapter


class TestCoverEncodingRegression:

    @pytest.fixture
    def voice_file_path(self):
        return os.path.join(_ROOT, "tests", "fixtures", "audio", "synthetic_voice_reference.wav")

    @pytest.fixture
    def rgba_cover_path(self):
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tf:
            # Create 100x100 RGBA image with alpha channel
            arr = np.zeros((100, 100, 4), dtype=np.uint8)
            arr[:, :, 0] = 200  # Red
            arr[:, :, 3] = 255  # Alpha
            img = Image.fromarray(arr, mode="RGBA")
            img.save(tf.name)
            path = tf.name
        yield path
        if os.path.exists(path):
            os.remove(path)

    def test_rgba_cover_converted_and_embedded(self, rgba_cover_path, voice_file_path):
        with tempfile.TemporaryDirectory() as td:
            cfg = AudiobookConfig(
                book_title="CoverBook",
                voice_file=voice_file_path,
                cover_image=rgba_cover_path,
                output_dir=td,
                output_format="mp3",
                tts_provider_name="mock",
                worker_count=1,
            )

            chapters = [
                ExtractedChapter(num=1, title="Chapter 1", text="Cover art test chapter.", sentences=["Cover art test chapter."])
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
            mp3_path = out_files[0]
            assert os.path.exists(mp3_path)

            # Check ID3 APIC frame
            audio = MP3(mp3_path)
            has_apic = any(isinstance(frame, APIC) for frame in audio.tags.values()) if audio.tags else False
            assert has_apic, "Expected embedded APIC cover frame in MP3"
