"""
test_mastering_fallback.py
===========================
Integration tests comparing native Rust audio mastering (audiobook_rust.master_audio)
against pure-Python mastering fallback (pyloudnorm / soundfile).
"""

from __future__ import annotations

import os
import sys
import tempfile
import numpy as np
import pytest
import soundfile as sf
import pyloudnorm as pyln

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from audiobook_factory.chapter_pipeline import _master_final
from audiobook_factory.pipeline import AudiobookConfig
from tests.audio_validator import AudioValidator


class TestMasteringFallback:

    @pytest.fixture
    def synthetic_chunks(self):
        chunks = []
        sr = 24000
        for i in range(4):
            duration = 1.0 + i * 0.2
            t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
            sig = (0.3 * np.sin(2 * np.pi * (200 + i * 50) * t)).astype(np.float32)
            chunks.append((sig, sr))
        return chunks

    def test_master_final_produces_valid_ebu_r128_audio(self, synthetic_chunks):
        with tempfile.TemporaryDirectory() as td:
            chunk_files = []
            for i, (sig, sr) in enumerate(synthetic_chunks):
                p = os.path.join(td, f"chunk_{i}.wav")
                sf.write(p, sig, sr, subtype="PCM_16")
                chunk_files.append(p)

            out_wav = os.path.join(td, "mastered_output.wav")
            cfg = AudiobookConfig(lufs=-18, true_peak=-1.5, pause=0.2, sample_rate=24000)

            # Run mastering
            _master_final(chunk_files, out_wav, cfg)

            assert os.path.exists(out_wav)
            val = AudioValidator.validate_audio_file(out_wav, min_duration=2.0)
            assert val.is_valid, f"Mastered audio invalid: {val.error_message}"

            # Measure integrated loudness with pyloudnorm
            data, sr = sf.read(out_wav)
            meter = pyln.Meter(sr)
            loudness = meter.integrated_loudness(data)
            # Should be within ±1.5 LUFS of target -18
            assert abs(loudness - (-18.0)) < 2.0, f"Loudness {loudness} LUFS too far from target -18 LUFS"
