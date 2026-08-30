"""
audio_validator.py
==================
Reusable audio and subtitle output validation utilities.
Verifies container validity, sample rates, channels, duration,
signal levels (RMS, peak amplitude), silence detection, NaN/Inf checks,
and subtitle timestamp monotonicity.
"""

from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass
from typing import Literal

import numpy as np
import soundfile as sf


@dataclass
class AudioValidationResult:
    is_valid: bool
    file_path: str
    format: str
    duration_sec: float
    sample_rate: int
    channels: int
    rms: float
    peak: float
    is_silent: bool
    has_nan: bool
    has_inf: bool
    error_message: str = ""


class AudioValidator:
    """Validator for synthesized and mastered audio files."""

    DEFAULT_MIN_RMS: float = 0.005  # Below this is considered silence
    DEFAULT_MIN_SIZE_BYTES: int = 1000

    @classmethod
    def validate_audio_file(
        cls,
        file_path: str,
        *,
        min_duration: float = 0.1,
        expected_sample_rate: int | None = None,
        min_rms: float = DEFAULT_MIN_RMS,
        allow_silence: bool = False,
    ) -> AudioValidationResult:
        """Thoroughly inspect an audio file on disk."""
        if not os.path.exists(file_path):
            return AudioValidationResult(
                is_valid=False,
                file_path=file_path,
                format="unknown",
                duration_sec=0.0,
                sample_rate=0,
                channels=0,
                rms=0.0,
                peak=0.0,
                is_silent=True,
                has_nan=False,
                has_inf=False,
                error_message=f"File does not exist: {file_path}"
            )

        file_size = os.path.getsize(file_path)
        if file_size < cls.DEFAULT_MIN_SIZE_BYTES:
            return AudioValidationResult(
                is_valid=False,
                file_path=file_path,
                format="unknown",
                duration_sec=0.0,
                sample_rate=0,
                channels=0,
                rms=0.0,
                peak=0.0,
                is_silent=True,
                has_nan=False,
                has_inf=False,
                error_message=f"File size ({file_size} bytes) below minimum threshold ({cls.DEFAULT_MIN_SIZE_BYTES} bytes)"
            )

        try:
            data, sr = sf.read(file_path, dtype="float32")
        except Exception as e:
            return AudioValidationResult(
                is_valid=False,
                file_path=file_path,
                format="corrupted",
                duration_sec=0.0,
                sample_rate=0,
                channels=0,
                rms=0.0,
                peak=0.0,
                is_silent=True,
                has_nan=False,
                has_inf=False,
                error_message=f"Soundfile failed to read audio container: {e}"
            )

        channels = 1 if data.ndim == 1 else data.shape[1]
        duration = len(data) / sr if sr > 0 else 0.0

        has_nan = bool(np.isnan(data).any())
        has_inf = bool(np.isinf(data).any())

        if has_nan or has_inf:
            return AudioValidationResult(
                is_valid=False,
                file_path=file_path,
                format=file_path.split(".")[-1],
                duration_sec=duration,
                sample_rate=sr,
                channels=channels,
                rms=0.0,
                peak=0.0,
                is_silent=True,
                has_nan=has_nan,
                has_inf=has_inf,
                error_message="Audio array contains NaN or Infinite floating point values"
            )

        rms = float(np.sqrt(np.mean(data**2)))
        peak = float(np.max(np.abs(data)))
        is_silent = rms < min_rms or peak == 0.0

        if not allow_silence and is_silent:
            return AudioValidationResult(
                is_valid=False,
                file_path=file_path,
                format=file_path.split(".")[-1],
                duration_sec=duration,
                sample_rate=sr,
                channels=channels,
                rms=rms,
                peak=peak,
                is_silent=True,
                has_nan=False,
                has_inf=False,
                error_message=f"Audio is silent (RMS: {rms:.6f} < threshold {min_rms:.6f}, peak: {peak:.6f})"
            )

        if duration < min_duration:
            return AudioValidationResult(
                is_valid=False,
                file_path=file_path,
                format=file_path.split(".")[-1],
                duration_sec=duration,
                sample_rate=sr,
                channels=channels,
                rms=rms,
                peak=peak,
                is_silent=is_silent,
                has_nan=False,
                has_inf=False,
                error_message=f"Audio duration ({duration:.3f}s) is shorter than required ({min_duration:.3f}s)"
            )

        if expected_sample_rate is not None and sr != expected_sample_rate:
            return AudioValidationResult(
                is_valid=False,
                file_path=file_path,
                format=file_path.split(".")[-1],
                duration_sec=duration,
                sample_rate=sr,
                channels=channels,
                rms=rms,
                peak=peak,
                is_silent=is_silent,
                has_nan=False,
                has_inf=False,
                error_message=f"Sample rate mismatch: {sr} != expected {expected_sample_rate}"
            )

        return AudioValidationResult(
            is_valid=True,
            file_path=file_path,
            format=file_path.split(".")[-1],
            duration_sec=duration,
            sample_rate=sr,
            channels=channels,
            rms=rms,
            peak=peak,
            is_silent=is_silent,
            has_nan=False,
            has_inf=False,
            error_message=""
        )

    @classmethod
    def validate_audio_bytes(
        cls,
        audio_bytes: bytes,
        *,
        min_duration: float = 0.1,
        min_rms: float = DEFAULT_MIN_RMS,
    ) -> AudioValidationResult:
        """Validate raw in-memory WAV audio bytes."""
        import io
        if len(audio_bytes) < cls.DEFAULT_MIN_SIZE_BYTES:
            return AudioValidationResult(
                is_valid=False,
                file_path="<bytes>",
                format="wav",
                duration_sec=0.0,
                sample_rate=0,
                channels=0,
                rms=0.0,
                peak=0.0,
                is_silent=True,
                has_nan=False,
                has_inf=False,
                error_message=f"Audio bytes ({len(audio_bytes)}) below minimum size {cls.DEFAULT_MIN_SIZE_BYTES}"
            )

        try:
            with io.BytesIO(audio_bytes) as buf:
                data, sr = sf.read(buf, dtype="float32")
        except Exception as e:
            return AudioValidationResult(
                is_valid=False,
                file_path="<bytes>",
                format="wav",
                duration_sec=0.0,
                sample_rate=0,
                channels=0,
                rms=0.0,
                peak=0.0,
                is_silent=True,
                has_nan=False,
                has_inf=False,
                error_message=f"Failed to read WAV bytes: {e}"
            )

        channels = 1 if data.ndim == 1 else data.shape[1]
        duration = len(data) / sr if sr > 0 else 0.0
        has_nan = bool(np.isnan(data).any())
        has_inf = bool(np.isinf(data).any())
        rms = float(np.sqrt(np.mean(data**2)))
        peak = float(np.max(np.abs(data)))
        is_silent = rms < min_rms or peak == 0.0

        is_valid = not (has_nan or has_inf or is_silent or duration < min_duration)
        err = ""
        if has_nan or has_inf:
            err = "NaN/Inf in audio buffer"
        elif is_silent:
            err = f"Silent audio buffer (RMS: {rms:.6f})"
        elif duration < min_duration:
            err = f"Audio duration too short ({duration:.3f}s)"

        return AudioValidationResult(
            is_valid=is_valid,
            file_path="<bytes>",
            format="wav",
            duration_sec=duration,
            sample_rate=sr,
            channels=channels,
            rms=rms,
            peak=peak,
            is_silent=is_silent,
            has_nan=has_nan,
            has_inf=has_inf,
            error_message=err
        )


class SubtitleValidator:
    """Validator for subtitle files (.lrc, .srt, .vtt)."""

    @classmethod
    def validate_lrc(cls, file_path: str) -> tuple[bool, str]:
        if not os.path.exists(file_path):
            return False, f"LRC file not found: {file_path}"
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        lines = content.strip().split("\n")
        timestamp_re = re.compile(r"\[(\d{2}):(\d{2}\.\d{2})\]")
        last_sec = -1.0
        entry_count = 0

        for line in lines:
            line = line.strip()
            if not line:
                continue
            m = timestamp_re.search(line)
            if m:
                mins, secs = float(m.group(1)), float(m.group(2))
                current_sec = mins * 60.0 + secs
                if current_sec < last_sec:
                    return False, f"Non-monotonic LRC timestamp: {current_sec:.2f}s after {last_sec:.2f}s"
                last_sec = current_sec
                entry_count += 1

        if entry_count == 0:
            return False, "No valid LRC timestamps found in file"
        return True, ""

    @classmethod
    def validate_srt(cls, file_path: str) -> tuple[bool, str]:
        if not os.path.exists(file_path):
            return False, f"SRT file not found: {file_path}"
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        srt_pattern = re.compile(
            r"(\d+)\s*\n(\d{2}:\d{2}:\d{2},\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2},\d{3})\s*\n(.*?)(?=\n\s*\n|\Z)",
            re.DOTALL
        )
        matches = list(srt_pattern.finditer(content))
        if not matches:
            return False, "No valid SRT subtitle blocks found"

        def _to_sec(t_str: str) -> float:
            h, m, rest = t_str.split(":")
            s, ms = rest.split(",")
            return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0

        last_end = 0.0
        for m in matches:
            idx = int(m.group(1))
            start_sec = _to_sec(m.group(2))
            end_sec = _to_sec(m.group(3))
            if start_sec >= end_sec:
                return False, f"SRT block {idx}: start time {start_sec:.3f} >= end time {end_sec:.3f}"
            if start_sec < last_end - 0.001:
                return False, f"SRT block {idx}: start {start_sec:.3f} before previous end {last_end:.3f}"
            last_end = end_sec

        return True, ""

    @classmethod
    def validate_vtt(cls, file_path: str) -> tuple[bool, str]:
        if not os.path.exists(file_path):
            return False, f"VTT file not found: {file_path}"
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        if not content.startswith("WEBVTT"):
            return False, "VTT file does not start with 'WEBVTT' signature"

        vtt_pattern = re.compile(
            r"(\d{2}:\d{2}:\d{2}\.\d{3})\s*-->\s*(\d{2}:\d{2}:\d{2}\.\d{3})"
        )
        matches = list(vtt_pattern.finditer(content))
        if not matches:
            return False, "No WebVTT cues found"

        def _to_sec(t_str: str) -> float:
            h, m, rest = t_str.split(":")
            s, ms = rest.split(".")
            return int(h) * 3600 + int(m) * 60 + int(s) + int(ms) / 1000.0

        last_end = 0.0
        for m in matches:
            start_sec = _to_sec(m.group(1))
            end_sec = _to_sec(m.group(2))
            if start_sec >= end_sec:
                return False, f"WebVTT cue: start {start_sec:.3f} >= end {end_sec:.3f}"
            if start_sec < last_end - 0.001:
                return False, f"WebVTT cue: start {start_sec:.3f} before previous end {last_end:.3f}"
            last_end = end_sec

        return True, ""
