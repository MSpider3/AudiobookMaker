"""
generate_test_audio.py
======================
Generates deterministic synthetic audio fixtures:
- synthetic_voice_reference.wav (24kHz, 16-bit PCM, mono, 4.0s)
- expected_audio_properties.json
"""

from __future__ import annotations

import json
import os
import numpy as np
import soundfile as sf


def get_fixtures_dir() -> str:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, "fixtures")


def generate_synthetic_voice(out_path: str, duration: float = 4.0, sample_rate: int = 24000) -> dict:
    """
    Generate synthetic harmonic audio mimicking voice formant structure.
    Has non-zero RMS, no NaNs, and clean smooth fades.
    """
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False, dtype=np.float32)

    # Fundamental voice frequency (F0) around 140 Hz with harmonics at 280, 560, 1120 Hz
    f0 = 140.0
    signal = (
        0.50 * np.sin(2 * np.pi * f0 * t) +
        0.25 * np.sin(2 * np.pi * (f0 * 2) * t) +
        0.15 * np.sin(2 * np.pi * (f0 * 4) * t) +
        0.10 * np.sin(2 * np.pi * (f0 * 8) * t)
    )

    # Modulate envelope to emulate syllables
    mod = 0.5 * (1.0 + np.sin(2 * np.pi * 3.5 * t))
    signal = signal * mod

    # Smooth attack and release (50ms fade in/out)
    fade_samples = int(0.05 * sample_rate)
    fade_in = np.linspace(0.0, 1.0, fade_samples, dtype=np.float32)
    fade_out = np.linspace(1.0, 0.0, fade_samples, dtype=np.float32)
    signal[:fade_samples] *= fade_in
    signal[-fade_samples:] *= fade_out

    # Scale to ~ -18 dBFS RMS
    rms = np.sqrt(np.mean(signal**2))
    target_rms = 0.12  # ~ -18 dBFS
    if rms > 0:
        signal = signal * (target_rms / rms)

    # Peak clamp
    signal = np.clip(signal, -0.95, 0.95)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sf.write(out_path, signal, sample_rate, subtype="PCM_16")

    # Compute actual properties
    actual_rms = float(np.sqrt(np.mean(signal**2)))
    peak = float(np.max(np.abs(signal)))
    properties = {
        "file_path": out_path,
        "sample_rate": sample_rate,
        "channels": 1,
        "duration_sec": float(duration),
        "total_samples": len(signal),
        "rms": actual_rms,
        "peak_amplitude": peak,
        "format": "WAV",
        "subtype": "PCM_16"
    }
    return properties


def generate_all_audio_fixtures() -> str:
    fixtures_dir = get_fixtures_dir()
    audio_dir = os.path.join(fixtures_dir, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    wav_path = os.path.join(audio_dir, "synthetic_voice_reference.wav")
    json_path = os.path.join(audio_dir, "expected_audio_properties.json")

    print(f"Generating synthetic voice reference: {wav_path}...")
    props = generate_synthetic_voice(wav_path)

    print(f"Saving expected properties to {json_path}...")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(props, f, indent=2)

    print("Audio fixtures generated successfully!")
    return wav_path


if __name__ == "__main__":
    generate_all_audio_fixtures()
