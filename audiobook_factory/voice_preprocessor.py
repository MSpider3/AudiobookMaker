"""
audiobook_factory/voice_preprocessor.py
========================================
Audio preprocessing pipeline for narrator voice samples
before TTS cloning.

Based on Mangio-RVC-Fork preprocessing techniques adapted
for single-file voice cleaning (no RVC model required).

Processing steps (all individually toggleable):
  1. Noise Reduction         - noisereduce
  2. Noise Gate              - librosa RMS gate
  3. High-Pass Filter        - scipy Butterworth
  4. Silence Removal         - custom RMS slicer
  5. Normalize Volume        - peak normalization to target dBFS
  6. Formant Shifting        - spectral envelope adjustment (optional)
  7. Resample                - to target sample rate
"""
from __future__ import annotations

import io
import numpy as np
import soundfile as sf
from dataclasses import dataclass, field


# ══════════════════════════════════════════════════════════════════════════════
# Config dataclass
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class PreprocessConfig:
    # Step 1: Noise Reduction
    noise_reduce:           bool  = True
    noise_reduce_strength:  float = 0.5    # 0.0 – 1.0

    # Step 2: Noise Gate
    noise_gate:             bool  = True
    noise_gate_threshold_db: float = -40.0  # dB, frames below this → silence

    # Step 3: High-Pass Filter
    highpass_filter:        bool  = True
    highpass_cutoff_hz:     int   = 80      # Hz

    # Step 4: Silence Removal
    silence_removal:        bool  = True
    silence_threshold_db:   float = -40.0   # dB
    min_segment_ms:         int   = 300     # ms — shorter segments discarded
    max_silence_kept_ms:    int   = 500     # ms retained between segments

    # Step 5: Normalize Volume
    normalize_volume:       bool  = True
    normalize_target_dbfs:  float = -3.0   # dBFS

    # Step 6: Formant Shifting (optional / experimental)
    formant_shift:          bool  = False
    formant_quefrency:      float = 1.0    # 0.0 – 16.0
    formant_timbre:         float = 1.0    # 0.0 – 16.0

    # Step 7: Resample
    resample:               bool  = False
    target_sample_rate:     int   = 44100


# ══════════════════════════════════════════════════════════════════════════════
# Pipeline
# ══════════════════════════════════════════════════════════════════════════════

def preprocess(
    input_bytes: bytes,
    config: PreprocessConfig | None = None,
    log_fn=None,
) -> bytes:
    """
    Process a raw WAV byte-string through the configured pipeline.

    Parameters
    ----------
    input_bytes : raw audio bytes (WAV format)
    config      : PreprocessConfig; uses defaults if None
    log_fn      : optional callable(str) for progress reporting

    Returns
    -------
    bytes : processed WAV audio bytes
    """
    if config is None:
        config = PreprocessConfig()

    def log(msg: str):
        if log_fn:
            log_fn(msg)
        else:
            print(msg)

    # Read input
    audio, sr = sf.read(io.BytesIO(input_bytes), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)   # mono

    log(f"[Preprocess] Loaded audio: {len(audio)/sr:.1f}s @ {sr}Hz")

    # ── Step 1: Noise Reduction ───────────────────────────────────────────────
    if config.noise_reduce:
        try:
            import noisereduce as nr
            log("[Preprocess] Step 1: Noise reduction...")
            audio = nr.reduce_noise(
                y=audio, sr=sr,
                prop_decrease=config.noise_reduce_strength,
            ).astype(np.float32)
        except ImportError:
            log("[Preprocess] WARNING: noisereduce not installed — skipping.")

    # ── Step 2: Noise Gate ────────────────────────────────────────────────────
    if config.noise_gate:
        try:
            import librosa
            log("[Preprocess] Step 2: Noise gate...")
            frame_length = 2048
            hop_length   = 1024
            rms = librosa.feature.rms(
                y=audio, frame_length=frame_length, hop_length=hop_length
            )[0]
            db  = librosa.amplitude_to_db(rms, ref=1.0)
            for i, below in enumerate(db < config.noise_gate_threshold_db):
                if below:
                    start = i * hop_length
                    end   = min(start + hop_length, len(audio))
                    audio[start:end] = 0.0
        except ImportError:
            log("[Preprocess] WARNING: librosa not installed — skipping noise gate.")

    # ── Step 3: High-Pass Filter ──────────────────────────────────────────────
    if config.highpass_filter:
        try:
            from scipy.signal import butter, filtfilt
            log(f"[Preprocess] Step 3: High-pass filter @ {config.highpass_cutoff_hz}Hz...")
            nyq = sr / 2.0
            wn  = config.highpass_cutoff_hz / nyq
            if 0 < wn < 1:
                b, a  = butter(N=5, Wn=wn, btype="high")
                audio = filtfilt(b, a, audio).astype(np.float32)
        except ImportError:
            log("[Preprocess] WARNING: scipy not installed — skipping high-pass filter.")

    # ── Step 4: Silence Removal ───────────────────────────────────────────────
    if config.silence_removal:
        log("[Preprocess] Step 4: Silence removal...")
        audio = _remove_silence(
            audio, sr,
            threshold_db=config.silence_threshold_db,
            min_segment_ms=config.min_segment_ms,
            max_silence_kept_ms=config.max_silence_kept_ms,
        )

    # ── Step 5: Normalize Volume ──────────────────────────────────────────────
    if config.normalize_volume:
        log(f"[Preprocess] Step 5: Normalize to {config.normalize_target_dbfs} dBFS...")
        audio = _normalize_peak(audio, config.normalize_target_dbfs)

    # ── Step 6: Formant Shifting ──────────────────────────────────────────────
    if config.formant_shift:
        try:
            log("[Preprocess] Step 6: Formant shifting (experimental)...")
            audio = _formant_shift(
                audio, sr,
                quefrency=config.formant_quefrency,
                timbre=config.formant_timbre,
            )
        except Exception as e:
            log(f"[Preprocess] Formant shift failed: {e} — skipping.")

    # ── Step 7: Resample ──────────────────────────────────────────────────────
    if config.resample and config.target_sample_rate != sr:
        try:
            import librosa
            log(f"[Preprocess] Step 7: Resample {sr}→{config.target_sample_rate}Hz...")
            audio = librosa.resample(
                audio, orig_sr=sr, target_sr=config.target_sample_rate
            ).astype(np.float32)
            sr = config.target_sample_rate
        except ImportError:
            log("[Preprocess] WARNING: librosa not installed — skipping resample.")

    # ── Encode output as WAV bytes ────────────────────────────────────────────
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
    log("[Preprocess] Done.")
    return buf.getvalue()


# ══════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ══════════════════════════════════════════════════════════════════════════════

def _remove_silence(
    audio: np.ndarray,
    sr: int,
    threshold_db: float,
    min_segment_ms: int,
    max_silence_kept_ms: int,
) -> np.ndarray:
    """
    RMS-based silence slicer, inspired by Mangio-RVC slicer2.py.
    Retains up to max_silence_kept_ms of silence between voiced segments.
    """
    hop   = int(sr * 0.010)     # 10ms frames
    rms   = _rms_frames(audio, hop)
    ref   = np.max(np.abs(audio)) if np.max(np.abs(audio)) > 0 else 1.0
    db    = 20 * np.log10(rms / ref + 1e-9)
    voiced = db > threshold_db

    min_frames     = int(min_segment_ms   / 10)
    max_sil_frames = int(max_silence_kept_ms / 10)

    segments = []
    in_seg   = False
    seg_start = 0
    sil_count = 0

    for i, v in enumerate(voiced):
        if v:
            if not in_seg:
                in_seg    = True
                seg_start = max(0, i - min(sil_count, max_sil_frames))
            sil_count = 0
        else:
            if in_seg:
                sil_count += 1
                if sil_count > max_sil_frames:
                    seg_end = i - sil_count + max_sil_frames
                    if seg_end - seg_start >= min_frames:
                        segments.append((seg_start, seg_end))
                    in_seg = False
                    sil_count = 0

    if in_seg:
        segments.append((seg_start, len(voiced)))

    if not segments:
        return audio

    parts = []
    silence = np.zeros(int(sr * 0.05), dtype=np.float32)  # 50ms gap between segments
    for s, e in segments:
        parts.append(audio[s * hop : e * hop])
        parts.append(silence)

    return np.concatenate(parts)


def _rms_frames(audio: np.ndarray, hop: int) -> np.ndarray:
    n_frames = len(audio) // hop + 1
    rms = np.zeros(n_frames)
    for i in range(n_frames):
        frame = audio[i * hop : (i + 1) * hop]
        rms[i] = np.sqrt(np.mean(frame ** 2)) if len(frame) > 0 else 0.0
    return rms


def _normalize_peak(audio: np.ndarray, target_dbfs: float) -> np.ndarray:
    peak = np.max(np.abs(audio))
    if peak == 0:
        return audio
    target_amplitude = 10 ** (target_dbfs / 20.0)
    return (audio / peak * target_amplitude).astype(np.float32)


def _formant_shift(
    audio: np.ndarray, sr: int, quefrency: float, timbre: float
) -> np.ndarray:
    """
    Simple cepstrum-based formant envelope shift.
    Adapted from Mangio-RVC StftPitchShift approach.
    """
    from scipy.fft import fft, ifft
    n     = len(audio)
    spec  = fft(audio)
    log_spec = np.log(np.abs(spec) + 1e-9)

    # Cepstrum
    cepstrum = np.real(ifft(log_spec))

    # Lifter — zero out high-quefrency components
    q = max(1, int(quefrency * sr / 1000))
    cepstrum[q : n - q] = 0.0

    # Envelope
    envelope = np.exp(np.real(fft(cepstrum)))

    # Mix timbre
    mixed = spec * (envelope ** (timbre - 1.0))
    return np.real(ifft(mixed)).astype(np.float32)[:n]
