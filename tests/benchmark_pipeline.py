"""
benchmark_pipeline.py
=====================
Profiling benchmark for AudiobookMaker pipeline.
Times each phase independently to measure optimization impact.

Usage:
    python benchmark_pipeline.py
"""
import os
import sys
import time
import shutil
import tempfile

# Ensure project root is importable
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def _timer(label):
    """Simple context-manager timer."""
    class Timer:
        def __init__(self, label):
            self.label = label
            self.elapsed = 0.0
        def __enter__(self):
            self.start = time.perf_counter()
            return self
        def __exit__(self, *args):
            self.elapsed = time.perf_counter() - self.start
            print(f"  ⏱  {self.label}: {self.elapsed:.3f}s")
    return Timer(label)


def run_benchmark():
    print("=" * 70)
    print("     AudiobookMaker Pipeline Benchmark")
    print("=" * 70)

    # ── Phase 1: Import timing ────────────────────────────────────────────
    with _timer("Import core modules") as t_import:
        import torch
        import numpy as np
        import soundfile as sf
        from audiobook_factory.pipeline import AudiobookConfig
        from audiobook_factory.text_processing import smart_sentence_splitter, normalize_text
        from audiobook_factory.tts_providers import get_tts_provider
        from audiobook_factory.filename_sanitizer import make_safe_filename

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n  Device: {device}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # ── Phase 2: Text processing benchmark ────────────────────────────────
    print(f"\n{'─' * 70}")
    print("Phase 2: Text Processing")

    sample_text = """
    The mysterious world of Tingen was silent under the silver moonlight.
    Klein Moretti woke up from a table, feeling a cold sensation on his temple.
    Inside the revolver, a single brass bullet sat quietly in the cylinder.
    He reached out his hand, touching the sticky red fluid on his forehead.
    The Beyonder characteristics were incredibly dangerous, capable of
    corrupting anyone who dared to consume them without proper preparation.
    """ * 20  # ~120 sentences worth

    with _timer("normalize_text()") as t_norm:
        for _ in range(100):
            normalize_text(sample_text)

    with _timer("smart_sentence_splitter()") as t_split:
        for _ in range(100):
            chunks = smart_sentence_splitter(sample_text, 399)
    print(f"    → {len(chunks)} chunks per call")

    # ── Phase 3: Model loading benchmark ──────────────────────────────────
    print(f"\n{'─' * 70}")
    print("Phase 3: TTS Model Loading")

    voice_file = "./narrator_voice/LOTM_narrator_voice_no_space.wav"
    if not os.path.exists(voice_file):
        # Try to find any WAV in narrator_voice/
        voice_dir = os.path.join(_ROOT, "narrator_voice")
        if os.path.exists(voice_dir):
            wavs = [f for f in os.listdir(voice_dir) if f.endswith(".wav")]
            if wavs:
                voice_file = os.path.join(voice_dir, wavs[0])
            else:
                print("  ⚠ No voice WAV found — skipping TTS benchmarks.")
                _print_summary(t_import, t_norm, t_split)
                return
        else:
            print("  ⚠ No narrator_voice/ dir — skipping TTS benchmarks.")
            _print_summary(t_import, t_norm, t_split)
            return

    config = AudiobookConfig(
        voice_file=voice_file,
        tts_provider_name="qwen",
        tts_model_name="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        device=device,
        worker_count=4,
        parallel_mode="chunks",
    )

    with _timer("Model initialization") as t_model:
        provider = get_tts_provider(config.tts_provider_name, config)

    # ── Phase 4: Single synthesis benchmark ───────────────────────────────
    print(f"\n{'─' * 70}")
    print("Phase 4: Single Chunk TTS Synthesis")

    test_texts = [
        "The mysterious world of Tingen was silent under the silver moonlight.",
        "Klein Moretti woke up from a table, feeling a cold sensation on his temple.",
        "Inside the revolver, a single brass bullet sat quietly in the cylinder.",
        "He reached out his hand, touching the sticky red fluid on his forehead.",
    ]

    test_dir = tempfile.mkdtemp(prefix="abm_bench_")
    try:
        # Sequential
        seq_paths = [os.path.join(test_dir, f"seq_{i}.wav") for i in range(len(test_texts))]
        with _timer(f"Sequential synthesis ({len(test_texts)} chunks)") as t_seq:
            for text, path in zip(test_texts, seq_paths):
                provider.synthesize(text, config.voice_file, path)

        # Batch
        batch_paths = [os.path.join(test_dir, f"batch_{i}.wav") for i in range(len(test_texts))]
        voice_bytes = b""
        if os.path.exists(config.voice_file):
            with open(config.voice_file, "rb") as vf:
                voice_bytes = vf.read()
        with _timer(f"Batch synthesis ({len(test_texts)} chunks, batch_size={len(test_texts)})") as t_batch:
            batch_results = provider.synthesize_batch(test_texts, voice_bytes, return_bytes=True)
            for (audio_bytes, dur), bp in zip(batch_results, batch_paths):
                if isinstance(audio_bytes, bytes):
                    with open(bp, "wb") as fh:
                        fh.write(audio_bytes)

        # ── Phase 5: In-memory concat benchmark ──────────────────────────────
        print(f"\n{'─' * 70}")
        print("Phase 5: Audio Post-Processing")

        wav_arrays = []
        for p in seq_paths:
            if os.path.exists(p):
                audio, sr = sf.read(p, dtype="float32")
                wav_arrays.append(audio)

        pause = np.zeros(int(0.5 * 24000), dtype=np.float32)

        with _timer("In-memory numpy concat") as t_concat:
            for _ in range(100):
                segments = []
                for i, a in enumerate(wav_arrays):
                    segments.append(a)
                    if i < len(wav_arrays) - 1:
                        segments.append(pause)
                combined = np.concatenate(segments)

        total_duration = len(combined) / 24000
        print(f"    → {total_duration:.1f}s of audio concatenated")

        with _timer("FFmpeg encode (stdin pipe)") as t_ffmpeg:
            import subprocess
            out_mp3 = os.path.join(test_dir, "bench_output.mp3")
            subprocess.run(
                ["ffmpeg", "-y",
                 "-f", "f32le", "-ar", "24000", "-ac", "1",
                 "-i", "pipe:0",
                 "-af", "loudnorm=I=-18:TP=-1.5:LRA=11",
                 "-ar", "44100", "-ac", "2", "-c:a", "libmp3lame", "-q:a", "0",
                 out_mp3],
                input=combined.tobytes(),
                check=True, capture_output=True,
            )

        with _timer("Rust master_audio (decodes + EBUR128 + LAME)") as t_rust_master:
            import audiobook_rust
            out_mp3_rust = os.path.join(test_dir, "bench_output_rust.mp3")
            audiobook_rust.master_audio(
                seq_paths,
                out_mp3_rust,
                0.5,
                24000,
                -18.0,
                -1.5,
                64
            )

    finally:
        shutil.rmtree(test_dir, ignore_errors=True)
        provider.cleanup()

    # ── Summary ───────────────────────────────────────────────────────────
    print(f"\n{'=' * 70}")
    print("BENCHMARK SUMMARY")
    print(f"{'=' * 70}")
    print(f"  Import modules:      {t_import.elapsed:8.3f}s")
    print(f"  Text normalize (×100):{t_norm.elapsed:8.3f}s  ({t_norm.elapsed/100*1000:.1f}ms/call)")
    print(f"  Sentence split (×100):{t_split.elapsed:8.3f}s  ({t_split.elapsed/100*1000:.1f}ms/call)")
    print(f"  Model load:          {t_model.elapsed:8.3f}s")
    print(f"  Sequential TTS ×{len(test_texts)}:   {t_seq.elapsed:8.3f}s  ({t_seq.elapsed/len(test_texts):.3f}s/chunk)")
    print(f"  Batch TTS ×{len(test_texts)}:        {t_batch.elapsed:8.3f}s  ({t_batch.elapsed/len(test_texts):.3f}s/chunk)")
    print(f"  Speedup (batch):     {t_seq.elapsed/max(t_batch.elapsed, 0.001):.2f}x")
    print(f"  Numpy concat (×100): {t_concat.elapsed:8.3f}s  ({t_concat.elapsed/100*1000:.2f}ms/call)")
    print(f"  FFmpeg encode:       {t_ffmpeg.elapsed:8.3f}s")
    print(f"  Rust master:         {t_rust_master.elapsed:8.3f}s")
    if t_rust_master.elapsed > 0.001:
        print(f"  Post-proc speedup:   {t_ffmpeg.elapsed/t_rust_master.elapsed:.2f}x")
    print()
    print(f"  Estimated chapter time (60 chunks, batch=4):")
    per_batch = t_batch.elapsed / len(test_texts) * 4  # ~4 per batch
    n_batches = 60 / 4
    est_tts = n_batches * per_batch
    est_total = est_tts + t_rust_master.elapsed
    print(f"    TTS inference:    ~{est_tts:.1f}s")
    print(f"    Post-processing:  ~{t_rust_master.elapsed:.1f}s")
    print(f"    Total estimated:  ~{est_total:.1f}s")
    print(f"{'=' * 70}")


def _print_summary(t_import, t_norm, t_split):
    """Partial summary when TTS couldn't run."""
    print(f"\n{'=' * 70}")
    print("PARTIAL BENCHMARK (no TTS)")
    print(f"{'=' * 70}")
    print(f"  Import modules:      {t_import.elapsed:8.3f}s")
    print(f"  Text normalize (×100):{t_norm.elapsed:8.3f}s  ({t_norm.elapsed/100*1000:.1f}ms/call)")
    print(f"  Sentence split (×100):{t_split.elapsed:8.3f}s  ({t_split.elapsed/100*1000:.1f}ms/call)")


if __name__ == "__main__":
    run_benchmark()

