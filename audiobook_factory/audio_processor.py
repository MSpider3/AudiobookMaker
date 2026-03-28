"""
audiobook_factory/audio_processor.py
======================================
Backward-compatible shim.

All TTS logic now lives in:
    audiobook_factory/tts_providers/qwen_provider.py  (QwenTTSProvider)

This file preserves the old `tts_consumer()` function signature so that any
remaining direct callers (e.g. old pipeline.py code) keep working.
New code should use the provider abstraction directly via
    from audiobook_factory.tts_providers import get_tts_provider
"""
from __future__ import annotations

import queue as _queue


# Keep a module-level provider instance so the model loads only once
# across multiple calls from the old tts_consumer() interface.
_provider_instance = None


def tts_consumer(job_queue, results_queue, args):
    """
    Backward-compatible TTS consumer loop.

    Reads (idx, text, out_path, voice_ref) tuples from *job_queue* until
    it receives the sentinel string "STOP", synthesises each chunk via
    QwenTTSProvider, and puts (idx, text, out_path) into *results_queue*.
    """
    global _provider_instance

    # Lazily build a minimal AudiobookConfig from the args object.
    from audiobook_factory.pipeline import AudiobookConfig  # avoid circular at import time
    from audiobook_factory.tts_providers import get_tts_provider

    cfg = AudiobookConfig(
        voice_file=getattr(args, "voice_file", ""),
        temperature=getattr(args, "temperature", 0.7),
        top_p=getattr(args, "top_p", 0.8),
        device=getattr(args, "device", "cuda"),
        tts_model_name=getattr(args, "tts_model_name", "qwen"),
    )

    if _provider_instance is None:
        _provider_instance = get_tts_provider("qwen", cfg)

    provider = _provider_instance
    gen_count = 0

    while True:
        try:
            job = job_queue.get(timeout=5)
        except _queue.Empty:
            continue

        if job == "STOP":
            print("\n    [audio_processor] STOP signal — exiting.")
            break

        # Accept 3-tuple (idx, text, path) or 4-tuple (idx, text, path, voice_ref)
        if len(job) == 4:
            idx, text, out_path, voice_ref = job
        else:
            idx, text, out_path = job
            voice_ref = cfg.voice_file

        print(f"\r  > [TTS] chunk {idx + 1}…", end="", flush=True)
        try:
            provider.synthesize(text, voice_ref, out_path)
            results_queue.put((idx, text, out_path))
        except Exception as exc:
            print(f"\n    [audio_processor] Error on chunk {idx}: {exc}")

        gen_count += 1
        if gen_count % 10 == 0:
            try:
                import torch
                torch.cuda.empty_cache()
            except ImportError:
                pass
