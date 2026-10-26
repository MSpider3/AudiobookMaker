"""
audiobook_factory/engine_trial.py
=================================
Hear the same text in several TTS engines, to choose one for a book.

Each engine reads the test text with the narrator clip; the result is one
WAV per engine plus its load time, speed and peak GPU memory. Engines are
run one after another and freed in between. Used by the "Try engines"
notebooks; also a command::

    python -m audiobook_factory.engine_trial --engines qwen,higgs \
        --voice narrator.wav --transcript "What the clip says." \
        --text "A paragraph of the book." --out trial_output

Install engines first with ``python -m audiobook_factory.engine_envs install``.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
import time
from dataclasses import dataclass
from typing import Callable

logger = logging.getLogger(__name__)

__all__ = ["TrialResult", "try_engines", "score_with_whisper"]

_HF_HUB: str = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")


@dataclass
class TrialResult:
    """How one engine did on the test text."""

    engine: str
    ok: bool = False
    wav_path: str = ""
    load_seconds: float = 0.0
    synth_seconds: float = 0.0
    audio_seconds: float = 0.0
    peak_vram_gb: float = 0.0
    error: str = ""
    word_error: float | None = None
    heard: str = ""

    @property
    def speed_x(self) -> float:
        """Seconds of audio per second of synthesis (above 1 is faster than real time)."""
        return round(self.audio_seconds / self.synth_seconds, 2) if self.synth_seconds else 0.0


def _peak_vram_gb() -> float:
    try:
        import torch
        if torch.cuda.is_available():
            return round(max(torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())) / 1e9, 1)
    except Exception:
        pass
    return 0.0


def _gpu_memory_used_gb() -> float:
    """GPU memory in use on device 0 by every process (engine workers included)."""
    try:
        import torch
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info(0)
            return round((total - free) / 1e9, 1)
    except Exception:
        pass
    return 0.0


def try_engines(
    engines: list[str],
    voice: str,
    transcript: str,
    text: str,
    out_dir: str,
    language: str = "English",
    device: str = "cuda:0",
    delete_models: bool = False,
    log: Callable[[str], None] = print,
) -> list[TrialResult]:
    """Has each engine read *text* in the voice of *voice*.

    Parameters
    ----------
    engines : list[str]
        Engine names, e.g. ``["qwen", "higgs"]``.
    voice, transcript : str
        Narrator clip and the words spoken in it.
    text : str
        Test text, cut into chunks like a chapter.
    out_dir : str
        Where ``<engine>.wav`` files are written.
    delete_models : bool
        Delete each engine's downloaded weights after its turn (for the
        small disks of Kaggle and Colab).
    """
    from audiobook_factory.chunk_planner import plan_chunks
    from audiobook_factory.pipeline import AudiobookConfig, _join_preview_audio
    from audiobook_factory.tts_providers import get_tts_provider

    os.makedirs(out_dir, exist_ok=True)
    with open(voice, "rb") as fh:
        voice_bytes = fh.read()
    results: list[TrialResult] = []
    for engine in engines:
        result = TrialResult(engine=engine)
        results.append(result)
        hub_before = set(os.listdir(_HF_HUB)) if os.path.isdir(_HF_HUB) else set()
        provider = None
        try:
            config = AudiobookConfig(tts_provider_name=engine, voice_file=voice, voice_transcript=transcript,
                                     language=language, seed=7)
            chunks = plan_chunks(text, None, config.max_len, config.pause, config.para_pause)
            log(f"▶ {engine}: loading...")
            provider = get_tts_provider(engine, config, device=device)
            started = time.time()
            provider.ensure_ready()
            result.load_seconds = time.time() - started
            started = time.time()
            outputs = provider.synthesize_batch([chunk.text for chunk in chunks], voice_bytes)
            result.synth_seconds = time.time() - started
            result.audio_seconds = round(sum(duration for _, duration in outputs), 1)
            result.peak_vram_gb = max(_peak_vram_gb(), _gpu_memory_used_gb())
            result.wav_path = os.path.join(out_dir, f"{engine}.wav")
            _join_preview_audio([audio for audio, _ in outputs], [c.pause_after for c in chunks], result.wav_path)
            result.ok = True
            log(f"  {engine}: {result.audio_seconds:.0f} s of audio in {result.synth_seconds:.0f} s "
                f"({result.speed_x}x), loaded in {result.load_seconds:.0f} s")
        except Exception as exc:
            result.error = f"{type(exc).__name__}: {exc}"
            log(f"  {engine}: ❌ {result.error}")
        finally:
            if provider is not None:
                try:
                    provider.cleanup()
                except Exception as exc:
                    logger.debug("cleanup of %s failed: %s", engine, exc)
            _free_gpu()
            if delete_models and os.path.isdir(_HF_HUB):
                for folder in set(os.listdir(_HF_HUB)) - hub_before:
                    if "whisper" not in folder:
                        shutil.rmtree(os.path.join(_HF_HUB, folder), ignore_errors=True)
    return results


def _free_gpu() -> None:
    import gc
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            for index in range(torch.cuda.device_count()):
                torch.cuda.reset_peak_memory_stats(index)
    except Exception:
        pass


def score_with_whisper(results: list[TrialResult], text: str, language: str = "English",
                       device: str = "cuda:0") -> None:
    """Fills ``word_error`` and ``heard`` of each result with a Whisper transcript."""
    from audiobook_factory.chunk_verifier import ChunkVerifier, error_rate
    import soundfile as sf

    verifier = ChunkVerifier("asr", language=language)
    try:
        for result in results:
            if not result.ok:
                continue
            samples, rate = sf.read(result.wav_path, dtype="float32")
            heard = verifier._transcribe(samples, rate, device) or ""
            result.heard = heard
            result.word_error = round(error_rate(text, heard), 3)
    finally:
        verifier.close()


def table(results: list[TrialResult]) -> str:
    """The results as a Markdown table."""
    rows = ["| Engine | Result | Speed | Load | GPU memory | Word errors |", "|---|---|---|---|---|---|"]
    for r in results:
        state = "✅" if r.ok else f"❌ {r.error[:80]}"
        errors = f"{r.word_error:.0%}" if r.word_error is not None else ""
        rows.append(f"| {r.engine} | {state} | {r.speed_x}x | {r.load_seconds:.0f} s | {r.peak_vram_gb} GB | {errors} |")
    return "\n".join(rows)


def _main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--engines", required=True, help="Comma-separated engine names.")
    parser.add_argument("--voice", required=True)
    parser.add_argument("--transcript", default="")
    parser.add_argument("--text", required=True)
    parser.add_argument("--language", default="English")
    parser.add_argument("--out", default="engine_trial")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--delete-models", action="store_true")
    parser.add_argument("--whisper", action="store_true", help="Score each engine with a Whisper transcript.")
    args = parser.parse_args(argv)
    results = try_engines([e.strip() for e in args.engines.split(",") if e.strip()], args.voice, args.transcript,
                          args.text, args.out, args.language, args.device, args.delete_models)
    if args.whisper:
        score_with_whisper(results, args.text, args.language, args.device)
    print(table(results))
    with open(os.path.join(args.out, "results.json"), "w", encoding="utf-8") as fh:
        json.dump([{**r.__dict__, "speed_x": r.speed_x} for r in results], fh, indent=1, ensure_ascii=False)
    return 0 if all(r.ok for r in results) else 1


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
