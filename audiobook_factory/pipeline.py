"""
audiobook_factory/pipeline.py
================================
Thread-safe audiobook generation orchestrator.
Replaces the monolithic main() in main.py.

Usage (from Gradio UI):
    from audiobook_factory.pipeline import AudiobookConfig, run_pipeline

    config  = AudiobookConfig(...)
    log_q   = queue.Queue()
    prog_q  = queue.Queue()

    thread = threading.Thread(
        target=run_pipeline,
        args=(config, chapters, log_q, prog_q)
    )
    thread.start()

    # Drain log_q in a loop to update Gradio textbox
"""
from __future__ import annotations

import os
import queue
import re
import shutil
import subprocess
import tempfile
import threading
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import soundfile as sf

from audiobook_factory.text_extractor import ExtractedChapter
from audiobook_factory.text_processing import smart_sentence_splitter


# ══════════════════════════════════════════════════════════════════════════════
# Config dataclass
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class AudiobookConfig:
    # Book metadata
    book_title:     str  = "Audiobook"
    author:         str  = "Unknown Author"
    cover_image:    str | None = None

    # Output
    output_dir:     str  = "./output"
    output_format:  str  = "mp3"     # mp3 | flac | wav | m4b

    # Voice
    voice_file:     str  = ""        # path to cloning WAV

    # TTS tuning
    temperature:    float = 0.3
    top_p:          float = 0.8
    max_len:        int   = 399      # max chars per TTS chunk

    # Pacing
    pause:          float = 0.5      # seconds between sentences
    para_pause:     float = 1.2      # seconds between paragraphs

    # Audio mastering
    lufs:           int   = -18
    true_peak:      float = -1.5

    # Device
    device:         str   = "cuda"
    tts_model_name: str   = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"

    # Misc
    force_reprocess: bool = False
    sample_rate:    int   = 24000


# ══════════════════════════════════════════════════════════════════════════════
# Cancellation token
# ══════════════════════════════════════════════════════════════════════════════

class CancelToken:
    """Shared flag so the UI 'Cancel' button can stop the pipeline."""
    def __init__(self):
        self._cancelled = threading.Event()

    def cancel(self):
        self._cancelled.set()

    @property
    def is_cancelled(self) -> bool:
        return self._cancelled.is_set()


# ══════════════════════════════════════════════════════════════════════════════
# Main orchestrator
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(
    config:      AudiobookConfig,
    chapters:    list[ExtractedChapter],
    log_queue:   "queue.Queue[str]",
    prog_queue:  "queue.Queue[tuple[int,int]]",   # (current, total)
    cancel:      CancelToken | None = None,
) -> list[str]:
    """
    Runs the full audiobook generation pipeline.

    Returns list of output file paths (one per chapter).
    Sends log strings to log_queue and (ch_num, total) to prog_queue.
    """
    if cancel is None:
        cancel = CancelToken()

    def log(msg: str):
        log_queue.put(msg)
        print(msg)

    def progress(cur: int, total: int):
        prog_queue.put((cur, total))

    os.makedirs(config.output_dir, exist_ok=True)
    total = len(chapters)
    output_files: list[str] = []

    log(f"[Pipeline] Starting — {total} chapter(s) to process")
    log(f"[Pipeline] Output: {config.output_dir}")
    log(f"[Pipeline] Format: {config.output_format}")

    for idx, chapter in enumerate(chapters, 1):
        if cancel.is_cancelled:
            log("[Pipeline] ⛔ Cancelled by user.")
            break

        log(f"\n[Chapter {idx}/{total}] '{chapter.title}'")
        progress(idx - 1, total)

        try:
            chapter_path = _process_chapter(config, chapter, idx, total, log, cancel)
            if chapter_path:
                output_files.append(chapter_path)
                log(f"[Chapter {idx}/{total}] ✅ Done → {os.path.basename(chapter_path)}")
        except Exception as e:
            log(f"[Chapter {idx}/{total}] ❌ Error: {e}")

    progress(total, total)
    log(f"\n[Pipeline] ✅ Complete — {len(output_files)} file(s) generated.")
    return output_files


# ══════════════════════════════════════════════════════════════════════════════
# Chapter processing
# ══════════════════════════════════════════════════════════════════════════════

def _process_chapter(
    config:  AudiobookConfig,
    chapter: ExtractedChapter,
    idx:     int,
    total:   int,
    log:     Callable,
    cancel:  CancelToken,
) -> str | None:
    """Generate audio for one chapter. Returns output file path."""
    from audiobook_factory.audio_processor import tts_consumer
    from audiobook_factory.ffmpeg_utils import get_format_settings

    temp_dir = tempfile.mkdtemp(prefix=f"abm_ch{idx:03d}_")
    try:
        # ── 1. Build TTS job list ─────────────────────────────────────────────
        sentences = chapter.sentences or smart_sentence_splitter(chapter.text, config.max_len)
        tts_jobs  = []
        for sent in sentences:
            for chunk in _chunk(sent, config.max_len):
                tts_jobs.append(chunk)

        if not tts_jobs:
            log(f"  [Ch{idx}] No text to synthesise — skipping.")
            return None

        log(f"  [Ch{idx}] {len(tts_jobs)} TTS chunks...")

        # ── 2. Run TTS using the worker (subprocess-safe) ─────────────────────
        job_q    = _ImmediateQueue()
        result_q = _ImmediateQueue()

        # Build a minimal args-like object for tts_consumer
        class _Args:
            device          = config.device
            tts_model_name  = config.tts_model_name
            temperature     = config.temperature
            top_p           = config.top_p
            voice_file      = config.voice_file

        # Spin up TTS consumer thread (in-process, no multiprocessing overhead)
        consumer_thread = threading.Thread(
            target=tts_consumer, args=(job_q, result_q, _Args()), daemon=True
        )
        consumer_thread.start()

        # Enqueue jobs
        collected: list[dict | None] = [None] * len(tts_jobs)
        for i, text in enumerate(tts_jobs):
            out_path = os.path.join(temp_dir, f"s_{i:04d}.wav")
            job_q.put((i, text, out_path, config.voice_file))

        job_q.put("STOP")
        consumer_thread.join(timeout=3600)

        # Collect results (the consumer writes files and puts (idx, text, path) in result_q)
        while not result_q.empty():
            try:
                i, text, path = result_q.get_nowait()
                if path and os.path.exists(path):
                    collected[i] = {"text": text, "path": path}
            except queue.Empty:
                break

        if cancel.is_cancelled:
            return None

        # ── 3. Build filelist + pauses ─────────────────────────────────────────
        sent_pause_path = os.path.join(temp_dir, "pause_sent.wav")
        sf.write(sent_pause_path, np.zeros(int(config.pause * config.sample_rate)), config.sample_rate)

        filelist_path = os.path.join(temp_dir, "filelist.txt")
        with open(filelist_path, "w", encoding="utf-8") as f:
            for i, item in enumerate(collected):
                if item and os.path.exists(item["path"]):
                    f.write(f"file '{os.path.basename(item['path'])}'\n")
                    if i < len(collected) - 1:
                        f.write(f"file '{os.path.basename(sent_pause_path)}'\n")

        # ── 4. FFmpeg concat ──────────────────────────────────────────────────
        raw_wav = os.path.join(temp_dir, "raw.wav")
        subprocess.run(
            ["ffmpeg", "-y", "-f", "concat", "-safe", "0",
             "-i", os.path.basename(filelist_path), "-c", "copy",
             os.path.basename(raw_wav)],
            check=True, capture_output=True, cwd=temp_dir,
        )

        # ── 5. Loudnorm mastering ─────────────────────────────────────────────
        master_wav = os.path.join(temp_dir, "master.wav")
        subprocess.run(
            ["ffmpeg", "-y", "-i", raw_wav,
             "-af", f"loudnorm=I={config.lufs}:TP={config.true_peak}:LRA=11",
             master_wav],
            check=True, capture_output=True,
        )

        # ── 6. Encode to final format ─────────────────────────────────────────
        safe_title   = re.sub(r'[\\/*?:"<>|]', "", chapter.title)
        out_filename = f"{idx:02d}_{safe_title}.{config.output_format}"
        out_path     = os.path.join(config.output_dir, out_filename)

        audio_settings, _, _ = get_format_settings(config.output_format)

        ffmpeg_cmd = (
            ["ffmpeg", "-y", "-i", master_wav]
            + audio_settings
            + ["-metadata", f"title={chapter.title}",
               "-metadata", f"artist={config.author}",
               "-metadata", f"album={config.book_title}",
               "-metadata", f"track={idx}",
               "-metadata", "genre=Audiobook",
               out_path]
        )
        subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
        return out_path

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _chunk(text: str, max_len: int) -> list[str]:
    """Split a long string at sentence boundaries to stay under max_len."""
    if len(text) <= max_len:
        return [text]
    return smart_sentence_splitter(text, max_len)


class _ImmediateQueue(queue.Queue):
    """A Queue subclass with no max size, usable as a drop-in for multiprocessing.Queue."""
    pass


def preview_tts(text: str, config: AudiobookConfig) -> bytes | None:
    """
    Generate a short TTS preview and return raw WAV bytes.
    Used by the Voice Studio tab.
    """
    from audiobook_factory.audio_processor import tts_consumer

    if not text.strip():
        return None

    with tempfile.TemporaryDirectory() as tmp:
        out_path = os.path.join(tmp, "preview.wav")

        job_q    = _ImmediateQueue()
        result_q = _ImmediateQueue()

        class _Args:
            device         = config.device
            tts_model_name = config.tts_model_name
            temperature    = config.temperature
            top_p          = config.top_p
            voice_file     = config.voice_file

        t = threading.Thread(
            target=tts_consumer, args=(job_q, result_q, _Args()), daemon=True
        )
        t.start()
        job_q.put((0, text.strip(), out_path, config.voice_file))
        job_q.put("STOP")
        t.join(timeout=120)

        if os.path.exists(out_path):
            with open(out_path, "rb") as f:
                return f.read()
    return None
