"""
audiobook_factory/pipeline.py
================================
Thread-safe audiobook generation orchestrator.

New in this version
-------------------
- Audiobookshelf-compatible filenames via filename_sanitizer.make_safe_filename()
- preview_mode      — returns chapter stats table without calling TTS
- export_text       — writes a .txt file per chapter alongside the audio
- worker_count      — ThreadPoolExecutor for parallel chapter processing
- pronunciation_map — regex search-replace applied to text before TTS
- tts_provider_name — selects which provider to use (currently: "qwen")
- TTS logic delegated to tts_providers.get_tts_provider()
"""
from __future__ import annotations

import json
import os
import queue
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import soundfile as sf

# ── project root & temp folder ────────────────────────────────────────────────
# NOTE: _has_rust is checked lazily at call time (not import time).
# This is important for environments like Kaggle/Colab where the Rust
# extension may be compiled *after* this module is first imported.
# Calling _check_rust() on each chapter ensures the freshly-compiled
# .so is found without needing a kernel restart.
def _check_rust() -> bool:
    """Return True if audiobook_rust.master_audio is importable right now."""
    try:
        import importlib
        importlib.invalidate_caches()
        import audiobook_rust
        return hasattr(audiobook_rust, "master_audio")
    except ImportError:
        return False

_ROOT = Path(__file__).resolve().parent.parent
_TEMP_DIR = _ROOT / "temp"
_TEMP_DIR.mkdir(parents=True, exist_ok=True)

from audiobook_factory.text_extractor import ExtractedChapter
from audiobook_factory.text_processing import smart_sentence_splitter
from audiobook_factory.filename_sanitizer import make_safe_filename
from audiobook_factory.utils import (
    load_or_create_progress_file, 
    update_progress_file,
    format_lrc_timestamp
)


# ══════════════════════════════════════════════════════════════════════════════
# Config dataclass
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class AudiobookConfig:
    # ── Book metadata ─────────────────────────────────────────────────────────
    book_title:          str   = "Audiobook"
    author:              str   = "Unknown Author"
    language:            str   = "English"
    cover_image:         str | None = None
    book_path:           str   = ""

    # ── Output ────────────────────────────────────────────────────────────────
    output_dir:          str   = "./output"
    output_format:       str   = "mp3"    # mp3 | flac | wav | m4b

    # ── Voice ─────────────────────────────────────────────────────────────────
    voice_file:          str   = ""       # path to cloning WAV

    # ── TTS ───────────────────────────────────────────────────────────────────
    tts_provider_name:   str   = "qwen"   # "qwen" (Qwen3-TTS)
    temperature:         float = 0.3
    top_p:               float = 0.8
    max_len:             int   = 399      # max chars per TTS chunk

    # ── Pacing ────────────────────────────────────────────────────────────────
    pause:               float = 0.5     # seconds between sentences
    para_pause:          float = 1.2     # seconds between paragraphs

    # ── Audio mastering ───────────────────────────────────────────────────────
    lufs:                int   = -18
    true_peak:           float = -1.5

    # ── Parallelism ───────────────────────────────────────────────────────────
    worker_count:        int   = 1       # chapters/chunks in parallel
    parallel_mode:       str   = "chunks" # "chapters" | "chunks"

    # ── Multi-Model Qwen3 ─────────────────────────────────────────────────────
    device:              str   = "cuda"
    tts_model_name:      str   = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    tts_instruct:        str   = ""       # For VoiceDesign/CustomVoice instructions
    tts_timbre:          str   = ""       # For CustomVoice premium speakers

    # ── Modes ─────────────────────────────────────────────────────────────────
    preview_mode:        bool  = False   # show stats, no TTS
    export_text:         bool  = False   # write .txt per chapter
    export_lrc:          bool  = True    # write .lrc timed lyrics
    export_srt:          bool  = False   # write .srt subtitles
    export_vtt:          bool  = False   # write .webvtt subtitles
    single_file_mode:    bool  = False   # combine all into one big file

    # ── Misc ──────────────────────────────────────────────────────────────────
    force_reprocess:     bool  = False
    sample_rate:         int   = 24000
    torch_compile:       bool  = False

    # ── Pronunciation fixes ───────────────────────────────────────────────────
    # { regex_pattern: replacement }  applied before TTS
    pronunciation_map:   dict  = field(default_factory=dict)

    # ── Resume / selection ────────────────────────────────────────────────────
    # Raw chapter labels chosen in the UI (e.g. "1. Chapter 1 (~500 words)")
    # Stored in progress JSON so the user doesn't have to re-select on resume.
    selected_chapters:   list  = field(default_factory=list)

    # When True (default), chapters marked 'completed' in the progress JSON
    # but whose audio file is missing on disk will be automatically re-generated.
    # When False, such chapters are logged and silently skipped.
    regen_missing:       bool  = True



# ══════════════════════════════════════════════════════════════════════════════
# Cancellation token
# ══════════════════════════════════════════════════════════════════════════════

class CancelToken:
    """Shared flag — the UI Cancel button sets this to stop mid-pipeline."""
    def __init__(self):
        self._cancelled = threading.Event()

    def cancel(self):
        self._cancelled.set()

    @property
    def is_cancelled(self) -> bool:
        return self._cancelled.is_set()


# ══════════════════════════════════════════════════════════════════════════════
# Pronunciation helper
# ══════════════════════════════════════════════════════════════════════════════

def _apply_pronunciation(text: str, pron_map: dict) -> str:
    """Apply all regex search-replace pairs to *text* before TTS."""
    for pattern, replacement in pron_map.items():
        try:
            text = re.sub(pattern, replacement, text)
        except re.error:
            # Treat as literal string if the pattern is invalid.
            text = text.replace(pattern, replacement)
    return text


# ══════════════════════════════════════════════════════════════════════════════
# Preview mode
# ══════════════════════════════════════════════════════════════════════════════

def preview_chapters(
    chapters:   list[ExtractedChapter],
    log_queue:  "queue.Queue[str]",
) -> list[dict]:
    """
    Preview mode — return a list of chapter-info dicts without generating audio.

    Returns
    -------
    List of { "idx", "title", "chars", "words", "sentences" } dicts.
    """
    rows = []
    total_chars = 0

    for idx, ch in enumerate(chapters, 1):
        chars  = len(ch.text)
        words  = len(ch.text.split())
        sents  = len(smart_sentence_splitter(ch.text, 9999))  # count only
        total_chars += chars
        rows.append({
            "idx":       idx,
            "title":     ch.title,
            "chars":     chars,
            "words":     words,
            "sentences": sents,
        })
        log_queue.put(
            f"[Preview] Ch {idx:>3}: {ch.title[:50]:<50} "
            f"| {chars:>7,} chars | {words:>6,} words"
        )

    log_queue.put(f"\n[Preview] Total characters: {total_chars:,}")
    log_queue.put(f"[Preview] Total chapters:   {len(rows)}")
    return rows


# ══════════════════════════════════════════════════════════════════════════════
# Main orchestrator
# ══════════════════════════════════════════════════════════════════════════════

def run_pipeline(
    config:      AudiobookConfig,
    chapters:    list[ExtractedChapter],
    log_queue:   "queue.Queue[str]",
    prog_queue:  "queue.Queue[tuple[int,int]]",
    cancel:      CancelToken | None = None,
) -> list[str]:
    """
    Run the full audiobook generation pipeline.

    Returns list of output file paths (one per chapter).
    In preview_mode returns an empty list (no audio files generated).
    """
    if cancel is None:
        cancel = CancelToken()

    def log(msg: str):
        log_queue.put(msg)
        print(msg)

    def progress(cur: float, total: int):
        prog_queue.put((cur, float(total)))

    os.makedirs(config.output_dir, exist_ok=True)
    total = len(chapters)

    # ── Preview mode ──────────────────────────────────────────────────────────
    if config.preview_mode:
        log(f"[Pipeline] Preview mode — {total} chapter(s)")
        preview_chapters(chapters, log_queue)
        progress(total, total)
        return []

    log(f"[Pipeline] Starting — {total} chapter(s)")
    log(f"[Pipeline] Output  : {config.output_dir}")
    log(f"[Pipeline] Format  : {config.output_format}")
    log(f"[Pipeline] Workers : {config.worker_count}")
    if config.pronunciation_map:
        log(f"[Pipeline] Pronunciation fixes: {len(config.pronunciation_map)}")
    if config.export_text:
        log("[Pipeline] Text export: enabled")

    # ── Progress tracking setup ───────────────────────────────────────────────
    progress_name = "generation_progress.json"
    prog_path_out = os.path.join(config.output_dir, progress_name)
    prog_path_tmp = os.path.join(str(_TEMP_DIR), progress_name)

    if config.force_reprocess:
        log("[Pipeline] 🔄 Force reprocess enabled. Clearing old progress.")
        for p in [prog_path_out, prog_path_tmp]:
            if os.path.exists(p):
                try: os.remove(p)
                except: pass

    # ── Build per-chapter tasks ───────────────────────────────────────────────
    tasks = list(enumerate(chapters, 1))

    # We only initialize with the output path first, then copy to temp
    chapters_data = [{"num": i, "title": ch.title} for i, ch in tasks]
    
    # Convert config dataclass to dict for settings
    from dataclasses import asdict
    settings_dict = {}
    try:
        # Ignore fields that are not JSON serializable or too large
        for k, v in asdict(config).items():
            if k not in ("pronunciation_map",):
                settings_dict[k] = v
    except Exception as e:
        print(f"Error serializing config: {e}")

    progress_data = load_or_create_progress_file(
        prog_path_out,
        chapters_data,
        config.book_title,
        book_path=getattr(config, "book_path", ""),
        voice_file=getattr(config, "voice_file", ""),
        settings=settings_dict
    )

    # Ensure settings, book_path, voice_file are up to date in the progress data
    dirty = False
    if "book_path" not in progress_data or not progress_data["book_path"]:
        progress_data["book_path"] = getattr(config, "book_path", "")
        dirty = True
    if "voice_file" not in progress_data or not progress_data["voice_file"]:
        progress_data["voice_file"] = getattr(config, "voice_file", "")
        dirty = True
    if "settings" not in progress_data or not progress_data["settings"]:
        progress_data["settings"] = settings_dict
        dirty = True
        
    if dirty:
        try:
            with open(prog_path_out, "w", encoding="utf-8") as f:
                json.dump(progress_data, f, indent=4)
        except Exception as e:
            print(f"Error writing updated progress json settings: {e}")
            
    # Sync to temp for user visibility
    try:
        with open(prog_path_tmp, "w", encoding="utf-8") as f:
            json.dump(progress_data, f, indent=4)
    except:
        pass

    # ── Shared TTS Provider Setup ─────────────────────────────────────────────
    from audiobook_factory.tts_providers import get_tts_provider
    provider = None
    if not config.preview_mode:
        provider = get_tts_provider(config.tts_provider_name, config)

    output_files: list[str] = []
    _lock = threading.Lock()
    
    chapter_progress = {i: 0.0 for i in range(1, total + 1)}
    def _update_chapter_prog(idx, frac):
        with _lock:
            chapter_progress[idx] = frac
            sum_frac = sum(chapter_progress.values())
            progress(sum_frac, total)

    def _process(idx_chapter):
        idx, chapter = idx_chapter
        if cancel.is_cancelled:
            return None

        # Check checkpoint
        ch_status = "pending"
        for c in progress_data.get("chapters", []):
            if c["num"] == idx:
                ch_status = c.get("status", "pending")
                break
        
        if ch_status == "completed" and not config.force_reprocess:
            log(f"[Chapter {idx}/{total}] ⏩ Already completed. Skipping.")
            _update_chapter_prog(idx, 1.0)
            # Find the existing file to return its path
            safe_name = make_safe_filename(chapter.title, idx, config.output_dir, f".{config.output_format}")
            existing_path = os.path.join(config.output_dir, safe_name)
            if os.path.exists(existing_path):
                with _lock:
                    output_files.append(existing_path)
                return existing_path
            # File is missing — check user's preference
            if not getattr(config, "regen_missing", True):
                log(f"  [Ch{idx}] ⚠ Warning: Marked 'completed' but file not found. Skipping (regen_missing=False).")
                return None
            log(f"  [Ch{idx}] ⚠ Warning: Marked 'completed' but file not found. Re-generating.")

        log(f"\n[Chapter {idx}/{total}] '{chapter.title}'")
        try:
            path = _process_chapter(
                config, chapter, idx, total, log, cancel, provider, 
                prog_cb=lambda f: _update_chapter_prog(idx, f)
            )
            if path:
                with _lock:
                    output_files.append(path)
                log(f"[Chapter {idx}/{total}] ✅ → {os.path.basename(path)}")
                
                # Update checkpoint files
                for p in [prog_path_out, prog_path_tmp]:
                    try:
                        update_progress_file(p, idx, "completed")
                    except:
                        pass
            return path
        except Exception as e:
            import traceback
            traceback.print_exc()
            log(f"[Chapter {idx}/{total}] ❌ Error: {e}")
            return None
        finally:
            _update_chapter_prog(idx, 1.0)

    try:
        if getattr(config, "parallel_mode", "chunks") == "chapters" and config.worker_count > 1:
            with ThreadPoolExecutor(max_workers=config.worker_count) as pool:
                futures = {pool.submit(_process, t): t for t in tasks}
                for fut in as_completed(futures):
                    if cancel.is_cancelled:
                        pool.shutdown(wait=False, cancel_futures=True)
                        break
                    fut.result()
        else:
            for t in tasks:
                if cancel.is_cancelled:
                    log("[Pipeline] ⛔ Cancelled.")
                    break
                _process(t)

        progress(total, total)

        if cancel.is_cancelled:
            log(f"\n[Pipeline] ⛔ Cancelled — {len(output_files)} file(s) saved.")
            output_files.sort()
            return output_files

        # ── Single File Mode (Combine all chapters) ───────────────────────────────
        if config.single_file_mode and len(output_files) > 1:
            log("\n[Pipeline] 📦 Combining chapters into a single file...")
            output_files.sort()
            
            # Use simple concat protocol for same-format files
            list_txt = os.path.join(config.output_dir, "concat_list.txt")
            full_name = make_safe_filename(config.book_title, 0, config.output_dir, f".{config.output_format}")
            full_path = os.path.join(config.output_dir, f"Combined_{full_name}")

            try:
                with open(list_txt, "w", encoding="utf-8") as f:
                    for p in output_files:
                        p_safe = os.path.abspath(p).replace('\\', '/')
                        f.write(f"file '{p_safe}'\n")
                
                subprocess.run(
                    ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_txt, "-c", "copy", full_path],
                    check=True, capture_output=True
                )
                log(f"[Pipeline] 📦 Combined file created: {os.path.basename(full_path)}")
                
                # Clean up chapters and list
                for p in output_files:
                    try: os.remove(p)
                    except: pass
                os.remove(list_txt)
                
                output_files = [full_path]
            except Exception as e:
                log(f"[Pipeline] ❌ Failed to combine: {e}")

        log(f"\n[Pipeline] ✅ Complete — {len(output_files)} file(s) generated.")
        output_files.sort()
        return output_files

    finally:
        if provider is not None:
            try:
                provider.cleanup()
            except Exception as e:
                log(f"[Pipeline] Cleanup error: {e}")



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
    provider: "BaseTTSProvider" = None,
    prog_cb: Callable[[float], None] = None,
) -> str | None:
    """Generate audio for one chapter. Returns output file path."""
    from audiobook_factory.ffmpeg_utils import get_format_settings
    from audiobook_factory.tts_providers import get_tts_provider

    temp_dir = tempfile.mkdtemp(prefix=f"abm_ch{idx:03d}_", dir=str(_TEMP_DIR))
    try:
        # ── Apply pronunciation fixes ─────────────────────────────────────────
        text = chapter.text
        if config.pronunciation_map:
            text = _apply_pronunciation(text, config.pronunciation_map)

        # ── Export text if requested ──────────────────────────────────────────
        if config.export_text:
            txt_name = make_safe_filename(chapter.title, idx, config.output_dir, ".txt")
            txt_path = os.path.join(config.output_dir, txt_name)
            try:
                with open(txt_path, "w", encoding="utf-8") as fh:
                    fh.write(f"{chapter.title}\n{'─' * 60}\n\n{text}")
                log(f"  [Ch{idx}] Text exported → {txt_name}")
            except OSError as e:
                log(f"  [Ch{idx}] Text export failed: {e}")

        # ── Build sentence/chunk list ─────────────────────────────────────────
        sentences = chapter.sentences or smart_sentence_splitter(text, config.max_len)
        tts_jobs  = []
        for sent in sentences:
            for chunk in _chunk(sent, config.max_len):
                tts_jobs.append(chunk)

        if not tts_jobs:
            log(f"  [Ch{idx}] No text to synthesise — skipping.")
            return None

        log(f"  [Ch{idx}] {len(tts_jobs)} TTS chunks…")

        # ── Synthesise via provider ───────────────────────────────────────────
        if provider is None:
            # Fallback for manual calls bypassing run_pipeline
            provider = get_tts_provider(config.tts_provider_name, config)

        chunk_paths: list[str | None] = [None] * len(tts_jobs)
        chunk_durations: list[float] = [0.0] * len(tts_jobs)

        use_parallel_chunks = (getattr(config, "parallel_mode", "chunks") == "chunks" and config.worker_count > 1)

        if use_parallel_chunks:
            # Batch size for GPU inference — can be larger than worker_count
            # since batched TTS runs in a single model forward pass
            batch_size = max(config.worker_count, 4)
            for batch_start in range(0, len(tts_jobs), batch_size):
                if cancel.is_cancelled:
                    break
                
                batch_end = min(batch_start + batch_size, len(tts_jobs))
                batch_texts = tts_jobs[batch_start:batch_end]
                batch_voice_refs = [config.voice_file] * len(batch_texts)
                batch_out_paths = [os.path.join(temp_dir, f"s_{i:04d}.wav") for i in range(batch_start, batch_end)]
                
                try:
                    durations = provider.synthesize_batch(batch_texts, batch_voice_refs, batch_out_paths)
                    for idx_chunk, out_wav in enumerate(batch_out_paths):
                        i = batch_start + idx_chunk
                        # Only record path if the file was actually written
                        if os.path.exists(out_wav) and os.path.getsize(out_wav) > 0:
                            chunk_paths[i] = out_wav
                            if config.export_lrc:
                                chunk_durations[i] = durations[idx_chunk]
                except Exception as e:
                    import traceback
                    error_trace = traceback.format_exc()
                    log(f"  [Ch{idx}] Batch starting at chunk {batch_start} failed ({type(e).__name__}): {e}")
                    log(f"  [Ch{idx}] Retrying batch chunks one-by-one...")
                    err_log_path = os.path.join(config.output_dir, f"error_ch{idx:03d}_batch_{batch_start}.txt")
                    with open(err_log_path, "w", encoding="utf-8") as err_f:
                        err_f.write(error_trace)

                    # ── One-by-one fallback for failed batch ──────────────────
                    recovered = 0
                    for sub_i, (chunk_text, out_wav) in enumerate(zip(batch_texts, batch_out_paths)):
                        global_i = batch_start + sub_i
                        try:
                            provider.synthesize(chunk_text, config.voice_file, out_wav)
                            chunk_paths[global_i] = out_wav
                            if config.export_lrc:
                                chunk_durations[global_i] = _get_wav_duration(out_wav)
                            recovered += 1
                        except Exception as sub_e:
                            log(f"  [Ch{idx}] ⚠ Chunk {global_i} failed even in single mode: {sub_e}")
                    log(f"  [Ch{idx}] One-by-one recovery: {recovered}/{len(batch_texts)} chunks saved.")
                
                if prog_cb:
                    prog_cb(batch_end / len(tts_jobs))
        else:
            for i, chunk_text in enumerate(tts_jobs):
                if cancel.is_cancelled:
                    return None
                out_wav = os.path.join(temp_dir, f"s_{i:04d}.wav")
                # Retry once on transient failures (e.g. brief GPU hiccup)
                for attempt in range(2):
                    try:
                        provider.synthesize(chunk_text, config.voice_file, out_wav)
                        if os.path.exists(out_wav) and os.path.getsize(out_wav) > 0:
                            chunk_paths[i] = out_wav
                            if config.export_lrc:
                                chunk_durations[i] = _get_wav_duration(out_wav)
                        break  # success
                    except Exception as e:
                        if attempt == 0:
                            import time as _time
                            log(f"  [Ch{idx}] chunk {i} failed ({e}), retrying...")
                            _time.sleep(1)
                        else:
                            import traceback
                            error_trace = traceback.format_exc()
                            log(f"  [Ch{idx}] chunk {i} failed after retry: {e}")
                            err_log_path = os.path.join(config.output_dir, f"error_ch{idx:03d}_{i}.txt")
                            with open(err_log_path, "w", encoding="utf-8") as err_f:
                                err_f.write(error_trace)

                if prog_cb:
                    prog_cb((i + 1) / len(tts_jobs))

        if cancel.is_cancelled:
            return None

        # ── Generate LRC timed lyrics ─────────────────────────────────────────
        if config.export_lrc:
            lrc_name = make_safe_filename(chapter.title, idx, config.output_dir, ".lrc")
            lrc_path = os.path.join(config.output_dir, lrc_name)
            try:
                curr_time = 0.0
                pause_len = config.pause
                with open(lrc_path, "w", encoding="utf-8") as fh:
                    for i, (text_chunk, dur) in enumerate(zip(tts_jobs, chunk_durations)):
                        m, s = divmod(curr_time, 60)
                        fh.write(f"[{int(m):02d}:{s:05.2f}]{text_chunk}\n")
                        curr_time += dur + pause_len
                log(f"  [Ch{idx}] LRC exported → {lrc_name}")
            except Exception as e:
                log(f"  [Ch{idx}] LRC export failed: {e}")

        # ── Generate SRT timed subtitles ──────────────────────────────────────
        if config.export_srt:
            srt_name = make_safe_filename(chapter.title, idx, config.output_dir, ".srt")
            srt_path = os.path.join(config.output_dir, srt_name)
            try:
                from audiobook_factory.utils import seconds_to_srt_time
                curr_time = 0.0
                pause_len = config.pause
                with open(srt_path, "w", encoding="utf-8") as fh:
                    for i, (text_chunk, dur) in enumerate(zip(tts_jobs, chunk_durations), 1):
                        start = seconds_to_srt_time(curr_time)
                        end = seconds_to_srt_time(curr_time + dur)
                        fh.write(f"{i}\n{start} --> {end}\n{text_chunk}\n\n")
                        curr_time += dur + pause_len
                log(f"  [Ch{idx}] SRT exported → {srt_name}")
            except Exception as e:
                log(f"  [Ch{idx}] SRT export failed: {e}")

        # ── Generate WebVTT timed subtitles ───────────────────────────────────
        if config.export_vtt:
            vtt_name = make_safe_filename(chapter.title, idx, config.output_dir, ".vtt")
            vtt_path = os.path.join(config.output_dir, vtt_name)
            try:
                from audiobook_factory.utils import seconds_to_vtt_time
                curr_time = 0.0
                pause_len = config.pause
                with open(vtt_path, "w", encoding="utf-8") as fh:
                    fh.write("WEBVTT\n\n")
                    for i, (text_chunk, dur) in enumerate(zip(tts_jobs, chunk_durations), 1):
                        start = seconds_to_vtt_time(curr_time)
                        end = seconds_to_vtt_time(curr_time + dur)
                        fh.write(f"{i}\n{start} --> {end}\n{text_chunk}\n\n")
                        curr_time += dur + pause_len
                log(f"  [Ch{idx}] WebVTT exported → {vtt_name}")
            except Exception as e:
                log(f"  [Ch{idx}] WebVTT export failed: {e}")

        # ── In-memory audio mastering (Rust first, Python fallback) ───────────
        if not any(chunk_paths):
            log(f"  [Ch{idx}] ❌ No audio chunks generated successfully. Skipping.")
            return None

        safe_name  = make_safe_filename(chapter.title, idx, config.output_dir,
                                        f".{config.output_format}")
        out_path   = os.path.join(config.output_dir, safe_name)

        if _check_rust():
            import audiobook_rust as _audiobook_rust  # fresh local import
            valid_paths = [p for p in chunk_paths if p and os.path.exists(p)]
            if not valid_paths:
                log(f"  [Ch{idx}] ❌ No audio chunks found. Skipping.")
                return None

            try:
                bitrate_kbps = getattr(config, "bitrate_kbps", 64)
                has_cover = bool(config.cover_image and os.path.exists(config.cover_image))
                use_pure_rust = (config.output_format in ("mp3", "wav")) and not has_cover

                # If pure rust, write directly to final out_path.
                # If cover image is specified or other container/codec (e.g. flac, m4b) is used,
                # master to a temporary WAV first and use FFmpeg to do packaging and copy streams.
                master_target = out_path if use_pure_rust else os.path.join(temp_dir, "mastered.wav")

                _audiobook_rust.master_audio(
                    valid_paths,
                    master_target,
                    float(config.pause),
                    int(config.sample_rate),
                    float(config.lufs),
                    float(config.true_peak),
                    int(bitrate_kbps)
                )

                log(f"  [Ch{idx}] ⚡ Mastered {len(valid_paths)} segments via Rust to {os.path.basename(master_target)}")

                if not use_pure_rust:
                    audio_settings, _, _ = get_format_settings(config.output_format)[:3]
                    ffmpeg_cmd = [
                        "ffmpeg", "-y",
                        "-i", master_target,
                    ]
                    if has_cover:
                        ffmpeg_cmd += ["-i", config.cover_image]
                    
                    ffmpeg_cmd += audio_settings

                    if has_cover:
                        ffmpeg_cmd += ["-map", "0:a", "-map", "1:v", "-c:v", "copy",
                                       "-disposition:v", "attached_pic", "-id3v2_version", "3"]

                    ffmpeg_cmd += [
                        "-metadata", f"title={chapter.title}",
                        "-metadata", f"artist={config.author}",
                        "-metadata", f"album={config.book_title}",
                        "-metadata", f"track={idx}",
                        "-metadata", "genre=Audiobook",
                        out_path,
                    ]

                    subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
                
                return out_path

            except Exception as rust_err:
                log(f"  [Ch{idx}] ⚠ Rust mastering failed ({rust_err}). Falling back to Python/FFmpeg.")

        # ── Python fallback: In-memory WAV concat ─────────────────────────────
        # Build the silence pause array once
        pause_samples = np.zeros(int(config.pause * config.sample_rate), dtype=np.float32)

        # Concatenate all chunk WAVs + pauses in-memory (no disk I/O)
        # Determine the last valid chunk index so we don't add a trailing silence
        valid_indices = [i for i, p in enumerate(chunk_paths) if p and os.path.exists(p)]
        last_valid_idx = valid_indices[-1] if valid_indices else -1
        audio_segments: list[np.ndarray] = []
        for i, p in enumerate(chunk_paths):
            if p and os.path.exists(p):
                try:
                    chunk_audio, _ = sf.read(p, dtype="float32")
                    audio_segments.append(chunk_audio)
                    if i != last_valid_idx:  # no trailing silence after last real chunk
                        audio_segments.append(pause_samples)
                except Exception:
                    pass  # skip corrupt chunks

        if not audio_segments:
            log(f"  [Ch{idx}] ❌ No valid audio segments. Skipping.")
            return None

        raw_audio = np.concatenate(audio_segments)
        log(f"  [Ch{idx}] Concatenated {len(audio_segments)} segments "
            f"({len(raw_audio)/config.sample_rate:.1f}s) in-memory (python fallback)")

        # ── Single FFmpeg call: loudnorm + encode (piped via stdin) ───────────
        audio_settings, _, _ = get_format_settings(config.output_format)[:3]

        # Build FFmpeg command: read raw PCM from stdin → loudnorm → encode
        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-f", "f32le",                       # 32-bit float PCM input
            "-ar", str(config.sample_rate),       # sample rate
            "-ac", "1",                           # mono
            "-i", "pipe:0",                       # read from stdin
        ]

        # Add cover image if exists
        has_cover = False
        if config.cover_image and os.path.exists(config.cover_image):
            ffmpeg_cmd += ["-i", config.cover_image]
            has_cover = True

        # Loudnorm filter (was a separate FFmpeg call before)
        ffmpeg_cmd += ["-af", f"loudnorm=I={config.lufs}:TP={config.true_peak}:LRA=11"]

        ffmpeg_cmd += audio_settings

        if has_cover:
            ffmpeg_cmd += ["-map", "0:a", "-map", "1:v", "-c:v", "copy",
                           "-disposition:v", "attached_pic", "-id3v2_version", "3"]

        ffmpeg_cmd += [
            "-metadata", f"title={chapter.title}",
            "-metadata", f"artist={config.author}",
            "-metadata", f"album={config.book_title}",
            "-metadata", f"track={idx}",
            "-metadata", "genre=Audiobook",
            out_path,
        ]

        # Pipe raw PCM bytes to FFmpeg stdin (no intermediate files)
        proc = subprocess.run(
            ffmpeg_cmd,
            input=raw_audio.tobytes(),
            check=True,
            capture_output=True,
        )
        return out_path


    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _get_wav_duration(path: str) -> float:
    """Return the duration of a WAV file in seconds."""
    if not os.path.exists(path):
        return 0.0
    with sf.SoundFile(path) as f:
        return f.frames / f.samplerate

def _chunk(text: str, max_len: int) -> list[str]:
    """Split a long string at sentence boundaries to stay under max_len."""
    if len(text) <= max_len:
        return [text]
    return smart_sentence_splitter(text, max_len)


class _ImmediateQueue(queue.Queue):
    """Queue subclass kept for backward compat with old callers."""
    pass


def preview_tts(text: str, config: AudiobookConfig) -> bytes | None:
    """
    Generate a short TTS preview and return raw WAV bytes.
    Used by the Voice Studio tab.
    """
    from audiobook_factory.tts_providers import get_tts_provider

    if not text.strip():
        return None

    with tempfile.TemporaryDirectory(dir=str(_TEMP_DIR)) as tmp:
        out_path = os.path.join(tmp, "preview.wav")
        try:
            provider = get_tts_provider(config.tts_provider_name, config)
            provider.synthesize(text.strip(), config.voice_file, out_path)
            if os.path.exists(out_path):
                with open(out_path, "rb") as f:
                    return f.read()
        except Exception as e:
            print(f"[preview_tts] Error: {e}")
    return None
