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

import atexit
import concurrent.futures
import json
import logging
import os
import queue
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed, CancelledError
from dataclasses import dataclass, field, fields, MISSING
from typing import Callable, Any

logger = logging.getLogger(__name__)

_VALID_QUANTIZATION_MODES: frozenset[str] = frozenset({"none", "int8"})

_subtitle_executor: concurrent.futures.ThreadPoolExecutor = (
    concurrent.futures.ThreadPoolExecutor(
        max_workers=2,
        thread_name_prefix="audiobookmaker_subtitles",
    )
)
atexit.register(_subtitle_executor.shutdown, wait=True)

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
from audiobook_factory.progress_io import (
    read_progress_file,
    write_progress_file,
    update_chapter_status,
    update_chapter_chunk,
)
from audiobook_factory.utils import format_lrc_timestamp


_MAX_PARALLEL_CHAPTERS: int = 0
# 0 = auto (matches GPU count). >0 = override. Set at module level.

def _get_chapter_parallelism(pool: Any) -> int:
    """Returns the number of chapters to process simultaneously.

    Equals pool.device_count when _MAX_PARALLEL_CHAPTERS == 0.
    Capped at pool.device_count to prevent VRAM contention.

    Returns:
        Number of chapters to process in parallel (always >= 1).
    """
    if pool is None or not hasattr(pool, "device_count") or pool.device_count <= 0:
        return 1
    if _MAX_PARALLEL_CHAPTERS > 0:
        return max(1, min(_MAX_PARALLEL_CHAPTERS, pool.device_count))
    return max(1, pool.device_count)


_CONFIG_SCHEMA_VERSION: int = 5
# Increment this integer whenever AudiobookConfig fields are added,
# removed, or renamed. Used to detect stale generation_progress.json
# files from older versions.


@dataclass
class AudiobookConfig:
    # ── Config version ────────────────────────────────────────────────────────
    config_version:      int   = _CONFIG_SCHEMA_VERSION

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
    gpu_count:           int   = 0       # 0 = auto-detect at runtime

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
    force_reprocess:          bool  = False  # When True, forces re-extraction & re-synthesis of all chunks from scratch
    resume_incomplete_chunks: bool  = True   # When True, resumes mid-chapter from last completed chunk using disk cache
    regen_missing:            bool  = True   # When True, regenerates missing/failed chapter audio
    sample_rate:              int   = 24000
    torch_compile:            bool  = False
    quantization:             str   = "none"   # "none" | "int8"
    selected_chapters:        list  = field(default_factory=list) # Selected chapter titles/labels

    # ── Pronunciation fixes ───────────────────────────────────────────────────
    # { regex_pattern: replacement }  applied before TTS
    pronunciation_map:   dict  = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> "AudiobookConfig":
        """Constructs AudiobookConfig from a dict, tolerating unknown and
        missing keys.

        Unknown keys are silently dropped. Missing keys use the field's
        default value. A version mismatch logs a warning but does not raise.

        Args:
            data: Dict from generation_progress.json settings section.

        Returns:
            Populated AudiobookConfig instance.
        """
        known_fields = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in known_fields}

        # Version check
        incoming_version = data.get("config_version", 0)
        if incoming_version < _CONFIG_SCHEMA_VERSION:
            logger.warning(
                "generation_progress.json was created with config schema "
                "version %d, current version is %d. Some settings may use "
                "defaults. Re-export config JSON to update.",
                incoming_version,
                _CONFIG_SCHEMA_VERSION,
            )

        return cls(**filtered)

    @classmethod
    def field_summary(cls) -> str:
        """Returns a human-readable summary of all config fields and defaults.

        Used in --dry-run output and error messages to show users what
        settings are available and what their current defaults are.

        Returns:
            Multi-line string, one field per line: "field_name: default_value"
        """
        lines = []
        for f in fields(cls):
            if f.name.startswith("_"):
                continue
            default = f.default if f.default is not MISSING else (
                f.default_factory() if f.default_factory is not MISSING
                else "<required>"
            )
            lines.append(f"  {f.name}: {default!r}")
        return "\n".join(lines)


def _validate_config(config: AudiobookConfig) -> None:
    """Validate AudiobookConfig options before running the pipeline.

    Raises:
        ValueError: If config.quantization is not one of _VALID_QUANTIZATION_MODES.
    """
    if config.quantization not in _VALID_QUANTIZATION_MODES:
        raise ValueError(
            f"Invalid quantization mode '{config.quantization}'. "
            f"Supported options: {sorted(_VALID_QUANTIZATION_MODES)}"
        )

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
    _validate_config(config)

    from audiobook_factory.preflight import run_preflight_checks, PreflightError

    recommended_dtype = "float16"
    try:
        preflight = run_preflight_checks(
            voice_ref=config.voice_file if config.voice_file else None,
            check_voice_ref=bool(config.voice_file),
        )
        recommended_dtype = preflight.recommended_dtype
    except PreflightError as exc:
        for error in exc.result.errors:
            logger.error("[Pipeline] Pre-flight failed: %s", error)
        raise

    if cancel is None:
        cancel = CancelToken()

    def log(msg: str):
        log_queue.put(msg)
        logger.info(msg)

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
                try:
                    os.remove(p)
                except OSError as exc:
                    logger.debug("Could not remove %s: %s", p, exc)

    # ── Build per-chapter tasks ───────────────────────────────────────────────
    tasks = list(enumerate(chapters, 1))

    # We only initialize with the output path first, then copy to temp
    chapters_data = [
        {"num": i, "title": ch.title, "text": ch.text, "sentences": ch.sentences}
        for i, ch in tasks
    ]
    
    # Convert config dataclass to dict for settings
    from dataclasses import asdict
    settings_dict = {}
    try:
        for k, v in asdict(config).items():
            settings_dict[k] = v
    except Exception as e:
        logger.warning("Error serializing config: %s", e)

    try:
        progress_data = read_progress_file(prog_path_out)
        for c in progress_data.get("chapters", []):
            if "completed_chunks" not in c:
                c["completed_chunks"] = []
    except (FileNotFoundError, ValueError):
        progress_data = {
            "book_title": config.book_title,
            "book_path": getattr(config, "book_path", ""),
            "voice_file": getattr(config, "voice_file", ""),
            "settings": settings_dict,
            "chapters": [
                {
                    "num": c["num"],
                    "title": c["title"],
                    "status": "pending",
                    "completed_chunks": [],
                    "text": c.get("text", ""),
                    "sentences": c.get("sentences", []),
                }
                for c in chapters_data
            ],
        }
        write_progress_file(prog_path_out, progress_data)

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
            write_progress_file(prog_path_out, progress_data)
        except Exception as e:
            logger.warning("Error writing updated progress json settings: %s", e)
            
    # Sync to temp for user visibility
    try:
        write_progress_file(prog_path_tmp, progress_data)
    except OSError as exc:
        logger.debug("Could not sync progress to temp: %s", exc)

    # ── Shared TTS Provider / GPU Pool Setup ──────────────────────────────────
    from audiobook_factory.gpu_pool import GPUPoolManager
    from audiobook_factory.tts_providers import get_tts_provider
    pool = None
    provider = None
    if not config.preview_mode:
        pool = GPUPoolManager.instance().get_pool(
            provider_name=config.tts_provider_name,
            provider_factory=lambda dev: get_tts_provider(
                config.tts_provider_name, config, device=dev, dtype_override=recommended_dtype
            ),
            min_vram_gb=5.0,
            gpu_count_override=config.gpu_count,
        )

    output_files: list[str] = []
    _lock = threading.Lock()
    
    subtitle_futures: list[tuple[int, concurrent.futures.Future]] = []
    subtitle_futures_lock = threading.Lock()

    chapter_progress = {i: 0.0 for i in range(1, total + 1)}
    def _update_chapter_prog(idx, frac):
        with _lock:
            chapter_progress[idx] = frac
            sum_frac = sum(chapter_progress.values())
            progress(sum_frac, total)

    def _process(idx_chapter, pinned_device: str | None = None):
        idx, chapter = idx_chapter
        if cancel.is_cancelled:
            return None

        # Check checkpoint
        from audiobook_factory.utils import normalize_chapter_title_for_matching
        ch_status = "pending"
        completed_chunks: list[int] = []
        ch_title_norm = chapter.title.strip().lower()
        ch_clean_norm = re.sub(r'\(~[\d,]+\s*words\)', '', chapter.title).strip().lower()
        ch_num_extracted, ch_core = normalize_chapter_title_for_matching(chapter.title)

        found_match = False
        # Phase 1: High priority Title Matching
        for c in progress_data.get("chapters", []):
            c_title_norm = c.get("title", "").strip().lower()
            c_clean_norm = re.sub(r'\(~[\d,]+\s*words\)', '', c.get("title", "")).strip().lower()
            c_num_extracted, c_core = normalize_chapter_title_for_matching(c.get("title", ""))

            if (
                (c_title_norm and c_title_norm == ch_title_norm)
                or (c_clean_norm and c_clean_norm == ch_clean_norm)
                or (c_core and ch_core and c_core == ch_core)
                or (c_num_extracted is not None and ch_num_extracted is not None and c_num_extracted == ch_num_extracted and c_core == ch_core)
            ):
                ch_status = c.get("status", "pending")
                completed_chunks = list(c.get("completed_chunks", []))
                found_match = True
                break

        # Phase 2: Fallback Index Matching if no title match was found
        if not found_match:
            for c in progress_data.get("chapters", []):
                c_num_extracted, _ = normalize_chapter_title_for_matching(c.get("title", ""))
                if (
                    c.get("num") == idx
                    or (hasattr(chapter, "num") and str(c.get("num")) == str(chapter.num))
                    or (c_num_extracted is not None and ch_num_extracted is not None and c_num_extracted == ch_num_extracted)
                ):
                    ch_status = c.get("status", "pending")
                    completed_chunks = list(c.get("completed_chunks", []))
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
                config, chapter, idx, total, log, cancel, provider, pool=pool,
                prog_cb=lambda f: _update_chapter_prog(idx, f),
                pinned_device=pinned_device,
                subtitle_futures=subtitle_futures,
                subtitle_futures_lock=subtitle_futures_lock,
                completed_chunks=completed_chunks,
            )
            if path:
                with _lock:
                    output_files.append(path)
                log(f"[Chapter {idx}/{total}] ✅ → {os.path.basename(path)}")
                
                # Update checkpoint files
                for p in [prog_path_out, prog_path_tmp]:
                    try:
                        update_chapter_status(p, idx, "completed", reset_chunks=True)
                    except Exception as exc:
                        logger.debug("Could not update progress file: %s", exc)
            return path
        except Exception as e:
            import traceback
            traceback.print_exc()
            log(f"[Chapter {idx}/{total}] ❌ Error: {e}")
            return None
        finally:
            _update_chapter_prog(idx, 1.0)

    try:
        max_parallel = _get_chapter_parallelism(pool)
        if max_parallel > 1:
            log(f"[Pipeline] 🚀 Inter-chapter parallelism active: processing up to {max_parallel} chapters simultaneously...")
            devices = pool.devices if pool else []
            with ThreadPoolExecutor(max_workers=max_parallel) as executor:
                futures = {}
                for i, t in enumerate(tasks):
                    if cancel.is_cancelled:
                        break
                    pinned = devices[i % len(devices)] if devices else None
                    fut = executor.submit(_process, t, pinned)
                    futures[fut] = t

                for future in as_completed(futures):
                    t = futures[future]
                    try:
                        future.result()
                    except CancelledError:
                        cancel.cancel()
                        break
                    except Exception as exc:
                        log(f"[Pipeline] Chapter execution error: {exc}")
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
                    try:
                        os.remove(p)
                    except OSError as exc:
                        logger.debug("Could not remove chunk file %s: %s", p, exc)
                os.remove(list_txt)
                
                output_files = [full_path]
            except Exception as e:
                log(f"[Pipeline] ❌ Failed to combine: {e}")

        log(f"\n[Pipeline] ✅ Complete — {len(output_files)} file(s) generated.")
        output_files.sort()
        return output_files

    finally:
        _await_subtitle_futures(subtitle_futures, cancel)
        if provider is not None:
            try:
                provider.cleanup()
            except Exception as e:
                logger.warning("[Pipeline] Cleanup error: %s", e)



def _generate_subtitles(
    config: AudiobookConfig,
    chapter: ExtractedChapter,
    idx: int,
    tts_jobs: list[str],
    chunk_durations: list[float],
    log: Callable[[str], None],
) -> None:
    """Generate LRC, SRT, and VTT subtitle files for one chapter.

    Called asynchronously via _subtitle_executor. All three formats are
    written in a single call. Failures are caught per-format and logged as
    warnings — subtitle files are non-critical output.

    Args:
        config: AudiobookConfig controlling which formats to export.
        chapter: ExtractedChapter providing the chapter title.
        idx: Chapter index used in log messages and filename generation.
        tts_jobs: List of text chunks in chapter order.
        chunk_durations: Duration in seconds for each chunk in tts_jobs.
        log: Callable for progress reporting.
    """
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


def _await_subtitle_futures(
    futures: list[tuple[int, concurrent.futures.Future]],
    cancel_token: CancelToken | None = None,
) -> None:
    """Waits for all subtitle generation tasks to complete.

    Cancels pending futures if cancel_token.is_cancelled. Logs warnings
    for timeouts and errors — subtitle files are non-critical output.

    Args:
        futures: List of (chapter_idx, Future) pairs to await.
        cancel_token: Optional token checked before awaiting each future.
    """
    for chapter_num, future in futures:
        if cancel_token is not None and cancel_token.is_cancelled:
            future.cancel()
            continue
        try:
            future.result(timeout=30.0)
        except concurrent.futures.CancelledError:
            pass
        except concurrent.futures.TimeoutError:
            logger.warning(
                "Subtitle generation timed out for chapter %d.", chapter_num
            )
        except Exception as exc:
            logger.warning(
                "Subtitle generation failed for chapter %d: %s", chapter_num, exc
            )


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
    pool: Any = None,
    prog_cb: Callable[[float], None] = None,
    pinned_device: str | None = None,
    subtitle_futures: list[tuple[int, concurrent.futures.Future]] | None = None,
    subtitle_futures_lock: threading.Lock | None = None,
    completed_chunks: list[int] | None = None,
) -> str | None:
    """Generate audio for one chapter. Returns output file path."""
    from audiobook_factory.ffmpeg_utils import get_format_settings
    from audiobook_factory.tts_providers import get_tts_provider

    temp_dir = os.path.join(config.output_dir, ".temp_chunks", f"abm_ch{idx:03d}")
    os.makedirs(temp_dir, exist_ok=True)

    if config.force_reprocess or not getattr(config, "resume_incomplete_chunks", True):
        completed_chunks = []
        if os.path.exists(temp_dir):
            try:
                for f_name in os.listdir(temp_dir):
                    if f_name.startswith(f"chunk_ch_{idx}_") and f_name.endswith(".wav"):
                        os.remove(os.path.join(temp_dir, f_name))
            except OSError as exc:
                logger.warning("Could not clear chunk files in %s: %s", temp_dir, exc)
    elif completed_chunks:
        from audiobook_factory.chapter_pipeline import _validate_chunk_file
        missing_files = [
            chunk_idx for chunk_idx in completed_chunks
            if not _validate_chunk_file(
                os.path.join(temp_dir, f"chunk_ch_{idx}_{chunk_idx}.wav")
            )
        ]
        if missing_files:
            stale_count = len(missing_files)
            total_claimed = len(completed_chunks)
            log(
                f"  [Ch{idx}] Stale checkpoint: {stale_count}/{total_claimed} "
                f"claimed-complete chunks have no WAV file on disk. These will be re-synthesized."
            )

    prog_path_out = os.path.join(config.output_dir, "generation_progress.json")
    def _chunk_cb(c_idx: int) -> None:
        update_chapter_chunk(prog_path_out, idx, c_idx)

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

        # ── Synthesis via 3-Stage Overlapped Chapter Pipeline ─────────────────
        if pool is not None:
            from audiobook_factory.chapter_pipeline import run_chapter_pipeline

            voice_bytes = b""
            if config.voice_file and os.path.exists(config.voice_file):
                try:
                    with open(config.voice_file, "rb") as vf:
                        voice_bytes = vf.read()
                except Exception as exc:
                    log(f"  [Ch{idx}] Warning: Could not read voice_file bytes: {exc}")

            logger.info(
                "[process_chapter] voice_ref bytes: %d bytes from %s",
                len(voice_bytes),
                config.voice_file,
            )

            chapter_wav_path = os.path.join(temp_dir, "chapter_mastered.wav")
            chunk_durations = run_chapter_pipeline(
                sentences=sentences,
                voice_ref=voice_bytes,
                out_wav_path=chapter_wav_path,
                out_dir=temp_dir,
                chapter_index=idx,
                config=config,
                pool=pool,
                cancel_token=cancel,
                log_callback=log,
                progress_callback=prog_cb,
                pinned_device=pinned_device,
                completed_chunks=completed_chunks,
                chunk_completed_cb=_chunk_cb,
            )
            chunk_paths = [chapter_wav_path] if os.path.exists(chapter_wav_path) else []
        else:
            def _synth_single(t_text: str, v_ref: str, o_path: str) -> None:
                if provider is not None:
                    provider.synthesize(t_text, v_ref, o_path)
                else:
                    p = get_tts_provider(config.tts_provider_name, config)
                    p.synthesize(t_text, v_ref, o_path)

            chunk_paths: list[str | None] = [None] * len(tts_jobs)
            chunk_durations: list[float] = [0.0] * len(tts_jobs)

            for i, chunk_text in enumerate(tts_jobs):
                if cancel.is_cancelled:
                    return None
                out_wav = os.path.join(temp_dir, f"s_{i:04d}.wav")
                for attempt in range(2):
                    try:
                        _synth_single(chunk_text, config.voice_file, out_wav)
                        if os.path.exists(out_wav) and os.path.getsize(out_wav) > 0:
                            chunk_paths[i] = out_wav
                            if config.export_lrc or config.export_srt or config.export_vtt:
                                chunk_durations[i] = _get_wav_duration(out_wav)
                        break
                    except Exception as e:
                        if attempt == 0:
                            import time as _time
                            log(f"  [Ch{idx}] chunk {i} failed ({e}), retrying...")
                            _time.sleep(1)
                        else:
                            log(f"  [Ch{idx}] chunk {i} failed after retry: {e}")

                if prog_cb:
                    prog_cb((i + 1) / len(tts_jobs))

        if cancel.is_cancelled:
            return None

        # ── Generate Subtitles Asynchronously ─────────────────────────────────
        if config.export_lrc or config.export_srt or config.export_vtt:
            if subtitle_futures is not None and subtitle_futures_lock is not None:
                sub_future = _subtitle_executor.submit(
                    _generate_subtitles,
                    config, chapter, idx, tts_jobs, chunk_durations, log,
                )
                with subtitle_futures_lock:
                    subtitle_futures.append((idx, sub_future))
            else:
                _generate_subtitles(config, chapter, idx, tts_jobs, chunk_durations, log)

        # ── Ensure cover image is in valid format (e.g. JPEG/PNG) ────────────
        raw_cover = config.cover_image
        valid_cover = ""
        if raw_cover and os.path.exists(raw_cover):
            try:
                ext = os.path.splitext(raw_cover)[1].lower()
                if ext in (".jpg", ".jpeg", ".png"):
                    valid_cover = raw_cover
                else:
                    from PIL import Image
                    img = Image.open(raw_cover)
                    if img.mode in ("RGBA", "P", "LA"):
                        img = img.convert("RGB")
                    conv_path = os.path.join(temp_dir, "cover_converted.jpg")
                    img.save(conv_path, format="JPEG", quality=95)
                    valid_cover = conv_path
            except Exception:
                valid_cover = raw_cover if os.path.exists(raw_cover) else ""

        def _get_cover_flags(fmt: str, include_cover: bool) -> list[str]:
            if not include_cover or not valid_cover:
                return []
            f = (fmt or "").lower()
            if f == "mp3":
                return ["-map", "0:a", "-map", "1:v", "-c:v", "copy", "-disposition:v", "attached_pic", "-id3v2_version", "3"]
            elif f in ("m4b", "m4a", "mp4", "flac"):
                return ["-map", "0:a", "-map", "1:v", "-c:v", "copy", "-disposition:v", "attached_pic"]
            else:
                return ["-map", "0:a", "-map", "1:v", "-c:v", "copy"]

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
                has_cover = bool(valid_cover and os.path.exists(valid_cover))
                use_pure_rust = (config.output_format in ("mp3", "wav")) and not has_cover

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
                    
                    def _build_cmd(include_cover: bool):
                        cmd = ["ffmpeg", "-y", "-i", master_target]
                        if include_cover and valid_cover:
                            cmd += ["-i", valid_cover]
                        cmd += audio_settings
                        cmd += _get_cover_flags(config.output_format, include_cover)
                        cmd += [
                            "-metadata", f"title={chapter.title}",
                            "-metadata", f"artist={config.author}",
                            "-metadata", f"album={config.book_title}",
                            "-metadata", f"track={idx}",
                            "-metadata", "genre=Audiobook",
                            out_path,
                        ]
                        return cmd

                    try:
                        cmd = _build_cmd(has_cover)
                        res = subprocess.run(cmd, check=True, capture_output=True)
                    except subprocess.CalledProcessError as e:
                        stderr_log = e.stderr.decode("utf-8", errors="replace") if e.stderr else str(e)
                        if has_cover:
                            log(f"  [Ch{idx}] ⚠ Cover embedding failed ({stderr_log[:200]}). Retrying without cover image...")
                            try:
                                cmd_no_cover = _build_cmd(False)
                                res = subprocess.run(cmd_no_cover, check=True, capture_output=True)
                            except subprocess.CalledProcessError as e2:
                                stderr_log2 = e2.stderr.decode("utf-8", errors="replace") if e2.stderr else str(e2)
                                raise RuntimeError(f"FFmpeg encoding failed: {stderr_log2}")
                        else:
                            raise RuntimeError(f"FFmpeg encoding failed: {stderr_log}")
                
                return out_path

            except Exception as rust_err:
                log(f"  [Ch{idx}] ⚠ Rust mastering failed ({rust_err}). Falling back to Python/FFmpeg.")

        # ── Python fallback: In-memory WAV concat ─────────────────────────────
        pause_samples = np.zeros(int(config.pause * config.sample_rate), dtype=np.float32)

        valid_indices = [i for i, p in enumerate(chunk_paths) if p and os.path.exists(p)]
        last_valid_idx = valid_indices[-1] if valid_indices else -1
        audio_segments: list[np.ndarray] = []
        for i, p in enumerate(chunk_paths):
            if p and os.path.exists(p):
                try:
                    chunk_audio, _ = sf.read(p, dtype="float32")
                    audio_segments.append(chunk_audio)
                    if i != last_valid_idx:
                        audio_segments.append(pause_samples)
                except Exception:
                    pass

        if not audio_segments:
            log(f"  [Ch{idx}] ❌ No valid audio segments. Skipping.")
            return None

        raw_audio = np.concatenate(audio_segments)
        log(f"  [Ch{idx}] Concatenated {len(audio_segments)} segments "
            f"({len(raw_audio)/config.sample_rate:.1f}s) in-memory (python fallback)")

        # ── Single FFmpeg call: loudnorm + encode (piped via stdin) ───────────
        audio_settings, _, _ = get_format_settings(config.output_format)[:3]

        def _build_py_cmd(include_cover: bool):
            cmd = [
                "ffmpeg", "-y",
                "-f", "f32le",
                "-ar", str(config.sample_rate),
                "-ac", "1",
                "-i", "pipe:0",
            ]
            if include_cover and valid_cover:
                cmd += ["-i", valid_cover]
            cmd += ["-af", f"loudnorm=I={config.lufs}:TP={config.true_peak}:LRA=11"]
            cmd += audio_settings
            cmd += _get_cover_flags(config.output_format, include_cover)
            cmd += [
                "-metadata", f"title={chapter.title}",
                "-metadata", f"artist={config.author}",
                "-metadata", f"album={config.book_title}",
                "-metadata", f"track={idx}",
                "-metadata", "genre=Audiobook",
                out_path,
            ]
            return cmd

        has_cover = bool(valid_cover and os.path.exists(valid_cover))
        raw_bytes = raw_audio.tobytes()

        try:
            cmd = _build_py_cmd(has_cover)
            proc = subprocess.run(cmd, input=raw_bytes, check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            stderr_log = e.stderr.decode("utf-8", errors="replace") if e.stderr else str(e)
            if has_cover:
                log(f"  [Ch{idx}] ⚠ Cover image encoding failed ({stderr_log[:200]}). Retrying without cover image...")
                try:
                    cmd_no_cover = _build_py_cmd(False)
                    proc = subprocess.run(cmd_no_cover, input=raw_bytes, check=True, capture_output=True)
                except subprocess.CalledProcessError as e2:
                    stderr_log2 = e2.stderr.decode("utf-8", errors="replace") if e2.stderr else str(e2)
                    raise RuntimeError(f"FFmpeg python fallback encoding failed: {stderr_log2}")
            else:
                raise RuntimeError(f"FFmpeg python fallback encoding failed: {stderr_log}")

        return out_path


    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def _cleanup_chunk_files(paths: list[str | None]) -> None:
    """Removes temporary chunk WAV files. Logs warnings for failures.

    Thread-safe. Safe to call with an empty list or paths that no longer exist.
    """
    for path in paths:
        if not path:
            continue
        try:
            if os.path.exists(path):
                os.remove(path)
        except OSError as exc:
            import logging
            logging.getLogger(__name__).warning("Failed to remove temp chunk file %s: %s", path, exc)

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
            logger.warning("[preview_tts] Error: %s", e)
    return None
