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

import asyncio
import atexit
import concurrent.futures
import hashlib
import json
import logging
import os
import queue
import re
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed, CancelledError
from dataclasses import dataclass, field, fields, MISSING
from typing import Callable, Any, TYPE_CHECKING

import soundfile as sf

if TYPE_CHECKING:
    from audiobook_factory.tts_providers.base_tts_provider import BaseTTSProvider

logger = logging.getLogger(__name__)

_VALID_QUANTIZATION_MODES: frozenset[str] = frozenset({"none", "int8"})

_RETIRED_PROVIDERS: frozenset[str] = frozenset({"vibevoice", "vibe-voice", "vibevoice-1.5b"})
# Providers that existed in earlier releases; configs naming them still load.

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
    update_chapter_chunks,
    update_chapter_fields,
    update_chapter_retry,
    _WRITE_LOCK,
    _write_unlocked,
)

_MINIMUM_CHAPTER_WAV_BYTES: int = 10_000

_CANCELLED_ERRORS: tuple[type[BaseException], ...] = (CancelledError, asyncio.CancelledError)
# chapter_pipeline / gpu_pool raise asyncio.CancelledError (a BaseException),
# ThreadPoolExecutor raises concurrent.futures.CancelledError. Both mean "stop".

_CHUNK_CACHE_KEY_FILE: str = "chunk_cache.key"
# Sidecar in each chapter temp dir recording what its cached chunk WAVs were
# synthesized from, so a resume never splices in audio made from other
# text, another voice, or different TTS settings.

_BITRATE_FORMATS: tuple[str, ...] = ("mp3", "m4b", "m4a", "aac", "ogg")
# Lossy formats whose bitrate is driven by AudiobookConfig.bitrate_kbps.

_CHAPTER_MARKER_FORMATS: tuple[str, ...] = ("m4b", "m4a", "mp4", "mov", "mp3", "ogg", "webm")
# Containers that can carry chapter markers in a combined (single-file) book.
_COVER_FORMATS: tuple[str, ...] = ("m4b", "m4a", "mp4", "mp3", "flac")
# Containers a cover image can be attached to when combining.

_MIN_SPEED: float = 0.5
_MAX_SPEED: float = 2.0
# Range of FFmpeg's atempo filter in a single stage.

_ETA_REPORT_INTERVAL_SEC: float = 60.0
_ETA_MIN_SHARE: float = 0.02
# Below this share of the work the extrapolation is too noisy to print.
_SUMMARY_MAX_FLAGGED_PER_CHAPTER: int = 5

_VALID_VERIFY_MODES: tuple[str, ...] = ("off", "duration", "asr")


def _mark_chapter_completed(progress_path: str, chapter_num: int, output_wav_path: str) -> None:
    """Validates chapter output WAV and marks chapter completed in progress JSON.

    Enforces file existence and minimum file size (_MINIMUM_CHAPTER_WAV_BYTES).
    Does not update status if file is missing or corrupted.
    """
    if not output_wav_path or not os.path.exists(output_wav_path):
        logger.warning(
            "[Pipeline] Cannot mark chapter %d completed — file not found: %s",
            chapter_num, output_wav_path,
        )
        return
    size = os.path.getsize(output_wav_path)
    if size < _MINIMUM_CHAPTER_WAV_BYTES:
        logger.warning(
            "[Pipeline] Cannot mark chapter %d completed — WAV size too small (%d < %d bytes): %s",
            chapter_num, size, _MINIMUM_CHAPTER_WAV_BYTES, output_wav_path,
        )
        return
    try:
        update_chapter_status(progress_path, chapter_num, "completed", reset_chunks=True)
    except Exception as exc:
        logger.debug("[Pipeline] Could not update chapter %d status in %s: %s", chapter_num, progress_path, exc)


def _finalize_progress_file(progress_path: str, chapters: list) -> None:
    """Writes top-level generation_summary to progress JSON on run completion."""
    if not os.path.exists(progress_path):
        return
    try:
        with _WRITE_LOCK:
            data = read_progress_file(progress_path)
            chapter_entries = data.get("chapters", [])
            completed_count = sum(1 for c in chapter_entries if c.get("status") == "completed")
            failed_entries = [c for c in chapter_entries if c.get("status") == "failed"]
            failed_count = len(failed_entries)
            failed_chapters = [c.get("num") for c in failed_entries if c.get("num") is not None]
            total_ch = len(chapter_entries) if chapter_entries else len(chapters)
            all_complete = (completed_count == total_ch and total_ch > 0)

            from datetime import datetime, timezone
            data["generation_summary"] = {
                "total_chapters": total_ch,
                "completed_count": completed_count,
                "failed_count": failed_count,
                "failed_chapters": failed_chapters,
                "all_complete": all_complete,
                "finalized_at": datetime.now(timezone.utc).isoformat(),
            }
            _write_unlocked(progress_path, data)
    except Exception as exc:
        logger.warning("[Pipeline] Could not finalize progress file %s: %s", progress_path, exc)



_MAX_PARALLEL_CHAPTERS: int = 0
# 0 = auto (matches GPU count). >0 = override. Set at module level.

def _get_chapter_parallelism(
    pool: Any,
    config: AudiobookConfig | None = None,
) -> int:
    """Returns the number of chapters to process simultaneously.

    When parallel_mode="chunks" (default): returns 1, meaning one
    chapter is processed at a time with ALL GPUs splitting its chunks
    via Stage A's static contiguous distribution. pinned_device=None
    must be passed to the chapter pipeline for this to take effect.

    When parallel_mode="chapters": returns pool.device_count, meaning
    multiple chapters run simultaneously with each chapter pinned to
    one GPU. Each chapter uses only its assigned GPU.

    Args:
        pool: The active ProviderPool with device count information.
        config: AudiobookConfig controlling parallelism mode.
                If None, defaults to "chunks" behavior.

    Returns:
        Integer >= 1. Number of concurrent chapter pipelines.
    """
    if pool is None or not hasattr(pool, "device_count") or pool.device_count <= 0:
        return 1

    if _MAX_PARALLEL_CHAPTERS > 0:
        return min(_MAX_PARALLEL_CHAPTERS, pool.device_count)

    parallel_mode = getattr(config, "parallel_mode", "chunks")

    if parallel_mode == "chapters":
        logger.info(
            "[pipeline] Chapter-mode: %d chapters simultaneously, "
            "one GPU per chapter.",
            pool.device_count,
        )
        return pool.device_count

    # Default: "chunks" mode — one chapter at a time, all GPUs
    # split the chapter's chunks across devices.
    logger.info(
        "[pipeline] Chunk-mode: 1 chapter at a time, all %d GPU(s) "
        "splitting chunks via Stage A static distribution.",
        pool.device_count,
    )
    return 1


_CONFIG_SCHEMA_VERSION: int = 7
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
    voice_transcript:    str   = ""       # optional text transcript of reference voice for prompt-based cloning

    # ── TTS ───────────────────────────────────────────────────────────────────
    tts_provider_name:   str   = "qwen"   # "qwen" (Qwen3-TTS)
    temperature:         float = 0.3
    top_p:               float = 0.8
    max_len:             int   = 399      # max chars per TTS chunk

    # ── Pacing ────────────────────────────────────────────────────────────────
    pause:               float = 0.5     # seconds between sentences
    para_pause:          float = 1.2     # seconds between paragraphs

    # ── Audio mastering & Encoding ────────────────────────────────────────────
    lufs:                int   = -18
    true_peak:           float = -1.5
    bitrate_kbps:        int   = 64       # Audio encoding bitrate (kbps)
    channels:            int   = 1        # Audio channels (1 = mono, 2 = stereo)

    # ── Parallelism & Hardware Optimization ───────────────────────────────────
    worker_count:        int   = 1       # chapters/chunks in parallel
    parallel_mode:       str   = "chunks" # "chapters" | "chunks"
    gpu_count:           int   = 0       # 0 = auto-detect at runtime
    vram_headroom_gb:    float = 2.0     # GB reserved VRAM headroom for dynamic batching

    # ── Multi-Model Qwen3 ─────────────────────────────────────────────────────
    device:              str   = "cuda"
    tts_model_name:      str   = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    tts_instruct:        str   = ""       # Natural-language style / voice-design prompt
    tts_timbre:          str   = ""       # Built-in preset speaker (providers with preset voices)
    voice_preset:        str   = ""       # Saved provider voice preset file; replaces voice_file when set
    # Provider-specific settings, keyed by ProviderOption.key (see tts_providers/).
    tts_options:         dict  = field(default_factory=dict)

    # ── Modes ─────────────────────────────────────────────────────────────────
    preview_mode:        bool  = False   # show stats, no TTS
    export_text:         bool  = False   # write .txt per chapter
    export_lrc:          bool  = True    # write .lrc timed lyrics
    export_srt:          bool  = False   # write .srt subtitles
    export_vtt:          bool  = False   # write .webvtt subtitles
    single_file_mode:    bool  = False   # combine all into one big file

    # ── Retries & Resilience ──────────────────────────────────────────────────
    max_chapter_retries: int   = 2       # Max retry attempts per chapter on failure
    retry_failed_at_end: bool  = True    # Run a final pass for remaining failed chapters at end of pipeline

    # ── Misc ──────────────────────────────────────────────────────────────────
    force_reprocess:          bool  = False  # When True, forces re-extraction & re-synthesis of all chunks from scratch
    resume_incomplete_chunks: bool  = True   # When True, resumes mid-chapter from last completed chunk using disk cache
    regen_missing:            bool  = True   # When True, regenerates missing/failed chapter audio
    sample_rate:              int   = 24000
    repetition_penalty:       float = 1.05
    top_k:                    int   = 50
    speed:                    float = 1.0
    nfe_step:                 int   = 32
    seed:                     int   = -1
    torch_compile:            bool  = False
    quantization:             str   = "none"   # "none" | "int8"
    selected_chapters:        list  = field(default_factory=list) # Selected chapter titles/labels
    redo_chapters:            list  = field(default_factory=list) # Chapter numbers to regenerate even if completed
    batch_size:               int   = 0        # TTS chunks per forward pass; 0 = size from free VRAM

    # ── Narration quality ─────────────────────────────────────────────────────
    pack_sentences:           bool  = True     # Speak a paragraph's sentences together (up to max_len) for natural prosody
    normalize_speech_text:    bool  = True     # Rewrite numerals, currency, dates, abbreviations into speakable form
    verify_chunks:            str   = "duration"  # "off" | "duration" | "asr" — check each chunk and re-synthesize failures
    verify_max_retries:       int   = 2        # Re-synthesis attempts for a chunk that fails verification
    verify_asr_model:         str   = "openai/whisper-large-v3-turbo"  # Whisper model for verify_chunks="asr"
    verify_max_wer:           float = 0.3      # Transcript error rate above which a chunk is rejected

    # ── Pronunciation fixes ───────────────────────────────────────────────────
    # { regex_pattern: replacement }  applied before TTS
    pronunciation_map:   dict  = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict) -> "AudiobookConfig":
        """Constructs AudiobookConfig from a dict, tolerating unknown and
        missing keys.

        Unknown keys are silently dropped. Missing keys use the field's
        default value. A version mismatch logs a warning but does not raise.

        Migration notes: every field added since schema version 1 has a
        default, so older files load unchanged. Version 7 added
        ``voice_preset``, ``tts_options``, ``redo_chapters``, ``batch_size``,
        ``pack_sentences``, ``normalize_speech_text`` and the ``verify_*``
        fields, and retired the ``vibevoice`` provider, which is mapped back
        to the default engine here.

        Args:
            data: Dict from generation_progress.json settings section.

        Returns:
            Populated AudiobookConfig instance.
        """
        known_fields = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in data.items() if k in known_fields}

        if str(filtered.get("tts_provider_name", "")).lower().strip() in _RETIRED_PROVIDERS:
            logger.warning(
                "TTS provider '%s' was removed; falling back to the default engine.",
                filtered["tts_provider_name"],
            )
            filtered["tts_provider_name"] = "qwen"
            filtered.pop("tts_model_name", None)
        if not isinstance(filtered.get("tts_options", {}), dict):
            filtered["tts_options"] = {}

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


_VALID_OUTPUT_FORMATS: tuple[str, ...] = (
    "mp3", "wav", "flac", "m4b", "m4a", "aac", "ogg", "webm", "mp4", "mov"
)


def _validate_config(config: AudiobookConfig) -> None:
    """Validate AudiobookConfig options before running the pipeline.

    Raises:
        ValueError: If configuration values are invalid.
    """
    if config.quantization not in _VALID_QUANTIZATION_MODES:
        raise ValueError(
            f"Invalid quantization mode '{config.quantization}'. "
            f"Supported options: {sorted(_VALID_QUANTIZATION_MODES)}"
        )
    if config.output_format not in _VALID_OUTPUT_FORMATS:
        raise ValueError(
            f"Invalid output_format '{config.output_format}'. "
            f"Supported options: {sorted(_VALID_OUTPUT_FORMATS)}"
        )
    if str(getattr(config, "verify_chunks", "duration")).lower() not in _VALID_VERIFY_MODES:
        raise ValueError(
            f"Invalid verify_chunks '{config.verify_chunks}'. "
            f"Supported options: {list(_VALID_VERIFY_MODES)}"
        )
    if config.parallel_mode not in ("chunks", "chapters"):
        raise ValueError(
            f"Invalid parallel_mode '{config.parallel_mode}'. "
            f"Supported options: ['chapters', 'chunks']"
        )



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

    from audiobook_factory.tts_providers import get_tts_provider, provider_info

    engine = provider_info(config.tts_provider_name)
    # Providers that change speed themselves get config.speed; for the rest
    # the finished chapter is time-stretched while it is encoded.
    post_speed = 1.0 if engine.supports_speed else _clamp_speed(config.speed)

    log(f"[Pipeline] Starting — {total} chapter(s)")
    log(f"[Pipeline] Output  : {config.output_dir}")
    log(f"[Pipeline] Format  : {config.output_format}")
    log(f"[Pipeline] Engine  : {engine.display_name}")
    if not engine.commercial_use:
        log(f"[Pipeline] ⚠ {engine.display_name} weights are licensed {engine.license} — non-commercial use only.")
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
    tasks = _number_chapters(chapters)
    redo = {int(n) for n in (getattr(config, "redo_chapters", None) or []) if str(n).lstrip("-").isdigit()}

    from dataclasses import asdict
    settings_dict = {}
    try:
        settings_dict = dict(asdict(config))
        # One-shot instructions describe this run, not the book: saved back
        # they would wipe or redo finished chapters on every later resume.
        settings_dict.update(force_reprocess=False, redo_chapters=[], preview_mode=False)
    except Exception as e:
        logger.warning("Error serializing config: %s", e)

    try:
        previous = read_progress_file(prog_path_out)
    except (FileNotFoundError, ValueError):
        previous = {}

    progress_data = dict(previous)
    progress_data["book_title"] = previous.get("book_title") or config.book_title
    progress_data["book_path"] = previous.get("book_path") or getattr(config, "book_path", "")
    progress_data["voice_file"] = previous.get("voice_file") or getattr(config, "voice_file", "")
    # The settings of the latest run win, so a later CLI resume uses them.
    progress_data["settings"] = settings_dict or previous.get("settings") or {}
    progress_data["chapters"] = _reconcile_chapter_entries(previous.get("chapters", []), tasks)
    try:
        write_progress_file(prog_path_out, progress_data)
    except OSError as e:
        logger.warning("Error writing progress json: %s", e)

    # Sync to temp for user visibility
    try:
        write_progress_file(prog_path_tmp, progress_data)
    except OSError as exc:
        logger.debug("Could not sync progress to temp: %s", exc)

    status_by_num: dict[int, str] = {
        int(entry["num"]): entry.get("status", "pending") for entry in progress_data["chapters"]
        if str(entry.get("num", "")).isdigit()
    }

    # A finished single-file book has no per-chapter files left (they are
    # removed once combined); without this check a re-run would see every
    # chapter as "completed but missing" and synthesize the whole book again.
    if config.single_file_mode and not config.force_reprocess and not redo:
        combined_path = _combined_book_path(config)
        if os.path.exists(combined_path) and all(
            status_by_num.get(num) == "completed" for num, _ in tasks
        ):
            log(f"[Pipeline] ⏩ Already complete: {os.path.basename(combined_path)}")
            progress(total, total)
            return [combined_path]

    # ── Shared TTS Provider / GPU Pool Setup ──────────────────────────────────
    from audiobook_factory.chunk_verifier import ChunkVerifier
    from audiobook_factory.gpu_pool import GPUPoolManager

    # Free the Voice Studio's cached preview model (if any) before the
    # real generation run claims GPU memory for its own provider pool —
    # a leftover preview model from a different engine can otherwise
    # starve the pool warmup of VRAM.
    _cleanup_preview_provider()
    pool = GPUPoolManager.instance().get_pool(
        provider_name=config.tts_provider_name,
        provider_factory=lambda dev: get_tts_provider(
            config.tts_provider_name, config, device=dev, dtype_override=recommended_dtype
        ),
        min_vram_gb=engine.min_vram_gb,
        gpu_count_override=config.gpu_count,
    )
    log(f"[Pipeline] Devices : {', '.join(pool.devices)}")

    verifier = ChunkVerifier(
        mode=getattr(config, "verify_chunks", "duration"),
        language=config.language,
        speed=config.speed if engine.supports_speed else 1.0,
        asr_model=getattr(config, "verify_asr_model", "openai/whisper-large-v3-turbo"),
        max_error_rate=getattr(config, "verify_max_wer", 0.3),
    )
    if verifier.enabled:
        log(f"[Pipeline] Chunk verification: {verifier.mode}")

    output_files: list[str] = []
    output_order: dict[str, int] = {}
    _lock = threading.Lock()

    def _sort_outputs() -> None:
        # Filenames are "Chapter 10 - …", which sorts before "Chapter 2 - …"
        # lexicographically, so order by chapter number instead.
        output_files.sort(key=lambda p: (output_order.get(p, 0), p))

    subtitle_futures: list[tuple[int, concurrent.futures.Future]] = []
    subtitle_futures_lock = threading.Lock()

    positions = {num: position for position, (num, _) in enumerate(tasks, 1)}
    chapter_progress = {num: 0.0 for num, _ in tasks}
    eta = _EtaTracker(
        {
            num: max(1, len(chapter.text or "") or sum(len(s) for s in (chapter.sentences or [])))
            for num, chapter in tasks
            if status_by_num.get(num) != "completed" or config.force_reprocess or num in redo
        },
        log,
    )

    def _update_chapter_prog(num, frac):
        with _lock:
            chapter_progress[num] = frac
            sum_frac = sum(chapter_progress.values())
            progress(sum_frac, total)
            eta.update(num, frac)

    def _process(num_chapter, pinned_device: str | None = None):
        num, chapter = num_chapter
        position = positions[num]
        if cancel.is_cancelled:
            return None

        ch_status = status_by_num.get(num, "pending")
        forced = config.force_reprocess or num in redo

        if ch_status == "completed" and not forced:
            log(f"[Chapter {position}/{total}] ⏩ Already completed. Skipping.")
            _update_chapter_prog(num, 1.0)
            # Find the existing file to return its path
            safe_name = make_safe_filename(chapter.title, num, config.output_dir, f".{config.output_format}")
            existing_path = os.path.join(config.output_dir, safe_name)
            if os.path.exists(existing_path):
                with _lock:
                    output_files.append(existing_path)
                    output_order[existing_path] = num
                return existing_path
            # File is missing — check user's preference
            if not getattr(config, "regen_missing", True):
                log(f"  [Ch{num}] ⚠ Warning: Marked 'completed' but file not found. Skipping (regen_missing=False).")
                return None
            log(f"  [Ch{num}] ⚠ Warning: Marked 'completed' but file not found. Re-generating.")

        log(f"\n[Chapter {position}/{total}] '{chapter.title}'")
        if forced and ch_status == "completed":
            for p in [prog_path_out, prog_path_tmp]:
                update_chapter_status(p, num, "pending", reset_chunks=True)
        try:
            path = _process_chapter_with_retry(
                config=config,
                chapter=chapter,
                idx=num,
                total=total,
                log=log,
                cancel=cancel,
                pool=pool,
                prog_cb=lambda f: _update_chapter_prog(num, f),
                pinned_device=pinned_device,
                subtitle_futures=subtitle_futures,
                subtitle_futures_lock=subtitle_futures_lock,
                prog_path_out=prog_path_out,
                prog_path_tmp=prog_path_tmp,
                verifier=verifier,
                post_speed=post_speed,
                discard_cache=forced,
            )
            if path:
                with _lock:
                    if path not in output_files:
                        output_files.append(path)
                    output_order[path] = num
                status_by_num[num] = "completed"
                log(f"[Chapter {position}/{total}] ✅ → {os.path.basename(path)}")
            return path
        finally:
            _update_chapter_prog(num, 1.0)

    try:
        max_parallel = _get_chapter_parallelism(pool, config)
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
                    except _CANCELLED_ERRORS:
                        cancel.cancel()
                        break
                    except Exception as exc:
                        log(f"[Pipeline] Chapter execution error: {exc}")
        else:
            for t in tasks:
                if cancel.is_cancelled:
                    log("[Pipeline] ⛔ Cancelled.")
                    break
                _process(t, pinned_device=None)

        if not cancel.is_cancelled:
            progress(total, total)

        if cancel.is_cancelled:
            log(f"\n[Pipeline] ⛔ Cancelled — {len(output_files)} file(s) saved.")
            _sort_outputs()
            return output_files

        # ── End-of-Run Retry Pass ─────────────────────────────────────────────────
        if getattr(config, "retry_failed_at_end", True) and not cancel.is_cancelled:
            try:
                curr_prog = read_progress_file(prog_path_out)
                failed_nums = {
                    int(c["num"]) for c in curr_prog.get("chapters", [])
                    if c.get("status") == "failed" and str(c.get("num", "")).isdigit()
                }
                failed_tasks = [t for t in tasks if t[0] in failed_nums]
                if failed_tasks:
                    log(f"\n[Pipeline] 🔄 End-of-run retry pass starting for {len(failed_tasks)} failed chapter(s)...")
                    for t in failed_tasks:
                        if cancel.is_cancelled:
                            break
                        for p in [prog_path_out, prog_path_tmp]:
                            update_chapter_status(p, t[0], "pending")
                        status_by_num[t[0]] = "pending"
                        _process(t, pinned_device=None)
            except Exception as exc:
                logger.warning("[Pipeline] End-of-run retry pass encountered error: %s", exc)

        _sort_outputs()
        _log_run_summary(prog_path_out, tasks, log)

        # ── Single File Mode (Combine all chapters) ───────────────────────────────
        if config.single_file_mode and len(output_files) > 1:
            log("\n[Pipeline] 📦 Combining chapters into a single file...")
            titles = {num: chapter.title for num, chapter in tasks}
            try:
                full_path = _combine_chapters(
                    config, output_files, [titles.get(output_order.get(p, 0), "") for p in output_files]
                )
                log(f"[Pipeline] 📦 Combined file created: {os.path.basename(full_path)}")
                for p in output_files:
                    try:
                        os.remove(p)
                    except OSError as exc:
                        logger.debug("Could not remove chapter file %s: %s", p, exc)
                output_files = [full_path]
            except Exception as e:
                log(f"[Pipeline] ❌ Failed to combine: {e}")

        log(f"\n[Pipeline] ✅ Complete — {len(output_files)} file(s) generated.")
        return output_files

    finally:
        for p in [prog_path_out, prog_path_tmp]:
            _finalize_progress_file(p, chapters)
        _await_subtitle_futures(subtitle_futures, cancel)
        verifier.close()


def _clamp_speed(speed: Any) -> float:
    """Returns a playback speed FFmpeg's atempo filter accepts in one stage."""
    try:
        value = float(speed or 1.0)
    except (TypeError, ValueError):
        return 1.0
    return min(_MAX_SPEED, max(_MIN_SPEED, value))


def _number_chapters(chapters: list[ExtractedChapter]) -> list[tuple[int, ExtractedChapter]]:
    """Pairs each chapter with the number used for its file, tags and progress entry.

    The extractor's own ``num`` is kept when it is usable, so a run over a
    subset of a book (chapters 50–60) writes "Chapter 50 - …" rather than
    renumbering from 1 and colliding with an earlier run. Falls back to the
    position in the list when the numbers are missing, repeated or not
    positive integers.
    """
    nums = [getattr(chapter, "num", None) for chapter in chapters]
    usable = all(isinstance(n, int) and not isinstance(n, bool) and n > 0 for n in nums)
    if usable and len(set(nums)) == len(nums):
        return list(zip(nums, chapters))
    return list(enumerate(chapters, 1))


def _title_key(title: str) -> str:
    """Normalises a chapter title for matching progress entries across runs."""
    from audiobook_factory.utils import normalize_chapter_title_for_matching

    _, core = normalize_chapter_title_for_matching(title or "")
    return " ".join((core or "").split())


def _reconcile_chapter_entries(
    previous: list[dict],
    tasks: list[tuple[int, ExtractedChapter]],
) -> list[dict]:
    """Rebuilds the progress file's chapter list for the chapters of this run.

    Status is carried over only from an earlier entry with the *same title*,
    preferring the one that also has the same number. Matching on number
    alone would let "chapter 3 of another selection (or another book) is
    done" skip a chapter that was never generated. Earlier entries for
    chapters outside this run are kept so a subset run does not erase the
    record of the rest of the book.
    """
    by_key: dict[str, list[dict]] = {}
    for entry in previous:
        if isinstance(entry, dict):
            by_key.setdefault(_title_key(entry.get("title", "")), []).append(entry)

    used: set[int] = set()
    reconciled: list[dict] = []
    for num, chapter in tasks:
        candidates = [e for e in by_key.get(_title_key(chapter.title), []) if id(e) not in used]
        match = next((e for e in candidates if str(e.get("num")) == str(num)), None)
        if match is None and candidates:
            match = candidates[0]
        entry = {
            "num": num,
            "title": chapter.title,
            "status": "pending",
            "completed_chunks": [],
            "text": chapter.text,
            "sentences": chapter.sentences,
        }
        if match is not None:
            used.add(id(match))
            entry["status"] = match.get("status", "pending")
            for key in ("retry_count", "last_error", "duration", "flagged_chunks"):
                if key in match:
                    entry[key] = match[key]
            if str(match.get("num")) == str(num):
                entry["completed_chunks"] = list(match.get("completed_chunks", []))
        reconciled.append(entry)

    taken = {str(num) for num, _ in tasks}
    for entry in previous:
        if isinstance(entry, dict) and id(entry) not in used and str(entry.get("num")) not in taken:
            reconciled.append(entry)
            taken.add(str(entry.get("num")))
    reconciled.sort(key=lambda e: (int(e["num"]) if str(e.get("num", "")).isdigit() else 10**9))
    return reconciled


class _EtaTracker:
    """Estimates time remaining from the share of text already synthesized.

    Chapters are weighted by their length, since "3 of 10 chapters done"
    says little when chapter lengths differ tenfold. Not thread-safe: the
    caller serialises ``update``.
    """

    def __init__(self, weights: dict[int, int], log: Callable[[str], None]) -> None:
        self._weights = weights
        self._total = float(sum(weights.values())) or 1.0
        self._done: dict[int, float] = {num: 0.0 for num in weights}
        self._log = log
        self._started = time.monotonic()
        self._last_report = self._started

    def update(self, num: int, fraction: float) -> None:
        """Records a chapter's progress and logs an estimate once a minute."""
        if num not in self._done:
            return
        self._done[num] = max(0.0, min(1.0, float(fraction)))
        now = time.monotonic()
        if now - self._last_report < _ETA_REPORT_INTERVAL_SEC:
            return
        share = sum(self._weights[n] * f for n, f in self._done.items()) / self._total
        if share < _ETA_MIN_SHARE or share >= 1.0:
            return
        self._last_report = now
        elapsed = now - self._started
        remaining = elapsed * (1.0 - share) / share
        self._log(
            f"[Pipeline] ⏱ {share * 100:.1f}% of remaining text done — "
            f"elapsed {_format_hms(elapsed)}, about {_format_hms(remaining)} left."
        )


def _format_hms(seconds: float) -> str:
    """Formats a duration as H:MM:SS."""
    seconds = int(max(0, seconds))
    return f"{seconds // 3600}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def _log_run_summary(
    progress_path: str,
    tasks: list[tuple[int, ExtractedChapter]],
    log: Callable[[str], None],
) -> None:
    """Logs which chapters failed and which chunks were kept despite failing verification."""
    try:
        data = read_progress_file(progress_path)
    except (FileNotFoundError, ValueError):
        return
    wanted = {str(num) for num, _ in tasks}
    entries = [c for c in data.get("chapters", []) if str(c.get("num")) in wanted]
    failed = [c for c in entries if c.get("status") == "failed"]
    flagged = [(c, c.get("flagged_chunks") or []) for c in entries if c.get("flagged_chunks")]
    if not failed and not flagged:
        return
    log("\n[Pipeline] ── Summary ──")
    for c in failed:
        log(f"[Pipeline] ❌ Chapter {c.get('num')} '{c.get('title', '')}' failed: {c.get('last_error') or 'unknown error'}")
    for c, chunks in flagged:
        log(
            f"[Pipeline] ⚠ Chapter {c.get('num')} '{c.get('title', '')}': {len(chunks)} chunk(s) "
            f"kept after failing verification — worth a listen:"
        )
        for item in chunks[:_SUMMARY_MAX_FLAGGED_PER_CHAPTER]:
            log(f"[Pipeline]     “{str(item.get('text', ''))[:70]}” ({item.get('reason', '')})")
    if failed:
        log(f"[Pipeline] Re-run to retry the {len(failed)} failed chapter(s); finished chapters are skipped.")


def _probe_duration(path: str) -> float:
    """Returns an audio file's duration in seconds (0.0 when it cannot be read)."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", path],
            capture_output=True, text=True, timeout=60,
        )
        return float(result.stdout.strip())
    except (OSError, ValueError, subprocess.SubprocessError):
        pass
    try:
        return float(sf.info(path).duration)
    except Exception:
        return 0.0


def _ffmetadata_escape(value: str) -> str:
    """Escapes a value for an FFMETADATA1 file."""
    out = str(value or "")
    for ch in ("\\", "=", ";", "#"):
        out = out.replace(ch, "\\" + ch)
    return out.replace("\n", " ")


def _combined_book_path(config: AudiobookConfig) -> str:
    """Path of the single-file audiobook for this config."""
    from audiobook_factory.filename_sanitizer import _sanitize_base_name

    return os.path.join(
        config.output_dir, f"{_sanitize_base_name(config.book_title)}.{config.output_format}"
    )


def _combine_chapters(config: AudiobookConfig, files: list[str], titles: list[str]) -> str:
    """Joins chapter files into one audiobook file with chapter markers.

    Audio is stream-copied (no re-encode). Containers that carry chapters
    (M4B/M4A/MP4, MP3, OGG, WebM) get one marker per chapter, so players show
    a navigable chapter list; the cover image and book tags are added too.

    Args:
        config: AudiobookConfig of the run.
        files: Chapter audio files in reading order.
        titles: Chapter titles aligned with ``files``.

    Returns:
        Path of the combined file.

    Raises:
        RuntimeError: If FFmpeg fails.
    """
    from audiobook_factory.filename_sanitizer import _sanitize_base_name

    fmt = config.output_format
    work_dir = os.path.join(config.output_dir, ".temp_chunks")
    os.makedirs(work_dir, exist_ok=True)
    list_txt = os.path.join(work_dir, "concat_list.txt")
    meta_txt = os.path.join(work_dir, "concat_meta.txt")
    full_path = _combined_book_path(config)
    if os.path.abspath(full_path) in {os.path.abspath(f) for f in files}:
        full_path = os.path.join(config.output_dir, f"{_sanitize_base_name(config.book_title)} (complete).{fmt}")

    with open(list_txt, "w", encoding="utf-8") as fh:
        for p in files:
            p_safe = os.path.abspath(p).replace('\\', '/')
            fh.write("file '" + p_safe.replace("'", "'\\''") + "'\n")

    with open(meta_txt, "w", encoding="utf-8") as fh:
        fh.write(";FFMETADATA1\n")
        fh.write(f"title={_ffmetadata_escape(config.book_title)}\n")
        fh.write(f"album={_ffmetadata_escape(config.book_title)}\n")
        fh.write(f"artist={_ffmetadata_escape(config.author)}\n")
        fh.write("genre=Audiobook\n")
        start_ms = 0
        for p, title in zip(files, titles):
            end_ms = start_ms + max(1, int(round(_probe_duration(p) * 1000)))
            fh.write("\n[CHAPTER]\nTIMEBASE=1/1000\n")
            fh.write(f"START={start_ms}\nEND={end_ms}\n")
            fh.write(f"title={_ffmetadata_escape(title or os.path.splitext(os.path.basename(p))[0])}\n")
            start_ms = end_ms

    with_chapters = fmt in _CHAPTER_MARKER_FORMATS
    cover = _ensure_valid_cover_image(config.cover_image, work_dir) if fmt in _COVER_FORMATS else ""

    def _build(include_cover: bool) -> list[str]:
        cmd = ["ffmpeg", "-y", "-f", "concat", "-safe", "0", "-i", list_txt, "-i", meta_txt]
        if include_cover and cover:
            cmd += ["-i", cover]
        cmd += ["-map", "0:a", "-map_metadata", "1"]
        cmd += ["-map_chapters", "1"] if with_chapters else ["-map_chapters", "-1"]
        if include_cover and cover:
            cmd += ["-map", "2:v", "-c:v", "copy", "-disposition:v", "attached_pic"]
            if fmt == "mp3":
                cmd += ["-id3v2_version", "3"]
        cmd += ["-c:a", "copy", full_path]
        return cmd

    try:
        try:
            subprocess.run(_build(bool(cover)), check=True, capture_output=True)
        except subprocess.CalledProcessError as exc:
            if not cover:
                raise
            logger.warning(
                "Combining with cover failed (%s); retrying without the cover.",
                (exc.stderr or b"").decode("utf-8", errors="replace")[:200],
            )
            subprocess.run(_build(False), check=True, capture_output=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(
            "FFmpeg concat failed: " + (exc.stderr or b"").decode("utf-8", errors="replace")[-400:]
        ) from exc
    finally:
        for tmp in (list_txt, meta_txt):
            try:
                os.remove(tmp)
            except OSError:
                pass
    return full_path


def _subtitle_cues(
    tts_jobs: list,
    chunk_durations: list[float],
    default_pause: float,
    post_speed: float = 1.0,
) -> list[tuple[float, float, str]]:
    """Builds (start, end, text) subtitle cues from the synthesized chunks.

    A chunk that packs several sentences is divided among them in proportion
    to their length, which keeps subtitles sentence-sized without forcing the
    TTS model to speak one sentence per call.
    """
    speed = post_speed if post_speed and post_speed > 0 else 1.0
    cues: list[tuple[float, float, str]] = []
    cursor = 0.0
    for job, duration in zip(tts_jobs, chunk_durations):
        if isinstance(job, str):
            sentences: tuple[str, ...] = (job,)
            pause_after = default_pause
        else:
            sentences = tuple(job.sentences) or (job.text,)
            pause_after = job.pause_after
        total_chars = sum(len(s) for s in sentences) or 1
        offset = cursor
        for sentence in sentences:
            share = duration * len(sentence) / total_chars
            cues.append((offset / speed, (offset + share) / speed, sentence))
            offset += share
        cursor += duration + pause_after
    return cues


def _generate_subtitles(
    config: AudiobookConfig,
    chapter: ExtractedChapter,
    idx: int,
    tts_jobs: list,
    chunk_durations: list[float],
    log: Callable[[str], None],
    post_speed: float = 1.0,
) -> None:
    """Generate LRC, SRT, and VTT subtitle files for one chapter.

    Called asynchronously via _subtitle_executor. All three formats are
    written in a single call. Failures are caught per-format and logged as
    warnings — subtitle files are non-critical output.

    Args:
        config: AudiobookConfig controlling which formats to export.
        chapter: ExtractedChapter providing the chapter title.
        idx: Chapter number used in log messages and filename generation.
        tts_jobs: Chunks in chapter order — ``SpeechChunk`` objects, or plain
            strings (each followed by ``config.pause``).
        chunk_durations: Duration in seconds for each chunk in tts_jobs.
        log: Callable for progress reporting.
        post_speed: Time-stretch applied to the chapter after synthesis.
    """
    cues = _subtitle_cues(tts_jobs, chunk_durations, float(config.pause), post_speed)

    # ── Generate LRC timed lyrics ─────────────────────────────────────────
    if config.export_lrc:
        lrc_name = make_safe_filename(chapter.title, idx, config.output_dir, ".lrc")
        lrc_path = os.path.join(config.output_dir, lrc_name)
        try:
            with open(lrc_path, "w", encoding="utf-8") as fh:
                for start, _end, text in cues:
                    m, s = divmod(start, 60)
                    fh.write(f"[{int(m):02d}:{s:05.2f}]{text}\n")
            log(f"  [Ch{idx}] LRC exported → {lrc_name}")
        except Exception as e:
            log(f"  [Ch{idx}] LRC export failed: {e}")

    # ── Generate SRT timed subtitles ──────────────────────────────────────
    if config.export_srt:
        srt_name = make_safe_filename(chapter.title, idx, config.output_dir, ".srt")
        srt_path = os.path.join(config.output_dir, srt_name)
        try:
            from audiobook_factory.utils import seconds_to_srt_time
            with open(srt_path, "w", encoding="utf-8") as fh:
                for i, (start, end, text) in enumerate(cues, 1):
                    fh.write(f"{i}\n{seconds_to_srt_time(start)} --> {seconds_to_srt_time(end)}\n{text}\n\n")
            log(f"  [Ch{idx}] SRT exported → {srt_name}")
        except Exception as e:
            log(f"  [Ch{idx}] SRT export failed: {e}")

    # ── Generate WebVTT timed subtitles ───────────────────────────────────
    if config.export_vtt:
        vtt_name = make_safe_filename(chapter.title, idx, config.output_dir, ".vtt")
        vtt_path = os.path.join(config.output_dir, vtt_name)
        try:
            from audiobook_factory.utils import seconds_to_vtt_time
            with open(vtt_path, "w", encoding="utf-8") as fh:
                fh.write("WEBVTT\n\n")
                for i, (start, end, text) in enumerate(cues, 1):
                    fh.write(f"{i}\n{seconds_to_vtt_time(start)} --> {seconds_to_vtt_time(end)}\n{text}\n\n")
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
# Chapter processing & retries
# ══════════════════════════════════════════════════════════════════════════════

def _process_chapter_with_retry(
    config: AudiobookConfig,
    chapter: ExtractedChapter,
    idx: int,
    total: int,
    log: Callable[[str], None],
    cancel: CancelToken,
    provider: Any = None,
    pool: Any = None,
    prog_cb: Callable[[float], None] | None = None,
    pinned_device: str | None = None,
    subtitle_futures: list | None = None,
    subtitle_futures_lock: Any = None,
    completed_chunks: list[int] | None = None,
    prog_path_out: str | None = None,
    prog_path_tmp: str | None = None,
    verifier: Any = None,
    post_speed: float = 1.0,
    discard_cache: bool = False,
) -> str | None:
    """Wraps _process_chapter with automatic retry logic and backoff.

    Retries up to config.max_chapter_retries times. Clears CUDA cache before retrying.
    Updates progress JSON with retry count and error messages on failure.
    """
    max_attempts = max(1, config.max_chapter_retries + 1)
    last_error: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        if cancel.is_cancelled:
            break

        if attempt > 1:
            backoff = min(30, 5 * (attempt - 1))
            log(f"  [Ch{idx}] 🔄 Retry attempt {attempt}/{max_attempts} after {backoff}s backoff...")
            deadline = time.monotonic() + backoff
            while time.monotonic() < deadline and not cancel.is_cancelled:
                time.sleep(0.25)
            import gc
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

        # Attempts resume from whatever earlier attempts (or an earlier run)
        # left in the chunk cache; the last retry starts clean in case a
        # cached chunk is the problem.
        fresh = discard_cache if attempt == 1 else attempt == max_attempts

        try:
            path = _process_chapter(
                config, chapter, idx, total, log, cancel, pool=pool,
                prog_cb=prog_cb,
                pinned_device=pinned_device,
                subtitle_futures=subtitle_futures,
                subtitle_futures_lock=subtitle_futures_lock,
                fresh=fresh,
                verifier=verifier,
                post_speed=post_speed,
            )

            if path and os.path.exists(path) and os.path.getsize(path) >= _MINIMUM_CHAPTER_WAV_BYTES:
                for p in [prog_path_out, prog_path_tmp]:
                    if p:
                        _mark_chapter_completed(p, idx, path)
                return path
            else:
                msg = f"Output file missing or under size threshold ({_MINIMUM_CHAPTER_WAV_BYTES} bytes)"
                if path is None and last_error is None and not _chapter_has_text(chapter):
                    # Nothing to narrate is not a failure worth retrying.
                    return None
                last_error = RuntimeError(msg)
                log(f"  [Ch{idx}] ⚠ Attempt {attempt} failed: {msg}")
        except _CANCELLED_ERRORS:
            log(f"  [Ch{idx}] ⛔ Cancelled.")
            return None
        except Exception as exc:
            if cancel.is_cancelled:
                log(f"  [Ch{idx}] ⛔ Cancelled.")
                return None
            logger.exception("[Ch%d] Attempt %d failed", idx, attempt)
            last_error = exc
            log(f"  [Ch{idx}] ❌ Attempt {attempt} failed with error: {exc}")

        if cancel.is_cancelled:
            return None

        # Persist retry details to progress JSON
        for p in [prog_path_out, prog_path_tmp]:
            if p:
                try:
                    update_chapter_retry(p, idx, attempt, str(last_error or "Unknown error"))
                except Exception as e:
                    logger.debug("Could not write retry status to %s: %s", p, e)

    if cancel.is_cancelled:
        return None
    log(f"[Chapter {idx}/{total}] ❌ Failed after {max_attempts} attempts.")
    return None


def _chapter_has_text(chapter: ExtractedChapter) -> bool:
    """Returns True if the chapter contains anything to synthesize."""
    if any((s or "").strip() for s in (chapter.sentences or [])):
        return True
    return bool((chapter.text or "").strip())


def _chunk_cache_fingerprint(config: AudiobookConfig, tts_jobs: list[str]) -> str:
    """Hashes everything that determines what a chapter's chunk WAVs sound like.

    Cached chunks are only reusable when the chunk text, the narrator voice
    and the TTS settings are all unchanged.
    """
    voice_digest = ""
    for source in (getattr(config, "voice_preset", ""), config.voice_file):
        if source and os.path.exists(source):
            try:
                with open(source, "rb") as vf:
                    voice_digest += hashlib.sha256(vf.read()).hexdigest()
            except OSError as exc:
                logger.debug("Could not hash voice source %s: %s", source, exc)
    payload = json.dumps(
        [
            tts_jobs, voice_digest, config.voice_transcript,
            config.tts_provider_name, config.tts_model_name,
            config.tts_instruct, config.tts_timbre, config.language,
            config.temperature, config.top_p, config.top_k,
            config.repetition_penalty, config.speed, config.nfe_step,
            config.seed, config.quantization,
            sorted((str(k), str(v)) for k, v in (getattr(config, "tts_options", None) or {}).items()),
        ],
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _prune_chapter_temp_dir(temp_dir: str, idx: int, keep_chunks: bool) -> None:
    """Removes a chapter's temp directory, optionally keeping resumable chunks.

    Args:
        temp_dir: The chapter's `.temp_chunks/abm_chNNN` directory.
        idx: Chapter number used in chunk filenames.
        keep_chunks: When True, chunk WAVs and their cache key survive so an
            interrupted chapter can resume instead of starting over.
    """
    if not keep_chunks:
        shutil.rmtree(temp_dir, ignore_errors=True)
        return
    chunk_prefix = f"chunk_ch_{idx}_"
    kept = 0
    try:
        for name in os.listdir(temp_dir):
            if name.startswith(chunk_prefix) and name.endswith(".wav"):
                kept += 1
            elif name != _CHUNK_CACHE_KEY_FILE:
                target = os.path.join(temp_dir, name)
                if os.path.isdir(target):
                    shutil.rmtree(target, ignore_errors=True)
                else:
                    os.remove(target)
    except OSError as exc:
        logger.debug("Could not prune %s: %s", temp_dir, exc)
    if kept == 0:
        shutil.rmtree(temp_dir, ignore_errors=True)


def _encoder_args(config: AudiobookConfig) -> list[str]:
    """Builds FFmpeg output-side sample-rate, channel, codec and bitrate args.

    `-ar`/`-ac` are emitted as *output* options: filters such as loudnorm
    upsample internally and FFmpeg keeps that rate unless told otherwise.
    Format presets that carry their own quality (`-q:a`) or bitrate are
    stripped for bitrate-driven formats, because libmp3lame and libvorbis
    ignore `-b:a` whenever `-q:a` is present.
    """
    from audiobook_factory.ffmpeg_utils import get_format_settings

    preset = list(get_format_settings(config.output_format)[0])
    args = ["-ar", str(int(config.sample_rate)), "-ac", str(int(getattr(config, "channels", 1) or 1))]
    if config.output_format not in _BITRATE_FORMATS:
        return args + preset

    codec_args: list[str] = []
    skip_next = False
    for token in preset:
        if skip_next:
            skip_next = False
            continue
        if token in ("-q:a", "-b:a"):
            skip_next = True
            continue
        codec_args.append(token)
    return args + codec_args + ["-b:a", f"{int(getattr(config, 'bitrate_kbps', 64) or 64)}k"]


def _tag_mp3(path: str, chapter: ExtractedChapter, config: AudiobookConfig, idx: int) -> None:
    """Writes ID3 tags onto an MP3 produced by the Rust encoder.

    The Rust fast path writes a bare MPEG stream, so without this the file
    carries no title/author/album/track for audiobook players to read.
    """
    from mutagen.id3 import ID3, TALB, TCON, TIT2, TPE1, TRCK

    tags = ID3()
    tags.add(TIT2(encoding=3, text=chapter.title))
    tags.add(TPE1(encoding=3, text=config.author))
    tags.add(TALB(encoding=3, text=config.book_title))
    tags.add(TRCK(encoding=3, text=str(idx)))
    tags.add(TCON(encoding=3, text="Audiobook"))
    tags.save(path, v2_version=3)


def _ensure_valid_cover_image(raw_cover: str | None, work_dir: str) -> str:
    """Returns a cover image path FFmpeg can embed, converting if necessary.

    JPEG and PNG are used as they are; anything else is converted to JPEG in
    ``work_dir``. Returns ``""`` when there is no usable cover.
    """
    if not raw_cover or not os.path.exists(raw_cover):
        return ""
    try:
        if os.path.splitext(raw_cover)[1].lower() in (".jpg", ".jpeg", ".png"):
            return raw_cover
        from PIL import Image
        img = Image.open(raw_cover)
        if img.mode in ("RGBA", "P", "LA"):
            img = img.convert("RGB")
        os.makedirs(work_dir, exist_ok=True)
        conv_path = os.path.join(work_dir, "cover_converted.jpg")
        img.save(conv_path, format="JPEG", quality=95)
        return conv_path
    except Exception as exc:
        logger.warning("Cover image conversion failed (%s). Using original.", exc)
        return raw_cover


def _get_cover_flags(fmt: str, include_cover: bool) -> list[str]:
    """Returns the FFmpeg stream-mapping flags that attach a cover image (input 1)."""
    if not include_cover:
        return []
    f = (fmt or "").lower()
    if f == "mp3":
        return ["-map", "0:a", "-map", "1:v", "-c:v", "copy", "-disposition:v", "attached_pic", "-id3v2_version", "3"]
    if f in ("m4b", "m4a", "mp4", "flac"):
        return ["-map", "0:a", "-map", "1:v", "-c:v", "copy", "-disposition:v", "attached_pic"]
    return ["-map", "0:a", "-map", "1:v", "-c:v", "copy"]


def _encode_chapter(
    master_wav: str,
    out_path: str,
    config: AudiobookConfig,
    chapter: ExtractedChapter,
    idx: int,
    valid_cover: str,
    post_speed: float,
    normalized: bool,
    log: Callable[[str], None],
) -> None:
    """Encodes a mastered chapter WAV to the requested output format.

    The WAV is already loudness-normalised by the chapter pipeline, so this
    step only converts: sample rate, channels, codec, tags, cover and, when
    the TTS engine cannot change speed itself, a pitch-preserving
    time-stretch. Loudness is applied here only if mastering could not do it.

    Raises:
        RuntimeError: If FFmpeg fails.
    """
    fmt = config.output_format
    native_rate = sf.info(master_wav).samplerate
    channels = int(getattr(config, "channels", 1) or 1)
    needs_convert = native_rate != int(config.sample_rate) or channels != 1
    has_cover = bool(valid_cover and os.path.exists(valid_cover))

    filters: list[str] = []
    if abs(post_speed - 1.0) > 1e-3:
        filters.append(f"atempo={post_speed:.4f}")
    if not normalized:
        filters.append(f"loudnorm=I={config.lufs}:TP={config.true_peak}:LRA=11")

    if not has_cover and not needs_convert and not filters:
        if fmt == "wav":
            shutil.copyfile(master_wav, out_path)
            return
        if fmt == "mp3" and _check_rust():
            import audiobook_rust as _audiobook_rust  # fresh local import
            try:
                # Loudness is re-measured here but the gain comes out at ~0 dB:
                # the input is already at target.
                _audiobook_rust.master_audio(
                    [master_wav], out_path, 0.0, int(native_rate),
                    float(config.lufs), float(config.true_peak),
                    int(getattr(config, "bitrate_kbps", 64)),
                )
                try:
                    _tag_mp3(out_path, chapter, config, idx)
                except Exception as tag_err:
                    log(f"  [Ch{idx}] ⚠ Could not write MP3 tags: {tag_err}")
                return
            except Exception as rust_err:
                log(f"  [Ch{idx}] ⚠ Rust MP3 encode failed ({rust_err}). Falling back to FFmpeg.")

    def _build_cmd(include_cover: bool) -> list[str]:
        cmd = ["ffmpeg", "-y", "-i", master_wav]
        if include_cover:
            cmd += ["-i", valid_cover]
        if filters:
            cmd += ["-af", ",".join(filters)]
        cmd += _encoder_args(config)
        cmd += _get_cover_flags(fmt, include_cover)
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
        subprocess.run(_build_cmd(has_cover), check=True, capture_output=True)
    except subprocess.CalledProcessError as e:
        stderr_log = e.stderr.decode("utf-8", errors="replace") if e.stderr else str(e)
        if not has_cover:
            raise RuntimeError(f"FFmpeg encoding failed: {stderr_log[-600:]}")
        log(f"  [Ch{idx}] ⚠ Cover embedding failed ({stderr_log[:200]}). Retrying without cover image...")
        try:
            subprocess.run(_build_cmd(False), check=True, capture_output=True)
        except subprocess.CalledProcessError as e2:
            stderr_log2 = e2.stderr.decode("utf-8", errors="replace") if e2.stderr else str(e2)
            raise RuntimeError(f"FFmpeg encoding failed: {stderr_log2[-600:]}")


def _prepare_speech_text(text: str, config: AudiobookConfig) -> str:
    """Applies the user's pronunciation fixes, then written-form → spoken-form rules."""
    if config.pronunciation_map:
        text = _apply_pronunciation(text, config.pronunciation_map)
    if getattr(config, "normalize_speech_text", True):
        try:
            from audiobook_factory.speech_text import normalize_for_speech
            text = normalize_for_speech(text, config.language)
        except ImportError:
            pass
        except Exception as exc:
            logger.warning("Speech text normalisation failed (%s); using the text as extracted.", exc)
    return text


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
    fresh: bool = False,
    verifier: Any = None,
    post_speed: float = 1.0,
) -> str | None:
    """Generate audio for one chapter. Returns output file path.

    Args:
        config: AudiobookConfig of the run.
        chapter: The chapter to narrate.
        idx: Chapter number (file name, tags, progress entry, chunk cache).
        total: Number of chapters in the run, for log messages.
        log: Log callback.
        cancel: Cancellation token.
        provider: Unused; kept for call compatibility.
        pool: ProviderPool supplying one TTS provider per device.
        prog_cb: Receives this chapter's progress as a 0–1 fraction.
        pinned_device: Restrict synthesis to one device (chapter-parallel mode).
        subtitle_futures: Shared list collecting async subtitle jobs.
        subtitle_futures_lock: Lock guarding ``subtitle_futures``.
        completed_chunks: Unused; the chunk cache on disk is authoritative.
        fresh: Discard cached chunks and synthesize the whole chapter.
        verifier: Optional ChunkVerifier.
        post_speed: Time-stretch applied while encoding (1.0 = none).
    """
    from audiobook_factory.chapter_pipeline import run_chapter_pipeline
    from audiobook_factory.chunk_planner import plan_chunks

    if pool is None:
        raise RuntimeError("A TTS provider pool is required to synthesize a chapter.")

    temp_dir = os.path.join(config.output_dir, ".temp_chunks", f"abm_ch{idx:03d}")
    os.makedirs(temp_dir, exist_ok=True)
    chunk_prefix = f"chunk_ch_{idx}_"
    resume = getattr(config, "resume_incomplete_chunks", True) and not config.force_reprocess and not fresh

    def _clear_chunk_cache() -> int:
        removed = 0
        try:
            for f_name in os.listdir(temp_dir):
                if f_name.startswith(chunk_prefix) and f_name.endswith((".wav", ".part")):
                    os.remove(os.path.join(temp_dir, f_name))
                    removed += 1
        except OSError as exc:
            logger.warning("Could not clear chunk files in %s: %s", temp_dir, exc)
        return removed

    if not resume:
        _clear_chunk_cache()

    prog_path_out = os.path.join(config.output_dir, "generation_progress.json")
    flagged: list[dict] = []
    flagged_lock = threading.Lock()

    # Both are read in the `finally` below, so they must exist even when the
    # body raises or returns before reaching the subtitle / encode stages.
    sub_future: concurrent.futures.Future | None = None
    chapter_done = False

    try:
        # ── Pronunciation fixes + written-form → spoken-form ──────────────────
        text = _prepare_speech_text(chapter.text or "", config)
        sentences = None
        if not text.strip() and chapter.sentences:
            sentences = [_prepare_speech_text(s, config) for s in chapter.sentences if s and s.strip()]

        # ── Export text if requested ──────────────────────────────────────────
        if config.export_text:
            txt_name = make_safe_filename(chapter.title, idx, config.output_dir, ".txt")
            txt_path = os.path.join(config.output_dir, txt_name)
            try:
                with open(txt_path, "w", encoding="utf-8") as fh:
                    fh.write(f"{chapter.title}\n{'─' * 60}\n\n{text or ' '.join(sentences or [])}")
                log(f"  [Ch{idx}] Text exported → {txt_name}")
            except OSError as e:
                log(f"  [Ch{idx}] Text export failed: {e}")

        # ── Plan the TTS chunks ───────────────────────────────────────────────
        chunks = plan_chunks(
            text, sentences, config.max_len,
            pause=float(config.pause),
            para_pause=float(getattr(config, "para_pause", config.pause)),
            pack_sentences=bool(getattr(config, "pack_sentences", True)),
        )
        if not chunks:
            log(f"  [Ch{idx}] No text to synthesise — skipping.")
            return None

        tts_jobs = [c.text for c in chunks]
        log(f"  [Ch{idx}] {len(tts_jobs)} TTS chunks…")

        # ── Validate the chunk cache against what is about to be synthesized ──
        cache_key = _chunk_cache_fingerprint(config, tts_jobs)
        cache_key_path = os.path.join(temp_dir, _CHUNK_CACHE_KEY_FILE)
        stored_key = ""
        try:
            with open(cache_key_path, encoding="utf-8") as kf:
                stored_key = kf.read().strip()
        except OSError:
            pass
        if stored_key != cache_key:
            removed = _clear_chunk_cache()
            if removed and resume:
                log(
                    f"  [Ch{idx}] Cached chunks were made from different text, voice or "
                    f"TTS settings — discarding {removed} and re-synthesizing."
                )
            try:
                with open(cache_key_path, "w", encoding="utf-8") as kf:
                    kf.write(cache_key)
            except OSError as exc:
                logger.warning("[Ch%d] Could not write chunk cache key: %s", idx, exc)

        # ── Synthesis ─────────────────────────────────────────────────────────
        voice_bytes = b""
        if config.voice_file and os.path.exists(config.voice_file):
            try:
                with open(config.voice_file, "rb") as vf:
                    voice_bytes = vf.read()
            except Exception as exc:
                log(f"  [Ch{idx}] Warning: Could not read voice_file bytes: {exc}")

        def _chunks_done(indices: list[int]) -> None:
            update_chapter_chunks(prog_path_out, idx, indices)

        def _chunk_flagged(chunk_index: int, reason: str) -> None:
            with flagged_lock:
                flagged.append({"chunk": chunk_index, "reason": reason, "text": tts_jobs[chunk_index][:120]})
            log(f"  [Ch{idx}] ⚠ Chunk {chunk_index} kept despite failing verification: {reason}")

        chapter_wav_path = os.path.join(temp_dir, "chapter_mastered.wav")
        master_info: dict = {}
        chunk_durations = run_chapter_pipeline(
            sentences=tts_jobs,
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
            completed_chunks=None if resume else [],
            chunk_pauses=[c.pause_after for c in chunks],
            verifier=verifier,
            chunks_completed_cb=_chunks_done,
            chunk_flagged_cb=_chunk_flagged,
            master_info=master_info,
        )

        if cancel.is_cancelled:
            return None
        if not os.path.exists(chapter_wav_path):
            log(f"  [Ch{idx}] ❌ No audio was produced. Skipping.")
            return None

        # ── Generate Subtitles Asynchronously ─────────────────────────────────
        if config.export_lrc or config.export_srt or config.export_vtt:
            if subtitle_futures is not None and subtitle_futures_lock is not None:
                sub_future = _subtitle_executor.submit(
                    _generate_subtitles,
                    config, chapter, idx, chunks, chunk_durations, log, post_speed,
                )
                with subtitle_futures_lock:
                    subtitle_futures.append((idx, sub_future))
            else:
                _generate_subtitles(config, chapter, idx, chunks, chunk_durations, log, post_speed)

        # ── Encode ────────────────────────────────────────────────────────────
        safe_name  = make_safe_filename(chapter.title, idx, config.output_dir,
                                        f".{config.output_format}")
        out_path   = os.path.join(config.output_dir, safe_name)
        _encode_chapter(
            chapter_wav_path, out_path, config, chapter, idx,
            _ensure_valid_cover_image(config.cover_image, temp_dir),
            post_speed, bool(master_info.get("normalized", True)), log,
        )

        audio_seconds = (sum(chunk_durations) + sum(c.pause_after for c in chunks)) / (post_speed or 1.0)
        update_chapter_fields(prog_path_out, idx, {
            "duration": round(audio_seconds, 2),
            "flagged_chunks": sorted(flagged, key=lambda item: item["chunk"]),
        })
        log(f"  [Ch{idx}] 🎧 {_format_hms(audio_seconds)} of audio.")

        chapter_done = True
        return out_path

    finally:
        if sub_future is not None:
            try:
                sub_future.result(timeout=30.0)
            except Exception as exc:
                logger.warning("[Ch%d] Subtitle future failed: %s", idx, exc)
        # An unfinished chapter keeps its chunk WAVs so the next attempt or
        # run resumes from them instead of re-synthesizing the whole chapter.
        keep_chunks = (
            not chapter_done
            and not config.force_reprocess
            and getattr(config, "resume_incomplete_chunks", True)
        )
        _prune_chapter_temp_dir(temp_dir, idx, keep_chunks)


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


# ── Voice Studio preview provider cache ─────────────────────────────────────
# preview_tts() is called on every "Preview voice" click in the Voice Studio
# tab and is separate from the GPUPoolManager pool used by run_pipeline().
# Previously it called get_tts_provider() fresh on every single call and
# never released the result, so each preview click (and every provider
# switch in the dropdown) permanently loaded another full model onto the
# GPU. Repeated previews, or previewing more than one TTS engine in the
# same session, eventually exhausted VRAM. Cache one instance and clean up
# the old one whenever the requested provider changes.
_preview_provider_cache: dict[str, Any] = {"name": None, "provider": None, "key": None}
_preview_cache_lock = threading.Lock()


def _cleanup_preview_provider() -> None:
    """Releases the cached Voice Studio preview provider, if any."""
    with _preview_cache_lock:
        provider = _preview_provider_cache.get("provider")
        if provider is not None:
            try:
                provider.cleanup()
            except Exception as exc:
                logger.warning("[preview_tts] Error cleaning up cached preview provider: %s", exc)
        _preview_provider_cache["provider"] = None
        _preview_provider_cache["name"] = None
        _preview_provider_cache["key"] = None


atexit.register(_cleanup_preview_provider)


def preview_tts(text: str, config: AudiobookConfig) -> bytes | None:
    """
    Generate a short TTS preview and return raw WAV bytes.
    Used by the Voice Studio tab.

    Reuses a single cached provider instance across calls instead of
    loading a fresh model per click; automatically frees the previous
    provider's GPU memory when the requested provider name or variant changes.
    """
    from audiobook_factory.tts_providers import get_tts_provider

    if not text.strip():
        return None

    with tempfile.TemporaryDirectory(dir=str(_TEMP_DIR)) as tmp:
        out_path = os.path.join(tmp, "preview.wav")
        try:
            with _preview_cache_lock:
                provider = _preview_provider_cache.get("provider")
                cached_name = _preview_provider_cache.get("name")
                cached_key = _preview_provider_cache.get("key")
                current_key = (
                    config.tts_provider_name,
                    getattr(config, "tts_model_name", getattr(config, "tts_model_variant", "")),
                    getattr(config, "quantization", ""),
                )
                if provider is None or cached_key != current_key or cached_name != config.tts_provider_name:
                    if provider is not None:
                        logger.info(
                            "[preview_tts] Switching preview provider %s -> %s: freeing previous model.",
                            cached_name, config.tts_provider_name,
                        )
                        try:
                            provider.cleanup()
                        except Exception as exc:
                            logger.warning("[preview_tts] Error cleaning up previous preview provider: %s", exc)
                    provider = get_tts_provider(config.tts_provider_name, config)
                    _preview_provider_cache["provider"] = provider
                    _preview_provider_cache["name"] = config.tts_provider_name
                    _preview_provider_cache["key"] = current_key
                else:
                    # The cached instance still holds the config it was built
                    # with; without this, edited sampling settings, language,
                    # instruct text or transcript are ignored until the model
                    # itself changes.
                    provider.config = config
            provider.synthesize(text.strip(), config.voice_file, out_path)
            if os.path.exists(out_path):
                with open(out_path, "rb") as f:
                    return f.read()
        except Exception as e:
            logger.warning("[preview_tts] Error: %s", e)
            # The cached provider may be in a bad state (e.g. OOM mid-load) —
            # drop it so the next preview call starts clean instead of
            # repeatedly failing against a half-initialised model.
            _cleanup_preview_provider()
            raise e
    return None
