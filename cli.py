"""
cli.py — AudiobookMaker Headless CLI
======================================
Generate audiobooks without the Gradio web interface: from a
generation_progress.json (exported by the UI or written by an earlier run),
or straight from a book file.

Usage examples
--------------
# Resume / run a config exported by the UI (chapter text is cached in it):
    python cli.py audiobook_output/MyBook/generation_progress.json

# Straight from a book file, no JSON needed:
    python cli.py --book book.epub --voice-file narrator.wav
    python cli.py --book book.epub --voice-file narrator.wav \\
        --provider qwen --language English --output-format m4b --single-file

# See what would happen (config, engine, chapters, audio length) — loads no model:
    python cli.py generation_progress.json --dry-run

# List the TTS engines, their licences, options and install commands:
    python cli.py --list-providers

# A JSON exported on another machine: point it at this machine's files:
    python cli.py generation_progress.json \\
        --voice-file /path/to/narrator.wav --book-path /path/to/book.epub

# Override settings saved in the JSON:
    python cli.py generation_progress.json \\
        --speed 1.1 --pause 0.4 --para-pause 1.0 --bitrate 96 --verify asr \\
        --tts-option num_step=16 --tts-option denoise=false

# Only some chapters; regenerate chapters 3 and 7-9 even though they are done:
    python cli.py generation_progress.json --chapters 1-12
    python cli.py generation_progress.json --redo 3,7-9

# Regenerate everything, ignoring saved progress:
    python cli.py generation_progress.json --force-reprocess

# Embed a cover into the audio files already in the output directory:
    python cli.py generation_progress.json --embed-cover-only --cover-image cover.jpg

Exit status
-----------
    0    every requested chapter was generated (or was already complete)
    1    a fatal error, or at least one chapter failed
    2    invalid command line
    130  cancelled (Ctrl+C; a second Ctrl+C exits immediately)

Environment
-----------
    ABM_API_URL             API server to dispatch to (default http://127.0.0.1:8000)
    ABM_API_SECRET          sent as the x-api-key header when the server requires a key
    ABM_SKIP_INSTALL_CHECK  set to 1 to skip the "is the engine installed" check
"""
from __future__ import annotations

import argparse
import base64
import dataclasses
import logging
import os
import queue
import re
import shlex
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import traceback
from typing import Any, Callable, Iterable

# ── Ensure project root is on sys.path ───────────────────────────────────────
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from audiobook_factory.pipeline import AudiobookConfig
from audiobook_factory.progress_io import read_progress_file
from audiobook_factory.text_extractor import ExtractedChapter
from audiobook_factory.utils import decode_done_message, encode_done_message


# ══════════════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════════════

_EXIT_OK: int = 0
_EXIT_FAILED: int = 1
_EXIT_CANCELLED: int = 130

_PROGRESS_FILE_NAME: str = "generation_progress.json"
_DEFAULT_OUTPUT_ROOT: str = os.path.join(_ROOT, "audiobook_output")
# Only these names, and only next to the output, the JSON and the book.
_COVER_FILE_NAMES: tuple[str, ...] = ("cover.jpg", "cover.jpeg", "cover.png", "cover.webp")
_COMPLETED_STATUSES: tuple[str, ...] = ("completed", "complete")

_DEFAULT_API_URL: str = "http://127.0.0.1:8000"
_API_HEALTH_ATTEMPTS: int = 3
_API_REQUEST_TIMEOUT_SEC: float = 60.0
_API_POLL_SEC: float = 2.0
# Consecutive failed polls before the run is declared lost (about a minute).
_API_MAX_POLL_FAILURES: int = 30
# How long to keep following a task after asking the server to cancel it.
_API_CANCEL_WAIT_SEC: float = 30.0

# Narration speed used for the --dry-run estimate: about 150 words per minute
# of English is 15 characters per second; CJK text is read character by character.
_DRY_RUN_CHARS_PER_SECOND: float = 15.0
_DRY_RUN_CJK_SECONDS_PER_CHAR: float = 0.22
_DRY_RUN_VALUE_WIDTH: int = 110
# Non-interactive output (notebooks, logs) gets a progress line per this many percent.
_PROGRESS_STEP_PERCENT: float = 5.0
_SUMMARY_MAX_FLAGGED: int = 5

_TRUE_WORDS: frozenset[str] = frozenset({"1", "true", "yes", "on", "y"})
_FALSE_WORDS: frozenset[str] = frozenset({"0", "false", "no", "off", "n", ""})

# Settings that are a one-off instruction, not a property of the book. They
# are honoured from the command line only: a JSON that carries them (the
# pipeline writes the last run's settings back) would otherwise wipe or redo
# finished chapters on every resume.
_ONE_SHOT_SETTINGS: tuple[str, ...] = ("force_reprocess", "redo_chapters")

# argparse destination → AudiobookConfig field, for flags that simply replace a value.
_VALUE_FLAGS: tuple[tuple[str, str], ...] = (
    ("output_format", "output_format"),
    ("worker_count", "worker_count"),
    ("device", "device"),
    ("tts_model_name", "tts_model_name"),
    ("quantization", "quantization"),
    ("language", "language"),
    ("voice_transcript", "voice_transcript"),
    ("speed", "speed"),
    ("temperature", "temperature"),
    ("top_p", "top_p"),
    ("seed", "seed"),
    ("bitrate", "bitrate_kbps"),
    ("sample_rate", "sample_rate"),
    ("channels", "channels"),
    ("pause", "pause"),
    ("para_pause", "para_pause"),
    ("max_len", "max_len"),
    ("batch_size", "batch_size"),
    ("gpu_count", "gpu_count"),
    ("verify", "verify_chunks"),
)
# argparse destination → (AudiobookConfig field, value set when the flag is present).
_SWITCH_FLAGS: tuple[tuple[str, str, bool], ...] = (
    ("force_reprocess", "force_reprocess", True),
    ("no_resume_chunks", "resume_incomplete_chunks", False),
    ("no_pack_sentences", "pack_sentences", False),
    ("no_normalize_text", "normalize_speech_text", False),
    ("single_file", "single_file_mode", True),
)

# ── Settings allowlists for progress JSON (BUG-R4-C1-A4-H1) ─────────────────
_ALLOWED_QWEN_MODELS: set[str] = {
    "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
    "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
}
_SCALAR_TYPES: tuple[type, ...] = (str, int, float, bool, type(None))


class _CliError(Exception):
    """A problem to report to the user in one message, without a traceback."""

    def __init__(self, message: str, exit_code: int = _EXIT_FAILED) -> None:
        super().__init__(message)
        self.exit_code = exit_code


class _UntrustedOutputDir(ValueError):
    """The progress JSON names an output_dir outside the allowed directories."""


class _ApiDeclined(Exception):
    """The API server did not take the task; nothing was enqueued."""


class _ApiTaskFailed(Exception):
    """A task accepted by the API server ended in the failed state."""


class _ApiError(Exception):
    """The API server rejected the request, or contact with it was lost."""


@dataclasses.dataclass
class _RunState:
    """Result of the generation thread, read by the main thread when it ends."""

    files: list[str] = dataclasses.field(default_factory=list)
    error: BaseException | None = None
    error_trace: str = ""


# ══════════════════════════════════════════════════════════════════════════════
# Console helpers
# ══════════════════════════════════════════════════════════════════════════════

_BOLD  = "\033[1m"
_GREEN = "\033[92m"
_CYAN  = "\033[96m"
_YELL  = "\033[93m"
_RED   = "\033[91m"
_RESET = "\033[0m"

def _h(text): return f"{_BOLD}{text}{_RESET}"
def _ok(text): return f"{_GREEN}{text}{_RESET}"
def _info(text): return f"{_CYAN}{text}{_RESET}"
def _warn(text): return f"{_YELL}{text}{_RESET}"
def _err(text): return f"{_RED}{text}{_RESET}"
def _print_banner():
    print()
    print(_h("━" * 50))
    print(_h("  📖  AudiobookMaker CLI"))
    print(_h("━" * 50))

_IS_TTY: bool = sys.stdout.isatty()


def _display_progress(
    chapter_num: int,
    total_chapters: int,
    chunk_num: int,
    total_chunks: int,
    chapter_title: str,
) -> None:
    """Displays progress in TTY mode (overwriting line) or notebook mode (new line)."""
    pct = (chunk_num / total_chunks) * 100 if total_chunks > 0 else 0
    title_display = chapter_title
    if len(title_display) > 35:
        title_display = title_display[:32] + "..."
    line = (
        f"[{chapter_num}/{total_chapters}] {title_display} "
        f"— chunk {chunk_num}/{total_chunks} ({pct:.1f}%)"
    )
    if _IS_TTY:
        sys.stdout.write(f"\r{line}   ")
        sys.stdout.flush()
    else:
        if chunk_num == total_chunks:
            print(f"[DONE] {line}")


def _format_hms(seconds: float) -> str:
    """Formats a duration as H:MM:SS."""
    seconds = int(max(0, seconds))
    return f"{seconds // 3600}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


# ══════════════════════════════════════════════════════════════════════════════
# Cancellation (Ctrl+C)
# ══════════════════════════════════════════════════════════════════════════════

def _hard_exit(code: int) -> None:
    """Ends the process at once, without waiting for worker threads."""
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass
    os._exit(code)


def _make_sigint_handler(
    cancel: Any,
    hard_exit: Callable[[int], None] = _hard_exit,
) -> Callable[[int, object], None]:
    """Builds the Ctrl+C handler for a run that owns the token *cancel*.

    The first interrupt asks the pipeline to stop after the batch in flight
    (finished chapters and chunks stay on disk and are resumed next time).
    A second one exits with status 130 immediately — for when the cooperative
    stop is itself stuck, for example in a model download.
    """
    def _handler(signum: int, frame: object) -> None:
        if cancel.is_cancelled:
            print()
            print(_err("Second interrupt — exiting now."))
            hard_exit(_EXIT_CANCELLED)
            return
        cancel.cancel()
        print()
        print(_warn(
            "Interrupt received — stopping after the current batch. "
            "Press Ctrl+C again to exit immediately."
        ))

    return _handler


def _install_sigint_handler(cancel: Any) -> Any:
    """Installs the Ctrl+C handler; returns the previous one (None off the main thread)."""
    if threading.current_thread() is not threading.main_thread():
        return None
    return signal.signal(signal.SIGINT, _make_sigint_handler(cancel))


def _restore_sigint_handler(previous: Any) -> None:
    """Puts back the handler returned by :func:`_install_sigint_handler`."""
    if previous is not None and threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGINT, previous)


# ══════════════════════════════════════════════════════════════════════════════
# FastAPI backend client
# ══════════════════════════════════════════════════════════════════════════════

def _api_base_url() -> str:
    """Base URL of the API server (``ABM_API_URL`` or the local default)."""
    return (os.environ.get("ABM_API_URL") or _DEFAULT_API_URL).rstrip("/")


def _api_headers() -> dict[str, str]:
    """Auth header for the API server when ``ABM_API_SECRET`` is set."""
    secret = os.environ.get("ABM_API_SECRET", "")
    return {"x-api-key": secret} if secret else {}


def _is_api_healthy() -> bool:
    import requests
    url = f"{_api_base_url()}/api/v1/health"
    for attempt in range(_API_HEALTH_ATTEMPTS):
        try:
            r = requests.get(url, timeout=4.0)
            if r.status_code == 200 and r.json().get("status") == "ok":
                return True
        except Exception:
            pass
        if attempt + 1 < _API_HEALTH_ATTEMPTS:
            time.sleep(0.5)
    return False


def _api_error_detail(response: Any) -> tuple[str, str, dict]:
    """Returns ``(code, message, detail)`` from an API error response."""
    try:
        detail = response.json().get("detail")
    except Exception:
        return "", (getattr(response, "text", "") or "")[:300], {}
    if isinstance(detail, dict):
        return str(detail.get("code") or ""), str(detail.get("message") or detail), detail
    return "", str(detail or ""), {}


def _run_via_api(
    cfg: AudiobookConfig,
    chapters: list[ExtractedChapter],
    log_q: queue.Queue,
    prog_q: queue.Queue,
    cancel: Any,
) -> list[str]:
    """Runs the job on the API server, relaying its log lines and progress.

    Returns the output files of the completed task (empty when cancelled).

    Raises
    ------
    _ApiDeclined
        The server did not accept the task — the caller may run it locally.
    _ApiTaskFailed
        The task was accepted and failed; the message is the server's
        ``error_message``.
    _ApiError
        The server rejected the request as invalid, or contact with it was
        lost after the task was enqueued. The task is not re-run locally:
        it may still be running there.
    """
    import requests

    base = _api_base_url()
    headers = _api_headers()
    payload = {
        "config": dataclasses.asdict(cfg),
        "chapters": [
            {"num": ch.num, "title": ch.title, "text": ch.text, "sentences": ch.sentences}
            for ch in chapters
        ],
    }
    try:
        r = requests.post(
            f"{base}/api/v1/generate", json=payload, headers=headers,
            timeout=_API_REQUEST_TIMEOUT_SEC,
        )
    except requests.RequestException as exc:
        raise _ApiDeclined(f"the API server could not be reached ({exc})") from exc

    if r.status_code == 401:
        raise _ApiDeclined(
            "the API server requires an API key — set ABM_API_SECRET to the server's secret"
        )
    if r.status_code == 429:
        raise _ApiDeclined("the API server is rate-limiting requests")
    if r.status_code >= 400:
        code, message, detail = _api_error_detail(r)
        if code == "output_dir_outside_base":
            raise _ApiDeclined(
                f"the output directory {cfg.output_dir} is outside the API server's "
                f"output base ({detail.get('output_base', 'audiobook_output')})"
            )
        raise _ApiError(f"The API server rejected the request (HTTP {r.status_code}): {message}")

    task_id = r.json()["task_id"]
    cancel.task_id = task_id
    log_q.put(f"✅ Enqueued. Task ID: {task_id}")

    total = max(1, len(chapters))
    seen = 0
    failures = 0
    cancel_deadline: float | None = None
    while True:
        if cancel.is_cancelled and cancel_deadline is None:
            cancel_deadline = time.monotonic() + _API_CANCEL_WAIT_SEC
            try:
                requests.post(f"{base}/api/v1/tasks/{task_id}/cancel", headers=headers, timeout=5)
            except Exception as exc:
                log_q.put(f"⚠️ Could not send the cancel request to the API server: {exc}")

        try:
            poll = requests.get(
                f"{base}/api/v1/tasks/{task_id}", params={"since": seen},
                headers=headers, timeout=10,
            )
            if poll.status_code == 404:
                raise _ApiError(
                    f"The API server no longer knows task {task_id} (was it restarted?)."
                )
            poll.raise_for_status()
            st = poll.json()
            failures = 0
        except _ApiError:
            raise
        except Exception as exc:
            failures += 1
            if failures >= _API_MAX_POLL_FAILURES:
                raise _ApiError(
                    f"Lost contact with the API server ({exc}). "
                    f"Task {task_id} may still be running there."
                ) from exc
            if cancel_deadline is not None and time.monotonic() > cancel_deadline:
                return []
            time.sleep(_API_POLL_SEC)
            continue

        # The server returns only the lines after `since`; a server without
        # that parameter returns the whole list every time.
        logs = st.get("logs") or []
        if "log_count" in st:
            new_lines = logs
            seen = int(st["log_count"])
        else:
            new_lines = logs[seen:]
            seen = len(logs)
        for line in new_lines:
            log_q.put(line)
        if st.get("progress"):
            prog_q.put((float(st["progress"]) * total, float(total)))

        status = st.get("status")
        if status == "completed":
            return [f for f in (st.get("output_files") or []) if isinstance(f, str)]
        if status == "failed":
            raise _ApiTaskFailed(st.get("error_message") or "the server gave no error message")
        if status == "cancelled":
            cancel.cancel()
            return []
        if cancel_deadline is not None and time.monotonic() > cancel_deadline:
            return []
        time.sleep(_API_POLL_SEC)


def _run_local(
    cfg: AudiobookConfig,
    chapters: list[ExtractedChapter],
    log_q: queue.Queue,
    prog_q: queue.Queue,
    cancel: Any,
) -> list[str]:
    """Runs the pipeline in this process."""
    from audiobook_factory.pipeline import run_pipeline
    return run_pipeline(cfg, chapters, log_q, prog_q, cancel)


# ══════════════════════════════════════════════════════════════════════════════
# Argument parsing
# ══════════════════════════════════════════════════════════════════════════════

def _parse_number_list(text: str) -> list[int]:
    """Parses ``"1-3,7,10-12"`` into ``[1, 2, 3, 7, 10, 11, 12]``."""
    numbers: list[int] = []
    for part in str(text).replace(" ", "").split(","):
        if not part:
            continue
        match = re.fullmatch(r"(\d+)(?:-(\d+))?", part)
        if not match:
            raise argparse.ArgumentTypeError(
                f"invalid chapter list {text!r}: use numbers and ranges such as 3 or 1-5,8,12-14"
            )
        first = int(match.group(1))
        last = int(match.group(2) or first)
        if first < 1 or last < first:
            raise argparse.ArgumentTypeError(f"invalid chapter range {part!r}")
        numbers.extend(range(first, last + 1))
    if not numbers:
        raise argparse.ArgumentTypeError("the chapter list is empty")
    return sorted(set(numbers))


def _output_format_choices() -> list[str]:
    """Output formats the pipeline accepts."""
    from audiobook_factory.pipeline import _VALID_OUTPUT_FORMATS
    return list(_VALID_OUTPUT_FORMATS)


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cli.py",
        description="AudiobookMaker — headless audiobook generation from a progress JSON or a book file.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    src = p.add_argument_group("input")
    src.add_argument(
        "config_json",
        nargs="?",
        default=None,
        help="Path to generation_progress.json (exported by the Gradio UI or written by an "
             "earlier run). Omit it when using --book.",
    )
    src.add_argument(
        "--book",
        metavar="PATH",
        default=None,
        help="Generate straight from a book file (EPUB, PDF, TXT, DOCX, ODT) with no JSON: "
             "chapters are extracted, settings are the defaults plus the flags given here, and "
             "output goes to audiobook_output/<title> unless --output-dir is set. "
             "Given together with a JSON it means --book-path.",
    )
    src.add_argument(
        "--book-path",
        metavar="PATH",
        default=None,
        help="Book file to use with a JSON: needed when the JSON has no cached chapter text, "
             "and used to extract the cover image.",
    )
    src.add_argument(
        "--chapters",
        metavar="N-M[,…]",
        type=_parse_number_list,
        default=None,
        help="Only run these chapter numbers, e.g. 1-12 or 3,7,20-25. "
             "Replaces any chapter selection saved in the JSON.",
    )

    out = p.add_argument_group("output")
    out.add_argument(
        "--output-dir",
        metavar="DIR",
        default=None,
        help="Directory for the generated audio files and the progress file.",
    )
    out.add_argument(
        "--output-format",
        choices=_output_format_choices(),
        default=None,
        help="Output audio format.",
    )
    out.add_argument(
        "--single-file",
        action="store_true",
        default=False,
        help="Combine all chapters into one file with chapter markers.",
    )
    out.add_argument("--bitrate", type=int, default=None, metavar="KBPS",
                     help="Encoder bitrate in kbps (lossy formats).")
    out.add_argument("--sample-rate", type=int, default=None, metavar="HZ",
                     help="Output sample rate in Hz.")
    out.add_argument("--channels", type=int, choices=[1, 2], default=None,
                     help="1 = mono, 2 = stereo.")
    out.add_argument(
        "--cover-image",
        metavar="PATH",
        default=None,
        help="Cover image file (PNG, JPG, WEBP). Without it the cover saved in the JSON, a "
             "cover.* file next to the output / JSON / book, or the book's own cover is used.",
    )

    eng = p.add_argument_group("engine")
    eng.add_argument(
        "--provider",
        metavar="NAME",
        default=None,
        help="TTS engine (see --list-providers). Switching engine drops the model name and "
             "engine options saved in the JSON.",
    )
    eng.add_argument(
        "--tts-model-name",
        metavar="MODEL",
        default=None,
        help="Model variant of the engine (e.g. Qwen/Qwen3-TTS-12Hz-1.7B-Base).",
    )
    eng.add_argument(
        "--tts-option",
        metavar="KEY=VALUE",
        action="append",
        default=None,
        help="Engine-specific option; repeatable. Keys, types and defaults are listed by "
             "--list-providers. The value is converted to the option's declared type.",
    )
    eng.add_argument("--language", metavar="NAME", default=None,
                     help="Language of the text, e.g. English.")
    eng.add_argument("--temperature", type=float, default=None, help="Sampling temperature.")
    eng.add_argument("--top-p", type=float, default=None, help="Nucleus sampling threshold.")
    eng.add_argument("--seed", type=int, default=None, help="Random seed (-1 = random).")
    eng.add_argument(
        "--device",
        default=None,
        choices=["cuda", "cpu"],
        help="Compute device.",
    )
    eng.add_argument("--gpu-count", type=int, default=None, metavar="N",
                     help="Number of GPUs to use (0 = all detected).")
    eng.add_argument("--batch-size", type=int, default=None, metavar="N",
                     help="Text chunks per forward pass (0 = size from free VRAM).")
    eng.add_argument(
        "--worker-count",
        type=int,
        default=None,
        metavar="N",
        help="Number of parallel TTS workers.",
    )
    eng.add_argument(
        "--quantization",
        choices=["none", "int8"],
        default=None,
        help="Model quantization mode. int8 reduces VRAM by ~50%% via bitsandbytes (disables torch.compile).",
    )

    voice = p.add_argument_group("voice")
    voice.add_argument(
        "--voice-file",
        metavar="PATH",
        default=None,
        help="Narrator reference clip to clone.",
    )
    transcript = voice.add_mutually_exclusive_group()
    transcript.add_argument("--voice-transcript", metavar="TEXT", default=None,
                            help="What is said in the reference clip.")
    transcript.add_argument("--voice-transcript-file", metavar="PATH", default=None,
                            help="Text file holding the reference clip's transcript.")
    voice.add_argument("--voice-preset", metavar="PATH", default=None,
                       help="Saved voice preset file of the engine; used instead of --voice-file.")

    nar = p.add_argument_group("narration")
    nar.add_argument("--speed", type=float, default=None, help="Speaking speed (1.0 = normal).")
    nar.add_argument("--pause", type=float, default=None, metavar="SEC",
                     help="Pause between sentences, in seconds.")
    nar.add_argument("--para-pause", type=float, default=None, metavar="SEC",
                     help="Pause between paragraphs, in seconds.")
    nar.add_argument("--max-len", type=int, default=None, metavar="CHARS",
                     help="Maximum characters per TTS chunk.")
    nar.add_argument("--no-pack-sentences", action="store_true", default=False,
                     help="Synthesize sentence by sentence instead of packing a paragraph's "
                          "sentences into one call.")
    nar.add_argument("--no-normalize-text", action="store_true", default=False,
                     help="Do not rewrite numerals, currency, dates and abbreviations into speakable form.")
    nar.add_argument("--verify", choices=["off", "duration", "asr"], default=None,
                     help="Check every chunk and re-synthesize failures: by duration/silence "
                          "(free) or by transcribing it with Whisper (asr).")

    run = p.add_argument_group("run control")
    run.add_argument(
        "--redo",
        metavar="N[,N…]",
        type=_parse_number_list,
        default=None,
        help="Regenerate these chapter numbers even if they are complete, e.g. 3 or 3,7-9.",
    )
    run.add_argument(
        "--force-reprocess",
        action="store_true",
        default=False,
        help="Re-generate all chapters, ignoring any saved progress.",
    )
    run.add_argument(
        "--no-resume-chunks",
        action="store_true",
        default=False,
        help="Disable chunk-level resume. Re-synthesize all chunks from scratch even if partial progress exists.",
    )
    run.add_argument(
        "--dry-run",
        action="store_true",
        default=False,
        help="Print the resolved configuration, the engine and its licence, the chapter list "
             "with character counts and an estimate of the audio length, then exit. Loads no model.",
    )
    run.add_argument(
        "--local",
        action="store_true",
        default=False,
        help="Run in this process even when the API server is up.",
    )

    util = p.add_argument_group("utilities")
    util.add_argument(
        "--list-providers",
        action="store_true",
        default=False,
        help="List the TTS engines with licence, default model, VRAM, options and install command, then exit.",
    )
    util.add_argument(
        "--embed-cover-only",
        action="store_true",
        default=False,
        help="Embed the cover image into the audio files already in the output directory, without running TTS.",
    )
    util.add_argument(
        "--clear-voice-cache",
        action="store_true",
        default=False,
        help="Clear all cached voice preprocessing files and exit.",
    )
    return p


# ══════════════════════════════════════════════════════════════════════════════
# Providers
# ══════════════════════════════════════════════════════════════════════════════

def _provider_info(name: str) -> Any:
    """Returns the ProviderInfo of *name* (imports the provider module)."""
    from audiobook_factory.tts_providers.registry import provider_info
    return provider_info(name)


def _canonical_provider(name: Any) -> str:
    """Registry key for *name*, or the cleaned name itself when it is unknown."""
    from audiobook_factory.tts_providers.registry import canonical_name
    try:
        return canonical_name(str(name))
    except ValueError:
        return str(name).lower().strip()


def _requirement_name(spec: str) -> str | None:
    """Distribution name of a pip requirement, or None when it does not apply here.

    Requirements whose environment marker is false on this machine, and
    arguments that are not a named requirement (a bare URL, an option), are
    skipped.
    """
    try:
        from packaging.requirements import InvalidRequirement, Requirement
    except ImportError:
        if ";" in spec:
            return None
        match = re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)\s*(?:$|[\[<>=!~@])", spec)
        return match.group(1) if match else None
    try:
        requirement = Requirement(spec)
    except InvalidRequirement:
        return None
    if requirement.marker is not None and not requirement.marker.evaluate():
        return None
    return requirement.name


def _missing_requirements(requirements: Iterable[str]) -> list[str]:
    """Returns the pip requirements whose distribution is not installed.

    Only presence is checked, not versions: a version mismatch is the
    engine's to report when it loads.
    """
    from importlib import metadata

    missing: list[str] = []
    for spec in requirements:
        name = _requirement_name(spec)
        if not name:
            continue
        try:
            metadata.distribution(name)
        except metadata.PackageNotFoundError:
            missing.append(spec)
        except Exception:
            continue
    return missing


def _install_command(info: Any) -> str:
    """The ``pip install`` line for a provider's own dependencies."""
    requirements = list(getattr(info, "pip_requirements", ()) or ())
    if not requirements:
        return ""
    return "pip install " + " ".join(shlex.quote(r) for r in requirements)


def _install_help(provider: str, info: Any) -> str:
    """Install command and notes for an engine, as indented text."""
    lines: list[str] = []
    command = _install_command(info)
    if command:
        lines.append(f"   Install it with:\n     {command}")
    notes = (getattr(info, "install_notes", "") or "").strip()
    if notes:
        lines.append("   Notes:")
        lines.extend(f"     {line}" for line in notes.splitlines())
    if not lines:
        lines.append(f"   See `python cli.py --list-providers` for engine '{provider}'.")
    return "\n".join(lines)


def _check_provider_installed(provider: str) -> None:
    """Fails with the engine's install command when it cannot run here.

    Raises
    ------
    _CliError
        The provider module cannot be imported, or one of the packages it
        declares in ``ProviderInfo.pip_requirements`` is not installed.
    """
    try:
        info = _provider_info(provider)
    except Exception as exc:
        raise _CliError(
            f"TTS engine '{provider}' is not available in this installation: {exc}\n"
            "   Run `python cli.py --list-providers` to see the engines that are."
        ) from exc
    if os.environ.get("ABM_SKIP_INSTALL_CHECK") == "1":
        return
    missing = _missing_requirements(getattr(info, "pip_requirements", ()) or ())
    if missing:
        raise _CliError(
            f"TTS engine '{info.display_name}' ({provider}) is not installed — "
            f"missing: {', '.join(missing)}\n"
            f"{_install_help(provider, info)}\n"
            "   (Set ABM_SKIP_INSTALL_CHECK=1 to run anyway.)"
        )


def _has_import_error(exc: BaseException | None) -> bool:
    """True when *exc* or anything in its cause chain is an ImportError."""
    seen: set[int] = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if isinstance(exc, ImportError):
            return True
        exc = exc.__cause__ or exc.__context__
    return False


def _print_providers() -> None:
    """Prints every registered TTS engine (the ``--list-providers`` output)."""
    from audiobook_factory.tts_providers.registry import provider_names

    names = provider_names()
    print(_h(f"TTS engines ({len(names)}):"))
    for name in names:
        print()
        try:
            info = _provider_info(name)
        except Exception as exc:
            print(f"{_h(name)} — {_warn('unavailable')}: {exc}")
            continue
        commercial = "commercial use allowed" if info.commercial_use else "NON-COMMERCIAL use only"
        missing = _missing_requirements(info.pip_requirements)
        installed = _ok("installed") if not missing else _warn("not installed (missing: " + ", ".join(missing) + ")")
        extra_models = max(0, len(info.models) - 1)
        languages = str(len(info.languages)) if info.languages else "not restricted"
        print(f"{_h(name)} — {info.display_name}")
        print(f"    Licence:       {info.license} ({commercial})")
        print(f"    Default model: {info.default_model or '(none)'}"
              + (f"  (+{extra_models} more)" if extra_models else ""))
        print(f"    Min VRAM:      {info.min_vram_gb:g} GB")
        print(f"    Languages:     {languages}")
        print(f"    Status:        {installed}")
        print(f"    Install:       {_install_command(info) or '(no extra packages)'}")
        if info.options:
            print("    Options (--tts-option KEY=VALUE):")
            for option in info.options:
                kind = option.kind
                if option.kind == "choice" and option.choices:
                    kind = "choice: " + "|".join(str(c) for c in option.choices)
                elif option.minimum is not None and option.maximum is not None:
                    kind = f"{option.kind} {option.minimum:g}..{option.maximum:g}"
                print(f"      {option.key}={option.default!r}  [{kind}]  {option.label}")
        else:
            print("    Options:       (none)")
    print()


def _coerce_option(option: Any, raw: str) -> Any:
    """Converts a ``--tts-option`` value to the type the provider declared."""
    kind = option.kind
    text = raw.strip()
    try:
        if kind == "float":
            return float(text)
        if kind == "int":
            return int(text)
    except ValueError:
        raise _CliError(f"--tts-option {option.key}: {raw!r} is not a valid {kind}.") from None
    if kind == "bool":
        low = text.lower()
        if low in _TRUE_WORDS:
            return True
        if low in _FALSE_WORDS:
            return False
        raise _CliError(f"--tts-option {option.key}: {raw!r} is not a boolean (use true or false).")
    if kind == "choice":
        choices = [str(c) for c in option.choices]
        if choices and text not in choices:
            raise _CliError(
                f"--tts-option {option.key}: {raw!r} is not one of {', '.join(choices)}."
            )
        return text
    if kind == "file":
        if not text:
            return ""
        path = os.path.abspath(os.path.expanduser(text))
        if not os.path.isfile(path):
            raise _CliError(f"--tts-option {option.key}: file not found: {path}")
        return path
    return raw


def _parse_tts_options(provider: str, pairs: list[str]) -> dict[str, Any]:
    """Turns ``KEY=VALUE`` strings into typed entries for ``config.tts_options``.

    Raises
    ------
    _CliError
        For a malformed pair, a key the provider does not declare (the
        message lists the valid ones) or a value of the wrong type.
    """
    try:
        info = _provider_info(provider)
    except Exception as exc:
        raise _CliError(
            f"--tts-option needs the option list of engine '{provider}', which is not available: {exc}"
        ) from exc
    declared = {option.key: option for option in info.options}
    parsed: dict[str, Any] = {}
    for pair in pairs:
        key, sep, raw = str(pair).partition("=")
        key = key.strip()
        if not sep or not key:
            raise _CliError(f"--tts-option expects KEY=VALUE, got {pair!r}.")
        option = declared.get(key)
        if option is None:
            valid = ", ".join(sorted(declared)) or "(this engine has no options)"
            raise _CliError(
                f"Unknown --tts-option '{key}' for engine '{provider}'. Valid options: {valid}"
            )
        value = _coerce_option(option, raw)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            low, high = option.minimum, option.maximum
            if (low is not None and value < low) or (high is not None and value > high):
                bounds = f"{'' if low is None else format(low, 'g')}..{'' if high is None else format(high, 'g')}"
                print(_warn(f"  ⚠️  --tts-option {key}={value} is outside the usual range {bounds}."))
        parsed[key] = value
    return parsed


# ══════════════════════════════════════════════════════════════════════════════
# Load settings (JSON or book) and merge CLI overrides
# ══════════════════════════════════════════════════════════════════════════════

def _validate_loaded_settings(settings: dict, args: Any, path: str) -> None:
    """Validate untrusted configuration loaded from progress JSON (BUG-R4-C1-A4-H1)."""
    # 1. Provider validation
    from audiobook_factory.tts_providers.registry import canonical_name, is_known_provider, provider_names

    provider = str(settings.get("tts_provider_name", "qwen")).lower().strip()
    if not is_known_provider(provider):
        raise ValueError(
            f"Untrusted or invalid tts_provider_name '{provider}' in progress JSON settings. "
            f"Known engines: {', '.join(provider_names())}."
        )
    provider = canonical_name(provider)

    # 2. Model name validation
    model = str(settings.get("tts_model_name", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")).strip()
    if provider == "qwen":
        if not (model in _ALLOWED_QWEN_MODELS or model.startswith("Qwen/Qwen3-TTS-")):
            raise ValueError(f"Untrusted or invalid tts_model_name '{model}' for Qwen provider in config settings.")
    # Every other provider only loads ids listed in its ProviderInfo.models
    # (BaseTTSProvider.resolve_model_id), so a foreign model name is ignored.

    # 3. Provider options: a flat mapping of scalars, nothing a provider could
    # be made to traverse or execute.
    options = settings.get("tts_options", {})
    if options is None:
        options = {}
    if not isinstance(options, dict):
        raise ValueError("Invalid tts_options in progress JSON settings: expected an object of KEY: VALUE pairs.")
    for key, value in options.items():
        if not isinstance(key, str) or not isinstance(value, _SCALAR_TYPES):
            raise ValueError(
                f"Invalid tts_options entry {key!r} in progress JSON settings: "
                "values must be strings, numbers, booleans or null."
            )

    # 4. Output dir validation
    # If not explicitly overridden by the user via CLI --output-dir, ensure output_dir stays within safe bounds
    if settings.get("output_dir") and not getattr(args, "output_dir", None):
        out_dir = str(settings["output_dir"])
        norm_out = os.path.realpath(os.path.abspath(out_dir))
        allowed_roots = [os.path.realpath(os.path.abspath(_DEFAULT_OUTPUT_ROOT))]
        if path:
            allowed_roots.append(os.path.realpath(os.path.abspath(os.path.dirname(path))))
        if not any(norm_out == r or norm_out.startswith(r + os.sep) for r in allowed_roots):
            raise _UntrustedOutputDir(
                f"Untrusted output_dir '{out_dir}' in progress JSON: path outside allowed output directories."
            )


def _existing_file_arg(value: str, flag: str) -> str:
    """Absolute path of a file given on the command line; fails if it is missing."""
    path = os.path.abspath(os.path.expanduser(value))
    if not os.path.isfile(path):
        raise _CliError(f"{flag}: file not found: {path}")
    return path


def _apply_cli_overrides(meta: dict, settings: dict, args: Any) -> None:
    """Writes every flag given on the command line into *meta* / *settings*.

    Uses ``getattr`` throughout so a namespace from an older or partial
    parser still works.
    """
    # One-off instructions never come from the file (see _ONE_SHOT_SETTINGS).
    if settings.get("force_reprocess") and not getattr(args, "force_reprocess", False):
        print(_info(
            "  ℹ️  'force_reprocess' saved in the JSON is not applied, so a resume keeps its "
            "progress — pass --force-reprocess to regenerate everything."
        ))
    for name in _ONE_SHOT_SETTINGS:
        settings.pop(name, None)

    book = getattr(args, "book_path", None) or (
        getattr(args, "book", None) if getattr(args, "config_json", None) else None
    )
    if book:
        meta["book_path"] = settings["book_path"] = _existing_file_arg(book, "--book-path")
    if getattr(args, "voice_file", None):
        meta["voice_file"] = settings["voice_file"] = _existing_file_arg(args.voice_file, "--voice-file")
    if getattr(args, "voice_preset", None):
        settings["voice_preset"] = _existing_file_arg(args.voice_preset, "--voice-preset")
    if getattr(args, "cover_image", None):
        settings["cover_image"] = _existing_file_arg(args.cover_image, "--cover-image")
        # An explicit cover wins over the one embedded in the JSON.
        settings.pop("cover_image_b64", None)
        meta["cover_image_b64"] = ""
    if getattr(args, "output_dir", None):
        settings["output_dir"] = os.path.abspath(os.path.expanduser(args.output_dir))

    transcript_file = getattr(args, "voice_transcript_file", None)
    if transcript_file:
        path = _existing_file_arg(transcript_file, "--voice-transcript-file")
        with open(path, encoding="utf-8-sig", errors="replace") as fh:
            settings["voice_transcript"] = fh.read().strip()

    provider = getattr(args, "provider", None)
    if provider:
        previous = settings.get("tts_provider_name", "qwen")
        if _canonical_provider(previous) != _canonical_provider(provider):
            # The model id and the options were saved for the other engine.
            if settings.get("tts_options"):
                print(_info(
                    f"  ℹ️  Engine changed ({previous} → {provider}): "
                    "engine options saved in the JSON are not carried over."
                ))
            settings.pop("tts_options", None)
            settings.pop("tts_model_name", None)
        settings["tts_provider_name"] = provider

    for dest, field_name in _VALUE_FLAGS:
        value = getattr(args, dest, None)
        if value is not None:
            settings[field_name] = value
    for dest, field_name, value in _SWITCH_FLAGS:
        if getattr(args, dest, False):
            settings[field_name] = value
    if getattr(args, "redo", None):
        settings["redo_chapters"] = list(args.redo)


def _finish_settings(
    meta: dict,
    settings: dict,
    args: Any,
    path: str,
    fallback_output_dir: bool,
) -> None:
    """Applies the command line to loaded settings, then validates them."""
    _apply_cli_overrides(meta, settings, args)

    # Validate settings against allowlists to prevent settings poisoning
    try:
        _validate_loaded_settings(settings, args, path)
    except _UntrustedOutputDir as exc:
        if not fallback_output_dir:
            raise
        # Typically a JSON exported on another machine: its absolute path
        # means nothing here. The path is never used; the default is.
        print(_warn(f"  ⚠️  {exc}"))
        print(_warn("      Using the default output directory instead (override with --output-dir)."))
        settings.pop("output_dir", None)
        _validate_loaded_settings(settings, args, path)

    settings["tts_provider_name"] = _canonical_provider(settings.get("tts_provider_name", "qwen"))

    pairs = getattr(args, "tts_option", None)
    if pairs:
        options = dict(settings.get("tts_options") or {})
        options.update(_parse_tts_options(settings["tts_provider_name"], list(pairs)))
        settings["tts_options"] = options


def _load_config(args: Any, *, fallback_output_dir: bool = False) -> tuple[dict, dict, list[dict], str]:
    """
    Returns (meta, settings, chapters_raw, path).
    meta     — top-level keys: book_title, book_path, voice_file
    settings — the 'settings' sub-dict, with the command line applied
    chapters_raw — the 'chapters' list (may include text/sentences)
    path     — absolute path to the loaded progress JSON

    With ``fallback_output_dir`` an ``output_dir`` outside the allowed
    directories is dropped with a warning (the default directory is used)
    instead of raising.
    """
    path = os.path.abspath(args.config_json)
    try:
        data = read_progress_file(path)
    except FileNotFoundError:
        raise _CliError(f"Config JSON not found: {path}") from None
    except ValueError as exc:
        raise _CliError(str(exc)) from None
    except Exception as exc:
        raise _CliError(f"Failed to read config JSON '{path}': {exc}") from None
    if not isinstance(data, dict):
        raise _CliError(f"Config JSON '{path}' does not contain a JSON object.")

    settings = data.get("settings") or {}
    if not isinstance(settings, dict):
        raise _CliError(f"'settings' in '{path}' is not a JSON object.")
    settings = dict(settings)
    chapters_raw = data.get("chapters") or []
    if not isinstance(chapters_raw, list):
        raise _CliError(f"'chapters' in '{path}' is not a list.")
    chapters_raw = [c for c in chapters_raw if isinstance(c, dict)]

    meta = {
        "book_title":  data.get("book_title") or settings.get("book_title") or "Audiobook",
        "book_path":   data.get("book_path", "") or "",
        "voice_file":  data.get("voice_file", "") or "",
        "cover_image_b64": data.get("cover_image_b64", "") or "",
        "_json_dir":   os.path.dirname(path),
    }

    _finish_settings(meta, settings, args, path, fallback_output_dir)
    return meta, settings, chapters_raw, path


def _load_book_settings(args: Any) -> tuple[dict, dict, list[dict], str]:
    """Builds (meta, settings, chapters_raw, path) for a ``--book`` run with no JSON.

    Settings are the defaults plus the command line; the chapter list is
    empty, so the chapters are extracted from the book.
    """
    book = _existing_file_arg(args.book, "--book")
    title = ""
    author = ""
    try:
        from audiobook_factory.text_extractor import scan
        scanned = scan(book)
        title = (scanned.title or "").strip()
        author = (scanned.author or "").strip()
    except Exception as exc:
        print(_warn(f"  ⚠️  Could not read the book's metadata ({exc}); using the file name as title."))
    if not title:
        title = os.path.splitext(os.path.basename(book))[0]

    meta = {
        "book_title": title,
        "book_path": book,
        "voice_file": "",
        "cover_image_b64": "",
        "_json_dir": "",
    }
    from audiobook_factory.pipeline import _CONFIG_SCHEMA_VERSION
    settings: dict = {"book_path": book, "config_version": _CONFIG_SCHEMA_VERSION}
    if author:
        settings["author"] = author
    _finish_settings(meta, settings, args, "", False)
    # --book is the book of this run, whatever --book-path says.
    meta["book_path"] = settings["book_path"] = book
    return meta, settings, [], ""


# ══════════════════════════════════════════════════════════════════════════════
# Build AudiobookConfig from merged settings
# ══════════════════════════════════════════════════════════════════════════════

def _coerce_setting(name: str, value: Any, default: Any) -> Any:
    """Converts a JSON value to the type of the config field's default."""
    if isinstance(default, bool):
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return bool(value)
        if isinstance(value, str):
            low = value.strip().lower()
            if low in _TRUE_WORDS:
                return True
            if low in _FALSE_WORDS:
                return False
        raise ValueError(f"Invalid value for '{name}' in settings: {value!r} is not a boolean.")
    try:
        if isinstance(default, int):
            if isinstance(value, bool):
                return int(value)
            return value if isinstance(value, int) else int(float(value))
        if isinstance(default, float):
            return float(value)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(f"Invalid value for '{name}' in settings: {value!r} is not a number.") from None
    if isinstance(default, str):
        if not isinstance(value, _SCALAR_TYPES):
            raise ValueError(f"Invalid value for '{name}' in settings: expected text.")
        return str(value)
    if isinstance(default, (list, dict)):
        if not isinstance(value, type(default)):
            raise ValueError(
                f"Invalid value for '{name}' in settings: expected a {type(default).__name__}."
            )
        return value
    return value


def _coerce_settings(settings: dict) -> dict:
    """Returns the AudiobookConfig fields found in *settings*, with corrected types.

    ``AudiobookConfig.from_dict`` keeps values as they are; a hand-edited or
    older JSON may hold ``"2"`` or ``2.0`` where an int is expected. ``null``
    means "use the default".
    """
    coerced: dict = {}
    for f in dataclasses.fields(AudiobookConfig):
        if f.name not in settings or settings[f.name] is None:
            continue
        if f.default is not dataclasses.MISSING:
            default = f.default
        elif f.default_factory is not dataclasses.MISSING:
            default = f.default_factory()
        else:
            default = None
        coerced[f.name] = _coerce_setting(f.name, settings[f.name], default)
    return coerced


def _first_usable_path(candidates: Iterable[Any]) -> str:
    """The first candidate that is an existing file, else the first non-empty one."""
    named = [str(c) for c in candidates if c]
    for candidate in named:
        if os.path.isfile(candidate):
            return os.path.abspath(candidate)
    return named[0] if named else ""


def _default_output_dir(book_title: str) -> str:
    """``audiobook_output/<title>`` under the project root."""
    safe_title = re.sub(r'[\\/\*\?:"<>|\x00-\x1f]', "", str(book_title)).strip().strip(".")
    return os.path.join(_DEFAULT_OUTPUT_ROOT, safe_title or "Audiobook")


def _find_cover_file(directory: str | None) -> str | None:
    """A ``cover.*`` image directly inside *directory*, if there is one."""
    if not directory or not os.path.isdir(directory):
        return None
    for name in _COVER_FILE_NAMES:
        candidate = os.path.join(directory, name)
        if os.path.isfile(candidate):
            return candidate
    return None


def _write_cover(data: bytes, output_dir: str) -> str:
    """Saves cover bytes into the output directory; returns the file path."""
    extension = ".png" if data[:8] == b"\x89PNG\r\n\x1a\n" else ".jpg"
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "cover" + extension)
    with open(path, "wb") as fh:
        fh.write(data)
    return path


def _resolve_cover_image(
    meta: dict,
    settings: dict,
    output_dir: str,
    book_path: str,
    write: bool = True,
) -> tuple[str | None, str]:
    """Finds the cover for this book; returns ``(path, where it came from)``.

    Looked for, in order: the configured file; the image embedded in the
    JSON; a ``cover.*`` file in the output directory, next to the JSON; the
    cover inside the book file; a ``cover.*`` file next to the book. Nothing
    else is searched — in particular not the current directory, where the
    first image or EPUB found would belong to whatever book lives there.

    With ``write=False`` nothing is saved to disk, and a cover that would
    have to be written first is reported with a ``None`` path.
    """
    explicit = settings.get("cover_image")
    if explicit and os.path.isfile(str(explicit)):
        return os.path.abspath(str(explicit)), "configured file"

    cover_b64 = settings.get("cover_image_b64") or meta.get("cover_image_b64")
    if cover_b64:
        try:
            data = base64.b64decode(cover_b64)
            if not data:
                raise ValueError("empty image data")
            if not write:
                return None, "embedded in the JSON (saved when the run starts)"
            return _write_cover(data, output_dir), "embedded in the JSON"
        except Exception as exc:
            print(_warn(f"  ⚠️  Could not decode the cover image embedded in the JSON: {exc}"))

    for directory, label in ((output_dir, "output directory"), (meta.get("_json_dir"), "next to the JSON")):
        found = _find_cover_file(directory)
        if found:
            return found, label

    if book_path and os.path.isfile(book_path):
        try:
            from audiobook_factory.text_extractor import scan
            data = scan(book_path).cover_data
        except Exception:
            data = None
        if data:
            if not write:
                return None, "inside the book file (saved when the run starts)"
            return _write_cover(data, output_dir), "inside the book file"
        found = _find_cover_file(os.path.dirname(os.path.abspath(book_path)))
        if found:
            return found, "next to the book file"

    return None, ""


def _build_audiobook_config(meta: dict, settings: dict, write_cover: bool = True) -> AudiobookConfig:
    """Builds the run's AudiobookConfig.

    Every field comes from ``AudiobookConfig.from_dict(settings)``, so a
    field added to the config later round-trips without a change here. Only
    what depends on this machine is resolved on top: the title, the book /
    voice / cover paths and the output directory.
    """
    from audiobook_factory.pipeline import _CONFIG_SCHEMA_VERSION

    cfg = AudiobookConfig.from_dict(_coerce_settings(settings))
    # from_dict has applied the migrations; what is written back is current.
    cfg.config_version = _CONFIG_SCHEMA_VERSION
    # The CLI's preview is --dry-run; a "preview" flag left in an exported
    # JSON must not turn a real run into one that generates nothing.
    cfg.preview_mode = False

    cfg.book_title = str(meta.get("book_title") or cfg.book_title or "Audiobook")
    cfg.book_path = _first_usable_path([settings.get("book_path"), meta.get("book_path")])
    cfg.voice_file = _first_usable_path([settings.get("voice_file"), meta.get("voice_file")])

    if settings.get("output_dir"):
        cfg.output_dir = os.path.abspath(os.path.expanduser(str(settings["output_dir"])))
    else:
        cfg.output_dir = _default_output_dir(cfg.book_title)

    for name in ("voice_preset",):
        value = getattr(cfg, name, "")
        if value:
            setattr(cfg, name, os.path.abspath(os.path.expanduser(str(value))))

    # The model id is only honoured when the engine lists it; show (and save)
    # what will really be loaded.
    try:
        info = _provider_info(cfg.tts_provider_name)
    except Exception:
        info = None
    if info is not None and info.models and cfg.tts_model_name not in info.models:
        if "tts_model_name" in settings:
            print(_warn(
                f"  ⚠️  Model '{cfg.tts_model_name}' is not offered by engine "
                f"'{cfg.tts_provider_name}'; using its default '{info.default_model}'."
            ))
        cfg.tts_model_name = info.default_model

    cfg.cover_image, source = _resolve_cover_image(
        meta, settings, cfg.output_dir, cfg.book_path, write=write_cover
    )
    meta["_cover_source"] = source
    return cfg


# ══════════════════════════════════════════════════════════════════════════════
# Chapter loading — cached or freshly extracted
# ══════════════════════════════════════════════════════════════════════════════

def _chapter_num(entry: dict, position: int) -> int:
    """Chapter number of a JSON chapter entry (its 1-based position when unusable)."""
    raw = entry.get("num")
    if isinstance(raw, bool):
        return position
    if isinstance(raw, int):
        return raw
    if isinstance(raw, str) and raw.strip().isdigit():
        return int(raw.strip())
    return position


def _selection_title(label: Any) -> str:
    """Title inside a UI selection label such as ``"3. Chapter Three  (~500 words)"``."""
    text = re.sub(r"\s*\(~[\d,]+\s*words\)\s*$", "", str(label))
    text = re.sub(r"^\s*\d+\.\s+", "", text)
    return " ".join(text.split()).lower()


def _filter_by_selection(entries: list[dict], selected: list) -> list[dict]:
    """Keeps the chapters named by the UI's saved selection labels.

    Titles are compared exactly (after whitespace/case normalisation): a
    substring test would make "Chapter 1" also select "Chapter 10". Labels
    that match no title fall back to their leading number.
    """
    if not selected:
        return entries
    wanted = {_selection_title(label) for label in selected}
    wanted.discard("")
    by_title = [e for e in entries if " ".join(str(e.get("title", "")).split()).lower() in wanted]
    if by_title:
        return by_title
    numbers: set[int] = set()
    for label in selected:
        match = re.match(r"\s*(\d+)\b", str(label))
        if match:
            numbers.add(int(match.group(1)))
    return [e for i, e in enumerate(entries, 1) if _chapter_num(e, i) in numbers]


def _load_chapters(
    chapters_raw: list[dict],
    meta: dict,
    cfg: AudiobookConfig,
    only_nums: list[int] | None = None,
) -> list[ExtractedChapter]:
    """
    Return a list of ExtractedChapter objects.

    If the JSON holds chapters and every one has non-empty 'text', the list
    is built directly from it — no book parsing needed. Otherwise (text
    missing, or no chapters at all) they are extracted from the book file.

    ``only_nums`` (``--chapters``) restricts the result to those chapter
    numbers and replaces the selection saved in the settings; without it
    that saved selection is applied.
    """
    selected = [] if only_nums else list(cfg.selected_chapters or [])

    # ── Try to use cached text ────────────────────────────────────────────────
    # all() of an empty list is True: a JSON with no chapters has no cache.
    has_text = bool(chapters_raw) and all(str(c.get("text") or "").strip() for c in chapters_raw)

    if has_text:
        print(_info("  📦 Using cached chapter text from JSON (no book re-parsing needed)."))
        entries = _filter_by_selection(chapters_raw, selected)
        if selected and not entries:
            raise _CliError(
                "The chapter selection saved in the JSON matches none of its chapters. "
                "Pass --chapters to choose chapters by number."
            )
        positions = {id(c): i for i, c in enumerate(chapters_raw, 1)}
        chapters = [
            ExtractedChapter(
                num       = _chapter_num(c, positions[id(c)]),
                title     = str(c.get("title") or ""),
                text      = str(c.get("text") or ""),
                sentences = list(c.get("sentences") or []),
            )
            for c in entries
        ]
    else:
        # ── Fall back to extracting from book file ────────────────────────────
        book_path = cfg.book_path
        if not book_path or not os.path.isfile(book_path):
            missing = f" ('{book_path}' does not exist)" if book_path else ""
            raise _CliError(
                "No cached chapter text in the JSON and no usable book file" + missing + ".\n"
                "   Pass --book-path /path/to/book, or export the JSON from the Gradio UI with\n"
                "   '📋 Export Config JSON' (which embeds the chapter text)."
            )

        print(_info(f"  📖 Extracting chapters from book file: {book_path}"))
        from audiobook_factory.text_extractor import extract

        # Build selection titles for the extractor
        selection_titles = None
        if selected:
            selection_titles = []
            for label in selected:
                after_num = str(label).split(". ", 1)[-1]
                title = re.sub(r'\s+\(~[\d,]+\s*words\)\s*$', '', after_num).strip()
                if title:
                    selection_titles.append(title)

        try:
            chapters, _cover = extract(
                book_path,
                selections=selection_titles or None,
                log_fn=lambda message: print(" ", message),
            )
        except Exception as exc:
            raise _CliError(f"Could not extract chapters from '{book_path}': {exc}") from exc

    if only_nums:
        wanted = set(only_nums)
        available = sorted({ch.num for ch in chapters if isinstance(ch.num, int)})
        chapters = [ch for ch in chapters if ch.num in wanted]
        if not chapters:
            span = f"{available[0]}-{available[-1]}" if available else "none"
            raise _CliError(
                f"--chapters matched no chapter. Chapter numbers available: {span} "
                f"({len(available)} chapter(s))."
            )
    return chapters


def _chapter_has_text(chapter: ExtractedChapter) -> bool:
    """True if the chapter contains anything to synthesize."""
    if any((s or "").strip() for s in (chapter.sentences or [])):
        return True
    return bool((chapter.text or "").strip())


def _chapter_chars(chapter: ExtractedChapter) -> int:
    """Number of characters the chapter's narration has."""
    text = chapter.text or ""
    return len(text) if text.strip() else sum(len(s or "") for s in (chapter.sentences or []))


def _read_chapter_entries(path: str) -> list[dict]:
    """Chapter entries of a progress file (empty when it cannot be read)."""
    try:
        data = read_progress_file(path)
    except (FileNotFoundError, ValueError, OSError):
        return []
    entries = data.get("chapters") if isinstance(data, dict) else None
    return [c for c in entries if isinstance(c, dict)] if isinstance(entries, list) else []


def _saved_entries(
    numbered: list[tuple[int, ExtractedChapter]],
    previous: list[dict],
) -> dict[int, dict]:
    """Maps each chapter number of this run to its saved progress entry.

    Uses the pipeline's own matching (same title, preferring the same
    number), so what is shown before the run is what the pipeline will do.
    """
    from audiobook_factory.pipeline import _reconcile_chapter_entries

    wanted = {num for num, _ in numbered}
    result: dict[int, dict] = {}
    for entry in _reconcile_chapter_entries(previous, numbered):
        num = entry.get("num")
        if num in wanted and num not in result:
            result[num] = entry
    return result


def _is_completed(entry: dict | None) -> bool:
    """True when a progress entry says its chapter is done."""
    return bool(entry) and entry.get("status") in _COMPLETED_STATUSES


def _all_outputs_present(
    cfg: AudiobookConfig,
    numbered: list[tuple[int, ExtractedChapter]],
    saved: dict[int, dict],
) -> bool:
    """True when the run would have nothing to do.

    Every chapter must be marked completed with its audio file on disk. Runs
    that regenerate, or that produce one combined file, are left to the
    pipeline to decide.
    """
    if cfg.force_reprocess or cfg.redo_chapters or cfg.single_file_mode or not numbered:
        return False
    from audiobook_factory.filename_sanitizer import make_safe_filename

    for num, chapter in numbered:
        if not _is_completed(saved.get(num)):
            return False
        try:
            name = make_safe_filename(chapter.title, num, cfg.output_dir, f".{cfg.output_format}")
        except Exception:
            return False
        if not os.path.isfile(os.path.join(cfg.output_dir, name)):
            return False
    return True


# ══════════════════════════════════════════════════════════════════════════════
# --dry-run
# ══════════════════════════════════════════════════════════════════════════════

def _estimate_audio_seconds(text: str, speed: float = 1.0) -> float:
    """Rough spoken length of *text* in seconds."""
    cjk = sum(
        1 for ch in text
        if "぀" <= ch <= "ヿ" or "㐀" <= ch <= "鿿" or "가" <= ch <= "힣"
    )
    other = len(text) - cjk
    seconds = cjk * _DRY_RUN_CJK_SECONDS_PER_CHAR + other / _DRY_RUN_CHARS_PER_SECOND
    return seconds / speed if speed and speed > 0 else seconds


def _config_summary(cfg: AudiobookConfig) -> str:
    """One ``name: value`` line per config field, like ``AudiobookConfig.field_summary()``
    but with this run's values."""
    lines = []
    for f in dataclasses.fields(cfg):
        text = repr(getattr(cfg, f.name))
        if len(text) > _DRY_RUN_VALUE_WIDTH:
            text = text[: _DRY_RUN_VALUE_WIDTH - 1] + "…"
        lines.append(f"  {f.name}: {text}")
    return "\n".join(lines)


def _print_dry_run(
    cfg: AudiobookConfig,
    meta: dict,
    numbered: list[tuple[int, ExtractedChapter]],
    saved: dict[int, dict],
    devices: list[str],
) -> None:
    """Prints what a run would do. Loads no model and writes nothing."""
    rule = "━" * 78
    print(_h(rule))
    print(_h("  DRY RUN — nothing is generated, no model is loaded"))
    print(_h(rule))

    try:
        info = _provider_info(cfg.tts_provider_name)
    except Exception as exc:
        info = None
        print(f"  {_h('Engine:')}    {cfg.tts_provider_name} — {_warn(f'unavailable: {exc}')}")
    if info is not None:
        commercial = "commercial use allowed" if info.commercial_use else "NON-COMMERCIAL use only"
        missing = _missing_requirements(info.pip_requirements)
        print(f"  {_h('Engine:')}    {info.display_name} ({cfg.tts_provider_name})")
        print(f"  {_h('Licence:')}   {info.license} — {commercial}")
        print(f"  {_h('Model:')}     {cfg.tts_model_name or info.default_model}")
        print(f"  {_h('Min VRAM:')}  {info.min_vram_gb:g} GB")
        if missing:
            print(f"  {_h('Installed:')} {_warn('NO — missing: ' + ', '.join(missing))}")
            print(_install_help(cfg.tts_provider_name, info))
        else:
            print(f"  {_h('Installed:')} yes")
    print(f"  {_h('Devices:')}   {', '.join(devices)}")
    print(f"  {_h('Cover:')}     {cfg.cover_image or meta.get('_cover_source') or '(none found)'}")
    print()
    print(_h("  Resolved configuration"))
    print(_config_summary(cfg))
    print()

    redo = {int(n) for n in (cfg.redo_chapters or []) if str(n).isdigit()}
    header = f"  {'Num':>4}  {'Chapter':<40} {'Chars':>9} {'Words':>8}  {'Status':<11} {'Est. audio':>10}"
    print(_h(f"  Chapters ({len(numbered)})"))
    print(_h(header))
    print("  " + "─" * (len(header) - 2))
    total_chars = 0
    total_words = 0
    total_seconds = 0.0
    todo_seconds = 0.0
    for num, chapter in numbered:
        text = chapter.text or " ".join(chapter.sentences or [])
        chars = _chapter_chars(chapter)
        words = len(text.split())
        seconds = _estimate_audio_seconds(text, cfg.speed)
        done = _is_completed(saved.get(num))
        if cfg.force_reprocess or num in redo:
            status = "redo" if done else "pending"
        else:
            status = str((saved.get(num) or {}).get("status") or "pending")
        if not _chapter_has_text(chapter):
            status = "empty"
        if status not in _COMPLETED_STATUSES and status != "empty":
            todo_seconds += seconds
        total_chars += chars
        total_words += words
        total_seconds += seconds
        title = chapter.title if len(chapter.title) <= 40 else chapter.title[:39] + "…"
        print(f"  {num:>4}  {title:<40} {chars:>9,} {words:>8,}  {status:<11} {_format_hms(seconds):>10}")
    print("  " + "─" * (len(header) - 2))
    print(_h(f"  {'':>4}  {'TOTAL':<40} {total_chars:>9,} {total_words:>8,}  {'':<11} {_format_hms(total_seconds):>10}"))
    print()
    print(f"  {_h('Estimated audio:')} {_format_hms(total_seconds)} for these chapters, "
          f"{_format_hms(todo_seconds)} still to generate")
    print(f"  (about {_DRY_RUN_CHARS_PER_SECOND:g} characters per second of speech at speed "
          f"{cfg.speed:g}; pauses not included — an estimate, not a measurement)")
    print(_h(rule))


# ══════════════════════════════════════════════════════════════════════════════
# --embed-cover-only
# ══════════════════════════════════════════════════════════════════════════════

def _embed_cover_only(cfg: AudiobookConfig) -> int:
    """Embeds ``cfg.cover_image`` into the audio files in ``cfg.output_dir``.

    Audio is stream-copied; an existing embedded picture is replaced. Each
    file is rewritten next to itself and swapped in atomically, so a failure
    leaves the original untouched.

    Returns the process exit status: 0 when every file that can carry a cover
    got one, 1 otherwise.
    """
    from audiobook_factory.pipeline import _COVER_FORMATS, _ensure_valid_cover_image, _get_cover_flags

    if not cfg.cover_image or not os.path.isfile(cfg.cover_image):
        raise _CliError(
            "No cover image found. Pass --cover-image /path/to/cover.jpg "
            "(or put cover.jpg in the output directory)."
        )
    if not os.path.isdir(cfg.output_dir):
        raise _CliError(f"Output directory does not exist: {cfg.output_dir}")
    if shutil.which("ffmpeg") is None:
        raise _CliError("ffmpeg was not found on PATH; it is needed to embed the cover.")

    audio_extensions = set(_output_format_choices())
    candidates: list[str] = []
    unsupported: list[str] = []
    for name in sorted(os.listdir(cfg.output_dir)):
        path = os.path.join(cfg.output_dir, name)
        extension = os.path.splitext(name)[1].lstrip(".").lower()
        if name.startswith(".") or not os.path.isfile(path) or extension not in audio_extensions:
            continue
        (candidates if extension in _COVER_FORMATS else unsupported).append(path)

    for path in unsupported:
        print(_warn(f"  ⚠️  Skipped {os.path.basename(path)}: this format cannot carry a cover image."))
    if not candidates:
        raise _CliError(
            f"No audio files that can carry a cover ({', '.join(_COVER_FORMATS)}) in {cfg.output_dir}"
        )

    print(_info(f"🎨 Embedding cover image ({cfg.cover_image}) into {len(candidates)} file(s) in {cfg.output_dir}..."))
    embedded = 0
    work_dir = tempfile.mkdtemp(prefix=".abm_cover_", dir=cfg.output_dir)
    try:
        cover = _ensure_valid_cover_image(cfg.cover_image, work_dir)
        for path in candidates:
            name = os.path.basename(path)
            fmt = os.path.splitext(name)[1].lstrip(".").lower()
            temp_out = os.path.join(work_dir, name)
            cmd = [
                "ffmpeg", "-y", "-loglevel", "error", "-i", path, "-i", cover, "-c:a", "copy",
                *_get_cover_flags(fmt, True), temp_out,
            ]
            try:
                result = subprocess.run(cmd, capture_output=True)
            except OSError as exc:
                print(_warn(f"  ⚠️  Failed {name}: {exc}"))
                continue
            if result.returncode == 0 and os.path.isfile(temp_out) and os.path.getsize(temp_out) > 0:
                os.replace(temp_out, path)
                print(_ok(f"  ✓ Embedded cover into: {name}"))
                embedded += 1
            else:
                message = result.stderr.decode("utf-8", errors="replace").strip()[-300:]
                print(_warn(f"  ⚠️  Could not embed cover into {name}: {message}"))
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    if embedded == len(candidates):
        print(_ok(f"🎉 Done! Embedded the cover image into {embedded} audio file(s)."))
        return _EXIT_OK
    print(_err(f"❌ Embedded the cover into {embedded} of {len(candidates)} audio file(s)."))
    return _EXIT_FAILED


# ══════════════════════════════════════════════════════════════════════════════
# Progress / log consumer (prints to console)
# ══════════════════════════════════════════════════════════════════════════════

def _consume_queues(
    log_q: queue.Queue,
    prog_q: queue.Queue,
    cancel: Any,
    runner_thread: threading.Thread,
) -> list[str]:
    """Drain log and progress queues while the runner thread is alive."""
    last_step = -1.0
    line_open = False

    while runner_thread.is_alive() or not log_q.empty() or not prog_q.empty():
        # Progress
        try:
            while not prog_q.empty():
                cur, tot = prog_q.get_nowait()
                if tot <= 0:
                    continue
                pct = cur / tot * 100
                text = f"[Progress] {pct:.1f}% ({cur:.1f}/{tot:.0f} chapters)"
                if _IS_TTY:
                    # Overwrite the progress line in-place
                    print(f"\r  {_info(text)}", end="", flush=True)
                    line_open = True
                elif pct - last_step >= _PROGRESS_STEP_PERCENT or (pct >= 100 and last_step < 100):
                    # One line per step: notebooks and log files keep every "\r" line.
                    last_step = pct
                    print(f"  {text}", flush=True)
        except queue.Empty:
            pass

        # Log messages
        try:
            msg = log_q.get(timeout=0.15)
            done_files = decode_done_message(msg)
            if done_files is not None:
                out_files = [p for p in done_files if p and os.path.exists(p)]
                if line_open:
                    print()  # newline after progress line
                return out_files
            if line_open:
                print(f"\r  {msg}                                    ")
                line_open = False
            else:
                print(f"  {msg}", flush=True)
        except queue.Empty:
            pass

    if line_open:
        print()
    return []


def _clear_voice_cache() -> None:
    """Clear all preprocessed voice audio cache files."""
    from audiobook_factory.voice_preprocessor import _get_cache_dir, _CACHE_ENTRY_SUFFIX
    cache_dir = _get_cache_dir()
    count = 0
    if os.path.exists(cache_dir):
        for fname in os.listdir(cache_dir):
            if fname.endswith(_CACHE_ENTRY_SUFFIX):
                fpath = os.path.join(cache_dir, fname)
                try:
                    os.remove(fpath)
                    count += 1
                except OSError as exc:
                    print(_warn(f"Failed to remove cache file {fname}: {exc}"))
    print(_ok(f"✅ Cleared {count} voice preprocessing cache file(s)."))


# ══════════════════════════════════════════════════════════════════════════════
# Run and report
# ══════════════════════════════════════════════════════════════════════════════

def _seed_progress_file(json_path: str, dest_json: str) -> None:
    """Gives a new output directory its first progress file.

    The JSON named on the command line is copied only when the output
    directory has no progress file yet. Once it has one, that file is the
    newer record — copying over it on every start (as re-running a notebook
    cell does) reset every chapter finished since to whatever the exported
    JSON said. Settings still come from the JSON given: the pipeline writes
    this run's settings into the output directory's file.
    """
    if not json_path or os.path.abspath(json_path) == os.path.abspath(dest_json):
        return
    if os.path.exists(dest_json):
        print(_info(
            f"  ↻  Resuming: chapter progress is read from {dest_json}\n"
            "     (settings and overrides are taken from the JSON given on the command line)."
        ))
        return
    try:
        shutil.copy2(json_path, dest_json)
    except OSError as exc:
        print(_warn(f"  ⚠️  Could not copy progress JSON to output dir: {exc}"))


def _report_fatal(state: _RunState, cfg: AudiobookConfig) -> None:
    """Prints the reason a run died."""
    exc = state.error
    if isinstance(exc, _ApiTaskFailed):
        print(_err(f"  ❌ Task failed on the API server: {exc}"))
        return
    try:
        info = _provider_info(cfg.tts_provider_name)
    except Exception:
        info = None
    if _has_import_error(exc):
        # A missing or mismatched package: the install command helps, a
        # traceback through the model loader does not.
        print(_err(f"  ❌ TTS engine '{cfg.tts_provider_name}' could not be loaded: {exc}"))
        if info is not None:
            print(_install_help(cfg.tts_provider_name, info))
        return
    print(_err(f"  ❌ Fatal error: {exc}"))
    preflight_errors = getattr(getattr(exc, "result", None), "errors", None) or []
    for problem in preflight_errors:
        print(_err(f"     - {problem}"))
    warmup_failed = "warmups failed" in str(exc)
    if warmup_failed and info is not None and _install_command(info):
        # The pool reports only that every device failed to load the model.
        print(_warn(
            "     The cause is in the log above. If the engine's packages are missing "
            "or the wrong version:"
        ))
        print(_install_help(cfg.tts_provider_name, info))
    # Errors that already say what to do need no traceback; anything else is
    # unexpected and the traceback is what a bug report needs.
    self_explanatory = isinstance(exc, (_ApiError, _CliError)) or preflight_errors or warmup_failed
    if state.error_trace and (not self_explanatory or os.environ.get("ABM_DEBUG") == "1"):
        print(state.error_trace)


def _report_outcome(
    cfg: AudiobookConfig,
    numbered: list[tuple[int, ExtractedChapter]],
    out_files: list[str],
    state: _RunState,
    cancelled: bool,
) -> int:
    """Prints the final summary and returns the process exit status.

    0 only when every requested chapter that has text is marked completed in
    the output directory's progress file.
    """
    rule = _h("━" * 50)
    entries = {
        _chapter_num(entry, position): entry
        for position, entry in enumerate(_read_chapter_entries(os.path.join(cfg.output_dir, _PROGRESS_FILE_NAME)), 1)
    }
    requested = [(num, chapter) for num, chapter in numbered if _chapter_has_text(chapter)]
    skipped_empty = len(numbered) - len(requested)
    not_done = [(num, chapter) for num, chapter in requested if not _is_completed(entries.get(num))]
    flagged = [
        (num, chapter, entries[num].get("flagged_chunks") or [])
        for num, chapter in requested
        if num in entries and entries[num].get("flagged_chunks")
    ]
    audio_seconds = sum(
        float(entries[num].get("duration") or 0.0)
        for num, _ in requested if num in entries and _is_completed(entries[num])
    )

    print()
    print(rule)
    if state.error is not None:
        _report_fatal(state, cfg)
    if cancelled:
        print(_warn("  ⛔ Generation cancelled."))

    done_count = len(requested) - len(not_done)
    print(f"  {_h('Chapters:')} {done_count} of {len(requested)} completed"
          + (f"  ({skipped_empty} without text skipped)" if skipped_empty else "")
          + (f"  — {_format_hms(audio_seconds)} of audio" if audio_seconds else ""))

    if not_done and not cancelled:
        print(_err(f"  ❌ {len(not_done)} chapter(s) did not complete:"))
        for num, chapter in not_done:
            entry = entries.get(num) or {}
            reason = entry.get("last_error") or (
                "failed" if entry.get("status") == "failed" else "not generated"
            )
            print(_err(f"     Chapter {num} '{chapter.title}': {reason}"))
    if flagged:
        total_flagged = sum(len(chunks) for _, _, chunks in flagged)
        print(_warn(
            f"  ⚠️  {total_flagged} chunk(s) in {len(flagged)} chapter(s) were kept after "
            "failing verification — worth a listen:"
        ))
        for num, chapter, chunks in flagged:
            print(_warn(f"     Chapter {num} '{chapter.title}': {len(chunks)} chunk(s)"))
            for item in chunks[:_SUMMARY_MAX_FLAGGED]:
                if isinstance(item, dict):
                    print(_warn(f"       “{str(item.get('text', ''))[:70]}” ({item.get('reason', '')})"))

    if out_files:
        print(_ok(f"  ✅ {len(out_files)} file(s) in {cfg.output_dir}:"))
        for f in out_files:
            print(f"     {f}")
    elif not cancelled and state.error is None:
        print(_warn("  ⚠️  No output files were generated. Check the log above for errors."))

    if cancelled:
        code = _EXIT_CANCELLED
        print(_info("  Run the same command again to resume; finished chapters are skipped."))
    elif state.error is not None or not_done:
        code = _EXIT_FAILED
        if not_done and state.error is None:
            print(_info("  Run the same command again to retry the failed chapter(s); finished chapters are skipped."))
    else:
        code = _EXIT_OK
    print(rule)
    print()
    return code


def _generate(
    cfg: AudiobookConfig,
    chapters: list[ExtractedChapter],
    numbered: list[tuple[int, ExtractedChapter]],
    force_local: bool,
) -> int:
    """Runs the job (through the API server when it is up, else in-process),
    printing its log, and returns the process exit status."""
    from audiobook_factory.pipeline import CancelToken

    cancel = CancelToken()
    log_q: queue.Queue = queue.Queue()
    prog_q: queue.Queue = queue.Queue()
    state = _RunState()

    use_api = False
    if force_local:
        print(_info("  🖥️  Running locally in-process (--local)."))
    else:
        use_api = _is_api_healthy()
        if use_api:
            print(_info(f"  📡 FastAPI orchestrator is running at {_api_base_url()} — dispatching task to it."))
        else:
            print(_info("  🖥️  FastAPI orchestrator not detected — running locally in-process."))
    print()

    def _runner() -> None:
        try:
            files: list[str] | None = None
            if use_api:
                try:
                    files = _run_via_api(cfg, chapters, log_q, prog_q, cancel)
                except _ApiDeclined as declined:
                    log_q.put(f"⚠️ Not using the API server: {declined}. Running locally in-process instead.")
            if files is None:
                files = _run_local(cfg, chapters, log_q, prog_q, cancel)
            state.files = list(files or [])
        except BaseException as exc:  # noqa: BLE001 - reported by the main thread
            state.error = exc
            state.error_trace = traceback.format_exc()
        finally:
            log_q.put(encode_done_message(state.files))

    # The handler exists only while there is a token for it to cancel. Before
    # this point Ctrl+C raises KeyboardInterrupt as usual.
    previous_handler = _install_sigint_handler(cancel)
    try:
        t = threading.Thread(target=_runner, daemon=True, name="abm-cli-runner")
        t.start()
        out_files = _consume_queues(log_q, prog_q, cancel, t)
        t.join(timeout=10)
    finally:
        _restore_sigint_handler(previous_handler)

    cancelled = cancel.is_cancelled
    if cancelled:
        # Whatever the interrupted run raised on its way out is not the news.
        state.error = None
    return _report_outcome(cfg, numbered, out_files or state.files, state, cancelled)


def _run(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    """Executes the parsed command line; returns the process exit status."""
    if getattr(args, "clear_voice_cache", False):
        _clear_voice_cache()
        return _EXIT_OK
    if getattr(args, "list_providers", False):
        _print_providers()
        return _EXIT_OK
    if not args.config_json and not args.book:
        parser.error("give a generation_progress.json, or --book PATH to run straight from a book file")

    dry_run = bool(getattr(args, "dry_run", False))
    _print_banner()
    from audiobook_factory.gpu_pool import GPUDetector
    devices = GPUDetector.detect_devices()
    if devices == ["cpu"]:
        print(_info("[GPU] No CUDA GPU detected — running on CPU"))
    else:
        dev_str = ", ".join(devices)
        print(_info(f"[GPU] Detected {len(devices)} GPU(s): {dev_str}"))

    # ── Load and merge config ──────────────────────────────────────────────────
    try:
        if args.config_json:
            meta, settings, chapters_raw, json_path = _load_config(args, fallback_output_dir=True)
        else:
            meta, settings, chapters_raw, json_path = _load_book_settings(args)
        embed_only = bool(getattr(args, "embed_cover_only", False))
        cfg = _build_audiobook_config(meta, settings, write_cover=embed_only or not dry_run)
        # The pipeline's own checks (output format, verify mode, …): fail
        # here with the message instead of as a traceback from the run.
        from audiobook_factory.pipeline import _validate_config
        _validate_config(cfg)
    except ValueError as exc:
        raise _CliError(str(exc)) from exc

    # ── Handle --embed-cover-only ──────────────────────────────────────────────
    if embed_only:
        print(f"  {_h('Book:')}     {cfg.book_title}")
        print(f"  {_h('Output:')}   {cfg.output_dir}")
        print(f"  {_h('CoverImg:')} {cfg.cover_image or '(none found)'}")
        print()
        return _embed_cover_only(cfg)

    # ── Load chapters ─────────────────────────────────────────────────────────
    from audiobook_factory.pipeline import _number_chapters

    has_text = bool(chapters_raw) and all(str(c.get("text") or "").strip() for c in chapters_raw)
    chapters = _load_chapters(chapters_raw, meta, cfg, only_nums=getattr(args, "chapters", None))
    if not chapters:
        raise _CliError(
            "No chapters to generate: the book yielded no chapter text. "
            "Check the book file (or pass --book-path)."
        )
    if not any(_chapter_has_text(ch) for ch in chapters):
        raise _CliError("None of the selected chapters contains any text to narrate.")
    if getattr(args, "chapters", None):
        # The explicit chapter list replaces the saved selection labels.
        cfg.selected_chapters = []
    # The pipeline's numbering: these are the numbers in file names and in
    # the progress file.
    numbered = _number_chapters(chapters)

    dest_json = os.path.join(cfg.output_dir, _PROGRESS_FILE_NAME)
    previous = _read_chapter_entries(dest_json) if os.path.exists(dest_json) else chapters_raw
    saved = {} if cfg.force_reprocess else _saved_entries(numbered, previous)

    # ── Print run summary ──────────────────────────────────────────────────────
    total = len(numbered)
    done = sum(1 for num, _ in numbered if _is_completed(saved.get(num)))
    redo = sorted({int(n) for n in (cfg.redo_chapters or [])} & {num for num, _ in numbered})

    print(f"  {_h('Book:')}     {cfg.book_title}")
    print(f"  {_h('Chapters:')} {total} requested  ({done} completed, {total - done} remaining)"
          + (f"  — redo: {', '.join(str(n) for n in redo)}" if redo else ""))
    print(f"  {_h('Engine:')}   {cfg.tts_provider_name}  ({cfg.tts_model_name})")
    print(f"  {_h('Voice:')}    {cfg.voice_preset or cfg.voice_file or '(built-in voice / style prompt)'}")
    print(f"  {_h('Output:')}   {cfg.output_dir}")
    print(f"  {_h('Format:')}   {cfg.output_format}" + ("  (single file)" if cfg.single_file_mode else ""))
    print(f"  {_h('CoverImg:')} {cfg.cover_image or meta.get('_cover_source') or '(none found)'}")
    print(f"  {_h('TextCache:')} {'✅ yes (fast resume)' if has_text else '⚠️  no  (extracted from the book file)'}")
    print(_h("━" * 50))
    print()

    # ── Handle --dry-run mode ──────────────────────────────────────────────────
    if dry_run:
        _print_dry_run(cfg, meta, numbered, saved, devices)
        return _EXIT_OK

    # ── Checks that need no model ─────────────────────────────────────────────
    if cfg.voice_file and not os.path.isfile(cfg.voice_file):
        raise _CliError(
            f"Voice file not found: {cfg.voice_file}\n"
            "   (a path saved on another machine?) Pass --voice-file /path/to/narrator.wav"
        )
    if cfg.voice_preset and not os.path.isfile(cfg.voice_preset):
        raise _CliError(f"Voice preset not found: {cfg.voice_preset}\n   Pass --voice-preset /path/to/preset")
    _check_provider_installed(cfg.tts_provider_name)

    if _all_outputs_present(cfg, numbered, saved):
        print(_ok(f"✅ Nothing to generate — all {total} requested chapter(s) are already completed."))
        print(_info("   Use --redo N or --force-reprocess to regenerate."))
        return _EXIT_OK

    os.makedirs(cfg.output_dir, exist_ok=True)
    if not cfg.force_reprocess:
        _seed_progress_file(json_path, dest_json)

    return _generate(cfg, chapters, numbered, force_local=bool(getattr(args, "local", False)))


def _is_pipeline_echo(record: logging.LogRecord) -> bool:
    """Logging filter: False for records the CLI already prints from the log queue.

    The pipeline sends every user-facing message to its log queue *and* to
    its logger (from the nested ``log`` helper of ``run_pipeline``). The CLI
    prints the queue, so letting the logger copy through shows each line
    twice.
    """
    return not (record.name == "audiobook_factory.pipeline" and record.funcName == "log")


def _configure_logging() -> None:
    """Sends library log records to stdout, unless the host already set logging up."""
    root = logging.getLogger()
    if root.handlers:
        return
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    handler.addFilter(_is_pipeline_echo)
    root.addHandler(handler)
    root.setLevel(logging.INFO)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns the process exit status (see the module docstring)."""
    _configure_logging()

    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        return _run(args, parser)
    except _CliError as exc:
        print(_err(f"❌ {exc}"))
        return exc.exit_code
    except KeyboardInterrupt:
        print()
        print(_warn("⛔ Interrupted."))
        return _EXIT_CANCELLED


if __name__ == "__main__":
    sys.exit(main())
