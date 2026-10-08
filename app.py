"""
app.py  —  AudiobookMaker Gradio UI
=====================================
Run:  python app.py
Opens: http://localhost:7860

One narrator voice reads the whole book. The UI collects the settings, then
runs generation in this process (``run_pipeline``) or hands it to the local
FastAPI backend (``api/``) when that is healthy.

Layout of this module
---------------------
* Helpers and event handlers are module-level functions so they can be
  tested without a browser. ``build_app()`` only creates components and
  wires them to those handlers.
* Every handler that needs the generation settings receives the same
  ordered list of component values (``_UI_KEYS``) and turns it into an
  ``AudiobookConfig`` with ``_build_config``. Test Voice, Generate, Redo,
  Export Config and the pronunciation audition all go through that one
  function.

Environment
-----------
ABM_API_URL
    Base URL of the FastAPI backend (default ``http://127.0.0.1:8000``).
    Set to an empty string to never use the backend.
ABM_API_SECRET
    Shared secret, sent as ``x-api-key`` on every backend request.
ABM_MULTI_USER
    ``1`` binds each book's progress to the browser that started it (for
    hosting the UI for several people). Off by default: on a single-user
    machine there is nothing to protect and a lock would only lock the
    user out after a page reload.
ABM_SHOW_MOCK_PROVIDER
    ``1`` lists hidden test engines in the provider dropdown.
ABM_SKIP_ENGINE_CHECK
    ``1`` lets Generate / Test Voice try an engine whose pip requirements
    look unmet in this environment.
"""
from __future__ import annotations

import atexit
import base64
import collections
import dataclasses
import functools
import hashlib
import html
import importlib
import importlib.metadata
import importlib.util
import inspect
import io
import json
import logging
import os
import queue
import re
import secrets
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import zipfile
from typing import Any, Callable, Iterator
from urllib.parse import quote

import gradio as gr  # type: ignore
import soundfile as sf

logger = logging.getLogger(__name__)

# ── project root on sys.path ──────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from audiobook_factory.text_extractor import (  # noqa: E402
    scan, extract,
    ScanResult, ExtractedChapter, ExtractionError,
)
from audiobook_factory.voice_preprocessor import (  # noqa: E402
    PreprocessConfig, analyze_voice, preprocess_with_report,
)
from audiobook_factory.pipeline import (  # noqa: E402
    AudiobookConfig, CancelToken, run_pipeline, preview_tts, preview_chapters,
)
from audiobook_factory import SOURCE_URL  # noqa: E402
from audiobook_factory.utils import decode_done_message  # noqa: E402
from audiobook_factory.progress_io import (  # noqa: E402
    read_progress_file, write_progress_file,
)

# ══════════════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════════════

_OUTPUT_DIR = os.path.join(_ROOT, "audiobook_output")
os.makedirs(_OUTPUT_DIR, exist_ok=True)

_PROGRESS_FILE_NAME: str = "generation_progress.json"
_LOG_FILE_NAME: str = "generation_log.txt"
_OWNER_FILE_NAME: str = ".abm_owner"

_MULTI_USER_ENV: str = "ABM_MULTI_USER"
_SHOW_MOCK_ENV: str = "ABM_SHOW_MOCK_PROVIDER"
_SKIP_ENGINE_CHECK_ENV: str = "ABM_SKIP_ENGINE_CHECK"
_API_URL_ENV: str = "ABM_API_URL"
_API_SECRET_ENV: str = "ABM_API_SECRET"
_DEFAULT_API_URL: str = "http://127.0.0.1:8000"
_API_HEALTH_TIMEOUT_SEC: float = 1.0
_API_SPEC_TTL_SEC: float = 60.0
_API_POLL_INTERVAL_SEC: float = 3.0

# Generation log / progress streaming.
_LOG_TAIL_LINES: int = 400
_STREAM_POLL_SEC: float = 0.25
_STREAM_HEARTBEAT_SEC: float = 1.0
_ETA_WINDOW_SEC: float = 180.0
_ETA_MIN_SPAN_SEC: float = 5.0

_AUDIO_EXTENSIONS: tuple[str, ...] = (
    ".mp3", ".m4b", ".m4a", ".flac", ".wav", ".ogg", ".aac", ".webm", ".mp4", ".mov",
)
_BOOK_FILE_TYPES: tuple[str, ...] = (".epub", ".mobi", ".azw3", ".azw", ".pdf", ".docx", ".odt", ".txt")
_OUTPUT_FORMATS: tuple[str, ...] = ("mp3", "m4b", "flac", "wav")
_SAMPLE_RATES: tuple[int, ...] = (22050, 24000, 44100, 48000)
_BITRATES: tuple[int, ...] = (64, 96, 128, 192, 256, 320)
_CHANNELS: tuple[int, ...] = (1, 2)
_QUANTIZATIONS: tuple[str, ...] = ("none", "int8")
_PARALLEL_MODES: tuple[str, ...] = ("chunks", "chapters", "auto")
_VERIFY_CHOICES: tuple[tuple[str, str], ...] = (
    ("Off", "off"),
    ("Duration check (free)", "duration"),
    ("Whisper transcript check (slower, ~2 GB VRAM)", "asr"),
)
_ASR_MODELS: tuple[str, ...] = (
    "openai/whisper-large-v3-turbo",
    "openai/whisper-large-v3",
    "openai/whisper-medium",
    "openai/whisper-small",
    "openai/whisper-base",
)
_PREPROCESS_SAMPLE_RATES: tuple[int, ...] = (16000, 22050, 24000, 44100, 48000)
# Offered when an engine does not publish a language list.
_FALLBACK_LANGUAGES: tuple[str, ...] = (
    "English", "Chinese", "Japanese", "Korean", "French", "Spanish", "Italian", "German",
)
# (minimum, maximum, step) of every numeric control; restoring a saved value
# clamps it into this range.
_RANGES: dict[str, tuple[float, float, float]] = {
    "lufs": (-24, -14, 1),
    "temperature": (0.05, 2.0, 0.05),
    "top_p": (0.05, 1.0, 0.05),
    "top_k": (0, 200, 1),
    "repetition_penalty": (0.8, 2.0, 0.05),
    "speed": (0.5, 2.0, 0.05),
    "pause": (0.0, 2.0, 0.1),
    "para_pause": (0.0, 3.0, 0.1),
    "max_len": (100, 600, 1),
    "true_peak": (-6.0, -0.5, 0.1),
    "verify_max_retries": (0, 5, 1),
    "verify_max_wer": (0.05, 1.0, 0.05),
    "batch_size": (0, 64, 1),
    "gpu_count": (0, 16, 1),
    "vram_headroom_gb": (0.0, 16.0, 0.5),
    "max_chapter_retries": (0, 5, 1),
}

_MATTER_SUFFIX: str = " (front/back matter)"
_WORDS_SUFFIX_RE = re.compile(r"\s+\(~[\d,]+\s*words\)\s*$")
_LABEL_NUM_RE = re.compile(r"^\s*(\d+)\.\s")
_UNSAFE_TITLE_CHARS_RE = re.compile(r'[\\/*?:"<>|\x00-\x1f]')

_DEFAULT_TEST_TEXT: str = "In the beginning, there was only darkness — and then, a single flame."


class _UiError(Exception):
    """A problem the user can act on; the message is shown as it is."""


def _env_flag(name: str) -> bool:
    """True when the environment variable *name* is set to a truthy value."""
    return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")


# ══════════════════════════════════════════════════════════════════════════════
# FastAPI backend client
# ══════════════════════════════════════════════════════════════════════════════

_api_spec_cache: dict[str, Any] = {"at": 0.0, "base": "", "spec": None}
_api_spec_lock = threading.Lock()


def _api_base_url() -> str:
    """Base URL of the backend, or ``""`` when the backend is disabled."""
    return os.environ.get(_API_URL_ENV, _DEFAULT_API_URL).strip().rstrip("/")


def _api_headers() -> dict[str, str]:
    """Headers for every backend request (the shared secret, when configured)."""
    secret = os.environ.get(_API_SECRET_ENV, "")
    return {"x-api-key": secret} if secret else {}


def _api_ws_url(task_id: str) -> str:
    """WebSocket URL of a task's event stream, including the API key."""
    base = _api_base_url()
    scheme = "wss" if base.startswith("https") else "ws"
    url = f"{scheme}://{base.split('://', 1)[-1]}/api/v1/ws/{quote(str(task_id), safe='')}"
    secret = os.environ.get(_API_SECRET_ENV, "")
    if secret:
        url += f"?api_key={quote(secret, safe='')}"
    return url


def _api_request(method: str, path: str, **kwargs: Any) -> Any:
    """Sends one request to the backend. The only place that talks HTTP to it.

    Parameters
    ----------
    method : str
        HTTP method.
    path : str
        Path below the base URL, e.g. ``/api/v1/generate``.
    **kwargs
        Passed to ``requests.request`` (``json``, ``data``, ``files``,
        ``timeout`` …).

    Returns
    -------
    requests.Response

    Raises
    ------
    requests.RequestException
        The backend could not be reached.
    """
    import requests

    base = _api_base_url()
    if not base:
        raise requests.ConnectionError("The API backend is disabled (ABM_API_URL is empty).")
    headers = dict(_api_headers())
    headers.update(kwargs.pop("headers", None) or {})
    return requests.request(method, base + path, headers=headers, **kwargs)


def _api_error_detail(response: Any) -> str:
    """Human-readable reason of a failed backend response."""
    try:
        detail = response.json().get("detail", "")
    except Exception:
        detail = (getattr(response, "text", "") or "").strip()
    if isinstance(detail, dict):
        detail = detail.get("message") or json.dumps(detail)
    elif isinstance(detail, list):
        detail = "; ".join(str(item.get("msg", item)) if isinstance(item, dict) else str(item) for item in detail)
    detail = str(detail or "").strip() or "no detail given"
    return f"{detail} (HTTP {getattr(response, 'status_code', '?')})"


def is_api_healthy() -> bool:
    """True when the FastAPI backend answers its health check."""
    if not _api_base_url():
        return False
    try:
        r = _api_request("GET", "/api/v1/health", timeout=_API_HEALTH_TIMEOUT_SEC)
        return r.status_code == 200 and r.json().get("status") == "ok"
    except Exception:
        return False


def _api_openapi() -> dict[str, Any]:
    """The backend's OpenAPI document (cached briefly; ``{}`` when unavailable)."""
    base = _api_base_url()
    with _api_spec_lock:
        fresh = time.monotonic() - _api_spec_cache["at"] < _API_SPEC_TTL_SEC
        if fresh and _api_spec_cache["base"] == base and _api_spec_cache["spec"] is not None:
            return _api_spec_cache["spec"]
    spec: dict[str, Any] = {}
    try:
        r = _api_request("GET", "/openapi.json", timeout=3.0)
        if r.status_code == 200 and isinstance(r.json(), dict):
            spec = r.json()
    except Exception as exc:
        logger.debug("Could not read the API schema: %s", exc)
    with _api_spec_lock:
        _api_spec_cache.update(at=time.monotonic(), base=base, spec=spec)
    return spec


def _api_form_fields(path: str) -> set[str] | None:
    """Names of the form fields the POST endpoint *path* accepts.

    Returns None when the schema cannot be read, so the caller can decide
    not to rely on the endpoint.
    """
    spec = _api_openapi()
    try:
        content = spec["paths"][path]["post"]["requestBody"]["content"]
        schema = next(iter(content.values()))["schema"]
        if "allOf" in schema and schema["allOf"]:
            schema = schema["allOf"][0]
        if "$ref" in schema:
            schema = spec["components"]["schemas"][schema["$ref"].rsplit("/", 1)[-1]]
        return set(schema.get("properties", {}))
    except Exception:
        return None


# ══════════════════════════════════════════════════════════════════════════════
# Small utilities
# ══════════════════════════════════════════════════════════════════════════════

_ui_temp_root: str | None = None
_ui_temp_lock = threading.Lock()


def _ui_temp_dir() -> str:
    """A private temp directory of this process, removed at exit."""
    global _ui_temp_root
    with _ui_temp_lock:
        if _ui_temp_root is None or not os.path.isdir(_ui_temp_root):
            _ui_temp_root = tempfile.mkdtemp(prefix="abm_ui_")
            atexit.register(shutil.rmtree, _ui_temp_root, True)
        return _ui_temp_root


def _temp_file(name: str) -> str:
    """Path for a new download named *name*, in a directory of its own.

    Each call gets a fresh directory, so two sessions never write the same
    path while both keep the readable file name.
    """
    directory = tempfile.mkdtemp(dir=_ui_temp_dir())
    return os.path.join(directory, os.path.basename(name) or "file")


def _file_path(obj: Any) -> str:
    """Path of an uploaded file, whatever shape Gradio passed it in."""
    if not obj:
        return ""
    if isinstance(obj, str):
        return obj
    if isinstance(obj, dict):
        return str(obj.get("path") or obj.get("name") or "")
    return str(getattr(obj, "name", "") or getattr(obj, "path", "") or "")


def _is_inside(path: str, base: str) -> bool:
    """True when the resolved *path* is *base* or lies beneath it."""
    path = os.path.realpath(path)
    base = os.path.realpath(base)
    return path == base or path.startswith(base + os.sep)


def _bytes_to_gradio_audio(wav_bytes: bytes) -> tuple[int, Any]:
    """Convert raw WAV bytes to (sample_rate, numpy_array) for gr.Audio."""
    audio, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
    return sr, audio


def _make_zip(files: list[str]) -> str:
    """Packs *files* into a new ZIP and returns its path.

    Audio is already compressed, so the entries are stored, not deflated.
    """
    zip_path = _temp_file("AudiobookMaker_output.zip")
    used: set[str] = set()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_STORED, allowZip64=True) as zf:
        for f in files:
            if not f or not os.path.isfile(f):
                continue
            name = os.path.basename(f)
            stem, ext = os.path.splitext(name)
            counter = 2
            while name in used:
                name = f"{stem} ({counter}){ext}"
                counter += 1
            used.add(name)
            zf.write(f, name)
    return zip_path


def _format_hms(seconds: float | None) -> str:
    """Formats a duration as H:MM:SS (``—`` when unknown)."""
    if seconds is None:
        return "—"
    try:
        seconds = int(max(0.0, float(seconds)))
    except (TypeError, ValueError):
        return "—"
    return f"{seconds // 3600}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def _num(value: Any, default: Any, cast: Callable[[Any], Any] = float) -> Any:
    """Casts a UI value to a number, using *default* for None / junk."""
    if value is None or value == "" or isinstance(value, bool):
        return default
    try:
        return cast(value)
    except (TypeError, ValueError):
        return default


def _clamp(key: str, value: Any, default: Any, cast: Callable[[Any], Any] = float) -> Any:
    """Casts *value* and clamps it into the range of the control *key*."""
    number = _num(value, default, float)
    low, high, _step = _RANGES[key]
    number = min(high, max(low, number))
    return cast(number)


def _one_of(value: Any, choices: Any, default: Any) -> Any:
    """Returns *value* when it is one of *choices*, else *default*.

    Numbers saved as strings ("24000") match their numeric choice.
    """
    allowed = [c[1] if isinstance(c, tuple) else c for c in choices]
    if value in allowed and not isinstance(value, bool):
        return value
    for candidate in allowed:
        if str(candidate).lower() == str(value).strip().lower():
            return candidate
    return default


# ══════════════════════════════════════════════════════════════════════════════
# Book titles, output folders, chapter labels
# ══════════════════════════════════════════════════════════════════════════════

def _safe_title(book_title: str | None) -> str:
    """Folder name for a book title (never empty, never a path)."""
    name = _UNSAFE_TITLE_CHARS_RE.sub("", str(book_title or "")).strip().strip(".").strip()
    return name or "audiobook"


def _book_output_dir(book_title: str | None) -> str:
    """Output folder of a book: ``audiobook_output/<title>``."""
    return os.path.join(_OUTPUT_DIR, _safe_title(book_title))


def _progress_path(book_title: str | None) -> str:
    return os.path.join(_book_output_dir(book_title), _PROGRESS_FILE_NAME)


def _chapter_value(num: Any, title: str, word_count: int) -> str:
    """Selection label stored in the config: ``"3. Title  (~1,234 words)"``."""
    return f"{num}. {title}  (~{int(word_count or 0):,} words)"


def _strip_label(label: str) -> str:
    """Removes the display-only parts of a chapter label, leaving the title."""
    text = str(label or "")
    if text.endswith(_MATTER_SUFFIX):
        text = text[: -len(_MATTER_SUFFIX)]
    text = _WORDS_SUFFIX_RE.sub("", text)
    return _LABEL_NUM_RE.sub("", text, count=1).strip()


def _label_num(label: str) -> int | None:
    """Chapter number at the start of a selection label, if it has one."""
    match = _LABEL_NUM_RE.match(str(label or ""))
    return int(match.group(1)) if match else None


def _parse_chapter_titles(labels: list[str] | None) -> list[str] | None:
    """Convert checkbox labels into plain titles."""
    if not labels:
        return None
    out = [title for title in (_strip_label(lbl) for lbl in labels) if title]
    return out or None


def _norm_title(title: Any) -> str:
    return " ".join(str(title or "").split()).lower()


def _selection_from_labels(labels: list[str] | None) -> list[int] | list[str] | None:
    """What to pass to ``extract(selections=…)`` for the ticked chapters.

    Chapter numbers when every label carries one (they are unique, titles
    are not), otherwise titles.
    """
    if not labels:
        return None
    nums = [_label_num(lbl) for lbl in labels]
    if all(n is not None for n in nums):
        return [int(n) for n in nums]  # type: ignore[arg-type]
    return _parse_chapter_titles(labels)


def _parse_page_ranges(text: str | None) -> list[tuple[int, int]] | None:
    """Parses ``"1-50, 51-120"`` into page ranges (None when there are none)."""
    ranges: list[tuple[int, int]] = []
    for part in str(text or "").split(","):
        match = re.fullmatch(r"\s*(\d+)\s*[-–]\s*(\d+)\s*", part)
        if match:
            start, end = int(match.group(1)), int(match.group(2))
            if start >= 1 and end >= start:
                ranges.append((start, end))
    return ranges or None


def _is_matter(chapter: Any) -> bool:
    """True for a scanned chapter flagged as probable front/back matter."""
    return bool(getattr(chapter, "probably_matter", False))


def _chapter_choices(chapters: list[Any]) -> tuple[list[tuple[str, str]], list[str]]:
    """Checkbox choices for scanned chapters.

    Returns
    -------
    tuple[list[tuple[str, str]], list[str]]
        ``(display, value)`` pairs in reading order, and the values ticked by
        default (everything except probable front/back matter).
    """
    choices: list[tuple[str, str]] = []
    default: list[str] = []
    for chapter in chapters:
        value = _chapter_value(
            getattr(chapter, "num", len(choices) + 1),
            getattr(chapter, "title", ""),
            getattr(chapter, "word_count", 0),
        )
        if _is_matter(chapter):
            choices.append((value + _MATTER_SUFFIX, value))
        else:
            choices.append((value, value))
            default.append(value)
    return choices, default


def _match_saved_selection(values: list[str], saved: list[str] | None) -> list[str]:
    """The checklist values that a saved selection (older labels) refers to."""
    if not saved:
        return []
    saved_set = set(saved)
    saved_titles = {_norm_title(_strip_label(s)) for s in saved}
    saved_nums = {_label_num(s) for s in saved} - {None}
    by_exact = [v for v in values if v in saved_set]
    if len(by_exact) == len(saved_set):
        return by_exact
    by_title = [v for v in values if _norm_title(_strip_label(v)) in saved_titles]
    if by_title:
        return by_title
    return [v for v in values if _label_num(v) in saved_nums]


# ══════════════════════════════════════════════════════════════════════════════
# Session ownership of a book's progress (opt-in, ABM_MULTI_USER=1)
# ══════════════════════════════════════════════════════════════════════════════
#
# The owner used to be Gradio's ``session_hash``, which changes on every page
# load and lived only in memory: after a reload or a tunnel change the user
# was locked out of their own book until the server restarted. Now
#   * the check only exists in multi-user mode,
#   * the owner is a token kept in the browser's localStorage, and
#   * it is recorded next to the progress file, so a server restart keeps it.

_PROGRESS_OWNERS: dict[str, str] = {}
_PROGRESS_LOCK = threading.Lock()


def _multi_user_enabled() -> bool:
    """True when books are bound to the browser that started them."""
    return _env_flag(_MULTI_USER_ENV)


def _get_session_id(request: Any | None, client_token: str | None = None) -> str:
    """Identifies who is asking.

    The browser token wins because it survives page reloads; Gradio's
    per-page-load ``session_hash`` is only a fallback for clients without one.
    """
    token = str(client_token or "").strip()
    if token:
        return token
    if request is None:
        return "local"
    return getattr(request, "session_hash", None) or getattr(request, "username", None) or "local"


def _owner_digest(session_id: str) -> str:
    return hashlib.sha256(str(session_id).encode("utf-8")).hexdigest()


def _canonical(path: str) -> str:
    return os.path.realpath(os.path.abspath(path))


def _register_progress_owner(prog_path: str, session_id: str) -> None:
    """Records *session_id* as the owner of a progress file (multi-user mode)."""
    canon = _canonical(prog_path)
    digest = _owner_digest(session_id)
    with _PROGRESS_LOCK:
        _PROGRESS_OWNERS[canon] = digest
    if not _multi_user_enabled() or session_id == "local":
        return
    marker = os.path.join(os.path.dirname(canon), _OWNER_FILE_NAME)
    try:
        if os.path.isdir(os.path.dirname(marker)):
            with open(marker, "w", encoding="utf-8") as fh:
                fh.write(digest)
    except OSError as exc:
        logger.debug("Could not write the owner marker %s: %s", marker, exc)


def _recorded_owner(canon: str) -> str | None:
    owner = _PROGRESS_OWNERS.get(canon)
    if owner is not None:
        return owner
    try:
        with open(os.path.join(os.path.dirname(canon), _OWNER_FILE_NAME), encoding="utf-8") as fh:
            return fh.read().strip() or None
    except OSError:
        return None


def _owns_progress(prog_path: str, session_id: str) -> bool:
    """True when *session_id* may read and write this book's progress.

    Always True unless ``ABM_MULTI_USER=1``. In multi-user mode a book
    belongs to whoever started it; nobody owns a book that was never started
    from the UI, and the local operator (``"local"``) owns everything.
    """
    if not _multi_user_enabled():
        return True
    canon = _canonical(prog_path)
    with _PROGRESS_LOCK:
        owner = _recorded_owner(canon)
    return owner is None or session_id == "local" or owner == _owner_digest(session_id)


def _browser_state_secret() -> str:
    """Stable key for the browser-stored session token.

    Gradio encrypts ``BrowserState`` with a key that is random per process
    unless one is given, which would invalidate every stored token on
    restart.
    """
    path = os.path.join(_OUTPUT_DIR, ".abm_ui_secret")
    try:
        with open(path, encoding="utf-8") as fh:
            secret = fh.read().strip()
        if len(secret) >= 16:
            return secret
    except OSError:
        pass
    secret = secrets.token_hex(16)
    try:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(secret)
        os.chmod(path, 0o600)
    except OSError as exc:
        logger.debug("Could not persist the UI secret: %s", exc)
    return secret


def ensure_client_token(token: str | None) -> str:
    """Returns the browser's session token, creating one on first visit."""
    token = str(token or "").strip()
    return token if len(token) >= 16 else secrets.token_urlsafe(18)


# ══════════════════════════════════════════════════════════════════════════════
# Pronunciation fixes
# ══════════════════════════════════════════════════════════════════════════════

def _read_pronunciation_file(file_obj: Any) -> dict[str, str]:
    """Reads ``search == replace`` lines; ``#`` starts a comment."""
    path = _file_path(file_obj)
    fixes: dict[str, str] = {}
    if not path:
        return fixes
    try:
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#") or "==" not in line:
                    continue
                search, repl = line.split("==", 1)
                if search.strip():
                    fixes[search.strip()] = repl.strip()
    except OSError as exc:
        logger.warning("Could not read the pronunciation file: %s", exc)
    return fixes


def _table_rows(table: Any) -> list[list[Any]]:
    """Rows of a ``gr.Dataframe`` value (list of lists, DataFrame or dict)."""
    if table is None:
        return []
    if isinstance(table, dict):
        table = table.get("data") or []
    if hasattr(table, "values") and hasattr(table, "columns"):
        table = table.values.tolist()
    return [list(row) for row in table if isinstance(row, (list, tuple))]


def _merge_pronunciation(file_obj: Any, table: Any) -> dict[str, str]:
    """Pronunciation map from the uploaded file plus the inline editor.

    A pattern present in both takes the editor's replacement.
    """
    fixes = _read_pronunciation_file(file_obj)
    for row in _table_rows(table):
        if not row:
            continue
        pattern = "" if row[0] is None else str(row[0]).strip()
        replacement = "" if len(row) < 2 or row[1] is None else str(row[1]).strip()
        if pattern and pattern.lower() != "nan":
            fixes[pattern] = "" if replacement.lower() == "nan" else replacement
    return fixes


def _pronunciation_rows(fixes: Any) -> list[list[str]]:
    """Editor rows for a saved pronunciation map (always at least one row)."""
    rows = [[str(k), str(v)] for k, v in fixes.items()] if isinstance(fixes, dict) else []
    return rows or [["", ""]]


# ══════════════════════════════════════════════════════════════════════════════
# TTS engines — everything comes from the provider registry
# ══════════════════════════════════════════════════════════════════════════════

@dataclasses.dataclass(frozen=True)
class _VoiceUi:
    """Which voice controls an engine + model combination uses."""

    mode: str             # "clone" | "speaker" | "design" | "any"
    show_clip: bool
    show_speaker: bool
    show_instruct: bool
    show_design: bool
    show_preset: bool
    show_transcript: bool
    clip_required: bool


_NO_VOICE_UI = _VoiceUi("any", True, False, False, False, False, True, False)


def _provider_keys() -> list[str]:
    """Registry keys of the engines whose module imports, in display order."""
    from audiobook_factory.tts_providers import registry

    keys = []
    for name in registry.provider_names(include_hidden=_env_flag(_SHOW_MOCK_ENV)):
        if _provider_info(name) is not None:
            keys.append(name)
    return keys


def _canonical_provider(name: Any) -> str:
    """Registry key for a provider name or alias (the input when unknown)."""
    from audiobook_factory.tts_providers import registry

    text = str(name or "").strip()
    try:
        return registry.canonical_name(text)
    except ValueError:
        return text.lower()


def _provider_info(name: Any) -> Any:
    """``ProviderInfo`` of an engine, or None when it is unknown / broken."""
    from audiobook_factory.tts_providers import registry

    try:
        return registry.provider_info(_canonical_provider(name))
    except Exception as exc:
        logger.debug("TTS provider %r is unavailable: %s", name, exc)
        return None


def _provider_class(name: Any) -> Any:
    from audiobook_factory.tts_providers import registry

    try:
        return registry.provider_class(_canonical_provider(name))
    except Exception:
        return None


def _provider_choices() -> list[tuple[str, str]]:
    """Dropdown choices: ``(display name, registry key)``."""
    choices = []
    for key in _provider_keys():
        info = _provider_info(key)
        label = getattr(info, "display_name", "") or key
        if not getattr(info, "commercial_use", True):
            label += " (non-commercial)"
        choices.append((label, key))
    return choices


def _default_provider() -> str:
    keys = _provider_keys()
    preferred = _canonical_provider(AudiobookConfig().tts_provider_name)
    return preferred if preferred in keys or not keys else keys[0]


def _missing_requirements(info: Any) -> list[str]:
    """Pip requirements of an engine that are not satisfied here.

    Uses installed-package metadata only — nothing is imported, so the check
    costs milliseconds even for engines that pull in torch.
    """
    try:
        from packaging.requirements import Requirement
    except Exception:
        Requirement = None  # type: ignore[assignment]

    missing: list[str] = []
    for spec in getattr(info, "pip_requirements", ()) or ():
        name = re.split(r"[\s\[<>=!~;@]", str(spec).strip(), maxsplit=1)[0]
        if not name:
            continue
        requirement = None
        if Requirement is not None:
            try:
                requirement = Requirement(str(spec))
                if requirement.marker is not None and not requirement.marker.evaluate():
                    continue
            except Exception:
                requirement = None
        try:
            installed = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            missing.append(name)
            continue
        except Exception:
            continue
        if requirement is not None and requirement.specifier:
            try:
                if not requirement.specifier.contains(installed, prereleases=True):
                    missing.append(f"{name}{requirement.specifier} (installed: {installed})")
            except Exception:
                pass
    return missing


def _install_command(key: str, info: Any) -> str:
    """The pip command that installs an engine's own dependencies."""
    requirements_file = os.path.join("requirements", f"tts-{key}.txt")
    if os.path.isfile(os.path.join(_ROOT, requirements_file)):
        return f"pip install -r {requirements_file}"
    reqs = getattr(info, "pip_requirements", ()) or ()
    return "pip install " + " ".join(f'"{r}"' for r in reqs) if reqs else ""


def _engine_problem(name: Any) -> str | None:
    """Why an engine cannot run in this environment, or None when it can.

    The engines need different ``transformers`` versions and cannot share
    one environment, so choosing one that is not installed here is normal.
    It is refused with its install command instead of crashing mid-load.
    """
    key = _canonical_provider(name)
    info = _provider_info(key)
    if info is None:
        return f"⚠️ The engine '{name}' is not available. Choose another one in Voice Studio."
    if _env_flag(_SKIP_ENGINE_CHECK_ENV):
        return None
    missing = _missing_requirements(info)
    if not missing:
        return None
    shown = ", ".join(missing[:6]) + (" …" if len(missing) > 6 else "")
    command = _install_command(key, info)
    message = f"⚠️ {info.display_name} is not installed in this environment (missing: {shown})."
    if command:
        message += f" Install it with `{command}`"
        message += " — in an environment of its own, see the engine notes in Voice Studio." if info.install_notes else "."
    return message + f" (Set {_SKIP_ENGINE_CHECK_ENV}=1 to try anyway.)"


def _instruct_hint(info: Any) -> str:
    """What the engine's instruction box accepts: its own description."""
    return str(getattr(info, "description", "") or "")


def _provider_markdown(name: Any) -> str:
    """Description panel of an engine: what it is, licence, VRAM, install."""
    key = _canonical_provider(name)
    info = _provider_info(key)
    if info is None:
        return f"⚠ Engine `{html.escape(str(name))}` is not available in this installation."
    lines = [f"**{info.display_name}** — {info.description}".rstrip(" —")]
    if not info.commercial_use:
        lines.append(
            f"> ⚠ **Non-commercial weights** ({info.license}). "
            "Audiobooks made with this engine may not be sold."
        )
    facts = [f"Licence: {info.license}", f"Needs about {info.min_vram_gb:g} GB VRAM"]
    if info.languages:
        facts.append(f"{len(info.languages)} languages")
    if info.homepage:
        facts.append(f"[Project page]({info.homepage})")
    lines.append(" · ".join(facts))
    missing = _missing_requirements(info)
    command = _install_command(key, info)
    if missing:
        shown = ", ".join(missing[:6]) + (" …" if len(missing) > 6 else "")
        lines.append(f"⚠ **Not installed here** — missing: {shown}")
    if command and (missing or info.install_notes):
        lines.append(f"Install: `{command}`")
    if info.install_notes:
        lines.append(f"<small>{html.escape(info.install_notes)}</small>")
    return "\n\n".join(lines)


def _resolve_model(info: Any, model: Any) -> str:
    """The model id to use: *model* when the engine lists it, else its default."""
    if info is None:
        return str(model or "")
    models = tuple(getattr(info, "models", ()) or ())
    if model and (not models or model in models):
        return str(model)
    return getattr(info, "default_model", "") or (models[0] if models else "")


def _voice_ui(provider: Any, model: Any = None) -> _VoiceUi:
    """Decides which voice controls apply to an engine and checkpoint.

    Engines whose checkpoints are named ``…-CustomVoice`` / ``…-VoiceDesign``
    choose the voice source by checkpoint (clip, built-in speaker or a
    described voice). Every other engine shows whatever its ``ProviderInfo``
    says it supports.
    """
    info = _provider_info(provider)
    if info is None:
        return _NO_VOICE_UI
    model = _resolve_model(info, model)
    cls = _provider_class(provider)
    can_design = cls is not None and callable(getattr(cls, "design_voice", None))
    by_checkpoint = any("CustomVoice" in m or "VoiceDesign" in m for m in info.models)
    has_speakers = bool(info.preset_voices)

    if by_checkpoint:
        if "CustomVoice" in model:
            mode = "speaker"
        elif "VoiceDesign" in model:
            mode = "design"
        else:
            mode = "clone"
        show_clip = info.supports_voice_clone and mode == "clone"
        show_speaker = has_speakers and mode == "speaker"
        # The small CustomVoice checkpoint takes no style instruction.
        show_instruct = info.supports_instruct and (
            mode == "design" or (mode == "speaker" and "0.6B" not in model)
        )
        show_design = can_design and mode == "design"
        clip_required = show_clip
    else:
        mode = "any"
        show_clip = info.supports_voice_clone
        show_speaker = has_speakers
        show_instruct = info.supports_instruct
        show_design = can_design
        # An engine that can invent a voice, or has built-in ones, works
        # without a clip.
        clip_required = show_clip and not can_design and not has_speakers
    return _VoiceUi(
        mode=mode,
        show_clip=show_clip,
        show_speaker=show_speaker,
        show_instruct=show_instruct,
        show_design=show_design,
        show_preset=bool(info.supports_voice_preset),
        show_transcript=show_clip and info.transcript != "unused",
        clip_required=clip_required,
    )


def _transcript_hint(info: Any) -> str:
    """Says how the engine uses the reference clip's transcript."""
    mode = getattr(info, "transcript", "optional")
    name = getattr(info, "display_name", "This engine")
    if mode == "required":
        return f"Required: {name} needs the exact words spoken in the clip."
    if mode == "unused":
        return f"{name} ignores the transcript."
    return f"Optional: {name} clones more faithfully when it knows the exact words spoken in the clip."


def _pick_language(info: Any, current: Any) -> str:
    """A language the engine supports, keeping *current* when possible."""
    languages = list(getattr(info, "languages", ()) or ())
    current = str(current or "").strip()
    if not languages:
        return current or AudiobookConfig().language
    for candidate in (current, AudiobookConfig().language, "Auto"):
        for language in languages:
            if language.lower() == candidate.lower():
                return language
    return languages[0]


def _pick_speaker(info: Any, current: Any) -> str | None:
    """A built-in speaker of the engine, matching *current* loosely."""
    voices = list(getattr(info, "preset_voices", ()) or ())
    if not voices:
        return None
    # Older files stored "[English] ryan"; the engine now lists "Ryan".
    wanted = str(current or "").split()[-1].lower() if str(current or "").strip() else ""
    for voice in voices:
        if voice.lower() == wanted:
            return voice
    return voices[0]


def _sampling_values(info: Any, saved: dict[str, Any] | None = None) -> dict[str, Any]:
    """Values for the shared sampling controls.

    The engine's recommended operating point where it publishes one, the
    application defaults otherwise; *saved* (a restored config) wins.
    """
    base = AudiobookConfig()
    recommended = dict(getattr(info, "recommended_settings", {}) or {})
    saved = saved or {}
    values: dict[str, Any] = {}
    for key, cast in (("temperature", float), ("top_p", float), ("top_k", int), ("repetition_penalty", float)):
        raw = saved.get(key, recommended.get(key, getattr(base, key)))
        values[key] = _clamp(key, raw, getattr(base, key), cast)
    return values


def _option_spec(option: Any) -> tuple[str, dict[str, Any]]:
    """Maps a ``ProviderOption`` to a component kind and its arguments.

    Returns
    -------
    tuple[str, dict[str, Any]]
        ``(component, kwargs)`` where component is ``"checkbox"``,
        ``"dropdown"``, ``"slider"``, ``"number"``, ``"textbox"`` or
        ``"file"``.
    """
    kwargs: dict[str, Any] = {"label": option.label or option.key, "info": option.help or None}
    kind = option.kind
    if kind == "bool":
        return "checkbox", {**kwargs, "value": bool(option.default)}
    if kind == "choice":
        choices = [str(c) for c in option.choices]
        value = option.default if option.default in choices else (choices[0] if choices else None)
        return "dropdown", {**kwargs, "choices": choices, "value": value}
    if kind in ("float", "int"):
        is_int = kind == "int"
        default = _num(option.default, 0, int if is_int else float)
        if option.minimum is not None and option.maximum is not None:
            step = option.step or (1 if is_int else round((option.maximum - option.minimum) / 100.0, 4) or 0.01)
            return "slider", {
                **kwargs, "minimum": option.minimum, "maximum": option.maximum,
                "step": step, "value": min(option.maximum, max(option.minimum, default)),
            }
        number: dict[str, Any] = {**kwargs, "value": default}
        if is_int:
            number["precision"] = 0
        return "number", number
    if kind == "file":
        return "file", {"label": kwargs["label"]}
    return "textbox", {**kwargs, "value": "" if option.default is None else str(option.default)}


def _coerce_option(option: Any, value: Any) -> Any:
    """Casts an option value to its declared kind (None when unusable)."""
    if value is None:
        return None
    kind = option.kind
    try:
        if kind == "bool":
            if isinstance(value, str):
                return value.strip().lower() in ("1", "true", "yes", "on")
            return bool(value)
        if kind == "int":
            return int(float(value))
        if kind == "float":
            return float(value)
        if kind == "file":
            return _file_path(value)
        if kind == "choice":
            return str(value) if not option.choices or str(value) in option.choices else None
        return str(value)
    except (TypeError, ValueError):
        return None


def _clean_tts_options(provider: Any, values: Any) -> dict[str, Any]:
    """The ``tts_options`` to send for an engine.

    Keeps only options the engine declares, cast to their kind, and drops
    those left at their default (the provider applies defaults itself).
    """
    info = _provider_info(provider)
    if info is None or not isinstance(values, dict):
        return {}
    cleaned: dict[str, Any] = {}
    for option in info.options:
        if option.key not in values:
            continue
        value = _coerce_option(option, values[option.key])
        if value is None:
            continue
        if option.kind == "file" and not os.path.isfile(value):
            continue   # cleared, or an upload that has since been removed
        default = _coerce_option(option, option.default)
        if value == default or (value == "" and default in (None, "")):
            continue
        cleaned[option.key] = value
    return cleaned


def _provider_options_for(state: Any, provider: Any) -> dict[str, Any]:
    """Raw option values the user set for *provider* in this session."""
    if not isinstance(state, dict):
        return {}
    values = state.get(_canonical_provider(provider))
    return dict(values) if isinstance(values, dict) else {}


def set_provider_option(value: Any, state: Any, provider: Any, key: str) -> dict[str, Any]:
    """Stores one engine option in the per-engine options state."""
    new_state = {k: dict(v) for k, v in state.items() if isinstance(v, dict)} if isinstance(state, dict) else {}
    new_state.setdefault(_canonical_provider(provider), {})[key] = value
    return new_state


# ══════════════════════════════════════════════════════════════════════════════
# UI values → AudiobookConfig (the one place a config is built)
# ══════════════════════════════════════════════════════════════════════════════

# Order of the settings every config-building handler receives. build_app()
# maps each key to its component and fails at start-up if one is missing.
_UI_KEYS: tuple[str, ...] = (
    # Book
    "book_title", "author", "language", "cover_image", "output_format", "lufs",
    "selected_chapters",
    # Voice
    "tts_provider_name", "tts_model_name", "tts_timbre", "tts_instruct",
    "voice_file", "voice_transcript", "voice_preset", "tts_options",
    "temperature", "top_p", "top_k", "repetition_penalty", "seed",
    "speed", "pause", "para_pause",
    # Narration and pipeline
    "max_len", "pack_sentences", "normalize_speech_text",
    "verify_chunks", "verify_max_retries", "verify_asr_model", "verify_max_wer",
    "batch_size", "gpu_count", "vram_headroom_gb", "max_chapter_retries",
    "parallel_mode", "torch_compile", "quantization",
    "sample_rate", "bitrate_kbps", "channels", "true_peak",
    "force_reprocess", "export_text",
    # Output
    "single_file_mode", "export_lrc", "export_srt", "export_vtt",
    "regen_missing", "resume_incomplete_chunks",
    "pronunciation_file", "pronunciation_table",
)

# AudiobookConfig fields that deliberately have no control. Every other field
# must be produced by _build_config (tests/unit/test_app_config.py checks it).
_CONFIG_FIELDS_NOT_IN_UI: frozenset[str] = frozenset({
    "config_version",        # schema constant
    "book_path",             # the uploaded file, passed in by the handler
    "output_dir",            # derived from the book title
    "redo_chapters",         # set by "Redo selected chapters"
    "worker_count",          # read by nothing; accepted in old files only
    "device",                # the GPU pool picks devices
    "preview_mode",          # "Preview Chapters" has its own handler
    "retry_failed_at_end",   # always on
    "nfe_step",              # an engine option (tts_options) now
})


def _ui_values(values: tuple[Any, ...] | list[Any]) -> dict[str, Any]:
    """Names the positional settings a handler received."""
    if len(values) != len(_UI_KEYS):
        raise ValueError(f"Expected {len(_UI_KEYS)} settings, got {len(values)}.")
    return dict(zip(_UI_KEYS, values))


def _build_config(
    ui: dict[str, Any],
    *,
    book_path: str = "",
    output_dir: str | None = None,
    redo_chapters: list[int] | None = None,
) -> AudiobookConfig:
    """Builds the ``AudiobookConfig`` for the current UI values.

    Used by Test Voice, Generate, Redo, Export Config and the pronunciation
    audition, so a setting can never be collected in one place and dropped
    in another. Missing keys fall back to the dataclass defaults.

    Parameters
    ----------
    ui : dict[str, Any]
        Component values keyed by ``_UI_KEYS``.
    book_path : str
        Path of the uploaded book file.
    output_dir : str | None
        Output folder; defaults to ``audiobook_output/<book title>``.
    redo_chapters : list[int] | None
        Chapter numbers to regenerate even if completed.
    """
    base = AudiobookConfig()

    def get(key: str) -> Any:
        value = ui.get(key)
        return getattr(base, key) if value is None and hasattr(base, key) else value

    provider = _canonical_provider(ui.get("tts_provider_name") or base.tts_provider_name)
    info = _provider_info(provider)
    model = _resolve_model(info, ui.get("tts_model_name")) or base.tts_model_name
    voice = _voice_ui(provider, model)

    # A hidden control keeps its last value; only what the chosen engine and
    # checkpoint actually use goes into the config.
    voice_preset = _file_path(ui.get("voice_preset")) if voice.show_preset else ""
    voice_file = _file_path(ui.get("voice_file")) if voice.show_clip else ""
    transcript = str(ui.get("voice_transcript") or "").strip() if voice.show_transcript else ""
    timbre = str(ui.get("tts_timbre") or "").strip() if voice.show_speaker else ""
    instruct = str(ui.get("tts_instruct") or "").strip() if voice.show_instruct else ""
    if info is None:
        voice_file = _file_path(ui.get("voice_file"))
        transcript = str(ui.get("voice_transcript") or "").strip()

    tts_options = _clean_tts_options(provider, _provider_options_for(ui.get("tts_options"), provider))
    cover_image = _file_path(ui.get("cover_image"))
    book_title = str(ui.get("book_title") or "").strip() or base.book_title
    verify = _one_of(ui.get("verify_chunks"), _VERIFY_CHOICES, base.verify_chunks)

    return AudiobookConfig(
        book_title=book_title,
        author=str(ui.get("author") or "").strip() or base.author,
        language=str(ui.get("language") or "").strip() or base.language,
        # The backend rejects a path that does not exist; "no cover" is None.
        cover_image=cover_image if os.path.isfile(cover_image) else None,
        book_path=book_path or "",
        output_dir=output_dir or _book_output_dir(book_title),
        output_format=_one_of(ui.get("output_format"), _OUTPUT_FORMATS, base.output_format),
        voice_file=voice_file,
        voice_transcript=transcript,
        tts_provider_name=provider,
        temperature=_num(ui.get("temperature"), base.temperature),
        top_p=_num(ui.get("top_p"), base.top_p),
        max_len=_num(ui.get("max_len"), base.max_len, int),
        pause=_num(ui.get("pause"), base.pause),
        para_pause=_num(ui.get("para_pause"), base.para_pause),
        lufs=_num(ui.get("lufs"), base.lufs, int),
        true_peak=_num(ui.get("true_peak"), base.true_peak),
        bitrate_kbps=_num(ui.get("bitrate_kbps"), base.bitrate_kbps, int),
        channels=_num(ui.get("channels"), base.channels, int),
        parallel_mode=_one_of(ui.get("parallel_mode"), _PARALLEL_MODES, base.parallel_mode),
        gpu_count=max(0, _num(ui.get("gpu_count"), base.gpu_count, int)),
        vram_headroom_gb=max(0.0, _num(ui.get("vram_headroom_gb"), base.vram_headroom_gb)),
        tts_model_name=model,
        tts_instruct=instruct,
        tts_timbre=timbre,
        voice_preset=voice_preset,
        tts_options=tts_options,
        export_text=bool(get("export_text")),
        export_lrc=bool(get("export_lrc")),
        export_srt=bool(get("export_srt")),
        export_vtt=bool(get("export_vtt")),
        single_file_mode=bool(get("single_file_mode")),
        max_chapter_retries=max(0, _num(ui.get("max_chapter_retries"), base.max_chapter_retries, int)),
        force_reprocess=bool(get("force_reprocess")),
        resume_incomplete_chunks=bool(get("resume_incomplete_chunks")),
        regen_missing=bool(get("regen_missing")),
        sample_rate=_num(ui.get("sample_rate"), base.sample_rate, int),
        repetition_penalty=_num(ui.get("repetition_penalty"), base.repetition_penalty),
        top_k=_num(ui.get("top_k"), base.top_k, int),
        speed=_num(ui.get("speed"), base.speed),
        nfe_step=_num(tts_options.get("nfe_step"), base.nfe_step, int),
        seed=_num(ui.get("seed"), base.seed, int),
        torch_compile=bool(get("torch_compile")),
        quantization=_one_of(ui.get("quantization"), _QUANTIZATIONS, base.quantization),
        selected_chapters=list(ui.get("selected_chapters") or []),
        redo_chapters=[int(n) for n in (redo_chapters or [])],
        batch_size=max(0, _num(ui.get("batch_size"), base.batch_size, int)),
        pack_sentences=bool(get("pack_sentences")),
        normalize_speech_text=bool(get("normalize_speech_text")),
        verify_chunks=verify,
        verify_max_retries=max(0, _num(ui.get("verify_max_retries"), base.verify_max_retries, int)),
        verify_asr_model=str(ui.get("verify_asr_model") or "").strip() or base.verify_asr_model,
        verify_max_wer=_num(ui.get("verify_max_wer"), base.verify_max_wer),
        pronunciation_map=_merge_pronunciation(ui.get("pronunciation_file"), ui.get("pronunciation_table")),
    )


def _voice_problem(cfg: AudiobookConfig) -> str | None:
    """Why this config cannot speak yet, or None when it can.

    The single voice check behind Test Voice, Generate, Redo and the
    pronunciation audition. Driven by ``ProviderInfo``, not by engine names.
    """
    problem = _engine_problem(cfg.tts_provider_name)
    if problem:
        return problem
    info = _provider_info(cfg.tts_provider_name)
    voice = _voice_ui(cfg.tts_provider_name, cfg.tts_model_name)
    if cfg.voice_preset:
        if not os.path.isfile(cfg.voice_preset):
            return "⚠️ The voice preset file is no longer available. Upload it again in Voice Studio."
        return None
    if voice.clip_required and not cfg.voice_file:
        return (
            f"⚠️ {info.display_name} needs a narrator voice. Upload a reference clip in Voice Studio"
            + (" or load a voice preset." if voice.show_preset else ".")
        )
    if cfg.voice_file and not os.path.isfile(cfg.voice_file):
        return "⚠️ The narrator voice clip is no longer available. Upload it again in Voice Studio."
    if voice.mode == "speaker" and not cfg.tts_timbre:
        return "⚠️ Choose a preset speaker in Voice Studio."
    if voice.mode == "design" and not cfg.tts_instruct:
        return "⚠️ Describe the narrator's voice in the voice instruction box in Voice Studio."
    if (
        info.transcript == "required"
        and cfg.voice_file
        and not cfg.voice_transcript
        and not cfg.tts_options.get("allow_missing_transcript")
    ):
        return (
            f"⚠️ {info.display_name} needs the transcript of the reference clip. "
            "Type the exact words spoken in it into the transcript box in Voice Studio."
        )
    return None


# ══════════════════════════════════════════════════════════════════════════════
# Preview synthesis (Test Voice, audition, voice design, presets)
# ══════════════════════════════════════════════════════════════════════════════

def _preview_provider(cfg: AudiobookConfig) -> Any:
    """The provider instance ``preview_tts`` uses, created if necessary.

    Shares the pipeline's one-model preview cache, so designing a voice or
    saving a preset reuses the model a voice test already loaded.
    """
    import audiobook_factory.pipeline as pipeline_module
    from audiobook_factory.tts_providers import get_tts_provider

    getter = getattr(pipeline_module, "get_preview_provider", None)
    if callable(getter):
        return getter(cfg)
    cache = getattr(pipeline_module, "_preview_provider_cache", None)
    lock = getattr(pipeline_module, "_preview_cache_lock", None)
    if not isinstance(cache, dict) or lock is None:
        return get_tts_provider(cfg.tts_provider_name, cfg)
    key = (cfg.tts_provider_name, getattr(cfg, "tts_model_name", ""), getattr(cfg, "quantization", ""))
    with lock:
        provider = cache.get("provider")
        if provider is None or cache.get("key") != key:
            if provider is not None:
                try:
                    provider.cleanup()
                except Exception as exc:
                    logger.warning("Error releasing the previous preview model: %s", exc)
            provider = get_tts_provider(cfg.tts_provider_name, cfg)
            cache.update(provider=provider, name=cfg.tts_provider_name, key=key)
        else:
            provider.config = cfg
    return provider


def _release_preview_provider() -> None:
    """Frees the cached preview model."""
    import audiobook_factory.pipeline as pipeline_module

    cleanup = getattr(pipeline_module, "_cleanup_preview_provider", None)
    if callable(cleanup):
        try:
            cleanup()
        except Exception as exc:
            logger.debug("Could not release the preview model: %s", exc)


def _stretch_preview(wav_bytes: bytes, speed: float) -> bytes:
    """Time-stretches a preview clip the way the pipeline stretches chapters."""
    speed = min(2.0, max(0.5, float(speed or 1.0)))
    if abs(speed - 1.0) < 0.01 or shutil.which("ffmpeg") is None:
        return wav_bytes
    source = _temp_file("preview_in.wav")
    target = _temp_file("preview_out.wav")
    try:
        with open(source, "wb") as fh:
            fh.write(wav_bytes)
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-i", source, "-filter:a", f"atempo={speed:.3f}", target],
            check=True, timeout=60, stdin=subprocess.DEVNULL,
        )
        with open(target, "rb") as fh:
            return fh.read()
    except Exception as exc:
        logger.debug("Could not time-stretch the preview: %s", exc)
        return wav_bytes
    finally:
        for path in (source, target):
            shutil.rmtree(os.path.dirname(path), ignore_errors=True)


def _spoken_text(text: str, cfg: AudiobookConfig) -> str:
    """The text the pipeline would hand to the engine for *text*."""
    import audiobook_factory.pipeline as pipeline_module

    prepare = getattr(pipeline_module, "_prepare_speech_text", None)
    if callable(prepare):
        return prepare(text, cfg)
    apply = getattr(pipeline_module, "_apply_pronunciation", None)
    return apply(text, cfg.pronunciation_map) if callable(apply) and cfg.pronunciation_map else text


def _synthesize_preview(cfg: AudiobookConfig, text: str) -> bytes:
    """Speaks *text* with *cfg* and returns WAV bytes.

    Goes through the backend when it is up. A backend that answers with an
    error is reported as that error: synthesizing here instead would load a
    second copy of the model next to the backend's.

    Raises
    ------
    _UiError
        With a message for the user.
    """
    import requests

    wav_bytes: bytes | None = None
    if is_api_healthy():
        try:
            r = _api_request("POST", "/api/v1/voice-test", json={"config": dataclasses.asdict(cfg), "text": text})
        except requests.ConnectionError as exc:
            logger.info("Voice test: backend went away (%s); synthesizing in the UI process.", exc)
        else:
            if r.status_code != 200:
                raise _UiError(f"❌ The backend could not synthesize the preview: {_api_error_detail(r)}")
            wav_bytes = r.content
    if wav_bytes is None:
        active = _active_local_run()
        if active is not None:
            raise _UiError(
                f"⚠️ '{active.book_title}' is being generated. Test the voice after it finishes "
                "(a preview would load a second copy of the model)."
            )
        try:
            wav_bytes = preview_tts(text, cfg)
        except Exception as exc:
            raise _UiError(f"❌ Preview failed: {exc}") from exc
    if not wav_bytes:
        raise _UiError("❌ The engine returned no audio. Check the voice settings.")
    info = _provider_info(cfg.tts_provider_name)
    if info is not None and not info.supports_speed:
        wav_bytes = _stretch_preview(wav_bytes, cfg.speed)
    return wav_bytes


def _with_preview_provider(cfg: AudiobookConfig, action: Callable[[Any], Any]) -> Any:
    """Runs *action(provider)* on the preview model.

    The backend has no endpoint for voice design or presets, so with the
    backend running the model is loaded here just for the call and released
    again straight away.
    """
    active = _active_local_run()
    if active is not None:
        raise _UiError(f"⚠️ '{active.book_title}' is being generated. Try again after it finishes.")
    backend_up = is_api_healthy()
    try:
        return action(_preview_provider(cfg))
    finally:
        if backend_up:
            _release_preview_provider()


# ══════════════════════════════════════════════════════════════════════════════
# Voice preprocessing
# ══════════════════════════════════════════════════════════════════════════════

def _report_dict(report: Any) -> dict[str, Any]:
    """A ``VoiceReport`` (or its JSON form) as a plain dict."""
    if report is None:
        return {}
    if isinstance(report, dict):
        return report
    to_dict = getattr(report, "to_dict", None)
    return to_dict() if callable(to_dict) else dict(getattr(report, "__dict__", {}))


def _report_metrics(report: Any) -> str:
    """One line of measurements: duration, loudness, SNR, speech share."""
    data = _report_dict(report)
    if not data:
        return ""
    parts = []
    if data.get("duration_s") is not None:
        parts.append(f"Duration {float(data['duration_s']):.1f} s")
    if data.get("loudness_lufs") is not None:
        parts.append(f"Loudness {float(data['loudness_lufs']):.1f} LUFS")
    if data.get("snr_db") is not None:
        parts.append(f"SNR {float(data['snr_db']):.0f} dB")
    if data.get("speech_ratio") is not None:
        parts.append(f"Speech {float(data['speech_ratio']) * 100:.0f} %")
    return " · ".join(parts)


def _report_markdown(report: Any, ok_text: str) -> str:
    """Metrics line plus warnings; the success tick only without warnings."""
    data = _report_dict(report)
    warnings = [str(w) for w in (data.get("warnings") or []) if str(w).strip()]
    lines = []
    if warnings:
        lines.append(f"⚠️ **{len(warnings)} thing(s) to check**")
    else:
        lines.append(f"✅ {ok_text}")
    metrics = _report_metrics(data)
    if metrics:
        lines.append(metrics)
    if warnings:
        lines.append("\n".join(f"- {html.escape(w)}" for w in warnings))
    return "\n\n".join(lines)


def analyze_voice_clip(path: Any) -> str:
    """Measures a clip uploaded straight into Voice Studio and lists its problems."""
    path = _file_path(path)
    if not path:
        return "*Upload a reference clip, or carry one over from Voice Preprocessing.*"
    try:
        report = analyze_voice(path)
    except Exception as exc:
        return f"⚠️ Could not analyse this clip: {exc}"
    return _report_markdown(report, "Clip looks fine for cloning.")


def _preprocess_config(
    noise_reduce: Any, noise_strength: Any,
    gate: Any, gate_db: Any, gate_range: Any,
    highpass: Any, highpass_hz: Any,
    trim_silence: Any, shorten_pauses: Any, min_segment_ms: Any, max_silence_ms: Any,
    loudness_lufs: Any,
    resample: Any, target_sr: Any,
    best_window: Any, best_window_seconds: Any,
) -> PreprocessConfig:
    """``PreprocessConfig`` from the Voice Preprocessing controls."""
    base = PreprocessConfig()
    return PreprocessConfig(
        noise_reduce=bool(noise_reduce),
        noise_reduce_strength=_num(noise_strength, base.noise_reduce_strength),
        noise_gate=bool(gate),
        noise_gate_threshold_db=_num(gate_db, base.noise_gate_threshold_db),
        noise_gate_range_db=_num(gate_range, base.noise_gate_range_db),
        highpass_filter=bool(highpass),
        highpass_cutoff_hz=_num(highpass_hz, base.highpass_cutoff_hz, int),
        trim_silence=bool(trim_silence),
        silence_removal=bool(shorten_pauses),
        min_segment_ms=_num(min_segment_ms, base.min_segment_ms, int),
        max_silence_kept_ms=_num(max_silence_ms, base.max_silence_kept_ms, int),
        loudness_target_lufs=_num(loudness_lufs, base.loudness_target_lufs),
        formant_shift=False,
        resample=bool(resample),
        target_sample_rate=_num(target_sr, base.target_sample_rate, int),
        select_best_window=bool(best_window),
        best_window_seconds=_num(best_window_seconds, base.best_window_seconds),
    )


def _report_from_response(response: Any) -> dict[str, Any] | None:
    """The voice report a preprocess response carries, if the backend sent one."""
    for header in ("x-voice-report", "x-abm-voice-report", "x-preprocess-report"):
        raw = response.headers.get(header)
        if not raw:
            continue
        for decode in (lambda s: s, lambda s: base64.b64decode(s).decode("utf-8")):
            try:
                data = json.loads(decode(raw))
                if isinstance(data, dict):
                    return data
            except Exception:
                continue
    return None


def _preprocess_via_api(raw_path: str, cfg: PreprocessConfig) -> tuple[bytes, dict[str, Any] | None] | None:
    """Runs preprocessing on the backend.

    Returns None when the backend should not be used: it is down, its schema
    is unknown, or it does not accept a setting the user changed (dropping
    that setting silently would process the clip differently from what the
    screen shows).

    Raises
    ------
    _UiError
        The backend answered with an error.
    """
    import requests

    if not is_api_healthy():
        return None
    accepted = _api_form_fields("/api/v1/preprocess")
    if not accepted:
        return None
    values = dataclasses.asdict(cfg)
    defaults = dataclasses.asdict(PreprocessConfig())
    dropped = [k for k, v in values.items() if k not in accepted and v != defaults[k]]
    if dropped:
        logger.info("Backend preprocess endpoint lacks %s; preprocessing in the UI process.", ", ".join(dropped))
        return None
    data = {
        key: (str(value).lower() if isinstance(value, bool) else value)
        for key, value in values.items() if key in accepted
    }
    try:
        with open(raw_path, "rb") as fh:
            r = _api_request(
                "POST", "/api/v1/preprocess", data=data,
                files={"audio_file": (os.path.basename(raw_path), fh, "application/octet-stream")},
            )
    except requests.ConnectionError as exc:
        logger.info("Preprocess: backend went away (%s); preprocessing in the UI process.", exc)
        return None
    if r.status_code != 200:
        raise _UiError(f"❌ The backend could not preprocess the clip: {_api_error_detail(r)}")
    if "json" in r.headers.get("content-type", ""):
        body = r.json()
        audio = body.get("audio_b64") or body.get("wav_b64") or body.get("audio")
        if not audio:
            return None
        return base64.b64decode(audio), body.get("report") if isinstance(body.get("report"), dict) else None
    return r.content, _report_from_response(r)


def run_preprocess(raw_audio: Any, *controls: Any) -> tuple[Any, str, Any]:
    """Cleans the uploaded clip and reports what the cloning model will hear.

    Returns
    -------
    tuple
        ``(audio for the player, status markdown, processed WAV bytes)``.
    """
    raw_path = _file_path(raw_audio)
    if not raw_path:
        return None, "⚠️ Upload a voice recording first.", None
    try:
        cfg = _preprocess_config(*controls)
        result = _preprocess_via_api(raw_path, cfg)
        if result is not None:
            out_bytes, report = result
            if report is None:
                report = analyze_voice(out_bytes)
        else:
            out_bytes, report = preprocess_with_report(raw_path, cfg, use_cache=False)
        sr, audio = _bytes_to_gradio_audio(out_bytes)
    except _UiError as exc:
        return None, str(exc), None
    except Exception as exc:
        logger.exception("Voice preprocessing failed")
        return None, f"❌ Preprocessing failed: {exc}", None
    return (sr, audio), _report_markdown(report, "Preprocessing complete."), out_bytes


def save_processed_voice(wav_bytes: Any, raw_audio: Any) -> tuple[str, Any]:
    """Hands the processed clip to Voice Studio."""
    if not wav_bytes:
        return "⚠️ Run preprocessing first.", gr.update()
    stem = os.path.splitext(os.path.basename(_file_path(raw_audio)))[0] or "narrator_voice"
    path = _temp_file(f"{stem}_processed.wav")
    try:
        with open(path, "wb") as fh:
            fh.write(wav_bytes)
    except OSError as exc:
        return f"❌ Could not save the processed clip: {exc}", gr.update()
    return "✅ Processed clip set as the narrator voice (see Voice Studio).", path


# ══════════════════════════════════════════════════════════════════════════════
# Reference transcription (optional helper)
# ══════════════════════════════════════════════════════════════════════════════

def _transcription_available() -> bool:
    """True when a Whisper backend and the verifier's ASR helper are present."""
    try:
        from audiobook_factory import chunk_verifier
    except Exception:
        return False
    verifier = getattr(chunk_verifier, "ChunkVerifier", None)
    if verifier is None or not callable(getattr(verifier, "_transcribe", None)):
        return False
    return any(importlib.util.find_spec(m) is not None for m in ("faster_whisper", "transformers"))


def transcribe_reference(voice_file: Any, language: Any, asr_model: Any) -> tuple[Any, str]:
    """Transcribes the reference clip with Whisper for the transcript box."""
    path = _file_path(voice_file)
    if not path:
        return gr.update(), "⚠️ Upload a reference clip first."
    try:
        from audiobook_factory.chunk_verifier import ChunkVerifier

        samples, rate = sf.read(path, dtype="float32", always_2d=True)
        verifier = ChunkVerifier(
            mode="asr",
            language=str(language or "English"),
            asr_model=str(asr_model or AudiobookConfig().verify_asr_model),
        )
        try:
            text = verifier._transcribe(samples.mean(axis=1), int(rate))
        finally:
            verifier.close()
    except Exception as exc:
        return gr.update(), f"❌ Transcription failed: {exc}"
    if not text:
        return gr.update(), "⚠️ Whisper returned nothing. Type the transcript by hand."
    return text, "✅ Transcribed. Read it once and correct any wrong word."


# ══════════════════════════════════════════════════════════════════════════════
# Generation runs
# ══════════════════════════════════════════════════════════════════════════════
#
# A run lives in a registry keyed by the book's output folder, not in the
# generator that started it. A reloaded page (new Gradio session) finds the
# run again, shows its log and progress, and can cancel it.

class _RunLogSink(queue.Queue):
    """``log_queue`` for ``run_pipeline`` that appends straight to the run."""

    def __init__(self, run: "_Run") -> None:
        super().__init__()
        self._run = run

    def put(self, item: Any, block: bool = True, timeout: float | None = None) -> None:  # noqa: ARG002
        files = decode_done_message(item) if isinstance(item, str) else None
        if files is not None:
            return
        self._run.add_log(str(item))


class _RunProgressSink(queue.Queue):
    """``prog_queue`` for ``run_pipeline`` that records the latest fraction."""

    def __init__(self, run: "_Run") -> None:
        super().__init__()
        self._run = run

    def put(self, item: Any, block: bool = True, timeout: float | None = None) -> None:  # noqa: ARG002
        try:
            current, total = item
            if float(total) > 0:
                self._run.set_progress(float(current) / float(total))
        except (TypeError, ValueError):
            pass


class _EtaEstimator:
    """Time remaining from the progress made over the last few minutes.

    A whole-run average would be thrown off by chapters that are skipped
    instantly because they are already complete; a recent window forgets
    those jumps.
    """

    def __init__(self, window_sec: float = _ETA_WINDOW_SEC) -> None:
        self._window = window_sec
        self._samples: collections.deque[tuple[float, float]] = collections.deque()

    def update(self, fraction: float, now: float | None = None) -> None:
        """Records the progress fraction observed at *now*."""
        now = time.monotonic() if now is None else now
        fraction = min(1.0, max(0.0, float(fraction)))
        if self._samples and fraction < self._samples[-1][1]:
            self._samples.clear()
        if not self._samples or fraction != self._samples[-1][1]:
            self._samples.append((now, fraction))
        while len(self._samples) > 2 and now - self._samples[0][0] > self._window:
            self._samples.popleft()

    def remaining(self, now: float | None = None) -> float | None:
        """Estimated seconds left, or None while there is too little to go on."""
        if len(self._samples) < 2:
            return None
        now = time.monotonic() if now is None else now
        start_time, start_fraction = self._samples[0]
        _, fraction = self._samples[-1]
        span = now - start_time
        gained = fraction - start_fraction
        if span < _ETA_MIN_SPAN_SEC or gained <= 0:
            return None
        return max(0.0, (1.0 - fraction) * span / gained)


class _Run:
    """One generation run: its log, progress, result and cancel token."""

    def __init__(self, key: str, book_title: str, output_dir: str, owner: str, output_format: str) -> None:
        self.key = key
        self.book_title = book_title
        self.output_dir = output_dir
        self.owner = owner
        self.output_format = output_format
        self.cancel = CancelToken()
        self.task_id: str | None = None
        self.local = True                 # False while the backend does the work
        self.remote_log_count = 0         # backend log lines received so far
        self.started = time.monotonic()
        self.finished_at: float | None = None
        self.out_files: list[str] = []
        self.error: str = ""
        self.final_status: str = ""       # backend verdict: completed / failed / cancelled
        self.chapters: list[ExtractedChapter] = []
        self.chapter_nums: list[int] = []
        self.extraction_key: Any = None
        self.reused_extraction = False
        self._lock = threading.Lock()
        self._lines: list[str] = []
        self._fraction = 0.0
        self._done = threading.Event()
        self.log_q = _RunLogSink(self)
        self.prog_q = _RunProgressSink(self)

    # ── written by the worker ────────────────────────────────────────────────
    def add_log(self, message: str) -> None:
        with self._lock:
            self._lines.extend(str(message).split("\n"))

    def set_progress(self, fraction: float) -> None:
        with self._lock:
            self._fraction = min(1.0, max(0.0, float(fraction)))

    def finish(self) -> None:
        self.finished_at = time.monotonic()
        self._done.set()

    # ── read by the UI ───────────────────────────────────────────────────────
    @property
    def done(self) -> bool:
        return self._done.is_set()

    @property
    def cancelled(self) -> bool:
        return self.cancel.is_cancelled or self.final_status == "cancelled"

    @property
    def elapsed(self) -> float:
        return (self.finished_at or time.monotonic()) - self.started

    def snapshot(self) -> tuple[int, float]:
        """``(number of log lines, progress fraction)`` right now."""
        with self._lock:
            return len(self._lines), self._fraction

    def tail(self, lines: int = _LOG_TAIL_LINES) -> str:
        """The last *lines* log lines, with a note when older ones are hidden."""
        with self._lock:
            total = len(self._lines)
            shown = self._lines[-lines:]
        text = "\n".join(shown)
        if total > lines:
            text = f"… {total - lines:,} earlier lines hidden — the full log is saved with the book …\n" + text
        return text

    def full_log(self) -> str:
        with self._lock:
            return "\n".join(self._lines)

    def request_cancel(self) -> None:
        """Stops the run, wherever it is executing."""
        self.cancel.cancel()
        task_id = self.task_id
        if task_id:
            try:
                _api_request("POST", f"/api/v1/tasks/{task_id}/cancel", timeout=5)
            except Exception as exc:
                logger.warning("Could not cancel backend task %s: %s", task_id, exc)


_RUNS: dict[str, _Run] = {}
_RUNS_LOCK = threading.Lock()


def _run_key(output_dir: str) -> str:
    return _canonical(output_dir)


def _active_run_for(output_dir: str) -> _Run | None:
    """The unfinished run of this book, if there is one."""
    with _RUNS_LOCK:
        run = _RUNS.get(_run_key(output_dir))
    return run if run is not None and not run.done else None


def _active_runs() -> list[_Run]:
    with _RUNS_LOCK:
        runs = [r for r in _RUNS.values() if not r.done]
    return sorted(runs, key=lambda r: r.started)


def _active_local_run() -> _Run | None:
    """A run that is using this process's GPU pool right now."""
    return next((r for r in _active_runs() if r.local), None)


def _visible_runs(session_id: str) -> list[_Run]:
    """Active runs this session may see: all of them unless multi-user."""
    runs = _active_runs()
    if _multi_user_enabled() and session_id != "local":
        runs = [r for r in runs if r.owner == session_id]
    return runs


def _chapter_numbers(chapters: list[ExtractedChapter]) -> list[int]:
    """The numbers the pipeline gives these chapters (files, tags, progress)."""
    import audiobook_factory.pipeline as pipeline_module

    numberer = getattr(pipeline_module, "_number_chapters", None)
    if callable(numberer):
        try:
            return [int(num) for num, _ in numberer(chapters)]
        except Exception:
            pass
    return list(range(1, len(chapters) + 1))


async def _listen_task_ws(run: _Run, task_id: str) -> None:
    """Follows a backend task's event stream until it ends.

    Raises when the stream breaks before the task finished, so the caller
    can fall back to polling.
    """
    import websockets

    finished = False
    try:
        async with websockets.connect(_api_ws_url(task_id), max_size=None) as ws:
            while True:
                data = json.loads(await ws.recv())
                kind = data.get("type")
                if kind == "log":
                    run.add_log(str(data.get("message", "")))
                    run.remote_log_count += 1
                elif kind == "progress":
                    run.set_progress(float(data.get("progress") or 0.0))
                elif kind == "status":
                    if data.get("status") in ("completed", "failed", "cancelled"):
                        run.final_status = data["status"]
                        finished = True
                elif kind == "completed":
                    run.out_files = [f for f in data.get("files") or [] if isinstance(f, str)]
                    run.final_status = "completed"
                    finished = True
                elif kind == "session_end":
                    if data.get("files"):
                        run.out_files = [f for f in data["files"] if isinstance(f, str)]
                    run.final_status = data.get("status") or run.final_status or (
                        "cancelled" if data.get("cancelled") else "completed" if data.get("success") else "failed"
                    )
                    if data.get("error"):
                        run.error = str(data["error"])
                    return
                elif kind == "error":
                    raise RuntimeError(str(data.get("message") or "backend stream error"))
    except Exception:
        if finished:
            return
        raise


def _poll_task(run: _Run, task_id: str) -> None:
    """Follows a backend task by polling, for when the WebSocket is gone."""
    offset = run.remote_log_count
    failures = 0
    while True:
        try:
            r = _api_request("GET", f"/api/v1/tasks/{task_id}", params={"since": offset}, timeout=10)
            if r.status_code == 404:
                run.error = "The backend no longer knows this task (was it restarted?)."
                run.final_status = "failed"
                return
            r.raise_for_status()
            state = r.json()
            failures = 0
        except Exception as exc:
            failures += 1
            if failures >= 20:
                run.error = f"Lost contact with the backend: {exc}"
                run.final_status = "failed"
                return
            time.sleep(_API_POLL_INTERVAL_SEC)
            continue
        logs = state.get("logs") or []
        if "log_offset" not in state:
            # Older backend: the whole log on every request.
            logs = logs[offset:]
        for line in logs:
            run.add_log(str(line))
        offset += len(logs)
        run.remote_log_count = offset
        run.set_progress(float(state.get("progress") or 0.0))
        status = state.get("status")
        if status in ("completed", "failed", "cancelled"):
            run.final_status = status
            run.out_files = [f for f in state.get("output_files") or [] if isinstance(f, str)]
            if state.get("error_message"):
                run.error = str(state["error_message"])
            return
        time.sleep(_API_POLL_INTERVAL_SEC)


def _follow_task(run: _Run, task_id: str) -> None:
    """Streams a backend task into *run* until the task ends."""
    import asyncio

    try:
        asyncio.run(_listen_task_ws(run, task_id))
    except Exception as exc:
        run.add_log(f"⚠️ Live stream interrupted ({exc}). Polling the backend instead…")
        _poll_task(run, task_id)


def _generate_via_api(run: _Run, cfg: AudiobookConfig, chapters: list[ExtractedChapter]) -> bool:
    """Hands the run to the backend.

    Returns False when the backend could not be reached at all (the caller
    then generates in this process). A backend that *rejects* the request is
    an error, not a reason to load a second model here.
    """
    import requests

    payload = {
        "config": dataclasses.asdict(cfg),
        "chapters": [
            {"num": ch.num, "title": ch.title, "text": ch.text, "sentences": ch.sentences}
            for ch in chapters
        ],
    }
    run.add_log("📡 Sending the job to the backend…")
    try:
        r = _api_request("POST", "/api/v1/generate", json=payload, timeout=120)
    except requests.ConnectionError as exc:
        run.add_log(f"⚠️ Backend unreachable ({exc}). Generating in the UI process instead.")
        return False
    if r.status_code != 200:
        raise _UiError(f"The backend rejected the job: {_api_error_detail(r)}")
    task_id = str(r.json().get("task_id") or "")
    if not task_id:
        raise _UiError("The backend accepted the job but returned no task id.")
    run.local = False
    run.task_id = task_id
    run.cancel.task_id = task_id  # type: ignore[attr-defined]
    run.add_log(f"✅ Queued on the backend. Task ID: {task_id}")
    # Cancel may have been pressed while the request was in flight; the
    # backend has never heard of it.
    if run.cancel.is_cancelled:
        run.request_cancel()
    _follow_task(run, task_id)
    return True


def _run_worker(run: _Run, cfg: AudiobookConfig, load_chapters: Callable[[_Run], list[ExtractedChapter]]) -> None:
    """Thread body of a run: extract, then generate here or on the backend."""
    try:
        chapters = load_chapters(run)
        run.chapters = chapters or []
        run.chapter_nums = _chapter_numbers(run.chapters)
        if not chapters:
            run.error = "No chapters with text were found for this selection."
            return
        # Cancel pressed during extraction: nothing has been queued or
        # loaded yet, so stop here.
        if run.cancel.is_cancelled:
            run.add_log("⛔ Cancelled before generation started.")
            return
        if is_api_healthy() and _generate_via_api(run, cfg, chapters):
            return
        run.local = True
        run.out_files = list(run_pipeline(cfg, chapters, run.log_q, run.prog_q, run.cancel) or [])
    except _UiError as exc:
        run.error = str(exc)
        run.add_log(f"❌ {exc}")
    except Exception as exc:
        logger.exception("Generation run crashed")
        run.error = str(exc)
        run.add_log(f"❌ [Fatal Error] Pipeline crashed: {exc}")
    finally:
        try:
            log_path = os.path.join(run.output_dir, _LOG_FILE_NAME)
            with open(log_path, "w", encoding="utf-8") as fh:
                fh.write(run.full_log() + "\n")
        except OSError as exc:
            logger.debug("Could not save the generation log: %s", exc)
        run.finish()


def _start_run(
    cfg: AudiobookConfig,
    owner: str,
    load_chapters: Callable[[_Run], list[ExtractedChapter]],
) -> _Run:
    """Registers and starts a run for ``cfg.output_dir``."""
    key = _run_key(cfg.output_dir)
    run = _Run(key, cfg.book_title, cfg.output_dir, owner, cfg.output_format)
    with _RUNS_LOCK:
        existing = _RUNS.get(key)
        if existing is not None and not existing.done:
            return existing
        _RUNS[key] = run
    threading.Thread(target=_run_worker, args=(run, cfg, load_chapters), daemon=True).start()
    return run


def _find_backend_task(book_title: str) -> str | None:
    """Task id of a queued / running backend task for this book title."""
    try:
        r = _api_request("GET", "/api/v1/tasks", timeout=5)
        if r.status_code != 200:
            return None
        for task in r.json().get("tasks") or []:
            if task.get("status") in ("queued", "running") and task.get("book_title") == book_title:
                return str(task.get("task_id") or "") or None
    except Exception as exc:
        logger.debug("Could not list backend tasks: %s", exc)
    return None


def _attach_backend_task(cfg: AudiobookConfig, owner: str, task_id: str) -> _Run:
    """Creates a run that follows a backend task started before this UI."""
    key = _run_key(cfg.output_dir)
    run = _Run(key, cfg.book_title, cfg.output_dir, owner, cfg.output_format)
    run.local = False
    run.task_id = task_id
    run.cancel.task_id = task_id  # type: ignore[attr-defined]
    with _RUNS_LOCK:
        existing = _RUNS.get(key)
        if existing is not None and not existing.done:
            return existing
        _RUNS[key] = run

    def _follow() -> None:
        try:
            run.add_log(f"🔗 Re-attached to backend task {task_id}.")
            _follow_task(run, task_id)
        finally:
            run.finish()

    threading.Thread(target=_follow, daemon=True).start()
    return run


# ══════════════════════════════════════════════════════════════════════════════
# Progress file → result panel, player, redo list
# ══════════════════════════════════════════════════════════════════════════════

_RESULT_HEADERS: list[str] = ["#", "Chapter", "Status", "Duration", "Flagged chunks", "Last error"]


def _read_progress(path: str) -> dict[str, Any]:
    """The progress file as a dict (``{}`` when missing or unreadable)."""
    try:
        data = read_progress_file(path)
        return data if isinstance(data, dict) else {}
    except (FileNotFoundError, ValueError, OSError):
        return {}


def _chapter_entries(data: dict[str, Any], nums: list[int] | None = None) -> list[dict[str, Any]]:
    entries = [c for c in data.get("chapters") or [] if isinstance(c, dict)]
    if nums:
        wanted = {str(n) for n in nums}
        picked = [c for c in entries if str(c.get("num")) in wanted]
        if picked:
            return picked
    return entries


def _is_completed(entry: dict[str, Any]) -> bool:
    return entry.get("status") in ("completed", "complete")


def _result_rows(entries: list[dict[str, Any]]) -> list[list[Any]]:
    """Table rows: number, title, status, duration, flagged count, last error."""
    rows = []
    for entry in entries:
        duration = entry.get("duration")
        rows.append([
            entry.get("num", ""),
            str(entry.get("title", "")),
            str(entry.get("status", "pending")),
            _format_hms(duration) if duration else "",
            len(entry.get("flagged_chunks") or []),
            str(entry.get("last_error") or "") if not _is_completed(entry) else "",
        ])
    return rows


def _flagged_markdown(entries: list[dict[str, Any]]) -> str:
    """Lists chunks kept after failing verification, with text and reason."""
    lines = []
    for entry in entries:
        for item in entry.get("flagged_chunks") or []:
            if not isinstance(item, dict):
                continue
            text = html.escape(" ".join(str(item.get("text", "")).split()))
            lines.append(
                f"- **Chapter {entry.get('num')}**, chunk {item.get('chunk', '?')} — "
                f"{html.escape(str(item.get('reason', '') or 'failed verification'))}: “{text}”"
            )
    if not lines:
        return ""
    return (
        f"**{len(lines)} chunk(s) kept after failing verification — worth a listen:**\n\n"
        + "\n".join(lines)
    )


def _result_markdown(
    entries: list[dict[str, Any]],
    *,
    cancelled: bool = False,
    error: str = "",
    out_files: list[str] | None = None,
    finished_run: bool = True,
) -> str:
    """Headline and summary of a run (or of the book's saved progress)."""
    completed = [e for e in entries if _is_completed(e)]
    failed = [e for e in entries if e.get("status") == "failed"]
    total_seconds = sum(float(e.get("duration") or 0) for e in completed)
    summary = f"{len(completed)} completed, {len(failed)} failed, total audio {_format_hms(total_seconds)}"
    if len(entries) > len(completed) + len(failed):
        summary += f", {len(entries) - len(completed) - len(failed)} not done"

    if not finished_run:
        headline = "### 📚 Saved progress for this book"
    elif cancelled:
        headline = f"### ⛔ Cancelled — {len(completed)} of {len(entries)} chapter(s) finished"
    elif error and not completed:
        headline = f"### ❌ Generation failed\n\n{html.escape(error)}"
    elif failed:
        headline = f"### ⚠ {len(failed)} chapter(s) failed"
    elif error:
        headline = f"### ⚠ Finished with an error\n\n{html.escape(error)}"
    elif not completed and not (out_files or []):
        headline = "### ⚠ No output files were generated"
    else:
        headline = "### ✅ Generation complete"
    parts = [headline, summary]
    if failed and finished_run and not cancelled:
        parts.append("Press **Generate** again to retry the failed chapters; finished ones are skipped.")
    flagged = _flagged_markdown(entries)
    if flagged:
        parts.append(flagged)
    return "\n\n".join(parts)


def _chapter_file(output_dir: str, entry: dict[str, Any], fmt: str) -> str | None:
    """The audio file of a completed chapter, when it is on disk."""
    from audiobook_factory.filename_sanitizer import make_safe_filename

    try:
        name = make_safe_filename(str(entry.get("title", "")), int(entry.get("num")), output_dir, f".{fmt}")
    except (TypeError, ValueError):
        return None
    path = os.path.join(output_dir, name)
    return path if os.path.isfile(path) else None


def _player_choices(output_dir: str, data: dict[str, Any], out_files: list[str] | None = None) -> list[tuple[str, str]]:
    """``(label, path)`` of every finished audio file of the book."""
    fmt = str((data.get("settings") or {}).get("output_format") or "")
    formats = [fmt] if fmt else []
    formats += [f for f in _OUTPUT_FORMATS if f not in formats]
    paths: list[str] = []
    for entry in _chapter_entries(data):
        if not _is_completed(entry):
            continue
        for candidate in formats:
            path = _chapter_file(output_dir, entry, candidate)
            if path:
                paths.append(path)
                break
    for path in out_files or []:
        if isinstance(path, str) and os.path.isfile(path) and path.lower().endswith(_AUDIO_EXTENSIONS):
            paths.append(path)
    seen: set[str] = set()
    choices = []
    for path in paths:
        if path not in seen and _is_inside(path, output_dir):
            seen.add(path)
            choices.append((os.path.basename(path), path))
    return choices


def _redo_choices(data: dict[str, Any]) -> list[tuple[str, int]]:
    """``(label, chapter number)`` of the chapters already completed."""
    choices = []
    for entry in _chapter_entries(data):
        if _is_completed(entry) and str(entry.get("num", "")).isdigit():
            choices.append((f"{entry['num']}. {entry.get('title', '')}", int(entry["num"])))
    return choices


def _progress_html(
    fraction: float,
    *,
    elapsed: float | None = None,
    remaining: float | None = None,
    chapter: int = 0,
    total_chapters: int = 0,
    note: str = "",
) -> str:
    """Progress bar with the chapter counter, elapsed time and ETA."""
    percent = min(100.0, max(0.0, float(fraction) * 100.0))
    facts = [f"{percent:.1f}%"]
    if total_chapters:
        facts.append(f"chapter {min(max(chapter, 1), total_chapters)} of {total_chapters}")
    if elapsed is not None:
        facts.append(f"elapsed {_format_hms(elapsed)}")
    if remaining is not None:
        facts.append(f"about {_format_hms(remaining)} left")
    elif elapsed is not None and 0.0 < percent < 100.0:
        facts.append("estimating time left…")
    if note:
        facts.append(html.escape(note))
    return (
        '<div style="text-align:center;margin-bottom:5px;font-weight:bold;">'
        + " · ".join(facts)
        + f'</div><progress value="{percent:.2f}" max="100" style="width:100%;height:25px;"></progress>'
    )


def _chapters_from_progress(prog_path: str, nums: list[int]) -> list[ExtractedChapter]:
    """Chapters *nums* rebuilt from the text cached in the progress file."""
    wanted = {int(n) for n in nums}
    chapters = []
    for entry in _chapter_entries(_read_progress(prog_path)):
        num = entry.get("num")
        if not str(num).isdigit() or int(num) not in wanted or not str(entry.get("text") or "").strip():
            continue
        chapters.append(ExtractedChapter(
            num=int(num), title=str(entry.get("title", "")),
            text=entry["text"], sentences=list(entry.get("sentences") or []),
        ))
    return chapters if len(chapters) == len(wanted) else []


def _load_cached_chapters_if_available(
    prog_json_path: str,
    selected_chapters_labels: list[str] | None = None,
    log_fn: Callable[[str], None] | None = None,
    request: Any | None = None,
    session_id: str | None = None,
) -> list[ExtractedChapter] | None:
    """Chapter text cached in a progress file, when it covers the selection.

    Returns None when the file is missing, belongs to another session
    (multi-user mode), has a chapter without text, or lacks a selected
    chapter — the caller then parses the book.
    """
    if not os.path.exists(prog_json_path):
        return None
    if session_id is None and request is not None:
        session_id = _get_session_id(request)
    if session_id is not None and not _owns_progress(prog_json_path, session_id):
        return None
    try:
        data = read_progress_file(prog_json_path)
        ch_list = [c for c in data.get("chapters", []) if isinstance(c, dict)]
        if not ch_list or not all(str(c.get("text") or "").strip() for c in ch_list):
            return None

        selected_titles = _parse_chapter_titles(selected_chapters_labels) if selected_chapters_labels else None
        wanted = {_norm_title(st) for st in selected_titles if st} if selected_titles else None
        if wanted is not None:
            # A substring test made "Chapter 1" also pull in "Chapter 10", and
            # a partial hit silently replaced the selection: only use the cache
            # when it holds every chapter that was asked for.
            cached_titles = {_norm_title(c.get("title", "")) for c in ch_list}
            if not wanted <= cached_titles:
                return None

        chapters = []
        for position, c in enumerate(ch_list, 1):
            title = c.get("title", "")
            if wanted is not None and _norm_title(title) not in wanted:
                continue
            num = c.get("num")
            chapters.append(ExtractedChapter(
                num=int(num) if str(num).isdigit() else position,
                title=title,
                text=c["text"],
                sentences=c.get("sentences") or [],
            ))
        if chapters:
            if log_fn:
                log_fn("📦 Using cached chapter text from progress JSON (skipping book re-parsing).")
            return chapters
    except Exception as e:
        if log_fn:
            log_fn(f"⚠️ Could not load cached text from progress JSON: {e}")
    return None


def _apply_uploaded_progress(upload_path: str, dest_path: str) -> str:
    """Brings an uploaded progress JSON into a book's output folder.

    The upload used to be copied over the destination on every Generate
    click, resetting whatever had been generated since. Now it is copied
    only when the book has no progress file yet; otherwise the destination
    keeps its chapter statuses and only gains what it lacks (cached text,
    chapters it does not know, settings when it has none).

    Returns
    -------
    str
        ``"copied"``, ``"merged"`` or ``"skipped"``.
    """
    if not upload_path or not os.path.isfile(upload_path):
        return "skipped"
    if os.path.realpath(upload_path) == os.path.realpath(dest_path):
        return "skipped"
    if not os.path.exists(dest_path):
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        shutil.copy2(upload_path, dest_path)
        return "copied"
    uploaded = _read_progress(upload_path)
    current = _read_progress(dest_path)
    if not uploaded:
        return "skipped"
    if not current:
        shutil.copy2(upload_path, dest_path)
        return "copied"

    if isinstance(uploaded.get("settings"), dict) and uploaded["settings"]:
        current["settings"] = uploaded["settings"]
    by_title: dict[str, dict[str, Any]] = {}
    for entry in _chapter_entries(current):
        by_title.setdefault(_norm_title(entry.get("title")), entry)
    known_nums = {str(e.get("num")) for e in _chapter_entries(current)}
    chapters = list(_chapter_entries(current))
    for entry in _chapter_entries(uploaded):
        match = by_title.get(_norm_title(entry.get("title")))
        if match is not None:
            for key in ("text", "sentences"):
                if not match.get(key) and entry.get(key):
                    match[key] = entry[key]
        elif str(entry.get("num")) not in known_nums:
            chapters.append(entry)
            known_nums.add(str(entry.get("num")))
    chapters.sort(key=lambda e: int(e["num"]) if str(e.get("num", "")).isdigit() else 10**9)
    current["chapters"] = chapters
    for key in ("book_title", "book_path", "voice_file", "cover_image_b64"):
        if not current.get(key) and uploaded.get(key):
            current[key] = uploaded[key]
    write_progress_file(dest_path, current)
    return "merged"


def check_existing_progress(
    book_title: str,
    client_token: str = "",
    request: gr.Request = None,  # type: ignore[assignment]
) -> str:
    """Says whether this book already has progress (nothing across sessions in multi-user mode)."""
    if not book_title or not str(book_title).strip():
        return ""
    prog_path = _progress_path(book_title)
    if not os.path.exists(prog_path):
        return ""
    if not _owns_progress(prog_path, _get_session_id(request, client_token)):
        return ""
    try:
        data = read_progress_file(prog_path)
        chapters = data.get("chapters", [])
        completed = sum(1 for c in chapters if c.get("status") in ("completed", "complete"))
        total = len(chapters)
        share = f" ({completed / total * 100:.1f}%)" if total else ""
        return (
            f"### 🔄 Existing Progress Found!\n"
            f"- **Book Title:** {data.get('book_title', book_title)}\n"
            f"- **Progress:** {completed} / {total} chapters generated{share}\n"
            f"Generation will automatically resume from the last completed chapter."
        )
    except Exception as e:
        return f"⚠️ Found progress file but failed to read it: {e}"


# ══════════════════════════════════════════════════════════════════════════════
# Event handlers
# ══════════════════════════════════════════════════════════════════════════════
#
# Each handler returns one value per output component. The output lists are
# declared here as tuples of slot names; build_app() turns a slot name into
# its component, so a handler and its outputs cannot drift apart silently
# (tests/unit/test_app_events.py calls every handler and counts).

_PROVIDER_SLOTS: tuple[str, ...] = (
    "provider_info", "tts_model_name", "language", "tts_timbre", "tts_instruct",
    "clip_group", "voice_transcript", "preset_group", "design_group", "seed", "speed_note",
    "temperature", "top_p", "top_k", "repetition_penalty",
)
_MODEL_SLOTS: tuple[str, ...] = (
    "tts_timbre", "tts_instruct", "clip_group", "voice_transcript", "preset_group", "design_group",
)
_BOOK_SLOTS: tuple[str, ...] = (
    "chapter_panel", "page_panel", "book_note", "scan_status", "selected_chapters",
    "book_title", "author", "scan_state", "total_pages", "cover_image",
    "all_choices", "json_selected", "chapters_cache",
)
_TITLE_SLOTS: tuple[str, ...] = (
    "existing_progress", "result_md", "result_table", "player_dd", "redo_select",
)
_GEN_SLOTS: tuple[str, ...] = (
    "run_status", "log", "progress", "download_col", "download_files", "run_key",
    "result_md", "result_table", "player_dd", "redo_select", "log_file",
    "progress_upload", "chapters_cache",
)
_RESTORE_SLOTS: tuple[str, ...] = (
    # The first four positions are fixed: the security tests index them.
    "restore_status", "book_title", "book_file", "voice_file",
    "author", "output_format", "lufs",
    "tts_provider_name", *_PROVIDER_SLOTS,
    "tts_options", "options_epoch",
    "speed", "pause", "para_pause",
    "max_len", "pack_sentences", "normalize_speech_text",
    "verify_chunks", "verify_max_retries", "verify_asr_model", "verify_max_wer",
    "batch_size", "gpu_count", "vram_headroom_gb", "max_chapter_retries",
    "parallel_mode", "torch_compile", "quantization",
    "sample_rate", "bitrate_kbps", "channels", "true_peak",
    "export_text",
    "single_file_mode", "export_lrc", "export_srt", "export_vtt",
    "regen_missing", "resume_incomplete_chunks",
    "pronunciation_table", "epub_ocr", "page_ranges",
    "chapter_panel", "selected_chapters", "all_choices", "json_selected",
)


def _slots(slots: tuple[str, ...], updates: dict[str, Any]) -> tuple[Any, ...]:
    """One value per slot; slots without an update are left unchanged."""
    unknown = set(updates) - set(slots)
    if unknown:
        raise KeyError(f"Not an output of this handler: {sorted(unknown)}")
    return tuple(updates.get(name, gr.update()) for name in slots)


def _gen_out(**updates: Any) -> tuple[Any, ...]:
    return _slots(_GEN_SLOTS, updates)


# ── Engine / model selection ──────────────────────────────────────────────────

def _speed_note(info: Any) -> str:
    if info is not None and getattr(info, "supports_speed", False):
        return "<small>This engine changes speed itself.</small>"
    return "<small>This engine has no speed control: the audio is time-stretched afterwards (pitch preserved).</small>"


def _provider_updates(
    provider: Any,
    *,
    model: Any = None,
    language: Any = None,
    timbre: Any = None,
    sampling: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Updates for every control that depends on the selected engine."""
    info = _provider_info(provider)
    model = _resolve_model(info, model)
    voice = _voice_ui(provider, model)
    models = list(getattr(info, "models", ()) or ())
    languages = list(getattr(info, "languages", ()) or ())
    voices = list(getattr(info, "preset_voices", ()) or ())
    updates: dict[str, Any] = {
        "provider_info": _provider_markdown(provider),
        "tts_model_name": gr.update(choices=models, value=model if model in models else None, visible=bool(models)),
        "language": gr.update(
            choices=languages or list(_FALLBACK_LANGUAGES),
            value=_pick_language(info, language),
            allow_custom_value=not languages,
        ),
        "tts_timbre": gr.update(choices=voices, value=_pick_speaker(info, timbre), visible=voice.show_speaker),
        "tts_instruct": gr.update(visible=voice.show_instruct, info=_instruct_hint(info)),
        "clip_group": gr.update(visible=voice.show_clip),
        "voice_transcript": gr.update(visible=voice.show_transcript, info=_transcript_hint(info)),
        "preset_group": gr.update(visible=voice.show_preset),
        "design_group": gr.update(visible=voice.show_design),
        "seed": gr.update(visible=bool(getattr(info, "supports_seed", True))),
        "speed_note": _speed_note(info),
    }
    updates.update(_sampling_values(info, sampling))
    return updates


def on_provider_change(provider: Any, language: Any) -> tuple[Any, ...]:
    """The user picked another engine: re-render everything that depends on it.

    The sampling controls move to the engine's recommended values.
    """
    return _slots(_PROVIDER_SLOTS, _provider_updates(provider, language=language))


def on_model_change(provider: Any, model: Any) -> tuple[Any, ...]:
    """Shows the voice controls the chosen checkpoint uses."""
    info = _provider_info(provider)
    voice = _voice_ui(provider, model)
    return _slots(_MODEL_SLOTS, {
        "tts_timbre": gr.update(visible=voice.show_speaker),
        "tts_instruct": gr.update(visible=voice.show_instruct),
        "clip_group": gr.update(visible=voice.show_clip),
        "voice_transcript": gr.update(visible=voice.show_transcript, info=_transcript_hint(info)),
        "preset_group": gr.update(visible=voice.show_preset),
        "design_group": gr.update(visible=voice.show_design),
    })


# ── Book tab ──────────────────────────────────────────────────────────────────

def _supports_page_ranges(scan_res: Any) -> bool:
    """True for formats where ``extract(page_ranges=…)`` is honoured (PDF)."""
    return bool(getattr(scan_res, "supports_page_ranges", False))


def _cover_extension(data: bytes) -> str:
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return ".png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return ".webp"
    return ".jpg"


def on_book_upload(file_obj: Any, json_sel: Any) -> tuple[Any, ...]:
    """Scans the uploaded book and fills the chapter checklist.

    ``json_sel`` is the chapter selection restored from a progress JSON, if
    one was uploaded; it is applied once and then forgotten.
    """
    path = _file_path(file_obj)
    if not path:
        return _slots(_BOOK_SLOTS, {
            "chapter_panel": gr.update(visible=False),
            "page_panel": gr.update(visible=False),
            "book_note": "",
            "scan_status": "*Upload a file to begin.*",
            "selected_chapters": gr.update(choices=[], value=[]),
            "scan_state": None,
            "total_pages": "",
            "all_choices": {"values": [], "default": []},
            "chapters_cache": None,
        })

    try:
        result: ScanResult = scan(path, include_matter=True)
    except Exception as exc:
        logger.exception("Book scan failed")
        return _slots(_BOOK_SLOTS, {
            "chapter_panel": gr.update(visible=False),
            "page_panel": gr.update(visible=False),
            "book_note": "",
            "scan_status": f"❌ Could not read this book: {exc}",
            "selected_chapters": gr.update(choices=[], value=[]),
            "scan_state": None,
            "all_choices": {"values": [], "default": []},
            "chapters_cache": None,
        })

    status_parts = [f"**Type:** `{str(result.file_type).upper()}`"]
    if result.title:
        status_parts.append(f"**Title:** {result.title}")
    if result.author:
        status_parts.append(f"**Author:** {result.author}")
    if result.page_count:
        status_parts.append(f"**Pages:** {result.page_count}")
    warning = getattr(result, "warning", "")
    if warning:
        status_parts.append(f"⚠️ {warning}")

    cover_update: Any = gr.update()
    if result.cover_data:
        try:
            cover_path = _temp_file("cover" + _cover_extension(result.cover_data))
            with open(cover_path, "wb") as fh:
                fh.write(result.cover_data)
            cover_update = cover_path
        except OSError as exc:
            logger.debug("Cover extraction failed: %s", exc)

    choices, default = _chapter_choices(list(result.chapters or []))
    values = [value for _, value in choices]
    matter = len(values) - len(default)
    if json_sel:
        selected = _match_saved_selection(values, json_sel) or default
    else:
        selected = default

    note = ""
    if result.has_toc and values:
        status_parts.append(f"✅ **{len(default)} chapters found.**")
        if matter:
            note = (
                f"ℹ️ {matter} front/back-matter item(s) (title page, copyright, “also by”…) are "
                "listed but not ticked. Tick them if you want them read."
            )
    elif values:
        note = "ℹ️ No chapter structure was found — the book is read as one piece."
    else:
        note = "ℹ️ No chapters could be listed — the whole file is read as one piece."

    pages = _supports_page_ranges(result)
    return _slots(_BOOK_SLOTS, {
        "chapter_panel": gr.update(visible=bool(values)),
        "page_panel": gr.update(visible=pages),
        "book_note": note,
        "scan_status": "\n\n".join(status_parts),
        "selected_chapters": gr.update(choices=choices, value=selected),
        "book_title": result.title or gr.update(),
        "author": result.author or gr.update(),
        "scan_state": result,
        "total_pages": f"**Total pages:** {result.page_count}" if pages and result.page_count else "",
        "cover_image": cover_update,
        "all_choices": {"values": values, "default": default},
        "json_selected": None,
        "chapters_cache": None,
    })


def select_default_chapters(all_choices: Any) -> Any:
    """Ticks every real chapter (front/back matter stays unticked)."""
    return gr.update(value=list((all_choices or {}).get("default") or []))


def deselect_all_chapters() -> Any:
    """Unticks every chapter (Generate then asks for at least one)."""
    return gr.update(value=[])


def _language_tag(language: Any) -> str | None:
    """Language tag ("en", "fr") for the extractor's OCR, from a language name."""
    name = str(language or "").strip()
    if not name or name.lower() == "auto":
        return None
    try:
        from audiobook_factory import chunk_verifier
        codes = getattr(chunk_verifier, "_WHISPER_LANGUAGE_CODES", {})
    except Exception:
        codes = {}
    return codes.get(name.lower(), name)


def _extraction_message(exc: Exception) -> str:
    """User-facing text for a failed extraction."""
    if isinstance(exc, ExtractionError):
        return f"❌ {exc}"
    return f"❌ Could not read the book: {exc}"


def _extraction_key(
    path: str, selection: Any, page_ranges: Any, ocr: Any, language: Any = None,
) -> tuple[Any, ...] | None:
    """Identifies an extraction: same key, same chapters."""
    try:
        stat = os.stat(path)
    except OSError:
        return None
    return (
        os.path.realpath(path), stat.st_mtime_ns, stat.st_size,
        tuple(selection or ()), tuple(page_ranges or ()), bool(ocr),
        # The language only steers OCR.
        _language_tag(language) if ocr else None,
    )


def _extract_chapters(
    path: str,
    selection: Any,
    page_ranges: Any,
    ocr: Any,
    cache: Any,
    log_fn: Callable[[str], None] | None = None,
    language: Any = None,
) -> tuple[list[ExtractedChapter], dict[str, Any] | None, bool]:
    """Extracts the selected chapters, reusing this session's last extraction.

    Preview, Generate and Export each used to parse the whole book again.

    Returns
    -------
    tuple
        ``(chapters, cache entry, reused)``.
    """
    key = _extraction_key(path, selection, page_ranges, ocr, language)
    if key is not None and isinstance(cache, dict) and cache.get("key") == key and cache.get("chapters"):
        if log_fn:
            log_fn("📦 Reusing the chapters already extracted in this session.")
        return cache["chapters"], cache, True
    chapters, _cover = extract(
        path, selections=selection, enable_ocr=bool(ocr),
        page_ranges=page_ranges, log_fn=log_fn or (lambda _msg: None),
        language=_language_tag(language),
    )
    chapters = list(chapters or [])
    return chapters, ({"key": key, "chapters": chapters} if key is not None and chapters else None), False


def _selection_inputs(
    scan_res: Any, page_ranges_str: Any, all_choices: Any, selected: Any,
) -> tuple[Any, Any, str | None]:
    """Resolves what to extract from the Book tab.

    Returns
    -------
    tuple
        ``(selection, page_ranges, problem)``; *problem* is a message when
        nothing is selected.
    """
    page_ranges = _parse_page_ranges(page_ranges_str) if _supports_page_ranges(scan_res) else None
    has_checklist = bool((all_choices or {}).get("values"))
    selected = list(selected or [])
    if page_ranges:
        return None, page_ranges, None
    if has_checklist and not selected:
        # An empty selection used to mean "the whole book".
        return None, None, "⚠️ Select at least one chapter on the Book tab."
    return (_selection_from_labels(selected) if has_checklist else None), None, None


def on_preview(
    file_obj: Any, scan_res: Any, page_ranges_str: Any, epub_ocr: Any,
    chapters_cache: Any, all_choices: Any, selected: Any, language: Any,
) -> tuple[Any, Any, Any]:
    """Lists the selected chapters with their size, without generating audio."""
    path = _file_path(file_obj)
    if not path:
        return "⚠️ Please upload a book file first.", gr.update(visible=False), gr.update()
    selection, page_ranges, problem = _selection_inputs(scan_res, page_ranges_str, all_choices, selected)
    if problem:
        return problem, gr.update(visible=False), gr.update()
    try:
        chapters, cache, _reused = _extract_chapters(
            path, selection, page_ranges, epub_ocr, chapters_cache, language=language,
        )
    except Exception as exc:
        return _extraction_message(exc), gr.update(visible=False), gr.update()
    if not chapters:
        return "⚠️ No chapters found.", gr.update(visible=False), gr.update()
    rows = preview_chapters(chapters, queue.Queue())
    nums = _chapter_numbers(chapters)
    data = [[num, r["title"], r["chars"], r["words"], r["sentences"]] for num, r in zip(nums, rows)]
    total_words = sum(r["words"] for r in rows)
    return (
        f"🔍 {len(rows)} chapter(s), {total_words:,} words.",
        gr.update(value=data, visible=True),
        cache if cache is not None else gr.update(),
    )


def _first_paragraph(chapter: ExtractedChapter, limit: int = 420) -> str:
    """The first paragraph of a chapter that is prose, not a heading."""
    title = _norm_title(chapter.title)
    for block in re.split(r"\n\s*\n", chapter.text or ""):
        paragraph = " ".join(block.split())
        if len(paragraph.split()) < 8 or _norm_title(paragraph) == title:
            continue
        if len(paragraph) > limit:
            cut = max(paragraph.rfind(mark, 0, limit) for mark in (". ", "! ", "? ", "。", "” "))
            paragraph = paragraph[: cut + 1] if cut > limit // 3 else paragraph[:limit]
        return paragraph.strip()
    return " ".join((chapter.sentences or [])[:3]).strip()


def on_use_book_paragraph(
    file_obj: Any, scan_res: Any, page_ranges_str: Any, epub_ocr: Any,
    chapters_cache: Any, all_choices: Any, selected: Any, language: Any,
) -> tuple[Any, str]:
    """Puts the first real paragraph of the first selected chapter into the test box."""
    path = _file_path(file_obj)
    if not path:
        return gr.update(), "⚠️ Upload a book on the Book tab first."
    selection, page_ranges, problem = _selection_inputs(scan_res, page_ranges_str, all_choices, selected)
    if problem:
        return gr.update(), problem
    try:
        key = _extraction_key(path, selection, page_ranges, epub_ocr, language)
        if isinstance(chapters_cache, dict) and chapters_cache.get("key") == key and chapters_cache.get("chapters"):
            chapters = chapters_cache["chapters"]
        else:
            # Only the first chapter is needed, so only that one is converted.
            first = selection[:1] if selection else None
            first_range = page_ranges[:1] if page_ranges else None
            chapters, _ = extract(path, selections=first, enable_ocr=bool(epub_ocr),
                                  page_ranges=first_range, log_fn=lambda _msg: None,
                                  language=_language_tag(language))
    except Exception as exc:
        return gr.update(), _extraction_message(exc)
    for chapter in chapters or []:
        paragraph = _first_paragraph(chapter)
        if paragraph:
            return paragraph, f"📖 Test text taken from “{chapter.title}”."
    return gr.update(), "⚠️ No paragraph found in the selected chapter."


# ── Voice Studio ──────────────────────────────────────────────────────────────

def on_test_voice(text: Any, *cfg_vals: Any) -> tuple[Any, str]:
    """Speaks the test sentence with exactly the settings Generate would use."""
    cfg = _build_config(_ui_values(cfg_vals))
    problem = _voice_problem(cfg)
    if problem:
        return None, problem
    if not str(text or "").strip():
        return None, "⚠️ Enter some text to test."
    try:
        wav_bytes = _synthesize_preview(cfg, _spoken_text(str(text).strip(), cfg))
    except _UiError as exc:
        return None, str(exc)
    return _bytes_to_gradio_audio(wav_bytes), "✅ Preview ready!"


def on_audition(text: Any, *cfg_vals: Any) -> tuple[Any, str]:
    """Applies the current pronunciation fixes to *text* and plays the result."""
    cfg = _build_config(_ui_values(cfg_vals))
    text = str(text or "").strip()
    if not text:
        return None, "⚠️ Type a word or sentence to audition."
    problem = _voice_problem(cfg)
    if problem:
        return None, problem
    spoken = _spoken_text(text, cfg)
    try:
        wav_bytes = _synthesize_preview(cfg, spoken)
    except _UiError as exc:
        return None, str(exc)
    fixes = len(cfg.pronunciation_map)
    return _bytes_to_gradio_audio(wav_bytes), f"✅ {fixes} fix(es) applied. Spoken as: “{spoken}”"


def _design_voice(force: bool, cfg_vals: tuple[Any, ...]) -> tuple[Any, str, Any]:
    cfg = _build_config(_ui_values(cfg_vals))
    voice = _voice_ui(cfg.tts_provider_name, cfg.tts_model_name)
    if not voice.show_design:
        return None, "⚠️ This engine or model cannot design a voice from a description.", None
    if voice.mode == "design" and not cfg.tts_instruct:
        return None, "⚠️ Describe the narrator's voice in the voice instruction box first.", None
    try:
        wav_bytes, _rate, spoken = _with_preview_provider(
            cfg, lambda provider: provider.design_voice(cfg.tts_instruct or None, None, cfg.language, force=force),
        )
        path = _temp_file("designed_voice.wav")
        with open(path, "wb") as fh:
            fh.write(wav_bytes)
    except _UiError as exc:
        return None, str(exc), None
    except Exception as exc:
        logger.exception("Voice design failed")
        return None, f"❌ Voice design failed: {exc}", None
    note = "The whole book is read in this voice." if not force else "New voice rolled; the book will use this one."
    return path, f"✅ {note} Sample text: “{spoken}”", {"wav": path, "text": spoken, "provider": cfg.tts_provider_name}


def on_design_voice(*cfg_vals: Any) -> tuple[Any, str, Any]:
    """Plays the voice the engine designs from the instruction."""
    return _design_voice(False, cfg_vals)


def on_reroll_voice(*cfg_vals: Any) -> tuple[Any, str, Any]:
    """Designs the voice again (a different one when the seed is random)."""
    return _design_voice(True, cfg_vals)


def _save_preset(cfg: AudiobookConfig, voice_ref: Any, transcript: str | None) -> tuple[Any, str]:
    info = _provider_info(cfg.tts_provider_name)
    if info is None or not info.supports_voice_preset:
        return gr.update(), "⚠️ This engine has no voice presets."
    path = _temp_file(f"{_safe_title(cfg.tts_provider_name)}_voice_preset.pt")
    try:
        described = _with_preview_provider(
            cfg, lambda provider: provider.save_voice_preset(path, voice_ref, transcript=transcript),
        )
    except _UiError as exc:
        return gr.update(), str(exc)
    except Exception as exc:
        logger.exception("Saving the voice preset failed")
        return gr.update(), f"❌ Could not save the preset: {exc}"
    saved = described.get("path") if isinstance(described, dict) else None
    saved = saved if saved and os.path.isfile(str(saved)) else path
    if not os.path.isfile(saved):
        return gr.update(), "❌ The engine reported success but wrote no preset file."
    return (
        gr.update(value=saved, visible=True),
        "✅ Preset saved. Download it, and upload it as “Voice preset file” next time "
        "instead of the reference clip.",
    )


def on_save_preset(*cfg_vals: Any) -> tuple[Any, str]:
    """Builds a voice preset from the current reference clip and transcript."""
    cfg = _build_config(_ui_values(cfg_vals))
    voice = _voice_ui(cfg.tts_provider_name, cfg.tts_model_name)
    if not cfg.voice_file and not voice.show_design:
        return gr.update(), "⚠️ Upload a reference clip first."
    problem = None if not cfg.voice_file else _voice_problem(dataclasses.replace(cfg, voice_preset=""))
    if problem:
        return gr.update(), problem
    return _save_preset(cfg, cfg.voice_file or None, cfg.voice_transcript or None)


def on_save_designed_preset(design: Any, *cfg_vals: Any) -> tuple[Any, str]:
    """Saves the voice that was just designed as a preset."""
    cfg = _build_config(_ui_values(cfg_vals))
    wav_path = (design or {}).get("wav") if isinstance(design, dict) else None
    if not wav_path or not os.path.isfile(wav_path):
        return gr.update(), "⚠️ Design a voice sample first."
    if design.get("provider") != cfg.tts_provider_name:
        return gr.update(), "⚠️ That sample was designed with another engine. Design it again."
    with open(wav_path, "rb") as fh:
        wav_bytes = fh.read()
    return _save_preset(cfg, wav_bytes, str(design.get("text") or "") or None)


def on_preset_upload(preset_file: Any, provider: Any) -> str:
    """Describes an uploaded voice preset without loading a model."""
    path = _file_path(preset_file)
    if not path:
        return ""
    info = _provider_info(provider)
    name = getattr(info, "display_name", "this engine")
    reader = getattr(_provider_class(provider), "read_voice_preset_info", None)
    if callable(reader):
        try:
            described = reader(path) or {}
        except Exception as exc:
            # The message may quote the upload's temporary server path.
            reason = str(exc).replace(path, os.path.basename(path))
            return f"⚠️ Not a usable {name} preset: {reason}"
        made_for = described.get("model_id") or described.get("model") or ""
        suffix = f" (made with {made_for})" if made_for else ""
        return f"✅ Preset loaded{suffix}. The reference clip is not needed while it is set."
    return "✅ Preset loaded. The reference clip is not needed while it is set."


# ── Restore from a progress JSON ──────────────────────────────────────────────

def _restore_failure(message: str) -> list[Any]:
    return list(_slots(_RESTORE_SLOTS, {"restore_status": message}))


def on_progress_upload_handler(file_obj: Any) -> list[Any]:
    """Restores every control from an uploaded ``generation_progress.json``.

    Never puts a server path into a file component and never echoes one
    (BUG-R2-C1-A2-H2 / H3). Unknown keys are ignored, missing ones leave the
    control at its default, and a value that is not one of a dropdown's
    choices is replaced by the default — an out-of-list value would block
    every later submit.
    """
    path = _file_path(file_obj)
    if not path:
        return _restore_failure("")
    if not os.path.exists(path):
        return _restore_failure("❌ Uploaded progress file not found on disk.")
    try:
        data = read_progress_file(path)
    except (FileNotFoundError, ValueError) as exc:
        gr.Warning(str(exc))
        return _restore_failure(f"❌ **Failed to parse progress file**: {exc}")
    except Exception as exc:
        return _restore_failure(f"❌ **Failed to parse progress file**: {exc}")

    try:
        if not isinstance(data, dict):
            raise ValueError("the file does not contain a JSON object")
        settings = data.get("settings") if isinstance(data.get("settings"), dict) else {}
        ui_extra = data.get("ui") if isinstance(data.get("ui"), dict) else {}
        entries = _chapter_entries(data)
        base = AudiobookConfig()

        def saved(key: str) -> Any:
            value = settings.get(key)
            return getattr(base, key) if value is None else value

        notes: list[str] = []
        title = str(data.get("book_title") or settings.get("book_title") or "")

        # ── Engine and everything that depends on it ─────────────────────────
        provider = _canonical_provider(saved("tts_provider_name"))
        if provider not in _provider_keys():
            notes.append(f"The saved engine `{html.escape(str(saved('tts_provider_name')))}` is not available here; using the default.")
            provider = _default_provider()
        info = _provider_info(provider)
        model = _resolve_model(info, settings.get("tts_model_name"))
        voice = _voice_ui(provider, model)
        updates = _provider_updates(
            provider, model=model, language=saved("language"),
            timbre=settings.get("tts_timbre"), sampling=settings,
        )
        updates["tts_provider_name"] = gr.update(value=provider)
        updates["tts_instruct"] = gr.update(
            value=str(saved("tts_instruct") or ""), visible=voice.show_instruct, info=_instruct_hint(info),
        )
        updates["voice_transcript"] = gr.update(
            value=str(saved("voice_transcript") or ""), visible=voice.show_transcript, info=_transcript_hint(info),
        )
        updates["seed"] = gr.update(
            value=_num(saved("seed"), base.seed, int), visible=bool(getattr(info, "supports_seed", True)),
        )

        options = dict(settings.get("tts_options")) if isinstance(settings.get("tts_options"), dict) else {}
        declared = {o.key for o in getattr(info, "options", ())}
        if "nfe_step" in declared and "nfe_step" not in options and settings.get("nfe_step") not in (None, base.nfe_step):
            options["nfe_step"] = settings["nfe_step"]   # older files kept it as a top-level setting
        updates["tts_options"] = {provider: {k: v for k, v in options.items() if k in declared}}
        updates["options_epoch"] = time.time()

        # ── Plain settings ───────────────────────────────────────────────────
        def number(key: str, cast: Callable[[Any], Any] = float) -> Any:
            return gr.update(value=_clamp(key, saved(key), getattr(base, key), cast))

        def flag(key: str) -> Any:
            return gr.update(value=bool(saved(key)))

        updates.update({
            "author": gr.update(value=str(saved("author") or "")),
            "output_format": gr.update(value=_one_of(saved("output_format"), _OUTPUT_FORMATS, base.output_format)),
            "lufs": number("lufs", int),
            "speed": number("speed"),
            "pause": number("pause"),
            "para_pause": number("para_pause"),
            "max_len": number("max_len", int),
            "pack_sentences": flag("pack_sentences"),
            "normalize_speech_text": flag("normalize_speech_text"),
            "verify_chunks": gr.update(value=_one_of(saved("verify_chunks"), _VERIFY_CHOICES, base.verify_chunks)),
            "verify_max_retries": number("verify_max_retries", int),
            "verify_asr_model": gr.update(value=str(saved("verify_asr_model") or base.verify_asr_model)),
            "verify_max_wer": number("verify_max_wer"),
            "batch_size": number("batch_size", int),
            "gpu_count": number("gpu_count", int),
            "vram_headroom_gb": number("vram_headroom_gb"),
            "max_chapter_retries": number("max_chapter_retries", int),
            "parallel_mode": gr.update(value=_one_of(saved("parallel_mode"), _PARALLEL_MODES, base.parallel_mode)),
            "torch_compile": flag("torch_compile"),
            "quantization": gr.update(value=_one_of(saved("quantization"), _QUANTIZATIONS, base.quantization)),
            "sample_rate": gr.update(value=_one_of(saved("sample_rate"), _SAMPLE_RATES, base.sample_rate)),
            "bitrate_kbps": gr.update(value=_one_of(saved("bitrate_kbps"), _BITRATES, base.bitrate_kbps)),
            "channels": gr.update(value=_one_of(saved("channels"), _CHANNELS, base.channels)),
            "true_peak": number("true_peak"),
            # "Force re-process" is a one-off instruction and is not restored.
            "export_text": flag("export_text"),
            "single_file_mode": flag("single_file_mode"),
            "export_lrc": flag("export_lrc"),
            "export_srt": flag("export_srt"),
            "export_vtt": flag("export_vtt"),
            "regen_missing": flag("regen_missing"),
            "resume_incomplete_chunks": flag("resume_incomplete_chunks"),
            "pronunciation_table": gr.update(value=_pronunciation_rows(settings.get("pronunciation_map"))),
            "epub_ocr": gr.update(value=bool(ui_extra.get("epub_ocr", settings.get("epub_ocr", False)))),
            "page_ranges": gr.update(value=str(ui_extra.get("page_ranges") or "")),
        })

        # ── Chapters cached in the file ──────────────────────────────────────
        values = [
            _chapter_value(e.get("num", i), str(e.get("title", "")), len(str(e.get("text") or "").split()))
            for i, e in enumerate(entries, 1)
        ]
        saved_selection = [str(s) for s in settings.get("selected_chapters") or [] if isinstance(s, str)]
        selected = (_match_saved_selection(values, saved_selection) if saved_selection else []) or values
        updates["chapter_panel"] = gr.update(visible=bool(values)) if values else gr.update()
        updates["selected_chapters"] = gr.update(choices=[(v, v) for v in values], value=selected) if values else gr.update()
        updates["all_choices"] = {"values": values, "default": values} if values else gr.update()
        updates["json_selected"] = saved_selection or None

        completed = sum(1 for e in entries if _is_completed(e))
        need = ["your book file (Book tab)"]
        if voice.show_clip:
            need.append("the narrator voice clip or a voice preset (Voice Studio)")
        message = (
            "### ✅ Progress File Loaded Successfully!\n"
            f"- **Book:** {title}\n"
            f"- **Total Chapters:** {len(entries)}\n"
            f"- **Completed:** {completed}\n\n"
            "All settings were restored. Files are never restored from a JSON: upload "
            + " and ".join(need) + " again."
        )
        if notes:
            message += "\n\n" + "\n".join(f"- ⚠️ {n}" for n in notes)
        updates["restore_status"] = message
        updates["book_title"] = gr.update(value=title) if title else gr.update()
        # book_file / voice_file: never preset server paths (BUG-R2-C1-A2-H2/H3).
        return list(_slots(_RESTORE_SLOTS, updates))
    except Exception as exc:
        logger.exception("Restoring the progress file failed")
        return _restore_failure(f"❌ Failed to parse progress file: {exc}")


# ── Export Config JSON ────────────────────────────────────────────────────────

_ONE_SHOT_SETTINGS: tuple[str, ...] = ("force_reprocess", "redo_chapters", "preview_mode")
_RESUME_FIELDS: tuple[str, ...] = (
    "status", "completed_chunks", "retry_count", "last_error", "duration", "flagged_chunks",
)


def _merge_chapter_entries(previous: list[dict[str, Any]], chapters: list[ExtractedChapter]) -> list[dict[str, Any]]:
    """Chapter list for an exported config.

    Each extracted chapter keeps the resume fields of an existing entry with
    the same title (preferring the one with the same number), and entries
    for chapters outside the current selection are kept as they are.
    """
    by_title: dict[str, list[dict[str, Any]]] = {}
    for entry in previous:
        by_title.setdefault(_norm_title(entry.get("title")), []).append(entry)
    nums = _chapter_numbers(chapters)
    used: set[int] = set()
    merged: list[dict[str, Any]] = []
    for num, chapter in zip(nums, chapters):
        candidates = [e for e in by_title.get(_norm_title(chapter.title), []) if id(e) not in used]
        match = next((e for e in candidates if str(e.get("num")) == str(num)), candidates[0] if candidates else None)
        entry: dict[str, Any] = {
            "num": num, "title": chapter.title, "status": "pending", "completed_chunks": [],
            "text": chapter.text, "sentences": chapter.sentences,
        }
        if match is not None:
            used.add(id(match))
            for key in _RESUME_FIELDS:
                if key in match:
                    entry[key] = match[key]
        merged.append(entry)
    taken = {str(num) for num in nums}
    for entry in previous:
        if id(entry) not in used and str(entry.get("num")) not in taken:
            merged.append(entry)
            taken.add(str(entry.get("num")))
    merged.sort(key=lambda e: int(e["num"]) if str(e.get("num", "")).isdigit() else 10**9)
    return merged


def on_export_config(
    request: gr.Request,
    file_obj: Any, scan_res: Any, page_ranges_str: Any, epub_ocr: Any,
    chapters_cache: Any, all_choices: Any, client_token: Any,
    *cfg_vals: Any,
) -> tuple[Any, Any, Any, Any]:
    """Writes a self-contained ``generation_progress.json`` without generating.

    The file holds every ``AudiobookConfig`` field and the chapter text, so
    ``python cli.py <file>`` can run it without the UI.
    """
    def fail(message: str) -> tuple[Any, Any, Any, Any]:
        return message, gr.update(visible=False), gr.update(open=True), gr.update()

    try:
        ui = _ui_values(cfg_vals)
        session_id = _get_session_id(request, client_token)
        path = _file_path(file_obj)
        book_out = _book_output_dir(ui.get("book_title"))
        prog_path = os.path.join(book_out, _PROGRESS_FILE_NAME)

        if not _owns_progress(prog_path, session_id):
            return fail("⚠️ Access denied: Progress file for this book title is owned by another session.")
        if _active_run_for(book_out) is not None:
            return fail("⚠️ This book is being generated. Export the config after the run finishes.")

        selected = list(ui.get("selected_chapters") or [])
        selection, page_ranges, problem = _selection_inputs(scan_res, page_ranges_str, all_choices, selected)
        if problem:
            return fail(problem)

        cache_update: Any = gr.update()
        chapters: list[ExtractedChapter] | None = None
        if path and os.path.exists(path):
            try:
                chapters, cache, _reused = _extract_chapters(
                    path, selection, page_ranges, epub_ocr, chapters_cache, language=ui.get("language"),
                )
            except Exception as exc:
                return fail(_extraction_message(exc))
            if cache is not None:
                cache_update = cache
        else:
            chapters = _load_cached_chapters_if_available(prog_path, selected or None, session_id=session_id)
            if not chapters:
                return fail("⚠️ Please upload a book file first.")
        if not chapters:
            return fail("⚠️ No chapters extracted from the book.")

        os.makedirs(book_out, exist_ok=True)
        cfg = _build_config(ui, book_path=path, output_dir=book_out)
        settings = dataclasses.asdict(cfg)
        # One-off instructions are not part of a saved config: a file that
        # carried "force re-process" would wipe the book on every CLI run.
        default = AudiobookConfig()
        for one_shot in _ONE_SHOT_SETTINGS:
            settings[one_shot] = getattr(default, one_shot)

        # The cover is embedded once, at the top level.
        cover_bytes: bytes | None = None
        if cfg.cover_image and os.path.isfile(cfg.cover_image):
            try:
                with open(cfg.cover_image, "rb") as fh:
                    cover_bytes = fh.read()
            except OSError:
                cover_bytes = None
        if not cover_bytes:
            cover_bytes = getattr(scan_res, "cover_data", None)
        cover_b64 = ""
        if cover_bytes:
            local_cover = os.path.join(book_out, "cover" + _cover_extension(cover_bytes))
            try:
                with open(local_cover, "wb") as fh:
                    fh.write(cover_bytes)
                settings["cover_image"] = local_cover
            except OSError as exc:
                logger.debug("Could not save the cover: %s", exc)
            cover_b64 = base64.b64encode(cover_bytes).decode("ascii")

        existing = _read_progress(prog_path)
        data = dict(existing)
        data.pop("generation_summary", None)
        data["book_title"] = cfg.book_title
        data["book_path"] = path
        data["voice_file"] = cfg.voice_file
        data["cover_image_b64"] = cover_b64 or existing.get("cover_image_b64", "")
        data["settings"] = settings
        data["ui"] = {"epub_ocr": bool(epub_ocr), "page_ranges": str(page_ranges_str or "")}
        data["chapters"] = _merge_chapter_entries(_chapter_entries(existing), chapters)
        write_progress_file(prog_path, data)
        _register_progress_owner(prog_path, session_id)

        return (
            f"✅ **Config exported!** {len(chapters)} chapter(s) with text, "
            f"{len(data['chapters'])} in the file.\n\n"
            f"Saved to `{os.path.relpath(prog_path, _ROOT)}`.\n\n"
            f"To generate without the UI, run:\n"
            f"```\npython cli.py \"{prog_path}\"\n```",
            gr.update(value=prog_path, visible=True),
            gr.update(open=True),
            cache_update,
        )
    except Exception as exc:
        logger.exception("Config export failed")
        return fail(f"❌ **Config export failed:** {exc}")


# ── Result panel, player, redo list ───────────────────────────────────────────

def _book_updates(output_dir: str, run: _Run | None = None) -> dict[str, Any]:
    """Result panel, player and redo list from the book's progress file."""
    prog_path = os.path.join(output_dir, _PROGRESS_FILE_NAME)
    data = _read_progress(prog_path)
    out_files = [p for p in (run.out_files if run else []) if isinstance(p, str) and os.path.exists(p)]
    if run is not None:
        entries = _chapter_entries(data, run.chapter_nums)
        markdown = _result_markdown(entries, cancelled=run.cancelled, error=run.error, out_files=out_files)
    else:
        entries = _chapter_entries(data)
        markdown = _result_markdown(entries, finished_run=False) if entries else ""
    rows = _result_rows(entries)
    player = _player_choices(output_dir, data, out_files)
    return {
        "result_md": markdown,
        "result_table": gr.update(value=rows, visible=bool(rows)),
        "player_dd": gr.update(choices=player, value=player[0][1] if player else None, visible=bool(player)),
        "redo_select": gr.update(choices=_redo_choices(data), value=[]),
    }


def on_title_commit(request: gr.Request, book_title: Any, client_token: Any) -> tuple[Any, ...]:
    """The book title was confirmed: show what already exists for that book.

    Runs when the title box loses focus or Enter is pressed, not on every
    keystroke (each call reads and parses the whole progress file).
    """
    title = str(book_title or "").strip()
    session_id = _get_session_id(request, client_token)
    output_dir = _book_output_dir(title)
    prog_path = os.path.join(output_dir, _PROGRESS_FILE_NAME)
    if not title or not os.path.exists(prog_path) or not _owns_progress(prog_path, session_id):
        return _slots(_TITLE_SLOTS, {
            "existing_progress": "",
            "result_md": "",
            "result_table": gr.update(value=[], visible=False),
            "player_dd": gr.update(choices=[], value=None, visible=False),
            "redo_select": gr.update(choices=[], value=[]),
        })
    updates = _book_updates(output_dir)
    updates["existing_progress"] = check_existing_progress(title, client_token, request)
    return _slots(_TITLE_SLOTS, updates)


def on_player_select(path: Any) -> Any:
    """Loads the chosen chapter file into the in-page player."""
    path = _file_path(path)
    if not path or not os.path.isfile(path) or not _is_inside(path, _OUTPUT_DIR):
        return gr.update(value=None)
    return gr.update(value=path)


# ── Generate / Redo / Re-attach / Cancel ──────────────────────────────────────

def _stream_run(run: _Run, first: dict[str, Any] | None = None) -> Iterator[tuple[Any, ...]]:
    """Yields UI updates for *run* until it ends, then its result.

    Yields only when something changed (new log lines, progress) or once a
    second for the clock; the log box receives a bounded tail, not the whole
    growing log.
    """
    eta = _EtaEstimator()
    pending = dict(first or {})
    last_lines, last_fraction, last_emit = -1, -1.0, 0.0
    while True:
        done = run.done
        line_count, fraction = run.snapshot()
        now = time.monotonic()
        eta.update(fraction, now)
        log_changed = line_count != last_lines
        if pending or log_changed or fraction != last_fraction or now - last_emit >= _STREAM_HEARTBEAT_SEC:
            total = len(run.chapters)
            updates: dict[str, Any] = {
                "progress": _progress_html(
                    fraction, elapsed=run.elapsed,
                    remaining=None if done else eta.remaining(now),
                    chapter=int(fraction * total) + 1, total_chapters=total,
                    note="cancelling…" if run.cancel.is_cancelled and not done else "",
                ),
                "run_key": run.key,
            }
            if log_changed:
                updates["log"] = run.tail()
            updates.update(pending)
            pending = {}
            yield _gen_out(**updates)
            last_lines, last_fraction, last_emit = line_count, fraction, now
        if done:
            break
        time.sleep(_STREAM_POLL_SEC)
    yield _gen_out(**_final_updates(run))


def _final_updates(run: _Run) -> dict[str, Any]:
    """Everything the Generate tab shows once a run has ended."""
    output_dir = run.output_dir
    out_files = [p for p in run.out_files if isinstance(p, str) and os.path.exists(p)]
    if out_files and not os.path.exists(os.path.join(output_dir, _PROGRESS_FILE_NAME)):
        # The backend may keep its output under another base directory.
        output_dir = os.path.dirname(out_files[0])
    updates = _book_updates(output_dir, run)
    headline = updates["result_md"].split("\n", 1)[0].lstrip("# ").strip()

    downloads = list(out_files)
    for name in (_PROGRESS_FILE_NAME, _LOG_FILE_NAME):
        candidate = os.path.join(output_dir, name)
        if os.path.exists(candidate):
            downloads.append(candidate)
    log_path = os.path.join(run.output_dir, _LOG_FILE_NAME)
    succeeded = headline.startswith("✅")
    # The pipeline reports 100 % when it stops for any reason; for a run that
    # did not succeed, show the share of chapters that really finished.
    entries = _chapter_entries(_read_progress(os.path.join(output_dir, _PROGRESS_FILE_NAME)), run.chapter_nums)
    fraction = sum(1 for e in entries if _is_completed(e)) / len(entries) if entries else 0.0
    updates.update({
        "run_status": f"**{headline}**",
        "log": run.tail() + f"\n\n{headline}",
        "progress": _progress_html(
            1.0 if succeeded else fraction, elapsed=run.elapsed,
            note="done" if succeeded else "cancelled" if run.cancelled else "stopped",
        ),
        "download_col": gr.update(visible=bool(downloads)),
        "download_files": downloads,
        "run_key": None,
        "log_file": gr.update(value=log_path, visible=True) if os.path.exists(log_path) else gr.update(),
    })
    if run.extraction_key is not None and run.chapters:
        updates["chapters_cache"] = {"key": run.extraction_key, "chapters": run.chapters}
    return updates


def _generate(
    request: Any,
    redo_nums: list[int] | None,
    file_obj: Any, scan_res: Any, page_ranges_str: Any, epub_ocr: Any,
    progress_file_obj: Any, chapters_cache: Any, all_choices: Any, client_token: Any,
    cfg_vals: tuple[Any, ...],
) -> Iterator[tuple[Any, ...]]:
    """Validates the request, starts (or re-joins) the run and streams it."""
    ui = _ui_values(cfg_vals)
    session_id = _get_session_id(request, client_token)
    book_path = _file_path(file_obj)
    book_out = _book_output_dir(ui.get("book_title"))
    prog_path = os.path.join(book_out, _PROGRESS_FILE_NAME)
    is_redo = redo_nums is not None

    # Validation messages go to the status line only. They used to be sent
    # as "hide" updates into the progress bar, which then stayed hidden.
    if not _owns_progress(prog_path, session_id):
        yield _gen_out(run_status="⚠️ Access denied: Progress file for this book title is owned by another session.")
        return

    running = _active_run_for(book_out)
    if running is not None:
        # Same book, already running (page reloaded, second click, second
        # tab): show that run instead of starting a duplicate.
        note = f"🔗 **{running.book_title}** is already being generated — showing its progress."
        yield from _stream_run(running, first={"run_status": note})
        return

    if is_redo:
        redo_nums = sorted({int(n) for n in (redo_nums or []) if str(n).lstrip("-").isdigit()})
        if not redo_nums:
            yield _gen_out(run_status="⚠️ Choose at least one completed chapter to redo.")
            return
    selected = list(ui.get("selected_chapters") or [])
    selection, page_ranges, problem = _selection_inputs(scan_res, page_ranges_str, all_choices, selected)
    if problem and not is_redo:
        yield _gen_out(run_status=problem)
        return

    cfg = _build_config(ui, book_path=book_path, output_dir=book_out, redo_chapters=redo_nums if is_redo else None)
    problem = _voice_problem(cfg)
    if problem:
        yield _gen_out(run_status=problem)
        return

    backend = is_api_healthy()
    if backend:
        task_id = _find_backend_task(cfg.book_title)
        if task_id:
            run = _attach_backend_task(cfg, session_id, task_id)
            note = f"🔗 The backend is already generating **{cfg.book_title}** — showing its progress."
            yield from _stream_run(run, first={"run_status": note})
            return
    else:
        busy = _active_local_run()
        if busy is not None:
            yield _gen_out(run_status=(
                f"⚠️ **{busy.book_title}** is being generated. Wait for it to finish or cancel it first."
            ))
            return

    # ── Progress file brought by the user ────────────────────────────────────
    upload_path = _file_path(progress_file_obj)
    first: dict[str, Any] = {}
    os.makedirs(book_out, exist_ok=True)
    if upload_path:
        try:
            outcome = _apply_uploaded_progress(upload_path, prog_path)
            logger.info("Uploaded progress file %s into %s", outcome, prog_path)
        except Exception as exc:
            logger.warning("Could not use the uploaded progress file: %s", exc)
        # Used once: a second Generate click must not apply the stale upload again.
        first["progress_upload"] = gr.update(value=None)

    use_saved_text = is_redo and not cfg.single_file_mode
    if not book_path:
        if use_saved_text:
            has_text = bool(_chapters_from_progress(prog_path, redo_nums or []))
        else:
            has_text = bool(_load_cached_chapters_if_available(prog_path, selected or None, session_id=session_id))
        if not has_text:
            yield _gen_out(run_status="⚠️ Please upload a book file first.", **first)
            return
    _register_progress_owner(prog_path, session_id)

    language = cfg.language
    extraction_key = _extraction_key(book_path, selection, page_ranges, epub_ocr, language) if book_path else None

    def load_chapters(run: _Run) -> list[ExtractedChapter]:
        log = run.add_log
        try:
            if use_saved_text:
                chapters = _chapters_from_progress(prog_path, redo_nums or [])
                if chapters:
                    log("📦 Using the chapter text saved with the book.")
                    return chapters
                if not book_path:
                    return []
                tag = _language_tag(language)
                if page_ranges:
                    chapters, _ = extract(book_path, enable_ocr=bool(epub_ocr), page_ranges=page_ranges,
                                          log_fn=log, language=tag)
                    wanted = set(redo_nums or [])
                    return [c for n, c in zip(_chapter_numbers(chapters), chapters) if n in wanted]
                chapters, _ = extract(book_path, selections=list(redo_nums or []), enable_ocr=bool(epub_ocr),
                                      log_fn=log, language=tag)
                return list(chapters or [])
            if not cfg.force_reprocess:
                if book_path and isinstance(chapters_cache, dict) and chapters_cache.get("key") == extraction_key \
                        and chapters_cache.get("chapters"):
                    log("📦 Reusing the chapters already extracted in this session.")
                    return chapters_cache["chapters"]
                if not page_ranges:
                    cached = _load_cached_chapters_if_available(prog_path, selected or None, log, session_id=session_id)
                    if cached:
                        return cached
            if not book_path:
                return []
            chapters, _cache, _reused = _extract_chapters(
                book_path, selection, page_ranges, epub_ocr, None, log, language=language,
            )
            run.extraction_key = extraction_key
            return chapters
        except Exception as exc:
            raise _UiError(_extraction_message(exc).lstrip("❌ ")) from exc

    run = _start_run(cfg, session_id, load_chapters)
    first["run_status"] = (
        f"🔁 Redoing chapter(s) {', '.join(str(n) for n in redo_nums or [])} of **{cfg.book_title}**…"
        if is_redo else f"🎧 Generating **{cfg.book_title}**…"
    )
    first["result_md"] = ""
    first["result_table"] = gr.update(visible=False)
    yield from _stream_run(run, first=first)


def on_generate(
    request: gr.Request,
    file_obj: Any, scan_res: Any, page_ranges_str: Any, epub_ocr: Any,
    progress_file_obj: Any, chapters_cache: Any, all_choices: Any, client_token: Any,
    *cfg_vals: Any,
) -> Iterator[tuple[Any, ...]]:
    """Generates the audiobook for the current settings and streams progress."""
    yield from _generate(
        request, None, file_obj, scan_res, page_ranges_str, epub_ocr,
        progress_file_obj, chapters_cache, all_choices, client_token, cfg_vals,
    )


def on_redo(
    request: gr.Request,
    redo_nums: Any,
    file_obj: Any, scan_res: Any, page_ranges_str: Any, epub_ocr: Any,
    progress_file_obj: Any, chapters_cache: Any, all_choices: Any, client_token: Any,
    *cfg_vals: Any,
) -> Iterator[tuple[Any, ...]]:
    """Regenerates only the chosen, already completed chapters."""
    yield from _generate(
        request, list(redo_nums or []), file_obj, scan_res, page_ranges_str, epub_ocr,
        progress_file_obj, chapters_cache, all_choices, client_token, cfg_vals,
    )


def on_page_load(client_token: Any) -> Any:
    """Gives a first-time browser its session token; a known browser keeps its own."""
    token = ensure_client_token(client_token)
    return gr.update() if token == client_token else token


def on_reattach(request: gr.Request, client_token: Any) -> Iterator[tuple[Any, ...]]:
    """Page (re)loaded: if a run is still going, show it again.

    The run lives on the server; only the page that was showing it is gone.
    """
    runs = _visible_runs(_get_session_id(request, client_token))
    if not runs:
        yield (*_gen_out(), gr.update(), gr.update())
        return
    run = runs[-1]
    note = f"🔗 Re-attached to the running generation of **{run.book_title}**. Cancel works as usual."
    extra: tuple[Any, Any] = (
        run.book_title,
        f"⏳ **{run.book_title}** is being generated — live progress and Cancel are on the Generate tab.",
    )
    for out in _stream_run(run, first={"run_status": note}):
        yield (*out, *extra)
        extra = (gr.update(), gr.update())
    yield (*_gen_out(), gr.update(), "")


def on_cancel(request: gr.Request, run_key: Any, book_title: Any, client_token: Any) -> str:
    """Stops the run this page is showing (or this book's, or this session's)."""
    session_id = _get_session_id(request, client_token)
    visible = _visible_runs(session_id)
    targets = [r for r in visible if r.key == run_key]
    if not targets and str(book_title or "").strip():
        key = _run_key(_book_output_dir(book_title))
        targets = [r for r in visible if r.key == key]
    if not targets:
        targets = visible
    if not targets:
        return "ℹ️ Nothing is being generated."
    for run in targets:
        run.request_cancel()
    names = ", ".join(f"**{r.book_title}**" for r in targets)
    return f"⛔ Cancellation requested for {names}. The current chunk finishes first."


def on_zip(files: Any) -> Any:
    """Packs the listed output files into one ZIP for download."""
    paths = [_file_path(f) for f in (files if isinstance(files, list) else [files] if files else [])]
    paths = [p for p in paths if p]
    if not paths:
        return gr.update(visible=False)
    return gr.update(value=_make_zip(paths), visible=True)


# ══════════════════════════════════════════════════════════════════════════════
# Gradio App
# ══════════════════════════════════════════════════════════════════════════════

_CSS: str = """
.header-banner { text-align:center; padding: 20px 0 10px; }
.header-banner h1 { font-size: 2.4rem; font-weight: 800; letter-spacing: -1px; }
.header-banner p  { color: #6b7280; font-size: 1rem; }
.warn-box { background: rgba(234,179,8,0.12); border-radius:8px;
            border:1px solid #ca8a04; padding:12px; font-size:0.9rem; }
"""

_FORMAT_GUIDE: str = """
| Feature          | FLAC                       | MP3               | WAV                       | M4B        |
| ---------------- | -------------------------- | ----------------- | ------------------------- | ---------- |
| **Cover Art**    | ✅ Yes                      | ✅ Yes             | ⚠️ Limited / inconsistent | ✅ Yes      |
| **Title / Artist / Album** | ✅ Yes            | ✅ Yes             | ⚠️ Limited                | ✅ Yes      |
| **Track Number** | ✅ Yes                      | ✅ Yes             | ⚠️ Limited                | ✅ Yes      |
| **Lyrics**       | ⚠️ Possible (not standard) | ✅ Yes             | ❌ No                      | ⚠️ Limited |
| **Chapter markers (single file)** | ❌ No      | ✅ Yes (ID3)       | ❌ No                      | ✅ Yes      |

- **One file for the whole book**: choose **M4B** and tick “Combine into a single file”.
- **Lossless**: choose **FLAC** (or WAV if you do not need metadata).
- **Plays everywhere**: choose **MP3**.
"""

_AUDIOBOOKSHELF_GUIDE: str = """
## Adding the book to Audiobookshelf

[Audiobookshelf](https://www.audiobookshelf.org/) is a self-hosted audiobook server.

1. **Where the files are**: AudiobookMaker writes each book to `audiobook_output/{Title}/`.
   Chapter files are numbered, so they sort in reading order; the cover is embedded in each file.
2. **Folder layout**: Audiobookshelf expects `{Author}/{Title}/`. Create a folder named after the
   author in your library and move the `{Title}` folder into it. (AudiobookMaker does not create
   the author folder.)
3. **Scan**: press *Scan* in Audiobookshelf. It reads the title, author and cover from the files.
4. **Chapters**: a single-file **M4B** or **MP3** carries chapter markers. With one file per
   chapter, Audiobookshelf orders the chapters by file name.
5. **Text**: `.lrc` files (when enabled) sit next to the audio and show timed text in players
   that support them.
"""


def _theme() -> Any:
    return gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="purple",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
    )


def _style_kwargs(target: Callable[..., Any]) -> dict[str, Any]:
    """``theme`` / ``css`` arguments for whichever of Blocks() and launch() takes them.

    Gradio 5 takes them in ``Blocks()``, Gradio 6 in ``launch()``.
    """
    try:
        params = inspect.signature(target).parameters
    except (TypeError, ValueError):
        return {}
    return {"theme": _theme(), "css": _CSS} if "theme" in params and "css" in params else {}


def _gpu_badge() -> str:
    try:
        from audiobook_factory.gpu_pool import GPUDetector
        devices = GPUDetector.detect_devices()
    except Exception as exc:
        logger.debug("GPU detection failed: %s", exc)
        return "GPU: unknown"
    if devices == ["cpu"]:
        return "GPU: none — CPU (slow)"
    if len(devices) == 1:
        return f"GPU: {devices[0]}"
    return f"GPU: {' + '.join(devices)} ({len(devices)}× parallel)"


def _slider(key: str, label: str, value: Any, **kwargs: Any) -> Any:
    low, high, step = _RANGES[key]
    return gr.Slider(label=label, minimum=low, maximum=high, step=step,
                     value=min(high, max(low, value)), **kwargs)


def build_app() -> gr.Blocks:
    """Creates the UI and wires every component to its module-level handler."""
    base = AudiobookConfig()
    pp = PreprocessConfig()
    provider0 = _default_provider()
    info0 = _provider_info(provider0)
    model0 = _resolve_model(info0, None)
    voice0 = _voice_ui(provider0, model0)
    sampling0 = _sampling_values(info0)
    languages0 = list(getattr(info0, "languages", ()) or ())
    voices0 = list(getattr(info0, "preset_voices", ()) or ())
    models0 = list(getattr(info0, "models", ()) or ())
    can_transcribe = _transcription_available()

    # slot name → component, for every handler input / output
    C: dict[str, Any] = {}

    with gr.Blocks(title="AudiobookMaker", **_style_kwargs(gr.Blocks.__init__)) as demo:

        gr.HTML(f"""
        <div class="header-banner">
          <h1>📖 AudiobookMaker</h1>
          <p>Turn a book into an audiobook read by one narrator voice.</p>
          <span style="background: rgba(128,128,128,0.18); padding: 4px 12px; border-radius: 12px; font-size: 0.9em; font-weight: bold; display: inline-block; margin-top: 6px;">{html.escape(_gpu_badge())}</span>
          <p style="font-size: 0.8em; opacity: 0.75; margin-top: 8px;">Free software under AGPL-3.0-or-later ·
            <a href="{html.escape(SOURCE_URL)}" target="_blank" rel="noopener">source code</a></p>
        </div>
        """)

        # ── Session state ─────────────────────────────────────────────────────
        C["scan_state"] = gr.State(None)                                # ScanResult
        C["chapters_cache"] = gr.State(None)                            # last extraction of this session
        C["run_key"] = gr.State(None)                                   # run shown by this page
        C["all_choices"] = gr.State({"values": [], "default": []})      # chapter checklist values
        C["json_selected"] = gr.State(None)                             # selection restored from JSON
        C["tts_options"] = gr.State({})                                 # {engine: {option: value}}
        C["options_epoch"] = gr.State(0.0)                              # bumped to re-render engine options
        preproc_state = gr.State(None)                                  # processed voice bytes
        design_state = gr.State(None)                                   # last designed voice
        if hasattr(gr, "BrowserState"):
            client_token = gr.BrowserState("", storage_key="abm_client_token", secret=_browser_state_secret())
        else:
            client_token = gr.State("")

        active_banner = gr.Markdown("")

        # ── Resume from Progress JSON (top-level, before tabs) ────────────────
        with gr.Accordion("🔄 Resume from Progress JSON", open=False):
            gr.HTML(
                '<div class="warn-box">💡 <strong>Start here if you are resuming.</strong> Upload your '
                "<code>generation_progress.json</code> to restore every setting and the chapter selection. "
                "Then upload the book file and the narrator voice again — files are never restored from a JSON.</div>"
            )
            C["progress_upload"] = gr.File(
                label="Upload existing generation_progress.json",
                file_types=[".json"], file_count="single",
            )
            C["restore_status"] = gr.Markdown("")

        with gr.Tabs():

            # ═══════════════════════════════════════════════════════════════ #
            # TAB 1 — BOOK                                                    #
            # ═══════════════════════════════════════════════════════════════ #
            with gr.Tab("📚 Book"):
                gr.Markdown("### Step 1 — Upload your book")
                with gr.Row():
                    C["book_file"] = gr.File(label="Book file", file_types=list(_BOOK_FILE_TYPES))
                    with gr.Column():
                        C["scan_status"] = gr.Markdown("*Upload a file to begin.*")
                        C["language"] = gr.Dropdown(
                            label="Book language",
                            choices=languages0 or list(_FALLBACK_LANGUAGES),
                            value=_pick_language(info0, base.language),
                            allow_custom_value=not languages0,
                            interactive=True,
                            info="The languages offered are those of the engine chosen in Voice Studio.",
                        )
                        C["book_title"] = gr.Textbox(label="Book title", interactive=True)
                        C["author"] = gr.Textbox(label="Author", interactive=True)
                        C["existing_progress"] = gr.Markdown("")

                with gr.Group(visible=False) as chapter_panel:
                    gr.Markdown("### Step 2 — Select chapters to convert")
                    with gr.Row():
                        select_all_btn = gr.Button("Select all chapters", size="sm", variant="secondary")
                        deselect_all_btn = gr.Button("Deselect all", size="sm", variant="secondary")
                    C["selected_chapters"] = gr.CheckboxGroup(label="Chapters", choices=[], value=[], interactive=True)
                C["chapter_panel"] = chapter_panel
                C["book_note"] = gr.Markdown("")

                with gr.Group(visible=False) as page_panel:
                    C["page_ranges"] = gr.Textbox(
                        label='Page ranges (optional, e.g. "1-50, 51-120")',
                        placeholder="Leave empty to use the chapters ticked above",
                        info="When filled in, each range becomes one chapter and the chapter list is ignored.",
                        lines=1,
                    )
                    C["total_pages"] = gr.Markdown("")
                C["page_panel"] = page_panel

                gr.Markdown("### Step 3 — Output settings")
                with gr.Row():
                    C["output_format"] = gr.Dropdown(
                        label="Output format", choices=list(_OUTPUT_FORMATS), value=base.output_format,
                    )
                    C["lufs"] = _slider("lufs", "Loudness target (LUFS)", base.lufs)
                with gr.Row():
                    C["cover_image"] = gr.Image(label="Cover image (optional)", type="filepath", height=160)

            # ═══════════════════════════════════════════════════════════════ #
            # TAB 2 — VOICE PREPROCESSING                                     #
            # ═══════════════════════════════════════════════════════════════ #
            with gr.Tab("🎧 Voice Preprocessing"):
                gr.Markdown(
                    "### Clean a voice recording before cloning\n"
                    "The defaults repair what is safe to repair and leave the performance alone."
                )
                with gr.Row():
                    voice_raw_upload = gr.Audio(label="WAV / FLAC / OGG / MP3", type="filepath", sources=["upload"])
                    voice_processed_player = gr.Audio(label="Processed preview", type="numpy", interactive=False)

                with gr.Accordion("🔇 Noise reduction", open=True):
                    with gr.Row():
                        pp_noise_reduce = gr.Checkbox(label="Enable", value=pp.noise_reduce)
                        pp_noise_strength = gr.Slider(
                            label="Strength", minimum=0.0, maximum=1.0, step=0.05, value=pp.noise_reduce_strength,
                        )
                with gr.Accordion("🔈 Noise gate", open=False):
                    with gr.Row():
                        pp_gate = gr.Checkbox(label="Enable", value=pp.noise_gate)
                        pp_gate_db = gr.Slider(
                            label="Threshold (dB below clip peak)", minimum=-60, maximum=-20, step=1,
                            value=pp.noise_gate_threshold_db,
                        )
                        pp_gate_range = gr.Slider(
                            label="Max attenuation (dB)", minimum=3, maximum=24, step=1, value=pp.noise_gate_range_db,
                        )
                with gr.Accordion("📡 High-pass filter", open=False):
                    with gr.Row():
                        pp_hp = gr.Checkbox(label="Enable", value=pp.highpass_filter)
                        pp_hp_hz = gr.Slider(
                            label="Cutoff (Hz)", minimum=40, maximum=400, step=10, value=pp.highpass_cutoff_hz,
                        )
                with gr.Accordion("✂️ Silence", open=False):
                    with gr.Row():
                        pp_trim = gr.Checkbox(label="Trim leading/trailing silence", value=pp.trim_silence)
                        pp_shorten = gr.Checkbox(
                            label="Shorten long pauses (edits inside the clip)", value=pp.silence_removal,
                        )
                    with gr.Row():
                        pp_min_segment = gr.Slider(
                            label="Ignore clicks shorter than (ms)", minimum=20, maximum=500, step=10,
                            value=pp.min_segment_ms,
                        )
                        pp_max_silence = gr.Slider(
                            label="Shorten pauses to (ms)", minimum=80, maximum=2000, step=10,
                            value=pp.max_silence_kept_ms,
                        )
                with gr.Accordion("🔊 Level", open=False):
                    pp_loudness = gr.Slider(
                        label="Loudness target (LUFS)", minimum=-30, maximum=-14, step=0.5,
                        value=pp.loudness_target_lufs,
                    )
                with gr.Accordion("🔁 Resample", open=False):
                    with gr.Row():
                        pp_resample = gr.Checkbox(label="Enable (downsamples only)", value=pp.resample)
                        pp_target_sr = gr.Dropdown(
                            label="Target sample rate (downsamples only)",
                            choices=list(_PREPROCESS_SAMPLE_RATES),
                            value=_one_of(pp.target_sample_rate, _PREPROCESS_SAMPLE_RATES, 24000),
                        )
                with gr.Accordion("🎯 Best window", open=False):
                    with gr.Row():
                        pp_best = gr.Checkbox(
                            label="Cut a long recording down to its best part", value=pp.select_best_window,
                        )
                        pp_best_seconds = gr.Slider(
                            label="Window length (s)", minimum=5, maximum=30, step=1, value=pp.best_window_seconds,
                        )

                with gr.Row():
                    preprocess_btn = gr.Button("▶ Preview Processed Audio", variant="primary")
                    save_voice_btn = gr.Button("💾 Use as narrator voice", variant="secondary")
                preprocess_status = gr.Markdown("")

            # ═══════════════════════════════════════════════════════════════ #
            # TAB 3 — VOICE STUDIO                                            #
            # ═══════════════════════════════════════════════════════════════ #
            with gr.Tab("🎙️ Voice Studio"):
                gr.Markdown("### 1 · Engine")
                with gr.Row():
                    with gr.Column(scale=1):
                        C["tts_provider_name"] = gr.Dropdown(
                            label="TTS engine", choices=_provider_choices(), value=provider0, interactive=True,
                        )
                        C["tts_model_name"] = gr.Dropdown(
                            label="Model", choices=models0, value=model0 if model0 in models0 else None,
                            visible=bool(models0), interactive=True,
                        )
                    with gr.Column(scale=2):
                        C["provider_info"] = gr.Markdown(_provider_markdown(provider0))

                gr.Markdown("### 2 · Narrator voice")
                C["tts_timbre"] = gr.Dropdown(
                    label="Preset speaker", choices=voices0, value=_pick_speaker(info0, None),
                    visible=voice0.show_speaker, interactive=True,
                )
                C["tts_instruct"] = gr.Textbox(
                    label="Voice / style instruction", lines=2, visible=voice0.show_instruct,
                    info=_instruct_hint(info0),
                )
                with gr.Group(visible=voice0.show_clip) as clip_group:
                    with gr.Row():
                        C["voice_file"] = gr.Audio(label="Reference clip", type="filepath", sources=["upload"])
                        voice_status_md = gr.Markdown(analyze_voice_clip(None))
                    C["voice_transcript"] = gr.Textbox(
                        label="Reference transcript", lines=3, visible=voice0.show_transcript,
                        placeholder="The exact words spoken in the reference clip",
                        info=_transcript_hint(info0),
                    )
                    transcribe_btn = None
                    if can_transcribe:
                        with gr.Row():
                            transcribe_btn = gr.Button("📝 Auto-transcribe", size="sm", variant="secondary", scale=0)
                            transcribe_status = gr.Markdown(
                                "<small>Runs Whisper on the clip (the model is downloaded on first use).</small>"
                            )
                C["clip_group"] = clip_group

                with gr.Accordion("💾 Voice preset", open=False, visible=voice0.show_preset) as preset_group:
                    gr.Markdown(
                        "A preset stores the prepared narrator voice. With one loaded, no reference clip is needed."
                    )
                    with gr.Row():
                        C["voice_preset"] = gr.File(
                            label="Voice preset file", file_types=[".pt", ".npy"], file_count="single",
                        )
                        with gr.Column():
                            save_preset_btn = gr.Button("Save voice as preset", variant="secondary")
                            preset_download = gr.File(label="Saved preset", interactive=False, visible=False)
                    preset_status = gr.Markdown("")
                C["preset_group"] = preset_group

                with gr.Group(visible=voice0.show_design) as design_group:
                    gr.Markdown(
                        "**Voice design** — hear the voice the engine creates from the instruction. "
                        "It is designed once and then used for the whole book."
                    )
                    with gr.Row():
                        design_btn = gr.Button("🎨 Design voice sample", variant="secondary")
                        reroll_btn = gr.Button("🎲 Re-roll", variant="secondary")
                        save_designed_btn = gr.Button("💾 Save designed voice as preset", variant="secondary")
                    design_audio = gr.Audio(label="Designed voice", type="filepath", interactive=False)
                    design_status = gr.Markdown("")
                    designed_preset_file = gr.File(label="Saved preset", interactive=False, visible=False)
                C["design_group"] = design_group

                gr.Markdown("### 3 · Delivery")
                with gr.Row():
                    C["speed"] = _slider("speed", "Speaking speed", base.speed)
                    C["pause"] = _slider("pause", "Sentence pause (s)", base.pause)
                    C["para_pause"] = _slider("para_pause", "Paragraph pause (s)", base.para_pause)
                C["speed_note"] = gr.Markdown(_speed_note(info0))
                with gr.Row():
                    C["temperature"] = _slider("temperature", "Temperature", sampling0["temperature"])
                    C["top_p"] = _slider("top_p", "Top-p", sampling0["top_p"])
                    C["top_k"] = _slider("top_k", "Top-k", sampling0["top_k"])
                    C["repetition_penalty"] = _slider(
                        "repetition_penalty", "Repetition penalty", sampling0["repetition_penalty"],
                    )
                    C["seed"] = gr.Number(
                        label="Seed (-1 = random)", value=base.seed, precision=0,
                        visible=bool(getattr(info0, "supports_seed", True)),
                    )
                gr.Markdown("<small>Choosing an engine sets these four to that engine's recommended values.</small>")

                with gr.Accordion("🧩 Engine options", open=False):
                    @gr.render(
                        inputs=[C["tts_provider_name"], C["tts_options"]],
                        triggers=[C["tts_provider_name"].change, demo.load, C["options_epoch"].change],
                    )
                    def render_engine_options(provider: Any, options_state: Any) -> None:
                        """One control per ``ProviderOption`` of the selected engine."""
                        key = _canonical_provider(provider)
                        info = _provider_info(key)
                        options = list(getattr(info, "options", ()) or ())
                        if not options:
                            gr.Markdown("*This engine has no extra options.*")
                            return
                        current = _provider_options_for(options_state, key)
                        factories = {
                            "checkbox": gr.Checkbox, "dropdown": gr.Dropdown, "slider": gr.Slider,
                            "number": gr.Number, "textbox": gr.Textbox, "file": gr.File,
                        }
                        for option in options:
                            kind, kwargs = _option_spec(option)
                            if option.key in current and kind != "file":
                                value = _coerce_option(option, current[option.key])
                                if value is not None:
                                    kwargs["value"] = value
                            if kind == "file":
                                kwargs.update(file_count="single", type="filepath")
                                if option.help:
                                    gr.Markdown(f"<small>{html.escape(option.help)}</small>")
                            component = factories[kind](key=f"opt-{key}-{option.key}", interactive=True, **kwargs)
                            component.change(
                                functools.partial(set_provider_option, provider=key, key=option.key),
                                inputs=[component, C["tts_options"]],
                                outputs=[C["tts_options"]],
                                show_progress="hidden",
                            )

                gr.Markdown("### 4 · Voice test")
                test_text = gr.Textbox(label="Test text", value=_DEFAULT_TEST_TEXT, lines=3)
                with gr.Row():
                    book_paragraph_btn = gr.Button("📖 Test with a paragraph from the book", variant="secondary")
                    test_btn = gr.Button("▶ Test Voice", variant="primary")
                test_audio = gr.Audio(label="Preview", type="numpy", interactive=False)
                test_status = gr.Markdown("")

            # ═══════════════════════════════════════════════════════════════ #
            # TAB 4 — ADVANCED                                                #
            # ═══════════════════════════════════════════════════════════════ #
            with gr.Tab("⚙️ Advanced"):
                gr.Markdown("### Narration")
                with gr.Row():
                    C["pack_sentences"] = gr.Checkbox(
                        label="Read a paragraph's sentences together", value=base.pack_sentences,
                        info="More natural flow. Off: one sentence per call.",
                    )
                    C["normalize_speech_text"] = gr.Checkbox(
                        label="Convert numbers, currency, dates and abbreviations to spoken form",
                        value=base.normalize_speech_text,
                    )
                    C["max_len"] = _slider("max_len", "Max chunk length (chars)", base.max_len)

                gr.Markdown("### Chunk verification")
                with gr.Row():
                    C["verify_chunks"] = gr.Dropdown(
                        label="Check each chunk", choices=list(_VERIFY_CHOICES), value=base.verify_chunks,
                        info="A chunk that fails is synthesized again.",
                    )
                    C["verify_max_retries"] = _slider("verify_max_retries", "Retries per chunk", base.verify_max_retries)
                with gr.Row():
                    C["verify_asr_model"] = gr.Dropdown(
                        label="Whisper model (transcript check)",
                        choices=list(dict.fromkeys((base.verify_asr_model, *_ASR_MODELS))),
                        value=base.verify_asr_model, allow_custom_value=True,
                    )
                    C["verify_max_wer"] = _slider(
                        "verify_max_wer", "Allowed share of wrong words (transcript check)", base.verify_max_wer,
                    )

                gr.Markdown("### Performance")
                with gr.Row():
                    C["batch_size"] = _slider("batch_size", "Batch size (0 = automatic)", base.batch_size)
                    C["gpu_count"] = _slider("gpu_count", "GPUs to use (0 = all)", base.gpu_count)
                    C["vram_headroom_gb"] = _slider("vram_headroom_gb", "VRAM headroom (GB)", base.vram_headroom_gb)
                    C["max_chapter_retries"] = _slider(
                        "max_chapter_retries", "Retries per failed chapter", base.max_chapter_retries,
                    )
                with gr.Row():
                    C["parallel_mode"] = gr.Dropdown(
                        label="Parallelism", choices=list(_PARALLEL_MODES), value=base.parallel_mode,
                        info="chunks: every GPU shares one chapter (default). chapters: one chapter per GPU. "
                             "auto: short chapters one per GPU, long ones shared (new, for books with short chapters).",
                    )
                    C["quantization"] = gr.Radio(
                        label="Model quantization", choices=list(_QUANTIZATIONS), value=base.quantization,
                        info="int8 roughly halves VRAM use (disables torch.compile).",
                    )
                    C["torch_compile"] = gr.Checkbox(label="torch.compile the model", value=base.torch_compile)

                gr.Markdown("### Audio encoding")
                with gr.Row():
                    C["sample_rate"] = gr.Dropdown(
                        label="Sample rate (Hz)", choices=list(_SAMPLE_RATES),
                        value=_one_of(base.sample_rate, _SAMPLE_RATES, 24000),
                    )
                    C["bitrate_kbps"] = gr.Dropdown(
                        label="Bitrate (kbps, MP3 / M4B)", choices=list(_BITRATES),
                        value=_one_of(base.bitrate_kbps, _BITRATES, 64),
                    )
                    C["channels"] = gr.Radio(
                        label="Channels", choices=list(_CHANNELS), value=_one_of(base.channels, _CHANNELS, 1),
                        info="1 = mono (recommended), 2 = stereo.",
                    )
                    C["true_peak"] = _slider("true_peak", "True peak (dBTP)", base.true_peak)

                gr.Markdown("### Book processing")
                with gr.Row():
                    C["epub_ocr"] = gr.Checkbox(label="OCR text inside images / scanned pages (EasyOCR)", value=False)
                    C["force_reprocess"] = gr.Checkbox(
                        label="Force re-process (ignore saved progress)", value=base.force_reprocess,
                    )
                    C["export_text"] = gr.Checkbox(label="Export chapter text as .txt", value=base.export_text)

            # ═══════════════════════════════════════════════════════════════ #
            # TAB 5 — GENERATE                                                #
            # ═══════════════════════════════════════════════════════════════ #
            with gr.Tab("🚀 Generate"):
                gr.Markdown("### Generate Audiobook")
                with gr.Row():
                    preview_btn = gr.Button("🔍 Preview Chapters", variant="secondary", scale=1)
                    generate_btn = gr.Button("🎧 Generate Audiobook", variant="primary", scale=3)
                    export_cfg_btn = gr.Button("📋 Export Config JSON", variant="secondary", scale=1)
                    cancel_btn = gr.Button("⛔ Cancel", variant="stop", scale=1)

                with gr.Row():
                    C["single_file_mode"] = gr.Checkbox(
                        label="📦 Combine into a single file", value=base.single_file_mode,
                        info="M4B and MP3 get chapter markers.",
                    )
                    C["export_lrc"] = gr.Checkbox(label="📜 Timed LRC text", value=base.export_lrc)
                    C["export_srt"] = gr.Checkbox(label="🎬 SRT subtitles", value=base.export_srt)
                    C["export_vtt"] = gr.Checkbox(label="WebVTT subtitles", value=base.export_vtt)
                with gr.Row():
                    C["regen_missing"] = gr.Checkbox(
                        label="🔄 Re-generate completed chapters whose audio file is missing",
                        value=base.regen_missing,
                    )
                    C["resume_incomplete_chunks"] = gr.Checkbox(
                        label="⏩ Resume interrupted chapters from the last finished chunk",
                        value=base.resume_incomplete_chunks,
                    )

                with gr.Accordion("🗣️ Pronunciation fixes", open=False):
                    gr.Markdown(
                        "Each fix replaces text before it is spoken. Patterns are regular expressions; "
                        "plain words work too. The table and the file are merged (the table wins)."
                    )
                    C["pronunciation_table"] = gr.Dataframe(
                        headers=["Find (regex allowed)", "Say instead"],
                        datatype=["str", "str"], type="array", interactive=True,
                        value=_pronunciation_rows(None), label="Fixes",
                    )
                    C["pronunciation_file"] = gr.File(
                        label="Fix file (.txt, one `find == say` per line, # for comments)",
                        file_types=[".txt"], file_count="single",
                    )
                    with gr.Row():
                        audition_text = gr.Textbox(
                            label="Audition", placeholder="A sentence containing the word you fixed", scale=3,
                        )
                        audition_btn = gr.Button("▶ Audition", variant="secondary", scale=0)
                    audition_audio = gr.Audio(label="Audition", type="numpy", interactive=False)
                    audition_status = gr.Markdown("")

                preview_table = gr.Dataframe(
                    headers=["#", "Chapter", "Chars", "Words", "Sentences"],
                    datatype=["number", "str", "number", "number", "number"],
                    label="Chapter preview", visible=False, interactive=False, wrap=True,
                )

                C["run_status"] = gr.Markdown("")
                C["progress"] = gr.HTML(_progress_html(0.0))
                C["log"] = gr.Textbox(
                    label=f"Generation log (last {_LOG_TAIL_LINES} lines)", lines=18, max_lines=18,
                    interactive=False, autoscroll=True,
                )
                C["log_file"] = gr.File(label="Full log", interactive=False, visible=False)

                gr.Markdown("#### Result")
                C["result_md"] = gr.Markdown("")
                C["result_table"] = gr.Dataframe(
                    headers=_RESULT_HEADERS,
                    datatype=["number", "str", "str", "str", "number", "str"],
                    label="Chapters", visible=False, interactive=False, wrap=True,
                )
                with gr.Row():
                    C["player_dd"] = gr.Dropdown(
                        label="Listen to a finished chapter", choices=[], value=None, visible=False, interactive=True,
                    )
                    player_audio = gr.Audio(label="Player", type="filepath", interactive=False)
                with gr.Row():
                    C["redo_select"] = gr.Dropdown(
                        label="Redo completed chapters", choices=[], value=[], multiselect=True, interactive=True,
                        info="Only the chapters picked here are generated again.", scale=3,
                    )
                    redo_btn = gr.Button("🔁 Redo selected chapters", variant="secondary", scale=1)

                with gr.Accordion("📋 Export Config JSON", open=False) as export_cfg_accordion:
                    gr.HTML(
                        '<div class="warn-box">ℹ️ <strong>Export Config JSON</strong> saves every setting and the '
                        "chapter text into one file for the CLI (<code>python cli.py config.json</code>). "
                        "Progress already made on the book is kept.</div>"
                    )
                    export_cfg_status = gr.Markdown("")
                    export_config_file = gr.File(
                        label="Download generation_progress.json", interactive=False, visible=False,
                    )

                gr.Markdown("#### Download outputs")
                with gr.Column(visible=False) as download_col:
                    C["download_files"] = gr.File(label="Output files", file_count="multiple", interactive=False)
                    zip_btn = gr.Button("⬇ Download All (ZIP)", variant="secondary")
                    zip_file = gr.File(label="ZIP", interactive=False, visible=False)
                C["download_col"] = download_col

                with gr.Accordion("ℹ️ Which format should I choose?", open=False):
                    gr.Markdown(_FORMAT_GUIDE)

            # ═══════════════════════════════════════════════════════════════ #
            # TAB 6 — AUDIOBOOKSHELF                                          #
            # ═══════════════════════════════════════════════════════════════ #
            with gr.Tab("📜 Audiobookshelf"):
                gr.Markdown(_AUDIOBOOKSHELF_GUIDE)

        # ══════════════════════════════════════════════════════════════════
        # EVENT WIRING
        # ══════════════════════════════════════════════════════════════════
        missing = [key for key in _UI_KEYS if key not in C]
        if missing:
            raise RuntimeError(f"UI settings without a component: {missing}")

        def out(slots: tuple[str, ...]) -> list[Any]:
            return [C[name] for name in slots]

        cfg_inputs = [C[key] for key in _UI_KEYS]
        book_inputs = [C["book_file"], C["scan_state"], C["page_ranges"], C["epub_ocr"]]
        title_inputs = [C["book_title"], client_token]
        gen_inputs = [
            *book_inputs, C["progress_upload"], C["chapters_cache"], C["all_choices"], client_token, *cfg_inputs,
        ]
        gen_outputs = out(_GEN_SLOTS)

        # ── Page load: session token, then re-attach to a run still going ─────
        demo.load(
            on_page_load, inputs=[client_token], outputs=[client_token], api_name="page_load",
        ).then(
            on_reattach, inputs=[client_token], outputs=[*gen_outputs, C["book_title"], active_banner],
            api_name="reattach", show_progress="hidden", concurrency_limit=None,
        )

        # ── Book ──────────────────────────────────────────────────────────────
        C["book_file"].upload(
            on_book_upload, inputs=[C["book_file"], C["json_selected"]], outputs=out(_BOOK_SLOTS),
            api_name="scan_book",
        ).then(
            on_title_commit, inputs=title_inputs, outputs=out(_TITLE_SLOTS),
        )
        select_all_btn.click(select_default_chapters, inputs=[C["all_choices"]], outputs=[C["selected_chapters"]])
        deselect_all_btn.click(deselect_all_chapters, outputs=[C["selected_chapters"]])
        # Not on every keystroke: each call parses the whole progress file.
        C["book_title"].blur(on_title_commit, inputs=title_inputs, outputs=out(_TITLE_SLOTS), api_name="title_commit")
        C["book_title"].submit(on_title_commit, inputs=title_inputs, outputs=out(_TITLE_SLOTS))

        C["progress_upload"].upload(
            on_progress_upload_handler, inputs=[C["progress_upload"]], outputs=out(_RESTORE_SLOTS),
            api_name="restore_progress",
        ).then(
            on_title_commit, inputs=title_inputs, outputs=out(_TITLE_SLOTS),
        )

        # ── Voice Preprocessing ───────────────────────────────────────────────
        preprocess_btn.click(
            run_preprocess,
            inputs=[
                voice_raw_upload,
                pp_noise_reduce, pp_noise_strength,
                pp_gate, pp_gate_db, pp_gate_range,
                pp_hp, pp_hp_hz,
                pp_trim, pp_shorten, pp_min_segment, pp_max_silence,
                pp_loudness,
                pp_resample, pp_target_sr,
                pp_best, pp_best_seconds,
            ],
            outputs=[voice_processed_player, preprocess_status, preproc_state],
            api_name="preprocess",
        )
        save_voice_btn.click(
            save_processed_voice, inputs=[preproc_state, voice_raw_upload],
            outputs=[preprocess_status, C["voice_file"]],
        )

        # ── Voice Studio ──────────────────────────────────────────────────────
        # .input, not .change: restoring a config sets the engine too, and must
        # not replace the restored sampling values with the recommended ones.
        C["tts_provider_name"].input(
            on_provider_change, inputs=[C["tts_provider_name"], C["language"]], outputs=out(_PROVIDER_SLOTS),
            api_name="provider_change",
        )
        C["tts_model_name"].input(
            on_model_change, inputs=[C["tts_provider_name"], C["tts_model_name"]], outputs=out(_MODEL_SLOTS),
            api_name="model_change",
        )
        C["voice_file"].change(analyze_voice_clip, inputs=[C["voice_file"]], outputs=[voice_status_md])
        if transcribe_btn is not None:
            transcribe_btn.click(
                transcribe_reference,
                inputs=[C["voice_file"], C["language"], C["verify_asr_model"]],
                outputs=[C["voice_transcript"], transcribe_status],
            )
        C["voice_preset"].change(
            on_preset_upload, inputs=[C["voice_preset"], C["tts_provider_name"]], outputs=[preset_status],
        )
        save_preset_btn.click(on_save_preset, inputs=cfg_inputs, outputs=[preset_download, preset_status])
        design_btn.click(on_design_voice, inputs=cfg_inputs, outputs=[design_audio, design_status, design_state])
        reroll_btn.click(on_reroll_voice, inputs=cfg_inputs, outputs=[design_audio, design_status, design_state])
        save_designed_btn.click(
            on_save_designed_preset, inputs=[design_state, *cfg_inputs],
            outputs=[designed_preset_file, design_status],
        )
        book_paragraph_btn.click(
            on_use_book_paragraph,
            inputs=[*book_inputs, C["chapters_cache"], C["all_choices"], C["selected_chapters"], C["language"]],
            outputs=[test_text, test_status],
        )
        test_btn.click(
            on_test_voice, inputs=[test_text, *cfg_inputs], outputs=[test_audio, test_status], api_name="test_voice",
        )
        audition_btn.click(
            on_audition, inputs=[audition_text, *cfg_inputs], outputs=[audition_audio, audition_status],
            api_name="audition",
        )

        # ── Generate ──────────────────────────────────────────────────────────
        preview_btn.click(
            on_preview,
            inputs=[*book_inputs, C["chapters_cache"], C["all_choices"], C["selected_chapters"], C["language"]],
            outputs=[C["run_status"], preview_table, C["chapters_cache"]],
            api_name="preview",
        )
        # The handlers only stream a run that lives in its own thread, so any
        # number may run at once; duplicates are refused inside _generate.
        generate_btn.click(
            on_generate, inputs=gen_inputs, outputs=gen_outputs,
            api_name="generate", show_progress="hidden", concurrency_limit=None,
        )
        redo_btn.click(
            on_redo, inputs=[C["redo_select"], *gen_inputs], outputs=gen_outputs,
            api_name="redo", show_progress="hidden", concurrency_limit=None,
        )
        cancel_btn.click(
            on_cancel, inputs=[C["run_key"], C["book_title"], client_token], outputs=[C["run_status"]],
            api_name="cancel", concurrency_limit=None,
        )
        export_cfg_btn.click(
            on_export_config,
            inputs=[*book_inputs, C["chapters_cache"], C["all_choices"], client_token, *cfg_inputs],
            outputs=[export_cfg_status, export_config_file, export_cfg_accordion, C["chapters_cache"]],
            api_name="export_config",
        )
        C["player_dd"].change(on_player_select, inputs=[C["player_dd"]], outputs=[player_audio], api_name="play_chapter")
        zip_btn.click(on_zip, inputs=[C["download_files"]], outputs=[zip_file], api_name="zip")

    return demo


def _launch_kwargs() -> dict[str, Any]:
    """Arguments for ``demo.launch()``."""
    allowed = [_ROOT]
    output_base = os.environ.get("ABM_OUTPUT_BASE", "").strip()
    if output_base:
        allowed.append(os.path.realpath(output_base))
    kwargs: dict[str, Any] = {
        "server_name": "localhost",
        "server_port": 7860,
        "share": False,
        "show_error": True,
        "allowed_paths": allowed,
    }
    kwargs.update(_style_kwargs(gr.Blocks.launch))
    return kwargs


def main() -> None:
    """Builds the UI and serves it."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    demo = build_app()
    demo.launch(**_launch_kwargs())


if __name__ == "__main__":
    main()
