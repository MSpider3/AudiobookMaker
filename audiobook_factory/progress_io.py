"""
audiobook_factory/progress_io.py

Single, robust I/O layer for generation_progress.json.
All reads and writes go through this module.
"""
from __future__ import annotations

__all__ = [
    "read_progress_file",
    "write_progress_file",
    "update_chapter_status",
    "update_chapter_chunk",
    "update_chapter_retry",
]

import json
import logging
import os
import re
import threading
from typing import Any

logger = logging.getLogger(__name__)

_WRITE_LOCK = threading.Lock()
# Module-level lock protecting all writes to progress JSON files.
# Prevents concurrent write corruption when chapters run in parallel.

_SUPPORTED_ENCODINGS: tuple[str, ...] = ("utf-8-sig", "utf-8", "latin-1")
# Tried in order on read. utf-8-sig strips BOM automatically.
# latin-1 is a last-resort fallback — every byte is valid latin-1.

_MIN_VALID_JSON_SIZE: int = 10
# Files smaller than this are treated as empty/corrupt.

_JSON_OBJECT_PATTERN = re.compile(r'\{', re.MULTILINE)
# Used to find the start of a JSON object when the file has leading garbage.


def _strip_leading_garbage(raw: str) -> str:
    """Removes any text before the first '{' character.

    Handles cases where files are accidentally prefixed with text
    (e.g. 'chp{...}' → '{...}').

    Args:
        raw: Raw file content string.

    Returns:
        String starting from the first '{', or original if no '{' found.
    """
    match = _JSON_OBJECT_PATTERN.search(raw)
    if match and match.start() > 0:
        logger.warning(
            "Progress file has %d leading garbage characters before '{'. "
            "Auto-stripping and continuing.",
            match.start(),
        )
        return raw[match.start():]
    return raw


def read_progress_file(path: str) -> dict[str, Any]:
    """Reads and parses a generation_progress.json file robustly.

    Handles: UTF-8 BOM, leading garbage text, multiple encodings,
    HTML content (raises with clear message), empty files (raises with
    clear message), and JSON syntax errors (raises with line/col info).

    Args:
        path: Absolute or relative path to the progress JSON file.

    Returns:
        Parsed dict. Never returns None.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is empty, HTML, or has unrecoverable
                    JSON syntax errors.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Progress file not found: {path}")

    file_size = os.path.getsize(path)
    if file_size < _MIN_VALID_JSON_SIZE:
        raise ValueError(
            f"Progress file is empty or too small ({file_size} bytes): {path}. "
            "Re-export config JSON from the Generate tab."
        )

    raw: str | None = None
    last_error: Exception | None = None

    for encoding in _SUPPORTED_ENCODINGS:
        try:
            with open(path, encoding=encoding) as fh:
                raw = fh.read()
            break
        except UnicodeDecodeError as exc:
            last_error = exc
            continue

    if raw is None:
        raise ValueError(
            f"Could not decode progress file with any supported encoding "
            f"({', '.join(_SUPPORTED_ENCODINGS)}): {path}. "
            f"Last error: {last_error}"
        )

    # Detect HTML (browser saved a webpage instead of the JSON file)
    stripped = raw.lstrip()
    if stripped.lower().startswith(("<!doctype", "<html")):
        raise ValueError(
            f"Progress file contains HTML, not JSON: {path}. "
            "You may have saved the web page instead of the raw JSON file. "
            "In Kaggle/Colab, right-click the file and choose "
            "'Download' or use the Files panel to download the raw file."
        )

    raw = _strip_leading_garbage(raw)

    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        snippet = raw[:120].replace("\n", " ")
        raise ValueError(
            f"Invalid JSON in progress file at line {exc.lineno}, "
            f"column {exc.colno}: {exc.msg}. "
            f"File preview: {snippet!r}"
        ) from exc


def write_progress_file(path: str, data: dict[str, Any]) -> None:
    """Writes data to a progress JSON file atomically.

    Uses temp file + os.replace() to guarantee the file is never
    partially written. Thread-safe via module-level _WRITE_LOCK.

    Args:
        path: Destination path.
        data: Dict to serialize as JSON.

    Raises:
        OSError: If the write fails.
    """
    with _WRITE_LOCK:
        _write_unlocked(path, data)


def update_chapter_status(
    path: str,
    chapter_num: int,
    status: str,
    reset_chunks: bool = False,
) -> None:
    """Updates a single chapter's status in the progress file.

    Thread-safe. Reads, modifies, and writes atomically.

    Args:
        path: Path to generation_progress.json.
        chapter_num: 1-based chapter number.
        status: New status string ("pending", "in_progress", "completed",
                "failed").
        reset_chunks: If True, resets completed_chunks to [] for this
                      chapter. Set True when marking "completed".
    """
    with _WRITE_LOCK:
        try:
            data = read_progress_file(path)
        except (FileNotFoundError, ValueError) as exc:
            logger.error(
                "Cannot update chapter status — progress file unreadable: %s",
                exc,
            )
            return

        for ch in data.get("chapters", []):
            if ch.get("num") == chapter_num or str(ch.get("num")) == str(chapter_num):
                ch["status"] = status
                if reset_chunks:
                    ch["completed_chunks"] = []
                break

        try:
            _write_unlocked(path, data)
        except OSError as exc:
            logger.error(
                "Failed to write chapter status update for chapter %d: %s",
                chapter_num,
                exc,
            )


def update_chapter_chunk(
    path: str,
    chapter_num: int,
    chunk_index: int,
) -> None:
    """Appends a completed chunk index to the chapter's completed_chunks list.

    Thread-safe. Uses _WRITE_LOCK to prevent concurrent write corruption.

    Args:
        path: Path to generation_progress.json.
        chapter_num: 1-based chapter number.
        chunk_index: Zero-based chunk index to mark as completed.
    """
    with _WRITE_LOCK:
        try:
            data = read_progress_file(path)
        except (FileNotFoundError, ValueError) as exc:
            logger.warning(
                "Cannot update chunk completion — progress file unreadable: %s",
                exc,
            )
            return

        for ch in data.get("chapters", []):
            if ch.get("num") == chapter_num or str(ch.get("num")) == str(chapter_num):
                completed = ch.setdefault("completed_chunks", [])
                if chunk_index not in completed:
                    completed.append(chunk_index)
                break

        try:
            _write_unlocked(path, data)
        except OSError as exc:
            logger.warning(
                "Failed to persist chunk completion for chapter %d chunk %d: %s",
                chapter_num,
                chunk_index,
                exc,
            )



def update_chapter_retry(
    path: str,
    chapter_num: int,
    attempt: int,
    error_message: str,
) -> None:
    """Updates a chapter's retry count, last error, and status in progress JSON.

    Thread-safe. Reads, modifies, and writes atomically under _WRITE_LOCK.

    Args:
        path: Path to generation_progress.json.
        chapter_num: 1-based chapter number.
        attempt: Retry attempt count.
        error_message: Error string (truncated to 500 characters).
    """
    with _WRITE_LOCK:
        try:
            data = read_progress_file(path)
        except (FileNotFoundError, ValueError) as exc:
            logger.warning(
                "Cannot update retry info — progress file unreadable: %s",
                exc,
            )
            return

        for ch in data.get("chapters", []):
            if ch.get("num") == chapter_num or str(ch.get("num")) == str(chapter_num):
                ch["retry_count"] = attempt
                ch["last_error"] = str(error_message)[:500]
                ch["status"] = "failed"
                break

        try:
            _write_unlocked(path, data)
        except OSError as exc:
            logger.warning(
                "Failed to write retry update for chapter %d: %s",
                chapter_num,
                exc,
            )


def _write_unlocked(path: str, data: dict[str, Any]) -> None:
    """Writes data without acquiring _WRITE_LOCK.

    INTERNAL USE ONLY. Caller must already hold _WRITE_LOCK.
    Used by update_chapter_status and update_chapter_chunk which read
    and write within a single lock acquisition.

    Args:
        path: Destination path.
        data: Dict to serialize.

    Raises:
        OSError: If write fails.
    """
    dir_name = os.path.dirname(os.path.abspath(path))
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    tmp_path = path + ".tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, path)
    except OSError:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        raise
