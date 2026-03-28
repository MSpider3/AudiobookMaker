"""
audiobook_factory/filename_sanitizer.py
========================================
Cross-platform, filesystem-safe filename generation for audiobook chapter files.

Ported and adapted from epub_to_audiobook (p0n1) — MIT licence.
Produces filenames that are compatible with Audiobookshelf's library scanner.

Output format:  {idx:04d}_{sanitized_title}.{ext}
Example:        0001_Chapter_I_The_Beginning.mp3
"""
from __future__ import annotations

import hashlib
import os
import unicodedata

# Characters forbidden on Windows (and some Linux FS).
_FORBIDDEN = frozenset('<>:"/\\|?*\n\r\t\x00')

# Windows reserved names (device names).
_WIN_RESERVED = frozenset({
    "CON", "PRN", "AUX", "NUL",
    "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9",
    "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9",
})


# ── Internal helpers ──────────────────────────────────────────────────────────

def _detect_name_max(path: str) -> int:
    """Return NAME_MAX for the filesystem containing *path*, or 255 as default."""
    directory = path if os.path.isdir(path) else os.path.dirname(path) or "."
    try:
        name_max = os.pathconf(directory, "PC_NAME_MAX")
        if isinstance(name_max, int) and name_max >= 64:
            return name_max
    except (OSError, AttributeError, ValueError):
        pass
    return 255


def _sanitize_base_name(name: str) -> str:
    """Normalize and sanitize text for use as a filename stem (no extension)."""
    if not name:
        return "chapter"

    # Unicode NFKC: normalise ligatures, width variants, etc.
    name = unicodedata.normalize("NFKC", name)

    # Replace forbidden / control characters with underscore.
    chars = []
    for ch in name:
        if ch in _FORBIDDEN or (ord(ch) < 32):
            chars.append("_")
        else:
            chars.append(ch)
    sanitized = "".join(chars)

    # Collapse whitespace and replace spaces with underscores.
    sanitized = "_".join(sanitized.split())

    # Strip leading/trailing dots and underscores (Windows quirk).
    sanitized = sanitized.strip("._")

    # Guard Windows reserved names.
    if sanitized.upper() in _WIN_RESERVED:
        sanitized = sanitized + "_file"

    return sanitized or "chapter"


# ── Public API ────────────────────────────────────────────────────────────────

def make_safe_filename(
    title: str,
    idx: int,
    output_dir: str,
    ext: str,
    reserve: int = 16,
) -> str:
    """
    Build a filesystem-safe, Audiobookshelf-compatible chapter filename.

    Parameters
    ----------
    title       : Chapter title (raw, may contain special characters).
    idx         : 1-based chapter index (zero-padded to 4 digits).
    output_dir  : Directory where the file will live (used to query NAME_MAX).
    ext         : File extension WITH leading dot (e.g. '.mp3').
    reserve     : Extra bytes to keep clear (for safety margin).

    Returns
    -------
    Filename string (no directory path).  Example: '0003_Chapter_III.mp3'
    """
    if not ext:
        raise ValueError("ext must be non-empty")
    if not ext.startswith("."):
        ext = "." + ext

    name_max = _detect_name_max(output_dir)
    effective_name_max = max(64, min(name_max, 255))

    prefix = f"{idx:04d}_"
    base   = _sanitize_base_name(title)

    prefix_bytes = prefix.encode("utf-8")
    ext_bytes    = ext.encode("utf-8")

    if len(prefix_bytes) + len(ext_bytes) + 8 >= effective_name_max:
        raise RuntimeError(
            "Cannot construct safe filename: prefix+extension exceeds filesystem limit."
        )

    max_base_bytes = effective_name_max - len(prefix_bytes) - len(ext_bytes) - reserve
    if max_base_bytes <= 0:
        raise RuntimeError("No room for base name under filesystem limits.")

    base_bytes = base.encode("utf-8")

    if len(base_bytes) <= max_base_bytes:
        return prefix + base + ext

    # Truncate safely at a UTF-8 character boundary.
    truncated = base_bytes[:max_base_bytes]
    while truncated and (truncated[-1] & 0b11000000) == 0b10000000:
        truncated = truncated[:-1]

    truncated_base = truncated.decode("utf-8", errors="ignore").rstrip("._- ")
    if not truncated_base:
        truncated_base = "chapter"

    # Append a short hash so truncated names stay unique.
    h = hashlib.sha1(base_bytes).hexdigest()[:8]
    candidate = f"{prefix}{truncated_base}_{h}{ext}"

    # Last-resort: just hash.
    if len(candidate.encode("utf-8")) > effective_name_max:
        candidate = f"{prefix}{h}{ext}"

    return candidate
