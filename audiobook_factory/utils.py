import os
import json
import re
import subprocess
import sys
import shutil
import tempfile
import logging

logger = logging.getLogger(__name__)


def seconds_to_srt_time(seconds):
    """Converts seconds to SRT timestamp format (HH:MM:SS,ms)."""
    ms = int(seconds * 1000)
    minutes, seconds = divmod(ms // 1000, 60)
    hours, minutes = divmod(minutes, 60)
    ms = ms % 1000
    return f"{hours:02d}:{minutes:02d}:{seconds % 60:02d},{ms:03d}"


def seconds_to_vtt_time(seconds):
    """Converts seconds to WebVTT timestamp format (HH:MM:SS.mmm)."""
    ms = int(seconds * 1000)
    minutes, seconds = divmod(ms // 1000, 60)
    hours, minutes = divmod(minutes, 60)
    ms = ms % 1000
    return f"{hours:02d}:{minutes:02d}:{seconds % 60:02d}.{ms:03d}"


def normalize_chapter_title_for_matching(title: str):
    if not title:
        return None, ""
    cleaned = re.sub(r'\(~[\d,]+\s*words\)', '', str(title)).strip().lower()
    num_match = re.search(r'^(?:chapter|chap|ch\.?)?\s*(\d+)\b', cleaned, re.IGNORECASE)
    ch_num = int(num_match.group(1)) if num_match else None
    core_text = re.sub(r'^(?:chapter|chap|ch\.?)?\s*\d+[:.\-\s]*', '', cleaned).strip()
    return ch_num, core_text or cleaned


def format_lrc_timestamp(seconds):
    """
    Converts a time in seconds to LRC timestamp format [mm:ss.xx].
    Used for synchronizing lyrics with audio.
    """
    minutes = int(seconds // 60)
    sec = int(seconds % 60)
    hundredths = int((seconds - (minutes * 60) - sec) * 100)
    return f"[{minutes:02d}:{sec:02d}.{hundredths:02d}]"


_DONE_PREFIX = "__DONE__::"


def encode_done_message(files=None) -> str:
    """Builds the log-queue sentinel that carries a run's output file list.

    The list is JSON-encoded: joining paths with a comma broke on any book or
    chapter title containing one ("The Lion, the Witch and the Wardrobe").
    """
    return _DONE_PREFIX + json.dumps(list(files or []))


def decode_done_message(msg: str):
    """Returns the file list from a done sentinel, or None if `msg` is not one."""
    if not isinstance(msg, str) or not msg.startswith(_DONE_PREFIX):
        return None
    payload = msg[len(_DONE_PREFIX):]
    if not payload:
        return []
    try:
        files = json.loads(payload)
    except ValueError:
        return [payload]
    return [f for f in files if isinstance(f, str)] if isinstance(files, list) else []
