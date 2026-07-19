import os
import json
import re
import ebooklib
import subprocess
import sys
import shutil
import tempfile
from ebooklib import epub
from bs4 import BeautifulSoup



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

import threading as _threading
_progress_lock = _threading.Lock()

def update_progress_file(progress_path, chapter_num, status):
    """
    Updates the progress JSON file for a specific chapter.
    Thread-safe: uses a module-level lock to prevent race conditions
    when multiple chapter workers complete simultaneously.
    """
    with _progress_lock:
        with open(progress_path, 'r', encoding='utf-8') as f:
            progress_data = json.load(f)
        for chapter in progress_data["chapters"]:
            if chapter["num"] == chapter_num:
                chapter["status"] = status
                break
        with open(progress_path, 'w', encoding='utf-8') as f:
            json.dump(progress_data, f, indent=4)

def load_or_create_progress_file(progress_path, chapters_data, book_title, book_path="", voice_file="", settings=None):
    """Loads a progress file if it exists, otherwise creates a new one.

    chapters_data entries may optionally contain 'text' and 'sentences' keys.
    When creating a new file these are persisted so the CLI can bypass book
    re-parsing on resume.  When loading an existing file that lacks those keys
    (older format) we backfill from chapters_data if available.
    """
    if os.path.exists(progress_path):
        print("Found existing progress file. Loading state.")
        with open(progress_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Backfill text/sentences for chapters that are missing them (old format)
        existing_by_num = {c["num"]: c for c in data.get("chapters", [])}
        dirty = False
        for cd in chapters_data:
            ch = existing_by_num.get(cd["num"])
            if ch is not None:
                if "text" not in ch and cd.get("text"):
                    ch["text"] = cd["text"]
                    dirty = True
                if "sentences" not in ch and cd.get("sentences"):
                    ch["sentences"] = cd["sentences"]
                    dirty = True
        if dirty:
            try:
                with open(progress_path, 'w', encoding='utf-8') as f:
                    json.dump(data, f, indent=4)
            except Exception as e:
                print(f"Warning: could not backfill text cache into progress file: {e}")
        return data
    else:
        print("No progress file found. Creating a new one.")
        progress_data = {
            "book_title": book_title,
            "book_path": book_path,
            "voice_file": voice_file,
            "settings": settings or {},
            "chapters": [
                {
                    "num": c["num"],
                    "title": c["title"],
                    "status": "pending",
                    "text": c.get("text", ""),
                    "sentences": c.get("sentences", []),
                }
                for c in chapters_data
            ],
        }
        with open(progress_path, 'w', encoding='utf-8') as f:
            json.dump(progress_data, f, indent=4)
        return progress_data


def format_lrc_timestamp(seconds):
    """
    Converts a time in seconds to LRC timestamp format [mm:ss.xx].
    Used for synchronizing lyrics with audio.
    """
    minutes = int(seconds // 60)
    sec = int(seconds % 60)
    hundredths = int((seconds - (minutes * 60) - sec) * 100)
    return f"[{minutes:02d}:{sec:02d}.{hundredths:02d}]"

