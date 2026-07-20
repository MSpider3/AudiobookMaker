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

def normalize_chapter_title_for_matching(title: str):
    if not title:
        return None, ""
    cleaned = re.sub(r'\(~[\d,]+\s*words\)', '', str(title)).strip().lower()
    num_match = re.search(r'^(?:chapter|chap|ch\.?)?\s*(\d+)\b', cleaned, re.IGNORECASE)
    ch_num = int(num_match.group(1)) if num_match else None
    core_text = re.sub(r'^(?:chapter|chap|ch\.?)?\s*\d+[:.\-\s]*', '', cleaned).strip()
    return ch_num, core_text or cleaned


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
        existing_chapters = data.get("chapters", [])

        dirty = False
        for cd in chapters_data:
            cd_title_raw = cd.get("title", "").strip().lower()
            cd_clean = re.sub(r'\(~[\d,]+\s*words\)', '', cd.get("title", "")).strip().lower()
            cd_num_extracted, cd_core = normalize_chapter_title_for_matching(cd.get("title", ""))

            ch = None
            # Phase 1: High Priority Title Matching
            for c in existing_chapters:
                c_title_raw = c.get("title", "").strip().lower()
                c_clean = re.sub(r'\(~[\d,]+\s*words\)', '', c.get("title", "")).strip().lower()
                c_num_extracted, c_core = normalize_chapter_title_for_matching(c.get("title", ""))

                if (
                    (c_title_raw and c_title_raw == cd_title_raw)
                    or (c_clean and c_clean == cd_clean)
                    or (c_core and cd_core and c_core == cd_core)
                    or (c_num_extracted is not None and cd_num_extracted is not None and c_num_extracted == cd_num_extracted and c_core == cd_core)
                ):
                    ch = c
                    break

            # Phase 2: Fallback Index Matching if no title match was found
            if ch is None:
                for c in existing_chapters:
                    c_num_extracted, _ = normalize_chapter_title_for_matching(c.get("title", ""))
                    if (
                        (c.get("num") is not None and c.get("num") == cd.get("num"))
                        or (c.get("num") is not None and str(c.get("num")) == str(cd.get("num")))
                        or (c_num_extracted is not None and cd_num_extracted is not None and c_num_extracted == cd_num_extracted)
                    ):
                        ch = c
                        break

            if ch is not None:
                if ("text" not in ch or not ch["text"]) and cd.get("text"):
                    ch["text"] = cd["text"]
                    dirty = True
                if ("sentences" not in ch or not ch["sentences"]) and cd.get("sentences"):
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

