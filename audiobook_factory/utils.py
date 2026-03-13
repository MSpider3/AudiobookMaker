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



def ms_to_srt_time(ms):
    """Converts milliseconds to SRT timestamp format (HH:MM:SS,ms)."""
    seconds, milliseconds = divmod(ms, 1000)
    minutes, seconds = divmod(seconds, 60)
    hours, minutes = divmod(minutes, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

def update_progress_file(progress_path, chapter_num, status):
    """
    Updates the progress JSON file for a specific chapter.
    """
    with open(progress_path, 'r', encoding='utf-8') as f:
        progress_data = json.load(f)
    for chapter in progress_data["chapters"]:
        if chapter["num"] == chapter_num:
            chapter["status"] = status
            break
    with open(progress_path, 'w', encoding='utf-8') as f:
        json.dump(progress_data, f, indent=4)

def load_or_create_progress_file(progress_path, chapters_data, book_title):
    """Loads a progress file if it exists, otherwise creates a new one."""
    if os.path.exists(progress_path):
        print("Found existing progress file. Loading state.")
        with open(progress_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    else:
        print("No progress file found. Creating a new one.")
        progress_data = {
            "book_title": book_title,
            "chapters": [
                {"num": c["num"], "title": c["title"], "status": "pending"} for c in chapters_data
            ]
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

