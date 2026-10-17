#!/usr/bin/env python3
"""
scripts/combine_book.py
=======================
Joins the chapter files of an audiobook into one file with chapter markers,
and its subtitle files into one subtitle file per type.

Made for books generated on Kaggle or Colab and downloaded in parts: point it
at the folders and ZIP files you downloaded and, ideally, at the book's
``generation_progress.json``. It needs only Python 3.9+ and FFmpeg - none of
the TTS dependencies - so it can be fetched on its own::

    curl -O https://raw.githubusercontent.com/MSpider3/AudiobookMaker/main/scripts/combine_book.py
    python combine_book.py part1.zip part2.zip --progress generation_progress.json

What it does:

* finds the chapter audio files (``Chapter 7 - Title.mp3``) and their
  ``.srt`` / ``.lrc`` / ``.vtt`` subtitles in every folder and ZIP given;
* orders them by the chapter list of the progress file (or by chapter number
  when there is no progress file) and stops if a chapter is missing;
* writes one audio file with a chapter marker per chapter, the book's title,
  author and cover - stream-copied when every part has the same format, so
  nothing is re-encoded;
* writes one subtitle file per type with every cue moved to its place in the
  whole book.
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile
from dataclasses import dataclass, field

logger = logging.getLogger("combine_book")

_AUDIO_EXTENSIONS: tuple[str, ...] = (".m4b", ".m4a", ".mp3", ".aac", ".ogg", ".opus", ".flac", ".wav", ".webm", ".mp4")
_SUBTITLE_EXTENSIONS: tuple[str, ...] = (".srt", ".lrc", ".vtt")
_COVER_NAMES: tuple[str, ...] = ("cover.jpg", "cover.jpeg", "cover.png", "cover.webp")
_PROGRESS_NAME: str = "generation_progress.json"
# Containers that carry chapter markers and an embedded cover.
_CHAPTER_FORMATS: tuple[str, ...] = ("m4b", "m4a", "mp4", "mp3", "ogg", "opus", "webm")
_COVER_FORMATS: tuple[str, ...] = ("m4b", "m4a", "mp4", "mp3", "flac")
# Encoder per output format when the parts have to be re-encoded.
_ENCODERS: dict[str, list[str]] = {
    "m4b": ["-c:a", "aac"], "m4a": ["-c:a", "aac"], "mp4": ["-c:a", "aac"], "aac": ["-c:a", "aac"],
    "mp3": ["-c:a", "libmp3lame"], "ogg": ["-c:a", "libopus"], "opus": ["-c:a", "libopus"],
    "webm": ["-c:a", "libopus"], "flac": ["-c:a", "flac"], "wav": ["-c:a", "pcm_s16le"],
}
_LOSSLESS_FORMATS: tuple[str, ...] = ("flac", "wav")
_DEFAULT_BITRATE_KBPS: int = 64
# "Chapter 12 - Title.mp3" (current names) and "0012_Title.mp3" (older names).
_CHAPTER_NAME = re.compile(r"^chapter\s+(\d+)\s*[-:.]?\s*(.*)$", re.IGNORECASE)
_NUMBERED_NAME = re.compile(r"^(\d{1,4})[_\s.-]+(.*)$")
_SRT_TIME = re.compile(r"(\d{1,2}):(\d{2}):(\d{2})[,.](\d{1,3})")
_VTT_TIME = re.compile(r"(?:(\d{1,2}):)?(\d{2}):(\d{2})\.(\d{3})")
_LRC_TAG = re.compile(r"\[(\d{1,3}):(\d{2})(?:[.:](\d{1,3}))?\]")
_LRC_META = re.compile(r"^\[[a-z]+:.*\]$", re.IGNORECASE)


class CombineError(Exception):
    """A problem with the input that stops the book from being combined."""


@dataclass
class Chapter:
    """One chapter of the book and the files found for it."""

    number: int
    title: str
    audio: str = ""
    subtitles: dict[str, str] = field(default_factory=dict)
    seconds: float = 0.0


@dataclass
class Book:
    """Book metadata from the progress file, the command line or defaults."""

    title: str = "Audiobook"
    author: str = ""
    output_format: str = ""
    bitrate_kbps: int = 0
    cover_b64: str = ""
    chapters: list[tuple[int, str, str]] = field(default_factory=list)   # (number, title, status)


# ── Finding the files ─────────────────────────────────────────────────────────

def _collect(paths: list[str], scratch: str) -> list[str]:
    """Every file inside the given folders and ZIP archives (ZIPs are unpacked into *scratch*)."""
    found: list[str] = []
    for index, path in enumerate(paths):
        if os.path.isdir(path):
            for folder, _dirs, files in os.walk(path):
                found.extend(os.path.join(folder, name) for name in files)
        elif zipfile.is_zipfile(path):
            target = os.path.join(scratch, f"part{index:03d}")
            with zipfile.ZipFile(path) as archive:
                for member in archive.infolist():
                    # Never write outside the target folder (absolute or "../" names).
                    destination = os.path.realpath(os.path.join(target, member.filename))
                    if not destination.startswith(os.path.realpath(target) + os.sep):
                        logger.warning("Skipping unsafe path in %s: %s", path, member.filename)
                        continue
                    archive.extract(member, target)
            for folder, _dirs, files in os.walk(target):
                found.extend(os.path.join(folder, name) for name in files)
        elif os.path.isfile(path):
            found.append(path)
        else:
            raise CombineError(f"Not found: {path}")
    return found


def _chapter_number(path: str) -> tuple[int | None, str]:
    """``(number, title)`` from a chapter file name, or ``(None, stem)``."""
    stem = os.path.splitext(os.path.basename(path))[0]
    for pattern in (_CHAPTER_NAME, _NUMBERED_NAME):
        match = pattern.match(stem)
        if match:
            return int(match.group(1)), match.group(2).replace("_", " ").strip() or stem
    return None, stem


def _title_key(title: str) -> str:
    """A chapter title reduced to its words, for matching a file name with the progress file.

    "Chapter 108: Deep Into The Night" and the file name
    "Chapter 108 - Deep Into The Night" (or "Chapter 109 - Chapter 108 Deep
    Into The Night") both become "deepintothenight".
    """
    core = title.strip()
    while True:
        match = _CHAPTER_NAME.match(core)
        if not match or not match.group(2):
            break
        core = match.group(2)
    return re.sub(r"[\W_]+", "", core.lower())


def _title_number(title: str) -> int | None:
    """The chapter number written in a title ("Chapter 108: ..." -> 108), or None."""
    match = _CHAPTER_NAME.match(title.strip())
    return int(match.group(1)) if match else None


def _natural_key(path: str) -> list:
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", os.path.basename(path))]


def load_progress(path: str) -> Book:
    """Book metadata and chapter list from a ``generation_progress.json``."""
    try:
        with open(path, encoding="utf-8-sig") as fh:
            data = json.load(fh)
    except (OSError, ValueError) as exc:
        raise CombineError(f"Cannot read the progress file {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise CombineError(f"{path} is not a generation_progress.json file.")
    settings = data.get("settings") if isinstance(data.get("settings"), dict) else {}
    chapters = []
    for entry in data.get("chapters") or []:
        if isinstance(entry, dict) and str(entry.get("num", "")).isdigit():
            chapters.append((int(entry["num"]), str(entry.get("title") or ""), str(entry.get("status") or "")))
    return Book(
        title=str(data.get("book_title") or settings.get("book_title") or "Audiobook"),
        author=str(settings.get("author") or ""),
        output_format=str(settings.get("output_format") or "").lower(),
        bitrate_kbps=int(settings.get("bitrate_kbps") or 0),
        cover_b64=str(data.get("cover_image_b64") or ""),
        chapters=chapters,
    )


def _assign_by_title(chapters: list[Chapter], audio: list[str]) -> None:
    """Gives each chapter of the progress file its audio file.

    File names do not reliably carry the progress file's chapter numbers:
    depending on the version that wrote them they hold that number or the one
    in the title ("Chapter 109 - Deduction" for entry 110, "Chapter 109:
    Deduction"). Titles are therefore matched first, and numbers only decide
    between chapters with the same title or stand in when the title differs.
    """
    files = []
    for path in audio:
        number, title = _chapter_number(path)
        files.append((path, number, _title_key(title)))
    # Largest first, so a chapter downloaded twice keeps its most complete copy.
    files.sort(key=lambda item: os.path.getsize(item[0]), reverse=True)
    taken: set[str] = set()

    def numbers_of(chapter: Chapter) -> list[int]:
        found = [_title_number(chapter.title), chapter.number]
        return [n for n in dict.fromkeys(found) if n is not None]

    def pick(chapter: Chapter, accept) -> None:
        for path, number, key in files:
            if path not in taken and accept(number, key):
                chapter.audio = path
                taken.add(path)
                return

    passes = (
        # 1. same title and a number that belongs to the chapter
        lambda c: (lambda number, key: key == _title_key(c.title) and number in numbers_of(c)),
        # 2. same title, when only one chapter has that title
        lambda c: (lambda number, key: key == _title_key(c.title)
                   and sum(_title_key(o.title) == key for o in chapters) == 1),
        # 3. the number written in the chapter's title, then its entry number
        lambda c: (lambda number, key: number is not None and number == (numbers_of(c) or [None])[0]),
        lambda c: (lambda number, key: number is not None and number == c.number),
    )
    for make in passes:
        for chapter in chapters:
            if not chapter.audio:
                pick(chapter, make(chapter))
    leftover = [os.path.basename(path) for path, _n, _k in files if path not in taken]
    if leftover:
        logger.warning("%d audio file(s) did not match any chapter of the progress file and are left out: %s",
                       len(leftover), ", ".join(leftover[:10]) + (" ..." if len(leftover) > 10 else ""))


def match_chapters(files: list[str], book: Book, allow_missing: bool) -> list[Chapter]:
    """Pairs chapter audio and subtitle files with the chapters of the book.

    Raises
    ------
    CombineError
        If no audio is found, or a chapter of the progress file has no audio
        and *allow_missing* is False.
    """
    audio = [f for f in files if f.lower().endswith(_AUDIO_EXTENSIONS)]
    subtitles = [f for f in files if f.lower().endswith(_SUBTITLE_EXTENSIONS)]
    if not audio:
        raise CombineError("No chapter audio files found in the given folders or ZIP files.")

    by_number: dict[int, list[str]] = {}
    unnumbered: list[str] = []
    for path in sorted(audio, key=_natural_key):
        number, _title = _chapter_number(path)
        if number is None:
            unnumbered.append(path)
        else:
            by_number.setdefault(number, []).append(path)

    if book.chapters:
        chapters = [Chapter(number, title or f"Chapter {number}") for number, title, _status in book.chapters]
    elif by_number:
        chapters = [Chapter(number, _chapter_number(paths[0])[1]) for number, paths in sorted(by_number.items())]
        if unnumbered:
            logger.warning("Ignoring files that are not named after a chapter: %s",
                           ", ".join(os.path.basename(p) for p in unnumbered))
    else:
        # No progress file and no chapter numbers: file-name order.
        chapters = [Chapter(index, _chapter_number(path)[1], audio=path) for index, path in enumerate(unnumbered, 1)]
        by_number = {chapter.number: [chapter.audio] for chapter in chapters}

    if book.chapters:
        _assign_by_title(chapters, audio)
    else:
        for chapter in chapters:
            candidates = by_number.get(chapter.number, [])
            if len(candidates) > 1:
                # The same chapter downloaded twice (or in two formats): keep the largest file.
                candidates = sorted(candidates, key=os.path.getsize, reverse=True)
                logger.warning("Chapter %d appears %d times; using %s", chapter.number, len(candidates),
                               os.path.basename(candidates[0]))
            if candidates:
                chapter.audio = candidates[0]

    missing = [chapter for chapter in chapters if not chapter.audio]
    if missing and not allow_missing:
        listed = ", ".join(f"{c.number} ({c.title})" for c in missing[:20])
        more = f" and {len(missing) - 20} more" if len(missing) > 20 else ""
        raise CombineError(
            f"{len(missing)} chapter(s) have no audio file: {listed}{more}. Add the missing parts, "
            "or pass --allow-missing to combine the book without them."
        )
    for chapter in missing:
        logger.warning("Chapter %d (%s) has no audio and is left out.", chapter.number, chapter.title)
    chapters = [chapter for chapter in chapters if chapter.audio]

    # Subtitles: same file stem as the audio first, otherwise the same chapter number.
    by_stem = {os.path.splitext(os.path.basename(p))[0].lower() + os.path.splitext(p)[1].lower(): p for p in subtitles}
    for chapter in chapters:
        stem = os.path.splitext(os.path.basename(chapter.audio))[0].lower()
        for extension in _SUBTITLE_EXTENSIONS:
            path = by_stem.get(stem + extension)
            if path is None:
                path = next((p for p in subtitles if p.lower().endswith(extension)
                             and _chapter_number(p)[0] == chapter.number), None)
            if path:
                chapter.subtitles[extension] = path
    return chapters


# ── Audio ─────────────────────────────────────────────────────────────────────

def _require(tool: str) -> str:
    found = shutil.which(tool)
    if not found:
        raise CombineError(f"{tool} was not found. Install FFmpeg (it includes {tool}) and try again.")
    return found


def probe(path: str) -> tuple[float, str, int, int]:
    """``(seconds, codec, sample_rate, channels)`` of the first audio stream of *path*."""
    seconds, codec, rate, channels, _bitrate = probe_full(path)
    return seconds, codec, rate, channels


def probe_full(path: str) -> tuple[float, str, int, int, int]:
    """``(seconds, codec, sample_rate, channels, bit_rate)`` of the first audio stream of *path*."""
    result = subprocess.run(
        [_require("ffprobe"), "-v", "error", "-select_streams", "a:0",
         "-show_entries", "stream=codec_name,sample_rate,channels,bit_rate:format=duration,bit_rate",
         "-of", "json", path],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise CombineError(f"Cannot read {path}: {result.stderr.strip()[-300:]}")
    info = json.loads(result.stdout or "{}")
    stream = (info.get("streams") or [{}])[0]
    try:
        seconds = float((info.get("format") or {}).get("duration") or 0.0)
    except (TypeError, ValueError):
        seconds = 0.0
    if seconds <= 0:
        raise CombineError(f"{path} has no audio.")
    bitrate = stream.get("bit_rate") or (info.get("format") or {}).get("bit_rate") or 0
    try:
        bitrate = int(bitrate)
    except (TypeError, ValueError):
        bitrate = 0
    return (seconds, str(stream.get("codec_name") or ""), int(stream.get("sample_rate") or 0),
            int(stream.get("channels") or 0), bitrate)


def _metadata_escape(value: str) -> str:
    """Escapes a value for an FFmpeg metadata file."""
    return re.sub(r"([=;#\\\n])", r"\\\1", value or "")


def _write_metadata(path: str, book: Book, chapters: list[Chapter], with_chapters: bool) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(";FFMETADATA1\n")
        fh.write(f"title={_metadata_escape(book.title)}\nalbum={_metadata_escape(book.title)}\n")
        if book.author:
            fh.write(f"artist={_metadata_escape(book.author)}\nalbum_artist={_metadata_escape(book.author)}\n")
        fh.write("genre=Audiobook\n")
        if not with_chapters:
            return
        start = 0
        for chapter in chapters:
            end = start + max(1, int(round(chapter.seconds * 1000)))
            fh.write(f"\n[CHAPTER]\nTIMEBASE=1/1000\nSTART={start}\nEND={end}\ntitle={_metadata_escape(chapter.title)}\n")
            start = end


def _cover_file(book: Book, cover: str, files: list[str], scratch: str) -> str:
    """Path of the cover image to embed, or ""."""
    if cover:
        if not os.path.isfile(cover):
            raise CombineError(f"Cover image not found: {cover}")
        return cover
    if book.cover_b64:
        try:
            data = base64.b64decode(book.cover_b64.split(",", 1)[-1], validate=False)
        except (ValueError, TypeError):
            data = b""
        if data:
            extension = ".png" if data[:8] == b"\x89PNG\r\n\x1a\n" else ".jpg"
            path = os.path.join(scratch, "cover" + extension)
            with open(path, "wb") as fh:
                fh.write(data)
            return path
    return next((f for f in files if os.path.basename(f).lower() in _COVER_NAMES), "")


def combine_audio(chapters: list[Chapter], book: Book, output: str, cover: str, bitrate_kbps: int, scratch: str) -> bool:
    """Writes the combined audio file. Returns True when the parts were stream-copied."""
    ffmpeg = _require("ffmpeg")
    output_format = os.path.splitext(output)[1].lstrip(".").lower()
    if output_format not in _ENCODERS:
        raise CombineError(f"Unsupported output format: .{output_format}")

    def family(extension: str) -> str:
        extension = extension.lstrip(".").lower()
        return "mp4" if extension in ("m4b", "m4a", "mp4") else extension

    kinds: dict[str, tuple[str, str, int, int]] = {}
    bitrates: dict[tuple[str, str, int, int], list[int]] = {}
    for chapter in chapters:
        chapter.seconds, codec, rate, channels, bitrate = probe_full(chapter.audio)
        kind = (family(os.path.splitext(chapter.audio)[1]), codec, rate, channels)
        kinds[chapter.audio] = kind
        bitrates.setdefault(kind, []).append(bitrate)
    # The format most chapters share. When it suits the output, only the
    # chapters that differ are converted to it and everything is then copied:
    # re-encoding a whole book because one chapter differs takes hours.
    majority = max(bitrates, key=lambda kind: len(bitrates[kind]))
    encoder_codec = {"mp4": "aac", "aac": "aac", "mp3": "mp3", "ogg": "opus", "opus": "opus", "webm": "opus",
                     "flac": "flac", "wav": "pcm_s16le"}.get(family(output_format))
    copy = majority[0] == family(output_format) and (majority[1] == encoder_codec or len(bitrates) == 1)
    odd = [chapter for chapter in chapters if kinds[chapter.audio] != majority] if copy else []
    parts = {chapter.audio: chapter.audio for chapter in chapters}
    if odd:
        kept = sorted(b for b in bitrates[majority] if b) or [bitrate_kbps * 1000]
        target_bitrate = kept[len(kept) // 2]
        logger.info("%d chapter(s) differ from the rest (%s, %d Hz, %d channel(s)); converting only those.",
                    len(odd), majority[1], majority[2], majority[3])
        for index, chapter in enumerate(odd):
            converted = os.path.join(scratch, f"conformed_{index:04d}.{output_format}")
            cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error", "-i", chapter.audio, "-vn",
                   *_ENCODERS[output_format], "-ar", str(majority[2]), "-ac", str(majority[3])]
            if output_format not in _LOSSLESS_FORMATS:
                cmd += ["-b:a", str(target_bitrate)]
            cmd += (["-f", "mp4"] if output_format == "m4b" else []) + [converted]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise CombineError(f"Could not convert {chapter.audio}: {result.stderr.strip()[-300:]}")
            parts[chapter.audio] = converted
            chapter.seconds = probe(converted)[0]

    with_chapters = output_format in _CHAPTER_FORMATS
    metadata = os.path.join(scratch, "metadata.txt")
    _write_metadata(metadata, book, chapters, with_chapters)
    use_cover = cover if output_format in _COVER_FORMATS else ""

    def command(include_cover: bool) -> list[str]:
        cmd = [ffmpeg, "-y", "-hide_banner", "-loglevel", "error"]
        if copy:
            listing = os.path.join(scratch, "parts.txt")
            with open(listing, "w", encoding="utf-8") as fh:
                for chapter in chapters:
                    fh.write("file '" + os.path.abspath(parts[chapter.audio]).replace("'", "'\\''") + "'\n")
            cmd += ["-f", "concat", "-safe", "0", "-i", listing]
            audio_inputs = 1
        else:
            for chapter in chapters:
                cmd += ["-i", chapter.audio]
            audio_inputs = len(chapters)
        cmd += ["-i", metadata]
        metadata_index = audio_inputs
        if include_cover:
            cmd += ["-i", use_cover]
        if copy:
            cmd += ["-map", "0:a"]
        else:
            joined = "".join(f"[{i}:a:0]" for i in range(audio_inputs))
            cmd += ["-filter_complex", f"{joined}concat=n={audio_inputs}:v=0:a=1[book]", "-map", "[book]"]
        cmd += ["-map_metadata", str(metadata_index), "-map_chapters", str(metadata_index) if with_chapters else "-1"]
        if include_cover:
            cmd += ["-map", f"{metadata_index + 1}:v", "-c:v", "copy", "-disposition:v", "attached_pic"]
            if output_format == "mp3":
                cmd += ["-id3v2_version", "3"]
        if copy:
            cmd += ["-c:a", "copy"]
        else:
            cmd += _ENCODERS[output_format]
            if output_format not in _LOSSLESS_FORMATS:
                cmd += ["-b:a", f"{bitrate_kbps}k"]
        if output_format == "m4b":
            cmd += ["-f", "mp4"]
        cmd.append(output)
        return cmd

    attempts = [bool(use_cover), False] if use_cover else [False]
    error = ""
    for include_cover in attempts:
        result = subprocess.run(command(include_cover), capture_output=True, text=True)
        if result.returncode == 0:
            if use_cover and not include_cover:
                logger.warning("The cover could not be embedded; the book was written without it.")
            return copy
        error = result.stderr.strip()[-600:]
    raise CombineError(f"FFmpeg could not combine the chapters: {error}")


# ── Subtitles ─────────────────────────────────────────────────────────────────

def _seconds_from(match: re.Match, hours_group: bool = True) -> float:
    hours = int(match.group(1) or 0) if hours_group else 0
    minutes, seconds, fraction = (int(match.group(2)), int(match.group(3)), match.group(4))
    return hours * 3600 + minutes * 60 + seconds + int(fraction.ljust(3, "0")[:3]) / 1000


def _srt_time(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def _vtt_time(seconds: float) -> str:
    return _srt_time(seconds).replace(",", ".")


def _lrc_time(seconds: float) -> str:
    centis = int(round(seconds * 100))
    return f"[{centis // 6000:02d}:{centis // 100 % 60:02d}.{centis % 100:02d}]"


def _read(path: str) -> str:
    with open(path, encoding="utf-8-sig", errors="replace") as fh:
        return fh.read().replace("\r\n", "\n").replace("\r", "\n")


def _clamp(start: float, end: float, limit: float | None) -> tuple[float, float]:
    """Keeps a cue inside its chapter: subtitles may run a little past the chapter's audio."""
    if limit is None:
        return start, end
    start = min(start, limit)
    return start, max(start, min(end, limit))


def shift_srt(text: str, offset: float, first_index: int, limit: float | None = None) -> tuple[list[str], int]:
    """SRT cues moved by *offset* seconds and renumbered from *first_index*.

    *limit* is the end of the chapter in book time; no cue goes past it, so
    a chapter's subtitles never overlap the next chapter's.
    """
    blocks, index = [], first_index
    for block in re.split(r"\n\s*\n", text.strip()):
        lines = block.split("\n")
        timing = next((i for i, line in enumerate(lines) if "-->" in line), None)
        if timing is None:
            continue
        times = list(_SRT_TIME.finditer(lines[timing]))
        if len(times) < 2:
            continue
        start, end = _clamp(*(_seconds_from(t) + offset for t in times[:2]), limit)
        body = lines[timing + 1:]
        blocks.append(f"{index}\n{_srt_time(start)} --> {_srt_time(end)}\n" + "\n".join(body))
        index += 1
    return blocks, index


def shift_vtt(text: str, offset: float, limit: float | None = None) -> list[str]:
    """WebVTT cues moved by *offset* seconds (the header and notes are dropped)."""
    cues = []
    for block in re.split(r"\n\s*\n", text.strip()):
        lines = block.split("\n")
        timing = next((i for i, line in enumerate(lines) if "-->" in line), None)
        if timing is None:
            continue
        left, _arrow, right = lines[timing].partition("-->")
        start_match, end_match = _VTT_TIME.search(left), _VTT_TIME.search(right)
        if not start_match or not end_match:
            continue
        settings = right[end_match.end():].rstrip()
        start, end = _clamp(_seconds_from(start_match) + offset, _seconds_from(end_match) + offset, limit)
        cues.append(f"{_vtt_time(start)} --> {_vtt_time(end)}{settings}\n" + "\n".join(lines[timing + 1:]))
    return cues


def shift_lrc(text: str, offset: float, limit: float | None = None) -> list[str]:
    """LRC lines moved by *offset* seconds (per-file tags such as [ti:] are dropped)."""
    lines = []
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped or _LRC_META.match(stripped) and not _LRC_TAG.match(stripped):
            continue
        tags = list(_LRC_TAG.finditer(stripped))
        if not tags or tags[0].start() != 0:
            continue
        lyric = stripped[tags[-1].end():]
        for tag in tags:
            minutes, seconds, fraction = int(tag.group(1)), int(tag.group(2)), tag.group(3) or "0"
            value = minutes * 60 + seconds + int(fraction.ljust(3, "0")[:3]) / 1000 + offset
            if limit is not None:
                value = min(value, limit)
            lines.append(f"{_lrc_time(value)}{lyric}")
    return lines


def combine_subtitles(chapters: list[Chapter], book: Book, output: str) -> list[str]:
    """Writes one subtitle file per type found; returns their paths."""
    written = []
    base = os.path.splitext(output)[0]
    for extension in _SUBTITLE_EXTENSIONS:
        with_type = [c for c in chapters if extension in c.subtitles]
        if not with_type:
            continue
        if len(with_type) < len(chapters):
            logger.warning("%d of %d chapters have a %s file; the others are left without subtitles.",
                           len(with_type), len(chapters), extension)
        offset, srt_blocks, vtt_cues, lrc_lines, next_index = 0.0, [], [], [], 1
        for chapter in chapters:
            path = chapter.subtitles.get(extension)
            if path:
                text = _read(path)
                if extension == ".srt":
                    blocks, next_index = shift_srt(text, offset, next_index, offset + chapter.seconds)
                    srt_blocks += blocks
                elif extension == ".vtt":
                    vtt_cues += shift_vtt(text, offset, offset + chapter.seconds)
                else:
                    lrc_lines += shift_lrc(text, offset, offset + chapter.seconds)
            offset += chapter.seconds
        target = base + extension
        with open(target, "w", encoding="utf-8") as fh:
            if extension == ".srt":
                fh.write("\n\n".join(srt_blocks) + "\n")
            elif extension == ".vtt":
                fh.write("WEBVTT\n\n" + "\n\n".join(vtt_cues) + "\n")
            else:
                header = [f"[ti:{book.title}]"] + ([f"[ar:{book.author}]"] if book.author else [])
                fh.write("\n".join(header + lrc_lines) + "\n")
        written.append(target)
    return written


# ── Command line ──────────────────────────────────────────────────────────────

def _safe_name(title: str) -> str:
    name = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', " ", title).strip(" .")
    return re.sub(r"\s+", " ", name)[:150] or "Audiobook"


def combine(argv: list[str] | None = None) -> int:
    """Runs the tool; returns the process exit code."""
    parser = argparse.ArgumentParser(
        description="Join the chapter files of an audiobook (folders or ZIP downloads) into one file "
                    "with chapter markers, and merge its subtitles.",
    )
    parser.add_argument("inputs", nargs="+", help="Folders and/or ZIP files holding the chapter files.")
    parser.add_argument("--progress", default="", help="The book's generation_progress.json "
                        "(default: one found among the inputs). Gives chapter order, titles, author and cover.")
    parser.add_argument("-o", "--output", default="", help="Output file (default: '<book title>.<format>' "
                        "in the current folder).")
    parser.add_argument("--format", default="", help="Output format, e.g. m4b or mp3 "
                        "(default: the book's format, else that of the chapter files).")
    parser.add_argument("--title", default="", help="Book title (overrides the progress file).")
    parser.add_argument("--author", default="", help="Author (overrides the progress file).")
    parser.add_argument("--cover", default="", help="Cover image to embed.")
    parser.add_argument("--bitrate", type=int, default=0, help="Bitrate in kbps when the parts must be re-encoded.")
    parser.add_argument("--allow-missing", action="store_true", help="Combine even if some chapters have no audio.")
    parser.add_argument("--no-subtitles", action="store_true", help="Do not merge subtitle files.")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    scratch = tempfile.mkdtemp(prefix="combine_book_")
    try:
        files = _collect(args.inputs, scratch)
        progress = args.progress or next((f for f in files if os.path.basename(f) == _PROGRESS_NAME), "")
        book = load_progress(progress) if progress else Book()
        if not progress:
            logger.info("No %s given or found: chapters are ordered by their numbers.", _PROGRESS_NAME)
        book.title = args.title or book.title
        book.author = args.author or book.author

        chapters = match_chapters(files, book, args.allow_missing)
        output_format = (args.format or (os.path.splitext(args.output)[1] if args.output else "")
                         or book.output_format or os.path.splitext(chapters[0].audio)[1]).lstrip(".").lower()
        output = args.output or f"{_safe_name(book.title)}.{output_format}"
        if not output.lower().endswith("." + output_format):
            output = f"{os.path.splitext(output)[0]}.{output_format}"
        if os.path.abspath(output) in {os.path.abspath(c.audio) for c in chapters}:
            raise CombineError(f"The output {output} would overwrite one of the chapter files.")

        cover = _cover_file(book, args.cover, files, scratch)
        bitrate = args.bitrate or book.bitrate_kbps or _DEFAULT_BITRATE_KBPS
        logger.info("Combining %d chapter(s) of '%s' into %s ...", len(chapters), book.title, output)
        copied = combine_audio(chapters, book, output, cover, bitrate, scratch)
        total = sum(chapter.seconds for chapter in chapters)
        logger.info("Audio: %s (%d:%02d:%02d, %s)", output, total // 3600, total // 60 % 60, total % 60,
                    "stream-copied" if copied else f"re-encoded at {bitrate} kbps")
        if not args.no_subtitles:
            for path in combine_subtitles(chapters, book, output):
                logger.info("Subtitles: %s", path)
        return 0
    except CombineError as exc:
        logger.error("%s", exc)
        return 1
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(combine())
