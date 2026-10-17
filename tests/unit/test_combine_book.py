"""
test_combine_book.py
====================
The standalone book combiner (``scripts/combine_book.py``): chapter files
downloaded in parts are joined into one audiobook with chapter markers, and
their subtitles into one file per type.
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
import zipfile

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_SCRIPT = os.path.join(_ROOT, "scripts", "combine_book.py")

pytestmark = pytest.mark.skipif(not shutil.which("ffmpeg") or not shutil.which("ffprobe"),
                                reason="FFmpeg is needed")


def _load():
    spec = importlib.util.spec_from_file_location("combine_book", _SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["combine_book"] = module
    spec.loader.exec_module(module)
    return module


cb = _load()

_TITLES = {1: "Prologue", 2: "Chapter 1: The Salt Road", 3: "Chapter 2: Plan B"}


def _tone(path, seconds: float, extension: str = "mp3") -> None:
    codec = {"mp3": ["-c:a", "libmp3lame", "-b:a", "64k"], "m4b": ["-c:a", "aac", "-b:a", "64k", "-f", "mp4"],
             "wav": ["-c:a", "pcm_s16le"]}[extension]
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i",
                    f"sine=frequency=440:duration={seconds}", "-ac", "1", "-ar", "24000", *codec, str(path)],
                   check=True)


def _srt(path, lines) -> None:
    blocks = [f"{i}\n00:00:0{start},000 --> 00:00:0{end},000\n{text}" for i, (start, end, text) in enumerate(lines, 1)]
    path.write_text("\n\n".join(blocks) + "\n", encoding="utf-8")


def _progress(path, chapters=_TITLES, **settings) -> None:
    data = {
        "book_title": "The Lighthouse",
        "settings": {"book_title": "The Lighthouse", "author": "Mara Ellison", "output_format": "mp3",
                     "bitrate_kbps": 64, **settings},
        "chapters": [{"num": n, "title": t, "status": "completed"} for n, t in chapters.items()],
    }
    path.write_text(json.dumps(data), encoding="utf-8")


def _chapters_of(path) -> list[tuple[str, float, float]]:
    out = subprocess.run(["ffprobe", "-v", "error", "-show_chapters", "-show_format", "-of", "json", str(path)],
                         capture_output=True, text=True, check=True).stdout
    data = json.loads(out)
    return [(c["tags"]["title"], float(c["start_time"]), float(c["end_time"])) for c in data["chapters"]], data["format"]


@pytest.fixture
def downloads(tmp_path):
    """Three chapters downloaded in two parts: a folder and a ZIP file."""
    part1, part2 = tmp_path / "part1", tmp_path / "part2_files"
    part1.mkdir()
    part2.mkdir()
    _tone(part1 / "Chapter 1 - Prologue.mp3", 2)
    _srt(part1 / "Chapter 1 - Prologue.srt", [(0, 1, "The tide went out."), (1, 2, "Nobody watched.")])
    (part1 / "Chapter 1 - Prologue.lrc").write_text("[ti:Prologue]\n[00:00.00]The tide went out.\n[00:01.00]Nobody watched.\n")
    _tone(part2 / "Chapter 2 - The Salt Road.mp3", 3)
    _srt(part2 / "Chapter 2 - The Salt Road.srt", [(0, 2, "Dr. Hale arrived.")])
    (part2 / "Chapter 2 - The Salt Road.lrc").write_text("[00:00.50]Dr. Hale arrived.\n")
    _tone(part2 / "Chapter 3 - Plan B.mp3", 1)
    _srt(part2 / "Chapter 3 - Plan B.srt", [(0, 1, "Plan A had been simple.")])
    archive = tmp_path / "part2.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for name in os.listdir(part2):
            zf.write(part2 / name, f"audiobook_output/The Lighthouse/{name}")
    shutil.rmtree(part2)
    _progress(tmp_path / "generation_progress.json")
    return tmp_path, part1, archive


class TestCombine:

    def test_parts_become_one_book_with_chapter_markers(self, downloads):
        root, part1, archive = downloads
        out = root / "book.m4b"
        assert cb.combine([str(part1), str(archive), "--progress", str(root / "generation_progress.json"),
                           "-o", str(out)]) == 0
        chapters, fmt = _chapters_of(out)
        assert [title for title, _s, _e in chapters] == list(_TITLES.values())   # titles from the progress file
        assert chapters[1][1] == pytest.approx(2.0, abs=0.15)
        assert chapters[2][1] == pytest.approx(5.0, abs=0.2)
        assert float(fmt["duration"]) == pytest.approx(6.0, abs=0.3)
        assert fmt["tags"]["title"] == "The Lighthouse"
        assert fmt["tags"].get("artist") == "Mara Ellison"

    def test_subtitles_are_shifted_to_their_place_in_the_book(self, downloads):
        root, part1, archive = downloads
        out = root / "book.mp3"
        assert cb.combine([str(part1), str(archive), "--progress", str(root / "generation_progress.json"),
                           "-o", str(out)]) == 0
        srt = (root / "book.srt").read_text(encoding="utf-8")
        blocks = srt.strip().split("\n\n")
        assert [b.split("\n")[0] for b in blocks] == ["1", "2", "3", "4"]       # renumbered
        assert "Dr. Hale arrived." in blocks[2]
        start = blocks[2].split("\n")[1].split(" --> ")[0]
        assert start.startswith("00:00:02,0") or start.startswith("00:00:01,9")  # after the 2 s prologue
        assert "00:00:05" in blocks[3] or "00:00:04,9" in blocks[3]             # after 2 s + 3 s
        lrc = (root / "book.lrc").read_text(encoding="utf-8").splitlines()
        assert lrc[0] == "[ti:The Lighthouse]" and lrc[1] == "[ar:Mara Ellison]"
        assert "[ti:Prologue]" not in lrc                                          # per-chapter headers dropped
        assert any(line.startswith("[00:02.") and "Dr. Hale" in line for line in lrc)

    def test_same_format_is_stream_copied(self, downloads):
        root, part1, archive = downloads
        out = root / "book.mp3"
        assert cb.combine([str(part1), str(archive), "--progress", str(root / "generation_progress.json"),
                           "-o", str(out)]) == 0
        book = cb.load_progress(str(root / "generation_progress.json"))
        files = cb._collect([str(part1), str(archive)], str(root / "scratch_check"))
        chapters = cb.match_chapters(files, book, allow_missing=False)
        assert cb.combine_audio(chapters, book, str(root / "again.mp3"), "", 64, str(root)) is True

    def test_progress_file_is_found_among_the_inputs(self, downloads):
        root, part1, archive = downloads
        shutil.copy(root / "generation_progress.json", part1 / "generation_progress.json")
        out = root / "found.m4b"
        assert cb.combine([str(part1), str(archive), "-o", str(out)]) == 0
        assert [t for t, _s, _e in _chapters_of(out)[0]] == list(_TITLES.values())

    def test_default_name_and_format_come_from_the_progress_file(self, downloads, monkeypatch):
        root, part1, archive = downloads
        monkeypatch.chdir(root)
        assert cb.combine([str(part1), str(archive), "--progress", "generation_progress.json"]) == 0
        assert (root / "The Lighthouse.mp3").exists()   # output_format "mp3" in the settings


class TestProblems:

    def test_missing_chapter_stops_with_its_name(self, downloads, caplog):
        root, part1, _archive = downloads
        assert cb.combine([str(part1), "--progress", str(root / "generation_progress.json"),
                           "-o", str(root / "x.m4b")]) == 1
        assert "2 chapter(s) have no audio file" in caplog.text and "Plan B" in caplog.text
        assert not (root / "x.m4b").exists()

    def test_allow_missing_combines_what_is_there(self, downloads):
        root, part1, _archive = downloads
        out = root / "partial.m4b"
        assert cb.combine([str(part1), "--progress", str(root / "generation_progress.json"),
                           "-o", str(out), "--allow-missing"]) == 0
        assert [t for t, _s, _e in _chapters_of(out)[0]] == ["Prologue"]

    def test_no_audio_found(self, tmp_path, caplog):
        (tmp_path / "notes.txt").write_text("nothing here")
        assert cb.combine([str(tmp_path), "-o", str(tmp_path / "x.mp3")]) == 1
        assert "No chapter audio files found" in caplog.text

    def test_zip_entries_cannot_escape_the_scratch_folder(self, tmp_path):
        evil = tmp_path / "evil.zip"
        with zipfile.ZipFile(evil, "w") as zf:
            zf.writestr("../../outside.mp3", b"x")
        scratch = tmp_path / "scratch"
        scratch.mkdir()
        files = cb._collect([str(evil)], str(scratch))
        assert files == []
        assert not (tmp_path / "outside.mp3").exists()


class TestWithoutProgressFile:

    def test_chapters_are_ordered_by_number_not_by_name(self, tmp_path):
        for number, seconds in ((10, 1), (2, 2), (1, 1)):
            _tone(tmp_path / f"Chapter {number} - Part {number}.mp3", seconds)
        out = tmp_path / "out" / "book.mp3"
        out.parent.mkdir()
        assert cb.combine([str(tmp_path), "-o", str(out), "--title", "No JSON"]) == 0
        chapters, fmt = _chapters_of(out)
        assert [t for t, _s, _e in chapters] == ["Part 1", "Part 2", "Part 10"]
        assert fmt["tags"]["title"] == "No JSON"

    def test_mixed_formats_are_re_encoded(self, tmp_path):
        _tone(tmp_path / "Chapter 1 - One.mp3", 1)
        _tone(tmp_path / "Chapter 2 - Two.wav", 1, "wav")
        out = tmp_path / "out" / "book.m4b"
        out.parent.mkdir()
        assert cb.combine([str(tmp_path), "-o", str(out)]) == 0
        chapters, fmt = _chapters_of(out)
        assert len(chapters) == 2 and float(fmt["duration"]) == pytest.approx(2.0, abs=0.2)


class TestSubtitleShifting:

    def test_srt(self):
        blocks, next_index = cb.shift_srt("1\n00:00:01,500 --> 00:00:02,000\nHello\n", 60.0, 7)
        assert blocks == ["7\n00:01:01,500 --> 00:01:02,000\nHello"] and next_index == 8

    def test_vtt_keeps_cue_settings(self):
        cues = cb.shift_vtt("WEBVTT\n\n00:01.000 --> 00:02.500 align:start\nHi\n", 3600.0)
        assert cues == ["01:00:01.000 --> 01:00:02.500 align:start\nHi"]

    def test_lrc_with_several_time_tags(self):
        assert cb.shift_lrc("[ar:x]\n[00:01.50][00:03.00]Again\n", 10.0) == ["[00:11.50]Again", "[00:13.00]Again"]

    def test_cue_running_past_the_chapter_is_kept_inside_it(self):
        # A real book had a last subtitle starting a second after its chapter's
        # audio ended, so it overlapped the next chapter in the combined file.
        text = "1\n00:14:18,920 --> 00:14:22,040\nLast line.\n"
        blocks, _ = cb.shift_srt(text, 100.0, 1, limit=100.0 + 857.84)
        assert blocks == ["1\n00:15:57,840 --> 00:15:57,840\nLast line."]
        assert cb.shift_lrc("[14:18.92]Last line.\n", 0.0, limit=857.84) == ["[14:17.84]Last line."]


class TestMatchingByTitle:
    """File names may carry a different number than the progress file (a real book did)."""

    def test_titles_decide_and_numbers_break_ties(self, tmp_path):
        names = {"Chapter 1 - Crimson.mp3": 1, "Chapter 109 - Deduction.mp3": 1, "Chapter 175 Deduction.mp3": 1,
                 "Chapter 174 - Madam Sharon.mp3": 1}
        for name in names:
            _tone(tmp_path / name, 1)
        progress = tmp_path / "generation_progress.json"
        _progress(progress, {2: "Chapter 1: Crimson", 110: "Chapter 109: Deduction",
                             175: "Chapter 174: Madam Sharon", 176: "Chapter 175: Deduction"})
        book = cb.load_progress(str(progress))
        chapters = cb.match_chapters(cb._collect([str(tmp_path)], str(tmp_path)), book, allow_missing=False)
        found = {c.number: os.path.basename(c.audio) for c in chapters}
        assert found == {2: "Chapter 1 - Crimson.mp3", 110: "Chapter 109 - Deduction.mp3",
                         175: "Chapter 174 - Madam Sharon.mp3", 176: "Chapter 175 Deduction.mp3"}

    def test_newer_names_with_the_entry_number_still_match(self, tmp_path):
        _tone(tmp_path / "Chapter 2 - Chapter 1 Crimson.mp3", 1)
        progress = tmp_path / "generation_progress.json"
        _progress(progress, {2: "Chapter 1: Crimson"})
        chapters = cb.match_chapters(cb._collect([str(tmp_path)], str(tmp_path)),
                                     cb.load_progress(str(progress)), allow_missing=False)
        assert os.path.basename(chapters[0].audio) == "Chapter 2 - Chapter 1 Crimson.mp3"


class TestMixedParts:

    def test_only_the_odd_chapter_is_converted_and_the_rest_copied(self, tmp_path):
        for n in (2, 3):
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "sine=duration=1",
                            "-ac", "1", "-ar", "24000", "-c:a", "aac", "-b:a", "64k", "-f", "mp4",
                            str(tmp_path / f"Chapter {n} - Part {n}.m4b")], check=True)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "sine=duration=1",
                        "-ac", "2", "-ar", "44100", "-c:a", "aac", "-b:a", "128k", "-f", "mp4",
                        str(tmp_path / "Chapter 1 - Part 1.m4b")], check=True)
        out = tmp_path / "out" / "book.m4b"
        out.parent.mkdir()
        book = cb.Book(title="Mixed")
        chapters = cb.match_chapters(cb._collect([str(tmp_path)], str(tmp_path)), book, allow_missing=False)
        scratch = tmp_path / "scratch"
        scratch.mkdir()
        assert cb.combine_audio(chapters, book, str(out), "", 64, str(scratch)) is True   # copied after conforming
        info = json.loads(subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-show_chapters", "-of", "json",
                                          str(out)], capture_output=True, text=True, check=True).stdout)
        stream = info["streams"][0]
        assert (stream["sample_rate"], stream["channels"]) == ("24000", 1)
        assert len(info["chapters"]) == 3

