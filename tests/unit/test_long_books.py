"""
test_long_books.py
==================
The twenty-page test books in seven languages
(``tests/fixture_generation/generate_long_books.py``): the built EPUBs match
their sources, extract into ten chapters, and are cut into chunks of a
sensible spoken length whatever the script.
"""

from __future__ import annotations

import json
import os
import sys
import zipfile

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from audiobook_factory.chunk_planner import plan_chunks
from audiobook_factory.chunk_verifier import expected_seconds
from audiobook_factory.text_extractor import extract
from tests.fixture_generation import generate_long_books as books

_FIXTURES: str = books.OUTPUT_DIR
# About twenty printed pages of narration.
_MIN_MINUTES: float = 25.0
_MAX_MINUTES: float = 60.0
# 399 Latin characters are about 26 seconds; no chunk may run far past that.
_MAX_CHUNK_SECONDS: float = 40.0


def _zip_members(path: str) -> dict[str, bytes]:
    """Name and uncompressed content of every member of a ZIP file."""
    with zipfile.ZipFile(path) as archive:
        return {name: archive.read(name) for name in archive.namelist()}


def _manifest() -> dict:
    with open(os.path.join(_FIXTURES, books.MANIFEST_NAME), encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture(scope="module")
def extracted() -> dict[str, list]:
    """Each long book, extracted once."""
    result = {}
    for entry in _manifest()["books"]:
        chapters, _cover = extract(os.path.join(_FIXTURES, entry["file"]), enable_ocr=False, log_fn=lambda *_: None)
        result[entry["code"]] = [chapter for chapter in chapters if chapter.text.strip()]
    return result


def test_one_book_per_language():
    codes = [entry["code"] for entry in _manifest()["books"]]
    assert codes == list(books.BOOK_CODES) == ["en", "fr", "ru", "hi", "zh", "ja", "ko"]
    languages = {entry["language"] for entry in _manifest()["books"]}
    assert languages == {"English", "French", "Russian", "Hindi", "Chinese", "Japanese", "Korean"}


def test_built_files_are_up_to_date(tmp_path):
    for code in books.BOOK_CODES:
        source = books.load_source(code)
        rebuilt = tmp_path / books.book_file_name(code)
        books.build_long_epub(source, str(rebuilt))
        committed = os.path.join(_FIXTURES, books.book_file_name(code))
        # Compared member by member: the compressed bytes differ between zlib
        # builds (this test failed on Kaggle when it compared whole files).
        assert _zip_members(str(rebuilt)) == _zip_members(committed), (
            f"{books.book_file_name(code)} is stale — run: python tests/fixture_generation/generate_long_books.py"
        )
        assert books.describe(source) == next(e for e in _manifest()["books"] if e["code"] == code)


@pytest.mark.parametrize("code", books.BOOK_CODES)
def test_extraction_finds_the_ten_chapters(extracted, code):
    entry = next(e for e in _manifest()["books"] if e["code"] == code)
    assert [chapter.title for chapter in extracted[code]] == entry["chapters"]
    assert len(entry["chapters"]) == 10


@pytest.mark.parametrize("code", books.BOOK_CODES)
def test_book_is_about_twenty_pages_of_narration(extracted, code):
    minutes = sum(expected_seconds(chapter.text) for chapter in extracted[code]) / 60
    assert _MIN_MINUTES <= minutes <= _MAX_MINUTES, f"{code}: {minutes:.0f} minutes of speech"


@pytest.mark.parametrize("code", books.BOOK_CODES)
def test_chunks_have_a_sensible_spoken_length(extracted, code):
    # A Chinese chunk of 399 characters would be over a minute of audio.
    chunks = [chunk for chapter in extracted[code] for chunk in plan_chunks(chapter.text, None, 399, 0.3, 0.8)]
    assert len(chunks) > 100
    longest = max(expected_seconds(chunk.text) for chunk in chunks)
    assert longest <= _MAX_CHUNK_SECONDS, f"{code}: longest chunk is about {longest:.0f}s"
    # Packing still happens: chunks are not single short sentences.
    assert sum(expected_seconds(chunk.text) for chunk in chunks) / len(chunks) >= 10.0
    # No word is lost or invented by the planner (a lone closing guillemet
    # after a space may go: it has nothing to say).
    def spoken(text: str) -> str:
        return "".join(char for char in text if char.isalnum())

    original = "".join(spoken(chapter.text) for chapter in extracted[code])
    planned = "".join(spoken(chunk.text) for chunk in chunks)
    assert planned == original


@pytest.mark.parametrize("code", ["zh", "ja"])
def test_spaceless_scripts_are_joined_without_spaces(extracted, code):
    packed = [
        chunk for chapter in extracted[code]
        for chunk in plan_chunks(chapter.text, None, 399, 0.3, 0.8) if len(chunk.sentences) > 1
    ]
    assert len(packed) > 50, "expected chunks holding several sentences"
    for chunk in packed:
        # Joining adds no space: the only spaces are those inside a sentence.
        assert chunk.text == "".join(chunk.sentences), chunk.text


@pytest.mark.parametrize("code", ["zh", "ja", "ko", "hi", "fr", "ru"])
def test_no_sentence_starts_or_ends_with_a_stray_quote(extracted, code):
    opening, closing = "“‘「『（«《", "”」』）»》"
    for chapter in extracted[code]:
        for chunk in plan_chunks(chapter.text, None, 399, 0.3, 0.8):
            for sentence in chunk.sentences:
                assert sentence[-1] not in opening, f"{code}: opening quote left at the end of {sentence!r}"
                assert sentence[0] not in closing, f"{code}: closing quote left at the start of {sentence!r}"


@pytest.mark.parametrize("code", ["fr", "ru"])
def test_dialogue_dashes_are_not_read_as_commas(extracted, code):
    # "— Rentre vite !" used to come out of extraction as ",  Rentre vite !".
    text = "\n\n".join(chapter.text for chapter in extracted[code])
    assert "\u2014" not in text
    for leftover in (". ,", "? ,", "! ,", ", ,", "\n,"):
        assert leftover not in text, f"{code}: {leftover!r} left where a dialogue dash was"

