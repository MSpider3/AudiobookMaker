"""
test_extraction_fixtures.py
===========================
End-to-end extraction tests: every document in
``tests/fixtures/source_documents`` goes through ``scan()`` and ``extract()``
and is checked against ``expected_chapters.json`` — chapter titles and order,
phrases that must / must not be narrated, per-chapter content, single-chapter
selection, and agreement between ``scan()`` and ``extract()``.

None of these tests needs Docling, EasyOCR or BookNLP.
"""

from __future__ import annotations

import json
import os
import re
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from audiobook_factory import text_processing  # noqa: E402
from audiobook_factory.text_extractor import ExtractedChapter, extract, scan  # noqa: E402

_FIXTURES = os.path.join(_ROOT, "tests", "fixtures", "source_documents")
_EXPECTED_PATH = os.path.join(_FIXTURES, "expected_chapters.json")

with open(_EXPECTED_PATH, encoding="utf-8") as _handle:
    EXPECTED: dict[str, dict] = json.load(_handle)

_NEEDS_MOBI = pytest.mark.skipif(
    __import__("importlib").util.find_spec("mobi") is None,
    reason="the 'mobi' package (pip install mobi) is needed to read .mobi files",
)
FIXTURE_PARAMS = [
    pytest.param(name, marks=_NEEDS_MOBI) if name.endswith(".mobi") else name
    for name in EXPECTED
]


def _flat(text: str) -> str:
    """Whitespace-insensitive form, so wrapping and paragraph breaks do not matter."""
    return " ".join(text.split())


def _quiet(_message: str) -> None:
    return None


@pytest.fixture(scope="module")
def extracted() -> dict[str, list[ExtractedChapter]]:
    """Each fixture is extracted once and shared by the tests of this module."""
    cache: dict[str, list[ExtractedChapter]] = {}

    def _get(name: str) -> list[ExtractedChapter]:
        if name not in cache:
            cache[name] = extract(os.path.join(_FIXTURES, name), log_fn=_quiet)[0]
        return cache[name]

    return _get  # type: ignore[return-value]


def test_expected_file_covers_every_document():
    # The long_book_* files are described by long_books.json and checked in
    # test_long_books.py; this file covers the small format fixtures.
    documents = {
        name for name in os.listdir(_FIXTURES)
        if name.rsplit(".", 1)[-1] in ("epub", "pdf", "docx", "odt", "txt", "mobi")
        and not name.startswith("long_book_")
    }
    assert documents == set(EXPECTED), "expected_chapters.json and the fixture directory disagree"
    for name in (
        "dummy_book.epub", "dummy_book.pdf", "dummy_book.docx", "dummy_book.txt", "dummy_book.mobi",
        "epub_no_toc.epub", "epub_multi_anchor_single_file.epub", "epub_percent_encoded_hrefs.epub",
        "epub_manifest_order_differs.epub", "epub_trailing_backmatter.epub",
    ):
        assert name in EXPECTED
        assert os.path.getsize(os.path.join(_FIXTURES, name)) <= 300 * 1024


@pytest.mark.parametrize("name", FIXTURE_PARAMS)
class TestEveryFixture:

    def test_extract_returns_expected_chapters_in_order(self, name, extracted):
        chapters = extracted(name)
        assert [c.title for c in chapters] == EXPECTED[name]["chapters"]
        assert [c.num for c in chapters] == list(range(1, len(chapters) + 1))
        for chapter in chapters:
            assert chapter.text.strip()
            assert chapter.sentences, f"{chapter.title!r} has no sentences"
            assert not chapter.probably_matter

    def test_scan_lists_the_same_chapters_as_extract(self, name, extracted):
        result = scan(os.path.join(_FIXTURES, name))
        assert result.file_type == EXPECTED[name]["format"]
        assert result.has_toc is True
        assert not result.warning
        chapters = extracted(name)
        assert [(c.num, c.title) for c in result.chapters] == [(c.num, c.title) for c in chapters]
        assert all(c.word_count > 0 and not c.probably_matter for c in result.chapters)

    def test_required_phrases_are_narrated(self, name, extracted):
        text = _flat(" ".join(c.text for c in extracted(name)))
        missing = [phrase for phrase in EXPECTED[name]["must_contain"] if _flat(phrase) not in text]
        assert not missing, f"missing from {name}: {missing}"

    def test_front_and_back_matter_is_not_narrated(self, name, extracted):
        text = _flat(" ".join(c.text for c in extracted(name)))
        leaked = [phrase for phrase in EXPECTED[name]["must_not_contain"] if _flat(phrase) in text]
        assert not leaked, f"narrated but must not be, in {name}: {leaked}"
        spoken = _flat(" ".join(" ".join(c.sentences) for c in extracted(name)))
        assert not [phrase for phrase in EXPECTED[name]["must_not_contain"] if _flat(phrase) in spoken]

    def test_each_chapter_holds_its_own_text(self, name, extracted):
        chapters = {c.title: _flat(c.text) for c in extracted(name)}
        for title, phrases in EXPECTED[name]["chapter_must_contain"].items():
            for phrase in phrases:
                assert _flat(phrase) in chapters[title], f"{phrase!r} not in chapter {title!r}"
                elsewhere = [t for t, text in chapters.items() if t != title and _flat(phrase) in text]
                assert not elsewhere, f"{phrase!r} also found in {elsewhere}"

    def test_no_markup_reaches_the_text(self, name, extracted):
        for chapter in extracted(name):
            for line in chapter.text.split("\n"):
                assert not re.match(r"\s*(?:#{1,6}\s|\||\* \* \*|-{3,}\s*$)", line), line
            assert "<" not in chapter.text and "&amp;" not in chapter.text
            assert "**" not in chapter.text and "_Unreliable_" not in chapter.text
            assert max(len(s) for s in chapter.sentences) <= 399

    def test_selecting_one_chapter_returns_exactly_that_chapter(self, name, extracted):
        everything = extracted(name)
        target = everything[len(everything) // 2]
        by_title, _ = extract(os.path.join(_FIXTURES, name), selections=[target.title], log_fn=_quiet)
        assert [(c.num, c.title) for c in by_title] == [(target.num, target.title)]
        assert by_title[0].text == target.text
        by_number, _ = extract(os.path.join(_FIXTURES, name), selections=[target.num], log_fn=_quiet)
        assert [(c.num, c.title, c.text) for c in by_number] == [(target.num, target.title, target.text)]

    def test_selecting_two_chapters_keeps_reading_order(self, name, extracted):
        everything = extracted(name)
        wanted = [everything[-1].title, everything[0].title]        # asked for in reverse
        chosen, _ = extract(os.path.join(_FIXTURES, name), selections=wanted, log_fn=_quiet)
        assert [c.title for c in chosen] == [everything[0].title, everything[-1].title]

    def test_flagged_matter_is_listed_only_on_request(self, name, extracted):
        path = os.path.join(_FIXTURES, name)
        listed = scan(path, include_matter=True).chapters
        flagged = [c for c in listed if c.probably_matter]
        assert [c.title for c in listed if not c.probably_matter] == EXPECTED[name]["chapters"]
        numbers = [c.num for c in listed]
        assert len(set(numbers)) == len(numbers), "every listed entry needs its own number"
        assert all(c.num > len(EXPECTED[name]["chapters"]) for c in flagged)
        for title in EXPECTED[name].get("matter", []):
            assert title in [c.title for c in flagged]
        if not flagged:
            return
        # A flagged entry is extracted when — and only when — it is asked for.
        entry = max(flagged, key=lambda c: c.word_count)
        if entry.word_count < 10:
            return
        chosen, _ = extract(path, selections=[entry.num], log_fn=_quiet)
        assert [(c.num, c.title, c.probably_matter) for c in chosen] == [(entry.num, entry.title, True)]

    def test_python_and_rust_normalisers_agree(self, name, extracted, monkeypatch):
        pytest.importorskip("audiobook_rust")
        try:
            text_processing._python_normalize_text("probe")
            __import__("nltk").sent_tokenize("One. Two.")
        except Exception:
            pytest.skip("NLTK punkt data is not available for the pure-Python splitter")
        with_rust = extracted(name)
        monkeypatch.setitem(sys.modules, "audiobook_rust", None)      # `import audiobook_rust` now fails
        monkeypatch.setattr(text_processing, "_RUST_AVAILABLE", False)
        pure_python, _ = extract(os.path.join(_FIXTURES, name), log_fn=_quiet)
        assert [c.title for c in pure_python] == [c.title for c in with_rust]
        assert [c.text for c in pure_python] == [c.text for c in with_rust]


class TestMetadata:

    @pytest.mark.parametrize("name", [
        pytest.param(n, marks=_NEEDS_MOBI) if n.endswith(".mobi") else n
        for n, entry in EXPECTED.items() if entry.get("title")
    ])
    def test_title_and_author(self, name):
        result = scan(os.path.join(_FIXTURES, name))
        assert result.title == EXPECTED[name]["title"]
        assert result.author == EXPECTED[name]["author"]

    @pytest.mark.parametrize("name", [
        pytest.param(n, marks=_NEEDS_MOBI) if n.endswith(".mobi") else n
        for n, entry in EXPECTED.items() if entry.get("has_cover")
    ])
    def test_cover_is_returned_by_scan_and_extract(self, name):
        path = os.path.join(_FIXTURES, name)
        cover = scan(path).cover_data
        assert isinstance(cover, bytes) and cover.startswith(b"\x89PNG")
        _, extracted_cover = extract(path, selections=[1], log_fn=_quiet)
        assert extracted_cover == cover

    def test_page_ranges_are_only_offered_for_pdf(self):
        for name, entry in EXPECTED.items():
            if name.endswith(".mobi"):
                continue
            result = scan(os.path.join(_FIXTURES, name))
            assert result.supports_page_ranges is (entry["format"] == "pdf"), name
            assert (result.page_count > 0) is (entry["format"] == "pdf"), name


class TestPdfFixture:

    PATH = os.path.join(_FIXTURES, "dummy_book.pdf")

    def test_no_page_numbers_or_running_headers(self, extracted):
        for chapter in extracted("dummy_book.pdf"):
            for line in chapter.text.split("\n"):
                assert not re.fullmatch(r"\s*\d{1,4}\s*", line), f"page number narrated in {chapter.title!r}"
            assert "The Lighthouse at Saltmarsh Point" not in chapter.text
            for sentence in chapter.sentences:
                assert not re.fullmatch(r"\W*\d{1,4}\W*", sentence)

    def test_line_end_hyphens_are_removed(self, extracted):
        import fitz
        doc = fitz.open(self.PATH)
        try:
            raw = "\n".join(page.get_text("text") for page in doc)
        finally:
            doc.close()
        broken = re.findall(r"([a-z]{3,})-\n([a-z]{3,})", raw)
        assert broken, "the fixture should contain words hyphenated at a line end"
        text = _flat(" ".join(c.text for c in extracted("dummy_book.pdf")))
        for head, tail in broken:
            assert f"{head}{tail}" in text, f"{head}-{tail} was not re-joined"
            assert f"{head}- {tail}" not in text and f"{head}-{tail}" not in text

    def test_page_ranges_give_one_chapter_per_range_without_headers(self):
        chapters, _ = extract(self.PATH, page_ranges=[(4, 4), (5, 6)], log_fn=_quiet)
        assert [c.title for c in chapters] == ["Chapter 1 (pp. 4–4)", "Chapter 2 (pp. 5–6)"]
        assert "The tide went out at dawn" in _flat(chapters[0].text)
        for chapter in chapters:
            assert "The Lighthouse at Saltmarsh Point" not in chapter.text
            assert not [ln for ln in chapter.text.split("\n") if re.fullmatch(r"\s*\d{1,4}\s*", ln)]

    def test_pdf_without_outline_finds_the_same_chapters(self, extracted):
        with_outline = extracted("dummy_book.pdf")
        without = extracted("dummy_book_no_outline.pdf")
        assert [c.title for c in without] == [c.title for c in with_outline]
        assert [_flat(c.text) for c in without] == [_flat(c.text) for c in with_outline]


@_NEEDS_MOBI
class TestMobiFixture:

    PATH = os.path.join(_FIXTURES, "dummy_book.mobi")

    def test_file_is_a_palm_database_with_mobi_header(self):
        with open(self.PATH, "rb") as handle:
            data = handle.read()
        assert data[60:68] == b"BOOKMOBI"
        record_count = int.from_bytes(data[76:78], "big")
        first_record = int.from_bytes(data[78:82], "big")
        assert record_count >= 5
        assert data[first_record + 16:first_record + 20] == b"MOBI"
        assert b"EXTH" in data[first_record:first_record + 400]

    def test_text_matches_the_epub_edition(self, extracted):
        mobi = extracted("dummy_book.mobi")
        epub = extracted("dummy_book.epub")
        assert [c.title for c in mobi] == [c.title for c in epub]
        assert [_flat(c.text) for c in mobi] == [_flat(c.text) for c in epub]

    def test_temporary_unpack_directory_is_removed(self):
        temp_dir = os.path.join(_ROOT, "temp")
        before = set(os.listdir(temp_dir)) if os.path.isdir(temp_dir) else set()
        scan(self.PATH)
        extract(self.PATH, selections=[1], log_fn=_quiet)
        after = set(os.listdir(temp_dir)) if os.path.isdir(temp_dir) else set()
        assert not {name for name in after - before if name.startswith("mobi_")}


class TestFixturesAreReproducible:

    def test_generator_rebuilds_identical_text_fixtures(self, tmp_path):
        from tests.fixture_generation import fixture_builders as build
        from tests.fixture_generation.generate_test_documents import generate_epub_fixture

        rebuilt = tmp_path / "dummy_book.txt"
        build.build_txt(str(rebuilt))
        with open(os.path.join(_FIXTURES, "dummy_book.txt"), "rb") as handle:
            assert rebuilt.read_bytes() == handle.read()

        rebuilt_mobi = tmp_path / "dummy_book.mobi"
        build.build_mobi(str(rebuilt_mobi))
        with open(os.path.join(_FIXTURES, "dummy_book.mobi"), "rb") as handle:
            assert rebuilt_mobi.read_bytes() == handle.read()

        first, second = tmp_path / "a.epub", tmp_path / "b.epub"
        generate_epub_fixture(str(first))
        generate_epub_fixture(str(second))
        assert first.read_bytes() == second.read_bytes()
