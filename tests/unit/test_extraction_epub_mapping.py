"""
test_extraction_epub_mapping.py
===============================
EPUB chapter mapping (TOC -> spine -> chapters), selection push-down, parse
counts, and image OCR handling. Small EPUBs are built on the fly with the
fixture builders; EasyOCR and Docling are replaced by in-test fakes, so no
optional dependency is needed.
"""

from __future__ import annotations

import os
import sys
import types

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from ebooklib import epub  # noqa: E402

from audiobook_factory import extractor_engine as engine  # noqa: E402
from audiobook_factory import text_extractor  # noqa: E402
from audiobook_factory.extractor_engine import (  # noqa: E402
    DocumentIngestor, HtmlDoc, MLClassifier, TextNormalizer, TocEntry, normalize_href, select_sections,
)
from audiobook_factory.text_extractor import extract, scan  # noqa: E402
from tests.fixture_generation import fixture_builders as build  # noqa: E402

_FIXTURES = os.path.join(_ROOT, "tests", "fixtures", "source_documents")
_WORDS = "The harbour was quiet and the tide was low that morning. "


def _body(heading: str, marker: str, repeat: int = 12, tag: str = "h1", anchor: str = "") -> str:
    ident = f' id="{anchor}"' if anchor else ""
    head = f"<{tag}{ident}>{heading}</{tag}>\n" if heading else ""
    return f"{head}<p>{marker} {_WORDS * repeat}</p>"


def _quiet(_message: str) -> None:
    return None


def _titles(path: str) -> list[str]:
    return [c.title for c in extract(path, log_fn=_quiet)[0]]


class TestHrefMatching:

    @pytest.mark.parametrize("href, expected", [
        ("Text/chapter%20one.xhtml#c1", ("Text/chapter one.xhtml", "c1")),
        ("./Text/../Text/caf%C3%A9.xhtml", ("Text/café.xhtml", "")),
        ("ch1.xhtml#sec%201", ("ch1.xhtml", "sec 1")),
        ("", ("", "")),
    ])
    def test_normalize_href(self, href, expected):
        assert normalize_href(href) == expected

    def test_ncx_hrefs_resolve_against_the_ncx_directory(self):
        assert normalize_href("../Text/ch1.xhtml", "toc") == ("Text/ch1.xhtml", "")

    def test_classifier_uses_exact_paths_not_substrings(self):
        label, score = MLClassifier().classify_item(
            item_name="1.xhtml", item_title="", word_count=500, position_idx=0,
            chapter_hrefs={"11.xhtml"}, skip_hrefs={"21.xhtml"}, doc_texts=[], avg_font=12.0,
        )
        assert (label, score) != ("chapter", 1.0)
        label, score = MLClassifier().classify_item(
            item_name="Text/chapter one.xhtml", item_title="", word_count=5, position_idx=0,
            chapter_hrefs={"Text/chapter%20one.xhtml"}, skip_hrefs=set(), doc_texts=[], avg_font=12.0,
        )
        assert (label, score) == ("chapter", 1.0)

    def test_walk_toc_survives_ebooklibs_bare_link_for_an_empty_navmap(self):
        chapter_hrefs, skip_hrefs, entries = DocumentIngestor()._walk_epub_toc(epub.Link("", "", ""))
        assert (chapter_hrefs, skip_hrefs, entries) == (set(), set(), [])

    def test_walk_toc_keeps_anchor_depth_and_parent_flag(self):
        toc = [
            epub.Link("a.xhtml#top", "One", "l1"),
            (epub.Section("Part Two", href="b.xhtml"), [epub.Link("b.xhtml#c2", " Two\n  again ", "l2")]),
            epub.Link("copy%20right.xhtml", "Copyright", "l3"),
        ]
        chapters, skips, entries = DocumentIngestor()._walk_epub_toc(toc)
        assert chapters == {"a.xhtml", "b.xhtml"} and skips == {"copy right.xhtml"}
        assert [(e.title, e.href, e.anchor, e.depth, e.is_parent, e.classification) for e in entries] == [
            ("One", "a.xhtml", "top", 0, False, "chapter"),
            ("Part Two", "b.xhtml", "", 0, True, "chapter"),
            ("Two again", "b.xhtml", "c2", 1, False, "chapter"),
            ("Copyright", "copy right.xhtml", "", 0, False, "skip"),
        ]


class TestChapterPlanning:

    def _plan(self, docs, toc):
        sections, skipped = DocumentIngestor().plan_html_sections(docs, toc)
        return sections, skipped

    def test_anchors_that_cannot_be_found_give_one_chapter_with_the_first_title(self, tmp_path):
        path = str(tmp_path / "book.epub")
        body = _body("One", "ALPHA") + _body("Two", "BETA") + _body("Three", "GAMMA")   # no ids at all
        toc = [("First", "book.xhtml#c1"), ("Second", "book.xhtml#c2"), ("Third", "book.xhtml#c3")]
        build.build_epub(path, files=[("book.xhtml", "Book", body)], spine=["book.xhtml"], toc=toc)
        assert [c.title for c in scan(path).chapters] == ["First"]
        chapters, _ = extract(path, log_fn=_quiet)
        assert [c.title for c in chapters] == ["First"]
        assert all(word in chapters[0].text for word in ("ALPHA", "BETA", "GAMMA"))

    def test_only_located_anchors_split_the_file(self, tmp_path):
        path = str(tmp_path / "book.epub")
        body = _body("One", "ALPHA", anchor="c1") + _body("Two", "BETA") + _body("Three", "GAMMA", anchor="c3")
        toc = [("First", "book.xhtml#c1"), ("Second", "book.xhtml#missing"), ("Third", "book.xhtml#c3")]
        build.build_epub(path, files=[("book.xhtml", "Book", body)], spine=["book.xhtml"], toc=toc)
        chapters, _ = extract(path, log_fn=_quiet)
        assert [c.title for c in chapters] == ["First", "Third"]
        assert "BETA" in chapters[0].text and "GAMMA" in chapters[1].text and "GAMMA" not in chapters[0].text

    def test_anchor_inside_a_heading_keeps_the_heading_with_its_chapter(self):
        markup = (
            '<div class="ch"><h2><a id="c1"></a>One</h2><p>ALPHA ' + _WORDS * 5 + "</p></div>"
            '<div class="ch"><h2><a id="c2"></a>Two</h2><p>BETA ' + _WORDS * 5 + "</p></div>"
        )
        toc = [TocEntry("One", "b.xhtml", "chapter", anchor="c1"), TocEntry("Two", "b.xhtml", "chapter", anchor="c2")]
        sections, _ = self._plan([HtmlDoc("b.xhtml", markup)], toc)
        assert [s.title for s in sections] == ["One", "Two"]
        assert sections[1].parts[0][1].startswith('<div class="ch"><h2><a id="c2">')
        assert "BETA" not in sections[0].parts[0][1]

    def test_section_and_first_child_with_one_target_use_the_child_title(self):
        docs = [HtmlDoc("p1.xhtml", _body("Chapter 1", "ALPHA")), HtmlDoc("c2.xhtml", _body("Chapter 2", "BETA"))]
        toc = [
            TocEntry("Part One", "p1.xhtml", "chapter", is_parent=True),
            TocEntry("Chapter 1", "p1.xhtml", "chapter", depth=1),
            TocEntry("Chapter 2", "c2.xhtml", "chapter", depth=1),
        ]
        sections, _ = self._plan(docs, toc)
        assert [s.title for s in sections] == ["Chapter 1", "Chapter 2"]

    def test_deep_subsections_do_not_fragment_a_listed_chapter(self):
        markup = _body("Chapter 1", "ALPHA") + '<h3 id="s1">1.1</h3><p>BETA ' + _WORDS * 5 + "</p>"
        toc = [
            TocEntry("Part", "part.xhtml", "chapter", is_parent=True),
            TocEntry("Chapter 1", "c1.xhtml", "chapter", depth=1, is_parent=True),
            TocEntry("1.1", "c1.xhtml", "chapter", anchor="s1", depth=2),
        ]
        docs = [HtmlDoc("part.xhtml", "<h1>Part</h1>"), HtmlDoc("c1.xhtml", markup)]
        sections, skipped = self._plan(docs, toc)
        assert [s.title for s in sections] == ["Chapter 1"]
        assert "BETA" in sections[0].parts[0][1]
        assert any("too short" in s.reason.lower() for s in skipped)       # the bare "Part" page

    def test_heading_only_fragment_joins_the_chapter_that_follows_in_the_same_file(self):
        markup = '<h1 id="p1">Part One</h1>' + _body("Chapter 1", "ALPHA", tag="h2", anchor="c1")
        toc = [TocEntry("Part One", "b.xhtml", "chapter", anchor="p1"),
               TocEntry("Chapter 1", "b.xhtml", "chapter", anchor="c1")]
        sections, _ = self._plan([HtmlDoc("b.xhtml", markup)], toc)
        assert [s.title for s in sections] == ["Chapter 1"]
        assert "Part One" in "".join(fragment for _doc, fragment in sections[0].parts)

    def test_toc_entry_to_a_missing_file_is_reported_not_fatal(self):
        docs = [HtmlDoc("c1.xhtml", _body("Chapter 1", "ALPHA"))]
        toc = [TocEntry("Chapter 1", "c1.xhtml", "chapter"), TocEntry("Ghost", "gone.xhtml", "chapter")]
        sections, skipped = self._plan(docs, toc)
        assert [s.title for s in sections] == ["Chapter 1"]
        assert any(s.title == "Ghost" for s in skipped)

    def test_unlisted_file_with_a_chapter_heading_starts_its_own_chapter(self):
        docs = [HtmlDoc("c1.xhtml", _body("Chapter 1", "ALPHA")), HtmlDoc("c1b.xhtml", _body("", "BETA", 3)),
                HtmlDoc("c2.xhtml", _body("Chapter 2", "GAMMA", 2))]
        sections, _ = self._plan(docs, [TocEntry("Chapter 1", "c1.xhtml", "chapter")])
        assert [s.title for s in sections] == ["Chapter 1", "Chapter 2"]
        assert len(sections[0].parts) == 2

    def test_continuation_file_that_mentions_a_first_edition_is_not_mistaken_for_matter(self):
        tail = "<p>She found the first edition of the paper on the table. " + _WORDS * 4 + "</p>"
        docs = [HtmlDoc("c1.xhtml", _body("Chapter 1", "ALPHA")), HtmlDoc("c1b.xhtml", tail)]
        sections, _ = self._plan(docs, [TocEntry("Chapter 1", "c1.xhtml", "chapter")])
        assert len(sections) == 1 and len(sections[0].parts) == 2

    def test_short_unlisted_front_file_is_flagged_not_narrated(self):
        docs = [HtmlDoc("ded.xhtml", "<p>For my mother.</p>"), HtmlDoc("c1.xhtml", _body("Chapter 1", "ALPHA"))]
        sections, _ = self._plan(docs, [TocEntry("Chapter 1", "c1.xhtml", "chapter")])
        assert [(s.title, s.probably_matter, s.num) for s in sections] == [
            ("For my mother.", True, 2), ("Chapter 1", False, 1)] or \
            [(s.probably_matter, s.num) for s in sections] == [(True, 2), (False, 1)]

    def test_epub_type_marks_matter_even_without_a_telling_title(self):
        docs = [
            HtmlDoc("c1.xhtml", _body("Chapter 1", "ALPHA")),
            HtmlDoc("x.xhtml", '<section epub:type="backmatter colophon"><p>Set in ten point type. '
                    + _WORDS * 10 + "</p></section>"),
        ]
        sections, _ = self._plan(docs, [TocEntry("Chapter 1", "c1.xhtml", "chapter")])
        assert [s.probably_matter for s in sections] == [False, True]
        assert len(sections[0].parts) == 1

    def test_no_ncx_and_no_nav_still_yields_chapters(self, tmp_path):
        path = str(tmp_path / "bare.epub")
        files = [("a.xhtml", "A", _body("The Arrival", "ALPHA")), ("b.xhtml", "B", _body("The Storm", "BETA"))]
        build.build_epub(path, files=files, spine=["a.xhtml", "b.xhtml"], toc=None, nav=False, ncx=False)
        assert [c.title for c in scan(path).chapters] == ["The Arrival", "The Storm"]
        assert _titles(path) == ["The Arrival", "The Storm"]

    def test_headingless_files_without_toc_are_numbered(self, tmp_path):
        path = str(tmp_path / "plain.epub")
        files = [("a.xhtml", "A", _body("", "ALPHA")), ("b.xhtml", "B", _body("Night", "BETA")),
                 ("c.xhtml", "C", _body("", "GAMMA", 4))]
        build.build_epub(path, files=files, spine=["a.xhtml", "b.xhtml", "c.xhtml"], toc=None, nav=False, ncx=True)
        chapters, _ = extract(path, log_fn=_quiet)
        assert [c.title for c in chapters] == ["Chapter 1", "Night"]
        assert "GAMMA" in chapters[1].text                      # headingless file continues "Night"

    def test_single_file_without_toc_is_split_at_its_headings(self, tmp_path):
        path = str(tmp_path / "onefile.epub")
        body = "<h1>My Book</h1><p>by Somebody</p>" + "".join(
            _body(title, marker, 6, tag="h2") for title, marker in
            (("Dawn", "ALPHA"), ("Noon", "BETA"), ("About the Author", "GAMMA"))
        )
        build.build_epub(path, files=[("book.xhtml", "Book", body)], spine=["book.xhtml"], toc=None,
                         nav=False, ncx=True)
        assert [c.title for c in scan(path).chapters] == ["Dawn", "Noon"]
        chapters, _ = extract(path, log_fn=_quiet)
        assert [c.title for c in chapters] == ["Dawn", "Noon"]
        assert "BETA" in chapters[1].text and "GAMMA" not in chapters[1].text and "Somebody" not in chapters[0].text

    def test_non_linear_spine_items_are_not_appended_to_a_chapter(self, tmp_path):
        path = str(tmp_path / "nonlinear.epub")
        files = [("c1.xhtml", "One", _body("Chapter 1", "ALPHA")), ("pop.xhtml", "", _body("", "POPUPNOTE", 3)),
                 ("c2.xhtml", "Two", _body("Chapter 2", "BETA"))]
        build.build_epub(path, files=files, spine=["c1.xhtml", "pop.xhtml", "c2.xhtml"],
                         toc=[("Chapter 1", "c1.xhtml"), ("Chapter 2", "c2.xhtml")], nonlinear=("pop.xhtml",))
        text = " ".join(c.text for c in extract(path, log_fn=_quiet)[0])
        assert "ALPHA" in text and "BETA" in text and "POPUPNOTE" not in text

    def test_footnote_markers_and_bodies_are_removed_but_exponents_stay(self):
        markup = (
            '<p>Water boils<sup><a href="#n1">1</a></sup> at 100 degrees<a epub:type="noteref" href="#n2">[2]</a>, '
            "and E = mc<sup>2</sup> is famous<sup>*</sup>.</p>"
            '<aside epub:type="footnote" id="n1"><p>1. At sea level.</p></aside>'
            '<div epub:type="endnotes"><p>2. Celsius.</p></div><script>var x = "SCRIPTED";</script>'
        )
        text = DocumentIngestor()._bs_fallback(markup)
        assert text == "Water boils at 100 degrees, and E = mc2 is famous."

    def test_table_rows_are_read_as_lines_and_layout_tables_are_left_alone(self):
        data = "<table><tr><th>Day</th><th>High</th></tr><tr><td>Monday</td><td>6:10</td></tr></table>"
        assert DocumentIngestor()._bs_fallback(data) == "Day, High\n\nMonday, 6:10"
        layout = "<table><tr><td><p>First paragraph.</p><p>Second paragraph.</p></td><td><p>Side.</p></td></tr></table>"
        assert DocumentIngestor()._bs_fallback(layout) == "First paragraph.\n\nSecond paragraph.\n\nSide."


class TestSkipList:

    @pytest.mark.parametrize("title", [
        "Dedication", "Acknowledgments", "Acknowledgements", "Acknowledgment", "Also by Mara Ellison",
        "Also Available", "Praise for The Lighthouse", "Advance Praise for the Author", "Notes", "Endnotes",
        "End Notes", "Footnotes", "Half Title", "Half-Title Page", "Halftitle", "Other Books by the Author",
        "By the Same Author", "Books by Mara Ellison", "Preview", "Preview of The Next Book",
        "An Excerpt from Book Two", "Sneak Peek", "List of Illustrations", "Frontispiece", "References",
        "Further Reading", "Works Cited", "Permissions", "Disclaimer", "Reading Group Guide",
        "Discussion Questions", "Table of Contents", "Copyright", "About the Author", "Title Page", "Index",
    ])
    def test_front_and_back_matter_titles_are_skipped(self, title):
        assert engine._SKIP_TOC_TITLE.match(title) is not None

    @pytest.mark.parametrize("title", [
        "Prologue", "Epilogue", "Introduction", "Preface", "Foreword", "Afterword", "Interlude",
        "About the Lighthouse", "Notes from Underground", "Dedication to Duty", "Praise", "Also",
        "Previews of Coming Attractions", "The Half-Blood Prince", "Reference Point", "Chapter 10: The Tenth Bell",
        "Part One", "A Note on the Text", "Epigraph", "Author's Note", "Appendix A", "Permission to Land",
        "Books and Their Makers", "End of the Road", "Cover of Darkness", "Maple Street",
    ])
    def test_narrable_titles_are_kept(self, title):
        assert engine._SKIP_TOC_TITLE.match(title) is None

    def test_select_sections_skips_matter_unless_asked_for(self):
        Section = types.SimpleNamespace
        sections = [
            Section(title="Dedication", num=3, probably_matter=True),
            Section(title="Chapter 1", num=1, probably_matter=False),
            Section(title="Chapter 2", num=2, probably_matter=False),
        ]
        assert [s.num for s in select_sections(sections, None)] == [1, 2]
        assert [s.num for s in select_sections(sections, [])] == [1, 2]
        assert [s.num for s in select_sections(sections, ["  chapter   2 "])] == [2]
        assert [s.num for s in select_sections(sections, ["Dedication", "Chapter 1"])] == [3, 1]
        assert [s.num for s in select_sections(sections, [3])] == [3]
        assert select_sections(sections, ["Chapter 10"]) == []


class TestPerformance:

    PATH = os.path.join(_FIXTURES, "dummy_book.epub")

    def test_unselected_chapters_are_never_converted(self, monkeypatch):
        converted = []
        original = DocumentIngestor.convert_section

        def _spy(self, section, *args, **kwargs):
            converted.append(section.title)
            return original(self, section, *args, **kwargs)

        monkeypatch.setattr(DocumentIngestor, "convert_section", _spy)
        chapters, _ = extract(self.PATH, selections=["Chapter 2: Plan B"], log_fn=_quiet)
        assert [c.title for c in chapters] == ["Chapter 2: Plan B"]
        assert converted == ["Chapter 2: Plan B"]

    def test_epub_is_safety_checked_and_opened_exactly_once(self, monkeypatch):
        calls = {"safe": 0, "read": 0}
        real_safe, real_read = text_extractor._assert_zip_safe, epub.read_epub

        def _safe(path):
            calls["safe"] += 1
            return real_safe(path)

        def _read(*args, **kwargs):
            calls["read"] += 1
            return real_read(*args, **kwargs)

        monkeypatch.setattr(text_extractor, "_assert_zip_safe", _safe)
        monkeypatch.setattr(epub, "read_epub", _read)
        chapters, cover = extract(self.PATH, log_fn=_quiet)
        assert len(chapters) == 6 and cover
        assert calls == {"safe": 1, "read": 1}
        calls.update(safe=0, read=0)
        scan(self.PATH)
        assert calls == {"safe": 1, "read": 1}

    def test_each_converted_file_is_parsed_by_beautifulsoup_once(self, monkeypatch):
        parsed = []
        real = engine.make_soup
        monkeypatch.setattr(engine, "make_soup", lambda markup: parsed.append(len(markup)) or real(markup))
        extract(os.path.join(_FIXTURES, "epub_manifest_order_differs.epub"),
                selections=["Chapter 1: The Salt Road"], log_fn=_quiet)
        assert len(parsed) == 2             # ch1.xhtml + its unlisted continuation, once each

    def test_lxml_is_the_html_parser(self):
        pytest.importorskip("lxml")
        soup = engine.make_soup("<p>one<p>two")
        assert soup.builder.NAME == "lxml"

    def test_single_file_with_many_anchors_is_indexed_once(self, tmp_path, monkeypatch):
        count = 150
        body = "".join(_body(f"Chapter {n}", f"MARK{n}", 2, tag="h2", anchor=f"c{n}") for n in range(1, count + 1))
        toc = [(f"Chapter {n}", f"book.xhtml#c{n}") for n in range(1, count + 1)]
        path = str(tmp_path / "many.epub")
        build.build_epub(path, files=[("book.xhtml", "Book", body)], spine=["book.xhtml"], toc=toc)

        indexed = []
        real = DocumentIngestor._anchor_positions
        monkeypatch.setattr(DocumentIngestor, "_anchor_positions",
                            staticmethod(lambda markup: indexed.append(1) or real(markup)))
        result = scan(path)
        assert [c.title for c in result.chapters] == [f"Chapter {n}" for n in range(1, count + 1)]
        assert len(indexed) == 1
        only, _ = extract(path, selections=["Chapter 77"], log_fn=_quiet)
        assert len(only) == 1 and "MARK77" in only[0].text and "MARK78" not in only[0].text

    def test_docling_converter_is_not_built_until_it_is_needed(self, monkeypatch):
        built = []
        monkeypatch.setattr(engine, "DOCLING_AVAILABLE", True)
        monkeypatch.setattr(engine, "DocumentConverter", lambda *a, **k: built.append(1) or object(), raising=False)
        ingestor = DocumentIngestor()
        assert built == []
        assert ingestor._converter is ingestor._converter
        assert built == [1]

    def test_pdf_path_does_not_build_an_ingestor(self, monkeypatch):
        def _boom(self):
            raise AssertionError("DocumentIngestor must not be constructed for PDF/DOCX/ODT/TXT")
        monkeypatch.setattr(DocumentIngestor, "__init__", _boom)
        for name in ("dummy_book.pdf", "dummy_book.docx", "dummy_book.odt", "dummy_book.txt"):
            chapters, _ = extract(os.path.join(_FIXTURES, name), selections=[1], log_fn=_quiet)
            assert len(chapters) == 1


# ══════════════════════════════════════════════════════════════════════════════
# OCR and Docling glue, with fakes
# ══════════════════════════════════════════════════════════════════════════════

class _FakeReader:
    instances: list["_FakeReader"] = []

    def __init__(self, languages, gpu=True):
        if "xx" in languages:
            raise ValueError("unsupported language")
        self.languages, self.gpu, self.reads = list(languages), gpu, 0
        _FakeReader.instances.append(self)

    def readtext(self, image, **kwargs):
        self.reads += 1
        if kwargs.get("detail") == 0:
            return ["SCANNED PAGE TEXT that was only a picture."]
        return [((0, 0), "PAINTED SIGN", 0.99)]


@pytest.fixture
def fake_easyocr(monkeypatch):
    pytest.importorskip("PIL")
    pytest.importorskip("numpy")
    _FakeReader.instances = []
    monkeypatch.setitem(sys.modules, "easyocr", types.SimpleNamespace(Reader=_FakeReader))
    monkeypatch.delenv("AUDIOBOOK_OCR_GPU", raising=False)
    DocumentIngestor.release_ocr_reader()
    yield _FakeReader
    DocumentIngestor.release_ocr_reader()


@pytest.fixture
def picture_book(tmp_path):
    """An EPUB whose chapter has an image between two paragraphs."""
    path = str(tmp_path / "picture.epub")
    body = ("<h1>The Sign</h1><p>BEFORE " + _WORDS * 3 + '</p><p><img src="images/cover.png" alt="sign"/></p>'
            "<p>AFTER " + _WORDS * 3 + "</p>")
    build.build_epub(path, files=[("c1.xhtml", "The Sign", body)], spine=["c1.xhtml"],
                     toc=[("The Sign", "c1.xhtml")], cover_png=build.make_cover_png())
    return path


class TestImageOcr:

    def test_ocr_text_stays_where_the_image_was(self, fake_easyocr, picture_book):
        chapters, _ = extract(picture_book, enable_ocr=True, log_fn=_quiet)
        text = chapters[0].text
        assert text.index("BEFORE") < text.index("PAINTED SIGN") < text.index("AFTER")
        assert "OCR_IMG_TEXT" not in text and "OCRIMGTOKEN" not in text

    def test_reader_runs_on_cpu_follows_book_language_and_is_released(self, fake_easyocr, picture_book):
        extract(picture_book, enable_ocr=True, log_fn=_quiet)
        assert [(r.languages, r.gpu) for r in fake_easyocr.instances] == [(["en"], False)]
        assert DocumentIngestor._easyocr_reader is None

    def test_gpu_only_when_the_environment_asks_for_it(self, fake_easyocr, picture_book, monkeypatch):
        monkeypatch.setenv("AUDIOBOOK_OCR_GPU", "1")
        extract(picture_book, enable_ocr=True, language="fr-FR", log_fn=_quiet)
        assert [(r.languages, r.gpu) for r in fake_easyocr.instances] == [(["fr", "en"], True)]

    def test_unsupported_language_falls_back_to_english(self, fake_easyocr, picture_book):
        chapters, _ = extract(picture_book, enable_ocr=True, language="xx", log_fn=_quiet)
        assert "PAINTED SIGN" in chapters[0].text
        assert [r.languages for r in fake_easyocr.instances] == [["en"]]

    def test_no_ocr_without_the_flag(self, fake_easyocr, picture_book):
        chapters, _ = extract(picture_book, log_fn=_quiet)
        assert "PAINTED SIGN" not in chapters[0].text and fake_easyocr.instances == []

    @pytest.mark.parametrize("tag, expected", [
        ("en", ["en"]), ("en-GB", ["en"]), (None, ["en"]), ("fr", ["fr", "en"]), ("zh", ["ch_sim", "en"]),
        ("zh-TW", ["ch_tra", "en"]), ("zh_Hant", ["ch_tra", "en"]), ("ja", ["ja", "en"]), ("RU", ["ru", "en"]),
    ])
    def test_language_mapping(self, tag, expected):
        assert DocumentIngestor.ocr_languages(tag) == expected

    def test_images_are_ocrd_once_when_docling_fails(self, fake_easyocr, picture_book, monkeypatch):
        class _Broken:
            def convert(self, _path):
                raise RuntimeError("docling exploded")

        monkeypatch.setattr(engine, "DOCLING_AVAILABLE", True)
        monkeypatch.setattr(engine, "DocumentConverter", _Broken, raising=False)
        chapters, _ = extract(picture_book, enable_ocr=True, log_fn=_quiet)
        assert chapters[0].text.count("PAINTED SIGN") == 1
        assert sum(r.reads for r in fake_easyocr.instances) == 1


class _FakeDoclingDoc:
    """Mimics docling-core: escapes HTML and marks images unless told not to."""

    def __init__(self, html_text: str):
        self._html = html_text
        self.texts = []
        self.kwargs: dict = {}

    def export_to_markdown(self, escape_html: bool = True, image_placeholder: str = "<!-- image -->"):
        self.kwargs = {"escape_html": escape_html, "image_placeholder": image_placeholder}
        import re
        token = re.search(r"OCRIMGTOKEN\d+X", self._html)
        amp = "&amp;" if escape_html else "&"
        lines = ["## The Sign", "", f"Smith {amp} Sons made the sign, see [the catalogue](http://example.com/c).",
                 "", image_placeholder, "", token.group(0) if token else "", "", "- first point", "- second point",
                 "", "| Day | High |", "|---|---|", "| Monday | 6:10 |", "", "AFTER the list."]
        return "\n".join(lines)

    def export_to_dict(self):
        return {"ok": True}


class TestDoclingGlue:

    def _install(self, monkeypatch, doc_class=_FakeDoclingDoc):
        made = []

        class _Converter:
            def convert(self, path):
                with open(path, encoding="utf-8") as handle:
                    made.append(doc_class(handle.read()))
                return types.SimpleNamespace(document=made[-1])

        monkeypatch.setattr(engine, "DOCLING_AVAILABLE", True)
        monkeypatch.setattr(engine, "DocumentConverter", _Converter, raising=False)
        return made

    def test_export_passes_supported_arguments_and_markdown_is_flattened(self, monkeypatch, fake_easyocr, picture_book):
        made = self._install(monkeypatch)
        chapters, _ = extract(picture_book, enable_ocr=True, log_fn=_quiet)
        assert made[0].kwargs == {"escape_html": False, "image_placeholder": ""}
        text = chapters[0].text
        assert "Smith & Sons made the sign, see the catalogue." in text
        # List items and table rows end like sentences, so they are not run together.
        assert "first point." in text and "second point." in text
        assert "Day, High.\n\nMonday, 6:10." in text
        assert text.index("catalogue") < text.index("PAINTED SIGN") < text.index("first point")
        for junk in ("|", "](", "- first", "<!--", "##", "OCRIMGTOKEN", "http"):
            assert junk not in text

    def test_export_falls_back_for_older_docling_core(self):
        class _Old:
            def export_to_markdown(self):
                return "old style"

        class _Picky:
            def export_to_markdown(self, **kwargs):
                if kwargs:
                    raise TypeError("unexpected keyword")
                return "picky"

        class _New:
            def export_to_markdown(self, escape_html=True, escape_underscores=True, image_placeholder="<!-- image -->"):
                return f"{escape_html}|{escape_underscores}|{image_placeholder!r}"

        assert DocumentIngestor._export_markdown(_Old()) == "old style"
        assert DocumentIngestor._export_markdown(_Picky()) == "picky"
        # Underscores stay escaped: the normaliser turns "\_" into a space.
        assert DocumentIngestor._export_markdown(_New()) == "False|True|''"

    def test_ocr_text_is_appended_when_docling_drops_the_placeholder(self):
        placed = DocumentIngestor._place_ocr_text("Only prose.", {"OCRIMGTOKEN00000X": "OCR_IMG_TEXT: LOST"})
        assert placed.startswith("Only prose.") and placed.rstrip().endswith("OCR_IMG_TEXT: LOST")

    def test_preprocess_html_keeps_its_contract(self):
        html, texts = DocumentIngestor._preprocess_html('<p><span class="dropcap">T</span>he tide.</p>')
        assert "The tide." in html and texts == []

    def test_markdown_prepass(self):
        normalizer = TextNormalizer()
        raw = ("- one\n* two\n+ three\n\n> quoted line\n\n| A | B |\n| :-- | --: |\n| 1 | 2 |\n\n"
               "See [the map](maps/1.png) and ![pic](x.png)[[3]](#fn3).\n\n* * *\n\n```\ncode\n```\n\n1\\. Escaped")
        flattened = normalizer.strip_markdown_structure(raw)
        assert [line for line in flattened.split("\n") if line] == [
            "one.", "two.", "three.", "quoted line", "A, B.", "1, 2.", "See the map and .", "* * *", "code",
            "1. Escaped",
        ]
        # every list item and table row is a paragraph of its own
        assert "one.\n\ntwo.\n\nthree." in flattened and "A, B.\n\n1, 2." in flattened
