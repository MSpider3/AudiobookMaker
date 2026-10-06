"""
generate_test_documents.py
===========================
Generates the deterministic synthetic book fixtures used by the extraction
tests and by the Kaggle test notebook:

- ``dummy_book.{epub,pdf,docx,odt,txt,mobi}`` — the same six-chapter book in
  every supported format, with front and back matter that must not be narrated;
- ``dummy_book_no_outline.pdf`` — the PDF without bookmarks (chapters have to
  be found from the heading type size);
- five edge-case EPUBs, one per chapter-mapping bug class;
- ``expected_chapters.json`` and ``README.md`` describing what extraction must
  return for each of them.

The book text lives in ``book_content.py`` (original prose, no third-party
text) and the file writers in ``fixture_builders.py``.

Usage::

    python tests/fixture_generation/generate_test_documents.py
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tests.fixture_generation import book_content as book  # noqa: E402
from tests.fixture_generation import fixture_builders as build  # noqa: E402
from tests.fixture_generation.book_content import CHAPTERS_DATA, TEST_BOOK_METADATA  # noqa: E402,F401

_ALL_KEYS: list[str] = [chapter["key"] for chapter in CHAPTERS_DATA]


def get_fixtures_dir() -> str:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, "fixtures")


# ══════════════════════════════════════════════════════════════════════════════
# Individual fixtures (names kept from the previous generator)
# ══════════════════════════════════════════════════════════════════════════════

def generate_txt_fixture(out_path: str) -> None:
    """Generate plaintext book fixture."""
    build.build_txt(out_path)


def generate_docx_fixture(out_path: str) -> None:
    """Generate DOCX book fixture using python-docx."""
    build.build_docx(out_path)


def generate_pdf_fixture(out_path: str, outline: bool = True) -> None:
    """Generate PDF book fixture using pymupdf."""
    build.build_pdf(out_path, outline=outline)


def generate_odt_fixture(out_path: str) -> None:
    """Generate ODT book fixture using odfpy."""
    build.build_odt(out_path)


def generate_mobi_fixture(out_path: str) -> None:
    """Generate a classic MOBI fixture (hand-built records, no external tools)."""
    build.build_mobi(out_path)


def _chapter_file(key: str) -> tuple[str, str, str]:
    chapter = book.chapter_by_key(key)
    return f"{key}.xhtml", chapter["title"], build.chapter_xhtml(chapter)


def generate_epub_fixture(out_path: str) -> None:
    """Generate the main EPUB 3 fixture: one file per chapter, nav + NCX, cover image."""
    files = [
        ("titlepage.xhtml", "Title Page", build.title_page_xhtml()),
        ("copyright.xhtml", "Copyright", build.copyright_xhtml(heading=True)),
        ("dedication.xhtml", "Dedication", f"<p>{book.DEDICATION}</p>"),
    ]
    files += [_chapter_file(key) for key in _ALL_KEYS]
    files += [
        ("about.xhtml", book.ABOUT_AUTHOR_TITLE, build.about_author_xhtml()),
        ("alsoby.xhtml", book.ALSO_BY_TITLE, build.also_by_xhtml()),
    ]
    toc = [("Title Page", "titlepage.xhtml"), ("Copyright", "copyright.xhtml"),
           ("Dedication", "dedication.xhtml")]
    toc += [(book.chapter_by_key(key)["title"], f"{key}.xhtml") for key in _ALL_KEYS]
    toc += [(book.ABOUT_AUTHOR_TITLE, "about.xhtml"), (book.ALSO_BY_TITLE, "alsoby.xhtml")]
    build.build_epub(out_path, files=files, spine=[name for name, _t, _b in files], toc=toc,
                     cover_png=build.make_cover_png())


# ── Edge-case EPUBs ───────────────────────────────────────────────────────────

def generate_epub_no_toc(out_path: str) -> list[str]:
    """No nav document and an NCX with an EMPTY navMap; one chapter is split over two files."""
    chapter_one = book.chapter_by_key("ch1")
    files = [
        ("title.xhtml", "Title", build.title_page_xhtml()),
        ("copyright.xhtml", "Copyright", build.copyright_xhtml()),
        _chapter_file("prologue"),
        ("ch1.xhtml", chapter_one["title"],
         build.chapter_xhtml({"title": chapter_one["title"], "blocks": chapter_one["blocks"][:5]})),
        ("ch1_part2.xhtml", "", build.render_blocks(chapter_one["blocks"][5:])),
        _chapter_file("lighthouse"),
        _chapter_file("epilogue"),
        ("about.xhtml", book.ABOUT_AUTHOR_TITLE, build.about_author_xhtml()),
    ]
    build.build_epub(out_path, files=files, spine=[name for name, _t, _b in files], toc=[],
                     nav=False, ncx=True, identifier="abm-edge-no-toc")
    return ["prologue", "ch1", "lighthouse", "epilogue"]


def generate_epub_multi_anchor(out_path: str) -> list[str]:
    """The whole book in ONE file; every TOC entry is ``book.xhtml#anchor``."""
    body = [build.title_page_xhtml(), build.copyright_xhtml()]
    body += [build.chapter_xhtml(book.chapter_by_key(key), heading="h2", anchor=key) for key in _ALL_KEYS]
    body += [build.about_author_xhtml(anchor="about").replace("h1", "h2"),
             build.also_by_xhtml(anchor="alsoby").replace("h1", "h2")]
    toc = [(book.chapter_by_key(key)["title"], f"book.xhtml#{key}") for key in _ALL_KEYS]
    toc += [(book.ABOUT_AUTHOR_TITLE, "book.xhtml#about"), (book.ALSO_BY_TITLE, "book.xhtml#alsoby")]
    build.build_epub(out_path, files=[("book.xhtml", TEST_BOOK_METADATA["title"], "\n".join(body))],
                     spine=["book.xhtml"], toc=toc, identifier="abm-edge-multi-anchor")
    return list(_ALL_KEYS)


def generate_epub_percent_encoded(out_path: str) -> list[str]:
    """EPUB 2 whose file names need percent-encoding (spaces, an accent) in OPF and NCX."""
    names = {"ch1": "Text/chapter one.xhtml", "ch2": "Text/chapter two.xhtml",
             "lighthouse": "Text/café lighthouse.xhtml", "epilogue": "Text/the epilogue.xhtml"}
    files = [("Text/title page.xhtml", "Title", build.title_page_xhtml())]
    files += [(names[key], book.chapter_by_key(key)["title"], build.chapter_xhtml(book.chapter_by_key(key)))
              for key in names]
    toc = [(book.chapter_by_key(key)["title"], names[key]) for key in names]
    build.build_epub(out_path, files=files, spine=[name for name, _t, _b in files], toc=toc,
                     nav=False, ncx=True, encode_hrefs=True, identifier="abm-edge-percent")
    return list(names)


def generate_epub_manifest_order(out_path: str) -> list[str]:
    """Manifest order is scrambled; only the spine gives the reading order.

    Chapter 1 continues in an unlisted second file that sits elsewhere in the
    manifest, so grouping by manifest order glues it to the wrong chapter.
    """
    chapter_one = book.chapter_by_key("ch1")
    keys = ["prologue", "ch1", "ch2", "ch10", "epilogue"]
    files = {
        "prologue.xhtml": _chapter_file("prologue"),
        "ch1.xhtml": ("ch1.xhtml", chapter_one["title"],
                      build.chapter_xhtml({"title": chapter_one["title"], "blocks": chapter_one["blocks"][:5]})),
        "ch1_part2.xhtml": ("ch1_part2.xhtml", "", build.render_blocks(chapter_one["blocks"][5:])),
        "ch2.xhtml": _chapter_file("ch2"),
        "ch10.xhtml": _chapter_file("ch10"),
        "epilogue.xhtml": _chapter_file("epilogue"),
    }
    spine = ["prologue.xhtml", "ch1.xhtml", "ch1_part2.xhtml", "ch2.xhtml", "ch10.xhtml", "epilogue.xhtml"]
    manifest = ["ch10.xhtml", "epilogue.xhtml", "ch1_part2.xhtml", "ch2.xhtml", "prologue.xhtml", "ch1.xhtml"]
    toc = [(book.chapter_by_key(key)["title"], f"{key}.xhtml") for key in keys]
    build.build_epub(out_path, files=[files[name] for name in spine], spine=spine, toc=toc,
                     manifest_order=manifest, identifier="abm-edge-manifest-order")
    return keys


def generate_epub_trailing_backmatter(out_path: str) -> list[str]:
    """Unlisted files around the TOC: a long letter in front, back matter and a preview behind.

    The copyright page is ``1.xhtml`` and Chapter 1 is ``11.xhtml`` so that a
    substring comparison of hrefs confuses the two.
    """
    letter = f"<h1>{book.LETTER_TITLE}</h1>\n" + "\n".join(f"<p>{p}</p>" for p in book.LETTER_PARAGRAPHS)
    preview = f"<h1>{book.PREVIEW_TITLE}</h1>\n<p>{book.PREVIEW_PARAGRAPHS[0]}</p>"
    files = [
        ("titlepage.xhtml", "Title", build.title_page_xhtml()),
        ("1.xhtml", "Copyright", build.copyright_xhtml()),
        ("letter.xhtml", book.LETTER_TITLE, letter),
        _chapter_file("prologue"),
        ("11.xhtml", book.chapter_by_key("ch1")["title"], build.chapter_xhtml(book.chapter_by_key("ch1"))),
        _chapter_file("ch2"),
        _chapter_file("epilogue"),
        ("alsoby.xhtml", book.ALSO_BY_TITLE, build.also_by_xhtml()),
        ("about.xhtml", book.ABOUT_AUTHOR_TITLE, build.about_author_xhtml()),
        ("preview.xhtml", book.PREVIEW_TITLE, preview),
        ("preview2.xhtml", "", f"<p>{book.PREVIEW_PARAGRAPHS[1]}</p>"),
    ]
    toc = [("Prologue", "prologue.xhtml"), (book.chapter_by_key("ch1")["title"], "11.xhtml"),
           (book.chapter_by_key("ch2")["title"], "ch2.xhtml"), ("Epilogue", "epilogue.xhtml")]
    build.build_epub(out_path, files=files, spine=[name for name, _t, _b in files], toc=toc,
                     identifier="abm-edge-trailing")
    return ["prologue", "ch1", "ch2", "epilogue"]


# ══════════════════════════════════════════════════════════════════════════════
# Expectations
# ══════════════════════════════════════════════════════════════════════════════

def _expectation(fmt: str, keys: list[str], what: str, **extra: Any) -> dict[str, Any]:
    """Expected extraction result for a fixture holding the chapters ``keys``."""
    chapters = [book.chapter_by_key(key) for key in keys]
    plain = " ".join(" ".join(book.chapter_plain_paragraphs(chapter)) for chapter in chapters)
    entry: dict[str, Any] = {
        "format": fmt,
        "exercises": what,
        "chapters": [chapter["title"] for chapter in chapters],
        "must_contain": [phrase for phrase in book.MUST_CONTAIN if phrase in plain],
        "must_not_contain": list(book.MUST_NOT_CONTAIN),
        "chapter_must_contain": {
            chapter["title"]: [book.CHAPTER_PHRASES[chapter["title"]]] for chapter in chapters
        },
    }
    for name, value in extra.items():
        if name == "chapters_before":
            entry["chapters"] = value + entry["chapters"]
        elif isinstance(value, dict) and isinstance(entry.get(name), dict):
            entry[name].update(value)
        elif isinstance(value, list) and isinstance(entry.get(name), list):
            entry[name] += value
        else:
            entry[name] = value
    return entry


_README_HEAD: str = """# Source-document fixtures

Everything in this directory is **generated** — do not edit the files by hand.
Regenerate with:

```bash
python tests/fixture_generation/generate_test_documents.py
```

The text is original prose written for this test-suite ("The Lighthouse at
Saltmarsh Point" by the invented author Mara Ellison), so the files carry no
third-party copyright. `expected_chapters.json` is the machine-readable
version of this page: for every fixture it lists the ordered chapter titles
that `scan()` and `extract()` must return, phrases that must appear in the
extracted text, phrases that must NOT appear (front/back matter, running
headers, footnote bodies) and one phrase per chapter that must be found in
that chapter and no other.

## The book

Narrated chapters, in order:

{chapter_list}

`Chapter 10` deliberately follows `Chapter 2`, and `About the Lighthouse`
starts with a skip-list word, so a sort or prefix-match bug changes the list.

Never narrated: title page, copyright page (`All rights reserved`, `ISBN`),
dedication, contents, footnote text, `Notes`, `About the Author`,
`Also by Mara Ellison`. With `scan(path, include_matter=True)` these come back
flagged `probably_matter=True`.

The prose exercises the normaliser: a drop cap, a sentence of more than 400
characters, `"No!" he cried.`, `Dr.` / `Mr.` / `e.g.,` / `etc.,`, `£120`,
`$4,000`, `1,250`, `2.5 miles`, `4:15 p.m.`, `the 3rd of March, 1926`,
`William IV`, `Chapter I`, an em-dash, an ellipsis, a `* * *` scene break, a
footnote marker, a three-column table, italic and bold mid-sentence, a forced
line break, `Plan B`, `Vitamin C`, `café`, `naïve`, `Привет` and `東京`.

## Files

| File | Size | What it is |
|------|------|------------|
{file_table}

## Expected chapters per fixture

{expected_lists}

## Format notes

- **dummy_book.epub** — EPUB 3, one XHTML file per chapter, nav + NCX, PNG
  cover. Footnote is `<sup><a epub:type="noteref">` plus an
  `<aside epub:type="footnote">`.
- **dummy_book.pdf** — A5 pages typeset with PyMuPDF. EVERY page carries the
  running header `The Lighthouse at Saltmarsh Point` and a page number;
  paragraphs are marked by first-line indents only, some lines end in a
  hyphenated word, paragraphs run across page breaks, the footnote sits at
  the foot of its page in small type. Chapters are in the PDF outline.
- **dummy_book_no_outline.pdf** — the same pages without bookmarks.
- **dummy_book.docx** — `Title` / `Heading 1` styles, a real
  `w:footnoteReference` (with a footnotes part), a Word drop cap (framed
  paragraph), a table and a `w:br` line break.
- **dummy_book.odt** — `text:h` headings, a `text:note`, `text:line-break`,
  `text:tab` (in the contents lines) and a `table:table`.
- **dummy_book.txt** — UTF-8, a `Contents` listing, `_italic_` /
  `**bold**`, a pipe table and a `[1]` footnote marker.
- **dummy_book.mobi** — classic MOBI 6: PalmDB container, MOBI header +
  EXTH, uncompressed UTF-8 text records, inline contents page with `filepos`
  links, a cover image record, FLIS/FCIS/EOF records. Built byte by byte in
  `fixture_builders.build_mobi` (no Calibre, no kindlegen) and opened in the
  tests through the same `mobi` (KindleUnpack) reader the application uses.
"""


def _write_readme(src_docs_dir: str, expected: dict[str, Any], notes: dict[str, str]) -> None:
    chapter_list = "\n".join(f"{n}. `{title}`" for n, title in enumerate(book.EXPECTED_TITLES, start=1))
    rows = []
    lists = []
    for name, entry in expected.items():
        size = os.path.getsize(os.path.join(src_docs_dir, name))
        rows.append(f"| `{name}` | {size / 1024:.1f} KB | {notes[name]} |")
        titles = "\n".join(f"{n}. `{title}`" for n, title in enumerate(entry["chapters"], start=1))
        lists.append(f"### {name}\n\n{titles}\n")
    text = _README_HEAD.format(chapter_list=chapter_list, file_table="\n".join(rows),
                               expected_lists="\n".join(lists))
    with open(os.path.join(src_docs_dir, "README.md"), "w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

def generate_all_fixtures() -> dict[str, str]:
    fixtures_dir = get_fixtures_dir()
    src_docs_dir = os.path.join(fixtures_dir, "source_documents")
    text_dir = os.path.join(fixtures_dir, "text")
    os.makedirs(src_docs_dir, exist_ok=True)
    os.makedirs(text_dir, exist_ok=True)

    def _doc(name: str) -> str:
        return os.path.join(src_docs_dir, name)

    paths = {
        "txt": _doc("dummy_book.txt"),
        "docx": _doc("dummy_book.docx"),
        "pdf": _doc("dummy_book.pdf"),
        "epub": _doc("dummy_book.epub"),
        "odt": _doc("dummy_book.odt"),
        "mobi": _doc("dummy_book.mobi"),
        "pdf_no_outline": _doc("dummy_book_no_outline.pdf"),
        "expected_chapters": _doc("expected_chapters.json"),
        "readme": _doc("README.md"),
        "expected_json": os.path.join(text_dir, "expected_extraction.json"),
        "source_txt": os.path.join(text_dir, "dummy_book_source.txt"),
    }
    expected: dict[str, Any] = {}
    notes: dict[str, str] = {}
    table_row = "Monday, 6:10 a.m., 12:25 p.m."

    def _main(name: str, fmt: str, note: str, writer, **extra: Any) -> None:
        print(f"Generating {name}...")
        try:
            writer(_doc(name))
        except ImportError as exc:
            print(f"  SKIPPED ({exc}) — install the missing library to build this fixture.")
            return
        expected[name] = _expectation(fmt, _ALL_KEYS, note, **extra)
        notes[name] = note

    _main("dummy_book.epub", "epub", "Reference EPUB 3: nav + NCX, cover, front and back matter listed in the TOC.",
          generate_epub_fixture, must_contain=[table_row], has_cover=True,
          title=TEST_BOOK_METADATA["title"], author=TEST_BOOK_METADATA["author"],
          matter=["Title Page", "Copyright", "Dedication", book.ABOUT_AUTHOR_TITLE, book.ALSO_BY_TITLE])
    _main("dummy_book.pdf", "pdf", "PDF with outline, running header and page number on every page.",
          generate_pdf_fixture, must_not_contain=[book.RUNNING_HEADER], no_bare_numbers=True,
          title=TEST_BOOK_METADATA["title"], author=TEST_BOOK_METADATA["author"])
    _main("dummy_book_no_outline.pdf", "pdf", "Same PDF without bookmarks: chapters come from heading type size.",
          lambda path: generate_pdf_fixture(path, outline=False),
          must_not_contain=[book.RUNNING_HEADER], no_bare_numbers=True)
    _main("dummy_book.docx", "docx", "DOCX with heading styles, real footnote, drop-cap frame, table, line break.",
          generate_docx_fixture, must_contain=[table_row])
    _main("dummy_book.odt", "odt", "ODT with text:h headings, text:note, line-break, tab, table.",
          generate_odt_fixture, must_contain=[table_row])
    _main("dummy_book.txt", "txt", "Plain text with a Contents listing, pipe table and [1] footnote marker.",
          generate_txt_fixture, must_contain=[table_row])
    _main("dummy_book.mobi", "mobi", "Hand-built classic MOBI 6 (PalmDB + MOBI/EXTH headers, filepos TOC, cover).",
          generate_mobi_fixture, must_contain=[table_row], has_cover=True,
          title=TEST_BOOK_METADATA["title"], author=TEST_BOOK_METADATA["author"])
    generate_txt_fixture(paths["source_txt"])

    edge_cases = [
        ("epub_no_toc.epub", generate_epub_no_toc,
         "A.1b — no nav, empty NCX: chapters come from per-file classification; a chapter split over two files.", {}),
        ("epub_multi_anchor_single_file.epub", generate_epub_multi_anchor,
         "A.1a — whole book in one XHTML file, TOC entries are anchors into it.",
         {"must_contain": [table_row]}),
        ("epub_percent_encoded_hrefs.epub", generate_epub_percent_encoded,
         "A.1e — file names with spaces and an accent, percent-encoded in OPF and NCX.",
         {"must_contain": [table_row]}),
        ("epub_manifest_order_differs.epub", generate_epub_manifest_order,
         "A.1g — scrambled manifest; spine is the reading order; unlisted second file of Chapter 1.",
         {"must_contain": [table_row],
          "chapter_must_contain": {"Chapter 1: The Salt Road": ["e.g., a broken lens"]}}),
        ("epub_trailing_backmatter.epub", generate_epub_trailing_backmatter,
         "A.1c/d/f — unlisted letter before the TOC is kept; unlisted back matter and preview are not; "
         "1.xhtml vs 11.xhtml.",
         {"chapters_before": [book.LETTER_TITLE],
          "must_contain": [table_row, "the town keeps its promises slowly", "mind the second milestone."],
          "must_not_contain": ["The map arrived folded into eighths", "The harbour was on the wrong side"],
          "chapter_must_contain": {book.LETTER_TITLE: ["My dear reader"]}}),
    ]
    for name, writer, note, extra in edge_cases:
        print(f"Generating {name}...")
        keys = writer(_doc(name))
        expected[name] = _expectation("epub", keys, note, **extra)
        notes[name] = note

    print("Saving expected_chapters.json and README.md...")
    with open(paths["expected_chapters"], "w", encoding="utf-8", newline="\n") as handle:
        json.dump(expected, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    _write_readme(src_docs_dir, expected, notes)

    print("Saving expected extraction JSON...")
    expected_data = {
        "metadata": TEST_BOOK_METADATA,
        "chapter_count": len(CHAPTERS_DATA),
        "chapters": [
            {
                "num": num,
                "title": chapter["title"],
                "paragraph_count": len(book.chapter_plain_paragraphs(chapter)),
                "full_text": "\n\n".join(book.chapter_plain_paragraphs(chapter)),
            }
            for num, chapter in enumerate(CHAPTERS_DATA, start=1)
        ],
    }
    with open(paths["expected_json"], "w", encoding="utf-8", newline="\n") as handle:
        json.dump(expected_data, handle, indent=2, ensure_ascii=False)
        handle.write("\n")

    print("All document fixtures generated successfully!")
    return paths


if __name__ == "__main__":
    generate_all_fixtures()
