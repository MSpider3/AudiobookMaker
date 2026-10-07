"""
generate_long_books.py
======================
Builds the long test books: one ten-chapter novella of about twenty pages in
each of seven languages (English, French, Russian, Hindi, Chinese, Japanese,
Korean).

They exist for what the small ``dummy_book`` fixtures cannot show: how long a
real book takes, whether two GPUs share a chapter of realistic length, and
whether text in other scripts is split, synthesized and scored correctly.

The text lives in ``long_books/<code>.json`` (original prose written for this
project, no third-party text). This script turns each file into
``tests/fixtures/source_documents/long_book_<code>.epub`` and writes
``long_books.json``, which lists every book with its chapter titles and size.

Usage::

    python tests/fixture_generation/generate_long_books.py
    python tests/kaggle/sync_assets.py        # copy them next to the notebook's other inputs
"""

from __future__ import annotations

import json
import os
import sys
import unicodedata
from html import escape
from typing import Any

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tests.fixture_generation.fixture_builders import write_zip  # noqa: E402

SOURCES_DIR: str = os.path.join(_ROOT, "tests", "fixture_generation", "long_books")
OUTPUT_DIR: str = os.path.join(_ROOT, "tests", "fixtures", "source_documents")
MANIFEST_NAME: str = "long_books.json"
# Reading order of the report and the notebook.
BOOK_CODES: tuple[str, ...] = ("en", "fr", "ru", "hi", "zh", "ja", "ko")

_XHTML_NS: str = "http://www.w3.org/1999/xhtml"
_OPS_NS: str = "http://www.idpf.org/2007/ops"
_CONTAINER_XML: str = (
    '<?xml version="1.0" encoding="utf-8"?>\n'
    '<container version="1.0" xmlns="urn:oasis:names:tc:opendocument:xmlns:container">\n'
    '  <rootfiles><rootfile full-path="OEBPS/content.opf" media-type="application/oebps-package+xml"/></rootfiles>\n'
    "</container>\n"
)


def book_file_name(code: str) -> str:
    """File name of the long book for a language code."""
    return f"long_book_{code}.epub"


def load_source(code: str) -> dict[str, Any]:
    """Reads and checks one book's source file.

    Raises
    ------
    ValueError
        If the file does not have ten chapters of plain-text paragraphs.
    """
    with open(os.path.join(SOURCES_DIR, f"{code}.json"), encoding="utf-8") as fh:
        book = json.load(fh)
    if book.get("code") != code or len(book.get("chapters", [])) != 10:
        raise ValueError(f"{code}.json: expected code {code!r} and ten chapters")
    for chapter in book["chapters"]:
        if not chapter.get("title", "").strip() or not chapter.get("paragraphs"):
            raise ValueError(f"{code}.json: a chapter has no title or no text")
        for paragraph in chapter["paragraphs"]:
            if not paragraph.strip() or "\n" in paragraph:
                raise ValueError(f"{code}.json: empty paragraph or line break inside one")
    return book


def _spoken_characters(text: str) -> int:
    """Characters that are neither whitespace nor punctuation."""
    return sum(1 for char in text if not char.isspace() and not unicodedata.category(char).startswith("P"))


def _page(lang: str, title: str, body: str) -> bytes:
    return (
        '<?xml version="1.0" encoding="utf-8"?>\n<!DOCTYPE html>\n'
        f'<html xmlns="{_XHTML_NS}" xmlns:epub="{_OPS_NS}" lang="{lang}" xml:lang="{lang}">\n'
        f"<head><title>{escape(title)}</title></head>\n<body>\n{body}\n</body>\n</html>\n"
    ).encode("utf-8")


def build_long_epub(book: dict[str, Any], path: str) -> None:
    """Writes *book* as an EPUB 3 file with a navigation document and an NCX."""
    lang, title, author = book["code"], book["title"], book["author"]
    identifier = f"urn:audiobookmaker:long-book:{lang}"
    chapters = [(f"chapter_{index:02d}.xhtml", chapter) for index, chapter in enumerate(book["chapters"], 1)]

    members: list[tuple[str, bytes]] = [
        ("mimetype", b"application/epub+zip"),
        ("META-INF/container.xml", _CONTAINER_XML.encode("utf-8")),
    ]
    manifest = ['<item id="nav" href="nav.xhtml" media-type="application/xhtml+xml" properties="nav"/>',
                '<item id="ncx" href="toc.ncx" media-type="application/x-dtbncx+xml"/>',
                '<item id="title" href="title.xhtml" media-type="application/xhtml+xml"/>']
    spine = ['<itemref idref="title"/>']
    for index, (name, _chapter) in enumerate(chapters, 1):
        manifest.append(f'<item id="ch{index:02d}" href="{name}" media-type="application/xhtml+xml"/>')
        spine.append(f'<itemref idref="ch{index:02d}"/>')
    opf = (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<package xmlns="http://www.idpf.org/2007/opf" version="3.0" unique-identifier="book-id">\n'
        '  <metadata xmlns:dc="http://purl.org/dc/elements/1.1/">\n'
        f'    <dc:identifier id="book-id">{identifier}</dc:identifier>\n'
        f"    <dc:title>{escape(title)}</dc:title>\n"
        f"    <dc:creator>{escape(author)}</dc:creator>\n"
        f"    <dc:language>{lang}</dc:language>\n"
        '    <meta property="dcterms:modified">2026-01-01T00:00:00Z</meta>\n'
        "  </metadata>\n"
        "  <manifest>\n    " + "\n    ".join(manifest) + "\n  </manifest>\n"
        '  <spine toc="ncx">\n    ' + "\n    ".join(spine) + "\n  </spine>\n"
        "</package>\n"
    )
    nav_items = "\n".join(
        f'      <li><a href="{name}">{escape(chapter["title"])}</a></li>' for name, chapter in chapters
    )
    nav = _page(lang, title, f'<nav epub:type="toc" id="toc">\n    <ol>\n{nav_items}\n    </ol>\n</nav>')
    nav_points = "\n".join(
        f'    <navPoint id="np{index}" playOrder="{index}"><navLabel><text>{escape(chapter["title"])}</text></navLabel>'
        f'<content src="{name}"/></navPoint>'
        for index, (name, chapter) in enumerate(chapters, 1)
    )
    ncx = (
        '<?xml version="1.0" encoding="utf-8"?>\n'
        '<ncx xmlns="http://www.daisy.org/z3986/2005/ncx/" version="2005-1">\n'
        f'  <head><meta name="dtb:uid" content="{identifier}"/></head>\n'
        f"  <docTitle><text>{escape(title)}</text></docTitle>\n"
        f"  <navMap>\n{nav_points}\n  </navMap>\n</ncx>\n"
    )
    members += [
        ("OEBPS/content.opf", opf.encode("utf-8")),
        ("OEBPS/nav.xhtml", nav),
        ("OEBPS/toc.ncx", ncx.encode("utf-8")),
        ("OEBPS/title.xhtml", _page(lang, title, f"<h1>{escape(title)}</h1>\n<p>{escape(author)}</p>")),
    ]
    for name, chapter in chapters:
        body = f"<h1>{escape(chapter['title'])}</h1>\n" + "\n".join(
            f"<p>{escape(paragraph)}</p>" for paragraph in chapter["paragraphs"]
        )
        members.append((f"OEBPS/{name}", _page(lang, chapter["title"], body)))
    write_zip(path, members)


def describe(book: dict[str, Any]) -> dict[str, Any]:
    """Manifest entry for one book: what extraction must find, and its size."""
    text = " ".join(paragraph for chapter in book["chapters"] for paragraph in chapter["paragraphs"])
    return {
        "file": book_file_name(book["code"]),
        "language": book["language"],
        "code": book["code"],
        "title": book["title"],
        "author": book["author"],
        "chapters": [chapter["title"] for chapter in book["chapters"]],
        "paragraphs": sum(len(chapter["paragraphs"]) for chapter in book["chapters"]),
        "words": len(text.split()),
        "characters": _spoken_characters(text),
    }


def generate_long_books() -> dict[str, Any]:
    """Builds every long book and the manifest; returns the manifest."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    manifest: dict[str, Any] = {"books": []}
    for code in BOOK_CODES:
        book = load_source(code)
        build_long_epub(book, os.path.join(OUTPUT_DIR, book_file_name(code)))
        manifest["books"].append(describe(book))
    with open(os.path.join(OUTPUT_DIR, MANIFEST_NAME), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, ensure_ascii=False, indent=1)
        fh.write("\n")
    return manifest


if __name__ == "__main__":
    for entry in generate_long_books()["books"]:
        print(f"{entry['file']}: {entry['language']}, {len(entry['chapters'])} chapters, "
              f"{entry['words']} words, {entry['characters']} characters — {entry['title']}")
