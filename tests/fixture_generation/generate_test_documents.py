"""
generate_test_documents.py
===========================
Generates deterministic synthetic book fixtures across multiple formats:
- EPUB (.epub)
- PDF (.pdf)
- DOCX (.docx)
- ODT (.odt)
- TXT (.txt)

The generated corpus contains varied sentence lengths, dialogue, numbers,
abbreviations (e.g., Mr., Dr., etc.), punctuation, and unicode characters
to thoroughly test text extraction and chapter detection pipelines.
"""

from __future__ import annotations

import json
import os
import sys

# Structured 3-chapter test book content
TEST_BOOK_METADATA = {
    "title": "The Chronicles of Antiquity",
    "author": "Dr. Arthur Pendelton, Ph.D.",
    "language": "en",
    "description": "A synthetic multi-chapter test novel for AudiobookMaker validation."
}

CHAPTERS_DATA = [
    {
        "num": 1,
        "title": "Chapter 1 - The Crimson Tower",
        "paragraphs": [
            "Dr. Arthur Pendelton adjusted his gold-rimmed spectacles, gazing up at the ominous silhouette of the Crimson Tower. The clock tower in the town square struck 12:45 PM with a resounding chime.",
            "\"We must proceed with utmost caution,\" whispered Eleanor, checking her brass compass. \"The magnetic anomalies began at exactly 3.14159 MHz, just as Prof. Moriarty predicted in his 1888 journal.\"",
            "The heavy iron doors groaned on rusty hinges as they stepped across the marble threshold. Dust particles danced in the amber sunbeams filtering through stained-glass arches. On the wall hung a faded tapestry depicting the battle of Saint-Germain in A.D. 1492.",
            "\"Look here, Arthur!\" she exclaimed, pointing to an inscribed inscription: 'Ignorantia juris non excusat—knowledge is the only true shield.' Could this ancient proverb hold the secret to the cipher?"
        ]
    },
    {
        "num": 2,
        "title": "Chapter 2 - Whispers in the Archive",
        "paragraphs": [
            "Inside the subterranean library, towering mahogany shelves held thousands of vellum folios. The air smelled of aged cedar, dried lavender, and old ink.",
            "Eleanor retrieved a leather-bound volume stamped with the emblem of the Royal Philosophical Society. She flipped to page 247, scanning the meticulous handwritten notes of Mrs. H. G. Wells.",
            "\"According to section 4(b),\" Arthur murmured, studying the celestial chart, \"the planetary alignment will occur on Nov. 15th, 2026. That gives us less than 72 hours.\"",
            "Suddenly, a sharp noise echoed from the corridor—a rhythmic clacking sound, like boots on flagstone. Step, click, step, click. Someone—or something—was approaching with chilling deliberate speed."
        ]
    },
    {
        "num": 3,
        "title": "Chapter 3 - The Obsidian Key",
        "paragraphs": [
            "Arthur grasped the obsidian key firmly in his right palm. Its surface was cold as glacial ice, vibrating with a subtle, rhythmic frequency of 440 Hz.",
            "\"Insert it into the central cylinder,\" Eleanor urged, holding her lantern high to illuminate the bronze keyhole.",
            "With a decisive twist, the tumblers aligned with three distinct clicks. A hidden compartment sprang open, revealing a velvet-lined casket containing the golden astrolabe of King Solomon.",
            "They had solved the riddle of the Crimson Tower. The truth of antiquity was finally theirs to safeguard for generations to come."
        ]
    }
]


def get_fixtures_dir() -> str:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(root, "fixtures")


def generate_txt_fixture(out_path: str) -> None:
    """Generate plaintext book fixture."""
    lines = [
        f"{TEST_BOOK_METADATA['title']}",
        f"By {TEST_BOOK_METADATA['author']}",
        "",
        "==================================================",
        ""
    ]
    for ch in CHAPTERS_DATA:
        lines.append(f"{ch['title']}")
        lines.append("")
        for p in ch["paragraphs"]:
            lines.append(p)
            lines.append("")
        lines.append("--------------------------------------------------")
        lines.append("")
    
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def generate_docx_fixture(out_path: str) -> None:
    """Generate DOCX book fixture using python-docx."""
    from docx import Document
    from docx.shared import Pt

    doc = Document()
    doc.add_heading(TEST_BOOK_METADATA["title"], level=0)
    doc.add_paragraph(f"By {TEST_BOOK_METADATA['author']}")
    
    for ch in CHAPTERS_DATA:
        doc.add_heading(ch["title"], level=1)
        for p in ch["paragraphs"]:
            doc.add_paragraph(p)
            
    doc.save(out_path)


def generate_pdf_fixture(out_path: str) -> None:
    """Generate PDF book fixture using pymupdf (fitz)."""
    import fitz

    doc = fitz.open()
    toc = []
    page_num = 1

    # Title page
    page = doc.new_page()
    page.insert_text(fitz.Point(72, 100), TEST_BOOK_METADATA["title"], fontsize=20)
    page.insert_text(fitz.Point(72, 140), f"By {TEST_BOOK_METADATA['author']}", fontsize=12)
    
    for ch in CHAPTERS_DATA:
        page_num += 1
        page = doc.new_page()
        toc.append([1, ch["title"], page_num])
        
        y = 72
        page.insert_text(fitz.Point(72, y), ch["title"], fontsize=16)
        y += 40
        for p in ch["paragraphs"]:
            rect = fitz.Rect(72, y, 520, y + 100)
            page.insert_textbox(rect, p, fontsize=11)
            y += 80

    doc.set_toc(toc)
    doc.save(out_path)
    doc.close()


def generate_epub_fixture(out_path: str) -> None:
    """Generate EPUB book fixture using EbookLib."""
    from ebooklib import epub

    book = epub.EpubBook()
    book.set_identifier("abm-synthetic-fixture-001")
    book.set_title(TEST_BOOK_METADATA["title"])
    book.set_language(TEST_BOOK_METADATA["language"])
    book.add_author(TEST_BOOK_METADATA["author"])

    chapters = []
    toc = []

    for i, ch in enumerate(CHAPTERS_DATA, start=1):
        c = epub.EpubHtml(title=ch["title"], file_name=f"chap_{i:02d}.xhtml", lang="en")
        body_html = [f"<h1>{ch['title']}</h1>"]
        for p in ch["paragraphs"]:
            body_html.append(f"<p>{p}</p>")
        c.set_content("\n".join(body_html))
        book.add_item(c)
        chapters.append(c)
        toc.append(c)

    book.toc = tuple(toc)
    book.add_item(epub.EpubNcx())
    book.add_item(epub.EpubNav())

    book.spine = ["nav"] + chapters
    epub.write_epub(out_path, book)


def generate_odt_fixture(out_path: str) -> None:
    """Generate ODT book fixture using odfpy."""
    from odf.opendocument import OpenDocumentText
    from odf.text import H, P

    doc = OpenDocumentText()
    doc.text.addElement(H(outlinelevel=1, text=TEST_BOOK_METADATA["title"]))
    doc.text.addElement(P(text=f"By {TEST_BOOK_METADATA['author']}"))

    for ch in CHAPTERS_DATA:
        doc.text.addElement(H(outlinelevel=2, text=ch["title"]))
        for p in ch["paragraphs"]:
            doc.text.addElement(P(text=p))

    doc.save(out_path)


def generate_all_fixtures() -> dict[str, str]:
    fixtures_dir = get_fixtures_dir()
    src_docs_dir = os.path.join(fixtures_dir, "source_documents")
    text_dir = os.path.join(fixtures_dir, "text")
    os.makedirs(src_docs_dir, exist_ok=True)
    os.makedirs(text_dir, exist_ok=True)

    paths = {
        "txt": os.path.join(src_docs_dir, "dummy_book.txt"),
        "docx": os.path.join(src_docs_dir, "dummy_book.docx"),
        "pdf": os.path.join(src_docs_dir, "dummy_book.pdf"),
        "epub": os.path.join(src_docs_dir, "dummy_book.epub"),
        "odt": os.path.join(src_docs_dir, "dummy_book.odt"),
        "expected_json": os.path.join(text_dir, "expected_extraction.json"),
        "source_txt": os.path.join(text_dir, "dummy_book_source.txt")
    }

    print("Generating TXT fixture...")
    generate_txt_fixture(paths["txt"])
    generate_txt_fixture(paths["source_txt"])

    print("Generating DOCX fixture...")
    generate_docx_fixture(paths["docx"])

    print("Generating PDF fixture...")
    generate_pdf_fixture(paths["pdf"])

    print("Generating EPUB fixture...")
    generate_epub_fixture(paths["epub"])

    print("Generating ODT fixture...")
    generate_odt_fixture(paths["odt"])

    print("Saving expected extraction JSON...")
    expected_data = {
        "metadata": TEST_BOOK_METADATA,
        "chapter_count": len(CHAPTERS_DATA),
        "chapters": [
            {
                "num": ch["num"],
                "title": ch["title"],
                "paragraph_count": len(ch["paragraphs"]),
                "full_text": "\n\n".join(ch["paragraphs"])
            }
            for ch in CHAPTERS_DATA
        ]
    }
    with open(paths["expected_json"], "w", encoding="utf-8") as f:
        json.dump(expected_data, f, indent=2)

    print("All document fixtures generated successfully!")
    return paths


if __name__ == "__main__":
    generate_all_fixtures()
