"""
test_text_extractor.py
======================
Unit tests for text extraction across all supported document formats
(EPUB, PDF, DOCX, ODT, TXT).
"""

from __future__ import annotations

import os
import sys
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from audiobook_factory.text_extractor import scan, extract, _detect_type


class TestTextExtractor:

    @pytest.fixture
    def fixtures_dir(self):
        return os.path.join(_ROOT, "tests", "fixtures", "source_documents")

    def test_detect_type_extensions(self):
        assert _detect_type("book.epub") == "epub"
        assert _detect_type("book.mobi") == "mobi"
        assert _detect_type("book.pdf") == "pdf"
        assert _detect_type("book.docx") == "docx"
        assert _detect_type("book.odt") == "odt"
        assert _detect_type("book.txt") == "txt"

    def test_extract_epub(self, fixtures_dir):
        path = os.path.join(fixtures_dir, "dummy_book.epub")
        if not os.path.exists(path):
            pytest.skip("Fixture not found")
        chapters, cover = extract(path)
        assert len(chapters) == 3
        assert "Crimson Tower" in chapters[0].title
        assert len(chapters[0].text) > 50
        assert len(chapters[0].sentences) > 0

    def test_extract_txt(self, fixtures_dir):
        path = os.path.join(fixtures_dir, "dummy_book.txt")
        if not os.path.exists(path):
            pytest.skip("Fixture not found")
        chapters, cover = extract(path)
        assert len(chapters) >= 1
        assert "Arthur" in chapters[0].text

    def test_extract_docx(self, fixtures_dir):
        path = os.path.join(fixtures_dir, "dummy_book.docx")
        if not os.path.exists(path):
            pytest.skip("Fixture not found")
        chapters, cover = extract(path)
        assert len(chapters) >= 1
        assert "Chronicles" in chapters[0].text or "Crimson" in chapters[0].text

    def test_extract_pdf(self, fixtures_dir):
        path = os.path.join(fixtures_dir, "dummy_book.pdf")
        if not os.path.exists(path):
            pytest.skip("Fixture not found")
        chapters, cover = extract(path)
        assert len(chapters) >= 1
        assert len(chapters[0].text) > 50

    def test_extract_odt(self, fixtures_dir):
        path = os.path.join(fixtures_dir, "dummy_book.odt")
        if not os.path.exists(path):
            pytest.skip("Fixture not found")
        chapters, cover = extract(path)
        assert len(chapters) >= 1
        assert len(chapters[0].text) > 50

    def test_scan_epub_metadata_and_chapters(self, fixtures_dir):
        path = os.path.join(fixtures_dir, "dummy_book.epub")
        if not os.path.exists(path):
            pytest.skip("Fixture not found")
        res = scan(path)
        assert res.file_type == "epub"
        assert len(res.chapters) == 3
        assert res.title == "The Chronicles of Antiquity"
