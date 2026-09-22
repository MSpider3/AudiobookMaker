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

    # ── Bug regression tests ──────────────────────────────────────────────

    def test_epub_metadata_handles_missing_get_properties(self):
        """Regression: EpubImage objects lack get_properties(), _epub_metadata
        must not crash when iterating ITEM_IMAGE items."""
        from audiobook_factory.text_extractor import _epub_metadata
        from unittest.mock import MagicMock

        # Build a fake book whose ITEM_IMAGE items lack get_properties
        fake_book = MagicMock()
        fake_book.get_metadata.side_effect = lambda ns, key: {
            "title": [("Test Book", {})],
            "creator": [("Test Author", {})],
        }.get(key, [])

        # Simulate an EpubImage that does NOT have get_properties
        fake_image = MagicMock(spec=["get_content", "file_name"])
        fake_image.get_content.return_value = b"\x89PNG" + b"\x00" * 2000
        fake_image.file_name = "images/cover.png"
        del fake_image.get_properties  # ensure it's absent

        import ebooklib
        fake_book.get_items_of_type.return_value = [fake_image]
        fake_book.file_name = None

        # Must not raise
        title, author, cover = _epub_metadata(fake_book)
        assert title == "Test Book"
        assert author == "Test Author"

    def test_skip_toc_allows_narrable_chapters(self):
        """Regression: _SKIP_TOC_TITLE must NOT skip prologue, epilogue,
        introduction, preface, foreword, afterword — these are narrable."""
        from audiobook_factory.extractor_engine import _SKIP_TOC_TITLE

        narrable = [
            "Prologue", "Epilogue", "Introduction",
            "Preface", "Foreword", "Afterword",
        ]
        for title in narrable:
            assert _SKIP_TOC_TITLE.match(title) is None, (
                f"_SKIP_TOC_TITLE incorrectly skips narrable chapter: {title!r}"
            )

    def test_skip_toc_still_skips_non_content(self):
        """_SKIP_TOC_TITLE should still skip truly non-content items."""
        from audiobook_factory.extractor_engine import _SKIP_TOC_TITLE

        non_content = [
            "Table of Contents", "Copyright", "Cover",
            "Title Page", "Index", "Bibliography",
            "Glossary", "Credits",
        ]
        for title in non_content:
            assert _SKIP_TOC_TITLE.match(title) is not None, (
                f"_SKIP_TOC_TITLE should skip non-content: {title!r}"
            )

    def test_extract_epub_returns_cover_bytes_not_toc_entries(self, fixtures_dir):
        """Regression: _extract_epub must return (chapters, cover_bytes),
        not (chapters, toc_entries)."""
        path = os.path.join(fixtures_dir, "dummy_book.epub")
        if not os.path.exists(path):
            pytest.skip("Fixture not found")
        chapters, cover = extract(path)
        # cover must be None or bytes, never a list of TocEntry objects
        assert cover is None or isinstance(cover, bytes), (
            f"Expected cover to be bytes or None, got {type(cover).__name__}"
        )
