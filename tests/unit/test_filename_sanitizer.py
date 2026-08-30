"""
test_filename_sanitizer.py
==========================
Unit tests for Audiobookshelf filename formatting, cross-platform illegal character
sanitization, Windows reserved device name guarding, and filesystem boundary truncation.
"""

from __future__ import annotations

import os
import sys
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from audiobook_factory.filename_sanitizer import (
    make_safe_filename,
    _sanitize_base_name,
    _WIN_RESERVED,
    _FORBIDDEN,
)


class TestFilenameSanitizer:

    def test_basic_chapter_naming(self):
        name = make_safe_filename("The Beginning", 1, "./output", ".mp3")
        assert name.startswith("Chapter ")
        assert "The Beginning" in name
        assert name.endswith(".mp3")

    def test_strip_illegal_characters(self):
        dirty = 'Chapter 1: "The Lost <City> / of *Gold*?" | [Full]'
        clean = _sanitize_base_name(dirty)
        for bad in ('<', '>', ':', '"', '/', '\\', '|', '?', '*'):
            assert bad not in clean

    def test_strip_redundant_chapter_prefix(self):
        name = make_safe_filename("Chapter 1: The Awakening", 1, "./output", ".mp3")
        assert name == "Chapter 1 - The Awakening.mp3"

    def test_windows_reserved_names(self):
        clean_con = _sanitize_base_name("CON")
        assert clean_con == "CON file"

        clean_aux = _sanitize_base_name("aux")
        assert clean_aux == "aux file"

        clean_nul = _sanitize_base_name("NUL")
        assert clean_nul == "NUL file"

        clean_com1 = _sanitize_base_name("COM1")
        assert clean_com1 == "COM1 file"

        clean_lpt9 = _sanitize_base_name("LPT9")
        assert clean_lpt9 == "LPT9 file"

        clean_valid = _sanitize_base_name("ValidTitle")
        assert clean_valid == "ValidTitle"

    def test_extreme_filename_length_truncation(self):
        very_long_title = "A" * 400
        safe_name = make_safe_filename(very_long_title, 1, "/tmp", ".mp3")
        assert len(safe_name.encode("utf-8")) <= 255
        assert safe_name.endswith(".mp3")
        assert "Chapter 1 - " in safe_name
