"""
test_text_processing.py
=======================
Unit tests for text normalization, abbreviation guarding,
dialogue quote preservation, and sentence segmentation.
"""

from __future__ import annotations

import os
import sys
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from audiobook_factory.text_processing import (
    normalize_text,
    smart_sentence_splitter,
    _python_normalize_text,
    _soft_split_long_sentence,
)


class TestTextProcessing:

    def test_normalize_dewrapping_and_dropcaps(self):
        raw = "he is about to come, we\nwill prepare"
        norm = _python_normalize_text(raw)
        assert "we will prepare" in norm

    def test_abbreviations_do_not_split(self):
        text = "Dr. Arthur Pendelton met with Prof. Moriarty and Mrs. Wells at 12:45 P.M. in the library."
        sentences = smart_sentence_splitter(text, max_len=399)
        assert len(sentences) == 1
        assert "Dr." in sentences[0]
        assert "Prof." in sentences[0]

    def test_dialogue_splitting(self):
        text = '"We must leave immediately," said Arthur. "The tower is unstable!" Eleanor nodded in agreement.'
        sentences = smart_sentence_splitter(text, max_len=399)
        assert len(sentences) >= 2

    def test_long_paragraph_soft_splitting_under_max_len(self):
        long_paragraph = (
            "The ancient library of Tingen stood silent beneath the silver moonlight, its towering shelves "
            "laden with thousands of vellum folios, handwritten manuscripts, and leather-bound treatises on astronomy, "
            "alchemy, and the lost histories of the Fourth Epoch. Outside, the cold wind whispered through the marble arches, "
            "rustling the amber leaves of the autumn trees that lined the secluded courtyard where few scholars dared tread."
        )
        max_limit = 100
        chunks = smart_sentence_splitter(long_paragraph, max_len=max_limit)
        for c in chunks:
            assert len(c) <= max_limit, f"Chunk exceeded max_len ({len(c)} > {max_limit}): '{c}'"

    def test_empty_or_whitespace_input(self):
        assert smart_sentence_splitter("", max_len=399) == []
        assert smart_sentence_splitter("   \n\t  ", max_len=399) == []
