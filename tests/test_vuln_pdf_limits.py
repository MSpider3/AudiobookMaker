"""Test PDF extraction resource limits (BUG-R2-C3-A3-H1).

PDF chapter extraction must bound total ranges (MAX_RANGES=64), deduplicate
identical ranges, and cap total pages/chars to prevent resource exhaustion.
"""
import pytest
from unittest.mock import MagicMock
from audiobook_factory.text_extractor import _extract_pdf_ranges


def test_extract_pdf_ranges_limits_and_deduplicates():
    """Ensure page ranges are capped at MAX_RANGES and duplicates are skipped."""
    mock_doc = MagicMock()
    mock_doc.page_count = 100
    mock_page = MagicMock()
    mock_page.get_text.return_value = "Sample page content. "
    mock_doc.__getitem__.return_value = mock_page

    mock_ingestor = MagicMock()
    mock_normalizer = MagicMock()
    mock_normalizer.normalize.side_effect = lambda text, **kw: text

    logs = []

    # Provide 100 duplicate ranges of (1, 2) plus distinct ones
    ranges = [(1, 2)] * 50 + [(i, i + 1) for i in range(3, 100)]
    
    with pytest.MonkeyPatch.context() as mp:
        import fitz
        mp.setattr(fitz, "open", lambda p: mock_doc)
        
        chapters = _extract_pdf_ranges(
            "dummy.pdf",
            page_ranges=ranges,
            ingestor=mock_ingestor,
            normalizer=mock_normalizer,
            log=logs.append,
        )

        # Must not exceed MAX_RANGES (64)
        assert len(chapters) <= 64
        # Duplicate (1, 2) must be extracted only once
        first_chapters = [c for c in chapters if "pp. 1–2" in c.title]
        assert len(first_chapters) == 1
