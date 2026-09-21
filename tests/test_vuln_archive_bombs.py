"""Test zip decompression bomb guards for EPUB, DOCX, and ODT (BUG-R2-C3-A1-H1, H2, H3).

Small archive files declaring excessive uncompressed size or member counts must be
rejected BEFORE passing to eager readers (ebooklib, python-docx, odfpy) to prevent OOM.
"""
import io
import zipfile
import pytest
from audiobook_factory.text_extractor import scan, extract


def _create_mock_zip_bomb(file_path: str):
    """Creates a zip file exceeding member count limit (2000 members)."""
    with zipfile.ZipFile(file_path, "w") as zf:
        for i in range(2005):
            zf.writestr(f"file_{i}.xml", b"x")


def test_epub_decompression_bomb_rejected(tmp_path):
    """Ensure scan() and extract() reject EPUB zip bombs."""
    bomb_path = str(tmp_path / "bomb.epub")
    _create_mock_zip_bomb(bomb_path)
    
    # scan() should handle or reject safely without crashing
    res = scan(bomb_path)
    assert not res.has_toc

    with pytest.raises(ValueError, match="Archive too large|uncompressed"):
        extract(bomb_path)


def test_docx_decompression_bomb_rejected(tmp_path):
    """Ensure scan() and extract() reject DOCX zip bombs."""
    bomb_path = str(tmp_path / "bomb.docx")
    _create_mock_zip_bomb(bomb_path)
    
    res = scan(bomb_path)
    assert res.page_count == 0

    chapters, _ = extract(bomb_path)
    assert len(chapters) == 0 or chapters[0].text == ""


def test_odt_decompression_bomb_rejected(tmp_path):
    """Ensure scan() and extract() reject ODT zip bombs."""
    bomb_path = str(tmp_path / "bomb.odt")
    _create_mock_zip_bomb(bomb_path)
    
    res = scan(bomb_path)
    assert res.page_count == 0

    chapters, _ = extract(bomb_path)
    assert len(chapters) == 0 or chapters[0].text == ""
