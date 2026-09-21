"""Test table colspan and rowspan clamping in HTML preprocessor (BUG-R2-C3-A4-H1).

Docling allocates table grids proportional to declared colspan/rowspan.
Unbounded span attributes (e.g. colspan="9999999") cause multi-GB allocations
leading to process OOM kill. DocumentIngestor._preprocess_html must clamp
colspan and rowspan to a safe maximum (e.g. 1000).
"""
import pytest
from bs4 import BeautifulSoup
from audiobook_factory.extractor_engine import DocumentIngestor


def test_table_span_clamped_in_preprocess_html():
    """Ensure excessive colspan/rowspan attributes are clamped to safe limit (1000)."""
    raw_html = """
    <html>
      <body>
        <table>
          <tr>
            <td colspan="99999999" rowspan="500000">Huge cell</td>
            <th colspan="2000">Header cell</th>
            <td colspan="5" rowspan="2">Normal cell</td>
          </tr>
        </table>
      </body>
    </html>
    """
    cleaned_html, _ = DocumentIngestor._preprocess_html(raw_html)
    soup = BeautifulSoup(cleaned_html, "html.parser")
    
    tds = soup.find_all(["td", "th"])
    # First cell
    assert tds[0]["colspan"] == "1000"
    assert tds[0]["rowspan"] == "1000"
    # Second cell
    assert tds[1]["colspan"] == "1000"
    # Normal cell untouched
    assert tds[2]["colspan"] == "5"
    assert tds[2]["rowspan"] == "2"
