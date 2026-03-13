"""
audiobook_factory/text_extractor.py
====================================
Public API for the 5-phase text extraction pipeline.
Wraps DocumentIngestor, MLClassifier, TextNormalizer from the
production-tested extractor_engine module.

Public API
----------
scan(path)                -> ScanResult          (fast, no OCR – for UI display)
extract(path, selections) -> list[ExtractedChapter]
"""
from __future__ import annotations

import os
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

# ── Ensure project root on sys.path ──────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# ── Import the production classes from the debug script ──────────────────────
# We import lazily so the app starts even if some optional deps are missing.
def _load_pipeline():
    from audiobook_factory.extractor_engine import (  # type: ignore
        DocumentIngestor,
        MLClassifier,
        TextNormalizer,
        ChapterItem,
    )
    return DocumentIngestor, MLClassifier, TextNormalizer, ChapterItem


# ══════════════════════════════════════════════════════════════════════════════
# Public data structures
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ScannedChapter:
    """Lightweight chapter info returned by scan() for the UI checklist."""
    num:        int
    title:      str
    word_count: int
    href:       str = ""   # EPUB/MOBI only


@dataclass
class ScanResult:
    """Result of a fast pre-scan before any heavy processing."""
    file_type:  str              # "epub" | "mobi" | "pdf" | "docx" | "txt" | "odt"
    has_toc:    bool
    chapters:   list[ScannedChapter] = field(default_factory=list)
    title:      str = ""
    author:     str = ""
    cover_data: bytes | None = None
    # For formats without chapter TOC: total page count (0 for TXT)
    page_count: int = 0


@dataclass
class ExtractedChapter:
    """A fully processed chapter ready for TTS."""
    num:        int
    title:      str
    text:       str          # normalized, TTS-ready text
    sentences:  list[str]


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _detect_type(path: str) -> str:
    ext = Path(path).suffix.lower()
    return {
        ".epub": "epub",
        ".mobi": "mobi",
        ".pdf":  "pdf",
        ".docx": "docx",
        ".odt":  "odt",
        ".txt":  "txt",
    }.get(ext, "unknown")


def _epub_metadata(book) -> tuple[str, str, bytes | None]:
    """Extract title, author, cover from an ebooklib Book object."""
    import ebooklib
    title  = (book.get_metadata("DC", "title")  or [("Unknown Title",  {})])[0][0]
    author = (book.get_metadata("DC", "creator") or [("Unknown Author", {})])[0][0]
    cover_data = None
    cover = book.get_item_with_id("cover")
    if cover:
        cover_data = cover.get_content()
    else:
        for item in book.get_items_of_type(ebooklib.ITEM_IMAGE):
            if "cover-image" in item.get_properties():
                cover_data = item.get_content()
                break
    return title, author, cover_data


# ══════════════════════════════════════════════════════════════════════════════
# scan() — fast, no OCR, just TOC extraction for the UI
# ══════════════════════════════════════════════════════════════════════════════

def scan(path: str) -> ScanResult:
    """
    Fast pre-scan; returns chapter list for EPUB/MOBI or page count for others.
    Never runs OCR or Docling — this must be snappy for the UI.
    """
    ftype = _detect_type(path)

    if ftype in ("epub", "mobi"):
        return _scan_epub(path, ftype)
    elif ftype == "pdf":
        return _scan_pdf(path)
    elif ftype == "docx":
        return _scan_docx(path)
    elif ftype == "odt":
        return _scan_odt(path)
    else:  # txt / unknown
        return ScanResult(file_type=ftype, has_toc=False, page_count=0)


def _scan_epub(path: str, ftype: str) -> ScanResult:
    from ebooklib import epub
    from audiobook_factory.extractor_engine import DocumentIngestor  # type: ignore

    try:
        book = epub.read_epub(path)
        title, author, cover_data = _epub_metadata(book)

        ingestor = DocumentIngestor.__new__(DocumentIngestor)
        chapter_hrefs, skip_hrefs, toc_entries = ingestor._walk_epub_toc(book.toc)

        if not chapter_hrefs:
            return ScanResult(file_type=ftype, has_toc=False,
                              title=title, author=author, cover_data=cover_data)

        from bs4 import BeautifulSoup
        chapters: list[ScannedChapter] = []
        num = 1
        for entry in toc_entries:
            if entry.classification != "chapter":
                continue
            # Try to get word count from the item
            item = book.get_item_with_href(entry.href.split("#")[0])
            wc  = 0
            if item:
                soup = BeautifulSoup(item.get_body_content(), "html.parser")
                wc   = len(soup.get_text().split())
            chapters.append(ScannedChapter(
                num=num, title=entry.title, word_count=wc, href=entry.href
            ))
            num += 1

        return ScanResult(
            file_type=ftype, has_toc=bool(chapters),
            chapters=chapters, title=title, author=author,
            cover_data=cover_data,
        )
    except Exception as e:
        print(f"[scan_epub] {e}")
        return ScanResult(file_type=ftype, has_toc=False)


def _scan_pdf(path: str) -> ScanResult:
    page_count = 0
    try:
        import fitz  # PyMuPDF
        doc = fitz.open(path)
        page_count = doc.page_count
        doc.close()
    except Exception:
        pass
    return ScanResult(file_type="pdf", has_toc=False, page_count=page_count)


def _scan_docx(path: str) -> ScanResult:
    page_count = 0
    try:
        import docx2txt  # type: ignore
        # docx doesn't expose page count easily; count sections as proxy
        from docx import Document as _DocxDoc  # type: ignore
        d = _DocxDoc(path)
        # Approximate: every 500 words ≈ 1 page
        words = sum(len(p.text.split()) for p in d.paragraphs)
        page_count = max(1, words // 500)
    except Exception:
        page_count = 0
    return ScanResult(file_type="docx", has_toc=False, page_count=page_count)


def _scan_odt(path: str) -> ScanResult:
    page_count = 0
    try:
        from odf.opendocument import load as odf_load  # type: ignore
        from odf.text import P  # type: ignore
        doc = odf_load(path)
        paras = doc.text.getElementsByType(P)
        words = sum(len(str(p).split()) for p in paras)
        page_count = max(1, words // 500)
    except Exception:
        page_count = 0
    return ScanResult(file_type="odt", has_toc=False, page_count=page_count)


# ══════════════════════════════════════════════════════════════════════════════
# extract() — full extraction with Docling + OCR + normalization
# ══════════════════════════════════════════════════════════════════════════════

def extract(
    path: str,
    selections: list[int] | str | None = None,
    *,
    enable_ocr: bool = False,
    page_ranges: list[tuple[int, int]] | None = None,
    log_fn=None,
) -> list[ExtractedChapter]:
    """
    Full extraction. Returns normalized, sentence-split chapters.

    Parameters
    ----------
    path        : path to the input file
    selections  : list of chapter numbers to extract (EPUB/MOBI); None = all
    enable_ocr  : run EasyOCR on embedded images (EPUB only)
    page_ranges : for PDF/DOCX/ODT — list of (start_page, end_page) tuples, each
                  becomes one chapter. None = whole document.
    log_fn      : optional callable(str) for progress messages
    """
    def log(msg):
        if log_fn:
            log_fn(msg)
        else:
            print(msg)

    ftype = _detect_type(path)

    if ftype in ("epub", "mobi"):
        return _extract_epub(path, selections, enable_ocr=enable_ocr, log=log)
    elif ftype == "txt":
        return _extract_txt(path, log=log)
    else:
        return _extract_paged(path, ftype, page_ranges, log=log)


# ── EPUB / MOBI extraction ────────────────────────────────────────────────────

def _extract_epub(
    path: str,
    selections: list[int] | None,
    *,
    enable_ocr: bool,
    log,
) -> list[ExtractedChapter]:
    from audiobook_factory.extractor_engine import (  # type: ignore
        DocumentIngestor, MLClassifier, TextNormalizer
    )
    from audiobook_factory.text_processing import smart_sentence_splitter

    ingestor   = DocumentIngestor()
    classifier = MLClassifier()
    normalizer = TextNormalizer()

    chapters, skipped, _ = ingestor.ingest_epub(path, classifier, normalizer)

    results: list[ExtractedChapter] = []
    for ch in chapters:
        if selections and ch.num not in selections:
            continue
        log(f"  ✓ Extracted chapter {ch.num}: {ch.title}")
        results.append(ExtractedChapter(
            num=ch.num,
            title=ch.title,
            text=ch.normalized,
            sentences=ch.sentences,
        ))

    return results


# ── TXT extraction ────────────────────────────────────────────────────────────

def _extract_txt(path: str, *, log) -> list[ExtractedChapter]:
    from audiobook_factory.text_processing import normalize_text, smart_sentence_splitter
    from audiobook_factory.extractor_engine import TextNormalizer  # type: ignore

    log("  Reading TXT file...")
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        raw = f.read()

    normalizer = TextNormalizer()
    text = normalizer.normalize(raw, title="", ocr_block_texts=[])
    sentences = smart_sentence_splitter(text)

    return [ExtractedChapter(num=1, title="Full Book", text=text, sentences=sentences)]


# ── PDF / DOCX / ODT page-range extraction ───────────────────────────────────

def _extract_paged(
    path: str,
    ftype: str,
    page_ranges: list[tuple[int, int]] | None,
    *,
    log,
) -> list[ExtractedChapter]:
    from audiobook_factory.extractor_engine import DocumentIngestor, TextNormalizer  # type: ignore
    from audiobook_factory.text_processing import smart_sentence_splitter

    normalizer = TextNormalizer()
    ingestor   = DocumentIngestor()

    if ftype == "pdf":
        return _extract_pdf_ranges(path, page_ranges, ingestor, normalizer, log)
    elif ftype == "docx":
        return _extract_docx(path, page_ranges, normalizer, log)
    elif ftype == "odt":
        return _extract_odt(path, page_ranges, normalizer, log)
    return []


def _extract_pdf_ranges(path, page_ranges, ingestor, normalizer, log):
    from audiobook_factory.extractor_engine import TextNormalizer  # type: ignore
    from audiobook_factory.text_processing import smart_sentence_splitter

    try:
        import fitz
    except ImportError:
        log("[ERROR] PyMuPDF not installed — cannot extract PDF.")
        return []

    doc = fitz.open(path)
    results = []

    if not page_ranges:
        # Whole document
        all_text = "\n\n".join(doc[i].get_text("text") for i in range(doc.page_count))
        text = normalizer.normalize(all_text, title="", ocr_block_texts=[])
        results.append(ExtractedChapter(
            num=1, title="Full Book", text=text,
            sentences=smart_sentence_splitter(text)
        ))
    else:
        for idx, (start, end) in enumerate(page_ranges, 1):
            pages = range(max(0, start - 1), min(end, doc.page_count))
            raw   = "\n\n".join(doc[i].get_text("text") for i in pages)
            text  = normalizer.normalize(raw, title="", ocr_block_texts=[])
            results.append(ExtractedChapter(
                num=idx,
                title=f"Chapter {idx} (pp. {start}–{end})",
                text=text,
                sentences=smart_sentence_splitter(text),
            ))
            log(f"  ✓ Extracted pages {start}–{end}")

    doc.close()
    return results


def _extract_docx(path, page_ranges, normalizer, log):
    from audiobook_factory.text_processing import smart_sentence_splitter
    try:
        from docx import Document as _DocxDoc  # type: ignore
    except ImportError:
        log("[ERROR] python-docx not installed.")
        return []
    d    = _DocxDoc(path)
    raw  = "\n\n".join(p.text for p in d.paragraphs if p.text.strip())
    text = normalizer.normalize(raw, title="", ocr_block_texts=[])
    return [ExtractedChapter(num=1, title="Full Book", text=text,
                             sentences=smart_sentence_splitter(text))]


def _extract_odt(path, page_ranges, normalizer, log):
    from audiobook_factory.text_processing import smart_sentence_splitter
    try:
        from odf.opendocument import load as odf_load  # type: ignore
        from odf.text import P  # type: ignore
    except ImportError:
        log("[ERROR] odfpy not installed.")
        return []
    doc  = odf_load(path)
    raw  = "\n\n".join(str(p) for p in doc.text.getElementsByType(P))
    text = normalizer.normalize(raw, title="", ocr_block_texts=[])
    return [ExtractedChapter(num=1, title="Full Book", text=text,
                             sentences=smart_sentence_splitter(text))]
