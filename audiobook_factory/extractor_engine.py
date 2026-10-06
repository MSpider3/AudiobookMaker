"""
audiobook_factory/extractor_engine.py  —  5-Phase Hybrid AI Extraction Engine
==============================================================================
Production extraction engine: DocumentIngestor, MLClassifier, TextNormalizer.
Imported by audiobook_factory/text_extractor.py as the public API backend.

  Phase 1: DocumentIngestor  — TOC baseline + drop-cap pre-processing
  Phase 2: DocumentIngestor  — Docling ingestion with explicit format routing
  Phase 3: MLClassifier      — XGBoost feature extraction + stub classifier
  Phase 4: TextNormalizer    — LLM OCR repair stub + broken-line heuristic + noise filtering
"""

from __future__ import annotations

import gc
import html as html_lib
import inspect
import json
import logging
import os
import posixpath
import re
import shutil
import statistics
import sys
import tempfile
import time
import traceback
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import unquote

import ebooklib
from bs4 import BeautifulSoup, NavigableString
from ebooklib import epub

logger = logging.getLogger(__name__)

# ── project root & temp folder ────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
_TEMP_DIR = _ROOT / "temp"
_TEMP_DIR.mkdir(parents=True, exist_ok=True)

# ── Safe table span limit to prevent Docling OOM (BUG-R2-C3-A4-H1) ───────────
_MAX_TABLE_SPAN: int = 1000

# ── sys.path so the project root is importable ───────────────────────────────
# This file lives in audiobook_factory/, so go up one level to find the project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from audiobook_factory.text_processing import normalize_text, smart_sentence_splitter

# ── Docling ───────────────────────────────────────────────────────────────────
try:
    from docling.document_converter import DocumentConverter  # type: ignore
    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False
    logger.info("Docling not installed — using the BeautifulSoup/PyMuPDF extraction paths.")

# ── PyMuPDF (optional, for PDF TOC) ──────────────────────────────────────────
try:
    try:
        import pymupdf as fitz  # PyMuPDF (new name)
    except ImportError:
        import fitz  # PyMuPDF (legacy)
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

# ══════════════════════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════════════════════
SUPPORTED_EXT    = {".epub", ".mobi", ".azw", ".azw3", ".pdf", ".docx", ".odt", ".txt"}
MAX_SENTENCE_LEN = 399   # matches config.py

# Chapter keywords for heuristic scoring
_CHAPTER_KW = re.compile(
    r"^\s*(chapter|part|book|prologue|epilogue|interlude|volume|act|section)\b",
    re.I,
)

# Skip-list for TOC entry titles (front/back-matter, gallery, legal…)
_SKIP_TOC_TITLE = re.compile(
    r"^\s*(?:"
    # Unambiguous front/back matter — a prefix is enough.
    r"(?:table\s*of\s*contents|copyright|title\s*page|postscript|newsletter|"
    r"image\s*gallery|to\s*be\s*continued|back\s*cover|coloph|errata|"
    r"bibliography|glossary|character\s*gallery|pathways\s*guide|contact\s*us|"
    r"half[\s-]*title|acknowledge?ments?\b|also\s+(?:by|from|available)\b|"
    r"(?:other|more)\s+(?:books|titles|works|novels)\b|books\s+by\b|"
    r"by\s+the\s+same\s+author|(?:advance\s+)?praise\s+for\b|"
    r"(?:an?\s+)?(?:excerpt|preview|sneak\s+peek)\s+(?:of|from)\b|"
    r"list\s+of\s+(?:illustrations|figures|tables|maps|plates)|frontispiece|"
    r"reading\s+group\s+guide|discussion\s+questions|questions\s+for\s+discussion)"
    r"|"
    # Ordinary words that also begin real chapter titles ("About a Boy",
    # "Maple Street", "End of the Road", "Cover of Darkness", "Notes from
    # Underground"): skip them only when they are the entire title.
    r"(?:toc|contents|index|(?:front\s*)?cover(?:\s*(?:page|image|art))?|"
    r"about(?:\s+(?:the\s+)?(?:authors?|book|publisher|translator|illustrator|series)"
    r"|\s+this\s+(?:book|edition))?|"
    r"characters?(?:\s+(?:list|profiles?|introduction))?|locations?|maps?|"
    r"pathways?|credits?|"
    r"dedication|(?:end|foot)?\s*notes|references|works\s+cited|further\s+reading|"
    r"permissions|legal\s+notice|disclaimer|imprint|preview|sneak\s+peek|"
    r"end\s*of\s*(?:the\s+)?(?:book|volume|vol\.?|part|preview|sample|excerpt)(?:\s+\w+)?)"
    r"\s*[.:!]*\s*$"
    r")",
    re.I,
)

# Headings that open a narrable unit on their own ("Chapter 7", "Prologue").
_STRICT_CHAPTER_HEADING = re.compile(
    r"^\s*(?:(?:chapter|part|book|volume|act)\s+(?:\d+|[ivxlcdm]+|[a-z]+)\b"
    r"|prologue|epilogue|interlude|introduction|preface|foreword|afterword)",
    re.I,
)

# ``epub:type`` values that mark a document as front/back matter.
_MATTER_EPUB_TYPES: frozenset[str] = frozenset({
    "cover", "titlepage", "halftitlepage", "copyright-page", "toc", "landmarks",
    "loi", "lot", "index", "colophon", "imprint", "imprimatur", "dedication",
    "acknowledgments", "bibliography", "glossary", "contributors",
    "other-credits", "errata", "seriespage", "footnotes", "endnotes",
    "rearnotes", "page-list", "ad", "advertisement",
})

# Phrases that only ever appear on copyright / licence pages.
_MATTER_MARKERS = re.compile(
    r"all\s+rights\s+reserved|\bisbn(?:-1[03])?\b[\s:]*[\dxX-]{9,}|"
    r"(?:copyright|\(c\)|©)\s*(?:©|\(c\))?\s*(?:by\s+)?(?:19|20)\d\d|"
    r"no\s+part\s+of\s+this\s+(?:book|publication|work)\s+may|library\s+of\s+congress|"
    r"cataloging[- ]in[- ]publication|project\s+gutenberg(?:\s+e-?book|\s+license|-tm)|"
    r"this\s+(?:e-?book|edition)\s+is\s+(?:licensed|for\s+the\s+use)",
    re.I,
)
# Weaker hints: ordinary prose can contain one of these, a copyright page has several.
_MATTER_HINTS = re.compile(
    r"first\s+(?:published|edition|printing)\b|published\s+by\b|printed\s+in\s+the\b|"
    r"cover\s+(?:design|art|illustration)\s+by\b|\bimprint\s+of\b|www\.\S+|https?://\S+",
    re.I,
)
_TOC_LINE_TAIL = re.compile(r"(?:[.…\s]{2,}|\s)\d{1,4}\s*$")

_WORD = re.compile(r"\S+")

# Minimum sizes used when a document has to be classified without a TOC.
_MIN_CHAPTER_CHARS: int = 50
_MIN_UNLISTED_NARRATIVE_WORDS: int = 150


def looks_like_matter(text: str) -> bool:
    """Tells whether a block of text reads like a copyright page or a contents list.

    Parameters
    ----------
    text : str
        Plain text of a page, file or leading section.

    Returns
    -------
    bool
        True for copyright/licence boilerplate and for table-of-contents
        listings; False for ordinary prose.
    """
    head = text[:1500]
    if _MATTER_MARKERS.search(head):
        return True
    if len({m.group(0).lower()[:8] for m in _MATTER_HINTS.finditer(head)}) >= 2:
        return True
    lines = [ln.strip() for ln in head.split("\n") if ln.strip()]
    if len(lines) >= 4:
        short = [ln for ln in lines if len(ln) <= 70]
        listed = [
            ln for ln in short
            if _TOC_LINE_TAIL.search(ln) or _CHAPTER_KW.match(ln) or _SKIP_TOC_TITLE.match(ln)
        ]
        if len(short) >= 0.8 * len(lines) and len(listed) >= 0.6 * len(lines):
            return True
    return False


def normalize_href(href: str, base_dir: str = "") -> tuple[str, str]:
    """Splits an EPUB href into a comparable (path, anchor) pair.

    ebooklib unquotes manifest hrefs but returns NCX/nav hrefs verbatim, so
    both sides are percent-decoded and path-normalised before comparing.

    Parameters
    ----------
    href : str
        Href as found in a TOC entry or a manifest item.
    base_dir : str
        Directory the href is relative to ("" for the OPF directory).

    Returns
    -------
    tuple[str, str]
        Normalised path and decoded fragment ("" when absent).
    """
    path, _, anchor = (href or "").partition("#")
    path = unquote(path).replace("\\", "/")
    if base_dir and path and not path.startswith("/"):
        path = posixpath.join(base_dir, path)
    path = posixpath.normpath(path) if path else ""
    if path == ".":
        path = ""
    return path.lstrip("/"), unquote(anchor)


def make_soup(markup: str) -> BeautifulSoup:
    """Parses HTML with lxml, falling back to the pure-Python parser."""
    try:
        return BeautifulSoup(markup, "lxml")
    except Exception:  # bs4.FeatureNotFound, or an lxml build problem
        return BeautifulSoup(markup, "html.parser")

# Inline HTML elements: markup inside a sentence, never a paragraph boundary.
_INLINE_TAGS: tuple[str, ...] = (
    "a", "abbr", "b", "bdi", "bdo", "cite", "code", "del", "dfn", "em", "font",
    "i", "ins", "kbd", "mark", "q", "s", "samp", "small", "span", "strike",
    "strong", "sub", "sup", "tt", "u", "var",
)

# Words that label a single-letter identifier ("Class D", "Plan B",
# "Vitamin C", "Mr. T"). A capital following one is a name, not a kerning split.
_LETTER_LABELS: frozenset[str] = frozenset({
    "class", "room", "section", "level", "floor", "group", "area", "zone",
    "rank", "type", "grade", "exam", "test", "point", "score", "phase",
    "stage", "category", "model", "series", "volume", "chapter", "year",
    "course", "subject", "unit", "part", "item", "step", "plan", "vitamin",
    "option", "team", "block", "wing", "gate", "platform", "appendix",
    "figure", "table", "exhibit", "mr", "mrs", "ms", "dr", "agent",
})

# ══════════════════════════════════════════════════════════════════════════════
# Data structures
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class ChapterItem:
    num:          int
    title:        str
    raw_md:       str            # Docling markdown, before Phase 4
    normalized:   str            # after Phase 4 normalization
    sentences:    list[str]      # after sentence splitter
    method:       str            # "docling" | "beautifulsoup" | "plain_text"
    ir_json:      dict           # Docling document dict (for 0_docling_ir.json)
    xgb_score:    float = 0.0
    probably_matter: bool = False   # front/back matter that was explicitly selected
    href:         str = ""       # where the chapter starts ("file.xhtml#anchor")

@dataclass
class SkippedItem:
    name:      str
    title:     str
    reason:    str
    xgb_score: float = 0.0

@dataclass
class TocEntry:
    title:          str
    href:           str
    classification: str   # "chapter" | "skip"
    anchor:         str = ""       # fragment of the original href, percent-decoded
    is_parent:      bool = False   # a TOC section that has child entries
    depth:          int = 0        # nesting level in the TOC (0 = top)


@dataclass
class HtmlDoc:
    """One HTML file of a book, in reading order."""
    name:   str            # normalised path inside the book
    html:   str            # body markup
    linear: bool = True    # False for spine items marked linear="no"
    is_nav: bool = False   # the EPUB3 navigation document
    item:   Any = None     # ebooklib item, when the book is an EPUB
    epub_types: frozenset[str] = frozenset()   # epub:type values of <body>


@dataclass
class EpubSection:
    """A planned chapter (or flagged front/back matter) of an HTML-based book.

    ``scan()`` lists these and ``extract()`` converts them, so both always
    agree on titles, order and numbering.
    """
    title:      str
    kind:       str                                  # "chapter" | "matter"
    href:       str = ""
    parts:      list[tuple[HtmlDoc, str]] = field(default_factory=list)  # (doc, html fragment)
    word_count: int = 0
    num:        int = 0
    reason:     str = ""                             # why it was classified this way
    position:   int = 0                              # index of the lead document

    @property
    def probably_matter(self) -> bool:
        return self.kind != "chapter"

# ══════════════════════════════════════════════════════════════════════════════
# PHASE 3  —  MLClassifier
# ══════════════════════════════════════════════════════════════════════════════

class MLClassifier:
    """
    Feature-based classifier for EPUB/PDF document items.
    Currently uses a hand-tuned heuristic stub in place of a real XGBoost model.
    Swap predict_is_chapter() body with xgb.Booster inference when
    chapter_classifier.json is available.
    """

    XGB_MODEL_PATH = os.path.join(PROJECT_ROOT, "chapter_classifier.json")

    def __init__(self):
        self._model = None
        self._try_load_model()

    def _try_load_model(self):
        """Attempt to load XGBoost model; silently continue if unavailable."""
        if not os.path.exists(self.XGB_MODEL_PATH):
            return
        try:
            import xgboost as xgb  # type: ignore
            self._model = xgb.Booster()
            self._model.load_model(self.XGB_MODEL_PATH)
            print(f"[MLClassifier] Loaded XGBoost model from {self.XGB_MODEL_PATH}")
        except Exception as e:
            print(f"[MLClassifier] Could not load XGBoost model: {e}")

    # ── Feature extraction from Docling IR ───────────────────────────────────

    def extract_features(self, doc_texts: list, avg_font: float) -> list[dict]:
        """
        Build a feature vector for each text block in the Docling IR.
        Returns one dict per block.
        """
        features = []
        for block in doc_texts:
            text = getattr(block, "text", "") or ""
            font_size = getattr(block, "font_size", None)
            is_bold   = int(bool(getattr(block, "bold", False)))
            label     = str(getattr(block, "label", "")).lower()

            # is_centered: if Docling exposes prov/bbox we can check;
            # otherwise fall back to label heuristic
            is_centered = int("title" in label)

            font_ratio = (font_size / avg_font) if (font_size and avg_font > 0) else 1.0

            features.append({
                "text":             text,
                "word_count":       len(text.split()),
                "font_size_ratio":  round(font_ratio, 3),
                "is_bold":          is_bold,
                "is_centered":      is_centered,
                "has_chapter_keyword": int(bool(_CHAPTER_KW.match(text))),
                "label":            label,
            })
        return features

    def avg_body_font(self, doc_texts: list) -> float:
        """Median font size of all body text blocks (robust to outlier headings)."""
        sizes = [
            getattr(b, "font_size", None)
            for b in doc_texts
            if getattr(b, "font_size", None)
        ]
        return statistics.median(sizes) if sizes else 12.0

    # ── XGBoost inference stub ────────────────────────────────────────────────

    def predict_is_chapter(self, feat: dict) -> float:
        """
        Returns a probability [0..1] that this block is a chapter heading.

        STUB — uses a hand-tuned heuristic score.
        When chapter_classifier.json is trained, replace the body with:

            import xgboost as xgb
            dmat = xgb.DMatrix([[
                feat["word_count"], feat["font_size_ratio"],
                feat["is_bold"], feat["is_centered"], feat["has_chapter_keyword"]
            ]])
            return float(self._model.predict(dmat)[0])
        """
        if self._model:
            try:
                import xgboost as xgb  # type: ignore
                dmat = xgb.DMatrix([[
                    feat["word_count"],
                    feat["font_size_ratio"],
                    feat["is_bold"],
                    feat["is_centered"],
                    feat["has_chapter_keyword"],
                ]])
                return float(self._model.predict(dmat)[0])
            except Exception:
                pass  # fall through to heuristic

        # Heuristic stub
        score = 0.0
        if feat["has_chapter_keyword"]:   score += 0.60
        if feat["font_size_ratio"] > 1.3: score += 0.20
        if feat["is_bold"]:               score += 0.10
        if feat["word_count"] < 15:       score += 0.10
        return min(score, 1.0)

    # ── Document-level classification ─────────────────────────────────────────

    def classify_item(
        self,
        *,
        item_name:      str,
        item_title:     str,
        word_count:     int,
        position_idx:   int,
        chapter_hrefs:  set[str],
        skip_hrefs:     set[str],
        doc_texts:      list,
        avg_font:       float,
    ) -> tuple[str, float]:
        """
        Returns (classification, xgb_score).
        classification: "chapter" | "front_matter" | "back_matter" | "toc" | "gallery" | "skipped"
        """
        # Exact, percent-decoded path equality: a substring test matched
        # "1.xhtml" against the TOC href "11.xhtml".
        name_path = normalize_href(item_name)[0]

        # Priority 1 — TOC says chapter
        if name_path and any(name_path == normalize_href(h)[0] for h in chapter_hrefs):
            return ("chapter", 1.0)

        # Priority 2 — TOC says skip
        if name_path and any(name_path == normalize_href(h)[0] for h in skip_hrefs):
            label = "toc" if "toc" in item_title.lower() or "table" in item_title.lower() \
                else "gallery" if "gallery" in item_title.lower() or "image" in item_title.lower() \
                else "back_matter"
            return (label, 0.0)

        # Priority 3 — Title matches skip pattern
        if _SKIP_TOC_TITLE.match(item_title.strip()):
            return ("front_matter" if position_idx <= 3 else "back_matter", 0.0)

        # Priority 4 — Too short to be a chapter
        if word_count < 80:
            return ("front_matter" if position_idx <= 5 else "back_matter", 0.0)

        # Priority 5 — XGBoost on first text block
        xgb_score = 0.0
        feats = self.extract_features(doc_texts, avg_font)
        if feats:
            xgb_score = self.predict_is_chapter(feats[0])

        # Long enough items default to chapter regardless of score
        return ("chapter", xgb_score)


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 4  —  TextNormalizer
# ══════════════════════════════════════════════════════════════════════════════

class TextNormalizer:
    """
    Cleans Docling markdown output for TTS consumption.
    """

    # Markdown patterns to remove / replace
    _IMG_TAG   = re.compile(r"!\[[^\]]*\]\([^)]*\)")      # ![alt](src)
    # Horizontal rules and scene breaks, contiguous or spaced: ---, ***, * * *
    _HR        = re.compile(r"^[ \t]*(?:[-*_~#\u2022\u00b7][ \t]*){3,}$", re.MULTILINE)
    _HEADING   = re.compile(r"^[ \t]*#{1,6}[ \t]+", re.MULTILINE)
    _HTML_CMT  = re.compile(r"<!--.*?-->", re.DOTALL)
    # Docling escapes these when exporting markdown; "&amp;" must come last.
    _ENTITIES: tuple[tuple[str, str], ...] = (
        ("&lt;", "<"), ("&gt;", ">"), ("&quot;", '"'), ("&#x27;", "'"),
        ("&#39;", "'"), ("&apos;", "'"), ("&nbsp;", " "), ("&amp;", "&"),
    )
    _BOLD_EM   = re.compile(r"\*{1,2}([^*]+?)\*{1,2}|_{1,2}([^_]+?)_{1,2}")
    _MULTI_BL  = re.compile(r"\n{3,}")
    _SOFT_WRAP = re.compile(r"(?<![.!?:;\"'\u2019\u201d])\n(?=[a-z])")
    _HYPHEN_WR = re.compile(r"-\n(\S)")

    # Smart-quote / typography normalisation
    _SMART_Q   = str.maketrans({
        "\u201c": '"', "\u201d": '"',
        "\u2018": "'", "\u2019": "'",
        "\u2014": ", ", "\u2013": "-",
        "\u00a0": " ",
    })

    # ── OCR repair stub ───────────────────────────────────────────────────────

    @staticmethod
    def llm_repair_ocr_block(text: str) -> str:
        """
        Stubs a targeted LLM OCR repair pass.

        ONLY called on blocks whose text was generated by Docling's OCR engine
        (i.e., block came from an embedded image, not native text).

        To activate with a local Qwen2.5-0.5B (or any HuggingFace model):
            from transformers import pipeline
            _pipe = pipeline("text-generation", model="Qwen/Qwen2.5-0.5B-Instruct")
            prompt = (
                "Fix any OCR spelling or grammar errors in this text. "
                "Output ONLY the fixed text.\\n\\n" + text
            )
            result = _pipe(prompt, max_new_tokens=512)[0]["generated_text"]
            return result.split(prompt)[-1].strip()
        """
        return text  # passthrough until model is plugged in

    # Patterns for PDF header/footer noise
    # Garbled OCR from images: no-spaces blobs that look like merged words.
    # Heuristic: >=18 chars, no spaces, and at least three capitalised words
    # run together ("ThisIsGarbledOcrSoup"). An ordinary long word such as
    # "Introduction" or "Acknowledgments" is NOT garbled.
    _GARBLED    = re.compile(r"[A-Z][a-z]+(?:[A-Z][a-z]+){2,}")
    _GARBLED_MIN_LEN: int = 18
    # Running page-header lines: short (≤60 chars) with no sentence-ending punctuation
    _PAGE_NUM   = re.compile(r"^\s*\d{1,4}\s*$", re.MULTILINE)
    # Docling image placeholder lines
    _IMG_BLOCK  = re.compile(r"^<!-- image -->\s*$", re.MULTILINE)

    # ── Individual normalisation steps ───────────────────────────────────────

    _MD_HEADING_PREFIX = re.compile(r"^#+\s*")
    # Sentence-final punctuation, optionally inside closing quotes/brackets.
    _SENTENCE_END = re.compile(r"[.!?:,;…][\"'”’)\]*_]*$")
    _DIALOGUE_START = re.compile(r"^[\"'“‘—–\-«]")
    _STRUCTURAL_LINE = re.compile(r"^(?:#{1,6}\s|[-*+]\s|\d+[.)]\s|\||>)")

    def _is_running_header_candidate(self, stripped: str) -> bool:
        """A short bare line that could be a running header or footer.

        Dialogue (``"What?"``), anything that ends a sentence and markdown
        structure (headings, list items, table rows) never qualifies.
        """
        if not 3 <= len(stripped) <= 60:
            return False
        if self._SENTENCE_END.search(stripped) or self._DIALOGUE_START.match(stripped):
            return False
        return not self._STRUCTURAL_LINE.match(stripped)

    def _strip_pdf_noise(self, text: str) -> str:
        """
        Removes PDF-specific extraction artefacts:
        1. Standalone page numbers.
        2. Docling <!-- image --> placeholder lines.
        3. Garbled OCR from image blocks (CamelCase word-soup with no spaces).
        4. Repeating short lines (running headers/footers): an identical bare
           line of ≤60 chars that appears 3+ times is almost certainly a
           header. The first occurrence is kept, because a one-word chapter
           heading ("Introduction") is usually also the running header of the
           pages that follow it; repeated dialogue is never touched.
        """
        # 1. Page numbers on their own line
        text = self._PAGE_NUM.sub("", text)
        # 2. Docling image placeholders
        text = self._IMG_BLOCK.sub("", text)
        # 3. Garbled OCR blobs — remove lines whose text content (after stripping
        #    any markdown heading markers like ## or ###) is ONLY a CamelCase blob
        lines = text.split("\n")
        cleaned = []
        for line in lines:
            stripped = line.strip()
            # Remove heading markers to get the bare text
            bare = self._MD_HEADING_PREFIX.sub("", stripped).strip()
            # A "garbled" line: bare text has no internal spaces AND matches CamelCase blob
            if len(bare) >= self._GARBLED_MIN_LEN and " " not in bare and self._GARBLED.fullmatch(bare):
                continue
            cleaned.append(line)
        # 4. Detect and remove repeating short header/footer lines
        line_counts = Counter(
            stripped for stripped in (ln.strip() for ln in cleaned)
            if self._is_running_header_candidate(stripped)
        )
        repeating = {ln for ln, cnt in line_counts.items() if cnt >= 3}
        if repeating:
            seen: set[str] = set()
            kept = []
            for line in cleaned:
                stripped = line.strip()
                if stripped in repeating:
                    if stripped in seen:
                        continue
                    seen.add(stripped)
                kept.append(line)
            cleaned = kept
        return "\n".join(cleaned)

    # ── Markdown structure (Docling output only) ─────────────────────────────

    _MD_TABLE_RULE = re.compile(r"^[ \t]*\|?[ \t]*:?-{2,}:?[ \t]*(?:\|[ \t]*:?-{2,}:?[ \t]*)+\|?[ \t]*$")
    _MD_TABLE_ROW  = re.compile(r"^[ \t]*\|(.*)\|[ \t]*$")
    _MD_CELL_SPLIT = re.compile(r"(?<!\\)\|")
    _MD_BULLET     = re.compile(r"^([ \t]*)(?:[-*+•◦▪])[ \t]+(?=\S)")
    _MD_QUOTE      = re.compile(r"^[ \t]*(?:>[ \t]?)+")
    _MD_FENCE      = re.compile(r"^[ \t]*(?:```|~~~)")
    _MD_LINK       = re.compile(r"(?<!!)\[([^\]\n]+)\]\((?:[^()\n]|\([^()\n]*\))*\)")
    _MD_ESCAPE     = re.compile(r"\\([\\`*{}\[\]()#+\-.!>|~])")

    @staticmethod
    def _close_item(text: str) -> str:
        """Ends a list item or table row like a sentence ("Tea, 3 shillings.")."""
        text = text.rstrip()
        return text + "." if text[-1:].isalnum() else text

    def strip_markdown_structure(self, text: str) -> str:
        """Flattens Markdown structure that a TTS voice would read aloud.

        List bullets, block-quote markers, code fences and table pipes are
        removed (a table row becomes its cells joined by commas) and
        ``[text](url)`` links are reduced to their text. List items and table
        rows become paragraphs of their own, closed with a full stop when they
        have no end punctuation, so they are not run together when spoken.
        This is a Python pre-pass for Docling output and Markdown-flavoured
        text files, applied *before* :meth:`normalize`, so the Rust and Python
        normalisers still receive identical input.

        Parameters
        ----------
        text : str
            Markdown as exported by Docling.

        Returns
        -------
        str
            The same text without list, table, quote and link syntax.
        """
        text = self._IMG_TAG.sub("", text)
        text = self._FOOTNOTE_LINK.sub("", text)
        out: list[str] = []
        for line in text.split("\n"):
            if self._MD_FENCE.match(line) or self._MD_TABLE_RULE.match(line):
                continue
            row = self._MD_TABLE_ROW.match(line)
            if row:
                cells = [c.strip() for c in self._MD_CELL_SPLIT.split(row.group(1))]
                out += [self._close_item(", ".join(c for c in cells if c)), ""]
                continue
            if not self._HR.match(line):
                line = self._MD_QUOTE.sub("", line)
                if self._MD_BULLET.match(line):
                    out += [self._close_item(self._MD_BULLET.sub(r"\1", line)), ""]
                    continue
            out.append(line)
        text = "\n".join(out)
        text = self._MD_LINK.sub(r"\1", text)
        return self._MD_ESCAPE.sub(r"\1", text)

    # Patterns for Markdown noise
    _FOOTNOTE_LINK = re.compile(r"\[\[\d+\]\]\([^)]+\)|\[\d+\]\([^)]+\)")
    _OCR_PREFIX    = re.compile(r"OCR_IMG_TEXT:\s*")

    def _strip_noise(self, text: str) -> str:
        # Remove the OCR_IMG_TEXT: prefix we maliciously injected in _preprocess_html
        # MUST happen before _BOLD_EM to prevent _IMG_ from being stripped as an italic tag!
        text = self._OCR_PREFIX.sub("", text)

        text = self._IMG_TAG.sub("", text)              # strip ![...](...)
        text = self._HTML_CMT.sub("", text)             # strip <!-- image --> etc.
        text = self._HR.sub("\n\n", text)               # strip --- / *** / * * *
        text = self._HEADING.sub("", text)              # "## Title" → "Title"
        text = text.replace("\\_", " ")                 # markdown-escaped underscore
        text = self._BOLD_EM.sub(r"\1\2", text)         # strip ** or _
        text = self._FOOTNOTE_LINK.sub("", text)        # strip footnotes like [[1]](#id_C0001)
        for entity, char in self._ENTITIES:
            text = text.replace(entity, char)

        text = text.translate(self._SMART_Q)            # normalise smart quotes
        text = self._MULTI_BL.sub("\n\n", text)         # collapse blank lines
        return text

    # Kerning splits anywhere in a line ("W ar", "T ohsaka") — PDF text only.
    _KERN_MIXED    = re.compile(r"(?<![a-zA-Z])([A-Z])[ \t]+([a-zA-Z]+)\b")
    _KERN_CAPS     = re.compile(r"\b([A-Z])[ \t]+([A-Z]+)\b")
    # Drop caps: a lone capital opening a line, split from the rest of its word.
    _DROPCAP_MIXED = re.compile(r"^([ \t]*(?:#+[ \t]*)?)([A-Z])[ \t]+([a-z]+)\b", re.MULTILINE)
    _DROPCAP_CAPS  = re.compile(r"^([ \t]*(?:#+[ \t]*)?)([A-Z])[ \t]+([A-Z]{2,})\b", re.MULTILINE)
    _LABEL_BEFORE  = re.compile(r"([A-Za-z]+)\.?[ \t]+$")

    @staticmethod
    def _merge_capital(cap: str, rest: str) -> str:
        """Joins a detached capital to the rest of its word, unless "A"/"I" is a real word here."""
        if cap in ("A", "I"):
            if rest.isupper():
                if cap == "A" and rest in ("ND", "S", "T", "RE", "N", "LL", "NY"):
                    return cap + rest
                if cap == "I" and rest in ("T", "S", "F", "N"):
                    return cap + rest
                return f"{cap} {rest}"  # "A THURSDAY", "I WANT"
            rest_lower = rest.lower()
            if cap == "A" and rest_lower in ("nd", "s", "t", "re", "n", "ll", "ny", "lthough", "gain", "nother", "lready", "lways"):
                return cap + rest
            if cap == "I" and rest_lower in ("t", "s", "f", "n", "ll", "nto", "ndeed", "tself"):
                return cap + rest
            return f"{cap} {rest}"  # "I didn't", "A New"
        return cap + rest

    def _fix_isolated_capitals(self, text: str, aggressive: bool = False) -> str:
        """
        Re-joins a single capital letter that extraction detached from its word.

        By default only drop-cap position is repaired — a lone capital opening
        a line ('T he sun' -> 'The sun'). Merging everywhere would corrupt
        ordinary prose ('Vitamin C is' -> 'Vitamin Cis', 'Plan B was' ->
        'Plan Bwas'), so that is reserved for ``aggressive=True``, used for PDF
        text where kerning splits words mid-line ('W ar', 'T ohsaka'). Even
        then a capital that follows a label word ('Class D students') is left
        alone. Never joins across a line break.
        """
        if not aggressive:
            text = self._DROPCAP_MIXED.sub(
                lambda m: m.group(1) + self._merge_capital(m.group(2), m.group(3)), text
            )
            return self._DROPCAP_CAPS.sub(
                lambda m: m.group(1) + self._merge_capital(m.group(2), m.group(3)), text
            )

        def _merge_unless_labelled(match: re.Match) -> str:
            before = match.string[max(0, match.start(1) - 24):match.start(1)]
            label = self._LABEL_BEFORE.search(before)
            if label and label.group(1).lower() in _LETTER_LABELS:
                return match.group(0)
            return self._merge_capital(match.group(1), match.group(2))

        text = self._KERN_MIXED.sub(_merge_unless_labelled, text)
        return self._KERN_CAPS.sub(_merge_unless_labelled, text)

    def _fix_broken_lines(self, text: str, is_pdf: bool = False) -> str:
        text = self._HYPHEN_WR.sub(r"\1", text)         # "conver-\nsion" → "conversion"
        text = self._SOFT_WRAP.sub(" ", text)            # soft-wrap join
        text = self._fix_isolated_capitals(text, aggressive=is_pdf)  # "T he" → "The"
        return text

    def _remove_duplicate_title(self, title: str, text: str) -> str:
        """
        If the chapter title appears verbatim in the first 3 lines (Docling
        often emits it as an h1 AND the EPUB has it as a paragraph), remove
        the duplicate.
        """
        stripped_title = title.strip().lower()
        if not stripped_title:
            # An empty title would "match" every blank line and glue the
            # opening paragraphs together.
            return text
        lines = text.split("\n")
        cleaned = []
        skipped = False
        for i, line in enumerate(lines):
            # Strip markdown heading markers for comparison
            if i >= 4:
                cleaned.extend(lines[i:])
                break
            bare = self._MD_HEADING_PREFIX.sub("", line).strip().lower()
            if bare == stripped_title and cleaned and not skipped:
                skipped = True
                continue  # skip duplicate title line
            cleaned.append(line)
        return "\n".join(cleaned)

    # ── Public API ────────────────────────────────────────────────────────────

    def normalize(self, raw_md: str, title: str, ocr_block_texts: list[str],
                  is_pdf: bool = False, fix_kerning: bool = False) -> str:
        """
        Full normalization pipeline.
        ocr_block_texts: list of text strings extracted by OCR (from Docling IR).
        is_pdf: if True, also run _strip_pdf_noise() to clean running headers/footers
                and repair kerning-split words anywhere in a line.
        fix_kerning: repair kerning-split words without the PDF noise stripping
                (for PDF text that did not come through Docling).

        Both PDF steps run here in Python, before the shared pipeline, so the
        Rust and Python back ends see the same already-repaired text and
        return identical output.
        """
        text = raw_md

        # 1. LLM OCR repair — targeted, only on OCR blocks
        for ocr_txt in ocr_block_texts:
            if ocr_txt and ocr_txt in text:
                repaired = self.llm_repair_ocr_block(ocr_txt)
                if repaired != ocr_txt:
                    text = text.replace(ocr_txt, repaired, 1)

        # 2. PDF-specific noise (headers, footers, garbled OCR images). The
        #    Rust clean_text() has its own copy of this step that deletes any
        #    short line repeated 3+ times (dialogue, one-word headings), so it
        #    is always done here and Rust is called with is_pdf=False.
        if is_pdf:
            text = self._strip_pdf_noise(text)
        if is_pdf or fix_kerning:
            text = self._fix_isolated_capitals(text, aggressive=True)

        # Try Rust compiled clean pipeline first for speed
        try:
            import audiobook_rust
            if hasattr(audiobook_rust, "clean_text"):
                return audiobook_rust.clean_text(text, title, False)
        except ImportError:
            pass

        # 3. Remove duplicate title heading
        text = self._remove_duplicate_title(title, text)

        # 4. Fix broken lines before noise strip to avoid stripping mid-word
        text = self._fix_broken_lines(text, is_pdf=False)

        # 5. Markdown noise strip
        text = self._strip_noise(text)

        # 6. Legacy audiobook pipeline normalisation (de-wrap, drop-cap)
        text = normalize_text(text)

        return text.strip()

    def split_sentences(self, text: str) -> list[str]:
        return smart_sentence_splitter(text, MAX_SENTENCE_LEN)


# ── HTML fragment inspection (shared by scan and extract) ────────────────────

_TAG_WITH_ID   = re.compile(r"<[a-zA-Z][^<>]*>")
_ID_ATTR       = re.compile(r"\s(?:xml:id|id|name)\s*=\s*([\"'])(.*?)\1", re.S)
_EPUB_TYPE     = re.compile(r"epub:type\s*=\s*[\"']([^\"']+)[\"']")
_BODY_TAG      = re.compile(r"<body\b[^>]*>", re.I)
_BODY_CONTENT  = re.compile(r"<body\b[^>]*>(.*)</body\s*>", re.I | re.S)
_ANY_TAG       = re.compile(r"<[^>]+>")
_HEADING_TAG   = re.compile(r"<h([1-3])\b[^>]*>(.*?)</h\1\s*>", re.I | re.S)
_SYNTH_ANCHOR: str = "abm-split-{}"
_HAS_ALNUM     = re.compile(r"[^\W_]")
# Opening tags that may wrap an anchor: the split point moves in front of them
# so a heading keeps its own <h2>/<div> instead of leaving it behind.
_OPENERS_BEFORE = re.compile(
    r"(?:<(?:div|section|article|header|hgroup|h[1-6]|p|span|a|b|i|em|strong|center|blockquote|font)\b[^<>]*>\s*)+$",
    re.I,
)
_NOTE_MARKER   = re.compile(r"^[\[(]?(?:\d{1,3}|[*†‡§]+|[a-z])[\])]?$")
_NOTE_SYMBOLS: tuple[str, ...] = ("[", "(", "*", "†", "‡", "§")
_NOTE_TYPES: frozenset[str] = frozenset({"footnote", "endnote", "rearnote", "footnotes", "endnotes", "rearnotes"})
_BLOCK_TAGS: tuple[str, ...] = (
    "p", "div", "table", "ul", "ol", "dl", "blockquote", "pre", "section",
    "h1", "h2", "h3", "h4", "h5", "h6",
)
_OCR_TOKEN_FMT: str = "OCRIMGTOKEN{:05d}X"
_OCR_GPU_ENV: str = "AUDIOBOOK_OCR_GPU"
_OCR_MIN_IMAGE_SIDE: int = 48


@dataclass
class _FragmentInfo:
    text:         str
    words:        int
    heading:      str               # first h1–h3 when it opens the fragment
    first_line:   str
    epub_types:   frozenset[str]
    link_ratio:   float
    has_images:   bool


def _collapse(text: str) -> str:
    return " ".join((text or "").split())


def fragment_info(markup: str) -> _FragmentInfo:
    """Parses an HTML fragment once and returns what classification needs."""
    epub_types = frozenset(
        t for m in _EPUB_TYPE.finditer(markup[:4000]) for t in m.group(1).lower().split()
    )
    pieces: list[str] = []
    heading = ""
    first_line = ""
    link_chars = 0
    has_images = False
    try:
        from lxml import html as lxml_html
        root = lxml_html.fragment_fromstring(markup, create_parent="div")
        for junk in root.xpath(".//script|.//style|.//head|.//title"):
            junk.drop_tree()
        pieces = [t.strip() for t in root.itertext() if t and t.strip()]
        headings = root.xpath(".//h1|.//h2|.//h3")
        if headings:
            heading = _collapse(headings[0].text_content())
        link_chars = sum(len(_collapse(a.text_content())) for a in root.xpath(".//a[@href]"))
        has_images = bool(root.xpath(".//img|.//image|.//svg"))
        for block in root.iter("p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "td"):
            first_line = _collapse(block.text_content())
            if first_line:
                break
    except Exception:
        pieces = [p.strip() for p in _ANY_TAG.sub("\n", markup).split("\n") if p.strip()]
        pieces = [html_lib.unescape(p) for p in pieces]
    if not first_line and pieces:
        first_line = pieces[0]
    text = "\n".join(pieces)
    words = len(_WORD.findall(text))
    if heading:
        # Only a heading that opens the fragment titles it.
        probe = heading.split(" ")[0]
        before = text.split(probe, 1)[0] if probe and probe in text else text
        if len(before.split()) > 12:
            heading = ""
    if len(first_line) > 80 or first_line[-1:] in ".,;:!?":
        first_line = ""
    return _FragmentInfo(
        text=text, words=words, heading=heading, first_line=_collapse(first_line),
        epub_types=epub_types, link_ratio=link_chars / max(1, len(text)),
        has_images=has_images,
    )


def select_sections(sections: Iterable[Any], selections: list[int] | list[str] | None) -> list[Any]:
    """Applies a chapter selection to planned sections.

    Parameters
    ----------
    sections : Iterable
        Objects with ``title``, ``num`` and ``probably_matter`` attributes.
    selections : list[int] | list[str] | None
        ``None``/empty selects every chapter that is not flagged as
        front/back matter. A list of titles (as shown by ``scan()``) or of
        chapter numbers selects exactly those entries — including flagged
        ones, which are only ever extracted when asked for by name or number.

    Returns
    -------
    list
        The selected sections, in reading order.
    """
    sections = list(sections)
    if not selections:
        return [s for s in sections if not s.probably_matter]
    if isinstance(selections, (str, int)):
        selections = [selections]  # type: ignore[list-item]
    wanted_titles = {_collapse(s).casefold() for s in selections if isinstance(s, str)}
    wanted_nums = {s for s in selections if isinstance(s, int) and not isinstance(s, bool)}
    return [
        s for s in sections
        if s.num in wanted_nums or _collapse(s.title).casefold() in wanted_titles
    ]


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 & 2  —  DocumentIngestor
# ══════════════════════════════════════════════════════════════════════════════

class DocumentIngestor:
    """
    Phase 1: TOC extraction + drop-cap pre-processing.
    Phase 2: Docling ingestion with explicit format routing + OCR.
    """

    # EasyOCR reader shared by one extraction run and released afterwards, so
    # it never sits in memory (or VRAM) next to the TTS model.
    _easyocr_reader: Any = None
    _easyocr_langs: tuple[str, ...] = ()
    _easyocr_failed: set[tuple[str, ...]] = set()

    def __init__(self):
        # Docling loads layout models when a converter is built, so that only
        # happens the first time a document is actually converted.
        self._docling_converter = None
        self._pdf_converter = None   # kept for backward compatibility; never used

    @property
    def _converter(self):
        if not DOCLING_AVAILABLE:
            return None
        if getattr(self, "_docling_converter", None) is None:
            self._docling_converter = DocumentConverter()
        return self._docling_converter

    @_converter.setter
    def _converter(self, value) -> None:
        self._docling_converter = value

    # ── Phase 1a: TOC extraction ──────────────────────────────────────────────

    def _walk_epub_toc(
        self, toc_items, base_dir: str = "", depth: int = 0,
    ) -> tuple[set[str], set[str], list[TocEntry]]:
        """Recursively walk epub TOC; returns (chapter_hrefs, skip_hrefs, entries).

        Hrefs are percent-decoded, normalised paths (see :func:`normalize_href`);
        the fragment of each entry is kept in ``TocEntry.anchor``.
        """
        chapter_hrefs: set[str] = set()
        skip_hrefs:    set[str] = set()
        entries:       list[TocEntry] = []

        if toc_items is None:
            return chapter_hrefs, skip_hrefs, entries
        if not isinstance(toc_items, (list, tuple)):
            # ebooklib returns a bare Link for an NCX whose navMap is empty.
            toc_items = [toc_items]

        def _add(title: str, raw_href: str, is_parent: bool) -> None:
            path, anchor = normalize_href(raw_href, base_dir)
            if not path:
                return
            title = _collapse(title)
            cls = "skip" if _SKIP_TOC_TITLE.match(title) else "chapter"
            (skip_hrefs if cls == "skip" else chapter_hrefs).add(path)
            entries.append(TocEntry(title, path, cls, anchor=anchor, is_parent=is_parent, depth=depth))

        for item in toc_items:
            if isinstance(item, tuple) and len(item) == 2 and isinstance(item[1], (list, tuple)):
                section, children = item
                _add(getattr(section, "title", "") or "", getattr(section, "href", "") or "", True)
                ch, sk, en = self._walk_epub_toc(children, base_dir, depth + 1)
                entries.extend(en)
                chapter_hrefs |= ch
                skip_hrefs    |= sk
                continue
            href = getattr(item, "href", None)
            if href is None and hasattr(item, "get_name"):
                href = item.get_name()     # an EpubHtml placed directly in the TOC
            _add(getattr(item, "title", "") or "", href or "", False)

        return chapter_hrefs, skip_hrefs, entries

    def _extract_pdf_toc(self, pdf_path: str) -> tuple[set[int], list[TocEntry]]:
        """Returns (chapter_pages, entries) from a PDF TOC via PyMuPDF."""
        chapter_pages: set[int] = set()
        entries: list[TocEntry] = []
        if not PYMUPDF_AVAILABLE:
            return chapter_pages, entries
        doc = None
        try:
            doc = fitz.open(pdf_path)
            for level, title, page in doc.get_toc():
                cls = "skip" if _SKIP_TOC_TITLE.match((title or "").strip()) else "chapter"
                if cls == "chapter":
                    chapter_pages.add(page)
                entries.append(TocEntry(title or "", str(page), cls))
        except Exception as e:
            print(f"    [PDF TOC] extraction failed: {e}")
        finally:
            if doc is not None:
                doc.close()
        return chapter_pages, entries

    # ── OCR reader lifecycle ──────────────────────────────────────────────────

    @staticmethod
    def ocr_languages(language: str | None) -> list[str]:
        """Maps a book language tag ("fr", "zh-TW", "en-GB") to EasyOCR codes."""
        code = (language or "en").strip().lower().replace("_", "-")
        if code.startswith(("zh-tw", "zh-hk", "zh-mo", "zh-hant")):
            return ["ch_tra", "en"]
        primary = code.split("-")[0] or "en"
        mapped = {"zh": "ch_sim", "nb": "no", "nn": "no", "iw": "he", "in": "id"}.get(primary, primary)
        return ["en"] if mapped == "en" else [mapped, "en"]

    @classmethod
    def get_ocr_reader(cls, language: str | None = None):
        """Returns a (cached) EasyOCR reader for the book language, or None.

        The reader runs on the CPU unless the ``AUDIOBOOK_OCR_GPU`` environment
        variable is set to a true value, so it does not compete with the TTS
        model for VRAM. Call :meth:`release_ocr_reader` when extraction ends.
        """
        try:
            import easyocr  # type: ignore
        except ImportError:
            return None
        langs = tuple(cls.ocr_languages(language))
        if cls._easyocr_reader is not None and cls._easyocr_langs == langs:
            return cls._easyocr_reader
        if cls._easyocr_reader is not None:
            cls.release_ocr_reader(keep_failures=True)
        use_gpu = os.environ.get(_OCR_GPU_ENV, "").strip().lower() in ("1", "true", "yes", "on")
        for attempt in dict.fromkeys((langs, ("en",))):
            if attempt in cls._easyocr_failed:
                continue
            try:
                cls._easyocr_reader = easyocr.Reader(list(attempt), gpu=use_gpu)
                cls._easyocr_langs = langs
                return cls._easyocr_reader
            except Exception as reader_err:
                # Unsupported language, no network for the model download,
                # CUDA OOM, … — image text is optional, the chapter text is not.
                cls._easyocr_failed.add(attempt)
                logger.warning("EasyOCR unavailable for %s (%s).", list(attempt), reader_err)
        return None

    @classmethod
    def release_ocr_reader(cls, keep_failures: bool = False) -> None:
        """Drops the cached EasyOCR reader and frees the memory it held."""
        reader = cls._easyocr_reader
        cls._easyocr_reader = None
        cls._easyocr_langs = ()
        if not keep_failures:
            cls._easyocr_failed = set()
        if reader is None:
            return
        del reader
        gc.collect()
        torch = sys.modules.get("torch")
        try:
            if torch is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as exc:
            logger.debug("Could not empty the CUDA cache after OCR: %s", exc)

    # ── Phase 1b: HTML pre-processing ─────────────────────────────────────────

    @classmethod
    def _preprocess_soup(
        cls,
        html_content: str,
        epub_book=None,
        epub_item_name: str = "",
        language: str | None = None,
    ) -> tuple[BeautifulSoup, dict[str, str]]:
        """Parses and cleans one HTML file; returns (soup, ocr_tokens).

        1. Unwraps drop-cap <span>s.
        2. Clamps table spans (Docling OOM guard).
        3. Removes scripts, styles, footnote markers and footnote bodies.
        4. OCRs embedded images (only when ``epub_book`` is given). Each image
           with text is replaced, *in place*, by a placeholder token; the
           returned dict maps token -> ``"OCR_IMG_TEXT: <text>"``.
        """
        soup = make_soup(html_content)
        tokens: dict[str, str] = {}

        for junk in soup.find_all(["script", "style", "template", "noscript"]):
            junk.decompose()

        for span in soup.find_all("span"):
            classes = " ".join(span.get("class", []))
            style   = span.get("style", "")
            text    = span.get_text(strip=True)
            if len(text) == 1 and (
                "drop" in classes.lower() or
                ("font-size" in style and "em" in style)
            ):
                # Heuristic: If it has a previous sibling that isn't just whitespace,
                # it's likely an identifier rather than a paragraph drop-cap (There).
                is_middle = False
                prev = span.previous_sibling
                if prev:
                    prev_txt = prev.get_text() if hasattr(prev, "get_text") else str(prev)
                    if prev_txt.strip():
                        is_middle = True

                # If middle and followed by lowercase, ensure space.
                nxt = span.next_sibling
                if is_middle and nxt and isinstance(nxt, str):
                    if nxt.lstrip() and nxt.lstrip()[0].islower():
                        # Prepend space to the next text node
                        span.next_sibling.replace_with(" " + nxt.lstrip())

                span.unwrap()   # merge single-char drop-cap with following word

        # ── Clamp table colspan and rowspan to prevent Docling OOM (BUG-R2-C3-A4-H1) ──
        for cell in soup.find_all(["td", "th"]):
            for attr in ("colspan", "rowspan"):
                val = cell.get(attr)
                if val:
                    try:
                        num = int(val)
                        if num > _MAX_TABLE_SPAN:
                            cell[attr] = str(_MAX_TABLE_SPAN)
                    except (ValueError, TypeError):
                        pass

        # ── Footnotes: a marker read aloud is a stray number mid-sentence ──
        doomed = [
            note for note in soup.find_all(attrs={"epub:type": True})
            if set(str(note.get("epub:type")).lower().split()) & _NOTE_TYPES
        ]
        for ref in soup.find_all("a"):
            kind = str(ref.get("epub:type") or "").lower()
            label = ref.get_text(strip=True)
            in_sup = ref.parent is not None and ref.parent.name == "sup"
            if "noteref" in kind or (
                _NOTE_MARKER.match(label) and (in_sup or ref.find("sup") is not None)
            ):
                doomed.append(ref.parent if in_sup and ref.parent.get_text(strip=True) == label else ref)
        for sup in soup.find_all("sup"):
            label = sup.get_text(strip=True)
            if label[:1] in _NOTE_SYMBOLS and _NOTE_MARKER.match(label):
                doomed.append(sup)
        for tag in doomed:
            if not getattr(tag, "decomposed", False):
                tag.decompose()

        # ── In-flight EPUB Image OCR ──
        if epub_book is not None:
            images = soup.find_all(["img", "image"])
            reader = cls.get_ocr_reader(language) if images else None
            if reader is not None:
                base = posixpath.dirname(epub_item_name)
                for img in images:
                    src = img.get("src") or img.get("xlink:href") or img.get("href")
                    if not src or src.startswith("data:"):
                        continue
                    full_src = normalize_href(src, base)[0]
                    img_item = epub_book.get_item_with_href(full_src) or epub_book.get_item_with_href(src)
                    if not img_item:
                        continue
                    try:
                        import io
                        from PIL import Image
                        import numpy as np

                        img_obj = Image.open(io.BytesIO(img_item.get_content()))
                        if min(img_obj.size) < _OCR_MIN_IMAGE_SIDE:
                            continue   # ornaments and scene-break glyphs
                        # Convert to RGB if it's not (e.g. RGBA or Grayscale) to prevent easyocr/cv2 errors
                        if img_obj.mode != "RGB":
                            img_obj = img_obj.convert("RGB")
                        res = reader.readtext(np.array(img_obj))
                        extracted_text = " ".join(line[1] for line in res).strip() if res else ""
                        if extracted_text:
                            # The token survives Docling untouched (plain letters and
                            # digits) and is swapped for the text afterwards, so image
                            # text stays where the image was instead of at the chapter end.
                            token = _OCR_TOKEN_FMT.format(len(tokens))
                            tokens[token] = f"OCR_IMG_TEXT: {extracted_text}"
                            holder = img.parent if img.parent is not None and img.parent.name == "svg" else img
                            holder.replace_with(NavigableString(f" {token} "))
                    except Exception as e:
                        logger.warning("OCR failed on image %s: %s", src, e)

        return soup, tokens

    @staticmethod
    def _preprocess_html(html_content: str, epub_book=None, epub_item_name: str = "") -> tuple[str, list[str]]:
        """
        Cleans EPUB HTML before Docling sees it:
        1. Unwraps drop-cap <span>s.
        2. Runs OCR on embedded images; the recognised text replaces each image in place.
        Returns: (processed_html_string, list_of_extracted_ocr_texts)
        """
        soup, tokens = DocumentIngestor._preprocess_soup(html_content, epub_book, epub_item_name)
        processed = str(soup)
        for token, ocr_text in tokens.items():
            processed = processed.replace(token, html_lib.escape(ocr_text, quote=False))
        return processed, list(tokens.values())

    @staticmethod
    def _place_ocr_text(text: str, tokens: dict[str, str]) -> str:
        """Swaps OCR placeholder tokens for their text; leftovers go to the end."""
        leftovers = []
        for token, ocr_text in tokens.items():
            if token in text:
                text = text.replace(token, f"\n\n{ocr_text}\n\n")
            else:
                leftovers.append(ocr_text)
        if leftovers:
            text += "\n\n" + "\n\n".join(leftovers) + "\n\n"
        return text

    # ── Phase 2: Docling ingestion ────────────────────────────────────────────

    @staticmethod
    def _export_markdown(doc) -> str:
        """Exports a Docling document without HTML escaping or image placeholders.

        Underscores stay escaped on purpose: both normalisers turn ``\\_`` into
        a space, while a bare ``snake_case_name`` would be read as italics.
        """
        export = doc.export_to_markdown
        try:
            params = inspect.signature(export).parameters
        except (TypeError, ValueError):
            params = {}
        kwargs = {
            name: value
            for name, value in (("escape_html", False), ("image_placeholder", ""))
            if name in params
        }
        try:
            return export(**kwargs)
        except TypeError:
            return export()   # older docling-core

    @staticmethod
    def _docling_ocr_blocks(doc) -> list[str]:
        """Text blocks Docling produced by OCR (for the Phase 4 repair hook)."""
        ocr_texts = []
        for block in getattr(doc, "texts", []):
            prov = getattr(block, "prov", [])
            for p in (prov if isinstance(prov, list) else [prov]):
                if getattr(p, "charspan", None) == (0, 0):
                    # zero charspan means Docling had no native text → OCR
                    ocr_texts.append(getattr(block, "text", ""))
        return ocr_texts

    def _docling_convert(self, processed_html: str, tokens: dict[str, str]) -> tuple[str, dict, list[str]]:
        """Runs Docling on already pre-processed HTML."""
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                suffix=".html", mode="w", encoding="utf-8", delete=False, dir=str(_TEMP_DIR)
            ) as tmp:
                tmp.write(processed_html)
                tmp_path = tmp.name

            result  = self._converter.convert(tmp_path)
            doc     = result.document
            raw_md  = self._place_ocr_text(self._export_markdown(doc), tokens)

            try:
                ir_dict = doc.export_to_dict()
            except Exception:
                ir_dict = {}

            return raw_md, ir_dict, self._docling_ocr_blocks(doc)

        except Exception as e:
            logger.warning("Docling HTML conversion failed, using BeautifulSoup instead: %s", e)
            return "", {}, []
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

    def _docling_html(self, html_content: str, epub_book=None, epub_item_name: str = "") -> tuple[str, dict, list[str]]:
        """
        Run Docling on preprocessed HTML.
        Returns (raw_markdown, ir_dict, ocr_block_texts).
        """
        soup, tokens = self._preprocess_soup(html_content, epub_book, epub_item_name)
        return self._docling_convert(str(soup), tokens)

    @classmethod
    def _soup_to_text(cls, soup: BeautifulSoup, tokens: dict[str, str] | None = None) -> str:
        """Plain text of a pre-processed soup, one paragraph per block element."""
        # A data table row reads best as one line ("Tea, 3 shillings").
        for row in soup.find_all("tr"):
            cells = row.find_all(["td", "th"], recursive=False)
            if len(cells) < 2 or any(cell.find(_BLOCK_TAGS) is not None for cell in cells):
                continue
            texts = [cell.get_text(" ", strip=True) for cell in cells]
            if any(len(t) > 200 for t in texts):
                continue
            line = soup.new_tag("p")
            line.string = ", ".join(t for t in texts if t)
            row.replace_with(line)
        # get_text() puts the separator around *every* tag, so "<i>Titanic</i>"
        # mid-sentence would become its own paragraph (and its own TTS chunk).
        # Dissolve inline markup first so only block boundaries split.
        for inline in soup.find_all(_INLINE_TAGS):
            inline.unwrap()
        soup.smooth()
        text = soup.get_text(separator="\n\n", strip=True)
        return cls._place_ocr_text(text, tokens or {})

    def _bs_fallback(self, html_content: str, epub_book=None, epub_item_name: str = "") -> str:
        """BeautifulSoup plain-text fallback."""
        soup, tokens = self._preprocess_soup(html_content, epub_book, epub_item_name)
        return self._soup_to_text(soup, tokens)

    # ── Chapter planning ──────────────────────────────────────────────────────

    @staticmethod
    def _anchor_positions(markup: str) -> dict[str, int]:
        """Offset of the first tag carrying each id/name in an HTML string."""
        positions: dict[str, int] = {}
        for tag in _TAG_WITH_ID.finditer(markup):
            body = tag.group(0)
            if "id" not in body and "name" not in body:
                continue
            for attr in _ID_ATTR.finditer(body):
                positions.setdefault(html_lib.unescape(attr.group(2)), tag.start())
        return positions

    def _split_at_anchors(self, doc: HtmlDoc, targets: list[TocEntry]) -> list[tuple[TocEntry | None, str]]:
        """Cuts one HTML file at the anchors its TOC entries point to.

        Returns (entry, fragment) pairs in document order. Text in front of
        the first anchor comes back with ``entry=None``. When no anchor can be
        located the whole file is returned under the FIRST entry.
        """
        markup = doc.html
        positions: dict[str, int] | None = None
        located: list[tuple[int, TocEntry]] = []
        for target in targets:
            if not target.anchor:
                located.append((0, target))
                continue
            if positions is None:
                positions = self._anchor_positions(markup)
            pos = positions.get(target.anchor, -1)
            if pos < 0:
                continue
            opener = _OPENERS_BEFORE.search(markup[max(0, pos - 600):pos])
            if opener:
                pos -= len(markup[max(0, pos - 600):pos]) - opener.start()
            located.append((pos, target))
        if not located:
            return [(targets[0], markup)]

        located.sort(key=lambda pair: pair[0])
        cuts: list[tuple[int, TocEntry]] = []
        for pos, target in located:
            if cuts and cuts[-1][0] == pos:
                if cuts[-1][1].is_parent and not target.is_parent:
                    cuts[-1] = (pos, target)      # "Part One" + "Chapter 1" at one spot
                continue
            cuts.append((pos, target))

        segments: list[tuple[TocEntry | None, str]] = []
        first_pos = cuts[0][0]
        if first_pos > 0:
            lead = markup[:first_pos]
            if _HAS_ALNUM.search(_ANY_TAG.sub("", lead)):
                segments.append((None, lead))
            else:
                cuts[0] = (0, cuts[0][1])
        for i, (pos, target) in enumerate(cuts):
            end = cuts[i + 1][0] if i + 1 < len(cuts) else len(markup)
            segments.append((target, markup[pos:end]))
        return segments

    @staticmethod
    def _heading_targets(doc: HtmlDoc) -> tuple[HtmlDoc, list[TocEntry]] | None:
        """For a book WITHOUT a TOC: finds the headings that divide one file.

        A file holding several chapters is cut at its most senior heading
        level that occurs at least twice. Returns the file with split anchors
        inserted plus one entry per heading, or None when there is nothing to
        split.
        """
        found = [
            (m.start(), int(m.group(1)), _collapse(html_lib.unescape(_ANY_TAG.sub(" ", m.group(2)))))
            for m in _HEADING_TAG.finditer(doc.html)
        ]
        found = [hit for hit in found if hit[2]]
        counts = Counter(level for _, level, _ in found)
        level = next((lv for lv in sorted(counts) if counts[lv] >= 2), 0)
        if not level:
            return None
        chosen = [hit for hit in found if hit[1] == level]
        markup = doc.html
        entries: list[TocEntry] = []
        for n in range(len(chosen) - 1, -1, -1):
            pos, _, title = chosen[n]
            anchor = _SYNTH_ANCHOR.format(n)
            markup = f'{markup[:pos]}<a id="{anchor}"></a>{markup[pos:]}'
            cls = "skip" if _SKIP_TOC_TITLE.match(title) else "chapter"
            entries.append(TocEntry(title, doc.name, cls, anchor=anchor))
        entries.reverse()
        split_doc = HtmlDoc(name=doc.name, html=markup, linear=doc.linear, is_nav=doc.is_nav,
                            item=doc.item, epub_types=doc.epub_types)
        return split_doc, entries

    @staticmethod
    def _unlisted_verdict(doc: HtmlDoc, info: _FragmentInfo) -> str:
        """Classifies a file no TOC entry points at: "matter", "heading" or "plain"."""
        if (info.epub_types | doc.epub_types) & _MATTER_EPUB_TYPES:
            return "matter"
        title = info.heading or info.first_line
        if title and _SKIP_TOC_TITLE.match(title):
            return "matter"
        if looks_like_matter(info.text):
            return "matter"
        if info.link_ratio > 0.6 and info.words >= 4:
            return "matter"                      # a linked contents page
        if info.heading and _STRICT_CHAPTER_HEADING.match(info.heading):
            return "heading"
        return "plain"

    def plan_html_sections(
        self,
        docs: list[HtmlDoc],
        toc_entries: list[TocEntry],
        classifier: MLClassifier | None = None,
    ) -> tuple[list[EpubSection], list[SkippedItem]]:
        """Maps a book's HTML files and TOC onto an ordered list of chapters.

        Parameters
        ----------
        docs : list[HtmlDoc]
            The book's HTML files in READING order.
        toc_entries : list[TocEntry]
            Flattened TOC (see :meth:`_walk_epub_toc`); may be empty.
        classifier : MLClassifier | None
            Classifier for files the TOC does not list.

        Returns
        -------
        tuple[list[EpubSection], list[SkippedItem]]
            Sections in reading order — chapters numbered 1..N, flagged
            front/back matter numbered after them — and the items dropped.
        """
        classifier = classifier or MLClassifier()
        skipped: list[SkippedItem] = []

        by_path: dict[str, int] = {}
        by_base: dict[str, list[int]] = {}
        for idx, doc in enumerate(docs):
            by_path.setdefault(doc.name, idx)
            by_base.setdefault(posixpath.basename(doc.name), []).append(idx)

        targets: dict[int, list[TocEntry]] = {}
        for entry in toc_entries:
            idx = by_path.get(entry.href)
            if idx is None:
                same_name = by_base.get(posixpath.basename(entry.href), [])
                idx = same_name[0] if len(same_name) == 1 else None
            if idx is None:
                skipped.append(SkippedItem(entry.href, entry.title, "TOC entry points at a file that is not in the book"))
                continue
            listed = targets.setdefault(idx, [])
            twin = next((t for t in listed if t.anchor == entry.anchor), None)
            if twin is not None:
                if twin.is_parent and not entry.is_parent:
                    # A section and its first child share a target: the child names it.
                    twin.title, twin.classification, twin.is_parent = entry.title, entry.classification, False
                continue
            if listed and entry.depth >= 2:
                continue   # sub-sections of a chapter that is already listed
            listed.append(TocEntry(entry.title, entry.href, entry.classification,
                                   anchor=entry.anchor, is_parent=entry.is_parent, depth=entry.depth))

        chapter_docs = [i for i, lst in targets.items() if any(t.classification == "chapter" for t in lst)]
        has_toc = bool(chapter_docs)
        last_chapter_doc = max(chapter_docs) if chapter_docs else -1
        readable = [d for d in docs if d.linear and not d.is_nav]
        if not has_toc and len(readable) <= 2:
            # No usable TOC and (nearly) the whole book in one file: cut that
            # file at its headings. With one file per chapter the files
            # themselves are the chapters and sub-headings stay inside them.
            docs = list(docs)
            for idx, doc in enumerate(docs):
                if idx in targets or doc.is_nav or not doc.linear:
                    continue
                split = self._heading_targets(doc)
                if split is not None:
                    docs[idx], targets[idx] = split

        sections: list[EpubSection] = []
        chars: dict[int, int] = {}          # id(section) -> text length
        images: dict[int, bool] = {}
        current: EpubSection | None = None
        tail_closed = False                 # back matter seen after the last listed chapter

        def _open(kind: str, title: str, doc: HtmlDoc, idx: int, frag: str,
                  info: _FragmentInfo, reason: str, anchor: str = "") -> EpubSection:
            section = EpubSection(
                title=title, kind=kind, href=doc.name + (f"#{anchor}" if anchor else ""),
                parts=[(doc, frag)], word_count=info.words, reason=reason, position=idx,
            )
            chars[id(section)] = len(info.text)
            images[id(section)] = info.has_images
            sections.append(section)
            return section

        for idx, doc in enumerate(docs):
            doc_targets = targets.get(idx)
            segments = self._split_at_anchors(doc, doc_targets) if doc_targets else [(None, doc.html)]
            for target, frag in segments:
                info = fragment_info(frag)
                if target is not None:
                    if target.classification == "chapter":
                        current = _open("chapter", target.title or info.heading, doc, idx, frag, info,
                                        "Listed in the table of contents", target.anchor)
                    else:
                        _open("matter", target.title, doc, idx, frag, info,
                              "Explicitly skipped (TOC)", target.anchor)
                        current = None
                        tail_closed = tail_closed or idx >= last_chapter_doc
                    continue

                # ── A file (or leading fragment) the TOC does not list ──
                if doc.is_nav or not doc.linear:
                    skipped.append(SkippedItem(doc.name, info.heading or doc.name,
                                               "Navigation or non-linear document"))
                    continue
                verdict = self._unlisted_verdict(doc, info)
                title = info.heading or info.first_line
                before_story = not any(s.kind == "chapter" for s in sections)
                matter_title = title or (
                    "Copyright" if looks_like_matter(info.text)
                    else "Front Matter" if before_story else "Back Matter"
                )
                if verdict == "matter":
                    _open("matter", matter_title, doc, idx, frag, info,
                          f"Front/back matter (words={info.words})")
                    current = None
                    tail_closed = tail_closed or (has_toc and idx > last_chapter_doc)
                    continue
                starts_new = (verdict == "heading") if has_toc else bool(info.heading)
                if current is not None and not starts_new:
                    # Continuation of a chapter that is split across files.
                    current.parts.append((doc, frag))
                    current.word_count += info.words
                    chars[id(current)] += len(info.text)
                    images[id(current)] = images[id(current)] or info.has_images
                    continue
                if current is None and has_toc and tail_closed and idx > last_chapter_doc:
                    _open("matter", matter_title, doc, idx, frag, info,
                          "Unlisted file after the back matter")
                    continue
                label, _score = classifier.classify_item(
                    item_name=doc.name, item_title=title, word_count=info.words,
                    position_idx=idx, chapter_hrefs=set(), skip_hrefs=set(),
                    doc_texts=[], avg_font=12.0,
                )
                if verdict == "heading":
                    label = "chapter"    # "Chapter 7" stays a chapter however short
                substantial = info.words >= _MIN_UNLISTED_NARRATIVE_WORDS or not has_toc or starts_new
                if label == "chapter" and substantial:
                    fallback = "" if not has_toc else ("Opening" if not any(
                        s.kind == "chapter" for s in sections) else "Untitled Section")
                    current = _open("chapter", info.heading or (info.first_line if has_toc else "") or fallback,
                                    doc, idx, frag, info, "Unlisted narrative file")
                else:
                    _open("matter", matter_title, doc, idx, frag, info,
                          f"Front/back matter (words={info.words})")
                    current = None

        # ── Tidy up: fold or drop fragments too small to be a chapter ──
        kept: list[EpubSection] = []
        pending: EpubSection | None = None
        for section in sections:
            if pending is not None:
                if section.kind == "chapter" and section.parts[0][0] is pending.parts[-1][0]:
                    # A heading-only fragment ("Part One") in front of a chapter
                    # in the same file: keep its text with that chapter.
                    section.parts = pending.parts + section.parts
                    section.word_count += pending.word_count
                    chars[id(section)] += chars[id(pending)]
                else:
                    skipped.append(SkippedItem(pending.href, pending.title, "Text too short (<50 chars)"))
                pending = None
            if section.kind == "chapter" and chars[id(section)] < _MIN_CHAPTER_CHARS:
                if images[id(section)]:
                    section.kind = "matter"
                    section.reason = "Image-only section (needs OCR to yield text)"
                else:
                    pending = section
                    continue
            if section.kind == "matter" and section.word_count == 0 and not images[id(section)]:
                skipped.append(SkippedItem(section.href, section.title, "Empty document"))
                continue
            kept.append(section)
        if pending is not None:
            skipped.append(SkippedItem(pending.href, pending.title, "Text too short (<50 chars)"))

        number = 0
        for section in kept:
            if section.kind == "chapter":
                number += 1
                section.num = number
                if not section.title:
                    section.title = f"Chapter {number}"
        for section in kept:
            if section.kind != "chapter":
                number += 1
                section.num = number
                section.title = section.title or "Untitled"
        return kept, skipped

    @staticmethod
    def epub_docs(book) -> list[HtmlDoc]:
        """The book's HTML documents in SPINE (reading) order."""
        by_id = {item.id: item for item in book.get_items() if getattr(item, "id", None)}
        ordered = []
        for entry in (book.spine or []):
            idref, linear = (tuple(entry) + ("yes",))[:2] if isinstance(entry, (tuple, list)) else (entry, "yes")
            item = by_id.get(idref) if isinstance(idref, str) else idref
            if item is None or not hasattr(item, "get_body_content"):
                continue
            if item.get_type() != ebooklib.ITEM_DOCUMENT and not isinstance(item, epub.EpubNav):
                continue
            ordered.append((item, str(linear).lower() != "no"))
        if not ordered:
            # No usable spine: manifest order is the only order there is.
            ordered = [(item, True) for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT)]

        docs: list[HtmlDoc] = []
        seen: set[int] = set()
        for item, linear in ordered:
            if id(item) in seen:
                continue
            seen.add(id(item))
            raw = item.content if isinstance(item.content, (bytes, bytearray)) else str(item.content or "").encode("utf-8")
            body_tag = _BODY_TAG.search(raw[:20000].decode("utf-8", errors="ignore"))
            body_types = frozenset(
                t for m in _EPUB_TYPE.finditer(body_tag.group(0) if body_tag else "")
                for t in m.group(1).lower().split()
            )
            docs.append(HtmlDoc(
                name=normalize_href(item.get_name() or "")[0],
                html=item.get_body_content().decode("utf-8", errors="replace"),
                linear=linear,
                is_nav=isinstance(item, epub.EpubNav) or "nav" in (getattr(item, "properties", None) or []),
                item=item,
                epub_types=body_types,
            ))
        return docs

    def plan_epub(
        self, book, classifier: MLClassifier | None = None,
    ) -> tuple[list[EpubSection], list[SkippedItem], list[TocEntry]]:
        """Plans the chapters of an opened EPUB (no conversion, no OCR).

        Parameters
        ----------
        book : ebooklib.epub.EpubBook
            Book returned by ``epub.read_epub``.
        classifier : MLClassifier | None
            Classifier for files the TOC does not list.

        Returns
        -------
        tuple[list[EpubSection], list[SkippedItem], list[TocEntry]]
            Planned sections, dropped items and the flattened TOC.
        """
        docs = self.epub_docs(book)
        base_dir = ""
        if not any(isinstance(item, epub.EpubNav) for item in book.get_items()):
            # NCX hrefs are relative to the NCX file, not to the OPF.
            ncx = next((item for item in book.get_items() if isinstance(item, epub.EpubNcx)), None)
            base_dir = posixpath.dirname(ncx.get_name() or "") if ncx is not None else ""
        _, _, toc_entries = self._walk_epub_toc(book.toc, base_dir)
        sections, skipped = self.plan_html_sections(docs, toc_entries, classifier)
        return sections, skipped, toc_entries

    # ── Conversion of planned sections ────────────────────────────────────────

    def convert_section(
        self,
        section: EpubSection,
        normalizer: TextNormalizer,
        ocr_book=None,
        language: str | None = None,
    ) -> ChapterItem | None:
        """Converts one planned section to normalised text (Docling, else BeautifulSoup)."""
        raw_parts: list[str] = []
        norm_parts: list[str] = []
        merged_ir: dict = {}
        merged_ocr: list[str] = []
        methods_used: set[str] = set()
        demarkdown = getattr(normalizer, "strip_markdown_structure", None)

        for doc, frag in section.parts:
            # Parsed (and OCR'd) exactly once, whichever converter ends up being used.
            soup, tokens = self._preprocess_soup(frag, ocr_book, doc.name, language)
            raw_md, ir_dict, ocr_texts = "", {}, []
            if DOCLING_AVAILABLE and self._converter is not None:
                raw_md, ir_dict, ocr_texts = self._docling_convert(str(soup), tokens)
            if raw_md:
                methods_used.add("docling")
                norm_parts.append(demarkdown(raw_md) if demarkdown else raw_md)
            else:
                raw_md = self._soup_to_text(soup, tokens)
                methods_used.add("beautifulsoup")
                norm_parts.append(raw_md)
            raw_parts.append(raw_md)
            merged_ocr.extend(ocr_texts)
            if not merged_ir:
                merged_ir = ir_dict   # keep the lead file's IR for 0_docling_ir.json

        method = "docling" if "docling" in methods_used else "beautifulsoup"
        if len(section.parts) > 1:
            method += f"+merged({len(section.parts)} files)"

        normalized = normalizer.normalize("\n\n".join(p for p in norm_parts if p), section.title, merged_ocr)
        if not normalized.strip():
            return None
        return ChapterItem(
            num=section.num,
            title=section.title,
            raw_md="\n\n".join(p for p in raw_parts if p),
            normalized=normalized,
            sentences=normalizer.split_sentences(normalized),
            method=method,
            ir_json=merged_ir,
            xgb_score=1.0 if section.kind == "chapter" else 0.0,
            probably_matter=section.probably_matter,
            href=section.href,
        )

    def convert_sections(
        self,
        sections: list[EpubSection],
        normalizer: TextNormalizer,
        *,
        ocr_book=None,
        language: str | None = None,
    ) -> tuple[list[ChapterItem], list[SkippedItem]]:
        """Converts planned sections; releases the OCR reader when done."""
        chapters: list[ChapterItem] = []
        skipped: list[SkippedItem] = []
        try:
            for section in sections:
                chapter = self.convert_section(section, normalizer, ocr_book, language)
                if chapter is None:
                    skipped.append(SkippedItem(section.href, section.title, "No text after normalisation"))
                else:
                    chapters.append(chapter)
        finally:
            if ocr_book is not None:
                self.release_ocr_reader()
        return chapters, skipped

    # ── Public EPUB ingestion ─────────────────────────────────────────────────

    def ingest_epub(
        self,
        epub_path: str,
        classifier: MLClassifier,
        normalizer: TextNormalizer,
        enable_ocr: bool = True,
        *,
        selections: list[int] | list[str] | None = None,
        book=None,
        language: str | None = None,
    ) -> tuple[list[ChapterItem], list[SkippedItem], list[TocEntry]]:
        """Extracts the chapters of an EPUB.

        Parameters
        ----------
        epub_path : str
            Path to the EPUB file.
        classifier : MLClassifier
            Classifier for documents the TOC does not list.
        normalizer : TextNormalizer
            Text normaliser.
        enable_ocr : bool
            OCR embedded images (EasyOCR) and put their text where the image was.
        selections : list[int] | list[str] | None
            Chapter titles or numbers to extract; unselected chapters are never
            converted. ``None`` extracts every chapter not flagged as matter.
        book : EpubBook | None
            An already opened (and safety-checked) book, to avoid reading the
            archive a second time.
        language : str | None
            OCR language override; defaults to the book's ``dc:language``.

        Returns
        -------
        tuple[list[ChapterItem], list[SkippedItem], list[TocEntry]]
            Extracted chapters, skipped items and the flattened TOC.
        """
        if book is None:
            from audiobook_factory.text_extractor import _assert_zip_safe  # type: ignore
            _assert_zip_safe(epub_path)
            book = epub.read_epub(epub_path)

        sections, skipped, toc_entries = self.plan_epub(book, classifier)
        logger.info(
            "EPUB plan: %d chapters, %d front/back-matter items, %d dropped.",
            sum(1 for s in sections if s.kind == "chapter"),
            sum(1 for s in sections if s.kind != "chapter"), len(skipped),
        )

        chosen = select_sections(sections, selections)
        chosen_ids = {id(s) for s in chosen}
        skipped.extend(
            SkippedItem(s.href, s.title, s.reason if s.kind != "chapter" else "Not selected")
            for s in sections if id(s) not in chosen_ids
        )
        if language is None:
            try:
                language = (book.get_metadata("DC", "language") or [(None, {})])[0][0]
            except Exception:
                language = None
        chapters, unconverted = self.convert_sections(
            chosen, normalizer, ocr_book=book if enable_ocr else None, language=language,
        )
        return chapters, skipped + unconverted, toc_entries

    # ── Public PDF ingestion ──────────────────────────────────────────────────

    def ingest_pdf(
        self,
        pdf_path: str,
        normalizer: TextNormalizer,
    ) -> tuple[list[ChapterItem], list[SkippedItem], list[TocEntry]]:

        if not DOCLING_AVAILABLE:
            print("    [WARNING] Docling not available — cannot extract PDF.")
            return [], [], []

        print("    Running Docling on PDF (may take several minutes)…")
        skipped: list[SkippedItem] = []

        _, toc_entries = self._extract_pdf_toc(pdf_path)

        try:
            result  = self._converter.convert(pdf_path)
            doc     = result.document
            raw_md  = self._export_markdown(doc)

            try:
                ir_dict = doc.export_to_dict()
            except Exception:
                ir_dict = {}

            ocr_texts = self._docling_ocr_blocks(doc)

            title   = os.path.splitext(os.path.basename(pdf_path))[0]
            # Pass is_pdf=True so header/footer noise stripping is enabled
            demarkdown = getattr(normalizer, "strip_markdown_structure", None)
            norm    = normalizer.normalize(
                demarkdown(raw_md) if demarkdown else raw_md, title, ocr_texts, is_pdf=True,
            )
            sents   = normalizer.split_sentences(norm)

            chapter = ChapterItem(
                num=1, title=title,
                raw_md=raw_md, normalized=norm,
                sentences=sents, method="docling",
                ir_json=ir_dict,
            )
            return [chapter], skipped, toc_entries

        except Exception as e:
            print(f"    [ERROR] Docling PDF extraction failed: {e}")
            traceback.print_exc()
            return [], [], toc_entries

    # ── Public TXT ingestion ──────────────────────────────────────────────────

    def ingest_txt(
        self,
        txt_path: str,
        normalizer: TextNormalizer,
    ) -> tuple[list[ChapterItem], list[SkippedItem], list[TocEntry]]:
        try:
            from audiobook_factory.text_extractor import read_text_file  # type: ignore
            raw = read_text_file(txt_path)
            title = os.path.splitext(os.path.basename(txt_path))[0]
            norm  = normalizer.normalize(raw, title, [])
            sents = normalizer.split_sentences(norm)
            return ([ChapterItem(
                num=1, title=title, raw_md=raw, normalized=norm,
                sentences=sents, method="plain_text", ir_json={},
            )], [], [])
        except Exception as e:
            print(f"    [ERROR] TXT read failed: {e}")
            return [], [], []


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 5  —  OutputWriter
# ══════════════════════════════════════════════════════════════════════════════

class OutputWriter:
    """Writes all debug output files for a single book."""

    MAX_IR_SIZE = 5 * 1024 * 1024  # 5 MB cap on IR JSON to avoid huge files

    def __init__(self, out_dir: str):
        self.out_dir = out_dir
        os.makedirs(out_dir, exist_ok=True)

    @staticmethod
    def _safe_name(text: str, maxlen: int = 60) -> str:
        return "".join(c if c.isalnum() or c in " _-" else "_" for c in text)[:maxlen]

    def write_chapter(self, ch: ChapterItem):
        folder = os.path.join(
            self.out_dir,
            f"ch{ch.num:03d} - {self._safe_name(ch.title)}"
        )
        os.makedirs(folder, exist_ok=True)

        # 0_docling_ir.json
        ir_str = json.dumps(ch.ir_json, indent=2, ensure_ascii=False)
        if len(ir_str) > self.MAX_IR_SIZE:
            ir_str = json.dumps({"note": "IR too large; truncated", "preview": ir_str[:2000]})
        with open(os.path.join(folder, "0_docling_ir.json"), "w", encoding="utf-8") as f:
            f.write(ir_str)

        # 1_docling_raw.md
        with open(os.path.join(folder, "1_docling_raw.md"), "w", encoding="utf-8") as f:
            f.write(f"<!-- extraction method: {ch.method} | xgb_score: {ch.xgb_score:.3f} -->\n\n")
            f.write(ch.raw_md)

        # 2_normalized.md
        with open(os.path.join(folder, "2_normalized.md"), "w", encoding="utf-8") as f:
            f.write(f"<!-- Phase 4 normalized — ready for TTS -->\n\n")
            f.write(ch.normalized)

        # 3_sentences.md
        with open(os.path.join(folder, "3_sentences.md"), "w", encoding="utf-8") as f:
            f.write(f"<!-- {len(ch.sentences)} TTS chunks (max_len={MAX_SENTENCE_LEN}) -->\n\n")
            for i, sent in enumerate(ch.sentences, 1):
                f.write(f"[{i:04d}] ({len(sent):3d} chars)  {sent}\n")

    def write_skipped(self, skipped: list[SkippedItem]):
        path = os.path.join(self.out_dir, "_skipped_items.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write("# Skipped Items\n\n")
            f.write("Items classified as non-chapter and excluded from the audiobook.\n\n")
            f.write("| # | Item Name | Title | Classification Reason | XGB Score |\n")
            f.write("|---|-----------|-------|----------------------|----------|\n")
            for i, s in enumerate(skipped, 1):
                f.write(f"| {i} | `{s.name}` | {s.title} | {s.reason} | {s.xgb_score:.3f} |\n")

    def write_toc_map(self, toc_entries: list[TocEntry]):
        path = os.path.join(self.out_dir, "_toc_map.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write("# TOC Map — Original vs Classification\n\n")
            f.write("| TOC Title | Href | Classification |\n")
            f.write("|-----------|------|----------------|\n")
            for e in toc_entries:
                icon = "✅" if e.classification == "chapter" else "❌"
                f.write(f"| {e.title} | `{e.href}` | {icon} {e.classification} |\n")

    def write_all_chapters(self, chapters: list[ChapterItem]):
        path = os.path.join(self.out_dir, "_all_chapters.md")
        with open(path, "w", encoding="utf-8") as f:
            f.write("# Full Book — Normalized Text\n\n")
            for ch in chapters:
                f.write(f"\n\n---\n\n## Chapter {ch.num}: {ch.title}\n\n")
                f.write(ch.normalized)
        return path

    def write_summary(self, file_info: dict):
        path = os.path.join(self.out_dir, "summary.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(file_info, f, indent=2, ensure_ascii=False)


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description="5-Phase Hybrid AI Extraction Engine")
    parser.add_argument("input_path", nargs="?", default=os.path.join(PROJECT_ROOT, "LOTM"), help="Path to book file or folder containing book files")
    parser.add_argument("--output", "-o", default=os.path.join(PROJECT_ROOT, "output"), help="Output directory")
    args = parser.parse_args()

    input_path = os.path.abspath(args.input_path)
    output_folder = os.path.abspath(args.output)
    os.makedirs(output_folder, exist_ok=True)

    print("=" * 70)
    print("  5-PHASE HYBRID EXTRACTION PIPELINE")
    print("  AudiobookMaker — Debug Text Extraction")
    print("=" * 70)
    print(f"  Input : {input_path}")
    print(f"  Output: {output_folder}")
    print()

    if os.path.isfile(input_path):
        input_folder = os.path.dirname(input_path)
        files = [os.path.basename(input_path)]
    elif os.path.isdir(input_path):
        input_folder = input_path
        files = [
            f for f in sorted(os.listdir(input_folder))
            if Path(f).suffix.lower() in SUPPORTED_EXT
        ]
    else:
        print(f"[ERROR] Input path not found: {input_path}")
        sys.exit(1)

    if not files:
        print("[ERROR] No supported files (.epub .pdf .txt .docx .odt) found.")
        sys.exit(1)

    print(f"Files to process ({len(files)}):\n")
    for f in files:
        size = os.path.getsize(os.path.join(input_folder, f)) / 1024 / 1024
        print(f"  • {f}  ({size:.1f} MB)")
    print()

    # Initialise the pipeline once (expensive models/converters)
    ingestor   = DocumentIngestor()
    classifier = MLClassifier()
    normalizer = TextNormalizer()

    all_summary: dict[str, Any] = {}

    for filename in files:
        filepath = os.path.join(input_folder, filename)
        stem     = Path(filename).stem
        ext      = Path(filename).suffix.lower()

        print("-" * 70)
        print(f"▶  {filename}")
        t0 = time.time()

        # Clear previous output for this book
        safe_stem = OutputWriter._safe_name(stem, 80)
        out_dir   = os.path.join(output_folder, safe_stem)
        if os.path.exists(out_dir):
            shutil.rmtree(out_dir)

        writer = OutputWriter(out_dir)

        try:
            if ext == ".epub":
                chapters, skipped, toc_entries = ingestor.ingest_epub(
                    filepath, classifier, normalizer
                )
            elif ext == ".pdf":
                chapters, skipped, toc_entries = ingestor.ingest_pdf(
                    filepath, normalizer
                )
            elif ext == ".txt":
                chapters, skipped, toc_entries = ingestor.ingest_txt(
                    filepath, normalizer
                )
            else:
                print("  [SKIP] Unknown extension.")
                continue
        except Exception as e:
            print(f"  [FATAL] Pipeline crashed: {e}")
            traceback.print_exc()
            continue

        elapsed = time.time() - t0
        print(f"  Extracted {len(chapters)} chapter(s), {len(skipped)} skipped  [{elapsed:.1f}s]")

        # Write outputs
        for ch in chapters:
            print(f"    ch{ch.num:03d}: {ch.title[:60]}  ({ch.method})")
            writer.write_chapter(ch)

        writer.write_skipped(skipped)
        writer.write_toc_map(toc_entries)
        all_chapters_path = writer.write_all_chapters(chapters)

        chapter_stats = [
            {
                "num":              ch.num,
                "title":            ch.title,
                "method":           ch.method,
                "xgb_score":        round(ch.xgb_score, 3),
                "raw_chars":        len(ch.raw_md),
                "normalized_chars": len(ch.normalized),
                "sentence_count":   len(ch.sentences),
                "avg_sentence_len": round(
                    sum(len(s) for s in ch.sentences) / max(1, len(ch.sentences)), 1
                ),
                "max_sentence_len": max((len(s) for s in ch.sentences), default=0),
            }
            for ch in chapters
        ]
        summary = {
            "file":             filename,
            "chapters":         len(chapters),
            "skipped":          len(skipped),
            "elapsed_seconds":  round(elapsed, 2),
            "chapter_details":  chapter_stats,
        }
        writer.write_summary(summary)
        all_summary[filename] = summary

        print(f"  ✓ Output → {out_dir}")

    # Master summary
    master_path = os.path.join(output_folder, "_master_summary.json")
    with open(master_path, "w", encoding="utf-8") as f:
        json.dump(all_summary, f, indent=2, ensure_ascii=False)

    print()
    print("=" * 70)
    print(f"  DONE.  Master summary: {master_path}")
    print(f"  Inspect output in:    {output_folder}")
    print("=" * 70)


if __name__ == "__main__":
    main()
