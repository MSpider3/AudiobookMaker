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

import json
import os
import re
import shutil
import statistics
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import ebooklib
from bs4 import BeautifulSoup
from ebooklib import epub

# ── sys.path so the project root is importable ───────────────────────────────
# This file lives in audiobook_factory/, so go up one level to find the project root
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from audiobook_factory.text_processing import normalize_text, smart_sentence_splitter

# ── Docling ───────────────────────────────────────────────────────────────────
try:
    from docling.document_converter import DocumentConverter
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import PdfPipelineOptions, RapidOcrOptions
    DOCLING_AVAILABLE = True
except ImportError:
    DOCLING_AVAILABLE = False
    print("[WARNING] Docling not installed — HTML/PDF extraction will be limited.")

# ── PyMuPDF (optional, for PDF TOC) ──────────────────────────────────────────
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

# ══════════════════════════════════════════════════════════════════════════════
# Config
# ══════════════════════════════════════════════════════════════════════════════
SUPPORTED_EXT    = {".epub", ".mobi", ".pdf", ".docx", ".odt", ".txt"}
MAX_SENTENCE_LEN = 399   # matches config.py

# Chapter keywords for heuristic scoring
_CHAPTER_KW = re.compile(
    r"^\s*(chapter|part|book|prologue|epilogue|interlude|volume|act|section)\b",
    re.I,
)

# Skip-list for TOC entry titles (front/back-matter, gallery, legal…)
_SKIP_TOC_TITLE = re.compile(
    r"^(table\s*of\s*contents|toc|index|preface|foreword|introduction|"
    r"copyright|cover|title\s*page|about|postscript|newsletter|"
    r"image\s*gallery|characters?|locations?|map|pathways?|"
    r"epilogue|afterword|end\s*of|to\s*be\s*continued|back\s*cover|"
    r"coloph|errata|bibliography|glossary|character\s*gallery|"
    r"pathways\s*guide|bonus\s*chapter|contact\s*us|credits?)",
    re.I,
)

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
        # Normalise href basename for matching
        name_base = item_name.split("#")[0].split("/")[-1]

        # Priority 1 — TOC says chapter
        if any(name_base in h for h in chapter_hrefs):
            return ("chapter", 1.0)

        # Priority 2 — TOC says skip
        if any(name_base in h for h in skip_hrefs):
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
    _HR        = re.compile(r"^\s*(-{3,}|\*{3,}|_{3,})\s*$", re.MULTILINE)
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
    # Heuristic: >=18 chars, no spaces, starts uppercase, contains multiple
    # lowercase sequences (rules out legitimate acronyms like "UNESCO").
    _GARBLED    = re.compile(r"[A-Z][a-zA-Z]{11,}")
    # Running page-header lines: short (≤60 chars) with no sentence-ending punctuation
    _PAGE_NUM   = re.compile(r"^\s*\d{1,4}\s*$", re.MULTILINE)
    # Docling image placeholder lines
    _IMG_BLOCK  = re.compile(r"^<!-- image -->\s*$", re.MULTILINE)

    # ── Individual normalisation steps ───────────────────────────────────────

    def _strip_pdf_noise(self, text: str) -> str:
        """
        Removes PDF-specific extraction artefacts:
        1. Standalone page numbers.
        2. Docling <!-- image --> placeholder lines.
        3. Garbled OCR from image blocks (CamelCase word-soup with no spaces).
        4. Repeating short lines (running headers/footers): if an identical
           line of ≤60 chars appears 3+ times it is almost certainly a header.
        """
        # 1. Page numbers on their own line
        text = self._PAGE_NUM.sub("", text)
        # 2. Docling image placeholders
        text = self._IMG_BLOCK.sub("", text)
        # 3. Garbled OCR blobs — remove lines whose text content (after stripping
        #    any markdown heading markers like ## or ###) is ONLY a CamelCase blob
        _MD_HEADING = re.compile(r"^#+\s*")
        lines = text.split("\n")
        cleaned = []
        for line in lines:
            stripped = line.strip()
            # Remove heading markers to get the bare text
            bare = _MD_HEADING.sub("", stripped).strip()
            # A "garbled" line: bare text has no internal spaces AND matches CamelCase blob
            if bare and " " not in bare and self._GARBLED.fullmatch(bare):
                continue
            cleaned.append(line)
        text = "\n".join(cleaned)
        # 4. Detect and remove repeating short header/footer lines
        from collections import Counter
        line_counts = Counter(
            l.strip() for l in text.split("\n")
            if 3 <= len(l.strip()) <= 60 and not l.strip().endswith((".", "!", "?", ":", ","))
        )
        repeating = {ln for ln, cnt in line_counts.items() if cnt >= 3}
        if repeating:
            text = "\n".join(
                l for l in text.split("\n")
                if l.strip() not in repeating
            )
        return text

    # Patterns for Markdown noise
    _FOOTNOTE_LINK = re.compile(r"\[\[\d+\]\]\([^)]+\)|\[\d+\]\([^)]+\)")

    def _strip_noise(self, text: str) -> str:
        # Remove the OCR_IMG_TEXT: prefix we maliciously injected in _preprocess_html
        # MUST happen before _BOLD_EM to prevent _IMG_ from being stripped as an italic tag!
        text = re.sub(r"OCR_IMG_TEXT:\s*", "", text)

        text = self._IMG_TAG.sub("", text)              # strip ![...](...)
        text = self._HR.sub("\n\n", text)               # strip --- and ***
        text = self._BOLD_EM.sub(r"\1\2", text)         # strip ** or _
        text = self._FOOTNOTE_LINK.sub("", text)        # strip footnotes like [[1]](#id_C0001)
        
        text = text.translate(self._SMART_Q)            # normalise smart quotes
        text = self._MULTI_BL.sub("\n\n", text)         # collapse blank lines
        return text

    def _fix_isolated_capitals(self, text: str) -> str:
        """
        Fixes PDF kerning issues where a single capital letter is detached from
        the rest of the word due to drop-cap or font spacing anomalies.
        Examples: 'T HE' -> 'THE', 'W ar' -> 'War', 'T ohsaka' -> 'Tohsaka'
        """
        # A single capital letter at the start of a word, followed by a space,
        # followed by lowercase letters or all-caps continuing the word.
        # We use a negative lookbehind for letters to ensure it's the start of a word.
        # Group 1: The isolated capital letter
        # Group 2: The rest of the word (must be letters)
        # We replace "C rest" with "Crest". We avoid merging "A dog" or "I am".
        
        # We need a function to conditionally merge because "A" and "I" are valid words.
        def _merge_if_not_word(match):
            cap = match.group(1)
            rest = match.group(2)
            
            # The word was split in the PDF/EPUB. 
            # E.g. cap="A", rest="nd" or cap="I", rest="t"
            if cap in ("A", "I"):
                rest_lower = rest.lower()
                # Strict list of valid single-letter prefixed words (and, as, at, are, an, all, any, it, is, if, in, ill)
                if cap == "A" and rest_lower in ("nd", "s", "t", "re", "n", "ll", "ny", "lthough", "gain", "nother", "lready", "lways"):
                    return cap + rest
                if cap == "I" and rest_lower in ("t", "s", "f", "n", "ll", "nto", "ndeed", "tself"):
                    return cap + rest
                # If it's anything else, it's likely a real word following "A" or "I"
                # (e.g. "I didn't", "A New")
                return f"{cap} {rest}"
                
            # For letters other than A and I, they are never standalone words in English 
            # (e.g. "T HE", "W ar", "S he", "Y our", "C rest")
            return cap + rest

        # Pattern: [not letter] [Capital] [space] [letters]
        _ISO_CAP = re.compile(r"(?<![a-zA-Z])([A-Z])\s+([a-zA-Z]{1,})\b")
        text = _ISO_CAP.sub(_merge_if_not_word, text)
        
        # Handle all-caps splits specifically (e.g., "T HE", "W AR", "A ND", "I T", "A THURSDAY")
        def _merge_all_caps(match):
            cap = match.group(1)
            rest = match.group(2)
            if cap in ("A", "I"):
                # "A ND", "I T", "A S", "I S"
                if cap == "A" and rest in ("ND", "S", "T", "RE", "N", "LL", "NY"): return cap + rest
                if cap == "I" and rest in ("T", "S", "F", "N"): return cap + rest
                return f"{cap} {rest}"  # "A THURSDAY", "I WANT"
            # "T HE", "W AR", "Y OUR"
            return cap + rest
            
        text = re.sub(r"\b([A-Z])\s+([A-Z]+)\b", _merge_all_caps, text)
        
        return text

    def _fix_broken_lines(self, text: str) -> str:
        text = self._HYPHEN_WR.sub(r"\1", text)         # "conver-\nsion" → "conversion"
        text = self._SOFT_WRAP.sub(" ", text)            # soft-wrap join
        text = self._fix_isolated_capitals(text)        # "W ar" → "War"
        return text

    def _remove_duplicate_title(self, title: str, text: str) -> str:
        """
        If the chapter title appears verbatim in the first 3 lines (Docling
        often emits it as an h1 AND the EPUB has it as a paragraph), remove
        the duplicate.
        """
        lines = text.split("\n")
        cleaned = []
        stripped_title = title.strip().lower()
        for i, line in enumerate(lines):
            # Strip markdown heading markers for comparison
            bare = re.sub(r"^#+\s*", "", line).strip().lower()
            if i < 4 and bare == stripped_title and cleaned:
                continue  # skip duplicate title line
            cleaned.append(line)
        return "\n".join(cleaned)

    # ── Public API ────────────────────────────────────────────────────────────

    def normalize(self, raw_md: str, title: str, ocr_block_texts: list[str],
                  is_pdf: bool = False) -> str:
        """
        Full normalization pipeline.
        ocr_block_texts: list of text strings extracted by OCR (from Docling IR).
        is_pdf: if True, also run _strip_pdf_noise() to clean running headers/footers.
        """
        text = raw_md

        # 1. LLM OCR repair — targeted, only on OCR blocks
        for ocr_txt in ocr_block_texts:
            if ocr_txt and ocr_txt in text:
                repaired = self.llm_repair_ocr_block(ocr_txt)
                if repaired != ocr_txt:
                    text = text.replace(ocr_txt, repaired, 1)

        # 2. PDF-specific noise (headers, footers, garbled OCR images)
        if is_pdf:
            text = self._strip_pdf_noise(text)

        # 3. Remove duplicate title heading
        text = self._remove_duplicate_title(title, text)

        # 4. Fix broken lines before noise strip to avoid stripping mid-word
        text = self._fix_broken_lines(text)

        # 5. Markdown noise strip
        text = self._strip_noise(text)

        # 6. Legacy audiobook pipeline normalisation (de-wrap, drop-cap)
        text = normalize_text(text)

        return text.strip()

    def split_sentences(self, text: str) -> list[str]:
        return smart_sentence_splitter(text, MAX_SENTENCE_LEN)


# ══════════════════════════════════════════════════════════════════════════════
# PHASE 1 & 2  —  DocumentIngestor
# ══════════════════════════════════════════════════════════════════════════════

class DocumentIngestor:
    """
    Phase 1: TOC extraction + drop-cap pre-processing.
    Phase 2: Docling ingestion with explicit format routing + OCR.
    """

    def __init__(self):
        if DOCLING_AVAILABLE:
            self._converter = DocumentConverter()
            # Separate PDF converter with forced OCR
            pdf_opts = PdfPipelineOptions()
            pdf_opts.do_ocr = True          # Enable OCR for embedded images
            pdf_opts.do_table_structure = False
            pdf_opts.ocr_options = RapidOcrOptions()
            self._pdf_converter = DocumentConverter(
                format_options={
                    InputFormat.PDF: type("FormatOption", (), {
                        "pipeline_options": pdf_opts
                    })()
                }
            )
        else:
            self._converter = None
            self._pdf_converter = None

    # ── Phase 1a: TOC extraction ──────────────────────────────────────────────

    def _walk_epub_toc(self, toc_items) -> tuple[set[str], set[str], list[TocEntry]]:
        """Recursively walk epub TOC; returns (chapter_hrefs, skip_hrefs, entries)."""
        chapter_hrefs: set[str] = set()
        skip_hrefs:    set[str] = set()
        entries:       list[TocEntry] = []

        for item in toc_items:
            if isinstance(item, epub.Link):
                href  = item.href.split("#")[0]   # strip anchor
                title = item.title or ""
                if _SKIP_TOC_TITLE.match(title.strip()):
                    skip_hrefs.add(href)
                    entries.append(TocEntry(title, href, "skip"))
                else:
                    chapter_hrefs.add(href)
                    entries.append(TocEntry(title, href, "chapter"))
            elif isinstance(item, tuple) and len(item) == 2:
                section, children = item
                # Recurse into section
                ch, sk, en = self._walk_epub_toc(children)
                section_title = getattr(section, "title", "") or ""
                section_href  = getattr(section, "href", "").split("#")[0]
                section_cls   = "skip" if _SKIP_TOC_TITLE.match(section_title.strip()) else "chapter"
                if section_cls == "skip":
                    sk.add(section_href)
                else:
                    ch.add(section_href)
                entries.append(TocEntry(section_title, section_href, section_cls))
                entries.extend(en)
                chapter_hrefs |= ch
                skip_hrefs    |= sk

        return chapter_hrefs, skip_hrefs, entries

    def _extract_pdf_toc(self, pdf_path: str) -> tuple[set[int], list[TocEntry]]:
        """Returns (chapter_pages, entries) from a PDF TOC via PyMuPDF."""
        chapter_pages: set[int] = set()
        entries: list[TocEntry] = []
        if not PYMUPDF_AVAILABLE:
            return chapter_pages, entries
        try:
            doc = fitz.open(pdf_path)
            for level, title, page in doc.get_toc():
                cls = "skip" if _SKIP_TOC_TITLE.match((title or "").strip()) else "chapter"
                if cls == "chapter":
                    chapter_pages.add(page)
                entries.append(TocEntry(title or "", str(page), cls))
            doc.close()
        except Exception as e:
            print(f"    [PDF TOC] extraction failed: {e}")
        return chapter_pages, entries

    # ── Phase 1b: HTML drop-cap pre-processing ────────────────────────────────

    @staticmethod
    def _preprocess_html(html_content: str, epub_book=None, epub_item_name: str = "") -> tuple[str, list[str]]:
        """
        Cleans EPUB HTML before Docling sees it:
        1. Unwraps drop-cap <span>s.
        2. Runs OCR on embedded images and extracts the text to bypass Docling stripping.
        Returns: (processed_html_string, list_of_extracted_ocr_texts)
        """
        soup = BeautifulSoup(html_content, "html.parser")
        extracted_images = []

        for span in soup.find_all("span"):
            classes = " ".join(span.get("class", []))
            style   = span.get("style", "")
            text    = span.get_text(strip=True)
            if len(text) == 1 and (
                "drop" in classes.lower() or
                ("font-size" in style and "em" in style)
            ):
                span.unwrap()   # merge single-char drop-cap with following word

        # ── In-flight EPUB Image OCR ──
        if epub_book is not None:
            try:
                import easyocr
                # We initialize lazily so we don't block startup or throw errors if missing
                if not hasattr(DocumentIngestor, "_easyocr_reader"):
                    DocumentIngestor._easyocr_reader = easyocr.Reader(["en"], gpu=True)
                
                for img in soup.find_all(["img", "image"]):
                    src = img.get("src") or img.get("xlink:href")
                    if not src: continue
                    
                    # Resolve relative path inside the EPUB archive
                    base = epub_item_name.rsplit('/', 1)[0] if '/' in epub_item_name else ''
                    full_src = f'{base}/{src}'.strip('/') if base else src
                    
                    img_item = epub_book.get_item_with_href(full_src) or epub_book.get_item_with_href(src)
                    if img_item:
                        try:
                            import io
                            from PIL import Image
                            import numpy as np

                            raw_data = img_item.get_content()
                            img_stream = io.BytesIO(raw_data)
                            img_obj = Image.open(img_stream)
                            
                            # Convert to RGB if it's not (e.g. RGBA or Grayscale) to prevent easyocr/cv2 errors
                            if img_obj.mode != "RGB":
                                img_obj = img_obj.convert("RGB")
                                
                            img_np = np.array(img_obj)

                            res = DocumentIngestor._easyocr_reader.readtext(img_np)
                            if res:
                                extracted_text = " ".join([line[1] for line in res]).strip()
                                if extracted_text:
                                    # Collect to append directly to the Markdown later, completely bypassing Docling parser
                                    extracted_images.append(f"OCR_IMG_TEXT: {extracted_text}")
                        except Exception as e:
                            print(f"      [OCR] Failed on image {src}: {e}")
            except ImportError:
                pass  # easyocr not installed

        return str(soup), extracted_images

    # ── Phase 2: Docling ingestion ────────────────────────────────────────────

    def _docling_html(self, html_content: str, epub_book=None, epub_item_name: str = "") -> tuple[str, dict, list[str]]:
        """
        Run Docling on preprocessed HTML.
        Returns (raw_markdown, ir_dict, ocr_block_texts).
        """
        processed, extracted_images = self._preprocess_html(html_content, epub_book, epub_item_name)

        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                suffix=".html", mode="w", encoding="utf-8", delete=False
            ) as tmp:
                tmp.write(processed)
                tmp_path = tmp.name

            result  = self._converter.convert(tmp_path)
            doc     = result.document
            raw_md  = doc.export_to_markdown()
            
            # Manually append the OCR text extracted from images directly into the Markdown
            # This cleanly bypasses Docling's aggressive tag-stripping behavior
            if extracted_images:
                raw_md += "\n\n" + "\n\n".join(extracted_images) + "\n\n"

            try:
                ir_dict = doc.export_to_dict()
            except Exception:
                ir_dict = {}

            # Identify OCR-sourced text blocks (for Phase 4 LLM repair)
            ocr_texts = []
            for block in getattr(doc, "texts", []):
                prov = getattr(block, "prov", [])
                for p in (prov if isinstance(prov, list) else [prov]):
                    if getattr(p, "charspan", None) == (0, 0):
                        # zero charspan means Docling had no native text → OCR
                        ocr_texts.append(getattr(block, "text", ""))

            return raw_md, ir_dict, ocr_texts

        except Exception as e:
            print(f"      [Docling HTML] failed: {e}")
            return "", {}, []
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

    def _bs_fallback(self, html_content: str, epub_book=None, epub_item_name: str = "") -> str:
        """BeautifulSoup plain-text fallback."""
        processed, extracted_images = self._preprocess_html(html_content, epub_book, epub_item_name)
        soup = BeautifulSoup(processed, "html.parser")
        text = soup.get_text(separator="\n\n", strip=True)
        if extracted_images:
            text += "\n\n" + "\n\n".join(extracted_images) + "\n\n"
        return text

    # ── Public EPUB ingestion ─────────────────────────────────────────────────

    def _group_spine_by_chapter(
        self,
        items: list,
        chapter_hrefs: set[str],
        skip_hrefs: set[str],
    ) -> list[list]:
        """
        Groups EPUB document items into chapter buckets.

        Many EPUBs split one narrative chapter across multiple HTML files
        (e.g. index_split_007 through index_split_017 all belong to Chapter 2
        but only 007 appears in the TOC). This method walks the spine in order
        and appends 'orphan' files (not in any TOC href) to the most recently
        opened chapter group.

        Returns a list of groups, where each group is a list of items that
        belong to the same logical chapter. Groups for explicitly skipped
        items are returned as single-item lists tagged with a sentinel.
        """
        groups: list[list] = []          # list of ["chapter"|"skip"|"front", item, item, ...]
        current_group: list | None = None

        for idx, item in enumerate(items):
            name = (item.get_name() or "").split("/")[-1]

            in_chapter = any(name in h for h in chapter_hrefs)
            in_skip    = any(name in h for h in skip_hrefs)

            if in_chapter:
                # Start a new chapter group
                current_group = ["chapter", item]
                groups.append(current_group)
            elif in_skip:
                # Explicitly skipped — own isolated group
                groups.append(["skip", item])
                # Don't update current_group; orphans after a skip still
                # attach to the last real chapter
            else:
                # Orphan — continuation of the previous chapter, or pre-chapter front matter
                if current_group is not None:
                    current_group.append(item)
                else:
                    # Before any chapter has started → front matter
                    groups.append(["front", item])

        return groups

    def ingest_epub(
        self,
        epub_path: str,
        classifier: MLClassifier,
        normalizer: TextNormalizer,
    ) -> tuple[list[ChapterItem], list[SkippedItem], list[TocEntry]]:

        book  = epub.read_epub(epub_path)
        items = list(book.get_items_of_type(ebooklib.ITEM_DOCUMENT))
        print(f"    Found {len(items)} EPUB document items.")

        # Phase 1: TOC extraction
        chapter_hrefs, skip_hrefs, toc_entries = self._walk_epub_toc(book.toc)
        print(f"    TOC: {len(chapter_hrefs)} chapter hrefs, {len(skip_hrefs)} skip hrefs.")

        # Group spine items so multi-file chapters are merged
        groups = self._group_spine_by_chapter(items, chapter_hrefs, skip_hrefs)
        print(f"    Spine groups: {sum(1 for g in groups if g[0]=='chapter')} chapters, "
              f"{sum(1 for g in groups if g[0]=='skip')} skip, "
              f"{sum(1 for g in groups if g[0]=='front')} front.")

        chapters: list[ChapterItem] = []
        skipped:  list[SkippedItem] = []
        chapter_num = 1

        for group in groups:
            group_type = group[0]
            group_items = group[1:]
            if not group_items:
                continue

            lead_item = group_items[0]
            lead_name = lead_item.get_name() or ""
            lead_html = lead_item.get_body_content().decode("utf-8", errors="replace")
            lead_soup = BeautifulSoup(lead_html, "html.parser")
            heading    = lead_soup.find(["h1", "h2", "h3"])
            item_title = heading.get_text(strip=True) if heading else lead_name

            # Handle explicitly skipped groups
            if group_type in ("skip", "front"):
                word_count = len(lead_soup.get_text().split())
                reason = "Explicitly skipped (TOC)" if group_type == "skip" \
                    else f"Front/back matter (words={word_count})"
                skipped.append(SkippedItem(
                    name=lead_name, title=item_title,
                    reason=reason, xgb_score=0.0,
                ))
                continue

            # ── Phase 2: Docling ingestion — one call per HTML file, then merge ──
            merged_raw_md   = ""
            merged_ir       = {}
            merged_ocr      = []
            methods_used    = set()

            for file_item in group_items:
                html_raw = file_item.get_body_content().decode("utf-8", errors="replace")
                raw_md, ir_dict, ocr_texts = "", {}, []

                if DOCLING_AVAILABLE and self._converter:
                    raw_md, ir_dict, ocr_texts = self._docling_html(
                        html_raw, epub_book=book, epub_item_name=file_item.get_name() or ""
                    )
                    if raw_md:
                        methods_used.add("docling")

                if not raw_md:
                    raw_md = self._bs_fallback(html_raw, epub_book=book, epub_item_name=file_item.get_name() or "")
                    methods_used.add("beautifulsoup")

                # Append with a blank line separator
                merged_raw_md += ("\n\n" if merged_raw_md else "") + raw_md
                merged_ocr.extend(ocr_texts)
                if not merged_ir:
                    merged_ir = ir_dict   # keep the lead file's IR for 0_docling_ir.json

            method = "docling" if "docling" in methods_used else "beautifulsoup"
            if len(group_items) > 1:
                method += f"+merged({len(group_items)} files)"

            total_words = sum(
                len(BeautifulSoup(
                    fi.get_body_content().decode("utf-8", errors="replace"), "html.parser"
                ).get_text().split())
                for fi in group_items
            )

            # Phase 3: classify — use the lead file's position and word count
            position_idx = items.index(lead_item)
            classification, xgb_score = classifier.classify_item(
                item_name=lead_name,
                item_title=item_title,
                word_count=total_words,
                position_idx=position_idx,
                chapter_hrefs=chapter_hrefs,
                skip_hrefs=skip_hrefs,
                doc_texts=[],
                avg_font=12.0,
            )

            if classification != "chapter":
                reason = {
                    "front_matter": f"Classified as front matter (words={total_words})",
                    "back_matter":  f"Classified as back matter (title={item_title!r})",
                    "toc":          "Table of contents item",
                    "gallery":      "Image/character gallery",
                }.get(classification, classification)
                skipped.append(SkippedItem(
                    name=lead_name, title=item_title,
                    reason=reason, xgb_score=xgb_score,
                ))
                continue

            # Phase 4: normalize the merged text
            normalized = normalizer.normalize(merged_raw_md, item_title, merged_ocr)
            sentences  = normalizer.split_sentences(normalized)

            if len(normalized.strip()) < 50:
                skipped.append(SkippedItem(
                    name=lead_name, title=item_title,
                    reason="Normalized text too short (<50 chars)",
                    xgb_score=xgb_score,
                ))
                continue

            chapters.append(ChapterItem(
                num=chapter_num,
                title=item_title,
                raw_md=merged_raw_md,
                normalized=normalized,
                sentences=sentences,
                method=method,
                ir_json=merged_ir,
                xgb_score=xgb_score,
            ))
            chapter_num += 1

        return chapters, skipped, toc_entries

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
            raw_md  = doc.export_to_markdown()

            try:
                ir_dict = doc.export_to_dict()
            except Exception:
                ir_dict = {}

            # OCR blocks
            ocr_texts = []
            for block in getattr(doc, "texts", []):
                prov = getattr(block, "prov", [])
                for p in (prov if isinstance(prov, list) else [prov]):
                    if getattr(p, "charspan", None) == (0, 0):
                        ocr_texts.append(getattr(block, "text", ""))

            title   = os.path.splitext(os.path.basename(pdf_path))[0]
            # Pass is_pdf=True so header/footer noise stripping is enabled
            norm    = normalizer.normalize(raw_md, title, ocr_texts, is_pdf=True)
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
            with open(txt_path, "r", encoding="utf-8", errors="replace") as f:
                raw = f.read()
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
    print("=" * 70)
    print("  5-PHASE HYBRID EXTRACTION PIPELINE")
    print("  AudiobookMaker — Debug Text Extraction")
    print("=" * 70)
    print(f"  Input : {LOTM_FOLDER}")
    print(f"  Output: {OUTPUT_FOLDER}")
    print()

    if not os.path.isdir(LOTM_FOLDER):
        print(f"[ERROR] LOTM folder not found: {LOTM_FOLDER}")
        sys.exit(1)

    files = [
        f for f in sorted(os.listdir(LOTM_FOLDER))
        if Path(f).suffix.lower() in SUPPORTED_EXT
    ]
    if not files:
        print("[ERROR] No supported files (.epub .pdf .txt) found.")
        sys.exit(1)

    print(f"Files to process ({len(files)}):\n")
    for f in files:
        size = os.path.getsize(os.path.join(LOTM_FOLDER, f)) / 1024 / 1024
        print(f"  • {f}  ({size:.1f} MB)")
    print()

    # Initialise the pipeline once (expensive models/converters)
    ingestor   = DocumentIngestor()
    classifier = MLClassifier()
    normalizer = TextNormalizer()

    all_summary: dict[str, Any] = {}

    for filename in files:
        filepath = os.path.join(LOTM_FOLDER, filename)
        stem     = Path(filename).stem
        ext      = Path(filename).suffix.lower()

        print("-" * 70)
        print(f"▶  {filename}")
        t0 = time.time()

        # Clear previous output for this book
        safe_stem = OutputWriter._safe_name(stem, 80)
        out_dir   = os.path.join(OUTPUT_FOLDER, safe_stem)
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
    master_path = os.path.join(OUTPUT_FOLDER, "_master_summary.json")
    with open(master_path, "w", encoding="utf-8") as f:
        json.dump(all_summary, f, indent=2, ensure_ascii=False)

    print()
    print("=" * 70)
    print(f"  DONE.  Master summary: {master_path}")
    print(f"  Inspect output in:    {OUTPUT_FOLDER}")
    print("=" * 70)


if __name__ == "__main__":
    main()
