"""
audiobook_factory/text_extractor.py
====================================
Public API for the text extraction pipeline.
Wraps DocumentIngestor, MLClassifier, TextNormalizer from the
production-tested extractor_engine module and adds chapter detection for
TXT, DOCX, ODT and PDF plus MOBI/AZW3 unpacking.

Public API
----------
scan(path, include_matter=False)  -> ScanResult     (fast, no OCR – for UI display)
extract(path, selections, ...)    -> (list[ExtractedChapter], cover_bytes | None)
read_text_file(path)              -> str
ExtractionError                   -> raised when a book cannot be read at all

``scan()`` and ``extract()`` are built on the same chapter plan, so the
chapters a user ticks in the UI are exactly the chapters that get extracted,
and unselected chapters are never converted.
"""
from __future__ import annotations

import logging
import os
import re
import shutil
import statistics
import sys
import tempfile
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

logger = logging.getLogger(__name__)

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

class ExtractionError(ValueError):
    """Raised when a book cannot be read at all, with a message fit for the UI.

    Examples: a DRM-protected Kindle file, a scanned PDF without a text layer
    (and no OCR), or a ``.mobi`` file when the ``mobi`` package is missing.
    """


@dataclass
class ScannedChapter:
    """Lightweight chapter info returned by scan() for the UI checklist."""
    num:        int
    title:      str
    word_count: int
    href:       str = ""   # EPUB/MOBI only
    # Front/back matter (copyright, dedication, "also by", …). Only listed by
    # ``scan(path, include_matter=True)``; show these UNTICKED. extract() skips
    # them unless they are selected explicitly by title or number.
    probably_matter: bool = False


@dataclass
class ScanResult:
    """Result of a fast pre-scan before any heavy processing."""
    file_type:  str              # "epub" | "mobi" | "pdf" | "docx" | "txt" | "odt"
    has_toc:    bool             # True when chapters were found/detected
    chapters:   list[ScannedChapter] = field(default_factory=list)
    title:      str = ""
    author:     str = ""
    cover_data: bytes | None = None
    # Real page count — PDF only. DOCX/ODT/TXT/EPUB have no fixed pages, so 0.
    page_count: int = 0
    # True only where extract(page_ranges=…) is honoured (PDF).
    supports_page_ranges: bool = False
    # Human-readable problem found while scanning ("" when all is well), e.g.
    # a DRM-protected MOBI or a scanned PDF without a text layer.
    warning:    str = ""


@dataclass
class ExtractedChapter:
    """A fully processed chapter ready for TTS."""
    num:        int
    title:      str
    text:       str          # normalized, TTS-ready text
    sentences:  list[str]
    probably_matter: bool = False   # an explicitly selected front/back-matter entry


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

_EXTENSION_TYPES: dict[str, str] = {
    ".epub": "epub",
    ".mobi": "mobi",
    ".azw":  "mobi",
    ".azw3": "mobi",
    ".prc":  "mobi",
    ".pdf":  "pdf",
    ".docx": "docx",
    ".odt":  "odt",
    ".txt":  "txt",
}


def _detect_type(path: str) -> str:
    ext = Path(path).suffix.lower()
    detected = _EXTENSION_TYPES.get(ext, "")
    if detected:
        return detected

    # Content-based magic detection fallback if file lacks extension
    if os.path.exists(path) and os.path.isfile(path):
        try:
            with open(path, "rb") as f:
                head = f.read(2048)
            if head.startswith(b"%PDF"):
                return "pdf"
            if b"BOOKMOBI" in head or b"TEXtREAd" in head or b"TEXtREAG" in head:
                return "mobi"
        except Exception:
            pass

        import zipfile
        if zipfile.is_zipfile(path):
            try:
                with zipfile.ZipFile(path, "r") as z:
                    names = z.namelist()
                    if "META-INF/container.xml" in names or "mimetype" in names:
                        return "epub"
                    if "word/document.xml" in names:
                        return "docx"
                    if "content.xml" in names:
                        return "odt"
            except Exception:
                pass
    return "unknown"


def _epub_metadata(book) -> tuple[str, str, bytes | None]:
    """Extract title, author, cover from an ebooklib Book object."""
    import ebooklib
    title  = (book.get_metadata("DC", "title")  or [("Unknown Title",  {})])[0][0]
    author = (book.get_metadata("DC", "creator") or [("Unknown Author", {})])[0][0]
    cover_data = None

    # Strategy 1: Check standard item IDs
    for cover_id in ["cover", "cover-image", "cover.jpg", "cover.png", "cover.jpeg", "cover-img"]:
        try:
            cover = book.get_item_with_id(cover_id)
            if cover:
                cover_data = cover.get_content()
                if cover_data:
                    break
        except Exception:
            pass

    # Strategy 2: Check ITEM_IMAGE properties
    if not cover_data:
        for item in book.get_items_of_type(ebooklib.ITEM_IMAGE):
            # EpubImage items may lack get_properties(); guard with getattr
            props = getattr(item, "properties", None)
            if props is None and hasattr(item, "get_properties"):
                try:
                    props = item.get_properties()
                except (AttributeError, TypeError):
                    props = []
            if props and "cover-image" in props:
                cover_data = item.get_content()
                if cover_data:
                    break

    # Strategy 3: Check image filenames containing 'cover'
    if not cover_data:
        for item in book.get_items_of_type(ebooklib.ITEM_IMAGE):
            name = (getattr(item, "file_name", "") or getattr(item, "name", "") or "").lower()
            if "cover" in name:
                cover_data = item.get_content()
                if cover_data:
                    break

    # Strategy 4: Fallback to the first substantial image item
    if not cover_data:
        for item in book.get_items_of_type(ebooklib.ITEM_IMAGE):
            try:
                content = item.get_content()
                if content and len(content) > 1000:
                    cover_data = content
                    break
            except Exception:
                pass

    # Strategy 5: Raw zipfile extraction fallback if book has file_name
    if not cover_data:
        book_file_path = getattr(book, "file_name", None)
        if book_file_path and os.path.exists(book_file_path):
            cover_data = extract_epub_cover_fallback(book_file_path)

    return title, author, cover_data


def extract_epub_cover_fallback(epub_path: str) -> bytes | None:
    """Raw zipfile fallback to extract cover image from an EPUB file."""
    import zipfile
    try:
        with zipfile.ZipFile(epub_path, "r") as z:
            for name in z.namelist():
                lname = name.lower()
                if ("cover" in lname) and lname.endswith((".jpg", ".jpeg", ".png", ".webp")):
                    return z.read(name)
            images = [n for n in z.namelist() if n.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))]
            if images:
                images.sort(key=lambda n: z.getinfo(n).file_size, reverse=True)
                return z.read(images[0])
    except Exception:
        pass
    return None


_ZIP_MAX_UNCOMPRESSED = 512 * 1024 * 1024   # 512 MiB total uncompressed
_ZIP_MAX_MEMBERS = 2000


def _assert_zip_safe(path: str) -> None:
    """Reject archive-based documents whose declared uncompressed payload
    exceeds safe bounds BEFORE any eager whole-archive reader decompresses it.
    CPython's zipfile never inflates beyond a member's declared file_size, so capping
    the declared sizes bounds the memory an archive reader can allocate."""
    import zipfile
    with zipfile.ZipFile(path, "r") as z:
        infos = z.infolist()
        if len(infos) > _ZIP_MAX_MEMBERS:
            raise ValueError(f"Archive too large: too many archive members ({len(infos)})")
        total = 0
        for info in infos:
            total += info.file_size
            if info.file_size > _ZIP_MAX_UNCOMPRESSED:
                raise ValueError(
                    f"Archive too large: member {info.filename!r} declares "
                    f"{info.file_size} bytes uncompressed (limit {_ZIP_MAX_UNCOMPRESSED})"
                )
        if total > _ZIP_MAX_UNCOMPRESSED:
            raise ValueError(
                f"Archive too large when decompressed "
                f"({total} bytes, limit {_ZIP_MAX_UNCOMPRESSED})"
            )


# ══════════════════════════════════════════════════════════════════════════════
# Chapter detection for formats without a TOC (TXT, DOCX, ODT, PDF)
# ══════════════════════════════════════════════════════════════════════════════

_FULL_BOOK_TITLE: str = "Full Book"
_MIN_SECTION_CHARS: int = 50            # shorter "chapters" are headings without text
_MIN_LOOSE_NARRATIVE_WORDS: int = 150   # unheaded text in front of the first chapter
_TOC_BODY_WORDS: int = 12               # a heading with fewer words under it may be a TOC line
_MATTER_MAX_WORDS: int = 3000
_MATTER_MAX_SHARE: float = 0.4
_HEADING_MAX_CHARS: int = 110

_WORD_RE = re.compile(r"\S+")
_ALNUM_RE = re.compile(r"[^\W_]")
_NUMWORD: str = (
    r"(?:one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|"
    r"fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|"
    r"fifty|sixty|seventy|eighty|ninety|hundred|first|second|third|fourth|fifth|"
    r"sixth|seventh|eighth|ninth|tenth|eleventh|twelfth|last|final)"
)
# "Chapter 12", "CHAPTER XII", "Part Two", "Book III", "Act 1: The Storm"
_KW_HEADING = re.compile(
    r"^(?P<kw>chapter|chap\.|part|book|volume|vol\.|act|section|canto|letter|stave|episode|scene)\s+"
    rf"(?P<num>(?:the\s+)?(?:\d{{1,4}}|[ivxlcdm]{{1,8}}|{_NUMWORD}(?:[-\s]{_NUMWORD})?))(?![\w(])"
    r"(?P<rest>.*)$",
    re.I,
)
_NUMWORD_FULL = re.compile(rf"^{_NUMWORD}(?:[-\s]{_NUMWORD})?$", re.I)
_ROMAN = re.compile(r"^M{0,3}(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3})$", re.I)
_HEADING_SEP = re.compile(r"^\s*[:.\-–—|]+\s*")
_SMALL_WORDS: frozenset[str] = frozenset({
    "a", "an", "and", "as", "at", "but", "by", "for", "from", "in", "into", "nor",
    "of", "on", "or", "the", "to", "with", "vs", "de", "la", "le", "du", "von", "van",
})
# Narrable sections that carry no number.
_SOLO_HEADING = re.compile(
    r"^(?:prologue|epilogue|introduction|preface|foreword|afterword|interlude|"
    r"prelude|intermission|coda|finale|overture|postlude|envoi|"
    r"author[’']?s?\s+note|a\s+note\s+from\s+the\s+author|appendix(?:\s+[A-Z0-9]{1,4})?)"
    r"(?:\s*[:.\-–—]\s*\S.*)?$",
    re.I,
)
_BARE_NUMBER = re.compile(r"^(\d{1,3}|[IVXLCDM]{1,7})\.?$")
_PART_TITLE = re.compile(r"^\s*(?:part|book|volume|vol\.|act)\b", re.I)
_CONTENTS_HEADING = re.compile(r"^(?:table\s+of\s+)?contents[.:]?$", re.I)
_TOC_TAIL = re.compile(r"(?:\s*[.\u2026\u00b7_]{2,}[\s.\u2026\u00b7_]*|\s{2,}|\t+)(?:\d{1,4}|[ivxlc]{1,7})\s*$")
_GUTENBERG_MARK = re.compile(
    r"^\s*\*{3}\s*(START|END)\s+OF\s+(?:THE|THIS)\s+PROJECT\s+GUTENBERG\s+E-?BOOK", re.I,
)
_TERMINAL_PUNCT: tuple[str, ...] = (".", ",", ";", "!", "?", ":")


@dataclass
class _Block:
    """One paragraph (or heading) of a document, in reading order."""
    text:  str
    level: int = 0     # 1..9 = styled / outlined heading, 0 = body text
    page:  int = -1    # PDF page index, -1 elsewhere
    title: str = ""    # heading title when it differs from the text (PDF outline)
    words: int = -1    # cached word count (see _block_words)


@dataclass
class _Section:
    """A detected chapter, or flagged front/back matter."""
    title:           str
    text:            str
    probably_matter: bool = False
    num:             int = 0
    word_count:      int = 0
    start:           int = 0    # index of the first block


@dataclass
class _Candidate:
    index: int
    title: str
    kind:  str          # "styled" | "kw" | "solo" | "matter" | "toc" | "num" | "caps"
    value: int = 0      # numeric value of a bare-number heading
    whole_block: bool = True   # False when the heading is the first line of a paragraph


def _word_count(text: str) -> int:
    return len(_WORD_RE.findall(text))


def _block_words(block: _Block) -> int:
    if block.words < 0:
        block.words = _word_count(block.text)
    return block.words


def _collapse_ws(text: str) -> str:
    return " ".join((text or "").split())


def _roman_value(token: str) -> int | None:
    token = token.upper()
    if not token or not _ROMAN.match(token):
        return None
    values = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000}
    total = 0
    for i, ch in enumerate(token):
        val = values[ch]
        total += -val if i + 1 < len(token) and values[token[i + 1]] > val else val
    return total


def _is_title_case(words: list[str]) -> bool:
    """True when every word is capitalised, a number, or a small joining word."""
    for word in words:
        bare = word.strip("\"'“”‘’()[]")
        if not bare:
            continue
        if not (bare[0].isupper() or bare[0].isdigit() or bare.lower() in _SMALL_WORDS):
            return False
    return True


def _keyword_heading(line: str) -> bool:
    """Tells a heading ("Chapter 3 The Return") from prose ("Chapter 3 was hard.")."""
    if len(line) > _HEADING_MAX_CHARS or not line[:1].isupper():
        return False
    match = _KW_HEADING.match(line)
    if not match:
        return False
    token = match.group("num").split()[-1]
    if not (token.isdigit() or _NUMWORD_FULL.match(token) or _roman_value(token) is not None):
        return False
    rest = match.group("rest")
    if not rest.strip(" .:"):
        return True
    separator = _HEADING_SEP.match(rest)
    body = rest[separator.end():] if separator else rest.strip()
    if not body:
        return True
    words = body.split()
    if separator:
        # "Chapter 1. The tide went out, and it did not come back for a long while."
        return not (body.endswith((",", ";")) or (body.endswith(".") and len(words) > 10))
    if not rest[:1].isspace():
        return False
    return len(words) <= 12 and not body.endswith((".", ",", ";")) and _is_title_case(words)


def _toc_key(line: str) -> str:
    """Comparable form of a contents line ("Chapter 1 ....... 7" -> "chapter 1")."""
    line = _TOC_TAIL.sub("", line.strip())
    return _collapse_ws(line).strip(" .:…").casefold()


def _line_candidate(block: _Block, index: int, skip_title: re.Pattern, matter_markers: re.Pattern) -> _Candidate | None:
    """Classifies one block as a heading line, without looking at its neighbours."""
    lines = [ln.strip() for ln in block.text.split("\n") if ln.strip()]
    if not lines:
        return None
    first = lines[0]
    if _keyword_heading(first):
        if len(lines) == 1:
            return _Candidate(index, first, "kw")
        if len(lines) <= 3 and all(len(ln) <= 80 for ln in lines) and not lines[-1].endswith((".", ",", ";")):
            # "CHAPTER I." / "Down the Rabbit-Hole" stacked in one paragraph.
            joiner = " " if first.endswith(_TERMINAL_PUNCT) else ": "
            return _Candidate(index, first + joiner + " ".join(lines[1:]), "kw")
        if len(first) <= 60:
            return _Candidate(index, first, "kw", whole_block=False)
        return None
    if len(lines) != 1 or len(first) > 80 or not (first[:1].isupper() or first[:1].isdigit()):
        return None                     # every other kind of heading is one short line
    if len(first) <= 60 and skip_title.match(first) and (
        not first.endswith((".", ",", ";", "!", "?")) or matter_markers.search(first)
    ):
        return _Candidate(index, first, "matter")
    if len(first) <= 80 and _SOLO_HEADING.match(first) and not first.endswith((",", ";")):
        return _Candidate(index, first, "solo")
    number = _BARE_NUMBER.match(first)
    if number:
        token = number.group(1)
        value = int(token) if token.isdigit() else _roman_value(token)
        if value:
            return _Candidate(index, first.rstrip("."), "num", value=value)
        return None
    letters = [ch for ch in first if ch.isalpha()]
    if 2 <= len(first) <= 60 and len(letters) >= 2 and all(ch.isupper() for ch in letters) \
            and not first.endswith(_TERMINAL_PUNCT + ("\"", "'", "”", "’")):
        return _Candidate(index, first, "caps")
    return None


def _find_contents_block(blocks: list[_Block]) -> tuple[int, int, set[str]] | None:
    """Locates a "Contents" heading and the listing under it.

    Returns (start, stop, keys): the listing occupies ``blocks[start:stop]`` and
    ``keys`` holds the comparable form of every listed title.
    """
    for i, block in enumerate(blocks):
        text = block.text.strip()
        if "\n" in text or not _CONTENTS_HEADING.match(text):
            continue
        keys: set[str] = set()
        count = 0
        j = i + 1
        while j < len(blocks) and count < 500:
            lines = [ln.strip() for ln in blocks[j].text.split("\n") if ln.strip()]
            if not lines:
                j += 1
                continue
            if any(len(ln) > 90 for ln in lines):
                break
            if any(ln.endswith((".", "!", "?", "\"", "”")) and len(ln.split()) > 6 for ln in lines):
                break
            line_keys = [_toc_key(ln) for ln in lines]
            if any(key in keys for key in line_keys):
                break          # the first real heading repeats a listed title
            keys.update(k for k in line_keys if k)
            count += len(lines)
            j += 1
        if count >= 2:
            return i, j, keys
    return None


def _choose_heading_level(headings: list[tuple[int, str]]) -> tuple[int, int]:
    """Picks the heading level that opens a chapter.

    Returns (chapter_level, part_level); 0 means "none". ``part_level`` is set
    when "Part One" / "Part Two" headings sit above the real chapters.
    """
    counts = Counter(level for level, _ in headings)
    for level in sorted(counts):
        if counts[level] < 2:
            continue
        deeper = [lv for lv in sorted(counts) if lv > level and counts[lv] >= 2]
        if deeper and all(_PART_TITLE.match(title) for lv, title in headings if lv == level):
            return deeper[0], level
        return level, 0
    return 0, 0


def _heuristic_candidates(blocks: list[_Block], offset: int = 0) -> tuple[list[_Candidate], tuple[int, int] | None]:
    """Finds heading LINES in unstyled text. Returns (candidates, contents_range)."""
    from audiobook_factory.extractor_engine import _MATTER_MARKERS, _SKIP_TOC_TITLE  # type: ignore

    contents = _find_contents_block(blocks)
    start = contents[1] if contents else 0
    raw = [
        cand for i in range(start, len(blocks))
        if (cand := _line_candidate(blocks[i], i, _SKIP_TOC_TITLE, _MATTER_MARKERS)) is not None
    ]
    by_kind: dict[str, list[_Candidate]] = {}
    for cand in raw:
        by_kind.setdefault(cand.kind, []).append(cand)
    matter, solo, keyword = by_kind.get("matter", []), by_kind.get("solo", []), by_kind.get("kw", [])

    def _words_between(chosen: list[_Candidate]) -> list[int]:
        sizes = []
        for k, cand in enumerate(chosen):
            end = chosen[k + 1].index if k + 1 < len(chosen) else len(blocks)
            sizes.append(sum(_block_words(blocks[i]) for i in range(cand.index, end)))
        return sizes

    listed: list[_Candidate] = []
    if contents:
        keys = contents[2]
        for i in range(start, len(blocks)):
            text = blocks[i].text.strip()
            if "\n" not in text and len(text) <= _HEADING_MAX_CHARS and _toc_key(text) in keys:
                listed.append(_Candidate(i, text, "toc"))

    chosen: list[_Candidate] = []
    if len(listed) >= 2:
        chosen = listed + keyword + solo
    elif len(keyword) >= 2 or (keyword and solo):
        chosen = keyword + solo
    else:
        # Bare numbers ("1", "II.") only count as chapters when they run 1, 2, 3 …
        numbered: list[_Candidate] = []
        for cand in by_kind.get("num", []):
            if cand.value == len(numbered) + 1:
                numbered.append(cand)
        sizes = _words_between(numbered)
        if len(numbered) >= 3 and statistics.median(sizes) >= 600:
            chosen = numbered + solo      # anything denser is page numbering
        else:
            caps = by_kind.get("caps", [])
            sizes = _words_between(caps)
            if len(caps) >= 2 and statistics.median(sizes) >= 300:
                chosen = caps + solo
            elif len(solo) >= 2 or (solo and matter):
                chosen = solo
    seen: set[int] = set()
    merged = []
    for cand in sorted(chosen + matter, key=lambda c: c.index):
        if cand.index not in seen:
            seen.add(cand.index)
            merged.append(cand)
    for cand in merged:
        cand.index += offset
    contents_range = (contents[0] + offset, contents[1] + offset) if contents else None
    return merged, contents_range


def _drop_toc_runs(cands: list[_Candidate], blocks: list[_Block]) -> tuple[list[_Candidate], list[tuple[int, int]]]:
    """Removes heading candidates that are really lines of a contents listing.

    A listing shows up as a run of headings with (almost) nothing between
    them. Returns the surviving candidates and the block ranges of the runs.
    """
    if len(cands) < 3:
        return cands, []

    def _body_words(k: int) -> int:
        cand = cands[k]
        end = cands[k + 1].index if k + 1 < len(cands) else len(blocks)
        total = 0
        for i in range(cand.index + 1, end):
            total += _block_words(blocks[i])
            if total >= _TOC_BODY_WORDS:
                return total            # enough to know this is not a contents line
        if not cand.whole_block:
            total += max(0, _block_words(blocks[cand.index]) - _word_count(cand.title))
        return total

    keys = [_toc_key(c.title) for c in cands]
    keep: list[_Candidate] = []
    runs: list[tuple[int, int]] = []
    k = 0
    while k < len(cands):
        j = k
        while j < len(cands) - 1 and _body_words(j) < _TOC_BODY_WORDS:
            j += 1
        run = list(range(k, j))                 # candidates with nothing under them
        later = set(keys[j + 1:])
        recurring = sum(1 for r in run if keys[r] in set(keys[j:]))
        if len(run) >= 3 or (len(run) >= 2 and recurring == len(run)):
            last = j - 1
            if keys[j] in later and cands[j].whole_block:
                last = j                         # final listing line, with loose text after it
            runs.append((cands[k].index, cands[last].index + 1))
            if last < j:
                keep.append(cands[j])
        else:
            keep.extend(cands[k:j + 1])
        k = j + 1
    return keep, runs


def _join_blocks(blocks: list[_Block]) -> str:
    return "\n\n".join(b.text for b in blocks if b.text.strip())


def _plan_sections(blocks: list[_Block]) -> tuple[list[_Section], bool]:
    """Splits a document into chapters and flagged front/back matter.

    Parameters
    ----------
    blocks : list[_Block]
        Paragraphs in reading order. ``level`` marks styled headings (DOCX and
        ODT heading styles, PDF outline entries, large PDF type).

    Returns
    -------
    tuple[list[_Section], bool]
        Sections in reading order (chapters numbered 1..N, flagged matter
        after them) and whether any chapter structure was detected. With no
        structure the result is a single "Full Book" chapter.
    """
    from audiobook_factory.extractor_engine import (  # type: ignore
        _MATTER_MARKERS, _SKIP_TOC_TITLE, looks_like_matter,
    )

    blocks = [b for b in blocks if b.text.strip() or b.title]
    if not blocks:
        return [], False
    total_words = max(1, sum(_block_words(b) for b in blocks))

    def _full_book() -> list[_Section]:
        return [_Section(_FULL_BOOK_TITLE, _join_blocks(blocks), num=1, word_count=total_words)]

    # Project Gutenberg wrappers: everything outside the START/END marks is licence text.
    core_start, core_end = 0, len(blocks)
    for i, block in enumerate(blocks):
        mark = _GUTENBERG_MARK.match(block.text)
        if mark and mark.group(1).upper() == "START" and i < len(blocks) // 2 + 1:
            core_start = i + 1
        elif mark and mark.group(1).upper() == "END" and i >= core_start:
            core_end = i
            break
    forced: list[tuple[int, int, str]] = []      # (start, stop, title) of forced matter ranges
    if core_start:
        forced.append((0, core_start, "Project Gutenberg Header"))
    if core_end < len(blocks):
        forced.append((core_end, len(blocks), "Project Gutenberg License"))
    core = blocks[core_start:core_end]

    # ── 1. Heading candidates ──
    headings = [(b.level, b.title or _collapse_ws(b.text)) for b in core if b.level > 0]
    cut, part_level = _choose_heading_level(headings)
    contents_range: tuple[int, int] | None = None
    if cut:
        # A shallower heading in front of the first chapter is the book title
        # (title page), not a chapter; later ones ("Part Two") do open a section.
        first_chapter = next(i for i, b in enumerate(core) if b.level == cut)
        cands = [
            _Candidate(core_start + i, (b.title or _collapse_ws(b.text))[:200], "styled")
            for i, b in enumerate(core)
            if (0 < b.level <= cut and i >= first_chapter) or (part_level and b.level == part_level)
        ]
        # Unstyled "About the Author" / "Copyright" lines still end a chapter.
        styled_at = {c.index for c in cands}
        extra, _ = _heuristic_candidates(core, core_start)
        cands += [c for c in extra if c.kind == "matter" and c.index not in styled_at]
        cands.sort(key=lambda c: c.index)
    else:
        cands, contents_range = _heuristic_candidates(core, core_start)
    if contents_range:
        forced.append((contents_range[0], contents_range[1], _collapse_ws(blocks[contents_range[0]].text)))
    # A copyright STATEMENT ("Copyright © 2026 …") under a matter heading is
    # that section's text, not another heading.
    tidy: list[_Candidate] = []
    for cand in cands:
        if cand.kind == "matter" and tidy and _SKIP_TOC_TITLE.match(tidy[-1].title) \
                and _MATTER_MARKERS.search(cand.title):
            continue
        tidy.append(cand)
    cands, toc_runs = _drop_toc_runs(tidy, blocks)
    forced.extend((a, b, "Contents") for a, b in toc_runs)

    if not cands and not forced:
        return _full_book(), False

    # ── 2. Cut the document at the candidates and forced ranges ──
    boundaries: dict[int, tuple[str, bool | None]] = {}     # start -> (title, is_matter)
    for a, b, title in forced:
        boundaries[a] = (title, True)
        if b < len(blocks) and b not in boundaries:
            boundaries[b] = ("", None)                      # loose text after a forced range
    for cand in cands:
        if any(a <= cand.index < b for a, b, _ in forced):
            continue
        boundaries[cand.index] = (cand.title, bool(_SKIP_TOC_TITLE.match(cand.title)))
    if 0 not in boundaries:
        boundaries[0] = ("", None)                          # loose text in front of everything
    starts = sorted(boundaries)

    sections: list[_Section] = []
    for k, start in enumerate(starts):
        stop = starts[k + 1] if k + 1 < len(starts) else len(blocks)
        title, is_matter = boundaries[start]
        text = _join_blocks(blocks[start:stop])
        if not _ALNUM_RE.search(text):
            continue
        words = sum(_block_words(b) for b in blocks[start:stop])
        if is_matter is None:
            # Unheaded text: narrate it only when it is substantial prose.
            first_line = text.split("\n", 1)[0].strip()
            is_matter = words < _MIN_LOOSE_NARRATIVE_WORDS or looks_like_matter(text)
            title = "Front Matter" if is_matter and start == 0 else (
                first_line if len(first_line) <= 80 and not first_line.endswith(_TERMINAL_PUNCT) else ""
            )
            if not title:
                title = "Untitled Section" if is_matter else ("Opening" if start == 0 else "Untitled Section")
        elif is_matter and words > _MATTER_MAX_WORDS and words > _MATTER_MAX_SHARE * total_words \
                and not title.startswith("Project Gutenberg"):
            is_matter = False        # a "Dedication" heading in front of a heading-less novel
        sections.append(_Section(title, text, bool(is_matter), word_count=words, start=start))

    # ── 3. A heading with no text under it joins the chapter that follows ──
    merged: list[_Section] = []
    carry: _Section | None = None
    for section in sections:
        if carry is not None:
            if not section.probably_matter:
                section.text = carry.text + "\n\n" + section.text
                section.word_count += carry.word_count
                section.start = carry.start
            carry = None
        if not section.probably_matter and len(section.text.strip()) < _MIN_SECTION_CHARS:
            carry = section
            continue
        merged.append(section)

    chapters = [s for s in merged if not s.probably_matter]
    if not chapters:
        return _full_book(), False
    detected = bool(cands) or len(chapters) > 1
    if len(chapters) == 1 and chapters[0].title in ("Opening", "Untitled Section", ""):
        chapters[0].title = _FULL_BOOK_TITLE
    number = 0
    for section in chapters:
        number += 1
        section.num = number
    for section in merged:
        if section.probably_matter:
            number += 1
            section.num = number
    return merged, detected


def _scanned_from_sections(sections: list[Any], include_matter: bool) -> list[ScannedChapter]:
    return [
        ScannedChapter(
            num=s.num, title=s.title, word_count=s.word_count,
            href=getattr(s, "href", ""), probably_matter=s.probably_matter,
        )
        for s in sections if include_matter or not s.probably_matter
    ]


def _chapters_from_sections(
    sections: list[_Section],
    selections: list[int] | list[str] | None,
    normalizer,
    log: Callable[[str], None],
    *,
    fix_kerning: bool = False,
) -> list[ExtractedChapter]:
    """Normalises and sentence-splits only the SELECTED sections."""
    from audiobook_factory.extractor_engine import select_sections  # type: ignore
    from audiobook_factory.text_processing import smart_sentence_splitter

    results: list[ExtractedChapter] = []
    for section in select_sections(sections, selections):
        title = "" if section.title == _FULL_BOOK_TITLE else section.title
        if fix_kerning:
            text = normalizer.normalize(section.text, title=title, ocr_block_texts=[], fix_kerning=True)
        else:
            text = normalizer.normalize(section.text, title=title, ocr_block_texts=[])
        if not text.strip():
            continue
        results.append(ExtractedChapter(
            num=section.num, title=section.title, text=text,
            sentences=smart_sentence_splitter(text),
            probably_matter=section.probably_matter,
        ))
        log(f"  ✓ Extracted chapter {section.num}: {section.title}")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# scan() — fast, no OCR, just chapter discovery for the UI
# ══════════════════════════════════════════════════════════════════════════════

def scan(path: str, *, include_matter: bool = False) -> ScanResult:
    """Fast pre-scan: lists the chapters ``extract()`` would return.

    Never runs OCR or Docling — this must be snappy for the UI.

    Parameters
    ----------
    path : str
        Book file (.epub, .mobi/.azw3, .pdf, .docx, .odt, .txt).
    include_matter : bool
        Also list front/back matter (copyright, dedication, "also by", …) in
        reading order, each with ``probably_matter=True``. The UI should show
        those entries unticked.

    Returns
    -------
    ScanResult
        ``chapters`` matches ``extract(path)`` title for title. ``has_toc`` is
        True when a chapter structure was found; otherwise ``chapters`` holds
        the single "Full Book" entry.
    """
    ftype = _detect_type(path)

    if ftype == "epub":
        return _scan_epub(path, ftype, include_matter=include_matter)
    elif ftype == "mobi":
        return _scan_mobi(path, include_matter=include_matter)
    elif ftype == "pdf":
        return _scan_pdf(path, include_matter=include_matter)
    elif ftype == "docx":
        return _scan_docx(path, include_matter=include_matter)
    elif ftype == "odt":
        return _scan_odt(path, include_matter=include_matter)
    elif ftype == "txt":
        return _scan_txt(path, include_matter=include_matter)
    return ScanResult(file_type=ftype, has_toc=False, page_count=0)


def _scan_epub(path: str, ftype: str, *, include_matter: bool = False) -> ScanResult:
    try:
        from ebooklib import epub
        from audiobook_factory.extractor_engine import DocumentIngestor  # type: ignore

        _assert_zip_safe(path)
        book = epub.read_epub(path)
        title, author, cover_data = _epub_metadata(book)
        sections, _skipped, _toc = DocumentIngestor().plan_epub(book)
        chapters = _scanned_from_sections(sections, include_matter)
        return ScanResult(
            file_type=ftype, has_toc=any(not s.probably_matter for s in sections),
            chapters=chapters, title=title, author=author, cover_data=cover_data,
        )
    except Exception as e:
        logger.warning("EPUB scan failed for %s: %s", path, e)
        return ScanResult(file_type=ftype, has_toc=False, warning=str(e))


def _scan_txt(path: str, *, include_matter: bool = False) -> ScanResult:
    try:
        sections, detected = _plan_sections(_txt_blocks(read_text_file(path)))
    except Exception as e:
        logger.warning("TXT scan failed for %s: %s", path, e)
        return ScanResult(file_type="txt", has_toc=False, warning=str(e))
    return ScanResult(
        file_type="txt", has_toc=detected,
        chapters=_scanned_from_sections(sections, include_matter),
    )


def _scan_pdf(path: str, *, include_matter: bool = False) -> ScanResult:
    result = ScanResult(file_type="pdf", has_toc=False, supports_page_ranges=True)
    doc = None
    try:
        doc = _open_pdf(path)
        result.page_count = doc.page_count
        meta = doc.metadata or {}
        result.title = (meta.get("title") or "").strip()
        result.author = (meta.get("author") or "").strip()
        blocks, _truncated = _pdf_blocks(doc, range(min(doc.page_count, _PDF_MAX_PAGES)), lambda _msg: None)
        if _pdf_is_textless(blocks, min(doc.page_count, _PDF_MAX_PAGES)):
            result.warning = _SCANNED_PDF_MESSAGE
            return result
        sections, detected = _pdf_sections(doc, blocks)
        result.has_toc = detected
        result.chapters = _scanned_from_sections(sections, include_matter)
    except Exception as e:
        logger.warning("PDF scan failed for %s: %s", path, e)
        result.warning = result.warning or str(e)
    finally:
        if doc is not None:
            doc.close()
    return result


def _scan_docx(path: str, *, include_matter: bool = False) -> ScanResult:
    # DOCX has no fixed pages, so page_count stays 0 and page ranges are not offered.
    result = ScanResult(file_type="docx", has_toc=False)
    try:
        _assert_zip_safe(path)
        blocks, result.title, result.author = _docx_blocks(path)
        sections, result.has_toc = _plan_sections(blocks)
        result.chapters = _scanned_from_sections(sections, include_matter)
    except Exception as e:
        logger.warning("DOCX scan failed for %s: %s", path, e)
        result.warning = str(e)
    return result


def _scan_odt(path: str, *, include_matter: bool = False) -> ScanResult:
    # ODT has no fixed pages either.
    result = ScanResult(file_type="odt", has_toc=False)
    try:
        _assert_zip_safe(path)
        blocks, result.title, result.author = _odt_blocks(path)
        sections, result.has_toc = _plan_sections(blocks)
        result.chapters = _scanned_from_sections(sections, include_matter)
    except Exception as e:
        logger.warning("ODT scan failed for %s: %s", path, e)
        result.warning = str(e)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# extract() — full extraction with Docling + OCR + normalization
# ══════════════════════════════════════════════════════════════════════════════

def extract(
    path: str,
    selections: list[int] | list[str] | None = None,
    *,
    enable_ocr: bool = False,
    page_ranges: list[tuple[int, int]] | None = None,
    log_fn=None,
    language: str | None = None,
) -> tuple[list[ExtractedChapter], bytes | None]:
    """Full extraction. Returns (chapters, cover_data).

    Parameters
    ----------
    path : str
        Book file (.epub, .mobi/.azw3, .pdf, .docx, .odt, .txt).
    selections : list[int] | list[str] | None
        Chapter titles (as listed by ``scan()``) or chapter numbers. ``None``
        extracts every chapter except flagged front/back matter; flagged
        entries are only extracted when selected explicitly. Unselected
        chapters are never converted.
    enable_ocr : bool
        OCR images inside EPUB/MOBI chapters, and scanned PDF pages, with
        EasyOCR when it is installed.
    page_ranges : list[tuple[int, int]] | None
        PDF only: 1-based inclusive page ranges, one chapter per range. Takes
        precedence over ``selections``. Ignored for every other format.
    log_fn : callable | None
        Receives progress lines; defaults to ``print``.
    language : str | None
        Book language ("en", "fr", "zh-TW", …) for OCR. EPUB/MOBI fall back
        to the language in the book's metadata.

    Returns
    -------
    tuple[list[ExtractedChapter], bytes | None]
        Chapters in reading order and the cover image when the format has one.

    Raises
    ------
    ExtractionError
        The book cannot be read: DRM-protected Kindle file, scanned PDF with
        no usable OCR, missing ``mobi`` package.
    """
    def log(msg):
        if log_fn:
            log_fn(msg)
        else:
            print(msg)

    ftype = _detect_type(path)

    if ftype == "epub":
        return _extract_epub(path, selections, enable_ocr=enable_ocr, log=log, language=language)
    elif ftype == "mobi":
        return _extract_mobi(path, selections, enable_ocr=enable_ocr, log=log,
                             language=language, page_ranges=page_ranges)
    elif ftype == "txt":
        return _extract_txt(path, log=log, selections=selections), None
    else:
        return _extract_paged(path, ftype, page_ranges, log=log, selections=selections,
                              enable_ocr=enable_ocr, language=language), None


# ── EPUB extraction ───────────────────────────────────────────────────────────

def _chapters_from_items(items, log) -> list[ExtractedChapter]:
    results: list[ExtractedChapter] = []
    for ch in items:
        log(f"  ✓ Extracted chapter {ch.num}: {ch.title}")
        results.append(ExtractedChapter(
            num=ch.num,
            title=ch.title,
            text=ch.normalized,
            sentences=ch.sentences,
            probably_matter=getattr(ch, "probably_matter", False),
        ))
    return results


def _extract_epub(
    path: str,
    selections: list[int] | list[str] | None,
    *,
    enable_ocr: bool,
    log,
    language: str | None = None,
) -> tuple[list[ExtractedChapter], bytes | None]:
    _assert_zip_safe(path)
    from ebooklib import epub
    from audiobook_factory.extractor_engine import (  # type: ignore
        DocumentIngestor, MLClassifier, TextNormalizer
    )

    # The archive is checked and opened exactly once; the same book object
    # serves chapter planning, conversion and the cover.
    book = epub.read_epub(path)

    # Selections can be:
    #   None        → every chapter that is not flagged front/back matter
    #   list[int]   → chapter numbers as reported by scan()
    #   list[str]   → chapter titles as reported by scan()
    # They are applied BEFORE conversion, so unselected chapters cost nothing.
    chapters, _skipped, _toc_entries = DocumentIngestor().ingest_epub(
        path, MLClassifier(), TextNormalizer(), enable_ocr=enable_ocr,
        selections=selections, book=book, language=language,
    )

    cover_data: bytes | None = None
    try:
        _, _, cover_data = _epub_metadata(book)
    except Exception as exc:
        logger.debug("Cover extraction failed for %s: %s", path, exc)

    return _chapters_from_items(chapters, log), cover_data


# ── MOBI / AZW3 extraction ────────────────────────────────────────────────────

_MOBI_MAX_BYTES: int = 512 * 1024 * 1024
_MOBI_INSTALL_HINT: str = (
    "Reading .mobi/.azw3 files needs the 'mobi' package: run `pip install mobi` "
    "(or convert the book to EPUB with Calibre)."
)
_MOBI_DRM_MESSAGE: str = (
    "This Kindle book is DRM-protected and cannot be read. Use a DRM-free copy "
    "(EPUB, or a MOBI/AZW3 you own without DRM)."
)
_FILEPOS_LINK = re.compile(
    r"<a\b[^>]*?\bhref\s*=\s*[\"'](?:[^\"'#]*)#([^\"']+)[\"'][^>]*>(.*?)</a>", re.I | re.S,
)
_TAG_RE = re.compile(r"<[^>]+>")
_HEADING_TAG = re.compile(r"<h([1-3])\b[^>]*>(.*?)</h\1\s*>", re.I | re.S)
_HTML_BODY = re.compile(r"<body\b[^>]*>(.*)</body\s*>", re.I | re.S)


def _mobi_precheck(path: str) -> None:
    """Rejects files that are not MOBI, are too large, or are DRM-protected."""
    import struct
    size = os.path.getsize(path)
    if size > _MOBI_MAX_BYTES:
        raise ExtractionError(f"MOBI file too large ({size} bytes, limit {_MOBI_MAX_BYTES}).")
    with open(path, "rb") as f:
        head = f.read(78)
        if len(head) < 78 or head[60:68] not in (b"BOOKMOBI", b"TEXtREAd"):
            raise ExtractionError(
                "Not a valid MOBI/AZW file (missing BOOKMOBI signature). "
                "If this is an EPUB with the wrong extension, rename it to .epub."
            )
        (count,) = struct.unpack(">H", head[76:78])
        if count < 1:
            raise ExtractionError("Not a valid MOBI/AZW file (no records).")
        table = f.read(8)
        if len(table) < 8:
            raise ExtractionError("Not a valid MOBI/AZW file (truncated record table).")
        (offset,) = struct.unpack(">L", table[:4])
        f.seek(offset)
        record0 = f.read(16)
    if len(record0) >= 14 and struct.unpack(">H", record0[12:14])[0] != 0:
        raise ExtractionError(_MOBI_DRM_MESSAGE)


@contextmanager
def _unpacked_mobi(path: str) -> Iterator[tuple[str, str]]:
    """Unpacks a MOBI/AZW3 with the pure-Python ``mobi`` package (KindleUnpack).

    Yields (kind, file): ("epub", …) for KF8 books, ("html", …) for classic
    MOBI and ("pdf", …) for print replicas. The temporary directory is
    removed when the block exits.
    """
    _mobi_precheck(path)
    try:
        from mobi.kindleunpack import unpackBook  # type: ignore
    except ImportError as exc:
        raise ExtractionError(_MOBI_INSTALL_HINT) from exc

    os.makedirs(os.path.join(_ROOT, "temp"), exist_ok=True)
    workdir = tempfile.mkdtemp(prefix="mobi_", dir=os.path.join(_ROOT, "temp"))
    try:
        try:
            unpackBook(path, workdir, epubver="A")
        except Exception as exc:
            message = str(exc)
            if "encrypt" in message.lower() or "drm" in message.lower():
                raise ExtractionError(_MOBI_DRM_MESSAGE) from exc
            raise ExtractionError(f"Could not unpack this MOBI/AZW file: {message or type(exc).__name__}") from exc
        base = os.path.splitext(os.path.basename(path))[0]
        candidates = (
            ("epub", os.path.join(workdir, "mobi8", base + ".epub")),
            ("html", os.path.join(workdir, "mobi7", "book.html")),
            ("pdf",  os.path.join(workdir, base + ".001.pdf")),
        )
        for kind, candidate in candidates:
            if os.path.exists(candidate):
                yield kind, candidate
                return
        raise ExtractionError("Could not unpack this MOBI/AZW file: no readable content found.")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def _decode_html(data: bytes) -> str:
    try:
        return data.decode("utf-8")
    except UnicodeDecodeError:
        return data.decode("cp1252", errors="replace")


def _opf_field(opf: str, name: str) -> str:
    import html as html_lib
    match = re.search(rf"<dc:{name}\b[^>]*>(.*?)</dc:{name}>", opf, re.I | re.S)
    return html_lib.unescape(_collapse_ws(_TAG_RE.sub("", match.group(1)))) if match else ""


def _mobi7_plan(html_path: str):
    """Plans the chapters of a classic (MOBI 6/7) book unpacked to one HTML file.

    Returns (ingestor, sections, title, author, language, cover_bytes).
    """
    import html as html_lib
    from audiobook_factory.extractor_engine import (  # type: ignore
        DocumentIngestor, HtmlDoc, TocEntry, _SKIP_TOC_TITLE, normalize_href,
    )

    folder = os.path.dirname(html_path)
    with open(html_path, "rb") as f:
        markup = _decode_html(f.read())
    body = _HTML_BODY.search(markup)
    body_html = body.group(1) if body else markup

    opf = ""
    opf_path = os.path.join(folder, "content.opf")
    if os.path.exists(opf_path):
        with open(opf_path, "rb") as f:
            opf = _decode_html(f.read())
    title, author, language = _opf_field(opf, "title"), _opf_field(opf, "creator"), _opf_field(opf, "language")

    entries: list[TocEntry] = []

    def _add(label: str, anchor: str, depth: int = 0) -> None:
        label = html_lib.unescape(_collapse_ws(_TAG_RE.sub("", label)))
        if not label or not anchor:
            return
        cls = "skip" if _SKIP_TOC_TITLE.match(label) else "chapter"
        entries.append(TocEntry(label, "book.html", cls, anchor=anchor, depth=depth))

    # 1. NCX written by KindleUnpack from the book's index records.
    ncx_path = os.path.join(folder, "toc.ncx")
    if os.path.exists(ncx_path):
        with open(ncx_path, "rb") as f:
            ncx = _decode_html(f.read())
        for point in re.finditer(
            r"<navLabel>\s*<text>(.*?)</text>\s*</navLabel>\s*<content\s+src\s*=\s*[\"']([^\"']+)[\"']",
            ncx, re.I | re.S,
        ):
            _add(point.group(1), normalize_href(point.group(2))[1])

    # 2. The inline contents page: the longest run of in-book links that have
    #    nothing but markup (or a page number) between them. Links scattered
    #    through prose — footnote markers, cross references — never qualify,
    #    so narrative text between two links is never mistaken for a listing.
    links = list(_FILEPOS_LINK.finditer(body_html))
    runs: list[list[re.Match]] = []
    for link in links:
        between = _TAG_RE.sub("", body_html[runs[-1][-1].end():link.start()]) if runs else ""
        if runs and link.start() - runs[-1][-1].end() < 600 and len("".join(between.split())) <= 12:
            runs[-1].append(link)
        else:
            runs.append([link])
    listing = max(runs, key=len) if runs else []
    worded = [link for link in listing if len(_ALNUM_RE.findall(_TAG_RE.sub("", link.group(2)))) >= 3]
    if len(listing) >= 3 and len(worded) >= 0.6 * len(listing):
        if len(entries) < 2:
            entries = []
            for link in listing:
                _add(link.group(2), html_lib.unescape(link.group(1)))
        # The listing itself must never be narrated.
        body_html = body_html[:listing[0].start()] + body_html[listing[-1].end():]

    # 3. Where the contents page starts (so its heading is not narrated either).
    guide_toc = re.search(
        r"<reference\b[^>]*\btype\s*=\s*[\"']toc[\"'][^>]*>", opf, re.I,
    )
    if guide_toc:
        target = re.search(r"href\s*=\s*[\"'][^\"'#]*#([^\"']+)[\"']", guide_toc.group(0), re.I)
        if target and not any(e.anchor == target.group(1) for e in entries):
            entries.append(TocEntry("Contents", "book.html", "skip", anchor=target.group(1)))

    cover: bytes | None = None
    images = os.path.join(folder, "Images")
    if os.path.isdir(images):
        names = sorted(os.listdir(images))
        chosen = next((n for n in names if "cover" in n.lower()), None)
        if chosen:
            with open(os.path.join(images, chosen), "rb") as f:
                cover = f.read()

    ingestor = DocumentIngestor()
    sections, _skipped = ingestor.plan_html_sections([HtmlDoc(name="book.html", html=body_html)], entries)
    return ingestor, sections, title, author, language, cover


def _scan_mobi(path: str, *, include_matter: bool = False) -> ScanResult:
    try:
        with _unpacked_mobi(path) as (kind, unpacked):
            if kind == "epub":
                result = _scan_epub(unpacked, "mobi", include_matter=include_matter)
            elif kind == "pdf":
                result = _scan_pdf(unpacked, include_matter=include_matter)
                result.file_type = "mobi"
                result.supports_page_ranges = True
            else:
                _ingestor, sections, title, author, _lang, cover = _mobi7_plan(unpacked)
                result = ScanResult(
                    file_type="mobi", has_toc=any(not s.probably_matter for s in sections),
                    chapters=_scanned_from_sections(sections, include_matter),
                    title=title, author=author, cover_data=cover,
                )
            return result
    except Exception as e:
        logger.warning("MOBI scan failed for %s: %s", path, e)
        return ScanResult(file_type="mobi", has_toc=False, warning=str(e))


def _extract_mobi(
    path: str,
    selections: list[int] | list[str] | None,
    *,
    enable_ocr: bool,
    log,
    language: str | None = None,
    page_ranges: list[tuple[int, int]] | None = None,
) -> tuple[list[ExtractedChapter], bytes | None]:
    from audiobook_factory.extractor_engine import TextNormalizer, select_sections  # type: ignore

    with _unpacked_mobi(path) as (kind, unpacked):
        if kind == "epub":
            return _extract_epub(unpacked, selections, enable_ocr=enable_ocr, log=log, language=language)
        if kind == "pdf":
            return _extract_paged(unpacked, "pdf", page_ranges, log=log, selections=selections,
                                  enable_ocr=enable_ocr, language=language), None
        ingestor, sections, _title, _author, book_language, cover = _mobi7_plan(unpacked)
        # Images of a classic MOBI live on disk, not in an EPUB container, so
        # image OCR is not available on this path.
        items, _skipped = ingestor.convert_sections(
            select_sections(sections, selections), TextNormalizer(),
            ocr_book=None, language=language or book_language or None,
        )
        return _chapters_from_items(items, log), cover


# ── TXT extraction ────────────────────────────────────────────────────────────

def read_text_file(path: str) -> str:
    """Reads a plain-text book, detecting its encoding.

    UTF-8 is tried first (a BOM is dropped). Anything else is detected with
    chardet when available and otherwise read as Windows-1252, so legacy files
    keep their apostrophes and quotes instead of turning into U+FFFD.

    Parameters
    ----------
    path : str
        Path to the text file.

    Returns
    -------
    str
        Decoded file contents.
    """
    with open(path, "rb") as f:
        data = f.read()
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        pass
    if data[:2] in (b"\xff\xfe", b"\xfe\xff"):
        try:
            return data.decode("utf-16")
        except UnicodeDecodeError:
            pass
    try:
        import chardet  # type: ignore
        guess = chardet.detect(data[:200_000])
        encoding = guess.get("encoding")
        if encoding and (guess.get("confidence") or 0.0) >= 0.5:
            return data.decode(encoding, errors="replace")
    except Exception as exc:
        logger.debug("Encoding detection failed for %s: %s", path, exc)
    return data.decode("cp1252", errors="replace")


_PARAGRAPH_BREAK = re.compile(r"\n[ \t]*(?:\n[ \t]*)+")
_TXT_NOTE_MARK = re.compile(r"(?<=[^\s\[(])\[\d{1,3}\]")       # "tables[1]"
_TXT_NOTE_BODY = re.compile(r"^\[\d{1,3}\][ \t]+\S")            # "[1] The Salt Road …"
_MD_ATX_HEADING = re.compile(r"^(#{1,6})[ \t]+(\S.*?)[ \t#]*$")


def _txt_blocks(raw: str) -> list[_Block]:
    """Paragraphs of a plain-text book (blank-line separated, wraps kept).

    Markdown-flavoured text is common in .txt books, so list bullets, pipe
    tables and links are flattened here (a Python pre-pass, see
    ``TextNormalizer.strip_markdown_structure``), ``# Heading`` lines become
    styled headings, and bracketed footnote markers and footnote paragraphs
    are dropped.
    """
    from audiobook_factory.extractor_engine import TextNormalizer  # type: ignore

    raw = raw.replace("\r\n", "\n").replace("\r", "\n")
    raw = TextNormalizer().strip_markdown_structure(raw)
    blocks: list[_Block] = []
    for part in _PARAGRAPH_BREAK.split(raw):
        text = part.strip("\n")
        if not text.strip() or _TXT_NOTE_BODY.match(text.lstrip()):
            continue
        text = _TXT_NOTE_MARK.sub("", text)
        heading = _MD_ATX_HEADING.match(text.strip()) if "\n" not in text.strip() else None
        if heading:
            blocks.append(_Block(text, level=len(heading.group(1)), title=heading.group(2)))
        else:
            blocks.append(_Block(text))
    return blocks


def _extract_txt(
    path: str, *, log, selections: list[int] | list[str] | None = None,
) -> list[ExtractedChapter]:
    from audiobook_factory.extractor_engine import TextNormalizer  # type: ignore

    log("  Reading TXT file...")
    sections, _detected = _plan_sections(_txt_blocks(read_text_file(path)))
    return _chapters_from_sections(sections, selections, TextNormalizer(), log)


# ── PDF / DOCX / ODT extraction ───────────────────────────────────────────────

def _extract_paged(
    path: str,
    ftype: str,
    page_ranges: list[tuple[int, int]] | None,
    *,
    log,
    selections: list[int] | list[str] | None = None,
    enable_ocr: bool = False,
    language: str | None = None,
) -> list[ExtractedChapter]:
    from audiobook_factory.extractor_engine import TextNormalizer  # type: ignore

    normalizer = TextNormalizer()

    if ftype == "pdf":
        # No DocumentIngestor here: this path never touches Docling.
        return _extract_pdf_ranges(path, page_ranges, None, normalizer, log,
                                   selections=selections, enable_ocr=enable_ocr, language=language)
    elif ftype == "docx":
        return _extract_docx(path, page_ranges, normalizer, log, selections=selections)
    elif ftype == "odt":
        return _extract_odt(path, page_ranges, normalizer, log, selections=selections)
    return []


# ── PDF: positional text extraction ──────────────────────────────────────────

_PDF_MAX_RANGES: int = 64
_PDF_MAX_PAGES: int = 5_000
_PDF_MAX_EXTRACT_CHARS: int = 20_000_000
_PDF_MARGIN_BAND: float = 0.12          # top/bottom share of a page where headers/footers live
_PDF_HEADING_RATIO: float = 1.15        # type this much larger than body text is a heading
_PDF_DROPCAP_RATIO: float = 1.5
_SCANNED_PDF_MESSAGE: str = (
    "This PDF has no text layer (it looks like a scan), so there is nothing to narrate. "
    "Run it through OCR first (e.g. `ocrmypdf in.pdf out.pdf`), or enable OCR with EasyOCR installed."
)
_PAGE_NUMBER_LINE = re.compile(
    r"^\W*(?:(?:page|p\.|pg\.?)\s*)?(?:\d{1,4}|[ivxlc]{1,7})(?:\s*(?:of|/)\s*\d{1,4})?\W*$", re.I,
)
_EDGE_NUMBER = re.compile(r"^(?:(\d{1,4})\s*[|·•–—-]?\s+)?(.*?)(?:\s+[|·•–—-]?\s*(\d{1,4}))?$")
_FOOTNOTE_START = re.compile(r"^(?:\d{1,3}|[*†‡§]+)[.)]?\s*\S")
_NOTE_MARK = re.compile(r"^\s*(?:\d{1,3}|[*†‡§]+)\s*$")
_PARA_END = re.compile(r"(?:[.!?\u2026:;][\"'\u201d\u2019)\]]*|[\u2014\u2013-][\"'\u201d\u2019])$")
_LOWER_START = re.compile(r"^[\"'“‘(]?[a-zß-ÿ]")
_HYPHEN_END = re.compile(r"[a-zß-ÿ]-$")


@dataclass
class _PdfLine:
    text:  str
    x0:    float
    y0:    float
    x1:    float
    y1:    float
    size:  float
    block: int


@dataclass
class _PdfPage:
    index:  int
    height: float
    lines:  list[_PdfLine] | None      # None when only plain text is available
    plain:  str = ""

    def char_count(self) -> int:
        if self.lines is None:
            return len(self.plain)
        return sum(len(line.text) for line in self.lines)


def _import_fitz():
    """PyMuPDF under whichever name is available (``pymupdf``, or legacy ``fitz``)."""
    if "fitz" in sys.modules:
        return sys.modules["fitz"]          # already loaded under the legacy name
    try:
        import pymupdf  # type: ignore
        return pymupdf
    except ImportError:
        import fitz  # type: ignore
        return fitz


def _open_pdf(path: str):
    return _import_fitz().open(path)


def _pdf_text_flags() -> int:
    fitz = _import_fitz()
    # No ligatures (so "ﬁ" comes out as "fi"), no images, hyphens at line ends removed.
    return (
        getattr(fitz, "TEXT_PRESERVE_WHITESPACE", 2)
        | getattr(fitz, "TEXT_MEDIABOX_CLIP", 64)
        | getattr(fitz, "TEXT_DEHYPHENATE", 16)
    )


def _pdf_load_page(doc, index: int, flags: int) -> _PdfPage:
    """Reads one page as positioned lines (falls back to plain text)."""
    page = doc[index]
    data = page.get_text("dict", flags=flags)
    if not isinstance(data, dict):
        # A text-only backend: positions are unknown, keep the text as it is.
        return _PdfPage(index, 0.0, None, str(data or ""))
    height = float(data.get("height") or 0.0) or 1.0
    lines: list[_PdfLine] = []
    for block_no, block in enumerate(data.get("blocks", [])):
        if block.get("type", 0) != 0:
            continue
        previous: _PdfLine | None = None
        for line in block.get("lines", []):
            direction = line.get("dir") or (1.0, 0.0)
            if abs(direction[1]) > 0.5:
                continue                      # vertical watermark / margin text
            spans = line.get("spans", [])
            if not spans:
                continue
            main_size = max(spans, key=lambda s: len(s.get("text", "").strip()))["size"]
            parts = []
            for span in spans:
                text = span.get("text", "")
                superscript = (span.get("flags", 0) & 1) or span["size"] <= 0.8 * main_size
                if superscript and _NOTE_MARK.match(text) and len(spans) > 1:
                    continue                  # footnote marker
                parts.append(text)
            text = "".join(parts)
            if not text.strip():
                continue
            x0, y0, x1, y1 = line["bbox"]
            current = _PdfLine(text.strip(), x0, y0, x1, y1, float(main_size), block_no)
            if previous is not None and current.x0 >= previous.x1 - 1.0:
                overlap = min(previous.y1, current.y1) - max(previous.y0, current.y0)
                if overlap > 0.5 * min(previous.y1 - previous.y0, current.y1 - current.y0):
                    if _NOTE_MARK.match(current.text) and current.size <= 0.8 * previous.size:
                        continue              # raised footnote marker after a word
                    # Same baseline, split only by a font change: one line.
                    glue = "" if current.x0 - previous.x1 < 0.15 * current.size else " "
                    previous.text = f"{previous.text}{glue}{current.text}"
                    previous.x1, previous.y0, previous.y1 = current.x1, min(previous.y0, y0), max(previous.y1, y1)
                    continue
            lines.append(current)
            previous = current
    return _PdfPage(index, height, lines)


def _running_key(text: str) -> str:
    return _collapse_ws(text).casefold()


def _strip_running_lines(pages: list[_PdfPage], body_size: float, context: list[_PdfPage] | None = None) -> None:
    """Deletes running headers, footers and page numbers BY POSITION.

    Only lines inside the top/bottom margin band are considered. A line goes
    when it is nothing but a page number, or when the same text (optionally
    with a page number at either edge) sits in that band on several pages.
    Body text, dialogue and headings outside the bands are never touched.
    """
    sample = [p for p in pages + (context or []) if p.lines is not None]
    needed = 3 if len(sample) > 5 else 2

    def _band(page: _PdfPage, line: _PdfLine) -> str:
        if line.y1 <= page.height * _PDF_MARGIN_BAND:
            return "top"
        if line.y0 >= page.height * (1.0 - _PDF_MARGIN_BAND):
            return "bottom"
        return ""

    exact: dict[tuple[str, str], set[int]] = {}
    edged: dict[tuple[str, str], list[tuple[int, int]]] = {}
    for page in sample:
        for line in page.lines or []:
            band = _band(page, line)
            if not band or line.size > 1.1 * body_size:
                continue
            key = _running_key(line.text)
            exact.setdefault((band, key), set()).add(page.index)
            match = _EDGE_NUMBER.match(_collapse_ws(line.text))
            if match and (match.group(1) or match.group(3)) and _ALNUM_RE.search(match.group(2) or ""):
                number = int(match.group(1) or match.group(3))
                edged.setdefault((band, match.group(2).casefold()), []).append((page.index, number))

    def _counts_like_pages(occurrences: list[tuple[int, int]]) -> bool:
        pages_seen = {p for p, _ in occurrences}
        if len(pages_seen) < needed:
            return False
        ordered = sorted(occurrences)
        steps = [(b[0] - a[0], b[1] - a[1]) for a, b in zip(ordered, ordered[1:]) if b[0] != a[0]]
        return bool(steps) and sum(1 for dp, dn in steps if dp == dn) >= 0.8 * len(steps)

    for page in pages:
        if page.lines is None:
            continue
        kept = []
        for line in page.lines:
            band = _band(page, line)
            if band:
                if _PAGE_NUMBER_LINE.match(line.text):
                    continue
                if line.size <= 1.1 * body_size:
                    if len(exact.get((band, _running_key(line.text)), ())) >= needed:
                        continue
                    match = _EDGE_NUMBER.match(_collapse_ws(line.text))
                    if match and (match.group(1) or match.group(3)) and not _keyword_heading(line.text) \
                            and _counts_like_pages(edged.get((band, (match.group(2) or "").casefold()), [])):
                        continue
            kept.append(line)
        # Footnotes: small type at the foot of the page that opens with a marker.
        tail = len(kept)
        while tail > 0 and kept[tail - 1].size < 0.92 * body_size and kept[tail - 1].y0 > page.height * 0.5:
            tail -= 1
        if tail < len(kept) and _FOOTNOTE_START.match(kept[tail].text):
            kept = kept[:tail]
        page.lines = kept


def _pdf_body_size(pages: list[_PdfPage]) -> float:
    weights: Counter[float] = Counter()
    for page in pages:
        for line in page.lines or []:
            weights[round(line.size * 2) / 2] += len(line.text)
    return weights.most_common(1)[0][0] if weights else 12.0


def _pdf_page_blocks(page: _PdfPage, body_size: float) -> list[tuple[str, float]]:
    """Paragraphs of one page as (text, font size)."""
    if page.lines is None:
        return [(page.plain, body_size)] if page.plain.strip() else []
    paragraphs: list[tuple[str, float]] = []
    current: list[_PdfLine] = []
    drop_cap = ""

    by_block: dict[int, list[_PdfLine]] = {}
    for line in page.lines:
        by_block.setdefault(line.block, []).append(line)
    left = {b: min(ln.x0 for ln in lines) for b, lines in by_block.items()}
    right = {b: max(ln.x1 for ln in lines) for b, lines in by_block.items()}

    def _flush() -> None:
        if not current:
            return
        text = current[0].text
        for line in current[1:]:
            if text.endswith("\u00ad"):
                text = text[:-1] + line.text
            elif _HYPHEN_END.search(text) and _LOWER_START.match(line.text):
                text = text[:-1] + line.text          # "extraordi-" + "narily"
            else:
                text = f"{text} {line.text}"
        size = max(current, key=lambda ln: len(ln.text)).size
        paragraphs.append((text, size))
        current.clear()

    for line in page.lines:
        stripped = line.text.strip()
        if len(stripped) == 1 and stripped.isupper() and line.size >= _PDF_DROPCAP_RATIO * body_size:
            _flush()
            drop_cap = stripped              # joins the word that follows it
            continue
        if drop_cap:
            line.text = drop_cap + line.text.lstrip()
            drop_cap = ""
        if current:
            previous = current[-1]
            width = max(1.0, right[line.block] - left[line.block])
            indent = line.x0 - left[line.block]
            # A first-line indent (relative to the line above) or a short line
            # that ends a sentence marks a paragraph boundary inside a block.
            new_paragraph = (
                line.block != previous.block
                or abs(line.size - previous.size) > 0.5
                or (len(by_block[line.block]) >= 3 and indent <= 0.25 * width
                    and indent - (previous.x0 - left[line.block]) >= 0.8 * line.size)
                or (previous.x1 < right[previous.block] - 0.25 * width and _PARA_END.search(previous.text))
                # Two indented first lines in a row: one-line paragraphs (dialogue).
                or (len(current) == 1 and 0.8 * line.size <= indent <= 0.25 * width
                    and abs(indent - (previous.x0 - left[line.block])) < 1.0
                    and _PARA_END.search(previous.text))
            )
            if new_paragraph:
                _flush()
        current.append(line)
    _flush()
    if drop_cap:
        paragraphs.append((drop_cap, body_size))
    return paragraphs


def _pdf_blocks(
    doc,
    page_indices,
    log: Callable[[str], None],
    *,
    budget: list[int] | None = None,
    context_indices=(),
) -> tuple[list[_Block], bool]:
    """Reads pages into paragraphs with headers/footers removed.

    Parameters
    ----------
    doc : fitz.Document
        Open PDF.
    page_indices : iterable of int
        0-based pages to read, in order.
    log : callable
        Receives warnings.
    budget : list[int] | None
        One-element list with the characters that may still be extracted; it
        is decremented in place (``MAX_EXTRACT_CHARS`` across all ranges).
    context_indices : iterable of int
        Extra pages used only to recognise running headers in short ranges.

    Returns
    -------
    tuple[list[_Block], bool]
        Paragraphs (heading levels set from the type size) and whether the
        character budget cut the extraction short.
    """
    flags = _pdf_text_flags()
    budget = budget if budget is not None else [_PDF_MAX_EXTRACT_CHARS]
    pages: list[_PdfPage] = []
    truncated = False
    for index in page_indices:
        page = _pdf_load_page(doc, index, flags)
        budget[0] -= page.char_count()
        pages.append(page)
        if budget[0] <= 0:
            truncated = True
            log("[WARN] Extracted-text limit reached — truncating extraction.")
            break
    context = [_pdf_load_page(doc, index, flags) for index in context_indices]

    body_size = _pdf_body_size(pages)
    _strip_running_lines(pages, body_size, context)

    blocks: list[_Block] = []
    sizes: list[float] = []
    for page in pages:
        first_of_page = True
        for text, size in _pdf_page_blocks(page, body_size):
            if first_of_page and blocks and sizes and blocks[-1].page != page.index:
                # A paragraph that runs over the page break is one paragraph.
                previous = blocks[-1].text
                same_type = abs(sizes[-1] - size) <= 0.5 and abs(size - body_size) <= 0.5
                if same_type and not _PARA_END.search(previous) and _LOWER_START.match(text):
                    joined = previous[:-1] + text if _HYPHEN_END.search(previous) else f"{previous} {text}"
                    blocks[-1].text = joined
                    first_of_page = False
                    continue
            first_of_page = False
            blocks.append(_Block(text, page=page.index))
            sizes.append(size)

    # Type clearly larger than the body marks a heading; rank the sizes into levels.
    heading_sizes = sorted({
        round(size * 2) / 2 for block, size in zip(blocks, sizes)
        if size >= _PDF_HEADING_RATIO * body_size and len(block.text) <= 120
        and _word_count(block.text) <= 14 and len(_ALNUM_RE.findall(block.text)) >= 2
    }, reverse=True)
    rank = {size: min(level, 4) for level, size in enumerate(heading_sizes, start=1)}
    for block, size in zip(blocks, sizes):
        key = round(size * 2) / 2
        if key in rank and len(block.text) <= 120 and _word_count(block.text) <= 14 \
                and len(_ALNUM_RE.findall(block.text)) >= 2:
            block.level = rank[key]
    return blocks, truncated


def _alnum_key(text: str) -> str:
    return "".join(ch for ch in text.casefold() if ch.isalnum())


def _apply_pdf_outline(doc, blocks: list[_Block]) -> bool:
    """Marks chapter starts from the PDF outline (bookmarks). True when used."""
    try:
        outline = [entry for entry in doc.get_toc() if len(entry) >= 3]
    except Exception as exc:
        logger.debug("PDF outline could not be read: %s", exc)
        return False
    usable = [
        (int(level), _collapse_ws(str(title)), int(page) - 1)
        for level, title, page in (entry[:3] for entry in outline)
        if isinstance(page, int) and page >= 1 and str(title).strip()
    ]
    if len(usable) < 2 or not blocks:
        return False
    first_on_page: dict[int, int] = {}
    for i, block in enumerate(blocks):
        first_on_page.setdefault(block.page, i)
    page_starts = sorted(first_on_page.items())

    marks: dict[int, tuple[int, str]] = {}
    for level, title, page in usable:
        key = _alnum_key(title)
        target = None
        for i in range(first_on_page.get(page, len(blocks)), len(blocks)):
            if blocks[i].page != page:
                break
            block_key = _alnum_key(blocks[i].text[:200])
            if len(block_key) >= 3 and (block_key == key or key.startswith(block_key) or (
                block_key.startswith(key) and len(blocks[i].text) <= 120
            )):
                target = i
                break
        if target is None:
            # No matching heading on the page: the chapter starts with the page.
            target = next((i for p, i in page_starts if p >= page), None)
        if target is not None and target not in marks:
            marks[target] = (level, title)
    if len(marks) < 2:
        return False
    for block in blocks:
        block.level = 0            # the outline outranks type-size guesses
    for index, (level, title) in marks.items():
        blocks[index].level = max(1, min(level, 9))
        blocks[index].title = title
    return True


def _pdf_sections(doc, blocks: list[_Block]) -> tuple[list[_Section], bool]:
    """Chapters of a PDF: outline first, then type size, then heading lines."""
    _apply_pdf_outline(doc, blocks)
    return _plan_sections(blocks)


def _pdf_is_textless(blocks: list[_Block], page_count: int) -> bool:
    letters = sum(len(_ALNUM_RE.findall(b.text)) for b in blocks)
    return letters < max(8, 4 * page_count)


def _pdf_ocr_blocks(doc, page_indices, language: str | None, log) -> list[_Block]:
    """OCRs scanned pages with EasyOCR. Returns [] when no OCR engine is usable."""
    from audiobook_factory.extractor_engine import DocumentIngestor  # type: ignore
    reader = DocumentIngestor.get_ocr_reader(language)
    if reader is None:
        return []
    blocks: list[_Block] = []
    try:
        import numpy as np
        for index in page_indices:
            pix = doc[index].get_pixmap(dpi=200, alpha=False)
            image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
            if pix.n == 1:
                image = np.repeat(image, 3, axis=2)
            for piece in reader.readtext(image[:, :, :3], detail=0, paragraph=True):
                if str(piece).strip():
                    blocks.append(_Block(str(piece).strip(), page=index))
            log(f"  OCR page {index + 1}")
    except Exception as exc:
        logger.warning("PDF OCR failed: %s", exc)
    finally:
        DocumentIngestor.release_ocr_reader()
    return blocks


def _extract_pdf_ranges(
    path, page_ranges, ingestor, normalizer, log,
    *,
    selections: list[int] | list[str] | None = None,
    enable_ocr: bool = False,
    language: str | None = None,
):
    """Extracts a PDF by page range, or by detected chapter when no range is given.

    ``ingestor`` is unused and only kept so existing callers keep working.
    """
    from audiobook_factory.text_processing import smart_sentence_splitter

    # ── Resource-safety caps (pre-auth anonymous surface) ──────────────────
    MAX_RANGES        = _PDF_MAX_RANGES
    MAX_PAGES         = _PDF_MAX_PAGES
    MAX_EXTRACT_CHARS = _PDF_MAX_EXTRACT_CHARS

    try:
        fitz = _import_fitz()
    except ImportError:
        log("[ERROR] PyMuPDF not installed — cannot extract PDF.")
        return []

    doc = fitz.open(path)
    try:
        results = []
        pages_left = MAX_PAGES
        budget = [MAX_EXTRACT_CHARS]

        def _read(pages, context=()) -> list[_Block]:
            blocks, _truncated = _pdf_blocks(doc, pages, log, budget=budget, context_indices=context)
            if _pdf_is_textless(blocks, len(pages)):
                ocr_blocks = _pdf_ocr_blocks(doc, pages, language, log) if enable_ocr else []
                if not ocr_blocks:
                    raise ExtractionError(_SCANNED_PDF_MESSAGE)
                return ocr_blocks
            return blocks

        if not page_ranges:
            # Whole document, split into chapters where an outline or headings exist.
            pages = range(min(doc.page_count, pages_left))
            sections, _detected = _pdf_sections(doc, _read(pages))
            return _chapters_from_sections(sections, selections, normalizer, log, fix_kerning=True)

        seen: set[tuple[int, int]] = set()
        idx = 0
        for start, end in page_ranges:
            if idx >= MAX_RANGES or pages_left <= 0 or budget[0] <= 0:
                log("[WARN] Page-range limits reached — stopping extraction.")
                break
            if (start, end) in seen:   # identical ranges are pure duplicates
                continue
            seen.add((start, end))
            pages = range(max(0, start - 1), min(end, doc.page_count))
            if len(pages) > pages_left:
                pages = pages[:pages_left]
            pages_left -= len(pages)
            if len(pages) == 0:
                continue
            context: list[int] = []
            if len(pages) < 6:
                # A short range cannot show that a header repeats; look at its neighbours too.
                context = [p for p in range(pages[0] - 3, pages[-1] + 4)
                           if 0 <= p < doc.page_count and p not in pages]
            raw   = _join_blocks(_read(pages, context))
            idx  += 1
            text  = normalizer.normalize(raw, title="", ocr_block_texts=[], fix_kerning=True)
            del raw
            results.append(ExtractedChapter(
                num=idx,
                title=f"Chapter {idx} (pp. {start}–{end})",
                text=text,
                sentences=smart_sentence_splitter(text),
            ))
            log(f"  ✓ Extracted pages {start}–{end}")
            if pages_left <= 0 or budget[0] <= 0:
                log("[WARN] Page/extracted-text limits reached — stopping extraction.")
                break
        return results
    finally:
        doc.close()


# ── DOCX ──────────────────────────────────────────────────────────────────────

_DOCX_SKIP_TAGS: frozenset[str] = frozenset({
    "del", "moveFrom", "rPrChange", "pPrChange", "instrText", "delText",
    "footnoteReference", "endnoteReference", "commentReference", "annotationRef",
    "drawing", "pict", "object", "AlternateContent", "fldSimple_disabled",
})
_DOCX_HEADING_NAME = re.compile(r"^heading\s*(\d)$", re.I)
_DOCX_TOC_STYLE = re.compile(r"^(?:toc\s*\d|toc\s*heading|table\s+of\s+figures)$", re.I)


def _local(tag: Any) -> str:
    return tag.rsplit("}", 1)[-1] if isinstance(tag, str) else ""


def _docx_paragraph_text(paragraph) -> str:
    """Text of a ``w:p`` element: tabs, line breaks and inserted text included;
    deleted text, field codes, footnote markers and drawings left out."""
    out: list[str] = []

    def _walk(element) -> None:
        for child in element:
            name = _local(child.tag)
            if name in _DOCX_SKIP_TAGS:
                continue
            if name == "t":
                out.append(child.text or "")
            elif name == "tab":
                out.append(" ")
            elif name in ("br", "cr"):
                out.append("\n")
            elif name == "noBreakHyphen":
                out.append("-")
            elif name in ("rPr", "pPr", "softHyphen", "sectPr"):
                continue
            else:
                _walk(child)

    _walk(paragraph)
    return "".join(out)


def _docx_blocks(path: str) -> tuple[list[_Block], str, str]:
    """Body paragraphs AND tables of a DOCX, in document order.

    Returns (blocks, title, author). Heading styles set ``_Block.level``
    ("Title" -> 1, "Heading n" -> n + 1).
    """
    from docx import Document as _DocxDoc  # type: ignore

    document = _DocxDoc(path)
    ns = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"

    styles: dict[str, tuple[str, int | None, str | None]] = {}   # id -> (name, outline, based_on)
    try:
        for style in document.styles.element.iter(f"{ns}style"):
            style_id = style.get(f"{ns}styleId") or ""
            name_el = style.find(f"{ns}name")
            outline_el = style.find(f"{ns}pPr/{ns}outlineLvl")
            based_el = style.find(f"{ns}basedOn")
            outline = None
            if outline_el is not None and (outline_el.get(f"{ns}val") or "").isdigit():
                outline = int(outline_el.get(f"{ns}val"))
            styles[style_id] = (
                (name_el.get(f"{ns}val") if name_el is not None else "") or style_id,
                outline,
                based_el.get(f"{ns}val") if based_el is not None else None,
            )
    except Exception as exc:
        logger.debug("Could not read DOCX styles: %s", exc)

    def _level(paragraph) -> tuple[int, bool]:
        """(heading level, is a TOC line) for a paragraph."""
        props = paragraph.find(f"{ns}pPr")
        style_id = None
        if props is not None:
            style_el = props.find(f"{ns}pStyle")
            style_id = style_el.get(f"{ns}val") if style_el is not None else None
            direct = props.find(f"{ns}outlineLvl")
            if direct is not None and (direct.get(f"{ns}val") or "").isdigit() and int(direct.get(f"{ns}val")) < 9:
                return int(direct.get(f"{ns}val")) + 2, False
        for _ in range(6):                      # follow basedOn a few steps
            if not style_id or style_id not in styles:
                break
            name, outline, parent = styles[style_id]
            if _DOCX_TOC_STYLE.match(name):
                return 0, True
            heading = _DOCX_HEADING_NAME.match(name)
            if heading:
                return int(heading.group(1)) + 1, False
            if name.lower() == "title":
                return 1, False
            if outline is not None and outline < 9:
                return outline + 2, False
            style_id = parent
        return 0, False

    blocks: list[_Block] = []
    drop_cap: list[str] = []          # letter(s) of a pending Word drop-cap frame

    def _table(table) -> None:
        for row in table.iter(f"{ns}tr"):
            cells = []
            for cell in row.findall(f"{ns}tc"):
                text = " ".join(
                    t for t in (_collapse_ws(_docx_paragraph_text(p)) for p in cell.iter(f"{ns}p")) if t
                )
                if text:
                    cells.append(text)
            if cells:
                blocks.append(_Block(", ".join(cells)))

    def _walk(container) -> None:
        for child in container:
            name = _local(child.tag)
            if name == "p":
                level, is_toc = _level(child)
                if is_toc:
                    continue                    # generated contents lines
                text = _docx_paragraph_text(child).strip()
                frame = child.find(f"{ns}pPr/{ns}framePr")
                if frame is not None and frame.get(f"{ns}dropCap") and len(text) <= 3:
                    # Word keeps a drop cap in its own framed paragraph; it
                    # belongs to the first word of the paragraph that follows.
                    drop_cap.append(text)
                    continue
                if drop_cap:
                    text = "".join(drop_cap) + text
                    drop_cap.clear()
                if text:
                    blocks.append(_Block(text, level=level))
            elif name == "tbl":
                _table(child)
            elif name == "sdt":
                gallery = child.find(f"{ns}sdtPr/{ns}docPartObj/{ns}docPartGallery")
                if gallery is not None and "table of contents" in (gallery.get(f"{ns}val") or "").lower():
                    continue
                content = child.find(f"{ns}sdtContent")
                if content is not None:
                    _walk(content)
            elif name in ("ins", "moveTo", "smartTag", "customXml"):
                _walk(child)

    _walk(document.element.body)

    title = author = ""
    try:
        props = document.core_properties
        title = (props.title or "").strip()
        author = (props.author or "").strip()
        if author.lower() == "python-docx":
            author = ""
    except Exception as exc:
        logger.debug("Could not read DOCX properties: %s", exc)
    return blocks, title, author


def _extract_docx(path, page_ranges, normalizer, log, *, selections=None):
    # ``page_ranges`` is accepted for compatibility; a DOCX has no fixed pages.
    try:
        _assert_zip_safe(path)
    except Exception as e:
        log(f"[ERROR] Rejected unsafe DOCX archive: {e}")
        return []
    try:
        import docx  # type: ignore  # noqa: F401
    except ImportError:
        log("[ERROR] python-docx not installed.")
        return []
    blocks, _title, _author = _docx_blocks(path)
    sections, _detected = _plan_sections(blocks)
    return _chapters_from_sections(sections, selections, normalizer, log)


# ── ODT ───────────────────────────────────────────────────────────────────────

_ODF_TEXT_NS: str = "urn:oasis:names:tc:opendocument:xmlns:text:1.0"
_ODF_TABLE_NS: str = "urn:oasis:names:tc:opendocument:xmlns:table:1.0"
_ODF_SKIP: frozenset[str] = frozenset({
    "note", "tracked-changes", "table-of-content", "illustration-index", "alphabetical-index",
    "user-index", "object-index", "table-index", "bibliography", "annotation",
    "annotation-end", "change", "change-start", "change-end", "sequence-decls",
    "variable-decls", "user-field-decls", "forms",
})


def _odt_inline_text(node) -> str:
    """Text of a paragraph/heading: line breaks, tabs and spaces included,
    footnotes, comments and tracked deletions left out."""
    out: list[str] = []
    for child in node.childNodes:
        if child.nodeType == 3:                 # text node
            out.append(child.data)
            continue
        if child.nodeType != 1:
            continue
        namespace, name = child.qname
        if name in _ODF_SKIP:
            continue
        if namespace == _ODF_TEXT_NS and name == "line-break":
            out.append("\n")
        elif namespace == _ODF_TEXT_NS and name == "tab":
            out.append(" ")
        elif namespace == _ODF_TEXT_NS and name == "s":
            try:
                out.append(" " * int(child.getAttribute("c") or 1))
            except (TypeError, ValueError):
                out.append(" ")
        elif namespace == _ODF_TEXT_NS and name in ("p", "h"):
            out.append(" " + _odt_inline_text(child))   # a text box inside the paragraph
        elif namespace.endswith(":drawing:1.0"):
            continue
        else:
            out.append(_odt_inline_text(child))
    return "".join(out)


def _odt_blocks(path: str) -> tuple[list[_Block], str, str]:
    """Headings, paragraphs, lists and tables of an ODT body, in document order.

    Returns (blocks, title, author). ``text:h`` sets ``_Block.level`` to its
    outline level + 1; a paragraph in the "Title" style is level 1.
    """
    from odf.opendocument import load as odf_load  # type: ignore

    document = odf_load(path)
    blocks: list[_Block] = []

    def _walk(container) -> None:
        for node in container.childNodes:
            if node.nodeType != 1:
                continue
            namespace, name = node.qname
            if name in _ODF_SKIP:
                continue
            if namespace == _ODF_TEXT_NS and name == "h":
                text = _odt_inline_text(node).strip()
                try:
                    level = int(node.getAttribute("outlinelevel") or 1)
                except (TypeError, ValueError):
                    level = 1
                if text:
                    blocks.append(_Block(text, level=max(1, min(level, 8)) + 1))
            elif namespace == _ODF_TEXT_NS and name == "p":
                text = _odt_inline_text(node).strip()
                style = str(node.getAttribute("stylename") or "")
                if text:
                    blocks.append(_Block(text, level=1 if style.lower() == "title" else 0))
            elif namespace == _ODF_TABLE_NS and name == "table":
                _table(node)
            elif namespace in (_ODF_TEXT_NS, _ODF_TABLE_NS):
                _walk(node)                     # lists, list items, sections, index bodies

    def _table(table) -> None:
        for node in table.childNodes:
            if node.nodeType != 1:
                continue
            name = node.qname[1]
            if name == "table-row":
                cells = []
                for cell in node.childNodes:
                    if cell.nodeType != 1 or cell.qname[1] != "table-cell":
                        continue
                    parts: list[str] = []
                    _cell_text(cell, parts)
                    text = _collapse_ws(" ".join(parts))
                    if text:
                        cells.append(text)
                if cells:
                    blocks.append(_Block(", ".join(cells)))
            else:
                _table(node)                    # header rows, row groups

    def _cell_text(node, parts: list[str]) -> None:
        for child in node.childNodes:
            if child.nodeType != 1 or child.qname[1] in _ODF_SKIP:
                continue
            if child.qname[0] == _ODF_TEXT_NS and child.qname[1] in ("p", "h"):
                parts.append(_odt_inline_text(child))
            else:
                _cell_text(child, parts)

    _walk(document.text)

    title = author = ""
    try:
        for node in document.meta.childNodes:
            if node.nodeType != 1:
                continue
            if node.qname[1] == "title":
                title = _collapse_ws("".join(c.data for c in node.childNodes if c.nodeType == 3))
            elif node.qname[1] in ("creator", "initial-creator") and not author:
                author = _collapse_ws("".join(c.data for c in node.childNodes if c.nodeType == 3))
    except Exception as exc:
        logger.debug("Could not read ODT metadata: %s", exc)
    return blocks, title, author


def _extract_odt(path, page_ranges, normalizer, log, *, selections=None):
    # ``page_ranges`` is accepted for compatibility; an ODT has no fixed pages.
    try:
        _assert_zip_safe(path)
    except Exception as e:
        log(f"[ERROR] Rejected unsafe ODT archive: {e}")
        return []
    try:
        import odf.opendocument  # type: ignore  # noqa: F401
    except ImportError:
        log("[ERROR] odfpy not installed.")
        return []
    blocks, _title, _author = _odt_blocks(path)
    sections, _detected = _plan_sections(blocks)
    return _chapters_from_sections(sections, selections, normalizer, log)
