# Source-document fixtures

Everything in this directory is **generated** — do not edit the files by hand.
Regenerate with:

```bash
python tests/fixture_generation/generate_test_documents.py
```

The text is original prose written for this test-suite ("The Lighthouse at
Saltmarsh Point" by the invented author Mara Ellison), so the files carry no
third-party copyright. `expected_chapters.json` is the machine-readable
version of this page: for every fixture it lists the ordered chapter titles
that `scan()` and `extract()` must return, phrases that must appear in the
extracted text, phrases that must NOT appear (front/back matter, running
headers, footnote bodies) and one phrase per chapter that must be found in
that chapter and no other.

## The book

Narrated chapters, in order:

1. `Prologue`
2. `Chapter 1: The Salt Road`
3. `Chapter 2: Plan B`
4. `About the Lighthouse`
5. `Chapter 10: The Tenth Bell`
6. `Epilogue`

`Chapter 10` deliberately follows `Chapter 2`, and `About the Lighthouse`
starts with a skip-list word, so a sort or prefix-match bug changes the list.

Never narrated: title page, copyright page (`All rights reserved`, `ISBN`),
dedication, contents, footnote text, `Notes`, `About the Author`,
`Also by Mara Ellison`. With `scan(path, include_matter=True)` these come back
flagged `probably_matter=True`.

The prose exercises the normaliser: a drop cap, a sentence of more than 400
characters, `"No!" he cried.`, `Dr.` / `Mr.` / `e.g.,` / `etc.,`, `£120`,
`$4,000`, `1,250`, `2.5 miles`, `4:15 p.m.`, `the 3rd of March, 1926`,
`William IV`, `Chapter I`, an em-dash, an ellipsis, a `* * *` scene break, a
footnote marker, a three-column table, italic and bold mid-sentence, a forced
line break, `Plan B`, `Vitamin C`, `café`, `naïve`, `Привет` and `東京`.

## Files

| File | Size | What it is |
|------|------|------------|
| `dummy_book.epub` | 8.7 KB | Reference EPUB 3: nav + NCX, cover, front and back matter listed in the TOC. |
| `dummy_book.pdf` | 17.0 KB | PDF with outline, running header and page number on every page. |
| `dummy_book_no_outline.pdf` | 15.6 KB | Same PDF without bookmarks: chapters come from heading type size. |
| `dummy_book.docx` | 38.8 KB | DOCX with heading styles, real footnote, drop-cap frame, table, line break. |
| `dummy_book.odt` | 4.5 KB | ODT with text:h headings, text:note, line-break, tab, table. |
| `dummy_book.txt` | 5.0 KB | Plain text with a Contents listing, pipe table and [1] footnote marker. |
| `dummy_book.mobi` | 7.2 KB | Hand-built classic MOBI 6 (PalmDB + MOBI/EXTH headers, filepos TOC, cover). |
| `epub_no_toc.epub` | 5.6 KB | A.1b — no nav, empty NCX: chapters come from per-file classification; a chapter split over two files. |
| `epub_multi_anchor_single_file.epub` | 4.7 KB | A.1a — whole book in one XHTML file, TOC entries are anchors into it. |
| `epub_percent_encoded_hrefs.epub` | 4.8 KB | A.1e — file names with spaces and an accent, percent-encoded in OPF and NCX. |
| `epub_manifest_order_differs.epub` | 5.6 KB | A.1g — scrambled manifest; spine is the reading order; unlisted second file of Chapter 1. |
| `epub_trailing_backmatter.epub` | 7.7 KB | A.1c/d/f — unlisted letter before the TOC is kept; unlisted back matter and preview are not; 1.xhtml vs 11.xhtml. |

## Expected chapters per fixture

### dummy_book.epub

1. `Prologue`
2. `Chapter 1: The Salt Road`
3. `Chapter 2: Plan B`
4. `About the Lighthouse`
5. `Chapter 10: The Tenth Bell`
6. `Epilogue`

### dummy_book.pdf

1. `Prologue`
2. `Chapter 1: The Salt Road`
3. `Chapter 2: Plan B`
4. `About the Lighthouse`
5. `Chapter 10: The Tenth Bell`
6. `Epilogue`

### dummy_book_no_outline.pdf

1. `Prologue`
2. `Chapter 1: The Salt Road`
3. `Chapter 2: Plan B`
4. `About the Lighthouse`
5. `Chapter 10: The Tenth Bell`
6. `Epilogue`

### dummy_book.docx

1. `Prologue`
2. `Chapter 1: The Salt Road`
3. `Chapter 2: Plan B`
4. `About the Lighthouse`
5. `Chapter 10: The Tenth Bell`
6. `Epilogue`

### dummy_book.odt

1. `Prologue`
2. `Chapter 1: The Salt Road`
3. `Chapter 2: Plan B`
4. `About the Lighthouse`
5. `Chapter 10: The Tenth Bell`
6. `Epilogue`

### dummy_book.txt

1. `Prologue`
2. `Chapter 1: The Salt Road`
3. `Chapter 2: Plan B`
4. `About the Lighthouse`
5. `Chapter 10: The Tenth Bell`
6. `Epilogue`

### dummy_book.mobi

1. `Prologue`
2. `Chapter 1: The Salt Road`
3. `Chapter 2: Plan B`
4. `About the Lighthouse`
5. `Chapter 10: The Tenth Bell`
6. `Epilogue`

### epub_no_toc.epub

1. `Prologue`
2. `Chapter 1: The Salt Road`
3. `About the Lighthouse`
4. `Epilogue`

### epub_multi_anchor_single_file.epub

1. `Prologue`
2. `Chapter 1: The Salt Road`
3. `Chapter 2: Plan B`
4. `About the Lighthouse`
5. `Chapter 10: The Tenth Bell`
6. `Epilogue`

### epub_percent_encoded_hrefs.epub

1. `Chapter 1: The Salt Road`
2. `Chapter 2: Plan B`
3. `About the Lighthouse`
4. `Epilogue`

### epub_manifest_order_differs.epub

1. `Prologue`
2. `Chapter 1: The Salt Road`
3. `Chapter 2: Plan B`
4. `Chapter 10: The Tenth Bell`
5. `Epilogue`

### epub_trailing_backmatter.epub

1. `A Letter Before the Story`
2. `Prologue`
3. `Chapter 1: The Salt Road`
4. `Chapter 2: Plan B`
5. `Epilogue`


## Format notes

- **dummy_book.epub** — EPUB 3, one XHTML file per chapter, nav + NCX, PNG
  cover. Footnote is `<sup><a epub:type="noteref">` plus an
  `<aside epub:type="footnote">`.
- **dummy_book.pdf** — A5 pages typeset with PyMuPDF. EVERY page carries the
  running header `The Lighthouse at Saltmarsh Point` and a page number;
  paragraphs are marked by first-line indents only, some lines end in a
  hyphenated word, paragraphs run across page breaks, the footnote sits at
  the foot of its page in small type. Chapters are in the PDF outline.
- **dummy_book_no_outline.pdf** — the same pages without bookmarks.
- **dummy_book.docx** — `Title` / `Heading 1` styles, a real
  `w:footnoteReference` (with a footnotes part), a Word drop cap (framed
  paragraph), a table and a `w:br` line break.
- **dummy_book.odt** — `text:h` headings, a `text:note`, `text:line-break`,
  `text:tab` (in the contents lines) and a `table:table`.
- **dummy_book.txt** — UTF-8, a `Contents` listing, `_italic_` /
  `**bold**`, a pipe table and a `[1]` footnote marker.
- **dummy_book.mobi** — classic MOBI 6: PalmDB container, MOBI header +
  EXTH, uncompressed UTF-8 text records, inline contents page with `filepos`
  links, a cover image record, FLIS/FCIS/EOF records. Built byte by byte in
  `fixture_builders.build_mobi` (no Calibre, no kindlegen) and opened in the
  tests through the same `mobi` (KindleUnpack) reader the application uses.
