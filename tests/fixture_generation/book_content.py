"""
book_content.py
===============
The text of the dummy test book, "The Lighthouse at Saltmarsh Point".

Everything here is original prose written for this test-suite (no third-party
text), so the generated fixtures can be committed and redistributed freely.

The book is deliberately awkward for a text-to-speech pipeline. It contains:

- front matter (title page, copyright, dedication, contents) and back matter
  (notes, about the author, also-by) that must NOT be narrated;
- a prologue, an epilogue, a chapter whose title starts with a skip-list
  lookalike word ("About the Lighthouse") and a "Chapter 10" that follows
  "Chapter 2", so sorting or prefix-matching bugs show up as a wrong order;
- dialogue with a tag after an exclamation, abbreviations, numbers, currency,
  dates, roman numerals, ordinals, an em-dash, an ellipsis, a scene break, a
  footnote marker, a table, inline italic/bold, a drop cap, single-letter
  names ("Plan B", "Vitamin C"), accented and non-Latin words, a forced line
  break and one sentence of more than 400 characters.

Content model
-------------
A chapter is a list of blocks::

    ("p", [segments])                 paragraph
    ("dropcap", "T", [segments])      paragraph opened by a drop cap
    ("scene",)                        scene break
    ("table", [[cell, ...], ...])     table, first row is the header

and a segment is a plain ``str`` or one of::

    ("i", text)   italic        ("b", text)   bold
    ("br",)       line break    ("fn", "1")   footnote marker (see FOOTNOTES)
"""

from __future__ import annotations

from typing import Any

TEST_BOOK_METADATA: dict[str, str] = {
    "title": "The Lighthouse at Saltmarsh Point",
    "author": "Mara Ellison",
    "language": "en",
    "description": "A synthetic multi-chapter test novel for AudiobookMaker validation.",
    "identifier": "abm-synthetic-fixture-002",
}

FOOTNOTES: dict[str, str] = {
    "1": "The Salt Road is covered for roughly four hours around each high tide.",
}

_LONG_SENTENCE: str = (
    "By the time the fishing boats had cleared the harbour wall and the gulls had finished "
    "quarrelling over the scraps on the quay, and the baker had set out her first trays, and "
    "the schoolmaster had rung his cracked brass bell, and the postman had begun his slow climb "
    "up the hill with a sack of letters that were mostly bills, the lamp at the top of the "
    "lighthouse was still burning, pale and stubborn against the brightening sky, because the "
    "keeper who should have put it out was nowhere to be found."
)
assert len(_LONG_SENTENCE) > 400

CHAPTERS_DATA: list[dict[str, Any]] = [
    {
        "key": "prologue",
        "title": "Prologue",
        "blocks": [
            ("dropcap", "T", ["he tide went out at dawn, as it had every morning since 1887, "
                              "and nobody in Saltmarsh thought to watch it go."]),
            ("p", [_LONG_SENTENCE]),
            ("p", ["It was the first time in thirty-nine years that the light had been left on."]),
        ],
    },
    {
        "key": "ch1",
        "title": "Chapter 1: The Salt Road",
        "blocks": [
            ("p", ["Dr. Imogen Hale arrived on the 3rd of March, 1926, with two trunks, a bicycle "
                   "and a letter from Mr. Abel Crane, the harbour master."]),
            ("p", ["The letter promised a salary of £120 a year, a cottage, and—underlined "
                   "twice—no visitors after dark."]),
            ("p", ["\"You'll want the Salt Road,\" said the carter. \"It floods at high water, mind.\""]),
            ("p", ["\"Does it flood often?\" she asked."]),
            ("p", ["\"Only twice a day.\""]),
            ("p", ["She laughed, but he did not, and by the time she understood that he was being ",
                   ("i", "perfectly"), " serious the cart had already turned for home."]),
            ("p", ["The notice on the cottage door read:", ("br",), "KEEPER ABSENT", ("br",),
                   "ENQUIRE AT THE HARBOUR OFFICE"]),
            ("p", ["She had packed for every emergency she could imagine, e.g., a broken lens, a "
                   "flooded cellar, a failed generator, etc., and for none of the ones that actually "
                   "happened. That night she consulted the tide tables", ("fn", "1"),
                   " twice before breakfast was even a possibility."]),
        ],
    },
    {
        "key": "ch2",
        "title": "Chapter 2: Plan B",
        "blocks": [
            ("p", ["Plan A had been simple: find the missing keeper, relight the lamp, go home."]),
            ("p", ["Plan B was Imogen's own invention, and it involved a rowing boat, a flask of tea "
                   "with a slice of lemon for the Vitamin C, and the tide tables."]),
            ("p", ["\"No!\" he cried. \"Not the north channel. Nobody rows the north channel in March.\""]),
            ("p", ["\"Then what do you suggest, Mr. Crane?\""]),
            ("p", ["He rubbed his jaw. \"I suggest… I suggest we wait for the ebb.\""]),
            ("scene",),
            ("p", ["They waited. At 4:15 p.m. the water began to slide away from the causeway, and by "
                   "five o'clock the Salt Road lay bare and shining, all 2.5 miles of it."]),
            ("table", [
                ["Day", "High water", "Low water"],
                ["Monday", "6:10 a.m.", "12:25 p.m."],
                ["Tuesday", "6:55 a.m.", "1:05 p.m."],
            ]),
            ("p", ["On the back of the table someone had written, in pencil, ",
                   ("b", "do not trust Tuesday"), "."]),
        ],
    },
    {
        "key": "lighthouse",
        "title": "About the Lighthouse",
        "blocks": [
            ("p", ["About the lighthouse itself there were three stories, and Imogen heard all of "
                   "them in her first week."]),
            ("p", ["The first said it was built in 1851 by King William IV's own engineer, which was "
                   "impossible, since that king had died in 1837."]),
            ("p", ["The second said the lamp had cost $4,000, shipped in 1,250 pieces from a workshop "
                   "in Paris, and that the man who assembled it kept a café on the quay until the "
                   "day he died."]),
            ("p", ["The third story was the one the children told. It said the keeper could speak to "
                   "ships in any language, that he had once answered a Russian trawler with a cheerful "
                   "\"Привет!\" and a steamer bound for 東京 "
                   "with a bow, and that only a naïve person would ask how."]),
            ("p", ["Imogen wrote all three down in her notebook, under the heading ",
                   ("i", "Unreliable"), ", and underlined the second."]),
        ],
    },
    {
        "key": "ch10",
        "title": "Chapter 10: The Tenth Bell",
        "blocks": [
            ("p", ["The bell in the lamp room rang ten times at midnight on the 21st, although nobody "
                   "had wound it since the 2nd."]),
            ("p", ["\"That makes ten,\" said Mr. Crane, who had been counting on his fingers. \"It only "
                   "ever rings nine.\""]),
            ("p", ["\"What happens on the tenth?\""]),
            ("p", ["\"I don't know. Nobody's ever stayed to find out—\""]),
            ("p", ["The door at the foot of the stairs opened. Wet footprints crossed the stone floor, "
                   "one after another, and stopped at the first step. Then a voice Imogen had never "
                   "heard, but recognised at once from Chapter I of the keeper's own logbook, said, "
                   "\"You left my light on.\""]),
            ("p", ["\"Somebody had to,\" she said."]),
        ],
    },
    {
        "key": "epilogue",
        "title": "Epilogue",
        "blocks": [
            ("p", ["The keeper stayed until spring. He would not say where he had been, only that the "
                   "journey had cost him exactly 39 years and one good pair of boots."]),
            ("p", ["Dr. Hale left Saltmarsh on the 1st of May. She took the bicycle, one trunk, and the "
                   "tide tables, which she never trusted again."]),
            ("p", ["The lamp is still lit every evening at dusk. If you take the Salt Road at low water "
                   "you can see it from the second milestone… and if it is a Tuesday, you should "
                   "walk a little faster."]),
        ],
    },
]

# ── Front and back matter: must never be narrated ────────────────────────────

COPYRIGHT_LINES: list[str] = [
    "Copyright © 2026 Mara Ellison. All rights reserved.",
    "No part of this book may be reproduced without written permission.",
    "ISBN 000-0-00-000000-0",
    "First published by Saltmarsh Test Press. Printed in the fixture directory.",
]
DEDICATION: str = "For the keepers of small lights."
ABOUT_AUTHOR_TITLE: str = "About the Author"
ABOUT_AUTHOR: str = (
    "Mara Ellison grew up beside a tidal causeway and has never fully trusted a timetable. "
    "She does not exist; she was invented for this test fixture."
)
ALSO_BY_TITLE: str = "Also by Mara Ellison"
ALSO_BY: list[str] = ["The Cartographer's Daughter", "Nine Bells for Winter", "A Field Guide to Fog"]
NOTES_TITLE: str = "Notes"

# An unlisted letter placed BEFORE the first TOC entry in one edge-case EPUB:
# long enough (> 150 words) that it must be kept as a chapter.
LETTER_TITLE: str = "A Letter Before the Story"
LETTER_PARAGRAPHS: list[str] = [
    "My dear reader, before the tide tables and the missing keeper there is something you ought "
    "to know about Saltmarsh, and it is this: the town keeps its promises slowly. A debt agreed in "
    "one generation is settled in the next, a quarrel begun over a fence is finished at a funeral, "
    "and a lamp that is lit at dusk is expected to be put out at dawn by the same pair of hands.",
    "I mention it because the story that follows is, at heart, about a promise that was kept late. "
    "Nobody in it is wicked. Several people are stubborn, one is frightened, and one is simply very "
    "tired, and between them they manage to leave a light burning that should have gone dark.",
    "If you find, by the last page, that you have begun to check the clock whenever you cross wet "
    "sand, then the book has done what its author hoped, and you may consider yourself an honorary "
    "citizen of the harbour. Walk carefully, and mind the second milestone.",
]
assert sum(len(p.split()) for p in LETTER_PARAGRAPHS) > 150

# A preview of "the next book", placed after the back matter in one edge-case EPUB.
PREVIEW_TITLE: str = "Preview of The Cartographer's Daughter"
PREVIEW_PARAGRAPHS: list[str] = [
    "The map arrived folded into eighths, and every fold had been made by someone in a hurry. "
    "Odile spread it across the kitchen table and weighted the corners with jam jars.",
    "It showed a coastline she had walked every day of her life, drawn by a hand that had plainly "
    "never seen it. The harbour was on the wrong side. The lighthouse was missing altogether.",
]

RUNNING_HEADER: str = "The Lighthouse at Saltmarsh Point"

EXPECTED_TITLES: list[str] = [chapter["title"] for chapter in CHAPTERS_DATA]

# Phrases that must survive extraction in every format (compared after
# collapsing whitespace, so line wrapping and paragraph breaks do not matter).
MUST_CONTAIN: list[str] = [
    "The tide went out at dawn, as it had every morning since 1887",   # drop cap re-joined
    "By the time the fishing boats had cleared the harbour wall",       # start of the long sentence
    "because the keeper who should have put it out was nowhere to be found.",
    "Dr. Imogen Hale arrived on the 3rd of March, 1926",
    "a letter from Mr. Abel Crane, the harbour master.",
    "a salary of £120 a year",
    "\"Does it flood often?\" she asked.",
    "he was being perfectly serious",                                   # inline italic mid-sentence
    "KEEPER ABSENT",
    "e.g., a broken lens",
    "a failed generator, etc., and for none",
    "the tide tables twice before breakfast",                           # footnote marker removed
    "Plan B was Imogen's own invention",
    "for the Vitamin C, and the tide tables.",
    "\"No!\" he cried.",
    "I suggest we wait for the ebb.",
    "At 4:15 p.m. the water began to slide away",
    "all 2.5 miles of it.",
    "6:10 a.m.",                                                        # table cell
    "do not trust Tuesday",                                             # inline bold
    "About the lighthouse itself there were three stories",
    "King William IV's own engineer",
    "the lamp had cost $4,000, shipped in 1,250 pieces",
    "kept a café on the quay",
    "Привет",
    "東京",
    "only a naïve person would ask how.",
    "under the heading Unreliable, and underlined the second.",
    "rang ten times at midnight on the 21st",
    "recognised at once from Chapter I of the keeper's own logbook",
    "\"Somebody had to,\" she said.",
    "exactly 39 years and one good pair of boots.",
    "Dr. Hale left Saltmarsh on the 1st of May.",
    "you should walk a little faster.",
]

# Front/back matter, running headers and footnote bodies.
MUST_NOT_CONTAIN: list[str] = [
    "All rights reserved",
    "ISBN",
    "Mara Ellison",
    "Saltmarsh Test Press",
    "For the keepers of small lights",
    "grew up beside a tidal causeway",
    "The Cartographer's Daughter",
    "Nine Bells for Winter",
    "covered for roughly four hours",       # footnote body
    "tables1",                              # footnote marker glued to its word
    "tables 1 twice",
    "tables[1]",
    "OCR_IMG_TEXT",
]

# One phrase per chapter, to prove each chapter holds its own text.
CHAPTER_PHRASES: dict[str, str] = {
    "Prologue": "the light had been left on.",
    "Chapter 1: The Salt Road": "Only twice a day.",
    "Chapter 2: Plan B": "Plan A had been simple",
    "About the Lighthouse": "The third story was the one the children told.",
    "Chapter 10: The Tenth Bell": "It only ever rings nine.",
    "Epilogue": "The keeper stayed until spring.",
}


def segments_text(segments: list[Any], *, note_marker: str = "", line_break: str = "\n") -> str:
    """Plain text of a segment list.

    Parameters
    ----------
    segments : list
        Inline segments (see the module docstring).
    note_marker : str
        Format for footnote markers, e.g. ``"[{}]"``; empty drops them.
    line_break : str
        Text used for a forced line break.

    Returns
    -------
    str
        The paragraph as plain text.
    """
    out = []
    for segment in segments:
        if isinstance(segment, str):
            out.append(segment)
        elif segment[0] in ("i", "b"):
            out.append(segment[1])
        elif segment[0] == "br":
            out.append(line_break)
        elif segment[0] == "fn" and note_marker:
            out.append(note_marker.format(segment[1]))
    return "".join(out)


def chapter_plain_paragraphs(chapter: dict[str, Any]) -> list[str]:
    """Paragraphs of a chapter as plain strings (tables become comma-joined rows)."""
    paragraphs = []
    for block in chapter["blocks"]:
        if block[0] == "p":
            paragraphs.append(segments_text(block[1], line_break=" "))
        elif block[0] == "dropcap":
            paragraphs.append(block[1] + segments_text(block[2], line_break=" "))
        elif block[0] == "table":
            paragraphs.extend(", ".join(row) for row in block[1])
    return paragraphs


def chapter_by_key(key: str) -> dict[str, Any]:
    return next(chapter for chapter in CHAPTERS_DATA if chapter["key"] == key)
