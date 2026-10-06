"""
audiobook_factory/chunk_planner.py
===================================
Turns a chapter's text into the list of chunks sent to the TTS model.

A chunk is what one TTS call speaks. How the text is cut decides how the
narration sounds:

* One sentence per call gives the model no context, so every sentence starts
  with the same reset intonation and very short lines ("Yes.") clone badly.
  Consecutive sentences of a paragraph are therefore packed into one chunk up
  to ``max_len`` characters.
* A dialogue tag split from its quote (``"No!" he cried.``) is read as two
  unrelated utterances, so a fragment that starts in lower case is re-joined
  to the sentence before it.
* The silence after a chunk depends on what follows: the sentence pause inside
  a paragraph, the longer paragraph pause at its end.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from audiobook_factory.text_processing import smart_sentence_splitter

_CLOSING_QUOTES: str = "\"'”’)]"


@dataclass(frozen=True)
class SpeechChunk:
    """One TTS call.

    Attributes
    ----------
    text : str
        Text spoken by this call (at most ``max_len`` characters).
    pause_after : float
        Seconds of silence inserted after the chunk's audio.
    sentences : tuple[str, ...]
        The sentences packed into ``text``, in order; used to give subtitles
        sentence-level timing inside a multi-sentence chunk.
    paragraph_end : bool
        True when this chunk closes a paragraph.
    """

    text: str
    pause_after: float
    sentences: tuple[str, ...] = field(default_factory=tuple)
    paragraph_end: bool = False


def _rejoin_dialogue_tags(sentences: list[str], max_len: int) -> list[str]:
    """Merges a lower-case continuation into the sentence it belongs to.

    ``['"No!"', 'he cried.']`` becomes ``['"No!" he cried.']``. Only applies
    when the previous piece ends in a closing quote/bracket, which is what a
    sentence tokenizer leaves behind when it splits after quoted punctuation.
    """
    merged: list[str] = []
    for sentence in sentences:
        previous = merged[-1] if merged else ""
        if (
            previous
            and sentence[:1].islower()
            and previous[-1:] in _CLOSING_QUOTES
            and len(previous) + 1 + len(sentence) <= max_len
        ):
            merged[-1] = f"{previous} {sentence}"
        else:
            merged.append(sentence)
    return merged


def _pack(sentences: list[str], max_len: int) -> list[tuple[str, ...]]:
    """Greedily groups consecutive sentences into runs of at most ``max_len`` chars."""
    groups: list[tuple[str, ...]] = []
    current: list[str] = []
    current_len = 0
    for sentence in sentences:
        added = len(sentence) + (1 if current else 0)
        if current and current_len + added > max_len:
            groups.append(tuple(current))
            current, current_len = [], 0
            added = len(sentence)
        current.append(sentence)
        current_len += added
    if current:
        groups.append(tuple(current))
    return groups


def plan_chunks(
    text: str,
    sentences: list[str] | None,
    max_len: int,
    pause: float,
    para_pause: float,
    pack_sentences: bool = True,
) -> list[SpeechChunk]:
    """Plans the TTS chunks for one chapter.

    Parameters
    ----------
    text : str
        Chapter text with paragraphs separated by blank lines. When empty,
        ``sentences`` is used instead and treated as a single paragraph.
    sentences : list[str] | None
        Pre-split sentences, used only when ``text`` is empty.
    max_len : int
        Maximum characters per chunk.
    pause : float
        Silence after a chunk inside a paragraph, in seconds.
    para_pause : float
        Silence after the last chunk of a paragraph, in seconds.
    pack_sentences : bool
        Pack consecutive sentences of a paragraph into one chunk. When False
        every sentence is its own chunk.

    Returns
    -------
    list[SpeechChunk]
        Chunks in reading order. The last chunk of the chapter keeps the
        sentence pause so the chapter does not end on a long silence.
    """
    max_len = max(1, int(max_len))
    paragraphs: list[list[str]] = []

    if text and text.strip():
        for paragraph in text.split("\n\n"):
            pieces = smart_sentence_splitter(paragraph, max_len)
            if pieces:
                paragraphs.append(pieces)
    else:
        # No paragraph structure survives in a flat sentence list, so it is
        # read as one paragraph rather than pausing long after every line.
        flat: list[str] = []
        for sentence in sentences or []:
            if sentence and sentence.strip():
                flat.extend(smart_sentence_splitter(sentence, max_len))
        if flat:
            paragraphs.append(flat)

    chunks: list[SpeechChunk] = []
    for pieces in paragraphs:
        pieces = _rejoin_dialogue_tags(pieces, max_len)
        groups = _pack(pieces, max_len) if pack_sentences else [(p,) for p in pieces]
        for position, group in enumerate(groups):
            last = position == len(groups) - 1
            chunks.append(SpeechChunk(
                text=" ".join(group),
                pause_after=float(para_pause if last else pause),
                sentences=group,
                paragraph_end=last,
            ))

    if chunks:
        final = chunks[-1]
        chunks[-1] = SpeechChunk(final.text, float(pause), final.sentences, final.paragraph_end)
    return chunks
