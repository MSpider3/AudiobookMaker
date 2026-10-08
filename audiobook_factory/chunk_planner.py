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
* ``max_len`` is a character count calibrated on Latin text (399 characters is
  about 25 seconds of speech). A Chinese character is a whole syllable, so the
  same count would be over a minute of audio in one TTS call; the limit is
  scaled down by how long the script takes to say.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from audiobook_factory.text_processing import smart_sentence_splitter

_CLOSING_QUOTES: str = "\"'”’)]"

# Speaking time of one character relative to a Latin letter.
_HAN_WEIGHT: float = 3.0
_KANA_WEIGHT: float = 1.5
_HANGUL_WEIGHT: float = 2.2
_MIN_CHAR_LIMIT: int = 40  # never cut finer than a short sentence


def _char_weight(char: str) -> float:
    """How long *char* takes to say, relative to a Latin letter."""
    code = ord(char)
    if 0x4E00 <= code <= 0x9FFF or 0x3400 <= code <= 0x4DBF or 0xF900 <= code <= 0xFAFF or code >= 0x20000:
        return _HAN_WEIGHT
    if 0x3040 <= code <= 0x30FF or 0x31F0 <= code <= 0x31FF:
        return _KANA_WEIGHT
    if 0xAC00 <= code <= 0xD7A3 or 0x1100 <= code <= 0x11FF or 0x3130 <= code <= 0x318F:
        return _HANGUL_WEIGHT
    return 1.0


def _is_spaceless(char: str) -> bool:
    """True for characters of scripts written without spaces between words."""
    code = ord(char)
    return (
        0x4E00 <= code <= 0x9FFF or 0x3400 <= code <= 0x4DBF or 0xF900 <= code <= 0xFAFF
        or 0x3040 <= code <= 0x30FF          # hiragana, katakana
        or 0x3000 <= code <= 0x303F          # CJK punctuation
        or 0xFF00 <= code <= 0xFFEF          # full-width forms
        or code >= 0x20000
    )


def char_limit(text: str, max_len: int) -> int:
    """``max_len`` adjusted for how long the script of *text* takes to say.

    Parameters
    ----------
    text : str
        The text about to be cut into chunks.
    max_len : int
        Chunk limit in characters of Latin text.

    Returns
    -------
    int
        Character limit that gives chunks of about the same spoken length:
        ``max_len`` for Latin, Cyrillic or Devanagari text, about a third of
        it for Chinese, between the two for Japanese and Korean.
    """
    max_len = max(1, int(max_len))
    counted = [char for char in text if not char.isspace()]
    if not counted:
        return max_len
    weight = sum(_char_weight(char) for char in counted) / len(counted)
    if weight <= 1.05:
        return max_len
    return max(min(_MIN_CHAR_LIMIT, max_len), int(max_len / weight))


_BOUNDARY_MARKS: str = "\"'“”‘’()[]"


def _edge_char(text: str, last: bool) -> str:
    """The first or last character of *text* that is not a quote or bracket."""
    for char in (reversed(text) if last else text):
        if char not in _BOUNDARY_MARKS and not char.isspace():
            return char
    return ""


def join_sentences(sentences: tuple[str, ...] | list[str]) -> str:
    """Joins sentences into chunk text: no space where the script uses none.

    The script is judged by the text on either side of the join, looking
    past quotes: straight quotes are used in Chinese text too.
    """
    text = ""
    for sentence in sentences:
        if not sentence:
            continue
        before, after = _edge_char(text, last=True), _edge_char(sentence, last=False)
        spaceless = (before and _is_spaceless(before)) or (after and _is_spaceless(after))
        if text and not spaceless:
            text += " "
        text += sentence
    return text


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
        Maximum characters per chunk of Latin text; scaled down for scripts
        that take longer to say per character (see :func:`char_limit`).
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
    # Each paragraph with the limit that suits its script.
    paragraphs: list[tuple[list[str], int]] = []

    if text and text.strip():
        for paragraph in text.split("\n\n"):
            limit = char_limit(paragraph, max_len)
            pieces = smart_sentence_splitter(paragraph, limit)
            if pieces:
                paragraphs.append((pieces, limit))
    else:
        # No paragraph structure survives in a flat sentence list, so it is
        # read as one paragraph rather than pausing long after every line.
        flat: list[str] = []
        limit = char_limit(" ".join(sentences or []), max_len)
        for sentence in sentences or []:
            if sentence and sentence.strip():
                flat.extend(smart_sentence_splitter(sentence, limit))
        if flat:
            paragraphs.append((flat, limit))

    chunks: list[SpeechChunk] = []
    for pieces, limit in paragraphs:
        pieces = _rejoin_dialogue_tags(pieces, limit)
        groups = _pack(pieces, limit) if pack_sentences else [(p,) for p in pieces]
        for position, group in enumerate(groups):
            last = position == len(groups) - 1
            chunks.append(SpeechChunk(
                text=join_sentences(group),
                pause_after=float(para_pause if last else pause),
                sentences=group,
                paragraph_end=last,
            ))

    if chunks:
        final = chunks[-1]
        chunks[-1] = SpeechChunk(final.text, float(pause), final.sentences, final.paragraph_end)
    return chunks
