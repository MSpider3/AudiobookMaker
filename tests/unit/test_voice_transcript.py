"""
test_voice_transcript.py
========================
The reference transcript is checked before a run starts
(``_resolve_voice_transcript`` in ``pipeline.py``): a path typed where the
words belong is read, and a transcript that cannot match the clip is reported.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import soundfile as sf

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from audiobook_factory.pipeline import AudiobookConfig, _resolve_voice_transcript, _transcript_from_file

_WORDS: str = "Something about the question irritates me. I drift back to sleep."


@pytest.fixture
def clip(tmp_path) -> str:
    """A four-second reference clip."""
    path = tmp_path / "narrator.wav"
    sf.write(path, np.zeros(4 * 24000, dtype=np.float32), 24000)
    return str(path)


def _resolve(config: AudiobookConfig) -> list[str]:
    messages: list[str] = []
    _resolve_voice_transcript(config, messages.append)
    return messages


class TestTranscriptGivenAsAPath:

    def test_path_to_a_text_file_is_replaced_by_its_contents(self, tmp_path, clip):
        # The second Kaggle run: the notebook was given the .txt path, every
        # engine was told the clip says "tests/kaggle/assets/voice/...txt".
        sidecar = tmp_path / "narrator.txt"
        sidecar.write_text(_WORDS + "\n", encoding="utf-8")
        config = AudiobookConfig(voice_file=clip, voice_transcript=str(sidecar))
        messages = _resolve(config)
        assert config.voice_transcript == _WORDS
        assert any("file path" in message and "narrator.txt" in message for message in messages)

    def test_byte_order_mark_is_dropped(self, tmp_path):
        sidecar = tmp_path / "narrator.txt"
        sidecar.write_text(_WORDS, encoding="utf-8-sig")
        assert _transcript_from_file(str(sidecar)) == _WORDS

    def test_ordinary_text_is_not_treated_as_a_path(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        assert _transcript_from_file(_WORDS) is None
        assert _transcript_from_file("") is None
        assert _transcript_from_file("first line\nsecond line") is None
        assert _transcript_from_file(str(tmp_path)) is None  # a directory is not a transcript

    def test_empty_file_clears_the_transcript_without_error(self, tmp_path, clip):
        sidecar = tmp_path / "empty.txt"
        sidecar.write_text("", encoding="utf-8")
        config = AudiobookConfig(voice_file=clip, voice_transcript=str(sidecar))
        _resolve(config)
        assert config.voice_transcript == ""


class TestImplausibleTranscript:

    def test_matching_transcript_is_accepted_silently(self, clip):
        config = AudiobookConfig(voice_file=clip, voice_transcript=_WORDS)
        assert _resolve(config) == []
        assert config.voice_transcript == _WORDS

    def test_one_word_for_a_long_clip_is_reported(self, clip):
        config = AudiobookConfig(voice_file=clip, voice_transcript="missing/transcript.txt")
        messages = _resolve(config)
        assert len(messages) == 1
        assert "1 word(s) for a 4-second clip" in messages[0]
        assert config.voice_transcript == "missing/transcript.txt"  # reported, not rewritten

    def test_far_too_many_words_is_reported(self, clip):
        config = AudiobookConfig(voice_file=clip, voice_transcript=" ".join(["word"] * 60))
        assert any("60 word(s)" in message for message in _resolve(config))

    def test_scripts_without_word_spacing_are_not_judged(self, clip):
        config = AudiobookConfig(voice_file=clip, voice_transcript="这个问题让我有点恼火。")
        assert _resolve(config) == []

    def test_nothing_to_check_without_a_transcript_or_clip(self, clip):
        assert _resolve(AudiobookConfig(voice_file=clip, voice_transcript="")) == []
        assert _resolve(AudiobookConfig(voice_file="", voice_transcript="one")) == []
        assert _resolve(AudiobookConfig(voice_file="/no/such/clip.wav", voice_transcript="one")) == []
