"""
test_multilingual_text.py
=========================
Text handling for books that are not in English: chunk length by script,
sentence breaks around quotes and abbreviations, and transcript comparison
for scripts with combining marks.
"""

from __future__ import annotations

import os
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import audiobook_factory.text_processing as text_processing
from audiobook_factory.chunk_planner import char_limit, join_sentences, plan_chunks
from audiobook_factory.chunk_verifier import _normalize_for_compare, error_rate, expected_seconds
from audiobook_factory.text_processing import _repair_sentence_edges, smart_sentence_splitter

_ZH = ("潮水在黎明时分退去，就像1887年以来的每个早晨一样，镇上没有人想起去看它。“今晚风暴会来吗？”面包店的男孩问道。"
       "她没有立刻回答。“快回家吧，”她说，“雨要来了。”到了傍晚，第一场雨来了。")
_JA = "潮は夜明けに引いていった。「今夜、嵐は来ますか？」とパン屋の少年が尋ねた。彼女はすぐには答えなかった。"
_KO = "조수는 새벽에 빠져나갔다. “오늘 밤 폭풍이 올까요?” 빵집 소년이 물었다. 그녀는 바로 대답하지 않았다."
_HI = "ज्वार भोर में उतर गया। डॉ. शर्मा ने कहा कि 250 रुपये काफ़ी हैं। “जल्दी घर जाओ!” उसने कहा।"
_FR = "La marée se retira à l'aube. « Est-ce que la tempête arrivera ce soir ? » demanda le garçon. Mme Leroy attendait."
_RU = "Отлив начался на рассвете. Он жил на ул. Морской, в г. Приморске. Она ответила не сразу."


@pytest.fixture(params=["rust", "python"])
def backend(request, monkeypatch):
    """Runs a test on the Rust splitter and on the pure-Python fallback."""
    if request.param == "rust":
        if not text_processing._RUST_AVAILABLE:
            pytest.skip("Rust extension not built")
    else:
        monkeypatch.setattr(text_processing, "_RUST_AVAILABLE", False)
    return request.param


class TestChunkLengthByScript:

    def test_latin_cyrillic_and_devanagari_keep_the_limit(self):
        for text in (_FR, _RU, _HI, "Plain English prose."):
            assert char_limit(text, 399) == 399

    def test_chinese_gets_about_a_third(self):
        assert 120 <= char_limit(_ZH, 399) <= 160

    def test_japanese_and_korean_sit_in_between(self):
        assert char_limit(_ZH, 399) < char_limit(_KO, 399) < 399
        assert char_limit(_ZH, 399) < char_limit(_JA, 399) < 399

    def test_limit_never_drops_below_a_short_sentence(self):
        assert char_limit(_ZH, 60) == 40
        assert char_limit(_ZH, 10) == 10          # an explicit tiny limit is respected
        assert char_limit("", 399) == 399 and char_limit("   ", 399) == 399

    def test_chunks_of_every_script_take_about_as_long_to_say(self):
        longest = {}
        for name, text in (("en", "The tide went out at dawn, as it had every morning. " * 40),
                           ("zh", _ZH * 8), ("ja", _JA * 12), ("ko", _KO * 12)):
            chunks = plan_chunks(text, None, 399, 0.3, 0.8)
            longest[name] = max(expected_seconds(chunk.text) for chunk in chunks)
        assert all(15.0 <= seconds <= 40.0 for seconds in longest.values()), longest

    def test_long_chinese_sentence_is_cut_at_a_full_width_comma(self, backend):
        sentence = "，".join(["风从中午起就变了"] * 30) + "。"
        pieces = smart_sentence_splitter(sentence, 50)
        assert len(pieces) > 3 and all(len(piece) <= 50 for piece in pieces)
        assert all(piece.endswith(("，", "。")) for piece in pieces)


class TestJoiningSentences:

    def test_spaceless_scripts_are_joined_without_a_space(self):
        assert join_sentences(("她没有立刻回答。", "风变了。")) == "她没有立刻回答。风变了。"
        assert join_sentences(("彼女は答えなかった。", "「早く帰りなさい。」")) == "彼女は答えなかった。「早く帰りなさい。」"

    def test_other_scripts_keep_the_space(self):
        assert join_sentences(("She waited.", "He came.")) == "She waited. He came."
        assert join_sentences(("그녀는 대답하지 않았다.", "바람이 바뀌었다.")) == "그녀는 대답하지 않았다. 바람이 바뀌었다."
        assert join_sentences(("उसने कहा।", "वह चला गया।")) == "उसने कहा। वह चला गया।"

    def test_script_is_judged_past_straight_quotes(self):
        # Extraction straightens curly quotes, in Chinese text as well.
        assert join_sentences(('"已经停了十年了。"', '"为什么没有人修呢？"')) == '"已经停了十年了。""为什么没有人修呢？"'
        assert join_sentences(('"Is it far?"', '"No," he said.')) == '"Is it far?" "No," he said.'

    def test_empty_pieces_are_ignored(self):
        assert join_sentences(("One.", "", "Two.")) == "One. Two."


class TestSentenceEdges:

    def test_chinese_opening_quote_starts_the_next_sentence(self, backend):
        pieces = smart_sentence_splitter(_ZH, 399)
        assert not any(piece.endswith("“") for piece in pieces)
        assert any(piece.startswith("“今晚风暴会来吗？”") for piece in pieces)
        assert "".join(pieces) == _ZH

    def test_japanese_opening_bracket_starts_the_next_sentence(self, backend):
        pieces = smart_sentence_splitter(_JA, 399)
        assert not any(piece.endswith("「") for piece in pieces)
        assert any(piece.startswith("「今夜、嵐は来ますか？」") for piece in pieces)

    def test_straightened_quotes_in_chinese_are_placed_by_counting(self, backend):
        # Extraction turns curly quotes into straight ones before splitting.
        text = _ZH.replace("“", '"').replace("”", '"')
        pieces = smart_sentence_splitter(text, 399)
        assert "".join(pieces) == text
        assert any(piece.startswith('"今晚风暴会来吗？"') for piece in pieces)
        for piece in pieces:
            # No piece ends with a quote that opens the next one.
            assert not (piece.endswith('。"') and piece.count('"') == 1 and not piece.startswith('"')), piece

    def test_french_closing_guillemet_stays_with_its_sentence(self, backend):
        pieces = smart_sentence_splitter(_FR, 399)
        assert any(piece.endswith("ce soir ? »") for piece in pieces)
        assert not any(piece.startswith("»") for piece in pieces)

    def test_abbreviations_do_not_end_a_sentence(self, backend):
        russian = smart_sentence_splitter(_RU, 399)
        assert any("ул. Морской, в г. Приморске." in piece for piece in russian), russian
        hindi = smart_sentence_splitter(_HI, 399)
        assert any(piece.startswith("डॉ. शर्मा") for piece in hindi), hindi
        french = smart_sentence_splitter(_FR, 399)
        assert any("Mme Leroy attendait." in piece for piece in french)

    def test_hindi_danda_ends_a_sentence(self, backend):
        pieces = smart_sentence_splitter(_HI, 399)
        assert pieces[0] == "ज्वार भोर में उतर गया।"

    def test_english_is_unchanged(self, backend):
        text = 'Dr. Hale arrived at 5 p.m. "Is it far?" she asked. He shook his head.'
        pieces = smart_sentence_splitter(text, 399)
        assert " ".join(pieces) == text
        assert pieces[0].startswith("Dr. Hale")

    def test_repair_never_loses_text(self):
        pieces = ["它。“", "今晚？”", "»", "fin.", "ул.", "Морской."]
        repaired = _repair_sentence_edges(pieces, 399)
        assert "".join(repaired).replace(" ", "") == "".join(pieces).replace(" ", "")
        assert _repair_sentence_edges([], 399) == []
        assert _repair_sentence_edges(["“"], 399) == ["“"]

    def test_abbreviation_merge_respects_the_limit(self):
        assert _repair_sentence_edges(["Он жил на ул.", "Морской."], 15) == ["Он жил на ул.", "Морской."]
        assert _repair_sentence_edges(["Он жил на ул.", "Морской."], 399) == ["Он жил на ул. Морской."]


class TestTranscriptComparison:

    def test_devanagari_words_stay_whole(self):
        assert _normalize_for_compare("ज्वार भोर में उतर गया। डॉ. शर्मा") == [
            "ज्वार", "भोर", "में", "उतर", "गया", "डॉ", "शर्मा",
        ]

    def test_other_scripts_with_combining_marks_stay_whole(self):
        assert _normalize_for_compare("สวัสดีครับ ผมชื่อ") == ["สวัสดีครับ", "ผมชื่อ"]
        assert _normalize_for_compare("مَرْحَبًا بِكَ") == ["مَرْحَبًا", "بِكَ"]

    def test_hindi_error_rate_counts_words(self):
        spoken = "ज्वार भोर में उतर गया।"
        assert error_rate(spoken, "ज्वार भोर में उतर गया") == 0.0
        assert error_rate(spoken, "ज्वार शाम में उतर गया") == pytest.approx(0.2)

    def test_english_and_cjk_tokens_are_as_before(self):
        assert _normalize_for_compare("Dr. O'Neil's café—it's 5 o'clock!") == [
            "dr", "o", "neil", "s", "café", "it", "s", "5", "o", "clock",
        ]
        assert _normalize_for_compare("潮は引いた。") == ["潮", "は", "引", "い", "た"]
        assert _normalize_for_compare("snake_case, under_score") == ["snake", "case", "under", "score"]
