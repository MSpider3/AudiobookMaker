"""
test_speech_text.py
===================
Unit tests for ``audiobook_factory.speech_text``: the written-form to
spoken-form table (what must change), the negative table (what must not),
and the structural guarantees (idempotence, paragraph preservation,
never raising, speed).
"""

from __future__ import annotations

import os
import random
import sys
import time

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from audiobook_factory.speech_text import (  # noqa: E402
    normalize_for_speech,
    supported_languages,
)

_FIXTURE_TXT = os.path.join(
    _ROOT, "tests", "fixtures", "source_documents", "dummy_book.txt"
)

# ── English: written form -> spoken form ─────────────────────────────────────

_ENGLISH_CASES: list[tuple[str, str]] = [
    # Roman numerals after a label word (cardinal)
    ("Chapter IV", "Chapter four"),
    ("Chapter I", "Chapter one"),
    ("Chapter I.", "Chapter one."),
    ('She said, "Chapter IV is next."', 'She said, "Chapter four is next."'),
    ("(Louis XIV)", "(Louis the fourteenth)"),
    ("CHAPTER XII", "CHAPTER TWELVE"),
    ("Chapter XL", "Chapter forty"),
    ("Part II: The Return", "Part two: The Return"),
    ("Book V", "Book five"),
    ("Volume III of the series", "Volume three of the series"),
    ("Act I, Scene II", "Act one, Scene two"),
    ("Appendix II", "Appendix two"),
    ("Canto XXXIV", "Canto thirty-four"),
    ("Section IX covers this.", "Section nine covers this."),
    ("see chapter IV for details", "see chapter four for details"),
    ("Chapters IV, V and VI", "Chapters four, five and six"),
    ("Parts I and II", "Parts one and two"),
    ("Vol. IV", "Volume four"),
    ("Title IX", "Title nine"),
    ("Stage IV cancer", "Stage four cancer"),
    ("Part I of the trilogy", "Part one of the trilogy"),
    # Regnal names (ordinal) and war / event numbering (cardinal)
    ("Louis XIV", "Louis the fourteenth"),
    ("Henry VIII had six wives.", "Henry the eighth had six wives."),
    ("Henry VIII's court", "Henry the eighth's court"),
    ("Elizabeth II", "Elizabeth the second"),
    ("Pope John Paul II", "Pope John Paul the second"),
    ("Pope Benedict XVI", "Pope Benedict the sixteenth"),
    ("King Charles I was beheaded.", "King Charles the first was beheaded."),
    ("Queen Elizabeth I", "Queen Elizabeth the first"),
    ("Tsar Nicholas II", "Tsar Nicholas the second"),
    ("Henry V was king.", "Henry the fifth was king."),
    ("King William IV's own engineer", "King William the fourth's own engineer"),
    ("World War II", "World War two"),
    ("World War I began in Europe.", "World War one began in Europe."),
    ("Super Bowl XLII", "Super Bowl forty-two"),
    # Standalone heading lines
    ("IV", "Four"),
    ("XII.", "Twelve."),
    ("I\n\nIt began.\n\nII\n\nIt ended.", "One\n\nIt began.\n\nTwo\n\nIt ended."),
    # Cardinals: separators, decimals, negatives, long round numbers
    (
        "1,234,567",
        "one million two hundred thirty-four thousand five hundred sixty-seven",
    ),
    ("It rose by 2,000 feet.", "It rose by two thousand feet."),
    ("3.14", "three point one four"),
    ("0.5", "zero point five"),
    ("It was -5 outside.", "It was minus five outside."),
    ("250000 people", "two hundred fifty thousand people"),
    ("1000000", "one million"),
    ("10,000-12,000 men", "ten thousand to twelve thousand men"),
    # Years, ranges, decades, eras
    ("1914-1918", "nineteen fourteen to nineteen eighteen"),
    ("1914-18", "nineteen fourteen to nineteen eighteen"),
    ("(1856-1939)", "(eighteen fifty-six to nineteen thirty-nine)"),
    ("the 1990s", "the nineteen nineties"),
    ("the '90s", "the nineties"),
    ("mid-1800s", "mid-eighteen hundreds"),
    ("the 2000s", "the two thousands"),
    ("in his 40s", "in his forties"),
    ("in the 50s or 60s", "in the fifties or sixties"),
    ("in 2005", "in two thousand five"),
    ("in 1905", "in nineteen oh five"),
    ("in 1900", "in nineteen hundred"),
    ("in 2021", "in twenty twenty-one"),
    (
        "In 1492 Columbus sailed the ocean blue.",
        "In fourteen ninety-two Columbus sailed the ocean blue.",
    ),
    ("In 2000, everything changed.", "In two thousand, everything changed."),
    ("since 1999", "since nineteen ninety-nine"),
    ("the summer of 1914", "the summer of nineteen fourteen"),
    ("from 1939 to 1945", "from nineteen thirty-nine to nineteen forty-five"),
    ("AD 79", "AD seventy-nine"),
    ("300 BC", "three hundred BC"),
    ("A.D. 1492", "A.D. fourteen ninety-two"),
    ("44 BCE", "forty-four BCE"),
    # Ordinals
    ("1st", "first"),
    ("2nd", "second"),
    ("3rd", "third"),
    ("21st", "twenty-first"),
    ("100th", "one hundredth"),
    ("the 19th century", "the nineteenth century"),
    ("her 12th birthday", "her twelfth birthday"),
    # Currency
    ("$5", "five dollars"),
    ("$1", "one dollar"),
    ("$5.50", "five dollars and fifty cents"),
    ("$1,200", "one thousand two hundred dollars"),
    ("$3 million", "three million dollars"),
    ("$2.5 billion", "two point five billion dollars"),
    ("$5M", "five million dollars"),
    ("$0.99", "ninety-nine cents"),
    ("$1.01", "one dollar and one cent"),
    ("£10", "ten pounds"),
    ("£1", "one pound"),
    ("€7.20", "seven euros and twenty cents"),
    ("¥500", "five hundred yen"),
    ("₹250", "two hundred fifty rupees"),
    ("5 USD", "five U.S. dollars"),
    ("US$20", "twenty U.S. dollars"),
    ("a $5 bill", "a five dollar bill"),
    ("two $5 bills", "two five dollar bills"),
    ("a £5 note", "a five pound note"),
    ("$5-$10", "five to ten dollars"),
    ("50¢", "fifty cents"),
    # Percentages and fractions
    ("12%", "twelve percent"),
    ("3.5 %", "three point five percent"),
    ("5-10%", "five to ten percent"),
    ("100%", "one hundred percent"),
    ("1/2", "one half"),
    ("3/4", "three quarters"),
    ("2 1/2", "two and a half"),
    ("1 3/4 cups", "one and three quarters cups"),
    ("1⁄2 cup", "one half cup"),
    ("2/3 of the men", "two thirds of the men"),
    ("½", "one half"),
    ("2½", "two and a half"),
    # Math and symbols
    ("2 + 2 = 4", "2 plus 2 equals 4"),
    ("3 × 4", "3 times 4"),
    ("5 - 3 = 2", "5 minus 3 equals 2"),
    ("20°C", "twenty degrees Celsius"),
    ("98.6°F", "ninety-eight point six degrees Fahrenheit"),
    ("100°", "one hundred degrees"),
    ("-5°C", "minus five degrees Celsius"),
    ("#5", "number five"),
    ("Tom & Jerry", "Tom and Jerry"),
    ("No. 5", "Number five"),
    ("Sealed Artifact No. 125.", "Sealed Artifact Number one hundred twenty-five."),
    ("Nos. 3 and 4", "Numbers three and four"),
    ("x²", "x squared"),
    ("~5 minutes", "about 5 minutes"),
    ("5+ years", "5 plus years"),
    ("E = mc²", "E equals mc squared"),
    ("Copyright © 2026 Mara", "Copyright 2026 Mara"),
    # Abbreviations
    ("Smith vs. Jones", "Smith versus Jones"),
    ("Roe v. Wade", "Roe versus Wade"),
    ("apples, pears, etc. and more", "apples, pears, et cetera and more"),
    ("apples, pears, etc.", "apples, pears, et cetera."),
    ("pens, paper, etc, and more", "pens, paper, et cetera, and more"),
    (
        "Bring pens, paper, etc. The test is hard.",
        "Bring pens, paper, et cetera. The test is hard.",
    ),
    ("fruit, e.g. apples", "fruit, for example, apples"),
    ("e.g., apples", "for example, apples"),
    ("i.e. the king", "that is, the king"),
    ("approx. 50 people", "approximately 50 people"),
    ("St. Louis", "Saint Louis"),
    ("He went with St. Louis in mind.", "He went with Saint Louis in mind."),
    ("the church of St. Peter", "the church of Saint Peter"),
    ("Main St.", "Main Street."),
    ("He lived on Baker St. for years.", "He lived on Baker Street for years."),
    ("42nd St.", "forty-second Street."),
    ("Mt. Everest", "Mount Everest"),
    ("Ft. Worth", "Fort Worth"),
    ("Martin Luther King Jr. was there.", "Martin Luther King Junior was there."),
    ("John Smith Sr.", "John Smith Senior."),
    ("Prof. Moriarty", "Professor Moriarty"),
    ("Gen. Lee", "General Lee"),
    ("Capt. Ahab", "Captain Ahab"),
    ("Lt. Dan", "Lieutenant Dan"),
    ("Sgt. Pepper", "Sergeant Pepper"),
    ("Rev. Brown", "Reverend Brown"),
    ("Hon. Jane Doe", "Honorable Jane Doe"),
    ("Gov. Brown", "Governor Brown"),
    ("Sen. Smith", "Senator Smith"),
    ("Rep. Jones", "Representative Jones"),
    ("Lt. Col. Sanders", "Lieutenant Colonel Sanders"),
    ("Acme Inc.", "Acme Incorporated."),
    ("p. 47", "page 47"),
    ("pp. 10-12", "pages 10 to 12"),
    ("Fig. 3", "Figure 3"),
    ("c. 1900", "circa nineteen hundred"),
    # Times and dates
    ("10:30", "ten thirty"),
    ("10:30 p.m.", "ten thirty p.m."),
    ("12:45 PM", "twelve forty-five PM"),
    ("7am", "seven a.m."),
    ("at 7 pm sharp", "at seven p.m. sharp"),
    ("We met at 7pm.", "We met at seven p.m."),
    ("12:00", "twelve o'clock"),
    ("9:05", "nine oh five"),
    ("John 3:16", "John three sixteen"),
    ("Jan. 5, 1920", "January fifth, nineteen twenty"),
    ("May 5th, 1920", "May fifth, nineteen twenty"),
    ("December 25, 2020", "December twenty-fifth, twenty twenty"),
    ("5th of May", "fifth of May"),
    ("on the 3rd of March, 1926, she came", "on the third of March, nineteen twenty-six, she came"),
    ("on 5 January 1920", "on the fifth of January nineteen twenty"),
    ("May 1920", "May nineteen twenty"),
    ('"12th May. Mr. Hale came."', '"the twelfth of May. Mr. Hale came."'),
    (
        "15th September 1323, 2:00PM.",
        "the fifteenth of September thirteen twenty-three, two PM.",
    ),
    # Units
    ("5km", "five kilometers"),
    ("1 km", "one kilometer"),
    ("10 kg", "ten kilograms"),
    ("6ft", "six feet"),
    ("He was 6 ft. tall.", "He was six feet tall."),
    ("a 10 lb baby", "a ten pound baby"),
    ("5'10\"", "five feet ten inches"),
    ("30mph", "thirty miles per hour"),
    ("3.14159 MHz", "three point one four one five nine megahertz"),
    ("a 5-km run", "a five-kilometer run"),
    ("5 km²", "five square kilometers"),
    ("5 m² of floor", "five square meters of floor"),
    ("a 6-ft fence", "a six-foot fence"),
    ("5-10 km", "five to ten kilometers"),
    # Phone numbers and identifiers
    ("555-123-4567", "five five five, one two three, four five six seven"),
    ("(555) 123-4567", "five five five, one two three, four five six seven"),
    ("Call 555-1234 now.", "Call five five five, one two three four now."),
    (
        "account number 12345678",
        "account number one two three four five six seven eight",
    ),
    ("PIN 4821", "PIN four eight two one"),
    ("1-800-555-0199", "one, eight zero zero, five five five, zero one nine nine"),
    ("area code 212", "area code two one two"),
    ("zip code 90210", "zip code nine zero two one zero"),
    # URLs, e-mail, footnote markers, superscripts
    ("www.example.com", "example dot com"),
    ("https://www.example.com/path?x=1", "example dot com"),
    ("Visit http://example.org.", "Visit example dot org."),
    ("Amazon.com", "Amazon dot com"),
    ("john.doe@example.com", "john dot doe at example dot com"),
    ("word[12]", "word"),
    ("It ended.[3] Next came war.", "It ended. Next came war."),
    ("a bold claim[a] indeed", "a bold claim indeed"),
    ("the treaty [12] was signed", "the treaty was signed"),
    ("word¹²", "word"),
    ("It ended.¹", "It ended."),
    # Symbol noise
    ("*sigh*", "sigh"),
    ("snake_case", "snake case"),
    ("Hello~", "Hello"),
    ("a | b", "a, b"),
    ("Wow!!!", "Wow!"),
    ("What?!?!", "What?!"),
    ("Really???", "Really?"),
    ("Wait . . . what", "Wait... what"),
    ("Wait.... what", "Wait... what"),
    ("• First item", "First item"),
    ("A → B", "A, B"),
    ("Nice ★ work", "Nice, work"),
    ("Brand™ name", "Brand name"),
    ("word--word", "word, word"),
    ("^_^ hello", "hello"),
    ("5~10 people", "5 to 10 people"),
    ("Fool~!", "Fool!"),
    ("*Rubs hands nefariously.*", "Rubs hands nefariously."),
    ("==== Title ====", "Title"),
    ("Sword of ______ Victory.", "Sword of ... Victory."),
    ("| Monday | 6:10 a.m. | 12:25 p.m. |", "Monday, six ten a.m. twelve twenty-five p.m."),
    # All-caps runs of three or more words
    ("THE CRIMSON TOWER", "The Crimson Tower"),
    ("GET OUT OF MY HOUSE!", "Get out of my house!"),
    (
        'He yelled, "GET OUT OF MY HOUSE!" and left.',
        'He yelled, "Get out of my house!" and left.',
    ),
    ("THE NASA REPORT WAS LATE.", "The NASA report was late."),
    ("I AM NOT GOING.", "I am not going."),
    ("CHAPTER IV: THE STORM BREAKS", "Chapter Four: The Storm Breaks"),
    ("STOP! DON'T MOVE! I MEAN IT!", "Stop! Don't move! I mean it!"),
    ("I AM A GOD. EPUB, MOBI, PDF, DOCX.", "I am a god. EPUB, MOBI, PDF, DOCX."),
    ("MR. SMITH GOES TO WASHINGTON", "Mr. Smith Goes to Washington"),
    ("O'BRIEN WAS HERE TODAY.", "O'Brien was here today."),
    ("CHAPTER 16: RAT-BAITING WITH DOGS", "Chapter 16: Rat-Baiting with Dogs"),
    ("CHAPTER 24: PENNY-PINCHER", "Chapter 24: Penny-Pincher"),
    ("CHAPTER 27: SIBLINGS' DINNER", "Chapter 27: Siblings' Dinner"),
    ("CHAPTER 98: MR. AZIK", "Chapter 98: Mr. Azik"),
    ("JOHN F. KENNEDY WAS SHOT TODAY", "John F. Kennedy Was Shot Today"),
    ("The CEO of IBM AND HP MET TODAY.", "The CEO of IBM and HP met today."),
    ("20000 Leagues Under the Sea", "twenty thousand Leagues Under the Sea"),
]

# ── English: must stay exactly as written ────────────────────────────────────

_UNCHANGED: list[str] = [
    "",
    "   ",
    "\n\n",
    "Plain prose with nothing to convert, only words and commas.",
    'She said, "Hello there," and smiled.',
    "Hello…",
    # The pronoun, initials and single letters
    "I",
    "I am here.",
    "I think, therefore I am.",
    "I'm sure I'll be fine, and I've said so.",
    "I, Claudius",
    "Plan B",
    "Vitamin C",
    "Vitamin D",
    "Malcolm X",
    "John D. Rockefeller",
    "J. R. R. Tolkien",
    "Mr. X",
    "MIX the batter",
    "a CIVIL matter",
    "He DID it",
    "LIVID with rage",
    "the X-Men",
    "Generation X",
    "Planet X",
    "the letter V",
    "the part I played",
    "The book I read was long.",
    "James I told you so.",
    "I told Charles I would come.",
    "Henry V. Smith",
    "Section C",
    "Appendix C",
    "Class C",
    "size XL",
    "XXX",
    "Rocky II",
    "Mark I think is right",
    "Peter I think you're wrong.",
    "They watched Henry V. It was long.",
    "Final Fantasy VII",
    "an IV line",
    "Volume I contains errors.",
    "the world war I mean is the second",
    # Acronyms and short all-caps tokens
    "OK",
    "NASA",
    "NASA and the FBI",
    "CHAPTER ONE",
    "A B C D",
    "He joined the CIA in secret.",
    "USA TODAY reported.",
    "EPUB, MOBI, PDF, DOCX",
    '"NO! NO! NO!" she screamed.',
    "the FBI AND CIA",
    "the FBI AND CIA. USA TODAY reported.",
    "H. H. OBIIT MDCCCXXXIII: 11.",
    "It was a TOP SECRET file.",
    # Titles deliberately left alone
    "Mr. Smith",
    "Mrs. Jones",
    "Ms. Lee",
    "Dr. Watson",
    "Dr. Arthur Pendelton, Ph.D.",
    # Ambiguous abbreviations
    "Main St. Louis walked",
    "King Jr. Boulevard",
    "No. I won't.",
    "no. 7",
    "He said no. 5 people came.",
    "The answer was No. 5 minutes later, he left.",
    "Gamma Inc. He left.",
    "a Sr. partner",
    "Jones v. the state",
    "the U.S. Army",
    # Plain numbers and number-like tokens
    "12 apples",
    "page 247",
    "his 1888 journal",
    "2005 was a good year.",
    "1905 soldiers",
    "in 1905 cases",
    "in 2000 years",
    "100 men",
    "the 3 of us",
    "room 101",
    "007",
    "Boeing 747",
    "A4 paper",
    "v2.3.1",
    "192.168.1.1",
    "Catch-22",
    "COVID-19",
    "R2-D2",
    "9/11",
    "24/7",
    "4x4",
    "M16",
    "B-52",
    "1080p",
    "5G network",
    "Windows 11",
    "Apollo 13",
    "3/4/2021",
    "on 3/4",
    "2021-03-04",
    "ISBN 978-3-16-148410-0",
    "0306406152",
    "123456789",
    "90210",
    "3:2 win",
    "10:30:15",
    "January 5",
    "5 May",
    "and/or",
    "1/5",
    "5m",
    "5 minutes",
    "11st",
    "On May 5 he left",
    "20 June.",
    "5 CAD drawings",
    "7/8 time",
    "page 1/2",
    "50/50",
    "iOS 14.5.1",
    "a .38 revolver",
    "1.5-2 hours",
    "X-5",
    "Sequence 9",
    "Sealed Artifact 2-049",
    "Job 24:8",
    # Symbols left alone
    "AT&T",
    "R&D",
    "@john",
    "C++",
    "A+",
    "#metoo",
    "$",
    "R$20",
    "f*ck",
    "[sic]",
    "[1] See Smith.",
    "[A]",
    "C# code",
    "F**k!",
    "as shown in [3]",
    "see [12] for details",
]

# ── Other languages: language-neutral cleanups only ──────────────────────────

_OTHER_LANGUAGE_CASES: list[tuple[str, str, str]] = [
    ("第12章[3]", "Chinese", "第12章"),
    ("太好了！！！", "Chinese", "太好了！"),
    ("★ 第一章", "Chinese", "第一章"),
    ("10~20人", "Japanese", "10~20人"),
    ("価格は5,000円です。", "Japanese", "価格は5,000円です。"),
    ("가격은 $5.50 입니다.", "Korean", "가격은 $5.50 입니다."),
    ("Das kostet 5,50 € und 12 %.", "German", "Das kostet 5,50 € und 12 %."),
    ("Kapitel IV", "German", "Kapitel IV"),
    ("Voir https://www.example.com/page !!!", "French", "Voir example.com !"),
    ("Louis XIV en 1914-1918", "French", "Louis XIV en 1914-1918"),
    ("palabra¹² aquí", "Spanish", "palabra aquí"),
    ("В 1905 году... что?!?!", "Russian", "В 1905 году... что?!"),
    ("Capitolo IV costa $5", "Italian", "Capitolo IV costa $5"),
    ("Chapter IV cost $5", "Auto", "Chapter IV cost $5"),
    ("Chapter IV cost $5", "Klingon", "Chapter IV cost $5"),
    ("Wow!!! *sigh*", "Auto", "Wow! sigh"),
]


@pytest.mark.parametrize(("written", "spoken"), _ENGLISH_CASES)
def test_english_written_to_spoken(written: str, spoken: str) -> None:
    assert normalize_for_speech(written) == spoken


@pytest.mark.parametrize("text", _UNCHANGED)
def test_english_left_unchanged(text: str) -> None:
    assert normalize_for_speech(text) == text


@pytest.mark.parametrize(("written", "language", "spoken"), _OTHER_LANGUAGE_CASES)
def test_other_languages_neutral_only(written: str, language: str, spoken: str) -> None:
    assert normalize_for_speech(written, language) == spoken


def test_table_is_large_enough() -> None:
    total = len(_ENGLISH_CASES) + len(_UNCHANGED) + len(_OTHER_LANGUAGE_CASES)
    assert total >= 150
    assert len(_UNCHANGED) >= 60


class TestApi:

    def test_supported_languages(self) -> None:
        languages = supported_languages()
        assert isinstance(languages, tuple)
        assert "English" in languages

    def test_default_language_is_english(self) -> None:
        assert normalize_for_speech("Chapter IV") == "Chapter four"

    @pytest.mark.parametrize("language", ["English", "english", " ENGLISH ", "en"])
    def test_language_name_is_forgiving(self, language: str) -> None:
        assert normalize_for_speech("Chapter IV", language) == "Chapter four"

    @pytest.mark.parametrize("language", [None, 5, "", "Auto"])
    def test_unknown_language_is_neutral(self, language: object) -> None:
        assert normalize_for_speech("Chapter IV!!!", language) == "Chapter IV!"

    def test_non_string_input(self) -> None:
        assert normalize_for_speech(None) == ""  # type: ignore[arg-type]
        assert normalize_for_speech(12) == "12"  # type: ignore[arg-type]

    def test_builtin_speller_needs_no_num2words(self) -> None:
        # The English rule set must work on Colab/Kaggle without extra deps:
        # the module spells numbers itself and never imports num2words.
        import audiobook_factory.speech_text as speech_text

        with open(speech_text.__file__, encoding="utf-8") as handle:
            assert "import num2words" not in handle.read()
        assert normalize_for_speech("$1,234.56") == (
            "one thousand two hundred thirty-four dollars and fifty-six cents"
        )


class TestStructure:

    _SEPARATORS: tuple[str, ...] = (" ", "\n", "\n\n", ". ", ", ", "  ", " - ")

    def _compositions(self, count: int, seed: int) -> list[str]:
        rng = random.Random(seed)
        pool = (
            [written for written, _ in _ENGLISH_CASES]
            + [spoken for _, spoken in _ENGLISH_CASES]
            + _UNCHANGED
            + [written for written, _, _ in _OTHER_LANGUAGE_CASES]
        )
        texts: list[str] = []
        for _ in range(count):
            parts = [rng.choice(pool) for _ in range(rng.randint(1, 5))]
            text = parts[0]
            for part in parts[1:]:
                text += rng.choice(self._SEPARATORS) + part
            texts.append(text)
        return texts

    @pytest.mark.parametrize("language", ["English", "Auto", "Chinese"])
    def test_idempotent_on_random_compositions(self, language: str) -> None:
        for text in self._compositions(3000, seed=20260607):
            once = normalize_for_speech(text, language)
            assert normalize_for_speech(once, language) == once, repr(text)

    @pytest.mark.parametrize(("written", "spoken"), _ENGLISH_CASES)
    def test_expected_outputs_are_stable(self, written: str, spoken: str) -> None:
        assert normalize_for_speech(spoken) == spoken

    @pytest.mark.parametrize("language", ["English", "Auto"])
    def test_paragraph_structure_preserved(self, language: str) -> None:
        for text in self._compositions(3000, seed=77):
            result = normalize_for_speech(text, language)
            assert result.count("\n\n") == text.count("\n\n"), repr(text)
            assert result.count("\n") == text.count("\n"), repr(text)
            blank_in = [not line.strip() for line in text.split("\n")]
            blank_out = [not line.strip() for line in result.split("\n")]
            assert blank_in == blank_out, repr(text)

    def test_line_made_only_of_removable_marks_is_kept(self) -> None:
        text = "First paragraph.\n***\nSecond paragraph.\n[12]\n★ ★ ★\nThird."
        result = normalize_for_speech(text)
        assert result.count("\n") == text.count("\n")
        assert result.count("\n\n") == 0

    def test_edge_whitespace_and_crlf_preserved(self) -> None:
        assert normalize_for_speech("  Chapter IV  \r\n\r\n\tIt cost $5.\r\n") == (
            "  Chapter four  \r\n\r\n\tIt cost five dollars.\r\n"
        )

    @pytest.mark.parametrize("language", ["English", "Auto", "Japanese"])
    def test_never_raises_on_random_unicode(self, language: str) -> None:
        rng = random.Random(4242)
        alphabet = (
            "abcIVXLC019 .,!?$%&*_~|^#@[]()'\"-/:°\n\t\r"
            "½²¹…•→★™©€£¥₹×÷±​﻿­\ud800\U0001f600章ñЖ"
        )
        for _ in range(1500):
            length = rng.randint(0, 60)
            if rng.random() < 0.5:
                text = "".join(rng.choice(alphabet) for _ in range(length))
            else:
                text = "".join(chr(rng.randint(0, 0x2FFF)) for _ in range(length))
            result = normalize_for_speech(text, language)
            assert isinstance(result, str)
            assert result.count("\n") == text.count("\n")

    @pytest.mark.parametrize("text", ["", " ", "\n", "\n\n\n", "\t \r\n ", "\x00", "\ud800"])
    def test_degenerate_input_returned_unchanged(self, text: str) -> None:
        assert normalize_for_speech(text) == text
        assert normalize_for_speech(text, "Auto") == text

    def test_one_megabyte_single_line(self) -> None:
        unit = "It cost $5 in 1905. "
        text = unit * (1_048_576 // len(unit) + 1)
        assert len(text) >= 1_048_576 and "\n" not in text
        result = normalize_for_speech(text)
        assert result.startswith("It cost five dollars in nineteen oh five. It cost")
        assert "$" not in result and "\n" not in result
        assert normalize_for_speech("x" * 1_048_576) == "x" * 1_048_576
        noise = "a1$ " * 262_144
        assert normalize_for_speech(noise) == noise

    def test_fast_on_a_book_sized_text(self) -> None:
        paragraph = (
            "The rain had not stopped for three days, and I was beginning to "
            "think it never would. \"We should go,\" Eleanor said, pulling her "
            "coat tighter. I nodded, though I did not move; the harbour lights "
            "were still burning, and somewhere below us a bell was ringing.\n\n"
            "Mr. Hale arrived at 10:30 p.m. with Prof. Arden, who had paid "
            "$1,250 for the map in 1905 and regretted it ever since.\n\n"
            "Nobody spoke. The clock ticked on, and the fire sank lower, and "
            "at last the old man laughed, a dry and private sound.\n\n"
        )
        text = paragraph * (1_000_000 // len(paragraph) + 1)
        assert len(text) >= 1_000_000
        started = time.perf_counter()
        result = normalize_for_speech(text)
        elapsed = time.perf_counter() - started
        assert "ten thirty p.m. with Professor Arden" in result
        assert result.count("\n\n") == text.count("\n\n")
        assert elapsed < 5.0, f"too slow: {elapsed:.2f}s for {len(text)} characters"


_SAMPLE_PAGE: str = (
    "CHAPTER IV\n\n"
    "Dr. Arthur Pendelton, Ph.D., reached the tower at 12:45 PM on Nov. 15th, "
    "2026. \"The anomalies began at exactly 3.14159 MHz,\" whispered Eleanor, "
    "\"just as Prof. Moriarty predicted in his 1888 journal.\"[12]\n\n"
    "The tapestry showed the battle of Saint-Germain in A.D. 1492. She turned "
    "to page 247 and read section 4(b) again: it had cost Louis XIV $1,250, "
    "i.e. 3% of the treasury.\n\n"
    "* * *\n\n"
    "I looked at her. \"I think Plan B is all we have,\" I said."
)


class TestSamplePage:

    def test_expected_readings(self) -> None:
        result = normalize_for_speech(_SAMPLE_PAGE)
        assert result.startswith("CHAPTER FOUR\n\n")
        assert "at twelve forty-five PM on November fifteenth, twenty twenty-six." in result
        assert "three point one four one five nine megahertz" in result
        assert "Professor Moriarty" in result
        assert 'journal."\n\n' in result          # footnote marker gone
        assert "A.D. fourteen ninety-two" in result
        assert "Louis the fourteenth one thousand two hundred fifty dollars" in result
        assert "that is, three percent of the treasury" in result

    def test_left_alone_on_purpose(self) -> None:
        result = normalize_for_speech(_SAMPLE_PAGE)
        assert "Dr. Arthur Pendelton, Ph.D." in result
        assert "1888 journal" in result
        assert "page 247" in result
        assert "section 4(b)" in result
        assert "\n\n* * *\n\n" in result
        assert result.endswith('I looked at her. "I think Plan B is all we have," I said.')

    def test_structure(self) -> None:
        result = normalize_for_speech(_SAMPLE_PAGE)
        assert result.count("\n\n") == _SAMPLE_PAGE.count("\n\n")
        assert normalize_for_speech(result) == result


@pytest.fixture(scope="module")
def fixture_book() -> str:
    if not os.path.exists(_FIXTURE_TXT):
        pytest.skip("dummy_book.txt fixture not present")
    with open(_FIXTURE_TXT, encoding="utf-8") as handle:
        return handle.read()


class TestFixtureBook:

    def test_structure_and_idempotence(self, fixture_book: str) -> None:
        result = normalize_for_speech(fixture_book)
        assert result.count("\n\n") == fixture_book.count("\n\n")
        assert len(result.split("\n")) == len(fixture_book.split("\n"))
        assert normalize_for_speech(result) == result

    def test_other_languages_only_get_neutral_cleanups(self, fixture_book: str) -> None:
        result = normalize_for_speech(fixture_book, "Auto")
        digits_in = [c for c in fixture_book if c.isdigit()]
        digits_out = [c for c in result if c.isdigit()]
        # Footnote markers such as "[1]" may go; nothing is ever spelled out.
        assert len(digits_in) - 8 <= len(digits_out) <= len(digits_in)
        for symbol in "$£€%":
            assert result.count(symbol) == fixture_book.count(symbol)
