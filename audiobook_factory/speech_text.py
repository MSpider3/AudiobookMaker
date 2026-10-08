"""
audiobook_factory/speech_text.py

Text normalisation for speech: rewrites written forms a neural TTS model
tends to misread ("Chapter IV", "$5.50", "1914-1918", "3rd", "No. 5",
"10:30 p.m.", "[12]", "www.example.com") into forms it can simply read.

Runs *after* extraction cleanup (``TextNormalizer`` in ``extractor_engine``
has already straightened smart quotes, turned em-dashes into ", " and
en-dashes into "-", and stripped markdown / HTML entities) and *before*
sentence splitting.

Rule zero: never make the text worse. A rule fires only when the written
form is unambiguous; anything it is not sure about is left exactly as written.

Guarantees
----------
* Paragraph structure is untouched: the text is processed line by line, a line
  never gains or loses a newline and a non-blank line never becomes blank.
* Idempotent: every line is rewritten to a fixed point, so
  ``normalize_for_speech(normalize_for_speech(x)) == normalize_for_speech(x)``.
* Never raises: each rule is isolated; a failing rule is logged at debug level
  and skipped.
* Text with no digits, symbols, abbreviations or all-caps runs is returned
  unchanged, byte for byte.

Policy (the judgement calls)
----------------------------
Plain integers
    Left as digits ("12 apples", "page 247", "his 1888 journal"). Modern TTS
    reads them correctly and context-free spelling can only add errors.
    A number is spelled out only when it is part of a written form that is
    rewritten anyway (currency, percent, ordinal, time, fraction, unit,
    "No. 5", negative sign) or is itself awkward: thousands separators
    ("1,234,567"), decimals ("3.14" -> "three point one four") and long round
    numbers without separators ("250000"). American style, no "and".
Years
    A bare four-digit number stays as digits ("1905 soldiers" cannot be told
    from a year). It is read as a year only in an explicit year context: a
    range ("1914-1918" -> "nineteen fourteen to nineteen eighteen", fully
    worded so nothing is left for the model to guess and nothing is left for
    a second pass to re-match), a decade ("the 1990s", "'90s", "mid-1800s"),
    an era ("AD 79", "300 BC"), a date ("Jan. 5, 1920", "May 1920") or after
    a year preposition ("in 1905", "since 2000", "the summer of 1914") when
    what follows is not a counted noun ("in 1905 cases" is left alone).
Mr. / Mrs. / Ms. / Dr.
    Left alone. They are among the most frequent tokens in any TTS training
    set and are read correctly; "Dr." is also ambiguous (Doctor / Drive).
    Rank and title abbreviations that models do stumble on ("Prof.", "Capt.",
    "Lt.", "Sgt.", "Gen.", "Rev.", "Gov.", "Sen.", "St.", "Mt.", "Ft.", ...)
    are expanded, and only directly before a capitalised name.
    Trailing abbreviations ("Jr.", "Inc.", "St." as Street) are left alone
    when it is unclear whether their period also ends the sentence.
ALL CAPS
    Left as written, except a run of three or more all-caps words (a shouted
    sentence or a heading), which becomes sentence case - or title case when
    the run is a whole heading line. Known acronyms, vowel-less tokens
    ("NYPD"), roman numerals and the pronoun "I" inside such a run keep their
    capitals. Isolated all-caps tokens ("NASA", "OK", "MIX"), two-word shouts
    and comma-separated acronym lists ("EPUB, MOBI, PDF") are never touched.
Roman numerals
    Converted only after a label word ("Chapter IV" -> "Chapter four"),
    after a regnal name or royal title ("Louis XIV" -> "Louis the
    fourteenth"), or as a standalone heading line. The pronoun "I", initials
    and single letters ("Plan B", "Vitamin C", "Malcolm X") are never touched.
Other languages
    Only the language-neutral cleanups run (footnote markers, URLs, symbol
    noise, repeated punctuation, superscripts, decorative unicode). Numbers
    are never rewritten outside English: the models read native numerals
    correctly, and number grammar is language specific. ``num2words`` is
    deliberately not used.
"""
from __future__ import annotations

__all__ = ["normalize_for_speech", "supported_languages"]

import logging
import re
from typing import Callable

logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════════════

_FULL_RULESET_LANGUAGES: tuple[str, ...] = ("English",)
# Languages with a written-form rule set. Everything else gets neutral cleanups.

_ENGLISH_ALIASES: frozenset[str] = frozenset(
    {"english", "en", "eng", "en-us", "en-gb", "en_us", "en_gb"}
)

_MAX_PASSES: int = 6
# Upper bound on fixed-point iterations per line (two are normally enough).

_MAX_CARDINAL_DIGITS: int = 21
# Longest integer the built-in speller will word (up to quintillions).

_ONES: tuple[str, ...] = (
    "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
    "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
    "sixteen", "seventeen", "eighteen", "nineteen",
)
_TENS: tuple[str, ...] = (
    "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy",
    "eighty", "ninety",
)
_SCALES: tuple[str, ...] = (
    "", "thousand", "million", "billion", "trillion", "quadrillion",
    "quintillion",
)
_ORDINAL_IRREGULAR: dict[str, str] = {
    "one": "first", "two": "second", "three": "third", "five": "fifth",
    "eight": "eighth", "nine": "ninth", "twelve": "twelfth",
}
_DIGIT_WORDS: dict[str, str] = {str(i): _ONES[i] for i in range(10)}
_DECADE_WORDS: dict[str, str] = {
    "2": "twenties", "3": "thirties", "4": "forties", "5": "fifties",
    "6": "sixties", "7": "seventies", "8": "eighties", "9": "nineties",
}

_ROMAN_VALUES: dict[str, int] = {
    "I": 1, "V": 5, "X": 10, "L": 50, "C": 100, "D": 500, "M": 1000,
}
# Label words after which a roman numeral is a plain number (cardinal).
_ROMAN_LABELS: frozenset[str] = frozenset({
    "chapter", "chapters", "chap", "ch", "part", "parts", "book", "books",
    "volume", "volumes", "vol", "vols", "act", "acts", "scene", "scenes",
    "section", "sections", "sec", "appendix", "appendices", "canto", "cantos",
    "episode", "episodes", "lesson", "lecture", "stanza", "article", "title",
    "phase", "stage", "type", "class", "grade", "mark", "tier", "category",
    "schedule", "psalm", "sonnet", "figure", "fig", "table", "plate", "unit",
    "module", "round", "tome", "interlude", "apollo", "saturn", "vatican",
    "wrestlemania", "olympiad",
})
# Labels that may legitimately end with a period ("Vol. IV").
_ROMAN_LABELS_DOTTED: frozenset[str] = frozenset(
    {"chap", "ch", "vol", "vols", "sec", "fig"}
)
# Labels after which a single capital I / V / X is trusted to be a numeral.
_ROMAN_STRONG: frozenset[str] = frozenset({
    "chapter", "chapters", "chap", "part", "parts", "book", "books",
    "volume", "volumes", "vol", "act", "acts", "scene", "scenes", "section",
    "sections", "canto", "cantos", "episode", "episodes", "lesson", "lecture",
    "stanza", "article", "title", "psalm", "sonnet", "saturn", "appendix",
    "appendices",
})
# Two-word labels; the numeral after them is always a number.
_ROMAN_LABELS_2: frozenset[str] = frozenset({"world war", "super bowl"})

_REGNAL_TITLES: frozenset[str] = frozenset({
    "King", "Queen", "Pope", "Emperor", "Empress", "Tsar", "Czar", "Tsarina",
    "Kaiser", "Sultan", "Pharaoh", "Prince", "Princess", "Duke", "Duchess",
    "Earl", "Count", "Countess", "Baron", "Archduke", "Shah", "Caliph",
    "Emir", "Khan", "Patriarch", "Antipope", "Lord", "Elector", "Margrave",
})
_REGNAL_NAMES: frozenset[str] = frozenset({
    "Henry", "Louis", "Charles", "George", "Edward", "Elizabeth", "William",
    "James", "Richard", "John Paul", "John", "Paul", "Pius", "Benedict",
    "Leo", "Gregory", "Innocent", "Clement", "Urban", "Alexander",
    "Nicholas", "Peter", "Catherine", "Mary", "Philip", "Philippe",
    "Frederick", "Friedrich", "Ferdinand", "Francis", "Franz Joseph",
    "Franz", "Napoleon", "Victor Emmanuel", "Wilhelm", "Ivan", "Constantine",
    "Ramses", "Ramesses", "Rameses", "Thutmose", "Amenhotep", "Ptolemy",
    "Seti", "Darius", "Xerxes", "Artaxerxes", "Cyrus", "Alfonso", "Carlos",
    "Juan Carlos", "Felipe", "Henri", "Leopold", "Albert", "Baldwin", "Otto",
    "Conrad", "Rudolf", "Maximilian", "Gustav", "Gustavus", "Christian",
    "Haakon", "Olaf", "Olav", "Harald", "Harold", "Carl", "Oscar", "Stephen",
    "Edmund", "Ethelred", "Robert", "David", "Kamehameha", "Selim",
    "Suleiman", "Mehmed", "Murad", "Justinian", "Theodosius", "Basil",
    "Michael", "Manuel", "Boris", "Vladimir", "Sigismund", "Casimir",
    "Augustus", "Umberto", "Pedro", "Isabella", "Anne", "Margaret",
    "Margrethe", "Rainier", "Hussein", "Abdullah", "Faisal", "Hassan",
    "Mohammed", "Rama", "Alexios", "Mithridates", "Antiochus", "Seleucus",
    "Demetrius", "Shapur", "Sixtus", "Julius", "Adrian", "Hadrian", "Martin",
    "Celestine", "Boniface", "Honorius", "Eugene", "Callixtus", "Victor",
    "Sylvester", "Anastasius", "Felix", "Lothair", "Pepin", "Clovis",
    "Dagobert", "Theodoric", "Canute", "Cnut", "Sweyn", "Duncan", "Kenneth",
    "Menelik", "Akbar", "Victoria", "Joseph", "Humbert", "Christina",
    "Eric", "Erik", "Magnus", "Valdemar", "Sancho", "Ramiro", "Afonso",
})

_MONTHS_FULL: frozenset[str] = frozenset({
    "January", "February", "March", "April", "May", "June", "July", "August",
    "September", "October", "November", "December",
})
_MONTH_ABBR: dict[str, str] = {
    "Jan": "January", "Feb": "February", "Mar": "March", "Apr": "April",
    "Jun": "June", "Jul": "July", "Aug": "August", "Sep": "September",
    "Sept": "September", "Oct": "October", "Nov": "November",
    "Dec": "December",
}
_MONTH_FULL_ALT: str = (
    "January|February|March|April|May|June|July|August|September|October"
    "|November|December"
)
_MONTH_ALT: str = (
    _MONTH_FULL_ALT + "|Sept|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Oct|Nov|Dec"
)

# symbol -> (singular, plural, sub-unit singular, sub-unit plural)
_CURRENCIES: dict[str, tuple[str, str, str | None, str | None]] = {
    "$": ("dollar", "dollars", "cent", "cents"),
    "£": ("pound", "pounds", "penny", "pence"),
    "€": ("euro", "euros", "cent", "cents"),
    "¥": ("yen", "yen", None, None),
    "₹": ("rupee", "rupees", "paisa", "paise"),
}
_CURRENCY_PREFIXES: dict[str, str] = {
    "US": "U.S.", "USD": "U.S.", "A": "Australian", "AU": "Australian",
    "C": "Canadian", "CA": "Canadian", "NZ": "New Zealand", "HK": "Hong Kong",
}
_CURRENCY_CODES: dict[str, tuple[str, str]] = {
    "USD": ("U.S. dollar", "U.S. dollars"), "EUR": ("euro", "euros"),
    "GBP": ("pound", "pounds"), "JPY": ("yen", "yen"),
    "INR": ("rupee", "rupees"),
}
_MAGNITUDE_LETTERS: dict[str, str] = {
    "k": "thousand", "K": "thousand", "m": "million", "M": "million",
    "MM": "million", "b": "billion", "B": "billion", "bn": "billion",
    "Bn": "billion",
}

# Fraction denominators that are always read as a fraction ("3/4"), and
# those that need a supporting cue ("2/5 of", mixed "1 3/10").
_FRACTION_ALWAYS: frozenset[int] = frozenset({2, 3, 4, 8, 16})
_FRACTION_CUED: frozenset[int] = frozenset({5, 6, 10, 32, 100})
_FRACTION_NAMES: dict[int, tuple[str, str]] = {
    2: ("half", "halves"), 3: ("third", "thirds"), 4: ("quarter", "quarters"),
    5: ("fifth", "fifths"), 6: ("sixth", "sixths"), 8: ("eighth", "eighths"),
    10: ("tenth", "tenths"), 16: ("sixteenth", "sixteenths"),
    32: ("thirty-second", "thirty-seconds"),
    100: ("hundredth", "hundredths"),
}
_UNICODE_FRACTIONS: dict[str, tuple[int, int]] = {
    "½": (1, 2), "⅓": (1, 3), "⅔": (2, 3), "¼": (1, 4), "¾": (3, 4),
    "⅛": (1, 8), "⅜": (3, 8), "⅝": (5, 8), "⅞": (7, 8), "⅕": (1, 5),
    "⅖": (2, 5), "⅗": (3, 5), "⅘": (4, 5), "⅙": (1, 6), "⅚": (5, 6),
}

# Unit abbreviations that are unambiguous after a number. Single letters
# ("m", "g", "s", "l") and "in" are deliberately absent.
_UNITS: dict[str, tuple[str, str]] = {
    "km": ("kilometer", "kilometers"), "cm": ("centimeter", "centimeters"),
    "mm": ("millimeter", "millimeters"), "kg": ("kilogram", "kilograms"),
    "mg": ("milligram", "milligrams"), "ml": ("milliliter", "milliliters"),
    "mL": ("milliliter", "milliliters"),
    "km/h": ("kilometer per hour", "kilometers per hour"),
    "kph": ("kilometer per hour", "kilometers per hour"),
    "mph": ("mile per hour", "miles per hour"), "ft": ("foot", "feet"),
    "lb": ("pound", "pounds"), "lbs": ("pound", "pounds"),
    "oz": ("ounce", "ounces"), "yd": ("yard", "yards"),
    "yds": ("yard", "yards"), "Hz": ("hertz", "hertz"),
    "kHz": ("kilohertz", "kilohertz"), "MHz": ("megahertz", "megahertz"),
    "GHz": ("gigahertz", "gigahertz"), "KB": ("kilobyte", "kilobytes"),
    "MB": ("megabyte", "megabytes"), "GB": ("gigabyte", "gigabytes"),
    "TB": ("terabyte", "terabytes"), "kW": ("kilowatt", "kilowatts"),
    "MW": ("megawatt", "megawatts"),
    "kWh": ("kilowatt hour", "kilowatt hours"), "hr": ("hour", "hours"),
    "hrs": ("hour", "hours"), "min": ("minute", "minutes"),
    "mins": ("minute", "minutes"), "sec": ("second", "seconds"),
    "secs": ("second", "seconds"), "yr": ("year", "years"),
    "yrs": ("year", "years"),
}

# Abbreviations expanded only directly before a capitalised name.
_TITLE_ABBREVIATIONS: dict[str, str] = {
    "Prof": "Professor", "Gen": "General", "Capt": "Captain",
    "Lt": "Lieutenant", "Sgt": "Sergeant", "Rev": "Reverend",
    "Hon": "Honorable", "Gov": "Governor", "Sen": "Senator",
    "Rep": "Representative", "Col": "Colonel", "Cmdr": "Commander",
    "Maj": "Major", "Cpl": "Corporal", "Pvt": "Private", "Adm": "Admiral",
    "Supt": "Superintendent", "Det": "Detective", "Insp": "Inspector",
    "Pres": "President", "Mt": "Mount", "Ft": "Fort", "Msgr": "Monsignor",
    "Brig": "Brigadier", "Fr": "Father",
}
# Abbreviations that follow a name; their period may also end the sentence.
_TAIL_ABBREVIATIONS: dict[str, str] = {
    "Jr": "Junior", "Sr": "Senior", "Inc": "Incorporated", "Ltd": "Limited",
    "Corp": "Corporation", "Co": "Company", "Bros": "Brothers",
    "Esq": "Esquire", "Ave": "Avenue", "Blvd": "Boulevard", "Rd": "Road",
}
_REFERENCE_ABBREVIATIONS: dict[str, str] = {
    "Fig": "Figure", "Vol": "Volume", "Ch": "Chapter", "Chap": "Chapter",
    "Sec": "Section",
}
# Capitalised words that can precede "St." without making it a street.
_NOT_STREET_WORDS: frozenset[str] = frozenset({
    "The", "In", "At", "On", "To", "From", "Of", "A", "An", "And", "But",
    "For", "Near", "By", "With", "Is", "Was", "Dear", "Old", "Poor", "Good",
    "Blessed", "Like", "As", "When", "Then", "If", "Not", "Or", "So", "That",
    "This", "Did", "Had", "Has", "Where", "Our", "His", "Her", "My", "Now",
    "Yet", "Before", "After", "Since", "Until", "While", "Because", "Though",
    "Although", "Into", "Toward", "Towards", "Through", "Visit", "See",
})

# All-caps tokens that keep their capitals inside a converted all-caps run.
# Ambiguous ones ("US", "IT", "AM", "AD", "WHO", "LA") are deliberately absent.
_KNOWN_ACRONYMS: frozenset[str] = frozenset({
    "USA", "UK", "EU", "UN", "FBI", "CIA", "NSA", "NASA", "NATO", "UFO",
    "DNA", "RNA", "TV", "CEO", "CFO", "CTO", "VIP", "ID", "OK", "AI", "PC",
    "CD", "DVD", "GPS", "ATM", "HIV", "AIDS", "IQ", "DC", "NYC", "UAE",
    "USSR", "BBC", "CNN", "IBM", "MIT", "UCLA", "NYPD", "LAPD", "SWAT",
    "POW", "AWOL", "RSVP", "ASAP", "DIY", "FAQ", "HR", "PR", "PM", "BC",
    "BCE", "KGB", "IRA", "IRS", "SOS", "RAF", "SS", "UV", "PDF", "URL",
    "USB", "CPU", "GPU", "RAM", "API", "HQ", "ER", "ICU", "MRI", "DJ", "MC",
    "MP", "VP", "PHD", "YMCA", "NFL", "NBA", "FIFA", "UNESCO", "OPEC",
    "WWII", "WWI", "GDP", "ETA", "RIP", "LED", "LCD", "SUV", "RV", "AC",
    "ESP", "PTSD", "ADHD", "OCD", "LGBT", "NGO", "EMP", "ROTC", "GI",
})
_US_FOLLOWERS: frozenset[str] = frozenset({
    "ARMY", "NAVY", "AIR", "MARINE", "MARINES", "GOVERNMENT", "MILITARY",
    "EMBASSY", "DOLLAR", "DOLLARS", "CITIZEN", "CITIZENS", "SENATE",
    "CONGRESS", "PRESIDENT", "TROOPS", "FORCES", "SOLDIERS", "BORDER",
    "COAST", "STATE", "DEPARTMENT", "SUPREME", "CONSTITUTION", "MARSHAL",
    "MARSHALS", "ATTORNEY", "CAPITOL", "TREASURY", "MINT", "POSTAL", "MAIL",
})
_CAPS_VOWELS: frozenset[str] = frozenset("AEIOUY")
# Vowel-less all-caps tokens that are words or abbreviations, not acronyms.
_CAPS_VOWELLESS_WORDS: frozenset[str] = frozenset({
    "MR", "MRS", "MS", "DR", "ST", "MT", "FT", "JR", "SR", "VS", "LTD", "HMM",
    "HM", "MM", "MMM", "SHH", "SH", "SSH", "PSST", "PST", "GRR", "BRR", "PFFT",
    "TSK", "ZZZ", "HMPH", "NTH", "CH", "TH", "ND", "RD",
})
_TITLE_SMALL_WORDS: frozenset[str] = frozenset({
    "a", "an", "the", "and", "but", "or", "nor", "for", "of", "in", "on",
    "at", "to", "by", "with", "from", "as", "into", "vs",
})

# ══════════════════════════════════════════════════════════════════════════════
# Number spelling (built in, English)
# ══════════════════════════════════════════════════════════════════════════════


def _under_thousand(n: int) -> str:
    """Words for 0..999."""
    parts: list[str] = []
    if n >= 100:
        parts.append(_ONES[n // 100] + " hundred")
        n %= 100
    if n >= 20:
        tens = _TENS[n // 10]
        parts.append(tens + ("-" + _ONES[n % 10] if n % 10 else ""))
    elif n or not parts:
        parts.append(_ONES[n])
    return " ".join(parts)


def _cardinal(n: int) -> str:
    """Words for a non-negative integer ("one thousand two hundred")."""
    if n < 1000:
        return _under_thousand(n)
    groups: list[str] = []
    scale = 0
    while n:
        n, rem = divmod(n, 1000)
        if rem:
            words = _under_thousand(rem)
            groups.append(words + " " + _SCALES[scale] if scale else words)
        scale += 1
    return " ".join(reversed(groups))


def _ordinal(n: int) -> str:
    """Words for an ordinal ("twenty-first", "one hundredth")."""
    words = _cardinal(n)
    cut = max(words.rfind(" "), words.rfind("-")) + 1
    last = words[cut:]
    if last in _ORDINAL_IRREGULAR:
        last = _ORDINAL_IRREGULAR[last]
    elif last.endswith("y"):
        last = last[:-1] + "ieth"
    else:
        last += "th"
    return words[:cut] + last


def _digits(text: str) -> str:
    """Reads the ASCII digits of ``text`` one by one."""
    return " ".join(_DIGIT_WORDS[c] for c in text if c in _DIGIT_WORDS)


def _year(n: int) -> str:
    """Words for a year ("nineteen oh five", "two thousand five")."""
    if n < 100 or n > 9999:
        return _cardinal(n)
    high, low = divmod(n, 100)
    if n < 1000:
        if low == 0:
            return _ONES[high] + " hundred"
        if low < 10:
            return _ONES[high] + " oh " + _ONES[low]
        return _ONES[high] + " " + _under_thousand(low)
    if low == 0:
        if high % 10 == 0:
            return _cardinal(n)
        return _under_thousand(high) + " hundred"
    if high % 10 == 0 and low < 10:
        return _cardinal(n)
    if low < 10:
        return _under_thousand(high) + " oh " + _ONES[low]
    return _under_thousand(high) + " " + _under_thousand(low)


def _say_number(token: str) -> str | None:
    """Words for a numeric token ("1,234.50"); None when it cannot be worded."""
    if not token.isascii():
        return None
    token = token.replace(",", "")
    whole, _, frac = token.partition(".")
    if not whole.isdigit() or len(whole) > _MAX_CARDINAL_DIGITS:
        return None
    if len(whole) > 1 and whole[0] == "0":
        return None
    words = _cardinal(int(whole))
    if frac:
        if not frac.isdigit():
            return None
        words += " point " + _digits(frac)
    return words


def _roman_value(token: str) -> int | None:
    """Value of a well-formed roman numeral, else None."""
    if not token or _ROMAN_VALID_RE.fullmatch(token) is None:
        return None
    total = 0
    highest = 0
    for ch in reversed(token):
        value = _ROMAN_VALUES[ch]
        total += -value if value < highest else value
        highest = max(highest, value)
    return total


# ══════════════════════════════════════════════════════════════════════════════
# Shared pattern fragments
# ══════════════════════════════════════════════════════════════════════════════

# Speed note: Python's ``re`` scans quickly only when a pattern *starts* by
# consuming a literal or a character class. Number patterns therefore begin
# with the digit itself and assert the left boundary right after it
# (``\d(?<![bad]\d)``) instead of opening with a look-behind, and rules that
# need a cue word look it up from the match callback.
_NUM: str = r"\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?"
_BAD_BEFORE: str = r"\w.,:/$£€¥₹-"       # what may not touch a number's left
_NB: str = r"(?<![" + _BAD_BEFORE + r"])"
_D1: str = r"\d(?<![" + _BAD_BEFORE + r"]\d)"          # first digit of a number
_NUM_START: str = _D1 + r"(?:\d{0,2}(?:,\d{3})+(?:\.\d+)?|\d*(?:\.\d+)?)"
_NA: str = r"(?![\w%°]|[.,:/-]\d)"       # nothing numeric-ish after a number
_YEAR: str = r"(?:1\d{3}|20\d{2})"
_LETTER: str = r"[^\W\d_]"

_ROMAN_VALID_RE = re.compile(
    r"(?=[IVXLCDM])M{0,3}(?:CM|CD|D?C{0,3})(?:XC|XL|L?X{0,3})(?:IX|IV|V?I{0,3})"
)
_ASCII_DIGIT_RE = re.compile(r"[0-9]")
_THREE_DIGIT_RE = re.compile(r"[0-9][0-9][0-9]")
_FOUR_DIGIT_RE = re.compile(r"[0-9][0-9][0-9][0-9]")


def _contains_any(text: str, keys: tuple[str, ...]) -> bool:
    """True when any of ``keys`` is a substring of ``text`` (a cheap rule gate)."""
    for key in keys:
        if key in text:
            return True
    return False


# ══════════════════════════════════════════════════════════════════════════════
# Language-neutral rules
# ══════════════════════════════════════════════════════════════════════════════

_INVISIBLE_RE = re.compile(r"[\u200b-\u200d\u2060\ufeff\u00ad]")

_URL_RE = re.compile(
    r"(?<![\w@.])(?:(?:https?|ftp)://(?:www\.)?|www\.)"
    r"(?P<host>[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)+)(?::\d+)?"
    r"(?P<path>[/?#][^\s<>\"'”’)\]]*)?",
    re.IGNORECASE,
)
_SIMPLE_PATH_RE = re.compile(r"(?:/[A-Za-z][A-Za-z-]{0,20}){1,2}/?")
_EMAIL_RE = re.compile(
    r"(?<![\w.+-])(?P<local>[A-Za-z0-9][A-Za-z0-9._+-]*)@"
    r"(?P<host>(?:[A-Za-z0-9-]+\.)+[A-Za-z]{2,})(?![\w-])"
)
_BARE_DOMAIN_RE = re.compile(
    r"(?<![\w@./-])(?P<host>(?:[A-Za-z0-9-]{2,}\.)+"
    r"(?:com|org|net|edu|gov|io|co\.uk|org\.uk|ac\.uk))"
    r"(?P<path>/[^\s<>\"'”’)\]]*)?(?![\w-]|\.[A-Za-z])"
)
_BARE_DOMAIN_KEYS: tuple[str, ...] = (
    ".com", ".org", ".net", ".edu", ".gov", ".io", ".uk",
)

_FOOTNOTE_RE = re.compile(
    r"(?<=\S)\[\d{1,3}(?:\s?[,\-–]\s?\d{1,3})*\]"      # word[1], word[1, 2]
    r"|(?P<spaced>(?<=\S) \[\d{1,3}\])"                 # word [12]
    r"|(?<=\S)\s?\[(?:note|fn\.?|n\.)\s?\d{1,3}\]"      # word[note 3]
    r"|(?<=[^\s\[(])\[[a-z]\]"                          # word[a]
    r"|(?<=\S)\s?\[(?:citation needed|\*+|†|‡)\]"
)
# "as shown in [3]": after these words a spaced marker is a citation that
# the sentence needs, not a footnote - it is left alone.
_CITATION_LEADS: frozenset[str] = frozenset({
    "in", "see", "by", "from", "of", "and", "to", "ref", "ref.", "refs",
    "refs.", "reference", "references", "cf.", "also", "or", "with", "per",
    "e.g.", "i.e.", "e.g.,", "i.e.,", "and,", "see,", "en", "de", "und", "et",
    "ver", "voir", "siehe", "vedi", "ver,", "em", "y", "e",
})

_SUPERSCRIPTS: str = "⁰¹²³⁴⁵⁶⁷⁸⁹"
_SUPERSCRIPT_NOTE_RE = re.compile(
    r"(?:(?<=" + _LETTER + _LETTER + r")|(?<=[.,;:!?\"'”’)\]]))[" + _SUPERSCRIPTS + r"]+"
)
_SUPERSCRIPT_GATE_RE = re.compile("[" + _SUPERSCRIPTS + "]")

# Bullets, daggers, pilcrow, trademark signs, arrows, box drawing, geometric
# shapes, dingbats, emoji, variation selectors - and the ASCII pipe.
_DECOR_CLASS: str = (
    "|\u2022\u2023\u2043\u204c\u204d\u2219\u25e6\u00b6\u2020\u2021\u2122\u00ae"
    "\u2190-\u21ff\u2300-\u23ff\u2500-\u259f\u25a0-\u25ff\u2600-\u26ff"
    "\u2700-\u27bf\u27f0-\u27ff\u2900-\u297f\u2b00-\u2bff"
    "\U0001f000-\U0001faff\ufe00-\ufe0f\u20e3"
)
_DECOR_RE = re.compile(r"[ \t]*(?:[" + _DECOR_CLASS + r"][ \t]*)+")
_DECOR_GATE_RE = re.compile("[" + _DECOR_CLASS + "·]")
_MIDDLE_DOT_RE = re.compile(r"(?:(?<=\s)|^)·(?=\s|$)")
_CLOSING_PUNCT: str = ".,;:!?)]}\"'”’"

# Substring gate for the repeated-punctuation rules.
_PUNCT_KEYS: tuple[str, ...] = (
    "!!", "??", "!?", "?!", "..", ". .", "….", ",,", "===", "----",
    "！！", "？？", "！？", "？！",
)
_RULE_RUN_RE = re.compile(r"[ \t]*(?:={3,}|-{4,})[ \t]*")
_REPEAT_BANG_RE = re.compile(r"[!?]{2,}|[！？]{2,}")
_ELLIPSIS_SPACED_RE = re.compile(r"(?:(?<=\w) )?\.(?:[ \t]\.){2,}")
_ELLIPSIS_LONG_RE = re.compile(r"\.{4,}")
_ELLIPSIS_MIXED_RE = re.compile(r"…\.+")
_REPEAT_COMMA_RE = re.compile(r",{2,}")

_EMOTICON_RE = re.compile(r"\^[_\-.o]?\^")
_ASTERISK_RE = re.compile(r"\*+")
_BLANK_RE = re.compile(r"(?<![^\W_])_{3,}(?![^\W_])")
_UNDERSCORE_TOKEN_RE = re.compile(r"\S*_\S*")
_UNDERSCORE_JOIN_RE = re.compile(r"(?<=[^\W_])_+(?=[^\W_])")
_TILDE_RE = re.compile(r"~+(?![\d~])")
_CARET_RE = re.compile(r"(?<![^\W_])\^+|\^+(?![^\W_])")

_TIDY_SPACES_RE = re.compile(r"[ \t]{2,}")
_TIDY_SPACE_PUNCT_RE = re.compile(r" +([,.])(?=\s|$|[\"'”’)\]])")
_TIDY_DUP_COMMA_RE = re.compile(r",(?:\s*,)+")
_TIDY_EMPTY_BRACKETS_RE = re.compile(r"\(\s*\)|\[\s*\]")
_ALNUM_RE = re.compile(r"[^\W_]")


def _bang_sub(match: re.Match) -> str:
    """"!!!" -> "!", "?!?!" -> "?!" (first two distinct marks, in order)."""
    run = match.group(0)
    first = run[0]
    for ch in run[1:]:
        if ch != first:
            return first + ch
    return first


def _fix_repeated_punctuation(text: str, english: bool) -> str:
    text = _REPEAT_BANG_RE.sub(_bang_sub, text)
    text = _ELLIPSIS_SPACED_RE.sub("...", text)
    text = _ELLIPSIS_LONG_RE.sub("...", text)
    text = _ELLIPSIS_MIXED_RE.sub("…", text)
    text = _RULE_RUN_RE.sub(" ", text)
    return _REPEAT_COMMA_RE.sub(",", text)


def _url_sub(match: re.Match, english: bool) -> str:
    host = match.group("host")
    path = match.group("path") or ""
    trailing = ""
    while path and path[-1] in ".,;:!?":
        trailing = path[-1] + trailing
        path = path[:-1]
    if not english:
        return host + trailing
    spoken = host.replace(".", " dot ")
    if path and _SIMPLE_PATH_RE.fullmatch(path):
        spoken += "".join(" slash " + seg for seg in path.strip("/").split("/"))
    return spoken + trailing


def _fix_urls(text: str, english: bool) -> str:
    return _URL_RE.sub(lambda m: _url_sub(m, english), text)


def _footnote_sub(match: re.Match) -> str:
    if match.group("spaced"):
        before = match.string[max(0, match.start() - 12):match.start()]
        words = before.split()
        if words and words[-1].lower() in _CITATION_LEADS:
            return match.group(0)
    return ""


def _fix_footnotes(text: str, english: bool) -> str:
    return _FOOTNOTE_RE.sub(_footnote_sub, text)


def _fix_superscript_notes(text: str, english: bool) -> str:
    return _SUPERSCRIPT_NOTE_RE.sub("", text)


def _decor_sub(match: re.Match, english: bool) -> str:
    text = match.string
    start, end = match.span()
    if start == 0 or end == len(text):
        return ""
    run = match.group(0)
    before, after = text[start - 1], text[end]
    spaced_before, spaced_after = run[0] in " \t", run[-1] in " \t"
    if after in _CLOSING_PUNCT:
        return ""
    if spaced_before and spaced_after:
        if english and before.isalnum() and after.isalnum():
            return ", "
        return " "
    if spaced_before or spaced_after:
        return " "
    if before.isascii() and before.isalnum() and after.isascii() and after.isalnum():
        return " "
    return ""


def _fix_decorations(text: str, english: bool) -> str:
    text = _DECOR_RE.sub(lambda m: _decor_sub(m, english), text)
    return _MIDDLE_DOT_RE.sub("", text)


def _asterisk_sub(match: re.Match, english: bool) -> str:
    text = match.string
    start, end = match.span()
    before = text[start - 1] if start else ""
    after = text[end] if end < len(text) else ""
    if before.isalpha() and after.isalpha():
        return match.group(0)                        # censored: f*ck, f**k
    if before.isdigit() and after.isdigit():
        return match.group(0)                        # 5*3
    if (
        end - start == 1 and before == " " and after == " "
        and start >= 2 and text[start - 2].isdigit()
        and end + 1 < len(text) and text[end + 1].isdigit()
    ):
        return "times" if english else match.group(0)
    return ""


def _underscore_sub(match: re.Match) -> str:
    token = match.group(0)
    if "@" in token or "://" in token:
        return token
    return _UNDERSCORE_JOIN_RE.sub(" ", token).replace("_", "")


def _fix_stray_symbols(text: str, english: bool) -> str:
    if "^" in text:
        text = _EMOTICON_RE.sub("", text)
        text = _CARET_RE.sub("", text)
    if "*" in text:
        text = _ASTERISK_RE.sub(lambda m: _asterisk_sub(m, english), text)
    if "_" in text:
        text = _BLANK_RE.sub("...", text)            # a blank: "Sword of ______"
        text = _UNDERSCORE_TOKEN_RE.sub(_underscore_sub, text)
    if "~" in text:
        text = _TILDE_RE.sub("", text)
    return text


def _tidy(text: str) -> str:
    """Repairs spacing left behind by removals (only run on changed lines)."""
    text = _TIDY_EMPTY_BRACKETS_RE.sub("", text)
    text = _TIDY_SPACES_RE.sub(" ", text)
    text = _TIDY_SPACE_PUNCT_RE.sub(r"\1", text)
    text = _TIDY_DUP_COMMA_RE.sub(",", text)
    return text.strip()


# ══════════════════════════════════════════════════════════════════════════════
# English: e-mail addresses and bare domains
# ══════════════════════════════════════════════════════════════════════════════


def _email_sub(match: re.Match) -> str:
    local = (
        match.group("local").replace(".", " dot ").replace("_", " underscore ")
        .replace("-", " dash ").replace("+", " plus ")
    )
    return local + " at " + match.group("host").replace(".", " dot ")


def _bare_domain_sub(match: re.Match) -> str:
    return _url_sub(match, True)


def _fix_addresses(text: str) -> str:
    if "@" in text:
        text = _EMAIL_RE.sub(_email_sub, text)
    if _contains_any(text, _BARE_DOMAIN_KEYS):
        text = _BARE_DOMAIN_RE.sub(_bare_domain_sub, text)
    return text


# ══════════════════════════════════════════════════════════════════════════════
# English: abbreviations
# ══════════════════════════════════════════════════════════════════════════════

_ABBR_RE = re.compile(
    r"(?<![\w.&])(?:"
    r"(?P<title>" + "|".join(_TITLE_ABBREVIATIONS) + r")\.(?=\s+[A-Z])"
    r"|(?P<st>St)\."
    r"|(?P<tail>" + "|".join(_TAIL_ABBREVIATIONS) + r")\."
    r"|(?P<etc>[Ee]tc)(?:\.|(?=[,;)]))"
    r"|(?P<eg>[Ee]\.\s?g|[Ii]\.\s?e)\.(?P<egcomma>,)?"
    r"|(?P<vs>[Vv]s)\.?(?![\w.])"
    r"|(?P<v>v)\.(?=\s+[A-Z])"
    r"|(?P<approx>[Aa]pprox)\."
    r"|(?P<aka>a\.k\.a)\.?(?![\w.])"
    r"|(?P<cf>cf)\.(?=\s)"
    r"|(?P<viz>viz)\.(?=\s)"
    r"|(?P<etal>et al)\."
    r"|(?P<ref>" + "|".join(_REFERENCE_ABBREVIATIONS) + r")\.(?=\s?(?:\d|[IVXLC]+\b))"
    r"|(?P<with>w/o|w/)(?=\s?[A-Za-z])"
    r")"
)
# Abbreviations that only mean something before a number ("p. 47", "c. 1900").
_ABBR_DIGIT_RE = re.compile(
    r"(?P<pp>p(?<![\w.&]p)p?)\.(?=\s?\d)"
    r"|(?P<circa>c(?<![\w.&]c)a?)\.(?=\s?[12]?\d{3}\b)"
)
# Token gate: the alternation above only runs on lines holding one of these
# whitespace-delimited tokens (punctuation around them is blanked first).
_ABBR_TOKENS: frozenset[str] = frozenset(
    [key + "." for key in _TITLE_ABBREVIATIONS]
    + [key + "." for key in _TAIL_ABBREVIATIONS]
    + [key + "." for key in _REFERENCE_ABBREVIATIONS]
    + [
        "St.", "etc.", "Etc.", "etc", "Etc", "e.g.", "E.g.", "i.e.", "I.e.", "e.g", "i.e",
        "vs", "vs.", "Vs", "Vs.", "v.", "approx.", "Approx.", "a.k.a.",
        "a.k.a", "cf.", "viz.", "al.",
    ]
)
_ABBR_PUNCT_TABLE: dict[int, str] = {ord(c): " " for c in "()[]\"',;:!?-“”‘’"}
_AFTER_END_RE = re.compile(r"[\"'’”)\]]*\s*$")
_AFTER_AMBIGUOUS_RE = re.compile(r"[\"'’”)\]]*\s+[\"'‘“(\[]*[A-Z]")
_NAME_BEFORE_RE = re.compile(r"(?:[A-Z][\w'’-]*,?|&|and)\s$")
_CAP_WORD_BEFORE_RE = re.compile(r"([A-Z][\w'’-]*)\s$")
_ORDINAL_BEFORE_RE = re.compile(
    r"(?:\d(?:st|nd|rd|th)|\b(?:[a-z]+-)?(?:first|second|third|fourth|fifth"
    r"|sixth|seventh|eighth|ninth|tenth|eleventh|twelfth|[a-z]+teenth"
    r"|[a-z]+tieth|hundredth))\s$"
)
_SAINT_AFTER_RE = re.compile(r"\s+[A-Z][a-z]")


def _tail_period(after: str) -> str | None:
    """Decides what becomes of an expanded abbreviation's period.

    Returns "." when the abbreviation closes the line, "" when the sentence
    plainly continues, and None when a capitalised word follows (the period
    may or may not end the sentence).
    """
    if _AFTER_END_RE.match(after):
        return "."
    if _AFTER_AMBIGUOUS_RE.match(after):
        return None
    return ""


def _abbr_sub(match: re.Match) -> str:
    kind = match.lastgroup
    whole = match.group(0)
    text = match.string
    after = text[match.end():match.end() + 16]

    if kind == "title":
        return _TITLE_ABBREVIATIONS[match.group("title")]
    if kind == "st":
        before = text[max(0, match.start() - 32):match.start()]
        cap = _CAP_WORD_BEFORE_RE.search(before)
        street = bool(_ORDINAL_BEFORE_RE.search(before)) or (
            cap is not None and cap.group(1) not in _NOT_STREET_WORDS
        )
        saint = _SAINT_AFTER_RE.match(after) is not None
        if saint and not street:
            return "Saint"
        if street and not saint:
            period = _tail_period(after)
            return whole if period is None else "Street" + period
        return whole
    if kind == "tail":
        before = text[max(0, match.start() - 32):match.start()]
        if not _NAME_BEFORE_RE.search(before):
            return whole
        period = _tail_period(after)
        if period is None:
            return whole
        return _TAIL_ABBREVIATIONS[match.group("tail")] + period
    if kind == "etc":
        period = _tail_period(after)
        word = "Et cetera" if match.group("etc")[0] == "E" else "et cetera"
        return word + ("." if period is None else period)
    if kind in ("eg", "egcomma"):
        abbr = match.group("eg")
        words = "for example" if abbr[0] in "eE" else "that is"
        if abbr[0].isupper():
            words = words.capitalize()
        return words + ("" if after[:1] in ",;:)" and not match.group("egcomma") else ",")
    if kind == "vs":
        return "Versus" if match.group("vs")[0] == "V" else "versus"
    if kind == "v":
        before = text[max(0, match.start() - 32):match.start()]
        return "versus" if _CAP_WORD_BEFORE_RE.search(before) else whole
    if kind == "approx":
        period = _tail_period(after)
        word = "Approximately" if match.group("approx")[0] == "A" else "approximately"
        return word + ("." if period == "." else "")
    if kind == "aka":
        return "also known as"
    if kind == "cf":
        return "compare"
    if kind == "viz":
        return "namely"
    if kind == "etal":
        period = _tail_period(after)
        return "and others" + ("." if period is None else period)
    if kind == "ref":
        return _REFERENCE_ABBREVIATIONS[match.group("ref")]
    if kind == "with":
        word = "without" if match.group("with") == "w/o" else "with"
        return word if after[:1] == " " else word + " "
    return whole


def _abbr_digit_sub(match: re.Match) -> str:
    if match.lastgroup == "circa":
        return "circa"
    return "pages" if match.group("pp") == "pp" else "page"


def _has_abbreviation(text: str) -> bool:
    """Cheap test for whether ``_ABBR_RE`` could match anywhere in ``text``."""
    if ("." in text or "etc" in text) and not _ABBR_TOKENS.isdisjoint(
        text.translate(_ABBR_PUNCT_TABLE).split()
    ):
        return True
    return "vs" in text or "Vs" in text or "w/" in text or "e. g" in text or "i. e" in text


def _fix_abbreviations(text: str) -> str:
    return _ABBR_RE.sub(_abbr_sub, text)


def _fix_number_abbreviations(text: str) -> str:
    if "p." in text or "c." in text or "ca." in text:
        text = _ABBR_DIGIT_RE.sub(_abbr_digit_sub, text)
    return text


# ══════════════════════════════════════════════════════════════════════════════
# English: roman numerals
# ══════════════════════════════════════════════════════════════════════════════

# A numeral token that follows whitespace; the word before it decides its fate.
_ROMAN_TOKEN_RE = re.compile(r"[IVXLC](?<=\s[IVXLC])[IVXLC]*(?![\w]|-\w)")
_ROMAN_ONLY_RE = re.compile(r"([IVXLC]+)([.:]?)")
_ROMAN_HEADING_MAX_LEN: int = 16
_ROMAN_I_FOLLOW_RE = re.compile(
    r"\s*$|[.,:;!?)\]\"”]|\s+[-–—:]|\s+(?:of|and|to|through|or|&)\b"
)
_REGNAL_I_FOLLOW_RE = re.compile(
    r"\s*$|[.,:;!?)\]\"”]|['’]s\b"
    r"|\s+(?:of|and|was|were|is|who|in|to|as|at|by|on|or|died|reigned|ruled"
    r"|became|succeeded|from|with|for)\b"
)
_SINGLE_NUMERAL_BLOCK_RE = re.compile(r"\.?\s+[A-Z]")
# "the part I played": a determiner before the label means "I" is the pronoun.
_LABEL_DETERMINERS: frozenset[str] = frozenset({
    "the", "a", "an", "this", "that", "my", "his", "her", "your", "our",
    "their", "every", "each", "which", "what", "whatever", "no", "only",
    "first", "last", "best", "same", "one",
})
_ROMAN_LIST_CONNECTIVES: frozenset[str] = frozenset(
    {"and", "to", "through", "&", "or"}
)
_ROMAN_LETTERS: frozenset[str] = frozenset("IVXLC")
_OPENING_PUNCT: str = "\"'“‘([{"
_ROMAN_LIST_BEFORE_RE = re.compile(
    r"\b(?P<label>(?i:chapters|parts|books|volumes|acts|scenes|sections"
    r"|appendices|cantos|episodes))\s+"
    r"(?:[IVXLC]+\s*(?:,\s*and\b|,|\band\b|\bto\b|\bthrough\b|&|\bor\b)\s*)+$"
)
_ROMAN_STANDALONE_UNSAFE: frozenset[str] = frozenset({"XX", "XXX"})


def _roman_after_label(
    token: str, label: str, words: list[str], after: str
) -> str | None:
    """Cardinal words for a numeral that follows a label word, or None."""
    if not (label.islower() or label.isupper() or label.istitle()):
        return None
    value = _roman_value(token)
    if value is None:
        return None
    key = label.lower()
    if len(token) == 1:
        if key in _ROMAN_LABELS_2:
            if token == "I" and label.islower():
                return None                           # "the world war I mean"
        elif token not in "IVX" or label.islower() or key not in _ROMAN_STRONG:
            return None
        elif token == "I":
            if not _ROMAN_I_FOLLOW_RE.match(after):
                return None
            if len(words) >= 2 and words[-2].lower() in _LABEL_DETERMINERS:
                return None
        elif key in ("appendix", "appendices"):
            return None
    spoken = _cardinal(value)
    return spoken.upper() if label.isupper() else spoken


def _is_name_word(word: str) -> bool:
    """True for a capitalised ASCII word such as "Henry" or "Paul"."""
    return len(word) > 1 and word.isascii() and word.isalpha() and word.istitle()


def _roman_regnal(token: str, context: list[str], after: str) -> str | None:
    """"the fourteenth" for a numeral that follows a regnal name, or None."""
    if not set(token) <= {"I", "V", "X"}:
        return None
    value = _roman_value(token)
    if value is None:
        return None
    words: list[str] = []
    for word in reversed(context[-4:]):
        if not _is_name_word(word):
            break
        words.insert(0, word)
    if not words:
        return None
    last = words[-1]
    if last in _REGNAL_TITLES:
        return None
    titled = any(w in _REGNAL_TITLES for w in words[:-1])
    named = last in _REGNAL_NAMES or " ".join(words[-2:]) in _REGNAL_NAMES
    if not (titled or named):
        return None
    if len(token) == 1:
        if token == "I":
            if not titled or not _REGNAL_I_FOLLOW_RE.match(after):
                return None
        elif not titled and _SINGLE_NUMERAL_BLOCK_RE.match(after):
            return None
    return "the " + _ordinal(value)


def _roman_list_item(token: str, before: str) -> str | None:
    """Cardinal for a later numeral of "Chapters IV, V and VI", or None."""
    found = _ROMAN_LIST_BEFORE_RE.search(before)
    if found is None or (len(token) == 1 and token not in "IVX"):
        return None
    value = _roman_value(token)
    if value is None:
        return None
    spoken = _cardinal(value)
    return spoken.upper() if found.group("label").isupper() else spoken


def _roman_token_sub(match: re.Match) -> str:
    token = match.group(0)
    text = match.string
    start = match.start()
    if len(token) == 1:
        # Fast path for the pronoun "I": a single-letter numeral needs a
        # capitalised word (label or regnal name) or a list connective before it.
        cut = text.rfind(" ", 0, start - 1) + 1
        previous = text[cut:start - 1].lstrip(_OPENING_PUNCT)
        if previous and not previous[0].isupper():
            if previous not in _ROMAN_LIST_CONNECTIVES:
                return token
    before = text[max(0, start - 64):start]
    words = [word.lstrip(_OPENING_PUNCT) for word in before.split()]
    if not words or not words[-1]:
        return token
    after = text[match.end():match.end() + 24]
    last = words[-1]
    dot = last.endswith(".")
    word1 = last[:-1] if dot else last
    if word1.isascii() and word1.isalpha():
        key = word1.lower()
        label: str | None = None
        if (
            not dot and len(words) >= 2
            and (words[-2].lower() + " " + key) in _ROMAN_LABELS_2
        ):
            label = words[-2] + " " + word1
        elif key in _ROMAN_LABELS and (not dot or key in _ROMAN_LABELS_DOTTED):
            label = word1
        if label is not None:
            return _roman_after_label(token, label, words, after) or token
        if not dot and word1[0].isupper():
            regnal = _roman_regnal(token, words, after)
            if regnal is not None:
                return regnal
    if last in _ROMAN_LIST_CONNECTIVES or last.endswith(","):
        # Only "<numeral>, <numeral>" / "<numeral> and <numeral>" can be a list.
        earlier = last[:-1] if last.endswith(",") else (words[-2] if len(words) > 1 else "")
        if earlier.rstrip(",") and set(earlier.rstrip(",")) <= _ROMAN_LETTERS:
            return _roman_list_item(token, before) or token
    return token


def _fix_roman_numerals(text: str) -> str:
    return _ROMAN_TOKEN_RE.sub(_roman_token_sub, text)


def _standalone_roman(core: str, roman_lines: frozenset[int]) -> str | None:
    """Words for a heading line that is only a roman numeral, or None."""
    match = _ROMAN_ONLY_RE.fullmatch(core)
    if match is None:
        return None
    token = match.group(1)
    value = _roman_value(token)
    if value is None:
        return None
    safe = (
        len(token) >= 2 and set(token) <= {"I", "V", "X"}
        and token not in _ROMAN_STANDALONE_UNSAFE
    )
    # Ambiguous numerals ("I", "X", "C", "XXX") need a neighbour in the
    # sequence elsewhere in the text to count as section headings.
    if not safe and (value - 1) not in roman_lines and (value + 1) not in roman_lines:
        return None
    return _cardinal(value).capitalize() + match.group(2)


# ══════════════════════════════════════════════════════════════════════════════
# English: numbers, dates, times, money, units
# ══════════════════════════════════════════════════════════════════════════════

_MONTH_KEYS: tuple[str, ...] = (
    "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct",
    "Nov", "Dec",
)
_DATE_MDY_RE = re.compile(
    r"(?<![\w.])(?P<mon>" + _MONTH_ALT + r")(?P<dot>\.)?\s+(?P<day>\d{1,2})"
    r"(?P<suf>st|nd|rd|th)?(?:(?P<sep>,?\s+)(?P<year>[12]\d{3}))?"
    r"(?![\w%°]|[.,:/-]\d|\s?[ap]\.?m\b)"
)
_DATE_DMY_RE = re.compile(
    r"(?P<day>" + _D1 + r"\d?)(?P<suf>st|nd|rd|th)?(?P<of>\s+of)?\s+"
    r"(?P<mon>" + _MONTH_ALT + r")(?![A-Za-z])(?P<dot>\.)?"
    r"(?:(?P<sep>,?\s+)(?P<year>[12]\d{3})(?![\w%°]|[.,:/-]\d))?"
)
_DATE_MY_RE = re.compile(
    r"(?<![\w.])(?P<mon>" + _MONTH_ALT + r")(?P<dot>\.)?(?P<sep>,?\s+)"
    r"(?P<year>" + _YEAR + r")(?![\w%°'’]|[.,:/-]\d)"
)
_THE_BEFORE_RE = re.compile(r"\b[Tt]he\s$")

_MERIDIEM: str = r"[ap]\.m\.?|[AP]\.M\.?|[ap]m|[AP]M"
_MERIDIEM_GATE_RE = re.compile(r"\d\s?[apAP]\.?[mM]")
_TIME_MERIDIEM_RE = re.compile(
    r"(?P<h>" + _D1 + r"\d?)[:.](?P<m>\d{2})(?P<sp>\s?)"
    r"(?P<mer>" + _MERIDIEM + r")(?![A-Za-z])"
)
_TIME_RE = re.compile(
    r"(?P<h>" + _D1 + r"\d?):(?P<m>\d{2})(?![\w:%°]|[.,/-]\d)"
)
_HOUR_MERIDIEM_RE = re.compile(
    r"(?P<h>" + _D1 + r"\d?)(?P<sp>\s?)(?P<mer>" + _MERIDIEM + r")(?![A-Za-z])"
)

_MAGNITUDE_WORDS: str = r"hundred|thousand|million|billion|trillion"
_CURRENCY_GATE_RE = re.compile(r"[$£€¥₹]")
_CURRENCY_RE = re.compile(
    r"(?<![\w$£€¥₹.,/:-])(?P<pre>USD|US|AU|A|CA|C|NZ|HK)?(?P<sym>[$£€¥₹])\s?"
    r"(?P<num>" + _NUM + r")"
    r"(?:\s?[-–]\s?(?P=sym)?(?P<num2>" + _NUM + r"))?"
    r"(?:\s(?P<magw>" + _MAGNITUDE_WORDS + r")|(?P<magl>MM|[bB]n|[kKmMbB]))?"
    r"(?![\w]|[.,:/-]\d)"
)
_CURRENCY_CODE_RE = re.compile(
    r"(?P<num>" + _NUM_START + r")(?:\s(?P<magw>" + _MAGNITUDE_WORDS + r"))?"
    r"\s?(?P<code>USD|EUR|GBP|JPY|INR)(?![\w])"
)
_CURRENCY_CODE_PRE_RE = re.compile(
    r"(?<![\w$])(?P<code>USD|EUR|GBP|JPY|INR)\s?(?P<num>" + _NUM + r")"
    r"(?:\s(?P<magw>" + _MAGNITUDE_WORDS + r")\b)?" + _NA
)
_CENT_SIGN_RE = re.compile(r"(?P<num>" + _D1 + r"\d*)\s?¢")
_ARTICLE_BEFORE_RE = re.compile(r"\b(?:[Aa]n?|[Aa]nother)\s$")
_LOWER_WORD_AFTER_RE = re.compile(r"\s[a-z]")
# Nouns an amount of money commonly modifies ("a $5 bill", "two $20 notes").
_MONEY_NOUN_AFTER_RE = re.compile(
    r"\s(?:bill|note|coin|check|cheque|fine|fee|tip|bet|ticket|prize|reward"
    r"|bounty|bonus|gift|loan|deposit|budget|contract|deal|lawsuit|settlement"
    r"|investment|jackpot|mortgage|grant|donation|payment|raise|ransom|debt"
    r"|question|price|salary|profit|loss)s?\b"
)

_PERCENT_RE = re.compile(
    r"(?P<a>" + _NUM_START + r")(?:\s?%?\s?(?:[-–]|to)\s?(?P<b>" + _NUM + r"))?"
    r"\s?%(?![\w%])"
)

# A sign glued to a number. An ASCII hyphen opening a line may be a list
# bullet, so it must follow whitespace or an opening bracket/quote.
_TRAILING_PLUS_RE = re.compile(r"(?<=\d)\+(?![\w+=(])")
_SIGN_RE = re.compile(
    r"-(?<=[\s(\[\"“]-)(?<!\d\s-)(?=\d)"
    r"|-(?<![^\s(\[\"“]-)(?=\d[\d.,]*\s?[°%℃℉])"
    r"|−(?<![\w.,:/)\]%]−)(?=\d)"
    r"|\+(?<![\w+.,)\]]\+)(?=\d)"
)
_MINUS_NUMBER_RE = re.compile(
    r"(?P<n>\d(?<=\bminus \d)(?<!\d minus \d)"
    r"(?:\d{0,2}(?:,\d{3})+(?:\.\d+)?|\d*(?:\.\d+)?))" + _NA
)

_ERA_PRE_RE = re.compile(
    r"(?<![\w.])(?P<era>A\.D\.|AD)(?P<sp>\s?)(?P<year>\d{1,4})" + _NA
)
_ERA_POST_RE = re.compile(
    r"(?P<year>" + _NUM_START + r")(?P<sp>\s?)"
    r"(?P<era>B\.C\.E\.|BCE|B\.C\.|BC|A\.D\.|AD|C\.E\.|CE)(?![\w])"
)
_ERA_KEYS: tuple[str, ...] = ("BC", "B.C", "AD", "A.D", "CE", "C.E")
_CURRENCY_CODE_KEYS: tuple[str, ...] = tuple(_CURRENCY_CODES)
_DECADE_RE = re.compile(
    r"(?P<y>[12](?<![\w$£€¥₹.,/:][12])(?<!\d-[12])\d\d0)['’]?s(?![\w])"
)
_DECADE_APOS_RE = re.compile(
    r"['’](?<![" + _BAD_BEFORE + r"]['’])(?P<dec>[2-9])0['’]?s(?![\w])"
)
_DECADE_BARE_RE = re.compile(r"(?P<dec>[2-9])(?<=[\s-][2-9])0s(?![\w])")
_DECADE_CUE_BEFORE_RE = re.compile(
    r"\b(?:[Tt]he|[Hh]is|[Hh]er|[Tt]heir|[Mm]y|[Yy]our|[Oo]ur|[Ee]arly|[Ll]ate"
    r"|[Mm]id|or|and|to)[\s-]$"
)
_YEAR_START: str = (
    r"[12](?<![" + _BAD_BEFORE + r"][12])(?:(?<=1)\d{3}|(?<=2)0\d{2})"
)
_YEAR_RANGE_RE = re.compile(
    r"(?P<y1>" + _YEAR_START + r")\s?[-–]\s?(?P<y2>" + _YEAR + r"|\d{2})"
    r"(?![\w%°]|[.,:/-]\d|\s?(?:BCE?|B\.C|AD|A\.D|CE)\b)"
)
_PAGES_BEFORE_RE = re.compile(r"(?:pp\.|[Pp]ages|[Ll]ines|[Nn]os\.)\s$")
# A year (optionally "X and Y" / "X to Y"); the callback requires a year cue
# such as "in", "since" or "the summer of" before it.
_YEAR_CONTEXT_RE = re.compile(
    r"(?P<year>[12](?<![\w.,:/$£€¥₹][12])(?:(?<=1)\d{3}|(?<=2)0\d{2}))"
    r"(?:(?P<mid>\s+(?:and|or|to|through|until|till)\s+)(?P<year2>" + _YEAR + r"))?"
    r"(?![\w%°'’]|[.,:/-]\d)"
)
_YEAR_CUE_BEFORE_RE = re.compile(
    r"(?<![\w-])(?:"
    r"(?P<span>[Ff]rom|[Bb]etween)\s+"
    r"|(?:[Ii]n|[Ss]ince|[Uu]ntil|[Tt]ill|[Dd]uring|[Cc]irca|[Bb]efore|[Aa]fter)"
    r"(?:\s+(?:the\s+)?(?:year|early|late|mid|spring|summer|autumn|fall|winter)"
    r"(?:\s+of)?)?(?:\s+|-)"
    r"|(?:[Tt]he\s+)?[Yy]ear\s+(?:of\s+)?"
    r"|(?:[Ee]arly|[Ll]ate|[Mm]id)(?:\s+|-)"
    r"|(?:[Ss]pring|[Ss]ummer|[Aa]utumn|[Ff]all|[Ww]inter|[Cc]lass|"
    + _MONTH_FULL_ALT + r")\s+of\s+"
    r")$"
)
# Last word a year cue can end with (quick reject before the regex above).
_YEAR_CUE_LAST_WORDS: frozenset[str] = frozenset({
    "in", "since", "until", "till", "during", "circa", "before", "after",
    "year", "early", "late", "mid", "of", "from", "between",
})
_YEAR_FOLLOW_RE = re.compile(
    r"\s*$|[.,;:!?)\]\"”…]|\s+[-–—(\[\"“]|\s+[A-Z][a-z]"
    r"|\s+(?:and|or|but|when|while|the|a|an|he|she|it|they|we|I|you|his|her"
    r"|their|was|were|is|had|has|at|on|as|with|by|for|to|that|this|there"
    r"|after|before|until|because|so|then|in|if|than|through|under|during"
    r"|did|would|could|will|saw|came|brought|marked)\b"
)

_CUE_RANGE_RE = re.compile(
    r"(?P<a>" + _D1 + r"\d{0,3})\s?[-–]\s?(?P<b>\d{1,4})" + _NA
)
_RANGE_CUE_BEFORE_RE = re.compile(
    r"\b(?:[Pp]ages|[Cc]hapters|[Vv]erses|[Ll]ines|[Aa]ges|[Aa]ged|[Ss]ections"
    r"|[Vv]olumes|[Rr]ooms|[Nn]umbers)\s+$"
)

_ORDINAL_GATE_RE = re.compile(r"\d(?:st|nd|rd|th|ST|ND|RD|TH)")
_ORDINAL_RE = re.compile(
    r"(?P<n>\d(?<![\w.,/:$£€¥₹]\d)(?<!\d-\d)(?:\d{0,2}(?:,\d{3})+|\d*))"
    r"(?P<suf>st|nd|rd|th|ST|ND|RD|TH)(?![\w])"
)

_FRACTION_END: str = r"(?![\w/⁄]|[.,:]\d|-\d|\stime\b)"
_FRACTION_MIXED_RE = re.compile(
    r"(?P<w>" + _D1 + r"\d{0,2})[\s-](?P<n>\d{1,2})[/⁄](?P<d>\d{1,3})" + _FRACTION_END
)
_FRACTION_RE = re.compile(
    r"(?P<n>" + _D1 + r"\d?)[/⁄](?P<d>\d{1,3})" + _FRACTION_END
)
_FRACTION_CUE_AFTER_RE = re.compile(
    r"\s+of\b|[\s-](?:inch|inches|cup|cups|mile|miles|teaspoons?|tablespoons?"
    r"|pounds?|ounces?|hours?|acres?|pints?|gallons?)\b"
)
# Before "3/4" these words make it a date or a counter, not a fraction.
_DATE_PREPOSITION_BEFORE_RE = re.compile(
    r"\b(?:[Oo]n|[Dd]ated|[Ss]ince|[Uu]ntil|[Bb]y|[Ff]rom|[Oo]f|[Pp]ages?"
    r"|[Pp]art|[Ss]tep|[Rr]ound|[Ee]pisode|[Cc]hapter|[Ss]core|[Rr]ated)\s$"
)
_UNICODE_FRACTION_CLASS: str = "[" + "".join(_UNICODE_FRACTIONS) + "]"
_UNICODE_FRACTION_GATE_RE = re.compile(_UNICODE_FRACTION_CLASS)
_UNICODE_FRACTION_RE = re.compile(
    r"(?:" + _NB + r"(?P<w>\d{1,3})\s?)?(?P<f>" + _UNICODE_FRACTION_CLASS + r")"
)

_FEET_INCHES_RE = re.compile(
    r"(?P<ft>\d(?<![\w.,:/'\"’”-]\d)\d?)['’′]\s?(?P<in>\d{1,2})(?:[\"”″]|'')(?![\w])"
)
_UNIT_ALT: str = "|".join(
    re.escape(u) for u in sorted(_UNITS, key=len, reverse=True)
)
_UNIT_RE = re.compile(
    r"(?P<a>" + _NUM_START + r")(?:\s?[-–]\s?(?P<b>" + _NUM + r"))?"
    r"(?P<sep>[ -])?(?P<unit>" + _UNIT_ALT + r")"
    r"(?P<pow>[²³])?(?P<dot>\.(?=\s+[a-z]))?(?![\w/²³°])"
)
_METER_POWER_RE = re.compile(
    r"(?P<n>" + _NUM_START + r")\s?m(?P<pow>[²³])(?![\w])"
)
_DEGREE_RE = re.compile(
    r"(?P<a>" + _NUM_START + r")(?:\s?[-–]\s?(?P<b>" + _NUM + r"))?"
    r"\s?(?:°\s?(?P<scale>[CF])?(?![A-Za-z])|(?P<sign>[℃℉]))"
)

_NUMBER_ABBR_RE = re.compile(
    r"(?<![\w.])(?P<w>Nos?|NOS?)\.\s?(?P<n>\d+)"
    r"(?P<rest>(?:(?:\s?,\s?|\s?[-–]\s?|\s+and\s+|\s+&\s+|\s+to\s+|\s+or\s+)\d+)*)"
    r"(?![\w%°]|[.:/]\d"
    r"|\s(?:seconds?|minutes?|hours?|days?|weeks?|months?|years?|people|men"
    r"|women|times)\b)"
)
_NUMBER_REST_RE = re.compile(r"\d+|[-–]|&")
_HASH_NUMBER_RE = re.compile(r"(?<![\w#&])#(?P<n>\d{1,4})(?![\w#%°]|[.,:/-]\d)")

_PHONE_GATE_RE = re.compile(r"\d\d\d[.-]\d{4}")
_PHONE_RE = re.compile(
    _NB + r"(?:(?P<cc>\+?1)[ .-])?\(?(?P<a>\d{3})\)?[ .-]"
    r"(?P<b>\d{3})[.-](?P<c>\d{4})(?![\w]|[.,:/-]\d)"
)
_PHONE_SHORT_RE = re.compile(
    r"(?P<a>" + _D1 + r"\d{2})-(?P<b>\d{4})(?![\w]|[.,:/-]\d)"
)
_PHONE_CUE_BEFORE_RE = re.compile(
    r"\b(?:[Cc]all(?:ed|ing|s)?|[Dd]ial(?:ed|ing|s)?|[Pp]hone|[Tt]elephone"
    r"|[Tt]el\.?|[Ff]ax|[Cc]ell|[Mm]obile|[Nn]umber)\b[^\d]{0,20}$"
)
_IDENTIFIER_RE = re.compile(
    r"(?P<n>" + _D1 + r"\d{2,})(?![\w%°]|[.,:/-]\d)"
)
_IDENTIFIER_CUE_BEFORE_RE = re.compile(
    r"\b(?:(?P<short>PIN|pin|[Cc]ode|[Pp]asscode|[Cc]ombination|[Ee]xtension"
    r"|[Ee]xt\.?)|[Aa]ccount|[Aa]cct\.?|ID|[Ss]erial|[Zz]ip|ZIP|[Tt]racking"
    r"|[Bb]adge|[Ii]nvoice|[Rr]eference|[Cc]onfirmation|[Rr]outing"
    r"|[Pp]assport|[Ll]icen[sc]e)"
    r"(?:\s+(?:number|no\.|code))?\s*:?\s*#?$"
)

# Last word a cue for a digit-by-digit identifier can end with (quick reject).
_IDENTIFIER_LAST_WORDS: frozenset[str] = frozenset({
    "pin", "code", "passcode", "combination", "extension", "ext", "ext.",
    "account", "acct", "acct.", "id", "serial", "zip", "tracking", "badge",
    "invoice", "reference", "confirmation", "routing", "passport", "license",
    "licence", "number", "no.",
})

_SEPARATED_TAIL: str = r"\d{0,2}(?:,\d{3})+(?:\.\d+)?"
_SEPARATED_RE = re.compile(
    r"(?P<n>" + _D1 + _SEPARATED_TAIL + r")"
    r"(?:\s?[-–]\s?(?P<n2>\d" + _SEPARATED_TAIL + r"))?" + _NA
)
_DECIMAL_RE = re.compile(r"(?P<n>" + _D1 + r"\d*\.\d+)" + _NA)
_ROUND_RE = re.compile(
    r"(?P<n>[1-9](?<![" + _BAD_BEFORE + r"][1-9])\d{0,2}(?:000)+)" + _NA
)

_PLUS_RE = re.compile(r"(?<=\d)\s?\+\s?(?=\d)")
_TIMES_RE = re.compile(r"(?<=\d)\s?[×✕✖]\s?(?=\d)")
_DIVIDE_RE = re.compile(r"(?<=\d)\s?÷\s?(?=\d)")
_EQUALS_RE = re.compile(r"(?<=[\w)])\s=\s(?=[\w(])|(?<=\d)=(?=\d)")
_MINUS_RE = re.compile(r"(?<=\d)\s-\s(?=\d)")
_TRUE_MINUS_RE = re.compile(r"(?<=\d)\s?−\s?(?=\d)")
_PLUS_MINUS_RE = re.compile(r"±\s?(?=\d)")
_APPROX_RE = re.compile(r"≈\s?(?=\d)")
_TILDE_ABOUT_RE = re.compile(r"(?<![\w~])~\s?(?=\d)")
_TILDE_RANGE_RE = re.compile(r"(?<=\d)\s?~\s?(?=\d)")
_SECTION_SIGN_RE = re.compile(r"(§§?)\s?(?=\d)")
_AMPERSAND_RE = re.compile(r"(?<=\S)\s+&\s+(?=\S)")
_COPYRIGHT_RE = re.compile(r"(?P<word>(?i:copyright)\s?)?©\s?")
_DOUBLE_DASH_RE = re.compile(r"(?<=[^\W_])\s?-{2,3}\s?(?=[^\W_])")
_ELLIPSIS_DOUBLE_RE = re.compile(r"…(?:\s?…)+")
_SQUARED_RE = re.compile(
    r"(?<![^\W_])(?P<base>mc|[b-hj-zB-HJ-Z]|\d+)(?P<pow>[²³])(?![" + _SUPERSCRIPTS + r"])"
)


def _plural(token: str, singular: str, plural: str) -> str:
    return singular if token == "1" else plural


def _year_words(token: str) -> str | None:
    """Year reading for a digit token (commas / decimals fall back to cardinal)."""
    if token.isdigit() and token.isascii():
        return _year(int(token))
    return _say_number(token)


def _date_mdy_sub(match: re.Match) -> str:
    mon, dot, year = match.group("mon"), match.group("dot"), match.group("year")
    full = mon in _MONTHS_FULL
    day = int(match.group("day"))
    if not 1 <= day <= 31 or (full and dot):
        return match.group(0)
    if full and not match.group("suf") and not year:
        return match.group(0)                         # "January 5": leave
    if not full and not dot and not year:
        return match.group(0)                         # "Jan 5": could be a name
    out = (mon if full else _MONTH_ABBR[mon]) + " " + _ordinal(day)
    if year:
        out += match.group("sep") + _year(int(year))
    return out


def _date_dmy_sub(match: re.Match) -> str:
    mon, dot, year = match.group("mon"), match.group("dot"), match.group("year")
    suf, of = match.group("suf"), match.group("of")
    full = mon in _MONTHS_FULL
    day = int(match.group("day"))
    if not 1 <= day <= 31:
        return match.group(0)
    if not suf and not of and not year:
        return match.group(0)                         # "5 May": leave
    if not full and not dot and not year:
        return match.group(0)
    out = _ordinal(day) + " of " + (mon if full else _MONTH_ABBR[mon])
    if not of:
        before = match.string[max(0, match.start() - 5):match.start()]
        if not _THE_BEFORE_RE.search(before):
            out = "the " + out
    if year:
        out += match.group("sep") + _year(int(year))
    elif dot and (full or _AFTER_END_RE.match(match.string, match.end())):
        out += "."                                    # the period closed the sentence
    return out


def _date_my_sub(match: re.Match) -> str:
    mon, dot = match.group("mon"), match.group("dot")
    full = mon in _MONTHS_FULL
    if full == bool(dot):                              # "May." or "Jan 1920"
        return match.group(0)
    if not _YEAR_FOLLOW_RE.match(match.string, match.end()):
        return match.group(0)
    return (
        (mon if full else _MONTH_ABBR[mon]) + match.group("sep")
        + _year(int(match.group("year")))
    )


def _fix_dates(text: str) -> str:
    if not _contains_any(text, _MONTH_KEYS):
        return text
    text = _DATE_MDY_RE.sub(_date_mdy_sub, text)
    text = _DATE_DMY_RE.sub(_date_dmy_sub, text)
    return _DATE_MY_RE.sub(_date_my_sub, text)


def _meridiem(marker: str, next_char: str) -> str:
    """Normalises "am"/"pm" to "a.m."/"p.m."; other spellings are kept."""
    if marker in ("am", "pm"):
        dotted = marker[0] + ".m."
        return dotted[:-1] if next_char == "." else dotted
    return marker


def _minutes_words(minutes: int) -> str:
    if minutes < 10:
        return "oh " + _ONES[minutes]
    return _under_thousand(minutes)


def _time_meridiem_sub(match: re.Match) -> str:
    hour, minute = int(match.group("h")), int(match.group("m"))
    if not 1 <= hour <= 12 or minute > 59:
        return match.group(0)
    next_char = match.string[match.end():match.end() + 1]
    words = _cardinal(hour)
    if minute:
        words += " " + _minutes_words(minute)
    return words + " " + _meridiem(match.group("mer"), next_char)


def _time_sub(match: re.Match) -> str:
    hour, minute = int(match.group("h")), int(match.group("m"))
    if not 1 <= hour <= 23 or minute > 59:
        return match.group(0)
    if minute == 0:
        return _cardinal(hour) + (" o'clock" if hour <= 12 else " hundred")
    return _cardinal(hour) + " " + _minutes_words(minute)


def _hour_meridiem_sub(match: re.Match) -> str:
    hour = int(match.group("h"))
    if not 1 <= hour <= 12:
        return match.group(0)
    next_char = match.string[match.end():match.end() + 1]
    return _cardinal(hour) + " " + _meridiem(match.group("mer"), next_char)


def _fix_times(text: str) -> str:
    meridiem = _MERIDIEM_GATE_RE.search(text) is not None
    if meridiem:
        text = _TIME_MERIDIEM_RE.sub(_time_meridiem_sub, text)
    if ":" in text:
        text = _TIME_RE.sub(_time_sub, text)
    if meridiem:
        text = _HOUR_MERIDIEM_RE.sub(_hour_meridiem_sub, text)
    return text


def _is_attributive(match: re.Match) -> bool:
    """True for "a <amount> <noun>" ("a $5 bill", "a 10 kg bag"): unit stays singular."""
    text = match.string
    before = text[max(0, match.start() - 9):match.start()]
    return (
        _ARTICLE_BEFORE_RE.search(before) is not None
        and _LOWER_WORD_AFTER_RE.match(text, match.end()) is not None
    )


def _currency_sub(match: re.Match) -> str:
    whole = match.group(0)
    singular, plural, sub_singular, sub_plural = _CURRENCIES[match.group("sym")]
    prefix = match.group("pre")
    if prefix:
        adjective = _CURRENCY_PREFIXES[prefix]
        singular, plural = adjective + " " + singular, adjective + " " + plural
    num, num2 = match.group("num"), match.group("num2")
    magnitude = match.group("magw") or _MAGNITUDE_LETTERS.get(match.group("magl") or "")
    words = _say_number(num)
    if words is None:
        return whole
    text = match.string
    attributive = _is_attributive(match) or (
        _MONEY_NOUN_AFTER_RE.match(text, match.end()) is not None
    )
    if num2:
        words2 = _say_number(num2)
        if words2 is None:
            return whole
        amount = words + " to " + words2 + (" " + magnitude if magnitude else "")
        return amount + " " + (singular if attributive else plural)
    if magnitude:
        return words + " " + magnitude + " " + (singular if attributive else plural)
    integer, _, frac = num.replace(",", "").partition(".")
    if len(frac) == 2 and sub_singular and sub_plural:
        major, minor = int(integer), int(frac)
        parts: list[str] = []
        if major or not minor:
            parts.append(_cardinal(major) + " " + (singular if major == 1 else plural))
        if minor:
            parts.append(_cardinal(minor) + " " + (sub_singular if minor == 1 else sub_plural))
        return " and ".join(parts)
    return words + " " + (singular if num == "1" or attributive else plural)


def _currency_code_sub(match: re.Match) -> str:
    words = _say_number(match.group("num"))
    if words is None:
        return match.group(0)
    singular, plural = _CURRENCY_CODES[match.group("code")]
    magnitude = match.group("magw")
    if magnitude:
        return words + " " + magnitude + " " + plural
    return words + " " + _plural(match.group("num"), singular, plural)


def _cent_sub(match: re.Match) -> str:
    num = match.group("num")
    words = _say_number(num)
    if words is None:
        return match.group(0)
    return words + " " + _plural(num, "cent", "cents")


def _fix_currency(text: str) -> str:
    if _CURRENCY_GATE_RE.search(text):
        text = _CURRENCY_RE.sub(_currency_sub, text)
    if _contains_any(text, _CURRENCY_CODE_KEYS):
        text = _CURRENCY_CODE_RE.sub(_currency_code_sub, text)
        text = _CURRENCY_CODE_PRE_RE.sub(_currency_code_sub, text)
    if "¢" in text:
        text = _CENT_SIGN_RE.sub(_cent_sub, text)
    return text


def _range_words(low: str, high: str | None) -> str | None:
    """Words for "low" or "low to high"; None when a part cannot be worded."""
    words = _say_number(low)
    if words is None or not high:
        return words
    high_words = _say_number(high)
    if high_words is None:
        return None
    return words + " to " + high_words


def _percent_sub(match: re.Match) -> str:
    words = _range_words(match.group("a"), match.group("b"))
    if words is None:
        return match.group(0)
    return words + " percent"


def _fix_percent(text: str) -> str:
    if "%" in text:
        text = _PERCENT_RE.sub(_percent_sub, text)
    return text


def _fix_signs(text: str) -> str:
    """"-5" -> "minus 5", "+5" -> "plus 5"; later rules word the number."""
    if "+" in text:
        text = _TRAILING_PLUS_RE.sub(" plus", text)
    if "-" in text or "−" in text or "+" in text:
        text = _SIGN_RE.sub(lambda m: "plus " if m.group(0) == "+" else "minus ", text)
    return text


def _era_pre_sub(match: re.Match) -> str:
    words = _year_words(match.group("year"))
    if words is None:
        return match.group(0)
    return match.group("era") + " " + words


def _era_post_sub(match: re.Match) -> str:
    words = _year_words(match.group("year"))
    if words is None:
        return match.group(0)
    return words + " " + match.group("era")


def _decade_sub(match: re.Match) -> str:
    token = match.group("y")
    century, decade = int(token[:2]), token[2]
    if not 11 <= century <= 20:
        return match.group(0)
    if decade == "0":
        if century == 20:
            return "two thousands"
        return _under_thousand(century) + " hundreds"
    if decade == "1":
        return _under_thousand(century) + " tens"
    return _under_thousand(century) + " " + _DECADE_WORDS[decade]


def _decade_bare_sub(match: re.Match) -> str:
    before = match.string[max(0, match.start() - 8):match.start()]
    if not _DECADE_CUE_BEFORE_RE.search(before):
        return match.group(0)
    return _DECADE_WORDS[match.group("dec")]


def _year_range_sub(match: re.Match) -> str:
    first, second = int(match.group("y1")), match.group("y2")
    last = int(second)
    if len(second) == 2:
        if last <= first % 100:
            return match.group(0)
        last += first - first % 100
    elif last <= first:
        return match.group(0)
    before = match.string[max(0, match.start() - 8):match.start()]
    if _PAGES_BEFORE_RE.search(before):
        return match.group(0)
    return _year(first) + " to " + _year(last)


def _year_context_sub(match: re.Match) -> str:
    text = match.string
    if not _YEAR_FOLLOW_RE.match(text, match.end()):
        return match.group(0)
    before = text[max(0, match.start() - 40):match.start()]
    head = before.rstrip(" -")
    if head[head.rfind(" ") + 1:].lower() not in _YEAR_CUE_LAST_WORDS:
        return match.group(0)
    cue = _YEAR_CUE_BEFORE_RE.search(before)
    if cue is None:
        return match.group(0)
    first, second = int(match.group("year")), match.group("year2")
    if cue.group("span") and (not second or int(second) <= first):
        return match.group(0)                         # "from 1500" what?
    out = _year(first)
    if second:
        out += match.group("mid") + _year(int(second))
    return out


def _fix_years(text: str) -> str:
    if _contains_any(text, _ERA_KEYS):
        text = _ERA_PRE_RE.sub(_era_pre_sub, text)
        text = _ERA_POST_RE.sub(_era_post_sub, text)
    if "0s" in text or "0'" in text or "0’" in text:
        text = _DECADE_RE.sub(_decade_sub, text)
        text = _DECADE_APOS_RE.sub(lambda m: _DECADE_WORDS[m.group("dec")], text)
        text = _DECADE_BARE_RE.sub(_decade_bare_sub, text)
    if _FOUR_DIGIT_RE.search(text):
        if "-" in text or "–" in text:
            text = _YEAR_RANGE_RE.sub(_year_range_sub, text)
        text = _YEAR_CONTEXT_RE.sub(_year_context_sub, text)
    return text


def _cue_range_sub(match: re.Match) -> str:
    if int(match.group("b")) <= int(match.group("a")):
        return match.group(0)
    before = match.string[max(0, match.start() - 12):match.start()]
    if not _RANGE_CUE_BEFORE_RE.search(before):
        return match.group(0)
    return match.group("a") + " to " + match.group("b")


def _fix_cue_ranges(text: str) -> str:
    if "-" in text or "–" in text:
        text = _CUE_RANGE_RE.sub(_cue_range_sub, text)
    return text


def _ordinal_sub(match: re.Match) -> str:
    token, suffix = match.group("n").replace(",", ""), match.group("suf")
    if len(token) > 9 or (len(token) > 1 and token[0] == "0"):
        return match.group(0)
    value = int(token)
    if 11 <= value % 100 <= 13:
        expected = "th"
    else:
        expected = {1: "st", 2: "nd", 3: "rd"}.get(value % 10, "th")
    if suffix.lower() != expected:
        return match.group(0)
    words = _ordinal(value)
    return words.upper() if suffix.isupper() else words


def _fix_ordinals(text: str) -> str:
    if _ORDINAL_GATE_RE.search(text):
        text = _ORDINAL_RE.sub(_ordinal_sub, text)
    return text


def _fraction_words(numerator: int, denominator: int, mixed: bool) -> str:
    singular, plural = _FRACTION_NAMES[denominator]
    if numerator == 1:
        if mixed:
            return ("an " if singular[0] in "aeiou" else "a ") + singular
        return "one " + singular
    return _cardinal(numerator) + " " + plural


def _fraction_mixed_sub(match: re.Match) -> str:
    numerator, denominator = int(match.group("n")), int(match.group("d"))
    whole = match.group("w")
    if denominator not in _FRACTION_NAMES or not 1 <= numerator < denominator:
        return match.group(0)
    if len(whole) > 1 and whole[0] == "0":
        return match.group(0)
    return _cardinal(int(whole)) + " and " + _fraction_words(numerator, denominator, True)


def _fraction_sub(match: re.Match) -> str:
    numerator, denominator = int(match.group("n")), int(match.group("d"))
    if not 1 <= numerator < denominator:
        return match.group(0)
    text = match.string
    if denominator in _FRACTION_CUED:
        if not _FRACTION_CUE_AFTER_RE.match(text, match.end()):
            return match.group(0)
    elif denominator not in _FRACTION_ALWAYS:
        return match.group(0)
    before = text[max(0, match.start() - 8):match.start()]
    if _DATE_PREPOSITION_BEFORE_RE.search(before):
        return match.group(0)
    return _fraction_words(numerator, denominator, False)


def _unicode_fraction_sub(match: re.Match) -> str:
    numerator, denominator = _UNICODE_FRACTIONS[match.group("f")]
    whole = match.group("w")
    if whole is None:
        start = match.start()
        if start and match.string[start - 1].isalnum():
            return match.group(0)
        return _fraction_words(numerator, denominator, False)
    return _cardinal(int(whole)) + " and " + _fraction_words(numerator, denominator, True)


def _fix_fractions(text: str) -> str:
    if "/" in text or "⁄" in text:
        text = _FRACTION_MIXED_RE.sub(_fraction_mixed_sub, text)
        text = _FRACTION_RE.sub(_fraction_sub, text)
    return text


def _feet_inches_sub(match: re.Match) -> str:
    feet, inches = match.group("ft"), match.group("in")
    if int(inches) > 11 or feet[0] == "0":
        return match.group(0)
    return (
        _cardinal(int(feet)) + " " + _plural(feet, "foot", "feet") + " "
        + _cardinal(int(inches)) + " " + _plural(str(int(inches)), "inch", "inches")
    )


def _unit_sub(match: re.Match) -> str:
    words = _range_words(match.group("a"), match.group("b"))
    if words is None:
        return match.group(0)
    singular, plural = _UNITS[match.group("unit")]
    power = {"²": "square ", "³": "cubic "}.get(match.group("pow") or "", "")
    if match.group("sep") == "-":
        return words + "-" + power + singular       # "a 5-km run"
    is_one = match.group("a") == "1" and not match.group("b")
    if is_one or (not match.group("b") and _is_attributive(match)):
        return words + " " + power + singular       # "a 10 kg bag"
    return words + " " + power + plural


def _meter_power_sub(match: re.Match) -> str:
    words = _say_number(match.group("n"))
    if words is None:
        return match.group(0)
    power = "square" if match.group("pow") == "²" else "cubic"
    return words + " " + power + " " + _plural(match.group("n"), "meter", "meters")


def _degree_sub(match: re.Match) -> str:
    words = _range_words(match.group("a"), match.group("b"))
    if words is None:
        return match.group(0)
    is_one = match.group("a") == "1" and not match.group("b")
    scale = match.group("scale") or {"℃": "C", "℉": "F"}.get(match.group("sign") or "", "")
    out = words + " " + ("degree" if is_one else "degrees")
    if scale:
        out += " Celsius" if scale == "C" else " Fahrenheit"
    return out


def _fix_units(text: str) -> str:
    if "'" in text or "’" in text or "′" in text:
        text = _FEET_INCHES_RE.sub(_feet_inches_sub, text)
    text = _UNIT_RE.sub(_unit_sub, text)
    if "²" in text or "³" in text:
        text = _METER_POWER_RE.sub(_meter_power_sub, text)
    if "°" in text or "℃" in text or "℉" in text:
        text = _DEGREE_RE.sub(_degree_sub, text)
    return text


def _small_cardinal(token: str) -> str:
    """Words for a short number; longer ones stay as digits."""
    if len(token) > 4 or (len(token) > 1 and token[0] == "0"):
        return token
    return _cardinal(int(token))


def _number_abbr_sub(match: re.Match) -> str:
    word, rest = match.group("w"), match.group("rest")
    plural = word.lower().endswith("s")
    if rest and not plural:
        return match.group(0)
    out = "Number"
    if plural:
        out += "s"

    def _rest_sub(m: re.Match) -> str:
        token = m.group(0)
        if token == "&":
            return "and"
        if token in "-–":
            return " to "
        return _small_cardinal(token)

    tail = _TIDY_SPACES_RE.sub(" ", _NUMBER_REST_RE.sub(_rest_sub, rest))
    out += " " + _small_cardinal(match.group("n")) + tail
    return out.upper() if word.isupper() else out


def _fix_number_labels(text: str) -> str:
    if "o." in text or "os." in text or "O." in text or "OS." in text:
        text = _NUMBER_ABBR_RE.sub(_number_abbr_sub, text)
    if "#" in text:
        text = _HASH_NUMBER_RE.sub(
            lambda m: "number " + _small_cardinal(m.group("n")), text
        )
    return text


def _phone_sub(match: re.Match) -> str:
    groups = [match.group("a"), match.group("b"), match.group("c")]
    if match.group("cc"):
        groups.insert(0, "1")
    return ", ".join(_digits(g) for g in groups)


def _phone_short_sub(match: re.Match) -> str:
    before = match.string[max(0, match.start() - 32):match.start()]
    if not _PHONE_CUE_BEFORE_RE.search(before):
        return match.group(0)
    return _digits(match.group("a")) + ", " + _digits(match.group("b"))


def _identifier_sub(match: re.Match) -> str:
    token = match.group("n")
    before = match.string[max(0, match.start() - 40):match.start()]
    head = before.rstrip(" :#")
    if head[head.rfind(" ") + 1:].lower() not in _IDENTIFIER_LAST_WORDS:
        return match.group(0)
    cue = _IDENTIFIER_CUE_BEFORE_RE.search(before)
    if cue is None or len(token) < (3 if cue.group("short") else 5):
        return match.group(0)
    return _digits(token)


def _fix_identifiers(text: str) -> str:
    if not _THREE_DIGIT_RE.search(text):
        return text
    if _PHONE_GATE_RE.search(text):
        text = _PHONE_RE.sub(_phone_sub, text)
        text = _PHONE_SHORT_RE.sub(_phone_short_sub, text)
    return _IDENTIFIER_RE.sub(_identifier_sub, text)


def _separated_sub(match: re.Match) -> str:
    return _range_words(match.group("n"), match.group("n2")) or match.group(0)


def _plain_number_sub(match: re.Match) -> str:
    return _say_number(match.group("n")) or match.group(0)


def _round_sub(match: re.Match) -> str:
    token = match.group("n")
    if len(token) < 5:
        return match.group(0)
    return _say_number(token) or match.group(0)


def _fix_plain_numbers(text: str) -> str:
    if "," in text:
        text = _SEPARATED_RE.sub(_separated_sub, text)
    if "." in text:
        text = _DECIMAL_RE.sub(_plain_number_sub, text)
    if "000" in text:
        text = _ROUND_RE.sub(_round_sub, text)
    if "minus " in text:
        text = _MINUS_NUMBER_RE.sub(_plain_number_sub, text)
    return text


def _fix_math(text: str) -> str:
    if "+" in text:
        text = _PLUS_RE.sub(" plus ", text)
    if not text.isascii():
        text = _TIMES_RE.sub(" times ", text)
        text = _DIVIDE_RE.sub(" divided by ", text)
        text = _TRUE_MINUS_RE.sub(" minus ", text)
        text = _PLUS_MINUS_RE.sub("plus or minus ", text)
        text = _APPROX_RE.sub("approximately ", text)
    if "=" in text:
        text = _MINUS_RE.sub(" minus ", text)
        text = _EQUALS_RE.sub(" equals ", text)
    if "~" in text:
        text = _TILDE_RANGE_RE.sub(" to ", text)
        text = _TILDE_ABOUT_RE.sub("about ", text)
    return text


# Applied in this order to every English line that contains an ASCII digit.
_DIGIT_RULES: tuple[tuple[str, Callable[[str], str]], ...] = (
    ("number abbreviations", _fix_number_abbreviations),
    ("signs", _fix_signs),
    ("dates", _fix_dates),
    ("times", _fix_times),
    ("currency", _fix_currency),
    ("percent", _fix_percent),
    ("years", _fix_years),
    ("cue ranges", _fix_cue_ranges),
    ("ordinals", _fix_ordinals),
    ("fractions", _fix_fractions),
    ("units", _fix_units),
    ("number labels", _fix_number_labels),
    ("identifiers", _fix_identifiers),
    ("plain numbers", _fix_plain_numbers),
    ("math", _fix_math),
)


def _fix_symbols(text: str) -> str:
    """English readings for symbols that need no digit beside them."""
    if "&" in text:
        text = _AMPERSAND_RE.sub(" and ", text)
    if "=" in text:
        text = _EQUALS_RE.sub(" equals ", text)
    if "--" in text:
        text = _DOUBLE_DASH_RE.sub(", ", text)
    if "§" in text:
        text = _SECTION_SIGN_RE.sub(
            lambda m: "sections " if len(m.group(1)) == 2 else "section ", text
        )
    if "©" in text:
        text = _COPYRIGHT_RE.sub(
            lambda m: m.group("word") if m.group("word") else "copyright ", text
        )
    return text


def _fix_unicode_english(text: str) -> str:
    """English readings for non-ASCII forms that precede the number rules."""
    if _UNICODE_FRACTION_GATE_RE.search(text):
        text = _UNICODE_FRACTION_RE.sub(_unicode_fraction_sub, text)
    if "……" in text or "… …" in text:
        text = _ELLIPSIS_DOUBLE_RE.sub("…", text)
    return text


def _fix_powers(text: str) -> str:
    """"x²" -> "x squared"; runs after the unit rules ("5 m²" is square meters)."""
    return _SQUARED_RE.sub(
        lambda m: m.group("base") + (" squared" if m.group("pow") == "²" else " cubed"),
        text,
    )


# ══════════════════════════════════════════════════════════════════════════════
# English: all-caps runs
# ══════════════════════════════════════════════════════════════════════════════

_CAPS_GATE_RE = re.compile(r"[A-Z][A-Z][^a-z]*[A-Z][A-Z]")
_WORD_RE = re.compile(_LETTER + r"+(?:['’]" + _LETTER + r"+)*")
# What may separate the words of one all-caps run. Anything else (a slash,
# a backtick, an equals sign) means code or a table, not a shouted sentence.
_CAPS_GAP_RE = re.compile(r"[\s\d,.;:!?'\"“”‘’()\-–—…&#]*")
_SENTENCE_GAP_RE = re.compile(r"[.!?…:][\"'”’)\]]*\s+[\"'“‘(\[]*$|[\"“(\[]$")
_CAPS_HEADING_MAX_WORDS: int = 14
_CAPS_ROMAN_RE = re.compile(r"[IVX]{2,}")
_CAPS_NAME_PREFIX_RE = re.compile(r"[OD]['’][A-Z]{2,}")
_CAPS_SENTENCE_BREAK_RE = re.compile(r"[.!?…][\"'”’)\]]*\s")
# Between two words of one phrase: no comma, semicolon or sentence stop.
_CAPS_PHRASE_GAP_RE = re.compile(r"[\s\d:'\"“”‘’()\-–—#&]*")


def _caps_keep(token: str, previous: str, following: str) -> bool:
    """True when an all-caps token keeps its capitals inside a converted run."""
    if token in _KNOWN_ACRONYMS:
        return True
    if len(token) == 1:
        return token != "A"
    if (
        (len(token) >= 4 or _CAPS_ROMAN_RE.fullmatch(token))
        and _roman_value(token) is not None
    ):
        return True                                   # "XIV", "MDCCCXXXIII"
    if token == "US":
        return previous == "THE" or following in _US_FOLLOWERS
    # No vowel at all: an acronym ("HP", "NYPD") unless it is a known word.
    if len(token) <= 5 and not (set(token) & _CAPS_VOWELS):
        return token not in _CAPS_VOWELLESS_WORDS
    return False


def _caps_segments(
    text: str, run: list[re.Match], kept: list[bool]
) -> list[tuple[int, int, bool]] | None:
    """Splits an all-caps run into sentences and decides whether to convert it.

    Returns ``(start, end, convert)`` index ranges into ``run``, or None when
    the run is left alone. A sentence counts as shouted prose when two of its
    neighbouring words form a phrase, with no comma between them ("GET OUT");
    a comma-separated list of acronyms ("EPUB, MOBI, PDF") never does. The
    run converts when its shouted sentences hold three words and one of them
    has two ordinary words (not an acronym, numeral or initial).
    """
    plain = [not keep and len(tok.group(0)) >= 2 for tok, keep in zip(run, kept)]
    bounds: list[int] = [0]
    for k in range(1, len(run)):
        gap_start = run[k - 1].end()
        if not _CAPS_SENTENCE_BREAK_RE.search(text, gap_start, run[k].start()):
            continue
        previous = run[k - 1].group(0)
        if text[gap_start] == "." and (
            len(previous) == 1 or previous in _CAPS_VOWELLESS_WORDS
        ):
            continue                                  # "MR. SMITH", "JOHN F. KENNEDY"
        bounds.append(k)
    bounds.append(len(run))
    segments: list[tuple[int, int, bool]] = []
    words = 0
    ordinary = 0
    for start, end in zip(bounds, bounds[1:]):
        phrase = any(
            (plain[k - 1] or plain[k])
            and _CAPS_PHRASE_GAP_RE.fullmatch(text, run[k - 1].end(), run[k].start())
            for k in range(start + 1, end)
        )
        if phrase:
            words += end - start
            ordinary = max(ordinary, sum(plain[start:end]))
        # A lone word ("STOP!") follows its neighbours; a list is left alone.
        segments.append((start, end, phrase or end - start == 1))
    if words < 3 or ordinary < 2:
        return None
    return segments


def _fix_all_caps(text: str) -> str:
    tokens = list(_WORD_RE.finditer(text))
    count = len(tokens)
    if count < 3:
        return text
    upper = [t.group(0).isupper() for t in tokens]
    pieces: list[str] = []
    cursor = 0
    index = 0
    while index < count:
        if not upper[index]:
            index += 1
            continue
        end = index + 1
        while (
            end < count and upper[end]
            and _CAPS_GAP_RE.fullmatch(text, tokens[end - 1].end(), tokens[end].start())
        ):
            end += 1
        run = tokens[index:end]
        whole_line = index == 0 and end == count
        index = end
        if len(run) < 3:
            continue
        words = [t.group(0) for t in run]
        kept = [
            _caps_keep(
                w,
                words[k - 1] if k else "",
                words[k + 1] if k + 1 < len(words) else "",
            )
            for k, w in enumerate(words)
        ]
        segments = _caps_segments(text, run, kept)
        if segments is None:
            continue
        tail = text[run[-1].end():].strip()
        heading = (
            whole_line and len(run) <= _CAPS_HEADING_MAX_WORDS
            and not tail.endswith((".", "!", "?", ",", ";", '"', "”"))
            and not text[:run[0].start()].strip().startswith(('"', "“"))
        )
        for seg_start, seg_end, convert in segments:
            if not convert:
                continue
            for k in range(seg_start, seg_end):
                tok, word = run[k], words[k]
                if k:
                    gap = text[run[k - 1].end():tok.start()]
                    starts = _SENTENCE_GAP_RE.search(gap) is not None
                else:
                    gap = text[max(0, tok.start() - 8):tok.start()]
                    starts = not gap.strip() or _SENTENCE_GAP_RE.search(gap) is not None
                dotted = text[tok.end():tok.end() + 1] == "." and len(word) == 1
                if kept[k] or (dotted and gap.endswith(".")):
                    new = word
                elif word[:2] in ("I'", "I’"):
                    new = "I" + word[1:].lower()
                elif word == "A" and dotted:
                    new = word
                elif _CAPS_NAME_PREFIX_RE.fullmatch(word):
                    new = word[:3] + word[3:].lower()        # O'BRIEN -> O'Brien
                else:
                    new = word.lower()
                    small = new in _TITLE_SMALL_WORDS and 0 < k < len(run) - 1
                    if starts or (heading and not small):
                        new = new[0].upper() + new[1:]
                pieces.append(text[cursor:tok.start()])
                pieces.append(new)
                cursor = tok.end()
    if not pieces:
        return text
    pieces.append(text[cursor:])
    return "".join(pieces)


# ══════════════════════════════════════════════════════════════════════════════
# Line pipeline
# ══════════════════════════════════════════════════════════════════════════════


def _run(name: str, rule: Callable[..., str], text: str, *args: object) -> str:
    """Applies one rule; a failing rule is logged and skipped."""
    try:
        return rule(text, *args)
    except Exception:  # noqa: BLE001 - a rule must never break narration
        logger.debug("speech_text rule %r failed; skipped", name, exc_info=True)
        return text


def _apply_rules(text: str, english: bool) -> str:
    """One pass of every applicable rule over a single line.

    Each rule sits behind a cheap gate (a substring test or a character-class
    search), so a line of ordinary prose costs a handful of scans.
    """
    ascii_only = text.isascii()

    # Language-neutral removals that must precede the English readings.
    if not ascii_only and _INVISIBLE_RE.search(text):
        text = _INVISIBLE_RE.sub("", text)
    if "://" in text or "www." in text or "WWW." in text:
        text = _run("urls", _fix_urls, text, english)
    if "[" in text:
        text = _run("footnotes", _fix_footnotes, text, english)

    if english:
        if "." in text:
            if "@" in text or _contains_any(text, _BARE_DOMAIN_KEYS):
                text = _run("addresses", _fix_addresses, text)
        if _has_abbreviation(text):
            text = _run("abbreviations", _fix_abbreviations, text)
        text = _run("roman numerals", _fix_roman_numerals, text)
        if not ascii_only:
            text = _run("unicode forms", _fix_unicode_english, text)
        if _ASCII_DIGIT_RE.search(text):
            for name, rule in _DIGIT_RULES:
                text = _run(name, rule, text)
        if "&" in text or "=" in text or "--" in text or not ascii_only:
            text = _run("symbols", _fix_symbols, text)
        if not ascii_only and ("²" in text or "³" in text):
            text = _run("powers", _fix_powers, text)

    # Language-neutral noise removal.
    if not ascii_only:
        if _SUPERSCRIPT_GATE_RE.search(text):
            text = _run("superscripts", _fix_superscript_notes, text, english)
        if _DECOR_GATE_RE.search(text):
            text = _run("decorations", _fix_decorations, text, english)
    elif "|" in text:
        text = _run("decorations", _fix_decorations, text, english)
    if _contains_any(text, _PUNCT_KEYS):
        text = _run("repeated punctuation", _fix_repeated_punctuation, text, english)
    if "*" in text or "_" in text or "~" in text or "^" in text:
        text = _run("stray symbols", _fix_stray_symbols, text, english)

    if english and _CAPS_GATE_RE.search(text):
        text = _run("all caps", _fix_all_caps, text)
    return text


def _process_line(line: str, english: bool) -> str:
    """Rewrites one line to a fixed point; whitespace at its edges is kept."""
    core = line.strip()
    if not core or _ALNUM_RE.search(core) is None:
        return line                 # blank, or pure decoration the splitter drops
    original = core
    for _ in range(_MAX_PASSES):
        new = _apply_rules(core, english)
        if new != core:
            new = _tidy(new)
        if new == core:
            break
        core = new
    if core == original:
        return line
    if not core or _ALNUM_RE.search(core) is None:
        return line                 # never blank out a line: paragraphs must hold
    lead = line[:len(line) - len(line.lstrip())]
    trail = line[len(line.rstrip()):]
    return lead + core + trail


def _convert_roman_headings(lines: list[str]) -> None:
    """Rewrites, in place, heading lines that are only a roman numeral.

    Runs after the line rules so that a heading freed of a footnote marker
    ("XII.¹") is seen, and looks at every such line of the text at once: an
    ambiguous numeral ("I", "X") only counts when its neighbour is there too.
    """
    candidates: list[tuple[int, str]] = []
    values: set[int] = set()
    for index, line in enumerate(lines):
        if 0 < len(line) <= _ROMAN_HEADING_MAX_LEN:
            core = line.strip()
            match = _ROMAN_ONLY_RE.fullmatch(core)
            if match is not None:
                value = _roman_value(match.group(1))
                if value is not None:
                    candidates.append((index, core))
                    values.add(value)
    roman_lines = frozenset(values)
    for index, core in candidates:
        heading = _standalone_roman(core, roman_lines)
        if heading is not None:
            line = lines[index]
            lead = line[:len(line) - len(line.lstrip())]
            lines[index] = lead + heading + line[len(line.rstrip()):]


def _is_english(language: object) -> bool:
    return isinstance(language, str) and language.strip().lower() in _ENGLISH_ALIASES


# ══════════════════════════════════════════════════════════════════════════════
# Public API
# ══════════════════════════════════════════════════════════════════════════════


def supported_languages() -> tuple[str, ...]:
    """Languages that have a full written-form rule set.

    Returns
    -------
    tuple[str, ...]
        Language names as the app stores them. Any other language (including
        "Auto") is still accepted by :func:`normalize_for_speech` and receives
        the language-neutral cleanups only.
    """
    return _FULL_RULESET_LANGUAGES


def normalize_for_speech(text: str, language: str = "English") -> str:
    """Rewrites written forms a TTS model misreads into speakable text.

    Parameters
    ----------
    text : str
        Cleaned chapter text (paragraphs separated by blank lines).
    language : str
        Book language as the app stores it ("English", "Chinese", "Auto", ...).
        English gets the full rule set; every other value gets only the
        language-neutral cleanups (footnote markers, URLs, symbol noise,
        repeated punctuation, superscripts).

    Returns
    -------
    str
        The normalised text. Line and paragraph structure is identical to the
        input; the function is idempotent and never raises.
    """
    if not isinstance(text, str):
        return "" if text is None else str(text)
    if not text:
        return text
    try:
        english = _is_english(language)
        lines = text.split("\n")
        for index, line in enumerate(lines):
            if not line:
                continue
            try:
                lines[index] = _process_line(line, english)
            except Exception:  # noqa: BLE001 - leave the line as written
                logger.debug("speech_text failed on a line; left unchanged", exc_info=True)
        if english:
            _convert_roman_headings(lines)
        return "\n".join(lines)
    except Exception:  # noqa: BLE001 - never break the pipeline over text polish
        logger.debug("speech_text failed; text left unchanged", exc_info=True)
        return text
