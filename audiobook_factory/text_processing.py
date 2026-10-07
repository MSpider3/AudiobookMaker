import re
import nltk
from nltk.tokenize import sent_tokenize

# Ensure NLTK data is ready
try:
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
except Exception as e:
    print(f"[WARNING] Failed to load/download NLTK punkt tokenizer: {e}")

try:
    try:
        nltk.data.find('tokenizers/punkt_tab')
    except LookupError:
        nltk.download('punkt_tab', quiet=True)
except Exception as e:
    print(f"[WARNING] Failed to load/download NLTK punkt_tab: {e}")

try:
    import audiobook_rust
    _RUST_AVAILABLE = hasattr(audiobook_rust, "normalize_text") and hasattr(audiobook_rust, "split_sentences")
except ImportError:
    _RUST_AVAILABLE = False

def normalize_text(text):
    """
    Sanitizes text *before* splitting.
    1. Fixes 'De-wrapping' (broken line breaks mid-sentence).
    2. Merges Drop Caps (e.g. "T" + "HERE" -> "THERE").
    """
    if _RUST_AVAILABLE:
        return audiobook_rust.normalize_text(text)
    return _python_normalize_text(text)

def _python_normalize_text(text):
    # 1. De-wrapping: Join lines if they don't look like distinct paragraphs.
    # Logic: If a line ends with non-punctuation and next starts with lowercase, it's a broken wrap.
    # Ex: "he is about to come, we\nwill prepare" -> "he is about to come, we will prepare"
    text = re.sub(r'(?<=[^.!?;:"\n])\n\s*(?=[a-z])', ' ', text)

    # 2. Drop Cap Merge (Simple Heuristic for "T\nHERE" -> "THERE")
    # Searches for Single Capital Char -> Newline/Space -> Capital Word(min 2 chars)
    # This is a bit aggressive, so we perform it carefully.
    text = re.sub(r'(^|\n)([A-Z])\s*\n\s*([A-Z]{2,})', r'\1\2\3', text)

    return text

# Quotation marks and brackets that can only open, or only close. Sentence
# tokenizers leave them on the wrong side of a break in text that puts no
# space around them (Chinese, Japanese) or a space inside them (French).
_OPENING_MARKS = "\u201c\u2018\u300c\u300e\uff08\u00ab\u300a\u3008\u3010"
_CLOSING_MARKS = "\u201d\u300d\u300f\uff09\u00bb\u300b\u3009\u3011"
# Sentence enders the Punkt tokenizer does not know: CJK full stops and the
# Devanagari danda, with any closing marks that follow them.
_WIDE_SENTENCE_END = re.compile(
    "(?<=[\u3002\uff01\uff1f\u0964\u0965])(?![\u3002\uff01\uff1f\u0964\u0965" + _CLOSING_MARKS + "\u2019])"
)
# Abbreviations that are followed by a name or a number, never by a new
# sentence, in languages the tokenizers have no abbreviation list for.
_NON_FINAL_ABBREVIATIONS = frozenset({
    "\u0443\u043b", "\u0433", "\u0438\u043c", "\u043f\u0440\u043e\u0444", "\u0442\u043e\u0432", "\u0441\u0442\u0440", "\u0441\u043c", "\u0440\u0438\u0441",
    "\u0921\u0949", "\u092a\u094d\u0930\u094b", "\u0936\u094d\u0930\u0940", "\u0936\u094d\u0930\u0940\u092e\u0924\u0940",
    "mme", "mlle", "mm", "cf", "sra", "srta", "dra", "sig", "nr", "hr",
})
_LAST_TOKEN_BEFORE_DOT = re.compile(r"(\S+?)\.$")


def _repair_sentence_edges(sentences, max_len):
    """Fixes breaks a sentence tokenizer puts in the wrong place.

    * An opening quote left at the end of a sentence moves to the start of
      the next one, and a closing quote that starts a sentence moves back.
    * The same for straight double quotes in Chinese and Japanese text, where
      extraction has already replaced the curly ones: whether a quote that
      follows a full stop opens or closes is told from the quotes before it.
    * A sentence that ends in an abbreviation such as "\u0443\u043b." or "\u0921\u0949." is
      joined to the one after it.
    """
    repaired = []
    carried = ""
    quote_open = False  # inside a straight-quoted passage at the start of the sentence
    for sentence in sentences:
        sentence = carried + sentence
        carried = ""
        if quote_open and sentence[:1] == '"' and repaired and repaired[-1][-1:] in "\u3002\uff01\uff1f":
            # A quotation is open, so this quote closes the previous sentence.
            repaired[-1] += '"'
            sentence = sentence[1:].lstrip()
            quote_open = False
            if not sentence:
                continue
        if len(sentence) >= 2 and sentence[-1] == '"' and sentence[-2] in "\u3002\uff01\uff1f":
            inside = quote_open ^ ((sentence.count('"') - 1) % 2 == 1)
            if not inside:
                # Nothing is open, so this quote starts the next sentence.
                carried = '"'
                sentence = sentence[:-1]
        quote_open ^= sentence.count('"') % 2 == 1
        # Closing marks at the start belong to the previous sentence.
        lead = 0
        while lead < len(sentence) and sentence[lead] in _CLOSING_MARKS:
            lead += 1
        if lead and repaired:
            marks = sentence[:lead]
            repaired[-1] += (" " + marks) if marks[0] == "\u00bb" else marks
            sentence = sentence[lead:].lstrip()
        # Opening marks at the end belong to the next sentence.
        tail = len(sentence)
        while tail > 0 and sentence[tail - 1] in _OPENING_MARKS:
            tail -= 1
        if tail < len(sentence) and tail > 0:
            marks = sentence[tail:]
            carried = (marks + " ") if marks[-1] == "\u00ab" else marks
            sentence = sentence[:tail].rstrip()
        if not sentence:
            continue
        previous = repaired[-1] if repaired else ""
        match = _LAST_TOKEN_BEFORE_DOT.search(previous)
        if (
            match
            and match.group(1).lstrip("\"'(\u00ab\u201c\u2018").casefold() in _NON_FINAL_ABBREVIATIONS
            and len(previous) + 1 + len(sentence) <= max_len
        ):
            repaired[-1] = previous + " " + sentence
        else:
            repaired.append(sentence)
    if carried.strip() and repaired:
        repaired[-1] += carried.strip()
    return repaired


def smart_sentence_splitter(text, max_len=399):
    """
    Hierarchical split strategy:
    Level 1: Paragraphs (\n\n)
    Level 2: Sentences (Rust segmenter, or NLTK when the extension is absent)
    Level 3: Soft Split on Punctuation (if sentence > max_len)

    Quotes and abbreviations that the tokenizer split badly are repaired
    afterwards, the same way for both backends.
    """
    max_len = max(1, int(max_len))
    if _RUST_AVAILABLE:
        chunks = audiobook_rust.split_sentences(text, max_len)
    else:
        chunks = _python_split_sentences(text, max_len)
    return _repair_sentence_edges(chunks, max_len)


def _python_split_sentences(text, max_len):
    paragraphs = text.split('\n\n')
    final_chunks = []

    for para in paragraphs:
        cleaned_para = para.strip()
        if not cleaned_para: 
            continue
            
        # Level 2: Sentence Split
        sentences = []
        for candidate in sent_tokenize(cleaned_para):
            sentences.extend(
                piece.strip() for piece in _WIDE_SENTENCE_END.split(candidate) if piece.strip()
            )
        
        for i, sentence in enumerate(sentences):
            # Check length
            if len(sentence) <= max_len:
                final_chunks.append(sentence)
            else:
                # Level 3: Soft Split
                # We need to break this long sentence down.
                sub_chunks = _soft_split_long_sentence(sentence, max_len)
                final_chunks.extend(sub_chunks)

    # A chunk with no letter or digit ("*", "...", "#") gives a TTS model
    # nothing to say and tends to come back as noise or a hallucinated word.
    return [c for c in final_chunks if any(ch.isalnum() for ch in c)]

def _soft_split_long_sentence(sentence, max_len):
    """
    Splits a long sentence trying to respect punctuation boundaries.
    """
    chunks = []
    current_text = sentence
    
    while len(current_text) > max_len:
        # Find best split point
        # Grade 1: "Major" stops (semicolon, em-dash, colon)
        match = re.search(r'[;:—]', current_text[:max_len])  # Search backwards is better usually
        # Actually standard rfind is safer for specific chars.
        
        best_split_idx = -1
        
        # Priority 1: Sentence-like pauses (ASCII and full-width forms)
        for char in [';', ':', '—', '\uff1b', '\uff1a']:
             idx = current_text.rfind(char, 0, max_len)
             if idx > best_split_idx:
                 best_split_idx = idx
        
        # Priority 2: Commas (very common, acceptable split)
        if best_split_idx == -1:
            best_split_idx = max(current_text.rfind(char, 0, max_len) for char in (',', '\uff0c', '\u3001'))
            
        # Priority 3: Spaces (Last resort)
        if best_split_idx == -1:
             best_split_idx = current_text.rfind(' ', 0, max_len)
             
        # Do the split
        # We include the punctuation in the first part usually to imply the pause
        if best_split_idx == -1:
            # Priority 4: Hard limit (just chop)
            split_point = max(1, max_len)
        else:
            split_point = best_split_idx + 1
        
        chunk = current_text[:split_point].strip()
        if chunk: chunks.append(chunk)
        
        current_text = current_text[split_point:].strip()
        
    if current_text:
        chunks.append(current_text)
        
    return chunks
