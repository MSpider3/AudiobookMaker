import re
import nltk
from nltk.tokenize import sent_tokenize

# Ensure NLTK data is ready
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

def normalize_text(text):
    """
    Sanitizes text *before* splitting.
    1. Fixes 'De-wrapping' (broken line breaks mid-sentence).
    2. Merges Drop Caps (e.g. "T" + "HERE" -> "THERE").
    """
    # 1. De-wrapping: Join lines if they don't look like distinct paragraphs.
    # Logic: If a line ends with non-punctuation and next starts with lowercase, it's a broken wrap.
    # Ex: "he is about to come, we\nwill prepare" -> "he is about to come, we will prepare"
    text = re.sub(r'(?<=[^.!?;:"\n])\n\s*(?=[a-z])', ' ', text)

    # 2. Drop Cap Merge (Simple Heuristic for "T\nHERE" -> "THERE")
    # Searches for Single Capital Char -> Newline/Space -> Capital Word(min 2 chars)
    # This is a bit aggressive, so we perform it carefully.
    text = re.sub(r'(^|\n)([A-Z])\s*\n\s*([A-Z]{2,})', r'\1\2\3', text)
    
    return text

def smart_sentence_splitter(text, max_len=399):
    """
    Hierarchical split strategy:
    Level 1: Paragraphs (\n\n)
    Level 2: NLTK Sentences
    Level 3: Soft Split on Punctuation (if sentence > max_len)
    """
    # 0. Normalize first
    # (Optional: caller might have already normalized, but doing it again is cheap)
    
    paragraphs = text.split('\n\n')
    final_chunks = []

    for para in paragraphs:
        cleaned_para = para.strip()
        if not cleaned_para: 
            continue
            
        # Level 2: Sentence Split
        sentences = sent_tokenize(cleaned_para)
        
        for i, sentence in enumerate(sentences):
            # Check length
            if len(sentence) <= max_len:
                final_chunks.append(sentence)
            else:
                # Level 3: Soft Split
                # We need to break this long sentence down.
                sub_chunks = _soft_split_long_sentence(sentence, max_len)
                final_chunks.extend(sub_chunks)
                
    return final_chunks

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
        
        # Priority 1: Sentence-like pauses
        for char in [';', ':', '—']:
             idx = current_text.rfind(char, 0, max_len)
             if idx > best_split_idx:
                 best_split_idx = idx
        
        # Priority 2: Commas (very common, acceptable split)
        if best_split_idx == -1:
            best_split_idx = current_text.rfind(',', 0, max_len)
            
        # Priority 3: Spaces (Last resort)
        if best_split_idx == -1:
             best_split_idx = current_text.rfind(' ', 0, max_len)
             
        # Priority 4: Hard limit (just chop)
        if best_split_idx == -1:
            best_split_idx = max_len
            
        # Do the split
        # We include the punctuation in the first part usually to imply the pause
        split_point = best_split_idx + 1
        
        chunk = current_text[:split_point].strip()
        if chunk: chunks.append(chunk)
        
        current_text = current_text[split_point:].strip()
        
    if current_text:
        chunks.append(current_text)
        
    return chunks
