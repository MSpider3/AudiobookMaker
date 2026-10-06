use unicode_segmentation::UnicodeSegmentation;

/// Smart sentence splitter matching Level 1 (paragraphs), Level 2 (sentence tokenize with abbreviation fixes),
/// and Level 3 (soft split on punctuation for long sentences).
pub fn split_sentences_rust(text: &str, max_len: usize) -> Vec<String> {
    let mut final_chunks = Vec::new();
    let paragraphs = text.split("\n\n");

    for para in paragraphs {
        let cleaned_para = para.trim();
        if cleaned_para.is_empty() {
            continue;
        }

        // Level 2: Sentence Split with NLTK-like abbreviation fixes
        let sentences = segment_sentences(cleaned_para);

        for sentence in sentences {
            // Lengths are in characters, matching Python's len(); byte
            // lengths would cut non-Latin text 2-3x shorter than requested.
            if sentence.chars().count() <= max_len {
                final_chunks.push(sentence.to_string());
            } else {
                // Level 3: Soft Split
                let sub_chunks = soft_split_long_sentence(&sentence, max_len);
                final_chunks.extend(sub_chunks);
            }
        }
    }

    final_chunks
}

/// Segment paragraph into sentences while avoiding splits on abbreviations.
fn segment_sentences(para: &str) -> Vec<String> {
    let raw_slices: Vec<&str> = para.unicode_sentences().collect();
    if raw_slices.is_empty() {
        return Vec::new();
    }

    let abbreviations = [
        "mr", "mrs", "ms", "dr", "prof", "sr", "jr",
        "lt", "col", "gen", "capt", "sgt",
        "st", "ave", "rd", "co", "inc", "ltd",
        "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
        "eg", "ie", "vs", "ca"
    ];

    let mut sentences = Vec::new();
    let mut current_sentence = String::new();

    for (idx, slice) in raw_slices.iter().enumerate() {
        let trimmed_slice = slice.trim();
        if trimmed_slice.is_empty() {
            continue;
        }

        if current_sentence.is_empty() {
            current_sentence.push_str(slice);
        } else {
            // Check if current_sentence ends with an abbreviation or a single uppercase letter (initial)
            let prev_trimmed = current_sentence.trim_end();
            
            // Find last word of current sentence (alphabetic chars before the ending punctuation)
            let last_word = prev_trimmed
                .split(|c: char| !c.is_alphabetic())
                .filter(|s| !s.is_empty())
                .last()
                .unwrap_or("");

            let word_lower = last_word.to_lowercase();
            let is_abbrev = abbreviations.contains(&word_lower.as_str());
            
            // Checks for middle initials like "John D. Rockefeller" - last_word is 1 capital letter.
            let is_initial = last_word.len() == 1 && last_word.chars().next().unwrap().is_uppercase();

            if is_abbrev || is_initial {
                // Merge since it's likely not a real sentence end
                current_sentence.push_str(slice);
            } else {
                // Push the complete sentence and start a new one
                sentences.push(current_sentence.trim().to_string());
                current_sentence = slice.to_string();
            }
        }

        // Push last item
        if idx == raw_slices.len() - 1 && !current_sentence.is_empty() {
            sentences.push(current_sentence.trim().to_string());
        }
    }

    sentences
}

/// Byte offset of the `n`-th character of `s` (or `s.len()` if it is shorter).
fn byte_offset_of_char(s: &str, n: usize) -> usize {
    s.char_indices().nth(n).map_or(s.len(), |(i, _)| i)
}

/// Splits a long sentence trying to respect punctuation boundaries.
///
/// `max_len` is a character count. Every slice index is derived from
/// `char_indices`/`rfind`, so it always lands on a UTF-8 boundary.
fn soft_split_long_sentence(sentence: &str, max_len: usize) -> Vec<String> {
    let max_len = max_len.max(1);
    let mut chunks = Vec::new();
    let mut current_text = sentence.trim();

    while current_text.chars().count() > max_len {
        let window_end = byte_offset_of_char(current_text, max_len);
        let sub = &current_text[..window_end];
        // End (exclusive, in bytes) of the chunk if we split after the best delimiter.
        let mut best_split_end: Option<usize> = None;

        // Priority 1: Sentence-like pauses (semicolons, colons, em-dashes)
        for c in [';', ':', '—'] {
            if let Some(idx) = sub.rfind(c) {
                let end = idx + c.len_utf8();
                if best_split_end.map_or(true, |best| end > best) {
                    best_split_end = Some(end);
                }
            }
        }

        // Priority 2: Commas
        if best_split_end.is_none() {
            best_split_end = sub.rfind(',').map(|idx| idx + 1);
        }

        // Priority 3: Spaces
        if best_split_end.is_none() {
            best_split_end = sub.rfind(' ').map(|idx| idx + 1);
        }

        // Priority 4: Hard limit (chop after max_len characters)
        let split_point = best_split_end.unwrap_or(window_end);

        let chunk = current_text[..split_point].trim();
        if !chunk.is_empty() {
            chunks.push(chunk.to_string());
        }

        current_text = current_text[split_point..].trim();
    }

    if !current_text.is_empty() {
        chunks.push(current_text.to_string());
    }

    chunks
}
