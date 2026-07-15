use unicode_segmentation::UnicodeSegmentation;
use regex::Regex;
use std::sync::OnceLock;

static RE_SOFT_SPLIT_CHAR: OnceLock<Regex> = OnceLock::new();

fn get_soft_split_re() -> &'static Regex {
    RE_SOFT_SPLIT_CHAR.get_or_init(|| Regex::new(r"[;:—]").unwrap())
}

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
            if sentence.len() <= max_len {
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

/// Splits a long sentence trying to respect punctuation boundaries.
fn soft_split_long_sentence(sentence: &str, max_len: usize) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut current_text = sentence.trim();

    while current_text.len() > max_len {
        let sub = &current_text[..max_len];
        let mut best_split_idx = None;

        // Priority 1: Sentence-like pauses (semicolons, colons, em-dashes)
        for c in [';', ':', '—'] {
            if let Some(idx) = sub.rfind(c) {
                if best_split_idx.map_or(true, |best| idx > best) {
                    best_split_idx = Some(idx);
                }
            }
        }

        // Priority 2: Commas
        if best_split_idx.is_none() {
            if let Some(idx) = sub.rfind(',') {
                best_split_idx = Some(idx);
            }
        }

        // Priority 3: Spaces
        if best_split_idx.is_none() {
            if let Some(idx) = sub.rfind(' ') {
                best_split_idx = Some(idx);
            }
        }

        // Priority 4: Hard limit (chop at character boundary)
        let split_point = match best_split_idx {
            Some(idx) => idx + 1,
            None => {
                // Find nearest UTF-8 character boundary
                let mut bound = max_len;
                while bound > 0 && !current_text.is_char_boundary(bound) {
                    bound -= 1;
                }
                if bound == 0 {
                    max_len
                } else {
                    bound
                }
            }
        };

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
