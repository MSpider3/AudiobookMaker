use regex::{Regex, Captures};
use std::sync::OnceLock;
use std::collections::HashMap;

// Initialize compile-once regular expressions
static RE_DEWRAP: OnceLock<Regex> = OnceLock::new();
static RE_DROPCAP: OnceLock<Regex> = OnceLock::new();
static RE_CLASS_C: OnceLock<Regex> = OnceLock::new();
static RE_ISO_CAP: OnceLock<Regex> = OnceLock::new();
static RE_ALL_CAPS: OnceLock<Regex> = OnceLock::new();
static RE_HYPHEN: OnceLock<Regex> = OnceLock::new();
static RE_SOFT_WRAP: OnceLock<Regex> = OnceLock::new();

fn get_deway_re() -> &'static Regex {
    RE_DEWRAP.get_or_init(|| Regex::new(r"([^.!?;:\x22\n])\n\s*([a-z])").unwrap())
}

fn get_dropcap_re() -> &'static Regex {
    RE_DROPCAP.get_or_init(|| Regex::new(r"(^|\n)([A-Z])\s*\n\s*([A-Z]{2,})").unwrap())
}

fn get_class_c_re() -> &'static Regex {
    RE_CLASS_C.get_or_init(|| {
        let descriptors = "Class|Room|Section|Level|Floor|Group|Area|Zone|Rank|Type|Grade|\
                           Exam|Test|Point|Score|Rank|Phase|Stage|Category|Model|Series|\
                           Volume|Chapter|Year|Course|Subject|Unit|Part|Item|Step";
        Regex::new(&format!(r"(?i)(\b(?:{})\s+)([A-Z])([a-z]{{2,}})", descriptors)).unwrap()
    })
}

fn get_iso_cap_re() -> &'static Regex {
    // Equivalent to (?<![a-zA-Z])([A-Z])\s+([a-zA-Z]{1,})\b
    // We match a boundary non-alphabet character or start of line
    RE_ISO_CAP.get_or_init(|| Regex::new(r"(^|[^a-zA-Z])([A-Z])\s+([a-zA-Z]{1,})\b").unwrap())
}

fn get_all_caps_re() -> &'static Regex {
    RE_ALL_CAPS.get_or_init(|| Regex::new(r"\b([A-Z])\s+([A-Z]+)\b").unwrap())
}

fn get_hyphen_re() -> &'static Regex {
    RE_HYPHEN.get_or_init(|| Regex::new(r"-\n(\S)").unwrap())
}

fn get_soft_wrap_re() -> &'static Regex {
    // Original: (?<![.!?:;\"'\u2019\u201d])\n(?=[a-z])
    // Since lookbehind is unsupported, we can match:
    // ([^.!?:;\"'\u2019\u201d])\n([a-z])
    // and replace with ${1} ${2}
    RE_SOFT_WRAP.get_or_init(|| Regex::new(r"([^.!?:;\x22'\u2019\u201d])\n([a-z])").unwrap())
}

/// Sanitizes text *before* splitting (legacy audiobook_factory/text_processing.py normalizer).
pub fn normalize_text_rust(text: &str) -> String {
    // 1. De-wrapping
    let text = get_deway_re().replace_all(text, "$1 $2");

    // 2. Drop cap merge
    let text = get_dropcap_re().replace_all(&text, "$1$2$3");

    // 3. Class C merged identifier
    let text = get_class_c_re().replace_all(&text, |caps: &Captures| {
        let prefix = caps.get(1).map_or("", |m| m.as_str());
        let cap = caps.get(2).map_or("", |m| m.as_str());
        let rest = caps.get(3).map_or("", |m| m.as_str());
        let combined = format!("{}{}", cap, rest);
        let combined_lower = combined.to_lowercase();
        
        const NO_SPLIT_WORDS: &[&str] = &[
            "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
            "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen", "twenty",
            "first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth", "tenth",
            "title", "name", "number", "code", "type", "level", "value", "status", "description", "details",
            "list", "info", "room", "leader", "member", "head", "base", "group", "area", "zone", "rank",
            "grade", "exam", "test", "point", "score", "phase", "stage", "category", "model", "series",
            "volume", "chapter", "year", "course", "subject", "unit", "part", "item", "step", "boss", "user"
        ];
        
        if NO_SPLIT_WORDS.contains(&combined_lower.as_str()) {
            format!("{}{}", prefix, combined)
        } else {
            format!("{}{} {}", prefix, cap, rest)
        }
    });

    text.into_owned()
}

/// Fixes PDF kerning issues where a single capital letter is detached.
fn fix_isolated_capitals(text: &str) -> String {
    // First, handle the mixed case splits
    let text = get_iso_cap_re().replace_all(text, |caps: &Captures| {
        let prefix = caps.get(1).map_or("", |m| m.as_str());
        let cap = caps.get(2).map_or("", |m| m.as_str());
        let rest = caps.get(3).map_or("", |m| m.as_str());
        
        let merged = if cap == "A" || cap == "I" {
            let rest_lower = rest.to_lowercase();
            if cap == "A" && matches!(rest_lower.as_str(), "nd" | "s" | "t" | "re" | "n" | "ll" | "ny" | "lthough" | "gain" | "nother" | "lready" | "lways") {
                format!("{}{}", cap, rest)
            } else if cap == "I" && matches!(rest_lower.as_str(), "t" | "s" | "f" | "n" | "ll" | "nto" | "ndeed" | "tself") {
                format!("{}{}", cap, rest)
            } else {
                format!("{} {}", cap, rest)
            }
        } else {
            format!("{}{}", cap, rest)
        };
        format!("{}{}", prefix, merged)
    });

    // Then, handle the all-caps splits
    let text = get_all_caps_re().replace_all(&text, |caps: &Captures| {
        let cap = caps.get(1).map_or("", |m| m.as_str());
        let rest = caps.get(2).map_or("", |m| m.as_str());
        
        if cap == "A" || cap == "I" {
            if cap == "A" && matches!(rest, "ND" | "S" | "T" | "RE" | "N" | "LL" | "NY") {
                format!("{}{}", cap, rest)
            } else if cap == "I" && matches!(rest, "T" | "S" | "F" | "N") {
                format!("{}{}", cap, rest)
            } else {
                format!("{} {}", cap, rest)
            }
        } else {
            format!("{}{}", cap, rest)
        }
    });

    text.into_owned()
}

/// Removes footnote links, images, headings, collapses spaces, etc. (TextNormalizer class level).
pub fn clean_text_full(raw_md: &str, title: &str, is_pdf: bool) -> String {
    let mut text = raw_md.to_string();

    // Remove PDF headers/footers noise if PDF
    if is_pdf {
        text = strip_pdf_noise(&text);
    }

    // Remove duplicate title heading
    text = remove_duplicate_title(title, &text);

    // Fix broken lines
    text = get_hyphen_re().replace_all(&text, "$1").into_owned();
    text = get_soft_wrap_re().replace_all(&text, "$1 $2").into_owned();
    text = fix_isolated_capitals(&text);

    // Smart quote translations & noise stripping
    text = strip_noise(&text);

    // Call fallback/legacy text normalizer
    normalize_text_rust(&text)
}

fn strip_pdf_noise(text: &str) -> String {
    let re_page = Regex::new(r"(?m)^\s*\d{1,4}\s*$").unwrap();
    let re_img_placeholder = Regex::new(r"(?m)^<!-- image -->\s*$").unwrap();
    let re_garbled = Regex::new(r"^[A-Z][a-zA-Z]{11,}$").unwrap();
    let re_md_heading = Regex::new(r"^#+\s*").unwrap();

    // 1. Page numbers
    let text = re_page.replace_all(text, "");
    // 2. Image placeholders
    let text = re_img_placeholder.replace_all(&text, "");

    // 3. Garbled OCR lines
    let mut lines: Vec<&str> = text.split('\n').collect();
    let mut cleaned_lines = Vec::with_capacity(lines.len());
    for line in lines {
        let stripped = line.trim();
        let bare = re_md_heading.replace(stripped, "").into_owned();
        let bare_trimmed = bare.trim();
        if !bare_trimmed.is_empty() && !bare_trimmed.contains(' ') && re_garbled.is_match(bare_trimmed) {
            continue;
        }
        cleaned_lines.push(line);
    }
    let text = cleaned_lines.join("\n");

    // 4. Repeating headers/footers
    let lines: Vec<&str> = text.split('\n').collect();
    let mut counts = HashMap::new();
    for line in &lines {
        let stripped = line.trim();
        let len = stripped.len();
        if len >= 3 && len <= 60 && !stripped.ends_with('.') && !stripped.ends_with('!') 
            && !stripped.ends_with('?') && !stripped.ends_with(':') && !stripped.ends_with(',') {
            *counts.entry(stripped).or_insert(0) += 1;
        }
    }

    let repeating: std::collections::HashSet<&str> = counts.into_iter()
        .filter(|&(_, cnt)| cnt >= 3)
        .map(|(ln, _)| ln)
        .collect();

    if !repeating.is_empty() {
        lines.into_iter()
            .filter(|l| !repeating.contains(l.trim()))
            .collect::<Vec<&str>>()
            .join("\n")
    } else {
        text
    }
}

fn remove_duplicate_title(title: &str, text: &str) -> String {
    let re_md_heading = Regex::new(r"^#+\s*").unwrap();
    let lines: Vec<&str> = text.split('\n').collect();
    let mut cleaned = Vec::with_capacity(lines.len());
    let stripped_title = title.trim().to_lowercase();
    
    let mut skipped = false;
    for (i, line) in lines.iter().enumerate() {
        let bare = re_md_heading.replace(line, "").into_owned();
        let bare_lower = bare.trim().to_lowercase();
        if i < 4 && bare_lower == stripped_title && !cleaned.is_empty() && !skipped {
            skipped = true;
            continue; // Skip duplicate title line
        }
        cleaned.push(*line);
    }
    cleaned.join("\n")
}

fn strip_noise(text: &str) -> String {
    let re_ocr_prefix = Regex::new(r"OCR_IMG_TEXT:\s*").unwrap();
    let re_img_tag = Regex::new(r"!\[[^\]]*\]\([^)]*\)").unwrap();
    let re_hr = Regex::new(r"(?m)^\s*(-{3,}|\*{3,}|_{3,})\s*$").unwrap();
    let re_bold_em = Regex::new(r"\*{1,2}([^*]+?)\*{1,2}|_{1,2}([^_]+?)_{1,2}").unwrap();
    let re_footnote = Regex::new(r"\[\[\d+\]\]\([^)]+\)|\[\d+\]\([^)]+\)").unwrap();
    let re_multi_bl = Regex::new(r"\n{3,}").unwrap();

    let text = re_ocr_prefix.replace_all(text, "");
    let text = re_img_tag.replace_all(&text, "");
    let text = re_hr.replace_all(&text, "\n\n");
    
    // Replacing **bold** and _italic_ with the inner contents group 1 or 2
    let text = re_bold_em.replace_all(&text, |caps: &Captures| {
        if let Some(m) = caps.get(1) {
            m.as_str().to_string()
        } else if let Some(m) = caps.get(2) {
            m.as_str().to_string()
        } else {
            "".to_string()
        }
    });

    let text = re_footnote.replace_all(&text, "");

    // Smart quotes & other characters mapping
    let mut cleaned = String::with_capacity(text.len());
    for c in text.chars() {
        match c {
            '\u{201c}' | '\u{201d}' => cleaning_push(&mut cleaned, '"'),
            '\u{2018}' | '\u{2019}' => cleaning_push(&mut cleaned, '\''),
            '\u{2014}' => cleaned.push_str(", "),
            '\u{2013}' => cleaning_push(&mut cleaned, '-'),
            '\u{00a0}' => cleaning_push(&mut cleaned, ' '),
            other => cleaning_push(&mut cleaned, other),
        }
    }

    let text = re_multi_bl.replace_all(&cleaned, "\n\n");
    text.trim().to_string()
}

#[inline]
fn cleaning_push(s: &mut String, c: char) {
    s.push(c);
}
