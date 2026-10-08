use regex::{Regex, Captures};
use std::sync::OnceLock;
use std::collections::HashMap;

// Initialize compile-once regular expressions
static RE_DEWRAP: OnceLock<Regex> = OnceLock::new();
static RE_DROPCAP: OnceLock<Regex> = OnceLock::new();
static RE_ISO_CAP: OnceLock<Regex> = OnceLock::new();
static RE_ALL_CAPS: OnceLock<Regex> = OnceLock::new();
static RE_DROPCAP_MIXED: OnceLock<Regex> = OnceLock::new();
static RE_DROPCAP_CAPS: OnceLock<Regex> = OnceLock::new();

/// Words that label a single-letter identifier ("Class D", "Plan B",
/// "Vitamin C", "Mr. T"). A capital following one is a name, not a kerning split.
const LETTER_LABELS: &[&str] = &[
    "class", "room", "section", "level", "floor", "group", "area", "zone",
    "rank", "type", "grade", "exam", "test", "point", "score", "phase",
    "stage", "category", "model", "series", "volume", "chapter", "year",
    "course", "subject", "unit", "part", "item", "step", "plan", "vitamin",
    "option", "team", "block", "wing", "gate", "platform", "appendix",
    "figure", "table", "exhibit", "mr", "mrs", "ms", "dr", "agent",
];
static RE_HYPHEN: OnceLock<Regex> = OnceLock::new();
static RE_SOFT_WRAP: OnceLock<Regex> = OnceLock::new();

fn get_deway_re() -> &'static Regex {
    RE_DEWRAP.get_or_init(|| Regex::new(r"([^.!?;:\x22\n])\n\s*([a-z])").unwrap())
}

fn get_dropcap_re() -> &'static Regex {
    RE_DROPCAP.get_or_init(|| Regex::new(r"(^|\n)([A-Z])\s*\n\s*([A-Z]{2,})").unwrap())
}

fn get_iso_cap_re() -> &'static Regex {
    // Equivalent to (?<![a-zA-Z])([A-Z])[ \t]+([a-zA-Z]+)\b
    // We match a boundary non-alphabet character or start of line
    RE_ISO_CAP.get_or_init(|| Regex::new(r"(^|[^a-zA-Z])([A-Z])[ \t]+([a-zA-Z]+)\b").unwrap())
}

fn get_all_caps_re() -> &'static Regex {
    RE_ALL_CAPS.get_or_init(|| Regex::new(r"\b([A-Z])[ \t]+([A-Z]+)\b").unwrap())
}

fn get_dropcap_mixed_re() -> &'static Regex {
    RE_DROPCAP_MIXED.get_or_init(|| Regex::new(r"(?m)^([ \t]*(?:#+[ \t]*)?)([A-Z])[ \t]+([a-z]+)\b").unwrap())
}

fn get_dropcap_caps_re() -> &'static Regex {
    RE_DROPCAP_CAPS.get_or_init(|| Regex::new(r"(?m)^([ \t]*(?:#+[ \t]*)?)([A-Z])[ \t]+([A-Z]{2,})\b").unwrap())
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

    text.into_owned()
}

/// Joins a detached capital to the rest of its word, unless "A"/"I" is a real word here.
fn merge_capital(cap: &str, rest: &str) -> String {
    if cap == "A" || cap == "I" {
        let is_upper = rest.chars().any(|c| c.is_alphabetic()) && !rest.chars().any(|c| c.is_lowercase());
        if is_upper {
            if cap == "A" && matches!(rest, "ND" | "S" | "T" | "RE" | "N" | "LL" | "NY") {
                return format!("{}{}", cap, rest);
            }
            if cap == "I" && matches!(rest, "T" | "S" | "F" | "N") {
                return format!("{}{}", cap, rest);
            }
            return format!("{} {}", cap, rest);
        }
        let rest_lower = rest.to_lowercase();
        if cap == "A" && matches!(rest_lower.as_str(), "nd" | "s" | "t" | "re" | "n" | "ll" | "ny" | "lthough" | "gain" | "nother" | "lready" | "lways") {
            return format!("{}{}", cap, rest);
        }
        if cap == "I" && matches!(rest_lower.as_str(), "t" | "s" | "f" | "n" | "ll" | "nto" | "ndeed" | "tself") {
            return format!("{}{}", cap, rest);
        }
        return format!("{} {}", cap, rest);
    }
    format!("{}{}", cap, rest)
}

/// True when the text right before `pos` ends in a label word ("Class ", "Mr. ").
fn preceded_by_label(text: &str, pos: usize) -> bool {
    let before = text[..pos].trim_end_matches(|c| c == ' ' || c == '\t');
    if before.len() == text[..pos].len() {
        return false; // no whitespace between the label and the capital
    }
    let before = before.strip_suffix('.').unwrap_or(before);
    let word_start = before
        .char_indices()
        .rev()
        .take_while(|(_, c)| c.is_ascii_alphabetic())
        .last()
        .map(|(i, _)| i);
    match word_start {
        Some(start) => LETTER_LABELS.contains(&before[start..].to_lowercase().as_str()),
        None => false,
    }
}

/// Re-joins a single capital letter that extraction detached from its word.
///
/// By default only drop-cap position is repaired (a lone capital opening a
/// line). Merging everywhere corrupts ordinary prose ("Vitamin C is" ->
/// "Vitamin Cis"), so that is reserved for `aggressive`, used for PDF text
/// where kerning splits words mid-line. Never joins across a line break.
fn fix_isolated_capitals(text: &str, aggressive: bool) -> String {
    if !aggressive {
        let text = get_dropcap_mixed_re().replace_all(text, |caps: &Captures| {
            format!("{}{}", &caps[1], merge_capital(&caps[2], &caps[3]))
        });
        let text = get_dropcap_caps_re().replace_all(&text, |caps: &Captures| {
            format!("{}{}", &caps[1], merge_capital(&caps[2], &caps[3]))
        });
        return text.into_owned();
    }

    let text = get_iso_cap_re().replace_all(text, |caps: &Captures| {
        let cap = caps.get(2).unwrap();
        if preceded_by_label(text, cap.start()) {
            return caps[0].to_string();
        }
        format!("{}{}", &caps[1], merge_capital(cap.as_str(), &caps[3]))
    });
    let text = text.into_owned();

    let merged = get_all_caps_re().replace_all(&text, |caps: &Captures| {
        let cap = caps.get(1).unwrap();
        if preceded_by_label(&text, cap.start()) {
            return caps[0].to_string();
        }
        merge_capital(cap.as_str(), &caps[2])
    });
    merged.into_owned()
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
    text = fix_isolated_capitals(&text, is_pdf);

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
    if stripped_title.is_empty() {
        // An empty title would "match" a blank line and glue paragraphs together.
        return text.to_string();
    }

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
    // Horizontal rules and scene breaks, contiguous or spaced: ---, ***, * * *
    let re_hr = Regex::new(r"(?m)^[ \t]*(?:[-*_~#\u{2022}\u{00b7}][ \t]*){3,}$").unwrap();
    let re_heading = Regex::new(r"(?m)^[ \t]*#{1,6}[ \t]+").unwrap();
    let re_html_comment = Regex::new(r"(?s)<!--.*?-->").unwrap();
    let re_bold_em = Regex::new(r"\*{1,2}([^*]+?)\*{1,2}|_{1,2}([^_]+?)_{1,2}").unwrap();
    let re_footnote = Regex::new(r"\[\[\d+\]\]\([^)]+\)|\[\d+\]\([^)]+\)").unwrap();
    let re_multi_bl = Regex::new(r"\n{3,}").unwrap();

    let text = re_ocr_prefix.replace_all(text, "");
    let text = re_img_tag.replace_all(&text, "");
    let text = re_html_comment.replace_all(&text, "");
    let text = re_hr.replace_all(&text, "\n\n");
    let text = re_heading.replace_all(&text, "");
    let text = text.replace("\\_", " ");

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

    // Docling escapes these when exporting markdown; "&amp;" must come last.
    let mut text = text.into_owned();
    for (entity, ch) in [
        ("&lt;", "<"), ("&gt;", ">"), ("&quot;", "\""), ("&#x27;", "'"),
        ("&#39;", "'"), ("&apos;", "'"), ("&nbsp;", " "), ("&amp;", "&"),
    ] {
        text = text.replace(entity, ch);
    }

    // A dash that opens a line of dialogue (French, Russian, Spanish ...)
    // is not a pause inside a sentence: drop it instead of reading ", ".
    let re_dialogue_dash = Regex::new("(?m)(^[ \\t]*|[.!?\u{2026}\u{00bb}\"\u{201d},;:][ \\t\u{00a0}]+)\u{2014}[ \\t\u{00a0}]*").unwrap();
    let text = re_dialogue_dash.replace_all(&text, "$1").into_owned();

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
