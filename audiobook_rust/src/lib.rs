use pyo3::prelude::*;

mod text;
mod audio;

use text::normalize::{normalize_text_rust, clean_text_full};
use text::splitter::split_sentences_rust;

/// Exposes the simple normalizer (legacy format).
#[pyfunction]
#[pyo3(name = "normalize_text")]
fn normalize_text(text: &str) -> PyResult<String> {
    Ok(normalize_text_rust(text))
}

/// Exposes the full Page/MD normalizer pipeline (TextNormalizer class level).
#[pyfunction]
#[pyo3(name = "clean_text")]
fn clean_text(raw_md: &str, title: &str, is_pdf: bool) -> PyResult<String> {
    Ok(clean_text_full(raw_md, title, is_pdf))
}

/// Exposes the smart sentence splitter.
#[pyfunction]
#[pyo3(name = "split_sentences")]
fn split_sentences(text: &str, max_len: usize) -> PyResult<Vec<String>> {
    Ok(split_sentences_rust(text, max_len))
}

/// Exposes the audio mastering pipeline.
#[pyfunction]
#[pyo3(name = "master_audio")]
fn master_audio(
    chunk_paths: Vec<String>,
    out_path: String,
    pause_sec: f64,
    sample_rate: u32,
    target_lufs: f64,
    target_tp_db: f64,
    bitrate_kbps: u32,
) -> PyResult<()> {
    audio::master::master_audio_rust(
        chunk_paths,
        out_path,
        pause_sec,
        sample_rate,
        target_lufs,
        target_tp_db,
        bitrate_kbps,
    ).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(e))
}

/// A Python module implemented in Rust.
#[pymodule]
fn audiobook_rust(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(normalize_text, m)?)?;
    m.add_function(wrap_pyfunction!(clean_text, m)?)?;
    m.add_function(wrap_pyfunction!(split_sentences, m)?)?;
    m.add_function(wrap_pyfunction!(master_audio, m)?)?;
    Ok(())
}
