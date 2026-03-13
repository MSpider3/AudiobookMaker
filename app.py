"""
app.py  —  AudiobookMaker Gradio UI
=====================================
Run:  python app.py
Opens: http://localhost:7860
"""
from __future__ import annotations

import io
import os
import queue
import sys
import threading
import time
import zipfile
from pathlib import Path

import gradio as gr
import numpy as np
import soundfile as sf

# ── project root on sys.path ──────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from audiobook_factory.text_extractor import (
    scan, extract,
    ScanResult, ScannedChapter, ExtractedChapter,
)
from audiobook_factory.voice_preprocessor import (
    PreprocessConfig, preprocess as voice_preprocess,
)
from audiobook_factory.pipeline import (
    AudiobookConfig, CancelToken, run_pipeline, preview_tts,
)

# ══════════════════════════════════════════════════════════════════════════════
# Shared state (per-session in Gradio, so we use gr.State)
# ══════════════════════════════════════════════════════════════════════════════

_OUTPUT_DIR = os.path.join(_ROOT, "audiobook_output")
os.makedirs(_OUTPUT_DIR, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
# Helper utilities
# ══════════════════════════════════════════════════════════════════════════════

def _bytes_to_gradio_audio(wav_bytes: bytes):
    """Convert raw WAV bytes to (sample_rate, numpy_array) for gr.Audio."""
    audio, sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
    return sr, audio


def _make_zip(files: list[str]) -> str:
    zip_path = os.path.join(_OUTPUT_DIR, "AudiobookMaker_output.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f in files:
            if os.path.exists(f):
                zf.write(f, os.path.basename(f))
    return zip_path


# ══════════════════════════════════════════════════════════════════════════════
# Gradio App
# ══════════════════════════════════════════════════════════════════════════════

def build_app():
    with gr.Blocks(title="AudiobookMaker") as demo:

        # ── Header ────────────────────────────────────────────────────────────
        gr.HTML("""
        <div class="header-banner">
          <h1>📖 AudiobookMaker</h1>
          <p>Convert any book into a high-quality narrated audiobook using AI voice cloning.</p>
        </div>
        """)

        # ── Session state ─────────────────────────────────────────────────────
        scan_state      = gr.State(None)     # ScanResult
        chapters_state  = gr.State([])       # list[ExtractedChapter]
        config_state    = gr.State(None)     # AudiobookConfig
        outputs_state   = gr.State([])       # list[str] output file paths
        cancel_state    = gr.State(None)     # CancelToken
        preproc_state   = gr.State(None)     # processed voice bytes

        with gr.Tabs() as tabs:

            # ═══════════════════════════════════════════════════════════════ #
            # TAB 1 — BOOK                                                    #
            # ═══════════════════════════════════════════════════════════════ #
            with gr.Tab("📚 Book"):
                gr.Markdown("### Step 1 — Upload your book")

                with gr.Row():
                    book_file = gr.File(
                        label="Book file",
                        file_types=[".epub", ".mobi", ".pdf", ".docx", ".odt", ".txt"],
                    )
                    with gr.Column():
                        scan_status = gr.Markdown("*Upload a file to begin.*")
                        book_title_box  = gr.Textbox(label="Book title", interactive=True)
                        book_author_box = gr.Textbox(label="Author", interactive=True)

                # ── EPUB / MOBI: chapter checklist ───────────────────────────
                with gr.Group(visible=False) as epub_panel:
                    gr.Markdown("### Step 2 — Select chapters to convert")
                    with gr.Row():
                        select_all_btn   = gr.Button("Select All",   size="sm", variant="secondary")
                        deselect_all_btn = gr.Button("Deselect All", size="sm", variant="secondary")
                    chapter_check = gr.CheckboxGroup(
                        label="Chapters",
                        choices=[],
                        value=[],
                        interactive=True,
                    )

                # ── PDF/DOCX/ODT/MOBI-fallback: page ranges ─────────────────
                with gr.Group(visible=False) as page_panel:
                    gr.Markdown("### Step 2 — Split by page ranges")
                    gr.HTML('<div class="warn-box">⚠️ No chapter structure detected in this file. Enter page ranges below — each range becomes a separate chapter file.</div>')
                    page_ranges_box = gr.Textbox(
                        label='Page ranges (e.g. "1-50, 51-120, 121-250")',
                        placeholder="1-50, 51-120",
                        lines=2,
                    )
                    total_pages_label = gr.Markdown("")

                # ── TXT: whole book notice ────────────────────────────────────
                with gr.Group(visible=False) as txt_panel:
                    gr.HTML('<div class="warn-box">ℹ️ TXT files have no page structure — the entire file will be converted as a single audio file.</div>')

                # ── Metadata & output settings ───────────────────────────────
                gr.Markdown("### Step 3 — Output settings")
                with gr.Row():
                    output_format = gr.Dropdown(
                        label="Output format",
                        choices=["mp3", "flac", "wav", "m4b"],
                        value="mp3",
                    )
                    lufs_slider = gr.Slider(
                        label="Loudness target (LUFS)",
                        minimum=-24, maximum=-14, value=-18, step=1,
                    )
                with gr.Row():
                    cover_image = gr.Image(
                        label="Cover image (optional)", type="filepath", height=160
                    )

            # ═══════════════════════════════════════════════════════════════ #
            # TAB 2 — VOICE PREPROCESSING                                     #
            # ═══════════════════════════════════════════════════════════════ #
            with gr.Tab("🎧 Voice Preprocessing"):
                gr.Markdown(
                    "### Clean your raw voice recording before cloning\n"
                    "Each step is independently toggleable."
                )

                with gr.Row():
                    voice_raw_upload = gr.Audio(
                        label="Upload raw voice WAV",
                        type="filepath",
                        sources=["upload"],
                    )
                    voice_processed_player = gr.Audio(
                        label="Processed preview",
                        type="numpy",
                        interactive=False,
                    )

                with gr.Accordion("🔇 Step 1 — Noise Reduction", open=True):
                    with gr.Row():
                        pp_noise_reduce  = gr.Checkbox(label="Enable", value=True)
                        pp_noise_strength = gr.Slider(label="Strength", minimum=0.0, maximum=1.0, value=0.5, step=0.05)

                with gr.Accordion("🔈 Step 2 — Noise Gate", open=False):
                    with gr.Row():
                        pp_gate       = gr.Checkbox(label="Enable", value=True)
                        pp_gate_db    = gr.Slider(label="Threshold (dB)", minimum=-60, maximum=0, value=-40, step=1)

                with gr.Accordion("📡 Step 3 — High-Pass Filter", open=False):
                    with gr.Row():
                        pp_hp         = gr.Checkbox(label="Enable", value=True)
                        pp_hp_hz      = gr.Slider(label="Cutoff (Hz)", minimum=40, maximum=400, value=80, step=10)

                with gr.Accordion("✂️ Step 4 — Silence Removal", open=False):
                    with gr.Row():
                        pp_sil        = gr.Checkbox(label="Enable", value=True)
                        pp_sil_db     = gr.Slider(label="Silence threshold (dB)", minimum=-60, maximum=-10, value=-40, step=1)
                    with gr.Row():
                        pp_sil_min    = gr.Slider(label="Min segment (ms)", minimum=50, maximum=2000, value=300, step=50)
                        pp_sil_keep   = gr.Slider(label="Max silence kept (ms)", minimum=0, maximum=2000, value=500, step=50)

                with gr.Accordion("🔊 Step 5 — Normalize Volume", open=False):
                    with gr.Row():
                        pp_norm       = gr.Checkbox(label="Enable", value=True)
                        pp_norm_db    = gr.Slider(label="Target dBFS", minimum=-12, maximum=-1, value=-3, step=0.5)

                with gr.Accordion("🎵 Step 6 — Formant Shift (experimental)", open=False):
                    with gr.Row():
                        pp_formant    = gr.Checkbox(label="Enable", value=False)
                        pp_quefrency  = gr.Slider(label="Quefrency (gender/timbre)", minimum=0.0, maximum=16.0, value=1.0, step=0.1)
                        pp_timbre     = gr.Slider(label="Timbre", minimum=0.0, maximum=16.0, value=1.0, step=0.1)

                with gr.Accordion("🔁 Step 7 — Resample", open=False):
                    with gr.Row():
                        pp_resample   = gr.Checkbox(label="Enable", value=False)
                        pp_target_sr  = gr.Dropdown(
                            label="Target sample rate",
                            choices=[22050, 44100, 48000],
                            value=44100,
                        )

                with gr.Row():
                    preprocess_btn   = gr.Button("▶ Preview Processed Audio", variant="primary")
                    save_voice_btn   = gr.Button("💾 Use as narrator voice", variant="secondary")

                preprocess_status = gr.Markdown("")

            # ═══════════════════════════════════════════════════════════════ #
            # TAB 3 — VOICE STUDIO                                            #
            # ═══════════════════════════════════════════════════════════════ #
            with gr.Tab("🎙️ Voice Studio"):
                gr.Markdown("### Configure TTS voice and test it")

                with gr.Row():
                    voice_studio_upload = gr.Audio(
                        label="Narrator voice (processed WAV)",
                        type="filepath",
                        sources=["upload"],
                    )
                    voice_status_md = gr.Markdown("*Upload or carry voice from preprocessing.*")

                gr.Markdown("#### Tuning parameters")
                with gr.Row():
                    speed_slider   = gr.Slider(label="Speaking speed", minimum=0.5, maximum=2.0, value=1.0, step=0.05)
                    temp_slider    = gr.Slider(label="Temperature",     minimum=0.1, maximum=1.0, value=0.3, step=0.05)
                    topp_slider    = gr.Slider(label="Top-p",           minimum=0.5, maximum=1.0, value=0.8, step=0.05)

                with gr.Row():
                    sent_pause_sl  = gr.Slider(label="Sentence pause (s)", minimum=0.2, maximum=2.0, value=0.5, step=0.1)
                    para_pause_sl  = gr.Slider(label="Paragraph pause (s)", minimum=0.5, maximum=3.0, value=1.2, step=0.1)

                gr.Markdown("#### Voice test")
                with gr.Row():
                    test_text  = gr.Textbox(
                        label="Test sentence",
                        value="In the beginning, there was only darkness — and then, a single flame.",
                        lines=2,
                    )
                    test_btn   = gr.Button("▶ Test Voice", variant="primary", scale=0)

                test_audio = gr.Audio(label="Preview", type="numpy", interactive=False)
                test_status = gr.Markdown("")

            # ═══════════════════════════════════════════════════════════════ #
            # TAB 4 — ADVANCED                                                #
            # ═══════════════════════════════════════════════════════════════ #
            with gr.Tab("⚙️ Advanced"):
                gr.Markdown("### Advanced Generation Settings")
                with gr.Row():
                    max_len_sl     = gr.Slider(label="Max chunk length (chars)", minimum=100, maximum=600, value=399, step=1)
                    lufs_adv       = gr.Slider(label="True peak (dBTP)",         minimum=-6,   maximum=-0.5, value=-1.5, step=0.1)
                with gr.Row():
                    epub_ocr_chk   = gr.Checkbox(label="Enable EasyOCR for EPUB image text", value=False)
                    force_repro_chk = gr.Checkbox(label="Force re-process (ignore saved progress)", value=False)

            # ═══════════════════════════════════════════════════════════════ #
            # TAB 5 — GENERATE                                                #
            # ═══════════════════════════════════════════════════════════════ #
            with gr.Tab("🚀 Generate"):
                gr.Markdown("### Generate Audiobook")

                with gr.Row():
                    generate_btn = gr.Button("🎧 Generate Audiobook", variant="primary", scale=3)
                    cancel_btn   = gr.Button("⛔ Cancel",              variant="stop",    scale=1)

                progress_bar = gr.Progress(track_tqdm=False)
                log_box      = gr.Textbox(
                    label="Generation log",
                    lines=20,
                    interactive=False,
                    max_lines=200,
                )

                gr.Markdown("#### Download outputs")
                download_col   = gr.Column(visible=False)
                with download_col:
                    download_files = gr.File(
                        label="Chapter files",
                        file_count="multiple",
                        interactive=False,
                    )
                    zip_btn        = gr.Button("⬇ Download All (ZIP)", variant="secondary")
                    zip_file       = gr.File(label="ZIP", interactive=False, visible=False)

        # ══════════════════════════════════════════════════════════════════
        # EVENT HANDLERS
        # ══════════════════════════════════════════════════════════════════

        # ── Book upload ───────────────────────────────────────────────────
        def on_book_upload(file_obj):
            if file_obj is None:
                return (
                    gr.update(visible=False),  # epub_panel
                    gr.update(visible=False),  # page_panel
                    gr.update(visible=False),  # txt_panel
                    "*Upload a file to begin.*",
                    gr.update(choices=[], value=[]),
                    "",
                    "",
                    None,
                    "",
                )

            path = file_obj.name if hasattr(file_obj, "name") else str(file_obj)
            result: ScanResult = scan(path)

            status_parts = [f"**Type:** `{result.file_type.upper()}`"]
            if result.title:      status_parts.append(f"**Title:** {result.title}")
            if result.author:     status_parts.append(f"**Author:** {result.author}")
            if result.page_count: status_parts.append(f"**Pages:** {result.page_count}")

            title  = result.title  or ""
            author = result.author or ""

            if result.has_toc and result.chapters:
                choices = [f"{c.num}. {c.title}  (~{c.word_count:,} words)" for c in result.chapters]
                status_parts.append(f"✅ **{len(result.chapters)} chapters found.**")
                return (
                    gr.update(visible=True),   # epub_panel
                    gr.update(visible=False),  # page_panel
                    gr.update(visible=False),  # txt_panel
                    "\n\n".join(status_parts),
                    gr.update(choices=choices, value=choices),
                    title, author,
                    result,
                    "",
                )
            elif result.file_type == "txt":
                status_parts.append("ℹ️ No chapters — whole book mode.")
                return (
                    gr.update(visible=False),  # epub_panel
                    gr.update(visible=False),  # page_panel
                    gr.update(visible=True),   # txt_panel
                    "\n\n".join(status_parts),
                    gr.update(choices=[], value=[]),
                    title, author,
                    result,
                    "",
                )
            else:
                pages_msg = f"**Total pages:** {result.page_count}" if result.page_count else ""
                status_parts.append(f"⚠️ No chapter TOC detected — page-range mode.")
                return (
                    gr.update(visible=False),  # epub_panel
                    gr.update(visible=True),   # page_panel
                    gr.update(visible=False),  # txt_panel
                    "\n\n".join(status_parts),
                    gr.update(choices=[], value=[]),
                    title, author,
                    result,
                    pages_msg,
                )

        book_file.upload(
            on_book_upload,
            inputs=[book_file],
            outputs=[epub_panel, page_panel, txt_panel, scan_status,
                     chapter_check, book_title_box, book_author_box,
                     scan_state, total_pages_label],
        )

        # ── Select/Deselect All ───────────────────────────────────────────
        select_all_btn.click(
            fn=lambda choices: gr.update(value=choices),
            inputs=[chapter_check],
            outputs=[chapter_check],
        )
        deselect_all_btn.click(
            fn=lambda: gr.update(value=[]),
            outputs=[chapter_check],
        )

        # ── Voice Preprocessing ───────────────────────────────────────────
        def run_preprocess(
            raw_audio_path,
            noise_reduce, noise_strength,
            gate, gate_db,
            hp, hp_hz,
            sil, sil_db, sil_min, sil_keep,
            norm, norm_db,
            formant, quefrency, timbre,
            resamp, target_sr,
        ):
            if raw_audio_path is None:
                return None, "⚠️ Please upload a voice WAV first.", None

            cfg = PreprocessConfig(
                noise_reduce=noise_reduce,       noise_reduce_strength=noise_strength,
                noise_gate=gate,                 noise_gate_threshold_db=gate_db,
                highpass_filter=hp,              highpass_cutoff_hz=hp_hz,
                silence_removal=sil,             silence_threshold_db=sil_db,
                min_segment_ms=int(sil_min),     max_silence_kept_ms=int(sil_keep),
                normalize_volume=norm,           normalize_target_dbfs=norm_db,
                formant_shift=formant,           formant_quefrency=quefrency,
                formant_timbre=timbre,
                resample=resamp,                 target_sample_rate=int(target_sr),
            )

            with open(raw_audio_path, "rb") as f:
                in_bytes = f.read()

            logs = []
            out_bytes = voice_preprocess(in_bytes, cfg, log_fn=logs.append)
            sr, audio = _bytes_to_gradio_audio(out_bytes)
            return (sr, audio), "✅ Preprocessing complete!", out_bytes

        preprocess_btn.click(
            run_preprocess,
            inputs=[
                voice_raw_upload,
                pp_noise_reduce, pp_noise_strength,
                pp_gate, pp_gate_db,
                pp_hp, pp_hp_hz,
                pp_sil, pp_sil_db, pp_sil_min, pp_sil_keep,
                pp_norm, pp_norm_db,
                pp_formant, pp_quefrency, pp_timbre,
                pp_resample, pp_target_sr,
            ],
            outputs=[voice_processed_player, preprocess_status, preproc_state],
        )

        # ── Save processed voice → Voice Studio ──────────────────────────────
        def save_processed_voice(wav_bytes):
            if wav_bytes is None:
                return "⚠️ Run preprocessing first.", None
            save_path = os.path.join(_OUTPUT_DIR, "narrator_voice_processed.wav")
            with open(save_path, "wb") as f:
                f.write(wav_bytes)
            return "✅ Voice saved! You can also load it in Voice Studio →", save_path

        save_voice_btn.click(
            save_processed_voice,
            inputs=[preproc_state],
            outputs=[preprocess_status, voice_studio_upload],
        )

        # ── Voice Studio: Test Voice ──────────────────────────────────────────
        def on_test_voice(
            voice_path, text, temp, top_p, speed,
        ):
            if not voice_path:
                return None, "⚠️ Upload or set a narrator voice first."
            if not text.strip():
                return None, "⚠️ Enter some text to test."

            cfg = AudiobookConfig(
                voice_file=voice_path,
                temperature=temp,
                top_p=top_p,
            )
            wav_bytes = preview_tts(text, cfg)
            if wav_bytes is None:
                return None, "❌ TTS generation failed — check your voice file and TTS model."
            sr, audio = _bytes_to_gradio_audio(wav_bytes)
            return (sr, audio), "✅ Preview ready!"

        test_btn.click(
            on_test_voice,
            inputs=[voice_studio_upload, test_text, temp_slider, topp_slider, speed_slider],
            outputs=[test_audio, test_status],
        )

        # ── Generate Audiobook ────────────────────────────────────────────────
        def on_generate(
            scan_res, file_obj,
            selected_chapters, page_ranges_str,
            book_title, author,
            voice_path, output_fmt, lufs,
            temp, top_p, pause, para_pause,
            max_len, true_peak,
            epub_ocr, force_repro,
        ):
            if file_obj is None:
                yield "⚠️ Please upload a book file first.", gr.update(visible=False), []
                return
            if not voice_path:
                yield "⚠️ Please set a narrator voice in the Voice Studio tab.", gr.update(visible=False), []
                return

            path = file_obj.name if hasattr(file_obj, "name") else str(file_obj)

            # Build selections
            selections = None
            page_ranges = None

            if scan_res and scan_res.has_toc and selected_chapters:
                # Parse chapter numbers from checkbox labels ("1. Title (~N words)")
                selections = [int(label.split(".")[0]) for label in selected_chapters]
            elif scan_res and not scan_res.has_toc and scan_res.file_type != "txt":
                # Parse page ranges "1-50, 51-120"
                page_ranges = []
                for part in page_ranges_str.split(","):
                    part = part.strip()
                    if "-" in part:
                        try:
                            s, e = part.split("-")
                            page_ranges.append((int(s.strip()), int(e.strip())))
                        except ValueError:
                            pass
                if not page_ranges:
                    page_ranges = None  # fall back to whole-book

            # Config
            book_out = os.path.join(
                _OUTPUT_DIR,
                re.sub(r'[\\/*?:"<>|]', "", book_title or "audiobook"),
            )
            os.makedirs(book_out, exist_ok=True)

            cfg = AudiobookConfig(
                book_title=book_title,
                author=author,
                output_dir=book_out,
                output_format=output_fmt,
                voice_file=voice_path,
                temperature=temp,
                top_p=top_p,
                pause=pause,
                para_pause=para_pause,
                max_len=int(max_len),
                lufs=int(lufs),
                true_peak=true_peak,
                force_reprocess=force_repro,
            )

            log_q  = queue.Queue()
            prog_q = queue.Queue()
            cancel = CancelToken()

            def _runner():
                # Extract
                chapters = extract(
                    path,
                    selections=selections,
                    enable_ocr=epub_ocr,
                    page_ranges=page_ranges,
                    log_fn=log_q.put,
                )
                if chapters:
                    # Generate
                    out_files = run_pipeline(cfg, chapters, log_q, prog_q, cancel)
                    log_q.put(f"__DONE__::{','.join(out_files)}")
                else:
                    log_q.put("__DONE__::")

            t = threading.Thread(target=_runner, daemon=True)
            t.start()

            log_text = ""
            out_files   = []

            while t.is_alive() or not log_q.empty():
                try:
                    msg = log_q.get(timeout=0.2)
                    if msg.startswith("__DONE__::"):
                        paths = msg.split("::", 1)[1]
                        out_files = [p for p in paths.split(",") if p and os.path.exists(p)]
                        break
                    log_text += msg + "\n"
                    yield log_text, gr.update(visible=False), []
                except queue.Empty:
                    yield log_text, gr.update(visible=False), []

            if out_files:
                yield (
                    log_text + "\n✅ Generation complete!",
                    gr.update(visible=True),
                    out_files,
                )
            else:
                yield log_text + "\n⚠️ No output files generated.", gr.update(visible=False), []

        generate_btn.click(
            on_generate,
            inputs=[
                scan_state, book_file,
                chapter_check, page_ranges_box,
                book_title_box, book_author_box,
                voice_studio_upload, output_format, lufs_slider,
                temp_slider, topp_slider, sent_pause_sl, para_pause_sl,
                max_len_sl, lufs_adv,
                epub_ocr_chk, force_repro_chk,
            ],
            outputs=[log_box, download_col, download_files],
        )

        # ── Cancel ────────────────────────────────────────────────────────────
        def on_cancel(cancel_tok):
            if cancel_tok:
                cancel_tok.cancel()
            return "⛔ Cancellation requested..."

        cancel_btn.click(on_cancel, inputs=[cancel_state], outputs=[log_box])

        # ── Download ZIP ──────────────────────────────────────────────────────
        def on_zip(files):
            if not files:
                return gr.update(visible=False)
            z = _make_zip(files if isinstance(files, list) else [files])
            return gr.update(value=z, visible=True)

        zip_btn.click(on_zip, inputs=[download_files], outputs=[zip_file])

    return demo


# ══════════════════════════════════════════════════════════════════════════════
import re  # needed inside on_generate

if __name__ == "__main__":
    _THEME = gr.themes.Soft(
        primary_hue="indigo",
        secondary_hue="purple",
        neutral_hue="slate",
        font=[gr.themes.GoogleFont("Inter"), "sans-serif"],
    )
    _CSS = """
    .header-banner { text-align:center; padding: 20px 0 10px; }
    .header-banner h1 { font-size: 2.4rem; font-weight: 800; letter-spacing: -1px; }
    .header-banner p  { color: #6b7280; font-size: 1rem; }
    .warn-box { background: rgba(234,179,8,0.12); border-radius:8px;
                border:1px solid #ca8a04; padding:12px; font-size:0.9rem; }
    """
    demo = build_app()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        theme=_THEME,
        css=_CSS,
    )
