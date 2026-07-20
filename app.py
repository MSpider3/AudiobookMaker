"""
app.py  —  AudiobookMaker Gradio UI
=====================================
Run:  python app.py
Opens: http://localhost:7860
"""
from __future__ import annotations

import dataclasses
import io
import json
import os
import queue
import re
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
    AudiobookConfig, CancelToken, run_pipeline, preview_tts, preview_chapters,
)

# ══════════════════════════════════════════════════════════════════════════════
# FastAPI orchestrator healthcheck helper
# ══════════════════════════════════════════════════════════════════════════════
def is_api_healthy() -> bool:
    import requests
    try:
        r = requests.get("http://127.0.0.1:8000/api/v1/health", timeout=1.0)
        return r.status_code == 200 and r.json().get("status") == "ok"
    except Exception:
        return False


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


def check_existing_progress(book_title):
    if not book_title or not book_title.strip():
        return ""
    import json
    import re
    # We must sanitize the book title just like in on_generate to get the output dir
    sanitized_book_title = re.sub(r'[\\/*?:"<>|]', "", book_title)
    prog_path = os.path.join(_OUTPUT_DIR, sanitized_book_title, "generation_progress.json")
    if os.path.exists(prog_path):
        try:
            with open(prog_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            chapters = data.get("chapters", [])
            completed = sum(1 for c in chapters if c.get("status") in ("completed", "complete"))
            total = len(chapters)
            return (
                f"### 🔄 Existing Progress Found!\n"
                f"- **Book Title:** {data.get('book_title', book_title)}\n"
                f"- **Progress:** {completed} / {total} chapters generated ({float(completed)/total*100:.1f}%)\n"
                f"Generation will automatically resume from the last completed chapter."
            )
        except Exception as e:
            return f"⚠️ Found progress file but failed to read it: {e}"
    return ""


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
        all_chapter_choices_state = gr.State([]) # full list of chapter labels
        # Saved chapter selections from an uploaded progress JSON.
        # on_book_upload reads this and selects only the matching chapters.
        json_selected_chapters_state = gr.State(None)  # list[str] | None

        # ── Resume from Progress JSON (top-level, before tabs) ────────────────
        with gr.Accordion("🔄 Resume from Progress JSON", open=False):
            gr.HTML('<div class="warn-box">💡 <strong>Start here if you are resuming.</strong> Upload your <code>generation_progress.json</code> to restore all settings and chapter selections automatically. Then upload your book file (in the Book tab) — only needed for cover-image extraction when text is already cached in the JSON.</div>')
            with gr.Row():
                progress_file_upload = gr.File(
                    label="Upload existing generation_progress.json",
                    file_types=[".json"],
                    file_count="single",
                )
            progress_upload_status = gr.Markdown("")

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
                        book_language_dd = gr.Dropdown(
                            label="Language (Supported by current TTS)",
                            choices=["English", "Chinese", "Japanese", "Korean", "French", "Spanish", "Italian", "German"],
                            value="English",
                            interactive=True,
                            info="Select the language of your book to match the TTS engine."
                        )
                        book_title_box  = gr.Textbox(label="Book title", interactive=True)
                        book_author_box = gr.Textbox(label="Author", interactive=True)
                        existing_progress_info = gr.Markdown("")

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

                gr.Markdown("#### Qwen3 Model Configuration")
                with gr.Row():
                    tts_model_name = gr.Dropdown(
                        label="TTS Model Variant",
                        choices=[
                            "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign",
                            "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice",
                            "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                            "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice",
                            "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
                        ],
                        value="Qwen/Qwen3-TTS-12Hz-1.7B-Base",
                    )
                    tts_timbre = gr.Dropdown(
                        label="Premium Timbre (CustomVoice only)",
                        choices=[
                            "[Chinese] vivian",
                            "[Chinese] serena",
                            "[Chinese] uncle_fu",
                            "[Chinese (Beijing Dialect)] dylan",
                            "[Chinese (Sichuan Dialect)] eric",
                            "[English] ryan",
                            "[English] aiden",
                            "[Japanese] ono_anna",
                            "[Korean] sohee"
                        ],
                        value="[English] ryan",
                        visible=False
                    )
                
                tts_instruct = gr.Textbox(
                    label="Voice Design / Style Instruction",
                    placeholder="e.g. 'A mature male narrator with a deep resonant voice' or 'Energetic and youthful'",
                    visible=False,
                    lines=2
                )

                def on_model_change(mname):
                    # Show/hide relevant options
                    show_clone = "Base" in mname
                    show_timbre = "CustomVoice" in mname
                    show_instruct = "VoiceDesign" in mname or "1.7B-CustomVoice" in mname
                    return [
                        gr.update(visible=show_clone),
                        gr.update(visible=show_timbre),
                        gr.update(visible=show_instruct)
                    ]

                tts_model_name.change(
                    on_model_change,
                    inputs=[tts_model_name],
                    outputs=[voice_studio_upload, tts_timbre, tts_instruct]
                )

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
                    max_len_sl      = gr.Slider(label="Max chunk length (chars)",  minimum=100, maximum=600, value=399, step=1)
                    lufs_adv        = gr.Slider(label="True peak (dBTP)",           minimum=-6,   maximum=-0.5, value=-1.5, step=0.1)
                with gr.Row():
                    worker_count_sl = gr.Slider(
                        label="Parallel workers count",
                        info="Enables batch TTS mode. Higher = larger GPU batches = faster (needs more VRAM). Minimum 2 to enable.",
                        minimum=1, maximum=8, value=2, step=1,
                    )
                    parallel_mode_dd = gr.Dropdown(
                        label="Parallelism level",
                        choices=["chapters", "chunks"],
                        value="chunks",
                        info="chapters: process multiple chapters at once (high VRAM). chunks: process chunks of the same chapter at once (recommended).",
                        interactive=True,
                    )
                with gr.Row():
                    tts_provider_dd = gr.Dropdown(
                        label="TTS Provider",
                        choices=["qwen"],
                        value="qwen",
                        info="Currently: Qwen3-TTS (local). More providers coming soon.",
                        interactive=True,
                    )
                with gr.Row():
                    epub_ocr_chk      = gr.Checkbox(label="Enable EasyOCR for EPUB image text", value=False)
                    force_repro_chk   = gr.Checkbox(label="Force re-process (ignore saved progress)", value=False)
                    export_text_chk   = gr.Checkbox(label="Export chapter text as .txt", value=False)
                    torch_compile_chk = gr.Checkbox(label="Enable GPU compile (torch.compile)", value=False)
                gr.Markdown("#### Pronunciation fixes")
                gr.HTML('<div class="warn-box">ℹ️ Upload a plain text file with one fix per line: <code>search==replace</code> (regex supported). Lines starting with # are comments.</div>')
                pronunciation_file = gr.File(
                    label="Pronunciation fix file (.txt)",
                    file_types=[".txt"],
                    file_count="single",
                )
                gr.Markdown("#### Resume / Sync Progress")
                gr.HTML('<div class="warn-box">ℹ️ The progress JSON upload has moved to the top of the page (above all tabs) for a faster resume workflow.</div>')

            # ═══════════════════════════════════════════════════════════════ #
            # TAB 5 — GENERATE                                                #
            # ═══════════════════════════════════════════════════════════════ #
            with gr.Tab("🚀 Generate"):
                gr.Markdown("### Generate Audiobook")

                with gr.Row():
                    preview_btn     = gr.Button("🔍 Preview Chapters",   variant="secondary", scale=1)
                    generate_btn    = gr.Button("🎧 Generate Audiobook", variant="primary",   scale=3)
                    export_cfg_btn  = gr.Button("📋 Export Config JSON", variant="secondary", scale=1)
                    cancel_btn      = gr.Button("⛔ Cancel",              variant="stop",      scale=1)

                with gr.Row():
                    single_file_mode = gr.Checkbox(label="📦 Combine into single file", value=False)
                    export_lrc_chk   = gr.Checkbox(label="📜 Generate timed LRC lyrics", value=True)
                    export_srt_chk   = gr.Checkbox(label="🎬 Generate SRT subtitles", value=False)
                    export_vtt_chk   = gr.Checkbox(label="WebVTT Generate WebVTT subtitles", value=False)

                with gr.Row():
                    regen_missing_chk = gr.Checkbox(
                        label="**🔄 Re-generate completed chapters whose audio file is missing**",
                        value=True,
                        info="When ON (default): if a chapter is marked 'completed' in the progress JSON but the audio file is gone, it will be re-generated automatically. Turn OFF to skip those chapters entirely."
                    )

                preview_table = gr.Dataframe(
                    headers=["#", "Chapter", "Chars", "Words", "Sentences"],
                    datatype=["number", "str", "number", "number", "number"],
                    label="Chapter preview",
                    visible=False,
                    interactive=False,
                    wrap=True,
                )

                progress_bar = gr.Progress(track_tqdm=False)
                prog_html = gr.HTML('<div style="text-align: center; margin-bottom: 5px; font-weight: bold;">Generation Progress: 0.00%</div><progress value="0" max="100" style="width:100%; height:25px;"></progress>')
                log_box      = gr.Textbox(
                    label="Generation log",
                    lines=20,
                    interactive=False,
                    max_lines=200,
                )

                with gr.Accordion("📋 Export Config JSON", open=False) as export_cfg_accordion:
                    gr.HTML('<div class="warn-box">ℹ️ Click <strong>Export Config JSON</strong> above to parse the book and save all settings + chapter text into a single JSON file. You can then use this file with the CLI (<code>python cli.py config.json</code>) without needing Gradio.</div>')
                    export_cfg_status  = gr.Markdown("")
                    export_config_file = gr.File(
                        label="Download generation_progress.json",
                        interactive=False,
                        visible=False,
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

                with gr.Accordion("ℹ️ Which format should I choose?", open=False):
                    gr.Markdown("""
| Feature          | FLAC                       | MP3               | WAV                       | M4B        |
| ---------------- | -------------------------- | ----------------- | ------------------------- | ---------- |
| **Cover Art**    | ✅ Yes                      | ✅ Yes             | ⚠️ Limited / inconsistent | ✅ Yes      |
| **Title**        | ✅ Yes                      | ✅ Yes             | ⚠️ Limited                | ✅ Yes      |
| **Artist**       | ✅ Yes                      | ✅ Yes             | ⚠️ Limited                | ✅ Yes      |
| **Album**        | ✅ Yes                      | ✅ Yes             | ⚠️ Limited                | ✅ Yes      |
| **Genre**        | ✅ Yes                      | ✅ Yes             | ⚠️ Limited                | ✅ Yes      |
| **Track Number** | ✅ Yes                      | ✅ Yes             | ⚠️ Limited                | ✅ Yes      |
| **Lyrics**       | ⚠️ Possible (not standard) | ✅ Yes             | ❌ No                      | ⚠️ Limited |
| **Chapters**     | ❌ No                       | ⚠️ Rare / limited | ❌ No                      | ✅ Yes      |

### Recommendation:
- **Single Combined Audiobook**: Choose **M4B**. It supports internal chapters and cover art, making it the industry standard for portable players.
- **Lossless Quality**: Choose **FLAC** (or WAV if you don't care about metadata).
- **Maximum Compatibility**: Choose **MP3**. It works on everything but lacks true chapter support (it will be split into many files).
""")

            # ═══════════════════════════════════════════════════════════════ #
            # TAB 6 — AUDIOBOOKSHELF                                          #
            # ═══════════════════════════════════════════════════════════════ #
            with gr.Tab("📜 Audiobookshelf"):
                gr.Markdown("""
## Setting up Audiobookshelf

[Audiobookshelf](https://www.audiobookshelf.org/) is a self-hosted audiobook and podcast server. Here is how to add your generated book:

1. **Install ABS**: If you haven't, install it via Docker or Windows Installer.
2. **Library Mapping**: Point a 'Library' in ABS to the folder where you save your Audiobooks.
3. **Organize**: AudiobookMaker follows the `{Author}/{Title}/{Chapter}.mp3` standard. 
   - Move the generated folder into your Library folder.
4. **Scan**: Click 'Scan' in the ABS dashboard. It will automatically find the book, the author, the metadata, and the cover art!
5. **Chapters**: If you used **M4B**, the chapters are embedded! If you used **MP3/FLAC**, ABS will use the filenames to determine chapter order.
6. **Lyrics**: The `.lrc` files generated by AudiobookMaker will provide timed lyrics/subtitles in the ABS player!
""")

        # ══════════════════════════════════════════════════════════════════
        # EVENT HANDLERS
        # ══════════════════════════════════════════════════════════════════

        # ── Book upload ───────────────────────────────────────────────────
        def on_book_upload(file_obj, json_sel):
            """json_sel is the value of json_selected_chapters_state — a list of
            raw chapter labels previously loaded from a progress JSON, or None.
            When present, only those chapters are pre-selected in the checklist.
            """
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
                    [],
                    json_sel,  # preserve state unchanged
                )

            path = file_obj.name if hasattr(file_obj, "name") else str(file_obj)
            result: ScanResult = scan(path)

            status_parts = [f"**Type:** `{result.file_type.upper()}`"]
            if result.title:      status_parts.append(f"**Title:** {result.title}")
            if result.author:     status_parts.append(f"**Author:** {result.author}")
            if result.page_count: status_parts.append(f"**Pages:** {result.page_count}")

            title  = result.title  or ""
            author = result.author or ""
            
            # Handle cover extraction
            cover_path = None
            if result.cover_data:
                try:
                    import tempfile
                    from PIL import Image
                    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                        f.write(result.cover_data)
                        cover_path = f.name
                except:
                    pass

            if result.has_toc and result.chapters:
                choices = [f"{c.num}. {c.title}  (~{c.word_count:,} words)" for c in result.chapters]
                status_parts.append(f"✅ **{len(result.chapters)} chapters found.**")
                # ── Apply saved JSON chapter selections if present ──────────────
                # json_sel is passed as an extra input from json_selected_chapters_state.
                # It is a list of raw labels previously saved by on_progress_upload.
                # We filter the choices list to only those that match so the user
                # doesn't have to re-select chapters manually on resume.
                def _match_json_selection(label, saved_labels):
                    """Return True if label matches any entry in saved_labels."""
                    if not saved_labels:
                        return True
                    for sl in saved_labels:
                        if sl == label:
                            return True
                        # Fuzzy: same chapter number prefix
                        try:
                            if int(label.split(".")[0]) == int(sl.split(".")[0]):
                                return True
                        except (ValueError, IndexError):
                            pass
                    return False

                if json_sel:
                    selected_value = [c for c in choices if _match_json_selection(c, json_sel)]
                    # Fallback: if nothing matched (label format changed), select all
                    if not selected_value:
                        selected_value = choices
                    reset_json_sel = None   # consume the saved selection
                else:
                    selected_value = choices  # default: all selected
                    reset_json_sel = None

                return (
                    gr.update(visible=True),   # epub_panel
                    gr.update(visible=False),  # page_panel
                    gr.update(visible=False),  # txt_panel
                    "\n\n".join(status_parts),
                    gr.update(choices=choices, value=selected_value),
                    title, author,
                    result,
                    "",
                    cover_path,
                    choices,
                    reset_json_sel,           # json_selected_chapters_state (consumed)
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
                    cover_path,
                    [],
                    None,  # reset json_selected_chapters_state
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
                    cover_path,
                    [],
                    None,  # reset json_selected_chapters_state
                )

        book_file.upload(
            on_book_upload,
            inputs=[book_file, json_selected_chapters_state],
            outputs=[epub_panel, page_panel, txt_panel, scan_status,
                     chapter_check, book_title_box, book_author_box,
                     scan_state, total_pages_label, cover_image,
                     all_chapter_choices_state, json_selected_chapters_state],
        )

        # ── Select/Deselect All ───────────────────────────────────────────
        select_all_btn.click(
            fn=lambda full_list: gr.update(value=full_list),
            inputs=[all_chapter_choices_state],
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

            out_bytes = None
            if is_api_healthy():
                try:
                    import requests
                    url = "http://127.0.0.1:8000/api/v1/preprocess"
                    with open(raw_audio_path, "rb") as f:
                        files = {"audio_file": (os.path.basename(raw_audio_path), f, "audio/wav")}
                        data = {
                            "noise_reduce": str(noise_reduce).lower(),
                            "noise_reduce_strength": float(noise_strength),
                            "noise_gate": str(gate).lower(),
                            "noise_gate_threshold_db": float(gate_db),
                            "highpass_filter": str(hp).lower(),
                            "highpass_cutoff_hz": float(hp_hz),
                            "silence_removal": str(sil).lower(),
                            "silence_threshold_db": float(sil_db),
                            "min_segment_ms": int(sil_min),
                            "max_silence_kept_ms": int(sil_keep),
                            "normalize_volume": str(norm).lower(),
                            "normalize_target_dbfs": float(norm_db),
                            "formant_shift": str(formant).lower(),
                            "formant_quefrency": float(quefrency),
                            "formant_timbre": float(timbre),
                            "resample": str(resamp).lower(),
                            "target_sample_rate": int(target_sr)
                        }
                        r = requests.post(url, data=data, files=files)
                        r.raise_for_status()
                        out_bytes = r.content
                except Exception as e:
                    print(f"⚠️ [UI Fallback] Preprocessor API failed: {e}. Running locally.")

            if out_bytes is None:
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
        def save_processed_voice(wav_bytes, raw_audio_path):
            if wav_bytes is None:
                return "⚠️ Run preprocessing first.", None
            if not raw_audio_path:
                # Fallback if raw path is missing
                save_path = os.path.join(_OUTPUT_DIR, "narrator_voice_processed.wav")
            else:
                orig_dir = os.path.dirname(raw_audio_path)
                orig_name = os.path.splitext(os.path.basename(raw_audio_path))[0]
                save_path = os.path.join(orig_dir, f"{orig_name}_processed.wav")

            try:
                with open(save_path, "wb") as f:
                    f.write(wav_bytes)
                return f"✅ Voice saved to: {os.path.basename(save_path)} (in original folder)", save_path
            except Exception as e:
                # Fallback to output dir if directory is not writable
                try:
                    orig_name = os.path.splitext(os.path.basename(raw_audio_path))[0] if raw_audio_path else "narrator_voice"
                    fallback_path = os.path.join(_OUTPUT_DIR, f"{orig_name}_processed.wav")
                    with open(fallback_path, "wb") as f:
                        f.write(wav_bytes)
                    return f"✅ Saved to fallback output folder: {os.path.basename(fallback_path)}", fallback_path
                except Exception as ex:
                    return f"❌ Failed to save: {e} | Fallback failed: {ex}", None

        save_voice_btn.click(
            save_processed_voice,
            inputs=[preproc_state, voice_raw_upload],
            outputs=[preprocess_status, voice_studio_upload],
        )

        # ── Voice Studio: Test Voice ──────────────────────────────────────────
        def on_test_voice(
            voice_path, text, temp, top_p, speed,
            mname, timbre, instruct
        ):
            if ("Base" in mname) and not voice_path:
                return None, "⚠️ Upload or set a narrator voice first for Base models."
            if not text.strip():
                return None, "⚠️ Enter some text to test."

            cfg = AudiobookConfig(
                voice_file=voice_path,
                temperature=temp,
                top_p=top_p,
                tts_model_name=mname,
                tts_timbre=timbre.split()[-1] if timbre else "",
                tts_instruct=instruct
            )

            wav_bytes = None
            if is_api_healthy():
                try:
                    import requests
                    import dataclasses
                    url = "http://127.0.0.1:8000/api/v1/voice-test"
                    config_dict = dataclasses.asdict(cfg)
                    payload = {"config": config_dict, "text": text}
                    r = requests.post(url, json=payload)
                    r.raise_for_status()
                    wav_bytes = r.content
                except Exception as e:
                    print(f"⚠️ [UI Fallback] Voice test API failed: {e}. Running locally.")

            if wav_bytes is None:
                wav_bytes = preview_tts(text, cfg)

            if wav_bytes is None:
                return None, "❌ TTS generation failed — check your voice file and TTS model."
            sr, audio = _bytes_to_gradio_audio(wav_bytes)
            return (sr, audio), "✅ Preview ready!"

        test_btn.click(
            on_test_voice,
            inputs=[
                voice_studio_upload, test_text, temp_slider, topp_slider, speed_slider,
                tts_model_name, tts_timbre, tts_instruct
            ],
            outputs=[test_audio, test_status],
        )

        # ── Preview helper ─────────────────────────────────────────────────────
        def _parse_chapter_titles(labels: list[str]) -> list[str] | None:
            """
            Convert checkbox labels like '5. Chapter 1: Crimson  (~1,793 words)'
            into plain titles like 'Chapter 1: Crimson'.

            Returning titles (not numbers) avoids the scan/extractor numbering
            mismatch: scan() numbers ALL TOC entries 1-N, but ingest_epub()
            only numbers ML-classified content chapters 1-M.
            """
            if not labels:
                return None
            import re as _re
            out = []
            for lbl in labels:
                after_num = lbl.split(". ", 1)[-1]          # strip "5. " prefix
                title = _re.sub(r'\s+\(~[\d,]+\s*words\)\s*$', '', after_num).strip()
                if title:
                    out.append(title)
            return out or None

        def _load_cached_chapters_if_available(prog_json_path: str, selected_chapters_labels: list[str] | None = None, log_fn=None):
            """If prog_json_path exists and has cached text for chapters, return list of ExtractedChapter objects.
            Otherwise return None."""
            if not os.path.exists(prog_json_path):
                return None
            try:
                with open(prog_json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                ch_list = data.get("chapters", [])
                if not ch_list or not all(c.get("text", "").strip() for c in ch_list):
                    return None
                
                selected_titles = _parse_chapter_titles(selected_chapters_labels) if selected_chapters_labels else None
                
                chapters = []
                for c in ch_list:
                    title = c.get("title", "")
                    num = c.get("num", 0)
                    if selected_titles:
                        matched = False
                        for st in selected_titles:
                            if st and (st == title or st in title):
                                matched = True
                                break
                        if not matched:
                            continue
                    chapters.append(ExtractedChapter(
                        num=num,
                        title=title,
                        text=c["text"],
                        sentences=c.get("sentences") or []
                    ))
                if chapters:
                    if log_fn:
                        log_fn("📦 Using cached chapter text from progress JSON (skipping book re-parsing).")
                    return chapters
            except Exception as e:
                if log_fn:
                    log_fn(f"⚠️ Could not load cached text from progress JSON: {e}")
            return None

        def on_preview(scan_res, file_obj, selected_chapters, page_ranges_str, epub_ocr):
            if file_obj is None:
                yield "", gr.update(visible=False)
                return
            path = file_obj.name if hasattr(file_obj, "name") else str(file_obj)
            selections   = None
            page_ranges  = None
            if scan_res and scan_res.has_toc and selected_chapters:
                selections = _parse_chapter_titles(selected_chapters)
            elif scan_res and not scan_res.has_toc and scan_res.file_type != "txt":
                page_ranges = []
                for part in page_ranges_str.split(","):
                    part = part.strip()
                    if "-" in part:
                        try:
                            s, e = part.split("-")
                            page_ranges.append((int(s.strip()), int(e.strip())))
                        except ValueError:
                            pass
            log_q  = queue.Queue()
            chapters, _ = extract(
                path, selections=selections, enable_ocr=epub_ocr,
                page_ranges=page_ranges, log_fn=log_q.put,
            )
            if not chapters:
                yield "⚠️ No chapters found.", gr.update(visible=False)
                return
            rows = preview_chapters(chapters, log_q)
            data = [[r["idx"], r["title"], r["chars"], r["words"], r["sentences"]] for r in rows]
            yield "", gr.update(value=data, visible=True)

        preview_btn.click(
            on_preview,
            inputs=[scan_state, book_file, chapter_check, page_ranges_box, epub_ocr_chk],
            outputs=[log_box, preview_table],
        )

        # ── Generate Audiobook ────────────────────────────────────────────────
        def on_generate(
            scan_res, file_obj,
            selected_chapters, page_ranges_str,
            book_language, book_title, author, cover_path,
            voice_path, output_fmt, lufs,
            temp, top_p, pause, para_pause,
            max_len, true_peak,
            epub_ocr, force_repro,
            worker_count, parallel_mode, export_text, pron_file_obj, progress_file_obj, tts_provider,
            mname, timbre, instruct,
            single_file, export_lrc, export_srt, export_vtt,
            torch_compile, regen_missing,
            progress=gr.Progress(track_tqdm=False)
        ):
            if file_obj is None:
                yield "⚠️ Please upload a book file first.", gr.update(visible=False), gr.update(visible=False), [], None
                return
            if ("Base" in mname) and not voice_path:
                yield "⚠️ Please set a narrator voice in the Voice Studio tab for Base models.", gr.update(visible=False), gr.update(visible=False), [], None
                return

            path = file_obj.name if hasattr(file_obj, "name") else str(file_obj)

            # Build selections
            selections  = None
            page_ranges = None

            if scan_res and scan_res.has_toc and selected_chapters:
                # Use title-based matching — avoids the numbering mismatch between
                # scan() (all 13 TOC entries, nums 1-13) and ingest_epub() (content
                # chapters only, re-numbered 1-9).
                selections = _parse_chapter_titles(selected_chapters)
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

            # ── Parse pronunciation file ──────────────────────────────────────
            pron_map = {}
            if pron_file_obj is not None:
                pron_path = pron_file_obj.name if hasattr(pron_file_obj, "name") else str(pron_file_obj)
                try:
                    with open(pron_path, encoding="utf-8") as fh:
                        for line in fh:
                            line = line.strip()
                            if not line or line.startswith("#") or "==" not in line:
                                continue
                            search, repl = line.split("==", 1)
                            pron_map[search.strip()] = repl.strip()
                except OSError:
                    pass

            # Config
            book_out = os.path.join(
                _OUTPUT_DIR,
                re.sub(r'[\\/*?:"<>|]', "", book_title or "audiobook"),
            )
            os.makedirs(book_out, exist_ok=True)

            # If a progress JSON file was uploaded, copy/overwrite it in the output directory
            if progress_file_obj is not None:
                uploaded_progress_path = progress_file_obj.name if hasattr(progress_file_obj, "name") else str(progress_file_obj)
                try:
                    dest_progress_path = os.path.join(book_out, "generation_progress.json")
                    import shutil
                    shutil.copy2(uploaded_progress_path, dest_progress_path)
                    print(f"[UI] Progress file uploaded. Copied to {dest_progress_path}")
                except Exception as e:
                    print(f"[UI] Failed to copy uploaded progress file: {e}")

            cfg = AudiobookConfig(
                book_title=book_title,
                book_path=path,
                author=author,
                language=book_language,
                cover_image=cover_path,
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
                worker_count=int(worker_count),
                parallel_mode=parallel_mode,
                export_text=bool(export_text),
                pronunciation_map=pron_map,
                tts_provider_name=tts_provider or "qwen",
                tts_model_name=mname,
                tts_timbre=timbre.split()[-1] if timbre else "",
                tts_instruct=instruct,
                single_file_mode=single_file,
                export_lrc=export_lrc,
                export_srt=export_srt,
                export_vtt=export_vtt,
                torch_compile=bool(torch_compile),
                selected_chapters=selected_chapters or [],
                regen_missing=bool(regen_missing),
            )

            log_q  = queue.Queue()
            prog_q = queue.Queue()
            cancel = CancelToken()

            async def listen_ws(task_id, log_q, prog_q):
                import websockets
                import json
                url = f"ws://127.0.0.1:8000/api/v1/ws/{task_id}"
                try:
                    async with websockets.connect(url) as ws:
                        while True:
                            msg = await ws.recv()
                            data = json.loads(msg)
                            if data["type"] == "log":
                                log_q.put(data["message"])
                            elif data["type"] == "progress":
                                prog_val = data["progress"]
                                prog_q.put((int(prog_val * 100), 100))
                            elif data["type"] == "status":
                                status = data["status"]
                                if status in ("failed", "cancelled"):
                                    break
                            elif data["type"] == "completed":
                                paths = ",".join(data["files"])
                                log_q.put(f"__DONE__::{paths}")
                                break
                except Exception as e:
                    log_q.put(f"⚠️ [WebSocket Error] Disconnected or failed to connect to API server: {e}")

            def _runner():
                try:
                    # Check if progress JSON already has cached chapter text
                    prog_json_path = os.path.join(book_out, "generation_progress.json")
                    chapters = _load_cached_chapters_if_available(prog_json_path, selected_chapters, log_q.put)
                    
                    if not chapters:
                        # Extract from book file
                        chapters, _ = extract(
                            path,
                            selections=selections,
                            enable_ocr=epub_ocr,
                            page_ranges=page_ranges,
                            log_fn=log_q.put,
                        )
                    if chapters:
                        if is_api_healthy():
                            try:
                                import requests
                                import dataclasses
                                import asyncio
                                
                                log_q.put("📡 Dispatching task to FastAPI orchestrator...")
                                config_dict = dataclasses.asdict(cfg)
                                chapters_list = [
                                    {
                                        "num": ch.num,
                                        "title": ch.title,
                                        "text": ch.text,
                                        "sentences": ch.sentences
                                    } for ch in chapters
                                ]
                                
                                payload = {
                                    "config": config_dict,
                                    "chapters": chapters_list
                                }
                                r = requests.post("http://127.0.0.1:8000/api/v1/generate", json=payload)
                                r.raise_for_status()
                                task_id = r.json()["task_id"]
                                
                                cancel.task_id = task_id
                                log_q.put(f"✅ Enqueued task successfully. Task ID: {task_id}")
                                
                                # Listen to WebSocket stream — errors here do NOT mean
                                # generation failed; the backend is still running.
                                try:
                                    asyncio.run(listen_ws(task_id, log_q, prog_q))
                                except Exception as ws_err:
                                    # WebSocket dropped but the API task is still alive.
                                    # Poll for completion instead of re-running locally.
                                    log_q.put(f"⚠️ [WebSocket] Stream interrupted ({ws_err}). Polling API for completion...")
                                    import time
                                    while True:
                                        try:
                                            poll = requests.get(f"http://127.0.0.1:8000/api/v1/tasks/{task_id}", timeout=5)
                                            if poll.ok:
                                                st = poll.json()
                                                if st["status"] == "completed":
                                                    paths = ",".join(st.get("output_files", []))
                                                    log_q.put(f"__DONE__::{paths}")
                                                    break
                                                elif st["status"] in ("failed", "cancelled"):
                                                    log_q.put(f"❌ API task {st['status']}.")
                                                    log_q.put("__DONE__::")
                                                    break
                                        except Exception:
                                            pass
                                        time.sleep(3)
                            except Exception as e:
                                # Only fall back locally if the task was NEVER enqueued
                                # (i.e., the error occurred before we got a task_id)
                                if not getattr(cancel, "task_id", None):
                                    log_q.put(f"⚠️ API dispatch failed: {e}. Falling back to local generation...")
                                    out_files = run_pipeline(cfg, chapters, log_q, prog_q, cancel)
                                    log_q.put(f"__DONE__::{','.join(out_files)}")
                                else:
                                    # Task was enqueued — cancel it cleanly so we don't
                                    # have an orphaned GPU job and a local duplicate running.
                                    log_q.put(f"⚠️ API error after task enqueue: {e}. Cancelling API task to avoid duplication.")
                                    try:
                                        requests.post(f"http://127.0.0.1:8000/api/v1/tasks/{cancel.task_id}/cancel", timeout=3)
                                    except Exception:
                                        pass
                                    log_q.put("__DONE__::")
                        else:
                            out_files = run_pipeline(cfg, chapters, log_q, prog_q, cancel)
                            log_q.put(f"__DONE__::{','.join(out_files)}")
                    else:
                        log_q.put("__DONE__::")
                except Exception as e:
                    import traceback
                    err_msg = f"❌ [Fatal Error] Pipeline crashed: {e}"
                    print(err_msg)
                    traceback.print_exc()
                    log_q.put(err_msg)
                    log_q.put("__DONE__::")

            t = threading.Thread(target=_runner, daemon=True)
            t.start()

            log_text = ""
            out_files   = []
            last_prog_val = 0.0

            while t.is_alive() or not log_q.empty() or not prog_q.empty():
                try:
                    while not prog_q.empty():
                        cur_prog, tot_prog = prog_q.get_nowait()
                        if tot_prog > 0:
                            prog_val = float(cur_prog) / float(tot_prog)
                            last_prog_val = prog_val
                            progress(prog_val, desc=f"{prog_val*100:.2f}% Complete")
                except queue.Empty:
                    pass
                
                prog_html_str = f'<div style="text-align: center; margin-bottom: 5px; font-weight: bold;">Generation Progress: {last_prog_val*100:.2f}%</div><progress value="{last_prog_val*100}" max="100" style="width:100%; height:25px;"></progress>'

                try:
                    msg = log_q.get(timeout=0.2)
                    if msg.startswith("__DONE__::"):
                        paths = msg.split("::", 1)[1]
                        out_files = [p for p in paths.split(",") if p and os.path.exists(p)]
                        break
                    log_text += msg + "\n"
                    yield log_text, prog_html_str, gr.update(visible=False), [], cancel
                except queue.Empty:
                    yield log_text, prog_html_str, gr.update(visible=False), [], cancel

            final_prog_html = '<div style="text-align: center; margin-bottom: 5px; font-weight: bold;">Generation Progress: 100.00%</div><progress value="100" max="100" style="width:100%; height:25px;"></progress>'
            if out_files:
                prog_json_path = os.path.join(book_out, "generation_progress.json")
                files_to_show = list(out_files)
                if os.path.exists(prog_json_path):
                    files_to_show.append(prog_json_path)
                yield (
                    log_text + "\n✅ Generation complete!",
                    final_prog_html,
                    gr.update(visible=True),
                    files_to_show,
                    cancel
                )
            else:
                yield log_text + "\n⚠️ No output files generated.", final_prog_html, gr.update(visible=False), [], cancel

        generate_btn.click(
            on_generate,
            inputs=[
                scan_state, book_file,
                chapter_check, page_ranges_box,
                book_language_dd, book_title_box, book_author_box, cover_image,
                voice_studio_upload, output_format, lufs_slider,
                temp_slider, topp_slider, sent_pause_sl, para_pause_sl,
                max_len_sl, lufs_adv,
                epub_ocr_chk, force_repro_chk,
                worker_count_sl, parallel_mode_dd, export_text_chk, pronunciation_file, progress_file_upload, tts_provider_dd,
                tts_model_name, tts_timbre, tts_instruct,
                single_file_mode, export_lrc_chk, export_srt_chk, export_vtt_chk,
                torch_compile_chk, regen_missing_chk
            ],
            outputs=[log_box, prog_html, download_col, download_files, cancel_state],
        )

        # ── Auto-Sync & Progress Listeners ────────────────────────────────────
        book_title_box.change(
            check_existing_progress,
            inputs=[book_title_box],
            outputs=[existing_progress_info]
        )

        def on_progress_upload(file_obj):
            if file_obj is None:
                return ["", gr.update()] + [gr.update() for _ in range(27)]
            path = file_obj.name if hasattr(file_obj, "name") else str(file_obj)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                title = data.get("book_title", "")
                book_path = data.get("book_path", "")
                voice_file = data.get("voice_file", "")
                settings = data.get("settings", {})
                
                # Check for existence of book and voice files
                book_found_msg = ""
                if book_path:
                    if os.path.exists(book_path):
                        book_found_msg = f"  - ✅ Found book file at: `{book_path}`\n"
                    else:
                        book_found_msg = f"  - ⚠️ Book file not found at: `{book_path}`\n"
                
                voice_found_msg = ""
                if voice_file:
                    if os.path.exists(voice_file):
                        voice_found_msg = f"  - ✅ Found voice file at: `{voice_file}`\n"
                    else:
                        voice_found_msg = f"  - ⚠️ Voice file not found at: `{voice_file}`\n"

                chapters = data.get("chapters", [])
                completed = sum(1 for c in chapters if c.get("status") in ("completed", "complete"))
                total = len(chapters)
                
                msg = (
                    f"### ✅ Progress File Loaded Successfully!\n"
                    f"- **Book:** {title}\n"
                    f"- **Total Chapters:** {total}\n"
                    f"- **Completed:** {completed}\n"
                    f"**Paths Checked:**\n"
                    f"{book_found_msg}"
                    f"{voice_found_msg}"
                    f"Settings and files have been restored to the UI."
                )

                def val(key, default_fallback):
                    return settings.get(key, default_fallback) if settings else default_fallback

                # Let's map settings values
                author_val = val("author", "")
                lang_val = val("language", "English")
                out_fmt_val = val("output_format", "mp3")
                lufs_val = val("lufs", -18)
                temp_val = val("temperature", 0.3)
                topp_val = val("top_p", 0.8)
                pause_val = val("pause", 0.5)
                para_pause_val = val("para_pause", 1.2)
                
                mname_val = val("tts_model_name", "Qwen/Qwen3-TTS-12Hz-1.7B-Base")
                timbre_val = val("tts_timbre", "[English] ryan")
                _valid_timbres = [
                    "[Chinese] vivian",
                    "[Chinese] serena",
                    "[Chinese] uncle_fu",
                    "[Chinese (Beijing Dialect)] dylan",
                    "[Chinese (Sichuan Dialect)] eric",
                    "[English] ryan",
                    "[English] aiden",
                    "[Japanese] ono_anna",
                    "[Korean] sohee"
                ]
                if timbre_val and timbre_val not in _valid_timbres:
                    for _choice in _valid_timbres:
                        if _choice.endswith(timbre_val.strip()) or _choice.split()[-1] == timbre_val.strip():
                            timbre_val = _choice
                            break
                instruct_val = val("tts_instruct", "")
                
                max_len_val = val("max_len", 399)
                true_peak_val = val("true_peak", -1.5)
                worker_count_val = val("worker_count", 2)
                parallel_mode_val = val("parallel_mode", "chunks")
                tts_provider_val = val("tts_provider_name", "qwen")
                
                ocr_val = val("docling_ocr", False) or val("epub_ocr", False)
                force_val = val("force_reprocess", False)
                exp_txt_val = val("export_text", False)
                single_file_val = val("single_file_mode", False) or val("single_file", False)
                exp_lrc_val = val("export_lrc", True)
                exp_srt_val = val("export_srt", False)
                exp_vtt_val = val("export_vtt", False)
                torch_compile_val = val("torch_compile", False)
                regen_missing_val = val("regen_missing", True)

                # Restore pronunciation map if present
                pron_map = val("pronunciation_map", {})
                pron_file_update = gr.update()
                if pron_map and isinstance(pron_map, dict):
                    try:
                        temp_pron_path = os.path.join(_OUTPUT_DIR, "restored_pronunciation_fixes.txt")
                        with open(temp_pron_path, "w", encoding="utf-8") as pf:
                            for search, repl in pron_map.items():
                                pf.write(f"{search} == {repl}\n")
                        pron_file_update = gr.update(value=temp_pron_path)
                    except Exception:
                        pass

                # Restore saved chapter selections (raw labels) if present
                saved_chapters = val("selected_chapters", [])

                return (
                    msg,
                    gr.update(value=title) if title else gr.update(),
                    gr.update(value=book_path) if book_path and os.path.exists(book_path) else gr.update(),
                    gr.update(value=voice_file) if voice_file and os.path.exists(voice_file) else gr.update(),
                    gr.update(value=author_val),
                    gr.update(value=lang_val),
                    gr.update(value=out_fmt_val),
                    gr.update(value=lufs_val),
                    gr.update(value=temp_val),
                    gr.update(value=topp_val),
                    gr.update(value=pause_val),
                    gr.update(value=para_pause_val),
                    gr.update(value=mname_val),
                    gr.update(value=timbre_val),
                    gr.update(value=instruct_val),
                    gr.update(value=max_len_val),
                    gr.update(value=true_peak_val),
                    gr.update(value=worker_count_val),
                    gr.update(value=parallel_mode_val),
                    gr.update(value=tts_provider_val),
                    gr.update(value=ocr_val),
                    gr.update(value=force_val),
                    gr.update(value=exp_txt_val),
                    gr.update(value=single_file_val),
                    gr.update(value=exp_lrc_val),
                    gr.update(value=exp_srt_val),
                    gr.update(value=exp_vtt_val),
                    gr.update(value=torch_compile_val),
                    gr.update(value=regen_missing_val),
                    pron_file_update,
                    gr.update(value=saved_chapters) if saved_chapters else gr.update(),
                    saved_chapters or None,   # json_selected_chapters_state
                )
            except Exception as e:
                return [f"❌ Failed to parse progress file: {e}", gr.update()] + [gr.update() for _ in range(30)]

        progress_file_upload.upload(
            on_progress_upload,
            inputs=[progress_file_upload],
            outputs=[
                progress_upload_status, book_title_box, book_file, voice_studio_upload,
                book_author_box, book_language_dd, output_format, lufs_slider,
                temp_slider, topp_slider, sent_pause_sl, para_pause_sl, tts_model_name,
                tts_timbre, tts_instruct, max_len_sl, lufs_adv, worker_count_sl,
                parallel_mode_dd, tts_provider_dd, epub_ocr_chk, force_repro_chk,
                export_text_chk, single_file_mode, export_lrc_chk, export_srt_chk, export_vtt_chk,
                torch_compile_chk, regen_missing_chk, pronunciation_file, chapter_check,
                json_selected_chapters_state,
            ]
        )

        # ── Export Config JSON ────────────────────────────────────────────────
        def on_export_config(
            scan_res, file_obj,
            selected_chapters, page_ranges_str,
            book_language, book_title, author, cover_path,
            voice_path, output_fmt, lufs,
            temp, top_p, pause, para_pause,
            max_len, true_peak,
            epub_ocr, force_repro,
            worker_count, parallel_mode, export_text, pron_file_obj, tts_provider,
            mname, timbre, instruct,
            single_file, export_lrc, export_srt, export_vtt,
            torch_compile, regen_missing,
        ):
            """Parse the book, cache chapter text, and write a self-contained
            generation_progress.json — without starting TTS generation."""
            if file_obj is None:
                return (
                    "⚠️ Please upload a book file first.",
                    gr.update(visible=False),
                )

            path = file_obj.name if hasattr(file_obj, "name") else str(file_obj)

            # Build chapter selections
            selections = None
            page_ranges = None
            if scan_res and scan_res.has_toc and selected_chapters:
                import re as _re
                selections = []
                for lbl in selected_chapters:
                    after_num = lbl.split(". ", 1)[-1]
                    title = _re.sub(r'\s+\(~[\d,]+\s*words\)\s*$', '', after_num).strip()
                    if title:
                        selections.append(title)
            elif scan_res and not scan_res.has_toc and scan_res.file_type != "txt":
                page_ranges = []
                for part in page_ranges_str.split(","):
                    part = part.strip()
                    if "-" in part:
                        try:
                            s, e = part.split("-")
                            page_ranges.append((int(s.strip()), int(e.strip())))
                        except ValueError:
                            pass

            # Build output directory and config
            book_out = os.path.join(
                _OUTPUT_DIR,
                re.sub(r'[\\/*?":"<>|]', "", book_title or "audiobook"),
            )
            os.makedirs(book_out, exist_ok=True)

            # Check if progress JSON already has cached chapter text
            prog_path = os.path.join(book_out, "generation_progress.json")
            chapters = _load_cached_chapters_if_available(prog_path, selected_chapters)

            if not chapters:
                # Extract chapters (with text + sentences) from book file
                try:
                    chapters, _ = extract(path, selections=selections or None,
                                           enable_ocr=epub_ocr, page_ranges=page_ranges)
                except Exception as exc:
                    return (
                        f"❌ Book extraction failed: {exc}",
                        gr.update(visible=False),
                    )

            if not chapters:
                return (
                    "⚠️ No chapters extracted from the book.",
                    gr.update(visible=False),
                )

            # Parse pronunciation file if provided
            pron_map = {}
            if pron_file_obj is not None:
                pf_path = pron_file_obj.name if hasattr(pron_file_obj, "name") else str(pron_file_obj)
                try:
                    with open(pf_path, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if not line or line.startswith("#") or "==" not in line:
                                continue
                            search, repl = line.split("==", 1)
                            pron_map[search.strip()] = repl.strip()
                except OSError:
                    pass

            cfg = AudiobookConfig(
                book_title=book_title, book_path=path, author=author,
                language=book_language, cover_image=cover_path, output_dir=book_out,
                output_format=output_fmt, voice_file=voice_path,
                temperature=temp, top_p=top_p, pause=pause, para_pause=para_pause,
                max_len=int(max_len), lufs=int(lufs), true_peak=true_peak,
                force_reprocess=force_repro, worker_count=int(worker_count),
                parallel_mode=parallel_mode, export_text=bool(export_text),
                pronunciation_map=pron_map,
                tts_provider_name=tts_provider or "qwen", tts_model_name=mname,
                tts_timbre=timbre.split()[-1] if timbre else "",
                tts_instruct=instruct, single_file_mode=single_file,
                export_lrc=export_lrc, export_srt=export_srt, export_vtt=export_vtt,
                torch_compile=bool(torch_compile),
                selected_chapters=selected_chapters or [],
                regen_missing=bool(regen_missing),
            )
            settings_dict = dataclasses.asdict(cfg)

            chapters_data = [
                {"num": ch.num, "title": ch.title, "text": ch.text, "sentences": ch.sentences}
                for ch in chapters
            ]

            prog_path = os.path.join(book_out, "generation_progress.json")
            # Force fresh creation so text is always included
            if os.path.exists(prog_path):
                import json as _json
                try:
                    with open(prog_path, "r", encoding="utf-8") as f:
                        existing = _json.load(f)
                    existing_by_num = {c["num"]: c for c in existing.get("chapters", [])}
                except Exception:
                    existing_by_num = {}
                    existing = {}

                merged_chapters = []
                for cd in chapters_data:
                    ec = existing_by_num.get(cd["num"], {})
                    merged_chapters.append({
                        "num": cd["num"],
                        "title": cd["title"],
                        "status": ec.get("status", "pending"),
                        "text": cd["text"],
                        "sentences": cd["sentences"],
                    })
                existing["settings"] = settings_dict
                existing["book_path"] = path
                existing["voice_file"] = voice_path or ""
                existing["chapters"] = merged_chapters
                with open(prog_path, "w", encoding="utf-8") as f:
                    _json.dump(existing, f, indent=4)
            else:
                load_or_create_progress_file(
                    prog_path, chapters_data, book_title,
                    book_path=path, voice_file=voice_path or "",
                    settings=settings_dict,
                )

            return (
                f"✅ **Config exported!** {len(chapters)} chapters cached.\n\n"
                f"File saved to:\n`{prog_path}`\n\n"
                f"To generate without Gradio, run:\n"
                f"```\npython cli.py \"{prog_path}\"\n```",
                gr.update(value=prog_path, visible=True),
            )

        export_cfg_btn.click(
            on_export_config,
            inputs=[
                scan_state, book_file,
                chapter_check, page_ranges_box,
                book_language_dd, book_title_box, book_author_box, cover_image,
                voice_studio_upload, output_format, lufs_slider,
                temp_slider, topp_slider, sent_pause_sl, para_pause_sl,
                max_len_sl, lufs_adv,
                epub_ocr_chk, force_repro_chk,
                worker_count_sl, parallel_mode_dd, export_text_chk, pronunciation_file, tts_provider_dd,
                tts_model_name, tts_timbre, tts_instruct,
                single_file_mode, export_lrc_chk, export_srt_chk, export_vtt_chk,
                torch_compile_chk, regen_missing_chk,
            ],
            outputs=[export_cfg_status, export_config_file],
        )

        # ── Cancel ────────────────────────────────────────────────────────────
        def on_cancel(cancel_tok):
            if cancel_tok:
                cancel_tok.cancel()
                if hasattr(cancel_tok, "task_id") and cancel_tok.task_id:
                    try:
                        import requests
                        requests.post(f"http://127.0.0.1:8000/api/v1/tasks/{cancel_tok.task_id}/cancel")
                        print(f"[UI] Cancelled API task: {cancel_tok.task_id}")
                    except Exception as e:
                        print(f"[UI] Failed to cancel API task: {e}")
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
        server_name="localhost",
        server_port=7860,
        share=False,
        show_error=True,
        theme=_THEME,
        css=_CSS,
        allowed_paths=[_ROOT],
    )
