# 📖 AudiobookMaker

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-Apache%202.0-blue)
![UI](https://img.shields.io/badge/UI-Gradio-orange)

An end-to-end AI audiobook generator with a **Gradio web UI**. Upload any book, clone a narrator voice, clean it up, and generate a chapterized, mastered audiobook — all locally, no cloud APIs needed.

---

## ✨ Features

- **Multi-format book support** — EPUB, MOBI, PDF, DOCX, ODT, TXT
- **Smart chapter detection** — EPUB/MOBI use a TOC-based chapter checklist; PDF/DOCX/ODT let you split by page ranges
- **AI text extraction** — 5-phase pipeline (Docling + OCR + ML classification + heuristic normalization) produces clean, TTS-ready text
- **EPUB image OCR** — EasyOCR reads text embedded in images inside EPUBs
- **Voice Design & Cloning** — Clone from a reference WAV or prompt an entire new voice using Qwen3-TTS. Supports **8 languages** (English, Chinese, Japanese, Korean, French, Spanish, Italian, German).
- **Voice preprocessing** — 7-step audio cleaning pipeline: noise reduction, noise gate, high-pass filter, silence removal, normalization, formant shifting, resampling
- **Voice test tab** — Type any sentence and preview the cloned voice before generating. Includes language-labeled premium timbres for optimized results.
- **Preview mode** — See chapter list with character + word counts before committing to a full audiobook run.
- **Pronunciation fixes** — Upload a `.txt` file with `search==replace` pairs to fix how the TTS pronounces specific words.
- **Parallel processing with Shared VRAM** — Process multiple chapters simultaneously using a **Global Shared Provider**. This allows worker counts up to 4 without multiplying VRAM usage, utilizing a thread-safe GPU lock with asynchronous disk I/O for maximum performance.
- **Synced Lyrics Export** — Automatically generates `.lrc` timed lyrics files perfect for Audiobookshelf syncing. 
- **Audiobookshelf-compatible output** — Zero-padded filenames + full ID3 tags (title, author, album, track) ready to drop into Audiobookshelf.
- **Mastered Output & Single File Mode** — Output mastered MP3, FLAC, WAV, or M4B files. Optionally combine all chapters into a massive single unified file with one click.
- **Live generation log & Decimal Progress** — Stream progress in real time with a **sub-chapter decimal progress bar** (e.g. 74.52%) and detailed live logs.
- **Modular TTS provider system** — Qwen3-TTS built-in; optimized with **Flash Attention 2** and async processing to keep your GPU at peak utilization.

---

> [!NOTE]
> 🚫 **The Voice Preprocessing tab** is currently unavailable. Please avoid using this feature until a fix is released.**
>
> ✅ All other features are functioning as expected.
>
> 🎧 For now, please perform voice preprocessing manually until this issue is resolved.


---


## 🖥️ UI Preview

### 📚 Book Tab — Upload & Chapter Selection
![Book Tab](docs/preview/01_book_tab.png)

### 🎧 Voice Preprocessing Tab — 7-Step Audio Cleaning 
![Voice Preprocessing Tab](docs/preview/02_voice_preprocessing_tab.png)

### 🎙️ Voice Studio Tab — Clone & Test Voice
![Voice Studio Tab](docs/preview/03_voice_studio_tab.png)

### ⚙️ Advanced Tab
![Advanced Tab](docs/preview/04_advanced_tab.png)

### 🚀 Generate Tab — Live Log & Download
![Generate Tab](docs/preview/05_generate_tab.png)

FYI: Ok, I know adding antigravity taken ss is not a good idea, but I am too lazy to take ss myself when antigravity already took them for me during the testing.

---

## 🗂️ Project Structure

```
AudiobookMaker/
├── install.sh / install.bat         ← One-click installer (detects OS + GPU)
├── run.sh / run.bat                 ← Start app + open browser automatically
├── app.py                           ← Gradio UI entry point
├── requirements.txt
├── docs/
│   └── preview/                     ← UI screenshots
└── audiobook_factory/
    ├── extractor_engine.py          ← Core AI text extraction engine
    │                                   (DocumentIngestor, MLClassifier, TextNormalizer)
    ├── text_extractor.py            ← Public API: scan() + extract()
    ├── voice_preprocessor.py        ← 7-step voice audio cleaning pipeline
    ├── pipeline.py                  ← Thread-safe audiobook generation orchestrator
    ├── audio_processor.py           ← Backward-compat shim (delegates to tts_providers)
    ├── filename_sanitizer.py        ← Cross-platform, Audiobookshelf-compatible filenames
    ├── story_analyzer.py            ← BookNLP story/character analysis
    ├── text_processing.py           ← Sentence splitting + text normalization
    ├── ffmpeg_utils.py              ← FFmpeg encoding helpers
    ├── config.py                    ← AudiobookConfig dataclass
    ├── utils.py                     ← Shared utilities
    └── tts_providers/               ← Modular TTS provider abstraction
        ├── base_tts_provider.py     ← BaseTTSProvider ABC + get_tts_provider() factory
        └── qwen_provider.py         ← Qwen3-TTS implementation (genesis + X-vector cloning)
```

---

## ⚙️ Prerequisites

- **Python 3.11+**
- **NVIDIA GPU with 6 GB+ VRAM** (strongly recommended — CPU is very slow for Qwen3-TTS)
- **CUDA Toolkit 11.8+**
- **FFmpeg** — the installer tries to handle this automatically

---

## 📦 Clone Repo
```
git clone https://github.com/MSpider3/AudiobookMaker.git
cd AudiobookMaker
```

---

## 🚀 Installation

The installer automatically:
- Detects your OS and installs **Python 3.11** via the native package manager
- Creates a **virtual environment**
- Detects your **GPU** and installs the correct PyTorch (CUDA 12.1, CUDA 11.8, or CPU)
- Installs all **dependencies** from `requirements.txt`
- Installs **FFmpeg** if missing

### Windows
```bat
install.bat
```

### macOS / Linux (Ubuntu, Fedora, Mint, Arch, openSUSE, …)
```bash
chmod +x install.sh
./install.sh
```

---

## ▶️ Running the App

The run script activates the environment, starts the server, and opens your browser automatically.

### Windows
```bat
run.bat
```

### macOS / Linux
```bash
chmod +x run.sh
./run.sh
```

Your browser will open at **http://localhost:7860** automatically.

---

## 📋 Step-by-Step Usage

### 1. 📚 Book Tab
1. Upload your book file (`.epub`, `.mobi`, `.pdf`, `.docx`, `.odt`, `.txt`)
2. **EPUB / MOBI with TOC** → A chapter checklist appears. Tick the chapters you want to convert. Use *Select All* / *Deselect All* for quick bulk selection.
3. **PDF / DOCX / ODT / MOBI (no TOC)** → Enter page ranges, e.g. `1-50, 51-120, 121-250`. Each range becomes a separate chapter file.
4. **TXT** → No page structure; the whole file becomes one audio file automatically.
5. **Language Selection** → Choose the language of your book from the dropdown. This tells the TTS engine which phonetic dictionary to use.
6. Fill in book title, author, choose output format and LUFS loudness target.

### 2. 🎧 Voice Preprocessing Tab *(recommended before cloning)*
Upload your raw voice WAV and run any combination of these steps:

| Step | What it does |
|------|-------------|
| Noise Reduction | Reduces background hiss/hum |
| Noise Gate | Silences frames below a dB threshold |
| High-Pass Filter | Removes low-frequency rumble |
| Silence Removal | Strips long silences between words |
| Normalize Volume | Peaks at your chosen dBFS |
| Formant Shift | Adjust voice gender/timbre *(experimental)* |
| Resample | Convert to 22k / 44.1k / 48k Hz |

Click **▶ Preview Processed Audio** to hear the result, then **💾 Use as narrator voice** to pass it to the next tab.

### 3. 🎙️ Voice Studio Tab
1. Upload or carry over the processed voice WAV.
2. Select a **TTS Model Variant** (Base for cloning, CustomVoice/VoiceDesign for prompting).
3. Choose a **Premium Timbre** (if using CustomVoice). Choices are prefixed with their native language (e.g., `[English] ryan`, `[Japanese] ono_anna`) for the best quality match.
4. Adjust TTS tuning parameters (speed, temperature, top-p, sentence/paragraph pauses).
5. Type any sentence in the **Voice Test** box and click **▶ Test Voice** to hear a preview.

### 4. ⚙️ Advanced Tab
- **Max chunk length** — TTS input character limit per sentence chunk (default 399).
- **Parallel chapter workers** — Process 1–4 chapters simultaneously. Thanks to our **Shared VRAM** architecture, increasing this does not significantly increase memory usage, but can dramatically speed up generation by pre-fetching the next sentence while the current one is speaking.
- **TTS Provider** — Currently: `qwen` (Qwen3-TTS). More providers will be added in future releases.
- **EasyOCR** — Enable to extract text from images embedded inside EPUB files
- **Force reprocess** — Re-extract text even if cached output exists
- **Export chapter text** — Write a `.txt` file alongside each audio file with the cleaned chapter text
- **Pronunciation fix file** — Upload a `.txt` with one fix per line in `search==replace` format (regex supported). Comments start with `#`.
  ```
  # Fix common TTS mispronunciations
  Barbadoes==Barbayduss
  N\.E\.==north east
  Dr\.==Doctor
  ```

### 5. 🚀 Generate Tab
1. Click **🔍 Preview Chapters** to see a table of chapter titles, character counts, word counts, and sentence counts — without generating any audio. Great for checking your chapter selections.
2. Click **🎧 Generate Audiobook** to start the full pipeline.
3. Watch the **Live Decimal Progress Bar** and log stream.
4. Use **⛔ Cancel** to stop at any time.
5. When complete, download individual chapter files or use **⬇ Download All (ZIP)**

---

## 🛠️ Customizing the Project

### Change the TTS model
Edit `audiobook_factory/tts_providers/qwen_provider.py`:
```python
# In _load_base_model():
"Qwen/Qwen3-TTS-12Hz-1.7B-Base"   # replace with any compatible Qwen3 checkpoint

# In _run_genesis():
"Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"   # design model used once for voice genesis
```

### Add a new TTS provider (future)
1. Create `audiobook_factory/tts_providers/my_provider.py`
2. Subclass `BaseTTSProvider` and implement `synthesize()`, `estimate_cost()`, `get_name()`
3. Register the name in `base_tts_provider.get_tts_provider()`
4. Add the name to the `tts_provider_dd` dropdown in `app.py`

### Tune the text extraction pipeline
Edit `audiobook_factory/extractor_engine.py`:
- **`TextNormalizer._strip_noise()`** — add/remove markdown patterns to clean
- **`TextNormalizer._fix_isolated_capitals()`** — font-kerning fixes (e.g. `T HE` → `THE`)
- **`_SKIP_TOC_TITLE`** regex — controls which TOC entries are excluded (copyright, gallery, etc.)
- **`MLClassifier.predict_is_chapter()`** — swap in a trained XGBoost model here when ready

### Tune audio mastering
Edit `audiobook_factory/pipeline.py`:
```python
lufs:      int   = -18    # loudness target
true_peak: float = -1.5   # max true peak dBTP
```
Or adjust these in the UI (LUFS slider in Book tab, True Peak in Advanced tab).

### Add a new output format
Edit `audiobook_factory/ffmpeg_utils.py` — add a new entry to `get_format_settings()`.

### Modify the voice preprocessing pipeline
Edit `audiobook_factory/voice_preprocessor.py`:
- Each step is a standalone function — easy to add, remove, or reorder
- `PreprocessConfig` dataclass controls all defaults

---

## 📦 Supported Input Formats

| Format | Chapter Detection | Fallback |
|--------|-----------------|---------|
| EPUB | ✅ TOC chapter list | — |
| MOBI | ✅ Try TOC | Page-range picker |
| PDF | ❌ | Page-range picker |
| DOCX | ❌ | Page-range picker |
| ODT | ❌ | Page-range picker |
| TXT | ❌ | Whole book |

---

## 📦 Output Formats

| Format | Notes |
|--------|-------|
| MP3 | Default, most compatible |
| FLAC | Lossless |
| WAV | Uncompressed |
| M4B | Audiobook format with chapter markers (Apple Books) |

---

## 🎧 Audiobookshelf Integration

[Audiobookshelf](https://github.com/advplyr/audiobookshelf) is a self-hosted audiobook library server. AudiobookMaker generates output that Audiobookshelf automatically detects:

1. **Drop the output folder** into your Audiobookshelf library directory
2. Audiobookshelf will auto-scan and import it as a book
3. Each chapter file has the correct **ID3 metadata** (title, author, album, track number) so chapter ordering and library display work correctly out of the box

Output filenames follow the `{NNNN}_{Chapter_Title}.mp3` format Audiobookshelf expects.

---

## 🙏 Acknowledgements

This project would not have been possible without the incredible work from these projects:

### [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) by QwenLM
The voice cloning and TTS engine powering all audio generation in this project.
State-of-the-art text-to-speech with zero-shot voice cloning from a short reference clip.

### [Mangio-RVC-Fork](https://github.com/Mangio621/Mangio-RVC-Fork) by Mangio621
The voice preprocessing pipeline in this project (noise reduction, noise gate, high-pass filter, silence removal, formant shifting) is directly inspired by the preprocessing architecture used in Mangio-RVC-Fork.

---

## 📄 License

Apache 2.0 — see [LICENSE](LICENSE) for details.
