# 📝 Changelog

All notable changes to **AudiobookMaker** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [v1.1.0] - 2026-08-05

### ⚡ Added
- **Thread-Safe Atomic Progress I/O Layer (`progress_io.py`)**: Created a dedicated, thread-safe file I/O layer for `generation_progress.json` featuring atomic writes (tmp file + `os.replace()`), module-level write lock (`_WRITE_LOCK`), UTF-8 BOM auto-decoding, leading garbage stripping, empty file detection, HTML response guarding, and multi-encoding fallbacks.
- **Atomic Read-Inside-Lock Update Pattern**: Enforced atomic read-modify-write inside single lock acquisitions for chapter status and chunk completion updates, completely eliminating TOCTOU race conditions under concurrent multi-chapter/multi-GPU execution.
- **Config Contract Schema Versioning (`AudiobookConfig`)**: Introduced schema versioning (`_CONFIG_SCHEMA_VERSION = 5`, `config_version`) and a hardened `from_dict()` method that tolerates unknown/missing keys and warns on stale progress versions without crashing. Added `field_summary()` classmethod for config diagnostics.
- **Eager Multi-GPU Pool Warmup (`GPUPoolManager`)**: Added blocking parallel provider warmup via `ThreadPoolExecutor` during GPU pool creation to eliminate lazy-initialization race conditions. Failed GPU warmups (`OutOfMemoryError`, `RuntimeError`) are automatically caught and excluded from the pool so healthy GPUs continue synthesizing.
- **Comprehensive Regression Test Suite (`tests/test_hardening.py`)**: Added a 21-case test suite covering config contract parsing, atomic progress I/O, concurrent chunk update race prevention, provider readiness guards, and health endpoint output.

### 🔄 Changed
- Updated `GET /api/v1/health` endpoint to report per-device model readiness via provider `is_ready` property.
- Replaced all legacy JSON progress file loading and saving calls in `pipeline.py`, `app.py`, and `cli.py` with `progress_io` methods.

### 🧹 Removed
- Deprecated legacy, non-atomic progress file helper functions (`load_or_create_progress_file`, `update_progress_file`, `update_progress_file_chunk`, `_progress_lock`) from `audiobook_factory/utils.py`.

---

## [v1.0.0] - 2026-08-02

### ⚡ Added
- **3-Stage Overlapped Chapter Pipeline (`chapter_pipeline.py`)**: Implemented an overlapped 3-stage execution pipeline separating CPU text preparation (Stage A), parallel GPU batch synthesis worker threads (Stage B), and streaming partial mastering with async disk I/O (Stage C).
- **Chunk-Level Mid-Chapter Resumption**: Audio sentence chunks are incrementally cached to `.temp_chunks/` during generation. Interrupted or restarted runs seamlessly resume from the exact sentence without re-synthesizing completed chunks.
- **Rust PyO3 SIMD & Audio Mastering Acceleration (`audiobook_rust`)**: High-performance Rust PyO3 extension module providing 5.5× faster audio mastering, SIMD sentence tokenization, and ultra-fast text normalization with transparent pure-Python fallbacks.
- **Native Kaggle Notebook & Dual-GPU Support (`AudiobookMaker_Kaggle.ipynb`)**: Dedicated Kaggle notebook environment with dual T4 GPU (T4x2) parallel execution support.
- **Diagnostic Pre-Run Verification Scripts (`colab_prerun_check.py` & `kaggle_prerun_check.py`)**: Lightweight (<5s execution) diagnostic verification scripts that validate GPU allocation, VRAM metrics, PyO3 Rust bindings, and GPU pool dispatch before loading heavy 1.7B TTS model weights.
- **CLI Cover Art Embedding Tool (`--embed-cover-only`)**: Added `--embed-cover-only` flag to `cli.py` to instantly inject album cover artwork and ID3 metadata into pre-generated audio files without loading the TTS model.
- **INT8 Model Quantization (`--quantization int8`)**: Integrated 8-bit model quantization via `bitsandbytes` to reduce VRAM requirements by ~50%.
- **Multi-Format Subtitle Export**: Added SRT (`.srt`) and WebVTT (`.vtt`) subtitle file generation alongside `.lrc` timed lyrics.
- **True Multi-GPU Parallel Engine (`GPUPoolManager` & `ProviderPool`)**: Work-stealing GPU pool dispatcher for multi-GPU systems with VRAM monitoring and concurrent API queueing.

### 🔄 Changed
- Converted background task consumer loop (`api/worker.py`) to run tasks concurrently up to the number of detected GPUs via `asyncio.Semaphore`.
- Updated Gradio header with real-time multi-GPU status badge and GPU VRAM memory monitoring.
- Updated Advanced tab parallel worker slider to automatically default to `min(gpu_count * 4, 8)`.

### 🐛 Fixed
- Fixed cancellation token attribute error (`'CancelToken' object has no attribute 'cancelled'`).
- Fixed Flash Attention 2 runtime fallback to PyTorch SDPA on GPUs without Flash Attention support (e.g. Tesla T4).

---

## [v0.5.0] - Initial Production Feature Release

### ✨ Added
- **Headless CLI Pipeline (`cli.py`)**: Full command-line interface for running audiobook generation headless in cloud or terminal environments without launching a browser UI.
- **FastAPI / WebSocket Orchestration Server (`start_api.py`, `api/server.py`)**: Detached FastAPI orchestration backend that offloads GPU jobs from Gradio and streams real-time logs and progress via WebSockets.
- **Cached Book Extraction & JSON-First Session Workflow**: Embedded fully extracted and segmented chapter text inside `generation_progress.json` to eliminate re-parsing large books. Added progress JSON upload at the top of the Gradio interface for instant session restoration.
- **Chapter Selection Memory**: Automatically saves selected chapter subsets into progress JSON and restores checkbox states when resuming sessions.
- **AI Text Extraction Engine (`extractor_engine.py`)**: 5-phase extraction pipeline combining Docling, OCR, ML classifier, and heuristic normalization for multi-format books (`.epub`, `.mobi`, `.pdf`, `.docx`, `.odt`, `.txt`).
- **EPUB Image OCR**: Integrated EasyOCR to extract text embedded in EPUB image pages.
- **Voice Studio & Qwen3-TTS Engine**: Zero-shot voice cloning from reference WAV, voice prompting (VoiceDesign), multi-language synthesis (8 languages), language-labeled premium timbres, and instant voice testing tab.
- **7-Step Voice Audio Preprocessing (`voice_preprocessor.py`)**: Cleaning pipeline featuring noise reduction, noise gate, high-pass filter, silence removal, volume normalization, formant shifting, and resampling.
- **Audiobookshelf Integration & Metadata Tagging**: Zero-padded Audiobookshelf-compatible filenames and full ID3 metadata tagging (title, author, album, track number).
- **Audio Mastering Pipeline**: Automatic LUFS loudness normalization and True Peak limiting, with optional single-file unified output mode.
- **Google Colab Notebook (`AudiobookMaker_Colab.ipynb`)**: End-to-end Google Colab notebook supporting shareable Gradio links.
- **Pronunciation Fix Dictionary**: Uploadable `search==replace` text file support for custom phonetic replacements.
- **NLTK `punkt_tab` Auto-Downloader**: Automatic detection and downloading of NLTK `punkt` and `punkt_tab` tokenization resources on first run.
