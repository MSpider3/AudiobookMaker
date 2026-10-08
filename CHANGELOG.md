# 📝 Changelog

All notable changes to **AudiobookMaker** will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

> Rebuild the Rust extension after pulling (`cd audiobook_rust && maturin develop --release`): three of these fixes are in Rust, and a stale binary keeps the old behaviour.

### ⚖️ Licence
- **Relicensed from Apache-2.0 to AGPL-3.0-or-later.** Releases up to v1.5.0 remain available under Apache-2.0. `NOTICE` carries the copyright statement, an additional permission (AGPL section 7) to combine AudiobookMaker with the separately installed TTS engines, and third-party attributions. The web UI and the API now link to the source code.

### ⚡ Added
- **Five new TTS engines** behind a shared provider contract (`tts_providers/base_tts_provider.py`, `registry.py`): IndexTTS-2.5, MOSS-TTS, OmniVoice, Fish Audio S2 Pro and Higgs Audio v3. Each exposes its own controls as provider options, declares its licence and VRAM needs, and installs from `requirements/tts-<engine>.txt`. Engines pin incompatible `transformers` versions, so install one per environment.
- **Qwen3-TTS reworked**: validated preset speakers (CustomVoice), designed voices that are designed once and then cloned for the whole book on every GPU (VoiceDesign), saved voice presets, per-chunk token budgets against runaway generation, and every upstream sampling control.
- **Natural pacing** (`chunk_planner.py`): a paragraph's sentences are spoken together up to `max_len`, paragraphs get `para_pause`, dialogue tags stay with their quote; subtitles keep sentence-level timing.
- **Chunk verification** (`chunk_verifier.py`, `verify_chunks`): free duration/silence check by default, optional Whisper transcript check, automatic re-synthesis, and a list of chunks worth a listen in the run summary.
- **Spoken-form text** (`speech_text.py`, `normalize_speech_text`): currency, dates, years, ordinals, roman numerals, units and abbreviations rewritten for narration (English).
- **Chapter detection for TXT, DOCX, ODT and PDF**, real MOBI/AZW3 support (optional `mobi` package), and front/back matter that can be listed unticked instead of silently dropped.
- **Single-file audiobooks with chapter markers** (M4B/MP3), speed control for every engine, per-chapter redo (`redo_chapters`, `--redo`), ETA and an end-of-run summary of failed chapters.
- **CLI**: `--book` (no JSON needed), `--list-providers`, `--dry-run`, `--chapters`, `--redo`, `--tts-option`, `--verify` and the rest of the config as flags; exit codes 0 / 1 / 130.
- **API**: `/api/v1/providers`, `/api/v1/tasks`, task file download, request validation.
- **Kaggle test notebook** (`AudiobookMaker_Kaggle_Test.ipynb`, `tests/kaggle/abm_gpu_suite.py`) that tests a branch on real GPUs and packs the results into one archive. Besides each engine it measures the two-GPU speed-up on a text several batches long, narrates a whole twenty-page book, speaks a passage in six more languages, scores how close each cloned voice is to the narrator clip (speaker embeddings), and starts the API server and the web UI as real programs to generate through each.
- **Long test books in seven languages** (`tests/fixture_generation/long_books/`, `generate_long_books.py`): an original ten-chapter novella of about twenty pages each in English, French, Russian, Hindi, Chinese, Japanese and Korean, built into `long_book_<code>.epub`.

### 🚀 Changed
- **Multi-GPU**: all GPUs pull length-sorted batches from one shared queue per chapter; a GPU that fails hands its batch to the others. Batch size no longer shrinks after the first batch and adapts after an out-of-memory retry.
- **Progress file** is written once per batch instead of once per chunk; chapter numbers are stable across subset runs and status is matched by title.
- **Mastering** normalises loudness once; Rust functions release the GIL.
- **Chapters now reach the loudness target (`loudness.py`, `audiobook_rust/src/audio/master.rs`)**: narration has a few peaks far above its average level, so one static gain stopped 2-5 LU short of the LUFS target on every engine tested (-20 to -22 instead of -18). The full gain is now applied and those peaks go through a look-ahead limiter (1 ms gain curve, held over a pitch period, smoothed so it cannot click). Chapters that already fit are untouched; the limiter never takes more than 9 dB.
- **Chunk length follows the script (`chunk_planner.py`)**: `max_len` counts Latin characters (399 is about 25 seconds). A Chinese character is a whole syllable, so 399 of them were over a minute of speech in one TTS call. The limit is now scaled by how long the script takes to say: about a third for Chinese, between the two for Japanese and Korean; Chinese and Japanese sentences are joined without spaces. The duration check uses separate speaking rates for Han characters, kana and hangul.
- **The pipeline log shows how the GPUs shared a chapter**: chunks, batches and busy seconds per device.
- **Short chapters use every GPU (`chapter_pipeline.py`)**: a device takes at most an even share of the chunks still queued, where the first device used to take a whole batch — all of a short chapter — and the second T4 sat idle. An idle device also waits while another is mid-batch, so work handed back by a failed device is picked up instead of lost.
- **Reference-voice preprocessing** rewritten: loudness (LUFS) target applied last, edge-only silence trim with fades, peak-relative soft gate, pause-learned noise profile, downsample to 24 kHz mono, and a report with warnings.
- **Config schema 7**: `voice_preset`, `tts_options`, `redo_chapters`, `batch_size`, `pack_sentences`, `normalize_speech_text`, `verify_*`.

### 🗑️ Removed
- **VibeVoice provider** — it could never synthesize (it called a method the model does not have). Saved configs naming it fall back to Qwen3-TTS.

### 🐛 Fixed
- **Sentence breaks in other languages (`text_processing.py`, `splitter.rs`)**: an opening quote no longer stays at the end of the previous sentence in Chinese and Japanese, a French closing guillemet stays with its sentence, abbreviations such as "ул.", "г." and "डॉ." no longer end one, the Python fallback splits at the CJK full stop and the Devanagari danda, and over-long sentences are cut at full-width commas.
- **Dialogue dashes (`extractor_engine.py`, `normalize.rs`)**: the dash that opens a line of dialogue in French or Russian was turned into a comma (", Rentre vite !"); it is now dropped.
- **ASR verification in other languages (`chunk_verifier.py`)**: combining marks were stripped before comparing, which cut every Hindi, Thai or Arabic word into loose consonants and made any transcript look wrong. A transcript is now compared the way its script needs: Chinese by syllable sound and Japanese by reading (with the optional `pypinyin` / `pykakasi`), so the recogniser's choice of homophone, traditional character, kanji or kana is not an error; Hindi and other scripts with varying spelling by character; English and similar by word, as before.
- **Fish Audio S2 Pro failed on every chunk in reduced precision (`fish_provider.py`)**: upstream builds its codec under `torch.inference_mode`; converting it to the model's precision outside that mode left the weights unusable ("Inference tensors do not track version counter"). Seen on a T4, where the codec runs in bfloat16.
- **IndexTTS could not load (`requirements/tts-indextts.txt`, `indextts_provider.py`)**: upstream's bundled DAC code imports `audiotools` while the model loads; `descript-audiotools` and what it imports are now part of the install steps.
- **Transcript given as a file path (`pipeline.py`)**: a path to a `.txt` file in the voice-transcript field was taken as the words spoken in the clip, which made cloning engines cut chunks short, ramble or go silent. The file is now read, and a transcript whose length cannot match the clip is reported before synthesis starts.
- **ASR chunk verification (`chunk_verifier.py`)**: Whisper no longer carries context between its 30-second windows, which made one misheard window spoil the rest of a transcript.
- **Loudness normalisation never boosted quiet audio (`audiobook_rust/src/audio/master.rs`)**: the true peak (a linear amplitude) was added to a dB gain, so any chapter needing a boost was turned *down* ~1.5 dB instead of reaching the LUFS target.
- **Real errors hidden behind `sub_future` `UnboundLocalError` (`pipeline.py`)**: any failure, cancel or empty chapter before the subtitle stage reported "cannot access local variable 'sub_future'" and was marked failed.
- **Cancellation (`pipeline.py`)**: `asyncio.CancelledError` from the chapter pipeline is now handled; a cancelled chapter stays `pending` instead of `failed`.
- **Chapter order (`pipeline.py`)**: outputs (and the single-file concat) are ordered by chapter number, not by filename — "Chapter 10" no longer precedes "Chapter 2".
- **Pronunciation map ignored (`pipeline.py`)**: fixes are now applied to pre-split sentences, which is what is actually synthesized.
- **Encoder settings (`pipeline.py`)**: `bitrate_kbps` is honoured for MP3/OGG (`-q:a` used to override it); the Python fallback no longer emits 48/96/192 kHz files; sample-rate and stereo choices no longer change playback speed; Rust-encoded MP3s get ID3 tags.
- **Resume cache (`pipeline.py`, `chapter_pipeline.py`)**: chunk WAVs survive a failed or cancelled chapter, retries reuse them, and a fingerprint discards them if the text, voice or TTS settings changed.
- **Stale TTS settings (`chapter_pipeline.py`, `pipeline.py`, `qwen_provider.py`)**: pooled and preview providers now use the current run's config; the voice-prompt cache is keyed on file contents and transcript.
- **Qwen provider (`qwen_provider.py`)**: reference auto-transcription no longer fails silently on an invalid Whisper kwarg; `top_k`, `repetition_penalty` and `seed` are passed through; CustomVoice/VoiceDesign no longer require a voice file.
- **Text normalisation (`extractor_engine.py`, `text_processing.py`, Rust)**: removed the rules that split ordinary words ("room w as") and glued single capitals ("Vitamin Cis"); headings, HTML comments, entities and scene breaks are stripped; chapter titles such as "About a Boy" or "End of the Road" are no longer dropped as front matter.
- **Rust sentence splitter panic on non-ASCII text (`splitter.rs`)**: lengths are counted in characters and slices land on UTF-8 boundaries.
- **Extraction (`extractor_engine.py`, `text_extractor.py`)**: the OCR checkbox is honoured and an OCR failure no longer aborts extraction; inline HTML tags no longer split sentences in the fallback parser; TXT encoding is detected.
- **API/UI (`api/server.py`, `api/worker.py`, `app.py`, `cli.py`)**: the WebSocket delivers the file list before closing, so API-mode runs no longer end in "No output files generated"; paths containing commas survive; restoring a progress JSON no longer blocks Generate; the chapter cache matches titles exactly; the CLI keeps every exported setting.

## [v1.5.0] - 2026-09-24

### ⚡ Added
- **API Secret Authentication & Rate Limiting (`api/server.py`)**: Added shared secret authentication via `ABM_API_SECRET` (`x-api-key` header, `Authorization: Bearer` header, and WebSocket query parameter `?api_key=`), safeguarding endpoints against unauthorized GPU task submission and data access. Implemented thread-safe in-memory sliding-window rate limiting on `/api/v1/generate` (30 req/min) and `/api/v1/preprocess` (60 req/min).
- **Task Lifecycle Eviction (`api/worker.py`, `api/server.py`)**: Added bounded task history management with `_MAX_COMPLETED_TASKS = 50` and `evict_old_tasks()`, eliminating unbounded memory growth from accumulated terminal jobs in long-running server sessions.
- **Dedicated Dev & Test Requirements (`requirements-dev.txt`)**: Created `requirements-dev.txt` specifying `pytest` and `pytest-asyncio>=0.23` to support async test execution in fresh environments.

### 🐛 Fixed
- **F5-TTS Temporary Voice Reference Cleanup (`f5tts_provider.py`)**: Wrapped inference in a `try/finally` block to ensure temporary WAV reference files created from `bytes` payloads are reliably unlinked, preventing disk exhaustion during multi-chapter synthesis.
- **Qwen Voice Reference Bounded LRU Cache (`qwen_provider.py`)**: Capped `_VOICE_REF_CACHE` at 8 entries with thread-safe LRU eviction and automatic deletion of evicted WAV files from disk.
- **Whisper ASR Pipeline Caching (`qwen_provider.py`)**: Cached the Whisper automatic speech recognition pipeline at the `QwenTTSProvider` instance level, preventing expensive repeated model re-downloads and GPU memory allocations across synthesis calls.
- **Chapter Pipeline Cancellation Sentinels (`chapter_pipeline.py`)**: Enforced dispatch of termination sentinels (`None`) to all active device queues in `_stage_a_worker`'s `finally` block, preventing Stage B workers from blocking indefinitely when cancellation occurs mid-dispatch.
- **Stage B Cancellation Dropped Chunk Accounting (`chapter_pipeline.py`)**: Ensured accumulated in-flight batches abandoned on cancellation emit `_StageError` events to Stage C before exiting, maintaining consistent chunk counting and clean `CancelledError` propagation.
- **Stage C Active Worker Counting (`chapter_pipeline.py`)**: Fixed worker exit tracking in Stage C so active worker counts are decremented strictly upon receiving worker `None` sentinels rather than chunk-level `_StageError` events.
- **Subtitle Generation Synchronization (`pipeline.py`)**: Synchronized chapter subtitle generation futures inside `_process_chapter`'s `finally` block before temporary working directory removal, preventing silent `FileNotFoundError` subtitle export failures.
- **Expanded Output Format Validation (`pipeline.py`)**: Updated `_validate_config()` to accept all media formats supported by `ffmpeg_utils.py` (`mp3`, `wav`, `flac`, `m4b`, `m4a`, `aac`, `ogg`, `webm`, `mp4`, `mov`).
- **FFmpeg Concat Demuxer Single-Quote Escaping (`pipeline.py`)**: Added proper shell escaping for single quotes and spaces in file paths written to `concat_list.txt` in single-file audio concatenation mode.
- **Atomic Progress Summary Finalization (`pipeline.py`, `progress_io.py`)**: Wrapped `_finalize_progress_file()` read-modify-write logic with `_WRITE_LOCK` to eliminate race conditions with concurrent chapter completion updates, and documented multi-process locking limitations.
- **Cover Image Conversion Logging (`pipeline.py`, `app.py`)**: Replaced bare `except:` blocks with specific `Exception` handling and logging, preventing silent cover art embedding failures and ensuring system-level exceptions are not masked.
- **Preflight bfloat16 Capability Detection (`preflight.py`)**: Fixed `torch.cuda.is_bf16_supported()` call signature by scoping checks under `torch.cuda.device(idx)` context rather than passing device index as an argument.
- **Lazy PyTorch Import in Worker (`api/worker.py`)**: Defended module-level imports in `api.worker` by scoping `import torch` inside `_get_active_gpu_count()`, avoiding import crashes in lightweight CLI/test environments.

---

## [v1.4.1] - 2026-09-22

### ⚡ Added
- **Multi-TTS Provider Serialization & Pool Eviction (`gpu_pool.py`, `api/worker.py`)**: Implemented `_wait_for_other_providers_idle()` to serialize task execution when switching between different TTS backends (Qwen, VibeVoice, F5-TTS). Added automatic eviction and cleanup of inactive provider pools in `GPUPoolManager.get_pool()` to eliminate multi-engine GPU OOM errors.
- **Voice Studio Preview Provider Cache (`pipeline.py`)**: Added cached preview provider instance in `preview_tts()` with automatic resource cleanup upon engine or model variant changes, preventing VRAM leaks on preview clicks.
- **Conditional GPU Warmup (`server.py`, `colab_prerun_check.py`, `kaggle_prerun_check.py`)**: Added `ABM_SKIP_GPU_WARMUP` environment variable check to bypass blocking warmup passes during testing or fast startup.

### 🐛 Fixed
- **Notebook & Tunnel Launch Stability (`AudiobookMaker_Colab.ipynb`, `AudiobookMaker_Kaggle.ipynb`)**: Fixed notebook startup hangs by starting Pinggy tunnels before Gradio initialization, generating SSH keys automatically, setting `inline=False` and `share=True`, and executing via `sys.executable`.
- **Gradio Type Annotation NameErrors (`app.py`, `cli.py`)**: Imported `Any` in `app.py` and `AudiobookConfig` in `cli.py` to fix runtime `get_type_hints()` failures during Gradio interface mounting.
- **Document Extractor Edge Case Robustness (`text_extractor.py`, `extractor_engine.py`)**: Hardened EPUB/PDF text parsing, TOC boundary detection, and fallback handling for missing chapter titles.

---

## [v1.4.0] - 2026-09-21

### 🛡️ Security
- **VibeVoice Model Allowlist (`vibevoice_provider.py`)**: Enforced strict allowlist validation on HuggingFace model identifiers for VibeVoice to prevent remote code execution via `trust_remote_code=True`.
- **Path Traversal Containment (`worker.py`, `filename_sanitizer.py`)**: Anchored output directories in `api/worker.py` to `ABM_OUTPUT_BASE` and sanitized audio file extension suffixes in `filename_sanitizer.py`.
- **Archive & Document Decompression Guards (`text_extractor.py`, `extractor_engine.py`)**: Added zip bomb safety checks, PDF decompression limits, and table span clamping in document extractors.
- **CLI Configuration Sanitization (`cli.py`)**: Hardened CLI settings loading against maliciously crafted configuration parameters and unapproved provider injection.
- **Session Progress Isolation (`app.py`)**: Isolated Gradio session progress ownership and removed server path reflection in UI responses.
- **Security Documentation & QA Reports (`docs/security/`, `docs/qa/`)**: Added security hardening proposals, vulnerability remediation reports, and automated reproduction tests across 9 test modules.

### ⚡ Added & Fixed
- **Qwen3 Audio Quality & Concatenation (`pipeline.py`, `qwen_provider.py`)**: Refined cross-chunk audio stitching to eliminate audible gaps and pops between generated segments.
- **FastAPI Lifespan Migration (`api/server.py`)**: Replaced deprecated `@app.on_event("startup")` and `@app.on_event("shutdown")` hooks with modern `lifespan` context manager.

---

## [v1.3.1] - 2026-08-31

### ⚡ Added
- **Automated QA & Benchmark Framework (`tests/`)**: Built comprehensive automated testing infrastructure including audio quality validator (`AudioValidator`), synthetic audio fixture generators, document parsing test suites, and pipeline speed benchmarking tools (`tests/benchmark_pipeline.py`).
- **Kaggle Validation Suite (`tests/kaggle/`)**: Added end-to-end multi-GPU verification suite, smoke tests, and automated Markdown report generation for Kaggle dual T4 environments.
- **EBU R128 Pure-Python Mastering (`pipeline.py`, `ffmpeg_utils.py`)**: Implemented pure-Python audio mastering fallback with EBU R128 loudness normalization and true peak limiting when Rust acceleration is unavailable.
- **File-Based Voice Preprocessing (`voice_preprocessor.py`)**: Enabled `voice_preprocess()` to accept file paths directly and automatically persist cleaned reference audio to disk.

---

## [v1.3.0] - 2026-08-11

### ⚡ Added
- **Fail-Safe Chapter Retry System with Exponential Backoff (`pipeline.py`, `progress_io.py`)**: Added automatic chapter-level retry handling (up to `config.max_chapter_retries` attempts) with CUDA memory flushing (`torch.cuda.empty_cache()` and `gc.collect()`) and backoff delays. Implemented `update_chapter_retry()` in `progress_io.py` to persist `retry_count` and `last_error` (truncated to 500 chars) under thread-safe write locks. Added an automatic end-of-run retry pass for remaining failed chapters (`retry_failed_at_end`).
- **New Modular TTS Providers (`VibeVoice-1.5B` & `F5-TTS`)**: Added `VibeVoiceTTSProvider` (`vibevoice_provider.py`) for `bezzam/VibeVoice-1.5B-hf` and `F5TTSProvider` (`f5tts_provider.py`) for zero-shot voice cloning. Updated `get_tts_provider()` registry and exported all providers in `audiobook_factory.tts_providers`.
- **Single-GPU VRAM Optimization & Dynamic Batching (`gpu_pool.py`)**: Added `vram_headroom_gb: float = 2.0` field to `AudiobookConfig` and updated `GPUDetector.suggest_batch_size()` to calculate free usable VRAM after headroom subtraction. Added preflight warnings for low-VRAM GPUs (<= 8.5 GB).
- **Completion Validation & Top-Level Summary Finalizer (`pipeline.py`)**: Implemented `_mark_chapter_completed()` with minimum WAV file size guard (`_MINIMUM_CHAPTER_WAV_BYTES = 10_000`) and disk existence checks. Implemented `_finalize_progress_file()` writing a top-level `generation_summary` block (completed/failed counts, failed chapter numbers, ISO timestamp) to progress JSON.
- **WebSocket Keep-Alive & Session End Event (`server.py`, `worker.py`, `app.py`)**: Added a 15-second background `_ws_keepalive()` ping loop in `api/server.py`, extended WebSocket connection closure grace period to 3.0s, and broadcasted `session_end` events from `api/worker.py`. Updated Gradio client loop (`app.py`) to process `session_end` and ignore `ping` messages.
- **API Production Hardening & Graceful Shutdown (`server.py`, `worker.py`)**: Added `@app.on_event("shutdown")` in FastAPI server to flag active task cancellation tokens, allow checkpoint flushing, and shut down `GPUPoolManager`. Isolated task execution in `_run_task_safely()` in `api/worker.py`.

- **Prominent TTS Provider Selection in Voice Studio (`app.py`)**: Moved `tts_provider_dd` (`qwen`, `vibevoice`, `f5tts`) to Tab 3 (Voice Studio) with dynamic UI group toggles for Qwen parameters vs VibeVoice / F5-TTS provider info boxes.
- **Audio Encoding Controls (`app.py`, `pipeline.py`)**: Added `sample_rate_dd` (22050–48000 Hz), `bitrate_dd` (64–320 kbps), and `channels_radio` (Mono/Stereo) to Tab 4 (Advanced) and `AudiobookConfig`. Updated FFmpeg command builders in `pipeline.py` to enforce `-ar`, `-ac`, and `-b:a` encoding flags across lossy/lossless audio format outputs.
- **Advanced TTS Tuning Parameters (`AudiobookConfig`, `app.py`)**: Added `repetition_penalty` (default: 1.05), `top_k` (default: 50), `speed` (default: 1.0), `nfe_step` (default: 32), and `seed` (default: -1) to `AudiobookConfig`, provider backends (`qwen`, `vibevoice`, `f5tts`), and Tab 4 UI controls.
- **Provider Preflight Dependency Checks (`preflight.py`)**: Added automatic preflight checking for optional provider dependencies (`f5_tts` and `protobuf`).

### 🔄 Changed
- Bumped `_CONFIG_SCHEMA_VERSION` from `5` to `6` in `audiobook_factory/pipeline.py`.
- Updated Gradio UI TTS Provider dropdown in `app.py` to support `qwen`, `vibevoice`, and `f5tts`.
- Extended unit test suite in `tests/test_hardening.py` to 53 tests covering completion guards, summary finalization, retry persistence, provider resolution, VRAM headroom batch suggestions, and audio encoding options.

---

## [v1.2.0] - 2026-08-05

### ⚡ Added
- **Kaggle & Cloud Pre-Flight Environment Validator (`preflight.py`)**: Created an automated pre-flight validator module (`run_preflight_checks()`) that completes comprehensive environment validation in under 30 seconds before any model loading or generation begins. Checks Python version (3.10+), PyTorch, CUDA device count, per-device bfloat16 capability, transformers API (`BitsAndBytesConfig`), soundfile, FFmpeg, voice reference integrity/existence, and Python 3.12 dict view picklability. Raises `PreflightError` immediately at startup with actionable error reports to prevent wasting GPU time mid-generation.
- **Voice Reference Safety & Session Hash Caching**: Added `BaseTTSProvider._validate_voice_ref()` boundary validation and implemented module-level SHA-256 hash caching (`_VOICE_REF_CACHE`) in `_resolve_voice_ref()` to prevent writing redundant temp WAV files on disk across hundreds of chunk synthesis calls.
- **Stale Checkpoint Detection & Corrupted Chunk Recovery**: Added `_validate_chunk_file()` (minimum size guard = 1,000 bytes) in `chapter_pipeline.py`. Missing or corrupted chunk WAV files are automatically detected, logged with a stale checkpoint warning at chapter start, and routed for re-synthesis.
- **Python 3.12 Pickling Compatibility**: Extended `_sanitize_dict_keys()` in `qwen_provider.py` to recursively sanitize `dict_keys`, `dict_values`, and `dict_items` view objects across model and generation configs for Python 3.12 strict pickling compatibility.
- **Expanded Hardening Test Suite (`tests/test_hardening.py`)**: Added test coverage for pre-flight check functions, `PreflightResult`, `PreflightError`, voice reference validation, chunk file validation, and Python 3.12 dict view sanitization.

### 🔄 Changed
- Integrated `run_preflight_checks()` into `run_pipeline()` entry point before GPU pool provider factory allocation.
- Passed `recommended_dtype` ("float16" or "bfloat16") determined by pre-flight check directly into TTS provider initialization (`dtype_override`).

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
