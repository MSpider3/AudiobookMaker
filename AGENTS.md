# AGENTS.md — AudiobookMaker

## Stack

- **Python 3.10+**, PyTorch, HuggingFace transformers (Qwen3-TTS)
- **Rust** (PyO3 via maturin) — optional audio hot-path; pure-Python fallbacks required
- **Gradio** (UI), **FastAPI + WebSocket** (API backend), **argparse CLI**

## Build / Test / Run

```bash
./install.sh                          # venv + deps + Rust extension
./run.sh                              # API (:8000) + Gradio (:7860)

source venv/bin/activate
python start_api.py &                 # FastAPI only
python app.py                         # Gradio only
python cli.py <progress.json>         # headless

python -m pytest tests/ -v            # tests
cd audiobook_rust && maturin develop --release  # Rust only
```

## Code Conventions

- **snake_case** everywhere — files, functions, variables
- Module-private constants: `_UPPER_SNAKE` with type annotations (`_DEFAULT_MIN_VRAM_GB: float = 5.0`)
- Private helpers: `_leading_underscore`
- Classes: `PascalCase`; dataclasses for config objects (`AudiobookConfig`, `PreprocessConfig`)
- Docstrings: triple-quote block at top of every module and public class; numpy-ish param style
- `from __future__ import annotations` at top of all `audiobook_factory/` modules
- Logging via `logging.getLogger(__name__)`, not `print()` (except CLI banners)
- Type hints on all public function signatures

## Structure

```
app.py                  # Gradio UI (entry point)
cli.py                  # Headless CLI
start_api.py            # FastAPI launcher
api/server.py           # FastAPI routes + WebSocket
api/worker.py           # Async GPU task queue consumer
audiobook_factory/      # Core library
  pipeline.py           # Top-level orchestrator
  chapter_pipeline.py   # 3-stage per-chapter pipeline (A: text, B: GPU TTS, C: master)
  gpu_pool.py           # GPU detection, provider pool, device dispatch
  progress_io.py        # Atomic thread-safe JSON progress I/O
  text_extractor.py     # Public extraction API
  extractor_engine.py   # 5-phase AI extraction backend
  tts_providers/        # Provider plugin interface
    base_tts_provider.py
    qwen_provider.py
  voice_preprocessor.py # 7-step audio cleaning
audiobook_rust/         # Optional PyO3 Rust extension (maturin)
tests/                  # pytest suite
```

## Guardrails

- **ASK FIRST** before touching `gpu_pool.py` device dispatch or `_get_chapter_parallelism` — active bug area, changes cascade
- **ASK FIRST** before modifying `progress_io.py` locking / atomic-write — concurrency-critical
- **DO NOT** bump `_CONFIG_SCHEMA_VERSION` without updating `AudiobookConfig.from_dict()` migration
- **DO NOT** remove pure-Python fallbacks for Rust extensions — Colab/Kaggle depend on them
- **DO NOT** commit directly to `main`

## Full Context

See `PROJECT.md` in repo root for architecture decisions, priorities, and known bugs.
