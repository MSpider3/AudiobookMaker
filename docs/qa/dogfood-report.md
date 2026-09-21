# Exploratory QA Dogfood Report: AudiobookMaker

## Executive Summary
- **Target Application**: AudiobookMaker (`MSpider3/AudiobookMaker`)
- **Testing Date**: 2026-09-21
- **Testing Scope**: Full System Integration (FastAPI REST API, Headless CLI, Gradio Web UI)
- **Total Issues Identified**: 0 Critical, 0 High, 0 Medium, 0 Low
- **Overall Assessment**: **PASS**. All core user journeys and security boundaries function reliably with zero console errors, zero runtime exceptions, and robust rejection of malicious payloads.

## Scope & Journeys Tested

### 1. Headless CLI (`cli.py`)
- **Help Command**: Verified complete help output (`cli.py --help`) and argument parser definitions.
- **Cache Management**: Tested `--clear-voice-cache` flag, successfully purging cached voice preprocessor files.
- **Security Validation**: Verified that progress JSON configuration loading strictly validates against allowlists for TTS providers, TTS model variants, and output directory boundaries.

### 2. FastAPI Orchestrator & Task Consumer (`api/server.py` & `api/worker.py`)
- **Service Healthcheck**: Verified `GET /api/v1/health` returned `200 OK` with active status and CPU/GPU hardware detection payload.
- **Path Traversal Defense**: Tested `POST /api/v1/generate` with path traversal payload (`output_dir: "../../etc"`). Confirmed worker immediately caught violation, marked task status as `failed`, and logged bounded error message without filesystem escapes.
- **Malicious Model Load Defense**: Tested `POST /api/v1/generate` with unapproved VibeVoice model (`evil-hacker/backdoored-model`). Confirmed task worker refused to load unapproved model and safely terminated warmup.
- **Task Lifecycle**: Verified task enqueueing, status polling (`GET /api/v1/tasks/{id}`), and clean cancellation.

### 3. Gradio Web UI (`app.py`)
- **Page Load & Rendering**: Verified clean initial render at `http://127.0.0.1:7860/` with zero JavaScript console errors.
- **Accordion Interactions**: Verified that "🔄 Resume from Progress JSON" accordion expands and collapses cleanly.
- **Tab Navigation**: Tested switching across 📚 Book, 🎧 Voice Preprocessing, 🎙️ Voice Studio, and ⚙️ Advanced tabs. All controls, sliders, dropzones, and text inputs adapt responsively.
- **Text Inputs**: Verified typing and reactive state updates in the "Book title" input field.
- **Session Isolation**: Verified that progress files and cached book chapters are strictly partitioned by session ID.

## Summary Table

| Journey / Area | Component | Result | Notes |
|---|---|---|---|
| CLI Interface | `cli.py` | PASS | Clean CLI execution, argument parsing, and settings validation. |
| API Orchestration | `api/server.py` | PASS | Healthcheck, task enqueueing, status tracking. |
| API Worker Security | `api/worker.py` | PASS | Path traversal and unapproved model rejection verified live. |
| Gradio UI Shell | `app.py` | PASS | Zero JS console errors, responsive layout across all tabs. |
| State Isolation | Session Handling | PASS | Cross-session progress overwrite and text leakage prevented. |

## Testing Notes & Limitations
- Hardware: Testing was conducted in a workstation environment with CPU device fallback enabled. GPU acceleration (CUDA) was detected as absent; the application appropriately displayed `GPU: CPU (slow)` in the UI status badge and executed CPU paths without crashing.
- Test artifacts and browser session recordings are preserved in the conversation artifact repository.
