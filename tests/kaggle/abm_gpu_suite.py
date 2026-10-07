"""
tests/kaggle/abm_gpu_suite.py
==============================
GPU test harness driven by ``AudiobookMaker_Kaggle_Test.ipynb``.

Each sub-command runs one group of checks and writes a JSON result file into
the results directory (``$ABM_RESULTS_DIR``, default ``kaggle_results/``), so
a failure in one group never hides the others and the whole run can be zipped
and handed back for diagnosis. Every command exits 0; pass/fail lives in the
JSON and in the final report.

Sub-commands
------------
env         Hardware, drivers and package versions.
unit        The pytest suite (mock provider, no GPU needed).
extraction  scan()/extract() on every fixture book against expected_chapters.json.
mastering   Rust extension: loudness target, Rust/Python text parity, GIL release.
make-voice  Create a real-speech narrator reference clip with a preset voice.
provider    Synthesize the test passage with one TTS engine through the real
            pipeline on all GPUs; measure speed, VRAM, loudness and word accuracy.
preset      Save the narrator as a voice preset, then narrate from the preset alone.
scaling     Same passage on 1 GPU and on all GPUs; reports the speed-up.
resume      Kill a run mid-chapter, resume it, confirm cached chunks are reused.
book        Extract a fixture book and produce an M4B with chapter markers.
cli         Drive cli.py for real: MOBI fixture -> M4B, dry run, provider listing, re-run.
report      Aggregate every result into REPORT.md and results.zip.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import queue
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import zipfile
from typing import Any, Callable

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

RESULTS_DIR: str = os.path.abspath(os.environ.get("ABM_RESULTS_DIR", os.path.join(_ROOT, "kaggle_results")))
ASSETS_DIR: str = os.path.abspath(os.environ.get("ABM_ASSETS_DIR", os.path.join(_ROOT, "tests", "kaggle", "assets")))
# Narrator clip and books the tests read; see tests/kaggle/assets/README.md.
_ASSET_BOOKS: str = os.path.join(ASSETS_DIR, "books")
_FIXTURES: str = (
    _ASSET_BOOKS if os.path.isdir(_ASSET_BOOKS)
    else os.path.join(_ROOT, "tests", "fixtures", "source_documents")
)
_SYNTHETIC_VOICE: str = os.path.join(_ROOT, "tests", "fixtures", "audio", "synthetic_voice_reference.wav")

_MAX_WORD_ERROR_RATE: float = 0.30
_LOUDNESS_TOLERANCE_LU: float = 2.0
_LOG_TAIL_LINES: int = 60

# Original prose, written for this test. Plain narration with one line of
# dialogue and two numerals, so word accuracy can be scored against it.
TEST_PARAGRAPHS: tuple[str, ...] = (
    "The harbour town woke slowly that morning. Fishing boats rocked against the "
    "quay while gulls argued over scraps, and the smell of salt drifted up the narrow streets.",
    "Marta had kept the lighthouse for eleven years. She knew every step of the spiral "
    "stair, every creak of the old iron door, and the exact moment the lamp needed trimming.",
    "\"Will the storm reach us tonight?\" asked the boy from the bakery. He stood in the "
    "doorway with flour on his sleeves, looking out at the grey line of the horizon.",
    "She did not answer at once. The wind had changed since noon, and the barometer in "
    "the hall had fallen further than she liked. Somewhere beyond the point, a bell was ringing.",
    "By evening the first rain arrived. It came in long silver sheets across the water, "
    "and the boats that had gone out at dawn turned for home one after another.",
    "Marta climbed the stair, lit the lamp, and set it turning. For the next 6 hours its "
    "beam swept the dark, steady as a heartbeat, until the last boat was safely in.",
)

VOICE_SAMPLE_TEXT: str = (
    "Good evening, and welcome. Tonight I will be reading a story about a lighthouse, "
    "a harbour town, and the keeper who watched over both for many years."
)


# ══════════════════════════════════════════════════════════════════════════════
# Result plumbing
# ══════════════════════════════════════════════════════════════════════════════

def _write_result(name: str, status: str, metrics: dict | None = None, error: str = "",
                  log_tail: list[str] | None = None, notes: list[str] | None = None) -> dict:
    """Writes one result JSON and prints a one-line summary."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    result = {
        "test": name,
        "status": status,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "metrics": metrics or {},
        "notes": notes or [],
        "error": error,
        "log_tail": (log_tail or [])[-_LOG_TAIL_LINES:],
    }
    path = os.path.join(RESULTS_DIR, f"{name}.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, ensure_ascii=False, indent=2, default=str)
    icon = {"pass": "✅", "fail": "❌", "skip": "⏭", "warn": "⚠️"}.get(status, "•")
    headline = error.splitlines()[0][:160] if error else ""
    print(f"\n{icon} [{name}] {status.upper()} {headline}")
    for key, value in (metrics or {}).items():
        if isinstance(value, (int, float, str, bool)) or value is None:
            print(f"     {key}: {value}")
    for note in notes or []:
        print(f"     note: {note}")
    return result


def _guarded(name: str, body: Callable[[], dict | None]) -> None:
    """Runs a test body; any exception becomes a failed result instead of a crash."""
    try:
        body()
    except BaseException as exc:  # includes SystemExit/KeyboardInterrupt from libraries
        _write_result(name, "fail", error=f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}")


def _gpu_summary() -> list[dict]:
    try:
        import torch
        if not torch.cuda.is_available():
            return []
        devices = []
        for index in range(torch.cuda.device_count()):
            free, total = torch.cuda.mem_get_info(index)
            devices.append({
                "device": f"cuda:{index}",
                "name": torch.cuda.get_device_name(index),
                "total_gb": round(total / 2**30, 2),
                "free_gb": round(free / 2**30, 2),
            })
        return devices
    except Exception as exc:
        return [{"error": str(exc)}]


def _decode_to_mono(path: str, rate: int = 16000):
    """Decodes any audio file to a mono float32 array at ``rate`` via FFmpeg."""
    import numpy as np
    import soundfile as sf

    with tempfile.TemporaryDirectory() as tmp:
        wav = os.path.join(tmp, "decoded.wav")
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", path, "-ac", "1", "-ar", str(rate), wav],
            check=True, capture_output=True,
        )
        samples, _ = sf.read(wav, dtype="float32")
    return np.ascontiguousarray(samples, dtype=np.float32)


def _loudness(samples, rate: int) -> float | None:
    try:
        import pyloudnorm
        return float(pyloudnorm.Meter(rate).integrated_loudness(samples))
    except Exception:
        return None


def _transcribe(samples, language: str = "English") -> str | None:
    """Transcribes with faster-whisper (preferred) or transformers Whisper. None if unavailable."""
    from audiobook_factory.chunk_verifier import ChunkVerifier

    verifier = ChunkVerifier("asr", language=language, asr_model=os.environ.get(
        "ABM_ASR_MODEL", "openai/whisper-large-v3-turbo"))
    try:
        # Whisper works on 30 s windows; fed a whole chapter it can repeat or
        # invent text across window borders, which reads as a high error rate.
        parts = [verifier._transcribe(window, 16000) for window in _speech_windows(samples, 16000)]
        text = " ".join(part for part in parts if part).strip()
        return text if (text or verifier._asr is not None) else None
    finally:
        verifier.close()


def _speech_windows(samples, rate: int, max_seconds: float = 24.0, min_seconds: float = 8.0) -> list:
    """Cuts *samples* into windows of at most *max_seconds*, each cut at the quietest point."""
    import numpy as np

    frame = max(1, int(rate * 0.02))
    frames = len(samples) // frame
    if frames * frame <= max_seconds * rate:
        return [samples]
    energy = np.sqrt(np.mean(np.square(samples[:frames * frame].reshape(frames, frame)), axis=1))
    # Quietness over 0.3 s, so a cut lands inside a pause and not between two syllables.
    span = 15
    smoothed = np.convolve(energy, np.ones(span) / span, mode="same")
    windows, start = [], 0
    low, high = int(min_seconds / 0.02), int(max_seconds / 0.02)
    while frames - start > high:
        cut = start + low + int(np.argmin(smoothed[start + low:start + high]))
        windows.append(samples[start * frame:cut * frame])
        start = cut
    windows.append(samples[start * frame:])
    return [window for window in windows if len(window) >= rate // 4]


def _spoken_text(config, paragraphs: tuple[str, ...]) -> str:
    """The text the TTS engine was actually asked to say (after normalisation)."""
    from audiobook_factory.pipeline import _prepare_speech_text
    return " ".join(_prepare_speech_text(p, config) for p in paragraphs)


# ══════════════════════════════════════════════════════════════════════════════
# env / unit / extraction / mastering
# ══════════════════════════════════════════════════════════════════════════════

def cmd_env(_args) -> None:
    def body():
        versions = {}
        for module in ("torch", "torchaudio", "transformers", "accelerate", "qwen_tts", "gradio",
                       "fastapi", "soundfile", "librosa", "pyloudnorm", "faster_whisper",
                       "bitsandbytes", "flash_attn", "docling", "easyocr", "mutagen", "numpy"):
            try:
                versions[module] = getattr(__import__(module), "__version__", "installed")
            except Exception as exc:
                versions[module] = f"missing ({type(exc).__name__})"
        try:
            import audiobook_rust
            rust = sorted(n for n in dir(audiobook_rust) if not n.startswith("_"))
        except Exception as exc:
            rust = f"missing ({exc})"
        ffmpeg = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True).stdout.splitlines()[:1]
        disk = shutil.disk_usage(_ROOT)
        try:
            ram_gb = round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 2**30, 1)
        except (ValueError, OSError):
            ram_gb = None
        commit = subprocess.run(["git", "-C", _ROOT, "log", "-1", "--format=%h %s"],
                                capture_output=True, text=True).stdout.strip()
        branch = subprocess.run(["git", "-C", _ROOT, "rev-parse", "--abbrev-ref", "HEAD"],
                                capture_output=True, text=True).stdout.strip()
        gpus = _gpu_summary()
        voice_file, voice_transcript = _default_voice()
        hf_home = os.path.expanduser(os.environ.get("HF_HOME", "~/.cache/huggingface"))
        probe = hf_home if os.path.isdir(hf_home) else os.path.expanduser("~")
        from audiobook_factory.tts_providers import list_providers
        providers = {
            info.name: {"display": info.display_name, "license": info.license,
                        "commercial_use": info.commercial_use, "default_model": info.default_model,
                        "min_vram_gb": info.min_vram_gb, "batch": info.supports_batch}
            for info in list_providers()
        }
        notes = []
        if len(gpus) < 2:
            notes.append(f"{len(gpus)} GPU(s) visible — pick 'GPU T4 x2' in Session options to test multi-GPU.")
        transcript_problem = _transcript_problem(voice_file, voice_transcript)
        status = "fail" if transcript_problem else ("pass" if gpus else "warn")
        _write_result("env", status, {
            "python": platform.python_version(), "platform": platform.platform(),
            "branch": branch, "commit": commit, "gpus": gpus, "gpu_count": len(gpus),
            "ram_gb": ram_gb, "disk_free_gb": round(disk.free / 2**30, 1),
            "model_cache_free_gb": round(shutil.disk_usage(probe).free / 2**30, 1),
            "narrator_voice": os.path.relpath(voice_file, _ROOT) if voice_file.startswith(_ROOT) else voice_file,
            "narrator_transcript_words": len(voice_transcript.split()),
            "books_dir": os.path.relpath(_FIXTURES, _ROOT),
            "ffmpeg": ffmpeg[0] if ffmpeg else "missing", "rust_extension": rust,
            "versions": versions, "providers": providers,
        }, error=transcript_problem, notes=notes)
    _guarded("env", body)


def cmd_unit(_args) -> None:
    def body():
        junit = os.path.join(RESULTS_DIR, "pytest_junit.xml")
        os.makedirs(RESULTS_DIR, exist_ok=True)
        # The unit suite tests logic with the mock engine; it is pinned to CPU
        # (tests/conftest.py does the same) so GPU count cannot change results.
        env = dict(os.environ, ABM_SKIP_GPU_WARMUP="1", CUDA_VISIBLE_DEVICES="")
        started = time.time()
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "tests", "--ignore=tests/kaggle", "-q",
             "-p", "no:cacheprovider", "--tb=short", "-rfE", f"--junitxml={junit}"],
            cwd=_ROOT, env=env, capture_output=True, text=True, timeout=3600,
        )
        lines = (proc.stdout + proc.stderr).splitlines()
        with open(os.path.join(RESULTS_DIR, "pytest_output.txt"), "w", encoding="utf-8") as fh:
            fh.write("\n".join(lines))
        failed = [line for line in lines if line.startswith(("FAILED", "ERROR"))]
        if failed:
            print("\nFailed tests:")
            for line in failed[:60]:
                print("  " + line[:300])
            errors = [line for line in lines if line.startswith("E  ")]
            print("\nFirst error lines:")
            for line in errors[:40]:
                print("  " + line[:300])
        _write_result(
            "unit", "pass" if proc.returncode == 0 else "fail",
            {"summary": lines[-1] if lines else "", "seconds": round(time.time() - started, 1),
             "failed_count": len(failed), "failed": failed[:40]},
            error="" if proc.returncode == 0 else f"pytest exited {proc.returncode}",
            log_tail=lines,
        )
    _guarded("unit", body)


def cmd_extraction(_args) -> None:
    def body():
        from audiobook_factory.text_extractor import extract, scan

        expected_path = os.path.join(_FIXTURES, "expected_chapters.json")
        expected = {}
        if os.path.exists(expected_path):
            with open(expected_path, encoding="utf-8") as fh:
                expected = json.load(fh)
        files = sorted(
            f for f in os.listdir(_FIXTURES)
            if f.lower().endswith((".epub", ".pdf", ".docx", ".odt", ".txt", ".mobi", ".azw3"))
        )
        per_file, failures = {}, []
        for name in files:
            path = os.path.join(_FIXTURES, name)
            entry: dict[str, Any] = {}
            try:
                started = time.time()
                scanned = scan(path)
                chapters, _cover = extract(path, enable_ocr=False, log_fn=lambda _m: None)
                titles = [c.title for c in chapters]
                text = "\n".join(c.text for c in chapters)
                entry.update({
                    "seconds": round(time.time() - started, 2),
                    "scan_chapters": len(getattr(scanned, "chapters", []) or []),
                    "chapters": titles, "characters": len(text),
                })
                want = expected.get(name) or {}
                want_titles = want.get("chapters") if isinstance(want, dict) else want
                if want_titles is not None and titles != want_titles:
                    entry["expected_chapters"] = want_titles
                    failures.append(f"{name}: chapters {titles} != expected {want_titles}")
                if isinstance(want, dict):
                    for phrase in want.get("must_contain", []):
                        if phrase not in text:
                            failures.append(f"{name}: missing phrase {phrase!r}")
                    for phrase in want.get("must_not_contain", []):
                        if phrase in text:
                            failures.append(f"{name}: unwanted phrase {phrase!r} was extracted")
                if not chapters or not text.strip():
                    failures.append(f"{name}: no text extracted")
            except Exception as exc:
                entry["error"] = f"{type(exc).__name__}: {exc}"
                failures.append(f"{name}: {type(exc).__name__}: {exc}")
            per_file[name] = entry
        _write_result(
            "extraction", "pass" if not failures else "fail",
            {"files": len(files), "failures": failures, "per_file": per_file},
            error="; ".join(failures[:5]),
            notes=[] if expected else ["expected_chapters.json not found — only checked that text was extracted."],
        )
    _guarded("extraction", body)


def cmd_mastering(_args) -> None:
    def body():
        import numpy as np
        import soundfile as sf

        notes, failures, metrics = [], [], {}
        try:
            import audiobook_rust
            has_rust = hasattr(audiobook_rust, "master_audio")
        except ImportError:
            has_rust = False
        metrics["rust_extension"] = has_rust
        if not has_rust:
            _write_result("mastering", "warn", metrics,
                          notes=["Rust extension not built — the pure-Python fallbacks are in use."])
            return

        rate = 24000
        t = np.arange(rate * 20) / rate
        with tempfile.TemporaryDirectory() as tmp:
            for label, amplitude in (("quiet", 0.02), ("loud", 0.6)):
                src, out = os.path.join(tmp, f"{label}.wav"), os.path.join(tmp, f"{label}_out.wav")
                signal_ = (amplitude * np.sin(2 * np.pi * 220 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t)))
                sf.write(src, signal_.astype(np.float32), rate, subtype="PCM_16")
                audiobook_rust.master_audio([src], out, 0.0, rate, -18.0, -1.5, 64)
                mastered, _ = sf.read(out, dtype="float32")
                lufs = _loudness(mastered, rate)
                metrics[f"{label}_out_lufs"] = None if lufs is None else round(lufs, 2)
                if lufs is not None and abs(lufs + 18.0) > 1.0:
                    failures.append(f"{label} input mastered to {lufs:.1f} LUFS, target -18")

            # GIL must be released while mastering.
            long_src = os.path.join(tmp, "long.wav")
            sf.write(long_src, (0.1 * np.sin(np.arange(rate * 300) / 20)).astype(np.float32), rate, subtype="PCM_16")
            ticks, stop = [], threading.Event()

            def ticker():
                while not stop.is_set():
                    ticks.append(time.monotonic())
                    time.sleep(0.005)
            thread = threading.Thread(target=ticker)
            thread.start()
            started = time.monotonic()
            audiobook_rust.master_audio([long_src], os.path.join(tmp, "long.mp3"), 0.0, rate, -18.0, -1.5, 64)
            elapsed = time.monotonic() - started
            stop.set()
            thread.join()
            during = [x for x in ticks if started <= x <= started + elapsed]
            metrics["master_5min_seconds"] = round(elapsed, 2)
            metrics["other_thread_ticks_during_master"] = len(during)
            if elapsed > 0.5 and len(during) < 5:
                failures.append("Rust mastering holds the GIL (other threads did not run)")

        from audiobook_factory import text_processing
        from audiobook_factory.extractor_engine import TextNormalizer
        normalizer = TextNormalizer()
        samples = [
            "The room was dark and the year after that he left.",
            "Vitamin C is good. Plan B was ready. Class D students arrived.",
            "## Chapter 3\n\n<!-- image -->\n\nSmith &amp; Sons sold it.\n\n* * *\n\nNext scene.",
            "T he sun rose.\n\nO nce upon a time.",
        ]
        mismatches = []
        for sample in samples:
            via_rust = audiobook_rust.clean_text(sample, "T", False)
            text = normalizer._remove_duplicate_title("T", sample)
            text = normalizer._fix_broken_lines(text)
            text = normalizer._strip_noise(text)
            via_python = text_processing._python_normalize_text(text).strip()
            if via_rust != via_python:
                mismatches.append({"input": sample, "rust": via_rust, "python": via_python})
        metrics["rust_python_text_mismatches"] = mismatches
        if mismatches:
            failures.append(f"{len(mismatches)} Rust/Python text normalisation mismatch(es)")
        for name, text in (("cyrillic", "Это очень длинное предложение " * 20), ("em_dash", "word — " * 120)):
            try:
                chunks = audiobook_rust.split_sentences(text, 399)
                if max(len(c) for c in chunks) > 399:
                    failures.append(f"Rust splitter exceeded max_len on {name}")
            except BaseException as exc:
                failures.append(f"Rust splitter crashed on {name}: {exc}")
        _write_result("mastering", "pass" if not failures else "fail", metrics,
                      error="; ".join(failures), notes=notes)
    _guarded("mastering", body)


# ══════════════════════════════════════════════════════════════════════════════
# Synthesis through the real pipeline
# ══════════════════════════════════════════════════════════════════════════════

def _parse_options(pairs: list[str] | None) -> dict:
    options: dict[str, Any] = {}
    for pair in pairs or []:
        key, _, raw = pair.partition("=")
        value: Any = raw
        if raw.lower() in ("true", "false"):
            value = raw.lower() == "true"
        else:
            for cast in (int, float):
                try:
                    value = cast(raw)
                    break
                except ValueError:
                    continue
        options[key.strip()] = value
    return options


def _transcript_text(value: str) -> str:
    """The transcript itself: *value*, or the contents of the text file it names."""
    value = (value or "").strip()
    if value and "\n" not in value and len(value) < 1024:
        for candidate in (value, os.path.join(_ROOT, value)):
            if os.path.isfile(candidate):
                with open(candidate, encoding="utf-8-sig", errors="replace") as fh:
                    return fh.read().strip()
    return value


def _transcript_problem(voice_file: str, transcript: str) -> str:
    """Why *transcript* cannot be what is said in *voice_file*, or ""."""
    if not transcript or not transcript.isascii():
        return ""
    try:
        import soundfile as sf
        info = sf.info(voice_file)
        seconds = info.frames / float(info.samplerate)
    except Exception:
        return ""
    words = len(transcript.split())
    if seconds >= 1.0 and not 0.5 <= words / seconds <= 6.0:
        return (f"the narrator transcript has {words} word(s) for a {seconds:.0f}-second clip; it must be "
                "the exact words spoken in the clip (or a path to a .txt file holding them)")
    return ""


def _sidecar_transcript(voice_file: str) -> str:
    sidecar = os.path.splitext(voice_file)[0] + ".txt"
    if os.path.exists(sidecar):
        with open(sidecar, encoding="utf-8") as fh:
            return fh.read().strip()
    return ""


def _asset_voice() -> str:
    """The narrator clip committed in the assets folder, or ""."""
    voice_dir = os.path.join(ASSETS_DIR, "voice")
    if os.path.isdir(voice_dir):
        clips = sorted(name for name in os.listdir(voice_dir) if name.lower().endswith((".wav", ".flac")))
        if clips:
            return os.path.join(voice_dir, clips[0])
    return ""


def _default_voice() -> tuple[str, str]:
    """Returns (voice_file, transcript) used as the narrator reference.

    Order: ``$ABM_VOICE_FILE``, the clip in the assets folder, a clip made by
    the ``make-voice`` command, and last the synthetic tone fixture (which is
    not speech and only keeps the plumbing testable).
    """
    explicit = os.environ.get("ABM_VOICE_FILE", "").strip()
    if explicit and os.path.exists(explicit):
        given = _transcript_text(os.environ.get("ABM_VOICE_TRANSCRIPT", ""))
        return explicit, given or _sidecar_transcript(explicit)
    asset = _asset_voice()
    if asset:
        return asset, _sidecar_transcript(asset)
    generated = os.path.join(RESULTS_DIR, "voice", "reference.wav")
    if os.path.exists(generated):
        with open(os.path.splitext(generated)[0] + ".txt", encoding="utf-8") as fh:
            return generated, fh.read().strip()
    return _SYNTHETIC_VOICE, ""


def _build_config(args, out_dir: str, **overrides):
    from audiobook_factory.pipeline import AudiobookConfig

    voice_file, transcript = _default_voice()
    if getattr(args, "voice", None):
        voice_file, transcript = args.voice, getattr(args, "transcript", "") or ""
    settings = dict(
        book_title="ABM Test Book", author="AudiobookMaker", language=args.language,
        output_dir=out_dir, output_format=args.format,
        voice_file="" if getattr(args, "no_voice", False) else voice_file,
        voice_transcript="" if getattr(args, "no_voice", False) else transcript,
        tts_provider_name=args.name, tts_instruct=args.instruct or "", tts_timbre=args.timbre or "",
        tts_options=_parse_options(args.option), verify_chunks=args.verify,
        gpu_count=args.gpus, seed=args.seed, export_lrc=True, export_srt=True,
        max_chapter_retries=0, retry_failed_at_end=False,
    )
    if args.model:
        settings["tts_model_name"] = args.model
    if getattr(args, "voice_preset", None):
        settings["voice_preset"] = args.voice_preset
    if args.batch_size:
        settings["batch_size"] = args.batch_size
    settings.update(overrides)
    # The shared sampling defaults are tuned for Qwen; test every engine at
    # the operating point its authors recommend.
    from audiobook_factory.tts_providers import apply_recommended_settings
    apply_recommended_settings(args.name, settings)
    return AudiobookConfig(**settings)


def _run_pipeline(config, chapters, cancel=None) -> tuple[list[str], list[tuple[float, str]], BaseException | None, float]:
    """Runs run_pipeline, echoing logs. Returns (files, timestamped logs, error, wall seconds)."""
    from audiobook_factory.pipeline import run_pipeline

    log_q: queue.Queue = queue.Queue()
    prog_q: queue.Queue = queue.Queue()
    logs: list[tuple[float, str]] = []
    done = threading.Event()

    def drain():
        while not done.is_set() or not log_q.empty():
            try:
                message = log_q.get(timeout=0.2)
            except queue.Empty:
                continue
            logs.append((time.monotonic(), message))
            print(message, flush=True)
            while not prog_q.empty():
                prog_q.get_nowait()

    thread = threading.Thread(target=drain, daemon=True)
    thread.start()
    started = time.monotonic()
    files, error = [], None
    try:
        files = run_pipeline(config, chapters, log_q, prog_q, cancel)
    except BaseException as exc:
        error = exc
        traceback.print_exc()
    wall = time.monotonic() - started
    done.set()
    thread.join(timeout=5)
    return files, logs, error, wall


def _reset_gpu_state() -> None:
    try:
        from audiobook_factory.gpu_pool import GPUPoolManager
        GPUPoolManager.instance().shutdown()
        import gc
        import torch
        gc.collect()
        if torch.cuda.is_available() and torch.cuda.is_initialized():
            torch.cuda.empty_cache()
            for index in range(torch.cuda.device_count()):
                try:
                    torch.cuda.reset_peak_memory_stats(index)
                except Exception:
                    pass  # a device nothing has touched yet has no stats to reset
    except Exception as exc:
        print(f"(could not reset GPU state: {exc})")


def _measure_run(config, paragraphs: tuple[str, ...], tag: str, score_words: bool = True) -> tuple[str, dict, str, list[str], list[str]]:
    """Runs one synthesis job and measures it. Returns (status, metrics, error, logs, notes)."""
    import torch
    from audiobook_factory.chunk_verifier import error_rate, expected_seconds
    from audiobook_factory.text_extractor import ExtractedChapter
    from audiobook_factory.tts_providers import provider_info

    info = provider_info(config.tts_provider_name)
    notes: list[str] = []
    if not info.commercial_use:
        notes.append(f"{info.display_name} weights are licensed {info.license}: non-commercial use only.")

    _reset_gpu_state()
    shutil.rmtree(config.output_dir, ignore_errors=True)
    chapter = ExtractedChapter(num=1, title="The Lighthouse", text="\n\n".join(paragraphs), sentences=[])
    files, logs, error, wall = _run_pipeline(config, [chapter])
    messages = [message for _, message in logs]

    metrics: dict[str, Any] = {
        "provider": config.tts_provider_name, "engine": info.display_name,
        "model": config.tts_model_name if config.tts_provider_name == "qwen" else info.default_model,
        "license": info.license, "commercial_use": info.commercial_use,
        "wall_seconds": round(wall, 1), "gpu_count_requested": config.gpu_count,
        "verify": config.verify_chunks, "format": config.output_format,
    }
    if torch.cuda.is_available():
        metrics["vram_peak_gb"] = {
            f"cuda:{i}": round(torch.cuda.max_memory_allocated(i) / 2**30, 2)
            for i in range(torch.cuda.device_count())
        }
    for message in messages:
        if "Devices :" in message:
            metrics["devices"] = message.split("Devices :", 1)[1].strip()
        if "Device share:" in message:
            metrics["device_share"] = message.split("Device share:", 1)[1].strip()
        if "TTS chunks" in message:
            metrics["chunks"] = message.strip()

    start = next((stamp for stamp, message in logs if "Synthesizing" in message), None)
    end = next((stamp for stamp, message in logs if "of audio." in message), None)
    if start is not None and end is not None:
        metrics["synthesis_seconds"] = round(end - start, 1)

    if error is not None:
        return "fail", metrics, f"{type(error).__name__}: {error}", messages, notes
    if not files or not os.path.exists(files[0]):
        failed = [m for m in messages if "failed" in m.lower() or "❌" in m]
        return "fail", metrics, (failed[-1] if failed else "no output file produced"), messages, notes

    out_path = files[0]
    samples = _decode_to_mono(out_path)
    audio_seconds = len(samples) / 16000.0
    spoken = _spoken_text(config, paragraphs)
    expected = expected_seconds(spoken, config.speed) + config.para_pause * (len(paragraphs) - 1)
    metrics.update({
        "output_file": os.path.basename(out_path),
        "output_bytes": os.path.getsize(out_path),
        "audio_seconds": round(audio_seconds, 1),
        "expected_seconds": round(expected, 1),
    })
    synthesis = metrics.get("synthesis_seconds")
    if synthesis:
        metrics["realtime_factor"] = round(synthesis / max(audio_seconds, 0.1), 2)
        metrics["speed_x_realtime"] = round(audio_seconds / max(synthesis, 0.1), 2)
    lufs = _loudness(samples, 16000)
    metrics["loudness_lufs"] = None if lufs is None else round(lufs, 2)

    problems: list[str] = []
    if not 0.5 * expected <= audio_seconds <= 2.2 * expected:
        problems.append(f"audio is {audio_seconds:.0f}s, expected about {expected:.0f}s")
    if lufs is not None and abs(lufs - config.lufs) > _LOUDNESS_TOLERANCE_LU:
        problems.append(f"loudness {lufs:.1f} LUFS, target {config.lufs}")

    if score_words:
        try:
            transcript = _transcribe(samples, config.language)
        except Exception as exc:
            transcript = None
            notes.append(f"ASR scoring failed: {exc}")
        if transcript is None:
            notes.append("ASR unavailable — word accuracy not scored.")
        else:
            rate = error_rate(spoken, transcript)
            metrics["word_error_rate"] = round(rate, 3)
            metrics["transcript_head"] = transcript[:240]
            if rate > _MAX_WORD_ERROR_RATE:
                problems.append(f"word error rate {rate:.0%} (limit {_MAX_WORD_ERROR_RATE:.0%})")

    try:
        with open(os.path.join(config.output_dir, "generation_progress.json"), encoding="utf-8") as fh:
            entry = json.load(fh)["chapters"][0]
        metrics["chapter_status"] = entry.get("status")
        metrics["flagged_chunks"] = entry.get("flagged_chunks", [])
        if entry.get("status") != "completed":
            problems.append(f"chapter status is {entry.get('status')!r}")
    except Exception as exc:
        notes.append(f"could not read progress file: {exc}")

    subtitles = [f for f in os.listdir(config.output_dir) if f.endswith((".lrc", ".srt"))]
    metrics["subtitle_files"] = sorted(subtitles)

    visible = torch.cuda.device_count() if torch.cuda.is_available() else 0
    if visible > 1 and config.gpu_count != 1:
        share = metrics.get("device_share", "")
        if "cuda:1" not in metrics.get("devices", ""):
            problems.append("only one GPU was used although two are visible")
        elif share and any(part.strip().endswith(" 0 chunk(s)") for part in share.split(",")):
            notes.append(f"one GPU synthesized nothing ({share}) — chapter may be too short to share.")

    samples_dir = os.path.join(RESULTS_DIR, "samples")
    os.makedirs(samples_dir, exist_ok=True)
    shutil.copyfile(out_path, os.path.join(samples_dir, f"{tag}{os.path.splitext(out_path)[1]}"))
    return ("pass" if not problems else "fail"), metrics, "; ".join(problems), messages, notes


def _paragraphs(count: int) -> tuple[str, ...]:
    """The first ``count`` test paragraphs, repeating the passage when more are asked for."""
    count = max(1, int(count))
    repeats = count // len(TEST_PARAGRAPHS) + 1
    return (TEST_PARAGRAPHS * repeats)[:count]


def _device_and_dtype() -> tuple[str, str | None]:
    """The first GPU and the precision the pipeline itself would pick for it.

    Uses the same pre-flight recommendation as ``run_pipeline``. Forcing
    float16 here made Qwen3-TTS sample from NaN logits on a T4 ("probability
    tensor contains either inf, nan or element < 0").
    """
    import torch
    if not torch.cuda.is_available():
        return "cpu", "float32"
    try:
        from audiobook_factory.preflight import run_preflight_checks
        return "cuda:0", run_preflight_checks(voice_ref=None, check_voice_ref=False).recommended_dtype
    except Exception as exc:
        print(f"(pre-flight dtype check failed: {exc}; letting the engine choose)")
        return "cuda:0", None


def cmd_provider(args) -> None:
    tag = args.tag or f"provider_{args.name}"

    def body():
        paragraphs = _paragraphs(args.paragraphs)
        config = _build_config(args, os.path.join(RESULTS_DIR, "work", tag))
        status, metrics, error, logs, notes = _measure_run(config, paragraphs, tag, score_words=not args.no_asr)
        _write_result(tag, status, metrics, error, logs, notes)
        _reset_gpu_state()
    _guarded(tag, body)


def cmd_make_voice(args) -> None:
    def body():
        from audiobook_factory.pipeline import AudiobookConfig
        from audiobook_factory.tts_providers import get_tts_provider

        out_dir = os.path.join(RESULTS_DIR, "voice")
        os.makedirs(out_dir, exist_ok=True)
        wav_path = os.path.join(out_dir, "reference.wav")
        _reset_gpu_state()
        config = AudiobookConfig(
            tts_provider_name="qwen", tts_model_name=args.model,
            tts_timbre=args.timbre, language=args.language, seed=7,
        )
        device, dtype = _device_and_dtype()
        provider = get_tts_provider("qwen", config, device=device, dtype_override=dtype)
        started = time.monotonic()
        try:
            provider.synthesize(VOICE_SAMPLE_TEXT, b"", wav_path)
        finally:
            provider.cleanup()
        with open(os.path.splitext(wav_path)[0] + ".txt", "w", encoding="utf-8") as fh:
            fh.write(VOICE_SAMPLE_TEXT)
        samples = _decode_to_mono(wav_path)
        _write_result("make_voice", "pass", {
            "voice_file": wav_path, "seconds": round(len(samples) / 16000.0, 1),
            "model": args.model, "speaker": args.timbre,
            "generation_seconds": round(time.monotonic() - started, 1),
        }, notes=["Every clone test below uses this clip as the narrator reference."])
        _reset_gpu_state()
    _guarded("make_voice", body)


def cmd_preset(args) -> None:
    tag = args.tag or f"preset_{args.name}"

    def body():
        import torch
        from audiobook_factory.tts_providers import get_tts_provider, provider_info

        info = provider_info(args.name)
        if not info.supports_voice_preset:
            _write_result(tag, "skip", notes=[f"{info.display_name} has no voice-preset support."])
            return
        voice_file, transcript = _default_voice()
        preset_path = os.path.join(RESULTS_DIR, "voice", f"{args.name}_preset.pt")
        os.makedirs(os.path.dirname(preset_path), exist_ok=True)
        _reset_gpu_state()
        config = _build_config(args, os.path.join(RESULTS_DIR, "work", tag))
        device, dtype = _device_and_dtype()
        provider = get_tts_provider(args.name, config, device=device, dtype_override=dtype)
        try:
            described = provider.save_voice_preset(preset_path, voice_file, transcript=transcript or None)
        finally:
            provider.cleanup()
        preset_path = described.get("path", preset_path) if isinstance(described, dict) else preset_path

        # Narrate from the preset alone: no reference clip, no transcript.
        args.voice_preset, args.no_voice = preset_path, True
        config = _build_config(args, os.path.join(RESULTS_DIR, "work", tag))
        status, metrics, error, logs, notes = _measure_run(
            config, _paragraphs(args.paragraphs), tag, score_words=not args.no_asr)
        metrics["preset"] = {k: v for k, v in (described or {}).items()
                             if isinstance(v, (str, int, float, bool)) or v is None}
        metrics["preset_bytes"] = os.path.getsize(preset_path) if os.path.exists(preset_path) else 0
        _write_result(tag, status, metrics, error, logs, notes)
        _reset_gpu_state()
    _guarded(tag, body)


def cmd_scaling(args) -> None:
    tag = args.tag or f"scaling_{args.name}"

    def body():
        import torch
        visible = torch.cuda.device_count() if torch.cuda.is_available() else 0
        if visible < 2:
            _write_result(tag, "skip", {"gpu_count": visible}, notes=["Needs two GPUs."])
            return
        paragraphs = TEST_PARAGRAPHS * max(1, args.repeat)
        runs = {}
        for label, gpus in (("one_gpu", 1), ("all_gpus", 0)):
            args.gpus = gpus
            config = _build_config(args, os.path.join(RESULTS_DIR, "work", f"{tag}_{label}"), verify_chunks="duration")
            status, metrics, error, _logs, _notes = _measure_run(config, paragraphs, f"{tag}_{label}", score_words=False)
            runs[label] = {"status": status, "error": error, **{
                k: metrics.get(k) for k in ("synthesis_seconds", "audio_seconds", "speed_x_realtime", "device_share", "vram_peak_gb")
            }}
        one, both = runs["one_gpu"].get("synthesis_seconds"), runs["all_gpus"].get("synthesis_seconds")
        speedup = round(one / both, 2) if one and both else None
        ok = all(r["status"] == "pass" for r in runs.values()) and speedup is not None
        notes = []
        if speedup is not None and speedup < 1.4:
            notes.append("Less than 1.4x from the second GPU — check 'device_share' and batch sizes.")
        _write_result(tag, "pass" if ok else "fail", {"speedup": speedup, **runs},
                      error="" if ok else "; ".join(r["error"] for r in runs.values() if r["error"]), notes=notes)
        _reset_gpu_state()
    _guarded(tag, body)


def cmd_resume(args) -> None:
    tag = args.tag or f"resume_{args.name}"

    def body():
        work = os.path.join(RESULTS_DIR, "work", tag)
        shutil.rmtree(work, ignore_errors=True)
        chunk_dir = os.path.join(work, ".temp_chunks", "abm_ch001")
        command = [sys.executable, os.path.abspath(__file__), "provider", "--name", args.name,
                   "--tag", f"{tag}_first_run", "--paragraphs", str(len(TEST_PARAGRAPHS) * 2),
                   "--no-asr", "--verify", "off", "--keep-work-dir", work, "--batch-size", "2"]
        if args.model:
            command += ["--model", args.model]
        child = subprocess.Popen(command, cwd=_ROOT, env=dict(os.environ, ABM_RESULTS_DIR=RESULTS_DIR))
        deadline = time.monotonic() + args.kill_timeout
        seen = 0
        while time.monotonic() < deadline and child.poll() is None:
            if os.path.isdir(chunk_dir):
                seen = len([f for f in os.listdir(chunk_dir) if f.startswith("chunk_ch_1_") and f.endswith(".wav")])
                if seen >= args.kill_after_chunks:
                    break
            time.sleep(1.0)
        finished_early = child.poll() is not None
        if not finished_early:
            child.send_signal(signal.SIGKILL)
            child.wait()
        if finished_early or seen < 1:
            _write_result(tag, "skip", {"chunks_seen": seen}, notes=[
                "The first run finished (or produced nothing) before it could be interrupted."])
            return

        config = _build_config(args, work, verify_chunks="off", batch_size=2)
        paragraphs = _paragraphs(len(TEST_PARAGRAPHS) * 2)
        from audiobook_factory.text_extractor import ExtractedChapter
        _reset_gpu_state()
        chapter = ExtractedChapter(num=1, title="The Lighthouse", text="\n\n".join(paragraphs), sentences=[])
        files, logs, error, wall = _run_pipeline(config, [chapter])
        messages = [m for _, m in logs]
        cached_line = next((m for m in messages if "cached" in m and "pending" in m), "")
        match = re.search(r"\((\d+) cached", cached_line)
        cached = int(match.group(1)) if match else 0
        ok = error is None and bool(files) and cached >= 1
        _write_result(tag, "pass" if ok else "fail", {
            "chunks_on_disk_when_killed": seen, "chunks_reused_on_resume": cached,
            "resume_line": cached_line.strip(), "resume_wall_seconds": round(wall, 1),
        }, error="" if ok else (str(error) if error else "no cached chunks were reused"), log_tail=messages)
        _reset_gpu_state()
    _guarded(tag, body)


def cmd_book(args) -> None:
    tag = args.tag or f"book_{args.name}"

    def body():
        from audiobook_factory.text_extractor import extract

        book = args.book or os.path.join(_FIXTURES, "dummy_book.epub")
        chapters, cover = extract(book, enable_ocr=False, log_fn=print)
        chapters = [c for c in chapters if c.text.strip()][: args.max_chapters]
        for chapter in chapters:
            # Keep the run short: the point is the end-to-end path, not the length.
            chapter.text = "\n\n".join(chapter.text.split("\n\n")[: args.max_paragraphs])
            chapter.sentences = []
        work = os.path.join(RESULTS_DIR, "work", tag)
        cover_path = ""
        if cover:
            os.makedirs(work, exist_ok=True)
            cover_path = os.path.join(RESULTS_DIR, "work", f"{tag}_cover.jpg")
            with open(cover_path, "wb") as fh:
                fh.write(cover)
        config = _build_config(args, work, output_format="m4b", single_file_mode=True,
                               book_title="The Dummy Book", cover_image=cover_path or None)
        _reset_gpu_state()
        shutil.rmtree(work, ignore_errors=True)
        files, logs, error, wall = _run_pipeline(config, chapters)
        messages = [m for _, m in logs]
        if error is not None or not files:
            _write_result(tag, "fail", {"wall_seconds": round(wall, 1)},
                          error=str(error) if error else "no output file", log_tail=messages)
            return
        probe = json.loads(subprocess.run(
            ["ffprobe", "-v", "error", "-show_chapters", "-show_format", "-show_streams", "-of", "json", files[0]],
            capture_output=True, text=True, check=True).stdout)
        markers = [c.get("tags", {}).get("title", "") for c in probe.get("chapters", [])]
        want = [c.title for c in chapters]
        has_cover = any(s.get("codec_type") == "video" for s in probe.get("streams", []))
        problems = []
        if markers != want:
            problems.append(f"chapter markers {markers} != chapters {want}")
        if cover_path and not has_cover:
            problems.append("cover image was not embedded")
        samples_dir = os.path.join(RESULTS_DIR, "samples")
        os.makedirs(samples_dir, exist_ok=True)
        shutil.copyfile(files[0], os.path.join(samples_dir, f"{tag}.m4b"))
        _write_result(tag, "pass" if not problems else "fail", {
            "book": os.path.basename(book), "chapters": want, "chapter_markers": markers,
            "cover_embedded": has_cover, "tags": probe.get("format", {}).get("tags", {}),
            "duration_seconds": round(float(probe["format"].get("duration", 0)), 1),
            "wall_seconds": round(wall, 1), "output_file": os.path.basename(files[0]),
        }, error="; ".join(problems), log_tail=messages)
        _reset_gpu_state()
    _guarded(tag, body)


def cmd_cli(args) -> None:
    tag = args.tag or f"cli_{args.name}"

    def body():
        book = args.book or os.path.join(_FIXTURES, "dummy_book.mobi")
        work = os.path.join(RESULTS_DIR, "work", tag)
        shutil.rmtree(work, ignore_errors=True)
        voice_file, transcript = _default_voice()
        cli = os.path.join(_ROOT, "cli.py")
        metrics: dict[str, Any] = {"book": os.path.basename(book)}
        problems: list[str] = []

        listing = subprocess.run([sys.executable, cli, "--list-providers"], cwd=_ROOT,
                                 capture_output=True, text=True, timeout=300)
        metrics["list_providers_exit"] = listing.returncode
        if listing.returncode != 0:
            problems.append("--list-providers failed")

        command = [sys.executable, cli, "--book", book, "--local", "--provider", args.name,
                   "--chapters", args.chapters, "--single-file", "--output-format", "m4b",
                   "--output-dir", work, "--verify", args.verify, "--seed", str(args.seed)]
        if args.model:
            command += ["--tts-model-name", args.model]
        if voice_file:
            command += ["--voice-file", voice_file]
        if transcript:
            command += ["--voice-transcript", transcript]

        dry = subprocess.run(command + ["--dry-run"], cwd=_ROOT, capture_output=True, text=True, timeout=600)
        metrics["dry_run_exit"] = dry.returncode
        if dry.returncode != 0:
            problems.append(f"--dry-run exited {dry.returncode}: {(dry.stdout + dry.stderr)[-300:]}")

        started = time.monotonic()
        run = subprocess.run(command, cwd=_ROOT, capture_output=True, text=True,
                             timeout=args.timeout_min * 60)
        output = (run.stdout + run.stderr).splitlines()
        print("\n".join(output[-60:]))
        metrics.update(exit_code=run.returncode, wall_seconds=round(time.monotonic() - started, 1))
        if run.returncode != 0:
            problems.append(f"cli.py exited {run.returncode}")
        books = [f for f in os.listdir(work) if f.endswith(".m4b")] if os.path.isdir(work) else []
        metrics["output_files"] = books
        if len(books) != 1:
            problems.append(f"expected one .m4b, found {books}")
        else:
            probe = json.loads(subprocess.run(
                ["ffprobe", "-v", "error", "-show_chapters", "-show_format", "-of", "json",
                 os.path.join(work, books[0])], capture_output=True, text=True, check=True).stdout)
            metrics["chapter_markers"] = [c.get("tags", {}).get("title", "") for c in probe.get("chapters", [])]
            metrics["duration_seconds"] = round(float(probe["format"].get("duration", 0)), 1)
            if len(metrics["chapter_markers"]) < 2:
                problems.append("combined file has fewer than two chapter markers")
            samples_dir = os.path.join(RESULTS_DIR, "samples")
            os.makedirs(samples_dir, exist_ok=True)
            shutil.copyfile(os.path.join(work, books[0]), os.path.join(samples_dir, f"{tag}.m4b"))

        # A second run of a finished book must do nothing and still succeed.
        rerun = subprocess.run(command, cwd=_ROOT, capture_output=True, text=True, timeout=1200)
        metrics["rerun_exit"] = rerun.returncode
        metrics["rerun_skipped"] = "Already complete" in (rerun.stdout + rerun.stderr)
        if rerun.returncode != 0 or not metrics["rerun_skipped"]:
            problems.append("re-running a finished book did not skip cleanly")

        _write_result(tag, "pass" if not problems else "fail", metrics,
                      error="; ".join(problems), log_tail=output)
    _guarded(tag, body)


# ══════════════════════════════════════════════════════════════════════════════
# Report
# ══════════════════════════════════════════════════════════════════════════════

def cmd_report(_args) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    results = []
    for name in sorted(os.listdir(RESULTS_DIR)):
        if name.endswith(".json") and name != "summary.json":
            try:
                with open(os.path.join(RESULTS_DIR, name), encoding="utf-8") as fh:
                    results.append(json.load(fh))
            except (OSError, ValueError) as exc:
                results.append({"test": name, "status": "fail", "error": f"unreadable result: {exc}", "metrics": {}})

    order = {"fail": 0, "warn": 1, "skip": 2, "pass": 3}
    results.sort(key=lambda r: (order.get(r.get("status"), 9), r.get("test", "")))
    counts: dict[str, int] = {}
    for result in results:
        counts[result.get("status", "?")] = counts.get(result.get("status", "?"), 0) + 1

    lines = ["# AudiobookMaker — Kaggle GPU test report", "",
             f"Generated {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}", "",
             "**Totals:** " + ", ".join(f"{count} {status}" for status, count in sorted(counts.items())), ""]
    env = next((r for r in results if r.get("test") == "env"), None)
    if env:
        m = env.get("metrics", {})
        lines += ["## Environment", "",
                  f"- Branch / commit: `{m.get('branch')}` — {m.get('commit')}",
                  f"- Python {m.get('python')}, torch {m.get('versions', {}).get('torch')}, "
                  f"transformers {m.get('versions', {}).get('transformers')}",
                  f"- GPUs: " + (", ".join(f"{g.get('name')} ({g.get('total_gb')} GB)" for g in m.get("gpus", [])) or "none"),
                  f"- Rust extension: {m.get('rust_extension')}", ""]

    lines += ["## Results", "", "| Test | Status | Key numbers | Problem |", "|---|---|---|---|"]
    for result in results:
        m = result.get("metrics", {})
        keys = []
        for key in ("speed_x_realtime", "realtime_factor", "audio_seconds", "synthesis_seconds",
                    "word_error_rate", "loudness_lufs", "device_share", "vram_peak_gb", "speedup",
                    "chunks_reused_on_resume", "summary", "files", "failed_count"):
            if key in m and m[key] not in (None, "", [], {}):
                keys.append(f"{key}={m[key]}")
        problem = (result.get("error") or "").splitlines()[0][:200] if result.get("error") else ""
        lines.append(f"| {result.get('test')} | {result.get('status')} | {'; '.join(keys)} | {problem} |")
    lines.append("")
    for result in results:
        if result.get("status") in ("fail", "warn") or result.get("notes"):
            lines += [f"### {result.get('test')} — {result.get('status')}", ""]
            for note in result.get("notes", []):
                lines.append(f"- note: {note}")
            if result.get("error"):
                lines += ["", "```", result["error"][:3000], "```"]
            if result.get("status") == "fail" and result.get("log_tail"):
                lines += ["", "Last log lines:", "", "```", *result["log_tail"][-25:], "```"]
            lines.append("")

    report_path = os.path.join(RESULTS_DIR, "REPORT.md")
    with open(report_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    with open(os.path.join(RESULTS_DIR, "summary.json"), "w", encoding="utf-8") as fh:
        json.dump({"counts": counts, "results": results}, fh, ensure_ascii=False, indent=2, default=str)

    zip_path = os.path.join(os.path.dirname(RESULTS_DIR), "abm_test_results.zip")
    archived = 0
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as archive:
        for folder, dirs, files in os.walk(RESULTS_DIR):
            if folder == RESULTS_DIR:
                # Intermediate pipeline output; the samples folder has the audio.
                # Compared by name inside the results folder: the folder itself
                # may live under a path such as /kaggle/working.
                dirs[:] = [name for name in dirs if name != "work"]
            for name in sorted(files):
                full = os.path.join(folder, name)
                archive.write(full, os.path.relpath(full, os.path.dirname(RESULTS_DIR)))
                archived += 1
    size_mb = os.path.getsize(zip_path) / 2**20
    print("\n".join(lines))
    print(f"\nReport: {report_path}\nArchive to send back: {zip_path} ({archived} files, {size_mb:.1f} MB)")
    if archived < 2:
        raise SystemExit(f"The results archive is empty ({archived} file) — nothing was collected.")


# ══════════════════════════════════════════════════════════════════════════════
# CLI
# ══════════════════════════════════════════════════════════════════════════════

def _add_synthesis_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--name", required=True, help="TTS provider key, e.g. qwen, indextts, moss.")
    parser.add_argument("--model", default="", help="Model id (default: the provider's default).")
    parser.add_argument("--tag", default="", help="Result file name (default: derived from the command).")
    parser.add_argument("--language", default="English")
    parser.add_argument("--format", default="mp3")
    parser.add_argument("--gpus", type=int, default=0, help="GPUs to use; 0 = all.")
    parser.add_argument("--verify", default="duration", choices=["off", "duration", "asr"])
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--batch-size", type=int, default=0)
    parser.add_argument("--instruct", default="")
    parser.add_argument("--timbre", default="")
    parser.add_argument("--voice", default="", help="Reference clip (default: the generated one).")
    parser.add_argument("--transcript", default="")
    parser.add_argument("--voice-preset", default="")
    parser.add_argument("--no-voice", action="store_true", help="Run without a reference clip (preset / designed voices).")
    parser.add_argument("--option", action="append", default=[], metavar="KEY=VALUE",
                        help="Provider option (repeatable).")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name, handler in (("env", cmd_env), ("unit", cmd_unit), ("extraction", cmd_extraction),
                          ("mastering", cmd_mastering), ("report", cmd_report)):
        sub.add_parser(name).set_defaults(handler=handler)

    voice = sub.add_parser("make-voice")
    voice.add_argument("--model", default="Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice")
    voice.add_argument("--timbre", default="ryan")
    voice.add_argument("--language", default="English")
    voice.set_defaults(handler=cmd_make_voice)

    provider = sub.add_parser("provider")
    _add_synthesis_args(provider)
    provider.add_argument("--paragraphs", type=int, default=len(TEST_PARAGRAPHS))
    provider.add_argument("--no-asr", action="store_true")
    provider.add_argument("--keep-work-dir", default="", help=argparse.SUPPRESS)
    provider.set_defaults(handler=cmd_provider)

    preset = sub.add_parser("preset")
    _add_synthesis_args(preset)
    preset.add_argument("--paragraphs", type=int, default=3)
    preset.add_argument("--no-asr", action="store_true")
    preset.set_defaults(handler=cmd_preset)

    scaling = sub.add_parser("scaling")
    _add_synthesis_args(scaling)
    scaling.add_argument("--repeat", type=int, default=2)
    scaling.set_defaults(handler=cmd_scaling)

    resume = sub.add_parser("resume")
    _add_synthesis_args(resume)
    resume.add_argument("--kill-after-chunks", type=int, default=3)
    resume.add_argument("--kill-timeout", type=float, default=900.0)
    resume.set_defaults(handler=cmd_resume)

    book = sub.add_parser("book")
    _add_synthesis_args(book)
    book.add_argument("--book", default="")
    book.add_argument("--max-chapters", type=int, default=3)
    book.add_argument("--max-paragraphs", type=int, default=3)
    book.set_defaults(handler=cmd_book)

    cli = sub.add_parser("cli")
    _add_synthesis_args(cli)
    cli.add_argument("--book", default="")
    cli.add_argument("--chapters", default="1-2")
    cli.add_argument("--timeout-min", type=float, default=45.0)
    cli.set_defaults(handler=cmd_cli)

    args = parser.parse_args()
    if getattr(args, "keep_work_dir", ""):
        # Used by the resume test: the child must write into the directory the parent watches.
        original = _build_config

        def _pinned(a, out_dir, **overrides):
            return original(a, args.keep_work_dir, **overrides)
        globals()["_build_config"] = _pinned
        globals()["_measure_run"] = _measure_run_keep_dir
    args.handler(args)


def _measure_run_keep_dir(config, paragraphs, tag, score_words=True):
    """Variant of _measure_run for the resume child: never deletes the work dir contents on start."""
    from audiobook_factory.text_extractor import ExtractedChapter

    _reset_gpu_state()
    chapter = ExtractedChapter(num=1, title="The Lighthouse", text="\n\n".join(paragraphs), sentences=[])
    files, logs, error, wall = _run_pipeline(config, [chapter])
    status = "pass" if files and error is None else "fail"
    return status, {"wall_seconds": round(wall, 1)}, str(error or ""), [m for _, m in logs], []


if __name__ == "__main__":
    main()
