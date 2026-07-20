"""
cli.py — AudiobookMaker Headless CLI
======================================
Generate audiobooks directly from a generation_progress.json config file,
without needing the Gradio web interface.

Usage examples
--------------
# Minimal — JSON has cached text, no book file needed for generation:
    python cli.py audiobook_output/MyBook/generation_progress.json

# Override book file (for cover extraction) and voice file:
    python cli.py generation_progress.json \\
        --book-path /path/to/book.epub \\
        --voice-file /path/to/narrator.wav

# Override generation settings:
    python cli.py generation_progress.json \\
        --worker-count 4 \\
        --output-format wav \\
        --output-dir ./my_audiobooks

# Force re-generate all chapters (ignore saved progress):
    python cli.py generation_progress.json --force-reprocess
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import signal
import sys
import threading
import time

# ── Ensure project root is on sys.path ───────────────────────────────────────
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


# ══════════════════════════════════════════════════════════════════════════════
# Console helpers
# ══════════════════════════════════════════════════════════════════════════════

_BOLD  = "\033[1m"
_GREEN = "\033[92m"
_CYAN  = "\033[96m"
_YELL  = "\033[93m"
_RED   = "\033[91m"
_RESET = "\033[0m"

def _h(text): return f"{_BOLD}{text}{_RESET}"
def _ok(text): return f"{_GREEN}{text}{_RESET}"
def _info(text): return f"{_CYAN}{text}{_RESET}"
def _warn(text): return f"{_YELL}{text}{_RESET}"
def _err(text): return f"{_RED}{text}{_RESET}"

def _print_banner():
    print()
    print(_h("━" * 50))
    print(_h("  📖  AudiobookMaker CLI"))
    print(_h("━" * 50))


# ══════════════════════════════════════════════════════════════════════════════
# FastAPI health check (same as app.py)
# ══════════════════════════════════════════════════════════════════════════════

def _is_api_healthy() -> bool:
    import requests
    import time
    urls = ["http://127.0.0.1:8000/api/v1/health", "http://localhost:8000/api/v1/health"]
    for _ in range(3):
        for url in urls:
            try:
                r = requests.get(url, timeout=4.0)
                if r.status_code == 200 and r.json().get("status") == "ok":
                    return True
            except Exception:
                pass
        time.sleep(0.5)
    return False


# ══════════════════════════════════════════════════════════════════════════════
# Argument parsing
# ══════════════════════════════════════════════════════════════════════════════

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cli.py",
        description="AudiobookMaker — headless CLI generation from a progress JSON config.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "config_json",
        help="Path to generation_progress.json (created by Gradio UI or Export Config button).",
    )
    p.add_argument(
        "--book-path",
        metavar="PATH",
        default=None,
        help="Override the book file path (EPUB, PDF, TXT, DOCX, ODT). "
             "Required if the JSON has no cached chapter text and you need to extract it. "
             "Also used for cover-image extraction.",
    )
    p.add_argument(
        "--voice-file",
        metavar="PATH",
        default=None,
        help="Override the narrator voice WAV file path.",
    )
    p.add_argument(
        "--output-dir",
        metavar="DIR",
        default=None,
        help="Override the output directory for generated audio files.",
    )
    p.add_argument(
        "--output-format",
        choices=["mp3", "flac", "wav", "m4b"],
        default=None,
        help="Override the output audio format.",
    )
    p.add_argument(
        "--worker-count",
        type=int,
        default=None,
        metavar="N",
        help="Override the number of parallel TTS workers.",
    )
    p.add_argument(
        "--device",
        default=None,
        choices=["cuda", "cpu"],
        help="Override the compute device.",
    )
    p.add_argument(
        "--tts-model-name",
        metavar="MODEL",
        default=None,
        help="Override the TTS model variant (e.g. Qwen/Qwen3-TTS-12Hz-1.7B-Base).",
    )
    p.add_argument(
        "--force-reprocess",
        action="store_true",
        default=False,
        help="Re-generate all chapters, ignoring any saved progress.",
    )
    return p


# ══════════════════════════════════════════════════════════════════════════════
# Load JSON config and merge CLI overrides
# ══════════════════════════════════════════════════════════════════════════════

def _load_config(args) -> tuple[dict, dict, list[dict]]:
    """
    Returns (meta, settings, chapters_raw).
    meta     — top-level keys: book_title, book_path, voice_file
    settings — the 'settings' sub-dict
    chapters_raw — the 'chapters' list (may include text/sentences)
    """
    path = os.path.abspath(args.config_json)
    if not os.path.exists(path):
        print(_err(f"❌ Config JSON not found: {path}"))
        sys.exit(1)

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    meta = {
        "book_title":  data.get("book_title", "Audiobook"),
        "book_path":   data.get("book_path", ""),
        "voice_file":  data.get("voice_file", ""),
        "cover_image_b64": data.get("cover_image_b64", ""),
        "_json_dir":   os.path.dirname(path),
    }
    settings = data.get("settings", {})
    chapters_raw = data.get("chapters", [])

    # Apply CLI overrides
    if args.book_path:
        meta["book_path"] = args.book_path
    if args.voice_file:
        meta["voice_file"] = args.voice_file
    if args.output_dir:
        settings["output_dir"] = args.output_dir
    if args.output_format:
        settings["output_format"] = args.output_format
    if args.worker_count is not None:
        settings["worker_count"] = args.worker_count
    if args.device:
        settings["device"] = args.device
    if args.tts_model_name:
        settings["tts_model_name"] = args.tts_model_name
    if args.force_reprocess:
        settings["force_reprocess"] = True

    return meta, settings, chapters_raw, path


# ══════════════════════════════════════════════════════════════════════════════
# Build AudiobookConfig from merged settings
# ══════════════════════════════════════════════════════════════════════════════

def _build_audiobook_config(meta: dict, settings: dict) -> "AudiobookConfig":
    from audiobook_factory.pipeline import AudiobookConfig

    book_title = meta["book_title"]
    book_path  = meta["book_path"]
    voice_file = meta["voice_file"]

    # Determine output dir — default to audiobook_output/<book_title>
    if settings.get("output_dir"):
        output_dir = settings["output_dir"]
    else:
        import re
        safe_title = re.sub(r'[\\/\*\?:"<>|]', "", book_title)
        output_dir = os.path.join(_ROOT, "audiobook_output", safe_title)

    cover_image = settings.get("cover_image", None)
    cover_b64 = settings.get("cover_image_b64", None) or meta.get("cover_image_b64", None)

    # Strategy 1: Decode base64 embedded cover data directly from JSON if present
    if cover_b64:
        try:
            import base64
            cov_bytes = base64.b64decode(cover_b64)
            os.makedirs(output_dir, exist_ok=True)
            cov_path = os.path.join(output_dir, "cover.jpg")
            with open(cov_path, "wb") as f_cov:
                f_cov.write(cov_bytes)
            cover_image = cov_path
        except Exception as e:
            print(f"⚠ Warning: Failed to decode base64 cover image: {e}")

    # Strategy 2: Validate existing cover_image path
    if cover_image and os.path.exists(cover_image):
        pass
    else:
        # Strategy 3: Check for cover.jpg / cover.png in output_dir
        cov_in_out = os.path.join(output_dir, "cover.jpg")
        cov_in_out_png = os.path.join(output_dir, "cover.png")
        if os.path.exists(cov_in_out):
            cover_image = cov_in_out
        elif os.path.exists(cov_in_out_png):
            cover_image = cov_in_out_png
        # Strategy 4: Check directory containing JSON file
        elif meta.get("_json_dir") and os.path.exists(os.path.join(meta["_json_dir"], "cover.jpg")):
            cover_image = os.path.join(meta["_json_dir"], "cover.jpg")
        # Strategy 5: Re-extract from EPUB if book_path exists
        elif book_path and os.path.exists(book_path):
            try:
                from audiobook_factory.text_extractor import scan
                scan_res = scan(book_path)
                if scan_res.cover_data:
                    os.makedirs(output_dir, exist_ok=True)
                    extracted_cov_path = os.path.join(output_dir, "cover.jpg")
                    with open(extracted_cov_path, "wb") as f_cov:
                        f_cov.write(scan_res.cover_data)
                    cover_image = extracted_cov_path
                else:
                    cover_image = None
            except Exception:
                cover_image = None
        else:
            cover_image = None

    return AudiobookConfig(
        book_title        = book_title,
        book_path         = book_path,
        author            = settings.get("author", "Unknown Author"),
        language          = settings.get("language", "English"),
        cover_image       = cover_image,
        output_dir        = output_dir,
        output_format     = settings.get("output_format", "mp3"),
        voice_file        = voice_file,
        tts_provider_name = settings.get("tts_provider_name", "qwen"),
        temperature       = float(settings.get("temperature", 0.3)),
        top_p             = float(settings.get("top_p", 0.8)),
        max_len           = int(settings.get("max_len", 399)),
        pause             = float(settings.get("pause", 0.5)),
        para_pause        = float(settings.get("para_pause", 1.2)),
        lufs              = int(settings.get("lufs", -18)),
        true_peak         = float(settings.get("true_peak", -1.5)),
        worker_count      = int(settings.get("worker_count", 1)),
        parallel_mode     = settings.get("parallel_mode", "chunks"),
        device            = settings.get("device", "cuda"),
        tts_model_name    = settings.get("tts_model_name", "Qwen/Qwen3-TTS-12Hz-1.7B-Base"),
        tts_timbre        = settings.get("tts_timbre", ""),
        tts_instruct      = settings.get("tts_instruct", ""),
        export_text       = bool(settings.get("export_text", False)),
        export_lrc        = bool(settings.get("export_lrc", True)),
        export_srt        = bool(settings.get("export_srt", False)),
        export_vtt        = bool(settings.get("export_vtt", False)),
        single_file_mode  = bool(settings.get("single_file_mode", False)),
        force_reprocess   = bool(settings.get("force_reprocess", False)),
        torch_compile     = bool(settings.get("torch_compile", False)),
        regen_missing     = bool(settings.get("regen_missing", True)),
        pronunciation_map = settings.get("pronunciation_map", {}),
        selected_chapters = settings.get("selected_chapters", []),
    )


# ══════════════════════════════════════════════════════════════════════════════
# Chapter loading — cached or freshly extracted
# ══════════════════════════════════════════════════════════════════════════════

def _load_chapters(chapters_raw: list[dict], meta: dict, cfg: "AudiobookConfig") -> list:
    """
    Return a list of ExtractedChapter objects.

    If every chapter in chapters_raw has non-empty 'text', we build the
    chapter list directly from the JSON — no book parsing needed.

    If text is missing, we fall back to extract() on the book file.
    The selected_chapters list from settings is applied as a filter in
    both paths.
    """
    from audiobook_factory.text_extractor import ExtractedChapter

    # Filter to only selected (non-completed) or pending chapters
    selected = cfg.selected_chapters  # raw labels like "1. Chapter 1 (~500 words)"

    # Determine which chapter nums are pending/not completed
    # (force_reprocess means treat all as pending)
    if cfg.force_reprocess:
        pending_nums = {c["num"] for c in chapters_raw}
    else:
        pending_nums = {
            c["num"] for c in chapters_raw
            if c.get("status", "pending") not in ("completed", "complete")
        }

    # Filter by selected_chapters if any are saved
    def _matches_selection(ch_dict):
        """True if this chapter is in the user's saved selection (or no selection)."""
        if not selected:
            return True
        num   = ch_dict["num"]
        title = ch_dict["title"]
        for label in selected:
            # Labels are like "1. Chapter Title  (~500 words)"
            # Match by num prefix or by title substring
            try:
                label_num = int(label.split(".")[0].strip())
                if label_num == num:
                    return True
            except (ValueError, IndexError):
                pass
            if title and title in label:
                return True
        return False

    # ── Try to use cached text ────────────────────────────────────────────────
    has_text = all(c.get("text", "").strip() for c in chapters_raw)

    if has_text:
        print(_info("  📦 Using cached chapter text from JSON (no book re-parsing needed)."))
        chapters = []
        for c in chapters_raw:
            if not _matches_selection(c):
                continue
            chapters.append(ExtractedChapter(
                num       = c["num"],
                title     = c["title"],
                text      = c["text"],
                sentences = c.get("sentences") or [],
            ))
        return chapters

    # ── Fall back to extracting from book file ────────────────────────────────
    book_path = meta.get("book_path", "")
    if not book_path or not os.path.exists(book_path):
        print(_err(
            "❌ No cached chapter text found in the JSON and no valid book file path.\n"
            "   Please provide --book-path or export the JSON from Gradio using the\n"
            "   '📋 Export Config JSON' button (which embeds chapter text)."
        ))
        sys.exit(1)

    print(_info(f"  📖 Extracting chapters from book file: {book_path}"))

    from audiobook_factory.text_extractor import extract
    import queue as _queue

    log_q = _queue.Queue()
    # Build selection titles for the extractor
    selection_titles = None
    if selected:
        import re as _re
        selection_titles = []
        for label in selected:
            after_num = label.split(". ", 1)[-1]
            title = _re.sub(r'\s+\(~[\d,]+\s*words\)\s*$', '', after_num).strip()
            if title:
                selection_titles.append(title)

    extracted, _ = extract(book_path, selections=selection_titles or None, log_fn=log_q.put)

    # Drain log queue to console
    while not log_q.empty():
        print(" ", log_q.get_nowait())

    return extracted


# ══════════════════════════════════════════════════════════════════════════════
# Progress / log consumer (prints to console)
# ══════════════════════════════════════════════════════════════════════════════

def _consume_queues(log_q: queue.Queue, prog_q: queue.Queue, cancel, runner_thread: threading.Thread):
    """Drain log and progress queues while the runner thread is alive."""
    total_chapters = 0
    completed = 0

    while runner_thread.is_alive() or not log_q.empty() or not prog_q.empty():
        # Progress
        try:
            while not prog_q.empty():
                cur, tot = prog_q.get_nowait()
                if tot > 0:
                    pct = cur / tot * 100
                    # Overwrite the progress line in-place
                    print(f"\r  {_info(f'[Progress] {pct:.1f}% ({cur:.0f}/{tot:.0f} chapters)')}", end="", flush=True)
        except queue.Empty:
            pass

        # Log messages
        try:
            msg = log_q.get(timeout=0.15)
            if msg.startswith("__DONE__::"):
                paths = msg.split("::", 1)[1]
                out_files = [p for p in paths.split(",") if p and os.path.exists(p)]
                print()  # newline after progress line
                return out_files
            print(f"\r  {msg}                                    ")
        except queue.Empty:
            pass

    print()
    return []


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    _print_banner()
    parser = _build_parser()
    args = parser.parse_args()

    # ── Load and merge config ──────────────────────────────────────────────────
    meta, settings, chapters_raw, json_path = _load_config(args)

    cfg = _build_audiobook_config(meta, settings)
    os.makedirs(cfg.output_dir, exist_ok=True)

    # ── Print run summary ──────────────────────────────────────────────────────
    total    = len(chapters_raw)
    done     = sum(1 for c in chapters_raw if c.get("status") in ("completed", "complete"))
    pending  = total - done
    has_text = all(c.get("text", "").strip() for c in chapters_raw)

    print(f"  {_h('Book:')}     {meta['book_title']}")
    print(f"  {_h('Chapters:')} {total} total  ({done} completed, {pending} remaining)")
    print(f"  {_h('Voice:')}    {cfg.voice_file or '(built-in timbre)'}")
    print(f"  {_h('Output:')}   {cfg.output_dir}")
    print(f"  {_h('Format:')}   {cfg.output_format}")
    print(f"  {_h('Workers:')}  {cfg.worker_count}")
    print(f"  {_h('TextCache:')} {'✅ yes (fast resume)' if has_text else '⚠️  no  (book will be re-parsed)'}")
    print(_h("━" * 50))
    print()

    # ── Load chapters ─────────────────────────────────────────────────────────
    chapters = _load_chapters(chapters_raw, meta, cfg)

    if not chapters:
        print(_ok("✅ Nothing to generate — all chapters already completed."))
        sys.exit(0)

    # ── Copy progress JSON to output dir if not already there ─────────────────
    import shutil as _shutil
    dest_json = os.path.join(cfg.output_dir, "generation_progress.json")
    if os.path.abspath(json_path) != os.path.abspath(dest_json):
        try:
            _shutil.copy2(json_path, dest_json)
        except Exception as e:
            print(_warn(f"  ⚠️  Could not copy progress JSON to output dir: {e}"))

    # ── Set up cancel token ───────────────────────────────────────────────────
    from audiobook_factory.pipeline import CancelToken, run_pipeline
    cancel = CancelToken()

    def _handle_sigint(sig, frame):
        print(_warn("\n\n  ⛔ Ctrl+C received — cancelling after current chunk…"))
        cancel.cancel()

    signal.signal(signal.SIGINT, _handle_sigint)

    # ── Check if FastAPI orchestrator is running ───────────────────────────────
    log_q  = queue.Queue()
    prog_q = queue.Queue()
    out_files: list[str] = []

    use_api = _is_api_healthy()
    if use_api:
        print(_info("  📡 FastAPI orchestrator is running — dispatching task to it."))
    else:
        print(_info("  🖥️  FastAPI orchestrator not detected — running locally in-process."))
    print()

    # ── Runner thread ─────────────────────────────────────────────────────────
    def _runner():
        nonlocal out_files
        try:
            if use_api:
                import requests
                import dataclasses
                import asyncio

                config_dict   = dataclasses.asdict(cfg)
                chapters_list = [
                    {"num": ch.num, "title": ch.title, "text": ch.text, "sentences": ch.sentences}
                    for ch in chapters
                ]
                payload = {"config": config_dict, "chapters": chapters_list}

                r = requests.post("http://127.0.0.1:8000/api/v1/generate", json=payload)
                r.raise_for_status()
                task_id = r.json()["task_id"]
                cancel.task_id = task_id
                log_q.put(f"✅ Enqueued. Task ID: {task_id}")

                # Poll for completion
                import time as _time
                while True:
                    if cancel.is_cancelled:
                        try:
                            requests.post(f"http://127.0.0.1:8000/api/v1/tasks/{task_id}/cancel", timeout=3)
                        except Exception:
                            pass
                        log_q.put("__DONE__::")
                        return
                    try:
                        poll = requests.get(f"http://127.0.0.1:8000/api/v1/tasks/{task_id}", timeout=5)
                        if poll.ok:
                            st = poll.json()
                            if st.get("progress"):
                                prog_q.put((st["progress"] * 100, 100))
                            if st["status"] == "completed":
                                paths = ",".join(st.get("output_files", []))
                                log_q.put(f"__DONE__::{paths}")
                                return
                            elif st["status"] in ("failed", "cancelled"):
                                log_q.put(f"❌ Task {st['status']}.")
                                log_q.put("__DONE__::")
                                return
                    except Exception:
                        pass
                    _time.sleep(3)
            else:
                # Local in-process run
                files = run_pipeline(cfg, chapters, log_q, prog_q, cancel)
                log_q.put(f"__DONE__::{','.join(files)}")
        except Exception as e:
            import traceback
            log_q.put(f"❌ Fatal error: {e}")
            traceback.print_exc()
            log_q.put("__DONE__::")

    t = threading.Thread(target=_runner, daemon=True)
    t.start()

    # ── Consume output while generation runs ──────────────────────────────────
    out_files = _consume_queues(log_q, prog_q, cancel, t)
    t.join(timeout=10)

    # ── Final summary ─────────────────────────────────────────────────────────
    print()
    print(_h("━" * 50))
    if cancel.is_cancelled:
        print(_warn("  ⛔ Generation cancelled."))
    elif out_files:
        print(_ok(f"  ✅ Complete — {len(out_files)} file(s) generated:"))
        for f in sorted(out_files):
            print(f"     {f}")
    else:
        print(_warn("  ⚠️  No output files were generated. Check the log above for errors."))
    print(_h("━" * 50))
    print()


if __name__ == "__main__":
    main()
