"""
test_cli_run.py
===============
End-to-end tests of ``cli.main()`` with the mock TTS provider: exit codes,
resume behaviour, --dry-run, --book, --embed-cover-only, Ctrl+C handling and
dispatch through a real API server.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from collections import defaultdict

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import cli
from audiobook_factory.pipeline import _CONFIG_SCHEMA_VERSION
from audiobook_factory.tts_providers.base_tts_provider import ProviderInfo

_VOICE = os.path.join(_ROOT, "tests", "fixtures", "audio", "synthetic_voice_reference.wav")
_EPUB = os.path.join(_ROOT, "tests", "fixtures", "source_documents", "dummy_book.epub")
_SENTENCES = [
    "The harbour was quiet when the first boat came in that morning.",
    "Nobody on the quay had expected a letter from the old lighthouse.",
    "She read it twice before she folded it into her coat pocket.",
]
_FAIL_MARKER = "FAILMARKER"


def _chapter(num: int, text: str | None = None) -> dict:
    text = text or _SENTENCES[(num - 1) % len(_SENTENCES)]
    return {"num": num, "title": f"Chapter {num}", "status": "pending", "text": text, "sentences": [text]}


def _job(tmp_path, chapters=None, output_dir=None, **settings):
    """Writes a progress JSON whose output directory is separate from it.

    Returns ``(json_path, output_dir)`` as strings.
    """
    job = tmp_path / "job"
    job.mkdir(exist_ok=True)
    out = output_dir or str(job / "out")
    merged = {
        "config_version": _CONFIG_SCHEMA_VERSION,
        "tts_provider_name": "mock",
        "output_dir": out,
        "output_format": "mp3",
        "export_lrc": False,
        "max_chapter_retries": 0,
        "retry_failed_at_end": False,
    }
    merged.update(settings)
    data = {
        "book_title": "Cli Run Book",
        "voice_file": _VOICE,
        "settings": merged,
        "chapters": [_chapter(1), _chapter(2)] if chapters is None else chapters,
    }
    path = job / "generation_progress.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return str(path), out


def _statuses(out_dir: str) -> dict:
    with open(os.path.join(out_dir, "generation_progress.json"), encoding="utf-8") as fh:
        return {c["num"]: c.get("status") for c in json.load(fh)["chapters"]}


def _mp3s(out_dir: str) -> list[str]:
    return sorted(f for f in os.listdir(out_dir) if f.endswith(".mp3"))


@pytest.fixture(autouse=True)
def _isolated(monkeypatch):
    monkeypatch.setenv("ABM_SKIP_GPU_WARMUP", "1")
    monkeypatch.delenv("ABM_API_SECRET", raising=False)
    monkeypatch.delenv("ABM_SKIP_INSTALL_CHECK", raising=False)
    # Never talk to an API server that happens to run on this machine.
    monkeypatch.setenv("ABM_API_URL", "http://127.0.0.1:9")


@pytest.fixture
def no_pipeline(monkeypatch):
    """Fails the test if the run reaches the pipeline (and so a model)."""
    def _forbidden(*args, **kwargs):
        raise AssertionError("the pipeline must not run in this test")
    monkeypatch.setattr(cli, "_run_local", _forbidden)
    monkeypatch.setattr(cli, "_run_via_api", _forbidden)
    from audiobook_factory.gpu_pool import GPUPoolManager
    monkeypatch.setattr(GPUPoolManager, "get_pool", _forbidden)


# ── Exit codes ────────────────────────────────────────────────────────────────

class TestExitCodes:

    def test_success_is_0_and_every_chapter_is_recorded(self, tmp_path, capsys):
        path, out = _job(tmp_path)
        assert cli.main([path, "--local"]) == 0
        assert len(_mp3s(out)) == 2
        assert _statuses(out) == {1: "completed", 2: "completed"}
        text = capsys.readouterr().out
        assert "2 of 2 completed" in text

    def test_second_run_has_nothing_to_generate(self, tmp_path, capsys, monkeypatch):
        path, out = _job(tmp_path)
        assert cli.main([path, "--local"]) == 0
        capsys.readouterr()

        def _forbidden(*args, **kwargs):
            raise AssertionError("nothing should be generated")
        monkeypatch.setattr(cli, "_run_local", _forbidden)
        assert cli.main([path, "--local"]) == 0
        assert "Nothing to generate" in capsys.readouterr().out

    def test_failed_chapter_is_1_with_its_error(self, tmp_path, capsys, monkeypatch):
        from tests.fixtures import mock_provider

        real_batch = mock_provider.MockTTSProvider.synthesize_batch
        real_single = mock_provider.MockTTSProvider.synthesize

        def _batch(self, texts, voice_ref, **kwargs):
            if any(_FAIL_MARKER in t for t in texts):
                raise RuntimeError("injected synthesis failure")
            return real_batch(self, texts, voice_ref, **kwargs)

        def _single(self, text, voice_ref, out_path=None, **kwargs):
            if _FAIL_MARKER in text:
                raise RuntimeError("injected synthesis failure")
            return real_single(self, text, voice_ref, out_path, **kwargs)

        monkeypatch.setattr(mock_provider.MockTTSProvider, "synthesize_batch", _batch)
        monkeypatch.setattr(mock_provider.MockTTSProvider, "synthesize", _single)

        bad = f"This chapter contains the {_FAIL_MARKER} word and can never be spoken aloud."
        path, out = _job(tmp_path, chapters=[_chapter(1), _chapter(2, bad), _chapter(3)])
        assert cli.main([path, "--local"]) == 1

        statuses = _statuses(out)
        assert statuses[1] == "completed" and statuses[3] == "completed"
        assert statuses[2] == "failed"
        text = capsys.readouterr().out
        assert "2 of 3 completed" in text
        assert "1 chapter(s) did not complete" in text
        assert "Chapter 2 'Chapter 2'" in text
        assert "injected synthesis failure" in text

    def test_fatal_error_is_1(self, tmp_path, capsys, monkeypatch):
        def _crash(*args, **kwargs):
            raise RuntimeError("the pipeline exploded")
        monkeypatch.setattr(cli, "_run_local", _crash)
        path, _out = _job(tmp_path)
        assert cli.main([path, "--local"]) == 1
        assert "Fatal error: the pipeline exploded" in capsys.readouterr().out

    def test_missing_json_is_1(self, tmp_path, capsys):
        assert cli.main([str(tmp_path / "nope.json")]) == 1
        assert "Config JSON not found" in capsys.readouterr().out

    def test_no_input_is_a_usage_error(self):
        with pytest.raises(SystemExit) as excinfo:
            cli.main([])
        assert excinfo.value.code == 2

    def test_invalid_settings_are_1_without_traceback(self, tmp_path, capsys, no_pipeline):
        path, _out = _job(tmp_path, tts_provider_name="no-such-engine")
        assert cli.main([path, "--local"]) == 1
        captured = capsys.readouterr()
        assert "invalid tts_provider_name" in captured.out
        assert "Traceback" not in captured.out + captured.err

    def test_invalid_output_format_is_1_without_traceback(self, tmp_path, capsys, no_pipeline):
        path, _out = _job(tmp_path, output_format="xyz")
        assert cli.main([path, "--local"]) == 1
        captured = capsys.readouterr()
        assert "Invalid output_format 'xyz'" in captured.out
        assert "Traceback" not in captured.out + captured.err

    def test_voice_file_from_another_machine_is_1(self, tmp_path, capsys, no_pipeline):
        path, _out = _job(tmp_path)
        data = json.load(open(path, encoding="utf-8"))
        data["voice_file"] = "/tmp/gradio/abc123/narrator.wav"
        json.dump(data, open(path, "w", encoding="utf-8"))
        assert cli.main([path, "--local"]) == 1
        text = capsys.readouterr().out
        assert "Voice file not found" in text and "--voice-file" in text

    def test_engine_not_installed_is_1_with_install_command(self, tmp_path, capsys, monkeypatch, no_pipeline):
        fake = ProviderInfo(
            name="mock", display_name="Fake Engine", license="Test-1.0",
            pip_requirements=("abm-engine-that-does-not-exist>=1.0",),
            install_notes="Install torch first.",
        )
        monkeypatch.setattr(cli, "_provider_info", lambda name: fake)
        path, _out = _job(tmp_path)
        assert cli.main([path, "--local"]) == 1
        captured = capsys.readouterr()
        assert "Fake Engine" in captured.out and "not installed" in captured.out
        assert "pip install 'abm-engine-that-does-not-exist>=1.0'" in captured.out
        assert "Install torch first." in captured.out
        assert "Traceback" not in captured.out + captured.err

    def test_engine_import_failure_at_load_time_prints_install_command(self, tmp_path, capsys, monkeypatch):
        """The engine's package turns out to be missing only when the model loads."""
        fake = ProviderInfo(
            name="mock", display_name="Fake Engine", pip_requirements=("pytest",),
            install_notes="Needs transformers 5.",
        )
        monkeypatch.setattr(cli, "_provider_info", lambda name: fake)

        def _load_fails(*args, **kwargs):
            try:
                raise ModuleNotFoundError("No module named 'fake_engine'")
            except ImportError as inner:
                raise RuntimeError("Fake Engine is not installed") from inner
        monkeypatch.setattr(cli, "_run_local", _load_fails)

        path, _out = _job(tmp_path)
        assert cli.main([path, "--local"]) == 1
        captured = capsys.readouterr()
        assert "could not be loaded" in captured.out
        assert "pip install pytest" in captured.out and "Needs transformers 5." in captured.out
        assert "Traceback" not in captured.out + captured.err


# ── Ctrl+C ────────────────────────────────────────────────────────────────────

class TestCancel:

    def test_ctrl_c_cancels_cooperatively_and_exits_130(self, tmp_path, capsys, monkeypatch):
        seen = {}

        def _interrupted_run(cfg, chapters, log_q, prog_q, cancel):
            seen["handler_during_run"] = signal.getsignal(signal.SIGINT)
            signal.raise_signal(signal.SIGINT)  # the user presses Ctrl+C
            deadline = time.monotonic() + 10
            while not cancel.is_cancelled and time.monotonic() < deadline:
                time.sleep(0.02)
            seen["cancelled"] = cancel.is_cancelled
            return []

        monkeypatch.setattr(cli, "_run_local", _interrupted_run)
        before = signal.getsignal(signal.SIGINT)
        path, _out = _job(tmp_path)

        assert cli.main([path, "--local"]) == 130

        assert seen["cancelled"] is True
        assert seen["handler_during_run"] is not before
        assert signal.getsignal(signal.SIGINT) is before
        text = capsys.readouterr().out
        assert "Press Ctrl+C again to exit immediately" in text
        assert "Generation cancelled" in text

    def test_second_ctrl_c_exits_immediately(self, tmp_path, monkeypatch):
        exits = []
        monkeypatch.setattr(cli, "_hard_exit", exits.append)
        # _make_sigint_handler binds its default at definition time.
        real_make = cli._make_sigint_handler
        monkeypatch.setattr(cli, "_make_sigint_handler", lambda cancel: real_make(cancel, hard_exit=exits.append))

        def _stuck_run(cfg, chapters, log_q, prog_q, cancel):
            signal.raise_signal(signal.SIGINT)
            deadline = time.monotonic() + 10
            while not cancel.is_cancelled and time.monotonic() < deadline:
                time.sleep(0.02)
            signal.raise_signal(signal.SIGINT)  # the cooperative stop did not help
            deadline = time.monotonic() + 10
            while not exits and time.monotonic() < deadline:
                time.sleep(0.02)
            return []

        monkeypatch.setattr(cli, "_run_local", _stuck_run)
        path, _out = _job(tmp_path)
        cli.main([path, "--local"])
        assert exits == [130]


# ── Resume ────────────────────────────────────────────────────────────────────

class TestResume:

    def test_rerun_with_the_exported_json_keeps_progress(self, tmp_path, capsys):
        """Re-running the same command (a Kaggle cell after an interruption)
        must not reset chapters finished since the JSON was exported."""
        path, out = _job(tmp_path, chapters=[_chapter(1), _chapter(2), _chapter(3)])
        exported = open(path, encoding="utf-8").read()

        assert cli.main([path, "--local", "--chapters", "1"]) == 0
        first = os.path.join(out, _mp3s(out)[0])
        stamp = os.stat(first).st_mtime_ns
        assert _statuses(out)[1] == "completed"
        capsys.readouterr()
        time.sleep(0.05)

        assert cli.main([path, "--local"]) == 0

        text = capsys.readouterr().out
        assert "Resuming" in text
        assert "[Chapter 1/3] ⏩ Already completed. Skipping." in text
        assert os.stat(first).st_mtime_ns == stamp, "chapter 1 was generated again"
        assert _statuses(out) == {1: "completed", 2: "completed", 3: "completed"}
        assert len(_mp3s(out)) == 3
        # The file given on the command line is input only.
        assert open(path, encoding="utf-8").read() == exported

    def test_settings_still_come_from_the_given_json(self, tmp_path):
        path, out = _job(tmp_path, chapters=[_chapter(1), _chapter(2)])
        assert cli.main([path, "--local", "--chapters", "1"]) == 0

        data = json.load(open(path, encoding="utf-8"))
        data["settings"]["bitrate_kbps"] = 96
        json.dump(data, open(path, "w", encoding="utf-8"))
        assert cli.main([path, "--local", "--pause", "0.25"]) == 0

        saved = json.load(open(os.path.join(out, "generation_progress.json"), encoding="utf-8"))["settings"]
        assert saved["bitrate_kbps"] == 96 and saved["pause"] == 0.25

    def test_redo_regenerates_a_finished_chapter(self, tmp_path, capsys):
        path, out = _job(tmp_path)
        assert cli.main([path, "--local"]) == 0
        files = [os.path.join(out, f) for f in _mp3s(out)]
        stamps = [os.stat(f).st_mtime_ns for f in files]
        capsys.readouterr()
        time.sleep(0.05)

        assert cli.main([path, "--local", "--redo", "2"]) == 0

        text = capsys.readouterr().out
        assert "redo: 2" in text
        assert os.stat(files[0]).st_mtime_ns == stamps[0]
        assert os.stat(files[1]).st_mtime_ns != stamps[1]
        # A later plain run must not redo it again.
        saved = json.load(open(os.path.join(out, "generation_progress.json"), encoding="utf-8"))
        again = cli.main([os.path.join(out, "generation_progress.json"), "--local", "--dry-run"])
        assert again == 0 and saved["settings"]["redo_chapters"] == [2]
        assert "redo" not in capsys.readouterr().out.split("Chapters (")[1]


# ── Empty chapter list ────────────────────────────────────────────────────────

class TestEmptyChapterList:

    def test_without_a_book_it_is_an_error_not_success(self, tmp_path, capsys, no_pipeline):
        path, _out = _job(tmp_path, chapters=[])
        assert cli.main([path, "--local"]) == 1
        text = capsys.readouterr().out
        assert "Nothing to generate" not in text
        assert "--book-path" in text

    def test_book_path_is_used_to_extract_chapters(self, tmp_path, capsys, no_pipeline):
        book = tmp_path / "story.txt"
        book.write_text("The tide went out at noon. The boats lay on the sand.", encoding="utf-8")
        path, _out = _job(tmp_path, chapters=[])
        assert cli.main([path, "--book-path", str(book), "--dry-run"]) == 0
        text = capsys.readouterr().out
        assert "Extracting chapters from book file" in text
        assert "Full Book" in text


# ── --dry-run ─────────────────────────────────────────────────────────────────

class TestDryRun:

    def test_prints_config_engine_chapters_and_estimate(self, tmp_path, capsys, no_pipeline):
        path, out = _job(tmp_path, chapters=[_chapter(1), _chapter(2, "x" * 1500)], speed=1.0)
        assert cli.main([path, "--dry-run", "--bitrate", "96"]) == 0
        text = capsys.readouterr().out

        assert "DRY RUN" in text
        assert "MockTTSProvider (mock)" in text and "Licence:" in text
        assert "Resolved configuration" in text
        assert "tts_provider_name: 'mock'" in text and "bitrate_kbps: 96" in text
        assert "verify_chunks: 'duration'" in text
        assert "Chapter 1" in text and "Chapter 2" in text
        assert "1,500" in text  # character count of chapter 2
        assert "0:01:40" in text  # 1500 characters at 15 per second
        assert "Estimated audio:" in text
        assert not os.path.exists(out), "a dry run must not create the output directory"

    def test_foreign_output_dir_falls_back_instead_of_aborting(self, tmp_path, capsys, no_pipeline):
        path, _out = _job(tmp_path, output_dir="/home/another-user/AudiobookMaker/audiobook_output/Cli Run Book")
        assert cli.main([path, "--dry-run"]) == 0
        text = capsys.readouterr().out
        assert "Untrusted output_dir" in text
        assert os.path.join(_ROOT, "audiobook_output", "Cli Run Book") in text


# ── --book ────────────────────────────────────────────────────────────────────

class TestBookMode:

    def test_runs_straight_from_a_book_file(self, tmp_path, capsys):
        out = tmp_path / "from_book"
        code = cli.main([
            "--book", _EPUB, "--provider", "mock", "--voice-file", _VOICE,
            "--output-dir", str(out), "--chapters", "1", "--local",
        ])
        assert code == 0
        assert len(_mp3s(str(out))) == 1
        saved = json.load(open(out / "generation_progress.json", encoding="utf-8"))
        assert saved["settings"]["tts_provider_name"] == "mock"
        assert saved["settings"]["book_path"] == _EPUB
        assert saved["book_title"] and saved["book_title"] != "Audiobook"
        assert "1 of 1 completed" in capsys.readouterr().out

    def test_default_output_dir_is_named_after_the_title(self, tmp_path, capsys, no_pipeline):
        book = tmp_path / "My Little Story.txt"
        book.write_text("Once there was a story. It was short.", encoding="utf-8")
        assert cli.main(["--book", str(book), "--provider", "mock", "--dry-run"]) == 0
        assert os.path.join(_ROOT, "audiobook_output", "My Little Story") in capsys.readouterr().out

    def test_missing_book_is_1(self, tmp_path, capsys):
        assert cli.main(["--book", str(tmp_path / "nope.epub")]) == 1
        assert "file not found" in capsys.readouterr().out


# ── --embed-cover-only ────────────────────────────────────────────────────────

@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
class TestEmbedCoverOnly:

    def _make_audio(self, path: str) -> None:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "sine=frequency=440:duration=1",
             "-metadata", "title=Chapter One", "-b:a", "64k", path],
            check=True,
        )

    def _make_cover(self, path: str) -> None:
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi", "-i", "color=c=red:s=64x64",
             "-frames:v", "1", path],
            check=True,
        )

    def test_embeds_the_cover_into_existing_mp3_files(self, tmp_path, capsys, no_pipeline):
        from mutagen.id3 import ID3

        path, out = _job(tmp_path)
        os.makedirs(out)
        audio = [os.path.join(out, "Chapter 1 - One.mp3"), os.path.join(out, "Chapter 2 - Two.mp3")]
        for target in audio:
            self._make_audio(target)
        cover = str(tmp_path / "art.png")
        self._make_cover(cover)
        assert not ID3(audio[0]).getall("APIC")

        assert cli.main([path, "--embed-cover-only", "--cover-image", cover]) == 0

        for target in audio:
            tags = ID3(target)
            pictures = tags.getall("APIC")
            assert len(pictures) == 1 and pictures[0].mime == "image/png"
            assert str(tags["TIT2"]) == "Chapter One", "existing tags must be kept"
        assert sorted(os.listdir(out)) == ["Chapter 1 - One.mp3", "Chapter 2 - Two.mp3"], "no temp files left"
        assert "Embedded the cover image into 2 audio file(s)" in capsys.readouterr().out

    def test_no_cover_is_1(self, tmp_path, capsys, no_pipeline):
        path, out = _job(tmp_path)
        os.makedirs(out)
        self._make_audio(os.path.join(out, "Chapter 1 - One.mp3"))
        assert cli.main([path, "--embed-cover-only"]) == 1
        assert "No cover image found" in capsys.readouterr().out

    def test_no_audio_files_is_1(self, tmp_path, capsys, no_pipeline):
        path, out = _job(tmp_path)
        os.makedirs(out)
        cover = str(tmp_path / "art.png")
        self._make_cover(cover)
        assert cli.main([path, "--embed-cover-only", "--cover-image", cover]) == 1
        assert "No audio files" in capsys.readouterr().out


# ── Dispatch through the API server ───────────────────────────────────────────

@pytest.fixture
def api_server(tmp_path, monkeypatch):
    """A real uvicorn server for api.server:app on a free port.

    Yields the server's output base directory. ``ABM_API_URL`` points the CLI
    at it. A watchdog stops the server after two minutes so a stalled task
    makes the CLI give up (lost contact) instead of hanging the suite.
    """
    uvicorn = pytest.importorskip("uvicorn")
    pytest.importorskip("requests")
    import api.server as server_mod
    import api.worker as worker_mod

    base = tmp_path / "server_base"
    base.mkdir()
    monkeypatch.setenv("ABM_OUTPUT_BASE", str(base))
    # The module-level queue stays bound to whichever event loop used it
    # first, so give this server's loop a queue of its own.
    fresh_queue: asyncio.Queue = asyncio.Queue()
    monkeypatch.setattr(worker_mod, "task_queue", fresh_queue)
    monkeypatch.setattr(server_mod, "task_queue", fresh_queue)
    monkeypatch.setattr(server_mod._generate_limiter, "_requests", defaultdict(list))

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(server_mod.app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True, name="test-api-server")
    thread.start()
    deadline = time.monotonic() + 30
    while not server.started and thread.is_alive() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert server.started, "API server did not start"

    monkeypatch.setenv("ABM_API_URL", f"http://127.0.0.1:{port}")
    monkeypatch.setattr(cli, "_API_POLL_SEC", 0.2)
    monkeypatch.setattr(cli, "_API_MAX_POLL_FAILURES", 10)
    watchdog = threading.Timer(120.0, lambda: setattr(server, "should_exit", True))
    watchdog.daemon = True
    watchdog.start()
    try:
        yield base
    finally:
        watchdog.cancel()
        server.should_exit = True
        thread.join(timeout=30)


class TestApiMode:

    def test_task_runs_on_the_server_and_its_log_is_streamed(self, tmp_path, capsys, api_server, monkeypatch):
        def _no_local(*args, **kwargs):
            raise AssertionError("the job must run on the API server")
        monkeypatch.setattr(cli, "_run_local", _no_local)

        out = str(api_server / "Api Book")
        path, _ = _job(tmp_path, output_dir=out)
        assert cli.main([path, "--output-dir", out]) == 0

        text = capsys.readouterr().out
        assert "dispatching task to it" in text
        assert "Enqueued. Task ID:" in text
        # Lines only the server produces, relayed by polling ?since=<n>:
        assert text.count("Starting generation task") == 1, "log lines must arrive exactly once"
        assert text.count("[Pipeline] Starting — 2 chapter(s)") == 1
        assert "Generation complete. Processed 2 files." in text
        assert len(_mp3s(out)) == 2
        assert "2 of 2 completed" in text

    def test_api_key_is_sent_when_the_server_requires_one(self, tmp_path, capsys, api_server, monkeypatch):
        monkeypatch.setenv("ABM_API_SECRET", "s3cret-for-tests")
        monkeypatch.setattr(cli, "_run_local", lambda *a, **k: (_ for _ in ()).throw(AssertionError("ran locally")))
        assert cli._api_headers() == {"x-api-key": "s3cret-for-tests"}

        out = str(api_server / "Keyed Book")
        path, _ = _job(tmp_path, output_dir=out)
        assert cli.main([path, "--output-dir", out]) == 0
        assert "Starting generation task" in capsys.readouterr().out

    def test_without_the_key_the_job_runs_locally(self, tmp_path, capsys, api_server, monkeypatch):
        monkeypatch.setenv("ABM_API_SECRET", "s3cret-for-tests")
        monkeypatch.setattr(cli, "_api_headers", lambda: {})

        path, out = _job(tmp_path)
        assert cli.main([path]) == 0
        text = capsys.readouterr().out
        assert "requires an API key" in text and "Running locally" in text
        assert len(_mp3s(out)) == 2

    def test_failed_task_prints_the_servers_error_message(self, tmp_path, capsys, api_server, monkeypatch):
        import api.worker as worker_mod

        def _server_side_crash(*args, **kwargs):
            raise RuntimeError("CUDA out of memory on the server")
        monkeypatch.setattr(worker_mod, "run_pipeline", _server_side_crash)
        monkeypatch.setattr(cli, "_run_local", lambda *a, **k: (_ for _ in ()).throw(AssertionError("ran locally")))

        out = str(api_server / "Broken Book")
        path, _ = _job(tmp_path, output_dir=out)
        assert cli.main([path, "--output-dir", out]) == 1
        text = capsys.readouterr().out
        assert "Task failed on the API server: CUDA out of memory on the server" in text
        # The server-side traceback arrives through the streamed task log.
        assert "Task crashed: CUDA out of memory on the server" in text

    def test_output_dir_outside_the_servers_base_runs_locally(self, tmp_path, capsys, api_server):
        import api.worker as worker_mod

        known = set(worker_mod.tasks)
        path, out = _job(tmp_path)  # tmp_path/job/out is not under the server's base
        assert os.path.isabs(out)
        assert cli.main([path]) == 0

        text = capsys.readouterr().out
        assert "outside the API server's output base" in text
        assert str(api_server) in text
        assert "Running locally in-process instead" in text
        assert set(worker_mod.tasks) == known, "nothing may be enqueued on the server"
        assert len(_mp3s(out)) == 2
        assert _statuses(out) == {1: "completed", 2: "completed"}

    def test_invalid_request_is_reported_not_retried_locally(self, tmp_path, capsys, api_server, monkeypatch):
        monkeypatch.setattr(cli, "_run_local", lambda *a, **k: (_ for _ in ()).throw(AssertionError("ran locally")))
        out = str(api_server / "Bad Options")
        path, _ = _job(tmp_path, output_dir=out)
        # Reaches the server only: the CLI's own check is bypassed here.
        monkeypatch.setattr(cli, "_check_provider_installed", lambda name: None)
        real_asdict = cli.dataclasses.asdict

        def _poisoned(cfg):
            data = real_asdict(cfg)
            data["voice_file"] = os.path.dirname(_VOICE)  # a directory
            return data
        monkeypatch.setattr(cli.dataclasses, "asdict", _poisoned)

        assert cli.main([path, "--output-dir", out]) == 1
        text = capsys.readouterr().out
        assert "The API server rejected the request (HTTP 400)" in text
        assert "voice_file is not an existing regular file" in text
