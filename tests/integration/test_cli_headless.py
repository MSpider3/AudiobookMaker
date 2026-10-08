"""
test_cli_headless.py
====================
Integration tests for CLI headless execution (cli.py).
"""

from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from cli import _build_parser, _load_config, _display_progress


class TestCliHeadless:

    def test_arg_parser_defaults_and_flags(self):
        parser = _build_parser()
        args = parser.parse_args(["/tmp/progress.json", "--output-format", "wav", "--worker-count", "4"])
        assert args.config_json == "/tmp/progress.json"
        assert args.output_format == "wav"
        assert args.worker_count == 4

    def test_load_config_merges_settings(self):
        with tempfile.TemporaryDirectory() as td:
            json_path = os.path.join(td, "generation_progress.json")
            data = {
                "book_title": "CLI Test Novel",
                "settings": {"config_version": 6, "output_format": "mp3", "worker_count": 2},
                "chapters": [{"num": 1, "title": "Chapter 1", "status": "pending"}]
            }
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(data, f)

            parser = _build_parser()
            args = parser.parse_args([json_path, "--worker-count", "8"])
            meta, settings, chapters, path = _load_config(args)

            assert meta["book_title"] == "CLI Test Novel"
            assert settings["output_format"] == "mp3"
            assert len(chapters) == 1

    def test_display_progress_truncates_long_titles(self, monkeypatch):
        # Emulate TTY
        monkeypatch.setattr("cli._IS_TTY", True)
        buf = io.StringIO()
        monkeypatch.setattr("sys.stdout", buf)

        very_long_title = "The Incredibly Long Title of a Forgotten Tome in the Deep Caves of Mount Terror " * 2
        _display_progress(
            chapter_num=1,
            total_chapters=5,
            chunk_num=2,
            total_chunks=10,
            chapter_title=very_long_title
        )
        output = buf.getvalue()
        assert "\r" in output
        assert "[1/5]" in output
        assert "chunk 2/10" in output


class TestCliProcess:
    """Runs cli.py as a real process: the exit status is what scripts and
    notebooks act on."""

    @staticmethod
    def _run(*argv, cwd=None):
        import subprocess

        env = dict(os.environ, ABM_SKIP_GPU_WARMUP="1", ABM_API_URL="http://127.0.0.1:9")
        return subprocess.run(
            [sys.executable, os.path.join(_ROOT, "cli.py"), *argv],
            cwd=cwd or _ROOT, env=env, capture_output=True, text=True, timeout=300,
        )

    def test_missing_config_exits_1(self, tmp_path):
        result = self._run(str(tmp_path / "missing.json"))
        assert result.returncode == 1
        assert "Config JSON not found" in result.stdout
        assert "Traceback" not in result.stdout + result.stderr

    def test_no_arguments_is_a_usage_error(self):
        result = self._run()
        assert result.returncode == 2
        assert "--book" in result.stderr

    def test_list_providers_exits_0(self):
        from audiobook_factory.tts_providers.registry import provider_names

        result = self._run("--list-providers")
        assert result.returncode == 0, result.stderr[-500:]
        for name in provider_names():
            assert name in result.stdout

    def test_book_run_with_mock_provider_exits_0(self, tmp_path):
        book = tmp_path / "tiny.txt"
        book.write_text(
            "The ferry left at dawn and nobody waved from the shore. "
            "By noon the island was a line on the water.",
            encoding="utf-8",
        )
        voice = os.path.join(_ROOT, "tests", "fixtures", "audio", "synthetic_voice_reference.wav")
        out = tmp_path / "out"
        # Run from another directory: nothing may depend on the working directory.
        result = self._run(
            "--book", str(book), "--provider", "mock", "--voice-file", voice,
            "--output-dir", str(out), "--local", cwd=str(tmp_path),
        )
        assert result.returncode == 0, result.stdout[-1500:] + result.stderr[-1500:]
        assert [f for f in os.listdir(out) if f.endswith(".mp3")]
        with open(out / "generation_progress.json", encoding="utf-8") as fh:
            saved = json.load(fh)
        assert [c["status"] for c in saved["chapters"]] == ["completed"]
        assert saved["book_title"] == "tiny"
