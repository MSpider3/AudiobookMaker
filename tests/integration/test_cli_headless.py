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
