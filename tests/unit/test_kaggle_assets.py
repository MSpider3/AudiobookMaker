"""
test_kaggle_assets.py
=====================
The Kaggle test notebook reads its narrator voice and books from
``tests/kaggle/assets``. These tests keep that folder complete and in step
with the fixtures it is copied from.
"""

from __future__ import annotations

import filecmp
import json
import os
import sys
import zipfile

import soundfile as sf

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from tests.kaggle import abm_gpu_suite, generate_test_notebook, sync_assets


class TestKaggleAssets:

    def test_books_match_the_fixtures(self):
        wanted = sync_assets.book_files()
        assert wanted, "no fixture books found"
        assert sorted(os.listdir(sync_assets.BOOKS_DIR)) == wanted, (
            "tests/kaggle/assets/books is out of date — run: python tests/kaggle/sync_assets.py"
        )
        for name in wanted:
            assert filecmp.cmp(
                os.path.join(sync_assets.FIXTURES_DIR, name),
                os.path.join(sync_assets.BOOKS_DIR, name),
                shallow=False,
            ), f"{name} differs from its fixture — run: python tests/kaggle/sync_assets.py"

    def test_every_book_format_is_present(self):
        suffixes = {os.path.splitext(name)[1] for name in os.listdir(sync_assets.BOOKS_DIR)}
        assert {".epub", ".pdf", ".docx", ".odt", ".txt", ".mobi"} <= suffixes
        assert os.path.exists(os.path.join(sync_assets.BOOKS_DIR, "expected_chapters.json"))

    def test_narrator_voice_is_a_usable_reference_with_a_transcript(self):
        voice_dir = os.path.join(sync_assets.ASSETS_DIR, "voice")
        clips = [name for name in os.listdir(voice_dir) if name.endswith(".wav")]
        assert len(clips) == 1, "exactly one narrator clip is expected"
        clip = os.path.join(voice_dir, clips[0])
        info = sf.info(clip)
        assert info.channels == 1 and info.samplerate == 24000
        assert 5.0 <= info.duration <= 30.0
        with open(os.path.splitext(clip)[0] + ".txt", encoding="utf-8") as fh:
            transcript = fh.read().strip()
        # Roughly two to four words a second for narration.
        assert info.duration * 1.5 <= len(transcript.split()) <= info.duration * 5


class TestHarness:
    """Faults of the test harness itself that made a Kaggle run unreadable."""

    def test_results_archive_is_not_empty_under_kaggle_working(self, tmp_path, monkeypatch, capsys):
        # The archive used to skip every folder whose path contained "/work",
        # and on Kaggle the results live under /kaggle/working.
        results = tmp_path / "kaggle" / "working" / "abm_results"
        (results / "logs").mkdir(parents=True)
        (results / "samples").mkdir()
        (results / "work" / "qwen_clone" / ".temp_chunks").mkdir(parents=True)
        (results / "env.json").write_text(json.dumps({"test": "env", "status": "pass", "metrics": {}}))
        (results / "logs" / "qwen_clone.log").write_text("log line\n")
        (results / "samples" / "qwen_clone.mp3").write_bytes(b"audio")
        (results / "work" / "qwen_clone" / ".temp_chunks" / "chunk_0.wav").write_bytes(b"x" * 100)
        monkeypatch.setattr(abm_gpu_suite, "RESULTS_DIR", str(results))

        abm_gpu_suite.cmd_report(None)

        with zipfile.ZipFile(results.parent / "abm_test_results.zip") as archive:
            names = set(archive.namelist())
        assert {"abm_results/REPORT.md", "abm_results/summary.json", "abm_results/env.json",
                "abm_results/logs/qwen_clone.log", "abm_results/samples/qwen_clone.mp3"} <= names
        assert not any("/work/" in name for name in names)  # intermediate chunks stay out
        assert "5 files" in capsys.readouterr().out

    def test_transcript_setting_accepts_a_path(self):
        clip = os.path.join(sync_assets.ASSETS_DIR, "voice", "LOTM_narrator_voice.wav")
        sidecar = os.path.relpath(os.path.splitext(clip)[0] + ".txt", _ROOT)
        text = abm_gpu_suite._transcript_text(sidecar)
        assert len(text.split()) > 30 and not text.endswith(".txt")
        assert abm_gpu_suite._transcript_text("These are the words.") == "These are the words."
        assert abm_gpu_suite._transcript_problem(clip, text) == ""
        assert "1 word(s)" in abm_gpu_suite._transcript_problem(clip, sidecar)

    def test_long_audio_is_cut_into_whisper_sized_windows(self):
        import numpy as np

        rate = 16000
        t = np.arange(70 * rate) / rate
        speech = (0.2 * np.sin(2 * np.pi * 200 * t) * (np.sin(2 * np.pi * 0.25 * t) > -0.8)).astype(np.float32)
        windows = abm_gpu_suite._speech_windows(speech, rate)
        assert len(windows) >= 3
        assert all(len(window) <= 24 * rate for window in windows)
        assert sum(len(window) for window in windows) == (len(speech) // 320) * 320
        assert abm_gpu_suite._speech_windows(speech[: 10 * rate], rate)[0].shape == (10 * rate,)

    def test_notebook_transcript_setting_names_the_committed_file(self):
        notebook = generate_test_notebook.build("some-branch")
        settings = "".join(notebook["cells"][2]["source"])
        assert 'VOICE_TRANSCRIPT = "tests/kaggle/assets/voice/LOTM_narrator_voice.txt"' in settings
        assert os.path.exists(os.path.join(_ROOT, "tests/kaggle/assets/voice/LOTM_narrator_voice.txt"))
        clone_cell = "".join(notebook["cells"][6]["source"])
        assert "os.path.isfile(in_repo(words))" in clone_cell  # a path is read, not spoken
        assert "TQDM_DISABLE" in clone_cell

    def test_engine_install_steps_include_audiotools(self):
        for engine in ("indextts", "fish"):
            steps = " ".join(generate_test_notebook.PROVIDER_SETUP[engine]["post"])
            assert "--no-deps" in steps and "descript-audiotools" in steps


class TestNewStages:
    """Scaling, long book, languages, similarity, API and UI stages of the notebook."""

    def _notebook_commands(self) -> list[str]:
        """Every harness command line the generated notebook can run."""
        import re

        notebook = generate_test_notebook.build("some-branch")
        code = "\n".join("".join(cell["source"]) for cell in notebook["cells"] if cell["cell_type"] == "code")
        commands = []
        for match in re.finditer(r'suite\(\s*f?"((?:[^"\\]|\\.)*)"', code):
            commands.append(match.group(1))
        return commands

    def test_every_notebook_command_exists_in_the_harness(self):
        import re
        import shlex

        parser = abm_gpu_suite.build_parser()
        commands = self._notebook_commands()
        names = {command.split()[0] for command in commands}
        assert {"scaling", "longbook", "languages", "api", "ui", "similarity", "report"} <= names
        for command in commands:
            # Fill the notebook's f-string fields with plausible values.
            filled = re.sub(r"\{(?:ASR_FLAG|extra)\}", "", command)      # optional flags
            filled = re.sub(r"\{[^{}]*\}", "1", filled).replace("\\\"", "\"")
            filled = filled.replace("--name 1", "--name qwen").replace("--langs 1", "--langs fr")
            arguments = shlex.split(filled)
            if arguments[0] in ("provider", "preset", "scaling", "resume", "book", "cli", "longbook",
                                "languages", "api", "ui") and "--name" not in arguments:
                continue   # a continuation line of a longer command
            parser.parse_args(arguments)

    def test_long_books_are_found_in_the_assets(self):
        codes = [book["code"] for book in abm_gpu_suite._long_books()]
        assert codes == ["en", "fr", "ru", "hi", "zh", "ja", "ko"]
        entry = abm_gpu_suite._long_book("ja")
        assert entry["language"] == "Japanese"
        assert os.path.exists(os.path.join(sync_assets.BOOKS_DIR, entry["file"]))
        assert abm_gpu_suite._long_book("xx") is None

    def test_language_passage_is_about_the_requested_length(self):
        from audiobook_factory.chunk_verifier import expected_seconds

        for code in ("fr", "zh", "hi"):
            chapter = abm_gpu_suite._book_chapters(abm_gpu_suite._long_book(code))[0]
            passage = abm_gpu_suite._leading_paragraphs(chapter.text, 60.0)
            seconds = sum(expected_seconds(paragraph) for paragraph in passage)
            assert 60.0 <= seconds <= 150.0, f"{code}: {seconds:.0f}s"
            assert all(paragraph in chapter.text for paragraph in passage)

    def test_unsupported_language_is_skipped_not_failed(self, tmp_path, monkeypatch):
        # Qwen3-TTS has no Hindi; the test must say so instead of producing noise.
        monkeypatch.setattr(abm_gpu_suite, "RESULTS_DIR", str(tmp_path))
        args = abm_gpu_suite.build_parser().parse_args(["languages", "--name", "qwen", "--langs", "hi"])
        abm_gpu_suite.cmd_languages(args)
        with open(tmp_path / "lang_qwen_hi.json", encoding="utf-8") as fh:
            result = json.load(fh)
        assert result["status"] == "skip"
        assert "Hindi" in result["notes"][0]

    def test_scaling_text_is_several_batches_long(self):
        # A dozen chunks fit one batch on one GPU, which is why the first
        # scaling test showed almost no gain from the second GPU.
        from audiobook_factory.chunk_planner import plan_chunks

        chapters = abm_gpu_suite._book_chapters(abm_gpu_suite._long_book("en"))[:4]
        text = "\n\n".join(chapter.text for chapter in chapters)
        assert len(plan_chunks(text, None, 399, 0.3, 0.8)) >= 45
        args = abm_gpu_suite.build_parser().parse_args(["scaling", "--name", "qwen"])
        assert args.book_chapters == 4 and args.chunks == 0

    def test_helpers(self):
        assert list(abm_gpu_suite._walk({"a": [1, ("x", {"b": "y"})], "c": None})) == [1, "x", "y", None]
        import socket

        with socket.socket() as free:
            free.bind(("127.0.0.1", 0))
            port = free.getsockname()[1]
            assert abm_gpu_suite._port_in_use(port) is False   # bound but not listening
            free.listen(1)
            assert abm_gpu_suite._port_in_use(port) is True

    def test_report_renders_a_result_table(self, tmp_path, monkeypatch):
        results = tmp_path / "abm_results"
        results.mkdir()
        (results / "similarity.json").write_text(json.dumps({
            "test": "similarity", "status": "pass", "notes": ["1.0 = same voice."],
            "metrics": {"lowest_clone": 0.98, "table": {"headers": ["Sample", "Similarity"],
                                                        "rows": [["qwen_clone", "0.995"]]}},
        }))
        (results / "env.json").write_text(json.dumps({"test": "env", "status": "pass", "metrics": {}}))
        monkeypatch.setattr(abm_gpu_suite, "RESULTS_DIR", str(results))
        abm_gpu_suite.cmd_report(None)
        report = (results / "REPORT.md").read_text(encoding="utf-8")
        assert "| Sample | Similarity |" in report and "| qwen_clone | 0.995 |" in report
        assert "lowest_clone=0.98" in report

