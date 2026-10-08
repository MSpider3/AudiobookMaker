"""
test_cli_config.py
==================
Unit tests for cli.py that need no pipeline run: argument parsing, building
the AudiobookConfig from a progress JSON plus command-line overrides,
validation of untrusted settings, cover lookup, provider listing and the
Ctrl+C handler.
"""

from __future__ import annotations

import base64
import dataclasses
import json
import os
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import cli
from audiobook_factory.pipeline import AudiobookConfig, _CONFIG_SCHEMA_VERSION
from audiobook_factory.tts_providers.base_tts_provider import ProviderInfo, ProviderOption
from audiobook_factory.tts_providers.registry import provider_info, provider_names

_VOICE = os.path.join(_ROOT, "tests", "fixtures", "audio", "synthetic_voice_reference.wav")
_PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)

# Fields the CLI resolves for the machine it runs on, or deliberately resets.
_RESOLVED_FIELDS = {
    "config_version", "book_title", "book_path", "voice_file", "voice_preset", "output_dir",
    "cover_image", "preview_mode", "force_reprocess", "redo_chapters",
}
# A valid non-default value for the fields the pipeline restricts.
_ALTERNATIVES = {
    "tts_provider_name": "mock",
    "output_format": "m4b",
    "parallel_mode": "chapters",
    "quantization": "int8",
    "verify_chunks": "asr",
    "device": "cpu",
}


def _write_json(tmp_path, settings=None, chapters=None, **top_level):
    """Writes a progress JSON into tmp_path/job and returns its path."""
    job = tmp_path / "job"
    job.mkdir(exist_ok=True)
    data = {
        "book_title": "Unit Test Book",
        "settings": dict(settings or {}),
        "chapters": chapters if chapters is not None else [
            {"num": 1, "title": "Chapter 1", "status": "pending",
             "text": "Some text for the first chapter.", "sentences": ["Some text for the first chapter."]},
        ],
    }
    data["settings"].setdefault("config_version", _CONFIG_SCHEMA_VERSION)
    data.update(top_level)
    path = job / "generation_progress.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def _parse(*argv):
    return cli._build_parser().parse_args([str(a) for a in argv])


def _non_default(field: dataclasses.Field):
    """A value for *field* that differs from its default and is valid."""
    if field.name in _ALTERNATIVES:
        return _ALTERNATIVES[field.name]
    default = field.default if field.default is not dataclasses.MISSING else field.default_factory()
    if isinstance(default, bool):
        return not default
    if isinstance(default, int):
        return default + 3
    if isinstance(default, float):
        return default + 0.25
    if isinstance(default, str):
        return default + "-changed"
    if isinstance(default, list):
        return ["7. Some Chapter  (~10 words)"]
    if isinstance(default, dict):
        return {"key": "value"}
    return default


# ── Parser ────────────────────────────────────────────────────────────────────

class TestParser:

    def test_dry_run_is_registered(self):
        assert _parse("p.json", "--dry-run").dry_run is True
        assert _parse("p.json").dry_run is False

    def test_json_is_optional_with_book(self):
        args = _parse("--book", "novel.epub")
        assert args.config_json is None
        assert args.book == "novel.epub"

    def test_every_documented_flag_parses(self):
        args = _parse(
            "p.json", "--provider", "mock", "--language", "German",
            "--voice-transcript", "hello", "--voice-preset", "v.pt",
            "--speed", "1.2", "--temperature", "0.5", "--top-p", "0.9", "--seed", "7",
            "--bitrate", "96", "--sample-rate", "44100", "--channels", "2",
            "--pause", "0.3", "--para-pause", "0.9", "--max-len", "250",
            "--batch-size", "4", "--gpu-count", "2", "--verify", "asr",
            "--no-pack-sentences", "--no-normalize-text", "--single-file",
            "--redo", "3,7-9", "--chapters", "1-5,8",
            "--tts-option", "a=1", "--tts-option", "b=two",
            "--no-resume-chunks", "--force-reprocess", "--local",
        )
        assert args.redo == [3, 7, 8, 9]
        assert args.chapters == [1, 2, 3, 4, 5, 8]
        assert args.tts_option == ["a=1", "b=two"]
        assert args.verify == "asr" and args.channels == 2 and args.single_file is True

    def test_transcript_text_and_file_are_exclusive(self):
        with pytest.raises(SystemExit):
            _parse("p.json", "--voice-transcript", "x", "--voice-transcript-file", "t.txt")

    @pytest.mark.parametrize("bad", ["abc", "3-1", "0", "1,,x", ""])
    def test_bad_chapter_lists_are_rejected(self, bad):
        with pytest.raises(SystemExit):
            _parse("p.json", "--chapters", bad)

    def test_help_epilog_mentions_new_modes(self):
        text = cli._build_parser().format_help()
        for needle in ("--book book.epub", "--dry-run", "--list-providers", "--redo 3,7-9", "Exit status"):
            assert needle in text


# ── Config building ───────────────────────────────────────────────────────────

class TestBuildConfig:

    def test_every_config_field_round_trips(self, tmp_path):
        """A field added to AudiobookConfig later must survive without touching cli.py."""
        settings = {f.name: _non_default(f) for f in dataclasses.fields(AudiobookConfig)}
        path = _write_json(tmp_path, settings={k: v for k, v in settings.items() if k != "output_dir"})
        meta, loaded, _chapters, _ = cli._load_config(_parse(path))
        cfg = cli._build_audiobook_config(meta, loaded, write_cover=False)

        checked = 0
        for f in dataclasses.fields(AudiobookConfig):
            if f.name in _RESOLVED_FIELDS:
                continue
            assert getattr(cfg, f.name) == settings[f.name], f"field '{f.name}' was dropped"
            checked += 1
        assert checked >= 40

    def test_new_schema_fields_survive(self, tmp_path):
        path = _write_json(tmp_path, settings={
            "tts_provider_name": "mock",
            "tts_options": {"alpha": 0.5, "flag": True, "name": "x"},
            "batch_size": 6, "pack_sentences": False, "normalize_speech_text": False,
            "verify_chunks": "asr", "verify_max_retries": 5,
            "verify_asr_model": "openai/whisper-small", "verify_max_wer": 0.15,
        })
        meta, settings, _, _ = cli._load_config(_parse(path))
        cfg = cli._build_audiobook_config(meta, settings, write_cover=False)
        assert cfg.tts_options == {"alpha": 0.5, "flag": True, "name": "x"}
        assert (cfg.batch_size, cfg.pack_sentences, cfg.normalize_speech_text) == (6, False, False)
        assert (cfg.verify_chunks, cfg.verify_max_retries) == ("asr", 5)
        assert (cfg.verify_asr_model, cfg.verify_max_wer) == ("openai/whisper-small", 0.15)

    def test_flags_reach_the_config(self, tmp_path):
        transcript = tmp_path / "transcript.txt"
        transcript.write_text("  What the narrator says.\n", encoding="utf-8")
        out = tmp_path / "elsewhere"
        path = _write_json(tmp_path, settings={
            "tts_provider_name": "qwen", "resume_incomplete_chunks": True,
            "pack_sentences": True, "normalize_speech_text": True, "single_file_mode": False,
        })
        args = _parse(
            path, "--provider", "mock", "--language", "German",
            "--voice-file", _VOICE, "--voice-preset", _VOICE,
            "--voice-transcript-file", transcript,
            "--speed", "1.2", "--temperature", "0.55", "--top-p", "0.91", "--seed", "7",
            "--bitrate", "96", "--sample-rate", "44100", "--channels", "2",
            "--pause", "0.3", "--para-pause", "0.9", "--max-len", "250",
            "--batch-size", "4", "--gpu-count", "2", "--verify", "off",
            "--no-pack-sentences", "--no-normalize-text", "--single-file",
            "--no-resume-chunks", "--force-reprocess", "--redo", "2,4-5",
            "--output-dir", out, "--output-format", "flac", "--worker-count", "3",
            "--device", "cpu", "--quantization", "int8",
        )
        meta, settings, _, _ = cli._load_config(args)
        cfg = cli._build_audiobook_config(meta, settings, write_cover=False)

        assert cfg.tts_provider_name == "mock"
        assert cfg.language == "German"
        assert cfg.voice_file == _VOICE and cfg.voice_preset == _VOICE
        assert cfg.voice_transcript == "What the narrator says."
        assert (cfg.speed, cfg.temperature, cfg.top_p, cfg.seed) == (1.2, 0.55, 0.91, 7)
        assert (cfg.bitrate_kbps, cfg.sample_rate, cfg.channels) == (96, 44100, 2)
        assert (cfg.pause, cfg.para_pause, cfg.max_len) == (0.3, 0.9, 250)
        assert (cfg.batch_size, cfg.gpu_count, cfg.verify_chunks) == (4, 2, "off")
        assert cfg.pack_sentences is False and cfg.normalize_speech_text is False
        assert cfg.single_file_mode is True
        assert cfg.resume_incomplete_chunks is False
        assert cfg.force_reprocess is True and cfg.redo_chapters == [2, 4, 5]
        assert cfg.output_dir == str(out) and cfg.output_format == "flac"
        assert (cfg.worker_count, cfg.device, cfg.quantization) == (3, "cpu", "int8")

    def test_one_shot_settings_come_only_from_the_command_line(self, tmp_path, capsys):
        """The pipeline writes the last run's settings back; a resume must not
        wipe or redo finished chapters because of them."""
        path = _write_json(tmp_path, settings={
            "tts_provider_name": "mock", "force_reprocess": True, "redo_chapters": [1, 2],
        })
        meta, settings, _, _ = cli._load_config(_parse(path))
        cfg = cli._build_audiobook_config(meta, settings, write_cover=False)
        assert cfg.force_reprocess is False
        assert cfg.redo_chapters == []
        assert "--force-reprocess" in capsys.readouterr().out

    def test_preview_mode_in_json_does_not_disable_generation(self, tmp_path):
        path = _write_json(tmp_path, settings={"tts_provider_name": "mock", "preview_mode": True})
        meta, settings, _, _ = cli._load_config(_parse(path))
        assert cli._build_audiobook_config(meta, settings, write_cover=False).preview_mode is False

    def test_string_numbers_are_coerced_and_garbage_rejected(self, tmp_path):
        path = _write_json(tmp_path, settings={
            "tts_provider_name": "mock", "worker_count": "2", "speed": "1.5",
            "export_srt": "true", "max_len": 300.0, "seed": None,
        })
        meta, settings, _, _ = cli._load_config(_parse(path))
        cfg = cli._build_audiobook_config(meta, settings, write_cover=False)
        assert cfg.worker_count == 2 and isinstance(cfg.worker_count, int)
        assert cfg.speed == 1.5 and cfg.export_srt is True
        assert cfg.max_len == 300 and isinstance(cfg.max_len, int)
        assert cfg.seed == -1

        settings["worker_count"] = "many"
        with pytest.raises(ValueError, match="worker_count"):
            cli._build_audiobook_config(meta, settings, write_cover=False)

    def test_config_version_is_current_after_loading_an_old_file(self, tmp_path):
        path = _write_json(tmp_path, settings={"tts_provider_name": "mock", "config_version": 5})
        meta, settings, _, _ = cli._load_config(_parse(path))
        assert cli._build_audiobook_config(meta, settings, write_cover=False).config_version == _CONFIG_SCHEMA_VERSION

    def test_switching_engine_drops_the_other_engines_model_and_options(self, tmp_path):
        path = _write_json(tmp_path, settings={
            "tts_provider_name": "f5tts", "tts_model_name": "F5TTS_v1_Base",
            "tts_options": {"nfe_step": 16},
        })
        meta, settings, _, _ = cli._load_config(_parse(path, "--provider", "qwen"))
        cfg = cli._build_audiobook_config(meta, settings, write_cover=False)
        assert cfg.tts_provider_name == "qwen"
        assert cfg.tts_options == {}
        assert cfg.tts_model_name == provider_info("qwen").default_model

    def test_default_output_dir_is_under_the_project(self, tmp_path):
        path = _write_json(tmp_path, settings={"tts_provider_name": "mock"}, book_title='My: "Book"?')
        meta, settings, _, _ = cli._load_config(_parse(path))
        cfg = cli._build_audiobook_config(meta, settings, write_cover=False)
        assert cfg.output_dir == os.path.join(_ROOT, "audiobook_output", "My Book")

    def test_voice_file_prefers_a_path_that_exists(self, tmp_path):
        path = _write_json(
            tmp_path, settings={"tts_provider_name": "mock", "voice_file": "/gradio/tmp/gone.wav"},
            voice_file=_VOICE,
        )
        meta, settings, _, _ = cli._load_config(_parse(path))
        assert cli._build_audiobook_config(meta, settings, write_cover=False).voice_file == _VOICE


# ── Untrusted settings ────────────────────────────────────────────────────────

class TestSettingsValidation:

    @pytest.mark.parametrize("options", [
        ["a", "b"], "num_step=4", 3,
        {"nested": {"a": 1}}, {"items": [1, 2]}, {"ok": 1, "bad": {"x": "y"}},
    ])
    def test_tts_options_must_be_a_dict_of_scalars(self, tmp_path, options):
        path = _write_json(tmp_path, settings={"tts_provider_name": "mock", "tts_options": options})
        with pytest.raises(ValueError, match="tts_options"):
            cli._load_config(_parse(path))

    def test_scalar_tts_options_are_accepted(self, tmp_path):
        options = {"a": 1, "b": 0.5, "c": True, "d": "text", "e": None}
        path = _write_json(tmp_path, settings={"tts_provider_name": "mock", "tts_options": options})
        _meta, settings, _, _ = cli._load_config(_parse(path))
        assert settings["tts_options"] == options

    def test_unknown_provider_is_rejected_with_the_known_names(self, tmp_path):
        path = _write_json(tmp_path, settings={"tts_provider_name": "evil-engine"})
        with pytest.raises(ValueError, match="invalid tts_provider_name") as excinfo:
            cli._load_config(_parse(path))
        for name in provider_names():
            assert name in str(excinfo.value)

    def test_foreign_output_dir_still_raises_by_default(self, tmp_path):
        path = _write_json(tmp_path, settings={
            "tts_provider_name": "mock", "output_dir": "/home/someone-else/audiobook_output/Book",
        })
        with pytest.raises(ValueError, match="Untrusted output_dir"):
            cli._load_config(_parse(path))

    def test_foreign_output_dir_falls_back_to_the_default(self, tmp_path, capsys):
        """A JSON exported on another machine must not abort the run (and its
        path must never be used)."""
        foreign = "/home/someone-else/audiobook_output/Book"
        path = _write_json(tmp_path, settings={"tts_provider_name": "mock", "output_dir": foreign})
        meta, settings, _, _ = cli._load_config(_parse(path), fallback_output_dir=True)
        cfg = cli._build_audiobook_config(meta, settings, write_cover=False)

        assert "output_dir" not in settings
        assert cfg.output_dir == os.path.join(_ROOT, "audiobook_output", "Unit Test Book")
        out = capsys.readouterr().out
        assert "Untrusted output_dir" in out and "default output directory" in out

    def test_explicit_output_dir_is_trusted(self, tmp_path):
        path = _write_json(tmp_path, settings={"tts_provider_name": "mock", "output_dir": "/etc/cron.d"})
        target = tmp_path / "chosen"
        _meta, settings, _, _ = cli._load_config(_parse(path, "--output-dir", target))
        assert settings["output_dir"] == str(target)

    def test_missing_file_arguments_fail_cleanly(self, tmp_path):
        path = _write_json(tmp_path, settings={"tts_provider_name": "mock"})
        for flag in ("--voice-file", "--cover-image", "--voice-preset", "--book-path", "--voice-transcript-file"):
            with pytest.raises(cli._CliError, match="file not found"):
                cli._load_config(_parse(path, flag, tmp_path / "nope.bin"))


# ── --tts-option ──────────────────────────────────────────────────────────────

def _fake_info(**overrides) -> ProviderInfo:
    values = dict(
        name="mock", display_name="Fake Engine", license="Test-1.0", commercial_use=False,
        default_model="fake/model", models=("fake/model",), min_vram_gb=3.5,
        languages=("English", "German"),
        options=(
            ProviderOption(key="steps", label="Steps", kind="int", default=32, minimum=4, maximum=64),
            ProviderOption(key="guidance", label="Guidance", kind="float", default=2.0),
            ProviderOption(key="denoise", label="Denoise", kind="bool", default=True),
            ProviderOption(key="mode", label="Mode", kind="choice", default="auto", choices=("auto", "fast")),
            ProviderOption(key="note", label="Note", kind="str", default=""),
            ProviderOption(key="emotion_clip", label="Emotion clip", kind="file", default=""),
        ),
        pip_requirements=("abm-engine-that-does-not-exist>=1.0",),
        install_notes="Needs a GPU with 4 GB.",
    )
    values.update(overrides)
    return ProviderInfo(**values)


class TestTtsOptionFlag:

    @pytest.fixture(autouse=True)
    def _fake_provider(self, monkeypatch):
        monkeypatch.setattr(cli, "_provider_info", lambda name: _fake_info())

    def test_values_are_coerced_to_the_declared_kind(self):
        parsed = cli._parse_tts_options("mock", [
            "steps=16", "guidance=1.5", "denoise=false", "mode=fast", "note=a=b", f"emotion_clip={_VOICE}",
        ])
        assert parsed == {
            "steps": 16, "guidance": 1.5, "denoise": False, "mode": "fast",
            "note": "a=b", "emotion_clip": _VOICE,
        }
        assert isinstance(parsed["steps"], int) and isinstance(parsed["guidance"], float)

    def test_unknown_key_lists_the_valid_ones(self):
        with pytest.raises(cli._CliError) as excinfo:
            cli._parse_tts_options("mock", ["stepz=16"])
        message = str(excinfo.value)
        assert "stepz" in message
        for key in ("steps", "guidance", "denoise", "mode", "note", "emotion_clip"):
            assert key in message

    @pytest.mark.parametrize("pair", [
        "steps=many", "guidance=x", "denoise=maybe", "mode=slow", "novalue", "=3",
        "emotion_clip=/no/such/clip.wav",
    ])
    def test_bad_values_are_rejected(self, pair):
        with pytest.raises(cli._CliError):
            cli._parse_tts_options("mock", [pair])

    def test_flag_merges_over_options_from_the_json(self, tmp_path):
        path = _write_json(tmp_path, settings={
            "tts_provider_name": "mock", "tts_options": {"steps": 8, "note": "keep"},
        })
        meta, settings, _, _ = cli._load_config(_parse(path, "--tts-option", "steps=24", "--tts-option", "denoise=no"))
        cfg = cli._build_audiobook_config(meta, settings, write_cover=False)
        assert cfg.tts_options == {"steps": 24, "note": "keep", "denoise": False}


# ── Providers ─────────────────────────────────────────────────────────────────

class TestProviders:

    def test_list_providers_prints_every_registered_engine(self, capsys):
        cli._print_providers()
        out = capsys.readouterr().out
        for name in provider_names():
            assert name in out
            try:
                info = provider_info(name)
            except Exception:
                assert "unavailable" in out
                continue
            assert info.display_name in out
            assert info.license in out
            assert f"{info.min_vram_gb:g} GB" in out
            if info.default_model:
                assert info.default_model in out
            for option in info.options:
                assert f"{option.key}={option.default!r}" in out
            if info.pip_requirements:
                assert "pip install" in out

    def test_list_providers_shows_licence_flag_and_install_command(self, capsys, monkeypatch):
        monkeypatch.setattr(cli, "_provider_info", lambda name: _fake_info())
        cli._print_providers()
        out = capsys.readouterr().out
        assert "Fake Engine" in out and "Test-1.0" in out
        assert "NON-COMMERCIAL" in out
        assert "fake/model" in out and "3.5 GB" in out
        assert "Languages:     2" in out
        assert "steps=32" in out and "mode='auto'" in out
        assert "pip install 'abm-engine-that-does-not-exist>=1.0'" in out
        assert "not installed" in out

    def test_missing_requirements_checks_installed_distributions(self):
        missing = cli._missing_requirements([
            "pytest", "abm-engine-that-does-not-exist>=1.0",
            'abm-windows-only>=1; sys_platform == "no-such-platform"',
            "pytest[extra]>=1.0",
        ])
        assert missing == ["abm-engine-that-does-not-exist>=1.0"]

    def test_not_installed_engine_fails_with_install_command(self, monkeypatch):
        monkeypatch.delenv("ABM_SKIP_INSTALL_CHECK", raising=False)
        monkeypatch.setattr(cli, "_provider_info", lambda name: _fake_info())
        with pytest.raises(cli._CliError) as excinfo:
            cli._check_provider_installed("mock")
        message = str(excinfo.value)
        assert "not installed" in message
        assert "pip install 'abm-engine-that-does-not-exist>=1.0'" in message
        assert "Needs a GPU with 4 GB." in message

    def test_unimportable_engine_fails_without_traceback(self, monkeypatch):
        def _boom(name):
            raise ModuleNotFoundError("No module named 'some_provider'")
        monkeypatch.setattr(cli, "_provider_info", _boom)
        with pytest.raises(cli._CliError, match="not available in this installation"):
            cli._check_provider_installed("fish")

    def test_installed_engine_passes(self, monkeypatch):
        monkeypatch.setattr(cli, "_provider_info", lambda name: _fake_info(pip_requirements=("pytest",)))
        cli._check_provider_installed("mock")

    def test_import_errors_are_found_in_exception_chains(self):
        try:
            try:
                raise ModuleNotFoundError("No module named 'omnivoice'")
            except ImportError as inner:
                raise RuntimeError("could not load model") from inner
        except RuntimeError as outer:
            assert cli._has_import_error(outer) is True
        assert cli._has_import_error(RuntimeError("CUDA out of memory")) is False


# ── Cover lookup ──────────────────────────────────────────────────────────────

class TestCoverLookup:

    def test_current_directory_is_never_searched(self, tmp_path, monkeypatch):
        """A cover.jpg or an EPUB lying in the working directory belongs to
        some other book."""
        cwd = tmp_path / "cwd"
        (cwd / "sub").mkdir(parents=True)
        (cwd / "cover.jpg").write_bytes(_PNG_1X1)
        (cwd / "cover.png").write_bytes(_PNG_1X1)
        for target in (cwd / "other.epub", cwd / "sub" / "another.epub"):
            target.write_bytes(open(os.path.join(
                _ROOT, "tests", "fixtures", "source_documents", "dummy_book.epub"), "rb").read())
        monkeypatch.chdir(cwd)

        path = _write_json(tmp_path, settings={"tts_provider_name": "mock", "output_dir": str(tmp_path / "job" / "out")})
        meta, settings, _, _ = cli._load_config(_parse(path))
        cfg = cli._build_audiobook_config(meta, settings)
        assert cfg.cover_image is None
        assert not os.path.exists(os.path.join(cfg.output_dir, "cover.jpg"))

    def test_cover_next_to_the_json_is_used(self, tmp_path):
        path = _write_json(tmp_path, settings={"tts_provider_name": "mock"})
        cover = path.parent / "cover.png"
        cover.write_bytes(_PNG_1X1)
        meta, settings, _, _ = cli._load_config(_parse(path))
        assert cli._build_audiobook_config(meta, settings, write_cover=False).cover_image == str(cover)

    def test_cover_in_the_output_dir_is_used(self, tmp_path):
        out = tmp_path / "job" / "out"
        out.mkdir(parents=True)
        (out / "cover.webp").write_bytes(_PNG_1X1)
        path = _write_json(tmp_path, settings={"tts_provider_name": "mock", "output_dir": str(out)})
        meta, settings, _, _ = cli._load_config(_parse(path))
        assert cli._build_audiobook_config(meta, settings, write_cover=False).cover_image == str(out / "cover.webp")

    def test_cover_next_to_the_book_is_used(self, tmp_path):
        library = tmp_path / "library"
        library.mkdir()
        book = library / "book.txt"
        book.write_text("A short book.", encoding="utf-8")
        (library / "cover.jpeg").write_bytes(_PNG_1X1)
        path = _write_json(tmp_path, settings={"tts_provider_name": "mock"})
        meta, settings, _, _ = cli._load_config(_parse(path, "--book-path", book))
        assert cli._build_audiobook_config(meta, settings, write_cover=False).cover_image == str(library / "cover.jpeg")

    def test_embedded_cover_is_written_to_the_output_dir(self, tmp_path):
        out = tmp_path / "job" / "out"
        path = _write_json(
            tmp_path, settings={"tts_provider_name": "mock", "output_dir": str(out)},
            cover_image_b64=base64.b64encode(_PNG_1X1).decode("ascii"),
        )
        meta, settings, _, _ = cli._load_config(_parse(path))

        dry = cli._build_audiobook_config(meta, dict(settings), write_cover=False)
        assert dry.cover_image is None and not out.exists()

        cfg = cli._build_audiobook_config(meta, settings)
        assert cfg.cover_image == str(out / "cover.png")
        assert (out / "cover.png").read_bytes() == _PNG_1X1

    def test_explicit_cover_wins_over_embedded(self, tmp_path):
        mine = tmp_path / "mine.png"
        mine.write_bytes(_PNG_1X1)
        path = _write_json(
            tmp_path, settings={"tts_provider_name": "mock"},
            cover_image_b64=base64.b64encode(b"not-the-one").decode("ascii"),
        )
        meta, settings, _, _ = cli._load_config(_parse(path, "--cover-image", mine))
        assert cli._build_audiobook_config(meta, settings, write_cover=False).cover_image == str(mine)


# ── Chapters ──────────────────────────────────────────────────────────────────

def _entries(*titles):
    return [
        {"num": i, "title": title, "status": "pending", "text": f"Text of {title}.", "sentences": [f"Text of {title}."]}
        for i, title in enumerate(titles, 1)
    ]


class TestChapterLoading:

    def _cfg(self, **kwargs):
        return AudiobookConfig(tts_provider_name="mock", **kwargs)

    def test_empty_chapter_list_is_not_a_text_cache(self, tmp_path):
        """all() of nothing is True: an empty list used to mean "all done"."""
        with pytest.raises(cli._CliError, match="No cached chapter text"):
            cli._load_chapters([], {}, self._cfg(book_path=""))

    def test_empty_chapter_list_extracts_from_the_book(self, tmp_path):
        book = tmp_path / "book.txt"
        book.write_text("It was a quiet morning. Nothing moved on the road.", encoding="utf-8")
        chapters = cli._load_chapters([], {}, self._cfg(book_path=str(book)))
        assert len(chapters) == 1 and "quiet morning" in chapters[0].text

    def test_chapters_flag_restricts_by_number(self):
        raw = _entries("One", "Two", "Three", "Four")
        chapters = cli._load_chapters(raw, {}, self._cfg(), only_nums=[2, 4])
        assert [(c.num, c.title) for c in chapters] == [(2, "Two"), (4, "Four")]

    def test_chapters_flag_replaces_the_saved_selection(self):
        raw = _entries("One", "Two", "Three")
        cfg = self._cfg(selected_chapters=["1. One  (~3 words)"])
        assert [c.num for c in cli._load_chapters(raw, {}, cfg, only_nums=[3])] == [3]

    def test_chapters_flag_matching_nothing_is_an_error(self):
        with pytest.raises(cli._CliError, match="1-3"):
            cli._load_chapters(_entries("One", "Two", "Three"), {}, self._cfg(), only_nums=[9])

    def test_saved_selection_matches_titles_exactly(self):
        raw = _entries("Chapter 1", "Chapter 10", "Chapter 11")
        cfg = self._cfg(selected_chapters=["1. Chapter 1  (~3 words)"])
        assert [c.title for c in cli._load_chapters(raw, {}, cfg)] == ["Chapter 1"]

    def test_string_and_missing_chapter_numbers_are_tolerated(self):
        raw = [
            {"num": "4", "title": "A", "text": "Text a.", "sentences": []},
            {"title": "B", "text": "Text b.", "sentences": []},
        ]
        assert [c.num for c in cli._load_chapters(raw, {}, self._cfg())] == [4, 2]


# ── Progress file seeding ─────────────────────────────────────────────────────

class TestSeedProgressFile:

    def test_copies_when_the_output_dir_has_none(self, tmp_path):
        given = tmp_path / "given.json"
        given.write_text('{"chapters": []}', encoding="utf-8")
        dest = tmp_path / "out.json"
        cli._seed_progress_file(str(given), str(dest))
        assert dest.read_text(encoding="utf-8") == '{"chapters": []}'

    def test_never_overwrites_newer_progress(self, tmp_path, capsys):
        given = tmp_path / "given.json"
        given.write_text('{"chapters": [{"num": 1, "status": "pending"}]}', encoding="utf-8")
        dest = tmp_path / "out.json"
        newer = '{"chapters": [{"num": 1, "status": "completed"}]}'
        dest.write_text(newer, encoding="utf-8")
        cli._seed_progress_file(str(given), str(dest))
        assert dest.read_text(encoding="utf-8") == newer
        assert "Resuming" in capsys.readouterr().out

    def test_same_file_is_left_alone(self, tmp_path):
        path = tmp_path / "generation_progress.json"
        path.write_text("{}", encoding="utf-8")
        cli._seed_progress_file(str(path), str(path))
        assert path.read_text(encoding="utf-8") == "{}"


# ── Ctrl+C ────────────────────────────────────────────────────────────────────

class _Token:
    def __init__(self):
        self.is_cancelled = False

    def cancel(self):
        self.is_cancelled = True


class TestSigint:

    def test_importing_cli_installs_no_handler(self):
        import signal
        import subprocess

        code = (
            "import signal, cli; "
            "print(signal.getsignal(signal.SIGINT) is signal.default_int_handler)"
        )
        result = subprocess.run(
            [sys.executable, "-c", code], cwd=_ROOT, capture_output=True, text=True, timeout=120,
        )
        assert result.stdout.strip().endswith("True"), result.stderr[-500:]
        assert signal.SIGINT  # keeps the import used

    def test_first_interrupt_cancels_second_exits_130(self, capsys):
        token = _Token()
        exits = []
        handler = cli._make_sigint_handler(token, hard_exit=exits.append)

        handler(2, None)
        assert token.is_cancelled is True and exits == []
        assert "again" in capsys.readouterr().out

        handler(2, None)
        assert exits == [130]

    def test_handler_is_installed_and_restored(self):
        import signal

        before = signal.getsignal(signal.SIGINT)
        previous = cli._install_sigint_handler(_Token())
        try:
            assert signal.getsignal(signal.SIGINT) is not before
        finally:
            cli._restore_sigint_handler(previous)
        assert signal.getsignal(signal.SIGINT) is before
