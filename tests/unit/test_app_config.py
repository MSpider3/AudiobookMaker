"""
tests/unit/test_app_config.py
=============================
The UI builds its ``AudiobookConfig`` in one place (``app._build_config``) and
restores it in one place (``app.on_progress_upload_handler``). These tests
pin down that

* no config field is silently left out of the UI,
* every engine's capabilities and options reach the config,
* a config exported from the UI comes back unchanged when it is restored,
* a damaged or foreign JSON never leaves a control in an invalid state.
"""
from __future__ import annotations

import dataclasses
import json
import os
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import gradio as gr  # noqa: E402

import app  # noqa: E402
from audiobook_factory.pipeline import AudiobookConfig  # noqa: E402
from audiobook_factory.tts_providers import registry  # noqa: E402

_VOICE = os.path.join(_ROOT, "tests", "fixtures", "audio", "synthetic_voice_reference.wav")
_QWEN_BASE = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
_QWEN_CUSTOM = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
_QWEN_CUSTOM_SMALL = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
_QWEN_DESIGN = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"

# Fields that hold a path of this machine or session; a JSON never restores them.
_PATH_FIELDS = {"voice_file", "voice_preset", "cover_image", "book_path", "output_dir"}


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "_OUTPUT_DIR", str(tmp_path / "audiobook_output"))
    monkeypatch.setenv("ABM_API_URL", "")
    monkeypatch.setenv("ABM_SKIP_ENGINE_CHECK", "1")   # the engines are not installed in CI
    monkeypatch.delenv("ABM_MULTI_USER", raising=False)


def _non_default_ui(tmp_path, **overrides) -> dict:
    """UI values that all differ from the dataclass defaults."""
    preset = tmp_path / "voice.pt"
    preset.write_bytes(b"preset")
    cover = tmp_path / "cover.png"
    cover.write_bytes(b"\x89PNG\r\n\x1a\n")
    fixes = tmp_path / "fixes.txt"
    fixes.write_text("# comment\nSaltmarsh == Salt-marsh\n", encoding="utf-8")
    ui = {
        "book_title": "My Book", "author": "Ann Author", "language": "German",
        "cover_image": str(cover), "output_format": "m4b", "lufs": -20,
        "selected_chapters": ["2. Chapter Two  (~10 words)"],
        "tts_provider_name": "qwen", "tts_model_name": _QWEN_BASE,
        "tts_timbre": "Ryan", "tts_instruct": "calm and warm",
        "voice_file": _VOICE, "voice_transcript": "hello there", "voice_preset": str(preset),
        "tts_options": {"qwen": {"x_vector_only_mode": True, "subtalker_top_k": 40}},
        "temperature": 0.7, "top_p": 0.9, "top_k": 30, "repetition_penalty": 1.2, "seed": 7,
        "speed": 1.25, "pause": 0.3, "para_pause": 0.9,
        "max_len": 250, "pack_sentences": False, "normalize_speech_text": False,
        "verify_chunks": "asr", "verify_max_retries": 4, "verify_asr_model": "openai/whisper-small",
        "verify_max_wer": 0.5,
        "batch_size": 3, "gpu_count": 1, "vram_headroom_gb": 3.5, "max_chapter_retries": 4,
        "parallel_mode": "chapters", "torch_compile": True, "quantization": "int8",
        "sample_rate": 44100, "bitrate_kbps": 128, "channels": 2, "true_peak": -2.0,
        "force_reprocess": True, "export_text": True,
        "single_file_mode": True, "export_lrc": False, "export_srt": True, "export_vtt": True,
        "regen_missing": False, "resume_incomplete_chunks": False,
        "pronunciation_file": str(fixes), "pronunciation_table": [["Hale", "Hayl"], ["", ""]],
    }
    ui.update(overrides)
    assert set(ui) == set(app._UI_KEYS)
    return ui


def _changed_fields(cfg: AudiobookConfig) -> set[str]:
    default = AudiobookConfig()
    return {f.name for f in dataclasses.fields(cfg) if getattr(cfg, f.name) != getattr(default, f.name)}


class TestBuildConfig:

    def test_every_config_field_comes_from_the_ui_or_is_declared_unexposed(self, tmp_path):
        """A field added to AudiobookConfig must get a control or an explicit exemption."""
        clone = app._build_config(_non_default_ui(tmp_path), book_path="/b.epub", redo_chapters=[3])
        speaker = app._build_config(_non_default_ui(tmp_path, tts_model_name=_QWEN_CUSTOM))
        other = app._build_config(_non_default_ui(tmp_path, tts_provider_name="indextts"))
        produced = _changed_fields(clone) | _changed_fields(speaker) | _changed_fields(other)
        all_fields = {f.name for f in dataclasses.fields(AudiobookConfig)}
        derived = {"book_path", "output_dir", "redo_chapters"}
        missing = all_fields - produced - app._CONFIG_FIELDS_NOT_IN_UI
        assert not missing, f"AudiobookConfig fields the UI never sets: {sorted(missing)}"
        assert app._CONFIG_FIELDS_NOT_IN_UI <= all_fields
        assert not (app._CONFIG_FIELDS_NOT_IN_UI - derived) & produced

    def test_values_arrive_with_the_right_types(self, tmp_path):
        cfg = app._build_config(_non_default_ui(tmp_path))
        assert cfg.voice_transcript == "hello there"
        assert cfg.voice_preset.endswith("voice.pt")
        assert cfg.tts_options == {"x_vector_only_mode": True, "subtalker_top_k": 40}
        assert cfg.verify_chunks == "asr" and cfg.verify_max_retries == 4
        assert cfg.batch_size == 3 and cfg.gpu_count == 1 and cfg.vram_headroom_gb == 3.5
        assert cfg.pack_sentences is False and cfg.normalize_speech_text is False
        assert cfg.pronunciation_map == {"Saltmarsh": "Salt-marsh", "Hale": "Hayl"}
        assert isinstance(cfg.sample_rate, int) and isinstance(cfg.seed, int) and isinstance(cfg.lufs, int)
        assert cfg.output_dir == os.path.join(app._OUTPUT_DIR, "My Book")

    def test_files_that_no_longer_exist_are_sent_as_no_file(self, tmp_path):
        """The backend answers 400 for a path that does not exist."""
        ui = _non_default_ui(tmp_path, cover_image="/gone/cover.png",
                             tts_provider_name="indextts",
                             tts_options={"indextts": {"emo_audio_prompt": "/gone/emotion.wav"}})
        cfg = app._build_config(ui)
        assert cfg.cover_image is None
        assert "emo_audio_prompt" not in cfg.tts_options
        ui["tts_options"] = {"indextts": {"emo_audio_prompt": _VOICE}}
        assert app._build_config(ui).tts_options == {"emo_audio_prompt": _VOICE}

    def test_missing_and_junk_values_fall_back_to_defaults(self):
        cfg = app._build_config({"sample_rate": "fast", "verify_chunks": "maybe", "output_format": "exe",
                                 "seed": None, "tts_provider_name": None})
        default = AudiobookConfig()
        assert cfg.sample_rate == default.sample_rate
        assert cfg.verify_chunks == default.verify_chunks
        assert cfg.output_format == default.output_format
        assert cfg.seed == default.seed
        assert cfg.tts_provider_name == default.tts_provider_name

    def test_controls_hidden_for_a_checkpoint_do_not_leak_into_the_config(self, tmp_path):
        speaker = app._build_config(_non_default_ui(tmp_path, tts_model_name=_QWEN_CUSTOM))
        assert speaker.voice_file == "" and speaker.voice_transcript == ""
        assert speaker.tts_timbre == "Ryan" and speaker.tts_instruct == "calm and warm"
        small = app._build_config(_non_default_ui(tmp_path, tts_model_name=_QWEN_CUSTOM_SMALL))
        assert small.tts_instruct == ""
        clone = app._build_config(_non_default_ui(tmp_path))
        assert clone.voice_file == _VOICE and clone.tts_timbre == "" and clone.tts_instruct == ""

    def test_a_model_of_another_engine_is_replaced_by_the_engines_default(self, tmp_path):
        cfg = app._build_config(_non_default_ui(tmp_path, tts_provider_name="indextts"))
        assert cfg.tts_model_name == registry.provider_info("indextts").default_model

    def test_book_title_cannot_escape_the_output_folder(self):
        for title in ("..", "../../etc", "a/b", "  ", "con:fig?"):
            out = app._book_output_dir(title)
            assert os.path.dirname(out) == app._OUTPUT_DIR
            assert os.path.basename(out) not in ("", ".", "..")


class TestProviderOptions:

    @pytest.mark.parametrize("name", registry.provider_names(include_hidden=True))
    def test_every_option_of_every_engine_gets_a_control(self, name):
        info = app._provider_info(name)
        if info is None:
            pytest.skip(f"{name} does not import here")
        factories = {
            "checkbox": gr.Checkbox, "dropdown": gr.Dropdown, "slider": gr.Slider,
            "number": gr.Number, "textbox": gr.Textbox, "file": gr.File,
        }
        for option in info.options:
            kind, kwargs = app._option_spec(option)
            assert kind in factories, f"{name}.{option.key}: no control for kind {option.kind!r}"
            component = factories[kind](**kwargs)          # the arguments must be accepted
            if kind == "dropdown":
                assert kwargs["value"] in kwargs["choices"]
            if kind == "slider":
                assert kwargs["minimum"] <= kwargs["value"] <= kwargs["maximum"]
            if option.help and kind != "file":
                assert component.info == option.help
            # The default round-trips to "nothing to send".
            assert app._clean_tts_options(name, {option.key: option.default}) == {}

    def test_only_changed_declared_options_are_sent(self):
        cleaned = app._clean_tts_options("qwen", {
            "x_vector_only_mode": True,          # changed
            "auto_transcribe": True,             # default
            "subtalker_top_k": "40",             # string from a form
            "asr_model": "not-a-model",          # not one of the choices
            "made_up": 1,                        # not declared
        })
        assert cleaned == {"x_vector_only_mode": True, "subtalker_top_k": 40}

    def test_options_are_kept_per_engine(self):
        state = app.set_provider_option(True, {}, "qwen", "x_vector_only_mode")
        state = app.set_provider_option(0.4, state, "indextts", "emo_alpha")
        assert app._clean_tts_options("qwen", app._provider_options_for(state, "qwen")) == {"x_vector_only_mode": True}
        assert app._clean_tts_options("indextts", app._provider_options_for(state, "indextts")) == {"emo_alpha": 0.4}
        cfg = app._build_config({"tts_provider_name": "indextts", "tts_options": state})
        assert cfg.tts_options == {"emo_alpha": 0.4}


class TestEngineDrivenUi:

    def test_provider_dropdown_comes_from_the_registry(self, monkeypatch):
        monkeypatch.delenv("ABM_SHOW_MOCK_PROVIDER", raising=False)
        keys = [key for _, key in app._provider_choices()]
        assert keys == [n for n in registry.provider_names() if app._provider_info(n) is not None]
        assert "mock" not in keys
        monkeypatch.setenv("ABM_SHOW_MOCK_PROVIDER", "1")
        assert "mock" in [key for _, key in app._provider_choices()]

    @pytest.mark.parametrize("name", registry.provider_names())
    def test_provider_change_only_offers_valid_values(self, name):
        info = app._provider_info(name)
        if info is None:
            pytest.skip(f"{name} does not import here")
        updates = dict(zip(app._PROVIDER_SLOTS, app.on_provider_change(name, "Klingon")))
        model = updates["tts_model_name"]
        assert model["value"] in model["choices"] or (model["value"] is None and not model["choices"])
        assert model["value"] == (info.default_model if info.models else None)
        language = updates["language"]
        assert language["value"] in language["choices"] or language["allow_custom_value"]
        speaker = updates["tts_timbre"]
        assert speaker["visible"] == bool(info.preset_voices and app._voice_ui(name, None).show_speaker)
        assert speaker["value"] is None or speaker["value"] in speaker["choices"]
        assert updates["tts_instruct"]["visible"] == app._voice_ui(name, None).show_instruct
        assert updates["preset_group"]["visible"] == info.supports_voice_preset
        assert updates["seed"]["visible"] == info.supports_seed
        for key in ("temperature", "top_p", "top_k", "repetition_penalty"):
            expected = info.recommended_settings.get(key, getattr(AudiobookConfig(), key))
            assert updates[key] == pytest.approx(expected), key
        markdown = updates["provider_info"]
        assert info.display_name in markdown and info.license in markdown
        assert ("Non-commercial weights" in markdown) == (not info.commercial_use)

    def test_qwen_checkpoints_choose_the_voice_source(self):
        base = app._voice_ui("qwen", _QWEN_BASE)
        assert (base.show_clip, base.show_speaker, base.show_instruct, base.show_design) == (True, False, False, False)
        custom = app._voice_ui("qwen", _QWEN_CUSTOM)
        assert (custom.show_clip, custom.show_speaker, custom.show_instruct) == (False, True, True)
        design = app._voice_ui("qwen", _QWEN_DESIGN)
        assert (design.show_clip, design.show_speaker, design.show_instruct, design.show_design) == (False, False, True, True)
        updates = dict(zip(app._MODEL_SLOTS, app.on_model_change("qwen", _QWEN_DESIGN)))
        assert updates["clip_group"]["visible"] is False and updates["design_group"]["visible"] is True

    def test_design_audition_follows_the_provider_class(self):
        for name in registry.provider_names():
            cls = app._provider_class(name)
            info = app._provider_info(name)
            if cls is None or info is None:
                continue
            has_design = callable(getattr(cls, "design_voice", None))
            shows = any(app._voice_ui(name, m).show_design for m in (info.models or (None,)))
            assert shows == has_design, name

    def test_transcript_hint_says_how_the_engine_uses_it(self):
        hints = {mode: app._transcript_hint(type("I", (), {"transcript": mode, "display_name": "X"})())
                 for mode in ("required", "optional", "unused")}
        assert hints["required"].startswith("Required")
        assert hints["optional"].startswith("Optional")
        assert "ignores" in hints["unused"]

    def test_instruction_box_shows_the_engines_own_description(self):
        info = app._provider_info("omnivoice")
        if info is None:
            pytest.skip("omnivoice does not import here")
        updates = dict(zip(app._PROVIDER_SLOTS, app.on_provider_change("omnivoice", "English")))
        assert updates["tts_instruct"]["info"] == info.description


class TestVoiceCheck:
    """One check decides whether Test Voice and Generate may run."""

    def _cfg(self, **settings) -> AudiobookConfig:
        return AudiobookConfig(**settings)

    def test_clone_checkpoint_needs_a_clip_or_a_preset(self, tmp_path):
        assert "needs a narrator voice" in app._voice_problem(self._cfg(tts_model_name=_QWEN_BASE))
        assert app._voice_problem(self._cfg(tts_model_name=_QWEN_BASE, voice_file=_VOICE)) is None
        preset = tmp_path / "v.pt"
        preset.write_bytes(b"x")
        assert app._voice_problem(self._cfg(tts_model_name=_QWEN_BASE, voice_preset=str(preset))) is None

    def test_speaker_and_design_checkpoints_do_not_need_a_clip(self):
        assert app._voice_problem(self._cfg(tts_model_name=_QWEN_CUSTOM, tts_timbre="Ryan")) is None
        assert "preset speaker" in app._voice_problem(self._cfg(tts_model_name=_QWEN_CUSTOM))
        assert app._voice_problem(self._cfg(tts_model_name=_QWEN_DESIGN, tts_instruct="deep voice")) is None
        assert "Describe" in app._voice_problem(self._cfg(tts_model_name=_QWEN_DESIGN))

    def test_engine_that_requires_a_transcript_refuses_without_one(self):
        required = [n for n in registry.provider_names()
                    if getattr(app._provider_info(n), "transcript", "") == "required"]
        if not required:
            pytest.skip("no engine requires a transcript")
        name = required[0]
        problem = app._voice_problem(self._cfg(tts_provider_name=name, tts_model_name="", voice_file=_VOICE))
        assert problem and "transcript" in problem
        ok = self._cfg(tts_provider_name=name, tts_model_name="", voice_file=_VOICE, voice_transcript="words")
        assert app._voice_problem(ok) is None

    def test_engine_that_can_design_a_voice_runs_without_a_clip(self):
        if app._provider_info("omnivoice") is None:
            pytest.skip("omnivoice does not import here")
        assert app._voice_problem(self._cfg(tts_provider_name="omnivoice", tts_model_name="")) is None

    def test_missing_clip_file_is_reported(self):
        cfg = self._cfg(tts_model_name=_QWEN_BASE, voice_file="/nonexistent/voice.wav")
        assert "no longer available" in app._voice_problem(cfg)

    def test_engine_not_installed_is_refused_with_its_install_command(self, monkeypatch):
        monkeypatch.delenv("ABM_SKIP_ENGINE_CHECK", raising=False)
        monkeypatch.setattr(app, "_missing_requirements", lambda info: ["transformers==5.0.0 (installed: 4.57.1)"])
        problem = app._voice_problem(self._cfg(tts_provider_name="moss", tts_model_name="", voice_file=_VOICE))
        assert "not installed" in problem and "pip install" in problem and "transformers==5.0.0" in problem
        assert "Not installed here" in app._provider_markdown("moss")
        monkeypatch.setattr(app, "_missing_requirements", lambda info: [])
        assert app._engine_problem("moss") is None

    def test_unknown_engine_is_refused(self):
        assert "not available" in app._voice_problem(self._cfg(tts_provider_name="vibevoice"))

    def test_test_voice_and_generate_refuse_with_the_same_message(self, tmp_path):
        ui = _non_default_ui(tmp_path, voice_file=None, voice_preset=None)
        values = tuple(ui[k] for k in app._UI_KEYS)
        _audio, message = app.on_test_voice("Hello.", *values)
        book = tmp_path / "book.txt"
        book.write_text("Chapter 1\n\nSome text here for the book.\n", encoding="utf-8")
        outputs = list(app.on_generate(None, str(book), None, "", False, None, None, {"values": []}, "", *values))
        status = dict(zip(app._GEN_SLOTS, outputs[0]))["run_status"]
        assert message == status
        assert "needs a narrator voice" in message
        assert len(outputs) == 1


def _restored(tmp_path, settings: dict, **top) -> dict:
    """Runs the restore handler on a JSON and returns ``{slot: update}``."""
    data = {"book_title": settings.get("book_title", "T"), "settings": settings, "chapters": []}
    data.update(top)
    path = tmp_path / "generation_progress.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return dict(zip(app._RESTORE_SLOTS, app.on_progress_upload_handler(str(path))))


def _value(update):
    return update.get("value") if isinstance(update, dict) and update.get("__type__") == "update" else update


def _ui_from_restore(restored: dict, files: dict) -> dict:
    """The UI values after a restore, with the files the user uploads again."""
    ui = {}
    for key in app._UI_KEYS:
        if key in files:
            ui[key] = files[key]
        elif key in restored:
            ui[key] = _value(restored[key])
    return ui


class TestRestore:

    @pytest.mark.parametrize("provider,model", [
        ("qwen", _QWEN_BASE), ("qwen", _QWEN_CUSTOM), ("qwen", _QWEN_DESIGN),
        ("indextts", None), ("moss", None), ("omnivoice", None), ("fish", None), ("higgs", None), ("f5tts", None),
    ])
    def test_exported_settings_survive_a_restore(self, tmp_path, provider, model):
        info = app._provider_info(provider)
        if info is None:
            pytest.skip(f"{provider} does not import here")
        # A non-default value for every option the engine declares.
        options = {}
        for option in info.options:
            if option.kind == "bool":
                options[option.key] = not option.default
            elif option.kind == "choice" and len(option.choices) > 1:
                options[option.key] = next(c for c in option.choices if c != option.default)
            elif option.kind in ("int", "float") and option.minimum is not None and option.maximum is not None:
                options[option.key] = option.minimum if option.default != option.minimum else option.maximum
            elif option.kind == "str":
                options[option.key] = "custom text"
        language = info.languages[-1] if info.languages else "Esperanto"
        ui = _non_default_ui(
            tmp_path, tts_provider_name=provider, tts_model_name=model or info.default_model,
            tts_options={provider: options}, language=language,
            tts_timbre=info.preset_voices[-1] if info.preset_voices else None,
        )
        exported = app._build_config(ui, book_path="/books/b.epub")
        settings = json.loads(json.dumps(dataclasses.asdict(exported)))

        restored = _restored(tmp_path, settings)
        assert "Loaded Successfully" in restored["restore_status"]
        files = {k: ui[k] for k in ("voice_file", "voice_preset", "cover_image", "pronunciation_file")}
        files["pronunciation_file"] = None            # the fixes come back in the table
        rebuilt = app._build_config(_ui_from_restore(restored, files), book_path="/books/b.epub")

        before, after = dataclasses.asdict(exported), dataclasses.asdict(rebuilt)
        # One-off instructions are never restored from a file.
        for field in sorted(set(before) - {"selected_chapters", "force_reprocess"}):
            assert after[field] == pytest.approx(before[field]), field
        assert rebuilt.force_reprocess is False
        assert exported.tts_options, "the test should exercise engine options"
        assert set(exported.tts_options) == set(options) or provider == "qwen"

    def test_file_components_are_never_given_a_server_path(self, tmp_path):
        restored = _restored(tmp_path, {"voice_file": _VOICE, "voice_preset": _VOICE, "cover_image": _VOICE},
                             book_path=_VOICE, voice_file=_VOICE)
        assert restored["book_file"] == gr.update() and restored["voice_file"] == gr.update()
        assert "voice_preset" not in app._RESTORE_SLOTS and "cover_image" not in app._RESTORE_SLOTS
        assert _VOICE not in restored["restore_status"]

    def test_junk_values_never_reach_a_dropdown(self, tmp_path):
        restored = _restored(tmp_path, {
            "tts_provider_name": "vibevoice", "tts_model_name": "someone/else", "language": "Klingon",
            "output_format": "exe", "sample_rate": "192k", "bitrate_kbps": 9999, "channels": "many",
            "quantization": "int4", "parallel_mode": "books", "verify_chunks": "always",
            "tts_timbre": "[English] ryan", "temperature": "hot", "top_k": 10**9, "lufs": -99,
            "seed": "x", "tts_options": "nope", "pronunciation_map": ["a"], "selected_chapters": "all",
            "unknown_future_field": {"a": 1}, "worker_count": 6,
        })
        assert "Loaded Successfully" in restored["restore_status"]
        assert "not available here" in restored["restore_status"]
        provider = _value(restored["tts_provider_name"])
        info = app._provider_info(provider)
        assert provider == app._default_provider()
        assert _value(restored["tts_model_name"]) in info.models
        assert _value(restored["language"]) in info.languages
        assert _value(restored["output_format"]) in app._OUTPUT_FORMATS
        assert _value(restored["sample_rate"]) in app._SAMPLE_RATES
        assert _value(restored["bitrate_kbps"]) in app._BITRATES
        assert _value(restored["channels"]) in app._CHANNELS
        assert _value(restored["quantization"]) in app._QUANTIZATIONS
        assert _value(restored["parallel_mode"]) in app._PARALLEL_MODES
        assert _value(restored["verify_chunks"]) in [v for _, v in app._VERIFY_CHOICES]
        speaker = _value(restored["tts_timbre"])
        assert speaker is None or speaker in info.preset_voices
        low, high, _ = app._RANGES["top_k"]
        assert low <= _value(restored["top_k"]) <= high
        low, high, _ = app._RANGES["lufs"]
        assert low <= _value(restored["lufs"]) <= high
        assert _value(restored["seed"]) == AudiobookConfig().seed
        assert restored["tts_options"] == {provider: {}}
        assert _value(restored["pronunciation_table"]) == [["", ""]]

    def test_old_speaker_label_is_mapped_to_the_engines_speaker(self, tmp_path):
        restored = _restored(tmp_path, {"tts_model_name": _QWEN_CUSTOM, "tts_timbre": "[English] ryan"})
        assert _value(restored["tts_timbre"]) == "Ryan"
        assert restored["tts_timbre"]["visible"] is True

    def test_legacy_nfe_step_becomes_an_engine_option(self, tmp_path):
        if app._provider_info("f5tts") is None:
            pytest.skip("f5tts does not import here")
        restored = _restored(tmp_path, {"tts_provider_name": "f5tts", "nfe_step": 48})
        assert restored["tts_options"] == {"f5tts": {"nfe_step": 48}}

    def test_chapters_in_the_json_become_valid_checklist_choices(self, tmp_path):
        chapters = [
            {"num": 1, "title": "One", "status": "completed", "text": "a b c"},
            {"num": 2, "title": "Two", "status": "pending", "text": "d e"},
        ]
        restored = _restored(tmp_path, {"selected_chapters": ["2. Two  (~999 words)"]}, chapters=chapters)
        check = restored["selected_chapters"]
        values = [value for _, value in check["choices"]]
        assert values == ["1. One  (~3 words)", "2. Two  (~2 words)"]
        assert check["value"] == ["2. Two  (~2 words)"]          # matched by title, and one of the choices
        assert restored["all_choices"]["values"] == values
        assert "Completed:** 1" in restored["restore_status"]

    def test_unreadable_files_are_reported_not_raised(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text("<html>not json</html>" * 5, encoding="utf-8")
        result = app.on_progress_upload_handler(str(bad))
        assert len(result) == len(app._RESTORE_SLOTS)
        assert "Failed to parse" in result[0]
        assert len(app.on_progress_upload_handler(None)) == len(app._RESTORE_SLOTS)
        listy = tmp_path / "list.json"
        listy.write_text(json.dumps([1, 2, 3] * 10), encoding="utf-8")
        assert "Failed to parse" in app.on_progress_upload_handler(str(listy))[0]
