"""
test_omnivoice_provider.py
===========================
Unit tests for the OmniVoice TTS provider.

No network, no GPU and no ``omnivoice`` package: a fake ``omnivoice`` module is
injected into ``sys.modules``. Its classes mirror the signatures of
``omnivoice`` 0.2.1 (``omnivoice/models/omnivoice.py``): ``OmniVoice.from_pretrained``,
``generate``, ``create_voice_clone_prompt``, ``load_asr_model`` and
``VoiceClonePrompt.save`` / ``load``.
"""
from __future__ import annotations

import io
import json
import os
import sys
import tempfile
import types
from dataclasses import dataclass
from typing import Any

import numpy as np
import pytest
import soundfile as sf
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from audiobook_factory.pipeline import AudiobookConfig
from audiobook_factory.tts_providers import omnivoice_provider as ov
from audiobook_factory.tts_providers.base_tts_provider import get_tts_provider
from audiobook_factory.tts_providers.omnivoice_provider import OmniVoiceProvider

_SAMPLE_RATE = 24000
_INSTALL_COMMAND = 'pip install "omnivoice>=0.2.1"'


# ── Fake upstream package ─────────────────────────────────────────────────────


@dataclass
class FakeVoiceClonePrompt:
    """Mirror of ``omnivoice.VoiceClonePrompt``."""

    ref_audio_tokens: torch.Tensor
    ref_text: str
    ref_rms: float

    def save(self, path: str) -> None:
        torch.save(
            {
                "format_version": 1,
                "ref_audio_tokens": self.ref_audio_tokens.detach().cpu(),
                "ref_text": self.ref_text,
                "ref_rms": float(self.ref_rms),
            },
            path,
        )

    @classmethod
    def load(cls, path: str, map_location: str = "cpu") -> "FakeVoiceClonePrompt":
        data = torch.load(path, map_location=map_location, weights_only=True)
        if data.get("format_version") != 1:
            raise ValueError("Unsupported VoiceClonePrompt format version")
        return cls(data["ref_audio_tokens"], data["ref_text"], data["ref_rms"])


def _waveform(text: str) -> np.ndarray:
    """A tone whose length encodes the text length, so order can be checked."""
    samples = _SAMPLE_RATE + 240 * len(text)
    t = np.arange(samples, dtype=np.float32) / _SAMPLE_RATE
    return (0.2 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)


class FakeOmniVoice:
    """Mirror of ``omnivoice.OmniVoice`` recording every call."""

    instances: list["FakeOmniVoice"] = []
    # Class-wide behaviour switches, reset by the fixture.
    audio_override: Any = None
    oom_above: int = 0
    generate_error: Exception | None = None
    transcript: str = "auto transcript"
    next_token: int = 0

    def __init__(self, name: str, load_kwargs: dict) -> None:
        self.name = name
        self.load_kwargs = load_kwargs
        self.sampling_rate = _SAMPLE_RATE
        # OmniVoiceConfig defaults (models/omnivoice.py:247-267).
        self.config = types.SimpleNamespace(num_audio_codebook=8, audio_vocab_size=1025, audio_mask_id=1024)
        self._asr_pipe = None
        self._asr_model_name = "openai/whisper-large-v3-turbo"
        self.generate_calls: list[dict] = []
        self.prompt_calls: list[dict] = []
        self.asr_loads: list[tuple] = []
        self.transcriptions = 0

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        model = cls(pretrained_model_name_or_path, dict(kwargs))
        cls.instances.append(model)
        return model

    def load_asr_model(self, model_name=None, device=None):
        self.asr_loads.append((model_name, device))
        self._asr_pipe = object()

    def transcribe(self, audio) -> str:
        if self._asr_pipe is None:
            raise RuntimeError("ASR model is not loaded. Call model.load_asr_model() first.")
        self.transcriptions += 1
        return FakeOmniVoice.transcript

    def create_voice_clone_prompt(self, ref_audio, ref_text=None, preprocess_prompt=True):
        self.prompt_calls.append(
            {"ref_audio": ref_audio, "ref_text": ref_text, "preprocess_prompt": preprocess_prompt}
        )
        if isinstance(ref_audio, str):
            sf.read(ref_audio)  # upstream loads the file; a missing one must fail
        if ref_text is None:
            if self._asr_pipe is None:
                self.load_asr_model()
            ref_text = self.transcribe(ref_audio)
        FakeOmniVoice.next_token += 1
        tokens = torch.full((8, 50), FakeOmniVoice.next_token, dtype=torch.long)
        return FakeVoiceClonePrompt(tokens, ref_text, 0.1)

    def generate(
        self,
        text,
        language=None,
        ref_text=None,
        ref_audio=None,
        voice_clone_prompt=None,
        instruct=None,
        duration=None,
        speed=None,
        generation_config=None,
        normalize_text=False,
        **kwargs,
    ):
        texts = [text] if isinstance(text, str) else list(text)
        self.generate_calls.append(
            {
                "text": texts, "language": language, "ref_text": ref_text, "ref_audio": ref_audio,
                "voice_clone_prompt": voice_clone_prompt, "instruct": instruct, "duration": duration,
                "speed": speed, "generation_config": generation_config,
                "normalize_text": normalize_text, "kwargs": dict(kwargs),
            }
        )
        if FakeOmniVoice.generate_error is not None:
            raise FakeOmniVoice.generate_error
        if FakeOmniVoice.oom_above and len(texts) > FakeOmniVoice.oom_above:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory (fake)")
        if FakeOmniVoice.audio_override is not None:
            return [FakeOmniVoice.audio_override for _ in texts]
        return [_waveform(t) for t in texts]


@pytest.fixture
def fake_omnivoice(monkeypatch, tmp_path):
    """Installs the fake ``omnivoice`` package and isolates temp files."""
    FakeOmniVoice.instances = []
    FakeOmniVoice.audio_override = None
    FakeOmniVoice.oom_above = 0
    FakeOmniVoice.generate_error = None
    FakeOmniVoice.transcript = "auto transcript"
    FakeOmniVoice.next_token = 0

    module = types.ModuleType("omnivoice")
    module.OmniVoice = FakeOmniVoice
    module.VoiceClonePrompt = FakeVoiceClonePrompt
    monkeypatch.setitem(sys.modules, "omnivoice", module)

    temp_dir = tmp_path / "tmp"
    temp_dir.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(temp_dir))
    return FakeOmniVoice


@pytest.fixture
def config(tmp_path) -> AudiobookConfig:
    return AudiobookConfig(output_dir=str(tmp_path / "out"), language="English")


@pytest.fixture
def voice_bytes() -> bytes:
    t = np.arange(_SAMPLE_RATE * 2, dtype=np.float32) / _SAMPLE_RATE
    buf = io.BytesIO()
    sf.write(buf, (0.3 * np.sin(2 * np.pi * 150.0 * t)).astype(np.float32), _SAMPLE_RATE, format="WAV")
    return buf.getvalue()


def _provider(config: AudiobookConfig, device: str = "cpu", **kwargs) -> OmniVoiceProvider:
    return OmniVoiceProvider(config, device=device, **kwargs)


def _model(index: int = -1) -> FakeOmniVoice:
    return FakeOmniVoice.instances[index]


def _design_calls(model: FakeOmniVoice) -> list[dict]:
    return [c for c in model.generate_calls if c["voice_clone_prompt"] is None]


# ── Capability description ────────────────────────────────────────────────────


def test_info_is_complete_and_flags_non_commercial_weights():
    info = OmniVoiceProvider.info()
    assert info.name == "omnivoice"
    assert info.display_name == "OmniVoice"
    assert info.commercial_use is False
    assert "NC" in info.license
    assert info.homepage == "https://github.com/k2-fsa/OmniVoice"
    assert info.default_model == "k2-fsa/OmniVoice"
    assert info.default_model in info.models
    assert info.native_sample_rate == 24000
    assert 0 < info.min_vram_gb <= 8
    assert "English" in info.languages and "Chinese" in info.languages
    assert info.transcript == "optional"
    assert info.supports_voice_clone and info.supports_instruct
    assert info.supports_batch and info.supports_speed and info.supports_seed
    assert info.preset_voices == ()
    assert info.supports_voice_preset is True
    # No shared sampling field (temperature, top_p, ...) is mapped, so there is
    # no upstream operating point to recommend for them.
    assert info.recommended_settings == {}
    assert info.pip_requirements == ("omnivoice>=0.2.1",)
    assert "transformers>=5.3.0" in info.install_notes

    keys = [option.key for option in info.options]
    assert len(keys) == len(set(keys))
    for key in ("num_step", "guidance_scale", "denoise", "postprocess_output", "language_id", "duration"):
        assert key in keys
    for option in info.options:
        assert option.kind in ("float", "int", "bool", "str", "choice")
        assert option.label and option.help
        assert option.default is not None
        if option.kind == "choice":
            assert option.default in option.choices


def test_generation_defaults_match_upstream_generation_config():
    # omnivoice 0.2.1, OmniVoiceGenerationConfig (models/omnivoice.py:176-189).
    upstream = {
        "num_step": 32, "guidance_scale": 2.0, "t_shift": 0.1, "layer_penalty_factor": 5.0,
        "position_temperature": 5.0, "class_temperature": 0.0, "denoise": True,
        "preprocess_prompt": True, "postprocess_output": True, "audio_chunk_duration": 15.0,
        "audio_chunk_threshold": 30.0, "pad_duration": 0.1, "fade_duration": 0.1,
    }
    defaults = OmniVoiceProvider.info().option_defaults()
    assert set(ov._GENERATION_OPTION_KEYS) == set(upstream)
    for key, value in upstream.items():
        assert defaults[key] == value


def test_registry_builds_the_provider_without_loading_a_model(config):
    provider = get_tts_provider("omni-voice", config, device="cuda:1")
    assert isinstance(provider, OmniVoiceProvider)
    assert provider.device == "cuda:1"
    assert provider.is_ready is False
    assert provider.get_name() == "OmniVoice"


# ── Loading ───────────────────────────────────────────────────────────────────


def test_missing_package_error_names_the_install_command(monkeypatch, config):
    monkeypatch.setitem(sys.modules, "omnivoice", None)  # makes `import omnivoice` fail
    provider = _provider(config)
    with pytest.raises(RuntimeError) as excinfo:
        provider.ensure_ready()
    assert _INSTALL_COMMAND in str(excinfo.value)
    assert "not installed" in str(excinfo.value)
    assert provider.is_ready is False


def test_outdated_package_error_names_the_install_command(monkeypatch, config):
    module = types.ModuleType("omnivoice")
    module.OmniVoice = FakeOmniVoice
    module.VoiceClonePrompt = type("VoiceClonePrompt", (), {})  # no save()/load()
    monkeypatch.setitem(sys.modules, "omnivoice", module)
    with pytest.raises(RuntimeError) as excinfo:
        _provider(config).ensure_ready()
    assert "too old" in str(excinfo.value) and _INSTALL_COMMAND in str(excinfo.value)


def test_broken_import_error_mentions_transformers_and_install_command(monkeypatch, config):
    # What `import omnivoice` does next to transformers 4.x: the package is
    # there but a name it needs is not.
    monkeypatch.setitem(sys.modules, "omnivoice", types.ModuleType("omnivoice"))
    with pytest.raises(RuntimeError) as excinfo:
        _provider(config).ensure_ready()
    assert "transformers>=5.3.0" in str(excinfo.value)
    assert _INSTALL_COMMAND in str(excinfo.value)


def test_module_imports_without_torch_or_omnivoice():
    import subprocess

    code = (
        "import sys; sys.modules['omnivoice'] = None; "
        "import audiobook_factory.tts_providers.omnivoice_provider as m; "
        "assert m.OmniVoiceProvider.info().name == 'omnivoice'; "
        "assert 'torch' not in sys.modules, 'torch imported at module level'"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=_ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_foreign_model_name_falls_back_to_default(fake_omnivoice, config):
    config.tts_model_name = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
    provider = _provider(config)
    provider.ensure_ready()
    assert provider.is_ready
    assert _model().name == "k2-fsa/OmniVoice"
    assert _model().load_kwargs == {"device_map": "cpu", "dtype": torch.float32}


def test_device_and_dtype_are_passed_to_from_pretrained(fake_omnivoice, config):
    _provider(config, device="cuda:1").ensure_ready()
    assert _model().load_kwargs == {"device_map": "cuda:1", "dtype": torch.float16}

    _provider(config, device="cuda:0", dtype_override="float32").ensure_ready()
    assert _model().load_kwargs == {"device_map": "cuda:0", "dtype": torch.float32}


def test_model_is_reloaded_when_the_configured_model_changes(fake_omnivoice, config, voice_bytes):
    config.voice_transcript = "Reference words."
    provider = _provider(config)
    provider.synthesize("First run.", voice_bytes, return_bytes=True)
    provider.ensure_ready()
    assert len(FakeOmniVoice.instances) == 1

    # The pool hands the instance a new config between runs.
    provider.config = AudiobookConfig(
        output_dir=config.output_dir, tts_model_name="k2-fsa/OmniVoice-Emilia",
        voice_transcript="Reference words.",
    )
    provider.synthesize("Second run.", voice_bytes, return_bytes=True)
    assert len(FakeOmniVoice.instances) == 2
    assert _model().name == "k2-fsa/OmniVoice-Emilia"
    # The Emilia checkpoint was trained without denoising or language ids.
    call = _model().generate_calls[-1]
    assert call["language"] is None
    assert call["kwargs"]["denoise"] is False
    # The reference is encoded again for the new model.
    assert len(_model().prompt_calls) == 1


def test_cleanup_drops_the_model(fake_omnivoice, config, voice_bytes):
    config.voice_transcript = "Reference words."
    provider = _provider(config)
    provider.synthesize("Hello.", voice_bytes, return_bytes=True)
    provider.cleanup()
    assert provider.is_ready is False
    assert provider._prompt_cache == {}
    provider.synthesize("Hello again.", voice_bytes, return_bytes=True)
    assert len(FakeOmniVoice.instances) == 2


# ── Arguments passed upstream ─────────────────────────────────────────────────


def test_generate_kwargs_reflect_config_and_options(fake_omnivoice, config, voice_bytes):
    config.language = "German"
    config.voice_transcript = "Reference words."
    config.tts_instruct = "female, low pitch"
    config.speed = 1.2
    config.seed = 7
    config.tts_options = {
        "num_step": "16",           # strings arrive from forms and JSON
        "guidance_scale": 3.0,
        "t_shift": 0.2,
        "denoise": "false",
        "class_temperature": 0.1,
        "position_temperature": 4.0,
        "layer_penalty_factor": 2.0,
        "postprocess_output": False,
        "pad_duration": 0.0,
        "fade_duration": 0.05,
        "audio_chunk_duration": 12.0,
        "audio_chunk_threshold": 24.0,
        "normalize_text": True,
    }
    provider = _provider(config)
    provider.synthesize_batch(["Eins.", "Zwei."], voice_bytes)

    call = _model().generate_calls[-1]
    assert call["text"] == ["Eins.", "Zwei."]
    assert call["language"] == "German"
    assert call["instruct"] == "female, low pitch"
    assert call["speed"] == pytest.approx(1.2)
    assert call["duration"] is None
    assert call["normalize_text"] is True
    assert call["ref_audio"] is None and call["ref_text"] is None
    assert isinstance(call["voice_clone_prompt"], FakeVoiceClonePrompt)
    assert call["kwargs"] == {
        "num_step": 16, "guidance_scale": 3.0, "t_shift": 0.2, "denoise": False,
        "class_temperature": 0.1, "position_temperature": 4.0, "layer_penalty_factor": 2.0,
        "preprocess_prompt": True, "postprocess_output": False, "pad_duration": 0.0,
        "fade_duration": 0.05, "audio_chunk_duration": 12.0, "audio_chunk_threshold": 24.0,
    }
    assert isinstance(call["kwargs"]["num_step"], int)
    assert torch.initial_seed() == 7


def test_defaults_leave_speed_unset_and_use_upstream_values(fake_omnivoice, config, voice_bytes):
    config.voice_transcript = "Reference words."
    provider = _provider(config)
    provider.synthesize("Hello.", voice_bytes, return_bytes=True)
    call = _model().generate_calls[-1]
    assert call["language"] == "English"
    assert call["instruct"] is None
    assert call["speed"] is None and call["duration"] is None
    assert call["normalize_text"] is False
    assert call["kwargs"]["num_step"] == 32
    assert call["kwargs"]["guidance_scale"] == 2.0
    assert call["kwargs"]["denoise"] is True


def test_language_aliases_code_override_and_duration(fake_omnivoice, config, voice_bytes):
    config.voice_transcript = "Reference words."
    provider = _provider(config)

    config.language = "Arabic"  # upstream only knows "Standard Arabic" / "arb"
    provider.synthesize("One.", voice_bytes, return_bytes=True)
    assert _model().generate_calls[-1]["language"] == "arb"

    config.language = "auto"
    provider.synthesize("Two.", voice_bytes, return_bytes=True)
    assert _model().generate_calls[-1]["language"] is None

    config.language = "zh-CN"
    provider.synthesize("二。", voice_bytes, return_bytes=True)
    assert _model().generate_calls[-1]["language"] == "zh"

    config.tts_options = {"language_id": "YUE", "duration": 6.5, "num_step": 500}
    provider.synthesize("Three.", voice_bytes, return_bytes=True)
    call = _model().generate_calls[-1]
    assert call["language"] == "yue"
    assert call["duration"] == pytest.approx(6.5)
    assert call["kwargs"]["num_step"] == 64  # clamped to the declared range


# ── Reference clip ────────────────────────────────────────────────────────────


def test_reference_is_prepared_once_and_transcript_forwarded(fake_omnivoice, config, voice_bytes):
    config.voice_transcript = "  The reference transcript.  "
    provider = _provider(config)
    for text in ("Chunk one.", "Chunk two.", "Chunk three."):
        provider.synthesize(text, voice_bytes, return_bytes=True)
    provider.synthesize_batch(["Chunk four.", "Chunk five."], voice_bytes)

    model = _model()
    assert len(model.prompt_calls) == 1
    prepared = model.prompt_calls[0]
    assert prepared["ref_text"] == "The reference transcript."
    assert prepared["preprocess_prompt"] is True
    assert isinstance(prepared["ref_audio"], str) and os.path.isfile(prepared["ref_audio"])
    with open(prepared["ref_audio"], "rb") as fh:
        assert fh.read() == voice_bytes

    # No transcription model was loaded or run.
    assert model.asr_loads == [] and model.transcriptions == 0
    prompts = {id(call["voice_clone_prompt"]) for call in model.generate_calls}
    assert len(model.generate_calls) == 4 and len(prompts) == 1


def test_reference_cache_is_keyed_on_content_and_transcript(fake_omnivoice, config, voice_bytes, tmp_path):
    config.voice_transcript = "First transcript."
    provider = _provider(config)
    provider.synthesize("One.", voice_bytes, return_bytes=True)

    # Same content from a file path: still the same clip.
    clip = tmp_path / "narrator.wav"
    clip.write_bytes(voice_bytes)
    provider.synthesize("Two.", str(clip), return_bytes=True)
    assert len(_model().prompt_calls) == 1

    config.voice_transcript = "Second transcript."
    provider.synthesize("Three.", voice_bytes, return_bytes=True)
    assert len(_model().prompt_calls) == 2

    provider.synthesize("Four.", voice_bytes + b"\x00\x00", return_bytes=True)
    assert len(_model().prompt_calls) == 3


def test_sidecar_transcript_of_voice_file_is_used_for_bytes(fake_omnivoice, config, voice_bytes, tmp_path):
    clip = tmp_path / "narrator.wav"
    clip.write_bytes(voice_bytes)
    (tmp_path / "narrator.txt").write_text("Sidecar transcript.", encoding="utf-8")
    config.voice_file = str(clip)
    provider = _provider(config)

    provider.synthesize("One.", voice_bytes, return_bytes=True)  # what the pipeline passes
    assert _model().prompt_calls[0]["ref_text"] == "Sidecar transcript."
    assert _model().asr_loads == []


def test_empty_voice_ref_uses_config_voice_file(fake_omnivoice, config, voice_bytes, tmp_path):
    clip = tmp_path / "narrator.wav"
    clip.write_bytes(voice_bytes)
    config.voice_file = str(clip)
    config.voice_transcript = "Reference words."
    provider = _provider(config)
    provider.synthesize_batch(["One."], b"")
    assert _model().prompt_calls[0]["ref_audio"] == str(clip)
    assert _design_calls(_model()) == []


def test_missing_voice_file_raises_instead_of_designing_a_voice(fake_omnivoice, config):
    config.voice_file = "/nonexistent/narrator.wav"
    provider = _provider(config)
    with pytest.raises(RuntimeError, match="not found"):
        provider.synthesize_batch(["One."], b"")
    assert _model().generate_calls == []


def test_auto_transcription_runs_once_and_whisper_is_released(fake_omnivoice, config, voice_bytes):
    config.tts_options = {"asr_model": "openai/whisper-small", "asr_device": "cpu"}
    first = _provider(config, device="cuda:0")
    first.synthesize_batch(["One.", "Two."], voice_bytes)
    first.synthesize("Three.", voice_bytes, return_bytes=True)

    model = _model()
    assert model.asr_loads == [("openai/whisper-small", "cpu")]
    assert model.transcriptions == 1
    assert len(model.prompt_calls) == 1 and model.prompt_calls[0]["ref_text"] is None
    assert model._asr_pipe is None  # not left resident for the rest of the book
    assert model.generate_calls[-1]["voice_clone_prompt"].ref_text == "auto transcript"

    # A second GPU instance reuses the saved prompt: no Whisper, no re-encoding.
    second = _provider(config, device="cuda:1")
    second.synthesize("Four.", voice_bytes, return_bytes=True)
    other = _model()
    assert other is not model
    assert other.asr_loads == [] and other.transcriptions == 0 and other.prompt_calls == []
    shared = other.generate_calls[-1]["voice_clone_prompt"]
    assert shared.ref_text == "auto transcript"
    assert torch.equal(shared.ref_audio_tokens, model.generate_calls[-1]["voice_clone_prompt"].ref_audio_tokens)


def test_hallucinated_auto_transcript_is_rejected_and_not_saved(fake_omnivoice, config, voice_bytes):
    # What whisper-tiny really returned for a 4 s clip during development.
    FakeOmniVoice.transcript = "Oh, " * 250
    provider = _provider(config)
    with pytest.raises(RuntimeError, match="could not transcribe the reference clip reliably"):
        provider.synthesize_batch(["One."], voice_bytes)
    assert _model()._asr_pipe is None
    assert _model().generate_calls == []
    cache_dir = os.path.join(config.output_dir, ".omnivoice_voices")
    assert not [name for name in os.listdir(cache_dir) if name.endswith(".pt")]

    FakeOmniVoice.transcript = ""
    with pytest.raises(RuntimeError, match="no speech recognised"):
        provider.synthesize_batch(["One."], voice_bytes)


def test_mismatched_user_transcript_warns_but_still_synthesizes(fake_omnivoice, config, voice_bytes, caplog):
    config.voice_transcript = "word " * 200  # far too much text for the 2 s fake prompt
    provider = _provider(config)
    with caplog.at_level("WARNING"):
        results = provider.synthesize_batch(["One.", "Two."], voice_bytes)
        provider.synthesize_batch(["Three."], voice_bytes)
    assert len(results) == 2
    warnings = [r for r in caplog.records if "does not seem to match" in r.getMessage()]
    assert len(warnings) == 1


def test_default_transcription_runs_on_the_provider_device(fake_omnivoice, config, voice_bytes):
    provider = _provider(config, device="cuda:1")
    provider.synthesize("One.", voice_bytes, return_bytes=True)
    assert _model().asr_loads == [("openai/whisper-large-v3-turbo", "cuda:1")]


def test_voice_preset_prompt_replaces_the_reference_clip(fake_omnivoice, config, voice_bytes, tmp_path):
    preset = tmp_path / "narrator.pt"
    FakeVoiceClonePrompt(torch.full((8, 30), 9, dtype=torch.long), "Preset words.", 0.2).save(str(preset))
    config.voice_preset = str(preset)
    provider = _provider(config)
    provider.synthesize_batch(["One.", "Two."], voice_bytes)
    provider.synthesize("Three.", voice_bytes, return_bytes=True)

    model = _model()
    assert model.prompt_calls == [] and model.asr_loads == []
    used = model.generate_calls[-1]["voice_clone_prompt"]
    assert used.ref_text == "Preset words."
    assert int(used.ref_audio_tokens[0, 0]) == 9
    assert model.generate_calls[0]["voice_clone_prompt"] is used


# ── Batching ──────────────────────────────────────────────────────────────────


def test_batch_is_one_upstream_call_with_results_in_order(fake_omnivoice, config, voice_bytes):
    config.voice_transcript = "Reference words."
    provider = _provider(config)
    texts = ["Short.", "A somewhat longer sentence for the second chunk.", "Mid length chunk."]
    results = provider.synthesize_batch(texts, voice_bytes)

    assert len(_model().generate_calls) == 1
    assert _model().generate_calls[0]["text"] == texts
    assert len(results) == len(texts)
    for text, (wav, duration) in zip(texts, results):
        assert isinstance(wav, bytes)
        audio, sample_rate = sf.read(io.BytesIO(wav), dtype="float32")
        assert sample_rate == _SAMPLE_RATE
        assert len(audio) == len(_waveform(text))
        assert duration == pytest.approx(len(audio) / _SAMPLE_RATE)
    assert provider.synthesize_batch([], voice_bytes) == []


def test_synthesize_writes_out_path(fake_omnivoice, config, voice_bytes, tmp_path):
    config.voice_transcript = "Reference words."
    out_path = str(tmp_path / "nested" / "chunk.wav")
    result, duration = _provider(config).synthesize("Hello there.", voice_bytes, out_path)
    assert result == out_path
    assert sf.info(out_path).samplerate == _SAMPLE_RATE
    assert duration == pytest.approx(len(_waveform("Hello there.")) / _SAMPLE_RATE)


def test_out_of_memory_falls_back_to_one_chunk_at_a_time(fake_omnivoice, config, voice_bytes):
    config.voice_transcript = "Reference words."
    FakeOmniVoice.oom_above = 1
    provider = _provider(config)
    texts = ["Alpha.", "Bravo bravo.", "Charlie charlie charlie.", "Delta."]
    results = provider.synthesize_batch(texts, voice_bytes)

    sizes = [len(call["text"]) for call in _model().generate_calls]
    assert sizes == [4, 1, 1, 1, 1]
    assert [round(d * _SAMPLE_RATE) for _, d in results] == [len(_waveform(t)) for t in texts]

    # Later batches are capped instead of running out of memory again.
    FakeOmniVoice.oom_above = 2
    provider.synthesize_batch(texts, voice_bytes)
    assert [len(call["text"]) for call in _model().generate_calls[5:]] == [2, 2]


def test_out_of_memory_on_a_single_chunk_raises(fake_omnivoice, config, voice_bytes):
    config.voice_transcript = "Reference words."
    FakeOmniVoice.generate_error = torch.cuda.OutOfMemoryError("CUDA out of memory (fake)")
    with pytest.raises(RuntimeError, match="out of GPU memory"):
        _provider(config).synthesize_batch(["One.", "Two."], voice_bytes)


# ── Failures must raise ───────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "bad_audio, message",
    [
        (np.zeros(0, dtype=np.float32), "produced no audio"),
        (np.full(2400, np.nan, dtype=np.float32), "NaN/inf"),
    ],
)
def test_empty_or_nan_audio_raises(fake_omnivoice, config, voice_bytes, bad_audio, message):
    config.voice_transcript = "Reference words."
    FakeOmniVoice.audio_override = bad_audio
    provider = _provider(config)
    with pytest.raises(RuntimeError, match=message):
        provider.synthesize("Hello.", voice_bytes, return_bytes=True)
    with pytest.raises(RuntimeError, match=message):
        provider.synthesize_batch(["Hello.", "World."], voice_bytes)


def test_empty_text_raises_before_any_generation(fake_omnivoice, config, voice_bytes):
    provider = _provider(config)
    with pytest.raises(RuntimeError, match="empty text"):
        provider.synthesize_batch(["Fine.", "   "], voice_bytes)
    assert FakeOmniVoice.instances == []


def test_upstream_errors_are_raised_not_swallowed(fake_omnivoice, config, voice_bytes):
    config.voice_transcript = "Reference words."
    FakeOmniVoice.generate_error = KeyError("boom")
    with pytest.raises(RuntimeError, match="synthesis failed"):
        _provider(config).synthesize_batch(["One.", "Two."], voice_bytes)
    assert len(_model().generate_calls) == 1  # no silent per-item retry for non-OOM errors


def test_rejected_style_prompt_explains_the_attribute_format(fake_omnivoice, config, voice_bytes):
    config.voice_transcript = "Reference words."
    config.tts_instruct = "a warm, calm storyteller"
    FakeOmniVoice.generate_error = ValueError("Unsupported instruct items found")
    with pytest.raises(RuntimeError) as excinfo:
        _provider(config).synthesize("Hello.", voice_bytes, return_bytes=True)
    assert "a warm, calm storyteller" in str(excinfo.value)
    assert "british accent" in str(excinfo.value)


def test_wrong_number_of_waveforms_raises(fake_omnivoice, config, voice_bytes, monkeypatch):
    config.voice_transcript = "Reference words."
    monkeypatch.setattr(FakeOmniVoice, "generate", lambda self, **kwargs: [_waveform("x")])
    with pytest.raises(RuntimeError, match="1 waveform"):
        _provider(config).synthesize_batch(["One.", "Two."], voice_bytes)


def test_truncated_voice_bytes_raise(fake_omnivoice, config):
    with pytest.raises(RuntimeError, match="empty or corrupted"):
        _provider(config).synthesize("Hello.", b"RIFF", return_bytes=True)


# ── Voice design ──────────────────────────────────────────────────────────────


def test_designed_voice_is_created_once_and_shared_between_instances(fake_omnivoice, config):
    config.tts_instruct = "Female, Low Pitch"
    config.seed = 11
    first = _provider(config, device="cuda:0")
    first.synthesize_batch(["Chunk one.", "Chunk two."], b"")
    first.synthesize("Chunk three.", b"", return_bytes=True)

    model = _model()
    design = _design_calls(model)
    assert len(design) == 1
    assert design[0]["instruct"] == "Female, Low Pitch"
    assert design[0]["language"] == "English"
    assert design[0]["text"] == [ov._DESIGN_TEXTS["en"]]
    assert design[0]["speed"] is None
    # The designed clip is encoded once, with the sentence that was spoken.
    assert len(model.prompt_calls) == 1
    assert model.prompt_calls[0]["ref_text"] == ov._DESIGN_TEXTS["en"]
    assert os.path.isfile(model.prompt_calls[0]["ref_audio"])
    narration = [c for c in model.generate_calls if c["voice_clone_prompt"] is not None]
    assert len(narration) == 2
    assert len({id(c["voice_clone_prompt"]) for c in narration}) == 1
    assert narration[0]["instruct"] == "Female, Low Pitch"

    cache_dir = os.path.join(config.output_dir, ".omnivoice_voices")
    saved = sorted(name for name in os.listdir(cache_dir) if not name.endswith(".lock"))
    assert [os.path.splitext(name)[1] for name in saved] == [".json", ".pt", ".wav"]
    with open(os.path.join(cache_dir, saved[0]), encoding="utf-8") as fh:
        described = json.load(fh)
    assert described["seed"] == 11  # the run seed drives the first design
    assert described["instruct"] == "Female, Low Pitch" and described["model"] == "k2-fsa/OmniVoice"

    # The second GPU instance narrates with the same voice; an equivalent
    # spelling of the style prompt maps to the same saved voice.
    config.tts_instruct = "female,low pitch"
    second = _provider(config, device="cuda:1")
    second.synthesize_batch(["Chunk four."], b"")
    other = _model()
    assert other is not model
    assert _design_calls(other) == [] and other.prompt_calls == []
    assert torch.equal(
        other.generate_calls[0]["voice_clone_prompt"].ref_audio_tokens,
        narration[0]["voice_clone_prompt"].ref_audio_tokens,
    )
    assert other.generate_calls[0]["voice_clone_prompt"].ref_text == ov._DESIGN_TEXTS["en"]


def test_designed_voice_key_covers_instruct_language_voice_seed_and_model(fake_omnivoice, config):
    config.tts_instruct = "male"
    config.tts_options = {"voice_seed": 1}
    provider = _provider(config)
    provider.synthesize("One.", b"", return_bytes=True)
    assert len(_design_calls(_model())) == 1

    provider.synthesize("Two.", b"", return_bytes=True)
    assert len(_design_calls(_model())) == 1

    config.tts_options = {"voice_seed": 2}
    provider.synthesize("Three.", b"", return_bytes=True)
    assert len(_design_calls(_model())) == 2
    assert torch.initial_seed() == 2  # the voice seed drives the design

    config.tts_instruct = "male, elderly"
    provider.synthesize("Four.", b"", return_bytes=True)
    assert len(_design_calls(_model())) == 3

    config.language = "Chinese"
    provider.synthesize("五。", b"", return_bytes=True)
    design = _design_calls(_model())
    assert len(design) == 4
    assert design[-1]["text"] == [ov._DESIGN_TEXTS["zh"]]

    config.tts_model_name = "k2-fsa/OmniVoice-Emilia"
    provider.synthesize("六。", b"", return_bytes=True)
    assert len(_design_calls(_model())) == 1  # a new model instance designed its own


def test_pipeline_retry_with_bumped_seed_keeps_the_designed_narrator(fake_omnivoice, config):
    """chapter_pipeline._verify_and_repair re-takes a chunk with seed + attempt."""
    import dataclasses

    config.tts_instruct = "female"
    config.seed = 40
    provider = _provider(config)
    provider.synthesize_batch(["Chunk one.", "Chunk two."], b"")
    model = _model()
    narrator = model.generate_calls[-1]["voice_clone_prompt"]
    assert torch.initial_seed() == 40

    provider.config = dataclasses.replace(config, seed=41)
    provider.synthesize("Chunk two.", b"", return_bytes=True)
    provider.config = config

    assert len(_design_calls(model)) == 1 and len(model.prompt_calls) == 1
    assert model.generate_calls[-1]["voice_clone_prompt"] is narrator
    assert torch.initial_seed() == 41  # but the re-take itself is a different draw

    # A fresh instance (second GPU, or a resumed run) agrees.
    other_provider = _provider(dataclasses.replace(config, seed=42), device="cuda:1")
    other_provider.synthesize_batch(["Chunk three."], b"")
    assert _design_calls(_model()) == []
    assert torch.equal(
        _model().generate_calls[-1]["voice_clone_prompt"].ref_audio_tokens, narrator.ref_audio_tokens,
    )


def test_auto_voice_without_instruct_is_still_fixed_once(fake_omnivoice, config):
    provider = _provider(config)
    provider.synthesize_batch(["One.", "Two."], b"")
    provider.synthesize_batch(["Three."], b"")
    design = _design_calls(_model())
    assert len(design) == 1 and design[0]["instruct"] is None
    assert all(c["voice_clone_prompt"] is not None for c in _model().generate_calls[1:])


def test_design_sentence_comes_from_the_book_for_other_languages(fake_omnivoice, config):
    config.language = "German"
    opening = "Es war einmal ein alter Leuchtturmwärter. Er stieg jeden Abend die Treppe hinauf."
    filler = " Und so ging es viele Jahre lang weiter, bei jedem Wetter und zu jeder Jahreszeit."
    provider = _provider(config)
    provider.synthesize_batch(["Kapitel eins.", opening + filler], b"")
    spoken = _design_calls(_model())[0]["text"][0]
    assert spoken.startswith("Kapitel eins. Es war einmal")
    assert spoken.endswith("hinauf.")
    assert ov._text_weight(spoken) <= ov._DESIGN_TEXT_MAX_WEIGHT

    config.tts_options = {"design_text": "Guten Abend, liebe Hörerinnen und Hörer."}
    provider.synthesize("Noch ein Satz.", b"", return_bytes=True)
    assert _design_calls(_model())[-1]["text"] == ["Guten Abend, liebe Hörerinnen und Hörer."]


def test_very_short_opening_is_repeated_to_a_usable_design_sentence(fake_omnivoice, config):
    config.language = "German"
    _provider(config).synthesize("Kapitel eins", b"", return_bytes=True)
    spoken = _design_calls(_model())[0]["text"][0]
    assert spoken.startswith("Kapitel eins. Kapitel eins.")
    assert ov._DESIGN_TEXT_MIN_WEIGHT <= ov._text_weight(spoken) <= ov._DESIGN_TEXT_MAX_WEIGHT


def test_leading_excerpt_cuts_an_overlong_first_sentence_at_a_word():
    long_sentence = "word " * 80
    excerpt = ov._leading_excerpt([long_sentence])
    assert 0 < ov._text_weight(excerpt) <= ov._DESIGN_TEXT_MAX_WEIGHT
    assert excerpt.endswith("word")
    assert ov._leading_excerpt(["", "   "]) == ""


def test_unusable_designed_voice_raises_and_is_not_cached(fake_omnivoice, config):
    FakeOmniVoice.audio_override = np.zeros(100, dtype=np.float32)  # far too short
    provider = _provider(config)
    with pytest.raises(RuntimeError, match="voice design"):
        provider.synthesize("Hello.", b"", return_bytes=True)
    cache_dir = os.path.join(config.output_dir, ".omnivoice_voices")
    assert not [name for name in os.listdir(cache_dir) if name.endswith(".pt")]


# ── FlashInfer ────────────────────────────────────────────────────────────────


def test_flashinfer_is_applied_when_enabled(fake_omnivoice, config, voice_bytes, monkeypatch):
    patched: list[Any] = []
    models_pkg = types.ModuleType("omnivoice.models")
    flash = types.ModuleType("omnivoice.models.omnivoice_flashinfer")
    flash.apply_flashinfer = lambda model, enable_cuda_graph=False: patched.append(model) or model
    monkeypatch.setitem(sys.modules, "omnivoice.models", models_pkg)
    monkeypatch.setitem(sys.modules, "omnivoice.models.omnivoice_flashinfer", flash)

    config.voice_transcript = "Reference words."
    config.tts_options = {"flashinfer": True}
    provider = _provider(config, device="cuda:0")
    provider.synthesize("Hello.", voice_bytes, return_bytes=True)
    assert patched == [_model()]

    # Not available on CPU: ignored rather than failing the run.
    cpu_provider = _provider(config, device="cpu")
    cpu_provider.synthesize("Hello.", voice_bytes, return_bytes=True)
    assert len(patched) == 1


def test_flashinfer_missing_module_raises_a_clear_error(fake_omnivoice, config, monkeypatch):
    monkeypatch.setitem(sys.modules, "omnivoice.models", None)
    config.tts_options = {"flashinfer": True}
    provider = _provider(config, device="cuda:0")
    with pytest.raises(RuntimeError, match="flashinfer"):
        provider.ensure_ready()
    assert provider.is_ready is False


# ── Voice presets ─────────────────────────────────────────────────────────────


def test_save_voice_preset_round_trips_through_config_voice_preset(fake_omnivoice, config, voice_bytes, tmp_path):
    clip = tmp_path / "narrator.wav"
    clip.write_bytes(voice_bytes)
    config.voice_file = str(clip)
    config.voice_transcript = "Reference words."
    saver = _provider(config, device="cuda:0")
    saver.synthesize_batch(["Warm up."], voice_bytes)

    preset = str(tmp_path / "presets" / "narrator.pt")
    info = saver.save_voice_preset(preset)  # defaults: config.voice_file + config.voice_transcript
    assert json.loads(json.dumps(info)) == info  # JSON-safe
    assert info["path"] == preset and info["mode"] == "clone"
    assert info["ref_text"] == "Reference words." and info["model_id"] == "k2-fsa/OmniVoice"
    assert info["num_codebooks"] == 8 and info["ref_frames"] == 50 and info["ref_seconds"] == 2.0
    assert os.path.isfile(preset)
    # The prompt already encoded for synthesis was reused, not rebuilt.
    assert len(_model().prompt_calls) == 1
    saved = _model().generate_calls[-1]["voice_clone_prompt"]

    # A new instance needs neither the clip nor a transcript any more.
    loader = _provider(AudiobookConfig(output_dir=config.output_dir, voice_preset=preset), device="cuda:1")
    loader.synthesize_batch(["One.", "Two."], b"")
    model = _model()
    assert model.prompt_calls == [] and model.asr_loads == [] and _design_calls(model) == []
    used = model.generate_calls[-1]["voice_clone_prompt"]
    assert isinstance(used, FakeVoiceClonePrompt)
    assert torch.equal(used.ref_audio_tokens, saved.ref_audio_tokens)
    assert used.ref_text == saved.ref_text and used.ref_rms == pytest.approx(saved.ref_rms)

    described = loader.load_voice_preset(preset)
    assert described["path"] == preset and described["ref_text"] == "Reference words."
    assert described["model_id"] == "k2-fsa/OmniVoice" and described["mode"] == "unknown"
    assert {k: v for k, v in described.items() if k != "mode"} == {k: v for k, v in info.items() if k != "mode"}
    assert json.loads(json.dumps(described)) == described


def test_save_voice_preset_takes_an_explicit_clip_and_transcript(fake_omnivoice, config, voice_bytes, tmp_path):
    config.voice_transcript = "Configured transcript."
    provider = _provider(config)
    preset = str(tmp_path / "narrator.pt")
    info = provider.save_voice_preset(preset, voice_bytes, transcript="Explicit transcript.")
    assert info["mode"] == "clone" and info["ref_text"] == "Explicit transcript."
    assert _model().prompt_calls[0]["ref_text"] == "Explicit transcript."
    assert _model().asr_loads == []
    with pytest.raises(ValueError, match="destination path"):
        provider.save_voice_preset("", voice_bytes)


def test_save_voice_preset_without_a_clip_saves_the_designed_voice(fake_omnivoice, config, tmp_path):
    config.tts_instruct = "female, low pitch"
    provider = _provider(config)
    provider.synthesize_batch(["Chunk one."], b"")
    narrator = _model().generate_calls[-1]["voice_clone_prompt"]

    preset = str(tmp_path / "designed.pt")
    info = provider.save_voice_preset(preset)
    assert info["mode"] == "design" and info["ref_text"] == ov._DESIGN_TEXTS["en"]
    assert len(_design_calls(_model())) == 1  # the book's voice, not a new one

    loader = _provider(AudiobookConfig(output_dir=str(tmp_path / "elsewhere"), voice_preset=preset))
    loader.synthesize("Chunk two.", b"", return_bytes=True)
    assert _design_calls(_model()) == []
    assert torch.equal(_model().generate_calls[-1]["voice_clone_prompt"].ref_audio_tokens, narrator.ref_audio_tokens)


class _NotATensor:
    """An arbitrary object: pickling it into a preset must not be loadable."""


def test_load_voice_preset_rejects_files_that_are_not_presets(fake_omnivoice, config, tmp_path):
    provider = _provider(config)

    def write(name: str, payload) -> str:
        path = str(tmp_path / name)
        torch.save(payload, path)
        return path

    good = {"format_version": 1, "ref_audio_tokens": torch.zeros((8, 20), dtype=torch.long),
            "ref_text": "Words.", "ref_rms": 0.1}
    assert provider.load_voice_preset(write("good.pt", good))["ref_frames"] == 20

    garbage = tmp_path / "garbage.pt"
    garbage.write_bytes(b"this is not a torch file")
    cases = {
        str(tmp_path / "missing.pt"): "not found",
        str(garbage): "not an OmniVoice voice preset",
        # weights_only=True refuses to unpickle arbitrary classes.
        write("object.pt", {**good, "ref_text": _NotATensor()}): "not an OmniVoice voice preset",
        write("list.pt", [1, 2, 3]): "unexpected content",
        write("version.pt", {**good, "format_version": 2}): "format 2",
        write("float.pt", {**good, "ref_audio_tokens": torch.zeros((8, 20))}): "no usable reference audio tokens",
        write("text.pt", {**good, "ref_text": "  "}): "no reference transcript",
        write("rms.pt", {**good, "ref_rms": float("nan")}): "invalid reference loudness",
        write("codebooks.pt", {**good, "ref_audio_tokens": torch.zeros((4, 20), dtype=torch.long)}): "4 audio codebooks",
        write("vocab.pt", {**good, "ref_audio_tokens": torch.full((8, 20), 5000)}): "outside the model's vocabulary",
    }
    for path, message in cases.items():
        with pytest.raises(ValueError, match=message):
            provider.load_voice_preset(path)

    # The same file set as config.voice_preset fails the run instead of being ignored.
    provider.config.voice_preset = str(garbage)
    with pytest.raises(RuntimeError, match="not an OmniVoice voice preset"):
        provider.synthesize("Hello.", b"", return_bytes=True)
    assert _model().generate_calls == []


# ── design_voice ──────────────────────────────────────────────────────────────


def test_design_voice_returns_the_clip_the_book_will_clone(fake_omnivoice, config):
    config.tts_instruct = "female, low pitch"
    studio = _provider(config, device="cuda:0")
    wav, sample_rate, spoken = studio.design_voice()

    assert sample_rate == _SAMPLE_RATE and spoken == ov._DESIGN_TEXTS["en"]
    audio, read_rate = sf.read(io.BytesIO(wav), dtype="float32")
    assert read_rate == _SAMPLE_RATE and len(audio) == len(_waveform(spoken))
    design = _design_calls(_model())
    assert len(design) == 1 and design[0]["instruct"] == "female, low pitch"
    assert design[0]["language"] == "English" and design[0]["text"] == [spoken]

    # Cached: asking again, on this instance or another, returns the same clip.
    assert studio.design_voice() == (wav, sample_rate, spoken)
    assert len(_design_calls(_model())) == 1
    other = _provider(config, device="cuda:1")
    assert other.design_voice() == (wav, sample_rate, spoken)
    assert _design_calls(_model()) == [] and _model().prompt_calls == []

    # The book narrates with exactly that voice: no second design, and the
    # clip that was encoded is the clip that was auditioned.
    other.synthesize_batch(["Chunk one.", "Chunk two."], b"")
    assert _design_calls(_model()) == []
    encoded = FakeOmniVoice.instances[0].prompt_calls[0]
    with open(encoded["ref_audio"], "rb") as fh:
        assert fh.read() == wav
    assert encoded["ref_text"] == spoken
    assert torch.equal(
        _model().generate_calls[-1]["voice_clone_prompt"].ref_audio_tokens,
        torch.full((8, 50), 1, dtype=torch.long),
    )


def test_design_voice_arguments_override_the_config(fake_omnivoice, config):
    config.tts_instruct = "female"
    provider = _provider(config)
    wav, _, spoken = provider.design_voice("male, elderly", "A sentence chosen in the studio.", "German")
    call = _design_calls(_model())[-1]
    assert call["instruct"] == "male, elderly" and call["language"] == "German"
    assert call["text"] == ["A sentence chosen in the studio."] and spoken == "A sentence chosen in the studio."
    assert provider.design_voice("male, elderly", "A sentence chosen in the studio.", "German")[0] == wav
    assert len(_design_calls(_model())) == 1

    # A different voice is a different cache entry.
    provider.design_voice()
    assert len(_design_calls(_model())) == 2
    assert _design_calls(_model())[-1]["instruct"] == "female"


def test_design_voice_force_re_rolls_for_every_instance(fake_omnivoice, config):
    config.tts_instruct = "male"
    first = _provider(config, device="cuda:0")
    second = _provider(config, device="cuda:1")
    first.design_voice()
    second.synthesize("Chunk.", b"", return_bytes=True)
    before = FakeOmniVoice.instances[1].generate_calls[-1]["voice_clone_prompt"].ref_audio_tokens.clone()

    first.design_voice(force=True)
    assert len(_design_calls(FakeOmniVoice.instances[0])) == 2
    # The other instance drops its in-memory copy of the replaced voice.
    second.synthesize("Chunk.", b"", return_bytes=True)
    after = FakeOmniVoice.instances[1].generate_calls[-1]["voice_clone_prompt"].ref_audio_tokens
    assert not torch.equal(before, after)
    assert _design_calls(FakeOmniVoice.instances[1]) == []
    mine = first.synthesize_batch(["Chunk."], b"")
    assert len(mine) == 1
    assert torch.equal(FakeOmniVoice.instances[0].generate_calls[-1]["voice_clone_prompt"].ref_audio_tokens.cpu(), after.cpu())


def test_design_voice_needs_a_sentence_for_languages_without_a_built_in_one(fake_omnivoice, config):
    config.language = "German"
    provider = _provider(config)
    with pytest.raises(ValueError, match="pass a sentence in that language"):
        provider.design_voice()
    assert _design_calls(_model()) == []
    _, _, spoken = provider.design_voice(text="Guten Abend, liebe Hörerinnen und Hörer.")
    assert spoken == "Guten Abend, liebe Hörerinnen und Hörer."


def test_auditioned_voice_can_be_saved_as_a_preset(fake_omnivoice, config, tmp_path):
    """The Voice Studio flow: design_voice -> save_voice_preset(path, wav, transcript=text)."""
    config.tts_instruct = "female, british accent"
    provider = _provider(config)
    wav, _, spoken = provider.design_voice()
    preset = str(tmp_path / "studio.pt")
    info = provider.save_voice_preset(preset, wav, transcript=spoken)
    assert info["mode"] == "clone" and info["ref_text"] == spoken
    encoded = _model().prompt_calls[-1]
    assert encoded["ref_text"] == spoken
    with open(encoded["ref_audio"], "rb") as fh:
        assert fh.read() == wav
    assert provider.load_voice_preset(preset)["ref_text"] == spoken
