"""
test_new_engines.py
===================
Breeze TTS 2 and Ming-3.0-TTS against fake upstream code.

The real models need a GPU and their own environments; these tests check how
each provider drives upstream: the requests it builds, batching and the
out-of-memory split, language handling, transcripts and the audio it returns.
"""

from __future__ import annotations

import io
import os
import sys
import types

import numpy as np
import pytest
import soundfile as sf
import torch

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from audiobook_factory.pipeline import AudiobookConfig
from audiobook_factory.tts_providers import get_tts_provider


def _voice(tmp_path, seconds: float = 1.0, rate: int = 22050) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, (0.2 * np.sin(np.linspace(0, 300, int(rate * seconds)))).astype(np.float32), rate, format="WAV")
    return buffer.getvalue()


def _install(monkeypatch, modules: dict[str, types.ModuleType]) -> None:
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)


def _clone(tmp_path, env_var: str, marker: str, monkeypatch) -> None:
    path = tmp_path / "clone"
    (path / marker).parent.mkdir(parents=True, exist_ok=True)
    (path / marker).touch()
    monkeypatch.setenv(env_var, str(path))


def _hub(monkeypatch, tmp_path) -> None:
    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda repo_id, **_: str(tmp_path))


# ── Breeze ────────────────────────────────────────────────────────────────────

class _BreezeModel:
    def __init__(self, fail_above: int = 0) -> None:
        self.generation_config = types.SimpleNamespace(temperature=0.9)
        self.depth_decoder = types.SimpleNamespace(generation_config=types.SimpleNamespace(temperature=0.9))
        self.fail_above = fail_above
        self.batches: list[int] = []

    def generate(self, *, requests, **kwargs):
        if self.fail_above and len(requests) > self.fail_above:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")
        self.batches.append(len(requests))
        return [torch.full((1, 2400 * (i + 1)), 0.1) for i, _ in enumerate(requests)]


@pytest.fixture
def breeze(monkeypatch, tmp_path):
    _clone(tmp_path, "ABM_BREEZE_REPO", "breeze_infer/__init__.py", monkeypatch)
    _hub(monkeypatch, tmp_path)
    model = _BreezeModel()
    seen: dict = {}
    runtime = types.ModuleType("breeze_infer.runtime")
    runtime.load_runtime = lambda ckpt, *, device, attn_implementation: ("tok", model, "codec")
    runtime.update_generation_config_for_breeze = lambda m: None
    templates = types.ModuleType("breeze_infer.templates")

    def _prepare(tokenizer, audio_tokenizer, m, requests, template, **kw):
        seen["requests"], seen["template"], seen["guidance"] = requests, template, kw
        return {"requests": requests}

    templates.prepare_inputs = _prepare
    templates.get_template = lambda name: name
    templates.select_template_name = lambda request: "ref_edit_tata" if request.get("instruction") else "ref_clone_tata"
    logits = types.ModuleType("models.logits_process")
    logits.GeneratedTokenRepetitionPenaltyLogitsProcessor = lambda penalty: object()
    _install(monkeypatch, {
        "breeze_infer": types.ModuleType("breeze_infer"), "breeze_infer.runtime": runtime,
        "breeze_infer.templates": templates, "models": types.ModuleType("models"),
        "models.logits_process": logits,
    })
    return model, seen


class TestBreeze:

    def test_batch_clones_with_the_transcript(self, breeze, tmp_path):
        model, seen = breeze
        config = AudiobookConfig(tts_provider_name="breeze", voice_transcript="Hello, this is me.")
        provider = get_tts_provider("breeze", config, device="cpu")
        results = provider.synthesize_batch(["One.", "Two.", "Three."], _voice(tmp_path))
        assert [round(duration, 2) for _, duration in results] == [0.1, 0.2, 0.3]
        assert all(r["ref_text"] == "Hello, this is me." for r in seen["requests"])
        assert seen["template"] == "ref_clone_tata" and seen["guidance"]["guidance_scale"] == 1.0
        assert model.batches == [3]

    def test_instruction_selects_the_edit_template(self, breeze, tmp_path):
        _, seen = breeze
        config = AudiobookConfig(voice_transcript="Hi.", tts_instruct="Calm and slow.")
        get_tts_provider("breeze", config, device="cpu").synthesize_batch(["One."], _voice(tmp_path))
        assert seen["template"] == "ref_edit_tata" and seen["requests"][0]["instruction"] == "Calm and slow."

    def test_out_of_memory_halves_the_batch(self, breeze, tmp_path):
        model, _ = breeze
        model.fail_above = 2
        provider = get_tts_provider("breeze", AudiobookConfig(voice_transcript="Hi."), device="cpu")
        assert len(provider.synthesize_batch(["a.", "b.", "c.", "d."], _voice(tmp_path))) == 4
        assert model.batches == [2, 2] and provider.batch_size_limit == 2

    def test_transcript_is_required(self, breeze, tmp_path):
        provider = get_tts_provider("breeze", AudiobookConfig(voice_transcript=""), device="cpu")
        with pytest.raises(RuntimeError, match="transcript"):
            provider.synthesize_batch(["One."], _voice(tmp_path))

    def test_missing_clone_is_explained(self, monkeypatch):
        monkeypatch.delenv("ABM_BREEZE_REPO", raising=False)
        provider = get_tts_provider("breeze", AudiobookConfig(voice_transcript="Hi."), device="cpu")
        with pytest.raises(RuntimeError, match="ABM_BREEZE_REPO"):
            provider.ensure_ready()


# ── Ming ──────────────────────────────────────────────────────────────────────

class _FakeMing:
    calls: list[dict] = []

    def __init__(self, model_id, device="cuda:0", text_normalization=True, cache_prompts=True):
        self.sample_rate = 44100
        self.text_normalization = text_normalization
        self.spkemb_extractor = lambda wav: torch.zeros(1, 192) + wav.shape[-1]
        _FakeMing.calls = []

    def _pad_prompt_waveform(self, waveform):
        return waveform

    def synthesize(self, **kwargs):
        _FakeMing.calls.append(kwargs)
        return torch.full((1, 44100), 0.1)


@pytest.fixture
def ming(monkeypatch, tmp_path):
    import transformers

    monkeypatch.setattr(transformers, "__version__", "4.46.3")
    _clone(tmp_path, "ABM_MING_REPO", "ming_tts.py", monkeypatch)
    module = types.ModuleType("ming_tts")
    module.MingTTS = _FakeMing
    _install(monkeypatch, {"ming_tts": module})


class TestMing:

    @pytest.mark.parametrize("language, task, code", [
        ("English", "TTS", "en"), ("Chinese", "TTS", "zh"), ("Japanese", "MULTILINGUAL", "ja"),
        ("French", "MULTILINGUAL", "fr"),
    ])
    def test_language_picks_the_upstream_task(self, ming, tmp_path, language, task, code):
        config = AudiobookConfig(language=language, voice_transcript="Reference words.")
        audio, duration = get_tts_provider("ming", config, device="cpu").synthesize(
            "Some text.", _voice(tmp_path), return_bytes=True)
        call = _FakeMing.calls[-1]
        assert (call["task"], call["language_id"]) == (task, code)
        assert call["prompt_text"] == "Reference words." and call["use_spk_emb"] is True
        assert duration == pytest.approx(1.0) and sf.info(io.BytesIO(audio)).samplerate == 44100

    def test_normalizer_is_off_and_prompt_reads_with_soundfile(self, ming, tmp_path):
        pytest.importorskip("torchaudio")
        provider = get_tts_provider("ming", AudiobookConfig(voice_transcript="Hi."), device="cpu")
        provider.ensure_ready()
        assert provider._model.text_normalization is False
        path = provider.resolve_voice_path(_voice(tmp_path, seconds=0.5, rate=22050))
        waveform, embeddings = provider._model._load_prompt([path])
        assert waveform.shape[0] == 1 and abs(waveform.shape[-1] - 22050) <= 2  # 0.5 s at 44.1 kHz
        assert float(embeddings[0][0, 0]) == pytest.approx(8000, abs=2)        # 0.5 s at 16 kHz

    def test_transcript_is_required(self, ming, tmp_path):
        provider = get_tts_provider("ming", AudiobookConfig(voice_transcript=""), device="cpu")
        with pytest.raises(RuntimeError, match="transcript"):
            provider.synthesize("Text.", _voice(tmp_path), return_bytes=True)


def test_ming_refuses_other_transformers_versions(ming, monkeypatch):
    import transformers

    monkeypatch.setattr(transformers, "__version__", "4.57.3")
    provider = get_tts_provider("ming", AudiobookConfig(voice_transcript="Hi."), device="cpu")
    with pytest.raises(RuntimeError, match="4.46"):
        provider.ensure_ready()
