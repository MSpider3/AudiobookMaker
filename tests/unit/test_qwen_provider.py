"""
tests/unit/test_qwen_provider.py
================================
Unit tests for the Qwen3-TTS provider.

They run without network, GPU or the ``qwen_tts`` package: a fake
``qwen_tts`` module is injected whose ``Qwen3TTSModel`` mirrors the call
signatures and validation of qwen-tts 0.1.1
(``qwen_tts/inference/qwen3_tts_model.py``). The fake is stricter than
upstream in one place on purpose: its ``_merge_generate_kwargs`` rejects
unknown names instead of silently dropping them, so a kwarg upstream would
ignore fails the test.
"""

from __future__ import annotations

import ast
import inspect
import json
import os
import pickle
import sys
import threading
import time
import types
from dataclasses import asdict, dataclass
from types import SimpleNamespace
from typing import Any, Optional

import numpy as np
import pytest

torch = pytest.importorskip("torch")

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from audiobook_factory.pipeline import AudiobookConfig
from audiobook_factory.tts_providers import qwen_provider as qp
from audiobook_factory.tts_providers.qwen_provider import QwenTTSProvider

_SAMPLES_PER_FRAME: int = 1920
_SAMPLE_RATE: int = 24000
_LANGUAGES: list[str] = [
    "auto", "chinese", "english", "german", "italian", "portuguese",
    "spanish", "japanese", "korean", "french", "russian",
]
_SPEAKERS: dict[str, int] = {
    "serena": 3066, "vivian": 3065, "uncle_fu": 3010, "ryan": 3061, "aiden": 2861,
    "ono_anna": 2873, "sohee": 2864, "eric": 2875, "dylan": 2878,
}
# model id -> (tts_model_type, tts_model_size, speaker embedding size)
_CHECKPOINTS: dict[str, tuple[str, str, int]] = {
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base": ("base", "1b7", 8),
    "Qwen/Qwen3-TTS-12Hz-0.6B-Base": ("base", "0b6", 4),
    "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice": ("custom_voice", "1b7", 8),
    "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice": ("custom_voice", "0b6", 4),
    "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign": ("voice_design", "1b7", 8),
}
_BASE = "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
_BASE_SMALL = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
_CUSTOM = "Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice"
_CUSTOM_SMALL = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
_DESIGN = "Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign"


# ── Fake qwen_tts ─────────────────────────────────────────────────────────────

@dataclass
class VoiceClonePromptItem:
    """Same fields as qwen_tts.VoiceClonePromptItem."""

    ref_code: Optional[Any]
    ref_spk_embedding: Any
    x_vector_only_mode: bool
    icl_mode: bool
    ref_text: Optional[str] = None


class _FakeSpeechTokenizer:
    def get_decode_upsample_rate(self) -> int:
        return _SAMPLES_PER_FRAME

    def get_output_sample_rate(self) -> int:
        return _SAMPLE_RATE


class _FakeInnerModel:
    """Stand-in for Qwen3TTSForConditionalGeneration."""

    def __init__(self, model_type: str, size: str, hidden: int) -> None:
        speakers = dict(_SPEAKERS) if model_type == "custom_voice" else {}
        self.tts_model_type = model_type
        self.tts_model_size = size
        self.tokenizer_type = "qwen3_tts_tokenizer_12hz"
        self.speaker_encoder_sample_rate = 24000
        self.speech_tokenizer = _FakeSpeechTokenizer()
        self.generation_config = SimpleNamespace(pad_token_id=None, eos_token_id=None)
        self.talker = SimpleNamespace(generation_config=SimpleNamespace(pad_token_id=None))
        self.config = SimpleNamespace(
            speaker_encoder_config=SimpleNamespace(enc_dim=hidden, sample_rate=24000),
            talker_config=SimpleNamespace(
                hidden_size=hidden, vocab_size=3072, num_code_groups=16, spk_id=speakers,
                codec_eos_token_id=2150,
                code_predictor_config=SimpleNamespace(vocab_size=2048),
            ),
        )
        self.supported_speakers = speakers.keys()
        self.supported_languages = list(_LANGUAGES)

    def get_supported_speakers(self):
        return self.supported_speakers

    def get_supported_languages(self):
        return self.supported_languages


class FakeQwen3TTSModel:
    """Mirrors the public surface of qwen_tts.Qwen3TTSModel (0.1.1)."""

    loads: list[tuple[str, dict[str, Any]]] = []
    instances: list["FakeQwen3TTSModel"] = []
    oom_above: int | None = None          # OOM when a batch is larger than this
    fail_batches_above: int | None = None  # plain RuntimeError above this size
    runaway: dict[str, int] = {}           # text -> how many more times it never stops
    design_delay: float = 0.0

    def __init__(self, model: _FakeInnerModel, name: str) -> None:
        self.model = model
        self.name = name
        self.generate_defaults = {"max_new_tokens": 8192}
        self.calls: list[dict[str, Any]] = []
        self.prompt_calls: list[dict[str, Any]] = []

    @classmethod
    def reset(cls) -> None:
        cls.loads = []
        cls.instances = []
        cls.oom_above = None
        cls.fail_batches_above = None
        cls.runaway = {}
        cls.design_delay = 0.0

    @classmethod
    def all_calls(cls, fn: str | None = None) -> list[dict[str, Any]]:
        return [c for inst in cls.instances for c in inst.calls if fn is None or c["fn"] == fn]

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str, **kwargs) -> "FakeQwen3TTSModel":
        speakers = None
        if os.path.isdir(pretrained_model_name_or_path):     # a fine-tuned local checkpoint
            with open(os.path.join(pretrained_model_name_or_path, "config.json"), encoding="utf-8") as fh:
                local = json.load(fh)
            model_type, size, hidden = local["tts_model_type"], local["tts_model_size"], 8
            speakers = local["talker_config"]["spk_id"]
        else:
            model_type, size, hidden = _CHECKPOINTS[pretrained_model_name_or_path]
        inner = _FakeInnerModel(model_type, size, hidden)
        if speakers is not None:
            inner.config.talker_config.spk_id = dict(speakers)
            inner.supported_speakers = inner.config.talker_config.spk_id.keys()
        instance = cls(inner, pretrained_model_name_or_path)
        cls.loads.append((pretrained_model_name_or_path, dict(kwargs)))
        cls.instances.append(instance)
        return instance

    # -- helpers mirroring upstream --------------------------------------------

    def _ensure_list(self, x):
        return x if isinstance(x, list) else [x]

    def _require(self, model_type: str, fn: str) -> None:
        if self.model.tts_model_type != model_type:
            raise ValueError(f"model with tts_model_type: {self.model.tts_model_type} does not support {fn}")

    def _languages(self, language, count: int) -> list[str]:
        if isinstance(language, list):
            languages = language
        else:
            languages = [language] * count if language is not None else ["Auto"] * count
        if len(languages) == 1 and count > 1:
            languages = languages * count
        if len(languages) != count:
            raise ValueError(f"Batch size mismatch: text={count}, language={len(languages)}")
        supported = set(self.get_supported_languages())
        bad = [lang for lang in languages if lang is None or str(lang).lower() not in supported]
        if bad:
            raise ValueError(f"Unsupported languages: {bad}. Supported: {sorted(supported)}")
        return languages

    def _merge_generate_kwargs(
        self,
        do_sample: Optional[bool] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        temperature: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
        subtalker_dosample: Optional[bool] = None,
        subtalker_top_k: Optional[int] = None,
        subtalker_top_p: Optional[float] = None,
        subtalker_temperature: Optional[float] = None,
        max_new_tokens: Optional[int] = None,
    ) -> dict[str, Any]:
        # Upstream has a **kwargs catch-all here that drops unknown names
        # silently; the fake raises TypeError for them instead.
        hard_defaults = dict(
            do_sample=True, top_k=50, top_p=1.0, temperature=0.9, repetition_penalty=1.05,
            subtalker_dosample=True, subtalker_top_k=50, subtalker_top_p=1.0,
            subtalker_temperature=0.9, max_new_tokens=2048,
        )
        given = dict(
            do_sample=do_sample, top_k=top_k, top_p=top_p, temperature=temperature,
            repetition_penalty=repetition_penalty, subtalker_dosample=subtalker_dosample,
            subtalker_top_k=subtalker_top_k, subtalker_top_p=subtalker_top_p,
            subtalker_temperature=subtalker_temperature, max_new_tokens=max_new_tokens,
        )
        merged = {}
        for name, value in given.items():
            merged[name] = value if value is not None else self.generate_defaults.get(name, hard_defaults[name])
        return merged

    def _emit(self, texts: list[str], gen_kwargs: dict[str, Any]) -> tuple[list[np.ndarray], int]:
        try:
            return self._emit_unchecked(texts, gen_kwargs)
        except Exception:
            self.calls[-1]["failed"] = True
            raise

    def _emit_unchecked(self, texts: list[str], gen_kwargs: dict[str, Any]) -> tuple[list[np.ndarray], int]:
        cls = type(self)
        if cls.oom_above is not None and len(texts) > cls.oom_above:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate 2.00 GiB")
        if cls.fail_batches_above is not None and len(texts) > cls.fail_batches_above:
            raise RuntimeError("probability tensor contains either `inf`, `nan` or element < 0")
        budget = int(gen_kwargs["max_new_tokens"])
        wavs = []
        for text in texts:
            frames = min(len(text), budget)           # one code frame per character
            if cls.runaway.get(text, 0) > 0:
                cls.runaway[text] -= 1
                # Never emits end-of-speech. Real upstream then returns
                # max_new_tokens - 1 frames (checked against qwen-tts 0.1.1).
                frames = budget - 1
            wavs.append(np.full(frames * _SAMPLES_PER_FRAME, 0.1, dtype=np.float32))
        return wavs, _SAMPLE_RATE

    # -- public API -------------------------------------------------------------

    def create_voice_clone_prompt(self, ref_audio, ref_text=None, x_vector_only_mode=False):
        self._require("base", "create_voice_clone_prompt")
        if not isinstance(ref_audio, str) or not os.path.isfile(ref_audio):
            raise TypeError(f"Unsupported audio input: {ref_audio!r}")
        if not x_vector_only_mode and (ref_text is None or ref_text == ""):
            raise ValueError("ref_text is required when x_vector_only_mode=False (ICL mode). Bad index=0")
        with open(ref_audio, "rb") as fh:
            data = fh.read()
        self.prompt_calls.append({
            "ref_audio": ref_audio, "ref_text": ref_text,
            "x_vector_only_mode": x_vector_only_mode, "size": len(data),
        })
        hidden = self.model.config.speaker_encoder_config.enc_dim
        embedding = torch.arange(hidden, dtype=torch.float32) / hidden + (sum(data[:64]) % 97) / 97.0
        code = torch.arange(5 * 16, dtype=torch.long).reshape(5, 16) % 2048
        return [VoiceClonePromptItem(
            ref_code=None if x_vector_only_mode else code,
            ref_spk_embedding=embedding,
            x_vector_only_mode=bool(x_vector_only_mode),
            icl_mode=bool(not x_vector_only_mode),
            ref_text=ref_text,
        )]

    def generate_voice_clone(
        self,
        text,
        language=None,
        ref_audio=None,
        ref_text=None,
        x_vector_only_mode=False,
        voice_clone_prompt=None,
        non_streaming_mode: bool = False,
        **kwargs,
    ):
        self._require("base", "generate_voice_clone")
        texts = self._ensure_list(text)
        languages = self._languages(language, len(texts))
        if voice_clone_prompt is None:
            if ref_audio is None:
                raise ValueError("Either `voice_clone_prompt` or `ref_audio` must be provided.")
            items = self.create_voice_clone_prompt(ref_audio, ref_text, x_vector_only_mode)
        else:
            assert isinstance(voice_clone_prompt, list), "pass the list of prompt items, not a dict"
            items = voice_clone_prompt
        if len(items) == 1 and len(texts) > 1:
            items = items * len(texts)
        if len(items) != len(texts):
            raise ValueError(f"Batch size mismatch: prompt={len(items)}, text={len(texts)}")
        for item in items:
            if item.icl_mode:
                assert item.ref_code is not None and item.ref_text, "ICL prompt needs codes and text"
        gen_kwargs = self._merge_generate_kwargs(**kwargs)
        self.calls.append({
            "fn": "clone", "texts": list(texts), "languages": languages, "ref_audio": ref_audio,
            "voice_clone_prompt": voice_clone_prompt, "non_streaming_mode": non_streaming_mode,
            "raw_kwargs": dict(kwargs), "gen_kwargs": gen_kwargs,
        })
        return self._emit(texts, gen_kwargs)

    def generate_voice_design(self, text, instruct, language=None, non_streaming_mode: bool = True, **kwargs):
        self._require("voice_design", "generate_voice_design")
        texts = self._ensure_list(text)
        languages = self._languages(language, len(texts))
        instructs = self._ensure_list(instruct)
        if len(instructs) == 1 and len(texts) > 1:
            instructs = instructs * len(texts)
        if len(instructs) != len(texts):
            raise ValueError("Batch size mismatch")
        gen_kwargs = self._merge_generate_kwargs(**kwargs)
        if type(self).design_delay:
            time.sleep(type(self).design_delay)
        self.calls.append({
            "fn": "design", "texts": list(texts), "languages": languages, "instructs": instructs,
            "non_streaming_mode": non_streaming_mode, "raw_kwargs": dict(kwargs), "gen_kwargs": gen_kwargs,
        })
        return self._emit(texts, gen_kwargs)

    def generate_custom_voice(
        self, text, speaker, language=None, instruct=None, non_streaming_mode: bool = True, **kwargs,
    ):
        self._require("custom_voice", "generate_custom_voice")
        texts = self._ensure_list(text)
        languages = self._languages(language, len(texts))
        speakers = self._ensure_list(speaker)
        received_instruct = instruct
        if self.model.tts_model_size in "0b6":   # upstream: 0b6 does not support instruct
            instruct = None
        if len(speakers) == 1 and len(texts) > 1:
            speakers = speakers * len(texts)
        if len(speakers) != len(texts):
            raise ValueError("Batch size mismatch")
        supported = set(self.get_supported_speakers())
        bad = [spk for spk in speakers if spk and str(spk).lower() not in supported]
        if bad:
            raise ValueError(f"Unsupported speakers: {bad}. Supported: {sorted(supported)}")
        gen_kwargs = self._merge_generate_kwargs(**kwargs)
        self.calls.append({
            "fn": "custom", "texts": list(texts), "languages": languages, "speakers": speakers,
            "instruct": received_instruct, "non_streaming_mode": non_streaming_mode,
            "raw_kwargs": dict(kwargs), "gen_kwargs": gen_kwargs,
        })
        return self._emit(texts, gen_kwargs)

    def get_supported_speakers(self):
        return sorted(str(name).lower() for name in self.model.get_supported_speakers())

    def get_supported_languages(self):
        return sorted(str(name).lower() for name in self.model.get_supported_languages())


# ── Fixtures ──────────────────────────────────────────────────────────────────

class _Env:
    """Builds providers that load the fake model and share one cache directory."""

    def __init__(self, cache_dir: str) -> None:
        self.cache_dir = cache_dir

    def provider(self, *, device: str = "cpu", dtype_override: str | None = None, **config) -> QwenTTSProvider:
        config.setdefault("tts_model_name", _BASE)
        return QwenTTSProvider(AudiobookConfig(**config), device=device, dtype_override=dtype_override)


@pytest.fixture
def env(monkeypatch, tmp_path):
    FakeQwen3TTSModel.reset()
    module = types.ModuleType("qwen_tts")
    module.Qwen3TTSModel = FakeQwen3TTSModel
    module.VoiceClonePromptItem = VoiceClonePromptItem
    monkeypatch.setitem(sys.modules, "qwen_tts", module)
    cache_dir = tmp_path / "qwen_cache"
    monkeypatch.setenv("ABM_QWEN_CACHE_DIR", str(cache_dir))
    monkeypatch.setattr(qp, "pipeline", _forbidden_pipeline)
    qp._SHARED_TRANSCRIPTS.clear()
    yield _Env(str(cache_dir))
    qp._SHARED_TRANSCRIPTS.clear()
    FakeQwen3TTSModel.reset()


def _forbidden_pipeline(*args, **kwargs):
    raise AssertionError("ASR must not run in this test")


@pytest.fixture
def voice_path():
    return os.path.join(_ROOT, "tests", "fixtures", "audio", "synthetic_voice_reference.wav")


@pytest.fixture
def voice_bytes(voice_path):
    with open(voice_path, "rb") as fh:
        return fh.read()


def _prompt_fields(item: Any) -> dict[str, Any]:
    return {
        "ref_code": item.ref_code, "ref_spk_embedding": item.ref_spk_embedding,
        "x_vector_only_mode": item.x_vector_only_mode, "icl_mode": item.icl_mode,
        "ref_text": item.ref_text,
    }


def _assert_same_prompt(left: Any, right: Any) -> None:
    a, b = _prompt_fields(left), _prompt_fields(right)
    assert (a["x_vector_only_mode"], a["icl_mode"], a["ref_text"]) == \
           (b["x_vector_only_mode"], b["icl_mode"], b["ref_text"])
    assert (a["ref_code"] is None) == (b["ref_code"] is None)
    if a["ref_code"] is not None:
        assert torch.equal(a["ref_code"], b["ref_code"])
    assert torch.equal(a["ref_spk_embedding"].float(), b["ref_spk_embedding"].float())


# ── Static contract ───────────────────────────────────────────────────────────

class TestProviderInfo:

    def test_module_imports_no_heavy_dependency_at_top_level(self):
        tree = ast.parse(inspect.getsource(qp))
        imported = set()
        for node in tree.body:
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add((node.module or "").split(".")[0])
        assert not imported & {"torch", "transformers", "qwen_tts", "numpy", "soundfile", "librosa"}

    def test_info_describes_the_real_checkpoints(self):
        info = QwenTTSProvider.info()
        assert info.name == "qwen"
        assert info.license == "Apache-2.0" and info.commercial_use is True
        assert info.default_model == _BASE
        assert set(info.models) == set(_CHECKPOINTS)
        assert info.native_sample_rate == 24000
        assert info.supports_batch and info.supports_seed and info.supports_instruct
        assert info.supports_voice_clone and info.transcript == "optional"
        assert info.pip_requirements and "qwen-tts" in info.pip_requirements[0]
        assert {name.lower() for name in info.preset_voices} == set(_SPEAKERS)
        assert {name.lower() for name in info.languages} == set(_LANGUAGES)
        assert 4.0 <= info.min_vram_gb <= 16.0

    def test_every_generation_argument_is_reachable(self):
        option_keys = {opt.key for opt in QwenTTSProvider.info().options}
        common_fields = {"temperature", "top_p", "top_k", "repetition_penalty"}
        assert qp._UPSTREAM_GENERATE_KWARGS <= option_keys | common_fields
        assert {"non_streaming_mode", "design_then_clone", "asr_model", "x_vector_only_mode"} <= option_keys

    def test_allowlist_matches_upstream_signature(self):
        params = set(inspect.signature(FakeQwen3TTSModel._merge_generate_kwargs).parameters) - {"self"}
        assert params == set(qp._UPSTREAM_GENERATE_KWARGS)

    def test_model_outside_allowlist_falls_back_to_default(self, env):
        p = env.provider(tts_model_name="someone/evil-repo")
        assert p._target_model_id() == _BASE


# ── 1. Base: voice clone ──────────────────────────────────────────────────────

class TestVoiceClone:

    def test_prompt_built_once_for_every_chunk_and_batch(self, env, voice_bytes, voice_path):
        p = env.provider(voice_transcript="The reference sentence.")
        out = p.synthesize_batch(["First chunk.", "Second chunk."], voice_bytes)
        p.synthesize_batch(["Third chunk."], voice_bytes)
        p.synthesize("Fourth chunk.", voice_bytes, return_bytes=True)
        # The same clip passed as a path is the same voice.
        p.synthesize("Fifth chunk.", voice_path, return_bytes=True)

        model = p._model
        assert len(model.prompt_calls) == 1
        assert model.prompt_calls[0]["ref_text"] == "The reference sentence."
        assert model.prompt_calls[0]["x_vector_only_mode"] is False
        assert len(model.calls) == 4
        for call in model.calls:
            assert call["fn"] == "clone"
            assert call["ref_audio"] is None
            assert call["voice_clone_prompt"] is model.calls[0]["voice_clone_prompt"]
        assert model.calls[0]["voice_clone_prompt"][0].icl_mode is True
        assert len(out) == 2 and all(isinstance(wav, bytes) and dur > 0 for wav, dur in out)
        assert out[0][0][:4] == b"RIFF"

    def test_without_transcript_uses_speaker_embedding_only(self, env, voice_bytes):
        p = env.provider(tts_options={"auto_transcribe": False})
        p.synthesize_batch(["A chunk."], voice_bytes)
        call = p._model.prompt_calls[0]
        assert call["x_vector_only_mode"] is True and call["ref_text"] is None
        item = p._model.calls[0]["voice_clone_prompt"][0]
        assert item.x_vector_only_mode is True and item.icl_mode is False and item.ref_code is None

    def test_x_vector_only_option_overrides_transcript(self, env, voice_bytes):
        p = env.provider(voice_transcript="Some text.", tts_options={"x_vector_only_mode": True})
        p.synthesize_batch(["A chunk."], voice_bytes)
        assert p._model.prompt_calls[0]["x_vector_only_mode"] is True

    def test_changed_transcript_rebuilds_the_prompt(self, env, voice_bytes):
        p = env.provider(voice_transcript="Version one.")
        p.synthesize_batch(["A chunk."], voice_bytes)
        p.config.voice_transcript = "Version two."
        p.synthesize_batch(["A chunk."], voice_bytes)
        assert [c["ref_text"] for c in p._model.prompt_calls] == ["Version one.", "Version two."]

    def test_missing_reference_is_a_clear_error(self, env):
        p = env.provider()
        with pytest.raises(ValueError, match="voice_file"):
            p.synthesize_batch(["A chunk."], b"")
        assert FakeQwen3TTSModel.loads == []

    def test_nonexistent_upstream_apis_are_not_used(self):
        source = inspect.getsource(qp)
        for dead in ("extract_x_vector", "get_speaker_embedding", "x_vector="):
            assert dead not in source


# ── 2. Voice presets ──────────────────────────────────────────────────────────

class TestVoicePreset:

    def _saved(self, env, voice_bytes, tmp_path):
        a = env.provider(voice_transcript="The reference sentence.", language="English")
        a.synthesize_batch(["Hello there."], voice_bytes)
        path = str(tmp_path / "narrator.pt")
        info = a.save_voice_preset(path, voice_bytes)
        return a, path, info

    def test_round_trip_gives_identical_generate_kwargs_without_voice_file(self, env, voice_bytes, tmp_path):
        a, path, info = self._saved(env, voice_bytes, tmp_path)
        assert info["mode"] == "icl" and info["ref_text"] == "The reference sentence."
        assert info["model_id"] == _BASE and info["tokenizer_type"] == "qwen3_tts_tokenizer_12hz"

        b = env.provider(voice_preset=path, voice_file="", language="English")
        assert b._needs_voice_ref() is False
        out = b.synthesize_batch(["Hello there."], b"")
        assert len(out) == 1 and out[0][1] > 0

        assert b._model is not a._model
        assert b._model.prompt_calls == [], "a preset must not re-extract features"
        call_a, call_b = a._model.calls[-1], b._model.calls[-1]
        for key in ("fn", "texts", "languages", "ref_audio", "non_streaming_mode", "raw_kwargs"):
            assert call_a[key] == call_b[key], key
        assert len(call_b["voice_clone_prompt"]) == 1
        _assert_same_prompt(call_a["voice_clone_prompt"][0], call_b["voice_clone_prompt"][0])
        assert isinstance(call_b["voice_clone_prompt"][0], VoiceClonePromptItem)

    def test_file_is_plain_tensors_and_metadata(self, env, voice_bytes, tmp_path):
        _, path, _ = self._saved(env, voice_bytes, tmp_path)
        payload = torch.load(path, map_location="cpu", weights_only=True)
        assert payload["format_version"] == qp._PRESET_FORMAT_VERSION
        assert payload["model_id"] == _BASE
        assert payload["tokenizer_type"] == "qwen3_tts_tokenizer_12hz"
        assert payload["sample_rate"] == 24000
        assert payload["created_at"].endswith("Z")
        item = payload["items"][0]
        assert set(item) == {"ref_code", "ref_spk_embedding", "x_vector_only_mode", "icl_mode", "ref_text"}
        assert item["ref_code"].device.type == "cpu" and item["ref_spk_embedding"].device.type == "cpu"
        assert item["icl_mode"] is True and item["x_vector_only_mode"] is False

    def test_info_is_readable_without_a_model(self, env, voice_bytes, tmp_path):
        _, path, _ = self._saved(env, voice_bytes, tmp_path)
        loads_before = len(FakeQwen3TTSModel.loads)
        info = QwenTTSProvider.read_voice_preset_info(path)
        assert info["mode"] == "icl" and info["speaker_embedding_dim"] == 8
        assert len(FakeQwen3TTSModel.loads) == loads_before

    def test_preset_for_another_model_size_is_rejected(self, env, voice_bytes, tmp_path):
        _, path, _ = self._saved(env, voice_bytes, tmp_path)
        small = env.provider(tts_model_name=_BASE_SMALL, voice_preset=path)
        with pytest.raises(ValueError, match="speaker embedding size 8"):
            small.load_voice_preset(path)
        with pytest.raises(ValueError, match="1.7B-Base"):
            small.synthesize_batch(["Hello."], b"")
        assert small._model.calls == []

    def test_preset_for_another_tokenizer_is_rejected(self, env, voice_bytes, tmp_path):
        _, path, _ = self._saved(env, voice_bytes, tmp_path)
        payload = torch.load(path, map_location="cpu", weights_only=True)
        payload["tokenizer_type"] = "qwen3_tts_tokenizer_25hz"
        torch.save(payload, path)
        p = env.provider(voice_preset=path)
        with pytest.raises(ValueError, match="tokenizer"):
            p.load_voice_preset(path)

    def test_newer_format_version_is_rejected(self, env, voice_bytes, tmp_path):
        _, path, _ = self._saved(env, voice_bytes, tmp_path)
        payload = torch.load(path, map_location="cpu", weights_only=True)
        payload["format_version"] = qp._PRESET_FORMAT_VERSION + 1
        torch.save(payload, path)
        with pytest.raises(ValueError, match="format version"):
            QwenTTSProvider.read_voice_preset_info(path)

    def test_custom_voice_checkpoint_cannot_load_or_save_presets(self, env, voice_bytes, tmp_path):
        _, path, _ = self._saved(env, voice_bytes, tmp_path)
        custom = env.provider(tts_model_name=_CUSTOM)
        with pytest.raises(ValueError, match="Base checkpoint"):
            custom.load_voice_preset(path)
        with pytest.raises(ValueError, match="Base checkpoint"):
            custom.save_voice_preset(str(tmp_path / "x.pt"), voice_bytes)

    def test_arbitrary_pickle_is_refused_and_not_executed(self, env, tmp_path):
        sentinel = tmp_path / "pwned"

        class _Exploit:
            def __reduce__(self):
                return (open, (str(sentinel), "w"))

        path = tmp_path / "evil.pt"
        path.write_bytes(pickle.dumps({"items": [_Exploit()]}))
        with pytest.raises(ValueError, match="not a valid Qwen3-TTS voice preset"):
            QwenTTSProvider.read_voice_preset_info(str(path))
        p = env.provider(voice_preset=str(path))
        with pytest.raises(ValueError, match="not a valid Qwen3-TTS voice preset"):
            p.synthesize_batch(["Hello."], b"")
        assert not sentinel.exists()

    def test_upstream_demo_format_loads(self, env, voice_bytes, tmp_path):
        a, _, _ = self._saved(env, voice_bytes, tmp_path)
        items = a._model.calls[-1]["voice_clone_prompt"]
        path = str(tmp_path / "upstream_demo.pt")
        torch.save({"items": [asdict(item) for item in items]}, path)   # qwen_tts/cli/demo.py save_prompt
        b = env.provider(voice_preset=path)
        b.synthesize_batch(["Hello."], b"")
        _assert_same_prompt(items[0], b._model.calls[-1]["voice_clone_prompt"][0])

    def test_out_of_range_codes_are_rejected(self, env, voice_bytes, tmp_path):
        _, path, _ = self._saved(env, voice_bytes, tmp_path)
        payload = torch.load(path, map_location="cpu", weights_only=True)
        payload["items"][0]["ref_code"][0, 3] = 999999
        torch.save(payload, path)
        p = env.provider(voice_preset=path)
        with pytest.raises(ValueError, match="outside the model's codebook"):
            p.load_voice_preset(path)

    def test_explicit_transcript_wins_when_saving(self, env, voice_bytes, tmp_path):
        p = env.provider(tts_options={"auto_transcribe": False})
        info = p.save_voice_preset(str(tmp_path / "v.pt"), voice_bytes, transcript="Typed by the user.")
        assert info["mode"] == "icl" and info["ref_text"] == "Typed by the user."


# ── 3. CustomVoice ────────────────────────────────────────────────────────────

class TestCustomVoice:

    def test_unknown_speaker_lists_the_valid_ones(self, env):
        p = env.provider(tts_model_name=_CUSTOM, tts_timbre="Bob")
        with pytest.raises(ValueError) as excinfo:
            p.synthesize_batch(["Hello."], b"")
        message = str(excinfo.value)
        assert "Bob" in message
        for name in ("Vivian", "Serena", "Uncle_Fu", "Dylan", "Eric", "Ryan", "Aiden", "Ono_Anna", "Sohee"):
            assert name in message
        assert p._model.calls == [], "a bad speaker must fail before generation and without retries"

    @pytest.mark.parametrize("timbre,expected", [
        ("RYAN", "ryan"), ("Uncle_Fu", "uncle_fu"), ("uncle fu", "uncle_fu"), ("ono-anna", "ono_anna"),
    ])
    def test_speaker_is_case_insensitive(self, env, timbre, expected):
        p = env.provider(tts_model_name=_CUSTOM, tts_timbre=timbre)
        p.synthesize_batch(["One.", "Two."], b"")
        assert p._model.calls[0]["speakers"] == [expected, expected]

    def test_needs_no_voice_reference(self, env):
        p = env.provider(tts_model_name=_CUSTOM, tts_timbre="Serena", voice_file="")
        assert p._needs_voice_ref() is False
        out = p.synthesize_batch(["Hello."], b"")
        assert out[0][1] > 0

    def test_instruct_is_sent_to_the_1b7_checkpoint(self, env):
        p = env.provider(tts_model_name=_CUSTOM, tts_timbre="Ryan", tts_instruct="Calm and slow.")
        p.synthesize_batch(["One.", "Two."], b"")
        assert p._model.calls[0]["instruct"] == ["Calm and slow.", "Calm and slow."]

    def test_instruct_is_withheld_from_the_0b6_checkpoint(self, env):
        p = env.provider(tts_model_name=_CUSTOM_SMALL, tts_timbre="Ryan", tts_instruct="Calm and slow.")
        p.synthesize_batch(["One."], b"")
        assert p._model.calls[0]["instruct"] is None

    def test_empty_instruct_is_not_sent(self, env):
        p = env.provider(tts_model_name=_CUSTOM, tts_timbre="Ryan", tts_instruct="  ")
        p.synthesize_batch(["One."], b"")
        assert p._model.calls[0]["instruct"] is None

    def test_default_speaker_follows_the_language(self, env):
        p = env.provider(tts_model_name=_CUSTOM, tts_timbre="", language="Japanese")
        p.synthesize_batch(["One."], b"")
        assert p._model.calls[0]["speakers"] == ["ono_anna"]

    def test_list_speakers(self, env):
        p = env.provider(tts_model_name=_CUSTOM)
        assert set(p.list_speakers()) == set(QwenTTSProvider.info().preset_voices)   # before loading
        p.ensure_ready()
        assert set(p.list_speakers()) == set(QwenTTSProvider.info().preset_voices)   # from the model
        assert env.provider(tts_model_name=_BASE).list_speakers() == []


# ── 4. VoiceDesign ────────────────────────────────────────────────────────────

class TestVoiceDesign:

    _INSTRUCT = "A calm, low-pitched male narrator in his fifties."

    def _config(self, **extra):
        config = dict(tts_model_name=_DESIGN, tts_instruct=self._INSTRUCT, language="English", seed=7)
        config.update(extra)
        return config

    def test_designs_once_then_clones_and_shares_the_clip(self, env):
        a = env.provider(**self._config())
        b = env.provider(**self._config())
        assert a._needs_voice_ref() is False

        a.synthesize_batch(["Chapter one.", "It was a dark night."], b"")
        a.synthesize_batch(["The rain fell."], b"")
        b.synthesize_batch(["Chapter two."], b"")

        design_calls = FakeQwen3TTSModel.all_calls("design")
        assert len(design_calls) == 1, "the voice is designed exactly once"
        calibration = qp._CALIBRATION_TEXTS["english"]
        assert design_calls[0]["texts"] == [calibration]
        assert design_calls[0]["instructs"] == [self._INSTRUCT]
        assert design_calls[0]["languages"] == ["English"]

        # A loads VoiceDesign then Base; B finds the clip and loads only Base.
        assert [name for name, _ in FakeQwen3TTSModel.loads] == [_DESIGN, _BASE, _BASE]
        assert a._loaded_model_name == _BASE and b._loaded_model_name == _BASE

        clone_calls = FakeQwen3TTSModel.all_calls("clone")
        assert [c["texts"] for c in clone_calls] == [
            ["Chapter one.", "It was a dark night."], ["The rain fell."], ["Chapter two."],
        ]
        prompt_a, prompt_b = a._model.prompt_calls, b._model.prompt_calls
        assert len(prompt_a) == 1 and len(prompt_b) == 1
        for call in (prompt_a[0], prompt_b[0]):
            assert call["ref_text"] == calibration and call["x_vector_only_mode"] is False
        assert prompt_a[0]["ref_audio"] == prompt_b[0]["ref_audio"], "both GPUs clone the same clip"
        assert prompt_a[0]["ref_audio"].startswith(env.cache_dir)
        _assert_same_prompt(clone_calls[0]["voice_clone_prompt"][0], clone_calls[2]["voice_clone_prompt"][0])

    def test_two_instances_warming_up_concurrently_design_once(self, env):
        FakeQwen3TTSModel.design_delay = 0.3
        providers = [env.provider(**self._config()) for _ in range(2)]
        errors: list[BaseException] = []

        def warm(provider):
            try:
                provider.ensure_ready()
            except BaseException as exc:   # noqa: BLE001 - surfaced below
                errors.append(exc)

        threads = [threading.Thread(target=warm, args=(p,)) for p in providers]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
        assert not errors
        assert len(FakeQwen3TTSModel.all_calls("design")) == 1
        assert sorted(name for name, _ in FakeQwen3TTSModel.loads) == [_BASE, _BASE, _DESIGN]

    def test_design_voice_returns_the_sample_the_book_will_clone(self, env):
        p = env.provider(**self._config())
        wav, sample_rate, text = p.design_voice()
        assert wav[:4] == b"RIFF" and sample_rate == 24000
        assert text == qp._CALIBRATION_TEXTS["english"]
        assert p._model is None, "the VoiceDesign checkpoint is released after designing"

        p.synthesize_batch(["Chapter one."], b"")
        assert len(FakeQwen3TTSModel.all_calls("design")) == 1
        with open(p._model.prompt_calls[0]["ref_audio"], "rb") as fh:
            assert fh.read() == wav

    def test_design_voice_arguments_and_force(self, env):
        p = env.provider(**self._config())
        _, _, text = p.design_voice(instruct="A bright young woman.", text="Custom sample sentence here.",
                                    language="fr")
        assert text == "Custom sample sentence here."
        call = FakeQwen3TTSModel.all_calls("design")[0]
        assert call["instructs"] == ["A bright young woman."] and call["languages"] == ["French"]
        p.design_voice(instruct="A bright young woman.", text="Custom sample sentence here.", language="fr")
        assert len(FakeQwen3TTSModel.all_calls("design")) == 1
        p.design_voice(instruct="A bright young woman.", text="Custom sample sentence here.", language="fr",
                       force=True)
        assert len(FakeQwen3TTSModel.all_calls("design")) == 2

    def test_different_instruct_or_seed_is_a_different_voice(self, env):
        p = env.provider(**self._config())
        base = p._design_spec().key
        assert p._design_spec(instruct="Someone else.").key != base
        p.config.seed = 8
        assert p._design_spec().key != base
        p.config.seed = 7
        p.config.language = "German"
        assert p._design_spec().key != base
        assert p._design_spec().text == qp._CALIBRATION_TEXTS["german"]

    def test_direct_mode_generates_every_chunk_with_voice_design(self, env):
        p = env.provider(**self._config(tts_options={"design_then_clone": False}))
        p.synthesize_batch(["One.", "Two."], b"")
        assert [name for name, _ in FakeQwen3TTSModel.loads] == [_DESIGN]
        call = p._model.calls[0]
        assert call["fn"] == "design" and call["texts"] == ["One.", "Two."]
        assert call["instructs"] == [self._INSTRUCT, self._INSTRUCT]

    def test_clone_model_option(self, env):
        p = env.provider(**self._config(tts_options={"design_clone_model": _BASE_SMALL}))
        p.synthesize_batch(["One."], b"")
        assert [name for name, _ in FakeQwen3TTSModel.loads] == [_DESIGN, _BASE_SMALL]

    def test_designed_voice_can_be_saved_and_reused_as_a_preset(self, env, tmp_path):
        a = env.provider(**self._config())
        path = str(tmp_path / "designed.pt")
        info = a.save_voice_preset(path)
        assert info["source"] == "voice_design" and info["instruct"] == self._INSTRUCT
        assert info["mode"] == "icl" and info["ref_text"] == qp._CALIBRATION_TEXTS["english"]

        FakeQwen3TTSModel.reset()
        b = env.provider(**self._config(voice_preset=path, tts_instruct="Now something else entirely."))
        b.synthesize_batch(["Chapter one."], b"")
        assert [name for name, _ in FakeQwen3TTSModel.loads] == [_BASE], "a preset needs no design pass"
        assert b._model.prompt_calls == []
        assert b._model.calls[0]["voice_clone_prompt"][0].ref_text == qp._CALIBRATION_TEXTS["english"]


# ── 5. Generation controls ────────────────────────────────────────────────────

class TestGenerationKwargs:

    def test_every_merge_argument_is_forwarded(self, env, voice_bytes):
        p = env.provider(
            voice_transcript="Ref.", temperature=0.6, top_p=0.9, top_k=40, repetition_penalty=1.1,
            tts_options={
                "subtalker_top_k": 30, "subtalker_top_p": 0.85, "subtalker_temperature": 0.7,
                "max_new_tokens": 500, "non_streaming_mode": "on",
            },
        )
        p.synthesize_batch(["A chunk."], voice_bytes)
        call = p._model.calls[0]
        assert call["raw_kwargs"] == {
            "do_sample": True, "temperature": 0.6, "top_p": 0.9, "top_k": 40,
            "repetition_penalty": 1.1, "subtalker_dosample": True, "subtalker_top_k": 30,
            "subtalker_top_p": 0.85, "subtalker_temperature": 0.7, "max_new_tokens": 500,
        }
        assert set(call["raw_kwargs"]) == set(qp._UPSTREAM_GENERATE_KWARGS)
        assert call["non_streaming_mode"] is True

    def test_only_accepted_kwargs_in_every_mode(self, env, voice_bytes):
        clone = env.provider(voice_transcript="Ref.")
        clone.synthesize_batch(["A."], voice_bytes)
        custom = env.provider(tts_model_name=_CUSTOM, tts_timbre="Ryan", tts_instruct="Slow.")
        custom.synthesize_batch(["A."], b"")
        design = env.provider(tts_model_name=_DESIGN, tts_instruct="Low voice.",
                              tts_options={"design_then_clone": False})
        design.synthesize_batch(["A."], b"")
        for call in FakeQwen3TTSModel.all_calls():
            assert set(call["raw_kwargs"]) <= set(qp._UPSTREAM_GENERATE_KWARGS)
        assert {c["fn"] for c in FakeQwen3TTSModel.all_calls()} == {"clone", "custom", "design"}

    def test_non_streaming_mode_defaults_to_upstream(self, env, voice_bytes):
        p = env.provider(voice_transcript="Ref.")
        p.synthesize_batch(["A."], voice_bytes)
        assert p._model.calls[0]["non_streaming_mode"] is False      # generate_voice_clone default
        p.config.tts_options = {"non_streaming_mode": "off"}
        custom = env.provider(tts_model_name=_CUSTOM, tts_timbre="Ryan",
                              tts_options={"non_streaming_mode": "off"})
        custom.synthesize_batch(["A."], b"")
        assert custom._model.calls[0]["non_streaming_mode"] is False  # overrides the True default

    def test_zero_temperature_means_greedy(self, env, voice_bytes):
        p = env.provider(voice_transcript="Ref.", temperature=0.0)
        p.synthesize_batch(["A."], voice_bytes)
        kwargs = p._model.calls[0]["raw_kwargs"]
        assert kwargs["do_sample"] is False
        assert "temperature" not in kwargs and "top_k" not in kwargs and "top_p" not in kwargs

    def test_unset_values_leave_the_checkpoint_default(self, env, voice_bytes):
        p = env.provider(voice_transcript="Ref.", top_k=0, repetition_penalty=0.0)
        p.synthesize_batch(["A."], voice_bytes)
        kwargs = p._model.calls[0]["raw_kwargs"]
        assert "top_k" not in kwargs and "repetition_penalty" not in kwargs

    def test_sampling_options_can_be_switched_off(self, env, voice_bytes):
        p = env.provider(voice_transcript="Ref.",
                         tts_options={"do_sample": False, "subtalker_dosample": False})
        p.synthesize_batch(["A."], voice_bytes)
        kwargs = p._model.calls[0]["raw_kwargs"]
        assert kwargs["do_sample"] is False and kwargs["subtalker_dosample"] is False
        assert "subtalker_temperature" not in kwargs

    def test_seed_makes_runs_repeatable(self, env, voice_bytes):
        p = env.provider(voice_transcript="Ref.", seed=1234)
        p.synthesize_batch(["A."], voice_bytes)
        first = torch.rand(1).item()
        p.synthesize_batch(["A."], voice_bytes)
        assert torch.rand(1).item() == first


# ── 6. Runaway guard ──────────────────────────────────────────────────────────

class TestTokenBudget:

    def test_scales_with_text_length(self):
        short = qp._token_budget(["a" * 150])
        long = qp._token_budget(["a" * 399])
        assert qp._MIN_NEW_TOKENS < short < long

    def test_cjk_needs_far_more_tokens_than_latin(self):
        latin = qp._token_budget(["a" * 300])
        assert qp._token_budget(["字" * 300]) > 2 * latin
        assert qp._token_budget(["あ" * 300]) > latin
        assert qp._token_budget(["한" * 300]) > latin

    def test_floor_and_cap(self):
        assert qp._token_budget(["Hi."]) == qp._MIN_NEW_TOKENS
        assert qp._token_budget([""]) == qp._MIN_NEW_TOKENS
        assert qp._token_budget(["字" * 100000]) == qp._MAX_NEW_TOKENS_CAP

    def test_at_least_twice_the_expected_length(self):
        for text in ("word " * 79, "字" * 399, "12345 " * 60):
            expected_frames = qp._estimate_speech_seconds(text) * qp._CODEC_FRAME_RATE_HZ
            assert qp._token_budget([text]) >= 2 * expected_frames

    def test_latin_chunk_is_far_below_the_checkpoint_default(self):
        text = ("The quick brown fox jumps over the lazy dog. " * 9)[:399]
        budget = qp._token_budget([text])
        assert budget < 2048
        # ~28 s of speech at 150 words per minute must fit comfortably.
        assert budget / qp._CODEC_FRAME_RATE_HZ > 2 * 28

    def test_batch_is_sized_for_its_longest_text(self):
        long = "a" * 399
        assert qp._token_budget(["short", long, "mid " * 20]) == qp._token_budget([long])

    def test_provider_passes_the_budget(self, env, voice_bytes):
        p = env.provider(voice_transcript="Ref.")
        texts = ["Short one.", "b" * 350]
        p.synthesize_batch(texts, voice_bytes)
        assert p._model.calls[0]["raw_kwargs"]["max_new_tokens"] == qp._token_budget(texts)

    def test_option_overrides_the_budget(self, env, voice_bytes):
        p = env.provider(voice_transcript="Ref.", tts_options={"max_new_tokens": 777})
        p.synthesize_batch(["Short one."], voice_bytes)
        assert p._model.calls[0]["raw_kwargs"]["max_new_tokens"] == 777

    def test_runaway_chunk_is_retried_alone_with_a_larger_budget(self, env, voice_bytes):
        FakeQwen3TTSModel.runaway = {"This one runs away.": 1}
        p = env.provider(voice_transcript="Ref.")
        texts = ["A normal chunk.", "This one runs away.", "Another normal chunk."]
        out = p.synthesize_batch(texts, voice_bytes)
        calls = p._model.calls
        budget = qp._token_budget(texts)
        assert [c["texts"] for c in calls] == [texts, ["This one runs away."]]
        assert calls[1]["raw_kwargs"]["max_new_tokens"] == 2 * budget
        expected = [len(t) * _SAMPLES_PER_FRAME / _SAMPLE_RATE for t in texts]
        assert [round(dur, 6) for _, dur in out] == [round(d, 6) for d in expected]

    def test_persistent_runaway_fails_the_chunk(self, env, voice_bytes):
        FakeQwen3TTSModel.runaway = {"This one runs away.": 99}
        p = env.provider(voice_transcript="Ref.")
        with pytest.raises(RuntimeError, match="runaway generation"):
            p.synthesize_batch(["A normal chunk.", "This one runs away."], voice_bytes)
        assert len(p._model.calls) == 2, "one batch pass and one retry, nothing more"
        assert p._model is not None


# ── 7. Language ───────────────────────────────────────────────────────────────

class TestLanguage:

    @pytest.mark.parametrize("configured,expected", [
        ("English", "English"), ("english", "English"), ("ENGLISH", "English"), ("en", "English"),
        ("en-US", "English"), ("zh", "Chinese"), ("zh-CN", "Chinese"), ("Japanese", "Japanese"),
        ("pt-BR", "Portuguese"), ("Deutsch", "German"), ("auto", "Auto"), ("", "Auto"),
        ("Klingon", "Auto"), ("Hindi", "Auto"),
    ])
    def test_language_is_mapped_or_falls_back_to_auto(self, env, voice_bytes, configured, expected):
        p = env.provider(voice_transcript="Ref.", language=configured)
        p.synthesize_batch(["One.", "Two."], voice_bytes)   # the fake raises on an unsupported language
        assert p._model.calls[0]["languages"] == [expected, expected]

    def test_list_languages(self, env):
        p = env.provider()
        assert p.list_languages() == list(QwenTTSProvider.info().languages)
        p.ensure_ready()
        listed = p.list_languages()
        assert listed[0] == "Auto" and set(listed) == set(QwenTTSProvider.info().languages)


# ── 8. Reference transcript ───────────────────────────────────────────────────

class TestReferenceTranscript:

    def _install_asr(self, monkeypatch, text="  The shared transcript. ", fail=False):
        created: list[dict[str, Any]] = []
        heard: list[Any] = []

        def fake_pipeline(task, **kwargs):
            assert task == "automatic-speech-recognition"
            created.append(kwargs)
            if fail:
                raise OSError("offline: cannot download the ASR model")

            def run(audio, **call_kwargs):
                heard.append(audio)
                return {"text": text}
            return run

        monkeypatch.setattr(qp, "pipeline", fake_pipeline)
        return created, heard

    def test_two_instances_share_one_transcription(self, env, voice_bytes, monkeypatch):
        created, heard = self._install_asr(monkeypatch)
        a, b = env.provider(), env.provider()
        a.synthesize_batch(["One."], voice_bytes)
        a.synthesize_batch(["Two."], voice_bytes)
        b.synthesize_batch(["Three."], voice_bytes)

        assert len(created) == 1 and len(heard) == 1, "Whisper runs once for both GPUs"
        assert created[0]["model"] == qp._DEFAULT_ASR_MODEL
        assert heard[0]["sampling_rate"] == 16000 and heard[0]["raw"].dtype == np.float32
        assert a._asr_pipe is None and b._asr_pipe is None, "the ASR model is released"
        assert a._model.prompt_calls[0]["ref_text"] == "The shared transcript."
        assert b._model.prompt_calls[0]["ref_text"] == "The shared transcript."
        assert a._model.prompt_calls[0]["x_vector_only_mode"] is False

    def test_another_process_reads_the_disk_cache(self, env, voice_bytes, monkeypatch):
        created, _ = self._install_asr(monkeypatch)
        env.provider().synthesize_batch(["One."], voice_bytes)
        qp._SHARED_TRANSCRIPTS.clear()                      # what a fresh process starts with
        monkeypatch.setattr(qp, "pipeline", _forbidden_pipeline)
        c = env.provider()
        c.synthesize_batch(["Two."], voice_bytes)
        assert len(created) == 1
        assert c._model.prompt_calls[0]["ref_text"] == "The shared transcript."

    def test_failure_is_not_retried_and_both_instances_agree(self, env, voice_bytes, monkeypatch):
        created, _ = self._install_asr(monkeypatch, fail=True)
        a, b = env.provider(), env.provider()
        for provider in (a, a, b, b):
            provider.synthesize_batch(["One."], voice_bytes)
        assert len(created) == 1, "a failed transcription is remembered, not retried per batch"
        for provider in (a, b):
            assert len(provider._model.prompt_calls) == 1
            assert provider._model.prompt_calls[0]["x_vector_only_mode"] is True

        # A fresh process within the negative TTL trusts the recorded failure too.
        qp._SHARED_TRANSCRIPTS.clear()
        env.provider().synthesize_batch(["One."], voice_bytes)
        assert len(created) == 1

    def test_configured_transcript_and_sidecar_skip_asr(self, env, voice_bytes, tmp_path):
        p = env.provider(voice_transcript="  Typed transcript.  ")
        assert p._get_voice_transcript("/nonexistent/clip.wav") == "Typed transcript."

        clip = tmp_path / "voice.wav"
        clip.write_bytes(voice_bytes)
        (tmp_path / "voice.txt").write_text("From the sidecar file.", encoding="utf-8")
        q = env.provider()
        assert q._get_voice_transcript(str(clip)) == "From the sidecar file."

    def test_asr_model_option_and_allowlist(self, env, voice_bytes, monkeypatch):
        created, _ = self._install_asr(monkeypatch)
        env.provider(tts_options={"asr_model": "openai/whisper-small"}).synthesize_batch(["One."], voice_bytes)
        assert created[0]["model"] == "openai/whisper-small"
        p = env.provider(tts_options={"asr_model": "someone/unreviewed-asr"})
        assert p._asr_model_id() == qp._DEFAULT_ASR_MODEL

    def test_auto_transcribe_off_never_loads_asr(self, env, voice_bytes):
        p = env.provider(tts_options={"auto_transcribe": False})   # env's pipeline raises if called
        assert p._get_voice_transcript("/nonexistent/clip.wav") is None
        p.synthesize_batch(["One."], voice_bytes)


# ── 9. Model lifecycle ────────────────────────────────────────────────────────

class TestLifecycle:

    def test_model_is_loaded_once_for_an_unchanged_config(self, env, voice_bytes):
        p = env.provider(voice_transcript="Ref.")
        p.ensure_ready()
        for _ in range(3):
            p.synthesize_batch(["One."], voice_bytes)
        assert len(FakeQwen3TTSModel.loads) == 1
        name, kwargs = FakeQwen3TTSModel.loads[0]
        assert name == _BASE and kwargs["device_map"] == "cpu" and kwargs["attn_implementation"] == "sdpa"
        assert "dtype" in kwargs

    def test_reloads_when_the_model_changes(self, env, voice_bytes):
        p = env.provider(voice_transcript="Ref.")
        p.synthesize_batch(["One."], voice_bytes)
        first = p._model
        p.config.tts_model_name = _BASE_SMALL
        p.synthesize_batch(["Two."], voice_bytes)
        assert [name for name, _ in FakeQwen3TTSModel.loads] == [_BASE, _BASE_SMALL]
        assert p._model is not first and p._loaded_model_name == _BASE_SMALL
        assert len(p._model.prompt_calls) == 1, "the prompt is rebuilt for the new model"

    def test_reloads_when_dtype_or_compile_changes(self, env, voice_bytes):
        p = env.provider(voice_transcript="Ref.")
        p.synthesize_batch(["One."], voice_bytes)
        p._dtype_override = "float32"
        p.synthesize_batch(["Two."], voice_bytes)
        assert len(FakeQwen3TTSModel.loads) == 2
        assert FakeQwen3TTSModel.loads[1][1]["dtype"] == torch.float32
        p.config.torch_compile = True
        p.synthesize_batch(["Three."], voice_bytes)
        assert len(FakeQwen3TTSModel.loads) == 3
        p.synthesize_batch(["Four."], voice_bytes)
        assert len(FakeQwen3TTSModel.loads) == 3

    def test_reloads_when_quantization_changes(self, env):
        p = env.provider()
        assert p._signature_for(_BASE) != (
            setattr(p.config, "quantization", "int8") or p._signature_for(_BASE)
        )

    def test_cleanup_frees_everything(self, env, voice_bytes):
        p = env.provider(voice_transcript="Ref.")
        p.synthesize_batch(["One."], voice_bytes)
        assert p.is_ready
        p.cleanup()
        assert p._model is None and p._asr_pipe is None and not p.is_ready
        assert p._loaded_model_name is None and p._loaded_signature is None
        assert not p._voice_prompt_cache and not p._transcript_cache
        p.cleanup()                                         # idempotent
        p.synthesize_batch(["Two."], voice_bytes)
        assert len(FakeQwen3TTSModel.loads) == 2

    def test_talker_pad_token_is_set_so_generate_does_not_warn_per_batch(self, env):
        p = env.provider()
        p.ensure_ready()
        assert p._model.model.talker.generation_config.pad_token_id == 2150

    def test_local_fine_tuned_checkpoint_is_accepted(self, env, tmp_path):
        checkpoint = tmp_path / "checkpoint-epoch-2"
        checkpoint.mkdir()
        (checkpoint / "config.json").write_text(json.dumps({
            "model_type": "qwen3_tts", "tts_model_type": "custom_voice", "tts_model_size": "1b7",
            "talker_config": {"spk_id": {"speaker_test": 3000}},
        }), encoding="utf-8")
        p = env.provider(tts_model_name=str(checkpoint), tts_timbre="Speaker_Test")
        assert p._mode() == "custom_voice" and p.list_speakers() == ["speaker_test"]
        p.synthesize_batch(["Hello."], b"")
        assert FakeQwen3TTSModel.loads[0][0] == str(checkpoint)
        assert p._model.calls[0]["speakers"] == ["speaker_test"]

        not_a_checkpoint = tmp_path / "random_dir"
        not_a_checkpoint.mkdir()
        (not_a_checkpoint / "config.json").write_text('{"model_type": "bert"}', encoding="utf-8")
        assert env.provider(tts_model_name=str(not_a_checkpoint))._target_model_id() == _BASE

    def test_missing_package_raises_install_hint(self, env, monkeypatch):
        monkeypatch.delitem(sys.modules, "qwen_tts")
        monkeypatch.setitem(sys.modules, "qwen_tts", None)   # makes "import qwen_tts" fail
        p = env.provider()
        with pytest.raises(RuntimeError, match="pip install"):
            p.ensure_ready()

    def test_import_does_not_leave_stdout_swapped(self, env):
        before_out, before_err = sys.stdout, sys.stderr
        env.provider().ensure_ready()
        assert sys.stdout is before_out and sys.stderr is before_err

    def test_empty_batch(self, env):
        assert env.provider().synthesize_batch([], b"") == []
        assert FakeQwen3TTSModel.loads == []


# ── 10. Batch OOM handling ────────────────────────────────────────────────────

class TestBatchFailures:

    def _texts(self, count: int) -> list[str]:
        return [f"Chunk number {i} " + "x" * (i * 3) for i in range(count)]

    def test_oom_halves_the_batch_and_keeps_the_model(self, env, voice_bytes):
        FakeQwen3TTSModel.oom_above = 2
        p = env.provider(voice_transcript="Ref.")
        p.ensure_ready()
        model = p._model
        texts = self._texts(5)
        out = p.synthesize_batch(texts, voice_bytes)

        assert [len(c["texts"]) for c in model.calls] == [5, 3, 2, 1, 2], "5 -> 3 + 2, then 3 -> 2 + 1"
        assert [c.get("failed", False) for c in model.calls] == [True, True, False, False, False]
        done = [c for c in model.calls if not c.get("failed")]
        assert [t for c in done for t in c["texts"]] == texts, "order is preserved"
        assert len(out) == 5
        expected = [len(t) * _SAMPLES_PER_FRAME / _SAMPLE_RATE for t in texts]
        assert [round(dur, 6) for _, dur in out] == [round(d, 6) for d in expected]
        assert p._model is model and len(FakeQwen3TTSModel.loads) == 1, "the model is never unloaded"
        assert len(model.prompt_calls) == 1, "the clone prompt survives the OOM"

    def test_oom_on_a_single_chunk_raises_without_unloading(self, env, voice_bytes):
        FakeQwen3TTSModel.oom_above = 0
        p = env.provider(voice_transcript="Ref.")
        with pytest.raises(RuntimeError, match="out of memory"):
            p.synthesize_batch(self._texts(2), voice_bytes)
        assert p._model is not None and len(FakeQwen3TTSModel.loads) == 1
        with pytest.raises(RuntimeError, match="out of memory"):
            p.synthesize("One chunk.", voice_bytes, return_bytes=True)
        assert p._model is not None

    def test_other_batch_failure_falls_back_to_per_item(self, env, voice_bytes):
        FakeQwen3TTSModel.fail_batches_above = 1
        p = env.provider(voice_transcript="Ref.")
        texts = self._texts(3)
        out = p.synthesize_batch(texts, voice_bytes)
        assert [c["texts"] for c in p._model.calls] == [texts] + [[t] for t in texts]
        assert len(out) == 3

    def test_single_item_failure_is_retried_once_then_raised(self, env, voice_bytes):
        FakeQwen3TTSModel.fail_batches_above = 0
        p = env.provider(voice_transcript="Ref.")
        with pytest.raises(RuntimeError, match="synthesis failed"):
            p.synthesize("One chunk.", voice_bytes, return_bytes=True)
        assert len(p._model.calls) == 2 and p._model is not None

    def test_synthesize_writes_a_file(self, env, voice_bytes, tmp_path):
        p = env.provider(voice_transcript="Ref.")
        target = str(tmp_path / "out" / "chunk.wav")
        result, duration = p.synthesize("Write me to disk.", voice_bytes, target)
        assert result == target and os.path.getsize(target) > 44 and duration > 0


# ── Shared cache plumbing ─────────────────────────────────────────────────────

class TestFileLock:

    def test_lock_is_exclusive_between_threads(self, tmp_path):
        lock_path = str(tmp_path / "locks" / "voice.lock")
        active, overlaps = [0], [0]

        def work():
            for _ in range(5):
                with qp._file_lock(lock_path):
                    active[0] += 1
                    if active[0] > 1:
                        overlaps[0] += 1
                    time.sleep(0.005)
                    active[0] -= 1

        threads = [threading.Thread(target=work) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)
        assert overlaps[0] == 0

    def test_lock_times_out(self, tmp_path):
        lock_path = str(tmp_path / "voice.lock")
        with qp._file_lock(lock_path):
            with pytest.raises(TimeoutError):
                with qp._file_lock(lock_path, timeout=0.3):
                    pass

    def test_marker_fallback_when_no_kernel_lock(self, tmp_path, monkeypatch):
        monkeypatch.setattr(qp, "_try_os_lock", lambda handle: None)
        lock_path = str(tmp_path / "voice.lock")
        with qp._file_lock(lock_path):
            assert os.path.exists(lock_path + ".excl")
            with pytest.raises(TimeoutError):
                with qp._file_lock(lock_path, timeout=0.3):
                    pass
        assert not os.path.exists(lock_path + ".excl")
