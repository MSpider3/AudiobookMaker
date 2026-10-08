"""
test_higgs_provider.py
======================
Unit tests for the Higgs Audio v3 provider.

No network, GPU or upstream package is needed: ``transformers``,
``huggingface_hub``, ``safetensors``, ``accelerate``, ``tokenizers`` and
``torchaudio`` are replaced in ``sys.modules`` by small fakes that mirror the
signatures the provider calls:

* ``huggingface_hub.snapshot_download(repo_id, *, allow_patterns=None, ...)``
* ``tokenizers.Tokenizer.from_file(path)``
* ``transformers.PreTrainedTokenizerFast(tokenizer_object=...)`` with
  ``get_added_vocab()`` and ``encode(text, add_special_tokens=False)``
* ``transformers.CONFIG_MAPPING[model_type](**text_config)`` and
  ``transformers.AutoModel.from_config(config)`` returning a module whose
  ``forward(input_ids=None, attention_mask=None, position_ids=None,
  past_key_values=None, inputs_embeds=None, use_cache=None)`` yields
  ``last_hidden_state`` / ``past_key_values`` (``Qwen3Model``)
* ``transformers.HiggsAudioV2TokenizerConfig.from_dict(d)`` and
  ``transformers.HiggsAudioV2TokenizerModel(config)`` with
  ``encode(input_values).audio_codes`` ``[B, N, T]`` and
  ``decode(audio_codes).audio_values`` ``[B, 1, L]``
* ``safetensors.safe_open(filename, framework, device)`` with ``keys()`` /
  ``get_tensor(name)``
* ``accelerate.init_empty_weights()``
* ``torchaudio.functional.resample(waveform, orig_freq, new_freq)``

The fake backbone has ``hidden = codebooks * vocab`` and the fake checkpoint an
identity audio embedding, so the hidden state *is* the logits and a fused
audio embedding is the multi-hot encoding of its codes.
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import re
import subprocess
import sys
import tempfile
import types
from dataclasses import replace

import numpy as np
import pytest
import soundfile as sf

torch = pytest.importorskip("torch")

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from audiobook_factory.pipeline import AudiobookConfig  # noqa: E402
from audiobook_factory.tts_providers import higgs_provider as hp  # noqa: E402
from audiobook_factory.tts_providers.base_tts_provider import BaseTTSProvider, get_tts_provider  # noqa: E402
from audiobook_factory.tts_providers.registry import canonical_name, provider_class, provider_info  # noqa: E402

_BOOKS = 4
_VOCAB = 12            # 10 real codes + BOC (10) + EOC (11)
_BOC, _EOC = _VOCAB - 2, _VOCAB - 1
_HIDDEN = _BOOKS * _VOCAB
_HOP = 960
_SAMPLE_RATE = 24000
_TEXT_VOCAB = 400
_PEAK = 100.0

_SPECIAL_IDS = {
    "<|tts|>": 300, "<|ref_text|>": 301, "<|ref_audio|>": 302, "<|text|>": 303, "<|audio|>": 304,
    "<|audio_end|>": 305, "<|env:music|>": 306,
}
_TAG_IDS = {tag: 320 + index for index, tag in enumerate(sorted(hp._CONTROL_TAGS))}
_SPLIT_ON_TAGS = re.compile(r"(<\|[^<>|\s]{1,48}\|>)")


# ── Fakes ─────────────────────────────────────────────────────────────────────

class FakeRawTokenizer:
    """Stands in for ``tokenizers.Tokenizer``."""

    loaded_from: list[str] = []

    @classmethod
    def from_file(cls, path):
        cls.loaded_from.append(path)
        return cls()


class FakeFastTokenizer:
    """Stands in for ``transformers.PreTrainedTokenizerFast``."""

    def __init__(self, tokenizer_object=None, **kwargs):
        assert isinstance(tokenizer_object, FakeRawTokenizer)
        self.vocab = {**_SPECIAL_IDS, **_TAG_IDS}

    def get_added_vocab(self):
        return dict(self.vocab)

    def encode(self, text, add_special_tokens=True):
        assert add_special_tokens is False
        ids = []
        for piece in _SPLIT_ON_TAGS.split(text):
            if piece in self.vocab:
                ids.append(self.vocab[piece])       # added tokens are matched in raw text
            else:
                ids.extend(word_id(word) for word in piece.split())
        return ids


def word_id(word: str) -> int:
    return 1 + sum(word.encode("utf-8")) % 250


class FakeQwen3Config:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.hidden_size = kwargs["hidden_size"]
        self.vocab_size = kwargs["vocab_size"]


class FakeBackbone(torch.nn.Module):
    """Stands in for ``transformers.Qwen3Model``.

    The hidden state at the last position holds one peaked block per codebook,
    so the code the provider must sample is known: codebook ``q`` of a row
    gets ``(step + q + prompt_tokens) % 10`` and codebook 0 switches to the
    end-of-codes id once ``frames_for(prompt_tokens)`` rows were produced.
    """

    instances: list["FakeBackbone"] = []

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_tokens = torch.nn.Embedding(config.vocab_size, config.hidden_size)
        self.norm = torch.nn.Linear(2, 2)
        self.prefills: list[dict] = []
        self.step_inputs: list[torch.Tensor] = []
        self.step_masks: list[int] = []
        self.never_end = False
        self.poison = False
        FakeBackbone.instances.append(self)

    def get_input_embeddings(self):
        return self.embed_tokens

    @staticmethod
    def frames_for(prompt_tokens: int) -> int:
        return 6 + prompt_tokens % 7

    def forward(self, input_ids=None, attention_mask=None, position_ids=None, past_key_values=None,
                inputs_embeds=None, use_cache=None, **kwargs):
        assert inputs_embeds is not None and input_ids is None and use_cache is True
        batch, length, hidden = inputs_embeds.shape
        if past_key_values is None:
            state = {"step": 0, "prompt_tokens": attention_mask.sum(dim=1).tolist(), "prefill": length}
            self.prefills.append({"inputs_embeds": inputs_embeds.clone(), "attention_mask": attention_mask.clone()})
        else:
            state = past_key_values
            state["step"] += 1
            assert length == 1
            assert attention_mask.shape[1] == state["prefill"] + state["step"]
            self.step_inputs.append(inputs_embeds.clone())
        out = torch.zeros(batch, length, hidden)
        for row in range(batch):
            tokens = int(state["prompt_tokens"][row])
            for book in range(_BOOKS):
                code = (state["step"] + book + tokens) % (_VOCAB - 2)
                if book == 0 and not self.never_end and state["step"] >= self.frames_for(tokens):
                    code = _EOC
                out[row, -1, book * _VOCAB + code] = _PEAK
        if self.poison:
            out = out * float("nan")
        return types.SimpleNamespace(last_hidden_state=out, past_key_values=state)


class FakeAutoModel:
    @staticmethod
    def from_config(config, **kwargs):
        return FakeBackbone(config)


class FakeCodecConfig:
    received: list[dict] = []

    def __init__(self, source):
        self.source = source
        self.sample_rate = source["sample_rate"]
        self.hop_length = _HOP
        self.num_quantizers = _BOOKS

    @classmethod
    def from_dict(cls, source, **kwargs):
        cls.received.append(source)
        return cls(source)


class FakeCodec(torch.nn.Module):
    """Stands in for ``transformers.HiggsAudioV2TokenizerModel``."""

    instances: list["FakeCodec"] = []

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.scale = torch.nn.Parameter(torch.ones(1))
        self.bias = torch.nn.Parameter(torch.zeros(1))
        self.encoded: list[torch.Tensor] = []
        self.decoded: list[torch.Tensor] = []
        self.mode = "ok"
        FakeCodec.instances.append(self)

    def encode(self, input_values, bandwidth=None, return_dict=None):
        assert input_values.ndim == 3 and input_values.shape[1] == 1 and input_values.dtype == torch.float32
        self.encoded.append(input_values.clone())
        frames = input_values.shape[-1] // _HOP
        index = torch.arange(frames).unsqueeze(0) + torch.arange(_BOOKS).unsqueeze(1)
        codes = (index % (_VOCAB - 2)).unsqueeze(0).repeat(input_values.shape[0], 1, 1)
        return types.SimpleNamespace(audio_codes=codes)

    def decode(self, audio_codes, return_dict=None):
        assert audio_codes.ndim == 3 and audio_codes.shape[1] == _BOOKS
        assert int(audio_codes.max()) < _VOCAB - 2 and int(audio_codes.min()) >= 0
        self.decoded.append(audio_codes.clone())
        if self.mode == "empty":
            return types.SimpleNamespace(audio_values=torch.zeros(1, 1, 0))
        level = (audio_codes.float().mean(dim=1) + 1.0) / _VOCAB
        audio = level.repeat_interleave(_HOP, dim=-1).unsqueeze(1) * 0.5 * self.scale + self.bias
        if self.mode == "nan":
            audio = audio * float("nan")
        return types.SimpleNamespace(audio_values=audio)


class FakeSafeOpen:
    """Stands in for ``safetensors.safe_open``."""

    tensors: dict[str, torch.Tensor] = {}
    opened: list[tuple[str, str, str]] = []

    def __init__(self, filename, framework, device="cpu"):
        FakeSafeOpen.opened.append((filename, framework, device))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def keys(self):
        return list(self.tensors)

    def get_tensor(self, name):
        return self.tensors[name].clone()


def make_voice(seconds: float = 2.0, sample_rate: int = _SAMPLE_RATE, pitch: float = 150.0) -> bytes:
    t = np.arange(int(seconds * sample_rate)) / sample_rate
    buffer = io.BytesIO()
    sf.write(buffer, (0.3 * np.sin(2 * np.pi * pitch * t)).astype(np.float32), sample_rate, format="WAV")
    return buffer.getvalue()


@pytest.fixture
def env(monkeypatch, tmp_path):
    """Installs the fake upstream stack and returns handles to it."""
    for cls in (FakeBackbone, FakeCodec):
        cls.instances = []
    FakeCodecConfig.received = []
    FakeRawTokenizer.loaded_from = []
    FakeSafeOpen.opened = []

    snapshot = tmp_path / "snapshot"
    snapshot.mkdir()
    (snapshot / "config.json").write_text(json.dumps({
        "model_type": "higgs_multimodal_qwen3",
        "architectures": ["HiggsMultimodalQwen3ForConditionalGeneration"],
        "audio_encoder_config": {
            "encoder_type": "discrete", "num_codebooks": _BOOKS, "vocab_size": _VOCAB,
            "out_dim": _HIDDEN, "tie_word_embeddings": True,
        },
        "text_config": {"model_type": "qwen3", "hidden_size": _HIDDEN, "vocab_size": _TEXT_VOCAB},
    }))
    (snapshot / "tokenizer.json").write_text("{}")
    (snapshot / "model.safetensors").write_bytes(b"\0")

    generator = torch.Generator().manual_seed(0)
    FakeSafeOpen.tensors = {
        "body.norm.weight": torch.randn(2, 2, generator=generator).to(torch.bfloat16),
        "body.norm.bias": torch.randn(2, generator=generator).to(torch.bfloat16),
        "tied.embedding.text_embedding.weight": torch.randn(_TEXT_VOCAB, _HIDDEN, generator=generator).to(torch.bfloat16),
        "tied.head.text_head.weight": torch.zeros(_TEXT_VOCAB, _HIDDEN, dtype=torch.bfloat16),
        hp._AUDIO_EMBEDDING_KEY: torch.eye(_HIDDEN, dtype=torch.bfloat16),
        hp._AUDIO_HEAD_KEY: torch.eye(_HIDDEN, dtype=torch.bfloat16),
        hp._CODEC_PREFIX + "scale": torch.ones(1, dtype=torch.bfloat16),
        hp._CODEC_PREFIX + "bias": torch.zeros(1, dtype=torch.bfloat16),
        hp._CODEC_PREFIX + "semantic_model.masked_spec_embed": torch.zeros(3, dtype=torch.bfloat16),
    }

    downloads: list[tuple[str, dict]] = []

    def snapshot_download(repo_id, *, revision=None, allow_patterns=None, **kwargs):
        downloads.append((repo_id, {"revision": revision, "allow_patterns": allow_patterns, **kwargs}))
        return str(snapshot)

    resampled: list[tuple[int, int]] = []

    def resample(waveform, orig_freq, new_freq):
        resampled.append((int(orig_freq), int(new_freq)))
        length = int(round(waveform.shape[-1] * new_freq / orig_freq))
        return torch.nn.functional.interpolate(waveform, size=length, mode="linear", align_corners=False)

    transformers = types.ModuleType("transformers")
    transformers.__version__ = "5.12.1"
    transformers.CONFIG_MAPPING = {"qwen3": FakeQwen3Config}
    transformers.AutoModel = FakeAutoModel
    transformers.HiggsAudioV2TokenizerConfig = FakeCodecConfig
    transformers.HiggsAudioV2TokenizerModel = FakeCodec
    transformers.PreTrainedTokenizerFast = FakeFastTokenizer
    hub = types.ModuleType("huggingface_hub")
    hub.snapshot_download = snapshot_download
    tokenizers = types.ModuleType("tokenizers")
    tokenizers.Tokenizer = FakeRawTokenizer
    safetensors = types.ModuleType("safetensors")
    safetensors.safe_open = FakeSafeOpen
    accelerate = types.ModuleType("accelerate")
    accelerate.init_empty_weights = lambda include_buffers=None: contextlib.nullcontext()
    torchaudio = types.ModuleType("torchaudio")
    torchaudio.functional = types.SimpleNamespace(resample=resample)
    for name, module in (
        ("transformers", transformers), ("huggingface_hub", hub), ("tokenizers", tokenizers),
        ("safetensors", safetensors), ("accelerate", accelerate), ("torchaudio", torchaudio),
    ):
        monkeypatch.setitem(sys.modules, name, module)
    # Voice files and the smart-voice cache go to the test's own temp dir.
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))

    def make_config(**overrides):
        base = dict(
            tts_provider_name="higgs", tts_model_name="", voice_transcript="reference words here",
            temperature=0.8, top_p=0.95, top_k=5, seed=-1,
        )
        base.update(overrides)
        return AudiobookConfig(**base)

    def make_provider(**overrides):
        return hp.HiggsAudioProvider(make_config(**overrides), device="cpu")

    return types.SimpleNamespace(
        transformers=transformers, downloads=downloads, resampled=resampled, snapshot=snapshot,
        make_config=make_config, make_provider=make_provider, voice=make_voice(), tmp_path=tmp_path,
    )


def spy_groups(provider):
    """Records every ``_generate_group`` call of *provider*."""
    calls = []
    original = provider._generate_group

    def wrapper(jobs, reference, settings, attempt=0):
        calls.append(types.SimpleNamespace(jobs=list(jobs), reference=reference, settings=settings, attempt=attempt))
        return original(jobs, reference, settings, attempt)

    provider._generate_group = wrapper
    return calls


def read_wav(data: bytes):
    return sf.read(io.BytesIO(data), dtype="float32")


# ── Static description ────────────────────────────────────────────────────────

class TestInfo:

    def test_registered_under_higgs(self):
        assert provider_class("higgs") is hp.HiggsAudioProvider
        assert canonical_name("higgs-audio-v3") == "higgs"
        assert issubclass(hp.HiggsAudioProvider, BaseTTSProvider)

    def test_info_is_complete_and_truthful(self):
        info = provider_info("higgs")
        assert info.name == "higgs"
        assert info.display_name
        assert info.default_model == "bosonai/higgs-tts-3-4b"
        assert info.default_model in info.models
        assert all(model.startswith("bosonai/") for model in info.models)
        assert info.native_sample_rate == 24000
        assert 9.0 <= info.min_vram_gb <= 16.0, "must fit a 16 GB T4 and cover ~9 GB of fp16 weights"
        assert len(info.languages) == 102 and "English" in info.languages and "Hindi" in info.languages
        assert info.supports_voice_clone is True
        assert info.transcript == "optional"
        assert info.supports_batch is True
        assert info.supports_seed is True
        assert info.supports_speed is False
        assert info.supports_instruct is False, "v3 has no natural-language style prompt"
        assert info.supports_voice_preset is True
        assert info.preset_voices == ()
        assert info.homepage.startswith("https://huggingface.co/bosonai/")
        assert any(req.startswith("transformers>=5.5") for req in info.pip_requirements)
        assert "qwen-tts" in info.install_notes and "Creator" in info.install_notes

    def test_weights_are_flagged_non_commercial(self):
        info = provider_info("higgs")
        assert info.commercial_use is False
        assert "Non-Commercial" in info.license
        assert "Creator Use Grant" in info.license

    def test_recommended_settings_are_upstreams_cloning_recipe(self):
        info = provider_info("higgs")
        # Model card / SGLang-Omni cookbook: temperature 0.8, top_k 50, top_p unset (1.0 = no filter).
        assert info.recommended_settings == {"temperature": 0.8, "top_p": 1.0, "top_k": 50}
        fields = set(AudiobookConfig.__dataclass_fields__)
        assert set(info.recommended_settings) <= fields
        assert "repetition_penalty" not in info.recommended_settings, "v3 has no repetition penalty to map"

    def test_recommended_settings_reach_the_sampler(self, env, monkeypatch):
        seen = []
        original = hp._sample_codes

        def recording(logits, temperature, top_k, top_p, generators=None):
            seen.append((temperature, top_k, top_p))
            return original(logits, temperature, top_k, top_p, generators)

        monkeypatch.setattr(hp, "_sample_codes", recording)
        provider = env.make_provider(**provider_info("higgs").recommended_settings)
        provider.synthesize("Call me Ishmael.", env.voice, return_bytes=True)
        assert seen[-1] == (0.8, None, None), "top_k 50 covers the fake's 12-code vocab; top_p 1.0 is no filter"
        assert provider._settings().top_k is None
        provider._codebook_vocab = 1026
        assert provider._settings().top_k == 50

    def test_options_are_well_formed(self):
        info = provider_info("higgs")
        keys = [option.key for option in info.options]
        assert len(keys) == len(set(keys))
        for expected in ("emotion", "style", "prosody_speed", "prosody_pitch", "expressiveness", "control_tags",
                         "use_reference_transcript", "max_new_tokens", "duration_margin", "max_batch_size", "dtype"):
            assert expected in keys
        for option in info.options:
            assert option.label and option.help
            assert option.kind in ("float", "int", "bool", "str", "choice", "file")
            if option.kind == "choice":
                assert option.default in option.choices
        emotion = next(option for option in info.options if option.key == "emotion")
        assert len(emotion.choices) == 22  # "none" + the 21 documented emotions
        assert len(hp._CONTROL_TAGS) == 43

    def test_module_imports_without_heavy_dependencies(self):
        code = (
            "import sys; sys.modules['torch'] = None; sys.modules['transformers'] = None; "
            "sys.modules['torchaudio'] = None; "
            "import audiobook_factory.tts_providers.higgs_provider as m; "
            "print(m.HiggsAudioProvider.INFO.name)"
        )
        result = subprocess.run([sys.executable, "-c", code], cwd=_ROOT, capture_output=True, text=True, timeout=120)
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "higgs"

    def test_constructor_signature_and_device(self, env):
        provider = get_tts_provider("higgs", env.make_config(), device="cuda:1", dtype_override="float16")
        assert isinstance(provider, hp.HiggsAudioProvider)
        assert provider.device == "cuda:1"
        assert provider.is_ready is False
        assert provider.estimate_cost(10_000) == 0.0
        assert provider.get_name() == "Higgs Audio v3"


# ── Pure helpers ──────────────────────────────────────────────────────────────

class TestHelpers:

    def test_delay_pattern_matches_upstream_layout(self):
        codes = torch.arange(12).view(3, 4) % 10
        delayed = hp._apply_delay_pattern(codes, _VOCAB)
        assert delayed.shape == (3 + 4 - 1, 4)
        for book in range(4):
            assert delayed[:book, book].tolist() == [_BOC] * book
            assert delayed[book:book + 3, book].tolist() == codes[:, book].tolist()
            assert delayed[book + 3:, book].tolist() == [_EOC] * (3 - book)
        assert torch.equal(hp._reverse_delay_pattern(delayed), codes)

    def test_reverse_delay_rejects_too_few_rows(self):
        with pytest.raises(RuntimeError, match="code rows"):
            hp._reverse_delay_pattern(torch.zeros(3, 4, dtype=torch.long))

    def test_prompt_layouts(self):
        specials = {"<|tts|>": 1, "<|ref_text|>": 2, "<|ref_audio|>": 3, "<|text|>": 4, "<|audio|>": 5}
        assert hp._build_prompt_ids(specials, [70, 71]) == [1, 4, 70, 71, 5]
        assert hp._build_prompt_ids(specials, [70], 3) == [1, 3, -100, -100, -100, 4, 70, 5]
        assert hp._build_prompt_ids(specials, [70], 2, [60, 61]) == [1, 2, 60, 61, 3, -100, -100, 4, 70, 5]
        # A transcript without reference audio is not part of the zero-shot prompt.
        assert hp._build_prompt_ids(specials, [70], 0, [60]) == [1, 4, 70, 5]
        # Older checkpoints have no <|ref_text|>: fall back to audio-only cloning.
        assert hp._build_prompt_ids({**specials, "<|ref_text|>": None}, [70], 1, [60]) == [1, 3, -100, 4, 70, 5]

    def test_sanitize_keeps_control_tags_and_drops_structural_tokens(self):
        text = "Hello <|audio|> there <|prosody:pause|> friend <|im_start|><|ref_audio|> <|emotion:joy|>"
        assert hp._sanitize_text(text, hp._CONTROL_TAGS) == "Hello there <|prosody:pause|> friend"
        assert hp._plain_text("<|emotion:awe|>Wow  <|sfx:sigh|>Uh") == "Wow Uh"

    def test_tag_slots(self):
        assert hp._tag_slot("<|emotion:awe|>") == "emotion"
        assert hp._tag_slot("<|style:whispering|>") == "style"
        assert hp._tag_slot("<|prosody:speed_slow|>") == "prosody:speed"
        assert hp._tag_slot("<|prosody:pitch_high|>") == "prosody:pitch"
        assert hp._tag_slot("<|prosody:pause|>") == "<|prosody:pause|>"

    def test_frame_budget_scales_with_text_length(self):
        short = hp._frame_budget("Hello there.", 25.0, 8)
        medium = hp._frame_budget("word " * 40, 25.0, 8)
        long = hp._frame_budget("word " * 80, 25.0, 8)
        assert short < medium < long
        # 399 characters at 14 chars/s is ~28.5 s; twice that plus the floor, at 25 frames/s.
        full = hp._frame_budget("x" * 399, 25.0, 8)
        assert 1400 <= full <= 1600
        assert hp._frame_budget("x" * 5000, 25.0, 8) == hp._DEFAULT_MAX_NEW_TOKENS
        assert hp._frame_budget("x" * 5000, 25.0, 8, cap=512) == 512
        assert hp._frame_budget("Hi.", 25.0, 8) >= 100, "floor keeps very short chunks from being cut off"

    def test_frame_budget_accounts_for_script_margin_and_tags(self):
        latin = hp._frame_budget("a" * 60, 25.0, 8)
        assert hp._frame_budget("漢" * 60, 25.0, 8) > latin, "CJK characters take longer each"
        assert hp._frame_budget("a" * 60, 25.0, 8, margin=3.0) > latin
        assert hp._frame_budget("<|prosody:speed_very_slow|>" + "a" * 60, 25.0, 8) > latin
        assert hp._frame_budget("a" * 30 + "<|prosody:long_pause|>" + "a" * 30, 25.0, 8) > latin
        assert hp._frame_budget("<|emotion:awe|>" + "a" * 60, 25.0, 8) == latin, "tags are not spoken"

    def test_version_parsing(self):
        assert hp._version_tuple("5.12.1") == (5, 12, 1)
        assert hp._version_tuple("5.6.0.dev0") == (5, 6, 0)
        assert hp._version_tuple("4.57.3") < hp._MIN_TRANSFORMERS <= hp._version_tuple("5.5.0")


class TestSampler:

    @staticmethod
    def peaked(codes):
        logits = torch.zeros(len(codes), _BOOKS, _VOCAB)
        for row, row_codes in enumerate(codes):
            for book, code in enumerate(row_codes):
                logits[row, book, code] = _PEAK
        return logits

    @staticmethod
    def oracle(scripted):
        """sglang-omni ``sampler.step`` (the upstream reference), driven by scripted codes."""
        delay, countdown, rows = 0, None, []
        for codes in scripted:
            codes = list(codes)
            done = False
            if delay < _BOOKS:
                for book in range(delay + 1, _BOOKS):
                    codes[book] = _BOC
                delay += 1
            elif countdown is not None:
                countdown -= 1
                done = countdown <= 0
            elif codes[0] == _EOC:
                countdown = _BOOKS - 2
            rows.append(codes)
            if done:
                break
        return torch.tensor(rows)

    def run(self, scripted_rows, temperature=0.0):
        batch = len(scripted_rows)
        delay = torch.zeros(batch, dtype=torch.long)
        countdown = torch.full((batch,), -1, dtype=torch.long)
        stopped = torch.zeros(batch, dtype=torch.bool)
        outputs = [[] for _ in range(batch)]
        for step in range(max(len(rows) for rows in scripted_rows)):
            codes = [rows[min(step, len(rows) - 1)] for rows in scripted_rows]
            sampled, delay, countdown, done_now = hp._sampler_step(
                self.peaked(codes), delay, countdown, stopped, temperature, 5, 0.9, None
            )
            for row in range(batch):
                if not bool(stopped[row]):
                    outputs[row].append(sampled[row].tolist())
            stopped = stopped | done_now
        return [torch.tensor(rows) for rows in outputs], stopped

    @pytest.mark.parametrize("frames", [4, 5, 9])
    def test_matches_upstream_state_machine(self, frames):
        scripted = [[(step + book) % 10 for book in range(_BOOKS)] for step in range(frames + _BOOKS + 3)]
        for step in range(frames, len(scripted)):
            scripted[step][0] = _EOC
        expected = self.oracle(scripted)
        (mine,), stopped = self.run([scripted])
        assert bool(stopped[0])
        assert mine.shape == expected.shape == (frames + _BOOKS - 1, _BOOKS)
        assert torch.equal(hp._reverse_delay_pattern(mine), hp._reverse_delay_pattern(expected))
        assert hp._reverse_delay_pattern(mine).shape[0] == frames
        assert int(hp._reverse_delay_pattern(mine).max()) < _VOCAB - 2, "no BOC/EOC inside decoded frames"
        # The rows are exactly a delay pattern: what the model was trained to continue.
        assert torch.equal(mine, hp._apply_delay_pattern(hp._reverse_delay_pattern(mine), _VOCAB))

    def test_rows_in_a_batch_finish_independently(self):
        def script(frames):
            rows = [[(step + book) % 10 for book in range(_BOOKS)] for step in range(20)]
            for step in range(frames, 20):
                rows[step][0] = _EOC
            return rows

        outputs, stopped = self.run([script(5), script(11)], temperature=0.8)
        assert stopped.tolist() == [True, True]
        assert [int(out.shape[0]) for out in outputs] == [5 + _BOOKS - 1, 11 + _BOOKS - 1]

    def test_impossible_codes_are_never_sampled(self):
        # Every codebook "wants" EOC from the first step and BOC is the runner-up.
        logits = torch.zeros(1, _BOOKS, _VOCAB)
        logits[..., _EOC] = _PEAK
        logits[..., _BOC] = _PEAK / 2
        delay = torch.zeros(1, dtype=torch.long)
        countdown = torch.full((1,), -1, dtype=torch.long)
        frozen = torch.zeros(1, dtype=torch.bool)
        for step in range(_BOOKS):  # delay window: codebook q starts at step q, nothing may end yet
            codes, delay, countdown, done = hp._sampler_step(logits.clone(), delay, countdown, frozen, 1.0, None, None)
            assert codes[0, step + 1:].tolist() == [_BOC] * (_BOOKS - step - 1)
            assert all(code < _VOCAB - 2 for code in codes[0, :step + 1].tolist())
            assert not bool(done[0])
        codes, delay, countdown, done = hp._sampler_step(logits.clone(), delay, countdown, frozen, 1.0, None, None)
        assert int(codes[0, 0]) == _EOC, "codebook 0 may end the stream once every codebook has started"
        assert all(code < _VOCAB - 2 for code in codes[0, 1:].tolist()), "only codebook 0 may emit EOC"
        assert int(countdown[0]) == _BOOKS - 2

    def test_sampling_filters_and_seeding(self):
        logits = torch.tensor([[[0.0, 1.0, 2.0, 3.0, 4.0]]]).repeat(2, 3, 1)
        assert hp._sample_codes(logits, 0.0, None, None).tolist() == [[4, 4, 4], [4, 4, 4]]
        draws = torch.stack([hp._sample_codes(logits, 1.0, 2, None) for _ in range(200)])
        assert set(draws.unique().tolist()) == {3, 4}, "top-k 2 keeps the two most likely codes"
        draws = torch.stack([hp._sample_codes(logits, 1.0, None, 0.5) for _ in range(200)])
        assert set(draws.unique().tolist()) == {4}, "the top code alone already covers top-p 0.5"

        def seeded(seed):
            generators = [torch.Generator().manual_seed(seed) for _ in range(2)]
            return hp._sample_codes(torch.zeros(2, 3, 50), 1.0, None, None, generators)

        first = seeded(3)
        assert torch.equal(first, seeded(3))
        assert torch.equal(first[0], first[1]), "a row's draw depends on its own seed, not on its batch mates"
        assert not torch.equal(first, seeded(4))


# ── Loading ───────────────────────────────────────────────────────────────────

class TestLoading:

    def test_loads_from_the_bundled_checkpoint(self, env):
        provider = env.make_provider()
        provider.ensure_ready()
        assert provider.is_ready
        repo_id, kwargs = env.downloads[0]
        assert repo_id == "bosonai/higgs-tts-3-4b"
        assert "model.safetensors" in kwargs["allow_patterns"] and "tokenizer.json" in kwargs["allow_patterns"]
        assert FakeRawTokenizer.loaded_from == [str(env.snapshot / "tokenizer.json")]
        assert all(framework == "pt" and device == "cpu" for _, framework, device in FakeSafeOpen.opened)
        # Backbone keys are remapped body.* -> *, text embedding -> embed_tokens.
        backbone = FakeBackbone.instances[0]
        assert torch.equal(
            backbone.embed_tokens.weight,
            FakeSafeOpen.tensors["tied.embedding.text_embedding.weight"].float(),
        )
        assert torch.equal(backbone.norm.weight, FakeSafeOpen.tensors["body.norm.weight"].float())
        assert backbone.training is False
        assert not any(parameter.requires_grad for parameter in backbone.parameters())
        # The fused audio embedding doubles as the (tied) output head.
        assert provider._audio_weight.shape == (_BOOKS * _VOCAB, _HIDDEN)
        assert provider._head_weight is provider._audio_weight
        # The codec comes from the same checkpoint, in float32, with the bundled architecture.
        codec = FakeCodec.instances[0]
        assert codec.scale.dtype == torch.float32 and codec.training is False
        assert FakeCodecConfig.received[0]["target_bandwidths"] == [0.5, 1, 1.5, 2]
        assert FakeCodecConfig.received[0]["acoustic_model_config"]["downsampling_ratios"] == [8, 5, 4, 2, 3]
        assert FakeCodecConfig.received[0] is not hp._CODEC_CONFIG, "the module constant must not be mutated"
        assert provider._frame_rate == 25.0 and provider._sample_rate == 24000
        assert provider._special_ids["<|tts|>"] == 300 and provider._special_ids["<|ref_text|>"] == 301
        assert provider._known_tags == hp._CONTROL_TAGS

    def test_foreign_model_name_falls_back_to_default(self, env):
        provider = env.make_provider(tts_model_name="Qwen/Qwen3-TTS-12Hz-1.7B-Base")
        provider.synthesize("Hello there.", env.voice, return_bytes=True)
        provider.config = env.make_config(tts_model_name="attacker/evil-remote-code")
        provider.synthesize("Hello again.", env.voice, return_bytes=True)
        assert [repo for repo, _ in env.downloads] == ["bosonai/higgs-tts-3-4b"]
        for _, kwargs in env.downloads:
            assert "trust_remote_code" not in kwargs

    def test_listed_alias_is_honoured_and_a_model_change_reloads(self, env):
        provider = env.make_provider()
        provider.ensure_ready()
        first = provider._model
        provider.ensure_ready()
        assert provider._model is first and len(env.downloads) == 1, "no reload when nothing changed"
        provider.config = env.make_config(tts_model_name="bosonai/higgs-audio-v3-tts-4b")
        provider.synthesize("Hello there.", env.voice, return_bytes=True)
        assert [repo for repo, _ in env.downloads] == ["bosonai/higgs-tts-3-4b", "bosonai/higgs-audio-v3-tts-4b"]
        assert provider._model is not first

    def test_too_old_transformers_names_the_install_command(self, env):
        env.transformers.__version__ = "4.57.3"
        provider = env.make_provider()
        with pytest.raises(RuntimeError) as error:
            provider.ensure_ready()
        message = str(error.value)
        assert hp._INSTALL_COMMAND in message
        assert "4.57.3" in message and "5.5.0" in message
        assert "qwen-tts" in message, "the user must learn the two cannot share an environment"
        assert env.downloads == [] and provider.is_ready is False

    @pytest.mark.parametrize("missing", ["transformers", "torchaudio", "accelerate", "safetensors", "huggingface_hub"])
    def test_missing_dependency_names_the_install_command(self, env, monkeypatch, missing):
        monkeypatch.setitem(sys.modules, missing, None)  # makes ``import <missing>`` raise ImportError
        provider = env.make_provider()
        with pytest.raises(RuntimeError) as error:
            provider.synthesize("Hello.", env.voice, return_bytes=True)
        assert hp._INSTALL_COMMAND in str(error.value)
        assert env.downloads == []

    def test_transformers_without_the_codec_class_is_rejected(self, env):
        del env.transformers.HiggsAudioV2TokenizerModel
        with pytest.raises(RuntimeError) as error:
            env.make_provider().ensure_ready()
        assert "HiggsAudioV2TokenizerModel" in str(error.value) and hp._INSTALL_COMMAND in str(error.value)

    def test_wrong_checkpoint_type_is_rejected(self, env):
        config = json.loads((env.snapshot / "config.json").read_text())
        config["model_type"] = "higgs_audio_v2"
        (env.snapshot / "config.json").write_text(json.dumps(config))
        with pytest.raises(RuntimeError, match="higgs_multimodal_qwen3"):
            env.make_provider().ensure_ready()

    def test_incomplete_codec_weights_are_rejected(self, env):
        del FakeSafeOpen.tensors[hp._CODEC_PREFIX + "bias"]
        with pytest.raises(RuntimeError, match="codec is missing 1 weights"):
            env.make_provider().ensure_ready()
        del FakeSafeOpen.tensors[hp._CODEC_PREFIX + "scale"]
        with pytest.raises(RuntimeError, match="bundles no audio codec"):
            env.make_provider().ensure_ready()

    def test_missing_backbone_weight_is_rejected(self, env):
        del FakeSafeOpen.tensors["body.norm.bias"]
        with pytest.raises(RuntimeError, match="failed to load"):
            env.make_provider().ensure_ready()

    def test_insufficient_vram_raises_before_loading(self, env, monkeypatch):
        gigabyte = 1024 ** 3
        monkeypatch.setattr(torch.cuda, "mem_get_info", lambda index=0: (6 * gigabyte, 15 * gigabyte), raising=False)
        monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
        monkeypatch.setattr(torch.cuda, "set_device", lambda index: None)
        provider = hp.HiggsAudioProvider(env.make_config(), device="cuda:0", dtype_override="float16")
        monkeypatch.setattr(provider, "_has_native_bf16", lambda: False)
        with pytest.raises(RuntimeError) as error:
            provider.ensure_ready()
        message = str(error.value)
        assert "cuda:0" in message and "6.0" in message and "VRAM" in message
        assert env.downloads == [], "nothing may be downloaded or allocated when the model cannot fit"
        # A free 16 GB T4 passes the same check in float16 but not in float32.
        monkeypatch.setattr(torch.cuda, "mem_get_info", lambda index=0: (int(14.5 * gigabyte), 15 * gigabyte))
        provider._check_free_vram("float16")
        with pytest.raises(RuntimeError):
            provider._check_free_vram("float32")

    def test_precision_selection(self, env, monkeypatch):
        assert env.make_provider()._requested_dtype_name() == "float32", "CPU always runs float32"
        turing = hp.HiggsAudioProvider(env.make_config(), device="cuda:0", dtype_override="bfloat16")
        monkeypatch.setattr(turing, "_has_native_bf16", lambda: False)
        assert turing._requested_dtype_name() == "float16", "a T4 would only emulate bfloat16"
        turing.config = env.make_config(tts_options={"dtype": "bfloat16"})
        assert turing._requested_dtype_name() == "bfloat16", "an explicit user choice wins"
        ampere = hp.HiggsAudioProvider(env.make_config(), device="cuda:1", dtype_override="bfloat16")
        monkeypatch.setattr(ampere, "_has_native_bf16", lambda: True)
        assert ampere._requested_dtype_name() == "bfloat16"
        forced = hp.HiggsAudioProvider(env.make_config(), device="cuda:1", dtype_override="float16")
        monkeypatch.setattr(forced, "_has_native_bf16", lambda: True)
        assert forced._requested_dtype_name() == "float16"

    def test_int8_request_is_ignored_not_fatal(self, env, caplog):
        provider = env.make_provider(quantization="int8")
        with caplog.at_level("WARNING"):
            audio, duration = provider.synthesize("Hello there.", env.voice, return_bytes=True)
        assert duration > 0
        assert any("quantization" in record.getMessage() for record in caplog.records)

    def test_cleanup_drops_everything(self, env):
        provider = env.make_provider()
        provider.synthesize("Hello there.", env.voice, return_bytes=True)
        assert provider.is_ready and provider._references
        provider.cleanup()
        assert provider.is_ready is False
        assert provider._model is None and provider._codec is None and provider._audio_weight is None
        assert provider._references == {}
        provider.synthesize("Hello again.", env.voice, return_bytes=True)
        assert provider.is_ready and len(env.downloads) == 2


# ── Prompt ────────────────────────────────────────────────────────────────────

class TestPrompt:

    def test_prompt_contains_reference_audio_transcript_and_style(self, env):
        provider = env.make_provider(tts_options={"emotion": "contemplation"})
        calls = spy_groups(provider)
        provider.synthesize("Call me Ishmael.", env.voice, return_bytes=True)
        job, reference = calls[0].jobs[0], calls[0].reference
        frames = int(2.0 * _SAMPLE_RATE) // _HOP
        rows = frames + _BOOKS - 1
        expected = (
            [300]                                                    # <|tts|>
            + [301] + [word_id(word) for word in "reference words here".split()]   # <|ref_text|> transcript
            + [302] + [-100] * rows                                  # <|ref_audio|> + one slot per delayed row
            + [303] + [_TAG_IDS["<|emotion:contemplation|>"]]        # <|text|> + style tag
            + [word_id(word) for word in "Call me Ishmael.".split()]
            + [304]                                                  # <|audio|>
        )
        assert job.prompt_ids == expected
        assert job.text == "<|emotion:contemplation|>Call me Ishmael."
        assert reference.transcript == "reference words here"
        assert tuple(reference.delayed_codes.shape) == (rows, _BOOKS)

    def test_reference_codes_are_embedded_at_the_placeholders(self, env):
        provider = env.make_provider()
        calls = spy_groups(provider)
        provider.synthesize("Call me Ishmael.", env.voice, return_bytes=True)
        job, reference = calls[0].jobs[0], calls[0].reference
        prefill = FakeBackbone.instances[0].prefills[0]
        embeds = prefill["inputs_embeds"][0]
        slots = [index for index, token in enumerate(job.prompt_ids) if token == -100]
        assert slots == list(range(slots[0], slots[0] + reference.delayed_codes.shape[0]))
        # Identity audio embedding: each slot is the multi-hot encoding of one delayed code row.
        recovered = embeds[slots].view(len(slots), _BOOKS, _VOCAB).argmax(dim=-1)
        assert torch.equal(recovered, reference.delayed_codes)
        assert torch.equal(embeds[slots].sum(dim=-1), torch.full((len(slots),), float(_BOOKS)))
        # The reference is the codec's own encoding of the clip, delay-patterned.
        raw = FakeCodec.instances[0].encode(torch.zeros(1, 1, 2 * _SAMPLE_RATE)).audio_codes[0].transpose(0, 1)
        assert torch.equal(reference.delayed_codes, hp._apply_delay_pattern(raw, _VOCAB))
        # Text positions use the backbone's own token embedding.
        table = FakeBackbone.instances[0].embed_tokens.weight
        assert torch.equal(embeds[0], table[300]) and torch.equal(embeds[-1], table[304])
        assert prefill["attention_mask"].tolist() == [[1] * len(job.prompt_ids)]

    def test_generated_codes_are_fed_back_through_the_audio_embedding(self, env):
        provider = env.make_provider()
        calls = spy_groups(provider)
        provider.synthesize("Call me Ishmael.", env.voice, return_bytes=True)
        backbone = FakeBackbone.instances[0]
        tokens = len(calls[0].jobs[0].prompt_ids)
        first = backbone.step_inputs[0][0, 0].view(_BOOKS, _VOCAB).argmax(dim=-1).tolist()
        assert first == [tokens % 10, _BOC, _BOC, _BOC], "step 0: only codebook 0 has started"
        second = backbone.step_inputs[1][0, 0].view(_BOOKS, _VOCAB).argmax(dim=-1).tolist()
        assert second == [(1 + tokens) % 10, (2 + tokens) % 10, _BOC, _BOC]

    def test_without_transcript_the_prompt_is_audio_only(self, env):
        provider = env.make_provider(voice_transcript="")
        calls = spy_groups(provider)
        provider.synthesize("Call me Ishmael.", env.voice, return_bytes=True)
        ids = calls[0].jobs[0].prompt_ids
        assert ids[:2] == [300, 302] and 301 not in ids
        provider.config = env.make_config(tts_options={"use_reference_transcript": False})
        provider.synthesize("Call me Ishmael.", env.voice, return_bytes=True)
        assert 301 not in calls[1].jobs[0].prompt_ids, "the option turns the transcript off even when one exists"

    def test_sidecar_transcript_is_used(self, env):
        clip = env.tmp_path / "narrator.wav"
        clip.write_bytes(env.voice)
        (env.tmp_path / "narrator.txt").write_text("sidecar transcript <|audio|> text")
        provider = env.make_provider(voice_transcript="", voice_file=str(clip))
        calls = spy_groups(provider)
        provider.synthesize("Call me Ishmael.", str(clip), return_bytes=True)
        assert calls[0].reference.transcript == "sidecar transcript text"
        provider.synthesize("Call me Ishmael.", env.voice, return_bytes=True)  # the pipeline passes bytes
        assert calls[1].reference.transcript == "sidecar transcript text"

    def test_structural_tokens_in_book_text_cannot_reach_the_prompt(self, env):
        provider = env.make_provider()
        calls = spy_groups(provider)
        provider.synthesize("He said <|audio|> stop <|ref_audio|> now <|prosody:pause|> ok.", env.voice, return_bytes=True)
        ids = calls[0].jobs[0].prompt_ids
        assert ids.count(304) == 1 and ids[-1] == 304
        assert ids.count(302) == 1
        assert _TAG_IDS["<|prosody:pause|>"] in ids, "documented inline tags pass through"

    def test_style_tags_from_options_and_instruct(self, env, caplog):
        provider = env.make_provider(
            tts_instruct="calm and warm <|prosody:expressive_low|> <|bogus:tag|>",
            tts_options={
                "emotion": "awe", "style": "whispering", "prosody_speed": "slow", "prosody_pitch": "low",
                "expressiveness": "none", "control_tags": "<|emotion:awe|><|sfx:sigh|>",
            },
        )
        calls = spy_groups(provider)
        with caplog.at_level("WARNING"):
            provider.synthesize("The sea was quiet.", env.voice, return_bytes=True)
        assert calls[0].settings.prefix_tags == (
            "<|emotion:awe|>", "<|style:whispering|>", "<|prosody:speed_slow|>", "<|prosody:pitch_low|>",
            "<|sfx:sigh|>", "<|prosody:expressive_low|>",
        )
        assert calls[0].jobs[0].text.endswith("<|prosody:expressive_low|>The sea was quiet.")
        messages = " ".join(record.getMessage() for record in caplog.records)
        assert "natural-language" in messages, "free-text instruct is reported as unsupported"
        assert "<|bogus:tag|>" in messages

    def test_a_tag_already_in_the_chunk_wins_over_the_default(self, env):
        provider = env.make_provider(tts_options={"emotion": "awe", "prosody_speed": "slow"})
        calls = spy_groups(provider)
        provider.synthesize("<|emotion:anger|>Get out!", env.voice, return_bytes=True)
        assert calls[0].jobs[0].text == "<|prosody:speed_slow|><|emotion:anger|>Get out!"

    def test_unspeakable_chunk_raises(self, env):
        provider = env.make_provider()
        for text in ("", "   ", "* * *", "<|prosody:pause|>"):
            with pytest.raises(RuntimeError, match="speakable"):
                provider.synthesize(text, env.voice, return_bytes=True)


# ── Generation ────────────────────────────────────────────────────────────────

class TestGeneration:

    def expected_seconds(self, provider, text, reference):
        job = provider._prepare_jobs([text], reference, provider._settings())[0]
        return FakeBackbone.frames_for(len(job.prompt_ids)) * _HOP / _SAMPLE_RATE

    def test_single_chunk_returns_valid_audio(self, env, tmp_path):
        provider = env.make_provider()
        calls = spy_groups(provider)
        audio, duration = provider.synthesize("Call me Ishmael.", env.voice, return_bytes=True)
        samples, rate = read_wav(audio)
        assert rate == 24000 and samples.ndim == 1
        assert duration == pytest.approx(len(samples) / rate)
        assert duration == pytest.approx(self.expected_seconds(provider, "Call me Ishmael.", calls[0].reference))
        assert np.isfinite(samples).all() and np.abs(samples).max() > 0.01
        out_path = str(tmp_path / "out" / "chunk.wav")
        written, duration_again = provider.synthesize("Call me Ishmael.", env.voice, out_path)
        assert written == out_path and os.path.getsize(out_path) > 1000
        assert duration_again == pytest.approx(duration)

    def test_decoded_frames_are_exactly_what_the_model_emitted(self, env):
        provider = env.make_provider()
        calls = spy_groups(provider)
        provider.synthesize("Call me Ishmael.", env.voice, return_bytes=True)
        tokens = len(calls[0].jobs[0].prompt_ids)
        frames = FakeBackbone.frames_for(tokens)
        decoded = FakeCodec.instances[0].decoded[0]
        assert tuple(decoded.shape) == (1, _BOOKS, frames)
        # Codebook q at frame t was produced at step t + q: (step + q + tokens) % 10.
        expected = torch.tensor([[(t + 2 * q + tokens) % 10 for t in range(frames)] for q in range(_BOOKS)])
        assert torch.equal(decoded[0], expected)

    def test_batch_keeps_input_order_and_count(self, env):
        provider = env.make_provider()
        texts = [
            "One.",
            "A much longer sentence that certainly needs many more words than the others do.",
            "Two words.",
            "Something of a middling length here.",
            "Three more words.",
        ]
        calls = spy_groups(provider)
        results = provider.synthesize_batch(texts, env.voice)
        assert len(results) == len(texts)
        assert all(isinstance(audio, bytes) and duration > 0 for audio, duration in results)
        assert len(calls) == 1 and len(calls[0].jobs) == 5, "one real batched forward pass"
        budgets = [job.budget for job in calls[0].jobs]
        assert budgets == sorted(budgets), "chunks are grouped by length"
        reference = calls[0].reference
        expected = [self.expected_seconds(provider, text, reference) for text in texts]
        assert len(set(expected)) > 1, "the fixture must give different chunks different lengths"
        assert [duration for _, duration in results] == pytest.approx(expected)
        singles = [provider.synthesize(text, env.voice, return_bytes=True)[1] for text in texts]
        assert singles == pytest.approx(expected), "a chunk sounds the same alone and in a batch"
        assert provider.synthesize_batch([], env.voice) == []

    def test_batched_prompts_are_left_padded(self, env):
        provider = env.make_provider()
        calls = spy_groups(provider)
        provider.synthesize_batch(["One.", "A much longer sentence with several more words."], env.voice)
        prefill = FakeBackbone.instances[0].prefills[0]
        lengths = [len(job.prompt_ids) for job in calls[0].jobs]
        longest = max(lengths)
        assert prefill["inputs_embeds"].shape[:2] == (2, longest)
        for row, length in enumerate(lengths):
            assert prefill["attention_mask"][row].tolist() == [0] * (longest - length) + [1] * length
            assert not prefill["inputs_embeds"][row, :longest - length].any()
            assert torch.equal(prefill["inputs_embeds"][row, -1], FakeBackbone.instances[0].embed_tokens.weight[304])

    def test_large_batches_are_split(self, env):
        provider = env.make_provider(tts_options={"max_batch_size": 2})
        calls = spy_groups(provider)
        results = provider.synthesize_batch([f"Sentence number {index} here." for index in range(5)], env.voice)
        assert len(results) == 5
        assert [len(call.jobs) for call in calls] == [2, 2, 1]

    def test_reference_is_encoded_once_across_chunks(self, env):
        provider = env.make_provider()
        calls = spy_groups(provider)
        provider.synthesize("First chunk.", env.voice, return_bytes=True)
        provider.synthesize_batch(["Second chunk.", "Third chunk.", "Fourth chunk."], env.voice)
        provider.synthesize("Fifth chunk.", env.voice, return_bytes=True)
        codec = FakeCodec.instances[0]
        assert len(codec.encoded) == 1
        assert len({id(call.reference) for call in calls}) == 1, "every chunk gets the very same reference object"
        assert all(job.prompt_ids.count(-100) == calls[0].reference.delayed_codes.shape[0]
                   for call in calls for job in call.jobs)
        # Same content under another path / as a path instead of bytes: still one encode.
        clip = env.tmp_path / "copy.wav"
        clip.write_bytes(env.voice)
        provider.synthesize("Sixth chunk.", str(clip), return_bytes=True)
        assert len(codec.encoded) == 1
        # A different clip, or a different transcript, is a different reference.
        provider.synthesize("Seventh chunk.", make_voice(pitch=220.0), return_bytes=True)
        assert len(codec.encoded) == 2
        provider.config = env.make_config(voice_transcript="other words")
        provider.synthesize("Eighth chunk.", env.voice, return_bytes=True)
        assert len(codec.encoded) == 3

    def test_reference_clip_is_resampled_and_validated(self, env):
        provider = env.make_provider()
        provider.synthesize("Hello there.", make_voice(seconds=2.0, sample_rate=16000), return_bytes=True)
        assert env.resampled == [(16000, 24000)]
        assert FakeCodec.instances[0].encoded[0].shape == (1, 1, 2 * _SAMPLE_RATE)
        with pytest.raises(RuntimeError, match="100 s"):
            provider.synthesize("Hello there.", make_voice(seconds=101.0, sample_rate=8000), return_bytes=True)
        with pytest.raises(RuntimeError, match="bytes"):
            provider.synthesize("Hello there.", b"RIFF", return_bytes=True)
        with pytest.raises(RuntimeError, match="not found"):
            provider.synthesize("Hello there.", str(env.tmp_path / "missing.wav"), return_bytes=True)
        with pytest.raises(RuntimeError, match="could not read"):
            provider.synthesize("Hello there.", b"not a wav file at all " * 20, return_bytes=True)

    def test_sampling_settings_come_from_config_at_call_time(self, env, monkeypatch):
        seen = []
        original = hp._sample_codes

        def recording(logits, temperature, top_k, top_p, generators=None):
            seen.append((temperature, top_k, top_p, generators))
            return original(logits, temperature, top_k, top_p, generators)

        monkeypatch.setattr(hp, "_sample_codes", recording)
        provider = env.make_provider(temperature=0.65, top_k=7, top_p=0.9, seed=1234)
        provider.synthesize("Call me Ishmael.", env.voice, return_bytes=True)
        temperature, top_k, top_p, generators = seen[-1]
        assert (temperature, top_k, top_p) == (0.65, 7, 0.9)
        assert len(generators) == 1
        assert generators[0].initial_seed() == 1234

        # The pool swaps provider.config between runs: the next call must see the new values.
        provider.config = env.make_config(temperature=1.0, top_k=0, top_p=1.0, seed=-1)
        provider.synthesize("Call me Ishmael.", env.voice, return_bytes=True)
        assert seen[-1] == (1.0, None, None, None), "0 / 1.0 switch the filters off; no seed means no generator"

    def test_provider_options_shape_the_generation(self, env):
        provider = env.make_provider(tts_options={"max_new_tokens": "300", "duration_margin": "3.5", "max_batch_size": 3})
        calls = spy_groups(provider)
        provider.synthesize("word " * 79, env.voice, return_bytes=True)
        settings = calls[0].settings
        assert (settings.max_new_tokens, settings.duration_margin, settings.max_batch_size) == (300, 3.5, 3)
        assert calls[0].jobs[0].budget == 300, "the cap from tts_options bounds the frame budget"
        provider.config = env.make_config()
        provider.synthesize("word " * 79, env.voice, return_bytes=True)
        default = calls[1].jobs[0].budget
        assert default > 300 and calls[1].settings.max_new_tokens == hp._DEFAULT_MAX_NEW_TOKENS
        provider.synthesize("word " * 10, env.voice, return_bytes=True)
        assert calls[2].jobs[0].budget < default, "the budget follows the text length"

    def test_runaway_generation_is_bounded_then_raises(self, env):
        provider = env.make_provider(seed=5)
        provider.ensure_ready()
        FakeBackbone.instances[0].never_end = True
        calls = spy_groups(provider)
        with pytest.raises(RuntimeError) as error:
            provider.synthesize("Hello there.", env.voice, return_bytes=True)
        assert "max_new_tokens" in str(error.value)
        assert [call.attempt for call in calls] == [0, 1, 2], "re-sampled with a shifted seed before giving up"
        budget = calls[0].jobs[0].budget
        assert budget < 300
        assert len(FakeBackbone.instances[0].step_inputs) == 3 * (budget - 1), "each attempt stops at the budget"
        assert FakeCodec.instances[0].decoded == [], "no audio is fabricated for an unfinished chunk"

    def test_a_retry_can_rescue_an_unfinished_chunk(self, env):
        provider = env.make_provider()
        provider.ensure_ready()
        backbone = FakeBackbone.instances[0]
        original = provider._generate_group

        def flaky(jobs, reference, settings, attempt=0):
            backbone.never_end = attempt == 0
            return original(jobs, reference, settings, attempt)

        provider._generate_group = flaky
        results = provider.synthesize_batch(["Hello there.", "Another chunk of text."], env.voice)
        assert len(results) == 2 and all(duration > 0 for _, duration in results)

    def test_cuda_oom_falls_back_to_one_chunk_at_a_time(self, env, monkeypatch):
        class FakeOutOfMemory(RuntimeError):
            pass

        monkeypatch.setattr(torch.cuda, "OutOfMemoryError", FakeOutOfMemory, raising=False)
        provider = env.make_provider()
        provider.ensure_ready()
        emptied = []
        monkeypatch.setattr(provider, "_release_cuda_memory", lambda: emptied.append(True))
        original = provider._generate_group
        sizes = []

        def limited(jobs, reference, settings, attempt=0):
            sizes.append(len(jobs))
            if len(jobs) > 1:
                raise FakeOutOfMemory("CUDA out of memory. Tried to allocate 2.00 GiB")
            return original(jobs, reference, settings, attempt)

        provider._generate_group = limited
        texts = ["One.", "A much longer sentence with several more words.", "Two words."]
        results = provider.synthesize_batch(texts, env.voice)
        assert sizes == [3, 1, 1, 1]
        assert emptied, "the CUDA cache is emptied before retrying"
        reference = next(iter(provider._references.values()))
        expected = [TestGeneration().expected_seconds(provider, text, reference) for text in texts]
        assert [duration for _, duration in results] == pytest.approx(expected)

        def always(jobs, reference, settings, attempt=0):
            raise FakeOutOfMemory("CUDA out of memory.")

        provider._generate_group = always
        with pytest.raises(RuntimeError, match="out of VRAM"):
            provider.synthesize_batch(texts, env.voice)

    def test_empty_audio_raises(self, env):
        provider = env.make_provider()
        provider.ensure_ready()
        FakeCodec.instances[0].mode = "empty"
        with pytest.raises(RuntimeError, match="no audio"):
            provider.synthesize("Hello there.", env.voice, return_bytes=True)
        with pytest.raises(RuntimeError, match="no audio"):
            provider.synthesize_batch(["Hello there.", "Again."], env.voice)

    def test_nan_audio_raises(self, env):
        provider = env.make_provider()
        provider.ensure_ready()
        FakeCodec.instances[0].mode = "nan"
        with pytest.raises(RuntimeError, match="NaN"):
            provider.synthesize("Hello there.", env.voice, return_bytes=True)

    def test_nan_logits_raise_instead_of_producing_noise(self, env):
        provider = env.make_provider()
        provider.ensure_ready()
        FakeBackbone.instances[0].poison = True
        with pytest.raises(RuntimeError, match="NaN/inf logits"):
            provider.synthesize("Hello there.", env.voice, return_bytes=True)
        assert FakeCodec.instances[0].decoded == []

    def test_float16_overflow_reloads_in_bfloat16(self, env, monkeypatch):
        provider = env.make_provider()
        precision = {"name": "float16"}
        monkeypatch.setattr(
            provider, "_requested_dtype_name",
            lambda: provider._dtype_fallback or precision["name"],
        )
        monkeypatch.setattr(torch, "float16", torch.float32)   # CPU stand-ins for the half precisions
        monkeypatch.setattr(torch, "bfloat16", torch.float32)
        provider.ensure_ready()
        assert provider._loaded_signature == ("bosonai/higgs-tts-3-4b", "float16")
        FakeBackbone.instances[0].poison = True
        audio, duration = provider.synthesize("Hello there.", env.voice, return_bytes=True)
        assert duration > 0
        assert provider._loaded_signature == ("bosonai/higgs-tts-3-4b", "bfloat16")
        assert len(FakeBackbone.instances) == 2 and len(env.downloads) == 2


# ── Voices without a clip ─────────────────────────────────────────────────────

class TestVoices:

    def test_voice_preset_round_trip(self, env):
        provider = env.make_provider()
        saved = provider.save_voice_preset(str(env.tmp_path / "narrator"), env.voice)
        path = saved["path"]
        assert path.endswith(".pt") and os.path.isfile(path)
        frames = 2 * _SAMPLE_RATE // _HOP
        assert saved == {
            "path": path, "provider": "higgs", "format": hp._PRESET_FORMAT, "model": "bosonai/higgs-tts-3-4b",
            "codebooks": _BOOKS, "frames": frames, "seconds": 2.0, "transcript": "reference words here",
            "has_transcript": True,
        }
        assert json.loads(json.dumps(saved)) == saved, "the description must be JSON-safe"
        payload = torch.load(path, map_location="cpu", weights_only=True)  # no pickled objects inside
        assert payload["format"] == hp._PRESET_FORMAT
        assert payload["reference_text"] == "reference words here"
        assert tuple(payload["reference_codes"].shape) == (frames, _BOOKS)
        assert provider.load_voice_preset(path) == saved

        fresh = env.make_provider(voice_preset=path, voice_transcript="ignored for presets")
        with pytest.raises(ValueError, match="expected"):
            fresh.load_voice_preset(path)  # an unloaded instance assumes the real 8-codebook geometry
        calls = spy_groups(fresh)
        fresh.synthesize_batch(["First chunk.", "Second chunk."], b"")
        fresh.synthesize("Third chunk.", env.voice, return_bytes=True)
        assert FakeCodec.instances[-1].encoded == [], "a preset needs no encoding at all"
        original = next(iter(provider._references.values()))
        assert all(torch.equal(call.reference.delayed_codes, original.delayed_codes) for call in calls)
        assert all(call.reference.transcript == "reference words here" for call in calls)

    def test_save_voice_preset_signature_matches_the_base_class(self, env):
        import inspect

        for name in ("save_voice_preset", "load_voice_preset"):
            mine = inspect.signature(getattr(hp.HiggsAudioProvider, name))
            base = inspect.signature(getattr(BaseTTSProvider, name))
            assert [(p.name, p.kind, p.default) for p in mine.parameters.values()] == \
                [(p.name, p.kind, p.default) for p in base.parameters.values()]

    def test_save_voice_preset_transcript_and_defaults(self, env):
        clip = env.tmp_path / "narrator.wav"
        clip.write_bytes(env.voice)
        provider = env.make_provider(voice_file=str(clip), voice_transcript="configured words")
        default = provider.save_voice_preset(str(env.tmp_path / "default.pt"))
        assert default["transcript"] == "configured words", "voice_ref and transcript default to the config"
        explicit = provider.save_voice_preset(
            str(env.tmp_path / "explicit.pt"), str(clip), transcript="explicit <|audio|> words"
        )
        assert explicit["transcript"] == "explicit words"
        silent = provider.save_voice_preset(str(env.tmp_path / "silent.pt"), env.voice, transcript="")
        assert silent["transcript"] == "" and silent["has_transcript"] is False
        assert len(FakeCodec.instances[0].encoded) == 3, "one encode per (clip, transcript)"

        fresh = env.make_provider(voice_preset=explicit["path"], voice_transcript="configured words")
        calls = spy_groups(fresh)
        fresh.synthesize("Hello there.", b"", return_bytes=True)
        ids = calls[0].jobs[0].prompt_ids
        assert ids[1:4] == [301, word_id("explicit"), word_id("words")], "the preset's own transcript is used"
        nobody = env.make_provider(voice_file="")
        with pytest.raises(RuntimeError, match="needs a reference clip"):
            nobody.save_voice_preset(str(env.tmp_path / "none.pt"))

    def test_load_voice_preset_does_not_need_the_model(self, env):
        provider = env.make_provider()
        saved = provider.save_voice_preset(str(env.tmp_path / "narrator.pt"), env.voice)
        fresh = env.make_provider()
        fresh._num_codebooks, fresh._codebook_vocab = _BOOKS, _VOCAB   # geometry of the fake checkpoint
        described = fresh.load_voice_preset(saved["path"])
        assert described == saved
        assert fresh.is_ready is False and len(env.downloads) == 1, "describing a preset loads no weights"
        assert len(fresh._references) == 1

    def test_load_voice_preset_rejects_incompatible_files(self, env):
        provider = env.make_provider()
        provider.ensure_ready()
        with pytest.raises(ValueError, match="not found"):
            provider.load_voice_preset(str(env.tmp_path / "missing.pt"))
        foreign = env.tmp_path / "foreign.pt"
        torch.save({"something": torch.zeros(2)}, foreign)
        with pytest.raises(ValueError, match="not a Higgs Audio v3 voice preset"):
            provider.load_voice_preset(str(foreign))
        garbage = env.tmp_path / "garbage.pt"
        garbage.write_bytes(b"this is not a torch file")
        with pytest.raises(ValueError, match="could not read"):
            provider.load_voice_preset(str(garbage))
        wrong_shape = env.tmp_path / "wrong.pt"
        torch.save({"format": hp._PRESET_FORMAT, "reference_codes": torch.zeros(5, _BOOKS + 1, dtype=torch.int16)}, wrong_shape)
        with pytest.raises(ValueError, match="expected"):
            provider.load_voice_preset(str(wrong_shape))
        out_of_range = env.tmp_path / "range.pt"
        torch.save({"format": hp._PRESET_FORMAT, "reference_codes": torch.full((5, _BOOKS), _EOC, dtype=torch.int16)}, out_of_range)
        with pytest.raises(ValueError, match="outside"):
            provider.load_voice_preset(str(out_of_range))

    def test_preset_with_pickled_object_is_refused(self, env):
        class Payload:
            def __reduce__(self):
                return (os.getcwd, ())

        path = env.tmp_path / "evil.pt"
        torch.save({"format": hp._PRESET_FORMAT, "reference_codes": Payload()}, path)
        provider = env.make_provider(voice_preset=str(path))
        with pytest.raises(ValueError, match="could not read"):
            provider.load_voice_preset(str(path))
        with pytest.raises(ValueError, match="could not read"):
            provider.synthesize("Hello there.", env.voice, return_bytes=True)

    def test_voice_preset_can_be_a_plain_clip(self, env):
        clip = env.tmp_path / "preset_clip.wav"
        clip.write_bytes(make_voice(seconds=3.0))
        provider = env.make_provider(voice_preset=str(clip))
        calls = spy_groups(provider)
        provider.synthesize("Hello there.", env.voice, return_bytes=True)
        assert calls[0].reference.delayed_codes.shape[0] == 3 * _SAMPLE_RATE // _HOP + _BOOKS - 1
        provider.config = env.make_config(voice_preset=str(env.tmp_path / "nope.pt"))
        with pytest.raises(RuntimeError, match="preset not found"):
            provider.synthesize("Hello there.", env.voice, return_bytes=True)

    def test_foreign_preset_file_is_rejected(self, env):
        path = env.tmp_path / "other.pt"
        torch.save({"something": torch.zeros(2)}, path)
        provider = env.make_provider(voice_preset=str(path))
        with pytest.raises(ValueError, match="not a Higgs Audio v3 voice preset"):
            provider.synthesize("Hello there.", env.voice, return_bytes=True)

    def test_smart_voice_is_invented_once_and_shared(self, env):
        first = env.make_provider(seed=42)
        calls = spy_groups(first)
        results = first.synthesize_batch(["The first chunk of the book.", "The second chunk."], b"")
        assert len(results) == 2
        assert calls[0].reference is None and len(calls[0].jobs) == 1, "one zero-shot utterance bootstraps the voice"
        assert calls[0].jobs[0].prompt_ids[:2] == [300, 303], "zero-shot prompt: <|tts|> <|text|> ..."
        voice = calls[1].reference
        assert voice is not None and voice.transcript == "The first chunk of the book."
        assert all(call.reference is voice for call in calls[1:])
        assert all(302 in job.prompt_ids for job in calls[1].jobs), "the book itself is cloned from that voice"
        cache = [name for name in os.listdir(env.tmp_path / hp._VOICE_CACHE_DIR_NAME) if name.startswith("higgs_smart_")]
        assert len(cache) == 1

        first.synthesize("A later chunk.", b"", return_bytes=True)
        assert calls[-1].reference is voice and sum(call.reference is None for call in calls) == 1

        # A second GPU instance (or a resumed run) loads the cached voice instead of inventing another.
        second = env.make_provider(seed=42)
        second_calls = spy_groups(second)
        second.synthesize("Chunk on the other GPU.", b"", return_bytes=True)
        assert all(call.reference is not None for call in second_calls)
        assert torch.equal(second_calls[0].reference.delayed_codes, voice.delayed_codes)
        assert second_calls[0].reference.transcript == voice.transcript

        # Another seed is another narrator.
        third = env.make_provider(seed=43)
        third_calls = spy_groups(third)
        third.synthesize("Chunk with another seed.", b"", return_bytes=True)
        assert third_calls[0].reference is None
