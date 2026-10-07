"""
test_fish_provider.py
=====================
Unit tests for the Fish Audio S2 Pro provider.

No network, no GPU and no ``fish_speech`` package: a fake upstream package is
injected into ``sys.modules``. Its functions mirror the signatures of
fish-speech at commit 214da3cd841bda85da2496b96cd3c4d7edb1337e
(``fish_speech/models/text2semantic/inference.py`` and ``llama.py``), so a
keyword the real code would reject is rejected here too.
"""

from __future__ import annotations

import inspect
import io
import json
import math
import os
import subprocess
import sys
import tempfile
import types
from dataclasses import dataclass, field
from typing import Any, Optional

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

torch = pytest.importorskip("torch")
np = pytest.importorskip("numpy")
sf = pytest.importorskip("soundfile")

from audiobook_factory.pipeline import AudiobookConfig  # noqa: E402
from audiobook_factory.tts_providers import fish_provider  # noqa: E402
from audiobook_factory.tts_providers.base_tts_provider import (  # noqa: E402
    BaseTTSProvider,
    get_tts_provider,
)
from audiobook_factory.tts_providers.fish_provider import FishSpeechProvider  # noqa: E402
from audiobook_factory.tts_providers.registry import (  # noqa: E402
    apply_recommended_settings,
    canonical_name,
    provider_class,
)

_SAMPLE_RATE = 44100
_FRAME_LENGTH = 2048
_NUM_CODEBOOKS = 10
_TRANSCRIPT = "This is the narrator reference."

# Real S2 Pro text_config values (fishaudio/s2-pro config.json).
_S2_CONFIG = {
    "model_type": "fish_qwen3_omni",
    "text_config": {
        "n_layer": 36, "n_head": 32, "n_local_heads": 8, "head_dim": 128,
        "dim": 2560, "max_seq_len": 32768, "vocab_size": 155776,
    },
    "audio_decoder_config": {"num_codebooks": 10, "vocab_size": 4096, "n_layer": 4},
}


# ── Fake upstream package ────────────────────────────────────────────────────


@dataclass
class _Recorder:
    """Everything the fake upstream saw, plus switches for failure modes."""

    mode: str = "normal"  # normal | runaway | silent | nan | empty_audio | oom_once
    generate_calls: list[dict] = field(default_factory=list)
    codec_loads: list[tuple] = field(default_factory=list)
    model_loads: list[dict] = field(default_factory=list)
    setup_caches: list[dict] = field(default_factory=list)
    encode_calls: int = 0
    decode_calls: int = 0
    snapshot_ids: list[str] = field(default_factory=list)
    init_skipped: list[bool] = field(default_factory=list)
    oom_raised: bool = False
    toggle_sdp: bool = False


@dataclass
class _GenerateResponse:
    action: str
    codes: Optional[Any] = None
    text: Optional[str] = None


class WindowLimitedTransformer(torch.nn.Module):
    """Same class name as upstream so the mask trimming finds it."""

    def __init__(self) -> None:
        super().__init__()
        self.register_buffer(
            "causal_mask", torch.tril(torch.ones(8, 8, dtype=torch.bool)), persistent=False
        )


class _FakeCodec(torch.nn.Module):
    sample_rate = _SAMPLE_RATE

    def __init__(self, recorder: _Recorder) -> None:
        super().__init__()
        self.recorder = recorder
        self.weight = torch.nn.Parameter(torch.zeros(2))
        self.projection = torch.nn.Linear(2, 2)
        self.pre_module = WindowLimitedTransformer()

    # modded_dac.DAC.encode(audio_data, audio_lengths=None, n_quantizers=None, **kwargs)
    def encode(self, audio_data, audio_lengths=None, n_quantizers=None, **kwargs):
        assert audio_data.ndim == 3 and audio_data.shape[:2] == (1, 1)
        assert audio_data.dtype == self.weight.dtype
        # A real forward through the weights: it raises when they were
        # converted outside the inference mode they were built in.
        self.projection(audio_data.new_zeros(1, 2))
        self.recorder.encode_calls += 1
        frames = int(math.ceil(int(audio_lengths[0]) / _FRAME_LENGTH))
        indices = torch.arange(frames).repeat(1, _NUM_CODEBOOKS, 1) % 1024
        return indices, torch.tensor([frames])


class _FakeModel(torch.nn.Module):
    def __init__(self, recorder: _Recorder, max_seq_len: int) -> None:
        super().__init__()
        self.recorder = recorder
        self.weight = torch.nn.Parameter(torch.zeros(2))
        self.config = types.SimpleNamespace(
            max_seq_len=max_seq_len, num_codebooks=_NUM_CODEBOOKS, codebook_size=4096
        )
        self.tokenizer = object()

    # llama.BaseTransformer.setup_caches(max_batch_size, max_seq_len, dtype=torch.bfloat16)
    def setup_caches(self, max_batch_size: int, max_seq_len: int, dtype=torch.bfloat16):
        self.recorder.setup_caches.append(
            {"max_batch_size": max_batch_size, "max_seq_len": max_seq_len, "dtype": dtype}
        )


def _build_fake_upstream(recorder: _Recorder) -> dict[str, types.ModuleType]:
    inference = types.ModuleType("fish_speech.models.text2semantic.inference")
    llama = types.ModuleType("fish_speech.models.text2semantic.llama")

    class Conversation:
        def visualize(self, tokenizer, **kwargs):  # prints the prompt upstream
            raise AssertionError("visualize must be silenced")

    def tqdm(iterable=None, *args, **kwargs):
        raise AssertionError("tqdm must be silenced")

    # inference.py:97
    def decode_one_token_ar(
        model, x, input_pos, temperature, top_p, top_k, semantic_logit_bias,
        audio_masks, audio_parts, previous_tokens=None, kv_len=None,
    ):
        raise AssertionError("not called by the fake generate_long")

    # inference.py:417-418 - upstream builds the codec under inference_mode,
    # so its weights are inference tensors.
    @torch.inference_mode()
    def load_codec_model(codec_checkpoint_path, device, precision=torch.bfloat16):
        recorder.codec_loads.append((str(codec_checkpoint_path), device, precision))
        return _FakeCodec(recorder).to(device=device, dtype=precision)

    # inference.py:461-462
    @torch.inference_mode()
    def decode_to_audio(codes, codec):
        recorder.decode_calls += 1
        assert codes.ndim == 2 and codes.shape[0] == _NUM_CODEBOOKS
        if recorder.mode == "empty_audio":
            return torch.zeros(0)
        samples = codes.shape[1] * _FRAME_LENGTH
        audio = torch.linspace(-0.5, 0.5, samples)
        if recorder.mode == "nan":
            audio[0] = float("nan")
        return audio

    # inference.py:545 - keyword-only, exactly these names
    def generate_long(
        *,
        model,
        device,
        decode_one_token,
        text: str,
        num_samples: int = 1,
        max_new_tokens: int = 0,
        top_p: float = 0.9,
        top_k: int = 30,
        repetition_penalty: float = 1.1,
        temperature: float = 1.0,
        compile: bool = False,
        iterative_prompt: bool = True,
        chunk_length: int = 512,
        prompt_text=None,
        prompt_tokens=None,
    ):
        assert 0 < top_p <= 1, "top_p must be in (0, 1]"
        assert 0 < temperature < 2, "temperature must be in (0, 2)"
        use_prompt = bool(prompt_text) and bool(prompt_tokens)  # raises on a bare tensor
        recorder.generate_calls.append(
            dict(
                model=model, device=device, decode_one_token=decode_one_token, text=text,
                num_samples=num_samples, max_new_tokens=max_new_tokens, top_p=top_p,
                top_k=top_k, repetition_penalty=repetition_penalty, temperature=temperature,
                compile=compile, iterative_prompt=iterative_prompt, chunk_length=chunk_length,
                prompt_text=prompt_text, prompt_tokens=prompt_tokens, use_prompt=use_prompt,
            )
        )
        if recorder.toggle_sdp:
            torch.backends.cuda.enable_flash_sdp(False)
            torch.backends.cuda.enable_mem_efficient_sdp(False)
        if recorder.mode == "oom_once" and not recorder.oom_raised:
            recorder.oom_raised = True
            raise torch.cuda.OutOfMemoryError("CUDA out of memory (fake)")
        if recorder.mode != "silent":
            if recorder.mode == "runaway":
                frames = max_new_tokens - 1
            else:
                frames = min(4 + len(text.split()), max(max_new_tokens - 2, 1))
            codes = torch.zeros((_NUM_CODEBOOKS, frames), dtype=torch.long)
            yield _GenerateResponse(action="sample", codes=codes, text=text)
        yield _GenerateResponse(action="next")

    class DualARTransformer:
        # llama.py:499
        @staticmethod
        def from_pretrained(
            path, load_weights: bool = False, max_length=None, lora_config=None, rope_base=None
        ):
            recorder.model_loads.append(
                {"path": str(path), "load_weights": load_weights, "max_length": max_length}
            )
            recorder.init_skipped.append(torch.nn.init.normal_.__name__ != "normal_")
            return _FakeModel(recorder, max_length or 32768)

    # llama.py:229
    def _remap_fish_qwen3_omni_keys(weights):
        if not any(k.startswith(("text_model.", "audio_decoder.")) for k in weights):
            return weights
        remapped = type(weights)()
        for key, value in weights.items():
            if key.startswith("text_model.model."):
                key = key[len("text_model.model."):]
            elif key.startswith("audio_decoder."):
                suffix = key[len("audio_decoder."):]
                key = suffix if suffix.startswith("codebook_embeddings.") else "fast_" + suffix
            remapped[key] = value
        return remapped

    inference.Conversation = Conversation
    inference.tqdm = tqdm
    inference.decode_one_token_ar = decode_one_token_ar
    inference.load_codec_model = load_codec_model
    inference.decode_to_audio = decode_to_audio
    inference.generate_long = generate_long
    inference.GenerateResponse = _GenerateResponse
    llama.DualARTransformer = DualARTransformer
    llama._remap_fish_qwen3_omni_keys = _remap_fish_qwen3_omni_keys

    root = types.ModuleType("fish_speech")
    models = types.ModuleType("fish_speech.models")
    text2semantic = types.ModuleType("fish_speech.models.text2semantic")
    for package in (root, models, text2semantic):
        package.__path__ = []  # mark as packages
    return {
        "fish_speech": root,
        "fish_speech.models": models,
        "fish_speech.models.text2semantic": text2semantic,
        "fish_speech.models.text2semantic.inference": inference,
        "fish_speech.models.text2semantic.llama": llama,
    }


# ── Fixtures ─────────────────────────────────────────────────────────────────


def _wav_bytes(seconds: float = 1.0, sample_rate: int = _SAMPLE_RATE, freq: float = 220.0) -> bytes:
    t = np.arange(int(seconds * sample_rate), dtype=np.float32) / sample_rate
    buf = io.BytesIO()
    sf.write(buf, 0.3 * np.sin(2 * np.pi * freq * t), sample_rate, format="WAV")
    return buf.getvalue()


@pytest.fixture
def recorder() -> _Recorder:
    return _Recorder()


@pytest.fixture
def checkpoint_dir(tmp_path):
    directory = tmp_path / "snapshot"
    directory.mkdir()
    (directory / "config.json").write_text(json.dumps(_S2_CONFIG), encoding="utf-8")
    (directory / "codec.pth").write_bytes(b"\0" * 4000)
    (directory / "model-00001-of-00002.safetensors").write_bytes(b"\0" * 6000)
    (directory / "model-00002-of-00002.safetensors").write_bytes(b"\0" * 2000)
    return directory


@pytest.fixture
def upstream(monkeypatch, recorder, checkpoint_dir, tmp_path):
    """Installs the fake fish_speech package and a fake Hugging Face download."""
    for name, module in _build_fake_upstream(recorder).items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.delenv("ABM_FISH_REPO", raising=False)
    monkeypatch.delenv("ABM_FISH_CHECKPOINT_DIR", raising=False)

    import huggingface_hub

    def fake_snapshot_download(repo_id, *args, **kwargs):
        recorder.snapshot_ids.append(repo_id)
        return str(checkpoint_dir)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))  # voice-ref temp files
    monkeypatch.setattr(  # do not depend on the test machine's free memory
        FishSpeechProvider, "_available_host_ram_bytes", staticmethod(lambda: 64 * 1024 ** 3)
    )
    return recorder


def _config(**overrides: Any) -> AudiobookConfig:
    values: dict[str, Any] = dict(
        tts_provider_name="fish",
        tts_model_name="fishaudio/s2-pro",
        voice_transcript=_TRANSCRIPT,
    )
    values.update(overrides)
    return AudiobookConfig(**values)


def _provider(config: AudiobookConfig | None = None, **kwargs: Any) -> FishSpeechProvider:
    return FishSpeechProvider(config or _config(), device=kwargs.pop("device", "cpu"), **kwargs)


# ── INFO ─────────────────────────────────────────────────────────────────────


class TestInfo:
    def test_info_is_complete_and_truthful(self):
        info = FishSpeechProvider.info()
        assert info.name == "fish"
        assert info.commercial_use is False
        assert "Research License" in info.license
        assert info.homepage.startswith("https://github.com/fishaudio/fish-speech")
        assert info.default_model == "fishaudio/s2-pro"
        assert info.default_model in info.models
        # S1-mini cannot be loaded by the S2 code path, so it must not be offered.
        assert "fishaudio/s1-mini" not in info.models
        assert info.native_sample_rate == 44100
        assert 10.0 <= info.min_vram_gb <= 16.0
        assert info.transcript == "required"
        assert info.supports_voice_clone and info.supports_instruct and info.supports_seed
        assert info.supports_batch is False and info.supports_speed is False
        assert info.preset_voices == ()
        assert info.supports_voice_preset is True
        assert "English" in info.languages and len(info.languages) >= 80
        assert info.pip_requirements
        for needle in ("--no-deps", "ABM_FISH_REPO", "ABM_FISH_CHECKPOINT_DIR", "non-commercial"):
            assert needle in info.install_notes

    def test_recommended_settings_are_upstream_defaults(self):
        recommended = FishSpeechProvider.info().recommended_settings
        # fish-speech API server / WebUI / SGLang runner defaults.
        assert recommended == {
            "temperature": 0.8, "top_p": 0.8, "top_k": 30, "repetition_penalty": 1.1,
        }
        config_fields = set(AudiobookConfig.__dataclass_fields__)
        assert set(recommended) <= config_fields

        settings = {"temperature": 0.3, "top_p": 0.95, "top_k": 50, "repetition_penalty": 1.05}
        apply_recommended_settings("fish", settings, explicit={"top_p"})
        assert settings == {
            "temperature": 0.8, "top_p": 0.95, "top_k": 30, "repetition_penalty": 1.1,
        }

    def test_voice_preset_signatures_match_the_base_class(self):
        for name in ("save_voice_preset", "load_voice_preset"):
            assert inspect.signature(getattr(FishSpeechProvider, name)) == inspect.signature(
                getattr(BaseTTSProvider, name)
            )
            assert getattr(FishSpeechProvider, name) is not getattr(BaseTTSProvider, name)

    def test_options_are_well_formed(self):
        options = FishSpeechProvider.info().options
        keys = [o.key for o in options]
        assert len(keys) == len(set(keys))
        for option in options:
            assert option.kind in {"float", "int", "bool", "str", "choice", "file"}
            assert option.label and option.help
            if option.kind == "choice":
                assert option.default in option.choices
            if option.kind in {"int", "float"}:
                assert option.minimum <= option.default <= option.maximum
        for key in ("precision", "max_seq_len", "max_new_tokens", "inline_tags", "instruct_as_tag"):
            assert key in keys

    def test_registry_resolves_provider(self):
        assert provider_class("fish") is FishSpeechProvider
        assert canonical_name("s2-pro") == "fish"
        provider = get_tts_provider("fish-speech", _config(), device="cpu")
        assert isinstance(provider, FishSpeechProvider)
        assert provider.device == "cpu"
        assert provider.is_ready is False
        assert provider.get_name() == "Fish Audio S2 Pro"

    def test_module_imports_without_torch_or_upstream(self):
        code = (
            "import sys; sys.path.insert(0, %r); "
            "import audiobook_factory.tts_providers.fish_provider as m; "
            "assert 'torch' not in sys.modules and 'fish_speech' not in sys.modules; "
            "print(m.FishSpeechProvider.INFO.name)" % _ROOT
        )
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "fish"


# ── Loading ──────────────────────────────────────────────────────────────────


class TestLoading:
    def test_missing_package_error_names_install_command(self, monkeypatch):
        monkeypatch.setitem(sys.modules, "fish_speech", None)  # import raises ImportError
        monkeypatch.delenv("ABM_FISH_REPO", raising=False)
        provider = _provider()
        with pytest.raises(RuntimeError) as excinfo:
            provider.ensure_ready()
        message = str(excinfo.value)
        assert fish_provider._INSTALL_COMMAND in message
        assert "pip install --no-deps" in message
        assert "git+https://github.com/fishaudio/fish-speech" in message
        assert "ABM_FISH_REPO" in message
        assert provider.is_ready is False

    def test_invalid_repo_env_var_is_reported(self, monkeypatch, tmp_path):
        monkeypatch.setenv("ABM_FISH_REPO", str(tmp_path))
        with pytest.raises(RuntimeError, match="not a fish-speech clone"):
            _provider().ensure_ready()

    def test_loads_default_model_with_upstream_signatures(self, upstream, checkpoint_dir):
        provider = _provider()
        provider.ensure_ready()
        assert provider.is_ready
        assert upstream.snapshot_ids == ["fishaudio/s2-pro"]
        # Codec is built on the CPU in float32, then moved by the provider.
        assert upstream.codec_loads == [(str(checkpoint_dir / "codec.pth"), "cpu", torch.float32)]
        assert upstream.model_loads == [
            {"path": str(checkpoint_dir), "load_weights": True, "max_length": 4096}
        ]
        assert upstream.setup_caches == [
            {"max_batch_size": 1, "max_seq_len": 4096, "dtype": torch.float32}
        ]
        assert provider._model._cache_setup_done is True
        # The unused codec mask was replaced.
        assert tuple(provider._codec.pre_module.causal_mask.shape) == (1, 1)
        provider.ensure_ready()  # idempotent
        assert len(upstream.model_loads) == 1

    def test_foreign_model_name_falls_back_to_default(self, upstream):
        provider = _provider(_config(tts_model_name="Qwen/Qwen3-TTS-12Hz-1.7B-Base"))
        provider.ensure_ready()
        assert upstream.snapshot_ids == ["fishaudio/s2-pro"]
        provider.config = _config(tts_model_name="evil/unreviewed-repo")
        provider.ensure_ready()
        assert set(upstream.snapshot_ids) == {"fishaudio/s2-pro"}

    def test_checkpoint_env_var_skips_download(self, upstream, checkpoint_dir, monkeypatch):
        monkeypatch.setenv("ABM_FISH_CHECKPOINT_DIR", str(checkpoint_dir))
        _provider().ensure_ready()
        assert upstream.snapshot_ids == []
        assert upstream.model_loads[0]["path"] == str(checkpoint_dir)

    def test_quantized_looking_path_is_rejected(self, upstream, checkpoint_dir, monkeypatch):
        renamed = checkpoint_dir.parent / "s2-pro-int8"
        checkpoint_dir.rename(renamed)
        monkeypatch.setenv("ABM_FISH_CHECKPOINT_DIR", str(renamed))
        with pytest.raises(RuntimeError, match="int8"):
            _provider().ensure_ready()
        assert upstream.model_loads == []

    def test_dtype_override_and_precision_option(self, upstream):
        provider = _provider(dtype_override="bfloat16")
        provider.ensure_ready()
        assert next(provider._model.parameters()).dtype == torch.bfloat16
        assert next(provider._codec.parameters()).dtype == torch.bfloat16  # codec follows model
        provider.cleanup()

        half_on_cpu = _provider(dtype_override="float16")
        assert half_on_cpu._resolve_precision() == "float32"

        fp32_codec = _provider(
            _config(tts_options={"precision": "bfloat16", "codec_precision": "float32"})
        )
        fp32_codec.ensure_ready()
        assert next(fp32_codec._model.parameters()).dtype == torch.bfloat16
        assert next(fp32_codec._codec.parameters()).dtype == torch.float32

    def test_reload_when_settings_change_between_runs(self, upstream):
        provider = _provider()
        provider.ensure_ready()
        provider.config = _config(tts_options={"max_seq_len": 8192})
        provider.ensure_ready()
        assert [load["max_length"] for load in upstream.model_loads] == [4096, 8192]
        provider.config = _config(tts_options={"max_seq_len": 0})  # checkpoint default
        provider.ensure_ready()
        assert upstream.model_loads[-1]["max_length"] is None
        provider.config = _config(tts_options={"max_seq_len": 1024})  # below the floor
        provider.ensure_ready()
        assert upstream.model_loads[-1]["max_length"] == 3072

    def test_insufficient_vram_raises_before_loading(self, upstream, monkeypatch):
        provider = _provider(device="cuda:0")
        monkeypatch.setattr(provider, "bind_device", lambda: None)
        monkeypatch.setattr(provider, "_estimate_vram_bytes", lambda d, k: 11 * 1024 ** 3)
        monkeypatch.setattr(provider, "_free_vram_bytes", lambda: 6 * 1024 ** 3)
        with pytest.raises(RuntimeError) as excinfo:
            provider.ensure_ready()
        message = str(excinfo.value)
        assert "6.0 GiB free" in message and "12.0 GiB" in message
        assert "max_seq_len" in message
        assert upstream.codec_loads == [] and upstream.model_loads == []
        assert provider.is_ready is False

    def test_vram_estimate_matches_s2_pro_geometry(self, upstream, checkpoint_dir):
        provider = _provider(dtype_override="float16")
        short = fish_provider._LoadKey(
            model_id="fishaudio/s2-pro", checkpoint_override="", precision="float16",
            codec_precision="float16", max_seq_len=4096, compile=False, trim_codec=True,
        )
        native = fish_provider._LoadKey(
            model_id="fishaudio/s2-pro", checkpoint_override="", precision="float16",
            codec_precision="float16", max_seq_len=0, compile=False, trim_codec=False,
        )
        files = 8000 + 4000 // 2  # weights at 2 bytes/param + codec halved
        kv_short = 36 * 2 * 8 * 4096 * 128 * 2
        kv_native = 36 * 2 * 8 * 32768 * 128 * 2
        assert provider._estimate_vram_bytes(checkpoint_dir, short) == files + kv_short + 4096 ** 2
        assert provider._estimate_vram_bytes(checkpoint_dir, native) == (
            files + kv_native + 32768 ** 2 + 3 * 32768 ** 2
        )
        # The context cap and mask trimming together save about 7.9 GiB.
        saved = provider._estimate_vram_bytes(checkpoint_dir, native) - provider._estimate_vram_bytes(
            checkpoint_dir, short
        )
        assert 7.5 * 1024 ** 3 < saved < 8.5 * 1024 ** 3

    def test_insufficient_host_ram_raises_before_loading(self, upstream, monkeypatch):
        monkeypatch.setattr(
            FishSpeechProvider, "_available_host_ram_bytes", staticmethod(lambda: 2 * 1024 ** 3)
        )
        with pytest.raises(RuntimeError, match="free host RAM"):
            _provider().ensure_ready()
        assert upstream.codec_loads == [] and upstream.model_loads == []

    def test_low_ram_load_skips_random_init_when_checkpoint_is_complete(
        self, upstream, checkpoint_dir
    ):
        # The fake model's only weight is "weight"; upstream's remap strips the prefix.
        (checkpoint_dir / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"text_model.model.weight": "model-00001-of-00002.safetensors"}}),
            encoding="utf-8",
        )
        provider = _provider()
        provider.ensure_ready()
        assert upstream.init_skipped == [True]
        assert torch.nn.init.normal_.__name__ == "normal_"  # restored
        assert float(torch.nn.Linear(4, 4).weight.detach().abs().sum()) > 0

        provider.config = _config(tts_options={"low_ram_load": False, "max_seq_len": 8192})
        provider.ensure_ready()
        assert upstream.init_skipped == [True, False]

    def test_low_ram_load_falls_back_when_a_weight_is_missing(self, upstream, checkpoint_dir):
        (checkpoint_dir / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"text_model.model.other": "model-00001-of-00002.safetensors"}}),
            encoding="utf-8",
        )
        _provider().ensure_ready()
        # First build skipped init, found "weight" unsupplied, rebuilt the upstream way.
        assert upstream.init_skipped == [True, False]
        assert torch.nn.init.normal_.__name__ == "normal_"

    def test_legacy_checkpoint_uses_upstream_initialisation(self, upstream):
        _provider().ensure_ready()  # fixture has no safetensors index to verify against
        assert upstream.init_skipped == [False]

    def test_cleanup_drops_models(self, upstream):
        provider = _provider()
        provider.synthesize("Hello there.", _wav_bytes(), return_bytes=True)
        assert provider.is_ready and provider._references
        provider.cleanup()
        assert provider.is_ready is False
        assert provider._codec is None and provider._decode_one_token is None
        assert not provider._references

    def test_upstream_display_helpers_are_silenced(self, upstream):
        provider = _provider()
        provider.ensure_ready()
        inference = sys.modules["fish_speech.models.text2semantic.inference"]
        assert inference.Conversation().visualize(object(), merge_semantic_tokens=True) is None
        assert list(inference.tqdm(range(3))) == [0, 1, 2]


# ── Synthesis ────────────────────────────────────────────────────────────────


class TestCodecBuiltInInferenceMode:
    """Upstream's ``load_codec_model`` runs under ``torch.inference_mode``."""

    def test_reduced_precision_codec_still_encodes_the_reference(self, upstream):
        # On a GPU the codec is converted to the model's precision after
        # loading. Done outside inference mode, that left the weights unusable:
        # "Inference tensors do not track version counter" on every chunk.
        provider = _provider(dtype_override="bfloat16")
        provider.ensure_ready()
        assert next(provider._codec.parameters()).dtype == torch.bfloat16
        wav, duration = provider.synthesize("Hello there.", _wav_bytes(), return_bytes=True)
        assert duration > 0 and wav[:4] == b"RIFF"
        assert upstream.encode_calls == 1

    def test_codec_weights_stay_inference_tensors(self, upstream):
        provider = _provider(dtype_override="bfloat16")
        provider.ensure_ready()
        assert all(parameter.is_inference() for parameter in provider._codec.parameters())


class TestSynthesis:
    def test_kwargs_reflect_config_and_options(self, upstream):
        config = _config(
            temperature=0.65, top_p=0.9, top_k=40, repetition_penalty=1.2, seed=11,
            tts_instruct="calm audiobook narration",
            tts_options={"max_new_tokens": 500, "tokens_per_byte": 2.0},
        )
        provider = _provider(config)
        audio, duration = provider.synthesize("Hello there, world.", _wav_bytes(), return_bytes=True)

        call = upstream.generate_calls[-1]
        assert call["temperature"] == pytest.approx(0.65)
        assert call["top_p"] == pytest.approx(0.9)
        assert call["top_k"] == 40
        assert call["repetition_penalty"] == pytest.approx(1.2)
        assert call["text"] == "<|speaker:0|>[calm audiobook narration] Hello there, world."
        assert call["device"] == "cpu"
        assert call["num_samples"] == 1 and call["compile"] is False
        assert call["model"] is provider._model
        assert call["decode_one_token"] is provider._decode_one_token
        assert call["prompt_text"] == [_TRANSCRIPT]
        assert isinstance(call["prompt_tokens"], list) and len(call["prompt_tokens"]) == 1
        codes = call["prompt_tokens"][0]
        assert codes.dtype == torch.long and codes.device.type == "cpu"
        assert tuple(codes.shape) == (_NUM_CODEBOOKS, math.ceil(_SAMPLE_RATE / _FRAME_LENGTH))
        assert call["use_prompt"] is True
        text_bytes = len(call["text"].encode("utf-8"))
        assert call["max_new_tokens"] == 64 + math.ceil(text_bytes * 2.0)
        assert call["chunk_length"] >= text_bytes  # the chunk is never split upstream

        data, sample_rate = sf.read(io.BytesIO(audio))
        assert sample_rate == 44100
        assert duration == pytest.approx(len(data) / 44100)
        assert duration == pytest.approx((4 + len(call["text"].split())) * _FRAME_LENGTH / 44100)

    def test_out_of_range_sampling_values_are_clamped(self, upstream):
        provider = _provider(_config(temperature=5.0, top_p=0.0, top_k=0))
        provider.synthesize("Hello.", _wav_bytes(), return_bytes=True)
        call = upstream.generate_calls[-1]
        assert 0 < call["temperature"] < 2
        assert 0 < call["top_p"] <= 1
        assert call["top_k"] >= 155776  # top-k disabled

    def test_options_change_prompt_shape(self, upstream):
        config = _config(
            tts_instruct="[whisper] [low voice]",
            tts_options={"speaker_tag": False},
        )
        provider = _provider(config)
        provider.synthesize("Quiet now.", _wav_bytes(), return_bytes=True)
        assert upstream.generate_calls[-1]["text"] == "[whisper] [low voice] Quiet now."

        provider.config = _config(tts_instruct="warm", tts_options={"instruct_as_tag": False})
        provider.synthesize("Quiet now.", _wav_bytes(), return_bytes=True)
        assert upstream.generate_calls[-1]["text"] == "<|speaker:0|>Quiet now."

    def test_inline_tags_and_control_tokens(self, upstream):
        text = "He said [laughing] hello <|speaker:1|><|im_end|> again [sic]."
        provider = _provider()
        provider.synthesize(text, _wav_bytes(), return_bytes=True)
        assert upstream.generate_calls[-1]["text"] == (
            "<|speaker:0|>He said [laughing] hello again [sic]."
        )
        provider.config = _config(tts_options={"inline_tags": "strip"})
        provider.synthesize(text, _wav_bytes(), return_bytes=True)
        assert upstream.generate_calls[-1]["text"] == "<|speaker:0|>He said hello again ."
        provider.config = _config(tts_options={"inline_tags": "speak"})
        provider.synthesize(text, _wav_bytes(), return_bytes=True)
        assert upstream.generate_calls[-1]["text"] == (
            "<|speaker:0|>He said (laughing) hello again (sic)."
        )

    def test_reference_encoded_once_across_chunks(self, upstream):
        provider = _provider()
        voice = _wav_bytes()
        for text in ("First chunk.", "Second chunk.", "Third chunk."):
            provider.synthesize(text, voice, return_bytes=True)
        provider.synthesize_batch(["Fourth.", "Fifth."], voice)
        assert upstream.encode_calls == 1
        assert len(upstream.generate_calls) == 5
        tokens = {id(call["prompt_tokens"][0]) for call in upstream.generate_calls}
        assert len(tokens) == 1  # the very same tensor is reused

        # The cache key is content hash + transcript.
        provider.config = _config(voice_transcript="A different transcript.")
        provider.synthesize("Sixth.", voice, return_bytes=True)
        assert upstream.encode_calls == 2
        provider.synthesize("Seventh.", _wav_bytes(freq=330.0), return_bytes=True)
        assert upstream.encode_calls == 3
        provider.synthesize("Eighth.", bytes(voice), return_bytes=True)  # equal content
        assert upstream.encode_calls == 3

    def test_path_reference_uses_sidecar_transcript_and_content_hash(self, upstream, tmp_path):
        clip = tmp_path / "narrator.wav"
        clip.write_bytes(_wav_bytes())
        (tmp_path / "narrator.txt").write_text("Sidecar transcript.", encoding="utf-8")
        provider = _provider(_config(voice_transcript=""))
        provider.synthesize("One.", str(clip), return_bytes=True)
        assert upstream.generate_calls[0]["prompt_text"] == ["Sidecar transcript."]

        # The pipeline passes bytes; the sidecar is then found through voice_file.
        provider.config = _config(voice_transcript="", voice_file=str(clip))
        out_path = str(tmp_path / "out" / "two.wav")
        result, duration = provider.synthesize("Two.", clip.read_bytes(), out_path)
        assert result == out_path and os.path.isfile(out_path) and duration > 0
        assert upstream.generate_calls[1]["prompt_text"] == ["Sidecar transcript."]
        assert upstream.encode_calls == 1  # same content by path and by bytes

    def test_resamples_reference_to_codec_rate(self, upstream):
        try:
            import torchaudio  # noqa: F401
        except Exception:
            pytest.importorskip("librosa")
        provider = _provider()
        provider.synthesize("Hello.", _wav_bytes(seconds=1.0, sample_rate=24000), return_bytes=True)
        frames = upstream.generate_calls[-1]["prompt_tokens"][0].shape[1]
        assert frames == math.ceil(_SAMPLE_RATE / _FRAME_LENGTH)

    def test_voice_preset_tokens_skip_the_codec(self, upstream, tmp_path):
        preset = tmp_path / "narrator.npy"
        np.save(preset, np.ones((_NUM_CODEBOOKS, 33), dtype=np.int64))
        provider = _provider(_config(voice_preset=str(preset)))
        provider.synthesize("Hello.", _wav_bytes(), return_bytes=True)
        assert upstream.encode_calls == 0
        assert tuple(upstream.generate_calls[-1]["prompt_tokens"][0].shape) == (_NUM_CODEBOOKS, 33)

    def test_voice_preset_round_trip(self, upstream, tmp_path):
        voice = _wav_bytes()
        saver = _provider(_config(voice_transcript="Ignored config transcript."))
        path = str(tmp_path / "presets" / "narrator.pt")
        info = saver.save_voice_preset(path, voice, transcript="Typed  by the <|im_end|> user.")

        assert json.loads(json.dumps(info)) == info  # JSON-safe
        assert info["path"] == path and os.path.isfile(path)
        assert info["transcript"] == "Typed by the user."
        assert info["format"] == "abm-fish-voice-preset" and info["model_id"] == "fishaudio/s2-pro"
        assert info["num_codebooks"] == _NUM_CODEBOOKS
        assert info["frames"] == math.ceil(_SAMPLE_RATE / _FRAME_LENGTH)
        assert info["sample_rate"] == 44100 and info["created_at"]
        assert upstream.encode_calls == 1

        # Tensors and plain values only: loads without unpickling objects.
        raw = torch.load(path, map_location="cpu", weights_only=True)
        assert tuple(raw["codes"].shape) == (_NUM_CODEBOOKS, info["frames"])
        assert raw["codes"].dtype == torch.long

        # A fresh instance needs neither the clip nor a configured transcript.
        user = _provider(_config(voice_preset=path, voice_transcript=""))
        assert user.load_voice_preset(path) == info
        user.synthesize("Hello there.", b"", return_bytes=True)
        user.synthesize("Again.", voice, return_bytes=True)  # the preset replaces the clip
        assert upstream.encode_calls == 1
        for call in upstream.generate_calls:
            assert call["prompt_text"] == ["Typed by the user."]
            assert torch.equal(call["prompt_tokens"][0], raw["codes"])
        assert upstream.generate_calls[0]["prompt_tokens"][0] is upstream.generate_calls[1]["prompt_tokens"][0]

    def test_save_voice_preset_defaults_to_config_voice(self, upstream, tmp_path):
        clip = tmp_path / "narrator.wav"
        clip.write_bytes(_wav_bytes())
        provider = _provider(_config(voice_file=str(clip)))
        info = provider.save_voice_preset(str(tmp_path / "narrator.pt"))
        assert info["transcript"] == _TRANSCRIPT and len(info["source_sha256"]) == 64

    def test_save_voice_preset_rejects_bad_input(self, upstream, tmp_path):
        provider = _provider(_config(voice_transcript=""))
        with pytest.raises(ValueError, match="transcript"):
            provider.save_voice_preset(str(tmp_path / "a.pt"), _wav_bytes())
        with pytest.raises(ValueError, match="no built-in voices"):
            provider.save_voice_preset(str(tmp_path / "a.pt"), None, transcript="Text.")
        with pytest.raises(ValueError, match=r"\.pt"):
            provider.save_voice_preset(str(tmp_path / "a.json"), _wav_bytes(), transcript="Text.")
        assert not list(tmp_path.glob("a.*"))

    @pytest.mark.filterwarnings("ignore:Detected pickle protocol")
    def test_load_voice_preset_rejects_incompatible_files(self, upstream, tmp_path):
        import pickle

        provider = _provider()
        with pytest.raises(ValueError, match="not found"):
            provider.load_voice_preset(str(tmp_path / "missing.pt"))

        wrong_books = tmp_path / "wrong_books.pt"
        torch.save(torch.zeros((4, 20), dtype=torch.long), wrong_books)
        with pytest.raises(ValueError, match="4 codebooks"):
            provider.load_voice_preset(str(wrong_books))

        out_of_range = tmp_path / "range.npy"
        np.save(out_of_range, np.full((_NUM_CODEBOOKS, 5), 5000, dtype=np.int64))
        with pytest.raises(ValueError, match="codebook range"):
            provider.load_voice_preset(str(out_of_range))

        floats = tmp_path / "floats.pt"
        torch.save(torch.zeros((_NUM_CODEBOOKS, 5)), floats)
        with pytest.raises(ValueError, match="codec-token array"):
            provider.load_voice_preset(str(floats))

        other_format = tmp_path / "other.pt"
        torch.save({"format": "qwen-voice-preset", "codes": torch.zeros((10, 5), dtype=torch.long)},
                   other_format)
        with pytest.raises(ValueError, match="not a Fish voice preset"):
            provider.load_voice_preset(str(other_format))

        other_model = tmp_path / "other_model.pt"
        torch.save({"format": "abm-fish-voice-preset", "model_id": "fishaudio/other",
                    "transcript": "x", "codes": torch.zeros((10, 5), dtype=torch.long)}, other_model)
        with pytest.raises(ValueError, match="fishaudio/other"):
            provider.load_voice_preset(str(other_model))

        class Payload:  # an arbitrary object must never be unpickled
            def __reduce__(self):
                return (os.system, ("echo unpickled > %s" % (tmp_path / "pwned"),))

        malicious = tmp_path / "malicious.pt"
        malicious.write_bytes(pickle.dumps(Payload()))
        with pytest.raises(ValueError, match="not a readable Fish voice preset"):
            provider.load_voice_preset(str(malicious))
        assert not (tmp_path / "pwned").exists()

        with pytest.raises(ValueError, match=r"\.pt or \.npy"):
            provider.load_voice_preset(__file__)

    def test_token_only_preset_uses_config_transcript(self, upstream, tmp_path):
        preset = tmp_path / "fake.npy"  # what fish-speech's own encoder writes
        np.save(preset, np.ones((_NUM_CODEBOOKS, 12), dtype=np.int64))
        provider = _provider(_config(voice_transcript=""))
        with pytest.raises(ValueError, match="transcript"):
            provider.load_voice_preset(str(preset))
        provider.config = _config(voice_transcript="Spoken words.")
        info = provider.load_voice_preset(str(preset))
        assert info["transcript"] == "Spoken words." and info["format"] == "codec-tokens"

    def test_missing_transcript_raises_unless_allowed(self, upstream):
        provider = _provider(_config(voice_transcript=""))
        with pytest.raises(RuntimeError, match="transcript"):
            provider.synthesize("Hello.", _wav_bytes(), return_bytes=True)
        assert upstream.generate_calls == []

        provider.config = _config(voice_transcript="", tts_options={"allow_missing_transcript": True})
        provider.synthesize("Hello.", _wav_bytes(), return_bytes=True)
        call = upstream.generate_calls[-1]
        assert call["prompt_text"] == [""] and call["use_prompt"] is True

    def test_missing_reference_raises(self, upstream):
        provider = _provider()
        with pytest.raises(RuntimeError, match="no built-in voices"):
            provider.synthesize("Hello.", b"", return_bytes=True)

    def test_batch_preserves_order_and_count(self, upstream):
        provider = _provider()
        texts = ["One.", "One two three four.", "One two.", "One two three four five six."]
        results = provider.synthesize_batch(texts, _wav_bytes())
        assert len(results) == len(texts)
        sent = [call["text"] for call in upstream.generate_calls]
        assert sent == [f"<|speaker:0|>{text}" for text in texts]
        for (audio, duration), text in zip(results, texts):
            assert isinstance(audio, bytes) and audio[:4] == b"RIFF"
            assert duration == pytest.approx((4 + len(text.split())) * _FRAME_LENGTH / 44100)

    def test_batch_retries_item_after_out_of_memory(self, upstream):
        upstream.mode = "oom_once"
        provider = _provider()
        results = provider.synthesize_batch(["One.", "Two words."], _wav_bytes())
        assert len(results) == 2 and upstream.oom_raised
        assert [c["text"] for c in upstream.generate_calls] == [
            "<|speaker:0|>One.", "<|speaker:0|>One.", "<|speaker:0|>Two words.",
        ]

    def test_token_budget_scales_with_text_length(self, upstream):
        provider = _provider(_config(tts_options={"max_new_tokens": 0}))
        voice = _wav_bytes()
        short, medium, long_text = "Hi.", "word " * 20, "word " * 79  # 3, 100, 395 chars
        for text in (short, medium, long_text):
            provider.synthesize(text, voice, return_bytes=True)
        budgets = [call["max_new_tokens"] for call in upstream.generate_calls]
        assert budgets[0] < budgets[1] < budgets[2]
        for call, budget in zip(upstream.generate_calls, budgets):
            assert budget == 64 + math.ceil(len(call["text"].encode("utf-8")) * 4.0)
        assert budgets[0] < 200  # a three-letter chunk cannot run for minutes
        assert budgets[2] < 2048

        # The explicit cap wins when it is lower.
        provider.config = _config(tts_options={"max_new_tokens": 300})
        provider.synthesize(long_text, voice, return_bytes=True)
        assert upstream.generate_calls[-1]["max_new_tokens"] == 300

        # Multi-byte scripts speak fewer characters per second and get more room.
        provider.config = _config(tts_options={"max_new_tokens": 0})
        provider.synthesize("你好世界你好世界", voice, return_bytes=True)
        cjk = upstream.generate_calls[-1]["max_new_tokens"]
        provider.synthesize("abcdefgh", voice, return_bytes=True)
        assert cjk > upstream.generate_calls[-1]["max_new_tokens"]

    def test_budget_is_limited_by_context(self, upstream):
        provider = _provider(
            _config(tts_options={"max_new_tokens": 0, "max_seq_len": 3072, "tokens_per_byte": 8.0})
        )
        voice = _wav_bytes(seconds=40.0)  # 862 reference frames
        provider.synthesize("word " * 79, voice, return_bytes=True)
        call = upstream.generate_calls[-1]
        prompt_bound = 862 + len(_TRANSCRIPT) + len(call["text"].encode("utf-8")) + 64
        assert call["max_new_tokens"] == 3072 - prompt_bound

        too_long = _wav_bytes(seconds=60.0)  # 1292 frames > 3072 - 2048
        with pytest.raises(RuntimeError, match="do not fit the model context"):
            provider.synthesize("Hello.", too_long, return_bytes=True)

    def test_runaway_generation_is_retried_then_raises(self, upstream):
        upstream.mode = "runaway"
        provider = _provider(_config(seed=5, tts_options={"runaway_retries": 2}))
        with pytest.raises(RuntimeError, match="did not finish within"):
            provider.synthesize("Hello there.", _wav_bytes(), return_bytes=True)
        assert len(upstream.generate_calls) == 3
        assert upstream.decode_calls == 0  # truncated audio is never decoded

    def test_no_tokens_raises(self, upstream):
        upstream.mode = "silent"
        provider = _provider(_config(tts_options={"runaway_retries": 0}))
        with pytest.raises(RuntimeError, match="produced no audio tokens"):
            provider.synthesize("Hello there.", _wav_bytes(), return_bytes=True)
        assert len(upstream.generate_calls) == 1

    def test_empty_audio_raises(self, upstream):
        upstream.mode = "empty_audio"
        with pytest.raises(RuntimeError, match="produced no audio"):
            _provider().synthesize("Hello there.", _wav_bytes(), return_bytes=True)

    def test_nan_audio_raises(self, upstream):
        upstream.mode = "nan"
        with pytest.raises(RuntimeError, match="NaN/inf"):
            _provider().synthesize("Hello there.", _wav_bytes(), return_bytes=True)

    def test_nan_in_reduced_precision_codec_retries_in_float32(self, upstream):
        upstream.mode = "nan"
        provider = _provider(dtype_override="bfloat16")
        with pytest.raises(RuntimeError, match="NaN/inf"):
            provider.synthesize("Hello there.", _wav_bytes(), return_bytes=True)
        assert upstream.decode_calls == 2
        assert next(provider._codec.parameters()).dtype == torch.float32

    def test_empty_text_raises(self, upstream):
        with pytest.raises(RuntimeError, match="empty text"):
            _provider().synthesize("  <|im_end|> ", _wav_bytes(), return_bytes=True)

    def test_attention_kernel_flags_are_restored(self, upstream):
        backend = torch.backends.cuda
        before = (backend.flash_sdp_enabled(), backend.mem_efficient_sdp_enabled())
        upstream.toggle_sdp = True  # upstream leaves the global flags changed
        try:
            _provider().synthesize("Hello there.", _wav_bytes(), return_bytes=True)
            assert (backend.flash_sdp_enabled(), backend.mem_efficient_sdp_enabled()) == before
        finally:
            backend.enable_flash_sdp(before[0])
            backend.enable_mem_efficient_sdp(before[1])

    def test_two_instances_share_no_model_state(self, upstream):
        first = _provider(device="cpu")
        second = _provider(device="cpu")
        voice = _wav_bytes()
        first.synthesize("From the first.", voice, return_bytes=True)
        second.synthesize("From the second.", voice, return_bytes=True)
        assert first._model is not second._model and first._codec is not second._codec
        assert first._lock is not second._lock
        assert upstream.encode_calls == 2  # one reference encode per instance
        assert upstream.generate_calls[0]["model"] is first._model
        assert upstream.generate_calls[1]["model"] is second._model
        first.cleanup()
        assert second.is_ready
