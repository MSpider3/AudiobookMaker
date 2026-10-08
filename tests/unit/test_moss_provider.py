"""
test_moss_provider.py
======================
Unit tests for the MOSS-TTS provider.

No network, GPU or upstream code is used: ``transformers``, ``huggingface_hub``
and ``torchaudio`` are replaced in ``sys.modules`` by fakes that mirror the
signatures of the remote code the provider was written against
(``processing_moss_tts.py`` / ``modeling_moss_tts.py`` of the pinned Hub
commits). In particular the fake Delay model's ``generate`` takes no
``**kwargs``, exactly like upstream's, so an unsupported keyword fails here.
"""
from __future__ import annotations

import importlib.machinery
import io
import json
import os
import subprocess
import sys
import types
from typing import Any

import pytest

torch = pytest.importorskip("torch")
sf = pytest.importorskip("soundfile")

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from audiobook_factory.pipeline import AudiobookConfig  # noqa: E402
from audiobook_factory.tts_providers import moss_provider as mp  # noqa: E402
from audiobook_factory.tts_providers.base_tts_provider import get_tts_provider  # noqa: E402
from audiobook_factory.tts_providers.registry import provider_class, provider_info  # noqa: E402

_LOCAL = "OpenMOSS-Team/MOSS-TTS-Local-Transformer"
_LOCAL_V15 = "OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5"
_DELAY_V15 = "OpenMOSS-Team/MOSS-TTS-v1.5"
_DELAY = "OpenMOSS-Team/MOSS-TTS"
_CODEC_V1 = "OpenMOSS-Team/MOSS-Audio-Tokenizer"
_CODEC_V2 = "OpenMOSS-Team/MOSS-Audio-Tokenizer-v2"
_ALLOWED_REPOS = {_LOCAL, _LOCAL_V15, _DELAY_V15, _DELAY, _CODEC_V1, _CODEC_V2}
_FPS = 12.5


def _default_frames(text: str) -> int:
    """Frames the fake model "speaks" for a text; longer text, longer audio."""
    return max(2, len(text) // 4)


class World:
    """Shared state of the fake upstream stack, and its behaviour switches."""

    def __init__(self, root: str) -> None:
        self.root = root
        self.downloads: list[dict[str, Any]] = []
        self.processor_loads: list[dict[str, Any]] = []
        self.config_loads: list[dict[str, Any]] = []
        self.model_loads: list[dict[str, Any]] = []
        self.class_lookups: list[tuple[str, str]] = []
        self.processors: list["FakeProcessor"] = []
        self.models: list[Any] = []
        self.frames_for = _default_frames
        self.runaway = False          # never emit the stop token
        self.no_audio = False         # decode returns None for every item
        self.nan_audio = False
        self.oom_above = 0            # raise CUDA OOM for batches larger than this
        self.pending: list[tuple[str, int]] = []

    def repo_of(self, path: str) -> str:
        """Maps a fake snapshot directory back to its repository id."""
        return os.path.basename(str(path)).replace("--", "/")


class FakeCodec(torch.nn.Module):
    """Stands in for ``MossAudioTokenizerModel``: encoder + decoder modules."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = torch.nn.ModuleList([torch.nn.Linear(2, 2)])
        self.decoder = torch.nn.ModuleList([torch.nn.Linear(2, 2)])
        self.moves: list[str] = []

    def to(self, device: Any = None, *args: Any, **kwargs: Any) -> "FakeCodec":  # type: ignore[override]
        self.moves.append(str(device))
        return self


class FakeProcessor:
    """Mirror of ``MossTTSDelayProcessor`` / ``MossTTSLocalProcessor``."""

    world: World

    def __init__(self, model_dir: str, kwargs: dict[str, Any]) -> None:
        world = self.world
        self.repo = world.repo_of(model_dir)
        spec = mp._MODEL_SPECS[self.repo]
        self.audio_tokenizer: Any = FakeCodec()
        self.model_config = types.SimpleNamespace(sampling_rate=spec.sample_rate, n_vq=spec.n_vq)
        self.arch = spec.architecture
        self.encode_calls: list[dict[str, Any]] = []
        self.calls: list[dict[str, Any]] = []
        self.user_messages: list[dict[str, Any]] = []

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
        record = dict(kwargs, path=str(pretrained_model_name_or_path))
        cls.world.processor_loads.append(record)
        allowed = {"trust_remote_code", "codec_path", "codec_weight_dtype",
                   "codec_compute_dtype", "codec_attention_implementation"}
        # upstream forwards unknown keywords to ProcessorMixin.__init__, which raises
        unexpected = set(kwargs) - allowed
        if unexpected:
            raise TypeError(f"Unexpected keyword argument {sorted(unexpected)[0]}.")
        processor = cls(str(pretrained_model_name_or_path), kwargs)
        cls.world.processors.append(processor)
        return processor

    def build_user_message(self, text=None, reference=None, instruction=None, tokens=None,
                           quality=None, sound_event=None, ambient_sound=None, language=None):
        message = {
            "role": "user", "text": text, "reference": reference, "instruction": instruction,
            "tokens": tokens, "language": language,
        }
        self.user_messages.append(message)
        return message

    @staticmethod
    def build_assistant_message(audio_codes_list, content="<|audio|>"):
        return {"role": "assistant", "audio_codes_list": audio_codes_list, "content": content}

    def encode_audios_from_wav(self, wav_list, sampling_rate, n_vq=None):
        codec = self.audio_tokenizer
        self.encode_calls.append({
            "shape": tuple(wav_list[0].shape), "sampling_rate": int(sampling_rate), "n_vq": n_vq,
            "encoder_modules": len(codec.encoder),
        })
        frames = max(1, int(wav_list[0].shape[-1] / sampling_rate * _FPS))
        return [torch.ones((frames, self.model_config.n_vq), dtype=torch.long) for _ in wav_list]

    def __call__(self, *args, **kwargs):
        conversations = args[0] if args else kwargs.pop("conversations")
        mode = kwargs.pop("mode", "generation")
        if (mode == "generation") != (conversations[0][-1]["role"] == "user"):
            raise ValueError("mode does not match the conversation")
        self.calls.append({"mode": mode, "conversations": conversations})
        self.world.pending = [(conv[0]["text"], 0) for conv in conversations]
        batch = len(conversations)
        width = 1 + self.model_config.n_vq
        return {
            "input_ids": torch.zeros((batch, 7, width), dtype=torch.long),
            "attention_mask": torch.ones((batch, 7), dtype=torch.bool),
        }

    def decode(self, output):
        world = self.world
        samples_per_frame = int(round(self.model_config.sampling_rate / _FPS))
        messages = []
        for index, (_, ids) in enumerate(output):
            frames = int(ids.shape[0])
            if world.no_audio or frames == 0:
                messages.append(None)
                continue
            if self.arch == "local":
                frames += 1          # the stop step's row decodes into one extra frame
            value = 0.01 * (index + 1)
            if self.arch == "local_v15":
                audio = torch.full((2, frames * samples_per_frame), value)
                audio[1] *= -0.5     # stereo with different channels
            else:
                audio = torch.full((frames * samples_per_frame,), value)
            if world.nan_audio:
                audio = audio * float("nan")
            messages.append(types.SimpleNamespace(audio_codes_list=[audio]))
        return messages


class _FakeModelBase:
    world: World

    def __init__(self, model_dir: str, kwargs: dict[str, Any]) -> None:
        self.repo = self.world.repo_of(model_dir)
        self.load_kwargs = kwargs
        self.device: str | None = None
        self.eval_called = False
        self.generate_calls: list[dict[str, Any]] = []

    def to(self, device):
        self.device = str(device)
        return self

    def eval(self):
        self.eval_called = True
        return self

    def _run(self, frames_budget: int, **recorded: Any) -> list[tuple[int, Any]]:
        world = self.world
        batch = int(recorded["input_ids"].shape[0])
        self.generate_calls.append(
            {k: (tuple(v.shape) if hasattr(v, "shape") else v) for k, v in recorded.items()}
        )
        if world.oom_above and batch > world.oom_above:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory (fake)")
        width = recorded["input_ids"].shape[-1]
        outputs = []
        for text, _ in world.pending:
            frames = frames_budget if world.runaway else min(world.frames_for(text), frames_budget)
            outputs.append((0, torch.zeros((frames, width), dtype=torch.long)))
        return outputs


class FakeLocalModel(_FakeModelBase):
    """``moss_tts_local/modeling_moss_tts.py``: MossTTSDelayModel.generate (1.7B)."""

    def generate(self, input_ids, attention_mask=None, generation_config=None,
                 max_new_tokens=None, text_temperature=None, text_top_p=None, text_top_k=None,
                 text_repetition_penalty=None, audio_temperature=None, audio_top_p=None,
                 audio_top_k=None, audio_repetition_penalty=None, n_vq_for_inference=None,
                 **kwargs):
        if kwargs:
            raise TypeError(f"GenerationMixin.generate got unused model kwargs: {sorted(kwargs)}")
        return self._run(
            max_new_tokens - 1,       # one step is the stop decision
            input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=max_new_tokens,
            text_temperature=text_temperature, text_top_p=text_top_p, text_top_k=text_top_k,
            audio_temperature=audio_temperature, audio_top_p=audio_top_p, audio_top_k=audio_top_k,
            audio_repetition_penalty=audio_repetition_penalty, n_vq_for_inference=n_vq_for_inference,
        )


class FakeDelayModel(_FakeModelBase):
    """``moss_tts_delay/modeling_moss_tts.py``: MossTTSDelayModel.generate (8B). No **kwargs."""

    def generate(self, input_ids, attention_mask=None, max_new_tokens=1000, text_temperature=1.5,
                 text_top_p=1.0, text_top_k=50, audio_temperature=1.7, audio_top_p=0.8,
                 audio_top_k=25, audio_repetition_penalty=1.0):
        n_vq = input_ids.shape[-1] - 1
        return self._run(
            max_new_tokens - (n_vq + 2),   # audio_start, delay flush, audio_end, im_end
            input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=max_new_tokens,
            text_temperature=text_temperature, text_top_p=text_top_p, text_top_k=text_top_k,
            audio_temperature=audio_temperature, audio_top_p=audio_top_p, audio_top_k=audio_top_k,
            audio_repetition_penalty=audio_repetition_penalty,
        )


class FakeLocalV15Model(_FakeModelBase):
    """``moss_tts_local_v1.5/modeling_moss_tts.py``: MossTTSLocalModel.generate (4B)."""

    def generate(self, input_ids, attention_mask=None, max_new_tokens=None, max_new_frames=None,
                 do_sample=True, text_temperature=1.0, text_top_p=1.0, text_top_k=50,
                 audio_temperature=None, audio_top_p=None, audio_top_k=None,
                 audio_repetition_penalty=None, temperature=1.0, top_p=0.95, top_k=50,
                 repetition_penalty=1.0, use_kv_cache=True, n_vq_for_inference=None, nq=None,
                 **kwargs):
        if n_vq_for_inference not in (None, input_ids.shape[-1] - 1):
            raise ValueError("fixed RVQ depth")
        if do_sample and (text_temperature <= 0 or audio_temperature <= 0):
            raise ValueError("temperature must be positive when do_sample=True.")
        return self._run(
            max_new_tokens - 1,
            input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=max_new_tokens,
            do_sample=do_sample, text_temperature=text_temperature, text_top_p=text_top_p,
            text_top_k=text_top_k, audio_temperature=audio_temperature, audio_top_p=audio_top_p,
            audio_top_k=audio_top_k, audio_repetition_penalty=audio_repetition_penalty,
            extra=sorted(kwargs),
        )


_FAKE_MODELS = {"local": FakeLocalModel, "delay": FakeDelayModel, "local_v15": FakeLocalV15Model}


class FakeUpstreamConfig:
    """Mirror of the remote config; v1.5 Local keeps public attention attributes."""

    def __init__(self, repo: str) -> None:
        self.repo = repo
        if mp._MODEL_SPECS[repo].architecture == "local_v15":
            self.attn_implementation = "flash_attention_2"
            self.local_transformer_attn_implementation = "flash_attention_2"


@pytest.fixture
def world(tmp_path, monkeypatch) -> World:
    """Installs the fake upstream stack and returns its shared state."""
    state = World(str(tmp_path / "hub"))
    FakeProcessor.world = state
    _FakeModelBase.world = state

    def snapshot_download(repo_id, revision=None, **kwargs):
        state.downloads.append({"repo_id": repo_id, "revision": revision, **kwargs})
        path = os.path.join(state.root, repo_id.replace("/", "--"))
        os.makedirs(path, exist_ok=True)
        if repo_id in mp._MODEL_SPECS:
            name = "MossTTSLocalProcessor" if repo_id == _LOCAL_V15 else "MossTTSDelayProcessor"
            with open(os.path.join(path, "processor_config.json"), "w", encoding="utf-8") as fh:
                json.dump({"auto_map": {"AutoProcessor": f"processing_moss_tts.{name}"}}, fh)
        return path

    def get_class_from_dynamic_module(class_reference, pretrained_model_name_or_path, **kwargs):
        state.class_lookups.append((class_reference, str(pretrained_model_name_or_path)))
        return FakeProcessor

    class AutoConfig:
        @staticmethod
        def from_pretrained(pretrained_model_name_or_path, *args, **kwargs):
            state.config_loads.append(dict(kwargs, path=str(pretrained_model_name_or_path)))
            return FakeUpstreamConfig(state.repo_of(pretrained_model_name_or_path))

    class AutoModel:
        @staticmethod
        def from_pretrained(pretrained_model_name_or_path, *args, **kwargs):
            repo = state.repo_of(pretrained_model_name_or_path)
            state.model_loads.append(dict(kwargs, path=str(pretrained_model_name_or_path)))
            model = _FAKE_MODELS[mp._MODEL_SPECS[repo].architecture](str(pretrained_model_name_or_path), kwargs)
            state.models.append(model)
            return model

    class AutoProcessor:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            raise AssertionError("AutoProcessor must not be used: it injects keywords upstream rejects")

    transformers = types.ModuleType("transformers")
    transformers.__version__ = "5.0.0"
    transformers.AutoConfig = AutoConfig
    transformers.AutoModel = AutoModel
    transformers.AutoProcessor = AutoProcessor
    dynamic = types.ModuleType("transformers.dynamic_module_utils")
    dynamic.get_class_from_dynamic_module = get_class_from_dynamic_module
    transformers.dynamic_module_utils = dynamic
    hub = types.ModuleType("huggingface_hub")
    hub.snapshot_download = snapshot_download
    torchaudio = types.ModuleType("torchaudio")
    torchaudio.__spec__ = importlib.machinery.ModuleSpec("torchaudio", None)

    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "transformers.dynamic_module_utils", dynamic)
    monkeypatch.setitem(sys.modules, "huggingface_hub", hub)
    monkeypatch.setitem(sys.modules, "torchaudio", torchaudio)
    state.transformers = transformers  # type: ignore[attr-defined]
    return state


@pytest.fixture
def voice_bytes() -> bytes:
    path = os.path.join(_ROOT, "tests", "fixtures", "audio", "synthetic_voice_reference.wav")
    with open(path, "rb") as fh:
        return fh.read()


def _provider(world: World, **config: Any) -> mp.MossTTSProvider:
    config.setdefault("tts_provider_name", "moss")
    config.setdefault("tts_model_name", "")
    provider = get_tts_provider("moss", AudiobookConfig(**config), device="cpu")
    assert isinstance(provider, mp.MossTTSProvider)
    return provider


def _duration_frames(result: tuple[Any, float]) -> float:
    return result[1] * _FPS


def _read(result: tuple[Any, float]) -> tuple[Any, int]:
    return sf.read(io.BytesIO(result[0]), dtype="float32")


# ── Static description ────────────────────────────────────────────────────────


class TestInfo:
    def test_registry_and_info_sanity(self):
        info = provider_info("moss")
        assert provider_class("moss-tts") is mp.MossTTSProvider
        assert info.name == "moss" and info.display_name == "MOSS-TTS"
        assert info.license == "Apache-2.0" and info.commercial_use is True
        assert info.default_model == _LOCAL
        assert info.default_model in info.models
        assert set(info.models) == {_LOCAL, _LOCAL_V15, _DELAY_V15, _DELAY}
        assert all(model.startswith("OpenMOSS-Team/") for model in info.models)
        assert not any("TTSD" in model or "Nano" in model for model in info.models)
        assert info.native_sample_rate == 24000
        assert 9.0 <= info.min_vram_gb <= 15.0, "the default model must fit a 16 GB T4"
        assert info.transcript == "optional"
        assert info.supports_batch and info.supports_speed and info.supports_seed
        assert info.supports_voice_clone and info.supports_voice_preset
        assert info.supports_instruct is False and info.preset_voices == ()
        assert len(info.languages) == 31 and "English" in info.languages
        assert "transformers==5.0.0" in info.pip_requirements
        assert "qwen" in info.install_notes.lower()

    def test_recommended_settings_are_upstreams_values_for_the_default_model(self):
        assert provider_info("moss").recommended_settings == {
            "temperature": 1.0, "top_p": 0.95, "top_k": 50, "repetition_penalty": 1.1,
        }

    def test_options_are_well_formed(self):
        info = provider_info("moss")
        keys = [option.key for option in info.options]
        assert len(keys) == len(set(keys))
        for option in info.options:
            assert option.kind in {"float", "int", "bool", "str", "choice"}
            assert option.label and option.help
            if option.kind == "choice":
                assert option.default in option.choices
            if option.minimum is not None and option.maximum is not None:
                assert option.minimum <= option.default <= option.maximum
        for key in ("clone_mode", "duration_control", "pause_tags", "max_new_tokens",
                    "sampling_preset", "text_temperature", "rvq_depth", "model_dtype"):
            assert key in keys

    def test_every_model_has_pinned_revisions(self):
        for model_id, spec in mp._MODEL_SPECS.items():
            assert len(spec.revision) == 40, model_id
            assert spec.codec in mp._CODEC_SPECS
        for codec in mp._CODEC_SPECS.values():
            assert len(codec.revision) == 40 and codec.repo.startswith("OpenMOSS-Team/")

    def test_module_imports_without_torch_or_transformers(self):
        code = (
            "import sys; sys.path.insert(0, %r)\n"
            "import audiobook_factory.tts_providers.moss_provider as m\n"
            "assert m.MossTTSProvider.INFO.name == 'moss'\n"
            "assert 'torch' not in sys.modules and 'transformers' not in sys.modules\n"
        ) % _ROOT
        done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
        assert done.returncode == 0, done.stderr

    def test_constructor_is_lazy(self, world):
        provider = _provider(world)
        assert provider.device == "cpu"
        assert provider.is_ready is False
        assert provider.sample_rate == 24000
        assert world.downloads == [] and world.model_loads == []
        provider.config = AudiobookConfig(tts_model_name=_LOCAL_V15)
        assert provider.sample_rate == 48000


# ── Loading ───────────────────────────────────────────────────────────────────


class TestLoading:
    def test_load_calls_upstream_with_pinned_snapshots(self, world):
        provider = _provider(world)
        provider.ensure_ready()
        assert provider.is_ready

        assert [d["repo_id"] for d in world.downloads] == [_LOCAL, _CODEC_V1]
        assert world.downloads[0]["revision"] == mp._MODEL_SPECS[_LOCAL].revision
        assert world.downloads[1]["revision"] == mp._CODEC_SPECS["v1"].revision

        model_dir = os.path.join(world.root, _LOCAL.replace("/", "--"))
        codec_dir = os.path.join(world.root, _CODEC_V1.replace("/", "--"))
        assert world.class_lookups == [("processing_moss_tts.MossTTSDelayProcessor", model_dir)]
        assert world.processor_loads == [
            {"trust_remote_code": True, "codec_path": codec_dir, "path": model_dir}
        ]
        assert world.config_loads == [{"trust_remote_code": True, "path": model_dir}]

        load = world.model_loads[0]
        assert load["path"] == model_dir and load["trust_remote_code"] is True
        assert load["dtype"] is torch.float32, "CPU runs in float32"
        assert load["attn_implementation"] == "eager"
        assert isinstance(load["config"], FakeUpstreamConfig)
        assert "device_map" not in load and "revision" not in load
        model = world.models[0]
        assert model.device == "cpu" and model.eval_called

    def test_codec_encoder_is_parked_off_the_device(self, world):
        provider = _provider(world)
        provider.ensure_ready()
        codec = world.processors[0].audio_tokenizer
        assert len(codec.encoder) == 0, "the codec must be moved without its encoder"
        assert len(provider._codec_encoder) == 1
        assert codec.moves == ["cpu"]

    def test_codec_encoder_stays_when_offload_is_off(self, world):
        provider = _provider(world, tts_options={"offload_codec_encoder": False})
        provider.ensure_ready()
        assert len(world.processors[0].audio_tokenizer.encoder) == 1
        assert provider._codec_encoder is None

    def test_v15_local_gets_codec_keywords_and_a_runnable_attention_backend(self, world):
        provider = _provider(world, tts_model_name=_LOCAL_V15)
        provider.ensure_ready()
        assert [d["repo_id"] for d in world.downloads] == [_LOCAL_V15, _CODEC_V2]
        load = world.processor_loads[0]
        assert load["codec_weight_dtype"] == "fp32" and load["codec_compute_dtype"] == "fp32"
        assert load["codec_attention_implementation"] == "sdpa"
        config = world.model_loads[0]["config"]
        # config.json ships flash_attention_2, which only Ampere+ with flash-attn can run
        assert config.attn_implementation == "eager"
        assert config.local_transformer_attn_implementation == "eager"
        assert provider.sample_rate == 48000

    def test_dtype_override_and_options(self, world, monkeypatch):
        monkeypatch.setattr(torch.cuda, "mem_get_info", lambda index: (15 << 30, 16 << 30))
        provider = get_tts_provider("moss", AudiobookConfig(tts_model_name=""), device="cuda:0",
                                    dtype_override="bfloat16")
        provider.ensure_ready()
        load = world.model_loads[0]
        assert load["dtype"] is torch.bfloat16
        assert load["attn_implementation"] == "sdpa"
        assert world.models[0].device == "cuda:0"

    def test_old_gpu_defaults_to_float16(self, world, monkeypatch):
        monkeypatch.setattr(torch.cuda, "mem_get_info", lambda index: (15 << 30, 16 << 30))
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index=None: (7, 5))
        provider = get_tts_provider("moss", AudiobookConfig(tts_model_name=""), device="cuda:0")
        provider.ensure_ready()
        assert world.model_loads[0]["dtype"] is torch.float16

    def test_new_gpu_defaults_to_bfloat16(self, world, monkeypatch):
        monkeypatch.setattr(torch.cuda, "mem_get_info", lambda index: (23 << 30, 24 << 30))
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index=None: (8, 9))
        provider = get_tts_provider("moss", AudiobookConfig(tts_model_name=""), device="cuda:0")
        provider.ensure_ready()
        assert world.model_loads[0]["dtype"] is torch.bfloat16

    def test_model_too_big_for_the_device_fails_before_downloading(self, world, monkeypatch):
        monkeypatch.setattr(torch.cuda, "mem_get_info", lambda index: (int(14.6 * (1 << 30)), 16 << 30))
        provider = get_tts_provider("moss", AudiobookConfig(tts_model_name=_DELAY_V15), device="cuda:0")
        with pytest.raises(RuntimeError) as excinfo:
            provider.ensure_ready()
        message = str(excinfo.value)
        assert _DELAY_V15 in message and "GiB" in message and "cuda:0" in message
        assert _LOCAL in message, "the error must name the model that does fit"
        assert world.downloads == [] and world.model_loads == []
        assert provider.is_ready is False

    def test_default_model_fits_a_t4(self, world, monkeypatch):
        monkeypatch.setattr(torch.cuda, "mem_get_info", lambda index: (int(14.6 * (1 << 30)), 16 << 30))
        monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index=None: (7, 5))
        provider = get_tts_provider("moss", AudiobookConfig(tts_model_name=""), device="cuda:0")
        provider.ensure_ready()
        assert provider.is_ready

    def test_int8_is_not_passed_to_upstream(self, world):
        provider = _provider(world, quantization="int8")
        provider.ensure_ready()
        assert "quantization_config" not in world.model_loads[0]
        assert "load_in_8bit" not in world.model_loads[0]

    def test_model_change_reloads(self, world, voice_bytes):
        provider = _provider(world)
        provider.synthesize("Hello there, world.", voice_bytes, return_bytes=True)
        assert len(world.models) == 1
        provider.synthesize("Hello again, world.", voice_bytes, return_bytes=True)
        assert len(world.models) == 1, "an unchanged config must not reload"

        provider.config = AudiobookConfig(tts_model_name=_LOCAL_V15)
        _, duration = provider.synthesize("Hello there, world.", voice_bytes, return_bytes=True)
        assert len(world.models) == 2 and world.models[1].repo == _LOCAL_V15
        assert provider.sample_rate == 48000 and duration > 0

    def test_cleanup_drops_everything(self, world, voice_bytes):
        provider = _provider(world)
        provider.synthesize("Hello there, world.", voice_bytes, return_bytes=True)
        processor = world.processors[0]
        provider.cleanup()
        assert provider.is_ready is False
        assert provider._model is None and provider._processor is None
        assert provider._codec_encoder is None and processor.audio_tokenizer is None
        assert len(provider._reference_cache) == 0
        provider.cleanup()   # idempotent


# ── Dependency errors ─────────────────────────────────────────────────────────


class TestDependencies:
    def test_missing_transformers_names_the_install_command(self, world, monkeypatch):
        monkeypatch.setitem(sys.modules, "transformers", None)
        provider = _provider(world)
        with pytest.raises(RuntimeError) as excinfo:
            provider.ensure_ready()
        assert "pip install -r requirements/tts-moss.txt" in str(excinfo.value)
        assert 'pip install "transformers==5.0.0"' in str(excinfo.value)

    def test_missing_torchaudio_names_the_install_command(self, world, monkeypatch):
        monkeypatch.setitem(sys.modules, "torchaudio", None)
        provider = _provider(world)
        with pytest.raises(RuntimeError) as excinfo:
            provider.ensure_ready()
        assert "torchaudio" in str(excinfo.value)
        assert "pip install -r requirements/tts-moss.txt" in str(excinfo.value)

    def test_transformers_4_is_rejected_with_the_fix(self, world):
        world.transformers.__version__ = "4.57.3"
        provider = _provider(world)
        with pytest.raises(RuntimeError) as excinfo:
            provider.ensure_ready()
        message = str(excinfo.value)
        assert "4.57.3" in message and "pip install -r requirements/tts-moss.txt" in message
        assert world.downloads == []

    def test_default_model_rejects_transformers_newer_than_5_0(self, world):
        world.transformers.__version__ = "5.3.0"
        provider = _provider(world)
        with pytest.raises(RuntimeError) as excinfo:
            provider.ensure_ready()
        assert 'pip install "transformers==5.0.0"' in str(excinfo.value)
        assert world.downloads == []

    def test_other_models_accept_newer_transformers(self, world):
        world.transformers.__version__ = "5.19.0"
        provider = _provider(world, tts_model_name=_LOCAL_V15)
        provider.ensure_ready()
        assert provider.is_ready

    def test_missing_dependency_error_reaches_synthesize(self, world, voice_bytes, monkeypatch):
        monkeypatch.setitem(sys.modules, "huggingface_hub", None)
        provider = _provider(world)
        with pytest.raises(RuntimeError, match="pip install"):
            provider.synthesize("Hello.", voice_bytes, return_bytes=True)


# ── Model allowlist ───────────────────────────────────────────────────────────


class TestAllowlist:
    @pytest.mark.parametrize("foreign", [
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base", "attacker/repo", "OpenMOSS-Team/MOSS-TTSD-v1.0",
        "../../etc/passwd", "OpenMOSS-Team/MOSS-TTS-Local-Transformer/../../x",
    ])
    def test_foreign_model_falls_back_to_default(self, world, voice_bytes, foreign):
        provider = _provider(world, tts_model_name=foreign)
        provider.synthesize("Hello there, world.", voice_bytes, return_bytes=True)
        assert provider.resolve_model_id() == _LOCAL
        assert {d["repo_id"] for d in world.downloads} == {_LOCAL, _CODEC_V1}
        seen = (
            [load["path"] for load in world.processor_loads + world.config_loads + world.model_loads]
            + [load["codec_path"] for load in world.processor_loads]
            + [path for _, path in world.class_lookups]
        )
        assert seen and all(world.repo_of(path) in _ALLOWED_REPOS for path in seen)
        assert not any(foreign in str(path) for path in seen)
        assert world.models[0].repo == _LOCAL

    def test_remote_code_is_only_trusted_for_listed_repositories(self, world):
        for model_id in mp.MossTTSProvider.INFO.models:
            provider = _provider(world, tts_model_name=model_id)
            provider.ensure_ready()
            provider.cleanup()
        assert {d["repo_id"] for d in world.downloads} <= _ALLOWED_REPOS
        for load in world.processor_loads + world.config_loads + world.model_loads:
            assert load["trust_remote_code"] is True
            assert world.repo_of(load["path"]) in mp._MODEL_SPECS

    def test_revision_option_follows_upstream_for_model_and_codec(self, world):
        provider = _provider(world, tts_options={"model_revision": "main"})
        provider.ensure_ready()
        assert [(d["repo_id"], d["revision"]) for d in world.downloads] == [
            (_LOCAL, "main"), (_CODEC_V1, "main"),
        ]


# ── Generation keywords ───────────────────────────────────────────────────────


class TestGenerationKeywords:
    def test_defaults_are_upstreams_recommended_values(self, world, voice_bytes):
        # AudiobookConfig defaults (0.3 / 0.8 / 50 / 1.05) are Qwen's, not MOSS's.
        provider = _provider(world)
        provider.synthesize("Hello there, world.", voice_bytes, return_bytes=True)
        call = world.models[0].generate_calls[0]
        assert call["audio_temperature"] == 1.0 and call["audio_top_p"] == 0.95
        assert call["audio_top_k"] == 50 and call["audio_repetition_penalty"] == 1.1
        assert call["text_temperature"] == 1.5 and call["text_top_p"] == 1.0 and call["text_top_k"] == 50
        assert call["n_vq_for_inference"] is None
        assert call["input_ids"][0] == 1 and call["input_ids"][2] == 33

    def test_user_values_of_shared_fields_are_honoured(self, world, voice_bytes):
        provider = _provider(world, temperature=0.8, top_p=0.9, top_k=30, repetition_penalty=1.2)
        provider.synthesize("Hello there, world.", voice_bytes, return_bytes=True)
        call = world.models[0].generate_calls[0]
        assert call["audio_temperature"] == 0.8 and call["audio_top_p"] == 0.9
        assert call["audio_top_k"] == 30 and call["audio_repetition_penalty"] == 1.2

    def test_only_untouched_fields_are_replaced(self, world, voice_bytes):
        provider = _provider(world, tts_model_name=_DELAY_V15, temperature=1.3)
        provider.synthesize("Hello there, world.", voice_bytes, return_bytes=True)
        call = world.models[0].generate_calls[0]
        assert call["audio_temperature"] == 1.3
        assert call["audio_top_p"] == 0.8 and call["audio_top_k"] == 25
        assert call["audio_repetition_penalty"] == 1.0

    def test_default_models_operating_point_is_not_forced_on_the_8b_model(self, world, voice_bytes):
        settings = provider_info("moss").recommended_settings
        provider = _provider(world, tts_model_name=_DELAY_V15, **settings)
        provider.synthesize("Hello there, world.", voice_bytes, return_bytes=True)
        call = world.models[0].generate_calls[0]
        assert call["audio_temperature"] == 1.7 and call["audio_top_p"] == 0.8
        assert call["audio_top_k"] == 25 and call["audio_repetition_penalty"] == 1.0

    def test_sampling_preset_config_passes_shared_fields_verbatim(self, world, voice_bytes):
        provider = _provider(world, tts_options={"sampling_preset": "config"})
        provider.synthesize("Hello there, world.", voice_bytes, return_bytes=True)
        call = world.models[0].generate_calls[0]
        assert call["audio_temperature"] == 0.3 and call["audio_top_p"] == 0.8
        assert call["audio_top_k"] == 50 and call["audio_repetition_penalty"] == 1.05

    def test_sampling_preset_recommended_ignores_shared_fields(self, world, voice_bytes):
        provider = _provider(world, temperature=0.8, tts_options={"sampling_preset": "recommended"})
        provider.synthesize("Hello there, world.", voice_bytes, return_bytes=True)
        assert world.models[0].generate_calls[0]["audio_temperature"] == 1.0

    def test_text_layer_and_rvq_depth_options(self, world, voice_bytes):
        provider = _provider(world, tts_options={
            "text_temperature": "0.9", "text_top_p": 0.7, "text_top_k": 12, "rvq_depth": 8,
        })
        provider.synthesize("Hello there, world.", voice_bytes, return_bytes=True)
        call = world.models[0].generate_calls[0]
        assert call["text_temperature"] == 0.9 and call["text_top_p"] == 0.7 and call["text_top_k"] == 12
        assert call["n_vq_for_inference"] == 8

    def test_delay_model_receives_only_keywords_it_declares(self, world, voice_bytes):
        provider = _provider(world, tts_model_name=_DELAY_V15, tts_options={"rvq_depth": 8})
        _, duration = provider.synthesize("Hello there, world.", voice_bytes, return_bytes=True)
        call = world.models[0].generate_calls[0]
        assert set(call) == {
            "input_ids", "attention_mask", "max_new_tokens", "text_temperature", "text_top_p",
            "text_top_k", "audio_temperature", "audio_top_p", "audio_top_k", "audio_repetition_penalty",
        }
        assert call["audio_temperature"] == 1.7 and call["audio_top_k"] == 25
        assert duration > 0

    def test_v15_local_gets_do_sample_and_positive_temperatures(self, world, voice_bytes):
        provider = _provider(world, tts_model_name=_LOCAL_V15, temperature=0.0,
                             tts_options={"sampling_preset": "config"})
        provider.synthesize("Hello there, world.", voice_bytes, return_bytes=True)
        call = world.models[0].generate_calls[0]
        assert call["do_sample"] is False and call["extra"] == []
        assert call["audio_temperature"] > 0 and call["text_temperature"] > 0
        assert call["input_ids"][2] == 13

    def test_prompt_fields_for_v15_models(self, world, voice_bytes):
        provider = _provider(world, tts_model_name=_DELAY_V15, language="fr",
                             tts_options={"instruction": "calm", "duration_control": "on"})
        provider.synthesize("Bonjour tout le monde.[pause 1.5s] Ca va.", voice_bytes, return_bytes=True)
        message = world.processors[0].user_messages[0]
        assert message["language"] == "French"
        assert message["instruction"] == "calm"
        assert message["text"] == "Bonjour tout le monde.[pause 1.5s] Ca va.", "v1.5 reads pause tags itself"
        assert isinstance(message["reference"], list) and len(message["reference"]) == 1
        assert message["reference"][0].shape[1] == 32
        assert message["tokens"] > int(1.5 * _FPS)
        assert world.processors[0].calls[0]["mode"] == "generation"

    def test_v1_models_get_no_language_tag_unless_asked(self, world, voice_bytes):
        provider = _provider(world, language="English")
        provider.synthesize("Hello there, world.", voice_bytes, return_bytes=True)
        assert world.processors[0].user_messages[0]["language"] is None
        assert world.processors[0].user_messages[0]["tokens"] is None
        assert world.processors[0].user_messages[0]["instruction"] is None

        provider.config = AudiobookConfig(tts_model_name="", language="German",
                                          tts_options={"language_tag": "always"})
        provider.synthesize("Hallo Welt, wie geht es.", voice_bytes, return_bytes=True)
        assert world.processors[0].user_messages[-1]["language"] == "German"

    def test_seed_is_applied(self, world, voice_bytes, monkeypatch):
        seeds: list[int] = []
        monkeypatch.setattr(torch, "manual_seed", lambda seed: seeds.append(seed))
        provider = _provider(world, seed=1234)
        provider.synthesize("Hello there, world.", voice_bytes, return_bytes=True)
        assert seeds == [1234]


# ── Reference clip ────────────────────────────────────────────────────────────


class TestReference:
    def test_reference_is_encoded_once_across_chunks_and_calls(self, world, voice_bytes):
        provider = _provider(world)
        provider.synthesize("First chunk of the book.", voice_bytes, return_bytes=True)
        provider.synthesize_batch(["Second chunk.", "Third chunk.", "Fourth chunk."], voice_bytes)
        provider.synthesize("Fifth chunk of the book.", voice_bytes, return_bytes=True)
        processor = world.processors[0]
        assert len(processor.encode_calls) == 1
        encode = processor.encode_calls[0]
        assert encode["sampling_rate"] == 24000 and encode["n_vq"] is None
        assert len(encode["shape"]) == 2 and encode["shape"][0] == 1, "(channels, samples)"
        assert encode["encoder_modules"] == 1, "the parked encoder must be attached while encoding"
        assert len(processor.audio_tokenizer.encoder) == 0, "and parked again afterwards"
        references = [m["reference"][0] for m in processor.user_messages]
        assert len(references) == 5 and all(ref is references[0] for ref in references)

    def test_new_clip_or_transcript_is_encoded_again(self, world, voice_bytes, tmp_path):
        provider = _provider(world)
        provider.synthesize("First chunk of the book.", voice_bytes, return_bytes=True)
        data, rate = sf.read(io.BytesIO(voice_bytes), dtype="float32")
        other = tmp_path / "other.wav"
        sf.write(str(other), data[: len(data) // 2] * 0.5, rate)
        provider.synthesize("Second chunk of the book.", str(other), return_bytes=True)
        assert len(world.processors[0].encode_calls) == 2

        provider.config = AudiobookConfig(tts_model_name="", voice_transcript="A new transcript.")
        provider.synthesize("Third chunk of the book.", str(other), return_bytes=True)
        assert len(world.processors[0].encode_calls) == 3
        provider.synthesize("Fourth chunk of the book.", str(other), return_bytes=True)
        assert len(world.processors[0].encode_calls) == 3

    def test_reference_is_trimmed_and_stereo_is_kept_for_upstream(self, world, tmp_path):
        import numpy as np

        rate = 44100
        t = np.arange(rate * 6, dtype=np.float32) / rate
        stereo = np.stack([0.2 * np.sin(2 * np.pi * 200 * t), 0.2 * np.sin(2 * np.pi * 300 * t)], axis=1)
        path = tmp_path / "stereo.wav"
        sf.write(str(path), stereo, rate)
        provider = _provider(world, tts_options={"max_reference_seconds": 2})
        provider.synthesize("Hello there, world.", str(path), return_bytes=True)
        encode = world.processors[0].encode_calls[0]
        assert encode["shape"] == (2, 2 * rate)
        assert encode["sampling_rate"] == rate, "upstream resamples; the provider passes the true rate"

    def test_missing_reference_raises(self, world):
        provider = _provider(world, voice_file="")
        with pytest.raises(RuntimeError, match="reference clip"):
            provider.synthesize("Hello there, world.", b"", return_bytes=True)

    def test_config_voice_file_is_the_fallback(self, world, voice_bytes, tmp_path):
        path = tmp_path / "narrator.wav"
        path.write_bytes(voice_bytes)
        provider = _provider(world, voice_file=str(path))
        _, duration = provider.synthesize("Hello there, world.", "", return_bytes=True)
        assert duration > 0 and len(world.processors[0].encode_calls) == 1

    def test_continuation_modes(self, world, voice_bytes):
        provider = _provider(world, voice_transcript="The reference sentence.", speed=1.2,
                             tts_options={"clone_mode": "reference+continuation"})
        provider.synthesize("Hello there, world.", voice_bytes, return_bytes=True)
        processor = world.processors[0]
        call = processor.calls[0]
        user, assistant = call["conversations"][0]
        assert call["mode"] == "continuation"
        assert user["text"] == "The reference sentence. Hello there, world."
        assert user["tokens"] is None, "upstream disables duration control when continuing"
        assert user["reference"][0] is assistant["audio_codes_list"][0]

        provider.config = AudiobookConfig(tts_model_name="", voice_transcript="The reference sentence.",
                                          tts_options={"clone_mode": "continuation"})
        provider.synthesize("Hello there, world.", voice_bytes, return_bytes=True)
        user, assistant = processor.calls[-1]["conversations"][0]
        assert user["reference"] is None and len(assistant["audio_codes_list"]) == 1

    def test_continuation_without_transcript_falls_back_to_reference(self, world, voice_bytes):
        provider = _provider(world, tts_options={"clone_mode": "continuation"})
        provider.synthesize("Hello there, world.", voice_bytes, return_bytes=True)
        call = world.processors[0].calls[0]
        assert call["mode"] == "generation" and len(call["conversations"][0]) == 1


# ── Voice presets ─────────────────────────────────────────────────────────────


class TestVoicePreset:
    def test_save_load_and_use(self, world, voice_bytes, tmp_path):
        path = str(tmp_path / "voices" / "narrator.pt")
        provider = _provider(world)
        saved = provider.save_voice_preset(path, voice_bytes, transcript="Preset transcript.")
        assert saved["path"] == path and saved["provider"] == "moss"
        assert saved["codebooks"] == 32 and saved["frames"] > 0 and saved["seconds"] > 0
        assert saved["transcript"] == "Preset transcript."
        assert saved["codec_repo"] == _CODEC_V1 and saved["model"] == _LOCAL
        json.dumps(saved)
        assert len(world.processors[0].encode_calls) == 1

        # weights_only=True must be enough to read the file back
        payload = torch.load(path, map_location="cpu", weights_only=True)
        assert payload["codes"].dtype == torch.long and payload["format"] == mp._PRESET_FORMAT

        other = _provider(world, voice_preset=path, voice_file="",
                          tts_options={"clone_mode": "continuation"})
        loaded = other.load_voice_preset(path)
        assert loaded == saved
        other.synthesize("Hello there, world.", b"", return_bytes=True)
        processor = world.processors[1]
        assert processor.encode_calls == [], "a preset needs no codec encoder pass"
        call = processor.calls[0]
        assert call["mode"] == "continuation"
        assert call["conversations"][0][0]["text"].startswith("Preset transcript. ")

    def test_default_arguments_use_the_configured_clip(self, world, voice_bytes, tmp_path):
        clip = tmp_path / "narrator.wav"
        clip.write_bytes(voice_bytes)
        provider = _provider(world, voice_file=str(clip), voice_transcript="From the config.")
        saved = provider.save_voice_preset(str(tmp_path / "narrator.pt"))
        assert saved["transcript"] == "From the config."

    def test_load_rejects_foreign_files(self, world, voice_bytes, tmp_path):
        provider = _provider(world)
        with pytest.raises(ValueError):
            provider.load_voice_preset(str(tmp_path / "missing.pt"))
        junk = tmp_path / "junk.pt"
        junk.write_bytes(b"not a checkpoint")
        with pytest.raises(ValueError):
            provider.load_voice_preset(str(junk))
        foreign = tmp_path / "foreign.pt"
        torch.save({"format": "something-else", "codes": torch.zeros(3, 32, dtype=torch.long)}, str(foreign))
        with pytest.raises(ValueError):
            provider.load_voice_preset(str(foreign))

    def test_preset_of_another_codec_is_rejected_and_falls_back_to_the_clip(self, world, voice_bytes, tmp_path):
        path = str(tmp_path / "narrator.pt")
        _provider(world).save_voice_preset(path, voice_bytes)
        provider = _provider(world, tts_model_name=_LOCAL_V15, voice_preset=path)
        with pytest.raises(ValueError, match="MOSS-Audio-Tokenizer"):
            provider.load_voice_preset(path)
        _, duration = provider.synthesize("Hello there, world.", voice_bytes, return_bytes=True)
        assert duration > 0
        assert len(world.processors[1].encode_calls) == 1, "the clip is tokenized instead"

    def test_preset_is_shared_by_models_with_the_same_codec(self, world, voice_bytes, tmp_path):
        path = str(tmp_path / "narrator.pt")
        _provider(world).save_voice_preset(path, voice_bytes)
        provider = _provider(world, tts_model_name=_DELAY_V15, voice_preset=path)
        provider.synthesize("Hello there, world.", b"", return_bytes=True)
        assert world.processors[1].encode_calls == []

    def test_a_clip_given_as_preset_is_used_as_reference(self, world, voice_bytes, tmp_path):
        clip = tmp_path / "narrator.wav"
        clip.write_bytes(voice_bytes)
        provider = _provider(world, voice_preset=str(clip), voice_file="")
        provider.synthesize("Hello there, world.", b"", return_bytes=True)
        assert len(world.processors[0].encode_calls) == 1

    def test_unusable_preset_without_a_clip_explains_itself(self, world, tmp_path):
        missing = str(tmp_path / "gone.pt")
        provider = _provider(world, voice_preset=missing, voice_file="")
        with pytest.raises(RuntimeError, match="gone.pt"):
            provider.synthesize("Hello there, world.", b"", return_bytes=True)

        junk = tmp_path / "junk.pt"
        junk.write_bytes(b"not a checkpoint")
        provider.config = AudiobookConfig(tts_model_name="", voice_preset=str(junk), voice_file="")
        for _ in range(2):   # the rejection is remembered, the error stays the same
            with pytest.raises(RuntimeError, match="not a MOSS-TTS voice preset"):
                provider.synthesize("Hello there, world.", b"", return_bytes=True)

    def test_preset_must_be_a_pt_file(self, world, voice_bytes, tmp_path):
        with pytest.raises(ValueError):
            _provider(world).save_voice_preset(str(tmp_path / "narrator.bin"), voice_bytes)


# ── Batching and budgets ──────────────────────────────────────────────────────


class TestBatch:
    def test_batch_is_one_forward_pass_in_input_order(self, world, voice_bytes):
        texts = ["A" * 20, "B" * 80, "C" * 40, "D" * 120]
        provider = _provider(world)
        results = provider.synthesize_batch(texts, voice_bytes)
        assert len(results) == len(texts)
        assert len(world.models[0].generate_calls) == 1
        assert world.models[0].generate_calls[0]["input_ids"][0] == 4
        for text, result in zip(texts, results):
            assert isinstance(result[0], bytes)
            # exactly the frames the model spoke: the stop step's extra frame is trimmed
            assert _duration_frames(result) == pytest.approx(_default_frames(text))
            audio, rate = _read(result)
            assert rate == 24000 and audio.ndim == 1
        conversations = world.processors[0].calls[0]["conversations"]
        assert [conv[0]["text"] for conv in conversations] == texts

    def test_stop_frame_is_kept_when_trimming_is_off(self, world, voice_bytes):
        provider = _provider(world, tts_options={"trim_eos_frame": "false"})
        result = provider.synthesize("A" * 40, voice_bytes, return_bytes=True)
        assert _duration_frames(result) == pytest.approx(_default_frames("A" * 40) + 1)

    def test_large_batches_are_split(self, world, voice_bytes):
        texts = [f"Chunk number {index} of the book." for index in range(5)]
        provider = _provider(world, tts_options={"max_batch_size": 2})
        results = provider.synthesize_batch(texts, voice_bytes)
        assert [call["input_ids"][0] for call in world.models[0].generate_calls] == [2, 2, 1]
        assert len(results) == 5
        spoken = [conv[0]["text"] for call in world.processors[0].calls for conv in call["conversations"]]
        assert spoken == texts

    def test_empty_batch(self, world, voice_bytes):
        provider = _provider(world)
        assert provider.synthesize_batch([], voice_bytes) == []
        assert world.downloads == []

    def test_synthesize_writes_a_file(self, world, voice_bytes, tmp_path):
        out = str(tmp_path / "out" / "chunk.wav")
        provider = _provider(world)
        path, duration = provider.synthesize("Hello there, world.", voice_bytes, out)
        assert path == out and os.path.getsize(out) > 44 and duration > 0

    def test_token_budget_scales_with_text_length(self, world, voice_bytes):
        provider = _provider(world)
        budgets = []
        for length in (100, 200, 400):
            provider.synthesize("x" * length, voice_bytes, return_bytes=True)
            budgets.append(world.models[0].generate_calls[-1]["max_new_tokens"])
        assert budgets[0] < budgets[1] < budgets[2]
        # 400 characters: about 28 s expected; the cap must be generous but far below
        # upstream's default budgets (4096 in the examples, 100000 in the 1.7B model).
        expected = 400 * mp._FRAMES_PER_CHAR
        assert 1.5 * expected < budgets[2] < 4 * expected
        assert budgets[2] < 1500

    def test_short_text_keeps_a_minimum_budget(self, world, voice_bytes):
        provider = _provider(world)
        provider.synthesize("Yes.", voice_bytes, return_bytes=True)
        assert world.models[0].generate_calls[0]["max_new_tokens"] >= mp._MIN_FRAME_CAP

    def test_batch_budget_is_the_longest_items(self, world, voice_bytes):
        provider = _provider(world)
        provider.synthesize("x" * 400, voice_bytes, return_bytes=True)
        single = world.models[0].generate_calls[-1]["max_new_tokens"]
        provider.synthesize_batch(["x" * 50, "x" * 400], voice_bytes)
        assert world.models[0].generate_calls[-1]["max_new_tokens"] == single

    def test_delay_model_gets_room_for_the_delay_flush(self, world, voice_bytes):
        local = _provider(world)
        local.synthesize("x" * 200, voice_bytes, return_bytes=True)
        delay = _provider(world, tts_model_name=_DELAY_V15)
        delay.synthesize("x" * 200, voice_bytes, return_bytes=True)
        local_budget = world.models[0].generate_calls[0]["max_new_tokens"]
        delay_budget = world.models[1].generate_calls[0]["max_new_tokens"]
        assert delay_budget - local_budget == 32 + 1

    def test_hard_limit_and_factor_options(self, world, voice_bytes):
        provider = _provider(world, tts_options={"max_new_tokens": 120})
        provider.synthesize("x" * 400, voice_bytes, return_bytes=True)
        assert world.models[0].generate_calls[0]["max_new_tokens"] == 120 + 1 + mp._BUDGET_SLACK_FRAMES

        provider.config = AudiobookConfig(tts_model_name="", tts_options={"max_duration_factor": 5})
        provider.synthesize("x" * 400, voice_bytes, return_bytes=True)
        assert world.models[0].generate_calls[1]["max_new_tokens"] > 4.5 * 400 * mp._FRAMES_PER_CHAR

    def test_speed_drives_duration_tokens(self, world, voice_bytes):
        text = "x" * 200
        tokens = {}
        for speed in (1.0, 1.25, 0.8):
            provider = _provider(world, speed=speed)
            provider.synthesize(text, voice_bytes, return_bytes=True)
            tokens[speed] = world.processors[-1].user_messages[0]["tokens"]
        assert tokens[1.0] is None, "natural pace leaves the length to the model"
        assert tokens[1.25] == round(200 * mp._FRAMES_PER_CHAR / 1.25)
        assert tokens[0.8] == round(200 * mp._FRAMES_PER_CHAR / 0.8)

    def test_duration_control_on_and_off(self, world, voice_bytes):
        provider = _provider(world, tts_options={"duration_control": "on", "duration_scale": 1.5})
        provider.synthesize_batch(["x" * 100, "x" * 300], voice_bytes)
        short, long = (m["tokens"] for m in world.processors[0].user_messages)
        assert short == round(100 * mp._FRAMES_PER_CHAR * 1.5)
        assert long == round(300 * mp._FRAMES_PER_CHAR * 1.5)

        provider.config = AudiobookConfig(tts_model_name="", speed=1.5, tts_options={"duration_control": "off"})
        provider.synthesize("x" * 100, voice_bytes, return_bytes=True)
        assert world.processors[0].user_messages[-1]["tokens"] is None

    def test_cjk_text_gets_a_larger_estimate(self):
        latin = mp._estimate_frames("a" * 40)
        cjk = mp._estimate_frames("你" * 40)
        assert cjk > 3 * latin
        assert latin == pytest.approx(40 * mp._FRAMES_PER_CHAR)

    def test_out_of_memory_falls_back_to_single_items(self, world, voice_bytes):
        world.oom_above = 1
        texts = ["A" * 20, "B" * 80, "C" * 40]
        provider = _provider(world)
        results = provider.synthesize_batch(texts, voice_bytes)
        assert [call["input_ids"][0] for call in world.models[0].generate_calls] == [3, 1, 1, 1]
        assert [_duration_frames(r) for r in results] == pytest.approx([_default_frames(t) for t in texts])

    def test_out_of_memory_on_a_single_item_is_a_clear_error(self, world, voice_bytes):
        provider = _provider(world)
        provider.ensure_ready()

        def always_oom(**kwargs):
            raise torch.cuda.OutOfMemoryError("CUDA out of memory (fake)")

        world.models[0].generate = always_oom
        with pytest.raises(RuntimeError, match="out of memory"):
            provider.synthesize("Hello there, world.", voice_bytes, return_bytes=True)


# ── Pause markers ─────────────────────────────────────────────────────────────


class TestPauseTags:
    def test_split_pauses(self):
        assert mp._split_pauses("One.[pause 1.5s] Two. [PAUSE 2s]") == ["One.", 1.5, "Two.", 2.0]
        assert mp._split_pauses("No tags here.") == ["No tags here."]

    def test_v1_model_gets_real_silence(self, world, voice_bytes):
        provider = _provider(world)
        text = "It was late.[pause 1.5s] Nobody spoke."
        result = provider.synthesize(text, voice_bytes, return_bytes=True)
        conversations = world.processors[0].calls[0]["conversations"]
        assert [conv[0]["text"] for conv in conversations] == ["It was late.", "Nobody spoke."]
        frames = _default_frames("It was late.") + _default_frames("Nobody spoke.")
        assert result[1] == pytest.approx(frames / _FPS + 1.5)
        audio, rate = _read(result)
        start = int(_default_frames("It was late.") / _FPS * rate)
        silence = audio[start + 10: start + int(1.5 * rate) - 10]
        assert abs(silence).max() == 0.0
        assert abs(audio[:start]).min() > 0 and abs(audio[-10:]).min() > 0

    def test_strip_removes_markers(self, world, voice_bytes):
        provider = _provider(world, tts_options={"pause_tags": "strip"})
        provider.synthesize("It was late.[pause 1.5s] Nobody spoke.", voice_bytes, return_bytes=True)
        assert world.processors[0].user_messages[0]["text"] == "It was late. Nobody spoke."

    def test_native_pause_extends_the_budget(self, world, voice_bytes):
        provider = _provider(world, tts_model_name=_DELAY_V15)
        provider.synthesize("x" * 100, voice_bytes, return_bytes=True)
        provider.synthesize("x" * 100 + "[pause 4.0s]", voice_bytes, return_bytes=True)
        plain, paused = (call["max_new_tokens"] for call in world.models[0].generate_calls)
        assert paused - plain >= int(4.0 * _FPS)

    def test_pause_only_text_raises(self, world, voice_bytes):
        provider = _provider(world)
        with pytest.raises(RuntimeError):
            provider.synthesize("[pause 2.0s]", voice_bytes, return_bytes=True)

    def test_whitespace_is_collapsed(self, world, voice_bytes):
        provider = _provider(world)
        provider.synthesize("Chapter one.\n\nThe door\topened.", voice_bytes, return_bytes=True)
        assert world.processors[0].user_messages[0]["text"] == "Chapter one. The door opened."


# ── Failure handling ──────────────────────────────────────────────────────────


class TestFailures:
    def test_empty_text_raises(self, world, voice_bytes):
        provider = _provider(world)
        with pytest.raises(RuntimeError, match="empty"):
            provider.synthesize("   \n ", voice_bytes, return_bytes=True)
        with pytest.raises(RuntimeError, match="empty"):
            provider.synthesize_batch(["Fine.", ""], voice_bytes)

    def test_no_audio_raises_after_retries(self, world, voice_bytes):
        world.no_audio = True
        provider = _provider(world, tts_options={"runaway_retries": 2})
        with pytest.raises(RuntimeError, match="produced no audio"):
            provider.synthesize("Hello there, world.", voice_bytes, return_bytes=True)
        assert len(world.models[0].generate_calls) == 3, "one take plus two retries"

    def test_no_audio_in_a_batch_is_retried_per_item(self, world, voice_bytes):
        world.no_audio = True
        provider = _provider(world, tts_options={"runaway_retries": 1})
        with pytest.raises(RuntimeError, match="produced no audio"):
            provider.synthesize_batch(["One chunk.", "Two chunks."], voice_bytes)
        assert [call["input_ids"][0] for call in world.models[0].generate_calls] == [2, 1]

    def test_nan_audio_raises(self, world, voice_bytes):
        world.nan_audio = True
        provider = _provider(world)
        with pytest.raises(RuntimeError, match="NaN"):
            provider.synthesize("Hello there, world.", voice_bytes, return_bytes=True)
        with pytest.raises(RuntimeError, match="NaN"):
            provider.synthesize_batch(["Hello there, world."], voice_bytes)

    @pytest.mark.parametrize("model_id", [_LOCAL, _LOCAL_V15, _DELAY_V15])
    def test_runaway_generation_raises(self, world, voice_bytes, model_id):
        world.runaway = True
        provider = _provider(world, tts_model_name=model_id, tts_options={"runaway_retries": 1})
        with pytest.raises(RuntimeError, match="runaway"):
            provider.synthesize("Hello there, world.", voice_bytes, return_bytes=True)
        assert len(world.models[0].generate_calls) == 2

    @pytest.mark.parametrize("model_id", [_LOCAL, _LOCAL_V15, _DELAY_V15])
    def test_a_take_that_ends_on_its_cap_is_not_a_runaway(self, world, voice_bytes, model_id):
        world.frames_for = lambda text: 10 ** 6       # clamp to the model's budget minus its overhead
        provider = _provider(world, tts_model_name=model_id)
        provider.ensure_ready()
        model = world.models[0]
        original = model._run
        # speak exactly frame_cap frames: budget minus overhead minus slack
        model._run = lambda budget, **kw: original(budget - mp._BUDGET_SLACK_FRAMES, **kw)
        _, duration = provider.synthesize("Hello there, world.", voice_bytes, return_bytes=True)
        assert duration * _FPS == pytest.approx(mp._MIN_FRAME_CAP)
        assert len(model.generate_calls) == 1

    def test_runaway_item_in_a_batch_is_regenerated_alone(self, world, voice_bytes):
        texts = ["x" * 40, "y" * 400]
        cap_short = max(mp._MIN_FRAME_CAP, int(40 * mp._FRAMES_PER_CHAR * 2.5) + 1)
        calls = {"count": 0}

        def frames(text: str) -> int:
            # the short item babbles past its own cap, but only in the batched take
            if text.startswith("x") and calls["count"] == 0:
                return cap_short + 40
            return _default_frames(text)

        world.frames_for = frames
        provider = _provider(world)
        provider.ensure_ready()
        model = world.models[0]
        original = model._run

        def counting_run(budget, **kw):
            result = original(budget, **kw)
            calls["count"] += 1
            return result

        model._run = counting_run
        results = provider.synthesize_batch(texts, voice_bytes)
        assert [call["input_ids"][0] for call in model.generate_calls] == [2, 1]
        assert [_duration_frames(r) for r in results] == pytest.approx([_default_frames(t) for t in texts])

    def test_decode_count_mismatch_raises(self, world, voice_bytes):
        provider = _provider(world)
        provider.ensure_ready()
        world.processors[0].decode = lambda output: []
        with pytest.raises(RuntimeError, match="decoded 0 results"):
            provider.synthesize_batch(["One chunk.", "Two chunks."], voice_bytes)


# ── Stereo ────────────────────────────────────────────────────────────────────


class TestStereo:
    def test_v15_stereo_is_downmixed_at_48k(self, world, voice_bytes):
        provider = _provider(world, tts_model_name=_LOCAL_V15)
        text = "A" * 40
        result = provider.synthesize(text, voice_bytes, return_bytes=True)
        audio, rate = _read(result)
        assert rate == 48000 and audio.ndim == 1
        assert len(audio) == _default_frames(text) * 3840
        # channels were (v, -0.5 v): their mean is 0.25 v
        assert audio[0] == pytest.approx(0.25 * 0.01, abs=1e-4)   # 16-bit PCM
        assert _duration_frames(result) == pytest.approx(_default_frames(text)), "no stop frame on v1.5"

    def test_v15_emulated_pause_keeps_the_channel_layout(self, world, voice_bytes):
        provider = _provider(world, tts_model_name=_LOCAL_V15, tts_options={"pause_tags": "emulate"})
        result = provider.synthesize("One.[pause 1.0s] Two.", voice_bytes, return_bytes=True)
        frames = _default_frames("One.") + _default_frames("Two.")
        assert result[1] == pytest.approx(frames / _FPS + 1.0)
