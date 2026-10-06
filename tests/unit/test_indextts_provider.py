"""
test_indextts_provider.py
=========================
Unit tests for the IndexTTS-2.5 / IndexTTS-2 provider.

No network, no GPU and no upstream package: a fake ``indextts`` package is
injected through ``sys.modules``. Its two ``IndexTTS2`` classes copy the
constructor and ``infer`` signatures of upstream's ``indextts/infer_v2_5.py``
and ``indextts/infer_v2.py`` (index-tts commit d9e41aac), including the
path-keyed conditioning cache and the ``(sample_rate, int16 (N, 1))`` return
value, so a wrong keyword argument fails here the way it would upstream.
"""
from __future__ import annotations

import io
import os
import subprocess
import sys
import tempfile
import types

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

np = pytest.importorskip("numpy")
sf = pytest.importorskip("soundfile")
torch = pytest.importorskip("torch")
huggingface_hub = pytest.importorskip("huggingface_hub")

from audiobook_factory.pipeline import AudiobookConfig  # noqa: E402
from audiobook_factory.tts_providers import indextts_provider as mod  # noqa: E402
from audiobook_factory.tts_providers.base_tts_provider import ProviderInfo, get_tts_provider  # noqa: E402
from audiobook_factory.tts_providers.indextts_provider import IndexTTSProvider  # noqa: E402

_SAMPLE_RATE = 22050
# Keyword arguments upstream pops from **generation_kwargs or forwards to
# UnifiedVoice.inference_speech(); anything else would reach HF generate().
_GENERATION_KWARGS = {
    "do_sample", "top_p", "top_k", "temperature", "length_penalty", "num_beams",
    "repetition_penalty", "max_mel_tokens", "typical_sampling", "typical_mass",
}


# ── Fake upstream ────────────────────────────────────────────────────────────

class _FakeNormalizer:
    """Mirrors the parts of indextts.utils.front.TextNormalizer the provider uses."""

    def __init__(self) -> None:
        self.term_glossary: dict = {}

    def load_glossary_from_yaml(self, glossary_path):
        if glossary_path and os.path.exists(glossary_path):
            glossary = {}
            with open(glossary_path, encoding="utf-8") as fh:
                for line in fh:
                    if ":" in line:
                        term, reading = line.split(":", 1)
                        glossary[term.strip()] = reading.strip()
            if glossary:
                self.term_glossary = glossary
                return True
        return False


class _FakeIndexTTSBase:
    """Behaviour shared by both fake upstream classes."""

    instances: list = []

    def _setup(self, init_kwargs: dict) -> None:
        self.init_kwargs = init_kwargs
        self.device = init_kwargs["device"]
        self.model_dir = init_kwargs["model_dir"]
        self.cfg = types.SimpleNamespace(gpt=types.SimpleNamespace(max_mel_tokens=1815, max_text_tokens=600))
        self.qwen_emo = object() if init_kwargs["use_qwen_emo"] else None
        self.cache_spk_cond = None
        self.cache_spk_audio_prompt = None
        self.cache_emo_cond = None
        self.cache_emo_audio_prompt = None
        self.spk_conditioning_calls = 0
        self.emo_conditioning_calls = 0
        self.calls: list[dict] = []
        self.low_vram_seen: list = []
        self.results: list = []        # queued return values / exceptions
        type(self).instances.append(self)

    def normalize_emo_vec(self, emo_vector, apply_bias=True):
        if apply_bias:
            emo_bias = [0.9375, 0.875, 1.0, 1.0, 0.9375, 0.9375, 0.6875, 0.5625]
            emo_vector = [vec * bias for vec, bias in zip(emo_vector, emo_bias)]
        emo_sum = sum(emo_vector)
        if emo_sum > 0.8:
            scale_factor = 0.8 / emo_sum
            emo_vector = [vec * scale_factor for vec in emo_vector]
        return emo_vector

    def _infer(self, named: dict, generation_kwargs: dict):
        unknown = set(generation_kwargs) - _GENERATION_KWARGS
        if unknown:
            raise TypeError(f"generate() got unexpected keyword arguments {sorted(unknown)}")
        self.calls.append({**named, **generation_kwargs})
        self.low_vram_seen.append(getattr(self, "low_vram", None))
        if self.results:
            queued = self.results.pop(0)
            if isinstance(queued, BaseException):
                raise queued
            return queued
        if named["use_emo_text"] and self.qwen_emo is None:
            raise RuntimeError("use_emo_text=True requires QwenEmotion, but it was not loaded at init")
        spk, emo = named["spk_audio_prompt"], named["emo_audio_prompt"]
        if named["use_emo_text"] or named["emo_vector"] is not None:
            emo = None
        if emo is None:
            emo = spk
        # Same cache rule as upstream: recompute only when the path changed.
        if self.cache_spk_cond is None or self.cache_spk_audio_prompt != spk:
            assert os.path.isfile(spk), "speaker prompt must exist when it is conditioned"
            self.spk_conditioning_calls += 1
            self.cache_spk_cond, self.cache_spk_audio_prompt = object(), spk
        if self.cache_emo_cond is None or self.cache_emo_audio_prompt != emo:
            self.emo_conditioning_calls += 1
            self.cache_emo_cond, self.cache_emo_audio_prompt = object(), emo
        assert named["output_path"] is None
        samples = np.full((100 * len(named["text"]), 1), 8000, dtype=np.int16)
        return (_SAMPLE_RATE, samples)


def _make_fake_classes():
    class FakeV25(_FakeIndexTTSBase):
        instances: list = []

        def __init__(
                self, cfg_path="checkpoints/config.yaml", model_dir="checkpoints", use_bf16=False, device=None,
                use_cuda_kernel=None, use_deepspeed=False, use_accel=False, use_torch_compile=False,
                use_qwen_emo=False
        ):
            self._setup(dict(locals()))
            self.low_vram = False
            self.text_process = _FakeNormalizer()

        def infer(self, spk_audio_prompt, text, output_path, lang,
                  emo_audio_prompt=None, emo_alpha=1.0,
                  emo_vector=None, use_emo_text=False, emo_text=None, use_random=False, interval_silence=200,
                  verbose=False, max_text_tokens_per_segment=120, stream_return=False, more_segment_before=0,
                  duration_factor=1.0, text_normalization=True, **generation_kwargs):
            named = {k: v for k, v in locals().items() if k not in ("self", "generation_kwargs")}
            return self._infer(named, generation_kwargs)

    class FakeV2(_FakeIndexTTSBase):
        instances: list = []

        def __init__(
                self, cfg_path="checkpoints/config.yaml", model_dir="checkpoints", use_fp16=False, device=None,
                use_cuda_kernel=None, use_deepspeed=False, use_accel=False, use_torch_compile=False,
                use_qwen_emo=True, aux_paths=None
        ):
            self._setup(dict(locals()))
            self.normalizer = _FakeNormalizer()

        def infer(self, spk_audio_prompt, text, output_path,
                  emo_audio_prompt=None, emo_alpha=1.0,
                  emo_vector=None,
                  use_emo_text=False, emo_text=None, use_random=False, interval_silence=200,
                  verbose=False, max_text_tokens_per_segment=120, stream_return=False, more_segment_before=0,
                  **generation_kwargs):
            named = {k: v for k, v in locals().items() if k not in ("self", "generation_kwargs")}
            return self._infer(named, generation_kwargs)

    return FakeV25, FakeV2


@pytest.fixture
def upstream(monkeypatch, tmp_path):
    """Installs the fake ``indextts`` package and a fake ``snapshot_download``."""
    fake_v25, fake_v2 = _make_fake_classes()
    package = types.ModuleType("indextts")
    package.__path__ = []  # type: ignore[attr-defined]
    module_v25 = types.ModuleType("indextts.infer_v2_5")
    module_v25.IndexTTS2 = fake_v25  # type: ignore[attr-defined]
    module_v2 = types.ModuleType("indextts.infer_v2")
    module_v2.IndexTTS2 = fake_v2  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "indextts", package)
    monkeypatch.setitem(sys.modules, "indextts.infer_v2_5", module_v25)
    monkeypatch.setitem(sys.modules, "indextts.infer_v2", module_v2)
    monkeypatch.delenv("ABM_INDEXTTS_REPO", raising=False)
    # Voice references are written under the system temp dir; keep them in tmp_path.
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))

    downloads: list[tuple[str, dict]] = []

    def fake_snapshot_download(repo_id, **kwargs):
        downloads.append((repo_id, kwargs))
        target = tmp_path / "hub" / repo_id.replace("/", "--")
        target.mkdir(parents=True, exist_ok=True)
        return str(target)

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
    return types.SimpleNamespace(v25=fake_v25, v2=fake_v2, downloads=downloads, tmp_path=tmp_path)


@pytest.fixture
def voice_bytes():
    return b"RIFF" + bytes(range(256)) * 8


def _config(**overrides) -> AudiobookConfig:
    values = dict(
        language="English", tts_provider_name="indextts", tts_model_name="IndexTeam/IndexTTS-2.5",
        temperature=0.7, top_p=0.9, top_k=40, speed=1.0, seed=-1,
    )
    values.update(overrides)
    return AudiobookConfig(**values)


def _read_wav(wav_bytes: bytes):
    samples, rate = sf.read(io.BytesIO(wav_bytes), dtype="float32")
    return samples, rate


# ── INFO ─────────────────────────────────────────────────────────────────────

class TestInfo:

    def test_info_is_complete_and_consistent(self):
        info = IndexTTSProvider.info()
        assert isinstance(info, ProviderInfo)
        assert info.name == "indextts"
        assert info.default_model == "IndexTeam/IndexTTS-2.5"
        assert info.default_model in info.models
        assert set(info.models) == {"IndexTeam/IndexTTS-2.5", "IndexTeam/IndexTTS-2"}
        assert set(info.models) == set(mod._UPSTREAM_MODULES) == set(mod._HALF_DTYPES) == set(mod._MODEL_LANGUAGES)
        assert info.native_sample_rate == 22050
        assert info.min_vram_gb > 0
        assert info.transcript == "unused"
        assert info.supports_voice_clone and info.supports_speed and info.supports_seed
        assert info.supports_batch is False
        assert info.preset_voices == ()
        assert info.languages == ("Chinese", "English", "Japanese", "Spanish", "Arabic")
        assert info.license and info.homepage.startswith("https://")
        assert "transformers==4.52.1" in info.pip_requirements
        assert "ABM_INDEXTTS_REPO" in info.install_notes
        assert mod._INSTALL_COMMAND in info.install_notes

    def test_every_option_has_a_usable_default(self):
        info = IndexTTSProvider.info()
        keys = [o.key for o in info.options]
        assert len(keys) == len(set(keys)), "duplicate option keys"
        for option in info.options:
            assert option.kind in {"float", "int", "bool", "str", "choice", "file"}
            assert option.default is not None, option.key
            assert option.label and option.help, option.key
            if option.kind == "choice":
                assert option.default in option.choices, option.key
            if option.kind in {"float", "int"}:
                assert option.minimum is not None and option.maximum is not None, option.key
                assert option.minimum <= option.default <= option.maximum, option.key
        assert set(info.option_defaults()) == set(keys)
        for expected in ("emotion_mode", "emo_audio_prompt", "emo_vector", "emo_alpha", "interval_silence",
                         "max_text_tokens_per_segment", "max_mel_tokens", "precision", "use_cuda_kernel",
                         "use_deepspeed", "use_accel", "use_torch_compile"):
            assert expected in keys
        assert next(o for o in info.options if o.key == "emo_audio_prompt").kind == "file"

    def test_requirements_file_matches_info(self):
        path = os.path.join(_ROOT, "requirements", "tts-indextts.txt")
        with open(path, encoding="utf-8") as fh:
            text = fh.read()
        lines = [ln.split(" #")[0].strip() for ln in text.splitlines()]
        requirements = {ln for ln in lines if ln and not ln.startswith("#")}
        for requirement in IndexTTSProvider.info().pip_requirements:
            assert requirement in requirements, requirement
        # The upstream package itself must not be a plain requirement line:
        # that would drag in its torch / numpy / keras pins.
        assert not any("index-tts" in ln for ln in requirements)
        assert mod._UPSTREAM_COMMIT in text

    def test_module_imports_without_torch_or_upstream(self):
        code = (
            "import sys\n"
            "import audiobook_factory.tts_providers.base_tts_provider\n"
            "had_torch = 'torch' in sys.modules\n"
            "from audiobook_factory.tts_providers.indextts_provider import IndexTTSProvider\n"
            "assert IndexTTSProvider.info().default_model == 'IndexTeam/IndexTTS-2.5'\n"
            "assert ('torch' in sys.modules) == had_torch, 'torch imported at module import'\n"
            "assert 'indextts' not in sys.modules\n"
            "assert 'huggingface_hub' not in sys.modules or had_torch\n"
        )
        result = subprocess.run(
            [sys.executable, "-c", code], cwd=_ROOT, capture_output=True, text=True, timeout=120,
        )
        assert result.returncode == 0, result.stderr

    def test_registry_resolves_provider(self):
        provider = get_tts_provider("indextts", _config(), device="cpu")
        assert isinstance(provider, IndexTTSProvider)
        assert provider.device == "cpu"
        assert provider.is_ready is False
        assert provider.get_name() == "IndexTTS-2.5"
        assert isinstance(get_tts_provider("index-tts-2.5", _config(), device="cpu"), IndexTTSProvider)

    def test_bare_cuda_is_bound_to_one_device(self):
        assert IndexTTSProvider(_config(device="cuda")).device == "cuda:0"
        assert IndexTTSProvider(_config(), device="cuda:1").device == "cuda:1"


# ── Loading ──────────────────────────────────────────────────────────────────

class TestLoading:

    def test_missing_package_error_names_install_command(self, monkeypatch):
        for name in [m for m in sys.modules if m == "indextts" or m.startswith("indextts.")]:
            monkeypatch.delitem(sys.modules, name)
        monkeypatch.delenv("ABM_INDEXTTS_REPO", raising=False)
        monkeypatch.setattr(
            huggingface_hub, "snapshot_download",
            lambda *a, **k: pytest.fail("must not download before the package import succeeded"),
        )
        provider = IndexTTSProvider(_config(), device="cpu")
        with pytest.raises(RuntimeError) as excinfo:
            provider.ensure_ready()
        message = str(excinfo.value)
        assert mod._INSTALL_COMMAND in message
        assert "pip install -r requirements/tts-indextts.txt" in message
        assert "git+https://github.com/index-tts/index-tts.git" in message
        assert "ABM_INDEXTTS_REPO" in message
        assert provider.is_ready is False

    def test_broken_dependency_error_names_install_command(self, monkeypatch, tmp_path):
        repo = tmp_path / "index-tts"
        (repo / "indextts").mkdir(parents=True)
        (repo / "indextts" / "__init__.py").write_text("")
        (repo / "indextts" / "infer_v2_5.py").write_text("import abm_missing_dependency_xyz\n")
        for name in [m for m in sys.modules if m == "indextts" or m.startswith("indextts.")]:
            monkeypatch.delitem(sys.modules, name)
        monkeypatch.setattr(sys, "path", list(sys.path))
        monkeypatch.setenv("ABM_INDEXTTS_REPO", str(repo))
        provider = IndexTTSProvider(_config(), device="cpu")
        try:
            with pytest.raises(RuntimeError) as excinfo:
                provider.ensure_ready()
        finally:
            for name in [m for m in sys.modules if m == "indextts" or m.startswith("indextts.")]:
                sys.modules.pop(name, None)
        assert "abm_missing_dependency_xyz" in str(excinfo.value)
        assert "transformers==4.52.1" in str(excinfo.value)
        assert mod._INSTALL_COMMAND in str(excinfo.value)

    def test_invalid_repo_env_var_is_reported(self, monkeypatch, tmp_path):
        monkeypatch.setenv("ABM_INDEXTTS_REPO", str(tmp_path / "nowhere"))
        provider = IndexTTSProvider(_config(), device="cpu")
        with pytest.raises(RuntimeError, match="ABM_INDEXTTS_REPO"):
            provider.ensure_ready()

    def test_repo_env_var_imports_clone_and_restores_hf_cache_env(self, monkeypatch, tmp_path, voice_bytes):
        """A clone is importable through ABM_INDEXTTS_REPO; upstream's env edit is undone."""
        repo = tmp_path / "index-tts"
        (repo / "indextts").mkdir(parents=True)
        (repo / "tests").mkdir()
        (repo / "tests" / "__init__.py").write_text("SHADOW = True\n")
        (repo / "indextts" / "__init__.py").write_text("")
        # First line mirrors upstream indextts/infer_v2_5.py line 4.
        (repo / "indextts" / "infer_v2_5.py").write_text(
            "import os\n"
            "os.environ['HF_HUB_CACHE'] = './checkpoints/hf_cache'\n"
            "import numpy as np\n"
            "class IndexTTS2:\n"
            "    def __init__(self, cfg_path='checkpoints/config.yaml', model_dir='checkpoints', use_bf16=False,\n"
            "                 device=None, use_cuda_kernel=None, use_deepspeed=False, use_accel=False,\n"
            "                 use_torch_compile=False, use_qwen_emo=False):\n"
            "        self.cfg_path, self.model_dir = cfg_path, model_dir\n"
            "    def infer(self, spk_audio_prompt, text, output_path, lang, emo_audio_prompt=None, emo_alpha=1.0,\n"
            "              emo_vector=None, use_emo_text=False, emo_text=None, use_random=False,\n"
            "              interval_silence=200, verbose=False, max_text_tokens_per_segment=120,\n"
            "              stream_return=False, more_segment_before=0, duration_factor=1.0,\n"
            "              text_normalization=True, **generation_kwargs):\n"
            "        return (22050, np.full((2205, 1), 4000, dtype=np.int16))\n"
        )
        for name in [m for m in sys.modules if m == "indextts" or m.startswith("indextts.")]:
            monkeypatch.delitem(sys.modules, name)
        monkeypatch.setattr(sys, "path", list(sys.path))
        monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
        monkeypatch.setenv("ABM_INDEXTTS_REPO", str(repo))
        monkeypatch.setenv("HF_HUB_CACHE", "/original/hub/cache")
        monkeypatch.setattr(huggingface_hub, "snapshot_download", lambda repo_id, **k: str(tmp_path / "ckpt"))

        provider = IndexTTSProvider(_config(), device="cpu")
        try:
            provider.ensure_ready()
            assert provider.is_ready
            assert os.environ["HF_HUB_CACHE"] == "/original/hub/cache"
            assert provider._model.model_dir == str(tmp_path / "ckpt")
            assert provider._model.cfg_path == os.path.join(str(tmp_path / "ckpt"), "config.yaml")
            # Appended, so the clone's top-level 'tests' package cannot shadow ours.
            assert sys.path[-1] == str(repo) and sys.path[0] != str(repo)
            import tests
            assert not hasattr(tests, "SHADOW")
            _, duration = provider.synthesize("Hello.", voice_bytes, return_bytes=True)
            assert duration == pytest.approx(0.1)
        finally:
            for name in [m for m in sys.modules if m == "indextts" or m.startswith("indextts.")]:
                sys.modules.pop(name, None)

        monkeypatch.delenv("HF_HUB_CACHE")
        fresh = IndexTTSProvider(_config(), device="cpu")
        try:
            fresh.ensure_ready()
            assert "HF_HUB_CACHE" not in os.environ
        finally:
            for name in [m for m in sys.modules if m == "indextts" or m.startswith("indextts.")]:
                sys.modules.pop(name, None)

    def test_default_load_arguments(self, upstream):
        provider = IndexTTSProvider(_config(), device="cpu")
        provider.ensure_ready()
        provider.ensure_ready()  # idempotent
        assert len(upstream.v25.instances) == 1 and not upstream.v2.instances
        init = upstream.v25.instances[0].init_kwargs
        repo_id, download_kwargs = upstream.downloads[0]
        assert repo_id == "IndexTeam/IndexTTS-2.5"
        assert download_kwargs == {"ignore_patterns": ["qwen0.6bemo4-merge/*"]}
        assert len(upstream.downloads) == 1
        assert init["model_dir"].endswith("IndexTeam--IndexTTS-2.5")
        assert init["cfg_path"] == os.path.join(init["model_dir"], "config.yaml")
        assert init["device"] == "cpu"
        assert init["use_bf16"] is False            # CPU is always full precision
        assert init["use_cuda_kernel"] is False
        assert init["use_deepspeed"] is False and init["use_accel"] is False
        assert init["use_torch_compile"] is False and init["use_qwen_emo"] is False
        assert provider.is_ready

    def test_foreign_model_name_falls_back_to_default(self, upstream):
        provider = IndexTTSProvider(_config(tts_model_name="Qwen/Qwen3-TTS-12Hz-1.7B-Base"), device="cpu")
        assert provider.resolve_model_id() == "IndexTeam/IndexTTS-2.5"
        provider.ensure_ready()
        assert upstream.downloads[0][0] == "IndexTeam/IndexTTS-2.5"
        assert len(upstream.v25.instances) == 1 and not upstream.v2.instances

        provider.config = _config(tts_model_name="some-user/evil-repo")
        provider.ensure_ready()
        assert len(upstream.v25.instances) == 1, "an unknown repo must not trigger a reload"
        assert [d[0] for d in upstream.downloads] == ["IndexTeam/IndexTTS-2.5"]

    def test_acceleration_toggles_reach_constructor(self, upstream):
        config = _config(tts_options={
            "use_cuda_kernel": True, "use_deepspeed": "true", "use_accel": True, "use_torch_compile": True,
        })
        provider = IndexTTSProvider(config, device="cuda:1", dtype_override="bfloat16")
        provider.ensure_ready()
        init = upstream.v25.instances[0].init_kwargs
        assert init["device"] == "cuda:1"
        assert init["use_bf16"] is True
        assert init["use_cuda_kernel"] is True and init["use_deepspeed"] is True
        assert init["use_accel"] is True and init["use_torch_compile"] is True

    def test_common_torch_compile_field_is_honoured(self, upstream):
        provider = IndexTTSProvider(_config(torch_compile=True), device="cpu")
        provider.ensure_ready()
        assert upstream.v25.instances[0].init_kwargs["use_torch_compile"] is True

    @pytest.mark.parametrize(
        ("model", "dtype_override", "precision_option", "expected"),
        [
            # IndexTTS-2 implements FP16 only; default on CUDA is FP16.
            ("IndexTeam/IndexTTS-2", None, "auto", {"use_fp16": True}),
            ("IndexTeam/IndexTTS-2", "float16", "auto", {"use_fp16": True}),
            ("IndexTeam/IndexTTS-2", "bfloat16", "auto", {"use_fp16": True}),
            ("IndexTeam/IndexTTS-2", "float32", "auto", {"use_fp16": False}),
            ("IndexTeam/IndexTTS-2", None, "full", {"use_fp16": False}),
            # IndexTTS-2.5 implements BF16 only. This test machine has no
            # CUDA, i.e. no native BF16: auto / float16 stay in FP32.
            ("IndexTeam/IndexTTS-2.5", None, "auto", {"use_bf16": False}),
            ("IndexTeam/IndexTTS-2.5", "float16", "auto", {"use_bf16": False}),
            ("IndexTeam/IndexTTS-2.5", "bfloat16", "auto", {"use_bf16": True}),
            ("IndexTeam/IndexTTS-2.5", None, "half", {"use_bf16": True}),
            ("IndexTeam/IndexTTS-2.5", "float32", "half", {"use_bf16": False}),
        ],
    )
    def test_precision_mapping(self, upstream, model, dtype_override, precision_option, expected):
        config = _config(tts_model_name=model, tts_options={"precision": precision_option})
        provider = IndexTTSProvider(config, device="cuda:0", dtype_override=dtype_override)
        provider.ensure_ready()
        fake = upstream.v2 if model.endswith("TTS-2") else upstream.v25
        init = fake.instances[0].init_kwargs
        for key, value in expected.items():
            assert init[key] is value
        assert "use_fp16" not in init or "use_bf16" not in init

    @pytest.mark.parametrize(("major", "total_gb", "expected"), [(8, 24, True), (7, 15, False), (7, 8, True)])
    def test_auto_precision_for_v25_follows_gpu(self, upstream, monkeypatch, major, total_gb, expected):
        """BF16 on Ampere+ or on small cards; FP32 on a T4 (capability 7.5, 15 GB)."""
        properties = types.SimpleNamespace(major=major, total_memory=total_gb * 1024 ** 3)
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "get_device_properties", lambda index: properties)
        monkeypatch.setattr(torch.cuda, "set_device", lambda index: None)
        monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
        provider = IndexTTSProvider(_config(), device="cuda:0")
        provider.ensure_ready()
        assert upstream.v25.instances[0].init_kwargs["use_bf16"] is expected

    def test_model_change_reloads(self, upstream, voice_bytes):
        provider = IndexTTSProvider(_config(), device="cpu")
        provider.synthesize("One.", voice_bytes, return_bytes=True)
        assert len(upstream.v25.instances) == 1 and provider.get_name() == "IndexTTS-2.5"

        provider.config = _config(tts_model_name="IndexTeam/IndexTTS-2")
        provider.synthesize("Two.", voice_bytes, return_bytes=True)
        assert len(upstream.v2.instances) == 1
        assert [d[0] for d in upstream.downloads] == ["IndexTeam/IndexTTS-2.5", "IndexTeam/IndexTTS-2"]
        assert provider._model is upstream.v2.instances[0]
        assert provider.get_name() == "IndexTTS-2"
        # IndexTTS-2 defaults to use_qwen_emo=True upstream; the provider turns it off.
        assert upstream.v2.instances[0].init_kwargs["use_qwen_emo"] is False
        assert upstream.v2.instances[0].init_kwargs["aux_paths"] is None

    def test_text_emotion_loads_qwen_emo_and_keeps_it(self, upstream, voice_bytes):
        provider = IndexTTSProvider(_config(), device="cpu")
        provider.synthesize("One.", voice_bytes, return_bytes=True)
        assert upstream.v25.instances[0].init_kwargs["use_qwen_emo"] is False

        provider.config = _config(tts_instruct="calm and a little sad")
        provider.synthesize("Two.", voice_bytes, return_bytes=True)
        assert len(upstream.v25.instances) == 2
        assert upstream.v25.instances[1].init_kwargs["use_qwen_emo"] is True
        assert upstream.downloads[1] == ("IndexTeam/IndexTTS-2.5", {"ignore_patterns": None})
        call = upstream.v25.instances[1].calls[0]
        assert call["use_emo_text"] is True and call["emo_text"] == "calm and a little sad"
        assert call["emo_alpha"] == pytest.approx(0.65)

        # Going back to speaker emotion must not throw the loaded model away.
        provider.config = _config()
        provider.synthesize("Three.", voice_bytes, return_bytes=True)
        assert len(upstream.v25.instances) == 2
        assert upstream.v25.instances[1].calls[1]["use_emo_text"] is False

    def test_qwen_emotion_model_is_pinned_to_provider_device(self, upstream):
        moved: list[str] = []

        class FakeQwen:
            def __init__(self):
                self.model = types.SimpleNamespace(to=moved.append)

        original_setup = _FakeIndexTTSBase._setup

        def setup(self, init_kwargs):
            original_setup(self, init_kwargs)
            self.qwen_emo = FakeQwen() if init_kwargs["use_qwen_emo"] else None

        upstream.v25._setup = setup
        provider = IndexTTSProvider(_config(tts_options={"emotion_mode": "script"}), device="cuda:1")
        provider.ensure_ready()
        assert moved == ["cuda:1"]

    def test_constructor_failure_raises_and_stays_unloaded(self, upstream):
        def boom(self, *args, **kwargs):
            raise FileNotFoundError("gpt.pth")

        upstream.v25.__init__ = boom
        provider = IndexTTSProvider(_config(), device="cpu")
        with pytest.raises(RuntimeError, match="failed to load IndexTeam/IndexTTS-2.5"):
            provider.ensure_ready()
        assert provider.is_ready is False

    def test_cleanup_drops_model_and_empties_cuda_cache(self, upstream, monkeypatch, voice_bytes):
        emptied: list[bool] = []
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "empty_cache", lambda: emptied.append(True))
        monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda seed: None)
        provider = IndexTTSProvider(_config(), device="cpu")
        provider.synthesize("One.", voice_bytes, return_bytes=True)
        assert provider.is_ready
        provider.cleanup()
        assert provider.is_ready is False and provider._model is None
        assert emptied
        provider.cleanup()  # idempotent
        provider.synthesize("Two.", voice_bytes, return_bytes=True)  # loads again on demand
        assert len(upstream.v25.instances) == 2


# ── Arguments passed to upstream infer() ─────────────────────────────────────

class TestInferArguments:

    def _call(self, upstream, voice_bytes, config, text="Hello there, this is a test sentence.", fake=None):
        provider = IndexTTSProvider(config, device="cpu")
        provider.synthesize(text, voice_bytes, return_bytes=True)
        return (fake or upstream.v25).instances[-1].calls[-1]

    def test_defaults_match_upstream_api(self, upstream, voice_bytes):
        call = self._call(upstream, voice_bytes, _config())
        assert call["text"] == "Hello there, this is a test sentence."
        assert os.path.isfile(call["spk_audio_prompt"])
        assert call["output_path"] is None
        assert call["lang"] == "EN"
        assert call["emo_audio_prompt"] is None and call["emo_vector"] is None
        assert call["use_emo_text"] is False and call["emo_text"] is None
        assert call["emo_alpha"] == 1.0 and call["use_random"] is False   # untouched upstream defaults
        assert call["interval_silence"] == 200
        assert call["max_text_tokens_per_segment"] == 120
        assert call["stream_return"] is False
        assert call["duration_factor"] == 1.0
        assert call["text_normalization"] is True
        assert call["verbose"] is False
        # common config fields
        assert call["temperature"] == pytest.approx(0.7)
        assert call["top_p"] == pytest.approx(0.9)
        assert call["top_k"] == 40
        # IndexTTS-specific sampling defaults (upstream's own)
        assert call["do_sample"] is True
        assert call["num_beams"] == 3
        assert call["repetition_penalty"] == pytest.approx(10.0)
        assert call["length_penalty"] == pytest.approx(0.0)
        assert "typical_sampling" not in call

    def test_shared_repetition_penalty_is_not_forwarded(self, upstream, voice_bytes):
        call = self._call(upstream, voice_bytes, _config(repetition_penalty=1.05))
        assert call["repetition_penalty"] == pytest.approx(10.0)
        call = self._call(upstream, voice_bytes, _config(tts_options={"repetition_penalty": "6.5"}))
        assert call["repetition_penalty"] == pytest.approx(6.5)

    def test_tts_options_are_forwarded(self, upstream, voice_bytes):
        config = _config(
            language="Chinese", speed=1.25, temperature=1.1, top_p=0.5, top_k=0,
            tts_options={
                "interval_silence": 80, "max_text_tokens_per_segment": "90", "max_mel_tokens": 900,
                "adaptive_max_mel_tokens": False, "do_sample": False, "num_beams": 5,
                "length_penalty": 0.5, "typical_sampling": True, "typical_mass": 0.8,
                "text_normalization": False, "verbose": True,
            },
        )
        call = self._call(upstream, voice_bytes, config)
        assert call["lang"] == "ZH"
        assert call["duration_factor"] == pytest.approx(0.8)      # speed 1.25 → 1 / 1.25
        assert call["temperature"] == pytest.approx(1.1) and call["top_p"] == pytest.approx(0.5)
        assert call["top_k"] is None                               # upstream's WebUI maps 0 → None
        assert call["interval_silence"] == 80
        assert call["max_text_tokens_per_segment"] == 90
        assert call["max_mel_tokens"] == 900
        assert call["do_sample"] is False and call["num_beams"] == 5
        assert call["length_penalty"] == pytest.approx(0.5)
        assert call["typical_sampling"] is True and call["typical_mass"] == pytest.approx(0.8)
        assert call["text_normalization"] is False and call["verbose"] is True

    def test_out_of_range_values_are_clamped(self, upstream, voice_bytes):
        config = _config(
            speed=5.0, temperature=0.0, top_p=3.0,
            tts_options={"max_text_tokens_per_segment": 5000, "max_mel_tokens": 99999,
                         "adaptive_max_mel_tokens": False, "num_beams": 0, "emo_alpha": 7},
        )
        call = self._call(upstream, voice_bytes, config)
        assert call["duration_factor"] == pytest.approx(0.5)       # upstream range 0.5 - 2.0
        assert call["temperature"] == pytest.approx(0.1)
        assert call["top_p"] == pytest.approx(1.0)
        assert call["max_text_tokens_per_segment"] == 600          # cfg.gpt.max_text_tokens
        assert call["max_mel_tokens"] == 1815                      # cfg.gpt.max_mel_tokens
        assert call["num_beams"] == 1
        slow = self._call(upstream, voice_bytes, _config(speed=0.25))
        assert slow["duration_factor"] == pytest.approx(2.0)

    def test_runaway_bound_scales_with_text_length(self, upstream, voice_bytes):
        short = self._call(upstream, voice_bytes, _config(), text="Chapter One.")
        assert short["max_mel_tokens"] == 200 + 10 * len("Chapter One.")
        assert short["max_mel_tokens"] < 1815
        long_text = ("The quick brown fox jumps over the lazy dog. " * 9)[:399]
        full = self._call(upstream, voice_bytes, _config(), text=long_text)
        assert full["max_mel_tokens"] == 1815, "a full 399-character chunk keeps the model limit"
        # Digits and CJK characters take far longer to say than their length suggests.
        assert mod._estimate_mel_token_bound("1,234,567") >= 500
        assert mod._estimate_mel_token_bound("克" * 50) >= 50 * 25
        # ... and the bound is always generous: 3 tokens/char is a normal English reading rate.
        for text in ("Yes.", "Chapter One.", "It was the best of times, it was the worst of times."):
            assert mod._estimate_mel_token_bound(text) >= 2 * 3.4 * len(text)

    def test_emotion_reference_audio(self, upstream, voice_bytes, tmp_path):
        emo = tmp_path / "sad.flac"
        emo.write_bytes(b"fLaC" + b"\x01" * 500)
        config = _config(tts_options={"emo_audio_prompt": str(emo), "emo_alpha": 0.4})
        call = self._call(upstream, voice_bytes, config)
        assert call["emo_audio_prompt"] != str(emo), "must be a content-addressed copy"
        assert call["emo_audio_prompt"].endswith(".flac")
        with open(call["emo_audio_prompt"], "rb") as fh:
            assert fh.read() == emo.read_bytes()
        assert call["emo_alpha"] == pytest.approx(0.4)
        assert call["emo_vector"] is None and call["use_emo_text"] is False
        assert upstream.v25.instances[0].init_kwargs["use_qwen_emo"] is False

    def test_emotion_vector_is_parsed_and_normalised_like_the_webui(self, upstream, voice_bytes):
        config = _config(tts_options={"emo_vector": "0, 0, 0.8, 0, 0, 0, 0, 0.6", "use_random": True})
        call = self._call(upstream, voice_bytes, config)
        biased = [0.0, 0.0, 0.8, 0.0, 0.0, 0.0, 0.0, 0.6 * 0.5625]
        scale = 0.8 / sum(biased)
        assert call["emo_vector"] == pytest.approx([v * scale for v in biased])
        assert sum(call["emo_vector"]) == pytest.approx(0.8)
        assert call["use_random"] is True
        assert call["emo_audio_prompt"] is None and call["use_emo_text"] is False

        as_list = self._call(upstream, voice_bytes, _config(tts_options={"emo_vector": [0, 0, 0.5, 0, 0, 0, 0, 0]}))
        assert as_list["emo_vector"] == pytest.approx([0, 0, 0.5, 0, 0, 0, 0, 0])

    @pytest.mark.parametrize("bad", ["0,0,1", "a,b,c,d,e,f,g,h", "0,0,5,0,0,0,0,0", "0,0,-1,0,0,0,0,0"])
    def test_invalid_emotion_vector_raises(self, upstream, voice_bytes, bad):
        provider = IndexTTSProvider(_config(tts_options={"emo_vector": bad}), device="cpu")
        with pytest.raises(RuntimeError, match="emo_vector"):
            provider.synthesize("Hello.", voice_bytes, return_bytes=True)

    def test_emotion_precedence_and_explicit_modes(self, upstream, voice_bytes, tmp_path):
        emo = tmp_path / "angry.wav"
        emo.write_bytes(b"RIFF" + b"\x02" * 500)
        options = {"emo_audio_prompt": str(emo), "emo_vector": "0.5,0,0,0,0,0,0,0"}
        # auto: text beats vector beats audio, as upstream resolves conflicts.
        call = self._call(upstream, voice_bytes, _config(tts_instruct="terrified", tts_options=dict(options)))
        assert call["use_emo_text"] is True and call["emo_text"] == "terrified"
        assert call["emo_vector"] is None and call["emo_audio_prompt"] is None
        call = self._call(upstream, voice_bytes, _config(tts_options=dict(options)))
        assert call["emo_vector"] is not None and call["emo_audio_prompt"] is None
        # explicit modes override the precedence
        call = self._call(upstream, voice_bytes, _config(tts_options={**options, "emotion_mode": "audio"}))
        assert call["emo_audio_prompt"] and call["emo_vector"] is None
        call = self._call(
            upstream, voice_bytes,
            _config(tts_instruct="terrified", tts_options={**options, "emotion_mode": "speaker"}),
        )
        assert call["emo_audio_prompt"] is None and call["emo_vector"] is None and call["use_emo_text"] is False
        assert upstream.v25.instances[-1].init_kwargs["use_qwen_emo"] is False
        call = self._call(upstream, voice_bytes, _config(tts_options={"emotion_mode": "script"}))
        assert call["use_emo_text"] is True and call["emo_text"] is None
        assert upstream.v25.instances[-1].init_kwargs["use_qwen_emo"] is True

    @pytest.mark.parametrize("mode", ["audio", "vector", "text"])
    def test_explicit_emotion_mode_without_input_raises(self, upstream, voice_bytes, mode):
        provider = IndexTTSProvider(_config(tts_options={"emotion_mode": mode}), device="cpu")
        with pytest.raises(RuntimeError, match="emotion_mode"):
            provider.synthesize("Hello.", voice_bytes, return_bytes=True)

    @pytest.mark.parametrize(
        ("language", "code"),
        [("English", "EN"), ("Chinese", "ZH"), ("Japanese", "JA"), ("Spanish", "ES"), ("Arabic", "AR"),
         ("en", "EN"), ("zh-CN", "ZH"), ("  mandarin ", "ZH")],
    )
    def test_language_mapping(self, upstream, voice_bytes, language, code):
        assert self._call(upstream, voice_bytes, _config(language=language))["lang"] == code

    def test_unsupported_language_raises_unless_forced(self, upstream, voice_bytes):
        provider = IndexTTSProvider(_config(language="German"), device="cpu")
        with pytest.raises(RuntimeError, match="cannot narrate German"):
            provider.synthesize("Guten Tag.", voice_bytes, return_bytes=True)
        assert not upstream.v25.instances[0].calls
        forced = self._call(upstream, voice_bytes, _config(language="German", tts_options={"lang": "EN"}))
        assert forced["lang"] == "EN"

    def test_indextts2_gets_only_its_own_arguments(self, upstream, voice_bytes):
        config = _config(tts_model_name="IndexTeam/IndexTTS-2", speed=1.5, tts_options={"low_vram_split": "on"})
        call = self._call(upstream, voice_bytes, config, fake=upstream.v2)
        for absent in ("lang", "duration_factor", "text_normalization"):
            assert absent not in call
        assert call["max_text_tokens_per_segment"] == 120 and call["repetition_penalty"] == pytest.approx(10.0)
        assert not hasattr(upstream.v2.instances[0], "low_vram")

        japanese = IndexTTSProvider(_config(tts_model_name="IndexTeam/IndexTTS-2", language="Japanese"), device="cpu")
        with pytest.raises(RuntimeError, match="IndexTTS-2 does not support Japanese"):
            japanese.synthesize("こんにちは。", voice_bytes, return_bytes=True)

    def test_low_vram_split_option(self, upstream, voice_bytes):
        provider = IndexTTSProvider(_config(tts_options={"low_vram_split": "on"}), device="cpu")
        provider.synthesize("One.", voice_bytes, return_bytes=True)
        fake = upstream.v25.instances[0]
        provider.config = _config(tts_options={"low_vram_split": "auto"})
        provider.synthesize("Two.", voice_bytes, return_bytes=True)
        provider.config = _config(tts_options={"low_vram_split": "off"})
        fake.low_vram = True
        provider.synthesize("Three.", voice_bytes, return_bytes=True)
        assert fake.low_vram_seen == [True, False, False]   # auto restores upstream's own detection

    def test_glossary_file_is_loaded_reloaded_and_removed(self, upstream, voice_bytes, tmp_path):
        glossary = tmp_path / "glossary.yaml"
        glossary.write_text("NVMe: N-V-M-E\n", encoding="utf-8")
        provider = IndexTTSProvider(_config(tts_options={"glossary_file": str(glossary)}), device="cpu")
        provider.synthesize("One.", voice_bytes, return_bytes=True)
        normalizer = upstream.v25.instances[0].text_process
        assert normalizer.term_glossary == {"NVMe: N-V-M-E".split(":")[0]: "N-V-M-E"}

        glossary.write_text("NVMe: N-V-M-E\nSSD: S-S-D\n", encoding="utf-8")
        os.utime(glossary, ns=(1, 1))
        provider.synthesize("Two.", voice_bytes, return_bytes=True)
        assert set(normalizer.term_glossary) == {"NVMe", "SSD"}

        provider.config = _config()
        provider.synthesize("Three.", voice_bytes, return_bytes=True)
        assert normalizer.term_glossary == {}

        provider.config = _config(tts_options={"glossary_file": str(tmp_path / "missing.yaml")})
        with pytest.raises(RuntimeError, match="glossary"):
            provider.synthesize("Four.", voice_bytes, return_bytes=True)

    def test_seed_is_applied_before_every_chunk(self, upstream, voice_bytes):
        draws: list[float] = []
        fake_cls = upstream.v25
        original = fake_cls._infer

        def infer(self, named, generation_kwargs):
            draws.append(float(torch.rand(1)))
            return original(self, named, generation_kwargs)

        fake_cls._infer = infer
        provider = IndexTTSProvider(_config(seed=1234), device="cpu")
        provider.synthesize_batch(["One.", "Two.", "Three."], voice_bytes)
        assert draws[0] == draws[1] == draws[2]
        provider.config = _config(seed=-1)
        provider.synthesize_batch(["Four.", "Five."], voice_bytes)
        assert draws[3] != draws[4]


# ── Synthesis behaviour ──────────────────────────────────────────────────────

class TestSynthesis:

    def test_returns_valid_wav_at_native_rate(self, upstream, voice_bytes, tmp_path):
        provider = IndexTTSProvider(_config(), device="cpu")
        text = "Hello world."
        wav_bytes, duration = provider.synthesize(text, voice_bytes, return_bytes=True)
        samples, rate = _read_wav(wav_bytes)
        assert rate == 22050 == IndexTTSProvider.info().native_sample_rate
        assert samples.ndim == 1 and samples.size == 100 * len(text)
        assert duration == pytest.approx(samples.size / 22050)
        # int16 8000 → 8000 / 32767 in float, not left at PCM scale
        assert float(np.abs(samples).max()) == pytest.approx(8000 / 32767, abs=1e-3)

        out_path = tmp_path / "out" / "chunk.wav"
        result, duration_2 = provider.synthesize(text, voice_bytes, str(out_path))
        assert result == str(out_path) and out_path.is_file()
        assert duration_2 == pytest.approx(duration)

    def test_voice_conditioning_computed_once_across_chunks(self, upstream, voice_bytes):
        provider = IndexTTSProvider(_config(), device="cpu")
        texts = [f"Sentence number {i}." for i in range(6)]
        provider.synthesize_batch(texts[:3], voice_bytes)
        for text in texts[3:]:
            provider.synthesize(text, voice_bytes, return_bytes=True)
        fake = upstream.v25.instances[0]
        assert len(fake.calls) == 6
        assert len({call["spk_audio_prompt"] for call in fake.calls}) == 1
        assert fake.spk_conditioning_calls == 1
        assert fake.emo_conditioning_calls == 1

        # A different narrator clip is conditioned again, once.
        other = b"RIFF" + bytes(reversed(range(256))) * 8
        provider.synthesize_batch(texts[:2], other)
        assert fake.spk_conditioning_calls == 2
        assert fake.calls[-1]["spk_audio_prompt"] != fake.calls[0]["spk_audio_prompt"]

    def test_voice_path_is_content_addressed(self, upstream, voice_bytes, tmp_path):
        """A file replaced under the same name must not hit upstream's path-keyed cache."""
        voice = tmp_path / "narrator.wav"
        voice.write_bytes(voice_bytes)
        provider = IndexTTSProvider(_config(), device="cpu")
        provider.synthesize("One.", str(voice), return_bytes=True)
        provider.synthesize("Two.", str(voice), return_bytes=True)
        fake = upstream.v25.instances[0]
        assert fake.spk_conditioning_calls == 1
        first_path = fake.calls[0]["spk_audio_prompt"]
        assert first_path != str(voice) and first_path == fake.calls[1]["spk_audio_prompt"]

        voice.write_bytes(b"RIFF" + b"\x07" * 4096)
        provider.synthesize("Three.", str(voice), return_bytes=True)
        assert fake.spk_conditioning_calls == 2
        assert fake.calls[2]["spk_audio_prompt"] != first_path
        # bytes with the same content as the first file share its conditioning path
        provider.synthesize("Four.", voice_bytes, return_bytes=True)
        with open(fake.calls[3]["spk_audio_prompt"], "rb") as fh:
            assert fh.read() == voice_bytes

    def test_config_voice_file_is_used_when_no_reference_is_passed(self, upstream, voice_bytes, tmp_path):
        voice = tmp_path / "narrator.wav"
        voice.write_bytes(voice_bytes)
        provider = IndexTTSProvider(_config(voice_file=str(voice)), device="cpu")
        provider.synthesize("One.", b"", return_bytes=True)
        with open(upstream.v25.instances[0].calls[0]["spk_audio_prompt"], "rb") as fh:
            assert fh.read() == voice_bytes

    def test_missing_voice_reference_raises(self, upstream):
        provider = IndexTTSProvider(_config(), device="cpu")
        with pytest.raises(RuntimeError, match="reference clip"):
            provider.synthesize("Hello.", b"", return_bytes=True)
        with pytest.raises(RuntimeError, match="does not exist"):
            provider.synthesize("Hello.", "/nonexistent/abm_voice.wav", return_bytes=True)

    def test_batch_output_order_and_count(self, upstream, voice_bytes):
        provider = IndexTTSProvider(_config(), device="cpu")
        texts = ["Short.", "A somewhat longer sentence follows here.", "Mid length one.", "X."]
        results = provider.synthesize_batch(texts, voice_bytes)
        assert len(results) == len(texts)
        for text, (wav_bytes, duration) in zip(texts, results):
            assert isinstance(wav_bytes, bytes)
            samples, rate = _read_wav(wav_bytes)
            assert rate == 22050
            assert samples.size == 100 * len(text), "result is out of order"
            assert duration == pytest.approx(100 * len(text) / 22050)
        assert [call["text"] for call in upstream.v25.instances[0].calls] == texts
        assert provider.synthesize_batch([], voice_bytes) == []

    def test_empty_text_raises(self, upstream, voice_bytes):
        provider = IndexTTSProvider(_config(), device="cpu")
        with pytest.raises(RuntimeError, match="empty text"):
            provider.synthesize("   ", voice_bytes, return_bytes=True)

    @pytest.mark.parametrize(
        ("result", "match"),
        [
            (None, "produced no audio"),                                             # upstream's IndexError path
            ((22050, np.zeros((0, 1), dtype=np.int16)), "produced no audio"),
            ((22050, np.array([[0.1], [float("nan")], [0.2]], dtype=np.float32)), "NaN/inf"),
            ((22050, np.array([[0.1], [float("inf")]], dtype=np.float32)), "NaN/inf"),
            ((0, np.ones((100, 1), dtype=np.int16)), "produced no audio"),
            ("gen.wav", "unexpected result"),
        ],
    )
    def test_bad_audio_raises_instead_of_returning_placeholder(self, upstream, voice_bytes, result, match):
        provider = IndexTTSProvider(_config(), device="cpu")
        provider.ensure_ready()
        upstream.v25.instances[0].results.append(result)
        with pytest.raises(RuntimeError, match=match):
            provider.synthesize("Hello world.", voice_bytes, return_bytes=True)

    def test_batch_raises_when_one_item_fails(self, upstream, voice_bytes):
        provider = IndexTTSProvider(_config(), device="cpu")
        provider.ensure_ready()
        fake = upstream.v25.instances[0]
        fake.results.extend([(22050, np.full((500, 1), 100, dtype=np.int16)), None])
        with pytest.raises(RuntimeError, match="produced no audio"):
            provider.synthesize_batch(["One.", "Two.", "Three."], voice_bytes)
        assert len(fake.calls) == 2

    def test_upstream_exception_is_wrapped_with_context(self, upstream, voice_bytes):
        provider = IndexTTSProvider(_config(), device="cpu")
        provider.ensure_ready()
        upstream.v25.instances[0].results.append(KeyError("char_rep_map"))
        with pytest.raises(RuntimeError, match="IndexTTS-2.5 synthesis failed on cpu: KeyError"):
            provider.synthesize("Hello.", voice_bytes, return_bytes=True)

    def test_cuda_oom_retries_once_with_smaller_segments(self, upstream, voice_bytes, monkeypatch):
        emptied: list[bool] = []
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(torch.cuda, "empty_cache", lambda: emptied.append(True))
        monkeypatch.setattr(torch.cuda, "manual_seed_all", lambda seed: None)
        provider = IndexTTSProvider(_config(), device="cpu")
        provider.ensure_ready()
        fake = upstream.v25.instances[0]
        fake.results.append(torch.cuda.OutOfMemoryError("CUDA out of memory"))
        wav_bytes, duration = provider.synthesize("Hello world.", voice_bytes, return_bytes=True)
        assert duration > 0 and len(wav_bytes) > 100
        assert [call["max_text_tokens_per_segment"] for call in fake.calls] == [120, 60]
        assert emptied, "the CUDA cache must be emptied before retrying"

        fake.results.extend([torch.cuda.OutOfMemoryError("oom"), torch.cuda.OutOfMemoryError("oom again")])
        with pytest.raises(RuntimeError, match="ran out of GPU memory"):
            provider.synthesize("Hello world.", voice_bytes, return_bytes=True)

    def test_inference_runs_without_autograd(self, upstream, voice_bytes):
        states: list[bool] = []
        fake_cls = upstream.v25
        original = fake_cls._infer

        def infer(self, named, generation_kwargs):
            states.append(torch.is_grad_enabled())
            return original(self, named, generation_kwargs)

        fake_cls._infer = infer
        IndexTTSProvider(_config(), device="cpu").synthesize("Hello.", voice_bytes, return_bytes=True)
        assert states == [False]
