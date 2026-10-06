"""
audiobook_factory/tts_providers/indextts_provider.py
=====================================================
IndexTTS-2.5 / IndexTTS-2 (bilibili) zero-shot voice-cloning provider.

Wraps the upstream ``IndexTTS2`` inference class of
https://github.com/index-tts/index-tts (signatures read at commit
``d9e41aac89fd00b3d71497fddb287b7f24613712``):

* ``indextts.infer_v2_5.IndexTTS2`` for ``IndexTeam/IndexTTS-2.5``
  (``use_bf16``; ``infer(..., lang, duration_factor, text_normalization)``).
* ``indextts.infer_v2.IndexTTS2`` for ``IndexTeam/IndexTTS-2``
  (``use_fp16``; no language, speed or normalisation arguments).

Both classes share the rest of the ``infer`` signature, so one provider
drives either: arguments that only one version accepts are passed only when
the loaded class declares them.

How the model is used here
--------------------------
* The narrator is cloned from the reference clip alone (first 15 s, no
  transcript). Emotion is controlled separately from timbre: taken from the
  narrator clip, from a second "emotion reference" clip, from an 8-value
  emotion vector, or from text through upstream's optional QwenEmotion model.
* Upstream splits each chunk into segments of at most
  ``max_text_tokens_per_segment`` text tokens, generates them one after the
  other and joins them with ``interval_silence`` ms of silence *between*
  segments only. Nothing is appended at the chunk edges, so the pipeline's
  own inter-chunk pauses stay the only pauses there.
* Generation is not batched upstream (one autoregressive sequence at a
  time), so ``synthesize_batch`` is the base per-item loop.
* Upstream caches the speaker / emotion conditioning by *file path*. This
  provider always hands it a content-addressed path, which makes that cache
  a content-keyed one (see :meth:`IndexTTSProvider._stable_audio_path`).

The upstream package is imported lazily; this module imports on a machine
that has neither torch nor ``indextts`` installed.
"""
from __future__ import annotations

import gc
import hashlib
import importlib
import inspect
import logging
import math
import os
import random
import re
import sys
import tempfile
import threading
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING, Any

from audiobook_factory.tts_providers.base_tts_provider import (
    BaseTTSProvider,
    ProviderInfo,
    ProviderOption,
)

if TYPE_CHECKING:
    from audiobook_factory.pipeline import AudiobookConfig

logger = logging.getLogger(__name__)

# ── Upstream identifiers ─────────────────────────────────────────────────────
_MODEL_V2_5: str = "IndexTeam/IndexTTS-2.5"
_MODEL_V2: str = "IndexTeam/IndexTTS-2"
_UPSTREAM_MODULES: dict[str, str] = {
    _MODEL_V2_5: "indextts.infer_v2_5",
    _MODEL_V2: "indextts.infer_v2",
}
_UPSTREAM_CLASS: str = "IndexTTS2"
_UPSTREAM_COMMIT: str = "d9e41aac89fd00b3d71497fddb287b7f24613712"
_UPSTREAM_GIT_URL: str = "https://github.com/index-tts/index-tts.git"
# Half-precision type each upstream class implements (the other one is absent).
_HALF_DTYPES: dict[str, str] = {_MODEL_V2_5: "bfloat16", _MODEL_V2: "float16"}
# Languages each checkpoint was trained on, as upstream ``lang`` codes.
_MODEL_LANGUAGES: dict[str, tuple[str, ...]] = {
    _MODEL_V2_5: ("ZH", "EN", "JA", "ES", "AR"),
    _MODEL_V2: ("ZH", "EN"),
}
_DISPLAY_NAMES: dict[str, str] = {_MODEL_V2_5: "IndexTTS-2.5", _MODEL_V2: "IndexTTS-2"}
# Sub-directory of the checkpoint repo that holds the QwenEmotion model
# (``qwen_emo_path`` in upstream's config.yaml). Skipped unless it is needed.
_QWEN_EMO_DIR: str = "qwen0.6bemo4-merge"

# ── Installation ─────────────────────────────────────────────────────────────
_REPO_ENV_VAR: str = "ABM_INDEXTTS_REPO"
_REQUIREMENTS_FILE: str = "requirements/tts-indextts.txt"
# Upstream pins torch==2.8.*, numpy==2.2.6, keras==2.9.0 ... and Python <3.12
# in its package metadata, so the package itself is installed without its
# declared dependencies; the ones inference really imports are in the
# requirements file.
_INSTALL_COMMAND: str = (
    f"pip install -r {_REQUIREMENTS_FILE} && "
    f'pip install --no-deps --ignore-requires-python "indextts @ git+{_UPSTREAM_GIT_URL}@{_UPSTREAM_COMMIT}"'
)

# ── Audio / generation constants ─────────────────────────────────────────────
_NATIVE_SAMPLE_RATE: int = 22050
_PCM16_MAX: float = 32767.0            # upstream returns int16-scaled samples
_MEL_TOKEN_RATE_HZ: float = 50.0       # 22050 / 256 hop / 1.72 frames per token
_MIN_MEL_TOKENS: int = 50
_MIN_SEGMENT_TOKENS: int = 20
# Ceiling for short chunks: floor + generous per-character allowance (about
# 2.5x a slow narrator). Digits expand into whole words when normalised.
_MEL_BOUND_FLOOR: int = 200
_MEL_TOKENS_PER_CHAR: float = 10.0
_MEL_TOKENS_PER_WIDE_CHAR: float = 30.0
_MEL_TOKENS_PER_DIGIT: float = 60.0
# Hiragana / katakana, CJK ideographs (incl. extension A and compatibility), hangul.
_WIDE_CHAR_RE: re.Pattern[str] = re.compile(
    r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af\uf900-\ufaff]"
)
_MIN_DURATION_FACTOR: float = 0.5
_MAX_DURATION_FACTOR: float = 2.0
_EMO_VECTOR_SIZE: int = 8
_EMO_VECTOR_MAX: float = 1.2           # upper clamp upstream applies to scores
_LOW_VRAM_GB: float = 10.0             # upstream's low-VRAM threshold
_MAX_STABLE_PATHS: int = 16

_LANGUAGE_CODES: dict[str, str] = {
    "zh": "ZH", "chinese": "ZH", "mandarin": "ZH", "cmn": "ZH", "中文": "ZH",
    "en": "EN", "english": "EN", "eng": "EN",
    "ja": "JA", "japanese": "JA", "jp": "JA", "jpn": "JA", "日本語": "JA",
    "es": "ES", "spanish": "ES", "castilian": "ES", "spa": "ES", "español": "ES",
    "ar": "AR", "arabic": "AR", "ara": "AR", "العربية": "AR",
}
_LANGUAGE_NAMES: dict[str, str] = {
    "ZH": "Chinese", "EN": "English", "JA": "Japanese", "ES": "Spanish", "AR": "Arabic",
}

# Serialises everything process-global about loading: sys.path / os.environ
# edits, upstream's auxiliary-model downloads into the shared checkpoint
# directory, and the constructor itself. One GPU loads at a time.
_UPSTREAM_LOAD_LOCK: threading.Lock = threading.Lock()


@dataclass(frozen=True)
class _LoadSpec:
    """Everything that is fixed when the upstream model object is built."""

    model_id: str
    precision: str            # "float32" | "float16" | "bfloat16"
    use_cuda_kernel: bool
    use_deepspeed: bool
    use_accel: bool
    use_torch_compile: bool
    use_qwen_emo: bool

    def satisfies(self, wanted: "_LoadSpec") -> bool:
        """True when a model loaded with this spec can serve *wanted*.

        A model that already carries QwenEmotion also serves runs that do
        not need it, so that alone never forces a reload.
        """
        if wanted.use_qwen_emo and not self.use_qwen_emo:
            return False
        return replace(wanted, use_qwen_emo=self.use_qwen_emo) == self


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _estimate_mel_token_bound(text: str) -> int:
    """Upper bound on the speech tokens a chunk of *text* can plausibly need.

    Deliberately loose (roughly 2.5x a slow reading): it exists to stop a
    hallucinating short chunk from babbling up to the 36 s model limit, not
    to predict the duration.
    """
    wide = len(_WIDE_CHAR_RE.findall(text))
    digits = sum(ch.isdigit() for ch in text)
    other = max(0, len(text) - wide - digits)
    return int(
        _MEL_BOUND_FLOOR
        + wide * _MEL_TOKENS_PER_WIDE_CHAR
        + digits * _MEL_TOKENS_PER_DIGIT
        + other * _MEL_TOKENS_PER_CHAR
    )


def _parse_emo_vector(value: Any) -> list[float] | None:
    """Parses the ``emo_vector`` option into eight floats, or None when unset.

    Accepts a list / tuple or a string such as ``"0, 0, 0.8, 0, 0, 0, 0, 0"``.

    Raises
    ------
    RuntimeError
        If the value is set but is not eight finite numbers in ``[0, 1.2]``.
    """
    if value is None:
        return None
    if isinstance(value, str):
        parts: list[Any] = [p for p in re.split(r"[,;\s]+", value.strip().strip("[]()")) if p]
    elif isinstance(value, (list, tuple)):
        parts = list(value)
    else:
        raise RuntimeError(f"IndexTTS emo_vector must be a list or a string, got {type(value).__name__}.")
    if not parts:
        return None
    try:
        vector = [float(p) for p in parts]
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"IndexTTS emo_vector contains a non-numeric value: {value!r}.") from exc
    if len(vector) != _EMO_VECTOR_SIZE:
        raise RuntimeError(
            f"IndexTTS emo_vector needs {_EMO_VECTOR_SIZE} values "
            "(happy, angry, sad, afraid, disgusted, melancholic, surprised, calm); "
            f"got {len(vector)}."
        )
    if any(not math.isfinite(v) or v < 0.0 or v > _EMO_VECTOR_MAX for v in vector):
        raise RuntimeError(f"IndexTTS emo_vector values must be between 0 and {_EMO_VECTOR_MAX}: {vector}.")
    return vector


def _import_upstream_class(model_id: str) -> Any:
    """Imports and returns upstream's inference class for *model_id*.

    Call with ``_UPSTREAM_LOAD_LOCK`` held: it edits ``sys.path`` and
    ``os.environ``.

    Raises
    ------
    RuntimeError
        If the upstream package (or one of its dependencies) cannot be
        imported. The message names the exact install command.
    """
    module_name = _UPSTREAM_MODULES[model_id]
    repo = os.environ.get(_REPO_ENV_VAR, "").strip()
    if repo:
        repo = os.path.abspath(os.path.expanduser(repo))
        if not os.path.isdir(os.path.join(repo, "indextts")):
            raise RuntimeError(
                f"{_REPO_ENV_VAR}={repo!r} is not an index-tts checkout (no 'indextts' package in it). "
                f"Clone it with: git clone {_UPSTREAM_GIT_URL}"
            )
        # Appended, not prepended: the checkout also contains top-level
        # 'tests' and 'tools' packages that must not shadow this project's.
        if repo not in sys.path:
            sys.path.append(repo)
            importlib.invalidate_caches()

    # infer_v2_5 sets os.environ['HF_HUB_CACHE'] = './checkpoints/hf_cache' at
    # import time. Import huggingface_hub first so its cache location is
    # already resolved, then put the variable back so no other provider ends
    # up downloading into a directory relative to the working directory.
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        pass
    saved_hub_cache = os.environ.get("HF_HUB_CACHE")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        missing = getattr(exc, "name", None) or ""
        if missing == "indextts" or missing.startswith("indextts."):
            raise RuntimeError(
                f"The IndexTTS package (module '{module_name}') is not installed. Install it with:\n"
                f"    {_INSTALL_COMMAND}\n"
                f"or point {_REPO_ENV_VAR} at a clone of {_UPSTREAM_GIT_URL} that contains "
                f"'{module_name.replace('.', '/')}.py'."
            ) from exc
        raise RuntimeError(
            f"IndexTTS is installed but could not be imported ({exc}). One of its dependencies is "
            "missing or has an incompatible version (it needs transformers==4.52.1). Install them with:\n"
            f"    {_INSTALL_COMMAND}"
        ) from exc
    except Exception as exc:
        raise RuntimeError(
            f"Importing IndexTTS failed ({type(exc).__name__}: {exc}). This usually means an incompatible "
            "transformers version (it needs transformers==4.52.1). Reinstall with:\n"
            f"    {_INSTALL_COMMAND}"
        ) from exc
    finally:
        if saved_hub_cache is None:
            os.environ.pop("HF_HUB_CACHE", None)
        else:
            os.environ["HF_HUB_CACHE"] = saved_hub_cache

    cls = getattr(module, _UPSTREAM_CLASS, None)
    if cls is None:
        raise RuntimeError(
            f"'{module_name}' has no class '{_UPSTREAM_CLASS}'; the installed index-tts is not the "
            f"version this provider was written for. Reinstall with:\n    {_INSTALL_COMMAND}"
        )
    return cls


class IndexTTSProvider(BaseTTSProvider):
    """IndexTTS-2.5 / IndexTTS-2 provider: one upstream model on one device.

    Parameters
    ----------
    config : AudiobookConfig
        Run configuration. Read again on every call; the pool reassigns it.
    device : str | None
        Torch device this instance is bound to (``"cuda:0"``, ``"cpu"``).
        Defaults to ``config.device``.
    dtype_override : str | None
        ``"float16"``, ``"bfloat16"`` or ``"float32"``. IndexTTS-2 implements
        FP16 and IndexTTS-2.5 implements BF16 only; a half-precision request
        for the type a model lacks falls back as described in
        :meth:`_resolve_precision`.
    """

    INFO = ProviderInfo(
        name="indextts",
        display_name="IndexTTS-2.5",
        description=(
            "Bilibili's autoregressive zero-shot TTS. Clones the narrator from one clip (first 15 s, "
            "no transcript) and controls emotion separately from timbre: from the narrator clip, an "
            "emotion reference clip, an 8-value emotion vector, or a text description. Native speed "
            "control and inline pronunciation tags such as <minute|M IH1 . N AH0 T> or <行|XING2> "
            "(IndexTTS-2.5)."
        ),
        license=(
            "bilibili Model Use License Agreement (custom; free use incl. commercial below "
            "100M MAU / RMB 1B revenue, with conditions on distributed outputs)"
        ),
        commercial_use=True,
        homepage="https://github.com/index-tts/index-tts",
        default_model=_MODEL_V2_5,
        models=(_MODEL_V2_5, _MODEL_V2),
        native_sample_rate=_NATIVE_SAMPLE_RATE,
        min_vram_gb=7.0,
        languages=("Chinese", "English", "Japanese", "Spanish", "Arabic"),
        supports_voice_clone=True,
        transcript="unused",
        supports_instruct=True,
        supports_batch=False,
        supports_speed=True,
        supports_seed=True,
        preset_voices=(),
        options=(
            # ── Emotion ──────────────────────────────────────────────────────
            ProviderOption(
                key="emotion_mode", label="Emotion source", kind="choice", default="auto",
                choices=("auto", "speaker", "audio", "vector", "text", "script"),
                help=(
                    "speaker = emotion of the narrator clip; audio = emotion reference clip; vector = "
                    "emotion vector; text = the style prompt describes the emotion; script = emotion "
                    "detected from each chunk's own text. auto picks text, vector or audio when the "
                    "matching setting is filled in, else speaker. text and script are experimental "
                    "upstream and load the extra QwenEmotion model (about 1.2 GB)."
                ),
            ),
            ProviderOption(
                key="emo_audio_prompt", label="Emotion reference audio", kind="file", default="",
                help="Clip whose emotion (not voice) is transferred to the narration; first 15 s are used.",
            ),
            ProviderOption(
                key="emo_vector", label="Emotion vector", kind="str", default="",
                help=(
                    "Eight values 0-1, comma separated: happy, angry, sad, afraid, disgusted, "
                    "melancholic, surprised, calm. Normalised like upstream's WebUI (sum capped at 0.8)."
                ),
            ),
            ProviderOption(
                key="emo_alpha", label="Emotion strength", kind="float", default=0.65,
                minimum=0.0, maximum=1.0, step=0.05,
                help=(
                    "How strongly the emotion reference, vector or text emotion is applied. "
                    "Ignored when the emotion comes from the narrator clip."
                ),
            ),
            ProviderOption(
                key="use_random", label="Randomise emotion sampling", kind="bool", default=False,
                help="Picks random emotion prototypes for vector / text emotion. Reduces voice-clone fidelity.",
            ),
            # ── Text front end ───────────────────────────────────────────────
            ProviderOption(
                key="lang", label="Language", kind="choice", default="auto",
                choices=("auto", "ZH", "EN", "JA", "ES", "AR"),
                help="auto follows the book language. IndexTTS-2 handles only Chinese and English.",
            ),
            ProviderOption(
                key="text_normalization", label="Text normalisation", kind="bool", default=True,
                help="Let IndexTTS-2.5 expand numbers, dates and symbols into words before synthesis.",
            ),
            ProviderOption(
                key="glossary_file", label="Pronunciation glossary (YAML)", kind="str", default="",
                help=(
                    "Path to a YAML file mapping terms to readings, e.g. 'NVMe: N-V-M-E' or "
                    "'M.2: {en: M dot two, zh: M 二}'. Applied by the Chinese / English normaliser."
                ),
            ),
            # ── Segmentation ─────────────────────────────────────────────────
            ProviderOption(
                key="max_text_tokens_per_segment", label="Max text tokens per segment", kind="int",
                default=120, minimum=_MIN_SEGMENT_TOKENS, maximum=600, step=2,
                help=(
                    "Each chunk is split at punctuation into segments of at most this many text tokens "
                    "(English: about 4 characters per token; Chinese: 1). Upstream recommends 80-200; "
                    "larger is smoother but needs more VRAM, and a segment longer than about 36 s of "
                    "speech is cut off."
                ),
            ),
            ProviderOption(
                key="interval_silence", label="Silence between segments (ms)", kind="int",
                default=200, minimum=0, maximum=2000, step=10,
                help="Inserted between the segments of one chunk only, never at the chunk edges.",
            ),
            ProviderOption(
                key="low_vram_split", label="Low-VRAM clause splitting", kind="choice", default="auto",
                choices=("auto", "on", "off"),
                help=(
                    "IndexTTS-2.5 synthesises clause by clause (about 40 characters) to bound memory. "
                    "auto enables it on GPUs under 10 GB, as upstream does."
                ),
            ),
            # ── Sampling ─────────────────────────────────────────────────────
            ProviderOption(
                key="max_mel_tokens", label="Max speech tokens per segment", kind="int",
                default=1815, minimum=_MIN_MEL_TOKENS, maximum=1815, step=5,
                help="Hard stop for one segment (50 tokens per second of speech; the model limit is 1815).",
            ),
            ProviderOption(
                key="adaptive_max_mel_tokens", label="Shorter limit for short chunks", kind="bool",
                default=True,
                help="Lowers the speech-token limit for short chunks so a runaway generation stops early.",
            ),
            ProviderOption(
                key="do_sample", label="Sampling", kind="bool", default=True,
                help="Off = deterministic beam search.",
            ),
            ProviderOption(
                key="num_beams", label="Beams", kind="int", default=3, minimum=1, maximum=10, step=1,
                help="Beam count of the speech-token decoder; more is slower and uses more VRAM.",
            ),
            ProviderOption(
                key="repetition_penalty", label="Repetition penalty (IndexTTS)", kind="float",
                default=10.0, minimum=0.1, maximum=20.0, step=0.1,
                help=(
                    "IndexTTS needs a far stronger penalty than other engines (upstream default 10), "
                    "so it has its own setting instead of the shared repetition penalty."
                ),
            ),
            ProviderOption(
                key="length_penalty", label="Length penalty", kind="float", default=0.0,
                minimum=-2.0, maximum=2.0, step=0.1,
                help="Beam-search length penalty; positive favours longer output.",
            ),
            ProviderOption(
                key="typical_sampling", label="Typical sampling", kind="bool", default=False,
                help="Upstream's typical-sampling filter. Not recommended by upstream.",
            ),
            ProviderOption(
                key="typical_mass", label="Typical mass", kind="float", default=0.9,
                minimum=0.05, maximum=0.95, step=0.05,
                help="Probability mass kept by typical sampling.",
            ),
            # ── Runtime ──────────────────────────────────────────────────────
            ProviderOption(
                key="precision", label="Precision", kind="choice", default="auto",
                choices=("auto", "half", "full"),
                help=(
                    "half = FP16 on IndexTTS-2, BF16 on IndexTTS-2.5. auto uses FP16 for IndexTTS-2; "
                    "for IndexTTS-2.5 it uses BF16 on GPUs with native BF16 or under 10 GB, else FP32."
                ),
            ),
            ProviderOption(
                key="use_cuda_kernel", label="BigVGAN CUDA kernel", kind="bool", default=False,
                help="Compiled vocoder activation kernel. Needs ninja and the CUDA toolkit (nvcc).",
            ),
            ProviderOption(
                key="use_deepspeed", label="DeepSpeed", kind="bool", default=False,
                help="DeepSpeed inference kernels (pip install deepspeed). May be faster or slower.",
            ),
            ProviderOption(
                key="use_accel", label="GPT acceleration engine", kind="bool", default=False,
                help=(
                    "flash-attn based decoder (pip install flash-attn). It samples with temperature "
                    "only: top-k, top-p, beams and repetition penalty are ignored."
                ),
            ),
            ProviderOption(
                key="use_torch_compile", label="torch.compile", kind="bool", default=False,
                help="Compile the speech-to-mel model; long warm-up on the first chunks.",
            ),
            ProviderOption(
                key="verbose", label="Verbose upstream log", kind="bool", default=False,
                help="Print upstream's per-segment debug output.",
            ),
        ),
        pip_requirements=(
            "transformers==4.52.1",
            "huggingface-hub>=0.30,<1.0",
            "accelerate>=1.8.1",
            "omegaconf>=2.3.0",
            "einops>=0.8.1",
            "munch>=4.0.0",
            "librosa>=0.10.2",
            "sentencepiece>=0.2.0",
            "tiktoken>=0.7.0",
            "openai-whisper>=20231117",
            "modelscope>=1.27.0",
            "safetensors>=0.5.2",
            "matplotlib",
            "scipy",
            "pyyaml",
            "fugashi[unidic-lite]>=1.2.0",
            'WeTextProcessing>=1.0.4.1; sys_platform == "linux"',
            'wetext>=0.0.9; sys_platform != "linux"',
        ),
        install_notes=(
            "Two steps, because upstream's own package metadata pins torch==2.8.*, numpy==2.2.6, "
            f"keras==2.9.0 and Python <3.12:\n  {_INSTALL_COMMAND}\n"
            f"Alternative to the second step: git clone {_UPSTREAM_GIT_URL} and set {_REPO_ENV_VAR} to the "
            "clone (it is appended to sys.path; an installed 'indextts' package takes precedence).\n"
            "IndexTTS needs transformers==4.52.1: its bundled generation code does not import under "
            "transformers 4.57.x, so it cannot share an environment with qwen-tts (transformers==4.57.3). "
            "Use a separate environment / notebook session.\n"
            "Weights: huggingface_hub.snapshot_download() into the HF cache, about 4.3 GB for "
            "IndexTeam/IndexTTS-2.5 (+1.2 GB QwenEmotion when text emotion is used). On first load upstream "
            "also downloads facebook/w2v-bert-2.0 (4.7 GB), nvidia/bigvgan_v2_22khz_80band_256x (0.45 GB), "
            "funasr/campplus and amphion/MaskGCT's semantic codec into an 'hf_cache' folder inside that "
            "snapshot, so the HF cache must be writable. No repository is gated; no HF token is needed. "
            "Set USE_MODELSCOPE=false to stop upstream probing ModelScope.\n"
            "VRAM: roughly 6-7 GB in half precision, 9-10 GB in FP32 for 400-character chunks. "
            "System packages: none beyond a working torchaudio; ninja + nvcc only for the optional "
            "BigVGAN CUDA kernel.\n"
            "Licence: IndexTTS-2 additionally loads amphion/MaskGCT's semantic codec (CC-BY-NC-4.0); "
            "IndexTTS-2.5 does not load it."
        ),
    )

    def __init__(
        self,
        config: "AudiobookConfig",
        device: str | None = None,
        dtype_override: str | None = None,
    ) -> None:
        super().__init__(config)
        bound = (device or getattr(config, "device", None) or "cuda").strip()
        self._device: str = "cuda:0" if bound == "cuda" else bound
        self._dtype_override: str | None = dtype_override
        self._model: Any = None
        self._load_spec: _LoadSpec | None = None
        self._infer_params: frozenset[str] = frozenset()
        self._upstream_low_vram: bool | None = None
        self._base_glossary: dict[str, Any] = {}
        self._glossary_key: tuple[str, int, int] | None = None
        self._stable_paths: dict[tuple[str, int, int], str] = {}
        self._warned: set[str] = set()
        # Re-entrant: synthesize() holds it across ensure-loaded + inference.
        self._lock = threading.RLock()

    # ── Identity ─────────────────────────────────────────────────────────────

    @property
    def device(self) -> str:
        """The torch device string this provider is bound to."""
        return self._device

    def get_name(self) -> str:
        """Display name of the model this instance will run."""
        return _DISPLAY_NAMES.get(self.resolve_model_id(), self.info().display_name)

    # ── Small helpers ────────────────────────────────────────────────────────

    def _warn_once(self, key: str, message: str, *args: Any) -> None:
        if key not in self._warned:
            self._warned.add(key)
            logger.warning(message, *args)

    def _choice(self, key: str) -> str:
        """Reads a ``kind="choice"`` option, falling back to its default."""
        declared = next(o for o in self.info().options if o.key == key)
        value = str(self.option(key, declared.default) or "").strip().lower()
        for choice in declared.choices:
            if value == choice.lower():
                return choice
        self._warn_once(
            f"choice:{key}:{value}", "[IndexTTS] Unknown value %r for option '%s'; using '%s'.",
            value, key, declared.default,
        )
        return str(declared.default)

    def _number(self, key: str) -> float:
        """Reads a numeric option clamped to its declared range."""
        declared = next(o for o in self.info().options if o.key == key)
        value = float(self.option(key, declared.default))
        if not math.isfinite(value):
            value = float(declared.default)
        if declared.minimum is not None:
            value = max(float(declared.minimum), value)
        if declared.maximum is not None:
            value = min(float(declared.maximum), value)
        return value

    def _config_float(self, name: str, default: float) -> float:
        value = getattr(self.config, name, None)
        try:
            number = float(value) if value is not None else default
        except (TypeError, ValueError):
            return default
        return number if math.isfinite(number) else default

    def _model_limit(self, name: str) -> int | None:
        """Reads ``model.cfg.gpt.<name>`` (the checkpoint's own limit), if present."""
        try:
            value = getattr(self._model.cfg.gpt, name)
            return int(value) if value else None
        except Exception:
            return None

    # ── Emotion ──────────────────────────────────────────────────────────────

    def _emotion_mode(self) -> str:
        """Resolves ``emotion_mode`` to speaker / audio / vector / text / script.

        ``auto`` follows upstream's own precedence when several inputs are
        given: text description, then vector, then reference audio.
        """
        mode = self._choice("emotion_mode")
        if mode != "auto":
            return mode
        if (getattr(self.config, "tts_instruct", "") or "").strip():
            return "text"
        vector = _parse_emo_vector(self.option("emo_vector"))
        if vector is not None and any(v > 0.0 for v in vector):
            return "vector"
        if str(self.option("emo_audio_prompt") or "").strip():
            return "audio"
        return "speaker"

    # ── Precision ────────────────────────────────────────────────────────────

    def _native_bf16(self) -> bool:
        """True when the bound GPU runs bfloat16 natively (not emulated)."""
        try:
            import torch
            if not torch.cuda.is_available():
                return False
            index = int(self._device.split(":")[1]) if ":" in self._device else 0
            return int(torch.cuda.get_device_properties(index).major) >= 8
        except Exception:
            return False

    def _small_vram(self) -> bool:
        """True when the bound GPU is below upstream's 10 GB low-VRAM threshold."""
        try:
            import torch
            index = int(self._device.split(":")[1]) if ":" in self._device else 0
            total = torch.cuda.get_device_properties(index).total_memory
            return total / (1024 ** 3) < _LOW_VRAM_GB
        except Exception:
            return False

    def _resolve_precision(self, model_id: str) -> str:
        """Picks float32 / float16 / bfloat16 for *model_id* on this device.

        Upstream implements exactly one half-precision type per model: FP16
        for IndexTTS-2 and BF16 for IndexTTS-2.5.

        * ``dtype_override`` wins over the ``precision`` option.
        * ``float32`` / ``full`` → FP32. CPU is always FP32.
        * IndexTTS-2: any half request, and ``auto``, → FP16.
        * IndexTTS-2.5: ``bfloat16`` / ``half`` → BF16. ``float16`` and
          ``auto`` → BF16 only where it helps: on GPUs with native BF16
          (compute capability 8+) or below 10 GB (upstream's own low-VRAM
          rule); otherwise FP32, because emulated BF16 on a T4-class GPU is
          slower than FP32 and upstream measures no speed gain from BF16.
        """
        if not self._device.startswith("cuda"):
            return "float32"
        half = _HALF_DTYPES[model_id]
        requested = (self._dtype_override or "").strip().lower().replace("torch.", "")
        if not requested:
            requested = {"half": "half", "full": "float32"}.get(self._choice("precision"), "auto")
        aliases = {
            "fp32": "float32", "float": "float32", "full": "float32",
            "fp16": "float16", "half": "half",
            "bf16": "bfloat16",
        }
        requested = aliases.get(requested, requested)
        if requested == "float32":
            return "float32"
        if requested not in ("auto", "half", "float16", "bfloat16"):
            self._warn_once(
                f"dtype:{requested}", "[IndexTTS] Unknown dtype %r; choosing precision automatically.", requested,
            )
            requested = "auto"
        if half == "float16":
            if requested == "bfloat16":
                self._warn_once("dtype:v2-bf16", "[IndexTTS] IndexTTS-2 has no BF16 mode; using FP16.")
            return "float16"
        if requested in ("bfloat16", "half"):
            return "bfloat16"
        if self._native_bf16() or self._small_vram():
            if requested == "float16":
                self._warn_once("dtype:v25-fp16", "[IndexTTS] IndexTTS-2.5 has no FP16 mode; using BF16.")
            return "bfloat16"
        if requested == "float16":
            self._warn_once(
                "dtype:v25-fp32",
                "[IndexTTS] IndexTTS-2.5 has no FP16 mode and %s has no native BF16; using FP32.",
                self._device,
            )
        return "float32"

    # ── Loading ──────────────────────────────────────────────────────────────

    def _wanted_spec(self) -> _LoadSpec:
        model_id = self.resolve_model_id()
        return _LoadSpec(
            model_id=model_id,
            precision=self._resolve_precision(model_id),
            use_cuda_kernel=bool(self.option("use_cuda_kernel")) and self._device.startswith("cuda"),
            use_deepspeed=bool(self.option("use_deepspeed")),
            use_accel=bool(self.option("use_accel")),
            use_torch_compile=bool(self.option("use_torch_compile"))
            or bool(getattr(self.config, "torch_compile", False)),
            use_qwen_emo=self._emotion_mode() in ("text", "script"),
        )

    def _ensure_initialised(self) -> None:
        """Loads the model, or reloads it when the wanted model / flags changed."""
        with self._lock:
            wanted = self._wanted_spec()
            if self._model is not None and self._load_spec is not None and self._load_spec.satisfies(wanted):
                return
            if self._model is not None:
                logger.info(
                    "[IndexTTS] Settings changed on %s (%s -> %s); reloading.",
                    self._device, self._load_spec, wanted,
                )
                self._unload()
            self._load(wanted)

    def _checkpoint_dir(self, spec: _LoadSpec) -> str:
        """Downloads (or finds in the HF cache) the checkpoint directory."""
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise RuntimeError(
                f"huggingface_hub is required to download {spec.model_id}. Install it with:\n"
                f"    {_INSTALL_COMMAND}"
            ) from exc
        ignore = None if spec.use_qwen_emo else [f"{_QWEN_EMO_DIR}/*"]
        try:
            return str(snapshot_download(spec.model_id, ignore_patterns=ignore))
        except Exception as exc:
            raise RuntimeError(f"Could not download {spec.model_id} from HuggingFace: {exc}") from exc

    def _load(self, spec: _LoadSpec) -> None:
        """Builds the upstream model for *spec* on this provider's device."""
        with _UPSTREAM_LOAD_LOCK:
            cls = _import_upstream_class(spec.model_id)  # before the multi-GB download: fail fast
            model_dir = self._checkpoint_dir(spec)
            kwargs: dict[str, Any] = {
                "cfg_path": os.path.join(model_dir, "config.yaml"),
                "model_dir": model_dir,
                "device": self._device,
                "use_cuda_kernel": spec.use_cuda_kernel,
                "use_deepspeed": spec.use_deepspeed,
                "use_accel": spec.use_accel,
                "use_torch_compile": spec.use_torch_compile,
                "use_qwen_emo": spec.use_qwen_emo,
            }
            if spec.precision == "bfloat16":
                kwargs["use_bf16"] = True
            elif spec.precision == "float16":
                kwargs["use_fp16"] = True

            parameters = inspect.signature(cls.__init__).parameters
            if not any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
                for name in [n for n in kwargs if n not in parameters]:
                    value = kwargs.pop(name)
                    if value:
                        logger.warning(
                            "[IndexTTS] The installed index-tts does not accept '%s'; it is ignored.", name,
                        )

            logger.info(
                "[IndexTTS] Loading %s on %s (%s%s)...",
                spec.model_id, self._device, spec.precision,
                ", QwenEmotion" if spec.use_qwen_emo else "",
            )
            # Upstream calls .cuda() / queries "the current device" in places.
            self.bind_device()
            try:
                model = cls(**kwargs)
            except Exception as exc:
                self._release_cuda_memory()
                raise RuntimeError(
                    f"IndexTTS failed to load {spec.model_id} on {self._device}: {type(exc).__name__}: {exc}"
                ) from exc

            # QwenEmotion is created with device_map="auto"; pin it to this
            # provider's device so one instance stays on one GPU.
            qwen_model = getattr(getattr(model, "qwen_emo", None), "model", None)
            if qwen_model is not None and hasattr(qwen_model, "to"):
                try:
                    qwen_model.to(self._device)
                except Exception as exc:
                    logger.warning("[IndexTTS] Could not move QwenEmotion to %s: %s", self._device, exc)

            normalizer = self._normalizer(model)
            self._base_glossary = dict(getattr(normalizer, "term_glossary", None) or {})
            self._glossary_key = None
            self._upstream_low_vram = getattr(model, "low_vram", None)
            self._infer_params = frozenset(inspect.signature(model.infer).parameters)
            self._model = model
            self._load_spec = spec
            logger.info("[IndexTTS] %s ready on %s.", spec.model_id, self._device)

    @staticmethod
    def _release_cuda_memory() -> None:
        gc.collect()
        torch = sys.modules.get("torch")
        if torch is None:
            return
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as exc:
            logger.debug("[IndexTTS] torch.cuda.empty_cache() failed: %s", exc)

    def _unload(self) -> None:
        self._model = None
        self._load_spec = None
        self._infer_params = frozenset()
        self._upstream_low_vram = None
        self._base_glossary = {}
        self._glossary_key = None
        self._release_cuda_memory()

    def cleanup(self) -> None:
        """Drops the model and frees its GPU memory."""
        with self._lock:
            self._unload()

    # ── Per-call preparation ─────────────────────────────────────────────────

    def _stable_audio_path(self, path: str) -> str:
        """Returns a content-addressed path for the audio file *path*.

        Upstream keeps one cached speaker conditioning and one cached emotion
        conditioning, each keyed on the *path string* it was computed from
        (``cache_spk_audio_prompt`` / ``cache_emo_audio_prompt``). Voice bytes
        from the pipeline already arrive as a content-addressed file via
        ``resolve_voice_path``. A caller-supplied path is copied to one here,
        so a file that is replaced under the same name gets new conditioning
        instead of a stale cache hit, and an unchanged file is conditioned
        exactly once per run.
        """
        stat = os.stat(path)
        key = (os.path.abspath(path), stat.st_mtime_ns, stat.st_size)
        cached = self._stable_paths.get(key)
        if cached and os.path.exists(cached):
            return cached
        digest = hashlib.sha256()
        with open(path, "rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                digest.update(block)
        directory = os.path.join(tempfile.gettempdir(), "abm_voice_refs")
        extension = os.path.splitext(path)[1].lower() or ".wav"
        stable = os.path.join(directory, f"{digest.hexdigest()[:20]}{extension}")
        if os.path.abspath(path) != stable and (
            not os.path.exists(stable) or os.path.getsize(stable) != stat.st_size
        ):
            os.makedirs(directory, exist_ok=True)
            tmp_path = f"{stable}.{os.getpid()}.{threading.get_ident()}.tmp"
            with open(path, "rb") as src, open(tmp_path, "wb") as dst:
                for block in iter(lambda: src.read(1 << 20), b""):
                    dst.write(block)
            os.replace(tmp_path, stable)
        if len(self._stable_paths) >= _MAX_STABLE_PATHS:
            self._stable_paths.clear()
        self._stable_paths[key] = stable
        return stable

    def _voice_path(self, voice_ref: str | bytes | None) -> str:
        path = self.resolve_voice_path(voice_ref)
        if not path:
            raise RuntimeError(
                "IndexTTS clones the narrator from a reference clip and has no built-in voices; "
                "no voice reference was given."
            )
        if not os.path.isfile(path):
            raise RuntimeError(f"IndexTTS voice reference does not exist: {path}")
        if isinstance(voice_ref, (bytes, bytearray)):
            return path  # written content-addressed by resolve_voice_path()
        return self._stable_audio_path(path)

    def _language_code(self) -> str:
        """Maps ``config.language`` (or the ``lang`` option) to an upstream code.

        Raises
        ------
        RuntimeError
            If the language is not one the loaded checkpoint was trained on.
        """
        model_id = self._load_spec.model_id if self._load_spec else self.resolve_model_id()
        supported = _MODEL_LANGUAGES[model_id]
        code = self._choice("lang")
        if code == "auto":
            name = (getattr(self.config, "language", "") or "").strip()
            key = name.lower()
            code = _LANGUAGE_CODES.get(key) or _LANGUAGE_CODES.get(re.split(r"[-_ (]", key)[0], "")
            if not code:
                raise RuntimeError(
                    f"{_DISPLAY_NAMES[model_id]} cannot narrate {name or 'an unspecified language'}; it supports "
                    f"{', '.join(_LANGUAGE_NAMES[c] for c in supported)}. Pick another TTS engine, or set the "
                    "IndexTTS 'lang' option to force one of its languages."
                )
        if code not in supported:
            raise RuntimeError(
                f"{_DISPLAY_NAMES[model_id]} does not support {_LANGUAGE_NAMES.get(code, code)}; it supports "
                f"{', '.join(_LANGUAGE_NAMES[c] for c in supported)}."
            )
        return code

    @staticmethod
    def _normalizer(model: Any) -> Any:
        """Upstream's text normaliser: ``text_process`` (2.5) or ``normalizer`` (2)."""
        return getattr(model, "text_process", None) or getattr(model, "normalizer", None)

    def _apply_glossary(self) -> None:
        """Loads, reloads or removes the user's pronunciation glossary."""
        path = str(self.option("glossary_file") or "").strip()
        key: tuple[str, int, int] | None = None
        if path:
            path = os.path.abspath(os.path.expanduser(path))
            if not os.path.isfile(path):
                raise RuntimeError(f"IndexTTS glossary file does not exist: {path}")
            stat = os.stat(path)
            key = (path, stat.st_mtime_ns, stat.st_size)
        if key == self._glossary_key:
            return
        normalizer = self._normalizer(self._model)
        if key is not None:
            loader = getattr(normalizer, "load_glossary_from_yaml", None)
            if loader is None or not loader(path):
                raise RuntimeError(
                    f"IndexTTS could not load the glossary {path}: it must be a YAML mapping of term to reading."
                )
        elif normalizer is not None and hasattr(normalizer, "term_glossary"):
            normalizer.term_glossary = dict(self._base_glossary)
        self._glossary_key = key

    def _build_infer_kwargs(self, text: str, voice_path: str) -> dict[str, Any]:
        """Builds the keyword arguments for upstream ``IndexTTS2.infer``."""
        model = self._model
        params = self._infer_params
        config = self.config
        kwargs: dict[str, Any] = {
            "spk_audio_prompt": voice_path,
            "text": text,
            "output_path": None,   # makes upstream return (sample_rate, int16 samples)
            "verbose": bool(self.option("verbose")),
        }

        code = self._language_code()
        if "lang" in params:
            kwargs["lang"] = code

        # ── Emotion ──────────────────────────────────────────────────────────
        mode = self._emotion_mode()
        if mode != "speaker":
            kwargs["emo_alpha"] = self._number("emo_alpha")
            kwargs["use_random"] = bool(self.option("use_random"))
        if mode == "audio":
            emo_path = str(self.option("emo_audio_prompt") or "").strip()
            if not emo_path or not os.path.isfile(os.path.expanduser(emo_path)):
                raise RuntimeError(
                    f"IndexTTS emotion_mode is 'audio' but the emotion reference clip is missing: {emo_path!r}."
                )
            kwargs["emo_audio_prompt"] = self._stable_audio_path(os.path.expanduser(emo_path))
        elif mode == "vector":
            vector = _parse_emo_vector(self.option("emo_vector"))
            if vector is None:
                raise RuntimeError("IndexTTS emotion_mode is 'vector' but no emo_vector is set.")
            normalise = getattr(model, "normalize_emo_vec", None)
            kwargs["emo_vector"] = list(normalise(vector, apply_bias=True)) if normalise else vector
        elif mode == "text":
            description = (getattr(config, "tts_instruct", "") or "").strip()
            if not description:
                raise RuntimeError(
                    "IndexTTS emotion_mode is 'text' but the style prompt (tts_instruct) is empty. "
                    "Describe the emotion there, or use emotion_mode 'script'."
                )
            kwargs["use_emo_text"] = True
            kwargs["emo_text"] = description
        elif mode == "script":
            kwargs["use_emo_text"] = True
            kwargs["emo_text"] = None   # upstream then classifies the chunk text itself

        # ── Segmentation ─────────────────────────────────────────────────────
        segment_tokens = int(self._number("max_text_tokens_per_segment"))
        text_limit = self._model_limit("max_text_tokens")
        if text_limit:
            segment_tokens = min(segment_tokens, text_limit)
        kwargs["max_text_tokens_per_segment"] = max(_MIN_SEGMENT_TOKENS, segment_tokens)
        kwargs["interval_silence"] = int(self._number("interval_silence"))

        # ── Speed ────────────────────────────────────────────────────────────
        speed = self._config_float("speed", 1.0)
        if speed <= 0.0:
            speed = 1.0
        if "duration_factor" in params:
            # duration_factor stretches the duration, so it is the inverse of speed.
            kwargs["duration_factor"] = round(_clamp(1.0 / speed, _MIN_DURATION_FACTOR, _MAX_DURATION_FACTOR), 4)
        elif abs(speed - 1.0) > 1e-3:
            self._warn_once("speed", "[IndexTTS] %s has no speed control; speed=%s is ignored.", self.get_name(), speed)
        if "text_normalization" in params:
            kwargs["text_normalization"] = bool(self.option("text_normalization"))

        # ── Sampling (forwarded by upstream to the GPT decoder) ──────────────
        top_k = int(self._config_float("top_k", 30))
        kwargs["do_sample"] = bool(self.option("do_sample"))
        kwargs["temperature"] = _clamp(self._config_float("temperature", 0.8), 0.1, 2.0)
        kwargs["top_p"] = _clamp(self._config_float("top_p", 0.8), 0.01, 1.0)
        kwargs["top_k"] = top_k if top_k > 0 else None
        kwargs["num_beams"] = int(self._number("num_beams"))
        kwargs["repetition_penalty"] = self._number("repetition_penalty")
        kwargs["length_penalty"] = self._number("length_penalty")
        if bool(self.option("typical_sampling")):
            kwargs["typical_sampling"] = True
            kwargs["typical_mass"] = self._number("typical_mass")

        max_mel_tokens = int(self._number("max_mel_tokens"))
        mel_limit = self._model_limit("max_mel_tokens")
        if mel_limit:
            max_mel_tokens = min(max_mel_tokens, mel_limit)
        if bool(self.option("adaptive_max_mel_tokens")):
            max_mel_tokens = min(max_mel_tokens, _estimate_mel_token_bound(text))
        kwargs["max_mel_tokens"] = max(_MIN_MEL_TOKENS, max_mel_tokens)
        return kwargs

    def _run_infer(self, kwargs: dict[str, Any]) -> Any:
        """Calls upstream ``infer``; on CUDA OOM retries once with half-size segments."""
        import torch

        model = self._model
        oom_error = getattr(torch, "OutOfMemoryError", None) or torch.cuda.OutOfMemoryError

        def run(call_kwargs: dict[str, Any]) -> Any:
            self.seed_everything()
            seed = getattr(self.config, "seed", -1)
            if seed is not None and int(seed) >= 0:
                random.seed(int(seed))   # upstream's use_random draws from `random`
            # Upstream computes the cached reference conditioning outside
            # torch.no_grad(); wrap the whole call so no graph is retained.
            with torch.no_grad():
                return model.infer(**call_kwargs)

        try:
            return run(kwargs)
        except oom_error:
            # Retry below, outside this handler: the exception's traceback
            # keeps the failed call's tensors alive until the block is left.
            pass
        self._release_cuda_memory()
        smaller = max(_MIN_SEGMENT_TOKENS, int(kwargs["max_text_tokens_per_segment"]) // 2)
        logger.warning(
            "[IndexTTS] CUDA out of memory on %s; retrying with %d text tokens per segment.",
            self._device, smaller,
        )
        try:
            return run(dict(kwargs, max_text_tokens_per_segment=smaller))
        except oom_error as exc:
            self._release_cuda_memory()
            raise RuntimeError(
                f"IndexTTS ran out of GPU memory on {self._device} even with {smaller}-token segments. "
                "Lower max_text_tokens_per_segment / num_beams or enable half precision."
            ) from exc

    def _to_waveform(self, result: Any) -> tuple[Any, int]:
        """Converts upstream's ``(sample_rate, int16 samples)`` to float32 in [-1, 1]."""
        import numpy as np

        if result is None:
            raise RuntimeError(
                f"{self.get_name()} produced no audio on {self._device} "
                "(the text was empty after normalisation, or every segment failed)."
            )
        if not isinstance(result, (tuple, list)) or len(result) != 2:
            raise RuntimeError(f"{self.get_name()} returned an unexpected result: {type(result).__name__}.")
        sample_rate, samples = result
        if hasattr(samples, "detach"):
            samples = samples.detach().cpu().numpy()
        samples = np.asarray(samples)
        if samples.dtype.kind in "iu":
            audio = samples.astype(np.float32) / _PCM16_MAX
        else:
            audio = samples.astype(np.float32)
        return audio, int(sample_rate)

    # ── Synthesis ────────────────────────────────────────────────────────────

    def synthesize(
        self,
        text: str,
        voice_ref: str | bytes,
        out_path: str | None = None,
        *,
        return_bytes: bool = False,
    ) -> tuple[str | bytes, float]:
        """Synthesizes one chunk in the narrator's cloned voice.

        Parameters
        ----------
        text : str
            Chunk text (up to ``config.max_len`` characters). IndexTTS-2.5
            pronunciation tags (``<word|reading>``) are passed through.
        voice_ref : str | bytes
            Narrator reference clip: a path or WAV bytes. Only the first
            15 s are used by the model.
        out_path : str | None
            Where to write the WAV when ``return_bytes`` is False.
        return_bytes : bool
            Return WAV bytes instead of writing ``out_path``.

        Returns
        -------
        tuple[str | bytes, float]
            ``(out_path or wav_bytes, duration_seconds)`` at 22.05 kHz.

        Raises
        ------
        RuntimeError
            If the model cannot be loaded or the chunk cannot be synthesized.
        """
        if not text or not text.strip():
            raise RuntimeError("IndexTTS was given an empty text chunk.")
        with self._lock:
            self._ensure_initialised()
            self.bind_device()
            voice_path = self._voice_path(voice_ref)
            self._apply_glossary()
            if self._upstream_low_vram is not None:
                split = self._choice("low_vram_split")
                self._model.low_vram = self._upstream_low_vram if split == "auto" else split == "on"
            kwargs = self._build_infer_kwargs(text.strip(), voice_path)
            try:
                result = self._run_infer(kwargs)
            except RuntimeError:
                raise
            except Exception as exc:
                raise RuntimeError(
                    f"{self.get_name()} synthesis failed on {self._device}: {type(exc).__name__}: {exc}"
                ) from exc
        audio, sample_rate = self._to_waveform(result)
        return self.finish(audio, sample_rate, out_path, return_bytes)
