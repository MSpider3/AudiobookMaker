"""
audiobook_factory/tts_providers/fish_provider.py
=================================================
Fish Audio S2 Pro voice-cloning provider, driving the upstream ``fish_speech``
package in-process (no HTTP server, no worker queue).

Upstream source read for this module
------------------------------------
https://github.com/fishaudio/fish-speech at commit
``214da3cd841bda85da2496b96cd3c4d7edb1337e`` (2026-09-17). The calls made here
are the ones the upstream command line makes
(``fish_speech/models/text2semantic/inference.py``):

* ``load_codec_model(codec_checkpoint_path, device, precision)`` - DAC codec.
* ``DualARTransformer.from_pretrained(path, load_weights=True, max_length=N)``
  followed by the same steps as upstream ``init_model`` (``init_model`` itself
  cannot pass ``max_length``, see *Memory* below).
* ``codec.encode(audios, audio_lengths)`` - reference clip to prompt tokens.
* ``generate_long(model=, device=, decode_one_token=, text=, ...)`` - text to
  codec tokens.
* ``decode_to_audio(codes, codec)`` - codec tokens to a 44.1 kHz waveform.

Memory
------
The published checkpoint declares ``max_seq_len=32768``. Upstream allocates
the KV cache and a square attention mask for that whole length (about
5.5 GiB), and the codec carries three more unused 1 GiB mask buffers. An
audiobook chunk needs a few thousand positions, so this provider loads the
model with a shorter context (``max_seq_len`` option, default 4096) and drops
the codec's unused masks. That is what brings the model from roughly 22 GB
down to about 12 GB.

Host RAM is the other limit. Upstream ``from_pretrained`` builds the 4.56 B
parameter model with random float32 weights (17 GiB resident) before it
reads the 8.5 GiB checkpoint. The ``low_ram_load`` option skips that random
initialisation, after checking that the checkpoint supplies every weight.

Voice presets
-------------
``save_voice_preset`` stores the reference clip's codec tokens and transcript
in a ``.pt`` file of tensors and plain values; ``config.voice_preset`` then
replaces the reference clip. The ``.npy`` token files written by fish-speech's
own tools are accepted too.

Concurrency
-----------
One instance owns one device. Every model object lives on the instance; the
only state shared between instances is a load lock and a guard that restores
PyTorch's process-wide attention-kernel flags, which upstream toggles for
every generated token.
"""
from __future__ import annotations

import contextlib
import gc
import hashlib
import importlib
import json
import logging
import math
import os
import re
import sys
import threading
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar, Iterator

from audiobook_factory.tts_providers.base_tts_provider import (
    BaseTTSProvider,
    ProviderInfo,
    ProviderOption,
)

if TYPE_CHECKING:
    from audiobook_factory.pipeline import AudiobookConfig

logger = logging.getLogger(__name__)

# ── Upstream package ─────────────────────────────────────────────────────────
_UPSTREAM_COMMIT: str = "214da3cd841bda85da2496b96cd3c4d7edb1337e"
_UPSTREAM_REQUIREMENT: str = (
    f"fish-speech @ git+https://github.com/fishaudio/fish-speech.git@{_UPSTREAM_COMMIT}"
)
_UPSTREAM_INFERENCE_MODULE: str = "fish_speech.models.text2semantic.inference"
_UPSTREAM_LLAMA_MODULE: str = "fish_speech.models.text2semantic.llama"
_REPO_ENV_VAR: str = "ABM_FISH_REPO"
_CHECKPOINT_ENV_VAR: str = "ABM_FISH_CHECKPOINT_DIR"
_CODEC_FILE_NAME: str = "codec.pth"

# Dependencies that are safe to install with normal dependency resolution.
_PIP_REQUIREMENTS: tuple[str, ...] = (
    "einops",
    "loguru",
    "loralib",
    "hydra-core>=1.3.2",
    "argbind>=0.3.7",
    "flatten-dict",
    "julius",
    "ffmpy",
    "importlib-resources",
    "randomname",
    "tensorboard",
    "soundfile",
    "librosa",
)
# fish-speech pins torch==2.8.0 and pulls pyaudio, gradio, wandb and more;
# descript-audiotools pins protobuf<3.20. Both must be installed without
# their declared dependencies.
_INSTALL_COMMAND: str = (
    "pip install -r requirements/tts-fish.txt && "
    "pip install --no-deps descript-audiotools descript-audio-codec "
    f'"{_UPSTREAM_REQUIREMENT}"'
)

# ── Model facts ──────────────────────────────────────────────────────────────
_DEFAULT_MODEL_ID: str = "fishaudio/s2-pro"
_NATIVE_SAMPLE_RATE: int = 44100
_CODEC_FRAME_RATE_HZ: float = 44100.0 / 2048.0  # ~21.5 codec frames per second
_CHECKPOINT_BYTES_PER_PARAM: int = 2  # published weights are bfloat16
_CODEC_CHECKPOINT_BYTES_PER_PARAM: int = 4  # codec.pth is float32
_CODEC_MASK_BUFFER_BYTES: int = 3 * 32768 * 32768  # three unused bool masks
_UPSTREAM_PROMPT_RESERVE: int = 2048  # generate_long: prompt <= max_seq_len - 2048
_DEFAULT_MAX_SEQ_LEN: int = 4096
_MIN_MAX_SEQ_LEN: int = 3072
_QUANTIZED_PATH_MARKERS: tuple[str, ...] = ("int8", "int4")

# ── Generation bounds ────────────────────────────────────────────────────────
_TOKEN_BUDGET_BASE: int = 64
_MIN_TOKEN_BUDGET: int = 32
_PROMPT_OVERHEAD_TOKENS: int = 64
_MIN_TEMPERATURE: float = 0.01
_MAX_TEMPERATURE: float = 1.99
_MIN_TOP_P: float = 0.01
_TOP_K_DISABLED: int = 1_000_000
_RECOMMENDED_MAX_REFERENCE_SECONDS: float = 30.0
_RECOMMENDED_MIN_REFERENCE_SECONDS: float = 5.0
_MAX_REFERENCE_CACHE: int = 4
_PRESET_FORMAT: str = "abm-fish-voice-preset"
_PRESET_FORMAT_VERSION: int = 1

# ── Memory guards ────────────────────────────────────────────────────────────
_BYTES_PER_GIB: float = 1024.0 ** 3
_VRAM_WORKING_MARGIN_BYTES: int = 1024 ** 3  # prompt prefill + codec activations
_OOM_RETRY_GAIN_BYTES: int = 1024 ** 3
_CODEC_BUILD_RAM_BYTES: int = 13 * 1024 ** 3 // 2  # ~6.5 GiB peak while the codec is built
_INIT_FUNCTION_NAMES: tuple[str, ...] = (
    "uniform_", "normal_", "trunc_normal_", "kaiming_uniform_", "kaiming_normal_",
    "xavier_uniform_", "xavier_normal_",
)

_PRECISIONS: tuple[str, ...] = ("bfloat16", "float16", "float32")
_PRECISION_ALIASES: dict[str, str] = {
    "bf16": "bfloat16", "fp16": "float16", "half": "float16", "fp32": "float32",
    "torch.bfloat16": "bfloat16", "torch.float16": "float16", "torch.float32": "float32",
}
_PRECISION_ITEMSIZE: dict[str, int] = {"bfloat16": 2, "float16": 2, "float32": 4}

_CONTROL_TOKEN_RE: re.Pattern[str] = re.compile(r"<\|[^<>|]{0,64}\|>")
_INLINE_TAG_RE: re.Pattern[str] = re.compile(r"\[([^\[\]\n]{1,120})\]")
_WHITESPACE_RE: re.Pattern[str] = re.compile(r"\s+")
_SPEAKER_TAG: str = "<|speaker:0|>"

_LANGUAGES: tuple[str, ...] = (
    "Japanese", "English", "Chinese",
    "Korean", "Spanish", "Portuguese", "Arabic", "Russian", "French", "German",
    "Swedish", "Italian", "Turkish", "Norwegian", "Dutch", "Welsh", "Basque",
    "Catalan", "Danish", "Galician", "Tamil", "Hungarian", "Finnish", "Polish",
    "Estonian", "Hindi", "Latin", "Urdu", "Thai", "Vietnamese", "Javanese",
    "Bengali", "Yoruba", "Slovenian", "Czech", "Swahili", "Norwegian Nynorsk",
    "Hebrew", "Malay", "Ukrainian", "Indonesian", "Kazakh", "Bulgarian",
    "Latvian", "Burmese", "Tagalog", "Slovak", "Nepali", "Persian", "Afrikaans",
    "Greek", "Tibetan", "Croatian", "Romanian", "Shona", "Maori", "Yiddish",
    "Amharic", "Belarusian", "Khmer", "Icelandic", "Azerbaijani", "Sindhi",
    "Breton", "Albanian", "Pashto", "Mongolian", "Haitian Creole", "Malayalam",
    "Serbian", "Sanskrit", "Telugu", "Georgian", "Bosnian", "Punjabi",
    "Lithuanian", "Kannada", "Sinhala", "Armenian", "Marathi", "Assamese",
    "Gujarati", "Faroese",
)

# Serialises imports, downloads and model construction across instances: the
# codec alone needs ~7 GiB of host RAM while it is being built.
_LOAD_LOCK = threading.RLock()


class _SdpBackendGuard:
    """Restores PyTorch's process-wide attention-kernel flags after generation.

    Upstream wraps every decoded token in ``sdpa_kernel(SDPBackend.MATH)``.
    That context manager saves and restores *global* flags, so two GPUs
    generating in two threads can leave the process stuck on the slow math
    kernel for every model. The flags seen before the first concurrent
    generation are put back when the last one finishes.
    """

    _FLAGS: ClassVar[tuple[tuple[str, str], ...]] = (
        ("flash_sdp_enabled", "enable_flash_sdp"),
        ("mem_efficient_sdp_enabled", "enable_mem_efficient_sdp"),
        ("math_sdp_enabled", "enable_math_sdp"),
        ("cudnn_sdp_enabled", "enable_cudnn_sdp"),
    )
    _lock: ClassVar[threading.Lock] = threading.Lock()
    _active: ClassVar[int] = 0
    _snapshot: ClassVar[dict[str, bool] | None] = None

    @classmethod
    @contextlib.contextmanager
    def preserve(cls) -> Iterator[None]:
        """Context manager around one upstream generation call."""
        import torch

        backend = torch.backends.cuda
        with cls._lock:
            if cls._active == 0:
                cls._snapshot = {
                    setter: bool(getattr(backend, getter)())
                    for getter, setter in cls._FLAGS
                    if hasattr(backend, getter) and hasattr(backend, setter)
                }
            cls._active += 1
        try:
            yield
        finally:
            with cls._lock:
                cls._active -= 1
                if cls._active == 0 and cls._snapshot is not None:
                    for setter, enabled in cls._snapshot.items():
                        try:
                            getattr(backend, setter)(enabled)
                        except Exception as exc:  # pragma: no cover - defensive
                            logger.debug("Could not restore %s: %s", setter, exc)
                    cls._snapshot = None


@contextlib.contextmanager
def _skip_random_init(llama: Any) -> Iterator[None]:
    """Makes weight initialisation a no-op while a model is constructed.

    Same technique as ``transformers``' ``no_init_weights``: the
    ``torch.nn.init`` functions used by ``nn.Linear`` and ``nn.Embedding`` and
    upstream's own ``_init_weights`` are replaced for the duration of the
    block, so freshly allocated weights are never written and never become
    resident memory. Process-wide while active; callers hold ``_LOAD_LOCK``.
    """
    import torch

    init = torch.nn.init
    saved = {name: getattr(init, name) for name in _INIT_FUNCTION_NAMES if hasattr(init, name)}
    base = getattr(llama, "BaseTransformer", None)
    saved_method = base.__dict__.get("_init_weights") if base is not None else None

    def _keep(tensor: Any, *args: Any, **kwargs: Any) -> Any:
        return tensor

    try:
        for name in saved:
            setattr(init, name, _keep)
        if saved_method is not None:
            base._init_weights = lambda self, module: None
        yield
    finally:
        for name, function in saved.items():
            setattr(init, name, function)
        if saved_method is not None:
            base._init_weights = saved_method


@dataclass(frozen=True)
class _LoadKey:
    """Everything that decides which weights are loaded and how."""

    model_id: str
    checkpoint_override: str
    precision: str
    codec_precision: str
    max_seq_len: int
    compile: bool
    trim_codec: bool


@dataclass(frozen=True)
class _ReferencePrompt:
    """A reference clip encoded to codec tokens, with its transcript."""

    codes: Any  # CPU LongTensor, shape (num_codebooks, frames)
    transcript: str
    frames: int


def _silence_upstream(inference: Any) -> None:
    """Stops fish-speech printing a prompt dump, progress bar and logs per chunk.

    Process-wide and display-only: it disables the ``fish_speech`` loguru
    namespace and replaces two display helpers on the upstream module.
    """
    try:
        from loguru import logger as upstream_logger

        upstream_logger.disable("fish_speech")
    except Exception as exc:
        logger.debug("Could not silence fish_speech logging: %s", exc)
    conversation_cls = getattr(inference, "Conversation", None)
    if conversation_cls is not None and hasattr(conversation_cls, "visualize"):
        conversation_cls.visualize = lambda self, *args, **kwargs: None
    if hasattr(inference, "tqdm"):
        inference.tqdm = lambda iterable=None, *args, **kwargs: iterable


class FishSpeechProvider(BaseTTSProvider):
    """Fish Audio S2 Pro: zero-shot voice cloning with inline ``[style]`` tags.

    Parameters
    ----------
    config : AudiobookConfig
        Pipeline configuration; read again on every call.
    device : str | None
        Torch device this instance is bound to, e.g. ``"cuda:0"``.
    dtype_override : str | None
        ``"float16"``, ``"bfloat16"`` or ``"float32"``; wins over the
        ``precision`` option.
    """

    INFO = ProviderInfo(
        name="fish",
        display_name="Fish Audio S2 Pro",
        description=(
            "4B-parameter multilingual voice cloning with free-form inline emotion and "
            "prosody tags such as [whisper] or [excited]. Non-commercial licence; needs "
            "about 12 GB of VRAM and is slower than real time on a T4."
        ),
        license="Fish Audio Research License (research / non-commercial only)",
        commercial_use=False,
        homepage="https://github.com/fishaudio/fish-speech",
        default_model=_DEFAULT_MODEL_ID,
        models=(_DEFAULT_MODEL_ID,),
        native_sample_rate=_NATIVE_SAMPLE_RATE,
        min_vram_gb=12.0,
        languages=_LANGUAGES,
        supports_voice_clone=True,
        transcript="required",
        supports_instruct=True,
        supports_batch=False,
        supports_speed=False,
        supports_seed=True,
        supports_voice_preset=True,
        preset_voices=(),
        # Defaults shared by upstream's API server, WebUI and SGLang runner.
        recommended_settings={
            "temperature": 0.8,
            "top_p": 0.8,
            "top_k": 30,
            "repetition_penalty": 1.1,
        },
        options=(
            ProviderOption(
                key="precision", label="Model precision", kind="choice", default="auto",
                choices=("auto", "bfloat16", "float16", "float32"),
                help=(
                    "auto = bfloat16 on GPUs that support it natively, float16 on older "
                    "GPUs such as the T4, float32 on CPU."
                ),
            ),
            ProviderOption(
                key="max_seq_len", label="Context length (tokens)", kind="int",
                default=_DEFAULT_MAX_SEQ_LEN, minimum=0, maximum=32768, step=1024,
                help=(
                    "Positions the KV cache is allocated for. 4096 covers a 30 s reference "
                    "plus one chunk; 0 uses the checkpoint's 32768 and costs about 5 GiB more."
                ),
            ),
            ProviderOption(
                key="max_new_tokens", label="Max codec frames per chunk", kind="int",
                default=2048, minimum=0, maximum=8192, step=64,
                help=(
                    "Hard cap on generated frames (about 21.5 per second of audio). "
                    "0 keeps only the text-length budget."
                ),
            ),
            ProviderOption(
                key="tokens_per_byte", label="Frame budget per text byte", kind="float",
                default=4.0, minimum=1.5, maximum=12.0, step=0.5,
                help="Frames allowed per UTF-8 byte of text before a chunk counts as runaway.",
            ),
            ProviderOption(
                key="runaway_retries", label="Retries for runaway chunks", kind="int",
                default=1, minimum=0, maximum=3, step=1,
                help="Extra attempts when a chunk hits its frame budget or yields no audio.",
            ),
            ProviderOption(
                key="speaker_tag", label="Bind text to the reference speaker", kind="bool",
                default=True,
                help="Prefixes each chunk with <|speaker:0|>, as the official server does.",
            ),
            ProviderOption(
                key="instruct_as_tag", label="Use style prompt as inline tag", kind="bool",
                default=True,
                help="Prepends the style prompt to every chunk as a [free-form tag].",
            ),
            ProviderOption(
                key="inline_tags", label="Bracketed text in the book", kind="choice",
                default="keep", choices=("keep", "strip", "speak"),
                help=(
                    "keep = [tags] steer emotion and prosody; strip = remove them; "
                    "speak = read them aloud as (text)."
                ),
            ),
            ProviderOption(
                key="allow_missing_transcript", label="Clone without a transcript", kind="bool",
                default=False,
                help="Proceed when the reference clip has no transcript (lower similarity).",
            ),
            ProviderOption(
                key="codec_precision", label="Codec precision", kind="choice",
                default="model", choices=("model", "float32"),
                help="model = same as the language model; float32 costs about 2 GiB more.",
            ),
            ProviderOption(
                key="trim_codec_buffers", label="Drop unused codec masks", kind="bool",
                default=True,
                help="Frees three unused 1 GiB attention-mask buffers in the codec.",
            ),
            ProviderOption(
                key="low_ram_load", label="Low-RAM model loading", kind="bool",
                default=True,
                help=(
                    "Skips the random initialisation fish-speech performs before reading "
                    "the weights; saves about 17 GiB of host RAM and a minute of load time."
                ),
            ),
            ProviderOption(
                key="verbose_upstream", label="Show fish-speech output", kind="bool",
                default=False,
                help="Keep fish-speech's per-chunk prompt dump, progress bar and logs.",
            ),
        ),
        pip_requirements=_PIP_REQUIREMENTS,
        install_notes=(
            "Install in two steps so pip does not replace torch or downgrade protobuf: "
            f"{_INSTALL_COMMAND} . torch, torchaudio (matching torch) and transformers "
            "must already be installed; upstream supports transformers<=4.57.3. "
            f"Alternative to the fish-speech wheel: git clone the repository and set "
            f"{_REPO_ENV_VAR} to the clone; the two descript packages are still required. "
            f"Weights (11 GB) download from Hugging Face on first use; set "
            f"{_CHECKPOINT_ENV_VAR} to a local copy to skip that. fishaudio/s2-pro was "
            "not gated when this was written; if it becomes gated, accept the terms on "
            "the model page and set HF_TOKEN. Code and weights are under the Fish Audio "
            "Research License: research and non-commercial use only, commercial use "
            "needs a licence from business@fish.audio. No system packages are needed "
            "for WAV or FLAC reference clips. Loading needs about 10 GiB of free host "
            "RAM per instance (27 GiB with the low_ram_load option off). Set "
            "torch_compile for a faster decode loop on Linux (first chunk compiles "
            "for a minute or two)."
        ),
    )

    def __init__(
        self,
        config: "AudiobookConfig",
        device: str | None = None,
        dtype_override: str | None = None,
    ) -> None:
        super().__init__(config)
        self._device: str = device or getattr(config, "device", "cuda") or "cuda"
        self._dtype_override: str | None = dtype_override
        self._lock = threading.RLock()

        self._model: Any = None
        self._codec: Any = None
        self._decode_one_token: Any = None
        self._inference: Any = None
        self._loaded: _LoadKey | None = None
        self._context_len: int = 0
        self._compile_warmed: bool = False

        self._references: OrderedDict[tuple[str, str], _ReferencePrompt] = OrderedDict()
        self._preset_meta: dict[str, dict[str, Any]] = {}
        self._path_digests: dict[str, tuple[int, int, str]] = {}
        self._oom_memo: tuple[_LoadKey, int, str] | None = None
        self._warned: set[str] = set()

    # ── Contract ─────────────────────────────────────────────────────────────

    @property
    def device(self) -> str:
        """The torch device string this provider is bound to."""
        return self._device

    def ensure_ready(self) -> None:
        """Loads the model, or reloads it when the configuration changed.

        Raises
        ------
        RuntimeError
            If fish-speech is not installed, the weights cannot be fetched,
            or the device plainly lacks the VRAM to hold the model.
        """
        self._ensure_initialised()

    def synthesize(
        self,
        text: str,
        voice_ref: str | bytes,
        out_path: str | None = None,
        *,
        return_bytes: bool = False,
    ) -> tuple[str | bytes, float]:
        """Generates speech for *text* in the voice of *voice_ref*.

        Parameters
        ----------
        text : str
            One chunk of narration. ``[tags]`` steer emotion and prosody.
        voice_ref : str | bytes
            Reference clip as a path or WAV bytes.
        out_path : str | None
            Destination WAV path when *return_bytes* is False.
        return_bytes : bool
            Return WAV bytes instead of writing *out_path*.

        Returns
        -------
        tuple[str | bytes, float]
            ``(out_path or wav_bytes, duration_seconds)`` at 44.1 kHz.

        Raises
        ------
        RuntimeError
            If the chunk cannot be synthesized.
        """
        with self._lock:
            self._ensure_initialised()
            self.bind_device()
            prompt = self._reference_prompt(voice_ref)
            audio, sample_rate = self._generate(text, prompt)
        return self.finish(audio, sample_rate, out_path, return_bytes)

    def synthesize_batch(
        self,
        texts: list[str],
        voice_ref: bytes,
        *,
        return_bytes: bool = True,
    ) -> list[tuple[bytes | str, float]]:
        """Synthesizes *texts* one at a time, in input order.

        Upstream generation is fixed at batch size 1, so there is no real
        batch. A chunk that runs out of GPU memory is retried once after the
        allocator cache is emptied.
        """
        self._ensure_initialised()
        results: list[tuple[bytes | str, float]] = []
        for text in texts:
            try:
                results.append(self.synthesize(text, voice_ref, return_bytes=True))
            except Exception as exc:
                if not self._is_out_of_memory(exc):
                    raise
                logger.warning(
                    "[Fish] Out of memory on %s; emptying the cache and retrying the chunk.",
                    self._device,
                )
                self._empty_cuda_cache()
                results.append(self.synthesize(text, voice_ref, return_bytes=True))
        return results

    def cleanup(self) -> None:
        """Drops every model and returns cached GPU memory."""
        with self._lock:
            self._unload()
            self._oom_memo = None

    # ── Settings ─────────────────────────────────────────────────────────────

    def _choice(self, key: str) -> str:
        """Reads a ``choice`` option, falling back to its default when invalid."""
        declared = next(o for o in self.INFO.options if o.key == key)
        value = str(self.option(key) or declared.default).strip().lower()
        return value if value in declared.choices else str(declared.default)

    def _warn_once(self, key: str, message: str, *args: Any) -> None:
        if key not in self._warned:
            self._warned.add(key)
            logger.warning(message, *args)

    def _is_cuda(self) -> bool:
        return self._device.startswith("cuda")

    def _cuda_index(self) -> int:
        return int(self._device.split(":")[1]) if ":" in self._device else 0

    def _resolve_precision(self) -> str:
        """Returns ``"bfloat16"``, ``"float16"`` or ``"float32"`` for the model."""
        requested = (self._dtype_override or self._choice("precision")).strip().lower()
        requested = _PRECISION_ALIASES.get(requested, requested)
        if not self._is_cuda():
            if requested == "float16":
                self._warn_once(
                    "cpu_fp16", "[Fish] float16 is not usable on %s; using float32.", self._device
                )
                return "float32"
            return requested if requested in _PRECISIONS else "float32"
        if requested in _PRECISIONS:
            return requested
        try:
            import torch

            major = int(torch.cuda.get_device_capability(self._cuda_index())[0])
        except Exception as exc:
            logger.debug("[Fish] Could not read device capability: %s", exc)
            major = 0
        # Upstream runs bfloat16 and documents --half for GPUs without it.
        return "bfloat16" if major >= 8 else "float16"

    def _resolve_max_seq_len(self) -> int:
        value = int(self.option("max_seq_len") or 0)
        if value <= 0:
            return 0
        if value < _MIN_MAX_SEQ_LEN:
            self._warn_once(
                "ctx_floor",
                "[Fish] max_seq_len=%d is below the minimum; using %d.",
                value, _MIN_MAX_SEQ_LEN,
            )
            return _MIN_MAX_SEQ_LEN
        return value

    def _current_load_key(self) -> _LoadKey:
        precision = self._resolve_precision()
        codec_precision = (
            "float32" if self._choice("codec_precision") == "float32" else precision
        )
        quantization = str(getattr(self.config, "quantization", "none") or "none").lower()
        if quantization != "none":
            self._warn_once(
                "quantization",
                "[Fish] quantization=%s is ignored: fish-speech has no supported "
                "quantized path for S2 Pro.",
                quantization,
            )
        return _LoadKey(
            model_id=self.resolve_model_id(),
            checkpoint_override=os.environ.get(_CHECKPOINT_ENV_VAR, "").strip(),
            precision=precision,
            codec_precision=codec_precision,
            max_seq_len=self._resolve_max_seq_len(),
            compile=bool(getattr(self.config, "torch_compile", False)),
            trim_codec=bool(self.option("trim_codec_buffers")),
        )

    # ── Loading ──────────────────────────────────────────────────────────────

    def _ensure_initialised(self) -> None:
        with self._lock:
            key = self._current_load_key()
            if self._model is not None and self._loaded == key:
                return
            if self._model is not None:
                logger.info("[Fish] Settings changed on %s; reloading the model.", self._device)
                self._unload()
            self.bind_device()
            with _LOAD_LOCK:
                self._load(key)

    def _import_upstream(self) -> tuple[Any, Any]:
        """Imports the fish-speech inference modules.

        Returns
        -------
        tuple[Any, Any]
            ``(inference module, llama module)``.

        Raises
        ------
        RuntimeError
            If the package or one of its dependencies is missing.
        """
        repo = os.environ.get(_REPO_ENV_VAR, "").strip()
        if repo:
            if not os.path.isdir(os.path.join(repo, "fish_speech")):
                raise RuntimeError(
                    f"{_REPO_ENV_VAR}={repo!r} is not a fish-speech clone "
                    "(no fish_speech/ directory inside it)."
                )
            if repo not in sys.path:
                sys.path.insert(0, repo)
        try:
            inference = importlib.import_module(_UPSTREAM_INFERENCE_MODULE)
            llama = importlib.import_module(_UPSTREAM_LLAMA_MODULE)
        except ImportError as exc:
            raise RuntimeError(
                f"Fish Audio S2 Pro needs the fish-speech package ({exc}). Install it with: "
                f"{_INSTALL_COMMAND}  - or clone https://github.com/fishaudio/fish-speech "
                f"and set {_REPO_ENV_VAR} to the clone."
            ) from exc
        if not bool(self.option("verbose_upstream")):
            _silence_upstream(inference)
        return inference, llama

    def _checkpoint_dir(self, key: _LoadKey) -> Path:
        """Returns the local directory holding the weights for *key*."""
        if key.checkpoint_override:
            directory = Path(key.checkpoint_override)
            if not directory.is_dir():
                raise RuntimeError(
                    f"{_CHECKPOINT_ENV_VAR}={key.checkpoint_override!r} is not a directory."
                )
        else:
            try:
                from huggingface_hub import snapshot_download
            except ImportError as exc:
                raise RuntimeError(
                    "huggingface_hub is required to download Fish Audio S2 Pro: "
                    "pip install huggingface-hub"
                ) from exc
            try:
                directory = Path(snapshot_download(key.model_id))
            except Exception as exc:
                raise RuntimeError(
                    f"Could not download {key.model_id} from Hugging Face ({exc}). If the "
                    "repository is gated, accept its terms on the model page and set "
                    f"HF_TOKEN, or point {_CHECKPOINT_ENV_VAR} at a local copy."
                ) from exc
        marker = next((m for m in _QUANTIZED_PATH_MARKERS if m in str(directory)), None)
        if marker:
            raise RuntimeError(
                f"The checkpoint path {str(directory)!r} contains {marker!r}; fish-speech "
                "treats that as a legacy quantized checkpoint and fails to load S2 Pro. "
                "Rename or move the directory."
            )
        if not (directory / _CODEC_FILE_NAME).is_file():
            raise RuntimeError(f"{directory} has no {_CODEC_FILE_NAME}; the download is incomplete.")
        return directory

    def _estimate_vram_bytes(self, directory: Path, key: _LoadKey) -> int | None:
        """Estimates the VRAM held once the model is loaded, or None if unknown."""
        try:
            with open(directory / "config.json", encoding="utf-8") as fh:
                data = json.load(fh)
            text_cfg = data.get("text_config", data)
            n_layer = int(text_cfg["n_layer"])
            head_dim = int(text_cfg.get("head_dim") or text_cfg["dim"] // text_cfg["n_head"])
            kv_heads = int(text_cfg.get("n_local_heads", -1))
            if kv_heads <= 0:
                kv_heads = int(text_cfg["n_head"])
            native_len = int(text_cfg.get("max_seq_len", 2048))
        except Exception as exc:
            logger.debug("[Fish] Could not read %s/config.json: %s", directory, exc)
            return None
        context = key.max_seq_len or native_len
        itemsize = _PRECISION_ITEMSIZE[key.precision]
        weight_files = list(directory.glob("*.safetensors")) or list(directory.glob("model.pth"))
        weights = sum(f.stat().st_size for f in weight_files)
        weights = weights * itemsize // _CHECKPOINT_BYTES_PER_PARAM
        kv_cache = n_layer * 2 * kv_heads * context * head_dim * itemsize
        causal_mask = context * context
        codec = (directory / _CODEC_FILE_NAME).stat().st_size
        codec = codec * _PRECISION_ITEMSIZE[key.codec_precision] // _CODEC_CHECKPOINT_BYTES_PER_PARAM
        if not key.trim_codec:
            codec += _CODEC_MASK_BUFFER_BYTES
        return weights + kv_cache + causal_mask + codec

    def _free_vram_bytes(self) -> int | None:
        if not self._is_cuda():
            return None
        try:
            import torch

            free, _total = torch.cuda.mem_get_info(self._cuda_index())
            return int(free)
        except Exception as exc:
            logger.debug("[Fish] Could not query free VRAM on %s: %s", self._device, exc)
            return None

    def _vram_advice(self) -> str:
        return (
            "Free the GPU, lower the max_seq_len option (4096 is enough for audiobook "
            "chunks), keep codec_precision=model and trim_codec_buffers=on, or use a "
            "smaller engine such as Qwen3-TTS on this device."
        )

    def _check_vram(self, directory: Path, key: _LoadKey) -> None:
        """Raises before loading when the device plainly cannot hold the model."""
        free = self._free_vram_bytes()
        if free is None:
            return
        if self._oom_memo is not None:
            memo_key, memo_free, memo_message = self._oom_memo
            if memo_key == key and free < memo_free + _OOM_RETRY_GAIN_BYTES:
                raise RuntimeError(memo_message)
            self._oom_memo = None
        needed = self._estimate_vram_bytes(directory, key)
        if needed is None:
            return
        needed += _VRAM_WORKING_MARGIN_BYTES
        if free < needed:
            raise RuntimeError(
                f"Fish Audio S2 Pro needs about {needed / _BYTES_PER_GIB:.1f} GiB of free "
                f"VRAM with the current settings, but {self._device} has "
                f"{free / _BYTES_PER_GIB:.1f} GiB free. {self._vram_advice()}"
            )

    @staticmethod
    def _available_host_ram_bytes() -> int | None:
        """Free host RAM from ``/proc/meminfo``, or None when it cannot be read."""
        try:
            with open("/proc/meminfo", encoding="ascii") as fh:
                for line in fh:
                    if line.startswith("MemAvailable:"):
                        return int(line.split()[1]) * 1024
        except (OSError, ValueError, IndexError):
            return None
        return None

    def _check_host_ram(self, directory: Path, low_ram: bool) -> None:
        """Raises before loading when the host plainly lacks the RAM to load.

        Running out of host RAM kills the process outright, so this is the
        only place the failure can be reported.
        """
        available = self._available_host_ram_bytes()
        if available is None:
            return
        weight_files = list(directory.glob("*.safetensors")) or list(directory.glob("model.pth"))
        weights = sum(f.stat().st_size for f in weight_files)
        # Without low-RAM loading the float32 random init (4 bytes per
        # parameter) is resident next to the 2-byte checkpoint.
        language_model = weights if low_ram else weights * 3
        needed = max(language_model, _CODEC_BUILD_RAM_BYTES)
        if available < needed:
            hint = "" if low_ram else " Enable the low_ram_load option to need far less."
            raise RuntimeError(
                f"Loading Fish Audio S2 Pro needs about {needed / _BYTES_PER_GIB:.0f} GiB "
                f"of free host RAM, but only {available / _BYTES_PER_GIB:.1f} GiB is "
                f"available.{hint} Close other programs or unload other models first."
            )

    @staticmethod
    def _checkpoint_keys(directory: Path, llama: Any) -> set[str] | None:
        """Weight names the checkpoint provides, as the model will see them.

        Returns None when they cannot be listed without reading the weights
        (legacy ``model.pth`` checkpoints).
        """
        index_file = directory / "model.safetensors.index.json"
        single_file = directory / "model.safetensors"
        try:
            if index_file.is_file():
                with open(index_file, encoding="utf-8") as fh:
                    names = list(json.load(fh)["weight_map"])
            elif single_file.is_file():
                from safetensors import safe_open

                with safe_open(str(single_file), framework="pt") as handle:
                    names = list(handle.keys())
            else:
                return None
            mapping: Any = OrderedDict.fromkeys(names)
            remap = getattr(llama, "_remap_fish_qwen3_omni_keys", None)
            if remap is not None:
                mapping = remap(mapping)
            return set(mapping)
        except Exception as exc:
            logger.debug("[Fish] Could not list checkpoint weights in %s: %s", directory, exc)
            return None

    @staticmethod
    def _weights_not_in_checkpoint(model: Any, checkpoint_keys: set[str]) -> list[str]:
        """Model weights the checkpoint does not supply."""
        missing = []
        for name in model.state_dict().keys():
            if name in checkpoint_keys:
                continue
            # Upstream's load hook builds wqkv from separate wq / wk / wv weights.
            if name.endswith("wqkv.weight") and (
                name[: -len("wqkv.weight")] + "wq.weight" in checkpoint_keys
            ):
                continue
            missing.append(name)
        return sorted(missing)

    def _load_language_model(
        self, llama: Any, directory: Path, key: _LoadKey, checkpoint_keys: set[str] | None
    ) -> Any:
        """Builds the Dual-AR transformer and loads its weights on the CPU."""
        max_length = key.max_seq_len or None
        if checkpoint_keys is not None:
            with _skip_random_init(llama):
                model = llama.DualARTransformer.from_pretrained(
                    directory, load_weights=True, max_length=max_length
                )
            missing = self._weights_not_in_checkpoint(model, checkpoint_keys)
            if not missing:
                return model
            # Those weights would be uninitialised memory; do what upstream does.
            logger.warning(
                "[Fish] %d weight(s) are not in the checkpoint (%s); reloading with "
                "fish-speech's default initialisation.", len(missing), ", ".join(missing[:3]),
            )
            del model
            gc.collect()
        return llama.DualARTransformer.from_pretrained(
            directory, load_weights=True, max_length=max_length
        )

    @staticmethod
    def _trim_codec_buffers(codec: Any) -> int:
        """Replaces the codec's unused square attention masks; returns bytes freed.

        ``WindowLimitedTransformer.forward`` always builds its own mask, so the
        32768 x 32768 ``causal_mask`` registered by its base class is never read.
        """
        import torch

        freed = 0
        for module in codec.modules():
            if type(module).__name__ != "WindowLimitedTransformer":
                continue
            mask = getattr(module, "causal_mask", None)
            if isinstance(mask, torch.Tensor) and mask.numel() > 1:
                freed += mask.numel() * mask.element_size()
                module.causal_mask = torch.ones((1, 1), dtype=torch.bool)
        return freed

    def _load(self, key: _LoadKey) -> None:
        """Loads codec and language model onto this instance's device."""
        import torch

        inference, llama = self._import_upstream()
        directory = self._checkpoint_dir(key)
        self._check_vram(directory, key)
        checkpoint_keys = (
            self._checkpoint_keys(directory, llama) if bool(self.option("low_ram_load")) else None
        )
        self._check_host_ram(directory, low_ram=checkpoint_keys is not None)
        free_before = self._free_vram_bytes()

        logger.info(
            "[Fish] Loading %s on %s (%s, codec %s, context %s, compile=%s)...",
            key.model_id, self._device, key.precision, key.codec_precision,
            key.max_seq_len or "native", key.compile,
        )
        codec: Any = None
        model: Any = None
        try:
            # Codec first and on the CPU, so its mask buffers never reach the GPU.
            codec = inference.load_codec_model(
                directory / _CODEC_FILE_NAME, "cpu", torch.float32
            )
            if key.trim_codec:
                freed = self._trim_codec_buffers(codec)
                logger.debug("[Fish] Dropped %.1f GiB of codec masks.", freed / _BYTES_PER_GIB)
            # Upstream builds the codec under inference_mode, so its weights are
            # inference tensors: converting them outside that mode leaves them
            # unusable ("Inference tensors do not track version counter").
            with torch.inference_mode():
                codec = codec.to(device=self._device, dtype=getattr(torch, key.codec_precision))
            codec.eval()
            self._codec = codec
            gc.collect()

            # Same steps as upstream init_model(), plus the max_length override.
            model = self._load_language_model(llama, directory, key, checkpoint_keys)
            if getattr(model, "tokenizer", None) is None:
                raise RuntimeError(
                    f"fish-speech could not load the tokenizer in {directory}; the "
                    "checkpoint is incomplete or not an S2 Pro checkpoint."
                )
            model = model.to(device=self._device, dtype=getattr(torch, key.precision))
            model._cache_setup_done = False
            decode_one_token = inference.decode_one_token_ar
            if key.compile:
                cuda = torch.cuda.is_available()
                decode_one_token = torch.compile(
                    decode_one_token,
                    backend="inductor" if cuda else "aot_eager",
                    mode="default" if cuda else None,
                    fullgraph=True,
                    dynamic=True,
                )
            model = model.eval()
            # Allocate the KV cache now so a too-small GPU fails at load time.
            with torch.device(self._device):
                model.setup_caches(
                    max_batch_size=1,
                    max_seq_len=model.config.max_seq_len,
                    dtype=next(model.parameters()).dtype,
                )
            model._cache_setup_done = True
        except Exception as exc:
            codec = model = None  # release the partial load before emptying the cache
            self._unload()
            if self._is_out_of_memory(exc):
                message = (
                    f"Fish Audio S2 Pro ran out of memory while loading on {self._device}. "
                    f"{self._vram_advice()}"
                )
                if free_before is not None:
                    self._oom_memo = (key, free_before, message)
                raise RuntimeError(message) from exc
            raise

        self._model = model
        self._decode_one_token = decode_one_token
        self._inference = inference
        self._loaded = key
        self._context_len = int(model.config.max_seq_len)
        self._compile_warmed = False
        self._oom_memo = None
        logger.info("[Fish] Ready on %s.", self._device)

    def _unload(self) -> None:
        self._model = None
        self._codec = None
        self._decode_one_token = None
        self._inference = None
        self._loaded = None
        self._context_len = 0
        self._compile_warmed = False
        self._references.clear()
        self._preset_meta.clear()
        gc.collect()
        self._empty_cuda_cache()

    @staticmethod
    def _empty_cuda_cache() -> None:
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as exc:
            logger.debug("[Fish] empty_cache failed: %s", exc)

    @staticmethod
    def _is_out_of_memory(exc: BaseException) -> bool:
        try:
            import torch
        except Exception:
            return False
        oom_type = getattr(torch.cuda, "OutOfMemoryError", None)
        seen: BaseException | None = exc
        while seen is not None:
            if oom_type is not None and isinstance(seen, oom_type):
                return True
            seen = seen.__cause__
        return False

    # ── Reference clip ───────────────────────────────────────────────────────

    def _file_digest(self, path: str) -> str:
        """Content hash of a file, re-read only when its size or mtime changes."""
        stat = os.stat(path)
        cached = self._path_digests.get(path)
        if cached and cached[0] == stat.st_mtime_ns and cached[1] == stat.st_size:
            return cached[2]
        digest = hashlib.sha256()
        with open(path, "rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                digest.update(block)
        value = digest.hexdigest()
        self._path_digests[path] = (stat.st_mtime_ns, stat.st_size, value)
        return value

    def _transcript_for(self, *paths: str | None) -> str:
        """Transcript from the config or a sidecar file of any of *paths*."""
        text = ""
        for path in (*paths, None):
            text = self.reference_transcript(path)
            if text:
                break
        return _WHITESPACE_RE.sub(" ", _CONTROL_TOKEN_RE.sub(" ", text)).strip()

    def _remember(self, cache_key: tuple[str, str], prompt: _ReferencePrompt) -> _ReferencePrompt:
        self._references[cache_key] = prompt
        self._references.move_to_end(cache_key)
        while len(self._references) > _MAX_REFERENCE_CACHE:
            self._references.popitem(last=False)
        return prompt

    def _require_transcript(self, transcript: str) -> None:
        """Enforces the transcript rule shared by clips and presets."""
        if transcript:
            return
        if not bool(self.option("allow_missing_transcript")):
            raise RuntimeError(
                "Fish Audio S2 Pro needs the transcript of the reference clip. Set "
                "voice_transcript, put the text in a .txt file next to the clip, or "
                "enable the allow_missing_transcript option (lower voice similarity)."
            )
        self._warn_once(
            "no_transcript",
            "[Fish] Reference clip has no transcript; voice similarity will suffer.",
        )

    def _reference_prompt(self, voice_ref: str | bytes | None) -> _ReferencePrompt:
        """Returns the narrator prompt: the saved preset if set, else the clip."""
        preset = (getattr(self.config, "voice_preset", "") or "").strip()
        if preset:
            return self._preset_prompt(preset)[0]
        return self._clip_prompt(voice_ref)[0]

    def _clip_prompt(
        self, voice_ref: str | bytes | None, transcript: str | None = None
    ) -> tuple[_ReferencePrompt, str]:
        """Returns ``(prompt, content digest)`` for a reference clip, encoding it once.

        The cache key is the SHA-256 of the clip's content plus the transcript,
        so the same narrator reaches the codec exactly once per model load.
        """
        voice_file = (getattr(self.config, "voice_file", "") or "").strip() or None
        source_path: str | None = None
        if isinstance(voice_ref, (bytes, bytearray)) and voice_ref:
            if isinstance(voice_ref, bytearray):
                voice_ref = bytes(voice_ref)
            self._validate_voice_ref(voice_ref)
            digest = hashlib.sha256(voice_ref).hexdigest()
        else:
            source_path = self.resolve_voice_path(voice_ref)
            if not source_path:
                raise RuntimeError(
                    "Fish Audio S2 Pro has no built-in voices: set a reference clip "
                    "(voice_file) of 10-30 seconds together with its transcript."
                )
            self._validate_voice_ref(source_path)
            digest = self._file_digest(source_path)

        if transcript is None:
            transcript = self._transcript_for(source_path, voice_file)
        else:
            transcript = _WHITESPACE_RE.sub(" ", _CONTROL_TOKEN_RE.sub(" ", transcript)).strip()
        self._require_transcript(transcript)

        cache_key = (digest, transcript)
        cached = self._references.get(cache_key)
        if cached is not None:
            self._references.move_to_end(cache_key)
            return cached, digest
        if source_path is None:
            source_path = self.resolve_voice_path(voice_ref)
        codes = self._encode_reference(source_path)  # type: ignore[arg-type]
        prompt = _ReferencePrompt(codes=codes, transcript=transcript, frames=int(codes.shape[1]))
        return self._remember(cache_key, prompt), digest

    # ── Voice presets ────────────────────────────────────────────────────────

    @staticmethod
    def _read_preset_file(path: str) -> tuple[Any, dict[str, Any]]:
        """Reads codec tokens and metadata from a preset file.

        Accepts presets written by :meth:`save_voice_preset`, a bare token
        tensor saved with ``torch.save``, and the ``.npy`` token files
        fish-speech's own tools write. Nothing is unpickled: ``.pt`` files are
        read with ``weights_only=True`` and ``.npy`` with ``allow_pickle=False``.

        Raises
        ------
        ValueError
            If the file is missing or is not a codec-token preset.
        """
        import numpy as np
        import torch

        if not os.path.isfile(path):
            raise ValueError(f"Fish voice preset not found: {path}")
        if not path.lower().endswith((".npy", ".pt")):
            raise ValueError(f"Fish voice presets are .pt or .npy codec-token files; got {path}.")
        try:
            if path.lower().endswith(".npy"):
                payload: Any = torch.from_numpy(np.load(path, allow_pickle=False))
            else:
                payload = torch.load(path, map_location="cpu", weights_only=True)
        except Exception as exc:
            raise ValueError(f"{path} is not a readable Fish voice preset: {exc}") from exc

        meta: dict[str, Any] = {}
        codes = payload
        if isinstance(payload, dict):
            codes = payload.get("codes")
            meta = {
                key: value for key, value in payload.items()
                if isinstance(value, (str, int, float, bool))
            }
            if meta.get("format") != _PRESET_FORMAT:
                raise ValueError(f"{path} is not a Fish voice preset (format {meta.get('format')!r}).")
        if (
            not isinstance(codes, torch.Tensor)
            or codes.ndim != 2
            or codes.shape[1] == 0
            or codes.is_floating_point()
            or codes.is_complex()
        ):
            raise ValueError(f"{path} does not hold a (num_codebooks, frames) codec-token array.")
        return codes.to(dtype=torch.long).cpu(), meta

    def _check_preset_compatible(self, path: str, codes: Any, meta: dict[str, Any]) -> None:
        """Rejects tokens that do not fit the loaded model's codebooks."""
        model_id = self._loaded.model_id if self._loaded is not None else ""
        model_cfg = self._model.config
        num_codebooks = int(getattr(model_cfg, "num_codebooks", codes.shape[0]))
        if int(codes.shape[0]) != num_codebooks:
            raise ValueError(
                f"{path} has {int(codes.shape[0])} codebooks but {model_id} uses "
                f"{num_codebooks}; it was made for a different model."
            )
        codebook_size = int(getattr(model_cfg, "codebook_size", 0) or 0)
        if int(codes.min()) < 0 or (codebook_size and int(codes.max()) >= codebook_size):
            raise ValueError(f"{path} holds token ids outside the model's codebook range.")
        saved_model = str(meta.get("model_id") or "")
        if saved_model and saved_model != model_id:
            raise ValueError(f"{path} was saved with {saved_model}, not {model_id}.")

    @staticmethod
    def _preset_info(path: str, prompt: _ReferencePrompt, meta: dict[str, Any]) -> dict[str, Any]:
        """JSON-safe description of a preset."""
        return {
            "path": path,
            "format": str(meta.get("format") or "codec-tokens"),
            "format_version": int(meta.get("format_version") or 0),
            "model_id": str(meta.get("model_id") or ""),
            "num_codebooks": int(prompt.codes.shape[0]),
            "frames": prompt.frames,
            "seconds": round(prompt.frames / _CODEC_FRAME_RATE_HZ, 2),
            "sample_rate": int(meta.get("sample_rate") or _NATIVE_SAMPLE_RATE),
            "transcript": prompt.transcript,
            "created_at": str(meta.get("created_at") or ""),
            "source_sha256": str(meta.get("source_sha256") or ""),
        }

    def _preset_prompt(self, path: str) -> tuple[_ReferencePrompt, dict[str, Any]]:
        """Returns ``(prompt, info)`` for a preset file, reading it once.

        The transcript stored in the preset wins, because it is the text of
        the audio that was encoded; token-only files fall back to
        ``config.voice_transcript`` or a sidecar ``.txt``.
        """
        if not os.path.isfile(path):
            raise ValueError(f"Fish voice preset not found: {path}")
        fallback = self._transcript_for(path)
        cache_key = (f"preset:{self._file_digest(path)}", fallback)
        cached = self._references.get(cache_key)
        if cached is not None:
            self._references.move_to_end(cache_key)
            return cached, self._preset_info(path, cached, self._preset_meta.get(cache_key[0], {}))

        codes, meta = self._read_preset_file(path)
        self._check_preset_compatible(path, codes, meta)
        stored = _CONTROL_TOKEN_RE.sub(" ", str(meta.get("transcript") or ""))
        transcript = _WHITESPACE_RE.sub(" ", stored).strip() or fallback
        self._require_transcript(transcript)
        prompt = _ReferencePrompt(codes=codes, transcript=transcript, frames=int(codes.shape[1]))
        self._preset_meta[cache_key[0]] = meta
        self._remember(cache_key, prompt)
        logger.info(
            "[Fish] Voice preset %s loaded on %s (%d codec frames).",
            os.path.basename(path), self._device, prompt.frames,
        )
        return prompt, self._preset_info(path, prompt, meta)

    def save_voice_preset(
        self,
        path: str,
        voice_ref: str | bytes | None = None,
        *,
        transcript: str | None = None,
    ) -> dict[str, Any]:
        """Saves the encoded narrator reference so later runs skip the codec.

        The file holds the reference clip's codec tokens, its transcript and
        metadata, as a tensor plus plain values, so it loads with
        ``torch.load(weights_only=True)``. Set ``config.voice_preset`` to the
        file to use it; no reference clip is needed then.

        Parameters
        ----------
        path : str
            Destination file; must end in ``.pt``.
        voice_ref : str | bytes | None
            Reference clip (path or WAV bytes). Defaults to ``config.voice_file``.
        transcript : str | None
            Transcript of the clip; overrides ``config.voice_transcript`` and
            sidecar files.

        Returns
        -------
        dict[str, Any]
            JSON-safe description of the preset, including ``"path"``.

        Raises
        ------
        ValueError
            If *path* is not a ``.pt`` file, or the clip or its transcript is missing.
        """
        import time

        import torch

        if not str(path).lower().endswith(".pt"):
            raise ValueError(f"Fish voice presets are saved as .pt files; got {path}.")
        with self._lock:
            self._ensure_initialised()
            self.bind_device()
            try:
                prompt, digest = self._clip_prompt(voice_ref, transcript)
            except RuntimeError as exc:
                if self._is_out_of_memory(exc):
                    raise
                raise ValueError(str(exc)) from exc
            meta: dict[str, Any] = {
                "format": _PRESET_FORMAT,
                "format_version": _PRESET_FORMAT_VERSION,
                "model_id": self._loaded.model_id if self._loaded is not None else "",
                "sample_rate": int(self._codec.sample_rate),
                "transcript": prompt.transcript,
                "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "source_sha256": digest,
            }
            payload = dict(meta, codes=prompt.codes.detach().to("cpu", torch.long).clone())

        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, exist_ok=True)
        tmp_path = f"{path}.{os.getpid()}.tmp"
        torch.save(payload, tmp_path)
        os.replace(tmp_path, path)
        logger.info("[Fish] Voice preset saved to %s (%d codec frames).", path, prompt.frames)
        return self._preset_info(path, prompt, meta)

    def load_voice_preset(self, path: str) -> dict[str, Any]:
        """Loads a preset and checks that it fits the loaded model.

        Synthesis does this by itself when ``config.voice_preset`` is set;
        call it directly to validate a file the user picked.

        Returns
        -------
        dict[str, Any]
            JSON-safe description of the preset.

        Raises
        ------
        ValueError
            If the file is missing, unreadable, or made for another model.
        """
        with self._lock:
            self._ensure_initialised()
            self.bind_device()
            try:
                return self._preset_prompt(path)[1]
            except RuntimeError as exc:
                if self._is_out_of_memory(exc):
                    raise
                raise ValueError(str(exc)) from exc

    @staticmethod
    def _resample(audio: Any, source_rate: int, target_rate: int) -> Any:
        """Resamples a mono float32 ndarray; torchaudio first, librosa otherwise."""
        if source_rate == target_rate:
            return audio
        import numpy as np

        try:
            import torch
            from torchaudio.functional import resample

            return resample(torch.from_numpy(audio), source_rate, target_rate).numpy()
        except Exception as exc:
            logger.debug("[Fish] torchaudio resample unavailable (%s); using librosa.", exc)
        import librosa

        return np.asarray(
            librosa.resample(audio, orig_sr=source_rate, target_sr=target_rate),
            dtype=np.float32,
        )

    def _encode_reference(self, path: str) -> Any:
        """Encodes a reference clip to codec tokens (upstream ``encode_audio``).

        The clip is read with soundfile rather than ``torchaudio.load`` so the
        provider does not depend on torchaudio's decoder backend.
        """
        import numpy as np
        import soundfile as sf
        import torch

        try:
            audio, source_rate = sf.read(path, dtype="float32", always_2d=True)
        except Exception as exc:
            raise RuntimeError(f"Could not read the reference clip {path}: {exc}") from exc
        mono = np.ascontiguousarray(audio.mean(axis=1), dtype=np.float32)
        if mono.size == 0 or not np.isfinite(mono).all():
            raise RuntimeError(f"The reference clip {path} is empty or corrupted.")

        codec = self._codec
        target_rate = int(codec.sample_rate)
        mono = np.ascontiguousarray(
            self._resample(mono, int(source_rate), target_rate), dtype=np.float32
        )
        seconds = mono.size / float(target_rate)
        if seconds > _RECOMMENDED_MAX_REFERENCE_SECONDS:
            logger.warning(
                "[Fish] Reference clip is %.0f s; 10-30 s is recommended. Every chunk "
                "re-reads the whole clip, so long references are slow.", seconds,
            )
        elif seconds < _RECOMMENDED_MIN_REFERENCE_SECONDS:
            logger.warning(
                "[Fish] Reference clip is only %.1f s; 10-30 s is recommended.", seconds
            )

        codec_dtype = next(codec.parameters()).dtype
        waveform = torch.from_numpy(mono)
        audios = waveform[None, None].to(device=self._device, dtype=codec_dtype)
        audio_lengths = torch.tensor(
            [waveform.shape[0]], device=self._device, dtype=torch.long
        )
        with torch.inference_mode():
            indices, feature_lengths = codec.encode(audios, audio_lengths)
            codes = indices[0, :, : int(feature_lengths[0])]
        codes = codes.detach().to(device="cpu", dtype=torch.long)
        if codes.ndim != 2 or codes.shape[1] == 0:
            raise RuntimeError(f"The codec produced no tokens for the reference clip {path}.")
        logger.info(
            "[Fish] Encoded %.1f s reference clip to %d codec frames on %s.",
            seconds, int(codes.shape[1]), self._device,
        )
        return codes

    # ── Text ─────────────────────────────────────────────────────────────────

    def _prepare_text(self, text: str) -> str:
        """Applies tag handling, the style tag and the speaker tag to one chunk."""
        cleaned = _CONTROL_TOKEN_RE.sub(" ", text or "")
        mode = self._choice("inline_tags")
        if mode == "strip":
            cleaned = _INLINE_TAG_RE.sub(" ", cleaned)
        elif mode == "speak":
            cleaned = _INLINE_TAG_RE.sub(r"(\1)", cleaned)
        cleaned = _WHITESPACE_RE.sub(" ", cleaned).strip()
        if not cleaned:
            raise RuntimeError("Fish Audio S2 Pro was given an empty text chunk.")

        instruct = (getattr(self.config, "tts_instruct", "") or "").strip()
        instruct = _WHITESPACE_RE.sub(" ", _CONTROL_TOKEN_RE.sub(" ", instruct)).strip()
        if instruct and bool(self.option("instruct_as_tag")):
            # A prompt that already contains [tags] is used verbatim; plain
            # prose becomes one free-form tag.
            tag = instruct if _INLINE_TAG_RE.search(instruct) else f"[{instruct.strip('[] ')}]"
            cleaned = f"{tag} {cleaned}"
        if bool(self.option("speaker_tag")):
            cleaned = f"{_SPEAKER_TAG}{cleaned}"
        return cleaned

    def _token_budget(self, prepared_text: str, prompt: _ReferencePrompt) -> int:
        """Upper bound on generated codec frames for one chunk.

        The budget grows with the UTF-8 length of the text (about 4 frames per
        byte, three times normal speech), is capped by ``max_new_tokens`` and
        by the room left in the model's context after the prompt.
        """
        text_bytes = len(prepared_text.encode("utf-8"))
        per_byte = float(self.option("tokens_per_byte") or 4.0)
        budget = _TOKEN_BUDGET_BASE + math.ceil(text_bytes * max(per_byte, 0.5))
        cap = int(self.option("max_new_tokens") or 0)
        if cap > 0:
            budget = min(budget, cap)

        # A BPE token covers at least one byte, so bytes bound the token count.
        prompt_bound = (
            prompt.frames
            + len(prompt.transcript.encode("utf-8"))
            + text_bytes
            + _PROMPT_OVERHEAD_TOKENS
        )
        context = self._context_len
        if context > 0:
            room = context - prompt_bound
            # Upstream refuses prompts longer than max_seq_len - 2048 tokens,
            # and every reference frame is one prompt token.
            if room < _MIN_TOKEN_BUDGET or prompt.frames > context - _UPSTREAM_PROMPT_RESERVE:
                raise RuntimeError(
                    f"The reference clip ({prompt.frames} codec frames) plus this chunk "
                    f"do not fit the model context of {context} tokens. Use a shorter "
                    "reference clip (10-30 s) or raise the max_seq_len option."
                )
            budget = min(budget, room)
        return max(budget, _MIN_TOKEN_BUDGET)

    def _sampling_kwargs(self) -> dict[str, Any]:
        """Maps the common sampling fields onto ``generate_long`` arguments."""
        cfg = self.config
        temperature = float(getattr(cfg, "temperature", 0.8))
        top_p = float(getattr(cfg, "top_p", 0.8))
        top_k = int(getattr(cfg, "top_k", 30))
        return {
            "temperature": min(max(temperature, _MIN_TEMPERATURE), _MAX_TEMPERATURE),
            "top_p": min(max(top_p, _MIN_TOP_P), 1.0),
            "top_k": top_k if top_k > 0 else _TOP_K_DISABLED,
            # Accepted by generate_long but not applied at the pinned commit,
            # which uses repetition-aware sampling instead.
            "repetition_penalty": float(getattr(cfg, "repetition_penalty", 1.1)),
        }

    # ── Generation ───────────────────────────────────────────────────────────

    def _seed(self, attempt: int) -> None:
        """Seeds the first attempt from the config and shifts the seed on retries."""
        if attempt == 0:
            self.seed_everything()
            return
        seed = getattr(self.config, "seed", -1)
        if seed is None or int(seed) < 0:
            return
        import torch

        torch.manual_seed(int(seed) + attempt)
        if self._is_cuda() and torch.cuda.is_available():
            torch.cuda.manual_seed(int(seed) + attempt)

    @staticmethod
    def _hit_token_limit(frames: int, budget: int) -> bool:
        """True when generation stopped at the budget instead of at end-of-speech."""
        return frames >= budget - 1

    def _run_upstream(
        self,
        prepared_text: str,
        prompt: _ReferencePrompt,
        budget: int,
        sampling: dict[str, Any],
    ) -> Any:
        """Runs ``generate_long`` once and returns the generated codes, or None."""
        import torch

        key = self._loaded
        assert key is not None
        collected: list[Any] = []
        first_compiled_call = key.compile and not self._compile_warmed
        guard = _LOAD_LOCK if first_compiled_call else contextlib.nullcontext()
        try:
            with guard, _SdpBackendGuard.preserve():
                for response in self._inference.generate_long(
                    model=self._model,
                    device=self._device,
                    decode_one_token=self._decode_one_token,
                    text=prepared_text,
                    num_samples=1,
                    max_new_tokens=budget,
                    compile=key.compile,
                    iterative_prompt=True,
                    # Only groups <|speaker:N|> turns; one chunk is one turn.
                    chunk_length=max(len(prepared_text.encode("utf-8")), 1),
                    prompt_text=[prompt.transcript],
                    prompt_tokens=[prompt.codes],
                    **sampling,
                ):
                    if getattr(response, "action", None) == "sample" and response.codes is not None:
                        collected.append(response.codes)
        except ValueError as exc:
            if "too long" in str(exc).lower():
                raise RuntimeError(
                    f"Fish Audio S2 Pro rejected the prompt ({exc}). Use a shorter "
                    "reference clip (10-30 s) or raise the max_seq_len option."
                ) from exc
            raise
        self._compile_warmed = True
        if not collected:
            return None
        return torch.cat(collected, dim=1)

    def _decode(self, codes: Any) -> Any:
        """Decodes codec tokens to a CPU float32 waveform tensor."""
        import torch

        audio = self._inference.decode_to_audio(codes.to(self._device), self._codec)
        audio = audio.detach().float().cpu()
        if bool(torch.isfinite(audio).all()):
            return audio
        if next(self._codec.parameters()).dtype == torch.float32:
            return audio
        logger.warning(
            "[Fish] The codec produced NaN/inf in reduced precision on %s; switching "
            "the codec to float32.", self._device,
        )
        with torch.inference_mode():  # see _load: the weights are inference tensors
            self._codec = self._codec.float()
        audio = self._inference.decode_to_audio(codes.to(self._device), self._codec)
        return audio.detach().float().cpu()

    def _generate(self, text: str, prompt: _ReferencePrompt) -> tuple[Any, int]:
        """Generates one chunk and returns ``(waveform, sample_rate)``."""
        prepared = self._prepare_text(text)
        budget = self._token_budget(prepared, prompt)
        sampling = self._sampling_kwargs()
        attempts = 1 + max(0, int(self.option("runaway_retries") or 0))

        codes: Any = None
        problem = ""
        for attempt in range(attempts):
            self._seed(attempt)
            codes = self._run_upstream(prepared, prompt, budget, sampling)
            frames = int(codes.shape[1]) if codes is not None else 0
            if frames == 0:
                problem = "produced no audio tokens"
            elif self._hit_token_limit(frames, budget):
                problem = (
                    f"did not finish within {budget} codec frames "
                    f"({budget / _CODEC_FRAME_RATE_HZ:.0f} s of audio)"
                )
            else:
                problem = ""
                break
            logger.warning(
                "[Fish] Attempt %d/%d on %s %s for a %d-character chunk.",
                attempt + 1, attempts, self._device, problem, len(text),
            )
        if problem:
            raise RuntimeError(
                f"Fish Audio S2 Pro {problem} on {self._device} after {attempts} "
                "attempt(s). Raise the max_new_tokens or tokens_per_byte option if the "
                "chunk is legitimately long, or lower max_len."
            )

        audio = self._decode(codes)
        return audio, int(self._codec.sample_rate)
