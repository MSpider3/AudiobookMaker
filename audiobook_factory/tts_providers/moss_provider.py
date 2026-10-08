"""
audiobook_factory/tts_providers/moss_provider.py
=================================================
MOSS-TTS (OpenMOSS / MOSI.AI) zero-shot voice-cloning provider.

MOSS-TTS is an autoregressive discrete-token TTS family: a Qwen3 backbone
predicts RVQ codes of the MOSS-Audio-Tokenizer at 12.5 frames per second, and
the codec turns the codes back into audio. Three architectures are supported
behind this one class, because they share one processor / ``generate`` shape:

=================================================  =========  ======  =============
Model id                                           Arch       Params  Audio
=================================================  =========  ======  =============
``OpenMOSS-Team/MOSS-TTS-Local-Transformer``       local      1.7B    24 kHz mono
``OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5``  local_v15  4B      48 kHz stereo
``OpenMOSS-Team/MOSS-TTS-v1.5``                    delay      8B      24 kHz mono
``OpenMOSS-Team/MOSS-TTS``                         delay      8B      24 kHz mono
=================================================  =========  ======  =============

"1.7B" is the backbone of the default model; its checkpoint holds 3.06B
parameters once the 33 embedding tables, 33 output heads and the local
transformer are counted.

Everything is loaded through Hugging Face *remote code* (``trust_remote_code``),
which upstream requires: none of these classes ship inside ``transformers``.
The provider therefore only ever loads the repositories listed in
``INFO.models`` plus the two codec repositories, each from a commit pinned in
``_MODEL_SPECS`` / ``_CODEC_SPECS`` and fetched with ``snapshot_download``.
(``revision=`` cannot be passed to the processor: upstream forwards every
keyword to the *codec* repository and to ``ProcessorMixin.__init__`` as well.)

transformers must be 5.0.x for the default model. On 4.57 upstream's
processor cannot be constructed; on 5.1.0, 5.3.0 and 5.19.0 the 1.7B Local
model's ``generate`` fails (it overrides ``GenerationMixin._sample``). The 4B
and 8B models also ran on 5.19.0. This was checked by running the pinned
remote code with tiny random weights on CPU; nothing here has been run with
the real weights or on a GPU.

Written against the Hub commits pinned below and GitHub
``OpenMOSS/MOSS-TTS`` commit ``934d682``. The upstream calls used are:

* ``<Processor>.from_pretrained(model_dir, trust_remote_code=True,
  codec_path=codec_dir)`` on the snapshot's own processor class (plus
  ``codec_weight_dtype`` / ``codec_compute_dtype`` /
  ``codec_attention_implementation`` on v1.5 Local)
* ``AutoConfig.from_pretrained(model_dir, trust_remote_code=True)`` and
  ``AutoModel.from_pretrained(model_dir, config=..., trust_remote_code=True,
  attn_implementation=..., dtype=...)``
* ``processor.encode_audios_from_wav([wav], sampling_rate, n_vq)`` which
  returns one ``LongTensor`` of codes shaped ``(frames, n_vq)`` per clip
* ``processor.build_user_message(text=, reference=[codes], tokens=,
  language=, instruction=)`` and
  ``processor.build_assistant_message(audio_codes_list=[codes])``
* ``processor(conversations, mode="generation" | "continuation")`` which
  returns left-padded ``input_ids`` ``(batch, seq, 1 + n_vq)`` and
  ``attention_mask``
* ``model.generate(input_ids=, attention_mask=, max_new_tokens=,
  text_temperature=, text_top_p=, text_top_k=, audio_temperature=,
  audio_top_p=, audio_top_k=, audio_repetition_penalty=)`` (plus
  ``n_vq_for_inference`` on Local 1.7B and ``do_sample`` on v1.5 Local; the
  Delay model accepts no other keyword)
* ``processor.decode(outputs)`` which returns one message per item whose
  ``audio_codes_list[0]`` is the waveform: ``(samples,)`` on the 24 kHz
  models and ``(2, samples)`` on v1.5 Local

How the narrator voice is resolved (first match wins)
-----------------------------------------------------
1. ``config.voice_preset`` - a ``.pt`` file written by
   :meth:`MossTTSProvider.save_voice_preset` holds the reference clip already
   tokenized, so no codec encoder pass is needed. Any other file is treated
   as a reference clip; an incompatible ``.pt`` falls back to the clip.
2. A reference clip (the ``voice_ref`` argument, else ``config.voice_file``).

The codes of the reference are cached per clip (key: SHA-256 of the file
content, transcript, model id and reference trim), so the codec encoder runs
once per book, not once per chunk. MOSS-TTS has no preset speakers and its
reference-free mode invents a new voice per call, so a reference is required.

Memory layout on the GPU
------------------------
The codec is as large as the default model (1.78B parameters, shipped in
fp32: 3.3 GiB encoder + 3.3 GiB decoder). The encoder is only needed to
tokenize the reference clip, so by default it stays in CPU RAM and visits the
GPU for that one call (``offload_codec_encoder``). The default model then
holds about 9 GiB of a 16 GB T4: 5.7 GiB of fp16 weights plus the fp32 codec
decoder.

Shared settings
---------------
``temperature``, ``top_p``, ``top_k`` and ``repetition_penalty`` drive the
audio layers (see the ``sampling_preset`` option for how stock defaults are
replaced by upstream's per-model values), ``speed`` drives token-level
duration control, ``seed``, ``language``, ``voice_file``, ``voice_preset``
and ``voice_transcript`` are honoured.

Not mapped, because MOSS-TTS has no equivalent: ``tts_timbre`` (no preset
speakers), ``nfe_step`` and ``quantization`` (bitsandbytes int8 is not
verified with these remote-code models, so the setting is ignored with a
warning). ``tts_instruct`` is not mapped either: upstream documents the
prompt's instruction field only for MOSS-VoiceGenerator; the experimental
``instruction`` option passes text through for anyone who wants to try.
"""
from __future__ import annotations

import contextlib
import gc
import hashlib
import importlib.util
import logging
import math
import os
import re
import threading
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterator

from audiobook_factory.tts_providers.base_tts_provider import (
    BaseTTSProvider,
    ProviderInfo,
    ProviderOption,
)

if TYPE_CHECKING:
    from audiobook_factory.pipeline import AudiobookConfig

logger = logging.getLogger(__name__)

# ── Upstream constants ────────────────────────────────────────────────────────

_FRAMES_PER_SECOND: float = 12.5
# Audio frames per character, from upstream's own duration slider
# (clis/moss_tts_app.py: EN_TOKENS_PER_CHAR / ZH_TOKENS_PER_CHAR).
_FRAMES_PER_CHAR: float = 0.8673376262755219
_FRAMES_PER_WIDE_CHAR: float = 3.098411951313033
_MIN_TRANSFORMERS: tuple[int, int] = (5, 0)
# The 1.7B Local model overrides GenerationMixin._sample; its remote code fails
# inside generate() on transformers 5.1.0, 5.3.0 and 5.19.0 (checked).
_LOCAL_MAX_TRANSFORMERS: tuple[int, int] = (5, 0)
_PINNED_TRANSFORMERS: str = "5.0.0"

_INSTALL_COMMAND: str = "pip install -r requirements/tts-moss.txt"
_INSTALL_COMMAND_INLINE: str = (
    f'pip install "transformers=={_PINNED_TRANSFORMERS}" torchaudio safetensors soundfile'
)

# ── Provider tuning ───────────────────────────────────────────────────────────

_MIN_FRAME_CAP: int = 75                 # never cap a chunk below 6 seconds
_BUDGET_SLACK_FRAMES: int = 2            # lets a take that ends exactly on its cap finish
_RUNAWAY_TOLERANCE_FRAMES: float = 1.5
_WORKING_VRAM_GIB: float = 1.0           # KV cache + codec activations for a small batch
_REFERENCE_CACHE_SIZE: int = 4
_MIN_SPEED: float = 0.5
_MAX_SPEED: float = 2.0
_BYTES_PER_GIB: float = float(1 << 30)
_PRESET_FORMAT: str = "abm-moss-voice-v1"
_SNAPSHOT_IGNORE: tuple[str, ...] = ("*.md", "*.png", "*.jpg", "*.jpeg", "*.wav", "images/*", "demo/*")

_PAUSE_TAG_RE: re.Pattern[str] = re.compile(r"\[\s*pause\s+(\d+(?:\.\d+)?)\s*s\s*\]", re.IGNORECASE)
# CJK ideographs, kana and hangul: scripts where one character is a syllable or more.
_WIDE_CHAR_RE: re.Pattern[str] = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af]")

_LANGUAGES_V1: tuple[str, ...] = (
    "Chinese", "English", "German", "Spanish", "French", "Japanese", "Italian",
    "Hebrew", "Korean", "Russian", "Persian (Farsi)", "Arabic", "Polish",
    "Portuguese", "Czech", "Danish", "Swedish", "Hungarian", "Greek", "Turkish",
)
_LANGUAGES_V15: tuple[str, ...] = _LANGUAGES_V1 + (
    "Cantonese", "Dutch", "Finnish", "Hindi", "Macedonian", "Malay", "Romanian",
    "Swahili", "Tagalog", "Thai", "Vietnamese",
)
_LANGUAGE_CODES: dict[str, str] = {
    "zh": "Chinese", "yue": "Cantonese", "en": "English", "ar": "Arabic",
    "cs": "Czech", "da": "Danish", "nl": "Dutch", "fi": "Finnish", "fr": "French",
    "de": "German", "el": "Greek", "he": "Hebrew", "hi": "Hindi", "hu": "Hungarian",
    "it": "Italian", "ja": "Japanese", "ko": "Korean", "mk": "Macedonian",
    "ms": "Malay", "fa": "Persian (Farsi)", "pl": "Polish", "pt": "Portuguese",
    "ro": "Romanian", "ru": "Russian", "es": "Spanish", "sw": "Swahili",
    "sv": "Swedish", "tl": "Tagalog", "th": "Thai", "tr": "Turkish", "vi": "Vietnamese",
}
_LANGUAGE_ALIASES: dict[str, str] = {
    **{name.lower(): name for name in _LANGUAGES_V15},
    **_LANGUAGE_CODES,
    "persian": "Persian (Farsi)", "farsi": "Persian (Farsi)",
    "mandarin": "Chinese", "filipino": "Tagalog",
}
_NO_LANGUAGE: frozenset[str] = frozenset({"", "auto", "none", "multilingual", "mixed"})


@dataclass(frozen=True)
class _CodecSpec:
    """One MOSS-Audio-Tokenizer release.

    Attributes
    ----------
    repo, revision : str
        Hub repository and the commit this provider was written against.
    encoder_gib, decoder_gib : float
        fp32 size of the encoder and of the decoder plus quantizer, in GiB.
    """

    repo: str
    revision: str
    encoder_gib: float
    decoder_gib: float


@dataclass(frozen=True)
class _ModelSpec:
    """Static facts about one allowed MOSS-TTS checkpoint.

    Attributes
    ----------
    architecture : str
        ``"local"`` (1.7B), ``"local_v15"`` (4B) or ``"delay"`` (8B). Decides
        which keywords ``generate`` accepts and how many steps a clip costs
        beyond its audio frames.
    revision : str
        Pinned commit of the model repository (weights and remote code).
    codec : str
        Key into ``_CODEC_SPECS``.
    sample_rate, n_vq : int
        Output sample rate and RVQ depth of the checkpoint.
    weights_gib : float
        Size of the weights in fp16 / bf16, in GiB.
    audio_temperature, audio_top_p, audio_top_k, audio_repetition_penalty
        Upstream's recommended decoding values for this checkpoint.
    text_temperature : float
        Upstream's default temperature of the text (frame / stop) layer.
    native_pause_tags : bool
        The checkpoint was trained on ``[pause X.Ys]`` markers.
    language_tags : bool
        The checkpoint was trained with the prompt's language field.
    languages : tuple[str, ...]
        Languages the checkpoint was trained on.
    """

    architecture: str
    revision: str
    codec: str
    sample_rate: int
    n_vq: int
    weights_gib: float
    audio_temperature: float
    audio_top_p: float
    audio_top_k: int
    audio_repetition_penalty: float
    text_temperature: float
    native_pause_tags: bool
    language_tags: bool
    languages: tuple[str, ...]


_CODEC_SPECS: dict[str, _CodecSpec] = {
    "v1": _CodecSpec(
        repo="OpenMOSS-Team/MOSS-Audio-Tokenizer",
        revision="3cd226ba2947efa357ef453bcad111b6eafba782",
        encoder_gib=3.30, decoder_gib=3.31,
    ),
    "v2": _CodecSpec(
        repo="OpenMOSS-Team/MOSS-Audio-Tokenizer-v2",
        revision="f6e20e543b33d2c252a7ef71bdf8aa71e5ff9169",
        encoder_gib=3.95, decoder_gib=3.96,
    ),
}

_DEFAULT_MODEL: str = "OpenMOSS-Team/MOSS-TTS-Local-Transformer"

_MODEL_SPECS: dict[str, _ModelSpec] = {
    _DEFAULT_MODEL: _ModelSpec(
        architecture="local",
        revision="12aa734e4f11a7b3fdf4eb0ad2aa2029675ffc2e",
        codec="v1", sample_rate=24000, n_vq=32, weights_gib=5.70,
        audio_temperature=1.0, audio_top_p=0.95, audio_top_k=50,
        audio_repetition_penalty=1.1, text_temperature=1.5,
        native_pause_tags=False, language_tags=False, languages=_LANGUAGES_V1,
    ),
    "OpenMOSS-Team/MOSS-TTS-Local-Transformer-v1.5": _ModelSpec(
        architecture="local_v15",
        revision="be7766a6735b98bd793f7c79fb720b4d0f5d13b8",
        codec="v2", sample_rate=48000, n_vq=12, weights_gib=8.48,
        audio_temperature=1.7, audio_top_p=0.8, audio_top_k=25,
        audio_repetition_penalty=1.0, text_temperature=1.0,
        native_pause_tags=True, language_tags=True, languages=_LANGUAGES_V15,
    ),
    "OpenMOSS-Team/MOSS-TTS-v1.5": _ModelSpec(
        architecture="delay",
        revision="cdd3b911b1585e3f2dbc7775ef10f9926f58850a",
        codec="v1", sample_rate=24000, n_vq=32, weights_gib=15.81,
        audio_temperature=1.7, audio_top_p=0.8, audio_top_k=25,
        audio_repetition_penalty=1.0, text_temperature=1.5,
        native_pause_tags=True, language_tags=True, languages=_LANGUAGES_V15,
    ),
    "OpenMOSS-Team/MOSS-TTS": _ModelSpec(
        architecture="delay",
        revision="b6b0229853ff63c68fa6aeceb380d8c016f55daf",
        codec="v1", sample_rate=24000, n_vq=32, weights_gib=15.81,
        audio_temperature=1.7, audio_top_p=0.8, audio_top_k=25,
        audio_repetition_penalty=1.0, text_temperature=1.5,
        native_pause_tags=False, language_tags=False, languages=_LANGUAGES_V1,
    ),
}


@dataclass
class _Piece:
    """One unit of work inside an input text: speech to generate, or a pause."""

    item: int                     # index of the input text this piece belongs to
    text: str = ""                # empty for an emulated pause
    pause_seconds: float = 0.0
    frame_cap: int = 0            # audio frames this piece may not exceed
    tokens: int | None = None     # duration-control target, None when off
    audio: Any = None             # generated waveform (torch tensor, CPU)


@dataclass
class _Settings:
    """Everything read from ``config`` for one synthesis call."""

    clone_mode: str               # "reference" | "continuation" | "reference+continuation"
    duration_control: bool
    speed: float
    duration_scale: float
    max_duration_factor: float
    hard_frame_cap: int
    runaway_retries: int
    pause_mode: str               # "native" | "emulate" | "strip"
    language: str | None
    instruction: str | None
    trim_eos_frame: bool
    max_batch_size: int
    generation: dict[str, Any] = field(default_factory=dict)


def _split_pauses(text: str) -> list[str | float]:
    """Splits *text* at ``[pause X.Ys]`` markers into text and seconds."""
    parts: list[str | float] = []
    position = 0
    for match in _PAUSE_TAG_RE.finditer(text):
        before = text[position:match.start()].strip()
        if before:
            parts.append(before)
        parts.append(float(match.group(1)))
        position = match.end()
    tail = text[position:].strip()
    if tail:
        parts.append(tail)
    return parts


def _estimate_frames(text: str) -> float:
    """Expected audio frames for *text* at natural pace (12.5 frames = 1 s).

    Uses upstream's own per-character rates: one for scripts where a character
    is a whole syllable (CJK, kana, hangul) and one for everything else.
    """
    wide = len(_WIDE_CHAR_RE.findall(text))
    narrow = max(len(text) - wide, 0)
    return wide * _FRAMES_PER_WIDE_CHAR + narrow * _FRAMES_PER_CHAR


def _join_prefix(prefix: str, text: str) -> str:
    """Prepends the reference transcript to *text* for continuation modes."""
    prefix = prefix.strip()
    if not prefix:
        return text
    if _WIDE_CHAR_RE.match(prefix[-1]) or (text and _WIDE_CHAR_RE.match(text[0])):
        return prefix + text
    return f"{prefix} {text}"


class MossTTSProvider(BaseTTSProvider):
    """MOSS-TTS narrator: one model, processor and codec bound to one device.

    Parameters
    ----------
    config : AudiobookConfig
        Run settings. Read again on every call, never cached.
    device : str | None
        Torch device string, e.g. ``"cuda:0"``. Defaults to ``config.device``.
    dtype_override : str | None
        ``"float16"``, ``"bfloat16"`` or ``"float32"``; overrides the
        ``model_dtype`` option.
    """

    INFO = ProviderInfo(
        name="moss",
        display_name="MOSS-TTS",
        description=(
            "Autoregressive voice cloning from OpenMOSS / MOSI.AI with token-level "
            "duration control, [pause 1.5s] markers and inline Pinyin or /IPA/ "
            "pronunciation. The default 1.7B Local-Transformer model fits a 16 GB T4; "
            "the 4B (48 kHz) and 8B models need bigger cards."
        ),
        license="Apache-2.0",
        commercial_use=True,
        homepage="https://github.com/OpenMOSS/MOSS-TTS",
        default_model=_DEFAULT_MODEL,
        models=tuple(_MODEL_SPECS),
        native_sample_rate=_MODEL_SPECS[_DEFAULT_MODEL].sample_rate,
        min_vram_gb=11.0,
        languages=_LANGUAGES_V15,
        supports_voice_clone=True,
        transcript="optional",
        supports_instruct=False,
        supports_batch=True,
        supports_speed=True,
        supports_seed=True,
        supports_voice_preset=True,
        preset_voices=(),
        # Upstream's decoding values for the default (1.7B Local) model. The
        # 4B and 8B models have their own; see the sampling_preset option.
        recommended_settings={
            "temperature": _MODEL_SPECS[_DEFAULT_MODEL].audio_temperature,
            "top_p": _MODEL_SPECS[_DEFAULT_MODEL].audio_top_p,
            "top_k": _MODEL_SPECS[_DEFAULT_MODEL].audio_top_k,
            "repetition_penalty": _MODEL_SPECS[_DEFAULT_MODEL].audio_repetition_penalty,
        },
        options=(
            ProviderOption(
                key="clone_mode", label="Cloning mode", kind="choice", default="reference",
                choices=("reference", "continuation", "reference+continuation"),
                help=(
                    "'reference' clones the timbre from the clip alone. The continuation "
                    "modes also make the model continue the clip itself, which copies its "
                    "delivery more closely but needs the clip's exact transcript and "
                    "switches duration control (and so native speed) off."
                ),
            ),
            ProviderOption(
                key="duration_control", label="Duration control", kind="choice", default="auto",
                choices=("auto", "on", "off"),
                help=(
                    "Tells the model how many audio tokens each chunk should take "
                    "(12.5 per second, estimated from its length and the speed setting). "
                    "'auto' only does so when speed is not 1.0; 'off' ignores speed."
                ),
            ),
            ProviderOption(
                key="duration_scale", label="Duration estimate scale", kind="float", default=1.0,
                minimum=0.5, maximum=2.0, step=0.05,
                help="Multiplies the estimated length of every chunk; raise it for a slow narrator.",
            ),
            ProviderOption(
                key="max_duration_factor", label="Runaway limit (x expected length)", kind="float",
                default=2.5, minimum=1.2, maximum=6.0, step=0.1,
                help=(
                    "A chunk may run this many times longer than estimated before it is "
                    "stopped and retried as a runaway generation."
                ),
            ),
            ProviderOption(
                key="max_new_tokens", label="Hard token limit per chunk", kind="int", default=0,
                minimum=0, maximum=8192, step=16,
                help="Absolute ceiling on audio tokens per chunk (12.5 per second); 0 = automatic.",
            ),
            ProviderOption(
                key="runaway_retries", label="Runaway retries", kind="int", default=2,
                minimum=0, maximum=5, step=1,
                help="How often a chunk that hit its limit or came back empty is regenerated before failing.",
            ),
            ProviderOption(
                key="pause_tags", label="[pause X.Ys] markers", kind="choice", default="auto",
                choices=("auto", "emulate", "strip"),
                help=(
                    "'auto' passes markers to v1.5 models, which were trained on them, and "
                    "inserts real silence for the others; 'emulate' always inserts silence; "
                    "'strip' removes the markers."
                ),
            ),
            ProviderOption(
                key="language_tag", label="Language tag", kind="choice", default="auto",
                choices=("auto", "always", "never"),
                help=(
                    "Sends the book language in the prompt. 'auto' does so for v1.5 "
                    "models, where upstream recommends it."
                ),
            ),
            ProviderOption(
                key="sampling_preset", label="Audio sampling", kind="choice", default="auto",
                choices=("auto", "recommended", "config"),
                help=(
                    "Where the audio layers' temperature, top-p, top-k and repetition "
                    "penalty come from. 'auto' uses the global settings, except that a "
                    "setting still at a stock default is replaced by upstream's value for "
                    "the loaded model (the 4B and 8B models want 1.7 / 0.8 / 25 / 1.0). "
                    "'recommended' always uses upstream's values; 'config' always uses the "
                    "global settings as they are."
                ),
            ),
            ProviderOption(
                key="text_temperature", label="Text-layer temperature", kind="float", default=-1.0,
                minimum=-1.0, maximum=3.0, step=0.05,
                help=(
                    "Temperature of the layer that decides when the audio ends. -1 = the "
                    "model's default (1.5, or 1.0 on v1.5 Local); 0 = greedy."
                ),
            ),
            ProviderOption(
                key="text_top_p", label="Text-layer top-p", kind="float", default=1.0,
                minimum=0.05, maximum=1.0, step=0.05,
                help="Nucleus cutoff of the text layer.",
            ),
            ProviderOption(
                key="text_top_k", label="Text-layer top-k", kind="int", default=50,
                minimum=1, maximum=200, step=1,
                help="Top-k cutoff of the text layer.",
            ),
            ProviderOption(
                key="rvq_depth", label="RVQ depth", kind="int", default=0,
                minimum=0, maximum=32, step=1,
                help=(
                    "Codebook layers generated per frame (1.7B Local model only, 1-32). "
                    "Fewer is faster and lower fidelity; 0 = all."
                ),
            ),
            ProviderOption(
                key="instruction", label="Instruction (experimental)", kind="str", default="",
                help=(
                    "Text for the prompt's instruction field. Upstream documents it only "
                    "for MOSS-VoiceGenerator; MOSS-TTS checkpoints may ignore it."
                ),
            ),
            ProviderOption(
                key="max_reference_seconds", label="Reference clip limit (s)", kind="float",
                default=30.0, minimum=0.0, maximum=120.0, step=1.0,
                help=(
                    "Only this much of the reference clip is used; every second costs 12.5 "
                    "prompt tokens on every chunk. 0 = whole clip."
                ),
            ),
            ProviderOption(
                key="max_batch_size", label="Chunks per forward pass", kind="int", default=8,
                minimum=1, maximum=32, step=1,
                help="Larger batches are split into groups of this size.",
            ),
            ProviderOption(
                key="offload_codec_encoder", label="Keep codec encoder off the GPU", kind="bool",
                default=True,
                help=(
                    "The codec's encoder (3.3-4 GiB) is only needed to tokenize the "
                    "reference clip; keep it in CPU RAM between uses."
                ),
            ),
            ProviderOption(
                key="trim_eos_frame", label="Trim end-of-audio frame", kind="bool", default=True,
                help=(
                    "The 1.7B Local model emits one extra 80 ms frame of untrained codes "
                    "on the step where it stops; cut it off."
                ),
            ),
            ProviderOption(
                key="model_dtype", label="Model precision", kind="choice", default="auto",
                choices=("auto", "float16", "bfloat16", "float32"),
                help=(
                    "'auto' is bfloat16 (what upstream documents) on GPUs with native "
                    "support and float16 on older ones such as the T4."
                ),
            ),
            ProviderOption(
                key="codec_dtype", label="Codec precision (v1.5 Local)", kind="choice",
                default="auto", choices=("auto", "float32", "bfloat16"),
                help=(
                    "MOSS-Audio-Tokenizer-v2 only: bfloat16 halves its memory. 'auto' picks "
                    "it on GPUs with native bfloat16. The 24 kHz codec always runs in float32."
                ),
            ),
            ProviderOption(
                key="attn_implementation", label="Attention backend", kind="choice", default="auto",
                choices=("auto", "sdpa", "flash_attention_2", "eager"),
                help="'auto' uses FlashAttention 2 when installed and supported, else SDPA.",
            ),
            ProviderOption(
                key="model_revision", label="Model revision", kind="str", default="",
                help=(
                    "Hub revision of the model repository. Empty = the commit this "
                    "provider was tested against; 'main' follows upstream (codec too)."
                ),
            ),
        ),
        pip_requirements=(
            f"transformers=={_PINNED_TRANSFORMERS}", "torchaudio", "safetensors", "soundfile",
        ),
        install_notes=(
            "Needs transformers 5.0.0, which upstream pins. On 4.57 the remote-code processor "
            "cannot be constructed, so MOSS-TTS and qwen-tts (transformers==4.57.3) cannot "
            "share an environment; on 5.1 and newer the default model's generate() fails, so "
            "it cannot share one with omnivoice (transformers>=5.3) either. The 4B and 8B "
            "models also run on newer 5.x. torchaudio must match the installed torch; no FFmpeg or "
            "torchcodec is needed because the reference clip is read with soundfile "
            "(WAV/FLAC/OGG). All repositories are public (Apache-2.0, no HF token or "
            "gating) and are loaded with trust_remote_code from pinned commits.\n"
            "Free VRAM, computed from the checkpoints (weights in 16-bit plus the fp32 "
            "codec decoder plus about 1 GiB of working memory):\n"
            "- MOSS-TTS-Local-Transformer (default, 24 kHz): about 10 GiB, 13.2 GB download "
            "(6.1 model + 7.1 codec). Fits a 16 GB T4 and 12 GB cards.\n"
            "- MOSS-TTS-Local-Transformer-v1.5 (48 kHz stereo, downmixed): about 13.5 GiB "
            "with the fp32 codec, 11.5 GiB with a bfloat16 codec; 17.6 GB download. Needs "
            "24 GB to be comfortable; a T4 is too tight to batch.\n"
            "- MOSS-TTS-v1.5 and MOSS-TTS (8B): about 20 GiB, 24.1 GB download. 24 GB cards "
            "only.\n"
            "The codec encoder additionally occupies 3.3-4 GiB of CPU RAM per GPU. "
            "MOSS-TTS-Nano and MOSS-TTSD use different remote code and are not loaded by "
            "this provider."
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
        self._processor: Any = None
        self._codec_encoder: Any = None          # codec encoder parked in CPU RAM
        self._loaded_model_id: str = ""
        self._loaded_key: tuple[Any, ...] = ()
        self._sample_rate: int = self.INFO.native_sample_rate
        self._reference_cache: OrderedDict[tuple[Any, ...], Any] = OrderedDict()
        self._digest_memo: dict[str, tuple[int, int, str]] = {}
        self._rejected_presets: dict[tuple[Any, ...], str] = {}
        self._warned: set[str] = set()

    # ── Identity ─────────────────────────────────────────────────────────────

    @property
    def device(self) -> str:
        """The torch device string this provider is bound to."""
        return self._device

    @property
    def sample_rate(self) -> int:
        """Sample rate of the audio this instance returns.

        24000 for the default model; 48000 once a v1.5 Local model is loaded
        or selected in ``config.tts_model_name``.
        """
        if self._model is not None:
            return self._sample_rate
        return _MODEL_SPECS[self.resolve_model_id()].sample_rate

    @classmethod
    def create_for_device(
        cls,
        device: str,
        config: "AudiobookConfig",
        dtype_override: str | None = None,
    ) -> "MossTTSProvider":
        """Constructs an instance pinned to *device*."""
        return cls(config, device=device, dtype_override=dtype_override)

    def _warn_once(self, key: str, message: str, *args: Any) -> None:
        if key not in self._warned:
            self._warned.add(key)
            logger.warning(message, *args)

    # ── Loading ──────────────────────────────────────────────────────────────

    def _import_dependencies(self) -> tuple[Any, Any, Any]:
        """Imports torch, transformers and huggingface_hub, or explains the fix.

        Raises
        ------
        RuntimeError
            If a dependency is missing or transformers is older than 5.0.
        """
        try:
            import torch
            import transformers
            import huggingface_hub
        except ImportError as exc:
            raise RuntimeError(
                f"MOSS-TTS dependencies are not installed ({exc}). "
                f"Run: {_INSTALL_COMMAND}  (or: {_INSTALL_COMMAND_INLINE})"
            ) from exc

        version = str(getattr(transformers, "__version__", "0"))
        if self._version_tuple(version) < _MIN_TRANSFORMERS:
            raise RuntimeError(
                f"MOSS-TTS needs transformers >= {_MIN_TRANSFORMERS[0]}.{_MIN_TRANSFORMERS[1]} "
                f"(found {version}): its remote-code processor does not import on older "
                f"releases. Run: {_INSTALL_COMMAND}  (or: {_INSTALL_COMMAND_INLINE}). "
                "This replaces the transformers 4.57 that qwen-tts needs, so use a "
                "separate environment for Qwen3-TTS."
            )
        try:
            import torchaudio  # noqa: F401  (imported by upstream's processor module)
        except Exception as exc:
            raise RuntimeError(
                f"MOSS-TTS needs a torchaudio build matching torch {torch.__version__} "
                f"(its remote-code processor imports it; import failed: {exc}). "
                f"Run: {_INSTALL_COMMAND}  (or: {_INSTALL_COMMAND_INLINE})"
            ) from exc
        return torch, transformers, huggingface_hub

    @staticmethod
    def _version_tuple(version: str) -> tuple[int, int]:
        """Returns ``(major, minor)`` of a version string such as ``"5.0.0"``."""
        numbers = [int(part) for part in re.findall(r"\d+", version)[:2]]
        return (numbers + [0, 0])[0], (numbers + [0, 0])[1]

    def _check_transformers(self, transformers: Any, model_id: str) -> None:
        """Rejects transformers releases the selected model's remote code breaks on.

        Raises
        ------
        RuntimeError
            If the 1.7B Local model is selected and transformers is newer than 5.0.
        """
        version = str(getattr(transformers, "__version__", "0"))
        if (
            _MODEL_SPECS[model_id].architecture == "local"
            and self._version_tuple(version) > _LOCAL_MAX_TRANSFORMERS
        ):
            raise RuntimeError(
                f"MOSS-TTS model '{model_id}' only runs on transformers "
                f"{_LOCAL_MAX_TRANSFORMERS[0]}.{_LOCAL_MAX_TRANSFORMERS[1]}.x (found {version}): its "
                f"remote code fails inside generate() on newer releases. "
                f'Run: pip install "transformers=={_PINNED_TRANSFORMERS}"  (or select a v1.5 '
                "model or the 8B model, whose remote code also runs on newer 5.x releases)."
            )

    def _cuda_index(self) -> int | None:
        """Index of the bound CUDA device, or None on CPU / MPS."""
        if not self._device.startswith("cuda"):
            return None
        return int(self._device.split(":")[1]) if ":" in self._device else 0

    def _native_bf16(self, torch: Any) -> bool:
        """True on GPUs with hardware bfloat16 (compute capability 8.0+)."""
        index = self._cuda_index()
        if index is None:
            return False
        try:
            return int(torch.cuda.get_device_capability(index)[0]) >= 8
        except Exception as exc:
            logger.debug("Could not read the compute capability of %s: %s", self._device, exc)
            return False

    def _dtype_name(self, torch: Any) -> str:
        """Resolves the model precision: override, then option, then device."""
        requested = (self._dtype_override or str(self.option("model_dtype", "auto"))).strip().lower()
        aliases = {
            "fp16": "float16", "half": "float16", "bf16": "bfloat16",
            "fp32": "float32", "float": "float32",
        }
        requested = aliases.get(requested, requested)
        if requested in ("float16", "bfloat16", "float32"):
            if self._cuda_index() is None and requested != "float32":
                self._warn_once(
                    "cpu-dtype", "MOSS-TTS on %s runs in float32; ignoring dtype %s.",
                    self._device, requested,
                )
                return "float32"
            return requested
        if requested not in ("", "auto"):
            self._warn_once("dtype", "Unknown MOSS-TTS dtype '%s'; using the default.", requested)
        if self._cuda_index() is None:
            return "float32"
        return "bfloat16" if self._native_bf16(torch) else "float16"

    def _attn_implementation(self, torch: Any, dtype_name: str) -> str:
        """Picks the attention backend the way upstream's examples do."""
        requested = str(self.option("attn_implementation", "auto")).strip().lower()
        if requested in ("sdpa", "flash_attention_2", "eager"):
            return requested
        if self._cuda_index() is None:
            return "eager"
        if (
            dtype_name in ("float16", "bfloat16")
            and self._native_bf16(torch)
            and importlib.util.find_spec("flash_attn") is not None
        ):
            return "flash_attention_2"
        return "sdpa"

    def _codec_dtype_name(self, torch: Any, spec: _ModelSpec) -> str:
        """Weight dtype of the codec: only MOSS-Audio-Tokenizer-v2 has a 16-bit mode."""
        if spec.codec != "v2":
            return "float32"
        requested = str(self.option("codec_dtype", "auto")).strip().lower()
        if requested in ("float32", "bfloat16"):
            return requested
        return "bfloat16" if self._native_bf16(torch) else "float32"

    def _revisions(self, model_id: str) -> tuple[str, str]:
        """Returns the (model, codec) revisions to download."""
        spec = _MODEL_SPECS[model_id]
        override = str(self.option("model_revision", "") or "").strip()
        codec_revision = _CODEC_SPECS[spec.codec].revision
        if not override:
            return spec.revision, codec_revision
        return override, ("main" if override == "main" else codec_revision)

    def _load_key(self, model_id: str) -> tuple[Any, ...]:
        """The raw settings a load depends on; cheap enough to compare per call."""
        return (
            model_id,
            self._dtype_override,
            str(self.option("model_dtype", "auto")),
            str(self.option("codec_dtype", "auto")),
            str(self.option("attn_implementation", "auto")),
            bool(self.option("offload_codec_encoder", True)),
            str(self.option("model_revision", "") or ""),
        )

    def _signature(self, torch: Any, model_id: str) -> tuple[Any, ...]:
        """Resolves the load settings: precision, attention backend, revisions."""
        spec = _MODEL_SPECS[model_id]
        dtype_name = self._dtype_name(torch)
        return (
            model_id,
            dtype_name,
            self._codec_dtype_name(torch, spec),
            self._attn_implementation(torch, dtype_name),
            bool(self.option("offload_codec_encoder", True)),
            self._revisions(model_id),
        )

    def _required_vram_gib(self, spec: _ModelSpec, signature: tuple[Any, ...]) -> float:
        """Free VRAM needed to load *spec* and synthesize a small batch."""
        _, dtype_name, codec_dtype, _, offload, _ = signature
        codec = _CODEC_SPECS[spec.codec]
        codec_scale = 0.5 if codec_dtype == "bfloat16" else 1.0
        weights = spec.weights_gib * (2.0 if dtype_name == "float32" else 1.0)
        codec_gib = codec.decoder_gib + (0.0 if offload else codec.encoder_gib)
        return weights + codec_gib * codec_scale + _WORKING_VRAM_GIB

    def _check_vram(self, torch: Any, model_id: str, signature: tuple[Any, ...]) -> None:
        """Refuses to start a load that cannot fit, instead of running out of memory.

        Raises
        ------
        RuntimeError
            If the device's free VRAM is below what the model needs.
        """
        index = self._cuda_index()
        if index is None:
            return
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(index)
            reclaimable = max(
                0, torch.cuda.memory_reserved(index) - torch.cuda.memory_allocated(index)
            )
        except Exception as exc:
            logger.debug("Could not read free VRAM of %s: %s", self._device, exc)
            return
        free_gib = (free_bytes + reclaimable) / _BYTES_PER_GIB
        needed_gib = self._required_vram_gib(_MODEL_SPECS[model_id], signature)
        if free_gib >= needed_gib:
            return
        default_gib = self._required_vram_gib(
            _MODEL_SPECS[_DEFAULT_MODEL], (_DEFAULT_MODEL, "float16", "float32", "sdpa", True, ())
        )
        hint = (
            f"Select '{_DEFAULT_MODEL}' (about {default_gib:.0f} GiB) or a GPU with more memory."
            if model_id != _DEFAULT_MODEL
            else "Free the GPU (another model may be loaded on it) or use a GPU with at least 12 GB."
        )
        raise RuntimeError(
            f"MOSS-TTS model '{model_id}' needs about {needed_gib:.1f} GiB of free VRAM "
            f"({signature[1]} weights plus the audio codec), but {self._device} has "
            f"{free_gib:.1f} GiB free of {total_bytes / _BYTES_PER_GIB:.1f} GiB. {hint}"
        )

    def _ensure_initialised(self) -> None:
        """Loads the model selected in ``config``; reloads when it changed.

        Raises
        ------
        RuntimeError
            If dependencies are missing, the model cannot fit the device, or
            loading fails.
        """
        with self._lock:
            model_id = self.resolve_model_id()
            load_key = self._load_key(model_id)
            if self._model is not None and load_key == self._loaded_key:
                return
            torch, transformers, hub = self._import_dependencies()
            self._check_transformers(transformers, model_id)
            signature = self._signature(torch, model_id)
            if self._model is not None:
                logger.info(
                    "[MOSS-TTS] Settings changed (%s -> %s); reloading on %s.",
                    self._loaded_model_id, model_id, self._device,
                )
                self._unload()
            self.bind_device()
            self._check_vram(torch, model_id, signature)
            try:
                self._load(torch, transformers, hub, model_id, signature)
                self._loaded_key = load_key
            except Exception as exc:
                self._unload()
                if isinstance(exc, getattr(torch.cuda, "OutOfMemoryError", ())):
                    raise RuntimeError(
                        f"MOSS-TTS model '{model_id}' ran out of memory while loading on "
                        f"{self._device}. Select '{_DEFAULT_MODEL}' or free the GPU."
                    ) from exc
                raise RuntimeError(
                    f"MOSS-TTS failed to load '{model_id}' on {self._device}: {exc}"
                ) from exc

    def _load(
        self, torch: Any, transformers: Any, hub: Any, model_id: str, signature: tuple[Any, ...]
    ) -> None:
        """Downloads the pinned snapshots and builds processor, codec and model."""
        spec = _MODEL_SPECS[model_id]
        codec_spec = _CODEC_SPECS[spec.codec]
        _, dtype_name, codec_dtype, attn, offload, (model_revision, codec_revision) = signature
        logger.info(
            "[MOSS-TTS] Loading %s@%s on %s (%s, attention=%s, codec %s).",
            model_id, model_revision[:12], self._device, dtype_name, attn, codec_dtype,
        )
        if self._quantization_requested():
            self._warn_once(
                "int8",
                "MOSS-TTS ignores quantization='%s': bitsandbytes is not verified with its "
                "remote-code models. Loading %s weights instead.",
                getattr(self.config, "quantization", ""), dtype_name,
            )
        self._configure_backends(torch)

        # Local snapshot directories pin both repositories; nothing below can
        # reach a repository other than these two. Order matters: transformers
        # copies only the first level of a local module's relative imports, so
        # the processor (whose module pulls in the text normalizer on v1.5)
        # must be loaded before the model class that imports the processor.
        model_dir = hub.snapshot_download(
            repo_id=model_id, revision=model_revision, ignore_patterns=list(_SNAPSHOT_IGNORE)
        )
        codec_dir = hub.snapshot_download(
            repo_id=codec_spec.repo, revision=codec_revision, ignore_patterns=list(_SNAPSHOT_IGNORE)
        )

        processor_kwargs: dict[str, Any] = {"trust_remote_code": True, "codec_path": codec_dir}
        if spec.architecture == "local_v15":
            half_codec = codec_dtype == "bfloat16"
            processor_kwargs["codec_weight_dtype"] = "bf16" if half_codec else "fp32"
            processor_kwargs["codec_compute_dtype"] = "bf16" if half_codec else "fp32"
            processor_kwargs["codec_attention_implementation"] = (
                "flash_attention_2" if attn == "flash_attention_2" and half_codec else "sdpa"
            )
        processor = self._processor_class(model_dir).from_pretrained(model_dir, **processor_kwargs)
        codec = processor.audio_tokenizer
        if codec is None:
            raise RuntimeError("the MOSS-TTS processor came back without an audio tokenizer")
        codec.eval()
        if offload and getattr(codec, "encoder", None) is not None:
            # Park the encoder before the codec moves, so it never needs VRAM
            # at load time. An empty ModuleList keeps the codec's own
            # "device of my first parameter" lookup pointing at the decoder.
            self._codec_encoder = codec.encoder
            codec.encoder = torch.nn.ModuleList()
        processor.audio_tokenizer = codec.to(self._device)
        self._processor = processor

        config = transformers.AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
        # v1.5 Local keeps its own public copies of the attention backend and
        # defaults them to flash_attention_2, which a T4 cannot run.
        for attribute in ("attn_implementation", "local_transformer_attn_implementation"):
            if hasattr(config, attribute):
                setattr(config, attribute, attn)
        model = transformers.AutoModel.from_pretrained(
            model_dir,
            config=config,
            trust_remote_code=True,
            attn_implementation=attn,
            dtype=getattr(torch, dtype_name),
        )
        model = model.to(self._device)
        model.eval()

        self._model = model
        self._loaded_model_id = model_id
        self._sample_rate = int(
            getattr(getattr(processor, "model_config", None), "sampling_rate", 0) or spec.sample_rate
        )
        logger.info("[MOSS-TTS] %s ready on %s (%d Hz).", model_id, self._device, self._sample_rate)

    @staticmethod
    def _processor_class(model_dir: str) -> Any:
        """Returns the snapshot's own processor class.

        ``AutoProcessor.from_pretrained`` is not used: newer transformers
        releases add a ``revision`` keyword on the way, and upstream's
        ``from_pretrained`` forwards every keyword it does not know to both the
        codec and ``ProcessorMixin.__init__``, which rejects it.
        """
        import json
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        with open(os.path.join(model_dir, "processor_config.json"), encoding="utf-8") as fh:
            class_reference = json.load(fh)["auto_map"]["AutoProcessor"]
        return get_class_from_dynamic_module(class_reference, model_dir)

    def _quantization_requested(self) -> bool:
        value = str(getattr(self.config, "quantization", "none") or "none").strip().lower()
        return value not in ("", "none", "off", "false")

    def _configure_backends(self, torch: Any) -> None:
        """Applies the SDPA backend switches every upstream example starts with."""
        if self._cuda_index() is None:
            return
        try:
            torch.backends.cuda.enable_cudnn_sdp(False)   # "broken cuDNN SDPA backend"
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
            torch.backends.cuda.enable_math_sdp(True)
        except Exception as exc:
            logger.debug("Could not configure SDPA backends: %s", exc)

    def _unload(self) -> None:
        """Drops model, processor and codec without touching CUDA."""
        processor = self._processor
        if processor is not None and getattr(processor, "audio_tokenizer", None) is not None:
            processor.audio_tokenizer = None
        self._model = None
        self._processor = None
        self._codec_encoder = None
        self._loaded_model_id = ""
        self._loaded_key = ()
        self._reference_cache.clear()
        self._rejected_presets.clear()

    def _empty_cache(self) -> None:
        gc.collect()
        try:
            import torch
            if self._cuda_index() is not None and torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as exc:
            logger.debug("Could not empty the CUDA cache on %s: %s", self._device, exc)

    def cleanup(self) -> None:
        """Releases the model, processor and codec and frees their VRAM."""
        with self._lock:
            self._unload()
            self._digest_memo.clear()
            self._empty_cache()

    # ── Reference voice ──────────────────────────────────────────────────────

    def _file_digest(self, path: str) -> str:
        """SHA-256 of a file's content, re-read only when the file changed."""
        stat = os.stat(path)
        memo = self._digest_memo.get(path)
        if memo is not None and memo[0] == stat.st_size and memo[1] == stat.st_mtime_ns:
            return memo[2]
        digest = hashlib.sha256()
        with open(path, "rb") as fh:
            for block in iter(lambda: fh.read(1 << 20), b""):
                digest.update(block)
        self._digest_memo[path] = (stat.st_size, stat.st_mtime_ns, digest.hexdigest())
        return digest.hexdigest()

    def _read_clip(self, torch: Any, path: str, max_seconds: float) -> tuple[Any, int]:
        """Reads a reference clip as a ``(channels, samples)`` float32 tensor.

        soundfile is used instead of upstream's ``torchaudio.load`` so that no
        torchcodec / FFmpeg install is needed.
        """
        import numpy as np
        import soundfile as sf

        try:
            data, sample_rate = sf.read(path, dtype="float32", always_2d=True)
        except Exception as exc:
            raise RuntimeError(
                f"MOSS-TTS could not read the reference clip '{path}': {exc}. "
                "Use a WAV or FLAC file."
            ) from exc
        if max_seconds > 0:
            data = data[: max(1, int(max_seconds * sample_rate))]
        if data.shape[0] == 0 or not np.isfinite(data).all():
            raise RuntimeError(f"MOSS-TTS reference clip '{path}' is empty or corrupted.")
        return torch.from_numpy(np.ascontiguousarray(data.T)), int(sample_rate)

    @contextlib.contextmanager
    def _codec_encoder_on(self, device: str) -> Iterator[None]:
        """Temporarily puts the parked codec encoder back, on *device*."""
        codec = self._processor.audio_tokenizer
        parked = self._codec_encoder
        if parked is None:
            yield
            return
        placeholder = codec.encoder
        try:
            codec.encoder = parked.to(device)
            yield
        finally:
            self._codec_encoder = parked.to("cpu")
            codec.encoder = placeholder

    def _tokenize_clip(self, torch: Any, wav: Any, sample_rate: int) -> Any:
        """Runs the codec encoder on one clip; returns codes ``(frames, n_vq)``."""
        processor = self._processor

        def encode() -> Any:
            codes = processor.encode_audios_from_wav([wav], sample_rate, None)
            if not codes or codes[0].ndim != 2 or codes[0].shape[0] == 0:
                raise RuntimeError("the MOSS audio tokenizer returned no codes for the reference clip")
            return codes[0].detach().to("cpu")

        if self._codec_encoder is None:
            return encode()
        try:
            with self._codec_encoder_on(self._device):
                return encode()
        except getattr(torch.cuda, "OutOfMemoryError", ()):
            logger.warning(
                "[MOSS-TTS] No VRAM on %s for the codec encoder; tokenizing the reference "
                "clip on the CPU (slow, but done once per clip).", self._device,
            )
        finally:
            self._empty_cache()
        codec = processor.audio_tokenizer
        codec.to("cpu")
        try:
            with self._codec_encoder_on("cpu"):
                return encode()
        finally:
            codec.to(self._device)

    def _read_preset(self, torch: Any, path: str, spec: _ModelSpec) -> dict[str, Any]:
        """Reads and validates a preset written by :meth:`save_voice_preset`.

        The file is opened with ``weights_only=True``: it holds one tensor
        and a few strings and numbers, and nothing else is ever unpickled.

        Raises
        ------
        ValueError
            If the file is not a preset for the loaded model's codec.
        """
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except Exception as exc:
            raise ValueError(f"'{path}' is not a MOSS-TTS voice preset: {exc}") from exc
        if not isinstance(payload, dict) or payload.get("format") != _PRESET_FORMAT:
            raise ValueError(f"'{path}' is not a MOSS-TTS voice preset.")
        codes = payload.get("codes")
        codec_repo = _CODEC_SPECS[spec.codec].repo
        if payload.get("codec_repo") != codec_repo:
            raise ValueError(
                f"Voice preset '{path}' was tokenized with {payload.get('codec_repo')}, but "
                f"{self._loaded_model_id} uses {codec_repo}. Save it again with this model."
            )
        if not hasattr(codes, "ndim") or codes.ndim != 2 or codes.shape[0] == 0 or int(codes.shape[1]) != spec.n_vq:
            raise ValueError(
                f"Voice preset '{path}' does not hold {spec.n_vq}-codebook codes for "
                f"{self._loaded_model_id}."
            )
        payload["codes"] = codes.to(torch.long)
        payload["transcript"] = str(payload.get("transcript", "") or "")
        return payload

    def _describe_preset(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        """JSON-safe summary of a preset."""
        codes = payload["codes"]
        return {
            "path": path,
            "provider": self.INFO.name,
            "format": _PRESET_FORMAT,
            "model": str(payload.get("model", "") or ""),
            "codec_repo": str(payload.get("codec_repo", "") or ""),
            "frames": int(codes.shape[0]),
            "codebooks": int(codes.shape[1]),
            "seconds": round(int(codes.shape[0]) / _FRAMES_PER_SECOND, 3),
            "sampling_rate": int(payload.get("sampling_rate", 0) or 0),
            "transcript": str(payload.get("transcript", "") or ""),
        }

    def _preset_key(self, path: str, transcript: str) -> tuple[Any, ...]:
        return (self._file_digest(path), transcript, self._loaded_model_id, "preset")

    def _reference(
        self, torch: Any, voice_ref: str | bytes | None, spec: _ModelSpec, allow_preset: bool = True
    ) -> tuple[Any, str]:
        """Returns ``(codes, transcript)`` of the narrator reference, cached.

        With *allow_preset* False ``config.voice_preset`` is ignored and the
        clip itself is tokenized.

        Raises
        ------
        RuntimeError
            If there is no usable reference clip.
        """
        max_seconds = max(0.0, float(self.option("max_reference_seconds", 30.0)))
        configured_transcript = (getattr(self.config, "voice_transcript", "") or "").strip()
        preset_path = (getattr(self.config, "voice_preset", "") or "").strip() if allow_preset else ""
        clip_path: str | None = None
        preset_problem = ""

        if preset_path and os.path.isfile(preset_path):
            key = self._preset_key(preset_path, configured_transcript)
            cached = self._reference_cache.get(key)
            if cached is not None:
                self._reference_cache.move_to_end(key)
                return cached
            if key in self._rejected_presets:
                preset_problem = self._rejected_presets[key]
            elif preset_path.lower().endswith(".pt"):
                try:
                    payload = self._read_preset(torch, preset_path, spec)
                except ValueError as exc:
                    preset_problem = self._rejected_presets[key] = str(exc)
                    logger.warning("[MOSS-TTS] %s Falling back to the reference clip.", exc)
                else:
                    entry = (payload["codes"], configured_transcript or payload["transcript"])
                    self._remember(key, entry)
                    return entry
            else:
                clip_path = preset_path
        elif preset_path:
            preset_problem = f"Voice preset '{preset_path}' does not exist."
            self._warn_once(f"preset:{preset_path}", "[MOSS-TTS] %s", preset_problem)

        if clip_path is None:
            if isinstance(voice_ref, (bytes, bytearray, str)) and voice_ref:
                self._validate_voice_ref(bytes(voice_ref) if isinstance(voice_ref, bytearray) else voice_ref)
            clip_path = self.resolve_voice_path(voice_ref)
        if not clip_path or not os.path.isfile(clip_path):
            raise RuntimeError(
                "MOSS-TTS needs a reference clip of the narrator: it has no preset voices, "
                "and without a clip every chunk would be spoken by a different random voice. "
                "Set a voice file." + (f" ({preset_problem})" if preset_problem else "")
            )

        transcript = self.reference_transcript(clip_path)
        key = (self._file_digest(clip_path), transcript, self._loaded_model_id, max_seconds)
        cached = self._reference_cache.get(key)
        if cached is not None:
            self._reference_cache.move_to_end(key)
            return cached
        wav, sample_rate = self._read_clip(torch, clip_path, max_seconds)
        codes = self._tokenize_clip(torch, wav, sample_rate)
        logger.info(
            "[MOSS-TTS] Tokenized reference clip on %s: %.1f s -> %d frames x %d codebooks.",
            self._device, wav.shape[-1] / float(sample_rate), codes.shape[0], codes.shape[1],
        )
        entry = (codes, transcript)
        self._remember(key, entry)
        return entry

    def _remember(self, key: tuple[Any, ...], entry: tuple[Any, str]) -> None:
        self._reference_cache[key] = entry
        while len(self._reference_cache) > _REFERENCE_CACHE_SIZE:
            self._reference_cache.popitem(last=False)

    def save_voice_preset(
        self,
        path: str,
        voice_ref: str | bytes | None = None,
        *,
        transcript: str | None = None,
    ) -> dict[str, Any]:
        """Tokenizes the reference clip once and saves it as a voice preset.

        Point ``config.voice_preset`` at the file to narrate with it: the codec
        encoder is then never needed. A preset works with every model that
        shares the codec (the 1.7B and both 8B models share one).

        Parameters
        ----------
        path : str
            Destination ``.pt`` file.
        voice_ref : str | bytes | None
            Reference clip (path or WAV bytes); ``config.voice_file`` if None.
        transcript : str | None
            Transcript of the clip; ``config.voice_transcript`` or the clip's
            sidecar file if None. Only the continuation modes use it.

        Returns
        -------
        dict[str, Any]
            JSON-safe description of the preset, including ``"path"``.

        Raises
        ------
        RuntimeError
            If there is no usable reference clip.
        """
        if not path.lower().endswith(".pt"):
            raise ValueError("MOSS-TTS voice presets must be saved with a .pt extension.")
        with self._lock:
            self._ensure_initialised()
            self.bind_device()
            import torch

            spec = _MODEL_SPECS[self._loaded_model_id]
            with torch.no_grad():
                # Tokenize the clip itself, not whatever preset is configured now.
                codes, clip_transcript = self._reference(torch, voice_ref, spec, allow_preset=False)
            payload = {
                "format": _PRESET_FORMAT,
                "model": self._loaded_model_id,
                "codec_repo": _CODEC_SPECS[spec.codec].repo,
                "sampling_rate": self._sample_rate,
                "codes": codes.to(torch.long).cpu(),
                "transcript": (clip_transcript if transcript is None else str(transcript)).strip(),
            }
            directory = os.path.dirname(path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            temporary = f"{path}.{os.getpid()}.tmp"
            torch.save(payload, temporary)
            os.replace(temporary, path)
            return self._describe_preset(path, payload)

    def load_voice_preset(self, path: str) -> dict[str, Any]:
        """Loads a preset written by :meth:`save_voice_preset` for this instance.

        The codes are kept in the reference cache, so a run whose
        ``config.voice_preset`` is *path* starts without reading it again.

        Parameters
        ----------
        path : str
            Preset file.

        Returns
        -------
        dict[str, Any]
            JSON-safe description of the preset.

        Raises
        ------
        ValueError
            If the file is missing or is not a preset for the loaded model's codec.
        """
        if not path or not os.path.isfile(path):
            raise ValueError(f"Voice preset '{path}' does not exist.")
        with self._lock:
            self._ensure_initialised()
            import torch

            spec = _MODEL_SPECS[self._loaded_model_id]
            payload = self._read_preset(torch, path, spec)
            configured = (getattr(self.config, "voice_transcript", "") or "").strip()
            self._remember(
                self._preset_key(path, configured),
                (payload["codes"], configured or payload["transcript"]),
            )
            return self._describe_preset(path, payload)

    # ── Settings ─────────────────────────────────────────────────────────────

    def _language(self, spec: _ModelSpec) -> str | None:
        """Maps ``config.language`` to the tag upstream's prompt expects."""
        mode = str(self.option("language_tag", "auto")).strip().lower()
        if mode == "never" or (mode != "always" and not spec.language_tags):
            return None
        raw = (getattr(self.config, "language", "") or "").strip()
        if raw.lower() in _NO_LANGUAGE:
            return None
        name = _LANGUAGE_ALIASES.get(raw.lower())
        if name is None:
            self._warn_once(
                f"lang:{raw}", "MOSS-TTS has no language tag for '%s'; sending none.", raw
            )
            return None
        if name not in spec.languages:
            self._warn_once(
                f"lang:{name}", "%s was not trained on %s; expect a foreign accent or errors.",
                self._loaded_model_id, name,
            )
        return name

    def _shared_setting(self, name: str, recommended: float) -> float:
        """Resolves one shared sampling field of ``config`` for the loaded model.

        Parameters
        ----------
        name : str
            ``AudiobookConfig`` field: ``temperature``, ``top_p``, ``top_k`` or
            ``repetition_penalty``.
        recommended : float
            Upstream's value of that field for the loaded model.

        Returns
        -------
        float
            The configured value, or *recommended* when ``sampling_preset`` is
            ``"recommended"``, or when it is ``"auto"`` and the configured value
            is a stock default: the ``AudiobookConfig`` default (tuned for
            Qwen3-TTS) or ``INFO.recommended_settings`` (the default model's
            operating point, which the UI fills in). A value the user moved
            away from both is always honoured.
        """
        preset = str(self.option("sampling_preset", "auto")).strip().lower()
        value = getattr(self.config, name, None)
        if preset == "recommended" or value is None:
            return recommended
        if preset == "config":
            return value
        stock: list[Any] = [self.INFO.recommended_settings.get(name)]
        class_default = getattr(type(self.config), name, None)
        if isinstance(class_default, (int, float)):
            stock.append(class_default)
        try:
            untouched = any(
                default is not None and abs(float(value) - float(default)) < 1e-9 for default in stock
            )
        except (TypeError, ValueError):
            return recommended
        return recommended if untouched else value

    def _generation_kwargs(self, spec: _ModelSpec) -> dict[str, Any]:
        """Builds the sampling keywords for ``model.generate``.

        Only keywords the loaded architecture's ``generate`` declares are
        returned: the Delay model takes no ``**kwargs``.
        """
        audio_temperature = float(self._shared_setting("temperature", spec.audio_temperature))
        audio_top_p = float(self._shared_setting("top_p", spec.audio_top_p))
        audio_top_k = int(self._shared_setting("top_k", spec.audio_top_k))
        audio_repetition_penalty = float(
            self._shared_setting("repetition_penalty", spec.audio_repetition_penalty)
        )
        text_temperature = float(self.option("text_temperature", -1.0))
        if text_temperature < 0:
            text_temperature = spec.text_temperature

        kwargs: dict[str, Any] = {
            "text_temperature": text_temperature,
            "text_top_p": min(1.0, max(0.01, float(self.option("text_top_p", 1.0)))),
            "text_top_k": max(1, int(self.option("text_top_k", 50))),
            "audio_temperature": max(0.0, audio_temperature),
            "audio_top_p": min(1.0, max(0.01, audio_top_p)),
            "audio_top_k": max(1, audio_top_k),
            "audio_repetition_penalty": max(0.1, audio_repetition_penalty),
        }
        if spec.architecture == "local":
            depth = int(self.option("rvq_depth", 0))
            if 0 < depth < spec.n_vq:
                kwargs["n_vq_for_inference"] = depth
        elif spec.architecture == "local_v15":
            # One do_sample flag covers both layers, and sampling rejects a
            # temperature of zero, so greedy audio means greedy everything.
            do_sample = kwargs["audio_temperature"] > 0
            kwargs["do_sample"] = do_sample
            if not do_sample:
                kwargs["audio_temperature"] = 1.0
            if not do_sample or kwargs["text_temperature"] <= 0:
                kwargs["text_temperature"] = spec.text_temperature
        return kwargs

    def _settings(self, spec: _ModelSpec, transcript: str) -> _Settings:
        """Reads every per-call setting from ``config`` and its options."""
        clone_mode = str(self.option("clone_mode", "reference")).strip().lower()
        if clone_mode not in ("reference", "continuation", "reference+continuation"):
            clone_mode = "reference"
        if clone_mode != "reference" and not transcript:
            self._warn_once(
                "continuation",
                "MOSS-TTS clone_mode '%s' needs the reference clip's transcript; none was "
                "given, so cloning from the clip alone.", clone_mode,
            )
            clone_mode = "reference"

        try:
            speed = float(getattr(self.config, "speed", 1.0) or 1.0)
        except (TypeError, ValueError):
            speed = 1.0
        speed = min(_MAX_SPEED, max(_MIN_SPEED, speed))
        duration_mode = str(self.option("duration_control", "auto")).strip().lower()
        duration_control = duration_mode == "on" or (duration_mode == "auto" and abs(speed - 1.0) > 1e-3)
        if clone_mode != "reference":
            # Upstream disables duration control in continuation modes.
            if duration_control and abs(speed - 1.0) > 1e-3:
                self._warn_once(
                    "speed", "MOSS-TTS cannot honour speed=%.2f in clone_mode '%s'.", speed, clone_mode
                )
            duration_control = False

        pause_mode = str(self.option("pause_tags", "auto")).strip().lower()
        if pause_mode == "auto":
            pause_mode = "native" if spec.native_pause_tags else "emulate"
        elif pause_mode not in ("emulate", "strip"):
            pause_mode = "emulate"

        instruction = str(self.option("instruction", "") or "").strip()
        return _Settings(
            clone_mode=clone_mode,
            duration_control=duration_control,
            speed=speed,
            duration_scale=min(4.0, max(0.25, float(self.option("duration_scale", 1.0)))),
            max_duration_factor=max(1.1, float(self.option("max_duration_factor", 2.5))),
            hard_frame_cap=max(0, int(self.option("max_new_tokens", 0))),
            runaway_retries=max(0, int(self.option("runaway_retries", 2))),
            pause_mode=pause_mode,
            language=self._language(spec),
            instruction=instruction or None,
            trim_eos_frame=bool(self.option("trim_eos_frame", True)) and spec.architecture == "local",
            max_batch_size=max(1, int(self.option("max_batch_size", 8))),
            generation=self._generation_kwargs(spec),
        )

    # ── Planning ─────────────────────────────────────────────────────────────

    def _budget(self, text: str, settings: _Settings) -> tuple[int, int | None]:
        """Returns ``(frame_cap, duration_tokens)`` for one piece of text.

        The cap is a generous multiple of the expected length, so a chunk can
        never run for the many minutes upstream's default budget allows.
        """
        spoken = _PAUSE_TAG_RE.sub(" ", text)
        native_pause = sum(float(m.group(1)) for m in _PAUSE_TAG_RE.finditer(text))
        expected = (
            _estimate_frames(spoken) * settings.duration_scale / settings.speed
            + native_pause * _FRAMES_PER_SECOND
        )
        frame_cap = max(_MIN_FRAME_CAP, int(math.ceil(expected * settings.max_duration_factor)))
        if settings.hard_frame_cap > 0:
            frame_cap = min(frame_cap, settings.hard_frame_cap)
        tokens = max(1, int(round(expected))) if settings.duration_control else None
        return frame_cap, tokens

    def _plan(self, texts: list[str], settings: _Settings) -> list[_Piece]:
        """Turns the input texts into speech pieces and emulated pauses."""
        pieces: list[_Piece] = []
        for item, raw in enumerate(texts):
            text = " ".join(str(raw or "").split())
            if settings.pause_mode == "strip":
                text = " ".join(_PAUSE_TAG_RE.sub(" ", text).split())
            if not text:
                raise RuntimeError(f"MOSS-TTS was given an empty text (item {item}).")
            parts: list[str | float] = (
                _split_pauses(text) if settings.pause_mode == "emulate" else [text]
            )
            spoken = [part for part in parts if isinstance(part, str)]
            if not spoken:
                raise RuntimeError(f"MOSS-TTS was given only pause markers (item {item}): {raw!r}")
            for part in parts:
                if isinstance(part, str):
                    frame_cap, tokens = self._budget(part, settings)
                    pieces.append(_Piece(item=item, text=part, frame_cap=frame_cap, tokens=tokens))
                else:
                    pieces.append(_Piece(item=item, pause_seconds=float(part)))
        return pieces

    def _conversation(
        self, piece: _Piece, codes: Any, transcript: str, settings: _Settings
    ) -> list[dict[str, Any]]:
        """Builds upstream's message list for one piece."""
        processor = self._processor
        fields: dict[str, Any] = {"text": piece.text}
        if settings.language:
            fields["language"] = settings.language
        if piece.tokens is not None:
            fields["tokens"] = int(piece.tokens)
        if settings.instruction:
            fields["instruction"] = settings.instruction
        if settings.clone_mode == "reference":
            fields["reference"] = [codes]
            return [processor.build_user_message(**fields)]
        # Continuation: the clip is the start of the assistant's audio, and
        # its transcript must lead the text.
        fields["text"] = _join_prefix(transcript, piece.text)
        if settings.clone_mode == "reference+continuation":
            fields["reference"] = [codes]
        return [
            processor.build_user_message(**fields),
            processor.build_assistant_message(audio_codes_list=[codes]),
        ]

    # ── Generation ───────────────────────────────────────────────────────────

    def _step_budget(self, spec: _ModelSpec, frame_cap: int) -> int:
        """``max_new_tokens`` that lets a clip of *frame_cap* frames finish.

        The Local models spend one step on the stop decision. The Delay model
        also emits ``<|audio_start|>``, ``n_vq - 1`` delay steps that flush
        the codebooks, ``<|audio_end|>`` and ``<|im_end|>``.
        """
        overhead = spec.n_vq + 2 if spec.architecture == "delay" else 1
        return frame_cap + overhead + _BUDGET_SLACK_FRAMES

    def _generate(
        self, torch: Any, spec: _ModelSpec, group: list[_Piece],
        codes: Any, transcript: str, settings: _Settings,
    ) -> list[Any]:
        """One forward pass for *group*; returns a waveform or None per piece."""
        processor, model = self._processor, self._model
        conversations = [self._conversation(piece, codes, transcript, settings) for piece in group]
        mode = "generation" if settings.clone_mode == "reference" else "continuation"
        batch = processor(conversations, mode=mode)
        outputs = model.generate(
            input_ids=batch["input_ids"].to(self._device),
            attention_mask=batch["attention_mask"].to(self._device),
            max_new_tokens=self._step_budget(spec, max(piece.frame_cap for piece in group)),
            **settings.generation,
        )
        messages = processor.decode(outputs)
        if len(messages) != len(group):
            raise RuntimeError(
                f"MOSS-TTS decoded {len(messages)} results for a batch of {len(group)} on {self._device}."
            )
        eos_samples = int(round(self._sample_rate / _FRAMES_PER_SECOND))
        waveforms: list[Any] = []
        for message in messages:
            clips = getattr(message, "audio_codes_list", None) if message is not None else None
            audio = clips[0] if clips else None
            if audio is not None:
                audio = audio.detach().to(device="cpu", dtype=torch.float32)
                if settings.trim_eos_frame and audio.shape[-1] > eos_samples:
                    audio = audio[..., :-eos_samples]
                if audio.shape[-1] == 0:
                    audio = None
            waveforms.append(audio)
        return waveforms

    def _is_runaway(self, piece: _Piece, audio: Any) -> bool:
        frames = audio.shape[-1] * _FRAMES_PER_SECOND / float(self._sample_rate)
        return frames > piece.frame_cap + _RUNAWAY_TOLERANCE_FRAMES

    def _run_group(
        self, torch: Any, spec: _ModelSpec, group: list[_Piece],
        codes: Any, transcript: str, settings: _Settings,
    ) -> None:
        """Generates *group*, falling back to one piece at a time when needed.

        Raises
        ------
        RuntimeError
            If a piece still has no usable audio after its retries, or the
            device runs out of memory on a single piece.
        """
        out_of_memory = getattr(torch.cuda, "OutOfMemoryError", ())
        self.seed_everything()
        batched: list[Any] | None = None
        if len(group) > 1:
            try:
                batched = self._generate(torch, spec, group, codes, transcript, settings)
            except out_of_memory:
                logger.warning(
                    "[MOSS-TTS] Out of memory on %s with a batch of %d; retrying one chunk at a time.",
                    self._device, len(group),
                )
                self._empty_cache()
                self.seed_everything()

        for index, piece in enumerate(group):
            audio = batched[index] if batched is not None else None
            # Without a batched take, the first single take is not a retry.
            retries = 0 if batched is not None else -1
            while audio is None or self._is_runaway(piece, audio):
                if retries >= settings.runaway_retries:
                    problem = (
                        "produced no audio" if audio is None else
                        f"ran past its limit of {piece.frame_cap / _FRAMES_PER_SECOND:.0f} s "
                        "(runaway generation)"
                    )
                    raise RuntimeError(
                        f"MOSS-TTS chunk {problem} on {self._device} after "
                        f"{settings.runaway_retries} retries: {piece.text[:80]!r}"
                    )
                if retries >= 0:
                    logger.warning(
                        "[MOSS-TTS] Regenerating a chunk on %s (%s): %r",
                        self._device, "no audio" if audio is None else "runaway", piece.text[:60],
                    )
                retries += 1
                try:
                    audio = self._generate(torch, spec, [piece], codes, transcript, settings)[0]
                except out_of_memory as exc:
                    self._empty_cache()
                    raise RuntimeError(
                        f"MOSS-TTS ran out of memory on {self._device} synthesizing a single "
                        f"chunk of {len(piece.text)} characters. Lower max_reference_seconds "
                        "or use a smaller model."
                    ) from exc
            piece.audio = audio

    def _synthesize_many(self, texts: list[str], voice_ref: str | bytes | None) -> list[Any]:
        """Returns one CPU float32 waveform per text, in input order."""
        with self._lock:
            self._ensure_initialised()
            self.bind_device()
            import torch

            spec = _MODEL_SPECS[self._loaded_model_id]
            with torch.no_grad():
                codes, transcript = self._reference(torch, voice_ref, spec)
                settings = self._settings(spec, transcript)
                pieces = self._plan(texts, settings)
                speech = [piece for piece in pieces if piece.text]
                for start in range(0, len(speech), settings.max_batch_size):
                    self._run_group(
                        torch, spec, speech[start:start + settings.max_batch_size],
                        codes, transcript, settings,
                    )

            results: list[Any] = []
            for item in range(len(texts)):
                parts: list[Any] = []
                item_pieces = [piece for piece in pieces if piece.item == item]
                shape = next(piece.audio.shape[:-1] for piece in item_pieces if piece.text)
                for piece in item_pieces:
                    if piece.text:
                        parts.append(piece.audio)
                    else:
                        samples = int(round(piece.pause_seconds * self._sample_rate))
                        parts.append(torch.zeros(*shape, samples, dtype=torch.float32))
                results.append(parts[0] if len(parts) == 1 else torch.cat(parts, dim=-1))
            return results

    # ── Public API ───────────────────────────────────────────────────────────

    def synthesize(
        self,
        text: str,
        voice_ref: str | bytes,
        out_path: str | None = None,
        *,
        return_bytes: bool = False,
    ) -> tuple[str | bytes, float]:
        """Generates speech for *text* in the reference voice.

        Parameters
        ----------
        text : str
            One chunk of text. May contain ``[pause 1.5s]`` markers, tone-numbered
            Pinyin and ``/IPA/`` spans, which are passed to the model as written.
        voice_ref : str | bytes
            Reference clip (path or WAV bytes); ``config.voice_file`` if empty.
        out_path : str | None
            Where to write the WAV when *return_bytes* is False.
        return_bytes : bool
            Return WAV bytes instead of writing *out_path*.

        Returns
        -------
        tuple[str | bytes, float]
            ``(out_path or wav_bytes, duration_seconds)`` at :attr:`sample_rate`.

        Raises
        ------
        RuntimeError
            If the chunk cannot be synthesized.
        """
        audio = self._synthesize_many([text], voice_ref)[0]
        return self.finish(audio, self._sample_rate, out_path, return_bytes)

    def synthesize_batch(
        self,
        texts: list[str],
        voice_ref: bytes,
        *,
        return_bytes: bool = True,
    ) -> list[tuple[bytes | str, float]]:
        """Synthesizes several chunks in batched forward passes, in input order.

        Parameters
        ----------
        texts : list[str]
            Chunks to synthesize, each at most ``config.max_len`` characters.
        voice_ref : bytes
            Reference clip WAV bytes, the same for every chunk.
        return_bytes : bool
            Always True from the pipeline; WAV bytes are returned.

        Returns
        -------
        list[tuple[bytes | str, float]]
            One ``(wav_bytes, duration_seconds)`` per text.

        Raises
        ------
        RuntimeError
            If any chunk cannot be synthesized.
        """
        if not texts:
            return []
        waveforms = self._synthesize_many(list(texts), voice_ref)
        return [self.finish(audio, self._sample_rate, None, True) for audio in waveforms]
