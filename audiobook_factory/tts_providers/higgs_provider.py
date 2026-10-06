"""
audiobook_factory/tts_providers/higgs_provider.py
==================================================
Higgs Audio v3 ("Higgs TTS 3", Boson AI) zero-shot voice-cloning provider.

What upstream ships (read 2026-10-07)
-------------------------------------
``bosonai/higgs-tts-3-4b`` (alias ``bosonai/higgs-audio-v3-tts-4b``) is a
weights-only repository: ``config.json``, ``tokenizer.json`` and one
``model.safetensors`` (4.65 B bf16 parameters, 9.3 GB). It contains no Python
code and ``transformers`` has no ``higgs_multimodal_qwen3`` model class (checked
up to 5.19.0), so there is **no official in-process inference API**. Boson's
supported paths are servers: SGLang-Omni (recommended), vLLM-Omni, the hosted
API, and MLX-Audio on Apple Silicon.

This module is therefore an in-process port of the serving reference
implementation, built only from stock ``transformers`` classes, so one instance
can live on one GPU inside the provider pool:

* backbone  — ``transformers.Qwen3Model`` built from ``config.json``'s
  ``text_config``; checkpoint keys ``body.*`` / ``tied.embedding.text_embedding.*``
  map 1:1 onto it (398 of 398 keys, verified against the checkpoint header).
* audio in/out — one fused ``[8 * 1026, 2560]`` matrix
  (``tied.embedding.modality_embeddings.0.embedding.weight``) used both as the
  summed multi-codebook input embedding and, tied, as the output head.
* codec — ``transformers.HiggsAudioV2TokenizerModel`` (24 kHz, 25 frames/s,
  8 codebooks); its weights are bundled in the same checkpoint under
  ``tied.embedding.modality_embeddings.0.model.*``.

Sources ported (all Apache-2.0):
``sgl-project/sglang-omni@4a3e962`` ``sglang_omni/models/higgs_tts/``
(``text_tokenizer.py`` prompt, ``utils.py`` delay pattern, ``sampler.py`` state
machine, ``model_runner.py`` prefill overlay, ``audio_codec.py`` codec loading,
``vocoder_scheduler.py`` de-delay) and, for the logit constraints,
``vllm-project/vllm-omni@e61cd42``
``vllm_omni/model_executor/models/higgs_audio_v3/higgs_audio_v3_talker.py``.

Prompt (token ids; identical in both reference servers)
-------------------------------------------------------
::

    <|tts|> [<|ref_text|> tok(transcript)] <|ref_audio|> [-100] x R <|text|> tok(text) <|audio|>

``R`` is the number of *delayed* reference rows (``frames + 8 - 1``); the
``-100`` slots are replaced by the fused embedding of the reference codes.
Without a reference the prompt is ``<|tts|> <|text|> tok(text) <|audio|>``.
There is no system prompt and no natural-language scene description in v3:
style is steered only by inline ``<|emotion:..|>``, ``<|style:..|>``,
``<|prosody:..|>`` and ``<|sfx:..|>`` tokens inside the text.

How the narrator voice is resolved (first match wins)
-----------------------------------------------------
1. ``config.voice_preset`` — a ``.pt`` file written by
   :meth:`HiggsAudioProvider.save_voice_preset` (pre-encoded reference codes,
   upstream's ``reference_codes``, read with ``torch.load(weights_only=True)``);
   any other file is used as a reference clip.
2. A reference clip (the ``voice_ref`` argument, else ``config.voice_file``),
   with its transcript from ``config.voice_transcript`` or a sidecar ``.txt``.
   The clip is encoded once per (file content, transcript) and every chunk is
   conditioned on exactly the same codes, so the narrator cannot drift.
3. No clip — *smart voice*. Upstream picks a new speaker for every request, so
   the provider synthesizes one short utterance, keeps its codes as the
   reference for the rest of the book and stores them in the voice cache on
   disk so every GPU instance and a resumed run speak with the same voice.

Not mapped because v3 has no equivalent: ``tts_timbre`` (the open weights ship
no preset speakers), ``repetition_penalty`` (and v2's repetition-aware
sampling, which v3 dropped), ``language`` (detected from the text), ``speed``
(only four coarse ``<|prosody:speed_*|>`` tokens; the pipeline time-stretches)
and free-text ``tts_instruct`` (only control tags found in it are used).
``quantization="int8"`` is ignored: there is no official quantized path and a
bitsandbytes build of this model could not be validated.
"""
from __future__ import annotations

import copy
import gc
import hashlib
import json
import logging
import math
import os
import re
import tempfile
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from audiobook_factory.tts_providers.base_tts_provider import (
    BaseTTSProvider,
    ProviderInfo,
    ProviderOption,
)

if TYPE_CHECKING:
    from audiobook_factory.pipeline import AudiobookConfig

logger = logging.getLogger(__name__)

# ── Model identity ────────────────────────────────────────────────────────────
_DEFAULT_MODEL: str = "bosonai/higgs-tts-3-4b"
_MODEL_ALIAS: str = "bosonai/higgs-audio-v3-tts-4b"  # redirects to _DEFAULT_MODEL on the Hub
_MODEL_TYPE: str = "higgs_multimodal_qwen3"
_SNAPSHOT_FILES: tuple[str, ...] = (
    "config.json",
    "tokenizer.json",
    "model.safetensors",
    "model.safetensors.index.json",
    "model-*.safetensors",
)

# ── Dependencies ──────────────────────────────────────────────────────────────
# HiggsAudioV2TokenizerModel first shipped in transformers 5.3.0; the checkpoint
# was exported with 5.5.0 and the reference server pins 5.12.1.
_MIN_TRANSFORMERS: tuple[int, int, int] = (5, 5, 0)
_INSTALL_COMMAND: str = 'pip install "transformers>=5.5.0,<6" "accelerate>=1.1.0" torchaudio'

# ── Checkpoint layout ─────────────────────────────────────────────────────────
_CODEC_PREFIX: str = "tied.embedding.modality_embeddings.0.model."
_AUDIO_EMBEDDING_KEY: str = "tied.embedding.modality_embeddings.0.embedding.weight"
_AUDIO_HEAD_KEY: str = "tied.head.modality_heads.0.weight"
_SKIPPED_PREFIXES: tuple[str, ...] = (_CODEC_PREFIX, "tied.head.")
_BACKBONE_KEY_MAP: tuple[tuple[str, str], ...] = (
    ("tied.embedding.text_embedding.", "embed_tokens."),
    ("body.", ""),
)
_QWEN3_ROPE_THETA: int = 1_000_000

# Architecture of the bundled codec. The checkpoint carries the codec weights
# but not their config; this is bosonai/higgs-audio-v2-tokenizer's config.json
# (the same file sglang-omni bundles), minus bookkeeping keys.
_CODEC_CONFIG: dict[str, Any] = {
    "model_type": "higgs_audio_v2_tokenizer",
    "sample_rate": 24000,
    "semantic_sample_rate": 16000,
    "downsample_factor": 320,
    "target_bandwidths": [0.5, 1, 1.5, 2],
    "codebook_size": 1024,
    "codebook_dim": 64,
    "kernel_size": 3,
    "unit_kernel_size": 3,
    "channel_ratios": [1, 1],
    "strides": [1, 1],
    "block_dilations": [1, 1],
    "initializer_range": 0.02,
    "acoustic_model_config": {
        "model_type": "dac",
        "codebook_dim": 8,
        "codebook_loss_weight": 1.0,
        "codebook_size": 1024,
        "commitment_loss_weight": 0.25,
        "decoder_hidden_size": 1024,
        "downsampling_ratios": [8, 5, 4, 2, 3],
        "encoder_hidden_size": 64,
        "hidden_size": 256,
        "hop_length": 960,
        "n_codebooks": 9,
        "quantizer_dropout": 0,
        "sampling_rate": 16000,
        "upsampling_ratios": [8, 5, 4, 2, 3],
    },
    "semantic_model_config": {
        "model_type": "hubert",
        "activation_dropout": 0.1,
        "apply_spec_augment": True,
        "attention_dropout": 0.1,
        "bos_token_id": 1,
        "classifier_proj_size": 256,
        "conv_bias": False,
        "conv_dim": [512, 512, 512, 512, 512, 512, 512],
        "conv_kernel": [10, 3, 3, 3, 3, 2, 2],
        "conv_pos_batch_norm": False,
        "conv_stride": [5, 2, 2, 2, 2, 2, 2],
        "ctc_loss_reduction": "sum",
        "ctc_zero_infinity": False,
        "do_stable_layer_norm": False,
        "eos_token_id": 2,
        "feat_extract_activation": "gelu",
        "feat_extract_norm": "group",
        "feat_proj_dropout": 0.0,
        "feat_proj_layer_norm": True,
        "final_dropout": 0.1,
        "hidden_act": "gelu",
        "hidden_dropout": 0.1,
        "hidden_size": 768,
        "initializer_range": 0.02,
        "intermediate_size": 3072,
        "layer_norm_eps": 1e-05,
        "layerdrop": 0.1,
        "mask_feature_length": 10,
        "mask_feature_min_masks": 0,
        "mask_feature_prob": 0.0,
        "mask_time_length": 10,
        "mask_time_min_masks": 2,
        "mask_time_prob": 0.0,
        "num_attention_heads": 12,
        "num_conv_pos_embedding_groups": 16,
        "num_conv_pos_embeddings": 128,
        "num_feat_extract_layers": 7,
        "num_hidden_layers": 12,
        "pad_token_id": 0,
        "use_weighted_layer_sum": False,
        "vocab_size": 32,
    },
}
# Present in the checkpoint, absent from the transformers module (unused: mask_time_prob is 0).
_CODEC_IGNORED_KEYS: frozenset[str] = frozenset({"semantic_model.masked_spec_embed"})

# ── Prompt ────────────────────────────────────────────────────────────────────
_AUDIO_PLACEHOLDER_ID: int = -100
_TOKEN_TTS: str = "<|tts|>"
_TOKEN_REF_TEXT: str = "<|ref_text|>"
_TOKEN_REF_AUDIO: str = "<|ref_audio|>"
_TOKEN_TEXT: str = "<|text|>"
_TOKEN_AUDIO: str = "<|audio|>"
_REQUIRED_SPECIALS: tuple[str, ...] = (_TOKEN_TTS, _TOKEN_REF_AUDIO, _TOKEN_TEXT, _TOKEN_AUDIO)
_CONTEXT_LENGTH: int = 8192  # training sequence length (model card)

# ── Control tags (the 43 documented in upstream PROMPTING.md) ─────────────────
_EMOTIONS: tuple[str, ...] = (
    "elation", "amusement", "enthusiasm", "determination", "pride", "contentment",
    "affection", "relief", "contemplation", "confusion", "surprise", "awe", "longing",
    "arousal", "anger", "fear", "disgust", "bitterness", "sadness", "shame", "helplessness",
)
_STYLES: tuple[str, ...] = ("singing", "shouting", "whispering")
_SOUND_EFFECTS: tuple[str, ...] = (
    "cough", "laughter", "crying", "screaming", "burping", "humming", "sigh", "sniff", "sneeze",
)
_SPEEDS: tuple[str, ...] = ("very_slow", "slow", "fast", "very_fast")
_PITCHES: tuple[str, ...] = ("low", "high")
_EXPRESSIVENESS: tuple[str, ...] = ("high", "low")
_PAUSES: tuple[str, ...] = ("pause", "long_pause")
_CONTROL_TAGS: frozenset[str] = frozenset(
    [f"<|emotion:{name}|>" for name in _EMOTIONS]
    + [f"<|style:{name}|>" for name in _STYLES]
    + [f"<|sfx:{name}|>" for name in _SOUND_EFFECTS]
    + [f"<|prosody:speed_{name}|>" for name in _SPEEDS]
    + [f"<|prosody:pitch_{name}|>" for name in _PITCHES]
    + [f"<|prosody:expressive_{name}|>" for name in _EXPRESSIVENESS]
    + [f"<|prosody:{name}|>" for name in _PAUSES]
)
_TAG_PATTERN: re.Pattern[str] = re.compile(r"<\|[^<>|\s]{1,48}\|>")
_NONE_CHOICE: str = "none"
# Documented slow-down of the speed tokens; used to widen the frame budget.
_SPEED_FACTORS: dict[str, float] = {
    "<|prosody:speed_very_slow|>": 0.65,
    "<|prosody:speed_slow|>": 0.85,
}

# ── Generation ────────────────────────────────────────────────────────────────
_GREEDY_TEMPERATURE: float = 1e-5
_RECOMMENDED_TEMPERATURE: float = 0.8   # upstream's voice-cloning recipe (with top_k 50)
_DEFAULT_MAX_NEW_TOKENS: int = 2048      # upstream server default (82 s of audio)
_DEFAULT_DURATION_MARGIN: float = 2.0
_DEFAULT_MAX_BATCH: int = 8
_CHARS_PER_SECOND: float = 14.0          # alphabetic scripts, narration pace
_CJK_CHARS_PER_SECOND: float = 4.0       # Han / kana / hangul
_BUDGET_FLOOR_SECONDS: float = 4.0
_PAUSE_TAG_SECONDS: float = 1.5
_SFX_TAG_SECONDS: float = 1.0
_TOKEN_LIMIT_RETRIES: int = 2
_MIN_FRAME_BUDGET: int = 64

# ── Reference clip ────────────────────────────────────────────────────────────
_MAX_REFERENCE_SECONDS: float = 100.0    # upstream hard limit
_LONG_REFERENCE_SECONDS: float = 30.0
_MIN_REFERENCE_SECONDS: float = 1.0      # shorter clips are zero-padded upstream
_MIN_VOICE_BYTES: int = 100
_REFERENCE_CACHE_SIZE: int = 4
_PRESET_SUFFIX: str = ".pt"
_PRESET_FORMAT: str = "higgs-v3-reference-codes"
_VOICE_CACHE_DIR_NAME: str = "abm_voice_refs"
_SMART_VOICE_CHARS: int = 160
_SMART_VOICE_LOCK = threading.Lock()

# ── Memory ────────────────────────────────────────────────────────────────────
_BYTES_PER_GB: float = 1024.0 ** 3
_BACKBONE_PARAMETERS: int = 4_043_480_576  # 36 layers + text embedding + fused audio embedding
_CODEC_PARAMETERS: int = 201_401_321       # kept in float32
_WORKING_SET_GB: float = 1.0               # KV cache + activations for one chunk
_NATIVE_BF16_CAPABILITY: int = 8           # CUDA compute capability major (Ampere and newer)

_LANGUAGES: tuple[str, ...] = (
    # WER/CER under 5 on the model card (85)
    "Afrikaans", "Arabic", "Armenian", "Assamese", "Asturian", "Azerbaijani", "Bashkir", "Basque",
    "Belarusian", "Bengali", "Bosnian", "Bulgarian", "Catalan", "Cebuano", "Central Kurdish",
    "Chinese", "Croatian", "Czech", "Danish", "Dutch", "Eastern Mari", "English", "Esperanto",
    "Estonian", "Finnish", "French", "Galician", "Georgian", "German", "Greek", "Gujarati",
    "Haitian Creole", "Hausa", "Hebrew", "Hindi", "Hungarian", "Indonesian", "Italian", "Japanese",
    "Javanese", "Kannada", "Kazakh", "Korean", "Kinyarwanda", "Kyrgyz", "Latvian", "Lingala",
    "Lithuanian", "Luo", "Macedonian", "Malay", "Malayalam", "Maltese", "Māori", "Marathi",
    "Mongolian", "Nepali", "Norwegian", "Occitan", "Persian", "Polish", "Portuguese", "Romanian",
    "Russian", "Sepedi", "Serbian", "Shona", "Slovak", "Slovene", "Spanish", "Swahili", "Swedish",
    "Tagalog", "Tajik", "Tamil", "Telugu", "Thai", "Turkish", "Ukrainian", "Urdu", "Uyghur",
    "Uzbek", "Vietnamese", "Xhosa", "Zulu",
    # WER/CER between 5 and 10 (17)
    "Albanian", "Chichewa/Nyanja", "Eastern Punjabi", "Ganda", "Icelandic", "Irish", "Kabyle",
    "Kabuverdianu", "Kamba", "Latin", "Luxembourgish", "Oromo", "Pashto", "Sindhi", "Somali",
    "Umbundu", "Welsh",
)


class _NonFiniteLogitsError(RuntimeError):
    """The backbone produced NaN/inf logits (float16 overflow)."""


@dataclass(frozen=True)
class _Reference:
    """One encoded narrator reference, shared by every chunk.

    Attributes
    ----------
    key : str
        Cache key (model id, clip content hash, transcript hash).
    delayed_codes : Any
        ``LongTensor [frames + codebooks - 1, codebooks]`` on the CPU.
    transcript : str
        Transcript placed after ``<|ref_text|>``; empty when unknown.
    seconds : float
        Duration of the reference audio.
    source : str
        Human-readable origin, for log and error messages.
    """

    key: str
    delayed_codes: Any
    transcript: str
    seconds: float
    source: str


@dataclass(frozen=True)
class _Settings:
    """Generation settings read from ``self.config`` for one call."""

    temperature: float
    top_k: int | None
    top_p: float | None
    seed: int
    max_new_tokens: int
    duration_margin: float
    max_batch_size: int
    prefix_tags: tuple[str, ...]
    use_transcript: bool


@dataclass(frozen=True)
class _Job:
    """One chunk ready for generation."""

    text: str
    prompt_ids: list[int]
    budget: int


# ── Pure helpers ──────────────────────────────────────────────────────────────

def _version_tuple(version: str) -> tuple[int, int, int]:
    """Parses ``"5.12.1"`` / ``"5.6.0.dev0"`` into a comparable 3-tuple."""
    parts = [int(piece) for piece in re.findall(r"\d+", version.split("+")[0])[:3]]
    while len(parts) < 3:
        parts.append(0)
    return parts[0], parts[1], parts[2]


def _tag_slot(tag: str) -> str:
    """Returns the mutually exclusive group a control tag belongs to."""
    body = tag[2:-2]
    category, _, value = body.partition(":")
    if category in ("emotion", "style"):
        return category
    if category == "prosody":
        for group in ("speed", "pitch", "expressive"):
            if value.startswith(group + "_"):
                return f"prosody:{group}"
    return tag


def _sanitize_text(text: str, allowed_tags: frozenset[str]) -> str:
    """Removes every ``<|...|>`` token that is not an allowed control tag.

    Book text must never be able to inject structural tokens such as
    ``<|audio|>`` or ``<|ref_audio|>`` into the prompt.
    """
    cleaned = _TAG_PATTERN.sub(lambda m: m.group(0) if m.group(0) in allowed_tags else " ", text or "")
    return " ".join(cleaned.split())


def _plain_text(text: str) -> str:
    """Returns *text* without any ``<|...|>`` token."""
    return " ".join(_TAG_PATTERN.sub(" ", text or "").split())


def _is_cjk(char: str) -> bool:
    """True for Han ideographs, kana and hangul, which are spoken one per syllable."""
    code = ord(char)
    return (
        0x3040 <= code <= 0x30FF       # hiragana, katakana
        or 0x3400 <= code <= 0x4DBF    # CJK extension A
        or 0x4E00 <= code <= 0x9FFF    # CJK unified
        or 0xAC00 <= code <= 0xD7AF    # hangul syllables
        or 0xF900 <= code <= 0xFAFF    # CJK compatibility
        or 0x20000 <= code <= 0x2FA1F  # CJK extensions B..F
    )


def _expected_seconds(text: str) -> float:
    """Estimates how long *text* takes to speak at an unhurried narration pace."""
    plain = _plain_text(text)
    cjk = sum(1 for char in plain if _is_cjk(char))
    seconds = (len(plain) - cjk) / _CHARS_PER_SECOND + cjk / _CJK_CHARS_PER_SECOND
    tags = _TAG_PATTERN.findall(text or "")
    seconds += _PAUSE_TAG_SECONDS * sum(1 for tag in tags if tag in ("<|prosody:pause|>", "<|prosody:long_pause|>"))
    seconds += _SFX_TAG_SECONDS * sum(1 for tag in tags if tag.startswith("<|sfx:"))
    slowest = min((_SPEED_FACTORS.get(tag, 1.0) for tag in tags), default=1.0)
    return seconds / slowest


def _frame_budget(
    text: str,
    frame_rate: float,
    num_codebooks: int,
    margin: float = _DEFAULT_DURATION_MARGIN,
    cap: int = _DEFAULT_MAX_NEW_TOKENS,
) -> int:
    """Returns the most audio frames (generation steps) *text* may take.

    The budget is ``margin`` times the expected spoken duration plus a fixed
    floor, so a chunk that never emits its end-of-audio code is cut off after
    roughly twice its natural length instead of running to the context limit.
    """
    seconds = _expected_seconds(text) * max(1.0, float(margin)) + _BUDGET_FLOOR_SECONDS
    frames = int(math.ceil(seconds * float(frame_rate))) + int(num_codebooks) - 1
    return max(_MIN_FRAME_BUDGET, min(int(cap), frames))


def _build_prompt_ids(
    specials: dict[str, int | None],
    text_ids: list[int],
    num_ref_tokens: int = 0,
    ref_text_ids: list[int] | None = None,
) -> list[int]:
    """Assembles the Higgs v3 TTS prompt (``HiggsTokenizerAdapter.build_prompt``)."""
    ids: list[int] = [int(specials[_TOKEN_TTS])]  # type: ignore[arg-type]
    ref_text_id = specials.get(_TOKEN_REF_TEXT)
    if ref_text_ids and num_ref_tokens > 0 and ref_text_id is not None:
        ids.append(int(ref_text_id))
        ids.extend(ref_text_ids)
    if num_ref_tokens > 0:
        ids.append(int(specials[_TOKEN_REF_AUDIO]))  # type: ignore[arg-type]
        ids.extend([_AUDIO_PLACEHOLDER_ID] * num_ref_tokens)
    ids.append(int(specials[_TOKEN_TEXT]))  # type: ignore[arg-type]
    ids.extend(text_ids)
    ids.append(int(specials[_TOKEN_AUDIO]))  # type: ignore[arg-type]
    return ids


def _apply_delay_pattern(codes: Any, codebook_vocab: int) -> Any:
    """``[T, N]`` raw codes -> ``[T + N - 1, N]`` delayed, BOC/EOC padded.

    Codebook ``c`` is shifted down by ``c`` rows; the rows above it hold the
    begin-of-codes id and the rows below it the end-of-codes id.
    """
    import torch

    frames, books = codes.shape
    boc, eoc = codebook_vocab - 2, codebook_vocab - 1
    delayed = torch.full((frames + books - 1, books), eoc, dtype=torch.long)
    for book in range(books):
        delayed[:book, book] = boc
        delayed[book:book + frames, book] = codes[:, book]
    return delayed


def _reverse_delay_pattern(delayed: Any) -> Any:
    """``[L, N]`` delayed rows -> ``[L - (N - 1), N]`` raw codes."""
    import torch

    length, books = delayed.shape
    frames = length - (books - 1)
    if frames <= 0:
        raise RuntimeError(
            f"Higgs Audio v3 generated only {length} code rows; at least {books} are needed for one frame."
        )
    return torch.stack([delayed[book:book + frames, book] for book in range(books)], dim=1)


def _sample_codes(
    logits: Any,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
    generators: list[Any] | None = None,
) -> Any:
    """Samples every codebook independently: temperature -> top-k -> top-p.

    Parameters
    ----------
    logits : Tensor
        ``[batch, codebooks, vocab]`` float32.
    generators : list[torch.Generator] | None
        One generator per batch row; ``None`` uses torch's global RNG.

    Returns
    -------
    Tensor
        ``[batch, codebooks]`` int64.
    """
    import torch

    if temperature <= _GREEDY_TEMPERATURE:
        return logits.argmax(dim=-1)
    scaled = logits / temperature
    vocab = scaled.shape[-1]
    if top_k is not None and 0 < top_k < vocab:
        kth = scaled.topk(top_k, dim=-1).values[..., -1:]
        scaled = scaled.masked_fill(scaled < kth, float("-inf"))
    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_index = torch.sort(scaled, descending=True, dim=-1)
        remove = sorted_logits.softmax(dim=-1).cumsum(dim=-1) > top_p
        # Shift right so the token that crosses the threshold is kept.
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        scaled = scaled.masked_fill(torch.zeros_like(remove).scatter(-1, sorted_index, remove), float("-inf"))
    probs = scaled.softmax(dim=-1)
    batch, books, _ = probs.shape
    if generators is None:
        return torch.multinomial(probs.reshape(batch * books, vocab), 1).view(batch, books)
    return torch.stack(
        [torch.multinomial(probs[row], 1, generator=generators[row]).squeeze(-1) for row in range(batch)]
    )


def _sampler_step(
    logits: Any,
    delay: Any,
    countdown: Any,
    frozen: Any,
    temperature: float,
    top_k: int | None,
    top_p: float | None,
    generators: list[Any] | None = None,
) -> tuple[Any, Any, Any, Any]:
    """Runs one step of the multi-codebook delay / end-of-codes state machine.

    Mirrors sglang-omni's ``batched_step_direct`` and adds vLLM-Omni's logit
    constraints so a code that is impossible under the delay pattern can never
    be sampled.

    Parameters
    ----------
    logits : Tensor
        ``[batch, codebooks, vocab]`` float32; modified in place.
    delay : Tensor
        ``[batch]`` rows produced so far, saturating at ``codebooks``.
    countdown : Tensor
        ``[batch]`` wind-down steps left after codebook 0 ended; ``-1`` before.
    frozen : Tensor
        ``[batch]`` bool; rows that are finished and must not change state.

    Returns
    -------
    tuple[Tensor, Tensor, Tensor, Tensor]
        ``(codes [batch, codebooks], new_delay, new_countdown, done_now)``.
    """
    import torch

    _, books, vocab = logits.shape
    boc, eoc = vocab - 2, vocab - 1
    book = torch.arange(books, device=logits.device).unsqueeze(0)
    in_delay = delay < books
    winding = (countdown >= 0) & ~in_delay
    normal = ~in_delay & ~winding

    # Only codebook 0 may end the stream, and only once every codebook started.
    logits[..., boc] = float("-inf")
    eoc_logits = logits[..., eoc]
    logits[..., eoc] = torch.where(
        normal.unsqueeze(1) & (book == 0), eoc_logits, torch.full_like(eoc_logits, float("-inf"))
    )
    codes = _sample_codes(logits, temperature, top_k, top_p, generators)

    # Codebook c starts c rows late and ends c rows late.
    not_started = in_delay.unsqueeze(1) & (book > delay.unsqueeze(1))
    codes = torch.where(not_started, torch.full_like(codes, boc), codes)
    wind_step = (books - 1) - countdown
    ended = winding.unsqueeze(1) & (book <= wind_step.unsqueeze(1))
    codes = torch.where(ended, torch.full_like(codes, eoc), codes)

    live = ~frozen
    wind_live = live & winding
    cb0_ended = live & normal & (codes[:, 0] == eoc)
    new_delay = torch.where(live & in_delay, delay + 1, delay)
    if books > 2:
        new_countdown = torch.where(
            cb0_ended,
            torch.full_like(countdown, books - 2),
            torch.where(wind_live, countdown - 1, countdown),
        )
        done_now = wind_live & (new_countdown <= 0)
    else:
        new_countdown = countdown
        done_now = cb0_ended
    return codes, new_delay, new_countdown, done_now


class HiggsAudioProvider(BaseTTSProvider):
    """Higgs Audio v3 TTS, one instance per device.

    Parameters
    ----------
    config : AudiobookConfig
        Read at call time; the pool reassigns ``provider.config`` between runs.
    device : str | None
        Torch device this instance is bound to, e.g. ``"cuda:0"``.
    dtype_override : str | None
        ``"float16"``, ``"bfloat16"`` or ``"float32"``. ``"bfloat16"`` is used
        only on GPUs with native bf16 (Ampere and newer); older GPUs such as
        the T4 run float16 unless the ``dtype`` option forces otherwise.
    """

    INFO = ProviderInfo(
        name="higgs",
        display_name="Higgs Audio v3",
        description=(
            "Boson AI's 4B Higgs TTS 3: zero-shot voice cloning in 102 languages with inline "
            "emotion, style, prosody and sound-effect tags. Non-commercial weights with a "
            "Creator Use Grant (monetized audiobooks allowed with attribution)."
        ),
        license=(
            "Boson Higgs TTS 3 Research and Non-Commercial License (includes the Section II-A "
            "Creator Use Grant)"
        ),
        commercial_use=False,
        homepage="https://huggingface.co/bosonai/higgs-tts-3-4b",
        default_model=_DEFAULT_MODEL,
        models=(_DEFAULT_MODEL, _MODEL_ALIAS),
        native_sample_rate=24000,
        min_vram_gb=11.0,
        languages=_LANGUAGES,
        supports_voice_clone=True,
        transcript="optional",
        supports_instruct=False,
        supports_batch=True,
        supports_speed=False,
        supports_seed=True,
        supports_voice_preset=True,
        preset_voices=(),
        # Upstream's voice-cloning recipe (model card, SGLang-Omni cookbook):
        # temperature 0.8, top_k 50 and no nucleus filter (top_p null = 1.0).
        recommended_settings={"temperature": _RECOMMENDED_TEMPERATURE, "top_p": 1.0, "top_k": 50},
        options=(
            ProviderOption(
                key="emotion", label="Emotion", kind="choice", default=_NONE_CHOICE,
                choices=(_NONE_CHOICE,) + _EMOTIONS,
                help="Emotion token placed at the start of every chunk.",
            ),
            ProviderOption(
                key="style", label="Speaking style", kind="choice", default=_NONE_CHOICE,
                choices=(_NONE_CHOICE,) + _STYLES,
                help="Style token placed at the start of every chunk.",
            ),
            ProviderOption(
                key="prosody_speed", label="Pace", kind="choice", default=_NONE_CHOICE,
                choices=(_NONE_CHOICE,) + _SPEEDS,
                help="Native pace token: very_slow 0.65x, slow 0.85x, fast 1.2x, very_fast 1.4x.",
            ),
            ProviderOption(
                key="prosody_pitch", label="Pitch", kind="choice", default=_NONE_CHOICE,
                choices=(_NONE_CHOICE,) + _PITCHES,
                help="Pitch token: low is about -3 semitones, high about +2.5.",
            ),
            ProviderOption(
                key="expressiveness", label="Expressiveness", kind="choice", default=_NONE_CHOICE,
                choices=(_NONE_CHOICE,) + _EXPRESSIVENESS,
                help="high gives a livelier delivery, low a flatter one.",
            ),
            ProviderOption(
                key="control_tags", label="Extra control tags", kind="str", default="",
                help=(
                    "Raw <|category:value|> tags added to the start of every chunk, e.g. "
                    "<|emotion:contemplation|><|prosody:expressive_low|>."
                ),
            ),
            ProviderOption(
                key="use_reference_transcript", label="Use reference transcript", kind="bool", default=True,
                help="Condition on the reference clip's transcript; it materially improves cloning.",
            ),
            ProviderOption(
                key="max_new_tokens", label="Max audio frames per chunk", kind="int",
                default=_DEFAULT_MAX_NEW_TOKENS, minimum=128, maximum=6000, step=64,
                help="Hard cap on generation steps (25 frames = 1 second of audio).",
            ),
            ProviderOption(
                key="duration_margin", label="Duration margin", kind="float",
                default=_DEFAULT_DURATION_MARGIN, minimum=1.2, maximum=5.0, step=0.1,
                help="Frame budget per chunk as a multiple of its expected spoken duration.",
            ),
            ProviderOption(
                key="max_batch_size", label="Max chunks per forward pass", kind="int",
                default=_DEFAULT_MAX_BATCH, minimum=1, maximum=32, step=1,
                help="Larger batches are split; lower it if the GPU runs out of memory.",
            ),
            ProviderOption(
                key="dtype", label="Backbone precision", kind="choice", default="auto",
                choices=("auto", "float16", "bfloat16", "float32"),
                help="auto: bfloat16 on GPUs with native support, float16 elsewhere (T4).",
            ),
        ),
        pip_requirements=(
            "transformers>=5.5.0,<6",
            "accelerate>=1.1.0",
            "safetensors>=0.4.3",
            "huggingface-hub",
            "tokenizers",
            "torchaudio",
            "soundfile",
        ),
        install_notes=(
            "Needs transformers >= 5.5.0 (HiggsAudioV2TokenizerModel), which conflicts with "
            "qwen-tts (pins transformers==4.57.3): install Higgs in its own environment. "
            "torchaudio must match the installed torch build. The repository is not gated, so no "
            "token or click-through is needed, but downloading the 9.3 GB weights means accepting "
            "the Boson Higgs TTS 3 Research and Non-Commercial License. Its Creator Use Grant "
            "lets you publish and monetize audiobooks only if you credit Boson AI's Higgs Audio "
            "in the audio or prominently in the accompanying text, e.g. \"This audio was created "
            "with Boson AI's Higgs Audio — https://www.boson.ai/higgs-audio\". Clone only voices "
            "you have explicit consent for. About 9 GB of VRAM for the weights (float16) plus "
            "1-3 GB while synthesizing; upstream only validated 40 GB GPUs."
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
        self._dtype_fallback: str | None = None
        self._lock = threading.RLock()

        self._model: Any = None            # transformers Qwen3Model
        self._codec: Any = None            # transformers HiggsAudioV2TokenizerModel
        self._tokenizer: Any = None        # transformers PreTrainedTokenizerFast
        self._audio_weight: Any = None     # fused [codebooks * vocab, hidden] embedding
        self._head_weight: Any = None      # output head (the same tensor when tied)
        self._special_ids: dict[str, int | None] = {}
        self._known_tags: frozenset[str] = frozenset()
        self._loaded_signature: tuple[str, str] | None = None
        self._num_codebooks: int = 8
        self._codebook_vocab: int = 1026
        self._sample_rate: int = 24000
        self._frame_rate: float = 25.0

        self._references: dict[str, _Reference] = {}
        self._file_digests: dict[tuple[str, int, int], str] = {}
        self._warned: set[str] = set()

    # ── Identity ──────────────────────────────────────────────────────────────

    @property
    def device(self) -> str:
        """The torch device string this provider is bound to."""
        return self._device

    @property
    def is_ready(self) -> bool:
        """True once the backbone, the codec and the tokenizer are loaded."""
        return self._model is not None and self._codec is not None and self._tokenizer is not None

    def ensure_ready(self) -> None:
        """Loads the model on this instance's device if it is not loaded yet.

        Raises
        ------
        RuntimeError
            If a dependency is missing or too old (the message names the
            install command), if the device plainly lacks free VRAM, or if the
            checkpoint cannot be loaded.
        """
        self._ensure_initialised()

    # ── Loading ───────────────────────────────────────────────────────────────

    def _ensure_initialised(self) -> None:
        """Loads the model, or reloads it when the model id / precision changed."""
        with self._lock:
            signature = (self.resolve_model_id(), self._requested_dtype_name())
            if self.is_ready and self._loaded_signature == signature:
                return
            if self._loaded_signature is not None and self._loaded_signature != signature:
                logger.info(
                    "[Higgs] Settings changed %s -> %s on %s; reloading.",
                    self._loaded_signature, signature, self._device,
                )
            self._unload()
            self._load(signature)

    def _warn_once(self, key: str, message: str, *args: Any) -> None:
        """Logs a warning the first time *key* is seen on this instance."""
        if key in self._warned:
            return
        self._warned.add(key)
        logger.warning(message, *args)

    def _is_cuda(self) -> bool:
        return self._device.startswith("cuda")

    def _cuda_index(self) -> int:
        return int(self._device.split(":")[1]) if ":" in self._device else 0

    def _has_native_bf16(self) -> bool:
        """True when the GPU computes bfloat16 natively (not through emulation)."""
        try:
            import torch
            return int(torch.cuda.get_device_capability(self._cuda_index())[0]) >= _NATIVE_BF16_CAPABILITY
        except Exception as exc:
            logger.debug("[Higgs] Could not read the compute capability of %s: %s", self._device, exc)
            return False

    def _requested_dtype_name(self) -> str:
        """Resolves the backbone precision from the option, the override and the GPU."""
        if not self._is_cuda():
            return "float32"
        if self._dtype_fallback:
            return self._dtype_fallback
        aliases = {"fp16": "float16", "half": "float16", "bf16": "bfloat16", "fp32": "float32", "float": "float32"}
        choice = str(self.option("dtype", "auto") or "auto").strip().lower()
        choice = aliases.get(choice, choice)
        if choice in ("float16", "bfloat16", "float32"):
            return choice
        override = str(self._dtype_override or "").strip().lower()
        override = aliases.get(override, override)
        if override in ("float16", "float32"):
            return override
        # The pool recommends bfloat16 whenever torch can emulate it; on a GPU
        # without native bf16 (T4) emulation is several times slower than fp16.
        return "bfloat16" if self._has_native_bf16() else "float16"

    def _check_dependencies(self) -> Any:
        """Imports the upstream packages, or explains exactly how to install them.

        Returns
        -------
        module
            The imported ``transformers`` module.
        """
        try:
            import torch  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "Higgs Audio v3 needs PyTorch, which is not installed. Install torch for your "
                f"CUDA version, then run: {_INSTALL_COMMAND}"
            ) from exc
        try:
            import transformers
        except ImportError as exc:
            raise RuntimeError(
                f"Higgs Audio v3 needs the transformers package. Install it with: {_INSTALL_COMMAND}"
            ) from exc
        found = str(getattr(transformers, "__version__", "0"))
        minimum = ".".join(str(part) for part in _MIN_TRANSFORMERS)
        if _version_tuple(found) < _MIN_TRANSFORMERS:
            raise RuntimeError(
                f"Higgs Audio v3 needs transformers >= {minimum} (found {found}): the codec class "
                f"HiggsAudioV2TokenizerModel does not exist in older releases. Install it with: "
                f"{_INSTALL_COMMAND} — note that qwen-tts pins transformers==4.57.3, so Higgs "
                "needs its own Python environment."
            )
        missing = [
            name for name in ("HiggsAudioV2TokenizerModel", "HiggsAudioV2TokenizerConfig", "AutoModel",
                              "CONFIG_MAPPING", "PreTrainedTokenizerFast")
            if not hasattr(transformers, name)
        ]
        if missing:
            raise RuntimeError(
                f"The installed transformers {found} lacks {', '.join(missing)}, which Higgs Audio v3 "
                f"needs. Reinstall it with: {_INSTALL_COMMAND}"
            )
        for module_name, package in (
            ("torchaudio", "torchaudio"),
            ("accelerate", "accelerate"),
            ("safetensors", "safetensors"),
            ("huggingface_hub", "huggingface-hub"),
            ("tokenizers", "tokenizers"),
        ):
            try:
                __import__(module_name)
            except Exception as exc:
                raise RuntimeError(
                    f"Higgs Audio v3 needs the {package} package, which failed to import ({exc}). "
                    f"Install it with: {_INSTALL_COMMAND}"
                    + (" — torchaudio must be the build that matches your torch version."
                       if module_name == "torchaudio" else "")
                ) from exc
        return transformers

    def _check_free_vram(self, dtype_name: str) -> None:
        """Refuses to start a load that cannot fit, instead of dying in CUDA OOM.

        Raises
        ------
        RuntimeError
            If the device has less free memory than the weights need.
        """
        if not self._is_cuda():
            return
        import torch

        try:
            torch.cuda.empty_cache()
            free_bytes, total_bytes = torch.cuda.mem_get_info(self._cuda_index())
        except Exception as exc:
            logger.debug("[Higgs] Could not read free VRAM on %s: %s", self._device, exc)
            return
        bytes_per_parameter = 4 if dtype_name == "float32" else 2
        weights_gb = (_BACKBONE_PARAMETERS * bytes_per_parameter + _CODEC_PARAMETERS * 4) / _BYTES_PER_GB
        needed_gb = weights_gb + _WORKING_SET_GB
        free_gb = free_bytes / _BYTES_PER_GB
        if free_gb < needed_gb:
            raise RuntimeError(
                f"Higgs Audio v3 does not fit on {self._device}: it needs about {needed_gb:.1f} GB of "
                f"free VRAM in {dtype_name} ({weights_gb:.1f} GB of weights plus working memory) but "
                f"only {free_gb:.1f} of {total_bytes / _BYTES_PER_GB:.1f} GB is free. Free the GPU "
                "(unload other models or providers, restart the kernel) or use a GPU with at least "
                "12 GB free; a 16 GB T4 is enough when nothing else is loaded. There is no "
                "quantized build of this model."
            )

    def _load(self, signature: tuple[str, str]) -> None:
        """Downloads (if needed) and loads tokenizer, backbone and codec."""
        model_id, dtype_name = signature
        transformers = self._check_dependencies()
        import torch
        from huggingface_hub import snapshot_download

        if str(getattr(self.config, "quantization", "none") or "none").lower() not in ("none", ""):
            self._warn_once(
                "quantization",
                "[Higgs] quantization=%r is ignored: Higgs Audio v3 has no supported quantized path; "
                "loading %s weights.", getattr(self.config, "quantization", ""), dtype_name,
            )
        self.bind_device()
        self._check_free_vram(dtype_name)
        logger.info("[Higgs] Loading %s on %s (%s)...", model_id, self._device, dtype_name)
        try:
            snapshot = snapshot_download(repo_id=model_id, allow_patterns=list(_SNAPSHOT_FILES))
            with open(os.path.join(snapshot, "config.json"), encoding="utf-8") as fh:
                model_config = json.load(fh)
            self._read_model_config(model_config, model_id)
            tokenizer = self._load_tokenizer(transformers, snapshot)
            backbone, audio_weight, head_weight = self._load_backbone(
                transformers, snapshot, model_config, getattr(torch, dtype_name)
            )
            codec = self._load_codec(transformers, snapshot)
        except Exception as exc:
            self._unload()
            out_of_memory = getattr(torch.cuda, "OutOfMemoryError", None)
            if out_of_memory is not None and isinstance(exc, out_of_memory):
                raise RuntimeError(
                    f"Higgs Audio v3 ran out of VRAM while loading on {self._device}. It needs about "
                    f"{self.info().min_vram_gb:.0f} GB free; unload other models or use a larger GPU."
                ) from exc
            if isinstance(exc, RuntimeError) and str(exc).startswith("Higgs Audio v3"):
                raise
            raise RuntimeError(f"Higgs Audio v3 failed to load {model_id} on {self._device}: {exc}") from exc

        self._tokenizer = tokenizer
        self._model = backbone
        self._audio_weight = audio_weight
        self._head_weight = head_weight
        self._codec = codec
        self._loaded_signature = signature
        logger.info(
            "[Higgs] Ready on %s: %d codebooks x %d codes, %d Hz, %.0f frames/s.",
            self._device, self._num_codebooks, self._codebook_vocab, self._sample_rate, self._frame_rate,
        )

    def _read_model_config(self, model_config: dict[str, Any], model_id: str) -> None:
        """Validates ``config.json`` and reads the audio-codebook geometry."""
        model_type = model_config.get("model_type")
        if model_type != _MODEL_TYPE:
            raise RuntimeError(
                f"Higgs Audio v3 cannot load {model_id}: model_type is {model_type!r}, expected {_MODEL_TYPE!r}."
            )
        encoder = model_config.get("audio_encoder_config") or {}
        if encoder.get("encoder_type", "discrete") != "discrete":
            raise RuntimeError(
                f"Higgs Audio v3 cannot load {model_id}: audio encoder type "
                f"{encoder.get('encoder_type')!r} is not the discrete TTS path."
            )
        self._num_codebooks = int(encoder.get("num_codebooks", 8))
        self._codebook_vocab = int(encoder.get("vocab_size", 1026))

    def _load_tokenizer(self, transformers: Any, snapshot: str) -> Any:
        """Loads ``tokenizer.json`` directly, as the reference server does."""
        from tokenizers import Tokenizer

        raw = Tokenizer.from_file(os.path.join(snapshot, "tokenizer.json"))
        tokenizer = transformers.PreTrainedTokenizerFast(tokenizer_object=raw)
        vocab = dict(tokenizer.get_added_vocab())
        missing = [token for token in _REQUIRED_SPECIALS if token not in vocab]
        if missing:
            raise RuntimeError(f"Higgs Audio v3 tokenizer is missing the special tokens {missing}.")
        self._special_ids = {token: int(vocab[token]) for token in _REQUIRED_SPECIALS}
        self._special_ids[_TOKEN_REF_TEXT] = int(vocab[_TOKEN_REF_TEXT]) if _TOKEN_REF_TEXT in vocab else None
        self._known_tags = frozenset(tag for tag in _CONTROL_TAGS if tag in vocab)
        return tokenizer

    def _checkpoint_shards(self, snapshot: str) -> list[str]:
        """Returns the safetensors files that make up the checkpoint."""
        index_path = os.path.join(snapshot, "model.safetensors.index.json")
        names: list[str] = []
        if os.path.isfile(index_path):
            with open(index_path, encoding="utf-8") as fh:
                names = sorted(set((json.load(fh).get("weight_map") or {}).values()))
        if not names:
            names = ["model.safetensors"]
        paths = [os.path.join(snapshot, name) for name in names]
        absent = [path for path in paths if not os.path.isfile(path)]
        if absent:
            raise RuntimeError(f"Higgs Audio v3 checkpoint file(s) missing: {absent}")
        return paths

    def _load_backbone(
        self,
        transformers: Any,
        snapshot: str,
        model_config: dict[str, Any],
        dtype: Any,
    ) -> tuple[Any, Any, Any]:
        """Builds the Qwen3 backbone and streams its weights straight to the device.

        The module is created with meta tensors and the checkpoint tensors are
        read directly onto the target device, so host RAM never holds a copy of
        the 8 GB of weights (notebook hosts have little RAM per GPU).

        Returns
        -------
        tuple[Module, Tensor, Tensor]
            ``(backbone, audio_embedding_weight, audio_head_weight)``.
        """
        from accelerate import init_empty_weights
        from safetensors import safe_open

        text_config = dict(model_config.get("text_config") or {})
        backbone_type = text_config.get("model_type", "qwen3")
        if backbone_type not in transformers.CONFIG_MAPPING:
            raise RuntimeError(f"Higgs Audio v3: transformers has no {backbone_type!r} backbone.")
        if backbone_type == "qwen3":
            # Qwen3 was trained with a RoPE base of 1e6; transformers defaults to 1e4.
            rope = dict(text_config.get("rope_parameters") or {})
            if not rope.get("rope_theta") and not text_config.get("rope_theta"):
                rope.update({"rope_theta": _QWEN3_ROPE_THETA, "rope_type": rope.get("rope_type", "default")})
                text_config["rope_parameters"] = rope
        backbone_config = transformers.CONFIG_MAPPING[backbone_type](**text_config)
        with init_empty_weights():
            backbone = transformers.AutoModel.from_config(backbone_config)

        encoder = model_config.get("audio_encoder_config") or {}
        tied = bool(encoder.get("tie_word_embeddings", True))
        state: dict[str, Any] = {}
        audio_weight: Any = None
        head_weight: Any = None
        for shard in self._checkpoint_shards(snapshot):
            with safe_open(shard, framework="pt", device=self._device) as reader:
                for key in reader.keys():
                    if key == _AUDIO_EMBEDDING_KEY:
                        audio_weight = reader.get_tensor(key).to(dtype)
                    elif key == _AUDIO_HEAD_KEY and not tied:
                        head_weight = reader.get_tensor(key).to(dtype)
                    elif key.startswith(_SKIPPED_PREFIXES):
                        continue
                    else:
                        for prefix, replacement in _BACKBONE_KEY_MAP:
                            if key.startswith(prefix):
                                state[replacement + key[len(prefix):]] = reader.get_tensor(key).to(dtype)
                                break
        if audio_weight is None:
            raise RuntimeError(f"Higgs Audio v3 checkpoint has no {_AUDIO_EMBEDDING_KEY}.")
        expected_rows = self._num_codebooks * self._codebook_vocab
        if int(audio_weight.shape[0]) != expected_rows:
            raise RuntimeError(
                f"Higgs Audio v3 audio embedding has {int(audio_weight.shape[0])} rows, expected "
                f"{self._num_codebooks} x {self._codebook_vocab} = {expected_rows}."
            )
        if head_weight is None:
            head_weight = audio_weight
        backbone.load_state_dict(state, strict=True, assign=True)
        del state
        backbone.to(self._device)
        backbone.eval()
        backbone.requires_grad_(False)
        if any(getattr(parameter, "is_meta", False) for parameter in backbone.parameters()):
            raise RuntimeError("Higgs Audio v3 backbone still has unloaded (meta) parameters.")
        return backbone, audio_weight, head_weight

    def _load_codec(self, transformers: Any, snapshot: str) -> Any:
        """Loads the audio codec from the weights bundled in the TTS checkpoint.

        The codec stays in float32: its transposed convolutions are unstable in
        half precision (upstream note in ``audio_codec.py``).
        """
        import torch
        from accelerate import init_empty_weights
        from safetensors import safe_open

        codec_config = transformers.HiggsAudioV2TokenizerConfig.from_dict(copy.deepcopy(_CODEC_CONFIG))
        with init_empty_weights():
            codec = transformers.HiggsAudioV2TokenizerModel(codec_config)
        state: dict[str, Any] = {}
        for shard in self._checkpoint_shards(snapshot):
            with safe_open(shard, framework="pt", device=self._device) as reader:
                for key in reader.keys():
                    if key.startswith(_CODEC_PREFIX) and key[len(_CODEC_PREFIX):] not in _CODEC_IGNORED_KEYS:
                        state[key[len(_CODEC_PREFIX):]] = reader.get_tensor(key).to(torch.float32)
        if not state:
            raise RuntimeError(
                f"Higgs Audio v3 checkpoint bundles no audio codec (no {_CODEC_PREFIX}* tensors)."
            )
        result = codec.load_state_dict(state, strict=False, assign=True)
        del state
        missing = list(getattr(result, "missing_keys", []) or [])
        if missing:
            raise RuntimeError(
                f"Higgs Audio v3 codec is missing {len(missing)} weights (e.g. {missing[:3]}); the "
                f"installed transformers does not match the checkpoint. Reinstall with: {_INSTALL_COMMAND}"
            )
        unexpected = list(getattr(result, "unexpected_keys", []) or [])
        if unexpected:
            logger.debug("[Higgs] Codec ignored %d checkpoint tensors, e.g. %s", len(unexpected), unexpected[:3])
        codec.to(self._device)
        codec.eval()
        codec.requires_grad_(False)
        self._sample_rate = int(getattr(codec_config, "sample_rate", 24000))
        hop_length = int(getattr(codec_config, "hop_length", 960) or 960)
        self._frame_rate = self._sample_rate / float(hop_length)
        quantizers = int(getattr(codec_config, "num_quantizers", self._num_codebooks))
        if quantizers != self._num_codebooks:
            raise RuntimeError(
                f"Higgs Audio v3 codec has {quantizers} codebooks but the model expects {self._num_codebooks}."
            )
        return codec

    def _unload(self) -> None:
        """Drops the model objects and returns their VRAM to the driver."""
        had_model = self._model is not None or self._codec is not None
        self._model = None
        self._codec = None
        self._tokenizer = None
        self._audio_weight = None
        self._head_weight = None
        self._loaded_signature = None
        if had_model:
            self._release_cuda_memory()

    def _release_cuda_memory(self) -> None:
        """Collects garbage and empties torch's CUDA cache."""
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as exc:
            logger.debug("[Higgs] Could not empty the CUDA cache: %s", exc)

    def cleanup(self) -> None:
        """Releases the backbone, the codec, cached references and their VRAM."""
        with self._lock:
            self._unload()
            self._references.clear()
            self._file_digests.clear()
            self._release_cuda_memory()

    # ── Settings ──────────────────────────────────────────────────────────────

    def _settings(self) -> _Settings:
        """Reads the generation settings from ``self.config`` (call time)."""
        config = self.config
        temperature = max(0.0, float(getattr(config, "temperature", _RECOMMENDED_TEMPERATURE)))
        top_k_raw = int(getattr(config, "top_k", 0) or 0)
        top_p_raw = float(getattr(config, "top_p", 1.0) or 1.0)
        seed_raw = getattr(config, "seed", -1)
        if _GREEDY_TEMPERATURE < temperature < 0.5:
            self._warn_once(
                "temperature",
                "[Higgs] temperature=%.2f is far below upstream's voice-cloning recipe "
                "(temperature 0.8, top_k 50); expect a flatter delivery.", temperature,
            )
        return _Settings(
            temperature=temperature,
            top_k=top_k_raw if 0 < top_k_raw < self._codebook_vocab else None,
            top_p=top_p_raw if 0.0 < top_p_raw < 1.0 else None,
            seed=int(seed_raw) if seed_raw is not None else -1,
            max_new_tokens=max(_MIN_FRAME_BUDGET, int(self.option("max_new_tokens", _DEFAULT_MAX_NEW_TOKENS))),
            duration_margin=float(self.option("duration_margin", _DEFAULT_DURATION_MARGIN)),
            max_batch_size=max(1, int(self.option("max_batch_size", _DEFAULT_MAX_BATCH))),
            prefix_tags=self._prefix_tags(),
            use_transcript=bool(self.option("use_reference_transcript", True)),
        )

    def _prefix_tags(self) -> tuple[str, ...]:
        """Collects the control tags that open every chunk."""
        tags: list[str] = []
        for key, template in (
            ("emotion", "<|emotion:{}|>"),
            ("style", "<|style:{}|>"),
            ("prosody_speed", "<|prosody:speed_{}|>"),
            ("prosody_pitch", "<|prosody:pitch_{}|>"),
            ("expressiveness", "<|prosody:expressive_{}|>"),
        ):
            value = str(self.option(key, _NONE_CHOICE) or _NONE_CHOICE).strip().lower()
            if value and value != _NONE_CHOICE:
                tags.append(template.format(value))
        instruct = str(getattr(self.config, "tts_instruct", "") or "")
        tags.extend(_TAG_PATTERN.findall(str(self.option("control_tags", "") or "")))
        tags.extend(_TAG_PATTERN.findall(instruct))
        if _plain_text(instruct):
            self._warn_once(
                "instruct:" + instruct,
                "[Higgs] tts_instruct text is ignored: Higgs Audio v3 has no natural-language style "
                "prompt. Use <|emotion:..|> / <|style:..|> / <|prosody:..|> tags or the provider options.",
            )
        allowed = self._known_tags or _CONTROL_TAGS
        kept: list[str] = []
        for tag in tags:
            if tag not in allowed:
                self._warn_once("tag:" + tag, "[Higgs] Unknown control tag %s dropped.", tag)
            elif tag not in kept:
                kept.append(tag)
        return tuple(kept)

    # ── Text ──────────────────────────────────────────────────────────────────

    def _encode_text(self, text: str) -> list[int]:
        """Tokenizes *text*; control tags map to their single special token."""
        return list(self._tokenizer.encode(text, add_special_tokens=False))

    def _prepare_text(self, text: str, prefix_tags: tuple[str, ...]) -> str:
        """Sanitizes a chunk and puts the delivery tags in front of it.

        Raises
        ------
        RuntimeError
            If the chunk contains nothing speakable.
        """
        allowed = self._known_tags or _CONTROL_TAGS
        cleaned = _sanitize_text(text, allowed)
        if not any(char.isalnum() for char in _plain_text(cleaned)):
            raise RuntimeError(f"Higgs Audio v3 cannot synthesize a chunk with no speakable text: {text!r}")
        present = {_tag_slot(tag) for tag in _TAG_PATTERN.findall(cleaned)}
        prefix = "".join(tag for tag in prefix_tags if _tag_slot(tag) not in present)
        return prefix + cleaned

    def _prepare_jobs(
        self,
        texts: list[str],
        reference: _Reference | None,
        settings: _Settings,
    ) -> list[_Job]:
        """Builds the prompt and the frame budget of every chunk."""
        ref_rows = int(reference.delayed_codes.shape[0]) if reference is not None else 0
        ref_text_ids: list[int] = []
        if reference is not None and settings.use_transcript and reference.transcript:
            ref_text_ids = self._encode_text(reference.transcript)
        jobs: list[_Job] = []
        for text in texts:
            prepared = self._prepare_text(text, settings.prefix_tags)
            prompt_ids = _build_prompt_ids(self._special_ids, self._encode_text(prepared), ref_rows, ref_text_ids)
            budget = _frame_budget(
                prepared, self._frame_rate, self._num_codebooks, settings.duration_margin, settings.max_new_tokens
            )
            room = _CONTEXT_LENGTH - len(prompt_ids)
            if room < min(budget, _MIN_FRAME_BUDGET):
                raise RuntimeError(
                    f"Higgs Audio v3 prompt is {len(prompt_ids)} tokens, which leaves no room to "
                    f"generate within its {_CONTEXT_LENGTH}-token context. Use a shorter reference "
                    "clip (10-30 s) or a smaller max_len."
                )
            jobs.append(_Job(text=prepared, prompt_ids=prompt_ids, budget=min(budget, room)))
        return jobs

    # ── Reference voice ───────────────────────────────────────────────────────

    def _remember(self, reference: _Reference) -> _Reference:
        """Stores *reference* in the small per-instance cache."""
        self._references.pop(reference.key, None)
        self._references[reference.key] = reference
        while len(self._references) > _REFERENCE_CACHE_SIZE:
            self._references.pop(next(iter(self._references)))
        return reference

    def _file_digest(self, path: str) -> str:
        """SHA-256 of a file's content, memoized on (path, mtime, size)."""
        stat = os.stat(path)
        memo_key = (os.path.abspath(path), int(stat.st_mtime_ns), int(stat.st_size))
        digest = self._file_digests.get(memo_key)
        if digest is None:
            hasher = hashlib.sha256()
            with open(path, "rb") as fh:
                for block in iter(lambda: fh.read(1 << 20), b""):
                    hasher.update(block)
            digest = hasher.hexdigest()
            self._file_digests[memo_key] = digest
        return digest

    def _transcript_for(self, path: str | None) -> str:
        """Transcript of the reference clip, stripped of any ``<|...|>`` token."""
        transcript = self.reference_transcript(path)
        configured = (getattr(self.config, "voice_file", "") or "").strip()
        if not transcript and configured and configured != path:
            transcript = self.reference_transcript(configured)
        return _plain_text(transcript)

    def _resolve_reference(self, voice_ref: str | bytes | None, texts: list[str], settings: _Settings) -> _Reference:
        """Returns the reference every chunk of this call is conditioned on."""
        preset = (getattr(self.config, "voice_preset", "") or "").strip()
        if preset:
            if not os.path.isfile(preset):
                raise RuntimeError(f"Higgs Audio v3 voice preset not found: {preset}")
            if preset.lower().endswith(_PRESET_SUFFIX):
                return self._preset_reference(preset)
            return self._clip_reference(preset, None, settings)
        if isinstance(voice_ref, (bytes, bytearray)):
            if 0 < len(voice_ref) < _MIN_VOICE_BYTES:
                raise RuntimeError(
                    f"Higgs Audio v3 voice reference is only {len(voice_ref)} bytes; the audio is empty or corrupted."
                )
            if voice_ref:
                return self._clip_reference(None, bytes(voice_ref), settings)
        path = self.resolve_voice_path(voice_ref if voice_ref else None)
        if path:
            if not os.path.isfile(path):
                raise RuntimeError(f"Higgs Audio v3 voice reference not found: {path}")
            return self._clip_reference(path, None, settings)
        return self._smart_voice_reference(texts, settings)

    def _clip_reference(
        self,
        path: str | None,
        data: bytes | None,
        settings: _Settings,
        transcript: str | None = None,
    ) -> _Reference:
        """Encodes a reference clip once per (content, transcript) and caches it.

        An explicit *transcript* replaces the configured / sidecar one.
        """
        digest = hashlib.sha256(data).hexdigest() if data is not None else self._file_digest(str(path))
        if transcript is not None:
            transcript = _plain_text(transcript)
        else:
            transcript = self._transcript_for(path) if settings.use_transcript else ""
        key = "|".join((
            self.resolve_model_id(), digest, hashlib.sha256(transcript.encode("utf-8")).hexdigest()[:16],
        ))
        cached = self._references.get(key)
        if cached is not None:
            return cached
        clip_path = path or self.resolve_voice_path(data)
        codes, seconds = self._encode_clip(str(clip_path))
        if not transcript and settings.use_transcript:
            self._warn_once(
                "transcript:" + digest,
                "[Higgs] No transcript for the reference clip; cloning is noticeably better with one "
                "(set voice_transcript or put a .txt next to the clip).",
            )
        logger.info(
            "[Higgs] Encoded %.1f s reference (%d frames, transcript %s) on %s.",
            seconds, int(codes.shape[0]), "yes" if transcript else "no", self._device,
        )
        return self._remember(_Reference(
            key=key,
            delayed_codes=_apply_delay_pattern(codes, self._codebook_vocab),
            transcript=transcript,
            seconds=seconds,
            source=str(clip_path),
        ))

    def _encode_clip(self, path: str) -> tuple[Any, float]:
        """Reads a clip and encodes it into ``[frames, codebooks]`` codec codes.

        Returns
        -------
        tuple[Tensor, float]
            ``(codes on the CPU, duration in seconds)``.
        """
        import soundfile as sf
        import torch
        import torch.nn.functional as F
        import torchaudio

        try:
            samples, sample_rate = sf.read(path, dtype="float32", always_2d=True)
        except Exception as exc:
            raise RuntimeError(f"Higgs Audio v3 could not read the reference clip {path}: {exc}") from exc
        mono = self.to_mono_float32(samples)
        if mono.size == 0 or sample_rate <= 0:
            raise RuntimeError(f"Higgs Audio v3 reference clip {path} contains no audio.")
        seconds = mono.size / float(sample_rate)
        if seconds > _MAX_REFERENCE_SECONDS:
            raise RuntimeError(
                f"Higgs Audio v3 reference clip is {seconds:.0f} s long; the model accepts at most "
                f"{_MAX_REFERENCE_SECONDS:.0f} s. Trim it to a clean 10-30 s passage (and its transcript)."
            )
        if seconds > _LONG_REFERENCE_SECONDS:
            self._warn_once(
                "long-reference",
                "[Higgs] The reference clip is %.0f s; every chunk re-reads it (25 tokens per second). "
                "10-30 s is enough and faster.", seconds,
            )
        if seconds < _MIN_REFERENCE_SECONDS:
            self._warn_once(
                "short-reference",
                "[Higgs] The reference clip is only %.1f s; a few seconds of speech clone far better.", seconds,
            )
        waveform = torch.from_numpy(mono).view(1, 1, -1)
        if int(sample_rate) != self._sample_rate:
            waveform = torchaudio.functional.resample(waveform, int(sample_rate), self._sample_rate)
        if waveform.shape[-1] < self._sample_rate:
            # The codec errors on clips shorter than one second.
            waveform = F.pad(waveform, (0, self._sample_rate - waveform.shape[-1]))
        with torch.inference_mode():
            codes = self._codec.encode(waveform.to(device=self._device, dtype=torch.float32)).audio_codes
        codes = codes[0].transpose(0, 1).to(torch.long).cpu()
        self._check_codes(codes, f"reference clip {path}")
        return codes, seconds

    def _check_codes(self, codes: Any, what: str, error: type[Exception] = RuntimeError) -> None:
        """Validates the shape and range of raw ``[frames, codebooks]`` codes."""
        if codes.ndim != 2 or int(codes.shape[1]) != self._num_codebooks or int(codes.shape[0]) == 0:
            raise error(
                f"Higgs Audio v3 {what} has codes of shape {tuple(codes.shape)}; expected [frames, {self._num_codebooks}]."
            )
        if int(codes.min()) < 0 or int(codes.max()) >= self._codebook_vocab - 2:
            raise error(
                f"Higgs Audio v3 {what} has codes outside 0..{self._codebook_vocab - 3}."
            )

    def _preset_reference(self, path: str) -> _Reference:
        """Loads pre-encoded reference codes written by :meth:`save_voice_preset`.

        The file is read with ``torch.load(weights_only=True)``: only tensors,
        strings and numbers are accepted, never pickled objects.

        Raises
        ------
        ValueError
            If the file is not a compatible Higgs Audio v3 preset.
        """
        import torch

        key = "|".join((self.resolve_model_id(), "preset", self._file_digest(path)))
        cached = self._references.get(key)
        if cached is not None:
            return cached
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except Exception as exc:
            raise ValueError(f"Higgs Audio v3 could not read the voice preset {path}: {exc}") from exc
        codes = payload.get("reference_codes") if isinstance(payload, dict) else None
        if not isinstance(payload, dict) or payload.get("format") != _PRESET_FORMAT or not hasattr(codes, "ndim"):
            raise ValueError(
                f"{path} is not a Higgs Audio v3 voice preset. Create one with "
                "HiggsAudioProvider.save_voice_preset(), or point voice_preset at an audio clip."
            )
        codes = codes.to(torch.long)
        self._check_codes(codes, f"voice preset {path}", ValueError)
        return self._remember(_Reference(
            key=key,
            delayed_codes=_apply_delay_pattern(codes, self._codebook_vocab),
            transcript=_plain_text(str(payload.get("reference_text", "") or "")),
            seconds=int(codes.shape[0]) / self._frame_rate,
            source=path,
        ))

    def _write_preset(self, path: str, reference: _Reference) -> None:
        """Atomically writes *reference* as a voice preset file."""
        import torch

        directory = os.path.dirname(os.path.abspath(path))
        os.makedirs(directory, exist_ok=True)
        tmp_path = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"
        torch.save(
            {
                "format": _PRESET_FORMAT,
                "version": 1,
                "model": self.resolve_model_id(),
                "codebooks": self._num_codebooks,
                "codebook_size": self._codebook_vocab,
                "reference_codes": _reverse_delay_pattern(reference.delayed_codes).to(torch.int16),
                "reference_text": reference.transcript,
            },
            tmp_path,
        )
        os.replace(tmp_path, path)

    def _describe_preset(self, path: str, reference: _Reference) -> dict[str, Any]:
        """JSON-safe description of a voice preset."""
        frames = int(reference.delayed_codes.shape[0]) - (self._num_codebooks - 1)
        return {
            "path": path,
            "provider": self.info().name,
            "format": _PRESET_FORMAT,
            "model": self.resolve_model_id(),
            "codebooks": int(self._num_codebooks),
            "frames": frames,
            "seconds": round(frames / self._frame_rate, 3),
            "transcript": reference.transcript,
            "has_transcript": bool(reference.transcript),
        }

    def save_voice_preset(
        self,
        path: str,
        voice_ref: str | bytes | None = None,
        *,
        transcript: str | None = None,
    ) -> dict[str, Any]:
        """Encodes the narrator reference once and saves it as a reusable preset.

        The preset holds the clip's codec codes (upstream's ``reference_codes``)
        and its transcript, so later runs need neither the clip nor the codec
        encoder pass.

        Parameters
        ----------
        path : str
            Destination file; ``.pt`` is appended when missing (the returned
            ``"path"`` is the file actually written).
        voice_ref : str | bytes | None
            Reference clip (path or WAV bytes). Defaults to ``config.voice_file``.
        transcript : str | None
            Transcript of the clip. Defaults to ``config.voice_transcript`` or
            the clip's sidecar ``.txt``.

        Returns
        -------
        dict[str, Any]
            JSON-safe description of the preset; assign ``result["path"]`` to
            ``config.voice_preset``.

        Raises
        ------
        RuntimeError
            If there is no reference clip or the model cannot be loaded.
        """
        if not path.lower().endswith(_PRESET_SUFFIX):
            path += _PRESET_SUFFIX
        self.ensure_ready()
        with self._lock:
            self.bind_device()
            settings = self._settings()
            if isinstance(voice_ref, (bytes, bytearray)) and voice_ref:
                reference = self._clip_reference(None, bytes(voice_ref), settings, transcript)
            else:
                clip = self.resolve_voice_path(voice_ref if voice_ref else None)
                if not clip or not os.path.isfile(clip):
                    raise RuntimeError("Higgs Audio v3 needs a reference clip to save a voice preset.")
                reference = self._clip_reference(clip, None, settings, transcript)
            self._write_preset(path, reference)
            return self._describe_preset(path, reference)

    def load_voice_preset(self, path: str) -> dict[str, Any]:
        """Loads a preset written by :meth:`save_voice_preset` into this instance.

        The codes stay on the CPU until a chunk is synthesized, so this does
        not need the model to be loaded. Set ``config.voice_preset`` to the
        same path to narrate with it.

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
            If the file is missing or is not a compatible Higgs Audio v3 preset.
        """
        with self._lock:
            if not path or not os.path.isfile(path):
                raise ValueError(f"Higgs Audio v3 voice preset not found: {path}")
            return self._describe_preset(path, self._preset_reference(path))

    def _smart_voice_reference(self, texts: list[str], settings: _Settings) -> _Reference:
        """Invents one narrator voice and reuses it for every chunk.

        Without a reference the model picks a new speaker per request. The
        first chunk is synthesized once, its codes become the reference, and
        they are saved in the voice cache so other GPU instances and a resumed
        run load the same voice instead of inventing another.
        """
        identity = "|".join((
            self.resolve_model_id(), str(settings.seed), str(getattr(self.config, "language", "") or ""),
            "".join(settings.prefix_tags), "smart-voice-v1",
        ))
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
        path = os.path.join(tempfile.gettempdir(), _VOICE_CACHE_DIR_NAME, f"higgs_smart_{digest}{_PRESET_SUFFIX}")
        memory_key = "|".join((self.resolve_model_id(), "smart", digest))
        with _SMART_VOICE_LOCK:
            cached = self._references.get(memory_key)
            if cached is not None:
                return cached
            if os.path.isfile(path):
                try:
                    return self._preset_reference(path)
                except (RuntimeError, ValueError) as exc:
                    logger.warning("[Higgs] Ignoring unreadable smart-voice cache %s: %s", path, exc)
            seed_text = next((text for text in texts if _plain_text(text)), "")
            cut = seed_text[:_SMART_VOICE_CHARS]
            if len(seed_text) > _SMART_VOICE_CHARS and " " in cut:
                cut = cut.rsplit(" ", 1)[0]
            job = self._prepare_jobs([cut], None, settings)[0]
            logger.info(
                "[Higgs] No reference clip: inventing a narrator voice on %s and caching it at %s "
                "(delete the file or change the seed for a different voice).", self._device, path,
            )
            delayed = self._generate_one(job, None, settings)
            codes = _reverse_delay_pattern(delayed)
            codes = codes.masked_fill(codes >= self._codebook_vocab - 2, 0)
            reference = _Reference(
                key=memory_key,
                delayed_codes=_apply_delay_pattern(codes, self._codebook_vocab),
                transcript=_plain_text(job.text),
                seconds=int(codes.shape[0]) / self._frame_rate,
                source=path,
            )
            try:
                self._write_preset(path, reference)
            except OSError as exc:
                logger.warning("[Higgs] Could not cache the smart voice at %s: %s", path, exc)
            return self._remember(reference)

    # ── Generation ────────────────────────────────────────────────────────────

    def _row_generators(self, batch: int, seed: int, attempt: int) -> list[Any] | None:
        """One seeded generator per row, so a chunk's draw ignores its batch mates."""
        if seed < 0:
            return None
        import torch

        generators = []
        for _ in range(batch):
            generator = torch.Generator(device=self._device)
            generator.manual_seed(int(seed) + attempt)
            generators.append(generator)
        return generators

    def _generate_group(
        self,
        jobs: list[_Job],
        reference: _Reference | None,
        settings: _Settings,
        attempt: int = 0,
    ) -> list[Any | None]:
        """Generates the delayed code rows of several chunks in one batch.

        Prompts are left-padded to a common length (RoPE is relative, so the
        padded rows compute exactly what they would alone) and decoded with a
        shared KV cache until every row has ended or used up its frame budget.

        Returns
        -------
        list[Tensor | None]
            Per chunk the ``[rows, codebooks]`` delayed codes on the CPU, or
            ``None`` when the chunk hit its budget before the end-of-audio code.
        """
        import torch
        import torch.nn.functional as F

        model = self._model
        weight = self._audio_weight
        head = self._head_weight
        books, vocab = self._num_codebooks, self._codebook_vocab
        device = weight.device
        batch = len(jobs)
        embed_tokens = model.get_input_embeddings()
        offsets = torch.arange(books, device=device) * vocab

        with torch.inference_mode():
            ref_embeds = None
            if reference is not None:
                ref_embeds = F.embedding(reference.delayed_codes.to(device) + offsets, weight).sum(dim=-2)
            longest = max(len(job.prompt_ids) for job in jobs)
            max_steps = max(job.budget for job in jobs)
            inputs = torch.zeros((batch, longest, int(weight.shape[1])), dtype=weight.dtype, device=device)
            mask = torch.ones((batch, longest + max_steps), dtype=torch.long, device=device)
            for row, job in enumerate(jobs):
                ids = torch.tensor(job.prompt_ids, dtype=torch.long, device=device)
                slots = ids == _AUDIO_PLACEHOLDER_ID
                embeds = embed_tokens(torch.where(slots, torch.zeros_like(ids), ids)).to(weight.dtype)
                if bool(slots.any()):
                    if ref_embeds is None or int(slots.sum()) != int(ref_embeds.shape[0]):
                        raise RuntimeError("Higgs Audio v3 prompt and reference codes are out of step.")
                    embeds[slots] = ref_embeds.to(embeds.dtype)
                pad = longest - len(job.prompt_ids)
                inputs[row, pad:] = embeds
                mask[row, :pad] = 0

            delay = torch.zeros(batch, dtype=torch.long, device=device)
            countdown = torch.full((batch,), -1, dtype=torch.long, device=device)
            done = torch.zeros(batch, dtype=torch.bool, device=device)
            budgets = torch.tensor([job.budget for job in jobs], dtype=torch.long, device=device)
            stopped = budgets <= 0
            counts = torch.zeros(batch, dtype=torch.long, device=device)
            rows = torch.zeros((batch, max_steps, books), dtype=torch.long, device=device)
            last_codes = torch.zeros((batch, books), dtype=torch.long, device=device)
            generators = self._row_generators(batch, settings.seed, attempt)

            output = model(inputs_embeds=inputs, attention_mask=mask[:, :longest], use_cache=True)
            past = output.past_key_values
            hidden = output.last_hidden_state[:, -1, :]
            del inputs
            for step in range(max_steps):
                logits = F.linear(hidden, head).view(batch, books, vocab).float()
                if not bool(torch.isfinite(logits).all()):
                    raise _NonFiniteLogitsError(
                        f"Higgs Audio v3 produced NaN/inf logits on {self._device} in {weight.dtype}."
                    )
                codes, delay, countdown, done_now = _sampler_step(
                    logits, delay, countdown, stopped,
                    settings.temperature, settings.top_k, settings.top_p, generators,
                )
                writing = ~stopped
                rows[:, step] = codes
                counts += writing.to(counts.dtype)
                last_codes = torch.where(writing.unsqueeze(1), codes, last_codes)
                done = done | done_now
                stopped = done | (budgets <= step + 1)
                if bool(stopped.all()):
                    break
                step_embeds = F.embedding(last_codes + offsets, weight).sum(dim=-2).unsqueeze(1)
                output = model(
                    inputs_embeds=step_embeds,
                    attention_mask=mask[:, :longest + step + 1],
                    past_key_values=past,
                    use_cache=True,
                )
                past = output.past_key_values
                hidden = output.last_hidden_state[:, -1, :]

            finished = done.tolist()
            lengths = counts.tolist()
            rows_cpu = rows.cpu()
        del past, output, hidden, rows
        return [
            rows_cpu[row, :lengths[row]].clone() if finished[row] else None
            for row in range(batch)
        ]

    def _generate_resilient(
        self,
        jobs: list[_Job],
        reference: _Reference | None,
        settings: _Settings,
        attempt: int = 0,
    ) -> list[Any | None]:
        """Runs one batch, falling back to one chunk at a time on CUDA OOM."""
        import torch

        out_of_memory = getattr(torch.cuda, "OutOfMemoryError", None) or getattr(torch, "OutOfMemoryError", MemoryError)
        failure = ""
        try:
            return self._generate_group(jobs, reference, settings, attempt)
        except out_of_memory as exc:
            failure = str(exc)
        # Outside the except block, so the traceback no longer pins the tensors.
        self._release_cuda_memory()
        if len(jobs) > 1:
            logger.warning(
                "[Higgs] CUDA out of memory for a batch of %d on %s; retrying one chunk at a time.",
                len(jobs), self._device,
            )
            outputs: list[Any | None] = []
            for job in jobs:
                try:
                    outputs.extend(self._generate_group([job], reference, settings, attempt))
                    continue
                except out_of_memory as exc:
                    failure = str(exc)
                self._release_cuda_memory()
                break
            else:
                return outputs
        raise RuntimeError(
            f"Higgs Audio v3 ran out of VRAM on {self._device} even for a single chunk "
            f"({failure.splitlines()[0] if failure else 'CUDA out of memory'}). Use a shorter reference "
            "clip, a smaller max_len, or free the GPU."
        )

    def _generate_one(self, job: _Job, reference: _Reference | None, settings: _Settings) -> Any:
        """Generates one chunk, re-sampling when it runs into its frame budget.

        Raises
        ------
        RuntimeError
            If every attempt failed to reach the end-of-audio code.
        """
        result = self._generate_resilient([job], reference, settings)[0]
        return result if result is not None else self._retry_unfinished(job, reference, settings)

    def _retry_unfinished(self, job: _Job, reference: _Reference | None, settings: _Settings) -> Any:
        """Re-samples a chunk that did not end within its frame budget."""
        for attempt in range(1, _TOKEN_LIMIT_RETRIES + 1):
            logger.warning(
                "[Higgs] Chunk did not finish within %d frames (%.0f s) on %s; re-sampling (%d/%d).",
                job.budget, job.budget / self._frame_rate, self._device, attempt, _TOKEN_LIMIT_RETRIES,
            )
            result = self._generate_resilient([job], reference, settings, attempt)[0]
            if result is not None:
                return result
        raise RuntimeError(
            f"Higgs Audio v3 never reached the end of the audio for a chunk within {job.budget} frames "
            f"({job.budget / self._frame_rate:.0f} s) after {_TOKEN_LIMIT_RETRIES + 1} attempts on "
            f"{self._device}; text starts {_plain_text(job.text)[:60]!r}. Raise the max_new_tokens / "
            "duration_margin options if the chunk is legitimately that long."
        )

    def _decode(self, delayed: Any) -> Any:
        """Turns delayed code rows into a mono float32 waveform on the CPU."""
        import torch

        codes = _reverse_delay_pattern(delayed)
        codes = codes.masked_fill(codes >= self._codebook_vocab - 2, 0)
        with torch.inference_mode():
            audio = self._codec.decode(codes.transpose(0, 1).unsqueeze(0).to(self._device)).audio_values
        return audio.reshape(-1).float().cpu()

    def _synthesize_locked(self, texts: list[str], voice_ref: str | bytes | None) -> list[Any]:
        """Synthesizes *texts* in order; the caller holds ``self._lock``."""
        self.bind_device()
        self.seed_everything()
        settings = self._settings()
        reference = self._resolve_reference(voice_ref, texts, settings)
        jobs = self._prepare_jobs(texts, reference, settings)
        delayed: list[Any] = [None] * len(jobs)
        # Batch chunks of similar length together; a batch runs as long as its longest row.
        order = sorted(range(len(jobs)), key=lambda index: jobs[index].budget)
        for start in range(0, len(order), settings.max_batch_size):
            group = order[start:start + settings.max_batch_size]
            outputs = self._generate_resilient([jobs[index] for index in group], reference, settings)
            for index, output in zip(group, outputs):
                delayed[index] = output if output is not None else self._retry_unfinished(
                    jobs[index], reference, settings
                )
        return [self._decode(rows) for rows in delayed]

    def _synthesize_texts(self, texts: list[str], voice_ref: str | bytes | None) -> list[Any]:
        """Loads the model if needed and synthesizes *texts*, guarding precision."""
        self.ensure_ready()
        with self._lock:
            try:
                return self._synthesize_locked(texts, voice_ref)
            except _NonFiniteLogitsError as exc:
                failure = str(exc)
            precision = self._requested_dtype_name()
            if precision != "float16":
                raise RuntimeError(
                    f"{failure} This is not a float16 overflow (precision is {precision}): the cached "
                    "checkpoint or the torch build is likely broken. Delete the model from the Hugging "
                    "Face cache and reinstall matching torch / transformers builds."
                )
            # float16 overflowed: the checkpoint is bfloat16, which has float32's range.
            logger.warning(
                "[Higgs] %s Reloading in bfloat16 (slower on GPUs without native bf16) and retrying.", failure
            )
            self._dtype_fallback = "bfloat16"
            self._ensure_initialised()
            try:
                return self._synthesize_locked(texts, voice_ref)
            except _NonFiniteLogitsError as exc:
                raise RuntimeError(
                    f"{exc} It also failed in bfloat16; the checkpoint or the torch build is broken."
                ) from exc

    # ── Public synthesis API ──────────────────────────────────────────────────

    def synthesize(
        self,
        text: str,
        voice_ref: str | bytes,
        out_path: str | None = None,
        *,
        return_bytes: bool = False,
    ) -> tuple[str | bytes, float]:
        """Generates speech for one chunk.

        Parameters
        ----------
        text : str
            Chunk text; may contain Higgs control tags such as ``<|prosody:pause|>``.
        voice_ref : str | bytes
            Reference clip path or WAV bytes; empty selects the smart voice.
        out_path : str | None
            Where to write the WAV when ``return_bytes`` is False.
        return_bytes : bool
            Return WAV bytes instead of writing ``out_path``.

        Returns
        -------
        tuple[str | bytes, float]
            ``(out_path or wav_bytes, duration_seconds)`` at 24 kHz.

        Raises
        ------
        RuntimeError
            If the chunk cannot be synthesized.
        """
        waveform = self._synthesize_texts([text], voice_ref)[0]
        return self.finish(waveform, self._sample_rate, out_path, return_bytes)

    def synthesize_batch(
        self,
        texts: list[str],
        voice_ref: bytes,
        *,
        return_bytes: bool = True,
    ) -> list[tuple[bytes | str, float]]:
        """Generates several chunks with batched forward passes, in input order.

        Parameters
        ----------
        texts : list[str]
            Chunk texts, each at most ``config.max_len`` characters.
        voice_ref : bytes
            Reference clip WAV bytes, the same for every chunk.
        return_bytes : bool
            Always True from the pipeline; WAV bytes are returned.

        Returns
        -------
        list[tuple[bytes | str, float]]
            One ``(wav_bytes, duration_seconds)`` per input text.

        Raises
        ------
        RuntimeError
            If any chunk cannot be synthesized.
        """
        if not texts:
            return []
        waveforms = self._synthesize_texts(list(texts), voice_ref)
        return [self.finish(waveform, self._sample_rate, None, True) for waveform in waveforms]

