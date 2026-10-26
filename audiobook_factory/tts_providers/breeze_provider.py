"""
audiobook_factory/tts_providers/breeze_provider.py
====================================================
Breeze TTS 2 (BreezeBlue/Breeze-TTS-2): zero-shot voice cloning for English
and Chinese.

Upstream: https://github.com/breezeblue-ai/breeze-tts. The code is not a pip
package; clone it and set ``ABM_BREEZE_REPO`` (see
``requirements/tts-breeze.txt``). It needs the Qwen3-TTS environment
(``qwen-tts``, transformers 4.57.3): the bundled codec is Qwen3-TTS's audio
tokenizer.

Generation follows upstream's eager batch path (``breeze_infer/api.py``,
``_generate_batch_pcm``) with classifier-free guidance off, which needs none
of the CUDA-graph modules. The weights are bfloat16, as upstream loads them.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

from audiobook_factory.tts_providers.base_tts_provider import (
    BaseTTSProvider,
    ProviderInfo,
    ProviderOption,
    add_clone_to_path,
    is_out_of_memory,
)

if TYPE_CHECKING:
    from audiobook_factory.pipeline import AudiobookConfig

logger = logging.getLogger(__name__)

_MODEL_ID: str = "BreezeBlue/Breeze-TTS-2"
_REPO_ENV_VAR: str = "ABM_BREEZE_REPO"
_SAMPLE_RATE: int = 24000
_MAX_NEW_TOKENS: int = 1500          # upstream MAX_NEW_TOKENS
_REPETITION_PENALTY: float = 1.1     # upstream REPETITION_PENALTY


class BreezeTTSProvider(BaseTTSProvider):
    """Breeze TTS 2 bound to one device."""

    INFO = ProviderInfo(
        name="breeze",
        display_name="Breeze TTS 2",
        description=(
            "3.5B English / Chinese voice cloning from a clip and its transcript, with optional "
            "style instructions. Research and non-commercial use only."
        ),
        license="BreezeBlue Research and Non-Commercial License v1.1",
        commercial_use=False,
        homepage="https://huggingface.co/BreezeBlue/Breeze-TTS-2",
        default_model=_MODEL_ID,
        models=(_MODEL_ID,),
        native_sample_rate=_SAMPLE_RATE,
        min_vram_gb=9.0,
        languages=("English", "Chinese"),
        supports_voice_clone=True,
        transcript="required",
        supports_instruct=True,
        supports_batch=True,
        supports_seed=True,
        options=(
            ProviderOption(
                key="temperature", label="Temperature", kind="float", default=0.9,
                minimum=0.1, maximum=1.5, step=0.05,
                help="Sampling temperature of the speech tokens (upstream default 0.9).",
            ),
        ),
        pip_requirements=("qwen-tts", "transformers==4.57.3"),
        install_notes=(
            "Clone https://github.com/breezeblue-ai/breeze-tts and set ABM_BREEZE_REPO to it; "
            "uses the Qwen3-TTS environment. Weights (7.7 GB) download on first use. "
            "The weights licence forbids commercial use, including selling the audiobook."
        ),
    )

    def __init__(
        self,
        config: "AudiobookConfig",
        device: str | None = None,
        dtype_override: str | None = None,
    ) -> None:
        super().__init__(config)
        self._device = device or getattr(config, "device", "cuda")
        self._model: Any = None
        self._tokenizer: Any = None
        self._audio_tokenizer: Any = None
        self._upstream: dict[str, Any] = {}
        self._lock = threading.RLock()

    @property
    def device(self) -> str:
        return self._device

    # ── Loading ──────────────────────────────────────────────────────────────

    def _ensure_initialised(self) -> None:
        with self._lock:
            if self._model is not None:
                return
            add_clone_to_path(_REPO_ENV_VAR, "breeze_infer", "Breeze TTS")
            try:
                from breeze_infer.runtime import load_runtime, update_generation_config_for_breeze  # type: ignore
                from breeze_infer.templates import get_template, prepare_inputs, select_template_name  # type: ignore
                from huggingface_hub import snapshot_download
                from models.logits_process import GeneratedTokenRepetitionPenaltyLogitsProcessor  # type: ignore
            except ImportError as exc:
                raise RuntimeError(
                    f"Breeze TTS could not import its code ({exc}). It needs the clone in "
                    f"{_REPO_ENV_VAR} and the Qwen3-TTS environment (qwen-tts, transformers 4.57.3)."
                ) from exc
            self.bind_device()
            logger.info("[Breeze] Loading %s on %s...", _MODEL_ID, self._device)
            checkpoint = Path(snapshot_download(_MODEL_ID))
            self._tokenizer, self._model, self._audio_tokenizer = load_runtime(
                checkpoint, device=self._device, attn_implementation="eager",
            )
            update_generation_config_for_breeze(self._model)
            self._upstream = {
                "prepare_inputs": prepare_inputs,
                "get_template": get_template,
                "select_template_name": select_template_name,
                "penalty": GeneratedTokenRepetitionPenaltyLogitsProcessor,
            }
            logger.info("[Breeze] Ready on %s.", self._device)

    def cleanup(self) -> None:
        with self._lock:
            self._model = self._tokenizer = self._audio_tokenizer = None
            self._upstream = {}
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as exc:
            logger.debug("Could not empty the CUDA cache: %s", exc)

    # ── Synthesis ────────────────────────────────────────────────────────────

    def _requests(self, texts: list[str], voice_path: str) -> list[dict[str, Any]]:
        transcript = self.reference_transcript(voice_path)
        if not transcript:
            raise RuntimeError(
                "Breeze TTS needs the transcript of the reference clip (voice transcript setting)."
            )
        instruction = (getattr(self.config, "tts_instruct", "") or "").strip()
        requests = []
        for index, text in enumerate(texts):
            request = {"id": f"batch-{index}", "text": text, "speaker": "S0",
                       "ref_audio_path": voice_path, "ref_text": transcript}
            if instruction:
                request["instruction"] = instruction
            requests.append(request)
        return requests

    def _generate(self, texts: list[str], voice_path: str) -> list[Any]:
        """One eager batch, as upstream's ``_generate_batch_pcm`` with CFG 1.0."""
        import torch
        from transformers.generation.logits_process import LogitsProcessorList

        up = self._upstream
        requests = self._requests(texts, voice_path)
        inputs = up["prepare_inputs"](
            self._tokenizer, self._audio_tokenizer, self._model, requests,
            up["get_template"](up["select_template_name"](requests[0])),
            guidance_scale=1.0, guidance_scale_ref=None, guidance_scale_ins=None,
        )
        temperature = float(self.option("temperature", 0.9))
        self._model.generation_config.temperature = temperature
        self._model.depth_decoder.generation_config.temperature = temperature
        self.seed_everything()
        with torch.inference_mode():
            audio = self._model.generate(
                **inputs,
                output_audio=True,
                audio_tokenizer=self._audio_tokenizer,
                logits_processor=LogitsProcessorList([up["penalty"](_REPETITION_PENALTY)]),
                max_new_tokens=_MAX_NEW_TOKENS,
            )
        if len(audio) != len(texts):
            raise RuntimeError(f"Breeze TTS returned {len(audio)} clips for {len(texts)} texts.")
        return list(audio)

    def _generate_resilient(self, texts: list[str], voice_path: str) -> list[Any]:
        """Generates *texts*, halving the batch on out-of-memory."""
        try:
            return self._generate(texts, voice_path)
        except Exception as exc:
            if not is_out_of_memory(exc) or len(texts) == 1:
                raise
            failure = exc.with_traceback(None)
        import torch
        torch.cuda.empty_cache()
        middle = (len(texts) + 1) // 2
        self.batch_size_limit = min(self.batch_size_limit or middle, middle)
        logger.warning("[Breeze] Out of memory on %s for %d chunks (%s); retrying as %d + %d.",
                       self._device, len(texts), failure, middle, len(texts) - middle)
        return self._generate_resilient(texts[:middle], voice_path) + self._generate_resilient(texts[middle:], voice_path)

    def synthesize_batch(
        self,
        texts: list[str],
        voice_ref: bytes,
        *,
        return_bytes: bool = True,
    ) -> list[tuple[bytes | str, float]]:
        if not texts:
            return []
        self.ensure_ready()
        voice_path = self.resolve_voice_path(voice_ref)
        if not voice_path:
            raise RuntimeError("Breeze TTS clones a voice: a reference clip is required.")
        with self._lock:
            self.bind_device()
            clips = self._generate_resilient(list(texts), voice_path)
        return [self.finish(clip, _SAMPLE_RATE, return_bytes=True) for clip in clips]

    def synthesize(
        self,
        text: str,
        voice_ref: str | bytes,
        out_path: str | None = None,
        *,
        return_bytes: bool = False,
    ) -> tuple[str | bytes, float]:
        self.ensure_ready()
        voice_path = self.resolve_voice_path(voice_ref)
        if not voice_path:
            raise RuntimeError("Breeze TTS clones a voice: a reference clip is required.")
        with self._lock:
            self.bind_device()
            clip = self._generate_resilient([text], voice_path)[0]
        return self.finish(clip, _SAMPLE_RATE, out_path=out_path, return_bytes=return_bytes or not out_path)
