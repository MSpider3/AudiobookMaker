"""
audiobook_factory/tts_providers/fish_s1_provider.py
=====================================================
Fish Audio S1-mini (fishaudio/s1-mini): 0.5B zero-shot voice cloning in 13
languages, distilled from S1. Smaller and faster than S2 Pro.

S1-mini needs the fish-speech code from before S2 (commit d3df505,
2026-01-08): the S2 rewrite changed the tokenizer and prompt format, so the
``fish`` engine's pinned fish-speech cannot load it. Both are the same
``fish_speech`` package, so this engine runs in its own environment from a
clone named by ``ABM_FISH_S1_REPO`` (see ``requirements/tts-fish_s1.txt``).

The weights repository is gated: accept the licence on Hugging Face and set
``HF_TOKEN``. Licence CC-BY-NC-SA-4.0 (non-commercial).
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
)

if TYPE_CHECKING:
    from audiobook_factory.pipeline import AudiobookConfig

logger = logging.getLogger(__name__)

_MODEL_ID: str = "fishaudio/s1-mini"
_REPO_ENV_VAR: str = "ABM_FISH_S1_REPO"
_CODEC_CONFIG: str = "modded_dac_vq"
# Generated codec frames per UTF-8 byte of text: about three times normal
# speech, so a chunk is never cut short; the model stops at its end token.
_FRAMES_PER_BYTE: int = 4
_FRAME_BUDGET_BASE: int = 64
_MAX_REFERENCES: int = 4


class FishS1Provider(BaseTTSProvider):
    """Fish Audio S1-mini bound to one device."""

    INFO = ProviderInfo(
        name="fish_s1",
        display_name="Fish Audio S1-mini",
        description=(
            "0.5B voice cloning in 13 languages from a clip and its transcript; lighter than S2 Pro. "
            "Gated model, non-commercial use only."
        ),
        license="CC-BY-NC-SA-4.0",
        commercial_use=False,
        homepage="https://huggingface.co/fishaudio/s1-mini",
        default_model=_MODEL_ID,
        models=(_MODEL_ID,),
        native_sample_rate=44100,
        min_vram_gb=4.0,
        languages=("English", "Chinese", "Japanese", "Korean", "German", "French", "Spanish",
                   "Arabic", "Dutch", "Russian", "Italian", "Polish", "Portuguese"),
        supports_voice_clone=True,
        transcript="required",
        supports_batch=False,
        supports_seed=True,
        recommended_settings={"temperature": 0.8, "top_p": 0.8, "repetition_penalty": 1.1},
        options=(
            ProviderOption(
                key="half", label="Half precision (fp16)", kind="bool", default=True,
                help="fp16 weights (upstream --half); off loads bfloat16, which GPUs before Ampere run slowly.",
            ),
        ),
        pip_requirements=("tiktoken", "loralib", "hydra-core>=1.3.2", "pyrootutils", "einx==0.2.2"),
        install_notes=(
            "Follow requirements/tts-fish_s1.txt (two pip steps). "
            "Clone https://github.com/fishaudio/fish-speech at d3df505 and set ABM_FISH_S1_REPO to it, "
            "in an environment without the fish engine. Accept the model licence on Hugging Face and set HF_TOKEN."
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
        self._decode_one_token: Any = None
        self._codec: Any = None
        self._generate_long: Any = None
        self._references: dict[str, Any] = {}
        self._lock = threading.RLock()

    @property
    def device(self) -> str:
        return self._device

    # ── Loading ──────────────────────────────────────────────────────────────

    def _ensure_initialised(self) -> None:
        with self._lock:
            if self._model is not None:
                return
            add_clone_to_path(_REPO_ENV_VAR, ".project-root", "fish-speech (S1)")
            try:
                import torch
                from fish_speech.models.text2semantic import inference  # type: ignore
                from huggingface_hub import snapshot_download
            except ImportError as exc:
                raise RuntimeError(f"Fish S1-mini could not import fish-speech ({exc}).") from exc
            if not hasattr(inference, "generate_long") or not hasattr(inference, "init_model"):
                raise RuntimeError(
                    f"{_REPO_ENV_VAR} must be fish-speech before S2 (commit d3df505); this clone has another API."
                )
            self.bind_device()
            checkpoint = Path(snapshot_download(_MODEL_ID))
            precision = torch.half if bool(self.option("half", True)) else torch.bfloat16
            logger.info("[FishS1] Loading %s on %s (%s)...", _MODEL_ID, self._device, precision)
            model, decode_one_token = inference.init_model(str(checkpoint), self._device, precision, compile=False)
            self._codec = _load_codec(checkpoint / "codec.pth", self._device)
            self._model, self._decode_one_token = model, decode_one_token
            self._generate_long = inference.generate_long
            logger.info("[FishS1] Ready on %s.", self._device)

    def cleanup(self) -> None:
        with self._lock:
            self._model = self._codec = self._decode_one_token = self._generate_long = None
            self._references.clear()
        import gc
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as exc:
            logger.debug("Could not empty the CUDA cache: %s", exc)

    # ── Synthesis ────────────────────────────────────────────────────────────

    def _reference_codes(self, path: str) -> Any:
        """Codec tokens of a reference clip, cached per path."""
        if path in self._references:
            return self._references[path]
        import soundfile as sf
        import torch
        import torchaudio.functional as F

        audio, rate = sf.read(path, dtype="float32", always_2d=True)
        if audio.size == 0:
            raise RuntimeError(f"The reference clip {path} is empty.")
        waveform = torch.from_numpy(audio.mean(axis=1))
        target = int(self._codec.sample_rate)
        if rate != target:
            waveform = F.resample(waveform, int(rate), target)
        audios = waveform[None, None].to(self._device, dtype=next(self._codec.parameters()).dtype)
        lengths = torch.tensor([waveform.shape[0]], device=self._device, dtype=torch.long)
        with torch.inference_mode():
            indices, feature_lengths = self._codec.encode(audios, lengths)
        codes = indices[0, :, : int(feature_lengths[0])].detach().cpu().long()
        while len(self._references) >= _MAX_REFERENCES:
            self._references.pop(next(iter(self._references)))
        self._references[path] = codes
        return codes

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
            raise RuntimeError("Fish S1-mini clones a voice: a reference clip is required.")
        transcript = self.reference_transcript(voice_path)
        if not transcript:
            raise RuntimeError("Fish S1-mini needs the transcript of the reference clip (voice transcript setting).")

        import torch

        with self._lock:
            self.bind_device()
            prompt = self._reference_codes(voice_path)
            self.seed_everything()
            codes = []
            for response in self._generate_long(
                model=self._model,
                device=self._device,
                decode_one_token=self._decode_one_token,
                text=text,
                max_new_tokens=_FRAME_BUDGET_BASE + _FRAMES_PER_BYTE * len(text.encode("utf-8")),
                top_p=float(getattr(self.config, "top_p", 0.8) or 0.8),
                repetition_penalty=float(getattr(self.config, "repetition_penalty", 1.1) or 1.1),
                temperature=float(getattr(self.config, "temperature", 0.8) or 0.8),
                prompt_text=[transcript],
                prompt_tokens=[prompt],
            ):
                if response.action == "sample":
                    codes.append(response.codes)
            if not codes:
                raise RuntimeError("Fish S1-mini generated no speech tokens.")
            indices = torch.cat(codes, dim=1).to(self._device).long()
            lengths = torch.tensor([indices.shape[1]], device=self._device, dtype=torch.long)
            with torch.inference_mode():
                audio, _ = self._codec.decode(indices, lengths)
            waveform = audio[0, 0].float().cpu()
        return self.finish(waveform, int(self._codec.sample_rate), out_path=out_path,
                           return_bytes=return_bytes or not out_path)


def _load_codec(checkpoint_path: Path, device: str) -> Any:
    """Upstream ``fish_speech.models.dac.inference.load_model``.

    Done here because importing that module pulls in ``fish_speech.utils``,
    which needs PyTorch Lightning for training-only helpers.
    """
    import fish_speech  # type: ignore
    import torch
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    if not OmegaConf.has_resolver("eval"):
        OmegaConf.register_new_resolver("eval", eval)
    GlobalHydra.instance().clear()
    config_dir = str(Path(fish_speech.__file__).parent / "configs")
    with initialize_config_dir(version_base="1.3", config_dir=config_dir):
        codec = instantiate(compose(config_name=_CODEC_CONFIG))
    state = torch.load(str(checkpoint_path), map_location=device, mmap=True, weights_only=True)
    state = state.get("state_dict", state)
    if any("generator" in key for key in state):
        state = {key.replace("generator.", ""): value for key, value in state.items() if "generator." in key}
    codec.load_state_dict(state, strict=False, assign=True)
    return codec.eval().to(device)
