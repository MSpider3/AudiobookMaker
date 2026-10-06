"""
audiobook_factory/tts_providers/base_tts_provider.py
======================================================
Abstract base class and capability metadata for all TTS backends.

Adding a new TTS provider
-------------------------
1. Create ``tts_providers/my_provider.py``.
2. Subclass :class:`BaseTTSProvider`, set the class attribute ``INFO`` to a
   :class:`ProviderInfo`, and implement ``synthesize`` (and ``synthesize_batch``
   when the model can generate several texts in one forward pass).
3. Register the module in ``tts_providers/registry.py``.

Rules every provider must follow
--------------------------------
* The module must import without the model's own package installed: import
  heavy dependencies (torch, the model library) inside methods, never at
  module top level. ``INFO`` is read by the UI and CLI on machines that have
  no GPU and none of the optional packages.
* One instance is bound to exactly one device (``self.device``). Never use
  ``device_map="auto"`` or pick a GPU yourself — the pool creates one
  instance per GPU and runs them from separate threads.
* Return audio at the model's native sample rate. The pipeline resamples to
  ``config.sample_rate`` when it encodes the chapter.
* Read settings from ``self.config`` at call time, not in ``__init__``: the
  pool reuses provider instances across runs and reassigns ``self.config``.
* A chunk that cannot be synthesized must raise. Never return silence or a
  placeholder waveform.
"""
from __future__ import annotations

import hashlib
import io
import logging
import os
import tempfile
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from audiobook_factory.pipeline import AudiobookConfig

logger = logging.getLogger(__name__)

_VOICE_REF_DIR_NAME: str = "abm_voice_refs"
_VOICE_REF_LOCK = threading.Lock()


@dataclass(frozen=True)
class ProviderOption:
    """One provider-specific setting, rendered generically by the UI and CLI.

    Values live in ``AudiobookConfig.tts_options`` under ``key`` and are read
    with :meth:`BaseTTSProvider.option`.

    Attributes
    ----------
    key : str
        Dictionary key in ``config.tts_options``.
    label : str
        Short human-readable label.
    kind : str
        ``"float"``, ``"int"``, ``"bool"``, ``"str"``, ``"choice"`` or
        ``"file"`` (a path to an audio file, e.g. an emotion reference).
    default : Any
        Value used when the user has not set one.
    minimum, maximum, step : float | None
        Slider bounds for numeric kinds.
    choices : tuple[str, ...]
        Allowed values for ``kind="choice"``.
    help : str
        One-sentence explanation shown next to the control.
    """

    key: str
    label: str
    kind: str = "float"
    default: Any = None
    minimum: float | None = None
    maximum: float | None = None
    step: float | None = None
    choices: tuple[str, ...] = ()
    help: str = ""


@dataclass(frozen=True)
class ProviderInfo:
    """Static description of a TTS provider: what it is and what it can do.

    Attributes
    ----------
    name : str
        Registry key, e.g. ``"qwen"``.
    display_name : str
        Name shown to users.
    description : str
        One or two sentences for the UI.
    license : str
        Licence of the *weights* (the code licence may differ).
    commercial_use : bool
        False when the weights are research / non-commercial only.
    homepage : str
        Upstream repository or model card URL.
    default_model : str
        HuggingFace id (or checkpoint name) loaded when the user chose none.
    models : tuple[str, ...]
        Every model id this provider may load. Acts as an allowlist.
    native_sample_rate : int
        Sample rate of the audio the model produces.
    min_vram_gb : float
        Free VRAM needed to load ``default_model`` and synthesize.
    languages : tuple[str, ...]
        Supported language names; empty means "not restricted / unknown".
    supports_voice_clone : bool
        Clones the narrator from a reference clip.
    transcript : str
        ``"required"``, ``"optional"`` or ``"unused"`` — how the provider uses
        the reference clip's transcript (``config.voice_transcript``).
    supports_instruct : bool
        Honours a natural-language style prompt (``config.tts_instruct``).
    supports_batch : bool
        ``synthesize_batch`` runs one forward pass for several texts.
    supports_speed : bool
        Honours ``config.speed`` natively (otherwise the pipeline time-stretches).
    supports_seed : bool
        Honours ``config.seed``.
    supports_voice_preset : bool
        Implements ``save_voice_preset`` / ``load_voice_preset`` and honours
        ``config.voice_preset``.
    preset_voices : tuple[str, ...]
        Built-in speaker names usable without a reference clip.
    recommended_settings : dict[str, Any]
        Upstream's own operating point for the shared ``AudiobookConfig``
        fields this provider reads (``temperature``, ``top_p``, ``top_k``,
        ``repetition_penalty`` …). The shared defaults were tuned for
        Qwen3-TTS; the UI and CLI apply these instead when the user has not
        chosen a value.
    options : tuple[ProviderOption, ...]
        Extra provider-specific settings.
    pip_requirements : tuple[str, ...]
        ``pip install`` arguments for the provider's own dependencies.
    install_notes : str
        Anything pip alone does not cover (git clone, system packages, gating).
    """

    name: str
    display_name: str
    description: str = ""
    license: str = "unknown"
    commercial_use: bool = True
    homepage: str = ""
    default_model: str = ""
    models: tuple[str, ...] = ()
    native_sample_rate: int = 24000
    min_vram_gb: float = 5.0
    languages: tuple[str, ...] = ()
    supports_voice_clone: bool = True
    transcript: str = "optional"
    supports_instruct: bool = False
    supports_batch: bool = False
    supports_speed: bool = False
    supports_seed: bool = False
    supports_voice_preset: bool = False
    preset_voices: tuple[str, ...] = ()
    recommended_settings: dict = field(default_factory=dict)
    options: tuple[ProviderOption, ...] = field(default_factory=tuple)
    pip_requirements: tuple[str, ...] = ()
    install_notes: str = ""

    def option_defaults(self) -> dict[str, Any]:
        """Returns ``{key: default}`` for every provider-specific option."""
        return {opt.key: opt.default for opt in self.options}


class BaseTTSProvider(ABC):
    """
    Minimal interface every TTS provider must implement.

    Methods
    -------
    synthesize(text, voice_ref, out_path)
        Convert *text* to speech, using *voice_ref* (path to a WAV clone
        reference), and write the result to *out_path* (WAV).

    estimate_cost(total_chars) -> float
        Return an approximate cost in USD for *total_chars* characters.
        Return 0.0 for local/free providers.

    get_name() -> str
        Return the short human-readable provider name (e.g. "Qwen3-TTS").

    cleanup()
        Release GPU / model resources.  Called after all chapters are done.
    """

    INFO: ClassVar[ProviderInfo | None] = None

    def __init__(self, config: "AudiobookConfig") -> None:
        self.config = config

    @classmethod
    def info(cls) -> ProviderInfo:
        """Returns this provider's capability description."""
        if cls.INFO is not None:
            return cls.INFO
        return ProviderInfo(name=cls.__name__.lower(), display_name=cls.__name__)

    @property
    @abstractmethod
    def device(self) -> str:
        """The torch device string this provider is bound to (e.g. 'cuda:0')."""
        ...

    @abstractmethod
    def synthesize(
        self,
        text: str,
        voice_ref: str | bytes,
        out_path: str | None = None,
        *,
        return_bytes: bool = False,
    ) -> tuple[str | bytes, float]:
        """Generate speech for *text* and save WAV to *out_path* or return WAV bytes.

        Returns
        -------
        tuple[str | bytes, float]
            ``(out_path or wav_bytes, duration_seconds)``.
        """

    def synthesize_batch(
        self,
        texts: list[str],
        voice_ref: bytes,
        *,
        return_bytes: bool = True,
    ) -> list[tuple[bytes | str, float]]:
        """Synthesizes several text chunks, in input order.

        The default implementation calls :meth:`synthesize` once per text.
        Providers whose model generates a real batch in one forward pass
        override this and set ``INFO.supports_batch``.

        Args:
            texts: Text strings to synthesize, each <= ``config.max_len`` chars.
            voice_ref: Voice reference WAV bytes. Same for all items.
            return_bytes: Always True from the pipeline; WAV bytes are returned.

        Returns:
            List of (wav_bytes, duration) tuples, one per input text.

        Raises:
            RuntimeError: If any item cannot be synthesized.
        """
        return [self.synthesize(text, voice_ref, return_bytes=True) for text in texts]

    def estimate_cost(self, total_chars: int) -> float:
        """Estimated USD cost for *total_chars* characters. 0.0 = free."""
        return 0.0

    def get_name(self) -> str:
        """Short display name, e.g. 'Qwen3-TTS'."""
        return self.info().display_name

    @property
    def is_ready(self) -> bool:
        """Returns True if the provider is loaded and ready for synthesis.

        Default checks for a non-None _model attribute. Subclasses may
        override for different readiness semantics.
        """
        return getattr(self, "_model", None) is not None

    def ensure_ready(self) -> None:
        """Ensures the provider is fully initialized and ready for synthesis.

        The default implementation calls _ensure_initialised() if it exists.
        Subclasses may override to perform additional readiness checks.

        This method is called by GPUPoolManager during pool warmup to
        guarantee initialization before any synthesis request arrives.

        Thread-safe: implementations must be safe to call from any thread.
        """
        if hasattr(self, "_ensure_initialised"):
            self._ensure_initialised()

    def cleanup(self) -> None:
        """Release resources. Override if the provider holds GPU models."""

    def save_voice_preset(
        self,
        path: str,
        voice_ref: str | bytes | None = None,
        *,
        transcript: str | None = None,
    ) -> dict[str, Any]:
        """Saves the conditioned narrator voice so later runs need no reference clip.

        Providers that support it set ``INFO.supports_voice_preset`` and
        override this and :meth:`load_voice_preset`.

        Args:
            path: Destination file.
            voice_ref: Reference clip (path or WAV bytes); defaults to
                ``config.voice_file``.
            transcript: Transcript of the clip; defaults to
                ``config.voice_transcript``.

        Returns:
            A JSON-safe description of the preset, including ``"path"``.
        """
        raise NotImplementedError(f"{self.get_name()} does not support voice presets.")

    def load_voice_preset(self, path: str) -> dict[str, Any]:
        """Loads a preset written by :meth:`save_voice_preset` onto this device.

        Returns:
            A JSON-safe description of the preset.

        Raises:
            ValueError: If the file is not a compatible preset.
        """
        raise NotImplementedError(f"{self.get_name()} does not support voice presets.")

    # ── Helpers shared by providers ──────────────────────────────────────────

    def option(self, key: str, default: Any = None) -> Any:
        """Reads a provider-specific option from ``config.tts_options``.

        Falls back to the default declared in ``INFO.options``, then to
        *default*. Values are coerced to the declared kind so a string that
        arrived from JSON or a form still behaves as a number or bool.
        """
        declared = next((o for o in self.info().options if o.key == key), None)
        values = getattr(self.config, "tts_options", None) or {}
        if key in values and values[key] is not None and values[key] != "":
            value = values[key]
        elif declared is not None and declared.default is not None:
            value = declared.default
        else:
            return default
        if declared is None:
            return value
        try:
            if declared.kind == "float":
                return float(value)
            if declared.kind == "int":
                return int(value)
            if declared.kind == "bool":
                if isinstance(value, str):
                    return value.strip().lower() in ("1", "true", "yes", "on")
                return bool(value)
        except (TypeError, ValueError):
            return declared.default if declared.default is not None else default
        return value

    def resolve_model_id(self) -> str:
        """Returns the model id to load, restricted to ``INFO.models``.

        ``config.tts_model_name`` is honoured only when this provider lists
        it; anything else (another provider's model left over in the config,
        or an arbitrary repository) falls back to ``INFO.default_model``. This
        is what keeps a crafted config from making a provider that enables
        ``trust_remote_code`` load an unreviewed repository.
        """
        info = self.info()
        requested = (getattr(self.config, "tts_model_name", "") or "").strip()
        if requested and requested in info.models:
            return requested
        return info.default_model

    def resolve_voice_path(self, voice_ref: str | bytes | None) -> str | None:
        """Turns a voice reference (path or WAV bytes) into a file path.

        Bytes are written once to a content-addressed file in the system
        temp directory and reused on later calls, so model libraries that
        cache by path keep hitting their cache.
        """
        voice_ref = voice_ref or getattr(self.config, "voice_file", "") or None
        if not voice_ref:
            return None
        if isinstance(voice_ref, str):
            return voice_ref
        if not isinstance(voice_ref, (bytes, bytearray)):
            raise TypeError(f"voice_ref must be bytes or str, got {type(voice_ref).__name__}")
        digest = hashlib.sha256(voice_ref).hexdigest()[:20]
        directory = os.path.join(tempfile.gettempdir(), _VOICE_REF_DIR_NAME)
        path = os.path.join(directory, f"{digest}.wav")
        with _VOICE_REF_LOCK:
            if not os.path.exists(path) or os.path.getsize(path) != len(voice_ref):
                os.makedirs(directory, exist_ok=True)
                tmp_path = f"{path}.{os.getpid()}.tmp"
                with open(tmp_path, "wb") as fh:
                    fh.write(voice_ref)
                os.replace(tmp_path, path)
        return path

    def reference_transcript(self, voice_path: str | None = None) -> str:
        """Returns the transcript of the reference clip, or ``""``.

        Uses ``config.voice_transcript`` first, then a sidecar text file next
        to the clip (``voice.wav.txt`` or ``voice.txt``).
        """
        configured = (getattr(self.config, "voice_transcript", "") or "").strip()
        if configured:
            return configured
        if not voice_path:
            return ""
        for sidecar in (voice_path + ".txt", os.path.splitext(voice_path)[0] + ".txt"):
            try:
                with open(sidecar, encoding="utf-8", errors="replace") as fh:
                    text = fh.read().strip()
                if text:
                    return text
            except OSError:
                continue
        return ""

    def seed_everything(self) -> None:
        """Seeds torch's RNGs for this provider's device when ``config.seed >= 0``.

        Only this instance's GPU is reseeded: ``manual_seed_all`` would reset
        the generator of the other GPU while its worker is mid-generation.
        """
        seed = getattr(self.config, "seed", -1)
        if seed is None or int(seed) < 0:
            return
        import torch
        torch.manual_seed(int(seed))
        dev = self.device or ""
        if dev.startswith("cuda") and torch.cuda.is_available():
            try:
                index = int(dev.split(":")[1]) if ":" in dev else torch.cuda.current_device()
                with torch.cuda.device(index):
                    torch.cuda.manual_seed(int(seed))
            except Exception as exc:
                logger.debug("Could not seed %s: %s", dev, exc)

    def bind_device(self) -> None:
        """Makes this provider's GPU the calling thread's current CUDA device."""
        dev = self.device
        if dev and dev.startswith("cuda"):
            try:
                import torch
                torch.cuda.set_device(int(dev.split(":")[1]) if ":" in dev else 0)
            except Exception as exc:
                logger.warning("Failed to set CUDA device to %s: %s", dev, exc)

    @staticmethod
    def to_mono_float32(audio: Any) -> Any:
        """Converts a tensor / list / ndarray waveform to a 1-D float32 ndarray."""
        import numpy as np

        if hasattr(audio, "detach"):
            audio = audio.detach().cpu().float().numpy()
        array = np.asarray(audio, dtype=np.float32)
        array = np.squeeze(array)
        if array.ndim > 1:
            # (channels, samples) or (samples, channels) → mono
            axis = 0 if array.shape[0] < array.shape[-1] else -1
            array = array.mean(axis=axis)
        return np.ascontiguousarray(array, dtype=np.float32)

    def finish(
        self,
        audio: Any,
        sample_rate: int,
        out_path: str | None = None,
        return_bytes: bool = False,
    ) -> tuple[str | bytes, float]:
        """Validates a generated waveform and returns it in the synthesize() shape.

        Raises:
            RuntimeError: If the waveform is empty or contains NaN/inf.
        """
        import numpy as np
        import soundfile as sf

        array = self.to_mono_float32(audio)
        if array.size == 0 or sample_rate <= 0:
            raise RuntimeError(f"{self.get_name()} produced no audio on {self.device}.")
        if not np.isfinite(array).all():
            raise RuntimeError(f"{self.get_name()} produced NaN/inf samples on {self.device}.")
        duration = array.size / float(sample_rate)
        if out_path and not return_bytes:
            directory = os.path.dirname(out_path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            sf.write(out_path, array, int(sample_rate))
            return out_path, duration
        buf = io.BytesIO()
        sf.write(buf, array, int(sample_rate), format="WAV")
        return buf.getvalue(), duration

    @staticmethod
    def _validate_voice_ref(voice_ref: object) -> None:
        """Validates voice_ref is bytes or str before synthesis.

        Raises:
            TypeError: If voice_ref is neither bytes nor str.
            ValueError: If voice_ref is empty or non-existent file.
        """
        if not isinstance(voice_ref, (bytes, str)):
            raise TypeError(
                f"voice_ref must be bytes or str, got {type(voice_ref).__name__}. "
                "Pass either WAV file bytes or a path string."
            )
        if isinstance(voice_ref, bytes) and len(voice_ref) < 100:
            raise ValueError(
                f"voice_ref bytes too short ({len(voice_ref)}). "
                "The WAV data appears to be empty or corrupted."
            )
        if isinstance(voice_ref, str) and not os.path.exists(voice_ref):
            raise ValueError(
                f"voice_ref path does not exist: {voice_ref}"
            )

    @classmethod
    def create_for_device(cls, device: str, config: "AudiobookConfig", dtype_override: str | None = None) -> "BaseTTSProvider":
        """Factory classmethod: constructs a provider instance pinned to `device`.

        Args:
            device: Target torch device string, e.g. "cuda:0", "cuda:1", "cpu".
            config: AudiobookConfig options.
            dtype_override: If set, overrides torch_dtype selection.

        Returns:
            An instantiated BaseTTSProvider pinned to the device.
        """
        return cls(config, device=device, dtype_override=dtype_override)  # type: ignore[call-arg]


# ── Factory ───────────────────────────────────────────────────────────────────

def get_tts_provider(
    name: str,
    config: "AudiobookConfig",
    device: str | None = None,
    dtype_override: str | None = None,
) -> BaseTTSProvider:
    """
    Return an instantiated provider for *name*.

    Provider names and aliases are defined in ``tts_providers/registry.py``.

    Raises
    ------
    ValueError
        If *name* is not a registered provider.
    """
    from audiobook_factory.tts_providers.registry import canonical_name, provider_class

    key = canonical_name(name)
    cls = provider_class(key)
    if key == "mock":
        if device is not None:
            return cls.create_for_device(device, config, dtype_override=dtype_override)
        return cls(config, device="cpu")
    if device is not None:
        return cls.create_for_device(device, config, dtype_override=dtype_override)
    return cls(config, device=device, dtype_override=dtype_override)
