"""
audiobook_factory/preflight.py

Pre-flight environment validation for AudiobookMaker.
Runs before any model loading or generation begins.
All checks complete in under 30 seconds total.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

__all__ = ["run_preflight_checks", "PreflightResult", "PreflightError"]


@dataclass
class PreflightResult:
    """Results of all pre-flight environment checks.

    Attributes:
        passed: True if all required checks passed.
        warnings: List of non-fatal issues (generation can continue).
        errors: List of fatal issues (generation must not start).
        device_info: Per-device capability information.
        recommended_dtype: "float16" or "bfloat16" based on hardware.
        python_version: Detected Python version tuple.
        library_versions: Dict of critical library versions.
    """
    passed: bool
    warnings: list[str]
    errors: list[str]
    device_info: list[dict]
    recommended_dtype: str
    python_version: tuple[int, ...]
    library_versions: dict[str, str]


class PreflightError(RuntimeError):
    """Raised when pre-flight checks find fatal environment issues.

    Attributes:
        result: The full PreflightResult with all check details.
    """
    def __init__(self, result: PreflightResult) -> None:
        self.result = result
        error_lines = "\n".join(f"  ✗ {e}" for e in result.errors)
        super().__init__(
            f"Environment pre-flight failed with {len(result.errors)} error(s):\n"
            f"{error_lines}\n"
            "Fix the above issues before starting generation."
        )


def _check_python_version() -> tuple[bool, str]:
    """Checks Python version is 3.10+.

    Returns:
        Tuple of (passed, message).
    """
    import sys
    v = sys.version_info
    if v < (3, 10):
        return False, (
            f"Python {v.major}.{v.minor} detected. "
            "AudiobookMaker requires Python 3.10+."
        )
    return True, f"Python {v.major}.{v.minor}.{v.micro}"


def _check_torch_version() -> tuple[bool, str]:
    """Checks PyTorch is importable and returns version.

    Returns:
        Tuple of (passed, message).
    """
    try:
        import torch
        return True, f"PyTorch {torch.__version__}"
    except ImportError:
        return False, "PyTorch not installed. Run: pip install torch"


def _check_cuda_available() -> tuple[bool, str]:
    """Checks CUDA is available and returns device count.

    Returns:
        Tuple of (passed, message). Passes even with 0 GPUs (CPU fallback).
    """
    try:
        import torch
        count = torch.cuda.device_count()
        if count == 0:
            return True, "No CUDA GPUs detected — will use CPU (very slow)"
        return True, f"{count} CUDA GPU(s) detected"
    except Exception as exc:
        return False, f"CUDA check failed: {exc}"


def _check_bfloat16_support() -> tuple[bool, str, str]:
    """Checks per-device bfloat16 support and recommends dtype.

    Returns:
        Tuple of (passed, message, recommended_dtype).
        recommended_dtype is "bfloat16" or "float16".
    """
    try:
        import torch
        if not torch.cuda.is_available() or torch.cuda.device_count() == 0:
            return True, "CPU: using float32", "float32"

        all_support_bf16 = all(
            hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported(i)
            for i in range(torch.cuda.device_count())
        )
        if all_support_bf16:
            return True, "All GPUs support bfloat16", "bfloat16"
        else:
            names = [
                torch.cuda.get_device_name(i)
                for i in range(torch.cuda.device_count())
            ]
            return True, (
                f"GPU(s) {names} do not support bfloat16 — "
                "will use float16 automatically"
            ), "float16"
    except Exception as exc:
        return True, f"bfloat16 check inconclusive: {exc}", "float16"


def _check_transformers_api() -> tuple[bool, str]:
    """Checks transformers version and BitsAndBytesConfig availability.

    Verifies that the modern BitsAndBytesConfig API exists (not the
    deprecated load_in_8bit=True kwarg pattern).

    Returns:
        Tuple of (passed, message).
    """
    try:
        import transformers
        version = transformers.__version__
        # BitsAndBytesConfig moved in 4.30+
        from transformers import BitsAndBytesConfig  # noqa: F401
        return True, f"transformers {version} (BitsAndBytesConfig available)"
    except ImportError:
        return False, (
            "transformers not installed or BitsAndBytesConfig unavailable. "
            "Run: pip install transformers>=4.30.0"
        )
    except Exception as exc:
        return False, f"transformers check failed: {exc}"


def _check_bitsandbytes() -> tuple[bool, str]:
    """Checks bitsandbytes is importable for INT8 quantization.

    Returns:
        Tuple of (passed, message). Not fatal if missing — only needed
        for quantization="int8".
    """
    try:
        import bitsandbytes
        return True, f"bitsandbytes {bitsandbytes.__version__}"
    except ImportError:
        return True, (
            "bitsandbytes not installed — INT8 quantization unavailable. "
            "Install with: pip install bitsandbytes>=0.41.0"
        )  # warning, not error


def _check_soundfile() -> tuple[bool, str]:
    """Checks soundfile is importable for WAV I/O.

    Returns:
        Tuple of (passed, message).
    """
    try:
        import soundfile
        return True, "soundfile available"
    except ImportError:
        return False, "soundfile not installed. Run: pip install soundfile"


def _check_ffmpeg() -> tuple[bool, str]:
    """Checks FFmpeg is available on PATH.

    Returns:
        Tuple of (passed, message).
    """
    import subprocess
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            first_line = result.stdout.split("\n")[0]
            return True, first_line
        return False, "ffmpeg returned non-zero exit code"
    except FileNotFoundError:
        return False, (
            "FFmpeg not found on PATH. "
            "Install with: apt-get install ffmpeg (Linux/Colab/Kaggle)"
        )
    except subprocess.TimeoutExpired:
        return False, "FFmpeg check timed out after 5 seconds"


def _check_voice_ref_type(voice_ref: object) -> tuple[bool, str]:
    """Checks voice_ref is a valid type (str path or bytes).

    This is a boundary type check — voice_ref must be either a valid
    file path string pointing to an existing WAV, or raw WAV bytes.
    Both are acceptable; this check catches None, int, or other wrong types.

    Args:
        voice_ref: The voice reference to validate.

    Returns:
        Tuple of (passed, message).
    """
    if voice_ref is None:
        return False, (
            "voice_ref is None. Upload a narrator voice WAV file before generating."
        )
    if isinstance(voice_ref, bytes):
        if len(voice_ref) < 100:
            return False, (
                f"voice_ref bytes are too short ({len(voice_ref)} bytes). "
                "The voice WAV file may be empty or corrupted."
            )
        return True, f"voice_ref: {len(voice_ref)} bytes (in-memory WAV)"
    if isinstance(voice_ref, str):
        import os
        if not os.path.exists(voice_ref):
            return False, (
                f"voice_ref path does not exist: {voice_ref}. "
                "The voice file may have been deleted or moved."
            )
        size = os.path.getsize(voice_ref)
        if size < 100:
            return False, f"voice_ref file is too small ({size} bytes): {voice_ref}"
        return True, f"voice_ref: {voice_ref} ({size} bytes)"
    return False, (
        f"voice_ref has unexpected type {type(voice_ref).__name__}. "
        "Expected str (file path) or bytes (WAV data)."
    )


def _check_dict_keys_picklability() -> tuple[bool, str]:
    """Checks whether dict_keys objects are picklable in this Python version.

    Python 3.12 introduced stricter pickling that rejects dict_keys views.
    HuggingFace transformers GenerationConfig may store dict_keys internally.
    This check detects the issue before model loading.

    Returns:
        Tuple of (passed, message). Never fatal — will apply sanitization.
    """
    import pickle
    import sys
    v = sys.version_info
    test_keys = {"a": 1, "b": 2}.keys()
    try:
        pickle.dumps(test_keys)
        return True, f"dict_keys picklable on Python {v.major}.{v.minor}"
    except (TypeError, Exception):
        return True, (
            f"Python {v.major}.{v.minor}: dict_keys not picklable — "
            "model config sanitization will be applied automatically after loading"
        )  # warning, not error — we handle this in _sanitize_dict_keys()


def run_preflight_checks(
    voice_ref: object = None,
    check_voice_ref: bool = True,
) -> PreflightResult:
    """Runs all pre-flight environment checks and returns results.

    Checks complete in under 30 seconds. Does NOT load any TTS model.
    Call this at the start of run_pipeline() before get_pool().

    Args:
        voice_ref: Optional voice reference to validate type/existence.
        check_voice_ref: If True, validates voice_ref type and existence.

    Returns:
        PreflightResult with passed=True if no errors found.

    Raises:
        PreflightError: If any fatal checks fail. Contains the full result.
    """
    import sys
    errors: list[str] = []
    warnings: list[str] = []
    library_versions: dict[str, str] = {}
    device_info: list[dict] = []
    recommended_dtype = "float16"

    # Run all checks
    checks = [
        ("Python version", _check_python_version),
        ("PyTorch", _check_torch_version),
        ("CUDA", _check_cuda_available),
        ("transformers API", _check_transformers_api),
        ("soundfile", _check_soundfile),
        ("FFmpeg", _check_ffmpeg),
    ]

    for name, check_fn in checks:
        try:
            passed, message = check_fn()
        except Exception as exc:
            passed, message = False, f"{name} check raised unexpectedly: {exc}"

        logger.info("[preflight] %s: %s", name, message)
        library_versions[name] = message

        if not passed:
            errors.append(f"{name}: {message}")

    # bfloat16 check — returns 3 values
    try:
        bf16_passed, bf16_msg, recommended_dtype = _check_bfloat16_support()
        logger.info("[preflight] dtype: %s", bf16_msg)
        if not bf16_passed:
            errors.append(bf16_msg)
        elif "float16" in bf16_msg and "bfloat16" not in bf16_msg.lower():
            warnings.append(bf16_msg)
    except Exception as exc:
        warnings.append(f"bfloat16 check failed: {exc}")
        recommended_dtype = "float16"

    # bitsandbytes — warning only
    try:
        _, bnb_msg = _check_bitsandbytes()
        logger.info("[preflight] bitsandbytes: %s", bnb_msg)
        if "not installed" in bnb_msg:
            warnings.append(bnb_msg)
        else:
            library_versions["bitsandbytes"] = bnb_msg
    except Exception as exc:
        warnings.append(f"bitsandbytes check failed: {exc}")

    # dict_keys check — always warning
    try:
        _, dk_msg = _check_dict_keys_picklability()
        logger.info("[preflight] dict_keys: %s", dk_msg)
        if "not picklable" in dk_msg:
            warnings.append(dk_msg)
    except Exception as exc:
        warnings.append(f"dict_keys check failed: {exc}")

    # voice_ref check — only if requested
    if check_voice_ref and voice_ref is not None:
        try:
            vr_passed, vr_msg = _check_voice_ref_type(voice_ref)
            logger.info("[preflight] voice_ref: %s", vr_msg)
            if not vr_passed:
                errors.append(vr_msg)
        except Exception as exc:
            warnings.append(f"voice_ref check failed: {exc}")

    # Collect per-device info
    try:
        import torch
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                free, total = torch.cuda.mem_get_info(i)
                device_info.append({
                    "device": f"cuda:{i}",
                    "name": torch.cuda.get_device_name(i),
                    "free_vram_gb": round(free / 1e9, 2),
                    "total_vram_gb": round(total / 1e9, 2),
                    "bf16_supported": hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported(i),
                })
    except Exception as exc:
        warnings.append(f"Device info collection failed: {exc}")

    result = PreflightResult(
        passed=len(errors) == 0,
        warnings=warnings,
        errors=errors,
        device_info=device_info,
        recommended_dtype=recommended_dtype,
        python_version=sys.version_info[:3],
        library_versions=library_versions,
    )

    if warnings:
        for w in warnings:
            logger.warning("[preflight] ⚠ %s", w)

    if not result.passed:
        raise PreflightError(result)

    logger.info(
        "[preflight] ✓ All checks passed. Recommended dtype: %s",
        recommended_dtype,
    )
    return result
