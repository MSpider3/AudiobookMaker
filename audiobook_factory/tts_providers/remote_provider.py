"""
audiobook_factory/tts_providers/remote_provider.py
====================================================
Stand-in for an engine that runs in its own environment.

``get_tts_provider`` returns a :class:`RemoteProvider` when the engine was
installed with ``engine_envs.install``. It starts ``engine_worker.py`` with
that environment's Python, bound to one device, and forwards each call over
a pipe. The GPU pool, the chapter pipeline and the previews use it like any
provider. ``cleanup()`` ends the worker, which frees all its GPU memory.
"""
from __future__ import annotations

import dataclasses
import json
import logging
import os
import subprocess
import threading
from typing import TYPE_CHECKING, Any

from audiobook_factory.tts_providers.base_tts_provider import BaseTTSProvider, ProviderInfo

if TYPE_CHECKING:
    from audiobook_factory.pipeline import AudiobookConfig

logger = logging.getLogger(__name__)

_ROOT: str = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WORKER_ENV_FLAG: str = "ABM_ENGINE_WORKER"


class RemoteProvider(BaseTTSProvider):
    """Forwards provider calls to a worker process in the engine's environment."""

    def __init__(self, engine: str, python: str, env_vars: dict[str, str], config: "AudiobookConfig",
                 device: str | None = None, dtype_override: str | None = None) -> None:
        super().__init__(config)
        from audiobook_factory.tts_providers.registry import provider_class

        self._engine = engine
        self._python = python
        self._env_vars = dict(env_vars)
        self._device = device or getattr(config, "device", "cuda")
        self._dtype = dtype_override or ""
        self._info = provider_class(engine).info()
        self._process: subprocess.Popen | None = None
        self._ready = False
        self._lock = threading.Lock()

    def info(self) -> ProviderInfo:  # type: ignore[override]
        return self._info

    @property
    def device(self) -> str:
        return self._device

    @property
    def is_ready(self) -> bool:
        return self._ready and self._process is not None and self._process.poll() is None

    def _start(self) -> subprocess.Popen:
        if self._process is not None and self._process.poll() is None:
            return self._process
        env = dict(os.environ, **self._env_vars)
        env[WORKER_ENV_FLAG] = "1"
        # The engine's own packages must come before anything inherited.
        env["PYTHONPATH"] = _ROOT
        worker_device = self._device
        if self._device.startswith("cuda:"):
            # The worker sees only its own GPU: a library that picks "the
            # default GPU" cannot land on another worker's.
            index = int(self._device.split(":")[1])
            visible = [d for d in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if d.strip()]
            env["CUDA_VISIBLE_DEVICES"] = visible[index] if index < len(visible) else str(index)
            worker_device = "cuda:0"
        logger.info("[remote] Starting %s worker on %s (%s).", self._engine, self._device, self._python)
        self._process = subprocess.Popen(
            [self._python, "-m", "audiobook_factory.engine_worker", self._engine, worker_device, self._dtype],
            cwd=_ROOT, env=env, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1,
        )
        return self._process

    def _call(self, method: str, *args: Any, **kwargs: Any) -> Any:
        from audiobook_factory.engine_worker import decode, encode

        with self._lock:
            process = self._start()
            request = {"method": method, "config": dataclasses.asdict(self.config),
                       "args": encode(list(args)), "kwargs": encode(kwargs)}
            try:
                process.stdin.write(json.dumps(request) + "\n")
                process.stdin.flush()
                line = process.stdout.readline()
            except (BrokenPipeError, OSError) as exc:
                line, failure = "", exc
            else:
                failure = None
            if not line:
                code = process.poll()
                self._process = None
                self._ready = False
                raise RuntimeError(
                    f"The {self._info.display_name} worker on {self._device} stopped (exit code {code}); "
                    "its error is in the log above."
                ) from failure
            reply = json.loads(line)
        if reply.get("batch_size_limit"):
            self.batch_size_limit = int(reply["batch_size_limit"])
        if not reply.get("ok"):
            # Keep the message: "out of memory" in it drives the batch split.
            raise RuntimeError(f"{reply.get('type', 'Error')}: {reply.get('error', '')}")
        return decode(reply.get("result"))

    # ── Provider interface ───────────────────────────────────────────────────

    def ensure_ready(self) -> None:
        self._call("ensure_ready")
        self._ready = True

    def synthesize(self, text: str, voice_ref: str | bytes, out_path: str | None = None, *,
                   return_bytes: bool = False) -> tuple[str | bytes, float]:
        audio, duration = self._call("synthesize", text, voice_ref, return_bytes=True)
        if out_path and not return_bytes:
            directory = os.path.dirname(out_path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            with open(out_path, "wb") as fh:
                fh.write(audio)
            return out_path, float(duration)
        return audio, float(duration)

    def synthesize_batch(self, texts: list[str], voice_ref: bytes, *,
                         return_bytes: bool = True) -> list[tuple[bytes | str, float]]:
        return [(audio, float(duration)) for audio, duration in
                self._call("synthesize_batch", list(texts), voice_ref, return_bytes=True)]

    def save_voice_preset(self, path: str, voice_ref: str | bytes | None = None, *,
                          transcript: str | None = None) -> dict[str, Any]:
        return self._call("save_voice_preset", path, voice_ref, transcript=transcript)

    def load_voice_preset(self, path: str) -> dict[str, Any]:
        return self._call("load_voice_preset", path)

    def estimate_cost(self, total_chars: int) -> float:
        return 0.0

    def get_name(self) -> str:
        return self._info.display_name

    def cleanup(self) -> None:
        """Ends the worker; its process exit frees the GPU memory."""
        process = self._process
        if process is None:
            return
        try:
            if process.poll() is None:
                self._call("cleanup")
        except Exception as exc:
            logger.debug("[remote] cleanup call failed: %s", exc)
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
        self._process = None
        self._ready = False
