"""
audiobook_factory/engine_worker.py
==================================
Runs one TTS engine on one device inside the engine's own environment.

Started by ``tts_providers/remote_provider.py`` with the engine
environment's Python::

    python -m audiobook_factory.engine_worker <engine> <device> [<dtype>]

Protocol: one JSON object per line. Requests on stdin, replies on the
original stdout (everything else the engine prints goes to stderr)::

    {"method": "synthesize_batch", "config": {...}, "args": [...], "kwargs": {...}}
    {"ok": true, "result": ..., "batch_size_limit": null}
    {"ok": false, "error": "...", "type": "RuntimeError"}

``bytes`` travel as ``{"__bytes__": "<base64>"}`` and tuples as lists.
"""
from __future__ import annotations

import base64
import json
import os
import sys
import traceback
from typing import Any

# Only these provider methods can be called from the main process.
ALLOWED_METHODS: frozenset[str] = frozenset({
    "ensure_ready", "synthesize", "synthesize_batch", "cleanup", "save_voice_preset", "load_voice_preset",
    "estimate_cost", "get_name",
})


def encode(value: Any) -> Any:
    """Makes *value* JSON-safe: bytes become base64 objects, tuples lists."""
    if isinstance(value, (bytes, bytearray)):
        return {"__bytes__": base64.b64encode(bytes(value)).decode("ascii")}
    if isinstance(value, (list, tuple)):
        return [encode(item) for item in value]
    if isinstance(value, dict):
        return {str(key): encode(item) for key, item in value.items()}
    return value


def decode(value: Any) -> Any:
    """Reverses :func:`encode`."""
    if isinstance(value, dict):
        if set(value) == {"__bytes__"}:
            return base64.b64decode(value["__bytes__"])
        return {key: decode(item) for key, item in value.items()}
    if isinstance(value, list):
        return [decode(item) for item in value]
    return value


def main(argv: list[str]) -> int:
    engine, device = argv[0], argv[1]
    dtype = argv[2] if len(argv) > 2 and argv[2] else None
    # Engines print freely; only protocol lines may reach the parent's pipe.
    protocol = os.fdopen(os.dup(1), "w", buffering=1, encoding="utf-8")
    os.dup2(2, 1)
    sys.stdout = sys.stderr

    from audiobook_factory.pipeline import AudiobookConfig
    from audiobook_factory.tts_providers.registry import provider_class

    cls = provider_class(engine)
    provider = None
    for line in sys.stdin:
        if not line.strip():
            continue
        request: dict = {}
        try:
            request = json.loads(line)
            method = request.get("method")
            if method not in ALLOWED_METHODS:
                raise ValueError(f"method {method!r} is not allowed")
            config = AudiobookConfig.from_dict(request.get("config") or {})
            if provider is None:
                provider = cls.create_for_device(device, config, dtype_override=dtype)
            provider.config = config
            result = getattr(provider, method)(*decode(request.get("args") or []), **decode(request.get("kwargs") or {}))
            reply = {"ok": True, "result": encode(result), "batch_size_limit": getattr(provider, "batch_size_limit", None)}
        except Exception as exc:
            traceback.print_exc()
            reply = {"ok": False, "error": str(exc), "type": type(exc).__name__,
                     "batch_size_limit": getattr(provider, "batch_size_limit", None)}
        protocol.write(json.dumps(reply) + "\n")
        if request.get("method") == "cleanup":
            break
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
