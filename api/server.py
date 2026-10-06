"""
api/server.py
=============
FastAPI routes and the WebSocket event stream of the AudiobookMaker backend.

Handlers never run model loading, synthesis or DSP on the event loop: a
blocked loop makes the health check time out, and a client that believes the
API is down loads a second copy of the model in its own process.
"""
from __future__ import annotations

import json
import asyncio
import dataclasses
import io
import logging
import mimetypes
import os
import sys
from typing import Any, Dict, List, Optional
import uuid

# Ensure project root is in sys.path
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from collections import defaultdict
from contextlib import asynccontextmanager
import secrets
import threading
import time
from fastapi import (
    FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Response,
    UploadFile, File, Form, Depends, Header, Query, Request
)
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from audiobook_factory.pipeline import AudiobookConfig, preview_tts
from audiobook_factory.voice_preprocessor import (
    PreprocessConfig,
    preprocess as voice_preprocess,
    preprocess_with_report,
)

_PREPROCESS_DEFAULTS = PreprocessConfig()
from api.worker import (
    ConfigError, Task, evict_old_tasks, prepare_config, task_queue, tasks, worker_loop,
)

logger = logging.getLogger(__name__)

_TERMINAL_STATUSES: tuple[str, ...] = ("completed", "failed", "cancelled")
# How long the socket stays open after the last event of a finished task, so
# proxies deliver the completion payload before the close frame.
_WS_CLOSE_GRACE_SEC: float = 3.0
_WS_PING_INTERVAL_SEC: float = 15.0
# mimetypes does not know the audiobook container.
_AUDIO_MEDIA_TYPES: dict[str, str] = {
    ".m4b": "audio/mp4",
    ".m4a": "audio/mp4",
    ".mp3": "audio/mpeg",
    ".flac": "audio/flac",
    ".wav": "audio/wav",
    ".ogg": "audio/ogg",
}


# ── Lifespan Event Handler ───────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Spin up background task queue worker in the event loop
    worker_task = asyncio.create_task(worker_loop())
    print("[API Server] Background worker consumer task spawned successfully.")

    # No model is loaded here. The pipeline builds the provider pool for the
    # engine and model a task actually asks for, the first time one runs;
    # warming the default engine at startup made every user of another engine
    # pay an evict + reload and kept VRAM occupied by an idle server.
    # ABM_SKIP_GPU_WARMUP used to disable that warmup and is still accepted
    # (it has nothing left to skip).
    if os.environ.get("ABM_SKIP_GPU_WARMUP"):
        logger.debug("ABM_SKIP_GPU_WARMUP is set; startup warmup no longer exists.")

    yield

    # Shutdown: Cancel active tasks and shutdown GPU pools cleanly
    print("[API Server] Shutdown event triggered. Cancelling active tasks...")
    active_found = False
    for task_id, task in list(tasks.items()):
        if task.status in ("queued", "running"):
            active_found = True
            task.cancel_token.cancel()
            await task.add_log("⛔ Server shutdown requested. Cancelling task.")
    if active_found:
        await asyncio.sleep(0.2)
    try:
        from audiobook_factory.gpu_pool import GPUPoolManager
        GPUPoolManager.instance().shutdown()
        print("[API Server] GPUPoolManager shut down cleanly.")
    except Exception as exc:
        print(f"[API Server] GPUPoolManager shutdown warning: {exc}")
    worker_task.cancel()


app = FastAPI(
    title="AudiobookMaker Backend Server",
    description="Decoupled high-performance async task runner and model server.",
    version="1.0.0",
    lifespan=lifespan,
)


# ── Pydantic Request Models ───────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    config: Dict[str, Any]
    chapters: List[Dict[str, Any]]


class VoiceTestRequest(BaseModel):
    config: Dict[str, Any]
    text: str


from audiobook_factory.gpu_pool import GPUDetector, GPUPoolManager

# ── Auth & Rate Limiting ───────────────────────────────────────────────────────

async def require_auth(
    x_api_key: str = Header(default="", alias="x-api-key"),
    authorization: str = Header(default="", alias="authorization"),
) -> None:
    """Validates shared secret auth if ABM_API_SECRET is configured."""
    secret = os.environ.get("ABM_API_SECRET", "")
    if not secret:
        return
    token = x_api_key
    if not token and authorization.startswith("Bearer "):
        token = authorization[7:].strip()
    if not token or not secrets.compare_digest(token, secret):
        raise HTTPException(status_code=401, detail="Unauthorized: invalid or missing API key.")


class SimpleRateLimiter:
    """In-memory sliding-window rate limiter per client IP."""

    def __init__(self, max_requests: int, window_seconds: float):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests: dict[str, list[float]] = defaultdict(list)
        self._lock = threading.Lock()

    def check(self, client_ip: str) -> bool:
        now = time.monotonic()
        cutoff = now - self.window_seconds
        with self._lock:
            history = self._requests[client_ip]
            self._requests[client_ip] = [t for t in history if t > cutoff]
            if len(self._requests[client_ip]) >= self.max_requests:
                return False
            self._requests[client_ip].append(now)
            return True


_generate_limiter = SimpleRateLimiter(max_requests=30, window_seconds=60.0)
_preprocess_limiter = SimpleRateLimiter(max_requests=60, window_seconds=60.0)


async def check_generate_rate_limit(request: Request) -> None:
    client_ip = request.client.host if request.client else "127.0.0.1"
    if not _generate_limiter.check(client_ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded for generation requests.")


async def check_preprocess_rate_limit(request: Request) -> None:
    client_ip = request.client.host if request.client else "127.0.0.1"
    if not _preprocess_limiter.check(client_ip):
        raise HTTPException(status_code=429, detail="Rate limit exceeded for preprocess requests.")


@app.get("/api/v1/health")
async def health_check():
    detected = GPUDetector.detect_devices()
    all_pools = GPUPoolManager.instance().all_pools()
    pools_data = {}
    for name, pool in all_pools.items():
        devices_data = []
        for device in pool.devices:
            dev_info = GPUDetector.get_device_info(device)
            provider_inst = pool._device_map.get(device)
            dev_info["model_loaded"] = getattr(provider_inst, "is_ready", False) if provider_inst is not None else False
            devices_data.append(dev_info)

        pools_data[name] = {
            "device_count": pool.device_count,
            "devices": devices_data,
        }

    return {
        "status": "ok",
        "message": "AudiobookMaker API Server is active.",
        "gpu": {
            "detected_devices": detected,
            "provider_pools": pools_data,
        },
    }


def _reject(exc: ConfigError) -> HTTPException:
    """Maps a rejected request config to an HTTP 400 with a structured detail."""
    return HTTPException(status_code=400, detail=exc.as_detail())


def _get_task_or_404(task_id: str) -> Task:
    task = tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found.")
    return task


@app.get("/api/v1/providers", dependencies=[Depends(require_auth)])
def list_tts_providers(include_hidden: bool = False):
    """Describes every TTS provider that can be imported on this server.

    Each entry is the provider's ``ProviderInfo`` as JSON (tuples become
    arrays), with ``name`` set to the registry key that ``tts_provider_name``
    accepts. Providers whose module fails to import are reported under
    ``unavailable`` instead of breaking the listing.
    """
    from audiobook_factory.tts_providers.registry import provider_info, provider_names

    providers: List[Dict[str, Any]] = []
    unavailable: List[Dict[str, str]] = []
    for name in provider_names(include_hidden=include_hidden):
        try:
            entry = dataclasses.asdict(provider_info(name))
        except Exception as exc:
            unavailable.append({"name": name, "error": str(exc)})
            continue
        entry["name"] = name
        providers.append(entry)
    return {
        "providers": providers,
        "unavailable": unavailable,
        "default": AudiobookConfig().tts_provider_name,
    }


@app.post("/api/v1/generate", dependencies=[Depends(require_auth), Depends(check_generate_rate_limit)])
async def enqueue_generation(payload: GenerateRequest):
    # Reject a bad request now, with a reason, rather than as a failed task.
    # The worker validates again when the task starts.
    try:
        await asyncio.to_thread(prepare_config, payload.config)
    except ConfigError as exc:
        raise _reject(exc)

    evict_old_tasks()
    task_id = str(uuid.uuid4())

    # Store task details
    task = Task(
        task_id=task_id,
        config_dict=payload.config,
        chapters=payload.chapters
    )
    tasks[task_id] = task

    # Push to queue
    await task_queue.put(task_id)
    print(f"[API Server] Enqueued task: {task_id}")
    return {"task_id": task_id, "status": "queued"}


@app.get("/api/v1/tasks", dependencies=[Depends(require_auth)])
async def list_tasks():
    """Lists known tasks, newest first, without logs or chapter text."""
    ordered = sorted(tasks.values(), key=lambda t: t.created_at, reverse=True)
    return {"tasks": [t.summary() for t in ordered]}


@app.post("/api/v1/tasks/{task_id}/cancel", dependencies=[Depends(require_auth)])
async def cancel_task(task_id: str):
    task = _get_task_or_404(task_id)

    task.cancel_token.cancel()
    if task.status in ("queued", "running"):
        await task.add_log("⛔ Cancellation requested by client.")
        if task.status == "queued":
            await task.update_status("cancelled")
            # The worker skips a cancelled task without another word, so the
            # end of the session has to be announced here; otherwise a
            # WebSocket client waits out its whole grace timeout.
            await task.end_session()

    return {"task_id": task_id, "status": task.status}


@app.get("/api/v1/tasks/{task_id}", dependencies=[Depends(require_auth)])
async def get_task_status(task_id: str, since: Optional[int] = Query(default=None, ge=0)):
    """Returns a task's state.

    Without ``since`` the whole log is returned. With ``?since=<n>`` only the
    log lines from index ``n`` on are; pass the previous response's
    ``log_count`` to receive just the new ones.
    """
    task = _get_task_or_404(task_id)
    offset = min(since or 0, len(task.logs))
    body = task.summary()
    body.update({
        "logs": task.logs[offset:],
        "log_offset": offset,
        "log_count": len(task.logs),
        "output_files": task.output_files,
    })
    return body


@app.get("/api/v1/tasks/{task_id}/files/{index}", dependencies=[Depends(require_auth)])
def download_task_file(task_id: str, index: int):
    """Downloads output file number ``index`` of a finished task.

    Only paths the pipeline recorded in the task's ``output_files`` are
    served; the client supplies an index, never a path.
    """
    task = _get_task_or_404(task_id)
    if not task.is_finished:
        raise HTTPException(status_code=409, detail="Task has not finished yet.")
    if index < 0 or index >= len(task.output_files):
        raise HTTPException(status_code=404, detail="No such output file.")
    path = task.output_files[index]
    if not isinstance(path, str) or not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="Output file is no longer on disk.")
    extension = os.path.splitext(path)[1].lower()
    media_type = (
        _AUDIO_MEDIA_TYPES.get(extension)
        or mimetypes.guess_type(path)[0]
        or "application/octet-stream"
    )
    return FileResponse(path, media_type=media_type, filename=os.path.basename(path))


@app.post("/api/v1/voice-test", dependencies=[Depends(require_auth)])
async def api_voice_test(payload: VoiceTestRequest):
    """
    Generates preview speech using the backend's shared loaded model.
    """
    try:
        cfg = await asyncio.to_thread(prepare_config, payload.config, contain_output_dir=False)
    except ConfigError as exc:
        raise _reject(exc)
    try:
        # Loads the model on first use and synthesizes: seconds to minutes.
        wav_bytes = await asyncio.to_thread(preview_tts, payload.text, cfg)
    except Exception as e:
        logger.exception("Voice test failed")
        raise HTTPException(status_code=500, detail=f"TTS synthesis error: {e}")
    if wav_bytes is None:
        raise HTTPException(status_code=500, detail="TTS synthesis error: TTS generation returned empty audio data.")
    return StreamingResponse(io.BytesIO(wav_bytes), media_type="audio/wav")


@app.post("/api/v1/preprocess", dependencies=[Depends(require_auth), Depends(check_preprocess_rate_limit)])
async def api_preprocess(
    noise_reduce: bool = Form(...),
    noise_reduce_strength: float = Form(...),
    noise_gate: bool = Form(...),
    noise_gate_threshold_db: float = Form(...),
    highpass_filter: bool = Form(...),
    highpass_cutoff_hz: float = Form(...),
    silence_removal: bool = Form(...),
    silence_threshold_db: float = Form(...),
    min_segment_ms: int = Form(...),
    max_silence_kept_ms: int = Form(...),
    normalize_volume: bool = Form(...),
    normalize_target_dbfs: float = Form(...),
    formant_shift: bool = Form(...),
    formant_quefrency: float = Form(...),
    formant_timbre: float = Form(...),
    resample: bool = Form(...),
    target_sample_rate: int = Form(...),
    use_cache: bool = Form(default=False),
    # Added with the rewritten preprocessor. Defaults mirror PreprocessConfig,
    # so clients that only send the original fields keep working.
    noise_gate_range_db: float = Form(default=_PREPROCESS_DEFAULTS.noise_gate_range_db),
    trim_silence: bool = Form(default=_PREPROCESS_DEFAULTS.trim_silence),
    edge_silence_ms: int = Form(default=_PREPROCESS_DEFAULTS.edge_silence_ms),
    edge_fade_ms: int = Form(default=_PREPROCESS_DEFAULTS.edge_fade_ms),
    normalize_mode: str = Form(default=_PREPROCESS_DEFAULTS.normalize_mode),
    loudness_target_lufs: float = Form(default=_PREPROCESS_DEFAULTS.loudness_target_lufs),
    true_peak_ceiling_dbfs: float = Form(default=_PREPROCESS_DEFAULTS.true_peak_ceiling_dbfs),
    allow_upsample: bool = Form(default=_PREPROCESS_DEFAULTS.allow_upsample),
    select_best_window: bool = Form(default=_PREPROCESS_DEFAULTS.select_best_window),
    best_window_seconds: float = Form(default=_PREPROCESS_DEFAULTS.best_window_seconds),
    audio_file: UploadFile = File(...)
):
    """
    Cleans raw uploaded audio files using voice preprocessor algorithms.

    The cleaned WAV is the response body. The analysis of the result
    (duration, loudness, SNR, warnings…) is returned as ASCII JSON in the
    ``X-Voice-Report`` response header.
    """
    try:
        cfg = PreprocessConfig(
            noise_reduce=noise_reduce,
            noise_reduce_strength=noise_reduce_strength,
            noise_gate=noise_gate,
            noise_gate_threshold_db=noise_gate_threshold_db,
            highpass_filter=highpass_filter,
            highpass_cutoff_hz=highpass_cutoff_hz,
            silence_removal=silence_removal,
            silence_threshold_db=silence_threshold_db,
            min_segment_ms=min_segment_ms,
            max_silence_kept_ms=max_silence_kept_ms,
            normalize_volume=normalize_volume,
            normalize_target_dbfs=normalize_target_dbfs,
            formant_shift=formant_shift,
            formant_quefrency=formant_quefrency,
            formant_timbre=formant_timbre,
            resample=resample,
            target_sample_rate=target_sample_rate,
            noise_gate_range_db=noise_gate_range_db,
            trim_silence=trim_silence,
            edge_silence_ms=edge_silence_ms,
            edge_fade_ms=edge_fade_ms,
            normalize_mode=normalize_mode,
            loudness_target_lufs=loudness_target_lufs,
            true_peak_ceiling_dbfs=true_peak_ceiling_dbfs,
            allow_upsample=allow_upsample,
            select_best_window=select_best_window,
            best_window_seconds=best_window_seconds,
        )

        in_bytes = await audio_file.read()
        # Decoding, noise reduction and resampling are CPU-bound.
        out_bytes, report = await asyncio.to_thread(
            preprocess_with_report, in_bytes, cfg, use_cache=use_cache
        )
        headers = {"X-Voice-Report": json.dumps(report.to_dict(), ensure_ascii=True)}
        return StreamingResponse(io.BytesIO(out_bytes), media_type="audio/wav", headers=headers)
    except ValueError as e:
        # Undecodable, empty or over-long audio is the caller's problem, not a server fault.
        raise HTTPException(status_code=400, detail=f"Preprocessing error: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Preprocessing error: {e}")


# ── WebSocket Event Streaming Route ──────────────────────────────────────────

@app.websocket("/api/v1/ws/{task_id}")
async def task_websocket_endpoint(websocket: WebSocket, task_id: str):
    secret = os.environ.get("ABM_API_SECRET", "")
    if secret:
        token = websocket.query_params.get("api_key") or websocket.headers.get("x-api-key") or ""
        if not token or not secrets.compare_digest(token, secret):
            await websocket.accept()
            await websocket.send_json({"type": "error", "message": "Unauthorized: invalid or missing API key."})
            await websocket.close(code=1008)
            return

    await websocket.accept()
    task = tasks.get(task_id)
    if not task:
        await websocket.send_json({"type": "error", "message": "Requested task not found."})
        await websocket.close()
        return

    # Subscribe and snapshot the history in the same step (no await between
    # them), so a log line is delivered exactly once: from the snapshot or
    # from the queue, never both and never neither.
    ws_queue: asyncio.Queue = asyncio.Queue()
    task.subscribers.append(ws_queue)
    history = list(task.logs)
    print(f"[WebSocket] Client connected to task subscription: {task_id}")

    async def _ws_keepalive():
        while True:
            await asyncio.sleep(_WS_PING_INTERVAL_SEC)
            try:
                await websocket.send_json({"type": "ping"})
            except Exception:
                break

    async def _wait_for_disconnect():
        # The client sends nothing in this protocol; anything it does send is
        # discarded. Returns when the peer has gone.
        while True:
            message = await websocket.receive()
            if message.get("type") == "websocket.disconnect":
                return

    ping_task = asyncio.create_task(_ws_keepalive())
    # Awaiting only the task queue never notices a client that went away: the
    # handler and its subscriber queue would live until the task ends (for a
    # stalled task, forever) and hold up the server's graceful shutdown.
    gone_task = asyncio.create_task(_wait_for_disconnect())
    next_event: asyncio.Task | None = None

    try:
        # Send historical logs first
        for log_msg in history:
            await websocket.send_json({"type": "log", "message": log_msg})
        # Send progress baseline
        await websocket.send_json({"type": "progress", "progress": task.progress})
        await websocket.send_json({"type": "status", "status": task.status})
        if task.status == "completed" and task.output_files:
            await websocket.send_json({"type": "completed", "files": task.output_files})

        # A client that connects after the task ended has nothing more to wait for.
        terminal_seen = task.status in _TERMINAL_STATUSES

        while True:
            # Forward updates from the task channel queue to the client.
            # The worker announces the terminal *status* first and the
            # "completed"/"session_end" events (which carry the file list)
            # after it, so keep draining for a short grace period instead of
            # closing on the status message and dropping them.
            if next_event is None:
                next_event = asyncio.create_task(ws_queue.get())
            done, _pending = await asyncio.wait(
                {next_event, gone_task},
                timeout=_WS_CLOSE_GRACE_SEC if terminal_seen else None,
                return_when=asyncio.FIRST_COMPLETED,
            )
            if next_event not in done:
                # Client disconnected, or the grace period ran out.
                break
            data = next_event.result()
            next_event = None
            await websocket.send_json(data)
            ws_queue.task_done()

            if data.get("type") == "session_end":
                # Grace period to ensure all network proxies process the
                # completion payload; cut short if the client hangs up first.
                await asyncio.wait({gone_task}, timeout=_WS_CLOSE_GRACE_SEC)
                break
            if data.get("type") == "status" and data.get("status") in _TERMINAL_STATUSES:
                terminal_seen = True

    except WebSocketDisconnect:
        pass
    except Exception as e:
        print(f"[WebSocket] Event transmission exception: {e}")
    finally:
        for pending in (ping_task, gone_task, next_event):
            if pending is not None and not pending.done():
                pending.cancel()
        if gone_task.done() and not gone_task.cancelled():
            # Retrieve the result so a failed receive is not reported as an
            # un-awaited task exception.
            gone_task.exception()
        if ws_queue in task.subscribers:
            task.subscribers.remove(ws_queue)
        print(f"[WebSocket] Client disconnected from task subscription: {task_id}")
        try:
            await websocket.close()
        except Exception:
            pass
