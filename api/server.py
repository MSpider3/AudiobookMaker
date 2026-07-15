import asyncio
import io
import os
import sys
from typing import Any, Dict, List, Optional
import uuid

# Ensure project root is in sys.path
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Response, UploadFile, File, Form
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from audiobook_factory.pipeline import AudiobookConfig, preview_tts
from audiobook_factory.voice_preprocessor import PreprocessConfig, preprocess as voice_preprocess
from api.worker import tasks, task_queue, Task, worker_loop

app = FastAPI(
    title="AudiobookMaker Backend Server",
    description="Decoupled high-performance async task runner and model server.",
    version="1.0.0"
)


# ── Pydantic Request Models ───────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    config: Dict[str, Any]
    chapters: List[Dict[str, Any]]


class VoiceTestRequest(BaseModel):
    config: Dict[str, Any]
    text: str


# ── Start-up Event ────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    # Spin up background task queue worker in the event loop
    asyncio.create_task(worker_loop())
    print("[API Server] Background worker consumer task spawned successfully.")


# ── REST API Endpoints ────────────────────────────────────────────────────────

@app.get("/api/v1/health")
async def health_check():
    return {"status": "ok", "message": "AudiobookMaker API Server is active."}


@app.post("/api/v1/generate")
async def enqueue_generation(payload: GenerateRequest):
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


@app.post("/api/v1/tasks/{task_id}/cancel")
async def cancel_task(task_id: str):
    task = tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found.")
        
    task.cancel_token.cancel()
    if task.status in ("queued", "running"):
        await task.add_log("⛔ Cancellation requested by client.")
        if task.status == "queued":
            await task.update_status("cancelled")
            
    return {"task_id": task_id, "status": task.status}


@app.get("/api/v1/tasks/{task_id}")
async def get_task_status(task_id: str):
    task = tasks.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found.")
        
    return {
        "task_id": task.task_id,
        "status": task.status,
        "progress": task.progress,
        "logs": task.logs,
        "output_files": task.output_files,
        "error_message": task.error_message
    }


@app.post("/api/v1/voice-test")
async def api_voice_test(payload: VoiceTestRequest):
    """
    Generates preview speech using the backend's shared loaded model.
    """
    try:
        cfg = AudiobookConfig(**payload.config)
        wav_bytes = preview_tts(payload.text, cfg)
        if wav_bytes is None:
            raise HTTPException(status_code=500, detail="TTS generation returned empty audio data.")
            
        return StreamingResponse(io.BytesIO(wav_bytes), media_type="audio/wav")
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"TTS synthesis error: {e}")


@app.post("/api/v1/preprocess")
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
    audio_file: UploadFile = File(...)
):
    """
    Cleans raw uploaded audio files using voice preprocessor algorithms.
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
            target_sample_rate=target_sample_rate
        )
        
        in_bytes = await audio_file.read()
        out_bytes = voice_preprocess(in_bytes, cfg)
        
        return StreamingResponse(io.BytesIO(out_bytes), media_type="audio/wav")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Preprocessing error: {e}")


# ── WebSocket Event Streaming Route ──────────────────────────────────────────

@app.websocket("/api/v1/ws/{task_id}")
async def task_websocket_endpoint(websocket: WebSocket, task_id: str):
    await websocket.accept()
    task = tasks.get(task_id)
    if not task:
        await websocket.send_json({"type": "error", "message": "Requested task not found."})
        await websocket.close()
        return
        
    # Create connection channel queue
    ws_queue = asyncio.Queue()
    task.subscribers.append(ws_queue)
    print(f"[WebSocket] Client connected to task subscription: {task_id}")
    
    # Send historical logs first
    for log_msg in task.logs:
        await websocket.send_json({"type": "log", "message": log_msg})
    # Send progress baseline
    await websocket.send_json({"type": "progress", "progress": task.progress})
    await websocket.send_json({"type": "status", "status": task.status})
    if task.status == "completed" and task.output_files:
        await websocket.send_json({"type": "completed", "files": task.output_files})
        
    try:
        while True:
            # Poll updates from the task channel queue and send to client
            data = await ws_queue.get()
            await websocket.send_json(data)
            ws_queue.task_done()
            
            # If task terminates, we can close the socket connection cleanly
            if data.get("type") in ("completed", "status") and data.get("status") in ("completed", "failed", "cancelled"):
                # Brief sleep to ensure any trailing logs finish sending
                await asyncio.sleep(0.5)
                break
                
    except WebSocketDisconnect:
        print(f"[WebSocket] Client disconnected from task subscription: {task_id}")
    except Exception as e:
        print(f"[WebSocket] Event transmission exception: {e}")
    finally:
        if ws_queue in task.subscribers:
            task.subscribers.remove(ws_queue)
        try:
            await websocket.close()
        except Exception:
            pass
