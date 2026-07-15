import asyncio
import queue
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import os

from audiobook_factory.pipeline import AudiobookConfig, CancelToken, run_pipeline
from audiobook_factory.text_extractor import ExtractedChapter

@dataclass
class Task:
    task_id: str
    config_dict: Dict[str, Any]
    chapters: List[Dict[str, Any]]
    status: str = "queued"  # queued, running, completed, failed, cancelled
    progress: float = 0.0
    logs: List[str] = field(default_factory=list)
    output_files: List[str] = field(default_factory=list)
    error_message: Optional[str] = None
    
    # Synchronization
    cancel_token: CancelToken = field(default_factory=CancelToken)
    # Subscribers for this task's WebSocket events
    subscribers: List[asyncio.Queue] = field(default_factory=list)
    
    async def add_log(self, text: str):
        self.logs.append(text)
        await self.broadcast({"type": "log", "message": text})
        
    async def set_progress(self, val: float):
        self.progress = val
        await self.broadcast({"type": "progress", "progress": val})
        
    async def update_status(self, new_status: str):
        self.status = new_status
        await self.broadcast({"type": "status", "status": new_status})
        
    async def broadcast(self, data: Dict[str, Any]):
        for sub in list(self.subscribers):
            try:
                sub.put_nowait(data)
            except Exception:
                pass


# Global task memory database & execution queue
tasks: Dict[str, Task] = {}
task_queue: asyncio.Queue = asyncio.Queue()


async def monitor_task(task: Task, log_q: queue.Queue, prog_q: queue.Queue, future: asyncio.Future):
    """
    Asynchronously monitors synchronous queues populated inside the pipeline thread
    and broadcasts updates via WebSocket channels.
    """
    while not future.done() or not log_q.empty() or not prog_q.empty():
        # Read logs
        while not log_q.empty():
            try:
                msg = log_q.get_nowait()
                await task.add_log(msg)
            except queue.Empty:
                break
                
        # Read progress ratio
        while not prog_q.empty():
            try:
                cur, tot = prog_q.get_nowait()
                if tot > 0:
                    await task.set_progress(float(cur) / float(tot))
            except queue.Empty:
                break
                
        await asyncio.sleep(0.1)


async def worker_loop():
    """
    Main background consumer queue loop executing generation tasks sequentially.
    Guarantees only one task utilizes Qwen TTS on the GPU at any single time.
    """
    print("[API Worker] Central task worker queue consumer started.")
    while True:
        task_id = await task_queue.get()
        task = tasks.get(task_id)
        if not task:
            task_queue.task_done()
            continue
            
        if task.status == "cancelled":
            task_queue.task_done()
            continue
            
        await task.update_status("running")
        await task.add_log(f"🚀 Starting generation task: {task_id}")
        
        try:
            # Instantiate AudiobookConfig dataclass
            cfg = AudiobookConfig(**task.config_dict)
            # Reconstruct list of ExtractedChapter dataclasses
            chapters = [
                ExtractedChapter(
                    num=ch.get("num", idx + 1),
                    title=ch.get("title", ""),
                    text=ch.get("text", ""),
                    sentences=ch.get("sentences", [])
                ) for idx, ch in enumerate(task.chapters)
            ]
            
            log_q = queue.Queue()
            prog_q = queue.Queue()
            
            # Setup thread executor to prevent blocking the async FastAPI event loop
            loop = asyncio.get_running_loop()
            
            def run_sync_pipeline():
                return run_pipeline(cfg, chapters, log_q, prog_q, task.cancel_token)
                
            # Submit to default ThreadPoolExecutor
            future = loop.run_in_executor(None, run_sync_pipeline)
            
            # Run loop monitoring task logs/progress
            monitor = asyncio.create_task(monitor_task(task, log_q, prog_q, future))
            
            # Await execution threads to complete
            out_files = await future
            await monitor  # wait for leftover queue frames
            
            if task.cancel_token.cancelled:
                await task.update_status("cancelled")
                await task.add_log("⛔ Generation task cancelled by user.")
            else:
                task.output_files = out_files
                await task.update_status("completed")
                await task.add_log(f"✅ Generation complete. Processed {len(out_files)} files.")
                await task.broadcast({"type": "completed", "files": out_files})
                
        except Exception as e:
            import traceback
            err_msg = f"❌ Task crashed: {e}\n{traceback.format_exc()}"
            print(err_msg)
            task.error_message = str(e)
            await task.add_log(err_msg)
            await task.update_status("failed")
            
        finally:
            task_queue.task_done()
