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


import torch
from audiobook_factory.gpu_pool import GPUDetector, GPUPoolManager

_current_semaphore: asyncio.Semaphore | None = None
_current_semaphore_value: int = 0


def _get_active_gpu_count() -> int:
    """Returns the number of active GPU providers in the pool, or 1 if none loaded."""
    manager = GPUPoolManager.instance()
    pools = manager.all_pools()
    if not pools:
        return max(1, torch.cuda.device_count() if torch.cuda.is_available() else 1)
    return max(1, max(p.device_count for p in pools.values()))


def _get_or_resize_semaphore(desired_count: int) -> asyncio.Semaphore:
    """Returns the current semaphore if capacity matches, otherwise creates a new one."""
    global _current_semaphore, _current_semaphore_value
    if _current_semaphore is None or _current_semaphore_value != desired_count:
        import logging
        logging.getLogger(__name__).info(
            "Updating task concurrency limit: %d → %d",
            _current_semaphore_value,
            desired_count,
        )
        _current_semaphore = asyncio.Semaphore(desired_count)
        _current_semaphore_value = desired_count
    return _current_semaphore


async def _process_single_task(task_id: str, sem: asyncio.Semaphore) -> None:
    async with sem:
        task = tasks.get(task_id)
        if not task:
            task_queue.task_done()
            return

        if task.status == "cancelled":
            task_queue.task_done()
            return

        await task.update_status("running")
        await task.add_log(f"🚀 Starting generation task: {task_id}")

        try:
            cfg = AudiobookConfig(**task.config_dict)
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

            loop = asyncio.get_running_loop()

            def run_sync_pipeline():
                return run_pipeline(cfg, chapters, log_q, prog_q, task.cancel_token)

            future = loop.run_in_executor(None, run_sync_pipeline)
            monitor = asyncio.create_task(monitor_task(task, log_q, prog_q, future))

            out_files = await future
            await monitor

            if task.cancel_token.is_cancelled:
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


async def worker_loop() -> None:
    """
    Main background consumer queue loop executing generation tasks concurrently.
    Allowed concurrency is dynamically evaluated per task dispatch.
    """
    import logging
    logging.getLogger(__name__).info("[API Worker] Central task worker queue consumer started.")
    while True:
        task_id = await task_queue.get()
        active_gpu_count = _get_active_gpu_count()
        semaphore = _get_or_resize_semaphore(active_gpu_count)
        asyncio.create_task(_process_single_task(task_id, semaphore))
