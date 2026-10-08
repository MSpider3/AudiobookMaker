"""
api/worker.py
=============
In-memory task store, request validation and the queue consumer that runs
generation tasks for the FastAPI backend.

Everything a client sends in a request body is untrusted: ``prepare_config``
is the single place where it is turned into an ``AudiobookConfig``.
"""
from __future__ import annotations

import asyncio
import logging
import os
import queue
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from audiobook_factory.pipeline import AudiobookConfig, CancelToken, run_pipeline
from audiobook_factory.text_extractor import ExtractedChapter

logger = logging.getLogger(__name__)

_TERMINAL_STATUSES: frozenset[str] = frozenset({"completed", "failed", "cancelled"})
_MAX_COMPLETED_TASKS: int = 50
# Config fields that name a file the pipeline reads.
_PATH_FIELDS: tuple[str, ...] = ("voice_file", "cover_image", "voice_preset")
_SCALAR_TYPES: tuple[type, ...] = (str, int, float, bool, type(None))
# How often a waiting task re-reads the concurrency limit, which changes when
# a provider pool is built or evicted.
_SLOT_RECHECK_SEC: float = 1.0


class ConfigError(ValueError):
    """A request's config was rejected.

    Attributes
    ----------
    code : str
        Stable machine-readable reason: ``invalid_config``,
        ``unknown_provider``, ``provider_unavailable``,
        ``output_dir_outside_base``, ``invalid_path`` or
        ``invalid_tts_options``.
    field : str
        Name of the offending config field, when there is one.
    """

    def __init__(self, code: str, message: str, field: str = "") -> None:
        super().__init__(message)
        self.code = code
        self.field = field

    def as_detail(self) -> Dict[str, Any]:
        """Returns the JSON body sent to the client as the error ``detail``."""
        detail: Dict[str, Any] = {"code": self.code, "message": str(self)}
        if self.field:
            detail["field"] = self.field
        if self.code == "output_dir_outside_base":
            detail["output_base"] = get_output_base()
        return detail


def get_output_base() -> str:
    """Returns the directory every task's ``output_dir`` must stay inside.

    Defaults to ``<project root>/audiobook_output``; override with the
    ``ABM_OUTPUT_BASE`` environment variable.
    """
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.realpath(
        os.environ.get("ABM_OUTPUT_BASE", os.path.join(root, "audiobook_output"))
    )


def _is_inside(path: str, base: str) -> bool:
    """True when the resolved *path* is *base* or lies beneath it."""
    return path == base or path.startswith(base + os.sep)


def _require_regular_file(value: Any, field_name: str) -> None:
    """Raises ConfigError unless *value* is empty or names an existing regular file.

    ``os.path.isfile`` follows symlinks and is False for directories, device
    nodes, FIFOs and sockets, so ``/dev/zero`` or a directory never reaches
    the audio decoders.
    """
    if value is None or value == "":
        return
    if not isinstance(value, str):
        raise ConfigError("invalid_path", f"{field_name} must be a file path string.", field_name)
    if "\x00" in value or not os.path.isfile(value):
        raise ConfigError(
            "invalid_path",
            f"{field_name} is not an existing regular file: {value!r}",
            field_name,
        )


def prepare_config(config_dict: Any, *, contain_output_dir: bool = True) -> AudiobookConfig:
    """Builds the AudiobookConfig of an API request, rejecting unsafe values.

    * ``tts_provider_name`` must be a registered provider (aliases are
      canonicalised so one engine never ends up in two pools).
    * ``output_dir`` is rebased under / contained in :func:`get_output_base`
      so a request can never make the pipeline create, overwrite or delete
      files elsewhere.
    * ``voice_file``, ``cover_image``, ``voice_preset`` and every provider
      option of kind ``"file"`` must be empty or an existing regular file.
    * ``tts_options`` must be a flat ``{str: scalar}`` mapping.

    ``tts_model_name`` is deliberately not checked here: each provider only
    loads ids listed in its ``ProviderInfo.models`` and falls back to its
    default for anything else (``BaseTTSProvider.resolve_model_id``), so an
    arbitrary repository id in a request is ignored rather than downloaded.

    Parameters
    ----------
    config_dict : Any
        The ``config`` object of the request body.
    contain_output_dir : bool
        False for requests that never write to ``output_dir`` (voice preview).

    Raises
    ------
    ConfigError
        With a ``code`` describing what was rejected.
    """
    from audiobook_factory.tts_providers.registry import (
        canonical_name, is_known_provider, provider_info,
    )

    if not isinstance(config_dict, dict):
        raise ConfigError("invalid_config", "config must be a JSON object.")

    raw_options = config_dict.get("tts_options", {})
    if raw_options is None:
        raw_options = {}
    if not isinstance(raw_options, dict):
        raise ConfigError("invalid_tts_options", "tts_options must be an object.", "tts_options")
    for key, value in raw_options.items():
        if not isinstance(key, str) or not isinstance(value, _SCALAR_TYPES):
            raise ConfigError(
                "invalid_tts_options",
                f"tts_options[{key!r}] must be a string, number, boolean or null.",
                "tts_options",
            )

    try:
        cfg = AudiobookConfig.from_dict(config_dict)
    except Exception as exc:
        raise ConfigError("invalid_config", f"Invalid config: {exc}") from exc
    cfg.tts_options = dict(raw_options)

    if not isinstance(cfg.tts_provider_name, str) or not is_known_provider(cfg.tts_provider_name):
        raise ConfigError(
            "unknown_provider",
            f"Unknown tts_provider_name {cfg.tts_provider_name!r}. See GET /api/v1/providers.",
            "tts_provider_name",
        )
    cfg.tts_provider_name = canonical_name(cfg.tts_provider_name)

    if contain_output_dir:
        base = get_output_base()
        raw = os.path.expanduser(str(cfg.output_dir or ""))
        if not os.path.isabs(raw):
            raw = os.path.join(base, raw)
        resolved = os.path.realpath(raw)
        if not _is_inside(resolved, base):
            raise ConfigError(
                "output_dir_outside_base",
                "output_dir must resolve inside the server output base directory",
                "output_dir",
            )
        cfg.output_dir = resolved

    for name in _PATH_FIELDS:
        _require_regular_file(getattr(cfg, name, None), name)

    if cfg.tts_options:
        try:
            info = provider_info(cfg.tts_provider_name)
        except Exception as exc:
            raise ConfigError(
                "provider_unavailable",
                f"TTS provider '{cfg.tts_provider_name}' cannot be loaded on this server: {exc}",
                "tts_provider_name",
            ) from exc
        for option in info.options:
            if option.kind == "file":
                _require_regular_file(cfg.tts_options.get(option.key), f"tts_options.{option.key}")

    return cfg


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
    # Wall-clock times (seconds since the epoch).
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    chapter_count: int = 0

    # Synchronization
    cancel_token: CancelToken = field(default_factory=CancelToken)
    # Subscribers for this task's WebSocket events
    subscribers: List[asyncio.Queue] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.chapter_count:
            self.chapter_count = len(self.chapters or [])

    @property
    def book_title(self) -> str:
        """Title from the request config (``""`` when it had none)."""
        config = self.config_dict if isinstance(self.config_dict, dict) else {}
        return str(config.get("book_title") or "")

    @property
    def is_finished(self) -> bool:
        """True once the task is completed, failed or cancelled."""
        return self.status in _TERMINAL_STATUSES

    def summary(self) -> Dict[str, Any]:
        """Returns the task's metadata without logs or chapter text."""
        return {
            "task_id": self.task_id,
            "status": self.status,
            "progress": self.progress,
            "book_title": self.book_title,
            "chapter_count": self.chapter_count,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "output_file_count": len(self.output_files),
            "error_message": self.error_message,
        }

    async def add_log(self, text: str):
        self.logs.append(text)
        await self.broadcast({"type": "log", "message": text})

    async def set_progress(self, val: float):
        self.progress = val
        await self.broadcast({"type": "progress", "progress": val})

    async def update_status(self, new_status: str):
        self.status = new_status
        if new_status == "running" and self.started_at is None:
            self.started_at = time.time()
        if new_status in _TERMINAL_STATUSES:
            if self.finished_at is None:
                self.finished_at = time.time()
            # Up to _MAX_COMPLETED_TASKS finished tasks stay in memory; none
            # of them needs the book's full text any more.
            self.chapters = []
        await self.broadcast({"type": "status", "status": new_status})

    async def end_session(self, files: Optional[List[str]] = None, error: Optional[str] = None):
        """Broadcasts the final ``session_end`` event of this task.

        Sent after the terminal status (and after ``completed``), it is the
        signal WebSocket clients wait for before disconnecting.
        """
        event: Dict[str, Any] = {
            "type": "session_end",
            "task_id": self.task_id,
            "status": self.status,
            "files": list(files or []),
            "success": self.status == "completed",
            "cancelled": self.status == "cancelled",
        }
        if error:
            event["error"] = error
        await self.broadcast(event)

    async def broadcast(self, data: Dict[str, Any]):
        for sub in list(self.subscribers):
            try:
                sub.put_nowait(data)
            except Exception:
                pass


# Global task memory database & execution queue
tasks: Dict[str, Task] = {}
task_queue: asyncio.Queue = asyncio.Queue()


def evict_old_tasks(max_completed: int = _MAX_COMPLETED_TASKS) -> None:
    """Remove oldest completed/failed/cancelled tasks over the limit."""
    terminal = [
        tid for tid, t in tasks.items()
        if t.status in _TERMINAL_STATUSES
    ]
    if len(terminal) > max_completed:
        for tid in terminal[:-max_completed]:
            tasks.pop(tid, None)


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


from audiobook_factory.gpu_pool import GPUDetector, GPUPoolManager


def _get_active_gpu_count() -> int:
    """Returns the number of active GPU providers in the pool, or 1 if none loaded."""
    manager = GPUPoolManager.instance()
    pools = manager.all_pools()
    if not pools:
        try:
            import torch
            return max(1, torch.cuda.device_count() if torch.cuda.is_available() else 1)
        except ImportError:
            return 1
    return max(1, max(p.device_count for p in pools.values()))


class _TaskSlots:
    """Limits how many tasks run at once, with a limit that can change.

    The limit follows the number of devices in the active provider pool, which
    is only known once a pool exists and changes when the engine does. A
    semaphore cannot be resized, and replacing it while running tasks still
    hold the old one admits more pipelines than there are devices. This
    counts the running tasks itself and re-reads the limit on every
    admission, so the number running never exceeds the limit in force when a
    task is admitted; after the limit drops, nothing new starts until enough
    running tasks have finished.

    Must be created inside the event loop that uses it.
    """

    def __init__(self, capacity_fn: Callable[[], int]) -> None:
        self._capacity_fn = capacity_fn
        self._in_use: int = 0
        self._freed = asyncio.Event()

    @property
    def in_use(self) -> int:
        """Number of slots currently held."""
        return self._in_use

    def _capacity(self) -> int:
        try:
            return max(1, int(self._capacity_fn()))
        except Exception as exc:
            logger.warning("Could not read the task concurrency limit (%s); using 1.", exc)
            return 1

    async def acquire(self) -> None:
        """Waits until a slot is free under the current limit, then takes it."""
        while self._in_use >= self._capacity():
            self._freed.clear()
            try:
                await asyncio.wait_for(self._freed.wait(), timeout=_SLOT_RECHECK_SEC)
            except asyncio.TimeoutError:
                pass
        self._in_use += 1

    def release(self) -> None:
        """Returns a slot taken with :meth:`acquire`."""
        self._in_use = max(0, self._in_use - 1)
        self._freed.set()


def _queue_task_done() -> None:
    """Marks one queue item as processed, tolerating an unbalanced call.

    Tests and callers may run a task that never went through the queue.
    """
    try:
        task_queue.task_done()
    except ValueError:
        pass


async def _wait_for_other_providers_idle(
    cfg: AudiobookConfig, task: Task, timeout: float = 600.0
) -> None:
    """Waits for any other-provider GPU pool to finish before this task starts.

    GPUPoolManager keeps only one TTS provider's weights resident at a time
    (get_pool() evicts other providers to avoid OOM — see gpu_pool.py). The
    task slots in worker_loop() only limit concurrency by GPU *count*, not by
    which TTS engine each queued task wants, so two tasks requesting
    different engines can otherwise be scheduled at the same time and race on
    the same pool slot. Serialize provider switches here instead of letting
    that race surface as a confusing mid-run error.
    """
    if cfg.preview_mode:
        return
    manager = GPUPoolManager.instance()
    start = time.monotonic()
    announced = False
    while True:
        if task.cancel_token.is_cancelled or task.status == "cancelled":
            return
        pools = manager.all_pools()
        blocking = [
            name for name, pool in pools.items()
            if name != cfg.tts_provider_name and not pool.is_idle
        ]
        if not blocking:
            return
        if not announced:
            await task.add_log(
                f"⏳ Waiting for provider(s) {', '.join(blocking)} to finish before "
                f"loading '{cfg.tts_provider_name}' (TTS engines can't share GPU memory)."
            )
            announced = True
        if time.monotonic() - start > timeout:
            await task.add_log(
                "⚠️ Timed out waiting for the other TTS provider to free up; proceeding anyway."
            )
            return
        await asyncio.sleep(1.0)


async def _process_single_task(task_id: str, sem: Any | None = None) -> None:
    """Runs one task to a terminal state.

    Parameters
    ----------
    task_id : str
        Key of the task in ``tasks``.
    sem : Any | None
        Optional async context manager held for the duration of the run.
        ``worker_loop`` limits concurrency itself and passes nothing.
    """
    if sem is None:
        await _execute_task(task_id)
        return
    async with sem:
        await _execute_task(task_id)


async def _execute_task(task_id: str) -> None:
    task = tasks.get(task_id)
    if not task:
        _queue_task_done()
        return

    if task.status == "cancelled":
        # cancel_task() already announced the end of this task's session.
        _queue_task_done()
        return

    await task.update_status("running")
    await task.add_log(f"🚀 Starting generation task: {task_id}")

    try:
        # Security: nothing in the request body reaches the pipeline's
        # makedirs/os.remove/write sinks or its file readers unchecked. The
        # endpoint validates too; this covers tasks created any other way and
        # files that disappeared while the task was queued.
        cfg = prepare_config(task.config_dict)
        chapters = [
            ExtractedChapter(
                num=ch.get("num", idx + 1),
                title=ch.get("title", ""),
                text=ch.get("text", ""),
                sentences=ch.get("sentences", [])
            ) for idx, ch in enumerate(task.chapters)
        ]

        await _wait_for_other_providers_idle(cfg, task)
        if task.cancel_token.is_cancelled or task.status == "cancelled":
            await task.update_status("cancelled")
            await task.add_log("⛔ Generation task cancelled before provider acquisition.")
            await task.end_session()
            # task_done() runs in the `finally` below; calling it here too
            # raised "task_done() called too many times".
            return

        log_q = queue.Queue()
        prog_q = queue.Queue()

        loop = asyncio.get_running_loop()

        def run_sync_pipeline():
            # The provider pool is built (and the model loaded) in here, on
            # first use, for the engine this task asked for.
            return run_pipeline(cfg, chapters, log_q, prog_q, task.cancel_token)

        future = loop.run_in_executor(None, run_sync_pipeline)
        monitor = asyncio.create_task(monitor_task(task, log_q, prog_q, future))

        try:
            out_files = await future
        finally:
            await monitor

        if task.cancel_token.is_cancelled:
            await task.update_status("cancelled")
            await task.add_log("⛔ Generation task cancelled by user.")
            await task.end_session()
        else:
            task.output_files = list(out_files or [])
            await task.update_status("completed")
            await task.add_log(f"✅ Generation complete. Processed {len(task.output_files)} files.")
            await task.broadcast({"type": "completed", "files": task.output_files})
            await task.end_session(files=task.output_files)

    except Exception as e:
        import traceback
        err_msg = f"❌ Task crashed: {e}\n{traceback.format_exc()}"
        print(err_msg)
        task.error_message = str(e)
        await task.add_log(err_msg)
        await task.update_status("failed")
        await task.end_session(error=str(e))

    finally:
        _queue_task_done()


async def _run_task_safely(task_id: str, slots: Any | None = None) -> None:
    """Isolated task runner ensuring exceptions never crash the worker consumer loop.

    Releases the slot ``worker_loop`` took for this task however it ends.
    """
    try:
        await _process_single_task(task_id)
    except Exception as exc:
        logger.error("Unhandled exception in task runner for task %s: %s", task_id, exc)
    finally:
        if slots is not None:
            slots.release()


async def worker_loop() -> None:
    """
    Main background consumer queue loop executing generation tasks concurrently.

    Tasks start in queue order. The concurrency limit is re-evaluated before
    every start (see :class:`_TaskSlots`).
    """
    logger.info("[API Worker] Central task worker queue consumer started.")
    slots = _TaskSlots(_get_active_gpu_count)
    running: set[asyncio.Task] = set()
    while True:
        task_id = await task_queue.get()
        await slots.acquire()
        runner = asyncio.create_task(_run_task_safely(task_id, slots))
        # The loop only keeps weak references to tasks.
        running.add(runner)
        runner.add_done_callback(running.discard)
