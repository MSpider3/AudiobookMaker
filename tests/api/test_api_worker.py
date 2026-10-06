"""
test_api_worker.py
==================
Tests for the API worker: the concurrency limit that follows the GPU count,
task memory after completion, and a full task processed by the queue consumer.
"""

from __future__ import annotations

import asyncio
import os
import sys
import threading
import time
from collections import defaultdict

import pytest
from fastapi.testclient import TestClient

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import api.server as server_mod
import api.worker as worker_mod
from api.server import app
from api.worker import Task, _TaskSlots, tasks

_VOICE = os.path.join(_ROOT, "tests", "fixtures", "audio", "synthetic_voice_reference.wav")
_TEXT = "The lamp at the end of the pier burned all through the long night."


class _Recorder:
    """Stands in for _process_single_task: records how many run at once."""

    def __init__(self, capacity: dict, hold: float = 0.05):
        self.capacity = capacity
        self.hold = hold
        self.running = 0
        self.peak = 0
        self.over_limit: list[tuple[int, int]] = []
        self.started: list[str] = []
        self.finished: list[str] = []
        self.gates: dict[str, asyncio.Event] = {}

    async def __call__(self, task_id: str, sem=None) -> None:
        self.running += 1
        self.started.append(task_id)
        self.peak = max(self.peak, self.running)
        if self.running > self.capacity["value"]:
            self.over_limit.append((self.running, self.capacity["value"]))
        try:
            gate = self.gates.get(task_id)
            if gate is not None:
                await gate.wait()
            else:
                await asyncio.sleep(self.hold)
        finally:
            self.running -= 1
            self.finished.append(task_id)


async def _until(predicate, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        await asyncio.sleep(0.01)
    return predicate()


class TestTaskSlots:

    @pytest.mark.asyncio
    async def test_limit_can_rise_without_over_admitting(self, monkeypatch):
        """Estimate goes 1 → 2 while a task runs: the old code built a second
        semaphore of size 2, so three pipelines ran on two devices."""
        capacity = {"value": 1}
        recorder = _Recorder(capacity)
        for name in ("a", "b", "c", "d"):
            recorder.gates[name] = asyncio.Event()
        fresh_queue: asyncio.Queue = asyncio.Queue()
        monkeypatch.setattr(worker_mod, "task_queue", fresh_queue)
        monkeypatch.setattr(worker_mod, "_get_active_gpu_count", lambda: capacity["value"])
        monkeypatch.setattr(worker_mod, "_process_single_task", recorder)
        monkeypatch.setattr(worker_mod, "_SLOT_RECHECK_SEC", 0.02)

        loop_task = asyncio.create_task(worker_mod.worker_loop())
        try:
            for name in ("a", "b", "c", "d"):
                fresh_queue.put_nowait(name)
            assert await _until(lambda: recorder.started == ["a"])
            await asyncio.sleep(0.1)
            assert recorder.running == 1, "limit 1 admits one task"

            capacity["value"] = 2  # the pool now reports two devices
            assert await _until(lambda: recorder.running == 2)
            await asyncio.sleep(0.15)
            assert recorder.running == 2, "a third task started on two devices"
            assert recorder.started == ["a", "b"], "tasks start in queue order"

            recorder.gates["a"].set()
            assert await _until(lambda: recorder.started == ["a", "b", "c"])
            for name in ("b", "c", "d"):
                recorder.gates[name].set()
            assert await _until(lambda: len(recorder.finished) == 4)
        finally:
            loop_task.cancel()

        assert recorder.peak == 2
        assert recorder.over_limit == []

    @pytest.mark.asyncio
    async def test_limit_can_drop_while_tasks_run(self, monkeypatch):
        capacity = {"value": 2}
        recorder = _Recorder(capacity)
        for name in ("a", "b", "c"):
            recorder.gates[name] = asyncio.Event()
        fresh_queue: asyncio.Queue = asyncio.Queue()
        monkeypatch.setattr(worker_mod, "task_queue", fresh_queue)
        monkeypatch.setattr(worker_mod, "_get_active_gpu_count", lambda: capacity["value"])
        monkeypatch.setattr(worker_mod, "_process_single_task", recorder)
        monkeypatch.setattr(worker_mod, "_SLOT_RECHECK_SEC", 0.02)

        loop_task = asyncio.create_task(worker_mod.worker_loop())
        try:
            for name in ("a", "b", "c"):
                fresh_queue.put_nowait(name)
            assert await _until(lambda: recorder.running == 2)

            capacity["value"] = 1  # one device dropped out of the pool
            recorder.gates["a"].set()
            assert await _until(lambda: recorder.finished == ["a"])
            await asyncio.sleep(0.15)
            assert recorder.started == ["a", "b"], "one is still running: limit 1 admits nothing"

            recorder.gates["b"].set()
            assert await _until(lambda: recorder.started == ["a", "b", "c"])
            recorder.gates["c"].set()
            assert await _until(lambda: len(recorder.finished) == 3)
        finally:
            loop_task.cancel()

    @pytest.mark.asyncio
    async def test_many_tasks_never_exceed_the_limit(self, monkeypatch):
        capacity = {"value": 3}
        recorder = _Recorder(capacity, hold=0.02)
        fresh_queue: asyncio.Queue = asyncio.Queue()
        monkeypatch.setattr(worker_mod, "task_queue", fresh_queue)
        monkeypatch.setattr(worker_mod, "_get_active_gpu_count", lambda: capacity["value"])
        monkeypatch.setattr(worker_mod, "_process_single_task", recorder)
        monkeypatch.setattr(worker_mod, "_SLOT_RECHECK_SEC", 0.02)

        loop_task = asyncio.create_task(worker_mod.worker_loop())
        try:
            for index in range(20):
                fresh_queue.put_nowait(f"t{index}")
            assert await _until(lambda: len(recorder.finished) == 20, timeout=10)
        finally:
            loop_task.cancel()
        assert recorder.peak == 3
        assert recorder.over_limit == []
        assert recorder.started == [f"t{index}" for index in range(20)]

    @pytest.mark.asyncio
    async def test_slot_is_released_when_a_task_crashes(self, monkeypatch):
        calls = []

        async def _crash(task_id, sem=None):
            calls.append(task_id)
            raise RuntimeError("task blew up")

        fresh_queue: asyncio.Queue = asyncio.Queue()
        monkeypatch.setattr(worker_mod, "task_queue", fresh_queue)
        monkeypatch.setattr(worker_mod, "_get_active_gpu_count", lambda: 1)
        monkeypatch.setattr(worker_mod, "_process_single_task", _crash)
        loop_task = asyncio.create_task(worker_mod.worker_loop())
        try:
            for name in ("a", "b", "c"):
                fresh_queue.put_nowait(name)
            assert await _until(lambda: calls == ["a", "b", "c"]), "a crashed task kept its slot"
        finally:
            loop_task.cancel()

    @pytest.mark.asyncio
    async def test_slots_count_and_floor(self):
        slots = _TaskSlots(lambda: 0)  # a nonsense estimate still admits one task
        await slots.acquire()
        assert slots.in_use == 1
        waiter = asyncio.create_task(slots.acquire())
        await asyncio.sleep(0.05)
        assert not waiter.done()
        slots.release()
        await asyncio.wait_for(waiter, timeout=2)
        slots.release()
        slots.release()  # an extra release never goes negative
        assert slots.in_use == 0


class TestTaskMemory:

    @pytest.mark.asyncio
    @pytest.mark.parametrize("status", ["completed", "failed", "cancelled"])
    async def test_chapter_text_is_dropped_at_a_terminal_state(self, status):
        task = Task(task_id="mem", config_dict={}, chapters=[{"num": 1, "title": "One", "text": _TEXT * 50}])
        assert task.chapter_count == 1
        await task.update_status("running")
        assert task.chapters, "the text is needed until the task ends"
        assert task.started_at is not None and task.finished_at is None

        await task.update_status(status)
        assert task.chapters == []
        assert task.chapter_count == 1
        assert task.finished_at is not None and task.is_finished

    @pytest.mark.asyncio
    async def test_failed_validation_ends_the_session_and_frees_the_text(self):
        task = Task(
            task_id="mem-invalid",
            config_dict={"tts_provider_name": "mock", "voice_file": "/dev/zero"},
            chapters=[{"num": 1, "title": "One", "text": _TEXT}],
        )
        tasks["mem-invalid"] = task
        subscriber: asyncio.Queue = asyncio.Queue()
        task.subscribers.append(subscriber)
        try:
            await worker_mod._process_single_task("mem-invalid")
        finally:
            tasks.pop("mem-invalid", None)

        assert task.status == "failed"
        assert "voice_file" in (task.error_message or "")
        assert task.chapters == []
        events = []
        while not subscriber.empty():
            events.append(subscriber.get_nowait())
        assert events[-1]["type"] == "session_end"
        assert events[-1]["status"] == "failed" and events[-1]["success"] is False
        assert "voice_file" in events[-1]["error"]


class TestProcessedByTheQueueConsumer:

    def test_task_completes_frees_its_text_and_serves_its_files(self, monkeypatch, tmp_path):
        monkeypatch.setenv("ABM_SKIP_GPU_WARMUP", "1")
        monkeypatch.delenv("ABM_API_SECRET", raising=False)
        monkeypatch.setenv("ABM_OUTPUT_BASE", str(tmp_path))
        monkeypatch.setattr(server_mod._generate_limiter, "_requests", defaultdict(list))
        # The module-level queue stays bound to whichever event loop used it
        # first, so give this TestClient's loop a queue of its own.
        fresh_queue: asyncio.Queue = asyncio.Queue()
        monkeypatch.setattr(worker_mod, "task_queue", fresh_queue)
        monkeypatch.setattr(server_mod, "task_queue", fresh_queue)

        payload = {
            "config": {
                "book_title": "Consumer Book",
                "output_dir": "consumer_book",
                "output_format": "mp3",
                "tts_provider_name": "mock",
                "voice_file": _VOICE,
                "export_lrc": False,
            },
            "chapters": [{"num": 1, "title": "Chapter 1", "text": _TEXT, "sentences": [_TEXT]}],
        }
        with TestClient(app) as client:
            task_id = client.post("/api/v1/generate", json=payload).json()["task_id"]
            # Never hang the suite: a stalled task is cancelled, which fails
            # the assertions below instead.
            watchdog = threading.Timer(60.0, lambda: client.post(f"/api/v1/tasks/{task_id}/cancel"))
            watchdog.daemon = True
            watchdog.start()
            try:
                deadline = time.monotonic() + 90
                status = {}
                seen = 0
                lines: list[str] = []
                while time.monotonic() < deadline:
                    status = client.get(f"/api/v1/tasks/{task_id}?since={seen}").json()
                    lines.extend(status["logs"])
                    seen = status["log_count"]
                    if status["status"] in ("completed", "failed", "cancelled"):
                        break
                    time.sleep(0.1)
            finally:
                watchdog.cancel()

            assert status["status"] == "completed", status
            assert len(lines) == seen == len(tasks[task_id].logs), "since= must neither repeat nor skip lines"
            assert any("Generation complete" in line for line in lines)
            assert status["started_at"] and status["finished_at"] >= status["started_at"]
            assert tasks[task_id].chapters == [], "finished tasks must not keep chapter text"

            listed = {t["task_id"]: t for t in client.get("/api/v1/tasks").json()["tasks"]}[task_id]
            assert listed["book_title"] == "Consumer Book"
            assert listed["output_file_count"] == 1 and listed["chapter_count"] == 1

            assert len(status["output_files"]) == 1
            download = client.get(f"/api/v1/tasks/{task_id}/files/0")
            assert download.status_code == 200
            with open(status["output_files"][0], "rb") as fh:
                assert download.content == fh.read()
            assert download.headers["content-type"] == "audio/mpeg"
            assert client.get(f"/api/v1/tasks/{task_id}/files/1").status_code == 404
        tasks.pop(task_id, None)
