"""
test_bounded_and_strict.py
==========================
* Mastering stops on a chunk or partial file it cannot read, instead of
  dropping that audio from the chapter (Rust and Python paths).
* A backend task keeps a bounded log and each WebSocket client a bounded
  event queue, without breaking ``?since=`` log offsets.
* Importing the library does not download NLTK data.
"""

from __future__ import annotations

import asyncio
import os
import subprocess
import sys

import numpy as np
import pytest
import soundfile as sf

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import api.worker as worker_mod
from audiobook_factory.chapter_pipeline import _concat_partial, _master_final
from audiobook_factory.pipeline import AudiobookConfig


def _tone(path: str, seconds: float = 0.5, rate: int = 24000) -> str:
    sf.write(path, (0.2 * np.sin(np.linspace(0, 400 * seconds, int(rate * seconds)))).astype(np.float32), rate)
    return path


class TestMasteringIsStrict:

    def test_python_concat_stops_on_a_missing_chunk(self, tmp_path):
        good = _tone(str(tmp_path / "a.wav"))
        with pytest.raises(RuntimeError, match="missing"):
            _concat_partial([good, str(tmp_path / "gone.wav")], str(tmp_path / "p.wav"), AudiobookConfig())

    def test_python_concat_stops_on_an_unreadable_chunk(self, tmp_path):
        good = _tone(str(tmp_path / "a.wav"))
        bad = tmp_path / "bad.wav"
        bad.write_bytes(b"RIFF" + b"\0" * 200)
        with pytest.raises(RuntimeError, match="Failed to read chunk"):
            _concat_partial([good, str(bad)], str(tmp_path / "p.wav"), AudiobookConfig())

    def test_final_master_stops_on_a_missing_partial(self, tmp_path):
        good = _tone(str(tmp_path / "partial_0.wav"))
        with pytest.raises(RuntimeError, match="missing"):
            _master_final([good, str(tmp_path / "partial_1.wav")], str(tmp_path / "out.wav"), AudiobookConfig())

    def test_rust_master_stops_on_an_unreadable_file(self, tmp_path):
        rust = pytest.importorskip("audiobook_rust")
        if not hasattr(rust, "master_audio"):
            pytest.skip("Rust extension without master_audio")
        good = _tone(str(tmp_path / "a.wav"))
        bad = tmp_path / "bad.wav"
        bad.write_bytes(b"not a wav file at all" * 10)
        with pytest.raises(Exception, match="bad.wav"):
            rust.master_audio([good, str(bad)], str(tmp_path / "out.wav"), 0.0, 24000, -18.0, -1.5, 64)
        # Readable files still master, with no pause after the last one.
        rust.master_audio([good, good], str(tmp_path / "ok.wav"), 0.5, 24000, -18.0, -1.5, 64)
        assert sf.info(str(tmp_path / "ok.wav")).duration == pytest.approx(1.5, abs=0.01)


class TestBoundedBackendMemory:

    def test_log_is_trimmed_and_offsets_stay_valid(self, monkeypatch):
        monkeypatch.setattr(worker_mod, "_MAX_TASK_LOG_LINES", 100)
        monkeypatch.setattr(worker_mod, "_LOG_TRIM_SLACK", 10)
        task = worker_mod.Task(task_id="t-log", config_dict={}, chapters=[])

        async def _fill():
            for index in range(250):
                await task.add_log(f"line {index}")

        asyncio.run(_fill())
        assert len(task.logs) <= 110
        assert task.log_base + len(task.logs) == 250
        assert task.logs[-1] == "line 249"

        import api.server as server_mod
        monkeypatch.setitem(worker_mod.tasks, "t-log", task)
        body = asyncio.run(server_mod.get_task_status("t-log", since=5))
        # Line 5 was dropped: the reply starts at the oldest line still kept.
        assert body["log_offset"] == task.log_base
        assert body["logs"][0] == f"line {task.log_base}"
        assert body["log_count"] == 250
        tail = asyncio.run(server_mod.get_task_status("t-log", since=245))
        assert tail["logs"] == [f"line {i}" for i in range(245, 250)]

    def test_log_events_carry_their_line_number(self, monkeypatch):
        monkeypatch.setattr(worker_mod, "_MAX_TASK_LOG_LINES", 5)
        monkeypatch.setattr(worker_mod, "_LOG_TRIM_SLACK", 1)
        task = worker_mod.Task(task_id="t-idx", config_dict={}, chapters=[])

        async def _run():
            queue: asyncio.Queue = asyncio.Queue()
            task.subscribers.append(queue)
            for index in range(12):
                await task.add_log(f"line {index}")
            return [queue.get_nowait()["index"] for _ in range(queue.qsize())]

        assert asyncio.run(_run()) == list(range(12))

    def test_slow_websocket_client_keeps_the_newest_events(self):
        task = worker_mod.Task(task_id="t-ws", config_dict={}, chapters=[])

        async def _run():
            queue: asyncio.Queue = asyncio.Queue(maxsize=3)
            task.subscribers.append(queue)
            for index in range(10):
                await task.broadcast({"type": "log", "message": str(index)})
            await task.broadcast({"type": "session_end"})
            return [queue.get_nowait() for _ in range(queue.qsize())]

        events = asyncio.run(_run())
        assert len(events) == 3
        assert events[-1] == {"type": "session_end"}


def test_importing_the_library_does_not_download_nltk_data():
    code = (
        "import nltk\n"
        "def _no(*a, **k): raise SystemExit('nltk.download called at import')\n"
        "nltk.download = _no\n"
        "import audiobook_factory.pipeline, audiobook_factory.text_processing\n"
    )
    result = subprocess.run([sys.executable, "-c", code], cwd=_ROOT, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stderr[-2000:]
