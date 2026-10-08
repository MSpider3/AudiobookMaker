"""
test_parallel_auto.py
=====================
``parallel_mode="auto"``: each chapter gets as many GPUs as it has batches
to fill (``_gpus_wanted`` / ``_run_chapters_auto`` in ``pipeline.py``).
"""

from __future__ import annotations

import os
import queue
import sys
import tempfile
import threading
import time

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import audiobook_factory.pipeline as pipeline_module
from audiobook_factory.gpu_pool import GPUPoolManager, ProviderPool
from audiobook_factory.pipeline import (
    AudiobookConfig, CancelToken, _gpus_wanted, _run_chapters_auto, _validate_config, run_pipeline,
)
from audiobook_factory.text_extractor import ExtractedChapter
from tests.fixtures.mock_provider import MockTTSProvider

_SENTENCE: str = "The tide went out at dawn, as it had every morning since the lighthouse was built."


def _chapter(num: int, paragraphs: int) -> ExtractedChapter:
    text = "\n\n".join(f"{_SENTENCE} Paragraph {index} of chapter {num}." for index in range(paragraphs))
    return ExtractedChapter(num=num, title=f"Part {num}", text=text, sentences=[])


class TestGpusWanted:

    def test_chapter_that_fits_one_batch_wants_one_gpu(self):
        config = AudiobookConfig()
        assert _gpus_wanted(_chapter(1, 6), config, batch_size=8, device_count=2) == 1
        assert _gpus_wanted(_chapter(1, 8), config, batch_size=8, device_count=2) == 1

    def test_one_gpu_per_batch_up_to_the_gpus_there_are(self):
        config = AudiobookConfig()
        assert _gpus_wanted(_chapter(1, 9), config, batch_size=8, device_count=2) == 2
        assert _gpus_wanted(_chapter(1, 17), config, batch_size=8, device_count=4) == 3
        assert _gpus_wanted(_chapter(1, 80), config, batch_size=8, device_count=4) == 4

    def test_engine_that_speaks_one_chunk_at_a_time_shares_every_chapter(self):
        # Batch size 1: every chunk is its own round, so more GPUs always help.
        config = AudiobookConfig()
        assert _gpus_wanted(_chapter(1, 2), config, batch_size=1, device_count=2) == 2
        assert _gpus_wanted(_chapter(1, 1), config, batch_size=1, device_count=2) == 1

    def test_empty_chapter_still_gets_a_gpu(self):
        empty = ExtractedChapter(num=1, title="Empty", text="", sentences=[])
        assert _gpus_wanted(empty, AudiobookConfig(), batch_size=8, device_count=2) == 1


class _Recorder:
    """Stands in for the pipeline's per-chapter function: records who ran what, and when."""

    def __init__(self, seconds: dict[int, float]) -> None:
        self.seconds = seconds
        self.events: list[tuple[str, int, object]] = []
        self.lock = threading.Lock()
        self.busy: set[str] = set()
        self.overlap = False

    def __call__(self, task, pinned) -> None:
        num = task[0]
        devices = self.all_devices if pinned is None else [pinned] if isinstance(pinned, str) else list(pinned)
        with self.lock:
            if self.busy & set(devices):
                self.overlap = True           # a GPU given to two chapters at once
            self.busy |= set(devices)
            self.events.append(("start", num, pinned))
        time.sleep(self.seconds.get(num, 0.05))
        with self.lock:
            self.busy -= set(devices)
            self.events.append(("end", num, pinned))

    def order(self, kind: str) -> list[int]:
        return [num for event, num, _ in self.events if event == kind]

    def pinned(self, num: int):
        return next(pinned for event, n, pinned in self.events if event == "start" and n == num)


def _schedule(wanted: dict[int, int], devices: list[str], seconds: dict[int, float] | None = None) -> _Recorder:
    recorder = _Recorder(seconds or {})
    recorder.all_devices = devices
    tasks = [(num, _chapter(num, 1)) for num in wanted]
    _run_chapters_auto(tasks, devices, wanted, recorder, CancelToken(), lambda message: None)
    return recorder


class TestAutoScheduling:

    def test_short_long_short_runs_the_short_ones_side_by_side_first(self):
        # The case from the design discussion: chapters 1 and 3 short, 2 long.
        recorder = _schedule({1: 1, 2: 2, 3: 1}, ["cuda:0", "cuda:1"], {1: 0.2, 2: 0.2, 3: 0.2})
        assert set(recorder.order("start")[:2]) == {1, 3}
        assert recorder.order("start")[2] == 2
        assert {recorder.pinned(1), recorder.pinned(3)} == {"cuda:0", "cuda:1"}
        assert recorder.pinned(2) is None          # the long chapter gets every GPU
        assert recorder.overlap is False
        # The long chapter starts only after both short ones are done.
        events = [(event, num) for event, num, _ in recorder.events]
        assert events.index(("start", 2)) > events.index(("end", 1))
        assert events.index(("start", 2)) > events.index(("end", 3))

    def test_all_short_chapters_run_one_per_gpu(self):
        recorder = _schedule({n: 1 for n in range(1, 7)}, ["cuda:0", "cuda:1"])
        assert sorted(recorder.order("end")) == [1, 2, 3, 4, 5, 6]
        assert all(isinstance(recorder.pinned(n), str) for n in range(1, 7))
        assert recorder.overlap is False

    def test_all_long_chapters_run_one_at_a_time_on_every_gpu(self):
        recorder = _schedule({1: 2, 2: 2, 3: 2}, ["cuda:0", "cuda:1"])
        assert recorder.order("start") == [1, 2, 3]
        assert all(recorder.pinned(n) is None for n in (1, 2, 3))
        assert recorder.overlap is False

    def test_four_gpus_split_between_a_medium_chapter_and_short_ones(self):
        devices = ["cuda:0", "cuda:1", "cuda:2", "cuda:3"]
        recorder = _schedule({1: 2, 2: 1, 3: 1, 4: 4}, devices, {1: 0.3, 2: 0.3, 3: 0.3, 4: 0.1})
        assert recorder.pinned(1) == ("cuda:0", "cuda:1")
        assert {recorder.pinned(2), recorder.pinned(3)} == {"cuda:2", "cuda:3"}
        assert recorder.pinned(4) is None
        assert recorder.order("start")[-1] == 4
        assert recorder.overlap is False

    def test_every_chapter_runs_exactly_once_whatever_the_mix(self):
        wanted = {n: (n % 3) + 1 for n in range(1, 16)}
        recorder = _schedule(wanted, ["cuda:0", "cuda:1", "cuda:2"])
        assert sorted(recorder.order("start")) == list(range(1, 16))
        assert sorted(recorder.order("end")) == list(range(1, 16))
        assert recorder.overlap is False

    def test_failing_chapter_frees_its_gpus_for_the_rest(self):
        messages: list[str] = []

        def process(task, pinned):
            if task[0] == 1:
                raise RuntimeError("chapter 1 broke")

        tasks = [(n, _chapter(n, 1)) for n in (1, 2, 3)]
        done = threading.Thread(
            target=_run_chapters_auto,
            args=(tasks, ["cuda:0", "cuda:1"], {1: 2, 2: 2, 3: 1}, process, CancelToken(), messages.append),
        )
        done.start()
        done.join(timeout=10)
        assert not done.is_alive(), "the scheduler hung after a chapter failed"
        assert any("chapter 1 broke" in message for message in messages)

    def test_cancel_stops_handing_out_chapters(self):
        cancel = CancelToken()
        started: list[int] = []

        def process(task, pinned):
            started.append(task[0])
            cancel.cancel()

        _run_chapters_auto([(n, _chapter(n, 1)) for n in range(1, 9)], ["cuda:0"], {n: 1 for n in range(1, 9)},
                           process, cancel, lambda message: None)
        assert len(started) < 8


class TestAutoModeEndToEnd:
    """The whole pipeline on the mock engine with simulated GPUs."""

    @pytest.fixture
    def devices(self, monkeypatch):
        names = ["dev0", "dev1"]
        manager = GPUPoolManager.instance()
        config = AudiobookConfig(tts_provider_name="mock")
        manager._pools["mock"] = ProviderPool(lambda device: MockTTSProvider(config, device=device), names, "mock")
        monkeypatch.setattr(pipeline_module, "_chapter_batch_size", lambda pool, config: 4)
        yield names
        manager._pools.pop("mock", None)

    def _run(self, chapters, **settings):
        with tempfile.TemporaryDirectory() as out:
            voice = os.path.join(_ROOT, "tests", "fixtures", "audio", "synthetic_voice_reference.wav")
            config = AudiobookConfig(
                tts_provider_name="mock", output_dir=out, output_format="mp3", voice_file=voice,
                parallel_mode="auto", max_chapter_retries=0, retry_failed_at_end=False,
                export_lrc=False, verify_chunks="off", **settings,
            )
            logs: queue.Queue = queue.Queue()
            files = run_pipeline(config, chapters, logs, queue.Queue())
            messages = []
            while not logs.empty():
                messages.append(logs.get())
            return [os.path.basename(path) for path in files], messages

    def test_mixed_book_is_narrated_completely_and_in_order(self, devices):
        chapters = [_chapter(1, 3), _chapter(2, 12), _chapter(3, 3), _chapter(4, 2)]
        files, messages = self._run(chapters)
        assert files == [f"Chapter {n} - Part {n}.mp3" for n in (1, 2, 3, 4)]
        summary = next(m for m in messages if "Automatic GPU sharing" in m)
        assert "3 short chapter(s) take one GPU each" in summary and "1 longer one(s)" in summary
        # The long chapter was shared by both devices.
        assert any("on 2 device(s): dev0, dev1" in m for m in messages)
        # The short ones each ran on a single device.
        assert sum("on 1 device(s)" in m for m in messages) == 3

    def test_single_file_book_keeps_chapter_order(self, devices):
        chapters = [_chapter(1, 2), _chapter(2, 10), _chapter(3, 2)]
        files, _messages = self._run(chapters, single_file_mode=True, book_title="Auto Mode Book")
        assert len(files) == 1 and files[0].endswith(".mp3")


def test_auto_is_a_valid_parallel_mode():
    _validate_config(AudiobookConfig(parallel_mode="auto"))
    with pytest.raises(ValueError, match="parallel_mode"):
        _validate_config(AudiobookConfig(parallel_mode="everything"))
    assert AudiobookConfig().parallel_mode == "chunks"   # the tested default is unchanged
