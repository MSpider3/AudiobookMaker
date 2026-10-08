"""
test_work_share.py
==================
How Stage B workers divide a chapter's chunks between GPUs
(``_WorkShare`` and ``_stage_b_device_worker`` in ``chapter_pipeline.py``).
"""

from __future__ import annotations

import os
import queue
import sys
import threading
import time


_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import audiobook_factory.chapter_pipeline as chapter_pipeline
from audiobook_factory.chapter_pipeline import _ProgressState, _WorkShare, _stage_b_device_worker
from audiobook_factory.pipeline import AudiobookConfig, CancelToken


def _queue_of(count: int) -> queue.Queue:
    work: queue.Queue = queue.Queue()
    for index in range(count):
        work.put((index, f"chunk {index}"))
    return work


class TestWorkShare:

    def test_short_chapter_is_split_between_two_devices(self):
        # The Kaggle run: 6 chunks, batch size 8, two T4s - one GPU took all six.
        share, work = _WorkShare(workers=2), _queue_of(6)
        first = share.take(work, batch_size=8)
        second = share.take(work, batch_size=8)
        assert len(first) == 3 and len(second) == 3
        assert work.empty()

    def test_free_device_takes_full_batches_while_the_other_is_busy(self):
        share, work = _WorkShare(workers=2), _queue_of(20)
        assert len(share.take(work, batch_size=8)) == 8   # 20 queued, two free: up to 10 each
        assert len(share.take(work, batch_size=8)) == 8   # the other device is busy now
        share.finish()                                    # one device finishes its batch
        assert len(share.take(work, batch_size=8)) == 4   # and takes the whole tail

    def test_single_device_takes_full_batches(self):
        share, work = _WorkShare(workers=1), _queue_of(10)
        assert len(share.take(work, batch_size=4)) == 4
        assert len(share.take(work, batch_size=8)) == 6

    def test_long_chapter_is_not_slowed_down(self):
        share, work = _WorkShare(workers=2), _queue_of(100)
        assert len(share.take(work, batch_size=8)) == 8

    def test_three_devices_split_a_short_chapter_evenly(self):
        share, work = _WorkShare(workers=3), _queue_of(7)
        sizes = [len(share.take(work, batch_size=8)) for _ in range(3)]
        assert sizes == [3, 2, 2]

    def test_every_chunk_is_taken_exactly_once(self):
        share, work = _WorkShare(workers=3), _queue_of(17)
        taken = []
        while True:
            batch = share.take(work, batch_size=4)
            if not batch:
                break
            taken.extend(index for index, _ in batch)
            share.finish()
        assert sorted(taken) == list(range(17))

    def test_busy_flag_follows_take_and_finish(self):
        share, work = _WorkShare(workers=2), _queue_of(2)
        assert not share.others_busy
        assert share.take(work, batch_size=1)
        assert share.others_busy
        share.finish()
        assert not share.others_busy
        assert share.take(_queue_of(0), batch_size=4) == []
        assert not share.others_busy  # an empty grab is not a batch in progress

    def test_retired_worker_leaves_everything_to_the_rest(self):
        share, work = _WorkShare(workers=2), _queue_of(6)
        share.retire()
        assert len(share.take(work, batch_size=8)) == 6


class _Recorder:
    """Stands in for ``_synthesize_batch``: records who synthesized what."""

    def __init__(self, failing_device: str | None = None, seconds_per_batch: float = 0.05) -> None:
        self.failing_device = failing_device
        self.seconds_per_batch = seconds_per_batch
        self.by_device: dict[str, list[int]] = {}
        self.lock = threading.Lock()

    def __call__(self, batch, provider, voice_ref, config, out_dir, master_queue,
                 cancel_token, progress_state, chapter_index, verifier,
                 chunk_completed_cb, note_written, chunk_flagged_cb):
        time.sleep(self.seconds_per_batch)
        if provider.device == self.failing_device:
            raise RuntimeError(f"{provider.device} broke")
        indices = [index for index, _ in batch]
        with self.lock:
            self.by_device.setdefault(provider.device, []).extend(indices)
        note_written(indices)


class _Provider:
    batch_size_limit = None

    def __init__(self, device: str) -> None:
        self.device = device


def _run_workers(monkeypatch, recorder: _Recorder, chunk_count: int, devices=("cuda:0", "cuda:1")):
    monkeypatch.setattr(chapter_pipeline, "_synthesize_batch", recorder)
    monkeypatch.setattr(chapter_pipeline, "_batch_size_for", lambda device, provider, config: 8)
    work, master = _queue_of(chunk_count), queue.Queue()
    errors: list[BaseException] = []
    stats: dict[str, int] = {device: 0 for device in devices}
    share = _WorkShare(workers=len(devices))
    threads = [
        threading.Thread(target=_stage_b_device_worker, args=(
            device, _Provider(device), work, master, b"", AudiobookConfig(), "", CancelToken(),
            _ProgressState(total=chunk_count, callback=None), 1, errors, None, None, None, None,
            stats, share,
        ))
        for device in devices
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=20)
        assert not thread.is_alive(), "a Stage B worker did not stop"
    sentinels = 0
    while not master.empty():
        sentinels += master.get() is None
    return work, errors, stats, sentinels


class TestStageBWorkers:

    def test_both_devices_work_on_a_short_chapter(self, monkeypatch):
        recorder = _Recorder()
        work, errors, stats, sentinels = _run_workers(monkeypatch, recorder, chunk_count=6)
        assert not errors and work.empty() and sentinels == 2
        assert sorted(recorder.by_device["cuda:0"] + recorder.by_device["cuda:1"]) == list(range(6))
        assert stats["cuda:0"] >= 1 and stats["cuda:1"] >= 1

    def test_surviving_device_finishes_the_work_of_a_failed_one(self, monkeypatch):
        # cuda:1 used to exit as soon as the queue looked empty, so the chunks
        # cuda:0 handed back were never synthesized and the chapter failed.
        recorder = _Recorder(failing_device="cuda:0", seconds_per_batch=0.2)
        work, errors, stats, sentinels = _run_workers(monkeypatch, recorder, chunk_count=6)
        assert len(errors) == 1 and "cuda:0 broke" in str(errors[0])
        assert work.empty() and sentinels == 2
        assert sorted(recorder.by_device["cuda:1"]) == list(range(6))
        assert stats == {"cuda:0": 0, "cuda:1": 6}

    def test_chunks_stay_queued_when_every_device_fails(self, monkeypatch):
        class _AlwaysFails(_Recorder):
            def __call__(self, batch, provider, *args):
                raise RuntimeError(f"{provider.device} broke")

        work, errors, stats, sentinels = _run_workers(monkeypatch, _AlwaysFails(), chunk_count=4)
        assert len(errors) == 2 and sentinels == 2
        assert work.qsize() == 4  # nothing lost: the chapter reports them as missing

    def test_single_device_behaves_as_before(self, monkeypatch):
        recorder = _Recorder()
        work, errors, stats, sentinels = _run_workers(
            monkeypatch, recorder, chunk_count=5, devices=("cuda:0",)
        )
        assert not errors and work.empty() and sentinels == 1
        assert sorted(recorder.by_device["cuda:0"]) == list(range(5))
