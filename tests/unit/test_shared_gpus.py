"""
test_shared_gpus.py
===================
Several jobs, previews and processes sharing the same GPUs and files.

* Two jobs on one two-GPU pool used to deadlock: each chapter took its GPUs
  one at a time, so job A could hold GPU 0 and job B GPU 1, both waiting
  forever for the other (``ProviderPool.acquire_devices``).
* A chapter that fell back to another GPU kept the pinned GPU's name.
* Voice previews loaded a second model next to a running job, and two
  previews could call one model from two threads (``preview_tts``).
* One shared Whisper made every GPU wait for the others' transcripts
  (``ChunkVerifier``).
* The progress file lock only worked inside one process (``progress_io``).
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import sys
import tempfile
import threading
import time

import numpy as np
import pytest
import soundfile as sf

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import audiobook_factory.pipeline as pipeline_mod
from audiobook_factory.chapter_pipeline import run_chapter_pipeline
from audiobook_factory.chunk_verifier import ChunkVerifier
from audiobook_factory.gpu_pool import GPUPoolManager, ProviderPool
from audiobook_factory.pipeline import AudiobookConfig, CancelToken
from audiobook_factory.progress_io import read_progress_file, write_progress_file
from tests.fixtures.mock_provider import MockTTSProvider

_DEVICES = ["cuda:0", "cuda:1"]


@pytest.fixture(autouse=True)
def _isolated_manager():
    GPUPoolManager._instance = None
    pipeline_mod._cleanup_preview_provider()
    yield
    try:
        GPUPoolManager.instance().shutdown()
    except Exception:
        pass
    GPUPoolManager._instance = None
    pipeline_mod._cleanup_preview_provider()


def _pool(config: AudiobookConfig, delay: float = 0.0) -> ProviderPool:
    def _factory(device: str) -> MockTTSProvider:
        provider = MockTTSProvider(config, device=device)
        provider.simulated_delay = delay
        return provider

    return ProviderPool(_factory, list(_DEVICES), "mock")


def _voice() -> bytes:
    with open(os.path.join(_ROOT, "tests", "fixtures", "audio", "synthetic_voice_reference.wav"), "rb") as fh:
        return fh.read()


class TestTakingSeveralGpus:

    def test_a_multi_gpu_take_holds_nothing_while_it_waits(self):
        pool = _pool(AudiobookConfig())
        held = pool.acquire(preferred_device="cuda:1")
        taken: dict = {}
        waiter = threading.Thread(target=lambda: taken.update(pool.acquire_devices(list(_DEVICES))))
        waiter.start()
        time.sleep(0.3)
        # Still waiting for GPU 1, and GPU 0 is left free for anyone else.
        assert waiter.is_alive() and not taken
        assert pool._device_queues["cuda:0"].full()
        pool.release(held)
        waiter.join(timeout=5)
        assert set(taken) == set(_DEVICES)

    def test_a_cancelled_wait_gives_up(self):
        pool = _pool(AudiobookConfig())
        held = pool.acquire(preferred_device="cuda:0")
        cancel = CancelToken()
        threading.Timer(0.2, cancel.cancel).start()
        with pytest.raises(BaseException, match="cancel"):
            pool.acquire_devices(list(_DEVICES), cancel_token=cancel)
        pool.release(held)
        assert pool.is_idle

    def test_unknown_device_is_an_error(self):
        with pytest.raises(ValueError):
            _pool(AudiobookConfig()).acquire_devices(["cuda:7"])

    def test_two_jobs_on_one_pool_both_finish(self):
        with tempfile.TemporaryDirectory() as td:
            config = AudiobookConfig(output_dir=td, sample_rate=24000, max_len=100, pause=0.1, batch_size=2)
            pool = _pool(config, delay=0.05)
            voice = _voice()
            start = threading.Barrier(2)
            # Force the bad timing: both jobs take one GPU before either takes
            # a second. A pool that hands out GPUs one at a time deadlocks here.
            first_take = threading.Barrier(2)
            single_take = pool.acquire

            def _one_at_a_time(*args, **kwargs):
                provider = single_take(*args, **kwargs)
                try:
                    first_take.wait(timeout=2)
                except threading.BrokenBarrierError:
                    pass
                return provider

            pool.acquire = _one_at_a_time
            finished: list[int] = []
            errors: list[BaseException] = []

            def _job(number: int) -> None:
                out_dir = os.path.join(td, f"job{number}")
                start.wait()
                try:
                    run_chapter_pipeline(
                        sentences=[f"Job {number} sentence {i} has a few words." for i in range(6)],
                        voice_ref=voice, out_wav_path=os.path.join(out_dir, "ch.wav"), out_dir=out_dir,
                        chapter_index=1, config=config, pool=pool, cancel_token=CancelToken(),
                        log_callback=lambda message: None,
                    )
                    finished.append(number)
                except BaseException as exc:  # noqa: BLE001 - reported below
                    errors.append(exc)

            threads = [threading.Thread(target=_job, args=(n,), daemon=True) for n in (1, 2)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(timeout=30)
            assert not errors, errors
            assert sorted(finished) == [1, 2], "a job is stuck waiting for a GPU"
            assert pool.is_idle

    def test_fallback_gpu_is_named_by_the_gpu_it_really_is(self):
        with tempfile.TemporaryDirectory() as td:
            config = AudiobookConfig(output_dir=td, sample_rate=24000, max_len=100, pause=0.1, batch_size=2)
            pool = _pool(config)
            held = pool.acquire(preferred_device="cuda:0")
            logs: list[str] = []
            run_chapter_pipeline(
                sentences=["One short sentence.", "Another short sentence."],
                voice_ref=_voice(), out_wav_path=os.path.join(td, "ch.wav"), out_dir=td,
                chapter_index=3, config=config, pool=pool, cancel_token=CancelToken(),
                log_callback=logs.append, pinned_device="cuda:0",
            )
            pool.release(held)
            line = next(message for message in logs if "Synthesizing" in message)
            assert line.endswith("cuda:1...")


class TestPreviewNextToAJob:

    def _register(self, pool: ProviderPool) -> None:
        GPUPoolManager.instance()._pools["mock"] = pool

    def test_preview_is_refused_while_a_job_holds_the_gpus(self):
        pool = _pool(AudiobookConfig())
        self._register(pool)
        held = pool.acquire_devices(list(_DEVICES))
        try:
            with pytest.raises(RuntimeError, match="being generated"):
                pipeline_mod.preview_tts("Hello there.", AudiobookConfig(tts_provider_name="mock"))
        finally:
            for provider in held.values():
                pool.release(provider)

    def test_preview_borrows_the_idle_job_model(self, monkeypatch):
        pool = _pool(AudiobookConfig())
        self._register(pool)
        monkeypatch.setattr(
            "audiobook_factory.tts_providers.get_tts_provider",
            lambda *a, **k: pytest.fail("a second model was loaded"),
        )
        wav = pipeline_mod.preview_tts("Hello there.", AudiobookConfig(tts_provider_name="mock"))
        assert wav and wav[:4] == b"RIFF"
        assert pool.is_idle

    def test_preview_gives_up_politely_when_the_gpus_stay_taken(self, monkeypatch):
        # The pool looks idle when checked, then a job takes every GPU.
        pool = _pool(AudiobookConfig())
        self._register(pool)
        held = pool.acquire_devices(list(_DEVICES))
        monkeypatch.setattr(pipeline_mod, "_pool_for_preview", lambda name: pool)
        monkeypatch.setattr(pipeline_mod, "_PREVIEW_POOL_WAIT_SEC", 0.3)
        try:
            with pytest.raises(RuntimeError, match="being generated"):
                pipeline_mod.preview_tts("Hello there.", AudiobookConfig(tts_provider_name="mock"))
        finally:
            for provider in held.values():
                pool.release(provider)

    def test_a_job_between_chapters_keeps_its_pool(self):
        pool = _pool(AudiobookConfig())
        self._register(pool)
        GPUPoolManager.instance()._pools["other"] = _pool(AudiobookConfig())
        pool.begin_job()
        try:
            assert not pool.is_idle   # no provider checked out, but a run is using it
            with pytest.raises(RuntimeError, match="being generated"):
                pipeline_mod.preview_tts("Hello there.", AudiobookConfig(tts_provider_name="other"))
            assert "mock" in GPUPoolManager.instance().all_pools()
        finally:
            pool.end_job()
        assert pool.is_idle

    def test_waiters_stop_when_the_pool_is_released(self):
        pool = _pool(AudiobookConfig())
        held = pool.acquire(preferred_device="cuda:0")
        errors: list = []

        def _wait():
            try:
                pool.acquire_devices(list(_DEVICES))
            except RuntimeError as exc:
                errors.append(exc)

        waiter = threading.Thread(target=_wait)
        waiter.start()
        time.sleep(0.2)
        pool.cleanup()
        waiter.join(timeout=5)
        assert not waiter.is_alive() and "released" in str(errors[0])
        assert held is not None

    def test_two_previews_never_share_the_model_at_once(self):
        config = AudiobookConfig(tts_provider_name="mock")
        active, peak = [0], [0]
        lock = threading.Lock()
        provider = pipeline_mod._cached_preview_provider(config)
        original = provider.synthesize

        def _counting(*args, **kwargs):
            with lock:
                active[0] += 1
                peak[0] = max(peak[0], active[0])
            time.sleep(0.2)
            try:
                return original(*args, **kwargs)
            finally:
                with lock:
                    active[0] -= 1

        provider.synthesize = _counting
        threads = [threading.Thread(target=pipeline_mod.preview_tts, args=("Hello there.", config)) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)
        assert peak[0] == 1


class _SlowWhisper:
    """Fake faster-whisper model that records how many run at once."""

    active = 0
    peak = 0
    lock = threading.Lock()

    def transcribe(self, samples, **kwargs):
        with _SlowWhisper.lock:
            _SlowWhisper.active += 1
            _SlowWhisper.peak = max(_SlowWhisper.peak, _SlowWhisper.active)
        time.sleep(0.3)
        with _SlowWhisper.lock:
            _SlowWhisper.active -= 1

        class _Segment:
            text = "this sentence has about sixty characters of ordinary narration"

        return [_Segment()], None


class TestWhisperPerGpu:

    @staticmethod
    def _wav(seconds: float) -> bytes:
        rate = 16000
        tone = 0.2 * np.sin(np.linspace(0, 2 * np.pi * 220 * seconds, int(rate * seconds))).astype(np.float32)
        buffer = io.BytesIO()
        sf.write(buffer, tone, rate, format="WAV")
        return buffer.getvalue()

    def test_two_gpus_transcribe_at_the_same_time(self, monkeypatch):
        verifier = ChunkVerifier("asr", max_error_rate=0.3)
        loaded: list[str] = []

        def _load(device):
            loaded.append(device)
            return _SlowWhisper(), "faster-whisper"

        monkeypatch.setattr(verifier, "_load_asr", _load)
        for device in _DEVICES:
            verifier.prepare(device)
        assert loaded == _DEVICES
        text = "This sentence has about sixty characters of ordinary narration."
        wav = self._wav(4.5)
        results: list = []
        threads = [
            threading.Thread(target=lambda d=device: results.append(verifier.check(text, wav, 4.5, device=d)))
            for device in _DEVICES
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=10)
        assert all(result.ok for result in results) and len(results) == 2
        assert _SlowWhisper.peak == 2
        assert loaded == _DEVICES  # loaded once per GPU, by prepare()

    def test_prepare_does_nothing_without_asr(self, monkeypatch):
        verifier = ChunkVerifier("duration")
        monkeypatch.setattr(verifier, "_load_asr", lambda device: pytest.fail("Whisper loaded"))
        verifier.prepare("cuda:0")


_WRITER = """
import sys
sys.path.insert(0, {root!r})
from audiobook_factory.progress_io import update_chapter_chunks
for index in range({start}, {start} + {count}):
    update_chapter_chunks({path!r}, 1, [index])
"""


class TestProgressFileAcrossProcesses:

    def test_two_processes_lose_no_update(self):
        count = 40
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "generation_progress.json")
            write_progress_file(path, {"chapters": [{"num": 1, "status": "in_progress", "completed_chunks": [],
                                                     "text": "x" * 200_000}]})
            writers = [
                subprocess.Popen([sys.executable, "-c", _WRITER.format(root=_ROOT, start=start, count=count, path=path)])
                for start in (0, count)
            ]
            for writer in writers:
                assert writer.wait(timeout=120) == 0
            done = read_progress_file(path)["chapters"][0]["completed_chunks"]
            assert sorted(done) == list(range(2 * count))
            json.dumps(done)


class TestVoiceDesignNextToAJobModel:

    def test_voice_design_borrows_the_idle_job_model(self, monkeypatch):
        import app

        pool = _pool(AudiobookConfig())
        GPUPoolManager.instance()._pools["mock"] = pool
        monkeypatch.setattr(app, "_active_local_run", lambda: None)
        monkeypatch.setattr(app, "is_api_healthy", lambda: False)
        monkeypatch.setattr(app, "_preview_provider", lambda cfg: pytest.fail("a second model was loaded"))
        seen = app._with_preview_provider(AudiobookConfig(tts_provider_name="mock"), lambda provider: provider.device)
        assert seen in _DEVICES and pool.is_idle
