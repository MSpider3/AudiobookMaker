"""
test_review_fixes.py
====================
Regression tests for the bugs fixed in the full-project review:
error masking, cancellation, chapter ordering, pronunciation fixes,
encoder settings, sample-rate handling, the chunk resume cache, stale
provider config, text normalisation, and the WebSocket completion event.
"""

from __future__ import annotations

import json
import os
import queue
import sys
import tempfile
import threading

import numpy as np
import pytest
import soundfile as sf

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from audiobook_factory import text_processing
from audiobook_factory.extractor_engine import TextNormalizer, _SKIP_TOC_TITLE
from audiobook_factory.gpu_pool import GPUPoolManager, ProviderPool
from audiobook_factory.pipeline import AudiobookConfig, CancelToken, run_pipeline
from audiobook_factory.text_extractor import ExtractedChapter, read_text_file
from audiobook_factory.utils import decode_done_message, encode_done_message
from tests.fixtures.mock_provider import MockTTSProvider

_VOICE = os.path.join(_ROOT, "tests", "fixtures", "audio", "synthetic_voice_reference.wav")


def _chapter(num: int, sentences: list[str] | None = None) -> ExtractedChapter:
    sentences = sentences or [
        f"This is sentence {i} of part {num}, long enough to be audible." for i in range(3)
    ]
    return ExtractedChapter(num=num, title=f"Part {num}", text=" ".join(sentences), sentences=sentences)


def _config(out_dir: str, **overrides) -> AudiobookConfig:
    settings = dict(
        book_title="ReviewBook",
        voice_file=_VOICE,
        output_dir=out_dir,
        tts_provider_name="mock",
        export_lrc=False,
        max_chapter_retries=0,
        retry_failed_at_end=False,
        pack_sentences=False,  # one chunk per sentence keeps chunk counts explicit
    )
    settings.update(overrides)
    return AudiobookConfig(**settings)


def _run(cfg: AudiobookConfig, chapters: list[ExtractedChapter], cancel: CancelToken | None = None):
    log_q: queue.Queue = queue.Queue()
    files = run_pipeline(cfg, chapters, log_q, queue.Queue(), cancel=cancel)
    logs = []
    while not log_q.empty():
        logs.append(log_q.get())
    return files, logs


def _mock_providers(cfg: AudiobookConfig, provider_cls=MockTTSProvider) -> list[MockTTSProvider]:
    # One explicit device: these tests count batches and chunks, which must
    # not depend on how many GPUs the machine running them happens to have.
    pool = ProviderPool(lambda dev: provider_cls(cfg, device=dev), ["cpu"], "mock")
    GPUPoolManager.instance()._pools["mock"] = pool
    return [pool.get_provider_for_device(dev) for dev in pool.devices]


@pytest.fixture(autouse=True)
def _fresh_pool():
    GPUPoolManager.instance().shutdown()
    yield
    GPUPoolManager.instance().shutdown()


class TestPipelineFailureReporting:

    def test_real_error_is_reported_not_masked(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = _config(td)
            for provider in _mock_providers(cfg):
                provider.inject_error = RuntimeError("REAL-CAUSE-XYZ")
            files, logs = _run(cfg, [_chapter(1)])
            assert files == []
            joined = "\n".join(logs)
            assert "REAL-CAUSE-XYZ" in joined
            assert "sub_future" not in joined

    def test_empty_chapter_is_skipped_not_failed(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = _config(td)
            empty = ExtractedChapter(num=1, title="Empty", text="", sentences=[])
            files, logs = _run(cfg, [empty, _chapter(2)])
            assert len(files) == 1
            assert not any("failed" in line.lower() for line in logs)
            with open(os.path.join(td, "generation_progress.json"), encoding="utf-8") as fh:
                statuses = {c["num"]: c["status"] for c in json.load(fh)["chapters"]}
            assert statuses[1] != "failed"
            assert statuses[2] == "completed"


class TestCancellationAndResume:

    def test_cancel_returns_cleanly_and_keeps_chapter_pending(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = _config(td)
            for provider in _mock_providers(cfg):
                provider.simulated_delay = 0.4
            cancel = CancelToken()
            threading.Timer(0.2, cancel.cancel).start()
            files, _ = _run(cfg, [_chapter(1), _chapter(2)], cancel=cancel)
            assert files == []
            with open(os.path.join(td, "generation_progress.json"), encoding="utf-8") as fh:
                chapters = json.load(fh)["chapters"]
            assert all(c["status"] != "failed" for c in chapters)
            assert all(not c.get("last_error") for c in chapters)

    def test_failed_chapter_resumes_from_cached_chunks(self):
        sentences = [f"Sentence {i} of the interrupted chapter is right here." for i in range(12)]

        class FailsOnThirdBatch(MockTTSProvider):
            def synthesize_batch(self, texts, voice_ref, *, return_bytes=True):
                if self.batch_call_count == 2 and not getattr(self, "healed", False):
                    self.batch_call_count += 1
                    raise RuntimeError("simulated mid-chapter crash")
                return super().synthesize_batch(texts, voice_ref, return_bytes=return_bytes)

        with tempfile.TemporaryDirectory() as td:
            cfg = _config(td)
            provider = _mock_providers(cfg, FailsOnThirdBatch)[0]
            files, _ = _run(cfg, [_chapter(1, sentences)])
            assert files == []
            chunk_dir = os.path.join(td, ".temp_chunks", "abm_ch001")
            kept = [f for f in os.listdir(chunk_dir) if f.startswith("chunk_ch_1_")]
            assert len(kept) == 8, "chunks synthesized before the failure must survive it"

            provider.healed = True
            calls_before = provider.batch_call_count
            files, logs = _run(cfg, [_chapter(1, sentences)])
            assert len(files) == 1
            assert any("8 cached, 4 pending" in line for line in logs)
            assert provider.batch_call_count - calls_before == 1
            assert not os.path.exists(chunk_dir), "chunk cache is removed once the chapter succeeds"

    def test_cached_chunks_are_discarded_when_the_text_changes(self):
        class FailsOnSecondBatch(MockTTSProvider):
            def synthesize_batch(self, texts, voice_ref, *, return_bytes=True):
                if self.batch_call_count == 1 and not getattr(self, "healed", False):
                    self.batch_call_count += 1
                    raise RuntimeError("simulated crash")
                return super().synthesize_batch(texts, voice_ref, return_bytes=return_bytes)

        first = [f"Original sentence number {i} goes here." for i in range(8)]
        second = [f"A completely rewritten line {i} appears instead." for i in range(8)]
        with tempfile.TemporaryDirectory() as td:
            cfg = _config(td)
            provider = _mock_providers(cfg, FailsOnSecondBatch)[0]
            _run(cfg, [_chapter(1, first)])
            provider.healed = True
            files, logs = _run(cfg, [_chapter(1, second)])
            assert len(files) == 1
            assert any("discarding" in line for line in logs)
            assert any("0 cached, 8 pending" in line for line in logs)


class TestOutputs:

    def test_outputs_are_ordered_by_chapter_number(self):
        with tempfile.TemporaryDirectory() as td:
            files, _ = _run(_config(td), [_chapter(i) for i in range(1, 13)])
            names = [os.path.basename(f) for f in files]
            assert names == [f"Chapter {i} - Part {i}.mp3" for i in range(1, 13)]

    def test_pronunciation_map_reaches_presplit_sentences(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = _config(td, pronunciation_map={"Klein": "Kline"}, export_lrc=True)
            _run(cfg, [_chapter(1, ["Klein walked home slowly through the fog."])])
            lrc = [f for f in os.listdir(td) if f.endswith(".lrc")][0]
            with open(os.path.join(td, lrc), encoding="utf-8") as fh:
                content = fh.read()
            assert "Kline walked home" in content
            assert "Klein" not in content

    @pytest.mark.parametrize("with_cover", [False, True])
    def test_mp3_honours_bitrate_sample_rate_and_tags(self, with_cover):
        from mutagen.mp3 import MP3

        sentences = [
            f"This is sentence number {i} and it is long enough to measure the bitrate." for i in range(10)
        ]
        with tempfile.TemporaryDirectory() as td:
            overrides = dict(bitrate_kbps=64, author="An Author", book_title="A Book")
            if with_cover:
                from PIL import Image
                cover = os.path.join(td, "cover.png")
                Image.new("RGB", (64, 64), "red").save(cover)
                overrides["cover_image"] = cover
            files, _ = _run(_config(td, **overrides), [_chapter(1, sentences)])
            audio = MP3(files[0])
            assert audio.info.sample_rate == 24000
            assert 56_000 <= audio.info.bitrate <= 72_000
            assert str(audio.tags["TIT2"]) == "Part 1"
            assert str(audio.tags["TPE1"]) == "An Author"
            assert str(audio.tags["TALB"]) == "A Book"

    @pytest.mark.parametrize("sample_rate,channels,fmt", [(48000, 1, "mp3"), (48000, 1, "flac"), (24000, 2, "flac")])
    def test_resample_and_stereo_do_not_change_speed(self, sample_rate, channels, fmt):
        class Native24k(MockTTSProvider):
            """Emits 24 kHz audio regardless of the requested output rate, like a real model."""
            def synthesize_batch(self, texts, voice_ref, *, return_bytes=True):
                out = []
                for text in texts:
                    signal, duration = self._generate_synthetic_waveform(text, sample_rate=24000)
                    out.append((self._waveform_to_wav_bytes(signal, 24000), duration))
                return out

        sentences = ["x" * 150] * 4  # mock: 9.75 s each, plus 0.5 s pauses → ~41 s
        with tempfile.TemporaryDirectory() as td:
            cfg = _config(td, sample_rate=sample_rate, channels=channels, output_format=fmt)
            _mock_providers(cfg, Native24k)
            files, _ = _run(cfg, [_chapter(1, sentences)])
            info = sf.info(files[0])
            assert info.samplerate == sample_rate
            assert info.channels == channels
            assert 39.0 <= info.duration <= 43.0


class TestProviderConfigRefresh:

    def test_pooled_provider_sees_the_current_runs_config(self):
        with tempfile.TemporaryDirectory() as td:
            first = _config(os.path.join(td, "a"), temperature=0.3)
            provider = _mock_providers(first)[0]
            _run(first, [_chapter(1)])
            assert provider.config is first

            second = _config(os.path.join(td, "b"), temperature=0.9)
            _run(second, [_chapter(1)])
            assert provider.config is second


class TestTextNormalisation:

    @staticmethod
    def _normalize(text: str, title: str = "T", **kwargs) -> str:
        return TextNormalizer().normalize(text, title, [], **kwargs)

    @pytest.mark.parametrize("text", [
        "The room was dark and the year after that he left.",
        "He took a step into the hall. Part time work. Point taken.",
        "Chapter One was long and Class D students arrived.",
        "Vitamin C is good. Plan B was ready. The X chromosome.",
        "Harry S Truman and O Captain! my Captain.",
    ])
    def test_ordinary_prose_is_left_alone(self, text):
        assert self._normalize(text) == text

    def test_paragraph_break_after_single_letter_heading_survives(self):
        assert self._normalize("Part I\n\nThe sun rose.") == "Part I\n\nThe sun rose."

    def test_drop_caps_are_still_repaired(self):
        assert self._normalize("T he sun rose.\n\nO nce upon a time.") == "The sun rose.\n\nOnce upon a time."

    def test_pdf_kerning_repair_spares_labelled_letters(self):
        fixed = self._normalize("W ar came to T ohsaka. Class D students left.", fix_kerning=True)
        assert fixed == "War came to Tohsaka. Class D students left."

    def test_markdown_and_entities_are_stripped(self):
        raw = "## Chapter 3\n\n<!-- image -->\n\nSmith &amp; Sons sold it.\n\n* * *\n\nNext scene."
        assert self._normalize(raw) == "Chapter 3\n\nSmith & Sons sold it.\n\nNext scene."

    def test_empty_title_keeps_paragraph_breaks(self):
        raw = "My Book\n\nFirst paragraph here.\n\nSecond paragraph."
        assert self._normalize(raw, title="") == raw

    @pytest.mark.parametrize("title", [
        "End of the Road", "Maple Street", "About a Boy", "Cover of Darkness", "Character Flaws",
    ])
    def test_real_chapter_titles_are_not_skipped(self, title):
        assert _SKIP_TOC_TITLE.match(title) is None

    @pytest.mark.parametrize("title", [
        "About the Author", "About", "Copyright © 2020", "Map", "Characters", "End of Volume 1", "Contents",
    ])
    def test_front_and_back_matter_is_still_skipped(self, title):
        assert _SKIP_TOC_TITLE.match(title) is not None

    @pytest.mark.parametrize("text", [
        "Это очень длинное предложение " * 20,
        "word — " * 120,
        "这是一个很长的句子" * 80,
        "x" * 1000,
    ], ids=["cyrillic", "em-dash", "cjk", "unbroken"])
    def test_splitter_handles_non_ascii_and_respects_max_len(self, text):
        chunks = text_processing.smart_sentence_splitter(text, 399)
        assert chunks
        assert all(len(chunk) <= 399 for chunk in chunks)
        assert text_processing._soft_split_long_sentence(text.strip(), 399) == chunks

    def test_splitter_drops_chunks_without_words(self):
        assert text_processing.smart_sentence_splitter("First.\n\n*\n\n...\n\nSecond.", 399) == ["First.", "Second."]

    def test_legacy_encoded_text_file_is_decoded(self, tmp_path):
        path = tmp_path / "book.txt"
        path.write_bytes("It’s a café.".encode("cp1252"))
        assert read_text_file(str(path)) == "It’s a café."


class TestDoneMessage:

    def test_paths_with_commas_round_trip(self):
        files = ["/out/The Lion, the Witch/Chapter 1 - Hello, World.mp3", "/out/b.mp3"]
        assert decode_done_message(encode_done_message(files)) == files

    def test_non_sentinel_and_empty(self):
        assert decode_done_message("just a log line") is None
        assert decode_done_message(encode_done_message()) == []


class TestRustMastering:

    def test_quiet_audio_is_raised_to_target_loudness(self, tmp_path):
        audiobook_rust = pytest.importorskip("audiobook_rust")
        if not hasattr(audiobook_rust, "master_audio"):
            pytest.skip("Rust extension not built")
        pyln = pytest.importorskip("pyloudnorm")

        rate = 24000
        t = np.arange(rate * 20) / rate
        quiet = (0.02 * np.sin(2 * np.pi * 220 * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 3 * t))).astype(np.float32)
        src, out = str(tmp_path / "quiet.wav"), str(tmp_path / "out.wav")
        sf.write(src, quiet, rate, subtype="PCM_16")
        audiobook_rust.master_audio([src], out, 0.0, rate, -18.0, -1.5, 64)
        mastered, _ = sf.read(out, dtype="float32")
        assert abs(pyln.Meter(rate).integrated_loudness(mastered) - (-18.0)) < 1.0


class TestWebSocketCompletion:

    def test_client_receives_file_list_before_socket_closes(self, monkeypatch, tmp_path):
        pytest.importorskip("fastapi")
        from fastapi.testclient import TestClient
        from starlette.websockets import WebSocketDisconnect

        import asyncio

        monkeypatch.setenv("ABM_SKIP_GPU_WARMUP", "1")
        monkeypatch.setenv("ABM_OUTPUT_BASE", str(tmp_path))
        import api.server as server_mod
        import api.worker as worker_mod
        from api.server import app

        # The module-level queue stays bound to whichever event loop used it
        # first, so give this TestClient's loop a queue of its own.
        fresh_queue: asyncio.Queue = asyncio.Queue()
        monkeypatch.setattr(worker_mod, "task_queue", fresh_queue)
        monkeypatch.setattr(server_mod, "task_queue", fresh_queue)

        payload = {
            "config": {
                "book_title": "WsDone",
                "output_dir": "ws_done",
                "output_format": "mp3",
                "tts_provider_name": "mock",
                "voice_file": _VOICE,
                "export_lrc": False,
            },
            "chapters": [{
                "num": 1,
                "title": "Chapter 1",
                "text": "The completion event must arrive.",
                "sentences": ["The completion event must arrive."],
            }],
        }
        events = []
        with TestClient(app) as client:
            task_id = client.post("/api/v1/generate", json=payload).json()["task_id"]
            # Never hang the suite: if the task stalls, cancelling it makes the
            # server close the socket and the assertions below fail instead.
            watchdog = threading.Timer(
                60.0, lambda: client.post(f"/api/v1/tasks/{task_id}/cancel")
            )
            watchdog.daemon = True
            watchdog.start()
            try:
                with client.websocket_connect(f"/api/v1/ws/{task_id}") as ws:
                    try:
                        while True:
                            event = ws.receive_json()
                            events.append(event)
                            if event.get("type") == "session_end":
                                break
                    except WebSocketDisconnect:
                        pass
            finally:
                watchdog.cancel()

        completed = [e for e in events if e.get("type") == "completed"]
        assert completed, f"no 'completed' event delivered; got {[e.get('type') for e in events]}"
        assert completed[0]["files"] and completed[0]["files"][0].endswith(".mp3")
