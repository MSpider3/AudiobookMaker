"""
test_pipeline_features.py
=========================
Tests for natural pacing (sentence packing, paragraph pauses), chunk
verification, multi-device work sharing, stable chapter numbering,
per-chapter redo, speed control and chapter markers in combined files.
"""

from __future__ import annotations

import io
import json
import os
import queue
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pytest
import soundfile as sf

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from audiobook_factory import pipeline as pipeline_module
from audiobook_factory.chunk_planner import plan_chunks
from audiobook_factory.chunk_verifier import ChunkVerifier, error_rate, expected_seconds
from audiobook_factory.gpu_pool import GPUPoolManager, ProviderPool
from audiobook_factory.pipeline import (
    AudiobookConfig,
    _EtaTracker,
    _number_chapters,
    _reconcile_chapter_entries,
    _subtitle_cues,
    run_pipeline,
)
from audiobook_factory.text_extractor import ExtractedChapter
from tests.fixtures.mock_provider import MockTTSProvider

_VOICE = os.path.join(_ROOT, "tests", "fixtures", "audio", "synthetic_voice_reference.wav")


def _config(out_dir: str, **overrides) -> AudiobookConfig:
    settings = dict(
        book_title="FeatureBook",
        voice_file=_VOICE,
        output_dir=out_dir,
        tts_provider_name="mock",
        export_lrc=False,
        max_chapter_retries=0,
        retry_failed_at_end=False,
    )
    settings.update(overrides)
    return AudiobookConfig(**settings)


def _chapter(num: int, text: str | None = None, title: str | None = None) -> ExtractedChapter:
    text = text or (
        f"This is the opening line of part {num}. It is followed by a second sentence.\n\n"
        f"A new paragraph begins here in part {num}. And it ends with this one."
    )
    return ExtractedChapter(num=num, title=title or f"Part {num}", text=text, sentences=[])


def _run(cfg: AudiobookConfig, chapters: list[ExtractedChapter]):
    log_q: queue.Queue = queue.Queue()
    files = run_pipeline(cfg, chapters, log_q, queue.Queue())
    logs = []
    while not log_q.empty():
        logs.append(log_q.get())
    return files, logs


def _install_pool(cfg: AudiobookConfig, devices: list[str], provider_cls=MockTTSProvider) -> ProviderPool:
    pool = ProviderPool(lambda dev: provider_cls(cfg, device=dev), devices, "mock")
    GPUPoolManager.instance()._pools["mock"] = pool
    return pool


def _progress(out_dir: str) -> dict:
    with open(os.path.join(out_dir, "generation_progress.json"), encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture(autouse=True)
def _fresh_pool():
    GPUPoolManager.instance().shutdown()
    yield
    GPUPoolManager.instance().shutdown()


class TestChunkPlanner:

    def test_sentences_of_a_paragraph_are_packed_up_to_max_len(self):
        text = "One short sentence. Another short sentence. A third one here.\n\nSecond paragraph."
        chunks = plan_chunks(text, None, max_len=399, pause=0.5, para_pause=1.2)
        assert [c.text for c in chunks] == [
            "One short sentence. Another short sentence. A third one here.",
            "Second paragraph.",
        ]
        assert chunks[0].sentences == (
            "One short sentence.", "Another short sentence.", "A third one here.",
        )

    def test_packing_never_exceeds_max_len_and_never_crosses_paragraphs(self):
        paragraph = " ".join(f"Sentence number {i} is right here." for i in range(40))
        chunks = plan_chunks(f"{paragraph}\n\n{paragraph}", None, max_len=120, pause=0.5, para_pause=1.2)
        assert all(len(c.text) <= 120 for c in chunks)
        assert sum(1 for c in chunks if c.paragraph_end) == 2
        assert " ".join(c.text for c in chunks) == f"{paragraph} {paragraph}"

    def test_paragraph_end_gets_the_paragraph_pause(self):
        chunks = plan_chunks("First para.\n\nSecond para.\n\nThird para.", None, 399, pause=0.5, para_pause=1.2)
        assert [c.pause_after for c in chunks] == [1.2, 1.2, 0.5]

    def test_unpacked_mode_keeps_one_sentence_per_chunk(self):
        chunks = plan_chunks("One. Two. Three.", None, 399, 0.5, 1.2, pack_sentences=False)
        assert [c.text for c in chunks] == ["One.", "Two.", "Three."]
        assert [c.pause_after for c in chunks] == [0.5, 0.5, 0.5]

    def test_dialogue_tag_stays_with_its_quote(self):
        chunks = plan_chunks('"No!" he cried. She turned away.', None, 399, 0.5, 1.2, pack_sentences=False)
        assert chunks[0].text == '"No!" he cried.'

    def test_flat_sentence_list_is_one_paragraph(self):
        chunks = plan_chunks("", ["One.", "Two.", "Three."], 399, 0.5, 1.2)
        assert [c.text for c in chunks] == ["One. Two. Three."]

    def test_subtitle_cues_stay_sentence_sized_inside_a_packed_chunk(self):
        chunks = plan_chunks("Short one. A sentence twice as long.", None, 399, 0.5, 1.2)
        cues = _subtitle_cues(chunks, [9.0], default_pause=0.5)
        assert [text for _, _, text in cues] == ["Short one.", "A sentence twice as long."]
        assert cues[0][0] == 0.0
        assert cues[0][1] == pytest.approx(cues[1][0])
        assert cues[1][1] == pytest.approx(9.0)
        assert cues[1][1] - cues[1][0] > cues[0][1] - cues[0][0]


class TestPacing:

    def test_paragraph_pause_lengthens_the_chapter(self):
        text = "The first paragraph is here.\n\nThe second paragraph is here.\n\nThe third paragraph is here."
        durations = {}
        for para_pause in (0.5, 2.5):
            GPUPoolManager.instance().shutdown()
            with tempfile.TemporaryDirectory() as td:
                cfg = _config(td, output_format="wav", pause=0.5, para_pause=para_pause)
                files, _ = _run(cfg, [_chapter(1, text)])
                durations[para_pause] = sf.info(files[0]).duration
        # Two paragraph boundaries, each 2.0 s longer.
        assert durations[2.5] - durations[0.5] == pytest.approx(4.0, abs=0.1)

    def test_packed_chapter_uses_fewer_tts_calls_and_keeps_sentence_subtitles(self):
        sentences = [f"Sentence number {i} is right here." for i in range(10)]
        with tempfile.TemporaryDirectory() as td:
            cfg = _config(td, export_lrc=True)
            _, logs = _run(cfg, [_chapter(1, " ".join(sentences))])
            assert any("1 TTS chunks" in line for line in logs)
            lrc = [f for f in os.listdir(td) if f.endswith(".lrc")][0]
            with open(os.path.join(td, lrc), encoding="utf-8") as fh:
                lines = [line for line in fh.read().splitlines() if line.strip()]
            assert len(lines) == len(sentences)
            assert [line.split("]", 1)[1] for line in lines] == sentences

    def test_speed_is_applied_when_the_engine_cannot_change_speed(self):
        text = "x" * 150
        durations = {}
        for speed in (1.0, 1.25):
            GPUPoolManager.instance().shutdown()
            with tempfile.TemporaryDirectory() as td:
                cfg = _config(td, output_format="flac", speed=speed)
                files, _ = _run(cfg, [_chapter(1, text)])
                durations[speed] = sf.info(files[0]).duration
        assert durations[1.25] == pytest.approx(durations[1.0] / 1.25, rel=0.03)


class TestChunkVerifier:

    @staticmethod
    def _wav(seconds: float, amplitude: float = 0.2, rate: int = 24000) -> bytes:
        t = np.arange(int(seconds * rate)) / rate
        buf = io.BytesIO()
        sf.write(buf, (amplitude * np.sin(2 * np.pi * 220 * t)).astype(np.float32), rate, format="WAV")
        return buf.getvalue()

    def test_expected_duration_scales_with_text_and_script(self):
        english = "This sentence has about sixty characters of ordinary narration."
        assert 3.0 < expected_seconds(english) < 7.0
        assert expected_seconds("这是一个很长的句子" * 4) > expected_seconds("abcdefghi" * 4)
        assert expected_seconds(english, speed=2.0) == pytest.approx(expected_seconds(english) / 2.0)

    def test_duration_mode_accepts_plausible_audio(self):
        text = "This sentence has about sixty characters of ordinary narration."
        verdict = ChunkVerifier("duration").check(text, self._wav(4.5), 4.5)
        assert verdict.ok

    @pytest.mark.parametrize("seconds,amplitude,expected_reason", [
        (0.6, 0.2, "too short"),
        (60.0, 0.2, "too long"),
        (4.5, 0.0, "silent"),
    ])
    def test_duration_mode_rejects_truncated_runaway_and_silent_audio(self, seconds, amplitude, expected_reason):
        text = "This sentence has about sixty characters of ordinary narration."
        verdict = ChunkVerifier("duration").check(text, self._wav(seconds, amplitude), seconds)
        assert not verdict.ok
        assert expected_reason in verdict.reason

    def test_off_mode_accepts_everything(self):
        assert ChunkVerifier("off").check("Some text here.", b"not audio", 0.0).ok

    def test_error_rate(self):
        assert error_rate("The quick brown fox.", "the quick brown fox") == 0.0
        assert error_rate("The quick brown fox jumps", "the quick brown") == pytest.approx(0.4)
        assert error_rate("这是一个句子", "这是一个句子") == 0.0
        assert error_rate("这是一个句子", "这是一个") == pytest.approx(2 / 6)

    def test_asr_mode_rejects_a_transcript_that_does_not_match(self, monkeypatch):
        verifier = ChunkVerifier("asr", max_error_rate=0.3)
        text = "This sentence has about sixty characters of ordinary narration."
        monkeypatch.setattr(verifier, "_asr", object())
        monkeypatch.setattr(verifier, "_transcribe", lambda samples, rate: "completely different words were spoken")
        assert not verifier.check(text, self._wav(4.5), 4.5).ok
        monkeypatch.setattr(verifier, "_transcribe", lambda samples, rate: text.lower())
        assert verifier.check(text, self._wav(4.5), 4.5).ok


class TestVerificationInPipeline:

    def test_truncated_chunk_is_resynthesized(self):
        class TruncatesFirstTake(MockTTSProvider):
            def synthesize_batch(self, texts, voice_ref, *, return_bytes=True):
                results = super().synthesize_batch(texts, voice_ref, return_bytes=return_bytes)
                audio, _ = results[0]
                samples, rate = sf.read(io.BytesIO(audio), dtype="float32")
                buf = io.BytesIO()
                sf.write(buf, samples[: rate // 4], rate, format="WAV")
                results[0] = (buf.getvalue(), 0.25)
                return results

        with tempfile.TemporaryDirectory() as td:
            cfg = _config(td, output_format="wav", verify_chunks="duration")
            provider = _install_pool(cfg, ["cpu"], TruncatesFirstTake).get_provider_for_device("cpu")
            files, logs = _run(cfg, [_chapter(1, "x" * 150)])
            assert provider.synthesis_call_count == 1, "the bad chunk is re-synthesized on its own"
            assert sf.info(files[0]).duration > 9.0
            assert _progress(td)["chapters"][0]["flagged_chunks"] == []

    def test_chunk_that_never_passes_is_kept_and_reported(self):
        class AlwaysTooLong(MockTTSProvider):
            def _generate_synthetic_waveform(self, text, sample_rate=24000):
                signal, duration = super()._generate_synthetic_waveform(text, sample_rate)
                return np.tile(signal, 8), duration * 8

        with tempfile.TemporaryDirectory() as td:
            cfg = _config(td, verify_chunks="duration", verify_max_retries=1)
            _install_pool(cfg, ["cpu"], AlwaysTooLong)
            files, logs = _run(cfg, [_chapter(1, "x" * 150)])
            assert len(files) == 1
            flagged = _progress(td)["chapters"][0]["flagged_chunks"]
            assert len(flagged) == 1 and "too long" in flagged[0]["reason"]
            assert any("worth a listen" in line for line in logs)

    def test_silent_chunk_fails_the_chapter(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = _config(td, verify_chunks="duration", verify_max_retries=1)
            provider = _install_pool(cfg, ["cpu"]).get_provider_for_device("cpu")
            provider.inject_silence = True
            files, logs = _run(cfg, [_chapter(1)])
            assert files == []
            assert any("no usable audio" in line for line in logs)
            assert _progress(td)["chapters"][0]["status"] == "failed"


class TestMultiDevice:

    def test_chunks_are_shared_across_devices(self):
        text = "\n\n".join(f"Paragraph number {i} has a sentence of its own." for i in range(16))
        with tempfile.TemporaryDirectory() as td:
            cfg = _config(td)
            pool = _install_pool(cfg, ["dev0", "dev1"])
            for device in pool.devices:
                pool.get_provider_for_device(device).simulated_delay = 0.05
            files, logs = _run(cfg, [_chapter(1, text)])
            assert len(files) == 1
            calls = [pool.get_provider_for_device(d).batch_call_count for d in pool.devices]
            assert all(count > 0 for count in calls), f"both devices must take work, got {calls}"
            assert any("2 device(s): dev0, dev1" in line for line in logs)

    def test_a_failing_device_hands_its_work_to_the_others(self):
        text = "\n\n".join(f"Paragraph number {i} has a sentence of its own." for i in range(16))
        with tempfile.TemporaryDirectory() as td:
            cfg = _config(td, output_format="wav")
            pool = _install_pool(cfg, ["dev0", "dev1"])
            pool.get_provider_for_device("dev0").simulated_delay = 0.05
            pool.get_provider_for_device("dev1").inject_error = RuntimeError("dev1 is broken")
            files, logs = _run(cfg, [_chapter(1, text)])
            assert len(files) == 1, "the healthy device must finish the chapter"
            assert any("remaining device(s) finished the chapter" in line for line in logs)
            # 16 chunks of ~3 s plus 16 pauses of 1.2 s (last one 0.5 s).
            assert sf.info(files[0]).duration > 16 * 2.5

    def test_progress_file_is_written_once_per_batch_not_per_chunk(self, monkeypatch):
        calls: list[list[int]] = []
        real = pipeline_module.update_chapter_chunks
        monkeypatch.setattr(
            pipeline_module, "update_chapter_chunks",
            lambda path, num, indices: (calls.append(list(indices)), real(path, num, indices))[1],
        )
        text = "\n\n".join(f"Paragraph number {i} has a sentence of its own." for i in range(12))
        with tempfile.TemporaryDirectory() as td:
            _run(_config(td, batch_size=4), [_chapter(1, text)])
        assert len(calls) == 3
        assert sorted(i for batch in calls for i in batch) == list(range(12))


class TestChapterNumbering:

    def test_extractor_numbers_are_kept_for_a_subset(self):
        with tempfile.TemporaryDirectory() as td:
            files, _ = _run(_config(td), [_chapter(50), _chapter(51)])
            assert [os.path.basename(f) for f in files] == [
                "Chapter 50 - Part 50.mp3", "Chapter 51 - Part 51.mp3",
            ]
            entries = _progress(td)["chapters"]
            assert [(e["num"], e["status"]) for e in entries] == [(50, "completed"), (51, "completed")]
            assert _progress(td)["generation_summary"]["all_complete"] is True

    def test_unusable_numbers_fall_back_to_positions(self):
        chapters = [_chapter(1), _chapter(1), _chapter(0)]
        assert [num for num, _ in _number_chapters(chapters)] == [1, 2, 3]

    def test_status_is_reused_only_for_the_same_title(self):
        previous = [
            {"num": 1, "title": "Some Other Chapter", "status": "completed"},
            {"num": 7, "title": "Part 2", "status": "completed", "duration": 12.5},
        ]
        entries = _reconcile_chapter_entries(previous, [(1, _chapter(1)), (2, _chapter(2))])
        by_num = {e["num"]: e for e in entries}
        assert by_num[1]["status"] == "pending", "a different chapter that shared the number must not count"
        assert by_num[2]["status"] == "completed", "the same chapter under an old number carries over"
        assert by_num[2]["duration"] == 12.5

    def test_chapters_outside_the_run_keep_their_progress(self):
        previous = [{"num": 9, "title": "Part 9", "status": "completed"}]
        entries = _reconcile_chapter_entries(previous, [(1, _chapter(1))])
        assert [(e["num"], e["status"]) for e in entries] == [(1, "pending"), (9, "completed")]

    def test_redo_regenerates_only_the_named_chapter(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = _config(td)
            provider = _install_pool(cfg, ["cpu"]).get_provider_for_device("cpu")
            _run(cfg, [_chapter(1), _chapter(2), _chapter(3)])
            first_run_batches = provider.batch_call_count

            redo_cfg = _config(td, redo_chapters=[2])
            files, logs = _run(redo_cfg, [_chapter(1), _chapter(2), _chapter(3)])
            assert len(files) == 3
            assert provider.batch_call_count - first_run_batches == first_run_batches // 3
            assert sum("Already completed" in line for line in logs) == 2


class TestCombinedBook:

    @pytest.mark.skipif(shutil.which("ffprobe") is None, reason="ffprobe not available")
    @pytest.mark.parametrize("fmt", ["m4b", "mp3"])
    def test_single_file_has_one_marker_per_chapter(self, fmt):
        with tempfile.TemporaryDirectory() as td:
            cfg = _config(td, output_format=fmt, single_file_mode=True, book_title="My Book", author="A. Writer")
            files, _ = _run(cfg, [_chapter(i, title=f"The Title of {i}") for i in (1, 2, 3)])
            assert len(files) == 1 and os.path.basename(files[0]) == f"My Book.{fmt}"
            probe = json.loads(subprocess.run(
                ["ffprobe", "-v", "error", "-show_chapters", "-show_entries", "format_tags", "-of", "json", files[0]],
                capture_output=True, text=True, check=True,
            ).stdout)
            chapters = probe["chapters"]
            assert [c["tags"]["title"] for c in chapters] == ["The Title of 1", "The Title of 2", "The Title of 3"]
            starts = [float(c["start_time"]) for c in chapters]
            assert starts == sorted(starts) and starts[0] == 0.0 and starts[1] > 1.0
            assert not [f for f in os.listdir(td) if f.startswith("Chapter ") and f.endswith(f".{fmt}")]


class TestEta:

    def test_reports_once_enough_work_is_done(self, monkeypatch):
        lines: list[str] = []
        clock = {"now": 1000.0}
        monkeypatch.setattr(pipeline_module.time, "monotonic", lambda: clock["now"])
        tracker = _EtaTracker({1: 3000, 2: 1000}, lines.append)
        clock["now"] += 120.0
        tracker.update(1, 1.0)   # 75 % of the text in two minutes
        assert len(lines) == 1
        assert "75.0%" in lines[0] and "0:02:00" in lines[0] and "0:00:40" in lines[0]
        tracker.update(2, 0.5)   # too soon after the last report
        assert len(lines) == 1


class TestRerunBehaviour:

    def test_finished_single_file_book_is_not_resynthesized(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = _config(td, single_file_mode=True, book_title="My Book")
            provider = _install_pool(cfg, ["cpu"]).get_provider_for_device("cpu")
            first, _ = _run(cfg, [_chapter(1), _chapter(2)])
            batches = provider.batch_call_count
            second, logs = _run(cfg, [_chapter(1), _chapter(2)])
            assert second == first and os.path.exists(second[0])
            assert provider.batch_call_count == batches, "nothing may be synthesized again"
            assert any("Already complete" in line for line in logs)

    def test_one_shot_settings_are_not_saved_to_the_progress_file(self):
        with tempfile.TemporaryDirectory() as td:
            _run(_config(td), [_chapter(1), _chapter(2)])
            _run(_config(td, redo_chapters=[2], force_reprocess=False), [_chapter(1), _chapter(2)])
            settings = _progress(td)["settings"]
            assert settings["redo_chapters"] == []
            assert settings["force_reprocess"] is False

    def test_warmup_failure_reports_the_real_cause(self):
        class Unloadable(MockTTSProvider):
            def ensure_ready(self) -> None:
                raise RuntimeError("the engine package is not installed: pip install example-tts")

        with tempfile.TemporaryDirectory() as td:
            cfg = _config(td)
            with pytest.raises(RuntimeError, match="pip install example-tts"):
                GPUPoolManager.instance().get_pool("mock", lambda dev: Unloadable(cfg, device=dev))
