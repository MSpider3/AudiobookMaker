"""
tests/test_hardening.py

Regression tests for config contract, provider initialization safety,
and progress I/O robustness. Added after Kaggle runtime failures in
Phase 5 to prevent recurrence.
"""
import concurrent.futures
import logging
import os
import unittest
from unittest.mock import MagicMock, patch

from audiobook_factory.pipeline import (
    AudiobookConfig,
    _validate_config,
    _CONFIG_SCHEMA_VERSION,
    _get_chapter_parallelism,
)
from audiobook_factory.progress_io import (
    read_progress_file,
    write_progress_file,
    update_chapter_status,
    update_chapter_chunk,
)
from audiobook_factory.tts_providers.qwen_provider import QwenTTSProvider

try:
    from api.server import app
    HAS_FASTAPI = True
except ImportError:
    app = None
    HAS_FASTAPI = False


class TestConfigContract(unittest.TestCase):

    def test_audiobook_config_from_dict_unknown_keys_ignored(self):
        data = {"worker_count": 4, "unknown_field_xyz": "value"}
        cfg = AudiobookConfig.from_dict(data)
        self.assertEqual(cfg.worker_count, 4)
        self.assertFalse(hasattr(cfg, "unknown_field_xyz"))

    def test_audiobook_config_from_dict_missing_keys_use_defaults(self):
        data = {}
        cfg = AudiobookConfig.from_dict(data)
        self.assertEqual(cfg.quantization, "none")
        self.assertTrue(cfg.resume_incomplete_chunks)

    def test_audiobook_config_version_mismatch_logs_warning(self):
        data = {"config_version": 0}
        with self.assertLogs("audiobook_factory.pipeline", level="WARNING") as cm:
            cfg = AudiobookConfig.from_dict(data)
        self.assertTrue(any("schema version" in log for log in cm.output))

    def test_validate_config_raises_on_invalid_quantization(self):
        cfg = AudiobookConfig(quantization="invalid_mode")
        with self.assertRaises(ValueError):
            _validate_config(cfg)

    def test_field_summary(self):
        summary = AudiobookConfig.field_summary()
        self.assertIn("worker_count", summary)
        self.assertIn("quantization", summary)


class TestProgressIO(unittest.TestCase):

    def test_read_progress_file_utf8_bom(self):
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            f.write(b'\xef\xbb\xbf{"book_title": "Test"}')
            path = f.name
        try:
            data = read_progress_file(path)
            self.assertEqual(data["book_title"], "Test")
        finally:
            if os.path.exists(path):
                os.remove(path)

    def test_read_progress_file_empty_file(self):
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            f.write(b"")
            path = f.name
        try:
            with self.assertRaises(ValueError) as cm:
                read_progress_file(path)
            self.assertIn("empty", str(cm.exception))
        finally:
            if os.path.exists(path):
                os.remove(path)

    def test_read_progress_file_html_content(self):
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            f.write(b"<!DOCTYPE html><html>error page</html>")
            path = f.name
        try:
            with self.assertRaises(ValueError) as cm:
                read_progress_file(path)
            self.assertIn("HTML", str(cm.exception))
        finally:
            if os.path.exists(path):
                os.remove(path)

    def test_read_progress_file_leading_garbage(self):
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            f.write(b'chp{"book_title": "Test"}')
            path = f.name
        try:
            data = read_progress_file(path)
            self.assertEqual(data["book_title"], "Test")
        finally:
            if os.path.exists(path):
                os.remove(path)

    def test_write_progress_file_atomic(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "progress.json")
            write_progress_file(path, {"book_title": "Test"})
            self.assertTrue(os.path.exists(path))
            data = read_progress_file(path)
            self.assertEqual(data["book_title"], "Test")
            self.assertFalse(os.path.exists(path + ".tmp"))

    def test_update_chapter_chunk_no_duplicates(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "progress.json")
            write_progress_file(path, {
                "chapters": [{"num": 1, "completed_chunks": [0, 1]}]
            })
            update_chapter_chunk(path, chapter_num=1, chunk_index=1)
            data = read_progress_file(path)
            self.assertEqual(data["chapters"][0]["completed_chunks"].count(1), 1)

    def test_update_chapter_status_resets_chunks(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "progress.json")
            write_progress_file(path, {
                "chapters": [{"num": 1, "completed_chunks": [0, 1, 2, 3]}]
            })
            update_chapter_status(path, chapter_num=1, status="completed", reset_chunks=True)
            data = read_progress_file(path)
            self.assertEqual(data["chapters"][0]["completed_chunks"], [])
            self.assertEqual(data["chapters"][0]["status"], "completed")

    def test_update_chapter_chunk_concurrent_no_lost_writes(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp_dir:
            path = os.path.join(tmp_dir, "progress.json")
            write_progress_file(path, {
                "chapters": [{"num": 1, "completed_chunks": []}]
            })

            chunk_indices = list(range(50))

            def write_chunk(idx):
                update_chapter_chunk(path, chapter_num=1, chunk_index=idx)

            with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
                list(executor.map(write_chunk, chunk_indices))

            data = read_progress_file(path)
            completed = data["chapters"][0]["completed_chunks"]
            self.assertEqual(sorted(completed), chunk_indices)


class TestProviderInitialization(unittest.TestCase):

    def test_ensure_ready_raises_if_model_none(self):
        with patch.object(QwenTTSProvider, "_load_model", return_value=None):
            provider = QwenTTSProvider.__new__(QwenTTSProvider)
            provider._model = None
            provider._device = "cpu"
            provider.config = AudiobookConfig()
            with self.assertRaises(RuntimeError):
                provider.ensure_ready()

    def test_synthesize_batch_calls_ensure_ready_if_model_none(self):
        provider = QwenTTSProvider.__new__(QwenTTSProvider)
        provider._model = None
        provider._device = "cpu"
        provider.config = AudiobookConfig()
        with patch.object(provider, "ensure_ready", side_effect=RuntimeError("not loaded")) as mock_ready:
            with self.assertRaises(RuntimeError):
                provider.synthesize_batch(["test"], b"\x00" * 1000)
            mock_ready.assert_called_once()

    def test_resolve_voice_ref_converts_bytes_to_filepath(self):
        provider = QwenTTSProvider.__new__(QwenTTSProvider)
        dummy_wav_bytes = b"RIFF1234WAVEfmt "
        path = provider._resolve_voice_ref(dummy_wav_bytes)
        self.assertIsInstance(path, str)
        self.assertTrue(os.path.exists(path))
        with open(path, "rb") as f:
            self.assertEqual(f.read(), dummy_wav_bytes)


class TestHealthEndpoint(unittest.TestCase):

    def test_health_endpoint_includes_model_loaded(self):
        if not HAS_FASTAPI:
            self.skipTest("fastapi is not installed in current environment")
        from starlette.testclient import TestClient
        client = TestClient(app)

        mock_provider_ready = MagicMock()
        mock_provider_ready.is_ready = True

        mock_pool_ready = MagicMock()
        mock_pool_ready.device_count = 1
        mock_pool_ready.devices = ["cuda:0"]
        mock_pool_ready._device_map = {"cuda:0": mock_provider_ready}

        with patch("audiobook_factory.gpu_pool.GPUPoolManager.instance") as mock_mgr_inst:
            mock_mgr_inst.return_value.all_pools.return_value = {"qwen": mock_pool_ready}
            resp = client.get("/api/v1/health")
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertTrue(data["gpu"]["provider_pools"]["qwen"]["devices"][0]["model_loaded"])

        mock_provider_unready = MagicMock()
        mock_provider_unready.is_ready = False

        mock_pool_unready = MagicMock()
        mock_pool_unready.device_count = 1
        mock_pool_unready.devices = ["cuda:0"]
        mock_pool_unready._device_map = {"cuda:0": mock_provider_unready}

        with patch("audiobook_factory.gpu_pool.GPUPoolManager.instance") as mock_mgr_inst:
            mock_mgr_inst.return_value.all_pools.return_value = {"qwen": mock_pool_unready}
            resp = client.get("/api/v1/health")
            self.assertEqual(resp.status_code, 200)
            data = resp.json()
            self.assertFalse(data["gpu"]["provider_pools"]["qwen"]["devices"][0]["model_loaded"])


class TestPreflightAndRecovery(unittest.TestCase):

    def test_preflight_voice_ref_none_is_error(self):
        from audiobook_factory.preflight import _check_voice_ref_type
        passed, msg = _check_voice_ref_type(None)
        self.assertFalse(passed)
        self.assertIn("None", msg)

    def test_preflight_voice_ref_bytes_valid(self):
        from audiobook_factory.preflight import _check_voice_ref_type
        passed, msg = _check_voice_ref_type(b"\x00" * 1000)
        self.assertTrue(passed)

    def test_preflight_voice_ref_bytes_too_short(self):
        from audiobook_factory.preflight import _check_voice_ref_type
        passed, msg = _check_voice_ref_type(b"\x00" * 10)
        self.assertFalse(passed)
        self.assertTrue("short" in msg.lower() or "bytes" in msg.lower())

    def test_preflight_voice_ref_wrong_type(self):
        from audiobook_factory.preflight import _check_voice_ref_type
        passed, msg = _check_voice_ref_type(12345)
        self.assertFalse(passed)
        self.assertIn("int", msg)

    def test_preflight_voice_ref_str_missing_file(self):
        import tempfile
        from audiobook_factory.preflight import _check_voice_ref_type
        with tempfile.TemporaryDirectory() as tmp:
            missing = os.path.join(tmp, "nonexistent.wav")
            passed, msg = _check_voice_ref_type(missing)
            self.assertFalse(passed)
            self.assertIn("not exist", msg)

    def test_preflight_voice_ref_str_valid_file(self):
        import tempfile
        from audiobook_factory.preflight import _check_voice_ref_type
        with tempfile.TemporaryDirectory() as tmp:
            wav = os.path.join(tmp, "voice.wav")
            with open(wav, "wb") as f:
                f.write(b"\x00" * 5000)
            passed, msg = _check_voice_ref_type(wav)
            self.assertTrue(passed)

    def test_preflight_result_passed_when_no_errors(self):
        from audiobook_factory.preflight import PreflightResult
        result = PreflightResult(
            passed=True, warnings=[], errors=[],
            device_info=[], recommended_dtype="float16",
            python_version=(3, 12, 0), library_versions={},
        )
        self.assertTrue(result.passed)

    def test_preflight_error_contains_errors(self):
        from audiobook_factory.preflight import PreflightResult, PreflightError
        result = PreflightResult(
            passed=False, warnings=[], errors=["FFmpeg missing"],
            device_info=[], recommended_dtype="float16",
            python_version=(3, 12, 0), library_versions={},
        )
        exc = PreflightError(result)
        self.assertIn("FFmpeg missing", str(exc))
        self.assertIs(exc.result, result)

    def test_validate_chunk_file_missing(self):
        import tempfile
        from audiobook_factory.chapter_pipeline import _validate_chunk_file
        with tempfile.TemporaryDirectory() as tmp:
            self.assertFalse(_validate_chunk_file(os.path.join(tmp, "nonexistent.wav")))

    def test_validate_chunk_file_too_small(self):
        import tempfile
        from audiobook_factory.chapter_pipeline import _validate_chunk_file
        with tempfile.TemporaryDirectory() as tmp:
            f = os.path.join(tmp, "tiny.wav")
            with open(f, "wb") as fh:
                fh.write(b"\x00" * 50)
            self.assertFalse(_validate_chunk_file(f))

    def test_validate_chunk_file_valid(self):
        import tempfile
        from audiobook_factory.chapter_pipeline import _validate_chunk_file
        with tempfile.TemporaryDirectory() as tmp:
            f = os.path.join(tmp, "valid.wav")
            with open(f, "wb") as fh:
                fh.write(b"\x00" * 2000)
            self.assertTrue(_validate_chunk_file(f))

    def test_sanitize_covers_dict_values(self):
        from audiobook_factory.tts_providers.qwen_provider import _sanitize_dict_keys

        class FakeModel:
            pass

        model = FakeModel()
        model.config = FakeModel()
        model.config.some_values = {"a": 1}.values()

        _sanitize_dict_keys(model)

        self.assertIsInstance(model.config.some_values, list)

    def test_pickle_patch_makes_dict_keys_picklable(self):
        """Verifies the Python 3.12 pickle patch enables dict view pickling."""
        import pickle
        from audiobook_factory.preflight import _apply_python312_pickle_patch

        _apply_python312_pickle_patch()

        # All three view types must be picklable after patch
        for name, obj in [
            ("dict_keys", {"a": 1}.keys()),
            ("dict_values", {"a": 1}.values()),
            ("dict_items", {"a": 1}.items()),
        ]:
            try:
                result = pickle.dumps(obj)
                restored = pickle.loads(result)
                self.assertIsInstance(
                    restored, list,
                    f"{name}: expected list after unpickling, got {type(restored)}"
                )
            except (TypeError, Exception) as exc:
                self.fail(f"pickle.dumps({name}) failed after patch: {exc}")

    def test_pickle_patch_idempotent(self):
        """Verifies calling _apply_python312_pickle_patch() twice is safe."""
        from audiobook_factory.preflight import _apply_python312_pickle_patch
        from audiobook_factory import preflight
        _apply_python312_pickle_patch()
        _apply_python312_pickle_patch()
        self.assertTrue(preflight._PICKLE_PATCH_APPLIED)

    def test_deepcopy_works_after_pickle_patch(self):
        """Verifies copy.deepcopy works on objects containing dict views
        after the patch — simulating what HuggingFace does internally."""
        import copy
        from audiobook_factory.preflight import _apply_python312_pickle_patch

        _apply_python312_pickle_patch()

        class FakeConfig:
            def __init__(self):
                self.allowed_keys = {"a": 1, "b": 2}.keys()
                self.values = {"x": 1}.values()

        cfg = FakeConfig()
        try:
            cfg_copy = copy.deepcopy(cfg)
            self.assertIsInstance(cfg_copy.allowed_keys, list)
            self.assertIsInstance(cfg_copy.values, list)
        except TypeError as exc:
            self.fail(f"copy.deepcopy failed after patch: {exc}")


class TestSubtitleAudioSync(unittest.TestCase):

    def test_concat_partial_includes_pause_for_all_chunks(self):
        """Verifies _concat_partial appends pause samples after every chunk segment."""
        import tempfile
        import types

        sr = 24000
        pause_sec = 0.5
        cfg = AudiobookConfig(sample_rate=sr, pause=pause_sec)

        try:
            import numpy as np
            import soundfile as sf
            HAS_SF = True
        except ImportError:
            HAS_SF = False

        if HAS_SF:
            with tempfile.TemporaryDirectory() as tmpdir:
                from audiobook_factory.chapter_pipeline import _concat_partial
                chunk_paths = []
                chunk_durations = [1.0, 1.5, 2.0]
                for i, dur in enumerate(chunk_durations):
                    p = os.path.join(tmpdir, f"chunk_{i}.wav")
                    samples = np.ones(int(dur * sr), dtype=np.float32) * 0.1
                    sf.write(p, samples, sr)
                    chunk_paths.append(p)

                out_wav = os.path.join(tmpdir, "partial.wav")
                _concat_partial(chunk_paths, out_wav, cfg)

                data, out_sr = sf.read(out_wav)
                total_expected_sec = sum(chunk_durations) + len(chunk_durations) * pause_sec
                actual_sec = len(data) / out_sr
                self.assertAlmostEqual(actual_sec, total_expected_sec, delta=0.01)
        else:
            # Fallback mock test when soundfile library is not installed
            import numpy as np
            fake_data = np.ones(24000, dtype=np.float32)
            mock_sf = MagicMock()
            mock_sf.read.return_value = (fake_data, sr)
            
            with patch.dict("sys.modules", {"soundfile": mock_sf}):
                from audiobook_factory.chapter_pipeline import _concat_partial
                with tempfile.TemporaryDirectory() as tmpdir:
                    p1 = os.path.join(tmpdir, "chunk_0.wav")
                    p2 = os.path.join(tmpdir, "chunk_1.wav")
                    with open(p1, "wb") as f: f.write(b"0"*200)
                    with open(p2, "wb") as f: f.write(b"0"*200)
                    out_wav = os.path.join(tmpdir, "out.wav")
                    _concat_partial([p1, p2], out_wav, cfg)
                    
                    self.assertTrue(mock_sf.write.called)
                    written_args = mock_sf.write.call_args[0]
                    raw_concat = written_args[1]
                    # Expected length: 2 chunk frames (2 * 24000) + 2 pause frames (2 * 12000) = 72000
                    expected_len = 2 * 24000 + 2 * int(pause_sec * sr)
                    self.assertEqual(len(raw_concat), expected_len)

    def test_subtitle_timing_matches_partial_concatenation_audio(self):
        """Verifies _generate_subtitles timestamp calculations match actual audio timing."""
        import tempfile
        import types
        from audiobook_factory.pipeline import _generate_subtitles

        cfg = AudiobookConfig(export_srt=True, pause=0.5, sample_rate=24000)
        chapter = types.SimpleNamespace(title="Test Chapter")
        tts_jobs = ["Line 1", "Line 2", "Line 3"]
        chunk_durations = [2.0, 3.0, 1.5]

        with tempfile.TemporaryDirectory() as tmpdir:
            from audiobook_factory.filename_sanitizer import make_safe_filename
            cfg.output_dir = tmpdir
            _generate_subtitles(cfg, chapter, 1, tts_jobs, chunk_durations, lambda msg: None)
            srt_name = make_safe_filename(chapter.title, 1, tmpdir, ".srt")
            srt_path = os.path.join(tmpdir, srt_name)
            self.assertTrue(os.path.exists(srt_path))

            with open(srt_path, "r", encoding="utf-8") as f:
                content = f.read()

            # Line 1: 0.0s -> 2.0s
            # Line 2: 2.5s (2.0 + 0.5 pause) -> 5.5s
            # Line 3: 6.0s (5.5 + 0.5 pause) -> 7.5s
            self.assertIn("00:00:00,000 --> 00:00:02,000", content)
            self.assertIn("00:00:02,500 --> 00:00:05,500", content)
            self.assertIn("00:00:06,000 --> 00:00:07,500", content)


    def test_mark_chapter_completed_guards(self):
        """Verifies _mark_chapter_completed enforces existence and _MINIMUM_CHAPTER_WAV_BYTES."""
        import tempfile
        from audiobook_factory.pipeline import _mark_chapter_completed
        with tempfile.TemporaryDirectory() as tmp_dir:
            prog_path = os.path.join(tmp_dir, "progress.json")
            write_progress_file(prog_path, {"chapters": [{"num": 1, "status": "pending"}]})

            # 1. Missing file -> status remains pending
            _mark_chapter_completed(prog_path, 1, os.path.join(tmp_dir, "nonexistent.wav"))
            data = read_progress_file(prog_path)
            self.assertEqual(data["chapters"][0]["status"], "pending")

            # 2. Too small file -> status remains pending
            small_wav = os.path.join(tmp_dir, "ch1.wav")
            with open(small_wav, "wb") as f:
                f.write(b"0" * 100)
            _mark_chapter_completed(prog_path, 1, small_wav)
            data = read_progress_file(prog_path)
            self.assertEqual(data["chapters"][0]["status"], "pending")

            # 3. Valid size file -> status updated to completed
            valid_wav = os.path.join(tmp_dir, "valid_ch1.wav")
            with open(valid_wav, "wb") as f:
                f.write(b"0" * 15000)
            _mark_chapter_completed(prog_path, 1, valid_wav)
            data = read_progress_file(prog_path)
            self.assertEqual(data["chapters"][0]["status"], "completed")

    def test_finalize_progress_file(self):
        """Verifies _finalize_progress_file writes top-level generation_summary."""
        import tempfile
        from audiobook_factory.pipeline import _finalize_progress_file
        with tempfile.TemporaryDirectory() as tmp_dir:
            prog_path = os.path.join(tmp_dir, "progress.json")
            write_progress_file(prog_path, {
                "chapters": [
                    {"num": 1, "status": "completed"},
                    {"num": 2, "status": "failed"},
                ]
            })
            _finalize_progress_file(prog_path, [1, 2])
            data = read_progress_file(prog_path)
            self.assertIn("generation_summary", data)
            summary = data["generation_summary"]
            self.assertEqual(summary["completed_count"], 1)
            self.assertEqual(summary["failed_count"], 1)
            self.assertEqual(summary["failed_chapters"], [2])
            self.assertFalse(summary["all_complete"])


class TestMultiGPUScheduling(unittest.TestCase):
    """Tests for _get_chapter_parallelism() adaptive dispatch."""

    def _make_mock_pool(self, device_count: int):
        pool = MagicMock()
        pool.device_count = device_count
        pool.devices = [f"cuda:{i}" for i in range(device_count)]
        return pool

    def _make_config(self, parallel_mode: str) -> AudiobookConfig:
        return AudiobookConfig(parallel_mode=parallel_mode)

    def test_chunks_mode_returns_1_on_single_gpu(self):
        pool = self._make_mock_pool(1)
        config = self._make_config("chunks")
        result = _get_chapter_parallelism(pool, config)
        self.assertEqual(result, 1)

    def test_chunks_mode_returns_1_on_dual_gpu(self):
        """chunks mode must return 1 regardless of GPU count.
        All GPUs are used via chunk splitting, not chapter splitting."""
        pool = self._make_mock_pool(2)
        config = self._make_config("chunks")
        result = _get_chapter_parallelism(pool, config)
        self.assertEqual(result, 1)

    def test_chapters_mode_returns_device_count(self):
        pool = self._make_mock_pool(2)
        config = self._make_config("chapters")
        result = _get_chapter_parallelism(pool, config)
        self.assertEqual(result, 2)

    def test_chapters_mode_single_gpu_returns_1(self):
        pool = self._make_mock_pool(1)
        config = self._make_config("chapters")
        result = _get_chapter_parallelism(pool, config)
        self.assertEqual(result, 1)

    def test_none_config_defaults_to_chunks_behavior(self):
        """None config must behave like parallel_mode='chunks'."""
        pool = self._make_mock_pool(2)
        result = _get_chapter_parallelism(pool, None)
        self.assertEqual(result, 1)

    def test_max_parallel_chapters_override_respected(self):
        """_MAX_PARALLEL_CHAPTERS > 0 must override config."""
        import audiobook_factory.pipeline as pipeline_module
        original = pipeline_module._MAX_PARALLEL_CHAPTERS
        try:
            pipeline_module._MAX_PARALLEL_CHAPTERS = 1
            pool = self._make_mock_pool(2)
            config = self._make_config("chunks")
            result = _get_chapter_parallelism(pool, config)
            self.assertEqual(result, 1)
        finally:
            pipeline_module._MAX_PARALLEL_CHAPTERS = original


class TestPhase1AndPhase2Hardening(unittest.TestCase):

    def test_mark_chapter_completed_valid_file(self):
        import tempfile
        from audiobook_factory.pipeline import _mark_chapter_completed
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = os.path.join(tmpdir, "progress.json")
            wav_path = os.path.join(tmpdir, "ch1.wav")
            write_progress_file(json_path, {"chapters": [{"num": 1, "status": "pending", "completed_chunks": [0]}]})
            with open(wav_path, "wb") as f:
                f.write(b"0" * 12_000)
            _mark_chapter_completed(json_path, 1, wav_path)
            data = read_progress_file(json_path)
            self.assertEqual(data["chapters"][0]["status"], "completed")

    def test_mark_chapter_completed_small_file(self):
        import tempfile
        from audiobook_factory.pipeline import _mark_chapter_completed
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = os.path.join(tmpdir, "progress.json")
            wav_path = os.path.join(tmpdir, "ch1.wav")
            write_progress_file(json_path, {"chapters": [{"num": 1, "status": "pending"}]})
            with open(wav_path, "wb") as f:
                f.write(b"0" * 100)
            _mark_chapter_completed(json_path, 1, wav_path)
            data = read_progress_file(json_path)
            self.assertEqual(data["chapters"][0]["status"], "pending")

    def test_finalize_progress_file(self):
        import tempfile
        from audiobook_factory.pipeline import _finalize_progress_file
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = os.path.join(tmpdir, "progress.json")
            write_progress_file(json_path, {"chapters": [{"num": 1, "status": "completed"}, {"num": 2, "status": "failed"}]})
            _finalize_progress_file(json_path, [])
            data = read_progress_file(json_path)
            self.assertIn("generation_summary", data)
            summary = data["generation_summary"]
            self.assertEqual(summary["total_chapters"], 2)
            self.assertEqual(summary["completed_count"], 1)
            self.assertEqual(summary["failed_count"], 1)
            self.assertEqual(summary["failed_chapters"], [2])
            self.assertFalse(summary["all_complete"])

    def test_update_chapter_retry(self):
        import tempfile
        from audiobook_factory.progress_io import update_chapter_retry
        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = os.path.join(tmpdir, "progress.json")
            write_progress_file(json_path, {"chapters": [{"num": 1, "status": "pending"}]})
            update_chapter_retry(json_path, 1, 2, "OOM Error")
            data = read_progress_file(json_path)
            ch = data["chapters"][0]
            self.assertEqual(ch["status"], "failed")
            self.assertEqual(ch["retry_count"], 2)
            self.assertEqual(ch["last_error"], "OOM Error")


class TestPhase3AndPhase4Features(unittest.TestCase):

    def test_get_tts_provider_vibevoice_and_f5tts(self):
        from audiobook_factory.tts_providers import get_tts_provider
        cfg = AudiobookConfig()
        p1 = get_tts_provider("vibevoice", cfg)
        self.assertEqual(p1.get_name(), "VibeVoice-1.5B")
        p2 = get_tts_provider("f5tts", cfg)
        self.assertEqual(p2.get_name(), "F5-TTS")

    def test_suggest_batch_size_headroom(self):
        from audiobook_factory.gpu_pool import GPUDetector
        b1 = GPUDetector.suggest_batch_size("cpu", vram_headroom_gb=1.0)
        self.assertGreaterEqual(b1, 1)


if __name__ == "__main__":
    unittest.main()



