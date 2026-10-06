"""
tests/unit/test_app_helpers.py
==============================
Unit tests for the helpers behind the Gradio UI: the backend client, voice
preprocessing, the progress-file upload, the result panel and the small
utilities each reviewed bug was fixed in.
"""
from __future__ import annotations

import dataclasses
import json
import os
import sys
import zipfile

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import gradio as gr  # noqa: E402

import app  # noqa: E402
from audiobook_factory.pipeline import AudiobookConfig  # noqa: E402
from audiobook_factory.text_extractor import ExtractedChapter  # noqa: E402
from audiobook_factory.voice_preprocessor import PreprocessConfig  # noqa: E402

_VOICE = os.path.join(_ROOT, "tests", "fixtures", "audio", "synthetic_voice_reference.wav")
_EPUB = os.path.join(_ROOT, "tests", "fixtures", "source_documents", "dummy_book.epub")
_PDF = os.path.join(_ROOT, "tests", "fixtures", "source_documents", "dummy_book.pdf")
_TXT = os.path.join(_ROOT, "tests", "fixtures", "source_documents", "dummy_book.txt")


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "_OUTPUT_DIR", str(tmp_path / "audiobook_output"))
    monkeypatch.setenv("ABM_API_URL", "")
    monkeypatch.delenv("ABM_API_SECRET", raising=False)
    monkeypatch.delenv("ABM_MULTI_USER", raising=False)
    monkeypatch.setattr(app, "_api_spec_cache", {"at": 0.0, "base": "", "spec": None})


class _Response:
    def __init__(self, status_code=200, body=None, content=b"", headers=None):
        self.status_code = status_code
        self._body = body
        self.content = content
        self.headers = headers or {}
        self.text = json.dumps(body) if body is not None else ""

    def json(self):
        if self._body is None:
            raise ValueError("no JSON")
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


# ══════════════════════════════════════════════════════════════════════════════
# Backend client
# ══════════════════════════════════════════════════════════════════════════════

class TestApiClient:

    def test_every_request_carries_the_api_key(self, monkeypatch):
        import requests

        seen = {}

        def fake_request(method, url, **kwargs):
            seen.update(method=method, url=url, headers=kwargs.get("headers"))
            return _Response(200, {"status": "ok"})

        monkeypatch.setattr(requests, "request", fake_request)
        monkeypatch.setenv("ABM_API_URL", "http://backend:9000/")
        monkeypatch.setenv("ABM_API_SECRET", "s3cret key")
        assert app.is_api_healthy() is True
        assert seen["url"] == "http://backend:9000/api/v1/health"
        assert seen["headers"] == {"x-api-key": "s3cret key"}
        app._api_request("POST", "/api/v1/tasks/abc/cancel", headers={"x-extra": "1"})
        assert seen["headers"] == {"x-api-key": "s3cret key", "x-extra": "1"}
        assert app._api_ws_url("task 1") == "ws://backend:9000/api/v1/ws/task%201?api_key=s3cret%20key"

    def test_no_secret_means_no_header_and_no_query(self, monkeypatch):
        monkeypatch.setenv("ABM_API_URL", "https://backend")
        assert app._api_headers() == {}
        assert app._api_ws_url("t") == "wss://backend/api/v1/ws/t"

    def test_empty_url_disables_the_backend(self, monkeypatch):
        import requests

        monkeypatch.setattr(requests, "request", lambda *a, **k: pytest.fail("no request expected"))
        assert app.is_api_healthy() is False
        with pytest.raises(requests.ConnectionError):
            app._api_request("GET", "/api/v1/health")

    def test_error_detail_reads_structured_and_plain_errors(self):
        structured = _Response(400, {"detail": {"code": "invalid_path", "message": "voice_file is missing"}})
        assert app._api_error_detail(structured) == "voice_file is missing (HTTP 400)"
        assert app._api_error_detail(_Response(500, {"detail": "TTS synthesis error: boom"})).startswith("TTS synthesis error: boom")
        validation = _Response(422, {"detail": [{"msg": "field required", "loc": ["body", "x"]}]})
        assert "field required" in app._api_error_detail(validation)

    def test_voice_test_shows_a_backend_error_instead_of_loading_a_second_model(self, monkeypatch):
        monkeypatch.setattr(app, "is_api_healthy", lambda: True)
        monkeypatch.setattr(app, "_api_request", lambda *a, **k: _Response(
            400, {"detail": {"code": "provider_unavailable", "message": "engine cannot be loaded"}}))
        monkeypatch.setattr(app, "preview_tts", lambda *a, **k: pytest.fail("must not synthesize locally"))
        with pytest.raises(app._UiError) as error:
            app._synthesize_preview(AudiobookConfig(tts_provider_name="mock"), "Hello")
        assert "engine cannot be loaded" in str(error.value)

    def test_voice_test_falls_back_only_when_the_backend_is_gone(self, monkeypatch):
        import requests

        def gone(*_args, **_kwargs):
            raise requests.ConnectionError("refused")

        monkeypatch.setattr(app, "is_api_healthy", lambda: True)
        monkeypatch.setattr(app, "_api_request", gone)
        monkeypatch.setattr(app, "preview_tts", lambda text, cfg: b"RIFFlocal")
        monkeypatch.setattr(app, "_stretch_preview", lambda wav, speed: wav)
        assert app._synthesize_preview(AudiobookConfig(tts_provider_name="mock"), "Hello") == b"RIFFlocal"

    def test_voice_test_sends_the_whole_config(self, monkeypatch):
        sent = {}

        def fake(method, path, **kwargs):
            sent.update(path=path, json=kwargs.get("json"))
            return _Response(200, content=b"RIFFbackend")

        monkeypatch.setattr(app, "is_api_healthy", lambda: True)
        monkeypatch.setattr(app, "_api_request", fake)
        monkeypatch.setattr(app, "_stretch_preview", lambda wav, speed: wav)
        cfg = AudiobookConfig(tts_provider_name="mock", voice_transcript="words", tts_options={"a": 1})
        assert app._synthesize_preview(cfg, "Hello") == b"RIFFbackend"
        assert sent["path"] == "/api/v1/voice-test"
        assert sent["json"]["text"] == "Hello"
        assert sent["json"]["config"] == dataclasses.asdict(cfg)


# ══════════════════════════════════════════════════════════════════════════════
# Voice preprocessing
# ══════════════════════════════════════════════════════════════════════════════

def _controls(**overrides):
    base = PreprocessConfig()
    values = dict(
        noise_reduce=base.noise_reduce, noise_strength=base.noise_reduce_strength,
        gate=base.noise_gate, gate_db=base.noise_gate_threshold_db, gate_range=base.noise_gate_range_db,
        highpass=base.highpass_filter, highpass_hz=base.highpass_cutoff_hz,
        trim_silence=base.trim_silence, shorten_pauses=base.silence_removal,
        min_segment_ms=base.min_segment_ms, max_silence_ms=base.max_silence_kept_ms,
        loudness_lufs=base.loudness_target_lufs,
        resample=base.resample, target_sr=base.target_sample_rate,
        best_window=base.select_best_window, best_window_seconds=base.best_window_seconds,
    )
    values.update(overrides)
    return tuple(values.values())


_OLD_FIELDS = {
    "noise_reduce", "noise_reduce_strength", "noise_gate", "noise_gate_threshold_db", "highpass_filter",
    "highpass_cutoff_hz", "silence_removal", "silence_threshold_db", "min_segment_ms", "max_silence_kept_ms",
    "normalize_volume", "normalize_target_dbfs", "formant_shift", "formant_quefrency", "formant_timbre",
    "resample", "target_sample_rate", "use_cache", "audio_file",
}


class TestPreprocess:

    def test_controls_at_their_defaults_give_the_default_config(self):
        cfg = app._preprocess_config(*_controls())
        assert cfg == PreprocessConfig()
        assert cfg.formant_shift is False

    def test_controls_map_to_the_new_fields(self):
        cfg = app._preprocess_config(*_controls(
            gate=True, gate_db=-50, gate_range=18, trim_silence=False, shorten_pauses=True,
            min_segment_ms=60, max_silence_ms=300, loudness_lufs=-23, target_sr=16000,
            best_window=True, best_window_seconds=12,
        ))
        assert (cfg.noise_gate, cfg.noise_gate_threshold_db, cfg.noise_gate_range_db) == (True, -50, 18)
        assert (cfg.trim_silence, cfg.silence_removal) == (False, True)
        assert (cfg.min_segment_ms, cfg.max_silence_kept_ms) == (60, 300)
        assert cfg.loudness_target_lufs == -23 and cfg.target_sample_rate == 16000
        assert (cfg.select_best_window, cfg.best_window_seconds) == (True, 12)

    def test_in_process_run_reports_metrics(self):
        audio, status, wav_bytes = app.run_preprocess(_VOICE, *_controls())
        assert audio is not None and wav_bytes[:4] == b"RIFF"
        assert "Duration" in status and "LUFS" in status and "SNR" in status and "Speech" in status

    def test_success_tick_only_without_warnings(self):
        clean = {"duration_s": 12.0, "loudness_lufs": -20.0, "snr_db": 40.0, "speech_ratio": 0.9, "warnings": []}
        assert app._report_markdown(clean, "Preprocessing complete.").startswith("✅ Preprocessing complete.")
        noisy = dict(clean, warnings=["The clip is only 2.0 s long.", "Clipping detected."])
        text = app._report_markdown(noisy, "Preprocessing complete.")
        assert "✅" not in text
        assert "- The clip is only 2.0 s long." in text and "- Clipping detected." in text
        assert "Duration 12.0 s · Loudness -20.0 LUFS · SNR 40 dB · Speech 90 %" in text

    def test_no_clip_and_unreadable_clip_are_reported(self, tmp_path):
        assert app.run_preprocess(None, *_controls())[1].startswith("⚠️")
        junk = tmp_path / "junk.wav"
        junk.write_bytes(b"not audio at all")
        audio, status, wav_bytes = app.run_preprocess(str(junk), *_controls())
        assert audio is None and wav_bytes is None and status.startswith("❌")

    def test_studio_upload_is_analysed(self):
        text = app.analyze_voice_clip(_VOICE)
        assert "Duration" in text
        assert app.analyze_voice_clip(None).startswith("*Upload")

    def test_backend_missing_a_changed_setting_is_not_used(self, monkeypatch):
        """Dropping a setting silently would process the clip differently from the screen."""
        monkeypatch.setattr(app, "is_api_healthy", lambda: True)
        monkeypatch.setattr(app, "_api_form_fields", lambda path: set(_OLD_FIELDS))
        monkeypatch.setattr(app, "_api_request", lambda *a, **k: pytest.fail("endpoint must not be called"))
        changed = app._preprocess_config(*_controls(best_window=True))
        assert app._preprocess_via_api(_VOICE, changed) is None
        # …and the handler then does the work in this process.
        audio, status, wav_bytes = app.run_preprocess(_VOICE, *_controls(best_window=True, best_window_seconds=5))
        assert wav_bytes is not None and "Duration" in status

    def test_backend_gets_every_field_it_accepts_and_its_report_is_used(self, monkeypatch):
        sent = {}
        fields = {f.name for f in dataclasses.fields(PreprocessConfig)} | {"use_cache", "audio_file"}
        with open(_VOICE, "rb") as fh:
            wav = fh.read()
        report = {"duration_s": 3.0, "loudness_lufs": -20.0, "snr_db": 30.0, "speech_ratio": 0.5,
                  "warnings": ["from the backend"]}

        def fake(method, path, **kwargs):
            sent.update(path=path, data=kwargs.get("data"), files=list(kwargs.get("files") or {}))
            return _Response(200, content=wav, headers={"content-type": "audio/wav",
                                                        "x-voice-report": json.dumps(report)})

        monkeypatch.setattr(app, "is_api_healthy", lambda: True)
        monkeypatch.setattr(app, "_api_form_fields", lambda path: fields)
        monkeypatch.setattr(app, "_api_request", fake)
        audio, status, wav_bytes = app.run_preprocess(_VOICE, *_controls(best_window=True, gate=True))
        assert wav_bytes == wav and audio is not None
        assert set(sent["data"]) == {f.name for f in dataclasses.fields(PreprocessConfig)}
        assert sent["data"]["select_best_window"] == "true" and sent["data"]["noise_gate"] == "true"
        assert sent["files"] == ["audio_file"]
        assert "from the backend" in status and "✅" not in status

    def test_backend_error_is_shown_not_hidden_by_a_fallback(self, monkeypatch):
        monkeypatch.setattr(app, "is_api_healthy", lambda: True)
        monkeypatch.setattr(app, "_api_form_fields", lambda path: set(_OLD_FIELDS))
        monkeypatch.setattr(app, "_api_request", lambda *a, **k: _Response(500, {"detail": "Preprocessing error: bad"}))
        monkeypatch.setattr(app, "preprocess_with_report", lambda *a, **k: pytest.fail("no local fallback"))
        audio, status, wav_bytes = app.run_preprocess(_VOICE, *_controls())
        assert audio is None and wav_bytes is None
        assert "Preprocessing error: bad" in status

    def test_form_fields_are_read_from_the_openapi_schema(self, monkeypatch):
        spec = {
            "paths": {"/api/v1/preprocess": {"post": {"requestBody": {"content": {
                "multipart/form-data": {"schema": {"$ref": "#/components/schemas/Body_pre"}}}}}}},
            "components": {"schemas": {"Body_pre": {"properties": {"noise_reduce": {}, "audio_file": {}}}}},
        }
        monkeypatch.setattr(app, "_api_openapi", lambda: spec)
        assert app._api_form_fields("/api/v1/preprocess") == {"noise_reduce", "audio_file"}
        assert app._api_form_fields("/api/v1/unknown") is None

    def test_processed_clip_goes_to_a_private_file(self):
        with open(_VOICE, "rb") as fh:
            wav = fh.read()
        _status_a, path_a = app.save_processed_voice(wav, _VOICE)
        _status_b, path_b = app.save_processed_voice(wav, _VOICE)
        assert path_a != path_b and os.path.basename(path_a) == "synthetic_voice_reference_processed.wav"
        assert app.save_processed_voice(None, _VOICE)[0].startswith("⚠️")


# ══════════════════════════════════════════════════════════════════════════════
# Book tab
# ══════════════════════════════════════════════════════════════════════════════

class TestBookScan:

    def _scan(self, path, saved=None):
        return dict(zip(app._BOOK_SLOTS, app.on_book_upload(path, saved)))

    def test_front_and_back_matter_is_listed_unticked_with_a_suffix(self):
        book = self._scan(_EPUB)
        choices = book["selected_chapters"]["choices"]
        ticked = book["selected_chapters"]["value"]
        matter = [display for display, value in choices if display.endswith(app._MATTER_SUFFIX)]
        assert matter, "the fixture has front/back matter"
        for display, value in choices:
            assert (value in ticked) != display.endswith(app._MATTER_SUFFIX)
            assert not value.endswith(app._MATTER_SUFFIX)          # the saved label stays CLI-compatible
        assert book["all_choices"]["default"] == ticked
        assert "front/back-matter" in book["book_note"]
        assert book["chapter_panel"]["visible"] is True

    def test_page_range_box_only_for_formats_with_pages(self):
        assert self._scan(_EPUB)["page_panel"]["visible"] is False
        assert self._scan(_TXT)["page_panel"]["visible"] is False
        pdf = self._scan(_PDF)
        assert pdf["page_panel"]["visible"] is True
        assert "Total pages" in pdf["total_pages"]

    def test_every_format_gets_a_chapter_checklist(self):
        for path in (_EPUB, _PDF, _TXT):
            book = self._scan(path)
            assert book["selected_chapters"]["value"], path

    def test_saved_selection_is_applied_once(self):
        saved = ["3. Chapter 2: Plan B  (~999 words)"]
        book = self._scan(_EPUB, saved)
        assert [app._strip_label(v) for v in book["selected_chapters"]["value"]] == ["Chapter 2: Plan B"]
        assert book["json_selected"] is None

    def test_no_file_and_bad_file(self, tmp_path):
        assert self._scan(None)["scan_status"].startswith("*Upload")
        broken = tmp_path / "broken.epub"
        broken.write_bytes(b"not a zip")
        book = self._scan(str(broken))
        assert len(book) == len(app._BOOK_SLOTS)
        assert book["selected_chapters"]["value"] == [] or "⚠️" in book["scan_status"] or "❌" in book["scan_status"]

    def test_empty_selection_is_refused_not_read_as_whole_book(self):
        choices = {"values": ["1. One  (~5 words)"], "default": ["1. One  (~5 words)"]}
        selection, ranges, problem = app._selection_inputs(None, "", choices, [])
        assert problem and "at least one chapter" in problem
        selection, ranges, problem = app._selection_inputs(None, "", choices, ["1. One  (~5 words)"])
        assert (selection, ranges, problem) == ([1], None, None)
        # No checklist at all (nothing could be listed): the whole file.
        assert app._selection_inputs(None, "", {"values": []}, []) == (None, None, None)

    def test_page_ranges_only_apply_where_supported(self):
        pdf = type("S", (), {"supports_page_ranges": True})()
        epub = type("S", (), {"supports_page_ranges": False})()
        choices = {"values": ["1. One  (~5 words)"]}
        assert app._selection_inputs(pdf, "1-3, 7-9, x, 9-2", choices, [])[1] == [(1, 3), (7, 9)]
        assert app._selection_inputs(epub, "1-3", choices, ["1. One  (~5 words)"])[1] is None

    def test_labels_round_trip(self):
        label = app._chapter_value(12, "Dr. Hale: A Life", 1234)
        assert label == "12. Dr. Hale: A Life  (~1,234 words)"
        assert app._strip_label(label) == "Dr. Hale: A Life"
        assert app._strip_label(label + app._MATTER_SUFFIX) == "Dr. Hale: A Life"
        assert app._label_num(label) == 12
        assert app._parse_chapter_titles([label, ""]) == ["Dr. Hale: A Life"]
        assert app._parse_chapter_titles([]) is None
        assert app._selection_from_labels([label]) == [12]
        assert app._selection_from_labels(["No number here"]) == ["No number here"]

    def test_extraction_is_reused_while_nothing_changed(self, monkeypatch):
        calls = []
        real_extract = app.extract

        def counting(path, **kwargs):
            calls.append(kwargs.get("selections"))
            return real_extract(path, **kwargs)

        monkeypatch.setattr(app, "extract", counting)
        chapters, cache, reused = app._extract_chapters(_EPUB, [2], None, False, None)
        assert not reused and len(chapters) == 1 and len(calls) == 1
        again, cache2, reused = app._extract_chapters(_EPUB, [2], None, False, cache)
        assert reused and again is chapters and len(calls) == 1
        _other, _cache3, reused = app._extract_chapters(_EPUB, [2, 3], None, False, cache)
        assert not reused and len(calls) == 2
        _ocr, _cache4, reused = app._extract_chapters(_EPUB, [2], None, True, cache)
        assert not reused

    def test_preview_uses_and_fills_the_session_cache(self):
        book = dict(zip(app._BOOK_SLOTS, app.on_book_upload(_EPUB, None)))
        selected = book["selected_chapters"]["value"][:2]
        status, table, cache = app.on_preview(_EPUB, book["scan_state"], "", False, None,
                                              book["all_choices"], selected, "English")
        assert "2 chapter(s)" in status and len(table["value"]) == 2
        assert [row[0] for row in table["value"]] == [1, 2]          # real chapter numbers
        assert cache["chapters"]
        refused = app.on_preview(_EPUB, book["scan_state"], "", False, None, book["all_choices"], [], "English")
        assert "at least one chapter" in refused[0]

    def test_first_paragraph_skips_the_heading(self):
        chapter = ExtractedChapter(
            num=1, title="Chapter 1: The Salt Road",
            text="Chapter 1: The Salt Road\n\nShort.\n\nDr. Imogen Hale arrived on the third of March with two "
                 "trunks, a bicycle and a letter from the harbour master.\n\nSecond paragraph.",
            sentences=[],
        )
        assert app._first_paragraph(chapter).startswith("Dr. Imogen Hale arrived")
        book = dict(zip(app._BOOK_SLOTS, app.on_book_upload(_EPUB, None)))
        text, status = app.on_use_book_paragraph(
            _EPUB, book["scan_state"], "", False, None, book["all_choices"],
            book["selected_chapters"]["value"], "English",
        )
        assert isinstance(text, str) and len(text.split()) >= 8 and "Test text taken from" in status


# ══════════════════════════════════════════════════════════════════════════════
# Progress file, result panel, downloads
# ══════════════════════════════════════════════════════════════════════════════

def _write(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh)


def _read(path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


class TestUploadedProgress:

    def test_copied_only_when_the_book_has_no_progress_yet(self, tmp_path):
        upload, dest = str(tmp_path / "up.json"), str(tmp_path / "book" / "generation_progress.json")
        _write(upload, {"book_title": "B", "settings": {"speed": 1.2},
                        "chapters": [{"num": 1, "title": "One", "status": "pending", "text": "t"}]})
        assert app._apply_uploaded_progress(upload, dest) == "copied"
        assert _read(dest)["chapters"][0]["status"] == "pending"

    def test_existing_statuses_survive_a_stale_upload(self, tmp_path):
        upload, dest = str(tmp_path / "up.json"), str(tmp_path / "book" / "generation_progress.json")
        _write(upload, {"book_title": "B", "settings": {"speed": 1.2}, "chapters": [
            {"num": 1, "title": "One", "status": "pending", "text": "one text", "sentences": ["one text"]},
            {"num": 2, "title": "Two", "status": "pending", "text": "two text"},
            {"num": 3, "title": "Three", "status": "pending", "text": "three text"},
        ]})
        _write(dest, {"book_title": "B", "settings": {"speed": 1.0}, "chapters": [
            {"num": 1, "title": "One", "status": "completed", "duration": 12.5, "completed_chunks": []},
            {"num": 2, "title": "Two", "status": "failed", "last_error": "boom", "text": "kept text"},
        ]})
        assert app._apply_uploaded_progress(upload, dest) == "merged"
        merged = _read(dest)
        by_num = {c["num"]: c for c in merged["chapters"]}
        assert by_num[1]["status"] == "completed" and by_num[1]["duration"] == 12.5
        assert by_num[1]["text"] == "one text"                      # gained the cached text it lacked
        assert by_num[2]["status"] == "failed" and by_num[2]["text"] == "kept text"
        assert by_num[3]["status"] == "pending"                     # unknown chapter was added
        assert merged["settings"] == {"speed": 1.2}                 # settings come from the upload

    def test_garbage_and_self_uploads_are_ignored(self, tmp_path):
        dest = str(tmp_path / "book" / "generation_progress.json")
        _write(dest, {"book_title": "B", "chapters": [{"num": 1, "title": "One", "status": "completed"}]})
        junk = tmp_path / "junk.json"
        junk.write_text("{ this is not json " * 5, encoding="utf-8")
        assert app._apply_uploaded_progress(str(junk), dest) == "skipped"
        assert app._apply_uploaded_progress(dest, dest) == "skipped"
        assert app._apply_uploaded_progress("", dest) == "skipped"
        assert _read(dest)["chapters"][0]["status"] == "completed"


class TestResultPanel:

    _ENTRIES = [
        {"num": 1, "title": "One", "status": "completed", "duration": 61.0,
         "flagged_chunks": [{"chunk": 4, "reason": "too short (0.4s)", "text": "A  flagged\nsentence."}]},
        {"num": 2, "title": "Two", "status": "failed", "last_error": "CUDA out of memory"},
        {"num": 3, "title": "Three", "status": "pending"},
    ]

    def test_rows(self):
        rows = app._result_rows(self._ENTRIES)
        assert rows[0] == [1, "One", "completed", "0:01:01", 1, ""]
        assert rows[1] == [2, "Two", "failed", "", 0, "CUDA out of memory"]
        assert len(rows[0]) == len(app._RESULT_HEADERS)

    def test_failed_run_is_not_reported_as_complete(self):
        text = app._result_markdown(self._ENTRIES)
        assert text.startswith("### ⚠ 1 chapter(s) failed")
        assert "complete" not in text.split("\n")[0].lower()
        assert "1 completed, 1 failed, total audio 0:01:01, 1 not done" in text
        assert "Chapter 1**, chunk 4 — too short (0.4s): “A flagged sentence.”" in text

    def test_cancelled_run_says_cancelled(self):
        text = app._result_markdown(self._ENTRIES, cancelled=True)
        assert text.startswith("### ⛔ Cancelled — 1 of 3 chapter(s) finished")

    def test_clean_run_says_complete(self):
        done = [{"num": 1, "title": "One", "status": "completed", "duration": 10}]
        assert app._result_markdown(done, out_files=["x.mp3"]).startswith("### ✅ Generation complete")
        assert app._result_markdown([], error="backend rejected the job").startswith("### ❌ Generation failed")
        assert app._result_markdown([]).startswith("### ⚠ No output files")
        assert app._result_markdown(done, finished_run=False).startswith("### 📚 Saved progress")

    def test_title_commit_shows_saved_progress_and_clears_it_again(self, tmp_path):
        prog = app._progress_path("Saved Book")
        _write(prog, {"book_title": "Saved Book", "settings": {"output_format": "mp3"}, "chapters": self._ENTRIES})
        shown = dict(zip(app._TITLE_SLOTS, app.on_title_commit(None, "Saved Book", "")))
        assert "Existing Progress Found" in shown["existing_progress"]
        assert shown["result_table"]["visible"] is True and len(shown["result_table"]["value"]) == 3
        assert shown["redo_select"]["choices"] == [("1. One", 1)]
        cleared = dict(zip(app._TITLE_SLOTS, app.on_title_commit(None, "Another Book", "")))
        assert cleared["existing_progress"] == "" and cleared["result_md"] == ""
        assert cleared["result_table"]["visible"] is False and cleared["redo_select"]["choices"] == []

    def test_player_only_lists_files_inside_the_book_folder(self, tmp_path):
        book = tmp_path / "audiobook_output" / "B"
        book.mkdir(parents=True)
        inside = book / "Chapter 1 - One.mp3"
        inside.write_bytes(b"x")
        outside = tmp_path / "secret.mp3"
        outside.write_bytes(b"x")
        choices = app._player_choices(str(book), {"chapters": []}, [str(inside), str(outside), str(book / "missing.mp3")])
        assert choices == [("Chapter 1 - One.mp3", str(inside))]
        assert app.on_player_select(str(outside))["value"] is None
        assert app.on_player_select(None)["value"] is None


class TestRunLogAndProgress:

    def _run(self):
        return app._Run("key", "Book", "/tmp/out", "owner", "mp3")

    def test_log_box_gets_a_bounded_tail(self):
        run = self._run()
        for i in range(1000):
            run.log_q.put(f"line {i}")
        tail = run.tail().split("\n")
        assert len(tail) == app._LOG_TAIL_LINES + 1
        assert tail[0].startswith("… 600 earlier lines hidden") and tail[-1] == "line 999"
        assert len(run.full_log().split("\n")) == 1000
        assert run.log_q.empty()                     # nothing piles up in the queue itself

    def test_progress_sink_keeps_the_latest_fraction(self):
        run = self._run()
        for item in ((0.5, 4.0), (1.5, 4.0), ("bad", 0), (9.0, 4.0)):
            run.prog_q.put(item)
        assert run.snapshot() == (0, 1.0)

    def test_stream_yields_only_on_change(self, monkeypatch):
        monkeypatch.setattr(app, "_STREAM_POLL_SEC", 0.01)
        monkeypatch.setattr(app, "_STREAM_HEARTBEAT_SEC", 3600.0)
        monkeypatch.setattr(app, "_final_updates", lambda run: {"run_status": "end"})
        run = self._run()
        run.add_log("first")
        stream = app._stream_run(run)
        first = dict(zip(app._GEN_SLOTS, next(stream)))
        assert first["log"] == "first" and first["run_key"] == "key"

        import threading
        import time

        def later():
            time.sleep(0.15)                          # ~15 idle polls: nothing may be yielded for them
            run.set_progress(0.5)
            time.sleep(0.05)
            run.finish()

        threading.Thread(target=later, daemon=True).start()
        rest = [dict(zip(app._GEN_SLOTS, out)) for out in stream]
        assert len(rest) <= 3, f"{len(rest)} updates for one progress change and the end"
        assert rest[0]["log"] == gr.update()          # log unchanged → not sent again
        assert "50.0%" in rest[0]["progress"]
        assert rest[-1]["run_status"] == "end"

    def test_eta_from_recent_progress(self):
        eta = app._EtaEstimator(window_sec=100.0)
        assert eta.remaining(0.0) is None
        eta.update(0.0, 0.0)
        eta.update(0.1, 10.0)
        assert eta.remaining(10.0) == pytest.approx(90.0)
        eta.update(0.5, 50.0)
        assert eta.remaining(50.0) == pytest.approx(50.0)
        # Instantly skipped chapters drop out of the window after a while.
        jumpy = app._EtaEstimator(window_sec=30.0)
        jumpy.update(0.0, 0.0)
        jumpy.update(0.5, 1.0)                        # half the book was already done
        jumpy.update(0.6, 41.0)
        jumpy.update(0.7, 81.0)
        assert jumpy.remaining(81.0) == pytest.approx(120.0)

    def test_progress_html_shows_counter_elapsed_and_eta(self):
        text = app._progress_html(0.25, elapsed=65, remaining=195, chapter=2, total_chapters=8)
        assert "25.0%" in text and "chapter 2 of 8" in text
        assert "elapsed 0:01:05" in text and "about 0:03:15 left" in text
        assert "estimating" in app._progress_html(0.25, elapsed=5)
        assert 'value="100.00"' in app._progress_html(7.0)


class TestDownloads:

    def test_zip_is_stored_and_private_to_the_call(self, tmp_path):
        files = []
        for name in ("a.mp3", "b.mp3"):
            path = tmp_path / name
            path.write_bytes(os.urandom(2048))
            files.append(str(path))
        twin = tmp_path / "other"
        twin.mkdir()
        (twin / "a.mp3").write_bytes(b"second a")
        first = app._make_zip(files + [str(twin / "a.mp3"), str(tmp_path / "missing.mp3")])
        second = app._make_zip(files)
        assert first != second and os.path.basename(first) == "AudiobookMaker_output.zip"
        assert not first.startswith(app._OUTPUT_DIR)
        with zipfile.ZipFile(first) as archive:
            assert archive.namelist() == ["a.mp3", "b.mp3", "a (2).mp3"]
            assert all(info.compress_type == zipfile.ZIP_STORED for info in archive.infolist())
        assert app.on_zip(None)["visible"] is False
        assert app.on_zip(files)["visible"] is True

    def test_merge_chapter_entries_keeps_resume_state_and_other_chapters(self):
        previous = [
            {"num": 1, "title": "One", "status": "completed", "duration": 9.0, "completed_chunks": [0, 1],
             "retry_count": 1, "flagged_chunks": [{"chunk": 0}], "text": "old"},
            {"num": 2, "title": "Two", "status": "failed", "last_error": "x", "text": "two"},
            {"num": 7, "title": "Seven", "status": "completed", "text": "seven"},
        ]
        chapters = [
            ExtractedChapter(num=1, title="One", text="new one", sentences=["new one"]),
            ExtractedChapter(num=3, title="Three", text="three", sentences=[]),
        ]
        merged = app._merge_chapter_entries(previous, chapters)
        by_num = {c["num"]: c for c in merged}
        assert [c["num"] for c in merged] == [1, 2, 3, 7]
        assert by_num[1]["status"] == "completed" and by_num[1]["duration"] == 9.0
        assert by_num[1]["completed_chunks"] == [0, 1] and by_num[1]["retry_count"] == 1
        assert by_num[1]["flagged_chunks"] == [{"chunk": 0}] and by_num[1]["text"] == "new one"
        assert by_num[3]["status"] == "pending" and by_num[3]["completed_chunks"] == []
        assert by_num[2]["last_error"] == "x" and by_num[7]["status"] == "completed"
