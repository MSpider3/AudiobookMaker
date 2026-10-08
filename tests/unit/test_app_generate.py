"""
tests/unit/test_app_generate.py
===============================
Drives the Generate tab's handlers against the mock TTS engine, in this
process: generate, result panel, in-page player, redo of one chapter, cancel,
re-attaching after a "page reload", the stale progress upload and the
Export Config → Restore round trip.

The handlers are the functions Gradio calls; only the browser is missing.
"""
from __future__ import annotations

import dataclasses
import json
import os
import sys
import threading
import time

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import gradio as gr  # noqa: E402

import app  # noqa: E402
from audiobook_factory.gpu_pool import GPUPoolManager  # noqa: E402
from audiobook_factory.pipeline import AudiobookConfig  # noqa: E402
from tests.fixtures.mock_provider import MockTTSProvider  # noqa: E402

_VOICE = os.path.join(_ROOT, "tests", "fixtures", "audio", "synthetic_voice_reference.wav")
_EPUB = os.path.join(_ROOT, "tests", "fixtures", "source_documents", "dummy_book.epub")
_TOKEN = "browser-token-0123456789"


@pytest.fixture(autouse=True)
def _isolated(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "_OUTPUT_DIR", str(tmp_path / "audiobook_output"))
    monkeypatch.setenv("ABM_API_URL", "")
    monkeypatch.setenv("ABM_SHOW_MOCK_PROVIDER", "1")
    monkeypatch.delenv("ABM_MULTI_USER", raising=False)
    monkeypatch.setattr(app, "_STREAM_POLL_SEC", 0.02)
    GPUPoolManager.instance().shutdown()
    with app._RUNS_LOCK:
        app._RUNS.clear()
    yield
    for run in app._active_runs():
        run.request_cancel()
    deadline = time.time() + 30
    while app._active_runs() and time.time() < deadline:
        time.sleep(0.05)
    GPUPoolManager.instance().shutdown()


@pytest.fixture
def book():
    return dict(zip(app._BOOK_SLOTS, app.on_book_upload(_EPUB, None)))


def _values(book, **overrides) -> tuple:
    ui = {
        "book_title": "UI Book", "author": "Mara Ellison", "language": "English",
        "output_format": "mp3", "lufs": -18,
        "selected_chapters": book["selected_chapters"]["value"][:2],
        "tts_provider_name": "mock", "voice_file": _VOICE, "tts_options": {},
        "export_lrc": False, "max_chapter_retries": 0,
    }
    ui.update(overrides)
    return tuple(ui.get(key) for key in app._UI_KEYS)


def _generate(book, values, *, redo=None, progress_upload=None, cache=None, file=_EPUB, token=_TOKEN):
    args = (file, book["scan_state"], "", False, progress_upload, cache, book["all_choices"], token)
    if redo is None:
        stream = app.on_generate(None, *args, *values)
    else:
        stream = app.on_redo(None, redo, *args, *values)
    outputs = [dict(zip(app._GEN_SLOTS, out)) for out in stream]
    assert all(len(out) == len(app._GEN_SLOTS) for out in outputs)
    return outputs


def _progress(title="UI Book") -> dict:
    with open(app._progress_path(title), encoding="utf-8") as fh:
        return json.load(fh)


def _slow_mock(monkeypatch, delay=0.25):
    real = MockTTSProvider.synthesize

    def slow(self, *args, **kwargs):
        time.sleep(delay)
        return real(self, *args, **kwargs)

    monkeypatch.setattr(MockTTSProvider, "synthesize", slow)


def test_generate_two_chapters_fills_result_player_and_redo(book):
    outputs = _generate(book, _values(book))
    final = outputs[-1]

    assert final["run_status"] == "**✅ Generation complete**"
    assert final["result_md"].startswith("### ✅ Generation complete")
    assert "2 completed, 0 failed" in final["result_md"]
    rows = final["result_table"]["value"]
    assert [row[:3] for row in rows] == [[1, "Prologue", "completed"], [2, "Chapter 1: The Salt Road", "completed"]]
    assert all(row[3] for row in rows), "durations are shown"

    player = final["player_dd"]["choices"]
    assert len(player) == 2 and all(os.path.isfile(path) for _, path in player)
    assert final["player_dd"]["value"] == player[0][1]
    assert app.on_player_select(player[1][1])["value"] == player[1][1]
    assert final["redo_select"]["choices"] == [("1. Prologue", 1), ("2. Chapter 1: The Salt Road", 2)]

    downloads = final["download_files"]
    assert sum(path.endswith(".mp3") for path in downloads) == 2
    assert any(path.endswith("generation_progress.json") for path in downloads)
    assert final["download_col"]["visible"] is True
    assert os.path.isfile(final["log_file"]["value"])
    assert final["run_key"] is None and final["chapters_cache"]["chapters"]

    # Progress: a bar with elapsed time in every update, never a "hide".
    for out in outputs:
        if isinstance(out["progress"], str):
            assert "<progress" in out["progress"] and "elapsed" in out["progress"]
        else:
            assert out["progress"] == gr.update()
    assert "100.0%" in final["progress"]


def test_redo_regenerates_only_the_chosen_chapter(book):
    first = _generate(book, _values(book))[-1]
    files = {os.path.basename(path): path for _, path in first["player_dd"]["choices"]}
    stamps = {name: os.stat(path).st_mtime_ns for name, path in files.items()}
    time.sleep(0.05)

    outputs = _generate(book, _values(book), redo=[2], cache=first["chapters_cache"])
    final = outputs[-1]
    assert final["result_md"].startswith("### ✅ Generation complete")
    assert [row[0] for row in final["result_table"]["value"]] == [2]
    log = outputs[-1]["log"]
    assert "[Chapter 1/1]" in log and "Prologue" not in log.split("[Pipeline] Starting")[-1]

    after = {name: os.stat(path).st_mtime_ns for name, path in files.items()}
    changed = [name for name in files if after[name] != stamps[name]]
    assert changed == [name for name in files if "Salt Road" in name]
    # Both chapters are still complete and playable afterwards.
    assert [c["status"] for c in _progress()["chapters"]] == ["completed", "completed"]
    assert len(final["player_dd"]["choices"]) == 2

    refused = _generate(book, _values(book), redo=[])
    assert len(refused) == 1 and "at least one completed chapter" in refused[0]["run_status"]


def test_validation_never_hides_the_progress_bar_and_starts_nothing(book):
    cases = {
        "upload a book": dict(file=None, values=_values(book)),
        "at least one chapter": dict(values=_values(book, selected_chapters=[])),
        "needs a narrator voice": dict(values=_values(book, voice_file=None)),
    }
    for expected, case in cases.items():
        outputs = _generate(book, case["values"], file=case.get("file", _EPUB))
        assert len(outputs) == 1, expected
        assert expected in outputs[0]["run_status"]
        assert outputs[0]["progress"] == gr.update(), "the progress bar slot must be left alone"
        assert outputs[0]["download_col"] == gr.update()
    assert not app._RUNS
    assert not os.path.exists(app._progress_path("UI Book"))


def test_cancel_reports_cancelled_not_complete(book, monkeypatch):
    _slow_mock(monkeypatch)
    values = _values(book, selected_chapters=book["selected_chapters"]["value"][:4])
    args = (_EPUB, book["scan_state"], "", False, None, None, book["all_choices"], _TOKEN)
    stream = app.on_generate(None, *args, *values)
    first = dict(zip(app._GEN_SLOTS, next(stream)))
    assert first["run_key"]

    deadline = time.time() + 30
    while "[Pipeline] Starting" not in app._RUNS[first["run_key"]].full_log() and time.time() < deadline:
        next(stream)
    message = app.on_cancel(None, first["run_key"], "UI Book", _TOKEN)
    assert "Cancellation requested" in message and "UI Book" in message

    final = [dict(zip(app._GEN_SLOTS, out)) for out in stream][-1]
    assert final["run_status"].startswith("**⛔ Cancelled")
    assert "complete" not in final["run_status"].lower()
    assert final["result_md"].startswith("### ⛔ Cancelled")
    assert "cancelled" in final["progress"] and "100.0%" not in final["progress"]
    assert app.on_cancel(None, None, "UI Book", _TOKEN) == "ℹ️ Nothing is being generated."


def test_cancel_during_extraction_stops_before_generation(book, monkeypatch):
    started = threading.Event()
    release = threading.Event()
    real_extract = app.extract

    def slow_extract(path, **kwargs):
        started.set()
        release.wait(10)
        return real_extract(path, **kwargs)

    monkeypatch.setattr(app, "extract", slow_extract)
    monkeypatch.setattr(app, "run_pipeline", lambda *a, **k: pytest.fail("the pipeline must not start"))
    args = (_EPUB, book["scan_state"], "", False, None, None, book["all_choices"], _TOKEN)
    stream = app.on_generate(None, *args, *_values(book, force_reprocess=True))
    first = dict(zip(app._GEN_SLOTS, next(stream)))
    assert started.wait(10)
    app.on_cancel(None, first["run_key"], "UI Book", _TOKEN)
    release.set()
    outputs = [dict(zip(app._GEN_SLOTS, out)) for out in stream]
    assert outputs[-1]["run_status"].startswith("**⛔ Cancelled")
    assert "Cancelled before generation started" in app._RUNS[first["run_key"]].full_log()


def test_cancel_before_the_backend_task_id_arrives_cancels_that_task(monkeypatch):
    """Item 20: the task is enqueued after Cancel was pressed; it must be cancelled, not run."""
    calls = []
    cfg = AudiobookConfig(tts_provider_name="mock", output_dir=os.path.join(app._OUTPUT_DIR, "B"))
    run = app._Run("k", "B", cfg.output_dir, "owner", "mp3")

    class _Reply:
        status_code = 200

        @staticmethod
        def json():
            return {"task_id": "task-1", "status": "queued"}

    def fake_request(method, path, **kwargs):
        calls.append((method, path))
        if path == "/api/v1/generate":
            run.cancel.cancel()               # the user presses Cancel while the request is in flight
        return _Reply()

    monkeypatch.setattr(app, "_api_request", fake_request)
    monkeypatch.setattr(app, "_follow_task", lambda run_, task_id: calls.append(("follow", task_id)))
    chapters = [app.ExtractedChapter(num=1, title="One", text="Some text.", sentences=[])]
    assert app._generate_via_api(run, cfg, chapters) is True
    assert calls == [("POST", "/api/v1/generate"), ("POST", "/api/v1/tasks/task-1/cancel"), ("follow", "task-1")]
    assert run.task_id == "task-1" and run.local is False


def test_backend_rejection_is_an_error_not_a_local_fallback(monkeypatch, book):
    class _Rejected:
        status_code = 400
        text = ""

        @staticmethod
        def json():
            return {"detail": {"code": "invalid_path", "message": "voice_file is not an existing regular file"}}

    monkeypatch.setattr(app, "is_api_healthy", lambda: True)
    monkeypatch.setattr(app, "_find_backend_task", lambda title: None)
    monkeypatch.setattr(app, "_api_request", lambda *a, **k: _Rejected())
    monkeypatch.setattr(app, "run_pipeline", lambda *a, **k: pytest.fail("must not generate locally"))
    final = _generate(book, _values(book))[-1]
    assert final["result_md"].startswith("### ❌ Generation failed")
    assert "voice_file is not an existing regular file" in final["result_md"]


def test_reload_reattaches_to_the_running_book_and_can_cancel_it(book, monkeypatch):
    _slow_mock(monkeypatch)
    values = _values(book, selected_chapters=book["selected_chapters"]["value"][:4])
    args = (_EPUB, book["scan_state"], "", False, None, None, book["all_choices"], _TOKEN)
    original = app.on_generate(None, *args, *values)
    first = dict(zip(app._GEN_SLOTS, next(original)))
    original.close()                                   # the page went away; the run did not
    run = app._RUNS[first["run_key"]]
    assert not run.done

    # A new page load (no Gradio state at all) finds the run again.
    reattached = app.on_reattach(None, _TOKEN)
    out = next(reattached)
    assert len(out) == len(app._GEN_SLOTS) + 2
    shown = dict(zip(app._GEN_SLOTS, out))
    assert "Re-attached" in shown["run_status"] and shown["run_key"] == run.key
    assert out[-2] == "UI Book" and "is being generated" in out[-1]

    # Pressing Generate for the same book joins that run instead of starting a second one.
    joined = app.on_generate(None, *args, *values)
    again = dict(zip(app._GEN_SLOTS, next(joined)))
    assert "already being generated" in again["run_status"] and again["run_key"] == run.key
    assert len(app._RUNS) == 1
    joined.close()

    # Another book is refused while this one occupies the engine.
    other = _generate(book, _values(book, book_title="Other Book"))
    assert len(other) == 1 and "is being generated" in other[0]["run_status"]
    # Test Voice would load a second model next to the run.
    _audio, message = app.on_test_voice("Hello.", *values)
    assert "after it finishes" in message

    # Cancel from the reloaded page: no run key in its state, only the title.
    assert "Cancellation requested" in app.on_cancel(None, None, "UI Book", _TOKEN)
    rest = list(reattached)
    final = dict(zip(app._GEN_SLOTS, rest[-2]))
    assert final["run_status"].startswith("**⛔ Cancelled")
    assert rest[-1][-1] == ""                           # the banner is cleared at the end
    assert run.done

    # Nothing running → the load event changes nothing.
    idle = list(app.on_reattach(None, _TOKEN))
    assert len(idle) == 1 and all(value == gr.update() for value in idle[0])


def test_uploaded_progress_is_used_once_and_never_resets_newer_progress(book, tmp_path):
    final = _generate(book, _values(book))[-1]
    assert [c["status"] for c in _progress()["chapters"]] == ["completed", "completed"]
    stale = tmp_path / "stale.json"
    data = _progress()
    for chapter in data["chapters"]:
        chapter["status"] = "pending"
    stale.write_text(json.dumps(data), encoding="utf-8")

    outputs = _generate(book, _values(book), progress_upload=str(stale), cache=final["chapters_cache"])
    cleared = [out["progress_upload"] for out in outputs if out["progress_upload"] != gr.update()]
    assert cleared == [gr.update(value=None)], "the upload component is cleared after its first use"
    log = outputs[-1]["log"]                      # each update carries the tail so far
    assert log.count("Already completed. Skipping.") == 2, "the stale upload must not reset finished chapters"
    assert [c["status"] for c in _progress()["chapters"]] == ["completed", "completed"]


def test_generation_without_the_book_file_uses_text_saved_in_the_json(book):
    _generate(book, _values(book))
    for name in os.listdir(os.path.dirname(app._progress_path("UI Book"))):
        if name.endswith(".mp3"):
            os.remove(os.path.join(os.path.dirname(app._progress_path("UI Book")), name))
    outputs = _generate(book, _values(book), file=None)
    assert outputs[-1]["result_md"].startswith("### ✅ Generation complete")
    assert len(outputs[-1]["player_dd"]["choices"]) == 2


def test_export_config_keeps_progress_and_restores_into_the_ui(book, tmp_path):
    values = _values(
        book, speed=1.2, verify_chunks="off", batch_size=2, pack_sentences=False,
        normalize_speech_text=False, voice_transcript="the words in the clip",
        pronunciation_table=[["Saltmarsh", "Salt marsh"]], cover_image=book["cover_image"],
        force_reprocess=False,
    )
    generated = _generate(book, values)[-1]

    # Export with a different selection: chapter 2 again plus chapter 3.
    choices = book["selected_chapters"]["value"]
    export_values = _values(
        book, speed=1.2, verify_chunks="off", batch_size=2, pack_sentences=False,
        normalize_speech_text=False, voice_transcript="the words in the clip",
        pronunciation_table=[["Saltmarsh", "Salt marsh"]], cover_image=book["cover_image"],
        selected_chapters=choices[1:3], force_reprocess=True,
    )
    status, file_update, _accordion, _cache = app.on_export_config(
        None, _EPUB, book["scan_state"], "", False, generated["chapters_cache"], book["all_choices"], _TOKEN,
        *export_values,
    )
    assert status.startswith("✅ **Config exported!**"), status
    exported = _progress()
    assert file_update["value"] == app._progress_path("UI Book")

    # Every config field is written, one-off instructions are not carried over.
    assert set(exported["settings"]) == {f.name for f in dataclasses.fields(AudiobookConfig)}
    assert exported["settings"]["force_reprocess"] is False and exported["settings"]["redo_chapters"] == []
    assert exported["settings"]["voice_transcript"] == "the words in the clip"
    assert exported["settings"]["pronunciation_map"] == {"Saltmarsh": "Salt marsh"}
    # The cover is embedded once.
    assert exported["cover_image_b64"] and "cover_image_b64" not in exported["settings"]
    assert os.path.isfile(exported["settings"]["cover_image"])

    by_num = {c["num"]: c for c in exported["chapters"]}
    assert sorted(by_num) == [1, 2, 3]
    assert by_num[1]["status"] == "completed", "a chapter outside the selection is kept"
    assert by_num[2]["status"] == "completed" and by_num[2]["duration"] > 0
    assert by_num[3]["status"] == "pending" and by_num[3]["completed_chunks"] == []
    assert all(c["text"] for c in exported["chapters"])
    for chapter in exported["chapters"]:
        assert chapter["num"] == {"Prologue": 1, "Chapter 1: The Salt Road": 2, "Chapter 2: Plan B": 3}[chapter["title"]]

    # The CLI reads the same file.
    cfg = AudiobookConfig.from_dict(exported["settings"])
    assert cfg.speed == 1.2 and cfg.verify_chunks == "off" and cfg.tts_provider_name == "mock"

    # Restore into a fresh UI.
    restored = dict(zip(app._RESTORE_SLOTS, app.on_progress_upload_handler(app._progress_path("UI Book"))))
    assert "Loaded Successfully" in restored["restore_status"]
    assert restored["tts_provider_name"]["value"] == "mock"
    assert restored["speed"]["value"] == 1.2 and restored["batch_size"]["value"] == 2
    assert restored["verify_chunks"]["value"] == "off"
    assert restored["pack_sentences"]["value"] is False and restored["normalize_speech_text"]["value"] is False
    assert restored["voice_transcript"]["value"] == "the words in the clip"
    assert restored["pronunciation_table"]["value"] == [["Saltmarsh", "Salt marsh"]]
    check = restored["selected_chapters"]
    assert [app._strip_label(v) for v in check["value"]] == ["Chapter 1: The Salt Road", "Chapter 2: Plan B"]
    assert set(check["value"]) <= {value for _, value in check["choices"]}

    # Uploading the book afterwards applies the restored selection to the real checklist.
    rescanned = dict(zip(app._BOOK_SLOTS, app.on_book_upload(_EPUB, restored["json_selected"])))
    assert [app._strip_label(v) for v in rescanned["selected_chapters"]["value"]] == [
        "Chapter 1: The Salt Road", "Chapter 2: Plan B",
    ]


def test_export_is_refused_while_the_book_is_generating(book, monkeypatch):
    _slow_mock(monkeypatch)
    args = (_EPUB, book["scan_state"], "", False, None, None, book["all_choices"], _TOKEN)
    stream = app.on_generate(None, *args, *_values(book))
    next(stream)
    status, _file, _accordion, _cache = app.on_export_config(
        None, _EPUB, book["scan_state"], "", False, None, book["all_choices"], _TOKEN, *_values(book),
    )
    assert "being generated" in status
    stream.close()


def test_audition_applies_the_pronunciation_fixes(book, monkeypatch):
    spoken = []
    monkeypatch.setattr(app, "_synthesize_preview", lambda cfg, text: spoken.append(text) or open(_VOICE, "rb").read())
    values = _values(book, pronunciation_table=[["Saltmarsh", "Salt marsh"], [r"Dr\.", "Doctor"]],
                     normalize_speech_text=False)
    audio, status = app.on_audition("Dr. Hale reached Saltmarsh.", *values)
    assert audio is not None
    assert spoken == ["Doctor Hale reached Salt marsh."]
    assert "2 fix(es) applied" in status and "Doctor Hale reached Salt marsh." in status
    assert app.on_audition("  ", *values)[1].startswith("⚠️")


def test_test_voice_speaks_through_the_mock_engine(book):
    audio, status = app.on_test_voice("A short test sentence for the narrator.", *_values(book, speed=1.5))
    assert status == "✅ Preview ready!"
    rate, samples = audio
    assert rate > 0 and len(samples) > 1000
    app._release_preview_provider()


class _FakeDesigner:
    """Stands in for an engine with voice design and presets."""

    calls: list = []

    def __init__(self):
        self.config = None

    def design_voice(self, instruct=None, text=None, language=None, *, force=False):
        self.calls.append(("design", instruct, language, force))
        with open(_VOICE, "rb") as fh:
            return fh.read(), 24000, "The designed sample sentence."

    def save_voice_preset(self, path, voice_ref=None, *, transcript=None):
        self.calls.append(("save", type(voice_ref).__name__, transcript))
        with open(path, "wb") as fh:
            fh.write(b"preset-bytes")
        return {"path": path}


def test_voice_design_and_presets_go_through_the_preview_provider(book, monkeypatch):
    if app._provider_info("omnivoice") is None:
        pytest.skip("omnivoice does not import here")
    monkeypatch.setenv("ABM_SKIP_ENGINE_CHECK", "1")
    fake = _FakeDesigner()
    _FakeDesigner.calls = []
    monkeypatch.setattr(app, "_preview_provider", lambda cfg: fake)
    values = _values(book, tts_provider_name="omnivoice", tts_instruct="female, low pitch",
                     voice_file=None, voice_transcript="spoken words")

    audio, status, design = app.on_design_voice(*values)
    assert os.path.isfile(audio) and "The designed sample sentence." in status
    assert design["text"] == "The designed sample sentence."
    app.on_reroll_voice(*values)
    assert [c for c in fake.calls if c[0] == "design"] == [
        ("design", "female, low pitch", "English", False), ("design", "female, low pitch", "English", True),
    ]

    file_update, message = app.on_save_designed_preset(design, *values)
    assert message.startswith("✅ Preset saved") and file_update["visible"] is True
    with open(file_update["value"], "rb") as fh:
        assert fh.read() == b"preset-bytes"
    assert fake.calls[-1] == ("save", "bytes", "The designed sample sentence.")
    assert app.on_save_designed_preset(None, *values)[1].startswith("⚠️ Design a voice sample first")

    with_clip = _values(book, tts_provider_name="omnivoice", voice_file=_VOICE, voice_transcript="spoken words")
    file_update, message = app.on_save_preset(*with_clip)
    assert message.startswith("✅ Preset saved")
    assert fake.calls[-1] == ("save", "str", "spoken words")

    # An engine without presets or design says so instead of failing.
    plain = _values(book)
    assert "cannot design" in app.on_design_voice(*plain)[1]
    assert "no voice presets" in app.on_save_preset(*plain)[1]


# ══════════════════════════════════════════════════════════════════════════════
# Backend (API) mode
# ══════════════════════════════════════════════════════════════════════════════

class _Json:
    def __init__(self, body, status_code=200):
        self._body, self.status_code, self.text = body, status_code, ""

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


def test_polling_fetches_only_new_log_lines(monkeypatch):
    monkeypatch.setattr(app, "_API_POLL_INTERVAL_SEC", 0.0)
    replies = [
        {"status": "running", "progress": 0.5, "logs": ["b", "c"], "log_offset": 1, "log_count": 3},
        {"status": "completed", "progress": 1.0, "logs": ["d"], "log_offset": 3, "log_count": 4,
         "output_files": ["/x/one.mp3"]},
    ]
    asked = []

    def fake(method, path, **kwargs):
        asked.append(kwargs["params"]["since"])
        return _Json(replies.pop(0))

    monkeypatch.setattr(app, "_api_request", fake)
    run = app._Run("k", "B", "/tmp/out", "owner", "mp3")
    run.add_log("a")                       # one line already arrived over the WebSocket
    run.remote_log_count = 1
    app._poll_task(run, "task-1")
    assert asked == [1, 3]
    assert run.full_log().split("\n") == ["a", "b", "c", "d"]
    assert run.final_status == "completed" and run.out_files == ["/x/one.mp3"]
    assert run.snapshot()[1] == 1.0


def test_polling_an_older_backend_that_returns_the_whole_log(monkeypatch):
    monkeypatch.setattr(app, "_API_POLL_INTERVAL_SEC", 0.0)
    replies = [
        {"status": "running", "progress": 0.2, "logs": ["a", "b"]},
        {"status": "failed", "progress": 0.2, "logs": ["a", "b", "c"], "error_message": "CUDA out of memory"},
    ]
    monkeypatch.setattr(app, "_api_request", lambda *a, **k: _Json(replies.pop(0)))
    run = app._Run("k", "B", "/tmp/out", "owner", "mp3")
    app._poll_task(run, "task-1")
    assert run.full_log().split("\n") == ["a", "b", "c"]
    assert run.final_status == "failed" and run.error == "CUDA out of memory"


def test_generate_joins_a_task_the_backend_is_already_running(monkeypatch, book):
    """UI restarted (or another tab started it): attach, do not enqueue a duplicate."""
    posted = []

    def fake(method, path, **kwargs):
        if method == "GET" and path == "/api/v1/tasks":
            return _Json({"tasks": [
                {"task_id": "old", "status": "completed", "book_title": "UI Book"},
                {"task_id": "live-task", "status": "running", "book_title": "UI Book"},
            ]})
        posted.append((method, path))
        return _Json({})

    def follow(run, task_id):
        run.add_log("line from the backend")
        run.final_status = "cancelled"

    monkeypatch.setattr(app, "is_api_healthy", lambda: True)
    monkeypatch.setattr(app, "_api_request", fake)
    monkeypatch.setattr(app, "_follow_task", follow)
    monkeypatch.setattr(app, "run_pipeline", lambda *a, **k: pytest.fail("must not generate locally"))
    outputs = _generate(book, _values(book))
    assert "backend is already generating" in outputs[0]["run_status"]
    assert not [call for call in posted if call[1] == "/api/v1/generate"]
    assert "Re-attached to backend task live-task" in outputs[-1]["log"]
    assert outputs[-1]["run_status"].startswith("**⛔ Cancelled")


def test_run_cancel_reaches_the_backend_task(monkeypatch):
    calls = []
    monkeypatch.setattr(app, "_api_request", lambda method, path, **k: calls.append((method, path)) or _Json({}))
    run = app._Run("k", "B", "/tmp/out", "owner", "mp3")
    run.task_id = "task-9"
    run.request_cancel()
    assert run.cancel.is_cancelled and calls == [("POST", "/api/v1/tasks/task-9/cancel")]
