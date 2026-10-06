"""
test_api_endpoints.py
=====================
Tests for the read-only endpoints, request validation, non-blocking handlers,
startup behaviour and the WebSocket lifecycle of the FastAPI backend.
"""

from __future__ import annotations

import asyncio
import json
import os
import socket
import sys
import threading
import time
from collections import defaultdict

import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import api.server as server_mod
import api.worker as worker_mod
from api.server import app
from api.worker import Task, tasks
from audiobook_factory.tts_providers.base_tts_provider import ProviderInfo, ProviderOption
from audiobook_factory.tts_providers.registry import provider_info, provider_names

_VOICE = os.path.join(_ROOT, "tests", "fixtures", "audio", "synthetic_voice_reference.wav")
_CHAPTER = {
    "num": 1,
    "title": "Chapter 1",
    "text": "The whole text of the chapter stays out of task listings.",
    "sentences": ["The whole text of the chapter stays out of task listings."],
}


@pytest.fixture(autouse=True)
def _api_env(monkeypatch, tmp_path):
    monkeypatch.setenv("ABM_SKIP_GPU_WARMUP", "1")
    monkeypatch.delenv("ABM_API_SECRET", raising=False)
    # A task the worker happens to pick up must not write into the project.
    monkeypatch.setenv("ABM_OUTPUT_BASE", str(tmp_path / "api_output"))
    # The limiter is shared by every test in the session.
    monkeypatch.setattr(server_mod._generate_limiter, "_requests", defaultdict(list))
    created = set(tasks)
    yield
    for task_id in set(tasks) - created:
        tasks.pop(task_id, None)


def _add_task(task_id: str, **fields) -> Task:
    task = Task(task_id=task_id, config_dict={"book_title": "Listed Book"}, chapters=[dict(_CHAPTER)])
    for name, value in fields.items():
        setattr(task, name, value)
    tasks[task_id] = task
    return task


def _payload(**config) -> dict:
    merged = {"book_title": "ValidationBook", "tts_provider_name": "mock"}
    merged.update(config)
    return {"config": merged, "chapters": [dict(_CHAPTER)]}


@pytest.fixture
def live_server(monkeypatch):
    """Runs api.server:app under a real uvicorn server; yields its port."""
    uvicorn = pytest.importorskip("uvicorn")
    # The module-level queue stays bound to whichever event loop used it
    # first, so give this server's loop a queue of its own.
    fresh_queue: asyncio.Queue = asyncio.Queue()
    monkeypatch.setattr(worker_mod, "task_queue", fresh_queue)
    monkeypatch.setattr(server_mod, "task_queue", fresh_queue)

    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True, name="test-api-server")
    thread.start()
    assert _wait_until(lambda: server.started or not thread.is_alive(), timeout=30) and server.started
    try:
        yield port
    finally:
        server.should_exit = True
        thread.join(timeout=30)


def _wait_until(predicate, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return predicate()


# ── GET /api/v1/providers ─────────────────────────────────────────────────────

class TestProvidersEndpoint:

    def test_lists_every_importable_provider_with_options(self):
        with TestClient(app) as client:
            res = client.get("/api/v1/providers")
        assert res.status_code == 200
        body = res.json()
        listed = {p["name"]: p for p in body["providers"]}
        unavailable = {p["name"] for p in body["unavailable"]}

        assert set(listed) | unavailable == set(provider_names())
        assert body["default"] in set(listed) | unavailable
        for name, entry in listed.items():
            info = provider_info(name)
            assert entry["display_name"] == info.display_name
            assert entry["license"] == info.license
            assert entry["commercial_use"] == info.commercial_use
            assert entry["default_model"] == info.default_model
            assert entry["models"] == list(info.models)
            assert entry["languages"] == list(info.languages)
            assert entry["preset_voices"] == list(info.preset_voices)
            assert entry["min_vram_gb"] == info.min_vram_gb
            assert entry["pip_requirements"] == list(info.pip_requirements)
            assert entry["install_notes"] == info.install_notes
            assert entry["supports_voice_clone"] == info.supports_voice_clone
            assert [o["key"] for o in entry["options"]] == [o.key for o in info.options]
            for option in entry["options"]:
                assert {"key", "label", "kind", "default", "minimum", "maximum", "step", "choices", "help"} <= set(option)

    def test_hidden_providers_only_on_request(self):
        with TestClient(app) as client:
            default = {p["name"] for p in client.get("/api/v1/providers").json()["providers"]}
            everything = {p["name"] for p in client.get("/api/v1/providers?include_hidden=true").json()["providers"]}
        assert "mock" not in default
        assert "mock" in everything

    def test_unimportable_provider_does_not_break_the_listing(self, monkeypatch):
        import audiobook_factory.tts_providers.registry as registry

        real = registry.provider_info

        def _flaky(name):
            if name == "qwen":
                raise ImportError("simulated broken module")
            return real(name)
        monkeypatch.setattr(registry, "provider_info", _flaky)
        with TestClient(app) as client:
            body = client.get("/api/v1/providers").json()
        assert {"name": "qwen", "error": "simulated broken module"} in body["unavailable"]
        assert "qwen" not in {p["name"] for p in body["providers"]}

    def test_requires_auth(self, monkeypatch):
        monkeypatch.setenv("ABM_API_SECRET", "key-123")
        with TestClient(app) as client:
            assert client.get("/api/v1/providers").status_code == 401
            assert client.get("/api/v1/providers", headers={"x-api-key": "key-123"}).status_code == 200


# ── GET /api/v1/tasks and ?since= ─────────────────────────────────────────────

class TestTaskEndpoints:

    def test_task_list_has_metadata_but_no_chapter_text(self):
        older = _add_task("list-older", created_at=1000.0, status="completed", finished_at=1010.0, progress=1.0)
        newer = _add_task("list-newer", created_at=2000.0)
        with TestClient(app) as client:
            res = client.get("/api/v1/tasks")
        assert res.status_code == 200
        listed = [t for t in res.json()["tasks"] if t["task_id"] in ("list-older", "list-newer")]

        assert [t["task_id"] for t in listed] == ["list-newer", "list-older"], "newest first"
        assert listed[0] == {
            "task_id": "list-newer", "status": "queued", "progress": 0.0,
            "book_title": "Listed Book", "chapter_count": 1,
            "created_at": 2000.0, "started_at": None, "finished_at": None,
            "output_file_count": 0, "error_message": None,
        }
        assert listed[1]["finished_at"] == 1010.0 and listed[1]["status"] == "completed"
        assert "stays out of task listings" not in res.text
        assert older.chapters and newer.chapters  # the listing does not touch them

    def test_task_list_requires_auth(self, monkeypatch):
        monkeypatch.setenv("ABM_API_SECRET", "key-123")
        with TestClient(app) as client:
            assert client.get("/api/v1/tasks").status_code == 401
            assert client.get("/api/v1/tasks", headers={"authorization": "Bearer key-123"}).status_code == 200

    def test_status_without_since_returns_the_whole_log(self):
        _add_task("since-all", logs=["one", "two", "three"])
        with TestClient(app) as client:
            body = client.get("/api/v1/tasks/since-all").json()
        assert body["logs"] == ["one", "two", "three"]
        assert body["log_count"] == 3 and body["log_offset"] == 0
        assert body["book_title"] == "Listed Book" and body["output_files"] == []

    def test_since_returns_only_new_log_lines(self):
        task = _add_task("since-some", logs=["one", "two", "three"])
        with TestClient(app) as client:
            first = client.get("/api/v1/tasks/since-some?since=2").json()
            assert first["logs"] == ["three"] and first["log_count"] == 3 and first["log_offset"] == 2

            caught_up = client.get(f"/api/v1/tasks/since-some?since={first['log_count']}").json()
            assert caught_up["logs"] == [] and caught_up["log_count"] == 3

            task.logs.append("four")
            later = client.get("/api/v1/tasks/since-some?since=3").json()
            assert later["logs"] == ["four"] and later["log_count"] == 4

            beyond = client.get("/api/v1/tasks/since-some?since=99").json()
            assert beyond["logs"] == [] and beyond["log_offset"] == 4

            assert client.get("/api/v1/tasks/since-some?since=-1").status_code == 422


# ── GET /api/v1/tasks/{id}/files/{index} ──────────────────────────────────────

class TestFileDownload:

    def test_serves_only_recorded_output_files(self, tmp_path):
        recorded = tmp_path / "Chapter 1 - One.mp3"
        recorded.write_bytes(b"ID3-fake-mp3-bytes")
        secret = tmp_path / "secret.txt"
        secret.write_text("not an output file", encoding="utf-8")
        _add_task("dl-done", status="completed", output_files=[str(recorded)])

        with TestClient(app) as client:
            ok = client.get("/api/v1/tasks/dl-done/files/0")
            assert ok.status_code == 200
            assert ok.content == b"ID3-fake-mp3-bytes"
            assert ok.headers["content-type"] == "audio/mpeg"
            assert "Chapter%201%20-%20One.mp3" in ok.headers["content-disposition"] \
                or "Chapter 1 - One.mp3" in ok.headers["content-disposition"]

            assert client.get("/api/v1/tasks/dl-done/files/1").status_code == 404
            assert client.get("/api/v1/tasks/dl-done/files/-1").status_code == 404
            # The client can only name an index, never a path.
            for attempt in ("secret.txt", "..%2Fsecret.txt", str(secret).replace("/", "%2F")):
                assert client.get(f"/api/v1/tasks/dl-done/files/{attempt}").status_code in (404, 422)
            assert client.get("/api/v1/tasks/no-such-task/files/0").status_code == 404

    def test_unfinished_task_is_409(self, tmp_path):
        recorded = tmp_path / "partial.mp3"
        recorded.write_bytes(b"x")
        _add_task("dl-running", status="running", output_files=[str(recorded)])
        with TestClient(app) as client:
            assert client.get("/api/v1/tasks/dl-running/files/0").status_code == 409

    def test_deleted_or_non_regular_output_is_404(self, tmp_path):
        _add_task("dl-gone", status="completed", output_files=[str(tmp_path / "gone.mp3"), str(tmp_path)])
        with TestClient(app) as client:
            assert client.get("/api/v1/tasks/dl-gone/files/0").status_code == 404
            assert client.get("/api/v1/tasks/dl-gone/files/1").status_code == 404

    def test_requires_auth(self, tmp_path, monkeypatch):
        recorded = tmp_path / "book.m4b"
        recorded.write_bytes(b"m4b")
        _add_task("dl-auth", status="completed", output_files=[str(recorded)])
        monkeypatch.setenv("ABM_API_SECRET", "key-123")
        with TestClient(app) as client:
            assert client.get("/api/v1/tasks/dl-auth/files/0").status_code == 401
            ok = client.get("/api/v1/tasks/dl-auth/files/0", headers={"x-api-key": "key-123"})
            assert ok.status_code == 200 and ok.headers["content-type"] == "audio/mp4"


# ── Request validation ────────────────────────────────────────────────────────

class TestRequestValidation:

    def _post(self, client, **config):
        return client.post("/api/v1/generate", json=_payload(**config))

    def test_valid_request_is_accepted(self):
        with TestClient(app) as client:
            res = self._post(client, voice_file=_VOICE, tts_options={"speed_hint": 1.0, "flag": True, "note": None})
        assert res.status_code == 200 and res.json()["status"] == "queued"

    def test_unknown_provider_is_rejected(self):
        with TestClient(app) as client:
            before = set(tasks)
            res = self._post(client, tts_provider_name="definitely-not-an-engine")
        assert res.status_code == 400
        detail = res.json()["detail"]
        assert detail["code"] == "unknown_provider" and detail["field"] == "tts_provider_name"
        assert set(tasks) == before, "a rejected request must not create a task"

    @pytest.mark.parametrize("field", ["voice_file", "cover_image", "voice_preset"])
    @pytest.mark.parametrize("bad", ["DIRECTORY", "/dev/zero", "/no/such/file.wav"])
    def test_path_fields_must_be_existing_regular_files(self, field, bad):
        value = os.path.dirname(_VOICE) if bad == "DIRECTORY" else bad
        with TestClient(app) as client:
            res = self._post(client, **{field: value})
        assert res.status_code == 400, res.text
        detail = res.json()["detail"]
        assert detail["code"] == "invalid_path" and detail["field"] == field

    @pytest.mark.parametrize("options", [{"nested": {"a": 1}}, {"items": [1, 2]}, ["a"], "a=b"])
    def test_tts_options_must_be_flat_scalars(self, options):
        with TestClient(app) as client:
            res = self._post(client, tts_options=options)
        assert res.status_code == 400
        assert res.json()["detail"]["code"] == "invalid_tts_options"

    def test_file_kind_options_must_be_existing_regular_files(self, monkeypatch):
        import audiobook_factory.tts_providers.registry as registry

        fake = ProviderInfo(name="mock", display_name="Mock", options=(
            ProviderOption(key="emotion_clip", label="Emotion clip", kind="file", default=""),
            ProviderOption(key="note", label="Note", kind="str", default=""),
        ))
        monkeypatch.setattr(registry, "provider_info", lambda name: fake)
        with TestClient(app) as client:
            bad = self._post(client, tts_options={"emotion_clip": "/etc"})
            assert bad.status_code == 400
            assert bad.json()["detail"] == {
                "code": "invalid_path",
                "message": "tts_options.emotion_clip is not an existing regular file: '/etc'",
                "field": "tts_options.emotion_clip",
            }
            # A str-kind option is not a path and is not checked as one.
            assert self._post(client, tts_options={"emotion_clip": _VOICE, "note": "/etc"}).status_code == 200
            assert self._post(client, tts_options={"emotion_clip": ""}).status_code == 200

    def test_output_dir_outside_base_is_rejected_with_the_base(self, tmp_path, monkeypatch):
        base = tmp_path / "base"
        base.mkdir()
        monkeypatch.setenv("ABM_OUTPUT_BASE", str(base))
        with TestClient(app) as client:
            res = self._post(client, output_dir=str(tmp_path / "elsewhere"))
            assert res.status_code == 400
            assert res.json()["detail"]["code"] == "output_dir_outside_base"
            assert res.json()["detail"]["output_base"] == str(base)
            assert self._post(client, output_dir=str(base / "Book")).status_code == 200
            assert self._post(client, output_dir="relative/Book").status_code == 200

    def test_voice_test_validates_the_same_way(self):
        with TestClient(app) as client:
            res = client.post("/api/v1/voice-test", json={
                "config": {"tts_provider_name": "nope"}, "text": "Hello.",
            })
            assert res.status_code == 400 and res.json()["detail"]["code"] == "unknown_provider"
            res = client.post("/api/v1/voice-test", json={
                "config": {"tts_provider_name": "mock", "voice_file": "/dev/zero"}, "text": "Hello.",
            })
            assert res.status_code == 400 and res.json()["detail"]["field"] == "voice_file"

    def test_prepare_config_canonicalises_and_contains(self, tmp_path, monkeypatch):
        base = tmp_path / "base"
        base.mkdir()
        monkeypatch.setenv("ABM_OUTPUT_BASE", str(base))
        cfg = worker_mod.prepare_config({"tts_provider_name": "Dummy", "output_dir": "My Book", "batch_size": 3})
        assert cfg.tts_provider_name == "mock"
        assert cfg.output_dir == str(base / "My Book")
        assert cfg.batch_size == 3
        with pytest.raises(worker_mod.ConfigError) as excinfo:
            worker_mod.prepare_config({"tts_provider_name": "mock", "output_dir": "../escape"})
        assert excinfo.value.code == "output_dir_outside_base"
        with pytest.raises(worker_mod.ConfigError):
            worker_mod.prepare_config(["not", "a", "dict"])


# ── Startup: no eager warmup ──────────────────────────────────────────────────

class TestStartup:

    @pytest.mark.parametrize("skip_flag", [None, "1"])
    def test_no_model_is_loaded_before_a_task_asks_for_one(self, monkeypatch, skip_flag):
        from audiobook_factory import tts_providers
        from audiobook_factory.gpu_pool import GPUPoolManager

        if skip_flag is None:
            monkeypatch.delenv("ABM_SKIP_GPU_WARMUP", raising=False)
        else:
            monkeypatch.setenv("ABM_SKIP_GPU_WARMUP", skip_flag)
        loaded = threading.Event()

        def _record(*args, **kwargs):
            loaded.set()
            raise RuntimeError("no pool may be built at startup")

        monkeypatch.setattr(GPUPoolManager, "get_pool", _record)
        monkeypatch.setattr(tts_providers, "get_tts_provider", _record)
        monkeypatch.setattr("audiobook_factory.tts_providers.base_tts_provider.get_tts_provider", _record)

        with TestClient(app) as client:
            assert client.get("/api/v1/health").json()["status"] == "ok"
            # The old warmup ran in a background thread right after startup.
            assert not loaded.wait(timeout=1.0), "a provider pool was built at startup"


# ── Handlers do not block the event loop ──────────────────────────────────────

class TestHandlersOffTheEventLoop:

    @staticmethod
    def _on_event_loop() -> bool:
        try:
            asyncio.get_running_loop()
            return True
        except RuntimeError:
            return False

    def test_voice_test_synthesizes_in_a_worker_thread(self, monkeypatch):
        seen = {}

        def _fake_preview(text, cfg):
            seen["on_loop"] = self._on_event_loop()
            seen["provider"] = cfg.tts_provider_name
            return b"RIFF-fake-wav"

        monkeypatch.setattr(server_mod, "preview_tts", _fake_preview)
        with TestClient(app) as client:
            res = client.post("/api/v1/voice-test", json={
                "config": {"tts_provider_name": "mock", "voice_file": _VOICE}, "text": "Hello there.",
            })
        assert res.status_code == 200 and res.content == b"RIFF-fake-wav"
        assert seen == {"on_loop": False, "provider": "mock"}

    def test_health_answers_while_a_preview_is_synthesizing(self, monkeypatch):
        started = threading.Event()
        release = threading.Event()

        def _slow_preview(text, cfg):
            started.set()
            release.wait(timeout=20)
            return b"RIFF-fake-wav"

        monkeypatch.setattr(server_mod, "preview_tts", _slow_preview)
        result = {}
        with TestClient(app) as client:
            def _request_preview():
                result["response"] = client.post("/api/v1/voice-test", json={
                    "config": {"tts_provider_name": "mock"}, "text": "A long preview.",
                })
            worker = threading.Thread(target=_request_preview, daemon=True)
            worker.start()
            try:
                assert started.wait(timeout=10), "the preview never started"
                began = time.monotonic()
                health = client.get("/api/v1/health")
                elapsed = time.monotonic() - began
                assert health.status_code == 200
                assert not release.is_set() and elapsed < 5, "health check waited for the synthesis"
            finally:
                release.set()
                worker.join(timeout=20)
        assert result["response"].status_code == 200

    def test_preprocess_runs_in_a_worker_thread(self, monkeypatch):
        seen = {}

        def _fake_preprocess(data, cfg, use_cache=False):
            seen["on_loop"] = self._on_event_loop()
            seen["bytes"] = len(data)
            seen["use_cache"] = use_cache
            return b"RIFF-clean"

        monkeypatch.setattr(server_mod, "voice_preprocess", _fake_preprocess)
        form = {
            "noise_reduce": "true", "noise_reduce_strength": "0.5", "noise_gate": "false",
            "noise_gate_threshold_db": "-40", "highpass_filter": "true", "highpass_cutoff_hz": "80",
            "silence_removal": "false", "silence_threshold_db": "-40", "min_segment_ms": "100",
            "max_silence_kept_ms": "300", "normalize_volume": "true", "normalize_target_dbfs": "-3",
            "formant_shift": "false", "formant_quefrency": "1.0", "formant_timbre": "1.0",
            "resample": "false", "target_sample_rate": "24000", "use_cache": "true",
        }
        with TestClient(app) as client:
            res = client.post(
                "/api/v1/preprocess", data=form,
                files={"audio_file": ("voice.wav", b"0123456789", "audio/wav")},
            )
        assert res.status_code == 200 and res.content == b"RIFF-clean"
        assert seen == {"on_loop": False, "bytes": 10, "use_cache": True}


# ── Cancel and WebSocket lifecycle ────────────────────────────────────────────

class TestCancelAndWebSocket:

    def test_cancelling_a_queued_task_broadcasts_session_end(self):
        task = _add_task("cancel-queued")
        subscriber: asyncio.Queue = asyncio.Queue()
        task.subscribers.append(subscriber)
        with TestClient(app) as client:
            res = client.post("/api/v1/tasks/cancel-queued/cancel")
        assert res.json() == {"task_id": "cancel-queued", "status": "cancelled"}

        events = []
        while not subscriber.empty():
            events.append(subscriber.get_nowait())
        kinds = [e["type"] for e in events]
        assert kinds[-2:] == ["status", "session_end"], kinds
        assert events[-1] == {
            "type": "session_end", "task_id": "cancel-queued", "status": "cancelled",
            "files": [], "success": False, "cancelled": True,
        }
        assert task.chapters == [], "a finished task must not keep the book text"
        assert task.finished_at is not None

    def test_websocket_client_hears_about_a_cancelled_queued_task_at_once(self):
        _add_task("cancel-ws")
        events = []
        with TestClient(app) as client:
            with client.websocket_connect("/api/v1/ws/cancel-ws") as ws:
                assert ws.receive_json()["type"] == "progress"
                assert ws.receive_json() == {"type": "status", "status": "queued"}
                began = time.monotonic()
                client.post("/api/v1/tasks/cancel-ws/cancel")
                try:
                    while True:
                        event = ws.receive_json()
                        events.append(event)
                        if event["type"] == "session_end":
                            break
                except WebSocketDisconnect:
                    pass
                elapsed = time.monotonic() - began
        kinds = [e["type"] for e in events]
        assert "session_end" in kinds, f"only got {kinds}"
        assert kinds.index("status") < kinds.index("session_end")
        # Delivered as an event, not discovered when the grace timeout closes the socket.
        assert elapsed < server_mod._WS_CLOSE_GRACE_SEC

    def test_disconnected_client_is_unsubscribed_in_testclient(self):
        task = _add_task("ws-leak")
        with TestClient(app) as client:
            with client.websocket_connect("/api/v1/ws/ws-leak") as ws:
                assert ws.receive_json()["type"] == "progress"
                assert ws.receive_json()["type"] == "status"
                assert len(task.subscribers) == 1
            # The client has hung up; the task never produces another event.
            assert _wait_until(lambda: not task.subscribers, timeout=5.0), "subscriber leaked after disconnect"
            assert task.status == "queued"

    def test_disconnected_client_is_noticed_by_a_real_server(self, live_server):
        """The handler used to wait on the task queue only, so under uvicorn
        (which does not cancel a handler when its client leaves) an idle
        task's handler and subscriber queue lived on, and held up shutdown.
        TestClient cancels the handler itself and cannot show this."""
        sync_client = pytest.importorskip("websockets.sync.client")
        task = _add_task("ws-real-leak")

        with sync_client.connect(f"ws://127.0.0.1:{live_server}/api/v1/ws/ws-real-leak") as ws:
            assert json.loads(ws.recv(timeout=10))["type"] == "progress"
            assert json.loads(ws.recv(timeout=10)) == {"type": "status", "status": "queued"}
            assert len(task.subscribers) == 1
        # The client has hung up; the task never produces another event.
        assert _wait_until(lambda: not task.subscribers, timeout=5.0), "subscriber leaked after disconnect"
        assert task.status == "queued"

    def test_real_server_delivers_session_end_for_a_cancelled_queued_task(self, live_server):
        sync_client = pytest.importorskip("websockets.sync.client")
        requests = pytest.importorskip("requests")
        task = _add_task("ws-real-cancel")

        kinds = []
        with sync_client.connect(f"ws://127.0.0.1:{live_server}/api/v1/ws/ws-real-cancel") as ws:
            json.loads(ws.recv(timeout=10))
            json.loads(ws.recv(timeout=10))
            res = requests.post(f"http://127.0.0.1:{live_server}/api/v1/tasks/ws-real-cancel/cancel", timeout=10)
            assert res.json()["status"] == "cancelled"
            began = time.monotonic()
            while "session_end" not in kinds:
                kinds.append(json.loads(ws.recv(timeout=10))["type"])
            assert time.monotonic() - began < server_mod._WS_CLOSE_GRACE_SEC
        assert kinds == ["log", "status", "session_end"]
        assert _wait_until(lambda: not task.subscribers, timeout=5.0)

    def test_history_is_replayed_once_then_live_events_follow(self):
        task = _add_task("ws-history", logs=["first line", "second line"], progress=0.25, status="running")
        with TestClient(app) as client:
            with client.websocket_connect("/api/v1/ws/ws-history") as ws:
                received = [ws.receive_json() for _ in range(4)]
                assert received == [
                    {"type": "log", "message": "first line"},
                    {"type": "log", "message": "second line"},
                    {"type": "progress", "progress": 0.25},
                    {"type": "status", "status": "running"},
                ]
                for sub in list(task.subscribers):
                    sub.put_nowait({"type": "log", "message": "live line"})
                assert ws.receive_json() == {"type": "log", "message": "live line"}

    def test_terminal_status_does_not_close_before_session_end(self):
        """Kept from the previous fix: 'completed' and 'session_end' follow the
        terminal status and must still be delivered."""
        task = _add_task("ws-terminal", status="running")
        with TestClient(app) as client:
            with client.websocket_connect("/api/v1/ws/ws-terminal") as ws:
                ws.receive_json()
                ws.receive_json()
                assert _wait_until(lambda: len(task.subscribers) == 1)
                sub = task.subscribers[0]
                sub.put_nowait({"type": "status", "status": "completed"})
                time.sleep(0.3)
                sub.put_nowait({"type": "completed", "files": ["/x/a.mp3"]})
                sub.put_nowait({"type": "session_end", "files": ["/x/a.mp3"], "status": "completed"})
                kinds = [ws.receive_json()["type"] for _ in range(3)]
        assert kinds == ["status", "completed", "session_end"]
