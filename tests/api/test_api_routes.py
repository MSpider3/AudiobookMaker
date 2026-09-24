"""
test_api_routes.py
==================
Tests for FastAPI REST endpoints (/api/v1/health, /api/v1/generate, /api/v1/tasks).
"""

from __future__ import annotations

import os
import sys
import pytest
from fastapi.testclient import TestClient

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from api.server import app


class TestApiRoutes:

    def test_health_endpoint(self):
        with TestClient(app) as client:
            res = client.get("/api/v1/health")
            assert res.status_code == 200
            data = res.json()
            assert data.get("status") == "ok"
            assert "gpu" in data

    def test_generate_and_task_status_lifecycle(self):
        with TestClient(app) as client:
            payload = {
                "config": {
                    "book_title": "ApiTestBook",
                    "output_format": "mp3",
                    "tts_provider_name": "mock",
                    "preview_mode": True,
                },
                "chapters": [
                    {
                        "num": 1,
                        "title": "Chapter 1",
                        "text": "Hello API world.",
                        "sentences": ["Hello API world."]
                    }
                ]
            }
            res = client.post("/api/v1/generate", json=payload)
            assert res.status_code == 200
            task_data = res.json()
            assert "task_id" in task_data
            task_id = task_data["task_id"]

            # Poll status
            status_res = client.get(f"/api/v1/tasks/{task_id}")
            assert status_res.status_code == 200
            st = status_res.json()
            assert st["status"] in ("queued", "running", "completed")

    def test_get_nonexistent_task_returns_404(self):
        with TestClient(app) as client:
            res = client.get("/api/v1/tasks/non_existent_task_id_99999")
            assert res.status_code == 404

    def test_cancel_task(self):
        with TestClient(app) as client:
            payload = {
                "config": {
                    "book_title": "CancelTestBook",
                    "output_format": "mp3",
                    "tts_provider_name": "mock",
                },
                "chapters": [
                    {
                        "num": 1,
                        "title": "Chapter 1",
                        "text": "Long text to cancel.",
                        "sentences": ["Long text to cancel."]
                    }
                ]
            }
            res = client.post("/api/v1/generate", json=payload)
            task_id = res.json()["task_id"]

            cancel_res = client.post(f"/api/v1/tasks/{task_id}/cancel")
            assert cancel_res.status_code in (200, 400)

    def test_api_auth_enforced_when_secret_set(self, monkeypatch):
        monkeypatch.setenv("ABM_API_SECRET", "super-secret-key")
        with TestClient(app) as client:
            payload = {
                "config": {"book_title": "AuthTest"},
                "chapters": [{"num": 1, "title": "Ch1", "text": "Hi", "sentences": ["Hi"]}]
            }
            # Without key -> 401
            res_no_key = client.post("/api/v1/generate", json=payload)
            assert res_no_key.status_code == 401

            # With wrong key -> 401
            res_wrong = client.post("/api/v1/generate", json=payload, headers={"x-api-key": "wrong"})
            assert res_wrong.status_code == 401

            # With correct key -> 200
            res_ok = client.post("/api/v1/generate", json=payload, headers={"x-api-key": "super-secret-key"})
            assert res_ok.status_code == 200

    def test_task_eviction_limits_completed_tasks(self):
        from api.worker import tasks, Task, evict_old_tasks
        # Create 60 terminal tasks
        for i in range(60):
            tid = f"task-evict-{i}"
            t = Task(task_id=tid, config_dict={}, chapters=[])
            t.status = "completed"
            tasks[tid] = t

        evict_old_tasks(max_completed=50)
        completed_count = sum(1 for tid, t in tasks.items() if tid.startswith("task-evict-"))
        assert completed_count == 50
        # Oldest 10 (0..9) must be evicted
        assert "task-evict-0" not in tasks
        assert "task-evict-9" not in tasks
        assert "task-evict-59" in tasks

        # Clean up test tasks
        for i in range(60):
            tasks.pop(f"task-evict-{i}", None)

