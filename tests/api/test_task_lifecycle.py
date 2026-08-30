"""
test_task_lifecycle.py
======================
Tests for complete API task lifecycle through background worker dispatch.
"""

from __future__ import annotations

import asyncio
import os
import sys
import pytest
from fastapi.testclient import TestClient

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from api.server import app
from api.worker import tasks, _process_single_task


class TestTaskLifecycle:

    @pytest.mark.asyncio
    async def test_task_process_lifecycle(self):
        payload = {
            "config": {
                "book_title": "LifecycleBook",
                "output_format": "mp3",
                "tts_provider_name": "mock",
                "preview_mode": True,
            },
            "chapters": [
                {
                    "num": 1,
                    "title": "Chapter 1",
                    "text": "Preview only test.",
                    "sentences": ["Preview only test."]
                }
            ]
        }
        with TestClient(app) as client:
            res = client.post("/api/v1/generate", json=payload)
            assert res.status_code == 200
            task_id = res.json()["task_id"]

            sem = asyncio.Semaphore(1)
            await _process_single_task(task_id, sem)

            final_st = client.get(f"/api/v1/tasks/{task_id}").json()
            assert final_st["status"] == "completed"
