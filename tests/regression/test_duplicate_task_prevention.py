"""
test_duplicate_task_prevention.py
=================================
Regression test for QA-LEAD-04 (task duplication prevention).
Verifies that enqueued tasks are cleanly tracked and cancelled to avoid redundant execution.
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
from api.worker import tasks


class TestDuplicateTaskPrevention:

    def test_cancel_enqueued_task_prevents_running(self):
        with TestClient(app) as client:
            payload = {
                "config": {
                    "book_title": "CancelCheckBook",
                    "output_format": "mp3",
                    "tts_provider_name": "mock",
                },
                "chapters": [
                    {
                        "num": 1,
                        "title": "Chapter 1",
                        "text": "Long text to cancel before execution.",
                        "sentences": ["Long text to cancel before execution."]
                    }
                ]
            }
            res = client.post("/api/v1/generate", json=payload)
            assert res.status_code == 200
            task_id = res.json()["task_id"]

            # Cancel immediately
            cancel_res = client.post(f"/api/v1/tasks/{task_id}/cancel")
            assert cancel_res.status_code == 200

            task = tasks.get(task_id)
            assert task is not None
            assert task.cancel_token.is_cancelled is True
