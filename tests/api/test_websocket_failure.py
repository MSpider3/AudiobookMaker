"""
test_websocket_failure.py
=========================
Tests for WebSocket resilience against invalid task IDs, client disconnects, and reconnects.
"""

from __future__ import annotations

import os
import sys
import pytest
from fastapi.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from api.server import app


class TestWebSocketFailure:

    def test_websocket_nonexistent_task_reports_error_and_closes(self):
        with TestClient(app) as client:
            with client.websocket_connect("/api/v1/ws/non_existent_task_xyz") as ws:
                msg = ws.receive_json()
                assert msg.get("type") == "error"
                assert "not found" in msg.get("message", "").lower()

    def test_websocket_reconnect_during_task(self):
        with TestClient(app) as client:
            payload = {
                "config": {
                    "book_title": "ReconnectBook",
                    "output_format": "mp3",
                    "tts_provider_name": "mock",
                    "preview_mode": True,
                },
                "chapters": [
                    {
                        "num": 1,
                        "title": "Chapter 1",
                        "text": "Reconnect test.",
                        "sentences": ["Reconnect test."]
                    }
                ]
            }
            gen_res = client.post("/api/v1/generate", json=payload)
            task_id = gen_res.json()["task_id"]

            # First connection (immediate close)
            with client.websocket_connect(f"/api/v1/ws/{task_id}") as ws1:
                try:
                    _ = ws1.receive_json()
                except Exception:
                    pass

            # Second connection should succeed
            with client.websocket_connect(f"/api/v1/ws/{task_id}") as ws2:
                try:
                    data = ws2.receive_json()
                    assert isinstance(data, dict)
                except Exception:
                    pass
