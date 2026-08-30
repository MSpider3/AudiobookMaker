"""
test_websocket_stream.py
========================
Tests for FastAPI WebSocket live event streaming (/api/v1/ws/{task_id}).
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


class TestWebSocketStream:

    def test_websocket_receives_initial_events(self):
        with TestClient(app) as client:
            payload = {
                "config": {
                    "book_title": "WsTestBook",
                    "output_format": "mp3",
                    "tts_provider_name": "mock",
                    "preview_mode": True,
                },
                "chapters": [
                    {
                        "num": 1,
                        "title": "Chapter 1",
                        "text": "WebSocket test sentence.",
                        "sentences": ["WebSocket test sentence."]
                    }
                ]
            }
            gen_res = client.post("/api/v1/generate", json=payload)
            task_id = gen_res.json()["task_id"]

            events = []
            with client.websocket_connect(f"/api/v1/ws/{task_id}") as ws:
                # The server automatically sends progress and status baseline upon connection
                msg1 = ws.receive_json()
                events.append(msg1)
                msg2 = ws.receive_json()
                events.append(msg2)

            assert len(events) == 2
            event_types = [e.get("type") for e in events]
            assert "progress" in event_types
            assert "status" in event_types
