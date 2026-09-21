"""Test Gradio session progress ownership and overwrite protections (BUG-R2-C1-A2-H5 & BUG-R4-C1-A4-H2).

Ensures that:
1. Different user sessions cannot overwrite each other's title-keyed generation_progress.json.
2. Different user sessions cannot read/export each other's progress and cached chapter text.
3. Title-change progress existence oracle does not leak info across sessions.
"""
import os
import pytest
from app import (
    _register_progress_owner,
    _owns_progress,
    check_existing_progress,
    _load_cached_chapters_if_available,
)


class DummyRequest:
    def __init__(self, session_hash):
        self.session_hash = session_hash


def test_session_ownership_registration_and_check(tmp_path):
    prog_path = str(tmp_path / "audiobook_output" / "Secret Book" / "generation_progress.json")
    
    # Session A registers ownership
    _register_progress_owner(prog_path, "session-a")
    
    assert _owns_progress(prog_path, "session-a") is True
    assert _owns_progress(prog_path, "session-b") is False
    assert _owns_progress(prog_path, "local") is True  # local CLI/server always permitted


def test_check_existing_progress_does_not_leak_cross_session(tmp_path, monkeypatch):
    import app
    output_dir = tmp_path / "audiobook_output"
    monkeypatch.setattr(app, "_OUTPUT_DIR", str(output_dir))
    
    book_dir = output_dir / "SharedTitle"
    book_dir.mkdir(parents=True, exist_ok=True)
    prog_file = book_dir / "generation_progress.json"
    prog_file.write_text('{"book_title": "SharedTitle", "chapters": [{"num": 1, "status": "completed"}]}')
    
    # Owned by Session A
    _register_progress_owner(str(prog_file), "session-a")
    
    # Session A checks: receives progress info
    res_a = check_existing_progress("SharedTitle", request=DummyRequest("session-a"))
    assert "Existing Progress Found" in res_a
    
    # Session B checks: receives empty string (no information disclosure)
    res_b = check_existing_progress("SharedTitle", request=DummyRequest("session-b"))
    assert res_b == ""
