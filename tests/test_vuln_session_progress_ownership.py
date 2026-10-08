"""Test Gradio session progress ownership and overwrite protections (BUG-R2-C1-A2-H5 & BUG-R4-C1-A4-H2).

The protection is opt-in (``ABM_MULTI_USER=1``, for hosting the UI for several
people). With it on:
1. Different users cannot overwrite each other's title-keyed generation_progress.json.
2. Different users cannot read/export each other's progress and cached chapter text.
3. The title-change progress existence oracle does not leak info across users.
4. A user keeps access after a page reload (the owner is a token stored in the
   browser, not Gradio's per-page-load session hash) and after a server restart.

Without it (the default, single-user machine) nobody is ever locked out: the
old always-on check locked users out of their own book after every reload.
"""
import json
import os

import pytest

import app
from app import (
    _get_session_id,
    _load_cached_chapters_if_available,
    _owns_progress,
    _register_progress_owner,
    check_existing_progress,
)

_TOKEN_A = "browser-token-aaaaaaaaaaaa"
_TOKEN_B = "browser-token-bbbbbbbbbbbb"


class DummyRequest:
    def __init__(self, session_hash):
        self.session_hash = session_hash


@pytest.fixture(autouse=True)
def _clean_owners(monkeypatch, tmp_path):
    monkeypatch.setattr(app, "_OUTPUT_DIR", str(tmp_path / "audiobook_output"))
    monkeypatch.setattr(app, "_PROGRESS_OWNERS", {})
    monkeypatch.delenv("ABM_MULTI_USER", raising=False)


@pytest.fixture
def multi_user(monkeypatch):
    monkeypatch.setenv("ABM_MULTI_USER", "1")


def _make_book(title="SharedTitle"):
    prog_file = app._progress_path(title)
    os.makedirs(os.path.dirname(prog_file), exist_ok=True)
    with open(prog_file, "w", encoding="utf-8") as fh:
        json.dump({
            "book_title": title,
            "chapters": [{"num": 1, "title": "One", "status": "completed", "text": "secret chapter text"}],
        }, fh)
    return prog_file


def _values(**settings):
    return tuple(settings.get(key) for key in app._UI_KEYS)


# ── Multi-user mode: the protection holds ────────────────────────────────────

def test_session_ownership_registration_and_check(tmp_path, multi_user):
    prog_path = str(tmp_path / "audiobook_output" / "Secret Book" / "generation_progress.json")

    # Session A registers ownership
    _register_progress_owner(prog_path, "session-a")

    assert _owns_progress(prog_path, "session-a") is True
    assert _owns_progress(prog_path, "session-b") is False
    assert _owns_progress(prog_path, "local") is True  # local CLI/server always permitted


def test_check_existing_progress_does_not_leak_cross_session(multi_user):
    prog_file = _make_book()

    # Owned by Session A
    _register_progress_owner(prog_file, "session-a")

    # Session A checks: receives progress info
    res_a = check_existing_progress("SharedTitle", request=DummyRequest("session-a"))
    assert "Existing Progress Found" in res_a

    # Session B checks: receives empty string (no information disclosure)
    res_b = check_existing_progress("SharedTitle", request=DummyRequest("session-b"))
    assert res_b == ""


def test_other_user_cannot_generate_export_or_read_cached_text(multi_user):
    prog_file = _make_book()
    _register_progress_owner(prog_file, _TOKEN_A)
    before = open(prog_file, encoding="utf-8").read()
    values = _values(book_title="SharedTitle", tts_provider_name="qwen")

    # Generate: refused before anything is read or written.
    outputs = list(app.on_generate(DummyRequest("hash-b"), None, None, "", False, None, None, None, _TOKEN_B, *values))
    assert len(outputs) == 1
    assert "Access denied" in dict(zip(app._GEN_SLOTS, outputs[0]))["run_status"]

    # Export: refused, nothing offered for download.
    status, file_update, _accordion, _cache = app.on_export_config(
        DummyRequest("hash-b"), None, None, "", False, None, None, _TOKEN_B, *values)
    assert "Access denied" in status and file_update.get("value") is None

    # Cached chapter text and the result panel stay private.
    assert _load_cached_chapters_if_available(prog_file, session_id=_TOKEN_B) is None
    assert _load_cached_chapters_if_available(prog_file, request=DummyRequest("hash-b")) is None
    shown = dict(zip(app._TITLE_SLOTS, app.on_title_commit(DummyRequest("hash-b"), "SharedTitle", _TOKEN_B)))
    assert shown["existing_progress"] == "" and shown["result_md"] == ""

    # The owner still gets all of it.
    assert _load_cached_chapters_if_available(prog_file, session_id=_TOKEN_A)[0].text == "secret chapter text"
    assert open(prog_file, encoding="utf-8").read() == before


def test_owner_survives_page_reload_and_server_restart(multi_user, monkeypatch):
    """The lock-out bug: ``session_hash`` changes on every page load."""
    prog_file = _make_book()
    first_load = _get_session_id(DummyRequest("hash-of-first-page-load"), _TOKEN_A)
    _register_progress_owner(prog_file, first_load)

    # Reload: Gradio hands out a new session hash, the browser token is the same.
    after_reload = _get_session_id(DummyRequest("hash-of-second-page-load"), _TOKEN_A)
    assert after_reload == first_load
    assert _owns_progress(prog_file, after_reload) is True
    assert "Existing Progress Found" in check_existing_progress(
        "SharedTitle", _TOKEN_A, DummyRequest("hash-of-second-page-load"))

    # Server restart: the in-memory table is gone, the marker next to the file is not.
    monkeypatch.setattr(app, "_PROGRESS_OWNERS", {})
    assert _owns_progress(prog_file, _TOKEN_A) is True
    assert _owns_progress(prog_file, _TOKEN_B) is False

    # A client without a token falls back to the session hash.
    assert _get_session_id(DummyRequest("hash-x"), "") == "hash-x"
    assert _get_session_id(None) == "local"


def test_other_user_cannot_see_or_cancel_a_running_book(multi_user):
    run = app._Run("key-a", "SharedTitle", app._book_output_dir("SharedTitle"), _TOKEN_A, "mp3")
    with app._RUNS_LOCK:
        app._RUNS[run.key] = run
    try:
        assert app._visible_runs(_TOKEN_B) == []
        assert app.on_cancel(DummyRequest("hash-b"), "key-a", "SharedTitle", _TOKEN_B) == "ℹ️ Nothing is being generated."
        assert run.cancel.is_cancelled is False
        assert list(app.on_reattach(DummyRequest("hash-b"), _TOKEN_B))[0][0] == app.gr.update()
        assert app._visible_runs(_TOKEN_A) == [run]
        assert "Cancellation requested" in app.on_cancel(DummyRequest("hash-a2"), None, "SharedTitle", _TOKEN_A)
        assert run.cancel.is_cancelled is True
    finally:
        run.finish()
        with app._RUNS_LOCK:
            app._RUNS.pop(run.key, None)


# ── Default single-user mode: nobody is locked out ───────────────────────────

def test_single_user_mode_never_locks_anyone_out():
    prog_file = _make_book()
    _register_progress_owner(prog_file, "hash-of-first-page-load")

    assert _owns_progress(prog_file, "hash-of-second-page-load") is True
    assert "Existing Progress Found" in check_existing_progress(
        "SharedTitle", request=DummyRequest("hash-of-second-page-load"))
    assert _load_cached_chapters_if_available(prog_file, request=DummyRequest("another")) is not None
    assert not os.path.exists(os.path.join(os.path.dirname(prog_file), app._OWNER_FILE_NAME))

    values = _values(book_title="SharedTitle", tts_provider_name="qwen")
    outputs = list(app.on_generate(DummyRequest("another"), None, None, "", False, None, None, None, "", *values))
    assert "Access denied" not in str(dict(zip(app._GEN_SLOTS, outputs[0]))["run_status"])


def test_client_token_is_created_once_and_kept():
    token = app.ensure_client_token("")
    assert len(token) >= 16 and app.ensure_client_token(token) == token
    assert app.ensure_client_token(None) != app.ensure_client_token(None)
    assert app.on_page_load(token) == app.gr.update()      # a known browser's token is left alone
    assert len(app.on_page_load("")) >= 16
