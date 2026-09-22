"""
tests/unit/test_app_type_hints.py
=================================
Regression test to ensure type annotations across app.py and cli.py can be
evaluated via typing.get_type_hints without raising NameError (such as 'Any'
or 'AudiobookConfig').

Gradio event listeners inspect callback signatures via typing.get_type_hints(),
which evaluates string annotations in the module namespace.
"""
from __future__ import annotations

import inspect
import typing
import pytest


def test_app_functions_type_hints():
    """Verify all functions in app.py have valid, evaluatable type annotations."""
    import app

    functions_checked = 0
    for name, obj in inspect.getmembers(app):
        if inspect.isfunction(obj) and getattr(obj, "__module__", None) == "app":
            hints = typing.get_type_hints(obj)
            assert isinstance(hints, dict)
            functions_checked += 1

    assert functions_checked >= 10, f"Expected at least 10 app functions, got {functions_checked}"


def test_check_existing_progress_type_hints():
    """Explicitly verify check_existing_progress type hints evaluate cleanly with Any."""
    from app import check_existing_progress, _get_session_id, on_progress_upload_handler

    hints_check = typing.get_type_hints(check_existing_progress)
    assert "book_title" in hints_check
    assert "request" in hints_check

    hints_sess = typing.get_type_hints(_get_session_id)
    assert "request" in hints_sess

    hints_upload = typing.get_type_hints(on_progress_upload_handler)
    assert "file_obj" in hints_upload


def test_cli_functions_type_hints():
    """Verify all functions in cli.py have valid, evaluatable type annotations."""
    import cli

    functions_checked = 0
    for name, obj in inspect.getmembers(cli):
        if inspect.isfunction(obj) and getattr(obj, "__module__", None) == "cli":
            hints = typing.get_type_hints(obj)
            assert isinstance(hints, dict)
            functions_checked += 1

    assert functions_checked >= 15, f"Expected at least 15 cli functions, got {functions_checked}"


def test_build_app_succeeds():
    """Verify build_app creates Gradio Blocks without crashing on event listener registration."""
    from app import build_app

    demo = build_app()
    assert demo is not None
