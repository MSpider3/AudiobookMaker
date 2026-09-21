"""Test Gradio progress-JSON restore handler does not preset server paths (BUG-R2-C1-A2-H2 & BUG-R2-C1-A2-H3).

Ensures that uploading a progress JSON does not populate the book_file (gr.File)
or voice_studio_upload (gr.Audio) components with arbitrary server paths, preventing
unauthenticated arbitrary file downloads and information disclosure.
"""
import json
import os
import pytest
from app import on_progress_upload_handler


def test_on_progress_upload_does_not_preset_server_paths(tmp_path):
    """Ensure on_progress_upload does not set server paths in gr.File and gr.Audio components."""
    # Create fake server files
    fake_book = tmp_path / "secret_book.epub"
    fake_book.write_text("dummy epub content")
    fake_voice = tmp_path / "secret_voice.wav"
    fake_voice.write_text("dummy wav content")

    progress_file = tmp_path / "generation_progress.json"
    progress_file.write_text(json.dumps({
        "book_title": "Test Book",
        "book_path": str(fake_book),
        "voice_file": str(fake_voice),
        "settings": {
            "tts_provider_name": "qwen",
            "tts_model_name": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        },
        "chapters": [{"num": 1, "title": "Chapter 1", "status": "completed"}],
    }))

    class DummyFile:
        def __init__(self, path):
            self.name = path

    outputs = on_progress_upload_handler(DummyFile(str(progress_file)))
    status_msg = outputs[0]
    book_file_update = outputs[2]
    voice_file_update = outputs[3]

    # Status message must not echo server file existence oracle
    assert str(fake_book) not in status_msg
    assert str(fake_voice) not in status_msg

    # Component updates must not contain server file paths
    # In Gradio, gr.update() without value has no 'value' key or value is None/undefined
    if isinstance(book_file_update, dict):
        assert "value" not in book_file_update or book_file_update["value"] is None
    else:
        assert getattr(book_file_update, "value", None) is None

    if isinstance(voice_file_update, dict):
        assert "value" not in voice_file_update or voice_file_update["value"] is None
    else:
        assert getattr(voice_file_update, "value", None) is None
