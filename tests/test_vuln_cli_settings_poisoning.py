"""Test CLI settings poisoning protection (BUG-R4-C1-A4-H1).

When resuming via cli.py <progress.json>, untrusted progress files must not be
permitted to load arbitrary HuggingFace repos with trust_remote_code=True,
invalid TTS providers, or arbitrary system paths as output_dir.
"""
import argparse
import json
import os
import pytest
from cli import _load_config, _build_audiobook_config


def test_cli_rejects_untrusted_vibevoice_model(tmp_path):
    """Ensure cli.py rejects an untrusted VibeVoice model in progress JSON settings."""
    poisoned_json = tmp_path / "poisoned_progress.json"
    poisoned_json.write_text(json.dumps({
        "book_title": "Victim Book",
        "book_path": "",
        "voice_file": "",
        "settings": {
            "tts_provider_name": "vibevoice",
            "tts_model_name": "attacker/VibeVoice-malicious-repo",
            "output_dir": str(tmp_path / "audiobook_output" / "Victim Book"),
        },
        "chapters": [],
    }))

    parser = argparse.ArgumentParser()
    parser.add_argument("config_json")
    parser.add_argument("--book-path", default=None)
    parser.add_argument("--voice-file", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--output-format", default=None)
    parser.add_argument("--worker-count", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--tts-model-name", default=None)
    parser.add_argument("--cover-image", default=None)
    parser.add_argument("--force-reprocess", action="store_true", default=False)
    parser.add_argument("--quantization", default=None)
    parser.add_argument("--no-resume-chunks", action="store_true", default=False)

    args = parser.parse_args([str(poisoned_json)])

    with pytest.raises(ValueError, match="Untrusted or invalid tts_model_name"):
        _load_config(args)


def test_cli_rejects_untrusted_output_dir(tmp_path):
    """Ensure cli.py rejects an arbitrary system path in progress JSON output_dir."""
    poisoned_json = tmp_path / "poisoned_progress.json"
    poisoned_json.write_text(json.dumps({
        "book_title": "Victim Book",
        "book_path": "",
        "voice_file": "",
        "settings": {
            "tts_provider_name": "qwen",
            "tts_model_name": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
            "output_dir": "/etc/cron.d",
        },
        "chapters": [],
    }))

    parser = argparse.ArgumentParser()
    parser.add_argument("config_json")
    parser.add_argument("--book-path", default=None)
    parser.add_argument("--voice-file", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--output-format", default=None)
    parser.add_argument("--worker-count", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--tts-model-name", default=None)
    parser.add_argument("--cover-image", default=None)
    parser.add_argument("--force-reprocess", action="store_true", default=False)
    parser.add_argument("--quantization", default=None)
    parser.add_argument("--no-resume-chunks", action="store_true", default=False)

    args = parser.parse_args([str(poisoned_json)])

    with pytest.raises(ValueError, match="Untrusted output_dir"):
        _load_config(args)
