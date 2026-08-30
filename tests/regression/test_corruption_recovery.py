"""
test_corruption_recovery.py
===========================
Regression tests for progress JSON self-healing against BOM, garbage prefix, and corruption.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from audiobook_factory.progress_io import (
    read_progress_file,
    write_progress_file,
    _strip_leading_garbage,
)


class TestCorruptionRecoveryRegression:

    def test_strip_leading_garbage_variations(self):
        assert _strip_leading_garbage('{"a": 1}') == '{"a": 1}'
        assert _strip_leading_garbage('junk before {"a": 1}') == '{"a": 1}'
        assert _strip_leading_garbage('\ufeff{"a": 1}') == '{"a": 1}'
        assert _strip_leading_garbage(' \n\r\t {"a": 1}') == '{"a": 1}'

    def test_read_progress_with_corrupted_headers(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "corrupt.json")
            valid_payload = {
                "book_title": "Healed Book",
                "settings": {"config_version": 6},
                "chapters": []
            }
            # Prepend terminal logs / control characters before valid JSON
            corrupt_str = "LOG [12:00:00] Initializing...\r\n== PROGRESS START ==\n" + json.dumps(valid_payload)
            with open(path, "w", encoding="utf-8") as f:
                f.write(corrupt_str)

            data = read_progress_file(path)
            assert data["book_title"] == "Healed Book"
