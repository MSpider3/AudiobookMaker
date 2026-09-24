"""
test_config_contract.py
========================
Unit tests for AudiobookConfig configuration contract, serialization,
schema migrations, and parameter validation.
"""

from __future__ import annotations

import os
import sys
import logging
import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from audiobook_factory.pipeline import AudiobookConfig, _CONFIG_SCHEMA_VERSION, _validate_config


class TestConfigContract:

    def test_default_values(self):
        cfg = AudiobookConfig()
        assert cfg.config_version == _CONFIG_SCHEMA_VERSION
        assert cfg.book_title == "Audiobook"
        assert cfg.output_format == "mp3"
        assert cfg.tts_provider_name == "qwen"
        assert cfg.temperature == 0.3
        assert cfg.top_p == 0.8
        assert cfg.max_len == 399
        assert cfg.pause == 0.5
        assert cfg.lufs == -18
        assert cfg.true_peak == -1.5
        assert cfg.worker_count == 1
        assert cfg.parallel_mode == "chunks"
        assert cfg.quantization == "none"
        assert cfg.resume_incomplete_chunks is True
        assert cfg.regen_missing is True

    def test_from_dict_unknown_keys_ignored(self):
        raw = {
            "book_title": "Custom Title",
            "non_existent_key": 9999,
            "another_spurious_field": "ignore_me",
        }
        cfg = AudiobookConfig.from_dict(raw)
        assert cfg.book_title == "Custom Title"
        assert not hasattr(cfg, "non_existent_key")
        assert not hasattr(cfg, "another_spurious_field")

    def test_from_dict_missing_keys_adopt_defaults(self):
        raw = {"author": "Leo Tolstoy"}
        cfg = AudiobookConfig.from_dict(raw)
        assert cfg.author == "Leo Tolstoy"
        assert cfg.book_title == "Audiobook"
        assert cfg.output_format == "mp3"
        assert cfg.max_len == 399

    def test_from_dict_schema_version_mismatch_warning(self, caplog):
        raw = {
            "config_version": 2,
            "book_title": "Old Version Book",
        }
        with caplog.at_level(logging.WARNING):
            cfg = AudiobookConfig.from_dict(raw)
        assert cfg.book_title == "Old Version Book"
        assert any("schema version 2" in record.message for record in caplog.records)

    def test_validate_config_quantization(self):
        cfg_valid_none = AudiobookConfig(quantization="none")
        _validate_config(cfg_valid_none)

        cfg_valid_int8 = AudiobookConfig(quantization="int8")
        _validate_config(cfg_valid_int8)

        cfg_invalid = AudiobookConfig(quantization="int4")
        with pytest.raises(ValueError, match="quantization"):
            _validate_config(cfg_invalid)

    def test_validate_config_parallel_mode(self):
        cfg_chunks = AudiobookConfig(parallel_mode="chunks")
        _validate_config(cfg_chunks)

        cfg_chapters = AudiobookConfig(parallel_mode="chapters")
        _validate_config(cfg_chapters)

        cfg_invalid = AudiobookConfig(parallel_mode="invalid_mode")
        with pytest.raises(ValueError, match="parallel_mode"):
            _validate_config(cfg_invalid)

    def test_validate_config_output_format(self):
        for fmt in ("mp3", "wav", "flac", "m4b", "m4a", "aac", "ogg", "webm", "mp4", "mov"):
            cfg = AudiobookConfig(output_format=fmt)
            _validate_config(cfg)

        cfg_invalid = AudiobookConfig(output_format="unsupported_fmt")
        with pytest.raises(ValueError, match="output_format"):
            _validate_config(cfg_invalid)

