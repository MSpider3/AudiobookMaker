"""Test output filename and format path traversal prevention (BUG-R2-C2-A2-H3).

make_safe_filename() and pipeline configuration validation must ensure that
extensions and output formats cannot contain path separators or parent directory
references (e.g. '../', '/').
"""
import pytest
from audiobook_factory.filename_sanitizer import make_safe_filename
from audiobook_factory.pipeline import AudiobookConfig, _validate_config


def test_make_safe_filename_rejects_path_traversal_ext(tmp_path):
    """Ensure make_safe_filename rejects an extension with path traversal."""
    out_dir = str(tmp_path)
    
    with pytest.raises(ValueError, match="ext must be a simple extension suffix"):
        make_safe_filename("Chapter 1", 1, out_dir, ext="../escaped.mp3")
        
    with pytest.raises(ValueError, match="ext must be a simple extension suffix"):
        make_safe_filename("Chapter 1", 1, out_dir, ext=".mp3/evil")

    with pytest.raises(ValueError, match="ext must be a simple extension suffix"):
        make_safe_filename("Chapter 1", 1, out_dir, ext="..mp3")


def test_make_safe_filename_accepts_valid_ext(tmp_path):
    """Ensure legitimate audio extensions are accepted."""
    out_dir = str(tmp_path)
    fn = make_safe_filename("Chapter 1", 1, out_dir, ext=".mp3")
    assert fn.endswith(".mp3")
    assert "/" not in fn and "\\" not in fn


def test_validate_config_rejects_traversal_output_format():
    """Ensure _validate_config rejects output_format with path characters."""
    cfg = AudiobookConfig(
        book_title="Test",
        output_format="../mp3",
    )
    with pytest.raises(ValueError, match="Invalid output_format"):
        _validate_config(cfg)
        
    cfg2 = AudiobookConfig(
        book_title="Test",
        output_format="sub/mp3",
    )
    with pytest.raises(ValueError, match="Invalid output_format"):
        _validate_config(cfg2)
