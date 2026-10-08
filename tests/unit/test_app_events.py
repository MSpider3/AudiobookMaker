"""
tests/unit/test_app_events.py
=============================
Walks every event of the built Gradio app and checks that the handler returns
exactly one value per output component.

app.py has had several "handler returns N values, event has M outputs" bugs,
which Gradio only reports when the event fires in a browser. Each handler is
called here with the components' default values, which lands in its cheap
validation branch (no book, no voice, nothing running).
"""
from __future__ import annotations

import copy
import inspect
import itertools
import os
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import gradio as gr  # noqa: E402
from gradio import helpers  # noqa: E402

import app  # noqa: E402

# Handlers that must all receive the settings through the same components.
_CONFIG_HANDLERS: tuple[str, ...] = (
    "on_test_voice", "on_audition", "on_generate", "on_redo", "on_export_config",
    "on_save_preset", "on_design_voice", "on_reroll_voice", "on_save_designed_preset",
)


@pytest.fixture(scope="module")
def demo(tmp_path_factory):
    saved_env = {k: os.environ.get(k) for k in ("ABM_API_URL", "ABM_MULTI_USER")}
    saved_dir = app._OUTPUT_DIR
    os.environ["ABM_API_URL"] = ""          # never talk to a backend that happens to be running
    os.environ.pop("ABM_MULTI_USER", None)
    app._OUTPUT_DIR = str(tmp_path_factory.mktemp("audiobook_output"))
    try:
        yield app.build_app()
    finally:
        app._OUTPUT_DIR = saved_dir
        for key, value in saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def _events(demo):
    for fn in demo.fns.values():
        if fn.fn is None or fn.renderable is not None or getattr(fn, "rendered_in", None) is not None:
            continue
        yield fn


def _default_inputs(fn) -> list:
    values = [copy.deepcopy(getattr(block, "value", None)) for block in fn.inputs]
    return helpers.special_args(fn.fn, values, None, None)[0]


def _results(fn) -> list:
    result = fn.fn(*_default_inputs(fn))
    if inspect.isgenerator(result):
        return list(itertools.islice(result, 5))
    return [result]


def test_app_has_the_expected_events(demo):
    names = {fn.name for fn in _events(demo)}
    for expected in (*_CONFIG_HANDLERS, "on_book_upload", "on_progress_upload_handler", "on_reattach",
                     "on_cancel", "on_title_commit", "run_preprocess", "on_provider_change", "on_preview"):
        assert expected in names, f"{expected} is not wired to any event"


def test_every_handler_returns_one_value_per_output(demo):
    checked = 0
    for fn in _events(demo):
        expected = len(fn.outputs)
        results = _results(fn)
        assert results, f"{fn.name} yielded nothing"
        for value in results:
            if expected == 1:
                # A single output takes the return value as it is.
                assert not (isinstance(value, tuple) and len(value) != 1), fn.name
            else:
                assert isinstance(value, (tuple, list)), f"{fn.name} returned {type(value).__name__}"
                assert len(value) == expected, (
                    f"{fn.name}: {len(value)} value(s) returned for {expected} output component(s)"
                )
        checked += 1
    assert checked >= 25


def test_slot_tuples_match_the_wired_outputs(demo):
    by_name = {}
    for fn in _events(demo):
        by_name.setdefault(fn.name, fn)
    assert len(by_name["on_generate"].outputs) == len(app._GEN_SLOTS)
    assert len(by_name["on_redo"].outputs) == len(app._GEN_SLOTS)
    assert len(by_name["on_reattach"].outputs) == len(app._GEN_SLOTS) + 2
    assert len(by_name["on_progress_upload_handler"].outputs) == len(app._RESTORE_SLOTS)
    assert len(by_name["on_book_upload"].outputs) == len(app._BOOK_SLOTS)
    assert len(by_name["on_provider_change"].outputs) == len(app._PROVIDER_SLOTS)
    assert len(by_name["on_model_change"].outputs) == len(app._MODEL_SLOTS)
    assert len(by_name["on_title_commit"].outputs) == len(app._TITLE_SLOTS)
    for slots in (app._GEN_SLOTS, app._RESTORE_SLOTS, app._BOOK_SLOTS, app._PROVIDER_SLOTS):
        assert len(set(slots)) == len(slots), "a slot is listed twice"


def test_config_handlers_share_one_list_of_setting_components(demo):
    """Test Voice, Generate, Redo, Export… read the settings from the same components."""
    count = len(app._UI_KEYS)
    tails = {}
    for fn in _events(demo):
        if fn.name in _CONFIG_HANDLERS:
            assert len(fn.inputs) >= count, fn.name
            tails[fn.name] = [block._id for block in fn.inputs[-count:]]
    assert set(tails) == set(_CONFIG_HANDLERS)
    reference = tails["on_generate"]
    assert len(set(reference)) == count
    for name, ids in tails.items():
        assert ids == reference, f"{name} collects its settings from different components than Generate"


def test_request_is_injected_where_ownership_is_checked(demo):
    """``request`` must be annotated ``gr.Request`` or Gradio never passes it."""
    sentinel = object()
    for name in ("on_generate", "on_redo", "on_export_config", "on_cancel", "on_title_commit",
                 "on_reattach", "check_existing_progress"):
        fn = getattr(app, name)
        inputs = helpers.special_args(fn, [None] * 3, sentinel, None)[0]
        assert any(item is sentinel for item in inputs), f"{name} would not receive the request"


def test_dropdown_defaults_are_among_their_choices(demo):
    for block in demo.blocks.values():
        if not isinstance(block, (gr.Dropdown, gr.Radio)) or getattr(block, "allow_custom_value", False):
            continue
        values = [choice[1] for choice in block.choices]
        current = block.value
        for item in (current if isinstance(current, list) else [current]):
            assert item is None or item in values, f"{block.label}: {item!r} is not one of {values}"


def test_dead_and_removed_controls_are_gone(demo):
    labels = " | ".join(str(getattr(b, "label", "") or "") for b in demo.blocks.values()).lower()
    assert "workers" not in labels and "worker" not in labels      # worker_count is read by nothing
    assert "formant" not in labels and "quefrency" not in labels    # formant shifting was removed
    assert "nfe" not in labels                                      # now an engine option
