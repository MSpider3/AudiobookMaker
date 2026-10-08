"""
tests/conftest.py
=================
Suite-wide test setup.

The unit and integration tests exercise the pipeline with the mock engine and
assert on exact batch and chunk counts. Those counts depend on how many
devices the GPU pool finds, so the suite is pinned to CPU: on a machine with
two GPUs the mock pool would otherwise get two workers and tests written for
one device fail for reasons that have nothing to do with the code under test.

Set ``ABM_TEST_USE_GPU=1`` to run the suite against the visible GPUs anyway.
"""
from __future__ import annotations

import os

if os.environ.get("ABM_TEST_USE_GPU") != "1":
    # Must happen before anything initialises CUDA; conftest is imported
    # ahead of every test module.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
