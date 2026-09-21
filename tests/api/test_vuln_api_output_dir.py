"""Test output_dir containment in API worker (BUG-R2-C1-A2-H9).

The unauthenticated /api/v1/generate endpoint accepts an arbitrary output_dir
in config_dict. The worker must sanitize and contain output_dir inside the
server output base directory, rejecting any path traversal or out-of-tree targets.
"""
import asyncio
import pytest
from unittest.mock import patch, MagicMock
from api.worker import _process_single_task, Task, tasks, task_queue


@pytest.mark.asyncio
async def test_worker_rejects_out_of_tree_output_dir():
    """Ensure output_dir resolving outside ABM_OUTPUT_BASE fails the task."""
    task = Task(
        task_id="test-traversal-task",
        config_dict={
            "book_title": "Test",
            "output_dir": "/tmp/arbitrary_escaped_output_dir",
        },
        chapters=[],
    )
    tasks["test-traversal-task"] = task
    task_queue.put_nowait("test-traversal-task")
    sem = asyncio.Semaphore(1)
    
    with patch("api.worker.run_pipeline") as mock_pipeline:
        await _process_single_task("test-traversal-task", sem)
        
        # Pipeline must NOT have been called
        assert not mock_pipeline.called
        # Task status must be failed
        assert task.status == "failed"


@pytest.mark.asyncio
async def test_worker_allows_valid_in_tree_output_dir(tmp_path, monkeypatch):
    """Ensure a safe output_dir within base directory is accepted."""
    base_dir = tmp_path / "audiobook_output"
    base_dir.mkdir()
    monkeypatch.setenv("ABM_OUTPUT_BASE", str(base_dir))
    
    safe_target = str(base_dir / "my_novel")
    task = Task(
        task_id="test-safe-task",
        config_dict={
            "book_title": "SafeNovel",
            "output_dir": safe_target,
        },
        chapters=[],
    )
    tasks["test-safe-task"] = task
    task_queue.put_nowait("test-safe-task")
    sem = asyncio.Semaphore(1)
    
    with patch("api.worker.run_pipeline", return_value=[]):
        await _process_single_task("test-safe-task", sem)
        
        assert task.status == "completed"
