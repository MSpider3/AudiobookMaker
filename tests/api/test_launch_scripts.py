"""
test_launch_scripts.py
======================
Tests for the launchers: start_api.py (uvicorn options) and run.sh (the API
server must not outlive the UI process).
"""

from __future__ import annotations

import os
import runpy
import shutil
import signal
import subprocess
import sys
import time

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

_FAKE_PYTHON = """#!/usr/bin/env bash
# Stand-in for the venv's python used by run.sh.
case "$1" in
    start_api.py)
        echo $$ > "$ABM_TEST_DIR/api.pid"
        exec sleep 300
        ;;
    app.py)
        echo $$ > "$ABM_TEST_DIR/app.pid"
        if [[ "${ABM_TEST_APP_MODE:-exit}" == "run" ]]; then
            exec sleep 300
        fi
        exit "${ABM_TEST_APP_EXIT:-0}"
        ;;
esac
"""


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    # A zombie still answers signal 0.
    try:
        with open(f"/proc/{pid}/stat", encoding="utf-8") as fh:
            return fh.read().rsplit(")", 1)[1].split()[0] != "Z"
    except OSError:
        return True


def _wait_for(predicate, timeout: float = 15.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.05)
    return predicate()


class TestStartApi:

    def test_uvicorn_gets_a_graceful_shutdown_timeout(self, monkeypatch):
        uvicorn = pytest.importorskip("uvicorn")
        calls = []
        monkeypatch.setattr(uvicorn, "run", lambda *args, **kwargs: calls.append((args, kwargs)))
        runpy.run_path(os.path.join(_ROOT, "start_api.py"), run_name="__main__")

        assert len(calls) == 1
        args, kwargs = calls[0]
        assert args == ("api.server:app",)
        assert kwargs["host"] == "127.0.0.1" and kwargs["port"] == 8000
        timeout = kwargs["timeout_graceful_shutdown"]
        assert isinstance(timeout, int) and 0 < timeout <= 60

    def test_importing_start_api_does_not_start_a_server(self, monkeypatch):
        uvicorn = pytest.importorskip("uvicorn")
        calls = []
        monkeypatch.setattr(uvicorn, "run", lambda *args, **kwargs: calls.append(kwargs))
        runpy.run_path(os.path.join(_ROOT, "start_api.py"), run_name="start_api")
        assert calls == []


@pytest.mark.skipif(
    sys.platform != "linux" or shutil.which("bash") is None,
    reason="run.sh process handling is checked on Linux with bash",
)
class TestRunSh:

    @pytest.fixture
    def sandbox(self, tmp_path):
        """A directory with run.sh, a fake venv and stand-ins for python/curl/xdg-open."""
        shutil.copy(os.path.join(_ROOT, "run.sh"), tmp_path / "run.sh")
        (tmp_path / "app.py").write_text("", encoding="utf-8")
        (tmp_path / "start_api.py").write_text("", encoding="utf-8")
        bin_dir = tmp_path / "venv" / "bin"
        bin_dir.mkdir(parents=True)
        (bin_dir / "activate").write_text(f'export PATH="{bin_dir}:$PATH"\n', encoding="utf-8")
        stubs = {
            "python": _FAKE_PYTHON,
            # The UI "answers" at once and no browser is opened by the test.
            "curl": "#!/usr/bin/env bash\nexit 0\n",
            "xdg-open": "#!/usr/bin/env bash\nexit 0\n",
        }
        for name, body in stubs.items():
            path = bin_dir / name
            path.write_text(body, encoding="utf-8")
            path.chmod(0o755)
        return tmp_path

    def _start(self, sandbox, **env):
        environment = dict(os.environ, ABM_TEST_DIR=str(sandbox), **env)
        return subprocess.Popen(
            ["bash", "run.sh"], cwd=sandbox, env=environment,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            start_new_session=True,
        )

    def _pid(self, sandbox, name: str) -> int:
        path = sandbox / f"{name}.pid"
        assert _wait_for(path.exists), f"{name} was never started"
        assert _wait_for(lambda: path.read_text().strip() != "")
        return int(path.read_text().strip())

    def _reap(self, process):
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except OSError:
                pass
            process.wait(timeout=10)

    def test_script_is_valid_bash(self):
        result = subprocess.run(["bash", "-n", os.path.join(_ROOT, "run.sh")], capture_output=True, text=True)
        assert result.returncode == 0, result.stderr

    @pytest.mark.parametrize("app_exit", [0, 3])
    def test_api_server_is_stopped_when_the_ui_exits_on_its_own(self, sandbox, app_exit):
        process = self._start(sandbox, ABM_TEST_APP_EXIT=str(app_exit))
        try:
            api_pid = self._pid(sandbox, "api")
            output, _ = process.communicate(timeout=60)
            assert process.returncode == app_exit, output
            assert _wait_for(lambda: not _alive(api_pid)), "the API server was left running"
            assert "AudiobookMaker stopped." in output
        finally:
            self._reap(process)

    @pytest.mark.parametrize("signum,expected", [(signal.SIGTERM, 143), (signal.SIGINT, 130)])
    def test_api_server_and_ui_are_stopped_on_a_signal(self, sandbox, signum, expected):
        process = self._start(sandbox, ABM_TEST_APP_MODE="run")
        try:
            api_pid = self._pid(sandbox, "api")
            app_pid = self._pid(sandbox, "app")
            # Let the script reach its final `wait`.
            time.sleep(1.0)
            assert _alive(api_pid) and _alive(app_pid)

            os.kill(process.pid, signum)
            output, _ = process.communicate(timeout=60)
            assert process.returncode == expected, output
            assert _wait_for(lambda: not _alive(api_pid)), "the API server was left running"
            assert _wait_for(lambda: not _alive(app_pid)), "the UI was left running"
        finally:
            self._reap(process)


class TestRunBat:

    def test_api_server_window_is_closed_after_the_ui_returns(self):
        with open(os.path.join(_ROOT, "run.bat"), encoding="utf-8") as fh:
            lines = [line.strip() for line in fh.read().splitlines()]
        start = next(i for i, line in enumerate(lines) if line.startswith("start /min") and "start_api.py" in line)
        ui = next(i for i, line in enumerate(lines) if line == 'python "%APP%"')
        kill = next(i for i, line in enumerate(lines) if line.startswith("taskkill"))
        assert start < ui < kill, "the API server must be stopped after app.py returns"
        assert "%API_WINDOW_TITLE%" in lines[start] and "%API_WINDOW_TITLE%" in lines[kill]
        assert "/T" in lines[kill], "child processes of the server window must be stopped too"
