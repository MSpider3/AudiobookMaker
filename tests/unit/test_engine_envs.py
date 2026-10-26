"""
test_engine_envs.py
===================
Engines installed side by side, each in its own environment
(``engine_envs.py``), and run through a worker process
(``engine_worker.py``, ``tts_providers/remote_provider.py``).

The mock engine stands in for a real one: a real environment is built the way
Kaggle needs it (``venv --without-pip`` plus the main environment's packages
behind the venv's own), and a whole chapter goes through the GPU pool, the
chapter pipeline and the worker.
"""

from __future__ import annotations

import io
import os
import subprocess
import sys
import tempfile

import pytest
import soundfile as sf

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from audiobook_factory import engine_envs
from audiobook_factory.chapter_pipeline import run_chapter_pipeline
from audiobook_factory.gpu_pool import ProviderPool
from audiobook_factory.pipeline import AudiobookConfig, CancelToken
from audiobook_factory.tts_providers import get_tts_provider
from audiobook_factory.tts_providers.remote_provider import RemoteProvider


def _voice() -> bytes:
    with open(os.path.join(_ROOT, "tests", "fixtures", "audio", "synthetic_voice_reference.wav"), "rb") as fh:
        return fh.read()


@pytest.fixture
def remote():
    providers: list[RemoteProvider] = []

    def _make(python: str = sys.executable, device: str = "cpu", config: AudiobookConfig | None = None):
        provider = RemoteProvider("mock", python, {}, config or AudiobookConfig(sample_rate=24000), device=device)
        providers.append(provider)
        return provider

    yield _make
    for provider in providers:
        provider.cleanup()


class TestWorker:

    def test_batch_and_single_synthesis_come_back_as_audio(self, remote, tmp_path):
        provider = remote()
        provider.ensure_ready()
        assert provider.is_ready and provider.get_name()
        results = provider.synthesize_batch(["One sentence.", "Another one here."], _voice())
        assert len(results) == 2
        for audio, duration in results:
            assert sf.info(io.BytesIO(audio)).duration == pytest.approx(duration, abs=0.05)
        path, duration = provider.synthesize("Written to a file.", _voice(), str(tmp_path / "out.wav"))
        assert os.path.getsize(path) > 1000 and duration > 0

    def test_errors_come_back_as_runtime_errors_and_the_worker_lives_on(self, remote):
        provider = remote()
        with pytest.raises(RuntimeError, match="TypeError"):
            provider._call("synthesize")    # missing arguments: the engine raises
        assert provider.synthesize_batch(["Still works."], _voice())

    def test_config_changes_reach_the_worker(self, remote):
        provider = remote(config=AudiobookConfig(sample_rate=24000))
        first = provider.synthesize_batch(["Hello there."], _voice())[0][0]
        provider.config = AudiobookConfig(sample_rate=16000)
        second = provider.synthesize_batch(["Hello there."], _voice())[0][0]
        assert sf.info(io.BytesIO(first)).samplerate != sf.info(io.BytesIO(second)).samplerate or first != second

    def test_cleanup_ends_the_worker_and_a_new_call_restarts_it(self, remote):
        provider = remote()
        provider.ensure_ready()
        process = provider._process
        provider.cleanup()
        assert process.poll() is not None and not provider.is_ready
        assert provider.synthesize_batch(["Back again."], _voice())

    def test_only_provider_methods_can_be_called(self, remote):
        provider = remote()
        with pytest.raises(RuntimeError, match="not allowed"):
            provider._call("__init__")

    def test_a_dead_worker_is_restarted_and_a_death_mid_call_reported(self, remote, monkeypatch):
        provider = remote(python=sys.executable)
        provider.ensure_ready()
        dead = provider._process
        dead.kill()
        dead.wait()
        assert provider.synthesize_batch(["Restarted."], _voice())   # next call starts a new worker
        provider._process.kill()
        provider._process.wait()
        monkeypatch.setattr(provider, "_start", lambda: dead)        # dies while answering
        with pytest.raises(RuntimeError, match="stopped"):
            provider.synthesize_batch(["Text."], _voice())


class TestOwnEnvironment:

    def test_engine_runs_in_an_environment_of_its_own(self, monkeypatch, tmp_path):
        """A real venv like on Kaggle; a chapter runs through the pool and two workers."""
        monkeypatch.setattr(engine_envs, "ENVS_DIR", str(tmp_path / "engines"))
        monkeypatch.setitem(engine_envs.SETUPS, "mock", engine_envs.EngineSetup())
        manifest = engine_envs.install("mock", log=lambda line: None)
        python = engine_envs.env_python("mock")
        assert python and python.startswith(str(tmp_path)) and engine_envs.is_installed("mock")
        # The venv sees the main environment's packages and has no copy of its own.
        probe = subprocess.run([python, "-c", "import sys, numpy, soundfile; print(sys.prefix)"],
                               capture_output=True, text=True)
        assert probe.returncode == 0, probe.stderr
        assert probe.stdout.strip().startswith(str(tmp_path))
        assert manifest["installed"]

        config = AudiobookConfig(output_dir=str(tmp_path), sample_rate=24000, max_len=100, pause=0.1, batch_size=2)
        pool = ProviderPool(lambda dev: RemoteProvider("mock", python, {}, config, device=dev), ["cuda:0", "cuda:1"], "mock")
        try:
            durations = run_chapter_pipeline(
                sentences=[f"Sentence {i} of a chapter from another environment." for i in range(5)],
                voice_ref=_voice(), out_wav_path=str(tmp_path / "ch.wav"), out_dir=str(tmp_path / "tmp"),
                chapter_index=1, config=config, pool=pool, cancel_token=CancelToken(), log_callback=lambda m: None,
            )
            assert len(durations) == 5 and sf.info(str(tmp_path / "ch.wav")).duration > 2
        finally:
            pool.cleanup()

    def test_factory_uses_the_worker_only_for_engines_in_their_own_environment(self, monkeypatch, tmp_path):
        monkeypatch.setattr(engine_envs, "ENVS_DIR", str(tmp_path / "engines"))
        config = AudiobookConfig()
        assert not isinstance(get_tts_provider("omnivoice", config, device="cpu"), RemoteProvider)
        folder = tmp_path / "engines" / "omnivoice"
        folder.mkdir(parents=True)
        (folder / "abm-engine.json").write_text(
            '{"installed": "now", "python": "%s", "env": {"X_TEST_VAR": "1"}}' % sys.executable)
        provider = get_tts_provider("omnivoice", config, device="cuda:1")
        assert isinstance(provider, RemoteProvider) and provider.device == "cuda:1"
        assert provider.info().name == "omnivoice"

    def test_main_environment_engines_get_their_clone_location(self, monkeypatch, tmp_path):
        monkeypatch.setattr(engine_envs, "ENVS_DIR", str(tmp_path / "engines"))
        monkeypatch.delenv("ABM_BREEZE_REPO", raising=False)
        folder = tmp_path / "engines" / "breeze"
        folder.mkdir(parents=True)
        (folder / "abm-engine.json").write_text('{"installed": "now", "python": null, "env": {"ABM_BREEZE_REPO": "/x/breeze"}}')
        provider = get_tts_provider("breeze", AudiobookConfig(), device="cpu")
        assert not isinstance(provider, RemoteProvider)
        assert os.environ["ABM_BREEZE_REPO"] == "/x/breeze"
        monkeypatch.delenv("ABM_BREEZE_REPO")

    def test_unknown_engine_is_refused(self):
        with pytest.raises(ValueError):
            engine_envs.install("no-such-engine", log=lambda line: None)


class TestEngineTrial:

    def test_each_engine_reads_the_text_and_a_failure_does_not_stop_the_rest(self, tmp_path, monkeypatch):
        from audiobook_factory import engine_trial

        monkeypatch.delenv("ABM_BREEZE_REPO", raising=False)
        voice = tmp_path / "voice.wav"
        voice.write_bytes(_voice())
        results = engine_trial.try_engines(
            ["mock", "breeze", "mock"], str(voice), "Hello.", "First sentence here. Second one follows.\n\nA new paragraph.",
            str(tmp_path / "out"), device="cpu", log=lambda line: None)
        assert [r.ok for r in results] == [True, False, True]
        assert "ABM_BREEZE_REPO" in results[1].error
        assert sf.info(results[0].wav_path).duration > 1 and results[0].speed_x > 0
        table = engine_trial.table(results)
        assert table.count("| mock |") == 2 and "❌" in table
