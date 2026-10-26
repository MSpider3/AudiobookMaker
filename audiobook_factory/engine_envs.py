"""
audiobook_factory/engine_envs.py
================================
One environment per TTS engine, all installed side by side.

The engines pin different ``transformers`` versions (Ming 4.46, IndexTTS 4.52,
Qwen 4.57, MOSS 5.0, OmniVoice and Higgs 5.3+), and one Python process can
load only one. So each engine that needs other versions gets its own
environment under ``ENVS_DIR/<engine>``: a venv whose own packages come first
and which falls back to the main environment's packages (torch, numpy, the
app's own dependencies) through a ``.pth`` file. Only what the engine pins
differently is installed into it, about 0.1-1 GB per engine.

The app runs such an engine in a worker process with that environment's
Python (``engine_worker.py``, ``tts_providers/remote_provider.py``).
Qwen3-TTS and Breeze run in the main environment, as before.

Usage::

    python -m audiobook_factory.engine_envs list
    python -m audiobook_factory.engine_envs install moss higgs
    python -m audiobook_factory.engine_envs remove moss
"""
from __future__ import annotations

import json
import logging
import os
import shutil
import site
import subprocess
import sys
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

__all__ = ["ENVS_DIR", "SETUPS", "EngineSetup", "env_python", "engine_env_vars", "is_installed",
           "runs_in_own_env", "install", "remove", "install_command"]

_ROOT: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ENVS_DIR: str = os.environ.get("ABM_ENGINE_ENVS") or os.path.join(
    os.path.expanduser("~"), ".cache", "audiobookmaker", "engines")
_MANIFEST: str = "abm-engine.json"
_BASE_PTH: str = "_abm_base_packages.pth"


@dataclass(frozen=True)
class EngineSetup:
    """How one engine is installed.

    Attributes
    ----------
    requirements : str
        Requirements file relative to the repository root, or "".
    extra_pip : tuple[tuple[str, ...], ...]
        Further ``pip install`` argument lists, run in order.
    clone : tuple[str, str, str] | None
        ``(git URL, commit, environment variable)`` for engines that run from
        a clone of their repository.
    own_env : bool
        False for engines that run in the main environment.
    env : dict[str, str]
        Environment variables the engine needs.
    """

    requirements: str = ""
    extra_pip: tuple[tuple[str, ...], ...] = ()
    clone: tuple[str, str, str] | None = None
    own_env: bool = True
    env: dict = field(default_factory=dict)


SETUPS: dict[str, EngineSetup] = {
    "qwen": EngineSetup(own_env=False),
    "breeze": EngineSetup(own_env=False, clone=(
        "https://github.com/breezeblue-ai/breeze-tts.git", "6eba901", "ABM_BREEZE_REPO")),
    "indextts": EngineSetup(
        requirements="requirements/tts-indextts.txt",
        extra_pip=(("--no-deps", "--ignore-requires-python", "descript-audiotools",
                    "indextts @ git+https://github.com/index-tts/index-tts.git@d9e41aac89fd00b3d71497fddb287b7f24613712"),),
        env={"USE_MODELSCOPE": "false"}),
    "moss": EngineSetup(requirements="requirements/tts-moss.txt"),
    "omnivoice": EngineSetup(requirements="requirements/tts-omnivoice.txt"),
    "fish": EngineSetup(
        requirements="requirements/tts-fish.txt",
        extra_pip=(("--no-deps", "descript-audiotools", "descript-audio-codec",
                    "fish-speech @ git+https://github.com/fishaudio/fish-speech.git@214da3cd841bda85da2496b96cd3c4d7edb1337e"),)),
    "higgs": EngineSetup(requirements="requirements/tts-higgs.txt"),
    "ming": EngineSetup(requirements="requirements/tts-ming.txt", clone=(
        "https://github.com/inclusionAI/Ming-3.0-tts.git", "028c1ea", "ABM_MING_REPO")),
    "f5tts": EngineSetup(extra_pip=(("f5-tts",),)),
}


def _engine_dir(engine: str) -> str:
    return os.path.join(ENVS_DIR, engine)


def _manifest(engine: str) -> dict:
    try:
        with open(os.path.join(_engine_dir(engine), _MANIFEST), encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return {}


def is_installed(engine: str) -> bool:
    """True once ``install(engine)`` has finished successfully."""
    return bool(_manifest(engine).get("installed"))


def runs_in_own_env(engine: str) -> bool:
    """True when *engine* is installed in an environment of its own."""
    return bool(_manifest(engine).get("python"))


def env_python(engine: str) -> str | None:
    """The interpreter of *engine*'s own environment, or None."""
    python = _manifest(engine).get("python")
    return python if python and os.path.exists(python) else None


def engine_env_vars(engine: str) -> dict[str, str]:
    """Environment variables *engine* needs (clone locations and the like)."""
    return dict(_manifest(engine).get("env") or {})


def install_command(engine: str) -> str:
    """The command that installs *engine* side by side with the others."""
    return f"python -m audiobook_factory.engine_envs install {engine}"


def _base_package_dirs() -> list[str]:
    """Package folders of the main environment, for the engine envs to fall back on."""
    dirs: list[str] = []
    extra = [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
    candidates = list(sys.path) + extra + [site.getusersitepackages()]
    for path in candidates:
        if not path or not os.path.isdir(path) or os.path.abspath(path) == _ROOT:
            continue
        if path.endswith(("site-packages", "dist-packages")) or path in extra:
            if path not in dirs:
                dirs.append(path)
    return dirs


def _venv_python(venv: str) -> str:
    return os.path.join(venv, "Scripts", "python.exe") if os.name == "nt" else os.path.join(venv, "bin", "python")


def _run(command: list[str], log: callable, env: dict | None = None) -> None:
    log("$ " + " ".join(command))
    result = subprocess.run(command, cwd=_ROOT, env=env, text=True,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    for line in (result.stdout or "").splitlines()[-40:]:
        log("  " + line)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed ({result.returncode}): {' '.join(command)}")


def install(engine: str, log: callable = print) -> dict:
    """Installs *engine* into its own environment (or clones what it needs).

    Safe to run again: an existing environment is reused and its packages
    updated. Returns the manifest written for the engine.

    Raises:
        ValueError: If *engine* is unknown.
        RuntimeError: If a step fails; nothing is marked installed then.
    """
    if engine not in SETUPS:
        raise ValueError(f"Unknown engine {engine!r}. Known: {', '.join(SETUPS)}")
    setup = SETUPS[engine]
    folder = _engine_dir(engine)
    os.makedirs(folder, exist_ok=True)
    manifest: dict = {"engine": engine, "python": None, "env": dict(setup.env)}

    if setup.clone:
        url, commit, variable = setup.clone
        clone = os.path.join(folder, "upstream")
        if not os.path.isdir(os.path.join(clone, ".git")):
            _run(["git", "clone", "-q", url, clone], log)
        _run(["git", "-C", clone, "checkout", "-q", commit], log)
        manifest["env"][variable] = clone

    if setup.own_env:
        venv = os.path.join(folder, "venv")
        python = _venv_python(venv)
        if not os.path.exists(python):
            # --without-pip: ensurepip fails on Kaggle; pip comes from the base packages.
            _run([sys.executable, "-m", "venv", "--without-pip", venv], log)
        purelib = subprocess.run([python, "-c", "import sysconfig; print(sysconfig.get_paths()['purelib'])"],
                                 capture_output=True, text=True, check=True).stdout.strip()
        os.makedirs(purelib, exist_ok=True)
        with open(os.path.join(purelib, _BASE_PTH), "w", encoding="utf-8") as fh:
            fh.write("\n".join(_base_package_dirs()) + "\n")
        env = dict(os.environ, **manifest["env"])
        env.pop("PYTHONPATH", None)   # the venv's own packages must come first
        if setup.requirements:
            _run([python, "-m", "pip", "install", "-q", "-r", setup.requirements], log, env)
        for args in setup.extra_pip:
            _run([python, "-m", "pip", "install", "-q", *args], log, env)
        manifest["python"] = python
        manifest["versions"] = subprocess.run(
            [python, "-c", "import torch, transformers; print(torch.__version__, transformers.__version__)"],
            capture_output=True, text=True, env=env).stdout.strip()
    elif setup.requirements:
        _run([sys.executable, "-m", "pip", "install", "-q", "-r", setup.requirements], log)

    manifest["installed"] = time.strftime("%Y-%m-%d %H:%M:%S")
    with open(os.path.join(folder, _MANIFEST), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1)
    log(f"✅ {engine} installed" + (f" ({manifest.get('versions')})" if manifest.get("versions") else ""))
    return manifest


def remove(engine: str) -> bool:
    """Deletes *engine*'s environment and clone. Returns False if there was none."""
    folder = _engine_dir(engine)
    if not os.path.isdir(folder):
        return False
    shutil.rmtree(folder)
    return True


def _main(argv: list[str]) -> int:
    if not argv or argv[0] not in ("list", "install", "remove"):
        print(__doc__)
        return 2
    command, engines = argv[0], argv[1:]
    if command == "list":
        for engine in SETUPS:
            manifest = _manifest(engine)
            where = "own environment" if manifest.get("python") else "main environment"
            state = f"installed {manifest['installed']} ({where})" if manifest.get("installed") else "not installed"
            print(f"{engine:10} {state}")
        return 0
    if not engines:
        print(f"Name the engine(s): {', '.join(SETUPS)}")
        return 2
    failed = 0
    for engine in engines:
        try:
            if command == "install":
                install(engine)
            else:
                print(f"{engine}: {'removed' if remove(engine) else 'was not installed'}")
        except Exception as exc:
            failed += 1
            print(f"❌ {engine}: {exc}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
