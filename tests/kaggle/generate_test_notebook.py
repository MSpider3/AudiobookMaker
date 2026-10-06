"""
tests/kaggle/generate_test_notebook.py
=======================================
Writes ``AudiobookMaker_Kaggle_Test.ipynb``: a Kaggle notebook whose only job
is to test this branch on real GPUs (T4 x2) and produce a results archive.

Run from the repository root:

    python tests/kaggle/generate_test_notebook.py [--branch NAME]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
_OUTPUT = os.path.join(_ROOT, "AudiobookMaker_Kaggle_Test.ipynb")
_REPO_URL = "https://github.com/MSpider3/AudiobookMaker.git"


def _md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)}


def _code(text: str) -> dict:
    return {
        "cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
        "source": text.strip("\n").splitlines(keepends=True),
    }


def build(branch: str) -> dict:
    cells = [
        _md(f"""
# 🧪 AudiobookMaker — Kaggle GPU Test Notebook

This notebook does not make an audiobook. It **tests the `{branch}` branch** on real GPUs and
packs everything it finds into one archive you can hand back for diagnosis.

**Before running**

1. Right sidebar → **Session options** → **Accelerator: GPU T4 x2**.
2. Same panel → **Internet: On**.
3. *(Optional)* Add-ons → **Secrets** → add `HF_TOKEN` (a HuggingFace read token) and attach it to
   this notebook. Only needed for gated models; everything else runs without it.

Then **Run All**. Each test writes its own result, so one failure never stops the rest.
When it finishes, download **`/kaggle/working/abm_test_results.zip`** (Output panel) and send it back.

| Stage | What it checks | Rough time on T4 x2 |
|---|---|---|
| Setup | clone, dependencies, Rust extension | 10–15 min |
| CPU tests | unit suite, book extraction on every fixture format, mastering | 5 min |
| Qwen3-TTS | cloning, preset voice, designed voice, saved preset, 1-vs-2 GPU speed-up, resume, full book → M4B | 30–45 min |
| Other engines | IndexTTS-2.5, MOSS-TTS, OmniVoice, Fish S2 Pro, Higgs Audio v3 — each in its own environment | 15–30 min each |
| Report | `REPORT.md` + `abm_test_results.zip` | 1 min |
"""),
        _md("## 1 · Settings"),
        _code(f"""
REPO_URL = "{_REPO_URL}"
BRANCH   = "{branch}"

# Engines to test after Qwen. Remove any you do not want; order = run order.
# fish and higgs are 4B-parameter models: they are attempted and reported, and may not fit a T4.
OTHER_PROVIDERS = ["indextts", "omnivoice", "moss", "fish", "higgs"]

RUN_UNIT_TESTS   = True    # pytest suite (mock engine, CPU)
RUN_QWEN_MODES   = True    # preset speaker, designed voice, saved voice preset
RUN_SCALING      = True    # same text on 1 GPU and on 2 GPUs
RUN_RESUME       = True    # kill a run mid-chapter and resume it
RUN_BOOK         = True    # fixture EPUB -> M4B with chapter markers
ASR_SCORING      = True    # transcribe each result with Whisper and score word accuracy

# Optional: your own narrator clip (5-30 s of clean speech) and its exact transcript.
# Leave empty to let the notebook create a reference clip with a Qwen preset voice.
VOICE_FILE       = ""
VOICE_TRANSCRIPT = ""

PROVIDER_TIMEOUT_MIN = 40  # per engine, including model download
"""),
        _md("## 2 · Environment check"),
        _code("""
import os, socket, subprocess, sys, time

try:
    print(subprocess.check_output("nvidia-smi", shell=True).decode())
except Exception:
    print("⚠️ No GPU detected. Set Accelerator to 'GPU T4 x2' in Session options and restart.")

try:
    socket.setdefaulttimeout(5)
    socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect(("github.com", 443))
    print("✅ Internet is on.")
except Exception as exc:
    raise RuntimeError("Internet is off. Turn it on in Session options.") from exc

HF_TOKEN = ""
try:
    from kaggle_secrets import UserSecretsClient
    HF_TOKEN = UserSecretsClient().get_secret("HF_TOKEN") or ""
    print("✅ HF_TOKEN secret found." if HF_TOKEN else "ℹ️ HF_TOKEN secret is empty.")
except Exception:
    print("ℹ️ No HF_TOKEN secret attached — gated models will be reported as skipped/failed.")
if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
    os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN
"""),
        _md("## 3 · Clone the branch"),
        _code("""
import shutil

WORK = "/kaggle/working" if os.path.isdir("/kaggle/working") else os.getcwd()
REPO = os.path.join(WORK, "AudiobookMaker")
RESULTS = os.path.join(WORK, "abm_results")
VENVS = os.path.join(WORK, "abm_venvs")

os.chdir(WORK)
shutil.rmtree(REPO, ignore_errors=True)          # always test the latest push
shutil.rmtree(RESULTS, ignore_errors=True)
subprocess.run(["git", "clone", "--depth", "1", "-b", BRANCH, REPO_URL, REPO], check=True)
os.chdir(REPO)
os.makedirs(RESULTS, exist_ok=True)
print(subprocess.check_output(["git", "log", "-1", "--format=%h %s (%cr)"]).decode())

os.environ["ABM_RESULTS_DIR"] = RESULTS
os.environ["ABM_SKIP_GPU_WARMUP"] = "1"
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
if VOICE_FILE:
    os.environ["ABM_VOICE_FILE"] = VOICE_FILE
    os.environ["ABM_VOICE_TRANSCRIPT"] = VOICE_TRANSCRIPT
"""),
        _md("## 4 · Install dependencies and build the Rust extension"),
        _code("""
def sh(command, check=False, log_name=None, timeout=None, env=None):
    \"\"\"Runs a shell command, streaming its output here and into a log file in the results folder.\"\"\"
    print(f"$ {command}", flush=True)
    log_path = os.path.join(RESULTS, "logs", f"{log_name}.log") if log_name else None
    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
    started = time.time()
    proc = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, env=env or os.environ.copy(), cwd=os.getcwd())
    handle = open(log_path, "a", encoding="utf-8") if log_path else None
    try:
        for line in proc.stdout:
            print(line, end="", flush=True)
            if handle:
                handle.write(line)
            if timeout and time.time() - started > timeout:
                proc.kill()
                message = f"\\n⏱ TIMEOUT after {timeout/60:.0f} min — killed.\\n"
                print(message)
                if handle:
                    handle.write(message)
                break
    finally:
        if handle:
            handle.close()
    code = proc.wait()
    if check and code != 0:
        raise RuntimeError(f"command failed ({code}): {command}")
    return code

sh("apt-get update -qq && apt-get install -y -qq ffmpeg sox libsox-fmt-all > /dev/null", log_name="setup")
sh("curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y -q", log_name="setup")
os.environ["PATH"] += os.pathsep + os.path.expanduser("~/.cargo/bin")
sh(f"{sys.executable} -m pip install -q -r requirements.txt", log_name="setup")
sh(f"{sys.executable} -m pip install -q -r requirements-dev.txt maturin", log_name="setup")
sh(f"cd audiobook_rust && {sys.executable} -m pip install -q . ", log_name="setup")
if ASR_SCORING:
    sh(f"{sys.executable} -m pip install -q faster-whisper", log_name="setup")
sh(f"{sys.executable} -c \\"import audiobook_rust; print('Rust extension:', [n for n in dir(audiobook_rust) if not n.startswith('_')])\\"")
"""),
        _code("""
SUITE = "tests/kaggle/abm_gpu_suite.py"

def suite(arguments, python=sys.executable, timeout_min=None, log_name=None, env=None):
    \"\"\"Runs one harness command. It records its own pass/fail; this never raises.\"\"\"
    name = log_name or arguments.split()[0]
    return sh(f"{python} {SUITE} {arguments}", log_name=name,
              timeout=timeout_min * 60 if timeout_min else None, env=env)

ASR_FLAG = "" if ASR_SCORING else "--no-asr"
suite("env")
"""),
        _md("## 5 · CPU tests — unit suite, extraction of every fixture format, mastering"),
        _code("""
if RUN_UNIT_TESTS:
    suite("unit", timeout_min=30)
suite("extraction", timeout_min=15)
suite("mastering", timeout_min=10)
"""),
        _md("""
## 6 · Qwen3-TTS (default engine)

First a real-speech narrator clip is made with a Qwen preset voice (skipped if you supplied `VOICE_FILE`).
Every cloning test after that — for every engine — uses the same clip, so results are comparable.
"""),
        _code("""
if not VOICE_FILE:
    suite("make-voice", timeout_min=20)

# Voice cloning through the full pipeline on all GPUs, with ASR chunk verification switched on.
suite(f"provider --name qwen --tag qwen_clone --verify asr {ASR_FLAG}", timeout_min=PROVIDER_TIMEOUT_MIN, log_name="qwen_clone")
"""),
        _code("""
if RUN_QWEN_MODES:
    # Preset speaker (CustomVoice) with a style instruction — no reference clip.
    suite("provider --name qwen --tag qwen_custom_voice --no-voice "
          "--model Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice --timbre ryan "
          "--instruct \\"Calm, warm audiobook narration at an unhurried pace.\\" " + ASR_FLAG,
          timeout_min=PROVIDER_TIMEOUT_MIN, log_name="qwen_custom_voice")

    # Designed voice (VoiceDesign): designed once, then cloned for every chunk on both GPUs.
    suite("provider --name qwen --tag qwen_voice_design --no-voice "
          "--model Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign "
          "--instruct \\"A middle-aged male narrator with a deep, steady, reassuring voice.\\" " + ASR_FLAG,
          timeout_min=PROVIDER_TIMEOUT_MIN, log_name="qwen_voice_design")

    # The smaller checkpoint, for the speed/VRAM comparison.
    suite(f"provider --name qwen --tag qwen_clone_0.6b --model Qwen/Qwen3-TTS-12Hz-0.6B-Base {ASR_FLAG}",
          timeout_min=PROVIDER_TIMEOUT_MIN, log_name="qwen_clone_0.6b")
"""),
        _code("""
if RUN_SCALING:
    suite("scaling --name qwen", timeout_min=PROVIDER_TIMEOUT_MIN, log_name="scaling_qwen")
if RUN_RESUME:
    suite("resume --name qwen", timeout_min=PROVIDER_TIMEOUT_MIN, log_name="resume_qwen")
if RUN_BOOK:
    suite("book --name qwen", timeout_min=PROVIDER_TIMEOUT_MIN, log_name="book_qwen")
"""),
        _md("""
## 7 · Other engines

Each engine gets its **own virtual environment** layered over the base one, so an engine that needs a
different `transformers` or `torch` build cannot break Qwen or the next engine. An engine that fails to
install or does not fit a T4 is recorded as failed with its error — that is a result, not a notebook error.
"""),
        _code("""
PROVIDER_SETUP = __PROVIDER_SETUP__

def provider_python(name):
    \"\"\"Creates (once) the engine's virtualenv and installs its requirements. Returns its python.\"\"\"
    venv = os.path.join(VENVS, name)
    python = os.path.join(venv, "bin", "python")
    setup = PROVIDER_SETUP.get(name, {})
    if not os.path.exists(python):
        sh(f"{sys.executable} -m venv --system-site-packages {venv}", log_name=f"install_{name}")
    for command in setup.get("pre", []):
        sh(command.format(python=python, venv=venv, work=WORK, repo=REPO), log_name=f"install_{name}", timeout=30 * 60)
    requirements = os.path.join("requirements", f"tts-{name}.txt")
    if os.path.exists(requirements):
        sh(f"{python} -m pip install -q -r {requirements}", log_name=f"install_{name}", timeout=30 * 60)
    for command in setup.get("post", []):
        sh(command.format(python=python, venv=venv, work=WORK, repo=REPO), log_name=f"install_{name}", timeout=30 * 60)
    return python

def provider_env(name):
    env = os.environ.copy()
    for key, value in PROVIDER_SETUP.get(name, {}).get("env", {}).items():
        env[key] = value.format(work=WORK, repo=REPO, venv=os.path.join(VENVS, name))
    return env

for name in OTHER_PROVIDERS:
    print("\\n" + "=" * 100 + f"\\n  {name}\\n" + "=" * 100)
    try:
        python = provider_python(name)
    except Exception as exc:
        print(f"❌ could not prepare an environment for {name}: {exc}")
        continue
    extra = PROVIDER_SETUP.get(name, {}).get("args", "")
    suite(f"provider --name {name} {extra} {ASR_FLAG}", python=python,
          timeout_min=PROVIDER_TIMEOUT_MIN, log_name=f"provider_{name}", env=provider_env(name))
    # Free disk for the next engine's weights.
    sh("rm -rf ~/.cache/huggingface/hub/models--* 2>/dev/null; df -h /kaggle/working ~ | tail -2")
"""),
        _md("## 8 · Report"),
        _code("""
suite("report")

from IPython.display import Markdown, display
with open(os.path.join(RESULTS, "REPORT.md"), encoding="utf-8") as fh:
    display(Markdown(fh.read()))
print("\\n📦 Download and send back:", os.path.join(WORK, "abm_test_results.zip"))
"""),
        _md("""
### Listening to the results

The archive contains a `samples/` folder with one audio file per test. Run the cell below to play them here.
"""),
        _code("""
from IPython.display import Audio, display
samples = os.path.join(RESULTS, "samples")
for name in sorted(os.listdir(samples)) if os.path.isdir(samples) else []:
    print(name)
    display(Audio(os.path.join(samples, name)))
"""),
    ]
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "kaggle": {"accelerator": "nvidiaTeslaT4", "isGpuEnabled": True, "isInternetEnabled": True},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


# Per-engine setup beyond `pip install -r requirements/tts-<name>.txt`.
#   pre / post : shell commands run before / after the requirements install.
#                {python} {venv} {work} {repo} are substituted.
#   env        : extra environment variables for the test run.
#   args       : extra arguments for `abm_gpu_suite.py provider`.
PROVIDER_SETUP: dict[str, dict] = {
    "indextts": {
        # Upstream pins torch/numpy/keras and Python <3.12 in its metadata, so
        # the package itself goes in without dependencies; the requirements
        # file carries the ones it really needs (transformers 4.52.1 among them,
        # which is why this engine cannot share Qwen's environment).
        "post": [
            "{python} -m pip install -q --no-deps --ignore-requires-python "
            "\"indextts @ git+https://github.com/index-tts/index-tts.git@d9e41aac89fd00b3d71497fddb287b7f24613712\"",
        ],
        "env": {"USE_MODELSCOPE": "false"},
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    default_branch = subprocess.run(
        ["git", "-C", _ROOT, "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, text=True
    ).stdout.strip() or "main"
    parser.add_argument("--branch", default=default_branch)
    args = parser.parse_args()

    notebook = build(args.branch)
    setup_literal = json.dumps(PROVIDER_SETUP, indent=4)
    for cell in notebook["cells"]:
        cell["source"] = [line.replace("__PROVIDER_SETUP__", setup_literal) for line in cell["source"]]
    with open(_OUTPUT, "w", encoding="utf-8") as fh:
        json.dump(notebook, fh, indent=1, ensure_ascii=False)
        fh.write("\n")
    print(f"Wrote {_OUTPUT} for branch {args.branch}")


if __name__ == "__main__":
    main()
