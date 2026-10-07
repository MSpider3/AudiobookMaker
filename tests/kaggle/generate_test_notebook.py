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

Nothing needs uploading: the narrator voice and the test books are in the repository under
`tests/kaggle/assets/` and are read from the clone.

Then **Run All**. Each test writes its own result, so one failure never stops the rest.
When it finishes, download **`/kaggle/working/abm_test_results.zip`** (Output panel) and send it back.

| Stage | What it checks | Rough time on T4 x2 |
|---|---|---|
| Setup | clone, dependencies, Rust extension | 10–15 min |
| CPU tests | unit suite, book extraction on every fixture format, mastering | 5 min |
| Qwen3-TTS | cloning, preset voice, designed voice, saved preset, resume, short book → M4B, CLI | 30–45 min |
| Two-GPU speed-up | four chapters of the long English book on 1 GPU, then on 2 | 15–25 min |
| Whole book | the 20-page English book, every chapter → one M4B; time, memory, accuracy over the whole book | 20–35 min |
| Languages | one passage each in French, Russian, Chinese, Japanese, Korean (Qwen) and Hindi (engines that support it) | 15–20 min |
| API server and web UI | started as real programs; a chapter generated through each | 10 min |
| Other engines | IndexTTS-2.5, OmniVoice, MOSS-TTS, Fish S2 Pro, Higgs Audio v3 — each in its own environment; cloning, Hindi where supported, 1-vs-2 GPU | 25–60 min each |
| Voice similarity | how close every cloned voice is to the narrator clip | 3 min |
| Report | `REPORT.md` + `abm_test_results.zip` | 1 min |

The full run takes roughly 5–6 hours. Every block has a switch in the settings cell; turn off what you
do not need.
"""),
        _md("## 1 · Settings"),
        _code(f"""
REPO_URL = "{_REPO_URL}"
BRANCH   = "{branch}"

# Engines to test after Qwen. Remove any you do not want; order = run order.
# None of these has run on a GPU before this notebook. fish and higgs are 4B-parameter
# models that should just fit a T4 on paper; fish is expected to be slower than real time.
OTHER_PROVIDERS = ["indextts", "omnivoice", "moss", "fish", "higgs"]

RUN_UNIT_TESTS   = True    # pytest suite (mock engine, CPU)
RUN_QWEN_MODES   = True    # preset speaker, designed voice, saved voice preset
RUN_SCALING      = True    # four chapters of the long English book on 1 GPU and on 2 GPUs (Qwen)
RUN_RESUME       = True    # kill a run mid-chapter and resume it
RUN_BOOK         = True    # fixture EPUB and MOBI -> M4B with chapter markers (pipeline and cli.py)
RUN_LONG_BOOK    = True    # the whole 20-page English book -> one M4B (Qwen)
RUN_LANGUAGES    = True    # one passage per language, see LANGUAGES below
RUN_API_AND_UI   = True    # start the API server and the web UI and generate through each (Qwen)
RUN_SIMILARITY   = True    # score how close each cloned voice is to the narrator clip
ASR_SCORING      = True    # transcribe each result with Whisper and score word accuracy

# 20-page test books exist in en, fr, ru, hi, zh, ja, ko (tests/kaggle/assets/books/long_book_<code>.epub).
LANGUAGES             = ["fr", "ru", "zh", "ja", "ko", "hi"]   # tried with Qwen; it skips what it does not support (Hindi)
OTHER_ENGINE_LANGUAGES = ["hi"]   # tried with each other engine that supports them; add more codes to widen the test
LANGUAGE_SECONDS      = 60        # length of the passage spoken per language
LONG_BOOK_CHAPTERS    = 0         # chapters of the long book to narrate; 0 = all ten
SCALING_OTHER_ENGINES = True      # also measure 1-vs-2 GPU for every other engine (adds 5-25 min each)
SCALING_OTHER_CHUNKS  = 16        # chunks for that measurement, in batches of 4

# Test inputs, read from the cloned repository (paths are relative to the repo root).
# To test another voice, point VOICE_FILE at any 5-30 s clip of clean speech (an absolute
# path such as /kaggle/input/... also works). VOICE_TRANSCRIPT is what is said in the clip:
# either the words themselves or the path of a .txt file holding them. Leave it empty to
# use the .txt file next to the clip. It must match the clip word for word — engines that
# clone from a transcript cut off, ramble or go silent when it does not.
ASSETS_DIR       = "tests/kaggle/assets"
VOICE_FILE       = "tests/kaggle/assets/voice/LOTM_narrator_voice.wav"
VOICE_TRANSCRIPT = "tests/kaggle/assets/voice/LOTM_narrator_voice.txt"

PROVIDER_TIMEOUT_MIN = 60  # per test, including model download
LONG_RUN_TIMEOUT_MIN = 120 # whole-book and all-language runs
"""),
        _md("## 2 · Environment check"),
        _code("""
import os, signal, socket, subprocess, sys, threading, time

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
# Per-engine environments are several GB each; /kaggle/working is capped at ~20 GB and is
# saved as notebook output, so they live on the scratch disk instead.
VENVS = "/tmp/abm_venvs"

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
# Progress bars redraw in a terminal but print a new line each time here
# (one engine wrote 5,900 of them), burying the lines that matter.
os.environ["TQDM_DISABLE"] = "1"
os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"

def in_repo(path):
    \"\"\"Resolves a settings path against the cloned repository.\"\"\"
    return path if os.path.isabs(path) else os.path.join(REPO, path)

ASSETS_PATH = in_repo(ASSETS_DIR)
VOICE_PATH = in_repo(VOICE_FILE) if VOICE_FILE else ""
os.environ["ABM_ASSETS_DIR"] = ASSETS_PATH
HAVE_VOICE = bool(VOICE_PATH) and os.path.exists(VOICE_PATH)
if HAVE_VOICE:
    words = VOICE_TRANSCRIPT.strip()
    if words and "\\n" not in words and os.path.isfile(in_repo(words)):
        with open(in_repo(words), encoding="utf-8-sig") as fh:   # a path to the transcript, not the words
            words = fh.read().strip()
    sidecar = os.path.splitext(VOICE_PATH)[0] + ".txt"
    if not words and os.path.exists(sidecar):
        with open(sidecar, encoding="utf-8-sig") as fh:
            words = fh.read().strip()
    os.environ["ABM_VOICE_FILE"] = VOICE_PATH
    os.environ["ABM_VOICE_TRANSCRIPT"] = words
    print(f"Narrator voice : {VOICE_PATH}")
    print(f"Transcript     : {len(words.split())} words — {words[:110]}{'…' if len(words) > 110 else ''}" if words
          else "Transcript     : none — engines that need one will transcribe the clip themselves")
    try:
        import soundfile
        clip = soundfile.info(VOICE_PATH)
        clip_seconds = clip.frames / clip.samplerate
    except Exception:
        clip_seconds = 0
    if words and words.isascii() and clip_seconds >= 1 and not 0.5 <= len(words.split()) / clip_seconds <= 6:
        raise RuntimeError(
            f"VOICE_TRANSCRIPT has {len(words.split())} word(s) for a {clip_seconds:.0f}-second clip, so it cannot be "
            "what is said in it. Give the exact words, or the path of a .txt file that holds them.")
else:
    print(f"⚠️ Narrator voice not found at {VOICE_PATH or '(not set)'} — a clip will be made with a Qwen preset voice.")
books = os.path.join(ASSETS_PATH, "books")
print("Test books     :", ", ".join(sorted(os.listdir(books))) if os.path.isdir(books) else f"missing ({books})")
"""),
        _md("## 4 · Install dependencies and build the Rust extension"),
        _code("""
def sh(command, check=False, log_name=None, timeout=None, env=None):
    \"\"\"Runs a shell command, streaming its output here and into a log file in the results folder.\"\"\"
    print(f"$ {command}", flush=True)
    log_path = os.path.join(RESULTS, "logs", f"{log_name}.log") if log_name else None
    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
    proc = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, env=env or os.environ.copy(), cwd=os.getcwd(),
                            start_new_session=True)
    timed_out = threading.Event()

    def _kill():
        # Kill the whole process group: a hung model download or a silent
        # deadlock prints nothing, so this cannot rely on output arriving.
        timed_out.set()
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    timer = threading.Timer(timeout, _kill) if timeout else None
    if timer:
        timer.start()
    handle = open(log_path, "a", encoding="utf-8") if log_path else None
    try:
        for line in proc.stdout:
            print(line, end="", flush=True)
            if handle:
                handle.write(line)
        code = proc.wait()
        if timed_out.is_set():
            message = f"\\n⏱ TIMEOUT after {timeout/60:.0f} min — killed: {command}\\n"
            print(message)
            if handle:
                handle.write(message)
    finally:
        if timer:
            timer.cancel()
        if handle:
            handle.close()
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

Every cloning test — for every engine — uses the narrator clip from the settings cell, so results are
comparable. (If that clip is missing, one is made first with a Qwen preset voice.)
"""),
        _code("""
if not HAVE_VOICE:
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

    # Saved voice preset: build it from the reference clip, then narrate with no clip at all.
    suite(f"preset --name qwen --tag qwen_voice_preset {ASR_FLAG}",
          timeout_min=PROVIDER_TIMEOUT_MIN, log_name="qwen_voice_preset")

    # The smaller checkpoint, for the speed/VRAM comparison.
    suite(f"provider --name qwen --tag qwen_clone_0.6b --model Qwen/Qwen3-TTS-12Hz-0.6B-Base {ASR_FLAG}",
          timeout_min=PROVIDER_TIMEOUT_MIN, log_name="qwen_clone_0.6b")
"""),
        _code("""
if RUN_SCALING:
    # Several batches per GPU: with a dozen chunks one GPU takes them all in one batch and a second GPU cannot help.
    suite("scaling --name qwen --book-chapters 4", timeout_min=PROVIDER_TIMEOUT_MIN, log_name="scaling_qwen")
if RUN_RESUME:
    suite("resume --name qwen", timeout_min=PROVIDER_TIMEOUT_MIN, log_name="resume_qwen")
if RUN_BOOK:
    suite("book --name qwen", timeout_min=PROVIDER_TIMEOUT_MIN, log_name="book_qwen")
    # The headless CLI on the MOBI fixture: dry run, real run to M4B, and a re-run that must skip.
    suite("cli --name qwen", timeout_min=PROVIDER_TIMEOUT_MIN, log_name="cli_qwen")
if RUN_LONG_BOOK:
    # A complete book of realistic length: all ten chapters, automatic batch size, both GPUs.
    suite(f"longbook --name qwen --max-chapters {LONG_BOOK_CHAPTERS} {ASR_FLAG}",
          timeout_min=LONG_RUN_TIMEOUT_MIN, log_name="longbook_qwen")
if RUN_LANGUAGES and LANGUAGES:
    suite(f"languages --name qwen --langs {','.join(LANGUAGES)} --seconds {LANGUAGE_SECONDS} {ASR_FLAG}",
          timeout_min=LONG_RUN_TIMEOUT_MIN, log_name="languages_qwen")
if RUN_API_AND_UI:
    # The server programs themselves, not their functions: start, use over HTTP, stop.
    suite("api --name qwen", timeout_min=PROVIDER_TIMEOUT_MIN, log_name="api_qwen")
    suite("ui --name qwen", timeout_min=PROVIDER_TIMEOUT_MIN, log_name="ui_qwen")

def free_model_cache():
    \"\"\"Deletes downloaded TTS weights (keeps Whisper, which every engine's scoring reuses).\"\"\"
    sh("find ~/.cache/huggingface/hub -maxdepth 1 -name 'models--*' ! -iname '*whisper*' -exec rm -rf {} + 2>/dev/null; "
       "df -h ~ /kaggle/working 2>/dev/null | tail -2")

free_model_cache()   # the Qwen checkpoints are not needed by the other engines
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
        # virtualenv rather than the stdlib venv module: Debian-based images often ship without ensurepip.
        sh(f"{sys.executable} -m pip install -q virtualenv && "
           f"{sys.executable} -m virtualenv -q --system-site-packages {venv}", log_name=f"install_{name}")
    if not os.path.exists(python):
        raise RuntimeError(f"could not create a virtual environment at {venv}")
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
    code = suite(f"provider --name {name} {extra} {ASR_FLAG}", python=python,
                 timeout_min=PROVIDER_TIMEOUT_MIN, log_name=f"provider_{name}", env=provider_env(name))
    if not os.path.exists(os.path.join(RESULTS, f"provider_{name}.json")):
        # The run died before it could record anything (killed for time or memory, or a crash at import).
        import json
        with open(os.path.join(RESULTS, f"provider_{name}.json"), "w") as fh:
            json.dump({"test": f"provider_{name}", "status": "fail", "metrics": {}, "notes": [],
                       "error": f"run ended with exit code {code} before writing a result — see logs/provider_{name}.log "
                                f"and logs/install_{name}.log", "log_tail": []}, fh)
    if RUN_LANGUAGES and OTHER_ENGINE_LANGUAGES:
        suite(f"languages --name {name} --langs {','.join(OTHER_ENGINE_LANGUAGES)} --seconds {LANGUAGE_SECONDS} {extra} {ASR_FLAG}",
              python=python, timeout_min=PROVIDER_TIMEOUT_MIN, log_name=f"languages_{name}", env=provider_env(name))
    if SCALING_OTHER_ENGINES:
        suite(f"scaling --name {name} --chunks {SCALING_OTHER_CHUNKS} --batch-size 4 {extra}",
              python=python, timeout_min=PROVIDER_TIMEOUT_MIN, log_name=f"scaling_{name}", env=provider_env(name))
    free_model_cache()   # make room for the next engine's weights
"""),
        _md("## 8 · Report"),
        _code("""
if RUN_SIMILARITY:
    # Compares the voice of every sample above with the narrator clip.
    suite("similarity", timeout_min=20)
suite("report")

from IPython.display import Markdown, display
with open(os.path.join(RESULTS, "REPORT.md"), encoding="utf-8") as fh:
    display(Markdown(fh.read()))

import zipfile
archive_path = os.path.join(WORK, "abm_test_results.zip")
with zipfile.ZipFile(archive_path) as archive:
    names = archive.namelist()
    broken = archive.testzip()
print(f"\\n📦 {archive_path}: {len(names)} files, {os.path.getsize(archive_path) / 2**20:.1f} MB")
for folder in ("logs", "samples"):
    print(f"   {folder}/: {sum(1 for n in names if f'/{folder}/' in n)} files")
if broken or len(names) < 3:
    raise RuntimeError(f"The results archive is incomplete (first bad file: {broken}); send the notebook itself instead.")
print("Download it from the Output panel (or the file browser on the right) and send it back.")
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
            "{python} -m pip install -q --no-deps --ignore-requires-python descript-audiotools "
            "\"indextts @ git+https://github.com/index-tts/index-tts.git@d9e41aac89fd00b3d71497fddb287b7f24613712\"",
        ],
        "env": {"USE_MODELSCOPE": "false"},
    },
    "fish": {
        # fish-speech pins torch/pydantic/datasets and pulls gradio, wandb and
        # lightning; its real inference dependencies are in the requirements file.
        "post": [
            "{python} -m pip install -q --no-deps descript-audiotools descript-audio-codec "
            "\"fish-speech @ git+https://github.com/fishaudio/fish-speech.git@214da3cd841bda85da2496b96cd3c4d7edb1337e\"",
        ],
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
