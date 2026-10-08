#!/usr/bin/env bash
# =============================================================================
# AudiobookMaker — run.sh
# Activates the virtual environment, starts app.py, and opens the browser
# =============================================================================
set -euo pipefail

VENV_DIR="venv"
APP="app.py"
PORT=7860
URL="http://localhost:$PORT"

# ── Colours ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

# ── Environment Overrides ───────────────────────────────────────────────────
export TF_ENABLE_ONEDNN_OPTS=0
export PYTHONWARNINGS="ignore"
export GRADIO_ANALYTICS_ENABLED="False"
info()    { echo -e "${CYAN}[INFO]${RESET}  $*"; }
success() { echo -e "${GREEN}[OK]${RESET}    $*"; }

# ── Sanity checks ─────────────────────────────────────────────────────────────
if [[ ! -d "$VENV_DIR" ]]; then
    echo "[ERROR] Virtual environment not found. Run ./install.sh first."
    exit 1
fi

if [[ ! -f "$APP" ]]; then
    echo "[ERROR] $APP not found. Make sure you are in the AudiobookMaker directory."
    exit 1
fi

# ── Activate venv ─────────────────────────────────────────────────────────────
# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"
success "Virtual environment activated"

# ── Stop both processes however this script ends ─────────────────────────────
# The API server holds the TTS model in VRAM. Trapping only INT/TERM left it
# running whenever app.py exited on its own (crash, port in use, closed from
# the UI), so everything is stopped from an EXIT trap instead.
API_PID=""
APP_PID=""
STOP_TIMEOUT=15   # seconds to wait for a graceful stop before SIGKILL

stop_process() {
    local pid="$1"
    [[ -n "$pid" ]] || return 0
    kill -0 "$pid" 2>/dev/null || return 0
    kill "$pid" 2>/dev/null || return 0
    local waited=0
    while kill -0 "$pid" 2>/dev/null; do
        if [[ $waited -ge $((STOP_TIMEOUT * 2)) ]]; then
            kill -9 "$pid" 2>/dev/null || true
            break
        fi
        sleep 0.5
        waited=$((waited + 1))
    done
    wait "$pid" 2>/dev/null || true
}

cleanup() {
    local status=$?
    trap - EXIT INT TERM
    stop_process "$APP_PID"
    stop_process "$API_PID"
    echo ''
    echo 'AudiobookMaker stopped.'
    exit "$status"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# ── Start API backend & app.py in background ───────────────────────────────────
info "Starting API Orchestration Backend ..."
python start_api.py &
API_PID=$!

info "Starting AudiobookMaker on $URL (VRAM-Safe Orchestration Enabled) ..."
python "$APP" &
APP_PID=$!

# ── Wait for server to be ready ───────────────────────────────────────────────
info "Waiting for server..."
TIMEOUT=30
ELAPSED=0
APP_ALIVE=1
while ! curl -s "$URL" > /dev/null 2>&1; do
    if ! kill -0 "$APP_PID" 2>/dev/null; then
        APP_ALIVE=0
        break
    fi
    sleep 1
    ELAPSED=$((ELAPSED + 1))
    if [[ $ELAPSED -ge $TIMEOUT ]]; then
        echo "[WARN]  Server did not respond in ${TIMEOUT}s — opening browser anyway."
        break
    fi
done

if [[ $APP_ALIVE -eq 1 ]]; then
    # ── Open browser ──────────────────────────────────────────────────────────
    OS="$(uname -s)"
    if [[ "$OS" == "Darwin" ]]; then
        open "$URL" || true
    elif command -v xdg-open &>/dev/null; then
        xdg-open "$URL" || true
    elif command -v wslview &>/dev/null; then
        wslview "$URL" || true
    else
        echo "[INFO]  Open your browser at: $URL"
    fi

    success "AudiobookMaker is running at $URL"
    echo -e "${BOLD}  Press Ctrl+C to stop.${RESET}"
    echo ""
else
    echo "[ERROR] $APP exited during startup."
fi

# ── Keep running until app.py exits or the user stops it ─────────────────────
# The EXIT trap stops the API server in both cases.
APP_STATUS=0
wait "$APP_PID" || APP_STATUS=$?
exit "$APP_STATUS"
