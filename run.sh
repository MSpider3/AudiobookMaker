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
while ! curl -s "$URL" > /dev/null 2>&1; do
    sleep 1
    ELAPSED=$((ELAPSED + 1))
    if [[ $ELAPSED -ge $TIMEOUT ]]; then
        echo "[WARN]  Server did not respond in ${TIMEOUT}s — opening browser anyway."
        break
    fi
done

# ── Open browser ──────────────────────────────────────────────────────────────
OS="$(uname -s)"
if [[ "$OS" == "Darwin" ]]; then
    open "$URL"
elif command -v xdg-open &>/dev/null; then
    xdg-open "$URL"
elif command -v wslview &>/dev/null; then
    wslview "$URL"
else
    echo "[INFO]  Open your browser at: $URL"
fi

success "AudiobookMaker is running at $URL"
echo -e "${BOLD}  Press Ctrl+C to stop.${RESET}"
echo ""

# ── Keep running until user stops it ─────────────────────────────────────────
trap "kill $APP_PID $API_PID 2>/dev/null; echo ''; echo 'AudiobookMaker stopped.'" INT TERM
wait $APP_PID
