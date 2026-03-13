#!/usr/bin/env bash
# =============================================================================
# AudiobookMaker — install.sh
# Cross-platform installer for macOS and Linux
# =============================================================================
set -euo pipefail

PYTHON_VERSION="3.11"
VENV_DIR="venv"
REQUIREMENTS="requirements.txt"

# ── Colours ───────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'

info()    { echo -e "${CYAN}[INFO]${RESET}  $*"; }
success() { echo -e "${GREEN}[OK]${RESET}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${RESET}  $*"; }
error()   { echo -e "${RED}[ERROR]${RESET} $*"; exit 1; }
header()  { echo -e "\n${BOLD}$*${RESET}"; }

# ── Detect OS ─────────────────────────────────────────────────────────────────
header "=== AudiobookMaker Installer ==="

OS="$(uname -s)"
DISTRO=""
PKG_MANAGER=""

if [[ "$OS" == "Darwin" ]]; then
    DISTRO="macos"
    info "Detected: macOS"
elif [[ "$OS" == "Linux" ]]; then
    if [ -f /etc/os-release ]; then
        # shellcheck source=/dev/null
        . /etc/os-release
        DISTRO="${ID:-unknown}"
    fi
    info "Detected: Linux ($DISTRO)"
else
    error "Unsupported OS: $OS. Use install.bat on Windows."
fi

# ── Step 1: Install Python 3.11 ───────────────────────────────────────────────
header "Step 1 — Installing Python $PYTHON_VERSION"

PYTHON_BIN=""

# Check if already installed
for cmd in python3.11 python3 python; do
    if command -v "$cmd" &>/dev/null; then
        VER=$("$cmd" --version 2>&1 | grep -oP '\d+\.\d+' | head -1)
        MAJOR=$(echo "$VER" | cut -d. -f1)
        MINOR=$(echo "$VER" | cut -d. -f2)
        if [[ "$MAJOR" -eq 3 && "$MINOR" -ge 11 ]]; then
            PYTHON_BIN="$cmd"
            success "Python $VER already installed ($cmd)"
            break
        fi
    fi
done

if [[ -z "$PYTHON_BIN" ]]; then
    info "Python $PYTHON_VERSION not found — installing..."

    case "$DISTRO" in
        macos)
            if ! command -v brew &>/dev/null; then
                error "Homebrew not found. Install it first: https://brew.sh"
            fi
            brew install "python@$PYTHON_VERSION"
            PYTHON_BIN="python3.11"
            ;;
        ubuntu|linuxmint|pop|elementary|zorin|kali)
            sudo apt-get update -qq
            sudo apt-get install -y "python${PYTHON_VERSION}" "python${PYTHON_VERSION}-venv" \
                "python${PYTHON_VERSION}-dev" python3-pip
            PYTHON_BIN="python${PYTHON_VERSION}"
            ;;
        debian)
            sudo apt-get update -qq
            sudo apt-get install -y "python${PYTHON_VERSION}" "python${PYTHON_VERSION}-venv" \
                "python${PYTHON_VERSION}-dev" python3-pip || {
                # Debian may need deadsnakes PPA via backports
                warn "Trying backports..."
                sudo apt-get install -y -t "$(lsb_release -cs)-backports" \
                    "python${PYTHON_VERSION}" "python${PYTHON_VERSION}-venv"
            }
            PYTHON_BIN="python${PYTHON_VERSION}"
            ;;
        fedora)
            sudo dnf install -y "python${PYTHON_VERSION//.}" python3-pip
            PYTHON_BIN="python${PYTHON_VERSION}"
            ;;
        rhel|centos|almalinux|rocky)
            sudo dnf install -y epel-release
            sudo dnf install -y "python${PYTHON_VERSION//.}" python3-pip
            PYTHON_BIN="python${PYTHON_VERSION}"
            ;;
        arch|manjaro|endeavouros|garuda)
            sudo pacman -Sy --noconfirm python python-pip
            PYTHON_BIN="python3"
            ;;
        opensuse*|suse*)
            sudo zypper install -y "python311" python3-pip
            PYTHON_BIN="python3.11"
            ;;
        *)
            warn "Unknown distro '$DISTRO' — trying generic apt/dnf/pacman..."
            if command -v apt-get &>/dev/null; then
                sudo apt-get update -qq
                sudo apt-get install -y "python${PYTHON_VERSION}" \
                    "python${PYTHON_VERSION}-venv" python3-pip
                PYTHON_BIN="python${PYTHON_VERSION}"
            elif command -v dnf &>/dev/null; then
                sudo dnf install -y python3 python3-pip
                PYTHON_BIN="python3"
            elif command -v pacman &>/dev/null; then
                sudo pacman -Sy --noconfirm python python-pip
                PYTHON_BIN="python3"
            else
                error "Could not find a known package manager. Install Python $PYTHON_VERSION manually."
            fi
            ;;
    esac
    success "Python installed: $($PYTHON_BIN --version)"
fi

# ── Step 2: Create virtual environment ────────────────────────────────────────
header "Step 2 — Creating virtual environment"

if [[ -d "$VENV_DIR" ]]; then
    warn "Virtual environment '$VENV_DIR' already exists — skipping creation."
else
    "$PYTHON_BIN" -m venv "$VENV_DIR"
    success "Virtual environment created in ./$VENV_DIR"
fi

# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"
success "Virtual environment activated"

pip install --upgrade pip --quiet

# ── Step 3: Detect GPU and install PyTorch ────────────────────────────────────
header "Step 3 — Detecting GPU and installing PyTorch"

HAS_CUDA=false
CUDA_VER=""

if command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "Unknown")
    success "NVIDIA GPU detected: $GPU_NAME"

    # Try to get CUDA version
    CUDA_VER=$(nvidia-smi | grep -oP 'CUDA Version: \K[\d.]+' | head -1 || echo "")
    if [[ -n "$CUDA_VER" ]]; then
        CUDA_MAJOR=$(echo "$CUDA_VER" | cut -d. -f1)
        CUDA_MINOR=$(echo "$CUDA_VER" | cut -d. -f2)
        info "CUDA version: $CUDA_VER"
    fi
    HAS_CUDA=true
elif [[ "$DISTRO" == "macos" ]]; then
    # Check for Apple Silicon (MPS)
    ARCH=$(uname -m)
    if [[ "$ARCH" == "arm64" ]]; then
        success "Apple Silicon (MPS) detected — PyTorch will use Metal Performance Shaders"
    else
        info "macOS Intel CPU — no GPU acceleration"
    fi
fi

if [[ "$HAS_CUDA" == true ]]; then
    # Pick highest compatible CUDA index URL
    if [[ -n "$CUDA_MAJOR" ]] && [[ "$CUDA_MAJOR" -ge 12 ]]; then
        TORCH_INDEX="https://download.pytorch.org/whl/cu121"
        info "Installing PyTorch with CUDA 12.1 support..."
    else
        TORCH_INDEX="https://download.pytorch.org/whl/cu118"
        info "Installing PyTorch with CUDA 11.8 support..."
    fi
    pip install torch torchvision torchaudio --index-url "$TORCH_INDEX" --quiet
    success "PyTorch installed with CUDA support"
else
    info "No NVIDIA GPU found — installing CPU-only PyTorch..."
    pip install torch torchvision torchaudio --quiet
    success "PyTorch installed (CPU)"
fi

# ── Step 4: Install all other requirements ────────────────────────────────────
header "Step 4 — Installing project dependencies"

if [[ ! -f "$REQUIREMENTS" ]]; then
    error "Cannot find $REQUIREMENTS in current directory."
fi

# Install everything except torch (already installed above)
grep -v -E "^#|^\s*$|^torch" "$REQUIREMENTS" | pip install -r /dev/stdin --quiet
success "All dependencies installed"

# ── Step 5: Install FFmpeg if missing ─────────────────────────────────────────
header "Step 5 — Checking FFmpeg"

if command -v ffmpeg &>/dev/null; then
    success "FFmpeg is already installed: $(ffmpeg -version 2>&1 | head -1)"
else
    warn "FFmpeg not found — attempting to install..."
    case "$DISTRO" in
        macos)                                brew install ffmpeg ;;
        ubuntu|linuxmint|pop|debian|kali|zorin) sudo apt-get install -y ffmpeg ;;
        fedora|rhel|centos|almalinux|rocky)   sudo dnf install -y ffmpeg ;;
        arch|manjaro|endeavouros|garuda)       sudo pacman -Sy --noconfirm ffmpeg ;;
        opensuse*|suse*)                       sudo zypper install -y ffmpeg ;;
        *) warn "Could not auto-install FFmpeg. Please install it manually: https://ffmpeg.org" ;;
    esac
    command -v ffmpeg &>/dev/null && success "FFmpeg installed" || \
        warn "FFmpeg install may have failed — check manually"
fi

# ── Done ──────────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}${BOLD}✅ Installation complete!${RESET}"
echo ""
echo -e "  To start AudiobookMaker, run:  ${CYAN}./run.sh${RESET}"
echo ""
