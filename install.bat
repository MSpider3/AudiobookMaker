@echo off
:: =============================================================================
:: AudiobookMaker — install.bat
:: Windows installer using winget + py launcher
:: =============================================================================
setlocal EnableDelayedExpansion

set PYTHON_VERSION=3.11
set VENV_DIR=venv
set REQUIREMENTS=requirements.txt

echo.
echo ========================================
echo   AudiobookMaker Installer (Windows)
echo ========================================
echo.

:: ── Step 1: Install Python 3.11 ─────────────────────────────────────────────
echo [Step 1] Checking Python %PYTHON_VERSION%...

:: Check if py launcher has 3.11
py -%PYTHON_VERSION% --version >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    for /f "tokens=*" %%v in ('py -%PYTHON_VERSION% --version 2^>^&1') do echo [OK]    Found: %%v
    set PYTHON_CMD=py -%PYTHON_VERSION%
    goto :python_ok
)

:: Check python3.11 directly on PATH
python3.11 --version >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    set PYTHON_CMD=python3.11
    goto :python_ok
)

:: Try winget
echo [INFO]  Python %PYTHON_VERSION% not found. Installing via winget...
winget install -e --id Python.Python.%PYTHON_VERSION% --accept-source-agreements --accept-package-agreements
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] winget install failed.
    echo         Please install Python %PYTHON_VERSION% manually from https://www.python.org/downloads/
    pause
    exit /b 1
)

:: Refresh PATH so py or python can be found
:: Re-check
py -%PYTHON_VERSION% --version >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    set PYTHON_CMD=py -%PYTHON_VERSION%
    goto :python_ok
)
echo [ERROR] Python was installed but still not found. Restart this terminal and try again.
pause
exit /b 1

:python_ok
echo [OK]    Python %PYTHON_VERSION% ready.
echo.

:: ── Step 2: Create virtual environment ──────────────────────────────────────
echo [Step 2] Creating virtual environment...

if exist "%VENV_DIR%\Scripts\activate.bat" (
    echo [WARN]  Virtual environment already exists — skipping creation.
) else (
    %PYTHON_CMD% -m venv %VENV_DIR%
    if %ERRORLEVEL% NEQ 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo [OK]    Virtual environment created in .\%VENV_DIR%
)

call %VENV_DIR%\Scripts\activate.bat
echo [OK]    Virtual environment activated.
echo.

python -m pip install --upgrade pip --quiet

:: ── Step 3: Detect GPU and install PyTorch ───────────────────────────────────
echo [Step 3] Detecting GPU...

set HAS_CUDA=0
set CUDA_VER=

nvidia-smi >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [OK]    NVIDIA GPU detected.
    for /f "tokens=9" %%c in ('nvidia-smi ^| findstr "CUDA Version"') do set CUDA_VER=%%c
    echo [INFO]  CUDA version: !CUDA_VER!
    set HAS_CUDA=1
) else (
    echo [INFO]  No NVIDIA GPU detected — will install CPU-only PyTorch.
)

echo.
if %HAS_CUDA%==1 (
    :: Pick CUDA index URL based on version
    set CUDA_MAJOR=0
    for /f "delims=." %%a in ("!CUDA_VER!") do set CUDA_MAJOR=%%a
    if !CUDA_MAJOR! GEQ 12 (
        echo [INFO]  Installing PyTorch for CUDA 12.1...
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --quiet
    ) else (
        echo [INFO]  Installing PyTorch for CUDA 11.8...
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 --quiet
    )
    echo [OK]    PyTorch installed with CUDA support.
) else (
    echo [INFO]  Installing CPU-only PyTorch...
    pip install torch torchvision torchaudio --quiet
    echo [OK]    PyTorch installed (CPU only).
)
echo.

:: ── Step 4: Install project dependencies ─────────────────────────────────────
echo [Step 4] Installing project dependencies...

if not exist %REQUIREMENTS% (
    echo [ERROR] %REQUIREMENTS% not found in current directory.
    pause
    exit /b 1
)

:: Install all requirements except torch lines (already installed)
:: Use Python to filter and install
python -c "
import subprocess, sys
reqs = []
with open('%REQUIREMENTS%') as f:
    for line in f:
        stripped = line.strip()
        if not stripped or stripped.startswith('#') or stripped.lower().startswith('torch'):
            continue
        reqs.append(stripped)
subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--quiet'] + reqs)
"
if %ERRORLEVEL% NEQ 0 (
    echo [ERROR] Dependency installation failed.
    pause
    exit /b 1
)
echo [OK]    All dependencies installed.
echo.

:: ── Step 5: Check FFmpeg ─────────────────────────────────────────────────────
echo [Step 5] Checking FFmpeg...

ffmpeg -version >nul 2>&1
if %ERRORLEVEL% EQU 0 (
    echo [OK]    FFmpeg is already installed.
) else (
    echo [INFO]  FFmpeg not found. Installing via winget...
    winget install -e --id Gyan.FFmpeg --accept-source-agreements --accept-package-agreements
    if %ERRORLEVEL% NEQ 0 (
        echo [WARN]  Could not auto-install FFmpeg.
        echo         Please install it manually from https://www.gyan.dev/ffmpeg/builds/
        echo         and add it to your system PATH.
    ) else (
        echo [OK]    FFmpeg installed. You may need to restart your terminal.
    )
)

:: ── Done ─────────────────────────────────────────────────────────────────────
echo.
echo ========================================
echo   Installation complete!
echo ========================================
echo.
echo   To start AudiobookMaker, run:
echo     run.bat
echo.
pause
endlocal
