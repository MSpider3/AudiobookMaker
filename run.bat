@echo off
:: =============================================================================
:: AudiobookMaker — run.bat
:: Activates the virtual environment, starts app.py, and opens the browser
:: =============================================================================
setlocal

set VENV_DIR=venv
set APP=app.py
set PORT=7860
set URL=http://localhost:%PORT%

:: ── Environment Overrides ────────────────────────────────────────────────────
:: Suppress TensorFlow oneDNN noise and standard Python warnings
set TF_ENABLE_ONEDNN_OPTS=0
set PYTHONWARNINGS=ignore
set GRADIO_ANALYTICS_ENABLED=False

echo.
echo ========================================
echo   Starting AudiobookMaker
echo   (VRAM-Safe Orchestration Enabled)
echo ========================================
echo.

:: ── Sanity checks ────────────────────────────────────────────────────────────
if not exist "%VENV_DIR%\Scripts\activate.bat" (
    echo [ERROR] Virtual environment not found.
    echo         Run install.bat first.
    pause
    exit /b 1
)

if not exist "%APP%" (
    echo [ERROR] %APP% not found.
    echo         Make sure you are running this from the AudiobookMaker folder.
    pause
    exit /b 1
)

:: ── Activate venv ─────────────────────────────────────────────────────────────
call %VENV_DIR%\Scripts\activate.bat
echo [OK]    Virtual environment activated.

:: ── Launch app.py & API Backend ────────────────────────────────────────────────
echo [INFO]  Launching API Orchestration Backend ...
start /min "AudiobookMaker API Server" python start_api.py

echo [INFO]  Starting AudiobookMaker on %URL% ...
echo [INFO]  Press Ctrl+C in this window to stop the server.
echo.

:: Open browser in the background first so it waits for the server
echo [INFO]  Opening browser at %URL%
start "" "%URL%"

echo.
echo [OK]    AudiobookMaker is launching. Keep this window open.
echo.

:: Start app.py in the foreground (blocks until closed by user)
python "%APP%"

endlocal
