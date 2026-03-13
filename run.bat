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

echo.
echo ========================================
echo   Starting AudiobookMaker
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

:: ── Launch app.py ─────────────────────────────────────────────────────────────
echo [INFO]  Starting AudiobookMaker on %URL% ...
echo [INFO]  Press Ctrl+C in this window to stop the server.
echo.

:: Start app.py and open the browser after a short delay.
:: We use start /B so the window stays alive and shows logs.
start /B python %APP%

:: Wait a few seconds for the server to start
echo [INFO]  Waiting for server to start...
timeout /t 5 /nobreak >nul

:: Open browser
echo [INFO]  Opening browser at %URL%
start "" "%URL%"

echo.
echo [OK]    AudiobookMaker is running. Close this window to stop.
echo.

:: Keep the console alive so logs keep printing
:: (app.py outputs to this console via start /B)
python %APP%

endlocal
