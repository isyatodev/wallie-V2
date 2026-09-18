@echo off
cd /d "%~dp0"
title Wallie - Setup

echo.
echo   ========================================
echo        WALLIE - One-Click Setup
echo   ========================================
echo.

REM --- Find Python 3.11+ ---
set PYTHON=

py -3 --version >nul 2>&1 && set PYTHON=py -3
if not defined PYTHON (
    python --version >nul 2>&1 && set PYTHON=python
)

if not defined PYTHON goto :install_python

REM --- Version check ---
%PYTHON% -c "import sys; exit(0 if sys.version_info >= (3,11) else 1)" 2>nul
if errorlevel 1 (
    echo [!] Python 3.11+ required. Current version:
    %PYTHON% --version
    goto :install_python
)

goto :python_ok

:install_python
echo [!] Python 3.11+ not found. Installing...
echo.
winget install Python.Python.3.12 --accept-package-agreements --accept-source-agreements
if errorlevel 1 (
    echo.
    echo [ERROR] Could not install Python automatically.
    echo         Download from: https://www.python.org/downloads/
    echo         IMPORTANT: Check "Add Python to PATH" during installation!
    echo.
    pause
    exit /b 1
)
echo.
echo [OK] Python installed successfully.
echo     Close this window and double-click install.bat AGAIN.
echo     (PATH needs to refresh)
pause
exit /b 0

:python_ok
echo [OK] %PYTHON% -^> & %PYTHON% --version
echo.

REM --- Create virtual environment ---
if not exist ".venv\Scripts\python.exe" (
    echo [*] Creating virtual environment...
    %PYTHON% -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created.
)

REM --- Install dependencies ---
echo [*] Installing dependencies (this may take a minute)...
.venv\Scripts\python.exe -m pip install --upgrade pip -q 2>nul
.venv\Scripts\pip.exe install -r requirements.txt -q
if errorlevel 1 (
    echo.
    echo [ERROR] Dependency installation failed.
    echo         Check your internet connection and try again.
    pause
    exit /b 1
)

REM --- Verify Hearing deps (soundcard + faster-whisper) ---
.venv\Scripts\python.exe -c "import soundcard, faster_whisper" 2>nul
if errorlevel 1 (
    echo [!] Hearing deps missing. Installing soundcard + faster-whisper...
    .venv\Scripts\pip.exe install soundcard faster-whisper -q
    .venv\Scripts\python.exe -c "import soundcard, faster_whisper" 2>nul
    if errorlevel 1 (
        echo [ERROR] Could not install Hearing dependencies. Hearing will be disabled.
        echo         Try manually: .venv\Scripts\pip install soundcard faster-whisper
        pause
        exit /b 1
    )
)
echo [OK] All dependencies installed.

REM --- Setup .env ---
if not exist ".env" (
    if exist ".env.example" (
        copy .env.example .env >nul
        echo [OK] Created .env from template
    )
)

REM --- Ensure directories ---
if not exist "profiles" mkdir profiles
if not exist "voices" mkdir voices

echo.
echo   ========================================
echo        Setup complete! Launching Wallie...
echo   ========================================
echo.
call start.bat
