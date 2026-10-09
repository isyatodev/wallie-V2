@echo off
cd /d "%~dp0"
title Wallie - Setup

echo.
echo   ========================================
echo        WALLIE - One-Click Setup
echo   ========================================
echo.

REM --- Find Python 3.11+ ---
REM Prefer a Kokoro-compatible interpreter: the multilingual voice needs
REM kokoro>=0.9.4, which ships wheels only for 3.10-3.12. `py -3` returns the
REM NEWEST Python, so on a machine that also has 3.13/3.14 it would build a venv
REM that can never install the free local voice.
set PYTHON=

py -3.12 --version >nul 2>&1 && set PYTHON=py -3.12
if not defined PYTHON (
    py -3.11 --version >nul 2>&1 && set PYTHON=py -3.11
)
if not defined PYTHON (
    py -3.10 --version >nul 2>&1 && set PYTHON=py -3.10
)
if not defined PYTHON (
    py -3 --version >nul 2>&1 && set PYTHON=py -3
)
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
%PYTHON% -c "import sys; exit(0 if sys.version_info < (3,13) else 1)" 2>nul
if errorlevel 1 (
    echo.
    echo [!] This Python is 3.13 or newer. Wallie will run, but Kokoro, the
    echo     free local voice, has no wheels for it and cannot be installed.
    echo     Install Python 3.12 ^(https://www.python.org/downloads/^) and run
    echo     install.bat again if you want that voice.
)
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

REM --- Auto-repair venv if launchers are stale (folder renamed/moved) ---
.venv\Scripts\python.exe "%~dp0scripts\repair_venv.py"
if errorlevel 1 (
    echo.
    echo [ERROR] Could not repair the virtual environment.
    echo         Recreate it:  rmdir /s /q .venv  &&  run install.bat again
    pause
    exit /b 1
)

echo [*] Installing dependencies (this may take a minute)...
.venv\Scripts\python.exe -m pip install --upgrade pip -q 2>nul
.venv\Scripts\python.exe -m pip install -r requirements.txt -q
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
    .venv\Scripts\python.exe -m pip install soundcard faster-whisper -q
    .venv\Scripts\python.exe -c "import soundcard, faster_whisper" 2>nul
    if errorlevel 1 (
        echo [ERROR] Could not install Hearing dependencies. Hearing will be disabled.
        echo         Try manually: .venv\Scripts\python -m pip install soundcard faster-whisper
        pause
        exit /b 1
    )
)

REM --- Sanity check: pip launcher must spawn (catches renamed-folder breakage) ---
.venv\Scripts\pip.exe --version >nul 2>&1
if errorlevel 1 (
    echo [!] pip.exe launcher still stale — running repair again...
    .venv\Scripts\python.exe "%~dp0scripts\repair_venv.py"
    .venv\Scripts\pip.exe --version >nul 2>&1
    if errorlevel 1 (
        echo [ERROR] Could not repair the virtual environment.
        echo         Recreate it:  rmdir /s /q .venv  &&  run install.bat again
        pause
        exit /b 1
    )
)
echo [OK] All dependencies installed.

REM --- Free local voice (Kokoro): install, then VERIFY it for real --------------
REM Kokoro is Wallie's free, fully local voice. The setup installs it and then
REM PROVES the installed environment can actually speak with it: a line is really
REM synthesised through the same provider class a live session builds, so the
REM closing report is evidence instead of an assumption.
REM   * ~500 MB on the first run (PyTorch + model). Skip with WALLIE_SKIP_KOKORO=1.
REM   * Defaults to the project's default voice (English US, af_heart). Prefer
REM     another language?  WALLIE_KOKORO_LANG=p WALLIE_KOKORO_VOICE=pf_dora
REM   * Any failure here is non-fatal: the hosted voices still work.
set KOKORO_SCRIPT=%~dp0scripts\install_kokoro.py
set KOKORO_LANG=%WALLIE_KOKORO_LANG%
if not defined KOKORO_LANG set KOKORO_LANG=a
set KOKORO_VOICE=%WALLIE_KOKORO_VOICE%
set KOKORO_ARGS=--lang %KOKORO_LANG%
if defined KOKORO_VOICE set KOKORO_ARGS=%KOKORO_ARGS% --voice %KOKORO_VOICE%
set KOKORO_STATE=NOT CHECKED
set KOKORO_HINT=
set KOKORO_RETRY=.venv\Scripts\python.exe scripts\install_kokoro.py --install --play %KOKORO_ARGS%

if not exist "%KOKORO_SCRIPT%" goto :kokoro_no_script
if defined WALLIE_SKIP_KOKORO goto :kokoro_skip

REM Kokoro >= 0.9.4 ships wheels for Python 3.10-3.12 only, so on a newer venv the
REM installer would refuse: report it here instead of failing halfway through pip.
.venv\Scripts\python.exe -c "import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] < (3,13) else 1)" 2>nul
if errorlevel 1 goto :kokoro_old_python

REM Installed by an earlier run? Then skip straight to the verification below.
.venv\Scripts\python.exe -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('kokoro') and importlib.util.find_spec('soundfile') else 1)" 2>nul
if errorlevel 1 goto :kokoro_install
echo [*] Free local voice ^(Kokoro^) is already installed - verifying it...
.venv\Scripts\python.exe "%KOKORO_SCRIPT%" --play %KOKORO_ARGS%
goto :kokoro_result

:kokoro_install
echo [*] Installing the free local voice ^(Kokoro^) - PyTorch + model, ~500 MB.
echo     The first run takes a few minutes; WALLIE_SKIP_KOKORO=1 skips it.
.venv\Scripts\python.exe "%KOKORO_SCRIPT%" --install --play %KOKORO_ARGS%

:kokoro_result
REM --play IS the verification: it speaks one line through the SAME provider and
REM player a live session builds (tts.kokoro.KokoroTTS into audio.player.AudioPlayer)
REM and out the output device the active profile saved. That proves both halves of
REM "the voice works" - it synthesises, and it reaches the configured device, which
REM is what breaks when a device was renamed or unplugged.
if errorlevel 1 set KOKORO_STATE=NOT WORKING - install or verify failed
if errorlevel 1 set KOKORO_HINT=Retry later: %KOKORO_RETRY%
if errorlevel 1 goto :kokoro_failed
set KOKORO_STATE=READY - it spoke a real line through the configured device, lang %KOKORO_LANG%
echo [OK] Free local voice ^(Kokoro^) verified - Voice tab, provider "kokoro".
goto :kokoro_done

:kokoro_failed
echo.
echo [!] The free local voice is NOT ready. Wallie still runs with the hosted
echo     voices ^(Fish / ElevenLabs^). Fix it later with:
echo       %KOKORO_RETRY%
goto :kokoro_done

:kokoro_old_python
set KOKORO_STATE=NOT AVAILABLE - the free voice needs Python 3.10-3.12
set KOKORO_HINT=Install Python 3.12, recreate .venv, then run install.bat again.
echo [!] The free local voice ^(Kokoro^) cannot be installed on this Python: it
echo     needs 3.10-3.12. Wallie runs fine without it ^(hosted voices^).
goto :kokoro_done

:kokoro_no_script
set KOKORO_STATE=NOT CHECKED - scripts\install_kokoro.py is missing
goto :kokoro_done

:kokoro_skip
set KOKORO_STATE=SKIPPED - WALLIE_SKIP_KOKORO is set
echo [*] Skipping the free local voice ^(Kokoro^) - WALLIE_SKIP_KOKORO is set.

:kokoro_done

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

REM --- Closing report: state the voice verdict BEFORE claiming the setup is done ---
echo.
echo   ----------------------------------------
echo    Free local voice ^(Kokoro^): %KOKORO_STATE%
if defined KOKORO_HINT echo    %KOKORO_HINT%
echo   ----------------------------------------

echo.
echo   ========================================
echo        Setup complete! Launching Wallie...
echo   ========================================
echo.
call start.bat
