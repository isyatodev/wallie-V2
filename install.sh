#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

echo ""
echo "  ========================================"
echo "       WALLIE - One-Click Setup"
echo "  ========================================"
echo ""

# --- Find Python 3.11+ ---
PYTHON=""
for cmd in python3.13 python3.12 python3.11 python3 python; do
    if command -v "$cmd" &>/dev/null; then
        if "$cmd" -c "import sys; exit(0 if sys.version_info >= (3,11) else 1)" 2>/dev/null; then
            PYTHON="$cmd"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    echo "[!] Python 3.11+ not found."
    echo ""
    if command -v brew &>/dev/null; then
        echo "[*] Installing via Homebrew..."
        brew install python@3.12
        PYTHON="python3.12"
    elif command -v apt-get &>/dev/null; then
        echo "[*] Installing via apt..."
        sudo apt-get update && sudo apt-get install -y python3.12 python3.12-venv python3-pip
        PYTHON="python3.12"
    elif command -v dnf &>/dev/null; then
        echo "[*] Installing via dnf..."
        sudo dnf install -y python3.12
        PYTHON="python3.12"
    elif command -v pacman &>/dev/null; then
        echo "[*] Installing via pacman..."
        sudo pacman -S --noconfirm python
        PYTHON="python3"
    else
        echo "[ERROR] Could not install Python automatically."
        echo "        Install Python 3.11+ manually:"
        echo "          macOS:  brew install python@3.12"
        echo "          Ubuntu: sudo apt install python3.12 python3.12-venv"
        echo "          Arch:   sudo pacman -S python"
        exit 1
    fi
fi

echo "[OK] Using: $PYTHON ($($PYTHON --version 2>&1))"
echo ""

# --- Create virtual environment ---
if [ ! -d ".venv" ]; then
    echo "[*] Creating virtual environment..."
    $PYTHON -m venv .venv
    echo "[OK] Virtual environment created."
fi

# --- Auto-repair venv if launchers are stale (folder renamed/moved) ---
if ! .venv/bin/python scripts/repair_venv.py; then
    echo "[ERROR] Could not repair the virtual environment."
    echo "        Recreate it:  rm -rf .venv && ./install.sh"
    exit 1
fi

# --- Install dependencies ---
echo "[*] Installing dependencies..."
.venv/bin/python -m pip install --upgrade pip -q 2>/dev/null
.venv/bin/python -m pip install -r requirements.txt -q

# --- Verify Hearing deps (soundcard + faster-whisper) ---
if ! .venv/bin/python -c "import soundcard, faster_whisper" 2>/dev/null; then
    echo "[!] Hearing deps missing. Installing soundcard + faster-whisper..."
    .venv/bin/python -m pip install soundcard faster-whisper -q
    if ! .venv/bin/python -c "import soundcard, faster_whisper" 2>/dev/null; then
        echo "[ERROR] Could not install Hearing dependencies. Hearing will be disabled."
        echo "        Try manually: .venv/bin/python -m pip install soundcard faster-whisper"
        exit 1
    fi
fi
echo "[OK] All dependencies installed."

# --- Setup .env ---
if [ ! -f ".env" ] && [ -f ".env.example" ]; then
    cp .env.example .env
    chmod 600 .env
    echo "[OK] Created .env from template"
fi

# --- Ensure directories ---
mkdir -p profiles voices

echo ""
echo "  ========================================"
echo "       Setup complete!"
echo "       Run: ./start.sh"
echo "  ========================================"
echo ""
