#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

# --- Auto-install if needed ---
if [ ! -d ".venv" ]; then
    echo "[!] First run detected. Running setup..."
    echo ""
    bash install.sh
    echo ""
fi

# --- Auto-repair venv if launchers are stale (folder renamed/moved) ---
if ! .venv/bin/pip --version >/dev/null 2>&1; then
    echo "[!] Broken venv launchers detected. Repairing..."
    if ! .venv/bin/python scripts/repair_venv.py; then
        echo "[ERROR] Could not repair the virtual environment."
        echo "        Recreate it:  rm -rf .venv && ./install.sh"
        exit 1
    fi
fi

echo ""
echo "  ========================================"
echo "       WALLIE - Starting..."
echo "  ========================================"
echo ""
echo "  Dashboard: http://127.0.0.1:8765"
echo "  Press Ctrl+C to stop."
echo ""

# --- Open browser after 3 seconds (background) ---
(sleep 3 && (open "http://127.0.0.1:8765" 2>/dev/null || xdg-open "http://127.0.0.1:8765" 2>/dev/null || true)) &

# --- Launch ---
.venv/bin/python wallie.py --dashboard
