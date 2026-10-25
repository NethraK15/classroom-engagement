#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# run.sh — One-click setup and launch for EngageAI Dashboard
# Usage: bash run.sh
# ──────────────────────────────────────────────────────────────────────────────

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR="$SCRIPT_DIR/.venv"

# ── Find Python 3.9+ ────────────────────────────────────────────────────────
PYTHON=""
for candidate in python3.11 python3.10 python3.9 python3 python; do
  if command -v "$candidate" &>/dev/null; then
    VER=$("$candidate" -c "import sys; print(sys.version_info.major, sys.version_info.minor)")
    MAJOR=$(echo $VER | cut -d' ' -f1)
    MINOR=$(echo $VER | cut -d' ' -f2)
    if [ "$MAJOR" -eq 3 ] && [ "$MINOR" -ge 9 ]; then
      PYTHON=$(command -v "$candidate")
      echo "✅ Using Python: $PYTHON (3.$MINOR)"
      break
    fi
  fi
done

# Also check Homebrew
if [ -z "$PYTHON" ] && [ -f "/opt/homebrew/bin/python3.11" ]; then
  PYTHON="/opt/homebrew/bin/python3.11"
fi

if [ -z "$PYTHON" ]; then
  echo "❌ Python 3.9+ not found. Please install Python 3.9 or newer."
  exit 1
fi

# ── Create virtualenv if not present ────────────────────────────────────────
if [ ! -d "$VENV_DIR" ]; then
  echo "📦 Creating virtual environment at $VENV_DIR ..."
  "$PYTHON" -m venv "$VENV_DIR"
fi

# ── Activate venv ────────────────────────────────────────────────────────────
source "$VENV_DIR/bin/activate"
VENV_PYTHON="$VENV_DIR/bin/python"

# ── Install / upgrade packages ────────────────────────────────────────────────
echo "📥 Checking / installing dependencies..."
"$VENV_PYTHON" -m pip install --upgrade pip -q

# Check if Flask is already installed (quick check)
if ! "$VENV_PYTHON" -c "import flask" 2>/dev/null; then
  echo "Installing requirements..."
  "$VENV_PYTHON" -m pip install -r "$SCRIPT_DIR/requirements.txt" -q
  echo "✅ Dependencies installed."
else
  echo "✅ Dependencies already satisfied."
fi

# ── Check for ultralytics ─────────────────────────────────────────────────────
if ! "$VENV_PYTHON" -c "import ultralytics" 2>/dev/null; then
  echo "📥 Installing ultralytics + lapx..."
  "$VENV_PYTHON" -m pip install ultralytics lapx -q
fi

# ── Check for ffmpeg ──────────────────────────────────────────────────────────
if ! command -v ffmpeg &>/dev/null; then
  echo "⚠️  ffmpeg not found. Video re-encoding will be skipped (some browsers may not play the raw video)."
  echo "   macOS: brew install ffmpeg"
  echo "   Linux: sudo apt install ffmpeg"
fi

# ── Launch ────────────────────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════"
echo "  🎓 EngageAI — Classroom Engagement Dashboard"
echo "  http://localhost:5001"
echo "═══════════════════════════════════════════════"
echo ""

"$VENV_PYTHON" "$SCRIPT_DIR/app.py" "$@"
