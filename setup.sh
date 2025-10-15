#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"
LOG_DIR="$PROJECT_ROOT/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/setup.log"
: > "$LOG_FILE"
exec > >(tee -a "$LOG_FILE") 2>&1
echo "=== RUN START (setup.sh) ==="

finish() {
  echo "=== RUN FINISH (setup.sh) ==="
}

trap finish EXIT

PYTHON_BIN=${PYTHON_BIN:-python3}
VENV_DIR="venv"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Error: $PYTHON_BIN is not available on this system." >&2
  exit 1
fi

if [ -d "$VENV_DIR" ]; then
  echo "Virtual environment '$VENV_DIR' already exists. Skipping creation."
else
  echo "Creating virtual environment '$VENV_DIR' with $PYTHON_BIN..."
  "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

# Activate the virtual environment and install requirements
source "$VENV_DIR/bin/activate"

python --version
pip install --upgrade pip

if [ -f "requirements.txt" ]; then
  echo "Installing Python dependencies from requirements.txt..."
  pip install -r requirements.txt
else
  echo "Warning: requirements.txt not found. Skipping dependency installation." >&2
fi

deactivate

echo "Setup complete. Virtual environment available at '$VENV_DIR'."
