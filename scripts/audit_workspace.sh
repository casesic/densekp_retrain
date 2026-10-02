#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -n "${PYTHON_BIN:-}" ]]; then
    python_bin="$PYTHON_BIN"
elif command -v python >/dev/null 2>&1; then
    python_bin=python
elif command -v python3 >/dev/null 2>&1; then
    python_bin=python3
else
    echo "Python executable not found; set PYTHON_BIN." >&2
    exit 1
fi
exec "$python_bin" tools/validate/check_workspace.py "$@"
