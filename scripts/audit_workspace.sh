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
"$python_bin" tools/annotation/inspect_surface_operator.py
"$python_bin" - <<'PY'
import ast
from pathlib import Path
for root in (Path('densekp'), Path('src'), Path('tools'), Path('tests')):
    for path in root.rglob('*.py'):
        ast.parse(path.read_text(), filename=str(path))
print('python syntax: PASS')
PY
find . -type f -size +100M -print
