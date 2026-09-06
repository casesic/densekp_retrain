#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python tools/inspect_surface_operator.py
python - <<'PY'
import ast
from pathlib import Path
for root in (Path('densekp'), Path('tools')):
    for path in root.rglob('*.py'):
        ast.parse(path.read_text(), filename=str(path))
print('python syntax: PASS')
PY
find . -type f -size +100M -print
