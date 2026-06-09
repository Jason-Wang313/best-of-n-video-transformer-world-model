#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [[ -z "${PYTHON:-}" ]]; then
  if command -v python >/dev/null 2>&1; then
    PYTHON="python"
  elif command -v python.exe >/dev/null 2>&1; then
    PYTHON="python.exe"
  else
    echo "python or python.exe is required" >&2
    exit 1
  fi
fi

if "$PYTHON" -c "import os, sys; sys.exit(0 if os.name == 'nt' else 1)" >/dev/null 2>&1; then
  if command -v cygpath >/dev/null 2>&1; then
    ROOT_FOR_PY="$(cygpath -w "$ROOT")"
  elif command -v wslpath >/dev/null 2>&1; then
    ROOT_FOR_PY="$(wslpath -w "$ROOT")"
  else
    ROOT_FOR_PY="$ROOT"
  fi
  export PYTHONPATH="${ROOT_FOR_PY}\\src;${ROOT_FOR_PY}${PYTHONPATH:+;${PYTHONPATH}}"
else
  export PYTHONPATH="$ROOT/src:$ROOT:${PYTHONPATH:-}"
fi

"$PYTHON" -m scripts.run_claim_audit
