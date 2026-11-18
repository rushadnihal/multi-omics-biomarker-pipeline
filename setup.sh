#!/usr/bin/env bash
# Safe to re-run: creates folders, venv, installs deps.
set -e
mkdir -p data/{raw,interim,processed} scripts results/{tables,figures,models} docs
PY=$(command -v python3.12 || command -v python3)
[ -d .venv ] || "$PY" -m venv .venv
.venv/bin/pip install -q -U pip
.venv/bin/pip install -q -r requirements-lock.txt
.venv/bin/pip install -q -e . --no-deps
git init -q 2>/dev/null || true
echo "setup done"
