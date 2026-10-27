#!/bin/bash
# One command: copies the code from Google Drive to ~/hf_run, builds a private venv there
# (~/hf_run/.venv, created once, reused), runs the study, copies results/ and figures/ back to Drive.
#   bash run_local.sh [--test | --app | --export | --site] [--quick] [--limit-folds K] [--no-holdout] [--jobs N]
set -e
MODE=run
case "${1:-}" in --test) MODE=test; shift;; --app) MODE=app; shift;; --export) MODE=export; shift;; --site) MODE=site; shift;; esac
SRC="$(cd "$(dirname "$0")" && pwd)"
WORK="$HOME/hf_run"
case "$WORK" in
  *"My Drive"*|*"CloudStorage"*|*"Google Drive"*|*iCloud*|*Dropbox*)
    echo "STOP: $WORK is inside a synced folder. The venv must stay outside Drive."; exit 1;;
esac

# Ignore any virtual environment that is active in this terminal (it belongs to another project).
if [ -n "$VIRTUAL_ENV" ]; then
  PATH="$(echo "$PATH" | tr ':' '\n' | grep -vxF "$VIRTUAL_ENV/bin" | paste -sd: -)"
  unset VIRTUAL_ENV
fi
PY=""
for c in python3.13 python3.12 python3.11 python3.10 python3; do
  if command -v "$c" >/dev/null 2>&1; then PY="$c"; break; fi
done
[ -n "$PY" ] || { echo "No python3 found. Install one (for example: brew install python@3.12)."; exit 1; }
PYV="$($PY --version 2>&1)"
echo "Using $PYV at $(command -v $PY)"

mkdir -p "$WORK"
rsync -a --delete --exclude .venv --exclude app/models --exclude .req_stamp --exclude .py_stamp --exclude results --exclude figures \
      --exclude paper --exclude docs --exclude '.DS_Store' --exclude __pycache__ \
      "$SRC/" "$WORK/"
cd "$WORK"
if [ -d .venv ] && [ "$(cat .py_stamp 2>/dev/null)" != "$PYV" ]; then rm -rf .venv .req_stamp; fi
if [ ! -d .venv ]; then "$PY" -m venv .venv; echo "$PYV" > .py_stamp; fi
source .venv/bin/activate
STAMP="$(shasum requirements.txt | cut -d' ' -f1)"
if [ "$(cat .req_stamp 2>/dev/null)" != "$STAMP" ]; then
  pip install --quiet --upgrade pip
  pip install --quiet -r requirements.txt
  echo "$STAMP" > .req_stamp
fi
case "$MODE" in
  test)   python -m pytest tests -q; exit;;
  app)    streamlit run app/app.py; exit;;
  site)   python scripts/05_site_validation.py | tee results/site_validation_log.txt
          mkdir -p "$SRC/results"; rsync -a "$WORK/results/" "$SRC/results/"; echo "Copied to: $SRC/results"; exit;;
  export) python scripts/run_all.py --export-only "$@"; mkdir -p "$SRC/app/models"; rsync -a "$WORK/app/models/" "$SRC/app/models/"
          echo "Models copied back to: $SRC/app/models"; exit;;
esac
python scripts/run_all.py "$@"
mkdir -p "$SRC/app/models"; [ -d "$WORK/app/models" ] && rsync -a "$WORK/app/models/" "$SRC/app/models/"
mkdir -p "$SRC/results" "$SRC/figures"
rsync -a "$WORK/results/" "$SRC/results/"
rsync -a "$WORK/figures/" "$SRC/figures/"
echo "Results copied back to: $SRC/results"
