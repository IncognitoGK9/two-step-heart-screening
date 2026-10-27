#!/bin/bash
# Builds a clean git repository from this v3 folder in ~/hf_github (outside Google Drive) and pushes it to GitHub.
#   bash publish_github.sh [repo-name] [--public]
# Needs: git, rsync. Needs the GitHub CLI (gh, already logged in) to create the repo; without it the script stops after the
# commit and prints the manual commands. The repo is PRIVATE unless you pass --public.
set -e
NAME="two-step-heart-screening"; VIS="--private"
for a in "$@"; do case "$a" in --public) VIS="--public";; *) NAME="$a";; esac; done
SRC="$(cd "$(dirname "$0")" && pwd)"
DEST="$HOME/hf_github/$NAME"
case "$DEST" in *"My Drive"*|*"CloudStorage"*|*"Google Drive"*|*iCloud*|*Dropbox*) echo "STOP: $DEST is inside a synced folder."; exit 1;; esac
mkdir -p "$DEST"
rsync -a --delete --exclude .git --exclude .venv --exclude __pycache__ --exclude .pytest_cache --exclude '.DS_Store' \
      --exclude '.req_stamp' --exclude '.py_stamp' --exclude '_pv*' --exclude '_preview' --exclude '*.tmp' \
      --exclude 'external/*.zip' "$SRC"/ "$DEST"/
cd "$DEST"
cat > .gitignore <<'GI'
.venv/
__pycache__/
.pytest_cache/
.DS_Store
*.tmp
GI
[ -d .git ] || git init -q -b main
git add -A
git -c user.name="${GIT_AUTHOR_NAME:-$(git config user.name || echo author)}" -c user.email="${GIT_AUTHOR_EMAIL:-$(git config user.email || echo author@example.com)}" \
    commit -q -m "Two-step heart disease screening: pipeline, app, tests, results and paper" || echo "(nothing new to commit)"
echo "Repository prepared at: $DEST  ($(git ls-files | wc -l | tr -d ' ') files)"
if command -v gh >/dev/null 2>&1 && gh auth status >/dev/null 2>&1; then
  if git remote get-url origin >/dev/null 2>&1; then git push -u origin main
  else gh repo create "$NAME" $VIS --source . --remote origin --push; fi
  echo "Done: $(gh repo view --json url --jq .url)"
else
  echo; echo "gh is not installed or not logged in. Either run:  brew install gh && gh auth login   and run this script again,"
  echo "or create an empty repository named $NAME on github.com (no README), then run:"
  echo "  cd \"$DEST\" && git remote add origin https://github.com/<your-username>/$NAME.git && git push -u origin main"
fi
