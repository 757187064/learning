#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REMOTE_URL="${1:-git@github.com:757187064/learning.git}"

cd "$REPO_ROOT"

rm -f .DS_Store site/.DS_Store
find scripts -name '__pycache__' -type d -prune -exec rm -rf {} +
find scripts -name '*.pyc' -type f -delete

if [ ! -d .git ]; then
  git init -b main
fi

git add .

if git diff --cached --quiet; then
  echo "No staged changes to commit."
else
  git commit -m "Publish review site"
fi

if git remote get-url origin >/dev/null 2>&1; then
  git remote set-url origin "$REMOTE_URL"
else
  git remote add origin "$REMOTE_URL"
fi

git push -u origin main
