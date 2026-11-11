#!/usr/bin/env bash
# Publica este directorio como repositorio en tu GitHub.
# Requiere GitHub CLI autenticado: `brew install gh && gh auth login`
# Uso: bash scripts/publish.sh [nombre-repo] [--private]
set -euo pipefail
cd "$(dirname "$0")/.."
NAME="${1:-swarmtrader}"
VIS="${2:---public}"
if ! command -v gh >/dev/null; then
  echo "instala GitHub CLI: brew install gh && gh auth login"; exit 1
fi
git rev-parse --is-inside-work-tree >/dev/null 2>&1 || git init -b main
git add -A
git diff --cached --quiet || git commit -m "swarmtrader: ranking a 1 mes con probabilidades calibradas y torneo evolutivo con control de overfitting"
gh repo create "$NAME" "$VIS" --source=. --remote=origin --push
echo
echo "publicado: https://github.com/$(gh api user -q .login)/$NAME"
