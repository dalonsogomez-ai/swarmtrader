#!/usr/bin/env bash
# Ciclo diario de paper trading. Programar tras el cierre (p.ej. 22:30 CET).
# launchd: ~/Library/LaunchAgents/com.swarmtrader.paper.plist -> ProgramArguments este script
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export ALPACA_API_KEY="${ALPACA_API_KEY:?}"
export ALPACA_SECRET_KEY="${ALPACA_SECRET_KEY:?}"
swarmtrader paper --broker alpaca --data parquet --path data/ \
  --champion runs/latest/champion.json --config config/default.toml \
  --log runs/live.jsonl >> runs/paper_daily.log 2>&1
