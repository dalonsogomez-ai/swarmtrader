#!/usr/bin/env bash
# Ranking completo sobre el S&P 500 (disponible en eToro y Lightyear) con datos reales.
# Requiere red. Tarda 3-6 min la primera vez (descarga ~4 anos de historico via yfinance).
set -euo pipefail
cd "$(dirname "$0")/.."
[ -d .venv ] || make install
source .venv/bin/activate
UNI="${1:-examples/sp500_universe.csv}"
TOP="${2:-5}"
mkdir -p runs
swarmtrader rank "$UNI" --groups examples/sp500_sectors.csv --neutralize \
  --top "$TOP" --model ensemble --years 4 --min-dollar-volume 5e6 \
  --out runs/rank_$(date +%F).json | tee runs/rank_$(date +%F).txt
echo
echo "resultado guardado en runs/rank_$(date +%F).json  (pegalo en el chat para interpretarlo)"
