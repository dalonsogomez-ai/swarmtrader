# Ejemplos

- `universe_tickers.csv` — lista de tickers reales. `swarmtrader rank examples/universe_tickers.csv` descarga el histórico con yfinance.
- `history_long_example.csv` / `history_wide_example.csv` — histórico SINTÉTICO con momentum inyectado; funciona sin red.
- `news_example.csv`, `analyst_example.csv`, `earnings_example.csv`, `groups_example.csv` — señales externas SINTÉTICAS
  construidas con una relación artificial con el retorno futuro para verificar que el pipeline point-in-time funciona.
  **No representan la fuerza real de esas señales**: en mercados reales la mejora de IC por señal se mide en centésimas.

```bash
swarmtrader rank examples/history_long_example.csv --top 3
swarmtrader rank examples/history_long_example.csv --top 3 --news examples/news_example.csv \
  --analyst examples/analyst_example.csv --earnings examples/earnings_example.csv \
  --groups examples/groups_example.csv --neutralize
```
