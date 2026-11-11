"""Carga de universos desde CSV, ficheros de historico o APIs de broker.

Formatos aceptados (autodetectados):

  1. Lista de tickers        columna `symbol` / `ticker` / `Symbol`, o un ticker
                             por linea sin cabecera. El historico se descarga con
                             yfinance. Es el formato que exporta cualquier broker
                             (eToro, Lightyear, IBKR, Robinhood: posiciones o
                             watchlist).
  2. Historico largo (long)  columnas date, symbol, close [, volume, open, high, low]
  3. Historico ancho (wide)  columna date + una columna por simbolo con el cierre;
                             opcionalmente un segundo fichero con volumenes.

Los formatos 2 y 3 funcionan sin red. Los datos de yfinance tienen sesgo de
supervivencia (solo tickers vivos hoy): valen para rankear el universo que TU
ya tienes, no para backtests historicos de seleccion.
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .loaders import MarketData

_SYMBOL_COLS = ("symbol", "ticker", "Symbol", "Ticker", "SYMBOL", "TICKER", "Instrument", "instrument")
_CLOSE_COLS = ("close", "Close", "adj_close", "Adj Close", "adjclose", "price", "Price")
_VOL_COLS = ("volume", "Volume", "vol")
_DATE_COLS = ("date", "Date", "timestamp", "Timestamp", "time")


def _first(cols, candidates):
    for c in candidates:
        if c in cols:
            return c
    return None


def read_symbols(path: str | Path) -> list[str]:
    """Extrae la lista de tickers de un CSV (con o sin cabecera) o de un .txt."""
    p = Path(path)
    text = p.read_text(encoding="utf-8-sig")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return []
    try:
        df = pd.read_csv(p, encoding="utf-8-sig")
        col = _first(df.columns, _SYMBOL_COLS)
        if col is not None:
            syms = df[col].astype(str).str.strip().str.upper().tolist()
            return sorted({s for s in syms if s and s != "NAN"})
    except Exception:  # noqa: BLE001
        pass
    # sin cabecera reconocible: un ticker por linea (o separados por comas)
    syms = []
    for ln in lines:
        for tok in ln.replace(";", ",").split(","):
            tok = tok.strip().upper()
            if tok and tok.isascii() and len(tok) <= 12 and tok not in ("SYMBOL", "TICKER"):
                syms.append(tok)
    return sorted(set(syms))


def detect_format(path: str | Path) -> str:
    df = pd.read_csv(path, nrows=5, encoding="utf-8-sig")
    cols = list(df.columns)
    has_date = _first(cols, _DATE_COLS) is not None
    has_sym = _first(cols, _SYMBOL_COLS) is not None
    has_close = _first(cols, _CLOSE_COLS) is not None
    if has_date and has_sym and has_close:
        return "long"
    if has_date and len(cols) > 2 and not has_sym:
        return "wide"
    return "symbols"


def load_long_csv(path: str | Path) -> MarketData:
    df = pd.read_csv(path, encoding="utf-8-sig")
    d, s, c = _first(df.columns, _DATE_COLS), _first(df.columns, _SYMBOL_COLS), _first(df.columns, _CLOSE_COLS)
    v = _first(df.columns, _VOL_COLS)
    df[d] = pd.to_datetime(df[d])
    df[s] = df[s].astype(str).str.upper()
    prices = df.pivot_table(index=d, columns=s, values=c, aggfunc="last").sort_index()
    volumes = (df.pivot_table(index=d, columns=s, values=v, aggfunc="last").sort_index()
               if v else pd.DataFrame(1e6, index=prices.index, columns=prices.columns))
    prices.index.name, volumes.index.name = "date", "date"
    return MarketData(prices.astype(float), volumes.astype(float))


def load_wide_csv(path: str | Path, volumes_path: str | Path | None = None) -> MarketData:
    df = pd.read_csv(path, encoding="utf-8-sig")
    d = _first(df.columns, _DATE_COLS)
    df[d] = pd.to_datetime(df[d])
    prices = df.set_index(d).sort_index().astype(float)
    prices.columns = [str(c).upper() for c in prices.columns]
    if volumes_path:
        vdf = pd.read_csv(volumes_path, encoding="utf-8-sig")
        vd = _first(vdf.columns, _DATE_COLS)
        vdf[vd] = pd.to_datetime(vdf[vd])
        volumes = vdf.set_index(vd).sort_index().astype(float)
        volumes.columns = [str(c).upper() for c in volumes.columns]
    else:
        volumes = pd.DataFrame(1e6, index=prices.index, columns=prices.columns)
    prices.index.name, volumes.index.name = "date", "date"
    return MarketData(prices, volumes)


def load_symbols_yfinance(symbols: list[str], years: float = 4.0) -> MarketData:
    from .loaders import load_yfinance
    start = (pd.Timestamp.today() - pd.Timedelta(days=int(365 * years))).strftime("%Y-%m-%d")
    md = load_yfinance(symbols, start=start)
    missing = sorted(set(symbols) - set(md.symbols))
    if missing:
        print(f"aviso: sin datos para {len(missing)} simbolos: {', '.join(missing[:10])}{'...' if len(missing) > 10 else ''}")
    return md


def load_universe(path: str | Path, volumes_path: str | Path | None = None,
                  years: float = 4.0) -> MarketData:
    """Punto de entrada unico: detecta el formato y devuelve MarketData."""
    p = Path(path)
    if p.suffix.lower() == ".parquet":
        from .loaders import load_parquet
        return load_parquet(p.parent) if p.name == "prices.parquet" else load_parquet(p)
    if p.suffix.lower() in (".txt", ".lst"):
        return load_symbols_yfinance(read_symbols(p), years)
    fmt = detect_format(p)
    if fmt == "long":
        return load_long_csv(p)
    if fmt == "wide":
        return load_wide_csv(p, volumes_path)
    return load_symbols_yfinance(read_symbols(p), years)


# ------------------------------------------------------------------ brokers
def etoro_watchlist_symbols(api_key: str | None = None, user_key: str | None = None,
                            endpoint: str | None = None, symbol_field: str = "symbol") -> list[str]:
    """Lee los simbolos de una watchlist de eToro via su Public API.

    Autenticacion verificada: cabeceras `x-api-key` y `x-user-key`
    (api-portal.etoro.com; requiere cuenta verificada). La RUTA exacta del
    endpoint de watchlists y el nombre del campo de simbolo debes confirmarlos
    en la documentacion vigente y pasarlos aqui (o via ETORO_WATCHLIST_URL /
    ETORO_SYMBOL_FIELD). No se inventa ninguna ruta.

    El historico de precios se descarga despues con yfinance: los candles de
    eToro no tienen la profundidad ni la calidad necesarias para rankear.
    """
    import urllib.request
    api_key = api_key or os.environ.get("ETORO_API_KEY")
    user_key = user_key or os.environ.get("ETORO_USER_KEY")
    endpoint = endpoint or os.environ.get("ETORO_WATCHLIST_URL")
    symbol_field = os.environ.get("ETORO_SYMBOL_FIELD", symbol_field)
    if not (api_key and user_key and endpoint):
        raise RuntimeError("faltan ETORO_API_KEY, ETORO_USER_KEY o ETORO_WATCHLIST_URL")
    req = urllib.request.Request(endpoint, headers={"x-api-key": api_key, "x-user-key": user_key,
                                                    "accept": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
        data = json.loads(resp.read().decode())
    items = data if isinstance(data, list) else next((v for v in data.values() if isinstance(v, list)), [])
    syms = []
    for it in items:
        if isinstance(it, dict):
            v = it.get(symbol_field) or it.get("symbolFull") or it.get("ticker")
            if v:
                syms.append(str(v).upper().split(".")[0])
    return sorted(set(syms))


def generic_rest_symbols(url: str, headers: dict[str, str], symbol_field: str) -> list[str]:
    """Cualquier API REST que devuelva JSON con una lista de objetos con simbolo."""
    import urllib.request
    req = urllib.request.Request(url, headers={**headers, "accept": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
        data = json.loads(resp.read().decode())
    items = data if isinstance(data, list) else next((v for v in data.values() if isinstance(v, list)), [])
    return sorted({str(it[symbol_field]).upper() for it in items if isinstance(it, dict) and it.get(symbol_field)})
