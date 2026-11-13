"""Adaptadores de plataforma: de donde sale el universo y el historico.

Cuatro entradas, misma salida (`MarketData`), para que el resto del sistema no
sepa ni le importe de donde vienen los datos:

  * **CSV** (Lightyear, eToro, IBKR, Robinhood: todos exportan CSV).
  * **Alpaca** — recomendado. Datos y ordenes en la misma API, plan gratuito con
    datos historicos IEX, y `paper=True` para operar en simulado con la MISMA
    ruta de codigo que en real. Es la unica de las tres con API de datos
    historicos oficial y documentada.
  * **eToro** — solo watchlist via Public API (ya existia en `data/universe.py`).
  * **yfinance** — por defecto para el historico cuando no hay broker.

Aviso sobre el sesgo de supervivencia, que aplica a TODAS estas fuentes salvo a
un CSV historico propio: la lista de activos que devuelven es la de los que
existen HOY. Backtestear seleccion sobre ese universo infla el resultado porque
las empresas que quebraron o fueron excluidas no aparecen. Sirve para rankear tu
lista actual; no sirve para concluir que el sistema habria funcionado.
"""

from __future__ import annotations

from pathlib import Path

import os
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .loaders import MarketData

__all__ = ["AlpacaConfig", "alpaca_universe", "alpaca_market_data", "load_from_broker"]


@dataclass
class AlpacaConfig:
    key_id: str | None = None
    secret_key: str | None = None
    paper: bool = True
    feed: str = "iex"               # iex (gratis) | sip (requiere plan de datos)
    years: float = 4.0
    min_price: float = 1.0
    max_symbols: int | None = None

    def resolved(self) -> tuple[str, str]:
        # Se aceptan los dos nombres porque la documentacion de Alpaca y varios
        # SDK usan convenciones distintas, y tener dos modulos leyendo nombres
        # diferentes significaba que exportabas uno y el otro fallaba.
        # Canonico: ALPACA_API_KEY / ALPACA_SECRET_KEY (el que usa execution/).
        k = (self.key_id or os.environ.get("ALPACA_API_KEY")
             or os.environ.get("ALPACA_API_KEY_ID"))
        s = (self.secret_key or os.environ.get("ALPACA_SECRET_KEY")
             or os.environ.get("ALPACA_API_SECRET_KEY"))
        if not k or not s:
            raise RuntimeError(
                "faltan credenciales de Alpaca. Exporta ALPACA_API_KEY y "
                "ALPACA_SECRET_KEY (o los alias ALPACA_API_KEY_ID / "
                "ALPACA_API_SECRET_KEY), o pasalas en AlpacaConfig."
            )
        return k, s


def alpaca_universe(cfg: AlpacaConfig = AlpacaConfig()) -> list[str]:
    """Todos los activos US equity negociables y fraccionables.

    Se filtran ADRs raros, unidades y warrants por sufijo, que son ruido para un
    ranker cross-seccional y suelen tener historico roto.
    """
    try:
        from alpaca.trading.client import TradingClient
        from alpaca.trading.enums import AssetClass, AssetStatus
        from alpaca.trading.requests import GetAssetsRequest
    except ImportError as exc:  # pragma: no cover
        raise ImportError("falta alpaca-py: pip install -e '.[broker]'") from exc

    k, s = cfg.resolved()
    client = TradingClient(k, s, paper=cfg.paper)
    assets = client.get_all_assets(GetAssetsRequest(status=AssetStatus.ACTIVE,
                                                    asset_class=AssetClass.US_EQUITY))
    syms = [
        a.symbol for a in assets
        if a.tradable and getattr(a, "fractionable", False)
        and "." not in a.symbol and len(a.symbol) <= 5
        and not a.symbol.endswith(("W", "U", "R"))
    ]
    syms = sorted(set(syms))
    if cfg.max_symbols:
        syms = syms[: cfg.max_symbols]
    return syms


def alpaca_market_data(symbols: list[str], cfg: AlpacaConfig = AlpacaConfig()) -> MarketData:
    """Barras diarias ajustadas por splits. Devuelve precios y volumenes wide."""
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
    except ImportError as exc:  # pragma: no cover
        raise ImportError("falta alpaca-py: pip install -e '.[broker]'") from exc

    k, s = cfg.resolved()
    client = StockHistoricalDataClient(k, s)
    end = pd.Timestamp.utcnow().normalize().tz_localize(None)
    start = end - pd.Timedelta(days=int(365 * cfg.years))

    frames = []
    CHUNK = 200          # el endpoint pagina; trocear evita timeouts y 429
    for i in range(0, len(symbols), CHUNK):
        chunk = symbols[i:i + CHUNK]
        req = StockBarsRequest(symbol_or_symbols=chunk, timeframe=TimeFrame.Day,
                               start=start, end=end, adjustment="split", feed=cfg.feed)
        bars = client.get_stock_bars(req).df
        if bars is None or bars.empty:
            continue
        frames.append(bars.reset_index()[["symbol", "timestamp", "close", "volume"]])

    if not frames:
        raise RuntimeError("Alpaca no devolvio barras: revisa credenciales, feed y fechas")

    long = pd.concat(frames, ignore_index=True)
    long["date"] = pd.to_datetime(long["timestamp"]).dt.tz_localize(None).dt.normalize()
    px = long.pivot_table(index="date", columns="symbol", values="close", aggfunc="last")
    vol = long.pivot_table(index="date", columns="symbol", values="volume", aggfunc="sum")

    px = px.sort_index().ffill(limit=5)
    keep = (px.notna().mean() > 0.9) & (px.iloc[-1] >= cfg.min_price)
    px, vol = px.loc[:, keep], vol.reindex(columns=px.columns[keep]).sort_index()
    return MarketData(prices=px, volumes=vol.fillna(0.0))


def alpaca_sectors(symbols: list[str]) -> pd.Series:
    """Alpaca no expone sector; se cae a yfinance, que si lo hace."""
    from .signals import groups_from_yfinance
    return groups_from_yfinance(symbols)


def load_from_broker(source: str, path: str | None = None, years: float = 4.0,
                     max_symbols: int | None = None, **kw) -> tuple[MarketData, list[str]]:
    """Punto de entrada unico. `source` en {csv, alpaca, etoro, yfinance}."""
    from .universe import etoro_watchlist_symbols, load_symbols_yfinance, load_universe

    src = source.lower()
    if src == "alpaca":
        cfg = AlpacaConfig(years=years, max_symbols=max_symbols, **kw)
        syms = alpaca_universe(cfg)
        return alpaca_market_data(syms, cfg), syms
    if src == "etoro":
        syms = etoro_watchlist_symbols()
        if max_symbols:
            syms = syms[:max_symbols]
        return load_symbols_yfinance(syms, years=years), syms
    if src in ("csv", "yfinance"):
        if not path:
            raise ValueError("se requiere una ruta de CSV para source=csv")
        md = load_universe(path, years=years)
        return md, list(md.prices.columns)
    raise ValueError(f"fuente desconocida: {source}")


# =====================================================================
# eToro e Interactive Brokers (v0.7)
# =====================================================================
def etoro_candles(instrument_ids: list[int], interval: str = "OneDay", count: int = 1000,
                  api_key: str | None = None, session: object | None = None) -> pd.DataFrame:
    """Velas diarias de la API publica de eToro, respetando su ritmo.

    Correccion de una premisa previa: eToro SI expone market data publica, no
    solo watchlist. El endpoint de velas admite como maximo 1.000 por peticion,
    asi que cuatro anos de diarias (~1.008 sesiones) NO caben en una sola
    llamada y hay que paginar. El limitador aplica sus 60 peticiones/60 s.
    """
    import os

    import requests

    from .pacing import ETORO_LIMITS, RateLimiter

    key = api_key or os.environ.get("ETORO_API_KEY")
    if not key:
        raise RuntimeError("falta ETORO_API_KEY")
    sess = session or requests.Session()
    base = "https://public-api.etoro.com/api/v1/market-data/instruments"
    rl = RateLimiter(ETORO_LIMITS)
    n = min(int(count), ETORO_LIMITS.max_bars_per_request or 1000)

    frames = []
    for iid in instrument_ids:
        rl.acquire(key=str(iid))
        r = sess.get(f"{base}/{iid}/history/candles/desc/{interval}/{n}",
                     headers={"X-API-KEY": key, "Accept": "application/json"}, timeout=30)
        r.raise_for_status()
        payload = r.json()
        rows = payload.get("candles", payload) if isinstance(payload, dict) else payload
        if not rows:
            continue
        df = pd.DataFrame(rows)
        df["instrument_id"] = iid
        frames.append(df)
    if not frames:
        raise RuntimeError("eToro no devolvio velas")
    return pd.concat(frames, ignore_index=True)


def ibkr_connect(host: str = "127.0.0.1", port: int = 7497, client_id: int = 17):
    """Conecta con TWS/IB Gateway usando `ib_async`.

    `ib_insync` quedo archivado en marzo de 2024 tras el fallecimiento de su
    autor; el sucesor comunitario mantenido es `ib_async`, compatible a nivel de
    API. Puerto 7497 = TWS paper, 7496 = TWS real, 4002 = Gateway paper.
    """
    try:
        from ib_async import IB
    except ImportError as exc:  # pragma: no cover
        raise ImportError("falta ib_async: pip install -e '.[ibkr]'") from exc
    ib = IB()
    ib.connect(host, port, clientId=client_id)
    return ib


def ibkr_market_data(symbols: list[str], years: float = 4.0, ib=None,
                     exchange: str = "SMART", currency: str = "USD",
                     max_hours: float = 2.0, force: bool = False) -> MarketData:
    """Barras diarias desde IBKR, con control de ritmo y NEGATIVA razonada.

    IBKR no es una fuente de ingesta masiva y el sistema lo dice ANTES de
    empezar en vez de morir a mitad: a 60 solicitudes historicas por cada 10
    minutos, 5.000 simbolos son ~13,9 horas. Para el historico usa un proveedor
    point-in-time; reserva IBKR para la ejecucion y para universos pequenos.
    """
    from .pacing import IBKR_LIMITS, RateLimiter, assess_bulk_download

    n_sessions = int(252 * years)
    report = assess_bulk_download(IBKR_LIMITS, len(symbols), n_sessions, max_hours=max_hours)
    if not report.feasible and not force:
        raise RuntimeError(str(report) + "\n    (pasa force=True si aun asi quieres intentarlo)")

    try:
        from ib_async import Stock
    except ImportError as exc:  # pragma: no cover
        raise ImportError("falta ib_async: pip install -e '.[ibkr]'") from exc

    own = ib is None
    ib = ib or ibkr_connect()
    rl = RateLimiter(IBKR_LIMITS)
    frames = []
    try:
        for sym in symbols:
            rl.acquire(key=f"{sym}|{exchange}|TRADES")
            bars = ib.reqHistoricalData(
                Stock(sym, exchange, currency), endDateTime="",
                durationStr=f"{max(int(years), 1)} Y", barSizeSetting="1 day",
                whatToShow="TRADES", useRTH=True, formatDate=1)
            if not bars:
                continue
            df = pd.DataFrame([{"date": b.date, "close": b.close, "volume": b.volume}
                               for b in bars])
            df["symbol"] = sym
            frames.append(df)
    finally:
        if own:
            ib.disconnect()

    if not frames:
        raise RuntimeError("IBKR no devolvio barras: revisa suscripciones de datos y permisos")
    long = pd.concat(frames, ignore_index=True)
    long["date"] = pd.to_datetime(long["date"])
    px = long.pivot_table(index="date", columns="symbol", values="close", aggfunc="last")
    vol = long.pivot_table(index="date", columns="symbol", values="volume", aggfunc="sum")
    return MarketData(prices=px.sort_index(), volumes=vol.reindex_like(px).fillna(0.0))


def sharadar_market_data(path: str | Path, min_price: float | None = None) -> tuple[MarketData, pd.DataFrame]:
    """Carga un volcado de Sharadar SEP (o equivalente) y construye el panel PIT.

    Se espera el formato tabular estandar del volcado masivo:
        ticker,date,open,high,low,close,volume,closeadj,closeunadj,lastupdated
    Se usa `closeadj` (ajustado por splits y dividendos) si esta presente.

    Por que un volcado y no llamadas simbolo a simbolo: el historico completo se
    baja de una vez, lo que elimina por completo el problema de ritmo que hace
    inviables IBKR y eToro para esto. Ademas trae los deslistados, que es la
    razon de ser de este camino: las fechas de alta y baja se derivan del propio
    panel y alimentan `PITPanel`.

    No se filtra NINGUN simbolo aqui. Hasta la v1.2 este adaptador descartaba los
    valores cuyo ULTIMO precio quedaba por debajo de `min_price`: eso eliminaba
    del historico entero justo a los que acabaron en centimos, es decir,
    reintroducia el sesgo de supervivencia que este camino existe para quitar.
    Los filtros de universo se aplican fecha a fecha en `RankConfig`
    (`min_price`, `min_dollar_volume`), y usan `closeunadj` como precio nominal
    si el volcado lo trae.
    """
    if min_price:
        raise ValueError(
            "sharadar_market_data ya no filtra por precio: hacerlo con el ultimo precio "
            "elimina del historico a los valores que quebraron. Usa RankConfig(min_price=...), "
            "que se aplica fecha a fecha.")
    from .pit import DEFAULT_DELIST_RETURN, DelistReason, PITPanel

    df = pd.read_csv(path)
    df.columns = [c.lower().strip() for c in df.columns]
    sym_col = "ticker" if "ticker" in df.columns else "symbol"
    close_col = "closeadj" if "closeadj" in df.columns else "close"
    df["date"] = pd.to_datetime(df["date"])
    df = df[df[close_col] > 0]

    px = df.pivot_table(index="date", columns=sym_col, values=close_col, aggfunc="last").sort_index()
    vol = df.pivot_table(index="date", columns=sym_col, values="volume",
                         aggfunc="sum").reindex_like(px)

    last_session = px.index.max()
    rows = []
    for sym in px.columns:
        s = px[sym].dropna()
        if s.empty:
            continue
        first, last = s.index.min(), s.index.max()
        # si la serie termina mucho antes que el panel, el valor dejo de cotizar
        active = (last_session - last).days <= 10
        reason = DelistReason.ACTIVE if active else DelistReason.DROPPED
        rows.append((sym, first, last, reason, DEFAULT_DELIST_RETURN[reason]))
    listings = pd.DataFrame(rows, columns=["symbol", "first_date", "last_date",
                                           "reason", "delist_return"]).set_index("symbol")
    panel = PITPanel(px[listings.index], vol[listings.index], listings)
    md = panel.to_market_data()
    if "closeunadj" in df.columns:
        nominal = df.pivot_table(index="date", columns=sym_col, values="closeunadj", aggfunc="last")
        md = MarketData(md.prices, md.volumes, nominal)
    return md, panel.listings
