"""Carga de datos de mercado.

Tres backends:
  synthetic  -> generador con estructura factorial conocida. Para tests y para
                calibrar cuanta senal necesitas antes de creerte nada.
  parquet    -> tus datos locales (Sharadar, Polygon, EODHD...). RECOMENDADO
                para research serio: point-in-time y con delistadas.
  yfinance   -> rapido para prototipar. AVISO: tiene sesgo de supervivencia
                (solo devuelve tickers vivos hoy) y no es point-in-time.
                No tomes ninguna decision de capital con un backtest de yfinance.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class MarketData:
    prices: pd.DataFrame   # fecha x simbolo, ajustado por splits y dividendos
    volumes: pd.DataFrame
    # Cierre SIN ajustar (el precio que realmente cotizaba ese dia). Opcional.
    # Los filtros de precio lo usan si existe: el precio ajustado incorpora
    # splits FUTUROS, asi que un valor que cotizaba a 0,50 USD y despues hizo un
    # split inverso 1:20 aparece a 10 USD en el historico ajustado y pasa un
    # filtro de precio minimo que en su dia no habria pasado.
    nominal: pd.DataFrame | None = None

    def __post_init__(self) -> None:
        self.volumes = self.volumes.reindex_like(self.prices)
        if self.nominal is not None:
            self.nominal = self.nominal.reindex_like(self.prices)

    @property
    def symbols(self) -> list[str]:
        return list(self.prices.columns)

    def slice(self, start, end) -> "MarketData":
        return MarketData(self.prices.loc[start:end], self.volumes.loc[start:end],
                          None if self.nominal is None else self.nominal.loc[start:end])


def load_synthetic(
    n_symbols: int = 120,
    n_days: int = 1500,
    signal_strength: float = 0.03,
    momentum_strength: float = 0.0,
    seed: int = 7,
) -> MarketData:
    """Universo sintetico con edge REAL e inyectado y conocido.

    signal_strength   : factor LATENTE (no observable directamente) que predice
                        el retorno. Es deliberadamente dificil de capturar con
                        factores de precio: IC realista ~0.02-0.05.
    momentum_strength : efecto momentum OBSERVABLE: el retorno de manana incluye
                        una fraccion del retorno medio de los ultimos 63 dias
                        (estandarizado cross-seccionalmente). Sirve para probar
                        que el ranker detecta senal cuando la hay. 0.0 = ninguno.

    Ambos a 0.0 => mercado sin ninguna senal: nada deberia certificarse.
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2018-01-02", periods=n_days)
    symbols = [f"SYN{i:03d}" for i in range(n_symbols)]

    market = rng.normal(0.0003, 0.011, n_days)
    beta = rng.uniform(0.5, 1.6, n_symbols)
    idio_vol = rng.uniform(0.012, 0.035, n_symbols)

    rets = np.zeros((n_days, n_symbols))
    latent = np.zeros(n_symbols)
    for t in range(n_days):
        eps = rng.normal(0, 1, n_symbols) * idio_vol
        # el factor latente persiste y predice parcialmente el retorno de manana
        rets[t] = beta * market[t] + eps + signal_strength * latent * idio_vol
        if momentum_strength > 0 and t >= 63:
            past = rets[t - 63:t].mean(axis=0)
            z = (past - past.mean()) / (past.std() + 1e-12)
            rets[t] += momentum_strength * z * idio_vol
        latent = 0.97 * latent + 0.03 * rng.normal(0, 1, n_symbols)

    prices = pd.DataFrame(100.0 * np.exp(np.cumsum(rets, axis=0)), index=dates, columns=symbols)
    base_vol = rng.uniform(3e5, 8e6, n_symbols)
    volumes = pd.DataFrame(
        base_vol * np.exp(rng.normal(0, 0.35, (n_days, n_symbols))),
        index=dates, columns=symbols,
    )
    return MarketData(prices, volumes)


def load_parquet(directory: str | Path) -> MarketData:
    """Espera prices.parquet y volumes.parquet, wide (fecha x simbolo)."""
    d = Path(directory)
    prices = pd.read_parquet(d / "prices.parquet")
    volumes = pd.read_parquet(d / "volumes.parquet")
    prices.index = pd.to_datetime(prices.index)
    volumes.index = pd.to_datetime(volumes.index)
    return MarketData(prices.sort_index(), volumes.sort_index())


def load_yfinance(symbols: list[str], start: str, end: str | None = None) -> MarketData:
    """Prototipado rapido. Recuerda: sesgo de supervivencia, no point-in-time."""
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover
        raise ImportError("pip install yfinance") from exc

    symbols = list(dict.fromkeys(s.upper() for s in symbols))
    raw = yf.download(symbols, start=start, end=end, auto_adjust=True,
                      progress=False, group_by="column", threads=True)
    if raw is None or raw.empty:
        raise RuntimeError("yfinance no devolvio datos; revisa tickers y conexion")
    close, vol = raw["Close"], raw["Volume"]
    if isinstance(close, pd.Series):            # un solo ticker
        close, vol = close.to_frame(symbols[0]), vol.to_frame(symbols[0])
    prices = close.dropna(how="all").dropna(axis=1, how="all")
    prices = prices.loc[:, prices.notna().mean() > 0.5]      # descarta tickers casi vacios
    volumes = vol.reindex_like(prices).fillna(0.0)
    prices.index = pd.to_datetime(prices.index).tz_localize(None)
    volumes.index = prices.index
    prices.index.name = volumes.index.name = "date"
    return MarketData(prices.astype(float), volumes.astype(float))


_MAX_DAILY_GROSS = 20.0


def _sanitize_prices(px: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Limpieza en la frontera de entrada de un panel de precios ajustados.

    Los proveedores gratuitos (Yahoo) tienen errores que, sin limpiar, producen
    retornos infinitos o de millones por ciento y contaminan cualquier media:
      * precio ajustado <= 0 (ajustes por dividendos mal aplicados) -> NaN;
      * pico aislado: una sesion que multiplica por mas de 5 y la siguiente
        divide por mas de 4 (o al reves) es un tick erroneo -> NaN esa sesion;
      * si tras eso queda algun salto diario mayor que x20 en cualquier sentido,
        el ajuste de la serie es inservible y se descarta el simbolo entero.
    """
    px = px.where(px > 0)
    up = px / px.shift(1)
    nxt = px.shift(-1) / px
    spike = ((up > 5) & (nxt < 0.25)) | ((up < 0.2) & (nxt > 4))
    px = px.mask(spike)
    g = px / px.ffill().shift(1)
    bad = ((g > _MAX_DAILY_GROSS) | (g < 1.0 / _MAX_DAILY_GROSS)).any()
    dropped = list(bad.index[bad])
    return px.drop(columns=dropped), dropped


def load_long_panel(path: str | Path, min_dollar_volume: float = 0.0, min_price: float = 0.0,
                    liquidity_window: int = 60, start: str | None = None) -> MarketData:
    """Panel largo en Parquet: date, ticker, adjclose[, close_nominal][, volume | dollar_volume].

    `close_nominal` es el cierre SIN ajustar; si existe, rellena `MarketData.nominal`
    y los filtros de precio usan el precio que de verdad cotizaba.

    `min_dollar_volume` y `min_price` hacen un PRE-filtro con la ultima ventana de
    `liquidity_window` sesiones (mediana), solo para que el panel quepa en memoria
    en maquinas pequenas. Usa informacion de HOY para elegir el universo de todo el
    historico: es un sesgo de seleccion adicional al de supervivencia, y por eso
    queda desactivado por defecto. Los filtros point-in-time del ranker se siguen
    aplicando fecha a fecha.
    """
    p = pd.read_parquet(path)
    p = p[p["ticker"].notna()]
    p["ticker"] = p["ticker"].astype(str)
    p["date"] = pd.to_datetime(p["date"]).dt.tz_localize(None)
    if start:
        p = p[p["date"] >= pd.Timestamp(start)]
    px = p.pivot(index="date", columns="ticker", values="adjclose").sort_index()
    nom = (p.pivot(index="date", columns="ticker", values="close_nominal").reindex_like(px)
           if "close_nominal" in p.columns else None)
    if "volume" in p.columns:
        vol = p.pivot(index="date", columns="ticker", values="volume").reindex_like(px)
    else:
        dv = p.pivot(index="date", columns="ticker", values="dollar_volume").reindex_like(px)
        vol = dv / (nom if nom is not None else px)
    px, dropped = _sanitize_prices(px)
    if dropped:
        import warnings
        warnings.warn(f"{len(dropped)} simbolos descartados por precios corruptos "
                      f"(no positivos o saltos de mas de x{_MAX_DAILY_GROSS:.0f} en una "
                      f"sesion): {', '.join(sorted(dropped)[:20])}"
                      + (" ..." if len(dropped) > 20 else ""), stacklevel=2)
    if nom is not None:
        nom = nom.where(nom > 0)
    if min_dollar_volume > 0 or min_price > 0:
        ref = nom if nom is not None else px
        recent_dv = (ref * vol).tail(liquidity_window).median()
        recent_px = ref.tail(liquidity_window).median()
        keep = recent_dv.index[(recent_dv >= min_dollar_volume) & (recent_px >= min_price)]
        keep = [k for k in keep if k in px.columns]
        px, vol = px[keep], vol[keep]
        nom = nom[keep] if nom is not None else None
    px = px.loc[:, px.notna().sum() > 0]
    vol = vol.reindex_like(px).fillna(0.0)
    nom = nom.reindex_like(px) if nom is not None else None
    px.index.name = vol.index.name = "date"
    return MarketData(px.astype(float), vol.astype(float),
                      None if nom is None else nom.astype(float))
