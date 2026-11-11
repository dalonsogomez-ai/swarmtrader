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
