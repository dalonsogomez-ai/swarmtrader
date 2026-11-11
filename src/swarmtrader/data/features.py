"""Construccion del panel de factores cross-seccional.

Convencion critica: toda feature en la fila de fecha t se calcula EXCLUSIVAMENTE
con informacion disponible al cierre de t. El target vive en columna aparte y se
desplaza hacia adelante. Cualquier violacion de esto es look-ahead bias y
convierte el backtest en ficcion.

Cada factor lleva su signo orientado para que "mayor = mejor esperado" segun la
literatura. El genoma puede invertirlo con un peso negativo si el mercado dice
lo contrario.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

FACTOR_NAMES: tuple[str, ...] = (
    "mom_12_1",       # momentum 12m excluyendo el ultimo mes (Jegadeesh-Titman)
    "mom_6_1",        # momentum 6m excluyendo el ultimo mes
    "reversal_21",    # reversion a 1 mes (signo invertido)
    "reversal_5",     # reversion a 1 semana (signo invertido)
    "low_vol_60",     # baja volatilidad realizada (signo invertido)
    "low_beta_60",    # baja beta de mercado (signo invertido; anomalia BAB)
    "anti_lottery",   # -max retorno diario 21d (Bali-Cakici-Whitelaw 2011)
    "dist_52w_high",  # cercania al maximo de 52 semanas
    "vol_trend",      # volumen 21d relativo a 126d
    "illiq_amihud",   # iliquidez de Amihud (signo invertido)
    "accel",          # aceleracion: mom 1m menos mom 3m normalizado
    "vol_price_div",  # correlacion retorno~cambio de volumen a 21d
    "seasonal_12m",   # retorno del mismo mes hace un ano (Heston-Sadka 2008)
    "idio_skew_63",   # -asimetria de retornos 63d (Boyer-Mitton-Vorkink 2010)
)

REGIME_COLUMNS: tuple[str, ...] = ("mkt_vol_21", "mkt_ret_63", "mkt_vol_z")


def _zscore_rows(df: pd.DataFrame, limit: float = 3.0) -> pd.DataFrame:
    """Z-score cross-seccional (por fila = por fecha) vectorizado, con winsorizacion.

    Los NaN se ignoran en media y desviacion; las filas sin dispersion quedan a 0.
    """
    x = df.to_numpy(dtype=float)
    with np.errstate(all="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        mu = np.nanmean(x, axis=1, keepdims=True)
        sd = np.nanstd(x, axis=1, keepdims=True)
    sd = np.where(np.isfinite(sd) & (sd > 1e-12), sd, np.nan)
    z = np.clip((x - mu) / sd, -limit, limit)
    z = np.where(np.isfinite(z), z, np.nan)
    return pd.DataFrame(z, index=df.index, columns=df.columns)


def _rolling_beta(ret: pd.DataFrame, mkt: pd.Series, window: int) -> pd.DataFrame:
    cov = ret.rolling(window, min_periods=window // 2).cov(mkt)
    var = mkt.rolling(window, min_periods=window // 2).var()
    return cov.div(var.replace(0, np.nan), axis=0)


def _neutralize(df: pd.DataFrame, groups: pd.Series | None) -> pd.DataFrame:
    """Resta la media del grupo (sector) por fecha. Elimina el componente sectorial
    del factor para que el ranking sea intra-sector (menos ruido, menos apuestas
    macro implicitas). Simbolos sin grupo se neutralizan contra el conjunto."""
    if groups is None:
        return df
    g = groups.reindex(df.columns).fillna("_none_")
    out = df.copy()
    for name, cols in g.groupby(g).groups.items():
        cols = list(cols)
        sub = df[cols]
        out.loc[:, cols] = sub.sub(sub.mean(axis=1), axis=0).to_numpy()
    return out


def compute_features(prices: pd.DataFrame, volumes: pd.DataFrame,
                     extra: pd.DataFrame | None = None,
                     groups: pd.Series | None = None,
                     neutralize: bool = False) -> pd.DataFrame:
    """Panel MultiIndex (date, symbol) x factores, z-scoreado por fecha.

    extra     : panel MultiIndex (date, symbol) con senales externas ya alineadas
                y con lag point-in-time aplicado (ver data/signals.py). Se
                z-scorean por fecha igual que los factores de precio.
    groups    : Series symbol -> sector. Anade `industry_mom_6` (Moskowitz-
                Grinblatt 1999) y permite neutralizacion sectorial.
    neutralize: si True, todos los factores se demean por sector y fecha.
    """
    px = prices.sort_index()
    vol = volumes.reindex_like(px)
    ret1 = px.pct_change()
    mkt = ret1.mean(axis=1)

    raw: dict[str, pd.DataFrame] = {}
    raw["mom_12_1"] = px.shift(21) / px.shift(252) - 1.0
    raw["mom_6_1"] = px.shift(21) / px.shift(126) - 1.0
    raw["reversal_21"] = -(px / px.shift(21) - 1.0)
    raw["reversal_5"] = -(px / px.shift(5) - 1.0)
    raw["low_vol_60"] = -ret1.rolling(60, min_periods=30).std()
    raw["low_beta_60"] = -_rolling_beta(ret1, mkt, 60)
    raw["anti_lottery"] = -ret1.rolling(21, min_periods=10).max()
    raw["dist_52w_high"] = px / px.rolling(252, min_periods=120).max() - 1.0
    raw["vol_trend"] = vol.rolling(21, min_periods=10).mean() / vol.rolling(126, min_periods=60).mean() - 1.0
    dollar_vol = (px * vol).rolling(21, min_periods=10).mean()
    raw["illiq_amihud"] = -(ret1.abs().rolling(21, min_periods=10).mean() / dollar_vol.replace(0, np.nan))
    raw["accel"] = (px / px.shift(21) - 1.0) - (px / px.shift(63) - 1.0) / 3.0
    dvol = vol.pct_change().replace([np.inf, -np.inf], np.nan)
    raw["vol_price_div"] = ret1.rolling(21, min_periods=10).corr(dvol)
    raw["seasonal_12m"] = px.shift(231) / px.shift(252) - 1.0
    raw["idio_skew_63"] = -(ret1.sub(mkt, axis=0)).rolling(63, min_periods=40).skew()

    names = list(FACTOR_NAMES)
    if groups is not None:
        g = groups.reindex(px.columns)
        m6 = raw["mom_6_1"]
        ind = pd.DataFrame(np.nan, index=m6.index, columns=m6.columns)
        for sec, cols in g.dropna().groupby(g.dropna()).groups.items():
            cols = [c for c in cols if c in m6.columns]
            if cols:
                sec_mean = m6[cols].mean(axis=1)
                for c in cols:
                    ind[c] = sec_mean
        raw["industry_mom_6"] = ind
        names.append("industry_mom_6")

    frames = []
    for name in names:
        df = raw[name].replace([np.inf, -np.inf], np.nan)
        if neutralize and name != "industry_mom_6":
            df = _neutralize(df, groups)
        frames.append(_zscore_rows(df).stack().rename(name))

    if extra is not None and not extra.empty:
        for name in extra.columns:
            wide = extra[name].unstack("symbol").reindex(index=px.index, columns=px.columns)
            if neutralize:
                wide = _neutralize(wide, groups)
            frames.append(_zscore_rows(wide).stack().rename(name))

    panel = pd.concat(frames, axis=1)
    panel.index.names = ["date", "symbol"]
    return panel.dropna(how="all").fillna(0.0)


def compute_regime(prices: pd.DataFrame) -> pd.DataFrame:
    """Series de regimen de mercado (una fila por fecha). Solo pasado."""
    ret1 = prices.sort_index().pct_change()
    mkt = ret1.mean(axis=1)
    vol21 = mkt.rolling(21, min_periods=10).std() * np.sqrt(252)
    vol_hist_mu = vol21.rolling(252, min_periods=60).mean()
    vol_hist_sd = vol21.rolling(252, min_periods=60).std()
    out = pd.DataFrame({
        "mkt_vol_21": vol21,
        "mkt_ret_63": (1.0 + mkt).rolling(63, min_periods=30).apply(np.prod, raw=True) - 1.0,
        "mkt_vol_z": ((vol21 - vol_hist_mu) / vol_hist_sd.replace(0, np.nan)).clip(-4, 4),
    }, index=prices.index)
    return out.fillna(0.0)


def forward_returns(prices: pd.DataFrame, horizon: int) -> pd.Series:
    """Retorno futuro a `horizon` sesiones. SOLO para etiquetar, nunca feature."""
    fwd = prices.shift(-horizon) / prices - 1.0
    s = fwd.stack()
    s.index.names = ["date", "symbol"]
    return s.rename(f"fwd_{horizon}")


def dollar_volume(prices: pd.DataFrame, volumes: pd.DataFrame, window: int = 21) -> pd.Series:
    s = (prices * volumes.reindex_like(prices)).rolling(window, min_periods=5).mean().stack()
    s.index.names = ["date", "symbol"]
    return s.rename("dollar_volume")
