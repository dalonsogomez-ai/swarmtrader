"""Panel point-in-time con delistings.

Por que este modulo es la pieza mas importante de la v0.7
---------------------------------------------------------
Hasta la v0.6 el panel eran los simbolos que existen HOY. Eso inyecta sesgo de
supervivencia por DOS vias, y la segunda es la que casi todo el mundo se deja:

  1. **Via obvia — el universo.** Una empresa que quebro en 2023 no aparece en
     el panel, asi que el backtest solo ve a los que sobrevivieron. Conocido y
     documentado.

  2. **Via silenciosa — la ETIQUETA.** `forward_returns` calcula
     `precio[t+h] / precio[t] - 1`. Si la accion deja de cotizar en t+j con
     j <= h, el precio en t+h es NaN, el target es NaN y esa observacion
     **se cae del entrenamiento y de la evaluacion**. Resultado: el modelo
     nunca ve como se comporta una accion justo antes de quebrar, y
     `precision@k` se mide solo sobre los que llegaron vivos al final. El
     sesgo vuelve a entrar por la puerta de atras aunque el universo sea
     correcto.

Este modulo cierra las dos. La clave de la segunda es el **retorno de
delisting**: cuando una accion sale por causa (quiebra, exclusion), el inversor
no vende al ultimo precio, se come una perdida terminal. Shumway (1997)
documenta que omitir ese retorno sesga los rendimientos al alza de forma
material, con medianas fuertemente negativas para exclusiones por causa. Aqui el
retorno terminal se aplica explicitamente y la etiqueta de una accion que muere
dentro del horizonte **existe y es mala**, que es justo lo que el modelo tiene
que aprender a evitar.

Contrato: `PITPanel.to_market_data()` devuelve un `MarketData` compatible con
todo el pipeline existente, asi que nada de lo que ya funciona se rompe.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .loaders import MarketData

__all__ = ["DelistReason", "PITPanel", "forward_returns_pit", "universe_asof",
           "load_synthetic_pit", "survivorship_view"]


class DelistReason:
    ACTIVE = "active"
    ACQUIRED = "acquired"        # fusion/OPA: el retorno terminal suele ser >= 0
    BANKRUPT = "bankrupt"        # quiebra: perdida terminal severa
    DROPPED = "dropped"          # exclusion por incumplimiento de requisitos


# Retornos terminales por defecto. Son valores de trabajo conservadores, no
# estimaciones de un dataset concreto: cuando se conecta Sharadar/EODHD, la
# columna real sustituye a estos.
DEFAULT_DELIST_RETURN = {
    DelistReason.ACTIVE: 0.0,
    DelistReason.ACQUIRED: 0.0,
    DelistReason.BANKRUPT: -0.70,
    DelistReason.DROPPED: -0.35,
}


@dataclass
class PITPanel:
    """Precios y volumenes con NaN despues del delisting, mas la tabla de vida.

    `listings` tiene indice `symbol` y columnas:
        first_date, last_date, reason, delist_return
    """

    prices: pd.DataFrame
    volumes: pd.DataFrame
    listings: pd.DataFrame

    def __post_init__(self) -> None:
        req = {"first_date", "last_date", "reason", "delist_return"}
        missing = req - set(self.listings.columns)
        if missing:
            raise ValueError(f"listings sin columnas {sorted(missing)}")
        # `reindex` con un Index sin nombre PISA el nombre del indice, y luego
        # `reset_index()` genera una columna llamada "index" en vez de "symbol".
        # Se preserva explicitamente.
        self.listings = self.listings.reindex(self.prices.columns)
        self.listings.index.name = "symbol"
        self.volumes = self.volumes.reindex_like(self.prices)

    # ------------------------------------------------------------- universo
    def universe_asof(self, date) -> list[str]:
        """Simbolos vivos y cotizando en `date`. Point-in-time de verdad."""
        d = pd.Timestamp(date)
        L = self.listings
        alive = (L["first_date"] <= d) & (L["last_date"] >= d)
        return list(L.index[alive])

    @property
    def n_delisted(self) -> int:
        return int((self.listings["reason"] != DelistReason.ACTIVE).sum())

    def summary(self) -> dict:
        r = self.listings["reason"].value_counts().to_dict()
        return {"n_symbols": int(self.prices.shape[1]), "n_dates": int(self.prices.shape[0]),
                "n_delisted": self.n_delisted, "por_motivo": r}

    # ------------------------------------------------------------- vistas
    def to_market_data(self) -> MarketData:
        """Vista compatible con el pipeline existente (incluye deslistados)."""
        return MarketData(prices=self.prices.copy(), volumes=self.volumes.copy())

    def survivors_only(self) -> "PITPanel":
        """La vista SESGADA: solo los que siguen vivos en la ultima fecha.

        Existe unicamente para poder MEDIR el sesgo comparandola con el panel
        completo. No se usa en produccion.
        """
        last = self.prices.index.max()
        alive = self.universe_asof(last)
        return PITPanel(self.prices[alive].copy(), self.volumes[alive].copy(),
                        self.listings.loc[alive].copy())


def universe_asof(listings: pd.DataFrame, date) -> list[str]:
    d = pd.Timestamp(date)
    alive = (listings["first_date"] <= d) & (listings["last_date"] >= d)
    return list(listings.index[alive])


# ------------------------------------------------------------------ etiquetas
def forward_returns_pit(prices: pd.DataFrame, listings: pd.DataFrame,
                        horizon: int) -> pd.Series:
    """Retorno futuro a `horizon` sesiones, CON retorno terminal de delisting.

    Tres casos, y el tercero es el que arregla el sesgo en la etiqueta:

      a) la accion sigue viva en t+h  -> precio[t+h]/precio[t] - 1
      b) la accion no cotiza en t     -> NaN (no es elegible, correcto)
      c) la accion muere en t+j<=h    -> (ultimo_precio/precio[t])*(1+r_delist) - 1

    En (c) la observacion EXISTE y normalmente es muy negativa. Devolver NaN
    ahi, que es lo que hace un `shift(-h)` ingenuo, equivale a decir "si quiebra,
    no cuenta", y es exactamente el sesgo que se quiere eliminar.
    """
    px = prices.sort_index()
    idx = px.index
    fwd_simple = px.shift(-horizon) / px - 1.0

    # ultimo precio observado de cada simbolo y su retorno terminal
    last_px = px.ffill().iloc[-1]
    last_valid = px.apply(lambda s: s.last_valid_index())
    dr = listings["delist_return"].reindex(px.columns).fillna(0.0)
    reason = listings["reason"].reindex(px.columns).fillna(DelistReason.ACTIVE)

    out = fwd_simple.copy()
    pos = pd.Series(np.arange(len(idx)), index=idx)

    for sym in px.columns:
        if reason[sym] == DelistReason.ACTIVE:
            continue
        lv = last_valid[sym]
        if lv is None or pd.isna(lv):
            continue
        p_end = float(px.at[lv, sym])
        if not np.isfinite(p_end) or p_end <= 0:
            continue
        terminal = p_end * (1.0 + float(dr[sym]))
        j = int(pos[lv])
        # fechas t tales que t < delisting <= t+h  ->  la posicion muere dentro
        lo = max(j - horizon, 0)
        affected = idx[lo:j]
        if len(affected) == 0:
            continue
        p_t = px.loc[affected, sym]
        out.loc[affected, sym] = (terminal / p_t - 1.0).where(p_t > 0)
        # despues del delisting no hay observacion posible
        out.loc[idx[j:], sym] = np.nan

    s = out.stack()
    s.index.names = ["date", "symbol"]
    return s.rename(f"fwd_{horizon}")


# ------------------------------------------------------------------ sintetico
def load_synthetic_pit(
    n_symbols: int = 80,
    n_days: int = 900,
    signal_strength: float = 0.0,
    momentum_strength: float = 0.0,
    annual_delist_rate: float = 0.06,
    bankrupt_share: float = 0.5,
    pre_death_drift_days: int = 120,
    seed: int = 7,
) -> PITPanel:
    """Mercado sintetico CON delistings, para medir el sesgo de supervivencia.

    El punto clave del generador: las acciones que van a ser excluidas por causa
    tienen deriva negativa ANTES de morir. Eso es lo que hace realista el
    experimento — si murieran de golpe desde una trayectoria normal, ningun
    modelo podria anticiparlo y el sesgo no se notaria en el IC.

    `annual_delist_rate` = 0.06 significa ~6% de bajas al ano, en el orden de
    magnitud de la rotacion historica de un indice amplio de EE. UU.

    `pre_death_drift_days` controla el REGIMEN de muerte y es el parametro que
    de verdad decide el signo del sesgo:
      * > 0  muerte anunciada: la accion se desangra durante N sesiones y los
             factores de precio la ven venir.
      * = 0  muerte subita: fraude, suspension, quiebra sorpresa. Ningun factor
             de precio puede anticiparla.
    Ambos regimenes existen en el mercado real y el sesgo se comporta de forma
    OPUESTA en cada uno, asi que hay que medir los dos.
    """
    from .loaders import load_synthetic

    base = load_synthetic(n_symbols=n_symbols, n_days=n_days,
                          signal_strength=signal_strength,
                          momentum_strength=momentum_strength, seed=seed)
    px, vol = base.prices.copy(), base.volumes.copy()
    idx = px.index
    rng = np.random.default_rng(seed + 991)

    years = max(len(idx) / 252.0, 1e-9)
    p_delist = 1.0 - (1.0 - annual_delist_rate) ** years
    n_del = int(round(p_delist * n_symbols))
    victims = rng.choice(px.columns, size=min(n_del, n_symbols - 5), replace=False)

    rows = []
    for sym in px.columns:
        if sym not in victims:
            rows.append((sym, idx[0], idx[-1], DelistReason.ACTIVE, 0.0))
            continue
        # muere en algun punto del ultimo 80% de la muestra
        j = int(rng.integers(int(0.2 * len(idx)), len(idx) - 1))
        bankrupt = rng.random() < bankrupt_share
        reason = DelistReason.BANKRUPT if bankrupt else DelistReason.ACQUIRED
        if bankrupt:
            k = min(int(pre_death_drift_days), j)
            if k > 0:                      # muerte anunciada
                decay = np.linspace(0.0, -0.9, k)
                px.iloc[j - k:j, px.columns.get_loc(sym)] *= np.exp(decay)
            dr = DEFAULT_DELIST_RETURN[DelistReason.BANKRUPT]
        else:
            # OPA: prima positiva el ultimo dia
            dr = float(rng.uniform(0.05, 0.30))
        px.iloc[j:, px.columns.get_loc(sym)] = np.nan
        vol.iloc[j:, vol.columns.get_loc(sym)] = np.nan
        rows.append((sym, idx[0], idx[j - 1], reason, dr))

    listings = pd.DataFrame(rows, columns=["symbol", "first_date", "last_date",
                                           "reason", "delist_return"]).set_index("symbol")
    return PITPanel(px, vol, listings)


def survivorship_view(panel: PITPanel) -> MarketData:
    """`MarketData` tal y como lo veria alguien que descarga tickers de hoy."""
    return panel.survivors_only().to_market_data()
