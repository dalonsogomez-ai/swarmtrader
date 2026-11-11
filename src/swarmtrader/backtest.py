"""Simulador de cartera long-only. Determinista y sin look-ahead.

Reglas que hacen el backtest honesto:
  * la senal de la fecha t se ejecuta al cierre de t+1 (execution_lag=1)
  * los costes se aplican sobre el turnover real, ida y vuelta, mas impacto
    de mercado proporcional a sqrt(notional / dollar_volume)
  * los simbolos sin liquidez suficiente se excluyen ANTES de rankear
  * si ningun simbolo supera min_conviction (o el meta-modelo), el agente se
    queda en LIQUIDEZ. La abstencion es una accion valida.
  * en regimen de alta volatilidad la exposicion se escala por regime_exposure
  * cada cierre de posicion emite un evento de P&L realizado con fecha y
    simbolo, para que tax.py pueda aplicar el IRPF y la regla de los 2 meses
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .agents.genome import Genome
from .data.features import FACTOR_NAMES, REGIME_COLUMNS
from .evolution.meta import META_FEATURE_NAMES, MetaLabeler, PickRecord, triple_barrier_outcome
from .stats.deflated_sharpe import max_drawdown, sharpe_ratio


@dataclass
class RealizedEvent:
    t: int
    symbol_idx: int
    pnl_frac: float          # P&L realizado como fraccion del capital inicial
    hold_days: int


@dataclass
class BacktestResult:
    returns: np.ndarray
    equity: np.ndarray
    dates: pd.DatetimeIndex
    turnover: float
    n_trades: int
    pct_days_in_cash: float
    sharpe: float
    total_return: float
    max_dd: float
    realized: list[RealizedEvent] = field(default_factory=list)
    buys: list[tuple[int, int]] = field(default_factory=list)      # (t relativo a la ventana, symbol_idx)
    picks: list[PickRecord] = field(default_factory=list)
    cost_drag: float = 0.0          # retorno total perdido en costes
    meta_rejections: int = 0

    def summary(self) -> dict:
        return {
            "sharpe": round(self.sharpe, 3),
            "total_return": round(self.total_return, 4),
            "max_dd": round(self.max_dd, 4),
            "turnover": round(self.turnover, 3),
            "n_trades": self.n_trades,
            "pct_cash": round(self.pct_days_in_cash, 3),
            "cost_drag": round(self.cost_drag, 4),
            "meta_rejections": self.meta_rejections,
        }


class FeaturePanel:
    """Panel (fechas x simbolos x factores) en numpy para scoring rapido."""

    def __init__(self, features: pd.DataFrame, prices: pd.DataFrame,
                 dollar_vol: pd.Series, regime: pd.DataFrame | None = None):
        self.dates = prices.index
        self.symbols = list(prices.columns)
        self.factor_names: list[str] = [c for c in features.columns if c not in ("px", "dollar_volume")]
        n_d, n_s, n_f = len(self.dates), len(self.symbols), len(self.factor_names)

        cube = np.zeros((n_d, n_s, n_f), dtype=np.float32)
        for k, name in enumerate(self.factor_names):
            wide = features[name].unstack("symbol").reindex(index=self.dates, columns=self.symbols)
            cube[:, :, k] = wide.to_numpy(dtype=np.float32, na_value=0.0)
        self.cube = cube

        self.prices = prices.to_numpy(dtype=np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = self.prices[1:] / self.prices[:-1] - 1.0
        self.returns = np.vstack([np.zeros((1, n_s)), r])
        self.returns = np.clip(np.nan_to_num(self.returns, nan=0.0, posinf=0.0, neginf=0.0), -0.9, 3.0)

        dv = dollar_vol.unstack("symbol").reindex(index=self.dates, columns=self.symbols)
        self.dollar_vol = dv.to_numpy(dtype=np.float64, na_value=0.0)
        self.valid = np.isfinite(self.prices) & (self.prices > 0)

        if regime is None:
            self.regime = np.zeros((n_d, len(REGIME_COLUMNS)))
        else:
            self.regime = regime.reindex(self.dates)[list(REGIME_COLUMNS)].to_numpy(dtype=np.float64, na_value=0.0)

    def scores(self, weights: np.ndarray) -> np.ndarray:
        return self.cube @ weights.astype(np.float32)


def _regime_multiplier(genome: Genome, vol_z: float) -> float:
    return genome.regime_exposure if vol_z > genome.vol_regime_cut else 1.0


def simulate(
    panel: FeaturePanel,
    genome: Genome,
    start: int,
    end: int,
    cost_bps: float = 10.0,
    execution_lag: int = 1,
    capital: float = 50.0,
    impact_coef: float = 0.0,
    meta: MetaLabeler | None = None,
    record_picks: bool = True,
) -> BacktestResult:
    """Simula el genoma sobre el rango [start, end) de indices de fecha."""
    scores_all = panel.scores(genome.weight_vector(panel.factor_names))
    n_sym = len(panel.symbols)
    cost = cost_bps / 10_000.0
    ivz = REGIME_COLUMNS.index("mkt_vol_z")
    ir63 = REGIME_COLUMNS.index("mkt_ret_63")

    w = np.zeros(n_sym)
    entry_px = np.zeros(n_sym)
    entry_t = np.full(n_sym, -1)
    entry_w = np.zeros(n_sym)
    daily_ret: list[float] = []
    turnovers: list[float] = []
    realized: list[RealizedEvent] = []
    buys: list[tuple[int, int]] = []
    picks: list[PickRecord] = []
    open_picks: dict[int, PickRecord] = {}
    n_trades = days_in_cash = meta_rej = 0
    cost_drag = 0.0
    next_rebalance = start

    def _trade_cost(trade_w: np.ndarray, t: int) -> float:
        base = cost * float(np.abs(trade_w).sum())
        if impact_coef <= 0:
            return base
        notional = np.abs(trade_w) * capital
        dv = np.where(panel.dollar_vol[t] > 0, panel.dollar_vol[t], np.inf)
        return base + float(impact_coef * np.sum(np.abs(trade_w) * np.sqrt(notional / dv)))

    def _close(mask: np.ndarray, t: int) -> None:
        nonlocal n_trades
        for j in np.where(mask)[0]:
            if entry_px[j] > 0:
                pnl = entry_w[j] * (panel.prices[t, j] / entry_px[j] - 1.0)
                realized.append(RealizedEvent(t - start, int(j), float(pnl), int(t - entry_t[j])))
                if record_picks and j in open_picks:
                    path = panel.returns[entry_t[j] + 1: t + 1, j]
                    open_picks[j].outcome = triple_barrier_outcome(path, genome.take_profit, genome.stop_loss)
                    open_picks.pop(j)
            n_trades += 1
        w[mask] = 0.0
        entry_px[mask] = 0.0
        entry_t[mask] = -1
        entry_w[mask] = 0.0

    for t in range(start, end):
        r = float(np.dot(w, panel.returns[t]))
        if w.sum() <= 1e-9:
            days_in_cash += 1
        else:
            # Los pesos se expresan sobre el NAV actual, no sobre el inicial.
            # Sin normalizar, buy-and-hold acumula exposicion ficticia y el
            # retorno termina contando dos veces el crecimiento de posiciones.
            w = w * (1.0 + panel.returns[t]) / (1.0 + r)

        held = w > 1e-9
        if held.any():
            px = panel.prices[t]
            with np.errstate(divide="ignore", invalid="ignore"):
                pnl = np.where(entry_px > 0, px / entry_px - 1.0, 0.0)
            exit_mask = held & ((pnl <= -genome.stop_loss) | (pnl >= genome.take_profit))
            if exit_mask.any():
                c = _trade_cost(w * exit_mask, t)
                turnovers.append(float(w[exit_mask].sum()))
                r -= c
                cost_drag += c
                _close(exit_mask, t)

        if t >= next_rebalance:
            sig_t = t - execution_lag
            if sig_t >= 0:
                s = scores_all[sig_t].astype(np.float64)
                eligible = panel.valid[t] & (panel.dollar_vol[sig_t] >= genome.min_dollar_volume) & np.isfinite(s)
                target = np.zeros(n_sym)
                if eligible.any():
                    s_m = np.where(eligible, s, -np.inf)
                    k = min(genome.top_k, int(eligible.sum()))
                    top = np.argpartition(-s_m, k - 1)[:k] if k > 0 else np.array([], int)
                    top = top[np.argsort(-s_m[top])]
                    top = top[s_m[top] >= genome.min_conviction]
                    if top.size > 0:
                        # meta-features del pick
                        med = float(np.median(s[eligible]))
                        nxt = float(np.sort(s_m[eligible])[-(top.size + 1)]) if eligible.sum() > top.size else med
                        vol_z, ret63 = float(panel.regime[sig_t, ivz]), float(panel.regime[sig_t, ir63])
                        feats = np.array([[s[j], s[j] - med, s[j] - nxt, top.size, vol_z, ret63] for j in top])
                        sizes = np.ones(top.size)
                        if meta is not None and meta.fitted:
                            p = meta.predict_proba(feats)
                            keep = p >= genome.meta_threshold
                            meta_rej += int((~keep).sum())
                            top, feats, sizes = top[keep], feats[keep], p[keep]
                        if top.size > 0:
                            raw = sizes / sizes.sum()
                            raw = np.minimum(raw, genome.max_weight)
                            raw *= _regime_multiplier(genome, vol_z)
                            target[top] = raw
                            if record_picks:
                                for j, f in zip(top, feats):
                                    if w[j] <= 1e-9:
                                        rec = PickRecord(t=t - start, symbol_idx=int(j), features=f)
                                        picks.append(rec)
                                        open_picks[int(j)] = rec

                trade_w = target - w
                if np.abs(trade_w).sum() > 1e-6:
                    closing = (target <= 1e-9) & (w > 1e-9)
                    if closing.any():
                        _close(closing, t)
                    c = _trade_cost(trade_w, t)
                    turnovers.append(float(np.abs(trade_w).sum()))
                    r -= c
                    cost_drag += c
                    n_trades += int((np.abs(trade_w) > 1e-6).sum())
                    newly = (target > 1e-9) & (entry_px <= 0)
                    entry_px = np.where(newly, panel.prices[t], entry_px)
                    entry_t = np.where(newly, t, entry_t)
                    entry_w = np.where(target > 1e-9, target, entry_w)
                    buys.extend((t - start, int(j)) for j in np.where(newly)[0])
                    w = target
            next_rebalance = t + max(1, genome.rebalance_days)

        daily_ret.append(r)

    # cierra lo que quede abierto para que el P&L realizado cuadre con el equity
    if (w > 1e-9).any():
        _close(w > 1e-9, end - 1)

    rets = np.asarray(daily_ret, dtype=float)
    equity = np.cumprod(1.0 + rets)
    n = max(1, len(rets))
    return BacktestResult(
        returns=rets, equity=equity, dates=panel.dates[start:end],
        turnover=float(np.mean(turnovers)) if turnovers else 0.0,
        n_trades=n_trades, pct_days_in_cash=days_in_cash / n,
        sharpe=sharpe_ratio(rets),
        total_return=float(equity[-1] - 1.0) if len(equity) else 0.0,
        max_dd=max_drawdown(equity),
        realized=realized, buys=buys, picks=picks,
        cost_drag=cost_drag, meta_rejections=meta_rej,
    )
