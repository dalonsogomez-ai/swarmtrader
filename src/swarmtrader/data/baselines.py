"""Baselines sin IA. Si el torneo no los bate NETO, no ha anadido nada.

Se construyen como Genomas degenerados para pasar exactamente por el mismo
simulador, mismos costes y mismo lag que los agentes evolucionados. Comparar
contra un baseline calculado con otro motor es una fuente clasica de
autoengano.
"""

from __future__ import annotations

from ..agents.genome import Genome
from ..backtest import BacktestResult, FeaturePanel, simulate
from ..data.features import FACTOR_NAMES


def _zero_weights(names=FACTOR_NAMES) -> dict[str, float]:
    return {n: 0.0 for n in names}


def equal_weight_buy_hold(panel: FeaturePanel) -> Genome:
    """Compra todo el universo elegible a partes iguales y no toca nada."""
    return Genome(weights=_zero_weights(panel.factor_names), top_k=len(panel.symbols), min_conviction=-1e9,
                  rebalance_days=10**6, stop_loss=1.0, take_profit=1e9,
                  min_dollar_volume=0.0, max_weight=1.0, agent_id="BASE_ew_bh")


def momentum_top_n(n: int = 20, rebalance_days: int = 21) -> Genome:
    """Momentum 12-1 clasico, top-N equiponderado, rebalanceo mensual."""
    w = _zero_weights()
    w["mom_12_1"] = 1.0
    return Genome(weights=w, top_k=n, min_conviction=-1e9, rebalance_days=rebalance_days,
                  stop_loss=1.0, take_profit=1e9, min_dollar_volume=0.0, max_weight=1.0,
                  agent_id=f"BASE_mom_top{n}")


def momentum_quality_top_n(n: int = 20, rebalance_days: int = 21) -> Genome:
    """Momentum + baja vol + anti-loteria: el baseline 'razonable' a batir."""
    w = _zero_weights()
    w.update({"mom_12_1": 0.5, "low_vol_60": 0.25, "anti_lottery": 0.25})
    return Genome(weights=w, top_k=n, min_conviction=-1e9, rebalance_days=rebalance_days,
                  stop_loss=1.0, take_profit=1e9, min_dollar_volume=0.0, max_weight=1.0,
                  agent_id=f"BASE_momq_top{n}")


def run_baselines(panel: FeaturePanel, start: int, end: int, cost_bps: float) -> dict[str, BacktestResult]:
    out = {}
    for g in (equal_weight_buy_hold(panel), momentum_top_n(), momentum_quality_top_n()):
        out[g.agent_id] = simulate(panel, g, start, end, cost_bps=cost_bps)
    return out
