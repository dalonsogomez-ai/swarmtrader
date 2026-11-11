"""Funcion de fitness. Aqui esta la diferencia entre ciencia y casino.

Seleccionar por P&L bruto selecciona por VARIANZA, no por edge: el agente que
concentra todo en un nombre tiene a la vez la mayor probabilidad de ser campeon
y de arruinarse. El fitness de este proyecto es una probabilidad ajustada por
riesgo, penalizada por turnover y drawdown, y calculada SIEMPRE en ventana
out-of-sample.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

from ..backtest import BacktestResult
from ..stats.deflated_sharpe import probabilistic_sharpe_ratio


@dataclass(frozen=True)
class FitnessConfig:
    turnover_penalty: float = 0.15     # coste de fitness por unidad de turnover medio
    drawdown_penalty: float = 0.80     # coste por unidad de max drawdown
    min_periods: int = 30
    benchmark_sharpe: float = 0.0      # anualizado; sube esto para exigir batir un baseline
    periods_per_year: int = 252


def compute_fitness(result: BacktestResult, cfg: FitnessConfig = FitnessConfig()) -> float:
    """Fitness en [-1, 1]. Cero = indistinguible de no tener edge.

    Base: PSR (probabilidad de que el Sharpe real supere el benchmark),
    reescalada a [-1,1], menos penalizaciones por friccion y por riesgo de ruina.
    """
    r = np.asarray(result.returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < cfg.min_periods:
        return -1.0
    sd = r.std(ddof=1)
    if sd <= 1e-12:
        return 0.0

    sr_period = r.mean() / sd
    bench_period = cfg.benchmark_sharpe / np.sqrt(cfg.periods_per_year)
    psr = probabilistic_sharpe_ratio(
        observed_sr=sr_period,
        n_obs=r.size,
        skew=float(stats.skew(r)),
        kurtosis=float(stats.kurtosis(r, fisher=False)),
        benchmark_sr=bench_period,
    )
    base = 2.0 * psr - 1.0
    penalty = cfg.turnover_penalty * result.turnover + cfg.drawdown_penalty * result.max_dd
    return float(np.clip(base - penalty, -1.0, 1.0))


def posterior_edge_probability(
    returns: np.ndarray,
    prior_sharpe_sd: float = 0.5,
    periods_per_year: int = 252,
) -> float:
    """P(mu > 0 | datos) con prior escéptico conjugado normal-normal.

    El prior dice: "antes de ver nada, creo que el Sharpe real de este agente
    esta centrado en 0 con desviacion `prior_sharpe_sd` anualizada". Con
    prior_sharpe_sd=0.5 hace falta bastante evidencia para convencerlo.

    Esta es la cantidad correcta sobre la que decidir la muerte de un agente:
    penaliza al que gana poco con mucho ruido y protege al que va mal por mala
    suerte pero tiene una serie coherente.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    if n < 5:
        return 0.5
    sigma = r.std(ddof=1)
    if sigma <= 1e-12:
        return 0.5

    tau = (prior_sharpe_sd / np.sqrt(periods_per_year)) * sigma  # sd del prior sobre mu
    if tau <= 1e-15:
        return 0.5
    prec_prior = 1.0 / tau**2
    prec_data = n / sigma**2
    post_prec = prec_prior + prec_data
    post_mean = (prec_data * r.mean()) / post_prec
    post_sd = np.sqrt(1.0 / post_prec)
    return float(stats.norm.cdf(post_mean / post_sd))
