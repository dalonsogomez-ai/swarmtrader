"""Controles de multiple testing para evaluacion de backtests.

Este es el modulo mas importante del proyecto. Un torneo evolutivo genera
cientos o miles de variantes de estrategia; la mejor mostrara un Sharpe alto
aunque TODAS las variantes sean ruido puro. Estas funciones cuantifican cuanto
del Sharpe observado se explica por la propia busqueda.

Referencia
----------
Bailey, D. H. & Lopez de Prado, M. (2014). "The Deflated Sharpe Ratio:
Correcting for Selection Bias, Backtest Overfitting and Non-Normality".
Journal of Portfolio Management 40(5), 94-107.
"""

from __future__ import annotations

import numpy as np
from scipy import stats

EULER_MASCHERONI = 0.5772156649015329


def sharpe_ratio(returns: np.ndarray, periods_per_year: int = 252) -> float:
    """Sharpe anualizado. Devuelve 0.0 si no hay dispersion."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 2:
        return 0.0
    sd = r.std(ddof=1)
    if sd <= 1e-12:
        return 0.0
    return float(r.mean() / sd * np.sqrt(periods_per_year))


def probabilistic_sharpe_ratio(
    observed_sr: float,
    n_obs: int,
    skew: float = 0.0,
    kurtosis: float = 3.0,
    benchmark_sr: float = 0.0,
) -> float:
    """P(SR_real > benchmark) corrigiendo por no-normalidad y tamano muestral.

    PSR es el caso N=1 del DSR: corrige asimetria y colas gruesas pero NO el
    sesgo de seleccion. Los SR van en unidades por periodo (sin anualizar).
    """
    if n_obs < 3:
        return 0.5
    denom = 1.0 - skew * observed_sr + 0.25 * (kurtosis - 1.0) * observed_sr**2
    if denom <= 0:
        return 0.5
    se = np.sqrt(denom / (n_obs - 1))
    if se <= 1e-12:
        return 1.0 if observed_sr > benchmark_sr else 0.0
    return float(stats.norm.cdf((observed_sr - benchmark_sr) / se))


def expected_max_sharpe(n_trials: int, var_trial_sharpes: float) -> float:
    """E[max SR] bajo H0: SR real = 0 en todas las pruebas.

    Este es el umbral que hay que superar. Crece con el numero de pruebas: si
    pruebas 1000 estrategias sin edge, la mejor tendra un SR aparente positivo
    y grande solo por azar.
    """
    if n_trials < 2 or var_trial_sharpes <= 0:
        return 0.0
    sd = np.sqrt(var_trial_sharpes)
    n = float(n_trials)
    term_a = stats.norm.ppf(1.0 - 1.0 / n)
    term_b = stats.norm.ppf(1.0 - 1.0 / (n * np.e))
    return float(sd * ((1.0 - EULER_MASCHERONI) * term_a + EULER_MASCHERONI * term_b))


def deflated_sharpe_ratio(
    returns: np.ndarray,
    n_trials: int,
    var_trial_sharpes: float,
) -> float:
    """DSR: probabilidad de que el SR real supere el maximo esperado bajo H0.

    Interpretacion estandar:
      DSR > 0.95  -> evidencia solida de alfa genuino
      DSR < 0.50  -> el sesgo de seleccion explica el resultado entero

    Parameters
    ----------
    returns : retornos por periodo de la estrategia candidata.
    n_trials : numero EFECTIVO de pruebas independientes (ver effective_n_trials).
    var_trial_sharpes : varianza de los SR (por periodo) de todas las pruebas.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 3:
        return 0.0
    sd = r.std(ddof=1)
    if sd <= 1e-12:
        return 0.0
    sr = r.mean() / sd  # por periodo, sin anualizar
    sr_star = expected_max_sharpe(n_trials, var_trial_sharpes)
    return probabilistic_sharpe_ratio(
        observed_sr=sr,
        n_obs=r.size,
        skew=float(stats.skew(r)),
        kurtosis=float(stats.kurtosis(r, fisher=False)),
        benchmark_sr=sr_star,
    )


def effective_n_trials(returns_matrix: np.ndarray) -> int:
    """Numero efectivo de pruebas INDEPENDIENTES via participation ratio.

    Contar los backtests literalmente sobrestima N: las variantes de un torneo
    evolutivo estan muy correlacionadas. Lopez de Prado propone clustering
    (ONC); aqui usamos un estimador espectral equivalente en espiritu:
    N_eff = (sum lambda)^2 / sum(lambda^2) sobre los autovalores de la matriz
    de correlacion entre las series de retornos de las pruebas.

    returns_matrix : shape (n_periods, n_trials)
    """
    X = np.asarray(returns_matrix, dtype=float)
    if X.ndim != 2 or X.shape[1] < 2:
        return max(1, X.shape[1] if X.ndim == 2 else 1)
    keep = X.std(axis=0, ddof=1) > 1e-12
    X = X[:, keep]
    if X.shape[1] < 2:
        return max(1, int(X.shape[1]))
    corr = np.nan_to_num(np.corrcoef(X, rowvar=False), nan=0.0)
    eigvals = np.clip(np.linalg.eigvalsh(corr), 0.0, None)
    total = eigvals.sum()
    if total <= 0:
        return 1
    return int(max(1, round((total**2) / (eigvals**2).sum())))


def max_drawdown(equity: np.ndarray) -> float:
    """Maximo drawdown como fraccion positiva (0.25 = -25%)."""
    e = np.asarray(equity, dtype=float)
    if e.size == 0:
        return 0.0
    peak = np.maximum.accumulate(e)
    dd = np.where(peak > 0, (peak - e) / peak, 0.0)
    return float(dd.max())
