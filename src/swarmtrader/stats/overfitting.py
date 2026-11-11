"""Probability of Backtest Overfitting (PBO) y validacion cruzada combinatoria.

DSR responde: "dado cuantas pruebas hice, es creible este Sharpe?"
PBO responde una pregunta distinta y complementaria: "si elijo la mejor
estrategia in-sample, con que probabilidad rinde peor que la mediana
out-of-sample?". Un PBO alto significa que la seleccion no generaliza aunque
cada Sharpe individual sea creible.

Referencia
----------
Bailey, Borwein, Lopez de Prado & Zhu (2017). "The Probability of Backtest
Overfitting". Journal of Computational Finance 20(4), 39-69.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform


@dataclass(frozen=True)
class PBOResult:
    pbo: float                      # P(el mejor IS queda bajo la mediana OOS)
    n_paths: int
    logits: np.ndarray              # lambda por combinacion
    oos_sharpe_of_is_best: np.ndarray
    prob_oos_loss: float            # P(Sharpe OOS del mejor IS < 0)
    degradation_slope: float        # pendiente Sharpe_OOS ~ Sharpe_IS (1 = sin degradacion)

    def summary(self) -> dict:
        return {
            "pbo": round(self.pbo, 4),
            "n_paths": self.n_paths,
            "prob_oos_loss": round(self.prob_oos_loss, 4),
            "degradation_slope": round(self.degradation_slope, 3),
            "median_oos_sharpe_of_is_best": round(float(np.median(self.oos_sharpe_of_is_best)), 3),
        }


def _sharpe_cols(M: np.ndarray) -> np.ndarray:
    mu = M.mean(axis=0)
    sd = M.std(axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        sr = np.where(sd > 1e-12, mu / sd, 0.0)
    return sr


def probability_of_backtest_overfitting(
    returns_matrix: np.ndarray,
    n_splits: int = 10,
    max_paths: int = 1000,
    seed: int = 0,
) -> PBOResult:
    """CSCV (combinatorially symmetric cross-validation) sobre la matriz de trials.

    returns_matrix : (T periodos x N trials), retornos por periodo de cada
                     estrategia evaluada, alineados en el tiempo.
    """
    M = np.asarray(returns_matrix, dtype=float)
    if M.ndim != 2 or M.shape[1] < 2 or M.shape[0] < 2 * n_splits:
        return PBOResult(pbo=float("nan"), n_paths=0, logits=np.array([]),
                         oos_sharpe_of_is_best=np.array([]), prob_oos_loss=float("nan"),
                         degradation_slope=float("nan"))
    T, N = M.shape
    T_used = (T // n_splits) * n_splits
    subs = np.array_split(M[:T_used], n_splits, axis=0)

    combos = list(combinations(range(n_splits), n_splits // 2))
    rng = np.random.default_rng(seed)
    if len(combos) > max_paths:
        idx = rng.choice(len(combos), size=max_paths, replace=False)
        combos = [combos[i] for i in idx]

    logits, oos_best, is_srs, oos_srs = [], [], [], []
    for train in combos:
        test = tuple(i for i in range(n_splits) if i not in train)
        J = np.vstack([subs[i] for i in train])
        Jc = np.vstack([subs[i] for i in test])
        sr_is, sr_oos = _sharpe_cols(J), _sharpe_cols(Jc)
        n_star = int(np.argmax(sr_is))
        # rango relativo del mejor IS dentro de los OOS
        rank = float((sr_oos < sr_oos[n_star]).sum() + 0.5 * (sr_oos == sr_oos[n_star]).sum())
        omega = (rank + 0.5) / (N + 1.0)
        omega = float(np.clip(omega, 1e-6, 1 - 1e-6))
        logits.append(np.log(omega / (1.0 - omega)))
        oos_best.append(sr_oos[n_star])
        is_srs.append(sr_is[n_star])
        oos_srs.append(sr_oos[n_star])

    logits = np.asarray(logits)
    oos_best = np.asarray(oos_best)
    is_srs, oos_srs = np.asarray(is_srs), np.asarray(oos_srs)
    if is_srs.std() > 1e-12:
        slope = float(np.polyfit(is_srs, oos_srs, 1)[0])
    else:
        slope = float("nan")
    return PBOResult(
        pbo=float((logits < 0).mean()),
        n_paths=len(combos),
        logits=logits,
        oos_sharpe_of_is_best=oos_best,
        prob_oos_loss=float((oos_best < 0).mean()),
        degradation_slope=slope,
    )


def effective_n_trials_clustering(returns_matrix: np.ndarray, corr_threshold: float = 0.5) -> int:
    """N efectivo por clustering jerarquico sobre distancia de correlacion.

    Dos pruebas cuya correlacion supera `corr_threshold` cuentan como UNA.
    Se corta el dendrograma (enlace medio) a la distancia equivalente
    sqrt((1 - rho)/2) y se cuentan los clusters resultantes.

    Es una version simplificada y determinista del ONC de Lopez de Prado;
    combinala con el estimador espectral y usa el mayor de los dos (mas
    conservador para el DSR).
    """
    X = np.asarray(returns_matrix, dtype=float)
    if X.ndim != 2 or X.shape[1] < 2:
        return max(1, X.shape[1] if X.ndim == 2 else 1)
    keep = X.std(axis=0, ddof=1) > 1e-12
    X = X[:, keep]
    N = X.shape[1]
    if N < 2:
        return max(1, N)
    corr = np.nan_to_num(np.corrcoef(X, rowvar=False), nan=0.0)
    dist = np.sqrt(np.clip(0.5 * (1.0 - corr), 0.0, 1.0))
    np.fill_diagonal(dist, 0.0)
    dist = 0.5 * (dist + dist.T)
    Z = linkage(squareform(dist, checks=False), method="average")
    cut = np.sqrt(0.5 * (1.0 - corr_threshold))
    labels = fcluster(Z, t=cut, criterion="distance")
    return int(max(1, np.unique(labels).size))
