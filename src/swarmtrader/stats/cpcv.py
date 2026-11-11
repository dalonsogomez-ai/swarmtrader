"""Validacion cruzada purgada y combinatoria (CPCV).

Por que existe este modulo
--------------------------
Con horizonte de 21 sesiones, la etiqueta de la fecha t se construye con precios
hasta t+21. Una k-fold normal mete en entrenamiento observaciones cuyo futuro
solapa con el periodo de test: el modelo ve, indirectamente, el resultado que
tiene que predecir. Ese solape infla el IC de forma silenciosa y es la fuga de
informacion mas frecuente en backtests de seleccion de acciones.

La solucion (Lopez de Prado, *Advances in Financial ML*, cap. 7) es doble:

  * **purga**: se eliminan del entrenamiento las observaciones cuya ventana de
    etiqueta [t, t+horizon] solape con la ventana de test.
  * **embargo**: se elimina ademas una banda de `embargo` sesiones DESPUES del
    test, porque la autocorrelacion serial de los factores hace que las
    observaciones inmediatamente posteriores sigan contaminadas.

`combinatorial_splits` genera ademas las C(n_groups, k_test) combinaciones de
bloques de test, lo que produce multiples caminos de backtest en lugar de uno
solo. Eso es lo que alimenta a PBO con una distribucion de resultados en vez de
con un unico numero.

Todo trabaja sobre indices de FECHAS unicas, no de filas: el panel es
cross-seccional y todas las filas de una misma fecha comparten ventana.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

__all__ = ["Split", "purged_kfold_splits", "combinatorial_splits", "n_backtest_paths"]


@dataclass(frozen=True)
class Split:
    """Un corte de validacion, en indices posicionales sobre el vector de fechas."""

    train: np.ndarray
    test: np.ndarray

    def __post_init__(self) -> None:
        if np.intersect1d(self.train, self.test).size:
            raise ValueError("train y test solapan: la purga ha fallado")


def _purge(
    n_dates: int,
    test_idx: np.ndarray,
    horizon: int,
    embargo: int,
) -> np.ndarray:
    """Indices de entrenamiento admisibles dado un bloque de test.

    Se excluye [min(test) - horizon, max(test) + horizon + embargo]. El margen
    por la izquierda es `horizon` porque una observacion en t - horizon + 1
    todavia tiene su etiqueta dentro del test.
    """
    if test_idx.size == 0:
        return np.arange(n_dates)
    lo = int(test_idx.min()) - horizon
    hi = int(test_idx.max()) + horizon + embargo
    all_idx = np.arange(n_dates)
    return all_idx[(all_idx < lo) | (all_idx > hi)]


def purged_kfold_splits(
    n_dates: int,
    n_splits: int = 6,
    horizon: int = 21,
    embargo: int = 5,
) -> list[Split]:
    """K-fold contigua con purga y embargo."""
    if n_splits < 2:
        raise ValueError("n_splits >= 2")
    if n_dates < n_splits:
        raise ValueError(f"muy pocas fechas ({n_dates}) para {n_splits} folds")
    bounds = np.linspace(0, n_dates, n_splits + 1).astype(int)
    out: list[Split] = []
    for i in range(n_splits):
        test = np.arange(bounds[i], bounds[i + 1])
        if test.size == 0:
            continue
        train = _purge(n_dates, test, horizon, embargo)
        if train.size == 0:
            continue
        out.append(Split(train=train, test=test))
    return out


def combinatorial_splits(
    n_dates: int,
    n_groups: int = 6,
    k_test: int = 2,
    horizon: int = 21,
    embargo: int = 5,
    max_splits: int | None = None,
) -> list[Split]:
    """CPCV: todas las combinaciones de `k_test` bloques de test entre `n_groups`.

    Con n_groups=6 y k_test=2 salen 15 cortes y 5 caminos de backtest
    independientes, frente al camino unico de un walk-forward simple.
    """
    if not 1 <= k_test < n_groups:
        raise ValueError("se requiere 1 <= k_test < n_groups")
    if n_dates < n_groups:
        raise ValueError(f"muy pocas fechas ({n_dates}) para {n_groups} grupos")
    bounds = np.linspace(0, n_dates, n_groups + 1).astype(int)
    groups = [np.arange(bounds[i], bounds[i + 1]) for i in range(n_groups)]
    out: list[Split] = []
    for combo in combinations(range(n_groups), k_test):
        test = np.concatenate([groups[i] for i in combo])
        if test.size == 0:
            continue
        # la purga se aplica por bloque contiguo, no sobre el rango global:
        # dos bloques de test lejanos no deben borrar todo lo que hay en medio.
        train = np.arange(n_dates)
        for i in combo:
            if groups[i].size:
                keep = _purge(n_dates, groups[i], horizon, embargo)
                train = np.intersect1d(train, keep, assume_unique=False)
        if train.size == 0:
            continue
        out.append(Split(train=train, test=test))
        if max_splits is not None and len(out) >= max_splits:
            break
    return out


def n_backtest_paths(n_groups: int, k_test: int) -> int:
    """Numero de caminos de backtest que CPCV reconstruye.

    phi = C(n_groups, k_test) * k_test / n_groups
    """
    from math import comb

    return comb(n_groups, k_test) * k_test // n_groups
