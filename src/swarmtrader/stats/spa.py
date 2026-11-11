"""Contraste de superioridad predictiva con correccion por comparacion multiple.

El problema que resuelve
------------------------
Un enjambre de N agentes es N hipotesis. Si eliges el mejor de 40 agentes y le
calculas un t-stat, ese t-stat esta roto: el maximo de 40 variables aleatorias
tiene media positiva aunque las 40 sean ruido. Deflated Sharpe corrige esto para
UN estadistico; aqui hace falta el analogo para la comparacion de N candidatos
contra una linea base.

Dos contrastes, ambos sobre la hipotesis nula
``H0: ningun candidato es mejor que el benchmark``:

  * **Reality Check** (White 2000): recentra todos los candidatos por su media
    muestral. Es conservador cuando hay candidatos muy malos en el conjunto.
  * **SPA** (Hansen 2005): descarta del recentrado los candidatos cuya media
    esta por debajo de un umbral, de modo que un puñado de agentes basura no
    diluya la potencia del contraste. Es el que se usa por defecto.

La distribucion nula se simula con **bootstrap estacionario** (Politis-Romano
1994): remuestreo por bloques de longitud geometrica, que preserva la
autocorrelacion serial de las series de rendimiento. Un bootstrap iid daria
p-valores optimistas porque los retornos por fecha no son independientes.

Entrada: matriz ``d`` de (n_periodos x n_candidatos) con el DIFERENCIAL de
rendimiento de cada candidato frente al benchmark en cada periodo (positivo =
el candidato bate al benchmark). Salida: p-valor. p < 0.05 significa que al
menos un candidato bate al benchmark *despues* de pagar el peaje de haber
mirado a todos.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = ["SPAResult", "stationary_bootstrap_indices", "spa_test", "reality_check"]


@dataclass
class SPAResult:
    p_value: float
    statistic: float
    best_index: int
    best_mean: float
    n_candidates: int
    n_periods: int
    method: str

    def to_dict(self) -> dict:
        return {
            "p_value": round(self.p_value, 4),
            "statistic": round(self.statistic, 4),
            "best_index": int(self.best_index),
            "best_mean": round(self.best_mean, 6),
            "n_candidates": int(self.n_candidates),
            "n_periods": int(self.n_periods),
            "method": self.method,
        }

    @property
    def rejects(self) -> bool:
        """True si hay evidencia de superioridad al 5%."""
        return self.p_value < 0.05


def stationary_bootstrap_indices(
    n: int, n_boot: int, block_length: float, rng: np.random.Generator
) -> np.ndarray:
    """Indices (n_boot x n) del bootstrap estacionario de Politis-Romano.

    Cada paso continua el bloque actual con probabilidad 1 - 1/block_length, o
    salta a una posicion aleatoria. La longitud esperada de bloque es
    `block_length`; con block_length = 1 degenera en bootstrap iid.
    """
    if n <= 0:
        raise ValueError("n > 0")
    block_length = max(1.0, float(block_length))
    p_new = 1.0 / block_length
    idx = np.empty((n_boot, n), dtype=np.int64)
    idx[:, 0] = rng.integers(0, n, size=n_boot)
    if n > 1:
        jumps = rng.random((n_boot, n - 1)) < p_new
        fresh = rng.integers(0, n, size=(n_boot, n - 1))
        for t in range(1, n):
            cont = (idx[:, t - 1] + 1) % n
            idx[:, t] = np.where(jumps[:, t - 1], fresh[:, t - 1], cont)
    return idx


def _auto_block_length(d: np.ndarray) -> float:
    """Longitud de bloque a partir de la autocorrelacion de orden 1 media.

    Regla practica b = max(1, (1+rho)/(1-rho)); acotada a un cuarto de la
    muestra para que el bootstrap conserve variabilidad.
    """
    n = d.shape[0]
    if n < 4:
        return 1.0
    rhos = []
    for k in range(d.shape[1]):
        x = d[:, k] - d[:, k].mean()
        denom = float((x * x).sum())
        if denom <= 1e-18:
            continue
        rhos.append(float((x[:-1] * x[1:]).sum()) / denom)
    if not rhos:
        return 1.0
    rho = float(np.clip(np.mean(rhos), -0.9, 0.9))
    b = (1.0 + rho) / (1.0 - rho)
    return float(np.clip(b, 1.0, max(1.0, n / 4.0)))


def _omega(d: np.ndarray, boot_idx: np.ndarray) -> np.ndarray:
    """Desviacion tipica bootstrap de sqrt(n)*media, por candidato."""
    n = d.shape[0]
    means = d[boot_idx].mean(axis=1)                      # (n_boot, k)
    om = np.sqrt(n) * means.std(axis=0, ddof=1)
    return np.maximum(om, 1e-12)


def _spa_core(
    d: np.ndarray,
    n_boot: int,
    block_length: float | None,
    seed: int,
    method: str,
) -> SPAResult:
    d = np.asarray(d, dtype=float)
    if d.ndim == 1:
        d = d[:, None]
    mask = np.isfinite(d).all(axis=1)
    d = d[mask]
    n, k = d.shape
    if n < 8:
        raise ValueError(f"muestra insuficiente para bootstrap: {n} periodos")

    rng = np.random.default_rng(seed)
    b = _auto_block_length(d) if block_length is None else float(block_length)
    boot_idx = stationary_bootstrap_indices(n, n_boot, b, rng)

    dbar = d.mean(axis=0)
    omega = _omega(d, boot_idx)
    t_stats = np.sqrt(n) * dbar / omega
    statistic = float(max(t_stats.max(), 0.0))
    best = int(np.argmax(dbar))

    if method == "spa":
        # umbral de Hansen: se recentran solo los candidatos no manifiestamente malos
        thresh = -np.sqrt(omega**2 * 2.0 * np.log(np.log(n))) / (4.0 * n**0.25)
        keep = dbar >= thresh
        centre = np.where(keep, dbar, 0.0)
    elif method == "reality_check":
        centre = dbar
    else:  # pragma: no cover
        raise ValueError(method)

    boot_means = d[boot_idx].mean(axis=1)                 # (n_boot, k)
    boot_t = np.sqrt(n) * (boot_means - centre[None, :]) / omega[None, :]
    boot_stat = np.maximum(boot_t.max(axis=1), 0.0)

    p = float((boot_stat >= statistic).mean())
    return SPAResult(
        p_value=p,
        statistic=statistic,
        best_index=best,
        best_mean=float(dbar[best]),
        n_candidates=k,
        n_periods=n,
        method=method,
    )


def spa_test(
    d: np.ndarray,
    n_boot: int = 2000,
    block_length: float | None = None,
    seed: int = 0,
) -> SPAResult:
    """Contraste SPA de Hansen (2005). Ver docstring del modulo."""
    return _spa_core(d, n_boot, block_length, seed, "spa")


def reality_check(
    d: np.ndarray,
    n_boot: int = 2000,
    block_length: float | None = None,
    seed: int = 0,
) -> SPAResult:
    """Reality Check de White (2000). Mas conservador que SPA."""
    return _spa_core(d, n_boot, block_length, seed, "reality_check")


# ------------------------------------------------------- validacion cruzada
def spa_crosscheck(d: np.ndarray, n_boot: int = 2000, block_length: float | None = None,
                   seed: int = 0) -> dict:
    """Compara nuestro SPA con `arch.bootstrap.SPA` (Kevin Sheppard).

    Por que existe esta funcion en vez de sustituir la implementacion propia:

      * `arch` es la implementacion de REFERENCIA del autor del contraste, y es
        el mejor detector de bugs disponible. Si ambas difieren mucho, la
        nuestra tiene un fallo.
      * Pero `arch.bootstrap.SPA(benchmark, models)` razona en terminos de
        PERDIDAS y de una serie de benchmark explicita, mientras que aqui el
        benchmark es la ausencia de destreza (cero) y la unidad de evaluacion es
        el rank IC por fecha. Mantener la nuestra permite fijar el bootstrap
        estacionario con longitud de bloque derivada de la autocorrelacion de
        los IC y devolver `best_index`, que el enjambre necesita.

    Mapeo: perdida = -IC (menor perdida = mejor), benchmark = 0.

    Devuelve un dict con ambos p-valores y su diferencia. `arch` es opcional:
    si no esta instalado, se indica en `available`.
    """
    own = spa_test(d, n_boot=n_boot, block_length=block_length, seed=seed)
    try:
        from arch.bootstrap import SPA as ArchSPA
    except ImportError:
        return {"available": False, "p_own": own.p_value, "p_arch": None, "abs_diff": None,
                "note": "arch no instalado: pip install -e '.[stats]'"}

    x = np.asarray(d, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    x = x[np.isfinite(x).all(axis=1)]
    losses = -x                                   # perdida = -diferencial
    benchmark = np.zeros(len(losses))
    bl = _auto_block_length(x) if block_length is None else float(block_length)

    sp = ArchSPA(benchmark, losses, reps=n_boot, block_size=int(max(1, round(bl))),
                 bootstrap="stationary", seed=seed)
    sp.compute()
    p_arch = float(sp.pvalues["consistent"])
    return {"available": True, "p_own": float(own.p_value), "p_arch": p_arch,
            "abs_diff": abs(float(own.p_value) - p_arch),
            "block_length": bl, "n_periods": int(losses.shape[0]),
            "n_candidates": int(losses.shape[1])}
