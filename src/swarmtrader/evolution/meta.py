"""Meta-labeling (Lopez de Prado, Advances in Financial ML, cap. 3).

El modelo primario (los pesos del genoma) decide QUE comprar. El meta-modelo
decide SI actuar y CUANTO, aprendiendo de los propios picks pasados del agente:
dada la configuracion en el momento del pick (fuerza del score, dispersion,
regimen de mercado), cual fue la probabilidad de que acabase en beneficio.

Es la palanca mas eficaz de precision en un top-k: sacrifica recall (opera
menos) a cambio de precision (acierta mas cuando opera). La abstencion deja de
ser un umbral fijo y pasa a ser una probabilidad calibrada.

Implementado en numpy puro: regresion logistica L2 con descenso de gradiente.
Se entrena SOLO con picks anteriores a la ventana que se va a evaluar.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

META_FEATURE_NAMES: tuple[str, ...] = (
    "score",            # score del pick
    "score_rel",        # score menos mediana de los elegibles ese dia
    "score_gap",        # distancia al siguiente no elegido (fuerza del corte)
    "n_picks",          # cuantos superaron el umbral (proxy de dispersion)
    "mkt_vol_z",        # regimen: vol de mercado vs su historia
    "mkt_ret_63",       # regimen: tendencia de mercado
)


@dataclass
class PickRecord:
    t: int
    symbol_idx: int
    features: np.ndarray       # len(META_FEATURE_NAMES)
    outcome: int | None = None # 1 exito, 0 fracaso, None sin resolver


@dataclass
class MetaLabeler:
    l2: float = 2.0
    min_samples: int = 25
    epochs: int = 300
    lr: float = 0.1
    mu: np.ndarray = field(default_factory=lambda: np.zeros(len(META_FEATURE_NAMES)))
    sd: np.ndarray = field(default_factory=lambda: np.ones(len(META_FEATURE_NAMES)))
    w: np.ndarray = field(default_factory=lambda: np.zeros(len(META_FEATURE_NAMES)))
    b: float = 0.0
    fitted: bool = False
    n_train: int = 0
    base_rate: float = 0.5

    def fit(self, records: list[PickRecord]) -> "MetaLabeler":
        rows = [(r.features, r.outcome) for r in records if r.outcome is not None]
        if len(rows) < self.min_samples:
            self.fitted = False
            return self
        X = np.vstack([r[0] for r in rows]).astype(float)
        y = np.asarray([r[1] for r in rows], dtype=float)
        self.base_rate = float(y.mean())
        if y.min() == y.max():          # una sola clase: no hay nada que aprender
            self.fitted = False
            return self
        self.mu = X.mean(axis=0)
        self.sd = np.where(X.std(axis=0) > 1e-9, X.std(axis=0), 1.0)
        Z = (X - self.mu) / self.sd
        n, d = Z.shape
        w, b = np.zeros(d), float(np.log(self.base_rate / (1 - self.base_rate)))
        for _ in range(self.epochs):
            p = 1.0 / (1.0 + np.exp(-(Z @ w + b)))
            g = p - y
            w -= self.lr * (Z.T @ g / n + self.l2 * w / n)
            b -= self.lr * g.mean()
        self.w, self.b, self.fitted, self.n_train = w, b, True, n
        return self

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        F = np.atleast_2d(np.asarray(features, dtype=float))
        if not self.fitted:
            return np.full(F.shape[0], self.base_rate)
        Z = (F - self.mu) / self.sd
        return 1.0 / (1.0 + np.exp(-(Z @ self.w + self.b)))

    def to_dict(self) -> dict:
        return {"fitted": self.fitted, "n_train": self.n_train, "base_rate": round(self.base_rate, 4),
                "weights": {k: round(float(v), 4) for k, v in zip(META_FEATURE_NAMES, self.w)},
                "bias": round(self.b, 4)}


def triple_barrier_outcome(path_returns: np.ndarray, take_profit: float, stop_loss: float) -> int:
    """1 si el camino toca take-profit antes que stop-loss (o acaba en positivo)."""
    if path_returns.size == 0:
        return 0
    eq = np.cumprod(1.0 + path_returns) - 1.0
    hit_tp = np.where(eq >= take_profit)[0]
    hit_sl = np.where(eq <= -stop_loss)[0]
    t_tp = hit_tp[0] if hit_tp.size else np.inf
    t_sl = hit_sl[0] if hit_sl.size else np.inf
    if t_tp < t_sl:
        return 1
    if t_sl < t_tp:
        return 0
    return int(eq[-1] > 0)
