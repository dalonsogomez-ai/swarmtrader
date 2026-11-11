"""Calibracion de probabilidades en numpy puro.

Un ranker devuelve scores, no probabilidades. Para responder "con que
probabilidad sube esta accion mas de X% en 21 sesiones" hace falta un mapa
score -> probabilidad ajustado SOLO con predicciones out-of-sample. Si el
modelo no tiene senal, la calibracion devolvera la tasa base para todo el
mundo: esa es la respuesta honesta, y es exactamente lo que queremos que pase.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _pav(y: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pool-adjacent-violators: regresion isotonica creciente ponderada.

    Devuelve (valores ajustados, soporte = n observaciones del bloque de cada punto).
    """
    n = y.size
    vals, wts, sizes = [], [], []
    for i in range(n):
        vals.append(float(y[i])); wts.append(float(w[i])); sizes.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            v = (vals[-2] * wts[-2] + vals[-1] * wts[-1]) / (wts[-2] + wts[-1])
            wts[-2] += wts[-1]; sizes[-2] += sizes[-1]
            vals[-2] = v; vals.pop(); wts.pop(); sizes.pop()
    out, sup = np.empty(n), np.empty(n)
    k = 0
    for v, s in zip(vals, sizes):
        out[k:k + s] = v; sup[k:k + s] = s; k += s
    return out, sup


@dataclass
class IsotonicCalibrator:
    x_knots: np.ndarray = field(default_factory=lambda: np.array([0.0, 1.0]))
    y_knots: np.ndarray = field(default_factory=lambda: np.array([0.5, 0.5]))
    base_rate: float = 0.5
    fitted: bool = False
    n: int = 0
    shrink_m: float = 10.0   # pseudo-observaciones hacia la tasa base por bloque

    def fit(self, scores: np.ndarray, outcomes: np.ndarray, min_samples: int = 50) -> "IsotonicCalibrator":
        s = np.asarray(scores, float); y = np.asarray(outcomes, float)
        m = np.isfinite(s) & np.isfinite(y)
        s, y = s[m], y[m]
        self.n = s.size
        self.base_rate = float(y.mean()) if s.size else 0.5
        if s.size < min_samples or y.min() == y.max():
            self.fitted = False
            return self
        order = np.argsort(s, kind="stable")
        s, y = s[order], y[order]
        iso, sup = _pav(y, np.ones_like(y))
        # shrinkage bayesiano hacia la tasa base: un bloque con 3 observaciones al
        # 100% NO debe traducirse en "P=100%". Cuanto menos soporte, mas se encoge.
        iso = (sup * iso + self.shrink_m * self.base_rate) / (sup + self.shrink_m)
        # el shrinkage por bloque puede romper la monotonia: segunda pasada PAV
        # ponderada por el soporte efectivo de cada bloque
        iso, _ = _pav(iso, sup + self.shrink_m)
        # comprime a knots unicos para interpolar
        xs, ys = [], []
        for xv, yv in zip(s, iso):
            if not xs or xv > xs[-1]:
                xs.append(xv); ys.append(yv)
            else:
                ys[-1] = yv
        self.x_knots, self.y_knots = np.asarray(xs), np.asarray(ys)
        self.fitted = True
        return self

    def predict(self, scores: np.ndarray) -> np.ndarray:
        s = np.atleast_1d(np.asarray(scores, float))
        if not self.fitted:
            return np.full(s.shape, self.base_rate)
        return np.clip(np.interp(s, self.x_knots, self.y_knots), 0.0, 1.0)

    def to_dict(self) -> dict:
        return {"fitted": self.fitted, "n": self.n, "base_rate": round(self.base_rate, 4),
                "x_knots": self.x_knots.round(5).tolist(), "y_knots": self.y_knots.round(5).tolist()}


def expected_calibration_error(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    p = np.asarray(p, float); y = np.asarray(y, float)
    bins = np.clip((p * n_bins).astype(int), 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            ece += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(ece)
