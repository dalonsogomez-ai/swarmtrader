"""Combinacion de agentes sin fuga de informacion.

El error clasico del stacking en series temporales es ajustar los pesos del
meta-modelo con TODAS las predicciones OOS y despues evaluar sobre esas mismas
predicciones. Los scores base son out-of-sample, pero los PESOS no: se han
elegido sabiendo que agente funciono en el periodo que luego se evalua. El
resultado es un ensemble que parece batir a todos sus componentes y que en
produccion no bate a ninguno.

Aqui los pesos de la fecha d se ajustan **solo con fechas estrictamente
anteriores a d** (ventana expansiva). El primer tramo, hasta tener historia
suficiente, usa equiponderacion.

Se calculan tres competidores, y los tres se evaluan con la misma regla:

  * ``eq``          media de los rangos de los agentes. Linea base naive.
  * ``best_prior``  el mejor agente segun IC pasado, reelegido en cada fecha.
                    Es la version honesta de "me quedo con el que va mejor".
  * ``stack``       pesos no negativos por minimos cuadrados sobre el rango.

Si `stack` no bate a `eq`, la complejidad no se ha ganado el sitio y el sistema
debe reportar `eq`. Esa comparacion es obligatoria, no opcional.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats as _st
from scipy.optimize import nnls

__all__ = ["StackConfig", "StackResult", "stack_agents", "per_date_ic"]


@dataclass
class StackConfig:
    min_train_dates: int = 6        # fechas necesarias antes de ajustar pesos
    ridge: float = 1.0              # regularizacion hacia equiponderacion
    weight_floor: float = 0.0       # pesos negativos prohibidos (NNLS)
    max_weight: float = 0.60        # ningun agente domina el conjunto
    # Multiplicar la puntuacion por el acuerdo entre agentes era una heuristica sin
    # validar. Medida sobre datos reales (experiments/geometry_audit.py, D1) no
    # mejora el IC (p=1,00) y lo empeora ligeramente (p=0,027). Queda desactivada;
    # la columna `agreement` se sigue calculando como informacion.
    shrink_on_disagreement: bool = False


@dataclass
class StackResult:
    scores: pd.DataFrame            # date, symbol, fwd, success, eq, best_prior, stack, agreement
    weights: pd.DataFrame           # date x agente
    agents: list[str]
    config: StackConfig = field(default_factory=StackConfig)

    def final_weights(self) -> pd.Series:
        if self.weights.empty:
            return pd.Series(dtype=float)
        return self.weights.iloc[-1].sort_values(ascending=False)


def _rank_norm(s: pd.Series) -> pd.Series:
    n = s.notna().sum()
    if n < 2:
        return pd.Series(0.0, index=s.index)
    r = s.rank(method="average", na_option="keep")
    return ((r - 1.0) / (n - 1.0) * 2.0 - 1.0).fillna(0.0)


def per_date_ic(df: pd.DataFrame, score_col: str, fwd_col: str = "fwd",
                min_names: int = 5) -> pd.Series:
    """Rank IC de Spearman por fecha. Es la unidad de evaluacion de todo el sistema."""
    return df.groupby("date").apply(
        lambda g: _st.spearmanr(g[score_col], g[fwd_col]).statistic
        if g[score_col].notna().sum() >= min_names and g[fwd_col].notna().sum() >= min_names
        else np.nan,
        include_groups=False).dropna()


def _fit_weights(train: pd.DataFrame, agents: list[str], cfg: StackConfig) -> np.ndarray:
    """NNLS con encogimiento hacia equiponderacion.

    Se anaden filas de regularizacion sqrt(ridge)*I con objetivo sqrt(ridge)/k:
    equivale a un prior que dice "sin evidencia, todos los agentes pesan igual".
    """
    k = len(agents)
    y = train["y"].to_numpy(float)
    X = train[agents].to_numpy(float)
    if len(y) < k + 2 or not np.isfinite(X).all():
        return np.full(k, 1.0 / k)
    lam = float(max(cfg.ridge, 0.0))
    if lam > 0:
        X = np.vstack([X, np.sqrt(lam) * np.eye(k)])
        y = np.concatenate([y, np.sqrt(lam) * np.full(k, 1.0 / k)])
    try:
        w, _ = nnls(X, y)
    except Exception:
        return np.full(k, 1.0 / k)
    if not np.isfinite(w).all() or w.sum() <= 1e-12:
        return np.full(k, 1.0 / k)
    w = w / w.sum()
    # Tope por agente (water-filling). El exceso se reparte UNIFORMEMENTE entre
    # los no topados: repartirlo de forma proporcional no progresa cuando esos
    # pesos son cero, y renormalizar al final deshace el tope. Ambos errores
    # producen un unico agente con peso 1.0, que es justo lo que el tope existe
    # para impedir.
    cap = float(cfg.max_weight)
    if cap < 1.0:
        if cap * k < 1.0 - 1e-12:          # tope infactible: no hay reparto valido
            return np.full(k, 1.0 / k)
        for _ in range(32):
            over = w > cap + 1e-12
            if not over.any():
                break
            excess = float((w[over] - cap).sum())
            w[over] = cap
            free = ~over
            if not free.any():
                return np.full(k, 1.0 / k)
            w[free] += excess / float(free.sum())
        w = np.clip(w, 0.0, cap)
    return w


def stack_agents(panel_scores: pd.DataFrame, agents: list[str],
                 cfg: StackConfig = StackConfig()) -> StackResult:
    """Combina los agentes con pesos ajustados solo sobre el pasado."""
    if panel_scores.empty or not agents:
        return StackResult(pd.DataFrame(), pd.DataFrame(), list(agents), cfg)

    df = panel_scores.sort_values(["date", "symbol"]).copy()
    # objetivo del meta-modelo: rango cross-seccional del retorno futuro
    df["y"] = df.groupby("date")["fwd"].transform(_rank_norm)

    dates = list(pd.unique(df["date"]))
    out_blocks, weight_rows = [], []

    for i, d in enumerate(dates):
        cur = df[df["date"] == d].copy()
        prior = df[df["date"].isin(dates[:i])]

        if i < cfg.min_train_dates or prior.empty:
            w = np.full(len(agents), 1.0 / len(agents))
            best = agents[0]
        else:
            w = _fit_weights(prior.dropna(subset=["y"]), agents, cfg)
            ic_prior = {a: float(per_date_ic(prior, a).mean() or 0.0) for a in agents}
            best = max(ic_prior, key=ic_prior.get)

        A = cur[agents].to_numpy(float)
        cur["eq"] = A.mean(axis=1)
        cur["best_prior"] = cur[best].to_numpy(float)
        cur["stack"] = A @ w

        # desacuerdo: dispersion del rango del mismo valor entre agentes.
        # 0 = agentes de acuerdo, 1 = maximo desacuerdo.
        disp = A.std(axis=1) / (2.0 / np.sqrt(3.0))   # sd de U(-1,1) como escala
        cur["agreement"] = np.clip(1.0 - disp, 0.0, 1.0)
        if cfg.shrink_on_disagreement:
            cur["stack"] = cur["stack"] * cur["agreement"]

        out_blocks.append(cur)
        weight_rows.append(pd.Series(w, index=agents, name=d))

    scores = pd.concat(out_blocks, ignore_index=True)
    weights = pd.DataFrame(weight_rows)
    keep = ["date", "symbol", "fwd", "success", "eq", "best_prior", "stack", "agreement"] + agents
    return StackResult(scores[[c for c in keep if c in scores.columns]], weights, list(agents), cfg)
