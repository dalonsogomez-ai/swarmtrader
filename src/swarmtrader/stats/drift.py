"""Monitor de deriva: ¿el mercado de hoy esta dentro de lo que vio el modelo?

El problema, y por que no es el obvio
-------------------------------------
La idea ingenua es comparar la distribucion de los factores de hoy con la del
periodo de entrenamiento. **En este sistema eso no detecta nada**: los factores
se normalizan a z-score por fecha (`data/features.py`), asi que su media es 0 y
su desviacion 1 todos los dias por construccion (salvo el recorte a +-3, que
solo deja pasar un resto de la forma de las colas). Un monitor sobre las
marginales de los factores seria ciego exactamente a lo que tiene que ver.

La normalizacion por fecha borra la escala pero deja intactas dos cosas, y esas
son las que se vigilan. Cada una es un punto en una variedad estadistica y se
mide con su distancia de Fisher-Rao (`stats/infogeo.py`):

  **dispersion** — lo que el z-score ELIMINA. La seccion cruzada de retornos a
  21 sesiones de cada dia se resume en (mediana, MAD escalada): un punto de la
  variedad gaussiana. Un mismo z=+2 significa +8% en un mes tranquilo y +40% en
  uno de crisis; el modelo recibe el mismo numero en los dos casos.

  **dependencia** — lo que el z-score NO puede eliminar. La matriz de
  correlacion cross-seccional entre factores de cada dia: un punto de la
  variedad de matrices definidas positivas. Si momentum y baja volatilidad
  pasan de ser independientes a moverse juntos, el modelo lineal entrenado con
  la estructura anterior extrapola.

Como se decide
--------------
No se usa un umbral absoluto de distancia, porque la distancia incluye el error
de estimacion de cada dia y este depende del tamano del universo. El umbral se
calibra solo: se calcula la distancia de cada fecha de entrenamiento al centro
del propio periodo, y el dia evaluado esta "fuera de soporte" si su distancia
supera el cuantil 1 - alpha de esas. Los dos canales reparten alpha por
Bonferroni.

alpha es NOMINAL. Medido en `experiments/infogeo.py` sobre datos estacionarios,
con alpha = 1% el monitor dispara el 1,9% de las fechas. La causa es que los
estados estan muy autocorrelacionados (se construyen con ventanas de 21 a 252
sesiones), de modo que tres anos de referencia contienen pocas decenas de
observaciones independientes y no pueden fijar un cuantil del 99%. Sobre 1.000
acciones reales (2018-2026) marco el 3,2% de las fechas, en tres episodios.

Que puede y que no puede hacer
------------------------------
Detecta cambio en la distribucion de las ENTRADAS (covariate shift). No detecta
que cambie la relacion entre entradas y retorno futuro con las entradas
intactas (concept shift): eso solo lo ve el IC fuera de muestra, tarde.

Es lento y solo ve rupturas grandes: una volatilidad x3 se detecta en el 95% de
los casos a las 5 sesiones, una x2 en el 50% a las 21, y una x1,5 no se
distingue de la linea base. El canal de dependencia necesita un universo ancho
(con 50 nombres apenas detecta nada).

Solo puede volver el veredicto MAS conservador. No genera senal, no elige
acciones y no entra en la comparacion multiple de G1-G3. Su valor economico NO
esta demostrado: en los datos reales disponibles solo 2 fechas de evaluacion
cayeron en dias marcados, asi que no se pudo contrastar si el modelo rinde peor
ahi. El coste si es conocido: abstenerse en torno al 2-3% de los dias, incluidos
regimenes nuevos que podrian haber sido favorables. `mode="warn"` informa sin vetar.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .infogeo import fisher_rao_gaussian, fisher_rao_spd, shrink_to_identity

__all__ = ["DriftConfig", "DriftReport", "MarketStates", "compute_states", "drift_at",
           "market_state_drift"]

CHANNELS = ("dispersion", "dependencia")


@dataclass
class DriftConfig:
    alpha: float = 0.01        # tasa conjunta de falsa alarma; 0 desactiva el monitor
    mode: str = "gate"         # gate = veta el veredicto | warn = solo informa
    smooth: int = 5            # mediana movil de la distancia (sesiones)
    ret_window: int = 21       # ventana del retorno cuya seccion cruzada se resume
    min_ref_dates: int = 120   # fechas de referencia minimas para calibrar
    min_names: int = 20        # nombres minimos por fecha para estimar un estado

    def __post_init__(self) -> None:
        if self.mode not in ("gate", "warn"):
            raise ValueError(f"mode debe ser 'gate' o 'warn', no {self.mode!r}")


@dataclass
class DriftReport:
    evaluated: bool
    out_of_support: bool
    reason: str
    p_value: float = float("nan")          # minimo de los canales, corregido por Bonferroni
    channels: dict = field(default_factory=dict)
    n_ref: int = 0

    def to_dict(self) -> dict:
        return {
            "evaluated": self.evaluated, "out_of_support": self.out_of_support,
            "reason": self.reason,
            "p_value": None if not np.isfinite(self.p_value) else round(float(self.p_value), 4),
            "n_ref": self.n_ref,
            "channels": {k: {kk: round(float(vv), 4) for kk, vv in v.items()}
                         for k, v in self.channels.items()},
        }


@dataclass
class MarketStates:
    """Estado del mercado por fecha. Cada fila usa solo datos de esa fecha o anteriores."""

    dates: pd.DatetimeIndex
    loc: np.ndarray            # mediana cross-seccional del retorno a `ret_window`
    scale: np.ndarray          # 1.4826 * MAD
    corr: np.ndarray           # (T, k, k) correlacion entre factores, encogida
    n_names: np.ndarray
    factor_cols: list


def compute_states(df: pd.DataFrame, prices: pd.DataFrame, cfg: DriftConfig = DriftConfig(),
                   factor_cols: list | None = None) -> MarketStates:
    """Resume cada fecha del panel en un punto de cada variedad.

    `df` es el panel (date, symbol) ya filtrado por elegibilidad; `prices` es
    ancho (fecha x simbolo). El estado de la fecha t se calcula con la seccion
    cruzada de los simbolos elegibles en t.
    """
    cols = list(factor_cols if factor_cols is not None else df.attrs.get("factor_cols", []))
    cols = [c for c in cols if c in df.columns]
    px = prices.sort_index()
    r = (px / px.shift(cfg.ret_window) - 1.0).stack(future_stack=True)
    r.index.names = ["date", "symbol"]
    ret = r.reindex(df.index).to_numpy(float)

    codes, dates = pd.factorize(df.index.get_level_values("date"), sort=True)
    order = np.argsort(codes, kind="stable")
    bounds = np.searchsorted(codes[order], np.arange(len(dates) + 1))
    X = df[cols].to_numpy(float) if cols else np.empty((len(df), 0))
    k, T = len(cols), len(dates)

    loc = np.full(T, np.nan)
    scale = np.full(T, np.nan)
    n_names = np.zeros(T, dtype=int)
    corr = np.full((T, k, k), np.nan)
    for t in range(T):
        idx = order[bounds[t]:bounds[t + 1]]
        n_names[t] = len(idx)
        if len(idx) < cfg.min_names:
            continue
        v = ret[idx]
        v = v[np.isfinite(v)]
        if len(v) >= cfg.min_names:
            med = np.median(v)
            mad = 1.4826 * np.median(np.abs(v - med))
            if mad > 0:
                loc[t], scale[t] = med, mad
        if k >= 2:
            with np.errstate(divide="ignore", invalid="ignore"):
                c = np.corrcoef(X[idx], rowvar=False)
            corr[t] = shrink_to_identity(c, len(idx))
    return MarketStates(pd.DatetimeIndex(dates), loc, scale, corr, n_names, cols)


def _smooth(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return x
    return pd.Series(x).rolling(w, min_periods=1).median().to_numpy()


def _empirical_p(ref: np.ndarray, value: float) -> float:
    ref = ref[np.isfinite(ref)]
    return float((1 + (ref >= value).sum()) / (1 + len(ref)))


def drift_at(states: MarketStates, ref_dates, as_of, cfg: DriftConfig = DriftConfig()) -> DriftReport:
    """Compara el estado en `as_of` con el del periodo de referencia.

    `ref_dates` deben ser anteriores a `as_of`: el centro y el umbral se
    calculan solo con ellas.
    """
    if cfg.alpha <= 0:
        return DriftReport(False, False, "monitor desactivado (alpha=0)")
    as_of = pd.Timestamp(as_of)
    pos = states.dates.get_indexer([as_of])[0]
    if pos < 0:
        return DriftReport(False, False, f"{as_of.date()} sin estado calculado")
    ref_pos = states.dates.get_indexer(pd.DatetimeIndex(ref_dates))
    ref_pos = np.unique(ref_pos[(ref_pos >= 0) & (ref_pos < pos)])

    channels: dict = {}
    # -------- dispersion: variedad gaussiana (hiperbolica) --------
    ok = ref_pos[np.isfinite(states.scale[ref_pos])]
    if len(ok) >= cfg.min_ref_dates and np.isfinite(states.scale[pos]):
        # centro robusto en las coordenadas naturales (mu, ln sigma)
        mu0 = float(np.median(states.loc[ok]))
        s0 = float(np.exp(np.median(np.log(states.scale[ok]))))
        d = _smooth(fisher_rao_gaussian(states.loc, states.scale, mu0, s0), cfg.smooth)
        channels["dispersion"] = {"distance": float(d[pos]), "p_value": _empirical_p(d[ok], d[pos]),
                                  "ref_scale": s0, "today_scale": float(states.scale[pos])}
    # -------- dependencia: variedad de matrices definidas positivas --------
    k = states.corr.shape[1]
    if k >= 2:
        has = np.isfinite(states.corr).all(axis=(1, 2))
        ok = ref_pos[has[ref_pos]]
        if len(ok) >= cfg.min_ref_dates and has[pos]:
            center = states.corr[ok].mean(axis=0)
            d = np.full(len(states.dates), np.nan)
            d[has] = fisher_rao_spd(center, states.corr[has])
            d = _smooth(d, cfg.smooth)
            channels["dependencia"] = {"distance": float(d[pos]),
                                       "p_value": _empirical_p(d[ok], d[pos])}

    if not channels:
        return DriftReport(False, False,
                           f"referencia insuficiente (< {cfg.min_ref_dates} fechas con estado)",
                           n_ref=int(len(ref_pos)))
    m = len(channels)
    p = min(1.0, m * min(c["p_value"] for c in channels.values()))
    fired = [name for name, c in channels.items() if c["p_value"] < cfg.alpha / m]
    if fired:
        parts = [f"{n} (distancia {channels[n]['distance']:.2f}, p={channels[n]['p_value']:.4f})"
                 for n in fired]
        reason = "estado de mercado fuera del soporte de entrenamiento: " + "; ".join(parts)
    else:
        reason = "estado de mercado dentro del soporte de entrenamiento"
    return DriftReport(True, bool(fired), reason, p, channels, int(len(ref_pos)))


def market_state_drift(df: pd.DataFrame, prices: pd.DataFrame, train_years: float,
                       cfg: DriftConfig = DriftConfig()) -> DriftReport:
    """Monitor para la ULTIMA fecha del panel frente a su ventana de entrenamiento.

    La referencia son las mismas fechas que usa el modelo final: las etiquetadas
    dentro de `train_years`. Entre la ultima fecha etiquetada y hoy hay al menos
    un horizonte de hueco; si el mercado cambio en ese hueco, es justo lo que
    este monitor tiene que ver.
    """
    if cfg.alpha <= 0:
        return DriftReport(False, False, "monitor desactivado (alpha=0)")
    if df.empty:
        return DriftReport(False, False, "panel vacio")
    dates = df.index.get_level_values("date")
    last = dates.max()
    labelled = dates[df["target"].notna().to_numpy()].unique()
    cutoff = last - pd.Timedelta(days=int(365 * train_years))
    ref = labelled[(labelled >= cutoff) & (labelled < last)]
    return drift_at(compute_states(df, prices, cfg), ref, last, cfg)
