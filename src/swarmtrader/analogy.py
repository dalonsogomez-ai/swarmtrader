"""Motor de casos historicos analogos.

Que problema resuelve
---------------------
"Una accion que ya tenia comprada sube un 120% en una sesion: ¿vendo, aguanto o
amplio?" El ranker del enjambre no responde a eso. El ranker ordena el universo
ANTES de entrar; esto decide cuanto tiempo quedarse DESPUES de entrar, y es una
pregunta distinta con una respuesta distinta.

La respuesta correcta no es una regla fija ("vende al +50%") porque el mismo
+50% significa cosas opuestas segun el contexto: en un valor liquido tras
resultados suele continuar; en un microcap sin volumen tras un pump suele
revertir entero. La respuesta es empirica: **busca situaciones pasadas
parecidas y mira que paso despues**, incluyendo las que salieron mal.

Como se define "parecido"
-------------------------
Un descriptor de 9 dimensiones, todas **adimensionales** para que un +120% en
una accion de 2 $ sea comparable con uno en una de 80 $:

  run_up_5 / run_up_21   magnitud del movimiento reciente
  accel                  cuanto del movimiento es de los ultimos dias
  vol_surge              volumen frente a su propia media de 60 sesiones
  rvol_20                volatilidad realizada frente a su media anual
  dd_from_peak           caida desde el maximo de la ventana
  dist_52w               distancia al maximo de 52 semanas
  turnover_z             liquidez relativa
  gap_open               hueco de apertura del dia del evento

Los descriptores se convierten a **rango cross-seccional por fecha** antes de
comparar. Es la misma normalizacion que usa el resto del sistema y tiene una
razon concreta: hace la distancia inmune a cambios de regimen. Un vol_surge de
3x en 2017 y en 2020 no significan lo mismo en absoluto, pero "estar en el
percentil 95 de vol_surge de su dia" si es comparable.

Disciplina point-in-time (la parte que hay que hacer bien)
----------------------------------------------------------
Al consultar en la fecha `d`, solo son admisibles los analogos cuya ventana de
resultado **ya habia terminado antes de d**. Si se permite un analogo de hace
10 sesiones para estimar un horizonte de 21, se esta usando informacion que en
`d` no existia. Es la fuga mas facil de cometer aqui y la que hace que estos
motores parezcan magicos en backtest y fallen en vivo. `_admissible` lo impone y
`tests/test_analogy.py` lo comprueba.

Con que regla se mide la distancia (`AnalogyConfig.metric`)
----------------------------------------------------------
  euclid        euclidea ponderada sobre los percentiles. Es la regla por
                defecto y la unica validada operativamente.
  fisher_rank   la misma idea en la coordenada de Fisher de cada percentil,
                theta = 2 arcsin(sqrt u). Mide la diferencia en desviaciones
                tipicas de error muestral, lo que estira las colas: del
                percentil 99 al 99,9 pesa mas de 7 veces lo que del 50 al 50,9.
                Sin parametros libres.
  fisher_gauss  distancia de Rao entre las gaussianas (media, desviacion) de
                los retornos diarios de las ultimas 21 sesiones de cada caso.
                Invariante a escala; ignora volumen, drawdown y el resto de
                descriptores.

Las dos reglas de Fisher son COMPETIDORAS, no mejoras: cambiar el valor por
defecto exige que `experiments/infogeo.py` rechace con datos reales la hipotesis
de que ninguna bate a la euclidea, corrigiendo por haber probado dos.

Que devuelve
------------
No devuelve una decision, devuelve una **distribucion empirica**: retorno
mediano a +1/+5/+10/+21 sesiones, fraccion de casos que continuaron subiendo,
maxima subida adicional y maximo drawdown posteriores, y el horizonte en el que
la mediana toco su maximo. Mas el tamano muestral efectivo: con menos de
`min_analogues` casos no emite recomendacion, porque una mediana de 4 casos no
es evidencia de nada.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .stats.infogeo import bernoulli_coordinate, fisher_rao_gaussian

__all__ = ["AnalogyConfig", "AnalogyResult", "EventDescriptor",
           "build_descriptors", "find_analogues", "analogue_distances",
           "HOLD_HORIZONS", "METRICS"]

METRICS = ("euclid", "fisher_rank", "fisher_gauss")
GAUSS_COLS = ("_mu_21", "_sd_21")
_SD_FLOOR = 1e-4

HOLD_HORIZONS = (1, 5, 10, 21, 32)

DESCRIPTOR_COLS = (
    "run_up_5", "run_up_21", "accel", "vol_surge", "rvol_20",
    "dd_from_peak", "dist_52w", "turnover_z", "gap_open",
)


@dataclass
class AnalogyConfig:
    k: int = 50                     # vecinos a recuperar
    min_analogues: int = 20         # por debajo, no se emite recomendacion
    max_horizon: int = 32           # ventana de resultado de los analogos
    embargo: int = 5                # colchon extra sobre la ventana
    same_symbol_weight: float = 1.0  # 1.0 = no privilegia la propia accion
    min_run_up: float = 0.15        # solo se analiza si hubo movimiento relevante
    weights: dict | None = None     # pesos por dimension; None = uniforme
    metric: str = "euclid"          # euclid | fisher_rank | fisher_gauss

    def __post_init__(self) -> None:
        if self.metric not in METRICS:
            raise ValueError(f"metric debe ser una de {METRICS}, no {self.metric!r}")


@dataclass
class AnalogyResult:
    n_analogues: int
    actionable: bool
    reason: str
    forward: pd.DataFrame = field(default_factory=pd.DataFrame)  # horizonte x percentiles
    p_continue: float = float("nan")      # fraccion con retorno futuro > 0 a 5 sesiones
    median_max_gain: float = float("nan")  # subida adicional mediana alcanzable
    median_max_dd: float = float("nan")    # drawdown mediano posterior
    best_horizon: int = 0                  # horizonte donde la mediana es maxima
    neighbours: pd.DataFrame = field(default_factory=pd.DataFrame)

    def summary(self) -> str:
        if not self.actionable:
            return f"casos analogos: {self.n_analogues} — NO ACCIONABLE ({self.reason})"
        L = [f"casos analogos: {self.n_analogues}",
             f"  continua al alza a 5 sesiones: {self.p_continue:.0%} de los casos",
             f"  subida adicional mediana alcanzable: {self.median_max_gain:+.1%}",
             f"  drawdown posterior mediano:          {self.median_max_dd:+.1%}",
             f"  horizonte de mediana maxima:         {self.best_horizon} sesiones", ""]
        for h, row in self.forward.iterrows():
            L.append(f"  +{h:>2} sesiones:  p25 {row['p25']:+6.1%}   mediana {row['p50']:+6.1%}"
                     f"   p75 {row['p75']:+6.1%}")
        return "\n".join(L)


class EventDescriptor:
    """Construye el descriptor adimensional de cada (fecha, simbolo)."""

    @staticmethod
    def compute(prices: pd.DataFrame, volumes: pd.DataFrame) -> pd.DataFrame:
        px = prices.sort_index()
        vol = volumes.reindex_like(px)
        out = {}

        out["run_up_5"] = px / px.shift(5) - 1.0
        out["run_up_21"] = px / px.shift(21) - 1.0
        # que fraccion del movimiento de 21 sesiones ocurrio en las ultimas 5
        with np.errstate(divide="ignore", invalid="ignore"):
            out["accel"] = out["run_up_5"] / out["run_up_21"].abs().replace(0, np.nan)

        dv = px * vol
        out["vol_surge"] = dv / dv.rolling(60, min_periods=20).mean() - 1.0
        out["turnover_z"] = ((np.log1p(dv) - np.log1p(dv).rolling(252, min_periods=60).mean())
                             / np.log1p(dv).rolling(252, min_periods=60).std())

        ret = np.log(px / px.shift(1))
        rv20 = ret.rolling(20, min_periods=10).std() * np.sqrt(252)
        rv252 = ret.rolling(252, min_periods=60).std() * np.sqrt(252)
        out["rvol_20"] = rv20 / rv252 - 1.0

        roll_max = px.rolling(21, min_periods=5).max()
        out["dd_from_peak"] = px / roll_max - 1.0
        out["dist_52w"] = px / px.rolling(252, min_periods=60).max() - 1.0
        out["gap_open"] = px / px.shift(1) - 1.0
        # parametros de la gaussiana de la ventana: solo los usa metric="fisher_gauss"
        out["_mu_21"] = ret.rolling(21, min_periods=10).mean()
        out["_sd_21"] = ret.rolling(21, min_periods=10).std()

        frames = []
        for name, df in out.items():
            s = df.stack(future_stack=True).rename(name)
            s.index.names = ["date", "symbol"]
            frames.append(s)
        d = pd.concat(frames, axis=1)
        return d.replace([np.inf, -np.inf], np.nan)


def _rank_by_date(df: pd.DataFrame, cols) -> pd.DataFrame:
    """Rango cross-seccional a [0,1] por fecha: inmuniza frente a cambios de regimen."""
    out = df.copy()
    g = out.groupby(level="date")
    for c in cols:
        r = g[c].rank(pct=True)
        out[c] = r
    return out


def build_descriptors(prices: pd.DataFrame, volumes: pd.DataFrame) -> pd.DataFrame:
    raw = EventDescriptor.compute(prices, volumes)
    ranked = _rank_by_date(raw, DESCRIPTOR_COLS)
    ranked["_run_up_5_raw"] = raw["run_up_5"]
    return ranked


def analogue_distances(query: np.ndarray, pool: np.ndarray, weights: np.ndarray,
                       metric: str = "euclid", query_gauss: np.ndarray | None = None,
                       pool_gauss: np.ndarray | None = None) -> np.ndarray:
    """Distancia del caso consultado a cada candidato, con la regla elegida.

    `query` (d,) y `pool` (n, d) son percentiles cross-seccionales en [0, 1].
    `*_gauss` son (media, desviacion) de los retornos diarios de la ventana y
    solo se usan con metric="fisher_gauss".
    """
    if metric == "euclid":
        return np.sqrt(((pool - query) ** 2 * weights).sum(axis=1))
    if metric == "fisher_rank":
        return np.sqrt(((bernoulli_coordinate(pool) - bernoulli_coordinate(query)) ** 2
                        * weights).sum(axis=1))
    if metric == "fisher_gauss":
        if query_gauss is None or pool_gauss is None:
            raise ValueError("fisher_gauss necesita la media y la desviacion de cada ventana")
        return fisher_rao_gaussian(pool_gauss[:, 0], np.maximum(pool_gauss[:, 1], _SD_FLOOR),
                                   query_gauss[0], max(float(query_gauss[1]), _SD_FLOOR))
    raise ValueError(f"metrica desconocida: {metric!r}")


def _forward_outcomes(prices: pd.DataFrame, max_h: int) -> dict:
    """Retornos futuros y extremos posteriores, por horizonte."""
    px = prices.sort_index()
    out = {}
    for h in HOLD_HORIZONS:
        if h > max_h:
            continue
        s = (px.shift(-h) / px - 1.0).stack(future_stack=True)
        s.index.names = ["date", "symbol"]
        out[f"fwd_{h}"] = s
    # maximo y minimo alcanzados dentro de la ventana (no del cierre final)
    fwd_max = px.shift(-1).rolling(max_h, min_periods=1).max().shift(-(max_h - 1))
    fwd_min = px.shift(-1).rolling(max_h, min_periods=1).min().shift(-(max_h - 1))
    mg = (fwd_max / px - 1.0).stack(future_stack=True)
    md = (fwd_min / px - 1.0).stack(future_stack=True)
    mg.index.names = md.index.names = ["date", "symbol"]
    out["max_gain"] = mg
    out["max_dd"] = md
    return out


def _admissible(candidate_dates: pd.Index, as_of, max_h: int, embargo: int,
                sessions: pd.Index) -> np.ndarray:
    """True para los analogos cuya ventana de resultado terminó ANTES de `as_of`.

    Esta es la regla que hace el motor honesto. Un analogo de la fecha c solo es
    utilizable en d si c + max_h + embargo <= d, porque su resultado no se
    conocia hasta entonces.
    """
    pos = pd.Series(np.arange(len(sessions)), index=sessions)
    d_pos = int(pos.get(pd.Timestamp(as_of), len(sessions) - 1))
    c_pos = pos.reindex(candidate_dates).to_numpy()
    return (c_pos + max_h + embargo) <= d_pos


def find_analogues(prices: pd.DataFrame, volumes: pd.DataFrame, symbol: str, as_of,
                   cfg: AnalogyConfig = AnalogyConfig()) -> AnalogyResult:
    """Busca situaciones pasadas parecidas a la de `symbol` en `as_of`."""
    px = prices.sort_index()
    sessions = px.index
    as_of = pd.Timestamp(as_of)
    if as_of not in sessions:
        return AnalogyResult(0, False, f"{as_of.date()} no es sesion del panel")

    desc = build_descriptors(px, volumes)
    try:
        query = desc.loc[(as_of, symbol)]
    except KeyError:
        return AnalogyResult(0, False, f"{symbol} sin descriptor en {as_of.date()}")
    if query[list(DESCRIPTOR_COLS)].isna().any():
        return AnalogyResult(0, False, "descriptor incompleto (historico insuficiente)")

    run_up = float(query["_run_up_5_raw"])
    if not np.isfinite(run_up) or abs(run_up) < cfg.min_run_up:
        return AnalogyResult(0, False,
                             f"movimiento de {run_up:+.1%} por debajo del umbral "
                             f"({cfg.min_run_up:+.0%}): no hay evento que analizar")

    need = list(DESCRIPTOR_COLS) + (list(GAUSS_COLS) if cfg.metric == "fisher_gauss" else [])
    if query[need].isna().any():
        return AnalogyResult(0, False, "descriptor incompleto (historico insuficiente)")
    pool = desc.dropna(subset=need)
    cand_dates = pool.index.get_level_values("date")
    ok = _admissible(cand_dates, as_of, cfg.max_horizon, cfg.embargo, sessions)
    pool = pool[ok]
    if pool.empty:
        return AnalogyResult(0, False, "sin historico admisible point-in-time")

    w = np.ones(len(DESCRIPTOR_COLS))
    if cfg.weights:
        w = np.array([float(cfg.weights.get(c, 1.0)) for c in DESCRIPTOR_COLS])
    w = w / w.sum()

    Q = query[list(DESCRIPTOR_COLS)].to_numpy(float)
    M = pool[list(DESCRIPTOR_COLS)].to_numpy(float)
    dist = analogue_distances(Q, M, w, cfg.metric,
                              query_gauss=query[list(GAUSS_COLS)].to_numpy(float),
                              pool_gauss=pool[list(GAUSS_COLS)].to_numpy(float))

    if cfg.same_symbol_weight != 1.0:
        same = pool.index.get_level_values("symbol") == symbol
        dist = np.where(same, dist * cfg.same_symbol_weight, dist)

    k = min(cfg.k, len(dist))
    idx = np.argpartition(dist, k - 1)[:k]
    nb = pool.iloc[idx].copy()
    nb["distance"] = dist[idx]
    nb = nb.sort_values("distance")

    outcomes = _forward_outcomes(px, cfg.max_horizon)
    for name, s in outcomes.items():
        nb[name] = s.reindex(nb.index).to_numpy()

    valid = nb.dropna(subset=["fwd_5"])
    n = len(valid)
    if n < cfg.min_analogues:
        return AnalogyResult(n, False,
                             f"solo {n} analogos con resultado observable "
                             f"(minimo {cfg.min_analogues})", neighbours=nb)

    rows = {}
    for h in HOLD_HORIZONS:
        col = f"fwd_{h}"
        if col not in valid:
            continue
        v = valid[col].dropna()
        if len(v) < cfg.min_analogues // 2:
            continue
        rows[h] = {"p25": v.quantile(0.25), "p50": v.median(), "p75": v.quantile(0.75),
                   "n": len(v)}
    fwd = pd.DataFrame(rows).T
    best_h = int(fwd["p50"].idxmax()) if not fwd.empty else 0

    return AnalogyResult(
        n_analogues=n, actionable=True, reason="muestra suficiente",
        forward=fwd,
        p_continue=float((valid["fwd_5"] > 0).mean()),
        median_max_gain=float(valid["max_gain"].median()),
        median_max_dd=float(valid["max_dd"].median()),
        best_horizon=best_h,
        neighbours=valid[["distance"] + [c for c in valid.columns if c.startswith("fwd_")]],
    )
