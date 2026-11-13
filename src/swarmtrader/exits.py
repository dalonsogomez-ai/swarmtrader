"""Plan de salida por accion: cuando vender, aprendido de casos parecidos.

Que decide
----------
Para cada accion que se compra: objetivo, stop, stop movil y plazo maximo
(entre 3 y 31 sesiones, es decir, hasta unos 45 dias naturales). No hay un plazo
unico para todas: una accion puede salir con plan de 3 sesiones y otra con plan
de 21, segun lo que hicieron los casos historicos comparables.

De donde salen los casos comparables
------------------------------------
De las predicciones walk-forward FUERA DE MUESTRA del propio enjambre. Para una
accion que hoy esta en el percentil u de la puntuacion, los casos son las
(fecha, accion) historicas que estaban en el mismo tramo de percentil en su dia,
con su trayectoria de precios REAL de las 31 sesiones siguientes. Solo se usan
ventanas ya cerradas: nada de lo que se mira ocurrio despues de hoy.

Para que una trayectoria de un valor tranquilo sirva para uno volatil, cada
trayectoria se expresa en unidades de la volatilidad diaria que tenia ese valor
al entrar (z = ln(P_t/P_0) / sigma). Las barreras se definen en esas mismas
unidades, multiplicadas por sqrt(T): una barrera de 1,5 es +1,5 sigma sqrt(T),
que en un valor con 2% de volatilidad diaria y T = 21 son +13,7%, y en uno con
6% son +41%. Asi el mismo plan es comparable entre valores.

El criterio: crecimiento por sesion (teorema de renovacion-recompensa)
-----------------------------------------------------------------------
Comparar un plan de 3 sesiones con uno de 21 por su retorno medio es comparar
cosas distintas: el de 3 sesiones libera el capital siete veces antes. Si el
capital liberado se reinvierte en una operacion de la misma calidad, que es lo
que hace el ciclo diario, la tasa de crecimiento a largo plazo es

        g = E[ ln(R neto por operacion) ] / E[ sesiones por operacion ]

(teorema de renovacion-recompensa). Se maximiza g. Los costes entran en R neto,
asi que un plan corto, que paga el spread mas a menudo, solo gana si su ventaja
por sesion lo compensa.

Contra el sobreajuste
---------------------
La rejilla es pequena y FIJA (no se ajusta tras mirar resultados): 6 plazos x
5 objetivos x 4 stops x 3 stops moviles = 360 planes. Elegir el mejor de 360
sobre la misma muestra infla su resultado, asi que se valida por validacion
cruzada en dos mitades temporales: se elige en una mitad y se mide en la otra.
El procedimiento completo compite con uno mas simple, que solo elige el plazo
(sin objetivo ni stops). Si el completo no gana fuera de muestra con
probabilidad de al menos el 95% (bootstrap bayesiano por fechas), se usa el
simple. Es una regla fijada de antemano, no una opinion.

Lo que no hace
--------------
No usa maximos intradia: las barreras se comprueban sobre cierres, que es lo
que el ciclo diario puede observar y ejecutar. Sin datos de deslistados, una
trayectoria que se interrumpe se congela en su ultimo precio, lo que favorece a
los planes largos en valores que acabaron mal.
"""

from __future__ import annotations

import zlib
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

__all__ = ["ExitConfig", "ExitPlan", "PathBank", "plan_exit", "cost_from_price"]

INF = float("inf")


@dataclass
class ExitConfig:
    verticals: tuple = (3, 5, 10, 15, 21, 31)       # 31 sesiones ~ 45 dias naturales
    targets: tuple = (1.0, 1.5, 2.0, 3.0, INF)      # en unidades de sigma*sqrt(T)
    stops: tuple = (1.0, 1.5, 2.0, INF)
    trails: tuple = (INF, 1.5, 2.5)
    vol_lookback: int = 60
    bandwidth: float = 0.10                          # ancho del tramo de percentil
    min_paths: int = 300
    max_paths: int = 6000                            # submuestreo determinista si hay mas
    adopt_prob: float = 0.95                         # P(rejilla completa mejora) para adoptarla
    n_boot: int = 400

    @property
    def max_horizon(self) -> int:
        return int(max(self.verticals))


def cost_from_price(price: pd.Series) -> pd.Series:
    """Coste de ida y vuelta por tramo de precio (fraccion del nominal).

    Alpaca no cobra comision en acciones; el coste es el spread. Sin cotizaciones
    bid/ask historicas se usa la misma tabla por tramos que el resto del
    proyecto: 0,05% a partir de 5 USD, 1% entre 1 y 5 USD, 3% por debajo de 1.
    """
    p = price.astype(float)
    return pd.Series(np.select([p >= 5.0, p >= 1.0], [0.0005, 0.01], 0.03), index=p.index)


@dataclass
class ExitPlan:
    symbol: str
    horizon: int                    # plazo maximo en sesiones
    target_pct: float               # inf si no hay objetivo
    stop_pct: float                 # -inf si no hay stop
    trail_pct: float                # inf si no hay stop movil (caida desde el maximo)
    target_price: float
    stop_price: float
    entry_ref_price: float
    daily_vol: float
    growth_per_session: float       # E[ln R neto] / E[sesiones]
    mean_log_return: float
    mean_return: float
    median_hold: float
    p_target: float
    p_stop: float
    p_trail: float
    p_vertical: float
    p_win: float
    n_paths: int
    source: str                     # "rejilla completa" | "solo plazo"
    crossfit: dict = field(default_factory=dict)
    policy_z: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        for k, v in d.items():
            if isinstance(v, float) and not np.isfinite(v):
                d[k] = None
        return d


class PathBank:
    """Trayectorias reales posteriores a cada prediccion fuera de muestra."""

    def __init__(self, prices: pd.DataFrame, oos: pd.DataFrame, cfg: ExitConfig = ExitConfig()):
        self.cfg = cfg
        px = prices.sort_index()
        K = cfg.max_horizon
        o = oos.dropna(subset=["score"]).copy()
        o["date"] = pd.to_datetime(o["date"])
        o = o[o["symbol"].isin(px.columns)]
        o["pct"] = o.groupby("date")["score"].rank(pct=True)
        pos = pd.Series(np.arange(len(px)), index=px.index)
        d_pos = pos.reindex(o["date"]).to_numpy()
        ok = np.isfinite(d_pos) & (d_pos + K <= len(px) - 1)
        o, d_pos = o[ok], d_pos[ok].astype(int)
        cols = pd.Series(np.arange(px.shape[1]), index=px.columns)
        s_pos = cols.reindex(o["symbol"]).to_numpy().astype(int)

        logp = np.log(px.ffill(limit=None).to_numpy(dtype=float))
        lr = np.diff(logp, axis=0, prepend=np.nan)
        sd = pd.DataFrame(lr).rolling(cfg.vol_lookback, min_periods=20).std().to_numpy()
        base = logp[d_pos, s_pos]
        steps = np.arange(1, K + 1)
        path = logp[d_pos[:, None] + steps[None, :], s_pos[:, None]] - base[:, None]
        sig = sd[d_pos, s_pos]
        good = np.isfinite(base) & np.isfinite(sig) & (sig > 1e-4) & np.isfinite(path).all(axis=1)
        self.z = (path[good] / sig[good, None]).astype(np.float64)
        self.pct = o["pct"].to_numpy()[good]
        self.dates = o["date"].to_numpy()[good]
        self.n = int(good.sum())

    def neighbours(self, u: float) -> np.ndarray:
        """Indices de los casos en el tramo de percentil de `u` (se ensancha si faltan)."""
        bw = self.cfg.bandwidth
        while True:
            idx = np.where(np.abs(self.pct - u) <= bw / 2)[0]
            if len(idx) >= self.cfg.min_paths or bw >= 1.0:
                break
            bw = min(1.0, bw * 1.5)
        if len(idx) > self.cfg.max_paths:
            step = len(idx) / self.cfg.max_paths
            idx = idx[(np.arange(self.cfg.max_paths) * step).astype(int)]
        return idx


def _simulate(Z: np.ndarray, T: int, up: float, dn: float, tr: float):
    """Primera barrera tocada en cierres. Devuelve (z de salida, sesiones, motivo)."""
    z = Z[:, :T]
    peak = np.maximum.accumulate(np.maximum(z, 0.0), axis=1)
    hit_up = z >= up
    hit_dn = z <= dn
    hit_tr = (peak - z >= tr) & (peak > 0)
    anyhit = hit_up | hit_dn | hit_tr
    first = np.where(anyhit.any(axis=1), anyhit.argmax(axis=1), T - 1)
    rows = np.arange(len(z))
    zx = z[rows, first]
    why = np.full(len(z), 3)                         # 3 = vertical
    hit = anyhit.any(axis=1)
    why[hit & hit_tr[rows, first]] = 2
    why[hit & hit_dn[rows, first]] = 1
    why[hit & hit_up[rows, first]] = 0
    return zx, first + 1, why


def _grid(cfg: ExitConfig, vertical_only: bool):
    if vertical_only:
        return [(T, INF, INF, INF) for T in cfg.verticals]
    return [(T, ku, kd, kt) for T in cfg.verticals for ku in cfg.targets
            for kd in cfg.stops for kt in cfg.trails]


def _growth(Z: np.ndarray, pol, sigma: float, cost: float) -> float:
    T, ku, kd, kt = pol
    rt = np.sqrt(T)
    zx, held, _ = _simulate(Z, T, ku * rt, -kd * rt, kt * rt)
    r = sigma * zx + np.log1p(-cost)
    return float(r.mean() / held.mean())


def _best(Z: np.ndarray, grid, sigma: float, cost: float):
    g = np.array([_growth(Z, p, sigma, cost) for p in grid])
    k = int(np.nanargmax(g))
    return grid[k], float(g[k])


def plan_exit(bank: PathBank, symbol: str, percentile: float, sigma: float, price: float,
              cost: float) -> ExitPlan:
    """Plan de salida para una accion con percentil de puntuacion `percentile`.

    sigma  volatilidad diaria de los log-retornos de la accion hoy
    price  ultimo cierre (precio de referencia de la entrada)
    cost   coste de ida y vuelta (fraccion)
    """
    cfg = bank.cfg
    idx = bank.neighbours(float(percentile))
    Z, dts = bank.z[idx], bank.dates[idx]
    full, simple = _grid(cfg, False), _grid(cfg, True)

    # Validacion cruzada temporal en dos mitades: cada procedimiento elige su plan
    # en una mitad y se mide en la otra. Las trayectorias de una misma fecha
    # comparten el movimiento del mercado, asi que la incertidumbre de la
    # diferencia se mide con bootstrap bayesiano POR FECHAS (no por trayectoria).
    # La rejilla completa solo se adopta si mejora al procedimiento simple con
    # probabilidad >= `cfg.adopt_prob`; con una superficie plana (ruido) su
    # mejor plan es suerte y gana la regla simple.
    di = dts.astype("datetime64[ns]").astype(np.int64)
    A = di <= np.quantile(di, 0.5)
    cf = {"full": 0.0, "simple": 0.0, "p_mejora": 0.0}
    usable = all(m.sum() >= 30 for m in (A, ~A))
    if usable:
        rng = np.random.default_rng(zlib.crc32(symbol.encode()))
        draws = np.zeros(cfg.n_boot)
        for fit, ev in ((A, ~A), (~A, A)):
            ud, inv = np.unique(di[ev], return_inverse=True)
            w = rng.dirichlet(np.ones(len(ud)), size=cfg.n_boot)      # B x fechas
            per = {}
            for name, grid in (("full", full), ("simple", simple)):
                pol_h, _ = _best(Z[fit], grid, sigma, cost)
                T, ku, kd, kt = pol_h
                zx, held, _ = _simulate(Z[ev], T, ku * np.sqrt(T), -kd * np.sqrt(T), kt * np.sqrt(T))
                r = sigma * zx + np.log1p(-cost)
                R = np.bincount(inv, weights=r, minlength=len(ud))
                H = np.bincount(inv, weights=held, minlength=len(ud))
                cf[name] += 0.5 * float(R.sum() / H.sum())
                per[name] = (w @ R) / (w @ H)
            draws += 0.5 * (per["full"] - per["simple"])
        cf["p_mejora"] = float((draws > 0).mean())
    use_full = usable and cf["p_mejora"] >= cfg.adopt_prob
    pol, g = _best(Z, full if use_full else simple, sigma, cost)

    T, ku, kd, kt = pol
    rt = np.sqrt(T)
    zx, held, why = _simulate(Z, T, ku * rt, -kd * rt, kt * rt)
    r = sigma * zx + np.log1p(-cost)
    tgt = float(np.expm1(ku * rt * sigma)) if np.isfinite(ku) else INF
    stp = float(np.expm1(-kd * rt * sigma)) if np.isfinite(kd) else -INF
    trl = float(-np.expm1(-kt * rt * sigma)) if np.isfinite(kt) else INF
    return ExitPlan(
        symbol=symbol, horizon=int(T), target_pct=tgt, stop_pct=stp, trail_pct=trl,
        target_price=float(price * (1 + tgt)) if np.isfinite(tgt) else INF,
        stop_price=float(price * (1 + stp)) if np.isfinite(stp) else 0.0,
        entry_ref_price=float(price), daily_vol=float(sigma),
        growth_per_session=float(g), mean_log_return=float(r.mean()),
        mean_return=float(np.expm1(r).mean()), median_hold=float(np.median(held)),
        p_target=float((why == 0).mean()), p_stop=float((why == 1).mean()),
        p_trail=float((why == 2).mean()), p_vertical=float((why == 3).mean()),
        p_win=float((r > 0).mean()), n_paths=int(len(idx)),
        source="rejilla completa" if use_full else "solo plazo",
        crossfit={k: round(v, 6) for k, v in cf.items()} | {"evaluada": usable},
        policy_z={"T": T, "target": ku, "stop": kd, "trail": kt},
    )
