"""Motor de decision: la mejor apuesta que el sistema sabe hacer, siempre.

Por que existe
--------------
`swarm_rank` contesta "¿hay evidencia estadistica de que el ranking sirva?". Si
no la hay, el veredicto es NO ACCIONABLE y la cartera queda vacia. Este modulo
contesta otra pregunta: **si hay que elegir de todos modos, ¿que se compra, en
que proporcion, cuando se entra y cuando se sale?** El veredicto se sigue
calculando y se informa al lado, pero aqui no veta.

Flujo
-----
  1. Enjambre (`swarm_rank`): 11 agentes walk-forward, combinador elegido por G2,
     filtros G0-G4. De aqui salen la puntuacion de HOY de cada accion y las
     predicciones FUERA DE MUESTRA del pasado.
  2. Para cada plazo h de la rejilla (3, 5, 10, 15, 21 y 31 sesiones), la
     distribucion predictiva de retornos: E[r_h | puntuacion] estimada con las
     predicciones fuera de muestra, su incertidumbre por bootstrap bayesiano, y
     la dependencia entre acciones por bootstrap de secciones cruzadas diarias
     (`portfolio/conviction.py`). Los costes se descuentan dentro de cada escenario.
  3. Cartera log-optima sobre el simplex completo para cada h: sin tope por
     nombre, sin tope sectorial, sin numero minimo de posiciones, todo el capital
     invertido. Puede salir el 100% en una accion, dos, o mas: lo decide la
     matematica (condiciones de Kuhn-Tucker), no una regla.
  4. Se elige el plazo de la cartera por crecimiento por sesion,
     ln(1 + equivalente cierto) / h, la misma unidad con la que se comparan los
     planes de salida.
  5. Plan de salida por accion (`exits.py`): objetivo, stop, stop movil y plazo
     maximo aprendidos de las trayectorias reales de casos con puntuacion
     parecida, validados en dos mitades temporales.
  6. Momento de entrada: apertura de la siguiente sesion (ver `ENTRY_RATIONALE`).
  7. Tabla del universo con valoracion de 0 a 1 y los catalizadores de cada accion.

Lo que no cambia
----------------
Paper por defecto. La ejecucion real sigue exigiendo SWARMTRADER_ALLOW_LIVE=1 y
la confirmacion explicita del plan concreto (`execution/plan.py`). La parada del
mandato por drawdown total sigue activa: es una salida, no un tope de
diversificacion.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .data.features import forward_returns
from .data.loaders import MarketData
from .exits import ExitConfig, PathBank, cost_from_price, plan_exit
from .market_calendar import next_session, session_open_madrid
from .portfolio.conviction import ConvictionConfig, allocate, build_bank
from .swarm_rank import SwarmRankConfig, SwarmRankResult, swarm_rank

__all__ = ["DecisionConfig", "DecisionResult", "decide", "format_decision_report",
           "ENTRY_RATIONALE"]

ENTRY_RATIONALE = (
    "Entrada en la APERTURA de la siguiente sesion, con orden de mercado de tipo "
    "day enviada antes de la apertura. Motivos: (1) la senal se calcula con el "
    "cierre, asi que el primer momento en que se puede actuar es despues; "
    "(2) operar en el after-hours con acciones fraccionarias obliga a usar "
    "ordenes limitadas en un mercado con poca liquidez y spreads amplios, y "
    "Alpaca solo acepta time_in_force=day para fraccionarias, de modo que no "
    "hay orden a la apertura (OPG) para ellas; (3) medido sobre datos reales, "
    "retrasar la entrada una sesion entera no cambia la capacidad predictiva "
    "(rank IC 0,0746 frente a 0,0749), asi que esperar a la apertura no cuesta "
    "ventaja medible y evita pagar el spread nocturno."
)


@dataclass
class DecisionConfig:
    horizons: tuple = (3, 5, 10, 15, 21, 31)
    # market_level="history": el nivel de mercado no puede salir de las fechas
    # fuera de muestra de cada plazo, porque entonces elegir plazo seria elegir
    # el ruido de mercado mas favorable (ver portfolio/conviction.py, punto 3b).
    conviction: ConvictionConfig = field(
        default_factory=lambda: ConvictionConfig(risk_aversion=1.0, allow_cash=False,
                                                 market_level="history"))
    exits: ExitConfig = field(default_factory=ExitConfig)
    report_top: int = 10
    success_threshold: float = 0.05
    vol_lookback: int = 60
    agent_strong: float = 0.6       # rango de agente (en [-1, 1]) que cuenta como catalizador


@dataclass
class DecisionResult:
    as_of: str
    entry_session: str
    entry_time_madrid: str
    mode: str
    horizon: int
    weights: pd.Series
    positions: pd.DataFrame
    top: pd.DataFrame
    horizons: pd.DataFrame
    exit_plans: dict
    alternatives: dict
    verdict: dict
    swarm: SwarmRankResult | None = None
    notes: list = field(default_factory=list)
    created_utc: str = field(default_factory=lambda: datetime.now(timezone.utc)
                             .isoformat(timespec="seconds"))

    def to_dict(self) -> dict:
        def _clean(df: pd.DataFrame) -> list:
            return [{k: (None if isinstance(v, float) and not np.isfinite(v) else v)
                     for k, v in r.items()} for r in df.round(6).to_dict(orient="records")]
        return {
            "created_utc": self.created_utc, "as_of": self.as_of,
            "entry_session": self.entry_session, "entry_time_madrid": self.entry_time_madrid,
            "entry_rationale": ENTRY_RATIONALE, "mode": self.mode,
            "horizon_sessions": self.horizon,
            "weights": {k: round(float(v), 6) for k, v in self.weights.items()},
            "positions": _clean(self.positions), "top": _clean(self.top),
            "horizons": _clean(self.horizons),
            "exit_plans": {k: v.to_dict() for k, v in self.exit_plans.items()},
            "alternatives": {k: round(float(v), 6) for k, v in self.alternatives.items()},
            "verdict": self.verdict, "notes": self.notes,
        }


# ----------------------------------------------------------------- utilidades
def _forward(md: MarketData, h: int, listings) -> pd.Series:
    if listings is not None:
        from .data.pit import forward_returns_pit
        return forward_returns_pit(md.prices, listings, h).rename(f"fwd_{h}")
    return forward_returns(md.prices, h)


def _oos_at(oos: pd.DataFrame, fwd: pd.Series) -> pd.DataFrame:
    o = oos[["date", "symbol", "score"]].copy()
    o["date"] = pd.to_datetime(o["date"])
    f = fwd.rename("fwd").reset_index()
    o = o.merge(f, on=["date", "symbol"], how="left")
    return o[np.isfinite(o["fwd"].to_numpy(dtype=float))]


def _catalysts(row: pd.Series, agents: list[str], strong: float) -> str:
    pro = [a for a in agents if np.isfinite(row.get(a, np.nan)) and row[a] >= strong]
    con = [a for a in agents if np.isfinite(row.get(a, np.nan)) and row[a] <= -strong]
    parts = []
    if pro:
        parts.append("a favor: " + ", ".join(pro))
    if con:
        parts.append("en contra: " + ", ".join(con))
    return "; ".join(parts) if parts else "sin agente destacado"


# ------------------------------------------------------------------- decision
def decide(md: MarketData, swarm_cfg: SwarmRankConfig = SwarmRankConfig(),
           cfg: DecisionConfig = DecisionConfig(), *, swarm_result: SwarmRankResult | None = None,
           held: dict | None = None, exclude=(), listings=None, extra=None, groups=None,
           now_utc: datetime | None = None, verbose: bool = False) -> DecisionResult:
    """Cartera, plazo, entrada y salidas para la ultima sesion de `md`."""
    # El enjambre se ejecuta en modo "gated": la asignacion la hace esta funcion
    # para cada plazo; pedirle tambien la suya seria calcular dos veces lo mismo.
    res = swarm_result or swarm_rank(md, replace(swarm_cfg, decision="gated"), extra=extra,
                                     groups=groups, listings=listings, verbose=verbose)
    signal = pd.Timestamp(res.as_of).date()
    entry = next_session(signal)
    now = now_utc or datetime.now(timezone.utc)
    entry_dt = session_open_madrid(entry)
    notes: list[str] = []
    if now.astimezone(entry_dt.tzinfo) > entry_dt:
        notes.append(f"la apertura de {entry} ya ha pasado ({now:%Y-%m-%d %H:%M} UTC): "
                     "la entrada sera a precio de mercado del momento de ejecucion")
    verdict = res.verdict.to_dict()

    empty = DecisionResult(str(signal), str(entry), entry_dt.isoformat(), "conviction", 0,
                           pd.Series(dtype=float), pd.DataFrame(), pd.DataFrame(),
                           pd.DataFrame(), {}, {}, verdict, res, notes)
    if res.oos is None or res.oos.empty or res.picks.empty:
        empty.notes.append("el enjambre no produjo predicciones fuera de muestra: no hay "
                           "con que estimar la distribucion de retornos")
        return empty

    picks = res.picks.copy()
    agents = [a for a in res.agents["agent"]] if not res.agents.empty else []
    scores = picks.set_index("symbol")["score"].astype(float)
    last_px = picks.set_index("symbol")["last_price"].astype(float)
    if md.nominal is not None:
        nom = md.nominal.iloc[-1].reindex(last_px.index)
        last_px_nom = nom.fillna(last_px)
    else:
        last_px_nom = last_px
    costs = cost_from_price(last_px_nom)
    exclude = set(exclude)

    # ---- 2-4. cartera log-optima para cada plazo y eleccion del plazo ----
    rows, allocs, banks, stand = [], {}, {}, {}
    for h in cfg.horizons:
        oos_h = _oos_at(res.oos, _forward(md, h, listings))
        if oos_h.empty:
            continue
        try:
            bank = build_bank(oos_h, scores, md.prices, h, cfg.conviction, costs,
                              seed_offset=h, held=tuple(held or {}))
        except ValueError as exc:
            notes.append(f"plazo {h}: {exc}")
            continue
        a = allocate(oos_h, scores, md.prices, h, cfg.conviction, held=held, exclude=exclude,
                     top_k=swarm_cfg.rank.top_k, success_threshold=cfg.success_threshold,
                     costs=costs, bank=bank)
        if a.weights.empty:
            notes.extend(f"plazo {h}: {n}" for n in a.notes)
            continue
        allocs[h], banks[h] = a, bank
        cols = [bank.col[s] for s in scores.index if s in bank.col]
        stand[h] = bank.standalone(cols, cfg.success_threshold)
        w = a.weights
        rows.append({"horizon": h, "growth_per_session": float(np.log1p(a.certainty_equivalent) / h),
                     "ce": a.certainty_equivalent, "n_positions": int(len(w)),
                     "top_weight": float(w.max()),
                     "effective_positions": float(w.sum() ** 2 / (w ** 2).sum()),
                     "gap": a.gap, "map_bins": a.info.get("n_map_bins", 0),
                     "map_dates": a.info.get("n_map_dates", 0),
                     # fraccion del efecto de la puntuacion que sobrevive al
                     # encogimiento jerarquico (0 = la puntuacion no mueve E[r])
                     "signal_kept": float(np.mean(a.info.get("shrinkage", [0.0]))),
                     "growth_se": float(a.info.get("log_wealth_se", 0.0) / h)})
    if not rows:
        empty.notes.extend(["ningun plazo produjo una cartera"] + notes)
        return empty
    htab = pd.DataFrame(rows).sort_values("horizon").reset_index(drop=True)
    # Plazo: el de mayor crecimiento por sesion. Si varios quedan dentro del error
    # de Monte Carlo del mejor, no hay forma de distinguirlos con estos escenarios
    # y se elige el MAS LARGO: rota menos, y el coste es la parte del modelo con
    # mas incertidumbre (tabla por tramos, sin cotizaciones reales).
    k = htab["growth_per_session"].idxmax()
    tol = float(htab.loc[k, "growth_se"])
    near = htab[htab["growth_per_session"] >= htab.loc[k, "growth_per_session"] - tol]
    h_star = int(near["horizon"].max())
    alloc = allocs[h_star]
    weights = alloc.weights

    # ---- 7. tabla del universo: valoracion 0-1 por crecimiento por sesion ----
    g = pd.DataFrame({h: stand[h]["mean_log"] / h for h in stand})
    best_h = g.idxmax(axis=1)
    best_g = g.max(axis=1)
    uni = pd.DataFrame({"best_horizon": best_h.astype(int), "growth_per_session": best_g})
    uni["valoracion"] = best_g.rank(pct=True, method="average")
    for c in ("mu", "ce", "p_up", "p_loss", "q10", "q90"):
        uni[c] = [float(stand[int(h)].loc[s, c]) for s, h in best_h.items()]
    uni["score_pct"] = scores.rank(pct=True).reindex(uni.index)
    uni["last_price"] = last_px_nom.reindex(uni.index)
    uni["cost_rt"] = costs.reindex(uni.index)
    pk = picks.set_index("symbol")
    uni["catalysts"] = [_catalysts(pk.loc[s], agents, cfg.agent_strong) if s in pk.index
                        else "" for s in uni.index]
    uni = uni.sort_values("growth_per_session", ascending=False)
    top = uni.head(cfg.report_top).reset_index().rename(columns={"index": "symbol"})
    top.insert(0, "rank", np.arange(1, len(top) + 1))

    # ---- 5. planes de salida ----
    pbank = PathBank(md.prices, res.oos, cfg.exits)
    lr = np.log(md.prices).diff().tail(cfg.vol_lookback)
    sig = lr.std()
    plans = {}
    for s in list(dict.fromkeys(list(weights.index) + list(top["symbol"]))):
        if not np.isfinite(sig.get(s, np.nan)) or s not in scores.index:
            continue
        plans[s] = plan_exit(pbank, s, float(uni.loc[s, "score_pct"]), float(sig[s]),
                             float(last_px.get(s)), float(costs.get(s, 0.0)))
    if pbank.n == 0:
        notes.append("sin trayectorias historicas cerradas: planes de salida no estimables")

    # ---- tabla de posiciones ----
    pos = []
    for s, w in weights.items():
        p = plans.get(s)
        pos.append({
            "symbol": s, "weight": float(w), "valoracion": float(uni.loc[s, "valoracion"]),
            "score_pct": float(uni.loc[s, "score_pct"]),
            "exp_return_h": float(stand[h_star].loc[s, "mu"]) if s in stand[h_star].index else np.nan,
            "p_up_h": float(stand[h_star].loc[s, "p_up"]) if s in stand[h_star].index else np.nan,
            "last_price": float(last_px_nom.get(s, np.nan)),
            "exit_horizon": p.horizon if p else np.nan,
            "target_pct": p.target_pct if p else np.nan, "stop_pct": p.stop_pct if p else np.nan,
            "trail_pct": p.trail_pct if p else np.nan,
            "catalysts": uni.loc[s, "catalysts"],
        })
    positions = pd.DataFrame(pos)

    if not res.verdict.actionable:
        notes.append("evidencia estadistica: NO ACCIONABLE (" +
                     "; ".join(r.split(":")[0] for r in res.verdict.reasons) +
                     "). Seleccion por esperanza condicionada, sin ventaja demostrada")
    notes.extend(alloc.notes)
    return DecisionResult(
        as_of=str(signal), entry_session=str(entry), entry_time_madrid=entry_dt.isoformat(),
        mode="conviction", horizon=h_star, weights=weights, positions=positions, top=top,
        horizons=htab, exit_plans=plans, alternatives=alloc.alternatives, verdict=verdict,
        swarm=res, notes=notes,
    )


# -------------------------------------------------------------------- informe
def _pct(x: float, signed: bool = True) -> str:
    if x is None or not np.isfinite(x):
        return "—"
    return f"{x:+.1%}" if signed else f"{x:.1%}"


def format_decision_report(d: DecisionResult) -> str:
    v = d.verdict
    L = [f"ANALISIS {d.as_of}   entrada: apertura del {d.entry_session} "
         f"({d.entry_time_madrid[11:16]} hora de Madrid)",
         f"Evidencia estadistica del enjambre: "
         f"{'ACCIONABLE' if v.get('actionable') else 'NO ACCIONABLE'}"
         f"   (G1 p={v.get('p_skill', 1):.3f}, G4 p={v.get('p_topk', 1):.3f}, "
         f"evidencia {v.get('evidence')})", ""]
    if not d.top.empty:
        L.append(f"TOP {len(d.top)} DEL UNIVERSO (valoracion 0 = peor, 1 = mejor)")
        L.append("   #  accion     valor  plazo  ret.esp  P(>+5%)  P(perder)  catalizadores")
        for _, r in d.top.iterrows():
            L.append(f"  {int(r['rank']):2d}  {r['symbol']:<9} {r['valoracion']:.3f}  "
                     f"{int(r['best_horizon']):>3}s  {_pct(r['mu']):>7}  {r['p_up']:6.0%}"
                     f"   {r['p_loss']:7.0%}    {r['catalysts']}")
        L.append("")
    if not d.horizons.empty:
        L.append("PLAZO DE LA CARTERA (crecimiento por sesion del optimo log)")
        for _, r in d.horizons.iterrows():
            mark = "  <- elegido" if int(r["horizon"]) == d.horizon else ""
            L.append(f"   {int(r['horizon']):>2} sesiones: {r['growth_per_session']*100:+.3f}%/sesion"
                     f"   {int(r['n_positions'])} posiciones   mayor peso "
                     f"{r['top_weight']:.0%}   senal retenida {r['signal_kept']:.0%}{mark}")
        L.append("")
    if not d.positions.empty:
        L.append(f"CARTERA (100% del capital disponible; plazo de cartera {d.horizon} sesiones)")
        for _, r in d.positions.iterrows():
            L.append(f"   {r['symbol']:<9} {r['weight']:6.1%}   valoracion {r['valoracion']:.3f}"
                     f"   ret.esperado {_pct(r['exp_return_h'])}   P(>+5%) {r['p_up_h']:.0%}")
        L.append("")
        L.append("PLAN DE SALIDA POR ACCION (comprobado al cierre; venta en la apertura siguiente)")
        for s in d.positions["symbol"]:
            p = d.exit_plans.get(s)
            if p is None:
                continue
            from .market_calendar import add_sessions
            last_exit = add_sessions(pd.Timestamp(d.as_of).date(), p.horizon + 1)
            L.append(f"   {s:<9} plazo max {p.horizon:>2} sesiones (venta como tarde en la apertura "
                     f"del {last_exit})")
            L.append(f"             objetivo {_pct(p.target_pct)}"
                     + (f" ({p.target_price:.2f})" if np.isfinite(p.target_price) else "")
                     + f"   stop {_pct(p.stop_pct)}"
                     + (f" ({p.stop_price:.2f})" if p.stop_price > 0 else "")
                     + f"   stop movil {_pct(-p.trail_pct) if np.isfinite(p.trail_pct) else '—'}"
                     f" desde maximo")
            L.append(f"             casos: {p.n_paths}  sale por objetivo {p.p_target:.0%}, "
                     f"stop {p.p_stop:.0%}, movil {p.p_trail:.0%}, plazo {p.p_vertical:.0%}; "
                     f"mediana {p.median_hold:.0f} sesiones; {p.source}")
        L.append("")
    if d.alternatives:
        L.append("ALTERNATIVAS (equivalente cierto por plazo, mismos escenarios)")
        for k, x in d.alternatives.items():
            L.append(f"   {k:<44} {_pct(x)}")
        L.append("")
    for n in d.notes:
        L.append(f"  - {n}")
    return "\n".join(L)
