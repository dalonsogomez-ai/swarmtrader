"""Orquestador del enjambre: de un universo a una cartera, o a un "no".

Pipeline
--------
    features (solo pasado)
      -> N agentes especialistas, cada uno walk-forward purgado
      -> combinacion con pesos ajustados solo sobre el pasado
      -> TRES filtros estadisticos, todos obligatorios
      -> calibracion isotonica de la probabilidad
      -> dimensionamiento de posiciones
      -> informe

Los tres filtros, y por que ninguno sobra
-----------------------------------------
  **G1 — ¿hay destreza en alguna parte del enjambre?**
  SPA de Hansen sobre los N agentes mas los combinadores, frente a la ausencia
  de destreza. Corrige por haber mirado a todos: sin esto, el mejor de 9
  agentes siempre parece bueno.

  **G2 — ¿la complejidad se gana el sitio?**
  SPA del combinador aprendido frente a la equiponderacion. Si no la bate, el
  sistema REPORTA la equiponderacion. Es la unica forma de que anadir agentes
  no sea automaticamente una mejora aparente.

  **G3 — ¿el resultado sobrevive a la seleccion?**
  PBO por validacion cruzada combinatoriamente simetrica sobre la matriz de IC
  por agente: mide con que frecuencia el agente que gana dentro de muestra cae
  por debajo de la mediana fuera de muestra. PBO alto = el ranking de agentes
  es ruido, aunque G1 haya pasado.

Si cualquiera falla, el veredicto es NO ACCIONABLE y el dimensionador devuelve
una cartera vacia. Eso no es un fallo del sistema: es el sistema funcionando.

  **G4 — ¿los K primeros, que son lo unico que se compra, baten al universo?**
  G1 mide el IC sobre TODA la seccion cruzada. Un ranking puede tener IC alto
  porque distingue bien a los peores y ser inutil en la parte alta, que es la
  unica que una cartera solo-largo monetiza. Sobre datos reales ocurrio
  exactamente eso: IC 0,075 (t = 4,6), todo el poder predictivo en la mitad
  baja del ranking, y los 5 primeros sin exceso sobre el universo (t = -0,3).
  G4 contrasta el exceso de retorno fuera de muestra de los `top_k` primeros
  frente a la media del universo elegible. Es un contraste, no una seleccion:
  solo puede vetar.

  **G0 — ¿el mercado de hoy se parece al que vio el modelo?**
  G1-G3 juzgan el historico fuera de muestra; ninguno mira si HOY es
  extrapolacion. `stats/drift.py` compara el estado del mercado de la ultima
  fecha con el de la ventana de entrenamiento mediante distancias de
  Fisher-Rao. No es un contraste de destreza ni entra en la comparacion
  multiple: solo puede vetar.

Dos modos de decision
---------------------
  ``gated`` (por defecto): lo de arriba. Sin veredicto accionable no se asigna.
  ``conviction``: el veredicto se calcula y se informa IGUAL, pero no veta. Se
  asigna la cartera de crecimiento optimo sobre todo el universo elegible
  (`portfolio/conviction.py`), sin topes por nombre ni por sector: puede ser una
  sola accion o varias, segun lo que justifiquen los escenarios. Responde a "si
  hay que elegir de todos modos, ¿que es lo mejor con lo que se sabe?", que es
  una pregunta distinta de "¿hay ventaja demostrada?". El informe lo deja claro.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .agents.specialists import DEFAULT_ROSTER, Specialist, agent_diagnostics, run_swarm
from .data.features import compute_regime
from .data.loaders import MarketData
from .ensemble.stacking import StackConfig, per_date_ic, stack_agents
from .portfolio.conviction import ConvictionConfig, ConvictionResult, allocate
from .portfolio.sizing import SizingConfig, size_positions
from .rank import RankConfig, _make_model, _prepare, diagnose
from .stats.calibration import IsotonicCalibrator
from .stats.drift import DriftConfig, DriftReport, market_state_drift
from .stats.overfitting import probability_of_backtest_overfitting
from .stats.spa import spa_test

__all__ = ["topk_excess_by_date", "SwarmRankConfig", "SwarmVerdict", "SwarmRankResult", "swarm_rank", "format_swarm_report"]

EVIDENCE_QUALITY = {"FUERTE": 1.0, "DEBIL": 0.35, "NINGUNA": 0.0}
DECISION_MODES = ("gated", "conviction")


@dataclass
class SwarmRankConfig:
    rank: RankConfig = field(default_factory=RankConfig)
    stack: StackConfig = field(default_factory=StackConfig)
    sizing: SizingConfig = field(default_factory=SizingConfig)
    roster: tuple[Specialist, ...] = DEFAULT_ROSTER
    deep: object = None                 # DeepConfig; None = por defecto
    allow_deep: bool = True             # False = enjambre solo clasico
    spa_boot: int = 2000
    spa_alpha: float = 0.05
    pbo_max: float = 0.50
    drift: DriftConfig = field(default_factory=DriftConfig)
    seed: int = 0
    decision: str = "gated"             # gated | conviction (ver docstring del modulo)
    conviction: ConvictionConfig = field(default_factory=ConvictionConfig)


@dataclass
class SwarmVerdict:
    actionable: bool
    combiner: str                   # "stack" | "eq" | "none"
    reasons: list[str]
    p_skill: float
    p_stack_vs_eq: float
    pbo: float
    evidence: str
    p_topk: float = 1.0             # G4: H0 = los K primeros no baten al universo
    topk_excess: float = 0.0        # exceso medio por horizonte, fuera de muestra

    def to_dict(self) -> dict:
        return {
            "actionable": self.actionable, "combiner": self.combiner,
            "p_skill": round(self.p_skill, 4),
            "p_stack_vs_eq": round(self.p_stack_vs_eq, 4),
            "pbo": None if not np.isfinite(self.pbo) else round(self.pbo, 3),
            "p_topk": round(self.p_topk, 4), "topk_excess": round(self.topk_excess, 5),
            "evidence": self.evidence, "reasons": self.reasons,
        }


@dataclass
class SwarmRankResult:
    as_of: str
    horizon: int
    picks: pd.DataFrame
    portfolio: pd.DataFrame
    verdict: SwarmVerdict
    diagnostics: object
    agents: pd.DataFrame
    weights: pd.Series
    skipped: dict
    regime: dict
    n_universe: int
    n_eligible: int
    drift: DriftReport | None = None
    decision: str = "gated"
    conviction: ConvictionResult | None = None
    # Predicciones walk-forward FUERA DE MUESTRA del combinador del veredicto
    # ("stack" o "eq"; existe aunque no sea accionable): date, symbol, score, fwd.
    # Solo ventanas de retorno ya cerradas. Las usan la asignacion por conviccion
    # y `decision.py` para estimar E[r | puntuacion] sin mirar el futuro.
    oos: pd.DataFrame | None = None

    def to_dict(self) -> dict:
        return {
            "as_of": self.as_of, "horizon_sessions": self.horizon,
            "decision": self.decision,
            "conviction": None if self.conviction is None else self.conviction.to_dict(),
            "verdict": self.verdict.to_dict(),
            "diagnostics": self.diagnostics.summary(),
            "agents": self.agents.round(4).to_dict(orient="records"),
            "agent_weights": {k: round(float(v), 4) for k, v in self.weights.items()},
            "picks": self.picks.round(4).to_dict(orient="records"),
            "portfolio": self.portfolio.round(4).to_dict(orient="records"),
            "skipped_agents": self.skipped,
            "regime": {k: round(float(v), 3) for k, v in self.regime.items()},
            "n_universe": self.n_universe, "n_eligible": self.n_eligible,
            "drift": None if self.drift is None else self.drift.to_dict(),
        }


def _ic_matrix(scores: pd.DataFrame, cols: list[str]) -> tuple[np.ndarray, list[str], pd.Index]:
    """Matriz (fechas x candidatos) de rank IC. Unidad comun de evaluacion."""
    series = {c: per_date_ic(scores, c) for c in cols}
    series = {c: s for c, s in series.items() if len(s) >= 3}
    if not series:
        return np.empty((0, 0)), [], pd.Index([])
    idx = None
    for s in series.values():
        idx = s.index if idx is None else idx.intersection(s.index)
    idx = idx.sort_values()
    names = list(series.keys())
    M = np.column_stack([series[c].reindex(idx).to_numpy(float) for c in names])
    return M, names, idx


def topk_excess_by_date(scores: pd.DataFrame, col: str, k: int) -> pd.Series:
    """Retorno medio de los `k` primeros por `col` menos el del universo, por fecha.

    Es la cantidad que una cartera solo-largo de `k` nombres realmente captura.
    """
    s = scores.dropna(subset=["fwd", col])
    top = s.sort_values(["date", col]).groupby("date").tail(max(int(k), 1))
    return (top.groupby("date")["fwd"].mean() - s.groupby("date")["fwd"].mean()).dropna()


def _final_scores(df: pd.DataFrame, cfg: RankConfig, roster: list[Specialist],
                  agents: list[str], weights: pd.Series, md=None, groups=None,
                  deep_cfg=None, shrink_on_disagreement: bool = False) -> pd.DataFrame:
    """Score de la ULTIMA fecha: cada agente reentrenado con todo lo etiquetado."""
    dates = df.index.get_level_values("date").unique().sort_values()
    last = dates[-1]
    labelled = df.dropna(subset=["target"])
    cutoff = last - pd.Timedelta(days=int(365 * cfg.train_years))
    labelled = labelled.loc[labelled.index.get_level_values("date") >= cutoff]
    today = df.loc[last]
    cols = list(df.attrs.get("factor_cols", []))
    by_name = {sp.name: sp for sp in roster}

    out = pd.DataFrame({"symbol": today.index, "last_price": today["px"].to_numpy()})
    n = len(today)
    for a in agents:
        sp = by_name[a]
        fac = list(sp.available(cols))
        if sp.kind == "deep" and md is not None:
            from .models.deep import DeepConfig
            from .models.trainer import DeepPanelBuilder, deep_final_scores
            dcfg = deep_cfg or DeepConfig()
            builder = DeepPanelBuilder(md.prices, md.volumes, fac,
                                       seq_len=dcfg.seq_len, groups=groups)
            s_ser = deep_final_scores(df, cfg, builder, dcfg)
            s = s_ser.reindex(today.index).fillna(0.0).to_numpy() if len(s_ser) else np.zeros(n)
            r = pd.Series(s).rank(method="average")
            out[a] = ((r - 1) / max(n - 1, 1) * 2 - 1).to_numpy()
            continue
        if len(labelled) >= 200 and n >= 2 and fac:
            m = _make_model(cfg).fit(labelled[fac].to_numpy(float),
                                     labelled["target"].to_numpy(float))
            s = m.predict(today[fac].to_numpy(float))
        else:
            s = today[fac].to_numpy(float).mean(axis=1) if fac else np.zeros(n)
        r = pd.Series(s).rank(method="average")
        out[a] = ((r - 1) / max(n - 1, 1) * 2 - 1).to_numpy()
    A = out[agents].to_numpy(float)
    w = weights.reindex(agents).fillna(1.0 / len(agents)).to_numpy(float)
    out["eq"] = A.mean(axis=1)
    out["stack"] = A @ (w / max(w.sum(), 1e-12))
    disp = A.std(axis=1) / (2.0 / np.sqrt(3.0))
    out["agreement"] = np.clip(1.0 - disp, 0.0, 1.0)
    if shrink_on_disagreement:
        out["stack"] = out["stack"] * out["agreement"]
    return out


def swarm_rank(md: MarketData, cfg: SwarmRankConfig = SwarmRankConfig(),
               extra: pd.DataFrame | None = None, groups: pd.Series | None = None,
               previous: pd.Series | None = None, verbose: bool = False,
               listings: pd.DataFrame | None = None, held: dict | None = None,
               exclude=()) -> SwarmRankResult:
    """`held` y `exclude` solo se usan en modo `conviction`: posiciones que se
    mantienen (fracciones del patrimonio) y simbolos que no pueden comprarse."""
    if cfg.decision not in DECISION_MODES:
        raise ValueError(f"decision debe ser uno de {DECISION_MODES}, no {cfg.decision!r}")
    rc = cfg.rank
    df = _prepare(md, rc, extra=extra, groups=groups, listings=listings)
    panel = run_swarm(df, rc, roster=cfg.roster, verbose=verbose, md=md, groups=groups,
                      deep_cfg=cfg.deep, allow_deep=cfg.allow_deep)

    reg = compute_regime(md.prices).iloc[-1].to_dict()
    drift = market_state_drift(df, md.prices, rc.train_years, cfg.drift)
    n_univ = md.prices.shape[1]
    n_elig = int(df.loc[df.index.get_level_values("date").max()].shape[0]) if len(df) else 0

    if panel.empty or len(panel.agents) < 2:
        v = SwarmVerdict(False, "none", ["enjambre sin agentes evaluables"], 1.0, 1.0,
                         float("nan"), "NINGUNA")
        return SwarmRankResult(str(df.index.get_level_values("date").max().date()), rc.horizon,
                               pd.DataFrame(), pd.DataFrame(), v, diagnose(pd.DataFrame(), rc),
                               pd.DataFrame(), pd.Series(dtype=float), panel.skipped, reg,
                               n_univ, n_elig, drift)

    stacked = stack_agents(panel.scores, panel.agents, cfg.stack)
    sc = stacked.scores
    agents_diag = agent_diagnostics(panel)

    # ---------------- G1: ¿destreza en alguna parte del enjambre? ----------------
    cands = panel.agents + ["eq", "best_prior", "stack"]
    M, names, _ = _ic_matrix(sc, cands)
    if M.size and M.shape[0] >= 8:
        g1 = spa_test(M, n_boot=cfg.spa_boot, seed=cfg.seed)
        p_skill = g1.p_value
    else:
        p_skill = 1.0

    # ---------------- G2: ¿la complejidad bate a la equiponderacion? -------------
    if M.size and "stack" in names and "eq" in names and M.shape[0] >= 8:
        d = (M[:, names.index("stack")] - M[:, names.index("eq")])[:, None]
        p_stack = spa_test(d, n_boot=cfg.spa_boot, seed=cfg.seed + 1).p_value
    else:
        p_stack = 1.0

    # ---------------- G3: ¿sobrevive a la seleccion del mejor agente? ------------
    Ma, _, _ = _ic_matrix(sc, panel.agents)
    pbo = float("nan")
    if Ma.size and Ma.shape[1] >= 2:
        # CSCV necesita T >= 2*n_splits y n_splits par. Con pocas fechas OOS se
        # baja el numero de bloques en vez de devolver n/d en silencio: un filtro
        # que no se evalua es un filtro que no protege.
        T = Ma.shape[0]
        n_sp = min(8, (T // 2) // 2 * 2)
        if n_sp >= 4 and T >= 2 * n_sp:
            pbo = probability_of_backtest_overfitting(Ma, n_splits=n_sp, seed=cfg.seed).pbo

    combiner = "stack" if p_stack < cfg.spa_alpha else "eq"
    diag = diagnose(sc.rename(columns={combiner: "score"}), rc)

    # ---------------- G4: ¿los K primeros baten al universo? ----------------------
    excess = topk_excess_by_date(sc, combiner, rc.top_k)
    topk_excess = float(excess.mean()) if len(excess) else 0.0
    p_topk = (spa_test(excess.to_numpy()[:, None], n_boot=cfg.spa_boot, seed=cfg.seed + 2).p_value
              if len(excess) >= 8 else 1.0)

    reasons: list[str] = []
    if p_skill >= cfg.spa_alpha:
        reasons.append(f"G1 destreza: SPA p={p_skill:.3f} >= {cfg.spa_alpha} — ningun agente "
                       "bate al azar tras corregir por comparacion multiple")
    if combiner == "eq":
        reasons.append(f"G2 complejidad: el stack no bate a equiponderar (p={p_stack:.3f}); "
                       "se reporta la linea base equiponderada")
    if np.isfinite(pbo) and pbo > cfg.pbo_max:
        reasons.append(f"G3 overfitting: PBO={pbo:.2f} > {cfg.pbo_max} — el orden de los "
                       "agentes no persiste fuera de muestra")
    if diag.evidence == "NINGUNA":
        reasons.append("evidencia OOS NINGUNA en este universo")
    if p_topk >= cfg.spa_alpha:
        reasons.append(f"G4 cartera: los {rc.top_k} primeros no baten al universo fuera de muestra "
                       f"(exceso medio {topk_excess:+.2%} a {rc.horizon} sesiones, p={p_topk:.3f}); "
                       "el ranking puede distinguir a los peores sin saber elegir a los mejores")

    vetoed = drift.out_of_support and cfg.drift.mode == "gate"
    if vetoed:
        reasons.append(f"G0 deriva: {drift.reason} — hoy el modelo extrapola")
    actionable = (p_skill < cfg.spa_alpha and diag.evidence != "NINGUNA"
                  and not (np.isfinite(pbo) and pbo > cfg.pbo_max)
                  and p_topk < cfg.spa_alpha and not vetoed)
    if actionable and not reasons:
        reasons.append("los filtros pasan")

    # ---------------- calibracion y ranking final ----------------
    cal = IsotonicCalibrator()
    if actionable:
        cal = cal.fit(sc[combiner].to_numpy(float), sc["success"].to_numpy(float))
    else:
        cal.base_rate = diag.base_rate if diag.n_oos_obs else 0.5

    final = _final_scores(df, rc, panel.roster, panel.agents,
                          stacked.final_weights() if combiner == "stack"
                          else pd.Series(1.0, index=panel.agents),
                          md=md, groups=groups, deep_cfg=cfg.deep,
                          shrink_on_disagreement=cfg.stack.shrink_on_disagreement)
    picks = final[["symbol", "last_price", "agreement"] + panel.agents].copy()
    picks["score"] = final[combiner].to_numpy()
    picks["prob_success"] = cal.predict(picks["score"].to_numpy())
    picks = picks.sort_values("score", ascending=False).reset_index(drop=True)
    picks["rank"] = np.arange(1, len(picks) + 1)

    # ---------------- cartera ----------------
    top = picks.head(max(rc.top_k, 1)).set_index("symbol")
    rets = md.prices.pct_change()
    vol = (rets.tail(60).std() * np.sqrt(252)).reindex(top.index)
    sizing = size_positions(
        top["prob_success"], base_rate=diag.base_rate, volatilities=vol,
        evidence_quality=EVIDENCE_QUALITY.get(diag.evidence, 0.0) if actionable else 0.0,
        groups=groups, previous=previous, success_threshold=rc.success_threshold,
        cfg=cfg.sizing,
    )
    port = sizing.to_frame().reset_index().rename(columns={"index": "symbol"})

    # ---------------- asignacion por conviccion (solo si se pide) ----------------
    # Usa el MISMO combinador que el veredicto ("stack" o "eq", existe aunque no
    # sea accionable) y solo predicciones fuera de muestra con ventana ya cerrada.
    oos = (sc[["date", "symbol", combiner, "fwd"]].rename(columns={combiner: "score"})
           .dropna(subset=["score"]).reset_index(drop=True))
    conv = None
    if cfg.decision == "conviction":
        try:
            conv = allocate(oos, picks.set_index("symbol")["score"], md.prices, rc.horizon,
                            cfg.conviction, held=held, exclude=exclude, top_k=rc.top_k,
                            success_threshold=rc.success_threshold)
        except ValueError as exc:
            conv = ConvictionResult(pd.Series(dtype=float), 1.0, pd.DataFrame(), 0.0,
                                    float("inf"), notes=[f"no se pudo asignar: {exc}"])

    verdict = SwarmVerdict(actionable, combiner if actionable else "none", reasons,
                           p_skill, p_stack, pbo, diag.evidence, p_topk, topk_excess)
    return SwarmRankResult(
        as_of=str(df.index.get_level_values("date").max().date()), horizon=rc.horizon,
        picks=picks, portfolio=port, verdict=verdict, diagnostics=diag,
        agents=agents_diag, weights=stacked.final_weights(), skipped=panel.skipped,
        regime=reg, n_universe=n_univ, n_eligible=n_elig, drift=drift,
        decision=cfg.decision, conviction=conv, oos=oos,
    )


def format_swarm_report(res: SwarmRankResult, cfg: SwarmRankConfig) -> str:
    d, v = res.diagnostics, res.verdict
    L = [
        f"Fecha de analisis : {res.as_of}    horizonte: {res.horizon} sesiones    "
        f"universo: {res.n_eligible}/{res.n_universe} elegibles",
        f"Exito definido    : retorno > {cfg.rank.success_threshold:+.0%} en {res.horizon} "
        f"sesiones   tasa base: {d.base_rate:.0%}",
        "",
        f"ENJAMBRE: {len(res.agents)} agentes activos" +
        (f"   ({len(res.skipped)} inactivos)" if res.skipped else ""),
    ]
    if not res.agents.empty:
        for _, r in res.agents.iterrows():
            L.append(f"   {r['agent']:<10} IC {r['ic']:+.3f}  t={r['t']:+5.2f}  "
                     f"ICIR {r['icir']:+.2f}  ({int(r['n_dates'])} fechas)")
    if res.skipped:
        L.append("   inactivos: " + ", ".join(f"{k} ({vv})" for k, vv in res.skipped.items()))

    dr = res.drift
    if dr is None or not dr.evaluated:
        g0 = "n/d" + (f"  ({dr.reason})" if dr is not None else "")
    else:
        state = "NO PASA" if dr.out_of_support else "PASA"
        if dr.out_of_support and cfg.drift.mode == "warn":
            state = "AVISO (modo warn, no veta)"
        g0 = f"p={dr.p_value:.4f}  {state}   [" + ", ".join(
            f"{k} d={v['distance']:.2f}" for k, v in dr.channels.items()) + "]"
    L += ["", "FILTROS ESTADISTICOS",
          f"   G0 deriva (hoy vs ventana de entrenamiento)     : {g0}",
          f"   G1 destreza (SPA vs azar, {len(res.agents)+3} candidatos) : p={v.p_skill:.4f}"
          f"  {'PASA' if v.p_skill < cfg.spa_alpha else 'NO PASA'}",
          f"   G2 stack vs equiponderar                        : p={v.p_stack_vs_eq:.4f}"
          f"  -> se usa '{v.combiner if v.actionable else 'ninguno'}'",
          f"   G3 PBO (persistencia del mejor agente)          : "
          + ("n/d" if not np.isfinite(v.pbo) else f"{v.pbo:.2f}  "
             f"{'PASA' if v.pbo <= cfg.pbo_max else 'NO PASA'}"),
          f"   G4 los {cfg.rank.top_k} primeros frente al universo          : p={v.p_topk:.4f}"
          f"  {'PASA' if v.p_topk < cfg.spa_alpha else 'NO PASA'}"
          f"   (exceso medio {v.topk_excess:+.2%} a {res.horizon} sesiones)",
          f"   evidencia OOS del combinador                    : {d.evidence}"
          f"   (IC {d.mean_rank_ic:+.3f}, t={d.ic_tstat:+.2f}, "
          f"prec@{cfg.rank.top_k} {d.precision_at_k:.0%} vs base {d.base_rate:.0%})",
          ""]

    if not v.actionable:
        L += ["VEREDICTO: NO ACCIONABLE"]
        L += [f"   - {r}" for r in v.reasons]
        if res.conviction is None:
            L += ["", "   No se asigna capital. El ranking se muestra abajo solo por transparencia;",
                  "   sus probabilidades son la tasa base del universo, no una prediccion.", ""]
        else:
            L += ["", "   En modo 'gated' no se asignaria capital. En modo 'conviction' se asigna",
                  "   igualmente (ver ASIGNACION POR CONVICCION). Las probabilidades del TOP",
                  "   son la tasa base del universo, no una prediccion.", ""]
    else:
        L += ["VEREDICTO: ACCIONABLE"] + [f"   - {r}" for r in v.reasons] + [""]
        if v.combiner == "stack" and not res.weights.empty:
            L.append("   pesos aprendidos: " +
                     "  ".join(f"{k} {float(x):.2f}" for k, x in res.weights.items()))
            L.append("")
        elif v.combiner == "eq":
            L.append(f"   combinador en uso: equiponderacion de {len(res.agents)} agentes "
                     "(el stack no demostro ser mejor)")
            L.append("")

    L.append(f"TOP {cfg.rank.top_k}")
    for _, r in res.picks.head(cfg.rank.top_k).iterrows():
        L.append(f"  {int(r['rank']):2d}. {r['symbol']:<10} P(exito)={r['prob_success']:.0%}"
                 f"   score={r['score']:+.3f}   acuerdo={r['agreement']:.0%}"
                 f"   ultimo={r['last_price']:.2f}")

    if v.actionable and not res.portfolio.empty and res.portfolio["weight"].sum() > 0:
        L += ["", "CARTERA SUGERIDA (fraccion del capital)"]
        for _, r in res.portfolio.iterrows():
            if r["weight"] > 0:
                L.append(f"  {r['symbol']:<10} {r['weight']:6.1%}   ventaja esperada {r['edge']:+.2%}")
    if res.conviction is not None:
        L += [""] + _format_conviction(res, cfg)
    L += ["", "  Esto no es asesoramiento de inversion. Ver README para las limitaciones."]
    return "\n".join(L)


def _format_conviction(res: SwarmRankResult, cfg: SwarmRankConfig) -> list[str]:
    c, v, i = res.conviction, res.verdict, res.conviction.info
    L = ["ASIGNACION POR CONVICCION" + ("" if v.actionable else " — ventaja NO demostrada")]
    if not v.actionable:
        L += ["   Es la mejor eleccion con la informacion disponible, no una prediccion",
              "   validada: el veredicto de arriba sigue siendo NO ACCIONABLE."]
    if i:
        lam = i.get("map_lambda_mean")
        L += [f"   aversion al riesgo {i.get('risk_aversion', 0):g}   escenarios "
              f"{i.get('n_scenarios', 0)}   candidatos {i.get('n_candidates', 0)}   "
              f"posiciones efectivas {i.get('effective_positions', 0):.2f}",
              f"   mapa puntuacion->retorno: {i.get('map_bins', 0)} tramos, "
              f"{i.get('map_dates', 0)} fechas fuera de muestra, encogimiento "
              f"lambda={(lam if lam is not None else 0.0):.2f} (0 = ruido, 1 = senal)"]
    thr = cfg.rank.success_threshold
    for _, r in c.table.iterrows():
        L.append(f"  {r['symbol']:<10} {r['weight']:6.1%}   retorno esperado {r['mu']:+.2%}"
                 f"   desviacion {r['sd']:.1%}   P(>{thr:+.0%}) {r['p_success']:.0%}")
    if c.cash > 0:
        L.append(f"  {'efectivo':<10} {c.cash:6.1%}")
    if not c.weights.empty:
        L.append(f"   equivalente cierto a {res.horizon} sesiones: {c.certainty_equivalent:+.2%}"
                 f"   (cota de optimalidad {c.gap:.1e})")
        L.append("   alternativas con los mismos escenarios (equivalente cierto):")
        L += [f"     {k:<44} {val:+.2%}" for k, val in c.alternatives.items()]
    L += [f"   nota: {n}" for n in c.notes]
    return L
