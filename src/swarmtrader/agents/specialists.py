"""Enjambre de agentes especialistas.

Filosofia de diseno
-------------------
Un enjambre util NO es "muchos agentes que opinan y votan". Eso multiplica
hipotesis sin anadir informacion y es exactamente como se fabrica un backtest
que no replica. Aqui cada agente cumple tres condiciones:

  1. **Tesis economica previa**: cada especialista agrupa factores que responden
     a la misma anomalia documentada, no a una busqueda de combinaciones. El
     conjunto de agentes se fija ANTES de ver los datos, asi que el numero de
     hipotesis es conocido y se puede deflactar.
  2. **Score propio out-of-sample**: cada uno produce su ranking con la misma
     maquinaria walk-forward purgada que el modelo global. Comparables entre si.
  3. **Se gana el sitio**: su peso en el conjunto sale de su IC pasado, y el
     conjunto entero tiene que batir a la linea base equiponderada y pasar el
     contraste SPA. Un agente que no aporta, pesa cero.

El desacuerdo entre agentes no es un problema a ocultar: es informacion. La
dispersion cross-seccional de los rankings de los especialistas se usa para
encoger la confianza cuando el enjambre no se pone de acuerdo.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

__all__ = ["Specialist", "DEFAULT_ROSTER", "active_roster", "run_swarm", "SwarmPanel"]


@dataclass(frozen=True)
class Specialist:
    """Un agente: un nombre, una tesis, un subconjunto de factores."""

    name: str
    factors: tuple[str, ...]
    thesis: str
    reference: str
    requires_external: bool = False
    kind: str = "linear"            # linear | deep
    requires_torch: bool = False

    def available(self, columns: list[str]) -> tuple[str, ...]:
        if self.factors == ("__all__",):     # el agente profundo consume todo el panel
            return tuple(columns)
        return tuple(f for f in self.factors if f in columns)


DEFAULT_ROSTER: tuple[Specialist, ...] = (
    Specialist(
        name="momentum",
        factors=("mom_12_1", "mom_6_1", "accel", "dist_52w_high"),
        thesis="los ganadores de 12 meses siguen ganando durante 1-3 meses; la cercania "
               "al maximo de 52 semanas captura el anclaje de los inversores",
        reference="Jegadeesh-Titman 1993; George-Hwang 2004",
    ),
    Specialist(
        name="reversal",
        factors=("reversal_21", "reversal_5", "idio_skew_63"),
        thesis="el movimiento extremo de corto plazo revierte por presion de liquidez; "
               "la asimetria idiosincratica positiva se paga de mas",
        reference="Jegadeesh 1990; Boyer-Mitton-Vorkink 2010",
    ),
    Specialist(
        name="lowrisk",
        factors=("low_vol_60", "low_beta_60", "anti_lottery"),
        thesis="beta y volatilidad altas rinden menos de lo que predice el CAPM; "
               "las acciones tipo loteria tienen retorno esperado negativo",
        reference="Frazzini-Pedersen 2014; Bali-Cakici-Whitelaw 2011",
    ),
    Specialist(
        name="liquidity",
        factors=("illiq_amihud", "vol_trend", "vol_price_div"),
        thesis="prima por iliquidez y confirmacion por volumen: un movimiento sin "
               "volumen detras es menos persistente",
        reference="Amihud 2002",
    ),
    Specialist(
        name="seasonal",
        factors=("seasonal_12m",),
        thesis="los retornos del mismo mes del calendario reaparecen por patrones "
               "recurrentes de flujo institucional",
        reference="Heston-Sadka 2008",
    ),
    # --- agentes de evento: solo se activan si hay senales externas cargadas ---
    Specialist(
        name="earnings",
        factors=("sue", "earn_react_3d", "days_since_earn"),
        thesis="post-earnings announcement drift: la sorpresa de resultados se "
               "incorpora al precio durante 1-3 meses, no de golpe",
        reference="Bernard-Thomas 1989",
        requires_external=True,
    ),
    Specialist(
        name="analyst",
        factors=("rev_chg_21", "rev_chg_63", "target_upside"),
        thesis="las REVISIONES de consenso predicen; el nivel del consenso no",
        reference="Jegadeesh-Kim-Krische-Lee 2004",
        requires_external=True,
    ),
    Specialist(
        name="flow",
        factors=("insider_net", "short_ratio", "short_chg"),
        thesis="compras de insiders y cambios en el interes corto son informacion "
               "de agentes con ventaja o con conviccion costosa",
        reference="Lakonishok-Lee 2001; Boehmer-Jones-Zhang 2008",
        requires_external=True,
    ),
    Specialist(
        name="news",
        factors=("news_score", "news_count", "news_neg"),
        thesis="el tono de la cobertura tiene deriva de corto plazo; el volumen de "
               "noticias proxy la atencion del inversor minorista",
        reference="Tetlock 2007; Barber-Odean 2008",
        requires_external=True,
    ),
    Specialist(
        name="political",
        factors=("pol_net_buy", "pol_n_filers", "pol_cluster", "pol_leader_buy",
                 "pol_days_since"),
        thesis="la DIVULGACION de operaciones de cargos publicos (STOCK Act) es un "
               "evento de atencion y posible informacion; la hipotesis NO es copiar la "
               "operacion, que llega con hasta 45 dias de retraso y por tanto es "
               "incopiable a 21-32 sesiones, sino si el precio reacciona al anuncio",
        reference="Ziobrowski 2004/2011 (pre-STOCK Act, a favor); Eggers-Hainmueller 2013, "
                  "Belmont et al. 2022, Chen-Sacerdote 2026 (post-STOCK Act, EN CONTRA); "
                  "Wei-Zhou NBER (solo lideres)",
        requires_external=True,
    ),
    # --- agentes profundos: requieren el extra [deep] ---
    Specialist(
        name="deep_rank",
        factors=("__all__",),
        thesis="GRU sobre la secuencia de retornos y volumen + atencion sobre TODO el "
               "corte transversal con sesgo por sector; entrenado con perdida de ranking "
               "(ListNet), no de regresion, y promediando varias semillas",
        reference="Feng et al. 2019 (RSR); Abe-Nakayama 2018; Gu-Kelly-Xiu 2020",
        kind="deep",
        requires_torch=True,
    ),
)


@dataclass
class SwarmPanel:
    """Salida del enjambre: un score OOS por especialista y fecha/simbolo."""

    scores: pd.DataFrame                    # date, symbol, fwd, success, <un col por agente>
    agents: list[str]
    roster: list[Specialist] = field(default_factory=list)
    skipped: dict[str, str] = field(default_factory=dict)

    @property
    def empty(self) -> bool:
        return self.scores.empty or not self.agents

    def agent_matrix(self) -> pd.DataFrame:
        return self.scores[self.agents]


def active_roster(columns: list[str], roster: tuple[Specialist, ...] = DEFAULT_ROSTER,
                  min_factors: int = 1, allow_deep: bool = True) -> tuple[list[Specialist], dict[str, str]]:
    """Filtra el roster a los agentes que tienen factores disponibles."""
    from ..models.deep import TORCH_AVAILABLE

    keep, skipped = [], {}
    for sp in roster:
        if sp.kind == "deep":
            if not allow_deep:
                skipped[sp.name] = "desactivado (--no-deep)"
                continue
            if sp.requires_torch and not TORCH_AVAILABLE:
                skipped[sp.name] = "requiere PyTorch: pip install -e '.[deep]'"
                continue
            keep.append(sp)
            continue
        av = sp.available(columns)
        if len(av) < min_factors:
            skipped[sp.name] = (
                "requiere senales externas no cargadas" if sp.requires_external
                else "factores no disponibles en este panel"
            )
            continue
        keep.append(sp)
    return keep, skipped


def _rank_norm(s: pd.Series) -> pd.Series:
    """Rango cross-seccional a [-1, 1]. Hace comparables agentes con escalas distintas."""
    n = s.notna().sum()
    if n < 2:
        return pd.Series(0.0, index=s.index)
    r = s.rank(method="average", na_option="keep")
    return ((r - 1.0) / (n - 1.0) * 2.0 - 1.0).fillna(0.0)


def run_swarm(df: pd.DataFrame, cfg, roster: tuple[Specialist, ...] = DEFAULT_ROSTER,
              verbose: bool = False, md=None, groups=None, deep_cfg=None,
              allow_deep: bool = True) -> SwarmPanel:
    """Ejecuta cada especialista con walk-forward purgado y alinea sus scores.

    Reutiliza `rank.walk_forward_oos`, de modo que la logica anti-fuga esta
    implementada UNA vez y auditada UNA vez.
    """
    from ..rank import walk_forward_oos  # import diferido: evita ciclo

    cols = list(df.attrs.get("factor_cols", []))
    active, skipped = active_roster(cols, roster, allow_deep=allow_deep and md is not None)
    if md is None:
        for sp in roster:
            if sp.kind == "deep" and sp.name not in skipped:
                skipped[sp.name] = "necesita precios/volumenes crudos (md)"
        active = [sp for sp in active if sp.kind != "deep"]
    if not active:
        return SwarmPanel(pd.DataFrame(), [], [], skipped)

    base: pd.DataFrame | None = None
    agents: list[str] = []
    for sp in active:
        fac = list(sp.available(cols))
        if sp.kind == "deep":
            from ..models.deep import DeepConfig
            from ..models.trainer import DeepPanelBuilder, deep_walk_forward
            dcfg = deep_cfg or DeepConfig()
            builder = DeepPanelBuilder(md.prices, md.volumes, fac,
                                       seq_len=dcfg.seq_len, groups=groups)
            if verbose:
                print(f"  agente {sp.name:<10} entrenando (torch)...")
            oos = deep_walk_forward(df, cfg, builder, dcfg, verbose=False)
        else:
            oos = walk_forward_oos(df, cfg, x_cols=fac)
        if oos.empty:
            skipped[sp.name] = "sin fechas OOS evaluables"
            continue
        part = oos[["date", "symbol", "score", "fwd", "success"]].copy()
        part["score"] = part.groupby("date")["score"].transform(_rank_norm)
        part = part.rename(columns={"score": sp.name})
        if base is None:
            base = part
        else:
            base = base.merge(part[["date", "symbol", sp.name]], on=["date", "symbol"], how="inner")
        agents.append(sp.name)
        if verbose:
            print(f"  agente {sp.name:<10} factores={len(fac)}  obs={len(part)}")

    if base is None or not agents:
        return SwarmPanel(pd.DataFrame(), [], active, skipped)
    return SwarmPanel(base.reset_index(drop=True), agents, active, skipped)


def agent_diagnostics(panel: SwarmPanel) -> pd.DataFrame:
    """IC medio, t-stat y ICIR OOS de cada agente por separado."""
    from scipy import stats as _st

    rows = []
    for a in panel.agents:
        ics = panel.scores.groupby("date").apply(
            lambda g, c=a: _st.spearmanr(g[c], g["fwd"]).statistic if len(g) >= 5 else np.nan,
            include_groups=False).dropna()
        n = len(ics)
        m = float(ics.mean()) if n else 0.0
        sd = float(ics.std(ddof=1)) if n > 1 else 0.0
        rows.append({
            "agent": a,
            "ic": m,
            "t": m / sd * np.sqrt(n) if sd > 1e-12 else 0.0,
            "icir": m / sd if sd > 1e-12 else 0.0,
            "n_dates": n,
        })
    return pd.DataFrame(rows).sort_values("t", ascending=False).reset_index(drop=True)
