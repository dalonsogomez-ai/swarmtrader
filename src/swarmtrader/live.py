"""Bucle de ejecucion diaria (paper o real) para un campeon certificado.

Un ciclo = una llamada a `run_once()`: carga datos, calcula la senal de HOY con
el genoma del campeon, pasa por el meta-modelo y por las guardas de riesgo, y
lleva la cartera a los pesos objetivo a traves del BrokerAdapter. Escribe un
registro JSONL por ciclo con equity, ordenes y rechazos.

Programalo con cron/launchd tras el cierre de mercado. No es un daemon.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .agents.genome import Genome
from .backtest import FeaturePanel
from .data.features import REGIME_COLUMNS
from .evolution.meta import MetaLabeler
from .execution.base import BrokerAdapter
from .execution.guards import RiskLimits, RiskManager


@dataclass
class CycleReport:
    ts: str
    equity_before: float
    targets: dict[str, float]
    orders: list[dict]
    rejected: list[dict]
    abstained: bool
    halted: bool
    halt_reason: str | None


def compute_targets(panel: FeaturePanel, genome: Genome, meta: MetaLabeler | None = None,
                    t: int | None = None) -> dict[str, float]:
    """Pesos objetivo para la fecha t (por defecto, la ultima). Puede ser {}."""
    t = len(panel.dates) - 1 if t is None else t
    s = panel.scores(genome.weight_vector(panel.factor_names))[t].astype(float)
    eligible = panel.valid[t] & (panel.dollar_vol[t] >= genome.min_dollar_volume) & np.isfinite(s)
    if not eligible.any():
        return {}
    s_m = np.where(eligible, s, -np.inf)
    k = min(genome.top_k, int(eligible.sum()))
    top = np.argpartition(-s_m, k - 1)[:k]
    top = top[np.argsort(-s_m[top])]
    top = top[s_m[top] >= genome.min_conviction]
    if top.size == 0:
        return {}
    med = float(np.median(s[eligible]))
    nxt = float(np.sort(s_m[eligible])[-(top.size + 1)]) if eligible.sum() > top.size else med
    vz = float(panel.regime[t, REGIME_COLUMNS.index("mkt_vol_z")])
    r63 = float(panel.regime[t, REGIME_COLUMNS.index("mkt_ret_63")])
    feats = np.array([[s[j], s[j] - med, s[j] - nxt, top.size, vz, r63] for j in top])
    sizes = np.ones(top.size)
    if meta is not None and meta.fitted:
        p = meta.predict_proba(feats)
        keep = p >= genome.meta_threshold
        top, sizes = top[keep], p[keep]
        if top.size == 0:
            return {}
    w = sizes / sizes.sum()
    w = np.minimum(w, genome.max_weight)
    if vz > genome.vol_regime_cut:
        w = w * genome.regime_exposure
    return {panel.symbols[int(j)]: float(x) for j, x in zip(top, w) if x > 1e-6}


def run_once(panel: FeaturePanel, genome: Genome, broker: BrokerAdapter,
             limits: RiskLimits, meta: MetaLabeler | None = None,
             log_path: str | Path = "runs/live.jsonl") -> CycleReport:
    eq = broker.equity
    risk = RiskManager(limits, eq)
    risk.start_of_day(eq)
    targets = compute_targets(panel, genome, meta)
    t = len(panel.dates) - 1
    dvols = {panel.symbols[j]: float(panel.dollar_vol[t, j]) for j in range(len(panel.symbols))}
    results = broker.rebalance_to(targets, risk, dvols) if not risk.state.halted else []
    rep = CycleReport(
        ts=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        equity_before=eq, targets=targets,
        orders=[r.__dict__ for r in results if r.status not in ("rejected",)],
        rejected=[r.__dict__ for r in results if r.status == "rejected"],
        abstained=not targets, halted=risk.state.halted, halt_reason=risk.state.halt_reason,
    )
    p = Path(log_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("a") as f:
        f.write(json.dumps(rep.__dict__, default=str) + "\n")
    return rep
