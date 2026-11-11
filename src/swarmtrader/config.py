"""Carga de configuracion desde TOML (stdlib, sin dependencias).

Los limites de riesgo se leen aqui pero NUNCA pasan al genoma: se entregan
directamente a RiskManager en la capa de ejecucion.
"""

from __future__ import annotations

import tomllib
from dataclasses import fields
from pathlib import Path

from .evolution.engine import SwarmConfig
from .evolution.fitness import FitnessConfig
from .evolution.selection import DeathConfig
from .execution.guards import RiskLimits


def _pick(cls, d: dict):
    names = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in d.items() if k in names})


def load_config(path: str | Path | None = None) -> tuple[SwarmConfig, RiskLimits]:
    if path is None:
        return SwarmConfig(), RiskLimits()
    data = tomllib.loads(Path(path).read_text())
    swarm = dict(data.get("swarm", {}))
    swarm["fitness"] = _pick(FitnessConfig, data.get("fitness", {}))
    swarm["death"] = _pick(DeathConfig, data.get("death", {}))
    return _pick(SwarmConfig, swarm), _pick(RiskLimits, data.get("risk", {}))
