"""Genoma de un agente: el espacio de busqueda del torneo.

Nota de diseno CRITICA: el genoma NO contiene ningun parametro de riesgo duro
(apalancamiento, tamano maximo de posicion, perdida diaria maxima). Esos viven
en execution/guards.py y son inmutables. Si la evolucion pudiese mutarlos,
descubriria en pocas generaciones que maximizar apalancamiento maximiza el
fitness a corto plazo, y la poblacion entera convergeria a la ruina.

Los pesos se indexan por NOMBRE de factor, no por posicion: el conjunto de
factores puede crecer (noticias, analistas, fundamentales...) sin invalidar
genomas antiguos. `weight_vector(names)` alinea al panel que se use.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace

import numpy as np

from ..data.features import FACTOR_NAMES


def _rng_id(rng: np.random.Generator) -> str:
    """ID derivado del RNG: los runs con la misma semilla son reproducibles bit a bit."""
    return f"{int(rng.integers(0, 2**40)):010x}"


BOUNDS: dict[str, tuple[float, float]] = {
    "top_k": (2, 10),
    "min_conviction": (0.0, 2.0),
    "rebalance_days": (5, 42),
    "stop_loss": (0.05, 0.35),
    "take_profit": (0.05, 1.00),
    "min_dollar_volume": (5e5, 5e7),
    "max_weight": (0.15, 0.50),
    "vol_regime_cut": (0.5, 4.0),
    "regime_exposure": (0.0, 1.0),
    "meta_threshold": (0.30, 0.70),
}


def _l1(w: np.ndarray) -> np.ndarray:
    d = np.abs(w).sum()
    return w / d if d > 1e-12 else w


@dataclass
class Genome:
    weights: dict[str, float] = field(default_factory=dict)
    top_k: int = 5
    min_conviction: float = 0.5
    rebalance_days: int = 21
    stop_loss: float = 0.15
    take_profit: float = 0.40
    min_dollar_volume: float = 2e6
    max_weight: float = 0.34
    vol_regime_cut: float = 4.0
    regime_exposure: float = 1.0
    meta_threshold: float = 0.5
    agent_id: str = "manual"
    lineage: tuple[str, ...] = ()

    @classmethod
    def random(cls, rng: np.random.Generator, names: tuple[str, ...] | list[str] = FACTOR_NAMES) -> "Genome":
        w = _l1(rng.normal(0.0, 1.0, len(names)))
        return cls(
            agent_id=_rng_id(rng),
            weights={n: float(v) for n, v in zip(names, w)},
            top_k=int(rng.integers(BOUNDS["top_k"][0], BOUNDS["top_k"][1] + 1)),
            min_conviction=float(rng.uniform(*BOUNDS["min_conviction"])),
            rebalance_days=int(rng.integers(BOUNDS["rebalance_days"][0], BOUNDS["rebalance_days"][1] + 1)),
            stop_loss=float(rng.uniform(*BOUNDS["stop_loss"])),
            take_profit=float(rng.uniform(*BOUNDS["take_profit"])),
            min_dollar_volume=float(np.exp(rng.uniform(np.log(BOUNDS["min_dollar_volume"][0]),
                                                       np.log(BOUNDS["min_dollar_volume"][1])))),
            max_weight=float(rng.uniform(*BOUNDS["max_weight"])),
            vol_regime_cut=float(rng.uniform(*BOUNDS["vol_regime_cut"])),
            regime_exposure=float(rng.uniform(*BOUNDS["regime_exposure"])),
            meta_threshold=float(rng.uniform(*BOUNDS["meta_threshold"])),
        )

    def weight_vector(self, names: tuple[str, ...] | list[str] = FACTOR_NAMES) -> np.ndarray:
        return np.array([self.weights.get(n, 0.0) for n in names], dtype=float)

    def mutate(self, rng: np.random.Generator, rate: float = 0.3, scale: float = 0.25) -> "Genome":
        names = list(self.weights)
        w = self.weight_vector(names)
        mask = rng.random(w.size) < rate
        w[mask] += rng.normal(0.0, scale, int(mask.sum()))
        w = _l1(w)

        def jitter(val, key, integer=False, log=False):
            lo, hi = BOUNDS[key]
            if rng.random() >= rate:
                return int(val) if integer else float(val)
            v = np.exp(np.log(max(val, lo)) + rng.normal(0, scale)) if log else val + rng.normal(0, scale * (hi - lo))
            v = float(np.clip(v, lo, hi))
            return int(round(v)) if integer else v

        return replace(
            self,
            weights={n: float(v) for n, v in zip(names, w)},
            top_k=jitter(self.top_k, "top_k", integer=True),
            min_conviction=jitter(self.min_conviction, "min_conviction"),
            rebalance_days=jitter(self.rebalance_days, "rebalance_days", integer=True),
            stop_loss=jitter(self.stop_loss, "stop_loss"),
            take_profit=jitter(self.take_profit, "take_profit"),
            min_dollar_volume=jitter(self.min_dollar_volume, "min_dollar_volume", log=True),
            max_weight=jitter(self.max_weight, "max_weight"),
            vol_regime_cut=jitter(self.vol_regime_cut, "vol_regime_cut"),
            regime_exposure=jitter(self.regime_exposure, "regime_exposure"),
            meta_threshold=jitter(self.meta_threshold, "meta_threshold"),
            agent_id=_rng_id(rng),
            lineage=(self.lineage + (self.agent_id,))[-6:],
        )

    def crossover(self, other: "Genome", rng: np.random.Generator) -> "Genome":
        names = list(dict.fromkeys(list(self.weights) + list(other.weights)))
        a, b = self.weight_vector(names), other.weight_vector(names)
        w = _l1(np.where(rng.random(len(names)) < 0.5, a, b))
        lam = float(rng.uniform(0.3, 0.7))
        blend = lambda x, y: lam * x + (1 - lam) * y  # noqa: E731
        return Genome(
            agent_id=_rng_id(rng),
            weights={n: float(v) for n, v in zip(names, w)},
            top_k=int(round(blend(self.top_k, other.top_k))),
            min_conviction=blend(self.min_conviction, other.min_conviction),
            rebalance_days=int(round(blend(self.rebalance_days, other.rebalance_days))),
            stop_loss=blend(self.stop_loss, other.stop_loss),
            take_profit=blend(self.take_profit, other.take_profit),
            min_dollar_volume=float(np.exp(blend(np.log(self.min_dollar_volume), np.log(other.min_dollar_volume)))),
            max_weight=blend(self.max_weight, other.max_weight),
            vol_regime_cut=blend(self.vol_regime_cut, other.vol_regime_cut),
            regime_exposure=blend(self.regime_exposure, other.regime_exposure),
            meta_threshold=blend(self.meta_threshold, other.meta_threshold),
            lineage=(self.agent_id, other.agent_id),
        )

    @classmethod
    def from_dict(cls, d: dict) -> "Genome":
        d = dict(d)
        d["lineage"] = tuple(d.get("lineage", ()))
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in known})

    def to_dict(self) -> dict:
        d = asdict(self)
        d["lineage"] = list(self.lineage)
        return d
