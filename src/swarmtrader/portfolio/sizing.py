"""De probabilidad calibrada a tamano de posicion.

Este es el eslabon que faltaba en todos los proyectos anteriores: todos
terminaban en una lista de tickers. Una lista no es una cartera. El tamano de
la posicion determina mas del resultado final que el orden del ranking, y es
donde se pierde el dinero cuando el edge es pequeno.

Reglas implementadas, en este orden:

  1. **Ventaja esperada** por nombre a partir de la probabilidad calibrada:
     e_i = p_i * ganancia_media - (1 - p_i) * perdida_media.
     Si el ranker no tiene senal, p_i = tasa base para todos, e_i es constante
     y la cartera colapsa a equiponderacion: no se inventa conviccion.
  2. **Kelly fraccional diagonal**: w_i propto e_i / sigma_i^2. Kelly completo
     es la maxima tasa de crecimiento asintotica pero tiene drawdowns
     intolerables y asume que conoces p exactamente. Con p estimado, la
     literatura y la practica usan una fraccion; el defecto aqui es 0.25.
  3. **Encogimiento por incertidumbre**: la fraccion efectiva se multiplica por
     la calidad de la evidencia (0 si NINGUNA). Sin evidencia, sin apuesta.
  4. **Topes**: por nombre, por sector y de exposicion bruta. Un top-5 mensual
     concentra riesgo idiosincratico; el tope por nombre es la unica defensa.
  5. **Objetivo de volatilidad**: se escala la cartera para apuntar a una
     volatilidad anualizada dada, usando las volatilidades realizadas.
  6. **Banda de no-negociacion**: no se rebalancea lo que se mueve menos que
     `turnover_band`, porque cada rotacion paga coste y, en Espana, impuesto.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = ["SizingConfig", "SizingResult", "size_positions"]


@dataclass
class SizingConfig:
    kelly_fraction: float = 0.25
    max_weight: float = 0.25          # por nombre
    max_sector_weight: float = 0.25   # alineado con RiskLimits.max_sector_pct:
                                      # la capa operativa nunca debe proponer lo
                                      # que la capa de seguridad va a rechazar
    max_gross: float = 1.00           # 1.00 = sin apalancamiento
    target_vol: float = 0.15          # volatilidad anual objetivo de la cartera
    min_weight: float = 0.02          # por debajo, no merece la pena la comision
    turnover_band: float = 0.03       # no rebalancear cambios menores
    avg_win: float | None = None      # si None, se estima del umbral de exito
    avg_loss: float | None = None


@dataclass
class SizingResult:
    weights: pd.Series
    expected_edge: pd.Series
    gross: float
    portfolio_vol: float
    turnover: float
    notes: list[str]

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({"weight": self.weights, "edge": self.expected_edge}) \
            .sort_values("weight", ascending=False)


def _cap_group(w: pd.Series, groups: pd.Series, cap: float) -> pd.Series:
    """Escala hacia abajo los grupos que superan el tope, sin tocar el resto."""
    if groups is None or w.empty:
        return w
    g = groups.reindex(w.index)
    if g.isna().all():
        return w
    out = w.copy()
    for _ in range(8):
        tot = out.groupby(g).sum()
        over = tot[tot > cap + 1e-9]
        if over.empty:
            break
        for sector, s in over.items():
            idx = out.index[g == sector]
            out.loc[idx] *= cap / s
    return out


def size_positions(
    probabilities: pd.Series,
    base_rate: float,
    volatilities: pd.Series,
    *,
    evidence_quality: float = 1.0,
    groups: pd.Series | None = None,
    previous: pd.Series | None = None,
    success_threshold: float = 0.05,
    cfg: SizingConfig = SizingConfig(),
) -> SizingResult:
    """Convierte probabilidades calibradas en pesos de cartera.

    `evidence_quality` en [0, 1]: 1.0 = evidencia FUERTE, 0.35 = DEBIL,
    0.0 = NINGUNA. Con 0.0 la funcion devuelve una cartera vacia; es
    deliberado y es la decision correcta.
    """
    notes: list[str] = []
    p = probabilities.dropna().astype(float)
    if p.empty:
        return SizingResult(pd.Series(dtype=float), pd.Series(dtype=float), 0.0, 0.0, 0.0,
                            ["universo vacio"])

    q = float(np.clip(evidence_quality, 0.0, 1.0))
    if q <= 0.0:
        return SizingResult(pd.Series(0.0, index=p.index), pd.Series(0.0, index=p.index),
                            0.0, 0.0, 0.0,
                            ["evidencia NINGUNA: cartera vacia, no se asigna capital"])

    vol = volatilities.reindex(p.index).astype(float)
    vol = vol.replace([np.inf, -np.inf], np.nan)
    vol = vol.fillna(vol.median() if np.isfinite(vol.median()) else 0.30)
    vol = vol.clip(lower=0.05)

    win = cfg.avg_win if cfg.avg_win is not None else max(success_threshold * 1.6, 0.04)
    loss = cfg.avg_loss if cfg.avg_loss is not None else max(success_threshold * 1.2, 0.03)
    edge = p * win - (1.0 - p) * loss

    # Solo el exceso sobre la tasa base es informacion. Si p == base_rate para
    # todos, el exceso es cero y no hay razon para inclinar la cartera.
    edge_base = float(base_rate) * win - (1.0 - float(base_rate)) * loss
    excess = edge - edge_base

    if float(np.abs(excess).max()) < 1e-9:
        w = pd.Series(1.0 / len(p), index=p.index)
        notes.append("sin dispersion de probabilidad: equiponderacion")
    else:
        raw = np.maximum(excess, 0.0) / (vol**2)
        if raw.sum() <= 1e-12:
            w = pd.Series(0.0, index=p.index)
            notes.append("ningun nombre con ventaja positiva sobre la tasa base")
            return SizingResult(w, edge, 0.0, 0.0, 0.0, notes)
        w = raw / raw.sum()

    w = w * cfg.kelly_fraction * q * 4.0     # 4.0 renormaliza: fraccion 0.25 -> bruto 1.0

    # ORDEN IMPORTANTE: el objetivo de volatilidad es un objetivo blando y puede
    # escalar la cartera HACIA ARRIBA; los topes son restricciones duras. Si se
    # aplican los topes primero, el escalado de volatilidad los viola. Por eso
    # se escala antes y se topa despues, nunca al reves.
    # (aproximacion diagonal: ignora la diversificacion, asi que subestima el
    # beneficio de diversificar y nunca pide mas apalancamiento del necesario)
    pv = float(np.sqrt(((w * vol.reindex(w.index)) ** 2).sum()))
    if pv > 1e-9:
        w = w * (cfg.target_vol / pv)

    w = w.clip(upper=cfg.max_weight)
    w = _cap_group(w, groups, cfg.max_sector_weight)
    if w.sum() > cfg.max_gross:
        w = w * cfg.max_gross / w.sum()
    w = w[w >= cfg.min_weight]
    if w.empty:
        return SizingResult(pd.Series(dtype=float), edge, 0.0, 0.0, 0.0,
                            ["todos los pesos por debajo del minimo operable"])
    pv = float(np.sqrt(((w * vol.reindex(w.index)) ** 2).sum()))

    turnover = 0.0
    if previous is not None and not previous.empty:
        prev = previous.reindex(w.index.union(previous.index)).fillna(0.0)
        cur = w.reindex(prev.index).fillna(0.0)
        delta = (cur - prev).abs()
        frozen = delta < cfg.turnover_band
        cur[frozen] = prev[frozen]
        if cur.sum() > cfg.max_gross:
            cur = cur * cfg.max_gross / cur.sum()
        turnover = float((cur - prev).abs().sum())
        w = cur[cur > 0]
        notes.append(f"{int(frozen.sum())} posiciones congeladas por banda de no-negociacion")

    return SizingResult(
        weights=w.sort_values(ascending=False),
        expected_edge=edge.reindex(w.index),
        gross=float(w.sum()),
        portfolio_vol=float(pv),
        turnover=turnover,
        notes=notes,
    )
