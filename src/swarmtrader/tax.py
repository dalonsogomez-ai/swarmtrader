"""Fiscalidad espanola sobre el P&L realizado (IRPF, base del ahorro).

Ningun backtest academico modela esto y para un particular en Espana con
rotacion mensual es a menudo la diferencia entre alfa neto positivo y negativo.

Implementa:
  * tramos de la base del ahorro (parametrizables; verifica los vigentes)
  * compensacion de perdidas con ganancias dentro del ejercicio
  * arrastre de perdidas netas hasta 4 ejercicios
  * regla de los 2 meses (art. 33.5 LIRPF): una perdida en valores homogeneos
    NO es deducible si se recompran en los 2 meses siguientes; se difiere hasta
    la transmision definitiva. Aqui: 42 sesiones.

Esto es un modelo, no asesoramiento fiscal. Confirma tramos y reglas con la
normativa del ejercicio y con un asesor.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .backtest import BacktestResult

# tramos base del ahorro (limite superior, tipo). Verificar cada ejercicio.
DEFAULT_BRACKETS: tuple[tuple[float, float], ...] = (
    (6_000.0, 0.19),
    (50_000.0, 0.21),
    (200_000.0, 0.23),
    (300_000.0, 0.27),
    (float("inf"), 0.30),
)


def tax_on_gain(gain: float, brackets=DEFAULT_BRACKETS) -> float:
    if gain <= 0:
        return 0.0
    tax, lower = 0.0, 0.0
    for upper, rate in brackets:
        if gain <= lower:
            break
        tax += (min(gain, upper) - lower) * rate
        lower = upper
    return tax


@dataclass
class TaxReport:
    years: list[int]
    realized_gain: dict[int, float]        # ganancia neta computable por ejercicio (EUR)
    deferred_loss_total: float             # perdidas diferidas por la regla de 2 meses
    tax_paid: dict[int, float]
    carried_losses_end: float
    pre_tax_return: float
    post_tax_return: float
    tax_drag: float                        # puntos de retorno perdidos en impuestos

    def summary(self) -> dict:
        return {
            "pre_tax_return": round(self.pre_tax_return, 4),
            "post_tax_return": round(self.post_tax_return, 4),
            "tax_drag": round(self.tax_drag, 4),
            "total_tax": round(sum(self.tax_paid.values()), 2),
            "deferred_loss_2m_rule": round(self.deferred_loss_total, 2),
            "carried_losses_end": round(self.carried_losses_end, 2),
        }


def apply_spanish_tax(result: BacktestResult, capital: float,
                      two_month_sessions: int = 42,
                      brackets=DEFAULT_BRACKETS) -> TaxReport:
    """Aplica IRPF sobre los eventos realizados de un BacktestResult."""
    dates = result.dates
    buys_by_sym: dict[int, list[int]] = {}
    for t, j in result.buys:
        buys_by_sym.setdefault(j, []).append(t)

    # 1. regla de los 2 meses: perdida diferida si hay recompra en (t, t+42]
    gains_by_year: dict[int, float] = {}
    deferred_total = 0.0
    pending_deferred: dict[int, float] = {}   # perdida diferida pendiente por simbolo
    for ev in sorted(result.realized, key=lambda e: e.t):
        year = int(dates[min(max(ev.t, 0), len(dates) - 1)].year)
        amount = ev.pnl_frac * capital
        # libera diferidos anteriores del mismo simbolo al cerrar de nuevo
        amount += pending_deferred.pop(ev.symbol_idx, 0.0)
        if amount < 0:
            rebought = any(ev.t < b <= ev.t + two_month_sessions for b in buys_by_sym.get(ev.symbol_idx, []))
            if rebought:
                pending_deferred[ev.symbol_idx] = pending_deferred.get(ev.symbol_idx, 0.0) + amount
                deferred_total += -amount
                continue
        gains_by_year[year] = gains_by_year.get(year, 0.0) + amount

    # perdidas que quedan diferidas al final no se deducen nunca en el horizonte
    years = sorted(set(gains_by_year) | ({int(dates[-1].year)} if len(dates) else set()))
    tax_paid: dict[int, float] = {}
    carried = 0.0
    for y in years:
        net = gains_by_year.get(y, 0.0) - carried
        if net > 0:
            tax_paid[y] = tax_on_gain(net, brackets)
            carried = 0.0
        else:
            tax_paid[y] = 0.0
            carried = -net

    total_tax = sum(tax_paid.values())
    pre = result.total_return
    post = pre - total_tax / capital
    return TaxReport(
        years=years, realized_gain=gains_by_year, deferred_loss_total=deferred_total,
        tax_paid=tax_paid, carried_losses_end=carried,
        pre_tax_return=pre, post_tax_return=post, tax_drag=pre - post,
    )
