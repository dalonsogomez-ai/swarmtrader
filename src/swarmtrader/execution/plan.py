"""Plan de ordenes: de la decision al broker, usando hasta el 100 % del saldo.

Por que hay un plan intermedio y no se envian las ordenes al decidir
---------------------------------------------------------------------
El analisis se hace tras el cierre (precios de cierre completos). Las ordenes
deben entrar en la apertura. Y en una rotacion, las compras dependen del dinero
que liberan las ventas: Alpaca no devuelve poder de compra por una venta hasta
que se ejecuta. Enviar ventas y compras a la vez tras el cierre haria que las
compras se rechazaran o se quedaran cortas.

Por eso `swarmtrader decide` escribe un PLAN (JSON con identificador) y la
ejecucion es un paso aparte, tras la apertura. El plan lo puede ejecutar el
propio sistema (`swarmtrader execute`) o un agente de IA que lea el JSON; en los
dos casos pasa por `execute_plan`, que es donde estan las comprobaciones.

Como se ejecuta
---------------
`execute_plan` no reimplementa la logica de ordenes: delega en
`BrokerAdapter.rebalance_to`, la misma que usa el ciclo diario. Orden: cerrar lo
que sobra, reducir lo que pesa de mas, esperar a que esas ventas se ejecuten
(`settle`) y comprar con el poder de compra REAL menos un colchon
(`cash_buffer`). Lo que no se puede financiar queda `deferred`. Las reducciones
no pasan por las guardas de entrada.

Las guardas de la ejecucion son las del modo conviccion: sin tope por nombre,
por sector ni numero de ordenes; se mantienen sin margen, sin cortos, precio y
liquidez minimos, perdida diaria y drawdown. Los activos que el broker
rechazaria (no fraccionables en Alpaca) se excluyen ya en el analisis; si
alguno llega hasta aqui se descarta con una nota y su parte queda en efectivo.

Lo que exige dinero real
------------------------
`AlpacaBroker(paper=False)` ya exige SWARMTRADER_ALLOW_LIVE=1. Ademas, aqui se
exige `confirm == plan.plan_id`: la confirmacion es de ESTE plan concreto, no un
permiso general. Un plan modificado cambia de identificador y se rechaza al
cargarlo.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .base import BrokerAdapter, OrderResult

__all__ = ["OrderPlan", "ExecutionReport", "build_order_plan", "execute_plan",
           "load_plan", "LiveConfirmationRequired", "plan_risk_limits"]


class LiveConfirmationRequired(RuntimeError):
    """Ejecucion con dinero real sin la confirmacion explicita del plan."""


def _finite(x):
    return None if isinstance(x, float) and not math.isfinite(x) else x


@dataclass
class OrderPlan:
    as_of: str
    entry_session: str
    entry_time_madrid: str
    mode: str
    horizon: int
    targets: dict                   # simbolo -> peso sobre el patrimonio
    sells: list                     # simbolos a cerrar por completo
    keep: list                      # simbolos que no se tocan
    exit_plans: dict                # simbolo -> plan de salida (se guarda en el libro)
    verdict: dict
    reasons: dict = field(default_factory=dict)   # simbolo -> motivo de la venta
    dollar_volumes: dict = field(default_factory=dict)  # volumen en $ del dia del analisis
    min_price: float = 1.0          # guardas de liquidez aplicadas en el analisis
    min_dollar_volume: float = 1_000_000.0
    cash_buffer: float = 0.005
    created_utc: str = field(default_factory=lambda: datetime.now(timezone.utc)
                             .isoformat(timespec="seconds"))
    plan_id: str = ""

    def __post_init__(self) -> None:
        if not self.plan_id:
            self.plan_id = self.fingerprint()

    def fingerprint(self) -> str:
        core = {k: v for k, v in asdict(self).items() if k not in ("plan_id", "created_utc")}
        raw = json.dumps(core, sort_keys=True, default=str, ensure_ascii=False)
        return hashlib.sha256(raw.encode()).hexdigest()[:12]

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False, default=str)

    def save(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_json())
        return p


def load_plan(path: str | Path) -> OrderPlan:
    raw = json.loads(Path(path).read_text())
    plan = OrderPlan(**raw)
    if plan.plan_id != plan.fingerprint():
        raise ValueError(f"el plan {path} se ha modificado tras generarse "
                         f"(id {plan.plan_id}, contenido {plan.fingerprint()}); regeneralo")
    return plan


def plan_risk_limits(min_price: float = 1.0, min_dollar_volume: float = 1_000_000.0):
    """Guardas del modo conviccion (las mismas que `DailyCycle._risk_limits`)."""
    from .guards import RiskLimits
    return RiskLimits(max_position_pct=1.0, max_sector_pct=1.0, max_orders_per_day=100,
                      min_price=min_price, min_dollar_volume=min_dollar_volume)


def build_order_plan(decision, positions: dict, keep: dict | None = None,
                     sell_reasons: dict | None = None, dollar_volumes: dict | None = None,
                     **kw) -> OrderPlan:
    """Plan a partir de una `DecisionResult` y las posiciones actuales del broker.

    keep            {simbolo: peso} que se mantienen sin tocar.
    dollar_volumes  volumen en dolares por simbolo el dia del analisis. Sin el, la
                    guarda de liquidez no puede comprobarse en la ejecucion y se
                    desactiva (`min_dollar_volume` = 0); el analisis ya la aplica.
    Todo lo que esta en el broker y no esta en `keep` ni en los objetivos se vende.
    """
    keep = dict(keep or {})
    targets = {s: round(float(w), 8) for s, w in decision.weights.items() if float(w) > 0}
    sells = sorted(s for s in positions if s not in keep and s not in targets)
    plans = {}
    for s in targets:
        p = decision.exit_plans.get(s)
        if p is not None:
            plans[s] = {k: _finite(v) for k, v in {
                "horizon": p.horizon, "target_pct": p.target_pct, "stop_pct": p.stop_pct,
                "trail_pct": p.trail_pct, "entry_ref_price": p.entry_ref_price,
                "growth_per_session": p.growth_per_session, "source": p.source}.items()}
    reasons = {s: (sell_reasons or {}).get(s, "fuera de la cartera objetivo") for s in sells}
    v = decision.verdict
    if dollar_volumes is None:
        kw.setdefault("min_dollar_volume", 0.0)
        dv = {}
    else:
        dv = {s: round(float(dollar_volumes.get(s, 0.0)), 2) for s in targets}
    return OrderPlan(
        as_of=decision.as_of, entry_session=decision.entry_session,
        entry_time_madrid=decision.entry_time_madrid, mode=decision.mode,
        horizon=int(decision.horizon), targets=targets, sells=sells, keep=sorted(keep),
        exit_plans=plans, reasons=reasons, dollar_volumes=dv,
        verdict={"actionable": v.get("actionable"), "evidence": v.get("evidence"),
                 "p_skill": v.get("p_skill"), "p_topk": v.get("p_topk")},
        **kw,
    )


@dataclass
class ExecutionReport:
    plan_id: str
    dry_run: bool
    live: bool
    equity: float
    budget: float
    orders: list
    notes: list

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False, default=str)


def _simulate(broker: BrokerAdapter, targets: dict, plan: OrderPlan) -> tuple[list, float]:
    """Ordenes que `rebalance_to` enviaria, sin tocar el broker."""
    eq = float(broker.equity)
    current = broker.positions()
    band = max(1.0, 0.01 * eq)
    orders, freed = [], 0.0
    for s, p in current.items():
        if targets.get(s, 0.0) <= 1e-9:
            orders.append(OrderResult(s, "sell", p.qty, p.market_value, "simulada",
                                      message=plan.reasons.get(s)))
            freed += p.market_value
    buys = []
    for s, w in sorted(targets.items(), key=lambda kv: -kv[1]):
        held = current[s].market_value if s in current else 0.0
        delta = w * eq - held
        if abs(delta) < band:
            continue
        if delta < 0:
            orders.append(OrderResult(s, "sell", 0.0, -delta, "simulada", message="reducir"))
            freed -= delta
        else:
            buys.append((s, delta))
    available = max(0.0, float(broker.buying_power) + freed) * (1.0 - plan.cash_buffer)
    budget = available
    for s, delta in buys:
        notional = math.floor(min(delta, available) * 100.0) / 100.0
        if notional < 1.0:
            orders.append(OrderResult(s, "buy", 0.0, 0.0, "deferred",
                                      message="sin poder de compra suficiente"))
            continue
        available -= notional
        orders.append(OrderResult(s, "buy", 0.0, notional, "simulada"))
    return orders, budget


def execute_plan(broker: BrokerAdapter, plan: OrderPlan, *, dry_run: bool = True,
                 confirm: str | None = None,
                 ledger_path: str | Path | None = None) -> ExecutionReport:
    """Ejecuta un plan: lleva la cartera a sus pesos con `rebalance_to`."""
    live = getattr(broker, "paper", True) is False
    if live and not dry_run and confirm != plan.plan_id:
        raise LiveConfirmationRequired(
            f"dinero real: hace falta confirmar este plan concreto con confirm='{plan.plan_id}'")
    notes: list[str] = []
    positions = broker.positions()
    equity = float(broker.equity)

    targets = dict(plan.targets)
    untradable = set(broker.untradable(list(targets)))
    if untradable:
        for s in sorted(untradable):
            targets.pop(s, None)
        notes.append("el broker rechazaria (no operable o no fraccionable): "
                     f"{', '.join(sorted(untradable))}; su parte queda en efectivo")
    # Lo que se mantiene entra en el objetivo con su peso ACTUAL: rebalance_to
    # cierra todo lo que no esta en el objetivo, y la banda del 1 % evita tocarlo.
    for s in plan.keep:
        if s in positions and equity > 0:
            targets.setdefault(s, float(positions[s].market_value) / equity)
    missing = [s for s in plan.sells if s not in positions]
    if missing:
        notes.append(f"ya no estaban en cartera: {', '.join(missing)}")

    if dry_run:
        orders, budget = _simulate(broker, targets, plan)
        return ExecutionReport(plan.plan_id, True, live, equity, budget,
                               [o.__dict__ for o in orders], notes)

    from .guards import RiskManager
    risk = RiskManager(plan_risk_limits(plan.min_price, plan.min_dollar_volume), equity)
    risk.start_of_day(equity)
    dv = dict(plan.dollar_volumes) if plan.min_dollar_volume > 0 else {}
    orders = broker.rebalance_to(targets, risk, dv, cash_buffer=plan.cash_buffer)
    budget = float(sum(o.notional for o in orders if o.side == "buy"
                       and o.status not in ("rejected", "deferred")))
    deferred = [o.symbol for o in orders if o.status == "deferred"]
    if deferred:
        notes.append(f"compras aplazadas por falta de poder de compra: {', '.join(deferred)} "
                     "(las ventas aun no se habian ejecutado: con el mercado cerrado quedan "
                     "en cola; vuelve a ejecutar el plan tras la apertura)")
    rejected = [f"{o.symbol} ({o.message})" for o in orders if o.status == "rejected"]
    if rejected:
        notes.append("rechazadas: " + "; ".join(rejected))

    # libro de posiciones: fecha de entrada, maximo y plan de salida
    if ledger_path:
        from ..cycle import LedgerEntry, PositionLedger
        led = PositionLedger(ledger_path)
        now_pos = broker.positions()
        for s in list(led.entries):
            if s not in now_pos:
                del led.entries[s]
        for s, pl in plan.exit_plans.items():
            # nueva posicion, o renovacion de una que la decision vuelve a elegir:
            # en ambos casos se ancla el plan nuevo al precio de entrada
            if s in now_pos and (s not in led.entries or s not in plan.keep):
                px = float(now_pos[s].avg_price)
                led.entries[s] = LedgerEntry(
                    s, plan.entry_session, px, px, target_weight=plan.targets.get(s, 0.0),
                    note=f"plan {plan.plan_id}", horizon=pl.get("horizon"),
                    target_pct=pl.get("target_pct"), stop_pct=pl.get("stop_pct"),
                    trail_pct=pl.get("trail_pct"))
        led.save()

    return ExecutionReport(plan.plan_id, False, live, equity, budget,
                           [o.__dict__ for o in orders], notes)
