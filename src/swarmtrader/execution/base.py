"""Interfaz de broker. La misma firma para simulacion, paper y real.

El motor de evolucion nunca importa un broker concreto. Cambiar de paper a real
es cambiar una linea de configuracion, no reescribir codigo.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(frozen=True)
class Position:
    symbol: str
    qty: float
    avg_price: float
    market_value: float


@dataclass(frozen=True)
class OrderResult:
    symbol: str
    side: str
    qty: float
    notional: float
    status: str
    broker_order_id: str | None = None
    message: str | None = None


class BrokerAdapter(ABC):
    @property
    @abstractmethod
    def equity(self) -> float: ...

    @property
    @abstractmethod
    def cash(self) -> float: ...

    @abstractmethod
    def positions(self) -> dict[str, Position]: ...

    @abstractmethod
    def last_price(self, symbol: str) -> float: ...

    @abstractmethod
    def submit(self, symbol: str, side: str, notional: float) -> OrderResult: ...

    @abstractmethod
    def liquidate(self, symbol: str) -> OrderResult: ...

    def rebalance_to(self, targets: dict[str, float], risk, dollar_volumes: dict[str, float]) -> list[OrderResult]:
        """Lleva la cartera a los pesos objetivo, pasando por las guardas.

        targets: {symbol: peso en [0,1]}. Lo que no sume 1.0 queda en liquidez.
        """
        from .guards import RiskViolation

        results: list[OrderResult] = []
        eq = self.equity
        risk.update_equity(eq)
        current = self.positions()

        # 1. cerrar lo que ya no esta en el objetivo
        for sym in list(current):
            if targets.get(sym, 0.0) <= 1e-9:
                results.append(self.liquidate(sym))

        # 2. ajustar / abrir
        gross = sum(abs(p.market_value) for s, p in current.items() if targets.get(s, 0.0) > 1e-9)
        for sym, w in sorted(targets.items(), key=lambda kv: -kv[1]):
            if w <= 1e-9:
                continue
            target_notional = w * eq
            held = current.get(sym)
            delta = target_notional - (held.market_value if held else 0.0)
            if abs(delta) < max(1.0, 0.01 * eq):
                continue
            try:
                price = self.last_price(sym)
                risk.check_order(
                    symbol=sym, notional=target_notional, price=price,
                    dollar_volume=dollar_volumes.get(sym, 0.0),
                    equity=eq, gross_after=gross + max(0.0, delta),
                )
            except RiskViolation as exc:
                results.append(OrderResult(sym, "buy" if delta > 0 else "sell", 0.0, 0.0,
                                           "rejected", message=str(exc)))
                continue
            gross += max(0.0, delta)
            results.append(self.submit(sym, "buy" if delta > 0 else "sell", abs(delta)))
        return results
