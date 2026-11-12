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

    # --- ganchos con comportamiento por defecto (el simulado ejecuta al instante) ---
    @property
    def buying_power(self) -> float:
        """Efectivo utilizable para COMPRAR ahora mismo, sin margen."""
        return float(self.cash)

    def settle(self, orders: list[OrderResult]) -> None:
        """Espera, si el broker lo permite, a que las ventas enviadas se ejecuten."""
        return None

    def untradable(self, symbols: list[str]) -> set[str]:
        """Simbolos que el broker rechazaria (no operables o no fraccionables)."""
        return set()

    def rebalance_to(self, targets: dict[str, float], risk, dollar_volumes: dict[str, float],
                     cash_buffer: float = 0.005) -> list[OrderResult]:
        """Lleva la cartera a los pesos objetivo, pasando por las guardas.

        targets: {symbol: peso en [0,1]}. Lo que no sume 1.0 queda en liquidez.

        Orden: liquidar, reducir, esperar a que las ventas se ejecuten y solo
        entonces comprar, dimensionando con el poder de compra REAL y un
        colchon. Con la cuenta invertida al 100 % una compra enviada antes que
        la venta que la financia se rechaza (Alpaca: las ventas no liberan poder
        de compra hasta ejecutarse). Lo que no se puede financiar se marca
        `deferred` y el ciclo siguiente, que reconcilia contra el broker, lo
        completa. Las reducciones no pasan por las guardas de ENTRADA: existen
        para impedir abrir riesgo, no para impedir cerrarlo.
        """
        from .guards import RiskViolation

        results: list[OrderResult] = []
        eq = self.equity
        risk.update_equity(eq)
        current = self.positions()
        band = max(1.0, 0.01 * eq)

        # 1. cerrar lo que ya no esta en el objetivo
        sells: list[OrderResult] = []
        for sym in list(current):
            if targets.get(sym, 0.0) <= 1e-9:
                r = self.liquidate(sym)
                results.append(r)
                sells.append(r)

        # 2. reducir lo que sobra; las compras se apartan para despues
        buys: list[tuple[str, float, float]] = []
        gross = 0.0
        for sym, w in sorted(targets.items(), key=lambda kv: -kv[1]):
            if w <= 1e-9:
                continue
            target_notional = w * eq
            held = current.get(sym)
            held_mv = held.market_value if held else 0.0
            delta = target_notional - held_mv
            gross += min(abs(held_mv), target_notional)
            if abs(delta) < band:
                continue
            if delta < 0:
                r = self.submit(sym, "sell", -delta)
                results.append(r)
                sells.append(r)
            else:
                buys.append((sym, target_notional, delta))

        # 3. las compras se financian con lo que las ventas hayan liberado de verdad
        if sells:
            self.settle(sells)
        available = max(0.0, float(self.buying_power)) * (1.0 - cash_buffer)

        # 4. comprar, de mayor a menor peso objetivo
        for sym, target_notional, delta in buys:
            notional = min(delta, available)
            if notional < 1.0:
                results.append(OrderResult(sym, "buy", 0.0, 0.0, "deferred",
                                           message=f"sin poder de compra suficiente "
                                                   f"({available:.2f} $ disponibles); se "
                                                   "completara en el ciclo siguiente"))
                continue
            try:
                risk.check_order(
                    symbol=sym, notional=target_notional, price=self.last_price(sym),
                    dollar_volume=dollar_volumes.get(sym, 0.0),
                    equity=eq, gross_after=gross + notional,
                )
            except RiskViolation as exc:
                results.append(OrderResult(sym, "buy", 0.0, 0.0, "rejected", message=str(exc)))
                continue
            gross += notional
            available -= notional
            results.append(self.submit(sym, "buy", notional))
        return results
