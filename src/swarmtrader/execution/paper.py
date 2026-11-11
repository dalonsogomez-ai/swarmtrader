"""Broker simulado en memoria. Sin red, sin claves, determinista.

Modela lo que la mayoria de backtests ignora y que con 50$ es decisivo:
comision, spread y slippage proporcional al tamano.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .base import BrokerAdapter, OrderResult, Position


@dataclass
class PaperBroker(BrokerAdapter):
    initial_cash: float = 50.0
    commission_per_order: float = 0.0
    spread_bps: float = 5.0
    slippage_bps: float = 5.0
    fractional: bool = True

    _cash: float = field(init=False)
    _positions: dict[str, Position] = field(init=False, default_factory=dict)
    _prices: dict[str, float] = field(init=False, default_factory=dict)
    blotter: list[OrderResult] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        self._cash = float(self.initial_cash)

    def set_prices(self, prices: dict[str, float]) -> None:
        self._prices.update(prices)
        for sym, pos in list(self._positions.items()):
            px = self._prices.get(sym, pos.avg_price)
            self._positions[sym] = Position(sym, pos.qty, pos.avg_price, pos.qty * px)

    @property
    def equity(self) -> float:
        return self._cash + sum(p.market_value for p in self._positions.values())

    @property
    def cash(self) -> float:
        return self._cash

    def positions(self) -> dict[str, Position]:
        return dict(self._positions)

    def last_price(self, symbol: str) -> float:
        return float(self._prices.get(symbol, 0.0))

    def _fill_price(self, symbol: str, side: str) -> float:
        px = self.last_price(symbol)
        edge = (self.spread_bps / 2 + self.slippage_bps) / 10_000.0
        return px * (1 + edge) if side == "buy" else px * (1 - edge)

    def submit(self, symbol: str, side: str, notional: float) -> OrderResult:
        px = self._fill_price(symbol, side)
        if px <= 0:
            return OrderResult(symbol, side, 0, 0, "rejected", message="sin precio")
        qty = notional / px
        if not self.fractional:
            qty = float(int(qty))
            if qty <= 0:
                return OrderResult(symbol, side, 0, 0, "rejected",
                                   message=f"notional {notional:.2f} < 1 accion a {px:.2f}")
        cost = qty * px + self.commission_per_order
        if side == "buy":
            if cost > self._cash + 1e-9:
                qty = max(0.0, (self._cash - self.commission_per_order) / px)
                if qty <= 0:
                    return OrderResult(symbol, side, 0, 0, "rejected", message="liquidez insuficiente")
                cost = qty * px + self.commission_per_order
            self._cash -= cost
            held = self._positions.get(symbol)
            new_qty = (held.qty if held else 0.0) + qty
            new_avg = ((held.avg_price * held.qty if held else 0.0) + qty * px) / new_qty
            self._positions[symbol] = Position(symbol, new_qty, new_avg, new_qty * self.last_price(symbol))
        else:
            held = self._positions.get(symbol)
            if not held:
                return OrderResult(symbol, side, 0, 0, "rejected", message="sin posicion")
            qty = min(qty, held.qty)
            self._cash += qty * px - self.commission_per_order
            rem = held.qty - qty
            if rem <= 1e-9:
                self._positions.pop(symbol)
            else:
                self._positions[symbol] = Position(symbol, rem, held.avg_price, rem * self.last_price(symbol))
        res = OrderResult(symbol, side, qty, qty * px, "filled")
        self.blotter.append(res)
        return res

    def liquidate(self, symbol: str) -> OrderResult:
        held = self._positions.get(symbol)
        if not held:
            return OrderResult(symbol, "sell", 0, 0, "noop")
        return self.submit(symbol, "sell", held.qty * self.last_price(symbol))
