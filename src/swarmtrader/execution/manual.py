"""Plataformas SIN API de ordenes (Lightyear): ordenes para ejecutar a mano.

Lightyear no ofrece API publica ni de datos ni de ordenes. Automatizar su web
con un navegador seria fragil y, con dinero real, peligroso: un cambio de
pagina puede convertir una venta en una compra o duplicar una orden. Por eso
la ejecucion aqui es MANUAL y explicita: el sistema calcula las ordenes y las
escribe en un CSV (simbolo, sentido, importe, acciones aproximadas, precio de
referencia, motivo); el usuario las introduce en la app y despues exporta su
cartera para el siguiente ciclo.

`ManualBroker` implementa `BrokerAdapter` para que `execute` y el router
funcionen igual que con un broker con API: las ordenes quedan en estado
`manual` (pendientes de que una persona las ejecute), nunca `filled`.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

from .base import BrokerAdapter, OrderResult, Position


class ManualBroker(BrokerAdapter):
    def __init__(self, venue: str, cash: float, positions_csv: str | None = None,
                 prices: dict[str, float] | None = None, tickets_path: str | None = None):
        """positions_csv: symbol,qty,avg_price[,market_value] (exportacion de la app)."""
        self.venue = venue
        self.paper = True               # el sistema no envia nada: decide una persona
        self._cash = float(cash)
        self._prices = {k.upper(): float(v) for k, v in (prices or {}).items()}
        self._pos: dict[str, Position] = {}
        if positions_csv:
            with open(positions_csv, newline="", encoding="utf-8-sig") as fh:
                for r in csv.DictReader(fh):
                    s = r["symbol"].strip().upper()
                    q, avg = float(r["qty"]), float(r.get("avg_price") or 0.0)
                    mv = float(r.get("market_value") or q * self._prices.get(s, avg))
                    self._pos[s] = Position(s, q, avg, mv)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.tickets_path = tickets_path or f"runs/manual_orders_{venue}_{stamp}.csv"
        self.tickets: list[dict] = []

    @property
    def equity(self) -> float:
        return self._cash + sum(p.market_value for p in self._pos.values())

    @property
    def cash(self) -> float:
        return self._cash

    def positions(self) -> dict[str, Position]:
        return dict(self._pos)

    def last_price(self, symbol: str) -> float:
        s = symbol.upper()
        if s in self._prices:
            return self._prices[s]
        p = self._pos.get(s)
        if p and p.qty:
            return p.market_value / p.qty
        raise KeyError(f"sin precio de referencia para {symbol}")

    def _ticket(self, symbol, side, notional, reason) -> OrderResult:
        try:
            px = self.last_price(symbol)
        except KeyError:
            px = 0.0
        qty = round(notional / px, 6) if px > 0 else 0.0
        self.tickets.append({"venue": self.venue, "symbol": symbol, "side": side,
                             "notional": round(notional, 2), "approx_qty": qty,
                             "ref_price": px, "reason": reason})
        self._flush()
        return OrderResult(symbol, side, qty, round(notional, 2), "manual",
                           message=f"ejecutar a mano en {self.venue} -> {self.tickets_path}")

    def submit(self, symbol: str, side: str, notional: float) -> OrderResult:
        # el CSV ordena primero las ventas: en la app se ejecutan antes y su importe
        # queda disponible para las compras de la misma sesion (cuenta sin margen)
        self._cash += -notional if side == "buy" else notional
        return self._ticket(symbol, side, abs(float(notional)), "rebalanceo")

    def liquidate(self, symbol: str) -> OrderResult:
        p = self._pos.get(symbol.upper())
        if not p:
            return OrderResult(symbol, "sell", 0.0, 0.0, "rejected", message="sin posicion")
        self._cash += p.market_value
        return self._ticket(symbol, "sell", p.market_value, "cerrar posicion completa")

    def _flush(self) -> None:
        Path(self.tickets_path).parent.mkdir(parents=True, exist_ok=True)
        with open(self.tickets_path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(self.tickets[0]))
            w.writeheader()
            w.writerows(self.tickets)
