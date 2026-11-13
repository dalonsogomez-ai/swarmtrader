"""Adaptador de Interactive Brokers (TWS / IB Gateway) con `ib_async`.

Por que IBKR: es la plataforma con MAS acciones pequenas y penny stocks de las
evaluadas. Ademas de todas las cotizadas de EE. UU., deja abrir posiciones en
OTC (OTCQX, OTCQB, Pink) con el permiso "United States (Penny Stocks)", salvo
las designadas Caveat Emptor o Grey Market (IBKR Knowledge Base, "U.S. Microcap
Stock Restrictions"). Alpaca solo deja cerrar posiciones OTC.

Estado: escrito contra la API de `ib_async` (sucesor mantenido de `ib_insync`)
y probado con un cliente FALSO. NO se ha ejecutado contra TWS/Gateway.

Reglas de este adaptador
  * Paper por defecto: puertos 7497 (TWS) / 4002 (Gateway). Los puertos reales
    (7496 / 4001) exigen SWARMTRADER_ALLOW_LIVE=1.
  * Acciones ENTERAS: el importe se convierte en floor(importe / precio).
  * Ordenes LIMITADAS "marcables" (precio de referencia +/- `slippage_cap`), nunca
    a mercado: en OTC y en acciones de pocos centimos una orden a mercado puede
    ejecutarse muy lejos del ultimo precio.
  * Comision: 0,0035 USD/accion, minimo 0,35 USD y tope del 1 % (IBKR Pro
    escalonado); el router decide si una orden compensa (data/venues.py).
"""

from __future__ import annotations

import math
import os

from .base import BrokerAdapter, OrderResult, Position

_LIVE_ENV = "SWARMTRADER_ALLOW_LIVE"
_LIVE_PORTS = {7496, 4001}


class IBKRBroker(BrokerAdapter):
    def __init__(self, host: str = "127.0.0.1", port: int = 7497, client_id: int = 17,
                 ib=None, slippage_cap: float = 0.02, account: str | None = None):
        if port in _LIVE_PORTS and os.environ.get(_LIVE_ENV) != "1":
            raise RuntimeError("Puerto REAL de IBKR bloqueado: exporta SWARMTRADER_ALLOW_LIVE=1 "
                               "solo tras validar el adaptador en la cuenta paper.")
        self.slippage_cap = float(slippage_cap)
        self.account = account
        self.paper = port not in _LIVE_PORTS
        if ib is None:
            try:
                from ib_async import IB
            except ImportError as exc:  # pragma: no cover
                raise ImportError("pip install ib_async") from exc
            ib = IB()
            ib.connect(host, port, clientId=client_id, readonly=False)
            ib.reqMarketDataType(3)                 # retrasados si no hay suscripcion
        self.ib = ib
        self._contracts: dict = {}

    # ---------------------------------------------------------------- cuenta
    def _summary(self, tag: str) -> float:
        for v in self.ib.accountSummary(self.account or ""):
            if v.tag == tag and v.currency in ("USD", "BASE", ""):
                return float(v.value)
        return 0.0

    @property
    def equity(self) -> float:
        return self._summary("NetLiquidation")

    @property
    def cash(self) -> float:
        return self._summary("TotalCashValue")

    @property
    def buying_power(self) -> float:
        # sin margen: el menor entre efectivo y fondos disponibles
        return max(0.0, min(self._summary("TotalCashValue"), self._summary("AvailableFunds")))

    def positions(self) -> dict[str, Position]:
        out = {}
        for p in self.ib.positions(self.account or ""):
            if getattr(p.contract, "secType", "STK") != "STK" or p.position <= 0:
                continue
            sym = p.contract.symbol.replace(" ", "-")
            px = self._price(p.contract) or p.avgCost
            out[sym] = Position(sym, float(p.position), float(p.avgCost),
                                float(p.position) * float(px))
        return out

    # ---------------------------------------------------------------- datos
    def _contract(self, symbol: str):
        if symbol not in self._contracts:
            from ib_async import Stock
            c = Stock(symbol.replace("-", " "), "SMART", "USD")
            q = self.ib.qualifyContracts(c)
            self._contracts[symbol] = q[0] if q else None
        return self._contracts[symbol]

    def _price(self, contract, side: str | None = None) -> float:
        t = self.ib.reqTickers(contract)[0]
        vals = {"ask": t.ask, "bid": t.bid, "last": t.last, "close": t.close}
        order = (["ask", "last", "close"] if side == "buy" else
                 ["bid", "last", "close"] if side == "sell" else ["last", "close", "bid", "ask"])
        for k in order:
            v = vals.get(k)
            if v is not None and v == v and v > 0:      # descarta NaN
                return float(v)
        return 0.0

    def last_price(self, symbol: str) -> float:
        c = self._contract(symbol)
        if c is None:
            raise KeyError(f"IBKR no reconoce {symbol}")
        return self._price(c)

    def untradable(self, symbols: list[str]) -> set[str]:
        return {s for s in symbols if self._contract(s) is None}

    # ---------------------------------------------------------------- ordenes
    def _limit(self, symbol: str, side: str, qty: int, ref: float) -> OrderResult:
        from ib_async import LimitOrder
        c = self._contract(symbol)
        k = 1 + self.slippage_cap if side == "buy" else 1 - self.slippage_cap
        tick = 0.0001 if ref < 1.0 else 0.01                    # regla de precios de la SEC
        lim = round(math.floor(ref * k / tick) * tick if side == "buy"
                    else math.ceil(ref * k / tick) * tick, 4)
        order = LimitOrder("BUY" if side == "buy" else "SELL", qty, lim, tif="DAY")
        if self.account:
            order.account = self.account
        trade = self.ib.placeOrder(c, order)
        st = getattr(getattr(trade, "orderStatus", None), "status", "Submitted")
        oid = getattr(getattr(trade, "order", None), "orderId", None)
        return OrderResult(symbol, side, float(qty), round(qty * ref, 2), str(st).lower(),
                           broker_order_id=None if oid is None else str(oid),
                           message=f"limitada {lim}")

    def submit(self, symbol: str, side: str, notional: float) -> OrderResult:
        c = self._contract(symbol)
        if c is None:
            return OrderResult(symbol, side, 0.0, 0.0, "rejected", message="contrato desconocido")
        ref = self._price(c, side)
        if ref <= 0:
            return OrderResult(symbol, side, 0.0, 0.0, "rejected", message="sin precio")
        qty = math.floor(abs(float(notional)) / ref + 1e-9)
        if side == "sell":
            held = self.positions().get(symbol)
            qty = min(qty, int(held.qty) if held else 0)
        if qty < 1:
            return OrderResult(symbol, side, 0.0, 0.0, "rejected",
                               message=f"{notional:.2f} $ no llega a una accion ({ref:.4f} $)")
        return self._limit(symbol, side, qty, ref)

    def liquidate(self, symbol: str) -> OrderResult:
        held = self.positions().get(symbol)
        if not held or held.qty < 1:
            return OrderResult(symbol, "sell", 0.0, 0.0, "rejected", message="sin posicion")
        ref = self._price(self._contract(symbol), "sell")
        return self._limit(symbol, "sell", int(held.qty), ref)
