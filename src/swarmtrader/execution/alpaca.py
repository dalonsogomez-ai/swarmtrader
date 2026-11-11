"""Adaptador Alpaca (paper y real).

Alpaca es la unica de las plataformas evaluadas que combina API pensada para
algoritmos, acciones fraccionales POR API (imprescindible para repartir 50$
entre 5 nombres) y sin minimo de cuenta. Interactive Brokers NO soporta
fraccionales via API.

Para pasar a real hacen falta DOS cosas simultaneas: paper=False y la variable
de entorno SWARMTRADER_ALLOW_LIVE=1. Es deliberadamente incomodo.
"""

from __future__ import annotations

import os

from .base import BrokerAdapter, OrderResult, Position

_LIVE_ENV = "SWARMTRADER_ALLOW_LIVE"


class AlpacaBroker(BrokerAdapter):
    def __init__(self, api_key: str | None = None, secret_key: str | None = None,
                 paper: bool = True):
        try:
            from alpaca.trading.client import TradingClient
            from alpaca.data.historical import StockHistoricalDataClient
        except ImportError as exc:  # pragma: no cover
            raise ImportError("pip install 'alpaca-py'") from exc

        if not paper and os.environ.get(_LIVE_ENV) != "1":
            raise RuntimeError(
                "Modo REAL bloqueado. Exporta SWARMTRADER_ALLOW_LIVE=1 solo despues de "
                "que el campeon haya superado el gate DSR y como minimo 6 meses de paper."
            )
        api_key = api_key or os.environ.get("ALPACA_API_KEY")
        secret_key = secret_key or os.environ.get("ALPACA_SECRET_KEY")
        if not api_key or not secret_key:
            raise RuntimeError("faltan ALPACA_API_KEY / ALPACA_SECRET_KEY")

        self.paper = paper
        self._trading = TradingClient(api_key, secret_key, paper=paper)
        self._data = StockHistoricalDataClient(api_key, secret_key)

    @property
    def equity(self) -> float:
        return float(self._trading.get_account().equity)

    @property
    def cash(self) -> float:
        return float(self._trading.get_account().cash)

    def positions(self) -> dict[str, Position]:
        out: dict[str, Position] = {}
        for p in self._trading.get_all_positions():
            out[p.symbol] = Position(p.symbol, float(p.qty), float(p.avg_entry_price),
                                     float(p.market_value))
        return out

    def last_price(self, symbol: str) -> float:
        from alpaca.data.requests import StockLatestTradeRequest
        req = StockLatestTradeRequest(symbol_or_symbols=symbol)
        return float(self._data.get_stock_latest_trade(req)[symbol].price)

    def submit(self, symbol: str, side: str, notional: float) -> OrderResult:
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest
        try:
            order = self._trading.submit_order(MarketOrderRequest(
                symbol=symbol,
                notional=round(abs(notional), 2),          # fraccional por notional
                side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
            ))
            return OrderResult(symbol, side, float(order.qty or 0.0), abs(notional),
                               str(order.status), broker_order_id=str(order.id))
        except Exception as exc:  # noqa: BLE001
            return OrderResult(symbol, side, 0.0, 0.0, "rejected", message=str(exc))

    def liquidate(self, symbol: str) -> OrderResult:
        try:
            self._trading.close_position(symbol)
            return OrderResult(symbol, "sell", 0.0, 0.0, "submitted")
        except Exception as exc:  # noqa: BLE001
            return OrderResult(symbol, "sell", 0.0, 0.0, "rejected", message=str(exc))
