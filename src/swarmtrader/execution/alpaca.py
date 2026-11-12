"""Adaptador Alpaca (paper y real).

Alpaca es la unica de las plataformas evaluadas que combina API pensada para
algoritmos, acciones fraccionales POR API (imprescindible para repartir 50$
entre 5 nombres) y sin minimo de cuenta. Interactive Brokers NO soporta
fraccionales via API.

Para pasar a real hacen falta DOS cosas simultaneas: paper=False y la variable
de entorno SWARMTRADER_ALLOW_LIVE=1. Es deliberadamente incomodo, y se exige
tambien cuando se inyectan clientes.

Reglas de Alpaca que condicionan este adaptador (docs.alpaca.markets,
"Fractional Trading" y "Orders at Alpaca", consultadas en noviembre de 2026):

  * Ordenes por nocional: 2 decimales como maximo (mas => rechazo) y minimo de
    1 USD. Aqui se truncan HACIA ABAJO a centimos y lo que no llega a 1 USD se
    rechaza en local, sin llamar a la API.
  * Solo activos con `fractionable=True` admiten nocional; el resto se rechaza.
    `untradable` los identifica para que el optimizador no les asigne capital.
  * Las compras abiertas reducen el poder de compra, pero las ventas largas NO
    lo reponen hasta ejecutarse. Por eso `settle` espera a las ventas si el
    mercado esta abierto. Con el mercado cerrado las ordenes quedan en cola para
    la sesion siguiente y no se espera: el ciclo siguiente compra con el
    efectivo real.
  * Con menos de 2.000 USD de patrimonio la cuenta opera a 1x, sin margen. El
    poder de compra se toma del campo mas conservador disponible.
  * Una orden a mercado se rechaza si hay otra abierta de sentido contrario en
    el mismo simbolo: `rebalance_to` nunca compra y vende el mismo simbolo en el
    mismo ciclo.
  * La regla Pattern Day Trader se elimino el 4 de junio de 2026 y los campos
    `daytrade_count` y `daytrading_buying_power` se retiraron: no se leen.
"""

from __future__ import annotations

import math
import os
import time

from .base import BrokerAdapter, OrderResult, Position

_LIVE_ENV = "SWARMTRADER_ALLOW_LIVE"
_MIN_NOTIONAL = 1.0
_TERMINAL = {"filled", "canceled", "expired", "rejected", "done_for_day", "replaced",
             "stopped", "suspended"}


def _status(value) -> str:
    """'OrderStatus.FILLED', OrderStatus.FILLED o 'filled' -> 'filled'."""
    return str(getattr(value, "value", value)).split(".")[-1].lower()


class AlpacaBroker(BrokerAdapter):
    settle_timeout_s: float = 60.0      # espera maxima a que se ejecuten las ventas
    poll_s: float = 2.0

    def __init__(self, api_key: str | None = None, secret_key: str | None = None,
                 paper: bool = True, trading_client=None, data_client=None):
        if not paper and os.environ.get(_LIVE_ENV) != "1":
            raise RuntimeError(
                "Modo REAL bloqueado. Exporta SWARMTRADER_ALLOW_LIVE=1 solo despues de "
                "que el campeon haya superado el gate DSR y como minimo 6 meses de paper."
            )
        self.paper = paper
        self._assets: set[str] | None = None
        if trading_client is not None:
            self._trading, self._data = trading_client, data_client
            return
        try:
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.trading.client import TradingClient
        except ImportError as exc:  # pragma: no cover
            raise ImportError("pip install 'alpaca-py'") from exc
        api_key = api_key or os.environ.get("ALPACA_API_KEY")
        secret_key = secret_key or os.environ.get("ALPACA_SECRET_KEY")
        if not api_key or not secret_key:
            raise RuntimeError("faltan ALPACA_API_KEY / ALPACA_SECRET_KEY")
        self._trading = TradingClient(api_key, secret_key, paper=paper)
        self._data = StockHistoricalDataClient(api_key, secret_key)

    @property
    def equity(self) -> float:
        return float(self._trading.get_account().equity)

    @property
    def cash(self) -> float:
        return float(self._trading.get_account().cash)

    @property
    def buying_power(self) -> float:
        """El campo mas conservador: nunca se compra con margen."""
        a = self._trading.get_account()
        vals = []
        for k in ("non_marginable_buying_power", "buying_power", "cash"):
            v = getattr(a, k, None)
            if v is not None:
                vals.append(float(v))
        return max(0.0, min(vals)) if vals else 0.0

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
        amount = math.floor(abs(float(notional)) * 100.0 + 1e-9) / 100.0
        if amount < _MIN_NOTIONAL:
            return OrderResult(symbol, side, 0.0, 0.0, "rejected",
                               message=f"nocional {amount:.2f} $ por debajo del minimo de "
                                       f"Alpaca ({_MIN_NOTIONAL:.0f} $)")
        from alpaca.trading.enums import OrderSide, TimeInForce
        from alpaca.trading.requests import MarketOrderRequest
        try:
            order = self._trading.submit_order(MarketOrderRequest(
                symbol=symbol,
                notional=amount,                            # fraccional por nocional
                side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
                time_in_force=TimeInForce.DAY,
            ))
            return OrderResult(symbol, side, float(order.qty or 0.0), amount,
                               _status(order.status), broker_order_id=str(order.id))
        except Exception as exc:  # noqa: BLE001
            return OrderResult(symbol, side, 0.0, 0.0, "rejected", message=str(exc))

    def liquidate(self, symbol: str) -> OrderResult:
        try:
            order = self._trading.close_position(symbol)
            return OrderResult(symbol, "sell", float(getattr(order, "qty", 0.0) or 0.0), 0.0,
                               _status(getattr(order, "status", "submitted")),
                               broker_order_id=str(getattr(order, "id", "")) or None)
        except Exception as exc:  # noqa: BLE001
            return OrderResult(symbol, "sell", 0.0, 0.0, "rejected", message=str(exc))

    def settle(self, orders: list[OrderResult]) -> None:
        pending = {o.broker_order_id for o in orders
                   if o.broker_order_id and o.status not in _TERMINAL}
        if not pending:
            return
        try:
            if not self._trading.get_clock().is_open:
                return          # en cola hasta la apertura: el ciclo siguiente completa
        except Exception:  # noqa: BLE001
            return
        deadline = time.monotonic() + float(self.settle_timeout_s)
        while pending:
            for oid in list(pending):
                try:
                    st = _status(self._trading.get_order_by_id(oid).status)
                except Exception:  # noqa: BLE001
                    pending.discard(oid)
                    continue
                if st in _TERMINAL:
                    pending.discard(oid)
            if not pending or time.monotonic() >= deadline:
                return
            time.sleep(float(self.poll_s))

    def untradable(self, symbols: list[str]) -> set[str]:
        if self._assets is None:
            try:
                from alpaca.trading.enums import AssetClass, AssetStatus
                from alpaca.trading.requests import GetAssetsRequest
                assets = self._trading.get_all_assets(GetAssetsRequest(
                    status=AssetStatus.ACTIVE, asset_class=AssetClass.US_EQUITY))
                self._assets = {a.symbol for a in assets if a.tradable and a.fractionable}
            except Exception:  # noqa: BLE001
                # Sin la lista no se veta nada: lo que Alpaca rechace queda como
                # orden rechazada y el ciclo siguiente reconcilia.
                return set()
        return {s for s in symbols if s not in self._assets}
