"""Adaptador de Alpaca contra un cliente FALSO: sin red, sin claves, sin ordenes.

Comprueba las reglas de Alpaca que afectan a una cuenta pequena solo largo:
nocional con 2 decimales y minimo de 1 USD, poder de compra sin margen, espera
a que las ventas se ejecuten (con el mercado cerrado no se espera: quedan en
cola) y exclusion de activos no fraccionables. Y que la doble llave para dinero
real sigue exigiendose aunque se inyecten clientes.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

pytest.importorskip("alpaca")

from swarmtrader.execution.alpaca import AlpacaBroker  # noqa: E402
from swarmtrader.execution.base import OrderResult  # noqa: E402


class FakeTrading:
    def __init__(self, is_open=True, final_status="filled"):
        self.account = SimpleNamespace(equity="100", cash="40", buying_power="40",
                                       non_marginable_buying_power="38.5")
        self.clock = SimpleNamespace(is_open=is_open)
        self.final_status = final_status
        self.submitted, self.closed, self.polled = [], [], []
        self._orders: dict[str, SimpleNamespace] = {}

    def get_account(self):
        return self.account

    def get_clock(self):
        return self.clock

    def get_all_positions(self):
        return []

    def submit_order(self, req):
        o = SimpleNamespace(id=f"o{len(self.submitted)}", qty=None, status="accepted")
        self.submitted.append(req)
        self._orders[o.id] = o
        return o

    def close_position(self, symbol):
        o = SimpleNamespace(id=f"c{len(self.closed)}", qty="1.5", status="accepted")
        self.closed.append(symbol)
        self._orders[o.id] = o
        return o

    def get_order_by_id(self, oid):
        self.polled.append(oid)
        o = self._orders[oid]
        o.status = self.final_status
        return o

    def get_all_assets(self, req):
        return [SimpleNamespace(symbol="AAA", tradable=True, fractionable=True),
                SimpleNamespace(symbol="BBB", tradable=True, fractionable=False),
                SimpleNamespace(symbol="CCC", tradable=False, fractionable=True)]


def _broker(**kw):
    t = FakeTrading(**kw)
    b = AlpacaBroker(paper=True, trading_client=t, data_client=object())
    b.poll_s = 0.0
    return b, t


def test_live_mode_still_needs_the_environment_key(monkeypatch):
    monkeypatch.delenv("SWARMTRADER_ALLOW_LIVE", raising=False)
    with pytest.raises(RuntimeError):
        AlpacaBroker(paper=False, trading_client=FakeTrading(), data_client=object())


def test_buying_power_takes_the_most_conservative_field():
    b, _ = _broker()
    assert b.buying_power == pytest.approx(38.5)


def test_notional_is_truncated_to_cents():
    b, t = _broker()
    b.submit("AAA", "buy", 10.239)
    assert float(t.submitted[-1].notional) == pytest.approx(10.23)


def test_orders_below_one_dollar_are_refused_locally():
    b, t = _broker()
    r = b.submit("AAA", "buy", 0.995)
    assert r.status == "rejected" and not t.submitted


def test_liquidate_returns_the_closing_order_id():
    b, t = _broker()
    r = b.liquidate("AAA")
    assert r.broker_order_id == "c0" and t.closed == ["AAA"]


def test_settle_polls_until_filled_when_market_is_open():
    b, t = _broker(is_open=True)
    b.settle([b.submit("AAA", "sell", 10.0)])
    assert t.polled == ["o0"]


def test_settle_ignores_orders_without_id_or_already_final():
    b, t = _broker(is_open=True)
    b.settle([OrderResult("AAA", "sell", 0.0, 0.0, "rejected"),
              OrderResult("BBB", "sell", 1.0, 5.0, "filled", broker_order_id="x")])
    assert t.polled == []


def test_settle_does_not_wait_when_market_is_closed():
    b, t = _broker(is_open=False)
    b.settle([b.submit("AAA", "sell", 10.0)])
    assert t.polled == []


def test_settle_gives_up_after_the_timeout():
    b, t = _broker(is_open=True, final_status="accepted")
    b.settle_timeout_s = 0.05
    b.settle([b.submit("AAA", "sell", 10.0)])          # no se queda colgado
    assert len(t.polled) >= 1


def test_untradable_flags_inactive_and_unknown_but_not_non_fractionable():
    # v1.6: un no fraccionable SE PUEDE comprar por acciones enteras
    b, _ = _broker()
    assert b.untradable(["AAA", "BBB", "CCC", "ZZZ"]) == {"CCC", "ZZZ"}
    assert b.non_fractionable(["AAA", "BBB", "CCC"]) == {"BBB"}


class _Px(AlpacaBroker):
    """Precio fijo: evita el cliente de datos en los tests."""
    px = 0.4

    def last_price(self, symbol):
        return self.px


def _px_broker(px=0.4):
    t = FakeTrading()
    b = _Px(paper=True, trading_client=t, data_client=object())
    b.px = px
    return b, t


def test_non_fractionable_buy_is_sent_as_whole_shares():
    b, t = _px_broker(0.4)
    r = b.submit("BBB", "buy", 10.0)
    req = t.submitted[-1]
    assert req.qty == 25 and req.notional is None
    assert r.notional == pytest.approx(10.0)


def test_non_fractionable_buy_below_one_share_is_refused_locally():
    b, t = _px_broker(30.0)
    r = b.submit("BBB", "buy", 12.0)
    assert r.status == "rejected" and "no llega a una" in r.message and not t.submitted


def test_fractionable_keeps_notional_orders():
    b, t = _px_broker(0.4)
    b.submit("AAA", "buy", 10.0)
    assert float(t.submitted[-1].notional) == pytest.approx(10.0)


def test_class_shares_are_translated_at_the_boundary():
    from swarmtrader.execution.alpaca import from_alpaca, to_alpaca
    assert to_alpaca("BRK-B") == "BRK.B" and from_alpaca("BRK.B") == "BRK-B"
    assert to_alpaca("AAPL") == "AAPL" and to_alpaca("BF-A") == "BF.A"
    b, t = _px_broker(400.0)
    b.liquidate("BRK-B")
    assert t.closed == ["BRK.B"]
