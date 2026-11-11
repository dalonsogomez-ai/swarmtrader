"""Tests del universo multi-plaza y del router.

La propiedad central: anadir plazas NO puede empeorar el sistema. El router
debe negarse a enrutar donde el coste se come la ventaja, en vez de ejecutar
una operacion que nace perdiendo.
"""
from __future__ import annotations

import pytest

from swarmtrader.data.venues import (ALPACA, ETORO_EU, ETORO_US, IBKR, VenueSpec,
                                     build_union, min_viable_notional, round_trip_cost)
from swarmtrader.execution.paper import PaperBroker
from swarmtrader.execution.router import VenueRouter


@pytest.fixture
def universe():
    return build_union({
        "alpaca": ["AAPL", "MSFT", "SOFI"],
        "etoro_eu": ["AAPL", "MSFT", "SAN.MC"],
        "ibkr": ["AAPL", "SAN.MC", "7203.T"],
    }, venues={"alpaca": ALPACA, "etoro_eu": ETORO_EU, "ibkr": IBKR})


@pytest.fixture
def router(universe):
    brokers = {"alpaca": PaperBroker(initial_cash=80.0),
               "etoro_eu": PaperBroker(initial_cash=20.0),
               "ibkr": PaperBroker(initial_cash=0.0)}
    for b in brokers.values():
        b.set_prices({s: 50.0 for s in universe.symbols})
    return VenueRouter(brokers=brokers, universe=universe,
                       fx_venues=("etoro_eu",), max_cost_pct=0.02)


# ------------------------------------------------------------------ universo
def test_union_is_not_the_sum_of_parts(universe):
    s = universe.summary()
    assert s["total_union"] < sum(s["por_plaza"].values()), "el solapamiento no se cruza"
    assert s["en_las_tres"] >= 1


def test_symbol_normalisation_strips_exchange_suffix(universe):
    assert "SAN" in universe.symbols
    assert "SAN.MC" not in universe.symbols


def test_executable_on_reflects_membership(universe):
    assert set(universe.executable_on("AAPL")) == {"alpaca", "etoro_eu", "ibkr"}
    assert universe.executable_on("SOFI") == ["alpaca"]
    assert universe.executable_on("NOEXISTE") == []


def test_venue_that_cannot_execute_is_excluded():
    solo_datos = VenueSpec(name="solo_datos", can_execute=False)
    u = build_union({"solo_datos": ["XYZ"]}, venues={"solo_datos": solo_datos})
    assert u.executable_on("XYZ") == []
    assert u.tradable_symbols() == []


# ------------------------------------------------------------------ costes
def test_fixed_commission_destroys_small_positions():
    barato = round_trip_cost(25.0, ALPACA)["pct"]
    caro = round_trip_cost(25.0, ETORO_EU, fx=True)["pct"]
    assert barato < 0.001
    assert caro > 0.15, "una comision fija sobre 25 $ debe ser demoledora"


def test_cost_falls_with_size_for_fixed_commission():
    a = round_trip_cost(25.0, IBKR)["pct"]
    b = round_trip_cost(1000.0, IBKR)["pct"]
    assert b < a / 10


def test_min_viable_notional_is_consistent_with_cost():
    m = min_viable_notional(ETORO_EU, 0.02, fx=True)
    assert round_trip_cost(m * 1.01, ETORO_EU, fx=True)["pct"] <= 0.02 + 1e-6
    assert round_trip_cost(m * 0.5, ETORO_EU, fx=True)["pct"] > 0.02


def test_zero_commission_venue_has_no_minimum():
    assert min_viable_notional(ALPACA, 0.02) == 0.0
    assert min_viable_notional(ETORO_US, 0.02) == 0.0


def test_proportional_cost_above_limit_is_never_viable():
    caro = VenueSpec(name="caro", spread_bps=500.0)      # 5% de spread
    assert min_viable_notional(caro, 0.02) == float("inf")


# ------------------------------------------------------------------ router
def test_router_consolidates_equity_and_cash(router):
    assert router.equity == pytest.approx(100.0)
    assert router.cash_by_venue() == {"alpaca": 80.0, "etoro_eu": 20.0, "ibkr": 0.0}


def test_router_rejects_symbol_no_venue_offers(router):
    d = router.route("NOEXISTE", 25.0)
    assert d.venue is None and "no operable" in d.reason


def test_router_refuses_venues_where_cost_eats_the_edge(router):
    """Con 25 $, eToro EU cuesta 17,6% e IBKR 8%: ninguna es viable."""
    d = router.route("SAN", 25.0)          # solo en etoro_eu e ibkr
    assert d.venue is None
    assert any("coste" in x for x in router.routing_log)


def test_router_picks_the_cheapest_viable_venue(router):
    d = router.route("AAPL", 25.0)
    assert d.venue == "alpaca" and d.cost_pct < 0.001


def test_router_checks_cash_per_venue_not_aggregate(universe):
    """80 $ en Alpaca + 20 $ en eToro NO son 100 $ para una compra en eToro."""
    brokers = {"alpaca": PaperBroker(initial_cash=80.0),
               "etoro_eu": PaperBroker(initial_cash=20.0)}
    for b in brokers.values():
        b.set_prices({s: 50.0 for s in universe.symbols})
    r = VenueRouter(brokers=brokers, universe=universe, fx_venues=(),
                    max_cost_pct=1.0)                     # sin filtro de coste
    d = r.route("SAN", 60.0)                              # solo en etoro_eu, que tiene 20 $
    assert d.venue is None
    assert any("efectivo" in x for x in r.routing_log)


def test_submit_returns_rejected_instead_of_failing_silently(router):
    res = router.submit("SAN", "buy", 25.0)
    assert res.status == "rejected" and res.notional == 0.0


def test_submit_routes_and_reports_the_venue(router):
    res = router.submit("AAPL", "buy", 25.0)
    assert res.status == "filled" and "[alpaca]" in res.message


def test_router_is_a_drop_in_broker_adapter(router):
    """El ciclo no debe necesitar cambios para operar en varias plazas."""
    from swarmtrader.execution.base import BrokerAdapter

    assert isinstance(router, BrokerAdapter)
    for m in ("equity", "cash", "positions", "last_price", "submit", "liquidate",
              "rebalance_to"):
        assert hasattr(router, m)


def test_positions_are_aggregated_across_venues(universe):
    from swarmtrader.execution.base import Position

    brokers = {"alpaca": PaperBroker(initial_cash=100.0),
               "etoro_eu": PaperBroker(initial_cash=100.0)}
    for b in brokers.values():
        b.set_prices({s: 50.0 for s in universe.symbols})
    brokers["alpaca"]._positions["AAPL"] = Position("AAPL", 1.0, 50.0, 50.0)
    brokers["etoro_eu"]._positions["AAPL"] = Position("AAPL", 2.0, 50.0, 100.0)
    r = VenueRouter(brokers=brokers, universe=universe)
    p = r.positions()["AAPL"]
    assert p.qty == 3.0 and p.market_value == 150.0


def test_liquidate_closes_in_every_venue(universe):
    from swarmtrader.execution.base import Position

    brokers = {"alpaca": PaperBroker(initial_cash=100.0),
               "etoro_eu": PaperBroker(initial_cash=100.0)}
    for b in brokers.values():
        b.set_prices({s: 50.0 for s in universe.symbols})
        b._positions["AAPL"] = Position("AAPL", 1.0, 50.0, 50.0)
    r = VenueRouter(brokers=brokers, universe=universe)
    res = r.liquidate("AAPL")
    assert res.status == "filled"
    assert r.positions().get("AAPL") is None


def test_adding_venues_cannot_make_execution_worse(router):
    """Propiedad central: si una plaza nueva es cara, el router no la usa."""
    solo_alpaca = VenueRouter(brokers={"alpaca": router.brokers["alpaca"]},
                              universe=router.universe, max_cost_pct=0.02)
    for sym in ("AAPL", "MSFT", "SOFI"):
        a = solo_alpaca.route(sym, 25.0)
        b = router.route(sym, 25.0)
        if a.venue is not None:
            assert b.venue is not None
            assert b.cost_pct <= a.cost_pct + 1e-9
