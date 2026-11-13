"""v1.6: fundamentales SEC point-in-time, costes por plaza, universos y adaptadores.

Sin red ni claves: hechos XBRL, sitemaps y clientes de broker son FALSOS.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from swarmtrader.data import fundamentals as F
from swarmtrader.data.labels import abdi_ranaldo_spread
from swarmtrader.data.venues import (IBKR, LIGHTYEAR_EU, canonical_symbol, lightyear_universe,
                                     min_viable_notional, round_trip_cost)

D = pd.Timestamp


def _facts(rows):
    df = pd.DataFrame(rows, columns=["cik", "concept", "prio", "start", "end", "filed",
                                     "form", "accn", "val"])
    for c in ("start", "end", "filed"):
        df[c] = pd.to_datetime(df[c])
    return df


def _company(cik=1, ni=(10.0, 20.0), assets=(100.0, 110.0, 120.0)):
    """Dos ejercicios anuales (2022, 2023) presentados en marzo del ano siguiente."""
    r = []
    a2021, a2022, a2023 = assets
    r += [(cik, "assets", 0, None, "2021-12-31", "2022-03-01", "10-K", "a1", a2021),
          (cik, "assets", 0, None, "2022-12-31", "2023-03-01", "10-K", "a2", a2022),
          (cik, "assets", 0, None, "2023-12-31", "2024-03-01", "10-K", "a3", a2023)]
    r += [(cik, "net_income", 0, "2022-01-01", "2022-12-31", "2023-03-01", "10-K", "a2", ni[0]),
          (cik, "net_income", 0, "2023-01-01", "2023-12-31", "2024-03-01", "10-K", "a3", ni[1])]
    r += [(cik, "cfo", 0, "2022-01-01", "2022-12-31", "2023-03-01", "10-K", "a2", 15.0),
          (cik, "cfo", 0, "2023-01-01", "2023-12-31", "2024-03-01", "10-K", "a3", 30.0)]
    r += [(cik, "equity", 0, None, "2023-12-31", "2024-03-01", "10-K", "a3", 60.0)]
    return r


# ------------------------------------------------------------ fundamentales PIT
def test_first_reported_keeps_the_first_filing_not_the_restatement():
    rows = _company()
    # reexpresion del beneficio de 2022 presentada en 2024: no debe sustituir al original
    rows.append((1, "net_income", 0, "2022-01-01", "2022-12-31", "2024-03-01", "10-K", "a3", 99.0))
    P = F.first_reported(_facts(rows))
    v = P[(P.concept == "net_income") & (P.end == D("2022-12-31"))]["val"]
    assert list(v) == [10.0]


def test_flow_duration_is_classified_so_quarter_and_ytd_never_mix():
    rows = [(1, "eps", 0, "2023-04-01", "2023-06-30", "2023-08-01", "10-Q", "q2", 0.5),
            (1, "eps", 0, "2023-01-01", "2023-06-30", "2023-08-01", "10-Q", "q2", 1.1)]
    P = F.first_reported(_facts(rows))
    assert set(P["kind"]) == {"quarter", "ytd"}


def test_snapshot_never_uses_a_value_filed_after_its_date():
    S = F.pit_snapshots(_facts(_company()))
    early = S[S["date"] == D("2023-03-01")].iloc[0]
    assert early["net_income"] == 10.0          # el de 2023 aun no era publico
    assert np.isnan(early["book_equity"])       # el patrimonio se presenta en 2024
    late = S[S["date"] == D("2024-03-01")].iloc[0]
    assert late["net_income"] == 20.0 and late["book_equity"] == 60.0
    assert late["asset_growth"] == pytest.approx(120 / 110 - 1)


def test_piotroski_counts_only_computable_components():
    ev = pd.DataFrame({k: [np.nan] for k in (
        "ni_t", "ni_p", "assets_p", "assets_pp", "cfo_t", "lt_debt_t", "lt_debt_p", "assets_t",
        "assets_cur_t", "liab_cur_t", "assets_cur_p", "liab_cur_p", "shares_dei_t",
        "shares_bs_t", "shares_dei_p", "shares_bs_p", "gp_t", "rev_t", "gp_p", "rev_p")})
    assert np.isnan(F.piotroski(ev).iloc[0]), "sin datos no se inventa una puntuacion"
    ev.loc[0, ["ni_t", "ni_p", "assets_p", "assets_pp", "cfo_t", "assets_t"]] = \
        [20, 10, 110, 100, 30, 120]
    ev.loc[0, ["assets_cur_t", "liab_cur_t", "assets_cur_p", "liab_cur_p"]] = [50, 20, 40, 20]
    ev.loc[0, ["shares_dei_t", "shares_dei_p"]] = [100, 100]
    s = F.piotroski(ev).iloc[0]
    # roa>0, cfo>0, droa>0, cfo>roa, sin deuda (<=), liquidez sube, sin emision: 7 de 7
    assert s == pytest.approx(9.0)


def test_quarterly_sue_derives_q4_and_has_the_right_sign():
    rows = []
    for y in range(2019, 2025):
        base = 1.0 + 0.1 * (y - 2019)
        for q, (s, e) in enumerate([("01-01", "03-31"), ("04-01", "06-30"), ("07-01", "09-30")]):
            rows.append((1, "eps", 0, f"{y}-{s}", f"{y}-{e}", f"{y}-{e[:2]}-28".replace(
                "03-28", "05-10").replace("06-28", "08-10").replace("09-28", "11-10"),
                "10-Q", f"{y}{q}", base + 0.01 * q))
        rows.append((1, "eps", 0, f"{y}-01-01", f"{y}-09-30", f"{y}-11-10", "10-Q", f"{y}9", 3 * base))
        rows.append((1, "eps", 0, f"{y}-01-01", f"{y}-12-31", f"{y + 1}-02-20", "10-K", f"{y}K",
                     4 * base + (2.0 if y == 2024 else 0.0)))   # sorpresa en el 4T de 2024
    P = F.first_reported(_facts(rows))
    sue = F.quarterly_sue(P)
    q4 = sue[sue["end"] == D("2024-12-31")]
    assert len(q4) == 1 and q4["sue"].iloc[0] > 3, "el 4T se deriva de anual - 9 meses"
    assert q4["date"].iloc[0] == D("2025-02-20"), "fecha del evento = presentacion del 10-K"


def test_signal_tables_use_the_current_ticker_map():
    S = F.pit_snapshots(_facts(_company(cik=7)))
    tick = pd.DataFrame({"cik": [7, 7], "ticker": ["ABC", "ABC-B"], "exchange": ["NYSE"] * 2})
    fund, _ = F.signal_tables(S, pd.DataFrame(columns=["cik", "date", "end", "sue"]), tick)
    assert set(fund["symbol"]) == {"ABC", "ABC-B"}


def test_market_cap_uses_nominal_price(tmp_path):
    from swarmtrader.data.signals import fundamentals_features
    dates = pd.bdate_range("2024-01-01", periods=10)
    pd.DataFrame({"date": [dates[0]], "symbol": ["X"], "book_equity": [100.0],
                  "shares": [10.0]}).to_csv(tmp_path / "f.csv", index=False)
    adj = pd.DataFrame({"X": 5.0}, index=dates)          # ajustado (split posterior)
    nom = pd.DataFrame({"X": 10.0}, index=dates)         # lo que cotizaba de verdad
    f = fundamentals_features(tmp_path / "f.csv", dates, ["X"], adj, 1, nominal=nom)
    assert f["bm"].dropna().iloc[-1] == pytest.approx(100 / (10.0 * 10))


def test_value_and_quality_agents_activate_only_with_their_data():
    from swarmtrader.agents.specialists import active_roster
    act, skip = active_roster(["mom_12_1", "bm", "fscore"])
    names = {a.name for a in act}
    assert {"value", "quality"} <= names and "earnings" in skip


# ------------------------------------------------------------ spread y exclusiones
def test_abdi_ranaldo_recovers_a_known_spread():
    rng = np.random.default_rng(3)
    n, s_true = 400, 0.04
    mid = 10 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    close = mid * (1 + s_true / 2 * rng.choice([-1, 1], n))     # rebote bid-ask
    high = np.maximum(close, mid * np.exp(np.abs(rng.normal(0, 0.006, n))))
    low = np.minimum(close, mid * np.exp(-np.abs(rng.normal(0, 0.006, n))))
    idx = pd.bdate_range("2020-01-01", periods=n)
    est = abdi_ranaldo_spread(pd.DataFrame({"A": close}, idx), pd.DataFrame({"A": high}, idx),
                              pd.DataFrame({"A": low}, idx), window=252).iloc[-2, 0]
    assert 0.5 * s_true < est < 1.6 * s_true


def test_universe_exclusions_by_price_liquidity_and_spread():
    from swarmtrader.cli import universe_exclusions
    from swarmtrader.data.loaders import MarketData
    from swarmtrader.execution.plan import plan_risk_limits
    idx = pd.bdate_range("2024-01-01", periods=80)
    px = pd.DataFrame({"OK": 2.0, "TINY": 0.005, "ILLIQ": 3.0, "WIDE": 1.0}, index=idx)
    vol = pd.DataFrame({"OK": 1e6, "TINY": 1e9, "ILLIQ": 10.0, "WIDE": 1e6}, index=idx)
    rng = np.random.default_rng(1)
    bounce = pd.DataFrame(rng.choice([-1, 1], size=px.shape), index=idx, columns=px.columns)
    close = px * (1 + 0.25 / 2 * bounce.mul([0, 0, 0, 1], axis=1))
    half = px * 0.125 * pd.Series([0, 0, 0, 1], index=px.columns)  # el rango cubre el spread
    md = MarketData(px, vol, close, np.maximum(close, px + half) * 1.001,
                    np.minimum(close, px - half) * 0.999)
    ex = universe_exclusions(md, plan_risk_limits(0.01, 100_000.0), max_spread=0.10)
    got = dict(zip(ex["symbol"], ex["reason"]))
    assert got == {"TINY": "precio", "ILLIQ": "liquidez", "WIDE": "spread"}


# ------------------------------------------------------------ costes por plaza
def test_ibkr_per_share_minimum_and_one_percent_cap():
    # 25 $ a 10 $/accion: 2,5 acciones * 0,0035 = 0,009 -> minimo 0,35 por orden
    assert round_trip_cost(25.0, IBKR, price=10.0)["breakdown"]["comision"] == pytest.approx(0.70)
    # 1.000 $ de una accion de 0,01 $: 100.000 acciones * 0,0035 = 350 $ -> tope 1 % = 10 $
    c = round_trip_cost(1000.0, IBKR, price=0.01)["breakdown"]["comision"]
    assert c == pytest.approx(20.0)


def test_lightyear_minimum_and_cap_per_order():
    assert round_trip_cost(4.0, LIGHTYEAR_EU)["breakdown"]["comision"] == pytest.approx(0.20)
    assert round_trip_cost(5000.0, LIGHTYEAR_EU)["breakdown"]["comision"] == pytest.approx(2.0)


@pytest.mark.parametrize("venue,kw", [(IBKR, {"price": 10.0}), (LIGHTYEAR_EU, {})])
def test_min_viable_notional_bisection_is_consistent(venue, kw):
    m = min_viable_notional(venue, 0.02, **kw)
    assert np.isfinite(m) and m > 0
    assert round_trip_cost(m * 1.01, venue, **kw)["pct"] <= 0.02 + 1e-6
    assert round_trip_cost(m * 0.9, venue, **kw)["pct"] > 0.02


def test_canonical_symbols():
    assert canonical_symbol("BRK.B", "NYSE") == "BRK-B"
    assert canonical_symbol("HUSQ B", "XSTO") == "HUSQ-B.ST"
    assert canonical_symbol("SAN", "BME") == "SAN.MC"
    assert canonical_symbol("KNIN", "XGAT") is None


def test_lightyear_universe_from_sitemap(tmp_path):
    xml = ("<urlset><url><loc>https://lightyear.com/en/stocks/explore</loc></url>"
           "<url><loc>https://lightyear.com/en/stock/AAPL:NASDAQ</loc></url>"
           "<url><loc>https://lightyear.com/en/stock/HUSQ%20B:XSTO</loc></url></urlset>")
    (tmp_path / "s.xml").write_text(xml)
    u = lightyear_universe(str(tmp_path / "s.xml"))
    assert u["canonical"].tolist() == ["AAPL", "HUSQ-B.ST"]


# ------------------------------------------------------------ adaptadores
class _Resp:
    def __init__(self, data, code=200):
        self._d, self.status_code, self.headers, self.content = data, code, {}, b"x"

    def json(self):
        return self._d

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)


class _Session:
    def __init__(self):
        self.posts = []

    def get(self, url, params=None, headers=None, timeout=None):
        if url.endswith("/portfolio"):
            return _Resp({"clientPortfolio": {"credit": 40.0, "positions": [
                {"positionID": 11, "instrumentID": 1001, "isBuy": True, "units": 2.0,
                 "amount": 20.0, "unrealizedPnL": 2.0},
                {"positionID": 12, "instrumentID": 1001, "isBuy": True, "units": 1.0,
                 "amount": 10.0, "unrealizedPnL": 1.0}]}})
        if url.endswith("/rates"):
            return _Resp({"rates": [{"instrumentID": 1001, "bid": 10.9, "ask": 11.1}]})
        return _Resp({"items": []})

    def post(self, url, json=None, headers=None, timeout=None):
        assert headers["x-request-id"] and headers["x-api-key"] == "k"
        self.posts.append((url, json))
        return _Resp({"orderForOpen": {"orderID": 77}})


def _etoro(**kw):
    from swarmtrader.execution.etoro import EToroBroker, EToroClient
    sess = _Session()
    c = EToroClient(api_key="k", user_key="u", session=sess)
    return EToroBroker(client=c, symbol_map={"AAA": 1001}, **kw), sess


def test_etoro_buy_is_long_unleveraged_by_amount_on_demo():
    b, sess = _etoro()
    r = b.submit("AAA", "buy", 12.345)
    url, body = sess.posts[-1]
    assert "/execution/demo/market-open-orders/by-amount" in url
    assert body == {"InstrumentID": 1001, "IsBuy": True, "Leverage": 1, "Amount": 12.35}
    assert r.broker_order_id == "77"


def test_etoro_positions_and_equity_aggregate_lots():
    b, _ = _etoro()
    p = b.positions()["AAA"]
    assert p.qty == 3.0 and p.market_value == pytest.approx(33.0)
    assert b.equity == pytest.approx(73.0) and b.cash == 40.0


def test_etoro_partial_sell_closes_oldest_lot_first():
    b, sess = _etoro()
    b.submit("AAA", "sell", 11.0)                 # ~1 accion a 11 $
    url, body = sess.posts[-1]
    assert url.endswith("/positions/11") and body["UnitsToDeduct"] == pytest.approx(1.0)


def test_etoro_real_needs_the_environment_key(monkeypatch):
    monkeypatch.delenv("SWARMTRADER_ALLOW_LIVE", raising=False)
    with pytest.raises(RuntimeError):
        _etoro(demo=False)


class _IB:
    def __init__(self):
        from types import SimpleNamespace as NS
        self.NS = NS
        self.orders = []

    def accountSummary(self, acct=""):
        NS = self.NS
        return [NS(tag="NetLiquidation", value="100", currency="USD"),
                NS(tag="TotalCashValue", value="60", currency="USD"),
                NS(tag="AvailableFunds", value="55", currency="USD")]

    def positions(self, acct=""):
        return [self.NS(contract=self.NS(symbol="PNY", secType="STK"), position=100,
                        avgCost=0.2)]

    def qualifyContracts(self, c):
        return [c] if c.symbol != "NOPE" else []

    def reqTickers(self, c):
        return [self.NS(ask=0.2345, bid=0.2301, last=0.232, close=0.23)]

    def placeOrder(self, c, o):
        self.orders.append((c, o))
        return self.NS(order=self.NS(orderId=5), orderStatus=self.NS(status="Submitted"))


def test_ibkr_buys_whole_shares_with_a_marketable_limit():
    pytest.importorskip("ib_async")
    from swarmtrader.execution.ibkr import IBKRBroker
    ib = _IB()
    b = IBKRBroker(ib=ib, slippage_cap=0.02)
    r = b.submit("PNY", "buy", 10.0)
    _, o = ib.orders[-1]
    assert o.totalQuantity == 42 and o.action == "BUY"
    assert o.lmtPrice == pytest.approx(0.2391)       # ask * 1,02 truncado a 0,0001
    assert r.qty == 42 and b.buying_power == 55.0


def test_ibkr_live_port_needs_the_environment_key(monkeypatch):
    from swarmtrader.execution.ibkr import IBKRBroker
    monkeypatch.delenv("SWARMTRADER_ALLOW_LIVE", raising=False)
    with pytest.raises(RuntimeError):
        IBKRBroker(ib=_IB(), port=7496)


def test_manual_broker_writes_tickets_and_never_claims_a_fill(tmp_path):
    from swarmtrader.execution.manual import ManualBroker
    pos = tmp_path / "p.csv"
    pos.write_text("symbol,qty,avg_price,market_value\nAAA,2,10,22\n")
    b = ManualBroker("lightyear", cash=10.0, positions_csv=str(pos),
                     prices={"BBB": 5.0}, tickets_path=str(tmp_path / "t.csv"))
    assert b.equity == pytest.approx(32.0)
    s = b.liquidate("AAA")
    r = b.submit("BBB", "buy", 25.0)
    assert {s.status, r.status} == {"manual"}
    rows = (tmp_path / "t.csv").read_text().splitlines()
    assert len(rows) == 3 and rows[1].startswith("lightyear,AAA,sell")
    assert b.cash == pytest.approx(7.0)


def test_costs_use_the_estimate_only_where_it_is_informative():
    from swarmtrader.data.loaders import MarketData
    from swarmtrader.decision import DecisionConfig, estimate_costs
    idx = pd.bdate_range("2024-01-01", periods=80)
    rng = np.random.default_rng(2)
    px = pd.DataFrame({"BIG": 50.0, "PNY": 0.5}, index=idx)
    vol = pd.DataFrame({"BIG": 1e6, "PNY": 1e6}, index=idx)          # 50 M$ / 0,5 M$
    b = pd.DataFrame(rng.choice([-1, 1], size=px.shape), index=idx, columns=px.columns)
    close = px * (1 + 0.05 * b)
    md = MarketData(px, vol, close, px * 1.06, px * 0.94)
    c = estimate_costs(md, close.iloc[-1], DecisionConfig())
    assert c["BIG"] == pytest.approx(0.0005)          # liquida: tabla
    assert 0.04 < c["PNY"] < 0.2                       # penny poco liquida: estimacion
