"""Tests de la v0.7: point-in-time, ritmo de brokers, almacen y validacion cruzada.

Cada test comprueba una PROPIEDAD que, si se rompe, invalida silenciosamente los
resultados. No comprueban "el codigo corre".
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from swarmtrader.data.pacing import (ALPACA_PLUS_LIMITS, ETORO_LIMITS, IBKR_LIMITS,
                                     BrokerLimits, RateLimiter, assess_bulk_download)
from swarmtrader.data.pit import (DelistReason, PITPanel, forward_returns_pit,
                                  load_synthetic_pit, universe_asof)
from swarmtrader.data.store import PanelStore
from swarmtrader.rank import RankConfig, _prepare, diagnose, walk_forward_oos
from swarmtrader.stats.spa import spa_crosscheck, spa_test


# ------------------------------------------------------------------ PIT
@pytest.fixture(scope="module")
def panel():
    return load_synthetic_pit(n_symbols=50, n_days=800, annual_delist_rate=0.12, seed=3)


def test_panel_contains_delisted_symbols(panel):
    assert panel.n_delisted > 0
    assert DelistReason.BANKRUPT in set(panel.listings["reason"])


def test_universe_shrinks_over_time(panel):
    first = panel.universe_asof(panel.prices.index[0])
    last = panel.universe_asof(panel.prices.index[-1])
    assert len(last) < len(first), "sin bajas no hay nada que probar"


def test_universe_asof_is_point_in_time(panel):
    """Un simbolo muerto no puede aparecer en el universo de una fecha posterior."""
    L = panel.listings
    dead = L.index[L["reason"] != DelistReason.ACTIVE]
    for sym in dead[:5]:
        after = L.at[sym, "last_date"] + pd.Timedelta(days=5)
        if after <= panel.prices.index[-1]:
            assert sym not in panel.universe_asof(after)


def test_pit_label_recovers_observations_the_naive_shift_drops(panel):
    """La propiedad central del modulo: la etiqueta ingenua tira las quiebras."""
    h = 21
    pit = forward_returns_pit(panel.prices, panel.listings, h)
    naive = (panel.prices.shift(-h) / panel.prices - 1).stack(future_stack=True)
    naive.index.names = ["date", "symbol"]
    extra = pit.dropna().index.difference(naive.dropna().index)
    assert len(extra) > 0, "deberia recuperar observaciones de los que mueren dentro del horizonte"
    # y esas observaciones tienen que ser MALAS: es el punto entero
    assert pit.loc[extra].mean() < 0


def test_bankruptcy_label_is_catastrophic(panel):
    h = 21
    pit = forward_returns_pit(panel.prices, panel.listings, h)
    bank = panel.listings.index[panel.listings["reason"] == DelistReason.BANKRUPT]
    if not len(bank):
        pytest.skip("esta semilla no genero quiebras")
    naive = (panel.prices.shift(-h) / panel.prices - 1).stack(future_stack=True)
    naive.index.names = ["date", "symbol"]
    extra = pit.dropna().index.difference(naive.dropna().index)
    sub = pit.loc[extra]
    sub_b = sub[sub.index.get_level_values("symbol").isin(bank)]
    assert sub_b.mean() < -0.30, f"retorno terminal de quiebra demasiado benigno: {sub_b.mean():.2%}"


def test_no_observation_after_delisting(panel):
    """Despues de morir no puede haber etiqueta: seria operar un valor inexistente."""
    pit = forward_returns_pit(panel.prices, panel.listings, 21)
    L = panel.listings
    dead = L.index[L["reason"] != DelistReason.ACTIVE]
    for sym in dead[:5]:
        after = pit.xs(sym, level="symbol")
        after = after[after.index > L.at[sym, "last_date"]]
        assert after.notna().sum() == 0


def test_survivors_view_is_strictly_smaller(panel):
    biased = panel.survivors_only()
    assert biased.prices.shape[1] < panel.prices.shape[1]
    assert biased.n_delisted == 0


def test_pipeline_accepts_listings_without_breaking(panel):
    cfg = RankConfig(oos_months=6)
    md = panel.to_market_data()
    df_naive = _prepare(md, cfg)
    df_pit = _prepare(md, cfg, listings=panel.listings)
    assert set(df_naive.attrs["factor_cols"]) == set(df_pit.attrs["factor_cols"])
    # con PIT hay MAS observaciones etiquetadas
    assert df_pit["target"].notna().sum() >= df_naive["target"].notna().sum()
    d = diagnose(walk_forward_oos(df_pit, cfg), cfg)
    assert d.n_oos_dates > 0


def test_universe_asof_helper_matches_method(panel):
    d = panel.prices.index[len(panel.prices) // 2]
    assert universe_asof(panel.listings, d) == panel.universe_asof(d)


def test_panel_rejects_malformed_listings():
    px = pd.DataFrame(np.ones((10, 2)), index=pd.date_range("2024-01-01", periods=10),
                      columns=["A", "B"])
    with pytest.raises(ValueError):
        PITPanel(px, px, pd.DataFrame({"foo": [1, 2]}, index=["A", "B"]))


# ------------------------------------------------------------------ ritmo
class _Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def sleep(self, s):
        self.t += s


def test_sliding_window_blocks_the_61st_ibkr_request():
    c = _Clock()
    rl = RateLimiter(IBKR_LIMITS, sleep=c.sleep, clock=c)
    for i in range(60):
        assert rl.acquire(f"s{i}") == 0.0
    assert rl.acquire("s60") == pytest.approx(600.0)


def test_identical_request_respects_15_second_rule():
    c = _Clock()
    rl = RateLimiter(IBKR_LIMITS, sleep=c.sleep, clock=c)
    rl.acquire("AAPL|SMART|TRADES")
    assert rl.acquire("AAPL|SMART|TRADES") == pytest.approx(15.0)


def test_burst_rule_for_same_contract():
    lim = BrokerLimits(name="t", max_requests=10_000, window_seconds=600,
                       burst_same_key=3, burst_window_seconds=2.0)
    c = _Clock()
    rl = RateLimiter(lim, sleep=c.sleep, clock=c)
    for _ in range(3):
        rl.acquire("X")
    assert rl.acquire("X") == pytest.approx(2.0)


def test_etoro_minute_window():
    c = _Clock()
    rl = RateLimiter(ETORO_LIMITS, sleep=c.sleep, clock=c)
    for i in range(60):
        rl.acquire()
    assert rl.acquire() == pytest.approx(60.0)


def test_window_frees_up_after_it_passes():
    c = _Clock()
    rl = RateLimiter(ETORO_LIMITS, sleep=c.sleep, clock=c)
    for i in range(60):
        rl.acquire()
    c.t += 61.0
    assert rl.acquire() == 0.0


def test_ibkr_bulk_download_is_refused_at_scale():
    r = assess_bulk_download(IBKR_LIMITS, 5000, 1008)
    assert not r.feasible
    assert r.hours > 10
    assert any("point-in-time" in x for x in r.reasons)


def test_alpaca_bulk_download_is_feasible():
    assert assess_bulk_download(ALPACA_PLUS_LIMITS, 5000, 1008).feasible


def test_etoro_needs_two_requests_per_symbol_for_four_years():
    r = assess_bulk_download(ETORO_LIMITS, 100, 1008)
    assert r.requests_needed == 200, "1008 sesiones no caben en 1000 velas"


# ------------------------------------------------------------------ almacen
def test_store_roundtrip_preserves_panel_and_listings(panel):
    with tempfile.TemporaryDirectory() as td:
        st = PanelStore(Path(td) / "p")
        info = st.write(panel.prices, panel.volumes, panel.listings)
        assert info["rows"] > 0
        md = st.to_market_data()
        assert md.prices.shape[1] == panel.prices.shape[1]
        L = st.listings()
        assert L is not None and L.index.name == "symbol"
        assert (L["reason"] != DelistReason.ACTIVE).sum() == panel.n_delisted


def test_store_cross_section_returns_one_date(panel):
    with tempfile.TemporaryDirectory() as td:
        st = PanelStore(Path(td) / "p")
        st.write(panel.prices, panel.volumes)
        d = panel.prices.index[400]
        cs = st.cross_section(d)
        assert len(cs) > 0
        assert (pd.to_datetime(cs["date"]) == d).all()


def test_store_filters_by_symbol_and_range(panel):
    with tempfile.TemporaryDirectory() as td:
        st = PanelStore(Path(td) / "p")
        st.write(panel.prices, panel.volumes)
        syms = list(panel.prices.columns[:3])
        out = st.read_long(start=panel.prices.index[100], end=panel.prices.index[200],
                           symbols=syms)
        assert set(out["symbol"]) <= set(syms)
        assert out["date"].min() >= panel.prices.index[100]
        assert out["date"].max() <= panel.prices.index[200]


def test_store_does_not_materialise_empty_cells(panel):
    """El formato largo debe omitir las celdas vacias de los deslistados."""
    with tempfile.TemporaryDirectory() as td:
        st = PanelStore(Path(td) / "p")
        info = st.write(panel.prices, panel.volumes)
        assert info["rows"] < panel.prices.shape[0] * panel.prices.shape[1]


# ------------------------------------------------------------------ SPA arch
def test_spa_crosscheck_agrees_with_reference_on_the_decision():
    """Lo que debe coincidir es la DECISION, no el p-valor al cuarto decimal."""
    cases = []
    r = np.random.default_rng(11)
    cases.append(r.normal(0, 0.01, (180, 40)))                       # nulo
    x = r.normal(0, 0.01, (180, 40)); x[:, 7] += 0.004
    cases.append(x)                                                  # edge claro
    y = np.zeros((300, 10)); e = r.normal(0, 0.01, (300, 10))
    for t in range(1, 300):
        y[t] = 0.6 * y[t - 1] + e[t]
    cases.append(y)                                                  # autocorrelado
    for d in cases:
        c = spa_crosscheck(d, n_boot=800, seed=5)
        if not c["available"]:
            pytest.skip("arch no instalado")
        assert (c["p_own"] < 0.05) == (c["p_arch"] < 0.05), c


def test_spa_crosscheck_reports_when_arch_missing(monkeypatch):
    import builtins
    real = builtins.__import__

    def fake(name, *a, **k):
        if name.startswith("arch"):
            raise ImportError("simulado")
        return real(name, *a, **k)

    monkeypatch.setattr(builtins, "__import__", fake)
    c = spa_crosscheck(np.random.default_rng(0).normal(0, 0.01, (60, 5)), n_boot=200)
    assert c["available"] is False and c["p_own"] is not None


# ------------------------------------------------------------------ arquitecturas
def test_architectures_are_permutation_equivariant():
    torch = pytest.importorskip("torch")
    from swarmtrader.models.deep import ARCHS, CrossSectionalRanker, DeepConfig

    for a in ARCHS:
        m = CrossSectionalRanker(14, 2, 8, DeepConfig(arch=a, seq_len=40)).eval()
        seq, ft = torch.randn(30, 40, 2), torch.randn(30, 14)
        sec = torch.randint(0, 8, (30,))
        p = torch.randperm(30)
        with torch.no_grad():
            a1, a2 = m(seq, ft, sec)[p], m(seq[p], ft[p], sec[p])
        assert float((a1 - a2).abs().max()) < 1e-5, f"{a} no es equivariante"


def test_architectures_accept_variable_universe_size():
    torch = pytest.importorskip("torch")
    from swarmtrader.models.deep import ARCHS, CrossSectionalRanker, DeepConfig

    for a in ARCHS:
        m = CrossSectionalRanker(14, 2, 8, DeepConfig(arch=a, seq_len=40)).eval()
        for n in (6, 50, 300):
            with torch.no_grad():
                s = m(torch.randn(n, 40, 2), torch.randn(n, 14), torch.randint(0, 8, (n,)))
            assert s.shape == (n,)


def test_auto_arch_is_a_cost_rule_not_a_score_rule():
    pytest.importorskip("torch")
    from swarmtrader.models.deep import DeepConfig
    from swarmtrader.models.trainer import resolve_arch

    assert resolve_arch(DeepConfig(), 100).arch == "ensemble"
    assert resolve_arch(DeepConfig(), 5000).arch == "mixer"        # O(N), no por puntuacion
    assert resolve_arch(DeepConfig(arch="attn"), 5000).arch == "attn"   # explicito manda
