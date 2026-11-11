"""Tests del motor de analogos y de las etiquetas de triple barrera.

La propiedad que decide si este modulo sirve o es un espejismo es la disciplina
point-in-time: si un analogo cuyo resultado aun no se conocia entra en la
muestra, el motor "acierta" en backtest y falla en vivo.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from swarmtrader.analogy import (DESCRIPTOR_COLS, HOLD_HORIZONS, AnalogyConfig,
                                 EventDescriptor, build_descriptors, find_analogues)
from swarmtrader.data.labels import (EXIT_DELIST, EXIT_PROFIT, EXIT_STOP, EXIT_VERTICAL,
                                     TripleBarrierConfig, corwin_schultz_spread,
                                     triple_barrier_labels)
from swarmtrader.data.loaders import load_synthetic
from swarmtrader.data.pit import load_synthetic_pit


# ------------------------------------------------------------------ analogos
@pytest.fixture(scope="module")
def market():
    md = load_synthetic(n_symbols=70, n_days=1000, signal_strength=0.0,
                        momentum_strength=0.10, seed=17)
    px, vol = md.prices.copy(), md.volumes.copy()
    sym = px.columns[2]
    loc, i0 = px.columns.get_loc(sym), px.index.get_loc(px.index[-80])
    px.iloc[i0:i0 + 3, loc] *= np.linspace(1.3, 2.0, 3)
    px.iloc[i0 + 3:, loc] *= 2.0
    vol.iloc[i0:i0 + 3, loc] *= 6
    return px, vol, sym, px.index[i0 + 2]


def test_analogues_are_strictly_point_in_time(market):
    """Ningun analogo puede tener su ventana de resultado abierta en `as_of`."""
    px, vol, sym, as_of = market
    cfg = AnalogyConfig(k=60, min_analogues=10, max_horizon=32, embargo=5)
    res = find_analogues(px, vol, sym, as_of, cfg)
    assert res.n_analogues > 0
    sessions = px.index
    pos = pd.Series(np.arange(len(sessions)), index=sessions)
    d_pos = int(pos[as_of])
    for d in res.neighbours.index.get_level_values("date"):
        assert int(pos[d]) + cfg.max_horizon + cfg.embargo <= d_pos, (
            f"fuga: analogo de {d.date()} usado en {as_of.date()}")


def test_no_analogue_comes_from_the_future(market):
    px, vol, sym, as_of = market
    res = find_analogues(px, vol, sym, as_of, AnalogyConfig(k=40, min_analogues=10))
    assert res.neighbours.index.get_level_values("date").max() < as_of


def test_refuses_when_sample_is_thin(market):
    px, vol, sym, as_of = market
    res = find_analogues(px, vol, sym, as_of, AnalogyConfig(k=5, min_analogues=40))
    assert res.actionable is False and "analogos" in res.reason


def test_refuses_when_there_is_no_event(market):
    """Sin movimiento relevante no hay nada que analizar; no se inventa respuesta."""
    px, vol, _, as_of = market
    quiet = px.columns[40]
    res = find_analogues(px, vol, quiet, as_of, AnalogyConfig(min_run_up=0.90))
    assert res.actionable is False and "umbral" in res.reason


def test_descriptors_are_scale_free(market):
    """Doblar el precio de una accion no puede cambiar su descriptor de rango."""
    px, vol, sym, _ = market
    d1 = build_descriptors(px, vol)
    px2 = px.copy()
    px2[sym] = px2[sym] * 7.0            # mismo camino, otra escala
    d2 = build_descriptors(px2, vol)
    a = d1.xs(sym, level="symbol")[["run_up_5", "run_up_21", "dd_from_peak"]].dropna()
    b = d2.xs(sym, level="symbol")[["run_up_5", "run_up_21", "dd_from_peak"]].dropna()
    common = a.index.intersection(b.index)
    assert np.allclose(a.loc[common].to_numpy(), b.loc[common].to_numpy(), atol=1e-9)


def test_forward_distribution_has_all_horizons(market):
    px, vol, sym, as_of = market
    res = find_analogues(px, vol, sym, as_of, AnalogyConfig(k=60, min_analogues=15))
    assert res.actionable
    assert set(res.forward.index) <= set(HOLD_HORIZONS)
    assert (res.forward["p25"] <= res.forward["p50"]).all()
    assert (res.forward["p50"] <= res.forward["p75"]).all()


def test_on_pure_noise_there_is_no_directional_edge():
    """Sobre ruido, la continuacion debe salir cerca del 50%: sin edge inventado."""
    md = load_synthetic(n_symbols=80, n_days=1000, signal_strength=0.0,
                        momentum_strength=0.0, seed=29)
    px, vol = md.prices.copy(), md.volumes.copy()
    sym = px.columns[1]
    loc, i0 = px.columns.get_loc(sym), px.index.get_loc(px.index[-80])
    px.iloc[i0:i0 + 3, loc] *= 1.35
    px.iloc[i0 + 3:, loc] *= 1.35
    res = find_analogues(px, vol, sym, px.index[i0 + 2],
                         AnalogyConfig(k=120, min_analogues=30))
    if res.actionable:
        assert 0.30 < res.p_continue < 0.70, f"edge fabricado sobre ruido: {res.p_continue:.0%}"


def test_descriptor_columns_are_complete(market):
    px, vol, _, _ = market
    raw = EventDescriptor.compute(px, vol)
    assert set(DESCRIPTOR_COLS) <= set(raw.columns)


# ------------------------------------------------------------ triple barrera
def _series(path: list[float], n: int = 60) -> pd.DataFrame:
    idx = pd.bdate_range("2024-01-01", periods=n)
    return pd.DataFrame({"X": np.array(path[:n], dtype=float)}, index=idx)


def test_profit_target_fires_first():
    px = _series(list(np.linspace(100, 140, 60)))
    r = triple_barrier_labels(px, TripleBarrierConfig(horizon=32, profit_target=0.10,
                                                      stop_loss=-0.08, trail_stop=None))
    row = r.xs(px.index[0], level="date").loc["X"]
    assert row["exit_reason"] == EXIT_PROFIT and row["ret"] >= 0.10


def test_stop_loss_fires_on_decline():
    px = _series(list(np.linspace(100, 70, 60)))
    r = triple_barrier_labels(px, TripleBarrierConfig(horizon=32, trail_stop=None))
    row = r.xs(px.index[0], level="date").loc["X"]
    assert row["exit_reason"] == EXIT_STOP and row["ret"] <= -0.08


def test_vertical_barrier_when_nothing_happens():
    px = _series([100.0 + 0.01 * i for i in range(60)])
    r = triple_barrier_labels(px, TripleBarrierConfig(horizon=32, trail_stop=None))
    row = r.xs(px.index[0], level="date").loc["X"]
    assert row["exit_reason"] == EXIT_VERTICAL and row["exit_h"] == 32


def test_trailing_stop_protects_gains_where_fixed_stop_does_not():
    """El caso que justifica el trailing: sube sin tocar objetivo y luego cae."""
    path = list(np.linspace(100, 108, 11)) + list(np.linspace(108, 80, 49))
    px = _series(path)
    fixed = triple_barrier_labels(px, TripleBarrierConfig(horizon=32, profit_target=0.10,
                                                          stop_loss=-0.08, trail_stop=None))
    trail = triple_barrier_labels(px, TripleBarrierConfig(horizon=32, profit_target=0.10,
                                                          stop_loss=-0.08, trail_stop=-0.04))
    rf = fixed.xs(px.index[0], level="date").loc["X"]
    rt = trail.xs(px.index[0], level="date").loc["X"]
    assert rt["ret"] > rf["ret"], "el trailing debe salir mejor en este camino"
    assert rt["exit_h"] < rf["exit_h"], "y debe salir antes"


def test_trailing_does_not_fire_before_any_profit():
    """Si nunca hubo ganancia, manda el stop fijo, no el movil."""
    px = _series(list(np.linspace(100, 93, 60)))
    r = triple_barrier_labels(px, TripleBarrierConfig(horizon=32, stop_loss=-0.08,
                                                      trail_stop=-0.02))
    row = r.xs(px.index[0], level="date").loc["X"]
    assert row["exit_reason"] in (EXIT_STOP, EXIT_VERTICAL)
    assert row["ret"] >= -0.09


def test_delisting_barrier_overrides_vertical():
    panel = load_synthetic_pit(n_symbols=30, n_days=500, annual_delist_rate=0.30,
                               pre_death_drift_days=0, seed=5)
    from swarmtrader.data.pit import DelistReason
    out = triple_barrier_labels(panel.prices, TripleBarrierConfig(horizon=21),
                                listings=panel.listings)
    dead = panel.listings.index[panel.listings["reason"] == DelistReason.BANKRUPT]
    if not len(dead):
        pytest.skip("esta semilla no genero quiebras")
    reasons = out.xs(dead[0], level="symbol")["exit_reason"]
    assert (reasons == EXIT_DELIST).any(), "la barrera de delisting no se aplico"


def test_no_label_after_delisting():
    panel = load_synthetic_pit(n_symbols=30, n_days=500, annual_delist_rate=0.30, seed=6)
    from swarmtrader.data.pit import DelistReason
    out = triple_barrier_labels(panel.prices, TripleBarrierConfig(horizon=21),
                                listings=panel.listings)
    L = panel.listings
    for sym in L.index[L["reason"] != DelistReason.ACTIVE][:3]:
        after = out.xs(sym, level="symbol")
        after = after[after.index > L.at[sym, "last_date"]]
        assert after["ret"].notna().sum() == 0


def test_exit_horizon_never_exceeds_the_vertical_barrier():
    md = load_synthetic(n_symbols=20, n_days=400, seed=3)
    cfg = TripleBarrierConfig(horizon=32)
    out = triple_barrier_labels(md.prices, cfg)
    assert out["exit_h"].max() <= cfg.horizon


# ------------------------------------------------------------ Corwin-Schultz
def test_corwin_schultz_is_nonnegative_and_finite():
    md = load_synthetic(n_symbols=25, n_days=400, seed=8)
    px = md.prices
    highs = px * 1.012
    lows = px * 0.988
    s = corwin_schultz_spread(highs, lows, window=21).dropna(how="all")
    assert (s.fillna(0) >= 0).all().all()
    assert np.isfinite(s.to_numpy()[np.isfinite(s.to_numpy())]).all()


def test_wider_range_implies_wider_estimated_spread():
    """Propiedad de sanidad: mas rango intradia -> mas spread estimado."""
    md = load_synthetic(n_symbols=15, n_days=400, seed=9)
    px = md.prices
    tight = corwin_schultz_spread(px * 1.004, px * 0.996, 21).mean().mean()
    wide = corwin_schultz_spread(px * 1.05, px * 0.95, 21).mean().mean()
    assert wide > tight


# ------------------------------------------------------------ universo y mandato
def test_market_cap_filter_refuses_without_point_in_time_data():
    """Configurar banda de capitalizacion sin datos PIT debe FALLAR, no degradar."""
    from swarmtrader.rank import RankConfig, _prepare

    md = load_synthetic(n_symbols=30, n_days=500, seed=4)
    cfg = RankConfig(max_market_cap=10e9)
    with pytest.raises(ValueError, match="lookahead|point-in-time"):
        _prepare(md, cfg)


def test_price_band_filters_both_ends():
    from swarmtrader.rank import RankConfig, _prepare

    md = load_synthetic(n_symbols=40, n_days=500, seed=4)
    lo = float(md.prices.iloc[-1].quantile(0.25))
    hi = float(md.prices.iloc[-1].quantile(0.75))
    df = _prepare(md, RankConfig(min_price=lo, max_price=hi))
    assert df["px"].min() >= lo - 1e-9
    assert df["px"].max() <= hi + 1e-9


def test_market_cap_band_applies_when_data_is_given():
    from swarmtrader.rank import RankConfig, _prepare

    md = load_synthetic(n_symbols=40, n_days=500, seed=4)
    rng = np.random.default_rng(0)
    caps = pd.DataFrame(rng.uniform(20e6, 50e9, md.prices.shape),
                        index=md.prices.index, columns=md.prices.columns)
    df = _prepare(md, RankConfig(min_market_cap=50e6, max_market_cap=10e9), market_cap=caps)
    assert df["market_cap"].min() >= 50e6
    assert df["market_cap"].max() <= 10e9


def test_mandate_start_is_immutable_across_restarts(tmp_path):
    from swarmtrader.mandate import Mandate, MandateConfig

    p = tmp_path / "m.json"
    m = Mandate(p, MandateConfig())
    m.start(nav=100.0)
    t0 = m.state.started_utc
    again = Mandate(p, MandateConfig())
    again.start(nav=999.0)
    assert again.state.started_utc == t0
    assert again.state.initial_nav == 100.0


def test_deposits_are_not_counted_as_return(tmp_path):
    from swarmtrader.mandate import Mandate, MandateConfig

    m = Mandate(tmp_path / "m.json", MandateConfig())
    m.start(nav=100.0)
    m.mark(nav=200.0, external_flow=100.0)
    assert abs(m.evaluate()["return_flow_adjusted"]) < 0.01


def test_drawdown_halt_fires_and_never_auto_liquidates(tmp_path):
    from swarmtrader.mandate import STATUS_HALTED_DRAWDOWN, Mandate, MandateConfig

    m = Mandate(tmp_path / "m.json", MandateConfig(max_drawdown=0.20))
    m.start(nav=100.0)
    m.mark(nav=130.0)
    m.mark(nav=90.0)
    st = m.evaluate()
    assert st["status"] == STATUS_HALTED_DRAWDOWN
    assert "confirmacion explicita" in st["recommended_action"]
    assert m.cfg.liquidate_on_halt is False


def test_terminal_state_keeps_its_recommended_action(tmp_path):
    from swarmtrader.mandate import Mandate, MandateConfig

    m = Mandate(tmp_path / "m.json", MandateConfig(max_drawdown=0.20))
    m.start(nav=100.0)
    m.mark(nav=130.0)
    m.mark(nav=90.0)
    first = m.evaluate()["recommended_action"]
    assert m.evaluate()["recommended_action"] == first != "none"


def test_target_is_not_reachable_from_decision_layer():
    """La meta no debe poder leerse desde la logica de ranking o sizing."""
    import inspect

    from swarmtrader import portfolio, swarm_rank

    for mod in (swarm_rank, portfolio.sizing):
        src = inspect.getsource(mod)
        assert "mandate" not in src.lower(), (
            f"{mod.__name__} importa el mandato: la meta estaria dirigiendo la decision")
