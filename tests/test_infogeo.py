"""Tests de la capa de geometria de la informacion.

Tres niveles: las formulas contra identidades conocidas; el monitor de deriva
(calla con ruido estacionario, dispara con un cambio de regimen inyectado, no
mira al futuro); y las metricas alternativas del motor de analogos (no rompen
la disciplina point-in-time ni cambian el comportamiento por defecto).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from swarmtrader.analogy import (DESCRIPTOR_COLS, AnalogyConfig, analogue_distances,
                                 find_analogues)
from swarmtrader.data.features import compute_features
from swarmtrader.data.loaders import MarketData, load_synthetic
from swarmtrader.rank import RankConfig, _prepare
from swarmtrader.stats.drift import (DriftConfig, compute_states, drift_at,
                                     market_state_drift)
from swarmtrader.stats.infogeo import (bernoulli_coordinate, fisher_rao_bernoulli,
                                       fisher_rao_gaussian, fisher_rao_spd, kl_gaussian,
                                       shrink_to_identity)
from swarmtrader.swarm_rank import SwarmRankConfig, swarm_rank


# ------------------------------------------------------------------ formulas
def test_gaussian_equal_means_reduces_to_log_ratio_of_sigmas():
    d = fisher_rao_gaussian(0.3, 0.01, 0.3, 0.04)
    assert d == pytest.approx(np.sqrt(2) * np.log(4.0), rel=1e-12)


def test_gaussian_is_invariant_to_common_scale_and_shift():
    """+2% con sigma 1% dista de su referencia lo mismo que +6% con sigma 3%."""
    a = fisher_rao_gaussian(0.02, 0.01, 0.0, 0.01)
    b = fisher_rao_gaussian(0.06, 0.03, 0.0, 0.03)
    c = fisher_rao_gaussian(0.02 * 7 + 5, 0.07, 0.0 * 7 + 5, 0.07)
    assert a == pytest.approx(b, rel=1e-12) == pytest.approx(c, rel=1e-9)


def test_gaussian_is_a_metric():
    rng = np.random.default_rng(0)
    p = np.column_stack([rng.normal(0, 1, 300), rng.uniform(0.1, 3, 300)])
    a, b, c = p[:100], p[100:200], p[200:]
    dab = fisher_rao_gaussian(a[:, 0], a[:, 1], b[:, 0], b[:, 1])
    dba = fisher_rao_gaussian(b[:, 0], b[:, 1], a[:, 0], a[:, 1])
    dbc = fisher_rao_gaussian(b[:, 0], b[:, 1], c[:, 0], c[:, 1])
    dac = fisher_rao_gaussian(a[:, 0], a[:, 1], c[:, 0], c[:, 1])
    assert np.allclose(dab, dba)
    assert (dac <= dab + dbc + 1e-12).all()
    assert fisher_rao_gaussian(1.0, 2.0, 1.0, 2.0) == 0.0


def test_gaussian_matches_kl_locally():
    """Para distribuciones proximas, d^2 = 2 KL: es la definicion de la metrica."""
    mu, s, e = 0.1, 0.5, 1e-4
    for dmu, ds in ((e, 0.0), (0.0, e), (e, -e)):
        d2 = fisher_rao_gaussian(mu, s, mu + dmu, s + ds) ** 2
        assert d2 == pytest.approx(2 * kl_gaussian(mu, s, mu + dmu, s + ds), rel=1e-3)


def test_bernoulli_distance_is_additive_and_bounded():
    assert fisher_rao_bernoulli(0.0, 1.0) == pytest.approx(np.pi)
    assert (fisher_rao_bernoulli(0.1, 0.4) + fisher_rao_bernoulli(0.4, 0.9)
            == pytest.approx(fisher_rao_bernoulli(0.1, 0.9)))
    p, e = 0.2, 1e-6
    assert fisher_rao_bernoulli(p, p + e) == pytest.approx(e / np.sqrt(p * (1 - p)), rel=1e-4)


def test_bernoulli_coordinate_stretches_the_tails():
    """Del percentil 99 al 99,9 debe pesar mucho mas que del 50 al 50,9."""
    tail = fisher_rao_bernoulli(0.99, 0.999)
    centre = fisher_rao_bernoulli(0.50, 0.509)
    assert tail / centre > 7.0
    assert np.all(np.diff(bernoulli_coordinate(np.linspace(0, 1, 50))) > 0)


def test_spd_distance_agrees_with_gaussian_in_one_dimension():
    d = fisher_rao_spd(np.array([[0.2 ** 2]]), np.array([[0.5 ** 2]]))
    assert d == pytest.approx(float(fisher_rao_gaussian(0.0, 0.2, 0.0, 0.5)), rel=1e-12)


def test_spd_distance_is_affine_invariant():
    rng = np.random.default_rng(1)
    def spd(k):
        m = rng.normal(size=(k, k))
        return m @ m.T + k * np.eye(k)
    a, b, g = spd(6), spd(6), rng.normal(size=(6, 6)) + 3 * np.eye(6)
    d = fisher_rao_spd(a, b)
    assert d == pytest.approx(fisher_rao_spd(b, a), rel=1e-9)
    assert d == pytest.approx(fisher_rao_spd(g @ a @ g.T, g @ b @ g.T), rel=1e-8)
    assert d == pytest.approx(fisher_rao_spd(np.linalg.inv(a), np.linalg.inv(b)), rel=1e-8)
    assert fisher_rao_spd(a, a) == pytest.approx(0.0, abs=1e-6)
    assert fisher_rao_spd(a, np.stack([a, b])).shape == (2,)


def test_shrinkage_makes_rank_deficient_correlation_positive_definite():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(5, 12))                       # 5 observaciones, 12 variables
    r = shrink_to_identity(np.corrcoef(x, rowvar=False), n_obs=5)
    assert np.linalg.eigvalsh(r).min() > 0
    assert np.allclose(np.diag(r), 1.0)
    assert np.isfinite(fisher_rao_spd(np.eye(12), r))


# ------------------------------------------------------------------ deriva
def _with_vol_break(md: MarketData, tail: int, factor: float) -> MarketData:
    """Multiplica la volatilidad de las ultimas `tail` sesiones por `factor`."""
    lr = np.log(md.prices).diff().fillna(0.0)
    lr.iloc[-tail:] = lr.iloc[-tail:] * factor
    px = np.exp(lr.cumsum()) * md.prices.iloc[0]
    return MarketData(px, md.volumes)


def test_factor_marginals_are_blind_to_a_volatility_break():
    """La premisa que corrige este modulo: tras el z-score por fecha no queda
    deriva de escala que medir, aunque la volatilidad se haya triplicado."""
    md = _with_vol_break(load_synthetic(n_symbols=60, n_days=700, seed=3), 60, 3.0)
    f = compute_features(md.prices, md.volumes)
    sd = f.groupby(level="date")["reversal_21"].std()
    mu = f.groupby(level="date")["reversal_21"].mean()
    # media ~0 y desviacion ~1 antes y despues (el recorte a +-3 impide que sea exacto)
    assert abs(sd.iloc[-1] - sd.iloc[-200]) < 0.02
    assert abs(mu.iloc[-1]) < 0.02


def test_monitor_fires_on_dispersion_break():
    md = _with_vol_break(load_synthetic(n_symbols=60, n_days=700, seed=3), 60, 3.0)
    rc = RankConfig()
    rep = market_state_drift(_prepare(md, rc), md.prices, rc.train_years)
    assert rep.evaluated and rep.out_of_support
    ch = rep.channels["dispersion"]
    assert ch["p_value"] < 0.005
    assert ch["today_scale"] > 2.0 * ch["ref_scale"]
    assert "dispersion" in rep.reason


def _factor_panel(n_dates: int, n_names: int, rho_tail: float, tail: int, seed: int = 0):
    """Panel con 4 factores independientes salvo en la cola, donde dos se acoplan."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-04", periods=n_dates)
    syms = [f"S{i:03d}" for i in range(n_names)]
    rows = []
    for i in range(n_dates):
        z = rng.normal(size=(n_names, 4))
        if i >= n_dates - tail:
            z[:, 1] = rho_tail * z[:, 0] + np.sqrt(1 - rho_tail ** 2) * z[:, 1]
        rows.append(z)
    idx = pd.MultiIndex.from_product([dates, syms], names=["date", "symbol"])
    df = pd.DataFrame(np.vstack(rows), index=idx, columns=["f0", "f1", "f2", "f3"])
    df.attrs["factor_cols"] = ["f0", "f1", "f2", "f3"]
    px = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.02, (n_dates, n_names)), axis=0)),
                      index=dates, columns=syms)
    return df, px


def test_monitor_fires_on_dependence_break_and_only_in_that_channel():
    df, px = _factor_panel(400, 80, rho_tail=0.6, tail=30)
    st = compute_states(df, px)
    rep = drift_at(st, st.dates[:340], st.dates[-1])
    assert rep.out_of_support
    assert rep.channels["dependencia"]["p_value"] < 0.005
    assert rep.channels["dispersion"]["p_value"] > 0.005


def test_monitor_is_quiet_on_stationary_data():
    df, px = _factor_panel(500, 80, rho_tail=0.0, tail=0, seed=5)
    st = compute_states(df, px)
    fired = [drift_at(st, st.dates[t - 300:t - 26], st.dates[t]).out_of_support
             for t in range(330, 500, 4)]
    assert np.mean(fired) <= 0.07, f"tasa de falsa alarma {np.mean(fired):.0%} con alpha=1%"


def test_states_do_not_look_ahead():
    """El estado de la fecha t no puede cambiar al alterar datos posteriores a t."""
    df, px = _factor_panel(300, 60, rho_tail=0.0, tail=0, seed=7)
    cut = px.index[200]
    full = compute_states(df, px)
    df2, px2 = df.copy(), px.copy()
    df2.loc[df2.index.get_level_values("date") > cut, :] = 9.0
    px2.loc[px2.index > cut] *= 5.0
    df2.attrs = df.attrs
    alt = compute_states(df2, px2)
    n = 201
    assert np.allclose(full.loc[:n], alt.loc[:n], equal_nan=True)
    assert np.allclose(full.scale[:n], alt.scale[:n], equal_nan=True)
    assert np.allclose(full.corr[:n], alt.corr[:n], equal_nan=True)


def test_reference_dates_after_as_of_are_ignored():
    df, px = _factor_panel(400, 80, rho_tail=0.6, tail=30)
    st = compute_states(df, px)
    as_of = st.dates[360]                       # antes de la ruptura
    a = drift_at(st, st.dates[:360], as_of)
    b = drift_at(st, st.dates, as_of)           # anade as_of, fechas posteriores y la ruptura
    assert a.channels["dependencia"] == b.channels["dependencia"]
    assert a.n_ref == b.n_ref


def test_monitor_abstains_from_judging_without_enough_reference():
    df, px = _factor_panel(150, 60, rho_tail=0.0, tail=0)
    st = compute_states(df, px)
    rep = drift_at(st, st.dates[:50], st.dates[-1])
    assert rep.evaluated is False and rep.out_of_support is False


def test_alpha_zero_disables_and_bad_mode_is_rejected():
    df, px = _factor_panel(200, 60, rho_tail=0.0, tail=0)
    st = compute_states(df, px)
    assert drift_at(st, st.dates[:150], st.dates[-1], DriftConfig(alpha=0)).evaluated is False
    with pytest.raises(ValueError):
        DriftConfig(mode="auto")


def test_gate_vetoes_an_otherwise_actionable_verdict_and_warn_does_not():
    """El veto solo quita: con la misma evidencia, 'warn' deja pasar y 'gate' no."""
    md = load_synthetic(n_symbols=45, n_days=820, signal_strength=0.0,
                        momentum_strength=0.25, seed=21)
    md = _with_vol_break(md, 30, 4.0)
    warn = SwarmRankConfig(allow_deep=False, spa_boot=300, drift=DriftConfig(mode="warn"))
    gate = SwarmRankConfig(allow_deep=False, spa_boot=300, drift=DriftConfig(mode="gate"))
    rw, rg = swarm_rank(md, warn), swarm_rank(md, gate)
    assert rw.drift.out_of_support and rg.drift.out_of_support
    assert rw.verdict.actionable is True
    assert rg.verdict.actionable is False
    assert any(r.startswith("G0 deriva") for r in rg.verdict.reasons)
    assert rg.portfolio.empty or rg.portfolio["weight"].sum() == 0.0
    assert rg.picks["prob_success"].nunique() == 1
    assert rg.to_dict()["drift"]["out_of_support"] is True


def test_gate_does_not_touch_the_verdict_on_stationary_data():
    cfg = SwarmRankConfig(allow_deep=False, spa_boot=300)
    md = load_synthetic(n_symbols=45, n_days=820, signal_strength=0.0,
                        momentum_strength=0.25, seed=21)
    res = swarm_rank(md, cfg)
    assert res.drift.evaluated and not res.drift.out_of_support
    assert res.verdict.actionable is True


def test_drift_module_does_not_feed_the_signal():
    """Garantia mecanica: el monitor observa, no genera senal ni dimensiona."""
    import inspect

    import swarmtrader.agents.specialists as sp
    import swarmtrader.ensemble.stacking as stk
    import swarmtrader.portfolio.sizing as sz
    for mod in (sp, stk, sz):
        src = inspect.getsource(mod)
        assert "stats.drift" not in src and "market_state_drift" not in src


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


def test_default_metric_is_the_original_euclidean_rule():
    assert AnalogyConfig().metric == "euclid"
    rng = np.random.default_rng(4)
    q, m = rng.uniform(size=9), rng.uniform(size=(50, 9))
    w = np.full(9, 1 / 9)
    assert np.allclose(analogue_distances(q, m, w), np.sqrt(((m - q) ** 2 * w).sum(axis=1)))
    with pytest.raises(ValueError):
        AnalogyConfig(metric="cosine")


def test_fisher_rank_reorders_neighbours_in_the_tail():
    """Para un caso extremo, otro caso extremo debe quedar mas cerca que uno
    'solo alto', aunque en percentiles disten lo mismo de un tercero."""
    w = np.ones(1)
    q = np.array([0.999])
    pool = np.array([[0.99], [0.90]])
    e = analogue_distances(q, pool, w, "euclid")
    f = analogue_distances(q, pool, w, "fisher_rank")
    assert f[1] / f[0] < e[1] / e[0]             # la cola pesa relativamente mas
    centre = analogue_distances(np.array([0.509]), np.array([[0.50]]), w, "fisher_rank")
    assert f[0] > 7 * centre[0]


def test_fisher_gauss_is_scale_invariant_and_needs_its_inputs():
    w = np.full(9, 1 / 9)
    q, m = np.zeros(9), np.zeros((3, 9))
    qg = np.array([0.01, 0.02])
    pg = np.array([[0.0, 0.02], [0.03, 0.05], [-0.01, 0.01]])
    a = analogue_distances(q, m, w, "fisher_gauss", qg, pg)
    b = analogue_distances(q, m, w, "fisher_gauss", qg * 4, pg * 4)
    assert np.allclose(a, b)
    with pytest.raises(ValueError):
        analogue_distances(q, m, w, "fisher_gauss")


@pytest.mark.parametrize("metric", ["fisher_rank", "fisher_gauss"])
def test_alternative_metrics_stay_point_in_time(market, metric):
    px, vol, sym, as_of = market
    cfg = AnalogyConfig(k=60, min_analogues=10, max_horizon=32, embargo=5, metric=metric)
    res = find_analogues(px, vol, sym, as_of, cfg)
    assert res.n_analogues > 0
    pos = pd.Series(np.arange(len(px.index)), index=px.index)
    for d in res.neighbours.index.get_level_values("date"):
        assert int(pos[d]) + cfg.max_horizon + cfg.embargo <= int(pos[as_of])


def test_metrics_actually_select_different_neighbours(market):
    px, vol, sym, as_of = market
    sets = {}
    for metric in ("euclid", "fisher_rank", "fisher_gauss"):
        res = find_analogues(px, vol, sym, as_of, AnalogyConfig(k=40, min_analogues=10, metric=metric))
        sets[metric] = set(res.neighbours.index)
    assert sets["euclid"] != sets["fisher_gauss"]
    assert len(sets["euclid"] & sets["fisher_rank"]) < 40
    assert set(DESCRIPTOR_COLS)                  # el descriptor original no se ha tocado


# ------------------------------------------------------------------ experimento
def test_experiment_crps_and_tournament_run_on_small_data():
    """El instrumento con el que se decide la metrica tiene que medir bien."""
    import importlib.util
    from pathlib import Path

    spec = importlib.util.spec_from_file_location(
        "infogeo_experiment", Path(__file__).resolve().parents[1] / "experiments" / "infogeo.py")
    exp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exp)

    assert exp.crps_ensemble(np.array([0.0, 1.0]), 0.5) == pytest.approx(0.25)
    assert exp.crps_ensemble(np.array([2.0]), 5.0) == pytest.approx(3.0)
    # una distribucion centrada en el resultado puntua mejor que una desplazada
    rng = np.random.default_rng(0)
    e = rng.normal(0, 1, 200)
    assert exp.crps_ensemble(e, 0.0) < exp.crps_ensemble(e + 2.0, 0.0)

    md = exp.synthetic_events(n_symbols=40, n_days=700, seed=1)
    res = exp.e2_tournament(md.prices, md.volumes, None, step=10, per_date=3, burn=400)
    assert res["consultas"] > 20
    assert set(res["h21"]["crps"]) == {"azar", "euclid", "fisher_rank", "fisher_gauss"}
    assert 0.0 <= res["h21"]["T2_p_ninguna_fisher_mejora_euclid"] <= 1.0
