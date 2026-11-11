"""Tests que protegen los invariantes que hacen valido el sistema."""
from __future__ import annotations

import numpy as np
from scipy import stats
import pytest

from swarmtrader.agents.genome import BOUNDS, Genome
from swarmtrader.backtest import FeaturePanel, simulate
from swarmtrader.data.features import compute_features, dollar_volume
from swarmtrader.data.loaders import load_synthetic
from swarmtrader.evolution.fitness import posterior_edge_probability
from swarmtrader.execution.guards import RiskLimits, RiskManager, RiskViolation
from swarmtrader.execution.paper import PaperBroker
from swarmtrader.stats.deflated_sharpe import deflated_sharpe_ratio, expected_max_sharpe


@pytest.fixture(scope="module")
def panel():
    md = load_synthetic(n_symbols=40, n_days=600, signal_strength=0.05, seed=1)
    return FeaturePanel(compute_features(md.prices, md.volumes), md.prices,
                        dollar_volume(md.prices, md.volumes))


def test_expected_max_sharpe_grows_with_trials():
    """Mas pruebas => umbral mas alto. Si esto falla, el DSR no protege de nada."""
    assert expected_max_sharpe(10, 0.01) < expected_max_sharpe(1000, 0.01)


def test_dsr_rejects_noise_champion():
    rng = np.random.default_rng(0)
    series = [rng.normal(0, 0.01, 252) for _ in range(300)]
    sr = [s.mean() / s.std(ddof=1) for s in series]
    best = series[int(np.argmax(sr))]
    assert deflated_sharpe_ratio(best, 300, float(np.var(sr, ddof=1))) < 0.95


def test_posterior_resists_short_lucky_streak():
    """Una racha corta con Sharpe brutal no debe convencer al prior."""
    rng = np.random.default_rng(4)
    lucky = rng.normal(0.005, 0.01, 8)
    assert posterior_edge_probability(lucky) < 0.85


def test_genome_has_no_risk_parameters():
    """El genoma NO puede contener limites de riesgo: la evolucion los rompería."""
    forbidden = {"leverage", "max_gross_exposure", "allow_short", "allow_margin",
                 "max_daily_loss_pct", "max_total_drawdown_pct"}
    assert not (set(Genome().to_dict()) & forbidden)


def test_mutation_respects_bounds():
    rng = np.random.default_rng(2)
    g = Genome.random(rng)
    for _ in range(200):
        g = g.mutate(rng, rate=1.0, scale=1.0)
        for key, (lo, hi) in BOUNDS.items():
            assert lo - 1e-6 <= getattr(g, key) <= hi + 1e-6, key


def test_abstention_holds_cash(panel):
    """Umbral de conviccion inalcanzable => 100% liquidez, retorno exactamente 0."""
    g = Genome.random(np.random.default_rng(0))
    g.min_conviction = 99.0
    res = simulate(panel, g, 300, 600)
    assert res.pct_days_in_cash == 1.0
    assert abs(res.total_return) < 1e-12


def test_costs_reduce_return(panel):
    g = Genome.random(np.random.default_rng(6))
    g.min_conviction, g.rebalance_days = 0.0, 5
    cheap = simulate(panel, g, 300, 600, cost_bps=0.0)
    dear = simulate(panel, g, 300, 600, cost_bps=100.0)
    assert dear.total_return < cheap.total_return


def test_no_lookahead_shift(panel):
    """Con execution_lag=0 el resultado debe diferir: prueba de que el lag actua."""
    g = Genome.random(np.random.default_rng(8))
    g.min_conviction = 0.0
    a = simulate(panel, g, 300, 600, execution_lag=1)
    b = simulate(panel, g, 300, 600, execution_lag=0)
    assert not np.allclose(a.returns, b.returns)


def test_risk_manager_blocks_oversized_position():
    rm = RiskManager(RiskLimits(max_position_pct=0.35), 50.0)
    with pytest.raises(RiskViolation):
        rm.check_order(symbol="X", notional=30.0, price=10.0, dollar_volume=5e6,
                       equity=50.0, gross_after=30.0)


def test_risk_manager_blocks_leverage():
    rm = RiskManager(RiskLimits(), 50.0)
    with pytest.raises(RiskViolation):
        rm.check_order(symbol="X", notional=10.0, price=10.0, dollar_volume=5e6,
                       equity=50.0, gross_after=80.0)


def test_risk_manager_halts_on_drawdown():
    rm = RiskManager(RiskLimits(max_total_drawdown_pct=0.30), 50.0)
    rm.update_equity(30.0)
    assert rm.state.halted


def test_paper_broker_cannot_spend_more_than_cash():
    b = PaperBroker(initial_cash=50.0)
    b.set_prices({"AAA": 10.0})
    b.submit("AAA", "buy", 500.0)
    assert b.cash >= -1e-9
    assert b.equity <= 50.5


def test_paper_broker_non_fractional_rejects_tiny_order():
    b = PaperBroker(initial_cash=50.0, fractional=False)
    b.set_prices({"BBB": 400.0})
    assert b.submit("BBB", "buy", 10.0).status == "rejected"


# ---------------------------------------------------------------- mejoras v0.2
from swarmtrader.stats.overfitting import (effective_n_trials_clustering,
                                           probability_of_backtest_overfitting)
from swarmtrader.evolution.meta import META_FEATURE_NAMES, MetaLabeler, PickRecord, triple_barrier_outcome
from swarmtrader.data.features import compute_regime
from swarmtrader.data.baselines import run_baselines
from swarmtrader.tax import apply_spanish_tax, tax_on_gain
from swarmtrader.evolution.engine import SwarmConfig, SwarmEngine
from swarmtrader.stats.registry import TrialRegistry
from swarmtrader.live import compute_targets
from swarmtrader.config import load_config


def test_pbo_high_on_noise_low_with_edge():
    rng = np.random.default_rng(0)
    M = rng.normal(0, 0.01, (500, 60))
    assert probability_of_backtest_overfitting(M).pbo > 0.35
    M2 = M.copy(); M2[:, :3] += 0.0008
    assert probability_of_backtest_overfitting(M2).pbo < 0.35


def test_effective_n_clustering_collapses_clones():
    rng = np.random.default_rng(1)
    base = rng.normal(0, 0.01, (400, 1))
    clones = base + rng.normal(0, 0.002, (400, 30))
    assert effective_n_trials_clustering(clones) <= 2
    indep = rng.normal(0, 0.01, (400, 30))
    assert effective_n_trials_clustering(indep) >= 25


def test_meta_labeler_recovers_signal():
    rng = np.random.default_rng(2)
    recs = []
    for _ in range(300):
        f = rng.normal(0, 1, len(META_FEATURE_NAMES))
        p = 1 / (1 + np.exp(-(2.0 * f[1])))
        recs.append(PickRecord(0, 0, f, int(rng.random() < p)))
    m = MetaLabeler().fit(recs)
    assert m.fitted
    assert m.w[1] > 0.5 and abs(m.w[1]) > max(abs(m.w[k]) for k in range(len(m.w)) if k != 1)


def test_meta_labeler_refuses_to_fit_on_few_samples():
    recs = [PickRecord(0, 0, np.zeros(len(META_FEATURE_NAMES)), i % 2) for i in range(10)]
    assert not MetaLabeler().fit(recs).fitted


def test_triple_barrier():
    assert triple_barrier_outcome(np.array([0.05, 0.06]), 0.10, 0.05) == 1
    assert triple_barrier_outcome(np.array([-0.03, -0.03, 0.5]), 0.10, 0.05) == 0
    assert triple_barrier_outcome(np.array([0.01, -0.005]), 0.10, 0.05) == 1


def test_regime_reduces_exposure(panel):
    """Con corte de regimen imposible y exposicion 0 en alta vol, ambos deben coincidir;
    con corte 0 (siempre alta vol) y exposicion 0, el agente nunca invierte."""
    g = Genome.random(np.random.default_rng(5)); g.min_conviction = -9.0
    g.vol_regime_cut, g.regime_exposure = -99.0, 0.0
    res = simulate(panel, g, 300, 600)
    assert abs(res.total_return) < 1e-9


def test_tax_brackets_progressive():
    assert tax_on_gain(0) == 0
    assert abs(tax_on_gain(6000) - 1140) < 1e-6
    assert tax_on_gain(10000) > tax_on_gain(6000) + 0.19 * 4000  # el segundo tramo es mas alto


def test_tax_drag_positive_when_gains(panel):
    g = Genome.random(np.random.default_rng(6)); g.min_conviction = 0.0; g.rebalance_days = 21
    res = simulate(panel, g, 300, 600, capital=10000.0)
    rep = apply_spanish_tax(res, 10000.0)
    if res.total_return > 0.02:
        assert rep.tax_drag > 0
    assert rep.post_tax_return <= rep.pre_tax_return + 1e-12


def test_realized_pnl_reconciles_with_equity(panel):
    """Sum(P&L realizado) debe aproximar retorno + costes (difieren solo por composicion)."""
    g = Genome.random(np.random.default_rng(7)); g.min_conviction = 0.0
    res = simulate(panel, g, 300, 600)
    realized = sum(e.pnl_frac for e in res.realized)
    # el P&L realizado se mide sobre el capital inicial; el equity compone: la
    # diferencia crece con el retorno total. Tolerancia relativa.
    assert abs(realized - (res.total_return + res.cost_drag)) < 0.05 + 0.3 * abs(res.total_return)


def test_windows_never_overlap_and_respect_holdout():
    md = load_synthetic(n_symbols=30, n_days=1300, signal_strength=0.05, seed=3)
    panel = FeaturePanel(compute_features(md.prices, md.volumes), md.prices,
                         dollar_volume(md.prices, md.volumes), compute_regime(md.prices))
    cfg = SwarmConfig(population_size=6, generations=20, holdout_days=252, embargo_days=5)
    eng = SwarmEngine(panel, cfg, TrialRegistry(":memory:"))
    prev_end = -1
    for g in range(eng.max_generations()):
        s, e = eng._window(g)
        assert s > prev_end                      # no solapado
        assert e <= eng.holdout_start - 5        # nunca toca el holdout
        prev_end = e


def test_engine_runs_and_certify_has_all_gates():
    md = load_synthetic(n_symbols=30, n_days=1100, signal_strength=0.05, seed=4)
    panel = FeaturePanel(compute_features(md.prices, md.volumes), md.prices,
                         dollar_volume(md.prices, md.volumes), compute_regime(md.prices))
    eng = SwarmEngine(panel, SwarmConfig(population_size=8, generations=4, holdout_days=200),
                      TrialRegistry(":memory:"))
    eng.run(verbose=False)
    v = eng.certify(eng.champion())
    assert set(v["gates"]) == {"dsr", "pbo", "beats_baseline", "holdout"}
    assert v["holdout"]["available"]
    assert eng.registry.n_trials(eng.run_id) == 8 * 4


def test_null_market_is_never_certified():
    """El test mas importante del proyecto."""
    md = load_synthetic(n_symbols=40, n_days=1300, signal_strength=0.0, seed=99)
    panel = FeaturePanel(compute_features(md.prices, md.volumes), md.prices,
                         dollar_volume(md.prices, md.volumes), compute_regime(md.prices))
    eng = SwarmEngine(panel, SwarmConfig(population_size=12, generations=6, seed=1),
                      TrialRegistry(":memory:"))
    eng.run(verbose=False)
    assert not eng.certify(eng.champion())["certified"]


def test_baselines_go_through_same_simulator(panel):
    b = run_baselines(panel, 300, 600, cost_bps=10.0)
    assert set(b) == {"BASE_ew_bh", "BASE_mom_top20", "BASE_momq_top20"}
    assert b["BASE_ew_bh"].pct_days_in_cash < 0.05


def test_compute_targets_matches_genome_constraints(panel):
    g = Genome.random(np.random.default_rng(8)); g.min_conviction = -9.0; g.top_k = 4
    t = compute_targets(panel, g)
    assert 0 < len(t) <= 4
    assert all(w <= g.max_weight + 1e-9 for w in t.values())
    assert sum(t.values()) <= 1.0 + 1e-9


def test_config_loads_and_keeps_risk_out_of_swarm(tmp_path):
    toml = tmp_path / "c.toml"
    toml.write_text('[swarm]\npopulation_size=7\n[risk]\nmax_position_pct=0.2\nallow_margin=false\n')
    cfg, risk = load_config(toml)
    assert cfg.population_size == 7 and risk.max_position_pct == 0.2
    assert not hasattr(cfg, "max_position_pct")


# ---------------------------------------------------------------- modo manual (rank)
import pandas as pd
from swarmtrader.stats.calibration import IsotonicCalibrator, expected_calibration_error
from swarmtrader.data.universe import detect_format, load_long_csv, load_wide_csv, read_symbols, load_universe
from swarmtrader.rank import RankConfig, rank_universe, diagnose, walk_forward_oos, _prepare


def test_calibrator_returns_base_rate_without_signal():
    rng = np.random.default_rng(0)
    s = rng.normal(0, 1, 1500); y = (rng.random(1500) < 0.3).astype(float)
    c = IsotonicCalibrator().fit(s, y)
    p = c.predict(np.array([-3.0, 0.0, 3.0]))
    assert np.all(np.abs(p - 0.3) < 0.10)   # isotonica sobre ruido: escalones leves en colas


def test_calibrator_monotone_and_shrunk():
    rng = np.random.default_rng(1)
    s = rng.normal(0, 1, 2000); y = (rng.random(2000) < 1 / (1 + np.exp(-2 * s))).astype(float)
    c = IsotonicCalibrator().fit(s, y)
    p = c.predict(np.linspace(-3, 3, 50))
    assert np.all(np.diff(p) >= -1e-9)
    assert p.max() < 0.995 and p.min() > 0.005          # el shrinkage evita 0% / 100%
    assert expected_calibration_error(c.predict(s), y) < 0.05


def test_read_symbols_variants(tmp_path):
    f1 = tmp_path / "a.csv"; f1.write_text("symbol\naapl\nMSFT\n\nnvda\n")
    f2 = tmp_path / "b.txt"; f2.write_text("AAPL, MSFT\nNVDA\n")
    f3 = tmp_path / "c.csv"; f3.write_text("Ticker,Name\nKO,Coca Cola\nPEP,Pepsi\n")
    assert read_symbols(f1) == ["AAPL", "MSFT", "NVDA"]
    assert read_symbols(f2) == ["AAPL", "MSFT", "NVDA"]
    assert read_symbols(f3) == ["KO", "PEP"]


def _synthetic_long(tmp_path, momentum=0.08, n=20, days=800):
    md = load_synthetic(n_symbols=n, n_days=days, momentum_strength=momentum, seed=5)
    md.prices.columns = [f"S{i:02d}" for i in range(n)]; md.volumes.columns = md.prices.columns
    long = md.prices.stack().rename("close").to_frame().join(md.volumes.stack().rename("volume")).reset_index()
    long.columns = ["date", "symbol", "close", "volume"]
    p = tmp_path / "long.csv"; long.to_csv(p, index=False)
    w = tmp_path / "wide.csv"; md.prices.rename_axis("date").to_csv(w)
    return p, w, md


def test_format_detection_and_loading(tmp_path):
    p, w, md = _synthetic_long(tmp_path)
    assert detect_format(p) == "long" and detect_format(w) == "wide"
    a, b = load_long_csv(p), load_wide_csv(w)
    assert a.prices.shape == md.prices.shape == b.prices.shape
    assert np.allclose(a.prices.to_numpy(), md.prices.to_numpy())
    assert load_universe(p).prices.shape == md.prices.shape


def test_rank_detects_signal_and_abstains_without(tmp_path):
    _, _, strong = _synthetic_long(tmp_path, momentum=0.15)
    r = rank_universe(strong, RankConfig(top_k=3))
    assert r.diagnostics.evidence == "FUERTE" and r.diagnostics.ic_tstat > 3
    assert len(r.picks) == 3 and r.picks["prob_success"].iloc[0] >= r.diagnostics.base_rate
    _, _, none = _synthetic_long(tmp_path, momentum=0.0)
    r0 = rank_universe(none, RankConfig(top_k=3))
    assert r0.diagnostics.evidence == "NINGUNA"
    assert np.all(np.abs(r0.picks["prob_success"] - r0.diagnostics.base_rate) < 0.1)


def test_walk_forward_has_no_lookahead(tmp_path):
    """Cada prediccion OOS en la fecha d solo puede usar targets observables antes de d."""
    _, _, md = _synthetic_long(tmp_path, momentum=0.1)
    cfg = RankConfig()
    df = _prepare(md, cfg)
    oos = walk_forward_oos(df, cfg)
    assert not oos.empty
    dates = df.index.get_level_values("date").unique().sort_values()
    for d in oos["date"].unique():
        pos = dates.get_loc(pd.Timestamp(d))
        assert pos >= cfg.horizon + cfg.embargo


def test_rank_rejects_short_history():
    md = load_synthetic(n_symbols=10, n_days=150)
    with pytest.raises(ValueError):
        rank_universe(md, RankConfig())


# ---------------------------------------------------------------- senales externas (v0.4)
from swarmtrader.data.signals import (analyst_features, build_extra, earnings_features,
                                      fundamentals_features, insider_features, lexicon_score,
                                      load_groups, news_features, short_interest_features)
from swarmtrader.data.features import compute_features as _cf
from swarmtrader.rank import _Ensemble, _Ridge


@pytest.fixture(scope="module")
def small_md():
    md = load_synthetic(n_symbols=10, n_days=320, seed=9)
    return md, list(md.prices.columns), md.prices.index


def test_lexicon_polarity_and_negation():
    assert lexicon_score("beats estimates and raises guidance") > 0
    assert lexicon_score("shares plunge after lawsuit") < 0
    assert lexicon_score("not strong results") < 0
    assert lexicon_score("the company reported results") == 0.0


def test_news_pit_lag(small_md, tmp_path):
    """Una noticia fechada en D no puede influir en la feature de D (lag=1)."""
    md, syms, dates = small_md
    D = dates[150]
    f = tmp_path / "n.csv"
    pd.DataFrame({"date": [D], "symbol": [syms[0]], "headline": ["record profit upgrade"]}).to_csv(f, index=False)
    s = news_features(f, dates, syms, available_lag=1)["news_sent_5"].unstack("symbol")[syms[0]]
    assert np.isnan(s.loc[D]) or s.loc[D] == 0.0
    assert s.loc[dates[151]] > 0
    s0 = news_features(f, dates, syms, available_lag=0)["news_sent_5"].unstack("symbol")[syms[0]]
    assert s0.loc[D] > 0


def test_earnings_sue_window_and_reaction(small_md, tmp_path):
    md, syms, dates = small_md
    D = dates[100]
    f = tmp_path / "e.csv"
    pd.DataFrame({"date": [D], "symbol": [syms[1]], "eps_actual": [1.3], "eps_estimate": [1.0]}).to_csv(f, index=False)
    ef = earnings_features(f, dates, syms, md.prices, available_lag=1, drift_window=63)
    sue = ef["sue"].unstack("symbol")[syms[1]]
    assert sue.loc[D] == 0.0 and abs(sue.loc[dates[101]] - 0.3) < 1e-9 and sue.loc[dates[170]] == 0.0
    react = ef["earn_react_3d"].unstack("symbol")[syms[1]]
    assert react.loc[dates[101]] == 0.0          # incluye t+1: no disponible hasta t+2
    assert react.loc[dates[102]] != 0.0


def test_analyst_uses_deltas(small_md, tmp_path):
    md, syms, dates = small_md
    f = tmp_path / "a.csv"
    pd.DataFrame({"date": [dates[50], dates[80]], "symbol": [syms[2]] * 2, "consensus": [3.0, 4.0]}).to_csv(f, index=False)
    af = analyst_features(f, dates, syms, md.prices, available_lag=1)
    d = af["rev_chg_21"].unstack("symbol")[syms[2]]
    assert np.isnan(d.loc[dates[60]])               # aun no hay 21 sesiones de historia: NaN (=> 0 tras z-score)
    assert d.loc[dates[75]] == 0.0                  # sin cambio todavia
    assert d.loc[dates[85]] > 0                     # subida de 3 -> 4 dentro de la ventana


def test_insider_short_fundamentals_shapes(small_md, tmp_path):
    md, syms, dates = small_md
    i = tmp_path / "i.csv"; pd.DataFrame({"date": [dates[10]], "symbol": [syms[0]], "value": [250000]}).to_csv(i, index=False)
    s = tmp_path / "s.csv"; pd.DataFrame({"date": [dates[10]], "symbol": [syms[0]], "short_interest": [0.12]}).to_csv(s, index=False)
    fu = tmp_path / "f.csv"; pd.DataFrame({"date": [dates[10]], "symbol": [syms[0]], "gross_profit": [50.0],
                                            "total_assets": [200.0], "net_income": [10.0], "cfo": [15.0]}).to_csv(fu, index=False)
    x = build_extra(dates, syms, md.prices, insider=i, short=s, fundamentals=fu)
    assert {"insider_net_63", "short_int", "short_int_chg", "gp_a", "accruals"} <= set(x.columns)
    assert x["insider_net_63"].unstack("symbol")[syms[0]].loc[dates[20]] > 0
    assert x["short_int"].unstack("symbol")[syms[0]].loc[dates[20]] < 0   # negado
    assert x["accruals"].unstack("symbol")[syms[0]].loc[dates[20]] > 0    # NI < CFO => accruals negativos => feature positiva


def test_groups_neutralization_and_industry_momentum(small_md, tmp_path):
    md, syms, dates = small_md
    g = tmp_path / "g.csv"
    pd.DataFrame({"symbol": syms, "sector": ["A"] * 5 + ["B"] * 5}).to_csv(g, index=False)
    groups = load_groups(g)
    f = _cf(md.prices, md.volumes, groups=groups, neutralize=True)
    assert "industry_mom_6" in f.columns
    # tras neutralizar, la media por sector y fecha de un factor es ~0 antes del z-score global;
    # comprobamos al menos que los dos sectores tienen la misma media de industry_mom en cada fecha
    im = f["industry_mom_6"].unstack("symbol").iloc[-1]
    assert np.allclose(im[syms[:5]].to_numpy(), im[syms[0]]) and np.allclose(im[syms[5:]].to_numpy(), im[syms[5]])


def test_extra_signals_improve_ic_when_informative(tmp_path):
    """Una senal externa correlacionada con el futuro debe subir el IC OOS y aparecer en factor_ic."""
    md = load_synthetic(n_symbols=30, n_days=800, momentum_strength=0.0, seed=21)
    syms, dates = list(md.prices.columns), md.prices.index
    fwd = md.prices.shift(-21) / md.prices - 1
    rows = [(d, s, float(np.sign(fwd.loc[d, s]) if pd.notna(fwd.loc[d, s]) else 0)) for d in dates[::5] for s in syms]
    f = tmp_path / "news.csv"; pd.DataFrame(rows, columns=["date", "symbol", "score"]).to_csv(f, index=False)
    x = build_extra(dates, syms, md.prices, news=f)
    base = rank_universe(md, RankConfig(top_k=3))
    withx = rank_universe(md, RankConfig(top_k=3), extra=x)
    assert withx.diagnostics.mean_rank_ic > base.diagnostics.mean_rank_ic + 0.05
    assert "news_sent_5" in withx.diagnostics.factor_ic


def test_ensemble_rank_average_bounded():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (300, 5)); y = X[:, 0] + rng.normal(0, 0.5, 300)
    m = _Ensemble([_Ridge(1.0), _Ridge(50.0)]).fit(X, y)
    p = m.predict(X)
    assert -0.5 <= p.min() and p.max() <= 0.5
    assert stats.spearmanr(p, y).statistic > 0.5


def test_genome_weights_align_to_any_factor_list():
    g = Genome(weights={"mom_12_1": 1.0, "news_sent_5": 0.5})
    v = g.weight_vector(["news_sent_5", "mom_12_1", "sue"])
    assert list(v) == [0.5, 1.0, 0.0]
