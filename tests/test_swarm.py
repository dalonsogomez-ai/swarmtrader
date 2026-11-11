"""Tests de la capa de enjambre.

El foco no es "el codigo corre" sino las tres propiedades que lo hacen fiable:
no hay fuga temporal, el contraste multiple tiene tamano y potencia correctos,
y el sistema se niega a asignar capital cuando no hay evidencia.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from swarmtrader.agents.specialists import DEFAULT_ROSTER, active_roster, run_swarm
from swarmtrader.data.loaders import load_synthetic
from swarmtrader.ensemble.stacking import StackConfig, per_date_ic, stack_agents
from swarmtrader.portfolio.sizing import SizingConfig, size_positions
from swarmtrader.rank import RankConfig, _prepare
from swarmtrader.stats.cpcv import (combinatorial_splits, n_backtest_paths,
                                    purged_kfold_splits)
from swarmtrader.stats.spa import reality_check, spa_test, stationary_bootstrap_indices
from swarmtrader.swarm_rank import SwarmRankConfig, swarm_rank


# ------------------------------------------------------------------ CPCV
def test_purged_kfold_no_overlap_with_label_window():
    h, emb = 21, 5
    for sp in purged_kfold_splits(240, n_splits=6, horizon=h, embargo=emb):
        lo, hi = sp.test.min() - h, sp.test.max() + h + emb
        assert not ((sp.train >= lo) & (sp.train <= hi)).any(), "fuga: train dentro de la purga"


def test_combinatorial_splits_count_and_paths():
    sp = combinatorial_splits(300, n_groups=6, k_test=2, horizon=21, embargo=5)
    assert len(sp) == 15                      # C(6,2)
    assert n_backtest_paths(6, 2) == 5
    for s in sp:
        assert np.intersect1d(s.train, s.test).size == 0


def test_cpcv_keeps_middle_block_between_distant_test_blocks():
    """Dos bloques de test lejanos no deben purgar todo lo que hay en medio."""
    sp = combinatorial_splits(600, n_groups=6, k_test=2, horizon=5, embargo=1)
    widest = max(sp, key=lambda s: s.test.max() - s.test.min())
    assert widest.train.size > 200


def test_cpcv_rejects_bad_params():
    with pytest.raises(ValueError):
        combinatorial_splits(100, n_groups=4, k_test=4)
    with pytest.raises(ValueError):
        purged_kfold_splits(3, n_splits=6)


# ------------------------------------------------------------------ SPA
def test_stationary_bootstrap_shape_and_range():
    rng = np.random.default_rng(0)
    idx = stationary_bootstrap_indices(50, 100, 5.0, rng)
    assert idx.shape == (100, 50)
    assert idx.min() >= 0 and idx.max() < 50


def test_spa_does_not_reject_under_the_null():
    """40 candidatos de puro ruido: el contraste no debe encontrar destreza."""
    rejects = 0
    for s in range(15):
        d = np.random.default_rng(s).normal(0, 0.01, size=(150, 40))
        if spa_test(d, n_boot=400, seed=s).rejects:
            rejects += 1
    assert rejects <= 2, f"tamano roto: {rejects}/15 rechazos bajo H0"


def test_spa_detects_a_real_edge_among_many_candidates():
    r = np.random.default_rng(3)
    d = r.normal(0, 0.01, size=(150, 40))
    d[:, 11] += 0.005
    res = spa_test(d, n_boot=400, seed=1)
    assert res.rejects and res.best_index == 11


def test_block_bootstrap_matters_under_autocorrelation():
    """Con series autocorreladas, el bootstrap iid rechaza de mas."""
    rng = np.random.default_rng(5)
    x = np.zeros((300, 10))
    e = rng.normal(0, 0.01, (300, 10))
    for t in range(1, 300):
        x[t] = 0.6 * x[t - 1] + e[t]
    p_block = spa_test(x, n_boot=600, seed=1).p_value
    p_iid = spa_test(x, n_boot=600, block_length=1, seed=1).p_value
    assert p_block > p_iid


def test_reality_check_is_more_conservative_than_spa():
    r = np.random.default_rng(9)
    d = r.normal(0, 0.01, size=(160, 30))
    d[:, 0] += 0.0022
    d[:, 1:] -= 0.02                      # muchos candidatos malos
    assert reality_check(d, n_boot=500, seed=2).p_value >= spa_test(d, n_boot=500, seed=2).p_value


def test_spa_requires_minimum_sample():
    with pytest.raises(ValueError):
        spa_test(np.zeros((4, 3)))


# ------------------------------------------------------------------ enjambre
@pytest.fixture(scope="module")
def null_panel():
    md = load_synthetic(n_symbols=40, n_days=800, signal_strength=0.0, seed=11)
    cfg = RankConfig()
    df = _prepare(md, cfg)
    return run_swarm(df, cfg), cfg


def test_roster_skips_agents_without_features():
    keep, skipped = active_roster(["mom_12_1", "mom_6_1"], DEFAULT_ROSTER)
    names = {s.name for s in keep}
    assert "momentum" in names
    assert "earnings" in skipped and "requiere senales externas" in skipped["earnings"]


def test_swarm_produces_one_score_column_per_agent(null_panel):
    panel, _ = null_panel
    assert len(panel.agents) >= 4
    for a in panel.agents:
        assert a in panel.scores.columns
    assert {"date", "symbol", "fwd", "success"} <= set(panel.scores.columns)


def test_agent_scores_are_rank_normalised(null_panel):
    panel, _ = null_panel
    g = panel.scores.groupby("date")[panel.agents[0]]
    assert g.min().min() >= -1.0001 and g.max().max() <= 1.0001


# ------------------------------------------------------------------ stacking
def test_stack_weights_use_only_past_dates(null_panel):
    """Propiedad anti-fuga: alterar el futuro no puede cambiar pesos pasados."""
    panel, _ = null_panel
    base = stack_agents(panel.scores, panel.agents, StackConfig())
    tampered = panel.scores.copy()
    last = tampered["date"].max()
    tampered.loc[tampered["date"] == last, "fwd"] *= -3.0
    after = stack_agents(tampered, panel.agents, StackConfig())
    pd.testing.assert_frame_equal(base.weights.iloc[:-1], after.weights.iloc[:-1])


def test_stack_weights_are_nonnegative_and_capped(null_panel):
    panel, _ = null_panel
    res = stack_agents(panel.scores, panel.agents, StackConfig(max_weight=0.5))
    assert (res.weights.to_numpy() >= -1e-9).all()
    assert res.weights.to_numpy().max() <= 0.5 + 1e-6
    assert np.allclose(res.weights.sum(axis=1), 1.0, atol=1e-6)


def test_stack_emits_all_three_competitors(null_panel):
    panel, _ = null_panel
    res = stack_agents(panel.scores, panel.agents, StackConfig())
    assert {"eq", "best_prior", "stack", "agreement"} <= set(res.scores.columns)
    assert res.scores["agreement"].between(0, 1).all()


def test_per_date_ic_is_empty_on_degenerate_input():
    df = pd.DataFrame({"date": ["a", "a"], "score": [1.0, 2.0], "fwd": [0.1, 0.2]})
    assert per_date_ic(df, "score").empty


# ------------------------------------------------------------------ sizing
def test_no_capital_without_evidence():
    p = pd.Series([0.6, 0.55, 0.5], index=["A", "B", "C"])
    vol = pd.Series([0.3, 0.3, 0.3], index=p.index)
    r = size_positions(p, base_rate=0.35, volatilities=vol, evidence_quality=0.0)
    assert r.gross == 0.0 and r.weights.sum() == 0.0


def test_flat_probabilities_give_equal_weights():
    p = pd.Series([0.4, 0.4, 0.4], index=["A", "B", "C"])
    vol = pd.Series([0.25, 0.25, 0.25], index=p.index)
    r = size_positions(p, base_rate=0.4, volatilities=vol, evidence_quality=1.0)
    assert np.allclose(r.weights.to_numpy(), r.weights.iloc[0], rtol=1e-6)


def test_higher_vol_gets_smaller_weight():
    p = pd.Series([0.6, 0.6], index=["LOW", "HIGH"])
    vol = pd.Series([0.20, 0.80], index=p.index)
    r = size_positions(p, base_rate=0.35, volatilities=vol, evidence_quality=1.0)
    assert r.weights["LOW"] > r.weights["HIGH"]


def test_caps_are_respected():
    idx = [f"S{i}" for i in range(6)]
    p = pd.Series(np.linspace(0.75, 0.45, 6), index=idx)
    vol = pd.Series(0.25, index=idx)
    groups = pd.Series(["tech"] * 4 + ["energy"] * 2, index=idx)
    cfg = SizingConfig(max_weight=0.20, max_sector_weight=0.35, max_gross=1.0)
    r = size_positions(p, 0.35, vol, evidence_quality=1.0, groups=groups, cfg=cfg)
    assert r.weights.max() <= 0.20 + 1e-6
    assert r.weights.groupby(groups.reindex(r.weights.index)).sum().max() <= 0.35 + 1e-6
    assert r.gross <= 1.0 + 1e-6


def test_turnover_band_freezes_small_changes():
    idx = ["A", "B"]
    p = pd.Series([0.6, 0.58], index=idx)
    vol = pd.Series(0.25, index=idx)
    prev = size_positions(p, 0.35, vol, evidence_quality=1.0).weights
    again = size_positions(p * 1.001, 0.35, vol, evidence_quality=1.0, previous=prev)
    assert again.turnover < 0.02


# ------------------------------------------------------------------ veredicto
def test_swarm_refuses_to_act_on_noise():
    cfg = SwarmRankConfig(allow_deep=False)      # rapido: la capa profunda se prueba aparte
    cfg.spa_boot = 300
    md = load_synthetic(n_symbols=45, n_days=820, signal_strength=0.0, seed=21)
    res = swarm_rank(md, cfg)
    assert res.verdict.actionable is False
    assert res.portfolio.empty or res.portfolio["weight"].sum() == 0.0
    # sin evidencia, toda probabilidad es la tasa base
    assert res.picks["prob_success"].nunique() == 1


def test_swarm_acts_when_signal_is_real():
    cfg = SwarmRankConfig(allow_deep=False)
    cfg.spa_boot = 300
    md = load_synthetic(n_symbols=45, n_days=820, signal_strength=0.0,
                        momentum_strength=0.25, seed=21)
    res = swarm_rank(md, cfg)
    assert res.verdict.actionable is True
    assert res.verdict.evidence in ("FUERTE", "DEBIL")
    assert res.portfolio["weight"].sum() > 0


def test_result_serialises_to_json_safe_dict():
    import json

    cfg = SwarmRankConfig(allow_deep=False)
    cfg.spa_boot = 200
    md = load_synthetic(n_symbols=40, n_days=800, signal_strength=0.0, seed=5)
    res = swarm_rank(md, cfg)
    json.dumps(res.to_dict(), default=str)


# ------------------------------------------------------------------ capa profunda
# Estos tests entrenan redes y tardan minutos. Se activan con SWARMTRADER_DEEP=1.
import os  # noqa: E402

from swarmtrader.models.deep import TORCH_AVAILABLE  # noqa: E402

deep_only = pytest.mark.skipif(
    not TORCH_AVAILABLE or os.environ.get("SWARMTRADER_DEEP") != "1",
    reason="requiere PyTorch y SWARMTRADER_DEEP=1",
)


@deep_only
def test_deep_ranker_does_not_hallucinate_signal_on_noise():
    """La prueba que decide si el deep learning se puede usar aqui."""
    from scipy import stats as st

    from swarmtrader.models.deep import DeepConfig
    from swarmtrader.models.trainer import DeepPanelBuilder, deep_walk_forward

    cfg = RankConfig(oos_months=8)
    dcfg = DeepConfig(epochs=8, seeds=1, patience=3, seq_len=40, max_dates_per_epoch=80)
    md = load_synthetic(n_symbols=40, n_days=850, signal_strength=0.0, seed=3)
    df = _prepare(md, cfg)
    b = DeepPanelBuilder(md.prices, md.volumes, list(df.attrs["factor_cols"]), seq_len=40)
    oos = deep_walk_forward(df, cfg, b, dcfg)
    ics = oos.groupby("date").apply(
        lambda g: st.spearmanr(g["score"], g["fwd"]).statistic, include_groups=False).dropna()
    t_stat = ics.mean() / ics.std(ddof=1) * np.sqrt(len(ics))
    assert abs(t_stat) < 2.5, f"la red inventa senal sobre ruido: t={t_stat:.2f}"


@deep_only
def test_deep_ranker_learns_injected_momentum():
    from scipy import stats as st

    from swarmtrader.models.deep import DeepConfig
    from swarmtrader.models.trainer import DeepPanelBuilder, deep_walk_forward

    cfg = RankConfig(oos_months=6)
    dcfg = DeepConfig(epochs=8, seeds=1, patience=3, seq_len=40, max_dates_per_epoch=80)
    md = load_synthetic(n_symbols=40, n_days=850, signal_strength=0.0,
                        momentum_strength=0.25, seed=3)
    df = _prepare(md, cfg)
    b = DeepPanelBuilder(md.prices, md.volumes, list(df.attrs["factor_cols"]), seq_len=40)
    oos = deep_walk_forward(df, cfg, b, dcfg)
    ics = oos.groupby("date").apply(
        lambda g: st.spearmanr(g["score"], g["fwd"]).statistic, include_groups=False).dropna()
    assert ics.mean() > 0.10, f"la red no aprende una senal fuerte inyectada: IC={ics.mean():.3f}"


@deep_only
def test_deep_agent_joins_the_swarm():
    cfg = SwarmRankConfig(allow_deep=True)
    cfg.spa_boot = 200
    cfg.rank.oos_months = 6
    from swarmtrader.models.deep import DeepConfig
    cfg.deep = DeepConfig(epochs=6, seeds=1, patience=3, seq_len=40, max_dates_per_epoch=60)
    md = load_synthetic(n_symbols=40, n_days=850, signal_strength=0.0, seed=3)
    res = swarm_rank(md, cfg)
    assert "deep_rank" in set(res.agents["agent"])
    assert res.verdict.actionable is False          # sobre ruido, sigue diciendo que no
