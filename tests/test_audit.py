"""Tests de los cambios que salieron de la auditoria sobre datos reales.

Tres defectos, cada uno con su test:
  * el veredicto validaba el IC de toda la seccion cruzada y no lo que se compra;
  * la puntuacion se multiplicaba por el acuerdo entre agentes sin validarlo;
  * el filtro de precio usaba el precio ajustado, que incorpora splits futuros.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from swarmtrader.data.loaders import MarketData, load_synthetic
from swarmtrader.ensemble.stacking import StackConfig, per_date_ic, stack_agents
from swarmtrader.rank import RankConfig, _prepare
from swarmtrader.swarm_rank import SwarmRankConfig, swarm_rank, topk_excess_by_date


def _load_experiment(name: str):
    path = Path(__file__).resolve().parents[1] / "experiments" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ------------------------------------------------------------------ G4
def _bottom_only_ranking(n_dates=40, n=100, seed=0) -> pd.DataFrame:
    """Ranking que acierta con los peores y no sabe nada de los mejores:
    los 10 ultimos caen, los 80 centrales suben y los 10 primeros rinden como la media."""
    rng = np.random.default_rng(seed)
    rows = []
    for d in pd.bdate_range("2022-01-03", periods=n_dates, freq="21B"):
        score = np.arange(n, dtype=float)
        fwd = np.full(n, 0.03)
        fwd[:10] = -0.20
        fwd[-10:] = (10 * -0.20 + 80 * 0.03) / 90.0       # = media del resto: sin exceso
        rows.append(pd.DataFrame({"date": d, "symbol": [f"S{i}" for i in range(n)],
                                  "score": score, "fwd": fwd + rng.normal(0, 0.01, n)}))
    return pd.concat(rows, ignore_index=True)


def test_rank_ic_can_be_high_while_the_bought_names_have_no_edge():
    """El caso que encontro la auditoria: IC claramente positivo, top-K sin exceso."""
    sc = _bottom_only_ranking()
    ic = per_date_ic(sc, "score")
    t_ic = ic.mean() / (ic.std(ddof=1) / np.sqrt(len(ic)))
    assert ic.mean() > 0.03 and t_ic > 3.0        # IC pequeno pero inequivocamente positivo
    exc = topk_excess_by_date(sc, "score", 5)
    assert len(exc) == sc["date"].nunique()
    assert abs(exc.mean()) < 0.01
    worst = topk_excess_by_date(sc.assign(neg=-sc["score"]), "neg", 5)
    assert worst.mean() < -0.15                  # donde si hay informacion: en los peores


def test_topk_excess_is_measured_against_the_same_date_universe():
    sc = pd.DataFrame({"date": pd.Timestamp("2024-01-02"), "symbol": list("abcd"),
                       "score": [1.0, 2.0, 3.0, 4.0], "fwd": [0.0, 0.0, 0.10, 0.30]})
    assert topk_excess_by_date(sc, "score", 2).iloc[0] == pytest.approx(0.20 - 0.10)


@pytest.fixture(scope="module")
def signal_run():
    md = load_synthetic(n_symbols=45, n_days=820, signal_strength=0.0, momentum_strength=0.25, seed=21)
    return swarm_rank(md, SwarmRankConfig(allow_deep=False, spa_boot=300))


@pytest.fixture(scope="module")
def noise_run():
    md = load_synthetic(n_symbols=45, n_days=820, signal_strength=0.0, momentum_strength=0.0, seed=7)
    return swarm_rank(md, SwarmRankConfig(allow_deep=False, spa_boot=300))


def test_gate_passes_when_the_top_names_really_beat_the_universe(signal_run):
    v = signal_run.verdict
    assert v.p_topk < 0.05 and v.topk_excess > 0
    assert v.actionable is True
    assert signal_run.to_dict()["verdict"]["p_topk"] == round(v.p_topk, 4)


def test_gate_blocks_and_explains_when_they_do_not(noise_run):
    v = noise_run.verdict
    assert v.p_topk >= 0.05
    assert v.actionable is False
    assert any(r.startswith("G4 cartera") for r in v.reasons)
    assert noise_run.portfolio.empty or noise_run.portfolio["weight"].sum() == 0.0


def test_gate_alone_is_enough_to_veto(signal_run, monkeypatch):
    """Con el resto de filtros pasando, un top-K sin exceso debe bastar para NO ACCIONABLE."""
    import swarmtrader.swarm_rank as sr

    monkeypatch.setattr(sr, "topk_excess_by_date",
                        lambda scores, col, k: pd.Series(np.zeros(scores["date"].nunique())))
    md = load_synthetic(n_symbols=45, n_days=820, signal_strength=0.0, momentum_strength=0.25, seed=21)
    res = swarm_rank(md, SwarmRankConfig(allow_deep=False, spa_boot=300))
    assert res.verdict.p_skill < 0.05                      # G1 sigue pasando
    assert res.verdict.actionable is False
    assert [r for r in res.verdict.reasons if r.startswith("G")][0].startswith("G4")


# ------------------------------------------------------------------ desacuerdo
def test_disagreement_shrink_is_off_by_default_but_still_reported(signal_run):
    assert StackConfig().shrink_on_disagreement is False
    assert signal_run.picks["agreement"].between(0, 1).all()


def test_disagreement_shrink_changes_the_stack_only_when_enabled():
    rng = np.random.default_rng(3)
    rows = []
    for d in pd.bdate_range("2022-01-03", periods=30, freq="21B"):
        a = rng.uniform(-1, 1, (40, 3))
        rows.append(pd.DataFrame({"date": d, "symbol": [f"S{i}" for i in range(40)],
                                  "x": a[:, 0], "y": a[:, 1], "z": a[:, 2],
                                  "fwd": rng.normal(0, 0.05, 40), "success": 0.0}))
    panel = pd.concat(rows, ignore_index=True)
    off = stack_agents(panel, ["x", "y", "z"], StackConfig(shrink_on_disagreement=False)).scores
    on = stack_agents(panel, ["x", "y", "z"], StackConfig(shrink_on_disagreement=True)).scores
    assert np.allclose(on["stack"], off["stack"] * off["agreement"])
    assert not np.allclose(on["stack"], off["stack"])
    assert np.allclose(on["eq"], off["eq"])


# ------------------------------------------------------------------ precio nominal
def test_price_filter_uses_the_price_that_actually_traded():
    """Un valor que cotizaba a 0,50 y despues hizo un split inverso 1:20 aparece a
    10 en el historico ajustado. Con el nominal disponible, el filtro lo excluye."""
    md = load_synthetic(n_symbols=30, n_days=500, seed=4)
    sym = md.prices.columns[0]
    nominal = md.prices.copy()
    split_at = md.prices.index[350]
    nominal.loc[nominal.index < split_at, sym] = nominal.loc[nominal.index < split_at, sym] / 200.0
    assert md.prices[sym].min() > 1.0 and nominal.loc[nominal.index < split_at, sym].max() < 1.0

    rc = RankConfig(min_price=1.0)
    naive = _prepare(md, rc)
    pit = _prepare(MarketData(md.prices, md.volumes, nominal), rc)
    d_naive = naive.xs(sym, level="symbol").index
    d_pit = pit.xs(sym, level="symbol").index
    assert (d_naive < split_at).any(), "sin nominal, el valor entra antes del split"
    assert not (d_pit < split_at).any(), "con nominal, no puede entrar mientras cotizaba bajo 1"
    assert (d_pit >= split_at).any()
    others = [s for s in md.prices.columns if s != sym]
    assert len(naive.loc[(slice(None), others), :]) == len(pit.loc[(slice(None), others), :])


def test_marketdata_keeps_nominal_aligned_and_sliced():
    md = load_synthetic(n_symbols=5, n_days=320, seed=1)
    m = MarketData(md.prices, md.volumes, md.prices.iloc[:, ::-1] * 2.0)
    assert list(m.nominal.columns) == list(m.prices.columns)
    s = m.slice(m.prices.index[10], m.prices.index[50])
    assert s.nominal.shape == s.prices.shape
    assert MarketData(md.prices, md.volumes).slice(md.prices.index[0], md.prices.index[5]).nominal is None


# ------------------------------------------------------------------ experimento
def test_experiment_geometric_features_do_not_look_ahead():
    exp = _load_experiment("geometry_audit")
    px = load_synthetic(n_symbols=40, n_days=600, seed=9).prices
    full = exp.geo_features(px)
    cut = px.index[450]
    px2 = px.copy()
    px2.loc[px2.index > cut] *= 3.0
    alt = exp.geo_features(px2)
    a = full[full.index.get_level_values("date") <= cut]
    b = alt[alt.index.get_level_values("date") <= cut]
    assert np.allclose(a.to_numpy(), b.to_numpy(), equal_nan=True)


def test_experiment_holm_adjustment_is_monotone_and_correct():
    exp = _load_experiment("geometry_audit")
    adj = exp.holm({"a": 0.01, "b": 0.04, "c": 0.03, "d": float("nan")})
    assert adj["a"] == pytest.approx(0.03)
    assert adj["c"] == pytest.approx(0.06)
    assert adj["b"] == pytest.approx(0.06)          # nunca menor que el anterior
    assert "d" not in adj


# ------------------------------------------------------------------ adaptador Sharadar
def _sharadar_csv(tmp_path) -> Path:
    dates = pd.bdate_range("2022-01-03", periods=260)
    rng = np.random.default_rng(0)
    rows = []
    ok = 50 * np.exp(np.cumsum(rng.normal(0, 0.01, 260)))
    bust = np.linspace(20.0, 0.20, 180)                       # se hunde y deja de cotizar
    unadj = np.r_[np.full(150, 0.5), np.full(110, 10.0)]      # split inverso 1:20 en la sesion 150
    for i, d in enumerate(dates):
        rows.append(("OKAY", d, ok[i], ok[i], 1e5))
        if i < 180:
            rows.append(("BUST", d, bust[i], bust[i], 1e5))
        rows.append(("RSPL", d, 10.0, unadj[i], 1e5))
    path = tmp_path / "sep.csv"
    pd.DataFrame(rows, columns=["ticker", "date", "closeadj", "closeunadj", "volume"]).to_csv(path, index=False)
    return path


def test_sharadar_adapter_keeps_the_stocks_that_went_to_cents(tmp_path):
    """El adaptador que existe para quitar el sesgo de supervivencia no puede
    descartar a los que quebraron por tener un ultimo precio bajo."""
    from swarmtrader.data.brokers import sharadar_market_data

    md, listings = sharadar_market_data(_sharadar_csv(tmp_path))
    assert "BUST" in md.prices.columns
    assert md.prices["BUST"].dropna().iloc[-1] < 1.0
    assert md.prices["BUST"].iloc[-1] != md.prices["BUST"].iloc[-1]      # NaN tras la baja
    assert str(listings.loc["BUST", "reason"]) != str(listings.loc["OKAY", "reason"])
    with pytest.raises(ValueError, match="fecha a fecha"):
        sharadar_market_data(_sharadar_csv(tmp_path), min_price=1.0)


def test_sharadar_adapter_carries_the_unadjusted_close_for_the_filters(tmp_path):
    from swarmtrader.data.brokers import sharadar_market_data

    md, _ = sharadar_market_data(_sharadar_csv(tmp_path))
    assert md.nominal is not None and md.nominal.shape == md.prices.shape
    assert md.prices["RSPL"].iloc[0] == pytest.approx(10.0)
    assert md.nominal["RSPL"].iloc[0] == pytest.approx(0.5)
    assert md.nominal["RSPL"].iloc[-1] == pytest.approx(10.0)
