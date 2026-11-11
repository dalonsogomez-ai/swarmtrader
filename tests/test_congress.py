"""Tests de la ingesta de declaraciones de cargos publicos (STOCK Act).

El test que importa es `test_transaction_date_creates_fake_alpha`: no comprueba
que el codigo corra, comprueba que la disciplina point-in-time cambia el
resultado en un orden de magnitud. Si algun dia deja de fallar la version con
fuga, es que se ha roto la disciplina.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import stats

from swarmtrader.agents.specialists import DEFAULT_ROSTER, active_roster
from swarmtrader.data.congress import (AMOUNT_RANGES, CONGRESS_FEATURES, CongressConfig,
                                       congress_features, range_midpoint)
from swarmtrader.data.loaders import load_synthetic


@pytest.fixture(scope="module")
def prescient():
    """Un declarante con presciencia PERFECTA y 31 sesiones de retraso legal."""
    md = load_synthetic(n_symbols=60, n_days=900, signal_strength=0.0, seed=13)
    px = md.prices
    sessions = px.index
    fwd = px.shift(-21) / px - 1.0
    rng = np.random.default_rng(7)
    rows = []
    for t in range(300, len(sessions) - 60, 5):
        d = sessions[t]
        disc = sessions[min(t + 31, len(sessions) - 1)]
        for s in fwd.loc[d].nlargest(3).index:
            rows.append(dict(symbol=s, transaction_date=d, disclosure_date=disc,
                             type="Purchase", amount="$15,001 - $50,000",
                             filer=f"M{rng.integers(0, 40)}", role="Member"))
        for s in fwd.loc[d].nsmallest(2).index:
            rows.append(dict(symbol=s, transaction_date=d, disclosure_date=disc,
                             type="Sale", amount="$15,001 - $50,000",
                             filer=f"M{rng.integers(0, 40)}", role="Member"))
    return px, pd.DataFrame(rows), fwd


def _ic(feat: pd.DataFrame, col: str, fwd: pd.DataFrame) -> float:
    fw = fwd.stack(future_stack=True).rename("fwd")
    fw.index.names = ["date", "symbol"]
    j = feat[[col]].join(fw).dropna()
    ics = j.groupby(level="date").apply(
        lambda g: stats.spearmanr(g[col], g["fwd"]).statistic if g[col].nunique() > 2
        else np.nan, include_groups=False).dropna()
    return float(ics.mean())


def test_transaction_date_creates_fake_alpha(prescient):
    """La fecha de operacion fabrica alfa; la de divulgacion no. Medido, no supuesto."""
    px, disc, fwd = prescient
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        leak = congress_features(disc, px, CongressConfig(use_transaction_date=True))
    honest = congress_features(disc, px, CongressConfig(use_transaction_date=False))
    ic_leak = _ic(leak, "pol_net_buy", fwd)
    ic_honest = _ic(honest, "pol_net_buy", fwd)
    assert ic_leak > 0.10, "el escenario de control deberia producir fuga evidente"
    assert ic_leak > 3 * abs(ic_honest), (
        f"la disciplina PIT no esta reduciendo la fuga: {ic_leak:.3f} vs {ic_honest:.3f}")


def test_using_transaction_date_warns_loudly(prescient):
    px, disc, _ = prescient
    with pytest.warns(RuntimeWarning, match="NO estaba disponible"):
        congress_features(disc, px, CongressConfig(use_transaction_date=True))


def test_disclosure_date_is_mandatory(prescient):
    px, disc, _ = prescient
    sin_fecha = disc.drop(columns=["disclosure_date"])
    with pytest.raises(ValueError, match="disclosure_date"):
        congress_features(sin_fecha, px)


def test_no_feature_before_first_disclosure(prescient):
    """Antes de la primera divulgacion no puede haber senal distinta de cero."""
    px, disc, _ = prescient
    feat = congress_features(disc, px)
    first = disc["disclosure_date"].min()
    early = feat.loc[feat.index.get_level_values("date") < first]
    assert (early["pol_net_buy"].abs() < 1e-12).all()
    assert (early["pol_n_filers"] == 0).all()


def test_accepts_raw_dataframe_without_loader(prescient):
    """No debe exigir que se haya pasado antes por load_disclosures."""
    px, disc, _ = prescient
    feat = congress_features(disc, px)
    assert set(CONGRESS_FEATURES) <= set(feat.columns)
    assert len(feat) > 0


def test_range_midpoints_are_sane():
    for label, (lo, hi) in AMOUNT_RANGES.items():
        m = range_midpoint(label)
        assert lo <= m <= hi, label
    assert np.isnan(range_midpoint(None))


def test_net_buy_is_bounded_and_dimensionless(prescient):
    px, disc, _ = prescient
    feat = congress_features(disc, px)
    assert feat["pol_net_buy"].between(-1.0001, 1.0001).all()


def test_political_agent_is_inactive_without_data():
    """Sin factores politicos cargados, el agente no entra en el enjambre."""
    keep, skipped = active_roster(["mom_12_1", "reversal_21"], DEFAULT_ROSTER)
    assert "political" in skipped
    assert "political" not in {s.name for s in keep}


def test_political_agent_activates_with_data():
    cols = ["mom_12_1"] + list(CONGRESS_FEATURES)
    keep, _ = active_roster(cols, DEFAULT_ROSTER)
    assert "political" in {s.name for s in keep}


def test_roster_growth_is_declared():
    """Anadir un agente anade una hipotesis: el roster debe seguir siendo finito
    y conocido de antemano, que es lo que permite deflactarlo."""
    names = [s.name for s in DEFAULT_ROSTER]
    assert len(names) == len(set(names)), "nombres duplicados en el roster"
    assert "political" in names
