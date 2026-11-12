"""Tests del asignador por conviccion (portfolio/conviction.py).

Lo que importa no es "devuelve pesos", sino:

* el optimizador alcanza el optimo de crecimiento y lo CERTIFICA (la cota que
  devuelve acota de verdad la distancia al optimo);
* concentra solo cuando los escenarios lo justifican (condicion de Kuhn-Tucker),
  y la aversion al riesgo es el mando que mueve esa frontera;
* la distribucion predictiva no inventa ganadores: con un mapa puntuacion->
  retorno que es ruido, el encogimiento lo aplana; con senal real, lo conserva;
* la deriva historica de cada accion no se cuenta dos veces;
* respeta presupuesto, posiciones mantenidas, exclusiones y minimo de orden.
"""
from __future__ import annotations

import inspect
import json

import numpy as np
import pandas as pd
import pytest

from swarmtrader.portfolio import conviction as cv
from swarmtrader.portfolio.conviction import (
    ConvictionConfig,
    allocate,
    certainty_equivalent,
    solve_simplex,
)


# ------------------------------------------------------------------ utilidades
def _util(W, gamma):
    if abs(gamma - 1.0) < 1e-12:
        return float(np.log(W).mean())
    return float(((W ** (1.0 - gamma) - 1.0) / (1.0 - gamma)).mean())


def _lognormal(S, mus, sigma, rho, seed):
    """Retornos brutos lognormales con correlacion comun rho."""
    rng = np.random.default_rng(seed)
    n = len(mus)
    C = np.full((n, n), rho) + (1.0 - rho) * np.eye(n)
    Z = rng.standard_normal((S, n)) @ np.linalg.cholesky(C).T
    return np.exp(np.asarray(mus) - 0.5 * sigma ** 2 + sigma * Z)


def _prices(n_sym=40, n_days=420, seed=0, drift=None):
    rng = np.random.default_rng(seed)
    r = rng.normal(0.0003, 0.02, (n_days, n_sym))
    if drift is not None:
        for j, d in drift.items():
            r[:, j] += d
    idx = pd.bdate_range("2023-01-02", periods=n_days)
    cols = [f"S{i:02d}" for i in range(n_sym)]
    return pd.DataFrame(100.0 * np.exp(np.cumsum(r, axis=0)), index=idx, columns=cols)


def _oos(symbols, n_dates=18, effect=0.0, level=0.01, noise=0.08, seed=0):
    """Predicciones walk-forward sinteticas: fwd = nivel + effect*(pct-0.5) + ruido."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2023-03-01", periods=n_dates, freq="21B")
    rows = []
    for d in dates:
        sc = rng.standard_normal(len(symbols))
        pct = (pd.Series(sc).rank().to_numpy() - 0.5) / len(symbols)
        fwd = level + effect * (pct - 0.5) + noise * rng.standard_normal(len(symbols))
        rows.append(pd.DataFrame({"date": d, "symbol": symbols, "score": sc, "fwd": fwd}))
    return pd.concat(rows, ignore_index=True)


def _today(symbols, seed=1):
    rng = np.random.default_rng(seed)
    return pd.Series(rng.standard_normal(len(symbols)), index=symbols)


FAST = dict(n_scenarios=800, lookback=300, min_lookback=120)


# ======================================================= 1. optimizador
def test_dominant_asset_takes_everything():
    X = _lognormal(2000, [0.0, 0.0, 0.0], 0.1, 0.3, seed=1)
    X[:, 0] = X[:, 1] * 1.05          # el activo 0 bate al 1 en TODOS los escenarios
    sol = solve_simplex(X)
    assert sol.converged
    assert sol.weights[0] == pytest.approx(1.0, abs=1e-6)


def test_identical_independent_assets_split_evenly():
    A = _lognormal(3000, [0.01, 0.01], 0.15, 0.0, seed=2)
    X = np.vstack([A, A[:, ::-1]])     # simetria exacta entre columnas
    sol = solve_simplex(X)
    assert sol.weights == pytest.approx([0.5, 0.5], abs=1e-4)


def test_binary_bet_against_cash_matches_closed_form():
    # activo: x2 o x0.5 con igual probabilidad; efectivo: 1. Kelly: f* = 3p - 1 = 0.5
    X = np.array([[2.0, 1.0], [0.5, 1.0]])
    sol = solve_simplex(X, tol=1e-10)
    assert sol.weights[0] == pytest.approx(0.5, abs=1e-5)


@pytest.mark.parametrize("gamma", [1.0, 2.0, 0.5])
def test_gap_certifies_distance_to_optimum(gamma):
    mus = np.linspace(-0.01, 0.03, 8)
    X = _lognormal(1500, mus, 0.12, 0.3, seed=3)
    rough = solve_simplex(X, gamma=gamma, max_iter=2)
    best = solve_simplex(X, gamma=gamma, tol=1e-11, max_iter=50000)
    assert rough.gap > 0
    assert best.utility - rough.utility <= rough.gap + 1e-12
    assert best.gap <= 1e-9


def test_objective_never_decreases():
    X = _lognormal(1000, np.linspace(0.0, 0.02, 6), 0.15, 0.2, seed=4)
    U = [solve_simplex(X, max_iter=k).utility for k in range(1, 25)]
    assert all(b >= a - 1e-12 for a, b in zip(U, U[1:]))


def test_kkt_conditions_hold_at_the_solution():
    X = _lognormal(2000, np.linspace(-0.02, 0.04, 10), 0.2, 0.4, seed=5)
    sol = solve_simplex(X, tol=1e-10)
    r = (X / (X @ sol.weights)[:, None]).mean(axis=0)     # E[X_i / b.X]
    # La parada es por la cota de utilidad, no por soporte exacto: un peso que
    # aun decae puede quedar en ~1e-8 (con gbar = 1, su coste es <= gap).
    on = sol.weights > 1e-6
    assert np.allclose(r[on], 1.0, atol=1e-6)
    assert (r[~on] <= 1.0 + 1e-6).all()


def test_corner_iff_expected_ratio_at_most_one():
    base = np.array([1.10, 1.10])
    corner = solve_simplex(np.column_stack([base, [1.28, 0.90]]), tol=1e-10)
    assert corner.weights[0] == pytest.approx(1.0, abs=1e-6)        # E[X1/X0] = 0.991
    inner = solve_simplex(np.column_stack([base, [1.35, 0.90]]), tol=1e-10)
    assert 0.0 < inner.weights[1] < 1.0                            # E[X1/X0] = 1.023


def test_corner_threshold_is_about_sigma2_one_minus_rho():
    sigma, rho = 0.12, 0.3
    thr = sigma ** 2 * (1 - rho)                                   # ~ 1 % por periodo
    below = solve_simplex(_lognormal(40000, [0.6 * thr, 0.0], sigma, rho, seed=6))
    above = solve_simplex(_lognormal(40000, [1.6 * thr, 0.0], sigma, rho, seed=6))
    assert below.weights[1] > 0.05          # por debajo del umbral: reparte
    assert above.weights[0] > 0.999         # por encima: todo a uno


def test_risk_neutral_picks_highest_mean():
    X = _lognormal(2000, [0.00, 0.03, 0.01], [0.1, 0.5, 0.05][0], 0.2, seed=7)
    sol = solve_simplex(X, gamma=0.0)
    assert np.argmax(sol.weights) == int(np.argmax(X.mean(axis=0)))
    assert sol.weights.max() == pytest.approx(1.0)


def test_higher_risk_aversion_diversifies_more():
    X = _lognormal(4000, np.linspace(0.0, 0.03, 8), 0.15, 0.3, seed=8)
    eff = []
    for g in (0.5, 1.0, 2.0, 4.0):
        w = solve_simplex(X, gamma=g, tol=1e-10).weights
        eff.append(1.0 / (w ** 2).sum())
    assert all(b >= a - 1e-6 for a, b in zip(eff, eff[1:]))
    assert eff[-1] > eff[0]


def test_partial_budget_with_fixed_holding_matches_brute_force():
    X = _lognormal(3000, [0.01, 0.02, 0.015], 0.15, 0.3, seed=9)
    base = 0.4 * X[:, 0]                       # se mantiene el 40 % en el activo 0
    F = X[:, 1:]
    sol = solve_simplex(F, budget=0.6, base=base, tol=1e-11)
    assert sol.weights.sum() == pytest.approx(0.6)
    grid = np.linspace(0.0, 0.6, 6001)
    best = max(_util(base + a * F[:, 0] + (0.6 - a) * F[:, 1], 1.0) for a in grid)
    assert sol.utility >= best - 1e-7


def test_asset_with_near_total_loss_never_gets_everything():
    X = _lognormal(1000, [0.0, 0.0], 0.05, 0.0, seed=10)
    X[:, 0] *= 1.20                            # media mucho mayor...
    X[0, 0] = 1e-6                             # ...salvo un escenario de ruina
    sol = solve_simplex(X)
    assert sol.weights[0] < 1.0 - 1e-3


def test_collapsed_weight_can_come_back():
    # El activo 2 parece malo al principio (correlado con el 0, que domina pronto)
    # pero entra en el optimo como cobertura. Con el suelo numerico y sin pasos
    # de Frank-Wolfe se quedaba fuera.
    X = _lognormal(4000, [0.04, 0.0, -0.005], 0.25, 0.0, seed=11)
    sol = solve_simplex(X, tol=1e-8)       # ~1e-10 es el suelo de coma flotante aqui
    r = (X / (X @ sol.weights)[:, None]).mean(axis=0)
    assert (r <= 1.0 + 1e-6).all() and sol.converged


@pytest.mark.parametrize("bad", [
    np.array([[1.0, -0.1]]), np.array([[1.0, np.nan]]), np.zeros((3, 0)),
])
def test_invalid_scenarios_are_rejected(bad):
    with pytest.raises(ValueError):
        solve_simplex(bad)


def test_invalid_parameters_are_rejected():
    X = np.ones((4, 2))
    with pytest.raises(ValueError):
        solve_simplex(X, gamma=-1.0)
    with pytest.raises(ValueError):
        solve_simplex(X, budget=0.0)


def test_certainty_equivalent_is_mean_for_risk_neutral_and_geometric_for_log():
    W = np.array([1.2, 0.9, 1.05])
    assert certainty_equivalent(W, 0.0) == pytest.approx(W.mean() - 1)
    assert certainty_equivalent(W, 1.0) == pytest.approx(np.exp(np.log(W).mean()) - 1)


# ======================================================= 2. escenarios
def _bank(oos, today, px, cfg, horizon=21):
    pct = (today.rank() - 0.5) / len(today)
    return cv._ScenarioBank(oos, pct, px, horizon, cfg, np.random.default_rng(cfg.seed))


def test_scenarios_have_expected_shape_and_are_positive():
    px = _prices()
    syms = list(px.columns)
    bank = _bank(_oos(syms), _today(syms), px, ConvictionConfig(**FAST))
    G = bank.gross(range(5))
    assert G.shape == (800, 5)
    assert (G > 0).all() and np.isfinite(G).all()


def test_historical_drift_is_not_counted_twice():
    # S00 subio con fuerza en el pasado; con un mapa plano su retorno esperado
    # predictivo debe ser el mismo que el de cualquier otra.
    px = _prices(drift={0: 0.004})
    syms = list(px.columns)
    today = pd.Series(0.0, index=syms)
    bank = _bank(_oos(syms, effect=0.0, noise=1e-9), today, px, ConvictionConfig(**FAST))
    mu = bank.gross(range(len(syms))).mean(axis=0) - 1.0
    assert mu[0] == pytest.approx(np.median(mu), abs=1e-6)


def test_shrinkage_never_widens_the_score_map():
    px = _prices()
    syms = list(px.columns)
    for seed in range(5):
        bank = _bank(_oos(syms, seed=seed), _today(syms), px, ConvictionConfig(**FAST))
        assert bank.info["map_spread_shrunk"] <= bank.info["map_spread_raw"] + 1e-12


def test_shrinkage_flattens_noise_and_keeps_real_signal():
    px = _prices(n_sym=60)
    syms = list(px.columns)
    lam_noise = [_bank(_oos(syms, effect=0.0, seed=s), _today(syms), px,
                       ConvictionConfig(**FAST)).info["map_lambda_mean"] for s in range(8)]
    lam_signal = [_bank(_oos(syms, effect=0.20, seed=s), _today(syms), px,
                        ConvictionConfig(**FAST)).info["map_lambda_mean"] for s in range(8)]
    assert np.mean(lam_noise) < 0.4
    assert min(lam_signal) > 0.8


def test_too_few_oos_dates_gives_a_flat_map():
    px = _prices()
    syms = list(px.columns)
    bank = _bank(_oos(syms, n_dates=3, effect=0.5), _today(syms), px, ConvictionConfig(**FAST))
    assert any("plana" in n for n in bank.notes)
    mu = bank.gross(range(len(syms))).mean(axis=0)
    assert np.ptp(mu) < 1e-9


def test_insufficient_history_is_an_error():
    px = _prices(n_days=100)
    syms = list(px.columns)
    with pytest.raises(ValueError):
        _bank(_oos(syms), _today(syms), px, ConvictionConfig(**FAST))


# ======================================================= 3. asignacion
def test_allocation_is_reproducible_with_a_seed():
    px = _prices()
    syms = list(px.columns)
    a = allocate(_oos(syms), _today(syms), px, 21, ConvictionConfig(**FAST))
    b = allocate(_oos(syms), _today(syms), px, 21, ConvictionConfig(**FAST))
    pd.testing.assert_series_equal(a.weights, b.weights)


def test_fully_invested_and_every_weight_reaches_the_minimum_order():
    px = _prices()
    syms = list(px.columns)
    res = allocate(_oos(syms), _today(syms), px, 21, ConvictionConfig(**FAST, min_weight=0.05))
    assert res.weights.sum() == pytest.approx(1.0)
    assert (res.weights >= 0.05 - 1e-9).all()
    assert res.cash == 0.0


def test_real_signal_concentrates_on_the_top_scores():
    px = _prices(n_sym=60)
    syms = list(px.columns)
    today = _today(syms, seed=3)
    res = allocate(_oos(syms, effect=0.30, noise=0.04), today, px, 21,
                   ConvictionConfig(**FAST))
    top_decile = set(today.sort_values().index[-6:])
    assert res.weights[res.weights.index.isin(top_decile)].sum() > 0.8
    assert res.info["map_lambda_mean"] > 0.8


def test_held_positions_shrink_the_budget_and_are_not_bought():
    px = _prices()
    syms = list(px.columns)
    res = allocate(_oos(syms), _today(syms), px, 21, ConvictionConfig(**FAST),
                   held={"S00": 0.4})
    assert res.weights.sum() == pytest.approx(0.6)
    assert "S00" not in res.weights.index


def test_excluded_symbols_get_nothing():
    px = _prices()
    syms = list(px.columns)
    today = _today(syms)
    first = allocate(_oos(syms), today, px, 21, ConvictionConfig(**FAST))
    banned = list(first.weights.index[:2])
    second = allocate(_oos(syms), today, px, 21, ConvictionConfig(**FAST), exclude=banned)
    assert not set(banned) & set(second.weights.index)


def test_cash_is_used_only_when_allowed():
    px = _prices()
    syms = list(px.columns)
    grim = _oos(syms, level=-0.15, noise=0.02)          # todo cae un 15 % por periodo
    forced = allocate(grim, _today(syms), px, 21, ConvictionConfig(**FAST))
    assert forced.cash == 0.0 and forced.weights.sum() == pytest.approx(1.0)
    free = allocate(grim, _today(syms), px, 21, ConvictionConfig(**FAST, allow_cash=True))
    assert free.cash > 0.5


def test_column_generation_reaches_the_full_universe_optimum(monkeypatch):
    monkeypatch.setattr(cv, "_INITIAL_ACTIVE", 3)
    monkeypatch.setattr(cv, "_ADD_PER_ROUND", 2)
    px = _prices(n_sym=30)
    syms = list(px.columns)
    cfg = ConvictionConfig(**FAST, min_weight=0.0)
    bank = _bank(_oos(syms, effect=0.1), _today(syms), px, cfg)
    cand = np.arange(len(bank.symbols))
    _, sol, _, info = cv._solve_universe(bank, cand, np.zeros(800), 1.0, cfg)
    full = solve_simplex(bank.gross(cand), tol=1e-10)
    assert info["rounds"] > 1
    assert sol.utility == pytest.approx(full.utility, abs=1e-7)


def test_alternatives_are_measured_on_the_same_scenarios():
    px = _prices()
    syms = list(px.columns)
    res = allocate(_oos(syms, effect=0.1), _today(syms), px, 21, ConvictionConfig(**FAST))
    alt = res.alternatives
    assert alt["optimo"] == pytest.approx(res.certainty_equivalent)
    assert all(alt["optimo"] >= v - 1e-6 for v in alt.values())


def test_result_serialises_to_json():
    px = _prices()
    syms = list(px.columns)
    res = allocate(_oos(syms), _today(syms), px, 21, ConvictionConfig(**FAST))
    json.dumps(res.to_dict())


def test_module_never_reads_the_evaluation_target():
    """La meta del mandato no debe poder dirigir la asignacion (ver test_analogy)."""
    assert "mandate" not in inspect.getsource(cv).lower()
