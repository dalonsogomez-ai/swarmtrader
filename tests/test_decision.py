"""Motor de decision por conviccion, planes de salida, calendario y ejecucion."""

from __future__ import annotations

import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from swarmtrader.data.loaders import load_synthetic  # noqa: E402
from swarmtrader.exits import ExitConfig, ExitPlan, _best, _grid, cost_from_price  # noqa: E402
from swarmtrader.execution.paper import PaperBroker  # noqa: E402
from swarmtrader.execution.plan import (LiveConfirmationRequired,  # noqa: E402
                                        build_order_plan, execute_plan, load_plan)
from swarmtrader.market_calendar import (add_sessions, is_session, next_session,  # noqa: E402
                                         session_open_madrid)
from swarmtrader.portfolio.conviction import solve_simplex  # noqa: E402


# ------------------------------------------------------------ optimizador log
def _kkt(X, b):
    g = (1.0 / (X @ b)) @ X / len(X)          # E[X_i / b.X]
    return g


def test_dominant_asset_takes_100_percent():
    """Si un activo domina, el optimo es el vertice: 100% en el, sin ningun tope."""
    rng = np.random.default_rng(0)
    S = 4000
    X = np.column_stack([1.08 + 0.05 * rng.standard_normal(S),     # dominante
                         1.00 + 0.05 * rng.standard_normal(S),
                         0.99 + 0.08 * rng.standard_normal(S)])
    sol = solve_simplex(np.maximum(X, 1e-3))
    assert sol.weights[0] == pytest.approx(1.0, abs=1e-6)
    g = _kkt(np.maximum(X, 1e-3), sol.weights)
    assert g[0] == pytest.approx(1.0, abs=1e-6) and (g[1:] <= 1.0 + 1e-6).all()


def test_similar_assets_split_by_math_not_by_rule():
    """Dos activos iguales e independientes: el optimo reparte ~50/50 (Kuhn-Tucker)."""
    rng = np.random.default_rng(1)
    S = 20000
    a = 1.02 + 0.10 * rng.standard_normal(S)
    X = np.column_stack([a, rng.permutation(a)])      # mismas marginales, independientes
    sol = solve_simplex(np.maximum(X, 1e-3))
    assert sol.weights == pytest.approx([0.5, 0.5], abs=0.03)
    g = _kkt(np.maximum(X, 1e-3), sol.weights)
    assert np.allclose(g, 1.0, atol=1e-5)


def test_second_asset_enters_only_if_it_raises_growth():
    """Correlacion alta y media peor: el segundo activo NO entra aunque sea bueno."""
    rng = np.random.default_rng(2)
    S = 20000
    z = rng.standard_normal(S)
    X = np.column_stack([1.05 + 0.10 * z, 1.03 + 0.10 * (0.98 * z + 0.2 * rng.standard_normal(S))])
    sol = solve_simplex(np.maximum(X, 1e-3))
    assert sol.weights[0] > 0.999


# ------------------------------------------------------------- plan de salida
def test_renewal_criterion_prefers_short_plan_when_move_is_early():
    """Toda la subida ocurre en 3 sesiones: el plan de 3 sesiones gana por sesion."""
    n, K = 400, 31
    Z = np.zeros((n, K))
    Z[:, :3] = np.array([0.5, 1.0, 1.5])
    Z[:, 3:] = 1.5
    Z += 0.01 * np.random.default_rng(3).standard_normal(Z.shape)
    pol, g = _best(Z, _grid(ExitConfig(), True), sigma=0.02, cost=0.0)
    assert pol[0] == 3
    # con un coste alto, rotar cada 3 sesiones deja de compensar frente a no operar
    pol_c, g_c = _best(Z, _grid(ExitConfig(), True), sigma=0.02, cost=0.05)
    assert g_c < g


def test_cost_tiers():
    c = cost_from_price(pd.Series({"A": 50.0, "B": 3.0, "C": 0.5}))
    assert c.tolist() == [0.0005, 0.01, 0.03]


# ----------------------------------------------------------------- calendario
def test_nyse_calendar():
    assert not is_session(date(2026, 4, 3))           # Viernes Santo
    assert not is_session(date(2026, 6, 19))          # Juneteenth
    assert not is_session(date(2026, 11, 26))         # Accion de Gracias
    assert is_session(date(2026, 10, 12))             # Columbus Day: la bolsa abre
    assert next_session(date(2026, 11, 25)) == date(2026, 11, 27)
    assert next_session(date(2026, 11, 13)) == date(2026, 11, 16)
    assert add_sessions(date(2026, 12, 23), 2) == date(2026, 12, 28)
    assert session_open_madrid(date(2026, 11, 16)).strftime("%H:%M") == "15:30"
    assert session_open_madrid(date(2026, 3, 10)).strftime("%H:%M") == "14:30"   # desfase DST


# ---------------------------------------------------------- plan y ejecucion
def _fake_decision(weights: dict):
    plans = {s: ExitPlan(s, 10, 0.12, -0.08, float("inf"), 112.0, 92.0, 100.0, 0.02,
                         0.001, 0.01, 0.011, 10.0, 0.3, 0.2, 0.0, 0.5, 0.55, 500,
                         "rejilla completa") for s in weights}
    return SimpleNamespace(weights=pd.Series(weights), exit_plans=plans, horizon=10,
                           mode="conviction", as_of="2026-11-12", entry_session="2026-11-13",
                           entry_time_madrid="2026-11-13T15:30:00+01:00",
                           verdict={"actionable": False, "evidence": "NINGUNA"})


def _broker(cash=1000.0, holdings=None):
    b = PaperBroker(initial_cash=cash, spread_bps=0.0, slippage_bps=0.0)
    b.set_prices({"AAA": 10.0, "BBB": 20.0, "OLD": 50.0})
    for s, notional in (holdings or {}).items():
        b.submit(s, "buy", notional)
    return b


def test_all_in_single_stock_uses_whole_balance(tmp_path):
    b = _broker(1000.0)
    plan = build_order_plan(_fake_decision({"AAA": 1.0}), b.positions())
    rep = execute_plan(b, plan, dry_run=False, ledger_path=tmp_path / "pos.json")
    assert [o["side"] for o in rep.orders] == ["buy"]
    invested = b.positions()["AAA"].market_value
    assert invested >= 1000.0 * (1 - plan.cash_buffer) - 0.01
    assert b.cash >= 0.0
    led = json.loads((tmp_path / "pos.json").read_text())
    assert led["AAA"]["horizon"] == 10 and led["AAA"]["stop_pct"] == -0.08


def test_rotation_sells_first_then_buys_with_freed_cash(tmp_path):
    b = _broker(1000.0, holdings={"OLD": 1000.0})
    assert b.cash == pytest.approx(0.0)
    plan = build_order_plan(_fake_decision({"AAA": 0.6, "BBB": 0.4}), b.positions())
    assert plan.sells == ["OLD"]
    rep = execute_plan(b, plan, dry_run=False)
    sides = [o["side"] for o in rep.orders]
    assert sides[0] == "sell" and set(sides[1:]) == {"buy"}
    pos = b.positions()
    assert "OLD" not in pos
    total = pos["AAA"].market_value + pos["BBB"].market_value
    assert total >= 1000.0 * (1 - plan.cash_buffer) - 0.02
    assert pos["AAA"].market_value / total == pytest.approx(0.6, abs=0.005)


def test_dry_run_sends_nothing():
    b = _broker(1000.0, holdings={"OLD": 500.0})
    plan = build_order_plan(_fake_decision({"AAA": 1.0}), b.positions())
    rep = execute_plan(b, plan, dry_run=True)
    assert all(o["status"] == "simulada" for o in rep.orders)
    assert "OLD" in b.positions() and "AAA" not in b.positions()


def test_live_requires_confirming_this_exact_plan():
    b = _broker(1000.0)
    b.paper = False                                # se comporta como cuenta real
    plan = build_order_plan(_fake_decision({"AAA": 1.0}), b.positions())
    with pytest.raises(LiveConfirmationRequired):
        execute_plan(b, plan, dry_run=False)
    with pytest.raises(LiveConfirmationRequired):
        execute_plan(b, plan, dry_run=False, confirm="otro-id")
    rep = execute_plan(b, plan, dry_run=False, confirm=plan.plan_id)
    assert rep.live and b.positions()["AAA"].market_value > 0


def test_tampered_plan_is_rejected(tmp_path):
    plan = build_order_plan(_fake_decision({"AAA": 1.0}), {})
    path = plan.save(tmp_path / "plan.json")
    raw = json.loads(path.read_text())
    raw["targets"] = {"BBB": 1.0}
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError):
        load_plan(path)


def test_overweight_position_is_trimmed_not_left_as_is():
    # Antes de la v1.5 el ejecutor solo compraba o completaba: una posicion que
    # pesaba mas que su objetivo se quedaba asi. Ahora pasa por rebalance_to.
    b = _broker(1000.0, holdings={"AAA": 900.0})
    plan = build_order_plan(_fake_decision({"AAA": 0.3, "BBB": 0.7}), b.positions())
    rep = execute_plan(b, plan, dry_run=False)
    sides = [o["side"] for o in rep.orders]
    assert sides[0] == "sell"                      # reducir antes de comprar
    pos = b.positions()
    total = pos["AAA"].market_value + pos["BBB"].market_value
    assert pos["AAA"].market_value / total == pytest.approx(0.3, abs=0.01)


def test_kept_positions_are_not_touched_and_untradable_are_skipped():
    b = _broker(1000.0, holdings={"OLD": 300.0})
    b.untradable = lambda syms: {"BBB"} & set(syms)
    plan = build_order_plan(_fake_decision({"AAA": 0.35, "BBB": 0.35}), b.positions(),
                            keep={"OLD": 0.3})
    rep = execute_plan(b, plan, dry_run=False)
    assert all(o["symbol"] != "OLD" for o in rep.orders)
    assert "BBB" not in b.positions()
    assert b.positions()["OLD"].market_value == pytest.approx(300.0, rel=1e-6)
    assert any("BBB" in n for n in rep.notes)


def test_liquidity_guard_uses_the_dollar_volume_recorded_in_the_plan():
    b = _broker(1000.0)
    plan = build_order_plan(_fake_decision({"AAA": 1.0}), b.positions(),
                            dollar_volumes={"AAA": 10_000.0})      # < 1 M$ de la guarda
    rep = execute_plan(b, plan, dry_run=False)
    assert [o["status"] for o in rep.orders] == ["rejected"]
    assert "AAA" not in b.positions()


def test_plan_id_changes_with_content():
    a = build_order_plan(_fake_decision({"AAA": 1.0}), {})
    b = build_order_plan(_fake_decision({"AAA": 0.5, "BBB": 0.5}), {})
    assert a.plan_id != b.plan_id and len(a.plan_id) == 12


# --------------------------------------------------- de extremo a extremo
@pytest.fixture(scope="module")
def decisions():
    from swarmtrader.decision import DecisionConfig, decide
    from swarmtrader.portfolio.conviction import ConvictionConfig
    from swarmtrader.rank import RankConfig
    from swarmtrader.swarm_rank import SwarmRankConfig
    cfg = SwarmRankConfig(allow_deep=False, spa_boot=200, rank=RankConfig(oos_months=24))
    dcfg = DecisionConfig(conviction=ConvictionConfig(n_scenarios=1500))
    now = datetime(2020, 1, 1, tzinfo=timezone.utc)
    out = {}
    for name, mom in (("ruido", 0.0), ("senal", 0.25)):
        md = load_synthetic(n_symbols=50, n_days=850, signal_strength=0.0,
                            momentum_strength=mom, seed=11)
        out[name] = decide(md, cfg, dcfg, now_utc=now)
    return out


def test_always_invests_even_without_evidence(decisions):
    d = decisions["ruido"]
    assert not d.verdict["actionable"]
    assert d.weights.sum() == pytest.approx(1.0, abs=1e-6)
    assert d.horizon in DecisionConfigHorizons()
    assert any("NO ACCIONABLE" in n for n in d.notes)


def test_noise_does_not_move_the_mean_signal_does(decisions):
    """Encogimiento de Bayes: en ruido la puntuacion apenas cambia E[r]; con senal, si."""
    def spread(d):
        u = d.swarm.picks.set_index("symbol")["score"]
        t = d.top.set_index("symbol")
        return float(t["mu"].max() - t["mu"].min()), u
    s_noise, _ = spread(decisions["ruido"])
    s_sig, _ = spread(decisions["senal"])
    assert s_sig > 3 * s_noise
    kept_noise = decisions["ruido"].horizons["signal_kept"].mean()
    kept_sig = decisions["senal"].horizons["signal_kept"].mean()
    assert kept_noise < 0.6 < 0.9 < kept_sig


def test_report_top10_scores_in_unit_interval(decisions):
    t = decisions["senal"].top
    assert len(t) == 10
    assert t["valoracion"].between(0, 1).all()
    assert t["valoracion"].is_monotonic_decreasing
    assert set(t["best_horizon"]).issubset(set(DecisionConfigHorizons()))


def test_every_position_has_an_exit_plan(decisions):
    d = decisions["senal"]
    for s in d.weights.index:
        p = d.exit_plans[s]
        assert p.horizon in ExitConfig().verticals
        assert p.p_target + p.p_stop + p.p_trail + p.p_vertical == pytest.approx(1.0)


def DecisionConfigHorizons():
    from swarmtrader.decision import DecisionConfig
    return DecisionConfig().horizons


# ------------------------------------------------------------------- ciclo
def test_cycle_uses_the_positions_own_exit_plan(tmp_path):
    from swarmtrader.cycle import CycleConfig, DailyCycle, LedgerEntry
    from swarmtrader.mandate import Mandate, MandateConfig
    md = load_synthetic(n_symbols=10, n_days=300, seed=1)
    dc = DailyCycle(md, PaperBroker(), Mandate(tmp_path / "m.json", MandateConfig()),
                    cfg=CycleConfig(state_dir=str(tmp_path), use_analogy_on_exit=False))
    s = md.prices.index[-1]
    e = LedgerEntry("X", str((s - pd.Timedelta(days=3)).date()), 100.0, 100.0,
                    horizon=31, target_pct=0.30, stop_pct=None, trail_pct=None)
    assert dc._evaluate_exit("X", e, 115.0, s)["decision"] == "hold"   # global +10% no aplica
    assert dc._evaluate_exit("X", e, 131.0, s)["decision"] == "sell"
    assert dc._evaluate_exit("X", e, 60.0, s)["decision"] == "hold"    # sin stop en el plan


# --------------------------------------------------------------------- CLI
def test_cli_decide_then_execute_end_to_end(tmp_path, capsys):
    """decide (panel largo) -> plan JSON -> execute simulado, por la CLI."""
    from swarmtrader.cli import main
    md = load_synthetic(n_symbols=30, n_days=700, signal_strength=0.0,
                        momentum_strength=0.2, seed=11)
    long = (md.prices.stack().rename("adjclose").to_frame()
            .join(md.volumes.stack().rename("volume")).reset_index())
    long.columns = ["date", "ticker", "adjclose", "volume"]
    long["volume"] = long["volume"] * 1e3            # liquidez por encima de la guarda
    panel = tmp_path / "panel.parquet"
    long.to_parquet(panel)
    plan_path = tmp_path / "plan.json"
    rc = main(["decide", "--panel", str(panel), "--no-deep", "--spa-boot", "150",
               "--oos-months", "18", "--scenarios", "600", "--plan", str(plan_path)])
    assert rc == 0 and plan_path.exists()
    plan = load_plan(plan_path)
    assert sum(plan.targets.values()) == pytest.approx(1.0, abs=1e-6)
    assert set(plan.exit_plans) == set(plan.targets)
    prices = tmp_path / "px.json"
    prices.write_text(json.dumps({s: float(md.prices[s].iloc[-1]) for s in md.prices.columns}))
    capsys.readouterr()
    rc = main(["execute", str(plan_path), "--broker", "paper", "--prices", str(prices),
               "--state-dir", str(tmp_path)])
    out = capsys.readouterr().out
    assert rc == 0 and "SIMULACION" in out and '"simulada"' in out


# ------------------------------------------- correcciones tras la revision (v1.5)
def test_exit_plan_survives_a_buy_that_fills_after_execution(tmp_path):
    """Compra aceptada pero aun sin llenar: el plan queda pendiente y el ciclo lo
    asigna cuando aparece la posicion (antes se perdia y se usaban barreras globales)."""
    from swarmtrader.cycle import PositionLedger
    b = _broker(1000.0)
    real_positions = b.positions
    calls = {"n": 0}

    def lagging_positions():                      # 2a lectura: el broker aun no refleja la compra
        calls["n"] += 1
        return {} if calls["n"] >= 2 else real_positions()
    b.positions = lagging_positions
    plan = build_order_plan(_fake_decision({"AAA": 1.0}), {})
    rep = execute_plan(b, plan, dry_run=False, ledger_path=tmp_path / "pos.json")
    led = PositionLedger(tmp_path / "pos.json")
    assert "AAA" not in led.entries and "AAA" in led.pending
    assert any("pendiente" in n for n in rep.notes)
    # la compra se llena; el ciclo sincroniza con el broker
    b.positions = real_positions
    led.sync(b.positions(), pd.Series({"AAA": 10.0}), "2026-11-13")
    e = led.entries["AAA"]
    assert e.horizon == 10 and e.target_pct == pytest.approx(0.12) and e.stop_pct == pytest.approx(-0.08)
    assert e.entry_price == pytest.approx(10.0) and not led.pending
    led.save()
    assert "AAA" in PositionLedger(tmp_path / "pos.json").entries        # formato compatible


def test_stale_pending_plans_are_dropped():
    from swarmtrader.cycle import LedgerEntry, PositionLedger
    led = PositionLedger("/nonexistent/pos.json")
    led.pending["ZZZ"] = LedgerEntry("ZZZ", "2026-11-01", 0.0, 0.0, horizon=5)
    led.sync({}, pd.Series(dtype=float), "2026-11-13")
    assert not led.pending


def test_halted_mandate_blocks_buys_in_execute(tmp_path):
    """Con el mandato parado por drawdown, `execute` solo vende o reduce."""
    from swarmtrader.mandate import STATUS_HALTED_DRAWDOWN, Mandate, MandateConfig
    b = _broker(1000.0, holdings={"OLD": 500.0})
    m = Mandate(tmp_path / "mandate.json", MandateConfig())
    m.start(nav=2000.0, cash=500.0)                 # pico de 2000; ahora vale ~1000: -50 %
    plan = build_order_plan(_fake_decision({"AAA": 0.6, "OLD": 0.4}), b.positions())
    rep = execute_plan(b, plan, dry_run=False, mandate=m)
    assert Mandate(tmp_path / "mandate.json", MandateConfig()).state.status == STATUS_HALTED_DRAWDOWN
    assert all(o["side"] == "sell" for o in rep.orders)
    assert "AAA" not in b.positions()
    assert any("mandato" in n for n in rep.notes)


def test_crossfit_folds_purge_overlapping_outcome_windows():
    dp = np.arange(0, 21 * 20, 21)                  # entradas cada 21 sesiones
    di = dp.astype(np.int64)
    K = 31                                          # trayectorias de 31 sesiones
    from swarmtrader.exits import _crossfit_folds
    for fit, ev in _crossfit_folds(di, dp, K):
        for f in dp[fit]:
            assert np.all(np.abs(dp[ev] - f) > K)   # ninguna ventana compartida
    # sin purga, la ultima fecha de la primera mitad se solaparia con la siguiente
    (fit0, ev0), _ = _crossfit_folds(di, dp, K)
    assert (dp <= np.quantile(di, 0.5)).sum() > fit0.sum()


def test_path_bank_applies_delisting_return():
    """Con listings point-in-time, la trayectoria que cruza la baja termina en el
    valor terminal en vez de congelarse en el ultimo precio."""
    from swarmtrader.exits import PathBank
    idx = pd.bdate_range("2024-01-01", periods=200)
    rng = np.random.default_rng(3)
    px = pd.DataFrame(10 * np.exp(np.cumsum(rng.normal(0, 0.02, (200, 2)), axis=0)),
                      index=idx, columns=["LIVE", "DEAD"])
    px.loc[idx[120]:, "DEAD"] = np.nan                       # deja de cotizar
    oos = pd.DataFrame({"date": [idx[100]] * 2, "symbol": ["LIVE", "DEAD"],
                        "score": [0.1, 0.2], "fwd": [0.0, 0.0]})
    cfg = ExitConfig()                              # max_horizon = 31
    listings = pd.DataFrame({"reason": ["active", "bankrupt"], "delist_return": [0.0, -0.9]},
                            index=["LIVE", "DEAD"])
    frozen = PathBank(px, oos, cfg)
    pit = PathBank(px, oos, cfg, listings=listings)
    j = list(oos["symbol"]).index("DEAD")
    assert frozen.n == pit.n == 2
    # el terminal (-90 %) aparece como una caida enorme frente al camino congelado
    assert pit.z[j, -1] < frozen.z[j, -1] - 10
    assert np.allclose(pit.z[1 - j], frozen.z[1 - j])        # la viva no cambia


def test_all_cash_decision_is_valid_and_its_plan_sells_holdings(monkeypatch):
    """Con allow_cash, 100 % efectivo es una decision: el plan vende lo que haya."""
    import swarmtrader.decision as dec
    from swarmtrader.decision import DecisionConfig, decide
    from swarmtrader.portfolio.conviction import ConvictionConfig, ConvictionResult
    from swarmtrader.rank import RankConfig
    from swarmtrader.swarm_rank import SwarmRankConfig

    def all_cash(*a, **k):
        return ConvictionResult(pd.Series(dtype=float), 1.0, pd.DataFrame(), 0.0, 0.0,
                                {"optimo": 0.0}, {"risk_aversion": 1.0, "n_map_bins": 0,
                                                  "n_map_dates": 0, "shrinkage": [0.0],
                                                  "log_wealth_se": 0.0}, [])
    monkeypatch.setattr(dec, "allocate", all_cash)
    md = load_synthetic(n_symbols=20, n_days=600, signal_strength=0.0, seed=5)
    scfg = SwarmRankConfig(rank=RankConfig(oos_months=12), allow_deep=False, spa_boot=100)
    cfg = DecisionConfig(conviction=ConvictionConfig(allow_cash=True, n_scenarios=200,
                                                     market_level="history"))
    d = decide(md, scfg, cfg)
    assert d.weights.empty and d.cash == pytest.approx(1.0)
    assert "efectivo" in dec.format_decision_report(d)
    b = _broker(1000.0, holdings={"OLD": 400.0})
    plan = build_order_plan(d, b.positions())
    assert plan.targets == {} and plan.sells == ["OLD"]
    execute_plan(b, plan, dry_run=False)
    assert b.positions() == {}


def test_decide_without_closed_exit_paths_does_not_crash():
    """Predicciones cerradas a 21 sesiones pero ninguna a `max_horizon`: no hay plan
    de salida propio, pero la decision se produce."""
    from swarmtrader.decision import DecisionConfig, decide
    from swarmtrader.portfolio.conviction import ConvictionConfig
    from swarmtrader.rank import RankConfig
    from swarmtrader.swarm_rank import SwarmRankConfig
    md = load_synthetic(n_symbols=20, n_days=600, signal_strength=0.0, seed=6)
    scfg = SwarmRankConfig(rank=RankConfig(oos_months=12), allow_deep=False, spa_boot=100)
    cfg = DecisionConfig(horizons=(5, 10), exits=ExitConfig(verticals=(3, 5000)),
                         conviction=ConvictionConfig(n_scenarios=200, market_level="history"))
    d = decide(md, scfg, cfg)
    assert not d.weights.empty and d.exit_plans == {}
    assert any("planes de salida no estimables" in n for n in d.notes)


# --------------------------- broker local sin fichero de precios (tras la v1.6)
def _saved_plan(tmp_path, weights: dict, ref_prices: dict):
    d = _fake_decision(weights)
    for s, px in ref_prices.items():
        d.exit_plans[s].entry_ref_price = px
    return build_order_plan(d, {}).save(tmp_path / "plan.json")


def _run_paper_execute(plan_path, tmp_path, capsys, *extra):
    from swarmtrader.cli import main
    capsys.readouterr()
    rc = main(["execute", str(plan_path), "--broker", "paper", "--capital", "1000",
               "--state-dir", str(tmp_path), "--execute", *extra])
    out = capsys.readouterr().out
    return rc, json.loads(out[out.index("{"):])


def test_plan_reference_prices_skip_targets_without_a_usable_price():
    d = _fake_decision({"AAA": 0.6, "BBB": 0.4})
    d.exit_plans["AAA"].entry_ref_price = 10.0
    d.exit_plans["BBB"].entry_ref_price = float("nan")
    assert build_order_plan(d, {}).reference_prices() == {"AAA": 10.0}


def test_paper_execute_without_prices_fills_at_the_plan_reference_prices(tmp_path, capsys):
    """Sin `--prices` el broker local no tenia precios y rechazaba todas las compras
    ("sin precio"): usa el cierre de referencia que viaja en el plan y lo dice."""
    path = _saved_plan(tmp_path, {"AAA": 0.6, "BBB": 0.4}, {"AAA": 10.0, "BBB": 20.0})
    rc, rep = _run_paper_execute(path, tmp_path, capsys)
    assert rc == 0
    assert [o["status"] for o in rep["orders"]] == ["filled", "filled"]
    assert sum(o["notional"] for o in rep["orders"]) == pytest.approx(995.0, abs=5.0)
    note = next(n for n in rep["notes"] if "referencia del plan" in n)
    assert "AAA" in note and "BBB" in note


def test_paper_execute_explicit_prices_win_over_the_plan_reference(tmp_path, capsys):
    path = _saved_plan(tmp_path, {"AAA": 0.6, "BBB": 0.4}, {"AAA": 10.0, "BBB": 20.0})
    px = tmp_path / "px.json"
    px.write_text(json.dumps({"AAA": 50.0}))            # BBB no viene: sigue la referencia
    rc, rep = _run_paper_execute(path, tmp_path, capsys, "--prices", str(px))
    by = {o["symbol"]: o for o in rep["orders"]}
    assert rc == 0
    assert by["AAA"]["notional"] / by["AAA"]["qty"] == pytest.approx(50.0, rel=0.01)
    assert by["BBB"]["notional"] / by["BBB"]["qty"] == pytest.approx(20.0, rel=0.01)
    note = next(n for n in rep["notes"] if "referencia del plan" in n)
    assert "BBB" in note and "AAA" not in note


def test_paper_execute_says_how_to_supply_a_missing_price(tmp_path, capsys):
    """Plan sin precio de referencia (decision sin planes de salida): la compra se
    rechaza igual, pero el informe dice como darle el precio."""
    d = _fake_decision({"AAA": 1.0})
    d.exit_plans = {}
    path = build_order_plan(d, {}).save(tmp_path / "plan.json")
    rc, rep = _run_paper_execute(path, tmp_path, capsys)
    assert rc == 3 and rep["orders"][0]["status"] == "rejected"
    assert any("--prices" in n and "AAA" in n for n in rep["notes"])
