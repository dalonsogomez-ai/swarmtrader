"""Integracion del modo por conviccion en el enjambre, el ciclo y la ejecucion.

Propiedades que importan:

* el modo por defecto (`gated`) no cambia en nada;
* en modo `conviction` se asigna capital AUNQUE el veredicto sea NO ACCIONABLE,
  y el informe lo dice sin ambiguedad;
* el ciclo puede concentrar por encima del tope del 25 % y no lo hace en modo
  `gated`; el mandato detenido sigue bloqueando entradas;
* el rebalanceo vende antes de comprar y nunca gasta mas efectivo del que hay.
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from swarmtrader.cycle import CycleConfig, DailyCycle
from swarmtrader.data.loaders import load_synthetic
from swarmtrader.execution.base import OrderResult
from swarmtrader.execution.guards import RiskLimits, RiskManager
from swarmtrader.execution.paper import PaperBroker
from swarmtrader.mandate import Mandate, MandateConfig
from swarmtrader.portfolio.conviction import ConvictionConfig
from swarmtrader.swarm_rank import SwarmRankConfig, format_swarm_report, swarm_rank

FAST = ConvictionConfig(n_scenarios=800, lookback=300, min_lookback=120)


def _cfg(decision="conviction", **kw):
    cfg = SwarmRankConfig(allow_deep=False, decision=decision,
                          conviction=ConvictionConfig(**{**FAST.__dict__, **kw}))
    cfg.spa_boot = 300
    return cfg


@pytest.fixture(scope="module")
def noise_md():
    return load_synthetic(n_symbols=45, n_days=820, signal_strength=0.0, seed=21)


@pytest.fixture(scope="module")
def noise_result(noise_md):
    return swarm_rank(noise_md, _cfg())


# ================================================================ enjambre
def test_default_mode_is_gated_and_computes_no_conviction(noise_md):
    cfg = SwarmRankConfig(allow_deep=False)
    cfg.spa_boot = 300
    assert cfg.decision == "gated"
    res = swarm_rank(noise_md, cfg)
    assert res.conviction is None
    assert res.portfolio.empty or res.portfolio["weight"].sum() == 0.0


def test_conviction_allocates_even_when_not_actionable(noise_result):
    res = noise_result
    assert res.verdict.actionable is False                 # el veredicto no cambia
    assert res.conviction.weights.sum() == pytest.approx(1.0)
    assert res.portfolio.empty or res.portfolio["weight"].sum() == 0.0   # gated intacto


def test_report_says_edge_is_not_demonstrated(noise_result):
    txt = format_swarm_report(noise_result, _cfg())
    assert "ASIGNACION POR CONVICCION" in txt
    assert "ventaja NO demostrada" in txt
    assert "VEREDICTO: NO ACCIONABLE" in txt


def test_result_with_conviction_serialises_to_json(noise_result):
    d = json.loads(json.dumps(noise_result.to_dict(), default=str))
    assert d["decision"] == "conviction"
    assert abs(sum(d["conviction"]["weights"].values()) - 1.0) < 1e-4


def test_conviction_respects_held_and_exclude(noise_md, noise_result):
    first = list(noise_result.conviction.weights.index)
    held = {first[0]: 0.3}
    res = swarm_rank(noise_md, _cfg(), held=held, exclude=first[1:2])
    w = res.conviction.weights
    assert w.sum() == pytest.approx(0.7)
    assert first[0] not in w.index and (len(first) < 2 or first[1] not in w.index)


def test_unknown_decision_mode_is_rejected(noise_md):
    with pytest.raises(ValueError):
        swarm_rank(noise_md, _cfg(decision="yolo"))


# ================================================================ ejecucion
class _RecordingBroker(PaperBroker):
    """Broker simulado que registra el orden de las ordenes."""

    def __post_init__(self):
        super().__post_init__()
        self.sequence: list[tuple[str, str]] = []

    def submit(self, symbol, side, notional):
        self.sequence.append((side, symbol))
        return super().submit(symbol, side, notional)


def _broker_with(holdings: dict[str, float], prices: dict[str, float], cash=0.0):
    b = _RecordingBroker(initial_cash=sum(holdings.values()) + cash,
                         spread_bps=0.0, slippage_bps=0.0)
    b.set_prices(prices)
    for s, notional in holdings.items():
        b.submit(s, "buy", notional)
    b.sequence.clear()
    return b


def test_rebalance_sells_before_buying_and_never_overspends():
    px = {"AAA": 10.0, "BBB": 20.0, "CCC": 5.0}
    b = _broker_with({"AAA": 60.0, "BBB": 40.0}, px)              # 100 % invertido
    risk = RiskManager(RiskLimits(max_position_pct=1.0, max_sector_pct=1.0), b.equity)
    dv = {s: 1e9 for s in px}
    res = b.rebalance_to({"CCC": 0.9, "BBB": 0.1}, risk, dv)       # objetivo: casi todo en CCC
    sides = [s for s, _ in b.sequence]
    assert sides == sorted(sides, key=lambda x: x != "sell")       # todas las ventas primero
    assert b.cash >= -1e-9
    assert b.positions()["CCC"].market_value == pytest.approx(90.0, rel=0.02)
    assert all(isinstance(r, OrderResult) for r in res)


def test_reductions_are_not_blocked_by_entry_guards():
    # BBB ha caido por debajo del precio minimo: se debe poder REDUCIR igualmente.
    px = {"AAA": 10.0, "BBB": 0.5}
    b = _broker_with({"AAA": 50.0, "BBB": 50.0}, px)
    risk = RiskManager(RiskLimits(), b.equity)
    b.rebalance_to({"AAA": 0.8, "BBB": 0.2}, risk, {s: 1e9 for s in px})
    assert b.positions()["BBB"].market_value == pytest.approx(20.0, rel=0.02)


def test_unfundable_buys_are_deferred_not_rejected():
    class _SlowSettle(_RecordingBroker):
        @property
        def buying_power(self):
            return 0.0                      # las ventas aun no han liberado efectivo

    b = _SlowSettle(initial_cash=100.0, spread_bps=0.0, slippage_bps=0.0)
    b.set_prices({"AAA": 10.0})
    risk = RiskManager(RiskLimits(max_position_pct=1.0), b.equity)
    res = b.rebalance_to({"AAA": 1.0}, risk, {"AAA": 1e9})
    assert [r.status for r in res] == ["deferred"]
    assert b.positions() == {}


# ================================================================ ciclo
def _market(n_days=900, seed=5, momentum=0.0):
    md = load_synthetic(n_symbols=40, n_days=n_days, signal_strength=0.0,
                        momentum_strength=momentum, seed=seed)
    n = len(md.prices)
    idx = pd.bdate_range(end=pd.Timestamp.now("UTC").tz_localize(None).normalize(),
                         periods=n + 5)[-n:]
    md.prices.index = idx
    md.volumes.index = idx
    return md


def _cycle(md, tmp_path, decision, mandate_dd=0.30, **conv):
    broker = PaperBroker(initial_cash=100.0)
    broker.set_prices({s: float(md.prices[s].iloc[-1]) for s in md.prices.columns})
    man = Mandate(tmp_path / "m.json", MandateConfig(max_drawdown=mandate_dd))
    man.start(nav=100.0, cash=100.0)
    scfg = _cfg(decision=decision, **conv)
    cfg = CycleConfig(state_dir=str(tmp_path), max_positions=3, dry_run=False,
                      decision=decision, as_of=str(md.prices.index[-1].date()))
    return DailyCycle(md, broker, man, scfg, cfg), broker, man


def test_cycle_in_conviction_mode_buys_even_without_a_verdict(tmp_path):
    md = _market()
    cyc, broker, _ = _cycle(md, tmp_path, "conviction")
    rep = cyc.run()
    assert rep.status == "completed"
    assert sum(rep.targets.values()) == pytest.approx(1.0, abs=1e-3)
    assert broker.positions()                                  # ha comprado
    assert any("conviccion" in n.lower() for n in rep.notes)


def test_cycle_in_conviction_mode_does_not_cap_what_the_optimizer_decides(tmp_path, monkeypatch):
    # El ciclo no debe recortar la concentracion (tope del 25 %, max_positions):
    # se aisla sustituyendo el enjambre por una asignacion fija de 90/10.
    from types import SimpleNamespace

    import swarmtrader.cycle as cycle_mod
    from swarmtrader.portfolio.conviction import ConvictionResult

    md = _market()
    a, b = list(md.prices.columns)[:2]
    table = pd.DataFrame({"symbol": [a, b], "weight": [0.9, 0.1], "score": [1.0, 0.5],
                          "mu": [0.02, 0.01], "sd": [0.1, 0.1], "p_success": [0.3, 0.2]})
    conv = ConvictionResult(pd.Series({a: 0.9, b: 0.1}), 0.0, table, 0.015, 1e-9,
                            {"optimo": 0.015}, {"n_positions": 2, "effective_positions": 1.2})
    fake = SimpleNamespace(
        verdict=SimpleNamespace(actionable=False, reasons=["G4"],
                                to_dict=lambda: {"actionable": False, "combiner": "none"}),
        diagnostics=SimpleNamespace(evidence="NINGUNA"), conviction=conv, horizon=21)
    monkeypatch.setattr(cycle_mod, "swarm_rank", lambda *args, **kw: fake)
    cyc, broker, _ = _cycle(md, tmp_path, "conviction")
    rep = cyc.run()
    assert rep.targets[a] == pytest.approx(0.9)
    assert broker.positions()[a].market_value > 85.0


def test_real_signal_with_risk_neutrality_buys_only_the_top_names(tmp_path):
    md = _market(momentum=0.25)
    cyc, _, _ = _cycle(md, tmp_path, "conviction", risk_aversion=0.0)
    rep = cyc.run()
    assert 1 <= len(rep.targets) <= 5            # solo el tramo superior del mapa


def test_risk_neutral_on_noise_splits_ties_instead_of_guessing(tmp_path):
    # Sin senal el mapa es plano: todas tienen la misma media y no hay "la mejor".
    md = _market()
    cyc, _, _ = _cycle(md, tmp_path, "conviction", risk_aversion=0.0)
    rep = cyc.run()
    w = list(rep.targets.values())
    assert len(w) > 5 and max(w) - min(w) < 1e-3


def test_gated_cycle_on_noise_still_buys_nothing(tmp_path):
    md = _market()
    cyc, broker, _ = _cycle(md, tmp_path, "gated")
    rep = cyc.run()
    assert rep.targets == {} and broker.positions() == {}


def test_halted_drawdown_stop_still_blocks_conviction_entries(tmp_path):
    md = _market()
    cyc, broker, man = _cycle(md, tmp_path, "conviction", mandate_dd=0.05)
    man.mark(nav=130.0)
    man.mark(nav=80.0)                                         # dispara la parada
    rep = cyc.run()
    assert rep.targets == {} and broker.positions() == {}
    assert any("no se abren posiciones" in n for n in rep.notes)
