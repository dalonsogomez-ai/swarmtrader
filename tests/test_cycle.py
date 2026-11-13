"""Tests del ciclo diario.

Las propiedades que importan no son "el ciclo corre", sino: que no opere con
datos viejos, que no duplique ordenes tras un reinicio, que el mandato tenga
prioridad sobre el enjambre, y que no venda y recompre lo mismo el mismo dia.
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from swarmtrader.cycle import CycleConfig, DailyCycle, LedgerEntry, PositionLedger
from swarmtrader.data.labels import TripleBarrierConfig
from swarmtrader.data.loaders import load_synthetic
from swarmtrader.execution.paper import PaperBroker
from swarmtrader.mandate import Mandate, MandateConfig
from swarmtrader.swarm_rank import SwarmRankConfig


def _market(n_days=900, momentum=0.25, seed=5):
    md = load_synthetic(n_symbols=40, n_days=n_days, signal_strength=0.0,
                        momentum_strength=momentum, seed=seed)
    n = len(md.prices)
    idx = pd.bdate_range(end=pd.Timestamp.now("UTC").tz_localize(None).normalize(),
                         periods=n + 5)[-n:]
    md.prices.index = idx
    md.volumes.index = idx
    return md


def _cycle(md, tmp_path, **kw):
    broker = PaperBroker(initial_cash=100.0)
    broker.set_prices({s: float(md.prices[s].iloc[-1]) for s in md.prices.columns})
    man = Mandate(tmp_path / "m.json", MandateConfig())
    man.start(nav=100.0, cash=100.0)
    scfg = SwarmRankConfig(allow_deep=False)
    scfg.spa_boot = 200
    kw.setdefault("as_of", str(md.prices.index[-1].date()))
    cfg = CycleConfig(state_dir=str(tmp_path), max_positions=3, **kw)
    return DailyCycle(md, broker, man, scfg, cfg), broker, man


# ------------------------------------------------------------------ guardas
def test_refuses_to_operate_on_stale_data(tmp_path):
    md = _market()
    md.prices.index = md.prices.index - pd.Timedelta(days=60)
    md.volumes.index = md.prices.index
    cyc, broker, _ = _cycle(md, tmp_path, dry_run=False, as_of=None)
    rep = cyc.run()
    assert rep.status == "aborted"
    assert "obsoletos" in rep.notes[0]
    assert broker.positions() == {}


def test_dry_run_sends_no_orders(tmp_path):
    md = _market()
    cyc, broker, _ = _cycle(md, tmp_path, dry_run=True)
    rep = cyc.run()
    assert rep.status == "completed"
    assert rep.orders == []
    assert broker.positions() == {}
    assert any("DRY RUN" in n for n in rep.notes)


def test_second_real_cycle_same_session_is_skipped(tmp_path):
    md = _market()
    cyc, _, _ = _cycle(md, tmp_path, dry_run=False)
    assert cyc.run().status == "completed"
    cyc2, _, _ = _cycle(md, tmp_path, dry_run=False)
    assert cyc2.run().status == "skipped"


def test_force_allows_repeating_a_session(tmp_path):
    md = _market()
    _cycle(md, tmp_path, dry_run=False)[0].run()
    cyc2, _, _ = _cycle(md, tmp_path, dry_run=False, force=True)
    assert cyc2.run().status == "completed"


def test_replay_mode_measures_staleness_against_as_of(tmp_path):
    """Sin `as_of` no se puede validar el ciclo sobre el pasado."""
    md = _market()
    past = md.prices.index[-40]
    sliced = type("S", (), {"prices": md.prices.loc[:past], "volumes": md.volumes.loc[:past]})()
    cyc, _, _ = _cycle(sliced, tmp_path, dry_run=True, as_of=str(past.date()))
    assert cyc.run().status == "completed"


# ------------------------------------------------------------------ mandato
def test_halted_mandate_blocks_new_entries(tmp_path):
    md = _market()
    broker = PaperBroker(initial_cash=100.0)
    broker.set_prices({s: float(md.prices[s].iloc[-1]) for s in md.prices.columns})
    man = Mandate(tmp_path / "m.json", MandateConfig(max_drawdown=0.05))
    man.start(nav=100.0)
    man.mark(nav=130.0)
    man.mark(nav=80.0)                      # dispara el kill-switch
    scfg = SwarmRankConfig(allow_deep=False)
    scfg.spa_boot = 200
    cfg = CycleConfig(dry_run=True, state_dir=str(tmp_path),
                      as_of=str(md.prices.index[-1].date()))
    rep = DailyCycle(md, broker, man, scfg, cfg).run()
    assert rep.targets == {}
    assert any("no se abren posiciones" in n for n in rep.notes)


def test_mandate_is_marked_every_cycle(tmp_path):
    md = _market()
    cyc, _, man = _cycle(md, tmp_path, dry_run=True)
    before = len(man.state.marks)
    cyc.run()
    assert len(man.state.marks) == before + 1


# ------------------------------------------------------------------ libro
def test_ledger_records_entry_date_and_peak(tmp_path):
    led = PositionLedger(tmp_path / "p.json")
    led.entries["X"] = LedgerEntry("X", "2026-01-05", 10.0, 10.0)
    led.save()
    again = PositionLedger(tmp_path / "p.json")
    assert again.entries["X"].entry_date == "2026-01-05"
    assert again.entries["X"].entry_price == 10.0


def test_ledger_drops_symbols_the_broker_no_longer_holds(tmp_path):
    led = PositionLedger(tmp_path / "p.json")
    led.entries["GONE"] = LedgerEntry("GONE", "2026-01-05", 10.0, 10.0)
    led.sync({}, pd.Series(dtype=float), "2026-02-01")
    assert "GONE" not in led.entries


def test_ledger_peak_only_rises(tmp_path):
    from swarmtrader.execution.base import Position

    led = PositionLedger(tmp_path / "p.json")
    led.entries["X"] = LedgerEntry("X", "2026-01-05", 10.0, 15.0)
    led.sync({"X": Position("X", 1.0, 10.0, 12.0)}, pd.Series({"X": 12.0}), "2026-02-01")
    assert led.entries["X"].peak_price == 15.0


# ------------------------------------------------------------------ salidas
def _forced_exit(tmp_path, ret: float, barriers: TripleBarrierConfig):
    from swarmtrader.execution.base import Position

    md = _market()
    broker = PaperBroker(initial_cash=100.0)
    sym = md.prices.columns[0]
    px = float(md.prices[sym].iloc[-1])
    broker.set_prices({s: float(md.prices[s].iloc[-1]) for s in md.prices.columns})
    broker._positions[sym] = Position(sym, 1.0, px / (1 + ret), px)
    man = Mandate(tmp_path / "m.json", MandateConfig())
    man.start(nav=100.0)
    scfg = SwarmRankConfig(allow_deep=False)
    scfg.spa_boot = 150
    cfg = CycleConfig(dry_run=True, state_dir=str(tmp_path), barriers=barriers,
                      use_analogy_on_exit=False,
                      as_of=str(md.prices.index[-1].date()))
    cyc = DailyCycle(md, broker, man, scfg, cfg)
    return cyc.run(), sym


def test_stop_loss_triggers_exit(tmp_path):
    rep, sym = _forced_exit(tmp_path, -0.20,
                            TripleBarrierConfig(stop_loss=-0.08, trail_stop=None))
    d = [e for e in rep.exits if e["symbol"] == sym][0]
    assert d["decision"] in ("sell", "roll")
    assert "stop" in d["reason"]


def test_profit_target_triggers_exit(tmp_path):
    rep, sym = _forced_exit(tmp_path, 0.30,
                            TripleBarrierConfig(profit_target=0.10, trail_stop=None))
    d = [e for e in rep.exits if e["symbol"] == sym][0]
    assert d["decision"] in ("sell", "roll")
    assert "objetivo" in d["reason"]


def test_position_stopped_out_is_not_rebought_same_cycle(tmp_path):
    """Vender por stop y recomprar el mismo dia desangra la cuenta."""
    rep, sym = _forced_exit(tmp_path, -0.20,
                            TripleBarrierConfig(stop_loss=-0.08, trail_stop=None))
    assert sym not in rep.targets


def test_report_never_says_sell_while_keeping_the_symbol(tmp_path):
    """El informe no puede contradecirse: si dice VENDER, no puede estar en el objetivo."""
    md = _market()
    cyc, _, _ = _cycle(md, tmp_path, dry_run=True)
    rep = cyc.run()
    for e in rep.exits:
        if e["decision"] == "sell":
            assert e["symbol"] not in rep.targets


# ------------------------------------------------------------------ topes
def test_targets_never_exceed_the_position_cap(tmp_path):
    md = _market()
    cyc, _, _ = _cycle(md, tmp_path, dry_run=True, max_position_pct=0.25)
    rep = cyc.run()
    for w in rep.targets.values():
        assert w <= 0.25 + 1e-6, f"tope violado: {w:.3f}"


def test_targets_never_exceed_full_investment(tmp_path):
    md = _market()
    cyc, _, _ = _cycle(md, tmp_path, dry_run=True)
    rep = cyc.run()
    assert sum(rep.targets.values()) <= 1.0 + 1e-6


# ------------------------------------------------------------------ registro
def test_every_cycle_is_journalled(tmp_path):
    md = _market()
    cyc, _, _ = _cycle(md, tmp_path, dry_run=True)
    cyc.run()
    lines = (tmp_path / "cycles.jsonl").read_text().splitlines()
    assert len(lines) >= 1
    rec = json.loads(lines[-1])
    assert {"ts_utc", "session", "status", "targets"} <= set(rec)


# ------------------------------------------------------------------ guardarrailes
def test_binary_catalyst_cap_is_enforced():
    from swarmtrader.execution.guards import RiskLimits, RiskManager, RiskViolation

    rm = RiskManager(RiskLimits(), 1000.0)
    rm.start_of_day(1000.0)
    with pytest.raises(RiskViolation, match="binario"):
        rm.check_order(symbol="BIO", notional=100.0, price=10.0, dollar_volume=5e6,
                       equity=1000.0, gross_after=200.0, binary_catalyst=True)


def test_sector_cap_is_enforced_cumulatively():
    from swarmtrader.execution.guards import RiskLimits, RiskManager, RiskViolation

    rm = RiskManager(RiskLimits(), 1000.0)
    rm.start_of_day(1000.0)
    blocked = False
    for i in range(5):
        try:
            rm.check_order(symbol=f"T{i}", notional=90.0, price=10.0, dollar_volume=5e6,
                           equity=1000.0, gross_after=300.0, sector="tech")
        except RiskViolation:
            blocked = True
            break
    assert blocked, "el tope sectorial acumulado no se aplica"


def test_wash_sale_blocks_repurchase_within_two_months():
    from swarmtrader.execution.guards import RiskLimits, RiskManager, RiskViolation

    rm = RiskManager(RiskLimits(), 1000.0)
    rm.start_of_day(1000.0)
    rm.record_loss_sale("X", "2026-09-01")
    with pytest.raises(RiskViolation, match="33.5 LIRPF"):
        rm.check_order(symbol="X", notional=50.0, price=10.0, dollar_volume=5e6,
                       equity=1000.0, gross_after=100.0, today="2026-09-20")
    rm.check_order(symbol="X", notional=50.0, price=10.0, dollar_volume=5e6,
                   equity=1000.0, gross_after=100.0, today="2026-11-20")


def test_sizing_sector_cap_matches_guard_sector_cap():
    """La capa operativa no debe proponer lo que la de seguridad rechaza."""
    from swarmtrader.execution.guards import RiskLimits
    from swarmtrader.portfolio.sizing import SizingConfig

    assert SizingConfig().max_sector_weight <= RiskLimits().max_sector_pct + 1e-9


def test_guard_position_cap_is_looser_than_sizing_cap():
    """Orden correcto: la guarda es el limite ultimo, el sizing el objetivo operativo."""
    from swarmtrader.execution.guards import RiskLimits
    from swarmtrader.portfolio.sizing import SizingConfig

    assert RiskLimits().max_position_pct >= SizingConfig().max_weight


def _unstarted_cycle(md, tmp_path, dry_run):
    broker = PaperBroker(initial_cash=100.0)
    broker.set_prices({s: float(md.prices[s].iloc[-1]) for s in md.prices.columns})
    man = Mandate(tmp_path / "m.json", MandateConfig())
    scfg = SwarmRankConfig(allow_deep=False)
    scfg.spa_boot = 200
    cfg = CycleConfig(state_dir=str(tmp_path), max_positions=3, dry_run=dry_run,
                      as_of=str(md.prices.index[-1].date()))
    return DailyCycle(md, broker, man, scfg, cfg), man


def test_dry_run_does_not_start_or_mark_an_unstarted_mandate(tmp_path):
    """Una simulacion no arranca el reloj del mandato: lo consulta y sigue."""
    cyc, man = _unstarted_cycle(_market(), tmp_path, dry_run=True)
    rep = cyc.run()
    assert rep.status == "completed"
    assert rep.mandate["status"] == "NOT_STARTED"
    assert man.state.started_utc is None and man.state.marks == []
    assert not (tmp_path / "m.json").exists()
    assert not any("no se abren posiciones" in n for n in rep.notes)


def test_real_cycle_still_requires_a_started_mandate(tmp_path):
    """Con envio de ordenes, un mandato sin iniciar sigue siendo un error."""
    cyc, _ = _unstarted_cycle(_market(), tmp_path, dry_run=False)
    with pytest.raises(RuntimeError, match="no ha sido iniciada"):
        cyc.run()
