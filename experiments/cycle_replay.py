"""Repeticion historica del ciclo diario sobre sesiones consecutivas.

Es una comprobacion de la TUBERIA (ordenes, libro, diario, idempotencia), no
una medicion de rentabilidad: no dice nada sobre si la cartera gana dinero.

Criterios fijados el 2026-11-13, ANTES de ejecutar este script. Son los del
protocolo de prueba previo a conectar Alpaca paper ("repetir el ciclo sobre
unas 20 fechas pasadas: ningun error y un informe que no se contradiga"):

  C1  ningun ciclo termina en un estado distinto de `completed`;
  C2  el efectivo nunca es negativo y ninguna orden sale rechazada;
  C3  repetir el ciclo de una sesion ya ejecutada no envia nada (`skipped`);
  C4  el diario tiene como mucho un ciclo real completado por sesion;
  C5  en ruido, el modo por defecto (`gated`) no envia ninguna orden;
  C6  en ruido, el modo `conviction` invierte aunque no haya veredicto
      accionable (al menos el 99 % del patrimonio tras el primer ciclo).

Datos: SINTETICOS (`load_synthetic`) y broker simulado en memoria. Sin red ni
claves. Cada dia el enjambre solo ve precios hasta esa sesion; el broker y el
directorio de estado persisten de un dia al siguiente.

Uso: python experiments/cycle_replay.py [--sessions 24] [--out docs/results/cycle_replay.json]
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from swarmtrader.cycle import CycleConfig, DailyCycle  # noqa: E402
from swarmtrader.data.loaders import load_synthetic  # noqa: E402
from swarmtrader.execution.paper import PaperBroker  # noqa: E402
from swarmtrader.mandate import Mandate, MandateConfig  # noqa: E402
from swarmtrader.portfolio.conviction import ConvictionConfig  # noqa: E402
from swarmtrader.swarm_rank import SwarmRankConfig  # noqa: E402

N_SYM, N_DAYS, SEED, CAPITAL = 40, 820, 21, 1000.0
SCENARIOS = (("conviction", 0.0), ("gated", 0.0), ("conviction", 0.6))


def replay(decision: str, signal: float, n_sessions: int, state_dir: Path) -> dict:
    full = load_synthetic(n_symbols=N_SYM, n_days=N_DAYS + n_sessions,
                          signal_strength=signal, seed=SEED)
    broker = PaperBroker(initial_cash=CAPITAL)
    mandate = Mandate(state_dir / "mandate.json", MandateConfig())
    scfg = SwarmRankConfig(
        allow_deep=False, decision=decision,
        conviction=ConvictionConfig(n_scenarios=800, lookback=300, min_lookback=120))
    scfg.spa_boot = 200

    rows, problems = [], []
    for d in full.prices.index[-n_sessions:]:
        md = full.slice(None, d)
        broker.set_prices({s: float(md.prices[s].iloc[-1]) for s in md.prices.columns})
        if mandate.state.started_utc is None:
            mandate.start(nav=float(broker.equity), cash=float(broker.cash))
        ccfg = CycleConfig(dry_run=False, state_dir=str(state_dir), as_of=str(d.date()),
                           decision=decision)
        t0 = time.perf_counter()
        rep = DailyCycle(md, broker, mandate, scfg, ccfg).run()
        secs = time.perf_counter() - t0

        if rep.status != "completed":                                           # C1
            problems.append(f"{d.date()}: estado {rep.status}: {rep.notes}")
        if broker.cash < -1e-6:                                                 # C2
            problems.append(f"{d.date()}: efectivo negativo {broker.cash:.4f}")
        rejected = [(o["symbol"], o.get("message")) for o in rep.orders
                    if o.get("status") == "rejected"]
        if rejected:
            problems.append(f"{d.date()}: ordenes rechazadas {rejected[:3]}")

        n_blotter = len(broker.blotter)                                         # C3
        again = DailyCycle(md, broker, mandate, scfg, ccfg).run()
        if again.status != "skipped" or len(broker.blotter) != n_blotter:
            problems.append(f"{d.date()}: la repeticion no se omitio ({again.status}, "
                            f"{len(broker.blotter) - n_blotter} ordenes nuevas)")

        eq = float(broker.equity)
        pos = broker.positions()
        filled = [o for o in rep.orders if o.get("status") == "filled"]
        rows.append({
            "session": rep.session, "status": rep.status,
            "mandate": rep.mandate.get("status"),
            "buys": sum(o["side"] == "buy" for o in filled),
            "sells": sum(o["side"] == "sell" for o in filled),
            "positions": len(pos),
            "invested": round(1.0 - broker.cash / eq, 4) if eq > 0 else 0.0,
            "max_weight": round(max((p.market_value / eq for p in pos.values()),
                                    default=0.0), 4),
            "equity": round(eq, 2), "seconds": round(secs, 2)})

    journal = [json.loads(x) for x in (state_dir / "cycles.jsonl").read_text().splitlines()]
    real = [r["session"] for r in journal
            if r.get("status") == "completed" and not r.get("dry_run", True)]
    if len(real) != len(set(real)):                                             # C4
        problems.append("mas de un ciclo real completado para la misma sesion")
    orders = sum(r["buys"] + r["sells"] for r in rows)
    if decision == "gated" and signal == 0.0 and orders:                        # C5
        problems.append(f"el modo gated envio {orders} ordenes sobre ruido")
    if decision == "conviction" and min(r["invested"] for r in rows) < 0.99:    # C6
        problems.append("el modo conviction dejo mas del 1 % sin invertir")

    return {"decision": decision, "signal_strength": signal, "sessions": len(rows),
            "orders_filled": orders, "real_cycles_in_journal": len(real),
            "journal_lines": len(journal),
            "positions_min_max": [min(r["positions"] for r in rows),
                                  max(r["positions"] for r in rows)],
            "max_weight_max": max(r["max_weight"] for r in rows),
            "mean_seconds_per_cycle": round(sum(r["seconds"] for r in rows) / len(rows), 2),
            "final_equity": rows[-1]["equity"], "problems": problems, "rows": rows}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sessions", type=int, default=24)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1]
                                         / "docs/results/cycle_replay.json"))
    args = ap.parse_args()
    warnings.filterwarnings("ignore")
    runs = []
    for decision, signal in SCENARIOS:
        with tempfile.TemporaryDirectory() as tmp:
            run = replay(decision, signal, args.sessions, Path(tmp))
        runs.append(run)
        print(f"  {decision:<10} senal={signal}: {run['orders_filled']} ordenes, "
              f"posiciones {run['positions_min_max']}, mayor peso {run['max_weight_max']:.1%}, "
              f"{len(run['problems'])} problemas", flush=True)
        for p in run["problems"]:
            print("    -", p)
    res = {"design": {"n_symbols": N_SYM, "n_days": N_DAYS, "seed": SEED, "capital": CAPITAL,
                      "data": "sintetico (load_synthetic); broker simulado en memoria",
                      "criteria": "C1-C6 del docstring, fijados el 2026-11-13 antes de ejecutar",
                      "scope": "tuberia de ordenes; no mide rentabilidad"},
           "passed": not any(r["problems"] for r in runs), "runs": runs}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(f"{'CUMPLE' if res['passed'] else 'NO CUMPLE'}  guardado en {args.out}")
    return 0 if res["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
