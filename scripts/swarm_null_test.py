#!/usr/bin/env python3
"""Test de control del ENJAMBRE. Ejecutalo antes de creerte cualquier salida.

Dos mitades, y las dos importan:

  * **Especificidad**: sobre un mercado sin ninguna senal, el veredicto debe ser
    NO ACCIONABLE y la cartera debe quedar vacia. Un sistema que opera sobre
    ruido te arruina despacio.
  * **Potencia**: sobre un mercado con momentum inyectado y conocido, el
    veredicto debe ser ACCIONABLE. Un detector que nunca dispara es tan inutil
    como uno que dispara siempre, y es un fallo mucho mas facil de no ver.

Si cualquiera de las dos mitades falla, hay un bug o una fuga y ningun
resultado posterior significa nada.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from swarmtrader.data.loaders import load_synthetic
from swarmtrader.swarm_rank import SwarmRankConfig, swarm_rank


DEEP = os.environ.get("SWARMTRADER_DEEP") == "1"


def _run(momentum: float, seed: int = 7):
    cfg = SwarmRankConfig(allow_deep=DEEP)
    cfg.spa_boot = 800
    if DEEP:
        from swarmtrader.models.deep import DeepConfig
        cfg.rank.oos_months = 8
        cfg.deep = DeepConfig(epochs=8, seeds=2, patience=3, seq_len=40,
                              max_dates_per_epoch=80)
    md = load_synthetic(n_symbols=60 if not DEEP else 40, n_days=900 if not DEEP else 850,
                        signal_strength=0.0, momentum_strength=momentum, seed=seed)
    return swarm_rank(md, cfg), cfg


def main() -> int:
    ok = True
    print(f"capa profunda: {'ACTIVA' if DEEP else 'inactiva (SWARMTRADER_DEEP=1 para activarla)'}")

    res, _ = _run(0.0)
    v = res.verdict
    gross = float(res.portfolio["weight"].sum()) if not res.portfolio.empty else 0.0
    print(f"[especificidad] mercado sin senal -> accionable={v.actionable}  "
          f"SPA p={v.p_skill:.3f}  PBO={v.pbo:.2f}  evidencia={v.evidence}  bruto={gross:.1%}")
    if v.actionable or gross > 1e-9:
        print("  FALLO: el enjambre opera sobre ruido.")
        ok = False
    else:
        print("  OK: el enjambre se niega a operar sobre ruido.")

    res, _ = _run(0.25)
    v = res.verdict
    gross = float(res.portfolio["weight"].sum()) if not res.portfolio.empty else 0.0
    print(f"[potencia]      momentum inyectado -> accionable={v.actionable}  "
          f"SPA p={v.p_skill:.3f}  PBO={v.pbo:.2f}  evidencia={v.evidence}  bruto={gross:.1%}")
    if not v.actionable:
        print("  FALLO: no detecta una senal que existe y es fuerte.")
        ok = False
    else:
        print("  OK: detecta la senal y dimensiona la cartera.")

    print("TODO OK" if ok else "HAY FALLOS")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
