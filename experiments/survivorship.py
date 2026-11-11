#!/usr/bin/env python3
"""Cuantifica el sesgo de supervivencia sobre LAS METRICAS DE ESTE SISTEMA.

La literatura dice que el sesgo de supervivencia infla los retornos historicos.
Eso no sirve para decidir nada aqui: lo que hace falta saber es cuanto infla el
**rank IC y la precision@k de este pipeline concreto**, porque son esas dos
cifras las que gobiernan las puertas G1/G3 y el nivel de evidencia.

El experimento compara tres configuraciones sobre EL MISMO mercado sintetico,
que incluye delistings con retorno terminal conocido:

  A) SESGADA         solo los simbolos vivos al final. Es lo que ves si
                     descargas tickers de Alpaca, eToro o yfinance hoy.
  B) PANEL COMPLETO  incluye a los deslistados, pero con la etiqueta ingenua
                     `precio[t+h]/precio[t]-1`, que devuelve NaN si la accion
                     muere dentro del horizonte.
  C) POINT-IN-TIME   panel completo Y etiqueta con retorno terminal.

La diferencia A-C es el sesgo total. La diferencia B-C aisla la parte que entra
**por la etiqueta**, que es la que casi nadie corrige y la que no se arregla
solo con comprar datos que incluyan deslistados.

Uso:
    python3 experiments/survivorship.py [--symbols 80] [--days 900] [--seeds 3]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from swarmtrader.data.loaders import MarketData
from swarmtrader.data.pit import load_synthetic_pit
from swarmtrader.rank import RankConfig, _prepare, diagnose, walk_forward_oos


def _run(md: MarketData, listings: pd.DataFrame | None, cfg: RankConfig):
    df = _prepare(md, cfg, listings=listings)
    oos = walk_forward_oos(df, cfg)
    d = diagnose(oos, cfg)
    return d, oos


def _topk_return(oos: pd.DataFrame, k: int) -> float:
    """Retorno medio realizado del top-k por fecha. La metrica economica.

    Es la que de verdad delata el sesgo: el IC mide calidad de ORDENACION y
    puede mejorar al incluir perdedores predecibles, pero el retorno del top-k
    mide lo que habrias ganado, que es lo que paga las facturas.
    """
    if oos.empty:
        return float("nan")
    picks = oos.sort_values("score", ascending=False).groupby("date").head(k)
    return float(picks["fwd"].mean())


def _fmt(tag: str, d, n_sym: int) -> str:
    return (f"  {tag:<28} n={n_sym:>3}  IC {d.mean_rank_ic:+.4f}  t={d.ic_tstat:+6.2f}  "
            f"prec@{d.precision_at_k and 5} {d.precision_at_k:.1%}  base {d.base_rate:.1%}  "
            f"lift {d.precision_lift:+.1%}  ev={d.evidence}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", type=int, default=80)
    ap.add_argument("--days", type=int, default=900)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--momentum", type=float, default=0.15,
                    help="senal observable inyectada; >0 para que haya algo que medir")
    ap.add_argument("--delist-rate", type=float, default=0.06)
    ap.add_argument("--drift-days", type=int, default=120,
                    help="sesiones de declive previo a la quiebra; 0 = muerte subita")
    args = ap.parse_args()

    cfg = RankConfig(oos_months=12)
    rows = []
    regimen = "muerte anunciada" if args.drift_days > 0 else "MUERTE SUBITA"
    print(f"Mercado sintetico: {args.symbols} simbolos, {args.days} sesiones, "
          f"tasa de baja anual {args.delist_rate:.0%}, momentum inyectado {args.momentum}, "
          f"regimen: {regimen} ({args.drift_days}d)")
    print("=" * 100)

    for s in range(args.seeds):
        panel = load_synthetic_pit(n_symbols=args.symbols, n_days=args.days,
                                   momentum_strength=args.momentum,
                                   annual_delist_rate=args.delist_rate,
                                   pre_death_drift_days=args.drift_days, seed=3 + s)
        full = panel.to_market_data()
        biased = panel.survivors_only().to_market_data()

        dA, oA = _run(biased, None, cfg)
        dB, oB = _run(full, None, cfg)
        dC, oC = _run(full, panel.listings, cfg)
        rA, rB, rC = (_topk_return(o, 5) for o in (oA, oB, oC))

        print(f"\nsemilla {3+s}   {panel.summary()['por_motivo']}")
        print(_fmt("A) SESGADA (solo vivos)", dA, biased.prices.shape[1]) + f"  top5 {rA:+.2%}")
        print(_fmt("B) panel completo, label naive", dB, full.prices.shape[1]) + f"  top5 {rB:+.2%}")
        print(_fmt("C) POINT-IN-TIME", dC, full.prices.shape[1]) + f"  top5 {rC:+.2%}")
        rows.append(dict(seed=3 + s,
                         ic_A=dA.mean_rank_ic, ic_B=dB.mean_rank_ic, ic_C=dC.mean_rank_ic,
                         p_A=dA.precision_at_k, p_B=dB.precision_at_k, p_C=dC.precision_at_k,
                         b_A=dA.base_rate, b_B=dB.base_rate, b_C=dC.base_rate,
                         l_A=dA.precision_lift, l_B=dB.precision_lift, l_C=dC.precision_lift,
                         r_A=rA, r_B=rB, r_C=rC))

    r = pd.DataFrame(rows)
    print("\n" + "=" * 100)
    print("MEDIA SOBRE SEMILLAS")
    print(f"  rank IC          A {r.ic_A.mean():+.4f}   B {r.ic_B.mean():+.4f}   C {r.ic_C.mean():+.4f}")
    print(f"  precision@5      A {r.p_A.mean():.1%}   B {r.p_B.mean():.1%}   C {r.p_C.mean():.1%}")
    print(f"  tasa base        A {r.b_A.mean():.1%}   B {r.b_B.mean():.1%}   C {r.b_C.mean():.1%}")
    print(f"  lift sobre base  A {r.l_A.mean():+.1%}   B {r.l_B.mean():+.1%}   C {r.l_C.mean():+.1%}")
    print(f"  retorno top-5    A {r.r_A.mean():+.2%}   B {r.r_B.mean():+.2%}   C {r.r_C.mean():+.2%}  <-- metrica economica")
    print()
    print(f"  SESGO TOTAL   (A - C):  IC {r.ic_A.mean()-r.ic_C.mean():+.4f}   "
          f"lift {r.l_A.mean()-r.l_C.mean():+.1%}   RETORNO TOP-5 {r.r_A.mean()-r.r_C.mean():+.2%}")
    print(f"  SOLO ETIQUETA (B - C):  IC {r.ic_B.mean()-r.ic_C.mean():+.4f}   "
          f"lift {r.l_B.mean()-r.l_C.mean():+.1%}   RETORNO TOP-5 {r.r_B.mean()-r.r_C.mean():+.2%}")
    print()
    print("  Interpretacion: la tasa base es la probabilidad de que una accion cualquiera")
    print("  supere el umbral. Si la vista sesgada la sube, todo el sistema esta calibrado")
    print("  contra un mundo donde nadie quiebra, y la P(exito) que reporta es optimista.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
