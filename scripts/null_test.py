#!/usr/bin/env python3
"""Test de control. Ejecutalo ANTES de creerte ningun resultado.

Corre el torneo sobre un mercado sin ninguna senal y comprueba que el gate DSR
no certifica al campeon. Si certifica, hay un bug o una fuga de informacion y
cualquier resultado positivo posterior no significa nada.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from swarmtrader.backtest import FeaturePanel
from swarmtrader.data.features import compute_features, dollar_volume
from swarmtrader.data.loaders import load_synthetic
from swarmtrader.evolution.engine import SwarmConfig, SwarmEngine
from swarmtrader.stats.registry import TrialRegistry


def main() -> int:
    md = load_synthetic(n_symbols=100, n_days=1400, signal_strength=0.0, seed=1234)
    panel = FeaturePanel(compute_features(md.prices, md.volumes), md.prices,
                         dollar_volume(md.prices, md.volumes))
    eng = SwarmEngine(panel, SwarmConfig(population_size=30, generations=12, seed=99),
                      TrialRegistry("runs/null_test.db"))
    eng.run(verbose=False)
    champ = eng.champion()
    v = eng.certify(champ)
    print(f"mercado sin senal -> Sharpe campeon {v['sharpe_annualized']:+.2f}  "
          f"P(edge)={v['edge_probability']:.3f}  DSR={v['deflated_sharpe_ratio']:.4f}")
    if v["certified"]:
        print("FALLO: se certifico un agente en un mercado sin senal. Hay un bug.")
        return 1
    print("OK: el gate rechaza correctamente el ruido.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
