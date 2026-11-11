"""Exportacion de resultados: CSV de curvas y grafico PNG (matplotlib opcional)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def export_run(engine, champion, verdict: dict, out_dir: str | Path) -> dict[str, str]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    files: dict[str, str] = {}

    # 1. curvas: campeon vs baselines sobre la union de ventanas
    n = len(champion.returns_history)
    curves = {"champion": np.cumprod(1.0 + np.asarray(champion.returns_history))}
    for k, v in engine.baseline_history.items():
        arr = np.asarray(v[-n:]) if len(v) >= n else np.asarray(v)
        curves[k] = np.cumprod(1.0 + arr)
    m = min(len(c) for c in curves.values())
    df = pd.DataFrame({k: c[-m:] for k, c in curves.items()})
    p = out / "equity_curves.csv"; df.to_csv(p, index=False); files["equity_curves"] = str(p)

    # 2. reports por generacion
    rows = [{**r.__dict__, "window": f"{r.window[0]}..{r.window[1]}"} for r in engine.reports]
    p = out / "generations.csv"; pd.DataFrame(rows).to_csv(p, index=False); files["generations"] = str(p)

    # 3. veredicto + genoma
    p = out / "champion.json"
    p.write_text(json.dumps({"run_id": engine.run_id, "verdict": verdict,
                             "genome": champion.genome.to_dict()}, indent=2, ensure_ascii=False, default=str))
    files["champion"] = str(p)

    # 4. grafico
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, 1, figsize=(10, 7), gridspec_kw={"height_ratios": [3, 1]})
        for k in df.columns:
            ax[0].plot(df[k].to_numpy(), label=k, lw=2 if k == "champion" else 1)
        ax[0].set_title(f"run {engine.run_id} — campeon vs baselines, ultimas {m} sesiones (vida del campeon, OOS)")
        ax[0].legend(); ax[0].grid(alpha=.3)
        g = pd.DataFrame(rows)
        ax[1].plot(g["best_sharpe"], label="mejor agente de CADA gen (cambia: sesgo de seleccion)"); ax[1].plot(g["baseline_sharpe"], label="baseline momentum+calidad")
        ax[1].set_xlabel("generacion"); ax[1].legend(); ax[1].grid(alpha=.3)
        fig.tight_layout()
        p = out / "report.png"; fig.savefig(p, dpi=110); plt.close(fig); files["plot"] = str(p)
    except Exception as exc:  # noqa: BLE001
        files["plot_error"] = str(exc)
    return files
