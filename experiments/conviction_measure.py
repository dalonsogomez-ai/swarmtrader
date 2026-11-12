"""Medicion PRE-REGISTRADA del asignador por conviccion (v1.4.0).

Criterios fijados el 2026-11-12, ANTES de ejecutar este script (copiados de
`.handoff/context/CONTINUACION_h4t8.md` §6, redactado el 2026-11-11/12). No se
cambian a la vista de los resultados, ni se cambia ningun valor por defecto por
lo que salga aqui.

Datos: SINTETICOS (`load_synthetic`). En este equipo no hay panel real de
precios; nada de lo que sale aqui dice como se comporta el asignador en el
mercado real.

Diseno: semillas 0-9, momentum_strength en {0; 0,25}, 60 acciones, 1.000
sesiones. En cada semilla, 6 fechas de analisis mensuales no solapadas
(repeticion historica: el enjambre solo ve precios hasta esa fecha); retorno
realizado a 21 sesiones despues. 60 observaciones pareadas por intensidad.

  H1 (ruido)  con senal 0: el mapa encogido sale plano (lambda medio < 0,2) y
              el crecimiento logaritmico realizado de la cartera de conviccion
              NO supera al de equiponderar el universo (contraste pareado
              unilateral, alfa = 0,05). Si lo supera, hay una fuga.
  H2 (senal)  con momentum 0,25: la cartera de conviccion concentra mas
              (posiciones efectivas menores) que en H1 y su crecimiento
              realizado supera al del universo (unilateral, alfa = 0,05).
  H3 (encogimiento) en ruido, shrink=True produce menos concentracion y no
              peor crecimiento realizado que shrink=False (no se rechaza, a
              alfa = 0,05, que shrink=False sea mejor).
  H4 (gamma)  las posiciones efectivas crecen de forma monotona con gamma en
              {0,5; 1; 2; 4} (en la media de las observaciones).

Uso: python experiments/conviction_measure.py [--out docs/results/conviction_measure.json]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import swarmtrader.swarm_rank as sr  # noqa: E402
from swarmtrader.data.loaders import MarketData, load_synthetic  # noqa: E402
from swarmtrader.portfolio.conviction import ConvictionConfig, allocate  # noqa: E402

SEEDS = range(10)
STRENGTHS = (0.0, 0.25)
N_SYM, N_DAYS, H, N_DATES = 60, 1000, 21, 6
GAMMAS = (0.5, 1.0, 2.0, 4.0)
ALPHA = 0.05


def _eff(w: pd.Series) -> float:
    s = float(w.sum())
    return s * s / float((w ** 2).sum()) if s > 0 else 0.0


def _growth(w: pd.Series, r: pd.Series) -> float:
    return float(np.log1p((w * r.reindex(w.index).fillna(0.0)).sum()))


def run_one(seed: int, strength: float) -> list[dict]:
    md = load_synthetic(n_symbols=N_SYM, n_days=N_DAYS, signal_strength=0.0,
                        momentum_strength=strength, seed=seed)
    cfg = sr.SwarmRankConfig(allow_deep=False, decision="conviction",
                             conviction=ConvictionConfig(n_scenarios=1500, seed=seed))
    cfg.spa_boot = 300
    captured: dict = {}
    original = sr.allocate

    def spy(*args, **kw):
        captured["args"], captured["kw"] = args, kw
        return original(*args, **kw)

    rows = []
    last = len(md.prices) - 1 - H
    for k in range(N_DATES):
        p = last - k * H
        sub = MarketData(md.prices.iloc[:p + 1], md.volumes.iloc[:p + 1])
        sr.allocate = spy
        try:
            res = sr.swarm_rank(sub, cfg)
        finally:
            sr.allocate = original
        c = res.conviction
        r = md.prices.iloc[p + H] / md.prices.iloc[p] - 1.0
        elig = list(res.picks["symbol"])
        ew = pd.Series(1.0 / len(elig), index=elig)
        top = res.picks.head(5)["symbol"]
        oos, scores, prices, horizon = captured["args"][:4]
        alt = {}
        for g in GAMMAS:
            a = allocate(oos, scores, prices, horizon, replace(cfg.conviction, risk_aversion=g))
            alt[f"eff_g{g}"] = _eff(a.weights)
        ns = allocate(oos, scores, prices, horizon, replace(cfg.conviction, shrink=False))
        rows.append({
            "seed": seed, "strength": strength, "date": str(md.prices.index[p].date()),
            "actionable": bool(res.verdict.actionable),
            "lambda": float(c.info.get("map_lambda_mean", np.nan)),
            "eff": _eff(c.weights), "top_weight": float(c.weights.max()),
            "g_conv": _growth(c.weights, r), "g_univ": _growth(ew, r),
            "g_top5": _growth(pd.Series(0.2, index=top), r),
            "eff_noshrink": _eff(ns.weights), "g_noshrink": _growth(ns.weights, r),
            **alt,
        })
    return rows


def _one_sided(d: np.ndarray) -> tuple[float, float]:
    """(media, p) del contraste unilateral H0: media <= 0 frente a media > 0."""
    t = stats.ttest_1samp(d, 0.0, alternative="greater")
    return float(d.mean()), float(t.pvalue)


def evaluate(df: pd.DataFrame) -> dict:
    noise, sig = df[df.strength == 0.0], df[df.strength > 0.0]
    m1, p1 = _one_sided((noise.g_conv - noise.g_univ).to_numpy())
    m2, p2 = _one_sided((sig.g_conv - sig.g_univ).to_numpy())
    m3, p3 = _one_sided((noise.g_noshrink - noise.g_conv).to_numpy())
    eff_by_g = [float(df[f"eff_g{g}"].mean()) for g in GAMMAS]
    out = {
        "H1": {"lambda_mean": float(noise["lambda"].mean()),
               "excess_growth_vs_universe": m1, "p_conv_beats_universe": p1,
               "pass": bool(noise["lambda"].mean() < 0.2 and p1 >= ALPHA)},
        "H2": {"eff_signal": float(sig.eff.mean()), "eff_noise": float(noise.eff.mean()),
               "excess_growth_vs_universe": m2, "p_conv_beats_universe": p2,
               "pass": bool(sig.eff.mean() < noise.eff.mean() and p2 < ALPHA)},
        "H3": {"eff_shrink": float(noise.eff.mean()),
               "eff_noshrink": float(noise.eff_noshrink.mean()),
               "growth_noshrink_minus_shrink": m3, "p_noshrink_better": p3,
               "pass": bool(noise.eff.mean() >= noise.eff_noshrink.mean() and p3 >= ALPHA)},
        "H4": {"gammas": list(GAMMAS), "eff_mean": eff_by_g,
               "pass": bool(all(b >= a - 1e-9 for a, b in zip(eff_by_g, eff_by_g[1:])))},
        "context": {
            "n_obs_per_strength": int(len(noise)),
            "actionable_rate_noise": float(noise.actionable.mean()),
            "actionable_rate_signal": float(sig.actionable.mean()),
            "growth_mean": {k: {"conv": float(g.g_conv.mean()), "univ": float(g.g_univ.mean()),
                                "top5": float(g.g_top5.mean())}
                            for k, g in (("noise", noise), ("signal", sig))},
            "top_weight_mean": {"noise": float(noise.top_weight.mean()),
                                "signal": float(sig.top_weight.mean())},
        },
    }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1]
                                         / "docs/results/conviction_measure.json"))
    args = ap.parse_args()
    t0 = time.time()
    rows = []
    for s in STRENGTHS:
        for seed in SEEDS:
            rows += run_one(seed, s)
            print(f"  momentum={s} semilla={seed}: {len(rows)} observaciones "
                  f"({time.time() - t0:.0f} s)", flush=True)
    df = pd.DataFrame(rows)
    res = evaluate(df)
    res["design"] = {"seeds": list(SEEDS), "strengths": list(STRENGTHS), "n_symbols": N_SYM,
                     "n_days": N_DAYS, "horizon": H, "dates_per_seed": N_DATES,
                     "data": "sintetico (load_synthetic); sin costes de transaccion",
                     "preregistered": "2026-11-12, criterios en el docstring"}
    res["rows"] = rows
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in res.items() if k not in ("rows",)}, indent=2,
                     ensure_ascii=False))
    print(f"guardado en {args.out}  ({time.time() - t0:.0f} s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
