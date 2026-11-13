"""Experimento: ¿ampliar el universo mejora la seleccion del top-K?

PREREGISTRO (escrito antes de ejecutar, 2026-11-13)
Datos: puntuaciones fuera de muestra reales del enjambre (--oos, p. ej. oos.parquet,
80 fechas no solapadas cada 21 sesiones, 2020-03 a 2026-10, ~2.500 acciones por fecha)
y panel de precios (--panel). Universo de supervivientes:
valido solo para comparar reglas entre si, nunca para afirmar rentabilidad.

Metrica: exceso a 21 sesiones del top-K (media de sus retornos menos la media del
universo considerado), por fecha; media sobre fechas; IC 95% por bootstrap de bloques
de 3 fechas (5.000 remuestreos).

H1 (ampliacion aleatoria, K=5): exceso(todo) - exceso(N=500) > 0 con IC 95% > 0.
H2 (ampliacion hacia lo pequeno: universo = N mas liquidas por volumen en dolares
    mediano de 21 sesiones previas, K=5): exceso(todo) - exceso(500 mas liquidas) > 0.
Correccion: Holm sobre H1 y H2 (p unilateral por bootstrap).
Control C1 (potencia del metodo): una puntuacion sintetica con IC uniforme ~0,075 debe
    mostrar pendiente positiva en la ampliacion aleatoria; si no, el metodo no sirve.
Exploratorio (no decide nada): K=1, P(>+25%), P(<-15%), IC por mitades.
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

RNG = np.random.default_rng(20261113)
GRID = [100, 250, 500, 1000, 1500, 2000]
DRAWS = 300
KS = (1, 5)


def load(oos_path, panel_path):
    o = pd.read_parquet(oos_path)
    p = pd.read_parquet(panel_path,
                        columns=["date", "ticker", "close_nominal", "dollar_volume"])
    p = p.sort_values(["ticker", "date"])
    # liquidez PIT: mediana de 21 sesiones que terminan en la fecha (inclusive: la señal usa el cierre)
    p["dv21"] = p.groupby("ticker")["dollar_volume"].transform(
        lambda s: s.rolling(21, min_periods=15).median())
    p = p.rename(columns={"ticker": "symbol"})
    o = o.merge(p[["date", "symbol", "dv21", "close_nominal"]], on=["date", "symbol"], how="left")
    return o


def block_boot(x: np.ndarray, B: int = 5000, block: int = 3) -> tuple[float, float, float]:
    n = len(x)
    nb = int(np.ceil(n / block))
    starts = RNG.integers(0, n - block + 1, size=(B, nb))
    idx = (starts[:, :, None] + np.arange(block)).reshape(B, -1)[:, :n]
    m = x[idx].mean(axis=1)
    lo, hi = np.percentile(m, [2.5, 97.5])
    p_one = float((m <= 0).mean())  # unilateral: H0 diferencia <= 0
    return float(lo), float(hi), p_one


def topk_stats(s, r, idx, k):
    top = idx[np.argpartition(-s[idx], k - 1)[:k]] if k < len(idx) else idx
    return (r[top].mean() - r[idx].mean(), (r[top] > 0.25).mean(), (r[top] < -0.15).mean())


def run_random(groups, score_col, k):
    out = {N: [] for N in GRID + ["todo"]}
    for _, g in groups:
        s = g[score_col].to_numpy(); r = g["fwd"].to_numpy(); n = len(s)
        allidx = np.arange(n)
        out["todo"].append(topk_stats(s, r, allidx, k))
        for N in GRID:
            acc = np.array([topk_stats(s, r, RNG.choice(n, N, replace=False), k) for _ in range(DRAWS)])
            out[N].append(acc.mean(axis=0))
    return {N: np.array(v) for N, v in out.items()}


def run_liquid(groups, score_col, k):
    out = {N: [] for N in GRID + ["todo"]}
    for _, g in groups:
        g = g.sort_values("dv21", ascending=False, na_position="last")
        s = g[score_col].to_numpy(); r = g["fwd"].to_numpy(); n = len(s)
        for N in GRID + ["todo"]:
            m = n if N == "todo" else min(N, n)
            out[N].append(topk_stats(s, r, np.arange(m), k))
    return {N: np.array(v) for N, v in out.items()}


def summarize(res, univ_p25=None):
    rows = {}
    for N, a in res.items():
        rows[str(N)] = {"exceso_medio": round(float(a[:, 0].mean()), 5),
                        "p_mas25_top": round(float(a[:, 1].mean()), 4),
                        "p_menos15_top": round(float(a[:, 2].mean()), 4)}
    return rows


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--oos", required=True)
    ap.add_argument("--panel", required=True)
    ap.add_argument("--out", default="universe_size_result.json")
    a = ap.parse_args()
    o = load(a.oos, a.panel)
    o = o.dropna(subset=["score", "fwd"])
    # control sintetico: IC uniforme ~0,075 sobre toda la distribucion
    def synth(g):
        z = (g["fwd"].rank().to_numpy() - 0.5) / len(g)
        from scipy.stats import norm
        zz = norm.ppf(z)
        return pd.Series(0.075 * zz + np.sqrt(1 - 0.075 ** 2) * RNG.standard_normal(len(g)), index=g.index)
    o["synth"] = o.groupby("date", group_keys=False).apply(synth)

    groups = list(o.groupby("date"))
    n_by_date = [len(g) for _, g in groups]

    # IC por mitades (replica de F7 sobre esta ejecucion)
    ics = {"alta": [], "baja": [], "total": []}
    for _, g in groups:
        med = g["score"].median()
        ics["total"].append(g["score"].corr(g["fwd"], method="spearman"))
        for name, sub in (("alta", g[g.score >= med]), ("baja", g[g.score < med])):
            ics[name].append(sub["score"].corr(sub["fwd"], method="spearman"))
    res = {"fechas": len(groups), "acciones_por_fecha_mediana": int(np.median(n_by_date)),
           "ic_mitades": {k: round(float(np.nanmean(v)), 4) for k, v in ics.items()}}

    for k in KS:
        rr = run_random(groups, "score", k)
        rl = run_liquid(groups, "score", k)
        rs = run_random(groups, "synth", k)
        d1 = rr["todo"][:, 0] - rr[500][:, 0]
        d2 = rl["todo"][:, 0] - rl[500][:, 0]
        dc = rs["todo"][:, 0] - rs[500][:, 0]
        b1, b2, bc = block_boot(d1), block_boot(d2), block_boot(dc)
        res[f"K={k}"] = {
            "aleatoria": summarize(rr), "hacia_lo_pequeno": summarize(rl), "control_sintetico": summarize(rs),
            "H1_todo_menos_500": {"media": round(float(d1.mean()), 5), "ic95": [round(b1[0], 5), round(b1[1], 5)], "p_unilateral": round(b1[2], 4)},
            "H2_todo_menos_500_liquidas": {"media": round(float(d2.mean()), 5), "ic95": [round(b2[0], 5), round(b2[1], 5)], "p_unilateral": round(b2[2], 4)},
            "C1_control_todo_menos_500": {"media": round(float(dc.mean()), 5), "ic95": [round(bc[0], 5), round(bc[1], 5)], "p_unilateral": round(bc[2], 4)},
        }
    # Holm para K=5 (preregistrado)
    ps = sorted([("H1", res["K=5"]["H1_todo_menos_500"]["p_unilateral"]),
                 ("H2", res["K=5"]["H2_todo_menos_500_liquidas"]["p_unilateral"])], key=lambda t: t[1])
    holm, prev = {}, 0.0
    for i, (h, p) in enumerate(ps):
        adj = min(1.0, max(prev, p * (len(ps) - i))); holm[h] = round(adj, 4); prev = adj
    res["holm_K5"] = holm
    json.dump(res, open(a.out, "w"), indent=2, ensure_ascii=False)
    print(json.dumps(res, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    sys.exit(main())
