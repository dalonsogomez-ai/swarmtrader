"""Medicion PRE-REGISTRADA de los agentes de valor, calidad y resultados (v1.6).

Escrito el 2026-11-13 ANTES de ejecutarlo sobre datos reales. Las hipotesis, la
metrica, la correccion por comparacion multiple y el umbral no se cambian despues
de ver el resultado.

DATOS
  * Puntuaciones fuera de muestra del enjambre de precio (oos.parquet: date,
    symbol, score, fwd), 80 fechas no solapadas cada 21 sesiones.
  * Panel de precios con cierre nominal (para la capitalizacion).
  * Fundamentales SEC point-in-time (data/fundamentals.py): fecha = presentacion,
    usados desde la sesion siguiente (available_lag = 1).
  * Universo de SUPERVIVIENTES: sirve para comparar reglas entre si, no para
    afirmar rentabilidad. Las empresas que quebraron no estan; el efecto de valor
    sobre supervivientes esta sesgado a favor.

COMPUESTOS (orientacion FIJADA por la tesis, sin ajustar pesos con los datos)
  value    media de rangos de bm, ep, cfp
  quality  media de rangos de gp_a, fscore, accruals, inv_ag, anti_issuance
  earnings rango de sue (solo vigente 63 sesiones tras la presentacion)
  combinado = media de rangos de (enjambre, value, quality, earnings): la misma
  regla equiponderada que usa el enjambre cuando el stacking no la bate.
  Un valor ausente cuenta como neutral (rango medio).

HIPOTESIS (unilaterales; bootstrap de bloques de 3 fechas, 5.000 remuestreos)
  H1  IC medio de quality a 21 sesiones > 0
  H2  IC medio de value > 0
  H3  IC medio de earnings > 0
  H4  exceso de los 5 primeros del combinado sobre el universo  >  el mismo
      exceso de los 5 primeros del enjambre solo (diferencia emparejada por fecha)
  H5  "valor profundo": dentro del tercio de mayor bm, los 5 de mayor fscore
      baten a la media de ese tercio
  Correccion de Holm sobre H1-H5 con alfa = 0,05.

CONTROL DE FUGA: se comprueba que ningun valor usado en la fecha d procede de una
presentacion posterior a d menos una sesion.

EXPLORATORIO (no decide nada): exceso del decil superior, P(>+25 %), P(<-15 %),
corte por precio < 5 USD frente a >= 5 USD, cobertura.

Uso:
  python experiments/fundamentals_measure.py --oos runs/oos.parquet \
      --panel data/stocks_panel_equity.parquet --fundamentals data/fundamentals_sec.csv \
      --earnings data/earnings_sec.csv --out docs/results/fundamentals_measure.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from swarmtrader.data.signals import earnings_features, fundamentals_features  # noqa: E402

RNG = np.random.default_rng(20261113)
VALUE = ("bm", "ep", "cfp")
QUALITY = ("gp_a", "fscore", "accruals", "inv_ag", "anti_issuance")


def block_boot_p(x: np.ndarray, B: int = 5000, block: int = 3) -> tuple[float, float, float]:
    x = x[np.isfinite(x)]
    n = len(x)
    nb = int(np.ceil(n / block))
    starts = RNG.integers(0, n - block + 1, size=(B, nb))
    idx = (starts[:, :, None] + np.arange(block)).reshape(B, -1)[:, :n]
    m = x[idx].mean(axis=1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)), float((m <= 0).mean())


def holm(p: dict[str, float]) -> dict[str, float]:
    out, prev = {}, 0.0
    items = sorted(p.items(), key=lambda kv: kv[1])
    for i, (k, v) in enumerate(items):
        prev = max(prev, min(1.0, v * (len(items) - i)))
        out[k] = round(prev, 4)
    return out


def rank_mean(df: pd.DataFrame, cols) -> pd.Series:
    cols = [c for c in cols if c in df.columns]
    r = df.groupby("date")[cols].rank(pct=True)
    return r.fillna(0.5).mean(axis=1)


def topk_excess(g: pd.DataFrame, col: str, k: int = 5) -> float:
    top = g.nlargest(k, col)
    return float(top["fwd"].mean() - g["fwd"].mean())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--oos", required=True)
    ap.add_argument("--panel", required=True)
    ap.add_argument("--fundamentals", required=True)
    ap.add_argument("--earnings", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    oos = pd.read_parquet(a.oos).dropna(subset=["score", "fwd"])
    syms = sorted(oos["symbol"].unique())
    p = pd.read_parquet(a.panel, columns=["date", "ticker", "adjclose", "close_nominal"])
    p = p[p["ticker"].isin(syms)]
    px = p.pivot(index="date", columns="ticker", values="adjclose").sort_index()
    nom = p.pivot(index="date", columns="ticker", values="close_nominal").reindex_like(px)
    dates = px.index

    fe = fundamentals_features(a.fundamentals, dates, syms, px, available_lag=1, nominal=nom)
    ee = earnings_features(a.earnings, dates, syms, px, available_lag=1)[["sue"]]
    feats = fe.join(ee, how="outer")
    feats.index.names = ["date", "symbol"]
    d = oos.merge(feats.reset_index(), on=["date", "symbol"], how="left")
    d.loc[d["sue"] == 0.0, "sue"] = np.nan            # fuera de la ventana de deriva

    # control de fuga: el valor usado en la sesion d debe ser el de la ultima
    # presentacion fechada en la sesion ANTERIOR o antes (available_lag = 1)
    fund = pd.read_csv(a.fundamentals, parse_dates=["date"]).sort_values("date")
    chk = d[["date", "symbol", "fscore"]].dropna().sample(min(2000, d["fscore"].notna().sum()),
                                                          random_state=1)
    pos = {t: i for i, t in enumerate(dates)}
    leaks = checked = 0
    for _, r in chk.iterrows():
        i = pos.get(r["date"])
        if not i:
            continue
        # el panel arrastra el ultimo valor NO nulo: se compara con ese
        f = fund[(fund["symbol"] == r["symbol"]) & (fund["date"] <= dates[i - 1])]
        f = f.dropna(subset=["fscore"])
        if f.empty:
            leaks += 1                       # hay valor sin presentacion previa: fuga
            continue
        checked += 1
        if not np.isclose(f.groupby("date")["fscore"].last().iloc[-1], r["fscore"],
                          equal_nan=True):
            leaks += 1

    d["value"] = rank_mean(d, VALUE)
    d["quality"] = rank_mean(d, QUALITY)
    d["earnings"] = d.groupby("date")["sue"].rank(pct=True).fillna(0.5)
    d["swarm"] = d.groupby("date")["score"].rank(pct=True)
    d["combined"] = d[["swarm", "value", "quality", "earnings"]].mean(axis=1)

    def ic(col):
        return d.groupby("date").apply(lambda g: g[col].corr(g["fwd"], method="spearman")).to_numpy()

    res = {"fechas": int(d["date"].nunique()), "obs": int(len(d)),
           "cobertura": {c: round(float(d[c].notna().mean()), 3) for c in VALUE + QUALITY + ("sue",)},
           "control_fuga": {"muestras": int(checked), "discrepancias": int(leaks)}}
    tests = {}
    for h, col in (("H1_quality", "quality"), ("H2_value", "value"), ("H3_earnings", "earnings")):
        x = ic(col)
        lo, hi, pv = block_boot_p(x)
        tests[h] = pv
        res[h] = {"ic_medio": round(float(np.nanmean(x)), 4), "ic95": [round(lo, 4), round(hi, 4)],
                  "p": round(pv, 4)}
    ex_c = d.groupby("date").apply(lambda g: topk_excess(g, "combined")).to_numpy()
    ex_s = d.groupby("date").apply(lambda g: topk_excess(g, "swarm")).to_numpy()
    lo, hi, pv = block_boot_p(ex_c - ex_s)
    tests["H4_combinado_top5"] = pv
    res["H4_combinado_top5"] = {"exceso_combinado": round(float(ex_c.mean()), 5),
                                "exceso_enjambre": round(float(ex_s.mean()), 5),
                                "diferencia": round(float((ex_c - ex_s).mean()), 5),
                                "ic95": [round(lo, 5), round(hi, 5)], "p": round(pv, 4)}

    def deep_value(g):
        g = g.dropna(subset=["bm", "fscore"])
        if len(g) < 30:
            return np.nan
        cheap = g[g["bm"] >= g["bm"].quantile(2 / 3)]
        return topk_excess(cheap, "fscore")
    dv = d.groupby("date").apply(deep_value).to_numpy()
    lo, hi, pv = block_boot_p(dv)
    tests["H5_valor_profundo"] = pv
    res["H5_valor_profundo"] = {"exceso": round(float(np.nanmean(dv)), 5),
                                "ic95": [round(lo, 5), round(hi, 5)], "p": round(pv, 4)}
    res["holm"] = holm(tests)

    # exploratorio
    ex = {}
    for col in ("swarm", "value", "quality", "earnings", "combined"):
        top10 = d.groupby("date").apply(lambda g: g.nlargest(max(1, len(g) // 10), col)["fwd"].mean()
                                        - g["fwd"].mean())
        top5 = d.groupby("date", group_keys=False).apply(lambda g: g.nlargest(5, col))
        ex[col] = {"exceso_decil_superior": round(float(top10.mean()), 5),
                   "top5_p_mas25": round(float((top5["fwd"] > 0.25).mean()), 4),
                   "top5_p_menos15": round(float((top5["fwd"] < -0.15).mean()), 4)}
    ex["universo"] = {"p_mas25": round(float((d["fwd"] > 0.25).mean()), 4),
                      "p_menos15": round(float((d["fwd"] < -0.15).mean()), 4)}
    last_nom = nom.reindex(d["date"].unique()).stack().rename("nom").reset_index()
    last_nom.columns = ["date", "symbol", "nom"]
    d = d.merge(last_nom, on=["date", "symbol"], how="left")
    for name, mask in (("precio<5", d["nom"] < 5), ("precio>=5", d["nom"] >= 5)):
        sub = d[mask]
        ex[f"ic_quality_{name}"] = round(float(sub.groupby("date").apply(
            lambda g: g["quality"].corr(g["fwd"], method="spearman")).mean()), 4)
        ex[f"ic_value_{name}"] = round(float(sub.groupby("date").apply(
            lambda g: g["value"].corr(g["fwd"], method="spearman")).mean()), 4)
    res["exploratorio"] = ex
    res["aviso"] = ("PRE-REGISTRADO salvo el bloque exploratorio. Universo de supervivientes "
                    "(Yahoo) y mapeo CIK-ticker actual: comparacion entre reglas, no rentabilidad.")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2, ensure_ascii=False))
    print(json.dumps(res, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
