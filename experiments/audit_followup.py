#!/usr/bin/env python3
"""Seguimiento de la auditoria. EXPLORATORIO: escrito DESPUES de ver el bloque A
de `geometry_audit.py`, asi que nada de aqui cuenta como contraste confirmatorio.

Tres preguntas que abrio A1:
  F1  ¿Cuanta parte del universo que pasa el filtro `min_price` tenia en realidad
      un precio NOMINAL inferior a 1 USD? El filtro usa el precio ajustado por
      splits, que incorpora splits inversos FUTUROS. ¿Y cual es el IC por tramo
      nominal estricto?
  F2  El apilado multiplica su puntuacion por el acuerdo entre agentes
      (`StackConfig.shrink_on_disagreement`). ¿Cambia el IC del apilado?
  F3  ¿Que dice el enjambre completo, con sus cuatro filtros, sobre estos datos
      reales? Se informa para acciones de 5 USD o mas y para todo el universo.
      ES UN UNIVERSO DE SUPERVIVIENTES Y SIN COSTES: el veredicto no autoriza
      nada; solo muestra que los filtros se ejecutan sobre datos reales.

  F4  La via "agente": un agente geometrico (ridge sobre las tres features
      geometricas) anadido al enjambre equiponderado, en vez de meter las
      features en un unico modelo, que es lo que contrasto el bloque B.
  F5  ¿Aporta el enjambre algo frente a UN solo modelo ridge con los 14 factores?
  F6  Version continua de la hipotesis de valor del filtro G0: ¿el IC del
      enjambre en una fecha es menor cuanto mas lejos esta el estado del mercado
      de su ventana de entrenamiento? (Con el indicador binario solo habia 2
      fechas marcadas y no se pudo contrastar.)
  F7  ¿De donde sale el IC? El sistema solo COMPRA los primeros puestos. Se mide
      el retorno a 21 sesiones por decil de puntuacion, el de los 5 primeros y
      el de los 5 ultimos frente al universo, y el IC dentro de cada mitad.
  F8  ¿Cambia algo entrenar con un objetivo orientado a la parte alta? Mismo
      ridge y mismos 14 factores, cuatro etiquetas: rango relativo (la actual),
      pertenecer al quintil superior, superar +5% y retorno recortado. Se mide
      lo que el sistema compra: los 5 primeros frente al universo. AVISO: en un
      universo de supervivientes faltan justo los valores volatiles que
      acabaron en cero, asi que este bloque esta sesgado A FAVOR de las
      etiquetas que persiguen subidas grandes.

Uso: python3 experiments/audit_followup.py --stocks-panel panel.parquet [--json salida.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import pandas as pd
from scipy import stats

from geometry_audit import GEO, geo_features, ic_series, load_panel, spa_p, stack, summarize
from swarmtrader.agents.specialists import run_swarm
from swarmtrader.ensemble.stacking import StackConfig, per_date_ic, stack_agents
from swarmtrader.rank import RankConfig, _prepare, walk_forward_oos
from swarmtrader.stats.drift import DriftConfig, compute_states, drift_at
from swarmtrader.swarm_rank import SwarmRankConfig, swarm_rank


def f3_swarm(md) -> dict:
    """Enjambre completo sobre datos reales (supervivientes, sin costes)."""
    res = {}
    for label, mp in (("precio>=5", 5.0), ("precio>=1", 1.0)):
        cfg = SwarmRankConfig(rank=RankConfig(min_price=mp, min_dollar_volume=1e6, oos_months=85),
                              allow_deep=False, spa_boot=2000)
        r = swarm_rank(md, cfg)
        d = r.to_dict()
        v = d["verdict"]
        res[label] = {
            "accionable": v["actionable"], "razones": v["reasons"],
            "G1_spa_p_destreza": v["p_skill"], "G2_spa_p_apilado_vs_eq": v["p_stack_vs_eq"],
            "G3_pbo": v["pbo"], "G4_p_top_k": v["p_topk"], "G4_exceso_top_k": v["topk_excess"],
            "combinador": v["combiner"], "evidencia": v["evidence"],
            "diagnostico": d["diagnostics"], "deriva": d["drift"],
            "pesos_agentes": d["agent_weights"], "elegibles": d["n_eligible"],
        }
        print(label, json.dumps({k: res[label][k] for k in res[label] if k not in ("diagnostico", "deriva")},
                                indent=1, default=str), flush=True)
    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stocks-panel", required=True)
    ap.add_argument("--tickers", type=int, default=1000)
    ap.add_argument("--json", default=None)
    ap.add_argument("--only-f6", action="store_true")
    ap.add_argument("--only-f7", action="store_true")
    ap.add_argument("--only-f8", action="store_true")
    ap.add_argument("--only-f3", action="store_true")
    a = ap.parse_args()
    md, nom, _ = load_panel(a.stocks_panel, a.tickers)
    rc = RankConfig(min_price=1.0, min_dollar_volume=1e6, oos_months=85)
    df = _prepare(md, rc)
    out: dict = {}

    if a.only_f3:
        res = f3_swarm(md)
        if a.json:
            Path(a.json).write_text(json.dumps({"F3_enjambre_real": res}, indent=1, default=str), encoding="utf-8")
        return 0

    if a.only_f8:
        res = {}
        for label, mp in (("precio>=5", 5.0), ("precio>=1", 1.0)):
            rcx = RankConfig(min_price=mp, min_dollar_volume=1e6, oos_months=85)
            base = df if mp == 1.0 else _prepare(md, rcx)
            f = base["fwd_21"]
            g = f.groupby(level="date")
            targets = {
                "rango relativo (actual)": base["target"],
                "quintil superior": (g.rank(pct=True) >= 0.8).astype(float).where(f.notna()),
                "supera +5%": (f > 0.05).astype(float).where(f.notna()),
                "retorno recortado": (f - g.transform("median")).clip(-0.5, 0.5),
            }
            res[label] = {}
            for name, tg in targets.items():
                d2 = base.copy()
                d2.attrs = dict(base.attrs)
                d2["target"] = tg
                o = walk_forward_oos(d2, rcx)
                rows = []
                for _, x in o.groupby("date"):
                    x = x.dropna(subset=["fwd"]).sort_values("score")
                    if len(x) < 50:
                        continue
                    top = x.tail(5)
                    rows.append({"exc": top["fwd"].mean() - x["fwd"].mean(), "top": top["fwd"].mean(),
                                 "p5": (top["fwd"] > 0.05).mean(), "p25": (top["fwd"] > 0.25).mean(),
                                 "m15": (top["fwd"] < -0.15).mean(), "u5": (x["fwd"] > 0.05).mean(),
                                 "u25": (x["fwd"] > 0.25).mean(), "um15": (x["fwd"] < -0.15).mean(),
                                 "ic": stats.spearmanr(x["score"], x["fwd"]).statistic})
                r = pd.DataFrame(rows)
                res[label][name] = {
                    "ic": round(float(r["ic"].mean()), 4),
                    "exceso_top5_21": round(float(r["exc"].mean()), 4),
                    "t_exceso": round(float(r["exc"].mean() / (r["exc"].std(ddof=1) / np.sqrt(len(r)))), 2),
                    "sd_exceso": round(float(r["exc"].std(ddof=1)), 4),
                    "top5_prob_mas_5pct": round(float(r["p5"].mean()), 3),
                    "top5_prob_mas_25pct": round(float(r["p25"].mean()), 4),
                    "top5_prob_menos_15pct": round(float(r["m15"].mean()), 3),
                    "universo": {"mas_5pct": round(float(r["u5"].mean()), 3), "mas_25pct": round(float(r["u25"].mean()), 4),
                                 "menos_15pct": round(float(r["um15"].mean()), 3)},
                }
        print(json.dumps(res, indent=1))
        if a.json:
            Path(a.json).write_text(json.dumps({"F8_etiquetas": res}, indent=1), encoding="utf-8")
        return 0

    if a.only_f7:
        res = {}
        for label, mp in (("precio>=1", 1.0), ("precio>=5", 5.0)):
            rcx = RankConfig(min_price=mp, min_dollar_volume=1e6, oos_months=85)
            dfx = df if mp == 1.0 else _prepare(md, rcx)
            pan = run_swarm(dfx, rcx, md=md, allow_deep=False)
            s = pan.scores.copy()
            s["eq"] = s[pan.agents].mean(axis=1)
            rows, dec = [], []
            for d, g in s.groupby("date"):
                g = g.dropna(subset=["fwd"])
                if len(g) < 50:
                    continue
                g = g.sort_values("eq")
                top, bot = g.tail(5), g.head(5)
                half = len(g) // 2
                lo, hi = g.iloc[:half], g.iloc[half:]
                rows.append({
                    "u_media": g["fwd"].mean(), "u_mediana": g["fwd"].median(),
                    "top_media": top["fwd"].mean(), "top_mediana": top["fwd"].median(),
                    "bot_media": bot["fwd"].mean(),
                    "u_p5": (g["fwd"] > 0.05).mean(), "top_p5": (top["fwd"] > 0.05).mean(),
                    "u_p25": (g["fwd"] > 0.25).mean(), "top_p25": (top["fwd"] > 0.25).mean(),
                    "u_m15": (g["fwd"] < -0.15).mean(), "top_m15": (top["fwd"] < -0.15).mean(),
                    "bot_m15": (bot["fwd"] < -0.15).mean(),
                    "u_sd": g["fwd"].std(), "top_sd_corte": top["fwd"].std(),
                    "ic_mitad_alta": stats.spearmanr(hi["eq"], hi["fwd"]).statistic,
                    "ic_mitad_baja": stats.spearmanr(lo["eq"], lo["fwd"]).statistic})
                q = pd.qcut(g["eq"].rank(method="first"), 10, labels=False)
                dec.append(g.groupby(q.to_numpy())["fwd"].agg(["mean", "median"]).assign(
                    p5=g.groupby(q.to_numpy())["fwd"].apply(lambda x: (x > 0.05).mean())))
            r = pd.DataFrame(rows)
            m = r.mean()
            exc = r["top_media"] - r["u_media"]
            excb = r["bot_media"] - r["u_media"]
            D = pd.concat(dec).groupby(level=0).mean()
            res[label] = {
                "fechas": int(len(r)),
                "retorno_medio_21": {"universo": round(float(m["u_media"]), 4), "top5": round(float(m["top_media"]), 4),
                                     "ultimos5": round(float(m["bot_media"]), 4)},
                "retorno_mediano_21": {"universo": round(float(m["u_mediana"]), 4), "top5": round(float(m["top_mediana"]), 4)},
                "exceso_top5": {"media": round(float(exc.mean()), 4),
                                "t": round(float(exc.mean() / (exc.std(ddof=1) / np.sqrt(len(exc)))), 2)},
                "exceso_ultimos5": {"media": round(float(excb.mean()), 4),
                                    "t": round(float(excb.mean() / (excb.std(ddof=1) / np.sqrt(len(excb)))), 2)},
                "prob_mas_5pct": {"universo": round(float(m["u_p5"]), 3), "top5": round(float(m["top_p5"]), 3)},
                "prob_mas_25pct": {"universo": round(float(m["u_p25"]), 4), "top5": round(float(m["top_p25"]), 4)},
                "prob_menos_15pct": {"universo": round(float(m["u_m15"]), 3), "top5": round(float(m["top_m15"]), 3),
                                     "ultimos5": round(float(m["bot_m15"]), 3)},
                "dispersion_retornos": {"universo": round(float(m["u_sd"]), 3), "top5": round(float(m["top_sd_corte"]), 3)},
                "ic_dentro_de_cada_mitad": {"alta": round(float(m["ic_mitad_alta"]), 4), "baja": round(float(m["ic_mitad_baja"]), 4)},
                "por_decil": {int(k) + 1: {"media": round(float(v["mean"]), 4), "mediana": round(float(v["median"]), 4),
                                           "prob_mas_5pct": round(float(v["p5"]), 3)} for k, v in D.iterrows()},
            }
        print(json.dumps(res, indent=1))
        if a.json:
            Path(a.json).write_text(json.dumps({"F7_origen_del_ic": res}, indent=1), encoding="utf-8")
        return 0

    if a.only_f6:
        panel = run_swarm(df, rc, md=md, allow_deep=False)
        sc = panel.scores.copy()
        sc["eq"] = sc[panel.agents].mean(axis=1)
        st = compute_states(df, md.prices, DriftConfig())
        rows = {}
        for d in sorted(sc["date"].unique()):
            t = st.dates.get_loc(pd.Timestamp(d))
            hi = t - 26
            r = drift_at(st, st.dates[max(0, hi - 756):max(0, hi)], st.dates[t], DriftConfig())
            if r.evaluated:
                rows[pd.Timestamp(d)] = {k: v["distance"] for k, v in r.channels.items()}
        dist = pd.DataFrame(rows).T
        res = {"fechas": int(len(dist))}
        for col in ["eq"] + list(panel.agents):
            ic = ic_series(sc, col, "fwd")
            idx = dist.index.intersection(ic.index)
            res[col] = {}
            for ch in dist.columns:
                s = stats.spearmanr(dist.loc[idx, ch], ic.reindex(idx))
                res[col][ch] = {"spearman": round(float(s.statistic), 3), "p_bilateral": round(float(s.pvalue), 4)}
            if col == "eq":
                far = dist.loc[idx].rank(pct=True).max(axis=1) >= 0.8        # quinto mas alejado
                res["eq"]["ic_quinto_mas_alejado"] = round(float(ic.reindex(idx)[far].mean()), 4)
                res["eq"]["ic_resto"] = round(float(ic.reindex(idx)[~far].mean()), 4)
                res["eq"]["sd_ic_quinto_mas_alejado"] = round(float(ic.reindex(idx)[far].std()), 4)
                res["eq"]["sd_ic_resto"] = round(float(ic.reindex(idx)[~far].std()), 4)
        print(json.dumps(res, indent=1))
        if a.json:
            Path(a.json).write_text(json.dumps({"F6_distancia_vs_ic": res}, indent=1), encoding="utf-8")
        return 0

    # ---- F1: precio nominal de lo que pasa el filtro ajustado
    n = stack(nom, "nominal").reindex(df.index)
    out["F1_filtro_precio"] = {
        "filas_elegibles": int(len(df)),
        "fraccion_nominal_menor_1": round(float((n < 1.0).mean()), 4),
        "fraccion_nominal_1_a_5": round(float(((n >= 1.0) & (n < 5.0)).mean()), 4),
        "fraccion_nominal_5_o_mas": round(float((n >= 5.0).mean()), 4),
        "fraccion_ajustado_difiere_mas_10pct": round(float(
            (np.abs(df["px"] / n - 1.0) > 0.10).mean()), 4),
    }
    panel = run_swarm(df, rc, md=md, allow_deep=False)
    sc = panel.scores.join(stack(nom, "nominal"), on=["date", "symbol"])
    sc["eq"] = sc[panel.agents].mean(axis=1)
    tiers = {"nominal<1": sc[sc["nominal"] < 1.0],
             "nominal 1-5": sc[(sc["nominal"] >= 1.0) & (sc["nominal"] < 5.0)],
             "nominal>=5": sc[sc["nominal"] >= 5.0],
             "nominal>=10": sc[sc["nominal"] >= 10.0]}
    out["F1_ic_eq_por_tramo_nominal"] = {k: {**summarize(ic_series(v, "eq", "fwd")),
                                             "filas": int(len(v))} for k, v in tiers.items()}
    out["F1_retorno_21_mediano_por_tramo"] = {k: round(float(v["fwd"].median()), 4) for k, v in tiers.items()}
    print(json.dumps(out, indent=1), flush=True)

    # ---- F2: encogimiento del apilado por desacuerdo
    res = {}
    for name, flag in (("con_encogimiento", True), ("sin_encogimiento", False)):
        st = stack_agents(panel.scores, panel.agents, StackConfig(shrink_on_disagreement=flag))
        res[name] = per_date_ic(st.scores, "stack")
        eq = per_date_ic(st.scores, "eq")
    out["F2_apilado"] = {
        "ic_eq": summarize(eq), "ic_apilado_con": summarize(res["con_encogimiento"]),
        "ic_apilado_sin": summarize(res["sin_encogimiento"]),
        "diferencia_sin_menos_con": round(float((res["sin_encogimiento"] - res["con_encogimiento"]).mean()), 4),
        "p_quitar_no_mejora": spa_p([res["sin_encogimiento"] - res["con_encogimiento"]], 11),
        "p_apilado_sin_no_bate_a_eq": spa_p([res["sin_encogimiento"] - eq], 12),
        "p_apilado_con_no_bate_a_eq": spa_p([res["con_encogimiento"] - eq], 13),
    }
    print(json.dumps(out["F2_apilado"], indent=1), flush=True)

    # ---- F4: agente geometrico dentro del enjambre equiponderado
    attrs = dict(df.attrs)
    dfx = df.join(geo_features(md.prices).reindex(df.index).fillna(0.0))
    dfx.attrs = attrs
    g = walk_forward_oos(dfx, rc, x_cols=GEO)[["date", "symbol", "score"]]
    g["geo"] = g.groupby("date")["score"].rank(pct=True) * 2 - 1
    m = sc.merge(g[["date", "symbol", "geo"]], on=["date", "symbol"])
    m["eq6"] = (m[panel.agents].sum(axis=1) + m["geo"]) / (len(panel.agents) + 1)
    i5, i6 = ic_series(m, "eq", "fwd"), ic_series(m, "eq6", "fwd")
    out["F4_agente_geometrico_en_enjambre"] = {
        "ic_eq_5_agentes": summarize(i5), "ic_eq_6_con_geo": summarize(i6),
        "ic_geo_solo": summarize(ic_series(m, "geo", "fwd")),
        "diferencia": round(float((i6 - i5).mean()), 4), "p_anadirlo_no_mejora": spa_p([i6 - i5], 21),
        "por_tramo": {k: round(float((ic_series(v, "eq6", "fwd") - ic_series(v, "eq", "fwd")).mean()), 4)
                      for k, v in {"nominal>=5": m[m["nominal"] >= 5.0], "nominal<5": m[m["nominal"] < 5.0]}.items()},
    }
    print(json.dumps(out["F4_agente_geometrico_en_enjambre"], indent=1), flush=True)

    # ---- F5: enjambre frente a un unico modelo con todos los factores
    one = walk_forward_oos(df, rc)[["date", "symbol", "score"]].rename(columns={"score": "ridge14"})
    m5 = sc.merge(one, on=["date", "symbol"])
    st0 = stack_agents(panel.scores, panel.agents, StackConfig(shrink_on_disagreement=False)).scores
    m5 = m5.merge(st0[["date", "symbol", "stack"]], on=["date", "symbol"], how="left")
    ia, ie, is_ = ic_series(m5, "ridge14", "fwd"), ic_series(m5, "eq", "fwd"), ic_series(m5.dropna(subset=["stack"]), "stack", "fwd")
    out["F5_enjambre_vs_modelo_unico"] = {
        "ic_ridge_14_factores": summarize(ia), "ic_enjambre_eq": summarize(ie), "ic_enjambre_apilado": summarize(is_),
        "diferencia_ridge_menos_eq": round(float((ia - ie).mean()), 4),
        "p_ridge_no_bate_a_eq": spa_p([ia - ie], 22), "p_eq_no_bate_a_ridge": spa_p([ie - ia], 23),
        "por_tramo_ridge_menos_eq": {
            k: round(float((ic_series(v, "ridge14", "fwd") - ic_series(v, "eq", "fwd")).mean()), 4)
            for k, v in {"nominal>=5": m5[m5["nominal"] >= 5.0], "nominal<5": m5[m5["nominal"] < 5.0]}.items()},
    }
    print(json.dumps(out["F5_enjambre_vs_modelo_unico"], indent=1), flush=True)

    out["F3_enjambre_real"] = f3_swarm(md)
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
