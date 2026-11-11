#!/usr/bin/env python3
"""Auditoria de modelos y agentes + todas las vias geometricas, sobre datos reales.

PRE-REGISTRO (escrito antes de ejecutar; no se edita despues de ver resultados)
==============================================================================
Datos: muestra aleatoria con semilla fija de un panel de acciones ordinarias
cotizadas HOY en EE. UU. (2018-2026). Es un universo de SUPERVIVIENTES: sirve
para comparar dos reglas sobre los mismos casos, no para afirmar rentabilidad.
Elegibilidad point-in-time: precio nominal >= 1 USD, volumen en dolares >= 1 M.

Bloque A — los modelos y agentes que ya existen
  A1  IC de cada agente contra (i) el retorno de la etiqueta actual, que entra
      al cierre de la misma sesion en que se calcula la senal, y (ii) el mismo
      retorno entrando UNA sesion despues; por tramo de precio (>= 5 USD y 1-5).
      Regla: si el IC del conjunto cae mas de un tercio con el retraso en algun
      tramo, la etiqueta sin retraso se considera contaminada por microestructura
      (rebote bid-ask) y se corrige en el sistema.
  A2  Redundancia: correlacion media por fecha entre las puntuaciones de agentes.
  A4  Clase de modelo: ridge (actual) frente a gradient boosting y a su media.
      H0: ninguno bate a ridge. Solo cambia el valor por defecto si SPA p < 0,05.

Bloque B — geometria como AGENTE (features por accion)
  geo_fr_shift, geo_fr_signed: distancia de Fisher-Rao entre la gaussiana de los
      retornos diarios de las ultimas 21 sesiones y la de las 231 anteriores.
  geo_periphery: periferia en la red de correlaciones (menos correlacion media
      con el resto, ventana de 126 sesiones). Tesis publicada: Pozzi et al. 2013.
  H0: anadirlas a los 14 factores no mejora el IC fuera de muestra (SPA sobre
  tres variantes). Regla: solo se crea el agente si p < 0,05 y la mejora
  sobrevive al retraso de entrada.

Bloque C — geometria como MODELO / filtro de estado de mercado
  C1  Deteccion de seis episodios de estres fijados aqui de antemano:
        2020-02-20..2020-04-30  COVID
        2021-01-22..2021-02-10  estrangulamiento de cortos (meme stocks)
        2022-01-03..2022-06-30  subida de tipos
        2023-03-09..2023-03-31  bancos regionales
        2024-08-01..2024-08-15  deshacer del carry en yenes
        2025-04-02..2025-04-30  aranceles
      Monitores comparados al mismo alpha nominal total (1%):
        M0 linea base trivial: volatilidad de mercado a 21 sesiones;
        M1 dispersion (Fisher-Rao gaussiana);
        M2 dispersion + dependencia (el G0 actual);
        M3 M2 + forma (Wasserstein-2 entre funciones cuantil estandarizadas).
      Regla: el canal de forma solo se anade si detecta al menos un episodio que
      M2 no detecta, sin sumar mas de 1 punto de dias marcados fuera de episodios.
      Si M0 detecta lo mismo que M2 con no mas dias fuera de episodio, se
      documenta que la geometria no mejora la deteccion sobre la linea base.
  C2  TDA (Gidea y Katz 2018): norma L1 del paisaje de persistencia H1 de la nube
      de 50 sesiones de 4 indices. H0: no anade capacidad de prediccion de la
      varianza realizada a 21 sesiones una vez controlada la varianza pasada.
      Regla: |t| >= 3 del coeficiente (umbral de Harvey, Liu y Zhu 2016).
  C3  Curvatura de Ollivier-Ricci media de la red de correlaciones (ventana 22,
      umbral 0,75 + arbol generador minimo). Regla: si |Spearman| >= 0,8 con la
      correlacion media, se trata como reformulacion de esta.

Bloque D — geometria sobre el propio ENJAMBRE
  D1  La puntuacion final se multiplica hoy por el "acuerdo" entre agentes. No
      estaba validado. H0a: no mejora el IC. H0b: no lo empeora.
  D2  Agregador: media de rangos (la actual; es la media de Frechet bajo la
      distancia de Spearman) frente a mediana y a media en logits. H0: ninguna
      bate a la media.
  D3  Coherencia del enjambre (correlacion media entre agentes en la fecha) como
      incertidumbre epistemica. H: mas coherencia -> mas IC en esa fecha.

Comparacion multiple: seis familias con contraste (A4, B, C2, D1, D2, D3); los
p-valores de familia se ajustan por Holm.

Uso:
    python3 experiments/geometry_audit.py --stocks-panel panel.parquet \
        [--indices-dir dir_con_json_de_yahoo] [--json salida.json]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd
from scipy import stats

from swarmtrader.agents.specialists import run_swarm
from swarmtrader.data.features import _zscore_rows
from swarmtrader.data.loaders import MarketData
from swarmtrader.rank import RankConfig, _prepare, walk_forward_oos
from swarmtrader.stats.drift import DriftConfig, compute_states, drift_at
from swarmtrader.stats.infogeo import fisher_rao_gaussian
from swarmtrader.stats.spa import spa_test

H, EMBARGO, TRAIN_SESSIONS = 21, 5, 756
EPISODES = {
    "COVID 2020": ("2020-02-20", "2020-04-30"),
    "meme squeeze 2021": ("2021-01-22", "2021-02-10"),
    "tipos 2022": ("2022-01-03", "2022-06-30"),
    "bancos regionales 2023": ("2023-03-09", "2023-03-31"),
    "carry yen 2024": ("2024-08-01", "2024-08-15"),
    "aranceles 2025": ("2025-04-02", "2025-04-30"),
}
GEO = ["geo_fr_shift", "geo_fr_signed", "geo_periphery"]


# ============================================================ utilidades
def load_panel(path: str, n_tickers: int, seed: int = 20261111):
    p = pd.read_parquet(path, columns=["date", "ticker", "adjclose", "close_nominal", "dollar_volume"])
    p = p[p["ticker"].notna() & (p["ticker"] != "SPY")]
    p["ticker"] = p["ticker"].astype(str)
    tickers = np.array(sorted(p["ticker"].unique()))
    keep = np.random.default_rng(seed).choice(tickers, min(n_tickers, len(tickers)), replace=False)
    p = p[p["ticker"].isin(set(keep))]
    px = p.pivot(index="date", columns="ticker", values="adjclose").sort_index()
    px.index = pd.to_datetime(px.index)
    nom = p.pivot(index="date", columns="ticker", values="close_nominal").reindex(columns=px.columns)
    dv = p.pivot(index="date", columns="ticker", values="dollar_volume").reindex(columns=px.columns)
    nom.index = dv.index = px.index
    return MarketData(px, (dv / px).fillna(0.0)), nom, dv


def stack(wide: pd.DataFrame, name: str) -> pd.Series:
    s = wide.stack(future_stack=True)
    s.index.names = ["date", "symbol"]
    return s.rename(name)


def ic_series(df: pd.DataFrame, col: str, y: str, min_n: int = 20) -> pd.Series:
    g = df[["date", col, y]].dropna()
    if g.empty:
        return pd.Series(dtype=float)
    r = g.groupby("date")[[col, y]].rank()
    r["date"] = g["date"].to_numpy()
    out = {}
    for d, x in r.groupby("date"):
        if len(x) >= min_n and x[col].std() > 0 and x[y].std() > 0:
            out[d] = float(np.corrcoef(x[col], x[y])[0, 1])
    return pd.Series(out, dtype=float)


def summarize(ic: pd.Series) -> dict:
    n = len(ic)
    if n < 3:
        return {"n": n, "ic": None, "t": None}
    return {"n": n, "ic": round(float(ic.mean()), 4),
            "t": round(float(ic.mean() / (ic.std(ddof=1) / np.sqrt(n))), 2)}


def spa_p(diff_cols: list[pd.Series], seed: int = 0) -> float:
    idx = diff_cols[0].index
    for c in diff_cols[1:]:
        idx = idx.intersection(c.index)
    D = np.column_stack([c.reindex(idx).to_numpy(float) for c in diff_cols])
    if len(D) < 8:
        return float("nan")
    return float(spa_test(D, n_boot=3000, seed=seed).p_value)


def rank_norm(g: pd.Series) -> pd.Series:
    return g.groupby(level=0).rank(pct=True) if isinstance(g.index, pd.MultiIndex) else g.rank(pct=True)


# ============================================================ A y D
def block_a_d(md: MarketData, nom: pd.DataFrame, df: pd.DataFrame, rc: RankConfig) -> tuple[dict, pd.DataFrame]:
    panel = run_swarm(df, rc, md=md, allow_deep=False)
    sc = panel.scores.copy()
    agents = list(panel.agents)
    px = md.prices
    lag = stack(px.shift(-(H + 1)) / px.shift(-1) - 1.0, "fwd_lag")
    price = stack(nom, "price")
    sc = sc.join(lag, on=["date", "symbol"]).join(price, on=["date", "symbol"])
    A = sc[agents].to_numpy(float)
    sc["eq"] = A.mean(axis=1)
    out: dict = {"agentes": agents, "fechas": int(sc["date"].nunique()),
                 "nombres_por_fecha": int(sc.groupby("date").size().median())}

    # ---- A1: IC por agente, con y sin retraso de entrada, por tramo de precio
    tiers = {"todos": sc, ">=5": sc[sc["price"] >= 5.0], "1-5": sc[sc["price"] < 5.0]}
    a1 = {}
    for name in agents + ["eq"]:
        a1[name] = {f"{t}|{lab}": summarize(ic_series(d, name, y))
                    for t, d in tiers.items() for lab, y in (("sin_retraso", "fwd"), ("retraso_1", "fwd_lag"))}
    out["A1_ic"] = a1
    out["A1_caida_eq_por_retraso"] = {
        t: (None if not a1["eq"][f"{t}|sin_retraso"]["ic"] else
            round(1 - a1["eq"][f"{t}|retraso_1"]["ic"] / a1["eq"][f"{t}|sin_retraso"]["ic"], 3))
        for t in tiers}
    out["A1_fraccion_filas_1_5"] = round(float((sc["price"] < 5.0).mean()), 3)

    # ---- A2: redundancia entre agentes
    cors = []
    for _, g in sc.groupby("date"):
        if len(g) >= 20:
            cors.append(g[agents].rank().corr().to_numpy())
    C = np.nanmean(np.stack(cors), axis=0)
    out["A2_correlacion_media"] = {a: {b: round(float(C[i, j]), 3) for j, b in enumerate(agents)}
                                   for i, a in enumerate(agents)}

    # ---- D1: multiplicador de acuerdo
    disp = A.std(axis=1) / (2.0 / np.sqrt(3.0))
    sc["disp"] = disp
    sc["eq_acuerdo"] = sc["eq"] * np.clip(1.0 - disp, 0.0, 1.0)
    d1 = {}
    for lab, y in (("sin_retraso", "fwd"), ("retraso_1", "fwd_lag")):
        i0, i1 = ic_series(sc, "eq", y), ic_series(sc, "eq_acuerdo", y)
        d1[lab] = {"ic_eq": summarize(i0), "ic_eq_x_acuerdo": summarize(i1),
                   "diferencia": round(float((i1 - i0).mean()), 4),
                   "p_no_mejora": spa_p([i1 - i0], 1), "p_no_empeora": spa_p([i0 - i1], 2)}
    # dentro del quintil superior: mitad de bajo desacuerdo menos mitad de alto
    rows = {}
    for d, g in sc.groupby("date"):
        g = g.dropna(subset=["fwd"])
        if len(g) < 50:
            continue
        top = g[g["eq"] >= g["eq"].quantile(0.8)]
        med = top["disp"].median()
        rel = top["fwd"] - g["fwd"].median()
        rel_l = top["fwd_lag"] - g["fwd_lag"].median()
        lo = top["disp"] <= med
        rows[d] = (rel[lo].mean() - rel[~lo].mean(), rel_l[lo].mean() - rel_l[~lo].mean())
    q = pd.DataFrame(rows, index=["sin_retraso", "retraso_1"]).T
    d1["quintil_superior_bajo_menos_alto_desacuerdo"] = {
        c: {"media": round(float(q[c].mean()), 4), "p_bajo_no_es_mejor": spa_p([q[c].dropna()], 3)}
        for c in q.columns}
    out["D1_acuerdo"] = d1

    # ---- D2: agregadores
    sc["mediana"] = np.median(A, axis=1)
    u = np.clip((A + 1.0) / 2.0, 0.005, 0.995)
    sc["logit"] = np.log(u / (1 - u)).mean(axis=1)
    d2 = {}
    for lab, y in (("sin_retraso", "fwd"), ("retraso_1", "fwd_lag")):
        base = ic_series(sc, "eq", y)
        alt = {c: ic_series(sc, c, y) for c in ("mediana", "logit")}
        d2[lab] = {"ic_media": summarize(base), **{f"ic_{c}": summarize(v) for c, v in alt.items()},
                   "p_ninguna_bate_a_la_media": spa_p([v - base for v in alt.values()], 4)}
    out["D2_agregadores"] = d2

    # ---- D3: coherencia del enjambre por fecha
    coh = {}
    iu = np.triu_indices(len(agents), 1)
    for d, g in sc.groupby("date"):
        if len(g) >= 20:
            coh[d] = float(g[agents].rank().corr().to_numpy()[iu].mean())
    coh = pd.Series(coh)
    d3 = {}
    for lab, y in (("sin_retraso", "fwd"), ("retraso_1", "fwd_lag")):
        ic = ic_series(sc, "eq", y)
        idx = coh.index.intersection(ic.index)
        r = stats.spearmanr(coh.reindex(idx), ic.reindex(idx))
        d3[lab] = {"n": int(len(idx)), "spearman": round(float(r.statistic), 3),
                   "p_unilateral_positiva": round(float(r.pvalue / 2 if r.statistic > 0 else 1 - r.pvalue / 2), 4)}
    out["D3_coherencia"] = d3
    return out, sc


def block_a4(df: pd.DataFrame, rc: RankConfig, sc_lag: pd.Series, months: int) -> dict:
    from dataclasses import replace
    r = walk_forward_oos(df, replace(rc, oos_months=months, model="ridge"))
    g = walk_forward_oos(df, replace(rc, oos_months=months, model="gbdt", gbdt_seeds=1))
    m = r[["date", "symbol", "score", "fwd"]].rename(columns={"score": "ridge"}).merge(
        g[["date", "symbol", "score"]].rename(columns={"score": "gbdt"}), on=["date", "symbol"])
    m = m.join(sc_lag, on=["date", "symbol"])
    m["media"] = (m.groupby("date")["ridge"].rank(pct=True) + m.groupby("date")["gbdt"].rank(pct=True)) / 2
    out = {}
    for lab, y in (("sin_retraso", "fwd"), ("retraso_1", "fwd_lag")):
        base = ic_series(m, "ridge", y)
        alt = {c: ic_series(m, c, y) for c in ("gbdt", "media")}
        out[lab] = {"ic_ridge": summarize(base), **{f"ic_{c}": summarize(v) for c, v in alt.items()},
                    "p_ninguno_bate_a_ridge": spa_p([v - base for v in alt.values()], 5)}
    return out


# ============================================================ B
def geo_features(px: pd.DataFrame) -> pd.DataFrame:
    lr = np.log(px).diff()
    mu_s = lr.rolling(21, min_periods=15).mean()
    sd_s = lr.rolling(21, min_periods=15).std().clip(lower=1e-4)
    base = lr.shift(21)
    mu_l = base.rolling(231, min_periods=120).mean()
    sd_l = base.rolling(231, min_periods=120).std().clip(lower=1e-4)
    d = pd.DataFrame(fisher_rao_gaussian(mu_s.to_numpy(), sd_s.to_numpy(), mu_l.to_numpy(), sd_l.to_numpy()),
                     index=px.index, columns=px.columns)
    signed = d * np.sign(mu_s - mu_l)
    per = pd.DataFrame(np.nan, index=px.index, columns=px.columns)
    X = lr.to_numpy(float)
    for t in range(126, len(px), 21):
        w = X[t - 126:t]
        ok = np.isfinite(w).sum(axis=0) >= 100
        if ok.sum() < 30:
            continue
        z = w[:, ok]
        z = (z - np.nanmean(z, axis=0)) / np.nanstd(z, axis=0)
        z = np.nan_to_num(z)
        c = (z.T @ z) / len(z)
        mean_corr = (c.sum(axis=1) - np.diag(c)) / (c.shape[0] - 1)
        per.iloc[t - 1, np.flatnonzero(ok)] = -mean_corr       # conocido al cierre de t-1
    per = per.ffill(limit=20)
    feats = {"geo_fr_shift": d, "geo_fr_signed": signed, "geo_periphery": per}
    return pd.concat([stack(_zscore_rows(v), k) for k, v in feats.items()], axis=1)


def block_b(df: pd.DataFrame, rc: RankConfig, geo: pd.DataFrame, sc: pd.DataFrame, agents: list) -> dict:
    f14 = list(df.attrs["factor_cols"])
    attrs = dict(df.attrs)
    dfx = df.join(geo.reindex(df.index).fillna(0.0))
    dfx.attrs = attrs
    lag = sc.set_index(["date", "symbol"])["fwd_lag"]
    variants = {"base_14": f14, "14+fisher_rao": f14 + GEO[:2], "14+periferia": f14 + GEO[2:],
                "14+las_tres": f14 + GEO, "solo_geo": GEO}
    oos, ic, ic_l = {}, {}, {}
    for name, cols in variants.items():
        o = walk_forward_oos(dfx, rc, x_cols=cols)[["date", "symbol", "score", "fwd"]]
        o = o.join(lag, on=["date", "symbol"])
        oos[name] = o
        ic[name], ic_l[name] = ic_series(o, "score", "fwd"), ic_series(o, "score", "fwd_lag")
    ch = ["14+fisher_rao", "14+periferia", "14+las_tres"]
    out = {"ic": {k: {"sin_retraso": summarize(ic[k]), "retraso_1": summarize(ic_l[k])} for k in variants},
           "mejora_vs_base": {k: {"sin_retraso": round(float((ic[k] - ic["base_14"]).mean()), 4),
                                  "retraso_1": round(float((ic_l[k] - ic_l["base_14"]).mean()), 4)} for k in ch},
           "p_ninguna_variante_mejora": {"sin_retraso": spa_p([ic[k] - ic["base_14"] for k in ch], 6),
                                         "retraso_1": spa_p([ic_l[k] - ic_l["base_14"] for k in ch], 7)}}
    # IC univariante de cada feature (bilateral, descriptivo) y solape con agentes existentes
    raw = dfx[GEO + [f"fwd_{rc.horizon}"]].reset_index().rename(columns={f"fwd_{rc.horizon}": "fwd"})
    raw = raw[raw["date"].isin(set(sc["date"].unique()))]
    out["ic_univariante"] = {g: summarize(ic_series(raw, g, "fwd")) for g in GEO}
    m = oos["solo_geo"].rename(columns={"score": "geo"}).merge(sc[["date", "symbol"] + agents], on=["date", "symbol"])
    cor = {}
    for a in agents:
        v = [g[["geo", a]].rank().corr().iloc[0, 1] for _, g in m.groupby("date") if len(g) >= 20]
        cor[a] = round(float(np.nanmean(v)), 3)
    out["correlacion_agente_geo_con"] = cor
    return out


# ============================================================ C1
def _emp_p(ref: np.ndarray, v: float) -> float:
    ref = ref[np.isfinite(ref)]
    return float((1 + (ref >= v).sum()) / (1 + len(ref))) if len(ref) and np.isfinite(v) else np.nan


def quantile_states(df: pd.DataFrame, prices: pd.DataFrame, grid: np.ndarray) -> np.ndarray:
    r = stack(prices / prices.shift(21) - 1.0, "r").reindex(df.index).to_numpy(float)
    codes, dates = pd.factorize(df.index.get_level_values("date"), sort=True)
    order = np.argsort(codes, kind="stable")
    bounds = np.searchsorted(codes[order], np.arange(len(dates) + 1))
    Q = np.full((len(dates), len(grid)), np.nan)
    for t in range(len(dates)):
        v = r[order[bounds[t]:bounds[t + 1]]]
        v = v[np.isfinite(v)]
        if len(v) < 30:
            continue
        med = np.median(v)
        mad = 1.4826 * np.median(np.abs(v - med))
        if mad > 0:
            Q[t] = np.quantile((v - med) / mad, grid)
    return Q


def monitor_pvalues(df: pd.DataFrame, prices: pd.DataFrame, start: int = 400) -> pd.DataFrame:
    cfg = DriftConfig()
    st = compute_states(df, prices, cfg)
    grid = np.linspace(0.01, 0.99, 99)
    Q = quantile_states(df, prices, grid)
    mkt = prices.pct_change(fill_method=None).mean(axis=1)
    vol = (mkt.rolling(21, min_periods=10).std()).reindex(st.dates).to_numpy(float)
    okq = np.isfinite(Q).all(axis=1)
    rows = []
    for t in range(start, len(st.dates)):
        hi = t - H - EMBARGO
        ref = np.arange(max(0, hi - TRAIN_SESSIONS), max(0, hi))
        r = drift_at(st, st.dates[ref], st.dates[t], cfg)
        row = {"date": st.dates[t],
               "p_disp": r.channels.get("dispersion", {}).get("p_value", np.nan),
               "p_dep": r.channels.get("dependencia", {}).get("p_value", np.nan),
               "p_vol": _emp_p(vol[ref], vol[t]), "p_forma": np.nan}
        rq = ref[okq[ref]]
        if len(rq) >= cfg.min_ref_dates and okq[t]:
            center = Q[rq].mean(axis=0)                 # baricentro de Wasserstein-2 en 1D
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                d = np.sqrt(np.nanmean((Q - center) ** 2, axis=1))
            d = pd.Series(d).rolling(cfg.smooth, min_periods=1).median().to_numpy()
            row["p_forma"] = _emp_p(d[rq], d[t])
        rows.append(row)
    return pd.DataFrame(rows).set_index("date")


def evaluate_monitors(p: pd.DataFrame, alpha: float = 0.01) -> dict:
    fired = {
        "M0 volatilidad (linea base)": p["p_vol"] < alpha,
        "M1 dispersion": p["p_disp"] < alpha,
        "M2 dispersion+dependencia (G0)": (p["p_disp"] < alpha / 2) | (p["p_dep"] < alpha / 2),
        "M3 M2+forma": (p["p_disp"] < alpha / 3) | (p["p_dep"] < alpha / 3) | (p["p_forma"] < alpha / 3),
        "solo forma": p["p_forma"] < alpha,
        "solo dependencia": p["p_dep"] < alpha,
    }
    inside = pd.Series(False, index=p.index)
    for a, b in EPISODES.values():
        inside |= (p.index >= a) & (p.index <= b)
    pos = pd.Series(np.arange(len(p)), index=p.index)
    out = {}
    for name, f in fired.items():
        f = f.fillna(False)
        det = {}
        for ep, (a, b) in EPISODES.items():
            w = f[(f.index >= a) & (f.index <= b)]
            if len(w) == 0:
                det[ep] = None
            elif w.any():
                det[ep] = int(pos[w.index[w.to_numpy()][0]] - pos[w.index[0]])
            else:
                det[ep] = -1
        out[name] = {"episodios_detectados": int(sum(v is not None and v >= 0 for v in det.values())),
                     "retraso_sesiones": det, "tasa_total": round(float(f.mean()), 4),
                     "dias_marcados_fuera_de_episodios": round(float(f[~inside].mean()), 4),
                     "dias_marcados_dentro": round(float(f[inside].mean()), 4)}
    return out


def synthetic_shape_power(seeds: int) -> dict:
    """Ruptura de FORMA: una fraccion de nombres salta +80%; mediana y MAD no cambian."""
    res = {}
    for frac in (0.0, 0.01, 0.03, 0.05):
        hits = {"M2": 0, "M3": 0}
        for s in range(seeds):
            rng = np.random.default_rng(700 + s)
            n, T = 300, 800
            dates = pd.bdate_range("2021-01-04", periods=T)
            r = 0.01 * rng.normal(size=(T, 1)) + 0.02 * rng.normal(size=(T, n))
            k = int(frac * n)
            if k:
                r[-5, rng.choice(n, k, replace=False)] += np.log(1.8)
            px = pd.DataFrame(100 * np.exp(np.cumsum(r, axis=0)), index=dates,
                              columns=[f"S{i:03d}" for i in range(n)])
            f = pd.DataFrame(rng.normal(size=(T * n, 3)), columns=["f0", "f1", "f2"],
                             index=pd.MultiIndex.from_product([dates, px.columns], names=["date", "symbol"]))
            f.attrs["factor_cols"] = ["f0", "f1", "f2"]
            p = monitor_pvalues(f, px, start=T - 1).iloc[-1]
            hits["M2"] += bool((p["p_disp"] < 0.005) or (p["p_dep"] < 0.005))
            hits["M3"] += bool((p["p_disp"] < 0.01 / 3) or (p["p_dep"] < 0.01 / 3) or (p["p_forma"] < 0.01 / 3))
        res[f"{frac:.0%} de nombres con +80%"] = {k: v / seeds for k, v in hits.items()}
    return res


# ============================================================ C2: TDA
def _ols_hac(y: np.ndarray, X: np.ndarray, lags: int = 2):
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ b
    xe = X * e[:, None]
    S = xe.T @ xe
    for L in range(1, lags + 1):
        w = 1 - L / (lags + 1)
        G = xe[L:].T @ xe[:-L]
        S += w * (G + G.T)
    XtXi = np.linalg.inv(X.T @ X)
    V = XtXi @ S @ XtXi
    r2 = 1 - e.var() / y.var()
    return b, b / np.sqrt(np.diag(V)), float(r2)


def block_tda(indices_dir: str, w: int = 50) -> dict:
    from ripser import ripser
    closes = {}
    for f in sorted(Path(indices_dir).glob("*.json")):
        r = json.loads(f.read_text())["chart"]["result"][0]
        idx = pd.to_datetime(r["timestamp"], unit="s").normalize()
        closes[f.stem] = pd.Series(r["indicators"]["quote"][0]["close"], index=idx)
    R = np.log(pd.DataFrame(closes)).diff().dropna()
    X = R.to_numpy(float)
    spx = R[[c for c in R.columns if "GSPC" in c][0]].to_numpy(float)
    T = len(R)
    L1, var = np.full(T, np.nan), np.full(T, np.nan)
    for t in range(w, T + 1):
        cloud = X[t - w:t]
        dg = ripser(cloud, maxdim=1)["dgms"][1]
        pers = dg[:, 1] - dg[:, 0] if len(dg) else np.zeros(0)
        L1[t - 1] = float((pers ** 2).sum() / 4.0)      # norma L1 exacta del paisaje completo
        var[t - 1] = float(cloud.var(axis=0).sum())
    s = pd.DataFrame({"L1": L1, "var": var}, index=R.index).dropna()
    s = s[s["L1"] > 0]
    out = {"indices": list(R.columns), "desde": str(R.index[0].date()), "hasta": str(R.index[-1].date()),
           "ventanas": int(len(s)),
           "spearman_L1_vs_varianza": round(float(stats.spearmanr(s["L1"], s["var"]).statistic), 3),
           "spearman_L1_normalizada_vs_varianza":
               round(float(stats.spearmanr(s["L1"] / s["var"], s["var"]).statistic), 3)}
    # regresion predictiva, muestreo cada 21 sesiones (sin solape en la variable dependiente)
    rv21 = pd.Series(spx ** 2, index=R.index).rolling(21).sum()
    fut = rv21.shift(-21)
    fret = pd.Series(spx, index=R.index).rolling(21).sum().shift(-21)
    d = pd.DataFrame({"y": np.log(fut), "rv21": np.log(rv21), "var50": np.log(s["var"]),
                      "L1": np.log(s["L1"]), "L1n": np.log(s["L1"] / s["var"]), "fret": fret}).dropna()
    d = d.iloc[::21]
    one = np.ones(len(d))
    Xb = np.column_stack([one, d["rv21"], d["var50"]])
    _, _, r2b = _ols_hac(d["y"].to_numpy(), Xb)
    b, t, r2 = _ols_hac(d["y"].to_numpy(), np.column_stack([Xb, d["L1"]]))
    bn, tn, r2n = _ols_hac(d["y"].to_numpy(), np.column_stack([Xb, d["L1n"]]))
    _, t0, r20 = _ols_hac(d["y"].to_numpy(), np.column_stack([one, d["L1"]]))
    _, tr, _ = _ols_hac(d["fret"].to_numpy(), np.column_stack([Xb, d["L1"]]))
    out["prediccion_varianza_21"] = {
        "observaciones": int(len(d)), "R2_solo_varianza_pasada": round(r2b, 4),
        "R2_con_L1": round(r2, 4), "t_L1_controlando_varianza": round(float(t[-1]), 2),
        "t_L1_normalizada_controlando_varianza": round(float(tn[-1]), 2),
        "R2_solo_L1": round(r20, 4), "t_L1_sola": round(float(t0[-1]), 2),
        "t_L1_sobre_retorno_futuro": round(float(tr[-1]), 2)}
    # antes de los tres maximos previos a crisis: percentil frente a los 3 anos anteriores
    pk = {}
    for name, day in (("2000-03-10", "2000-03-10"), ("2008-09-12", "2008-09-12"), ("2020-02-19", "2020-02-19")):
        i = s.index.searchsorted(pd.Timestamp(day))
        ref, win = s.iloc[max(0, i - 776):i - 20], s.iloc[i - 20:i]
        pk[name] = {"percentil_L1": round(float((ref["L1"] < win["L1"].median()).mean()), 3),
                    "percentil_varianza": round(float((ref["var"] < win["var"].median()).mean()), 3)}
    out["veinte_sesiones_antes_del_maximo"] = pk
    return out


# ============================================================ C3: Ricci
def block_ricci(md: MarketData, dv: pd.DataFrame, n: int = 100, w: int = 22, step: int = 5,
                thr: float = 0.75, alpha: float = 0.5) -> dict:
    import networkx as nx
    import ot
    px = md.prices
    full = px.columns[px.notna().all(axis=0)]
    top = dv.loc[dv.index.year == 2018, full].median().nlargest(n).index
    n = len(top)
    lr = np.log(px[top]).diff().dropna().to_numpy(float)
    dates = px.index[1:]
    rec = []
    for t in range(w, len(lr) + 1, step):
        c = np.corrcoef(lr[t - w:t], rowvar=False)
        dist = np.sqrt(np.clip(2 * (1 - c), 0, None))
        iu = np.triu_indices(n, 1)
        G = nx.Graph()
        G.add_nodes_from(range(n))
        full_g = nx.Graph()
        full_g.add_weighted_edges_from((int(i), int(j), float(dist[i, j])) for i, j in zip(*iu))
        G.add_weighted_edges_from(nx.minimum_spanning_tree(full_g).edges(data="weight"))
        for i, j in zip(*iu):
            if c[i, j] >= thr:
                G.add_edge(int(i), int(j), weight=float(dist[i, j]))
        sp = dict(nx.all_pairs_dijkstra_path_length(G, weight="weight"))
        kap = []
        for x, y in G.edges():
            nx_, ny_ = list(G[x]), list(G[y])
            sx, sy = [x] + nx_, [y] + ny_
            a = np.r_[alpha, np.full(len(nx_), (1 - alpha) / len(nx_))]
            b = np.r_[alpha, np.full(len(ny_), (1 - alpha) / len(ny_))]
            M = np.array([[sp[u][v] for v in sy] for u in sx])
            kap.append(1 - ot.emd2(a, b, M) / sp[x][y])
        rec.append({"date": dates[t - 1], "orc": float(np.mean(kap)), "mean_corr": float(c[iu].mean()),
                    "edges": G.number_of_edges(), "vol": float(lr[t - w:t].mean(axis=1).std())})
    r = pd.DataFrame(rec).set_index("date")
    sr = lambda a, b: round(float(stats.spearmanr(r[a], r[b]).statistic), 3)
    return {"acciones": n, "ventanas": int(len(r)), "spearman_orc_vs_correlacion_media": sr("orc", "mean_corr"),
            "spearman_orc_vs_numero_de_aristas": sr("orc", "edges"),
            "spearman_orc_vs_volatilidad": sr("orc", "vol"),
            "spearman_correlacion_media_vs_volatilidad": sr("mean_corr", "vol")}


def holm(pvals: dict) -> dict:
    items = sorted(((k, v) for k, v in pvals.items() if v is not None and np.isfinite(v)), key=lambda kv: kv[1])
    out, m, prev = {}, len(items), 0.0
    for i, (k, p) in enumerate(items):
        adj = max(prev, min(1.0, (m - i) * p))
        out[k], prev = round(adj, 4), adj
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Auditoria de modelos, agentes y vias geometricas sobre datos reales")
    ap.add_argument("--stocks-panel", required=True)
    ap.add_argument("--indices-dir", default=None)
    ap.add_argument("--tickers", type=int, default=1000)
    ap.add_argument("--oos-months", type=int, default=85)
    ap.add_argument("--gbdt-months", type=int, default=48)
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--skip", default="", help="bloques a omitir separados por coma: a4,b,c1,c2,c3")
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    skip = {s.strip() for s in a.skip.split(",") if s.strip()}
    t0 = time.time()
    out: dict = {}

    def say(k):
        print(f"[{time.time() - t0:6.0f}s] {k}", flush=True)
        if a.json and out:                      # guarda lo ya medido antes de cada bloque
            Path(a.json).write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")

    md, nom, dv = load_panel(a.stocks_panel, a.tickers)
    rc = RankConfig(min_price=1.0, min_dollar_volume=1e6, oos_months=a.oos_months)
    df = _prepare(md, rc)
    out["panel"] = {"tickers": int(md.prices.shape[1]), "sesiones": int(md.prices.shape[0]),
                    "desde": str(md.prices.index[0].date()), "hasta": str(md.prices.index[-1].date()),
                    "aviso": "universo de supervivientes"}
    say("A/D: agentes, redundancia, acuerdo, agregadores, coherencia")
    out["A_D"], sc = block_a_d(md, nom, df, rc)
    print(json.dumps(out["A_D"], indent=1)[:4500], flush=True)
    if "a4" not in skip:
        say("A4: ridge frente a gradient boosting")
        out["A4_modelo"] = block_a4(df, rc, sc.set_index(["date", "symbol"])["fwd_lag"], a.gbdt_months)
        print(json.dumps(out["A4_modelo"], indent=1), flush=True)
    if "b" not in skip:
        say("B: features geometricas por accion")
        out["B_agente_geo"] = block_b(df, rc, geo_features(md.prices), sc, out["A_D"]["agentes"])
        print(json.dumps(out["B_agente_geo"], indent=1), flush=True)
    if "c1" not in skip:
        say("C1: monitores frente a episodios")
        p = monitor_pvalues(df, md.prices)
        out["C1_monitores"] = {"fechas": int(len(p)), "desde": str(p.index[0].date()),
                               "resultado": evaluate_monitors(p),
                               "potencia_forma_sintetica": synthetic_shape_power(a.seeds)}
        print(json.dumps(out["C1_monitores"], indent=1), flush=True)
    if "c2" not in skip and a.indices_dir:
        say("C2: TDA")
        out["C2_tda"] = block_tda(a.indices_dir)
        print(json.dumps(out["C2_tda"], indent=1), flush=True)
    if "c3" not in skip:
        say("C3: curvatura de Ollivier-Ricci")
        out["C3_ricci"] = block_ricci(md, dv)
        print(json.dumps(out["C3_ricci"], indent=1), flush=True)

    fam = {"D1 acuerdo mejora": out["A_D"]["D1_acuerdo"]["sin_retraso"]["p_no_mejora"],
           "D2 agregador": out["A_D"]["D2_agregadores"]["sin_retraso"]["p_ninguna_bate_a_la_media"],
           "D3 coherencia": out["A_D"]["D3_coherencia"]["sin_retraso"]["p_unilateral_positiva"]}
    if "A4_modelo" in out:
        fam["A4 modelo"] = out["A4_modelo"]["sin_retraso"]["p_ninguno_bate_a_ridge"]
    if "B_agente_geo" in out:
        fam["B agente geometrico"] = out["B_agente_geo"]["p_ninguna_variante_mejora"]["sin_retraso"]
    if "C2_tda" in out:
        tt = abs(out["C2_tda"]["prediccion_varianza_21"]["t_L1_controlando_varianza"])
        fam["C2 TDA"] = float(2 * (1 - stats.norm.cdf(tt)))
    out["familias_p"] = {k: round(float(v), 4) for k, v in fam.items()}
    out["familias_p_holm"] = holm(fam)
    out["segundos"] = round(time.time() - t0, 1)
    print("familias:", out["familias_p"], "\nHolm:", out["familias_p_holm"], flush=True)
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
        print("resultados en", a.json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
