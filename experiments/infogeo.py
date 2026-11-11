#!/usr/bin/env python3
"""Geometria de la informacion: ¿donde aporta y donde no? Medido, no supuesto.

Tres preguntas, cada una con su criterio fijado ANTES de ejecutar (este
docstring es el pre-registro; no se edita despues de ver resultados):

E1  MONITOR DE DERIVA (`stats/drift.py`)
    a) Tamano: con datos estacionarios, la fraccion de fechas que dispara debe
       estar cerca de alpha (1%).
    b) Potencia: con una ruptura inyectada de magnitud conocida, fraccion que
       dispara a 3, 5, 10 y 21 sesiones de la ruptura. La fila vol_x = 1,0 es
       la linea base sin ruptura sobre las mismas semillas.
    c) Datos reales (si se pasa --stocks-panel): que episodios marca y, como
       unica hipotesis de valor, H: el rank IC del modelo base es MENOR en las
       fechas marcadas que en las no marcadas. Se informa la diferencia con su
       intervalo bootstrap por bloques; no se ajusta nada para que salga.

E2  DISTANCIA DEL MOTOR DE ANALOGOS (`analogy.py`)
    Unidad de evaluacion: CRPS de la distribucion empirica de retornos a 21
    sesiones de los 50 vecinos, frente al retorno realizado del caso consultado
    (regla de puntuacion propia; menor es mejor). Secundaria: 5 sesiones.
    T1 (¿mide algo el instrumento?)  H0: la distancia euclidea no mejora a
        elegir 50 casos admisibles al azar entre los movimientos del mismo signo.
    T2 (la pregunta)  H0: NINGUNA de las dos metricas de Fisher mejora a la
        euclidea. SPA de Hansen sobre las dos diferencias, que corrige por
        haber probado dos.
    Regla de decision: el valor por defecto de `AnalogyConfig.metric` solo
    cambia si T2 se rechaza al 5% CON DATOS REALES y el signo de la mejora se
    mantiene en las dos mitades de la muestra. Los datos sinteticos solo
    validan el instrumento.

E3  COMBINACION DE AGENTES: media lineal frente a media en logits
    (m-geodesica frente a e-geodesica). H0: la media en logits no mejora el
    rank IC de la equiponderacion lineal. Solo se informa; no hay candidato a
    produccion salvo rechazo con datos reales.

Datos reales
------------
`--stocks-panel` espera un parquet largo con columnas date, ticker, adjclose,
close_nominal, dollar_volume. El panel usado en la ejecucion documentada son
5.550 acciones ordinarias cotizadas HOY en EE. UU. (Yahoo, 2018-2026): es un
universo de SUPERVIVIENTES. Sirve para comparar dos reglas entre si sobre los
mismos casos; no sirve para afirmar rentabilidad de nada.

Uso:
    python3 experiments/infogeo.py [--stocks-panel ruta.parquet] [--json salida.json]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pandas as pd

from swarmtrader.agents.specialists import run_swarm
from swarmtrader.analogy import (DESCRIPTOR_COLS, GAUSS_COLS, METRICS, analogue_distances,
                                 build_descriptors)
from swarmtrader.data.loaders import MarketData, load_synthetic
from swarmtrader.ensemble.stacking import per_date_ic
from swarmtrader.rank import RankConfig, _prepare, walk_forward_oos
from swarmtrader.stats.drift import DriftConfig, compute_states, drift_at
from swarmtrader.stats.spa import spa_test

HORIZON, EMBARGO, TRAIN_SESSIONS = 21, 5, 756
K_NEIGHBOURS, MAX_H, MIN_RUN_UP = 50, 32, 0.15


# ============================================================ utilidades
def vol_break(md: MarketData, tail: int, factor: float) -> MarketData:
    lr = np.log(md.prices).diff().fillna(0.0)
    lr.iloc[-tail:] = lr.iloc[-tail:] * factor
    return MarketData(np.exp(lr.cumsum()) * md.prices.iloc[0], md.volumes)


def ref_window(dates: pd.DatetimeIndex, t: int) -> pd.DatetimeIndex:
    """Ventana de referencia de la fecha t: 3 anos que acaban un horizonte+embargo antes."""
    hi = t - HORIZON - EMBARGO
    return dates[max(0, hi - TRAIN_SESSIONS):max(0, hi)]


def block_bootstrap_mean_diff(a: np.ndarray, b: np.ndarray, block: int = 6, n: int = 4000,
                              seed: int = 0) -> tuple[float, float, float]:
    """Diferencia de medias a - b con IC 95% remuestreando bloques de cada serie."""
    rng = np.random.default_rng(seed)

    def resample(x):
        nb = int(np.ceil(len(x) / block))
        starts = rng.integers(0, max(len(x) - block + 1, 1), nb)
        return np.concatenate([x[s:s + block] for s in starts])[:len(x)]
    d = [resample(a).mean() - resample(b).mean() for _ in range(n)]
    return float(a.mean() - b.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def crps_ensemble(ens: np.ndarray, y: float) -> float:
    ens = np.sort(ens)
    m = len(ens)
    spread = (2 * np.arange(1, m + 1) - m - 1) @ ens / (m * m)     # = 0.5 * E|X - X'|
    return float(np.abs(ens - y).mean() - spread)


# ============================================================ E1 deriva
def e1_size(seeds: int, cfg: DriftConfig) -> dict:
    fired = total = 0
    for s in range(seeds):
        md = load_synthetic(n_symbols=80, n_days=1100, seed=100 + s)
        rc = RankConfig()
        st = compute_states(_prepare(md, rc), md.prices, cfg)
        for t in range(560, len(st.dates), 5):
            r = drift_at(st, ref_window(st.dates, t), st.dates[t], cfg)
            if r.evaluated:
                total += 1
                fired += r.out_of_support
    return {"alpha": cfg.alpha, "evaluadas": total, "disparos": int(fired),
            "tasa": fired / max(total, 1)}


def e1_power_dispersion(seeds: int, cfg: DriftConfig) -> list[dict]:
    rows = []
    for factor in (1.0, 1.25, 1.5, 2.0, 3.0):      # 1.0 = sin ruptura (linea base)
        for lag in (3, 5, 10, 21):
            hits = 0
            for s in range(seeds):
                md = vol_break(load_synthetic(n_symbols=80, n_days=900, seed=200 + s), lag, factor)
                st = compute_states(_prepare(md, RankConfig()), md.prices, cfg)
                t = len(st.dates) - 1
                hits += drift_at(st, ref_window(st.dates, t), st.dates[t], cfg).out_of_support
            rows.append({"vol_x": factor, "sesiones_tras_ruptura": lag, "deteccion": hits / seeds})
    return rows


def _factor_panel(n_dates, n_names, rho, tail, seed):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-04", periods=n_dates)
    syms = [f"S{i:04d}" for i in range(n_names)]
    z = rng.normal(size=(n_dates, n_names, 6))
    if tail:
        z[-tail:, :, 1] = rho * z[-tail:, :, 0] + np.sqrt(1 - rho ** 2) * z[-tail:, :, 1]
    idx = pd.MultiIndex.from_product([dates, syms], names=["date", "symbol"])
    cols = [f"f{i}" for i in range(6)]
    df = pd.DataFrame(z.reshape(-1, 6), index=idx, columns=cols)
    df.attrs["factor_cols"] = cols
    px = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.02, (n_dates, n_names)), axis=0)),
                      index=dates, columns=syms)
    return df, px


def e1_power_dependence(seeds: int, cfg: DriftConfig) -> list[dict]:
    rows = []
    for n_names in (50, 300):
        for rho in (0.1, 0.2, 0.4):
            for lag in (3, 10):
                hits = 0
                for s in range(seeds):
                    df, px = _factor_panel(500, n_names, rho, lag, 300 + s)
                    st = compute_states(df, px, cfg)
                    t = len(st.dates) - 1
                    hits += drift_at(st, ref_window(st.dates, t), st.dates[t], cfg).out_of_support
                rows.append({"nombres": n_names, "rho_inyectada": rho,
                             "sesiones_tras_ruptura": lag, "deteccion": hits / seeds})
    return rows


def e1_real(md: MarketData, cfg: DriftConfig) -> dict:
    rc = RankConfig(min_price=1.0, min_dollar_volume=1e6, oos_months=200)
    df = _prepare(md, rc)
    st = compute_states(df, md.prices, cfg)
    recs = []
    for t in range(400, len(st.dates)):
        r = drift_at(st, ref_window(st.dates, t), st.dates[t], cfg)
        if r.evaluated:
            recs.append({"date": st.dates[t], "fired": r.out_of_support,
                         **{f"p_{k}": v["p_value"] for k, v in r.channels.items()},
                         **{f"d_{k}": v["distance"] for k, v in r.channels.items()}})
    tl = pd.DataFrame(recs).set_index("date")
    # episodios: rachas de fechas marcadas separadas por menos de 10 sesiones
    episodes, start, prev = [], None, None
    pos = {d: i for i, d in enumerate(tl.index)}
    for d in tl.index[tl["fired"]]:
        if start is None or pos[d] - pos[prev] > 10:
            if start is not None:
                episodes.append((start, prev))
            start = d
        prev = d
    if start is not None:
        episodes.append((start, prev))
    ep = []
    for a, b in episodes:
        seg = tl.loc[a:b]
        seg = seg[seg["fired"]]
        ch = [c[2:] for c in ("p_dispersion", "p_dependencia")
              if c in seg and (seg[c] < cfg.alpha / 2).any()]
        ep.append({"desde": str(a.date()), "hasta": str(b.date()),
                   "sesiones_marcadas": int(len(seg)), "canales": ch})
    by_year = tl["fired"].groupby(tl.index.year).mean().round(4).to_dict()

    # hipotesis de valor: IC del modelo base en fechas marcadas frente a no marcadas
    oos = walk_forward_oos(df, rc)
    ic = per_date_ic(oos, "score")
    flag = tl["fired"].reindex(ic.index)
    ok = flag.notna()
    ic, flag = ic[ok], flag[ok].astype(bool)
    val = {"fechas_oos": int(len(ic)), "marcadas": int(flag.sum()),
           "ic_no_marcadas": float(ic[~flag].mean()) if (~flag).any() else None,
           "ic_marcadas": float(ic[flag].mean()) if flag.any() else None}
    if flag.sum() >= 5 and (~flag).sum() >= 5:
        d, lo, hi = block_bootstrap_mean_diff(ic[flag].to_numpy(), ic[~flag].to_numpy(), block=2)
        val.update({"diferencia": d, "ic95": [lo, hi]})
    return {"fechas_evaluadas": int(len(tl)), "tasa_disparo": float(tl["fired"].mean()),
            "tasa_por_anio": {int(k): float(v) for k, v in by_year.items()},
            "episodios": ep, "valor": val,
            "nombres_por_fecha_mediana": int(np.median(st.n_names[st.n_names > 0]))}


# ============================================================ E2 analogos
def e2_tournament(px: pd.DataFrame, vol: pd.DataFrame, eligible: pd.DataFrame | None,
                  step: int, per_date: int, burn: int, seed: int = 0) -> dict:
    desc = build_descriptors(px, vol)
    need = list(DESCRIPTOR_COLS) + list(GAUSS_COLS)
    fwd = {h: (px.shift(-h) / px - 1.0).stack(future_stack=True) for h in (5, 21)}
    for h, s in fwd.items():
        s.index.names = ["date", "symbol"]
        desc[f"fwd_{h}"] = s.reindex(desc.index).to_numpy()
    pos = pd.Series(np.arange(len(px.index)), index=px.index)
    desc["_pos"] = pos.reindex(desc.index.get_level_values("date")).to_numpy()
    if eligible is not None:
        e = eligible.stack(future_stack=True)
        e.index.names = ["date", "symbol"]
        desc = desc[e.reindex(desc.index).fillna(False).to_numpy(bool)]
    pool = desc.dropna(subset=need + ["fwd_5", "fwd_21"]).sort_values("_pos", kind="stable")

    P = pool[list(DESCRIPTOR_COLS)].to_numpy(np.float32)
    G = pool[list(GAUSS_COLS)].to_numpy(float)
    F = {h: pool[f"fwd_{h}"].to_numpy(float) for h in (5, 21)}
    ppos = pool["_pos"].to_numpy(int)
    ru = pool["_run_up_5_raw"].to_numpy(float)
    ev_up = np.flatnonzero(ru >= MIN_RUN_UP)
    ev_dn = np.flatnonzero(ru <= -MIN_RUN_UP)
    w = np.full(len(DESCRIPTOR_COLS), 1.0 / len(DESCRIPTOR_COLS), dtype=np.float32)

    rng = np.random.default_rng(seed)
    last_q = len(px.index) - 1 - HORIZON
    q_idx = []
    for t in range(burn, last_q, step):
        cand = np.flatnonzero((ppos == t) & (np.abs(ru) >= MIN_RUN_UP))
        if len(cand) > per_date:
            cand = rng.choice(cand, per_date, replace=False)
        q_idx.extend(cand.tolist())

    names = ["azar"] + list(METRICS)
    rows = []
    for qi in q_idx:
        end = int(np.searchsorted(ppos, ppos[qi] - MAX_H - EMBARGO, side="right"))
        ev = ev_up if ru[qi] > 0 else ev_dn
        ev = ev[:np.searchsorted(ev, end)]
        if end < 500 or len(ev) < K_NEIGHBOURS:
            continue
        sel = {"azar": rng.choice(ev, K_NEIGHBOURS, replace=False)}
        for m in METRICS:
            d = analogue_distances(P[qi], P[:end], w, m, query_gauss=G[qi], pool_gauss=G[:end])
            sel[m] = np.argpartition(d, K_NEIGHBOURS - 1)[:K_NEIGHBOURS]
        row = {"pos": ppos[qi]}
        for h in (5, 21):
            for n in names:
                row[f"{n}_{h}"] = crps_ensemble(F[h][sel[n]], F[h][qi])
        rows.append(row)
    if not rows:
        return {"consultas": 0}
    res = pd.DataFrame(rows)
    by_date = res.groupby("pos").mean()
    out = {"consultas": int(len(res)), "fechas": int(len(by_date)), "filas_del_fondo": int(len(pool))}
    for h in (21, 5):
        crps = {n: float(res[f"{n}_{h}"].mean()) for n in names}
        t1 = spa_test((by_date[f"azar_{h}"] - by_date[f"euclid_{h}"]).to_numpy()[:, None],
                      n_boot=2000, seed=1)
        D = np.column_stack([(by_date[f"euclid_{h}"] - by_date[f"{m}_{h}"]).to_numpy()
                             for m in ("fisher_rank", "fisher_gauss")])
        t2 = spa_test(D, n_boot=2000, seed=2)
        half = len(by_date) // 2
        out[f"h{h}"] = {
            "crps": crps,
            "mejora_vs_azar": {n: 1 - crps[n] / crps["azar"] for n in METRICS},
            "mejora_vs_euclid": {m: 1 - crps[m] / crps["euclid"] for m in ("fisher_rank", "fisher_gauss")},
            "T1_p_euclid_no_mejora_azar": float(t1.p_value),
            "T2_p_ninguna_fisher_mejora_euclid": float(t2.p_value),
            "mejora_media_por_mitad": {
                m: [float(D[:half, i].mean()), float(D[half:, i].mean())]
                for i, m in enumerate(("fisher_rank", "fisher_gauss"))},
        }
    return out


def synthetic_events(n_symbols=120, n_days=1300, seed=0) -> MarketData:
    """Mercado con estructura de eventos CONOCIDA: tras un salto, el precio
    continua o revierte segun el tamano del salto medido en sigmas del valor."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-02", periods=n_days)
    sig = rng.uniform(0.01, 0.05, n_symbols)
    r = rng.normal(0, 1, (n_days, n_symbols)) * sig
    for j in range(n_symbols):
        for t in rng.choice(np.arange(30, n_days - 30), n_days // 60, replace=False):
            z = rng.uniform(3, 12) * rng.choice([-1, 1])
            r[t, j] += z * sig[j]
            drift = 0.35 * (abs(z) - 7.5) * np.sign(z) * sig[j]      # >7,5 sigmas continua
            r[t + 1:t + 22, j] += drift / np.sqrt(21) * 1.5
    px = pd.DataFrame(50 * np.exp(np.cumsum(r, axis=0)), index=dates,
                      columns=[f"E{i:03d}" for i in range(n_symbols)])
    v = pd.DataFrame(1e6 * np.exp(rng.normal(0, 0.3, r.shape)) * (1 + 20 * np.abs(r) / sig),
                     index=dates, columns=px.columns)
    return MarketData(px, v)


# ============================================================ E3 combinacion
def e3_pool(md: MarketData, rc: RankConfig) -> dict:
    df = _prepare(md, rc)
    panel = run_swarm(df, rc, md=md, allow_deep=False)
    if panel.empty or len(panel.agents) < 2:
        return {"agentes": 0}
    sc = panel.scores.copy()
    A = sc[panel.agents].to_numpy(float)
    u = np.clip((A + 1.0) / 2.0, 0.005, 0.995)
    sc["lineal"] = A.mean(axis=1)
    sc["logit"] = np.log(u / (1 - u)).mean(axis=1)
    il, ig = per_date_ic(sc, "lineal"), per_date_ic(sc, "logit")
    idx = il.index.intersection(ig.index)
    d = (ig.reindex(idx) - il.reindex(idx)).to_numpy()
    p = spa_test(d[:, None], n_boot=2000, seed=3).p_value if len(d) >= 8 else float("nan")
    return {"agentes": len(panel.agents), "fechas": int(len(idx)),
            "ic_lineal": float(il.reindex(idx).mean()), "ic_logit": float(ig.reindex(idx).mean()),
            "diferencia": float(d.mean()), "p_logit_no_mejora_lineal": float(p)}


# ============================================================ datos reales
def load_real(path: str, n_tickers: int, seed: int = 20261111):
    p = pd.read_parquet(path, columns=["date", "ticker", "adjclose", "close_nominal", "dollar_volume"])
    p = p[p["ticker"].notna() & (p["ticker"] != "SPY")]
    tickers = np.array(sorted(p["ticker"].astype(str).unique()))
    keep = np.random.default_rng(seed).choice(tickers, min(n_tickers, len(tickers)), replace=False)
    p = p[p["ticker"].astype(str).isin(set(keep))]
    p["ticker"] = p["ticker"].astype(str)
    px = p.pivot(index="date", columns="ticker", values="adjclose").sort_index()
    nom = p.pivot(index="date", columns="ticker", values="close_nominal").reindex_like(px)
    dv = p.pivot(index="date", columns="ticker", values="dollar_volume").reindex_like(px)
    px.index = pd.to_datetime(px.index)
    nom.index = dv.index = px.index
    vol = (dv / px).fillna(0.0)                       # px * vol reproduce el volumen en dolares
    eligible = (nom >= 1.0) & (dv.rolling(21, min_periods=10).median() >= 1e6)
    return MarketData(px, vol), eligible


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--stocks-panel", default=None)
    ap.add_argument("--tickers", type=int, default=1000, help="muestra aleatoria del panel real")
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--json", default=None)
    ap.add_argument("--skip-synthetic", action="store_true", help="solo la parte de datos reales")
    ap.add_argument("--skip-real-swarm", action="store_true",
                    help="omite E3 sobre datos reales (es la parte lenta)")
    a = ap.parse_args()
    cfg = DriftConfig()
    out: dict = {"config": {"alpha": cfg.alpha, "smooth": cfg.smooth, "k": K_NEIGHBOURS,
                            "min_run_up": MIN_RUN_UP, "seeds": a.seeds}}
    t0 = time.time()

    if not a.skip_synthetic:
        print("E1a tamano del monitor sobre datos estacionarios ...", flush=True)
        out["E1a_tamano"] = e1_size(a.seeds, cfg)
        print("   ", out["E1a_tamano"], flush=True)
        print("E1b potencia: ruptura de dispersion ...", flush=True)
        out["E1b_potencia_dispersion"] = e1_power_dispersion(a.seeds, cfg)
        print(pd.DataFrame(out["E1b_potencia_dispersion"]).pivot(
            index="vol_x", columns="sesiones_tras_ruptura", values="deteccion").to_string(), flush=True)
        print("E1b potencia: ruptura de dependencia ...", flush=True)
        out["E1b_potencia_dependencia"] = e1_power_dependence(a.seeds, cfg)
        print(pd.DataFrame(out["E1b_potencia_dependencia"]).to_string(index=False), flush=True)

        print("E2 torneo de distancias: ruido (no debe haber nada) ...", flush=True)
        md = load_synthetic(n_symbols=120, n_days=1300, seed=11)
        # el ruido sintetico rara vez mueve un 15% en 5 sesiones: se le sube la volatilidad
        md = vol_break(md, len(md.prices), 2.5)
        out["E2_ruido"] = e2_tournament(md.prices, md.volumes, None, step=5, per_date=6, burn=400)
        print("   ", json.dumps(out["E2_ruido"].get("h21", out["E2_ruido"]), indent=1)[:900], flush=True)
        print("E2 torneo de distancias: eventos con estructura conocida ...", flush=True)
        md = synthetic_events(seed=5)
        out["E2_eventos"] = e2_tournament(md.prices, md.volumes, None, step=5, per_date=6, burn=400)
        print("   ", json.dumps(out["E2_eventos"].get("h21", out["E2_eventos"]), indent=1)[:900], flush=True)

        print("E3 combinacion lineal frente a logit (sintetico con y sin senal) ...", flush=True)
        rc = RankConfig()
        out["E3_sin_senal"] = e3_pool(load_synthetic(n_symbols=60, n_days=1000, seed=21), rc)
        out["E3_con_senal"] = e3_pool(load_synthetic(n_symbols=60, n_days=1000, momentum_strength=0.25,
                                                     seed=21), rc)
        print("   sin senal:", out["E3_sin_senal"], "\n   con senal:", out["E3_con_senal"], flush=True)

    if a.stocks_panel:
        print(f"DATOS REALES: {a.tickers} tickers al azar de {a.stocks_panel}", flush=True)
        md, eligible = load_real(a.stocks_panel, a.tickers)
        out["real_panel"] = {"tickers": int(md.prices.shape[1]), "sesiones": int(md.prices.shape[0]),
                             "desde": str(md.prices.index[0].date()),
                             "hasta": str(md.prices.index[-1].date()),
                             "aviso": "universo de supervivientes (cotizados hoy)"}
        print("E1c monitor sobre datos reales ...", flush=True)
        out["E1c_real"] = e1_real(md, cfg)
        print(json.dumps(out["E1c_real"], indent=1, default=str)[:2500], flush=True)
        print("E2 torneo de distancias sobre datos reales ...", flush=True)
        out["E2_real"] = e2_tournament(md.prices, md.volumes, eligible, step=5, per_date=5, burn=504)
        print(json.dumps(out["E2_real"], indent=1)[:2500], flush=True)
        if not a.skip_real_swarm:
            print("E3 combinacion sobre datos reales ...", flush=True)
            out["E3_real"] = e3_pool(md, RankConfig(min_price=1.0, min_dollar_volume=1e6,
                                                    oos_months=60))
            print("   ", out["E3_real"], flush=True)

    out["segundos"] = round(time.time() - t0, 1)
    if a.json:
        Path(a.json).write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
        print(f"resultados en {a.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
