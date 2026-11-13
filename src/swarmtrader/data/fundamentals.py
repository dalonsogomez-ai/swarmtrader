"""Fundamentales point-in-time desde los datos XBRL de la SEC (gratuitos).

Por que esta fuente
-------------------
La SEC publica cada noche `companyfacts.zip`: todos los hechos XBRL de todas las
empresas que presentan informes (10-K, 10-Q, 20-F...), con la FECHA DE
PRESENTACION de cada dato (`filed`). Esa fecha es lo que hace posible el
point-in-time: un dato solo se usa a partir del dia en que fue publico. Incluye
empresas que despues dejaron de cotizar (se identifican por CIK, no por ticker),
asi que la fuente no fuerza el sesgo de supervivencia. Cobertura XBRL: desde
2009 (grandes) y 2011 (pequenas). Coste: 0.

Lo que la fuente NO resuelve y hay que tratar aqui:
  * Mapeo CIK -> ticker en el tiempo. La SEC solo publica el mapeo ACTUAL
    (`company_tickers_exchange.json`). Con un panel de precios de supervivientes
    indexado por ticker actual es coherente; con datos con deslistados hace falta
    un mapeo historico (Sharadar `tickers`, CRSP).
  * Etiquetas heterogeneas: la misma magnitud aparece con varias etiquetas
    US-GAAP (p. ej. `Revenues` o `RevenueFromContractWithCustomer...`). Se usa una
    lista de prioridad por concepto.
  * Reexpresiones: se usa el valor TAL COMO SE PUBLICO POR PRIMERA VEZ para cada
    periodo (primera presentacion). Es la convencion "as first reported": nunca
    usa un dato que no existia en la fecha, aunque ignore correcciones posteriores.
  * Duracion de los flujos: un 10-Q del segundo trimestre trae a la vez el dato
    de 3 meses y el acumulado de 6 con la misma fecha de cierre. Se clasifica por
    duracion (`end - start`) y solo se usan trimestres (80-100 dias) y anos
    (340-390 dias).

Salida
------
`pit_snapshots` produce una fila por (empresa, fecha de presentacion) con el
estado conocido en esa fecha. `signal_tables` la convierte en dos tablas con el
contrato de `data/signals.py` (columna `date` = fecha en que el dato fue publico):
  * fundamentals: book_equity, total_assets, gross_profit, net_income, cfo,
    shares, fscore, asset_growth, net_issuance.
  * earnings: sue (sorpresa estandarizada por paseo aleatorio estacional,
    Bernard-Thomas 1989; sin estimaciones de analistas, que no son gratuitas).

El dia de la presentacion del 10-Q suele ser posterior al anuncio de resultados
(8-K/nota de prensa, 2-4 semanas antes): la deriva medida desde aqui es la que
queda DESPUES de la presentacion, una cota inferior de la PEAD clasica.
"""

from __future__ import annotations

import io
import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = ["CONCEPTS", "read_companyfacts_zip", "read_companyfacts_parquet",
           "load_ticker_map", "first_reported", "pit_snapshots", "piotroski",
           "quarterly_sue", "signal_tables"]

# concepto -> etiquetas US-GAAP en orden de prioridad
CONCEPTS: dict[str, tuple[str, ...]] = {
    # saldos (instantes)
    "equity": ("StockholdersEquity",
               "StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest"),
    "assets": ("Assets",),
    "liabilities": ("Liabilities",),
    "assets_cur": ("AssetsCurrent",),
    "liab_cur": ("LiabilitiesCurrent",),
    "lt_debt": ("LongTermDebtNoncurrent", "LongTermDebt"),
    "shares_bs": ("CommonStockSharesOutstanding",),
    # flujos (duraciones)
    "revenue": ("Revenues", "RevenueFromContractWithCustomerExcludingAssessedTax",
                "SalesRevenueNet", "RevenueFromContractWithCustomerIncludingAssessedTax"),
    "gross_profit": ("GrossProfit",),
    "cost_rev": ("CostOfRevenue", "CostOfGoodsAndServicesSold", "CostOfGoodsSold"),
    "net_income": ("NetIncomeLoss", "ProfitLoss"),
    "cfo": ("NetCashProvidedByUsedInOperatingActivities",
            "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"),
    "eps": ("EarningsPerShareDiluted", "EarningsPerShareBasic"),
}
INSTANT = ("equity", "assets", "liabilities", "assets_cur", "liab_cur", "lt_debt", "shares_bs")
DEI_SHARES = "EntityCommonStockSharesOutstanding"
_FORMS = re.compile(r"^(10-K|10-Q|10-KT|10-QT|20-F|40-F)(/A)?$")
_TAG2 = {t: (c, i) for c, tags in CONCEPTS.items() for i, t in enumerate(tags)}
_COLS = ["cik", "concept", "prio", "start", "end", "filed", "form", "accn", "val"]


def _cik_from_folder(s: pd.Series) -> pd.Series:
    return s.astype(str).str.extract(r"CIK(\d{1,10})", expand=False).astype("int64")


def _finish(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["form"].astype(str).str.match(_FORMS)].copy()
    for c in ("start", "end", "filed"):
        df[c] = pd.to_datetime(df[c], errors="coerce")
    df["val"] = pd.to_numeric(df["val"], errors="coerce")
    df = df.dropna(subset=["end", "filed", "val"])
    df["prio"] = df["prio"].astype("int8")
    return df[_COLS].reset_index(drop=True)


# ---------------------------------------------------------------- lectura
def read_companyfacts_zip(path: str | Path, ciks: set[int] | None = None) -> pd.DataFrame:
    """Lee el `companyfacts.zip` OFICIAL de la SEC (un JSON por CIK)."""
    rows = []
    with zipfile.ZipFile(path) as z:
        for name in z.namelist():
            m = re.search(r"CIK(\d{10})\.json$", name)
            if not m:
                continue
            cik = int(m.group(1))
            if ciks is not None and cik not in ciks:
                continue
            facts = json.load(io.TextIOWrapper(z.open(name), encoding="utf-8")).get("facts", {})
            for tag, (concept, prio) in _TAG2.items():
                node = facts.get("us-gaap", {}).get(tag)
                if not node:
                    continue
                for unit, obs in node.get("units", {}).items():
                    if unit not in ("USD", "shares", "USD/shares"):
                        continue
                    for o in obs:
                        rows.append((cik, concept, prio, o.get("start"), o["end"], o["filed"],
                                     o.get("form", ""), o.get("accn", ""), o["val"]))
            node = facts.get("dei", {}).get(DEI_SHARES)
            if node:
                for o in node.get("units", {}).get("shares", []):
                    rows.append((cik, "shares_dei", 0, None, o["end"], o["filed"],
                                 o.get("form", ""), o.get("accn", ""), o["val"]))
    return _finish(pd.DataFrame(rows, columns=_COLS))


def read_companyfacts_parquet(usgaap: str | Path, dei_shares: str | Path | None = None,
                              ciks: set[int] | None = None) -> pd.DataFrame:
    """Lee un volcado en Parquet de companyfacts con columnas item, start, end,
    filed, form, accn, val_dec, source_folder (p. ej. el espejo publico en Hugging
    Face). Es una COPIA DE TERCEROS: el codigo de produccion debe leer el zip
    oficial; esta funcion existe para poder medir sin acceso directo a la SEC.
    """
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    tags = pa.array(list(_TAG2))
    forms = pa.array([f + a for f in ("10-K", "10-Q", "10-KT", "10-QT", "20-F", "40-F")
                      for a in ("", "/A")])
    cik_set = pa.array(sorted(ciks), pa.int64()) if ciks is not None else None

    def compact(t: pa.Table, with_item: bool) -> pa.Table | None:
        # filtrar y convertir en Arrow: las cadenas en pandas multiplican la memoria
        t = t.filter(pc.is_in(t.column("form"), value_set=forms))
        if with_item:
            t = t.filter(pc.is_in(t.column("item"), value_set=tags))
        if not t.num_rows:
            return None
        ex = pc.extract_regex(t.column("source_folder").combine_chunks(), r"CIK(?P<c>\d{1,10})")
        cik = pc.cast(pc.struct_field(ex, [0]), pa.int64())
        keep = pc.is_in(cik, value_set=cik_set) if cik_set is not None else None
        cols = {"cik": cik,
                "item": t.column("item").combine_chunks() if with_item else pa.repeat(pa.scalar(DEI_SHARES),
                                                                       t.num_rows),
                "start": t.column("start").combine_chunks() if with_item else pa.nulls(t.num_rows, pa.string()),
                "end": t.column("end").combine_chunks(), "filed": t.column("filed").combine_chunks(),
                "form": t.column("form").combine_chunks(),
                "accn": t.column("accn").combine_chunks() if not with_item else pa.nulls(t.num_rows, pa.string()),
                "val": pc.cast(t.column("val_dec").combine_chunks(), pa.float64())}
        out = pa.table(cols)
        if keep is not None:
            out = out.filter(keep)
        for c in ("item", "form"):
            out = out.set_column(out.schema.get_field_index(c), c, pc.dictionary_encode(out[c]))
        return out if out.num_rows else None

    parts = []
    f = pq.ParquetFile(usgaap)
    cols = ["item", "start", "end", "filed", "form", "val_dec", "source_folder"]
    for i in range(f.metadata.num_row_groups):
        t = compact(f.read_row_group(i, columns=cols), True)
        if t is not None:
            parts.append(t)
    if dei_shares is not None:
        d = pq.read_table(dei_shares, columns=["end", "filed", "form", "accn", "val_dec",
                                               "source_folder"])
        t = compact(d, False)
        if t is not None:
            parts.append(t)
    if not parts:
        return pd.DataFrame(columns=_COLS)
    df = pa.concat_tables(parts, promote_options="permissive").to_pandas()
    item = df.pop("item").astype(str)
    df["concept"] = item.map(lambda x: _TAG2[x][0] if x in _TAG2 else "shares_dei")
    df["prio"] = item.map(lambda x: _TAG2[x][1] if x in _TAG2 else 0)
    df["form"] = df["form"].astype(str)
    return _finish(df)


def load_ticker_map(path: str | Path) -> pd.DataFrame:
    """`company_tickers_exchange.json` de la SEC (o el mismo en Parquet/CSV).

    Devuelve cik, ticker (formato Yahoo: clases con guion), exchange. Es el mapeo
    ACTUAL: valido para un panel de supervivientes indexado por ticker actual.
    """
    p = Path(path)
    if p.suffix == ".json":
        j = json.loads(p.read_text())
        t = pd.DataFrame(j["data"], columns=j["fields"])
    elif p.suffix == ".parquet":
        t = pd.read_parquet(p)
    else:
        t = pd.read_csv(p)
    t = t.rename(columns=str.lower)
    t["cik"] = t["cik"].astype("int64")
    t["ticker"] = t["ticker"].astype(str).str.upper().str.replace(".", "-", regex=False)
    t["exchange"] = t.get("exchange", pd.Series("", index=t.index)).fillna("").astype(str)
    return t[["cik", "ticker", "exchange"]].drop_duplicates("ticker")


# ---------------------------------------------------------------- periodos
def first_reported(facts: pd.DataFrame) -> pd.DataFrame:
    """Un valor por (empresa, concepto, periodo): el de la PRIMERA presentacion.

    `kind`: instant | quarter (80-100 dias) | annual (340-390) | ytd (otros).
    Las acciones de portada (`shares_dei`) se suman por presentacion: con varias
    clases de accion la SEC publica una fila por clase.
    """
    f = facts.copy()
    dei = f["concept"] == "shares_dei"
    if dei.any():
        d = (f[dei].groupby(["cik", "accn", "end", "filed", "form"], as_index=False)["val"].sum())
        d["concept"], d["prio"], d["start"] = "shares_dei", 0, pd.NaT
        f = pd.concat([f[~dei], d[_COLS]], ignore_index=True)
    dur = (f["end"] - f["start"]).dt.days
    f["kind"] = np.select(
        [f["start"].isna(), dur.between(80, 100), dur.between(340, 390)],
        ["instant", "quarter", "annual"], "ytd")
    f = f.sort_values(["cik", "concept", "kind", "start", "end", "filed", "prio"],
                      na_position="first")
    p = f.drop_duplicates(["cik", "concept", "kind", "start", "end"], keep="first")
    return p.reset_index(drop=True)


def _latest(P: pd.DataFrame, ev: pd.DataFrame, concept: str, kind: str, name: str) -> pd.DataFrame:
    """Valor del periodo MAS RECIENTE conocido en cada fecha de `ev` (date, cik)."""
    r = P[(P["concept"] == concept) & (P["kind"] == kind)][["cik", "end", "filed", "val"]]
    if r.empty:
        return ev.assign(**{name: np.nan, f"{name}_end": pd.NaT})
    r = r.sort_values(["cik", "filed", "end"])
    # un periodo antiguo presentado tarde (enmienda) no es "el mas reciente"
    r = r[r["end"] >= r.groupby("cik")["end"].cummax()]
    r = r.sort_values("filed").rename(columns={"val": name, "end": f"{name}_end"})
    out = pd.merge_asof(ev.sort_values("date"), r, left_on="date", right_on="filed",
                        by="cik", direction="backward").drop(columns="filed")
    return out


def _at(P: pd.DataFrame, ev: pd.DataFrame, concept: str, kind: str, anchor: str,
        days: int, name: str, tol: int = 45) -> pd.DataFrame:
    """Valor del periodo que termina ~`days` antes de `ev[anchor]`, si ya era publico."""
    r = P[(P["concept"] == concept) & (P["kind"] == kind)][["cik", "end", "filed", "val"]]
    ev = ev.copy()
    ev["_t"] = ev[anchor] - pd.Timedelta(days=days)
    if r.empty or ev["_t"].isna().all():
        return ev.drop(columns="_t").assign(**{name: np.nan})
    r = r.sort_values("end").rename(columns={"end": "_e", "filed": "_f", "val": name})
    ok = ev[ev["_t"].notna()].sort_values("_t")
    m = pd.merge_asof(ok, r, left_on="_t", right_on="_e", by="cik", direction="nearest",
                      tolerance=pd.Timedelta(days=tol))
    m.loc[~(m["_f"] <= m["date"]), name] = np.nan            # aun no era publico
    m = m.drop(columns=["_t", "_e", "_f"])
    rest = ev[ev["_t"].isna()].drop(columns="_t").assign(**{name: np.nan})
    return pd.concat([m, rest], ignore_index=True)


def pit_snapshots(facts: pd.DataFrame, P: pd.DataFrame | None = None) -> pd.DataFrame:
    """Estado conocido de cada empresa en cada una de sus fechas de presentacion."""
    P = first_reported(facts) if P is None else P
    ev = P[["cik", "filed"]].drop_duplicates().rename(columns={"filed": "date"})
    ev = ev.sort_values(["cik", "date"]).reset_index(drop=True)
    for c in ("equity", "assets", "liabilities", "shares_bs"):
        ev = _latest(P, ev, c, "instant", c)
    ev = _latest(P, ev, "shares_dei", "instant", "shares_dei")
    # ultimo ejercicio anual (ancla: beneficio neto anual)
    ev = _latest(P, ev, "net_income", "annual", "ni_t")
    ev = ev.rename(columns={"ni_t_end": "fy_end"})
    for c, nm in (("revenue", "rev"), ("gross_profit", "gp"), ("cost_rev", "cogs"),
                  ("cfo", "cfo")):
        ev = _at(P, ev, c, "annual", "fy_end", 0, f"{nm}_t", tol=10)
        ev = _at(P, ev, c, "annual", "fy_end", 365, f"{nm}_p", tol=20)
    ev = _at(P, ev, "net_income", "annual", "fy_end", 365, "ni_p", tol=20)
    for c in ("assets", "assets_cur", "liab_cur", "lt_debt"):
        ev = _at(P, ev, c, "instant", "fy_end", 0, f"{c}_t", tol=10)
        ev = _at(P, ev, c, "instant", "fy_end", 365, f"{c}_p", tol=20)
    ev = _at(P, ev, "assets", "instant", "fy_end", 730, "assets_pp", tol=20)
    for src in ("shares_dei", "shares_bs"):
        ev = _at(P, ev, src, "instant", "fy_end", 0, f"{src}_t", tol=100)
        ev = _at(P, ev, src, "instant", "fy_end", 365, f"{src}_p", tol=100)
    # acciones de hace un ano respecto al dato de portada mas reciente
    ev = _at(P, ev, "shares_dei", "instant", "shares_dei_end", 365, "shares_dei_1y", tol=60)
    ev["gp_t"] = ev["gp_t"].fillna(ev["rev_t"] - ev["cogs_t"])
    ev["gp_p"] = ev["gp_p"].fillna(ev["rev_p"] - ev["cogs_p"])
    ev["fscore"] = piotroski(ev)
    ev["shares"] = ev["shares_dei"].fillna(ev["shares_bs"])
    pos = lambda x: x.where(x > 0)  # noqa: E731  (denominadores positivos o NaN)
    ev["asset_growth"] = ev["assets_t"] / pos(ev["assets_p"]) - 1.0
    ev["net_issuance"] = ev["shares_dei"] / pos(ev["shares_dei_1y"]) - 1.0
    ev = ev.rename(columns={"equity": "book_equity", "assets": "total_assets",
                            "gp_t": "gross_profit", "ni_t": "net_income", "cfo_t": "cfo"})
    keep = ["date", "cik", "book_equity", "total_assets", "gross_profit", "net_income",
            "cfo", "shares", "fscore", "asset_growth", "net_issuance", "fy_end"]
    return ev[keep].sort_values(["cik", "date"]).reset_index(drop=True)


def piotroski(ev: pd.DataFrame, min_components: int = 6) -> pd.Series:
    """F-score de Piotroski (2000) con datos ANUALES, reescalado a 0-9.

    Solo se cuentan los componentes calculables; con menos de `min_components`
    devuelve NaN (neutral en el ranking) en vez de inventar ceros.
    """
    def safe(a, b):
        return a / b.where(b.abs() > 0)

    roa = safe(ev["ni_t"], ev["assets_p"])
    roa_p = safe(ev["ni_p"], ev["assets_pp"])
    lev = safe(ev["lt_debt_t"].fillna(0.0), ev["assets_t"])
    lev_p = safe(ev["lt_debt_p"].fillna(0.0), ev["assets_p"])
    cr = safe(ev["assets_cur_t"], ev["liab_cur_t"])
    cr_p = safe(ev["assets_cur_p"], ev["liab_cur_p"])
    sh = ev["shares_dei_t"].fillna(ev["shares_bs_t"])
    sh_p = ev["shares_dei_p"].fillna(ev["shares_bs_p"])
    gm = safe(ev["gp_t"], ev["rev_t"])
    gm_p = safe(ev["gp_p"], ev["rev_p"])
    turn = safe(ev["rev_t"], ev["assets_p"])
    turn_p = safe(ev["rev_p"], ev["assets_pp"])
    comps = [
        (roa > 0, roa),
        (ev["cfo_t"] > 0, ev["cfo_t"]),
        (roa > roa_p, roa + roa_p),
        (safe(ev["cfo_t"], ev["assets_p"]) > roa, roa + ev["cfo_t"]),
        (lev <= lev_p, lev + lev_p),
        (cr > cr_p, cr + cr_p),
        (sh <= sh_p * 1.001, sh + sh_p),          # sin emision neta (tolerancia 0,1 %)
        (gm > gm_p, gm + gm_p),
        (turn > turn_p, turn + turn_p),
    ]
    pts = pd.concat([c.astype(float).where(v.notna()) for c, v in comps], axis=1)
    n = pts.notna().sum(axis=1)
    return (pts.sum(axis=1) / n * 9.0).where(n >= min_components)


def quarterly_sue(P: pd.DataFrame, window: int = 8, min_hist: int = 4,
                  sd_floor: float = 0.01) -> pd.DataFrame:
    """Sorpresa estandarizada por paseo aleatorio estacional, por trimestre.

    SUE_q = (BPA_q - BPA_{q-4}) / desviacion de esas diferencias en los
    `window` trimestres anteriores (al menos `min_hist`), con un suelo de
    `sd_floor` USD por accion para que una serie casi constante no dispare el
    cociente. El cuarto trimestre se
    obtiene como anual menos acumulado de 9 meses cuando no se publica aparte.
    Fecha del evento: primera presentacion del dato (10-Q o 10-K).
    """
    e = P[P["concept"] == "eps"]
    q = e[e["kind"] == "quarter"][["cik", "start", "end", "filed", "val"]]
    a = e[e["kind"] == "annual"][["cik", "start", "end", "filed", "val"]]
    y = e[e["kind"] == "ytd"][["cik", "start", "end", "filed", "val"]]
    y = y[(y["end"] - y["start"]).dt.days.between(260, 290)]
    q4 = a.merge(y, on=["cik", "start"], suffixes=("", "_9m"))
    q4 = q4[(q4["end"] - q4["end_9m"]).dt.days.between(80, 100)]
    q4 = q4.assign(val=q4["val"] - q4["val_9m"], start=q4["end_9m"] + pd.Timedelta(days=1))
    q = pd.concat([q, q4[["cik", "start", "end", "filed", "val"]]], ignore_index=True)
    q = q.sort_values(["cik", "end", "filed"]).drop_duplicates(["cik", "end"], keep="first")
    out = []
    for cik, g in q.groupby("cik", sort=False):
        g = g.sort_values("end").reset_index(drop=True)
        ends = g["end"].to_numpy()
        vals = g["val"].to_numpy(float)
        diffs = np.full(len(g), np.nan)
        for i in range(len(g)):
            j = np.searchsorted(ends, ends[i] - np.timedelta64(365, "D"))
            for k in (j - 1, j):
                if 0 <= k < i and abs((ends[i] - ends[k]) / np.timedelta64(1, "D") - 365) <= 20:
                    diffs[i] = vals[i] - vals[k]
        sd = pd.Series(diffs).shift(1).rolling(window, min_periods=min_hist).std()
        sue = diffs / np.maximum(sd.to_numpy(), sd_floor)
        out.append(pd.DataFrame({"cik": cik, "date": g["filed"], "end": g["end"], "sue": sue}))
    if not out:
        return pd.DataFrame(columns=["cik", "date", "end", "sue"])
    return pd.concat(out, ignore_index=True).dropna(subset=["sue"])


def signal_tables(snaps: pd.DataFrame, sue: pd.DataFrame,
                  tickers: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Tablas con el contrato de `data/signals.py`: date (publico), symbol, valores.

    Una empresa con varias clases cotizadas aparece una vez por ticker.
    """
    m = tickers[["cik", "ticker"]]
    f = snaps.merge(m, on="cik").rename(columns={"ticker": "symbol"})
    f = f.drop(columns=["cik", "fy_end"]).sort_values(["symbol", "date"])
    e = sue.merge(m, on="cik").rename(columns={"ticker": "symbol"})[["date", "symbol", "sue"]]
    return f.reset_index(drop=True), e.sort_values(["symbol", "date"]).reset_index(drop=True)
