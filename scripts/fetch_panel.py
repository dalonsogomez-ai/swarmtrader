"""Descarga un panel real de acciones de EE. UU. (Yahoo) para pruebas de extremo a extremo.
Universo de SUPERVIVIENTES: solo cotizadas hoy. Sirve para probar la tuberia y
comparar reglas entre si, no para medir alfa.

Uso:
  python3 scripts/fetch_panel.py data/stocks_panel.parquet 2018-01-01 data/nasdaqtraded.txt \
      [--otc data/company_tickers_exchange.json]

Universo de cotizadas (nasdaqtraded.txt): se quitan warrants, derechos, unidades,
preferentes, deuda y fondos cotizados (ETF), que no son acciones ordinarias. Desde
la v1.6 se CONSERVAN dos grupos que el filtro anterior tiraba sin querer:
  * ADR ("American Depositary Shares"): son acciones de empresas extranjeras; el
    filtro antiguo los quitaba por la palabra "Depositary Shares", pensada para
    las preferentes depositadas (~470 simbolos, muchos de pequena capitalizacion).
  * Clases de accion con punto (BRK.B, BF.B, LEN.B...): Yahoo las escribe con guion.

--otc anade las acciones OTC de empresas que presentan informes a la SEC
(fichero company_tickers_exchange de la SEC, exchange == "OTC"). No incluye las
OTC sin informacion publica (Pink Limited/Expert/Grey), que IBKR tampoco deja comprar.

Columnas: date, ticker, adjclose, close_nominal, high, low, volume, dollar_volume,
venue_hint ("listed" | "otc"). high/low son NOMINALES (sin ajustar) y sirven para
estimar el spread (Corwin-Schultz / Abdi-Ranaldo) cuando no hay cotizaciones.
"""
import argparse
import pathlib
import re
import time

import pandas as pd
import yfinance as yf

_NOT_COMMON = (r"Warrant|\bRights?\b|\bUnits?\b|Preferred|\bNotes?\b|Debenture|Trust Pref|"
               r"Subordinated|% ")
# "Depositary Shares" solo es preferente si va con Series / Preferred / fraccion de accion
_PREF_DEPOSITARY = r"Depositary Shares.*(?:Series|Preferred|interest in a share of)"


def listed_symbols(path: str) -> list[str]:
    d = pd.read_csv(path, sep="|", keep_default_na=False, dtype=str)
    d = d[(d["Nasdaq Traded"] == "Y") & (d["ETF"] == "N") & (d["Test Issue"] == "N")]
    name = d["Security Name"]
    bad = name.str.contains(_NOT_COMMON, case=False) & ~name.str.contains(
        "American Depositary", case=False)
    bad |= name.str.contains(_PREF_DEPOSITARY, case=False)
    d = d[~bad & d["Listing Exchange"].isin(list("NQAPZ"))]
    out = []
    for s in d["Symbol"].astype(str):
        if re.fullmatch(r"[A-Z]{1,5}", s):
            out.append(s)
        elif re.fullmatch(r"[A-Z]{1,4}\.[A-Z]", s):      # clase de accion
            out.append(s.replace(".", "-"))
    return sorted(set(out))


def otc_symbols(path: str, exclude: set[str]) -> list[str]:
    p = pathlib.Path(path)
    t = pd.read_parquet(p) if p.suffix == ".parquet" else _sec_json(p)
    t = t[t["exchange"].astype(str).str.upper() == "OTC"]
    syms = {str(s).upper().replace(".", "-") for s in t["ticker"]}
    return sorted(s for s in syms if re.fullmatch(r"[A-Z]{1,5}(-[A-Z])?", s) and s not in exclude)


def _sec_json(p: pathlib.Path) -> pd.DataFrame:
    import json
    j = json.loads(p.read_text())
    return pd.DataFrame(j["data"], columns=j["fields"])


def download(syms: list[str], start: str, hint: dict[str, str], batch: int = 150) -> pd.DataFrame:
    frames = []
    fields = ("Adj Close", "Close", "High", "Low", "Volume")
    for i in range(0, len(syms), batch):
        chunk = syms[i:i + batch]
        x = None
        for attempt in range(4):
            try:
                x = yf.download(chunk, start=start, auto_adjust=False, progress=False,
                                threads=True, group_by="column")
                break
            except Exception as e:  # noqa: BLE001
                print("reintento", i, e, flush=True)
                time.sleep(20 * (attempt + 1))
        if x is None or x.empty or not all(f in x.columns.get_level_values(0) for f in fields):
            continue
        long = pd.concat({f: x[f] for f in fields}, axis=1).stack(level=1, future_stack=True)
        long.columns = ["adjclose", "close_nominal", "high", "low", "volume"]
        long.index.names = ["date", "ticker"]
        frames.append(long.dropna(subset=["adjclose"]).reset_index())
        print(f"{i + len(chunk)}/{len(syms)}", flush=True)
        time.sleep(2)
    p = pd.concat(frames, ignore_index=True)
    p["dollar_volume"] = p["close_nominal"] * p["volume"]
    p["venue_hint"] = p["ticker"].map(hint).fillna("listed")
    return p


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("start", nargs="?", default="2018-01-01")
    ap.add_argument("listing", nargs="?", default="data/nasdaqtraded.txt")
    ap.add_argument("--otc", default=None, help="company_tickers_exchange.json|parquet de la SEC")
    a = ap.parse_args()
    listed = listed_symbols(a.listing)
    otc = otc_symbols(a.otc, set(listed)) if a.otc else []
    hint = {s: "listed" for s in listed} | {s: "otc" for s in otc}
    print("simbolos cotizados", len(listed), "OTC", len(otc), flush=True)
    p = download(listed + otc, a.start, hint)
    p.to_parquet(a.out, index=False)
    print("filas", len(p), "tickers", p["ticker"].nunique(), "hasta", p["date"].max(), flush=True)


if __name__ == "__main__":
    main()
