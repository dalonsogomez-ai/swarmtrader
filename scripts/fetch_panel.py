"""Descarga un panel real de acciones de EE. UU. (Yahoo) para pruebas de extremo a extremo.
Universo de SUPERVIVIENTES: solo cotizadas hoy. Sirve para probar la tuberia, no para medir alfa.

Uso: python3 scripts/fetch_panel.py data/stocks_panel.parquet 2018-01-01 data/nasdaqtraded.txt
"""
import sys, time, pathlib
import pandas as pd, yfinance as yf

out = pathlib.Path(sys.argv[1]); start = sys.argv[2] if len(sys.argv) > 2 else "2018-01-01"
d = pd.read_csv(sys.argv[3] if len(sys.argv) > 3 else "data/nasdaqtraded.txt", sep="|", keep_default_na=False, dtype=str)
d = d[(d["Nasdaq Traded"] == "Y") & (d["ETF"] == "N") & (d["Test Issue"] == "N")]
bad = d["Security Name"].str.contains(r"Warrant|Right|Unit|Preferred|Notes|Debenture|Depositary Shares|% |Trust Pref|Subordinated", case=False)
d = d[~bad & d["Listing Exchange"].isin(list("NQAPZ"))]
syms = sorted(s for s in d["Symbol"].astype(str) if s.isalpha() and len(s) <= 5)
print("simbolos", len(syms), flush=True)
frames = []
for i in range(0, len(syms), 150):
    chunk = syms[i:i + 150]
    for attempt in range(4):
        try:
            x = yf.download(chunk, start=start, auto_adjust=False, progress=False, threads=True, group_by="column")
            break
        except Exception as e:
            print("reintento", i, e, flush=True); time.sleep(20 * (attempt + 1)); x = None
    if x is None or x.empty:
        continue
    for f in ("Adj Close", "Close", "Volume"):
        if f not in x.columns.get_level_values(0):
            break
    else:
        long = pd.concat({f: x[f] for f in ("Adj Close", "Close", "Volume")}, axis=1).stack(level=1, future_stack=True)
        long.columns = ["adjclose", "close_nominal", "volume"]
        long.index.names = ["date", "ticker"]
        frames.append(long.dropna(subset=["adjclose"]).reset_index())
    print(f"{i + len(chunk)}/{len(syms)}", flush=True)
    time.sleep(2)
p = pd.concat(frames, ignore_index=True)
p["dollar_volume"] = p["close_nominal"] * p["volume"]
p.to_parquet(out, index=False)
print("filas", len(p), "tickers", p["ticker"].nunique(), "hasta", p["date"].max(), flush=True)
