"""Excluye fondos cerrados, ETN y productos apalancados del panel (heuristica por nombre).

Uso: python3 scripts/equity_filter.py data/nasdaqtraded.txt data/stocks_panel.parquet data/stocks_panel_equity.parquet
(nasdaqtraded.txt: https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqtraded.txt)
"""
import sys
import pandas as pd
src, panel, dst = (sys.argv[1:4] + ["data/nasdaqtraded.txt", "data/stocks_panel.parquet", "data/stocks_panel_equity.parquet"][len(sys.argv[1:4]):])
d = pd.read_csv(src, sep="|", keep_default_na=False, dtype=str)
fund = (r"\bFund\b|Municipal|\bMuni\b|Portfolio|Income Trust|Term Trust|\bETN\b|ETNs|Leveraged|Inverse|"
        r"MicroSectors|Closed[- ]End|Exchange[- ]Traded|Index Trust")
sponsors = (r"BlackRock|Gabelli|Nuveen|Eaton Vance|Cohen & Steers|Calamos|Western Asset|PIMCO|Aberdeen|abrdn|"
            r"Allspring|John Hancock|Virtus|Tekla|Adams|Neuberger|Kayne Anderson|Clough|Saba|Royce|Liberty All-Star|"
            r"Tortoise|Flaherty|Duff & Phelps|MFS|Putnam|Templeton|Central Securities|General American|Tri-Continental|"
            r"Source Capital|Bancroft|Ellsworth|Zweig|Voya|First Trust|Franklin|Invesco|Pioneer|DWS|Guggenheim|"
            r"Ares|Apollo|Blackstone|Brookfield|Highland|Oxford Lane|Eagle Point|Swiss Helvetia|Mexico|China Fund|India Fund")
cef_words = r"Trust|Fund|Income|Opportunit|Strateg|Municipal|Credit|Dividend|Securities|Investors|Equity|Total Return|Premium|Allocation"
name = d["Security Name"]
mask = name.str.contains(fund, case=False) | (name.str.contains(sponsors, case=False) & name.str.contains(cef_words, case=False))
funds = set(d.loc[mask, "Symbol"])
p = pd.read_parquet(panel)
b = p["ticker"].nunique(); p = p[~p["ticker"].isin(funds)]
print("excluidos:", b - p["ticker"].nunique(), "quedan", p["ticker"].nunique())
print({t: t in funds for t in ["BDJ","CET","GAB","BOE","GAM","BME","BLK","BX","BEN","FRT","IVZ","ARES","APO","BN","CHT","SO"]})
p.to_parquet(dst, index=False)
