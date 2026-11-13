"""Senales externas con lag point-in-time. Cada funcion devuelve un panel
MultiIndex (date, symbol) alineado al calendario de precios, listo para
`compute_features(extra=...)`.

CONTRATO COMUN
  * CSV con columnas `date`, `symbol` y las de valor de cada fuente.
  * `date` es la fecha en que la informacion fue PUBLICA. Para fundamentales es
    la fecha de filing (datekey / first_filed), NUNCA el cierre del periodo.
  * `available_lag` (defecto 1 sesion): un dato fechado en t se usa a partir de
    t+1. Es conservador a proposito: un CSV con solo fecha no dice si la noticia
    salio antes o despues del cierre. Poner 0 solo si tienes timestamp intradia
    y lo has filtrado tu.
  * Los simbolos sin datos quedan neutros (0 tras el z-score por fecha).

EVIDENCIA POR FUENTE (horizonte ~1 mes)
  analyst    cambios de consenso/revisiones: Jegadeesh et al. 2004; el NIVEL es
             debil (Farago-Hjalmarsson). Aqui se usan deltas.
  earnings   PEAD / SUE: Bernard-Thomas 1989; drift 1-3 meses tras el anuncio.
  news       sentimiento LLM/lexico: Lopez-Lira & Tang 2023 (efecto sobre todo a
             1-5 dias; a 21 dias queda atencion y sentimiento agregado).
  insider    compras netas de insiders: Lakonishok-Lee 2001; cluster buys.
  short      short interest / days-to-cover: predictor NEGATIVO (Boehmer et al.).
  fundamentals gross profitability (Novy-Marx 2013), accruals (Sloan 1996,
             negativo), book-to-market (debil a 1 mes).
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------- utilidades
_DATE = ("date", "Date", "timestamp", "datetime", "published", "filed", "datekey")
_SYM = ("symbol", "ticker", "Symbol", "Ticker")


def _read(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    d = next((c for c in _DATE if c in df.columns), None)
    s = next((c for c in _SYM if c in df.columns), None)
    if d is None or s is None:
        raise ValueError(f"{path}: hacen falta columnas date y symbol")
    df = df.rename(columns={d: "date", s: "symbol"})
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None).dt.normalize()
    df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()
    return df


def _to_calendar(daily: pd.DataFrame, dates: pd.DatetimeIndex, lag: int, how: str) -> pd.DataFrame:
    """daily: wide (fecha x simbolo). Alinea al calendario de precios y aplica lag.
    how='state' -> forward-fill (ultimo valor conocido); how='flow' -> 0 donde no hay."""
    idx = dates.union(daily.index).sort_values()
    w = daily.reindex(idx)
    w = w.ffill() if how == "state" else w.fillna(0.0)
    w = w.reindex(dates)
    return w.shift(lag)


def _stack(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    out = pd.concat([v.stack().rename(k) for k, v in frames.items()], axis=1)
    out.index.names = ["date", "symbol"]
    return out


# ---------------------------------------------------------------- noticias
_POS = {"beat", "beats", "surge", "surges", "soar", "soars", "rally", "record", "upgrade", "upgrades",
        "upgraded", "outperform", "strong", "growth", "profit", "profits", "gain", "gains", "raises",
        "raised", "guidance raised", "exceeds", "exceeded", "buyback", "dividend increase", "wins", "won",
        "approval", "approved", "expands", "expansion", "partnership", "breakthrough", "bullish",
        "optimistic", "positive", "boost", "boosts", "rebound", "rebounds", "recovery", "milestone",
        "innovation", "momentum", "accelerates", "acquires", "acquisition", "demand", "robust", "solid"}
_NEG = {"miss", "misses", "missed", "plunge", "plunges", "slump", "slumps", "fall", "falls", "drop", "drops",
        "downgrade", "downgrades", "downgraded", "underperform", "weak", "loss", "losses", "cuts", "cut",
        "guidance cut", "lowers", "lowered", "warning", "warns", "lawsuit", "sued", "probe",
        "investigation", "recall", "fraud", "bankruptcy", "default", "layoffs", "layoff", "delay",
        "delays", "halted", "suspends", "suspended", "decline", "declines", "bearish", "pessimistic",
        "negative", "concern", "concerns", "risk", "risks", "volatile", "shortfall", "writedown",
        "impairment", "restatement", "resigns", "resignation", "fine", "fined", "penalty", "breach"}


def lexicon_score(text: str) -> float:
    """Puntuacion [-1, 1] con lexico financiero compacto. Fallback sin modelo."""
    t = re.sub(r"[^a-z ]", " ", str(text).lower())
    toks = t.split()
    if not toks:
        return 0.0
    joined = " " + " ".join(toks) + " "
    pos = sum(1 for w in _POS if f" {w} " in joined)
    neg = sum(1 for w in _NEG if f" {w} " in joined)
    # negacion simple: "not strong" invierte
    if re.search(r"\b(not|no|never)\s+(\w+\s+){0,2}(strong|good|positive|beat)", t):
        pos, neg = neg, pos
    return float((pos - neg) / max(1, pos + neg)) if (pos + neg) else 0.0


def finbert_scores(texts: list[str], model_name: str = "ProsusAI/finbert", batch: int = 32) -> np.ndarray:
    """Sentimiento con FinBERT en local (requiere transformers + torch; MPS en Apple Silicon).
    Devuelve P(pos) - P(neg) en [-1, 1]."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_name).to(device).eval()
    labels = [mdl.config.id2label[i].lower() for i in range(mdl.config.num_labels)]
    ip, ineg = labels.index("positive"), labels.index("negative")
    out = []
    with torch.no_grad():
        for i in range(0, len(texts), batch):
            enc = tok(texts[i:i + batch], padding=True, truncation=True, max_length=128, return_tensors="pt").to(device)
            p = torch.softmax(mdl(**enc).logits, dim=-1).cpu().numpy()
            out.append(p[:, ip] - p[:, ineg])
    return np.concatenate(out) if out else np.array([])


def news_features(path: str | Path, dates: pd.DatetimeIndex, symbols: list[str],
                  available_lag: int = 1, scorer: str = "auto") -> pd.DataFrame:
    """CSV: date, symbol, headline | text | score. scorer: auto | lexicon | finbert.
    Features: news_sent_5, news_sent_21, news_cnt_21 (log), news_sent_chg."""
    df = _read(path)
    if "score" not in df.columns:
        col = next((c for c in ("headline", "title", "text") if c in df.columns), None)
        if col is None:
            raise ValueError("news: hace falta score, headline, title o text")
        texts = df[col].astype(str).tolist()
        use_finbert = scorer == "finbert"
        if scorer == "auto":
            try:
                import torch, transformers  # noqa: F401
                use_finbert = True
            except ImportError:
                use_finbert = False
        df["score"] = finbert_scores(texts) if use_finbert else [lexicon_score(t) for t in texts]
    df = df[df["symbol"].isin(symbols)]
    daily = df.groupby(["date", "symbol"]).agg(score=("score", "mean"), n=("score", "size"))
    sc = daily["score"].unstack("symbol").reindex(columns=symbols)
    cnt = daily["n"].unstack("symbol").reindex(columns=symbols)
    sc = _to_calendar(sc, dates, available_lag, "flow")
    cnt = _to_calendar(cnt, dates, available_lag, "flow")
    # media ponderada por numero de noticias en la ventana
    def wmean(win):
        num = (sc * cnt).rolling(win, min_periods=1).sum()
        den = cnt.rolling(win, min_periods=1).sum()
        return num / den.replace(0, np.nan)
    s5, s21 = wmean(5), wmean(21)
    return _stack({
        "news_sent_5": s5,
        "news_sent_21": s21,
        "news_cnt_21": np.log1p(cnt.rolling(21, min_periods=1).sum()),
        "news_sent_chg": s5 - s21,
    })


# ---------------------------------------------------------------- analistas
def analyst_features(path: str | Path, dates: pd.DatetimeIndex, symbols: list[str],
                     prices: pd.DataFrame | None = None, available_lag: int = 1) -> pd.DataFrame:
    """CSV: date, symbol, consensus (numerico, mayor = mejor) [, target].
    Features: rev_chg_21, rev_chg_63 (DELTAS del consenso), target_gap (nivel; debil)."""
    df = _read(path)
    if "consensus" not in df.columns:
        raise ValueError("analyst: hace falta columna consensus")
    df = df[df["symbol"].isin(symbols)]
    cons = df.pivot_table(index="date", columns="symbol", values="consensus", aggfunc="last").reindex(columns=symbols)
    cons = _to_calendar(cons, dates, available_lag, "state")
    out = {"rev_chg_21": cons - cons.shift(21), "rev_chg_63": cons - cons.shift(63)}
    if "target" in df.columns and prices is not None:
        tgt = df.pivot_table(index="date", columns="symbol", values="target", aggfunc="last").reindex(columns=symbols)
        tgt = _to_calendar(tgt, dates, available_lag, "state")
        out["target_gap"] = tgt / prices.reindex(index=dates, columns=symbols) - 1.0
    return _stack(out)


# ---------------------------------------------------------------- resultados
def earnings_features(path: str | Path, dates: pd.DatetimeIndex, symbols: list[str],
                      prices: pd.DataFrame | None = None, available_lag: int = 1,
                      drift_window: int = 63) -> pd.DataFrame:
    """CSV: date (anuncio), symbol, eps_actual, eps_estimate [| surprise_pct].
    Features: sue (sorpresa estandarizada, vigente drift_window sesiones),
    earn_react_3d (retorno t-1..t+1 del anuncio, vigente drift_window),
    days_since_earn (negado y acotado: mas reciente = mayor)."""
    df = _read(path)
    if "sue" in df.columns:                 # ya estandarizada (p. ej. data/fundamentals.py)
        df["sue"] = df["sue"].astype(float)
    elif "surprise_pct" in df.columns:
        df["sue"] = df["surprise_pct"].astype(float)
    elif {"eps_actual", "eps_estimate"} <= set(df.columns):
        est = df["eps_estimate"].astype(float)
        df["sue"] = (df["eps_actual"].astype(float) - est) / est.abs().clip(lower=0.05)
    else:
        raise ValueError("earnings: hace falta surprise_pct o eps_actual+eps_estimate")
    df["sue"] = df["sue"].clip(-3, 3)
    df = df[df["symbol"].isin(symbols)]
    sue = df.pivot_table(index="date", columns="symbol", values="sue", aggfunc="last").reindex(columns=symbols)
    idx = dates.union(sue.index).sort_values()
    sue_w = sue.reindex(idx).reindex(dates)
    # vigente drift_window sesiones tras el anuncio, luego 0
    sue_f = sue_w.ffill(limit=drift_window).fillna(0.0).shift(available_lag)
    ev = sue_w.notna().astype(float)
    days = ev.copy()
    for c in days.columns:
        cnt, col = np.nan, []
        for v in ev[c].to_numpy():
            cnt = 0.0 if v > 0 else (cnt + 1.0 if cnt == cnt else np.nan)
            col.append(cnt)
        days[c] = col
    days_since = (-days.clip(upper=drift_window)).shift(available_lag)
    out = {"sue": sue_f, "days_since_earn": days_since}
    if prices is not None:
        px = prices.reindex(index=dates, columns=symbols)
        react = (px.shift(-1) / px.shift(1) - 1.0).where(ev > 0)
        # la reaccion incluye t+1, asi que esta disponible desde t+2
        out["earn_react_3d"] = react.reindex(dates).ffill(limit=drift_window).fillna(0.0).shift(max(available_lag, 2))
    return _stack(out)


# ---------------------------------------------------------------- insiders / short
def insider_features(path: str | Path, dates: pd.DatetimeIndex, symbols: list[str],
                     available_lag: int = 1) -> pd.DataFrame:
    """CSV: date, symbol, value (USD netos; compra > 0, venta < 0). Feature: insider_net_63 (signed log)."""
    df = _read(path)
    if "value" not in df.columns:
        raise ValueError("insider: hace falta columna value")
    df = df[df["symbol"].isin(symbols)]
    v = df.groupby(["date", "symbol"])["value"].sum().unstack("symbol").reindex(columns=symbols)
    v = _to_calendar(v, dates, available_lag, "flow")
    net = v.rolling(63, min_periods=1).sum()
    return _stack({"insider_net_63": np.sign(net) * np.log1p(net.abs())})


def short_interest_features(path: str | Path, dates: pd.DatetimeIndex, symbols: list[str],
                            available_lag: int = 1) -> pd.DataFrame:
    """CSV: date, symbol, short_interest (float o pct) [| days_to_cover]. Feature: short_int (negada)."""
    df = _read(path)
    col = "days_to_cover" if "days_to_cover" in df.columns else "short_interest"
    if col not in df.columns:
        raise ValueError("short: hace falta short_interest o days_to_cover")
    df = df[df["symbol"].isin(symbols)]
    si = df.pivot_table(index="date", columns="symbol", values=col, aggfunc="last").reindex(columns=symbols)
    si = _to_calendar(si, dates, available_lag, "state")
    return _stack({"short_int": -si, "short_int_chg": -(si - si.shift(21))})


# ---------------------------------------------------------------- fundamentales
def fundamentals_features(path: str | Path, dates: pd.DatetimeIndex, symbols: list[str],
                          prices: pd.DataFrame | None = None, available_lag: int = 1,
                          nominal: pd.DataFrame | None = None) -> pd.DataFrame:
    """CSV: date (= fecha de DISPONIBILIDAD/filing), symbol, y alguna de:
    gross_profit, total_assets, net_income, cfo, book_equity, market_cap, shares,
    fscore, asset_growth, net_issuance (las genera `data/fundamentals.py`).

    Features (todas orientadas: mayor = mejor segun la tesis previa):
      valor   bm, ep, cfp           (Fama-French 1992; Lakonishok-Shleifer-Vishny 1994)
      calidad gp_a, fscore, accruals (negado), inv_ag (crecimiento de activos
              negado), anti_issuance (emision neta negada)
              (Novy-Marx 2013; Piotroski 2000; Sloan 1996; Cooper-Gulen-Schill
              2008; Pontiff-Woodgate 2008)

    La capitalizacion se calcula con el precio NOMINAL (`nominal`) por las acciones
    publicadas: el precio ajustado ya incorpora splits y dividendos futuros y,
    multiplicado por las acciones de la fecha, da una capitalizacion que no
    existio. Sin `nominal` se usa `prices` y se avisa.
    """
    df = _read(path)
    df = df[df["symbol"].isin(symbols)]

    def wide(col):
        if col not in df.columns:
            return None
        w = df.pivot_table(index="date", columns="symbol", values=col, aggfunc="last").reindex(columns=symbols)
        return _to_calendar(w, dates, available_lag, "state")

    gp, ta, ni, cfo, be, mc = (wide(c) for c in ("gross_profit", "total_assets", "net_income", "cfo", "book_equity", "market_cap"))
    pos = lambda x: x.where(x > 0)  # noqa: E731
    out = {}
    if gp is not None and ta is not None:
        out["gp_a"] = gp / pos(ta)
    if ni is not None and cfo is not None and ta is not None:
        out["accruals"] = -((ni - cfo) / pos(ta))
    if mc is None and "shares" in df.columns:
        px = nominal if nominal is not None else prices
        if px is not None:
            if nominal is None:
                import warnings
                warnings.warn("fundamentals: capitalizacion con precio ajustado (falta el "
                              "cierre nominal); bm/ep/cfp quedan sesgados", stacklevel=2)
            mc = px.reindex(index=dates, columns=symbols) * wide("shares")
    if mc is not None:
        mc = pos(mc)
        if be is not None:
            out["bm"] = be / mc
        if ni is not None:
            out["ep"] = ni / mc
        if cfo is not None:
            out["cfp"] = cfo / mc
    for col, name, sign in (("fscore", "fscore", 1.0), ("asset_growth", "inv_ag", -1.0),
                            ("net_issuance", "anti_issuance", -1.0)):
        w = wide(col)
        if w is not None:
            out[name] = sign * w.replace([np.inf, -np.inf], np.nan)
    if not out:
        raise ValueError("fundamentals: ninguna combinacion de columnas reconocida")
    return _stack(out)


# ---------------------------------------------------------------- grupos
def load_groups(path: str | Path) -> pd.Series:
    """CSV: symbol, sector (o industry / group). Devuelve Series symbol -> grupo."""
    df = pd.read_csv(path, encoding="utf-8-sig")
    s = next((c for c in _SYM if c in df.columns), None)
    g = next((c for c in ("sector", "industry", "group", "Sector", "Industry") if c in df.columns), None)
    if s is None or g is None:
        raise ValueError("groups: hacen falta columnas symbol y sector/industry/group")
    return df.set_index(df[s].astype(str).str.upper())[g].astype(str)


def groups_from_yfinance(symbols: list[str]) -> pd.Series:
    """Sector via yfinance (lento: una llamada por simbolo). Cachea el resultado en CSV."""
    import yfinance as yf
    out = {}
    for s in symbols:
        try:
            out[s] = yf.Ticker(s).info.get("sector") or "_none_"
        except Exception:  # noqa: BLE001
            out[s] = "_none_"
    return pd.Series(out)


# ---------------------------------------------------------------- orquestacion
def build_extra(dates: pd.DatetimeIndex, symbols: list[str], prices: pd.DataFrame | None = None,
                news: str | None = None, analyst: str | None = None, earnings: str | None = None,
                insider: str | None = None, short: str | None = None, fundamentals: str | None = None,
                available_lag: int = 1, news_scorer: str = "auto",
                nominal: pd.DataFrame | None = None) -> pd.DataFrame | None:
    parts = []
    if news:
        parts.append(news_features(news, dates, symbols, available_lag, news_scorer))
    if analyst:
        parts.append(analyst_features(analyst, dates, symbols, prices, available_lag))
    if earnings:
        parts.append(earnings_features(earnings, dates, symbols, prices, available_lag))
    if insider:
        parts.append(insider_features(insider, dates, symbols, available_lag))
    if short:
        parts.append(short_interest_features(short, dates, symbols, available_lag))
    if fundamentals:
        parts.append(fundamentals_features(fundamentals, dates, symbols, prices, available_lag,
                                           nominal=nominal))
    if not parts:
        return None
    return pd.concat(parts, axis=1)
