"""Modo manual: dado un universo, las k acciones con mayor probabilidad
calibrada de subir de forma notable en `horizon` sesiones.

Como funciona, y por que cada paso existe:

  1. Factores z-scoreados por fecha (data/features.py). Solo pasado.
  2. Target: retorno futuro a `horizon` sesiones, RELATIVO a la mediana del
     universo ese dia y normalizado por rango. Predecir el nivel absoluto es
     casi imposible; predecir quien lo hara mejor que sus vecinos es donde hay
     algo de senal (Gu-Kelly-Xiu 2020).
  3. Walk-forward: para cada fecha de evaluacion d de los ultimos `oos_months`
     meses, se entrena SOLO con datos cuyo target ya era observable en
     d - horizon - embargo, y se predice d. Asi obtenemos predicciones
     genuinamente out-of-sample sobre TU universo.
  4. Diagnostico OOS: rank IC medio, su t-stat, precision@k frente a la tasa
     base. Esto es lo que te dice si en tu universo hay senal explotable.
  5. Calibracion isotonica score -> P(retorno > umbral), ajustada solo con las
     predicciones OOS. Si no hay senal, todas las probabilidades colapsan a la
     tasa base: la herramienta se niega a inventar certeza.
  6. Ranking final en la ultima fecha con probabilidades calibradas y un nivel
     de evidencia (FUERTE / DEBIL / NINGUNA) que acompana a cada salida.

Modelo por defecto: ridge en numpy (cerrado, estable, sin dependencias).
Opcional: LightGBM si esta instalado (`--model lightgbm`).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

from .data.features import FACTOR_NAMES, compute_features, compute_regime, dollar_volume, forward_returns
from .data.loaders import MarketData
from .stats.calibration import IsotonicCalibrator, expected_calibration_error


@dataclass
class RankConfig:
    horizon: int = 21
    top_k: int = 5
    success_threshold: float = 0.05     # "subida notable" = +5% absoluto en `horizon`
    train_years: float = 3.0
    embargo: int = 5
    oos_months: int = 18
    model: str = "ridge"                # ridge | gbdt | ensemble (rank-average ridge+gbdt)
    ridge_alpha: float = 20.0
    gbdt_seeds: int = 2                 # bagging de semillas para gbdt/ensemble
    neutralize: bool = False            # demean sectorial por fecha (requiere groups)
    available_lag: int = 1              # sesiones de retraso para senales externas
    news_scorer: str = "auto"           # auto | lexicon | finbert
    min_dollar_volume: float = 0.0      # 0 = sin filtro (CSV sin volumen)
    min_price: float = 1.0              # suelo de precio (1.0 excluye penny stocks)
    max_price: float = 0.0              # 0 = sin techo
    # Banda de capitalizacion. Convencion SEC/industria: nano <50M, micro <300M,
    # small <2B, mid <10B, large >10B. REQUIERE market_cap point-in-time: usar el
    # `scalemarketcap` de la tabla `tickers` seria lookahead, porque clasifica por
    # capitalizacion MAXIMA historica.
    min_market_cap: float = 0.0         # 0 = sin suelo
    max_market_cap: float = 0.0         # 0 = sin techo (10e9 = hasta mid cap)
    max_spread: float = 0.0             # 0 = sin filtro; spread Corwin-Schultz maximo
    min_history_days: int = 300
    strong_tstat: float = 3.0
    weak_tstat: float = 2.0
    seed: int = 0


@dataclass
class RankDiagnostics:
    n_oos_dates: int
    n_oos_obs: int
    mean_rank_ic: float
    ic_tstat: float
    icir: float
    precision_at_k: float
    base_rate: float
    precision_lift: float
    precision_ci95: tuple[float, float]
    ece: float
    evidence: str                       # FUERTE | DEBIL | NINGUNA
    factor_ic: dict = field(default_factory=dict)   # {factor: (ic_medio, t)} univariante OOS
    per_date_ic: pd.Series = field(repr=False, default=None)

    def summary(self) -> dict:
        return {
            "evidence": self.evidence,
            "n_oos_dates": self.n_oos_dates, "n_oos_obs": self.n_oos_obs,
            "mean_rank_ic": round(self.mean_rank_ic, 4), "ic_tstat": round(self.ic_tstat, 2),
            "icir": round(self.icir, 3),
            "precision_at_k": round(self.precision_at_k, 3), "base_rate": round(self.base_rate, 3),
            "precision_lift": round(self.precision_lift, 3),
            "precision_ci95": [round(self.precision_ci95[0], 3), round(self.precision_ci95[1], 3)],
            "ece": round(self.ece, 4),
            "factor_ic": {k: [round(v[0], 4), round(v[1], 2)] for k, v in self.factor_ic.items()},
        }


# ---------------------------------------------------------------- modelos
class _Ridge:
    def __init__(self, alpha: float):
        self.alpha, self.w, self.b = alpha, None, 0.0

    def fit(self, X, y):
        Xc = X - X.mean(axis=0); yc = y - y.mean()
        A = Xc.T @ Xc + self.alpha * np.eye(X.shape[1])
        self.w = np.linalg.solve(A, Xc.T @ yc)
        self.b = float(y.mean() - X.mean(axis=0) @ self.w)
        return self

    def predict(self, X):
        return X @ self.w + self.b


def _gbdt(seed: int):
    """LightGBM si esta instalado; si no, HistGradientBoosting de sklearn (ARM64 nativo)."""
    try:
        import lightgbm as lgb
        return lgb.LGBMRegressor(n_estimators=300, learning_rate=0.03, num_leaves=15,
                                 min_child_samples=50, subsample=0.8, subsample_freq=1,
                                 colsample_bytree=0.8, reg_lambda=5.0, random_state=seed, verbose=-1)
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingRegressor
        return HistGradientBoostingRegressor(max_iter=200, learning_rate=0.03, max_leaf_nodes=15,
                                             min_samples_leaf=50, l2_regularization=5.0,
                                             random_state=seed)


class _Ensemble:
    """Rank-average de varios modelos: reduce varianza y descorrelaciona errores.
    Cada modelo emite un score; se convierten a rangos [0,1] por lote y se promedian."""

    def __init__(self, models):
        self.models = models

    def fit(self, X, y):
        for m in self.models:
            m.fit(X, y)
        return self

    def predict(self, X):
        preds = []
        for m in self.models:
            p = np.asarray(m.predict(X), float)
            r = stats.rankdata(p) / max(1, p.size)
            preds.append(r)
        return np.mean(preds, axis=0) - 0.5


def _make_model(cfg: RankConfig):
    if cfg.model == "ridge":
        return _Ridge(cfg.ridge_alpha)
    if cfg.model in ("gbdt", "lightgbm"):
        return _Ensemble([_gbdt(cfg.seed + i) for i in range(cfg.gbdt_seeds)])
    if cfg.model == "ensemble":
        return _Ensemble([_Ridge(cfg.ridge_alpha)] + [_gbdt(cfg.seed + i) for i in range(cfg.gbdt_seeds)])
    raise ValueError(f"modelo desconocido: {cfg.model}")


# ---------------------------------------------------------------- target
def _relative_rank_target(fwd: pd.Series) -> pd.Series:
    """Rango cross-seccional por fecha centrado en 0 (~[-0.5, 0.5]). Robusto a colas."""
    g = fwd.groupby(level="date")
    n = g.transform("count")
    r = g.rank(pct=True) - 0.5 - 0.5 / n
    return r.where(n >= 3)


# ---------------------------------------------------------------- nucleo
def _prepare(md: MarketData, cfg: RankConfig, extra: pd.DataFrame | None = None,
             groups: pd.Series | None = None, listings: pd.DataFrame | None = None,
             market_cap: pd.DataFrame | None = None, spread: pd.DataFrame | None = None):
    px = md.prices.sort_index()
    if len(px) < cfg.min_history_days:
        raise ValueError(f"historico insuficiente: {len(px)} sesiones < {cfg.min_history_days}")
    feats = compute_features(px, md.volumes, extra=extra, groups=groups, neutralize=cfg.neutralize)
    if listings is not None:
        # etiqueta point-in-time: incluye el retorno terminal de las acciones que
        # mueren dentro del horizonte. Ver data/pit.py para por que importa.
        from .data.pit import forward_returns_pit
        fwd = forward_returns_pit(px, listings, cfg.horizon)
    else:
        fwd = forward_returns(px, cfg.horizon)
    dv = dollar_volume(px, md.volumes)
    df = feats.join(fwd, how="left").join(dv, how="left")
    # filtros de universo: precio y liquidez (si hay volumen real)
    # precio para los filtros: el nominal del dia si se conoce; si no, el ajustado
    # (que NO es point-in-time cuando hay splits posteriores: ver MarketData.nominal)
    ref_px = md.nominal.sort_index() if getattr(md, "nominal", None) is not None else px
    last_px = ref_px.stack(); last_px.index.names = ["date", "symbol"]
    df = df.join(last_px.rename("px"), how="left")
    df = df[(df["px"] >= cfg.min_price)]
    if cfg.max_price > 0:
        df = df[df["px"] <= cfg.max_price]
    if cfg.min_dollar_volume > 0:
        df = df[df["dollar_volume"] >= cfg.min_dollar_volume]
    # capitalizacion point-in-time: fecha a fecha, nunca la maxima historica
    if (cfg.min_market_cap > 0 or cfg.max_market_cap > 0):
        if market_cap is None:
            raise ValueError(
                "se han configurado limites de capitalizacion pero no se ha pasado "
                "`market_cap` point-in-time. Usar una clasificacion estatica (p. ej. "
                "`scalemarketcap` de Sharadar `tickers`) seria lookahead: clasifica por "
                "capitalizacion MAXIMA historica. Fuente correcta: tabla `daily` "
                "(plan Fundamentals) o marketcap calculado con acciones en circulacion PIT.")
        mc = market_cap.reindex_like(px).stack(future_stack=True)
        mc.index.names = ["date", "symbol"]
        df = df.join(mc.rename("market_cap"), how="left")
        df = df[df["market_cap"].notna()]
        if cfg.min_market_cap > 0:
            df = df[df["market_cap"] >= cfg.min_market_cap]
        if cfg.max_market_cap > 0:
            df = df[df["market_cap"] <= cfg.max_market_cap]
    # spread estimado: critico si se incluyen penny stocks, donde 2 centimos de
    # spread en una accion de 0.50 $ son el 4% de ida y vuelta
    if cfg.max_spread > 0 and spread is not None:
        sp = spread.reindex_like(px).stack(future_stack=True)
        sp.index.names = ["date", "symbol"]
        df = df.join(sp.rename("spread_est"), how="left")
        df = df[(df["spread_est"].isna()) | (df["spread_est"] <= cfg.max_spread)]
    df["target"] = _relative_rank_target(df[f"fwd_{cfg.horizon}"])
    df["success"] = (df[f"fwd_{cfg.horizon}"] > cfg.success_threshold).astype(float)
    df.loc[df[f"fwd_{cfg.horizon}"].isna(), "success"] = np.nan
    df.attrs["factor_cols"] = list(feats.columns)
    return df


def walk_forward_oos(df: pd.DataFrame, cfg: RankConfig,
                     x_cols: list[str] | None = None) -> pd.DataFrame:
    """Predicciones OOS mensuales. Devuelve (date, symbol, score, fwd, success).

    `x_cols` permite entrenar sobre un SUBCONJUNTO de factores sin duplicar la
    logica anti-fuga (ventana de entrenamiento, embargo, purga por horizonte).
    Los especialistas del enjambre la usan asi: una sola implementacion
    auditada, muchos modelos.
    """
    dates = df.index.get_level_values("date").unique().sort_values()
    # fechas de evaluacion: una por mes, en los ultimos oos_months, con target observable
    last_labelled = dates[-(cfg.horizon + 1)]
    cand = dates[dates <= last_labelled]
    eval_dates = cand[::-1][::cfg.horizon][:cfg.oos_months][::-1]
    X_cols = list(x_cols) if x_cols else df.attrs.get("factor_cols", list(FACTOR_NAMES))
    out = []
    for d in eval_dates:
        pos = dates.get_loc(d)
        train_end = dates[max(0, pos - cfg.horizon - cfg.embargo)]
        train_start = train_end - pd.Timedelta(days=int(365 * cfg.train_years))
        tr = df.loc[(df.index.get_level_values("date") >= train_start) &
                    (df.index.get_level_values("date") <= train_end)]
        tr = tr.dropna(subset=["target"])
        if len(tr) < 200:
            continue
        te = df.loc[d]
        te = te.dropna(subset=["target"])
        if len(te) < 5:
            continue
        model = _make_model(cfg).fit(tr[X_cols].to_numpy(float), tr["target"].to_numpy(float))
        sc = model.predict(te[X_cols].to_numpy(float))
        block = pd.DataFrame({"date": d, "symbol": te.index, "score": sc,
                              "fwd": te[f"fwd_{cfg.horizon}"].to_numpy(),
                              "success": te["success"].to_numpy()})
        for c in X_cols:
            block[c] = te[c].to_numpy()
        out.append(block)
    if not out:
        return pd.DataFrame(columns=["date", "symbol", "score", "fwd", "success"])
    return pd.concat(out, ignore_index=True)


def diagnose(oos: pd.DataFrame, cfg: RankConfig) -> RankDiagnostics:
    if oos.empty:
        return RankDiagnostics(0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, (0.0, 0.0), 0.0, "NINGUNA")
    ics = oos.groupby("date").apply(
        lambda g: stats.spearmanr(g["score"], g["fwd"]).statistic if len(g) >= 5 else np.nan,
        include_groups=False).dropna()
    n = len(ics)
    mean_ic = float(ics.mean()) if n else 0.0
    sd_ic = float(ics.std(ddof=1)) if n > 1 else 0.0
    t = mean_ic / sd_ic * np.sqrt(n) if sd_ic > 1e-12 else 0.0
    icir = mean_ic / sd_ic if sd_ic > 1e-12 else 0.0

    # precision@k: fraccion de exitos en el top-k de cada fecha OOS
    topk = oos.sort_values(["date", "score"], ascending=[True, False]).groupby("date").head(cfg.top_k)
    hits = topk["success"].dropna()
    prec = float(hits.mean()) if len(hits) else 0.0
    base = float(oos["success"].dropna().mean()) if len(oos) else 0.0
    # IC 95% bootstrap por fecha (las observaciones dentro de una fecha no son independientes)
    rng = np.random.default_rng(cfg.seed)
    by_date = topk.groupby("date")["success"].mean().dropna().to_numpy()
    if by_date.size >= 3:
        bs = [rng.choice(by_date, by_date.size, replace=True).mean() for _ in range(2000)]
        ci = (float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5)))
    else:
        ci = (0.0, 1.0)

    cal = IsotonicCalibrator().fit(oos["score"].to_numpy(), oos["success"].to_numpy())
    ece = expected_calibration_error(cal.predict(oos["score"].to_numpy()), oos["success"].fillna(0).to_numpy()) if cal.fitted else 0.0

    if t >= cfg.strong_tstat and ci[0] > base:
        ev = "FUERTE"
    elif t >= cfg.weak_tstat:
        ev = "DEBIL"
    else:
        ev = "NINGUNA"

    # IC univariante por factor: que senal pesa en ESTE universo
    fac_cols = [c for c in oos.columns if c not in ("date", "symbol", "score", "fwd", "success")]
    factor_ic: dict[str, tuple[float, float]] = {}
    for c in fac_cols:
        ic_c = oos.groupby("date").apply(
            lambda g, c=c: stats.spearmanr(g[c], g["fwd"]).statistic if g[c].std() > 1e-12 and len(g) >= 5 else np.nan,
            include_groups=False).dropna()
        if len(ic_c) >= 3:
            m, sdv = float(ic_c.mean()), float(ic_c.std(ddof=1))
            factor_ic[c] = (m, m / sdv * np.sqrt(len(ic_c)) if sdv > 1e-12 else 0.0)
    factor_ic = dict(sorted(factor_ic.items(), key=lambda kv: -abs(kv[1][1])))
    return RankDiagnostics(n, int(len(oos)), mean_ic, float(t), float(icir), prec, base,
                           prec - base, ci, float(ece), ev, factor_ic=factor_ic, per_date_ic=ics)


@dataclass
class RankResult:
    as_of: str
    horizon: int
    picks: pd.DataFrame            # symbol, score, prob_success, rank, last_price
    diagnostics: RankDiagnostics
    calibrator: IsotonicCalibrator
    n_universe: int
    n_eligible: int
    regime: dict

    def to_dict(self) -> dict:
        return {
            "as_of": self.as_of, "horizon_sessions": self.horizon,
            "evidence": self.diagnostics.evidence,
            "picks": self.picks.round(4).to_dict(orient="records"),
            "diagnostics": self.diagnostics.summary(),
            "n_universe": self.n_universe, "n_eligible": self.n_eligible,
            "regime": {k: round(v, 3) for k, v in self.regime.items()},
            "calibrator": self.calibrator.to_dict(),
        }


def rank_universe(md: MarketData, cfg: RankConfig = RankConfig(),
                  extra: pd.DataFrame | None = None, groups: pd.Series | None = None,
                  listings: pd.DataFrame | None = None) -> RankResult:
    df = _prepare(md, cfg, extra=extra, groups=groups, listings=listings)
    oos = walk_forward_oos(df, cfg)
    diag = diagnose(oos, cfg)
    # calibrar SOLO si hay evidencia de senal: sobre ruido, la regresion isotonica
    # produce escalones espurios en las colas. Sin senal, todos reciben la tasa base.
    cal = IsotonicCalibrator()
    if not oos.empty and diag.evidence != "NINGUNA":
        cal = cal.fit(oos["score"].to_numpy(), oos["success"].to_numpy())
    else:
        cal.base_rate = diag.base_rate if diag.n_oos_obs else 0.5

    # modelo final: entrenado con TODO lo etiquetado, prediccion en la ultima fecha
    dates = df.index.get_level_values("date").unique().sort_values()
    last = dates[-1]
    labelled = df.dropna(subset=["target"])
    cutoff = last - pd.Timedelta(days=int(365 * cfg.train_years))
    labelled = labelled.loc[labelled.index.get_level_values("date") >= cutoff]
    X_cols = df.attrs.get("factor_cols", list(FACTOR_NAMES))
    today = df.loc[last]
    if len(labelled) >= 200 and len(today) >= 1:
        model = _make_model(cfg).fit(labelled[X_cols].to_numpy(float), labelled["target"].to_numpy(float))
        sc = model.predict(today[X_cols].to_numpy(float))
    else:
        sc = today[X_cols].to_numpy(float).mean(axis=1)   # fallback: composite equiponderado
    picks = pd.DataFrame({"symbol": today.index, "score": sc, "last_price": today["px"].to_numpy()})
    picks["prob_success"] = cal.predict(picks["score"].to_numpy())
    picks = picks.sort_values("score", ascending=False).reset_index(drop=True)
    picks["rank"] = np.arange(1, len(picks) + 1)
    reg = compute_regime(md.prices).iloc[-1].to_dict()
    return RankResult(as_of=str(last.date()), horizon=cfg.horizon,
                      picks=picks.head(cfg.top_k), diagnostics=diag, calibrator=cal,
                      n_universe=md.prices.shape[1], n_eligible=int(len(today)), regime=reg)


def format_report(res: RankResult, cfg: RankConfig) -> str:
    d = res.diagnostics
    lines = [
        f"Fecha de analisis : {res.as_of}    horizonte: {res.horizon} sesiones    universo: {res.n_eligible}/{res.n_universe} elegibles",
        f"Exito definido    : retorno > {cfg.success_threshold:+.0%} en {res.horizon} sesiones   tasa base del universo: {d.base_rate:.0%}",
        "",
        f"EVIDENCIA DE SENAL EN ESTE UNIVERSO: {d.evidence}",
        f"  rank IC medio {d.mean_rank_ic:+.3f}  t={d.ic_tstat:.2f}  ICIR {d.icir:.2f}  sobre {d.n_oos_dates} fechas OOS ({d.n_oos_obs} obs)",
        f"  precision@{cfg.top_k} OOS {d.precision_at_k:.0%}  (tasa base {d.base_rate:.0%}, lift {d.precision_lift:+.0%}, IC95 [{d.precision_ci95[0]:.0%}, {d.precision_ci95[1]:.0%}])",
        "",
    ]
    if d.evidence == "NINGUNA":
        lines += ["  El modelo NO muestra capacidad predictiva out-of-sample en este universo.",
                  "  Las probabilidades de abajo son la tasa base: comprar estos nombres no es mejor",
                  "  que elegir al azar dentro del universo. Se muestran para transparencia, no como recomendacion.", ""]
    elif d.evidence == "DEBIL":
        lines += ["  Senal debil: estadisticamente marginal. Trata las probabilidades como orientativas.", ""]
    if d.factor_ic:
        lines.append("  IC univariante OOS por factor (|t| desc.): " + ", ".join(
            f"{k} {v[0]:+.3f} (t={v[1]:.1f})" for k, v in list(d.factor_ic.items())[:8]))
        lines.append("")
    lines.append(f"TOP {len(res.picks)}")
    for _, r in res.picks.iterrows():
        lines.append(f"  {int(r['rank']):>2}. {r['symbol']:<10} P(exito)={r['prob_success']:.0%}   score={r['score']:+.3f}   ultimo={r['last_price']:.2f}")
    vz = res.regime.get("mkt_vol_z", 0.0)
    if vz > 1.0:
        lines += ["", f"  Aviso: volatilidad de mercado {vz:.1f} sd por encima de su historico. Los rankings degradan en estos regimenes."]
    lines += ["", "  Esto no es asesoramiento de inversion. Ver README para las limitaciones."]
    return "\n".join(lines)
