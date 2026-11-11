"""Entrenamiento walk-forward del ranker profundo.

Regla innegociable de este modulo: **las fechas de entrenamiento, validacion y
evaluacion se calculan exactamente igual que en `rank.walk_forward_oos`**. Si
una red profunda usara una ventana distinta, cualquier comparacion contra los
agentes lineales seria invalida y, peor, una fuga pasaria desapercibida porque
los agentes clasicos seguirian pareciendo limpios.

La validacion para la parada temprana se toma del FINAL del tramo de
entrenamiento y se purga del propio entrenamiento con el mismo horizonte y
embargo. Sin esa purga, la parada temprana elige el checkpoint que mejor predice
datos que el modelo ya vio de forma indirecta, que es una fuga silenciosa y de
las mas caras.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from dataclasses import replace as _replace

from .deep import ARCHS, TORCH_AVAILABLE, DeepConfig, best_device, fit_ranker, predict_scores

if TORCH_AVAILABLE:
    import torch

__all__ = ["DeepPanelBuilder", "deep_walk_forward", "deep_final_scores"]


@dataclass
class DeepPanelBuilder:
    """Convierte el panel tabular en tensores (secuencia, factores, sector)."""

    prices: pd.DataFrame
    volumes: pd.DataFrame
    factor_cols: list[str]
    seq_len: int = 60
    groups: pd.Series | None = None

    def __post_init__(self) -> None:
        px = self.prices.sort_index()
        self.ret = np.log(px / px.shift(1)).fillna(0.0).clip(-0.25, 0.25)
        vol = self.volumes.sort_index().replace(0, np.nan)
        lv = np.log(vol).replace([np.inf, -np.inf], np.nan)
        # z-score temporal del volumen por accion: escala estable entre tickers
        self.vz = ((lv - lv.rolling(126, min_periods=20).mean())
                   / lv.rolling(126, min_periods=20).std()).fillna(0.0).clip(-4, 4)
        self.dates = px.index
        self._pos = {d: i for i, d in enumerate(self.dates)}
        codes = None
        if self.groups is not None:
            codes = pd.Categorical(self.groups.reindex(px.columns)).codes
        self.sector_codes = pd.Series(
            codes if codes is not None else np.full(px.shape[1], -1), index=px.columns)
        self.n_sectors = int(max(self.sector_codes.max(), 0)) + 1

    def batch(self, date, symbols, feats: np.ndarray, target: np.ndarray | None,
              fwd: np.ndarray | None, device: str):
        i = self._pos.get(date)
        if i is None or i < self.seq_len:
            return None
        sl = slice(i - self.seq_len + 1, i + 1)
        r = self.ret.iloc[sl][symbols].to_numpy(float).T          # (N, L)
        v = self.vz.iloc[sl][symbols].to_numpy(float).T
        seq = np.stack([r, v], axis=-1)                           # (N, L, 2)
        if not np.isfinite(seq).all():
            seq = np.nan_to_num(seq)
        t = lambda a, dt=None: torch.tensor(a, dtype=dt or torch.float32, device=device)
        return (
            t(seq), t(np.nan_to_num(feats)),
            t(self.sector_codes.reindex(symbols).fillna(-1).to_numpy(), torch.long),
            t(target) if target is not None else None,
            t(fwd) if fwd is not None else None,
        )


def resolve_arch(dcfg: DeepConfig, n_symbols: int) -> DeepConfig:
    """Resuelve `arch="auto"` por tamano de universo. Regla de coste, no de score."""
    if dcfg.arch != "auto":
        return dcfg
    return _replace(dcfg, arch="mixer" if n_symbols > dcfg.auto_mixer_above else "ensemble")


def _variants(dcfg: DeepConfig) -> list[DeepConfig]:
    """Configuraciones a entrenar. `ensemble` entrena las tres arquitecturas.

    Promediar rangos de varias arquitecturas NO es seleccionar: no gasta un
    grado de libertad y es lo que la evidencia empirica senala como mas robusto.
    Seleccionar la mejor de tres si lo gastaria, y habria que deflactarlo.
    """
    if dcfg.arch == "ensemble":
        return [_replace(dcfg, arch=a) for a in ARCHS]
    return [dcfg]


def _eval_dates(dates: pd.DatetimeIndex, cfg) -> pd.DatetimeIndex:
    """Identico a rank.walk_forward_oos: una fecha por horizonte, ultimos oos_months."""
    last_labelled = dates[-(cfg.horizon + 1)]
    cand = dates[dates <= last_labelled]
    return cand[::-1][::cfg.horizon][: cfg.oos_months][::-1]


def _build_batches(df, builder, dates_subset, cfg, device, fwd_col):
    out = []
    for d in dates_subset:
        try:
            blk = df.loc[d].dropna(subset=["target"])
        except KeyError:
            continue
        if len(blk) < 8:
            continue
        b = builder.batch(d, list(blk.index), blk[builder.factor_cols].to_numpy(float),
                          blk["target"].to_numpy(float), blk[fwd_col].to_numpy(float), device)
        if b is not None:
            out.append(b)
    return out


def deep_walk_forward(df: pd.DataFrame, cfg, builder: DeepPanelBuilder,
                      dcfg: DeepConfig = DeepConfig(), verbose: bool = False) -> pd.DataFrame:
    """Scores OOS del ranker profundo, una fecha de evaluacion por horizonte."""
    if not TORCH_AVAILABLE:
        return pd.DataFrame(columns=["date", "symbol", "score", "fwd", "success"])

    device = best_device(dcfg.device)
    dcfg = resolve_arch(dcfg, builder.prices.shape[1])
    dates = df.index.get_level_values("date").unique().sort_values()
    fwd_col = f"fwd_{cfg.horizon}"
    rows = []

    for d in _eval_dates(dates, cfg):
        pos = dates.get_loc(d)
        train_end = dates[max(0, pos - cfg.horizon - cfg.embargo)]
        train_start = train_end - pd.Timedelta(days=int(365 * cfg.train_years))
        tr_dates = dates[(dates >= train_start) & (dates <= train_end)]
        if len(tr_dates) < 150:
            continue

        # validacion: ultimo 15% del tramo, purgado del entrenamiento
        n_val = max(8, int(0.15 * len(tr_dates)))
        val_dates = tr_dates[-n_val:]
        cut = len(tr_dates) - n_val - (cfg.horizon + cfg.embargo)
        fit_dates = tr_dates[:max(cut, 0)]
        if len(fit_dates) < 100:
            continue

        tb = _build_batches(df, builder, fit_dates, cfg, device, fwd_col)
        vb = _build_batches(df, builder, val_dates, cfg, device, fwd_col)
        if len(tb) < 30:
            continue

        models = []
        for vcfg in _variants(dcfg):
            for s in range(dcfg.seeds):
                m, _ = fit_ranker(tb, vb, len(builder.factor_cols), 2, builder.n_sectors,
                                  vcfg, seed=1000 + s)
                models.append(m)

        te = df.loc[d].dropna(subset=["target"])
        if len(te) < 5:
            continue
        b = builder.batch(d, list(te.index), te[builder.factor_cols].to_numpy(float),
                          None, None, device)
        if b is None:
            continue
        sc = predict_scores(models, b)
        rows.append(pd.DataFrame({"date": d, "symbol": te.index, "score": sc,
                                  "fwd": te[fwd_col].to_numpy(),
                                  "success": te["success"].to_numpy()}))
        if verbose:
            print(f"      deep {d.date()}  train={len(tb)}  val={len(vb)}  n={len(te)}")

    if not rows:
        return pd.DataFrame(columns=["date", "symbol", "score", "fwd", "success"])
    return pd.concat(rows, ignore_index=True)


def deep_final_scores(df: pd.DataFrame, cfg, builder: DeepPanelBuilder,
                      dcfg: DeepConfig = DeepConfig()) -> pd.Series:
    """Reentrena con todo lo etiquetado y puntua la ULTIMA fecha disponible."""
    if not TORCH_AVAILABLE:
        return pd.Series(dtype=float)
    device = best_device(dcfg.device)
    dcfg = resolve_arch(dcfg, builder.prices.shape[1])
    dates = df.index.get_level_values("date").unique().sort_values()
    last = dates[-1]
    train_end = dates[max(0, len(dates) - 1 - cfg.horizon - cfg.embargo)]
    train_start = train_end - pd.Timedelta(days=int(365 * cfg.train_years))
    tr = dates[(dates >= train_start) & (dates <= train_end)]
    n_val = max(8, int(0.15 * len(tr)))
    val_dates, fit_dates = tr[-n_val:], tr[: max(len(tr) - n_val - cfg.horizon - cfg.embargo, 0)]

    fwd_col = f"fwd_{cfg.horizon}"
    tb = _build_batches(df, builder, fit_dates, cfg, device, fwd_col)
    vb = _build_batches(df, builder, val_dates, cfg, device, fwd_col)
    if len(tb) < 30:
        return pd.Series(dtype=float)

    models = [fit_ranker(tb, vb, len(builder.factor_cols), 2, builder.n_sectors,
                         vcfg, seed=2000 + s)[0]
              for vcfg in _variants(dcfg) for s in range(dcfg.seeds)]
    today = df.loc[last]
    b = builder.batch(last, list(today.index), today[builder.factor_cols].to_numpy(float),
                      None, None, device)
    if b is None:
        return pd.Series(dtype=float)
    return pd.Series(predict_scores(models, b), index=today.index)
