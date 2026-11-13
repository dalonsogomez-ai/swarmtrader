"""Etiquetas de triple barrera y estimador de spread.

Por que hace falta este modulo
------------------------------
"Comprar hoy y vender como muy tarde en mes y medio, pero si a las dos semanas ya
ha subido, vender entonces" NO es un horizonte fijo mas largo. Es una regla de
salida dependiente del camino, y etiquetarla con `precio[t+h]/precio[t]-1` la
describe mal: una accion que sube +12% el dia 9 y cae a -3% el dia 32 seria un
"fracaso" con la etiqueta fija cuando la operacion real habria salido ganando.

El metodo de **triple barrera** (Lopez de Prado, *Advances in Financial
Machine Learning*, cap. 3) modela exactamente esa regla:

  * barrera superior  = objetivo de beneficio (`profit_target`)
  * barrera inferior  = stop de perdidas, en dos formas simultaneas:
      - `stop_loss`  fijo respecto al precio de ENTRADA
      - `trail_stop` movil respecto al MAXIMO alcanzado desde la entrada
  * barrera vertical  = periodo maximo de tenencia (`horizon` sesiones)

El trailing stop no es un adorno: con stop fijo, una accion que sube +15% el dia
10 sigue teniendo el stop al -8% del precio de ENTRADA, es decir, ya no protege
nada de lo ganado. La barrera movil convierte la regla en "aguanta mientras
acompanie, suelta cuando pierda fuerza", que es exactamente la logica de salida
que se queria modelar.

La operacion termina en la PRIMERA barrera que se toque. Aqui se anade una
cuarta, obligatoria por coherencia con `data/pit.py`:

  * barrera de delisting = la accion deja de cotizar dentro de la ventana; sale
    con su retorno terminal (quiebra = perdida severa, OPA = prima).

Sin esa cuarta barrera, el sesgo de supervivencia volveria a entrar por la
etiqueta exactamente igual que con el `shift` ingenuo.

Decision conservadora y deliberada: las barreras se evaluan sobre el CIERRE
diario, no sobre el maximo/minimo intradia. Declarar que el objetivo "se toco"
porque el maximo intradia lo roz o asume que tenias una orden limitada
descansando justo ahi y que se ejecuto; es la version optimista. Con cierres, la
etiqueta refleja lo que un rebalanceo diario habria obtenido de verdad.

Corwin-Schultz
--------------
Incluir penny stocks obliga a un modelo de coste que dependa del precio y de la
liquidez: un spread de 2 centimos en una accion de 50 centimos son 4% de ida y
vuelta, mas que cualquier ventaja realista. El estimador de Corwin y Schultz
(*Journal of Finance*, 2012) recupera el spread bid-ask efectivo a partir
unicamente de los maximos y minimos diarios, que Sharadar `stocks` ya trae.
Aqui se usa para (a) excluir nombres cuyo spread estimado se come el objetivo y
(b) netear el retorno de cada operacion con su coste de ida y vuelta.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = ["TripleBarrierConfig", "triple_barrier_labels", "corwin_schultz_spread",
           "abdi_ranaldo_spread",
           "EXIT_PROFIT", "EXIT_STOP", "EXIT_VERTICAL", "EXIT_DELIST", "EXIT_NONE"]

EXIT_PROFIT = "profit_target"
EXIT_STOP = "stop_loss"
EXIT_VERTICAL = "vertical"
EXIT_DELIST = "delisting"
EXIT_NONE = "unlabelled"


@dataclass(frozen=True)
class TripleBarrierConfig:
    horizon: int = 32               # barrera vertical: ~1,5 meses de sesiones
    profit_target: float = 0.10     # +10%: barrera superior
    stop_loss: float = -0.08        # -8%: barrera inferior fija (desde la entrada)
    trail_stop: float | None = -0.08  # barrera movil desde el MAXIMO alcanzado
    min_hold: int = 1               # no se evalua salida antes de esta sesion


def triple_barrier_labels(prices: pd.DataFrame, cfg: TripleBarrierConfig,
                          listings: pd.DataFrame | None = None) -> pd.DataFrame:
    """Etiqueta de triple barrera para cada (fecha, simbolo).

    Devuelve un DataFrame indexado por (date, symbol) con:
        ret          retorno realizado en la salida (NaN si no es etiquetable)
        exit_h       sesiones hasta la salida (1..horizon)
        exit_reason  profit_target | stop_loss | vertical | delisting | unlabelled

    Implementacion vectorizada por desplazamientos: para cada k en 1..horizon se
    calcula el retorno a k sesiones y se registra la primera k en que se cruza
    una barrera. Coste O(horizon x N x T), suficiente para miles de simbolos.
    """
    px = prices.sort_index()
    T, N = px.shape
    idx = px.index
    H = int(cfg.horizon)
    if H < 1:
        raise ValueError("horizon >= 1")

    base = px.to_numpy(float)                          # (T, N)
    peak = base.copy()                                 # maximo alcanzado desde la entrada
    ret = np.full((T, N), np.nan)
    exit_h = np.full((T, N), 0, dtype=np.int32)
    reason = np.full((T, N), EXIT_NONE, dtype=object)
    decided = np.zeros((T, N), dtype=bool)

    # --- barreras de precio: primera k en que se cruza objetivo o stop -----
    for k in range(max(1, cfg.min_hold), H + 1):
        if k >= T:
            break
        fut = np.full((T, N), np.nan)
        fut[: T - k] = base[k:]
        r_k = fut / base - 1.0
        # el maximo desde la entrada se actualiza ANTES de evaluar el trailing:
        # una vela que hace nuevo maximo no puede dispararlo en la misma sesion
        peak = np.fmax(peak, np.nan_to_num(fut, nan=-np.inf))
        valid = np.isfinite(r_k) & ~decided
        hit_pt = valid & (r_k >= cfg.profit_target)
        hit_sl = valid & (r_k <= cfg.stop_loss)
        if cfg.trail_stop is not None:
            with np.errstate(invalid="ignore", divide="ignore"):
                dd_from_peak = fut / peak - 1.0        # <= 0 por construccion
            # el trailing solo actua si ya hubo ganancia: si no, manda el stop fijo
            in_profit = np.isfinite(peak) & (peak > base)
            hit_sl = hit_sl | (valid & in_profit & (dd_from_peak <= cfg.trail_stop))
        # si en la misma sesion se cruzan ambas (imposible con cierres, salvo
        # objetivo <= |stop| mal configurado) gana el stop: es la version prudente
        first_sl = hit_sl
        first_pt = hit_pt & ~hit_sl
        for mask, tag in ((first_sl, EXIT_STOP), (first_pt, EXIT_PROFIT)):
            if mask.any():
                ret[mask] = r_k[mask]
                exit_h[mask] = k
                reason[mask] = tag
                decided |= mask

    # --- barrera vertical: quien llega vivo a t+H sale al cierre de t+H ------
    if H < T:
        fut = np.full((T, N), np.nan)
        fut[: T - H] = base[H:]
        r_H = fut / base - 1.0
        vert = np.isfinite(r_H) & ~decided & np.isfinite(base)
        ret[vert] = r_H[vert]
        exit_h[vert] = H
        reason[vert] = EXIT_VERTICAL
        decided |= vert

    out = pd.DataFrame({
        "ret": ret.ravel(order="C"),
        "exit_h": exit_h.ravel(order="C"),
        "exit_reason": reason.ravel(order="C"),
    }, index=pd.MultiIndex.from_product([idx, px.columns], names=["date", "symbol"]))

    # --- barrera de delisting: coherente con data/pit.py ---------------------
    if listings is not None:
        from .pit import DelistReason

        L = listings.reindex(px.columns)
        pos = pd.Series(np.arange(T), index=idx)
        for sym in px.columns:
            if L.at[sym, "reason"] == DelistReason.ACTIVE if sym in L.index else True:
                continue
            s = px[sym].dropna()
            if s.empty:
                continue
            lv = s.index[-1]
            j = int(pos[lv])
            p_end = float(s.iloc[-1])
            terminal = p_end * (1.0 + float(L.at[sym, "delist_return"]))
            lo = max(j - H, 0)
            # fechas t con lo <= t < j: la accion muere dentro de su ventana.
            # Solo se sobreescribe si ninguna barrera de precio se cruzo ANTES de
            # la muerte; si el objetivo se toco el dia 5 y murio el dia 20, la
            # operacion ya habia salido con beneficio.
            for t in range(lo, j):
                key = (idx[t], sym)
                cur_h = int(out.at[key, "exit_h"])
                cur_reason = out.at[key, "exit_reason"]
                died_at = j - t
                if cur_reason in (EXIT_PROFIT, EXIT_STOP) and cur_h < died_at:
                    continue
                p_t = float(px.iat[t, px.columns.get_loc(sym)])
                if not np.isfinite(p_t) or p_t <= 0:
                    continue
                out.at[key, "ret"] = terminal / p_t - 1.0
                out.at[key, "exit_h"] = died_at
                out.at[key, "exit_reason"] = EXIT_DELIST
            # despues de morir no hay observacion
            after = idx[j:]
            if len(after):
                out.loc[(after, sym), ["ret", "exit_h", "exit_reason"]] = [np.nan, 0, EXIT_NONE]

    return out


def corwin_schultz_spread(highs: pd.DataFrame, lows: pd.DataFrame,
                          window: int = 21) -> pd.DataFrame:
    """Spread bid-ask efectivo estimado (fraccion del precio), Corwin-Schultz 2012.

    Para dos dias consecutivos:
        beta  = ln(H1/L1)^2 + ln(H2/L2)^2
        gamma = ln(max(H1,H2)/min(L1,L2))^2
        alpha = (sqrt(2 beta) - sqrt(beta)) / (3 - 2 sqrt 2) - sqrt(gamma / (3 - 2 sqrt 2))
        S     = 2 (e^alpha - 1) / (1 + e^alpha)
    Los valores negativos se truncan a cero (recomendacion de los autores) y se
    promedian en una ventana movil para reducir ruido. Devuelve el spread de
    IDA, no de ida y vuelta: para netear una operacion completa se usa 2*S... o
    mas exactamente S (comprar en el ask, vender en el bid = un spread entero).
    """
    H = highs.sort_index().astype(float)
    Lw = lows.reindex_like(H).astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        hl = np.log(H / Lw) ** 2
        beta = hl + hl.shift(-1)
        hh = np.maximum(H, H.shift(-1))
        ll = np.minimum(Lw, Lw.shift(-1))
        gamma = np.log(hh / ll) ** 2
        k = 3.0 - 2.0 * np.sqrt(2.0)
        alpha = (np.sqrt(2.0 * beta) - np.sqrt(beta)) / k - np.sqrt(gamma / k)
        spread = 2.0 * (np.exp(alpha) - 1.0) / (1.0 + np.exp(alpha))
    spread = spread.clip(lower=0.0)
    return spread.rolling(window, min_periods=max(5, window // 3)).mean()


def abdi_ranaldo_spread(closes: pd.DataFrame, highs: pd.DataFrame, lows: pd.DataFrame,
                        window: int = 63) -> pd.DataFrame:
    """Spread efectivo estimado con cierre, maximo y minimo (Abdi-Ranaldo 2017, RFS).

        eta_t = (ln H_t + ln L_t) / 2           (punto medio del rango)
        S^2   = 4 E[(c_t - eta_t)(c_t - eta_{t+1})]

    con c_t = ln(cierre). La esperanza se estima con la media movil de `window`
    sesiones y los valores negativos se truncan a cero antes de la raiz, como
    proponen los autores. En acciones poco liquidas es mas preciso que
    Corwin-Schultz porque usa el cierre, que es donde se concentran las
    operaciones. Devuelve el spread COMPLETO (comprar en el ask y vender en el
    bid): es el coste de ida y vuelta por spread. Precios NOMINALES (sin ajustar):
    el ajuste por dividendos desplaza el cierre respecto al rango del dia.
    """
    C = np.log(closes.sort_index().astype(float))
    H = np.log(highs.reindex_like(C).astype(float))
    L = np.log(lows.reindex_like(C).astype(float))
    eta = (H + L) / 2.0
    prod = (C - eta) * (C - eta.shift(-1))
    m = prod.rolling(window, min_periods=max(10, window // 3)).mean()
    return np.sqrt((4.0 * m).clip(lower=0.0))
