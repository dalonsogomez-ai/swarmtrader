"""Operaciones declaradas por cargos publicos de EE. UU. (STOCK Act).

Que dice la evidencia antes de escribir el codigo
-------------------------------------------------
La idea intuitiva —"copiar lo que compran los congresistas"— esta contradicha
por la mejor evidencia disponible:

  * Ziobrowski et al. (2004, 2011): el Senado batia al mercado ~12% anual y la
    Camara ~6%. Pero ese periodo es ANTERIOR a la STOCK Act (2012).
  * Eggers y Hainmueller (2013, *Journal of Politics*): con datos corregidos de
    2004-2008, las carteras del Congreso rindieron 2-3% anual POR DEBAJO de un
    indexado pasivo.
  * Belmont, Sacerdote, Sehgal y Van Hoek (2022): sin rendimiento superior en
    2012-2020, ni en agregado ni entre los senadores acusados de operar con
    informacion privilegiada.
  * Chen y Sacerdote (2026), hasta 2023: por debajo o, como mucho, igualando.
  * La excepcion es estrecha: Wei y Zhou (NBER) documentan rendimiento anormal
    en LIDERES del Congreso, y solo a partir del momento en que lo son.

Por que entonces existe este modulo
-----------------------------------
Porque el sistema no necesita creer en una senal para incorporarla: necesita
poder MEDIRLA. Este modulo convierte las declaraciones en factores
point-in-time que pasan por las mismas puertas (IC OOS, SPA, PBO) que
`momentum` o `reversal`. Si no aportan, su peso sera cero y quedara registrado.
Esa es la diferencia entre anadir una fuente y anadir una creencia.

El detalle que decide si esto es senal o fuga
---------------------------------------------
La STOCK Act da **45 dias naturales** de plazo (30 desde la notificacion) y la
multa por retraso son 200 $ que rara vez se aplican, asi que muchas llegan aun
mas tarde. Consecuencia para un horizonte de 21-32 sesiones (30-45 dias
naturales): **cuando la operacion se hace publica, la ventana de tenencia ya ha
transcurrido entera**. Copiar la operacion es imposible, no dificil.

Por eso aqui la fecha que gobierna TODO es `disclosure_date`, nunca
`transaction_date`. Usar la fecha de transaccion produce un backtest
espectacular y completamente falso: estarias operando con informacion que no
existia. `tests/test_congress.py::test_transaction_date_creates_fake_alpha`
demuestra la magnitud del espejismo en lugar de limitarse a advertirlo.

La hipotesis comprobable que queda no es "copiar al congresista", es: **¿reacciona
el precio a la DIVULGACION?** Eso si ocurre despues de la fecha de divulgacion y
si es explotable en principio.

Fuentes (gratuitas y oficiales)
-------------------------------
  * Camara: disclosures-clerk.house.gov (descarga masiva anual en ZIP/XML)
  * Senado: efdsearch.senate.gov
Ambas permiten descarga masiva en formato legible por maquina. Los agregadores
comerciales (Capitol Trades, Quiver, Unusual Whales) revenden esto mismo.

Limitaciones que hay que tener presentes al interpretar el resultado
--------------------------------------------------------------------
  * El importe se declara en RANGOS ($1.001-$15.000, etc.), no en cifras
    exactas: cualquier ponderacion por tamano usa el punto medio del rango y
    arrastra ese error.
  * No se declara precio ni numero de acciones.
  * Solo operaciones > $1.000.
  * Incluye operaciones del conyuge e hijos dependientes, que pueden no tener
    ninguna relacion con informacion del cargo.
  * Muchas declaraciones llegan fuera de plazo, lo que sesga la muestra hacia
    los que cumplen.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = ["CongressConfig", "AMOUNT_RANGES", "range_midpoint", "load_disclosures",
           "congress_features", "CONGRESS_FEATURES"]

# Rangos oficiales del PTR. El punto medio es una aproximacion declarada, no un
# dato: el error que introduce debe recordarse al leer cualquier ponderacion.
AMOUNT_RANGES = {
    "$1,001 - $15,000": (1_001, 15_000),
    "$15,001 - $50,000": (15_001, 50_000),
    "$50,001 - $100,000": (50_001, 100_000),
    "$100,001 - $250,000": (100_001, 250_000),
    "$250,001 - $500,000": (250_001, 500_000),
    "$500,001 - $1,000,000": (500_001, 1_000_000),
    "$1,000,001 - $5,000,000": (1_000_001, 5_000_000),
    "$5,000,001 - $25,000,000": (5_000_001, 25_000_000),
    "$25,000,001 - $50,000,000": (25_000_001, 50_000_000),
    "Over $50,000,000": (50_000_001, 100_000_000),
}

CONGRESS_FEATURES = ("pol_net_buy", "pol_n_filers", "pol_cluster",
                     "pol_leader_buy", "pol_days_since")


@dataclass
class CongressConfig:
    window: int = 63               # sesiones sobre las que se agrega la actividad
    extra_lag: int = 1             # colchon sobre la fecha de divulgacion
    use_transaction_date: bool = False   # SOLO para demostrar la fuga en tests
    leader_roles: tuple = ("Speaker", "Majority Leader", "Minority Leader",
                           "Committee Chair", "Ranking Member")
    min_amount: float = 1_001.0


def range_midpoint(label: str) -> float:
    """Punto medio del rango declarado. Devuelve NaN si el rango no se reconoce."""
    if not isinstance(label, str):
        return float("nan")
    key = label.strip()
    if key in AMOUNT_RANGES:
        lo, hi = AMOUNT_RANGES[key]
        return float((lo + hi) / 2.0)
    # tolerancia a variantes de formato: se extraen los numeros presentes
    nums = [float(x.replace(",", "")) for x in
            pd.Series([key]).str.findall(r"[\d,]+").iloc[0] or []]
    if len(nums) >= 2:
        return float((nums[0] + nums[1]) / 2.0)
    if len(nums) == 1:
        return float(nums[0])
    return float("nan")


def load_disclosures(path: str) -> pd.DataFrame:
    """Carga un CSV de declaraciones y normaliza sus columnas.

    Columnas esperadas (nombres flexibles, se normalizan):
        symbol / ticker
        transaction_date     fecha real de la operacion
        disclosure_date      fecha de publicacion  <-- la que gobierna
        type                 purchase | sale | exchange (se acepta 'P'/'S')
        amount               rango textual del PTR
        filer                nombre del declarante        [opcional]
        role                 cargo                        [opcional]
        chamber              house | senate               [opcional]
    """
    df = pd.read_csv(path)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    ren = {"ticker": "symbol", "transactiondate": "transaction_date",
           "disclosuredate": "disclosure_date", "transaction_type": "type",
           "owner_type": "owner", "asset_ticker": "symbol"}
    df = df.rename(columns={k: v for k, v in ren.items() if k in df.columns})

    required = {"symbol", "disclosure_date", "type"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"faltan columnas {sorted(missing)}. `disclosure_date` es OBLIGATORIA: "
            "sin ella no se puede construir un factor point-in-time, y usar "
            "`transaction_date` produciria un backtest falso (ver docstring)."
        )
    for c in ("transaction_date", "disclosure_date"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    df["symbol"] = df["symbol"].astype(str).str.upper().str.strip()
    df = df[df["symbol"].str.len().between(1, 6)]
    df["amount_mid"] = (df["amount"].map(range_midpoint) if "amount" in df.columns
                        else np.nan)
    t = df["type"].astype(str).str.lower().str[0]
    df["direction"] = np.where(t == "p", 1.0, np.where(t == "s", -1.0, 0.0))
    return df.dropna(subset=["disclosure_date"])


def _ensure_normalised(d: pd.DataFrame) -> pd.DataFrame:
    """Acepta tanto la salida de `load_disclosures` como un DataFrame crudo.

    Acoplar esta funcion a que el cargador se haya ejecutado antes era un error:
    quien construye las declaraciones desde otra fuente (un scraper propio, un
    agregador) se encontraba con un KeyError opaco.
    """
    d = d.copy()
    d.columns = [c.strip().lower().replace(" ", "_") for c in d.columns]
    if "symbol" not in d.columns and "ticker" in d.columns:
        d = d.rename(columns={"ticker": "symbol"})
    if "disclosure_date" not in d.columns:
        raise ValueError(
            "falta `disclosure_date`: es la unica fecha que hace point-in-time "
            "este factor. Ver el docstring del modulo.")
    for c in ("transaction_date", "disclosure_date"):
        if c in d.columns:
            d[c] = pd.to_datetime(d[c], errors="coerce")
    if "amount_mid" not in d.columns:
        d["amount_mid"] = (d["amount"].map(range_midpoint) if "amount" in d.columns
                           else np.nan)
    if "direction" not in d.columns:
        tt = d["type"].astype(str).str.lower().str[0] if "type" in d.columns else ""
        d["direction"] = np.where(tt == "p", 1.0, np.where(tt == "s", -1.0, 0.0))
    d["symbol"] = d["symbol"].astype(str).str.upper().str.strip()
    return d


def congress_features(disclosures: pd.DataFrame, prices: pd.DataFrame,
                      cfg: CongressConfig = CongressConfig()) -> pd.DataFrame:
    """Factores point-in-time a partir de las declaraciones.

    Cada declaracion se ancla en su `disclosure_date` + `extra_lag`, que es el
    primer dia en que la informacion estaba realmente disponible. La ventana
    agrega `cfg.window` sesiones hacia atras.

    Devuelve un DataFrame indexado por (date, symbol) listo para pasar a
    `rank._prepare(extra=...)`, donde los factores se z-scorean por fecha y
    tienen que ganarse su IC como cualquier otro.
    """
    px = prices.sort_index()
    sessions = px.index
    d = _ensure_normalised(disclosures)

    date_col = "transaction_date" if cfg.use_transaction_date else "disclosure_date"
    if cfg.use_transaction_date and "transaction_date" not in d.columns:
        raise ValueError("no hay transaction_date en los datos")
    if cfg.use_transaction_date:
        import warnings
        warnings.warn(
            "use_transaction_date=True usa informacion que NO estaba disponible "
            "en esa fecha. Solo para demostrar la fuga en tests; jamas en produccion.",
            RuntimeWarning, stacklevel=2)

    d = d.dropna(subset=[date_col])
    d = d[d["symbol"].isin(px.columns)]
    if d.empty:
        return pd.DataFrame(columns=list(CONGRESS_FEATURES))

    # ancla: primera sesion >= fecha efectiva, mas el colchon
    idx = sessions.searchsorted(d[date_col].to_numpy(), side="left") + cfg.extra_lag
    idx = np.clip(idx, 0, len(sessions) - 1)
    d = d.assign(anchor=sessions[idx])

    amount = d["amount_mid"].fillna(d["amount_mid"].median() if
                                    d["amount_mid"].notna().any() else 1.0)
    d = d.assign(
        signed_amount=d["direction"] * amount,
        is_leader=(d["role"].astype(str).isin(cfg.leader_roles)
                   if "role" in d.columns else False),
        filer_key=(d["filer"].astype(str) if "filer" in d.columns else "unknown"),
    )

    g = d.groupby(["anchor", "symbol"])
    daily = pd.DataFrame({
        "signed": g["signed_amount"].sum(),
        "n_filers": g["filer_key"].nunique(),
        "n_events": g.size(),
        "leader_signed": g.apply(
            lambda x: float((x["signed_amount"] * x["is_leader"]).sum()),
            include_groups=False),
    })
    daily.index.names = ["date", "symbol"]

    wide_signed = daily["signed"].unstack(fill_value=0.0).reindex(
        index=sessions, columns=px.columns, fill_value=0.0)
    wide_filers = daily["n_filers"].unstack(fill_value=0.0).reindex(
        index=sessions, columns=px.columns, fill_value=0.0)
    wide_leader = daily["leader_signed"].unstack(fill_value=0.0).reindex(
        index=sessions, columns=px.columns, fill_value=0.0)
    # n_events es entero: unstack con 0.0 fuerza un upcast que pandas 4 rechaza
    wide_events = daily["n_events"].unstack(fill_value=0).astype(float).reindex(
        index=sessions, columns=px.columns, fill_value=0.0)

    W = int(cfg.window)
    roll_signed = wide_signed.rolling(W, min_periods=1).sum()
    roll_filers = wide_filers.rolling(W, min_periods=1).sum()
    roll_leader = wide_leader.rolling(W, min_periods=1).sum()
    roll_events = wide_events.rolling(W, min_periods=1).sum()

    # presion neta normalizada por el total operado: adimensional y comparable
    gross = wide_signed.abs().rolling(W, min_periods=1).sum()
    net = (roll_signed / gross.replace(0.0, np.nan)).fillna(0.0)

    # sesiones desde la ultima divulgacion: la informacion se degrada con el tiempo
    had = (wide_events > 0)
    order = pd.DataFrame(np.arange(len(sessions))[:, None].repeat(len(px.columns), 1),
                         index=sessions, columns=px.columns)
    last = order.where(had).ffill()
    days_since = (order - last).fillna(float(W))

    out = pd.DataFrame({
        "pol_net_buy": net.stack(future_stack=True),
        "pol_n_filers": roll_filers.stack(future_stack=True),
        # agrupamiento: varios declarantes distintos sobre el mismo valor. Es la
        # unica dimension donde la literatura deja algo de margen, porque un
        # movimiento coordinado es mas dificil de atribuir al azar que uno suelto
        "pol_cluster": (roll_filers / roll_events.replace(0, np.nan)).fillna(0.0)
                        .stack(future_stack=True) * (roll_filers > 1).stack(future_stack=True),
        "pol_leader_buy": np.sign(roll_leader).stack(future_stack=True),
        "pol_days_since": days_since.clip(upper=W).stack(future_stack=True),
    })
    out.index.names = ["date", "symbol"]
    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
