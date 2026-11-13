"""Universo unificado de varias plataformas, con ejecutabilidad y coste.

El problema que este modulo hace visible
-----------------------------------------
"Analiza todas las acciones de Alpaca, eToro e Interactive Brokers y compra las
3-5 mejores" da por supuestas tres cosas que no son ciertas, y el sistema falla
en silencio si no se modelan:

**1. No toda accion se puede comprar en toda plataforma.** Si el ranking elige
un valor que solo cotiza en eToro, no se puede ejecutar desde Alpaca. Un
universo sin etiqueta de ejecutabilidad produce recomendaciones imposibles de
operar, que es peor que no producirlas: pareces tener una oportunidad y no la
tienes.

**2. El capital NO es fungible entre brokers.** No tienes un saldo unico: tienes
una cuenta en cada plataforma. Si la mejor oportunidad esta solo en eToro y el
dinero esta en Alpaca, no la compras. Transferir tarda dias. Por eso el router
comprueba efectivo POR PLAZA, no en agregado.

**3. La union no es la suma.** Las tres son mayoritariamente renta variable
estadounidense: el solapamiento es enorme. Lo que aporta cada una al margen es
distinto — eToro e IBKR anaden bolsas no estadounidenses que Alpaca no cubre.

Y la cuarta, que es la que decide en la practica
------------------------------------------------
**El coste de ida y vuelta depende de la plaza y del tamano.** Con posiciones
pequenas una comision fija es demoledora: 2 $ de apertura mas 2 $ de cierre
sobre una posicion de 25 $ son el 16% antes de empezar, mas la conversion de
divisa. Ninguna ventaja realista sobrevive a eso. `round_trip_cost` calcula el
coste efectivo y `min_viable_notional` devuelve el tamano por debajo del cual
operar en esa plaza no tiene sentido.

Los parametros de coste son DATOS, no constantes incrustadas: varian por pais,
entidad legal y bolsa, y el usuario debe verificarlos contra su propia cuenta
antes de confiar en ellos.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

__all__ = ["VenueSpec", "ALPACA", "ETORO_EU", "ETORO_US", "IBKR", "LIGHTYEAR_EU",
           "TRADING212", "KNOWN_VENUES", "VenueUniverse", "build_union", "round_trip_cost",
           "min_viable_notional", "canonical_symbol", "lightyear_universe",
           "sec_otc_universe", "etoro_universe", "YAHOO_SUFFIX"]


@dataclass(frozen=True)
class VenueSpec:
    """Caracteristicas operativas de una plaza. Verificar contra la cuenta propia."""

    name: str
    commission_open: float = 0.0        # comision fija por operacion de apertura
    commission_close: float = 0.0
    commission_pct: float = 0.0         # comision proporcional
    spread_bps: float = 5.0             # spread tipico estimado
    fx_conversion_pct: float = 0.0      # conversion de divisa, por conversion
    fractional: bool = True
    can_execute: bool = True
    can_provide_history: bool = True
    bulk_history_viable: bool = True    # ¿se puede bajar el historico masivo?
    notes: str = ""
    # (v1.6) comision por accion con minimo y maximo por orden (modelo IBKR) y
    # minimo/maximo de una comision proporcional (modelo Lightyear)
    per_share: float = 0.0
    min_commission: float = 0.0         # por orden (apertura y cierre por separado)
    max_commission_pct: float = 0.0     # tope por orden como fraccion del importe (0 = sin tope)
    max_commission: float = 0.0         # tope por orden en dinero (0 = sin tope)
    otc: bool = False                   # deja ABRIR posiciones en OTC (penny stocks no cotizados)
    execution: str = "api"              # api | manual (sin API de ordenes)
    verified: str = ""                  # que esta verificado y donde (fecha)


# Especificaciones de trabajo (revisadas el 2026-11-13). NO son verdad revelada:
# dependen del pais, de la entidad legal y del plan; verificar en la cuenta propia.
ALPACA = VenueSpec(
    name="alpaca", commission_open=0.0, commission_close=0.0, spread_bps=5.0,
    fractional=True, bulk_history_viable=True,
    notes="sin comision en renta variable US; solo bolsas estadounidenses; OTC solo "
          "para cerrar posiciones; fraccionable si cotiza > 1 USD y > 10.000 acciones/dia, "
          "el resto por acciones enteras",
    verified="docs.alpaca.markets y alpaca.markets/support (fraccionables, OTC), nov-2026",
)

ETORO_US = VenueSpec(
    name="etoro_us", commission_open=0.0, commission_close=0.0, spread_bps=10.0,
    fractional=True, bulk_history_viable=False,
    notes="entidad estadounidense: sin comision en acciones; 1000 velas por peticion",
)

ETORO_EU = VenueSpec(
    name="etoro_eu", commission_open=1.0, commission_close=1.0, spread_bps=10.0,
    fx_conversion_pct=0.0075, fractional=True, bulk_history_viable=False,
    notes="entidad europea: ~1 USD por operacion (fuentes de terceros; antes 2 USD en "
          "este fichero) y conversion de divisa; demoledor con posiciones pequenas. "
          "API publica con entorno demo; ejecucion en execution/etoro.py",
    verified="comision de terceros (sin pagina oficial consultable), nov-2026",
)

IBKR = VenueSpec(
    name="ibkr", per_share=0.0035, min_commission=0.35, max_commission_pct=0.01,
    spread_bps=4.0, fractional=True, bulk_history_viable=False, otc=True,
    notes="IBKR Pro escalonado: 0,0035 USD/accion, minimo 0,35 USD y maximo 1 % del "
          "importe por orden, mas tasas de bolsa/regulatorias. Mayor universo: OTC "
          "(OTCQX/OTCQB/Pink con permiso 'United States (Penny Stocks)'; no Caveat Emptor "
          "ni Grey). El ritmo de su API hace inviable la ingesta masiva",
    verified="interactivebrokers.ie commissions-home y ibkb microcap, nov-2026; si el "
             "maximo del 1 % prevalece sobre el minimo NO esta verificado: se cobra el "
             "minimo (lo conservador)",
)

LIGHTYEAR_EU = VenueSpec(
    name="lightyear", commission_pct=0.001, min_commission=0.10, max_commission=1.0,
    spread_bps=10.0, fx_conversion_pct=0.0035, fractional=True, can_provide_history=False,
    bulk_history_viable=False, execution="manual",
    notes="cuenta UE: 0,1 % por orden en acciones US (min 0,10 USD, max 1 USD), 1 EUR en "
          "acciones UE, cambio 0,35 %. SIN API: universo desde su sitemap publico y "
          "ordenes manuales (execution/manual.py). Su parte US se custodia en Alpaca",
    verified="minimo 0,10 USD en lightyear.com/en-eu/stocks; 0,1 %, tope y cambio de "
             "investingintheweb (jul-2026)",
)

TRADING212 = VenueSpec(
    name="trading212", commission_open=0.0, commission_close=0.0, spread_bps=15.0,
    fx_conversion_pct=0.0015, fractional=True, can_provide_history=False,
    bulk_history_viable=False, otc=True,
    notes="sin comision; cambio 0,15 %; algunas OTC; API v0 beta (en real solo ordenes "
          "de mercado, sin ordenes por importe, no idempotente, sin historico). Sin "
          "adaptador de ejecucion todavia",
    verified="fuentes de terceros, nov-2026",
)

KNOWN_VENUES = {v.name: v for v in (ALPACA, ETORO_US, ETORO_EU, IBKR, LIGHTYEAR_EU,
                                       TRADING212)}


# ------------------------------------------------------------------ costes
def _commission(notional: float, venue: VenueSpec, fixed: float, price: float | None) -> float:
    """Comision de UNA orden (apertura o cierre)."""
    c = fixed + notional * venue.commission_pct
    if venue.per_share > 0:
        shares = notional / price if price and price > 0 else notional / 10.0
        c += venue.per_share * shares
        if venue.max_commission_pct > 0:
            c = min(c, venue.max_commission_pct * notional)
    if venue.max_commission > 0:
        c = min(c, venue.max_commission)
    # el minimo se aplica al final: es lo conservador (ver IBKR.verified)
    return max(c, venue.min_commission) if (venue.min_commission > 0 and notional > 0) else c


def round_trip_cost(notional: float, venue: VenueSpec, fx: bool = False,
                    price: float | None = None) -> dict:
    """Coste total de abrir y cerrar una posicion, en dinero y en porcentaje.

    Se cuenta el spread UNA vez (compras en el ask, vendes en el bid = un spread
    completo) y la conversion de divisa DOS veces si aplica (entrada y salida).
    `price` solo importa con comision por accion; sin el se suponen 10 USD.
    """
    if notional <= 0:
        return {"total": 0.0, "pct": 0.0, "breakdown": {}}
    commission = (_commission(notional, venue, venue.commission_open, price)
                  + _commission(notional, venue, venue.commission_close, price))
    spread = notional * venue.spread_bps / 10_000.0
    fx_cost = notional * venue.fx_conversion_pct * 2 if fx else 0.0
    total = commission + spread + fx_cost
    return {
        "total": round(total, 4),
        "pct": round(total / notional, 6),
        "breakdown": {"comision": round(commission, 4), "spread": round(spread, 4),
                      "divisa": round(fx_cost, 4)},
    }


def min_viable_notional(venue: VenueSpec, max_cost_pct: float = 0.02,
                        fx: bool = False, price: float | None = None) -> float:
    """Tamano minimo para que el coste de ida y vuelta no supere `max_cost_pct`.

    Por debajo de este importe, operar en esa plaza es tirar dinero: la comision
    fija no se diluye y se come cualquier ventaja esperada realista. Devuelve
    infinito si el coste proporcional ya supera el limite por si solo. Con
    comisiones con minimo y tope no hay forma cerrada: se busca por biseccion.
    """
    prop = venue.spread_bps / 10_000.0 + venue.commission_pct * 2 \
        + (venue.fx_conversion_pct * 2 if fx else 0.0)
    if prop >= max_cost_pct:
        return float("inf")
    fixed = venue.commission_open + venue.commission_close
    simple = (venue.per_share == 0 and venue.min_commission == 0 and venue.max_commission == 0)
    if simple:
        return 0.0 if fixed <= 0 else fixed / (max_cost_pct - prop)

    def ok(n):
        return round_trip_cost(n, venue, fx=fx, price=price)["pct"] <= max_cost_pct

    if ok(1.0):
        return 0.0
    lo, hi = 1.0, 1e7
    if not ok(hi):
        return float("inf")
    for _ in range(80):
        mid = (lo + hi) / 2
        lo, hi = (lo, mid) if ok(mid) else (mid, hi)
    return hi


# ------------------------------------------------------------------ universo
@dataclass
class VenueUniverse:
    """Union de simbolos con la lista de plazas donde cada uno se puede operar."""

    venues: dict = field(default_factory=dict)          # name -> VenueSpec
    membership: pd.DataFrame = field(default_factory=pd.DataFrame)  # symbol x venue (bool)

    @property
    def symbols(self) -> list[str]:
        return list(self.membership.index)

    def executable_on(self, symbol: str) -> list[str]:
        """Plazas donde ese simbolo se puede COMPRAR (no solo consultar)."""
        if symbol not in self.membership.index:
            return []
        row = self.membership.loc[symbol]
        return [v for v in row.index
                if bool(row[v]) and self.venues[v].can_execute]

    def tradable_symbols(self) -> list[str]:
        return [s for s in self.symbols if self.executable_on(s)]

    def cheapest_venue(self, symbol: str, notional: float, fx_venues=()) -> str | None:
        """Plaza mas barata para ese importe. None si no es operable en ninguna."""
        opts = self.executable_on(symbol)
        if not opts:
            return None
        return min(opts, key=lambda v: round_trip_cost(
            notional, self.venues[v], fx=v in fx_venues)["total"])

    def summary(self) -> dict:
        m = self.membership
        only = {}
        for v in m.columns:
            others = [c for c in m.columns if c != v]
            only[v] = int((m[v] & ~m[others].any(axis=1)).sum()) if others else int(m[v].sum())
        return {
            "total_union": int(len(m)),
            "por_plaza": {v: int(m[v].sum()) for v in m.columns},
            "exclusivos_de_cada_plaza": only,
            "en_las_tres": int(m.all(axis=1).sum()) if m.shape[1] > 1 else int(len(m)),
            "operables": int(len(self.tradable_symbols())),
        }


def _normalise(sym: str) -> str:
    """Normaliza un ticker para poder cruzar plazas.

    Deliberadamente CONSERVADORA: prefiere no cruzar dos simbolos a cruzar mal dos
    instrumentos, porque un cruce erroneo mete precios de uno en el historico de
    otro y eso no lo detecta ningun test estadistico posterior.

    (v1.6) El sufijo de bolsa ya NO se recorta: `SAN.MC` (Santander en Madrid, en
    euros) y `SAN` (su ADR en Nueva York, en dolares) son instrumentos distintos,
    con otra divisa, otro horario y otro precio; la version anterior los fundia en
    uno. Solo se quitan los designadores de EE. UU. (`.US`, `-US`), que no cambian
    de instrumento. Para crear el simbolo canonico desde el formato de cada
    plataforma, `canonical_symbol`.
    """
    s = str(sym).upper().strip()
    for suf in (".US", "-US"):
        if s.endswith(suf) and len(s) > len(suf) + 1:
            return s[: -len(suf)]
    return s


# bolsa (codigo de la plataforma) -> sufijo de Yahoo. "" = EE. UU.; None = sin
# historico en Yahoo (no se puede analizar con el panel actual).
YAHOO_SUFFIX: dict[str, str | None] = {
    "NASDAQ": "", "NYSE": "", "AMEX": "", "OTC": "", "TXSE": "", "ARCA": "", "BATS": "",
    "LSE": ".L", "AIM": ".L", "XETRA": ".DE", "XSTO": ".ST", "PAR": ".PA", "XSWX": ".SW",
    "XOSL": ".OL", "AEX": ".AS", "BME": ".MC", "CSE": ".CO", "XHEL": ".HE", "VSE": ".VI",
    "BSE": ".BR", "BUX": ".BD", "XLIT": ".VS", "XTAL": ".TL", "XLIS": ".LS", "XRIS": ".RG",
    "IRE": ".IR", "XGAT": None,
}


def canonical_symbol(symbol: str, exchange: str) -> str | None:
    """Simbolo canonico (formato Yahoo) a partir de simbolo + bolsa de la plataforma.

    EE. UU.: el ticker con las clases en guion (BRK.B / "BRK B" -> BRK-B).
    Resto: ticker + sufijo de bolsa (SAN + BME -> SAN.MC). None si la bolsa no
    tiene equivalente con historico (p. ej. Tradegate).
    """
    suf = YAHOO_SUFFIX.get(str(exchange).upper())
    if suf is None:
        return None
    s = str(symbol).upper().strip().replace(" ", "-")
    if suf == "":
        s = s.replace(".", "-")
    return s + suf


def lightyear_universe(sitemap: str | None = None) -> pd.DataFrame:
    """Acciones de Lightyear desde su sitemap PUBLICO de acciones.

    Lightyear no tiene API. Su `robots.txt` permite rastrear las paginas de
    acciones y publica el sitemap con una URL por accion (`/en/stock/SIMBOLO:BOLSA`).
    `sitemap`: ruta a un XML ya descargado, o None para descargarlo de
    https://lightyear.com/en/stocks/sitemap.xml. Devuelve symbol, exchange, canonical.
    """
    import re
    if sitemap and not str(sitemap).startswith("http"):
        from pathlib import Path
        xml = Path(sitemap).read_text()
    else:
        import urllib.request
        url = sitemap or "https://lightyear.com/en/stocks/sitemap.xml"
        req = urllib.request.Request(url, headers={"User-Agent": "swarmtrader-universe/1.6"})
        with urllib.request.urlopen(req, timeout=60) as r:          # noqa: S310
            xml = r.read().decode("utf-8")
    rows = []
    for loc in re.findall(r"<loc>([^<]+)</loc>", xml):
        m = re.search(r"/stock/([^/]+)$", loc)
        if not m:
            continue
        sym, _, exch = __import__("urllib.parse").parse.unquote(m.group(1)).rpartition(":")
        if sym and exch:
            rows.append((sym, exch, canonical_symbol(sym, exch)))
    return pd.DataFrame(rows, columns=["symbol", "exchange", "canonical"]).drop_duplicates()


def sec_otc_universe(tickers_path: str) -> pd.DataFrame:
    """Acciones OTC de empresas que presentan informes a la SEC.

    Fuente: `company_tickers_exchange.json` de la SEC (exchange == "OTC"). Quedan
    fuera las OTC sin informacion publica, que IBKR tampoco deja abrir (Caveat
    Emptor / Grey Market) y que son las de mayor riesgo de manipulacion.
    """
    from .fundamentals import load_ticker_map
    t = load_ticker_map(tickers_path)
    t = t[t["exchange"].str.upper() == "OTC"]
    return pd.DataFrame({"symbol": t["ticker"], "exchange": "OTC", "canonical": t["ticker"]})


def etoro_universe(client=None, stock_type_id: int = 5) -> pd.DataFrame:
    """Acciones de eToro desde su API publica (`/api/v1/market-data/instruments`).

    Solo el tipo "acciones" (`instrumentTypeID` 5 en la API; verificar con
    `/market-data/instrument-types`): asi no se cuelan criptomonedas con el mismo
    ticker que una accion (la colision de 178 simbolos de versiones anteriores).
    `client`: un `execution.etoro.EToroClient` (claves ETORO_API_KEY y
    ETORO_USER_KEY). Devuelve symbol, exchange, canonical, instrument_id.
    """
    from ..execution.etoro import EToroClient
    c = client or EToroClient()
    ex = {e.get("exchangeID"): (e.get("exchangeDescription") or "")
          for e in c.get("/api/v1/market-data/exchanges").get("exchangeInfo", [])}
    data = c.get("/api/v1/market-data/instruments",
                 {"instrumentTypeIds": str(stock_type_id)}).get("instrumentDisplayDatas", [])
    rows = []
    for d in data:
        if d.get("isInternalInstrument"):
            continue
        sym = str(d.get("symbolFull", ""))
        base, _, suf = sym.partition(".")
        exch = _ETORO_EXCHANGE.get(suf.upper(), "") if suf else "NASDAQ"
        rows.append((sym, ex.get(d.get("exchangeID"), exch), canonical_symbol(base, exch)
                     if exch else None, d.get("instrumentID")))
    return pd.DataFrame(rows, columns=["symbol", "exchange", "canonical", "instrument_id"])


# sufijo de eToro -> bolsa (los de EE. UU. van sin sufijo; verificar con /exchanges)
_ETORO_EXCHANGE = {"L": "LSE", "DE": "XETRA", "PA": "PAR", "MC": "BME", "MI": "",
                   "AS": "AEX", "ST": "XSTO", "OL": "XOSL", "CO": "CSE", "HE": "XHEL",
                   "ZU": "XSWX", "LSB": "LSE", "US": "NASDAQ"}


def build_union(venue_symbols: dict, venues: dict | None = None) -> VenueUniverse:
    """Construye la union etiquetando en que plazas vive cada simbolo.

    `venue_symbols`: {nombre_plaza: [tickers]}.
    """
    venues = venues or {k: KNOWN_VENUES.get(k, VenueSpec(name=k)) for k in venue_symbols}
    norm = {v: {_normalise(s) for s in syms} for v, syms in venue_symbols.items()}
    all_syms = sorted(set().union(*norm.values())) if norm else []
    data = {v: [s in norm[v] for s in all_syms] for v in norm}
    membership = pd.DataFrame(data, index=pd.Index(all_syms, name="symbol"))
    return VenueUniverse(venues=venues, membership=membership)
