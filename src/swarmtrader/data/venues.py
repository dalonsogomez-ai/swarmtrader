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

__all__ = ["VenueSpec", "ALPACA", "ETORO_EU", "ETORO_US", "IBKR", "VenueUniverse",
           "build_union", "round_trip_cost", "min_viable_notional"]


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


# Especificaciones de trabajo. NO son verdad revelada: el coste de eToro depende
# del pais y de la entidad legal, y el usuario debe confirmarlo en su cuenta.
ALPACA = VenueSpec(
    name="alpaca", commission_open=0.0, commission_close=0.0, spread_bps=5.0,
    fractional=True, bulk_history_viable=True,
    notes="sin comision en renta variable US; solo bolsas estadounidenses",
)

ETORO_US = VenueSpec(
    name="etoro_us", commission_open=0.0, commission_close=0.0, spread_bps=10.0,
    fractional=True, bulk_history_viable=False,
    notes="entidad estadounidense: sin comision en acciones; 1000 velas por peticion",
)

ETORO_EU = VenueSpec(
    name="etoro_eu", commission_open=2.0, commission_close=2.0, spread_bps=10.0,
    fx_conversion_pct=0.0075, fractional=True, bulk_history_viable=False,
    notes="entidad europea: comision fija por operacion y conversion de divisa; "
          "demoledor con posiciones pequenas",
)

IBKR = VenueSpec(
    name="ibkr", commission_open=1.0, commission_close=1.0, spread_bps=4.0,
    fractional=True, bulk_history_viable=False,
    notes="mejor ejecucion y mayor universo, pero el ritmo de su API hace "
          "inviable la ingesta masiva (60 solicitudes historicas / 10 min)",
)

KNOWN_VENUES = {v.name: v for v in (ALPACA, ETORO_US, ETORO_EU, IBKR)}


# ------------------------------------------------------------------ costes
def round_trip_cost(notional: float, venue: VenueSpec, fx: bool = False) -> dict:
    """Coste total de abrir y cerrar una posicion, en euros y en porcentaje.

    Se cuenta el spread UNA vez (compras en el ask, vendes en el bid = un spread
    completo) y la conversion de divisa DOS veces si aplica (entrada y salida).
    """
    if notional <= 0:
        return {"total": 0.0, "pct": 0.0, "breakdown": {}}
    commission = venue.commission_open + venue.commission_close
    commission += notional * venue.commission_pct * 2
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
                        fx: bool = False) -> float:
    """Tamano minimo para que el coste de ida y vuelta no supere `max_cost_pct`.

    Por debajo de este importe, operar en esa plaza es tirar dinero: la comision
    fija no se diluye y se come cualquier ventaja esperada realista.
    Devuelve infinito si el coste proporcional ya supera el limite por si solo.
    """
    prop = venue.spread_bps / 10_000.0 + venue.commission_pct * 2 \
        + (venue.fx_conversion_pct * 2 if fx else 0.0)
    fixed = venue.commission_open + venue.commission_close
    if prop >= max_cost_pct:
        return float("inf")
    if fixed <= 0:
        return 0.0
    return fixed / (max_cost_pct - prop)


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

    Cada plataforma usa convenciones distintas (puntos, guiones, sufijos de
    bolsa). Esta normalizacion es deliberadamente CONSERVADORA: prefiere no
    cruzar dos simbolos a cruzar mal dos empresas distintas, porque un cruce
    erroneo mete precios de una empresa en el historico de otra y eso no lo
    detecta ningun test estadistico posterior.
    """
    s = str(sym).upper().strip()
    for sep in (".", "-"):
        if sep in s and len(s.split(sep)[0]) >= 2:
            head, tail = s.split(sep, 1)
            # sufijos de bolsa conocidos: se recortan; clases de accion: se conservan
            if tail in ("US", "NAS", "NYSE", "L", "DE", "PA", "MC", "MI", "AS", "SW"):
                return head
    return s


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
