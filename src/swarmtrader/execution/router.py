"""Router multi-plaza. Implementa `BrokerAdapter`, asi que el ciclo no cambia.

Decision de diseno principal
----------------------------
`VenueRouter` implementa la MISMA interfaz que un broker suelto. Eso significa
que `cycle.py`, `guards.py` y todo lo demas funcionan sin tocar una linea: el
ciclo cree que habla con un broker y en realidad habla con tres. Si en vez de
esto se hubiera metido logica de plazas dentro del ciclo, cada regla de riesgo
habria que duplicarla por plaza.

Las tres reglas que el router impone y que un agregador ingenuo se salta
-----------------------------------------------------------------------
**1. Ejecutabilidad.** Un simbolo solo se enruta a una plaza que realmente lo
ofrezca. Si ninguna lo ofrece, se devuelve `rejected` con el motivo, nunca se
ignora en silencio.

**2. Efectivo POR PLAZA, no agregado.** El capital no es fungible entre
brokers: transferir tarda dias. Tener 80 $ en Alpaca y 20 $ en eToro NO es
tener 100 $ para una posicion de 60 $ en eToro. El router comprueba el efectivo
de la plaza concreta y, si no llega, lo dice.

**3. Umbral de coste.** Cada plaza tiene un tamano minimo por debajo del cual el
coste de ida y vuelta se come cualquier ventaja realista. El router se NIEGA a
enrutar por debajo de ese umbral en vez de ejecutar una operacion que nace
perdiendo. Es la regla que impide que anadir plazas empeore el sistema.

Posiciones del mismo simbolo en varias plazas se agregan para la vista
consolidada (es la exposicion real al riesgo) pero se liquidan plaza a plaza.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..data.venues import VenueSpec, VenueUniverse, min_viable_notional, round_trip_cost
from .base import BrokerAdapter, OrderResult, Position

__all__ = ["VenueRouter", "RoutingDecision"]


@dataclass
class RoutingDecision:
    symbol: str
    venue: str | None
    reason: str
    cost_pct: float = 0.0


@dataclass
class VenueRouter(BrokerAdapter):
    """Varias plazas tras una sola interfaz de broker."""

    brokers: dict                       # nombre_plaza -> BrokerAdapter
    universe: VenueUniverse
    fx_venues: tuple = ()               # plazas que cobran conversion de divisa
    max_cost_pct: float = 0.02          # coste maximo tolerado de ida y vuelta
    routing_log: list = field(default_factory=list)

    # ------------------------------------------------------------ consolidado
    @property
    def equity(self) -> float:
        return float(sum(b.equity for b in self.brokers.values()))

    @property
    def cash(self) -> float:
        return float(sum(b.cash for b in self.brokers.values()))

    def cash_by_venue(self) -> dict:
        return {v: float(b.cash) for v, b in self.brokers.items()}

    def positions(self) -> dict:
        """Vista agregada: la exposicion al riesgo es la suma, viva donde viva."""
        merged: dict[str, Position] = {}
        for pos in (p for b in self.brokers.values() for p in b.positions().values()):
            cur = merged.get(pos.symbol)
            if cur is None:
                merged[pos.symbol] = pos
            else:
                qty = cur.qty + pos.qty
                mv = cur.market_value + pos.market_value
                avg = ((cur.avg_price * cur.qty + pos.avg_price * pos.qty) / qty
                       if qty else cur.avg_price)
                merged[pos.symbol] = Position(pos.symbol, qty, avg, mv)
        return merged

    def positions_by_venue(self) -> dict:
        return {v: b.positions() for v, b in self.brokers.items()}

    def last_price(self, symbol: str) -> float:
        for v in self._venues_for(symbol) or list(self.brokers):
            try:
                p = float(self.brokers[v].last_price(symbol))
                if p > 0:
                    return p
            except Exception:
                continue
        raise KeyError(f"ninguna plaza tiene precio para {symbol}")

    # ------------------------------------------------------------ enrutado
    def _venues_for(self, symbol: str) -> list[str]:
        return [v for v in self.universe.executable_on(symbol) if v in self.brokers]

    def route(self, symbol: str, notional: float, side: str = "buy") -> RoutingDecision:
        """Elige plaza aplicando las tres reglas. No ejecuta nada."""
        options = self._venues_for(symbol)
        if not options:
            return RoutingDecision(symbol, None,
                                   "no operable en ninguna plaza conectada")

        viable = []
        for v in options:
            spec: VenueSpec = self.universe.venues[v]
            fx = v in self.fx_venues
            cost = round_trip_cost(notional, spec, fx=fx)
            if cost["pct"] > self.max_cost_pct:
                minimo = min_viable_notional(spec, self.max_cost_pct, fx=fx)
                self.routing_log.append(
                    f"{symbol}: {v} descartada, coste {cost['pct']:.1%} > "
                    f"{self.max_cost_pct:.0%} (minimo viable {minimo:,.0f} $)")
                continue
            if side == "buy" and self.brokers[v].cash < notional:
                self.routing_log.append(
                    f"{symbol}: {v} descartada, efectivo {self.brokers[v].cash:.2f} $ "
                    f"< {notional:.2f} $ (el capital no es fungible entre brokers)")
                continue
            viable.append((cost["pct"], v))

        if not viable:
            return RoutingDecision(symbol, None,
                                   "ninguna plaza viable: coste excesivo o sin efectivo")
        cost_pct, venue = min(viable)
        return RoutingDecision(symbol, venue, "enrutada", cost_pct)

    # ------------------------------------------------------------ ejecucion
    def submit(self, symbol: str, side: str, notional: float) -> OrderResult:
        d = self.route(symbol, notional, side)
        if d.venue is None:
            return OrderResult(symbol, side, 0.0, 0.0, "rejected", message=d.reason)
        res = self.brokers[d.venue].submit(symbol, side, notional)
        return OrderResult(res.symbol, res.side, res.qty, res.notional, res.status,
                           res.broker_order_id,
                           f"[{d.venue}] coste {d.cost_pct:.2%}"
                           + (f" | {res.message}" if res.message else ""))

    def liquidate(self, symbol: str) -> OrderResult:
        """Cierra la posicion en TODAS las plazas que la tengan."""
        done, total = [], 0.0
        for v, b in self.brokers.items():
            if symbol in b.positions():
                r = b.liquidate(symbol)
                done.append(f"{v}:{r.status}")
                total += r.notional
        if not done:
            return OrderResult(symbol, "sell", 0.0, 0.0, "rejected",
                               message="sin posicion en ninguna plaza")
        return OrderResult(symbol, "sell", 0.0, total, "filled",
                           message="liquidada en " + ", ".join(done))

    # ------------------------------------------------------------ informe
    def routing_report(self) -> str:
        if not self.routing_log:
            return "  sin incidencias de enrutado"
        return "\n".join(f"  {x}" for x in self.routing_log[-20:])
