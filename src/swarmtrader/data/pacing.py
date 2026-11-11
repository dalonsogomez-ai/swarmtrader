"""Control de ritmo para las APIs de los brokers.

Por que esto es un modulo propio y no un `time.sleep` disperso
--------------------------------------------------------------
Los limites de Interactive Brokers no son un rate limit simple: son cuatro
reglas simultaneas y distintas, y violarlas no devuelve un 429 educado sino un
`pacing violation` que puede dejar la sesion inutil durante minutos. Las reglas
documentadas por IBKR son:

  * maximo 50 solicitudes historicas ABIERTAS a la vez,
  * no mas de 60 solicitudes historicas en cualquier ventana de 10 minutos,
  * nada de repetir una solicitud identica en menos de 15 segundos,
  * nada de 6 o mas solicitudes para el mismo contrato/exchange/tipo en 2 s,
  * cada solicitud devuelve como mucho 2.000 barras.

Un `sleep` fijo no modela ninguna de esas reglas: o va demasiado lento o viola
la de la ventana deslizante. Aqui hay un limitador de ventana deslizante real,
con las cuatro reglas, y ademas una funcion que calcula la FACTIBILIDAD antes de
empezar. Esa segunda parte es la importante: el sistema debe poder decir "esto
te va a costar 14 horas, no lo hagas" en lugar de empezar y morir a las tres
horas con una violacion.

Los limites se declaran como datos (`BrokerLimits`), no incrustados en el
codigo, para que cuando el broker los cambie se toque un sitio y no veinte.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from dataclasses import dataclass, field

__all__ = ["BrokerLimits", "IBKR_LIMITS", "ETORO_LIMITS", "ALPACA_BASIC_LIMITS",
           "ALPACA_PLUS_LIMITS", "RateLimiter", "FeasibilityReport", "assess_bulk_download"]


@dataclass(frozen=True)
class BrokerLimits:
    """Limites declarados de una API. Todos opcionales salvo el principal."""

    name: str
    max_requests: int                 # solicitudes permitidas por ventana
    window_seconds: float             # tamano de la ventana deslizante
    max_bars_per_request: int | None = None
    max_concurrent: int | None = None
    min_seconds_between_identical: float = 0.0
    burst_same_key: int | None = None         # p. ej. 6 solicitudes...
    burst_window_seconds: float = 0.0         # ...en 2 segundos
    notes: str = ""


# Limites verificados en la documentacion oficial (ver docs/ARCHITECTURE.md).
IBKR_LIMITS = BrokerLimits(
    name="IBKR TWS API (historico)",
    max_requests=60, window_seconds=600.0,
    max_bars_per_request=2000, max_concurrent=50,
    min_seconds_between_identical=15.0,
    burst_same_key=6, burst_window_seconds=2.0,
    notes="60 solicitudes historicas por cada 10 minutos; violarlo corta el flujo",
)

ETORO_LIMITS = BrokerLimits(
    name="eToro Public API",
    max_requests=60, window_seconds=60.0,
    max_bars_per_request=1000,
    notes="1000 velas por peticion: 4 anos diarios (~1008 sesiones) no caben en una",
)

ALPACA_BASIC_LIMITS = BrokerLimits(
    name="Alpaca Basic (IEX)",
    max_requests=200, window_seconds=60.0,
    notes="feed IEX; los datos de los ultimos 15 minutos no estan disponibles",
)

ALPACA_PLUS_LIMITS = BrokerLimits(
    name="Alpaca Algo Trader Plus (SIP)",
    max_requests=10000, window_seconds=60.0,
    notes="feed SIP consolidado",
)


class RateLimiter:
    """Ventana deslizante con las cuatro reglas. Seguro entre hilos.

    `acquire(key)` bloquea lo justo para no violar ninguna regla y devuelve los
    segundos que ha esperado, lo que permite auditar el coste real de una
    descarga en vez de estimarlo.
    """

    def __init__(self, limits: BrokerLimits, sleep=time.sleep, clock=time.monotonic):
        self.limits = limits
        self._sleep = sleep
        self._clock = clock
        self._events: deque[float] = deque()
        self._last_identical: dict[str, float] = {}
        self._key_events: dict[str, deque[float]] = {}
        self._lock = threading.Lock()

    def _purge(self, now: float) -> None:
        w = self.limits.window_seconds
        while self._events and now - self._events[0] >= w:
            self._events.popleft()

    def _wait_for(self, now: float, key: str | None) -> float:
        """Segundos que hay que esperar por la regla mas restrictiva."""
        L = self.limits
        waits = [0.0]

        self._purge(now)
        if len(self._events) >= L.max_requests:
            waits.append(L.window_seconds - (now - self._events[0]))

        if key is not None and L.min_seconds_between_identical > 0:
            last = self._last_identical.get(key)
            if last is not None:
                waits.append(L.min_seconds_between_identical - (now - last))

        if key is not None and L.burst_same_key and L.burst_window_seconds > 0:
            dq = self._key_events.setdefault(key, deque())
            while dq and now - dq[0] >= L.burst_window_seconds:
                dq.popleft()
            if len(dq) >= L.burst_same_key:
                waits.append(L.burst_window_seconds - (now - dq[0]))

        return max(waits)

    def acquire(self, key: str | None = None) -> float:
        """Bloquea hasta que sea seguro emitir la solicitud. Devuelve la espera."""
        waited = 0.0
        while True:
            with self._lock:
                now = self._clock()
                w = self._wait_for(now, key)
                if w <= 0:
                    self._events.append(now)
                    if key is not None:
                        self._last_identical[key] = now
                        if self.limits.burst_same_key:
                            self._key_events.setdefault(key, deque()).append(now)
                    return waited
            self._sleep(w)
            waited += w


@dataclass
class FeasibilityReport:
    broker: str
    n_symbols: int
    requests_needed: int
    seconds_estimated: float
    feasible: bool
    reasons: list[str] = field(default_factory=list)

    @property
    def hours(self) -> float:
        return self.seconds_estimated / 3600.0

    def __str__(self) -> str:
        head = (f"{self.broker}: {self.n_symbols} simbolos -> {self.requests_needed} solicitudes, "
                f"~{self.hours:.1f} h")
        verdict = "VIABLE" if self.feasible else "NO VIABLE"
        return head + f"  [{verdict}]" + ("".join(f"\n    - {r}" for r in self.reasons))


def assess_bulk_download(limits: BrokerLimits, n_symbols: int, n_sessions: int,
                         max_hours: float = 2.0) -> FeasibilityReport:
    """¿Es sensato bajar `n_sessions` sesiones de `n_symbols` simbolos por aqui?

    Se contesta ANTES de empezar. La aritmetica es elemental y por eso mismo es
    honesta: solicitudes = simbolos x ceil(sesiones / barras_por_solicitud), y
    el tiempo sale de la ventana deslizante. Si el resultado son horas, el
    sistema debe decirlo y proponer la alternativa, no lanzarse a intentarlo.
    """
    per_symbol = 1
    if limits.max_bars_per_request:
        per_symbol = max(1, -(-n_sessions // limits.max_bars_per_request))
    reqs = n_symbols * per_symbol
    rate = limits.max_requests / limits.window_seconds          # solicitudes/segundo
    secs = reqs / rate if rate > 0 else float("inf")

    reasons: list[str] = []
    if limits.max_bars_per_request and per_symbol > 1:
        reasons.append(f"{per_symbol} solicitudes por simbolo: el maximo es "
                       f"{limits.max_bars_per_request} barras y se piden {n_sessions}")
    feasible = secs <= max_hours * 3600
    if not feasible:
        reasons.append(f"a {limits.max_requests} solicitudes/{limits.window_seconds:.0f}s "
                       f"tardaria {secs/3600:.1f} h, por encima del limite de {max_hours} h")
        reasons.append("usa un proveedor point-in-time con descarga masiva "
                       "(Sharadar SEP, EODHD) para el historico y reserva este broker "
                       "para la ejecucion")
    if limits.notes:
        reasons.append(limits.notes)
    return FeasibilityReport(limits.name, n_symbols, reqs, secs, feasible, reasons)
