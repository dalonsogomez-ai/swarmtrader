"""Adaptador de eToro (API publica, entorno DEMO por defecto).

Estado: escrito contra la documentacion publica (api-portal.etoro.com,
builders.etoro.com y el catalogo de rutas del conector oficial, consultados el
2026-11-13) y probado con un cliente HTTP FALSO. NO se ha ejecutado contra la
API real: los nombres de campo de las respuestas de cartera se leen de forma
tolerante y deben confirmarse con una cuenta demo antes de confiar en el.

Reglas que impone este adaptador
  * Demo por defecto (rutas con `/demo/`). Real exige demo=False Y la variable
    SWARMTRADER_ALLOW_LIVE=1, igual que Alpaca.
  * Solo largo y apalancamiento 1. En eToro UE una compra a 1x es la accion real;
    con apalancamiento o en corto es un CFD, que este proyecto no opera.
  * Sin reintentos automaticos de ordenes: la API no documenta clave de
    idempotencia, y reintentar un POST puede duplicar la orden. Los GET si se
    reintentan tras un 429 respetando Retry-After.
  * Limites de ritmo: datos de mercado 120 peticiones / 60 s (cupo compartido).

Autenticacion: cabeceras `x-api-key` (aplicacion) y `x-user-key` (usuario), mas
`x-request-id` unico por peticion. Claves en ETORO_API_KEY y ETORO_USER_KEY.
"""

from __future__ import annotations

import os
import time
import uuid

from .base import BrokerAdapter, OrderResult, Position

_LIVE_ENV = "SWARMTRADER_ALLOW_LIVE"
BASE_URL = "https://public-api.etoro.com"


class EToroClient:
    """Cliente HTTP minimo. `session` inyectable para tests (interfaz de requests)."""

    def __init__(self, api_key: str | None = None, user_key: str | None = None,
                 session=None, base_url: str = BASE_URL, timeout: float = 20.0):
        self.api_key = api_key or os.environ.get("ETORO_API_KEY")
        self.user_key = user_key or os.environ.get("ETORO_USER_KEY")
        if session is None:
            if not self.api_key or not self.user_key:
                raise RuntimeError("faltan ETORO_API_KEY / ETORO_USER_KEY")
            import requests
            session = requests.Session()
        self.session, self.base_url, self.timeout = session, base_url.rstrip("/"), timeout

    def _headers(self) -> dict:
        return {"x-api-key": self.api_key or "", "x-user-key": self.user_key or "",
                "x-request-id": str(uuid.uuid4()), "Content-Type": "application/json"}

    def get(self, path: str, params: dict | None = None, retries: int = 2) -> dict:
        for attempt in range(retries + 1):
            r = self.session.get(self.base_url + path, params=params, headers=self._headers(),
                                 timeout=self.timeout)
            if r.status_code == 429 and attempt < retries:
                time.sleep(float(r.headers.get("Retry-After", 5)))
                continue
            r.raise_for_status()
            return r.json()
        return {}

    def post(self, path: str, body: dict) -> dict:
        r = self.session.post(self.base_url + path, json=body, headers=self._headers(),
                              timeout=self.timeout)
        r.raise_for_status()
        return r.json() if r.content else {}


def _pick(d: dict, *names, default=None):
    """Lee un campo aceptando variantes de mayusculas (la API mezcla estilos)."""
    low = {k.lower(): v for k, v in d.items()}
    for n in names:
        if n.lower() in low:
            return low[n.lower()]
    return default


class EToroBroker(BrokerAdapter):
    def __init__(self, client: EToroClient | None = None, demo: bool = True,
                 symbol_map: dict[str, int] | None = None):
        if not demo and os.environ.get(_LIVE_ENV) != "1":
            raise RuntimeError("Modo REAL de eToro bloqueado: exporta SWARMTRADER_ALLOW_LIVE=1 "
                               "solo tras validar el adaptador en la cuenta demo.")
        self.c = client or EToroClient()
        self.demo = demo
        self.paper = demo               # lo que mira execute_plan para pedir confirmacion
        self._ids: dict[str, int] = dict(symbol_map or {})
        self._syms: dict[int, str] = {v: k for k, v in self._ids.items()}

    # ---------------------------------------------------------------- rutas
    def _info(self, what: str) -> str:
        return f"/api/v1/trading/info/{'demo/' if self.demo else ''}{what}"

    def _exec(self, what: str) -> str:
        return f"/api/v1/trading/execution/{'demo/' if self.demo else ''}{what}"

    # ---------------------------------------------------------------- simbolos
    def instrument_id(self, symbol: str) -> int | None:
        if symbol in self._ids:
            return self._ids[symbol]
        r = self.c.get("/api/v1/market-data/search", {
            "internalSymbolFull": symbol, "fields": "instrumentId,internalSymbolFull"})
        for it in _pick(r, "items", default=[]) or []:
            if str(_pick(it, "internalSymbolFull", default="")).upper() == symbol.upper():
                iid = int(_pick(it, "instrumentId"))
                self._ids[symbol], self._syms[iid] = iid, symbol
                return iid
        return None

    # ---------------------------------------------------------------- cuenta
    def _portfolio(self) -> dict:
        r = self.c.get(self._info("portfolio"))
        return _pick(r, "clientPortfolio", default=r) or {}

    @property
    def cash(self) -> float:
        return float(_pick(self._portfolio(), "credit", default=0.0) or 0.0)

    @property
    def equity(self) -> float:
        pf = self._portfolio()
        cash = float(_pick(pf, "credit", default=0.0) or 0.0)
        return cash + sum(p.market_value for p in self._positions_from(pf).values())

    def _positions_from(self, pf: dict) -> dict[str, Position]:
        agg: dict[str, list] = {}
        for p in _pick(pf, "positions", default=[]) or []:
            if not _pick(p, "isBuy", default=True):
                continue                          # cortos (CFD) no son del sistema
            iid = int(_pick(p, "instrumentID", "instrumentId"))
            sym = self._syms.get(iid, str(iid))
            units = float(_pick(p, "units", default=0.0) or 0.0)
            amount = float(_pick(p, "amount", default=0.0) or 0.0)
            pnl = float(_pick(p, "unrealizedPnL", "netProfit", default=0.0) or 0.0)
            a = agg.setdefault(sym, [0.0, 0.0, 0.0])
            a[0] += units
            a[1] += amount
            a[2] += amount + pnl
        return {s: Position(s, u, (amt / u if u else 0.0), mv) for s, (u, amt, mv) in agg.items()}

    def positions(self) -> dict[str, Position]:
        return self._positions_from(self._portfolio())

    def last_price(self, symbol: str) -> float:
        iid = self.instrument_id(symbol)
        if iid is None:
            raise KeyError(f"eToro no tiene {symbol}")
        r = self.c.get("/api/v1/market-data/instruments/rates", {"instrumentIds": str(iid)})
        rates = _pick(r, "rates", default=[]) or []
        if not rates:
            raise KeyError(f"sin cotizacion de eToro para {symbol}")
        q = rates[0]
        bid, ask = float(_pick(q, "bid", default=0) or 0), float(_pick(q, "ask", default=0) or 0)
        return (bid + ask) / 2 if bid > 0 and ask > 0 else max(bid, ask)

    def untradable(self, symbols: list[str]) -> set[str]:
        out = set()
        for s in symbols:
            try:
                if self.instrument_id(s) is None:
                    out.add(s)
            except Exception:  # noqa: BLE001
                return set()        # sin la API no se veta nada; el ciclo reconcilia
        return out

    # ---------------------------------------------------------------- ordenes
    def submit(self, symbol: str, side: str, notional: float) -> OrderResult:
        iid = self.instrument_id(symbol)
        amount = round(abs(float(notional)), 2)
        if iid is None:
            return OrderResult(symbol, side, 0.0, 0.0, "rejected", message="no existe en eToro")
        if side == "buy":
            body = {"InstrumentID": iid, "IsBuy": True, "Leverage": 1, "Amount": amount}
            try:
                r = self.c.post(self._exec("market-open-orders/by-amount"), body)
            except Exception as exc:  # noqa: BLE001
                return OrderResult(symbol, side, 0.0, 0.0, "rejected", message=str(exc))
            oid = _pick(_pick(r, "orderForOpen", default=r) or {}, "orderID", "orderId")
            return OrderResult(symbol, side, 0.0, amount, "accepted",
                               broker_order_id=str(oid) if oid is not None else None)
        return self._reduce(symbol, iid, amount)

    def _reduce(self, symbol: str, iid: int, amount: float | None) -> OrderResult:
        """Vende `amount` (None = todo) cerrando posiciones de la mas antigua a la nueva."""
        pf = self._portfolio()
        px = self.last_price(symbol) if amount is not None else 0.0
        left = amount / px if amount is not None and px > 0 else None
        done = 0.0
        ids = []
        for p in _pick(pf, "positions", default=[]) or []:
            if int(_pick(p, "instrumentID", "instrumentId")) != iid or not _pick(p, "isBuy",
                                                                                   default=True):
                continue
            units = float(_pick(p, "units", default=0.0) or 0.0)
            take = units if left is None else min(units, left)
            if take <= 0:
                break
            pid = _pick(p, "positionID", "positionId")
            body = {"InstrumentId": iid}
            if take < units:
                body["UnitsToDeduct"] = round(take, 6)
            try:
                r = self.c.post(self._exec(f"market-close-orders/positions/{pid}"), body)
                ids.append(str(_pick(r, "orderID", "orderId", default=pid)))
            except Exception as exc:  # noqa: BLE001
                return OrderResult(symbol, "sell", done, 0.0, "rejected", message=str(exc))
            done += take
            if left is not None:
                left -= take
        if not ids:
            return OrderResult(symbol, "sell", 0.0, 0.0, "rejected", message="sin posicion")
        return OrderResult(symbol, "sell", done, amount or 0.0, "accepted",
                           broker_order_id=",".join(ids))

    def liquidate(self, symbol: str) -> OrderResult:
        iid = self.instrument_id(symbol)
        if iid is None:
            return OrderResult(symbol, "sell", 0.0, 0.0, "rejected", message="no existe en eToro")
        return self._reduce(symbol, iid, None)
