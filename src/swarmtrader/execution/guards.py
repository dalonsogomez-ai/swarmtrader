"""Limites de riesgo INMUTABLES. El genoma no puede tocarlos.

Motivo de diseno: si los limites de riesgo formasen parte del espacio de
busqueda, la evolucion descubriria en pocas generaciones que maximizar
apalancamiento y concentracion maximiza el fitness a corto plazo. La poblacion
convergeria a la ruina y el torneo lo llamaria exito.

Estas guardas se evaluan en el ultimo momento, justo antes de mandar la orden,
y son las mismas para todos los agentes.
"""

from __future__ import annotations

from dataclasses import dataclass, field


class RiskViolation(Exception):
    """Orden rechazada por las guardas. Nunca capturar y reintentar sin revisar."""


@dataclass(frozen=True)
class RiskLimits:
    max_position_pct: float = 0.35      # % de la cartera en un solo nombre
    max_gross_exposure: float = 1.00    # 1.00 = sin apalancamiento
    max_daily_loss_pct: float = 0.10    # corta el dia si se pierde mas
    max_total_drawdown_pct: float = 0.30  # apaga el sistema
    min_price: float = 1.00             # nada de penny stocks
    min_dollar_volume: float = 1_000_000.0
    allow_short: bool = False
    allow_margin: bool = False
    max_orders_per_day: int = 20
    # Regla de los 2 meses (art. 33.5 LIRPF): recomprar valores homogeneos antes
    # de 2 meses impide computar la perdida. En EE. UU. la wash sale son 30 dias;
    # en Espana son 2 meses para cotizados, asi que manda la mas restrictiva.
    wash_sale_days: int = 61
    max_binary_catalyst_pct: float = 0.03   # tope duro para eventos binarios
    max_sector_pct: float = 0.25


@dataclass
class RiskState:
    equity_start_of_day: float
    peak_equity: float
    orders_today: int = 0
    halted: bool = False
    halt_reason: str | None = None
    # ventas con perdida: {symbol: fecha ISO}. Necesario para la regla de los
    # 2 meses del art. 33.5 LIRPF, que es mas restrictiva que la wash sale de
    # 30 dias de EE. UU.
    loss_sales: dict = field(default_factory=dict)
    sector_exposure: dict = field(default_factory=dict)
    binary_catalysts: set = field(default_factory=set)


class RiskManager:
    def __init__(self, limits: RiskLimits, initial_equity: float):
        self.limits = limits
        self.state = RiskState(equity_start_of_day=initial_equity, peak_equity=initial_equity)

    def start_of_day(self, equity: float) -> None:
        self.state.equity_start_of_day = equity
        self.state.orders_today = 0

    def update_equity(self, equity: float) -> None:
        self.state.peak_equity = max(self.state.peak_equity, equity)
        if self.state.peak_equity > 0:
            dd = (self.state.peak_equity - equity) / self.state.peak_equity
            if dd > self.limits.max_total_drawdown_pct:
                self.halt(f"drawdown total {dd:.1%} > {self.limits.max_total_drawdown_pct:.1%}")
        if self.state.equity_start_of_day > 0:
            day = (equity - self.state.equity_start_of_day) / self.state.equity_start_of_day
            if day < -self.limits.max_daily_loss_pct:
                self.halt(f"perdida diaria {day:.1%} supera el limite")

    def halt(self, reason: str) -> None:
        self.state.halted = True
        self.state.halt_reason = reason

    def check_order(self, *, symbol: str, notional: float, price: float,
                    dollar_volume: float, equity: float, gross_after: float,
                    side: str = "buy", sector: str | None = None,
                    binary_catalyst: bool = False, today: str | None = None) -> None:
        L = self.limits
        if self.state.halted:
            raise RiskViolation(f"sistema detenido: {self.state.halt_reason}")
        if side == "sell_short" and not L.allow_short:
            raise RiskViolation("short no permitido")
        if self.state.orders_today >= L.max_orders_per_day:
            raise RiskViolation("limite de ordenes diarias alcanzado")
        if price < L.min_price:
            raise RiskViolation(f"{symbol}: precio {price} < minimo {L.min_price}")
        if dollar_volume < L.min_dollar_volume:
            raise RiskViolation(f"{symbol}: liquidez insuficiente ({dollar_volume:,.0f})")
        if equity > 0 and abs(notional) / equity > L.max_position_pct + 1e-9:
            raise RiskViolation(
                f"{symbol}: posicion {abs(notional)/equity:.1%} > maximo {L.max_position_pct:.0%}")
        if equity > 0 and gross_after / equity > L.max_gross_exposure + 1e-9:
            raise RiskViolation(f"exposicion bruta {gross_after/equity:.1%} > maximo")

        # --- limites que antes estaban declarados pero no se aplicaban ---
        if binary_catalyst and equity > 0 and side == "buy":
            # Kelly asume que conoces la probabilidad. En un resultado binario
            # (ensayo clinico, fallo judicial) no la conoces, asi que el tope es
            # duro e independiente de lo que sugiera el dimensionador.
            if abs(notional) / equity > L.max_binary_catalyst_pct + 1e-9:
                raise RiskViolation(
                    f"{symbol}: {abs(notional)/equity:.1%} sobre catalizador binario > "
                    f"maximo {L.max_binary_catalyst_pct:.0%}")

        if sector and equity > 0 and side == "buy":
            after = self.state.sector_exposure.get(sector, 0.0) + abs(notional)
            if after / equity > L.max_sector_pct + 1e-9:
                raise RiskViolation(
                    f"{symbol}: sector {sector} quedaria al {after/equity:.1%} > "
                    f"maximo {L.max_sector_pct:.0%}")

        if side == "buy" and symbol in self.state.loss_sales:
            from datetime import date as _date
            sold = _date.fromisoformat(self.state.loss_sales[symbol])
            now = _date.fromisoformat(today) if today else _date.today()
            days = (now - sold).days
            if days < L.wash_sale_days:
                raise RiskViolation(
                    f"{symbol}: vendido con perdida hace {days} dias; recomprar antes de "
                    f"{L.wash_sale_days} impide computar la perdida (art. 33.5 LIRPF)")

        if sector and side == "buy":
            self.state.sector_exposure[sector] = (
                self.state.sector_exposure.get(sector, 0.0) + abs(notional))
        self.state.orders_today += 1

    def record_loss_sale(self, symbol: str, when: str) -> None:
        """Registra una venta con perdida para el bloqueo de recompra."""
        self.state.loss_sales[symbol] = when
