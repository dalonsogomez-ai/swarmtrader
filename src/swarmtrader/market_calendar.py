"""Calendario de sesiones de la NYSE y conversion de horas a Europe/Madrid.

Se necesita para dar fechas y horas concretas de entrada y salida. El
calendario federal de EE. UU. no sirve: incluye dias en que la bolsa abre
(Columbus Day, Veterans Day) y omite uno en que cierra (Viernes Santo). Aqui se
implementan las reglas de festivos de la NYSE vigentes desde 2022 (con
Juneteenth). No se modelan cierres extraordinarios (luto nacional, catastrofes)
ni las sesiones de media jornada: una orden de mercado de tipo `day` enviada un
dia de cierre anticipado se ejecuta igual en la apertura.
"""

from __future__ import annotations

from datetime import date, datetime, time, timedelta
from functools import lru_cache
from zoneinfo import ZoneInfo

__all__ = ["nyse_holidays", "is_session", "next_session", "add_sessions",
           "session_open_madrid", "session_close_madrid", "OPEN_ET", "CLOSE_ET"]

ET = ZoneInfo("America/New_York")
MADRID = ZoneInfo("Europe/Madrid")
OPEN_ET = time(9, 30)
CLOSE_ET = time(16, 0)


def _observed(d: date) -> date:
    if d.weekday() == 5:            # sabado -> viernes
        return d - timedelta(days=1)
    if d.weekday() == 6:            # domingo -> lunes
        return d + timedelta(days=1)
    return d


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    d = date(year, month, 1)
    d += timedelta(days=(weekday - d.weekday()) % 7)
    return d + timedelta(weeks=n - 1)


def _last_weekday(year: int, month: int, weekday: int) -> date:
    d = date(year, month + 1, 1) - timedelta(days=1) if month < 12 else date(year, 12, 31)
    return d - timedelta(days=(d.weekday() - weekday) % 7)


def _easter(year: int) -> date:
    """Domingo de Pascua (algoritmo anonimo gregoriano)."""
    a, b, c = year % 19, year // 100, year % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    lv = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * lv) // 451
    month = (h + lv - 7 * m + 114) // 31
    day = (h + lv - 7 * m + 114) % 31 + 1
    return date(year, month, day)


@lru_cache(maxsize=64)
def nyse_holidays(year: int) -> frozenset:
    hol = {
        _nth_weekday(year, 1, 0, 3),               # Martin Luther King Jr.
        _nth_weekday(year, 2, 0, 3),               # Presidents' Day
        _easter(year) - timedelta(days=2),         # Viernes Santo
        _last_weekday(year, 5, 0),                 # Memorial Day
        _observed(date(year, 7, 4)),               # Independencia
        _nth_weekday(year, 9, 0, 1),               # Labor Day
        _nth_weekday(year, 11, 3, 4),              # Accion de Gracias
        _observed(date(year, 12, 25)),             # Navidad
    }
    if year >= 2022:
        hol.add(_observed(date(year, 6, 19)))      # Juneteenth
    ny = date(year, 1, 1)
    if ny.weekday() != 5:                          # si cae en sabado, la NYSE no lo traslada
        hol.add(_observed(ny))
    return frozenset(hol)


def is_session(d: date) -> bool:
    return d.weekday() < 5 and d not in nyse_holidays(d.year)


def next_session(d: date) -> date:
    """Primera sesion estrictamente posterior a `d`."""
    d = d + timedelta(days=1)
    while not is_session(d):
        d += timedelta(days=1)
    return d


def add_sessions(d: date, n: int) -> date:
    """La sesion que queda `n` sesiones despues de `d` (n >= 0)."""
    for _ in range(int(n)):
        d = next_session(d)
    return d


def session_open_madrid(d: date) -> datetime:
    return datetime.combine(d, OPEN_ET, tzinfo=ET).astimezone(MADRID)


def session_close_madrid(d: date) -> datetime:
    return datetime.combine(d, CLOSE_ET, tzinfo=ET).astimezone(MADRID)
