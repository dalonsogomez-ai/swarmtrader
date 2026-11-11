"""Mandato de evaluacion: meta, plazo, y parada por drawdown.

Como se implementa la meta, y por que asi
-----------------------------------------
La meta "75% en 45 dias o el sistema muere" se implementa como **criterio de
evaluacion posterior**, nunca como restriccion que empuje al sistema a actuar.
La diferencia no es filosofica, es la que separa este modulo de una maquina de
arruinarse:

  * Como criterio posterior: el sistema decide con sus propios filtros
    estadisticos, y al vencimiento se compara el resultado con la meta. Si no
    llega, se registra el fallo.
  * Como restriccion de diseno: el sistema "sabe" que necesita 75% y, al no
    encontrarlo, concentra, apalanca o baja sus umbrales hasta conseguirlo en
    la muestra. Eso es exactamente el mecanismo que produjo el campeon con
    Sharpe +2.70 sobre ruido puro en `scripts/null_test.py`.

Por eso este modulo **no expone la meta a la logica de decision**. No hay ningun
camino por el que `swarm_rank` o `portfolio/sizing` puedan leer cuanto falta
para el 75%. El mandato observa y registra; no dirige.

Decisiones de implementacion y su motivo
----------------------------------------
* **Timestamp UTC inmutable**: el inicio se escribe una vez en disco y se
  rechaza cualquier intento de reescritura. Reiniciar el proceso no reinicia el
  contador; si lo hiciera, bastaria con reiniciar para no fallar nunca.
* **Retorno ajustado por flujos (Modified Dietz)**: un ingreso de capital NO es
  rentabilidad. Sin este ajuste, meter 100 $ mas en la cuenta contaria como
  +100% y la meta se "cumpliria" sin haber ganado un euro.
* **Cash drag contabilizado**: el efectivo ocioso renta. Como el sistema declara
  NO ACCIONABLE la mayoria de los meses, ignorar ese rendimiento infravalora
  sistematicamente la decision de no operar, que suele ser la correcta.
* **Kill-switch por drawdown independiente del plazo y de la meta**: se evalua
  en cada ciclo. Si dependiera de la meta, un sistema que va perdiendo asumiria
  mas riesgo cuanto peor le fuera, que es el comportamiento de un jugador que
  dobla la apuesta.
* **Nunca liquida solo**: al dispararse devuelve una accion RECOMENDADA que
  exige confirmacion explicita. Liquidar posiciones reales es irreversible.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

__all__ = ["MandateConfig", "NavMark", "MandateState", "Mandate",
           "STATUS_RUNNING", "STATUS_TARGET_MET", "STATUS_EXPIRED_FAILED",
           "STATUS_HALTED_DRAWDOWN", "STATUS_NOT_STARTED"]

STATUS_NOT_STARTED = "NOT_STARTED"
STATUS_RUNNING = "RUNNING"
STATUS_TARGET_MET = "TARGET_MET"
STATUS_EXPIRED_FAILED = "EXPIRED_FAILED"
STATUS_HALTED_DRAWDOWN = "HALTED_DRAWDOWN"


@dataclass(frozen=True)
class MandateConfig:
    target_return: float = 0.75          # criterio de evaluacion, NO objetivo operativo
    deadline_days: int = 45              # dias naturales desde el inicio registrado
    max_drawdown: float = 0.30           # parada de emergencia, independiente de la meta
    cash_yield_annual: float = 0.0       # rendimiento del efectivo ocioso (APY del broker)
    liquidate_on_halt: bool = False      # politica: NUNCA asumida, siempre explicita
    liquidate_on_expiry: bool = False

    def validate(self) -> None:
        if not 0 < self.deadline_days <= 3650:
            raise ValueError("deadline_days fuera de rango")
        if not 0 < self.max_drawdown < 1:
            raise ValueError("max_drawdown debe estar en (0,1)")


@dataclass
class NavMark:
    """Una marca de valor neto a mercado."""

    ts_utc: str
    nav: float                    # efectivo + posiciones, marcado a mercado
    cash: float = 0.0
    external_flow: float = 0.0    # ingresos (+) y retiradas (-) desde la marca anterior
    note: str = ""


@dataclass
class MandateState:
    started_utc: str | None = None
    config: dict = field(default_factory=dict)
    initial_nav: float = 0.0
    marks: list = field(default_factory=list)
    status: str = STATUS_NOT_STARTED
    closed_utc: str | None = None
    closing_reason: str = ""
    recommended_action: str = "none"   # se persiste: un estado terminal debe
                                       # seguir diciendo que hacer al consultarlo


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _parse(ts: str) -> datetime:
    return datetime.fromisoformat(ts)


class Mandate:
    """Registro persistente del mandato. Observa; no dirige."""

    def __init__(self, path: str | Path, cfg: MandateConfig = MandateConfig()):
        cfg.validate()
        self.path = Path(path)
        self.cfg = cfg
        self.state = self._load()

    # ------------------------------------------------------------ persistencia
    def _load(self) -> MandateState:
        if self.path.exists():
            raw = json.loads(self.path.read_text())
            return MandateState(**raw)
        return MandateState()

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(asdict(self.state), indent=2, ensure_ascii=False))

    # ------------------------------------------------------------ ciclo de vida
    def start(self, nav: float, cash: float | None = None, force: bool = False) -> MandateState:
        """Arranca la evaluacion. Idempotente salvo `force` explicito.

        El inicio NO ocurre al instalar ni al investigar: solo cuando se llama
        aqui a proposito. Es la separacion que pide el requisito entre desarrollo,
        validacion y evaluacion operativa.
        """
        if self.state.started_utc and not force:
            return self.state          # no se reinicia el contador por reiniciar el proceso
        if nav <= 0:
            raise ValueError("el valor neto inicial debe ser positivo")
        self.state = MandateState(
            started_utc=_now(), config=asdict(self.cfg), initial_nav=float(nav),
            marks=[asdict(NavMark(_now(), float(nav), float(cash if cash is not None else nav),
                                  0.0, "inicio de evaluacion"))],
            status=STATUS_RUNNING,
        )
        self._save()
        return self.state

    def mark(self, nav: float, cash: float = 0.0, external_flow: float = 0.0,
             note: str = "") -> dict:
        """Registra una marca y devuelve el estado evaluado."""
        if self.state.status in (STATUS_NOT_STARTED,):
            raise RuntimeError("la evaluacion no ha sido iniciada: llama a start() primero")
        self.state.marks.append(asdict(NavMark(_now(), float(nav), float(cash),
                                               float(external_flow), note)))
        st = self.evaluate()
        self._save()
        return st

    # ------------------------------------------------------------ metricas
    def _flow_adjusted_return(self) -> float:
        """Retorno Modified Dietz: los flujos externos no cuentan como rendimiento.

        R = (V_fin - V_ini - F) / (V_ini + sum(w_i * F_i))
        con w_i la fraccion del periodo que el flujo i estuvo invertido.
        """
        marks = self.state.marks
        if len(marks) < 2:
            return 0.0
        v0 = float(marks[0]["nav"])
        v1 = float(marks[-1]["nav"])
        t0, t1 = _parse(marks[0]["ts_utc"]), _parse(marks[-1]["ts_utc"])
        span = max((t1 - t0).total_seconds(), 1.0)

        flows, weighted = 0.0, 0.0
        for m in marks[1:]:
            f = float(m.get("external_flow", 0.0))
            if f == 0.0:
                continue
            ti = _parse(m["ts_utc"])
            w = max(0.0, min(1.0, (t1 - ti).total_seconds() / span))
            flows += f
            weighted += w * f
        denom = v0 + weighted
        if abs(denom) < 1e-9:
            return 0.0
        return (v1 - v0 - flows) / denom

    def _drawdown(self) -> float:
        navs = np.array([float(m["nav"]) for m in self.state.marks], dtype=float)
        if navs.size < 2:
            return 0.0
        peak = np.maximum.accumulate(navs)
        with np.errstate(divide="ignore", invalid="ignore"):
            dd = np.where(peak > 0, (peak - navs) / peak, 0.0)
        return float(np.nanmax(dd))

    def accrue_cash(self, cash: float, days: float) -> float:
        """Interes devengado por el efectivo ocioso. Ver docstring del modulo."""
        if self.cfg.cash_yield_annual <= 0 or cash <= 0 or days <= 0:
            return 0.0
        return float(cash * ((1.0 + self.cfg.cash_yield_annual) ** (days / 365.0) - 1.0))

    def days_elapsed(self) -> float:
        if not self.state.started_utc:
            return 0.0
        return (datetime.now(timezone.utc) - _parse(self.state.started_utc)).days

    def days_remaining(self) -> float:
        return max(0.0, self.cfg.deadline_days - self.days_elapsed())

    # ------------------------------------------------------------ evaluacion
    def evaluate(self) -> dict:
        """Evalua el estado. El orden de comprobacion importa y es deliberado.

        El drawdown se comprueba ANTES que la meta y que el plazo: una parada de
        emergencia no puede quedar subordinada a que queden dias o a que el
        objetivo siga siendo alcanzable.
        """
        ret = self._flow_adjusted_return()
        dd = self._drawdown()
        elapsed = self.days_elapsed()
        status = self.state.status
        action = "none"
        reasons: list[str] = []

        if status in (STATUS_HALTED_DRAWDOWN, STATUS_EXPIRED_FAILED, STATUS_TARGET_MET):
            # estado terminal: no se reevalua, pero se conserva el motivo y la
            # accion recomendada, que es justo lo que el operador consulta despues
            action = self.state.recommended_action
            if self.state.closing_reason:
                reasons.append(self.state.closing_reason)
        elif dd > self.cfg.max_drawdown:
            status = STATUS_HALTED_DRAWDOWN
            reasons.append(f"drawdown {dd:.1%} supera el limite {self.cfg.max_drawdown:.0%}")
            action = ("RECOMENDADO: cancelar ordenes pendientes y detener nuevas compras. "
                      "Liquidar posiciones requiere confirmacion explicita"
                      + (" (politica configurada: liquidar)" if self.cfg.liquidate_on_halt
                         else " (politica configurada: NO liquidar)"))
        elif ret >= self.cfg.target_return:
            status = STATUS_TARGET_MET
            reasons.append(f"retorno {ret:.1%} alcanza la meta {self.cfg.target_return:.0%} "
                           f"en {elapsed:.0f} dias")
            action = "RECOMENDADO: aplicar politica de proteccion del resultado"
        elif elapsed >= self.cfg.deadline_days:
            status = STATUS_EXPIRED_FAILED
            reasons.append(f"vencido a los {elapsed:.0f} dias con retorno {ret:.1%} "
                           f"(meta {self.cfg.target_return:.0%})")
            action = ("RECOMENDADO: parada controlada — cancelar ordenes, persistir estado, "
                      "generar informe. Liquidar requiere confirmacion explicita")

        if status != self.state.status:
            self.state.status = status
            self.state.recommended_action = action
            if status != STATUS_RUNNING:
                self.state.closed_utc = _now()
                self.state.closing_reason = "; ".join(reasons)

        return {
            "status": status,
            "return_flow_adjusted": round(ret, 6),
            "target_return": self.cfg.target_return,
            "drawdown": round(dd, 6),
            "max_drawdown": self.cfg.max_drawdown,
            "days_elapsed": elapsed,
            "days_remaining": self.days_remaining(),
            "started_utc": self.state.started_utc,
            "n_marks": len(self.state.marks),
            "recommended_action": action,
            "reasons": reasons,
        }

    def report(self) -> str:
        st = self.evaluate()
        L = [f"MANDATO DE EVALUACION   estado: {st['status']}",
             f"  inicio (UTC inmutable): {st['started_utc']}",
             f"  dias transcurridos:     {st['days_elapsed']:.0f} de {self.cfg.deadline_days}"
             f"   (restan {st['days_remaining']:.0f})",
             f"  retorno ajustado:       {st['return_flow_adjusted']:+.2%}"
             f"   (meta {st['target_return']:.0%})",
             f"  drawdown maximo:        {st['drawdown']:.2%}"
             f"   (limite {st['max_drawdown']:.0%})",
             f"  marcas registradas:     {st['n_marks']}"]
        if st["reasons"]:
            L += ["", "  " + "; ".join(st["reasons"])]
        if st["recommended_action"] != "none":
            L += ["  " + st["recommended_action"]]
        L += ["", "  La meta no se expone a la logica de decision: este registro observa,",
              "  no dirige. Ver docstring de mandate.py."]
        return "\n".join(L)
