"""Ciclo diario: el puente entre el enjambre y el broker.

Hasta ahora `swarm_rank` imprimia una lista y `execution/` sabia mandar ordenes,
pero nadie los unia. Este modulo es esa union, y su diseno esta gobernado por
cuatro decisiones que conviene entender antes de leer el codigo.

**1. Reconciliar, no reproducir.** El ciclo no guarda "ordenes pendientes" para
reenviarlas. En cada ejecucion lee el estado REAL del broker y calcula la
diferencia contra el objetivo. Si el proceso murio a mitad del envio, la
siguiente ejecucion ve las posiciones que si entraron y solo manda lo que falta.
Un sistema que reprodujera un log de ordenes duplicaria compras tras un
reinicio; este no puede, por construccion.

**2. El broker es la verdad.** Efectivo, posiciones y patrimonio se leen de el,
nunca del estado local. El estado local solo guarda lo que el broker NO sabe: la
fecha de entrada y el maximo alcanzado de cada posicion, que es justo lo que la
triple barrera necesita y ninguna API de broker expone.

**3. Orden deliberado: parada, salidas, entradas.** El kill-switch se evalua
ANTES que nada; si salta, no se abren posiciones aunque el enjambre tenga
veredicto accionable. Las salidas se procesan antes que las entradas para
liberar efectivo, y porque cerrar una posicion que ha tocado su stop es mas
urgente que abrir una nueva.

**4. Simulacion por defecto.** `dry_run=True` calcula todo e imprime el plan sin
mandar una sola orden. Operar de verdad exige desactivarlo a proposito, y
ademas, para dinero real, la variable de entorno que ya exige
`execution/alpaca.py`.

Programacion: una ejecucion al dia tras el cierre de EE. UU. (22:15 hora
peninsular), con cron o launchd. No es un demonio: un proceso vivo 45 dias
acumula estado en memoria y si cae a mitad no sabes que se envio. Con el
mercado cerrado las ventas quedan en cola y no liberan poder de compra hasta
ejecutarse, asi que una rotacion completa tarda dos ciclos: el segundo ve el
efectivo real y compra lo que falta.

**5. Modo `conviction`.** Con `decision="conviction"` se compra aunque el
veredicto sea NO ACCIONABLE: la cartera de crecimiento optimo de
`portfolio/conviction.py`, sin `max_positions` ni tope por nombre o sector, que
puede ser una sola accion. Se mantienen la parada por drawdown del mandato, el
veto de recompra tras stop, la triple barrera, la exposicion maxima de 1,0 (sin
margen) y la simulacion por defecto. Las posiciones abiertas no se tocan: el
optimizador reparte solo el presupuesto libre teniendo en cuenta lo que ya hay.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .analogy import AnalogyConfig, find_analogues
from .data.labels import TripleBarrierConfig
from .data.loaders import MarketData
from .mandate import STATUS_RUNNING, Mandate
from .swarm_rank import SwarmRankConfig, swarm_rank

__all__ = ["CycleConfig", "PositionLedger", "CycleReport", "DailyCycle"]


@dataclass
class CycleConfig:
    dry_run: bool = True
    max_positions: int = 5
    barriers: TripleBarrierConfig = field(default_factory=TripleBarrierConfig)
    analogy: AnalogyConfig = field(default_factory=AnalogyConfig)
    use_analogy_on_exit: bool = True
    max_data_staleness_days: int = 4      # viernes -> lunes son 3 dias naturales
    state_dir: str = "runs"
    force: bool = False                   # repetir un ciclo ya ejecutado hoy
    as_of: str | None = None              # repeticion historica: fecha simulada "hoy"
    max_position_pct: float = 0.25        # tope por nombre DESPUES de renormalizar
    allow_roll: bool = True               # renovar en vez de vender y recomprar
    stop_cooldown_cycles: int = 1         # ciclos de veto tras salir por stop
    decision: str = "gated"               # gated | conviction (ver docstring, punto 5)


@dataclass
class LedgerEntry:
    symbol: str
    entry_date: str
    entry_price: float
    peak_price: float
    target_weight: float = 0.0
    note: str = ""


class PositionLedger:
    """Lo que el broker no guarda: cuando entraste y cual fue el maximo.

    Sin esto la triple barrera es inaplicable en vivo: `avg_price` da el precio
    de entrada pero no la fecha, y el maximo alcanzado no lo guarda nadie. El
    stop movil depende de ambos.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.entries: dict[str, LedgerEntry] = {}
        if self.path.exists():
            raw = json.loads(self.path.read_text())
            self.entries = {k: LedgerEntry(**v) for k, v in raw.items()}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({k: asdict(v) for k, v in self.entries.items()},
                                        indent=2, ensure_ascii=False))

    def sync(self, positions: dict, prices: pd.Series, as_of: str) -> None:
        """Alinea el libro con la verdad del broker.

        Posiciones nuevas que el libro no conoce (compradas a mano, o tras
        perder el fichero) se registran con la fecha de HOY. Eso subestima el
        tiempo de tenencia, que es el error conservador: hace que la barrera
        vertical tarde mas en cerrar, nunca menos.
        """
        for sym, pos in positions.items():
            px = float(prices.get(sym, pos.avg_price))
            if sym not in self.entries:
                self.entries[sym] = LedgerEntry(sym, as_of, float(pos.avg_price),
                                                max(px, float(pos.avg_price)),
                                                note="alta por sincronizacion")
            else:
                e = self.entries[sym]
                e.peak_price = max(e.peak_price, px)
        for sym in list(self.entries):
            if sym not in positions:
                del self.entries[sym]


@dataclass
class CycleReport:
    ts_utc: str
    session: str
    status: str
    mandate: dict = field(default_factory=dict)
    exits: list = field(default_factory=list)
    entries: list = field(default_factory=list)
    targets: dict = field(default_factory=dict)
    orders: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    dry_run: bool = True

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, ensure_ascii=False, default=str)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class DailyCycle:
    def __init__(self, md: MarketData, broker, mandate: Mandate,
                 swarm_cfg: SwarmRankConfig = SwarmRankConfig(),
                 cfg: CycleConfig = CycleConfig(), listings=None, extra=None, groups=None):
        self.md = md
        self.broker = broker
        self.mandate = mandate
        self.swarm_cfg = swarm_cfg
        self.cfg = cfg
        self.listings = listings
        self.extra = extra
        self.groups = groups
        self.state_dir = Path(cfg.state_dir)
        self.ledger = PositionLedger(self.state_dir / "positions.json")
        self.journal = self.state_dir / "cycles.jsonl"

    # ------------------------------------------------------------- guardas
    def _last_session(self) -> pd.Timestamp:
        return self.md.prices.index.max()

    def _today(self) -> pd.Timestamp:
        """Fecha de referencia. `as_of` permite REPETIR un ciclo historico.

        Sin esto el ciclo solo se puede ejecutar contra datos de hoy, lo que
        impide validarlo sobre el pasado: la guarda de frescura aborta siempre.
        El modo repeticion mide la antiguedad contra la fecha simulada, no
        contra el reloj.
        """
        if self.cfg.as_of:
            return pd.Timestamp(self.cfg.as_of).normalize()
        return pd.Timestamp.now("UTC").tz_localize(None).normalize()

    def _check_freshness(self) -> str | None:
        last = self._last_session()
        age = (self._today() - last.normalize()).days
        if age > self.cfg.max_data_staleness_days:
            return (f"datos obsoletos: la ultima sesion es {last.date()}, hace {age} dias. "
                    "Operar con precios viejos es peor que no operar")
        return None

    def _already_ran(self, session: str) -> bool:
        if not self.journal.exists():
            return False
        for line in self.journal.read_text().splitlines()[::-1]:
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("session") == session and rec.get("status") == "completed" \
                    and not rec.get("dry_run", True):
                return True
        return False

    # ------------------------------------------------------------- salidas
    def _evaluate_exit(self, sym: str, entry: LedgerEntry, last_px: float,
                       session: pd.Timestamp) -> dict:
        """Aplica la triple barrera a una posicion abierta. Ver data/labels.py."""
        b = self.cfg.barriers
        ret = last_px / entry.entry_price - 1.0 if entry.entry_price > 0 else 0.0
        held = int(np.busday_count(pd.Timestamp(entry.entry_date).date(), session.date()))
        dd_peak = last_px / entry.peak_price - 1.0 if entry.peak_price > 0 else 0.0

        decision, reason = "hold", ""
        if ret >= b.profit_target:
            decision, reason = "sell", f"objetivo alcanzado ({ret:+.1%} >= {b.profit_target:+.0%})"
        elif ret <= b.stop_loss:
            decision, reason = "sell", f"stop fijo ({ret:+.1%} <= {b.stop_loss:+.0%})"
        elif (b.trail_stop is not None and entry.peak_price > entry.entry_price
              and dd_peak <= b.trail_stop):
            decision, reason = "sell", (f"stop movil: {dd_peak:+.1%} desde el maximo "
                                        f"{entry.peak_price:.2f}")
        elif held >= b.horizon:
            decision, reason = "sell", f"barrera vertical ({held} >= {b.horizon} sesiones)"

        out = {"symbol": sym, "held_sessions": held, "return": round(ret, 4),
               "drawdown_from_peak": round(dd_peak, 4), "decision": decision,
               "reason": reason}

        # El motor de analogos puede REVOCAR una venta por objetivo, nunca un stop.
        # Un stop existe para limitar perdidas y no se negocia con estadistica;
        # un objetivo si es negociable: si los casos comparables dicen que esto
        # suele seguir subiendo, vender ahi deja dinero sobre la mesa.
        if (self.cfg.use_analogy_on_exit and decision == "sell"
                and reason.startswith("objetivo")):
            res = find_analogues(self.md.prices, self.md.volumes, sym, session,
                                 self.cfg.analogy)
            out["analogy"] = {"actionable": res.actionable, "n": res.n_analogues,
                              "reason": res.reason}
            if res.actionable:
                out["analogy"].update({
                    "p_continue": round(res.p_continue, 3),
                    "median_max_gain": round(res.median_max_gain, 4),
                    "best_horizon": res.best_horizon})
                if res.p_continue >= 0.60 and res.median_max_gain > b.profit_target:
                    out["decision"] = "hold"
                    out["reason"] = (f"objetivo alcanzado pero {res.n_analogues} casos "
                                     f"comparables continuaron al alza en el "
                                     f"{res.p_continue:.0%} (+{res.median_max_gain:.1%} "
                                     f"mediana adicional)")
        return out

    # ------------------------------------------------------------- ciclo
    def run(self) -> CycleReport:
        session = self._last_session()
        rep = CycleReport(ts_utc=_now(), session=str(session.date()), status="running",
                          dry_run=self.cfg.dry_run)

        stale = self._check_freshness()
        if stale:
            rep.status, rep.notes = "aborted", [stale]
            self._write(rep)
            return rep

        if not self.cfg.force and not self.cfg.dry_run and self._already_ran(rep.session):
            rep.status = "skipped"
            rep.notes = [f"ya se ejecuto un ciclo real para la sesion {rep.session}; "
                         "usa force=True para repetirlo"]
            self._write(rep)
            return rep

        # --- 1. verdad del broker ---
        positions = self.broker.positions()
        equity, cash = float(self.broker.equity), float(self.broker.cash)
        last_prices = self.md.prices.loc[session]
        self.ledger.sync(positions, last_prices, rep.session)

        # --- 2. mandato: se evalua ANTES de cualquier decision ---
        st = self.mandate.mark(nav=equity, cash=cash, note=f"ciclo {rep.session}")
        rep.mandate = st
        can_enter = st["status"] == STATUS_RUNNING
        if not can_enter:
            rep.notes.append(f"mandato en estado {st['status']}: no se abren posiciones. "
                             f"{st['recommended_action']}")

        # --- 3. salidas ---
        keep: dict[str, float] = {}
        rolled: set[str] = set()
        banned: set[str] = set()          # no se pueden reabrir en este ciclo
        for sym, pos in positions.items():
            entry = self.ledger.entries[sym]
            px = float(last_prices.get(sym, pos.avg_price))
            d = self._evaluate_exit(sym, entry, px, session)
            rep.exits.append(d)
            if d["decision"] == "hold":
                keep[sym] = float(pos.market_value) / equity if equity > 0 else 0.0
            elif d["reason"].startswith(("stop fijo", "stop movil")):
                # Salir por stop y recomprar el mismo dia es la conducta que
                # desangra cuentas, y ademas en Espana la recompra de valores
                # homogeneos antes de 2 meses impide computar la perdida
                # (art. 33.5 LIRPF). Veto explicito.
                banned.add(sym)
            elif self.cfg.allow_roll:
                # Salida por objetivo o por vencimiento: si el enjambre la
                # sigue queriendo, RENOVAR es mejor que vender y recomprar,
                # porque el viaje de ida y vuelta paga el spread dos veces sin
                # cambiar la exposicion. La renovacion reancla las barreras al
                # precio de hoy, asi que la disciplina se mantiene.
                rolled.add(sym)

        # --- 4. entradas (solo si el mandato lo permite) ---
        conviction = self.cfg.decision == "conviction"
        if can_enter and conviction:
            self._enter_by_conviction(rep, keep, banned, session, equity)
        elif can_enter and len(keep) < self.cfg.max_positions:
            res = swarm_rank(self.md, self.swarm_cfg, extra=self.extra, groups=self.groups,
                             listings=self.listings)
            rep.entries.append({"verdict": res.verdict.to_dict(),
                                "evidence": res.diagnostics.evidence})
            if res.verdict.actionable and not res.portfolio.empty:
                free = self.cfg.max_positions - len(keep)
                budget = max(0.0, 1.0 - sum(keep.values()))
                cand = res.portfolio[res.portfolio["weight"] > 0]
                cand = cand[~cand["symbol"].isin(keep) & ~cand["symbol"].isin(banned)]
                cand = cand.head(free)
                total = float(cand["weight"].sum())
                # Renormalizar al presupuesto libre puede empujar un peso por
                # encima del tope y deshacer lo que `portfolio/sizing.py` ya
                # habia limitado. Se vuelve a topar y el exceso se reparte
                # uniformemente entre los que aun tienen holgura.
                raw = {r["symbol"]: (float(r["weight"]) / total * budget if total > 0 else 0.0)
                       for _, r in cand.iterrows()}
                cap = float(self.cfg.max_position_pct)
                for _ in range(16):
                    over = {s: w for s, w in raw.items() if w > cap + 1e-12}
                    if not over:
                        break
                    excess = sum(w - cap for w in over.values())
                    for s in over:
                        raw[s] = cap
                    free = [s for s, w in raw.items() if w < cap - 1e-12]
                    if not free:
                        break
                    for s in free:
                        raw[s] = min(cap, raw[s] + excess / len(free))
                for sym, w in raw.items():
                    keep[sym] = w
                    r = cand[cand["symbol"] == sym].iloc[0]
                    rep.entries.append({"symbol": sym, "weight": round(w, 4),
                                        "edge": round(float(r["edge"]), 4)})
            else:
                rep.notes.append("el enjambre no emite recomendacion: "
                                 + "; ".join(res.verdict.reasons))

        # marcar en el informe que salidas fueron renovaciones y cuales ventas
        chosen = set(keep)
        for d in rep.exits:
            if d["decision"] != "sell":
                continue
            s = d["symbol"]
            if s in banned:
                d["action"] = "vender (veto de recompra este ciclo)"
            elif s in rolled and s in chosen:
                d["action"] = "renovar (el enjambre la mantiene; se reanclan barreras)"
                d["decision"] = "roll"
            else:
                d["action"] = "vender"
        for s in rolled & chosen:
            e = self.ledger.entries.get(s)
            if e is not None:
                px = float(last_prices.get(s, e.entry_price))
                e.entry_date, e.entry_price, e.peak_price = rep.session, px, px
                e.note = "reanclada por renovacion"

        rep.targets = {k: round(v, 4) for k, v in keep.items()}

        # --- 5. ejecucion ---
        if self.cfg.dry_run:
            rep.notes.append("DRY RUN: no se ha enviado ninguna orden")
        else:
            from .execution.guards import RiskLimits, RiskManager
            risk = RiskManager(self._risk_limits(), equity)
            risk.start_of_day(equity)
            dv = (self.md.prices.loc[session] * self.md.volumes.loc[session]).to_dict()
            orders = self.broker.rebalance_to(keep, risk, dv)
            rep.orders = [o.__dict__ for o in orders]
            for sym, w in keep.items():
                if sym not in self.ledger.entries:
                    px = float(last_prices.get(sym, 0.0))
                    self.ledger.entries[sym] = LedgerEntry(sym, rep.session, px, px,
                                                           target_weight=w)
                else:
                    self.ledger.entries[sym].target_weight = w
            for sym in list(self.ledger.entries):
                if sym not in keep:
                    del self.ledger.entries[sym]
            rep.notes.append(f"renovadas {len(rolled & chosen)}, "
                             f"vetadas por stop {len(banned)}")
            self.ledger.save()

        rep.status = "completed"
        self._write(rep)
        return rep

    # ------------------------------------------------------------- conviccion
    def _risk_limits(self):
        """Guardas de la ejecucion. En modo conviccion caen los topes de
        DIVERSIFICACION (por nombre, por sector, numero de ordenes); se mantienen
        sin margen, precio y liquidez minimos, perdida diaria y drawdown."""
        from .execution.guards import RiskLimits
        if self.cfg.decision == "conviction":
            return RiskLimits(max_position_pct=1.0, max_sector_pct=1.0, max_orders_per_day=100)
        return RiskLimits()

    def _enter_by_conviction(self, rep: CycleReport, keep: dict[str, float],
                             banned: set[str], session: pd.Timestamp, equity: float) -> None:
        budget = max(0.0, 1.0 - sum(keep.values()))
        # Minimo de orden de Alpaca: 1 USD. Con 100 USD eso es el 1 % del capital.
        min_w = max(self.swarm_cfg.conviction.min_weight, 1.05 / equity if equity > 0 else 1.0)
        if budget < min_w:
            rep.notes.append("modo conviccion: sin presupuesto libre para nuevas entradas")
            return
        # Lo que las guardas o el broker rechazarian no debe recibir capital: una
        # orden rechazada deja ese dinero sin invertir.
        lim = self._risk_limits()
        px = self.md.prices.loc[session]
        dv = (px * self.md.volumes.loc[session]).reindex(px.index)
        exclude = set(banned)
        exclude |= set(px.index[(px < lim.min_price) | ~(dv >= lim.min_dollar_volume)])
        exclude |= set(self.broker.untradable(list(px.index)))
        scfg = replace(self.swarm_cfg, decision="conviction",
                       conviction=replace(self.swarm_cfg.conviction, min_weight=min_w))
        res = swarm_rank(self.md, scfg, extra=self.extra, groups=self.groups,
                         listings=self.listings, held=dict(keep), exclude=exclude)
        rep.entries.append({"verdict": res.verdict.to_dict(),
                            "evidence": res.diagnostics.evidence, "decision": "conviction"})
        c = res.conviction
        if c is None or c.weights.empty:
            why = "; ".join(c.notes) if c is not None else "; ".join(res.verdict.reasons)
            rep.notes.append(f"modo conviccion: no hay asignacion ({why})")
            return
        tbl = c.table.set_index("symbol") if not c.table.empty else pd.DataFrame()
        for sym, w in c.weights.items():
            keep[sym] = float(w)
            row = tbl.loc[sym] if sym in tbl.index else None
            rep.entries.append({"symbol": sym, "weight": round(float(w), 4),
                                "edge": round(float(row["mu"]), 4) if row is not None else 0.0,
                                "p_success": round(float(row["p_success"]), 4)
                                if row is not None else None})
        i = c.info
        rep.notes.append(
            "MODO CONVICCION: " + ("ventaja NO demostrada (veredicto NO ACCIONABLE); "
                                   if not res.verdict.actionable else "")
            + f"{i.get('n_positions', 0)} posiciones, efectivas "
            f"{i.get('effective_positions', 0):.2f}, equivalente cierto "
            f"{c.certainty_equivalent:+.2%} a {res.horizon} sesiones, alternativas "
            + ", ".join(f"{k} {v:+.2%}" for k, v in c.alternatives.items()))
        rep.notes += [f"modo conviccion: {n}" for n in c.notes]

    def _write(self, rep: CycleReport) -> None:
        self.journal.parent.mkdir(parents=True, exist_ok=True)
        with self.journal.open("a") as f:
            f.write(json.dumps(asdict(rep), default=str) + "\n")

    # ------------------------------------------------------------- informe
    @staticmethod
    def format_report(rep: CycleReport) -> str:
        L = [f"CICLO {rep.session}   estado: {rep.status}"
             + ("   [DRY RUN]" if rep.dry_run else "   [ORDENES REALES]")]
        if rep.mandate:
            m = rep.mandate
            L.append(f"  mandato: {m['status']}  retorno {m['return_flow_adjusted']:+.2%}"
                     f"  drawdown {m['drawdown']:.2%}"
                     f"  dia {m['days_elapsed']:.0f}/{m['days_elapsed']+m['days_remaining']:.0f}")
        if rep.exits:
            L.append("\n  POSICIONES ABIERTAS")
            for e in rep.exits:
                tag = {"sell": "VENDER", "roll": "RENOVAR"}.get(e["decision"], "mantener")
                L.append(f"    {e['symbol']:<8} {e['return']:+7.1%}  "
                         f"{e['held_sessions']:>2} sesiones  -> {tag}"
                         + (f"  ({e['reason']})" if e["reason"] else ""))
                if "analogy" in e and e["analogy"].get("actionable"):
                    a = e["analogy"]
                    L.append(f"             analogos: {a['n']} casos, continua "
                             f"{a['p_continue']:.0%}, +{a['median_max_gain']:.1%} mediana")
        if rep.entries:
            L.append("\n  ENTRADAS")
            for e in rep.entries:
                if "verdict" in e:
                    L.append(f"    enjambre: {e['verdict']['combiner']}  "
                             f"evidencia {e['evidence']}  "
                             f"accionable={e['verdict']['actionable']}")
                else:
                    L.append(f"    {e['symbol']:<8} peso {e['weight']:.1%}  "
                             f"ventaja {e['edge']:+.2%}")
        if rep.targets:
            L.append("\n  CARTERA OBJETIVO")
            for s, w in sorted(rep.targets.items(), key=lambda kv: -kv[1]):
                L.append(f"    {s:<8} {w:6.1%}")
        else:
            L.append("\n  CARTERA OBJETIVO: vacia (todo en efectivo)")
        if rep.orders:
            L.append("\n  ORDENES")
            for o in rep.orders:
                L.append(f"    {o['side']:<4} {o['symbol']:<8} {o['notional']:>8.2f} $  "
                         f"{o['status']}" + (f"  {o['message']}" if o.get("message") else ""))
        for n in rep.notes:
            L.append(f"\n  {n}")
        return "\n".join(L)
