"""Registro persistente de trials. Sin esto el DSR no es calculable.

Cada genoma evaluado se escribe aqui ANTES de mirar el resultado. Es la unica
forma honesta de conocer N, el numero de pruebas realizadas, que es el input
critico del Deflated Sharpe Ratio.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path

import numpy as np

_SCHEMA = """
CREATE TABLE IF NOT EXISTS trials (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id        TEXT NOT NULL,
    generation    INTEGER NOT NULL,
    agent_id      TEXT NOT NULL,
    genome_json   TEXT NOT NULL,
    window_start  TEXT,
    window_end    TEXT,
    sharpe        REAL,
    total_return  REAL,
    max_dd        REAL,
    turnover      REAL,
    n_periods     INTEGER,
    returns_json  TEXT,
    created_at    TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_trials_run ON trials(run_id);
"""


@dataclass(frozen=True)
class TrialRecord:
    run_id: str
    generation: int
    agent_id: str
    genome: dict
    window_start: str
    window_end: str
    sharpe: float
    total_return: float
    max_dd: float
    turnover: float
    returns: np.ndarray


class TrialRegistry:
    """SQLite append-only. Nunca borres filas: falsearia el N del DSR."""

    def __init__(self, path: str | Path = "runs/trials.db"):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.path))
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    def log(self, rec: TrialRecord) -> int:
        cur = self._conn.execute(
            """INSERT INTO trials
               (run_id, generation, agent_id, genome_json, window_start, window_end,
                sharpe, total_return, max_dd, turnover, n_periods, returns_json)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                rec.run_id, rec.generation, rec.agent_id,
                json.dumps(rec.genome, sort_keys=True, default=float),
                rec.window_start, rec.window_end,
                float(rec.sharpe), float(rec.total_return),
                float(rec.max_dd), float(rec.turnover),
                int(len(rec.returns)),
                json.dumps([round(float(x), 8) for x in rec.returns]),
            ),
        )
        self._conn.commit()
        return int(cur.lastrowid)

    def n_trials(self, run_id: str | None = None) -> int:
        q = "SELECT COUNT(*) FROM trials" + (" WHERE run_id = ?" if run_id else "")
        row = self._conn.execute(q, (run_id,) if run_id else ()).fetchone()
        return int(row[0])

    def sharpe_variance(self, run_id: str | None = None) -> float:
        """Varianza de los Sharpe por periodo de todas las pruebas: input del DSR."""
        q = "SELECT sharpe, n_periods FROM trials" + (" WHERE run_id = ?" if run_id else "")
        rows = [r for r in self._conn.execute(q, (run_id,) if run_id else ())
                if r[0] is not None and r[1]]
        if len(rows) < 2:
            return 0.0
        per_period = [r[0] / np.sqrt(252.0) for r in rows]
        return float(np.var(np.asarray(per_period), ddof=1))

    def returns_matrix(self, run_id: str | None = None, max_trials: int = 500):
        """Matriz (periodos x trials) para estimar el N efectivo."""
        q = "SELECT returns_json FROM trials" + (" WHERE run_id = ?" if run_id else "")
        q += f" ORDER BY id DESC LIMIT {int(max_trials)}"
        series = [json.loads(r[0]) for r in self._conn.execute(q, (run_id,) if run_id else ()) if r[0]]
        series = [s for s in series if len(s) >= 3]
        if len(series) < 2:
            return np.zeros((0, 0))
        n = min(len(s) for s in series)
        return np.column_stack([np.asarray(s[-n:], dtype=float) for s in series])

    def close(self) -> None:
        self._conn.close()
