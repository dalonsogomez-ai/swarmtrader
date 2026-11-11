"""Almacen columnar del panel: Parquet particionado + DuckDB.

Por que Parquet + DuckDB y no otra cosa
---------------------------------------
El pipeline tiene exactamente DOS patrones de acceso y conviene optimizar los
dos, no uno:

  1. **Corte transversal**: "todos los simbolos en la fecha d" — lo usa el
     calculo de factores, el ranking y cada fecha de evaluacion walk-forward.
  2. **Secuencia por simbolo**: "las ultimas L sesiones de estos N simbolos" —
     lo usa el GRU del agente profundo.

Parquet particionado por ano da poda de particiones para (1) y lectura
columnar para (2); DuckDB resuelve ambas con SQL sin cargar el panel entero en
memoria y sin servidor que mantener.

Se evaluaron las alternativas y se descartaron con motivo:

  * **CSV**: sin tipos, sin compresion, relectura completa siempre. Descartado.
  * **HDF5**: rapido, pero el ecosistema en ARM64 es fragil y el formato es
    opaco a herramientas externas. Descartado.
  * **SQLite**: orientado a filas; un corte transversal de 5.000 simbolos toca
    5.000 filas dispersas. Sirve para el registro de trials (y ahi se usa), no
    para el panel.
  * **Formato binario propio tipo Qlib**: mas rapido en su nicho, pero obliga a
    mantener un formato y su conversion. No compensa aqui.
  * **Solo pandas en memoria**: con 64 GB funciona hasta ~5.000 simbolos y 4
    anos (~5M filas). Deja de funcionar con el historico completo de Sharadar
    (21.000+ tickers desde 1998, del orden de 10^8 filas). Por eso DuckDB.

Dimensionamiento real, medido con `benchmark()`: un panel de 5.000 simbolos y
1.008 sesiones son ~5,04 M de filas, que en Parquet ZSTD ocupan decenas de MB.
El historico completo de Sharadar esta dos ordenes de magnitud por encima y
sigue siendo manejable en disco.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .loaders import MarketData

__all__ = ["PanelStore", "DUCKDB_AVAILABLE"]

try:
    import duckdb
    DUCKDB_AVAILABLE = True
except ImportError:  # pragma: no cover
    DUCKDB_AVAILABLE = False
    duckdb = None


@dataclass
class PanelStore:
    """Panel largo (date, symbol, close, volume) sobre Parquet particionado.

    El formato largo es deliberado: el panel es disperso por naturaleza (cada
    simbolo cotiza en un rango distinto y los deslistados terminan antes), y
    guardarlo ancho obligaria a materializar NaN para todas las celdas vacias.
    Con 21.000 tickers y 25 anos, la matriz ancha seria mayoritariamente NaN.
    """

    root: Path

    def __post_init__(self) -> None:
        self.root = Path(self.root)
        self.root.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------- escritura
    def write(self, prices: pd.DataFrame, volumes: pd.DataFrame,
              listings: pd.DataFrame | None = None, compression: str = "zstd") -> dict:
        """Vuelca el panel a Parquet particionado por ano."""
        px = prices.sort_index()
        long = px.stack(future_stack=True).rename("close").reset_index()
        long.columns = ["date", "symbol", "close"]
        vol = volumes.reindex_like(px).stack(future_stack=True).rename("volume").reset_index()
        vol.columns = ["date", "symbol", "volume"]
        long = long.merge(vol, on=["date", "symbol"], how="left")
        long = long.dropna(subset=["close"])
        long["year"] = pd.to_datetime(long["date"]).dt.year

        data_dir = self.root / "panel"
        for y, blk in long.groupby("year"):
            d = data_dir / f"year={int(y)}"
            d.mkdir(parents=True, exist_ok=True)
            blk.drop(columns="year").to_parquet(d / "part.parquet", index=False,
                                                compression=compression)
        if listings is not None:
            lst = listings.copy()
            lst.index.name = "symbol"          # defensa: ver data/pit.py
            lst.reset_index().to_parquet(self.root / "listings.parquet", index=False,
                                         compression=compression)
        size = sum(f.stat().st_size for f in self.root.rglob("*.parquet"))
        return {"rows": int(len(long)), "symbols": int(px.shape[1]),
                "dates": int(px.shape[0]), "bytes": int(size),
                "bytes_per_row": round(size / max(len(long), 1), 2)}

    # ------------------------------------------------------------- lectura
    def _files(self) -> list[str]:
        return [str(p) for p in sorted((self.root / "panel").rglob("*.parquet"))]

    def read_long(self, start=None, end=None, symbols: list[str] | None = None) -> pd.DataFrame:
        files = self._files()
        if not files:
            return pd.DataFrame(columns=["date", "symbol", "close", "volume"])
        if DUCKDB_AVAILABLE:
            con = duckdb.connect()
            q = f"SELECT date, symbol, close, volume FROM read_parquet({files!r})"
            w = []
            if start is not None:
                w.append(f"date >= DATE '{pd.Timestamp(start).date()}'")
            if end is not None:
                w.append(f"date <= DATE '{pd.Timestamp(end).date()}'")
            if symbols:
                lst = ",".join(f"'{s}'" for s in symbols)
                w.append(f"symbol IN ({lst})")
            if w:
                q += " WHERE " + " AND ".join(w)
            out = con.execute(q + " ORDER BY date, symbol").df()
            con.close()
        else:  # pragma: no cover
            out = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
            if start is not None:
                out = out[out["date"] >= pd.Timestamp(start)]
            if end is not None:
                out = out[out["date"] <= pd.Timestamp(end)]
            if symbols:
                out = out[out["symbol"].isin(symbols)]
        out["date"] = pd.to_datetime(out["date"])
        return out

    def cross_section(self, date) -> pd.DataFrame:
        """Patron de acceso 1: todos los simbolos de una fecha."""
        d = pd.Timestamp(date)
        return self.read_long(start=d, end=d).set_index("symbol")

    def to_market_data(self, start=None, end=None,
                       symbols: list[str] | None = None) -> MarketData:
        long = self.read_long(start, end, symbols)
        if long.empty:
            raise ValueError("el almacen no contiene datos para ese filtro")
        px = long.pivot(index="date", columns="symbol", values="close").sort_index()
        vol = long.pivot(index="date", columns="symbol", values="volume").reindex_like(px)
        return MarketData(prices=px, volumes=vol.fillna(0.0))

    def listings(self) -> pd.DataFrame | None:
        f = self.root / "listings.parquet"
        if not f.exists():
            return None
        df = pd.read_parquet(f)
        if "symbol" not in df.columns and "index" in df.columns:
            df = df.rename(columns={"index": "symbol"})
        df = df.set_index("symbol")
        for c in ("first_date", "last_date"):
            if c in df.columns:
                df[c] = pd.to_datetime(df[c])
        return df

    # ------------------------------------------------------------- utilidad
    def benchmark(self) -> dict:
        """Mide de verdad el tamano y el tiempo de las dos consultas tipicas."""
        import time

        info = {"files": len(self._files())}
        info["bytes"] = sum(f.stat().st_size for f in self.root.rglob("*.parquet"))
        t0 = time.time()
        long = self.read_long()
        info["rows"] = int(len(long))
        info["read_all_s"] = round(time.time() - t0, 3)
        if not long.empty:
            d = long["date"].iloc[len(long) // 2]
            t0 = time.time()
            self.cross_section(d)
            info["cross_section_s"] = round(time.time() - t0, 4)
        info["mb"] = round(info["bytes"] / 1e6, 2)
        info["bytes_per_row"] = round(info["bytes"] / max(info["rows"], 1), 2)
        return info
