"""Descarga los ficheros OFICIALES de la SEC para `swarmtrader fundamentals`.

  * companyfacts.zip (todos los hechos XBRL, ~1-2 GB, se regenera cada noche)
  * company_tickers_exchange.json (mapeo CIK -> ticker -> bolsa, incluye OTC)

La SEC exige identificarse: la cabecera User-Agent debe llevar un nombre y un
correo de contacto (https://www.sec.gov/os/accessing-edgar-data). No se incrusta
ninguno aqui: se lee de la variable SEC_USER_AGENT, p. ej.
  export SEC_USER_AGENT="Nombre Apellido nombre@dominio.com"
Limite de la SEC: 10 peticiones por segundo; este script hace dos.

Uso: python3 scripts/fetch_sec.py data/
Despues:
  swarmtrader fundamentals --companyfacts data/companyfacts.zip \
      --tickers data/company_tickers_exchange.json --out-dir data
"""
import os
import pathlib
import sys
import urllib.request

FILES = {
    "companyfacts.zip": "https://www.sec.gov/Archives/edgar/daily-index/xbrl/companyfacts.zip",
    "company_tickers_exchange.json": "https://www.sec.gov/files/company_tickers_exchange.json",
}


def main() -> int:
    ua = os.environ.get("SEC_USER_AGENT", "").strip()
    if "@" not in ua:
        print("define SEC_USER_AGENT con nombre y correo de contacto (lo exige la SEC)")
        return 2
    out = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "data")
    out.mkdir(parents=True, exist_ok=True)
    for name, url in FILES.items():
        req = urllib.request.Request(url, headers={"User-Agent": ua,
                                                   "Accept-Encoding": "identity"})
        dst = out / name
        with urllib.request.urlopen(req, timeout=600) as r, open(dst, "wb") as fh:  # noqa: S310
            while chunk := r.read(1 << 20):
                fh.write(chunk)
        print(f"{name}: {dst.stat().st_size / 1e6:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
