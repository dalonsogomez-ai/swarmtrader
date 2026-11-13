"""Aislamiento de los tests respecto al directorio del repositorio.

Varias rutas por defecto de la CLI son relativas al directorio de trabajo
(`runs/mandate.json`, `runs/positions.json`, `runs/trials.db`...). Sin este
fixture, un test que llama a la CLI sin pasar todas las rutas escribe estado en
el `runs/` real: asi se arranco por accidente el mandato de evaluacion desde una
ejecucion simulada (2026-11-13).
"""

import pytest


@pytest.fixture(autouse=True)
def _isolated_cwd(tmp_path, monkeypatch):
    """Cada test corre en su propio directorio temporal."""
    monkeypatch.chdir(tmp_path)
