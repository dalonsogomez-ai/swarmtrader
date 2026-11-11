# Notas de progreso

Registro en texto libre de que se hizo y que sigue. El estado estructurado esta
en `ESTADO.json`. Antes de continuar trabajo previo, leer ambos.

## 2026-11-11 — v1.3.0: auditoria de modelos, agentes y vias geometricas

Hecho:
- `experiments/geometry_audit.py` (pre-registrado) y `experiments/audit_followup.py`
  (exploratorio), ejecutados sobre 1.000 acciones reales. Informe en
  `docs/AUDITORIA_GEOMETRIA.md`.
- Filtro G4 en `swarm_rank` (los K primeros frente al universo).
- `sharadar_market_data` ya no descarta por ultimo precio; rellena `MarketData.nominal`.
- Filtros de precio sobre el precio nominal cuando existe.
- `shrink_on_disagreement` desactivado por defecto.
- 13 tests nuevos (`tests/test_audit.py`). Suite: 210.

Conclusiones medidas:
- El IC del enjambre (0,075) sale de la mitad baja del ranking; los 5 primeros no
  baten al universo. Con G4 el veredicto sobre datos reales es NO ACCIONABLE.
- Perseguir subidas grandes las hace mas probables y el retorno medio negativo.
- Ningun modelo, agente ni agregador geometrico nuevo supera su contraste.
- G0 no detecta mas episodios que una linea base de volatilidad.

Sigue:
1. Datos point-in-time y repetir `audit_followup.py --only-f7`.
2. Decidir el objetivo del sistema (exclusion, seleccion o loteria).
3. Contrastar las cuatro hipotesis anotadas en `ESTADO.json`.

## 2026-11-11 — v1.2.0: geometria de la informacion

Hecho:
- `stats/infogeo.py`: distancias de Fisher-Rao cerradas (gaussiana, Bernoulli, SPD).
- `stats/drift.py`: monitor de deriva en dos canales (dispersion, dependencia),
  integrado en `swarm_rank` como veto G0. Flags `--drift-alpha`, `--drift-mode`.
- `analogy.py`: `metric` = euclid | fisher_rank | fisher_gauss. Por defecto euclid.
- `experiments/infogeo.py`: experimento pre-registrado; ejecutado con 20 semillas
  y 1.000 acciones reales. Resultados en `docs/results/infogeo_2026-11-11.json`.
- 27 tests nuevos (`tests/test_infogeo.py`). `Makefile`: faltaba definir `PY`.
- Primera medicion del proyecto sobre datos reales (supervivientes, Yahoo).

Conclusiones medidas:
- El monitor dispara el 1,9% con alpha nominal 1% (autocorrelacion de estados).
  Solo ve rupturas grandes. En datos reales marca marzo-abril de 2020.
- El motor de analogos bate al azar con datos reales (+1,7% CRPS, p=0,0045).
- Ninguna metrica de Fisher bate a la euclidea en 21 sesiones (p=0,283).
- La media en logits no mejora a la lineal al combinar agentes.

No hecho:
- Gradiente natural en `deep_rank`: no ejecutable aqui (sin PyTorch).
- El valor economico del veto G0 sigue sin demostrar.

Sigue:
1. Datos point-in-time (Sharadar) y repetir `experiments/infogeo.py` sobre ellos.
2. Corregir la colision de tickers accion/cripto en `data/venues.py`.
3. Validacion historica con `cycle --as-of` antes de conectar Alpaca paper.

## 2026-10-31 — analisis de plataformas y cripto: INTERRUMPIDO

Se lanzo un analisis en paralelo (plataformas, literatura, datos reales de
cripto y acciones, simulacion) que se corto antes de producir informes. No hay
conclusiones. Quedaron en disco datos en bruto no versionados (klines diarias de
Binance con pares deslistados; precios de Yahoo de 5.550 acciones). De ahi sale
el panel real usado el 2026-11-11. Hallazgos previos al corte, ya anotados en
`ESTADO.json`: colision de 178 tickers y calendario de 252 sesiones incrustado.
