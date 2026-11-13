# Notas de progreso

Registro en texto libre de que se hizo y que sigue. El estado estructurado esta
en `ESTADO.json`. Antes de continuar trabajo previo, leer ambos.

## 2026-11-13 — v1.5.0: decision con plazo variable, salidas y plan de ordenes

Origen: una rama paralela desarrollada fuera del repositorio (13-nov) resolvio
la misma peticion que la v1.4 con otro enfoque. Se integra sobre esta base.

Hecho:
- `decision.py`, `exits.py`, `market_calendar.py`: `swarmtrader decide` (plazo
  3-31 sesiones, entrada en la apertura siguiente, plan de salida por accion,
  top 10 con valoracion 0-1 y catalizadores, plan de ordenes JSON).
- `execution/plan.py`: plan con `plan_id`; `swarmtrader execute` delega en
  `rebalance_to` (recorta, aplaza, espera ventas); libro con plan de salida.
- `cycle.py`: aplica el plan de salida propio de cada posicion.
- `conviction.py`: `costs`, `standalone`, `build_bank`, `market_level`, `tau`.
  Con valores por defecto, identico fila a fila al modulo v1.4.
- `swarm_rank`: `oos` siempre en el resultado. `load_long_panel` y
  `_sanitize_prices`. Scripts `fetch_panel.py` y `equity_filter.py`.
- 21 tests nuevos. Suite: 297 pasan, 7 omitidos. Controles en verde.

Medido:
- Criterio fijado antes de medir para `tau`: "integrated" tambien falla H1
  (lambda 0,35) -> se mantiene "point".
- Datos reales (12-nov, 3.313 acciones, supervivientes): NO ACCIONABLE;
  senal retenida 0-10 %; 23 posiciones, mayor peso 23,8 %, +1,7 % esperado a
  31 sesiones; 448 acciones excluidas por historia < 1.260 sesiones.

Sigue:
1. Validar decide + execute contra Alpaca paper con credenciales.
2. Datos point-in-time y repetir las mediciones.
3. Agentes de eventos (insiders, resultados) para que la puntuacion distinga.

## 2026-11-12 — v1.4.0: modo de decision por conviccion

Hecho:
- `portfolio/conviction.py`: cartera de crecimiento optimo (CRRA, Kelly por
  defecto) sobre todo el universo, con cota certificada, generacion de columnas,
  escenarios predictivos (bootstrap estacionario, mapa por percentil con
  encogimiento empirico-bayesiano) y posiciones abiertas fijas.
- `--decision conviction` en `swarm` y `cycle`: compra aunque el veredicto sea
  NO ACCIONABLE, sin topes por nombre ni sector; el veredicto se informa igual.
- `rebalance_to` vende, espera y compra con el poder de compra real (ambos modos).
- `AlpacaBroker`: poder de compra conservador, nocional a centimos con minimo de
  1 USD, espera de ventas segun horario, exclusion de no fraccionables.
- 61 tests nuevos; suite de la rama publicada: 277 pasan, 6 omitidos.
- Medicion pre-registrada en sintetico (`experiments/conviction_measure.py`).

Conclusiones medidas (sintetico, sin costes):
- Sin senal no bate al universo (-0,43 %, p = 0,83): no hay fuga. H1 incumple
  por el criterio del encogimiento (0,2015 frente a 0,2).
- Con senal concentra mas y bate al universo (+23,2 puntos), pero queda por
  debajo de equiponderar las 5 primeras (+27,3 % frente a +32,4 %).
- El encogimiento reduce la concentracion a la mitad sin coste significativo.

Sigue:
1. Validar el ciclo contra la cuenta paper de Alpaca (claves en el entorno).
2. Datos point-in-time y repetir la medicion sobre mercado real.
3. Decisiones del usuario anotadas en `ESTADO.json`.

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
