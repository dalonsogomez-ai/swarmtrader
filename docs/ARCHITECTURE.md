# Arquitectura del sistema final

Este documento explica **de donde sale cada pieza** y, mas importante, **que se
dejo fuera a proposito**. La segunda lista es la que hace que el sistema
funcione.

---

## 1. La decision de diseno central

La pregunta original era "el mejor enjambre multiagente con los mejores
algoritmos". La respuesta que implementa este repositorio es distinta de la que
sugiere la pregunta, y conviene entender por que antes de leer el codigo.

El experimento de control (`scripts/null_test.py`) genera un mercado **sin
ninguna senal** y lanza el torneo evolutivo encima. El campeon sale con un
Sharpe anualizado de **+2.70**. No hay nada que descubrir en esos datos: el
+2.70 es enteramente el resultado de haber probado muchas hipotesis y quedarse
con la mejor.

De ahi se sigue todo lo demas:

> Anadir agentes **no** anade alfa. Anade hipotesis. Un enjambre de N agentes es
> un problema de comparacion multiple con N grados de libertad, y sin
> correccion el mejor de N siempre parece bueno.

Por eso el enjambre de este repositorio esta construido al reves de lo habitual:
cada agente tiene que **ganarse** su sitio contra su propia linea base, y el
conjunto tiene que superar tres filtros estadisticos antes de que se asigne un
solo euro. El resultado por defecto del sistema es **NO ACCIONABLE**. Producir
una recomendacion es la excepcion, no la salida normal.

---

## 2. De donde viene cada pieza

| Componente | Origen | Que se conservo |
|---|---|---|
| Motor de backtest, `execution_lag=1`, costes sobre turnover real | `swarmtrader v0.4.1` | Intacto. Es la unica implementacion correcta de las diez. |
| DSR / PBO / registro de trials | `swarmtrader v0.4.1` | Intacto, mas PBO aplicado ahora tambien a la matriz de agentes. |
| Calibracion isotonica + ECE | `swarmtrader v0.4.1` | Intacto. Es lo que hace que `P(exito)` signifique algo. |
| Senales externas point-in-time (PEAD, revisiones, insiders, short, noticias) | `swarmtrader v0.4` | Intacto, y ahora ademas agrupadas en agentes de evento. |
| Fiscalidad espanola (IRPF, regla de los 2 meses) | `swarmtrader v0.4.1` | Intacto. |
| Guardarrailes de riesgo fuera del genoma | `swarmtrader v0.4.1` | Intacto. |
| **Idea** de multiples analistas especializados | `Scraping_Stocks-Lightyear`, `Cometagent-StockAI-Pro`, `portfolia/Gearly-AI` | Solo la idea. La implementacion era `random.gauss`; aqui cada agente es un modelo real con su IC medido. |
| **Idea** de coste de transaccion y slippage explicitos | `Cometagent` (`backtesting_service.py`) | Validada y ya presente en el motor base; confirmo que el diseno era correcto. |
| Ingesta real de precios con `yfinance` | `workspace-main/enhanced_stocker_pro.py` | El unico de los nueve que operaba sobre datos reales sin fabricar nada. |
| **Idea** de horizonte parametrizable y filtro por capitalizacion | `StockMarketAnalyst` | Parametros `--horizon`, `--threshold`, `--min-dollar-volume`. |
| Scraping de catalogo de broker | `Scraping_Stocks`, `portfolia` | Solo el patron; la carga de universo ya existia en `data/universe.py`. |

### Lo nuevo de esta version

| Modulo | Que hace | Por que hacia falta |
|---|---|---|
| `stats/cpcv.py` | Validacion cruzada purgada y combinatoria | Con etiquetas a 21 sesiones, una k-fold normal filtra el futuro al entrenamiento. Purga + embargo lo cierra. |
| `stats/spa.py` | SPA de Hansen y Reality Check de White, con bootstrap estacionario | Es **el** algoritmo para el problema del enjambre: contrastar N candidatos a la vez sin que el maximo de N ruidos parezca destreza. |
| `agents/specialists.py` | 9 agentes con tesis economica previa | Roster fijado antes de ver los datos: el numero de hipotesis es conocido y deflactable. |
| `ensemble/stacking.py` | Combinacion con pesos ajustados solo sobre el pasado | El stacking ingenuo elige pesos sabiendo que agente funciono en el periodo que luego evalua. |
| `portfolio/sizing.py` | Kelly fraccional + objetivo de vol + topes + banda de no-negociacion | Los diez proyectos anteriores terminaban en una lista. Una lista no es una cartera. |
| `swarm_rank.py` | Orquestador y triple filtro | Donde se decide si hay algo que hacer. |

---

## 3. Los tres filtros

Los tres son obligatorios. Si cualquiera falla, no se asigna capital.

**G1 — ¿hay destreza en alguna parte?**
SPA de Hansen sobre los N agentes mas los tres combinadores, contra la hipotesis
de que ninguno bate al azar. El bootstrap es **estacionario** (Politis-Romano),
no iid: los IC por fecha estan autocorrelados y un bootstrap iid da p-valores
optimistas. Esto no es teorico — en el test
`test_block_bootstrap_matters_under_autocorrelation` la version iid rechaza
(p=0.048) donde la correcta no (p=0.454).

**G2 — ¿la complejidad se gana el sitio?**
SPA del combinador aprendido frente a la equiponderacion. Si el stack no la bate,
el sistema **reporta la equiponderacion**. Sin este filtro, anadir agentes
siempre parece mejorar el resultado.

**G3 — ¿sobrevive a la seleccion?**
PBO sobre la matriz de IC por agente: con que frecuencia el agente que gana
dentro de muestra cae por debajo de la mediana fuera de muestra. En el mercado
sin senal, PBO sale **0.99**; con senal real, **0.00**.

A los tres se suma el nivel de evidencia heredado de v0.4 (t-stat del IC y
precision@k con intervalo de confianza frente a la tasa base).

---

## 4. Lo que se rechazo, y por que

Esta seccion es tan importante como el codigo.

**Agentes LLM que generan senal.** Todos los proyectos anteriores tenian un
"agente de sentimiento" o un "debate alcista contra bajista" que producia un
score. Se rechaza: un LLM no tiene ventaja comparativa estimando un numero que
un modelo cross-seccional estima mejor y de forma auditable, y su salida no es
reproducible ni deflactable. Donde un LLM **si** tiene ventaja comparativa es en
convertir texto no estructurado en variables estructuradas point-in-time; esa
puerta ya existe en `data/signals.py` (`news_features`, con FinBERT opcional) y
lo que produzca tiene que ganarse su IC como cualquier otro factor.

**Votacion o consenso entre agentes.** Un voto por mayoria descarta la magnitud
de la conviccion y no es deflactable. Se sustituye por combinacion ponderada con
pesos ajustados sobre el pasado, y el desacuerdo se aprovecha como senal de
confianza (`agreement`) en vez de ocultarse.

**Agentes que "mueren" con dinero real.** La idea original del enjambre era que
cada agente operase con 50 $ reales y muriese al quebrar. Se rechaza: es
seleccion sobre una sola trayectoria y con muestra minuscula, es decir, la
maquina de overfitting mas cara posible — pagas con dinero por generar el sesgo
de supervivencia. La seleccion ocurre en simulacion con DSR/PBO/SPA; el dinero
real entra despues, y a traves de `execution/paper.py` primero.

**Mas factores.** No se han anadido factores nuevos a los 14 existentes.
Anadirlos sin ampliar la muestra sube la varianza de la seleccion. El limite no
es el numero de factores, es el numero de fechas OOS independientes.

**Redes neuronales de series temporales.** `Cometagent` definia LSTM y
Transformer. Con ~18 fechas OOS independientes y horizonte de 21 sesiones no hay
muestra para ajustarlas honestamente. Ridge y GBDT con regularizacion fuerte
estan mas cerca del optimo sesgo-varianza aqui, y son auditables.

---

---

## 7. La capa profunda (v0.6)

### Por que esta arquitectura y no un LSTM de precios

La literatura de prediccion cross-seccional converge en tres puntos, y los tres
estan implementados en `models/deep.py`:

1. **Ranking, no regresion.** Predecir el retorno absoluto es casi imposible y
   ademas no es lo que necesitas. El modelo se entrena con **ListNet**
   (entropia cruzada entre las distribuciones softmax de scores y targets),
   opcionalmente mezclada con un **Sharpe diferenciable** sobre una cartera
   long-short suave. Eso alinea el entrenamiento con el objetivo de inversion
   (Feng et al., *Temporal Relational Ranking*, 2019).
2. **Modelar el corte completo, no cada accion aislada.** Un LSTM por ticker
   ignora que el retorno relativo depende del resto del universo. Aqui un GRU
   codifica la secuencia de cada accion y despues una capa de **atencion
   multi-cabeza opera sobre todas las acciones de la fecha**, con un **sesgo
   aprendido por par de sectores** que se suma a los logits: es una Temporal
   Graph Convolution barata, sin necesitar un grafo externo de relaciones.
3. **Promediar semillas, y promediar RANGOS.** La varianza por numero de
   parametros es el mayor problema de una DNN en este dominio (Abe-Nakayama
   2018). Se entrenan N semillas y se promedian sus rangos, no sus scores: una
   semilla mal escalada no puede dominar el conjunto.

```
seq (N,L,2)  --GRU--+
                    +- concat - MLP - atencion cross-seccional - residual - head - score (N,)
feat (N,F) --MLP----+            ^ sesgo aprendido por par de sectores (TGC barata)
```

### La unica pregunta que importaba

Deep learning aqui no es una mejora automatica: **anade capacidad de
sobreajustar**. La pregunta no es si aprende, es si inventa. Medido en este
repositorio, con validacion purgada y parada temprana:

| Mercado | IC medio OOS | t | Veredicto |
|---|---|---|---|
| Ruido puro | **-0.034** | -0.75 | no inventa senal |
| Momentum inyectado | **+0.413** | +12.39 | la aprende |

Si la primera fila hubiera dado un t significativo, la capa profunda no estaria
en el repositorio.

### Como evita las fugas

`models/trainer.py` calcula las fechas de entrenamiento, validacion y evaluacion
**exactamente igual** que `rank.walk_forward_oos`. Ademas, la validacion para la
parada temprana se toma del final del tramo de entrenamiento y se **purga del
propio entrenamiento** con el mismo horizonte y embargo. Sin esa purga, la
parada temprana elige el checkpoint que mejor predice datos que el modelo ya vio
de forma indirecta: una fuga silenciosa y de las mas caras.

El agente profundo entra al enjambre como uno mas: pasa por G1, G2 y G3 igual
que `momentum` o `reversal`, y si no aporta, su peso es cero.

---

## 8. Fuentes de datos (v0.6)

`data/brokers.py` unifica cuatro entradas en un `MarketData`:

| Fuente | Universo | Historico | Notas |
|---|---|---|---|
| CSV | tu fichero | yfinance | Lightyear, eToro, IBKR y Robinhood exportan CSV |
| **Alpaca** | `get_all_assets` filtrado a negociables y fraccionables | barras diarias ajustadas por splits, troceadas de 200 en 200 | datos y ordenes en la misma API; `paper=True` usa la MISMA ruta de codigo |
| eToro | watchlist via Public API | yfinance | solo watchlist |
| yfinance | lista de tickers | yfinance | por defecto |

El sesgo de supervivencia aplica a las cuatro salvo a un CSV historico propio:
todas devuelven los activos que existen **hoy**.

---

## 9. v0.7 — lo que la investigacion cambio, y lo que midio al aplicarlo

La investigacion concluyo que el cuello de botella no eran los modelos ni los
agentes, sino los datos. La v0.7 implementa eso y, al implementarlo, produjo
cuatro resultados medidos que conviene leer antes que el codigo.

### 9.1 El sesgo de supervivencia tiene DOS vias, y la segunda es silenciosa

La conocida es el universo: los que quebraron no estan. La que casi nadie
corrige esta en la ETIQUETA. `forward_returns` calcula `precio[t+h]/precio[t]-1`;
si la accion muere en t+j con j <= h, el precio futuro es NaN, el target es NaN
y **esa observacion desaparece del entrenamiento y de la evaluacion**. El modelo
nunca ve como se comporta una accion antes de quebrar y `precision@k` se mide
solo sobre supervivientes.

Medido sobre un panel sintetico de 60 simbolos con 12 bajas: la etiqueta
point-in-time recupera **240 observaciones** que el `shift` ingenuo tira, y esas
observaciones promedian **-24,5%** (**-72,2%** las quiebras).

### 9.2 La direccion del sesgo depende del regimen de muerte

Este fue el resultado inesperado. Con senal realista (momentum 0,02) y 12% de
bajas anuales, comparando A = solo supervivientes, B = panel completo con
etiqueta ingenua, C = point-in-time:

| Regimen | IC: A / B / C | Retorno top-5: A / B / C | Sesgo de etiqueta (B-C) |
|---|---|---|---|
| Muerte anunciada (declive 120d) | +0,035 / +0,085 / +0,106 | +0,30% / +1,10% / +1,42% | **-0,32 pp** |
| **Muerte subita** (fraude, suspension) | +0,035 / +0,037 / +0,040 | +0,30% / **+0,89%** / **+0,59%** | **+0,30 pp** |

Lectura correcta, que contradice la narrativa habitual:

* El sesgo del **universo** (A frente a C) **no infla el IC, lo reduce**: quitar
  a los deslistados elimina dispersion cross-seccional explotable.
* El sesgo de la **etiqueta** (B frente a C) **si infla el retorno** en el
  regimen de muerte subita: **+0,30 puntos porcentuales al mes**, del orden de
  3,6 pp anualizados de rentabilidad que no existe. Y la muerte subita es
  precisamente el caso que ningun factor de precio puede anticipar.
* La narrativa "el sesgo de supervivencia infla tu backtest" es cierta para
  **retornos de cartera** y NO se traslada sin mas al **rank IC**, porque el IC
  mide calidad de ordenacion y los perdedores predecibles la mejoran.

**Limitacion honesta:** el signo depende de que fraccion de las bajas reales es
anunciada y que fraccion subita, y eso un panel sintetico no lo puede decidir.
Solo datos reales con deslistados (Sharadar SEP, EODHD) lo zanjan. Lo que si
queda establecido es la MAGNITUD: entre 0,004 y 0,09 de IC, es decir, del mismo
orden que la senal realista completa (0,02-0,05). No es ignorable en ninguna
direccion.

Reproducir: `python3 experiments/survivorship.py --momentum 0.02 --delist-rate 0.12 --drift-days 0`

### 9.3 Ritmo de brokers: la aritmetica antes de empezar

`data/pacing.py` implementa las cuatro reglas simultaneas de IBKR (60
solicitudes por 10 minutos, 50 concurrentes, 15 s entre identicas, 6 por
contrato en 2 s) con ventana deslizante real y reloj inyectable, mas
`assess_bulk_download`, que responde a la pregunta ANTES de lanzarse:

| Broker | 500 simbolos | 5.000 simbolos |
|---|---|---|
| IBKR | 1,4 h — viable | **13,9 h — NO viable** |
| eToro | 0,3 h (2 peticiones/simbolo) | 2,8 h — no viable |
| Alpaca Basic (IEX) | segundos | 0,4 h — viable |
| Alpaca Plus (SIP) | segundos | viable |

Por eso `ibkr_market_data` **se niega** por defecto a barrer miles de simbolos y
explica por que, en vez de morir a las tres horas con un pacing violation.
IBKR es para ejecucion; el historico viene de un proveedor con volcado masivo.

### 9.4 Almacen: Parquet + DuckDB, con las cifras medidas

Formato LARGO, no ancho, porque el panel es disperso por naturaleza: cada
simbolo cotiza en un rango distinto y los deslistados terminan antes. Medido
sobre 500 simbolos x 1.008 sesiones: 446.461 filas, 8,6 MB, **19,19 bytes por
fila**, corte transversal en **23 ms**. Extrapolando con esa densidad:

| Escenario | Filas | Tamano |
|---|---|---|
| 5.000 simbolos x 4 anos | 5,0 M | 0,10 GB |
| Sharadar completo (21.000 tickers, 25 anos) | 132 M | **2,54 GB** |

Alternativas descartadas y por que: CSV (sin tipos ni compresion), HDF5
(ecosistema fragil en ARM64), SQLite (orientado a filas: un corte transversal
toca 5.000 filas dispersas; se usa para el registro de trials, no para el
panel), formato binario propio tipo Qlib (obliga a mantener formato y
conversion), pandas en memoria (funciona hasta ~5 M filas, se rompe con el
historico completo).

### 9.5 SPA: validacion cruzada contra la implementacion de referencia

`spa_crosscheck` compara la implementacion propia con `arch.bootstrap.SPA` de
Kevin Sheppard, mapeando perdida = -IC y benchmark = 0. Resultado medido:

| Escenario | p propio | p arch | decision |
|---|---|---|---|
| nulo, 40 candidatos | 0,4953 | 0,4327 | coincide |
| nulo, 10 candidatos | 0,3253 | 0,3893 | coincide |
| **edge real entre 40** | **0,0173** | **0,0173** | coincide |
| edge debil entre 20 | 0,2487 | 0,3120 | coincide |
| autocorrelado AR(0,6) | 0,2873 | 0,3553 | coincide |

Coinciden **exactamente** en el caso que decide y difieren ~0,06 lejos del
umbral, sin cambiar ninguna decision. No se sustituye la propia porque necesita
devolver `best_index` y fijar la longitud de bloque desde la autocorrelacion de
los IC; se conserva como test de regresion permanente.

### 9.6 Arquitecturas profundas: medidas, y elegidas por coste

Se reimplementaron desde cero dos patrones de la literatura (compuerta guiada
por mercado de MASTER; mezcla por agregacion inspirada en StockMixer, adaptada
porque su "stock mixing" exige N fijo y aqui el universo cambia con los
delistings). Medido con 1 semilla, 8 epocas, 8 fechas OOS:

| arch | Ruido: IC / t | Senal: IC / t | Coste |
|---|---|---|---|
| attn | -0,077 / -1,54 | +0,432 / +11,09 | O(N^2) |
| master | -0,079 / -1,56 | +0,433 / +10,64 | O(N^2) |
| **mixer** | -0,075 / -1,69 | **+0,493 / +14,79** | **O(N)** |
| ensemble | -0,073 / -1,54 | +0,458 / +12,10 | 3x |

Las cuatro son igual de honestas sobre ruido. `mixer` puntua mejor y ademas es
la unica que escala. **Pero el defecto NO es `mixer`**: elegir arquitectura
mirando cual puntuo mejor en un dataset es una comparacion multiple encubierta,
exactamente lo que el resto del sistema existe para evitar. El defecto es
`auto`, con una regla de **coste**, no de puntuacion: `ensemble` (promedia las
tres, no selecciona) por debajo de 800 simbolos, `mixer` por encima porque la
atencion completa deja de ser practica. La regla no mira resultados y por tanto
no gasta grados de libertad.

`master` no aporto nada aqui (+0,433 frente a +0,432), y la razon es
metodologica: el generador sintetico no tiene cambios de regimen, asi que la
compuerta no tiene nada que condicionar. Eso no prueba que el patron sea inutil,
prueba que este test no puede evaluarlo.

---

## 10. v0.8 — gestion de posiciones, mandato y banda de universo

### 10.1 Motor de casos analogos (`analogy.py`)

Es la pieza funcional que pedian los tres prompts y que no existia: el ranker
ordena el universo ANTES de entrar; esto decide cuanto quedarse DESPUES.

Descriptor de 9 dimensiones, todas adimensionales (run-up a 5 y 21 sesiones,
aceleracion, sobrevolumen, volatilidad relativa, caida desde el maximo,
distancia al maximo de 52 semanas, liquidez relativa, hueco de apertura),
convertido a **rango cross-seccional por fecha**. Esa normalizacion es la que
hace comparable un +120% en una accion de 2 $ con otro en una de 80 $, y ademas
inmuniza frente a cambios de regimen: un sobrevolumen de 3x no significa lo
mismo en 2017 que en 2020, pero "percentil 95 de su dia" si.

**La regla que hace honesto el motor**: al consultar en la fecha d, solo son
admisibles los analogos cuya ventana de resultado ya habia terminado antes de d
(`c + max_horizon + embargo <= d`). Sin esa regla el motor parece magico en
backtest y falla en vivo. `test_analogues_are_strictly_point_in_time` lo
comprueba analogo a analogo.

Devuelve una **distribucion empirica**, no una decision: percentiles p25/p50/p75
a 1, 5, 10, 21 y 32 sesiones, fraccion de casos que continuaron, subida adicional
mediana, drawdown mediano y horizonte de mediana maxima. Con menos de
`min_analogues` casos no emite recomendacion.

Verificado sobre ruido puro: `p_continue` se queda entre 30% y 70%, es decir, no
fabrica direccionalidad donde no la hay.

### 10.2 Triple barrera con stop movil (`data/labels.py`)

"Vender como muy tarde en mes y medio, pero antes si ya ha subido" no es un
horizonte fijo mas largo: es una regla dependiente del camino. Se modela con
cuatro barreras — objetivo, stop fijo, stop movil desde el maximo alcanzado,
barrera vertical — mas la de delisting, obligatoria por coherencia con `pit.py`.

El stop movil no es un adorno. Medido sobre un camino que sube a +8% (sin tocar
el objetivo del 10%) y luego se desploma:

| Configuracion | Retorno | Salida |
|---|---|---|
| Stop fijo -8% | **-4,25%** | barrera vertical, dia 32 |
| Trailing -8% | **-0,75%** | dia 26 |
| Trailing -4% | **+3,33%** | dia 19 |

Las barreras se evaluan sobre CIERRES, no sobre maximos intradia: decir que el
objetivo "se toco" porque el maximo lo rozo asume una orden limitada colocada
justo ahi y ejecutada. Con cierres, la etiqueta refleja lo que un rebalanceo
diario habria obtenido de verdad.

### 10.3 Mandato de evaluacion (`mandate.py`)

La meta "75% en 45 dias o muere" se implementa como **criterio de evaluacion
posterior**, nunca como restriccion que empuje al sistema. La diferencia es la
que separa este modulo de una maquina de arruinarse: como restriccion, el
sistema que no encuentra su 75% concentra, apalanca o baja umbrales hasta
conseguirlo en la muestra — el mismo mecanismo que produjo el campeon con Sharpe
+2,70 sobre ruido puro.

Por eso el mandato **no se expone a la logica de decision**. Hay un test
(`test_target_is_not_reachable_from_decision_layer`) que inspecciona el codigo
fuente de `swarm_rank` y `portfolio.sizing` y falla si alguno importa el
mandato.

Propiedades verificadas:

* **Inicio UTC inmutable**: reiniciar el proceso no reinicia el contador. Si lo
  hiciera, bastaria con reiniciar para no fallar nunca.
* **Retorno ajustado por flujos (Modified Dietz)**: ingresar 100 $ sobre un NAV
  de 100 $ lo lleva a 200 $ y el retorno registrado es **0,00%**. Sin este
  ajuste, la meta se "cumpliria" ingresando dinero.
* **Kill-switch por drawdown evaluado ANTES que la meta y el plazo**: una parada
  de emergencia no puede quedar subordinada a que queden dias.
* **Nunca liquida solo**: devuelve una accion RECOMENDADA que exige confirmacion
  explicita, y la conserva al reevaluar (bug corregido: un estado terminal
  perdia la accion y el operador se quedaba sin saber que hacer).
* **Cash drag**: 100 $ al 4% APY durante 45 dias = 0,485 $. Como el sistema
  declara NO ACCIONABLE la mayoria de los meses, ignorarlo infravalora
  sistematicamente la decision de no operar.

### 10.4 Banda de universo: de penny a mid cap

Dos filtros INDEPENDIENTES, porque son cosas distintas: *penny stock* se define
por **precio** (SEC Rule 3a51-1, < 5 $) y *mid cap* por **capitalizacion**
(convencion SEC: nano <50M, micro <300M, small <2B, mid <10B).

`min_price`/`max_price` son point-in-time por construccion. `min_market_cap`/
`max_market_cap` **exigen capitalizacion point-in-time y el codigo se niega a
funcionar sin ella**: usar `scalemarketcap` de la tabla `tickers` de Sharadar
seria lookahead, porque clasifica por capitalizacion MAXIMA historica — una
nanocap de hoy que fue midcap en 2010 sale etiquetada "Mid". La fuente correcta
es la tabla `daily`, que esta en el plan Fundamentals (19 $/mes), no en el de
9 $. El test `test_market_cap_filter_refuses_without_point_in_time_data`
garantiza que falle en vez de degradar en silencio.

Se anade ademas `max_spread` con el estimador de **Corwin-Schultz** (JF 2012),
que recupera el spread efectivo solo con maximos y minimos diarios. Es
imprescindible si se baja al segmento penny: 2 centimos de spread en una accion
de 0,50 $ son el 4% de ida y vuelta, mas que cualquier ventaja realista.

### 10.5 Importado del documento de arquitectura externa

Cinco reglas de gestion de capital, ninguna de las cuales anade un grado de
libertad estadistico porque ninguna toca la generacion de senal: rendimiento del
efectivo ocioso, stop movil, tope de exposicion por catalizador binario (3%),
concentracion sectorial al 25% (antes 40%), y la regla de los 2 meses del art.
33.5 LIRPF como bloqueo de recompra — mas restrictiva que la wash sale de 30
dias de EE. UU., asi que manda la espanola.

Rechazado del mismo documento, con motivo: todo el stack cloud (TimescaleDB,
Airflow, Kafka, Terraform, Vault, Datadog) por sobreingenieria para un sistema
de un solo usuario cuyo panel completo son 2,54 GB; el Feature Store, porque su
proposito declarado ya lo garantiza el hecho de que el mismo codigo calcule las
features en backtest y en vivo; PPO para el meta-agente, por falta de muestra; y
la etiqueta "IA Causal" aplicada a Transformers, que aprenden asociaciones
condicionales, no causalidad.

---

## 11. v0.9 — operaciones de cargos publicos (STOCK Act)

### 11.1 Que dice la evidencia, antes del codigo

La idea "copiar lo que compran los congresistas" esta contradicha por la mejor
evidencia disponible, y el modulo se escribio SABIENDO eso:

| Estudio | Periodo | Hallazgo |
|---|---|---|
| Ziobrowski et al. 2004/2011 | 1985-2001 | Senado +12%/ano, Camara +6% — **pre-STOCK Act** |
| Eggers-Hainmueller 2013 (*J. of Politics*) | 2004-2008 | **-2 a -3%/ano** frente a indexado |
| Belmont, Sacerdote et al. 2022 | 2012-2020 | sin rendimiento superior |
| Chen-Sacerdote 2026 | hasta 2023 | por debajo o, como mucho, igualando |
| Wei-Zhou (NBER) | — | rendimiento anormal **solo en lideres**, y solo desde que lo son |

### 11.2 El problema estructural: 45 dias

La STOCK Act da 45 dias naturales de plazo (30 desde la notificacion) y la multa
por retraso son 200 $ rara vez aplicados. Con horizonte de 21-32 sesiones
(30-45 dias naturales), **cuando la operacion se publica la ventana de tenencia
ya ha transcurrido entera**. Copiar la operacion no es dificil: es imposible.

Eso reformula la hipotesis. Lo unico comprobable es si el precio reacciona a la
**divulgacion**, no a la operacion. El agente `political` prueba eso y nada mas.

### 11.3 Cuanta alfa falsa fabrica el error, medida

Escenario de control: un declarante con presciencia PERFECTA (compra los 3
valores que mas subiran, vende los 2 que mas caeran) con 31 sesiones de retraso
legal. Mismo dato, dos anclajes:

| Fecha de anclaje | IC medio | t |
|---|---|---|
| **Fecha de operacion** (fuga) | **+0,212** | **+35,1** |
| **Fecha de divulgacion** (correcto) | +0,031 | +5,1 |

Un factor de 7 en IC, enteramente fabricado. `disclosure_date` es obligatoria en
el cargador: sin ella el modulo se niega a construir el factor.

**Nota honesta sobre el 0,031 residual**: con presciencia simulada el declarante
elige los valores de mayor recorrido, que tienden a ser los mas volatiles, asi
que el factor probablemente captura **volatilidad**, no informacion politica. Es
precisamente por esto que el agente tiene que ganarse su IC dentro del pipeline
(SPA/PBO/G2) en vez de confiarse: un factor puede correlacionar con el futuro
por un confusor y seguir sin aportar nada sobre los agentes existentes.

### 11.4 Cinco factores, y por que esos

`pol_net_buy` (presion neta normalizada por volumen operado: adimensional),
`pol_n_filers`, `pol_cluster` (varios declarantes distintos sobre el mismo valor
— la unica dimension donde la literatura deja margen, porque un movimiento
coordinado es mas dificil de atribuir al azar), `pol_leader_buy` (Wei-Zhou) y
`pol_days_since` (la informacion se degrada).

Limitaciones que arrastra el dato y hay que recordar al leer cualquier resultado:
importes en RANGOS (se usa el punto medio, con su error), sin precio ni numero de
acciones, solo operaciones > 1.000 $, incluye conyuge e hijos dependientes, y
muchas llegan fuera de plazo, lo que sesga la muestra hacia quien cumple.

Fuentes gratuitas y oficiales: `disclosures-clerk.house.gov` y
`efdsearch.senate.gov`, ambas con descarga masiva. Los agregadores comerciales
revenden exactamente esto.

### 11.5 El roster pasa de 10 a 11 agentes

Anadir un agente anade una hipotesis. Eso NO es gratis y el sistema lo paga
automaticamente: SPA contrasta ahora 11 agentes mas 3 combinadores, con el
deflactor correspondiente. Verificado tras el cambio: sobre ruido puro el
veredicto sigue siendo NO ACCIONABLE (SPA p=0,882, PBO=0,99) y sobre momentum
inyectado sigue detectando (p=0,000, evidencia FUERTE). El roster sigue siendo
finito y fijado de antemano, que es lo que permite deflactarlo.

### 11.6 Lo que NO se implemento, y por que

**Mineria de patrones sin restriccion.** "Identificar todos los patrones
posibles" es la definicion del problema que esta arquitectura existe para
evitar: cuantas mas formas se buscan, mas alto sale el maximo del ruido. El
null test ya da Sharpe +2,70 sobre datos sin senal alguna. Mas busqueda no
mejora eso, lo empeora. El aprendizaje automatico legitimo ya esta presente y
acotado: los especialistas se reentrenan walk-forward en cada fecha de
evaluacion, y el motor de analogos aprende de casos pasados con la ventana de
resultado cerrada antes de la consulta.

**Rumores como fuente.** No verificables, manipulables por construccion y sin
fecha de publicacion fiable, que es justo lo que hace imposible el anclaje
point-in-time. Un rumor sin timestamp auditable no se puede backtestear sin
fuga.

---

## 12. v1.0 — el ciclo diario, y lo que la revision encontro

### 12.1 Cuatro decisiones de diseno del ciclo

**Reconciliar, no reproducir.** El ciclo no guarda ordenes pendientes para
reenviarlas: en cada ejecucion lee el estado REAL del broker y calcula la
diferencia contra el objetivo. Si el proceso murio a mitad del envio, la
siguiente ejecucion ve lo que si entro y manda solo lo que falta. Un sistema
que reprodujera un log duplicaria compras tras un reinicio; este no puede.

**El broker es la verdad.** Efectivo, posiciones y patrimonio se leen de el. El
estado local guarda solo lo que el broker NO sabe: **fecha de entrada y maximo
alcanzado** de cada posicion. Sin eso la triple barrera es inaplicable en vivo,
porque `avg_price` da el precio pero no la fecha, y el maximo no lo guarda nadie.

**Orden: parada, salidas, entradas.** El kill-switch se evalua antes que nada.
Las salidas van antes que las entradas para liberar efectivo, y porque cerrar
algo que toco su stop es mas urgente que abrir nada.

**Simulacion por defecto.** `--execute` es obligatorio para mandar una orden.

### 12.2 Renovar frente a vender y recomprar

La primera simulacion encadenada destapo un fallo: el ciclo marcaba VENDER una
posicion y el enjambre la recompraba en la misma ejecucion, asi que no se emitia
orden y **el informe se contradecia**. Si la hubiera emitido, habria pagado el
spread dos veces sin cambiar la exposicion.

La resolucion distingue por MOTIVO de salida, y no es arbitraria:

* Salida por **objetivo o vencimiento** con el enjambre aun a favor -> **renovar**:
  se mantiene la posicion y se reanclan las barreras al precio de hoy. Se evita
  el viaje de ida y vuelta y la disciplina se conserva, porque el nuevo stop
  cuelga del precio nuevo.
* Salida por **stop fijo o movil** -> **veto de recompra en ese ciclo**. Vender
  por stop y recomprar el mismo dia es la conducta que desangra cuentas. Y en
  Espana hay ademas motivo fiscal: recomprar valores homogeneos antes de dos
  meses impide computar la perdida (art. 33.5 LIRPF).

### 12.3 Modo repeticion (`as_of`)

La guarda de frescura aborta si la ultima sesion tiene mas de 4 dias naturales,
lo que es correcto en produccion pero impedia **validar el ciclo sobre el
pasado**: toda repeticion historica abortaba. `as_of` mide la antiguedad contra
una fecha simulada en vez de contra el reloj, y permite reproducir un mes
completo de ciclos antes de arriesgar nada.

### 12.4 Hallazgos de la revision del proyecto completo

**Tres limites estaban declarados pero nunca aplicados.** `wash_sale_days`,
`max_binary_catalyst_pct` y `max_sector_pct` existian como campos de
`RiskLimits` desde que se importaron del documento externo, y ninguno se
comprobaba en `check_order`. Configuracion muerta, que es peor que no tenerla
porque aparenta proteccion. Ahora los tres se aplican y estan cubiertos por
tests.

**El tope sectorial estaba invertido.** `guards.py` tenia 0,25 y
`portfolio/sizing.py` tenia 0,40: la capa operativa podia proponer carteras que
la capa de seguridad iba a rechazar. Alineados a 0,25, con un test que falla si
vuelven a divergir. El tope POR NOMBRE, en cambio, es correcto que difiera
(guardas 0,35 frente a sizing 0,25): la guarda es el limite ultimo, el sizing es
el objetivo operativo, y el orden correcto es que la guarda sea la mas holgada.
Tambien hay test.

**El ciclo deshacia los topes al renormalizar.** Al repartir el presupuesto
libre entre las entradas, un peso llego al 42,8% y el guardarrail tuvo que
rechazar la orden. Corregido con reparto por water-filling, el mismo patron que
ya se uso en `ensemble/stacking.py`.

**Una deprecacion de pandas 4** (`Timestamp.utcnow`) en la guarda de frescura.

### 12.5 APIs necesarias para operar el ciclo

| Para | Que hace falta | Coste |
|---|---|---|
| Ejecutar ordenes | Alpaca paper: `ALPACA_API_KEY`, `ALPACA_SECRET_KEY` | gratis |
| Historico con deslistados | Sharadar SEP (volcado masivo) | 9 $/mes |
| Dinero real | ademas `SWARMTRADER_ALLOW_LIVE=1` y `paper=False` | — |
| Programacion | cron o launchd, 22:15 hora peninsular | — |

Ninguna API nueva respecto a v0.9: el ciclo usa lo que ya estaba integrado.

---

## 13. v1.1 — universo multi-plaza (Alpaca + eToro + IBKR)

### 13.1 Tres supuestos que la peticion daba por ciertos y no lo son

**No toda accion se puede comprar en toda plaza.** Un universo sin etiqueta de
ejecutabilidad produce recomendaciones imposibles de operar, que es peor que no
producirlas: parece que tienes una oportunidad y no la tienes.

**El capital NO es fungible entre brokers.** No hay un saldo unico: hay una
cuenta en cada plataforma. 80 $ en Alpaca mas 20 $ en eToro **no son** 100 $
para una compra en eToro, y transferir tarda dias. Por eso `VenueRouter`
comprueba efectivo POR PLAZA.

**La union no es la suma.** Las tres son mayoritariamente renta variable
estadounidense. Lo que anaden eToro e IBKR al margen son bolsas no
estadounidenses (eToro declara 6.000+ acciones en 20-25 bolsas, advirtiendo que
no toda accion de cada bolsa esta disponible; y la disponibilidad varia por pais
y entidad legal).

### 13.2 El coste de ida y vuelta, que es lo que realmente decide

Medido con `round_trip_cost`, contando el spread una vez (compras en ask, vendes
en bid) y la conversion de divisa dos veces:

| Posicion | Alpaca | eToro US | eToro EU | IBKR |
|---|---|---|---|---|
| 25 $ | 0,05% | 0,10% | **17,60%** | **8,04%** |
| 100 $ | 0,05% | 0,10% | 5,60% | 2,04% |
| 500 $ | 0,05% | 0,10% | 2,40% | 0,44% |
| 1.000 $ | 0,05% | 0,10% | 2,00% | 0,24% |

Tamano minimo para que el coste no supere el 2%: Alpaca y eToro US **0 $**,
IBKR **102 $**, eToro EU **1.000 $**.

Contexto que lo convierte en decisivo: el retorno MEDIANO a 32 sesiones con
Sharpe 1 y volatilidad 35% es **+3,8%**. Un coste del 17,6% de ida y vuelta no
reduce la ventaja, la invierte.

Los parametros de coste son DATOS (`VenueSpec`), no constantes incrustadas: la
comision de eToro depende del pais, la entidad legal y la bolsa, y deben
verificarse contra la cuenta propia antes de confiar en ellos.

### 13.3 `VenueRouter`: mismo interfaz, tres plazas

Implementa `BrokerAdapter`, asi que **`cycle.py` funciona sin tocar una linea**.
Si la logica de plazas viviera dentro del ciclo, cada regla de riesgo habria que
duplicarla por plaza.

Tres reglas que impone y que un agregador ingenuo se salta: **ejecutabilidad**
(solo enruta a plazas que ofrecen el simbolo), **efectivo por plaza** (no
agregado), y **umbral de coste** (se NIEGA a enrutar por debajo del tamano
viable, en vez de ejecutar una operacion que nace perdiendo). Esa tercera regla
es la que garantiza la propiedad central, cubierta por test: **anadir plazas no
puede empeorar el sistema**.

Demostracion con 100 $ repartidos (80 Alpaca / 20 eToro EU / 0 IBKR), posiciones
de 25 $:

```
AAPL  -> alpaca    enrutada (coste 0.05%)
SAN   -> None      ninguna plaza viable: coste excesivo o sin efectivo
SOFI  -> alpaca    enrutada (coste 0.05%)
7203  -> None      no operable en ninguna plaza conectada
  AAPL: etoro_eu descartada, coste 17.6% > 2% (minimo viable 1,000 $)
  AAPL: ibkr descartada, coste 8.0% > 2% (minimo viable 102 $)
```

### 13.4 Alcance honesto de lo implementado

`--venues` amplia lo que se **analiza**, no donde se ejecuta. La union de
universos funciona con los tres; la **ejecucion** sigue yendo por Alpaca, porque
es el unico con adaptador de ordenes probado. No se escribieron adaptadores de
ejecucion para eToro e IBKR por dos razones: no se pueden probar sin cuentas
reales, y al capital actual el filtro de coste los rechazaria igualmente.
`ibkr_market_data` ya existe pero se niega a la ingesta masiva por su ritmo
(13,9 h para 5.000 simbolos).

La normalizacion de tickers entre plazas es deliberadamente **conservadora**:
prefiere no cruzar dos simbolos a cruzar mal dos empresas distintas, porque un
cruce erroneo mete precios de una empresa en el historico de otra y **eso no lo
detecta ningun test estadistico posterior**.

## 14. v1.2 — geometria de la informacion

### 14.1 Que es y que no es

La metrica de Fisher mide la separacion entre dos distribuciones en unidades de
distinguibilidad estadistica, y es invariante a como se parametricen. Es una
regla de medir, no una fuente de senal: no anade informacion que el sistema no
tuviera. Por eso solo se ha puesto donde ya habia una distancia o hacia falta
una, y cada uso se ha medido. `stats/infogeo.py` contiene tres formulas cerradas
(gaussiana univariante, Bernoulli, gaussiana centrada sobre matrices definidas
positivas), verificadas contra identidades conocidas: invariancia afin, limite
local d^2 = 2 KL, coincidencia entre familias en dimension 1.

### 14.2 Una premisa que hubo que corregir

La propuesta inicial era vigilar la deriva de los factores entre el
entrenamiento y el dia evaluado. **No habria detectado nada**: los factores se
normalizan a z-score por fecha, asi que su media y su desviacion son las mismas
todos los dias por construccion. `test_factor_marginals_are_blind_to_a_volatility_break`
lo fija: con la volatilidad triplicada, el factor sigue con desviacion ~1.

El monitor (`stats/drift.py`) vigila lo que esa normalizacion deja fuera:

| Canal | Que resume cada fecha | Variedad | Por que |
|---|---|---|---|
| dispersion | (mediana, MAD) de la seccion cruzada de retornos a 21 sesiones | gaussiana (hiperbolica) | es lo que el z-score BORRA: un z=+2 es +8% un mes y +40% otro |
| dependencia | matriz de correlacion entre factores | matrices definidas positivas | es lo que el z-score NO PUEDE borrar |

El umbral no es absoluto: se calibra con las distancias de las propias fechas de
entrenamiento a su centro, y los dos canales reparten alpha por Bonferroni.
Entra en `swarm_rank` como **G0**: solo puede dejar el veredicto en NO
ACCIONABLE. Un test inspecciona el codigo de agentes, stacking y sizing y falla
si alguno importa el monitor.

### 14.3 Lo que midio el experimento (`experiments/infogeo.py`)

Criterios fijados en la cabecera del script antes de ejecutar. Resultados
completos en `docs/results/infogeo_2026-11-11.json`. Los datos reales son 1.000
acciones al azar de un panel de 5.550 cotizadas HOY en EE. UU. (Yahoo,
2018-01-02 a 2026-09-30): **universo de supervivientes**, valido para comparar
dos reglas entre si, no para afirmar rentabilidad.

**Monitor de deriva.**

| Medida | Resultado |
|---|---|
| Tamano con datos estacionarios, alpha nominal 1% | dispara el **1,9%** de las fechas (40 de 2.140) |
| Volatilidad x3: deteccion a 3 / 5 / 21 sesiones | 30% / 95% / 100% |
| Volatilidad x2: deteccion a 5 / 21 sesiones | 20% / 50% |
| Volatilidad x1,5 o menor | indistinguible de la linea base |
| Correlacion inyectada 0,4 entre dos factores, 300 nombres, a 3 sesiones | 100% |
| Correlacion inyectada 0,2, 300 nombres, a 10 sesiones | 60% |
| Correlacion inyectada 0,4, 50 nombres, a 10 sesiones | 30% |
| Datos reales: fechas marcadas | **3,2%** (58 de 1.793), en tres episodios |

El tamano real duplica el nominal porque los estados estan muy
autocorrelacionados: tres anos de referencia no contienen muestra independiente
suficiente para fijar un cuantil del 99%. La linea base de la tabla de potencia
(sin ruptura) sale en 15% porque 3 de las 20 semillas terminan por azar en un
extremo genuino; una comprobacion aparte sobre 300 semillas dio 1,7% en esa
misma fecha, sin efecto de borde.

Los tres episodios reales: 2019-12-31 a 2020-01-03 (dependencia, 3 sesiones);
**2020-03-10 a 2020-04-17** (dispersion y dependencia, 24 sesiones); 2025-01-29
a 2025-03-13 (dependencia, 31 sesiones). No marco 2022 ni abril de 2025.

**El valor economico del veto no esta demostrado.** La hipotesis pre-registrada
era que el rank IC del modelo base es menor en las fechas marcadas. Solo 2 de
las 85 fechas de evaluacion cayeron en dias marcados: no se puede contrastar. El
veto queda activo por coherencia con el resto del sistema (sin evidencia de que
el modelo sirva en un estado nunca visto, no se actua), con coste medido del
2-3% de los dias. `--drift-mode warn` informa sin vetar; `--drift-alpha 0` lo
apaga.

**Distancia del motor de analogos.** Tres reglas en `AnalogyConfig.metric`:
`euclid` (la original), `fisher_rank` (la misma en la coordenada de Fisher de
cada percentil, 2 arcsin sqrt u, que estira las colas) y `fisher_gauss`
(distancia de Rao entre las gaussianas de los retornos de 21 sesiones). Medida:
CRPS de la distribucion de retornos de los 50 vecinos frente al retorno
realizado; 1.675 consultas reales en 335 fechas sobre un fondo de 1,03 millones
de casos.

| Datos, horizonte | euclid vs azar | fisher_rank vs euclid | fisher_gauss vs euclid | T2: ¿alguna de Fisher mejora? |
|---|---|---|---|---|
| Reales, 21 sesiones (principal) | +1,67% (p=0,0045) | +0,27% | −1,22% | p=0,283: no |
| Reales, 5 sesiones (secundario) | +1,08% (p=0,039) | +0,62% | +0,88% | p=0,033 |
| Eventos sinteticos, 21 sesiones | +1,95% (p=0,012) | −0,09% | −5,08% | p=1,00: no |
| Ruido sintetico, 21 sesiones | +0,40% (p=0,29) | +0,19% | +0,81% | p=0,28: no |

Tres lecturas. Primera: el motor de analogos **si contiene informacion** frente a
elegir casos del mismo signo al azar, tambien con datos reales, aunque el efecto
es pequeno (1,7% de CRPS). Segunda: en el horizonte principal ninguna metrica de
Fisher bate a la euclidea, asi que por la regla pre-registrada **el valor por
defecto no cambia**. Tercera: a 5 sesiones las dos mejoran con el mismo signo en
las dos mitades de la muestra, pero es la metrica secundaria y con dos
horizontes probados p=0,033 no sobrevive a la correccion. `fisher_gauss` mejora
incluso sobre ruido a 5 sesiones (p=0,058), lo que indica que gana por emparejar
volatilidades (CRPS premia la anchura bien calibrada), no por direccion.
`fisher_rank` queda como candidata a confirmar con datos point-in-time.

**Combinacion de agentes.** Media en logits (e-geodesica) frente a media lineal
(m-geodesica) de las puntuaciones: diferencia de IC −0,005 con datos reales
(p=1,00) y +0,006 con senal sintetica (p=0,13). No se adopta. El nivel de IC de
esa prueba sobre datos reales (~0,09) **no es interpretable**: universo de
supervivientes que incluye valores de 1 USD, sin costes; lo unico valido es la
diferencia entre las dos reglas.

### 14.4 Lo que se evaluo y no se implemento

- **Gradiente natural / K-FAC en `deep_rank`.** No se pudo ejecutar ni probar en
  este entorno (sin PyTorch) y no hay evidencia de que la red pequena actual
  tenga un problema de optimizacion. Mejoraria velocidad de convergencia, no
  informacion.
- **Kelly.** Ya es la pieza informacional del dimensionado; no cambia.
- **Mas canales en el monitor.** Cada canal divide alpha. Dos es lo que justifica
  la estructura del problema (lo que el z-score borra y lo que no puede borrar).

## 15. v1.3 — auditoria sobre datos reales y filtro G4

Informe completo, con todas las cifras y fuentes: `docs/AUDITORIA_GEOMETRIA.md`.
Resultados: `docs/results/geometry_audit_2026-11-11.json` (pre-registrado) y
`docs/results/audit_followup_2026-11-11.json` (exploratorio). Datos: 1.000
acciones al azar de un panel de supervivientes (2018-2026), 85 fechas mensuales
fuera de muestra, sin costes.

### 15.1 Lo que encontro

El enjambre tiene rank IC 0,075 (t = 4,6) sobre datos reales, pero **todo el
poder predictivo esta en la mitad baja del ranking** (IC +0,080 en la mitad baja,
-0,003 en la alta). Los 5 primeros, que son lo unico que se compra, no baten al
universo (exceso -0,15% a 21 sesiones, t = -0,29); los 5 ultimos lo pierden por
8,8% (t = -4,6). El sistema elige valores de baja volatilidad: evita desastres y
renuncia a las subidas. Con la v1.2 el veredicto sobre estos datos era
ACCIONABLE.

Entrenar con una etiqueta orientada a subidas grandes (pertenecer al quintil
superior) triplica la probabilidad de un +25% mensual en los 5 primeros, casi
cuadruplica la de un -15% y deja el exceso medio en -6,2% (t = -3,7), y eso en
un universo sesgado a favor.

### 15.2 Lo que cambio

| Cambio | Motivo |
|---|---|
| **G4** en `swarm_rank`: contraste del exceso fuera de muestra de los `top_k` primeros sobre el universo | G1-G3 validan el IC de toda la seccion cruzada, que puede salir entero de lo que no se compra |
| `sharadar_market_data` ya no filtra por ultimo precio | Eliminaba del historico a los valores que quebraron: sesgo de supervivencia en el adaptador que existe para quitarlo |
| `MarketData.nominal`; los filtros de precio lo usan si existe | El precio ajustado incorpora splits futuros. Afecta al 1,1% de las filas, con mediana de retorno -12,3% |
| `StackConfig.shrink_on_disagreement = False` | Heuristica sin validar; medida, no mejora el IC (p = 1,00) y lo empeora algo (p = 0,027) |

### 15.3 Geometria: lo que se probo y no entro

Ninguna de las seis familias de contraste es significativa (Holm >= 0,79).
Ridge sigue siendo el modelo base (bate a gradient boosting); las features
geometricas por accion son informativas solas (IC 0,060) y redundantes con los
14 factores; el canal de forma con Wasserstein duplica los dias marcados sin
detectar mas episodios; la norma topologica replica 2000 y 2008, falla 2020 y no
predice varianza una vez controlada la varianza pasada (t = 1,43); la curvatura
de Ollivier-Ricci no es solo correlacion media (0,61) pero no se contrasto su
valor predictivo; la media de rangos ya es la media de Frechet bajo la distancia
de Spearman y ni la mediana ni la media en logits la baten.

Sobre G0: detecta el mismo episodio (COVID) que una linea base trivial de
volatilidad, tres sesiones mas tarde. El canal de dispersion solo lo hace sin
marcar ningun dia fuera de episodios. No se cambia el valor por defecto.

## 16. v1.4 — modo de decision por conviccion

Peticion: que el sistema, aunque no pueda demostrar ventaja, compre «las mas
probables o las mejores de todas las analizadas», una, dos o la mejor
combinacion, sin barreras de diversificacion, pudiendo poner todo el capital en
una sola accion. Es opcional (`--decision conviction`); el modo por defecto
(`gated`) no cambia.

### 16.1 Dos preguntas distintas

El veredicto G0-G4 responde «¿hay evidencia de que el ranking elija bien?». El
modo conviccion responde «si hay que elegir de todos modos, ¿que es lo mejor con
lo que se sabe?». La segunda no invalida la primera: el veredicto se calcula y
se imprime igual, y el informe dice «ventaja NO demostrada» cuando toca.

### 16.2 Criterio: crecimiento optimo en el simplex

`portfolio/conviction.py` maximiza E[u(b . X)] con b >= 0 y sum(b) = 1, con u
CRRA (gamma = 1 es Kelly). No hay tope por nombre: el optimo es una esquina (todo
a una accion j) exactamente cuando E[X_i / X_j] <= 1 para todas las demas
(Kuhn-Tucker; Cover y Thomas, teorema 16.2.1). Con dos acciones de volatilidad
sigma y correlacion rho, la mejor se lo lleva todo si su ventaja supera a
sigma^2 (1 - rho), alrededor de un 1 % mensual con sigma = 12 % y rho = 0,3.
Con la evidencia real (exceso aproximadamente nulo) lo esperable es repartir
entre pocos nombres de baja volatilidad, no «todo a una».

Solver: actualizacion multiplicativa de Cover (1984) con paso adaptativo,
aceptado solo si sube el objetivo, mas pasos de Frank-Wolfe con busqueda lineal
cuando el multiplicativo se atasca o cuando la accion que mas mejoraria tiene un
peso colapsado al suelo numerico. Devuelve una cota certificada de la distancia
al optimo (ln max_i g_i en el caso logaritmico; brecha de Frank-Wolfe en
general). El universo entero se trata por generacion de columnas: la condicion
de Kuhn-Tucker se comprueba en TODAS las acciones, no en un «top N».

### 16.3 Distribucion predictiva

| Pieza | Decision y motivo |
|---|---|
| Media por puntuacion | Predicciones walk-forward fuera de muestra del propio enjambre; tramos por percentil DENTRO de cada fecha (la escala de `stack` cambia entre fechas); sin forzar monotonia, porque sobre datos reales la parte alta no era mejor |
| Encogimiento empirico-bayesiano | lambda_k = tau^2 / (tau^2 + se_k^2), tau^2 por momentos. Con ~18 fechas y 10 tramos el tramo con mejor suerte aparenta varias decimas al mes de exceso aunque sea ruido; sin encoger, Kelly apuesta justo por ese tramo. El bootstrap bayesiano por si solo no lo evita: bajo utilidad logaritmica la incertidumbre de la media solo penaliza del orden de 0,006 % |
| Incertidumbre | Bootstrap bayesiano (Rubin, 1981) sobre las fechas, con la posterior normal-normal de cada tramo |
| Dependencia | Bootstrap estacionario de secciones cruzadas diarias compuesto al horizonte: correlaciones y colas reales |
| Media exacta | El residuo se reescala para que la media de cada accion sea exactamente su media predictiva: no cuenta dos veces la deriva historica y quita el ruido de Monte Carlo que el optimizador perseguiria entre acciones casi iguales |
| Empates | Con gamma = 0 y medias iguales (mapa plano), el capital se reparte entre los empatados en vez de dejar que decida el redondeo |

### 16.4 Integracion

| Pieza | Cambio |
|---|---|
| `swarm_rank` | `decision`, `conviction`, `held`, `exclude`; usa el mismo combinador que el veredicto; bloque de informe propio |
| `cycle` | Entradas sin `max_positions` ni tope por nombre o sector; las posiciones abiertas quedan fijas y se reparte solo el presupuesto libre; se excluyen antes los valores que las guardas o el broker rechazarian (precio, liquidez, no fraccionables), porque una orden rechazada deja capital sin invertir |
| Se mantiene | Parada por drawdown del mandato, veto de recompra tras stop, triple barrera, exposicion bruta 1,0 (sin margen), simulacion por defecto, doble llave para dinero real |
| `rebalance_to` (ambos modos) | Liquidar, reducir, esperar ejecucion y despues comprar con el poder de compra real y un colchon del 0,5 %; las reducciones no pasan por las guardas de entrada; lo que no se puede financiar queda `deferred` |
| `AlpacaBroker` | Poder de compra del campo mas conservador; nocional truncado a centimos con minimo local de 1 USD; `settle` espera a las ventas solo con el mercado abierto; `untradable` excluye no fraccionables e inactivos; clientes inyectables para tests |
| CLI | `--decision`, `--risk-aversion`, `--allow-cash`, `--scenarios` en `swarm` y `cycle`; el codigo de salida de `swarm` sigue ligado al veredicto |

Con el mercado cerrado (el ciclo corre a las 22:15) las ventas quedan en cola y
no liberan poder de compra hasta ejecutarse: una rotacion completa tarda dos
ciclos.

### 16.5 Medicion pre-registrada

`experiments/conviction_measure.py`; resultados en
`docs/results/conviction_measure.json`. Datos sinteticos (`load_synthetic`), 10
semillas, 60 acciones, 6 fechas mensuales no solapadas por semilla, retorno
realizado a 21 sesiones, sin costes. Criterios fijados antes de ejecutar.

| Hipotesis | Resultado | Cumple |
|---|---|---|
| H1 sin senal: lambda medio < 0,2 y no bate al universo | lambda 0,2015; crecimiento -0,43 % frente al universo (p = 0,83) | No (por el criterio de lambda) |
| H2 con senal: concentra mas y bate al universo | 3,6 frente a 8,4 posiciones efectivas; +23,2 puntos (p = 1,8e-25) | Si |
| H3 encogimiento: menos concentracion y no peor | 8,4 frente a 4,0 posiciones efectivas; sin encoger +0,31 % (p = 0,27) | Si |
| H4 mas aversion al riesgo, mas reparto | 5,6 / 6,0 / 6,5 / 7,2 posiciones efectivas con gamma 0,5 / 1 / 2 / 4 | Si |

Observacion no pre-registrada: con senal, equiponderar las 5 primeras por
puntuacion (+32,4 %) supera a la cartera de conviccion (+27,3 %). Con 60
acciones el mapa tiene 7 tramos y no distingue dentro del superior. Tramos mas
finos en la parte alta podrian ayudar cuando hay senal, pero con datos reales la
parte alta no era mejor; no se cambia nada sin medirlo sobre datos
point-in-time. Sin senal la conviccion queda por debajo del universo (-0,43 %,
no significativo): con medias iguales elige la cartera de menor riesgo, que en
este sintetico es la de beta baja.

### 16.6 Limites

Nada de esto demuestra ventaja: sobre datos reales el veredicto sigue siendo NO
ACCIONABLE y la medicion de este modo es sintetica. No se ha ejecutado contra la
cuenta paper de Alpaca. Quedan sin verificar en la documentacion de Alpaca la
liquidacion T+1 y las violaciones de buena fe al rotar en cuentas pequenas, y si
una orden por nocional se acepta con el mercado cerrado; el diseno no depende de
ello (lo que se rechace, el ciclo siguiente lo reconcilia).

## 17. v1.5 — decision con plazo variable, salidas por accion y plan de ordenes

La v1.4 de este repositorio (seccion 16) y una rama paralela desarrollada fuera
de el (13-nov-2026) resolvieron la misma peticion por separado. La v1.5 parte de
este repositorio e incorpora de la rama paralela lo que aqui no existia. Nada de
la seccion 16 se ha retirado ni ha cambiado de valor por defecto.

### 17.1 Que pregunta responde

`swarmtrader decide` responde, para la ultima sesion: que comprar (cartera de
crecimiento optimo, seccion 16, siempre invertida salvo `--allow-cash`), a que
plazo (3, 5, 10, 15, 21 o 31 sesiones; 31 ~ 45 dias naturales), cuando entrar
(apertura de la sesion siguiente) y cuando salir de cada accion (objetivo, stop,
stop movil y plazo maximo propios). Escribe un plan de ordenes JSON que ejecuta
`swarmtrader execute` (o un agente que lea el JSON) tras la apertura.

### 17.2 Lo que se ha incorporado

| Pieza | Que hace |
|---|---|
| `decision.py` | Ejecuta el enjambre (en modo `gated`: la asignacion la hace el), construye un banco de escenarios por plazo y asigna en cada uno; elige el plazo de mayor crecimiento por sesion, ln(1 + CE) / h, y entre los que quedan dentro del error de Monte Carlo del mejor, el MAS LARGO (rota menos; el coste es la parte con mas incertidumbre). Tabla del universo con valoracion 0-1 (percentil del crecimiento por sesion en su mejor plazo) y catalizadores (agentes con rango >= 0,6) |
| `exits.py` | Plan de salida por accion: trayectorias de los casos fuera de muestra del mismo tramo de puntuacion, en unidades de volatilidad diaria; rejilla de 360 planes (objetivo x stop x movil x plazo, en sigma raiz(T)); criterio de renovacion-recompensa E[ln R neto] / E[sesiones]; validacion cruzada en dos mitades temporales y adopcion de la rejilla completa solo si bate a «solo plazo» con probabilidad >= 95 % en bootstrap bayesiano por fechas |
| `market_calendar.py` | Calendario NYSE (festivos y cierres) y hora de apertura en Madrid con el desfase de cambio de hora |
| `execution/plan.py` | Plan JSON con `plan_id` (huella SHA-256 del contenido); `execute_plan` delega en `rebalance_to` (vender, reducir, esperar, comprar con el poder de compra real); libro de posiciones con el plan de salida de cada una; dinero real solo con `confirm == plan_id` ademas de la doble llave |
| `cycle.py` | `LedgerEntry` admite plan de salida propio (`horizon`, `target_pct`, `stop_pct`, `trail_pct`); `_evaluate_exit` lo usa cuando existe y, si no, las barreras globales. None en el plan = sin esa barrera |
| `conviction.py` | Costes por escenario (`costs`), `standalone` (distribucion de cada accion por separado), `build_bank` reutilizable, `market_level` y `tau` (17.3) |
| `data/loaders.py` | `load_long_panel` (Parquet largo con cierre sin ajustar opcional y pre-filtro de memoria) y `_sanitize_prices` (precios no positivos, picos aislados, saltos de mas de x20) |
| `swarm_rank` | `oos` en el resultado siempre: predicciones fuera de muestra del combinador del veredicto, con `symbol` |
| CLI | `decide` y `execute`; `decide` excluye antes de optimizar lo que las guardas o el broker rechazarian, igual que el ciclo |

### 17.3 Decisiones al fusionar

| Conflicto | Decision | Motivo |
|---|---|---|
| Optimizador | El de este repositorio (Cover + Frank-Wolfe con cota certificada) | Garantia mas fuerte y ya probada |
| Media por puntuacion | Tramos por percentil DENTRO de cada fecha y reescalado a la media exacta (este repositorio) | La rama paralela usaba cuantiles globales del valor de la puntuacion, cuya escala cambia entre fechas |
| Nivel de mercado | `market_level`: "oos" por defecto (seccion 16); `decide` usa "history" | Para un solo plazo, las fechas fuera de muestra dejan que un mercado esperado negativo lleve al efectivo (lo prueba `test_cash_is_used_only_when_allowed`). Al comparar plazos ese nivel es casi todo ruido y distinto en cada plazo: elegir plazo seria elegir el ruido mas favorable. "history" usa la media diaria de toda la ventana, compuesta al plazo, con numeros aleatorios comunes |
| Tau del encogimiento | `tau`: "point" (este repositorio) por defecto; "integrated" disponible | Criterio fijado ANTES de medir: adoptar "integrated" solo si pasaba H1-H4, "point" fallaba alguna, y sin ruido no era peor en mas de 2 errores estandar. Resultado (`docs/results/conviction_measure_tau_integrated.json`): "integrated" tambien falla H1 (lambda 0,35; nunca encoge exactamente a 0), asi que se queda "point". En ruido su crecimiento fue +0,46 % mejor (EE 0,27 %), no significativo |
| Ejecucion | `rebalance_to` de este repositorio; el plan JSON se conserva encima | Un solo camino de ordenes para el ciclo y para `execute`; recorta posiciones que pesan de mas (la rama paralela solo compraba o completaba) y aplaza lo no financiable |
| Adaptador Alpaca | El de este repositorio, sin cambios | Truncado a centimos, minimo local de 1 USD, `settle` segun horario, `untradable`, tests con clientes inyectados. Los metodos equivalentes de la rama paralela (`wait_closed`, `submit_qty` para acciones enteras) se descartan: los no fraccionables se excluyen en el analisis |
| `cycle --decision conviction` | Sin cambios (plazo fijo del enjambre) | Cubierto por seis tests de integracion. El plazo variable y las salidas propias estan en `decide` + `execute`; el ciclo diario respeta despues el plan de salida de cada posicion |

Regresion verificada: con los valores por defecto, `conviction_measure.py`
produce en esta maquina las mismas cifras, fila a fila, con el modulo original
que con el fusionado. Las diferencias de orden 1e-7 frente al JSON publicado
vienen de la aritmetica de coma flotante del equipo en que se genero.

### 17.3b Correcciones tras la revision automatica de la PR

La revision de la PR (Codex) senalo seis defectos; los seis se comprobaron
contra el codigo, eran reales y estan corregidos con un test cada uno.

| Defecto | Correccion |
|---|---|
| Una compra aceptada pero aun sin llenar perdia su plan de salida: el ciclo la daba de alta con las barreras globales | `PositionLedger.pending`: el plan queda pendiente y `sync` lo asigna al aparecer la posicion, anclado al precio real; se descarta a los 7 dias si la compra no se llena |
| `execute` no consultaba el mandato: con la parada por drawdown activa podia abrir una cartera nueva | `execute_plan(mandate=...)` lo arranca y lo marca como el ciclo; si no esta en RUNNING solo vende o reduce. En simulacion lo consulta sin escribir |
| La validacion cruzada de las salidas no purgaba: con fechas cada 21 sesiones y trayectorias de 31, las vecinas a la frontera compartian 10 sesiones de retornos | `_crossfit_folds`: del ajuste se excluye toda trayectoria cuya ventana [entrada, entrada + 31] se solape con alguna de evaluacion |
| Con historia corta (predicciones cerradas a 21 sesiones, ninguna a 31) `decide` fallaba al pedir planes de salida | Banco vacio: no hay plan propio, se informa y el ciclo usa las barreras globales |
| Con datos PIT las trayectorias de salida se congelaban en el ultimo precio de una accion dada de baja | `PathBank(listings=...)` aplica el retorno terminal de baja, igual que `forward_returns_pit` |
| Con `--allow-cash`, un optimo de 100 % efectivo se trataba como fallo y no se escribia plan | Es una decision valida: `DecisionResult.cash`; el plan no compra y vende lo que haya |

### 17.4 Hora de entrada

Apertura de la sesion siguiente al analisis, orden de mercado por nocional. (1)
La senal usa el cierre: no se puede actuar antes. (2) Fuera de horario, Alpaca
exige orden limitada para fraccionarias, con poca liquidez y spreads amplios.
(3) Con datos reales, retrasar la entrada una sesion entera no cambio la
capacidad predictiva (rank IC 0,0746 frente a 0,0749; `docs/AUDITORIA_GEOMETRIA.md`).

### 17.5 Prueba con datos reales

Cierre del 12-nov-2026, 3.313 acciones de EE. UU. de Yahoo (solo
supervivientes; pre-filtro de memoria de 1 M$ de volumen y 1 USD), 363 s y
4,7 GB. Informe y JSON en `docs/results/decision_real_2026-11-12_v15.*`.

| Medida | Resultado |
|---|---|
| Veredicto del enjambre | NO ACCIONABLE (G1 p = 0,039; G4 p = 0,282; evidencia NINGUNA) |
| Senal retenida por el encogimiento ("point") | 0 % en 3-15 sesiones, 10 % a 21, 6 % a 31 |
| Plazo elegido | 31 sesiones (crecimiento +0,054 %/sesion; las diferencias entre plazos quedan cerca del error de Monte Carlo) |
| Cartera | 23 posiciones, mayor peso 23,8 % (CHT); retorno esperado +1,7 % a 31 sesiones, igual para todas |
| Alternativas, mismos escenarios | optimo +1,7 %; todo en CHT +1,6 %; 5 primeras equiponderadas +1,6 % |
| Top 10 | CHT, TXNM, KO, FTS, L, PG, ATO, DUK, JNJ, EPD |
| Salidas | En 17 de 23 posiciones la rejilla completa no supera a «solo plazo» (21 de 23 antes de purgar la validacion cruzada, 17.3b); la cartera no cambia |
| Ejecucion simulada, 100 USD | 23 compras, 99,49 USD invertidos (colchon 0,5 %) |
| Excluidas del optimizador | 448 acciones sin 1.260 sesiones de historia completa |

Lectura: con solo precio y volumen la puntuacion del enjambre no distingue
ganadores (la senal retenida es casi nula), asi que la mejor eleccion forzada
es la cartera de menor riesgo con medias iguales, no una apuesta. La rama
paralela, con `tau="integrated"` y 756 sesiones de historia, obtuvo sobre el
mismo panel 24 posiciones con mayor peso del 14,8 % y senal retenida del
12-29 %: la diferencia sale de esos dos ajustes, no de la seleccion. Que una
accion destaque exige informacion nueva (eventos), no mas matematicas sobre las
mismas variables.

### 17.6 Limites

Las mismas de la seccion 16, mas: la eleccion del plazo y del plan de salida es
un maximo sobre varias estimaciones (sesgo del ganador, mitigado con el
desempate al plazo mas largo y el bootstrap por fechas); las barreras de salida
no entran en los escenarios del optimizador; los costes son una tabla por
tramos de precio (0,05 % desde 5 USD, 1 % entre 1 y 5, 3 % por debajo), no
cotizaciones reales; `cycle --decision conviction` no usa plazo variable; con
`lookback` = 1.260 sesiones las acciones con menos de 5 anos de historia
completa no son asignables (448 de 3.313 en la prueba real), un efecto de
seleccion contra las empresas jovenes que no se ha medido.

## 5. Flujo completo

```
universo (CSV / watchlist / yfinance)
  -> data/features.py        14 factores, z-score por fecha, solo pasado
  -> data/signals.py         senales externas con lag point-in-time  [opcional]
  -> data/congress.py        STOCK Act, anclado en fecha de DIVULGACION  [opcional]
  -> data/pit.py             panel point-in-time: deslistados + retorno terminal
  -> data/store.py           Parquet particionado + DuckDB
  -> agents/specialists.py   10 agentes (9 clasicos + 1 profundo), walk-forward purgado
  -> models/trainer.py       GRU + atencion cross-seccional, ListNet, bagging semillas
  -> ensemble/stacking.py    eq / best_prior / stack  (pesos solo del pasado)
  -> stats/drift.py          G0: ¿el mercado de hoy esta dentro del soporte de entrenamiento?
  -> swarm_rank.py           G1 SPA · G2 stack-vs-eq · G3 PBO · G4 top-K vs universo · evidencia OOS
  -> stats/calibration.py    score -> P(exito), isotonica, solo si hay evidencia
  -> portfolio/sizing.py     Kelly fraccional, objetivo de vol, topes, banda
  -> informe                 cartera, o NO ACCIONABLE con el motivo
  -> data/venues.py          union de plazas con ejecutabilidad y coste
  -> execution/router.py     enruta a la plaza viable; mismo interfaz de broker
  -> cycle.py                ciclo diario: salidas -> entradas -> ordenes -> mandato
  -> analogy.py              posicion abierta -> distribucion de casos analogos
  -> mandate.py              registro de meta/plazo/drawdown (observa, no dirige)
```

---

## 6. Como saber si sigue siendo fiable

Despues de **cualquier** cambio en features, agentes o modelos:

```bash
python3 scripts/null_test.py        # el gate DSR debe rechazar al campeon del ruido
python3 scripts/swarm_null_test.py  # especificidad Y potencia del enjambre
python3 -m pytest -q                            # 69 tests rapidos
SWARMTRADER_DEEP=1 python3 -m pytest -q         # + 3 tests de la capa profunda
SWARMTRADER_DEEP=1 python3 scripts/swarm_null_test.py
python3 experiments/survivorship.py             # cuantifica el sesgo en TUS metricas
```

Si `swarm_null_test` certifica sobre ruido, hay una fuga y **cualquier
resultado positivo posterior carece de significado**. Si deja de detectar el
momentum inyectado, el sistema se ha vuelto ciego. Las dos mitades importan.
