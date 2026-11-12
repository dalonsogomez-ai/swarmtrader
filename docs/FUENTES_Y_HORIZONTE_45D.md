# Fuentes de datos y horizonte de 0 a 45 días

Fecha: 2026-11-12. Versión del código de referencia: 1.4.0.

Pregunta: ¿añadir plataformas o fuentes de datos, aceptando que el análisis tarde
más, permite a `swarmtrader` encontrar más acciones con una subida fuerte que se
materialice en un máximo de 45 días (el mismo día, una semana o hasta 45 días)?

Cada afirmación lleva su grado: **[fuente]** (comprobado en esta fecha contra la
referencia citada), **[memoria]** (resultado conocido de la literatura, no
reverificado en esta sesión), **[inferencia]** (razonamiento propio) o **[no
medido]**. Nada de esto es asesoramiento de inversión ni promete rentabilidad.

## Resumen

Sí, pero solo para un tipo concreto de fuente y solo en la parte de la ventana que
va de unos días a 45 días. No para el intradía.

- **Lo que suma** son fuentes de **eventos de empresa** que el precio y el volumen
  no contienen: compras de directivos (Form 4), resultados y su reacción, y
  noticias agregadas por semana. Las tres son gratuitas o ya están incluidas en
  Alpaca, y alimentan agentes que hoy tienen código pero no datos (`flow`,
  `earnings`, `news`).
- **Lo que no suma** es añadir más brókers o más fuentes de precio. eToro e IBKR
  amplían la lista de acciones, no la información sobre cada una.
- **El tiempo de cómputo no es el problema.** El sistema decide una vez al día
  tras el cierre y ejecuta en la apertura siguiente. Con ese retraso, una señal
  que dura semanas se conserva casi entera y una que dura minutos se pierde
  entera, tarde el análisis diez segundos o dos horas.
- **El intradía queda fuera** mientras solo haya barras diarias y un ciclo al día.
  Es un sistema distinto, no una fuente más.
- **Ninguna fuente nueva está validada.** Todo lo medido hasta hoy es sobre 1.000
  acciones supervivientes. La primera compra útil sigue siendo un histórico con
  empresas deslistadas; sin él, cualquier fuente nueva se validaría sobre una
  muestra sesgada.

## 1. Veredicto y razonamiento

### 1.1 Qué mide el sistema hoy

El enjambre tiene un rank IC de 0,075 (t = 4,6) sobre datos reales, pero
concentrado en la mitad baja del ranking: distingue a las peores y no a las
mejores. Sus 5 primeras no baten al universo (`docs/AUDITORIA_GEOMETRIA.md`). Los
tres agentes que llevan esa señal (`lowrisk`, `momentum`, `liquidity`) usan solo
precio y volumen. Los cinco que usarían otra información (`earnings`, `analyst`,
`flow`, `news`, `political`, en `agents/specialists.py`) quedan inactivos porque
sus columnas solo entran por CSV (`data/signals.py`) y no hay ningún CSV.

Por tanto la pregunta correcta no es «¿más fuentes?» sino «¿hay información sobre
la parte alta del ranking que el precio no tenga?».

### 1.2 Cuándo una fuente nueva mejora: la condición exacta

La Ley Fundamental de la Gestión Activa (Grinold 1989; Grinold y Kahn) dice
**[memoria]**:

    IR ≈ TC · IC · √BR

donde IC es la correlación entre predicción y resultado, BR el número de apuestas
**independientes** al año y TC el coeficiente de transferencia (Clarke, de Silva y
Thorley 2002), que mide cuánto de la señal llega a la cartera.

Tres consecuencias para este sistema **[inferencia]**:

1. **Añadir acciones no añade amplitud si la señal es la misma.** Mil acciones
   puntuadas con los mismos factores de precio son apuestas muy correlacionadas.
   Más plataformas de ejecución suben el recuento, no el BR efectivo.
2. **Una cartera solo-largo de 1 a 5 nombres tiene un TC bajo.** Solo monetiza la
   cola alta del ranking, que es justo donde la señal actual es nula. El modo
   convicción lo acentúa: con un nombre, el resultado depende de una sola
   predicción.
3. **Una fuente nueva suma solo por su parte no redundante.** Para dos señales con
   IC₁ e IC₂ y correlación ρ entre ellas, el IC de la combinación óptima es

       IC² = (IC₁² + IC₂² − 2ρ·IC₁·IC₂) / (1 − ρ²)

   La mejora es cero cuando IC₂ = ρ·IC₁. Ejemplo: con IC₁ = 0,075, una señal
   nueva de IC₂ = 0,03 e independiente (ρ = 0) sube el IC a 0,081. Otra de
   IC₂ = 0,05 pero correlacionada 0,7 con la actual lo deja en 0,075: no aporta
   nada aunque por separado parezca mejor. En términos de información, lo que
   cuenta es I(Y; X_nueva | X_actual), no I(Y; X_nueva).

De ahí la condición: **una fuente merece entrar si su IC residual (tras quitarle
lo que ya explican los agentes actuales) es distinto de cero fuera de muestra y,
además, mejora la parte alta del ranking**, que es lo que contrasta el filtro G4.

### 1.3 Cuándo empeora

Cada fuente añadida es un contraste más. McLean y Pontiff (2016) midieron 97
predictores publicados: rinden un 26 % menos fuera de muestra y un 58 % menos tras
publicarse **[fuente]**. Harvey, Liu y Zhu (2016) proponen exigir t > 3 a un factor
nuevo por este motivo **[memoria]**. En el sistema, cada agente nuevo entra en el
contraste SPA de G1 y en el PBO de G3: diez agentes sin señal hacen más difícil
que pase el que sí la tiene.

El tamaño de muestra también limita. Una señal de eventos con IC de 0,05 necesita
del orden de (3 / 0,05)² = 3.600 eventos independientes para alcanzar t = 3
**[inferencia]**. Las compras agrupadas de directivos o los 13D son unos pocos
cientos al año: se tarda años de histórico en validarlas, y con 18 fechas
mensuales fuera de muestra no se puede.

### 1.4 Latencia frente a duración de la señal

Si la ventaja de una señal decae con vida media h y el sistema actúa con retraso
d, conserva una fracción 2^(−d/h) **[inferencia]**. El retraso del sistema es de
aproximadamente una sesión: dato hasta el cierre, decisión por la noche, orden en
la apertura. El tiempo de cómputo cabe dentro de esa noche.

| Clase de señal | Vida media orientativa | Fracción conservada con 1 sesión de retraso |
|---|---|---|
| Microestructura, flujo de órdenes | minutos | ≈ 0 |
| Titular de noticia aislado | 1 a 2 días | 50 a 70 % |
| Sentimiento de noticias agregado por semana | semanas | > 90 % |
| Deriva tras resultados | semanas | > 90 % |
| Compras de directivos | meses | ≈ 100 % |
| Valor sin catalizador | trimestres o años | ≈ 100 %, pero no madura en 45 días |

Las duraciones de noticias salen de Heston y Sinha (2017): la noticia diaria
predice uno o dos días; el sentimiento semanal, hasta un trimestre **[fuente]**.

Conclusión: **tardar más en analizar no cuesta nada para las señales que
interesan a este sistema, y no rescata las que ya se pierden por el ciclo
diario.**

### 1.5 La paradoja del valor profundo

Una acción barata por múltiplos y sin catalizador puede tardar años en corregirse;
con un techo de 45 días es capital inmovilizado **[memoria]**. Lo que el usuario
llama joya oculta, dentro de 45 días, es la intersección de dos cosas: una
asimetría (infravaloración, poca cobertura, posición extrema) y **un evento
fechado** que obligue al mercado a mirar. La fuente útil es la que fecha el
evento, no la que mide lo barata que está.

Qué exige cada horizonte:

| Horizonte | Qué tendría que saber el sistema | Dato necesario | ¿Lo tiene? |
|---|---|---|---|
| Mismo día | Desequilibrio de órdenes, noticia en el minuto | Barras intradía, cotizaciones, noticias en tiempo real, ejecución intradía | No. Solo barras diarias y un ciclo al día |
| 2 a 7 días | Reacción inicial a un evento ya publicado | Fecha del evento, reacción del precio en 1 a 3 días, tono de la noticia | Código sí (`earn_react_3d`, `news_score`); datos no |
| 8 a 45 días | Deriva tras el evento, compras de directivos, revisiones | Sorpresa de resultados, Form 4, consenso | Código sí (`sue`, `insider_net`, `rev_chg_21`); datos no |

## 2. Mapa de fuentes

### 2.1 Tabla

Precios en USD al mes. «Tiempo» es el impacto sobre el ciclo diario para un
universo de varios miles de acciones.

| Fuente | Dato que aporta | Horizonte donde tiene evidencia | Tiempo | Coste | Veredicto |
|---|---|---|---|---|---|
| Histórico con deslistadas (Sharadar SEP o equivalente) | Quita el sesgo de supervivencia | Todos | Carga única | 9 a 39 (sin confirmar) | **Primero.** Sin esto nada se valida |
| SEC EDGAR, Form 4 | Compras de directivos, con fecha de presentación | Semanas a meses | Minutos al día | 0 | **Integrar.** Alimenta `flow` |
| Alpaca News (Benzinga) | Titulares por símbolo desde 2015 | 1 a 2 días por titular; semanas si se agrega | Minutos al día | 0 con el plan actual (sin confirmar) | **Integrar agregado por semana.** Alimenta `news` |
| Fechas y sorpresa de resultados (EDGAR 8-K para la fecha; FMP o Finnhub para el consenso) | Evento fechado y reacción | Días a semanas, sobre todo microcaps | Minutos al día | 0 a 20 | **Integrar la fecha y la reacción;** la sorpresa frente a consenso, después |
| FINRA, interés corto quincenal | Posición corta abierta | Semanas; señal **negativa** | Segundos | 0 | **Integrar solo como veto.** Alimenta `flow` |
| FINRA, volumen corto diario | Ventas en corto del día | Días; evidencia débil | Segundos | 0 | Opcional, baja prioridad |
| SEC EDGAR, 13D | Entrada de un fondo activista | ±20 días alrededor del anuncio | Minutos al día | 0 | Posponer: pocos eventos, sube antes de publicarse |
| Revisiones de analistas (FMP, Finnhub) | Cambios de consenso | Semanas | Minutos | 20 a 100 | Posponer hasta validar las gratuitas |
| Alpaca Algo Trader Plus | SIP en tiempo real, 10.000 llamadas/min | Ninguno nuevo para un ciclo diario | Descarga más rápida | 99 | No hace falta |
| Massive (antes Polygon) | Precios, agregados, opciones | Ninguno nuevo | — | 29 a 199 por clase de activo | Descartar: duplica a Alpaca |
| Flujo de opciones (Unusual Whales, ORATS, Tradier) | Volumen y posicionamiento en opciones | 1 día a 1 semana | Alto | 150 a 375 | Descartar por ahora |
| Exposición gamma (SpotGamma) | Cobertura de creadores de mercado | Intradía | Alto | De pago | Descartar: no hay intradía |
| Impresiones de dark pools | Bloques fuera de mercado | Sin evidencia sólida | — | De pago | Descartar |
| Declaraciones del Congreso (Quiver) | Compras de congresistas | Ninguno tras 2012 | Bajo | ~30 | Descartar como señal; el cargador ya existe |
| Reddit, StockTwits, X | Atención minorista | Predice volumen y volatilidad | Medio | Variable | Descartar como señal de compra |
| eToro, IBKR como universo | Más símbolos | Ninguno | Alto | Costes de ejecución altos | No amplía información |

### 2.2 Las tres que primero merecen integrarse

**Form 4 desde EDGAR.** Es gratuito, tiene fecha legal clara y horizonte dentro de
45 días. El plazo de presentación es de dos días hábiles **[fuente]**. EDGAR
permite 10 peticiones por segundo y exige declarar un `User-Agent` **[fuente]**.
Los ficheros trimestrales de la SEC sirven para construir el histórico; la señal
diaria sale del índice diario de presentaciones **[fuente]**. La evidencia:
las compras predicen mejor que las ventas y sobre todo en empresas pequeñas
(Lakonishok y Lee 2001) **[fuente secundaria]**; separar a los directivos que
operan siempre en el mismo mes («rutinarios», en torno al 55 % de las
operaciones) deja en los demás un exceso de 82 puntos básicos al mes, y en los
rutinarios cero (Cohen, Malloy y Pomorski 2012) **[fuente]**. Encaja con el
universo que el usuario quiere, de penny a mid cap. Columna de destino:
`insider_net` en `data/signals.py::insider_features`. Faltaría el filtro
rutinario/oportunista, que hoy no existe.

**Resultados: fecha y reacción.** La deriva tras resultados está discutida.
Martineau (2022) la da por desaparecida en las grandes desde 2006 y recientemente
también en microcaps; Subrahmanyam (2025) encuentra t = 2,18 con todas las
acciones y t = 1,43 sin microcaps **[fuente]**. Es decir, lo que queda vive en las
empresas más pequeñas e ilíquidas, donde el coste de operar es mayor. Aun así es
el evento fechado más frecuente (cuatro por empresa y año) y el agente `earnings`
ya calcula `earn_react_3d` y `days_since_earn`, que por construcción solo
dependen de la fecha y del precio. La fecha sale de los 8-K (plazo de cuatro días
hábiles **[fuente]**). La sorpresa frente al consenso (`sue`) exige un proveedor
de estimaciones y puede esperar. Hoy `earnings_features` rechaza un CSV sin
sorpresa ni BPA: admitir uno solo con fechas es un cambio pequeño de código que
este documento no hace.

**Noticias de Alpaca, agregadas.** Alpaca sirve noticias de Benzinga desde 2015,
unas 130 al día, por REST y WebSocket **[fuente]**. No requiere otra plataforma.
La evidencia dice cómo usarlas: un titular suelto predice uno o dos días, que el
ciclo diario captura a medias; el sentimiento agregado por semana predice hasta
un trimestre, y las malas noticias tardan más en incorporarse que las buenas
(Heston y Sinha 2017) **[fuente]**. Los modelos de lenguaje mejoran la
clasificación del titular, con una correlación pequeña y un acierto por titular
cercano al 51 % (Lopez-Lira y Tang) **[fuente]**. Uso recomendado: `news_score`
como media semanal y `news_neg` como veto, no como disparador de compra en el día.

### 2.3 Fuentes que solo deben vetar

El interés corto alto predice retornos **peores**, no mejores. Engelberg, Reed y
Ringgenberg (2018) miden +0,58 % al mes en el quintil de menor riesgo de préstamo
y −0,49 % en el de mayor, con más fuerza en micro y pequeñas **[fuente]**. Buscar
un *short squeeze* es apostar contra la media. FINRA publica el interés corto dos
veces al mes, el séptimo día hábil tras la fecha de liquidación **[fuente]**; hay
una propuesta para hacerlo semanal (SR-FINRA-2026-012) cuyo resultado no se ha
podido confirmar. Con ese retraso el dato sirve para excluir, no para anticipar.
Esto coincide con la regla del proyecto: los filtros nuevos solo pueden vetar.

### 2.4 Por qué se descartan las de opciones

La evidencia existe pero no es explotable aquí. Pan y Poteshman (2006) miden más
de 40 puntos básicos al día siguiente y más del 1 % en una semana, con datos de
1990 a 2001 y con el volumen de apertura comprador, que no es público
**[fuente]**. Johnson y So (2012) encuentran 0,34 % por semana, y la relación es
negativa: más volumen de opciones respecto a acciones anuncia peores retornos
**[fuente]**. Son señales de uno a cinco días, sirven más para evitar que para
comprar, cuestan entre 150 y 375 USD al mes en los servicios comerciales
**[fuente parcial: precios de 2025]** y muchas de las acciones pequeñas del
universo no tienen opciones líquidas. Los servicios de «flujo inusual» no tienen
valor demostrado en literatura publicada **[no medido]**.

### 2.5 Qué cubre Alpaca sin añadir nada

Con el plan gratuito: barras diarias e intradía con IEX en tiempo real y SIP con
15 minutos de retraso, 200 llamadas por minuto, y SIP histórico de más de 15
minutos **[fuente]**; noticias de Benzinga desde 2015 **[fuente]**; órdenes
límite fraccionarias en pre y post mercado **[fuente]**; negociación de domingo
20:00 a viernes 20:00 ET con liquidez nocturna baja **[fuente secundaria]**. El
plan de 99 USD añade SIP en tiempo real y 10.000 llamadas por minuto: acelera la
descarga pero no añade información para un ciclo diario. La documentación menciona
además un plan intermedio de 1.000 llamadas por minuto; esa cifra está en
conflicto con las otras y hay que comprobarla antes de depender de ella.

## 3. Arquitectura en dos velocidades

La separación ya existe en parte y conviene hacerla explícita en vez de añadir
piezas.

**Capa lenta (descubrimiento, por la noche).** Hoy es `swarm_rank`: características,
agentes, combinación, filtros G0 a G4. Las fuentes nuevas entran aquí como una
ingesta previa que escribe los CSV que `data/signals.py::build_extra` ya sabe
leer, con su retraso `available_lag` de una sesión. No hace falta tocar el
orquestador. Presupuesto de tiempo: toda la noche. Con el límite de EDGAR, leer
el índice diario y los Form 4 del día son segundos; el histórico es una carga
única.

**Capa rápida (ejecución, en la apertura).** Hoy es `cycle.py` más
`execution/base.py::rebalance_to`: lee el estado real del bróker, aplica la
triple barrera a lo abierto y envía ventas, espera y compras. No depende de
ninguna fuente lenta y no debe depender.

Dos reglas para que la capa lenta no dañe a la rápida **[inferencia]**:

- La ingesta falla en abierto: si una fuente no responde, sus columnas quedan
  neutras (ya ocurre: símbolos sin dato valen cero tras normalizar) y el ciclo
  sigue con las demás.
- Cada dato lleva la fecha en que fue público, no la del hecho. `signals.py` ya
  lo exige en su contrato y `data/congress.py` falla si recibe la fecha de la
  operación en lugar de la de divulgación.

El intradía sería una tercera capa con otra fuente de datos (barras de minutos,
noticias por WebSocket), otra validación y un proceso vivo durante la sesión. El
propio `cycle.py` explica por qué no es un demonio. No se recomienda abrirla
hasta que la capa diaria demuestre ventaja.

## 4. Modelado de la ventana de 0 a 45 días

### 4.1 Lo que ya existe

`data/labels.py` implementa la triple barrera de López de Prado (2018, cap. 3):
objetivo (+10 %), stop fijo (−8 %), stop móvil desde el máximo y barrera vertical
a 32 sesiones, que son unos 45 días naturales. Devuelve el retorno, las sesiones
hasta la salida (`exit_h`) y el motivo. `cycle.py` aplica las mismas barreras en
vivo y `analogy.py` puede revocar una venta por objetivo.

### 4.2 Tocar una barrera es sobre todo volatilidad

Para un precio con deriva logarítmica diaria μ y volatilidad diaria σ, la
probabilidad de tocar una barrera superior a distancia a = ln(1 + objetivo) antes
de T sesiones es **[memoria]**:

    P(τ ≤ T) = Φ((μT − a) / (σ√T)) + exp(2μa / σ²) · Φ((−a − μT) / (σ√T))

Con μ = 0, σ = 3 % diario (una acción pequeña), T = 32 y objetivo +10 %, da
aproximadamente un 57 %. Es decir, **más de la mitad de las acciones volátiles
tocan +10 % en 45 días sin ninguna ventaja**, y una proporción parecida toca el
stop. Con dos barreras y sin límite de tiempo, la probabilidad de tocar antes la
superior con μ = 0 es b / (a + b), donde b es la distancia al stop: con +10 % y
−8 %, un 47 %.

Dos consecuencias **[inferencia]**:

- Una acción que «sube un 10 % en dos semanas» no prueba nada sobre la selección.
  La etiqueta útil es el exceso sobre lo que la volatilidad sola predice.
- El tiempo esperado hasta el objetivo, cuando hay deriva positiva, es a / μ. Con
  las derivas que la evidencia permite (del orden del 1 % al mes) ese tiempo es de
  muchos meses, no de días: dentro de 45 días la mayoría de las salidas por
  objetivo las produce la volatilidad.

### 4.3 Estimar cuándo madura

La forma de pasar de «qué» a «qué y cuándo» con barras diarias, sin datos nuevos,
es tratar la salida como riesgos en competencia en tiempo discreto: para cada
sesión k de 1 a 32, la probabilidad de salir por objetivo, por stop o seguir,
condicionada a las características del día de entrada. Las etiquetas ya están
(`exit_h` y `exit_reason`). De ahí salen la probabilidad acumulada de objetivo a
5, 10, 21 y 32 sesiones y una duración esperada E[τ]. Es una propuesta, no está
implementada **[no medido]**. Una alternativa más barata es entrenar el mismo
modelo de ranking con el retorno a 5, 10, 21 y 32 sesiones y mirar a qué horizonte
tiene IC cada agente; `RankConfig.horizon` ya es un parámetro.

### 4.4 Conexión con el asignador de convicción

`portfolio/conviction.py` maximiza el crecimiento esperado sobre escenarios a un
único horizonte. Para comparar una jugada de una semana con otra de 45 días hay
que comparar **crecimiento por unidad de tiempo**: g / E[τ], con g el crecimiento
logarítmico esperado de la posición. Una posición de +3 % esperado en 5 sesiones
vale más que una de +6 % en 32 si el capital liberado puede volver a colocarse.
Ese «si» es la condición: solo vale cuando hay otra oportunidad esperando y el
coste de rotar (el spread, que en penny stocks puede ser de varios puntos) no se
come la diferencia **[inferencia]**.

Sobre los stops: Kaminski y Lo (2014) muestran que con un paseo aleatorio un stop
reduce el retorno esperado, y que solo añade valor cuando hay momentum o cambios
de régimen **[fuente]**. El stop del sistema es una regla de control de pérdidas
exigida por el proyecto, no una fuente de rentabilidad, y así debe presentarse.

## 5. Plan de integración por fases

Los criterios se fijan aquí, antes de medir. Ningún valor por defecto cambia con
resultados exploratorios.

| Fase | Qué se hace | Criterio para pasar, fijado de antemano |
|---|---|---|
| 0 | Histórico con deslistadas; repetir la auditoría | Las hipótesis ya registradas en `ESTADO.json` |
| 1 | Ingesta de Form 4 desde EDGAR a CSV; activar `flow` | IC residual de `insider_net` con t ≥ 3 fuera de muestra y exceso de los 5 primeros (G4) no peor que sin el agente |
| 2 | Fechas de resultados y noticias semanales de Alpaca; activar `earnings` y `news` | El mismo, agente por agente; `news_neg` se evalúa solo como veto |
| 3 | Interés corto de FINRA como veto | Excluir el decil más corto no empeora el exceso de los 5 primeros |
| 4 | Modelo de salida por riesgos en competencia | Mejor calibración de la probabilidad de objetivo que la fórmula de volatilidad de §4.2 |

Si una fuente no pasa, se retira: no se deja activa «por si acaso», porque
penaliza a las demás en G1 y G3.

## 6. Limitaciones

- Ninguna de las fuentes recomendadas se ha medido en este sistema. La
  recomendación se apoya en literatura y en que alimentan código ya escrito.
- Los tamaños de efecto citados son brutos, de muestras que terminan hace años y
  en su mayoría antes de costes. La evidencia de decaimiento tras la publicación
  es general.
- Varios precios proceden de agregadores y no de la página oficial (Massive, FMP,
  Finnhub, Tiingo, Quiver, Unusual Whales, Sharadar). Hay que confirmarlos antes
  de contratar.
- No se ha confirmado si las noticias de Alpaca siguen siendo gratuitas, ni el
  resultado de la propuesta de interés corto semanal de FINRA.
- Las fórmulas de la sección 4 suponen deriva y volatilidad constantes y precios
  continuos; con huecos de apertura la probabilidad real de tocar una barrera es
  distinta.
- Los resultados marcados **[memoria]** no se han reverificado en esta sesión.

## Referencias

- Alpaca, About Market Data API: https://docs.alpaca.markets/docs/about-market-data-api
- Alpaca, Historical News Data: https://docs.alpaca.markets/docs/historical-news-data
- Alpaca, fracciones con órdenes límite y horario extendido: https://alpaca.markets/blog/fractional-shares-trading-supports-limit-orders-and-extended-hours/
- Alpaca, 24/5 Trading: https://docs.alpaca.markets/docs/245-trading-for-trading-api
- SEC, Accessing EDGAR Data: https://www.sec.gov/os/accessing-edgar-data
- SEC, Insider Transactions Data Sets: https://www.sec.gov/data-research/sec-markets-data/insider-transactions-data-sets
- Plazos 13D/13G tras la reforma de 2024: https://www.foley.com/en/insights/publications/2023/10/sec-adopts-changes-schedule-13d-schedule-13g
- FINRA, Short Interest Reporting: https://www.finra.org/filing-reporting/short-interest
- FINRA, Daily Short Sale Volume Files: https://www.finra.org/finra-data/short-sale-volume-data/daily-short-sale-volume-files
- FINRA, SR-FINRA-2026-012: https://www.finra.org/rules-guidance/rule-filings/sr-finra-2026-012
- Martineau (2022), Rest in Peace Post-Earnings Announcement Drift: https://ideas.repec.org/a/now/jnlcfr/104.00000122.html
- Subrahmanyam (2025), resumen de UCLA Anderson: https://anderson-review.ucla.edu/is-post-earnings-announcement-drift-a-thing-again
- Cohen, Malloy y Pomorski (2012), Decoding Inside Information: https://www.nber.org/papers/w16454
- Lakonishok y Lee (2001), Are Insider Trades Informative?: https://ideas.repec.org/a/oup/rfinst/v14y2001i1p79-111.html
- Heston y Sinha (2017), News vs. Sentiment: https://rpc.cfainstitute.org/research/financial-analysts-journal/2017/news-vs-sentiment-predicting-stock-returns-from-news-stories
- Lopez-Lira y Tang, Can ChatGPT Forecast Stock Price Movements?: https://arxiv.org/abs/2304.07619
- Engelberg, Reed y Ringgenberg (2018), Short-Selling Risk: https://ideas.repec.org/a/bla/jfinan/v73y2018i2p755-786.html
- Johnson y So (2012), The option to stock volume ratio and future returns: https://econpapers.repec.org/RePEc:eee:jfinec:v:106:y:2012:i:2:p:262-286
- Pan y Poteshman (2006), The Information in Option Volume for Future Stock Prices: https://www.nber.org/papers/w10925
- Brav, Jiang, Partnoy y Thomas (2008), Hedge Fund Activism: https://corpgov.law.harvard.edu/wp-content/uploads/2008/03/hedge_fund_activism-final.pdf
- Kaminski y Lo (2014), When Do Stop-Loss Rules Stop Losses?: https://ideas.repec.org/a/eee/finmar/v18y2014icp234-254.html
- McLean y Pontiff (2016), Does Academic Research Destroy Stock Return Predictability?: https://ivey.uwo.ca/media/3775549/pontiff.pdf
- Chen y Sacerdote (2026), Capital in the Capitol: https://www.nber.org/papers/w35041
- Belmont, Sacerdote y Van Hoek, Relief Rally: https://www.cato.org/research-briefs-economic-policy/relief-rally-senators-feckless-rest-us-stock-picking
