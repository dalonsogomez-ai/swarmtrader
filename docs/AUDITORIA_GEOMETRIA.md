# Auditoría de swarmtrader y de todas las vías geométricas

Versión del sistema: 1.3.0 · Fecha: 2026-11-11 · Resultados completos en
`docs/results/geometry_audit_2026-11-11.json` (pre-registrado) y
`docs/results/audit_followup_2026-11-11.json` (exploratorio).

## Cómo leer este documento

Cada afirmación lleva una etiqueta que dice de dónde sale:

| Etiqueta | Significado |
|---|---|
| **[MEDIDO]** | Sale de `experiments/geometry_audit.py`, cuyos criterios se fijaron en la cabecera del script antes de ejecutarlo |
| **[EXPLORATORIO]** | Sale de `experiments/audit_followup.py`, escrito después de ver los primeros resultados. Orienta, no confirma |
| **[VERIFICADO]** | Fuente externa consultada hoy; en casi todos los casos a nivel de resumen, no del texto completo |
| **[RAZONADO]** | Deducción matemática o lectura del código, sin medición |
| **[DE MEMORIA]** | Referencia clásica no consultada en esta sesión; sin cifras |

**Limitación que afecta a todas las mediciones.** Los datos reales son 1.000
acciones elegidas al azar de un panel de 5.550 que cotizan **hoy** en EE. UU.
(Yahoo, 2018-01-02 a 2026-09-30): es un universo de supervivientes. Faltan las
empresas que quebraron o fueron excluidas. Sirve para comparar dos reglas entre
sí sobre los mismos casos. No sirve para afirmar rentabilidad, y sesga a favor
cualquier estrategia que persiga valores volátiles, porque los que acabaron en
cero no están. No hay costes de transacción en ninguna cifra. La evaluación son
85 fechas mensuales fuera de muestra con una mediana de 544 valores por fecha.

## Resumen

1. **El hallazgo más importante no es geométrico.** Sobre datos reales el
   enjambre tiene un rank IC de 0,075 (t = 4,6), pero todo ese poder predictivo
   está en la mitad baja del ranking: el sistema distingue a los peores y no
   sabe elegir a los mejores. Los 5 primeros, que son lo único que compra, no
   baten al universo (exceso de −0,15% a 21 sesiones, t = −0,3). Con la versión
   1.2 el veredicto sobre esos datos era ACCIONABLE.
2. **Cuatro defectos corregidos en la v1.3.0**: el veredicto no comprobaba lo
   que se compra (nuevo filtro G4); el adaptador de Sharadar eliminaba del
   histórico a los valores que acabaron en céntimos; el filtro de precio usaba
   precios ajustados, que incorporan splits futuros; y la puntuación se
   multiplicaba por el «acuerdo» entre agentes sin que nadie lo hubiera validado.
3. **Geometría: ninguna de las vías nuevas se gana el sitio.** Seis familias de
   contraste, ninguna significativa ni antes ni después de corregir por
   comparación múltiple. No se añade ningún modelo, ningún agente y ningún
   agregador nuevo.
4. **Modelo, agente o enjambre.** La respuesta la da el tipo de objeto que
   produce cada técnica. Una medida del estado del mercado es un número por
   fecha, igual para todas las acciones, así que no puede ordenar acciones: no
   puede ser un agente. Solo puede ser un filtro. Las medidas geométricas por
   acción sí pueden ser un agente, y medidas resultan informativas pero
   redundantes con volatilidad y momentum. Y el agregador actual del enjambre ya
   es, matemáticamente, una media de Fréchet.
5. **Lo que queda de la geometría** es lo que ya estaba en la v1.2: el filtro G0
   y las dos métricas opcionales del motor de análogos. Hay evidencia nueva
   sobre G0 y no es favorable: no detecta más episodios de estrés que una línea
   base trivial de volatilidad.

---

## 1. Auditoría y verificación del sistema actual

### 1.1 Qué es realmente cada pieza

Leído del código, no de la documentación. **[RAZONADO]**

| Pieza | Qué hace de verdad | Supuestos que hace |
|---|---|---|
| Factores | 14 factores de precio y volumen, convertidos a z-score **por fecha** y recortados a ±3 | Solo importa la posición relativa dentro del día. Borra la escala: un z = +2 es +8% un mes y +40% otro |
| Agentes de precio (momentum, reversal, lowrisk, liquidity, seasonal) | Cada uno es una regresión ridge (α = 20) sobre 1 a 4 factores, reentrenada cada mes con purga y embargo; su salida se convierte a rango en [−1, 1] | Relación lineal y estable entre factor y retorno relativo durante la ventana de 3 años |
| Agentes con datos externos (earnings, analyst, flow, news, political) | Lo mismo, sobre datos que hoy no están cargados | **Están inactivos.** En la práctica el enjambre son 5 agentes, no 11 |
| `deep_rank` | GRU con atención cross-seccional | No ejecutable en este entorno (sin PyTorch); nunca medido sobre datos reales |
| Etiqueta de entrenamiento | Rango relativo del retorno a 21 sesiones dentro de la fecha | Trata igual acertar el orden entre los peores que entre los mejores |
| Combinación | Media de los rangos (`eq`), o apilado NNLS (`stack`) si demuestra batirla | — |
| Filtros G1-G3 | SPA sobre el IC, apilado frente a media, PBO | Validan el IC de **toda** la sección cruzada |
| G0 (v1.2) | Distancia de Fisher-Rao del estado de mercado a la ventana de entrenamiento | Los estados de entrenamiento son una muestra representativa |
| Calibración y dimensionado | Isotónica y Kelly fraccional 0,25 con topes | La puntuación es monótona en la probabilidad de éxito |

### 1.2 Lo que rinden sobre datos reales

**[MEDIDO]** Rank IC a 21 sesiones, media de 85 fechas (entre paréntesis, t):

| Agente | Todo el universo | Precio ≥ 5 USD | Precio < 5 USD |
|---|---|---|---|
| lowrisk | 0,085 (4,2) | 0,051 (2,6) | 0,165 (8,1) |
| momentum | 0,069 (3,4) | 0,039 (2,0) | 0,118 (5,6) |
| liquidity | 0,055 (5,8) | 0,031 (3,5) | 0,134 (7,8) |
| reversal | 0,019 (1,7) | 0,005 (0,5) | 0,079 (5,1) |
| seasonal | 0,010 (1,0) | 0,005 (0,5) | 0,017 (1,2) |
| **enjambre (media)** | **0,075 (4,6)** | **0,044 (2,9)** | **0,139 (8,0)** |

Lecturas:

- **Tres agentes llevan la señal** (lowrisk, momentum, liquidity). Es coherente
  con Gu, Kelly y Xiu (2020), que encuentran que los predictores dominantes en
  todos los métodos son variantes de momentum, liquidez y volatilidad.
  **[VERIFICADO]**
- **`reversal` no muestra nada en valores de 5 USD o más** (0,005; t = 0,5) y
  solo aparece por debajo de 5 USD. Coincide con Avramov, Chordia y Goyal
  (2006): la reversión a corto plazo se concentra en valores ilíquidos de alta
  rotación y sus beneficios son menores que los costes de transacción.
  **[VERIFICADO]**
- **`seasonal` no muestra nada en ningún tramo.**
- **El IC se concentra en los valores baratos.** En el tramo estricto de 1 a 5
  USD nominales es 0,118; en 10 USD o más, 0,036 (t = 2,4). **[EXPLORATORIO]**
  El signo del sesgo de supervivencia aquí no se puede determinar: para lowrisk
  y momentum la ausencia de quiebras probablemente lo atenúa; para reversal
  probablemente lo infla.
- **Redundancia.** La correlación media entre las puntuaciones de momentum y
  lowrisk es 0,58: los dos agentes más fuertes son en buena parte la misma
  apuesta. El resto de pares queda por debajo de 0,30 y seasonal es ortogonal a
  todos. El número efectivo de agentes independientes es 4,1 de 5.

### 1.3 ¿Es ridge el mejor modelo base?

**[MEDIDO]** Sí, en este régimen de datos. Sobre las últimas 48 fechas:

| Modelo | Rank IC (t) |
|---|---|
| ridge (actual) | 0,101 (4,3) |
| gradient boosting | 0,065 (3,4) |
| media de los dos | 0,086 (4,1) |

H0 «ninguno bate a ridge»: p = 1,00. La literatura encuentra que árboles y redes
neuronales ganan a los modelos lineales (Gu, Kelly y Xiu, 2020), pero con cientos
de predictores y décadas de historia; aquí hay 14 factores y ventanas de 3 años.
Además, Avramov, Cheng y Metzker (2023) muestran que la rentabilidad de las
señales de aprendizaje profundo se atenúa considerablemente al excluir
microcaps y empresas en dificultades, y cae más con costes razonables.
**[VERIFICADO]**

Otras alternativas que la evidencia no respalda adoptar:

- **Modelos fundacionales de series temporales** (Chronos, TimesFM y similares).
  Rahimikia, Ni y Wang (noviembre de 2025) concluyen que los preentrenados
  genéricos rinden mal sobre retornos diarios en exceso, tanto sin ajuste como
  con ajuste fino; solo ven ganancias entrenando desde cero con datos
  financieros, algo fuera del alcance de este proyecto. **[VERIFICADO]**
- **PEAD** (el agente `earnings`). Martineau (2022) lo da por inexistente desde
  2006 en valores grandes y desaparecido recientemente en microcaps.
  **[VERIFICADO]** No es motivo para borrar el agente, que está inactivo, pero sí
  para no esperar mucho de él cuando haya datos.

### 1.4 ¿Aporta el enjambre algo frente a un único modelo?

**[EXPLORATORIO]** Con solo los agentes de precio activos, no hay evidencia de
que sí:

| Sistema | Rank IC (t) |
|---|---|
| Un solo ridge con los 14 factores | 0,082 (4,5) |
| Enjambre, apilado NNLS | 0,078 (4,4) |
| Enjambre, media | 0,075 (4,6) |

La diferencia a favor del modelo único (+0,008) no es significativa (p = 0,10),
y el enjambre tampoco lo bate (p = 1,00). El apilado no bate a la media
(p = 0,35), así que el filtro G2 elige la media, que es el resultado habitual en
combinación de pronósticos. **[DE MEMORIA]**

La estructura de agentes se justifica por dos cosas que hoy no se ejercen: el
diagnóstico por tesis económica y la integración de fuentes heterogéneas
(resultados, insiders, noticias). Mientras esas fuentes no estén cargadas, el
enjambre es una forma más complicada de hacer lo que hace un ridge.

### 1.5 El hallazgo central: el sistema sabe a quién evitar, no a quién comprar

**[EXPLORATORIO]**, pero el defecto lógico que revela no depende de la muestra.

Retorno a 21 sesiones según el puesto en el ranking del enjambre (universo de
precio ≥ 1 USD):

| Grupo | Retorno medio | P(> +5%) | P(> +25%) | P(< −15%) |
|---|---|---|---|---|
| Los 5 primeros (lo que se compra) | +0,70% | 24,2% | 1,4% | 3,5% |
| Universo | +0,86% | 32,3% | 5,2% | 11,2% |
| Los 5 últimos | −7,92% | — | — | 42,1% |

- Exceso de los 5 primeros sobre el universo: **−0,15% (t = −0,29)**.
- Exceso de los 5 últimos: **−8,77% (t = −4,59)**.
- Rank IC calculado solo dentro de la mitad alta del ranking: **−0,003**.
  Dentro de la mitad baja: **+0,080**.
- Retorno medio por decil, del peor al mejor: −0,5%, 0,8%, 0,8%, 0,9%, 1,0%,
  1,3%, 1,4%, 1,0%, 1,0%, 0,8%. Sube hasta el séptimo decil y luego baja.
- La dispersión de retornos entre los 5 primeros es del 6,5%, frente al 18,4%
  del universo.

Con precio ≥ 5 USD el patrón es idéntico (exceso de los primeros −0,07%,
t = −0,15; de los últimos −10,0%, t = −4,9).

Lo que hace el sistema es elegir valores de baja volatilidad. Eso evita
desastres (la probabilidad de caer más de un 15% baja del 11% al 3,5%) y a
cambio renuncia a las subidas: la probabilidad de subir más de un 25% en el mes
es casi cuatro veces **menor** que eligiendo al azar. Para un objetivo de
«subida notable a corto plazo», el ranking actual apunta en sentido contrario.

Que las anomalías de volatilidad concentren su rentabilidad en el lado de los
perdedores es un resultado conocido. **[DE MEMORIA]** Una cartera que solo
compra no puede monetizar el lado corto.

**El defecto.** Los filtros G1 a G3 contrastan el IC sobre toda la sección
cruzada, y ese IC puede salir enteramente de la parte que no se compra. Con la
v1.2, el veredicto sobre estos datos (universo de 5 USD o más) era
**ACCIONABLE** (SPA p = 0,000, PBO 0,00), mientras la precisión de los 5
primeros era del 24,2% frente a una tasa base del 31,4%, con el intervalo de confianza entero por debajo de la tasa base
([18,8%, 29,9%]). El sistema habría operado, a tamaño reducido, una cartera
demostrablemente peor que el azar en su propio criterio de éxito.

### 1.6 ¿Y si se entrena para buscar subidas grandes?

**[EXPLORATORIO]** Mismo ridge, mismos 14 factores, cuatro etiquetas distintas.
Lo que entregan los 5 primeros (precio ≥ 5 USD):

| Etiqueta de entrenamiento | Exceso medio (t) | P(> +25%) | P(< −15%) |
|---|---|---|---|
| Rango relativo (la actual) | +0,60% (0,9) | 4,2% | 6,8% |
| Retorno recortado | +0,55% (0,6) | 7,3% | 11,8% |
| Superar +5% | −1,71% (−1,0) | 13,7% | 29,6% |
| Pertenecer al quintil superior | **−6,20% (−3,7)** | 12,7% | 37,4% |
| Universo | — | 4,6% | 10,4% |

Entrenar el modelo para encontrar grandes subidas funciona en un sentido muy
literal: triplica la probabilidad de un +25% mensual. Lo consigue eligiendo
volatilidad, y la volatilidad viene con las dos colas: la probabilidad de un
−15% se multiplica por tres o cuatro, y el retorno medio pasa a ser claramente
negativo. Esto ocurre **en un universo de supervivientes, que está sesgado a
favor de estas etiquetas**; con las quiebras incluidas el resultado solo puede
empeorar.

Es la medición directa, sobre datos reales, de la idea de que «más
posibilidades de subida enorme» equivale a más oportunidad: lo que se compra es
varianza, y a un precio medio negativo.

Ninguna de las cuatro etiquetas produce un exceso significativamente positivo
en los 5 primeros. No hay, en estos 14 factores de precio y volumen, evidencia
de capacidad para elegir ganadores.

### 1.7 Otras comprobaciones

- **Momento de entrada.** La etiqueta entra al cierre del mismo día en que se
  calcula la señal, mientras que el ciclo real envía la orden después del
  cierre. Sospechaba que eso inflaba el IC por rebote entre precio comprador y
  vendedor. **No es así**: entrando una sesión más tarde el IC pasa de 0,0746 a
  0,0749, y en valores de menos de 5 USD cae un 4%. La regla fijada era corregir
  si caía más de un tercio. No se toca. **[MEDIDO]**
- **Multiplicador de acuerdo.** La puntuación final se multiplicaba por
  1 − dispersión entre agentes. No mejora el IC (p = 1,00) y lo empeora
  ligeramente (−0,0018; p = 0,027). Dentro del quintil superior, los valores con
  menos desacuerdo no rinden más (p = 0,36). Queda desactivado. **[MEDIDO]**
- **Coherencia del enjambre como incertidumbre.** La correlación entre la
  coherencia de los agentes en una fecha y el IC de esa fecha es 0,12
  (p = 0,13). Sin evidencia. **[MEDIDO]**

### 1.8 Defectos encontrados y corregidos en la v1.3.0

| Defecto | Consecuencia | Corrección |
|---|---|---|
| El veredicto validaba el IC global y no lo que se compra | ACCIONABLE con una cartera sin ventaja | Filtro **G4**: contraste del exceso fuera de muestra de los `top_k` primeros sobre el universo. Sobre los datos reales el veredicto pasa a NO ACCIONABLE (exceso −0,07%, p = 1,00) |
| `sharadar_market_data` descartaba los símbolos cuyo **último** precio era inferior a 1 USD | Eliminaba del histórico entero a las empresas que quebraron: reintroducía el sesgo de supervivencia en el adaptador que existe para quitarlo. Nunca se había ejecutado con datos reales y no tenía tests | El adaptador ya no filtra nada; pasar `min_price` da error con la explicación |
| El filtro `min_price` usaba el precio ajustado | Un valor que cotizaba a 0,50 USD y después hizo un split inverso aparece a 10 USD en el histórico y entra en el universo. Afecta al 1,1% de las filas elegibles, cuya mediana de retorno a 21 sesiones es **−12,3%** | `MarketData.nominal` opcional; los filtros lo usan si existe. El adaptador de Sharadar lo rellena con `closeunadj` |
| Multiplicador de acuerdo sin validar | Ligero perjuicio medido | `StackConfig.shrink_on_disagreement = False`; la columna se sigue informando |

Los adaptadores de Alpaca y yfinance siguen sin proporcionar el precio nominal:
con ellos el filtro de precio continúa sin ser point-in-time. Queda anotado como
pendiente.

### 1.9 Veredicto por componente

| Componente | ¿Es la mejor opción para este caso? |
|---|---|
| Ridge como modelo base | Sí, con la evidencia disponible |
| lowrisk, momentum, liquidity | Llevan la señal; momentum y lowrisk se solapan mucho |
| reversal | Sin evidencia en valores operables a coste razonable |
| seasonal | Sin evidencia |
| earnings, analyst, flow, news, political | No evaluables: sin datos |
| `deep_rank` | No evaluable aquí; la literatura aconseja no esperar mejora |
| Media como combinador | Sí; el apilado no la bate |
| Enjambre frente a un modelo único | Sin ventaja demostrada mientras solo haya agentes de precio |
| Etiqueta de rango relativo | **Desalineada con una cartera que solo compra.** Las alternativas probadas son peores. Arreglarlo exige datos sin sesgo de supervivencia |
| Filtros G1-G3 | Correctos pero insuficientes; ahora hay G4 |

No retiro reversal ni seasonal a mano. Quitarlos por lo que muestran en esta
muestra y volver a evaluar sobre la misma muestra sería selección a posteriori.
El apilado ya les da un peso casi nulo (0,00 a reversal; entre 0,02 y 0,06 a
seasonal) y el SPA ya los descuenta.

---

## 2. Geometría general y geometría de la información

Para cada método: qué es, dónde encajaría, qué dice la literatura, qué medí y
qué se decide.

### 2.1 Variedades estadísticas y distancia de Fisher-Rao

**Qué es.** Tratar cada distribución como un punto de una variedad y medir la
separación entre dos puntos en unidades de distinguibilidad estadística. Para la
gaussiana univariante la distancia tiene forma cerrada y es hiperbólica; con
medias iguales se reduce a √2·|ln(σ₁/σ₂)|.

**Dónde está ya (v1.2).** En el canal de dispersión de G0 y como métrica
opcional del motor de análogos.

**Qué dice la literatura reciente.** Hammond (mayo de 2026) evalúa una familia
de observables geométricos sobre SPY y DIA de 1998 a 2024, con 17 crisis y 46
métodos de comparación, y concluye que «los observables responden al estrés, no
lo anticipan». Los contrastes de Granger van sobre todo del mercado a los
observables (17 de 45) y no al revés (6 de 45). Con umbrales calibrados a la
misma tasa de falsa alarma, un simple z-score de volatilidad móvil detecta 6 de
7 crisis con 13 días de retraso mediano; el mejor observable geométrico, 5 de 7
con 24 días. **[VERIFICADO]**

**Qué medí.** **[MEDIDO]** Seis episodios de estrés fijados de antemano, 1.793
fechas desde agosto de 2019, todos los monitores al mismo α nominal total (1%):

| Monitor | Episodios detectados | Retraso en COVID | Días marcados fuera de episodios |
|---|---|---|---|
| Línea base: volatilidad de mercado a 21 sesiones | 1 de 6 | 10 sesiones | 1,5% |
| Dispersión (Fisher-Rao) sola | 1 de 6 | 12 sesiones | **0,0%** |
| Dispersión + dependencia (el G0 actual) | 1 de 6 | 13 sesiones | 2,2% |
| G0 + forma (Wasserstein) | 1 de 6 | 13 sesiones | 5,3% |
| Dependencia sola | 2 de 6 | 17 sesiones | 2,5% |
| Forma sola | 1 de 6 | no lo detecta | 3,9% |

Todos detectan COVID y casi nada más. Ninguno marca los bancos regionales de
2023, el deshacer del carry de agosto de 2024 ni los aranceles de abril de 2025.

Por la regla fijada de antemano, queda documentado que **el G0 actual no mejora
la detección de una línea base trivial de volatilidad**: detecta el mismo
episodio, tres sesiones más tarde y con más días marcados fuera. Es el mismo
patrón que encuentra Hammond con otra familia de observables.

Dentro de ese resultado hay un matiz a favor del canal de Fisher-Rao: la
dispersión sola detecta lo mismo que la línea base sin marcar ningún día fuera
de los episodios. Con un único episodio detectado no es base para cambiar
nada.

**¿Predice la distancia cuándo falla el modelo?** **[EXPLORATORIO]** Era la
hipótesis de valor pendiente de la v1.2. En forma continua, sobre 85 fechas:

- Correlación entre la distancia y el IC del enjambre en esa fecha: −0,19 para
  dispersión (p = 0,08) y −0,16 para dependencia (p = 0,15).
- En la quinta parte de fechas más alejadas, el IC medio es 0,067 con desviación
  0,186; en el resto, 0,079 con desviación 0,129.
- Por agente, 8 de 10 correlaciones son negativas. Las más claras: reversal con
  dispersión (−0,31; p = 0,003) y momentum con dispersión (−0,27; p = 0,014).
  Son 12 pruebas sin corregir.

La dirección apoya la premisa del filtro: lejos del soporte de entrenamiento el
IC baja algo y, sobre todo, se vuelve más errático. Pero sigue siendo positivo.
El cociente media/varianza del IC, que es lo que usaría un dimensionado tipo
Kelly, queda en torno a 0,4 del valor normal. Eso sugiere que el uso adecuado de
la distancia sería **reducir tamaño**, no vetar. Es una hipótesis para contrastar
con datos point-in-time, no un cambio que haga ahora.

**Decisión.** G0 se queda como estaba. No lo paso a `warn` ni lo reduzco a un
canal: cambiar un valor por defecto con evidencia exploratoria y un solo
episodio detectado sería hacer justo lo que el sistema existe para impedir.

### 2.2 Divergencias y transporte óptimo (Wasserstein)

**Qué es.** Medir la distancia entre dos distribuciones como el coste mínimo de
transportar una sobre otra. En una dimensión se reduce a comparar funciones
cuantil, y el baricentro de Wasserstein-2 es la media de las funciones cuantil.
**[RAZONADO]**

**Dónde encajaría.** Como tercer canal de G0. El canal de dispersión resume cada
día en mediana y MAD, que por diseño ignoran las colas; un canal de forma vería
los días en que unos pocos valores hacen movimientos enormes.

**Literatura.** Horvath, Issa y Muguruza (2021) agrupan regímenes de mercado con
k-means sobre distancias de Wasserstein y lo validan en datos sintéticos y
reales con métricas de agrupamiento; no afirman rentabilidad fuera de muestra.
**[VERIFICADO]**

**Qué medí.** **[MEDIDO]**

- En sintético funciona como se esperaba: si un 3% de los valores salta un +80%,
  el canal de forma lo detecta en el 100% de los casos y el G0 actual en el 0%.
- En datos reales, la forma sola es el único monitor que detecta el
  estrangulamiento de cortos de enero de 2021, a las 5 sesiones.
- Pero añadida a G0 con el α repartido en tres, deja de detectarlo, no añade
  ningún episodio y sube los días marcados fuera de episodios del 2,2% al 5,3%.

**Decisión.** No se añade: incumple la regla fijada (detectar algo que G0 no
detecta sin sumar más de un punto de días marcados).

### 2.3 Gradiente natural

**Qué es.** Actualizar los parámetros de un modelo respetando la métrica de
Fisher del espacio de distribuciones en lugar de la euclídea.

**Dónde encajaría.** Solo en `deep_rank`. El resto de modelos no lo admiten: el
ridge tiene solución cerrada y el apilado NNLS es un problema convexo pequeño.
**[RAZONADO]**

**Literatura.** Kunstner, Balles y Hennig (2019) muestran que el Fisher
empírico, que es la aproximación barata habitual, no captura en general
información de segundo orden y puede perjudicar la optimización incluso en
problemas simples. **[VERIFICADO]**

**Decisión.** No implementado. No se puede ejecutar ni probar aquí, afecta a la
velocidad de convergencia y no a la información disponible, y el cuello de
botella del sistema no es la optimización.

### 2.4 Matrices definidas positivas y geometría de Riemann

**Qué es.** Las matrices de covarianza y correlación viven en un cono curvo.
La métrica afín-invariante mide la distancia entre dos de ellas de forma que no
cambia si las variables se reescalan o se mezclan.

**Dónde está ya (v1.2).** En el canal de dependencia de G0.

**Literatura.** Bucci, Palma y Zhang (2024) extienden el modelo HAR a matrices
de covarianza realizada de 50 valores del S&P 500 respetando esa geometría y
reportan mejor precisión, sin cuantificar la mejora en el resumen. Camara, Gaba
y Mabanza Nsiloulou (agosto de 2026) proponen un indicador de turbulencia
basado en geodésicas log-euclídeas y reportan ellos mismos que el contraste de
Diebold-Mariano no rechaza la igualdad de precisión, que el indicador va por
detrás de la volatilidad realizada y que la ganancia en backtest es modesta.
**[VERIFICADO]**

**Qué medí.** **[MEDIDO]** El canal de dependencia es el único que detectó dos
episodios (COVID y, con 54 sesiones de retraso, 2022), y también el que más días
marca fuera de episodios (2,5%).

**¿Y para estimar riesgo en el dimensionado?** **[RAZONADO]** No aporta. El
dimensionado usa volatilidades individuales y topes, sin matriz de covarianza.
Con 3 a 5 posiciones, estimar y promediar covarianzas en la variedad no cambia
nada que los topes por posición y por sector no hagan ya.

**Decisión.** Sin cambios.

### 2.5 Curvatura de Ricci en redes de mercado

**Qué es.** Construir una red de correlaciones entre acciones y medir su
curvatura; la tesis es que la curvatura media sube cuando el mercado se vuelve
frágil.

**Literatura.** Sandhu, Georgiou y Tannenbaum (2016, *Science Advances*) lo
muestran sobre 388 valores del S&P 500 de 1998 a 2013. La evaluación es
descriptiva y dentro de muestra; no hay contraste predictivo fuera de muestra y
no se compara con líneas base simples como la correlación media o la
volatilidad. Samal y coautores comparan cuatro nociones de curvatura en el S&P
500 y el Nikkei 225 de 1985 a 2016 y encuentran que la mejor es la de
Forman-Ricci, que es también la que más correlaciona con los indicadores
tradicionales. **[VERIFICADO]**

**Qué medí.** **[MEDIDO]** Curvatura de Ollivier-Ricci media sobre 100 valores
líquidos, 436 ventanas:

| Correlación de Spearman entre | Valor |
|---|---|
| Curvatura y correlación media | 0,61 |
| Curvatura y número de aristas de la red | 0,78 |
| Curvatura y volatilidad | 0,58 |
| Correlación media y volatilidad | 0,90 |

El umbral fijado para tratarla como una reformulación de la correlación media
era 0,8 y **no se alcanza**: la curvatura contiene algo distinto. Está muy
ligada a la densidad de la red. No contrasté si ese algo distinto predice nada.

**Decisión.** No se implementa. Es un número por fecha, así que solo podría ser
otro canal de G0, y los resultados de 2.1 y 2.2 muestran que añadir canales sube
los días marcados sin mejorar la detección. Además su cálculo exige resolver un
problema de transporte por cada arista.

### 2.6 Análisis topológico de datos

**Qué es.** Tomar una ventana de retornos de varios índices como una nube de
puntos y medir la persistencia de sus «agujeros»; la tesis es que la norma del
paisaje de persistencia crece antes de los desplomes.

**Literatura.** Gidea y Katz lo muestran para cuatro índices de EE. UU. antes de
los desplomes de 2000 y 2008: las normas crecen antes del máximo y la densidad
espectral de baja frecuencia sube durante unas 250 sesiones previas. No reportan
contrastes formales de significación. **[VERIFICADO]**

**Una simplificación útil.** **[RAZONADO]** La norma L¹ del paisaje completo es
exactamente Σ pᵢ²/4, donde pᵢ es la vida de cada rasgo, y la L² es
√(Σ pᵢ³/12). No hace falta ninguna librería de paisajes: basta el diagrama de
persistencia.

**Qué medí.** **[MEDIDO]** Réplica sobre los mismos cuatro índices (S&P 500, Dow
Jones, Nasdaq, Russell 2000), de 1992 a 2026, 8.697 ventanas de 50 sesiones:

- **La réplica funciona donde el artículo la mostró.** En las 20 sesiones
  previas al máximo de marzo de 2000, la norma estaba en el percentil 99,1 de
  los tres años anteriores; antes de septiembre de 2008, en el 99,6. La varianza
  estaba en el percentil 89 en ambos casos.
- **Falla en el único desplome posterior a la publicación.** Antes del máximo de
  febrero de 2020 estaba en el percentil 23.
- **No añade capacidad de predicción.** Para predecir la varianza realizada de
  las 21 sesiones siguientes, la varianza pasada explica un R² de 0,4754; al
  añadir la norma topológica sube a 0,4774, con t = 1,43. El umbral fijado era
  |t| ≥ 3. Sobre el retorno futuro, t = −1,41.
- Sola, la norma «predice» la varianza (t = 5,4) únicamente porque correlaciona
  con la varianza pasada (0,48).

**Decisión.** Descartado.

### 2.7 Lo que no medí

Mapas de difusión, embeddings hiperbólicos y otras técnicas de reducción no
lineal no se han medido. **[RAZONADO]** Producirían o un estado por fecha (otro
candidato a canal de G0, con el problema ya visto) o coordenadas por acción
(otro candidato a feature, con el resultado de la sección 3.2). No hay motivo
para esperar algo distinto y cada candidato añadido cuesta potencia estadística.

### 2.8 Resumen de la sección

| Método | Qué produce | Resultado | Decisión |
|---|---|---|---|
| Fisher-Rao, dispersión | Estado por fecha | Detecta lo mismo que la volatilidad, sin días fuera de episodio | Se mantiene (v1.2) |
| SPD, dependencia | Estado por fecha | Detecta un episodio más, tarde; más días marcados | Se mantiene (v1.2) |
| Wasserstein, forma | Estado por fecha | Ve el único episodio de colas; duplica los días marcados | No se añade |
| Fisher en análogos | Distancia entre casos | No bate a la euclídea a 21 sesiones (v1.2) | Opcional, no por defecto |
| Gradiente natural | Optimizador | No ejecutable; no afecta a la información | No implementado |
| Curvatura de Ricci | Estado por fecha | No es solo correlación media (0,61); valor predictivo sin contrastar | No implementado |
| TDA | Estado por fecha | Replica 2000 y 2008, falla 2020; t = 1,43 | Descartado |
| Fisher-Rao por acción, periferia | Feature por acción | Informativa sola, redundante con los 14 factores | No se añade |

Seis familias con contraste (modelo base, agente geométrico, TDA, multiplicador
de acuerdo, agregador, coherencia). El p-valor más bajo sin ajustar es 0,13; con
el ajuste de Holm, todos quedan en 0,79 o más.

---

## 3. ¿Modelo, agente o enjambre?

### 3.1 El criterio que lo decide

**[RAZONADO]** En swarmtrader un agente es un ordenador de acciones: produce una
puntuación por acción y fecha, y se le juzga por su rank IC dentro de cada
fecha. De ahí sale una regla que no depende de ninguna medición:

- Una técnica que produce **un número por fecha** (estado del mercado, curvatura
  de la red, norma topológica) da el mismo valor a todas las acciones ese día.
  No puede ordenarlas. **No puede ser un agente.** Solo puede ser un filtro, un
  modulador de tamaño o una variable que condicione a otros modelos.
- Una técnica que produce **un número por acción y fecha** sí puede ser un
  agente o una feature. Es una pregunta empírica.
- Una técnica que opera sobre **las salidas de los agentes** pertenece a la capa
  de combinación.

Toda la geometría «de mercado» (regímenes, curvatura, topología) cae en el
primer grupo. Preguntarse si conviene implementarla como agente no tiene
respuesta porque la pregunta no está bien formada.

### 3.2 Opción 2: un agente geométrico

**Qué probé.** Tres features por acción:

- `geo_fr_shift`: distancia de Fisher-Rao entre la gaussiana de los retornos
  diarios de las últimas 21 sesiones y la de las 231 anteriores. Mide cuánto ha
  cambiado el comportamiento del valor.
- `geo_fr_signed`: la misma con el signo del cambio de media.
- `geo_periphery`: periferia en la red de correlaciones (menos correlación media
  con el resto). Tesis de Pozzi, Di Matteo y Aste (2013): las carteras de valores
  periféricos tienen mejor relación señal-ruido. Ese trabajo no considera costes
  de transacción. **[VERIFICADO]**

**Resultado.** **[MEDIDO]**

| Modelo | Rank IC (t) |
|---|---|
| 14 factores (base) | 0,0823 (4,5) |
| 14 + las dos de Fisher-Rao | 0,0819 |
| 14 + periferia | 0,0803 |
| 14 + las tres | 0,0813 |
| Solo las tres geométricas | 0,0603 (4,6) |

H0 «ninguna variante mejora»: p = 1,00 (p = 0,71 entrando un día más tarde).

Las features **son informativas por sí solas**: un agente geométrico tendría un
IC de 0,060 con t = 4,6, comparable al de liquidity. Y **son redundantes**:
añadidas a lo que ya hay, la mejora es nula o negativa. La razón es que
reexpresan volatilidad. La distancia de Fisher-Rao de un valor a su propio
pasado es grande sobre todo cuando su volatilidad ha cambiado, y la periferia en
la red es en esencia la parte del riesgo que no es de mercado.

Un detalle: el signo de la periferia sale **al revés** que en Pozzi y coautores.
En esta muestra los valores periféricos rinden peor (IC univariante −0,068,
t = −4,6).

**La vía «agente» propiamente dicha.** **[EXPLORATORIO]** Añadirlo como sexto
agente al enjambre equiponderado sube el IC de 0,0746 a 0,0811 (p = 0,076). No
llega al 5%, es posterior a ver los datos, y sobre todo tiene una explicación
que no es favorable: 0,081 es lo que un único ridge con los 14 factores ya
consigue sin geometría (0,082). El agente geométrico no aporta información
nueva; compensa parte de lo que el enjambre pierde por ponderar igual a agentes
solapados.

**Decisión.** No se crea el agente. Incumple la regla fijada (p < 0,05 y
sobrevivir al retraso de entrada).

### 3.3 Opción 3: geometría sobre el propio enjambre

**El agregador actual ya es geométrico.** **[RAZONADO]** Cada agente emite un
rango por acción. La media de rangos es el punto que minimiza la suma de
distancias al cuadrado a los rangos de todos los agentes; ordenar por esa media
da la permutación que minimiza la suma de distancias de Spearman al cuadrado. Es
decir, la equiponderación actual es la media de Fréchet del enjambre bajo la
distancia de Spearman. No hay nada que sustituir.

**Baricentro de Wasserstein.** **[RAZONADO]** Se aplica a distribuciones de
probabilidad, y los agentes no emiten distribuciones sino rangos. Si las
emitieran, en una dimensión el baricentro sería la media de cuantiles, que
Lichtendahl, Grushka-Cockayne y Winkler (2013) encuentran más nítida que la
media de probabilidades. **[VERIFICADO]** Para aprovecharlo habría que rediseñar
los agentes para que emitan distribuciones predictivas, y nada medido aquí
justifica ese rediseño.

**Agregadores alternativos.** **[MEDIDO]**

| Agregador | Rank IC (t) |
|---|---|
| Media de rangos (actual) | 0,0746 (4,6) |
| Mediana | 0,0768 (4,8) |
| Media en logits | 0,0715 (4,6) |

H0 «ninguno bate a la media»: p = 0,14.

**Desacuerdo como incertidumbre.** **[MEDIDO]** Era la parte más atractiva de la
propuesta: medir cuánto discrepan los agentes y reducir tamaño o bloquear cuando
discrepan mucho. Resulta que el sistema ya lo hacía, con el multiplicador de
acuerdo, y medido no ayuda: empeora ligeramente el IC, los valores con menos
desacuerdo no rinden más dentro del quintil superior, y la coherencia del
enjambre en una fecha no predice su IC.

**Decisión.** Nada que añadir. Se retira el multiplicador.

### 3.4 Comparativa

| Opción | A favor | En contra | Latencia y coste | Resultado |
|---|---|---|---|---|
| 1. Modelo o motor de estado | Encaja con el tipo de objeto; determinista; solo puede vetar | No detecta más que una línea base trivial; valor económico sin demostrar | Milisegundos; solo numpy y scipy | Es lo que ya hay (G0); no se amplía |
| 2. Agente geométrico | Informativo por sí solo | Redundante; suma un candidato a la comparación múltiple | Bajo | No se crea |
| 3. Geometría del enjambre | La media actual ya es una media de Fréchet | El desacuerdo no predice nada; el baricentro exige rediseñar los agentes | Nulo | Nada que añadir; se quita una heurística |
| 4. Híbrido | — | Suma las complejidades de 1 y 3 sin que ninguna aporte | — | No justificado |
| 5. Descarte | Coherente con la evidencia | — | — | TDA, Ricci, forma, gradiente natural, agente |

---

## 4. Veredicto y diseño

### 4.1 Respuesta directa

**No tiene sentido crear un modelo geométrico nuevo ni un agente geométrico, y
no hay nada que cambiar en la combinación del enjambre.** La geometría de la
información tiene un papel legítimo y limitado en swarmtrader como regla de
medida dentro de la capa de filtros, que es donde ya estaba desde la v1.2. Todo
lo demás que se podía probar se ha probado y no se gana el sitio.

La auditoría ha servido para otra cosa: encontrar que el sistema validaba la
cantidad equivocada, y que su modelo, tal como está, no muestra capacidad de
elegir ganadores.

### 4.2 El sistema en la v1.3.0

```
datos (precio ajustado + precio nominal si existe)
  -> filtros de universo fecha a fecha, sobre el precio que cotizaba
  -> 14 factores en z-score por fecha
  -> agentes: ridge por tesis economica  (5 activos, 6 a la espera de datos)
  -> combinacion: media de rangos; apilado solo si demuestra batirla
  -> filtros del veredicto
       G0  ¿el mercado de hoy esta dentro del soporte de entrenamiento?   (Fisher-Rao)
       G1  ¿hay destreza en alguna parte del enjambre?                    (SPA)
       G2  ¿el apilado bate a la media?
       G3  ¿sobrevive a la seleccion del mejor agente?                    (PBO)
       G4  ¿los K primeros baten al universo?                             (NUEVO)
  -> calibracion -> Kelly fraccional -> ciclo diario -> Alpaca (paper por defecto)
```

Cambios respecto a la v1.2.0:

| Cambio | Tipo |
|---|---|
| Filtro G4 | Nuevo contraste en el veredicto; solo puede vetar |
| Adaptador de Sharadar sin filtro por último precio | Corrección de sesgo de supervivencia |
| `MarketData.nominal` y filtros de precio point-in-time | Corrección de anticipación |
| Multiplicador de acuerdo desactivado | Retirada de heurística sin validar |
| Dos experimentos y 13 tests nuevos | Evidencia y garantía |

Ningún modelo nuevo, ningún agente nuevo, ninguna dependencia nueva en
producción. Pasan 210 tests; el control de ruido sigue dando NO ACCIONABLE y el
de señal inyectada ACCIONABLE.

Efecto del cambio en el control de señal: el peso bruto de la cartera pasa del
74,5% al 72,3%, por la retirada del multiplicador.

### 4.3 Decisiones que son tuyas

No las tomo yo porque la evidencia no las zanja.

**Modo del filtro G0.**

- `gate` (por defecto): abstenerse alrededor del 3% de los días. Coherente con
  «sin evidencia en este estado, no se actúa». La evidencia nueva dice que no
  detecta más que la volatilidad y que el IC lejos del soporte baja pero sigue
  siendo positivo.
- `--drift-mode warn`: informa sin vetar. No pierde días y acumula historial
  para contrastar más adelante la hipótesis de reducir tamaño en vez de vetar.

**Enjambre o modelo único.** Mientras solo haya datos de precio, `swarmtrader
rank` (un ridge con los 14 factores) rinde al menos igual que `swarmtrader
swarm` y es más simple. El enjambre se justifica cuando entren fuentes
heterogéneas.

**Universo.** El IC se concentra por debajo de 5 USD, que es también donde los
costes son mayores y donde el sesgo de supervivencia es más grave. Con precio de
10 USD o más el IC es 0,036 (t = 2,4).

### 4.4 Lo que esto significa para el objetivo del proyecto

Con los datos y factores actuales, no hay evidencia de que el sistema pueda
entregar «las 3 a 5 acciones con mayor probabilidad de una subida notable». Lo
que hay es evidencia de dos cosas distintas:

- Sabe identificar valores que van a ir mal. Eso tiene utilidad como filtro de
  exclusión, no como selector de compras.
- Perseguir directamente la probabilidad de subida grande la aumenta, y a la vez
  hace negativo el retorno medio.

La geometría no cambia ninguna de las dos cosas, porque ninguna de las dos es un
problema de cómo se mide una distancia. Es un problema de información: qué datos
entran.

---

## 5. Hoja de ruta de implementación

### 5.1 Especificación de lo implementado

**G4, exceso de los K primeros.** Para cada fecha fuera de muestra *t*, con *S*
la puntuación del combinador y *r* el retorno a *h* sesiones:

    e_t = media{ r_i : i entre los K con mayor S en t } − media{ r_i : i elegible en t }

Se contrasta H0: E[e] ≤ 0 con el mismo bootstrap estacionario que G1 (`spa_test`
con una sola columna). El veredicto exige p < `spa_alpha`. Código:
`swarm_rank.topk_excess_by_date` y el bloque G4 de `swarm_rank.swarm_rank`.
Limitación: si el universo contiene muchos valores muy malos, evitarlos basta
para batir a la media del universo; eso es una ventaja real y G4 la deja pasar.

**Distancias (sin cambios desde la v1.2).** En `stats/infogeo.py`:

    Gaussiana:   d = √2 · arccosh(1 + ((μ₁−μ₂)²/2 + (σ₁−σ₂)²) / (2σ₁σ₂))
    Bernoulli:   d = 2·|arcsin√p − arcsin√q|
    SPD:         d = √(½ Σ ln² λᵢ),  λ = autovalores de A^(−1/2) B A^(−1/2)

**Precio nominal.** `MarketData(prices, volumes, nominal=None)`. Si `nominal`
existe, `min_price` y `max_price` se aplican sobre él; si no, sobre el ajustado,
con la limitación documentada.

### 5.2 Librerías

| Librería | Uso | ¿Hace falta? |
|---|---|---|
| numpy, scipy, pandas | Todo lo que está en producción | Sí; ya eran dependencias |
| ripser 0.6.15 | Diagramas de persistencia (bloque C2) | Solo para reproducir el experimento |
| POT 0.9.7 | Transporte óptimo por arista (bloque C3) | Solo para reproducir el experimento |
| networkx 3.7 | Red de correlaciones (bloque C3) | Solo para reproducir el experimento |
| geomstats | Variedades en general | **No.** Las tres distancias que se usan tienen forma cerrada |
| giotto-tda, gudhi | Paisajes de persistencia | **No.** Las normas L¹ y L² salen del diagrama (2.6), y TDA queda descartado |

Las versiones son las de este entorno (Linux, Python 3.13). No he comprobado la
instalación de ripser ni de POT en macOS ARM64; el código de producción no
depende de ellas.

### 5.3 Cómo reproducirlo

```bash
python3 -m pytest -q                                   # 210 tests
python3 scripts/swarm_null_test.py                     # control de ruido y de señal
python3 experiments/geometry_audit.py --stocks-panel panel.parquet \
        --indices-dir indices/ --json resultados.json  # pre-registrado, ~5 min
python3 experiments/audit_followup.py --stocks-panel panel.parquet   # exploratorio
python3 experiments/audit_followup.py --only-f7 --stocks-panel panel.parquet
```

El panel de precios no va en el paquete (191 MB). Formato esperado: parquet
largo con `date, ticker, adjclose, close_nominal, dollar_volume`.

### 5.4 Siguientes pasos, por orden

1. **Datos sin sesgo de supervivencia.** Es el bloqueo de todo lo demás. Con el
   adaptador de Sharadar corregido, el plan de 9 USD al mes ya sirve. Criterio de
   éxito: repetir `audit_followup.py --only-f7` y ver si el exceso de los 5
   primeros sigue siendo nulo.
2. **Decidir qué se quiere del sistema** a la vista de 1.5 y 1.6. Hay tres
   caminos y son excluyentes en esfuerzo:
   - *Filtro de exclusión.* Usar el ranking para descartar el decil inferior y
     elegir dentro del resto con otro criterio. Es lo único para lo que hay
     evidencia hoy.
   - *Selector de compras.* Exige información que distinga ganadores. Los
     candidatos son las fuentes que ya tienen agente y no tienen datos, más una
     función de pérdida que pese la parte alta del ranking. Poh, Lim, Zohren y
     Roberts (2020) reportan mejoras grandes con aprendizaje de ordenación en
     momentum cross-seccional; no he podido verificar sus datos ni si incluyen
     costes. **[VERIFICADO]** a nivel de resumen.
   - *Aceptar el perfil de lotería.* Posible, pero con el coste medido en 1.6 y
     sabiendo que el dato real será peor.
3. **Hipótesis para contrastar con datos point-in-time**, fijadas aquí de
   antemano:
   - El tamaño de posición debe reducirse en función del percentil de la
     distancia de G0 (2.1). Criterio: mejora del crecimiento logarítmico de la
     cartera frente a tamaño constante, p < 0,05.
   - `fisher_rank` bate a la euclídea a 5 sesiones en el motor de análogos
     (pendiente de la v1.2).
   - El canal de dispersión solo es mejor filtro que los dos canales juntos.
4. **Adaptadores de Alpaca y yfinance**: obtener también el cierre sin ajustar
   para que el filtro de precio sea point-in-time con ellos.
5. **Defectos anteriores sin corregir**: la colisión de 178 tickers entre
   acciones y criptomonedas en `data/venues.py` y el calendario de 252 sesiones
   incrustado.

### 5.5 Lo que no conviene hacer

- Añadir más canales, más agentes o más métricas geométricas antes de tener
  datos sin sesgo. Cada candidato cuesta potencia estadística y ninguno de los
  probados ha aportado.
- Retirar agentes o cambiar valores por defecto a partir de esta muestra.
- Leer el IC de 0,075 como señal de que el sistema funciona. El único número
  que importa para una cartera que solo compra es el exceso de los K primeros, y
  es cero.

---

## 6. Lo que no está demostrado

- El valor económico del filtro G0, en cualquiera de sus configuraciones.
- Que el resultado de 1.5 se mantenga con las empresas excluidas de cotización
  dentro de la muestra. Puede cambiar en cualquier dirección.
- Cualquier cifra neta de costes.
- El comportamiento de `deep_rank` y de los seis agentes sin datos.
- Que la curvatura de Ricci carezca de valor predictivo: no se contrastó.
- Las fuentes externas se consultaron casi todas a nivel de resumen. Donde digo
  que un trabajo «no compara» o «no reporta» algo, es lo que se desprende del
  texto consultado.

## 7. Fuentes

Consultadas el 2026-11-11.

- Hammond, W. (2026). *Geometric Observables for Financial Regime Detection.* [arXiv:2605.17117](https://arxiv.org/abs/2605.17117)
- Camara, E. H. B., Gaba, Y. U., Mabanza Nsiloulou, I. G. (2026). *Geometric turbulence: a geodesic-regression crisis indicator for equity covariance dynamics.* [arXiv:2608.15205](https://arxiv.org/abs/2608.15205)
- Bucci, A., Palma, M., Zhang, C. (2024). *Geometric Deep Learning for Realized Covariance Matrix Forecasting.* [arXiv:2412.09517](https://arxiv.org/abs/2412.09517v1)
- Samal, A. et al. *Network geometry and market instability.* [arXiv:2009.12335](https://arxiv.org/abs/2009.12335v2)
- Sandhu, R. S., Georgiou, T. T., Tannenbaum, A. R. (2016). *Ricci curvature: An economic indicator for market fragility and systemic risk.* Science Advances. [PMC4928924](https://pmc.ncbi.nlm.nih.gov/articles/PMC4928924)
- Gidea, M., Katz, Y. *Topological Data Analysis of Financial Time Series: Landscapes of Crashes.* [arXiv:1703.04385](https://arxiv.org/abs/1703.04385v1)
- Horvath, B., Issa, Z., Muguruza, A. (2021). *Clustering Market Regimes using the Wasserstein Distance.* [arXiv:2110.11848](https://arxiv.org/abs/2110.11848v1)
- Lichtendahl, K. C., Grushka-Cockayne, Y., Winkler, R. L. (2013). *Is It Better to Average Probabilities or Quantiles?* Management Science 59(7). [RePEc](https://ideas.repec.org/a/inm/ormnsc/v59y2013i7p1594-1611.html)
- Pozzi, F., Di Matteo, T., Aste, T. (2013). *Spread of risk across financial markets: better to invest in the peripheries.* Scientific Reports 3, 1665. [Nature](https://www.nature.com/articles/srep01665)
- Rahimikia, E., Ni, H., Wang, W. (2025). *Re(Visiting) Time Series Foundation Models in Finance.* [arXiv:2511.18578](https://arxiv.org/abs/2511.18578v1)
- Avramov, D., Cheng, S., Metzker, L. (2023). *Machine Learning vs. Economic Restrictions: Evidence from Stock Return Predictability.* Management Science 69(5). [RePEc](https://econpapers.repec.org/RePEc:inm:ormnsc:v:69:y:2023:i:5:p:2587-2619)
- Martineau, C. (2022). *Rest in Peace Post-Earnings Announcement Drift.* Critical Finance Review 11(3-4). [RePEc](https://ideas.repec.org/a/now/jnlcfr/104.00000122.html)
- Gu, S., Kelly, B. T., Xiu, D. (2020). *Empirical Asset Pricing via Machine Learning.* Review of Financial Studies. [AQR](https://www.aqr.com/Insights/Research/Journal-Article/Empirical-Asset-Pricing-via-Machine-Learning)
- Avramov, D., Chordia, T., Goyal, A. (2006). *Liquidity and Autocorrelations in Individual Stock Returns.* Journal of Finance 61(5). [RePEc](https://ideas.repec.org/a/bla/jfinan/v61y2006i5p2365-2394.html)
- Poh, D., Lim, B., Zohren, S., Roberts, S. (2020). *Building Cross-Sectional Systematic Strategies By Learning to Rank.* [arXiv:2012.07149](https://arxiv.org/abs/2012.07149)
- Kunstner, F., Balles, L., Hennig, P. (2019). *Limitations of the Empirical Fisher Approximation for Natural Gradient Descent.* [arXiv:1905.12558](https://arxiv.org/abs/1905.12558v3)
