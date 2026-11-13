"""Asignacion por conviccion: la mejor cartera que el sistema sabe construir,
sin veredicto que la bloquee y sin topes que la recorten.

Que resuelve
------------
El modo por defecto (`decision="gated"`) responde "¿hay evidencia de que el
ranking sirva?" y, si no la hay, no compra. Este modulo responde otra pregunta:
"si hay que elegir de todos modos, ¿cual es la mejor eleccion con lo que se
sabe?". Son preguntas distintas y ninguna sustituye a la otra: el veredicto se
sigue calculando e informando, pero aqui no veta.

El problema
-----------
    max_b  E[ u(b . X) ]    sujeto a  b >= 0,  sum(b) = 1

X es el vector de retornos BRUTOS (1 + r) a `horizon` sesiones y la esperanza
se toma sobre la distribucion PREDICTIVA (ver mas abajo). No hay tope por
nombre, ni por sector, ni numero minimo de posiciones: el conjunto factible es
el simplex entero, vertices incluidos. Un vertice es el 100 % en una accion.

u es la utilidad CRRA con aversion al riesgo gamma:

    gamma = 1   u = ln x. Cartera log-optima (criterio de Kelly): maximiza la
                tasa de crecimiento geometrico esperada. Es el defecto.
    gamma = 0   u = x. Neutral al riesgo: la solucion es SIEMPRE un vertice, el
                activo de mayor retorno esperado.
    gamma > 1   mas conservador que Kelly; gamma < 1, mas concentrado.

Cuando sale una sola accion y cuando varias (gamma = 1)
-------------------------------------------------------
Condiciones de Kuhn-Tucker de la cartera log-optima (Bell y Cover, 1980;
Cover y Thomas, 2.a ed., teorema 16.2.1):

    E[ X_i / (b* . X) ]  = 1   si b*_i > 0
    E[ X_i / (b* . X) ] <= 1   si b*_i = 0

De ahi sale el criterio exacto, sin ninguna regla anadida:

  * todo el capital va al activo j  <=>  E[X_i / X_j] <= 1 para todo i;
  * un segundo activo i entra       <=>  E[X_i / X_j] > 1. A segundo orden,
    mu_i - mu_j > cov(R_i, R_j) - var(R_j): entra cuando lo que pierde en
    retorno esperado es menos que lo que reduce la varianza. Con dos acciones
    de volatilidad sigma y correlacion rho, la mejor se lo lleva todo si su
    ventaja supera a sigma^2 (1 - rho): con sigma = 12 % mensual y rho = 0,3,
    alrededor de un 1 % de retorno esperado al mes.

La diversificacion no se impone: aparece solo si sube el crecimiento esperado.
Y la matematica tiene un limite propio: con utilidad logaritmica, una accion
con algun escenario de perdida total nunca recibe el 100 % (ln 0 = -inf). No es
un tope de seguridad, es el optimo.

Por que no un softmax con temperatura
-------------------------------------
softmax(s / tau) nunca da ceros exactos ni un 100 % exacto salvo tau -> 0, y
tau no tiene unidades financieras, asi que habria que elegirlo a mano. La
aproximacion cuadratica del problema de arriba, max b'mu - (gamma/2) b'Sigma b
sobre el simplex, es una proyeccion euclidea sobre el simplex cuando Sigma es
esferica, es decir sparsemax(mu / (gamma sigma^2)) (Martins y Astudillo, 2016):
la "temperatura" existe, pero es gamma * sigma^2 y sigma^2 la ponen los datos.

La distribucion predictiva
--------------------------
Meter estimaciones puntuales en Kelly sobreapuesta, porque trata como conocido
lo que es incierto (Baker y McHale, 2013). Aqui la incertidumbre entra en los
propios escenarios:

  1. **Media condicionada E[r | puntuacion].** Se estima con las predicciones
     walk-forward fuera de muestra del propio enjambre: en cada fecha, las
     acciones se ordenan por percentil de puntuacion DENTRO de esa fecha y se
     agrupan en K tramos de igual anchura; el dato es el retorno medio de cada
     tramo. Por percentil y no por valor, porque la escala de la puntuacion
     cambia entre fechas. No se asume que sea creciente: sobre datos reales no
     lo era (subia hasta el septimo decil y luego bajaba), asi que una
     calibracion isotonica lo habria falseado.
  2. **Encogimiento jerarquico.** Con ~18 fechas y 10 tramos, el tramo
     con mejor suerte aparenta un exceso de varias decimas al mes aunque todo
     sea ruido, y un optimizador de crecimiento apuesta exactamente por ese
     tramo. El exceso de cada tramo sobre la media del dia, e_k, se encoge hacia
     cero con lambda_k = tau^2 / (tau^2 + se_k^2), donde se_k^2 es su error
     estandar al cuadrado y tau la dispersion REAL entre tramos. Dos formas de
     tratar tau (`ConvictionConfig.tau`):
       * "point" (empirico-bayesiano): tau^2 por momentos,
         max(0, media(e_k^2) - media(se_k^2)). Si la dispersion entre tramos no
         supera al ruido, tau^2 = 0 y el mapa queda plano.
       * "integrated" (bayesiano completo): cada escenario sortea tau de su
         posterior (a priori uniforme en tau, Gelman 2006) y encoge con ese
         sorteo; la incertidumbre sobre tau entra en los escenarios en vez de
         fijarse en un punto.
     No hay ningun factor elegido a mano en ninguna de las dos.
  3. **Incertidumbre de esa media.** Bootstrap bayesiano (Rubin, 1981) sobre las
     fechas: cada escenario pondera las fechas con pesos Dirichlet(1,...,1) y
     usa la posterior normal-normal del tramo, media lambda e_k y desviacion
     sqrt(lambda) veces la del bootstrap.
  3b. **Nivel de mercado** (`market_level`). Por defecto ("oos") el retorno
     comun sale de las fechas fuera de muestra, con su bootstrap bayesiano: es
     lo adecuado para un solo plazo, y deja que un mercado esperado negativo
     lleve al efectivo cuando `allow_cash`. Al COMPARAR plazos (decision.py) eso
     no vale: con ~18 fechas el nivel es casi todo ruido y distinto en cada
     plazo, y elegir plazo seria elegir el ruido mas favorable. Con "history"
     sale de la media diaria de la seccion cruzada en toda la ventana de
     `lookback`, compuesta a `horizon`, con numeros aleatorios comunes: por
     sesion es identico en todos los plazos y no decide el plazo.
  3c. **Costes.** El coste de ida y vuelta de cada accion (`costs`) se descuenta
     del retorno bruto de cada escenario: el optimizador ve el retorno NETO.
  4. **Dependencia conjunta.** Bootstrap estacionario (Politis y Romano, 1994)
     de secciones cruzadas DIARIAS de log-retornos, compuestas a `horizon`
     sesiones: conserva correlaciones, colas y asimetrias reales entre las
     acciones, sin suponer normalidad. Se reescala para que la media de cada
     accion en los escenarios sea exactamente su media predictiva: asi no se
     cuenta dos veces la deriva historica (eso ya lo mide el momentum) y el
     optimizador no persigue ruido de simulacion en las medias.
  5. Escenario = (1 + media sorteada del tramo) x (residuo reescalado).

Garantia numerica
-----------------
El optimizador es una actualizacion multiplicativa, b_i <- b_i (g_i / gbar)^eta
con g_i = E[X_i u'(riqueza)], que para gamma = 1 y eta = 1 es el algoritmo de
Cover (1984). Un paso solo se acepta si sube el objetivo, y eta se adapta. Si el
paso multiplicativo no mejora, o si la accion que mas mejoraria tiene un peso
casi nulo (de donde un paso multiplicativo tarda en sacarla), se da un paso de
Frank-Wolfe con busqueda lineal hacia ese vertice. Se devuelve una cota superior
CERTIFICADA de lo que falta hasta el optimo: ln max_i g_i en el caso
logaritmico puro (Cover y Thomas, teorema 16.2.2) y la brecha de Frank-Wolfe
B (max_i g_i - gbar) en general (Jaggi, 2013). El universo completo se trata
por generacion de columnas: se resuelve sobre un subconjunto, se comprueba la
condicion de Kuhn-Tucker en TODAS las demas acciones y, si alguna la viola,
entra y se repite. La cota final se evalua sobre el universo entero.

Lo que este modulo NO afirma
----------------------------
Que la cartera resultante tenga ventaja. Es la mejor decision bajo la
informacion disponible; si esa informacion no distingue ganadores, el mapa sale
plano y la cartera es la de mayor crecimiento con medias iguales (en la
practica, la de menor riesgo), no una apuesta.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

__all__ = ["CASH", "ConvictionConfig", "ConvictionResult", "SimplexSolution",
           "certainty_equivalent", "solve_simplex", "allocate", "build_bank"]

CASH = "__EFECTIVO__"
_FLOOR = 1e-30          # peso minimo interno de la actualizacion multiplicativa
_INITIAL_ACTIVE = 40    # generacion de columnas: tamano del primer subconjunto
_ADD_PER_ROUND = 20     # acciones que entran por ronda
_MAX_ROUNDS = 60


@dataclass
class ConvictionConfig:
    risk_aversion: float = 1.0      # gamma. 1 = Kelly, 0 = neutral al riesgo (siempre un vertice)
    allow_cash: bool = False        # False = eleccion forzada: todo invertido
    n_scenarios: int = 3000
    lookback: int = 1260            # sesiones de historia para la dependencia conjunta
    min_lookback: int = 252
    block: float = 5.0              # longitud media de bloque del bootstrap, en sesiones
    map_bins: int = 10              # tramos de percentil de la media condicionada
    min_map_dates: int = 6          # fechas fuera de muestra minimas para estimarla
    shrink: bool = True             # encogimiento jerarquico del mapa
    tau: str = "point"              # "point" (empirico-bayesiano) | "integrated"
    # Nivel de mercado comun: "oos" = media de las fechas fuera de muestra (sirve
    # para UN plazo); "history" = media diaria de toda la ventana compuesta al
    # plazo, con numeros aleatorios comunes (necesario al COMPARAR plazos).
    market_level: str = "oos"
    # Minimo OPERATIVO, no de riesgo: por debajo, la orden no llega al minimo
    # del broker (1 USD en Alpaca). Fraccion del patrimonio.
    min_weight: float = 0.01
    tol: float = 1e-7               # cota de suboptimalidad aceptada (unidades de utilidad)
    max_iter: int = 20000
    seed: int = 0


@dataclass
class SimplexSolution:
    weights: np.ndarray             # sobre las columnas; suman `budget`
    utility: float
    gap: float                      # cota superior certificada de U* - U(b)
    iterations: int
    gradient: np.ndarray            # E[X_i u'(riqueza)]
    converged: bool


@dataclass
class ConvictionResult:
    weights: pd.Series              # simbolo -> fraccion del patrimonio (> 0)
    cash: float                     # fraccion que el optimizador deja en efectivo
    table: pd.DataFrame             # symbol, weight, score, mu, sd, p_success
    certainty_equivalent: float     # retorno equivalente cierto por horizonte
    gap: float
    alternatives: dict = field(default_factory=dict)
    info: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        def _num(v):
            return v if not isinstance(v, (np.floating, np.integer)) else v.item()
        return {
            "weights": {k: round(float(v), 6) for k, v in self.weights.items()},
            "cash": round(float(self.cash), 6),
            "certainty_equivalent": round(float(self.certainty_equivalent), 6),
            "gap": float(self.gap),
            "alternatives": {k: round(float(v), 6) for k, v in self.alternatives.items()},
            "table": self.table.round(6).to_dict(orient="records"),
            "info": {k: _num(v) for k, v in self.info.items()},
            "notes": list(self.notes),
        }


# ------------------------------------------------------------------ utilidad
def _is_log(gamma: float) -> bool:
    return abs(gamma - 1.0) < 1e-12


def _utility(wealth: np.ndarray, gamma: float) -> float:
    if _is_log(gamma):
        return float(np.log(wealth).mean())
    return float(((wealth ** (1.0 - gamma) - 1.0) / (1.0 - gamma)).mean())


def certainty_equivalent(wealth: np.ndarray, gamma: float = 1.0) -> float:
    """Retorno cierto que da la misma utilidad que la distribucion `wealth` (bruta)."""
    wealth = np.asarray(wealth, dtype=float)
    if _is_log(gamma):
        return float(np.exp(np.log(wealth).mean()) - 1.0)
    return float(np.mean(wealth ** (1.0 - gamma)) ** (1.0 / (1.0 - gamma)) - 1.0)


# ---------------------------------------------------------------- optimizador
def _frank_wolfe_step(X: np.ndarray, base: np.ndarray, b: np.ndarray, budget: float,
                      gamma: float) -> np.ndarray | None:
    """Paso hacia el vertice con mayor gradiente, con busqueda lineal exacta.

    phi(t) = U((1-t) b + t B e_j) es concava en t; se busca la raiz de phi' por
    biseccion. La riqueza es combinacion convexa de riquezas positivas, asi que
    nunca se sale del dominio.
    """
    Wb = X @ b
    W0 = base + Wb
    grad = (W0 ** (-gamma)) @ X / X.shape[0]
    j = int(np.argmax(grad))
    d = budget * X[:, j] - Wb

    def dphi(t: float) -> float:
        return float(np.mean((W0 + t * d) ** (-gamma) * d))

    if dphi(0.0) <= 0.0:
        return None
    if dphi(1.0) >= 0.0:
        t = 1.0
    else:
        lo, hi = 0.0, 1.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if dphi(mid) > 0.0:
                lo = mid
            else:
                hi = mid
        t = lo
    if t <= 0.0:
        return None
    new = (1.0 - t) * b
    new[j] += t * budget
    return new


def solve_simplex(X: np.ndarray, gamma: float = 1.0, budget: float = 1.0,
                  base: np.ndarray | None = None, tol: float = 1e-7,
                  max_iter: int = 20000) -> SimplexSolution:
    """max E[u(base + X b)] con b >= 0 y sum(b) = budget.

    X     (S, n) retornos brutos por escenario, estrictamente positivos.
    base  (S,) riqueza bruta que aportan las posiciones que NO se tocan
          (sum_k w_k X_k). None = no hay ninguna.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[0] == 0 or X.shape[1] == 0:
        raise ValueError("X debe ser una matriz (escenarios x activos) no vacia")
    if not np.isfinite(X).all() or (X <= 0.0).any():
        raise ValueError("los retornos brutos deben ser finitos y estrictamente positivos")
    if not np.isfinite(gamma) or gamma < 0.0:
        raise ValueError("la aversion al riesgo debe ser >= 0")
    if not np.isfinite(budget) or budget <= 0.0:
        raise ValueError("el presupuesto debe ser > 0")
    S, n = X.shape
    base = np.zeros(S) if base is None else np.asarray(base, dtype=float)
    if base.shape != (S,) or not np.isfinite(base).all() or (base < 0.0).any():
        raise ValueError("base debe ser un vector (S,) finito y no negativo")
    plain_log = _is_log(gamma) and not base.any() and abs(budget - 1.0) < 1e-12

    if gamma <= 1e-12:                      # objetivo lineal: un vertice, sin iterar
        # Empates dentro de la precision numerica se reparten a partes iguales:
        # con medias iguales cualquier reparto es optimo y elegir uno al azar
        # seria dejar que decida el redondeo.
        m = X.mean(axis=0)
        best = np.isclose(m, m.max(), rtol=0.0, atol=1e-12)
        b = np.where(best, budget / best.sum(), 0.0)
        return SimplexSolution(b, _utility(base + X @ b, 0.0), 0.0, 0, m, True)

    def _U(bb: np.ndarray) -> float:
        return _utility(base + X @ bb, gamma)

    def _state(bb: np.ndarray):
        grad = ((base + X @ bb) ** (-gamma)) @ X / S
        gbar = float(bb @ grad) / budget
        gmax = float(grad.max())
        if plain_log:
            return grad, gbar, float(np.log(max(gmax / gbar, 1.0)))
        return grad, gbar, float(budget * max(gmax - gbar, 0.0))

    b = np.full(n, budget / n)
    U = _U(b)
    eta, it, converged = 1.0, 0, False
    for it in range(1, max_iter + 1):
        g, gbar, gap = _state(b)
        if gap <= tol:
            converged = True
            break
        ratio = g / gbar
        moved = False
        while eta >= 1e-4:
            cand = np.maximum(b * ratio ** eta, _FLOOR)
            cand *= budget / cand.sum()
            Uc = _U(cand)
            if Uc > U:
                b, U, moved = cand, Uc, True
                eta = min(eta * 2.0, 64.0)
                break
            eta *= 0.5
        if not moved:
            eta = 1.0
        # Una accion cuyo peso cayo al suelo numerico tarda cientos de pasos
        # multiplicativos en volver aunque deba entrar. El paso de Frank-Wolfe la
        # trae de golpe y, si el multiplicativo se ha atascado, garantiza
        # progreso mientras la brecha sea positiva.
        j = int(np.argmax(g))
        if not moved or b[j] < 1e-6 * budget:
            fw = _frank_wolfe_step(X, base, b, budget, gamma)
            if fw is not None:
                Uf = _U(fw)
                if Uf > U:
                    b, U, moved = fw, Uf, True
        if not moved:                        # sin mejora representable: limite numerico
            break

    b = np.where(b < 1e-12 * budget, 0.0, b)
    b *= budget / b.sum()
    g, _, gap = _state(b)
    return SimplexSolution(b, _U(b), gap, it, g, converged or gap <= tol)


# ------------------------------------------------------- distribucion predictiva
def _bootstrap_days(n_days: int, n_scen: int, length: int, block: float,
                    rng: np.random.Generator) -> np.ndarray:
    """Indices (n_scen x length) del bootstrap estacionario sobre `n_days` sesiones."""
    p_new = 1.0 / max(1.0, float(block))
    idx = np.empty((n_scen, length), dtype=np.int64)
    idx[:, 0] = rng.integers(0, n_days, size=n_scen)
    for t in range(1, length):
        jump = rng.random(n_scen) < p_new
        idx[:, t] = np.where(jump, rng.integers(0, n_days, size=n_scen),
                             (idx[:, t - 1] + 1) % n_days)
    return idx


class _ScenarioBank:
    """Escenarios predictivos bajo demanda, por columnas.

    Con un universo de miles de simbolos la matriz completa (escenarios x
    simbolos) no cabe en memoria con holgura. Aqui solo se guardan los
    ingredientes (mapas sorteados, dias sorteados, retornos diarios) y cada
    bloque de columnas se materializa cuando hace falta.

    pct_today: percentil de puntuacion de HOY en (0, 1) por simbolo; NaN = sin
    puntuacion (se trata como la mediana).
    """

    def __init__(self, oos: pd.DataFrame, pct_today: pd.Series, prices: pd.DataFrame,
                 horizon: int, cfg: ConvictionConfig, rng: np.random.Generator,
                 costs: pd.Series | None = None):
        self.notes: list[str] = []
        self.info: dict = {}
        self.cfg = cfg
        self.horizon = int(horizon)
        S = int(cfg.n_scenarios)
        if S < 10:
            raise ValueError("n_scenarios debe ser >= 10")
        if int(horizon) < 1:
            raise ValueError("horizon debe ser >= 1")
        if cfg.tau not in ("point", "integrated"):
            raise ValueError(f"tau debe ser 'point' o 'integrated', no {cfg.tau!r}")
        if cfg.market_level not in ("oos", "history"):
            raise ValueError(f"market_level debe ser 'oos' o 'history', no {cfg.market_level!r}")

        # --- dependencia conjunta: retornos diarios con historia completa ---
        syms = [s for s in pct_today.index if s in prices.columns]
        px = prices[syms]

        def _logret(L: int):
            win = px.iloc[-(L + 1):]
            with np.errstate(divide="ignore", invalid="ignore"):
                lr = np.log(win.where(win > 0)).diff().iloc[1:]
            ok = lr.notna().all() & np.isfinite(lr).all()
            return lr, ok

        L = min(int(cfg.lookback), len(px) - 1)
        if L < cfg.min_lookback:
            raise ValueError(f"historia insuficiente: {L} sesiones < {cfg.min_lookback}")
        lr, ok = _logret(L)
        if len(ok) and ok.mean() < 0.8 and L > cfg.min_lookback:
            L = int(cfg.min_lookback)
            lr, ok = _logret(L)
        self.symbols = list(lr.columns[ok.to_numpy()])
        if not self.symbols:
            raise ValueError(f"ningun simbolo con historia completa en {L} sesiones")
        self.col = {s: i for i, s in enumerate(self.symbols)}
        self._lr = lr[self.symbols].to_numpy(dtype=np.float32)
        self._days = _bootstrap_days(L, S, int(horizon), cfg.block, rng)
        self.n_scenarios = S
        self.lookback = L
        dropped = len(syms) - len(self.symbols)
        if dropped:
            self.notes.append(f"{dropped} simbolos sin historia completa en {L} sesiones: "
                              "no asignables")
        # Coste de ida y vuelta por simbolo (fraccion), descontado de cada escenario.
        c = (pd.Series(costs, dtype=float).reindex(self.symbols) if costs is not None
             else pd.Series(0.0, index=self.symbols)).fillna(0.0).clip(0.0, 0.5)
        self._keep = 1.0 - c.to_numpy(dtype=float)
        lvl_hist = self._level(S) if cfg.market_level == "history" else None

        # --- media condicionada E[r | percentil] y su posterior ---
        o = oos.loc[:, ["date", "score", "fwd"]].dropna()
        o = o[np.isfinite(o["fwd"].astype(float)) & np.isfinite(o["score"].astype(float))]
        self.n_map_dates = int(o["date"].nunique())
        pct = pct_today.reindex(self.symbols).astype(float).fillna(0.5) \
            .clip(1e-9, 1.0 - 1e-9).to_numpy()
        K = 0
        if self.n_map_dates >= cfg.min_map_dates:
            per_date = float(o.groupby("date").size().median())
            K = int(min(cfg.map_bins, per_date // 8))
        if K >= 2:
            cnt = o.groupby("date")["score"].transform("count")
            rk = o.groupby("date")["score"].rank(method="average")
            p = ((rk - 0.5) / cnt).to_numpy()
            o = o.assign(bin=np.clip(np.floor(p * K).astype(int), 0, K - 1))
            M = o.pivot_table(index="date", columns="bin", values="fwd", aggfunc="mean") \
                .reindex(columns=range(K))
            day = o.groupby("date")["fwd"].mean().reindex(M.index)
            M = M.apply(lambda c: c.fillna(day)).to_numpy(dtype=float)
            D = M.shape[0]
            level = M.mean(axis=1)                       # retorno medio del dia
            E = M - level[:, None]                       # exceso de cada tramo
            e_bar = E.mean(axis=0)
            se2 = E.var(axis=0, ddof=1) / D
            omega = rng.dirichlet(np.ones(D), size=S)
            lvl_s = omega @ level if lvl_hist is None else lvl_hist      # S
            e_s = omega @ E                              # S x K
            if cfg.shrink and cfg.tau == "integrated":
                # tau sorteado de su posterior en cada escenario: e_k ~ N(0, tau^2
                # + se_k^2) con a priori uniforme en tau. Con ruido la posterior
                # se pega a 0; con senal clara, tau >> se y apenas se encoge.
                s2 = np.maximum(se2, 1e-300)
                hi = 4.0 * max(float(np.abs(e_bar).max()), float(np.sqrt(s2.mean())), 1e-9)
                taus = np.linspace(0.0, hi, 400)
                v = taus[:, None] ** 2 + s2[None, :]
                ll = -0.5 * (np.log(v) + e_bar[None, :] ** 2 / v).sum(axis=1)
                post = np.exp(ll - ll.max())
                post /= post.sum()
                tau_s = rng.choice(taus, size=S, p=post)
                B = tau_s[:, None] ** 2 / (tau_s[:, None] ** 2 + s2[None, :])   # S x K
                self._maps = lvl_s[:, None] + B * e_bar + np.sqrt(B) * (e_s - e_bar)
                tau2, lam = float(np.mean(tau_s ** 2)), B.mean(axis=0)
            elif cfg.shrink:
                tau2 = max(0.0, float(np.mean(e_bar ** 2) - np.mean(se2)))
                lam = np.where(se2 > 0.0, tau2 / np.maximum(tau2 + se2, 1e-300),
                               1.0 if tau2 > 0.0 else 0.0)
                self._maps = lvl_s[:, None] + lam * e_bar + np.sqrt(lam) * (e_s - e_bar)
            else:
                tau2, lam = float("nan"), np.ones(K)
                self._maps = lvl_s[:, None] + e_s
            centers = (np.arange(K) + 0.5) / K
            self._j = np.clip(np.floor(pct * K - 0.5).astype(int), 0, K - 2)
            self._w = np.clip((pct - centers[self._j]) * K, 0.0, 1.0)
            self.shrink = np.asarray(lam, dtype=float)
            self.info.update({
                "map_bins": K, "map_dates": D, "map_level": float(lvl_s.mean()),
                "map_tau": float(np.sqrt(tau2)) if np.isfinite(tau2) else None,
                "map_lambda_mean": float(lam.mean()),
                "map_spread_raw": float(e_bar.max() - e_bar.min()),
                "map_spread_shrunk": float((lam * e_bar).max() - (lam * e_bar).min()),
                "map_excess_by_bin": [round(float(x), 5) for x in lam * e_bar],
            })
        else:
            # Sin historia fuera de muestra suficiente no se distingue a nadie por
            # puntuacion: misma media para todos, con su incertidumbre.
            if lvl_hist is not None:
                lvl_s = lvl_hist
            else:
                day = o.groupby("date")["fwd"].mean().to_numpy(dtype=float)
                lvl_s = (rng.dirichlet(np.ones(len(day)), size=S) @ day if len(day)
                         else np.zeros(S))
            self.shrink = np.zeros(2)
            self._maps = np.column_stack([lvl_s, lvl_s])
            self._j = np.zeros(len(self.symbols), dtype=int)
            self._w = np.zeros(len(self.symbols))
            self.notes.append(f"media condicionada plana: {self.n_map_dates} fechas fuera de "
                              f"muestra (minimo {cfg.min_map_dates}) o universo demasiado "
                              "pequeno para formar tramos")
            self.info.update({"map_bins": 0, "map_dates": self.n_map_dates,
                              "map_level": float(lvl_s.mean()),
                              "map_tau": 0.0, "map_lambda_mean": 0.0,
                              "map_spread_raw": 0.0, "map_spread_shrunk": 0.0})
        self.n_bins = int(self.info["map_bins"])

    def _level(self, S: int) -> np.ndarray:
        """Retorno esperado del universo a `horizon` sesiones, con su incertidumbre.

        Media de la seccion cruzada de retornos diarios en toda la ventana de
        historia (no solo en las fechas fuera de muestra), compuesta a
        `horizon`: la misma tasa diaria para todos los plazos, asi que por sesion
        no decide el plazo. Incertidumbre: bootstrap bayesiano sobre los dias,
        con una semilla FIJA (numeros aleatorios comunes): la comparacion entre
        plazos no depende del sorteo.
        """
        daily = np.expm1(self._lr.astype(np.float64)).mean(axis=1)
        daily = daily[np.isfinite(daily)]
        if len(daily) < 20:
            return np.zeros(S)
        w = np.random.default_rng(self.cfg.seed + 7919).dirichlet(np.ones(len(daily)), size=S)
        return (1.0 + w @ daily) ** self.horizon - 1.0

    def gross(self, cols) -> np.ndarray:
        """Retornos brutos predictivos (S x len(cols)) de las columnas pedidas."""
        cols = np.asarray(list(cols), dtype=int)
        acc = np.zeros((self._days.shape[0], len(cols)), dtype=np.float64)
        sub = self._lr[:, cols]
        for t in range(self._days.shape[1]):
            acc += sub[self._days[:, t]]
        G = np.exp(acc)
        j, w = self._j[cols], self._w[cols]
        m = 1.0 + self._maps[:, j] * (1.0 - w) + self._maps[:, j + 1] * w
        # Residuo reescalado para que la media de cada accion en los escenarios
        # sea EXACTAMENTE la media predictiva, 1 + E[mu]. Quita la deriva
        # historica (no se cuenta dos veces) y el ruido de Monte Carlo en las
        # medias, que el optimizador perseguiria entre acciones casi iguales.
        G *= m.mean(axis=0, keepdims=True) / (m * G).mean(axis=0, keepdims=True)
        return np.maximum(m * G * self._keep[cols], 1e-6)

    def standalone(self, cols, threshold: float = 0.05, chunk: int = 512) -> pd.DataFrame:
        """Distribucion predictiva NETA de cada columna por separado (100 % en ella).

        mean_log  E[ln X]: crecimiento por horizonte si se apuesta todo a ella
        ce        equivalente cierto logaritmico, exp(E ln X) - 1
        mu        E[X] - 1;  p_up  P(X - 1 > threshold);  p_loss  P(X < 1)
        """
        cols = np.asarray(list(cols), dtype=int)
        rows = []
        for a in range(0, len(cols), chunk):
            G = self.gross(cols[a:a + chunk])
            rows.append(pd.DataFrame({
                "mean_log": np.log(G).mean(axis=0), "mu": G.mean(axis=0) - 1.0,
                "p_up": (G - 1.0 > threshold).mean(axis=0), "p_loss": (G < 1.0).mean(axis=0),
                "q10": np.quantile(G, 0.10, axis=0) - 1.0,
                "q50": np.quantile(G, 0.50, axis=0) - 1.0,
                "q90": np.quantile(G, 0.90, axis=0) - 1.0,
            }, index=[self.symbols[c] for c in cols[a:a + chunk]]))
        out = pd.concat(rows) if rows else pd.DataFrame(
            columns=["mean_log", "mu", "p_up", "p_loss", "q10", "q50", "q90"])
        out["ce"] = np.expm1(out["mean_log"].astype(float))
        return out

    def scan(self, cols, weights_by_scenario: np.ndarray, chunk: int = 512) -> np.ndarray:
        """E[X_i * v] para cada columna, sin materializar toda la matriz."""
        cols = np.asarray(list(cols), dtype=int)
        out = np.empty(len(cols))
        S = len(weights_by_scenario)
        for a in range(0, len(cols), chunk):
            out[a:a + chunk] = weights_by_scenario @ self.gross(cols[a:a + chunk]) / S
        return out


def _solve_universe(bank: _ScenarioBank, cand: np.ndarray, base: np.ndarray, budget: float,
                    cfg: ConvictionConfig) -> tuple[np.ndarray, SimplexSolution, np.ndarray, dict]:
    """Optimo sobre TODOS los candidatos por generacion de columnas.

    Devuelve (columnas activas, solucion, matriz de las activas, info). Si
    `allow_cash`, la ultima columna de la matriz es el efectivo.
    """
    gamma, S = float(cfg.risk_aversion), bank.n_scenarios
    plain_log = _is_log(gamma) and not base.any() and abs(budget - 1.0) < 1e-12
    ones = np.ones(S)
    means = bank.scan(cand, ones)
    active = list(cand[np.argsort(-means)[:max(1, int(_INITIAL_ACTIVE))]])
    cache: dict[int, np.ndarray] = {}

    def _matrix(cols):
        missing = [c for c in cols if c not in cache]
        if missing:
            block = bank.gross(missing)
            for k, c in enumerate(missing):
                cache[c] = block[:, k]
        M = np.column_stack([cache[c] for c in cols])
        return np.column_stack([M, ones]) if cfg.allow_cash else M

    rounds = 0
    while True:
        rounds += 1
        Xa = _matrix(active)
        sol = solve_simplex(Xa, gamma, budget, base, cfg.tol, cfg.max_iter)
        if gamma <= 1e-12:
            break                       # el vertice de mayor media ya esta entre los activos
        marg = (base + Xa @ sol.weights) ** (-gamma)
        g_all = bank.scan(cand, marg)
        gbar = float(sol.weights @ sol.gradient) / budget
        inside = np.isin(cand, active)
        viol = np.where(~inside & (g_all > gbar * (1.0 + 1e-7)))[0]
        if len(viol) == 0 or rounds >= _MAX_ROUNDS:
            g_max = max(float(g_all.max()), float(sol.gradient.max()))
            sol.gap = (float(np.log(max(g_max / gbar, 1.0))) if plain_log
                       else float(budget * max(g_max - gbar, 0.0)))
            sol.converged = sol.converged and len(viol) == 0
            break
        order = viol[np.argsort(-g_all[viol])[:max(1, int(_ADD_PER_ROUND))]]
        active += list(cand[order])
    return np.asarray(active, dtype=int), sol, Xa, {"rounds": rounds, "n_active": len(active)}


# ------------------------------------------------------------------- interfaz
def _empty(budget: float, notes: list[str]) -> ConvictionResult:
    return ConvictionResult(pd.Series(dtype=float), max(budget, 0.0),
                            pd.DataFrame(columns=["symbol", "weight", "score", "mu", "sd",
                                                  "p_success"]),
                            0.0, 0.0, notes=list(notes))


def _percentiles(scores: pd.Series, held=()) -> pd.Series:
    """Percentil de puntuacion de hoy en (0, 1); las mantenidas sin puntuacion, NaN."""
    pct = (scores.rank(method="average") - 0.5) / len(scores)
    universe = list(scores.index) + [s for s in held if s not in scores.index]
    return pct.reindex(universe)


def build_bank(oos: pd.DataFrame, scores: pd.Series, prices: pd.DataFrame, horizon: int,
               cfg: ConvictionConfig | None = None, costs: pd.Series | None = None,
               seed_offset: int = 0, held=()) -> _ScenarioBank:
    """Banco de escenarios predictivos para `horizon`, reutilizable entre llamadas.

    `held`: simbolos mantenidos que necesitan escenarios aunque hoy no puntuen.
    """
    cfg = cfg or ConvictionConfig()
    scores = pd.Series(scores, dtype=float).dropna()
    if scores.empty:
        raise ValueError("sin puntuaciones para hoy")
    rng = np.random.default_rng(cfg.seed + int(seed_offset))
    return _ScenarioBank(oos, _percentiles(scores, held), prices, horizon, cfg, rng,
                         costs=costs)


def allocate(oos: pd.DataFrame, scores: pd.Series, prices: pd.DataFrame, horizon: int,
             cfg: ConvictionConfig | None = None, *, held: dict | None = None,
             exclude=(), top_k: int = 5, success_threshold: float = 0.05,
             costs: pd.Series | None = None,
             bank: _ScenarioBank | None = None) -> ConvictionResult:
    """Cartera optima sobre el simplex completo para la fecha mas reciente.

    oos      predicciones walk-forward fuera de muestra: date, score, fwd. Solo
             deben contener ventanas de retorno YA cerradas.
    scores   puntuacion de hoy por simbolo (cualquier escala: se usa su percentil).
    prices   precios ajustados (fecha x simbolo) hasta hoy inclusive.
    held     {simbolo: peso} de posiciones que se mantienen y no se tocan. El
             presupuesto libre, 1 - sum(held), se asigna teniendo en cuenta su
             dependencia con ellas.
    exclude  simbolos que no pueden comprarse en esta decision (no operables,
             vetados por stop, etc.).
    costs    coste de ida y vuelta por simbolo (fraccion), descontado de cada escenario.
    bank     banco ya construido con `build_bank` para este horizonte (se reutiliza;
             entonces `oos`, `prices` y `costs` no se vuelven a leer).
    """
    cfg = cfg or ConvictionConfig()
    rng = np.random.default_rng(cfg.seed)
    gamma = float(cfg.risk_aversion)
    held = {s: float(w) for s, w in dict(held or {}).items() if float(w) > 0}
    budget = 1.0 - sum(held.values())
    if budget < max(cfg.min_weight, 1e-9):
        return _empty(budget, ["sin presupuesto libre: las posiciones mantenidas ocupan "
                               "el capital"])
    scores = pd.Series(scores, dtype=float).dropna()
    if scores.empty:
        return _empty(budget, ["sin puntuaciones para hoy"])

    if bank is None:
        bank = _ScenarioBank(oos, _percentiles(scores, held), prices, horizon, cfg, rng,
                             costs=costs)
    notes = list(bank.notes)
    S = bank.n_scenarios

    base = np.zeros(S)
    for s, w in held.items():
        if s in bank.col:
            base += w * bank.gross([bank.col[s]])[:, 0]
        else:
            base += w                       # sin historia utilizable: valor estable
            notes.append(f"{s} mantenida sin historia utilizable: tratada como efectivo")

    blocked = set(exclude) | set(held)
    cand = np.array([bank.col[s] for s in scores.index
                     if s in bank.col and s not in blocked], dtype=int)
    if len(cand) == 0:
        return _empty(budget, notes + ["ningun candidato asignable"])

    active, sol, Xa, info = _solve_universe(bank, cand, base, budget, cfg)
    w = sol.weights.copy()
    gap_free, u_free = float(sol.gap), float(sol.utility)

    # --- minimo operativo: lo que no llega al tamano de orden se reasigna ---
    n_sym = len(active)
    for _ in range(n_sym + 1):
        ws = w[:n_sym]
        small = (ws > 0) & (ws < cfg.min_weight)
        if not small.any():
            break
        keep = ws >= cfg.min_weight
        if not keep.any():
            if cfg.allow_cash:
                w[:n_sym] = 0.0
                w[-1] = budget
                notes.append("ninguna posicion llega al minimo de orden: todo en efectivo")
                break
            # Sin efectivo los pesos suman `budget`: entre los B/min mayores
            # alguno supera el minimo, asi que la siguiente vuelta avanza.
            top = np.argsort(-ws)[:max(1, int(budget / cfg.min_weight))]
            keep = np.zeros(n_sym, dtype=bool)
            keep[top] = True
        cols = np.where(keep)[0]
        Xk = Xa[:, cols] if not cfg.allow_cash else np.column_stack([Xa[:, cols], Xa[:, -1]])
        sub = solve_simplex(Xk, gamma, budget, base, cfg.tol, cfg.max_iter)
        w = np.zeros_like(w)
        w[cols] = sub.weights[:len(cols)]
        if cfg.allow_cash:
            w[-1] = sub.weights[-1]
    wealth = base + Xa @ w
    u_final = _utility(wealth, gamma)
    if u_free - u_final > 10 * cfg.tol:
        notes.append("el minimo de orden ha eliminado posiciones pequenas "
                     f"(coste en utilidad: {u_free - u_final:.2e})")

    names = [bank.symbols[c] for c in active]
    cash = float(w[-1]) if cfg.allow_cash else 0.0
    ws = pd.Series(w[:n_sym], index=names)
    ws = ws[ws > 0].sort_values(ascending=False)

    rows = []
    for s in ws.index:
        x = Xa[:, names.index(s)]
        rows.append({"symbol": s, "weight": float(ws[s]),
                     "score": float(scores.get(s, np.nan)),
                     "mu": float(x.mean() - 1.0), "sd": float(x.std()),
                     "p_success": float((x - 1.0 > success_threshold).mean())})
    table = pd.DataFrame(rows, columns=["symbol", "weight", "score", "mu", "sd", "p_success"])

    # --- alternativas, medidas con la MISMA utilidad y los MISMOS escenarios ---
    def _ce(vec):
        return certainty_equivalent(vec, gamma)

    single = [_ce(base + budget * Xa[:, k]) for k in range(n_sym)]
    k_best = int(np.argmax(single))
    top_syms = [s for s in scores.sort_values(ascending=False).index
                if s in bank.col and s not in blocked][:max(int(top_k), 1)]
    Xt = bank.gross([bank.col[s] for s in top_syms])
    ce_opt = _ce(wealth)
    alternatives = {
        "optimo": ce_opt,
        f"todo_en_{names[k_best]}": float(single[k_best]),
        f"equiponderar_top{len(top_syms)}_por_puntuacion": _ce(base + budget * Xt.mean(axis=1)),
    }

    invested = float(ws.sum())
    eff_n = float(invested ** 2 / (ws ** 2).sum()) if invested > 0 else 0.0
    info.update(bank.info)
    info.update({
        "risk_aversion": gamma, "allow_cash": bool(cfg.allow_cash),
        "budget": round(budget, 6), "n_candidates": int(len(cand)),
        "n_scenarios": int(S), "lookback_sessions": int(bank.lookback),
        "n_positions": int(len(ws)), "effective_positions": round(eff_n, 3),
        "top_weight": round(float(ws.iloc[0]) if len(ws) else 0.0, 6),
        "gap_before_min_order": gap_free, "converged": bool(sol.converged),
        "n_map_dates": int(bank.n_map_dates), "n_map_bins": int(bank.n_bins),
        "tau": cfg.tau if cfg.shrink else "none",
        # fraccion del efecto de cada tramo que sobrevive al encogimiento
        "shrinkage": [round(float(x), 4) for x in np.atleast_1d(bank.shrink)],
        # error de Monte Carlo del crecimiento: sd(ln riqueza) / sqrt(escenarios)
        "log_wealth_se": float(np.log(np.maximum(wealth, 1e-12)).std() / np.sqrt(S)),
    })
    if not sol.converged:
        notes.append(f"el optimizador no alcanzo la tolerancia (cota {gap_free:.2e})")
    return ConvictionResult(ws, cash, table, ce_opt, gap_free, alternatives, info, notes)
