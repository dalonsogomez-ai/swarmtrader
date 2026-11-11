"""Geometria de la informacion: distancias de Fisher-Rao en forma cerrada.

Por que existe este modulo
--------------------------
Una distancia euclidea entre parametros no sabe nada de estadistica: tratar
igual "sigma pasa de 1% a 2%" y "sigma pasa de 10% a 11%" es un error, porque
lo primero es duplicar el riesgo y lo segundo es ruido. La metrica de Fisher
mide la separacion entre dos distribuciones en unidades de **distinguibilidad**:
cuantas desviaciones tipicas de error de estimacion las separan. Es la unica
metrica riemanniana invariante a reparametrizaciones y a estadisticos
suficientes (teorema de Chentsov), asi que el resultado no depende de si se
trabaja con sigma, con la varianza o con su logaritmo.

Solo se implementan las tres familias que el sistema usa y que tienen forma
cerrada. No hay optimizacion numerica ni geodesicas aproximadas: cada funcion
es una formula verificable contra identidades conocidas en `tests/test_infogeo.py`.

  gaussiana univariante   -> semiplano hiperbolico (mu/sqrt 2, sigma)
  Bernoulli               -> arco de circunferencia (transformacion arcoseno)
  gaussiana centrada (SPD)-> metrica afin-invariante sobre matrices de covarianza

Lo que este modulo NO es: una fuente de senal. Son reglas de medir. El valor de
usarlas en un sitio concreto se demuestra midiendo (ver `experiments/infogeo.py`),
no por el hecho de que la formula sea elegante.
"""

from __future__ import annotations

import numpy as np

__all__ = ["fisher_rao_gaussian", "kl_gaussian", "fisher_rao_bernoulli",
           "bernoulli_coordinate", "fisher_rao_spd", "shrink_to_identity"]

_SQRT2 = float(np.sqrt(2.0))


def fisher_rao_gaussian(mu1, sigma1, mu2, sigma2):
    """Distancia de Rao entre N(mu1, sigma1^2) y N(mu2, sigma2^2). Vectorizada.

    La metrica de Fisher de la gaussiana es ds^2 = (dmu^2 + 2 dsigma^2) / sigma^2,
    es decir, sqrt(2) veces la del semiplano de Poincare en (mu/sqrt 2, sigma):

        d = sqrt 2 * arccosh(1 + ((mu1-mu2)^2 / 2 + (sigma1-sigma2)^2) / (2 sigma1 sigma2))

    Dos propiedades que importan en la practica:
      * invariante a cambios de escala y origen comunes (mu -> a*mu + b,
        sigma -> a*sigma): un +2% en un valor con sigma 1% dista lo mismo de su
        referencia que un +6% en uno con sigma 3%;
      * con medias iguales se reduce a sqrt 2 * |ln(sigma1 / sigma2)|: lo que
        cuenta es el cociente de volatilidades, no su diferencia.
    """
    mu1, sigma1, mu2, sigma2 = (np.asarray(x, dtype=float) for x in (mu1, sigma1, mu2, sigma2))
    num = 0.5 * (mu1 - mu2) ** 2 + (sigma1 - sigma2) ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        arg = 1.0 + num / (2.0 * sigma1 * sigma2)
    arg = np.where(np.isfinite(arg), np.maximum(arg, 1.0), np.nan)
    return _SQRT2 * np.arccosh(arg)


def kl_gaussian(mu1, sigma1, mu2, sigma2):
    """KL(N1 || N2) en nats. Asimetrica; se incluye para contrastar la distancia
    de Rao en el limite local (d^2 ~ 2 KL cuando las distribuciones son proximas)."""
    mu1, sigma1, mu2, sigma2 = (np.asarray(x, dtype=float) for x in (mu1, sigma1, mu2, sigma2))
    return np.log(sigma2 / sigma1) + (sigma1 ** 2 + (mu1 - mu2) ** 2) / (2.0 * sigma2 ** 2) - 0.5


def bernoulli_coordinate(p):
    """Coordenada en la que la metrica de Fisher de Bernoulli(p) es euclidea.

    theta = 2 arcsin(sqrt p). Es la transformacion estabilizadora de varianza:
    un paso unitario en theta es una desviacion tipica de error muestral en
    cualquier punto de (0, 1). Aplicada a un percentil, estira las colas: pasar
    del percentil 99 al 99,9 mide mas de 7 veces lo que pasar del 50 al 50,9.
    """
    p = np.clip(np.asarray(p, dtype=float), 0.0, 1.0)
    return 2.0 * np.arcsin(np.sqrt(p))


def fisher_rao_bernoulli(p, q):
    """Distancia de Rao entre Bernoulli(p) y Bernoulli(q): 2|arcsin sqrt p - arcsin sqrt q|."""
    return np.abs(bernoulli_coordinate(p) - bernoulli_coordinate(q))


def shrink_to_identity(corr: np.ndarray, n_obs: int) -> np.ndarray:
    """Encoge una matriz de correlacion hacia la identidad: (1-d) R + d I, d = k/(k+n).

    Garantiza que sea definida positiva aunque haya menos observaciones que
    variables, que es condicion para que `fisher_rao_spd` exista. La intensidad
    es una regla fija y sin parametro libre a proposito: se aplica identica a
    todas las fechas, asi que no puede introducir una diferencia entre el
    periodo de referencia y el dia evaluado.
    """
    r = np.array(corr, dtype=float, copy=True)
    k = r.shape[-1]
    r = np.nan_to_num(r, nan=0.0, posinf=0.0, neginf=0.0)
    r = 0.5 * (r + np.swapaxes(r, -1, -2))
    idx = np.arange(k)
    r[..., idx, idx] = 1.0
    d = k / float(k + max(int(n_obs), 1))
    return (1.0 - d) * r + d * np.eye(k)


def _inv_sqrt(a: np.ndarray) -> np.ndarray:
    w, v = np.linalg.eigh(a)
    w = np.maximum(w, 1e-12)
    return (v / np.sqrt(w)) @ v.T


def fisher_rao_spd(ref: np.ndarray, mats: np.ndarray) -> np.ndarray:
    """Distancia de Rao entre N(0, ref) y N(0, mats[i]). `mats` es (k,k) o (n,k,k).

    d = sqrt(1/2 * sum_i ln^2 lambda_i), con lambda los autovalores de
    ref^{-1/2} M ref^{-1/2}. Es la metrica afin-invariante de las matrices
    definidas positivas: no cambia si las variables se reescalan o se mezclan
    linealmente, que es lo que la hace adecuada para comparar estructuras de
    dependencia. En dimension 1 coincide con `fisher_rao_gaussian` a medias
    iguales (comprobado en los tests).
    """
    ref = np.asarray(ref, dtype=float)
    m = np.asarray(mats, dtype=float)
    single = m.ndim == 2
    if single:
        m = m[None]
    s = _inv_sqrt(ref)
    w = np.linalg.eigvalsh(s @ m @ s)
    w = np.maximum(w, 1e-12)
    d = np.sqrt(0.5 * (np.log(w) ** 2).sum(axis=-1))
    return float(d[0]) if single else d
