"""Reglas de supervivencia y reproduccion.

Tres modos de muerte, configurables:

  bayesian (DEFECTO, recomendado)
      Muere cuando P(edge > 0 | historial) cae por debajo del umbral, tras un
      minimo de observaciones. Mata al afortunado-pero-ruidoso y perdona al
      desafortunado-pero-solido. Es la unica de las tres que selecciona por
      edge en vez de por varianza.

  drawdown
      Muere al superar un drawdown maximo. Util como capa adicional de riesgo.

  capital  (el diseno "sobrevive con 50$")
      Muere al agotar el capital. Se incluye para que puedas MEDIR lo malo que
      es: correlaciona con varianza, no con habilidad, y genera exactamente el
      sesgo de supervivencia que el DSR existe para detectar. No lo uses como
      unico criterio.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .fitness import posterior_edge_probability


@dataclass
class AgentState:
    genome: object
    capital: float
    returns_history: list[float] = field(default_factory=list)
    fitness_history: list[float] = field(default_factory=list)
    generations_alive: int = 0
    alive: bool = True
    cause_of_death: str | None = None
    picks: list = field(default_factory=list)       # PickRecord acumulados (pasado)
    meta: object | None = None                        # MetaLabeler ajustado sobre picks
    last_result: object | None = None

    @property
    def fitness(self) -> float:
        return self.fitness_history[-1] if self.fitness_history else -1.0

    @property
    def mean_fitness(self) -> float:
        return float(np.mean(self.fitness_history)) if self.fitness_history else -1.0

    def edge_probability(self, prior_sharpe_sd: float = 0.5) -> float:
        return posterior_edge_probability(np.asarray(self.returns_history), prior_sharpe_sd)


@dataclass(frozen=True)
class DeathConfig:
    mode: str = "bayesian"             # bayesian | drawdown | capital | any
    edge_threshold: float = 0.15       # muere si P(edge>0) < esto
    min_observations: int = 60         # no matar antes de tener evidencia
    max_drawdown: float = 0.35
    capital_floor: float = 0.25        # fraccion del capital inicial
    grace_generations: int = 2


def apply_deaths(population: list[AgentState], cfg: DeathConfig, initial_capital: float) -> list[AgentState]:
    """Marca agentes muertos in place y devuelve la lista de fallecidos."""
    dead: list[AgentState] = []
    for a in population:
        if not a.alive or a.generations_alive < cfg.grace_generations:
            continue
        cause = None
        r = np.asarray(a.returns_history, dtype=float)

        if cfg.mode in ("bayesian", "any") and r.size >= cfg.min_observations:
            if a.edge_probability() < cfg.edge_threshold:
                cause = "sin_edge_bayesiano"

        if cause is None and cfg.mode in ("drawdown", "any") and r.size >= 5:
            eq = np.cumprod(1.0 + r)
            peak = np.maximum.accumulate(eq)
            dd = float(np.max((peak - eq) / np.where(peak > 0, peak, 1.0)))
            if dd > cfg.max_drawdown:
                cause = "drawdown"

        if cause is None and cfg.mode in ("capital", "any"):
            if a.capital <= cfg.capital_floor * initial_capital:
                cause = "capital_agotado"

        if cause:
            a.alive = False
            a.cause_of_death = cause
            dead.append(a)
    return dead


def tournament_select(
    population: list[AgentState],
    rng: np.random.Generator,
    k: int = 3,
) -> AgentState:
    """Seleccion por torneo sobre fitness MEDIO, no sobre el ultimo periodo.

    Usar el fitness medio historico en vez del ultimo reduce la probabilidad de
    promover a un agente que simplemente tuvo un buen mes.
    """
    alive = [a for a in population if a.alive]
    if not alive:
        raise RuntimeError("poblacion extinta")
    idx = rng.choice(len(alive), size=min(k, len(alive)), replace=False)
    return max((alive[i] for i in idx), key=lambda a: a.mean_fitness)


def breed(
    population: list[AgentState],
    n_offspring: int,
    rng: np.random.Generator,
    initial_capital: float,
    mutation_rate: float = 0.3,
    crossover_prob: float = 0.6,
    max_inherited_picks: int = 400,
) -> list[AgentState]:
    children: list[AgentState] = []
    for _ in range(n_offspring):
        p1 = tournament_select(population, rng)
        inherited = list(p1.picks)
        if rng.random() < crossover_prob:
            p2 = tournament_select(population, rng)
            g = p1.genome.crossover(p2.genome, rng).mutate(rng, rate=mutation_rate * 0.5)
            inherited += list(p2.picks)
        else:
            g = p1.genome.mutate(rng, rate=mutation_rate)
        # el hijo hereda la experiencia de sus padres: su funcion de scoring es
        # parecida, asi que los picks pasados siguen siendo informativos para el
        # meta-modelo. Se limita para que los linajes antiguos no dominen.
        children.append(AgentState(genome=g, capital=initial_capital,
                                   picks=inherited[-max_inherited_picks:]))
    return children
