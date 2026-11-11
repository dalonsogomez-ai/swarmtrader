"""Motor del torneo evolutivo con validacion walk-forward.

Invariante central: cada generacion evalua a la poblacion en una ventana
temporal NUEVA, posterior y NO solapada con las anteriores, separada por un
embargo. Ningun agente es seleccionado sobre los mismos datos con los que sera
juzgado despues, y ningun retorno entra dos veces en su historial.

El meta-modelo de cada agente se ajusta EXCLUSIVAMENTE con picks de ventanas
anteriores a la que se evalua.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field

import numpy as np

from ..agents.genome import Genome
from ..backtest import BacktestResult, FeaturePanel, simulate
from ..data.baselines import run_baselines
from ..stats.deflated_sharpe import deflated_sharpe_ratio, effective_n_trials, sharpe_ratio
from ..stats.overfitting import effective_n_trials_clustering, probability_of_backtest_overfitting
from ..stats.registry import TrialRecord, TrialRegistry
from .fitness import FitnessConfig, compute_fitness
from .meta import MetaLabeler
from .selection import AgentState, DeathConfig, apply_deaths, breed


@dataclass
class SwarmConfig:
    population_size: int = 40
    generations: int = 20
    warmup_days: int = 300
    eval_window: int = 126
    window_step: int | None = None   # None => no solapado (= eval_window + embargo)
    embargo_days: int = 5             # hueco entre ventanas: evita filtrado de informacion
    initial_capital: float = 50.0
    cost_bps: float = 10.0
    impact_coef: float = 0.0
    use_meta: bool = True
    elite_fraction: float = 0.2
    mutation_rate: float = 0.3
    n_jobs: int = 1                   # >1 usa joblib/loky; -1 = todos los nucleos
    seed: int = 42
    dsr_threshold: float = 0.95
    pbo_threshold: float = 0.20       # PBO maximo aceptable para certificar
    holdout_days: int = 252           # tramo final NUNCA visto por el torneo: examen final
    holdout_psr_threshold: float = 0.90
    min_generations_for_champion: int = 3
    fitness: FitnessConfig = field(default_factory=FitnessConfig)
    death: DeathConfig = field(default_factory=DeathConfig)

    def step(self) -> int:
        return self.window_step if self.window_step else self.eval_window + self.embargo_days


@dataclass
class GenerationReport:
    generation: int
    window: tuple[str, str]
    n_alive: int
    n_deaths: int
    best_fitness: float
    median_fitness: float
    best_sharpe: float
    mean_pct_cash: float
    baseline_sharpe: float           # momentum+calidad top-20 en la misma ventana
    pct_beating_baseline: float
    causes: dict[str, int]


def _evaluate_one(panel: FeaturePanel, genome: Genome, meta: MetaLabeler | None,
                  start: int, end: int, cost_bps: float, capital: float,
                  impact_coef: float) -> BacktestResult:
    return simulate(panel, genome, start, end, cost_bps=cost_bps, capital=capital,
                    impact_coef=impact_coef, meta=meta, record_picks=True)


class SwarmEngine:
    def __init__(self, panel: FeaturePanel, config: SwarmConfig, registry: TrialRegistry):
        self.panel = panel
        self.cfg = config
        self.registry = registry
        self.rng = np.random.default_rng(config.seed)
        self.run_id = uuid.uuid4().hex[:12]
        self.population: list[AgentState] = [
            AgentState(genome=Genome.random(self.rng, panel.factor_names), capital=config.initial_capital)
            for _ in range(config.population_size)
        ]
        self.graveyard: list[AgentState] = []
        self.reports: list[GenerationReport] = []
        self.windows: list[tuple[int, int]] = []
        self.baseline_history: dict[str, list[float]] = {}

    # ---------------------------------------------------------------- ventanas
    @property
    def holdout_start(self) -> int:
        return max(self.cfg.warmup_days, len(self.panel.dates) - self.cfg.holdout_days)

    def _window(self, generation: int) -> tuple[int, int]:
        start = self.cfg.warmup_days + generation * self.cfg.step()
        end = min(start + self.cfg.eval_window, self.holdout_start - self.cfg.embargo_days)
        return start, end

    def max_generations(self) -> int:
        g = 0
        while True:
            s, e = self._window(g)
            if e - s < self.cfg.eval_window // 2:
                return g
            g += 1

    # ------------------------------------------------------------- evaluacion
    def _fit_meta(self, agent: AgentState) -> MetaLabeler | None:
        if not self.cfg.use_meta:
            return None
        m = MetaLabeler().fit(agent.picks)   # solo picks de ventanas ANTERIORES
        agent.meta = m
        return m if m.fitted else None

    def _evaluate_population(self, alive: list[AgentState], generation: int,
                             start: int, end: int) -> None:
        metas = [self._fit_meta(a) for a in alive]
        args = [(self.panel, a.genome, m, start, end, self.cfg.cost_bps,
                 self.cfg.initial_capital, self.cfg.impact_coef) for a, m in zip(alive, metas)]
        if self.cfg.n_jobs == 1 or len(alive) < 4:
            results = [_evaluate_one(*a) for a in args]
        else:
            from joblib import Parallel, delayed
            results = Parallel(n_jobs=self.cfg.n_jobs, backend="loky")(
                delayed(_evaluate_one)(*a) for a in args)

        for agent, res in zip(alive, results):
            fit = compute_fitness(res, self.cfg.fitness)
            agent.returns_history.extend(res.returns.tolist())
            agent.fitness_history.append(fit)
            agent.capital *= (1.0 + res.total_return)
            agent.generations_alive += 1
            agent.last_result = res
            agent.picks.extend(res.picks)      # disponibles para el meta de la SIGUIENTE ventana
            self.registry.log(TrialRecord(
                run_id=self.run_id, generation=generation, agent_id=agent.genome.agent_id,
                genome=agent.genome.to_dict(),
                window_start=str(self.panel.dates[start].date()),
                window_end=str(self.panel.dates[end - 1].date()),
                sharpe=res.sharpe, total_return=res.total_return, max_dd=res.max_dd,
                turnover=res.turnover, returns=res.returns,
            ))

    # ------------------------------------------------------------------- loop
    def run(self, generations: int | None = None, verbose: bool = True) -> list[GenerationReport]:
        n_gen = min(generations or self.cfg.generations, self.max_generations())
        if n_gen <= 0:
            raise ValueError("no hay datos suficientes para una sola generacion")

        for g in range(n_gen):
            start, end = self._window(g)
            self.windows.append((start, end))
            alive = [a for a in self.population if a.alive]
            self._evaluate_population(alive, g, start, end)

            bases = run_baselines(self.panel, start, end, self.cfg.cost_bps)
            for k, v in bases.items():
                self.baseline_history.setdefault(k, []).extend(v.returns.tolist())
            base_sr = bases["BASE_momq_top20"].sharpe

            dead = apply_deaths(self.population, self.cfg.death, self.cfg.initial_capital)
            self.graveyard.extend(dead)
            alive = [a for a in self.population if a.alive]
            if not alive:
                alive = [AgentState(genome=Genome.random(self.rng, self.panel.factor_names), capital=self.cfg.initial_capital)
                         for _ in range(self.cfg.population_size // 2)]

            n_elite = max(1, int(self.cfg.elite_fraction * self.cfg.population_size))
            elite = sorted(alive, key=lambda a: a.mean_fitness, reverse=True)[:n_elite]
            children = breed(alive, self.cfg.population_size - len(elite), self.rng,
                             self.cfg.initial_capital, mutation_rate=self.cfg.mutation_rate)
            self.population = elite + children

            fits = [a.fitness for a in alive]
            srs = [a.last_result.sharpe for a in alive if a.last_result is not None]
            causes: dict[str, int] = {}
            for d in dead:
                causes[d.cause_of_death or "?"] = causes.get(d.cause_of_death or "?", 0) + 1
            rep = GenerationReport(
                generation=g,
                window=(str(self.panel.dates[start].date()), str(self.panel.dates[end - 1].date())),
                n_alive=len(alive), n_deaths=len(dead),
                best_fitness=float(max(fits)) if fits else -1.0,
                median_fitness=float(np.median(fits)) if fits else -1.0,
                best_sharpe=float(max(srs)) if srs else 0.0,
                mean_pct_cash=float(np.mean([a.last_result.pct_days_in_cash for a in alive
                                             if a.last_result is not None])) if alive else 1.0,
                baseline_sharpe=float(base_sr),
                pct_beating_baseline=float(np.mean([s > base_sr for s in srs])) if srs else 0.0,
                causes=causes,
            )
            self.reports.append(rep)
            if verbose:
                print(f"gen {g:02d} [{rep.window[0]}..{rep.window[1]}] vivos={rep.n_alive:3d} "
                      f"muertes={rep.n_deaths:3d} fit_best={rep.best_fitness:+.2f} "
                      f"sr_best={rep.best_sharpe:+.2f} sr_base={rep.baseline_sharpe:+.2f} "
                      f"baten_base={rep.pct_beating_baseline:.0%} cash={rep.mean_pct_cash:.0%}")
        return self.reports

    # ---------------------------------------------------------------- campeon
    def champion(self) -> AgentState | None:
        cands = [a for a in self.population
                 if a.alive and a.generations_alive >= self.cfg.min_generations_for_champion]
        if not cands:
            cands = [a for a in self.population if a.alive and a.fitness_history]
        if not cands:
            return None
        return max(cands, key=lambda a: (a.mean_fitness, a.edge_probability()))

    def _holdout_exam(self, agent: AgentState) -> dict:
        """Una sola evaluacion sobre el tramo reservado. N=1: el DSR se reduce al PSR.

        Esto resuelve la tension estructural del torneo: buscar exige muchos
        trials y muchos trials hunden el DSR. La solucion es separar exploracion
        (torneo, N grande, DSR/PBO) de confirmacion (holdout, N=1, PSR). Si el
        campeon rinde en el holdout, ese resultado no esta contaminado por la
        busqueda porque el torneo nunca lo vio.
        """
        from scipy import stats as _st
        from ..stats.deflated_sharpe import probabilistic_sharpe_ratio
        hs, he = self.holdout_start, len(self.panel.dates)
        if he - hs < 60:
            return {"available": False, "psr": 0.0, "sharpe": 0.0, "baseline_sharpe": 0.0,
                    "beats_baseline": False, "days": he - hs}
        meta = agent.meta if (agent.meta is not None and agent.meta.fitted) else None
        res = simulate(self.panel, agent.genome, hs, he, cost_bps=self.cfg.cost_bps,
                       capital=self.cfg.initial_capital, impact_coef=self.cfg.impact_coef,
                       meta=meta, record_picks=False)
        bases = run_baselines(self.panel, hs, he, self.cfg.cost_bps)
        r = res.returns
        sd = r.std(ddof=1) if r.size > 2 else 0.0
        psr = probabilistic_sharpe_ratio(r.mean() / sd if sd > 1e-12 else 0.0, r.size,
                                         float(_st.skew(r)), float(_st.kurtosis(r, fisher=False))) if sd > 1e-12 else 0.0
        b_sr = bases["BASE_momq_top20"].sharpe
        return {"available": True, "days": he - hs,
                "window": (str(self.panel.dates[hs].date()), str(self.panel.dates[he - 1].date())),
                "sharpe": round(res.sharpe, 3), "total_return": round(res.total_return, 4),
                "max_dd": round(res.max_dd, 4), "pct_cash": round(res.pct_days_in_cash, 3),
                "psr": round(psr, 4), "baseline_sharpe": round(b_sr, 3),
                "ew_bh_sharpe": round(bases["BASE_ew_bh"].sharpe, 3),
                "beats_baseline": bool(res.sharpe > b_sr)}

    def certify(self, agent: AgentState) -> dict:
        """Veredicto final. Tres gates independientes, los tres deben pasar.

        1. DSR >= umbral       : el Sharpe sobrevive al numero de pruebas
        2. PBO <= umbral       : elegir el mejor IS generaliza OOS
        3. bate al baseline    : ha anadido algo sobre momentum+calidad sin IA
        """
        r = np.asarray(agent.returns_history, dtype=float)
        n_raw = self.registry.n_trials(self.run_id)
        matrix = self.registry.returns_matrix(self.run_id)
        n_spec = effective_n_trials(matrix) if matrix.size else max(1, n_raw)
        n_clus = effective_n_trials_clustering(matrix) if matrix.size else max(1, n_raw)
        n_eff = max(n_spec, n_clus)                     # el mas conservador
        var_sr = self.registry.sharpe_variance(self.run_id)
        dsr = deflated_sharpe_ratio(r, n_eff, var_sr)
        pbo = probability_of_backtest_overfitting(matrix) if matrix.size else None

        base = self.baseline_history.get("BASE_momq_top20", [])
        base_sr = sharpe_ratio(np.asarray(base)) if base else 0.0
        agent_sr = sharpe_ratio(r)

        # 4. EXAMEN FINAL: holdout nunca visto, evaluado UNA sola vez (N=1 => PSR)
        holdout = self._holdout_exam(agent)

        gate_dsr = dsr >= self.cfg.dsr_threshold
        gate_pbo = (pbo is not None and np.isfinite(pbo.pbo) and pbo.pbo <= self.cfg.pbo_threshold)
        gate_base = agent_sr > base_sr
        gate_hold = holdout["psr"] >= self.cfg.holdout_psr_threshold and holdout["beats_baseline"]
        certified = bool(gate_dsr and gate_pbo and gate_base and gate_hold)
        reasons = []
        if not gate_dsr:
            reasons.append(f"DSR {dsr:.3f} < {self.cfg.dsr_threshold}: compatible con azar dado N")
        if not gate_pbo:
            reasons.append(f"PBO {(pbo.pbo if pbo else float('nan')):.3f} > {self.cfg.pbo_threshold}: la seleccion no generaliza")
        if not gate_base:
            reasons.append(f"Sharpe {agent_sr:.2f} no bate al baseline sin IA ({base_sr:.2f})")
        if not gate_hold:
            reasons.append(f"holdout: PSR {holdout['psr']:.3f} / Sharpe {holdout['sharpe']:.2f} vs baseline {holdout['baseline_sharpe']:.2f}")

        return {
            "agent_id": agent.genome.agent_id,
            "generations_alive": agent.generations_alive,
            "n_trials_raw": n_raw,
            "n_trials_effective": {"spectral": n_spec, "clustering": n_clus, "used": n_eff},
            "sharpe_annualized": round(agent_sr, 3),
            "baseline_sharpe_annualized": round(base_sr, 3),
            "edge_probability": round(agent.edge_probability(), 3),
            "deflated_sharpe_ratio": round(dsr, 4),
            "pbo": pbo.summary() if pbo else None,
            "meta_model": agent.meta.to_dict() if agent.meta is not None else None,
            "holdout": holdout,
            "gates": {"dsr": gate_dsr, "pbo": gate_pbo, "beats_baseline": gate_base, "holdout": gate_hold},
            "certified": certified,
            "verdict": "APTO para paper trading" if certified else "NO APTO: " + " | ".join(reasons),
        }
