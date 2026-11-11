# Swarmtrader · Open-source overview

**Reproducible equity-ranking research with explicit statistical evidence.**

[Quick start](../README.md#instalación) · [Architecture](ARCHITECTURE.md) ·
[Audit](AUDITORIA_GEOMETRIA.md) · [Current status](ESTADO.json) ·
[Contributing](../CONTRIBUTING.md)

## The problem

Testing many models and selecting the most attractive backtest can manufacture
apparent skill. Adding agents also adds hypotheses. A useful research tool must
measure that selection effect, compare against simple baselines and report when
the evidence does not support action.

Swarmtrader makes those checks part of an inspectable Python workflow. The goal
is to help researchers and students examine what a ranking model can support
with the available data.

## What is available

| Component | Purpose | Entry point |
|---|---|---|
| Single-model ranking | Walk-forward evaluation and calibrated probability estimates | [`rank.py`](../src/swarmtrader/rank.py) |
| Specialist ensemble | Compare agents, test learned aggregation and allow abstention | [`swarm_rank.py`](../src/swarmtrader/swarm_rank.py) |
| Statistical controls | Hansen SPA, Deflated Sharpe Ratio and Probability of Backtest Overfitting | [`stats/`](../src/swarmtrader/stats/) |
| Temporal data handling | Availability lags, point-in-time constraints and delisting handling | [`data/pit.py`](../src/swarmtrader/data/pit.py) |
| Information geometry | Closed-form Fisher–Rao distances and drift experiments | [`stats/infogeo.py`](../src/swarmtrader/stats/infogeo.py) |
| Reproduction materials | Synthetic controls, unit tests and saved research outputs | [`tests/`](../tests/), [`experiments/`](../experiments/), [`results/`](results/) |

## Try it locally

Python 3.11 or newer is required. The default model uses NumPy, pandas, SciPy
and joblib; larger ML libraries are optional extras.

```bash
git clone https://github.com/dalonsogomez-ai/swarmtrader
cd swarmtrader
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m swarmtrader.cli rank examples/history_long_example.csv --top 3
python -m pytest -q
python scripts/swarm_null_test.py
```

The bundled price examples and synthetic controls run without API keys, broker
accounts or external data downloads. Optional adapters have separate data,
licensing and credential requirements.

## Evidence and current limits

The local publication check on 11 November 2026 passed **216 tests**, with
**6 skipped** and **6 pandas compatibility warnings**, on Python 3.13.13.
The [CI matrix](../.github/workflows/ci.yml) runs on Linux and macOS with Python
3.11 and 3.12. Optional backends are not all exercised by that matrix.

The saved real-data audit uses a survivor-only universe. In that audit, the
top-ranked stocks did not outperform the universe, and the G4 portfolio check
produces **NO ACCIONABLE**. The information-geometry experiments did not establish
an advantage over the Euclidean baseline at the main horizon.

Those saved experiments are prior project evidence; the underlying real-data
panel is not bundled, and the full real-data audit was not rerun for publication.
The publication fixes include a correction to drifting portfolio weights;
historical outputs that depend on the simulator need to be recalculated. See
[publication notes](PUBLICATION_NOTES.md) for scope and verification.
No profitable live performance, external adoption, download volume or downstream
dependents are claimed. External broker paper trading remains unvalidated.

## Useful contributions

Current priorities follow the [documented project status](ESTADO.json):

1. Validate with point-in-time universes that include delisted securities.
2. Improve instrument identity across venues and trading-calendar assumptions.
3. Reproduce statistical controls, compare simple baselines and report failures.

The project is MIT-licensed. Small, reviewable contributions with reproducible
evidence are welcome; see [CONTRIBUTING.md](../CONTRIBUTING.md).
