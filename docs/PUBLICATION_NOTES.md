# Publication notes · 11 November 2026

The public import preserves the v1.3.0 source, examples and saved research
reports, with the following corrections and documentation improvements.

## Packaging and command-line interface

- Anchor the local-data ignore rule at `/data/`, so the source package
  `src/swarmtrader/data/` is included in Git.
- Register `--congress` and `--congress-window` for `rank`. Previously, the
  normal offline ranking command accessed an absent argparse attribute.
- Add an integration test of the bundled offline CSV command.
- Install the existing storage extra in CI and development instructions: the
  Parquet round-trip tests require pyarrow. Restrict CI permissions to reading
  repository contents and retain all four platform/version checks on failures.

## Portfolio accounting

Between rebalances, portfolio weights must be measured against current NAV.
The prior update multiplied weights by asset growth without dividing by
portfolio growth. This introduced fictitious exposure, especially for long
buy-and-hold periods. The update now normalizes by `1 + portfolio_return`.

Two deterministic, cost-free regression cases check fully invested and
partially invested buy-and-hold against the analytical asset-plus-cash NAV.
This is a targeted correction, not a full audit of costs, realized tax P&L,
execution or production broker integration.

Historical tournament and baseline metrics produced through the simulator
predate this correction. They must be rerun before being used as current
performance evidence. Saved audit artifacts retain their original values and
are historical project records.

## Positive-control validation

The previous `validate` command described injected momentum but generated a
different, latent signal. It also treated momentum beating buy-and-hold on a
particular price path as a correctness condition. Predictive rank information
does not guarantee that a rebalanced strategy beats a drifting buy-and-hold
portfolio after costs.

The corrected command explicitly injects observable momentum, requires the
out-of-sample ranking to detect it, and still reports net baseline results.
The null-market certification check is retained. A regression test removes the
positive-control signal and verifies that validation fails.

## Publication checks

- Local suite: **216 passed, 6 skipped**, Python 3.13.13.
- Six pandas warnings concern a future dtype-compatibility change.
- Bundled offline ranking, including external synthetic signals, executes.
- Provider-token, private-key, JWT, credential-assignment and local-contact/path
  pattern scans cover publication files and Git history; no findings were
  detected. Pattern scans do not prove the absence of every possible secret.
- GitHub Secret Scanning and Push Protection are enabled. Private vulnerability
  reporting is available through the repository Security tab.

Optional backends, live broker execution and the full external real-data panel
were not validated as part of this publication.
