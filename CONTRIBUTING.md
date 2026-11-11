# Contributing

Thanks for helping make the research easier to inspect and reproduce.
Documentation, bug reports, regression tests and negative results are useful
contributions.

## Development

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
python -m pytest -q
```

Use the existing `src/` layout and pytest conventions. Keep optional ML and
broker integrations optional. Explain any new dependency and the concrete need
it addresses.

## Reporting a bug or proposing a change

For bugs, include the command, Python and package versions, expected behavior
and a minimal example using synthetic or shareable data. Remove credentials,
account information and local paths from logs before sharing them.

For model or statistical changes, state the hypothesis, data provenance,
information available at each date, evaluation split and baseline. Report the
number of comparisons, seeds and limitations. Separate synthetic results from
real-data evidence and preserve the untouched holdout.

Keep pull requests focused. Include the relevant verification and update the
documentation when public behavior changes. Do not weaken statistical gates to
make a particular backtest pass.

## Security

Report security issues privately using the process in [SECURITY.md](SECURITY.md).
Do not post API keys, tokens, broker account details or exploitable private
configuration in issues or pull requests.
