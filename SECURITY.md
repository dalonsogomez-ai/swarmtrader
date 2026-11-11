# Security policy

## Reporting a vulnerability

Use GitHub's **Report a vulnerability** action in this repository's
[Security tab](https://github.com/dalonsogomez-ai/swarmtrader/security).
Keep reports private while the issue is assessed. Include a minimal reproduction
with synthetic data and describe the affected component and impact.

Do not include live API keys, tokens, account identifiers, personal portfolios
or authentication codes. If a credential has been exposed, revoke it at the
provider and remove it from any shared material.

## Local configuration

Supply credentials through environment variables or a local secret manager.
Keep `.env` files, local data, run outputs and virtual environments outside
version control. The CSV examples and synthetic controls require no credentials.

The source includes optional broker adapters. Research tests use local data and
simulated brokers; those tests do not validate production trading integrations.
Review configuration and provider permissions before using an adapter.

The currently published research version is 1.3.0. This is a maintainer-run
project; no security-response SLA is promised.
