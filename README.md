# TOWER

A robotics simulation benchmark for long-horizon tower manipulation.

Read the [documentation](docs/index.md), start as a
[benchmark user](docs/getting-started/quickstart-benchmark-user.md), or connect a
[policy](docs/getting-started/quickstart-policy-author.md).

Public benchmark revision 2026.11.1 uses a redacted scene representation with
unchanged physical scenes. See [versioning](docs/reference/versioning.md).
Isaac 6.1 provides benchmark physics; CPU smoke tests only check integration.

Code is Apache-2.0; see [LICENSE](LICENSE), [NOTICE](NOTICE), and
[asset terms](LICENSE-ASSETS). Cite TOWER using [CITATION.cff](CITATION.cff).
See [CONTRIBUTING.md](CONTRIBUTING.md) for development.

## CPU installation check

```sh
uv sync --locked
uv run --package tower-eval tower smoke
```

This smoke check does not evaluate benchmark physics.
