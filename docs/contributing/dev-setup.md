# Development setup

Use the locked workspace and checks in the [contributor quickstart](../getting-started/quickstart-contributor.md).
Keep library diagnostics in named loggers and public APIs typed. Run
`uv run pre-commit install` to enable the repository hooks.

Build documentation in an isolated environment, separate from the workspace:

```sh
python3 -m venv .venv-docs
.venv-docs/bin/python -m pip install -r docs/requirements.txt
.venv-docs/bin/python -m mkdocs build --strict
python3 tests/docs/test_internal_links.py
```

The link check uses only Python's standard library and also runs under pytest.
Documentation CI runs the link check and strict build on pull requests in the
public repository. Remove the temporary environment when finished.

Before building wheels, set `export PYTHONDONTWRITEBYTECODE=1`. Benchmark
entry points disable bytecode writes too, but wheel builds run separately.
See the [runtime build instructions](../reference/bench-public.md) for the three
wheels and release manifest required before preparing benchmark requests.
