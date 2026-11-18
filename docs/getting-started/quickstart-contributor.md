# Contributor quickstart

```sh
GIT_LFS_SKIP_SMUDGE=1 git submodule update --init --depth 1 third_party/XPolicyLab
uv sync --locked --all-packages
uv run python -m scripts.maintainer.fetch_golden
uv run python -m scripts.maintainer.fetch_public_test_data
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run pytest -ra -n auto
uv run python -m scripts.maintainer.check_public_tree
```

The XPolicyLab integration tests use the pinned checkout. Skipping Git LFS
downloads keeps setup limited to the code those tests need.

These checks use public hash-pinned recordings, scenes and assets and require no
GPU or private archive. Native IK checks also run in public CI. Offline input
selection and the remaining maintainer checks are described in
[CONTRIBUTING](../../CONTRIBUTING.md). Missing optional native dependencies or
public inputs are reported as skips, never passes.
Follow [development setup](../contributing/dev-setup.md) to build documentation.
Before changing task or scoring semantics, describe the proposed behavior in an
issue and include regression evidence. Follow the [control safety contract](../guides/eef-control.md).
