# Contributing

Start with the [contributor quickstart](docs/getting-started/quickstart-contributor.md),
[development guide](docs/contributing/dev-setup.md) and
[control safety contract](docs/guides/eef-control.md#control-safety-contract).

Every contributor runs:

```bash
GIT_LFS_SKIP_SMUDGE=1 git submodule update --init --depth 1 third_party/XPolicyLab
uv sync --locked --all-packages
uv run python -m scripts.maintainer.fetch_golden
uv run python -m scripts.maintainer.fetch_public_test_data
uv run ruff check .
uv run ruff format --check .
uv run pytest -ra -n auto
uv run python -m scripts.maintainer.check_public_tree
```

Golden recordings are public and pinned by SHA-256, size and an immutable dataset
commit. For offline work, set `TOWER_GOLDEN_ROOT` or pass `pytest --golden-root`
with either the golden recording directory or the complete staged asset root.
Missing data is an explicit skip with a reason; it is never evidence of a pass.
An explicit but incomplete or changed local copy is an error. Public CI fetches
all recordings before testing and requires public data, so unavailable inputs fail CI.

Use `TOWER_ASSET_ROOT` or `pytest --asset-root` for a local public asset folder.
The scene suite checks all 600 public scenes and their actual asset hashes.

Public CI runs the CPU suite on Python 3.11 and 3.12, golden scoring checks,
NumPy characterization, native IK jobs, Ruff, typing, package builds, and leak
checks over the repository and built wheels. These checks need no private
archive. Archive-marked checks are the one-time comparisons with frozen source,
v1 scenes, private expert commands or the reference GPU recordings.

Maintainers run the exact byte tier in its archived Linux numerical environment
and the remaining migration checks with `pytest --archive-root="$TOWER_ARCHIVE_ROOT"`.
The [maintainer release guide](docs/contributing/release.md) explains how to fetch
the original assets for the remaining private checks. An explicit archive run
must report zero archive-input skips. Run native IK cases in the pinned native
environment as well.
For the public golden corpus, both `scripts/maintainer/check_golden_exact.py` and
`scripts/maintainer/bench_scoring_replay.py --exact` accept `--golden-root` in
addition to `--archive-root`.

Changes to simulator runtime, physics, rendering, cameras, scenes or assets need
the GPU comparison (G), which compares physics and images with reference GPU
recordings on the reference GPU model, the import-source audit (I), and recorded safety/posture
checks. Request this in the PR's validation section, identify the affected tasks,
and leave the GPU result pending until a maintainer attaches the receipt for the
final commit. Fork PRs never run automatically on maintainer GPU hosts. There is
no public GPU workflow; maintainers run the GPU harness after reviewing the code.
Scoring or golden changes
also need the authoritative exact byte tier; tolerances and thresholds do not
change to accommodate a result.

Follow the [release process](docs/contributing/release.md) for distribution changes.
