# Scoring and runtime characterization

The public corpus contains 24 recordings for the 12 benchmark tasks. Scene
identifiers use the public 2026.11.1 selectors. Every other recording member,
including physical arrays, finger data and stored scores, retains its original
bytes. `benchmark-2026.11/MANIFEST.json` pins the public archive hashes and sizes.
The exact original recordings and frozen source remain in the private archive.

Fetch public inputs and run the CPU checks:

```sh
uv run python -m scripts.maintainer.fetch_golden
uv run python -m scripts.maintainer.fetch_public_test_data
uv run pytest -ra -n auto
```

For offline work, `TOWER_GOLDEN_ROOT` or `pytest --golden-root` accepts the golden
directory or complete public asset folder. `TOWER_ASSET_ROOT` or
`pytest --asset-root` supplies public scenes, station and textures. The fetchers
verify immutable commit, file size and SHA-256. Missing data without network is
an explicitly marked skip; an incomplete or changed explicit root is an error.
Public CI fetches the inputs before running tests, so fetch failures fail CI.

## Portable and exact comparisons

The public portable tier checks library scoring, online streams, every public
scene, and the off/fast/audit scoring modes. JSON types, keys, list lengths,
integers, booleans, strings and null remain exact. Float leaves use absolute and
relative tolerances of `1e-12`, defined in `characterization/golden_compare.py`.
Non-finite representations and signed zero remain exact. Accepted floating
point differences are reported with their JSON pointer and both deltas.
Only declared Qhull vertex lists permit cyclic rotation; reflection or arbitrary
reordering is never accepted.

The authoritative exact tier runs in the archived Linux x86-64 environment:
Python 3.12.3, NumPy 2.3.1, SciPy 1.17.0, h5py 3.15.1, Qhull 2020.2.r and
glibc/libm 2.39. Maintainers run both commands against the public recordings:

```sh
python scripts/maintainer/check_golden_exact.py \
  --archive-root "$TOWER_ARCHIVE_ROOT" --golden-root "$TOWER_GOLDEN_ROOT" \
  --receipt exact.json
python scripts/maintainer/bench_scoring_replay.py --exact \
  --archive-root "$TOWER_ARCHIVE_ROOT" --golden-root "$TOWER_GOLDEN_ROOT" \
  --receipt bench-exact.json
```

Both commands must pass all 24 trials. They retain the original frozen corpus
and source identities while checking the public recordings. No comparator,
tolerance, original archive member or archived score is rewritten.

Archive-marked tests are limited to one-time migration comparisons requiring
frozen 2026.11.0 code, v1 scenes, private expert commands or the reference GPU
recordings. Maintainers run them with an explicit archive root and zero archive
skips. Public contributors can run all other CPU checks without that archive.

## Native and GPU checks

Public native IK CI runs the package solvers in the pinned environments.
Joint and FK tolerances stay at `1e-9` with zero relative tolerance. Command
bounds, J2 conversion and simulation observation tolerance remain separate.
Rejected ordinary grasps never fall back to a reversed target.

Simulator runtime, physics, rendering, camera, scene and asset changes need a
maintainer G check on the reference GPU model, import check I and recorded
safety/posture review, including Task7 S where applicable. Request these checks
in the PR template. Fork PRs never run automatically on maintainer GPU hosts.
The comparator retains the 35 dB image floor and existing physical thresholds;
bit-identical physics is required for metadata-only asset changes. See
[CONTRIBUTING](../../CONTRIBUTING.md) for the review process.
