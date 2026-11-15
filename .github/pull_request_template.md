## Summary

Describe the problem and resulting behavior.

## Contributor validation

- [ ] `uv sync --locked --all-packages`
- [ ] Public golden fetch (or `TOWER_GOLDEN_ROOT` / `--golden-root`)
- [ ] Ruff lint and format checks
- [ ] `uv run pytest -ra -n auto` — include pass and skip counts
- [ ] Public tree and wheel leak checks

Public CI runs CPU tests with public inputs on Python 3.11/3.12, native IK jobs,
characterization, typing, packaging and leak checks. Missing golden data is a
marked skip locally and a failure in CI, never a passing score check.

## Maintainer validation

Describe effects on simulator runtime, physics, rendering, cameras, scenes,
assets, scoring or control safety. Request the affected checks here.

- [ ] Simulator/runtime/physics/rendering/camera/scene/asset changes: request
  GPU comparison (G) against reference GPU recordings, import-source audit (I),
  and recorded safety/posture review.
- [ ] Scoring/golden changes: request both exact commands in the archived
  numerical environment, with the public `--golden-root`.
- [ ] Changes affecting frozen code, v1 scenes, expert commands or reference GPU recording
  proofs: run the remaining private archive gate with zero archive skips.

Attach receipts for the final commit. Fork PRs never run automatically on
maintainer GPU hosts. There is no public GPU workflow; maintainers review the code
before running the GPU harness.
