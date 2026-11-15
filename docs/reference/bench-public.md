# Public benchmark runtime

The public runtime uses `tower_sim.bench`, benchmark `2026.11.1`, data version
`benchmark-2026.11.1`, and request schema `tower-public-benchmark-v2`.
The three wheels are `tower-sim==0.1.0`, `tower-eval==0.1.0`, and
`nero-kinematics==0.2.0`. A release manifest identifies the exact build.

## Build and verify

Build from a fresh, clean committed checkout. Keep caches outside bound trees:

```sh
export PYTHONDONTWRITEBYTECODE=1
uv build --wheel --out-dir dist packages/nero-kinematics
uv build --wheel --out-dir dist packages/tower-sim
uv build --wheel --out-dir dist src/tower-eval
python -m scripts.maintainer.check_public_tree --wheel dist/tower_sim-0.1.0-py3-none-any.whl
python -m scripts.maintainer.candidate_release --output dist/release-manifest.json
```

Install all three wheels in isolated package directories used by both the Isaac
interpreter and the independent IK interpreter. Calibration, runtime profiles,
shared IK specification and camera configuration are installed package data.
Physical assets and evaluation scenes use explicit external roots.

The release binds the tracked `tools/simulator/` and `scripts/` files and every
wheel payload. All tracked files must remain clean. Untracked files, including
bytecode, are rejected inside `src/`, `packages/`, `tools/simulator/`
and `scripts/`. Environments and outputs can live outside those bound trees.
Installed package file sets and bytes must equal their wheels. Startup validates
loaded sources before GPU work; final source receipts repeat that validation.
Unsupported planning entry points, private archive compatibility modules and
private identifiers are rejected in the release tree and wheel payloads.

Interpreter and SDK roots are explicit. They cannot overlap bound source trees.
Third-party packages under those roots are admitted by location; the dependency
manifest independently binds selected numerical and SDK dependencies. The IK
manifest pins Pinocchio 3.8.0, CasADi 3.7.2, pin-pink 3.4.0 and NumPy 1.26.4.
Conda distributions are checked against their installed package manifests.

## Evaluation data

```sh
python tools/simulator/benchmark_assets.py download --assets /path/to/assets --data-root /path/to/benchmark-2026.11.1
python tools/simulator/benchmark.py verify --assets /path/to/assets --data-root /path/to/benchmark-2026.11.1
python tools/simulator/benchmark.py describe
python tools/simulator/benchmark_portable.py prepare-request --help
python tools/simulator/benchmark.py run --help
```

Each task has fifty fixed scenes, `scene-000` through `scene-049`. The request
field `seeds` selects integer case numbers 0 through 49. It does not create scenes.
The default data directory is `<assets>/benchmark-2026.11.1`. Before the dataset
revision is pinned, supply the verified local staging directory with `--data-root`.
CPU verification without assets checks metadata and reports zero verified payloads.

Every trial records its benchmark, data version, request schema, package versions,
release manifest hash, commit, and data manifest hash. Offline commands leave
release identifiers null because they do not authorize simulation.

## Control and review contract

Preparation binds protocol, scenes, assets, policy identity, release, dependencies
and current posture-review evidence. Reject reverse 180-degree symmetric grasps;
never retry with a reversed target. Preserve each task's contact geometry. Review
planned approach/carry trajectories and recorded approach phases before accepting
a benchmark run. An old pass or the absence of a symmetry flag is insufficient.
Supply the posture-review document with the public `--pilot-review` option.

Command bounds remain the real normal IK soft bounds intersected with simulation
URDF bounds, inset by 1.5 degrees, with J2 Tower hard limits inset by 5 degrees.
J2 and coordinate conversion, the separate 0.5-degree observation tolerance, and
real hardware protections retain their existing meanings. Learned-policy runs
reject `command_replay`. The exact-flip tripwire cannot replace posture review.

Option abbreviations are disabled. Runtime evidence does not establish suitability
for hardware use. Public scene v2 and historical recording readers preserve their
physical/scoring contracts. Private source history and maintainer replay are not
public CLI features.

See the canonical [control safety contract](../guides/eef-control.md#control-safety-contract)
for the scope of runtime posture checks and the distinct control limits.
