# Public benchmark runtime

The public runtime uses `tower_sim.bench`, benchmark `2026.11.1`, data version
`benchmark-2026.11.1`, and request schema `tower-public-benchmark-v2`.
The three wheels are `tower-sim==0.1.0`, `tower-eval==0.1.0`, and
`nero-kinematics==0.2.0`. A release manifest identifies the exact build.

## Install the pinned SDK

Run from the checkout root. Install [uv](https://docs.astral.sh/uv/) and a
Micromamba first (to create the separate conda-forge prefixes). Isaac uses Python 3.12:

```sh
uv venv --python 3.12 .venv-isaac61
uv pip install --python .venv-isaac61/bin/python "isaacsim[all,extscache]==6.1.0" --default-index https://pypi.org/simple --index https://pypi.nvidia.com --index-strategy unsafe-best-match
```

Both public indexes and `unsafe-best-match` are required: first-index resolution
can hide the required `mujoco-usd-converter==0.5.0` candidate on PyPI.
Review and accept NVIDIA's license as described in the [Isaac guide](../isaac-sim.md).

On Debian/Ubuntu, install the system prerequisite with `sudo apt-get install libxt6`.
It supplies `libXt.so.6`, needed by the MaterialX/USD plugins even on headless hosts.
Without root, install it in a separate conda-forge prefix:

```sh
micromamba create --prefix "$PWD/.isaac-system-libs" --override-channels -c conda-forge xorg-libxt -y
LD_LIBRARY_PATH="$PWD/.isaac-system-libs/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" uv run python tools/simulator/benchmark.py check-environment --isaac-python .venv-isaac61/bin/python
```

Use that same process-scoped `LD_LIBRARY_PATH` prefix for each Isaac benchmark
invocation. Do not add it globally to shell startup files or to the IK environment.
The check loads `libMaterialXRenderHw.so`, `libMaterialXRenderGlsl.so` and
`libusd_usdBakeMtlx.so` without starting Kit or creating GPU contexts.
A metadata-only version match is insufficient.

The selected GPU chooses the render and physics device. The launcher deliberately
unsets `CUDA_VISIBLE_DEVICES`; the SDK can create small CUDA contexts on other
visible GPUs. On shared hosts, use whole-host reservation or container-level GPU
isolation; `CUDA_VISIBLE_DEVICES` alone does not isolate this runtime.

For the independent native solver, follow the complete
[IK install and acceptance recipe](kinematics.md#install-and-check).

## Build and verify

Build from a fresh, clean committed checkout. Keep caches outside bound trees:

```sh
export PYTHONDONTWRITEBYTECODE=1
GIT_LFS_SKIP_SMUDGE=1 git submodule update --init third_party/XPolicyLab
uv build --wheel --out-dir dist packages/nero-kinematics
uv build --wheel --out-dir dist packages/tower-sim
uv build --wheel --out-dir dist src/tower-eval
uv run python -m scripts.maintainer.check_public_tree --wheel dist/tower_sim-0.1.0-py3-none-any.whl
uv run python -m scripts.maintainer.candidate_release --output dist/release-manifest.json
```

Install all three wheels in isolated package directories used by both the Isaac
interpreter and the independent IK interpreter. Calibration, runtime profiles,
shared IK specification and camera configuration are installed package data.
Physical assets and evaluation scenes use explicit external roots.

The release binds the tracked `tools/simulator/` and `scripts/` files and every
wheel payload. It also binds `third_party/XPolicyLab/client_server/` at the pinned
gitlink, reading committed bytes from the submodule repository. An initialized
submodule with its own Git top-level and the exact pinned revision is required;
an empty submodule directory is rejected explicitly. All tracked files must remain clean. Untracked files, including
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
python tools/simulator/benchmark_assets.py download --assets /path/to/assets
python tools/simulator/benchmark.py verify --assets /path/to/assets --data-root /path/to/benchmark-2026.11.1
python tools/simulator/benchmark.py describe
python tools/simulator/benchmark_portable.py prepare-request --help
python tools/simulator/benchmark.py run --help
```

Each task has fifty fixed scenes, `scene-000` through `scene-049`. The request
field `seeds` selects integer case numbers 0 through 49. It does not create scenes.
The default data directory is `<assets>/benchmark-2026.11.1`. Downloads use the anonymous pinned TOWER-Assets revision in the release lock.
Use `--data-root` only to verify an existing evaluation-data directory.
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

The WebSocket extra requires `pydantic>=2.11,<3`, whose dependencies include
`typing-inspection`. This makes the installed dependency manifest consistent
with the supported extra; older Pydantic installations must be upgraded.
