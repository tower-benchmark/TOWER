# Nero kinematics

Convert an absolute tool-center-point (TCP) pose and measured joint angles into
absolute joint targets using Pinocchio, CasADi and IPOPT. Supports seven named
revolute joints per arm, with one solver instance per arm. No simulator or hardware
connection is required. The included Nero model and TCP offset are simulation
references; hardware calibration and testing are incomplete.

## Install and check

From the TOWER checkout root, create the native environment separately from Isaac:

```bash
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
micromamba create --prefix "$PWD/.venv-benchmark-ik" -f packages/nero-kinematics/environment.yml -y
ik_python="$PWD/.venv-benchmark-ik/bin/python"
"$ik_python" -m pip install --index-url https://pypi.org/simple --no-deps -r packages/nero-kinematics/requirements-pip.txt
"$ik_python" -m pip install --index-url https://pypi.org/simple --no-deps ./packages/nero-kinematics ./packages/tower-sim ./src/tower-eval
export TOWER_REQUIRE_NATIVE_IK=1
"$ik_python" packages/nero-kinematics/examples/offline_check.py
"$ik_python" -m pytest packages/nero-kinematics/tests tests/characterization/test_ik.py -v
```

The environment file provisions native dependencies, including Quadprog, through
conda-forge. The pip requirements pin Pink, QPsolvers and loop-rate-limiters with
`--no-deps`, avoiding optional solver backends. Ordinary pip
resolution can install a conflicting `pin==2.7.0`. Keep conda Pinocchio 3.8.0
and its CasADi bindings; do not install the pip `pin` distribution.
These acceptance tests use the source packages installed into the native prefix.
The alternative `configs/benchmark-ik.yml` uses the same two-step recipe; replace
the environment-file argument above to test that prefix.

After the [release build](bench-public.md#build-and-verify), install the three
wheels into a fresh shared package directory and run the release identity probe
with the native interpreter. This probe uses the wheel installation in
`runtime-packages` selected by `PYTHONPATH`, rather than the source installation:

```bash
"$ik_python" -m pip install --index-url https://pypi.org/simple --no-deps --target "$PWD/runtime-packages" dist/*.whl
export PYTHONPATH="$PWD/runtime-packages"
release_sha256=$(sha256sum dist/release-manifest.json | cut -d ' ' -f 1)
"$ik_python" -m tower_sim.bench.verification --ik --release dist/release-manifest.json --release-sha256 "$release_sha256" --checkout "$PWD"
```

Retain the probe's dependency and source identities alongside the acceptance and
parity test results. Use this same package directory for both runtime interpreters.

The tested environment is Linux x86_64, Python 3.11, Pinocchio 3.8.0 with CasADi
bindings, CasADi 3.7.2 and IPOPT 3.14.19; exact versions are in `environment.yml`.
Pinocchio is provisioned through conda, separately from pip dependencies. The wheel
includes all model and profile data. Native robotics dependencies are external.
The base package needs only NumPy/SciPy, and imports no Pinocchio, CasADi or Pink
until a native solver is constructed.

## Python API

```python
from nero_kinematics import ArmIK, solve_pair

arm = ArmIK("examples/nero_sim_reference.json")
measured_joints = [0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0]
# Offline hold example. In a control loop, supply actual measured joint angles
# and the desired TCP pose; model FK is not a physical TCP measurement.
target_pose = arm.fk(measured_joints)
result = arm.solve(target_pose, measured_joints, frame="arm_base")
if result["success"]:
    joint_targets = result["joints"]
else:
    print(result["diagnostics"])

other_arm = ArmIK("examples/nero_sim_reference.json")
pair = solve_pair([arm, other_arm], [target_pose, target_pose], [measured_joints, measured_joints])
```

Inputs use metres, radians and `[x, y, z, qx, qy, qz, qw]`. Joint arrays contain
seven angles in configured `joint_names` order. Only `arm_base` poses are accepted;
convert world poses with the calibrated world-to-arm-base transform first.

Results contain `success`, `joints`, `joint_names`, `backend` and `diagnostics`.
Numerical failure returns `joints: null`; malformed inputs raise `ValueError`.
`solve_pair` withholds both targets if either arm fails. Each instance requires
serial calls. `arm.identity` records model and solver settings; `arm.reset()` clears
diagnostics without moving anything. Gripper commands use a separate controller.

## Persistent JSONL CLI

```bash
eef-to-joint --config examples/nero_sim_reference.json --identity
eef-to-joint --config examples/nero_sim_reference.json
```

The second command reads one request per stdin line and writes one result per
stdout line, reusing the solver. Each request requires exactly these fields:

```text
{"pose": [x,y,z,qx,qy,qz,qw], "measured_joints": [q1,q2,q3,q4,q5,q6,q7], "frame": "arm_base"}
```

Replace the symbols with numbers. Invalid requests return `success: false` and
`joints: null`; later requests continue. Check every result before using its targets.

## Configure another arm

Copy `examples/nero_sim_reference.json` and update:

- `urdf`: model path relative to the configuration file.
- `joint_names`, `lower`, `upper`: seven controlled joints and deployment limits.
- `base_frame`, `tcp_frame`: model frames defining the arm base and tool attachment.
- `tcp_pose`: fixed attachment-to-TCP transform; use identity if `tcp_frame` is the TCP.
- `reference_configuration`: optional full Pinocchio configuration fixing other joints.

Deployment limits intersect URDF limits. Continuous joints still take scalar angles
and require finite deployment limits. Verify encoder direction, zero offsets and TCP
calibration against measured joint/TCP samples before hardware integration.
`model_status` records a description; it does not perform calibration.

## Acceptance and limits

Defaults require IPOPT `Solve_Succeeded`, finite joints within limits, FK residuals
at most 1 mm and 0.01 rad, and elapsed solve time at most 25 ms. The iteration limit
is 80. CPU limiting and post-solve wall-time rejection are soft limits; a call can
return later than its budget. There is no fallback or automatic tolerance relaxation.

The output is an IK target, not a collision-free trajectory. Hardware integration
must provide speed/acceleration limits, tracking and failure handling. Physical
tracking and 30 Hz timing targets remain unmet in simulation; numerical FK accuracy
does not establish physical accuracy. Test results are retained in CI artifacts.


## Bounds, seeds and posture evidence

`CommandBounds` and `SimulationObservationTolerance` are distinct immutable types;
command validation has no tolerance argument. Observation slack does not modify IK
bounds or the encoder/physical conversion. This library introduces no environment-variable reader.

`PinocchioIK` additionally accepts `optimizer_tolerance` and opt-in
`verified_current_seed`; `ik` and `ArmIK.solve` accept `joint_radius_rad` and
`margin_rad`. A measured seed in a soft margin may retreat but may not advance
toward its hard limit. Canonical defaults retain the original optimization path.
The seed shortcut requires finite FK agreement within 1e-9 and honors the time
budget. It is enabled only for the legacy vendor profile or explicit opt-in.

For grasp planning, supply the ordinary contact frame for the same task to
`posture_evidence(target, ordinary_target)`. TCP x is approach and z is closing by
default; other conventions must pass explicit axes. Signed closing-axis agreement
rejects a half-turn about approach without relying on a symmetric flag.
`solve_grasp(callback, target, seed, ordinary_target)` attempts an ordinary target
once and returns failure unchanged. Check all generated approach/carry samples
and recorded approach phases as well; this library does not certify unobserved
physical trajectories or replace their review.


See the [control safety contract](../guides/eef-control.md#control-safety-contract)
for scope, command bounds and required trajectory review.

Public benchmark runs do not require `--pilot-review`. Recorded TCP posture is
reported in `posture-flags.json`, independently of scores and action mode.
The automatic flag also covers joint commands and near-half-turn grasps; it
does not replace maintainer contact and approach/carry review. See
[posture flags](scoring.md#posture-flags-reported-not-scored).
