# Absolute Joint / EEF control

TOWER accepts absolute joint targets and absolute TCP poses through
`DualModeController`. Both modes use the same joint/gripper driver. The optional
Pinocchio/CasADi/IPOPT backend computes joint targets; the simulator executes them.
For a standalone converter, see the [Nero package](../reference/kinematics.md).

## Input contract

| Field | Convention |
|---|---|
| Arm order | Left, then back |
| Joint action | `joints`: two arrays of seven absolute angles, radians |
| EEF action | `poses`: two `[x,y,z,qx,qy,qz,qw]` TCP poses, metres and xyzw |
| Gripper | Two full jaw widths, metres, normally 0–0.1 m |
| Frame | `arm_base` or `environment_world` |

Each action must have exactly its mode's fields. Quaternion norms must be within
1e-4 of one; malformed, nonfinite, out-of-limit and untagged EEF inputs are rejected.
`environment_world` means environment-local world coordinates: clone offsets are
added before converting into the corresponding arm base. Delta commands are unsupported at this controller boundary. XPolicyLab’s
`eef_delta` mode describes policy encoding and decodes to absolute targets; see
[policy actions](../reference/policy-interface.md#actions-frames-and-units).
Legacy untagged 16-element arrays remain absolute joint commands.

## Public API

With an initialized simulator backend:

```python
from tower_eval.controllers import DualModeController
from tower_eval.envs import TowerEnv
from tower_eval.kinematics.pinocchio_ik import PinocchioKinematics

kinematics = PinocchioKinematics.from_simulator(sim)
controller = DualModeController(
    sim.robot, kinematics=kinematics, arm_speed_rad_s=1.0, interpolation_fraction=0.8
)
env = TowerEnv(sim, controller, action_dt_s=0.1)
env.reset()
state = sim.read()
env.step(
    {
        "input_space": "eef",
        "poses": state.eef.tolist(),
        "gripper": state.qpos[[7, 15]].tolist(),
        "frame": "arm_base",
    }
)
env.step(
    {
        "input_space": "joint",
        "joints": state.qpos[[*range(7), *range(8, 15)]].reshape(2, 7).tolist(),
        "gripper": state.qpos[[7, 15]].tolist(),
    }
)
```

Set `action_dt_s` explicitly when migrating callers. With a `DualModeController`,
its default is one control tick (1/30 s on the simulator backend). With the original
controller, the default remains 0.1 s. Durations must be integer multiples of the
control period; controller/backend period mismatches are rejected before execution.

`controller.validate(action)` checks and returns parsed mode, values and widths.
`controller.applied_target` returns a copy of the last command submitted to
`sim.step`, or None before execution. A submitted command does not confirm physics
completion; failure records include the measured state separately.
`TowerEnv.step` returns per-tick IK reports in its info dictionary.

`ActionChunk` supports tagged Joint/EEF sequences; its period must match
`env.action_dt_s`. Persisted chunks use `tower-action-chunk-v1`.
See [policy adapters](write-a-policy-server.md). For `tower run`, remote inference uses XPolicyLab's WebSocket transport; the Isaac
benchmark runtime uses [WebSocket](../reference/policy-interface.md#isaac-benchmark-runtime). Encoder-frame policy chunks require an environment converter
and are rejected by this generic controller boundary; they are not world-frame
TCP targets.

## Execution and failure behavior

The shared driver owns the full action schedule. For each EEF tick it requests
new IK solutions seeded from the actual measured angles, validates both arms,
and interpolates from the action's initial measurement toward that tick's accepted
target. Direct gripper widths follow the same schedule; simulator gripper profiles
retain their existing rate/contact envelopes. This is joint-space interpolation,
not a Cartesian path or collision-clearance guarantee.

Both arms must pass before either arm or gripper is dispatched. `IKFailure.report`
contains failure evidence; earlier physics steps remain. Reset clears reports and
controller/provider state. Optional `arm_speed_rad_s` bounds command increments in
both modes; its default None preserves existing Joint behavior.

Acceptance requires IPOPT `Solve_Succeeded`, finite in-limit joints, numerical FK
position error at most 1 mm, orientation error at most 0.01 rad, and the configured
elapsed-time limit. Default limits are 80 iterations and 25 ms per arm. CPU limiting
and post-solve wall-time rejection are soft limits, not hard real-time guarantees.
No automatic fallback, random restart or tolerance relaxation is used.

Pinocchio maps named joints, supports continuous revolute configurations, and fixes
noncontrolled joints at the configured reference. Limits intersect URDF and
articulation limits. The deployed URDF, TCP calibration and arm bases come from the
simulator. Numerical FK acceptance does not establish physical tracking accuracy.


## Install native dependencies

Joint-only installation does not require Pinocchio/CasADi/IPOPT. The `eef` extra
installs CasADi; Pinocchio with CasADi bindings is provisioned separately using the
[standalone environment file](../../packages/nero-kinematics/environment.yml):

```bash
conda env create -p "$PWD/.venv-pin" -f packages/nero-kinematics/environment.yml
"$PWD/.venv-pin/bin/python" -m pip install --no-deps ./packages/nero-kinematics
.venv-pin/bin/python -c 'import pinocchio.casadi, casadi'
```

For the `pink` method, also install [the pip requirements](../../packages/nero-kinematics/requirements-pip.txt) with `python -m pip install --no-deps -r packages/nero-kinematics/requirements-pip.txt` using the native prefix interpreter.

Do not replace Isaac's NumPy or shared native libraries. Keep native dependencies
in an isolated environment.

## Control safety contract

Command IK bounds are real normal soft bounds intersected with simulation URDF bounds,
inset by 1.5 degrees, with J2 additionally bounded by Tower hard limits inset by
5 degrees. Keep coordinate/J2 conversion and the simulation-only 0.5-degree
observation tolerance separate. Observation tolerance never expands command
bounds. Real hardware protections remain unchanged.

Expert data must reject reverse 180-degree symmetric grasps with no reversed fallback. Preserve
task-specific contact geometry. Inspect the approach and carry trajectory and
recorded approach phases; absence of a symmetry flag alone does not establish
posture acceptance. A failed ordinary grasp remains a failure.

Use `nero_kinematics.posture_evidence`/`solve_grasp` for task-relative checks.

Scope: `TowerEnv` and `DualModeController` perform no posture check. The Isaac
learned-policy route applies only an exact-flip tripwire to EEF targets
([policy route limits](../reference/bench-policy.md)); it does not reject an
orientation hemisphere, and joint-absolute commands bypass it. Benchmark request
preparation instead requires a reviewed posture record. Learned-rollout posture
compliance rests on review of the planned approach/carry and recorded approach
phases. No runtime check establishes posture acceptance.

See the benchmark [control and review contract](../reference/bench-public.md#control-and-review-contract) for required recorded posture evidence.
