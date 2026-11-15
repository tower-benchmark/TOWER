# TOWER ↔ XPolicyLab policy interface

TOWER owns evaluation, scoring, simulation and robot control. XPolicyLab owns
policy loading and serving. `third_party/XPolicyLab` pins the TOWER integration
branch of Apache-2.0 XPolicyLab. The environment imports only
`client_server/ws/model_client.WsModelClient` and its transport dependencies.

Initialize the submodule with
`GIT_LFS_SKIP_SMUDGE=1 git submodule update --init third_party/XPolicyLab`
and install `tower-eval[xpolicylab]`. The policy environment separately installs
XPolicyLab and the selected model dependencies. See
`third_party/XPolicyLab/policy/OpenWAM_TOWER/README.md` for checkpoint setup.
No checkpoints are distributed by TOWER.

```toml
[policy]
adapter = "xpolicylab"
server = "ws://localhost:19000"
timeout_s = 120

[policy.config]
action_mode = "eef_delta"
max_connect_attempts = 200
connect_timeout_s = 10
max_connect_seconds = 180
# Installed wheels may specify client_root = "/path/to/XPolicyLab".
```

The Python adapter accepts full encoder observations; the current generic CPU
backend supplies only joint state. For CPU protocol verification, use the
explicit synthetic fixture in `tower smoke --policy xpolicylab://demo_policy`.
This launches the actual XPolicyLab demo server, exercises single and batch
requests, runs three CPU evaluation steps and stops its server. It is not a
benchmark or a physical-policy evaluation.

## Embodiment and observations

`env_cfg_type = nero_dual`. Arm index 0 is **left**; index 1 is **back**, named
`right` in XPolicyLab action fields. Parent `third_party/env_cfg` supplies the
runtime robot configuration; the submodule maintains its training registry.

| TOWER observation | XPolicyLab observation | Shape / units |
| --- | --- | --- |
| `images/<camera>` | `vision/<camera>/color` | H×W×3 uint8 RGB |
| `joint_encoder` | `state/joint_encoder` | 2×7 radians |
| `eef_encoder` | `state/eef_encoder` | 2×4×4 rigid TCP transforms |
| `gripper_width_m` | `state/gripper_width_m` | 2, full aperture in metres |
| `instruction` (or `prompt`) | `instruction` | nonempty string |
| `coordinate_frame` | `coordinate_frame` | `encoder_arm_base` |
| optional `env_idx` | `env_idx` | unique integer per active batch |

Camera names are `right_front_camera`, `left_back_camera`, `left_wrist_camera`,
and `back_wrist_camera`. Arrays are transported losslessly with msgpack; RGB
channels are never swapped. The XPolicyLab server owns encoded-image decoding.
OpenWAM_TOWER applies the checkpoint's verified training image composition.
No camera, EEF pose, or prompt is fabricated for real observations.

Joint and EEF observations must already be in **encoder** coordinates. J2
compensation remains in the environment: physical → encoder before inference,
encoder → physical exactly once at command execution. The adapter does not
change IK, command limits, simulation observation tolerances, the exact-flip tripwire,
or hardware protections. Negative observed gripper undershoot is passed through
for the checkpoint's existing observation policy to validate.

## Actions, frames and units

| Mode | XPolicyLab endpoint fields | TOWER decoded action |
| --- | --- | --- |
| `joint_absolute` | `left/right_arm_joint_state` (7), `left/right_ee_joint_state` (1) | 16-vector: left joints, left width, back joints, back width |
| `eef_delta` | `left/right_ee_transform` (4×4), `left/right_ee_joint_state` (1), `chunk_anchor_encoder` (2×4×4) | tagged encoder target matrices plus widths and anchor |

`ActionChunk.metadata` retains mode, `encoder_arm_base` frame, and policy identity.
EEF dictionaries retain their explicit frame and mode. These are runtime
controller inputs, not generic world-frame `ee_pose` actions. The future unified
simulator consumes them through its existing encoder controller; the current CPU
joint-only environment cannot execute an EEF chunk.

OpenWAM predicts a 32-step horizon at 10 Hz, executes its first 16 steps, then
replans. Translation is `p_anchor + delta_p`; rotation is
`delta_R @ R_anchor`, using the checkpoint’s orthonormalized rot6d columns. All 32
steps use one observation-time anchor. Returned EEF targets are absolute,
although the checkpoint's mode remains `eef_delta`. Translations and gripper
full widths are metres; joints are radians. Gripper openness is projected by
the unchanged checkpoint decoder and scaled by its calibrated open width.
Predicted J8 values are diagnostic millimetres and remain masked/held fixed.
The client rejects mode/frame/timing mismatches instead of guessing conversions.
Every EEF row's anchor must exactly equal the request's `eef_encoder`; batches
are checked against their corresponding observations. The generic evaluator
rejects encoder-frame chunks before normalization discards metadata. It has no
J2 converter. CPU configs using `policy.adapter = "xpolicylab"` fail validation
before connecting; only the explicit synthetic smoke fixture reinterprets its
fake joint actions as physical CPU targets.

An explicitly configured `joint_absolute` client also accepts standard untagged
XPolicyLab joint action dictionaries, including `demo_policy`. Real EEF policies
must supply frame tags. OpenWAM_TOWER emits identity, units and timing on every
endpoint; `get_prediction()` exposes its last full horizon for parity audits.

## Lifecycle, seeds and batches

`reset(seed=..., metadata=...)` sends `prepare_case` metadata with `episode_seed`,
then the argument-free model `reset`. The client sends `trial_end` when an episode ends or before the next reset.
This seed identifies the environment
trial. Configured OpenWAM inference uses **seed 42 on every call**, independently
of the environment seed; non-42 inference configurations are rejected.
Reset clears observations, predictions and anchors.

Single calls use `update_obs` then `get_action`. `act_batch` sends
`update_obs_batch` then `get_action_batch` with the ordered active environment
indices; results preserve that order. Duplicate indices or missing observations
fail. OpenWAM batches are serialized independent calls, matching the reference behaviour;
tensor batching is not validated. Each client is called from one thread, and
owns its connection, trial identifiers and cleanup.


## Isaac benchmark runtime

The Isaac runtime uses `RuntimePolicyAdapter`, a thin wrapper around
`XPolicyLabAdapter`, on one WebSocket connection per process. `policy_url` and
`--policy-url` accept only `ws://` and `wss://`. The unreleased request schema
remains `tower-public-benchmark-v2`; benchmark, data and scene identities do not change.
The runtime uses the XPolicyLab client defaults: 120 seconds per request, 10
seconds per connection attempt, and a 180-second connection budget. Installed
runtimes receive the pinned checkout client path explicitly. Release verification
binds its client source files and the installed transport dependency records.

Before simulator startup, `get_identity` authorizes the package digest. Startup
writes `policy-health.json` with `ready: true` and the full identity. Each prediction
must equal both the first and latest startup identity, using existing dictionary
equality rules, including all unknown and nested fields.

A batch start sends `prepare_case` with episode seed and environment seed metadata,
then `reset`. Each fixed tick visits active environments in index order and sends
`update_obs` (with `env_idx`), `get_action`, then `get_prediction` if supported.
These calls remain consecutive on the same connection. One environment's action
rejection does not prevent later environments from receiving their actions.
`trial_end` closes the batch. OpenWAM_TOWER stores only observations, anchors and
last predictions; reset clears those caches. Every inference uses seed 42.

Pixels are decoded from the exact validated archive PNG bytes and sent as raw
RGB uint8 arrays. The server passes those arrays through its image decoder.
Every endpoint is validated with `decode_chunk`, including the exact request anchor.
Controller responses are reconstructed from float64 endpoint rows. With
`get_prediction`, all executed rows must match the full prediction bitwise;
the remaining horizon and auxiliary fields are retained unchanged. Without that
method, only available rows are recorded, with explicit `missing_evidence` fields.
No unexecuted horizon or inference timing is synthesized.

Remote `PolicyOutputError` becomes an action failure. Remote `ValueError` with
exact message `Negative gripper aperture` preserves the frozen input-rejection
record without clipping or changing tolerances. Other model errors, unknown
exception types, connection, timeout and protocol failures are infrastructure
failures. Invalid endpoint contracts fail before control. The server must expose
`details.exception_type` in model errors.

`rpc_start`, `rpc_end` and `policy.rpc_elapsed_sum` cover each environment's WS
exchange, including full-prediction retrieval; payload preparation and response
validation are outside the timer. `rpc_end` now records `transport: websocket`.
Request IDs remain local metrics; transport headers and server profiling fields
are removed. Native inference timing remains available in full predictions.

MME-VLA packages fail authorization: an XPolicyLab MME adapter is not available
in this release. Adding one with qualified history semantics is follow-up work.

See the [control safety contract](../guides/eef-control.md#control-safety-contract).
