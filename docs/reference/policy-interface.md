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


## HTTP compatibility

The Isaac benchmark runtime consumes HTTP `/health` and `/predict` through
`LegacyPredictAdapter`. It does not consume XPolicyLab WebSocket messages yet.
WebSocket serving applies to `tower run` integrations with a suitable environment
and in-process integration; the generic CPU backend rejects it except for the
explicit synthetic smoke fixture.

The endpoint defaults are **30 seconds for health** and **300 seconds for
prediction**. Configure `health_timeout_s` and `predict_timeout_s` directly or:

```toml
[policy]
server = "http://localhost:19000"

[policy.config]
health_timeout_s = 30
predict_timeout_s = 300
```

Both timeouts must be finite and positive. The runtime can use
`LegacyPredictAdapter.from_config(config.policy, expected_identity=authorized_identity)`.
This factory does not enable HTTP physics evaluation in the generic CPU CLI.

`GET /health` returns a JSON object with `ready` and `identity`; `ready` must
be true. `POST /predict` sends JSON with `Content-Type: application/json`,
including the authorized `package_digest`. A prediction returns `identity`,
`mode`, `coordinate_frame`, `units`, `control_hz`, `execute_steps`,
`j8_actuator_mask` and per-step `gripper_width_m`. Joint mode supplies
`joint_encoder_targets`; EEF mode supplies `eef_encoder_targets` and
`chunk_anchor_encoder`. The frame is `encoder_arm_base`. These HTTP response
arrays are distinct from the WebSocket endpoint dictionaries above.

Health must report readiness and match `expected_identity["package_digest"]`.
Other fields in the caller's expected identity are not authorization requirements.
The adapter snapshots the complete first health identity, including nested and
unknown fields. A repeated health read checks the package again; each prediction
must match both the original and latest complete identity. Missing, added or
changed fields fail even when the package digest is unchanged. Mutating a returned
health response cannot alter a snapshot. Call `health` or `reset` before inference;
a first `predict` also establishes health if needed. Each `reset` reads health;
there is no reset RPC or client inference seed. Use a new adapter for a new identity.

RGB images use lossless base64 PNG encoding. Already encoded images and MME
`history` pass through without re-encoding, resampling or reordering. The ordinary
request contains the observation, `images` keyed by camera name, and `prompt`.
MME adds `task`, `episode_id`, `observation_time_s` and complete-prefix `history`.
`predict(..., headers=...)` forwards `X-Baseline-Request-ID`
(`trial:scene_seed:physics_tick`) and `X-Baseline-Capture` (empty, `start`, or `stop`).
The transport does not derive those headers from reset metadata or inference seed.
`Content-Type` is sent first. The `act` convenience method sends no baseline headers.

On prediction errors, `profiling` is removed before classification. HTTP 422
`invalid_policy_output`, and HTTP 422 with the exact body
`{"code": "invalid_observation", "error": "Negative gripper aperture"}`, raise
`PolicyOutputError`. The latter rejects an action without clipping observations
or changing tolerances. Other HTTP/input errors, timeouts, connection failures
and truncated HTTP responses raise `PolicyTransportError`. Health errors are
always transport failures. Malformed successful JSON raises the JSON decoder
error; malformed HTTP error bodies remain transport failures with diagnostic text.

`predict` returns successful parsed responses unchanged. The controller validates
the executed prefix, finite targets, timing, frame, limits and IK before enqueue.
NaN/Infinity in executed targets are action rejections; unused tail values are
not newly validated by the transport. `act` decodes into `ActionChunk` and does
not replace controller validation or coordinate conversion.

See the [control safety contract](../guides/eef-control.md#control-safety-contract).
