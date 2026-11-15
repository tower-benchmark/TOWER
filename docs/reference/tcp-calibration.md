# TCP calibration caches

The calibration cache uses `tower-recorded-tcp-v2`.

## Validity and provenance

V2 dispatches on `tower-recorded-tcp-v2` and checks this semantic key:

| Component | Why it is required |
| --- | --- |
| `fit_algorithm` | `tower-tcp-median-rotation-mean-v1` fixes the fitting operations below. Any change requires a new version. |
| `coordinate_convention` | `nero-base-link-gripper-base-to-eef-m-xyzw-rad-v1` fixes frames, transform direction and units. |
| `kinematics_sha256` | Exact `nero.urdf` and `nero.yaml` bytes define the serial chain, frames and joint ordering. |
| `stage_sha256` | Exact verified station/stage identity prevents reuse across unrelated installations. |
| `kinematic_geometry_sha256` | The retained geometry is the URDF chain, so this equals its digest. A runtime with a separate geometry artifact must supply that artifact's digest. |
| `fit_source.episode_id` | Identity of the first source whose split is `fit`. |
| `fit_source.profile_channel_sha256` | Preserves the historical motion-profile channel identity. |
| `fit_source.fit_channel_sha256` | Binds the actual `observations/qpos` and `observations/eef` inputs; the historical digest does **not** cover observation joints. |
| `profile_selection` | First-fit-source selection and the two channel-hash specifications. File location, FPS, other sources and unrelated motion primitives do not affect this fit. |

`release_provenance` records the source commit and historical backend/profile
hashes separately. These are **not validity keys**. Moving or changing unrelated
backend code does not invalidate v2. Algorithm, coordinate and selection
versions are explicitly supported: an unknown version fails even if both
supplied and cached keys claim it. Unknown schemas fail with a schema error.

Build the expected key from verified runtime inputs, not from the cache being
validated:

```python
from tower_eval.calibration import cached_tcp_calibration_v2, tcp_calibration_key

key = tcp_calibration_key(
    profile=verified_profile,
    stage_sha256=verified_stage_hash,
    kinematics=kinematics_directory,
    fit_channel_sha256=verified_fit_input_receipt["key"]["fit_source"]["fit_channel_sha256"],
)
with cached_tcp_calibration_v2(
    actual_model_class, cache_path, key=key, kinematics=kinematics_directory
):
    environment = construct_environment()
```

The schema dispatcher also supports
`cached_tcp_calibration(actual_model_class, cache_path, key=key, kinematics=...)`
and rejects v1-only arguments for a v2 cache. Neither v2 API needs a backend
path. When `kinematics` is given, its URDF/YAML are re-hashed against the key
when the fit is restored, because the model may export them during
construction (v1 does the same). Both patch the supplied **class object**, so module aliases
observe the same method; restoration runs on normal exit and exceptions, and
nested contexts restore the enclosing patch. These process-wide contexts must
not overlap across threads. Each instance receives separate arrays/report data.
At application time the actual profile's first fit source is checked again.

The caller verifies station and fit-input receipts independently and constructs
the key immediately before use. Cached operation needs no raw episode; it cannot
independently detect a forged receipt or a model that falsely claims this
algorithm. Release/source provenance verification remains a separate concern.

## Fit algorithm specification

`tower-tcp-median-rotation-mean-v1` describes the unchanged
`LulaModel.calibrate_tcp` in `packages/tower-sim/src/tower_sim/backend/backend.py`:

1. Start with an **uncalibrated** model (`tcp_offset = 0`, identity TCP rotation).
   Choose the first `profile.sources` entry with `split == "fit"`, in list order.
   Read `observations/qpos[::15]` and `observations/eef[::15]`, starting at frame
   zero; reshape EEF rows to `(2, 7)`. Valid paired channels have equal frame
   counts. The existing fitter iterates `zip(qpos, measured)`.
2. Iterate sampled rows in order, arm 0 then arm 1. Joint columns are `0:7` and
   `8:15`; columns 7 and 15 are grippers and are excluded. Joint angles are
   radians passed directly to FK. This fitter performs **no** encoder/J2
   conversion, world installation transform, delay compensation, filtering or
   resampling. It changes no IK bounds or hardware protections.
3. FK gives the `gripper_base` pose in `base_link` coordinates. Recorded EEF
   poses are in that same arm-local coordinate convention. Positions are metres,
   quaternion components are **xyzw**. For raw pose `(p, R)` and measured pose
   `(e, E)`, accumulate `d = R.inv().apply(e - p)` and
   `Q = R.inv() * E`. The fitted transform maps gripper-base coordinates to the
   recorded EEF TCP: `p_tcp = p + R @ d`, `R_tcp = R * Q`.
4. Pool both arms without weights. Translation is the componentwise NumPy
   median of all `d`; rotation is SciPy `Rotation.from_quat(Q).mean()` (the
   unweighted chordal rotation mean). No quaternion sign canonicalization or
   averaging of Euler angles is applied.
5. Position residuals are `norm(d - median_d)` in metres. Angular residuals are
   `(Rotation.from_quat(Q) * mean_Q.inv()).magnitude()` in radians. Record NumPy
   percentiles `[50, 95]` with the default **linear** method; convert the angular
   percentiles to degrees. Reject if position p95 **exceeds** 0.005 m or angular
   p95 **exceeds** 3 degrees. Keep the operation order. Numerical dependencies
   are recorded in the [fit-equivalence receipt](../../benchmarks/2026.11/calibration/fit-equivalence.json); they are release provenance rather
   than semantic-key fields.

The profile channel digest retains the historical concatenation of C-order
bytes: `leader/qpos.astype(float)`, `observations/eef.astype(float)` reshaped to
`(-1,2,7)`, `observations/gripper_status.astype(float)`, then
`timestamps/row_ros_ns[:,0]` in its stored dtype. The calculation uses
little-endian native float64. Its scope is
`leader_qpos_observation_eef_gripper_timestamps`.

The additional `tower-tcp-observation-channels-v1` digest is SHA-256 over the
UTF-8 scope plus NUL, followed by `observations/qpos` then `observations/eef`.
For each channel append compact, sorted-key JSON
`{"channel": name, "dtype": array.dtype.str, "shape": list(array.shape)}`, NUL,
then C-order bytes of **all frames in the stored dtype**, without float
conversion. This covers both sampled data and shape/order metadata. Images and
other channels are never read or distributed by the checker.

Raw fit data is neither committed nor redistributed. CI uses a small invented
HDF5 fixture and injected CPU FK to exercise the **actual fitter**: first-source
selection, frame stride, both arms, coordinate composition, median versus mean,
rotation mean, residual quantiles and both rejection thresholds. It also checks
v1 compatibility, exact copied floats, key rejection and module-alias behavior.

See the [control safety contract](../guides/eef-control.md#control-safety-contract).
