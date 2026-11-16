# Camera capture and recorded evidence

`tower_sim.bench.cameras.make_camera` creates synchronous batch capture with
persistent 320×240 products, CPU RGB and global masks. `/rtx/post/aa/op=4`
selects the antialiasing mode; it is not a samples-per-pixel count. The first
capture uses 32 synchronous steps at six subframes. Later captures adapt between
one and six subframes using the existing motion thresholds; startup also has
separate two-subframe setup steps. Isaac imports remain inside runtime functions.

## Factory and guard

Call `make_camera(sim, output, *, resources, near_guard)` with a simulation
object exposing `.app` and `.env`, an output path, `resources=Resources(...)`
and `near_guard=guard.check`. These resource and guard arguments are required.
Construct `TerminalGuard(original_guard, env, output)` with the lazy `wrist_guard`
adapter as the original callable. Each guard owns its check records, terminal
latch and `reference_environment(index)` context. Consume latched terminals with
`consume`; there is no global active guard.

The fixed capture chain is `SynchronousBatchCapture` →
`BatchedLiveCameraObservation` → `LiveFourCameraObservation` →
`OrdinaryFourCapture` (near10) → `Native61Capture`. It preserves timestamp,
replicator-completion and application-dispatch freshness checks, including the
SDK timestamp-epsilon assertion. Class substitution, module eviction and late constant patching are absent.

`SerialCameraReference(camera, index, guard=guard)` uses an independent ordinary
renderer to check tiled row assignment. Pose evaluation enters the indexed guard
context. Ambiguous or incorrect assignments fail closed. Products remain attached
until process shutdown to preserve SDK handles. The caller must explicitly check
physical state before and after capture with `assert_unchanged`.

**Delivery validity.** Delivered RGB must have four correctly formatted views,
each with `maximum > 0`, and at least one with quality `valid: true`. If a subset
fails quality, an independent same-state render must meet the same predicate
and match every view within MAE ≤8 and absolute channel bias ≤6. Missing, invalid
or all-zero views and four quality-failed views are infrastructure rejections.
A partial blackout with low nonzero noise in both capture paths can still pass
the pixel comparison; this remains indistinguishable from confirmed local
occlusion under these checks. Four quality-failed views are always rejected.

## Mount evidence

`adopt_usd_wrist_mounts` preserves source-authored local wrist mounts in every
environment. Its `tower-bench-source-wrist-mount-adoption-v1` receipt binds the
helper's module-and-function name, source path and hash, caller-provided source
hashes, reference receipt hash and every mount matrix.

Call `verify_mount_evidence(camera_report, *, source_hashes, batch_root,
expected_helper_sha256=None)`. It verifies identities and transforms at the
existing 1e-10 tolerance. The receipt must be inside the batch; receipts and
serial references must be regular, non-symlink files. The recorded helper
path/hash must match `source_hashes`, independently of the current installation
path. `expected_helper_sha256` can pin a separately trusted digest. A copied
filename suffix is not evidence. Every matrix is checked against the hash-bound
reference with exact tolerance metadata.

The verifier accepts batches of 2..10 environments; the capture factory supports
1..16. The twelve recorded reference tasks each have N=1. Their serial evidence
cannot satisfy this batch check, which has no N=1 bypass. Verify serial references
and 2..10-environment batches separately.

## Writers and recorded-evidence tests

`BatchEvidence` writes physical arrays and finger feedback. `CameraArchive`
writes camera HDF5/JPEG/PNG observations and requires an explicit shared
`CameraQuality` instance. Its audit interval is 256 fast-path calls by default;
fallback calls use the reference directly and do not advance the interval. An
audit mismatch raises. `BatchContacts` and `ScreenContacts` retain per-environment
routing. `reference_windows=False` is the screen-only default; reference-window
checks require `True` and explicit window and matching functions.

CPU construction tests cover 1, 3 and 16 environments, freshness, guards,
rotations, reports and teardown. Recorded-evidence tests cover all 12 task
fixtures using four PNG views plus every physical/finger row. Supply the baseline
recordings and their matching reference source archive explicitly:

```sh
TOWER_ARCHIVE_ROOT=/path/to/reference-archive \
TOWER_P706_RECORDED_ROOT=/path/to/baseline \
uv run pytest packages/tower-sim/tests/test_evidence.py -k recorded_camera_and_physical_writers
```

Unset archive or baseline inputs produce skips. Missing evidence under an
explicitly supplied root is an error. The checks compare `states.hdf5`, NPZ and
JSON output byte-for-byte with the baseline, and compare JPEG/PNG encodings
between implementations. They also cover contact and collision receipts, scoped
readbacks, SI conversions, settling boundaries, shared quality counters, repeated
guard rejections and tick mismatches. HDF5 and Pillow dependencies ship with
`tower-sim`; the SDK is provisioned separately.

These checks establish data and construction behavior, not acceptance of new
rendered frames or trajectories. The [control and review contract](bench-public.md#control-and-review-contract)
still applies, including its separate command, observation and hardware limits.
