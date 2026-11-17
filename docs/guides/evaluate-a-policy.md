# Evaluate a policy

The public benchmark is **2026.11.1**, using `tower-scene-v2`. Use its matching
runtime and data identities; see [versioning](../reference/versioning.md).

## Download and verify assets

Run from the checkout root after [installation](../getting-started/install.md).
Keep physical assets and evaluation data outside the checkout. The downloader
uses immutable commits and manifest digests and needs no Hugging Face login:

```sh
uv run python tools/simulator/benchmark_assets.py download --assets /path/to/assets
```

This downloads physical assets and calls `tower_sim.bench.data_release.download`
for evaluation data, stored by default in `/path/to/assets/benchmark-2026.11.1`.
The default is an anonymous, pinned and verified download from TOWER-Assets
at revision `d78e4ac0b2bfc1e10a15d31ac7e7d723be2764f1` (tag `benchmark-2026.11.1`), as recorded in the release lock.
No token or login is needed. Verify an existing download without network access:

```sh
uv run python tools/simulator/benchmark_assets.py verify --assets /path/to/assets
uv run python tools/simulator/benchmark.py verify --benchmark 2026.11.1 --assets /path/to/assets
```

With `--data-root`, download fetches physical assets and verifies the existing
complete evaluation-data directory rather than downloading its contents.
The first verification command checks physical assets only. The second also
verifies evaluation data at `<assets>/benchmark-2026.11.1`, including
`MANIFEST.json`, the exact file set, sizes and hashes. No `--data-root` is needed
for that default location; supply it to verify a different existing data directory.
The request below passes the evaluation-data directory explicitly.
A checksum mismatch is an error. See [asset terms](../../LICENSE-ASSETS).

## Check the environment and prepare a request

Install [Isaac](../isaac-sim.md) and native IK separately. Build the three wheels
and release manifest using the [runtime build instructions](../reference/bench-public.md),
starting with `export PYTHONDONTWRITEBYTECODE=1`. Both interpreters need the
private installed package directory on `PYTHONPATH`. Keep the release-bound
checkout clean, then check the interpreter:

```sh
uv run python tools/simulator/benchmark.py check-environment --isaac-python /path/to/isaac/python
```

On hosts without root, follow the [X11 library recipe](../reference/bench-public.md#x11-system-libraries)
and use its process-scoped library path for this check and benchmark commands.

Use **one runtime process per policy server**. Start the XPolicyLab server in its policy environment, following
[OpenWAM_TOWER deployment](https://github.com/tower-benchmark/XPolicyLab/blob/tower-1.0.3/policy/OpenWAM_TOWER/README.md):

```sh
cd third_party/XPolicyLab
python setup_policy_server.py --config_path policy/OpenWAM_TOWER/deploy.yml \
  --overrides port=19000 package_path=/path/to/package.json
```

Run the benchmark commands below from the TOWER checkout. Install
`tower-eval[xpolicylab]` in the runtime environment and initialize the pinned
submodule with `GIT_LFS_SKIP_SMUDGE=1 git submodule update --init`.
See the [runtime transport contract](../reference/policy-interface.md#isaac-benchmark-runtime).
Supply its package identity file; TOWER does not download or start model weights.

## Run the benchmark

Use the Isaac interpreter while the policy server is running:

```sh
/path/to/isaac/python tools/simulator/benchmark_run.py run \
  --task cross-transfer-2 --assets /path/to/assets \
  --data-root /path/to/assets/benchmark-2026.11.1 \
  --package /path/to/package.json --policy-url ws://localhost:19000 \
  --ik-python /path/to/ik/python --output /path/to/new-run --accept-eula
```

This evaluates all 50 released scenes under `bench-2026.11`, with no
`--pilot-review`. The hash-bound protocol identifies the qualification as
`unreviewed`. Scores use the unchanged benchmark rules. Automatic TCP posture
flags are reported separately and never change scores or exit codes.

Use `--scenes 0-49` or a subset such as `--scenes 3,7,10-12`. Sequential
batches default to one scene; `--batch-size` accepts 1–10. The command stops at
the first failed batch and prints scene ids not yet run. `--keep-going` runs
the remaining batches. Neither mode retries automatically.

The immutable `run.json` records all selected scenes before preparation. Each
`attempts/batch-<id>` retains its start, bound inputs and completion or error record.
Inputs are in `requests/batch-<id>` and recordings in
`runs/batch-<id>/simulation/seed-<id>`. Each completed scene retains its original
`score.json` and gains `posture-flags.json`. The top-level `score.json` aggregates
complete episodes and lists infrastructure failures in `excluded`.

For independent shards, use disjoint `--scenes` selections and fresh outputs,
then combine their complete run bundles with:

```sh
uv run python tools/simulator/benchmark_run.py aggregate \
  --runs /path/to/shard-1 /path/to/shard-2 --output /path/to/aggregate
```

Aggregation rejects duplicate scenes and mixed task, package, benchmark,
ruleset or qualification. Keep the entire run directory, including `run.json`, `attempts/` and `runs/`.
Retained input bytes are addressed by hash, so the bundle can be moved without
rewriting historical paths. Aggregating a batch directory includes its original
run's full selected set and attempt history.

A retry uses `--scenes` and a new output with a distinct run id. Keep and disclose
both attempts. Duplicate scenes are always rejected, including excluded plus
valid attempts; arbitrary reruns cannot be combined into a clean full-task score.
An attributed replacement policy is deferred. There is no latest-wins selection.

Coverage lists requested, attempted, scored, excluded and not-run scenes and their
sources. `requested_complete` means all selected scenes have eligible scores;
`full_task_coverage` additionally requires all 50. A complete subset remains a
subset. Metrics are conditional on eligible scored episodes; exclusions keep
the frozen denominators unchanged. Transport and some policy-contract errors can
be exclusions, so coverage disclosure cannot prevent deliberate selection or
prove an unbiased all-task result. Report coverage and attempts with the metrics.
Only parent-qualified durable completion admits a batch. Nonzero teardown alone
does not disqualify complete evidence; execution-failed batches are not salvaged.

## Debug one scene

Use `tools/simulator/benchmark_diagnostic.py` with one `--seed` and the same
asset, policy and runtime flags. It writes `diagnostic-result.json`,
`result.json` and, when evidence is complete, `score.json`. Its results remain
diagnostic and do not qualify a pilot.

## Maintainer posture qualification

Maintainers may supply `--pilot-review` to qualify a reviewed run. A review is
not required for a public benchmark score. The reviewed path remains fail-closed.

Reviewed request preparation requires a reviewed posture record
(`tower-no-reversed-grasp-review-v1`): planned approach/carry and recorded
approach-phase review, no reverse 180-degree grasp, and preserved task contact
geometry, each bound to its evidence file by SHA-256.
[The exact-flip tripwire](../reference/bench-policy.md) cannot replace this review.
The record must match `task` and `package_digest`, with true
`planned_approach_carry_passed`, `recorded_approach_carry_passed`,
`reverse_180_grasp_absent` and `task_contact_geometry_preserved`. Its
`planned_trajectory` and `recorded_approach_carry` objects bind actual reviewed
files by absolute `path` and `sha256`; preparation does not create a review.
For this qualification, the CLI spells the required input `--pilot-review`.
The same file must also contain the policy's `package_digest`, a
`shared_spec_sha256` matching the canonical identity `efa69beb12a5bb972a2f570a9b2310f9dcfa852a91bd095295b205a66b8a0d03` (nero `command` profile `source_sha256`); see [shared IK specification identity](../reference/versioning.md#shared-ik-specification-identity), and true
`passed`, `native_inference_qualified`, `scene_reference_review_passed` and
`shared_ik_parity_passed`; preparation rejects it otherwise.

```sh
uv run python tools/simulator/benchmark_portable.py prepare-request \
  --benchmark 2026.11.1 --data-root /path/to/assets/benchmark-2026.11.1 \
  --assets /path/to/assets --task two-block-transfer-2 \
  --seeds 0 --gpu 0 --package /path/to/policy-package.json \
  --pilot-review /path/to/posture-review.json --policy-url ws://localhost:19000 \
  --ik-python /path/to/ik/python --isaac-python /path/to/isaac/python \
  --output /path/to/new-request --run-output /path/to/new-run
uv run python tools/simulator/benchmark.py check-request --request /path/to/new-request/request.json
uv run python tools/simulator/benchmark.py run --request /path/to/new-request/request.json --isaac-python /path/to/isaac/python
```

Use new output directories. The scene ids selected by `--seeds` above are
evaluation inputs: neutral case numbers 0–49, corresponding to `scene-000`
through `scene-049` within the task. Preparation resolves these released scenes;
it does not create new scenes. Use each command's `--help` for all flags.
Keep resolved configuration, input identities, complete recordings and failure
status with results. Use [offline scoring](offline-scoring.md) for saved recordings.

## Warm the renderer cache before concurrent starts

Benchmark workers enable render-completion synchronization on every start.
A first-time start also needs two different caches: the Kit renderer shader
cache and the SDK CUDA kernel cache. Warm-up and seeding cover both. When seed
or warm-up options are given, each simulator uses a private CUDA cache selected
by its own `CUDA_CACHE_PATH`; the parent process and policy service environment
are unchanged. Cache files are specific to the GPU and driver; rebuild both
after either changes.
TOWER ships no cache.

Use the configured simulator interpreter and the same license consent as for
other simulator runs. Start serially, with new output and export directories:

```sh
/path/to/isaac/python tools/simulator/benchmark_portable.py smoke \
  --assets /path/to/assets --data-root /path/to/benchmark-2026.11.1 \
  --task cross-transfer-2 --seeds 0 --gpu 0 --accept-eula \
  --output /path/to/warmup-1 --shader-cache-warmup /path/to/cache-pass-1
```

The first fully cold pass can fail during camera initialization. Its command
still fails, and `error.json` preserves the error before native shutdown.
Warm-up exports compilation output even after such a failure: `cache/kit` and
`cache/cuda`, a sorted SHA-256 file list in `digest.sha256`, and `warmup.json`
recording whether the run passed. An exported cache is not a passing scene result.

If the first pass failed, make one defined second serial pass using both
exported caches. Both the run output and the new export destination must be new
directories:

```sh
/path/to/isaac/python tools/simulator/benchmark_portable.py smoke \
  --assets /path/to/assets --data-root /path/to/benchmark-2026.11.1 \
  --task cross-transfer-2 --seeds 0 --gpu 0 --accept-eula \
  --output /path/to/warmup-2 \
  --shader-cache-seed /path/to/cache-pass-1/cache \
  --shader-cache-digest /path/to/cache-pass-1/digest.sha256 \
  --shader-cache-warmup /path/to/cache-pass-2
```

Proceed only after a serial pass exits successfully with a passing smoke result
and `source_run_passed: true` in its exported `warmup.json`. A seed without this
positive status is rejected before a child starts, unless another warm-up export
is explicitly requested. If the second pass fails, stop and retain both failures
for investigation. Every pass uses the original camera and image checks; no
failed result becomes an accepted trial.

In verification on one GPU/driver combination, a seed from a passing
`cross-transfer-2` warm-up passed initial-image validation for all 12 task
variants concurrently, with six starts on each of two cards. Warm up on each
GPU/driver combination you use. Every subsequent run still performs its own
strict image checks.

Use the successful pass's export for concurrent `benchmark.py run`,
`benchmark_portable.py run` or `smoke`, or `benchmark_diagnostic.py` commands.
For example, after the second pass:

```sh
--shader-cache-seed /path/to/cache-pass-2/cache \
--shader-cache-digest /path/to/cache-pass-2/digest.sha256
```

Each worker copies and hash-checks the complete Kit and CUDA seed before
renderer startup. Its CUDA cache is private to that worker. The parent records
warm-up digests after worker exit and removes the worker's private copies of
both caches after a completed export, as well as after ordinary seeded runs.
Use separate output directories and available GPUs. A digest mismatch rejects
startup. The invalid-image guard, including the all-zero initial-image
rejection, stays strict with no image fallback.

For troubleshooting, `simulation/private-shader-cache.json` records the private
Kit and CUDA paths, seed digest, and CUDA cache size and file count before
startup, together with the CUDA cache environment settings. The Kit cache is
under the worker's portable root at `<output>-kit/cache`; the simulator's
`CUDA_CACHE_PATH` points to `<output>-kit/cuda-cache`. File counts are evidence,
not readiness criteria. Exposure initialization has a bounded render-only wait
and writes `capture/renderer-readiness.json`; a timeout remains a startup failure.

Startup-only validation with four processes on each of two cards passed 8/8
starts when both caches were seeded. Concurrent starts with both caches empty
passed 0/8; a fresh Kit cache with a retained CUDA cache also passed 0/8. A serial
start with a seeded Kit cache but an empty CUDA cache passed 0/1. The last two
controls reached and failed the unchanged all-zero initial-image check. These
checks supplied no policy actions; warm-up is not an accepted evaluation trial.

## Rerun infrastructure rejections

A startup error such as `Initial scene has four all-zero camera views`,
`Renderer dispatcher did not advance` or `Renderer returned before scheduled
frame delivery`, or an episode status of `infrastructure_failure`, indicates an
infrastructure rejection, not a policy result. The delivery guard also rejects
`Renderer failure in delivered camera views` and `Renderer failure in independent
camera views`. Inspect the log, `error.json`, and each
`simulation/seed-<id>/status.json` that was written; startup can fail before
episode status exists. Per-frame rejection diagnostics and raw pixels are saved
relative to the run output in
`simulation/seed-<id>/camera-frames.hdf5.diagnostics/<frame>.json` and the matching
`.npz`. For diagnostics, inspect the run-level `diagnostic-result.json` too.
Incomplete evidence (per-seed `status.json` `evidence_complete: false`) must not
be counted as a policy failure.
After resolving the infrastructure problem, rerun exactly the rejected scene
ids, retaining the rejected attempt and its inputs, logs and evidence. For
formal batches, repeat `prepare-request` above with the same task, benchmark,
data, package, policy settings and review, set `--seeds` to only those ids, and
choose fresh `--output` and `--run-output` directories; check and run that new
request. For diagnostics, repeat the diagnostic command for each rejected
`--seed` with a fresh `--output`. Do not rerun completed episodes or overwrite
the old request. There is no automatic retry. If an identical rerun reproduces
the same rejection, report it with the retained evidence instead of retrying.

A recorded worker exit with `phase: "teardown"` means the worker crossed the
durable completion boundary. The command exits zero only if all evidence checks
also pass. The native return code and signal remain in the result's `worker_exit`
record; a zero return code has `phase: null`. Only negative subprocess return
codes are decoded into `signal`. A shell-wrapped interpreter can report `128+N`
for signal N; that positive code is retained with `signal: null`.
