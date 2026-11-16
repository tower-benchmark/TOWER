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

Diagnostic runs are the supported way to test a policy now. Run
`tools/simulator/benchmark_diagnostic.py` with `--task`, one scene `--seed`,
`--assets`, `--data-root`, `--package`, `--policy-url`, `--ik-python`, a new
`--output` and `--accept-eula`, using the Isaac interpreter. This entrypoint
selects diagnostic mode itself; no `--diagnostic` or `--pilot-review` is needed.
It records per-episode status and `result.json`, with `score.json` when the
physical evidence is complete and scoring succeeds. These results remain
diagnostic and do not approve a pilot or qualify a formal trial. Formal runs
additionally require the posture review (`--pilot-review`); its per-task
contact-frame reference and review tooling are not yet published and will follow.

Formal request preparation requires a reviewed posture record
(`tower-no-reversed-grasp-review-v1`): planned approach/carry and recorded
approach-phase review, no reverse 180-degree grasp, and preserved task contact
geometry, each bound to its evidence file by SHA-256.
[The exact-flip tripwire](../reference/bench-policy.md) cannot replace this review.
The record must match `task` and `package_digest`, with true
`planned_approach_carry_passed`, `recorded_approach_carry_passed`,
`reverse_180_grasp_absent` and `task_contact_geometry_preserved`. Its
`planned_trajectory` and `recorded_approach_carry` objects bind actual reviewed
files by absolute `path` and `sha256`; preparation does not create a review.
The current CLI spells this required input `--pilot-review`.
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

A startup error such as `Initial scene has four all-zero camera views` or
`Renderer failed to advance`, or an episode status of `infrastructure_failure`,
indicates an infrastructure rejection, not a policy result. Inspect the log,
`error.json`, camera diagnostics and each `simulation/seed-<id>/status.json`
that was written; startup can fail before episode status exists. Incomplete
evidence (`evidence_complete: false`) must not be counted as a policy failure.
After resolving the infrastructure problem, rerun exactly the rejected scene
ids, retaining the rejected attempt and its inputs, logs and evidence. For
formal batches, repeat `prepare-request` above with the same task, benchmark,
data, package, policy settings and review, set `--seeds` to only those ids, and
choose fresh `--output` and `--run-output` directories; check and run that new
request. For diagnostics, repeat the diagnostic command for each rejected
`--seed` with a fresh `--output`. Do not rerun completed episodes or overwrite
the old request. There is no automatic retry. A recorded worker exit with
`phase: "teardown"` and complete evidence is different: verification succeeded,
the command exits zero, and the native return code and signal remain in the
result's `worker_exit` record.
