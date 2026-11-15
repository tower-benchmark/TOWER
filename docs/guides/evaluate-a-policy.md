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
The data locator still awaits owner pinning. Until it is pinned, the command
fails with `data release not yet pinned; pass --data-root <dir>`.
Obtain the exact release directory from the owner; do not substitute scene files.
With that directory already available, use:

```sh
uv run python tools/simulator/benchmark_assets.py download --assets /path/to/assets --data-root /path/to/benchmark-2026.11.1
uv run python tools/simulator/benchmark_assets.py verify --assets /path/to/assets --data-root /path/to/benchmark-2026.11.1
uv run python tools/simulator/benchmark.py verify --benchmark 2026.11.1 --assets /path/to/assets --data-root /path/to/benchmark-2026.11.1
```

With `--data-root`, download fetches physical assets and verifies the existing
complete evaluation-data directory rather than downloading its contents.
Verification checks `MANIFEST.json`, the exact file set, sizes and hashes.
After downloading through a pinned locator, use the default data directory as
`--data-root` in these verification commands and in the request below.
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
Request preparation requires a reviewed posture record
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
  --benchmark 2026.11.1 --data-root /path/to/benchmark-2026.11.1 \
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
cache and the SDK CUDA kernel cache. Warm-up and seeding cover both. Each
simulator uses a private CUDA cache selected by its own `CUDA_CACHE_PATH`;
the parent process and policy service environment are unchanged. Cache files
are specific to the GPU and driver; rebuild both after either changes, and never distribute them with the
benchmark.

Use the configured simulator interpreter and the same license consent as for
other simulator runs. Start serially, with new output and export directories:

```sh
python tools/simulator/benchmark_portable.py smoke \
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
exported caches. Both the run output and the new export destination must be new directories:

```sh
python tools/simulator/benchmark_portable.py smoke \
  --assets /path/to/assets --data-root /path/to/benchmark-2026.11.1 \
  --task cross-transfer-2 --seeds 0 --gpu 0 --accept-eula \
  --output /path/to/warmup-2 \
  --shader-cache-seed /path/to/cache-pass-1/cache \
  --shader-cache-digest /path/to/cache-pass-1/digest.sha256 \
  --shader-cache-warmup /path/to/cache-pass-2
```

Proceed only after a serial pass exits successfully with a passing smoke result
and `source_run_passed: true` in its exported `warmup.json`. If the second pass
fails, stop and retain both failures for investigation. Every pass uses the
original camera and image checks; no failed result becomes an accepted trial.

Use the successful pass's export for concurrent `benchmark.py run`,
`benchmark_portable.py run` or `smoke`, or `benchmark_diagnostic.py` commands.
For example, after the second pass:

```sh
--shader-cache-seed /path/to/cache-pass-2/cache \
--shader-cache-digest /path/to/cache-pass-2/digest.sha256
```

Each worker copies and hash-checks the complete Kit and CUDA seed before
renderer startup. Its CUDA cache is private to that worker. The parent records
warm-up digests after worker exit and removes ordinary seeded workers' private
copies of both caches. Use separate output directories and available GPUs. A digest mismatch rejects startup. The invalid-image guard,
including the all-zero initial-image rejection, stays strict with no image
fallback.

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
