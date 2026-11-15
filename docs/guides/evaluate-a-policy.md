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

Start a compatible [HTTP policy server](../reference/policy-interface.md#http-compatibility).
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
  --pilot-review /path/to/posture-review.json --policy-url http://localhost:19000 \
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
