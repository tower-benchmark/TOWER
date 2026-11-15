# Cross-revision benchmark comparisons

These are maintainer tools. They do not create evaluation scenes, approve posture,
change command/observation bounds, or replace recorded approach/carry/contact review.

## Identities and compatibility

`gpu_reference_contract.json` is a new reference document, outside the immutable
goldens. `reference_identity` binds the original benchmark revision, the original
`SOURCE_MANIFEST.json` SHA-256, the golden corpus manifest SHA-256 (which binds
all archive hashes), the twelve expert NPZ hashes/seeds, and the original recorded
hardware/runtime profile. The source digest always means the **original**
manifest bytes, never a implementation build.

`contract` describes the equal benchmark/data/physics inputs: benchmark and data
version, immutable data digests, task order, simulation/recording rates, drive
and camera modes, and hardware/runtime profile. Its immutable data includes the
asset-release manifest, protocol, scene index, all frozen scene documents and
physics resources. Aggregate digests hash UTF-8 compact sorted JSON mappings
of repository-relative POSIX path to file SHA-256. `scenes_sha256` covers every
file under `simulator/benchmark-202611/scenes`; `physics_resources_sha256` covers
`.json`, `.urdf`, `.yaml`, `.yml` under its `runtime` and the archived pinned
`data/*.json`. These are stable **reference resource keys**, not implementation
installation paths. Moved resources must retain byte bindings or supply an
explicit, reviewed equivalence mapping; changing this contract is not an
implementation-only comparison. `test_contract_lineage` verifies the document
against the original source/data/receipt bytes.

New implementation run summaries use `schema: tower-gpu-baseline-v2` with:

```json
{
  "schema": "tower-gpu-baseline-v2",
  "reference_identity": {"...": "the immutable reference identity"},
  "contract": {"...": "verified actual benchmark/data/physics inputs"},
  "candidate_identity": {
    "schema": "tower-gpu-candidate-v2",
    "implementation": "tower-sim",
    "commit": "<40 lowercase hex characters>",
    "wheel_sha256": {"tower_sim-<version>.whl": "<SHA-256>"},
    "dependency_sha256": {"<dependency artifact name>": "<SHA-256>"},
    "data_version": "benchmark-2026.11.1",
    "benchmark_version": "2026.11.1",
    "frozen_components": []
  },
  "trials": []
}
```

Wheel and dependency maps must be nonempty and contain actual artifact digests.
The runner must verify the installed distribution matches those artifacts,
and report its real commit; an editable tree with unbound changes is not a
release identity. Dependencies include numerical/native runtime inputs; a
version string alone is not a hash. Per-task v2 summaries use the same envelope
with exactly one trial. Trials carry task, seed, expert hash, trace hash,
samples, duration, four camera filenames, completion and both terminal flags.
They must **not** contain the legacy `source_manifest_sha256` field.

`read_summary` also reads unchanged `tower-gpu-baseline-v1` run summaries and
the committed `tower-p0-gpu-reference-v1` receipt (`repeat=True` selects its
repeat pass). It normalizes them in memory as `tower-gpu-legacy-v1-identity`.
Their original source digest, seed and expert hashes must match the reference.
Commit/wheel/dependency identities are `null`: v1 never recorded them. The
hardware/runtime provenance is explicitly the original archived receipt, **not a new
measurement of a legacy run**. Thus v1 input is trusted historical evidence
from that campaign, not a way to attest a new environment. No old summary or
golden is rewritten. `run --implementation frozen` remains the default and
keeps the original execution, collection and v1 output format.

`compare` emits `tower-gpu-comparison-v2`, recording `reference_identity`,
`baseline_identity`, `candidate_identity` and `contract`. Reference/data/physics
fields must equal the pinned document and each other; implementation commits and
artifact digests may differ. Seeds, expert trajectories, timestamps, sample
counts, duration and terminal identities must match exactly. Each trace is
hash-checked and validated against its own summary, so two matching but false
summaries do not pass. Physical/image formulas and tolerances are unchanged.
Numerical/image divergence returns `passed: false`; invalid identity/data or
malformed evidence raises a field-specific error. `acceptance_eligible` is false
when either side used frozen component substitutions; such a comparison only
bisects a divergence, is never an acceptance result, and the CLI refuses it
without `--diagnostic`. The implementation contract is reported by the runner; the
harness independently checks the asset release `manifest.json` digest, and the
Runner review must cover how the remaining data digests are verified. Null camera PSNR means
identical pixels. Every protocol task is required by the CLI.

Resume has a separate purpose: it must not mix implementations within a run.
Frozen resume retains the original source-manifest check. Implementation resume
requires the full current implementation identity, substitutions and contract to
match, plus valid stored traces/cameras; a v1 task cannot resume as tower-sim.

## Replay runner interface

`gpu_runner.py` defines the CPU-importable Protocol. The entry point is
`tower_sim.bench.gpu_baseline.create_runner() -> CandidateRunner`. The runtime implements the command-only runner in `bench.replay` with the checkout
binding in `scripts/maintainer/candidate_replay.py`. Build the three workspace
wheels into `dist/` before running; wheel payloads are checked against the
loaded implementation. Missing dependencies fail directly, without a fallback.

The lifecycle runs in a fresh worker process for each task:

1. `identity(assets, gpu, frozen_components, reference_root)` performs CPU-only
   verification of actual artifact/data hashes, physics options and the runtime
   profile. It returns `candidate_identity` and `contract`. It must not start
   Kit or a GPU context. The harness validates the result against the reference
   before admission, then revalidates it in each worker and when aggregating.
2. The harness verifies the archived command NPZ hash, task/seed, finite
   strictly increasing timestamps starting at zero, and `(N,16)` action shape.
3. `setup(ReplayRequest)` constructs the selected scene/runtime. Request fields
   include task, seed, expert hash, assets/output paths, GPU, shared contract
   and explicit substitutions. Setup owns cleanup if construction raises.
4. `PreparedReplay.replay(CommandTrajectory)` verifies the first command
   against the initial scene (the frozen path uses absolute `2e-6`, zero
   relative tolerance), executes without a policy server, and persists
   `trace.npz` (`time`, `qpos`, `blocks`) and four first-frame RGB PNGs. It returns
   `ReplayEvidence` with camera filenames and exact completion/terminal flags.
   The runtime must preserve its full status/evidence/source receipts as well.
5. The harness validates and writes the task summary **before** `close()`;
   Kit's fast close can exit the process. `close()` runs in `finally`.

Per-component substitution is a diagnostic hook:

```sh
python scripts/maintainer/gpu_baseline.py run --implementation tower-sim \
  --assets "$ASSETS" --expert-root "$EXPERT_ROOT" --output "$IMPLEMENTATION_RUN" \
  --gpu "$ADMITTED_GPU" --reference-root "$FROZEN_CHECKOUT" \
  --frozen-component cameras --frozen-component observation
```

Allowed names are `bootstrap`, `command_mapping`, `observation`, `cameras`,
`scoring`, `early_stop`. The runtime implements the actual injection boundaries.
Only explicitly selected components may use the verified frozen checkout;
all other components use the installed runtime. Unsupported combinations fail before
execution. Identity records the sorted exact list; reference bytes must be
verified before loading any substitute. No failure-based substitution is
allowed. A gate without substitutions uses an empty list; a passing diagnostic with
substitutions is **not** full implementation acceptance. GPU admission and locks
still apply to future authorized executions; execution requires explicit authorization.

```sh
python scripts/maintainer/gpu_baseline.py compare "$RUN_A" "$IMPLEMENTATION_RUN" \
  --output "$COMPARISON_JSON"
```

## Exact scorer routing

The exact scorer retains its verified explicit archive reference lookup. The frozen
runtime itself is not part of this distribution; maintainers reproduce it
from the private archive. Set `FROZEN_CHECKOUT` to the verified reference
checkout from that archive, independently of golden fixtures and the implementation
library:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  "$ARCHIVED_PYTHON" scripts/maintainer/check_golden_exact.py \
  --reference-root "$FROZEN_CHECKOUT" --candidate-scorer bench_2026_11 \
  --receipt "$EXACT_RECEIPT"
```

The loader verifies the original manifest digest and every member, then loads
the scorer in a root-specific private namespace. It never redirects implementation
imports to the reference checkout. `bench_2026_11` uses the independent
campaign loader and `bench-2026.11` scoring API. Both scorers process the same
24 archives; the existing complete online-stream checks remain enabled.

`--root` selects external archives with fallback to
the fixture directory. `--fixture-root` independently selects the golden
manifest, archives and online snapshots (default `tests/golden/benchmark-2026.11`).
No fixture lookup needs the frozen runtime. Optional JSON receipts contain
only content identities and exact result digests, so default and explicit
reference roots produce identical receipts in the archived environment.

Where the private archive is available, local route tests use its real reference
checkout. Otherwise the test uses a temporary fixture tree containing the
unchanged, manifest-verified bundle.
The test makes the in-tree bundle unavailable and rejects calls to the old
loader while exercising the external route. Synthetic comparison fixtures are
generated in pytest temporary directories, never under `tests/golden/`.
