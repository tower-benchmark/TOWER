# Versioning

TOWER versions software, benchmark data, and persisted wire formats separately.

| Surface | Scheme | Release contract |
| --- | --- | --- |
| `nero-kinematics`, `tower-eval`, `tower-sim` | Semantic Versioning (`MAJOR.MINOR.PATCH`) | All three reach `1.0.0` at the public release. Breaking public API changes increment major; compatible features increment minor; fixes increment patch. |
| Benchmark | Calendar Versioning (`YYYY.MM.REVISION`) | `2026.11.1` is the public benchmark revision, using `tower-scene-v2` and a published all-600 physical equivalence map. |
| Wire schemas | `tower-*-vN` | Increment `N` for an incompatible format; do not infer it from a package or benchmark version. |

Current package versions are development versions, not a published 1.0 release.
The package release workflow uses coordinated stable tags `vMAJOR.MINOR.PATCH`;
all three package metadata versions must match the tag. It does not change
versions or dependency constraints itself. Package maintainers must align their
inter-package constraints and the workspace lock before the public release.

Persisted `jenga_*` schema strings remain readable through historical data readers. Renaming the project does not authorize rewriting old recordings,
changing their schema identifiers, or dropping old readers. New formats use the
`tower-*-vN` family. For example, `tower-infer-v2` is a protocol version, not
package version 2.0 or a benchmark revision.

The public repository starts with fresh history. The internal predecessor
2026.11.0 has identical resolved physics and remains internal archival evidence;
its tag and scene files are not part of the public repository. Historical
receipts never substitute for current posture and safety acceptance.

## Release machinery

`.github/workflows/release.yml` runs only on package tag pushes. Branch pushes,
PRs, benchmark tags and asset tags do not start it. It requires matching package
versions and final `docs/release-notes/<version>.md` notes (the `Status: Draft`
marker blocks release), verifies the public HF asset tag against its locked
commit and manifest SHA-256, then builds three wheels and three sdists. The
GitHub Release includes these six distributions, the asset lock and `SHA256SUMS`.
The current lock is `packages/tower-sim/src/tower_sim/data/benchmark/assets.lock.json`; any future
relocation must update the workflow's `ASSET_LOCK` in the same revision.

Builds use isolated build environments and do not install Isaac or run GPU jobs.
There is no PyPI publication step; publication requires maintainer approval.

See the [changelog](../../CHANGELOG.md) and [1.0.0 notes draft](../release-notes/1.0.0.md).

## Scene revision 2026.11.1

The [scene schema](../../packages/tower-sim/src/tower_sim/bench/scene-v2.schema.json)
contains fixed evaluation state, neutral scene identifiers and content-addressed
resources. Scene payloads are external data; the committed
[manifest](../../benchmarks/2026.11.1/data-manifest.json) and
[equivalence table](../../benchmarks/2026.11.1/scene-equivalence.json) bind them.
Only evaluation inputs and content hashes are public. Internal lineage is not
part of the repository or public dataset.

The request remains `tower-public-benchmark-v2`: its scene-selection values identify
fixed cases, and its policy/action semantics are unchanged. Envelopes remain
`tower-public-envelope-v1` and record the selected benchmark, data version and
manifest hash. Case numbers are independent of historical source identifiers.

The complete v1 reader and lossless compatibility view require archival evidence.
Maintainer rechecks accept an explicit `--v1-root`; no archive is discovered from checkout history. Without that explicit source, compatibility checks skip clearly
and cannot produce a successful equivalence receipt. A supplied invalid archive
is an error. Full-output golden values are unchanged; scene identifiers use neutral names.

The new scene manifest binds only new metadata and external evaluation payloads.
All new files have zero host/IP/path leak exemptions. The
[data lock](../../benchmarks/2026.11.1/assets.lock.json) pins the public dataset
at immutable commit `d78e4ac0b2bfc1e10a15d31ac7e7d723be2764f1`. Local verification uses the same hashes as
the public release. Package versions are unchanged. Staging performs no upload.

The public wire stores each fixed state under `episode`. The closed schema retains
`measured_asset_id`, which binds the collision/visual asset, and removes archival
provenance. After hash checks the decoder supplies the runtime asset-closure flag.
The full resolved runtime input is compared for all 600 scenes after the explicit
archive-only dropped-field list and mechanical identifier changes.

Data identity binds payload and metadata hashes plus the upload MANIFEST hash.
The HF repository/revision/tag/directory is a separate locator. Filling those
fields leaves `data_manifest_sha256` and existing GPU receipts valid; changing
`manifest_sha256` is rejected. The locator is not part of the upload.

## Shared IK specification identity

The packaged shared IK specification has provenance-redacted bytes with SHA-256
`3247cd2eed229c367341f26e2d7bf95dca845b94f74b6ade575c345cf32a7911`.
Its canonical identity remains
`efa69beb12a5bb972a2f570a9b2310f9dcfa852a91bd095295b205a66b8a0d03`,
read from the `nero-kinematics` command profile. The runner verifies the packaged
byte hash before binding that canonical identity. Bound values, IK profiles and
command limits are unchanged; request `ik_bounds` must still match the package
file bytes.


## Station distribution and calibration identity

The public station digest is
`54fe7ea5f90b8919f6fece5b2d3e926f9f9f14a0bf7c69659b5e3aa38d474ad9`.
Only this pinned digest maps to the canonical calibration identity
`59d3443b7020a365beb648ebbf428cae750559ca320824ca99d77824d4c26d27`.
The metadata-only re-authoring replaces internal path strings with `redacted`,
including the layer documentation. Every USD field outside the enumerated string
metadata, all numeric arrays, references and texture bytes were compared exactly.
The runtime configuration comparison is exact again; changed configuration
values fail with their JSON pointer. TCP and fit-equivalence calibration pins
retain their canonical identity.

The public asset manifest is independently pinned by its lock. Maintainer GPU
comparison maps only the sealed public manifest digest to the historical asset
identity, after verifying every actual public file. The private proof covers the
required asset subset, unchanged texture/physics data and redacted golden members.
This mapping changes no comparison thresholds or scoring behavior.

## Policy transport revision

The unreleased `tower-public-benchmark-v2` request now requires XPolicyLab
WebSockets. `policy_url` and `--policy-url` keep their names. This is a runtime
transport revision, with no benchmark, data or scene identity change. See the
[Isaac runtime contract](policy-interface.md#isaac-benchmark-runtime) for lifecycle,
response reconstruction, metrics and MME availability.

## Benchmark run schemas

The unreviewed request protocol is
`tower-public-benchmark-protocol-unreviewed-v1`; the reviewed protocol remains
`tower-public-benchmark-protocol-v1`. Only the explicit unreviewed schema skips
maintainer review. Missing or other schemas still require full review.
Aggregate scores use `tower-benchmark-score-v1`; informational posture reports
use `tower-posture-flags-v1` and `tower-ordinary-tcp-reference-v1`.
The benchmark revision `2026.11.1`, ruleset `bench-2026.11`, package versions
and data hashes are unchanged.

Run selection uses `tower-benchmark-run-v1`; retained request bindings use
`tower-benchmark-attempt-v1`. Keep these records with recordings for offline
coverage, attempt lineage and relocatable aggregation. They establish local
content consistency, not signed authenticity or protection against scene selection.
