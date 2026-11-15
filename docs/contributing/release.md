# Release evidence

Before a release, resolve all owner placeholders, verify locked assets and build
packages and documentation. Attach wheels, source distributions, the public
asset lock and SHA256SUMS through the release workflow.

Prepare these additional attachments for owner review:

- A validation summary naming package and benchmark versions, checks executed,
  pass/fail status, skipped checks, hardware class and known limitations.
- Portability and policy-diagnostic summaries with explicit evidence scope.
  A diagnostic result does not establish posture acceptance or benchmark success.
- A comparison summary binding public scene identities, runtime and checkpoint
  digests, scoring ruleset and the public old-to-new scene map.
- Policy transport parity results, clearly separated from physical evaluation.

Build new public summaries from reviewed evidence. Never attach raw internal
receipts, machine paths, private service addresses or training provenance.
Review every attachment, hash the final bytes, and include them in SHA256SUMS.
The existing release workflow does not automatically collect these summaries;
maintainers must add only reviewed files before the owner publishes a release.

Model weights and model-card licensing remain owner responsibilities.

## Private maintainer checks

With authenticated Hugging Face access to the private archive, run
`uv run python -m scripts.maintainer.archive_assets --destination ~/.cache/tower/archive-assets`
and set `TOWER_ARCHIVE_ASSET_ROOT` to that folder. The helper verifies the frozen
commit, manifest, file hashes and sizes. These assets are used only for archive
reproduction; public tests use `TOWER_ASSET_ROOT`.

GPU checks use the maintainer harness, with no self-hosted workflow in the public
repository. Any future self-hosted runner requires a dedicated runner group with
`restricted_to_workflows: true` limited to approved workflows on `refs/heads/main`,
ephemeral or JIT runners, and approval for all external contributors. Fork PRs
must never run automatically on maintainer GPU hosts.

## Shared IK specification identity

The packaged shared IK specification has provenance-redacted bytes with SHA-256
`3247cd2eed229c367341f26e2d7bf95dca845b94f74b6ade575c345cf32a7911`.
Its canonical identity remains
`efa69beb12a5bb972a2f570a9b2310f9dcfa852a91bd095295b205a66b8a0d03`,
read from the `nero-kinematics` command profile. The runner verifies the packaged
byte hash before binding that canonical identity. Bound values, IK profiles and
command limits are unchanged; request `ik_bounds` must still match the package
file bytes.
