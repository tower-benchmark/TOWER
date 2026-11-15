"""Canonical IK identity, independently bound to redacted distribution bytes."""

import hashlib

from nero_kinematics.profiles import load_profile

from tower_sim.backend.resources import DATA

REDACTED_SHARED_SPEC_SHA256 = "3247cd2eed229c367341f26e2d7bf95dca845b94f74b6ade575c345cf32a7911"


def shared_spec_identity():
    """Verify package provenance redaction and return the unchanged command identity."""
    path = DATA / "benchmark/runtime/shared/shared-ik-spec.json"
    if hashlib.sha256(path.read_bytes()).hexdigest() != REDACTED_SHARED_SPEC_SHA256:
        raise ValueError("Changed redacted shared IK specification")
    return load_profile("command")["source_sha256"]
