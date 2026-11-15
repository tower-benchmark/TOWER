"""Metadata-only public station bytes retain the calibrated physical identity.

Only the pinned distribution digest maps to the canonical calibration digest.
Every other digest is unchanged and must pass the caller's exact comparison.
"""

PUBLIC_STATION_SHA256 = "54fe7ea5f90b8919f6fece5b2d3e926f9f9f14a0bf7c69659b5e3aa38d474ad9"
CANONICAL_STATION_SHA256 = "59d3443b7020a365beb648ebbf428cae750559ca320824ca99d77824d4c26d27"
PUBLIC_ASSET_MANIFEST_SHA256 = "2e1bc0be31699060dbaaedf1e61f6ebb35d2bdaca36b2c6eea2786635adabe2c"
CANONICAL_ASSET_MANIFEST_SHA256 = "a2065d0b4872b579096caa1b3e4e2efd1b6ac95d4fd8df8b130a743c87207432"


def station_identity(digest):
    return CANONICAL_STATION_SHA256 if digest == PUBLIC_STATION_SHA256 else digest


def asset_manifest_identity(digest):
    """Identity for the proven metadata-only public subset and its public recordings."""
    return CANONICAL_ASSET_MANIFEST_SHA256 if digest == PUBLIC_ASSET_MANIFEST_SHA256 else digest
