"""Task-relative signed TCP geometry, including approach/carry/recorded samples.

Callers supply ordinary task contact frames. No global world-axis convention or
candidate flag can substitute for that contact geometry. This check does not
certify unobserved motion or replace review of recorded approach phases.
"""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PostureEvidence:
    accepted: bool
    approach_alignment: tuple[float, ...]
    closing_alignment: tuple[float, ...]
    reason: str


def _rotations(transforms):
    value = np.asarray(transforms, dtype=float)
    if value.shape[-2:] == (4, 4):
        if not np.allclose(value[..., 3, :], [0, 0, 0, 1], atol=1e-9, rtol=0):
            raise ValueError("Invalid homogeneous transform")
        value = value[..., :3, :3]
    if value.shape[-2:] != (3, 3) or not np.isfinite(value).all():
        raise ValueError("Expected finite rotation matrices or transforms")
    if not np.allclose(
        value.swapaxes(-1, -2) @ value, np.eye(3), atol=1e-6, rtol=0
    ) or not np.allclose(np.linalg.det(value), 1.0, atol=1e-6, rtol=0):
        raise ValueError("Expected proper rotations")
    return value


def posture_evidence(
    targets, ordinary_targets, *, approach_axis=0, closing_axis=2, minimum_alignment=0.0
) -> PostureEvidence:
    """Require signed ordinary axes at every supplied phase/sample.

    A half-turn about TCP approach preserves approach but reverses finger closing,
    so it fails regardless of candidate metadata. Other task TCP conventions can
    explicitly choose different axes. Ordinary frames may be one frame or a
    matching trajectory; positions/contact planning remain the caller's contract.
    """
    if (
        approach_axis not in (0, 1, 2)
        or closing_axis not in (0, 1, 2)
        or approach_axis == closing_axis
    ):
        raise ValueError("Expected distinct TCP approach and finger-closing axes")
    if not np.isfinite(minimum_alignment) or not 0 <= minimum_alignment < 1:
        raise ValueError("Alignment threshold must be in [0, 1)")
    actual, ordinary = _rotations(targets), _rotations(ordinary_targets)
    approach = np.sum(actual[..., :, approach_axis] * ordinary[..., :, approach_axis], axis=-1)
    closing = np.sum(actual[..., :, closing_axis] * ordinary[..., :, closing_axis], axis=-1)
    ok = bool(
        approach.size
        and np.all(approach > minimum_alignment)
        and np.all(closing > minimum_alignment)
    )
    return PostureEvidence(
        ok,
        tuple(np.ravel(approach)),
        tuple(np.ravel(closing)),
        "ordinary" if ok else "reversed_or_nonordinary_grasp",
    )


def solve_grasp(solve, target, seed, ordinary_target, **geometry):
    """Check task geometry then attempt exactly the supplied target once.

    Numerical rejection is returned unchanged; there is no symmetric retry.
    Use posture_evidence on generated approach/carry and recorded phases as well.
    """
    evidence = posture_evidence(target, ordinary_target, **geometry)
    if not evidence.accepted:
        return {"accepted": False, "joints": None, "reason": evidence.reason}
    return solve(target, seed)
