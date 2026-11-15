"""Frozen CPU scorer dependency; no scene-generation entry point."""

from __future__ import annotations

import numpy as np

from ...geometry import Rotation


def box_penetration(a, a_dimensions, b, b_dimensions):
    """SAT overlap in metres; nonpositive means separated/touching boxes."""
    ra, rb = (
        Rotation.from_quat(a[3:]).as_matrix(),
        Rotation.from_quat(b[3:]).as_matrix(),
    )
    axes = [*ra.T, *rb.T, *(np.cross(x, y) for x in ra.T for y in rb.T)]
    axes = np.asarray([x / np.linalg.norm(x) for x in axes if np.linalg.norm(x) > 1e-8])
    ar = np.abs(axes @ ra) @ (np.asarray(a_dimensions) / 2)
    br = np.abs(axes @ rb) @ (np.asarray(b_dimensions) / 2)
    return float(np.min(ar + br - np.abs(axes @ (np.asarray(a[:3]) - b[:3]))))
