# SPDX-License-Identifier: Apache-2.0
"""Numerically stable compatibility kernels for all scoring geometry.

The broadcast quaternion kernel and SciPy footprint kernel intentionally retain
their historical operation order; rewriting either changes archived float bytes.
Block half-turn symmetry describes object geometry, never allowed robot grasps."""

from __future__ import annotations

import itertools

import numpy as np
from scipy.spatial import ConvexHull, QhullError
from scipy.spatial.transform import Rotation as Rotation


def polygon(points):
    points = np.unique(np.asarray(points, dtype=float), axis=0)
    if len(points) < 3:
        return np.empty((0, 2))
    try:
        return points[ConvexHull(points).vertices]
    except QhullError:
        return np.empty((0, 2))


def area(poly):
    if len(poly) < 3:
        return 0.0
    return float(
        abs(np.dot(poly[:, 0], np.roll(poly[:, 1], 1)) - np.dot(poly[:, 1], np.roll(poly[:, 0], 1)))
        / 2
    )


def inside(point, poly, margin=0.0):
    if len(poly) < 3:
        return False
    edges = np.roll(poly, -1, axis=0) - poly
    delta = np.asarray(point) - poly
    cross = edges[:, 0] * delta[:, 1] - edges[:, 1] * delta[:, 0]
    return bool(np.all(cross >= -margin * np.linalg.norm(edges, axis=1) - 1e-12))


def intersection(subject, clip):
    result = list(np.asarray(subject))
    for a, b in zip(clip, np.roll(clip, -1, axis=0)):
        old, result = result, []
        if not old:
            break
        edge = b - a

        def side(p, a=a, edge=edge):
            d = p - a
            return edge[0] * d[1] - edge[1] * d[0]

        previous = old[-1]
        for current in old:
            ps, cs = side(previous), side(current)
            if (ps >= 0) != (cs >= 0):
                result.append(previous + (current - previous) * ps / (ps - cs))
            if cs >= 0:
                result.append(current)
            previous = current
    return np.asarray(result).reshape(-1, 2)


def geometry(pose, dimensions):
    corners = np.asarray(list(itertools.product((-1, 1), repeat=3))) * np.asarray(dimensions) / 2
    corners = Rotation.from_quat(pose[3:]).apply(corners) + pose[:3]
    return polygon(corners[:, :2]), float(corners[:, 2].min()), float(corners[:, 2].max())


def angle(a, b):
    difference = Rotation.from_quat(a).inv() * Rotation.from_quat(b)
    return float(
        min(
            difference.magnitude(),
            (difference * Rotation.from_euler("z", 180, degrees=True)).magnitude(),
        )
    )


def rotate(q, points):
    q = np.asarray(q, dtype=float)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    if not np.isfinite(q).all() or np.any(norm < 1e-8):
        raise ValueError("Invalid quaternion")
    q = q / norm
    t = 2 * np.cross(q[..., :3], points)
    return points + q[..., 3:] * t + np.cross(q[..., :3], t)


def inverse(q):
    out = np.array(q, copy=True)
    out[..., :3] *= -1
    return out


def angular_error(q, reference):
    """Rectangular blocks are invariant under 180-degree local-axis rotations."""
    q = np.asarray(q) / np.linalg.norm(q, axis=-1, keepdims=True)
    reference = np.asarray(reference) / np.linalg.norm(reference)
    variants = (Rotation.from_quat(reference) * Rotation.from_quat(np.eye(4))).as_quat()
    dots = np.max(np.abs(q @ variants.T), axis=-1)
    return 2 * np.arccos(np.clip(dots, 0, 1))


def corners(poses, dimensions):
    local = np.asarray(list(itertools.product((-1, 1), repeat=3))) * np.asarray(dimensions) / 2
    return poses[..., None, :3] + rotate(poses[..., None, 3:], local)


def fully_clear(poses, dimensions, foundation_corners):
    """Separating-axis test of the entire block against the source footprint."""
    base = np.concatenate(foundation_corners)[:, :2]
    block = corners(poses, dimensions)[..., :2]
    # Foundation edges plus moving block edges form the complete 2-D SAT set.
    from scipy.spatial import ConvexHull

    hull = base[ConvexHull(base).vertices]
    edges = np.roll(hull, -1, axis=0) - hull
    fixed_axes = np.column_stack((-edges[:, 1], edges[:, 0]))
    block_axes = rotate(poses[:, None, 3:], np.eye(3)[None, :, :])[..., :2]
    axes = np.concatenate(
        (
            np.broadcast_to(fixed_axes, (len(poses), *fixed_axes.shape)),
            np.stack((-block_axes[..., 1], block_axes[..., 0]), axis=-1),
        ),
        axis=1,
    )
    lengths = np.linalg.norm(axes, axis=-1)
    axes = axes / np.maximum(lengths[..., None], 1e-12)
    bp = np.einsum("nvd,nad->nva", block, axes)
    fp = np.einsum("vd,nad->nva", base, axes)
    separated = (bp.min(axis=1) >= fp.max(axis=1) + 1e-6) | (
        fp.min(axis=1) >= bp.max(axis=1) + 1e-6
    )
    return np.any(separated & (lengths > 1e-8), axis=1)


def boxes_touch(a, a_dimensions, b, b_dimensions, margin):
    """Oriented-box contact proximity using all 15 separating axes."""
    result = np.zeros(len(a), dtype=bool)
    a_half, b_half = np.asarray(a_dimensions) / 2, np.asarray(b_dimensions) / 2
    close = np.linalg.norm(a[:, :3] - b[:, :3], axis=1) <= (
        np.linalg.norm(a_half) + np.linalg.norm(b_half) + np.sqrt(3) * margin
    )
    indices = np.flatnonzero(close)
    if not len(indices):
        return result
    a, b = a[indices], b[indices]
    a_axes = rotate(a[:, None, 3:], np.eye(3)[None])
    b_axes = rotate(b[:, None, 3:], np.eye(3)[None])
    cross_axes = np.cross(a_axes[:, :, None], b_axes[:, None]).reshape(-1, 9, 3)
    axes = np.concatenate((a_axes, b_axes, cross_axes), axis=1)
    lengths = np.linalg.norm(axes, axis=-1)
    axes /= np.maximum(lengths[..., None], 1e-12)
    offset = np.abs(np.einsum("ni,nki->nk", a[:, :3] - b[:, :3], axes))
    radius = (
        np.abs(np.einsum("nki,nji->nkj", axes, a_axes)) @ a_half
        + np.abs(np.einsum("nki,nji->nkj", axes, b_axes)) @ b_half
    )
    result[indices] = np.all((offset - radius <= margin) | (lengths < 1e-8), axis=1)
    return result
