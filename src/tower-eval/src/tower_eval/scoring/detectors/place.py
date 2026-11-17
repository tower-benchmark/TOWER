# SPDX-License-Identifier: Apache-2.0
"""Released stable placement and authored task contact geometry."""

from __future__ import annotations

import numpy as np

from ..computation import REFERENCE
from ..constants import BLOCK_SIZE_M
from ..geometry import angular_error, area, corners, inside, intersection, inverse, polygon, rotate
from ..geometry import geometry as block_geometry


def stable_dwell(mask, poses, velocities, times, rules, *, computation=REFERENCE):
    """Continuous released placement, rejecting drift, rotation and late collapse."""
    quiet = (
        mask
        & (np.linalg.norm(velocities[:, :3], axis=1) <= rules.linear_velocity_m_s)
        & (np.linalg.norm(velocities[:, 3:], axis=1) <= rules.angular_velocity_rad_s)
    )
    result = np.zeros(len(times), dtype=bool)
    anchor = None
    for k in range(len(times)):
        if not quiet[k]:
            anchor = None
            continue
        if (
            anchor is None
            or np.linalg.norm(poses[k, :3] - poses[anchor, :3]) > rules.stable_translation_m
            or computation.angular_error(poses[k, 3:], poses[anchor, 3:])
            > np.deg2rad(rules.stable_rotation_deg)
        ):
            anchor = k
        result[k] = times[k] - times[anchor] >= rules.placement_dwell_s - 1e-8
    return result


def support_mask(r, target, supporters, dimensions, candidate, rules):
    """Support area and COM over the convex hull of current contact patches."""
    supported = np.zeros(len(r.times), dtype=bool)
    quiet = (np.linalg.norm(r.velocities[:, target, :3], axis=1) <= rules.linear_velocity_m_s) & (
        np.linalg.norm(r.velocities[:, target, 3:], axis=1) <= rules.angular_velocity_rad_s
    )
    supporters = tuple(supporters)
    blocks = [target, *supporters]
    previous_poses = None
    previous_supported = False
    for k in np.flatnonzero(candidate & quiet):
        # Within this call, dimensions, supporters and rules are fixed. Identical
        # pose bytes therefore imply the same geometric decision. Keep only the
        # last sample; do not quantize poses or retain state across calls.
        pose_bytes = r.poses[k, blocks].tobytes()
        if pose_bytes == previous_poses:
            supported[k] = previous_supported
            continue
        footprint, bottom, _ = block_geometry(r.poses[k, target], dimensions[target])
        patches = []
        for j in supporters:
            foot, _, top = block_geometry(r.poses[k, j], dimensions[j])
            if abs(bottom - top) <= rules.support_height_m:
                patch = intersection(footprint, foot)
                if area(patch) > 1e-9:
                    patches.append(patch)
        if patches and area(footprint) > 1e-9:
            supported[k] = sum(area(patch) for patch in patches) / area(
                footprint
            ) >= rules.minimum_support_fraction and inside(
                r.poses[k, target, :2], polygon(np.concatenate(patches)), rules.support_com_margin_m
            )
        previous_poses = pose_bytes
        previous_supported = supported[k]
    return supported


def placement_geometry(r, targets, task, rules, *, computation=REFERENCE):
    """Authored position/orientation, actual slot walls and intact supporters."""
    dims = [b.get("dimensions_m") or BLOCK_SIZE_M for b in r.spec["blocks"]]
    masks = {}
    for t in targets:
        i, goal = t["block_index"], np.asarray(t["goal"])
        delta = r.poses[:, i, :3] - goal[:3]
        masks[i] = (
            (np.linalg.norm(delta[:, :2], axis=1) <= rules.position_m)
            & (np.abs(delta[:, 2]) <= (rules.roof_height_m if task == 6 else rules.position_m))
            & (angular_error(r.poses[:, i, 3:], goal[3:]) <= np.deg2rad(rules.orientation_deg))
        )
    initial_ok = {}
    for j, _b in enumerate(r.spec["blocks"]):
        initial_ok[j] = (
            np.linalg.norm(r.poses[:, j, :3] - r.poses[0, j, :3], axis=1) <= rules.position_m
        ) & (
            angular_error(r.poses[:, j, 3:], r.poses[0, j, 3:]) <= np.deg2rad(rules.orientation_deg)
        )
    result = {}
    target_ids = set(masks)
    for t in targets:
        i = t["block_index"]
        if task in (1, 2, 6):
            tower = "destination" if task in (1, 2) else r.spec["blocks"][i]["tower"]
            foundation = [
                j
                for j, block in enumerate(r.spec["blocks"])
                if block["tower"] == tower and block.get("fixed_on_table")
            ]
            if not foundation:
                raise ValueError("Destination has no foundation")
            base = polygon(
                np.concatenate([corners(r.poses[0, j], dims[j])[:, :2] for j in foundation])
            )
            edges = np.roll(base, -1, axis=0) - base
            delta = r.poses[:, i, None, :2] - base
            in_base = np.all(
                edges[:, 0] * delta[..., 1] - edges[:, 1] * delta[..., 0] >= -1e-9, axis=1
            )
            good = (
                in_base
                & (
                    np.abs(r.poses[:, i, 2] - t["goal"][2])
                    <= (rules.roof_height_m if task == 6 else rules.support_height_m)
                )
                & (
                    angular_error(r.poses[:, i, 3:], t["goal"][3:])
                    <= np.deg2rad(rules.orientation_deg)
                )
            )
            if task in (1, 2):
                lower = [
                    u["block_index"]
                    for u in targets
                    if u["destination_layer"] == t["destination_layer"] - 1
                ]
                supporters = lower or foundation
            else:
                lower = [
                    u["block_index"]
                    for u in targets
                    if u["destination_layer"] == t["destination_layer"] - 1
                ]
                roof_layer = max(b["layer"] for b in r.spec["blocks"] if b["tower"] == tower)
                supporters = lower or [
                    j
                    for j, b in enumerate(r.spec["blocks"])
                    if b["tower"] == tower and b["layer"] == roof_layer
                ]
            # Physical support comes from actual poses, not authored XY goals.
            good &= computation.support_mask(r, i, supporters, dims, good, rules)
        else:
            good = masks[i].copy()
        if task in (5, 6):
            tower = rules.insertion_tower if task == 5 else r.spec["blocks"][i]["tower"]
            for j, block in enumerate(r.spec["blocks"]):
                if block["tower"] == tower and j not in target_ids:
                    if task == 5:
                        good &= initial_ok[j]
                    else:
                        # Disturbance is not structural failure: an upright roof
                        # can shift on its supports while remaining intact.
                        center_delta = r.poses[:, j, None, :2] - base
                        within_base = np.all(
                            edges[:, 0] * center_delta[..., 1] - edges[:, 1] * center_delta[..., 0]
                            >= -1e-9,
                            axis=1,
                        )
                        if rules.untouched_translation_m is not None:
                            translation = np.linalg.norm(
                                r.poses[:, j, :3] - r.poses[0, j, :3], axis=1
                            )
                            quat = r.poses[:, j, 3:] / np.linalg.norm(
                                r.poses[:, j, 3:], axis=1, keepdims=True
                            )
                            rotation = 2 * np.arccos(np.clip(np.abs(quat @ quat[0]), 0, 1))
                            within_base = (
                                np.maximum.accumulate(translation) <= rules.untouched_translation_m
                            ) & (
                                np.maximum.accumulate(rotation)
                                <= np.deg2rad(rules.untouched_rotation_deg)
                            )
                        good &= (
                            within_base
                            & (
                                np.abs(r.poses[:, j, 2] - r.poses[0, j, 2])
                                <= rules.support_height_m
                            )
                            & (
                                angular_error(r.poses[:, j, 3:], r.poses[0, j, 3:])
                                <= np.deg2rad(rules.orientation_deg)
                            )
                        )
            if task == 5:
                walls = [
                    j
                    for j, b in enumerate(r.spec["blocks"])
                    if b["tower"] == tower
                    and b["layer"] == t["destination_layer"]
                    and (
                        (j not in target_ids and not b.get("fixed_on_table"))
                        if rules.insertion_walls_by_membership
                        else "side" in b["name"]
                    )
                ]
                if len(walls) != 2:
                    raise ValueError("Insertion slot must have two authored side blocks")
                for j in walls:
                    inv = inverse(r.poses[:, j, 3:])
                    local = rotate(
                        inv[:, None, :], corners(r.poses[:, i], dims[i]) - r.poses[:, j, None, :3]
                    )
                    sign = np.sign(
                        rotate(
                            inverse(r.poses[0, j, 3:]),
                            np.asarray(t["goal"])[:3] - r.poses[0, j, :3],
                        )[1]
                    )
                    good &= (
                        np.min(sign * local[..., 1], axis=1)
                        >= dims[j][1] / 2 - rules.slot_clearance_m
                    )
        result[i] = good
    return result
