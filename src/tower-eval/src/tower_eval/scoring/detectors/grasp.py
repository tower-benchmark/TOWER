# SPDX-License-Identifier: Apache-2.0
"""Closed-hand carrying and bilateral effort evidence."""

from __future__ import annotations

import numpy as np

from ..computation import REFERENCE
from ..constants import BLOCK_SIZE_M
from ..geometry import corners, fully_clear, inverse, rotate
from .common import first, spans, sustained
from .place import stable_dwell


def carry_mask(recording, index, arm, rules):
    """Closed-hand transport evidence independent of the grasp force proxy."""
    r = recording
    center = r.poses[:, index, :3]
    relative = rotate(r.inverse_flange[:, arm], center - r.pinch[:, arm])
    dimensions = np.asarray(r.spec["blocks"][index].get("dimensions_m") or BLOCK_SIZE_M)
    local = rotate(inverse(r.poses[:, index, 3:]), r.pinch[:, arm] - center)
    surface_distance = np.linalg.norm(np.maximum(np.abs(local) - dimensions / 2, 0), axis=1)
    near = surface_distance <= rules.grasp_surface_distance_m
    candidate = sustained(near & (r.opening[:, arm] < rules.closed_width_m), r.times, rules.hold_s)
    valid = np.zeros(len(r.times), dtype=bool)
    for k in np.flatnonzero(candidate):
        a = int(np.searchsorted(r.times, r.times[k] - rules.hold_s - 1e-8))
        if r.times[k] - r.times[a] < rules.hold_s - 1e-8:
            continue
        if (
            np.linalg.norm(np.ptp(relative[a : k + 1], axis=0)) <= rules.relative_range_m
            and np.linalg.norm(center[k] - center[a]) >= rules.minimum_carry_m
            and np.linalg.norm(r.pinch[k, arm] - r.pinch[a, arm]) >= rules.minimum_carry_m
        ):
            valid[k] = True
    return valid


def hold_mask(recording, index, arm, rules, carried=None):
    """Prior carrying plus bilateral effort for the archived grasp proxy."""
    r = recording
    carried = carry_mask(r, index, arm, rules) if carried is None else carried
    weak = r.effort[:, arm].min(axis=1) < rules.jaw_effort_n
    valid = np.zeros(len(r.times), dtype=bool)
    for k in np.flatnonzero(carried & ~weak):
        a = int(np.searchsorted(r.times, r.times[k] - rules.hold_s - 1e-8))
        gaps = [r.times[min(a + y, k)] - r.times[a + x] for x, y in spans(weak[a : k + 1])]
        valid[k] = (
            np.mean(~weak[a : k + 1]) + 1e-8 >= rules.effort_fraction
            and max(gaps, default=0) <= rules.effort_gap_s + 1e-8
        )
    return valid


def score_transfer(
    r,
    t,
    order,
    start,
    dims,
    foundation,
    distances,
    rules,
    report,
    ordered_samples,
    final_masks,
    planned_g,
    geometry,
    detector,
    result,
    step_order,
    *,
    computation=REFERENCE,
):
    i, arm, name = t["block_index"], t["arm"], t["block_id"]
    n = len(r.times)
    windows = report["operations"]
    g = p = 0
    carried = carry_mask(r, i, arm, rules)
    carried[:start] = False
    carried_before = np.maximum.accumulate(carried)
    result["carry_sample"] = first(carried)
    held = hold_mask(r, i, arm, rules, carried)
    if detector.extraction:
        source_tower = r.spec["blocks"][i]["tower"]
        source_top = max(
            corners(r.poses[0, j], dims[j])[:, 2].max()
            for j, b in enumerate(r.spec["blocks"])
            if b["tower"] == source_tower
        )
        above_tower = corners(r.poses[:, i], dims[i])[..., 2].min(axis=1) > source_top + 1e-6
        held &= fully_clear(r.poses[:, i], dims[i], foundation) | above_tower
    else:
        held &= r.poses[:, i, 2] - r.poses[0, i, 2] >= rules.lift_m
    held[:start] = False
    grasp = first(held)
    result["grasp_sample"] = grasp
    window = dict(id=f"{name}:grasp", target=i, arm=arm, start=start, end=n - 1)
    windows.append(window)
    if grasp is not None:
        g += 1
        ordered_samples.append((step_order, grasp))
        report["events"].append(
            dict(
                operation=window["id"],
                group="G",
                sample=grasp,
                time_s=float(r.times[grasp]),
            )
        )
    # Placement is scored independently of the acquisition proxy. A missed
    # G detector must not erase a directly observed released placement.
    placement_start = (
        grasp + 1
        if grasp is not None
        else first(
            geometry[i]
            & carried_before
            & (np.arange(n) > start)
            & (np.linalg.norm(r.poses[:, i, :3] - r.poses[0, i, :3], axis=1) > rules.disturbance_m)
        )
    )
    if placement_start is None or placement_start >= n:
        final_masks.append(np.zeros(n, dtype=bool))
        return g, p
    # Both hands must release or be clear of the target.
    released = np.ones(n, dtype=bool)
    for hand in (0, 1):
        local = rotate(inverse(r.poses[:, i, 3:]), r.pinch[:, hand] - r.poses[:, i, :3])
        clearance = np.linalg.norm(np.maximum(np.abs(local) - np.asarray(dims[i]) / 2, 0), axis=1)
        released &= (r.opening[:, hand] >= rules.open_width_m) | (
            clearance >= rules.release_clearance_m
        )
    eligible = geometry[i] & released & carried_before
    eligible[:placement_start] = False
    verified = stable_dwell(
        eligible, r.poses[:, i], r.velocities[:, i], r.times, rules, computation=computation
    )
    placement = first(verified)
    result["place_sample"] = placement
    tail = next((a for a, b in spans(eligible) if b == n), None)
    result["terminal_released_geometry_s"] = (
        float(r.times[-1] - r.times[tail]) if tail is not None else 0.0
    )
    result["terminal_position_error_m"] = float(
        np.linalg.norm(r.poses[-1, i, :3] - np.asarray(t["goal"])[:3])
    )
    window["end"] = placement_start - 1
    # A dropped or misplaced block is also released. Goal geometry decides
    # placement credit, not whether a later subtask may disturb this block.
    release = first(released & (np.arange(n) >= placement_start))
    windows.append(
        dict(
            id=f"{name}:place",
            target=i,
            arm=arm,
            start=placement_start,
            end=n - 1,
            active_end=release if release is not None else n - 1,
        )
    )
    if placement is not None:
        p += 1
        release_start = int(
            np.searchsorted(r.times, r.times[placement] - rules.placement_dwell_s - 1e-8)
        )
        ordered_samples.append((step_order + 1, release_start))
        report["events"].append(
            dict(
                operation=f"{name}:place",
                group="P",
                sample=placement,
                time_s=float(r.times[placement]),
            )
        )
    final_masks.append(verified)
    return g, p
