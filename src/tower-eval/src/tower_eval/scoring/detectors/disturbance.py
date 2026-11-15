# SPDX-License-Identifier: Apache-2.0
"""Operation attribution and renewed-contact disturbance evidence."""

from __future__ import annotations

import numpy as np

from ..constants import BLOCK_SIZE_M
from ..geometry import boxes_touch, inverse, rotate
from .common import first, spans


def target_intervals(r, windows, distances, rules):
    """Each arm's observed target interaction, independent of scoring windows."""
    intervals = []
    for arm in (0, 1):
        ordered = sorted((w for w in windows if w["arm"] == arm), key=lambda w: w["start"])
        for k, w in enumerate(ordered):
            following = next(
                (v["start"] for v in ordered[k + 1 :] if v["start"] > w["start"]), len(r.times)
            )
            w["active_end"] = min(w.get("active_end", w["end"]), following - 1)
            a, b = w["start"], w["active_end"]
            engaged = distances[w["target"]][a : b + 1] <= rules.release_clearance_m
            for start, end in spans(engaged):
                intervals.append(
                    dict(arm=arm, target=w["target"], start=a + start, end=a + end - 1)
                )
    return intervals


def renewed_contact(r, index, assigned, previous, rules):
    """Contact from a later manipulation can interrupt a target's own inertia."""
    foreign = (assigned >= 0) & (assigned != index)
    hand_moving = np.linalg.norm(r.pinch - r.pinch[previous], axis=-1) > rules.disturbance_motion_m
    points = (
        r.pinch[:, :, None]
        if r.finger_corners is None
        else r.finger_corners.reshape(len(r.times), 2, -1, 3)
    )
    dims = r.spec["blocks"][index].get("dimensions_m") or BLOCK_SIZE_M
    local = rotate(
        inverse(r.poses[:, index, None, None, 3:]), points - r.poses[:, index, None, None, :3]
    )
    gap = np.linalg.norm(np.maximum(np.abs(local) - np.asarray(dims) / 2, 0), axis=-1).min(axis=-1)
    preceding = np.maximum(np.arange(len(r.times)) - 1, 0)
    touching = gap <= rules.disturbance_contact_m
    # A new operation label does not turn an existing, continuous contact into
    # a fresh impact. Require contact onset or the participant starting motion.
    hand_onset = ~touching[preceding] | ~hand_moving[preceding]
    contact = np.any(foreign & hand_moving & touching & hand_onset, axis=1)
    # The carried or pushed target can strike a former target even when the
    # fingers do not touch it directly. Static scene contacts are insufficient.
    for other in np.unique(assigned[foreign]):
        moving = (
            np.linalg.norm(r.poses[:, other, :3] - r.poses[previous, other, :3], axis=1)
            > rules.disturbance_motion_m
        )
        samples = np.flatnonzero(np.any(assigned == other, axis=1) & moving & ~contact)
        if len(samples):
            other_dims = r.spec["blocks"][other].get("dimensions_m") or BLOCK_SIZE_M
            queries = np.unique(np.r_[samples, preceding[samples]])
            touching = np.zeros(len(r.times), dtype=bool)
            touching[queries] = boxes_touch(
                r.poses[queries, index],
                dims,
                r.poses[queries, other],
                other_dims,
                rules.disturbance_contact_m,
            )
            contact[samples] = touching[samples] & (
                ~touching[preceding[samples]] | ~moving[preceding[samples]]
            )
    return contact


def disturbance_events(r, windows, rules, ownership=None):
    """Count once per block/operation, excluding continuing target motion."""
    active = np.full((len(r.times), 2), -1, dtype=int)
    for w in windows if ownership is None else ownership:
        active[w["start"] : w.get("active_end", w["end"]) + 1, w["arm"]] = w["target"]
    assigned = np.full_like(active, -1)
    for w in windows:
        assigned[w["start"] : w["end"] + 1, w["arm"]] = w["target"]
    assigned = np.where(active >= 0, active, assigned)
    previous = np.maximum(0, np.searchsorted(r.times, r.times - rules.disturbance_motion_window_s))
    events = []
    for i, block in enumerate(r.spec["blocks"]):
        if (
            block.get("fixed_on_table")
            or block.get("is_tool")
            or block.get("tower") in ("tool", "tools")
        ):
            continue
        center = r.poses[:, i, :3]
        owned = np.any(active == i, axis=1)
        target_motion = owned.copy()
        moving = np.linalg.norm(center - center[previous], axis=1) > rules.disturbance_motion_m
        bouts = []
        for a, b in spans(moving):
            if bouts and r.times[a] - r.times[bouts[-1][1] - 1] < rules.disturbance_quiet_s:
                bouts[-1][1] = b
            else:
                bouts.append([a, b])
        # Rest or renewed contact from another manipulation ends the exclusion
        # for a former target's own continuing movement.
        recontact = None
        for a, b in bouts:
            if not owned[a:b].any():
                continue
            if recontact is None:
                recontact = renewed_contact(r, i, assigned, previous, rules) & ~owned
            samples = np.arange(a, b)
            last_target = np.maximum.accumulate(np.where(owned[a:b], samples, -1))
            last_contact = np.maximum.accumulate(np.where(recontact[a:b], samples, -1))
            target_motion[a:b] |= (last_target >= 0) & (last_target > last_contact)
        for w in windows:
            if w["target"] == i:
                continue
            a, b = w["start"], w["end"]
            for start, end in spans(~target_motion[a : b + 1]):
                baseline = a + start
                # A fresh impact may cross 10 mm in its first recorded frame.
                # Reference the preceding pose only for that new interaction,
                # not for ordinary release (which must discard carried motion).
                fresh_impact = (
                    baseline > 0
                    and not owned[baseline - 1]
                    and recontact is not None
                    and recontact[baseline]
                )
                reference = baseline - 1 if fresh_impact else baseline
                distance = np.linalg.norm(center[baseline : a + end] - center[reference], axis=1)
                k = first(distance > rules.disturbance_m)
                if k is not None:
                    events.append(
                        dict(
                            operation=w["id"],
                            block=block["name"],
                            sample=baseline + k,
                            time_s=float(r.times[baseline + k]),
                            translation_m=float(distance[k]),
                            baseline_sample=reference,
                        )
                    )
                    break
    return sorted(events, key=lambda e: (e["sample"], e["operation"], e["block"]))
