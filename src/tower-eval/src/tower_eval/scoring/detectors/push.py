# SPDX-License-Identifier: Apache-2.0
"""Directed push contact, motion and complete-footprint clearance detector."""

import numpy as np

from ..computation import REFERENCE
from ..geometry import fully_clear, rotate
from .common import first, spans, sustained


def score_push(
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
    axis = rotate(r.poses[0, i, 3:], [1.0, 0.0, 0.0])[:2]
    axis /= np.linalg.norm(axis)
    base = np.concatenate(foundation)[:, :2] @ axis
    sign = 1 if (r.poses[0, i, :2] - r.pinch[0, arm, :2]) @ axis >= 0 else -1
    projection = r.poses[:, i, :2] @ axis
    departed = projection > base.max() if sign > 0 else projection < base.min()
    recent = np.zeros(n, dtype=bool)
    contact = distances[i] <= rules.approach_margin_m
    contact[:start] = False
    confirmed_contact = sustained(contact, r.times, rules.approach_s)
    hand_projection = r.pinch[:, arm, :2] @ axis
    hand_motion = np.r_[0.0, np.diff(hand_projection)] * sign > 1e-6
    block_motion = np.r_[0.0, np.diff(projection)] * sign > 1e-6
    driven = np.zeros(n, dtype=bool)
    for a, b in spans(contact):
        driven[a:b] = (
            ((hand_projection[a:b] - hand_projection[a]) * sign >= rules.minimum_push_m)
            & ((projection[a:b] - projection[a]) * sign >= rules.minimum_push_m)
            & hand_motion[a:b]
            & block_motion[a:b]
            & confirmed_contact[a:b]
        )
    for a, b in spans(driven):
        recent[a : np.searchsorted(r.times, r.times[b - 1] + rules.push_coast_s, side="right")] = (
            True
        )
    cleared = fully_clear(r.poses[:, i], dims[i], foundation) & departed
    candidate = sustained(cleared, r.times, rules.push_clearance_s) & recent
    candidate[:start] = False
    success = first(candidate)
    phase = "G" if order < planned_g else "P"
    result.update(push_sample=success, phase=phase)
    if success is not None:
        g += phase == "G"
        p += phase == "P"
        report["push_successes_by_arm"][str(arm)] += 1
        ordered_samples.append((order, success))
        report["events"].append(
            dict(
                operation=f"{name}:push",
                group=phase,
                sample=success,
                time_s=float(r.times[success]),
            )
        )
    windows.append(dict(id=f"{name}:push", target=i, arm=arm, start=start, end=n - 1))
    final_masks.append(cleared if success is not None else np.zeros(n, dtype=bool))
    return g, p
