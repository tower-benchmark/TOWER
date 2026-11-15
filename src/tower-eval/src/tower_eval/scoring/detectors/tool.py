# SPDX-License-Identifier: Apache-2.0
"""Tool acquisition, pushes, return and structural survival."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

import numpy as np

from ..computation import REFERENCE
from ..constants import BLOCK_SIZE_M
from ..geometry import angular_error, corners, fully_clear, inverse, polygon, rotate
from ..rules import DEFAULT_RULES
from .common import first, spans, sustained
from .grasp import carry_mask, hold_mask
from .place import stable_dwell


def tool_plan(spec, layers, rules=DEFAULT_RULES):
    """Resolve the authored scene identities, never controller success events."""
    blocks, transfers = spec["blocks"], spec["transfers"]
    names = [b["name"] for b in blocks]
    if len(set(names)) != len(names):
        raise ValueError("Ambiguous Task 7 block identities")
    stages = []
    for cycle in range(layers - 1):
        layer = cycle + 1
        name = "pusher_upper" if layer == 1 else "pusher_lower"
        if name not in names:
            raise ValueError(f"Missing Task 7 tool: {name}")
        tool = names.index(name)
        if blocks[tool]["tower"] != "tools" or blocks[tool].get("fixed_on_table"):
            raise ValueError("Task 7 requires a free tool")
        pushes = [f"source_l{layer}_{slot}" for slot in (2, 0)]
        extracts = pushes[::-1] if layer == 1 else pushes
        pair = transfers[cycle * 2 : cycle * 2 + 2]
        if [t["block_id"] for t in pair] != extracts or any(
            t["arm"] != cycle or t["source_layer"] != layer or t["destination_layer"] != layer
            for t in pair
        ):
            raise ValueError("Task 7 transfer sequence differs from the authored cycle")
        indices = [names.index(name) for name in pushes]
        if any(
            blocks[i]["tower"] != "source"
            or blocks[i]["layer"] != layer
            or blocks[i]["slot"] not in (-1, 1)
            for i in indices
        ):
            raise ValueError("Task 7 pushes must target the two outer blocks")
        stages.append(
            dict(
                tool=tool,
                arm=1 - cycle,
                targets=indices,
                direction=np.asarray(
                    spec["layout_metadata"]["cycles"][cycle]["direction"], dtype=float
                )
                if rules.tool_direction_from_metadata
                and spec["layout_metadata"].get("evaluation_task", "").startswith("task7-")
                else np.array([-1.0, 0.0, 0.0] if layer == 1 else [0.0, -1.0, 0.0]),
            )
        )
    return stages


def surface_distance(r, index, points):
    dims = np.asarray(r.spec["blocks"][index].get("dimensions_m") or BLOCK_SIZE_M)
    local = rotate(inverse(r.poses[:, index, 3:]), points - r.poses[:, index, :3])
    return np.linalg.norm(np.maximum(np.abs(local) - dims / 2, 0), axis=-1)


def released_mask(r, index, rules):
    return np.all(
        np.stack(
            [
                (r.opening[:, arm] >= rules.open_width_m)
                | (surface_distance(r, index, r.pinch[:, arm]) >= rules.release_clearance_m)
                for arm in (0, 1)
            ]
        ),
        axis=0,
    )


def score_tools(r, layers, rules, report, ordered_samples, final_masks, *, computation=REFERENCE):
    """Append three tool G steps and one return P step per collection cycle."""
    n = len(r.times)
    dims = [b.get("dimensions_m") or BLOCK_SIZE_M for b in r.spec["blocks"]]
    foundation = [
        corners(r.poses[0, j], dims[j])
        for j, b in enumerate(r.spec["blocks"])
        if b["tower"] == "source" and b.get("fixed_on_table")
    ]
    if not foundation:
        raise ValueError("Task 7 source has no fixed foundation")
    # Retain the source centers and roof; displaced but upright supported
    # structure is not automatically a failure merely because D is nonzero.
    base = np.concatenate(foundation)[:, :2]

    hull = polygon(base)
    edges = np.roll(hull, -1, axis=0) - hull
    transfer_ids = {t["block_index"] for t in r.spec["transfers"]}
    for i, block in enumerate(r.spec["blocks"]):
        if block["tower"] != "source" or i in transfer_ids or block.get("fixed_on_table"):
            continue
        delta = r.poses[:, i, None, :2] - hull
        intact = (
            np.all(
                edges[:, 0] * delta[..., 1] - edges[:, 1] * delta[..., 0] >= -1e-9,
                axis=1,
            )
            & (np.abs(r.poses[:, i, 2] - r.poses[0, i, 2]) <= rules.support_height_m)
            & (
                angular_error(r.poses[:, i, 3:], r.poses[0, i, 3:])
                <= np.deg2rad(rules.orientation_deg)
            )
        )
        if (
            rules.tool_collapse_veto
            and r.spec["layout_metadata"].get("evaluation_task") == "task7-3"
        ):
            # Frozen uniform617/bond583 scenes use the approved collapse486
            # criterion. Static COM/footprint displacement is diagnostic only.
            up = rotate(r.poses[:, i, 3:], np.array([0.0, 0.0, 1.0]))
            initial_up = rotate(np.asarray(block["pose"][3:]), np.array([0.0, 0.0, 1.0]))
            tilt = np.arccos(np.clip(up @ initial_up, -1.0, 1.0))
            loss = float(block["pose"][2]) - r.poses[:, i, 2]
            intact = np.logical_and.accumulate(
                (loss < rules.tool_collapse_thickness_fraction * min(dims[i]))
                & (tilt < np.deg2rad(rules.tool_collapse_tilt_deg))
            )
        final_masks.append(intact)
    ownership = []
    stage_reports = report["tool_cycles"] = []

    def event(identifier, group, sample, order, order_sample=None):
        if sample is not None:
            report["events"].append(
                dict(
                    operation=identifier,
                    group=group,
                    sample=sample,
                    time_s=float(r.times[sample]),
                )
            )
            ordered_samples.append((order, sample if order_sample is None else order_sample))

    def window(identifier, target, arm, start, active, end=None):
        if start is None:
            return
        end = n - 1 if end is None else end
        report["operations"].append(
            dict(
                id=identifier,
                target=target,
                arm=arm,
                start=start,
                end=end,
                tool_step=True,
            )
        )
        eligible = active.copy()
        eligible[:start] = False
        eligible[end + 1 :] = False
        for a, b in spans(eligible):
            ownership.append(dict(arm=arm, target=target, start=a, end=b - 1))

    for cycle, stage in enumerate(tool_plan(r.spec, layers, rules)):
        tool, arm, direction = stage["tool"], stage["arm"], stage["direction"]
        name = r.spec["blocks"][tool]["name"]
        near = surface_distance(r, tool, r.pinch[:, arm]) <= rules.approach_margin_m
        attempt = first(sustained(near, r.times, rules.approach_s))
        if attempt is not None:
            attempt = int(np.searchsorted(r.times, r.times[attempt] - rules.approach_s - 1e-8))
        carried = carry_mask(r, tool, arm, rules)
        held = hold_mask(r, tool, arm, rules, carried)
        grasp = first(held & (r.poses[:, tool, 2] - r.poses[0, tool, 2] >= rules.lift_m))
        event(f"{name}:grasp", "G", grasp, cycle * 8)
        active = near & (r.opening[:, arm] < rules.closed_width_m)
        # Acquisition cannot survive a release/regrasp without new carry evidence.
        transported = np.zeros(n, dtype=bool)
        for a, b in spans(active):
            transported[a:b] = np.maximum.accumulate(carried[a:b])
        stage_report: dict[str, Any] = dict(
            tool=name, arm=arm, grasp_sample=grasp, pushes=[], return_sample=None
        )
        stage_reports.append(stage_report)
        tip = r.poses[:, tool, :3] + rotate(r.poses[:, tool, 3:], rules.tool_tip_local_m)
        tool_projection = tip @ direction
        hand_projection = r.pinch[:, arm] @ direction
        base_edge = (np.concatenate(foundation) @ direction).max()
        push_starts, push_ends = [], []
        for push_order, i in enumerate(stage["targets"]):
            block_name = r.spec["blocks"][i]["name"]
            identifier = f"{block_name}:tool_push"
            contact = surface_distance(r, i, tip) <= rules.push_face_margin_m
            attempt_mask = contact & transported
            start = first(sustained(attempt_mask, r.times, rules.approach_s))
            if start is not None:
                start = int(np.searchsorted(r.times, r.times[start] - rules.approach_s - 1e-8))
                push_starts.append(start)
                push_ends.append(int(np.flatnonzero(attempt_mask)[-1]))
            projection = r.poses[:, i, :3] @ direction
            advancing = (
                (np.r_[0.0, np.diff(projection)] > 1e-6)
                & (np.r_[0.0, np.diff(tool_projection)] > 1e-6)
                & (np.r_[0.0, np.diff(hand_projection)] > 1e-6)
            )
            driven = np.zeros(n, dtype=bool)
            for a, b in spans(attempt_mask):
                driven[a:b] = (
                    advancing[a:b]
                    & (projection[a:b] - projection[a] >= rules.minimum_push_m)
                    & (tool_projection[a:b] - tool_projection[a] >= rules.minimum_push_m)
                    & (hand_projection[a:b] - hand_projection[a] >= rules.minimum_push_m)
                )
            recent = np.zeros(n, dtype=bool)
            for a, b in spans(driven):
                recent[
                    a : np.searchsorted(r.times, r.times[b - 1] + rules.push_coast_s, side="right")
                ] = True
            # Expose a graspable end while retaining the block in its source layer.
            front = (corners(r.poses[:, i], dims[i]) @ direction).max(axis=1)
            protruding = (
                (front - base_edge >= rules.tool_protrusion_m)
                & ~fully_clear(r.poses[:, i], dims[i], foundation)
                & (np.abs(r.poses[:, i, 2] - r.poses[0, i, 2]) <= rules.support_height_m)
                & (
                    angular_error(r.poses[:, i, 3:], r.poses[0, i, 3:])
                    <= np.deg2rad(rules.orientation_deg)
                )
            )
            success = first(sustained(protruding & recent, r.times, rules.push_clearance_s))
            stage_report["pushes"].append(
                dict(block=block_name, attempt_sample=start, push_sample=success)
            )
            event(identifier, "G", success, cycle * 8 + push_order + 1)
            window(identifier, i, arm, start, attempt_mask)
        # Return is distinct from acquisition and cannot be credited at t=0.
        released = released_mask(r, tool, rules)
        home = r.poses[0, tool]
        distance = np.linalg.norm(r.poses[:, tool, :3] - home[:3], axis=1)
        q = r.poses[:, tool, 3:]
        q = q / np.linalg.norm(q, axis=1, keepdims=True)
        # The T-shaped tool does not share rectangular-block half-turn symmetry.
        angle = 2 * np.arccos(np.clip(np.abs(q @ (home[3:] / np.linalg.norm(home[3:]))), 0, 1))
        eligible = (
            released
            & np.maximum.accumulate(carried)
            & (distance <= rules.position_m)
            & (angle <= np.deg2rad(rules.orientation_deg))
        )
        verified = stable_dwell(
            eligible,
            r.poses[:, tool],
            r.velocities[:, tool],
            r.times,
            replace(rules, placement_dwell_s=rules.tool_return_dwell_s),
            computation=computation,
        )
        returned = first(verified)
        return_start = None
        if carried.any():
            moving_home = np.r_[False, np.diff(distance) < -1e-6] & np.maximum.accumulate(carried)
            # Begin returning only after the last observed tool–block contact, if any.
            if push_ends:
                moving_home[: max(push_ends) + 1] = False
            return_start = first(moving_home)
            if return_start is None:
                return_start = first(eligible)
        release_start = (
            int(np.searchsorted(r.times, r.times[returned] - rules.tool_return_dwell_s - 1e-8))
            if returned is not None
            else None
        )
        stage_report["return_sample"] = returned
        event(f"{name}:return", "P", returned, cycle * 8 + 3, release_start)
        window(
            f"{name}:grasp",
            tool,
            arm,
            attempt,
            active,
            min(push_starts + [return_start if return_start is not None else n]) - 1,
        )
        window(f"{name}:return", tool, arm, return_start, active)
        final_masks.append(verified)
    return ownership
