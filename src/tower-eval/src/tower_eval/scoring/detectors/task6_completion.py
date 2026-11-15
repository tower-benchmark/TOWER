# SPDX-License-Identifier: Apache-2.0
"""Task6 TSR: released targets supported above the roof, no recorded falls.

This opt-in offline revision leaves the archived grasp, placement and disturbance
detectors unchanged. It does not alter generation acceptance or robot controls.
"""

from copy import deepcopy
from typing import Any

import numpy as np

from ..constants import BLOCK_SIZE_M
from ..geometry import (
    area,
    corners,
    inside,
    intersection,
    inverse,
    polygon,
    rotate,
)
from ..geometry import geometry as block_geometry
from ..rules import DEFAULT_RULES, task_counts
from .common import first, spans

REVISION = "tower-task6-roof-no-fall-v1"


def scene(recording, task) -> dict[str, Any]:
    r = recording
    number, layers, _, _ = task_counts(task)
    if number != 6 or r.spec.get("layout_metadata", {}).get("catalog_task") != 6:
        raise ValueError("Task6 completion requires a matching recorded Task6 scene")
    blocks = r.spec["blocks"]
    targets = r.spec["transfers"]
    ids = [t["block_index"] for t in targets]
    if len(ids) != layers - 1 or len(set(ids)) != len(ids):
        raise ValueError("Task6 requires exactly one/two distinct extraction targets")
    for t in targets:
        i = t["block_index"]
        if (
            not 0 <= i < len(blocks)
            or blocks[i]["name"] != t["block_id"]
            or blocks[i].get("fixed_on_table")
            or t["arm"] not in (0, 1)
        ):
            raise ValueError("Invalid Task6 target mapping")
    towers = {blocks[i]["tower"] for i in ids}
    if len(towers) != 1:
        raise ValueError("Task6 targets must belong to the same source tower")
    tower = next(iter(towers))
    members = [i for i, b in enumerate(blocks) if b["tower"] == tower]
    bases = [i for i in members if blocks[i].get("fixed_on_table")]
    structural = [i for i in members if i not in bases and i not in ids]
    if not bases or not structural:
        raise ValueError("Source tower requires fixed bases and remaining structure")
    roof_layer = max(blocks[i]["layer"] for i in structural)
    roof = [i for i in structural if blocks[i]["layer"] == roof_layer]
    dims = [b.get("dimensions_m") or BLOCK_SIZE_M for b in blocks]
    return dict(
        targets=ids,
        bases=bases,
        structural=structural,
        roof=roof,
        members=members,
        dims=dims,
    )


def support_graph(r, info, sample, rules) -> dict[str, Any]:
    """Reuse the frozen support geometry, without pose goals or 2 mm drift gates."""
    poses = r.poses[sample]
    members = info["members"]
    dims = info["dims"]
    geometry = {i: block_geometry(poses[i], dims[i]) for i in members}
    edges = {}
    diagnostics = {}
    for i in members:
        if i in info["bases"]:
            edges[i] = []
            continue
        footprint, bottom, _ = geometry[i]
        patches = []
        supporters = []
        for j in members:
            if i == j or poses[j, 2] >= poses[i, 2]:
                continue
            lower, _, top = geometry[j]
            if abs(bottom - top) > rules.support_height_m:
                continue
            patch = intersection(footprint, lower)
            if area(patch) > 1e-9:
                patches.append(patch)
                supporters.append(j)
        fraction = (
            sum(area(p) for p in patches) / area(footprint)
            if patches and area(footprint) > 1e-9
            else 0.0
        )
        com_supported = bool(
            patches
            and inside(
                poses[i, :2],
                polygon(np.concatenate(patches)),
                rules.support_com_margin_m,
            )
        )
        # A loaded structural beam may be balanced by blocks above it. Its own
        # COM need not lie over its bearing patch in isolation. Require a
        # geometric bearing path for the existing structure, and use the
        # entire-recording fall detector for collapse. Newly placed targets
        # retain the frozen support-area and COM checks.
        supported = (
            fraction >= rules.minimum_support_fraction and com_supported
            if i in info["targets"]
            else bool(patches)
        )
        edges[i] = supporters if supported else []
        diagnostics[i] = dict(
            support_fraction=float(fraction),
            com_supported=com_supported,
            support_geometry=bool(supported),
            bottom_m=float(bottom),
            candidate_supporters=supporters,
        )
    rooted = set(info["bases"])
    for _ in members:
        rooted.update(i for i in members if any(j in rooted for j in edges[i]))
    roof_rooted = set(info["roof"]) & rooted
    above_roof = set(roof_rooted)
    # Paths through extraction targets only: a target on a lower structural
    # tier cannot become a roof placement through an unrelated support path.
    for _ in info["targets"]:
        above_roof.update(
            i for i in info["targets"] if i in rooted and any(j in above_roof for j in edges[i])
        )
    return dict(edges=edges, rooted=rooted, above_roof=above_roof, diagnostics=diagnostics)


def controlled_by_hand(r, index, dims, rules):
    """Geometric co-motion proxy; independent of archived GSR/force credit."""
    previous = np.maximum(0, np.searchsorted(r.times, r.times - rules.disturbance_motion_window_s))
    controlled = np.zeros(len(r.times), dtype=bool)
    for arm in (0, 1):
        local = rotate(
            inverse(r.poses[:, index, 3:]),
            r.pinch[:, arm] - r.poses[:, index, :3],
        )
        distance = np.linalg.norm(
            np.maximum(np.abs(local) - np.asarray(dims[index]) / 2, 0), axis=1
        )
        relative = rotate(r.inverse_flange[:, arm], r.poses[:, index, :3] - r.pinch[:, arm])
        follows = np.linalg.norm(relative - relative[previous], axis=1) <= rules.relative_range_m
        controlled |= (
            (distance <= rules.grasp_surface_distance_m)
            & (r.opening[:, arm] < rules.closed_width_m)
            & follows
        )
    return controlled


def fall_events(r, info, rules):
    """Recorded descent beyond half a measured block thickness.

    Structural blocks cannot lose their support level. For extraction targets,
    normal hand-controlled lowering is exempt, as is settling onto the roof.
    An uncontrolled drop below roof level is a fall even if later recovered.
    This is a pose/motion proxy, not measured contact-force classification.
    """
    events = []
    roof_top = max(float(corners(r.poses[0, i], info["dims"][i])[:, 2].max()) for i in info["roof"])
    for i in info["structural"] + info["targets"]:
        target = i in info["targets"]
        controlled = (
            controlled_by_hand(r, i, info["dims"], rules)
            if target
            else np.zeros(len(r.times), dtype=bool)
        )
        threshold = float(info["dims"][i][2]) / 2
        below_roof = (
            (
                corners(r.poses[:, i], info["dims"][i])[..., 2].min(axis=1)
                < roof_top - rules.support_height_m
            )
            if target
            else np.ones(len(r.times), dtype=bool)
        )
        for a, b in spans(~controlled):
            # Include the preceding sample so a drop in the first released
            # frame is not discarded when the co-motion proxy becomes false.
            anchor = max(0, a - 1)
            z = r.poses[anchor:b, i, 2]
            peak = np.maximum.accumulate(z)
            mask = (peak - z > threshold) & below_roof[anchor:b]
            mask[: a - anchor] = False
            at = first(mask)
            if at is not None:
                k = anchor + at
                events.append(
                    dict(
                        block=r.spec["blocks"][i]["name"],
                        block_index=i,
                        role="extraction_target" if target else "structural",
                        sample=k,
                        time_s=float(r.times[k]),
                        descent_m=float(peak[at] - z[at]),
                        threshold_m=threshold,
                        reference_sample=anchor + int(np.argmax(z[: at + 1])),
                    )
                )
                break
    return sorted(events, key=lambda x: (x["sample"], x["block"]))


def completion(r, task, rules=DEFAULT_RULES) -> dict[str, Any]:
    info = scene(r, task)
    graph = support_graph(r, info, -1, rules)
    falls = fall_events(r, info, rules)
    targets = []
    for i in info["targets"]:
        released = True
        for arm in (0, 1):
            local = rotate(
                inverse(r.poses[-1, i, 3:]),
                r.pinch[-1, arm] - r.poses[-1, i, :3],
            )
            distance = np.linalg.norm(
                np.maximum(np.abs(local) - np.asarray(info["dims"][i]) / 2, 0)
            )
            released &= bool(
                r.opening[-1, arm] >= rules.open_width_m or distance >= rules.release_clearance_m
            )
        quiet = bool(
            np.linalg.norm(r.velocities[-1, i, :3]) <= rules.linear_velocity_m_s
            and np.linalg.norm(r.velocities[-1, i, 3:]) <= rules.angular_velocity_rad_s
        )
        on_top = i in graph["above_roof"]
        targets.append(
            dict(
                block=r.spec["blocks"][i]["name"],
                block_index=i,
                on_top=on_top,
                released=released,
                quiet=quiet,
                placed=bool(on_top and released and quiet),
                **graph["diagnostics"][i],
            )
        )
    unsupported = [
        r.spec["blocks"][i]["name"] for i in info["structural"] if i not in graph["rooted"]
    ]
    return dict(
        passed=all(t["placed"] for t in targets) and not falls and not unsupported,
        targets=targets,
        falls=falls,
        no_fall_entire_recording=not falls,
        unsupported_structural_blocks=unsupported,
        support_edges={
            r.spec["blocks"][i]["name"]: [r.spec["blocks"][j]["name"] for j in js]
            for i, js in graph["edges"].items()
        },
        roof_blocks=[r.spec["blocks"][i]["name"] for i in info["roof"]],
        final_time_s=float(r.times[-1]),
        fall_scope="entire recorded trajectory",
        support_height_tolerance_m=rules.support_height_m,
        minimum_support_fraction=rules.minimum_support_fraction,
        support_com_margin_m=rules.support_com_margin_m,
        fall_threshold="strictly more than half each measured block thickness",
        no_grasp_credit_or_exact_order_required=True,
        no_absolute_goal_pose_or_cumulative_2mm_5deg_gate=True,
        no_added_placement_dwell=True,
    )


def finalize(r, task, rules, original):
    result = deepcopy(original)
    result.update(
        protocol=REVISION,
        scoring_revision=REVISION,
        original_protocol=original["protocol"],
        original_scores=original["scores"],
        original_terminal_geometry=original.get("terminal_geometry"),
        recorded_terminal_status=r.status.get("status"),
        scoring_scope="final released roof support and no fall anywhere in recording",
    )
    if not original.get("valid"):
        return result
    gate = completion(r, task, rules)
    s = int(gate["passed"])
    result.update(
        task6_completion=gate,
        terminal_geometry=all(t["placed"] for t in gate["targets"]),
    )
    result["scores"].update(s=s, E=original["scores"]["C"] + 1 - s)
    scores = result["scores"]
    result["odr"] = dict(
        D=scores["D"],
        E=scores["E"],
        value=scores["D"] / scores["E"] if scores["E"] else None,
        denominator_rule="legacy C + 1 - s",
        undefined_reason=None if scores["E"] else "zero legacy denominator",
    )
    return result
