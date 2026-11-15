# SPDX-License-Identifier: Apache-2.0
"""Task4 final-state success, with the frozen push/ODR event detectors.

This opt-in revision does not modify historical runtime capsules. Clearance
uses the existing full-block/source-footprint SAT geometry, independently of
approach, recognized push, prescribed direction, or prescribed push order.
"""

from copy import deepcopy
from typing import Any

import numpy as np
from scipy.spatial import ConvexHull, QhullError

from ..constants import BLOCK_SIZE_M
from ..geometry import (
    boxes_touch,
    corners,
    fully_clear,
)
from ..rules import task_counts

REVISION = "tower-task4-final-state-v3"
# Reused from the qualified Task4 table-support40 checker and TABLE_GEOMETRY.
TABLE_MIN = np.array([0.0, -0.66, -0.025])
TABLE_MAX = np.array([0.96, 0.0, 0.0])
TABLE_CONTACT_TOLERANCE_M = 0.005


def table_supported(vertices, center):
    """COM inside the clipped near-table contact patch, not just a low corner.

    Reuses the qualified Task4 support40 geometry. A block leaning from the
    base onto one edge/corner of the table does not have an adequate contact
    polygon and is not automatically classified as off the base.
    """
    bottom = float(vertices[:, 2].min())
    result = dict(
        passed=False,
        bottom_to_table_m=bottom - TABLE_MAX[2],
        contact_tolerance_m=TABLE_CONTACT_TOLERANCE_M,
        com_margin_m=None,
    )
    if abs(bottom - TABLE_MAX[2]) > TABLE_CONTACT_TOLERANCE_M:
        return result
    contact = vertices[np.abs(vertices[:, 2] - TABLE_MAX[2]) <= TABLE_CONTACT_TOLERANCE_M, :2]
    if len(contact) < 3:
        return result
    try:
        poly = contact[ConvexHull(contact).vertices]
    except QhullError:
        return result
    for axis, bound, sign in (
        (0, TABLE_MIN[0], 1),
        (0, TABLE_MAX[0], -1),
        (1, TABLE_MIN[1], 1),
        (1, TABLE_MAX[1], -1),
    ):
        clipped = []
        for a, b in zip(poly, np.roll(poly, -1, axis=0)):
            da, db = sign * (a[axis] - bound), sign * (b[axis] - bound)
            if da >= 0:
                clipped.append(a)
            if (da >= 0) != (db >= 0):
                clipped.append(a + (b - a) * da / (da - db))
        poly = np.asarray(clipped)
        if len(poly) < 3:
            return result
    try:
        poly = poly[ConvexHull(poly).vertices]
    except QhullError:
        return result
    edges = np.roll(poly, -1, axis=0) - poly
    delta = center[:2] - poly
    margins = (edges[:, 0] * delta[:, 1] - edges[:, 1] * delta[:, 0]) / np.linalg.norm(
        edges, axis=1
    )
    margin = float(margins.min())
    result.update(passed=margin > 0, com_margin_m=margin, contact_polygon_m=poly.tolist())
    return result


def final_state(recording, task) -> dict[str, Any]:
    r = recording
    number, layers, _, _ = task_counts(task)
    if number != 4 or r.spec.get("layout_metadata", {}).get("catalog_task") != 4:
        raise ValueError("This revision is only for recorded Task4 scenes")
    blocks = r.spec["blocks"]
    targets = r.spec["layout_metadata"]["push_targets"]
    ids = [t["block_index"] for t in targets]
    if len(ids) != 3 * layers or len(set(ids)) != len(ids):
        raise ValueError("Task4 requires exactly 6/9 distinct movable targets")
    for t in targets:
        i = t["block_index"]
        if (
            not 0 <= i < len(blocks)
            or blocks[i]["name"] != t["block_id"]
            or blocks[i].get("fixed_on_table")
        ):
            raise ValueError("Invalid Task4 target mapping")
    towers = {blocks[i]["tower"] for i in ids}
    movable = {
        i for i, b in enumerate(blocks) if b["tower"] in towers and not b.get("fixed_on_table")
    }
    if movable != set(ids):
        raise ValueError("Target list must cover every movable source-tower block")
    dims = [b.get("dimensions_m") or BLOCK_SIZE_M for b in blocks]
    finals = np.asarray(r.poses[-1:], dtype=float)
    if not np.isfinite(finals).all() or np.any(np.linalg.norm(finals[..., 3:], axis=-1) < 1e-8):
        raise ValueError("Invalid final block pose")
    results = []
    for i in ids:
        bases = [
            j
            for j, b in enumerate(blocks)
            if b.get("fixed_on_table") and b["tower"] == blocks[i]["tower"]
        ]
        if not bases:
            raise ValueError("Missing fixed source base")
        foundation = [corners(r.poses[0, j], dims[j]) for j in bases]
        clear = bool(fully_clear(finals[:, i], dims[i], foundation)[0])
        table = table_supported(corners(finals[0, i], dims[i]), finals[0, i, :3])
        # The qualified support40 checker permits only 0.5 mm numerical
        # interpenetration. A fabricated block inside the base volume is not
        # tabletop support merely because its bottom happens to be at z=0.
        deep_base_overlap = any(
            bool(boxes_touch(finals[:, i], dims[i], finals[:, j], dims[j], -0.0005)[0])
            for j in bases
        )
        off = clear or (table["passed"] and not deep_base_overlap)
        results.append(
            dict(
                block=blocks[i]["name"],
                block_index=i,
                off_source_footprint=clear,
                off_source_base=off,
                removal_route=(
                    "clear_source_footprint"
                    if clear
                    else "table_supported_despite_xy_overlap"
                    if off
                    else None
                ),
                table_support=table,
                deep_base_overlap=deep_base_overlap,
                final_pose=finals[0, i].tolist(),
                source_base_blocks=[blocks[j]["name"] for j in bases],
            )
        )
    return dict(
        passed=all(t["off_source_base"] for t in results),
        removed_blocks=sum(t["off_source_base"] for t in results),
        expected_blocks=3 * layers,
        targets=results,
        remaining_blocks=[t["block"] for t in results if not t["off_source_base"]],
        geometry="Existing full-block/source-footprint SAT, or direct tabletop contact polygon containing COM (qualified Task4 support40 rule)",
        sample=len(r.times) - 1,
        time_s=float(r.times[-1]),
        scope="Final recorded state only; no inferred early stop, contact-force claim, or new tabletop/stability requirement",
    )


def finalize(recording, task, rules, original):
    report = deepcopy(original)
    report.update(
        protocol=REVISION,
        scoring_revision=REVISION,
        original_protocol=original["protocol"],
        original_scores=original["scores"],
        original_terminal_geometry=original.get("terminal_geometry"),
        recorded_terminal_status=recording.status.get("status"),
        scoring_scope="entire recorded trajectory; success from final block state",
    )
    if not original.get("valid"):
        return report
    terminal = final_state(recording, task)
    s = int(terminal["passed"])
    # Runtime stop evidence remains unchanged. A valid failed policy action does
    # not erase task completion already visible in the recorded final state.
    report["scores"].update(s=s, E=original["scores"]["C"] + 1 - s)
    report.update(
        terminal_geometry=terminal["passed"],
        final_state=terminal,
        success_requires_push_credit=False,
        success_requires_prescribed_order=False,
    )
    scores = report["scores"]
    report["odr"] = dict(
        D=scores["D"],
        E=scores["E"],
        value=scores["D"] / scores["E"] if scores["E"] else None,
        denominator_rule="legacy C + 1 - s, C = g + p",
        undefined_reason=None if scores["E"] else "zero legacy denominator",
        detected_operation_windows=len(report["operations"]),
        note="E is the legacy count formula, not a measured count of all attempted pushes",
    )
    return report
