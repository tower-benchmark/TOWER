# SPDX-License-Identifier: Apache-2.0
"""Shared event pipeline; task selection is delegated to the detector registry."""

from __future__ import annotations

from dataclasses import asdict, replace
from typing import Any

import numpy as np

from .computation import REFERENCE
from .constants import BLOCK_SIZE_M
from .detectors import for_task
from .detectors.common import first, sustained
from .detectors.disturbance import disturbance_events, target_intervals
from .detectors.place import placement_geometry
from .geometry import corners, inverse, rotate
from .outcomes import outcome_fields
from .rules import DEFAULT_RULES, FOUR_PLACE_HOLD, PROTOCOL, task_counts


def score_recording(
    r,
    task,
    rules=DEFAULT_RULES,
    *,
    success_rule=PROTOCOL,
    include_disturbances=True,
    computation=REFERENCE,
):
    if success_rule not in ((PROTOCOL, FOUR_PLACE_HOLD) if rules.four_place_hold else (PROTOCOL,)):
        raise ValueError("Unknown success rule")
    if success_rule == FOUR_PLACE_HOLD and task != "task1-2":
        raise ValueError("Four-place hold success is only defined for task1-2")
    number, layers, planned_g, planned_p = task_counts(task)
    detector = for_task(task)
    if detector.insertion and rules.insertion_rule_overrides:
        rules = replace(rules, **rules.insertion_rule_overrides)
    metadata = r.spec.get("layout_metadata", {})
    if metadata.get("catalog_task", 1) != number:
        raise ValueError("Task argument does not match the recorded scene")
    targets = metadata["push_targets"] if detector.push else r.spec["transfers"]
    expected = (
        planned_g + planned_p if detector.push else 2 * (layers - 1) if detector.tool else planned_g
    )
    if len(targets) != expected or len({t["block_index"] for t in targets}) != expected:
        raise ValueError("Scene target count differs from the paper task/layer protocol")
    for t in targets:
        if (
            t["arm"] not in (0, 1)
            or not 0 <= t["block_index"] < len(r.spec["blocks"])
            or r.spec["blocks"][t["block_index"]]["name"] != t["block_id"]
            or r.spec["blocks"][t["block_index"]].get("fixed_on_table")
        ):
            raise ValueError("Invalid task target mapping")
    report: dict[str, Any] = dict(
        protocol=success_rule,
        task=task,
        rules=asdict(rules),
        source_hashes=r.hashes,
        duration_s=float(r.times[-1]),
        planned_g=planned_g,
        planned_p=planned_p,
        events=[],
        operations=[],
        targets=[],
        planned_pushes_by_arm={
            str(arm): sum(t["arm"] == arm for t in targets) if detector.push else 0
            for arm in (0, 1)
        },
        push_successes_by_arm={"0": 0, "1": 0},
    )
    outcome = outcome_fields(r.status.get("status"), returncode=r.status.get("returncode", 0))
    if r.status.get("valid_trial") is False or not outcome["valid_trial"]:
        return {
            **report,
            "valid": False,
            "invalid_reason": r.status.get("status"),
            "scores": None,
        }
    n = len(r.times)
    dims = [b.get("dimensions_m") or BLOCK_SIZE_M for b in r.spec["blocks"]]
    geometry = (
        {}
        if detector.push
        else placement_geometry(r, targets, detector.geometry_task, rules, computation=computation)
    )
    distances, source_distances = {}, {}
    for t in targets:
        i, arm = t["block_index"], t["arm"]
        local = rotate(inverse(r.poses[:, i, 3:]), r.pinch[:, arm] - r.poses[:, i, :3])
        distances[i] = np.linalg.norm(
            np.maximum(np.abs(local) - np.asarray(dims[i]) / 2, 0), axis=1
        )
        source_local = rotate(inverse(r.poses[0, i, 3:]), r.pinch[:, arm] - r.poses[0, i, :3])
        source_distances[i] = np.linalg.norm(
            np.maximum(np.abs(source_local) - np.asarray(dims[i]) / 2, 0), axis=1
        )
    nearest = {}
    for arm in (0, 1):
        indices = [t["block_index"] for t in targets if t["arm"] == arm]
        if indices:
            nearest[arm] = np.asarray(indices)[
                np.argmin([source_distances[i] for i in indices], axis=0)
            ]
    ordered_samples, final_masks = [], []
    windows = report["operations"]
    g = p = 0
    tool_ownership = []
    if detector.tool:
        from .detectors.tool import score_tools

        tool_ownership = score_tools(
            r, layers, rules, report, ordered_samples, final_masks, computation=computation
        )
        g = sum(e["group"] == "G" for e in report["events"])
        p = sum(e["group"] == "P" for e in report["events"])
    for order, t in enumerate(targets):
        i, arm, name = t["block_index"], t["arm"], t["block_id"]
        approaching = (source_distances[i] <= rules.approach_margin_m) & (nearest[arm] == i)
        if detector.tool:
            approaching = distances[i] <= rules.approach_margin_m
        step_order = (order // 2 * 8 + 4 + order % 2 * 2) if detector.tool else order * 2
        if detector.push:
            local = rotate(inverse(r.poses[0, i, 3:]), r.pinch[:, arm] - r.poses[0, i, :3])
            approaching &= (
                (np.abs(local[:, 1]) <= dims[i][1] / 2 + rules.push_face_margin_m)
                & (np.abs(local[:, 2]) <= dims[i][2] / 2 + rules.push_face_margin_m)
                & (np.abs(local[:, 0]) >= dims[i][0] / 2 - rules.push_face_margin_m)
            )
        approach = sustained(approaching, r.times, rules.approach_s)
        start = first(approach)
        result: dict[str, Any] = dict(
            block=name,
            arm=arm,
            grasp_sample=None,
            place_sample=None,
            attempt_sample=start,
        )
        report["targets"].append(result)
        if start is None:
            final_masks.append(np.zeros(n, dtype=bool))
            continue
        # Include the approach confirmation interval in disturbance measurement.
        start = int(np.searchsorted(r.times, r.times[start] - rules.approach_s - 1e-8))
        foundation = [
            corners(r.poses[0, j], dims[j])
            for j, b in enumerate(r.spec["blocks"])
            if b.get("fixed_on_table") and b["tower"] == r.spec["blocks"][i]["tower"]
        ]
        if not foundation:
            raise ValueError("Task source has no fixed foundation")
        gained_g, gained_p = detector.operation(
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
            computation=computation,
        )
        g += gained_g
        p += gained_p
    finish = None
    if success_rule == FOUR_PLACE_HOLD and g == planned_g and p == planned_p:
        eligible = np.logical_and.reduce(final_masks)
        last_grasp = max(e["sample"] for e in report["events"] if e["group"] == "G")
        eligible[:last_grasp] = False
        # Match the online detector's one-second simulation-time cadence.
        eligible &= np.isclose(r.times, np.round(r.times), atol=1e-8, rtol=0)
        if r.status.get("status") == "safety_abort":
            eligible[-1] = False
        finish = first(eligible)
        # The batch loop checks success immediately after physics, before the
        # next policy request. A rejected next action at that same timestamp
        # would never have been requested after this success stop.
        if finish is not None and (finish < n - 1 or r.status.get("status") == "task_failure"):
            prefix = replace(
                r,
                **{
                    name: getattr(r, name)[: finish + 1]
                    for name in (
                        "times",
                        "poses",
                        "velocities",
                        "pinch",
                        "inverse_flange",
                        "effort",
                        "opening",
                    )
                },
                finger_corners=None if r.finger_corners is None else r.finger_corners[: finish + 1],
                status=dict(
                    r.status,
                    status="task_success",
                    simulated_seconds=float(r.times[finish]),
                ),
            )
            result = score_recording(
                prefix,
                task,
                rules,
                success_rule=success_rule,
                include_disturbances=include_disturbances,
                computation=computation,
            )
            result.update(
                recorded_duration_s=float(r.times[-1]),
                recorded_terminal_status=r.status.get("status"),
                scoring_scope="prefix ending at first eligible online success check",
            )
            return result
    report["target_intervals"] = tool_ownership + target_intervals(
        r, [w for w in windows if not w.get("tool_step")], distances, rules
    )
    # Attribute withdrawal and retries to the operation until the next starts.
    ordered = sorted(windows, key=lambda w: w["start"])
    for k, w in enumerate(ordered):
        following = next((v["start"] for v in ordered[k + 1 :] if v["start"] > w["start"]), n)
        w["end"] = min(w["end"], following - 1)
    windows[:] = [w for w in windows if w["end"] >= w["start"]]
    disturbance = (
        disturbance_events(r, windows, rules, report["target_intervals"])
        if include_disturbances
        else []
    )
    report["disturbances"] = disturbance
    chronological = [k for _, k in sorted(ordered_samples)]
    in_order = all(a <= b for a, b in zip(chronological, chronological[1:]))
    if detector.tool:
        # Starting extraction before returning the tool violates the prescribed
        # sequence even if extraction only finishes after the tool is released.
        plan_ids = []
        for cycle, stage in enumerate(report["tool_cycles"]):
            plan_ids.extend(
                [
                    f"{stage['tool']}:grasp",
                    *[f"{u['block']}:tool_push" for u in stage["pushes"]],
                    f"{stage['tool']}:return",
                ]
            )
            for t in targets[cycle * 2 : cycle * 2 + 2]:
                plan_ids.extend([f"{t['block_id']}:grasp", f"{t['block_id']}:place"])
        starts = {w["id"]: w["start"] for w in windows}
        completed = dict(ordered_samples)
        for order in range(1, len(plan_ids)):
            if order - 1 in completed and plan_ids[order] in starts:
                in_order &= starts[plan_ids[order]] >= completed[order - 1]
    terminal = all(bool(mask[-1]) for mask in final_masks)
    if detector.paired and success_rule == PROTOCOL:
        for level in range(1, layers if detector.tool else layers + 1):
            pair = [t for t in targets if t["destination_layer"] == level]
            if len(pair) != 2:
                raise ValueError("Paired tower must have two blocks per destination layer")
            actual = (
                r.poses[-1, pair[0]["block_index"], :2] - r.poses[-1, pair[1]["block_index"], :2]
            )
            expected = np.asarray(pair[0]["goal"])[:2] - np.asarray(pair[1]["goal"])[:2]
            terminal &= np.linalg.norm(actual - expected) <= rules.paired_spacing_m
    stopped = r.status.get("status") in ("safety_abort", "task_failure")
    s = int(
        g == planned_g
        and p == planned_p
        and terminal
        and (in_order or success_rule == FOUR_PLACE_HOLD)
        and not stopped
        and (success_rule == PROTOCOL or finish == n - 1)
    )
    report.update(
        valid=True,
        terminal_geometry=bool(terminal),
        prescribed_order=in_order,
        scores=dict(
            s=s,
            g=int(g),
            p=int(p),
            C=int(g + p),
            E=int(g + p + 1 - s),
            D=len(disturbance),
        ),
    )
    if success_rule == FOUR_PLACE_HOLD:
        report.update(
            success_time_s=float(r.times[-1]) if s else None,
            recorded_duration_s=float(r.times[-1]),
            recorded_terminal_status=r.status.get("status"),
            scoring_scope="entire recorded trajectory"
            if not s
            else "prefix ending at first eligible online success check",
        )
    if detector.insertion and rules.measured_insertion_gate:
        from .detectors.task5_score import measured_gate

        gate = measured_gate(r)
        report["task5_measured_gate"] = gate
        s = int(bool(s) and gate["passed"])
        report["scores"].update(s=s, E=int(g + p + 1 - s))
        report["scoring_revision"] = "task5-measured-insertion-v1"
    if rules.untouched_translation_m is not None and detector.number == 6:
        report["scoring_revision"] = "task6-generation-consistent-v1"
    return report
