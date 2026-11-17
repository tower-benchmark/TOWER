# SPDX-License-Identifier: Apache-2.0
"""Informational TCP posture flags; independent of benchmark scoring."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial.transform import Rotation

REFERENCE = "tower-ordinary-tcp-reference-v1"


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def classify(states, report):
    """Evaluate engaged operation windows plus one second of recorded approach.

    Base rotation must remain constant to within 0.001 radians. This tolerance
    accommodates recorded quaternion precision; it is not a grasp threshold.
    """
    with h5py.File(states, "r") as handle:
        missing = sorted({"time", "eef", "eef_world"} - set(handle))
        if missing:
            return dict(
                schema="tower-posture-flags-v1",
                reference=REFERENCE,
                status="not_evaluated",
                operations=[],
                reason="Missing TCP evidence: " + ", ".join(missing),
            )
        times = handle["time"][:]
        local = handle["eef"][:]
        world = handle["eef_world"][:]
    operations = []
    for operation in report.get("operations", []):
        arm = operation["arm"]
        result = dict(
            operation=operation["id"],
            arm=arm,
            status="not_evaluated",
            min_closing_dot=None,
            min_approach_down_dot=None,
        )
        operations.append(result)
        mask = np.zeros(len(times), dtype=bool)
        for interval in report.get("target_intervals", []):
            if interval["arm"] != arm or interval["target"] != operation["target"]:
                continue
            start = max(interval["start"], operation["start"])
            end = min(interval["end"], operation.get("active_end", operation["end"]))
            if 0 <= start <= end < len(times):
                mask |= (times >= times[start] - 1.0) & (times <= times[end])
        if not mask.any():
            result["reason"] = "No samples in target interval or approach window"
            continue
        try:
            rotations = Rotation.from_quat(world[:, arm, 3:7]).as_matrix()
            bases = rotations @ Rotation.from_quat(local[:, arm, 3:7]).as_matrix().transpose(
                0, 2, 1
            )
            drift = Rotation.from_matrix(bases @ bases[0].T).magnitude()
            result["base_rotation_max_drift_rad"] = float(drift.max())
            if not np.isfinite(drift).all() or drift.max() > 0.001:
                result["reason"] = "Arm base orientation is not near-constant"
                continue
            closing = -bases[0, :, 1].copy()
            closing[2] = 0
            if np.linalg.norm(closing) < 1e-8:
                result["reason"] = "Base minus-y has no horizontal projection"
                continue
            closing /= np.linalg.norm(closing)
            dots = rotations[:, :, 2] @ closing
            down = -rotations[:, 2, 0]
            if not np.isfinite(dots).all() or not np.isfinite(down).all():
                raise ValueError("Nonfinite TCP orientation")
        except ValueError as error:
            result["reason"] = str(error)
            continue
        result["min_closing_dot"] = float(dots[mask].min())
        result["min_approach_down_dot"] = float(down[mask].min())
        longest = streak = 0
        for sample, reversed_sample in enumerate(mask & (dots <= -0.5)):
            if sample and times[sample] - times[sample - 1] > 0.075:
                streak = 0
            streak = streak + 1 if reversed_sample else 0
            longest = max(longest, streak)
        result["reversed_consecutive_samples"] = longest
        result["status"] = "reversed_grasp" if longest >= 5 else "ordinary"
    statuses = {operation["status"] for operation in operations}
    status = (
        "reversed_grasp"
        if "reversed_grasp" in statuses
        else "not_evaluated"
        if not statuses or "not_evaluated" in statuses
        else "ordinary"
    )
    return dict(
        schema="tower-posture-flags-v1",
        reference=REFERENCE,
        status=status,
        operations=operations,
        **({"reason": "No operations"} if not operations else {}),
    )


def flags(states, score):
    result = classify(states, json.loads(Path(score).read_text()))
    result.update(score_sha256=sha256(score), states_sha256=sha256(states))
    return result
