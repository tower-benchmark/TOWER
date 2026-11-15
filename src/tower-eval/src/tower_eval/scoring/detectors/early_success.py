# SPDX-License-Identifier: Apache-2.0
"""Campaign online stop latches, ported without changing the frozen runtime."""

from dataclasses import replace

import numpy as np

from ..api import score_recording as _score_recording
from ..computation import REFERENCE
from ..recording import from_physical_rows as from_physical_rows
from ..rules import FOUR_PLACE_HOLD, PROTOCOL, task_counts
from ..rulesets.bench_2026_11 import DEFAULT_RULES
from .place import placement_geometry


def score_recording(
    recording,
    task,
    rules=DEFAULT_RULES,
    *,
    success_rule=PROTOCOL,
    include_disturbances=True,
    computation=REFERENCE,
):
    return _score_recording(
        recording,
        task,
        rules=rules,
        ruleset="bench-2026.11",
        success_rule=success_rule,
        include_disturbances=include_disturbances,
        computation=computation,
    )


class FourPlaceHoldStop:
    """Evaluate at 1 Hz simulation time; stop at most one second after eligibility."""

    interval_ticks = 400

    def __init__(self, *, computation=REFERENCE):
        self.success = None
        self.computation = computation

    def check(self, recording, tick, *, failed=False):
        if self.success is not None:
            return self.success
        if failed or tick < 4000 or tick % self.interval_ticks:
            return None
        if recording.times[-1] != tick / 400:
            raise ValueError("Success detector must inspect the actual current physics tick")
        # Cheap necessary conditions avoid rescoring the full history while
        # the policy is still carrying blocks. The final decision always uses
        # the common scorer, including its complete ten-second dwell history.
        targets = recording.spec["transfers"]
        latest = replace(
            recording,
            times=recording.times[[0, -1]],
            poses=recording.poses[[0, -1]],
            velocities=recording.velocities[[0, -1]],
        )
        geometry = placement_geometry(
            latest, targets, 1, DEFAULT_RULES, computation=self.computation
        )
        if not all(mask[-1] for mask in geometry.values()):
            return None
        tail = recording.times >= recording.times[-1] - 10.0 - 1e-8
        for target in targets:
            velocities = recording.velocities[tail, target["block_index"]]
            if np.any(
                np.linalg.norm(velocities[:, :3], axis=1) > DEFAULT_RULES.linear_velocity_m_s
            ) or np.any(
                np.linalg.norm(velocities[:, 3:], axis=1) > DEFAULT_RULES.angular_velocity_rad_s
            ):
                return None
        report = score_recording(
            recording,
            "task1-2",
            success_rule=FOUR_PLACE_HOLD,
            include_disturbances=False,
            computation=self.computation,
        )
        if report.get("valid") and report["scores"]["s"] == 1:
            self.success = dict(
                protocol=FOUR_PLACE_HOLD,
                terminal_tick=tick,
                simulated_seconds=tick / 400,
                grasps=4,
                placements=4,
                stable_hold_s=10.0,
                check_interval_s=1.0,
            )
        return self.success


class TaskAwareHoldStop(FourPlaceHoldStop):
    """Online stop for new variants delegates to unchanged task-aware scorer."""

    def __init__(self, task, *, computation=REFERENCE):
        super().__init__(computation=computation)
        if task not in (
            "task1-3",
            "task2-3",
            "task4-2",
            "task4-3",
            "task5-2",
            "task5-3",
            "task6-2",
            "task6-3",
        ):
            raise ValueError("New-variant stop requires explicit supported task")
        self.task = task

    def check(self, recording, tick, *, failed=False):

        if self.success is not None:
            return self.success
        if (
            failed
            or tick < (800 if self.task.startswith("task5") else 4000)
            or tick % self.interval_ticks
        ):
            return None
        if recording.times[-1] != tick / 400:
            raise ValueError("Success clock mismatch")
        number, layers, grasps, placements = task_counts(self.task)
        if number == 4:
            if (
                recording.spec["layout_metadata"]["catalog_task"] != 4
                or recording.spec["transfers"]
                or len(recording.spec["layout_metadata"]["push_targets"]) != grasps + placements
            ):
                raise ValueError("Task4 pushdown identity/count mismatch")
            report = score_recording(
                recording,
                self.task,
                success_rule=PROTOCOL,
                include_disturbances=False,
                computation=self.computation,
            )
            if report.get("valid") and report["scores"]["s"] == 1:
                self.success = dict(
                    protocol=PROTOCOL,
                    terminal_tick=tick,
                    simulated_seconds=tick / 400,
                    pushes=grasps + placements,
                    check_interval_s=1.0,
                )
            return self.success
        if (
            recording.spec["layout_metadata"]["catalog_task"] != number
            or len(recording.spec["transfers"]) != grasps
        ):
            raise ValueError("Task stop identity/count mismatch")
        latest = replace(
            recording,
            times=recording.times[[0, -1]],
            poses=recording.poses[[0, -1]],
            velocities=recording.velocities[[0, -1]],
        )
        geometry = placement_geometry(
            latest, recording.spec["transfers"], number, DEFAULT_RULES, computation=self.computation
        )
        if not all(mask[-1] for mask in geometry.values()):
            return None
        report = score_recording(
            recording,
            self.task,
            success_rule=PROTOCOL,
            include_disturbances=False,
            computation=self.computation,
        )
        if report.get("valid") and report["scores"]["s"] == 1:
            self.success = dict(
                protocol=PROTOCOL,
                terminal_tick=tick,
                simulated_seconds=tick / 400,
                grasps=grasps,
                placements=placements,
                stable_hold_s=10.0,
                check_interval_s=1.0,
            )
        return self.success


class Task2HoldStop(FourPlaceHoldStop):
    """Task2-2 uses the existing task-aware scorer and prescribed order."""

    def check(self, recording, tick, *, failed=False):

        if self.success is not None:
            return self.success
        if failed or tick < 4000 or tick % self.interval_ticks:
            return None
        if recording.times[-1] != tick / 400:
            raise ValueError("Success clock mismatch")
        if len(recording.spec["transfers"]) != 2:
            raise ValueError("Task2-2 requires two transfers")
        latest = replace(
            recording,
            times=recording.times[[0, -1]],
            poses=recording.poses[[0, -1]],
            velocities=recording.velocities[[0, -1]],
        )
        geometry = placement_geometry(
            latest, recording.spec["transfers"], 2, DEFAULT_RULES, computation=self.computation
        )
        if not all(mask[-1] for mask in geometry.values()):
            return None
        report = score_recording(
            recording,
            "task2-2",
            success_rule=PROTOCOL,
            include_disturbances=False,
            computation=self.computation,
        )
        if report.get("valid") and report["scores"]["s"] == 1:
            self.success = dict(
                protocol=PROTOCOL,
                terminal_tick=tick,
                simulated_seconds=tick / 400,
                grasps=2,
                placements=2,
                stable_hold_s=10.0,
                check_interval_s=1.0,
            )
        return self.success
