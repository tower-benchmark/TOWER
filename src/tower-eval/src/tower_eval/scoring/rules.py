# SPDX-License-Identifier: Apache-2.0
"""Threshold dataclass and legacy task counts; ruleset selectors are not score data."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import ClassVar

PROTOCOL = "tower-sim-trajectory-v5"
FOUR_PLACE_HOLD = "tower-task12-four-place-hold-v1"


@dataclass(frozen=True)
class Rules:
    # Class variables select semantics without changing the historical Rules JSON.
    ruleset: ClassVar[str] = "paper-v1"
    insertion_tower: ClassVar[str] = "destination"
    insertion_walls_by_membership: ClassVar[bool] = False
    insertion_rule_overrides: ClassVar[dict[str, float]] = {}
    measured_insertion_gate: ClassVar[bool] = False
    untouched_translation_m: ClassVar[float | None] = None
    untouched_rotation_deg: ClassVar[float | None] = None
    four_place_hold: ClassVar[bool] = False
    tool_direction_from_metadata: ClassVar[bool] = False
    tool_collapse_veto: ClassVar[bool] = False
    tool_collapse_thickness_fraction: ClassVar[float] = 0.5
    tool_collapse_tilt_deg: ClassVar[float] = 45.0
    # Paper Appendix B.1.
    placement_dwell_s: float = 10.0
    disturbance_m: float = 0.01
    roof_height_m: float = 0.015
    time_limit_s: float = 420.0
    # Numerical implementation of visually judged geometry and acquisition.
    hold_s: float = 0.3
    grasp_surface_distance_m: float = 0.015
    closed_width_m: float = 0.07
    jaw_effort_n: float = 0.5
    effort_fraction: float = 0.8
    effort_gap_s: float = 0.1
    relative_range_m: float = 0.01
    lift_m: float = 0.005
    minimum_carry_m: float = 0.005
    open_width_m: float = 0.075
    release_clearance_m: float = 0.04
    approach_margin_m: float = 0.025
    approach_s: float = 0.1
    push_face_margin_m: float = 0.015
    minimum_push_m: float = 0.005
    push_coast_s: float = 0.5
    push_clearance_s: float = 0.1
    tool_protrusion_m: float = 0.025
    tool_tip_local_m: tuple[float, float, float] = (0.0, -0.166, 0.01)
    tool_return_dwell_s: float = 0.3
    disturbance_contact_m: float = 0.002
    disturbance_motion_window_s: float = 0.2
    disturbance_motion_m: float = 0.00025
    disturbance_quiet_s: float = 0.3
    position_m: float = 0.01
    paired_spacing_m: float = 0.02
    support_height_m: float = 0.008
    minimum_support_fraction: float = 0.2
    support_com_margin_m: float = 0.002
    orientation_deg: float = 10.0
    slot_clearance_m: float = 0.0005
    stable_translation_m: float = 0.002
    stable_rotation_deg: float = 2.0
    linear_velocity_m_s: float = 0.005
    angular_velocity_rad_s: float = 0.05
    maximum_sample_gap_s: float = 0.1


def task_counts(task):
    match = re.fullmatch(r"task([124567])-([23])", task)
    if match is None:
        raise ValueError(
            f"Unsupported task {task!r}; expected Tasks 1, 2, 4, 5, 6, 7 with 2/3 layers"
        )
    number, layers = map(int, match.groups())
    g, p = {
        1: (2 * layers, 2 * layers),
        2: (layers, layers),
        4: (3 * (layers - 1), 3),
        5: (layers - 1, layers - 1),
        6: (layers - 1, layers - 1),
        7: (5 * (layers - 1), 3 * (layers - 1)),
    }[number]
    return number, layers, g, p


DEFAULT_RULES = Rules()
