# SPDX-License-Identifier: Apache-2.0
"""Named November campaign amendments; no changes to paper-v1 defaults."""

from dataclasses import dataclass
from typing import ClassVar

from ..rules import Rules as BaseRules


@dataclass(frozen=True)
class Rules(BaseRules):
    ruleset: ClassVar[str] = "bench-2026.11"
    insertion_tower: ClassVar[str] = "source"
    insertion_walls_by_membership: ClassVar[bool] = True
    insertion_rule_overrides: ClassVar[dict[str, float]] = {
        "placement_dwell_s": 0.5,
        "position_m": 0.005,
        "orientation_deg": 5.0,
        "stable_translation_m": 0.0005,
        "stable_rotation_deg": 0.5,
        "linear_velocity_m_s": 0.01,
        "angular_velocity_rad_s": 0.1,
    }
    measured_insertion_gate: ClassVar[bool] = True
    untouched_translation_m: ClassVar[float | None] = 0.002
    untouched_rotation_deg: ClassVar[float | None] = 5.0
    four_place_hold: ClassVar[bool] = True
    tool_direction_from_metadata: ClassVar[bool] = True
    tool_collapse_veto: ClassVar[bool] = True


DEFAULT_RULES = Rules()
# Terminal success revisions are registered separately from legacy event credit.
FINAL_STATE_REVISIONS = {4: "tower-task4-final-state-v3", 6: "tower-task6-roof-no-fall-v1"}
