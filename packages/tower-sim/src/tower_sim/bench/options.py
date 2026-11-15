"""Explicit runtime choices; no environment reads and no command-bound overrides."""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ControlOptions:
    exact_j2_cache: bool = True
    observed_joints: Literal["off", "fast", "shadow"] = "off"
    observation_audit_every: int = 64

    def __post_init__(self):
        if type(self.exact_j2_cache) is not bool or self.observed_joints not in (
            "off",
            "fast",
            "shadow",
        ):
            raise ValueError("Invalid control optimization option")
        if type(self.observation_audit_every) is not int or self.observation_audit_every < 1:
            raise ValueError("Observation audit interval must be a positive integer")


DEFAULT_CONTROL_OPTIONS = ControlOptions()
