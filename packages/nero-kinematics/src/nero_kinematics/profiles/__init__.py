"""Versioned solver settings. Command bounds and observation tolerance are separate types."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent


def load_profile(name: str) -> dict:
    """Return fresh profile data; no implicit environment overrides."""
    aliases = {"package": "canonical", "standalone": "canonical", "pink": "posture-ipopt"}
    name = aliases.get(name, name)
    if name not in {
        "canonical",
        "vendor",
        "posture-ipopt",
        "command",
        "simulation-observation",
        "numpy-dls",
        "legacy-clearance",
        "tcp",
    }:
        raise ValueError(f"Unknown kinematics profile: {name}")
    return json.loads((ROOT / (name + ".json")).read_text())


@dataclass(frozen=True)
class CommandBounds:
    """Strict encoder command bounds. Never accepts observation slack."""

    lower_rad: tuple[float, ...]
    upper_rad: tuple[float, ...]

    def __post_init__(self):
        lo, hi = np.asarray(self.lower_rad), np.asarray(self.upper_rad)
        if (
            lo.shape != (7,)
            or hi.shape != (7,)
            or not np.isfinite([lo, hi]).all()
            or np.any(lo >= hi)
        ):
            raise ValueError("Expected seven finite ordered command bounds")

    def contains(self, joints) -> bool:
        q = np.asarray(joints, dtype=float)
        return bool(
            q.shape[-1:] == (7,)
            and np.isfinite(q).all()
            and np.all(q >= self.lower_rad)
            and np.all(q <= self.upper_rad)
        )


@dataclass(frozen=True)
class SimulationObservationTolerance:
    """Simulation feedback only; not an IK margin or coordinate conversion."""

    radians: float


_data = load_profile("command")["command_bounds"]
COMMAND_BOUNDS = CommandBounds(tuple(_data["lower_rad"]), tuple(_data["upper_rad"]))
SIMULATION_OBSERVATION_TOLERANCE = SimulationObservationTolerance(
    float(np.radians(load_profile("simulation-observation")["observation_tolerance"]["deg"]))
)
