"""Pinned evaluation data installed with tower-sim; external assets are explicit."""

from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "data"
PATHS = {
    "motion_profile": "motion_profile_v1.json",
    "runtime_profiles": "runtime_profiles_20260902",
    "calibration_report": "calibration/robot_dynamics.json",
    "stage": "external-assets-required.usdz",
    "replay_runner": "explicit-replay-output-required",
}


def resource_path(checkout, name):
    """Return installed data; checkout is retained for call-site compatibility."""
    return DATA / PATHS[name]


def default_path(name):
    return DATA / PATHS[name]
