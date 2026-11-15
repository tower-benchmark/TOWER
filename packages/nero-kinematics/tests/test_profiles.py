"""Independent policy derivation and package-only FK; no robotics native imports."""

import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
from nero_kinematics import create_solver
from nero_kinematics.factory import MODELS
from nero_kinematics.profiles import (
    COMMAND_BOUNDS,
    ROOT,
    SIMULATION_OBSERVATION_TOLERANCE,
    CommandBounds,
    SimulationObservationTolerance,
    load_profile,
)

REPO = Path(__file__).resolve().parents[3]
FROZEN = REPO / "packages/tower-sim/src/tower_sim/data/benchmark/runtime"


def test_command_bounds_recomputed_from_independent_sources():
    hardware = json.loads(
        (
            FROZEN / "ik-capsule/runtime-profiles/real-machine-20261027/pinocchio_config.json"
        ).read_text()
    )
    model = ET.parse(MODELS / "agilex/nero_description.urdf")
    urdf = [model.find(f".//joint[@name='joint{i}']/limit") for i in range(1, 8)]
    hard_lo, hard_hi = np.array(hardware["lower"]), np.array(hardware["upper"])
    # Real normal soft limits are hard limits inset 0.03 radians. URDF J6 is narrower.
    soft_lo, soft_hi = hard_lo + 0.03, hard_hi - 0.03
    lo = np.maximum(soft_lo, [float(j.attrib["lower"]) for j in urdf]) + np.radians(1.5)
    hi = np.minimum(soft_hi, [float(j.attrib["upper"]) for j in urdf]) - np.radians(1.5)
    lo[1] = max(lo[1], hard_lo[1] + np.radians(5.0))
    hi[1] = min(hi[1], hard_hi[1] - np.radians(5.0))
    np.testing.assert_allclose(COMMAND_BOUNDS.lower_rad, lo, atol=1e-15, rtol=0)
    np.testing.assert_allclose(COMMAND_BOUNDS.upper_rad, hi, atol=1e-15, rtol=0)
    frozen = json.loads((FROZEN / "shared/shared-ik-spec.json").read_text())["limits"]["ik_command"]
    assert list(COMMAND_BOUNDS.lower_rad) == frozen["lower_rad"]
    assert list(COMMAND_BOUNDS.upper_rad) == frozen["upper_rad"]


def test_observation_slack_cannot_be_used_as_command_tolerance():
    assert type(COMMAND_BOUNDS) is CommandBounds
    assert type(SIMULATION_OBSERVATION_TOLERANCE) is SimulationObservationTolerance
    assert not isinstance(SIMULATION_OBSERVATION_TOLERANCE, CommandBounds)
    assert not hasattr(COMMAND_BOUNDS, "tolerance")
    assert not hasattr(SIMULATION_OBSERVATION_TOLERANCE, "lower_rad")
    assert SIMULATION_OBSERVATION_TOLERANCE.radians == np.radians(0.5)
    command = load_profile("command")
    assert "observation_tolerance" not in command
    assert "command_bounds" not in load_profile("simulation-observation")
    # A command only 0.25 degrees outside is still rejected, despite 0.5-degree feedback slack.
    q = np.array(COMMAND_BOUNDS.upper_rad)
    q[0] += SIMULATION_OBSERVATION_TOLERANCE.radians / 2
    assert not COMMAND_BOUNDS.contains(q)
    with pytest.raises(TypeError):
        COMMAND_BOUNDS.contains(q, tolerance=SIMULATION_OBSERVATION_TOLERANCE)
    with pytest.raises(TypeError):
        CommandBounds(
            COMMAND_BOUNDS.lower_rad, COMMAND_BOUNDS.upper_rad, SIMULATION_OBSERVATION_TOLERANCE
        )


def test_real_hardware_profile_is_byte_identical():
    source = FROZEN / "ik-capsule/runtime-profiles/real-machine-20261027"
    destination = ROOT / "real_hardware/real-machine-20261027"
    assert sorted(p.name for p in source.iterdir()) == sorted(p.name for p in destination.iterdir())
    for path in source.iterdir():
        assert path.read_bytes() == (destination / path.name).read_bytes(), path.name


@pytest.mark.archive
def test_model_copies_preserve_source_bytes():
    import re

    from scripts.maintainer.archive_root import archive_bundle

    frozen = archive_bundle() / "runtime"
    sources = {
        "nero_sdk_mdh.urdf": "support/task2_fixed/vendor/eef_ik_movejs/models/nero_sdk_mdh.urdf",
        "agilex/nero_description.urdf": "ik-capsule/runtime/urdf/nero_description.urdf",
        "agilex/nero_with_flange_description.xacro": "ik-capsule/runtime/urdf/nero_with_flange_description.xacro",
        "agilex/nero_with_gripper_description.xacro": "ik-capsule/runtime/urdf/nero_with_gripper_description.xacro",
    }
    for name, source in sources.items():
        assert re.sub(rb"<!--.*?-->", b"", (MODELS / name).read_bytes(), flags=re.S) == re.sub(
            rb"<!--.*?-->", b"", (frozen / source).read_bytes(), flags=re.S
        )


def test_fk_import_and_installed_data_need_no_native_solver(tmp_path):
    code = """
import sys
from nero_kinematics import create_solver
from nero_kinematics.profiles import load_profile
kin = create_solver("numpy-dls")
assert kin.fk([0.] * 7).shape == (4, 4)
assert create_solver("sdk-mdh").sdk_transform([0.] * 7).shape == (4, 4)
assert not any(n.split(".")[0] in {"pinocchio", "casadi", "pink"} for n in sys.modules)
"""
    subprocess.run([sys.executable, "-c", code], cwd=tmp_path, check=True)


def test_profile_reads_are_independent():
    a, b = load_profile("canonical"), load_profile("canonical")
    a["solver"]["orientation_weight"] = 123
    assert b["solver"]["orientation_weight"] == 0.1
    with pytest.raises(ValueError, match="Unknown"):
        load_profile("../command")
    assert callable(create_solver("numpy-dls").ik)


# Literal defaults of tower_eval.kinematics.PinocchioIK at the archived source revision (before the library
# move), including the then-hard-coded ipopt.tol. The live EEF controller takes its
# defaults from this profile, so any edit here must be a deliberate behaviour change.
CANONICAL_SOLVER_AT_ARCHIVED_SOURCE = {
    "position_tolerance_m": 0.001,
    "orientation_tolerance_rad": 0.01,
    "position_weight": 1.0,
    "orientation_weight": 0.1,
    "seed_weight": 1e-5,
    "max_iter": 80,
    "max_time_s": 0.025,
    "optimizer_tolerance": 1e-7,
    "verified_current_seed": False,
}


def test_canonical_solver_defaults_are_unchanged():
    assert load_profile("canonical")["solver"] == CANONICAL_SOLVER_AT_ARCHIVED_SOURCE
