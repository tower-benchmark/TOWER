import json
import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from nero_kinematics.fk import urdf
from nero_kinematics.profiles import COMMAND_BOUNDS, SIMULATION_OBSERVATION_TOLERANCE
from scipy.spatial.transform import Rotation
from tower_sim.bench import control
from tower_sim.bench.observation import encoder_observation
from tower_sim.bench.options import ControlOptions

from tests.characterization import runtime_scenarios
from tests.characterization.golden_compare import assert_equivalent

GOLDEN = runtime_scenarios.ROOT / "tests/golden/benchmark-2026.11"


@pytest.mark.archive
def test_command_queue_golden_and_original(monkeypatch):
    expected = runtime_scenarios.command_queue_stream()
    monkeypatch.setattr(runtime_scenarios, "runtime_module", lambda name: control)
    actual = runtime_scenarios.command_queue_stream()
    assert actual == expected
    assert_equivalent(json.loads((GOLDEN / "command-queue.json").read_text()), actual)


@pytest.mark.archive
def test_observation_mapping_golden_and_original():
    q = np.arange(16, dtype=float) / 100
    q[[7, 15]] = [0.04, 0.08]

    def compensate(joints, width, arm):
        result = joints.copy()
        result[1] -= width * (1 if arm == "left" else 2)
        return result

    def fk(joints, width):
        matrix = np.eye(4)
        matrix[:3, 3] = joints[:3]
        return {"gripper_base": matrix}

    actual = encoder_observation(
        q,
        SimpleNamespace(encoder_from_physical=compensate),
        Rotation.from_euler("z", 90, degrees=True),
        np.array([0.01, 0.02, 0.03]),
        fk=fk,
    )
    assert actual == runtime_scenarios.encoder_observation_output()
    assert_equivalent(json.loads((GOLDEN / "encoder-observation.json").read_text()), actual)


@pytest.mark.archive
def test_command_bounds_and_observation_tolerance_are_distinct(frozen_control):
    np.testing.assert_array_equal(control.GENERATION_LOWER, frozen_control.GENERATION_LOWER)
    np.testing.assert_array_equal(control.GENERATION_UPPER, frozen_control.GENERATION_UPPER)
    np.testing.assert_array_equal(control.GENERATION_LOWER, COMMAND_BOUNDS.lower_rad)
    assert (
        control.OBSERVED_TRACKING_TOLERANCE_RAD
        == np.radians(0.5)
        == SIMULATION_OBSERVATION_TOLERANCE.radians
    )
    for options in (
        dict(command_profile="simulation-observation"),
        dict(observation_profile="command"),
    ):
        with pytest.raises(TypeError):
            ControlOptions(**options)
    q = (control.GENERATION_LOWER + control.GENERATION_UPPER) / 2
    q[0] = control.GENERATION_UPPER[0] + np.radians(0.25)
    for module in (control, frozen_control):
        with pytest.raises(module.PolicyActionError):
            module.validate_encoder_joints(q)
        module.validate_observed_encoder_joints(q)
    with pytest.raises(TypeError):
        control.validate_encoder_joints(q, tolerance=SIMULATION_OBSERVATION_TOLERANCE.radians)


def response(mode="joint_absolute", count=1):
    q = np.tile((control.GENERATION_LOWER + control.GENERATION_UPPER) / 2, (count, 2, 1))
    return dict(
        coordinate_frame="encoder_arm_base",
        j8_actuator_mask=[False, False],
        control_hz=10,
        execute_steps=count,
        mode=mode,
        gripper_width_m=np.full((count, 2), 0.05),
        joint_encoder_targets=q,
        eef_encoder_targets=np.tile(np.eye(4), (count, 2, 1, 1)),
    )


def test_rejected_ik_does_not_retry_a_reversed_target():
    calls = []
    request = response("eef_delta")
    request["eef_encoder_targets"][:, :, :3, :3] = np.diag([1.0, -1.0, -1.0])

    def reject(targets, seed, **kwargs):
        calls.append(targets.copy())
        return dict(accepted=False, diagnostics=[dict(reason="reversed target rejected")])

    with pytest.raises(control.PolicyActionError, match="IK rejected policy target"):
        control.encoder_endpoints(
            request,
            {"joint_encoder": request["joint_encoder_targets"][0]},
            solver=SimpleNamespace(solve=reject),
            prefix=1,
        )
    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0], request["eef_encoder_targets"])


def outcome(fn, *args, **kwargs):
    try:
        return "ok", fn(*args, **kwargs)
    except (ValueError, RuntimeError) as exc:
        return "error", str(exc)


@pytest.mark.archive
@pytest.mark.parametrize("mode", ["joint_absolute", "eef_delta"])
@pytest.mark.parametrize(
    "case", ["ok", "bound", "nan", "width", "frame", "timing", "shape", "rejected"]
)
def test_encoder_endpoints_equivalence(frozen_control, mode, case):
    request = response(mode)
    if case == "bound":
        request["joint_encoder_targets"][0, 0, 0] = control.GENERATION_UPPER[0] + 0.001
    if case == "nan":
        request["joint_encoder_targets"][0, 0, 0] = np.nan
    if case == "width":
        request["gripper_width_m"][0, 0] = 0.100005
    if case == "frame":
        request["coordinate_frame"] = "world"
    if case == "timing":
        request["control_hz"] = 20
    if case == "shape":
        request["eef_encoder_targets"] = np.zeros((1, 2, 3, 3))
        request["joint_encoder_targets"] = np.zeros((1, 7))
    result = dict(
        accepted=case != "rejected",
        joints=request["joint_encoder_targets"],
        diagnostics=[dict(reason="fixture")],
    )

    def solve(*a, **kw):
        return result

    frozen_control.deployed_solve = solve
    observation = {"joint_encoder": response()["joint_encoder_targets"][0]}
    expected = outcome(
        frozen_control.encoder_endpoints,
        request,
        observation,
        SimpleNamespace(LIMITS=urdf.limits()),
        prefix=1,
    )
    actual = outcome(
        control.encoder_endpoints,
        request,
        observation,
        solver=SimpleNamespace(solve=solve),
        prefix=1,
    )
    assert actual[0] == expected[0]
    if actual[0] == "ok":
        np.testing.assert_array_equal(actual[1][0], expected[1][0])
        assert actual[1][1] == expected[1][1]
    else:
        assert actual[1] == expected[1]


@pytest.mark.archive
@pytest.mark.parametrize("observed", [False, True])
def test_bounds_edges_and_nonfinite(frozen_control, observed):
    name = "validate_observed_encoder_joints" if observed else "validate_encoder_joints"
    lo = (
        control.KILL_LOWER - control.OBSERVED_TRACKING_TOLERANCE_RAD
        if observed
        else control.GENERATION_LOWER - 1e-7
    )
    hi = (
        control.KILL_UPPER + control.OBSERVED_TRACKING_TOLERANCE_RAD
        if observed
        else control.GENERATION_UPPER + 1e-7
    )
    for q in (
        lo,
        hi,
        np.nextafter(lo, -np.inf),
        np.nextafter(hi, np.inf),
        np.full(7, np.nan),
        np.full(7, np.inf),
        np.zeros(6),
    ):
        assert outcome(getattr(control, name), q) == outcome(getattr(frozen_control, name), q)


@pytest.mark.archive
def test_physical_targets_and_queue_errors(frozen_control):
    limits = np.tile([-1.0, 1.0], (16, 1))
    for q in (
        np.zeros(16),
        limits[:, 1] + 2e-6,
        limits[:, 1] + 4e-6,
        np.full(16, np.nan),
        np.zeros(15),
    ):
        assert outcome(control.validate_physical_targets, q, limits) == outcome(
            frozen_control.validate_physical_targets, q, limits
        )
    for home, delay in (
        (np.zeros(15), np.zeros((2, 7))),
        (np.zeros(16), np.full((2, 7), np.nan)),
        (np.zeros(16), np.full((2, 7), -0.1)),
    ):
        assert outcome(control.CommandQueue, home, delay, lambda x: x) == outcome(
            frozen_control.CommandQueue, home, delay, lambda x: x
        )
    for module in (control, frozen_control):
        queue = module.CommandQueue(np.zeros(16), np.zeros((2, 7)), lambda x: x)
        with pytest.raises(ValueError, match="No known command"):
            queue.target(100)


@pytest.mark.archive
def test_bounds_ignore_environment_before_import(frozen_control, tmp_path):
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            """
import json
from tower_sim.bench import control
print(json.dumps([control.GENERATION_LOWER.tolist(), control.GENERATION_UPPER.tolist(),
                  control.OBSERVED_TRACKING_TOLERANCE_RAD]))
""",
        ],
        cwd=tmp_path,
        env=dict(os.environ, TOWER_EVAL_IK_BOUNDS="simulation-observation"),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    lower, upper, tolerance = json.loads(result.stdout)
    np.testing.assert_array_equal(lower, frozen_control.GENERATION_LOWER)
    np.testing.assert_array_equal(upper, frozen_control.GENERATION_UPPER)
    assert tolerance == np.radians(0.5)


def test_encoder_endpoints_reuses_readonly_urdf_limits(monkeypatch):
    def unexpected_parse():
        raise AssertionError("URDF limits parsed in command loop")

    monkeypatch.setattr(urdf, "limits", unexpected_parse)
    for _ in range(2):
        request = response()
        control.encoder_endpoints(request, {}, prefix=1)
    with pytest.raises(ValueError):
        control._URDF_LIMITS[0, 0] = 0
