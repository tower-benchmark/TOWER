"""Signed task geometry and single-attempt dispatch, independent of native IK."""

import numpy as np
import pytest
from nero_kinematics import posture_evidence, solve_grasp
from scipy.spatial.transform import Rotation


@pytest.mark.parametrize(
    "ordinary", [np.eye(3), Rotation.from_euler("xyz", [0.3, -0.7, 1.1]).as_matrix()]
)
def test_half_turn_about_tcp_approach_is_geometrically_rejected(ordinary):
    reversed_target = ordinary @ np.diag([1.0, -1.0, -1.0])
    evidence = posture_evidence(reversed_target, ordinary)
    assert not evidence.accepted
    np.testing.assert_allclose(evidence.approach_alignment, [1.0], atol=1e-15)
    np.testing.assert_allclose(evidence.closing_alignment, [-1.0], atol=1e-15)
    assert posture_evidence(ordinary, ordinary).accepted


def test_check_all_approach_carry_and_recorded_phase_samples():
    ordinary = Rotation.from_euler("z", np.linspace(0, 0.4, 9)[:, None]).as_matrix()
    actual = ordinary.copy()
    assert posture_evidence(actual, ordinary).accepted
    for index in range(len(actual)):
        actual[index] = ordinary[index] @ np.diag([1.0, -1.0, -1.0])
        assert not posture_evidence(actual, ordinary).accepted
        actual[index] = ordinary[index]


def test_rejected_ik_does_not_retry_a_reversed_target():
    calls = []
    ordinary = np.eye(4)

    def rejected(target, seed):
        calls.append(target.copy())
        return dict(accepted=False, joints=None, reason="no_solution")

    result = solve_grasp(rejected, ordinary, np.zeros(7), ordinary)
    assert result == dict(accepted=False, joints=None, reason="no_solution")
    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0], ordinary)
    reversed_target = ordinary.copy()
    reversed_target[:3, :3] = np.diag([1.0, -1.0, -1.0])
    result = solve_grasp(rejected, reversed_target, np.zeros(7), ordinary)
    assert not result["accepted"] and result["joints"] is None
    assert len(calls) == 1


def test_task_axis_convention_is_explicit():
    # A TCP with z approach and x closing uses its own task convention.
    assert not posture_evidence(
        np.diag([-1.0, -1.0, 1.0]), np.eye(3), approach_axis=2, closing_axis=0
    ).accepted
    with pytest.raises(ValueError):
        posture_evidence(np.eye(3), np.eye(3), approach_axis=0, closing_axis=0)
    with pytest.raises(ValueError):
        posture_evidence(np.full((3, 3), np.nan), np.eye(3))
    with pytest.raises(ValueError):
        posture_evidence(-np.eye(3), np.eye(3))
