"""Behavioral checks of opt-in vendor features and canonical failure handling."""

import importlib.util
from types import SimpleNamespace

import numpy as np
import pytest
from nero_kinematics import create_solver
from nero_kinematics.solvers.ipopt import PinocchioIK


@pytest.fixture
def numerical_stub():
    """Exercise public solve control flow without importing a robotics runtime."""
    arm = PinocchioIK.__new__(PinocchioIK)
    arm.lower, arm.upper = -np.ones(7), np.ones(7)
    arm.position_tolerance_m, arm.orientation_tolerance_rad = 0.001, 0.01
    arm.max_time_s, arm.verified_current_seed = 5.0, False
    arm.fk = lambda q: np.array([q[0], 0, 0, 0, 0, 0, 1.0])
    calls = []

    class Optimizer:
        def __call__(self, **kwargs):
            calls.append(kwargs)
            return {"x": kwargs["x0"].copy()}

        def stats(self):
            return dict(success=True, return_status="Solve_Succeeded", iter_count=1)

    arm.solver = Optimizer()
    return SimpleNamespace(arm=arm, calls=calls)


def test_canonical_default_still_optimizes_current_seed_and_clips(numerical_stub):
    arm, calls = numerical_stub.arm, numerical_stub.calls
    arm.ik(arm.fk(np.zeros(7)), np.zeros(7))
    assert len(calls) == 1
    arm.ik(arm.fk(np.ones(7)), np.full(7, 2.0))
    np.testing.assert_array_equal(calls[-1]["x0"], np.ones(7))


def test_verified_seed_shortcut_is_opt_in_and_has_time_guard(numerical_stub):
    arm = numerical_stub.arm
    arm.verified_current_seed = True
    seed = np.zeros(7)
    q, ok = arm.ik(arm.fk(seed), seed)
    assert ok and not numerical_stub.calls
    assert arm.last_report["solver_status"] == "verified_current_seed"
    np.testing.assert_array_equal(q, seed)
    arm.max_time_s = 1e-12
    _, ok = arm.ik(arm.fk(seed), seed)
    assert not ok


def test_soft_margin_allows_retreat_but_no_advance_and_radius_intersects(numerical_stub):
    arm, calls = numerical_stub.arm, numerical_stub.calls
    seed = np.array([-0.95, 0.95, 0, 0, 0, 0, 0])
    arm.ik(arm.fk(seed), seed, margin_rad=0.2, joint_radius_rad=0.1)
    np.testing.assert_allclose(
        calls[0]["lbx"], [-0.95, 0.85, -0.1, -0.1, -0.1, -0.1, -0.1], atol=1e-15
    )
    np.testing.assert_allclose(calls[0]["ubx"], [-0.85, 0.95, 0.1, 0.1, 0.1, 0.1, 0.1], atol=1e-15)


@pytest.mark.parametrize(
    "options",
    [
        {"margin_rad": -1.0},
        {"margin_rad": 1.0},
        {"margin_rad": np.nan},
        {"joint_radius_rad": 0.0},
        {"joint_radius_rad": np.inf},
    ],
)
def test_bad_bounds_options_are_rejected(numerical_stub, options):
    arm = numerical_stub.arm
    with pytest.raises(ValueError):
        arm.ik(arm.fk(np.zeros(7)), np.zeros(7), **options)
    assert not numerical_stub.calls


def test_verified_shortcut_and_margin_reject_seed_outside_hard_limits(numerical_stub):
    arm = numerical_stub.arm
    with pytest.raises(ValueError, match="hard joint"):
        arm.ik(arm.fk(np.ones(7)), np.full(7, 2.0), margin_rad=0.1)
    arm.verified_current_seed = True
    with pytest.raises(ValueError, match="hard joint"):
        arm.ik(arm.fk(np.ones(7)), np.full(7, 2.0))


def test_nonfinite_fk_cannot_be_accepted_or_shortcut(numerical_stub):
    arm = numerical_stub.arm
    arm.verified_current_seed = True
    arm.fk = lambda q: np.array([np.nan, 0, 0, 0, 0, 0, 1.0])
    _, ok = arm.ik([0, 0, 0, 0, 0, 0, 1], np.zeros(7))
    assert not ok and len(numerical_stub.calls) == 1
    assert arm.last_report["error"] == "Nonfinite FK residual"
    assert arm.last_report["position_error_m"] is None


@pytest.mark.skipif(importlib.util.find_spec("pinocchio") is None, reason="native-eef required")
def test_native_vendor_options_and_shared_wrapper():
    from nero_kinematics.fk import numpy_oracle
    from nero_kinematics.solvers.shared import SharedIK

    vendor = create_solver("vendor")
    assert vendor.identity["optimizer_tolerance"] == 1e-10
    q = np.array([0.1, -0.3, 0.4, 0.6, -0.2, 0.15, 0.1])
    result, accepted = vendor.ik(vendor.fk(q), q, joint_radius_rad=0.2, margin_rad=0.03)
    assert accepted and vendor.last_report["solver_status"] == "verified_current_seed"
    np.testing.assert_array_equal(result, q)
    shared = SharedIK()
    r = shared.encoder(numpy_oracle.fk(q), q, 0)
    assert r["accepted"], r
    with pytest.raises(ValueError, match="conversion"):
        shared.physical(numpy_oracle.fk(q), q, 0.05, 0)
