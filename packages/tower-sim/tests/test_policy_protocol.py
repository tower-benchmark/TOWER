"""Controller and authorization contracts, independent of transport encoding."""

import ast
from types import SimpleNamespace as NS

import numpy as np
import pytest
from tower_sim.bench import control, policy_entry
from tower_sim.bench.policy_ik import PolicyIK


def action_block(path):
    tree = ast.parse(path.read_text())
    owner = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Try)
        and any(
            isinstance(b, ast.Assign) and ast.unparse(b.targets[0]) == "(endpoints, diagnostics)"
            for b in n.body
        )
    )
    return compile(ast.Module(body=[owner], type_ignores=[]), str(path), "exec")


def decisions(scope, code):
    try:
        exec(code, scope)
        return "reject" if scope["failures"][0] else "enqueue"
    except (RuntimeError, ValueError, AssertionError, OSError):
        return "abort"


def test_public_rejects_command_replay_before_other_inputs():
    with pytest.raises(ValueError, match="reference-replay"):
        policy_entry.validate_entry({"command_replay": True}, [], None)


def test_exact_flip_tripwire_without_native_call():
    from nero_kinematics.fk.numpy_oracle import fk

    calls = []
    native = NS(solve=lambda *a, **k: calls.append((a, k)))
    solver = PolicyIK(None, client=native)
    seed = np.tile((control.GENERATION_LOWER + control.GENERATION_UPPER) / 2, (2, 1))
    poses = np.tile(fk(seed[0]), (16, 2, 1, 1))
    poses[0, 0, :3, :3] @= np.diag([1, -1, -1])
    result = solver.solve(poses, seed)
    assert result["accepted"] is False and result["joints"] is None
    assert result["diagnostics"][0]["reason"] == "reversed_180_grasp"
    assert not calls


@pytest.mark.parametrize(
    "reason",
    [
        "seed_outside_kill_line",
        "pose_bounds_or_time_rejection",
        "step_from_unclipped_seed",
        "physical_pose_bounds_or_time",
    ],
)
def test_planner_rejection_never_retries(reason):
    from nero_kinematics.fk.numpy_oracle import fk

    calls = []
    result = dict(accepted=False, joints=None, diagnostics=[dict(reason=reason)])

    def solve(*a, **k):
        calls.append((a, k))
        return result

    solver = PolicyIK(None, client=NS(solve=solve))
    seed = np.zeros((2, 7))
    poses = np.tile(fk(seed[0]), (16, 2, 1, 1))
    assert solver.solve(poses, seed) is result
    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0][0][0], poses)


def test_policy_imports_without_optional_image_or_simulator_dependencies():
    import subprocess
    import sys

    script = """import importlib.abc,sys
class Block(importlib.abc.MetaPathFinder):
 def find_spec(self,fullname,path=None,target=None):
  if fullname.split('.')[0] in {'PIL','h5py','isaacsim','omni','pxr','carb','pinocchio','casadi','jenga_generator','batch_policy_worker'}:raise AssertionError(fullname)
sys.meta_path.insert(0,Block())
import tower_sim.bench.policy_entry,tower_sim.bench.policy_ik
"""
    subprocess.run([sys.executable, "-c", script], check=True)


@pytest.mark.parametrize("degrees,tilt", [(179, 0), (180, 0.2), (120, 0)])
def test_near_flip_targets_pass_unchanged_once(degrees, tilt):
    from nero_kinematics.fk.numpy_oracle import fk
    from scipy.spatial.transform import Rotation

    calls = []
    result = dict(accepted=False)

    def solve(*args, **kwargs):
        calls.append((args, kwargs))
        return result

    solver = PolicyIK(None, client=NS(solve=solve))
    seed = np.tile((control.GENERATION_LOWER + control.GENERATION_UPPER) / 2, (2, 1))
    poses = np.tile(fk(seed[0]), (16, 2, 1, 1))
    poses[0, 0, :3, :3] @= Rotation.from_euler("xy", [degrees, tilt], degrees=True).as_matrix()
    original = poses.copy()
    assert solver.solve(poses, seed) is result
    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0][0][0], original)
