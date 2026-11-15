# SPDX-License-Identifier: Apache-2.0
"""Real numerical regression plus standalone protocol/failure checks."""

import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from nero_eef_ik import ArmIK, solve_pair
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "examples/nero_sim_reference.json"


@pytest.mark.skipif(
    importlib.util.find_spec("pinocchio") is None, reason="native-eef environment required"
)
class StandaloneTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.arm = ArmIK(CONFIG)
        cls.other = ArmIK(CONFIG)
        cls.q = (np.array(cls.arm.identity["lower"]) + cls.arm.identity["upper"]) / 2

    def test_actual_reachable_changed_pose_and_quaternion_sign(self):
        targets = [self.q, self.q + [0.1, 0.04, 0, 0.1, 0, 0.03, 0.06]]
        previous = None
        for q in targets:
            target = np.array(self.arm.fk(q))
            result = self.arm.solve(target, self.q + 0.01)
            self.assertTrue(result["success"], result)
            actual = np.array(self.arm.fk(result["joints"]))
            self.assertLessEqual(np.linalg.norm(actual[:3] - target[:3]), 0.001)
            self.assertLessEqual(
                (Rotation.from_quat(actual[3:]).inv() * Rotation.from_quat(target[3:])).magnitude(),
                0.01,
            )
            target[3:] *= -1
            self.assertTrue(self.arm.solve(target, q + 0.01)["success"])
            if previous is not None:
                self.assertGreater(np.linalg.norm(np.array(result["joints"]) - previous), 0.05)
            previous = np.array(result["joints"])

    def test_near_limits_and_seed_variation(self):
        rng = np.random.default_rng(110)
        low, high = np.array(self.arm.identity["lower"]), np.array(self.arm.identity["upper"])
        for fraction in [np.full(7, 0.92), *rng.uniform(0.2, 0.8, (20, 7))]:
            q = low + fraction * (high - low)
            result = self.arm.solve(self.arm.fk(q), q + 0.01)
            self.assertTrue(result["success"], result)

    def test_invalid_inputs_and_units(self):
        target = self.arm.fk(self.q)
        for invalid in [
            [0] * 7,
            [0] * 6,
            [float("nan")] * 7,
            [float("inf")] * 7,
            target[:3] + [0, 0, 0, 2],
        ]:
            with self.assertRaises(ValueError):
                self.arm.solve(invalid, self.q)
        for seed in [[0] * 6, [float("nan")] * 7, [complex(1, 2)] * 7, [100] * 7]:
            with self.assertRaises(ValueError):
                self.arm.solve(target, seed)
        with self.assertRaises(ValueError):
            self.arm.solve(target, self.q, frame="world")
        with tempfile.TemporaryDirectory() as tmp:
            config = json.loads(CONFIG.read_text())
            config["units"]["position"] = "mm"
            path = Path(tmp) / "invalid.json"
            path.write_text(json.dumps(config))
            with self.assertRaises(ValueError):
                ArmIK(path)

    def test_failure_withholds_both_targets_and_reset_isolation(self):
        target = self.arm.fk(self.q)
        unreachable = [10, 10, 10] + target[3:]
        result = self.arm.solve(unreachable, self.q)
        self.assertFalse(result["success"])
        self.assertIsNone(result["joints"])
        self.arm.reset()
        self.other.reset()
        self.arm.solve(target, self.q)
        self.assertIsNone(self.other._solver.last_report)
        result = solve_pair([self.arm, self.other], [target, unreachable], [self.q] * 2)
        self.assertFalse(result["success"])
        self.assertIsNone(result["joints"])
        self.assertNotIn("joints", result["diagnostics"][0])
        self.assertTrue(solve_pair([self.arm, self.other], [target] * 2, [self.q] * 2)["success"])
        self.arm.reset()
        self.assertIsNone(self.arm._solver.last_report)
        self.assertIsNotNone(self.other._solver.last_report)

    def test_soft_timeout_withholds_candidate(self):
        with patch.object(self.arm._solver, "max_time_s", 1e-12):
            result = self.arm.solve(self.arm.fk(self.q), self.q)
        self.assertFalse(result["success"])
        self.assertIsNone(result["joints"])
        self.assertTrue(result["diagnostics"]["solver_status"].startswith("wall_time_exceeded"))

    def test_same_pose_from_distinct_measured_seeds(self):
        target = self.arm.fk(self.q)
        for offset in (-0.15, -0.05, 0.05, 0.15):
            result = self.arm.solve(target, self.q + offset)
            self.assertTrue(result["success"], result)

    def test_smooth_trajectory_with_lagging_measurements(self):
        measured = self.q.copy()
        previous = measured.copy()
        for phase in np.linspace(0, 2 * np.pi, 61):
            target_q = self.q + 0.03 * np.sin(phase) * np.array([1, 0.5, 0, 1, 0, 0.5, 1])
            result = self.arm.solve(self.arm.fk(target_q), measured)
            self.assertTrue(result["success"], result)
            command = np.asarray(result["joints"])
            self.assertLess(np.max(np.abs(command - previous)), 0.08)
            measured += 0.35 * (command - measured)
            previous = command

    def test_overflow_failure_is_json_serializable_and_cli_continues(self):
        invalid = dict(
            pose=[1e200, 0, 0, 0, 0, 0, 1], measured_joints=self.q.tolist(), frame="arm_base"
        )
        result = self.arm.solve(invalid["pose"], self.q)
        self.assertFalse(result["success"])
        self.assertIsNone(result["joints"])
        json.dumps(result, allow_nan=False)
        valid = dict(invalid, pose=self.arm.fk(self.q))
        env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
        run = subprocess.run(
            [sys.executable, "-m", "nero_eef_ik", "--config", str(CONFIG)],
            input=json.dumps(invalid) + "\n" + json.dumps(valid) + "\n",
            text=True,
            capture_output=True,
            check=True,
            cwd=tempfile.gettempdir(),
            env=env,
            timeout=30,
        )
        rejected, accepted = [json.loads(line) for line in run.stdout.splitlines()]
        self.assertFalse(rejected["success"])
        self.assertIsNone(rejected["joints"])
        self.assertTrue(accepted["success"], accepted)

    def test_cli_roundtrip_without_tower_or_simulator_path(self):
        request = dict(pose=self.arm.fk(self.q), measured_joints=self.q.tolist(), frame="arm_base")
        env = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
        result = subprocess.run(
            [sys.executable, "-m", "nero_eef_ik", "--config", str(CONFIG)],
            input="{}\n" + json.dumps(request) + "\n",
            text=True,
            capture_output=True,
            check=True,
            cwd=tempfile.gettempdir(),
            env=env,
            timeout=30,
        )
        invalid, valid = [json.loads(line) for line in result.stdout.splitlines()]
        self.assertFalse(invalid["success"])
        self.assertIsNone(invalid["joints"])
        self.assertTrue(valid["success"], valid)
        self.assertEqual(valid["backend"], "pinocchio_casadi")
        self.assertEqual(len(valid["joints"]), 7)


if __name__ == "__main__":
    unittest.main()
