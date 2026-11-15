"""Calibrated USD drives, v4 gains and physical readback, with instance state."""

import json
from copy import deepcopy
from pathlib import Path

import numpy as np

from .redaction import matches_exact
from .resources import save, sha


class CalibratedDrives:
    PREFIX = "/PNP_DUALD1200_STATION/StationModel/InstalledRobots/"

    def __init__(self, scene, configuration):
        self.scene, self.configuration = Path(scene), configuration
        self._stage = None
        self.readbacks = {}

    def robot_root(self, arm):
        return (
            self.PREFIX
            + ("StandA_Robot" if arm == "left" else "StandB_Robot")
            + "/NeroRobotWithWristCamera"
        )

    def stage(self):
        if self._stage is None:
            # Import Kit's USD only after SimulationApp starts. Importing the pip
            # usd-core first binds a different ABI and breaks schema extensions.
            from pxr import Usd  # pyright: ignore[reportMissingImports]

            stage = Usd.Stage.Open(str(self.scene))
            matches_exact(
                json.loads(
                    stage.GetPrimAtPath("/RSSUnified").GetAttribute("rss:configuration").Get()
                ),
                self.configuration,
            )
            self._stage = stage
        return self._stage

    def joint(self, arm, name):
        from pxr import Usd, UsdPhysics  # pyright: ignore[reportMissingImports]

        root = self.stage().GetPrimAtPath(self.robot_root(arm))
        return next(
            p for p in Usd.PrimRange(root) if p.GetName() == name and p.IsA(UsdPhysics.Joint)
        )

    def apply_usd(self, robot, arm, profile=None, finger_effort_limit_n=None):
        from pxr import UsdPhysics  # pyright: ignore[reportMissingImports]

        kp, kd = [a.copy() for a in robot.get_gains()]
        effort = robot.get_max_efforts().copy()
        velocity = robot.get_joint_max_velocities().copy()
        inertia = robot.get_armatures().copy()
        friction = robot._physics_view.get_dof_friction_properties().copy()
        delays = []
        parameters = {}
        for i, name in enumerate(robot.dof_names):
            p = self.joint(arm, name)
            angular = name.startswith("joint")
            mode = "angular" if angular else "linear"
            d = UsdPhysics.DriveAPI(p, mode)
            scale = 180 / np.pi if angular else 1.0
            kp[:, i] = d.GetStiffnessAttr().Get() * scale
            kd[:, i] = d.GetDampingAttr().Get() * scale
            effort[:, i] = d.GetMaxForceAttr().Get()
            velocity[:, i] = p.GetAttribute("physxJoint:maxJointVelocity").Get() / scale
            if angular:
                inertia[:, i] = p.GetAttribute("physxJoint:armature").Get()
                friction[:, i] = [
                    p.GetAttribute("physxJointAxis:angular:" + n).Get() * s
                    for n, s in [
                        ("staticFrictionEffort", 1.0),
                        ("dynamicFrictionEffort", 1.0),
                        ("viscousFrictionCoefficient", scale),
                    ]
                ]
                delays.append(p.GetAttribute("rss:commandDelaySeconds").Get())
                parameters[name] = {
                    "kp_runtime_per_rad": float(kp[0, i] / 1e6),
                    "kv_runtime_per_rad": float(kd[0, i] / 1e6),
                    "armature_kg_m2": float(inertia[0, i] / 1e6),
                    "coulomb_friction": float(friction[0, i, 0] / 1e6),
                    "viscous_friction": float(friction[0, i, 2] / 1e6),
                    "command_delay_seconds": delays[-1],
                }
            else:
                inertia[:, i] = 0.0
                friction[:, i] = 0.0
        indices = np.arange(len(kp), dtype=np.int32)
        robot.set_gains(kp, kd, save_to_usd=False)
        robot.set_armatures(inertia)
        robot.set_max_efforts(effort)
        robot._physics_view.set_dof_max_velocities(velocity.astype(np.float32), indices)
        robot._physics_view.set_dof_friction_properties(friction.astype(np.float32), indices)
        actual = dict(
            kp=robot.get_gains()[0],
            kd=robot.get_gains()[1],
            armatures=robot.get_armatures(),
            friction=robot._physics_view.get_dof_friction_properties(),
            max_efforts=robot.get_max_efforts(),
            max_velocities=robot.get_joint_max_velocities(),
        )
        expected = dict(
            kp=kp,
            kd=kd,
            armatures=inertia,
            friction=friction,
            max_efforts=effort,
            max_velocities=velocity,
        )
        for k in actual:
            assert np.allclose(actual[k], expected[k], rtol=2e-6, atol=1e-6), (arm, k)
        report = dict(
            verified=True,
            arm=arm,
            parameters_source="USD authored attributes",
            parameter_source={"path": str(self.scene), "sha256": sha(self.scene)},
            parameters_si=parameters,
            delays_seconds=delays,
            dof_names=list(robot.dof_names),
            applied_finger_effort_limit_n=3.0,
            source_finger_effort_limit_n=3.0,
            save_to_usd=False,
            parameters_baked_into_usd=True,
            readback={k: v.tolist() for k, v in actual.items()},
        )
        self.readbacks[arm] = deepcopy(report)
        return report


def apply_v4(env, output, profile_path):
    from pxr import UsdGeom, UsdPhysics  # pyright: ignore[reportMissingImports]

    p = Path(profile_path)
    profile = json.loads(p.read_text())
    assert sha(p) == "7a120927b6c0be5c879ec849e097aab797f37791f2c4b7679b6ea0527bbc84b6"
    assert abs(UsdGeom.GetStageMetersPerUnit(env.stage) - 0.001) < 1e-12
    assert abs(UsdPhysics.GetStageKilogramsPerUnit(env.stage) - 1.0) < 1e-12
    records = []
    for arm, robot in zip(("left", "back"), env.robots):
        kp, kv = [x.copy() for x in robot.get_gains()]
        before_kp = kp.copy()
        before_kv = kv.copy()
        ids = [robot.dof_names.index(x["joint"]) for x in profile["joints"]]
        fingers = [robot.dof_names.index(n) for n in ("gripper_joint1", "gripper_joint2")]
        effort = robot.get_max_efforts().copy()
        velocity = robot.get_joint_max_velocities().copy()
        kp[:, ids] = np.asarray([x["kp_Nm_per_rad"] for x in profile["joints"]]) * 1e6
        kv[:, ids] = np.asarray([x["kv_Nm_s_per_rad"] for x in profile["joints"]]) * 1e6
        # Stage uses kg/mm. Prismatic force and distance conversion cancel for Kp/Kv.
        kp[:, fingers] = 800.0
        kv[:, fingers] = 20.0
        effort[:, fingers] = 3000.0
        robot.set_gains(kp, kv, save_to_usd=False)
        robot.set_max_efforts(effort)
        got_kp, got_kv = robot.get_gains()
        assert np.array_equal(got_kp, kp) and np.array_equal(got_kv, kv)
        assert np.array_equal(robot.get_max_efforts(), effort)
        assert np.array_equal(robot.get_joint_max_velocities(), velocity)
        other = [i for i in range(kp.shape[1]) if i not in ids + fingers]
        assert np.array_equal(got_kp[:, other], before_kp[:, other]) and np.array_equal(
            got_kv[:, other], before_kv[:, other]
        )
        records.append(
            dict(
                arm=arm,
                dof_names=robot.dof_names,
                kp=kp.tolist(),
                kv=kv.tolist(),
                max_efforts=effort.tolist(),
                joint_max_velocities=velocity.tolist(),
                baseline_kp=before_kp.tolist(),
                baseline_kv=before_kv.tolist(),
            )
        )
    save(
        output / "v4-gain-readback.json",
        dict(
            passed=True,
            scene_version="tower_v4",
            profile=profile,
            all_environments=env.n,
            records=records,
        ),
    )
    return records


def verify_v4(env, records):
    for robot, record in zip(env.robots, records):
        kp, kv = robot.get_gains()
        assert np.array_equal(kp, np.asarray(record["kp"], dtype=kp.dtype))
        assert np.array_equal(kv, np.asarray(record["kv"], dtype=kv.dtype))
        assert np.array_equal(robot.get_max_efforts(), np.asarray(record["max_efforts"]))
