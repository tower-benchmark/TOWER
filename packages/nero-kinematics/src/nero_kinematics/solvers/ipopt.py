"""Absolute IK using Pinocchio symbolic FK and CasADi's IPOPT plugin.

Algorithm reference: AgileX piper_ros piper_pinocchio at ac41fcbc (MIT).
This is an independent implementation for named, calibrated robot chains.
No ROS, hardware, visualization, collision planner, or alternative IK fallback.
"""

import hashlib
import time
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from ..poses import pose, pose_error
from ..profiles import load_profile

_DEFAULT = load_profile("canonical")["solver"]


class PinocchioIK:
    """One optimizer and numerical FK data per arm. Limits are scalar joint angles.

    Continuous revolute joints use cos/sin in Pinocchio configuration space;
    IPOPT still optimizes physical angles, with finite deployment limits.
    All noncontrolled joints are locked at an explicit reference (neutral by
    default). Gripper-dependent TCP chains must supply a fixed calibrated TCP.
    """

    backend = "pinocchio_casadi"

    def __init__(
        self,
        urdf,
        joint_names,
        *,
        lower,
        upper,
        base_frame="base_link",
        tcp_frame="gripper_base",
        tcp_pose=(0, 0, 0, 0, 0, 0, 1),
        reference_configuration=None,
        position_tolerance_m=_DEFAULT["position_tolerance_m"],
        orientation_tolerance_rad=_DEFAULT["orientation_tolerance_rad"],
        position_weight=_DEFAULT["position_weight"],
        orientation_weight=_DEFAULT["orientation_weight"],
        seed_weight=_DEFAULT["seed_weight"],
        max_iter=_DEFAULT["max_iter"],
        max_time_s=_DEFAULT["max_time_s"],
        optimizer_tolerance=_DEFAULT["optimizer_tolerance"],
        verified_current_seed=_DEFAULT["verified_current_seed"],
    ):
        try:
            import casadi as ca  # pyright: ignore[reportMissingImports] - Optional native environment.
            import pinocchio as pin  # pyright: ignore[reportMissingImports] - Optional native environment.
            import pinocchio.casadi as cpin  # pyright: ignore[reportMissingImports] - Optional native environment.
        except ImportError as exc:
            raise RuntimeError(
                "pinocchio_casadi requires Pinocchio with CasADi bindings "
                "and CasADi/IPOPT; see docs/eef-pinocchio-control.md"
            ) from exc
        self.pin = pin
        self.position_tolerance_m = position_tolerance_m
        self.orientation_tolerance_rad = orientation_tolerance_rad
        self.max_time_s = max_time_s
        if not isinstance(verified_current_seed, bool):
            raise ValueError("verified_current_seed must be a boolean")
        self.verified_current_seed = verified_current_seed
        settings = [
            position_tolerance_m,
            orientation_tolerance_rad,
            position_weight,
            orientation_weight,
            seed_weight,
            max_time_s,
            optimizer_tolerance,
        ]
        if not np.isfinite(settings).all() or min(settings) <= 0:
            raise ValueError("IK tolerances, weights and soft time limit must be positive")
        if isinstance(max_iter, bool) or not isinstance(max_iter, int) or max_iter < 1:
            raise ValueError("max_iter must be a positive integer")
        names = tuple(joint_names)
        if len(names) != 7 or len(set(names)) != 7:
            raise ValueError("Expected seven unique controlled joint names")
        full = pin.buildModelFromUrdf(str(urdf))
        if any(not full.existJointName(n) for n in names):
            raise ValueError("Controlled joint missing from URDF")
        reference = (
            pin.neutral(full)
            if reference_configuration is None
            else np.asarray(reference_configuration, dtype=float)
        )
        if reference.shape != (full.nq,) or not np.isfinite(reference).all():
            raise ValueError("Invalid fixed reference configuration")
        if not pin.isNormalized(full, reference, 1e-6):
            raise ValueError("Reference configuration is not normalized")
        locked = [i for i in range(1, full.njoints) if full.names[i] not in names]
        self.model = pin.buildReducedModel(full, locked, reference)
        model = self.model
        self.joints = [model.joints[model.getJointId(n)] for n in names]
        if any(
            j.nv != 1 or j.nq not in (1, 2) or not j.shortname().startswith("JointModelR")
            for j in self.joints
        ):
            raise ValueError("Controlled joints must be scalar revolute or continuous revolute")
        self.lower, self.upper = (
            np.asarray(lower, dtype=float).copy(),
            np.asarray(upper, dtype=float).copy(),
        )
        if self.lower.shape != (7,) or self.upper.shape != (7,):
            raise ValueError("Deployment limits must have shape (7,)")
        for k, j in enumerate(self.joints):
            if j.nq == 1:
                self.lower[k] = max(self.lower[k], model.lowerPositionLimit[j.idx_q])
                self.upper[k] = min(self.upper[k], model.upperPositionLimit[j.idx_q])
        if not np.isfinite([self.lower, self.upper]).all() or np.any(self.lower >= self.upper):
            raise ValueError("Empty or nonfinite URDF/articulation limit intersection")
        if not model.existFrame(base_frame) or not model.existFrame(tcp_frame):
            raise ValueError("Base/TCP frame missing from URDF")
        self.base_id, self.tcp_id = model.getFrameId(base_frame), model.getFrameId(tcp_frame)
        tcp = pose(tcp_pose)
        self.offset = pin.SE3(Rotation.from_quat(tcp[3:]).as_matrix(), tcp[:3])
        self.data = model.createData()
        cm = cpin.Model(model)
        cd = cm.createData()
        x, target, seed = ca.SX.sym("q", 7), ca.SX.sym("target", 4, 4), ca.SX.sym("seed", 7)
        cq = ca.SX(pin.neutral(model))
        for k, j in enumerate(self.joints):
            if j.nq == 1:
                cq[j.idx_q] = x[k]
            else:
                cq[j.idx_q], cq[j.idx_q + 1] = ca.cos(x[k]), ca.sin(x[k])
        cpin.framesForwardKinematics(cm, cd, cq)
        transform = cd.oMf[self.base_id].inverse() * cd.oMf[self.tcp_id] * cpin.SE3(self.offset)
        error_transform = cpin.SE3(target).inverse() * transform
        # Smooth SE(3) embedding avoids log6's undefined CasADi derivatives at
        # identity in Pinocchio 3.8. Rotation acceptance still uses geodesic angle.
        cost = (
            position_weight * ca.sumsqr(error_transform.translation)
            + orientation_weight * 0.5 * ca.sumsqr(error_transform.rotation - ca.SX.eye(3))
            + seed_weight * ca.sumsqr(x - seed)
        )
        self.solver = ca.nlpsol(
            "ik",
            "ipopt",
            {"x": x, "p": ca.vertcat(ca.vec(target), seed), "f": cost},
            {
                "print_time": False,
                "ipopt.print_level": 0,
                "ipopt.sb": "yes",
                "ipopt.max_iter": max_iter,
                "ipopt.max_cpu_time": max_time_s,
                "ipopt.tol": optimizer_tolerance,
                "ipopt.acceptable_iter": 0,
            },
        )
        self.identity = dict(
            backend=self.backend,
            pinocchio=pin.__version__,
            casadi=ca.__version__,
            model_sha256=hashlib.sha256(Path(urdf).read_bytes()).hexdigest(),
            joint_names=names,
            base_frame=base_frame,
            tcp_frame=tcp_frame,
            tcp_pose=tcp.tolist(),
            lower=self.lower.tolist(),
            upper=self.upper.tolist(),
            nq=model.nq,
            nv=model.nv,
            reference_configuration=reference.tolist(),
            optimizer_tolerance=optimizer_tolerance,
            verified_current_seed=verified_current_seed,
            max_iter=max_iter,
            max_time_s=max_time_s,
            time_limit="soft CPU limit plus post-solve wall-time rejection",
            position_tolerance_m=position_tolerance_m,
            orientation_tolerance_rad=orientation_tolerance_rad,
            weights=[position_weight, orientation_weight, seed_weight],
        )
        self.reset()

    def reset(self):
        self.last_report = None

    def fk(self, q):
        angles = np.asarray(q, dtype=float)
        if angles.shape != (7,) or not np.isfinite(angles).all():
            raise ValueError("Expected seven finite joint angles")
        config = self.pin.neutral(self.model)
        for angle, j in zip(angles, self.joints):
            config[j.idx_q : j.idx_q + j.nq] = (
                angle if j.nq == 1 else [np.cos(angle), np.sin(angle)]
            )
        self.pin.framesForwardKinematics(self.model, self.data, config)
        tcp = self.data.oMf[self.base_id].inverse() * self.data.oMf[self.tcp_id] * self.offset
        return np.r_[tcp.translation, Rotation.from_matrix(tcp.rotation).as_quat()]

    def ik(self, target, seed, *, joint_radius_rad: float | None = None, margin_rad: float = 0.0):
        checked_at = time.perf_counter()
        target = pose(target)
        seed = np.asarray(seed, dtype=float)
        if seed.shape != (7,) or not np.isfinite(seed).all():
            raise ValueError("IK seed must be seven finite measured joint angles")
        lo, hi = self.lower.copy(), self.upper.copy()
        if (
            not bool(np.isscalar(margin_rad))
            or not np.isfinite(margin_rad)
            or margin_rad < 0
            or np.any(2 * margin_rad >= hi - lo)
        ):
            raise ValueError("Invalid joint margin")
        # Opt-in vendor behavior preserves the canonical out-of-bounds seed clipping.
        bounded_request = (
            margin_rad != 0 or joint_radius_rad is not None or self.verified_current_seed
        )
        if bounded_request and (np.any(seed < lo) or np.any(seed > hi)):
            raise ValueError("IK seed outside hard joint limits")
        if margin_rad:
            # A seed inside a soft margin may retreat; never move toward its hard limit.
            lo, hi = np.minimum(lo + margin_rad, seed), np.maximum(hi - margin_rad, seed)
        if joint_radius_rad is not None:
            if (
                not bool(np.isscalar(joint_radius_rad))
                or not np.isfinite(joint_radius_rad)
                or joint_radius_rad <= 0
            ):
                raise ValueError("Invalid joint step/velocity radius")
            lo, hi = (
                np.maximum(lo, seed - joint_radius_rad),
                np.minimum(hi, seed + joint_radius_rad),
            )
        if self.verified_current_seed:
            seed_fk = self.fk(seed)
            seed_residual = (
                pose_error(seed_fk, target) if np.isfinite(seed_fk).all() else (np.inf, np.inf)
            )
            if np.isfinite(seed_residual).all() and max(seed_residual) <= 1e-9:
                elapsed = time.perf_counter() - checked_at
                success = elapsed <= self.max_time_s
                self.last_report = dict(
                    backend=self.backend,
                    success=success,
                    solver_status="verified_current_seed",
                    iterations=0,
                    wall_s=elapsed,
                    error=None,
                    position_error_m=seed_residual[0],
                    orientation_error_rad=seed_residual[1],
                    joint_radius_rad=joint_radius_rad,
                    margin_rad=margin_rad,
                    max_joint_step_rad=0.0,
                )
                return seed.copy(), success
        matrix = np.eye(4)
        matrix[:3, :3], matrix[:3, 3] = Rotation.from_quat(target[3:]).as_matrix(), target[:3]
        started = time.perf_counter()
        solution, success, status = seed.copy(), False, "exception"
        error = None
        iterations = None
        residuals = (None, None)
        try:
            result = self.solver(
                x0=np.clip(seed, lo, hi),
                p=np.r_[matrix.ravel(order="F"), seed],
                lbx=lo,
                ubx=hi,
            )
            stats = self.solver.stats()
            status, iterations = stats["return_status"], stats.get("iter_count")
            solution = np.asarray(result["x"]).ravel()
            finite = np.isfinite(solution).all()
            if finite:
                actual = self.fk(solution)
                residuals = (
                    pose_error(actual, target) if np.isfinite(actual).all() else (np.inf, np.inf)
                )
                if not np.isfinite(residuals).all():
                    error = "Nonfinite FK residual"
                    residuals = (None, None)
            success = bool(
                stats["success"]
                and status == "Solve_Succeeded"
                and finite
                and residuals[0] is not None
                and np.all(solution >= lo - 1e-8)
                and np.all(solution <= hi + 1e-8)
                and residuals[0] <= self.position_tolerance_m
                and residuals[1] <= self.orientation_tolerance_rad
            )
        except RuntimeError as exc:
            error = str(exc)
        elapsed = time.perf_counter() - started
        if elapsed > self.max_time_s:
            success = False
            status = "wall_time_exceeded:" + status
        self.last_report = dict(
            backend=self.backend,
            success=success,
            solver_status=status,
            iterations=iterations,
            wall_s=elapsed,
            error=error,
            position_error_m=residuals[0],
            orientation_error_rad=residuals[1],
        )
        return solution, success


class PinocchioKinematics:
    """Two independent arm solvers, with environment-local world coordinates."""

    backend = "pinocchio_casadi"

    def __init__(self, arms, bases, environment_offset=(0, 0, 0)):
        if len(arms) != 2:
            raise ValueError("Expected left and back solvers")
        self.arms = tuple(arms)
        if arms[0] is arms[1]:
            raise ValueError("Each arm needs an independent solver instance")
        self.bases = np.asarray([pose(b) for b in bases])
        self.environment_offset = np.asarray(environment_offset, dtype=float)
        if (
            self.bases.shape != (2, 7)
            or self.environment_offset.shape != (3,)
            or not np.isfinite(self.environment_offset).all()
        ):
            raise ValueError("Invalid arm bases or clone offset")
        self.identity = dict(
            backend=self.backend,
            arms=[a.identity for a in arms],
            bases=self.bases.tolist(),
            environment_offset=self.environment_offset.tolist(),
        )

    @classmethod
    def from_simulator(cls, sim, **options):
        env = sim.env
        tcp = np.r_[env.lula.tcp_offset, env.lula.tcp_rotation.as_quat()]
        arms = []
        for mapping in sim.robot_config["arms"]:
            indices = mapping["joint_indices"]
            arms.append(
                PinocchioIK(
                    sim.output / "kinematics/nero.urdf",
                    mapping["joint_names"],
                    lower=sim.robot.lower[indices],
                    upper=sim.robot.upper[indices],
                    tcp_pose=tcp,
                    **options,
                )
            )
        return cls(arms, env.bases[0], env.offsets[0])

    def to_base(self, arm, target, frame):
        target = pose(target)
        if frame == "arm_base":
            return target
        if frame != "environment_world":
            raise ValueError("Unsupported EEF frame")
        base = self.bases[arm]
        inverse = Rotation.from_quat(base[3:]).inv()
        return np.r_[
            inverse.apply(target[:3] + self.environment_offset - base[:3]),
            (inverse * Rotation.from_quat(target[3:])).as_quat(),
        ]

    def ik(self, arm, target, seed):
        return self.arms[arm].ik(target, seed)

    def fk(self, arm, q):
        return self.arms[arm].fk(q)

    def report(self, arm):
        return self.arms[arm].last_report

    def reset(self):
        for arm in self.arms:
            arm.reset()
