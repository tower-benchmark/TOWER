"""Isolated offline candidates; no hardware imports or actuator calls."""

import time

import casadi as ca  # pyright: ignore[reportMissingImports] - Optional native environment.
import numpy as np
import pinocchio as pin  # pyright: ignore[reportMissingImports] - Optional native environment.

from ..profiles import load_profile


class Solvers:
    radius: float

    def __init__(self, baseline, reference):
        self.profile = load_profile("posture-ipopt")
        spec = self.profile
        cost = spec["cost"]
        self.g = baseline
        self.ref = np.asarray(reference)
        self.lo = baseline.LOWER
        self.hi = baseline.UPPER
        q = ca.SX.sym("q", 7)
        target = ca.SX.sym("target", 4, 4)
        seed = ca.SX.sym("seed", 7)
        ref = ca.SX.sym("ref", 7)
        T = ca.SX.eye(4)
        model = pin.Model()
        parent = 0
        fixed = np.eye(4)
        for origin, idx, axis in baseline.kin.CHAIN:
            T = T @ ca.SX(origin)
            fixed = fixed @ origin
            if idx is not None:
                a = np.asarray(axis) / np.linalg.norm(axis)
                x, y, z = a
                K = ca.SX([[0, -z, y], [z, 0, -x], [-y, x, 0]])
                M = ca.SX.eye(4)
                M[:3, :3] = ca.SX.eye(3) + ca.sin(q[idx]) * K + (1 - ca.cos(q[idx])) * (K @ K)
                T = T @ M
                parent = model.addJoint(
                    parent, pin.JointModelRevoluteUnaligned(a), pin.SE3(fixed), f"joint{idx + 1}"
                )
                fixed = np.eye(4)
        T = T @ ca.SX(baseline.kin.TCP_LOCAL)
        model.lowerPositionLimit = self.lo.copy()
        model.upperPositionLimit = self.hi.copy()
        model.velocityLimit = np.full(7, spec["step_cap"]["equivalent_rad_s"])
        self.model = model
        # Both experimental solvers optimize exactly this residual and use the same FK.
        span = ca.DM(self.hi - self.lo)
        margin = np.radians(cost["limit_hinge"]["margin_deg"])
        residual = ca.vertcat(
            np.sqrt(cost["position_m"]) * (T[:3, 3] - target[:3, 3]),
            np.sqrt(cost["rotation_frobenius"]) * ca.vec(T[:3, :3] - target[:3, :3]),
            np.sqrt(cost["seed_pull"]) * (q - seed),
            np.sqrt(cost["reference_posture_pull_normalized_by_span"]) * (q - ref) / span,
            np.sqrt(cost["limit_hinge"]["weight"]) * ca.fmax(margin - (q - ca.DM(self.lo)), 0),
            np.sqrt(cost["limit_hinge"]["weight"]) * ca.fmax(margin - (ca.DM(self.hi) - q), 0),
        )
        self.fk = ca.Function("encoder_fk", [q], [T])
        self.res = ca.Function("residual", [q, target, seed, ref], [residual])
        self.jac = ca.Function("jacobian", [q, target, seed, ref], [ca.jacobian(residual, q)])
        self.ip = ca.nlpsol(
            "posture",
            "ipopt",
            {"x": q, "p": ca.vertcat(ca.vec(target), seed, ref), "f": ca.sumsqr(residual)},
            {
                "print_time": False,
                "ipopt.print_level": 0,
                "ipopt.sb": "yes",
                "ipopt.max_iter": spec["ipopt"]["max_iter"],
                "ipopt.max_cpu_time": spec["ipopt"]["max_cpu_time_s"],
                "ipopt.tol": spec["ipopt"]["tol"],
                "ipopt.acceptable_iter": spec["ipopt"]["acceptable_iter"],
            },
        )
        # Independent FK parity check, not an assumption about URDF/model frames.
        rng = np.random.default_rng(628)
        self.fk_error = max(
            float(np.max(np.abs(np.array(self.fk(v)) - baseline.kin.fk(v))))
            for v in rng.uniform(self.lo, self.hi, (32, 7))
        )
        assert self.fk_error < 1e-10, self.fk_error

    def solve(self, method, target, observed, arm):
        # The frozen module imported Pink before timing solves. Keep lazy loading
        # outside the same 0.2-second numerical acceptance budget.
        if method == "pink":
            import pink  # pyright: ignore[reportMissingImports] - Optional native environment.
            from pink.limits import Limit  # pyright: ignore[reportMissingImports]
            from pink.tasks import Task  # pyright: ignore[reportMissingImports]

        start = time.perf_counter()
        observed = np.asarray(observed)
        seed = np.clip(observed, self.lo, self.hi)
        radius = getattr(self, "radius", self.profile["step_cap"]["rad_per_row"])
        lo = np.maximum(self.lo, seed - radius)
        hi = np.minimum(self.hi, seed + radius)
        error = None
        if method == "baseline":
            r = self.g.solve_one(target, observed, arm, 0.1)
            if r["joints"] is None:
                return dict(
                    accepted=False,
                    reason="baseline_rejected",
                    latency_s=time.perf_counter() - start,
                    position_error_m=r["position_error_m"],
                    rotation_error_rad=r["rotation_error_rad"],
                )
            q = np.array(r["joints"])
            ok = r["accepted"]
        else:
            ref = self.ref[arm]
            q = seed.copy()
            ok = True
            try:
                if method == "posture_ipopt":
                    out = self.ip(x0=q, p=np.r_[target.ravel(order="F"), seed, ref], lbx=lo, ubx=hi)
                    q = np.array(out["x"]).ravel()
                    ok = bool(self.ip.stats()["success"])
                elif method == "pink":
                    qp = self.profile["pink"]
                    owner = self

                    class Objective(Task):
                        def __repr__(self):
                            return "MatchedEncoderPosePostureObjective"

                        def compute_error(self, c):
                            return np.array(owner.res(c.q, target, seed, ref)).ravel()

                        def compute_jacobian(self, c):
                            return np.array(owner.jac(c.q, target, seed, ref))

                    class Bounds(Limit):
                        def compute_qp_inequalities(self, q, dt):
                            return np.vstack([np.eye(7), -np.eye(7)]), np.r_[hi - q, q - lo]

                    config = pink.Configuration(self.model, self.model.createData(), q)
                    for _ in range(qp["iterations"]):
                        v = pink.solve_ik(
                            config,
                            [Objective(cost=1.0)],
                            qp["dt_s"],
                            solver=qp["solver"],
                            limits=[Bounds()],
                            damping=qp["damping"],
                        )
                        old = config.q.copy()
                        config.integrate_inplace(v, qp["dt_s"])
                        q = config.q.copy()
                        if np.max(np.abs(q - old)) < qp["convergence_rad"]:
                            break
                        if time.perf_counter() - start > self.profile["acceptance"]["latency_s"]:
                            break
                else:
                    raise ValueError(method)
            except Exception as exc:  # noqa: BLE001 - Pink version-specific NoSolutionFound boundary.
                if type(exc).__name__ in ("NoSolutionFound",):
                    ok = False
                    error = repr(exc)
                else:
                    raise RuntimeError(f"{method} implementation failure") from exc
        ep, er = self.g.kin.pose_error(self.g.kin.fk(q), target)
        pe = float(np.linalg.norm(ep))
        re = float(np.linalg.norm(er))
        elapsed = time.perf_counter() - start
        bounds = bool(
            np.all(q >= lo - self.profile["bounds_roundoff_rad"])
            and np.all(q <= hi + self.profile["bounds_roundoff_rad"])
        )
        accepted = bool(
            ok
            and np.isfinite(q).all()
            and bounds
            and pe <= self.profile["acceptance"]["position_error_m"]
            and re <= self.profile["acceptance"]["rotation_error_rad"]
            and elapsed <= self.profile["acceptance"]["latency_s"]
        )
        return dict(
            accepted=accepted,
            joints=q.tolist() if accepted else None,
            position_error_m=pe,
            rotation_error_rad=re,
            step_rad=float(np.max(abs(q - seed))),
            headroom_deg=float(np.degrees(np.minimum(q - self.lo, self.hi - q)).min()),
            latency_s=elapsed,
            reason=error or ("accepted" if accepted else "pose_bounds_or_time_rejection"),
        )
