"""Versioned shared encoder IK; physical-target conversion is explicit.

No task-specific posture, bounds, fallback solver or hardware imports.
Offline keypoint feasibility is separate from the 10 Hz execution chain.
"""

import time
from types import SimpleNamespace

import numpy as np

from ..fk import numpy_oracle as kin
from ..profiles import COMMAND_BOUNDS, load_profile
from .posture_ipopt import Solvers

_SPEC = load_profile("command")
_SOLVER = load_profile("posture-ipopt")
SPEC_SHA = _SPEC["source_sha256"]
LOWER, UPPER = np.array(COMMAND_BOUNDS.lower_rad), np.array(COMMAND_BOUNDS.upper_rad)
KILL_LO, KILL_HI = (
    np.array(_SPEC["kill_line"]["lower_rad"]),
    np.array(_SPEC["kill_line"]["upper_rad"]),
)
REFERENCE = np.array(
    [load_profile("posture-ipopt")["reference_posture"][a] for a in ("left", "back")]
)


class SharedIK:
    def __init__(self, *, physical_from_encoder=None):
        self.physical_from_encoder = physical_from_encoder
        self.core = Solvers(
            SimpleNamespace(LOWER=LOWER.copy(), UPPER=UPPER.copy(), kin=kin), REFERENCE.copy()
        )
        self.identity = dict(
            protocol="v5-shared-ik-v1",
            spec_sha256=SPEC_SHA,
            model_fk_max_error=self.core.fk_error,
            fallback="none",
        )

    def encoder(self, target, seed, arm, *, offline_keypoint=False):
        target, seed = np.asarray(target, float), np.asarray(seed, float)
        if (
            seed.shape != (7,)
            or target.shape != (4, 4)
            or not np.isfinite(seed).all()
            or not np.isfinite(target).all()
        ):
            raise ValueError("Invalid IK shapes/nonfinite input")
        if not np.allclose(target[3], [0, 0, 0, 1]) or not np.allclose(
            target[:3, :3].T @ target[:3, :3], np.eye(3), atol=1e-6
        ):
            raise ValueError("Invalid transform")
        if np.any(seed < KILL_LO - 1e-7) or np.any(seed > KILL_HI + 1e-7):
            return dict(accepted=False, joints=None, reason="seed_outside_kill_line")
        self.core.radius = (
            _SOLVER["offline_radius_rad"]
            if offline_keypoint
            else _SOLVER["step_cap"]["rad_per_row"]
        )
        r = self.core.solve("posture_ipopt", target, seed, arm)
        r.update(offline_keypoint=offline_keypoint, spec_sha256=SPEC_SHA)
        if (
            r["accepted"]
            and not offline_keypoint
            and np.max(np.abs(np.asarray(r["joints"]) - seed))
            > _SOLVER["step_cap"]["rad_per_row"] + _SOLVER["bounds_roundoff_rad"]
        ):
            r.update(accepted=False, joints=None, reason="step_from_unclipped_seed")
        return r

    def physical(self, target, seed, width, arm, *, offline_keypoint=False):
        """Physical desired pose -> encoder target; apply J2 conversion once."""
        if self.physical_from_encoder is None:
            raise ValueError("Physical conversion must be supplied explicitly")
        started = time.perf_counter()
        target = np.asarray(target)
        seed = np.asarray(seed)
        current = seed.copy()
        iterations = []
        name = ("left", "back")[arm]
        for _ in range(8):
            ph = self.physical_from_encoder(current, width, name)
            adjusted = kin.fk(current) @ np.linalg.inv(kin.fk(ph)) @ target
            r = self.encoder(adjusted, seed, arm, offline_keypoint=offline_keypoint)
            iterations.append(r)
            if not r["accepted"]:
                return dict(
                    accepted=False,
                    joints=None,
                    reason=r["reason"],
                    inner=iterations,
                    latency_s=time.perf_counter() - started,
                )
            current = np.asarray(r["joints"])
            ph = self.physical_from_encoder(current, width, name)
            ep, er = kin.pose_error(kin.fk(ph), target)
            pe, re = float(np.linalg.norm(ep)), float(np.linalg.norm(er))
            if pe < 2e-5 and re < 2e-4:
                break
        elapsed = time.perf_counter() - started
        ok = (
            pe <= _SOLVER["acceptance"]["position_error_m"]
            and re <= _SOLVER["acceptance"]["rotation_error_rad"]
            and elapsed <= _SOLVER["acceptance"]["latency_s"]
        )
        ok = ok and bool(
            np.all(ph >= kin.LIMITS[:, 0] - 1e-7) and np.all(ph <= kin.LIMITS[:, 1] + 1e-7)
        )
        return dict(
            accepted=ok,
            joints=current.tolist() if ok else None,
            physical_joints=ph.tolist(),
            adjusted_encoder_target=adjusted.tolist(),
            position_error_m=pe,
            rotation_error_rad=re,
            reason="accepted" if ok else "physical_pose_bounds_or_time",
            inner=iterations,
            latency_s=elapsed,
        )

    def chunk(self, targets, seed):
        targets = np.asarray(targets)
        state = np.asarray(seed).copy()
        rows = []
        reports = []
        if targets.ndim != 4 or targets.shape[1:] != (2, 4, 4) or state.shape != (2, 7):
            raise ValueError("Invalid chunk")
        for pair in targets:
            q = []
            for a in range(2):
                r = self.encoder(pair[a], state[a], a)
                reports.append(r)
                if not r["accepted"]:
                    return dict(accepted=False, joints=None, diagnostics=reports)
                q.append(r["joints"])
            state = np.asarray(q)
            rows.append(q)
        return dict(accepted=True, joints=rows, diagnostics=reports)


def headroom(q):
    q = np.asarray(q)
    return np.degrees(np.minimum(q - LOWER, UPPER - q))
