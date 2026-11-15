"""Policy EEF native route with an exact-flip tripwire, no orientation fallback."""

import numpy as np
from nero_kinematics.fk.numpy_oracle import fk

from .ik_client import IKServiceClient
from .ik_service import SCHEMA, service_identity, validate_request


class PolicyIK:
    def __init__(self, options, *, client=None):
        self.client = client or IKServiceClient(options)
        self.identity = service_identity()

    def solve(self, poses, seed, dt_s=0.1):
        poses, seed = validate_request(
            dict(
                schema=SCHEMA,
                request_id="policy",
                poses=poses,
                seed=seed,
                dt_s=dt_s,
            )
        )
        # Exact-flip tripwire relative to CURRENT encoder FK (about 0.08 degrees).
        # This is not a hemisphere/posture guard: near flips and joint_absolute
        # bypass it, and a return from an already flipped pose can trip it. Keep
        # this tiny frozen-runtime divergence unchanged pending the owner decision.
        # Qualification rests on generation checks and S: planned approach/carry
        # plus recorded approach-phase inspection, per AGENTS.md. Keep the legacy
        # diagnostic reason below for receipt compatibility; no flip/retry/fallback.
        for arm in (0, 1):
            anchor = np.asarray(fk(seed[arm]))[:3, :3]
            for index, pair in enumerate(poses):
                rotation = pair[arm, :3, :3]
                if (
                    np.dot(anchor[:, 0], rotation[:, 0]) >= 1 - 1e-6
                    and np.dot(anchor[:, 2], rotation[:, 2]) <= -1 + 1e-6
                ):
                    return dict(
                        accepted=False,
                        joints=None,
                        diagnostics=[
                            dict(
                                row=index,
                                arm=arm,
                                accepted=False,
                                joints=None,
                                reason="reversed_180_grasp",
                                fallback="none",
                            )
                        ],
                    )
        # One call with the original targets. Numerical/planner rejection is
        # returned unchanged; a broken service aborts rather than resubmitting.
        return self.client.solve(poses, seed, dt_s=dt_s)

    def close(self):
        self.client.close()
