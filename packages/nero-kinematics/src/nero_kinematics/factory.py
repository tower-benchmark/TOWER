"""Named profiles; importing this module does not import native solvers."""

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from .profiles import COMMAND_BOUNDS, load_profile

MODELS = Path(__file__).resolve().parent / "models"


def create_solver(profile: str, *, reference=None) -> Any:
    """Build the versioned profile, with per-arm reference posture if requested.

    Legacy profiles reproduce historical numerics. New generation must explicitly
    use COMMAND_BOUNDS and task-relative posture_evidence/solve_grasp.
    """
    if profile == "numpy-dls":
        from .fk import numpy_oracle

        return numpy_oracle
    if profile == "sdk-mdh":
        from .fk import sdk_mdh

        return sdk_mdh
    spec = load_profile(profile)
    if profile in ("posture-ipopt", "pink"):
        from .fk import numpy_oracle
        from .solvers.posture_ipopt import Solvers

        if reference is None:
            reference = [spec["reference_posture"][arm] for arm in ("left", "back")]
        baseline = SimpleNamespace(
            LOWER=np.array(COMMAND_BOUNDS.lower_rad),
            UPPER=np.array(COMMAND_BOUNDS.upper_rad),
            kin=numpy_oracle,
        )
        return Solvers(baseline, reference)
    if profile not in ("canonical", "package", "standalone", "vendor"):
        raise ValueError(f"Not a solver profile: {profile}")
    from .solvers.ipopt import PinocchioIK

    return PinocchioIK(
        MODELS / spec["model"],
        [f"joint{i}" for i in range(1, 8)],
        lower=spec["lower"],
        upper=spec["upper"],
        tcp_frame=spec["tcp_frame"],
        **spec["solver"],
    )
