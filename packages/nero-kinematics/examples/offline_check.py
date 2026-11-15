# SPDX-License-Identifier: Apache-2.0
"""Offline numerical check only: no hardware initialization or commands."""

import json
from pathlib import Path

import numpy as np
from nero_eef_ik import ArmIK

arm = ArmIK(Path(__file__).with_name("nero_sim_reference.json"))
identity = arm.identity
# Test state from model limits, not an instruction to move hardware here.
q = (np.array(identity["lower"]) + identity["upper"]) / 2
target = arm.fk(q)
result = arm.solve(target, q + 0.01)
print(json.dumps({"identity": identity, "target": target, "result": result}, indent=2))
if not result["success"]:
    raise SystemExit("Numerical check failed; see diagnostics")
