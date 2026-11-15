"""Fixed pose inputs for all six Appendix G implementations; run in isolation."""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[variable] = "1"

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from scripts.maintainer.archive_root import ArchivedPath, ArchivedToolPath

RUNTIME = ArchivedPath("runtime")
JOINTS = np.array(
    [
        [0.1, -0.3, 0.4, 0.6, -0.2, 0.15, 0.1],
        [-0.4, 0.2, -0.3, 0.8, 0.25, -0.1, -0.2],
        [0.3, -0.45, 0.2, 1.1, -0.35, 0.25, 0.4],
    ]
)
OFFSET = np.array([0.003, -0.004, 0.002, -0.003, 0.004, -0.002, 0.003])


def module_file(name: str, path: Path | ArchivedPath) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def kin_module() -> Any:
    sys.path.insert(0, str(ArchivedToolPath("")))
    return module_file("_p0_kin", RUNTIME / "ik-capsule/tools/kin.py")


def scenario(kind: str, *, library: bool = False) -> dict[str, Any]:
    if library:
        from nero_kinematics import create_solver

        kin = create_solver("numpy-dls")
    else:
        kin = kin_module()
    lower = np.array([-2.70526, -1.74, -2.75, -1.01, -2.75, -0.73, -1.5707963])
    upper = np.array([2.70526, 1.74, 2.75, 2.14, 2.75, 0.95, 1.5707963])
    rows: list[dict[str, Any]] = []
    if kind == "numpy-dls":
        for q in JOINTS:
            target = kin.fk(q)
            result, pe, re = kin.ik(target, q + OFFSET)
            assert pe < 1e-4 and re < 1e-3
            rows.append(
                dict(
                    input=q.tolist(),
                    target=target.tolist(),
                    seed=(q + OFFSET).tolist(),
                    joints=result.tolist(),
                )
            )
    elif kind == "sdk-mdh":
        if library:
            sdk = create_solver("sdk-mdh")
        else:
            sys.path.insert(0, str(RUNTIME / "support"))
            sdk = importlib.import_module("task2_fixed.eef_ik_movejs")
        rows = [dict(input=q.tolist(), transform=sdk.sdk_transform(q).tolist()) for q in JOINTS]
    elif kind in ("posture-ipopt", "pink"):
        if library:
            solver = create_solver(kind, reference=[JOINTS[0], JOINTS[1]])
        else:
            control = module_file("_p0_control", ArchivedToolPath("control.py"))
            solvers = module_file("_p0_controlled", RUNTIME / "shared/controlled_solvers.py")
            baseline = SimpleNamespace(
                LOWER=control.GENERATION_LOWER, UPPER=control.GENERATION_UPPER, kin=kin
            )
            solver = solvers.Solvers(baseline, [JOINTS[0], JOINTS[1]])
        for index, q in enumerate(JOINTS):
            target = kin.fk(q)
            result = solver.solve(kind.replace("-", "_"), target, q + OFFSET, index % 2)
            assert result["accepted"], result
            rows.append(
                dict(
                    input=q.tolist(),
                    target=target.tolist(),
                    seed=(q + OFFSET).tolist(),
                    joints=result["joints"],
                )
            )
        # A reversed tool target must fail the bounded single-step solve. No alternate
        # target or seed is attempted. Dispatch no-retry is tested independently.
        target = kin.fk(JOINTS[0])
        # The task TCP uses x for approach and z for finger closing. A symmetric
        # reverse grasp rotates about x, preserving approach and reversing z.
        target[:3, :3] = target[:3, :3] @ np.diag([1.0, -1.0, -1.0])
        rejected = solver.solve(kind.replace("-", "_"), target, JOINTS[0], 0)
        assert not rejected["accepted"] and rejected["joints"] is None
    else:
        if library:
            solver = create_solver(kind)
        else:
            if kind == "package":
                sys.path.insert(0, str(ROOT / "src/tower-eval/src"))
                cls = importlib.import_module("tower_eval.kinematics.pinocchio_ik").PinocchioIK
            elif kind == "standalone":
                sys.path.insert(0, str(ROOT / "packages/nero-kinematics/src"))
                cls = importlib.import_module("nero_eef_ik._solver").PinocchioIK
            elif kind == "vendor":
                vendor = RUNTIME / "support/task2_fixed/vendor/eef_ik_movejs"
                cls = module_file("_p0_vendor", vendor / "nero_eef_ik/_solver.py").PinocchioIK
            else:
                raise ValueError(kind)
            model = (
                ROOT / "packages/nero-kinematics/src/nero_kinematics/models/nero_sim_reference.urdf"
            )
            options: dict[str, Any] = {}
            if kind == "vendor":
                vendor = RUNTIME / "support/task2_fixed/vendor/eef_ik_movejs"
                config = json.loads((vendor / "models/pinocchio_config.json").read_text())
                model = vendor / "models/nero_sdk_mdh.urdf"
                options = dict(config["solver"], tcp_frame="link7")
            solver = cls(
                model, [f"joint{i}" for i in range(1, 8)], lower=lower, upper=upper, **options
            )
        for q in JOINTS:
            target = solver.fk(q)
            result, accepted = solver.ik(target, q + OFFSET)
            assert accepted, solver.last_report
            rows.append(
                dict(
                    input=q.tolist(),
                    target=target.tolist(),
                    seed=(q + OFFSET).tolist(),
                    joints=result.tolist(),
                )
            )
    return dict(implementation=kind, cases=rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("kind")
    parser.add_argument("output", type=Path)
    parser.add_argument("--library", action="store_true")
    args = parser.parse_args()
    args.output.write_text(
        json.dumps(scenario(args.kind, library=args.library), indent=2, sort_keys=True) + "\n"
    )


if __name__ == "__main__":
    main()
