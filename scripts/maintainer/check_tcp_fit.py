# SPDX-License-Identifier: Apache-2.0
"""Maintainer-only, CPU-only authentic TCP refit; never copy raw channels.

Run with the archived interpreter and its nvidia-lula-no-cuda pip_prebundle
on PYTHONPATH, together with the installed evaluator sources.
See docs/reference/tcp-calibration.md. This is not a public runtime entry point.
"""

import argparse
import copy
import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path
from typing import cast

import h5py
import numpy as np
import scipy
from scipy.spatial.transform import Rotation
from tower_eval.artifacts import sha256
from tower_eval.calibration import cached_tcp_calibration_v2, tcp_calibration_key
from tower_eval.calibration.semantic import FIT_CHANNEL_SCOPE
from tower_sim.bench.asset_identity import station_identity

ROOT = Path(__file__).resolve().parents[2]
V1 = Path("simulator/benchmark-202611/runtime/ik-capsule/tcp-cache.json")
PROFILE = Path("packages/tower-sim/src/tower_sim/data/motion_profile_v1.json")
V2 = Path("benchmarks/2026.11/calibration/tcp-v2.json")
FIELDS = (
    "translation_m",
    "rotation_xyzw",
    "position_residual_p50_p95_m",
    "rotation_residual_p50_p95_deg",
)


def _dataset(handle, name) -> np.ndarray:
    return np.asarray(cast(h5py.Dataset, handle[name])[:])


def channel_identity(path):
    """Hash profile channels and the *actual* fitter inputs without exporting them.

    The latter is SHA-256(scope + NUL +, for each observations/qpos,eef in
    that order: compact sorted JSON {channel,dtype,shape} + NUL + C-order
    bytes). Arrays retain their stored dtype and every frame is covered.
    The legacy profile hash uses its original native float64 conversions.
    """
    profile_hash = hashlib.sha256()
    fit_hash = hashlib.sha256(FIT_CHANNEL_SCOPE.encode() + b"\0")
    with h5py.File(path, "r") as handle:
        episode = str(handle.attrs["episode_id"])
        for array in (
            _dataset(handle, "leader/qpos").astype(float),
            _dataset(handle, "observations/eef").astype(float).reshape(-1, 2, 7),
            _dataset(handle, "observations/gripper_status").astype(float),
            _dataset(handle, "timestamps/row_ros_ns")[:, 0],
        ):
            profile_hash.update(array.tobytes())
        for channel in ("observations/qpos", "observations/eef"):
            array = _dataset(handle, channel)
            header = {"channel": channel, "dtype": array.dtype.str, "shape": list(array.shape)}
            fit_hash.update(
                json.dumps(header, sort_keys=True, separators=(",", ":")).encode() + b"\0"
            )
            fit_hash.update(array.tobytes(order="C"))
    return {
        "episode_id": episode,
        "profile_channel_sha256": profile_hash.hexdigest(),
        "fit_channel_sha256": fit_hash.hexdigest(),
    }


def reject_key_changes(cache_path, key):
    """Exercise rejection for every semantic leaf, including separate geometry."""

    class Model:
        def calibrate_tcp(self, profile):
            raise AssertionError("No refit during cache validation")

    def leaves(value, path=()):
        for name, item in value.items():
            if isinstance(item, dict):
                yield from leaves(item, (*path, name))
            else:
                yield (*path, name)

    rejected = []
    original = Model.calibrate_tcp
    for path in leaves(key):
        changed = copy.deepcopy(key)
        node = changed
        for name in path[:-1]:
            node = node[name]
        old = node[path[-1]]
        node[path[-1]] = ("0" if old[0] != "0" else "1") + old[1:]
        try:
            with cached_tcp_calibration_v2(Model, cache_path, key=changed):
                pass
        except ValueError:
            rejected.append(".".join(path))
        else:
            raise AssertionError(f"Accepted changed TCP identity: {path}")
        _require(Model.calibrate_tcp is original, "Rejection probe did not restore the class")
    return rejected


def _require(condition, message):
    # Explicit, so `python -O` cannot skip an authenticity gate.
    if not condition:
        raise SystemExit(f"TCP fit check failed: {message}")


def run(args):
    from scripts.maintainer.archive_root import archive_checkout

    archived_v1 = archive_checkout(args.archive_root) / V1
    # Never initialize Kit/SimulationApp. Require NVIDIA's CPU-only Lula build.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    lula_version = importlib.metadata.version("nvidia-lula-no-cuda")
    import lula  # pyright: ignore[reportMissingImports] - Optional native environment.
    from tower_sim.backend import backend

    v1 = json.loads(archived_v1.read_text())
    profile = json.loads((ROOT / PROFILE).read_text())
    source = next(s for s in profile["sources"] if s["split"] == "fit")
    identity = channel_identity(args.episode)
    _require(
        identity["episode_id"] == source["episode_id"] == v1["tcp"]["fit_episode_id"],
        "fit episode identity",
    )
    _require(
        identity["profile_channel_sha256"]
        == source["channel_sha256"]
        == v1["tcp"]["source_channel_sha256"],
        "profile channel digest",
    )
    _require(sha256(ROOT / PROFILE) == v1["profile_sha256"], "motion profile digest")
    for name, digest in v1["kinematics_sha256"].items():
        _require(sha256(args.kinematics / name) == digest, f"kinematics {name}")
    stage_hash = station_identity(sha256(args.stage))
    _require(stage_hash == v1["usd_sha256"], "stage digest")
    key = tcp_calibration_key(
        profile=profile,
        stage_sha256=stage_hash,
        kinematics=args.kinematics,
        fit_channel_sha256=identity["fit_channel_sha256"],
    )
    # __new__ avoids export_kinematics and every simulator constructor. Only the
    # real model's unmodified fk/calibrate_tcp methods are used, with fresh TCP.
    model = backend.LulaModel.__new__(backend.LulaModel)
    robot = lula.load_robot(str(args.kinematics / "nero.yaml"), str(args.kinematics / "nero.urdf"))
    model.kinematics = robot.kinematics()
    model.frame = "gripper_base"
    model.tcp_offset = np.zeros(3)
    model.tcp_rotation = Rotation.identity()
    model.tcp_report = {}
    source["path"] = str(args.episode)
    model.calibrate_tcp(profile)
    differences = {}
    for field in FIELDS:
        actual = np.asarray(model.tcp_report[field])
        expected = np.asarray(v1["tcp"][field])
        np.testing.assert_allclose(actual, expected, atol=1e-9, rtol=0, err_msg=field)
        differences[field] = float(np.max(np.abs(actual - expected)))
    v2 = json.loads((ROOT / V2).read_text())
    _require(v2["tcp"] == v1["tcp"], "migration must copy original floats, not refit output")
    _require(v2["lineage"] == {"path": V1.as_posix(), "sha256": sha256(archived_v1)}, "v2 lineage")
    original = backend.LulaModel.calibrate_tcp
    with cached_tcp_calibration_v2(
        backend.LulaModel, ROOT / V2, key=key, kinematics=args.kinematics
    ):
        model.calibrate_tcp(profile)
        _require(model.tcp_report == v1["tcp"], "restored v2 report")
    _require(backend.LulaModel.calibrate_tcp is original, "class restored after v2 context")
    return {
        "schema": "tower-tcp-fit-equivalence-receipt-v1",
        "status": "passed",
        "execution": "CPU-only Lula FK; no SimulationApp; CUDA_VISIBLE_DEVICES empty",
        "environment": {
            "python": platform.python_version(),
            "machine": platform.machine(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "h5py": h5py.__version__,
            "nvidia-lula-no-cuda": lula_version,
        },
        "lineage": v2["lineage"],
        "backend_sha256": sha256(backend.__file__),
        "profile_sha256": sha256(ROOT / PROFILE),
        "key": key,
        "comparison": {"atol": 1e-9, "rtol": 0, "max_abs_difference": differences},
        "rejected_key_changes": reject_key_changes(ROOT / V2, key),
        "raw_fit_data_redistributed": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--kinematics", type=Path, required=True)
    parser.add_argument("--stage", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--archive-root", type=Path)
    parser.add_argument(
        "--allow-skip",
        action="store_true",
        help="Allow missing archive evidence (still reported as skipped)",
    )
    args = parser.parse_args()
    from scripts.maintainer.archive_root import ArchiveUnavailable

    try:
        receipt = run(args)
    except ArchiveUnavailable as error:
        receipt = dict(passed=False, skipped=True, reason=str(error))
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
    return 77 if receipt.get("skipped") and not args.allow_skip else 0


if __name__ == "__main__":
    raise SystemExit(main())
