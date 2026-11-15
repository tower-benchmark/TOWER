"""Explicit roots for pinned 2026.11 data, never Python source discovery.

Calibration, profiles and IK specifications are installed package data.
Scenes and asset payloads use caller-supplied public release roots.
"""

import errno
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path


def sha(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, allow_nan=False, default=lambda x: x.tolist()) + "\n"
    )
    temporary.replace(path)


@dataclass(frozen=True)
class Resources:
    paths: Mapping[str, Path]

    def path(self, name):
        path = Path(self.paths[name]).resolve()
        if not path.exists():
            raise FileNotFoundError(f"Missing bench resource {name}: {path}")
        return path

    def read(self, name):
        return json.loads(self.path(name).read_text())

    @classmethod
    def from_roots(cls, *, checkout, assets):
        checkout, assets = Path(checkout), Path(assets)
        from tower_sim.backend.resources import DATA

        runtime = DATA / "benchmark/runtime"
        capsule = runtime / "ik-capsule"
        template = assets / "configuration/template"
        appearance = assets / "configuration/appearance-v22"
        from tower_sim.backend.resources import resource_path

        return cls(
            dict(
                tasks=DATA / "registry/tasks.json",
                protocol=checkout / "benchmarks/2026.11.1/protocol.json",
                asset_lock=DATA / "benchmark/assets.lock.json",
                scene=assets / "station/station-dual-j2-wrist-shared-v3.usdz",
                scene_config=template / "scene.json",
                settings=template / "settings.json",
                robot=assets / "configuration/robot.json",
                kinematics=template / "kinematics",
                wrist_mounts=template / "wrist-mount-adoption.json",
                appearance_scene=appearance
                / "PNP_DUALD1200_RF_CALIB_BACK_CURTAIN_WRIST_V2_2_REAL71_REFINED_20261005_STANDALONE.usdz",
                renderer_controls=appearance / "renderer-controls.json",
                scene_cache=capsule / "runtime/scene-cache.json",
                runtime_profiles=capsule / "runtime-profiles",
                shared_ik_spec=runtime / "shared/shared-ik-spec.json",
                camera_profile=runtime / "shared/camera-profile-near10.json",
                motion_profile=resource_path(checkout, "motion_profile"),
                calibration_report=resource_path(checkout, "calibration_report"),
                tcp_cache=DATA / "calibration/tcp-v2.json",
                tcp_fit_receipt=DATA / "calibration/fit-equivalence.json",
                drive_profile=assets / "configuration/official-nero-profile.json",
                center_bond=assets / "configuration/USER_CENTER_BOND_AMENDMENT_C74_583.json",
            )
        )


def save_exclusive(path, value):
    """Publish complete bytes atomically; never replace a previous receipt."""
    path = Path(path)
    payload = json.dumps(value, indent=2, allow_nan=False) + "\n"
    fd, temporary = tempfile.mkstemp(prefix=".receipt-", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            os.fchmod(stream.fileno(), 0o644)
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except OSError as exc:
            if exc.errno in (errno.EXDEV, errno.ENOSYS, errno.EOPNOTSUPP, errno.EPERM):
                raise OSError("Atomic receipt publication requires hard-link support") from exc
            raise
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        Path(temporary).unlink(missing_ok=True)
