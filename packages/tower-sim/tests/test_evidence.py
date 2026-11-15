import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace as NS

import h5py
import numpy as np
import pytest
from PIL import Image
from tower_sim.bench import cameras, evidence, render_freshness
from tower_sim.bench.camera_guard import TerminalGuard
from tower_sim.bench.camera_quality import CameraQuality
from tower_sim.bench.resources import sha

from scripts.maintainer.archive_root import ArchivedToolPath, ArchiveUnavailable

TASKS = [f"task{f}-{n}" for f in (1, 2, 4, 5, 6, 7) for n in (2, 3)]
NAMES = tuple(cameras.Native61Capture.names)


def compare_hdf5(left, right):
    with h5py.File(left) as a, h5py.File(right) as b:
        assert dict(a.attrs) == dict(b.attrs)
        assert set(a) == set(b)
        for name in a:
            assert a[name].dtype == b[name].dtype
            assert a[name].shape == b[name].shape
            if a[name].dtype.kind == "O":
                for x, y in zip(a[name], b[name], strict=True):
                    assert x.tobytes() == y.tobytes()
            else:
                np.testing.assert_array_equal(a[name], b[name])


def archive_pair(frozen_loader, tmp_path, pixels, *, valid=True):
    old = frozen_loader("_p706_archive", ArchivedToolPath("runtime_worker.py"))
    paths = []
    latest = []
    for label, cls in (("old", old.CameraArchive), ("new", evidence.CameraArchive)):
        path = tmp_path / (label + ".hdf5")
        paths.append(path)
        camera = NS(names=NAMES, observe=lambda: {"rgb": pixels[None].copy()})
        if not valid:
            camera.validate_observation = lambda *args: dict(
                valid=False, reason="recorded rejection"
            )
        archive = cls(camera, path, **({"quality": CameraQuality()} if label == "new" else {}))
        if valid:
            archive.append(0.0, encode_policy=False)
            latest.append(archive.latest)
            archive.append(0.05)
            archive.close(complete=True)
        else:
            with pytest.raises(RuntimeError, match="Invalid camera"):
                archive.append(0.0)
            assert archive.latest is None
            archive.close()
    compare_hdf5(*paths)
    if valid:
        assert latest[0] == latest[1]
    else:
        for label in ("old", "new"):
            raw = np.load(tmp_path / (label + ".hdf5.diagnostics") / "00000000.npz")
            np.testing.assert_array_equal(raw["pixels"], pixels)


@pytest.mark.archive
@pytest.mark.parametrize("valid", [True, False])
def test_camera_archives_and_rejection_equal(frozen_loader, tmp_path, valid):
    pixels = np.random.default_rng(16).integers(0, 256, (4, 240, 320, 3), dtype=np.uint8)
    archive_pair(frozen_loader, tmp_path, pixels, valid=valid)


@pytest.mark.archive
@pytest.mark.parametrize("task", TASKS)
def test_recorded_camera_and_physical_writers(frozen_loader, tmp_path, task):
    configured = os.environ.get("TOWER_P706_RECORDED_ROOT")
    if not configured:
        raise ArchiveUnavailable(
            "SKIP: supply --archive-root and TOWER_P706_RECORDED_ROOT for the frozen reference GPU recordings"
        )
    root = Path(configured) / task
    pixels = np.stack(
        [np.array(Image.open(root / f"first-{name}.png").convert("RGB")) for name in NAMES]
    )
    archive_pair(frozen_loader, tmp_path, pixels)
    old = frozen_loader("_p706_recorded_worker", ArchivedToolPath("batch_reference_worker.py"))
    runtime = old.runtime
    quality = CameraQuality(audit_every=1)
    for rgb in pixels:
        assert json.dumps(quality(rgb)) == json.dumps(runtime.camera_quality(rgb))
    seed = next((root / "simulation").glob("seed-*"))
    with h5py.File(seed / "episode/states.hdf5") as handle:
        values = {name: np.asarray(handle[name]) for name in handle}
    with (
        np.load(seed / "finger-link-poses.npz") as finger,
        np.load(seed / "finger-feedback.npz") as feedback,
    ):
        values.update(
            finger_poses=finger["poses_m"],
            finger_forces=feedback["forces_n"],
            finger_positions=feedback["positions_m"],
        )
    records = [
        {
            name: values[name][i]
            for name in (
                "qpos",
                "qvel",
                "eef",
                "eef_world",
                "blocks",
                "block_velocities",
                "time",
                "finger_poses",
                "finger_forces",
                "finger_positions",
            )
        }
        for i in range(len(values["time"]))
    ]
    env = NS(finger_geometry=json.loads((seed / "finger_geometry.json").read_text()))
    paths = []
    for label, cls in (("old", old.BatchEvidence), ("new", evidence.BatchEvidence)):
        writer = cls(env, 1)
        writer.records[0] = copy.deepcopy(records)
        path = tmp_path / label
        paths.append(path)
        writer.write(0, path)
    for name in (
        "episode/states.hdf5",
        "finger-link-poses.npz",
        "finger-feedback.npz",
        "finger_geometry.json",
    ):
        assert (
            (paths[0] / name).read_bytes()
            == (paths[1] / name).read_bytes()
            == (seed / name).read_bytes()
        ), name


@pytest.mark.archive
@pytest.mark.parametrize(
    "near,tick,label,enabled",
    [
        (10.0, 40, "v3_probe_env_1:back_wrist", True),
        (9.0, 40, "v3_probe_env_1:back_wrist", True),
        (10.0, 0, "native61_v2_2:left_wrist", True),
        (10.0, 40, "native61_v2_2:left_wrist", False),
        (10.0, 40, "invalid", True),
        (10.0, 40, "v3_probe_env_2:back_wrist", True),
    ],
)
def test_guard_matches_frozen(frozen_loader, tmp_path, near, tick, label, enabled):
    old = frozen_loader("_p706_guard", ArchivedToolPath("camera_guard_terminal.py"))

    def reject(*args):
        raise RuntimeError(f"Wrist near plane may clip a task block: {label} (9.9 < 10.0 mm)")

    env = NS(elapsed=np.array([tick / 400] * 2), n=2)
    args = ([], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0], np.eye(3), near, label)
    results = []
    for name, cls in (("old", old.TerminalGuard), ("new", TerminalGuard)):
        guard = cls(reject, env, tmp_path / name)
        guard.enabled = enabled
        try:
            result = guard.check(*args)
        except RuntimeError as exc:
            result = str(exc)
        rows = guard.consume(tick)
        for row in rows.values():
            row.pop("evidence_path")
        results.append((result, rows))
    assert results[0] == results[1]
    guard = TerminalGuard(reject, env, tmp_path / "scoped")
    with guard.reference_environment(1):
        assert guard.reference_index == 1
    assert guard.reference_index is None


@pytest.mark.archive
@pytest.mark.parametrize(
    "mode", ["ok", "initializing", "missing", "stale", "timeout", "bad-rational", "bad-target"]
)
def test_render_delivery_exact(frozen_loader, monkeypatch, mode):
    old = frozen_loader("_p706_fresh", ArchivedToolPath("render_freshness.py"))
    results = []
    for module in (old, render_freshness):
        snapshots = iter(
            [
                dict(exists=True, compute_count=1),
                dict(
                    exists=mode != "missing",
                    compute_count=1 if mode == "timeout" else 2,
                    reference_numerator=0 if mode == "stale" else 100,
                    reference_denominator=0 if mode == "bad-rational" else 1,
                ),
            ]
        )
        monkeypatch.setattr(
            module, "snapshot_dispatcher", lambda snapshots=snapshots: next(snapshots)
        )
        monkeypatch.setattr(
            module,
            "scheduled_target",
            lambda: dict(
                available=mode != "bad-target", numerator=100, denominator=1, tolerance_s=1e-7
            ),
        )
        calls = []
        try:
            result = module.verified_step(
                lambda calls=calls: calls.append("step"), initializing=mode == "initializing"
            )
        except RuntimeError as exc:
            result = str(exc)
        results.append((result, calls))
    assert results[0] == results[1]
    assert results[0][1] == ["step"]


def test_mount_provenance_has_no_filename_shape_exception(tmp_path):
    reference = {role: {"adopted_usd": np.eye(4).tolist()} for role in ("left_wrist", "back_wrist")}
    path = tmp_path / "reference.json"
    path.write_text(json.dumps(reference))
    env = NS(
        n=2,
        mounts=[{role: {"camera_to_gripper": np.eye(4)} for role in reference} for _ in range(2)],
        report={
            "wrist_camera_proxies": {
                role: {"camera_to_gripper_matrix": np.eye(4).tolist()} for role in reference
            }
        },
    )
    receipt = tmp_path / "receipt.json"
    digest = cameras.adopt_source_mounts(env, path, receipt)
    report = dict(
        source_camera_calibration_verified=True,
        source_wrist_mount_adoption_path=str(receipt),
        source_wrist_mount_adoption_sha256=digest,
        environment_count=2,
    )
    source = str(Path(cameras.__file__).resolve())
    hashes = {source: sha(source)}
    assert (
        cameras.verify_mount_evidence(report, source_hashes=hashes, batch_root=tmp_path)[
            "environment_count"
        ]
        == 2
    )
    with pytest.raises(ValueError, match="provenance"):
        cameras.verify_mount_evidence(report, source_hashes={source: "wrong"}, batch_root=tmp_path)
    value = json.loads(receipt.read_text())
    value["records"].append(value["records"][0])
    receipt.write_text(json.dumps(value))
    report["source_wrist_mount_adoption_sha256"] = sha(receipt)
    with pytest.raises(ValueError, match="Duplicate"):
        cameras.verify_mount_evidence(report, source_hashes=hashes, batch_root=tmp_path)
