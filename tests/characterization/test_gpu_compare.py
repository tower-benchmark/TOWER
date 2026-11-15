"""Exercise comparison boundaries without a simulator or GPU."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from scripts.maintainer import gpu_baseline
from scripts.maintainer.golden_support import read, sha, write
from scripts.maintainer.gpu_identity import reference_contract


def compare(baseline: Path, candidate: Path) -> dict:
    return gpu_baseline.compare(baseline, candidate, required_tasks=["task1-2"])


def artifact(
    root: Path, *, joint: float = 0, position: float = 0, angle: float = 0, pixel: int = 0
) -> None:
    reference = reference_contract()["reference_identity"]
    expert = reference["experts"]["task1-2"]
    directory = root / "task1-2"
    directory.mkdir(parents=True)
    qpos = np.zeros((2, 16))
    qpos[1, 0] = joint
    blocks = np.zeros((2, 1, 7))
    blocks[..., 6] = 1
    blocks[1, 0, 0] = position
    blocks[1, 0, 5:] = [np.sin(np.deg2rad(angle) / 2), np.cos(np.deg2rad(angle) / 2)]
    np.savez_compressed(directory / "trace.npz", time=[0.0, 0.1], qpos=qpos, blocks=blocks)
    Image.fromarray(np.full((4, 4, 3), pixel, dtype=np.uint8)).save(directory / "first.png")
    write(
        root / "summary.json",
        dict(
            schema="tower-gpu-baseline-v1",
            trials=[
                dict(
                    task="task1-2",
                    scene_id=expert["scene_id"],
                    expert_sha256=expert["sha256"],
                    source_manifest_sha256=reference["source_manifest_sha256"],
                    samples=2,
                    duration_s=0.1,
                    replay_completed=True,
                    task_success=True,
                    policy_action_failure=False,
                    trace_sha256=sha(directory / "trace.npz"),
                    cameras={name: "first.png" for name in ("left", "right", "front", "back")},
                )
            ],
        ),
    )


@pytest.mark.parametrize(
    "changes,passes",
    [
        ({}, True),
        ({"joint": 9e-5, "position": 9e-5, "angle": 0.04, "pixel": 4}, True),
        ({"joint": 1.01e-4}, False),
        ({"position": 1.01e-4}, False),
        ({"angle": 0.051}, False),
        ({"pixel": 5}, False),
    ],
)
def test_gpu_compare_boundaries(tmp_path: Path, changes: dict, passes: bool) -> None:
    artifact(tmp_path / "a")
    artifact(tmp_path / "b", **changes)
    assert compare(tmp_path / "a", tmp_path / "b")["passed"] is passes


def test_gpu_compare_rejects_changed_trace(tmp_path: Path) -> None:
    artifact(tmp_path / "a")
    artifact(tmp_path / "b")
    (tmp_path / "b/task1-2/trace.npz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Trace digest"):
        compare(tmp_path / "a", tmp_path / "b")


def test_quaternion_sign_is_the_same_orientation(tmp_path: Path) -> None:
    artifact(tmp_path / "a")
    artifact(tmp_path / "b")
    path = tmp_path / "b/task1-2/trace.npz"
    with np.load(path) as stream:
        arrays = dict(stream)
    arrays["blocks"][..., 3:] *= -1
    np.savez_compressed(path, **arrays)
    summary = read(tmp_path / "b/summary.json")
    summary["trials"][0]["trace_sha256"] = sha(path)
    write(tmp_path / "b/summary.json", summary)
    result = compare(tmp_path / "a", tmp_path / "b")
    assert result["passed"] and result["trials"][0]["max_block_orientation_deg"] == 0


@pytest.mark.parametrize("key,value", [("task_success", False), ("cameras", {})])
def test_gpu_compare_rejects_missing_or_changed_evidence(
    tmp_path: Path, key: str, value: object
) -> None:
    artifact(tmp_path / "a")
    artifact(tmp_path / "b")
    summary = read(tmp_path / "b/summary.json")
    summary["trials"][0][key] = value
    write(tmp_path / "b/summary.json", summary)
    with pytest.raises(ValueError, match="Replay identity|four-camera"):
        compare(tmp_path / "a", tmp_path / "b")


def test_gpu_compare_requires_every_protocol_task(tmp_path: Path) -> None:
    artifact(tmp_path / "a")
    artifact(tmp_path / "b")
    with pytest.raises(ValueError, match="every protocol task"):
        gpu_baseline.compare(tmp_path / "a", tmp_path / "b")
