"""Sharded GPU evidence must retain serial provenance and complete coverage."""

import copy
import shutil

import pytest

from scripts.maintainer import gpu_baseline as gpu
from scripts.maintainer.golden_support import read, write
from scripts.maintainer.gpu_identity import candidate_summary, reference_contract
from tests.characterization.test_comparison_routes import candidate_identity
from tests.characterization.test_gpu_compare import artifact


def shards(root):
    contract = reference_contract()
    for task in contract["contract"]["task_order"]:
        shard = root.parent / ("shard-" + task)
        artifact(shard)
        row = read(shard / "summary.json")["trials"][0]
        row.pop("source_manifest_sha256")
        expert = contract["reference_identity"]["experts"][task]
        row.update(task=task, scene_id=expert["scene_id"], expert_sha256=expert["sha256"])
        document = candidate_summary(candidate_identity(), contract["contract"], [row])
        destination = root / task
        shutil.copytree(shard / "task1-2", destination)
        write(destination / "summary.json", document)
    return contract


def test_assembly_matches_serial_schema_and_preserves_workers(tmp_path):
    root = tmp_path / "combined"
    contract = shards(root)
    paths = sorted(root.glob("*/summary.json"))
    before = {p: p.read_bytes() for p in paths}
    expected = candidate_summary(
        candidate_identity(),
        contract["contract"],
        [
            read(root / task / "summary.json")["trials"][0]
            for task in contract["contract"]["task_order"]
        ],
    )
    assert gpu.assemble(root) == expected == read(root / "summary.json")
    assert all(p.read_bytes() == value for p, value in before.items())
    with pytest.raises(FileExistsError):
        gpu.assemble(root)


@pytest.mark.parametrize(
    "damage",
    [
        "missing",
        "extra",
        "mixed_commit",
        "mixed_wheel",
        "incomplete",
        "wrong_task",
        "changed_trace",
        "missing_camera",
        "substitution",
    ],
)
def test_assembly_rejects_partial_mixed_or_changed_evidence(tmp_path, damage):
    root = tmp_path / "combined"
    shards(root)
    directory = root / "task2-3"
    path = directory / "summary.json"
    data = read(path)
    if damage == "missing":
        shutil.rmtree(directory)
    elif damage == "extra":
        (root / "task-unexpected").mkdir()
    elif damage == "mixed_commit":
        data["candidate_identity"]["commit"] = "d" * 40
    elif damage == "mixed_wheel":
        data["candidate_identity"]["wheel_sha256"]["tower_sim.whl"] = "d" * 64
    elif damage == "incomplete":
        data["trials"][0]["replay_completed"] = False
    elif damage == "wrong_task":
        data = copy.deepcopy(read(root / "task1-2/summary.json"))
    elif damage == "changed_trace":
        (directory / "trace.npz").write_bytes(b"changed")
    elif damage == "missing_camera":
        (directory / "first.png").unlink()
    elif damage == "substitution":
        data["candidate_identity"]["frozen_components"] = ["control"]
    if path.exists():
        write(path, data)
    with pytest.raises(ValueError):
        gpu.assemble(root)
    assert not (root / "summary.json").exists()
