# SPDX-License-Identifier: Apache-2.0
"""Public asset binding must not change scene semantics or escape its root."""

import copy
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from scripts.maintainer.archive_root import ArchivedToolPath

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS))
spec = importlib.util.spec_from_file_location(
    "public_benchmark_assets", TOOLS / "benchmark_assets.py"
)
assets = importlib.util.module_from_spec(spec)
spec.loader.exec_module(assets)


@pytest.mark.parametrize("name", ["../outside", "/absolute", "a/../../outside", "a\\outside", ""])
def test_asset_paths_cannot_escape_download_root(tmp_path, name):
    with pytest.raises(ValueError):
        assets.safe_path(tmp_path, name)


def test_symlink_cannot_redirect_an_asset_download(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    (root / "redirect").symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="escapes"):
        assets.safe_path(root, "redirect/outside")


@pytest.mark.archive
def test_scene_relocation_preserves_physics_contact_and_initial_state(tmp_path, v1_export):
    original = assets.benchmark.read(v1_export / "scenes/task7-3.json")["scenes"][0]
    files = {}
    for block in original["candidate"]["episode_spec"]["blocks"]:
        for key in ("asset_sha256", "texture_sha256"):
            if block[key]:
                files[block[key]] = {"sha256": block[key], "path": "objects/" + block[key]}
    (tmp_path / "manifest.json").write_text(
        json.dumps({"schema": "tower-benchmark-assets-v1", "files": list(files.values())})
    )
    resolved = assets.resolve_scene(
        "task7-3", original["seed"], tmp_path, benchmark_version="2026.11.0", v1_root=v1_export
    )
    assert resolved["initial_state"] == original["initial_state"]
    assert resolved["candidate"]["physics"] == original["candidate"]["physics"]
    assert (
        resolved["candidate"]["episode_spec"]["transfers"]
        == original["candidate"]["episode_spec"]["transfers"]
    )
    for before, after in zip(
        original["candidate"]["episode_spec"]["blocks"],
        resolved["candidate"]["episode_spec"]["blocks"],
        strict=True,
    ):
        restored = copy.deepcopy(after)
        for field in ("asset_reference", "texture_atlas"):
            if restored[field]:
                assert Path(restored[field]).is_relative_to(tmp_path)
            restored[field] = before[field]
        assert restored == before
    tool = next(b for b in resolved["candidate"]["episode_spec"]["blocks"] if b["tower"] == "tools")
    assert tool["texture_atlas"] is None
    (tmp_path / "manifest.json").write_text(
        json.dumps({"schema": "tower-benchmark-assets-v1", "files": []})
    )
    with pytest.raises(ValueError, match="lacks required"):
        assets.resolve_scene(
            "task7-3", original["seed"], tmp_path, benchmark_version="2026.11.0", v1_root=v1_export
        )


def test_downloader_requires_an_immutable_commit(tmp_path, monkeypatch):
    lock = tmp_path / "lock.json"
    lock.write_text(json.dumps({"revision": "main"}))
    monkeypatch.setattr(assets, "LOCK", lock)
    with pytest.raises(ValueError, match="pinned to a Hugging Face commit"):
        assets.download(tmp_path / "download")


def test_historical_pilot_pass_does_not_qualify_current_posture(tmp_path):
    review = {"passed": True, "native_inference_qualified": True}
    with pytest.raises(ValueError, match="current posture review"):
        assets.benchmark.posture_evidence(review, "task6-2", "policy-a")
    planned, recorded = tmp_path / "planned.npz", tmp_path / "recorded.npz"
    planned.write_bytes(b"planned approach/carry evidence")
    recorded.write_bytes(b"recorded approach/carry evidence")
    review["posture_review"] = dict(
        schema="tower-no-reversed-grasp-review-v1",
        task="task6-2",
        package_digest="policy-a",
        planned_approach_carry_passed=True,
        recorded_approach_carry_passed=True,
        reverse_180_grasp_absent=True,
        task_contact_geometry_preserved=True,
        planned_trajectory=dict(path=str(planned), sha256=assets.benchmark.sha256(planned)),
        recorded_approach_carry=dict(path=str(recorded), sha256=assets.benchmark.sha256(recorded)),
    )
    assert assets.benchmark.posture_evidence(review, "task6-2", "policy-a") == [planned, recorded]
    recorded.write_bytes(b"different recording")
    with pytest.raises(ValueError, match="changed posture evidence"):
        assets.benchmark.posture_evidence(review, "task6-2", "policy-a")


def test_kit_zero_exit_without_evidence_is_not_a_smoke_pass(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import benchmark_portable

    from scripts.maintainer import public_smoke

    monkeypatch.setattr(
        public_smoke.subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0)
    )
    with pytest.raises(RuntimeError, match="passing receipt"):
        benchmark_portable.main(
            [
                "smoke",
                "--assets",
                str(tmp_path / "assets"),
                "--task",
                "task1-2",
                "--seeds",
                "17",
                "--output",
                str(tmp_path / "new-output"),
                "--accept-eula",
            ]
        )


@pytest.mark.archive
def test_source_closure_ignores_synthetic_relative_module_names(tmp_path, monkeypatch):
    from types import SimpleNamespace

    path = ArchivedToolPath("runtime_worker.py")
    spec = importlib.util.spec_from_file_location("closure_test_worker", path)
    worker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(worker)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TOWER_SOURCE_ROOT", str(tmp_path))
    monkeypatch.delenv("TOWER_SDK_ROOT", raising=False)
    monkeypatch.setitem(
        sys.modules, "synthetic_sdk_classes", SimpleNamespace(__file__="_classes.py")
    )
    real = tmp_path / "real_source.py"
    real.write_text("value = 1\n")
    monkeypatch.setitem(sys.modules, "real_bound_source", SimpleNamespace(__file__=str(real)))
    assert worker.imported_sources() == {str(real): assets.benchmark.sha256(real)}
