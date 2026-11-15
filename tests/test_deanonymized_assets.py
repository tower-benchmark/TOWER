# SPDX-License-Identifier: Apache-2.0
"""Current and historical asset manifests remain usable after attribution edits."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from tower_eval.cli import _validate_task_assets

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    "repository", ["tower-benchmark/TOWER-Assets", "anonymous" + "-review-assets"]
)
def test_asset_manifest_repository_compatibility(tmp_path, repository):
    paths = ["station/station.usdz", "tasks/tower_transfer_2.usda"]
    for name in paths:
        path = tmp_path / name
        path.parent.mkdir(parents=True)
        path.write_bytes(b"asset")
    (tmp_path / "manifest.json").write_text(
        json.dumps({"repository": repository, "assets": [{"path": p} for p in paths]})
    )
    config = SimpleNamespace(environment={"asset_manifest": "manifest.json"})
    _validate_task_assets(config, tmp_path / "config.toml")
    (tmp_path / paths[0]).unlink()
    with pytest.raises(ValueError, match="not staged"):
        _validate_task_assets(config, tmp_path / "config.toml")


@pytest.mark.parametrize("repository", [None, "", " ", 42, "someone/other-assets"])
def test_asset_manifest_requires_repository_metadata(tmp_path, repository):
    (tmp_path / "manifest.json").write_text(json.dumps({"repository": repository, "assets": []}))
    with pytest.raises(ValueError, match="must target one of"):
        _validate_task_assets(
            SimpleNamespace(environment={"asset_manifest": "manifest.json"}),
            tmp_path / "config.toml",
        )


def test_legacy_downloader_points_to_working_public_download(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "legacy_download", ROOT / "tools/download_assets.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps({"schema": "tower-assets-v1", "assets": [{"path": "station.usdz"}]})
    )
    with pytest.raises(ValueError, match="benchmark_assets.py download"):
        module.download(manifest, tmp_path / "assets")
    assert not (tmp_path / "assets").exists()
