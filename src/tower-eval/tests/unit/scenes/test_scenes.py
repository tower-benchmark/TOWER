# SPDX-License-Identifier: Apache-2.0
import json
from copy import deepcopy
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest
from tower_eval.artifacts import sha256
from tower_eval.scenes import compare_initial_state, load_scene, verify_layout


def test_diagnostic_initial_block_overrides_preserve_physics_and_other_blocks():
    from tower_eval.model import Block, EpisodeSpec
    from tower_eval.scenes import initial_block_overrides

    base = Block("base", "source", 0, 0, (0, 0, 0.015, 0, 0, 0, 1), fixed_on_table=True)
    block = Block("movable", "source", 2, 0, (0, 0, 0.075, 0, 0, 0, 1))
    spec = EpisodeSpec(0, 0, [base, block], [])
    pose = [0.2, 0.3, 0.045, 0, 0, 0, 1]
    result = initial_block_overrides({"initial_block_pose_overrides": {"movable": pose}}, spec)
    assert result.blocks[0] == base
    assert result.blocks[1].pose == tuple(pose)
    assert not result.blocks[1].fixed_on_table
    assert spec.blocks[1].pose == block.pose
    with pytest.raises(ValueError):
        initial_block_overrides({"initial_block_pose_overrides": {"base": pose}}, spec)


def preset_files(tmp_path):
    usd, manifest, texture = [
        tmp_path / name for name in ("stage.usdz", "manifest.json", "texture.png")
    ]
    for file in (usd, manifest, texture):
        file.write_bytes(b"original")
    server = {
        "checkpoint_id": "5307",
        "artifact_identity": {"model_sha256": "weights", "norm_stats_sha256": "stats"},
    }
    preset = {
        "schema": "tower-training-scene-v1",
        **deepcopy(server),
        "source": {"training_episode": "archived"},
        "config": {
            "stage": str(usd),
            "randomization_manifest": str(manifest),
            "grasp_point_offsets_m": {"source_l2_a": [0, 0, 0.004]},
        },
        "files_sha256": {str(p): sha256(p) for p in (usd, manifest, texture)},
    }
    path = tmp_path / "scene.json"
    path.write_text(json.dumps(preset))
    return path, server, texture


def test_scene_load_binds_checkpoint_and_rejects_changed_texture(tmp_path):
    path, server, texture = preset_files(tmp_path)
    config = SimpleNamespace(seed=42)
    load_scene(path, config, server)
    assert config.seed == 42
    assert config.randomization_manifest.endswith("manifest.json")
    texture.write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_scene(path, config, server)


def test_scene_load_allows_canonical_blocks_without_randomization_manifest(tmp_path):
    path, server, _ = preset_files(tmp_path)
    preset = json.loads(path.read_text())
    preset["config"]["randomization_manifest"] = None
    path.write_text(json.dumps(preset))
    config = SimpleNamespace(seed=42)
    load_scene(path, config, server)
    assert config.randomization_manifest is None


def test_scene_rejects_non_path_randomization_manifest(tmp_path):
    path, server, _ = preset_files(tmp_path)
    preset = json.loads(path.read_text())
    preset["config"]["randomization_manifest"] = {"unexpected": "inline manifest"}
    path.write_text(json.dumps(preset))
    with pytest.raises(ValueError, match="path or null"):
        load_scene(path, SimpleNamespace(), server)


@pytest.mark.parametrize("key", ["checkpoint_id", "model_sha256", "norm_stats_sha256"])
def test_scene_rejects_wrong_checkpoint(tmp_path, key):
    path, server, _ = preset_files(tmp_path)
    if key == "checkpoint_id":
        server[key] = "other"
    else:
        server["artifact_identity"][key] = "other"
    with pytest.raises(ValueError, match="mismatch"):
        load_scene(path, SimpleNamespace(), server)


def test_scene_rejects_conflicting_usd(tmp_path):
    path, server, _ = preset_files(tmp_path)
    with pytest.raises(ValueError, match="conflicts"):
        load_scene(path, SimpleNamespace(), server, tmp_path / "different.usdz")


@dataclass
class Layout:
    blocks: list
    transfers: list


@pytest.mark.parametrize("change", ["pose", "asset_scale", "mass_kg", "asset_reference", "goal"])
def test_layout_rejects_geometry_or_goal_drift(change):
    expected = {
        "blocks": [
            {
                "pose": [0, 0, 0, 0, 0, 0, 1],
                "asset_scale": [333, 3000, 1000],
                "mass_kg": 0.106,
                "asset_reference": "variant.usda",
            }
        ],
        "transfers": [{"goal": [0, 0, 0, 0, 0, 0, 1]}],
    }
    actual = deepcopy(expected)
    if change == "goal":
        actual["transfers"][0]["goal"][0] += 0.001
    elif change == "asset_reference":
        actual["blocks"][0][change] = "default.usda"
    elif change == "mass_kg":
        actual["blocks"][0][change] = 0.1056
    else:
        actual["blocks"][0][change][0] += 0.01
    with pytest.raises(ValueError, match="Scene mismatch"):
        verify_layout({"expected_layout": expected}, Layout(**actual))


def test_initial_measurements_reject_mismatched_settled_scene():
    preset = {"initial_observation": {"qpos": [0.0] * 16, "blocks": [[0.0] * 7] * 12}}
    state = SimpleNamespace(qpos=np.zeros(16), blocks=np.zeros((12, 7)))
    assert compare_initial_state(preset, state)["max_block_position_error_m"] == 0
    state.blocks[10, 2] = 0.002
    with pytest.raises(ValueError, match="initial-state mismatch"):
        compare_initial_state(preset, state)
