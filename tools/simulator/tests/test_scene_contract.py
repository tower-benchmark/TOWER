# SPDX-License-Identifier: Apache-2.0
"""Retained scene identity, physics validation and source portability contracts."""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]

from tower_eval.scenes import frozen_layout


def scene():
    # Synthetic geometry tests the shared parser without retaining 5.1 configs.
    return {
        "expected_layout": {
            "logical_id": 0,
            "seed": 0,
            "blocks": [
                {
                    "name": name,
                    "tower": "source",
                    "layer": index,
                    "slot": 0,
                    "pose": [0, 0, 0.015 + index * 0.03, 0, 0, 0, 1],
                    "dimensions_m": [0.15, 0.05, 0.03],
                    "mass_kg": 0.1056,
                    "fixed_on_table": index == 0,
                }
                for index, name in enumerate(("base", "target"))
            ],
            "transfers": [
                {
                    "block_id": "target",
                    "block_index": 1,
                    "source_layer": 1,
                    "destination_layer": 0,
                    "arm": 0,
                    "goal": [0.3, 0, 0.015, 0, 0, 0, 1],
                }
            ],
        }
    }


def test_external_model_identity_is_not_bound_to_scene():
    import tempfile

    from tower_eval.scenes import load_scene

    with tempfile.TemporaryDirectory() as directory:
        scene = Path(directory) / "scene.json"
        scene.write_text(
            json.dumps({"schema": "tower-scene-v1", "config": {}, "source": {}, "files_sha256": {}})
        )
        preset, _ = load_scene(
            scene,
            SimpleNamespace(),
            server_config={"checkpoint_id": "a-completely-different-model"},
        )
        assert preset["schema"] == "tower-scene-v1"


def test_frozen_runtime_verification_checks_mass_and_reports_reference_identity():
    from types import SimpleNamespace

    from tower_eval.scenes import verify_runtime

    class Prim:
        def GetAttribute(self, name):
            assert name == "physics:mass"
            return SimpleNamespace(Get=lambda: 0.1056)

    preset = {"expected_visuals": [{"mass_kg": 0.1056}], "expected_cameras": []}
    env = SimpleNamespace(
        visual_reports=preset["expected_visuals"],
        camera_report=[],
        env_paths=["/env"],
        stage=SimpleNamespace(GetPrimAtPath=lambda _: Prim()),
    )
    result = verify_runtime(preset, env, frozen=True)
    assert result["reference_kind"] == "frozen-task-profile"
    assert result["visuals_match_reference"] and result["physics_properties_match_reference"]
    preset["expected_visuals"][0]["mass_kg"] = 0.2
    with pytest.raises(ValueError, match="mass mismatch"):
        verify_runtime(preset, env, frozen=True)


@pytest.mark.parametrize("bad", ["pose", "target", "mass", "duplicate", "dimension"])
def test_reject_invalid_frozen_physics_or_targets(bad):
    data = scene()
    block = data["expected_layout"]["blocks"][0]
    if bad == "pose":
        block["pose"][3:] = [0, 0, 0, 0]
    elif bad == "target":
        data["expected_layout"]["transfers"][0]["block_index"] = 999
    elif bad == "mass":
        block["mass_kg"] = -1
    elif bad == "duplicate":
        block["name"] = data["expected_layout"]["blocks"][1]["name"]
    else:
        block["dimensions_m"] = [0.15, -0.05, 0.03]
    with pytest.raises(ValueError):
        frozen_layout(data)


def test_valid_frozen_layout_preserves_physics_and_targets():
    data = scene()
    spec = frozen_layout(data)
    assert spec.blocks[0].fixed_on_table
    assert not spec.blocks[1].fixed_on_table
    assert spec.blocks[1].mass_kg == 0.1056
    assert spec.transfers[0].block_id == "target"


def test_maintainer_and_golden_sources_have_no_host_paths():
    for root in (ROOT / "scripts/maintainer", ROOT / "tests/golden"):
        for path in root.rglob("*"):
            if path.suffix in (".py", ".json", ".md"):
                assert all(token not in path.read_text() for token in ("/" + "root/", "/" + "mnt/"))
