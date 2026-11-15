"""Every released scene must construct valid CPU runtime inputs using public assets."""

import json
from pathlib import Path

import numpy as np
import pytest
from tower_eval.tasks import benchmark_task
from tower_sim.bench import data_release, scene, scene_format, tasks

from scripts.maintainer.fetch_public_test_data import public_asset_root

ROOT = Path(__file__).resolve().parents[1]
SCENES = json.loads((ROOT / "benchmarks/2026.11.1/scene-index.json").read_bytes())["scenes"]


@pytest.fixture(scope="module")
def public_scenes():
    root = public_asset_root()
    return root, json.loads((root / "manifest.json").read_bytes())


@pytest.mark.public_assets
@pytest.mark.parametrize("identity", SCENES, ids=lambda r: r["task"] + "/" + r["scene_id"])
def test_public_scene_runtime_construction(public_scenes, identity):
    root, manifest = public_scenes
    task, scene_id = identity["task"], identity["scene_id"]
    body = data_release.scene(task, scene_id, assets=root)
    assert scene_format.digest(body) == identity["new_sha256"]
    resolved = scene_format.resolve_physical(body, root, manifest)
    candidate = resolved["candidate"]
    spec = candidate["episode_spec"]
    assert tasks.validate_spec(spec, expected_task=task) == task
    plans = scene.plans_for_candidates([candidate], tasks.validate_spec)
    assert plans[0]["task"] == task
    assert plans[0]["seed"] == scene_format.scene_number(scene_id)
    assert len(plans[0]["blocks"]) == benchmark_task(task).block_count
    assert np.isfinite(resolved["initial_state"]["qpos"]).all()
    assert np.isfinite(resolved["initial_state"]["physical_qpos"]).all()
    assert len({block["root"] for block in plans[0]["blocks"]}) == len(spec["blocks"])
