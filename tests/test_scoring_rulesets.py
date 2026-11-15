"""Ruleset isolation, input contracts, and exported scoring entry points."""

import importlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest
from tower_eval.scoring import api, cli
from tower_eval.scoring.detectors import DETECTORS, task6_completion, tool
from tower_eval.scoring.recording import Recording, from_campaign_rows
from tower_eval.scoring.rulesets.bench_2026_11 import Rules as BenchRules
from tower_eval.scoring.rulesets.paper_v1 import Rules as PaperRules
from tower_eval.tasks import load_tasks
from tower_sim.bench import data_release, scene_format

from scripts.maintainer.fetch_public_test_data import public_asset_root

ROOT = Path(__file__).resolve().parents[1]


def initial_recording(task, seconds=0.25):
    root = public_asset_root()
    body = data_release.scene(task, "scene-000", assets=root)
    scene = scene_format.decode_scene(body)["candidate"]["episode_spec"]
    times = np.arange(0, seconds + 0.001, 0.05)
    poses = np.repeat(np.asarray([b["pose"] for b in scene["blocks"]])[None], len(times), axis=0)
    inverse_flange = np.zeros((len(times), 2, 4))
    inverse_flange[..., 3] = 1
    return Recording(
        scene,
        times,
        poses,
        np.zeros((*poses.shape[:2], 6)),
        np.full((len(times), 2, 3), 10.0),
        inverse_flange,
        np.zeros((len(times), 2, 2)),
        np.full((len(times), 2), 0.1),
        dict(valid_trial=True, status="duration_reached", simulated_seconds=float(times[-1])),
        {},
    )


def test_registry_covers_every_benchmark_and_excludes_experimental():
    assert set(DETECTORS) == {t.id for t in load_tasks() if t.status == "benchmark"}
    # Semantic switches must not alter the original threshold report shape.
    assert asdict(BenchRules()) == asdict(PaperRules())


@pytest.mark.public_assets
def test_task4_final_success_is_ruleset_specific():
    r = initial_recording("task4-2")
    for slot, target in enumerate(r.spec["layout_metadata"]["push_targets"]):
        r.poses[-1, target["block_index"], :3] = [0.7, -0.05 - slot * 0.07, 0.015]
    bench = api.score_recording(r, "E2")
    paper = api.score_recording(r, "E2", ruleset="paper-v1")
    assert bench["scores"]["s"] == 1 and paper["scores"]["s"] == 0
    assert bench["original_scores"] == paper["scores"]
    assert bench["scores"]["E"] == 0 and bench["odr"]["value"] is None


@pytest.mark.public_assets
def test_task6_recovered_structural_fall_is_preserved():
    r = initial_recording("task6-2")
    info = task6_completion.scene(r, "task6-2")
    index = info["structural"][0]
    r.poses[2, index, 2] -= info["dims"][index][2] * 0.6
    assert np.array_equal(r.poses[-1, index], r.poses[0, index])
    result = api.score_recording(r, "C2")
    assert any(e["block_index"] == index for e in result["task6_completion"]["falls"])
    assert not result["task6_completion"]["passed"]


@pytest.mark.public_assets
def test_task7_collapse_veto_is_ruleset_specific():
    r = initial_recording("task7-3")
    index = next(
        i
        for i, b in enumerate(r.spec["blocks"])
        if b["tower"] == "source"
        and not b["fixed_on_table"]
        and i not in {t["block_index"] for t in r.spec["transfers"]}
    )
    r.poses[2, index, 2] -= 0.03
    masks = []
    for rules in (PaperRules(), BenchRules()):
        final = []
        tool.score_tools(r, 3, rules, dict(events=[], operations=[]), [], final)
        masks.append(final[0])
    assert masks[0][-1] and not masks[1][-1]
    assert masks[1][:2].all() and not masks[1][2:].any()


@pytest.mark.public_assets
def test_task7_metadata_direction_does_not_change_paper():
    r = initial_recording("task7-3")
    r.spec["layout_metadata"]["cycles"][0]["direction"] = [0.0, 1.0, 0.0]
    np.testing.assert_array_equal(
        tool.tool_plan(r.spec, 3, BenchRules())[0]["direction"], [0, 1, 0]
    )
    np.testing.assert_array_equal(
        tool.tool_plan(r.spec, 3, PaperRules())[0]["direction"], [-1, 0, 0]
    )


def test_campaign_constructor_keeps_historical_inputs():
    fingers = np.zeros((2, 3, 7))
    fingers[..., 6] = 1
    rows = [
        dict(
            time=t,
            finger_poses=fingers,
            blocks=[[0, 0, 0, 0, 0, 0, 1]],
            block_velocities=np.zeros((1, 6)),
            finger_forces=-np.ones((2, 2)),
            qpos=np.arange(16),
        )
        for t in (0, 0.05)
    ]
    record = from_campaign_rows({}, rows, [dict(pinch_center_in_flange_m=[1, 2, 3])] * 2)
    assert record.hashes == {} and record.finger_corners is None
    assert record.status == dict(
        status="duration_reached", valid_trial=True, returncode=0, simulated_seconds=0.05
    )
    np.testing.assert_array_equal(record.opening, [[7, 15], [7, 15]])
    np.testing.assert_array_equal(record.pinch, [[[1, 2, 3]] * 2] * 2)
    np.testing.assert_array_equal(record.effort, np.ones((2, 2, 2)))
    rows[0]["time"] = 0.01
    with pytest.raises(ValueError, match="from time zero"):
        from_campaign_rows({}, rows, [])


@pytest.mark.parametrize("input_path", ["files", "campaign"])
@pytest.mark.parametrize("ruleset", ["paper-v1", "bench-2026.11"])
def test_tower_score_exports_selection(monkeypatch, tmp_path, input_path, ruleset):
    from tower_eval.cli import main

    calls = []

    def score(run, task, **kwargs):
        calls.append((run, task, kwargs))
        return dict(valid=True, **api.provenance(task, kwargs["ruleset"], kwargs["input_path"]))

    monkeypatch.setattr(cli, "score_run", score)
    output = tmp_path / "score.json"
    assert (
        main(
            [
                "score",
                str(tmp_path / "run"),
                "--task",
                "D2",
                "--output",
                str(output),
                "--input-path",
                input_path,
                "--ruleset",
                ruleset,
            ]
        )
        == 0
    )
    assert calls == [(tmp_path / "run", "task5-2", dict(input_path=input_path, ruleset=ruleset))]
    assert json.loads(output.read_text())["canonical_task_id"] == "lateral-insertion-2"
    with pytest.raises(SystemExit):
        main(["score", str(tmp_path / "run"), "--task", "D2", "--output", str(output)])


def test_legacy_reexports_are_real_implementations():
    from tower_eval.scoring import StackScorer, early_success, online
    from tower_eval.scoring import geometry as _public_geometry
    from tower_eval.scoring.offline import annotations, trajectory

    geometry = importlib.import_module("tower_eval.scoring.geometry")
    from reproduction.paper_metrics import annotations as paper_annotations

    assert StackScorer is online.StackScorer
    assert trajectory.block_geometry is geometry.geometry is _public_geometry
    assert early_success.from_physical_rows is from_campaign_rows
    assert annotations.read_annotations is paper_annotations.read_annotations
    assert trajectory.Rules().ruleset == "paper-v1"
