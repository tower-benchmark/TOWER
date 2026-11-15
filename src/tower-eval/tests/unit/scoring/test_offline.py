# SPDX-License-Identifier: Apache-2.0
import json
from zipfile import ZipFile

import numpy as np
import pytest
from tower_eval.scoring.offline.annotations import (
    aggregate,
    read_annotations,
    resolve_run,
    validate,
)
from tower_eval.scoring.offline.trajectory import (
    Recording,
    Rules,
    angular_error,
    boxes_touch,
    corners,
    disturbance_events,
    fully_clear,
    hold_mask,
    load_recording,
    placement_geometry,
    score_recording,
    score_run,
    stable_dwell,
    task_counts,
)


def recording(seconds=12.0, blocks=3):
    times = np.arange(0, seconds + 0.001, 0.05)
    n = len(times)
    poses = np.zeros((n, blocks, 7))
    poses[..., 6] = 1
    spec = dict(
        blocks=[
            dict(
                name=f"b{i}",
                tower="source",
                layer=1,
                fixed_on_table=i == 0,
                dimensions_m=[0.15, 0.05, 0.03],
            )
            for i in range(blocks)
        ]
    )
    inv = np.zeros((n, 2, 4))
    inv[..., 3] = 1
    return Recording(
        spec,
        times,
        poses,
        np.zeros((n, blocks, 6)),
        np.zeros((n, 2, 3)),
        inv,
        np.ones((n, 2, 2)),
        np.full((n, 2), 0.05),
        dict(valid_trial=True, status="duration_reached", simulated_seconds=seconds),
        {},
    )


def test_counts_and_skip_before_io(tmp_path):
    assert task_counts("task4-3")[2:] == (6, 3)
    assert task_counts("task6-3")[2:] == (2, 2)
    with pytest.raises(ValueError, match="Unsupported"):
        score_run(tmp_path / "missing", "task8-2")


def test_grasp_rejects_fall_and_unilateral_contact():
    r = recording(1.0)
    r.poses[:, 1, 2] = r.times * 0.04
    assert not hold_mask(r, 1, 0, Rules()).any()  # Stationary hand, moving block.
    r.pinch[:, 0] = r.poses[:, 1, :3]
    assert hold_mask(r, 1, 0, Rules()).any()
    r.effort[:, 0, 1] = 0
    assert not hold_mask(r, 1, 0, Rules()).any()


def test_extraction_requires_whole_block_to_clear_even_if_initially_protruding():
    poses = np.array(
        [[0.075, 0, 0.045, 0, 0, 0, 1], [0.14, 0, 0.045, 0, 0, 0, 1], [0.151, 0, 0.045, 0, 0, 0, 1]]
    )
    foundation = [corners(np.array([0, 0, 0.015, 0, 0, 0, 1]), [0.15, 0.15, 0.03])]
    assert fully_clear(poses, [0.15, 0.05, 0.03], foundation).tolist() == [False, False, True]


def test_release_dwell_and_late_collapse():
    r = recording()
    released = r.times >= 1
    result = stable_dwell(released, r.poses[:, 1], r.velocities[:, 1], r.times, Rules())
    assert not result[r.times < 11].any()
    assert result[-1]
    released[-1] = False
    assert not stable_dwell(released, r.poses[:, 1], r.velocities[:, 1], r.times, Rules())[-1]
    # Continuous drift with low instantaneous velocity cannot earn a stable dwell.
    r.poses[:, 1, 0] = r.times * 0.001
    assert not stable_dwell(
        np.ones(len(r.times), bool), r.poses[:, 1], r.velocities[:, 1], r.times, Rules()
    ).any()


def test_disturbance_per_operation_excludes_foundation_and_other_active_target():
    r = recording(2.0, blocks=4)
    r.poses[:, :, 0] = r.times[:, None] * 0.02
    windows = [
        dict(id="first", target=1, arm=0, start=0, end=20),
        dict(id="second", target=1, arm=0, start=20, end=40),
        dict(id="other-arm", target=2, arm=1, start=0, end=40),
    ]
    events = disturbance_events(r, windows, Rules())
    assert {(e["operation"], e["block"]) for e in events} == {
        ("first", "b3"),
        ("second", "b3"),
        ("other-arm", "b3"),
    }
    # Exactly 10 mm does not exceed the paper threshold.
    r.poses[:, 3, 0] = np.linspace(0, 0.01, len(r.times))
    assert disturbance_events(r, windows, Rules()) == []


def test_released_target_does_not_inherit_its_carried_displacement():
    r = recording(2.0)
    r.poses[:21, 2, 0] = np.linspace(0, 0.1, 21)
    r.poses[21:, 2, 0] = 0.1
    windows = [
        dict(id="first", target=1, arm=0, start=0, end=40),
        dict(id="carry", target=2, arm=1, start=0, end=40, active_end=20),
    ]
    assert disturbance_events(r, windows, Rules()) == []
    r.poses[31:, 2, 0] = 0.12
    assert len(disturbance_events(r, windows, Rules())) == 1


def test_rectangular_symmetry_and_wrong_orientation():
    assert angular_error([0, 0, 1, 0], [0, 0, 0, 1]) == 0
    assert angular_error([0, 0, 2**-0.5, 2**-0.5], [0, 0, 0, 1]) == pytest.approx(np.pi / 2)


def write_workbook(path, rows, coordinates=False):
    from xml.sax.saxutils import escape

    columns = [
        "Task",
        "No.",
        "Episode",
        "任务成功",
        "抓取成功",
        "放置成功",
        "长程进度",
        "操作扰动",
        "状态",
        "更新时间",
        "源文件",
    ]
    content = []
    for k, row in enumerate([columns, *rows], 1):
        cells = []
        for j, value in enumerate(row):
            ref = f' r="{chr(65 + j)}{k}"' if coordinates else ""
            cells.append(f'<c{ref} t="inlineStr"><is><t>{escape(str(value))}</t></is></c>')
        content.append("<row>" + "".join(cells) + "</row>")
    with ZipFile(path, "w") as z:
        z.writestr(
            "xl/workbook.xml",
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"><sheets><sheet name="rater_sim_policy" r:id="id1"/></sheets></workbook>',
        )
        z.writestr(
            "xl/_rels/workbook.xml.rels",
            '<Relationships><Relationship Id="id1" Target="worksheets/sheet1.xml"/></Relationships>',
        )
        z.writestr(
            "xl/worksheets/sheet1.xml",
            '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>'
            + "".join(content)
            + "</sheetData></worksheet>",
        )


@pytest.mark.parametrize("coordinates", [False, True])
def test_workbook_preserves_zero_and_skips_task7(tmp_path, coordinates):
    path = tmp_path / "annotations.xlsx"
    write_workbook(
        path,
        [
            [
                "task1-2",
                1,
                "ep1",
                "0.0",
                "1.0",
                "0.0",
                "2.0",
                "3.0",
                "",
                "",
                "/runs/x/run/episode/states.hdf5",
            ],
            ["task7-2", 1, "ep2", "", "", "", "", "", "", "", "/missing"],
            ["task2-2", 1, "ep3", "", "", "", "", "", "", "", "/runs/z/run/episode/states.hdf5"],
        ],
        coordinates,
    )
    episodes, ratings, skipped = read_annotations(path)
    assert len(episodes) == 2 and len(ratings) == 1
    assert ratings[0]["s"] == 0 and ratings[0]["C"] == 1 and ratings[0]["raw_progress"] == 2
    assert skipped == {"task7_rows": 1, "unrated_rows": 1}


def test_validation_unanimous_fields_and_pooled_denominators():
    episodes = [dict(source="x", task="task2-2", policy="p")]
    base = dict(source="x", s=0, g=1, p=0, C=1, D=3, raw_progress=2)
    ratings = [dict(base, annotator="a"), dict(base, annotator="b", D=4)]
    report = dict(valid=True, planned_g=2, planned_p=2, scores=dict(s=0, g=1, p=0, C=1, E=2, D=4))
    result = validate(episodes, ratings, {"x": report})
    assert result["agreed"]["g"]["exact_match"] == 1
    assert result["agreed"]["D"]["n"] == 0
    assert len(result["progress_column_conflicts"]) == 2
    assert aggregate([report])["ODR"] == 2
    assert aggregate([])["ODR"] is None


def test_remap_and_traversal(tmp_path):
    assert resolve_run("/old/runs/abc/run/episode/states.hdf5", tmp_path) == tmp_path / "abc/run"
    with pytest.raises(ValueError, match="escapes"):
        resolve_run("/old/runs/../../run/episode/states.hdf5", tmp_path)


def test_loader_rejects_missing_final_state(tmp_path):
    r = recording(1.0)
    import h5py

    (tmp_path / "episode").mkdir()
    (tmp_path / "episode_spec.json").write_text(json.dumps(r.spec))
    (tmp_path / "status.json").write_text(json.dumps({**r.status, "simulated_seconds": 2}))
    (tmp_path / "finger_geometry.json").write_text("[]")
    np.savez(tmp_path / "finger-link-poses.npz", time=r.times)
    np.savez(tmp_path / "finger-feedback.npz", time=r.times)
    with h5py.File(tmp_path / "episode/states.hdf5", "w") as f:
        for k, v in dict(
            time=r.times,
            blocks=r.poses,
            block_velocities=r.velocities,
            qpos=np.zeros((len(r.times), 16)),
        ).items():
            f[k] = v
    with pytest.raises(ValueError, match="final evaluator"):
        load_recording(tmp_path, Rules())


def transfer_recording():
    r = recording(30.0, blocks=4)
    r.spec["layout_metadata"] = {"catalog_task": 2}
    r.spec["blocks"][0].update(tower="source", layer=0)
    r.spec["blocks"][1].update(tower="destination", layer=0, fixed_on_table=True)
    r.poses[:, 0, :3] = [0, 0, 0.015]
    r.poses[:, 1, :3] = [1, 0, 0.015]
    r.spec["transfers"] = [
        dict(
            block_index=i,
            block_id=f"b{i}",
            arm=i - 2,
            source_layer=3 - i,
            destination_layer=i - 1,
            goal=[1, 0, 0.045 + 0.03 * (i - 2), 0, 0, 0, 1],
        )
        for i in (2, 3)
    ]
    r.pinch[:] = 5
    r.opening[:] = 0.1
    for i, begin in ((2, 0), (3, 14)):
        arm = i - 2
        r.poses[:, i, :3] = [0, 0, 0.045 + 0.03 * (3 - i)]
        for k, time in enumerate(r.times):
            dt = time - begin
            if 0 <= dt <= 2:
                r.poses[k, i, :3] += [max(0, dt - 1), 0, min(dt, 1) * 0.04]
                r.pinch[k, arm] = r.poses[k, i, :3]
                r.opening[k, arm] = 0.05
            elif dt > 2:
                r.poses[k, i] = r.spec["transfers"][arm]["goal"]
    return r


def test_complete_trajectory_and_terminal_failure_preserves_operations():
    r = transfer_recording()
    result = score_recording(r, "task2-2")
    assert result["scores"] == dict(s=1, g=2, p=2, C=4, E=4, D=0)
    r.poses[-1, 2, 0] = 2
    result = score_recording(r, "task2-2")
    assert result["scores"]["s"] == 0
    assert result["scores"]["C"] == 4 and result["scores"]["E"] == 5
    # Source evaluator's positive flag cannot override trajectory geometry.
    r.status["status"] = "task_success"
    assert score_recording(r, "task2-2")["scores"]["s"] == 0


def test_partial_credit_is_not_a_completed_prefix():
    r = transfer_recording()
    r.effort[:, 0] = 0
    result = score_recording(r, "task2-2")
    assert result["scores"]["g"] == 1
    assert result["scores"]["p"] == 2
    assert result["scores"]["C"] == 3
    assert result["scores"]["s"] == 0


def test_policy_safety_stop_is_valid_failure():
    r = transfer_recording()
    r.status["status"] = "safety_abort"
    result = score_recording(r, "task2-2")
    assert result["valid"] and result["scores"]["s"] == 0
    r.status["valid_trial"] = False
    assert score_recording(r, "task2-2")["scores"] is None


def test_insertion_requires_actual_slot_and_intact_roof():
    r = recording(1.0, blocks=5)
    r.spec["blocks"][0].update(tower="destination", layer=0)
    for j, y in ((1, -0.051), (2, 0.051)):
        r.spec["blocks"][j].update(name=f"destination_l1_side_{j}", tower="destination", layer=1)
        r.poses[:, j, :3] = [0, y, 0.045]
    r.spec["blocks"][3].update(name="roof", tower="destination", layer=2)
    r.poses[:, 3, :3] = [0, 0, 0.077]
    r.poses[:, 4, :3] = [0, 0, 0.045]
    t = dict(block_index=4, goal=[0, 0, 0.045, 0, 0, 0, 1], destination_layer=1)
    assert placement_geometry(r, [t], 5, Rules())[4].all()
    # Goal proximity alone is insufficient: target overlaps a side block.
    r.poses[-1, 4, 1] = 0.005
    assert not placement_geometry(r, [t], 5, Rules())[4][-1]
    r.poses[-1, 4, 1] = 0
    r.poses[-1, 3, 2] -= 0.03
    assert not placement_geometry(r, [t], 5, Rules())[4][-1]


def test_end_grasp_counts_without_requiring_pinch_at_block_center():
    r = recording(1.0)
    r.poses[:, 1, 0] = r.times * 0.04
    r.pinch[:, 0] = r.poses[:, 1, :3] + [0.06, 0, 0]
    assert hold_mask(r, 1, 0, Rules()).any()


def test_held_at_goal_never_counts_as_released_placement():
    r = transfer_recording()
    r.opening[:, 0] = 0.05
    r.pinch[:, 0] = r.poses[:, 2, :3]
    result = score_recording(r, "task2-2")
    assert result["targets"][0]["grasp_sample"] is not None
    assert result["targets"][0]["place_sample"] is None
    assert result["scores"]["s"] == 0


def pushing_recording():
    r = recording(20.0, blocks=7)
    r.spec["blocks"][0]["dimensions_m"] = [0.15, 0.15, 0.03]
    targets = [
        dict(block_index=i, block_id=f"b{i}", source_layer=2 if i < 4 else 1, arm=0 if i < 4 else 1)
        for i in range(1, 7)
    ]
    r.spec.update(transfers=[], layout_metadata=dict(catalog_task=4, push_targets=targets))
    r.pinch[:] = [-1, 0, 1]
    for i in range(1, 7):
        r.poses[:, i, :3] = [0, 0.05 * ((i - 1) % 3 - 1), 0.075 if i < 4 else 0.045]
        begin = (i - 1) * 3 + 1
        displacement = np.clip((r.times - begin) * 0.2, 0, 0.2)
        r.poses[:, i, 0] = displacement
        # Include the observed approach before translating the block.
        active = (r.times >= begin - 0.2) & (r.times <= begin + 1)
        r.pinch[active, 0 if i < 4 else 1] = r.poses[active, i, :3] - [0.075, 0, 0]
    return r


def test_push_phase_counts_order_and_passive_collapse():
    r = pushing_recording()
    result = score_recording(r, "task4-2")
    assert (result["scores"]["s"], result["scores"]["g"], result["scores"]["p"]) == (1, 3, 3)
    # Identical block departure without a hand near the block is a collapse.
    r.pinch[:] = [-1, 0, 1]
    result = score_recording(r, "task4-2")
    assert result["scores"]["C"] == 0 and result["scores"]["s"] == 0


def test_unknown_error_is_invalid_even_with_a_stale_positive_flag():
    r = transfer_recording()
    r.status.update(status="error", valid_trial=True)
    assert not score_recording(r, "task2-2")["valid"]


def test_task1_result_is_json_serializable():
    r = recording(1.0, blocks=6)
    r.spec["blocks"][1].update(tower="destination", fixed_on_table=True, layer=0)
    r.spec["layout_metadata"] = {"catalog_task": 1}
    r.spec["transfers"] = [
        dict(
            block_index=i,
            block_id=f"b{i}",
            arm=(i - 2) // 2,
            source_layer=2 - (i - 2) // 2,
            destination_layer=1 + (i - 2) // 2,
            goal=[1, 0.05 if i % 2 else -0.05, 0.045 + 0.03 * ((i - 2) // 2), 0, 0, 0, 1],
        )
        for i in range(2, 6)
    ]
    result = score_recording(r, "task1-2")
    assert json.loads(json.dumps(result, allow_nan=False))["scores"]["s"] == 0


def test_shared_strings_and_sparse_coordinates(tmp_path):
    from tower_eval.scoring.offline.annotations import worksheet_rows

    path = tmp_path / "sparse.xlsx"
    with ZipFile(path, "w") as archive:
        archive.writestr(
            "sheet.xml",
            '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData><row><c r="A1" t="s"><v>0</v></c><c r="C1"><v>0</v></c></row></sheetData></worksheet>',
        )
    with ZipFile(path) as archive:
        assert list(worksheet_rows(archive, "sheet.xml", ["任务成功"])) == [["任务成功", "", "0"]]


def test_supported_offset_placement_is_valid_but_floating_is_not():
    r = transfer_recording()
    # The destination foundation is 150 mm wide; a supported 15 mm offset
    # is not a failure merely because it differs from an authored center.
    placed = r.times > 2
    r.poses[placed, 2, 0] += 0.015
    masks = placement_geometry(r, r.spec["transfers"], 2, Rules())
    assert masks[2][-1]
    r.poses[-1, 1, 2] -= 0.03
    masks = placement_geometry(r, r.spec["transfers"], 2, Rules())
    assert not masks[2][-1]


def test_incomplete_ten_second_observation_does_not_receive_placement_credit():
    r = transfer_recording()
    r.times = r.times[:81]
    for field in ("poses", "velocities", "pinch", "inverse_flange", "effort", "opening"):
        setattr(r, field, getattr(r, field)[:81])
    report = score_recording(r, "task2-2")
    assert report["scores"]["g"] == 1 and report["scores"]["p"] == 0


def test_arm_annotation_mapping_does_not_change_paper_counts():
    episodes = [dict(source="x", task="task4-2", policy="p")]
    rating = dict(source="x", s=0, g=0, p=2, C=2, D=0, raw_progress=2)
    ratings = [dict(rating, annotator="a"), dict(rating, annotator="b")]
    report = dict(
        task="task4-2",
        valid=True,
        planned_g=3,
        planned_p=3,
        scores=dict(s=0, g=2, p=0, C=2, E=3, D=0),
        push_successes_by_arm={"0": 0, "1": 2},
    )
    assert validate(episodes, ratings, {"x": report}, "arm")["agreed"]["g"]["exact_match"] == 1
    assert validate(episodes, ratings, {"x": report})["agreed"]["g"]["exact_match"] == 0
    assert report["scores"]["g"] == 2 and report["scores"]["p"] == 0


def test_partially_pushed_block_does_not_receive_completion_credit():
    r = pushing_recording()
    r.poses[:, 1, 0] = np.minimum(r.poses[:, 1, 0], 0.12)
    report = score_recording(r, "task4-2")
    assert report["targets"][0]["push_sample"] is None
    assert report["scores"]["C"] == 5  # Later independent pushes still count.


def test_extraction_above_tower_and_partial_lift():
    r = recording(5.0)
    r.spec["layout_metadata"] = {"catalog_task": 6}
    r.spec["blocks"][0].update(layer=0, dimensions_m=[0.15, 0.15, 0.03])
    r.spec["blocks"][2].update(layer=2, dimensions_m=[0.15, 0.15, 0.03])
    r.poses[:, 0, :3] = [0, 0, 0.015]
    r.poses[:, 2, :3] = [0, 0, 0.075]
    r.poses[:, 1, :3] = [-0.075, 0, 0.045]
    r.poses[:, 1, 2] += r.times * 0.04
    r.pinch[:, 0] = r.poses[:, 1, :3]
    r.pinch[:, 1] = 1
    r.spec["transfers"] = [
        dict(
            block_index=1,
            block_id="b1",
            arm=0,
            source_layer=1,
            destination_layer=3,
            goal=[0, 0, 0.105, 0, 0, 0, 1],
        )
    ]
    assert score_recording(r, "task6-2")["scores"]["g"] == 1
    r.poses[:, 1, 2] = np.minimum(r.poses[:, 1, 2], 0.1)
    r.pinch[:, 0] = r.poses[:, 1, :3]
    assert score_recording(r, "task6-2")["scores"]["g"] == 0


def test_shifted_supported_roof_is_disturbance_not_structural_failure():
    r = recording(1.0)
    r.spec["blocks"][0].update(layer=0, dimensions_m=[0.15, 0.15, 0.03])
    r.spec["blocks"][2].update(layer=2, dimensions_m=[0.15, 0.15, 0.03])
    r.poses[:, 0, 2] = 0.015
    r.poses[:, 1, 2] = 0.105
    r.poses[:, 2, 2] = 0.075
    r.poses[-1, 2, 0] = 0.012
    t = dict(block_index=1, destination_layer=3, goal=[0, 0, 0.105, 0, 0, 0, 1])
    assert placement_geometry(r, [t], 6, Rules())[1][-1]
    r.poses[-1, 2, 2] -= 0.03
    assert not placement_geometry(r, [t], 6, Rules())[1][-1]


def test_push_inertia_is_not_a_new_disturbance_in_the_next_operation():
    r = recording(2.0)
    r.poses[:, 1, 0] = r.times * 0.04
    windows = [
        dict(id="push1", target=1, arm=0, start=0, end=15),
        dict(id="push2", target=2, arm=0, start=16, end=40),
    ]
    assert disturbance_events(r, windows, Rules()) == []
    r.poses[11:31, 1, 0] = r.poses[10, 1, 0]
    r.poses[31:, 1, 0] = r.poses[10, 1, 0] + 0.02
    events = disturbance_events(r, windows, Rules())
    assert len(events) == 1 and events[0]["operation"] == "push2"


def test_push_withdrawal_disturbance_counts_after_completion():
    r = pushing_recording()
    r.poses[r.times >= 2.8, 6, 1] += 0.02
    report = score_recording(r, "task4-2")
    event = next(e for e in report["disturbances"] if e["block"] == "b6")
    assert event["operation"] == "b1:push"
    assert event["sample"] > report["targets"][0]["push_sample"]


def test_out_of_order_pushes_retain_independent_credit():
    r = pushing_recording()
    targets = r.spec["layout_metadata"]["push_targets"]
    targets[0], targets[1] = targets[1], targets[0]
    report = score_recording(r, "task4-2")
    assert report["scores"]["C"] == 6
    assert report["scores"]["s"] == 0 and not report["prescribed_order"]


def test_other_arm_target_remains_excluded_after_operation_boundary():
    r = recording(3.0)
    r.poses[21:26, 1, 0] = np.linspace(0, 0.02, 5)
    r.poses[26:, 1, 0] = 0.02
    windows = [
        dict(id="first", target=1, arm=0, start=0, end=5, active_end=30),
        dict(id="second", target=2, arm=1, start=6, end=60),
    ]
    assert disturbance_events(r, windows, Rules()) == []
    r.poses[45:, 1, 0] += 0.02
    events = disturbance_events(r, windows, Rules())
    assert len(events) == 1 and events[0]["operation"] == "second"


def test_arm_annotation_bounds_use_recorded_assignment():
    episodes = [dict(source="x", task="task4-3", policy="p")]
    rating = dict(source="x", s=1, g=3, p=6, C=9, D=0, raw_progress=9)
    ratings = [dict(rating, annotator="a"), dict(rating, annotator="b")]
    report = dict(
        task="task4-3",
        valid=True,
        planned_g=6,
        planned_p=3,
        scores=dict(s=1, g=6, p=3, C=9, E=9, D=0),
        planned_pushes_by_arm={"0": 3, "1": 6},
        push_successes_by_arm={"0": 3, "1": 6},
    )
    assert validate(episodes, ratings, {"x": report}, "arm")["annotation_count_conflicts"] == []


def test_abandoned_push_target_can_be_disturbed_by_other_arm():
    r = pushing_recording()
    r.poses[:, 3, 0] = np.minimum(r.poses[:, 3, 0], 0.12)
    r.poses[r.times >= 13.2, 3, 0] += 0.02
    report = score_recording(r, "task4-2")
    assert report["targets"][2]["push_sample"] is None
    assert any(e["block"] == "b3" and e["operation"] == "b5:push" for e in report["disturbances"])


def test_repeated_motion_counts_once_per_block_per_subtask():
    r = recording(4.0, blocks=4)
    r.poses[10:20, 3, 0] = 0.02
    r.poses[30:40, 3, 0] = 0.03
    r.poses[50:60, 3, 0] = 0.02
    r.poses[70:, 3, 0] = 0.03
    windows = [
        dict(id="grasp", target=1, arm=0, start=0, end=40),
        dict(id="place", target=1, arm=0, start=41, end=80),
    ]
    events = disturbance_events(r, windows, Rules())
    assert [(e["operation"], e["block"]) for e in events] == [("grasp", "b3"), ("place", "b3")]


def test_failed_placement_release_allows_later_disturbance():
    r = transfer_recording()
    r.poses[r.times > 2, 2, 1] = 0.3  # Release outside the destination.
    r.poses[r.times >= 20, 2, 0] += 0.02
    report = score_recording(r, "task2-2")
    assert report["targets"][0]["place_sample"] is None
    assert [(e["operation"], e["block"]) for e in report["disturbances"]] == [("b3:place", "b2")]


def test_already_moving_block_can_become_the_current_target():
    r = recording(3.0)
    r.poses[:, 1, 0] = r.times * 0.04
    windows = [
        dict(id="before", target=2, arm=0, start=0, end=10),
        dict(id="push", target=1, arm=0, start=11, end=30),
        dict(id="after", target=2, arm=0, start=31, end=60),
    ]
    events = disturbance_events(r, windows, Rules())
    assert [(e["operation"], e["block"]) for e in events] == [("before", "b1")]


def test_passive_arrival_with_open_stationary_hands_is_not_placement():
    r = transfer_recording()
    r.pinch[:] = 5
    r.effort[:] = 0
    r.opening[:] = 0.1
    for i, begin in ((2, 0), (3, 14)):
        near = (r.times >= begin) & (r.times <= begin + 0.5)
        r.pinch[near, i - 2] = r.poses[0, i, :3]
    report = score_recording(r, "task2-2")
    assert report["scores"]["g"] == report["scores"]["p"] == 0
    assert all(t["carry_sample"] is None for t in report["targets"])


def test_stationary_nearby_hands_cannot_earn_push_credit():
    r = pushing_recording()
    for i in range(1, 7):
        begin, arm = (i - 1) * 3 + 1, 0 if i < 4 else 1
        r.poses[:, i, 0] = np.clip((r.times - begin) * 0.5, 0, 0.2)
        near = (r.times >= begin - 0.2) & (r.times <= begin + 1)
        r.pinch[near, arm] = r.poses[0, i, :3] - [0.075, 0, 0]
    report = score_recording(r, "task4-2")
    assert report["scores"]["C"] == 0


def test_slow_directed_push_retains_credit():
    r = pushing_recording()
    r.poses[:, 1, 0] = np.clip((r.times - 1) * 0.02, 0, 0.2)
    near = (r.times >= 0.8) & (r.times <= 11)
    r.pinch[near, 0] = r.poses[near, 1, :3] - [0.075, 0, 0]
    assert score_recording(r, "task4-2")["targets"][0]["push_sample"] is not None


@pytest.mark.parametrize("finger_geometry", [False, True])
def test_new_hand_contact_interrupts_a_former_targets_inertia(finger_geometry):
    r = recording(3.0)
    r.poses[:, 1, 0] = r.times * 0.02
    r.poses[:, 1, 1] = np.maximum(r.times - 1.5, 0) * 0.08
    r.pinch[:] = 5
    collision = r.times >= 1.5
    r.pinch[collision, 0] = r.poses[collision, 1, :3]
    if finger_geometry:
        r.pinch[collision, 0, 1] += 0.05
        r.finger_corners = np.broadcast_to(
            r.pinch[:, :, None, None], (len(r.times), 2, 2, 8, 3)
        ).copy()
        r.finger_corners[collision, 0, :, :, 1] -= 0.025
    windows = [
        dict(id="first", target=1, arm=0, start=0, end=20),
        dict(id="next", target=2, arm=0, start=21, end=60),
    ]
    events = disturbance_events(r, windows, Rules())
    assert [(e["operation"], e["block"]) for e in events] == [("next", "b1")]
    assert events[0]["baseline_sample"] == 29


def test_contact_by_the_new_target_also_interrupts_inertia():
    r = recording(3.0)
    r.pinch[:] = 5  # No direct finger contact with the former target.
    r.poses[:, 1, 0] = r.times * 0.02
    r.poses[:, 2, 0] = np.maximum(r.times, 1.05) * 0.02
    r.poses[:, 2, 1] = np.maximum(0.15 - 0.2 * np.maximum(r.times - 1.4, 0), -0.15)
    r.poses[:, 1, 1] = np.minimum(0, r.poses[:, 2, 1] - 0.05)
    windows = [
        dict(id="first", target=1, arm=0, start=0, end=20),
        dict(id="next", target=2, arm=0, start=21, end=60),
    ]
    events = disturbance_events(r, windows, Rules())
    assert [(e["operation"], e["block"]) for e in events] == [("next", "b1")]


def test_box_contact_handles_crossing_faces_and_separation():
    a = np.array([[0, 0, 0, 0, 0, 0, 1]] * 3, dtype=float)
    b = np.array(
        [
            [0, 0, 0.03, 0, 0, 2**-0.5, 2**-0.5],
            [0, 0, 0.035, 0, 0, 2**-0.5, 2**-0.5],
            [0.11, 0.11, 0, 0, 0, 2**-0.5, 2**-0.5],
        ]
    )
    assert boxes_touch(a, [0.15, 0.05, 0.03], b, [0.15, 0.05, 0.03], 0.002).tolist() == [
        True,
        False,
        False,
    ]


def test_overlapping_grasps_preserve_both_current_targets():
    r = transfer_recording()
    r.pinch[:, 1] = 5
    r.opening[:, 1] = 0.1
    r.poses[:, 3, :3] = [0, 0, 0.045]
    for k, time in enumerate(r.times):
        dt = time - 0.05
        if 0 <= dt <= 2:
            r.poses[k, 3, :3] += [max(0, dt - 1), 0, min(dt, 1) * 0.04]
            r.pinch[k, 1] = r.poses[k, 3, :3]
            r.opening[k, 1] = 0.05
        elif dt > 2:
            r.poses[k, 3] = r.spec["transfers"][1]["goal"]
    held = r.times <= 2
    r.poses[held, 2, 2] += 0.02 * np.minimum(r.times[held], 1)
    r.pinch[held, 0] = r.poses[held, 2, :3]
    report = score_recording(r, "task2-2")
    assert report["scores"]["g"] == report["scores"]["p"] == 2
    assert report["scores"]["D"] == 0
    assert not report["prescribed_order"]  # Order affects success, not ownership.


def test_single_frame_impact_uses_the_pre_contact_pose():
    r = recording(1.8)
    r.poses[:, 1, 0] = r.times * 0.02
    r.pinch[:] = 5
    collision = r.times >= 1.5
    r.poses[collision, 1, 1] = 0.02
    r.pinch[collision, 0] = r.poses[collision, 1, :3]
    windows = [
        dict(id="first", target=1, arm=0, start=0, end=20),
        dict(id="next", target=2, arm=0, start=21, end=36),
    ]
    events = disturbance_events(r, windows, Rules())
    assert len(events) == 1
    assert events[0]["sample"] == 30 and events[0]["baseline_sample"] == 29
    assert events[0]["translation_m"] > 0.02


def test_operation_boundary_alone_does_not_renew_continuous_contact():
    r = recording(3.0)
    r.pinch[:] = 5
    r.poses[:, 1, 0] = r.times * 0.02
    r.poses[:, 2, 0] = r.times * 0.02
    r.poses[:, 2, 1] = 0.05  # The pair stays in contact throughout its movement.
    windows = [
        dict(id="first", target=1, arm=0, start=0, end=20),
        dict(id="next", target=2, arm=0, start=21, end=60),
    ]
    events = disturbance_events(r, windows, Rules())
    assert [(e["operation"], e["block"]) for e in events] == [("first", "b2")]
