# SPDX-License-Identifier: Apache-2.0
"""Task F synthetic evidence tests; no human annotation comparisons."""

import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from tower_eval.scoring.offline.tool_extraction import tool_plan
from tower_eval.scoring.offline.trajectory import Recording, Rules, score_recording, task_counts


def task7_recording(layers=2):
    times = np.arange(0, 48 * (layers - 1), 0.05)
    blocks = []

    def block(name, tower, layer, slot, xyz, dims=None, yaw=0):
        q = Rotation.from_euler("z", yaw, degrees=True).as_quat().tolist()
        blocks.append(
            dict(
                name=name,
                tower=tower,
                layer=layer,
                slot=slot,
                pose=[*xyz, *q],
                dimensions_m=dims,
                fixed_on_table=layer == 0 and tower != "tools",
            )
        )
        return len(blocks) - 1

    block("base", "source", 0, 0, [0, 0, 0.015], [0.15, 0.15, 0.03])
    block("destination", "destination", 0, 0, [1, 0, 0.015], [0.15, 0.15, 0.03])
    for layer in range(1, layers + 1):
        for slot in (-1, 0, 1):
            xyz = (
                [0, slot * 0.05, 0.015 + layer * 0.03]
                if layer % 2
                else [slot * 0.05, 0, 0.015 + layer * 0.03]
            )
            block(
                f"source_l{layer}_{slot + 1}",
                "source",
                layer,
                slot,
                xyz,
                yaw=0 if layer % 2 else -90,
            )
    block("pusher_upper", "tools", 0, 0, [0.6, -0.5, 0.01], [0.1, 0.19, 0.02], -90)
    block("pusher_lower", "tools", 0, 1, [0.5, 0.5, 0.01], [0.1, 0.19, 0.02])
    poses = np.broadcast_to(
        np.array([b["pose"] for b in blocks]), (len(times), len(blocks), 7)
    ).copy()
    pinch = np.full((len(times), 2, 3), 5.0)
    opening = np.full((len(times), 2), 0.1)
    inv = np.zeros((len(times), 2, 4))
    inv[..., 3] = 1
    spec = dict(blocks=blocks, transfers=[], layout_metadata=dict(catalog_task=7))
    r = Recording(
        spec,
        times,
        poses,
        np.zeros((len(times), len(blocks), 6)),
        pinch,
        inv,
        np.ones((len(times), 2, 2)),
        opening,
        dict(valid_trial=True, status="duration_reached", simulated_seconds=times[-1]),
        {},
    )
    names = [b["name"] for b in blocks]
    for cycle in range(layers - 1):
        layer = cycle + 1
        begin = cycle * 48
        tool = names.index("pusher_upper" if cycle == 0 else "pusher_lower")
        arm = 1 - cycle
        direction = np.array([-1.0, 0.0, 0.0] if cycle == 0 else [0.0, -1.0, 0.0])
        offset = Rotation.from_quat(poses[0, tool, 3:]).apply(Rules().tool_tip_local_m)
        targets = [names.index(f"source_l{layer}_{slot}") for slot in (2, 0)]
        home = poses[0, tool, :3].copy()
        xyz1 = poses[0, targets[0], :3] - direction * 0.075 - offset
        xyz2 = poses[0, targets[1], :3] - direction * 0.075 - offset
        key_times = np.array([0, 1, 3, 4, 6, 7, 9, 10]) + begin
        key_xyz = np.array(
            [
                home,
                home + [0, 0, 0.04],
                xyz1,
                xyz1 + direction * 0.06,
                xyz2,
                xyz2 + direction * 0.06,
                home + [0, 0, 0.04],
                home,
            ]
        )
        for axis in range(3):
            poses[:, tool, axis] = np.interp(times, key_times, key_xyz[:, axis])
        held = (times >= begin) & (times <= begin + 10)
        pinch[held, arm] = poses[held, tool, :3]
        opening[held, arm] = 0.05
        for index, push_begin in zip(targets, (3, 6)):
            poses[:, index, :3] += (
                np.clip(times - begin - push_begin, 0, 1)[:, None] * direction * 0.06
            )
        extract_ids = targets[::-1] if cycle == 0 else targets
        for order, index in enumerate(extract_ids):
            start = begin + 12 + 16 * order
            goal = np.array([1.0, 0.0, 0.015 + layer * 0.03, 0, 0, 0, 1])
            goal[1 if cycle == 0 else 0] += (-1 if order == 0 else 1) * 0.05
            goal[3:] = poses[0, index, 3:]
            spec["transfers"].append(
                dict(
                    block_index=index,
                    block_id=names[index],
                    arm=cycle,
                    source_layer=layer,
                    destination_layer=layer,
                    goal=goal.tolist(),
                )
            )
            source = poses[np.searchsorted(times, start), index, :3].copy()
            path_times = [start, start + 1, start + 2, start + 3]
            path_xyz = np.array(
                [source, source + direction * 0.15, goal[:3] + [0, 0, 0.08], goal[:3]]
            )
            after = times >= start
            for axis in range(3):
                poses[after, index, axis] = np.interp(times[after], path_times, path_xyz[:, axis])
            held = (times >= start) & (times <= start + 3)
            pinch[held, cycle] = poses[held, index, :3]
            opening[held, cycle] = 0.05
    return r


@pytest.mark.parametrize("layers", [2, 3])
def test_complete_task7_cycle_counts(layers):
    r = task7_recording(layers)
    report = score_recording(r, f"task7-{layers}")
    assert task_counts(f"task7-{layers}")[2:] == (5 * (layers - 1), 3 * (layers - 1))
    assert report["scores"] == dict(
        s=1, g=5 * (layers - 1), p=3 * (layers - 1), C=8 * (layers - 1), E=8 * (layers - 1), D=0
    )
    assert len({e["operation"] for e in report["events"]}) == 8 * (layers - 1)
    json.dumps(report, allow_nan=False)


def test_task7_passive_blocks_do_not_earn_push_or_transfer_credit():
    r = task7_recording()
    r.pinch[:] = 5
    r.opening[:] = 0.1
    report = score_recording(r, "task7-2")
    assert report["scores"]["C"] == 0


def test_task7_return_requires_release_and_original_tool_orientation():
    r = task7_recording()
    tool = tool_plan(r.spec, 2)[0]["tool"]
    mask = r.times > 10
    r.poses[mask, tool, 3:] = Rotation.from_euler("z", 90, degrees=True).as_quat()
    report = score_recording(r, "task7-2")
    assert report["scores"]["g"] == 5 and report["scores"]["p"] == 2
    assert report["scores"]["s"] == 0
    r = task7_recording()
    r.pinch[mask, 1] = r.poses[mask, tool, :3]
    r.opening[mask, 1] = 0.05
    assert score_recording(r, "task7-2")["tool_cycles"][0]["return_sample"] is None


def test_task7_partial_progress_does_not_require_successful_prefix():
    r = task7_recording()
    r.effort[:, 1] = 0
    report = score_recording(r, "task7-2")
    assert report["scores"]["g"] == 4 and report["scores"]["p"] == 3
    assert report["scores"]["s"] == 0


def test_task7_tool_contact_required_for_push():
    r = task7_recording()
    mask = (r.times >= 2) & (r.times <= 8)
    tool = tool_plan(r.spec, 2)[0]["tool"]
    r.poses[mask, tool, 2] += 0.2
    r.pinch[mask, 1, 2] += 0.2
    report = score_recording(r, "task7-2")
    assert all(p["push_sample"] is None for p in report["tool_cycles"][0]["pushes"])
    assert report["scores"]["g"] == 3


def test_task7_invalid_arm_assignment_rejected():
    r = task7_recording()
    r.spec["transfers"][0]["arm"] = 1
    with pytest.raises(ValueError, match="sequence"):
        score_recording(r, "task7-2")


def test_task7_late_return_breaks_order_without_erasing_progress():
    r = task7_recording()
    tool = tool_plan(r.spec, 2)[0]["tool"]
    r.poses[(r.times > 10) & (r.times < 42), tool, 2] += 0.1
    report = score_recording(r, "task7-2")
    assert report["scores"]["C"] == 8
    assert not report["prescribed_order"] and report["scores"]["s"] == 0


def test_task7_late_collapse_preserves_completed_steps():
    r = task7_recording()
    roof = next(i for i, b in enumerate(r.spec["blocks"]) if b["layer"] == 2)
    r.poses[-1, roof, 2] -= 0.03
    report = score_recording(r, "task7-2")
    assert report["scores"]["C"] == 8
    assert not report["terminal_geometry"] and report["scores"]["s"] == 0


def test_task7_disturbance_once_per_step_and_again_in_next_step():
    r = task7_recording()
    center = next(i for i, b in enumerate(r.spec["blocks"]) if b["name"] == "source_l1_1")
    r.poses[:, center, 0] += np.clip(r.times - 3, 0, 1) * 0.012
    r.poses[:, center, 0] += np.clip(r.times - 6, 0, 1) * 0.012
    report = score_recording(r, "task7-2")
    events = [e for e in report["disturbances"] if e["block"] == "source_l1_1"]
    assert {e["operation"] for e in events} == {"source_l1_2:tool_push", "source_l1_0:tool_push"}
    assert len(events) == 2
    assert all(not e["block"].startswith("pusher") for e in report["disturbances"])


def test_task7_placement_needs_ten_seconds_after_release():
    r = task7_recording()
    keep = r.times <= 36
    for name in ("times", "poses", "velocities", "pinch", "inverse_flange", "effort", "opening"):
        setattr(r, name, getattr(r, name)[keep])
    report = score_recording(r, "task7-2")
    assert report["scores"]["g"] == 5 and report["scores"]["p"] == 2


def test_task7_cli_without_workbook(tmp_path):
    import h5py
    from tower_eval.scoring.offline.__main__ import main

    r = task7_recording()
    run = tmp_path / "run"
    (run / "episode").mkdir(parents=True)
    (run / "episode_spec.json").write_text(json.dumps(r.spec))
    (run / "status.json").write_text(json.dumps(r.status))
    geometry = [
        dict(
            pinch_center_in_flange_m=[0, 0, 0],
            fingers=[dict(distal_link_corners_m=np.zeros((8, 3)).tolist()) for _ in (0, 1)],
        )
        for _ in (0, 1)
    ]
    (run / "finger_geometry.json").write_text(json.dumps(geometry))
    fingers = np.zeros((len(r.times), 2, 3, 7))
    fingers[..., 6] = 1
    fingers[..., :3] = r.pinch[:, :, None]
    np.savez(run / "finger-link-poses.npz", time=r.times, poses_m=fingers)
    np.savez(run / "finger-feedback.npz", time=r.times, forces_n=r.effort)
    qpos = np.zeros((len(r.times), 16))
    qpos[:, [7, 15]] = r.opening
    with h5py.File(run / "episode/states.hdf5", "w") as f:
        for key, data in dict(
            time=r.times, blocks=r.poses, block_velocities=r.velocities, qpos=qpos
        ).items():
            f[key] = data
    out = tmp_path / "result"
    assert main(["--run", str(run), "--task", "task7-2", "--output", str(out)]) == 0
    report = json.loads((out / "scores.json").read_text())[str(run)]
    assert report["scores"]["s"] == 1
    assert len(report["source_hashes"]) == 6
    assert not (out / "validation.json").exists()


def test_task7_extraction_cannot_start_before_tool_return():
    r = task7_recording()
    i = r.spec["transfers"][0]["block_index"]
    after = r.times >= 9
    for axis in range(3):
        r.poses[after, i, axis] = np.interp(r.times[after] + 2.3, r.times, r.poses[:, i, axis])
    # Start at 9.7 s, clear the tower after the tool's 10.05 s release.
    r.pinch[(r.times >= 12) & (r.times <= 15), 0] = 5
    r.opening[(r.times >= 12) & (r.times <= 15), 0] = 0.1
    held = (r.times >= 9.7) & (r.times <= 12.7)
    r.pinch[held, 0] = r.poses[held, i, :3]
    r.opening[held, 0] = 0.05
    report = score_recording(r, "task7-2")
    assert report["scores"]["C"] == 8
    assert not report["prescribed_order"] and report["scores"]["s"] == 0
