# SPDX-License-Identifier: Apache-2.0
"""Regression checks for the shipped campaign code, without replacing its physics."""

import copy
import importlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts.maintainer import archived_benchmark
from scripts.maintainer.archive_root import ArchivedPath, ArchivedToolPath

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location(
    "benchmark_202611", ROOT / "tools/simulator/benchmark.py"
)
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)

BUNDLE = ArchivedPath()
TOOLS = ArchivedToolPath()


def helper(name):
    spec = importlib.util.spec_from_file_location("benchmark_test_" + name, TOOLS / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def initial_recording(task, seconds=0.25):
    scorer = benchmark.load_scorer()
    trajectory = importlib.import_module(scorer.__package__ + ".offline.trajectory")
    scene = benchmark.read(BUNDLE / "scenes" / (task + ".json"))["scenes"][0]["candidate"][
        "episode_spec"
    ]
    times = np.arange(0, seconds + 0.001, 0.05)
    poses = np.repeat(np.asarray([b["pose"] for b in scene["blocks"]])[None], len(times), axis=0)
    inverse_flange = np.zeros((len(times), 2, 4))
    inverse_flange[..., 3] = 1
    return trajectory.Recording(
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


@pytest.mark.archive
def test_snapshot_and_all_600_scene_topologies():
    assert benchmark.verify_bundle()["scenes"] == 600
    contract = helper("six_task_contract")
    pairs = set()
    for path in sorted((BUNDLE / "scenes").glob("*.json")):
        body = benchmark.read(path)
        assert len(body["scenes"]) == 50
        for row in body["scenes"]:
            assert (
                contract.validate_spec(row["candidate"]["episode_spec"], body["task"])
                == body["task"]
            )
            assert row["candidate"]["seed"] == row["seed"]
            assert np.asarray(row["initial_state"]["qpos"]).shape == (16,)
            assert np.isfinite(row["initial_state"]["qpos"]).all()
            pairs.add((row["task"], row["seed"]))
    assert len(pairs) == 600


def test_snapshot_rejects_changed_source(tmp_path):
    (tmp_path / "source.py").write_text("changed")
    (tmp_path / "SOURCE_MANIFEST.json").write_text(
        json.dumps({"files": {"source.py": {"sha256": "0" * 64}}})
    )
    with pytest.raises(ValueError, match="Changed benchmark source"):
        benchmark.verify_bundle(tmp_path)


@pytest.mark.archive
@pytest.mark.parametrize(
    "task", [f"task{task}-{layers}" for task in (1, 2, 4, 5, 6, 7) for layers in (2, 3)]
)
def test_unmoved_initial_scene_is_valid_but_not_success(task):
    scorer = benchmark.load_scorer()
    rule = benchmark.read(BUNDLE / "protocol.json")["tasks"][task]["success_rule"]
    result = scorer.score_recording(initial_recording(task), task, success_rule=rule)
    assert result["valid"]
    assert result["scores"]["s"] == 0
    assert result["scores"]["E"] == result["scores"]["C"] + 1
    assert result.get("scoring_revision", result["protocol"]) == (
        "task5-measured-insertion-v1" if task.startswith("task5-") else rule
    )


@pytest.mark.archive
def test_task4_final_geometry_can_pass_without_operation_credit():
    scorer = benchmark.load_scorer()
    r = initial_recording("task4-2")
    for slot, target in enumerate(r.spec["layout_metadata"]["push_targets"]):
        r.poses[-1, target["block_index"], :3] = [0.7, -0.05 - slot * 0.07, 0.015]
    result = scorer.score_recording(r, "task4-2")
    assert result["scores"]["s"] == 1
    assert result["scores"]["g"] == result["scores"]["p"] == 0
    assert result["recorded_terminal_status"] == "duration_reached"
    assert result["scores"]["E"] == 0 and result["odr"]["value"] is None


@pytest.mark.archive
def test_task4_edge_contact_is_not_table_support():
    scorer = benchmark.load_scorer()
    module = importlib.import_module(scorer.__package__ + ".offline.task4_final_state")
    assert not module.table_supported(
        np.array([[0.3, -0.2, 0], [0.4, -0.2, 0], [0.3, -0.1, 0.04], [0.4, -0.1, 0.04]]),
        np.array([0.35, -0.15, 0.02]),
    )["passed"]


@pytest.mark.archive
def test_task6_late_recovery_does_not_erase_structural_fall():
    scorer = benchmark.load_scorer()
    module = importlib.import_module(scorer.__package__ + ".offline.task6_completion")
    r = initial_recording("task6-2")
    info = module.scene(r, "task6-2")
    index = info["structural"][0]
    r.poses[2, index, 2] -= info["dims"][index][2] * 0.6
    assert np.array_equal(r.poses[-1, index], r.poses[0, index])
    events = module.fall_events(r, info, module.BASELINE.DEFAULT_RULES)
    assert any(event["block_index"] == index for event in events)
    assert not module.completion(r, "task6-2")["passed"]


@pytest.mark.archive
def test_task7_continuous_collapse_cannot_recover():
    scorer = benchmark.load_scorer()
    r = initial_recording("task7-3")
    module = importlib.import_module(scorer.__package__ + ".offline.tool_extraction")
    trajectory = importlib.import_module(scorer.__package__ + ".offline.trajectory")
    baseline = []
    module.score_tools(r, 3, trajectory.DEFAULT_RULES, dict(events=[], operations=[]), [], baseline)
    index = next(
        i
        for i, b in enumerate(r.spec["blocks"])
        if b["tower"] == "source"
        and not b["fixed_on_table"]
        and i not in {t["block_index"] for t in r.spec["transfers"]}
    )
    r.poses[2, index, 2] -= 0.03
    changed = []
    module.score_tools(r, 3, trajectory.DEFAULT_RULES, dict(events=[], operations=[]), [], changed)
    assert baseline[0].all() and not changed[0][-1]
    assert changed[0][:2].all() and not changed[0][2:].any()


@pytest.mark.archive
def test_command_bounds_are_not_observation_tolerance(monkeypatch):
    monkeypatch.syspath_prepend(str(TOOLS))
    control = helper("control")
    spec = benchmark.read(archived_benchmark.RUNTIME / "shared/shared-ik-spec.json")["limits"]
    lower = np.asarray(spec["base_soft_cap_urdf"]["lower_rad"]) + np.deg2rad(1.5)
    upper = np.asarray(spec["base_soft_cap_urdf"]["upper_rad"]) - np.deg2rad(1.5)
    mask = np.array([0, 2, 3, 4, 5, 6])
    assert np.allclose(control.GENERATION_LOWER[mask], lower[mask], rtol=0, atol=1e-12)
    assert np.allclose(control.GENERATION_UPPER[mask], upper[mask], rtol=0, atol=1e-12)
    assert control.GENERATION_UPPER[1] == pytest.approx(1.74 - np.deg2rad(5))
    q = control.GENERATION_UPPER.copy()
    q[0] += 0.001
    with pytest.raises(control.PolicyActionError):
        control.validate_encoder_joints(q)
    control.validate_observed_encoder_joints(q)


@pytest.mark.archive
def test_history_grid_sampling_and_rejection(monkeypatch):
    monkeypatch.syspath_prepend(str(TOOLS))
    history = importlib.import_module("mme_history")
    hook = importlib.import_module("mme_history_hook")
    pixels = np.zeros((4, 240, 320, 3), dtype=np.uint8)
    collector = hook.EpisodeHistory("episode-1", history.CAMERAS, 4041)
    archive = SimpleNamespace(times=[0.0], _latest_pixels=pixels)
    assert not collector.append_validated(archive, 0, rejected=True, terminal=False)
    assert not collector.append_validated(archive, 0, rejected=False, terminal=True)
    for index in range(41):
        archive.times = [index / 10]
        pixels[:] = index
        assert collector.append_validated(archive, index * 40, rejected=False, terminal=False)
    request = collector.request(1600)
    assert request["indices"] == np.linspace(0, 40, 32, dtype=np.int32).tolist()
    current = history.resize_pad(np.concatenate((pixels[0], pixels[1]), axis=1))
    images, indices = history.validate_history(request, current, 4041)
    assert images.shape == (32, 224, 224, 3) and indices[-1] == 40
    tampered = copy.deepcopy(request)
    tampered["png_sha256"][0] = "0" * 64
    with pytest.raises(ValueError, match="digest mismatch"):
        history.validate_history(tampered, current, 4041)
    with pytest.raises(ValueError, match="timestamps differ"):
        collector.request(1640)
    collector.close()
    with pytest.raises(ValueError):
        collector.request(1600)


def test_existing_output_is_rejected_before_external_io(tmp_path):
    path = tmp_path / "request.json"
    path.write_text(
        json.dumps(
            {
                "task": "task6-2",
                "duration": 40,
                "success_rule": "tower-task6-roof-no-fall-v1",
                "output": str(tmp_path),
            }
        )
    )
    with pytest.raises(ValueError, match="never reuse"):
        benchmark.preflight_request(path)


@pytest.mark.archive
def test_shipped_kinematics_matches_urdf_chain():
    # Isolate the archived flat-module imports from the rest of the test suite.
    script = """
import sys
import numpy as np
sys.path.insert(0, sys.argv[1])
import kin
import urdf_fk
assert kin.LIMITS.shape == (7, 2)
for q in (np.zeros(7), kin.LIMITS.mean(axis=1)):
    expected = urdf_fk.fk(q)['gripper_base'] @ kin.TCP_LOCAL
    np.testing.assert_allclose(kin.fk(q), expected, rtol=0, atol=1e-12)
"""
    subprocess.run(
        [sys.executable, "-c", script, str(archived_benchmark.RUNTIME / "ik-capsule/tools")],
        check=True,
    )


@pytest.mark.archive
def test_camera_rejection_is_scene_local_and_consumed_at_exact_tick(tmp_path):
    camera = helper("camera_guard_terminal")
    label = "v3_probe_env_1:back_wrist"

    def reject(*args):
        raise RuntimeError(f"Wrist near plane may clip a task block: {label} (9.9 < 10.0 mm)")

    env = SimpleNamespace(elapsed=np.array([0.1, 0.1]), n=2)
    guard = camera.TerminalGuard(reject, env, tmp_path)
    args = ([], [0, 0, 0], [0, 0, 0], [0, 0, 0, 1], np.eye(3), 10.0, label)
    with pytest.raises(RuntimeError):  # Startup rejection remains fatal.
        guard.check(*args)
    guard.enabled = True
    with pytest.raises(RuntimeError):
        guard.check(*args[:5], 9.0, label)
    assert guard.check(*args) == 9.9
    with pytest.raises(AssertionError, match="Unconsumed camera rejection"):
        guard.consume(41)
    row = guard.consume(40)[1]
    assert set(guard.rejections) == {1}
    assert row["failure_tick"] == 40 and row["near_mm"] == 10.0
    assert row["original_guard_rejected"]
    assert not row["policy_observation_accepted"]
    assert not row["renderer_parameters_changed"]
    assert guard.consume(40) == {}


@pytest.mark.archive
def test_rejected_ik_does_not_retry_a_reversed_target(monkeypatch):
    monkeypatch.syspath_prepend(str(TOOLS))
    control = helper("control")
    calls = []

    def reject(poses, seed, **kwargs):
        calls.append(poses.copy())
        return dict(accepted=False, diagnostics=[dict(reason="no ordinary solution")])

    monkeypatch.setattr(control, "deployed_solve", reject)
    poses = np.tile(np.eye(4), (1, 2, 1, 1))
    response = dict(
        coordinate_frame="encoder_arm_base",
        j8_actuator_mask=[False, False],
        control_hz=10,
        execute_steps=1,
        mode="eef_delta",
        gripper_width_m=[[0.05, 0.05]],
        eef_encoder_targets=poses,
    )
    seed = np.tile((control.GENERATION_LOWER + control.GENERATION_UPPER) / 2, (2, 1))
    with pytest.raises(control.PolicyActionError, match="IK rejected policy target"):
        control.encoder_endpoints(response, {"joint_encoder": seed}, None, prefix=1)
    assert len(calls) == 1
    np.testing.assert_array_equal(calls[0], poses)
