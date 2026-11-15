"""Campaign streams and registry integration; no native simulator startup."""

from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest
from tower_eval.scoring.computation import REFERENCE
from tower_eval.scoring.detectors import early_success
from tower_eval.scoring.recording import Recording
from tower_eval.scoring.rulesets.bench_2026_11 import DEFAULT_RULES
from tower_eval.tasks import load_tasks
from tower_sim.bench.fast_early_stop import CachedComputation
from tower_sim.bench.resources import Resources
from tower_sim.bench.scoring import BenchScorer
from tower_sim.bench.tasks import BenchTask
from tower_sim.bench.termination import terminal_evidence

from scripts.maintainer.archive_root import ArchivedToolPath
from scripts.maintainer.bench_scoring_replay import replay
from scripts.maintainer.golden_support import GOLDEN, ROOT, golden_archive, read, sha, unpack
from tests.characterization.golden_compare import assert_equivalent

TASKS = [t for t in load_tasks() if t.status == "benchmark"]
RESOURCES = Resources.from_roots(checkout=ROOT, assets=ROOT)


@pytest.mark.archive
@pytest.mark.parametrize("task", TASKS, ids=lambda t: t.legacy_id)
def test_task_protocol_mapping(task, frozen_loader):
    frozen = frozen_loader("_p707_tasks", ArchivedToolPath("six_task_contract.py"))
    protocol = RESOURCES.read("protocol")["tasks"][task.legacy_id]
    mapped = BenchTask.from_resources(task.id, RESOURCES)
    assert mapped.identity == task
    number, layers, blocks, _, deadline = frozen.TASKS[task.legacy_id]
    assert (task.number, task.layers, task.block_count) == (number, layers, blocks)
    assert mapped.deadline_s == task.deadline_s == protocol["duration_s"] == deadline
    assert mapped.duration_ticks == protocol["duration_s"] * 400
    assert mapped.early_success_stop is protocol["early_success_stop"]
    assert mapped.success_rule == protocol["success_rule"]
    assert (
        frozen.validate_request(
            dict(task=task.legacy_id, duration=mapped.deadline_s, success_rule=mapped.success_rule)
        )
        == task.legacy_id
    )
    with pytest.raises(ValueError, match="deadline"):
        frozen.validate_request(
            dict(task=task.legacy_id, duration=135, success_rule=mapped.success_rule)
        )
    assert BenchTask.from_resources(task.paper, RESOURCES) == mapped
    assert (
        not task.legacy_id.startswith("task7-")
        and mapped.success_rule in ("tower-task12-four-place-hold-v1", "tower-sim-trajectory-v5")
    ) is mapped.early_success_stop


@pytest.mark.parametrize(
    "field,value", [("duration_s", 135), ("early_success_stop", 1), ("success_rule", "wrong")]
)
def test_protocol_disagreement_rejected(tmp_path, field, value):
    import json

    protocol = RESOURCES.read("protocol")
    protocol["tasks"]["task2-2"][field] = value
    path = tmp_path / "protocol.json"
    path.write_text(json.dumps(protocol))
    with pytest.raises(ValueError, match="mismatch"):
        BenchTask.from_resources("task2-2", Resources(dict(RESOURCES.paths, protocol=path)))


@pytest.mark.golden
@pytest.mark.parametrize(
    "reference", [False, pytest.param(True, marks=pytest.mark.archive)], ids=["public", "frozen"]
)
@pytest.mark.parametrize("trial", read(GOLDEN / "MANIFEST.json")["trials"], ids=lambda t: t["id"])
def test_full_campaign_streams(trial, tmp_path, reference):
    archive = golden_archive(trial["archive"])
    assert sha(archive) == trial["sha256"]
    unpack(archive, tmp_path)
    results = replay(tmp_path, trial["task"], RESOURCES, assert_equivalent, reference=reference)
    if trial["id"] == "task1-3-openwam":
        assert results["fast"]["cache_stats"]["hits"] > 0
        assert results["fast"]["cache_stats"]["angular_hits"] > 0


def sample():
    poses = np.zeros((2, 2, 7))
    poses[:, :, -1] = 1
    poses[:, 0, 2], poses[:, 1, 2] = 0.03, 0.01
    return Recording(
        {},
        np.array([0.0, 0.05]),
        poses,
        np.zeros((2, 2, 6)),
        np.zeros((2, 2, 3)),
        np.zeros((2, 2, 4)),
        np.zeros((2, 2, 2)),
        np.zeros((2, 2)),
        {},
        {},
    )


def support(cache, r, *, rules=DEFAULT_RULES, dimensions=None):
    dims = dimensions or [[0.1, 0.1, 0.02]] * 2
    args = r, 0, [1], dims, np.ones(len(r.times), dtype=bool), rules
    actual = cache.support_mask(*args)
    np.testing.assert_array_equal(actual, REFERENCE.support_mask(*args))
    return actual


def test_cache_hits_across_ticks_and_state_changes():
    cache, r = CachedComputation(), sample()
    assert support(cache, r).all()
    first = cache.stats.copy()
    assert support(cache, replace(r, times=r.times + 0.05)).all()
    assert cache.stats["hits"] == first["hits"] + 2
    for block, channel in ((0, 0), (1, 0)):
        changed = deepcopy(r)
        changed.poses[:, block, channel] += 0.3
        assert not support(cache, changed).any()
    changed = deepcopy(r)
    changed.velocities[:, 0, 0] = 10
    assert not support(cache, changed).any()
    computed = cache.stats["computed"]
    support(cache, r, rules=replace(DEFAULT_RULES, support_com_margin_m=0.003))
    support(cache, r, dimensions=[[0.12, 0.1, 0.02], [0.1, 0.1, 0.02]])
    assert cache.stats["computed"] == computed + 4
    assert cache.stats["audited_samples"] > 0


def test_angular_dtype_bytes_audit_and_fallback():
    cache = CachedComputation(angular_audit_every=1)
    q = np.array([0.0, 0.0, 0.0, 1.0])
    for dtype in (np.float32, np.float64):
        value = q.astype(dtype)
        assert (
            cache.angular_error(value, value).tobytes()
            == REFERENCE.angular_error(value, value).tobytes()
        )
        cache.angular_error(value.copy(), value.copy())
    assert cache.stats["angular_computed"] == 2
    assert cache.stats["angular_hits"] == cache.stats["angular_audited"] == 2
    np.testing.assert_array_equal(
        cache.angular_error(q[None], q), REFERENCE.angular_error(q[None], q)
    )
    key = next(iter(cache.angles))
    cache.angles[key] += 1
    with pytest.raises(RuntimeError, match="angular audit"):
        cache.angular_error(q.astype(np.float32), q.astype(np.float32))


@pytest.mark.parametrize("full", [False, True])
def test_support_audit_detects_corruption(full):
    cache = CachedComputation(audit_samples=0 if full else 4, full_audit_every=2)
    r = sample()
    support(cache, r)
    table = next(iter(cache.cache.values()))
    for key in table:
        table[key] = False
    with pytest.raises(RuntimeError, match="full audit" if full else "sample audit"):
        support(cache, r)


def test_cache_capacity_and_reference_fallback():
    cache, r = CachedComputation(max_entries=1), sample()
    support(cache, r)
    assert cache.stats["cache_clears"] > 0
    changed = replace(
        r, poses=r.poses.astype(np.float16), velocities=r.velocities.astype(np.float16)
    )
    support(cache, changed)
    assert cache.stats["fallback"] == 1


@pytest.mark.parametrize("task", [t for t in TASKS if t.number in (4, 6, 7)], ids=lambda t: t.id)
def test_protocol_disabled_latch_cannot_stop(task):
    scorer = BenchScorer(BenchTask.from_resources(task.id, RESOURCES), mode="audit")
    assert scorer.detector is None
    scorer.detector = SimpleNamespace(
        success={"terminal_tick": 4000},
        check=lambda *a, **k: pytest.fail("disabled detector called"),
    )
    assert scorer.check(None, 4000) is None


def test_canonical_identity_cadence_failure_and_audit():
    scorer = BenchScorer(BenchTask.from_resources("task5-2", RESOURCES), mode="audit")
    assert type(scorer.detector) is early_success.TaskAwareHoldStop
    scorer.detector.success = {"terminal_tick": 4000}
    assert scorer.check(None, 800) is None
    assert scorer.check(None, 4001) is None
    assert scorer.check(None, 4000, failed=True) is None
    scorer.reference_detector.success = {"terminal_tick": 4400}
    with pytest.raises(RuntimeError, match="detector audit"):
        scorer.check(None, 4000)


@pytest.mark.archive
@pytest.mark.parametrize(
    "trial",
    [
        r
        for r in read(GOLDEN / "MANIFEST.json")["trials"]
        if r["task"][4] in "4567" and r["id"].endswith("openwam")
    ],
    ids=lambda r: r["id"],
)
def test_task4567_detector_computations(trial, tmp_path):
    from tower_eval.scoring import api

    from scripts.maintainer.golden_support import arrays, json_value, reference_scorer

    unpack(golden_archive(trial["archive"]), tmp_path)
    data = arrays(tmp_path)
    rows = [{key: v[i] for key, v in data.items()} for i in range(len(data["time"]))]
    task = BenchTask.from_resources(trial["task"], RESOURCES)
    adapter = BenchScorer(task)
    record = adapter.recording(
        read(tmp_path / "episode_spec.json"), rows, read(tmp_path / "finger_geometry.json")
    )
    cache = CachedComputation()
    reference = api.score_recording(record, task.identity.id, ruleset="bench-2026.11")
    for _ in range(2):
        fast = api.score_recording(
            record, task.identity.id, ruleset="bench-2026.11", computation=cache
        )
        assert json_value(fast) == json_value(reference)
    # Task4/6 have detector implementations even though the protocol disables
    # their runtime early stop. Task7 deliberately has no early-stop detector.
    frozen = reference_scorer()
    if task.identity.number == 7:
        for module in (frozen, early_success):
            with pytest.raises(ValueError, match="supported task"):
                module.TaskAwareHoldStop(trial["task"])
    else:
        candidate = early_success.TaskAwareHoldStop(trial["task"], computation=cache)
        original = frozen.TaskAwareHoldStop(trial["task"])
        for tick in (4000, 4400):
            # Repeating an unchanged physical state at successive check clocks
            # covers cross-tick reuse without modifying immutable inputs.
            checked = replace(record, times=np.linspace(0, tick / 400, len(record.times)))
            assert candidate.check(checked, tick) == original.check(checked, tick)


@pytest.mark.archive
def test_frozen_execution_uses_request_deadline_not_formal_default():
    import ast

    runtime = ArchivedToolPath()
    for name, attribute in (
        ("batch_policy_worker.py", False),
        ("batch_policy_worker_mme.py", False),
        ("v4_evaluation.py", True),
    ):
        tree = ast.parse((runtime / name).read_text())
        candidates = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.Assign)
            and any(
                isinstance(target, ast.Attribute) and target.attr == "duration_ticks"
                if attribute
                else isinstance(target, ast.Name) and target.id == "duration_ticks"
                for target in n.targets
            )
        ]
        assert len(candidates) == 1
        expression = compile(ast.Expression(candidates[0].value), name, "eval")
        for task in TASKS:
            mapped = BenchTask.from_resources(task.id, RESOURCES)
            assert (
                eval(
                    expression,
                    {
                        "args": SimpleNamespace(duration=mapped.deadline_s),
                        "request": dict(duration=mapped.deadline_s),
                    },
                )
                == mapped.duration_ticks
            )


def test_angular_cache_key_includes_both_quaternions():
    cache = CachedComputation(angular_audit_every=10**9)
    q = np.array([0.0, 0.0, 0.0, 1.0])
    r1 = np.array([0.0, 0.0, np.sin(0.1), np.cos(0.1)])
    r2 = np.array([0.0, np.sin(0.2), 0.0, np.cos(0.2)])
    # Same pose against different anchors, then a different pose against the same anchor.
    for pose, anchor in ((q, r1), (q, r2), (r1, r2), (q, r2)):
        assert (
            cache.angular_error(pose, anchor).tobytes()
            == REFERENCE.angular_error(pose, anchor).tobytes()
        )
    assert cache.stats["angular_computed"] == 3 and cache.stats["angular_hits"] == 1


def _terminal_inputs(task, *, tick):
    times = [0.0, tick / task.physics_hz]
    rows = [dict(time=value) for value in times]
    report = dict(valid=True, scores=dict(s=0), protocol="bench-2026.11")
    return rows, list(times), list(times), report


@pytest.mark.parametrize(
    "case,match",
    [
        ("tick-zero", "Invalid terminal physics tick"),
        ("tick-past-deadline", "Invalid terminal physics tick"),
        ("camera-clock", "Terminal evidence clocks differ"),
        ("command-clock", "Terminal evidence clocks differ"),
        ("terminal-clock", "Terminal evidence clocks differ"),
        ("failure-and-success", "both failure and early success"),
        ("success-protocol", "Early success terminal differs"),
        ("success-tick", "Early success terminal differs"),
        ("abort-before-failure", "Abort evidence tail differs"),
        ("abort-tail-at-zero", "Abort evidence tail differs"),
        ("rejected-action-applied", "Abort evidence tail differs"),
        ("unfailed-early-end", "did not reach its deadline"),
        ("scorer-invalid", "Scorer rejected physical recording evidence"),
        ("offline-success-on-early-stop-task", "Offline success disagrees"),
    ],
)
def test_terminal_evidence_rejections(case, match):
    task = BenchTask.from_resources("task2-2", RESOURCES)
    assert task.early_success_stop
    deadline = task.duration_ticks
    tick = deadline
    kwargs = {}
    if case == "tick-zero":
        tick = 0
    elif case == "tick-past-deadline":
        tick = deadline + 1
    elif case in ("failure-and-success", "success-protocol", "success-tick"):
        tick = 800
        kwargs["success"] = dict(protocol=task.success_rule, terminal_tick=tick)
        if case == "failure-and-success":
            kwargs["failure"] = dict(failure_tick=tick, rejected_action_applied=False)
        elif case == "success-protocol":
            kwargs["success"]["protocol"] = "other"
        else:
            kwargs["success"]["terminal_tick"] = tick + 1
    elif case in ("abort-before-failure", "abort-tail-at-zero", "rejected-action-applied"):
        tick = 800
        failure = dict(failure_tick=tick, rejected_action_applied=False)
        if case == "abort-before-failure":
            failure["failure_tick"] = tick + 1
        elif case == "abort-tail-at-zero":
            failure["failure_tick"] = 0  # The frozen tail after a tick-0 abort is exactly 20.
        else:
            failure["rejected_action_applied"] = True
        kwargs["failure"] = failure
    elif case == "unfailed-early-end":
        tick = deadline - 1
    # terminal-clock: rows, cameras and commands agree with each other, not with the tick.
    clock_tick = tick - 1 if case == "terminal-clock" else max(tick, 1)
    rows, cameras, commands, report = _terminal_inputs(task, tick=clock_tick)
    if case == "camera-clock":
        cameras[-1] += 1 / task.physics_hz
    elif case == "command-clock":
        commands[-1] += 1 / task.physics_hz
    elif case == "scorer-invalid":
        report["valid"] = False
    elif case == "offline-success-on-early-stop-task":
        report["scores"]["s"] = 1
    with pytest.raises(ValueError, match=match):
        terminal_evidence(task, rows, cameras, commands, report, tick=tick, **kwargs)


def test_terminal_evidence_accepts_valid_endings():
    task = BenchTask.from_resources("task2-2", RESOURCES)
    rows, cameras, commands, report = _terminal_inputs(task, tick=task.duration_ticks)
    result = terminal_evidence(task, rows, cameras, commands, report, tick=task.duration_ticks)
    assert result["status"] == "duration_reached"
    rows, cameras, commands, report = _terminal_inputs(task, tick=20)
    failure = dict(failure_tick=0, rejected_action_applied=False)
    result = terminal_evidence(task, rows, cameras, commands, report, tick=20, failure=failure)
    assert result["status"] == "task_failure"
