"""Execute the command lifecycle against deterministic CPU I/O boundaries."""

import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from tower_sim.bench import bootstrap, cameras, contacts, replay, resources, source_receipt
from tower_sim.bench.camera_guard import TerminalGuard
from tower_sim.bench.fast_early_stop import CachedComputation
from tower_sim.bench.metrics import Metrics
from tower_sim.bench.scoring import BenchScorer
from tower_sim.bench.source_receipt import ImportReceipt
from tower_sim.bench.tasks import BenchTask, validate_spec

from scripts.maintainer.archive_root import ArchivedPath, ArchivedToolPath

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def simulation(tmp_path, monkeypatch, request):
    count = getattr(request, "param", 1)
    events = []
    q = np.zeros(16)
    q[[7, 15]] = 0.05
    state = NS(
        time=0.0,
        qpos=q,
        qvel=np.zeros(16),
        eef=np.zeros((2, 7)),
        eef_world=np.zeros((2, 7)),
        blocks=np.array([[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]]),
        block_velocities=np.zeros((1, 6)),
    )
    names = [f"joint{i}" for i in range(1, 8)] + ["gripper_joint1", "gripper_joint2"]
    robot = NS(
        dof_names=names,
        body_names=["gripper_base", "gripper_link1", "gripper_link2"],
        set_joint_position_targets=lambda *a, **kw: events.append("command"),
        set_joint_velocity_targets=lambda *a, **kw: None,
    )
    prim = NS(
        GetRelationship=lambda name: NS(GetTargets=lambda: [f"cam{i}" for i in range(count)]),
        GetAppliedSchemas=lambda: [],
        GetAttributes=lambda: [],
    )
    env = NS(
        n=count,
        read=lambda: [state] * count,
        elapsed=np.zeros(count),
        simulation_time=0.0,
        robots=[robot, robot],
        previous=np.zeros((count, 16)),
        cameras=[{}] * count,
        stage=NS(GetPrimAtPath=lambda path: prim),
        lula=NS(tcp_rotation=Rotation.identity(), tcp_offset=np.zeros(3)),
        offsets=np.zeros((count, 3)),
        link_poses=[np.zeros((count, 3, 7))] * 2,
        raw_joint_efforts=[np.zeros((count, 9))] * 2,
        raw_joint_states=[(np.zeros((count, 9)),)] * 2,
        finger_geometry=[],
    )
    env._commands = lambda targets, arm: (
        np.arange(9),
        np.c_[targets[:, arm * 8 : arm * 8 + 7], np.zeros((count, 2))],
    )
    env.physics = NS(simulate=lambda *a: events.append("physics"), fetch_results=lambda: None)
    compensation = NS(
        physical_from_encoder16=lambda x: x.copy(),
        command_targets=lambda *a: (q[None].copy(), q[None].copy()),
    )
    context = bootstrap.BenchContext(
        NS(close=lambda: events.append("app.close")),
        NS(),
        env,
        NS(read=lambda name: {}),
        compensation,
        np.zeros((2, 7)),
        {},
        tmp_path,
    )
    monkeypatch.setattr(bootstrap, "verify_v4", lambda *a: events.append("drive"))
    context.physical_command_limits = lambda i: np.array([[-10.0, 10.0]] * 16)
    row = dict(
        env_id=0,
        output=tmp_path / "seed-1",
        candidate={"seed": 1, "episode_spec": {}},
        initial={"qpos": q.copy(), "physical_qpos": q.copy()},
        plan=tmp_path,
        times=np.array([0.0, 0.05]),
        actions=np.stack([q, q]),
        ticks=20,
    )
    rows = [
        dict(
            row,
            env_id=i,
            output=tmp_path / f"seed-{i + 1}",
            candidate={"seed": i + 1, "episode_spec": {}},
        )
        for i in range(count)
    ]
    row = rows[0]
    for item in rows:
        item["output"].mkdir()

    def make_contacts(env, rows):
        for item in rows:
            resources.save(item["output"] / "contact-reporting.json", {})
        return NS(close=lambda: events.append("contacts.close"), settling=[[] for _ in rows])

    def settle(rows, env, contact):
        events.append("settled")
        for item in rows:
            resources.save(item["output"] / "settling.json", {"passed": True})

    monkeypatch.setattr(replay, "settle", settle)
    rgb = np.random.default_rng(0).integers(0, 255, (count, 4, 240, 320, 3), dtype=np.uint8)
    camera = NS(
        names={str(i): str(i) for i in range(4)},
        products={str(i): NS(path="p") for i in range(4)},
        batch_camera_paths={str(i): [f"cam{j}" for j in range(count)] for i in range(4)},
        reference_paths={},
        verified_capture_steps=0,
        last_render_delivery={},
        controls={},
        profile={"renderer_controls": []},
        report=lambda: {"environment_count": count},
        close=lambda: events.append("camera.close"),
    )

    def observe():
        camera.verified_capture_steps += 1
        events.append("observe")
        return {"rgb": rgb}

    camera.observe = observe

    def make_camera(*a, **kw):
        assert "settled" in events
        events.append("camera.create")
        return camera

    monkeypatch.setattr(cameras, "verify_mount_evidence", lambda *a, **kw: events.append("c19"))
    monkeypatch.setattr(cameras, "ordinary_references", lambda camera: (rgb.copy(), {}))
    monkeypatch.setitem(
        sys.modules,
        "pxr",
        NS(UsdGeom=NS(XformCache=lambda: NS(GetLocalToWorldTransform=lambda prim: np.eye(4)))),
    )
    guard = TerminalGuard(lambda *a: 0.0, env, tmp_path / "guard")
    factories = replay.ReplayFactories(
        contacts=make_contacts,
        camera=make_camera,
        guard=lambda *a: guard,
        serial=lambda camera, index=0, **kw: NS(observe=lambda: (rgb[index].copy(), {})),
        observation=lambda *a: dict(
            joint_encoder=np.zeros((2, 7)).tolist(), gripper_width_m=[0.05, 0.05]
        ),
    )
    task = replace(
        BenchTask.from_resources(
            "task1-2", resources.Resources.from_roots(checkout=ROOT, assets=ROOT)
        ),
        deadline_s=0.05,
    )
    records = []
    metrics = Metrics(
        NS(emit=records.append, flush=lambda: None), lambda seed, tick: f"injected/{seed}/{tick}"
    )

    def sources():
        assert "camera.close" not in events
        events.append("sources")
        return {}

    def run(**kwargs):
        return replay.execute(
            context,
            rows,
            task,
            metrics=metrics,
            source_collector=sources,
            observed_validator=lambda *a: None,
            expert_sha256="a" * 64,
            factories=factories,
            **kwargs,
        )

    return NS(
        run=run,
        task=task,
        context=context,
        row=row,
        rows=rows,
        events=events,
        guard=guard,
        factories=factories,
        records=records,
        camera=camera,
        env=env,
    )


def test_complete_lifecycle_and_collection(simulation):
    s = simulation
    assert s.run() == 0
    assert s.events.count("drive") == 2
    assert s.events.count("sources") == 2
    assert s.events.index("settled") < s.events.index("camera.create") < s.events.index("command")
    assert s.events[-1] == "camera.close"
    receipt = json.loads((s.context.output / "v4-gain-final-readback.json").read_text())
    assert receipt["checks"] == 2
    assert s.guard.enabled
    assert any(r.get("request_id") == "injected/1/0" for r in s.records)
    assert json.loads((s.row["output"] / "status.json").read_text())["evidence_complete"]


def test_invalid_physical_row_rejects_before_motion(simulation):
    s = simulation
    s.context.compensation.command_targets = lambda *a: (np.zeros((2, 16)), np.full((2, 16), 20.0))
    with pytest.raises(ValueError, match="physical"):
        s.run()
    assert not s.events


def test_guard_rejection_is_consumed_and_terminal(simulation):
    s = simulation
    original = s.camera.observe

    def observe():
        if s.env.elapsed[0] > 0 and not s.guard.rejections:

            def reject(*args):
                raise RuntimeError(
                    "Wrist near plane may clip a task block: live_env_0:left_wrist (9 < 10 mm)"
                )

            s.guard.original = reject
            s.guard.check([], [], [], [], [], 10.0, "live_env_0:left_wrist")
        return original()

    s.camera.observe = observe
    assert s.run() == 0
    assert s.guard.consumed == {0}
    result = json.loads((s.row["output"] / "policy-action-failure.json").read_text())
    assert result["terminal_evidence_complete"]
    assert result["category"] == "camera_near_plane_guard"


def test_contact_windows_fail_closed(monkeypatch):
    def construct(env, rows, **kw):
        assert kw["reference_windows"] is True
        return kw["contact_windows"](rows[0]["parameters"], rows[0]["candidate"])

    monkeypatch.setattr(contacts, "BatchContacts", construct)
    with pytest.raises(KeyError, match="keyframes"):
        contacts.make_contacts(
            None,
            [{"parameters": {}, "candidate": {"episode_spec": {"blocks": []}}}],
            reference_windows=True,
        )


def test_shared_cache_and_capacity():
    task = BenchTask.from_resources(
        "task1-2", resources.Resources.from_roots(checkout=ROOT, assets=ROOT)
    )
    cache = CachedComputation(max_entries=1)
    scorers = [BenchScorer(task, mode="fast", computation=cache) for _ in range(2)]
    assert scorers[0].computation is scorers[1].computation is cache
    with pytest.raises(ValueError, match="capacity"):
        CachedComputation(max_entries=0)


def test_atomic_exclusive_receipt(tmp_path, monkeypatch):
    path = tmp_path / "receipt.json"

    def fail(source, target):
        assert json.loads(Path(source).read_text()) == {"checks": 2}
        assert not Path(target).exists()
        raise OSError("publication interrupted")

    monkeypatch.setattr(resources.os, "link", fail)
    with pytest.raises(OSError, match="interrupted"):
        resources.save_exclusive(path, {"checks": 2})
    assert not path.exists()
    assert list(tmp_path.iterdir()) == []


def test_yaml_receipt_timing_and_actual_bytes(tmp_path, monkeypatch):
    path = tmp_path / "yaml.py"
    path.write_text("# actual implementation\n")
    modules = {}
    monkeypatch.setattr(source_receipt, "sys", NS(modules=modules, addaudithook=sys.addaudithook))
    receipt = ImportReceipt()
    try:
        receipt.observe("import", ("yaml", None))
        receipt.app_factory(lambda: None)()
        receipt.observe("import", ("yaml._yaml", str(path)))
        monkeypatch.setitem(modules, "yaml", NS(__file__=str(path), __version__="test"))
        result = receipt.collect()["yaml"]
        assert result["loaded"]["yaml"]["sha256"] == resources.sha(path)
        assert [r["phase"] for r in result["first_import_events"]][-2:] == [
            "before_simulation_app",
            "after_simulation_app",
        ]
    finally:
        receipt.close()


@pytest.mark.archive
def test_registry_topology_all_scenes(frozen_loader):
    frozen = frozen_loader("_p708_topology", ArchivedToolPath("six_task_contract.py"))
    for path in (ArchivedPath("scenes")).glob("*.json"):
        for row in json.loads(path.read_text())["scenes"]:
            spec = row["candidate"]["episode_spec"]
            assert validate_spec(spec) == frozen.validate_spec(spec)


def test_capture_scope_separates_serial_from_batch(tmp_path, monkeypatch):
    proof = {label: {"comparison": {"passed": True}} for label in ("initial", "batch_end")}
    calls = []
    monkeypatch.setattr(cameras, "verify_mount_evidence", lambda *a, **kw: calls.append("batch"))
    replay.qualify_capture(1, {}, proof, proof, {}, tmp_path)
    assert calls == []
    assert json.loads((tmp_path / "camera-qualification.json").read_text())["c19_applied"] is False
    replay.qualify_capture(2, {}, proof, proof, {}, tmp_path)
    assert calls == ["batch"]
    with pytest.raises(ValueError, match="2..10"):
        replay.qualify_capture(11, {}, proof, proof, {}, tmp_path)
    bad = {**proof, "batch_end": {"comparison": {"passed": False}}}
    with pytest.raises(ValueError, match="Serial"):
        replay.qualify_capture(1, {}, proof, bad, {}, tmp_path)


def test_sources_failure_still_closes_camera(simulation):
    s = simulation

    def fail(*a):
        raise ValueError("drive drift")

    s.context.collect_sources = fail
    with pytest.raises(ValueError, match="drive drift"):
        s.run()
    assert s.events[-1] == "camera.close"
    assert not (s.context.output / "v4-gain-final-readback.json").exists()


def test_bootstrap_extent_prevalidation_visits_every_environment(simulation):
    s = simulation
    other = dict(s.row, env_id=1)
    seen = []

    def limits(i):
        seen.append(i)
        return np.array([[-10.0, 10.0]] * 16) if i == 0 else np.zeros((16, 2))

    s.context.physical_command_limits = limits
    with pytest.raises(ValueError, match="physical"):
        replay.validate_trajectory(s.context, [s.row, other])
    assert seen == [0, 1]


def test_injected_metrics_invalid_request_id_fails(simulation):
    s = simulation
    s.records.clear()
    # A non-JSON request identifier must fail at the injected Metrics boundary.
    metrics = Metrics(NS(emit=s.records.append, flush=lambda: None), lambda *a: object())
    with pytest.raises(TypeError):
        metrics.emit("command_replay_request", request_id=metrics.request_id(1, 0))
    assert s.records == []


def test_process_cache_is_passed_into_loop_scorers(simulation):
    seen = []

    def scorer(task, **kwargs):
        assert isinstance(kwargs["computation"], CachedComputation)
        seen.append(kwargs["computation"])
        return BenchScorer(task, **kwargs)

    simulation.factories.scorer = scorer
    assert simulation.run() == 0
    assert len(seen) == 1 and seen[0].max_entries == 1_000_000


def test_yaml_first_import_is_observed_in_fresh_process():
    import subprocess

    script = """
from tower_sim.bench.source_receipt import ImportReceipt
r=ImportReceipt()
import yaml
assert any(e['module']=='yaml' and e['phase']=='before_simulation_app' for e in r.events), r.events
r.app_factory(lambda:None)()
assert r.module('yaml')['sha256']
r.close()
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_live_import_receipt_rejects_generation_and_frozen_paths(tmp_path):
    from tower_sim.bench.source_receipt import validate_imports

    for name, path in [
        ("jenga_generator.motion", tmp_path / "motion.py"),
        ("hidden", tmp_path / "frozen/helper.py"),
    ]:
        with pytest.raises(ValueError, match="Forbidden candidate import"):
            validate_imports(
                {"modules": {name: {"path": str(path)}}}, forbidden_roots=(tmp_path / "frozen",)
            )


@pytest.mark.parametrize("simulation", [1, 2], indirect=True)
def test_execute_multi_environment_contract(simulation):
    s = simulation
    seen, scorers = [], []
    original_limits = s.context.physical_command_limits

    def limits(index):
        if "settled" not in s.events:
            seen.append(index)
        return original_limits(index)

    def scorer(task, **kwargs):
        result = BenchScorer(task, **kwargs)
        scorers.append(result)
        return result

    s.context.physical_command_limits = limits
    s.factories.scorer = scorer
    assert s.run() == 0
    assert seen == list(range(s.env.n))
    assert len(scorers) == s.env.n
    assert all(s.computation is scorers[0].computation for s in scorers)
    receipt = json.loads((s.context.output / "camera-qualification.json").read_text())
    assert receipt["environment_count"] == s.env.n
    assert receipt["c19_applied"] is (s.env.n == 2)
    assert s.events.count("c19") == (s.env.n == 2)
    assert s.events.count("drive") == s.env.n + 1
    gains = json.loads((s.context.output / "v4-gain-final-readback.json").read_text())
    assert gains["checks"] == s.env.n + 1
    bounds = json.loads((s.row["output"] / "observed-joint-bounds.json").read_text())
    from nero_kinematics.profiles import ROOT as profiles

    assert bounds["profile_sha256"] == resources.sha(profiles / "command.json")
    assert bounds["ik_bounds"] is None and "joint_absolute" in bounds["ik_bounds_scope"]


@pytest.mark.parametrize("cleanup_failure", [None, "sources", "camera", "results"])
def test_loop_error_preserved_without_passing_gain_receipt(
    simulation, monkeypatch, cleanup_failure
):
    s = simulation
    error = ValueError("physics simulate original failure")

    def fail(*a, **kw):
        raise error

    def fail_cleanup(*a, **kw):
        raise OSError("secondary cleanup failure")

    s.env.physics.simulate = fail
    if cleanup_failure == "sources":
        s.context.collect_sources = fail_cleanup
    elif cleanup_failure == "camera":
        s.camera.close = fail_cleanup
    elif cleanup_failure == "results":
        monkeypatch.setattr(replay, "save_batch_results", fail_cleanup)
    with pytest.raises(ValueError, match="physics simulate original failure") as caught:
        s.run()
    assert caught.value is error
    if cleanup_failure:
        assert "secondary cleanup failure" in " ".join(error.__notes__)
    assert not (s.context.output / "v4-gain-final-readback.json").exists()
    assert not (s.context.output / "camera-qualification.json").exists()


def test_cleanup_failure_after_completed_loop_still_fails(simulation):
    s = simulation

    def fail():
        raise OSError("camera cleanup failure")

    s.camera.close = fail
    with pytest.raises(RuntimeError, match="camera cleanup failure"):
        s.run()
    assert not (s.context.output / "v4-gain-final-readback.json").exists()


def test_cleanup_failure_inside_caller_handler_is_raised(simulation):
    s = simulation

    def fail():
        raise OSError("camera cleanup failure")

    s.camera.close = fail
    with pytest.raises(RuntimeError, match="camera cleanup failure"):
        try:
            raise KeyError("unrelated caller exception")
        except KeyError:
            s.run()
    assert not (s.context.output / "v4-gain-final-readback.json").exists()


@pytest.mark.parametrize("count", [0, 11])
def test_environment_count_rejected_at_entry(simulation, count):
    simulation.env.n = count
    with pytest.raises(ValueError, match="1..10"):
        simulation.run()
    assert not simulation.events


def test_atomic_receipt_mode_and_directory_fsync(tmp_path, monkeypatch):
    import os
    import stat

    calls = []
    real = os.fsync

    def fsync(fd):
        calls.append(stat.S_ISDIR(os.fstat(fd).st_mode))
        real(fd)

    monkeypatch.setattr(resources.os, "fsync", fsync)
    path = tmp_path / "receipt.json"
    resources.save_exclusive(path, {"passed": True})
    assert path.stat().st_mode & 0o777 == 0o644
    assert calls == [False, True]


def test_atomic_receipt_requires_hard_links(tmp_path, monkeypatch):
    import errno

    def fail(*a):
        raise OSError(errno.EOPNOTSUPP, "unsupported")

    monkeypatch.setattr(resources.os, "link", fail)
    with pytest.raises(OSError, match="requires hard-link support"):
        resources.save_exclusive(tmp_path / "receipt.json", {})
    assert list(tmp_path.iterdir()) == []


def test_frozen_stem_allowance_requires_namespace_and_expected_root(tmp_path):
    from tower_sim.bench.source_receipt import validate_imports

    known = tmp_path / "known_package"
    kwargs = dict(forbidden_stems={"model"}, allowed_packages={"package": [known]})
    validate_imports({"modules": {"package.model": {"path": str(known / "model.py")}}}, **kwargs)
    for name, path in (
        ("package.model", tmp_path / "outside/model.py"),
        ("hidden_alias", known / "model.py"),
    ):
        with pytest.raises(ValueError, match="Forbidden candidate import"):
            validate_imports({"modules": {name: {"path": str(path)}}}, **kwargs)


def assert_infrastructure_batch(s):
    for row in s.rows:
        status = json.loads((row["output"] / "status.json").read_text())
        assert status["status"] == "infrastructure_failure"
        assert status["evidence_complete"] is False
        assert not (row["output"] / "score.json").exists()


@pytest.mark.parametrize("simulation", [2], indirect=True)
def test_first_delivered_zero_view_never_reaches_policy(simulation):
    s = simulation
    original = s.camera.observe
    requests = []

    def observe():
        out = original()
        if s.camera.verified_capture_steps == 3:
            # Startup and assignment proof use earlier, healthy renders.
            pixels = out["rgb"].copy()
            pixels[0, 3] = 0
            return dict(rgb=pixels)
        return out

    s.camera.observe = observe
    policy = NS(
        package=dict(package_digest="a" * 64, checkpoint_sha256="b" * 64),
        solver=NS(identity={}),
        start=lambda *a: {},
        initialize=lambda *a: None,
        close=lambda: None,
        inference="test",
        prefetch=lambda *a: requests.append(a),
        append_camera=lambda *a, **kw: None,
    )
    with pytest.raises(RuntimeError, match="Renderer failure in delivered camera views"):
        s.run(policy=policy)
    assert_infrastructure_batch(s)
    assert requests == []
    assert not any("request_id" in record for record in s.records)


@pytest.mark.parametrize("simulation", [2], indirect=True)
@pytest.mark.parametrize("failed_views", [1, 4])
def test_mid_episode_blackout_aborts_every_environment(simulation, failed_views):
    s = simulation
    original = s.camera.observe
    serial = s.factories.serial

    def blacken(pixels):
        pixels = pixels.copy()
        pixels[-failed_views:] = 0
        return pixels

    def observe():
        out = original()
        if s.env.elapsed[1] > 0:
            pixels = out["rgb"].copy()
            pixels[1] = blacken(pixels[1])
            return dict(rgb=pixels)
        return out

    def reference(camera, index=0, **kwargs):
        source = serial(camera, index, **kwargs)

        def observe_reference():
            pixels, receipt = source.observe()
            if index == 1 and s.env.elapsed[1] > 0:
                pixels = blacken(pixels)
            return pixels, receipt

        return NS(observe=observe_reference)

    s.camera.observe = observe
    s.factories.serial = reference
    with pytest.raises(RuntimeError, match="Renderer failure in delivered camera views"):
        s.run()
    assert_infrastructure_batch(s)
    assert s.env.elapsed[1] > 0
    assert any("request_id" in record for record in s.records)
    detail = next((s.rows[1]["output"] / "camera-frames.hdf5.diagnostics").glob("*.json"))
    delivery = json.loads(detail.read_text())
    assert delivery["time_s"] > 0
    assert sum(q["maximum"] == 0 for q in delivery["cameras"].values()) == failed_views


def test_bright_low_quality_near_plane_terminal_is_policy_failure(simulation, monkeypatch):
    s = simulation
    original = s.camera.observe
    serial = s.factories.serial
    ordinary = cameras.ordinary_references

    def occlude(pixels):
        pixels = pixels.copy()
        if s.env.elapsed[0] > 0:
            pixels[3] = 128  # Bright, flat wrist occlusion; three healthy fixed/other views.
        return pixels

    def observe():
        if s.env.elapsed[0] > 0 and not s.guard.rejections:

            def reject(*args):
                raise RuntimeError(
                    "Wrist near plane may clip a task block: live_env_0:left_wrist (9 < 10 mm)"
                )

            s.guard.original = reject
            s.guard.check([], [], [], [], [], 10.0, "live_env_0:left_wrist")
        return dict(rgb=occlude(original()["rgb"][0])[None])

    def reference(camera, index=0, **kwargs):
        source = serial(camera, index, **kwargs)

        def observe_reference():
            pixels, receipt = source.observe()
            return occlude(pixels), receipt

        return NS(observe=observe_reference)

    def ordinary_reference(camera):
        pixels, receipt = ordinary(camera)
        return occlude(pixels[0])[None], receipt

    s.camera.observe = observe
    s.factories.serial = reference
    monkeypatch.setattr(cameras, "ordinary_references", ordinary_reference)
    assert s.run() == 0
    status = json.loads((s.row["output"] / "status.json").read_text())
    assert status["status"] == "diagnostic_prefix_complete"
    assert status["evidence_complete"] is True
    terminal = json.loads((s.row["output"] / "policy-action-failure.json").read_text())
    assert terminal["category"] == "camera_near_plane_guard"
    assert terminal["terminal_evidence_complete"] is True
    assert terminal["rejected_observation_sent_to_policy"] is False
    import h5py

    with h5py.File(s.row["output"] / "camera-frames.hdf5") as archive:
        visibility = json.loads(archive.attrs["verified_low_visibility_json"])
        assert visibility
        for frame in visibility.values():
            assert frame["cameras"]["3"]["maximum"] == 128
            assert frame["cameras"]["3"]["valid"] is False
            assert frame["delivery"]["valid"] is True
