import ast
import json
import sys
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pytest
from tower_eval.tasks import load_tasks
from tower_sim.backend.resources import DATA
from tower_sim.bench import app, bootstrap, drives, scene
from tower_sim.bench.kinematics import EefModel
from tower_sim.bench.resources import Resources

from scripts.maintainer.archive_root import ArchivedPath, ArchivedToolPath
from tests.characterization.runtime_scenarios import ROOT, RUNTIME


def test_resolver_and_no_pinned_data_copies(tmp_path):
    resources = Resources.from_roots(checkout=ROOT, assets=tmp_path)
    for name in (
        "runtime_profiles",
        "scene_cache",
        "shared_ik_spec",
        "camera_profile",
        "motion_profile",
        "tcp_cache",
        "tcp_fit_receipt",
        "calibration_report",
    ):
        assert resources.path(name).is_relative_to(DATA)
        assert resources.path(name).suffix != ".py"
    assert resources.paths["kinematics"] == tmp_path / "configuration/template/kinematics"
    assert (
        resources.paths["center_bond"]
        == tmp_path / "configuration/USER_CENTER_BOND_AMENDMENT_C74_583.json"
    )
    with pytest.raises(FileNotFoundError, match="Missing bench resource"):
        resources.path("scene")


def test_semantic_cache_uses_local_type_and_rechecks_loaded_bytes(monkeypatch, tmp_path):
    from tower_eval.calibration import cached_tcp_calibration_v2, tcp_calibration_key

    cache = json.loads((ROOT / "benchmarks/2026.11/calibration/tcp-v2.json").read_text())
    profile = json.loads(
        (ROOT / "packages/tower-sim/src/tower_sim/data/motion_profile_v1.json").read_text()
    )
    model_source = (
        ROOT / "packages/nero-kinematics/src/nero_kinematics/models/nero_sim_reference.urdf"
    )
    (tmp_path / "nero.urdf").write_bytes(model_source.read_bytes())
    (tmp_path / "nero.yaml").write_text("test-only yaml")
    key = tcp_calibration_key(
        profile=profile,
        stage_sha256="a" * 64,
        kinematics=tmp_path,
        fit_channel_sha256=cache["key"]["fit_source"]["fit_channel_sha256"],
    )
    cache["key"] = key
    path = tmp_path / "tcp.json"
    path.write_text(json.dumps(cache))
    session = NS(models=[])
    local_type = bootstrap.model_type(session, None)
    original = local_type.calibrate_tcp
    with cached_tcp_calibration_v2(local_type, path, key=key, kinematics=tmp_path):
        model = local_type(None, tmp_path)
        model.calibrate_tcp(profile)
        np.testing.assert_array_equal(model.tcp_offset, cache["tcp"]["translation_m"])
        assert not hasattr(EefModel, "calibrate_tcp")
        (tmp_path / "nero.yaml").write_text("changed after context entered")
        with pytest.raises(ValueError, match="kinematics mismatch"):
            model.calibrate_tcp(profile)
    assert local_type.calibrate_tcp is original
    assert session.models == [model]


def test_missing_limit_attribute_is_value_error(tmp_path):
    source = ROOT / "packages/nero-kinematics/src/nero_kinematics/models/nero_sim_reference.urdf"
    import xml.etree.ElementTree as ET

    tree = ET.fromstring(source.read_bytes())
    joint = tree.find("joint[@name='joint1']/limit")
    del joint.attrib["upper"]
    path = tmp_path / "model.urdf"
    path.write_bytes(ET.tostring(tree))
    with pytest.raises(ValueError, match="lower and upper"):
        EefModel(scene_urdf=path)


@pytest.mark.archive
def test_shader_flags_match_frozen_and_cleanup(frozen_loader, monkeypatch, tmp_path):
    old = frozen_loader("_p706_shader", ArchivedToolPath("private_shader_cache.py"))
    seed = tmp_path / "seed"
    seed.mkdir()
    (seed / "shader.bin").write_bytes(b"unchanged verified seed")
    hashes = tmp_path / "seed.sha256"
    hashes.write_text(app.digest_tree(seed))
    private = tmp_path / "private"
    monkeypatch.setattr(old, "SEED", seed)
    monkeypatch.setattr(old, "SEED_SHA", hashes)
    monkeypatch.setattr(old, "PRIVATE_ROOT", private)
    monkeypatch.setattr(old.time, "time", lambda: 10.0)
    flags, expected = old.prepare("fixture")
    import shutil

    shutil.rmtree(private / "fixture")
    argv = ["worker", "--existing"]
    with (
        pytest.raises(RuntimeError, match="construction failure"),
        app.private_shader_cache(
            seed=seed,
            seed_sha=hashes,
            private_root=private,
            name="fixture",
            argv=argv,
            clock=lambda: 10.0,
        ) as actual,
    ):
        assert actual == expected
        assert argv == ["worker", "--existing", *flags]
        raise RuntimeError("construction failure")
    assert argv == ["worker", "--existing"]
    assert not (private / "fixture").exists()
    hashes.write_text("bad seed")
    with (
        pytest.raises(RuntimeError, match="differs"),
        app.private_shader_cache(
            seed=seed, seed_sha=hashes, private_root=private, name="bad", argv=argv
        ),
    ):
        pass
    assert not (private / "bad").exists()


def test_app_failure_closes_without_patching_global(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "sys", NS(version_info=(3, 12)))
    monkeypatch.setitem(sys.modules, "isaacsim", NS(__file__=str(tmp_path / "__init__.py")))
    closed = []

    def renderer():
        raise RuntimeError("renderer failed")

    with (
        app.native_render_initialization(
            lambda options: NS(close=lambda: closed.append(True)), renderer
        ) as factory,
        pytest.raises(RuntimeError, match="renderer failed"),
    ):
        factory({})
    assert closed == [True]
    closed.clear()

    def setup(session):
        raise RuntimeError("model failed")

    with (
        app.native_render_initialization(
            lambda options: NS(close=lambda: closed.append(True)), lambda: None
        ) as native,
        app.grouped_runtime(
            native,
            (0,),
            tmp_path,
            sdk=tmp_path,
            experience=lambda *a: "fixture.kit",
            model_setup=setup,
        ) as launch,
        pytest.raises(RuntimeError, match="model failed"),
    ):
        launch({"headless": True})
    assert closed == [True]


@pytest.mark.parametrize("startup", ["headless", "stock"])
def test_candidate_launch_disables_new_crash_uploads(tmp_path, monkeypatch, startup):
    monkeypatch.setattr(app, "sys", NS(version_info=(3, 12)))
    monkeypatch.setitem(sys.modules, "isaacsim", NS(__file__=str(tmp_path / "__init__.py")))
    captured = []
    config = dict(headless=True, enable_crashreporter=True)
    with app.grouped_runtime(
        lambda options: captured.append(options),
        (0,),
        tmp_path,
        sdk=tmp_path,
        startup=startup,
        experience=lambda *a: "fixture.kit",
    ) as launch:
        launch(config)
    assert config["enable_crashreporter"] is True
    assert captured[0]["enable_crashreporter"] is False
    assert json.loads((tmp_path / "launch.json").read_text())["enable_crashreporter"] is False


class Robot:
    def __init__(self, count=3):
        self.dof_names = [f"joint{i}" for i in range(1, 8)] + ["gripper_joint1", "gripper_joint2"]
        self.kp = np.ones((count, 9))
        self.kd = self.kp * 2
        self.effort = self.kp * 3
        self.velocity = self.kp * 4
        self.inertia = self.kp * 5
        self.friction = np.ones((count, 9, 3))
        self._physics_view = NS(
            get_dof_friction_properties=lambda: self.friction,
            set_dof_friction_properties=lambda a, ids: setattr(self, "friction", a.copy()),
            set_dof_max_velocities=lambda a, ids: setattr(self, "velocity", a.copy()),
        )

    def get_gains(self):
        return self.kp, self.kd

    def set_gains(self, kp, kd, **kwargs):
        self.kp, self.kd = kp.copy(), kd.copy()

    def get_max_efforts(self):
        return self.effort

    def set_max_efforts(self, a):
        self.effort = a.copy()

    def get_joint_max_velocities(self):
        return self.velocity

    def get_armatures(self):
        return self.inertia

    def set_armatures(self, a):
        self.inertia = a.copy()


@pytest.mark.archive
def test_calibrated_usd_adapter_exact(frozen_loader, monkeypatch, tmp_path):
    path = tmp_path / "station.usd"
    path.write_text("CPU-only drive fixture")
    config = {"calibration": "fixture"}
    monkeypatch.setitem(sys.modules, "j2comp", NS(SCENE=path, CONFIG=config))
    original = frozen_loader("_p706_drives", "ik-capsule/runtime/adapter.py")

    def joint(arm, name):
        return NS(
            GetAttribute=lambda key: NS(
                Get=lambda: {
                    "physxJoint:maxJointVelocity": 40.0,
                    "physxJoint:armature": 7000.0,
                    "physxJointAxis:angular:staticFrictionEffort": 2.0,
                    "physxJointAxis:angular:dynamicFrictionEffort": 3.0,
                    "physxJointAxis:angular:viscousFrictionCoefficient": 4.0,
                    "rss:commandDelaySeconds": 0.035,
                }[key]
            )
        )

    api = NS(
        GetStiffnessAttr=lambda: NS(Get=lambda: 123.5),
        GetDampingAttr=lambda: NS(Get=lambda: 17.5),
        GetMaxForceAttr=lambda: NS(Get=lambda: 3000.0),
    )
    monkeypatch.setitem(sys.modules, "pxr", NS(UsdPhysics=NS(DriveAPI=lambda *args: api)))
    new = drives.CalibratedDrives(path, config)
    monkeypatch.setattr(original, "joint", joint)
    monkeypatch.setattr(new, "joint", joint)
    for arm in ("left", "back"):
        a, b = Robot(), Robot()
        expected = original.apply_usd(a, arm)
        actual = new.apply_usd(b, arm)
        assert json.dumps(actual) == json.dumps(expected)
        for field in ("kp", "kd", "effort", "velocity", "inertia", "friction"):
            assert getattr(a, field).tobytes() == getattr(b, field).tobytes()
    assert new.readbacks == original.READBACKS


def test_clone_hook_is_instance_owned_and_fails_closed(tmp_path):
    events = []

    class Cloner:
        def __init__(self, stage):
            pass

        def clone(self, *args, **kwargs):
            events.append((args, kwargs))
            return 42

    original = Cloner.clone
    plans = [{"env_path": "/World/envs/env_0"}, {"env_path": "/World/envs/env_1"}]
    hook = scene.CloneHook(plans, None, author=lambda *args: events.append("author"))
    clone = hook.factory(Cloner, None)
    assert Cloner.clone is original
    assert (
        clone.clone("/World/envs/env_0", [p["env_path"] for p in plans], replicate_physics=True)
        == 42
    )
    hook.verify()
    assert events[0][1] == dict(replicate_physics=False, copy_from_source=True)
    assert events[1] == "author"
    with pytest.raises(ValueError, match="Only one"):
        clone.clone("/World/envs/env_0", [p["env_path"] for p in plans])
    with pytest.raises(RuntimeError, match="did not invoke"):
        scene.CloneHook(plans, None).verify()


@pytest.mark.archive
def test_v4_gains_and_readback_exact(monkeypatch, tmp_path):
    """Execute frozen gain functions directly; only the profile file pin is stubbed."""
    source = (ArchivedToolPath("v4_policy_pilot.py")).read_text()
    functions = [
        node
        for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name in ("apply_v4", "verify_v4")
    ]
    namespace = {
        "Path": Path,
        "np": np,
        "json": json,
        "os": NS(environ={"TOWER_ASSET_ROOT": str(tmp_path)}),
    }
    exec(
        compile(ast.Module(body=functions, type_ignores=[]), "<frozen-v4-gain-functions>", "exec"),
        namespace,
    )
    profile = {
        "joints": [
            {"joint": f"joint{i}", "kp_Nm_per_rad": 100.0 * i, "kv_Nm_s_per_rad": 10.0 * i}
            for i in range(1, 8)
        ]
    }
    (tmp_path / "configuration").mkdir()
    path = tmp_path / "configuration/official-nero-profile.json"
    path.write_text(json.dumps(profile))
    pin = "7a120927b6c0be5c879ec849e097aab797f37791f2c4b7679b6ea0527bbc84b6"
    monkeypatch.setattr(drives, "sha", lambda p: pin)
    monkeypatch.setitem(
        sys.modules,
        "pxr",
        NS(
            UsdGeom=NS(GetStageMetersPerUnit=lambda s: 0.001),
            UsdPhysics=NS(GetStageKilogramsPerUnit=lambda s: 1.0),
        ),
    )
    outputs = []
    from tower_sim.bench.resources import save

    for name, fn in (("old", namespace["apply_v4"]), ("new", drives.apply_v4)):
        output = tmp_path / name
        output.mkdir()
        env = NS(robots=[Robot(), Robot()], n=3, stage=None)
        records = (
            fn(env, output, NS(sha=lambda p: pin, save=save))
            if name == "old"
            else fn(env, output, path)
        )
        outputs.append((output / "v4-gain-readback.json").read_bytes())
        drives.verify_v4(env, records)
        namespace["verify_v4"](env, records)
        env.robots[0].kp[1, 3] += 1
        with pytest.raises(AssertionError):
            drives.verify_v4(env, records)
    assert outputs[0] == outputs[1]


@pytest.mark.archive
def test_task7_authored_geometry_and_bond_source_parity():
    """AST equality of the actual author/verification algorithms (imports ignored)."""
    from tower_sim.bench import task7

    for old_path, old_name, new_name in [
        (ArchivedToolPath("task7_runtime.py"), "verify_tools", "verify_tools"),
        (ArchivedToolPath("source_center_bond_c74_583.py"), "install", "install_bond"),
        (ArchivedToolPath("source_center_bond_c74_583.py"), "assess", "assess"),
    ]:
        trees = []
        for source, name in (
            ((RUNTIME / old_path).read_text(), old_name),
            (Path(task7.__file__).read_text(), new_name),
        ):
            node = next(
                n
                for n in ast.parse(source).body
                if isinstance(n, ast.FunctionDef) and n.name == name
            )
            node.name = "compared"
            node.body = [n for n in node.body if not isinstance(n, (ast.Import, ast.ImportFrom))]
            trees.append(ast.dump(node, include_attributes=False))
        assert trees[0] == trees[1]


def test_final_readback_precedes_source_collection_and_fails_closed(monkeypatch, tmp_path):
    from contextlib import ExitStack

    events = []
    context = bootstrap.BenchContext(ExitStack(), None, NS(n=3), None, None, [], [], tmp_path)
    path = tmp_path / "v4-gain-final-readback.json"
    monkeypatch.setattr(bootstrap, "verify_v4", lambda *args: events.append("readback"))
    with pytest.raises(RuntimeError, match="without successful checks"):
        context.finalize_drive_receipt()
    assert not path.exists()
    for count in (1, 2):
        assert context.collect_sources(lambda: events.append("sources") or "result") == "result"
        assert context._drive_checks == count
        assert not path.exists()
    assert events == ["readback", "sources", "readback", "sources"]

    def fail(*args):
        raise ValueError("physical readback changed")

    monkeypatch.setattr(bootstrap, "verify_v4", fail)
    with pytest.raises(ValueError, match="physical readback changed"):
        context.collect_sources(lambda: events.append("must not collect"))
    assert len(events) == 4 and context._drive_checks == 2
    monkeypatch.setattr(bootstrap, "verify_v4", lambda *args: events.append("readback"))
    # Frozen loop includes an additional direct/final check, counted with collectors.
    assert context.verify_drives()["checks"] == 3
    receipt = context.finalize_drive_receipt()
    assert (
        receipt
        == json.loads(path.read_text())
        == dict(passed=True, all_environments=3, checks=3, checked_before_camera_teardown=True)
    )
    original = path.read_bytes()
    context.verify_drives()
    with pytest.raises(FileExistsError):
        context.finalize_drive_receipt()
    assert path.read_bytes() == original


@pytest.mark.archive
def test_construction_block_counts_match_frozen(frozen_loader):
    frozen = frozen_loader("_p706_task_counts", ArchivedToolPath("six_task_contract.py"))
    assert {key: value[2] for key, value in frozen.TASKS.items()} == {
        t.legacy_id: t.block_count for t in load_tasks() if t.status == "benchmark"
    }


@pytest.mark.archive
@pytest.mark.parametrize(
    "path",
    [f"{t.legacy_id}.json" for t in load_tasks() if t.status == "benchmark"],
)
def test_stored_scene_topologies_pass(frozen_loader, monkeypatch, path):
    path = ArchivedPath("scenes") / path
    frozen = frozen_loader("_p706_task_topologies", ArchivedToolPath("six_task_contract.py"))
    candidate = json.loads(path.read_text())["scenes"][0]["candidate"]
    # Assets live outside the checkout; stub only their file reads. The stored
    # topology, task metadata, physical properties and digest logic remain real.
    hashes = {
        str(Path(block[field]).resolve()): block[digest]
        for block in candidate["episode_spec"]["blocks"]
        for field, digest in (
            ("asset_reference", "asset_sha256"),
            ("texture_atlas", "texture_sha256"),
        )
        if block[field] is not None
    }
    monkeypatch.setattr(scene, "_file_hash", hashes.__getitem__)
    (plan,) = scene.plans_for_candidates([candidate], frozen.validate_spec)
    assert plan["task"] == path.stem
    assert len(plan["blocks"]) == frozen.TASKS[path.stem][2]


@pytest.mark.archive
def test_construction_topology_rejects_inconsistent_layers():
    with pytest.raises(ValueError, match="layer count"):
        bootstrap.ConstructionTask("task7-3", 2, lambda spec: "task7-3")
    for task in {t.legacy_id: t.block_count for t in load_tasks() if t.status == "benchmark"}:
        candidate = json.loads((ArchivedPath(f"scenes/{task}.json")).read_text())["scenes"][0][
            "candidate"
        ]
        candidate["episode_spec"]["blocks"].pop()
        with pytest.raises(ValueError, match="named block slots"):
            scene.plans_for_candidates([candidate], lambda spec, task=task: task)
        candidate["episode_spec"]["layout_metadata"]["movable_layers"] = 10
        with pytest.raises(ValueError, match="movable_layers"):
            scene.plans_for_candidates([candidate], lambda spec, task=task: task)


def test_live_physical_command_limits_and_gate(tmp_path):
    from contextlib import ExitStack

    from tower_sim.bench.control import validate_physical_targets

    robots = [Robot(2), Robot(2)]
    actual = np.arange(36).reshape(2, 9, 2) / 100
    actual[:, :, 0] *= -1
    for robot in robots:
        robot.dof_names.reverse()
        robot._physics_view.get_dof_limits = lambda: actual
    context = bootstrap.BenchContext(
        ExitStack(), None, NS(n=2, robots=robots), None, None, [], [], tmp_path
    )
    limits = context.physical_command_limits(1)
    for arm, robot in enumerate(robots):
        np.testing.assert_array_equal(
            limits[arm * 8 : arm * 8 + 7],
            actual[1, [robot.dof_names.index(f"joint{i}") for i in range(1, 8)]],
        )
        np.testing.assert_array_equal(limits[arm * 8 + 7], [0, 0.1])
    validate_physical_targets(np.zeros((2, 16)), limits)
    targets = np.zeros((2, 16))
    targets[1, 7] = 0.1001
    with pytest.raises(ValueError):
        validate_physical_targets(targets, limits)
    actual[1, 8, 1] = 0.001
    assert context.physical_command_limits(1)[0, 1] == 0.001
    with pytest.raises(IndexError):
        context.physical_command_limits(2)


def test_startup_environment_unsets_after_updates(monkeypatch):
    import os

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
    with app.process_environment({"CUDA_VISIBLE_DEVICES": "3"}, unset=("CUDA_VISIBLE_DEVICES",)):
        assert "CUDA_VISIBLE_DEVICES" not in os.environ
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "7"


@pytest.mark.parametrize("entry", ["grouped", "direct"])
@pytest.mark.parametrize("version", [(3, 11), (3, 13)])
def test_python_guard_precedes_sdk_import(monkeypatch, tmp_path, entry, version):
    import builtins

    monkeypatch.setattr(app, "sys", NS(version_info=version))
    original = builtins.__import__

    def blocked(name, *args, **kwargs):
        if name == "isaacsim":
            pytest.fail("Isaac import occurred before the Python guard")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", blocked)
    with pytest.raises(RuntimeError, match="requires Python 3.12"):
        if entry == "direct":
            app.simulation_app({})
        else:
            with app.grouped_runtime(
                lambda config: pytest.fail("App launched before the Python guard"),
                (0,),
                tmp_path,
                sdk=tmp_path,
            ) as launch:
                launch({"headless": True})
    assert not (tmp_path / "launch.json").exists()


@pytest.mark.parametrize("with_project", [True, False])
def test_source_package_version_without_metadata(monkeypatch, tmp_path, with_project):
    import importlib.metadata
    import runpy
    import tomllib

    import tower_sim

    def missing(name):
        assert name == "tower-sim"
        raise importlib.metadata.PackageNotFoundError(name)

    monkeypatch.setattr(importlib.metadata, "version", missing)
    package = tmp_path / "src/tower_sim"
    package.mkdir(parents=True)
    initializer = package / "__init__.py"
    initializer.write_bytes(Path(tower_sim.__file__).read_bytes())
    expected = "0+unknown"
    if with_project:
        project = (ROOT / "packages/tower-sim/pyproject.toml").read_text()
        (tmp_path / "pyproject.toml").write_text(project)
        expected = tomllib.loads(project)["project"]["version"]
    assert runpy.run_path(str(initializer))["__version__"] == expected
