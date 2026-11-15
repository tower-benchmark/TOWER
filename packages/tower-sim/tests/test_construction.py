"""Execute original and candidate construction with the same CPU SDK doubles.

The doubles expose SDK call boundaries, not alternate physics. Numerical scene,
cache, drive and camera contracts have separate tests; GPU equivalence is checked independently.
"""

import contextlib
import copy
import importlib
import json
import sys
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace as NS
from unittest.mock import MagicMock

import numpy as np
import pytest
from tower_sim.bench import bootstrap, environment, experience, scene
from tower_sim.bench.resources import Resources, save, sha

from scripts.maintainer.archive_root import ArchivedPath, ArchivedToolPath
from tests.characterization.runtime_scenarios import ROOT, RUNTIME


def install(monkeypatch, name, **attributes):
    module = ModuleType(name)
    module.__dict__.update(attributes)
    monkeypatch.setitem(sys.modules, name, module)
    if "." in name:
        parent, child = name.rsplit(".", 1)
        if parent not in sys.modules:
            install(monkeypatch, parent)
        monkeypatch.setattr(sys.modules[parent], child, module, raising=False)
    return module


def normalized(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [normalized(v) for v in value]
    if isinstance(value, dict):
        return {k: normalized(v) for k, v in value.items()}
    if isinstance(value, Path):
        return str(value)
    return value


class SDK:
    def __init__(self, monkeypatch, count, blocks):
        self.events = []
        self.count, self.blocks = count, blocks
        self.stage = MagicMock(name="stage")
        self.prim = MagicMock(name="prim")
        self.prim.GetName.return_value = "collider"
        self.prim.GetPath.return_value = "/World/envs/env_0/Station/InstalledRobots/link"
        self.prim.GetPrim.return_value = self.prim
        self.stage.GetPrimAtPath.return_value = self.prim
        self.stage.DefinePrim.return_value = self.prim
        self.settings = {}
        self.robots = []

        def set_setting(key, value):
            self.settings[key] = value
            self.record("setting", key, value)

        controls = NS(set=set_setting, get=self.settings.get)
        install(monkeypatch, "carb", settings=NS(get_settings=lambda: controls))
        pxr = install(monkeypatch, "pxr")
        geom = MagicMock()
        geom.Xform.Define.return_value = self.prim
        geom.Tokens.z = "z"
        geom.XformCache.return_value.GetLocalToWorldTransform.return_value = np.eye(4)
        geom.SetStageMetersPerUnit.side_effect = lambda stage, v: self.record("meters", v)
        geom.SetStageUpAxis.side_effect = lambda stage, v: self.record("up", v)
        geom.GetStageMetersPerUnit.return_value = 0.001
        gf = NS(Vec3d=lambda *v: v, Vec3f=lambda *v: v, Quatf=lambda *v: v, Vec2f=lambda *v: v)
        phys = MagicMock()
        phys.GetStageKilogramsPerUnit.return_value = 1.0
        usd = NS(PrimRange=lambda *a: [self.prim], TraverseInstanceProxies=lambda: None)
        for name, value in dict(
            Gf=gf,
            Usd=usd,
            UsdGeom=geom,
            UsdPhysics=phys,
            UsdShade=MagicMock(),
            PhysxSchema=MagicMock(),
        ).items():
            setattr(pxr, name, value)
        self.usd_mocks = {
            "stage": self.stage,
            "prim": self.prim,
            "UsdGeom": geom,
            "UsdPhysics": phys,
            "UsdShade": pxr.UsdShade,
            "PhysxSchema": pxr.PhysxSchema,
        }
        self.stage.SetTimeCodesPerSecond.side_effect = lambda v: self.record("time_codes", v)
        install(
            monkeypatch,
            "omni.usd",
            get_context=lambda: NS(
                new_stage=lambda: self.record("new_stage"), get_stage=lambda: self.stage
            ),
        )
        timeline = NS(
            **{
                name: lambda name=name: self.record("timeline", name)
                for name in ("play", "pause", "commit")
            }
        )
        install(monkeypatch, "omni.timeline", get_timeline_interface=lambda: timeline)
        install(monkeypatch, "omni.physx", get_physx_simulation_interface=lambda: NS())
        pc = NS(
            prim_path="/physics",
            **{
                name: lambda v, name=name: self.record(name, v)
                for name in ("enable_gpu_dynamics", "set_broadphase_type", "set_solver_type")
            },
        )
        sdk = self

        class Context:
            def __init__(self, **kwargs):
                sdk.record("context", kwargs)

            def get_physics_context(self):
                return pc

            def initialize_physics(self):
                sdk.record("initialize_physics")

            def set_simulation_dt(self, **kwargs):
                sdk.record("dt", kwargs)

        install(monkeypatch, "isaacsim.core.api", SimulationContext=Context)

        class Cloner:
            def __init__(self, stage):
                sdk.record("cloner")

            def define_base_env(self, path):
                sdk.record("base_env", path)

            def clone(self, source, paths, **kwargs):
                sdk.record("clone", source, paths, kwargs)

            def filter_collisions(self, *args):
                sdk.record("filter_collisions", args)

        self.Cloner = Cloner
        install(monkeypatch, "isaacsim.core.cloner", Cloner=Cloner)

        class Robot:
            def __init__(self, paths, **kwargs):
                self.prim_paths = paths
                self.dof_names = [f"joint{i}" for i in range(1, 8)] + [
                    "gripper_joint1",
                    "gripper_joint2",
                ]
                self.body_names = ["gripper_base", "gripper_link", "gripper_link1", "gripper_link2"]
                self.count = count
                sdk.robots.append(self)
                sdk.record("articulation", paths, kwargs)

            def initialize(self):
                sdk.record("robot_initialize")

        class Blocks:
            def __init__(self, pattern, **kwargs):
                self.prim_paths = [
                    f"/World/envs/env_{i}/Blocks/block_{b:02}"
                    for i in range(count)
                    for b in range(blocks)
                ]
                sdk.record("blocks", pattern, kwargs)

            def initialize(self):
                sdk.record("blocks_initialize")

        install(monkeypatch, "isaacsim.core.prims", Articulation=Robot, RigidPrim=Blocks)
        install(
            monkeypatch,
            "isaacsim.core.simulation_manager",
            SimulationManager=NS(get_simulation_time=lambda: 0.25),
        )
        backend = importlib.import_module("tower_sim.backend.backend")
        assets = importlib.import_module("tower_sim.backend.assets")
        replay = importlib.import_module("tower_sim.replay.dual_replay_worker")
        profile = {
            "provenance": {},
            "gripper": {"finger_effort_limit": 3.0},
            "arms": {arm: {"command_delays_seconds": [0.01] * 7} for arm in ("left", "back")},
        }
        monkeypatch.setattr(
            backend, "load_runtime_profiles", lambda *args, **kwargs: copy.deepcopy(profile)
        )
        monkeypatch.setattr(
            importlib.import_module("tower_sim.backend.runtime_profiles"),
            "load_runtime_profiles",
            lambda *args, **kwargs: copy.deepcopy(profile),
        )
        monkeypatch.setattr(backend, "calibration", lambda: ({}, {}))
        monkeypatch.setattr(backend, "rigid_pose", lambda matrix: np.r_[np.zeros(6), 1.0])
        for module in (backend, assets):
            monkeypatch.setattr(
                module, "normalize_reset_stacks", lambda *args: sdk.record("normalize")
            )
        monkeypatch.setattr(assets, "block_reference", lambda *args: ("block.usd", "/Block"))

        class PhysicsOutput:
            def __init__(self, settings, **kwargs):
                self.report = kwargs
                sdk.record("writeback", kwargs)

            def finish_initialization(self, **kwargs):
                sdk.record("finish_initialization", kwargs)

        monkeypatch.setattr(
            importlib.import_module("tower_sim.backend.physics_output"),
            "PhysicsOutputPolicy",
            PhysicsOutput,
        )
        monkeypatch.setattr(
            importlib.import_module("tower_sim.backend.grasp_geometry"),
            "inspect_fingers",
            lambda *args: {},
        )
        monkeypatch.setattr(
            importlib.import_module("tower_sim.backend.physics_capacity"),
            "configure_capacity",
            lambda pc, *args: sdk.record("capacity", args),
        )

        def prepare(stage, root, **kwargs):
            return sdk.record("prepare_scene", root, kwargs)

        monkeypatch.setattr(
            importlib.import_module("tower_sim.backend.physics_scene"),
            "prepare_physics_scene",
            prepare,
        )
        monkeypatch.setattr(
            replay,
            "_apply_gripper_linear_unit_scale",
            lambda stage, arm, units, root: sdk.record("gripper_units", arm, units, root),
        )

        def camera_proxies(stage, paths, robots, root):
            sdk.record("camera_proxies", paths, robots, root)
            return paths, {}, {}

        def cameras(stage, paths, intrinsic, prefix, **kwargs):
            sdk.record("cameras", paths, prefix, kwargs)
            cams = {name: NS(get_clipping_range=lambda: (10.0, 3000.0)) for name in paths}
            return cams, {name: {} for name in paths}

        monkeypatch.setattr(replay, "_create_wrist_camera_proxies", camera_proxies)
        monkeypatch.setattr(replay, "_create_calibrated_cameras", cameras)
        self.backend, self.replay = backend, replay
        self.profile = profile

    def record(self, *event):
        self.events.append(normalized(event))
        return {}

    def renderer(self, stage, **kwargs):
        self.record("renderer", kwargs)

    def apply_drives(self, robot, arm, *args):
        return self.record("drives", arm, args)


TASKS = [(f"task{family}-{layers}", layers) for family in (1, 2, 4, 5, 6, 7) for layers in (2, 3)]


@pytest.mark.archive
@pytest.mark.parametrize("task,layers", TASKS)
@pytest.mark.parametrize("count", [1, 3])
def test_real_constructor_call_order(frozen_loader, monkeypatch, tmp_path, task, layers, count):
    """Both real constructors execute: only SDK/calibration I/O is stubbed."""
    model = importlib.import_module("tower_sim.config")
    stored = json.loads((ArchivedPath(f"scenes/{task}.json")).read_text())["scenes"][0]["candidate"]
    specs = [environment.native_spec(stored) for _ in range(count)]
    sdk = SDK(monkeypatch, count, len(specs[0].blocks))
    stage_file = tmp_path / "station.usd"
    stage_file.write_text("CPU SDK contract")
    config = model.SimConfig(
        task=task,
        layers_above_base=layers,
        stage=str(stage_file),
        render=True,
        camera_batch_mode="individual",
        physics_usd_writeback="on",
        reset_settle_seconds=0.0,
        envs_per_gpu=count,
        renderer="RealTimePathTracing",
        anti_aliasing=3,
    )
    profile = {"home_qpos": [0.0] * 16}
    monkeypatch.setattr(
        importlib.import_module("tower_sim.backend.profiles"), "load_profile", lambda path: profile
    )

    class Model:
        def __init__(self, source, cache):
            sdk.record("model", source, cache)

        def calibrate_tcp(self, profile):
            sdk.record("calibrate", profile)

    monkeypatch.setattr(sdk.backend, "LulaModel", Model)
    monkeypatch.setattr(sdk.backend, "apply_runtime_profile", sdk.apply_drives)
    monkeypatch.setattr(sdk.replay, "_configure_calibration_renderer", sdk.renderer)
    monkeypatch.syspath_prepend(str(ArchivedToolPath("")))
    frozen_hooks = frozen_loader("_p706_constructor_hooks", ArchivedToolPath("batch_backend.py"))
    frozen_worker = frozen_loader(
        "_p706_constructor_worker", ArchivedToolPath("batch_reference_worker.py")
    )
    plans = [{"env_path": f"/World/envs/env_{i}"} for i in range(count)]
    monkeypatch.setattr(frozen_hooks, "plans_for_candidates", lambda candidates: plans)
    monkeypatch.setattr(
        frozen_hooks,
        "author_heterogeneous_blocks",
        lambda *args: sdk.record("author_heterogeneous", task, count),
    )
    # Execute the actual nested frozen gravity wrapper without booting its app.
    import ast

    namespace = {
        "original_prepare": importlib.import_module(
            "tower_sim.backend.physics_scene"
        ).prepare_physics_scene
    }
    tree = ast.parse(Path(frozen_worker.__file__).read_text())
    prepare = next(
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "prepare"
    )
    exec(
        compile(ast.Module(body=[prepare], type_ignores=[]), "<frozen-prepare>", "exec"), namespace
    )

    def run(cls, **kwargs):
        class Env(cls):
            def reset(self, ids, specs):
                sdk.record("reset", ids, [spec.seed for spec in specs])

            def _validate_fk(self):
                sdk.record("validate_fk")

        application = NS(update=lambda: sdk.record("app_update"))
        env = Env(
            application, config, specs, np.zeros(16), tmp_path, instanceable_blocks=False, **kwargs
        )
        env.context.set_simulation_dt(physics_dt=1 / 400, rendering_dt=1 / 30)
        return copy.deepcopy(sdk.events)

    physics_scene = importlib.import_module("tower_sim.backend.physics_scene")
    original_prepare = physics_scene.prepare_physics_scene
    monkeypatch.setattr(physics_scene, "prepare_physics_scene", namespace["prepare"])
    with frozen_hooks.heterogeneous_clone_context(
        [stored] * count, cloner_class=sdk.Cloner, stage_getter=lambda: sdk.stage
    ):
        expected = run(sdk.backend.IsaacEnvironment)
    expected_usd = {name: list(mock.mock_calls) for name, mock in sdk.usd_mocks.items()}
    monkeypatch.setattr(physics_scene, "prepare_physics_scene", original_prepare)
    sdk.events.clear()
    for mock in sdk.usd_mocks.values():
        mock.reset_mock()
    hook = scene.CloneHook(
        [{"env_path": f"/World/envs/env_{i}"} for i in range(count)],
        None,
        author=lambda *args: sdk.record("author_heterogeneous", task, count),
    )
    actual = run(
        environment.environment_class(),
        model_factory=Model,
        renderer=sdk.renderer,
        prepare_scene=scene.prepare_physics_scene,
        apply_drives=sdk.apply_drives,
        clone_hook=hook,
        resources=NS(read=lambda name: {}, path=lambda name: Path(name)),
        robot_config={},
    )
    assert actual == expected
    assert {name: list(mock.mock_calls) for name, mock in sdk.usd_mocks.items()} == expected_usd
    assert any(
        "CreateDisableGravityAttr(True)" in str(call) for call in expected_usd["PhysxSchema"]
    )
    names = [row[0] for row in actual]
    assert names.index("model") < names.index("context") < names.index("clone")
    assert (
        names.index("clone")
        < names.index("author_heterogeneous")
        < names.index("initialize_physics")
    )
    assert names.index("renderer") < names.index("cameras") < names.index("initialize_physics")
    assert names.index("drives") < names.index("dt")
    assert sum(name == "app_update" for name in names) == 34
    assert next(row[1]["physics_dt"] for row in actual if row[0] == "context") == 1 / 120
    assert actual[-1] == ["dt", {"physics_dt": 1 / 400, "rendering_dt": 1 / 30}]


@pytest.mark.archive
def test_startup_experience_equal_and_fail_closed(frozen_loader, tmp_path):
    sdk = tmp_path / "sdk"
    (sdk / "apps").mkdir(parents=True)
    base = """[package]
version = "6.1.0"
[dependencies]
"isaacsim.robot.policy.examples" = {}
"omni.kit.viewport.legacy_gizmos" = {}
"isaacsim.gui.sensors.icon" = {}
"keep" = {}
[settings.app]
name = "Isaac-Sim Base"
[settings.app.window]
title = "Isaac Sim Base"
[settings.app.exts]
folders = ["${app}"]
"""
    (sdk / "apps/isaacsim.exp.base.kit").write_text(base)
    (sdk / "apps/isaacsim.exp.base.python.kit").write_text("""[settings.app]
name = "Isaac-Sim Python"
[settings.app.window]
title = "Isaac Sim Python"
""")
    stock = frozen_loader("_p706_stock", "support/render_startup.py")
    headless = frozen_loader("_p706_headless", "support/headless_render.py")
    output = tmp_path / "output"
    output.mkdir()
    expected = headless.omit_viewport_overlays(stock.local_stock_experience)(
        sdk, output
    ).read_bytes()
    receipt = (output / "startup-experience.json").read_bytes()
    assert (
        experience.omit_viewport_overlays(experience.local_stock_experience)(
            sdk, output
        ).read_bytes()
        == expected
    )
    assert (output / "startup-experience.json").read_bytes() == receipt
    (sdk / "apps/isaacsim.exp.base.kit").write_text(
        base.replace('version = "6.1.0"', 'version = "6.2.0"')
    )
    with pytest.raises(AssertionError):
        experience.local_stock_experience(sdk, output)


@pytest.mark.archive
@pytest.mark.parametrize(
    "task,layers,failure",
    [(task, layers, None) for task, layers in TASKS]
    + [
        ("task1-2", 2, failure)
        for failure in ("gravity", "mass", "initial", "fit_hash", "fit_status", "missing_stage")
    ],
)
def test_frozen_bootstrap_and_new_factory_sequence(
    frozen_loader, monkeypatch, tmp_path, task, layers, failure, frozen_import_dependencies
):
    """Run the frozen bootstrap and both real wrapper chains with shared doubles.

    The environment boundary is recorded here; test_real_constructor_call_order
    separately executes the entire actual constructor on both sides.
    """
    monkeypatch.syspath_prepend(str(RUNTIME / "support"))
    monkeypatch.syspath_prepend(str(ArchivedToolPath("")))
    # Frozen bootstrap intentionally changes sys.path and environment; isolate them.
    monkeypatch.setattr(sys, "path", list(sys.path))
    # Only the SDK double is used; test interpreter rejection separately.
    monkeypatch.setattr(bootstrap.startup, "sys", NS(version_info=(3, 12)))
    import os

    monkeypatch.setattr(os, "environ", os.environ.copy())
    original = frozen_loader("_p706_worker", ArchivedToolPath("batch_reference_worker.py"))
    importlib.import_module("task2_fixed.native61")
    grouped = importlib.import_module("task2_fixed.runtime")
    cli = importlib.import_module("task2_fixed.cli")
    eef = importlib.import_module("task2_fixed.eef_ik_movejs")
    support_startup = importlib.import_module("task2_fixed.startup")
    backend = importlib.import_module("tower_sim.backend.backend")
    physics_scene = importlib.import_module("tower_sim.backend.physics_scene")
    calibration = importlib.import_module("tower_eval.calibration")
    compensation = importlib.import_module("tower_sim.bench.j2comp")
    task_module = importlib.import_module("six_task_contract")
    events = []

    def record(*args):
        events.append(normalized(args))

    template = tmp_path / "template"
    kin = template / "kinematics"
    kin.mkdir(parents=True)
    for name in ("nero.urdf", "nero.yaml", "model.json"):
        (kin / name).write_text("test-only native boundary")
    stage = tmp_path / "station.usd"
    stage.write_text("test-only scene")
    profile = json.loads(
        (ROOT / "packages/tower-sim/src/tower_sim/data/motion_profile_v1.json").read_text()
    )
    profile_path = tmp_path / "profile.json"
    save(profile_path, profile)
    config = {
        "stage": str(stage),
        "profile_path": str(profile_path),
        "renderer": "RealTimePathTracing",
        "anti_aliasing": 3,
        "task6_neighbor_height_extra_m": 0.0,
        "task6_protrusion_m": 0.075,
    }
    save(template / "scene.json", {"config": config})
    settings = {
        "gripper_friction": {"static": 1.0, "dynamic": 1.0},
    }
    save(template / "settings.json", settings)
    robot_config = {
        "arms": [
            dict(
                joint_names=[f"joint{i}" for i in range(1, 8)],
                joint_indices=list(range(arm * 8, arm * 8 + 7)),
                gripper_index=arm * 8 + 7,
                gripper_targets=[
                    dict(name="gripper_joint1", multiplier=0.5),
                    dict(name="gripper_joint2", multiplier=-0.5),
                ],
            )
            for arm in range(2)
        ],
        "stage_units_per_meter": 1000,
    }
    save(template / "robot.json", robot_config)
    startup_file = tmp_path / "startup.json"
    startup_values = {
        "CUDA_VISIBLE_DEVICES": "3",
        "RSS_RUN_ROOT": str(tmp_path / "output"),
        "RSS_CAPTURE_WRIST": "left",
        "RSS_J2_ENABLED": "1",
        "OMNI_KIT_ACCEPT_EULA": "YES",
        "ACCEPT_EULA": "Y",
    }
    save(startup_file, startup_values)
    (tmp_path / "runtime-profiles").mkdir()
    fit = tmp_path / "fit.json"
    save(fit, {"status": "passed", "key": {"fit_source": {"fit_channel_sha256": "a" * 64}}})
    resources = Resources(
        dict(
            scene=stage,
            scene_config=template / "scene.json",
            settings=template / "settings.json",
            robot=template / "robot.json",
            kinematics=kin,
            motion_profile=profile_path,
            runtime_profiles=tmp_path / "runtime-profiles",
            scene_cache=stage,
            tcp_fit_receipt=fit,
            tcp_cache=fit,
            center_bond=fit,
            drive_profile=fit,
        )
    )
    stored = json.loads((ArchivedPath(f"scenes/{task}.json")).read_text())["scenes"][0]["candidate"]
    candidate_file = tmp_path / "candidate.json"
    save(candidate_file, stored)
    rows = [
        dict(
            candidate=stored,
            candidate_path=candidate_file,
            plan=tmp_path,
            env_id=0,
            portable_initial_state_only=True,
            times=np.array([0.0, 0.05]),
            actions=np.zeros((2, 16)),
            ticks=20,
            initial={"physical_qpos": [0.01] * 16, "qpos": [0.01] * 16},
        )
    ]
    save(tmp_path / "reference-initial-state.json", rows[0]["initial"])
    comp = NS(
        SCENE=stage,
        CONFIG={},
        command_targets=lambda *args: (np.zeros((20, 16)), np.zeros((20, 16))),
        physical_from_encoder16=lambda q: q.copy(),
        encoder_from_physical=lambda q, *args: q.copy(),
    )
    monkeypatch.setattr(compensation.J2Compensation, "from_cache", lambda *args, **kwargs: comp)

    class Robot:
        dof_names = [f"joint{i}" for i in range(1, 8)] + ["gripper_joint1", "gripper_joint2"]
        body_names = ["gripper_base", "gripper_link", "gripper_link1", "gripper_link2"]

        def __init__(self):
            self._physics_view = NS(get_dof_limits=lambda: np.tile([-3.0, 3.0], (1, 9, 1)))

        def get_body_disable_gravity(self):
            return np.array([[True, failure != "gravity", True, True]])

        def get_body_masses(self):
            values = np.ones((1, 4))
            values[0, 1] += 1e-3 if failure == "mass" else 0
            return values

        def set_joint_positions(self, target, **kwargs):
            record("joint_positions", target, kwargs)

        def set_joint_velocities(self, target, **kwargs):
            record("joint_velocities", target, kwargs)

        def set_joint_position_targets(self, target, **kwargs):
            record("joint_position_targets", target, kwargs)

        def set_joint_velocity_targets(self, target, **kwargs):
            record("joint_velocity_targets", target, kwargs)

    class Attribute:
        def Set(self, value):
            self.value = value
            record("friction", value)

        def Get(self):
            return getattr(self, "value", 1.0)

    material = NS(GetAttribute=lambda key: attrs.setdefault(key, Attribute()))
    attrs = {}
    stage_double = NS(GetPrimAtPath=lambda path: material)

    class Drives:
        def __init__(self, *args):
            pass

        def apply_usd(self, robot, name):
            record("apply_usd", name)
            return {"readback": {"kp": [[1.0] * 9]}, "delays_seconds": [0.01] * 7}

        def stage(self):
            return NS(GetPrimAtPath=lambda path: NS(GetAttribute=lambda key: NS(Get=lambda: 1.0)))

        def robot_root(self, name):
            return name

    drives = Drives()

    class Model:
        def calibrate_tcp(self, profile):
            pass

        def close(self):
            pass

    class Environment:
        def __init__(self, app, cfg, specs, home, cache, **kwargs):
            # Frozen dependencies are captured globals/closure; candidate dependencies
            # are explicit. Assert every candidate kwarg before the shared trace.
            if "model_factory" in kwargs:
                assert kwargs == dict(
                    instanceable_blocks=False,
                    coupled_gripper_fingers=False,
                    model_factory=Model,
                    renderer=renderer,
                    prepare_scene=scene.prepare_physics_scene,
                    apply_drives=drives.apply_usd,
                    clone_hook=clone_hook,
                    resources=resources,
                    robot_config=resources.read("robot"),
                )
                dependency_calls.append(dict(kwargs))
                public = {k: kwargs[k] for k in ("instanceable_blocks", "coupled_gripper_fingers")}
            else:
                public = kwargs
            # Compare every original config value, including v1-only inputs,
            # independently of how the split stores those inputs.
            from dataclasses import asdict

            from test_config_split import baseline_config

            effective = asdict(baseline_config()())
            effective.update(getattr(cfg, "_v1_generation_values", {}))
            effective.update(asdict(cfg))
            record("environment", effective, [s.seed for s in specs], home, public)
            self.robots = [Robot(), Robot()]
            self.stage, self.n = stage_double, len(specs)
            self.context = NS(
                set_simulation_dt=lambda **kw: record("dt", kw),
                physics_sim_view=NS(update_articulations_kinematic=lambda: record("kinematic")),
            )
            self.physics_output_report = {}
            self.previous = np.zeros((self.n, 16))

        def _commands(self, actions, arm):
            from tower_eval.simulator.commands import joint_commands

            return joint_commands(
                actions, robot_config["arms"][arm], self.robots[arm].dof_names, 1000
            )

        def read(self):
            qpos = self.previous[0].copy()
            qpos[0] += 1e-3 if failure == "initial" else 0
            return [
                NS(
                    qpos=qpos,
                    blocks=np.array([b["pose"] for b in stored["episode_spec"]["blocks"]]),
                )
            ]

    monkeypatch.setattr(backend, "IsaacEnvironment", Environment)
    monkeypatch.setattr(backend, "LulaModel", Model)
    monkeypatch.setattr(backend, "apply_runtime_profile", drives.apply_usd)
    monkeypatch.setattr(physics_scene, "prepare_physics_scene", lambda *a, **kw: {})
    monkeypatch.setattr(cli, "register_recording_task", lambda: None)
    monkeypatch.setattr(cli, "DEFAULT_SETTINGS", str(template / "settings.json"))
    monkeypatch.setattr(
        importlib.import_module("tower_sim.backend.profiles"), "load_profile", lambda p: profile
    )
    monkeypatch.setattr(grouped, "gpu_readback", lambda group: record("gpu", group))
    monkeypatch.setattr(eef, "reference_model_runtime", lambda: contextlib.nullcontext([]))
    monkeypatch.setattr(
        support_startup, "headless_experience", lambda sdk, output: output / "experience.kit"
    )
    monkeypatch.setattr(
        importlib.import_module("headless_render"), "omit_viewport_overlays", lambda fn: fn
    )
    replay = NS(startup=support_startup, j2comp=comp, placement_runtime=drives)
    monkeypatch.setattr(original.runtime, "capsule_module", lambda *args: replay)
    monkeypatch.setattr(original.runtime, "imported_sources", lambda: {})
    monkeypatch.setattr(
        original, "load_local", lambda name: NS(validate_physical_targets=lambda *args: None)
    )
    monkeypatch.setattr(
        original.backend_hooks,
        "heterogeneous_clone_context",
        lambda *args: contextlib.nullcontext({}),
    )
    monkeypatch.setattr(
        original.backend_hooks, "verify_physics_readback", lambda *args: record("physics_readback")
    )

    dependency_calls, tcp_calls = [], []
    renderer = object()
    clone_hook = NS(report={})

    @contextlib.contextmanager
    def cached(*args, **kwargs):
        record("cache_enter")
        yield
        record("cache_exit")

    monkeypatch.setattr(calibration, "cached_tcp_calibration", cached)

    @contextlib.contextmanager
    def cached_v2(model, path, *, key, kinematics):
        tcp_calls.append((model, path, key, kinematics))
        with cached():
            yield

    monkeypatch.setattr(calibration, "cached_tcp_calibration_v2", cached_v2)

    monkeypatch.setattr(bootstrap, "apply_v4", lambda env, out, path: record("v4"))
    monkeypatch.setattr(
        scene,
        "plans_for_candidates",
        lambda cs, validate: [{"task": validate(c["episode_spec"])} for c in cs],
    )

    def launched(options):
        record(
            "SimulationApp",
            options,
            {
                key: os.environ.get(key)
                for key in (
                    *startup_values,
                    "PYTHONDONTWRITEBYTECODE",
                    "OPENBLAS_NUM_THREADS",
                    "OMP_NUM_THREADS",
                    "MKL_NUM_THREADS",
                )
            },
        )
        assert "CUDA_VISIBLE_DEVICES" not in os.environ
        return NS(close=lambda *a, **kw: record("close"))

    sdk = tmp_path / "sdk"
    sdk.mkdir()
    install(monkeypatch, "isaacsim", SimulationApp=launched, __file__=str(sdk / "__init__.py"))
    install(
        monkeypatch,
        "omni.physx",
        get_physx_interface=lambda: NS(
            update_transformations=lambda *args: record("transformations", args)
        ),
    )
    raw_adapt = original.adapt_task12_config
    monkeypatch.setattr(
        original,
        "adapt_task12_config",
        lambda cfg, cls: raw_adapt(dict(cfg, render=True, camera_batch_mode="individual"), cls),
    )
    output = tmp_path / "output"
    args = NS(
        output=output,
        startup_environment=startup_file,
        capsule=tmp_path,
        live_tools=RUNTIME / "support",
        template_runtime=template,
        gpu=0,
    )
    errors = {
        "gravity": "Robot gravity compensation differs",
        "mass": "Robot mass readback differs",
        "initial": "Initial robot readback differs",
    }

    n1 = failure in {"fit_hash", "fit_status", "missing_stage"}
    errors.update(
        fit_hash="TCP fit receipt differs",
        fit_status="TCP fit inputs have not been verified",
        missing_stage="stage file does not exist",
    )

    def outcome(*, frozen=False):
        return (
            pytest.raises(ValueError, match=errors[failure])
            if failure and not (frozen and n1)
            else contextlib.nullcontext()
        )

    with outcome(frozen=True):
        old_app, _, _ = original.bootstrap(args, copy.deepcopy(rows))
        record("v4")
        old_app.close()
    frozen_initial = output / f"seed-{stored['seed']}" / "initial-state-readback.json"
    expected_initial = frozen_initial.read_bytes() if frozen_initial.exists() else None
    expected = copy.deepcopy(events)
    # Explicit startup differences: disable crash uploads and synchronize render
    # completion. Preserve the historical source and all scene/physics settings.
    for event in expected:
        if event[0] == "SimulationApp":
            event[1]["enable_crashreporter"] = False
            flags = event[1]["extra_args"]
            position = flags.index("--/crashreporter/skipOldDumpUpload=1")
            flags[position:position] = [
                "--/app/updateOrder/checkForHydraRenderComplete=1000",
                "--/app/renderer/waitIdle=true",
                "--/app/hydraEngine/waitIdle=true",
            ]
    import shutil

    shutil.rmtree(output)
    events.clear()
    factories = bootstrap.ConstructionFactories(
        app=launched,
        experience=lambda sdk, out: out / "experience.kit",
        renderer=lambda resources: renderer,
        model_type=lambda *args: Model,
        environment_class=lambda: Environment,
        profile=lambda path: profile,
        drives=lambda *args: drives,
        clone_hook=lambda *args: clone_hook,
        physics_readback=lambda *args: record("physics_readback"),
        gpu_readback=lambda group: record("gpu", group),
    )
    fit_digest = sha(fit)
    if n1:
        if failure == "fit_hash":
            fit_digest = "0" * 64
        elif failure == "fit_status":
            save(fit, {"status": "failed", "key": {"fit_source": {"fit_channel_sha256": "a" * 64}}})
            fit_digest = sha(fit)
        else:

            class MissingStageConfig(environment.config_class()):
                def __init__(self, **kwargs):
                    super().__init__(**kwargs)
                    self.stage = str(tmp_path / "missing.usd")

            factories.config_class = lambda: MissingStageConfig
            factories.native_spec = lambda candidate: pytest.fail("cfg.validate was bypassed")
    with outcome():
        result = bootstrap.bootstrap(
            resources=resources,
            rows=copy.deepcopy(rows),
            task=bootstrap.ConstructionTask(task, layers, task_module.validate_spec),
            output=output,
            gpu=0,
            sdk=sdk,
            fit_receipt_sha256=fit_digest,
            factories=factories,
            startup_environment=startup_values,
        )
        result.close()
    if n1:
        assert not tcp_calls
        assert not events
        return
    launch_record = json.loads((output / "launch.json").read_text())
    assert {
        "--/app/updateOrder/checkForHydraRenderComplete=1000",
        "--/app/renderer/waitIdle=true",
        "--/app/hydraEngine/waitIdle=true",
    } <= set(launch_record["extra_args"])
    assert (frozen_initial.read_bytes() if frozen_initial.exists() else None) == expected_initial
    assert len(dependency_calls) == 1
    assert tcp_calls == [
        (
            Model,
            fit,
            calibration.tcp_calibration_key(
                profile=profile,
                stage_sha256=sha(stage),
                kinematics=output / "kinematics",
                fit_channel_sha256="a" * 64,
            ),
            output / "kinematics",
        )
    ]
    assert events == expected
