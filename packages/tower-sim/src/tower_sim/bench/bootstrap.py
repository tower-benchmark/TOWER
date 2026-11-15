"""Explicit batch construction; no command loop, scorer or default entry point.

Task identity/topology validation is injected by the caller through the
canonical task adapter. The reference command queue remains outside this API.
"""

import shutil
import traceback
from contextlib import ExitStack
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import numpy as np

from tower_sim.bench.asset_identity import station_identity

from . import app as startup
from . import environment, scene
from .drives import CalibratedDrives, apply_v4, verify_v4
from .j2comp import J2Compensation
from .kinematics import EefModel
from .options import ControlOptions
from .resources import save, save_exclusive, sha


@dataclass(frozen=True)
class ConstructionTask:
    name: str
    layers: int
    validate_spec: Any

    def __post_init__(self):
        if self.layers != scene.task_layers(self.name):
            raise ValueError("ConstructionTask layer count differs from task identity")


def adapt_config(configuration, config_class, task):
    from tower_sim._config_v1 import defaults
    from tower_sim.config import SimConfig

    supported = (
        set(defaults())
        if config_class is SimConfig
        else {field.name for field in fields(config_class)}
    )
    unknown = set(configuration) - supported
    irrelevant = {"task6_neighbor_height_extra_m": 0.0, "task6_protrusion_m": 0.075}
    if configuration.get("task") != task.name or configuration.get(
        "layers_above_base"
    ) != scene.task_layers(task.name):
        raise ValueError("Config layer count differs from explicit task")
    if unknown - irrelevant.keys() or any(configuration[key] != irrelevant[key] for key in unknown):
        raise ValueError(
            "Unsupported scene settings require explicit compatibility review: "
            + str(sorted(unknown))
        )
    values = {key: value for key, value in configuration.items() if key in supported}
    config = SimConfig.from_v1(values) if config_class is SimConfig else config_class(**values)
    return config, {
        "omitted_unrelated_task6_defaults": {key: configuration[key] for key in sorted(unknown)},
        "relevant_settings_preserved": True,
    }


def model_type(session, resources):
    class CalibratedModel(EefModel):
        def __init__(self, source, cache):
            super().__init__(scene_urdf=Path(cache) / "nero.urdf")
            session.models.append(self)

        def calibrate_tcp(self, profile):
            raise RuntimeError("The semantic TCP context is required")

    return CalibratedModel


@dataclass
class ConstructionFactories:
    """Inject native boundaries for CPU contracts; production uses lazy defaults."""

    app: Any = startup.simulation_app
    experience: Any = None
    renderer: Any = lambda resources: startup.native_renderer(
        resources, environment.renderer_base()
    )
    model_type: Any = model_type
    environment_class: Any = environment.environment_class
    config_class: Any = environment.config_class
    native_spec: Any = environment.native_spec
    profile: Any = environment.load_profile
    drives: Any = CalibratedDrives
    clone_hook: Any = scene.CloneHook
    physics_readback: Any = scene.verify_physics_readback
    gpu_readback: Any = startup.gpu_readback
    synchronize: Any = None


class BenchContext:
    """Owned app, model and startup contexts; final readback remains explicit."""

    def __init__(self, stack, app, env, resources, compensation, delays, gains, output):
        self._stack, self.app, self.env = stack, app, env
        self.resources, self.compensation = resources, compensation
        self.delays, self.gains = np.asarray(delays), gains
        self.output = Path(output)
        self._drive_checks = 0
        self.public_provenance: dict[str, Any] = {}

    def verify_drives(self):
        verify_v4(self.env, self.gains)
        self._drive_checks += 1
        return dict(
            passed=True,
            all_environments=self.env.n,
            checks=self._drive_checks,
            checked_before_camera_teardown=True,
        )

    def physical_command_limits(self, index):
        """Live articulation bounds for the pre-command gate; grippers stay [0, .1]."""
        if not 0 <= index < self.env.n:
            raise IndexError("Environment index out of range")
        limits = np.empty((16, 2))
        for arm, robot in enumerate(self.env.robots):
            actual = np.asarray(robot._physics_view.get_dof_limits())[index]
            limits[arm * 8 : arm * 8 + 7] = actual[
                [robot.dof_names.index(f"joint{i}") for i in range(1, 8)]
            ]
            limits[arm * 8 + 7] = [0.0, 0.1]
        return limits

    def close(self):
        self._stack.close()

    def collect_sources(self, collector):
        """Perform the frozen pre-source readback before the caller collects evidence."""
        self.verify_drives()
        return collector()

    def finalize_drive_receipt(self):
        """Publish after successful cleanup; counted readbacks precede camera teardown."""
        if self._drive_checks == 0:
            raise RuntimeError("Cannot finalize drive receipt without successful checks")
        receipt = dict(
            passed=True,
            all_environments=self.env.n,
            checks=self._drive_checks,
            checked_before_camera_teardown=True,
        )
        save_exclusive(self.output / "v4-gain-final-readback.json", receipt)
        return receipt

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def synchronize(env):
    env.context.physics_sim_view.update_articulations_kinematic()
    import omni.physx  # pyright: ignore[reportMissingImports]

    omni.physx.get_physx_interface().update_transformations(False, True, True, False)


def bootstrap(
    *,
    resources,
    rows,
    task,
    output,
    gpu,
    sdk,
    fit_receipt_sha256,
    factories=None,
    startup_environment=None,
    shader_cache=None,
):
    """Construct once, in a dedicated process, with explicit data/SDK inputs.

    The caller supplies previously authorized, resolved candidates and a task
    topology validator. This does not generate or qualify grasp trajectories.
    """
    from tower_eval.calibration import cached_tcp_calibration_v2, tcp_calibration_key

    factories = factories or ConstructionFactories()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    candidates = [row["candidate"] for row in rows]
    plans = scene.plans_for_candidates(candidates, task.validate_spec)
    if {plan["task"] for plan in plans} != {task.name}:
        raise ValueError("Mixed task topologies in one simulator")
    for row in rows:
        row["output"] = output / f"seed-{row['candidate']['seed']}"
        row["output"].mkdir()
        save(row["output"] / "candidate.json", row["candidate"])
        save(row["output"] / "episode_spec.json", row["candidate"]["episode_spec"])
    save(
        output / "source-inspection.json",
        dict(source=str(output / "no-trajectory.npz"), trajectory_used=False),
    )
    compensation = J2Compensation.from_cache(
        resources.path("scene_cache"),
        resources.path("scene"),
        options=ControlOptions(exact_j2_cache=False),
    )
    template, settings = resources.read("scene_config"), resources.read("settings")
    configuration = dict(
        template["config"],
        task=task.name,
        layers_above_base=task.layers,
        stage=str(resources.path("scene")),
        profile_path=str(resources.path("motion_profile")),
        runtime_profile_dir=str(resources.path("runtime_profiles")),
        envs_per_gpu=len(rows),
        gpus=[gpu],
        output=str(output),
        reset_settle_seconds=0.0,
        render=True,
        camera_batch_mode="individual",
        physics_scene_mode="full",
        physics_usd_writeback="on",
    )
    cfg, compatibility = adapt_config(configuration, factories.config_class(), task)
    cfg.validate()
    save(output / "config-compatibility.json", compatibility)
    specs = [factories.native_spec(candidate) for candidate in candidates]
    cache = output / "kinematics"
    cache.mkdir()
    for name in ("nero.urdf", "nero.yaml", "model.json"):
        shutil.copy2(resources.path("kinematics") / name, cache / name)
    if sha(resources.path("tcp_fit_receipt")) != fit_receipt_sha256:
        raise ValueError("TCP fit receipt differs from the caller's verified identity")
    fit_receipt = resources.read("tcp_fit_receipt")
    if fit_receipt["status"] != "passed":
        raise ValueError("TCP fit inputs have not been verified")
    profile = factories.profile(resources.path("motion_profile"))
    key = tcp_calibration_key(
        profile=profile,
        stage_sha256=station_identity(sha(cfg.stage)),
        kinematics=cache,
        fit_channel_sha256=fit_receipt["key"]["fit_source"]["fit_channel_sha256"],
    )
    drives = factories.drives(resources.path("scene"), compensation.CONFIG)
    hook = factories.clone_hook(
        plans, resources.path("center_bond") if task.name == "task7-3" else None
    )
    stack = ExitStack()
    try:
        # Process settings are explicit; SDK EULA consent must be supplied by the caller.
        values = dict(startup_environment or {})
        values.update(
            PYTHONDONTWRITEBYTECODE="1",
            OPENBLAS_NUM_THREADS="1",
            OMP_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
        )
        stack.enter_context(startup.process_environment(values, unset=("CUDA_VISIBLE_DEVICES",)))
        cache_args = []
        if shader_cache is not None:
            receipt = stack.enter_context(startup.private_shader_cache(**shader_cache))
            save(output / "private-shader-cache.json", receipt)
            cache_args = receipt["kit_args"]
        native = stack.enter_context(
            startup.native_render_initialization(
                factories.app, lambda: factories.renderer(resources)
            )
        )
        model = []

        def setup(session):
            model.append(factories.model_type(session, resources))

        launch = stack.enter_context(
            startup.grouped_runtime(
                native,
                (gpu,),
                output,
                sdk=sdk,
                zero_delay=True,
                experience=factories.experience,
                model_setup=setup,
            )
        )
        app = launch(
            dict(
                headless=True,
                extra_args=cache_args,
                active_gpu=gpu,
                physics_gpu=gpu,
                multi_gpu=False,
                renderer=cfg.renderer,
                anti_aliasing=cfg.anti_aliasing,
                width=640,
                height=480,
                limit_cpu_threads=12,
            )
        )
        stack.callback(app.close)
        environment = factories.environment_class()
        # Patch only this factory-local model type, never EefModel or a backend global.
        with cached_tcp_calibration_v2(
            model[0], resources.path("tcp_cache"), key=key, kinematics=cache
        ):
            env = environment(
                app,
                cfg,
                specs,
                np.asarray(profile["home_qpos"]),
                cache,
                instanceable_blocks=False,
                coupled_gripper_fingers=settings.get("coupled_gripper_fingers", False),
                model_factory=model[0],
                renderer=app.renderer,
                prepare_scene=scene.prepare_physics_scene,
                apply_drives=drives.apply_usd,
                clone_hook=hook,
                resources=resources,
                robot_config=resources.read("robot"),
            )
        env.context.set_simulation_dt(physics_dt=1 / 400, rendering_dt=1 / 30)
        readback = factories.physics_readback(env, candidates, task.validate_spec)
        applied, delays = {}, []
        for arm, name in enumerate(("left", "back")):
            robot = env.robots[arm]
            applied[name] = drives.apply_usd(robot, name)
            if any(
                np.asarray(value).shape[0] != len(rows)
                for value in applied[name]["readback"].values()
            ):
                raise ValueError("Robot calibration readback did not cover every environment")
            if not np.asarray(robot.get_body_disable_gravity()).all():
                raise ValueError("Robot gravity compensation differs across environments")
            masses = np.asarray(robot.get_body_masses())
            checks = {}
            for link in ("gripper_base", "gripper_link", "gripper_link1", "gripper_link2"):
                expected_mass = (
                    drives.stage()
                    .GetPrimAtPath(drives.robot_root(name) + "/" + link)
                    .GetAttribute("physics:mass")
                    .Get()
                )
                measured = masses[:, robot.body_names.index(link)]
                if measured.shape != (len(rows),) or not np.allclose(
                    measured, expected_mass, rtol=1e-5
                ):
                    raise ValueError("Robot mass readback differs between environments")
                checks[link] = measured.tolist()
            applied[name]["masses_kg_all_environments"] = checks
            delays.append(applied[name]["delays_seconds"])
        friction = settings["gripper_friction"]
        material = env.stage.GetPrimAtPath("/__DualReplayGripperPhysicsMaterial")
        for name, expected in friction.items():
            attr = material.GetAttribute(f"physics:{name}Friction")
            attr.Set(float(expected))
            if not np.isclose(attr.Get(), expected, atol=1e-6, rtol=0):
                raise ValueError("Gripper friction readback failed")
        initialize_homes(env, rows, compensation)
        (factories.synchronize or synchronize)(env)
        verify_initial_readback(env, rows, compensation)
        save(output / "heterogeneous-construction.json", hook.report)
        save(output / "heterogeneous-readback.json", readback)
        save(
            output / "robot-runtime-readback.json",
            dict(
                arms=applied,
                gripper_friction=friction,
                gpu=factories.gpu_readback((gpu,)),
                # Frozen receipt: camera delivery is not yet qualified at bootstrap.
                render_enabled=False,
                camera_qualified=False,
                physics_hz=400,
                command_hz=200,
                physics_output=env.physics_output_report,
            ),
        )
        gains = apply_v4(env, output, resources.path("drive_profile"))
        return BenchContext(stack, app, env, resources, compensation, delays, gains, output)
    except BaseException:
        try:
            save(output / "bootstrap-error.json", {"error": traceback.format_exc()})
        finally:
            stack.close()
        raise


def initialize_homes(env, rows, compensation):
    expected = np.array([r["initial"]["physical_qpos"] for r in rows])
    encoder = np.array([r["initial"]["qpos"] for r in rows])
    converted = np.array([compensation.physical_from_encoder16(q) for q in encoder])
    if not np.allclose(converted, expected, atol=2e-6, rtol=0):
        raise ValueError("Source physical/encoder calibration differs")
    indices = np.arange(len(rows), dtype=np.int32)
    for arm, robot in enumerate(env.robots):
        names, target = env._commands(expected, arm)
        robot.set_joint_positions(target, indices=indices, joint_indices=names)
        robot.set_joint_velocities(np.zeros_like(target), indices=indices, joint_indices=names)
        robot.set_joint_position_targets(target, indices=indices, joint_indices=names)
        robot.set_joint_velocity_targets(
            np.zeros_like(target), indices=indices, joint_indices=names
        )
    env.previous[:] = expected
    return expected, encoder


def verify_initial_readback(env, rows, compensation):
    states = env.read()
    for index, (row, state) in enumerate(zip(rows, states, strict=True)):
        measured_encoder = state.qpos.copy()
        for arm, name in enumerate(("left", "back")):
            part = slice(arm * 8, arm * 8 + 7)
            measured_encoder[part] = compensation.encoder_from_physical(
                state.qpos[part], state.qpos[arm * 8 + 7], name
            )
        physical, encoder = row["initial"]["physical_qpos"], row["initial"]["qpos"]
        passed = bool(
            np.allclose(state.qpos, physical, atol=2e-6, rtol=0)
            and np.allclose(measured_encoder, encoder, atol=2e-6, rtol=0)
        )
        save(
            row["output"] / "initial-state-readback.json",
            dict(
                env_id=index,
                passed=passed,
                source_initial_state_sha256=sha(row["plan"] / "reference-initial-state.json"),
                commanded_encoder_qpos=encoder,
                commanded_physical_qpos=physical,
                measured_encoder_qpos=measured_encoder,
                measured_physical_qpos=state.qpos,
                tolerance_rad_m=2e-6,
                phase="immediately_after_reset_before_settling",
            ),
        )
        if not passed:
            raise ValueError(f"Initial robot readback differs in env{index}")
        wanted = np.array([b["pose"] for b in row["candidate"]["episode_spec"]["blocks"]])
        if not np.allclose(
            state.blocks[:, :3], wanted[:, :3], atol=1e-5, rtol=0
        ) or not np.allclose(
            np.abs(np.sum(state.blocks[:, 3:] * wanted[:, 3:], axis=1)), 1.0, atol=1e-5, rtol=0
        ):
            raise ValueError(f"Initial block poses leaked or differ in env{index}")
