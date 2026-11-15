"""Explicit bench environment construction using the shared replay library.

The model, renderer, scene, drives and clone factories are supplied before use.
Native imports remain after SimulationApp; module globals are never patched.
"""

from dataclasses import fields
from pathlib import Path

import numpy as np


def config_class():
    from tower_sim.config import SimConfig as Config

    return Config


def load_profile(path):
    from tower_sim.backend.profiles import load_profile as load

    return load(Path(path))


def native_spec(candidate):
    from tower_eval.simulator import spec as model

    source = candidate["episode_spec"]

    def construct(cls, value):
        names = {field.name for field in fields(cls)}
        return cls(**{key: value[key] for key in names if key in value})

    return model.EpisodeSpec(
        logical_id=source["logical_id"],
        seed=source["seed"],
        blocks=[construct(model.Block, b) for b in source["blocks"]],
        transfers=[construct(model.Transfer, t) for t in source["transfers"]],
        layout_metadata=source.get("layout_metadata", {}),
    )


def renderer_base():
    from tower_sim.replay.dual_replay_worker import _configure_calibration_renderer

    return _configure_calibration_renderer


def wrist_guard(*args, **kwargs):
    from tower_sim.backend.render_validation import check_wrist_near_plane

    return check_wrist_near_plane(*args, **kwargs)


def environment_class():
    # Called after SimulationApp. Inheritance captures the original backend
    # only here, and construction uses our explicit model and renderer inputs.
    from tower_eval.simulator import joint_commands

    from tower_sim.backend.assets import ROOT, checksum, normalize_reset_stacks
    from tower_sim.backend.backend import IsaacEnvironment, rigid_pose
    from tower_eval.simulator.spec import ARM_ORDER
    from tower_sim.backend.runtime_profiles import load_runtime_profiles

    class MappedEnvironment(IsaacEnvironment):
        def __init__(
            self,
            app,
            config,
            specs,
            home,
            cache,
            *,
            model_factory,
            renderer,
            prepare_scene,
            apply_drives,
            clone_hook,
            resources,
            robot_config,
            instanceable_blocks: bool = True,
            isolate_camera_passes: bool = False,
            coupled_gripper_fingers: bool = False,
            block_physics_material: dict | None = None,
        ):
            if config.camera_batch_mode != "individual" or isolate_camera_passes:
                raise ValueError("The bench bridge requires ordinary individual cameras")
            import omni.physx  # pyright: ignore[reportMissingImports]
            import omni.timeline  # pyright: ignore[reportMissingImports]
            import omni.usd  # pyright: ignore[reportMissingImports]
            from isaacsim.core.api import SimulationContext  # pyright: ignore[reportMissingImports]
            from isaacsim.core.cloner import Cloner  # pyright: ignore[reportMissingImports]
            from isaacsim.core.prims import (  # pyright: ignore[reportMissingImports]
                Articulation,
                RigidPrim,
            )
            from pxr import (  # pyright: ignore[reportMissingImports]
                Gf,
                PhysxSchema,
                Usd,
                UsdGeom,
                UsdPhysics,
                UsdShade,
            )
            from tower_sim.replay.dual_replay_worker import (
                CAMERA_PATHS,
                ROBOT_PATHS,
                _apply_gripper_linear_unit_scale,
                _create_calibrated_cameras,
                _create_wrist_camera_proxies,
            )

            if not isinstance(coupled_gripper_fingers, bool):
                raise ValueError("coupled_gripper_fingers must be boolean")

            self.app, self.config, self.n = app, config, len(specs)
            self.capture_count = 0
            import carb  # pyright: ignore[reportMissingImports]

            settings = carb.settings.get_settings()
            self._render_settings = settings
            settings.set("/rtx/post/aa/op", config.anti_aliasing)
            settings.set("/rtx-transient/dlssg/enabled", False)
            settings.set("/exts/isaacsim.core.throttling/enable_async", False)
            settings.set("/app/asyncRendering", False)
            self.home = np.asarray(home).reshape(16).copy()
            self.home[[7, 15]] = 0.1
            self.elapsed = np.zeros(self.n)
            self.recording = np.zeros(self.n, dtype=bool)
            self.offsets = np.array(
                [[8 * (i % 4), 8 * (i // 4), 0] for i in range(self.n)], dtype=float
            )
            self.report = resources.read("calibration_report")
            self.runtime_profile = load_runtime_profiles(
                config.runtime_profile_dir,
                checksum(Path(config.stage)),
                calibration_report=resources.path("calibration_report"),
            )
            self.calibration_provenance = self.runtime_profile["provenance"]
            self.calibration_provenance["generator_gripper_effort_application"] = {
                "source_profile_limit_n": float(
                    self.runtime_profile["gripper"]["finger_effort_limit"]
                ),
                "applied_limit_n": config.gripper_finger_effort_limit_n,
                "matches_replay_profile": bool(
                    np.isclose(
                        config.gripper_finger_effort_limit_n,
                        self.runtime_profile["gripper"]["finger_effort_limit"],
                    )
                ),
                "arm_pd_parameters_changed": False,
            }
            self.command_delays = np.array(
                [self.runtime_profile["arms"][arm]["command_delays_seconds"] for arm in ARM_ORDER]
            )
            self.lula = model_factory(config.stage, cache)
            from tower_sim.backend.profiles import load_profile

            self.lula.calibrate_tcp(load_profile(Path(config.profile_path)))
            self.kinematics = self.lula
            self.robot_config = robot_config
            omni.usd.get_context().new_stage()
            from tower_sim.backend.physics_output import PhysicsOutputPolicy

            self.physics_output = PhysicsOutputPolicy(
                settings, render=config.render, mode=config.physics_usd_writeback
            )
            stage = self.stage = omni.usd.get_context().get_stage()
            settings.set("/rtx/post/aa/op", config.anti_aliasing)
            settings.set("/rtx/rendermode", config.renderer)
            UsdGeom.SetStageMetersPerUnit(stage, 0.001)
            UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
            stage.SetTimeCodesPerSecond(120)
            world = UsdGeom.Xform.Define(stage, "/World")
            stage.SetDefaultPrim(world.GetPrim())
            env0 = "/World/envs/env_0"
            UsdGeom.Xform.Define(stage, env0)
            station = stage.DefinePrim(env0 + "/Station", "Xform")
            station.GetReferences().AddReference(config.stage, ROOT)
            material = stage.DefinePrim("/__DualReplayGripperPhysicsMaterial", "Material")
            material.GetReferences().AddReference(
                config.stage, "/__DualReplayGripperPhysicsMaterial"
            )
            disabled = [
                prim.GetPath()
                for prim in Usd.PrimRange(station)
                if prim.GetName() in ("ReplayTaskBlocks", "JengaBlocks")
            ]
            for path in disabled:
                stage.GetPrimAtPath(path).SetActive(False)
            normalize_reset_stacks(stage, station)
            gripper_limit_reports = {}
            for arm, path in ROBOT_PATHS.items():
                root_path = path.replace(ROOT, env0 + "/Station")
                gripper_limit_reports[arm] = _apply_gripper_linear_unit_scale(
                    stage, arm, 1000, root_path
                )
                if coupled_gripper_fingers:
                    first = stage.GetPrimAtPath(root_path + "/joints/gripper_joint1")
                    second = stage.GetPrimAtPath(root_path + "/joints/gripper_joint2")
                    if not all(joint.IsA(UsdPhysics.PrismaticJoint) for joint in (first, second)):
                        raise ValueError("Finger coupling requires two prismatic finger joints")
                    # Mechanical symmetry: q1 + q2 = 0. Both position-drive
                    # targets and their effort limits remain unchanged.
                    mimic = PhysxSchema.PhysxMimicJointAPI.Apply(second, UsdPhysics.Tokens.rotX)
                    mimic.CreateReferenceJointRel().SetTargets([first.GetPath()])
                    mimic.CreateGearingAttr(1.0)
                    mimic.CreateOffsetAttr(0.0)
                for finger in ("gripper_link1", "gripper_link2"):
                    prim = stage.GetPrimAtPath(root_path + "/" + finger)
                    UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                        UsdShade.Material(material),
                        UsdShade.Tokens.strongerThanDescendants,
                        "physics",
                    )
            self.calibration_provenance["gripper_limit_normalization"] = gripper_limit_reports
            self.calibration_provenance["coupled_gripper_fingers"] = coupled_gripper_fingers
            self.visual_reports = []
            self.block_material_report = []
            from tower_sim.backend.assets import block_reference

            canonical_asset, canonical_prim = block_reference(config.stage)
            canonical_dimensions = np.asarray([0.150, 0.050, 0.030], dtype=float)
            stage_inertia_scale = (1.0 / 0.001) ** 2  # kg*m^2 -> kg*stage_unit^2 (millimetre stage)
            for i, block in enumerate(specs[0].blocks):
                prim = stage.DefinePrim(f"{env0}/Blocks/block_{i:02}", "Xform")
                asset_reference = getattr(block, "asset_reference", None) or canonical_asset
                asset_prim = "" if getattr(block, "asset_reference", None) else canonical_prim
                dimensions_value = getattr(block, "dimensions_m", None)
                dimensions = np.asarray(
                    dimensions_value if dimensions_value is not None else canonical_dimensions,
                    dtype=float,
                )
                scale_value = getattr(block, "asset_scale", None)
                scale = np.asarray(
                    scale_value
                    if scale_value is not None
                    else 1000.0 * dimensions / canonical_dimensions,
                    dtype=float,
                )
                mass_value = getattr(block, "mass_kg", None)
                mass = float(mass_value if mass_value is not None else 0.1056)
                diagonal_inertia = getattr(block, "diagonal_inertia_kg_m2", None)
                prim.GetReferences().AddReference(asset_reference, asset_prim)
                if not any(
                    child.HasAPI(UsdPhysics.CollisionAPI)
                    for child in Usd.PrimRange(prim, Usd.TraverseInstanceProxies())
                ):
                    raise RuntimeError(
                        f"Block reference has no collision geometry: {asset_reference}:{asset_prim}"
                    )
                prim.SetInstanceable(instanceable_blocks)
                if block_physics_material is not None:
                    from tower_sim.backend.material_contract import bind_block_material

                    self.block_material_report.extend(
                        bind_block_material(stage, prim, block_physics_material)
                    )
                xform = UsdGeom.Xformable(prim)
                xform.ClearXformOpOrder()
                xform.AddTranslateOp().Set(Gf.Vec3d(*(np.asarray(block.pose[:3]) * 1000)))
                xform.AddOrientOp().Set(Gf.Quatf(block.pose[6], Gf.Vec3f(*block.pose[3:6])))
                xform.AddScaleOp().Set(Gf.Vec3f(*scale))
                # Foundations participate in collision response but are anchored
                # to the tabletop. This mirrors the replay scene's fixed base
                # bodies and prevents contact impulses from propagating through
                # the tower during a pickup.
                fixed_on_table = bool(getattr(block, "fixed_on_table", False))
                UsdPhysics.RigidBodyAPI(prim).CreateKinematicEnabledAttr(fixed_on_table)
                mass_api = UsdPhysics.MassAPI(prim)
                mass_api.CreateMassAttr(mass)
                if diagonal_inertia is not None:
                    mass_api.CreateDiagonalInertiaAttr(
                        Gf.Vec3f(*(np.asarray(diagonal_inertia, dtype=float) * stage_inertia_scale))
                    )
                    mass_api.CreateCenterOfMassAttr(Gf.Vec3f(0, 0, 0))
                body = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
                body.CreateSolverPositionIterationCountAttr(16)
                body.CreateSolverVelocityIterationCountAttr(4)
                body.CreateMaxDepenetrationVelocityAttr(1000)
                self.visual_reports.append(
                    {
                        "mode": "source"
                        if asset_reference.startswith(config.stage)
                        else "resolved",
                        "instanceable": instanceable_blocks,
                        "asset": asset_reference,
                        "asset_prim": asset_prim,
                        "scale": scale.tolist(),
                        "mass_kg": mass,
                        "diagonal_inertia_kg_m2": None
                        if diagonal_inertia is None
                        else list(diagonal_inertia),
                        "fixed_on_table": fixed_on_table,
                        "material_overrides": bool(
                            asset_reference.startswith("/")
                            and not asset_reference.startswith(config.stage)
                        ),
                        "geometry_overrides": dimensions_value is not None,
                    }
                )

            self.context = SimulationContext(
                physics_dt=1 / 120,
                rendering_dt=1 / 30,
                stage_units_in_meters=0.001,
                backend="numpy",
            )
            from tower_sim.backend.grasp_geometry import inspect_fingers

            self.finger_geometry = [
                inspect_fingers(stage, path.replace(ROOT, env0 + "/Station"))
                for path in ROBOT_PATHS.values()
            ]
            pc = self.context.get_physics_context()
            pc.enable_gpu_dynamics(True)
            pc.set_broadphase_type("GPU")
            pc.set_solver_type("TGS")
            from tower_sim.backend.physics_capacity import configure_capacity

            self.physics_capacity_report = configure_capacity(
                pc,
                config.gpu_total_aggregate_pairs_capacity,
                config.gpu_found_lost_aggregate_pairs_capacity,
            )
            from tower_sim.backend.physics_scene import prepare_physics_scene

            self.physics_scene_report = prepare_scene(
                prepare_physics_scene,
                stage,
                env0,
                render=config.render,
                mode=config.physics_scene_mode,
            )
            self.env_paths = [f"/World/envs/env_{i}" for i in range(self.n)]
            cloner = clone_hook.factory(Cloner, stage)
            cloner.define_base_env("/World/envs")
            # Inherited environment roots can lose these instanced block visuals.
            # Independent root copies retain shared asset references and GPU physics replication.
            cloner.clone(
                env0,
                self.env_paths,
                positions=self.offsets * 1000,
                replicate_physics=True,
                base_env_path="/World/envs",
                root_path="/World/envs/env_",
                enable_env_ids=True,
                copy_from_source=True,
            )
            clone_hook.verify()
            cloner.filter_collisions(pc.prim_path, "/World/collisions", self.env_paths, [])
            if config.render:
                renderer(stage, anti_aliasing=config.anti_aliasing, dlss_mode=config.dlss_mode)
            from tower_sim.backend.render_validation import SAMPLING_KEYS

            if config.lighting_samples is not None:
                for key in SAMPLING_KEYS:
                    settings.set(key, config.lighting_samples)
            for _ in range(30):
                app.update()
            self.robot_paths = [
                {arm: path.replace(ROOT, env + "/Station") for arm, path in ROBOT_PATHS.items()}
                for env in self.env_paths
            ]
            self.cameras, self.mounts, self.camera_report = [], [], []
            self.tiled_capture = None
            self.camera_passes = None
            if config.render:
                for env, robots in zip(self.env_paths, self.robot_paths):
                    camera_paths = {
                        name: path.replace(ROOT, env + "/Station")
                        for name, path in CAMERA_PATHS.items()
                    }
                    paths, intrinsic, mounts = _create_wrist_camera_proxies(
                        stage, camera_paths, robots, env + "/Cameras"
                    )
                    for name, path in paths.items():
                        near = config.wrist_near_clip_mm if name.endswith("_wrist") else 1
                        UsdGeom.Camera(stage.GetPrimAtPath(path)).GetClippingRangeAttr().Set(
                            Gf.Vec2f(near, 3000)
                        )
                    cams, report = _create_calibrated_cameras(
                        stage,
                        paths,
                        intrinsic,
                        env.replace("/", "_") + "_",
                        initialize_render_products=config.camera_batch_mode == "individual",
                    )
                    self.cameras.append(cams)
                    self.mounts.append(mounts)
                    self.camera_report.append(report)
                    for name, camera in cams.items():
                        report[name]["clipping_range_mm"] = list(camera.get_clipping_range())
            self.timeline = omni.timeline.get_timeline_interface()
            self.context.initialize_physics()
            self.timeline.play()
            self.timeline.commit()
            for _ in range(3):
                app.update()
            self.timeline.pause()
            self.timeline.commit()
            app.update()
            self.physics = omni.physx.get_physx_simulation_interface()
            from isaacsim.core.simulation_manager import (  # pyright: ignore[reportMissingImports]
                SimulationManager,
            )

            # Keep PhysX/Fabric sample times monotonic across Kit's startup steps.
            # Per-episode clocks are separate and reset only in self.elapsed.
            self.simulation_time = float(SimulationManager.get_simulation_time())
            self.robots = []
            self.calibration_provenance["runtime_readback"] = {}
            for arm in ARM_ORDER:
                paths = [robots[arm] + "/root_joint" for robots in self.robot_paths]
                robot = Articulation(paths, name=arm, reset_xform_properties=False)
                robot.initialize()
                if list(robot.prim_paths) != paths:
                    raise RuntimeError(
                        f"{arm} articulation view order does not match logical environment order"
                    )
                self.robots.append(robot)
                self.calibration_provenance["runtime_readback"][arm] = apply_drives(
                    robot, arm, self.runtime_profile, config.gripper_finger_effort_limit_n
                )
            # Contact sensor preparation also changes sleep thresholds. Keep the
            # source dynamics while reading poses directly from the physics view.
            self.block_view = RigidPrim(
                "/World/envs/env_.*/Blocks/block_.*",
                name="blocks",
                reset_xform_properties=False,
                track_contact_forces=False,
                prepare_contact_sensors=False,
            )
            self.block_view.initialize()
            self.block_count = len(specs[0].blocks)
            self.block_paths = list(self.block_view.prim_paths)
            self.block_indices = np.array(
                [
                    [
                        self.block_paths.index(f"{env}/Blocks/block_{b:02}")
                        for b in range(self.block_count)
                    ]
                    for env in self.env_paths
                ]
            )
            self.fixed_block_mask = np.asarray(
                [bool(getattr(block, "fixed_on_table", False)) for block in specs[0].blocks],
                dtype=bool,
            )
            cache_usd = UsdGeom.XformCache()
            self.bases = np.array(
                [
                    [
                        rigid_pose(
                            cache_usd.GetLocalToWorldTransform(
                                stage.GetPrimAtPath(robots[arm] + "/base_link")
                            )
                        )
                        for arm in ARM_ORDER
                    ]
                    for robots in self.robot_paths
                ]
            )
            self.previous = np.tile(self.home, (self.n, 1))
            self.reset(list(range(self.n)), specs)
            for _ in range(round(self.config.reset_settle_seconds * 30)):
                self.step(self.previous)
            self.elapsed[:] = 0
            # Kit/Replicator can leave the global timeline at a nonzero sample
            # time after camera warm-up.  Episode physics must start from a fresh
            # local clock so rendered and no-render runs integrate identically.
            self.simulation_time = 0.0
            self._validate_fk()
            self.render_report = {
                "schema_version": "jenga_render_capture_v3",
                "capture_profile": "single_pass_v3"
                if config.camera_batch_mode == "individual" and not config.replicator_sync_only
                else "experimental",
                "visibility_passes": 1,
                "wrist_self_mask": False,
                "wrist_near_clip_mm": config.wrist_near_clip_mm,
                "near_plane_block_guard": True,
                "anti_aliasing": "DLAA" if config.anti_aliasing == 4 else "DLSS",
                "dlss_mode": config.dlss_mode,
                "camera_batch_mode": config.camera_batch_mode,
                "replicator_sync_only": config.replicator_sync_only,
                "render_subframes": config.render_subframes,
                "camera_resolution": [640, 480],
                "jpeg_quality": config.jpeg_quality,
                "settings": {
                    key: settings.get(key)
                    for key in (
                        "/rtx/rendermode",
                        "/rtx/post/aa/op",
                        "/rtx/post/dlss/execMode",
                        "/rtx/post/motionblur/enabled",
                        "/rtx-transient/dlssg/enabled",
                        "/app/useFabricSceneDelegate",
                        *SAMPLING_KEYS,
                    )
                },
            }
            self.physics_output.finish_initialization(
                camera_count=sum(len(batch) for batch in self.cameras)
            )
            self.physics_output_report = self.physics_output.report

        def _commands(self, actions, arm):
            return joint_commands(
                actions,
                self.robot_config["arms"][arm],
                self.robots[arm].dof_names,
                self.robot_config["stage_units_per_meter"],
            )

    return MappedEnvironment
