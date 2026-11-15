"""Isaac Sim adapter. Object and robot state setters exist only in reset()."""

from __future__ import annotations

from pathlib import Path
import json
import time

import numpy as np
from scipy.spatial.transform import Rotation

from .assets import ROOT, calibration, checksum, export_kinematics, normalize_reset_stacks
from tower_eval.simulator.spec import ARM_ORDER, CAMERAS, EpisodeSpec, State
from tower_sim.config import SimConfig as Config
from .runtime_profiles import apply_runtime_profile, delayed_actions, load_runtime_profiles


def rigid_pose(matrix) -> np.ndarray:
    from pxr import Gf
    transform = Gf.Transform(matrix)
    quat = transform.GetRotation().GetQuat()
    return np.r_[np.asarray(transform.GetTranslation()) * 0.001, quat.GetImaginary(), quat.GetReal()]


class LulaModel:
    def __init__(self, source: str, cache: Path):
        import lula
        import fcntl
        import xml.etree.ElementTree as ET
        cache.mkdir(parents=True, exist_ok=True)
        urdf, yaml = cache / "nero.urdf", cache / "nero.yaml"
        with (cache / "export.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not urdf.exists() or not yaml.exists() or not (cache / "model.json").exists():
                urdf, yaml = export_kinematics(source, cache)
        tree = ET.parse(urdf)
        limits = {joint.get("name"): joint.find("limit") for joint in tree.getroot().findall("joint") if joint.get("type") == "revolute"}
        self.lower = np.array([float(limits[f"joint{i}"].get("lower")) for i in range(1, 8)])
        self.upper = np.array([float(limits[f"joint{i}"].get("upper")) for i in range(1, 8)])
        self.lula = lula
        self.robot = lula.load_robot(str(yaml), str(urdf))
        self.kinematics = self.robot.kinematics()
        self.config = lula.CyclicCoordDescentIkConfig()
        self.config.position_tolerance = 0.0003
        self.config.orientation_tolerance = 0.005
        self.config.max_num_descents = 5
        self.config.bfgs_max_iterations = 100
        self.frame = "gripper_base"
        self.tcp_offset = np.zeros(3)
        self.tcp_rotation = Rotation.identity()
        self.tcp_report = {}
        # Keep a small set of recent valid joint solutions as deterministic
        # CCD seeds.  Returning to a previous workspace region (for example
        # from the destination corner to the next source block) can otherwise
        # put CCD in a different basin even though the pose is reachable.
        self._successful_seeds: list[np.ndarray] = []

    def calibrate_tcp(self, profile: dict) -> None:
        import h5py
        offsets, rotations = [], []
        source = next(s for s in profile["sources"] if s["split"] == "fit")
        with h5py.File(source["path"]) as handle:
            qpos = handle["observations/qpos"][::15]
            measured = handle["observations/eef"][::15].reshape(-1, 2, 7)
        for row, eef in zip(qpos, measured):
            for arm in range(2):
                raw = self.fk(row[arm * 8:arm * 8 + 7])
                rotation = Rotation.from_quat(raw[3:])
                offsets.append(rotation.inv().apply(eef[arm, :3] - raw[:3]))
                rotations.append((rotation.inv() * Rotation.from_quat(eef[arm, 3:])).as_quat())
        self.tcp_offset = np.median(offsets, axis=0)
        self.tcp_rotation = Rotation.from_quat(rotations).mean()
        position_error = np.linalg.norm(np.asarray(offsets) - self.tcp_offset, axis=1)
        angle_error = (Rotation.from_quat(rotations) * self.tcp_rotation.inv()).magnitude()
        self.tcp_report = {"fit_episode_id": source["episode_id"], "source_channel_sha256": source["channel_sha256"],
                           "translation_m": self.tcp_offset.tolist(), "rotation_xyzw": self.tcp_rotation.as_quat().tolist(),
                           "position_residual_p50_p95_m": np.percentile(position_error, [50, 95]).tolist(),
                           "rotation_residual_p50_p95_deg": np.rad2deg(np.percentile(angle_error, [50, 95])).tolist()}
        print("TCP_CALIBRATION", json.dumps(self.tcp_report), flush=True)
        if np.percentile(position_error, 95) > 0.005 or np.percentile(angle_error, 95) > np.deg2rad(3):
            raise RuntimeError("real EEF cannot be represented by a verified fixed TCP transform")

    def fk(self, q: np.ndarray) -> np.ndarray:
        pose = self.kinematics.pose(q[:7].astype(float), self.frame)
        rotation = np.asarray(pose.rotation.matrix())
        return np.r_[np.asarray(pose.translation).reshape(3) + rotation @ self.tcp_offset, (Rotation.from_matrix(rotation) * self.tcp_rotation).as_quat()]

    def ik(self, target: np.ndarray, seed: np.ndarray) -> tuple[np.ndarray, bool]:
        rotation = (Rotation.from_quat(target[3:]) * self.tcp_rotation.inv()).as_matrix()
        position = target[:3] - rotation @ self.tcp_offset
        desired = self.lula.Pose3(self.lula.Rotation3(rotation), position)
        seed = np.asarray(seed[:7], dtype=float)

        # Keep the normal path exactly as before.  A moving Cartesian spline can
        # occasionally land on a CCD basin boundary even when the endpoint is
        # reachable; deterministic nearby seeds recover that sample without
        # changing the commanded pose or writing a robot state directly.
        candidates = [seed]
        # The 7-DOF arm has an equivalent elbow branch that primarily differs
        # in joint 1.  The old seed family perturbed only joints 2/4/6, so a
        # corner-side transport could remain in the opposite branch even
        # though a nearby real-replay posture was reachable.  That produced
        # the visible lift-then-bow path (joint 1 swept to the opposite side)
        # during the first central-to-corner transfer.  Probe both bounded
        # joint-1 branches as IK seeds; the existing nearest-solution choice
        # keeps the solution continuous and does not alter the calibrated PD
        # gains or any simulator pose.
        for value in (-1.2, 1.2):
            branch_seed = seed.copy()
            branch_seed[0] = np.clip(value, self.lower[0], self.upper[0])
            candidates.append(branch_seed)
        for direction in (1.0, -1.0):
            candidates.append(np.clip(
                seed + direction * np.array([0.0, 0.35, 0.0, 0.35, 0.0, 0.35, 0.0]),
                self.lower, self.upper))
        midpoint = np.clip(0.5 * (self.lower + self.upper), self.lower, self.upper)
        candidates.append(midpoint)
        # Rendering and JPEG publication can leave the articulation a little
        # farther from the previous CCD basin than a no-render probe.  Try a
        # small deterministic family of elbow/wrist branches before declaring
        # a waypoint unreachable; these are seed choices only, never pose or
        # joint-state writes.
        for signs in ((1.0, 1.0, 1.0), (1.0, -1.0, -1.0),
                      (-1.0, 1.0, -1.0), (-1.0, -1.0, 1.0)):
            branch = 0.65 * np.array([0.0, signs[0], 0.0, signs[1], 0.0, signs[2], 0.0])
            candidates.append(np.clip(midpoint + branch, self.lower, self.upper))
        # Cached solutions are only consulted after the inexpensive local
        # seeds above fail.  They are joint seeds, never pose overwrites.
        cached = [value for value in reversed(self._successful_seeds)
                  if np.linalg.norm(value - seed) > 1e-4]

        original_descents = self.config.max_num_descents
        original_iterations = self.config.bfgs_max_iterations
        solutions = []
        try:
            for index, candidate in enumerate(candidates):
                self.config.cspace_seeds = [candidate]
                if index:
                    self.config.max_num_descents = max(original_descents, 12)
                    self.config.bfgs_max_iterations = max(original_iterations, 150)
                result = self.lula.compute_ik_ccd(self.kinematics, desired, self.frame, self.config)
                if bool(result.success):
                    joints = np.asarray(result.cspace_position, dtype=float)
                    solutions.append(joints)
                    if index == 0 and np.linalg.norm(joints - seed) < 0.25:
                        # The common case is already continuous. Avoid
                        # paying for all branch probes at every 30 Hz tick;
                        # only investigate alternatives after the primary
                        # CCD result jumps a substantial amount.
                        self._successful_seeds.append(joints.copy())
                        self._successful_seeds = self._successful_seeds[-24:]
                        return joints, True
            if not solutions:
                for candidate in cached[:12]:
                    self.config.cspace_seeds = [candidate]
                    self.config.max_num_descents = max(original_descents, 16)
                    self.config.bfgs_max_iterations = max(original_iterations, 200)
                    result = self.lula.compute_ik_ccd(self.kinematics, desired, self.frame, self.config)
                    if bool(result.success):
                        solutions.append(np.asarray(result.cspace_position, dtype=float))
        finally:
            self.config.max_num_descents = original_descents
            self.config.bfgs_max_iterations = original_iterations
        if solutions:
            # Prefer the solution closest to the previous command even when
            # the primary CCD seed succeeds. Returning the first valid branch
            # lets a moving Cartesian spline hop between equivalent elbow and
            # wrist configurations, which is visible as arm shake at contact.
            selected = min(solutions, key=lambda joints: float(np.linalg.norm(joints - seed)))
            self._successful_seeds.append(selected.copy())
            self._successful_seeds = self._successful_seeds[-24:]
            return selected, True
        # Preserve Lula's last result shape for callers that record a failed
        # sample, while ensuring the configuration is ready for the next tick.
        self.config.cspace_seeds = [seed]
        result = self.lula.compute_ik_ccd(self.kinematics, desired, self.frame, self.config)
        return np.asarray(result.cspace_position), bool(result.success)


class IsaacEnvironment:
    def __init__(self, app, config: Config, specs: list[EpisodeSpec], home: np.ndarray, cache: Path, *, instanceable_blocks: bool = True, isolate_camera_passes: bool = False, coupled_gripper_fingers: bool = False, block_physics_material: dict | None = None):
        import omni.usd
        import omni.timeline
        import omni.physx
        from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade, PhysxSchema
        from isaacsim.core.api import SimulationContext
        from isaacsim.core.cloner import Cloner
        from isaacsim.core.prims import Articulation, RigidPrim
        from tower_sim.replay.dual_replay_worker import ROBOT_PATHS, CAMERA_PATHS, _apply_gripper_linear_unit_scale, _create_wrist_camera_proxies, _create_calibrated_cameras, _configure_calibration_renderer

        if not isinstance(coupled_gripper_fingers, bool):
            raise ValueError('coupled_gripper_fingers must be boolean')

        self.app, self.config, self.n = app, config, len(specs)
        self.capture_count = 0
        import carb
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
        self.offsets = np.array([[8 * (i % 4), 8 * (i // 4), 0] for i in range(self.n)], dtype=float)
        self.report, _ = calibration()
        self.runtime_profile = load_runtime_profiles(config.runtime_profile_dir, checksum(Path(config.stage)))
        self.calibration_provenance = self.runtime_profile["provenance"]
        self.calibration_provenance["generator_gripper_effort_application"] = {
            "source_profile_limit_n": float(
                self.runtime_profile["gripper"]["finger_effort_limit"]),
            "applied_limit_n": config.gripper_finger_effort_limit_n,
            "matches_replay_profile": bool(np.isclose(
                config.gripper_finger_effort_limit_n,
                self.runtime_profile["gripper"]["finger_effort_limit"])),
            "arm_pd_parameters_changed": False,
        }
        self.command_delays = np.array([self.runtime_profile["arms"][arm]["command_delays_seconds"] for arm in ARM_ORDER])
        self.lula = LulaModel(config.stage, cache)
        from .profiles import load_profile
        self.lula.calibrate_tcp(load_profile(Path(config.profile_path)))
        omni.usd.get_context().new_stage()
        from .physics_output import PhysicsOutputPolicy
        self.physics_output = PhysicsOutputPolicy(settings, render=config.render, mode=config.physics_usd_writeback)
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
        material.GetReferences().AddReference(config.stage, "/__DualReplayGripperPhysicsMaterial")
        disabled = [prim.GetPath() for prim in Usd.PrimRange(station) if prim.GetName() in ("ReplayTaskBlocks", "JengaBlocks")]
        for path in disabled:
            stage.GetPrimAtPath(path).SetActive(False)
        normalize_reset_stacks(stage, station)
        gripper_limit_reports = {}
        for arm, path in ROBOT_PATHS.items():
            root_path = path.replace(ROOT, env0 + "/Station")
            gripper_limit_reports[arm] = _apply_gripper_linear_unit_scale(stage, arm, 1000, root_path)
            if coupled_gripper_fingers:
                first = stage.GetPrimAtPath(root_path + '/joints/gripper_joint1')
                second = stage.GetPrimAtPath(root_path + '/joints/gripper_joint2')
                if not all(joint.IsA(UsdPhysics.PrismaticJoint) for joint in (first, second)):
                    raise ValueError('Finger coupling requires two prismatic finger joints')
                # Mechanical symmetry: q1 + q2 = 0. Both position-drive
                # targets and their effort limits remain unchanged.
                mimic = PhysxSchema.PhysxMimicJointAPI.Apply(second, UsdPhysics.Tokens.rotX)
                mimic.CreateReferenceJointRel().SetTargets([first.GetPath()])
                mimic.CreateGearingAttr(1.0)
                mimic.CreateOffsetAttr(0.0)
            for finger in ("gripper_link1", "gripper_link2"):
                prim = stage.GetPrimAtPath(root_path + "/" + finger)
                UsdShade.MaterialBindingAPI.Apply(prim).Bind(UsdShade.Material(material), UsdShade.Tokens.strongerThanDescendants, "physics")
        self.calibration_provenance["gripper_limit_normalization"] = gripper_limit_reports
        self.calibration_provenance['coupled_gripper_fingers'] = coupled_gripper_fingers
        self.visual_reports = []
        self.block_material_report = []
        from .assets import block_reference
        canonical_asset, canonical_prim = block_reference(config.stage)
        canonical_dimensions = np.asarray([0.150, 0.050, 0.030], dtype=float)
        stage_inertia_scale = (1.0 / 0.001) ** 2  # kg*m^2 -> kg*stage_unit^2 (millimetre stage)
        for i, block in enumerate(specs[0].blocks):
            prim = stage.DefinePrim(f"{env0}/Blocks/block_{i:02}", "Xform")
            asset_reference = getattr(block, "asset_reference", None) or canonical_asset
            asset_prim = "" if getattr(block, "asset_reference", None) else canonical_prim
            dimensions_value = getattr(block, "dimensions_m", None)
            dimensions = np.asarray(dimensions_value if dimensions_value is not None else canonical_dimensions, dtype=float)
            scale_value = getattr(block, "asset_scale", None)
            scale = np.asarray(scale_value if scale_value is not None else 1000.0 * dimensions / canonical_dimensions, dtype=float)
            mass_value = getattr(block, "mass_kg", None)
            mass = float(mass_value if mass_value is not None else 0.1056)
            diagonal_inertia = getattr(block, "diagonal_inertia_kg_m2", None)
            prim.GetReferences().AddReference(asset_reference, asset_prim)
            if not any(child.HasAPI(UsdPhysics.CollisionAPI) for child in Usd.PrimRange(prim, Usd.TraverseInstanceProxies())):
                raise RuntimeError(f"Block reference has no collision geometry: {asset_reference}:{asset_prim}")
            prim.SetInstanceable(instanceable_blocks)
            if block_physics_material is not None:
                from .material_contract import bind_block_material
                self.block_material_report.extend(bind_block_material(stage, prim, block_physics_material))
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
            self.visual_reports.append({"mode": "source" if asset_reference.startswith(config.stage) else "resolved",
                                        "instanceable": instanceable_blocks, "asset": asset_reference, "asset_prim": asset_prim,
                                        "scale": scale.tolist(), "mass_kg": mass,
                                        "diagonal_inertia_kg_m2": None if diagonal_inertia is None else list(diagonal_inertia),
                                        "fixed_on_table": fixed_on_table,
                                        "material_overrides": bool(asset_reference.startswith("/") and not asset_reference.startswith(config.stage)),
                                        "geometry_overrides": dimensions_value is not None})

        self.context = SimulationContext(physics_dt=1 / 120, rendering_dt=1 / 30, stage_units_in_meters=0.001, backend="numpy")
        from .grasp_geometry import inspect_fingers
        self.finger_geometry = [inspect_fingers(stage, path.replace(ROOT, env0 + "/Station"))
                                for path in ROBOT_PATHS.values()]
        pc = self.context.get_physics_context()
        pc.enable_gpu_dynamics(True)
        pc.set_broadphase_type("GPU")
        pc.set_solver_type("TGS")
        from .physics_capacity import configure_capacity
        self.physics_capacity_report = configure_capacity(pc, config.gpu_total_aggregate_pairs_capacity,
                                                         config.gpu_found_lost_aggregate_pairs_capacity)
        from .physics_scene import prepare_physics_scene
        self.physics_scene_report = prepare_physics_scene(stage, env0, render=config.render, mode=config.physics_scene_mode)
        self.env_paths = [f"/World/envs/env_{i}" for i in range(self.n)]
        cloner = Cloner(stage)
        cloner.define_base_env("/World/envs")
        # Inherited environment roots can lose these instanced block visuals.
        # Independent root copies retain shared asset references and GPU physics replication.
        cloner.clone(env0, self.env_paths, positions=self.offsets * 1000, replicate_physics=True,
                     base_env_path="/World/envs", root_path="/World/envs/env_", enable_env_ids=True, copy_from_source=True)
        cloner.filter_collisions(pc.prim_path, "/World/collisions", self.env_paths, [])
        if config.render:
            _configure_calibration_renderer(stage, anti_aliasing=config.anti_aliasing, dlss_mode=config.dlss_mode)
        from .render_validation import SAMPLING_KEYS
        if config.lighting_samples is not None:
            for key in SAMPLING_KEYS:
                settings.set(key, config.lighting_samples)
        for _ in range(30):
            app.update()
        self.robot_paths = [{arm: path.replace(ROOT, env + "/Station") for arm, path in ROBOT_PATHS.items()} for env in self.env_paths]
        self.cameras, self.mounts, self.camera_report = [], [], []
        self.tiled_capture = None
        self.camera_passes = None
        if config.render:
            for env, robots in zip(self.env_paths, self.robot_paths):
                camera_paths = {name: path.replace(ROOT, env + "/Station") for name, path in CAMERA_PATHS.items()}
                paths, intrinsic, mounts = _create_wrist_camera_proxies(stage, camera_paths, robots, env + "/Cameras")
                for name, path in paths.items():
                    near = config.wrist_near_clip_mm if name.endswith("_wrist") else 1
                    UsdGeom.Camera(stage.GetPrimAtPath(path)).GetClippingRangeAttr().Set(Gf.Vec2f(near, 3000))
                cams, report = _create_calibrated_cameras(stage, paths, intrinsic, env.replace("/", "_") + "_",
                                                         initialize_render_products=config.camera_batch_mode == "individual")
                self.cameras.append(cams)
                self.mounts.append(mounts)
                self.camera_report.append(report)
                for name, camera in cams.items():
                    report[name]["clipping_range_mm"] = list(camera.get_clipping_range())
            if config.camera_batch_mode in ("tiled", "tiled_split"):
                from .camera_batch import SplitTiledRgbaCapture, TiledRgbaCapture
                capture = SplitTiledRgbaCapture if config.camera_batch_mode == "tiled_split" else TiledRgbaCapture
                self.tiled_capture = capture(self.cameras)
                for report in self.camera_report:
                    for camera in report.values():
                        camera["capture_api"] = self.tiled_capture.report["capture_api"]
                        camera["tiled_render_product"] = self.tiled_capture.report["render_product"]
            elif isolate_camera_passes:
                from .camera_batch import IndividualCameraPasses
                self.camera_passes = IndividualCameraPasses(self.cameras)
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
        from isaacsim.core.simulation_manager import SimulationManager
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
                raise RuntimeError(f"{arm} articulation view order does not match logical environment order")
            self.robots.append(robot)
            print("ARTICULATION", arm, robot.count, robot.dof_names, robot.body_names, flush=True)
            self.calibration_provenance["runtime_readback"][arm] = apply_runtime_profile(
                robot, arm, self.runtime_profile,
                config.gripper_finger_effort_limit_n)
        print("RUNTIME_CALIBRATION", json.dumps(self.calibration_provenance), flush=True)
        # Contact sensor preparation also changes sleep thresholds. Keep the
        # source dynamics while reading poses directly from the physics view.
        self.block_view = RigidPrim("/World/envs/env_.*/Blocks/block_.*", name="blocks", reset_xform_properties=False,
                                    track_contact_forces=False, prepare_contact_sensors=False)
        self.block_view.initialize()
        self.block_count = len(specs[0].blocks)
        self.block_paths = list(self.block_view.prim_paths)
        self.block_indices = np.array([[self.block_paths.index(f"{env}/Blocks/block_{b:02}") for b in range(self.block_count)] for env in self.env_paths])
        self.fixed_block_mask = np.asarray(
            [bool(getattr(block, "fixed_on_table", False)) for block in specs[0].blocks], dtype=bool
        )
        cache_usd = UsdGeom.XformCache()
        self.bases = np.array([[rigid_pose(cache_usd.GetLocalToWorldTransform(stage.GetPrimAtPath(robots[arm] + "/base_link"))) for arm in ARM_ORDER] for robots in self.robot_paths])
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
            "capture_profile": "single_pass_v3" if config.camera_batch_mode == "individual" and not config.replicator_sync_only else "experimental",
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
            "settings": {key: settings.get(key) for key in (
                "/rtx/rendermode", "/rtx/post/aa/op", "/rtx/post/dlss/execMode",
                "/rtx/post/motionblur/enabled", "/rtx-transient/dlssg/enabled", "/app/useFabricSceneDelegate", *SAMPLING_KEYS)},
        }
        self.physics_output.finish_initialization(camera_count=sum(len(batch) for batch in self.cameras))
        self.physics_output_report = self.physics_output.report

    def _commands(self, actions: np.ndarray, arm: int) -> tuple[np.ndarray, np.ndarray]:
        from tower_sim.replay.replay_actions import command_for_state
        from tower_sim.replay.dual_replay_worker import _scale_gripper_command_for_stage
        names = self.robots[arm].dof_names
        rows = []
        for action in actions:
            indices, positions, velocity, _ = command_for_state(action[arm * 8:arm * 8 + 8], np.zeros(8), names)
            positions, _ = _scale_gripper_command_for_stage(indices, positions, velocity, names, 1000)
            rows.append(positions)
        return indices, np.asarray(rows, dtype=np.float32)

    def reset(self, env_ids: list[int], episode_specs: list[EpisodeSpec]) -> None:
        if len(env_ids) != len(episode_specs) or any(self.recording[env_ids]):
            raise RuntimeError("reset requires matching specs and non-recording environments")
        self.physics_output.verify()
        self.physics_output.verify_tensor_views(*self.robots, self.block_view)
        indices = np.asarray(env_ids, dtype=np.int32)
        self.elapsed[indices] = 0.0
        self.previous[indices] = self.home
        for arm, robot in enumerate(self.robots):
            joints, positions = self._commands(np.tile(self.home, (len(env_ids), 1)), arm)
            robot.set_joint_positions(positions, indices=indices, joint_indices=joints)
            robot.set_joint_velocities(np.zeros_like(positions), indices=indices, joint_indices=joints)
            robot.set_joint_position_targets(positions, indices=indices, joint_indices=joints)
        ids = self.block_indices[indices].ravel()
        poses = np.concatenate([np.asarray([b.pose for b in spec.blocks]) for spec in episode_specs])
        world = poses[:, :3] + np.repeat(self.offsets[indices], self.block_count, axis=0)
        self.block_view.set_world_poses(world * 1000, np.roll(poses[:, 3:], 1, axis=1), indices=ids)
        # PhysX rejects velocity writes on kinematic bodies.  Foundation
        # poses are initialized above, but only dynamic blocks need their
        # velocity state cleared on reset.
        dynamic_ids = ids[~np.tile(self.fixed_block_mask, len(env_ids))]
        if len(dynamic_ids):
            self.block_view.set_velocities(np.zeros((len(dynamic_ids), 6)), indices=dynamic_ids)
        # The worker settles resetting environments while recording the others normally.

    def _validate_fk(self) -> None:
        errors = []
        for env, state in enumerate(self.read()):
            for arm in range(2):
                predicted = self.lula.fk(state.qpos[arm * 8:arm * 8 + 7])
                actual = state.eef[arm]
                errors.append({"env": env, "arm": ARM_ORDER[arm], "meters": float(np.linalg.norm(predicted[:3] - actual[:3])),
                               "radians": float((Rotation.from_quat(predicted[3:]).inv() * Rotation.from_quat(actual[3:])).magnitude()),
                               "fk": predicted.tolist(), "physics": actual.tolist()})
        print("FK_VALIDATION", json.dumps(errors), flush=True)
        self.fk_report = errors
        if any(e["meters"] > 0.002 or e["radians"] > 0.02 for e in errors):
            raise RuntimeError("Lula FK does not match the physics TCP frame")

    def step(self, actions: np.ndarray) -> list[State]:
        self.physics_output.verify()
        self.physics_output.verify_tensor_views(*self.robots, self.block_view)
        commands = self.compile_commands(actions)
        return self.dispatch_substeps(commands, actions)

    def compile_commands(self, actions: np.ndarray) -> list[dict]:
        """Resolve calibration delays and native-unit targets exactly once."""
        actions = np.asarray(actions, dtype=float)
        if actions.shape != (self.n, 16) or not np.isfinite(actions).all():
            raise ValueError("actions must be finite (envs, 16) absolute joint commands")
        commands = []
        for arm in range(2):
            positions = []
            for substep in range(4):
                delayed = delayed_actions(self.previous, actions, self.command_delays, substep)
                joints, target = self._commands(delayed, arm)
                positions.append(target)
            targets = np.asarray(positions, dtype=np.float32)
            commands.append({"indices": np.asarray(joints, dtype=np.int32),
                             "positions": targets, "velocities": np.zeros_like(targets)})
        return commands

    def dispatch_substeps(self, commands: list[dict], actions: np.ndarray) -> list[State]:
        """Execute recorded native targets; never reapply delays or regenerate IK."""
        self.physics_output.verify()
        self.physics_output.verify_tensor_views(*self.robots, self.block_view)
        actions = np.asarray(actions, dtype=float)
        if actions.shape != (self.n, 16) or not np.isfinite(actions).all():
            raise ValueError("actions must be finite (envs, 16) absolute joint commands")
        if len(commands) != 2:
            raise ValueError("two arm command streams are required")
        for arm, command in enumerate(commands):
            indices = np.asarray(command["indices"])
            expected, _ = self._commands(actions, arm)
            if not np.array_equal(indices, expected):
                raise ValueError("recorded command DOF order differs from backend")
            for field in ("positions", "velocities"):
                values = command[field]
                if values.shape != (4, self.n, len(indices)) or not np.isfinite(values).all():
                    raise ValueError("invalid substep target tensor")
        for substep in range(4):
            for arm, robot in enumerate(self.robots):
                command = commands[arm]
                robot.set_joint_position_targets(command["positions"][substep], joint_indices=command["indices"])
                robot.set_joint_velocity_targets(command["velocities"][substep], joint_indices=command["indices"])
            self.physics.simulate(1 / 120, self.simulation_time)
            self.physics.fetch_results()
            self.simulation_time += 1 / 120
        self.elapsed += 1 / 30
        self.previous = actions.copy()
        return self.read()

    def snapshots(self) -> list[dict]:
        states = self.read()
        raw = []
        for robot in self.robots:
            view = robot._physics_view
            raw.append((robot.get_world_poses(), robot.get_joint_positions(), robot.get_joint_velocities(),
                        view.get_dof_position_targets(), view.get_dof_velocity_targets(),
                        view.get_dof_actuation_forces(), robot.get_velocities()))
        return [self.snapshot(i, (states, raw)) for i in range(self.n)]

    def snapshot(self, env_id: int, cached=None) -> dict:
        """Accessible state, in native units; not a PhysX solver checkpoint."""
        if cached is None:
            return self.snapshots()[env_id]
        self.physics_output.verify_tensor_views(*self.robots, self.block_view)
        states, raw = cached
        state = states[env_id]
        arms = []
        for robot, data in zip(self.robots, raw):
            (xyz, quat), positions, velocities, targets, velocity_targets, efforts, root_velocity = data
            arms.append({"dof_names": list(robot.dof_names),
                         "dof_types": ["translation" if int(kind) == 1 else "rotation" for kind in robot.get_dof_types()],
                         "positions": positions[env_id].tolist(),
                         "velocities": velocities[env_id].tolist(),
                         "position_targets": targets[env_id].tolist(),
                         "velocity_targets": velocity_targets[env_id].tolist(),
                         "actuation_forces": efforts[env_id].tolist(),
                         "root_position_mm": (xyz[env_id] - self.offsets[env_id] * 1000).tolist(),
                         "root_orientation_wxyz": quat[env_id].tolist(),
                         "root_velocity_native": root_velocity[env_id].tolist()})
        return {"schema": "jenga_accessible_state_v1", "native_stage_meters_per_unit": .001,
                "arms": arms, "blocks": state.blocks.tolist(),
                "block_velocities": state.block_velocities.tolist(),
                "fixed_block_mask": self.fixed_block_mask.tolist(),
                "previous": self.previous[env_id].tolist(), "elapsed": float(self.elapsed[env_id]),
                "simulation_time": self.simulation_time, "original_offset_m": self.offsets[env_id].tolist(),
                "unrestorable_state": ["PhysX contact manifolds and solver warm-start impulses",
                                       "PhysX internal sleep/island/broadphase and GPU scheduling state"],
                "restore_is_exact_simulator_checkpoint": False}

    def restore(self, env_id: int, snapshot: dict) -> None:
        if self.recording[env_id] or snapshot.get("schema") != "jenga_accessible_state_v1":
            raise ValueError("restore requires a supported snapshot and non-recording slot")
        self.physics_output.verify()
        self.physics_output.verify_tensor_views(*self.robots, self.block_view)
        if not np.array_equal(snapshot["fixed_block_mask"], self.fixed_block_mask):
            raise ValueError("snapshot rigid-body topology mismatch")
        indices = np.array([env_id], dtype=np.int32)
        for robot, saved in zip(self.robots, snapshot["arms"]):
            if list(robot.dof_names) != saved["dof_names"]:
                raise ValueError("snapshot articulation DOF order mismatch")
            robot.set_world_poses(np.asarray([saved["root_position_mm"]]) + self.offsets[env_id] * 1000,
                                  np.asarray([saved["root_orientation_wxyz"]]), indices=indices)
            robot.set_velocities(np.asarray([saved["root_velocity_native"]]), indices=indices)
            robot.set_joint_positions(np.asarray([saved["positions"]]), indices=indices)
            robot.set_joint_velocities(np.asarray([saved["velocities"]]), indices=indices)
            robot.set_joint_position_targets(np.asarray([saved["position_targets"]]), indices=indices)
            robot.set_joint_velocity_targets(np.asarray([saved["velocity_targets"]]), indices=indices)
            robot.set_joint_efforts(np.asarray([saved["actuation_forces"]]), indices=indices)
        poses = np.asarray(snapshot["blocks"])
        ids = self.block_indices[env_id]
        self.block_view.set_world_poses((poses[:, :3] + self.offsets[env_id]) * 1000,
                                       np.roll(poses[:, 3:], 1, axis=1), indices=ids)
        dynamic = ~self.fixed_block_mask
        velocity = np.asarray(snapshot["block_velocities"])[dynamic].copy()
        velocity[:, :3] *= 1000
        self.block_view.set_velocities(velocity, indices=ids[dynamic])
        self.previous[env_id] = snapshot["previous"]
        self.elapsed[env_id] = snapshot["elapsed"]
        # The shared PhysX/Fabric clock is monotonic; never rewind it per slot.
        # Episode-relative timing is restored separately and commands use integer steps.

    def capture_guarded(self, *args, **kwargs):
        """Capture must be a read-only operation with respect to physics."""
        from isaacsim.core.simulation_manager import SimulationManager
        before = self.read()
        raw_before = [(p.copy(), v.copy()) for p, v in self.raw_joint_states]
        def roots():
            return [tuple(np.asarray(value).copy() for value in (*robot.get_world_poses(), robot.get_velocities()))
                    for robot in getattr(self, "robots", [])]
        roots_before = roots()
        clock = (self.simulation_time, float(SimulationManager.get_simulation_time()))
        frames = self.capture(*args, **kwargs)
        after = self.read()
        if clock != (self.simulation_time, float(SimulationManager.get_simulation_time())):
            raise RuntimeError("camera capture advanced physics clock")
        for a, b in zip(before, after):
            for name in ("qpos", "qvel", "eef_world", "blocks", "block_velocities"):
                if not np.allclose(getattr(a, name), getattr(b, name), rtol=0, atol=1e-9):
                    raise RuntimeError(f"camera capture changed physics state: {name}")
        for a, b in zip(roots_before, roots()):
            if any(not np.array_equal(x, y) for x, y in zip(a, b)):
                raise RuntimeError("camera capture changed articulation root state")
        for a, b in zip(raw_before, self.raw_joint_states):
            if any(not np.array_equal(x, y) for x, y in zip(a, b)):
                raise RuntimeError("camera capture changed full articulation state")
        return frames

    def read(self) -> list[State]:
        self.physics_output.verify()
        self.physics_output.verify_tensor_views(*self.robots, self.block_view)
        from tower_sim.replay.dual_replay_worker import _measured_episode_vector
        qpos, qvel, efforts = [np.zeros((self.n, 16)) for _ in range(3)]
        eef_world = np.zeros((self.n, 2, 7))
        eef = np.zeros_like(eef_world)
        gripper = np.zeros((self.n, 8))
        self.link_poses = []
        self.raw_joint_states = []
        self.raw_joint_efforts = []
        for arm, robot in enumerate(self.robots):
            positions = robot.get_joint_positions()
            velocities = robot.get_joint_velocities()
            self.raw_joint_states.append((np.asarray(positions).copy(), np.asarray(velocities).copy()))
            force = robot.get_measured_joint_efforts()
            self.raw_joint_efforts.append(np.asarray(force).copy())
            names = robot.dof_names
            links = np.asarray(robot._physics_view.get_link_transforms()).copy()
            self.link_poses.append(links)
            tool = links[:, robot.body_names.index("gripper_base")]
            for env in range(self.n):
                qpos[env, arm * 8:arm * 8 + 8] = _measured_episode_vector(None, names, 0.001, positions[env])
                qvel[env, arm * 8:arm * 8 + 8] = _measured_episode_vector(None, names, 0.001, velocities[env])
                efforts[env, arm * 8:arm * 8 + 7] = [force[env, names.index(f"joint{i}")] * 1e-6 for i in range(1, 8)]
                efforts[env, arm * 8 + 7] = 0.0005 * (force[env, names.index("gripper_joint1")] - force[env, names.index("gripper_joint2")])
                gripper[env, arm * 4:arm * 4 + 4] = [qpos[env, arm * 8 + 7], abs(force[env, names.index("gripper_joint1")]) * 0.001, 1, 0]
                rotation = Rotation.from_quat(tool[env, 3:])
                xyz = tool[env, :3] * 0.001 + rotation.apply(self.lula.tcp_offset)
                tcp_rotation = rotation * self.lula.tcp_rotation
                eef_world[env, arm] = np.r_[xyz - self.offsets[env], tcp_rotation.as_quat()]
                base = self.bases[env, arm]
                inverse = Rotation.from_quat(base[3:]).inv()
                eef[env, arm] = np.r_[inverse.apply(xyz - base[:3]), (inverse * tcp_rotation).as_quat()]
        positions, orientations = self.block_view.get_world_poses()
        block_pose = np.c_[positions * 0.001, np.roll(orientations, -1, axis=1)][self.block_indices]
        block_pose[:, :, :3] -= self.offsets[:, None, :]
        self._latest_block_poses = block_pose.copy()
        velocity = self.block_view.get_velocities()[self.block_indices].copy()
        velocity[:, :, :3] *= 0.001
        return [State(float(self.elapsed[i]), qpos[i], qvel[i], efforts[i], eef[i], eef_world[i], gripper[i], block_pose[i], velocity[i]) for i in range(self.n)]

    def to_base(self, env: int, arm: int, target: np.ndarray) -> np.ndarray:
        base = self.bases[env, arm]
        inverse = Rotation.from_quat(base[3:]).inv()
        return np.r_[inverse.apply(target[:3] + self.offsets[env] - base[:3]), (inverse * Rotation.from_quat(target[3:])).as_quat()]

    def finger_lowest_z(self, env: int, arm: int) -> float:
        heights = []
        for finger in self.finger_geometry[arm]['fingers']:
            pose = self.link_poses[arm][env, self.robots[arm].body_names.index(finger['link'])]
            points = Rotation.from_quat(pose[3:]).apply(finger['distal_link_corners_m'])
            heights.append(float(points[:, 2].min() + pose[2] * .001 - self.offsets[env, 2]))
        return min(heights)


    def capture(self, rt_subframes: int | None = None, *, mask_wrist_mounts: bool = False) -> list[dict[str, np.ndarray]]:
        import omni.replicator.core as rep
        from .render_validation import check_wrist_near_plane, verify_capture_settings
        from tower_sim.replay.dual_replay_worker import CAMERA_NAMES, _capture_calibrated_camera_batches
        if not self.config.render:
            raise RuntimeError("RGB capture is disabled; diagnostic runs cannot be published")
        self.physics_output.verify_capture()
        verify_capture_settings(self._render_settings, self.config)
        for env, cameras in enumerate(self.cameras):
            for arm, name in enumerate(("left_wrist", "back_wrist")):
                mount = self.report["wrist_camera_proxies"][name]
                relative = np.asarray(mount["camera_to_gripper_matrix"])
                relative_rotation = Rotation.from_matrix((relative[:3, :3] * 1000).T)
                tool = self.link_poses[arm][env, self.robots[arm].body_names.index("gripper_base")]
                rotation = Rotation.from_quat(tool[3:])
                position = tool[:3] + rotation.apply(relative[3, :3] * 1000)
                orientation = (rotation * relative_rotation).as_quat()
                cameras[name].set_world_pose(position, np.roll(orientation, 1), camera_axes="usd")
                if not mask_wrist_mounts:
                    near = cameras[name].get_clipping_range()[0]
                    if near != self.config.wrist_near_clip_mm:
                        raise RuntimeError(f"Wrist near clipping changed during capture: {near}")
                    check_wrist_near_plane(self._latest_block_poses[env], self.offsets[env], position, orientation,
                                          np.asarray(self.camera_report[env][name]["intrinsic_matrix_k"]), near,
                                          f"env_{env}:{name}")
        masks = [path + "/gripper_base/WristCameraMount" for robots in self.robot_paths for path in robots.values()]
        if not mask_wrist_mounts:
            masks = []
        subframes = rt_subframes if rt_subframes is not None else (3 if self.capture_count == 0 else self.config.render_subframes)
        batches = _capture_calibrated_camera_batches(self.cameras, rep, self.app, subframes, 180, self.stage, masks,
                                                    self.tiled_capture.read if self.tiled_capture else None,
                                                    self.camera_passes.prepare if self.camera_passes else getattr(self.tiled_capture, "prepare", None),
                                                    self.config.replicator_sync_only)
        verify_capture_settings(self._render_settings, self.config)
        self.capture_count += 1
        self.physics_output.verify_capture()
        return [{dest: batch[source] for source, dest in zip(CAMERA_NAMES, CAMERAS)} for batch in batches]

    def validate_initial_rgb(self, frames: list[dict[str, np.ndarray]], specs: list[EpisodeSpec]) -> list[dict]:
        from itertools import product
        from .render_validation import check_wood_region
        corners = np.array(list(product((-1, 1), repeat=3))) * [0.075, 0.025, 0.015]
        checks = []
        for env, (state, spec) in enumerate(zip(self.read(), specs)):
            for tower in sorted({block.tower for block in spec.blocks}):
                points = np.concatenate([Rotation.from_quat(pose[3:]).apply(corners) + pose[:3] + self.offsets[env]
                                         for block, pose in zip(spec.blocks, state.blocks) if block.tower == tower])
                position, orientation = self.cameras[env]["low"].get_world_pose(camera_axes="usd")
                optical = Rotation.from_quat(np.roll(orientation, -1)).inv().apply(points * 1000 - position) * [1, -1, -1]
                intrinsic = np.asarray(self.camera_report[env]["low"]["intrinsic_matrix_k"])
                projected = optical @ intrinsic.T
                pixels = projected[:, :2] / projected[:, 2:]
                pixels = np.array([639, 479]) - pixels
                checks.append(check_wood_region(frames[env][CAMERAS[0]], pixels, f"env_{env}:{tower}"))
        return checks
