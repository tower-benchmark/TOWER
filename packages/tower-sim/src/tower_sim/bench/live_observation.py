"""Effective persistent CPU camera path; synchronous delivery lives in cameras.py.

CUDA staging and unscheduled direct rendering are deliberately not candidate
variants. They were overridden by the benchmark's synchronous subclass.
"""

import time

import numpy as np

from .ordinary_capture import OrdinaryFourCapture


class LiveFourCameraObservation(OrdinaryFourCapture):
    def __init__(
        self,
        sim,
        output,
        *,
        device="cpu",
        subframes=6,
        persistent_views=True,
        profile_timings=True,
        mask_policy="global_secondary",
        refresh_physics_once=False,
        adaptive_min_subframes=None,
        adaptive_joint_threshold=0.005,
        resolution=(640, 480),
        renderer_overrides=None,
        resources,
        near_guard,
    ):
        if device != "cpu":
            raise ValueError("Use the synchronous CPU capture factory")
        if subframes < 1:
            raise ValueError("At least one render subframe is required")
        self.device = device
        if mask_policy not in ("self", "global_secondary"):
            raise ValueError("Unknown wrist housing shadow-mask policy")
        self.mask_policy = mask_policy
        self.persistent_views = persistent_views
        self.initializing = True
        self.app_updates = 0
        self.refresh_physics_once = refresh_physics_once
        self.physics_refreshes = 0
        if adaptive_min_subframes is not None:
            if not 1 <= adaptive_min_subframes <= subframes:
                raise ValueError("Adaptive minimum must be between one and the full subframe count")
            if mask_policy != "global_secondary" or not persistent_views:
                raise ValueError("Adaptive scheduling requires persistent global-mask views")
            if adaptive_joint_threshold <= 0:
                raise ValueError("Adaptive motion threshold must be positive")
        self.adaptive_min_subframes = adaptive_min_subframes
        self.adaptive_joint_threshold = adaptive_joint_threshold
        self.full_subframes = subframes
        self._previous_motion_state = None
        self.adaptive_schedule_counts = {}
        schedule = (
            (("high", "low"), 1, subframes),
            (("left_wrist",), 1, subframes),
            (("back_wrist",), 1, subframes),
        )
        if mask_policy == "global_secondary":
            schedule = ((("high", "low", "left_wrist", "back_wrist"), 1, subframes),)
        super().__init__(
            sim,
            output,
            preset="fast",
            schedule=schedule,
            rgb_device="cpu",
            profile_timings=profile_timings,
            resolution=resolution,
            renderer_overrides=renderer_overrides,
            resources=resources,
            near_guard=near_guard,
        )
        from pxr import Sdf  # pyright: ignore[reportMissingImports]

        # Zero selects the sensor's autotrigger mode. No exposure, projection,
        # image size or attachment transform is changed.
        for path in self.reference_paths.values():
            prim = sim.env.stage.GetPrimAtPath(path)
            prim.AddAppliedSchema("OmniSensorAPI")
            prim.CreateAttribute("omni:sensor:tickRate", Sdf.ValueTypeNames.Float).Set(0.0)
        self.initializing = False
        if persistent_views:
            for product in self.products.values():
                product.hydra_texture.set_updates_enabled(True)

    def _activate_roles(self, roles):
        if self.initializing or not self.persistent_views:
            # Avoid redundant calls: changing the enabled state can rebuild
            # sensor viewports in this Kit version.
            for role, product in self.products.items():
                texture = product.hydra_texture
                enabled = role in roles
                if texture.get_updates_enabled() != enabled:
                    texture.set_updates_enabled(enabled)

    def _select_subframes(self, states=None):
        if self.adaptive_min_subframes is None:
            return
        from scipy.spatial.transform import Rotation

        states = self.sim.read_batch() if states is None else states
        current = {
            name: np.stack([getattr(state, name) for state in states])
            for name in ("qpos", "blocks", "eef_world")
        }
        previous = self._previous_motion_state
        full = previous is None or not self.history_ready
        if previous is not None:
            qdelta = abs(current["qpos"] - previous["qpos"])
            arm_motion = float(qdelta[:, np.r_[0:7, 8:15]].max())
            finger_motion = float(qdelta[:, [7, 15]].max())
            block_motion = float(
                np.linalg.norm(
                    current["blocks"][..., :3] - previous["blocks"][..., :3], axis=-1
                ).max()
            )
            old_rot = Rotation.from_quat(previous["blocks"][..., 3:].reshape(-1, 4))
            new_rot = Rotation.from_quat(current["blocks"][..., 3:].reshape(-1, 4))
            block_angle = float((old_rot.inv() * new_rot).magnitude().max())
            wrist_motion = float(
                np.linalg.norm(
                    current["eef_world"][..., :3] - previous["eef_world"][..., :3], axis=-1
                ).max()
            )
            old_wrist = Rotation.from_quat(previous["eef_world"][..., 3:].reshape(-1, 4))
            new_wrist = Rotation.from_quat(current["eef_world"][..., 3:].reshape(-1, 4))
            wrist_angle = float((old_wrist.inv() * new_wrist).magnitude().max())
            full |= (
                arm_motion > self.adaptive_joint_threshold
                or finger_motion > 0.0005
                or block_motion > 0.0005
                or block_angle > 0.005
                or wrist_motion > 0.001
                or wrist_angle > 0.005
            )
        count = self.full_subframes if full else self.adaptive_min_subframes
        self.schedule = ((("high", "low", "left_wrist", "back_wrist"), 1, count),)
        self.adaptive_schedule_counts[count] = self.adaptive_schedule_counts.get(count, 0) + 1
        self._previous_motion_state = current

    def _before_capture(self):
        self._capture_before_states = self.sim.read_batch()
        self._select_subframes(self._capture_before_states)
        return self._capture_before_states[0]

    def _after_capture(self):
        self._capture_states = self.sim.read_batch()
        for index, (old, new) in enumerate(zip(self._capture_before_states, self._capture_states)):
            assert old.time == new.time, index
            for field in ("qpos", "qvel", "blocks", "block_velocities", "eef_world"):
                assert np.array_equal(getattr(old, field), getattr(new, field)), (index, field)
        return self._capture_states[0]

    def invalidate_history(self):
        self._previous_motion_state = None
        super().invalidate_history()

    def _apply_masks(self, roles):
        for role, attributes in self.masks.items():
            enabled = role in roles if self.mask_policy == "self" else True
            for attribute in attributes:
                if attribute.Get() != enabled:
                    attribute.Set(enabled)

    def _read_rgb(self, name):
        rgba = np.asarray(self.annotators[name].get_data(device="cpu", do_array_copy=False))
        assert rgba.shape == (*self.resolution[::-1], 4), (name, rgba.shape)
        return rgba[..., :3].copy()

    def observe(self):
        started = time.perf_counter()
        images = self()
        rgb = np.stack([images[name] for name in self.names])[None]
        state = self._capture_states[0]
        robot, cpu = self._pack_robot_state([state])
        return {
            "rgb": rgb,
            "robot_state": robot,
            "qpos": cpu["qpos"],
            "time_s": np.array([state.time]),
            "observation_wall_s": time.perf_counter() - started,
        }

    def _pack_robot_state(self, states):
        fields = ("qpos", "qvel", "effort", "eef", "eef_world", "gripper")
        cpu = {name: np.stack([getattr(state, name) for state in states]) for name in fields}
        return cpu, cpu

    def report(self):
        report = super().report()
        report.update(
            mode="live_persistent_camera_observation",
            output_device=self.device,
            direct_app_updates=self.app_updates,
            persistent_views=self.persistent_views,
            refresh_physics_once_per_pass=self.refresh_physics_once,
            direct_physics_refreshes=self.physics_refreshes,
            adaptive_min_subframes=self.adaptive_min_subframes,
            adaptive_full_subframes=self.full_subframes,
            adaptive_joint_threshold_rad=self.adaptive_joint_threshold,
            adaptive_schedule_counts=self.adaptive_schedule_counts,
            adaptive_motion_guards={
                "gripper_width_m": 0.0005,
                "block_translation_m": 0.0005,
                "block_rotation_rad": 0.005,
                "wrist_translation_m": 0.001,
                "wrist_rotation_rad": 0.005,
            },
            sensor_tick_rate_hz=0,
            steady_replicator_steps=0,
            response_and_pose_checks="Every frame; external moving-pose fidelity validation.",
            housing_mask_policy=self.mask_policy,
            rgb_transfer_mode="owned_cpu_array",
            rgb_zero_copy=False,
            pinned_host_reuse_protected_by_cuda_events=False,
            state_reads_per_observation=2,
            housing_geometry_visible=True,
            delivery_verified=False,
            dispatch_gate_opened=False,
            delivery_pump_counts={},
            unverified_observations=0,
            extra_delivery_calls=0,
            housing_mask_description=(
                "Original per-view self-only secondary-ray and shadow masks."
                if self.mask_policy == "self"
                else "Both housings suppress secondary rays/shadows in every view."
            ),
        )
        scale = np.diag([self.resolution[0] / 640, self.resolution[1] / 480, 1.0])
        report["intrinsic_matrices_k"] = {
            name: (
                scale @ np.asarray(self.sim.env.camera_report[0][role]["intrinsic_matrix_k"])
            ).tolist()
            for name, role in self.names.items()
        }
        report["intrinsics_note"] = (
            "Render resolution scaled from calibrated 640x480; optics, field of view and extrinsics unchanged."
        )
        return report


class BatchedLiveCameraObservation(LiveFourCameraObservation):
    def __init__(self, sim, output, **kwargs):
        self.environment_count = sim.env.n
        super().__init__(sim, output, **kwargs)
        if self.environment_count == 1:
            return
        import math

        import carb  # pyright: ignore[reportMissingImports]
        import omni.replicator.core as rep  # pyright: ignore[reportMissingImports]
        from isaacsim.core.utils.render_product import (  # pyright: ignore[reportMissingImports]
            get_resolution,
        )
        from pxr import Gf, Sdf, Usd, UsdGeom  # pyright: ignore[reportMissingImports]

        env = sim.env
        self.batch_camera_paths = {
            role: [self.reference_paths[role]] for role in self.names.values()
        }
        self.batch_camera_xforms = {
            role: [self.reference_xforms[role]] for role in self.names.values()
        }
        self.additional_original_pause = []
        self.additional_original_ranges = []
        flattened = env.stage.Flatten()
        for index, cameras in enumerate(env.cameras[1:], 1):
            UsdGeom.Xform.Define(env.stage, f"/__LiveBatchCameras/env_{index}")
            for role, camera in cameras.items():
                self.additional_original_pause.append((camera, camera.is_paused()))
                camera.pause()
                camera._render_product.hydra_texture.set_updates_enabled(False)
                if role.endswith("_wrist"):
                    self.additional_original_ranges.append((camera, camera.get_clipping_range()))
                    camera.set_clipping_range(self.profile["capture"]["wrist_near_clip_mm"], 3000.0)
                target = f"/__LiveBatchCameras/env_{index}/{role}"
                assert Sdf.CopySpec(flattened, camera.prim_path, env.stage.GetRootLayer(), target)
                prim = env.stage.GetPrimAtPath(target)
                transform = UsdGeom.Xformable(prim).MakeMatrixXform()
                transform.Set(
                    UsdGeom.XformCache().GetLocalToWorldTransform(
                        env.stage.GetPrimAtPath(camera.prim_path)
                    )
                )
                prim.AddAppliedSchema("OmniSensorAPI")
                prim.CreateAttribute("omni:sensor:tickRate", Sdf.ValueTypeNames.Float).Set(0.0)
                self.batch_camera_paths[role].append(target)
                self.batch_camera_xforms[role].append(transform)
        # Keep setup annotators attached and inactive. Detaching them here
        # stops Replicator and deletes the legacy Articulation tensor handles.
        self._setup_products, self._setup_annotators = self.products, self.annotators
        for product in self._setup_products.values():
            product.hydra_texture.set_updates_enabled(False)
        self.products, self.annotators = {}, {}
        self.atlas_resolution = {}
        padded = math.ceil(math.sqrt(self.environment_count)) ** 2
        UsdGeom.Xform.Define(env.stage, "/__LiveBatchPadding")
        carb.settings.get_settings().set_int("/rtx/viewTile/limit", max(16, padded))
        for name, role in self.names.items():
            paths = list(self.batch_camera_paths[role])
            for index in range(padded - self.environment_count):
                path = f"/__LiveBatchPadding/{role}_{index}"
                assert Sdf.CopySpec(
                    env.stage.GetRootLayer(), paths[0], env.stage.GetRootLayer(), path
                )
                paths.append(path)
            product = rep.create.render_product_tiled(
                paths, tile_resolution=self.resolution, name="live_batch_" + role
            )
            annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu", do_array_copy=True)
            annotator.attach(product.path)
            self.products[role], self.annotators[name] = product, annotator
            self.atlas_resolution[role] = tuple(get_resolution(product.path))
        for arm, role in enumerate(("left_wrist", "back_wrist")):
            for paths in env.robot_paths[1:]:
                root = list(paths.values())[arm]
                mount = env.stage.GetPrimAtPath(root + "/gripper_base/WristCameraMount")
                UsdGeom.Imageable(mount).MakeVisible()
                for prim in Usd.PrimRange(mount):
                    if prim.GetName() == "visual" and prim.IsA(UsdGeom.Mesh):
                        for flag in ("doNotCastShadows", "invisibleToSecondaryRays"):
                            attr = (
                                UsdGeom.PrimvarsAPI(prim)
                                .CreatePrimvar(flag, Sdf.ValueTypeNames.Bool)
                                .GetAttr()
                            )
                            attr.Set(False)
                            self.masks[role].append(attr)
        self.initializing = True
        for _ in range(2):
            self._step(2)
        self.initializing = False
        self.expected_attributes = []
        for name, role in self.names.items():
            response = self.profile["cameras"][name]
            for path in self.batch_camera_paths[role]:
                prim = env.stage.GetPrimAtPath(path)
                prim.AddAppliedSchema("OmniRtxCameraExposureAPI_1")
                for suffix, value in [
                    ("time", response["exposure_time_s"]),
                    ("iso", self.controls.get("/rtx/post/tonemap/filmIso")),
                    ("fStop", self.controls.get("/rtx/post/tonemap/fNumber")),
                    ("responsivity", self.controls.get("/rtx/post/tonemap/cm2Factor")),
                ]:
                    attr = prim.CreateAttribute("exposure:" + suffix, Sdf.ValueTypeNames.Float)
                    attr.Set(value)
                    self.expected_attributes.append((attr, value))
            attr = env.stage.GetPrimAtPath(self.products[role].path).CreateAttribute(
                "omni:rtx:post:tonemap:whitepoint", Sdf.ValueTypeNames.Color3f
            )
            attr.Set(Gf.Vec3f(*response["whitepoint"]))
            self.expected_attributes.append((attr, response["whitepoint"]))
        self._verify_response()
        self._wrist_intrinsics = [
            {
                role: np.asarray(env.camera_report[index][role]["intrinsic_matrix_k"])
                for role in ("left_wrist", "back_wrist")
            }
            for index in range(env.n)
        ]
        self._wrist_near_mm = [
            {role: cameras[role].get_clipping_range()[0] for role in ("left_wrist", "back_wrist")}
            for cameras in env.cameras
        ]

    def _legacy_batch_pose(self):
        """Original per-environment camera poses, evaluated on every synchronous frame."""
        from pxr import UsdGeom  # pyright: ignore[reportMissingImports]
        from scipy.spatial.transform import Rotation

        env = self.sim.env
        result = {role: {} for role in self.names.values()}
        for index, cameras in enumerate(env.cameras[1:], 1):
            for arm, role in enumerate(("left_wrist", "back_wrist")):
                relative = np.asarray(
                    env.report["wrist_camera_proxies"][role]["camera_to_gripper_matrix"]
                )
                tool = env.link_poses[arm][index, env.robots[arm].body_names.index("gripper_base")]
                rotation = Rotation.from_quat(tool[3:])
                xyz = tool[:3] + rotation.apply(relative[3, :3] * 1000)
                quaternion = (
                    rotation * Rotation.from_matrix((relative[:3, :3] * 1000).T)
                ).as_quat()
                cameras[role].set_world_pose(xyz, np.roll(quaternion, 1), camera_axes="usd")
            cache = UsdGeom.XformCache()
            for role, camera in cameras.items():
                result[role][index] = np.asarray(
                    cache.GetLocalToWorldTransform(env.stage.GetPrimAtPath(camera.prim_path))
                )
        return result

    def _read_rgb(self, name):
        if self.environment_count == 1:
            return super()._read_rgb(name)
        role = self.names[name]
        width, height = self.atlas_resolution[role]
        value = np.asarray(self.annotators[name].get_data(device="cpu", do_array_copy=False))
        assert value.shape == (height, width, 4), (name, value.shape)
        return self._unpack_tiles(value).copy()

    def _unpack_tiles(self, atlas):
        """(rows*H, columns*W, 4) atlas -> (N,H,W,3) tiles in environment order; NumPy."""
        tile_width, tile_height = self.resolution
        rows, columns = atlas.shape[0] // tile_height, atlas.shape[1] // tile_width
        tiles = atlas.reshape(rows, tile_height, columns, tile_width, 4)
        tiles = tiles.transpose(0, 2, 1, 3, 4)
        return tiles.reshape(-1, tile_height, tile_width, 4)[: self.environment_count, ..., :3]

    def _rotate_rgb(self, value):
        if self.environment_count == 1:
            return super()._rotate_rgb(value)
        return np.ascontiguousarray(np.rot90(value, 2, axes=(1, 2)))

    def observe(self):
        if self.environment_count == 1:
            return super().observe()
        images = self()
        rgb = np.stack([images[name] for name in self.names], axis=1)
        states = self._capture_states
        robot, cpu = self._pack_robot_state(states)
        return {
            "rgb": rgb,
            "robot_state": robot,
            "qpos": cpu["qpos"],
            "time_s": np.array([state.time for state in states]),
        }

    def report(self):
        return dict(
            super().report(),
            environment_count=self.environment_count,
            batch_pose_checks_passed=getattr(self, "pose_checks_passed", None),
            # Historical report field; synchronous pose checks run every frame.
            batch_pose_check_interval=64 if self.environment_count > 1 else None,
            active_render_products=len(self.products),
            inactive_setup_products=len(getattr(self, "_setup_products", {})),
            camera_batching="One tiled product per camera role across environments.",
        )

    def close(self):
        super().close()
        for name, annotator in getattr(self, "_setup_annotators", {}).items():
            product = self._setup_products[self.names[name]]
            annotator.detach(product.path)
            product.destroy()
        for camera, clip in getattr(self, "additional_original_ranges", []):
            camera.set_clipping_range(*clip)
        for camera, paused in getattr(self, "additional_original_pause", []):
            if not paused:
                camera.resume()
