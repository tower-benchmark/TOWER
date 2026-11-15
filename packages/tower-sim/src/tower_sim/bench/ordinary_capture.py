"""Ordinary RTX RGB products at the same calibrated poses and responses."""

import copy
import json
import time
from pathlib import Path

import numpy as np

from .four_capture import Native61Capture

CAPTURE_SCHEDULES = {
    "original": ((("high", "low"), 1, 2), (("left_wrist",), 2, 2), (("back_wrist",), 4, 4)),
    "balanced": ((("high", "low"), 1, 2), (("left_wrist",), 2, 2), (("back_wrist",), 2, 2)),
    "fast": ((("high", "low"), 1, 2), (("left_wrist",), 1, 2), (("back_wrist",), 1, 2)),
}


class OrdinaryFourCapture(Native61Capture):
    def __init__(
        self,
        sim,
        output,
        *,
        preset="original",
        schedule=None,
        renderer_overrides=None,
        profile_timings=False,
        rgb_device="cpu",
        resolution=(640, 480),
        resources,
        near_guard,
    ):
        import carb  # pyright: ignore[reportMissingImports]
        import omni.replicator.core as rep  # pyright: ignore[reportMissingImports]
        from pxr import Gf, Sdf, Usd, UsdGeom  # pyright: ignore[reportMissingImports]

        self.sim, self.output = sim, output
        self.resources, self.near_guard = resources, near_guard
        self.resolution = tuple(resolution)
        if len(self.resolution) != 2 or any(
            not isinstance(v, int) or v <= 0 for v in self.resolution
        ):
            raise ValueError("Resolution must contain two positive integers")
        if self.resolution[0] * 480 != self.resolution[1] * 640:
            raise ValueError("Retain the calibrated 4:3 camera aspect ratio")
        profile_path = resources.path("camera_profile")
        self.profile = copy.deepcopy(json.loads(profile_path.read_text()))
        self.profile["renderer_controls"].update(renderer_overrides or {})
        self.schedule = schedule if schedule is not None else CAPTURE_SCHEDULES[preset]
        self.preset = preset if schedule is None else "custom"
        self.profile_timings = profile_timings
        self.frame_timings = []
        self.controls = carb.settings.get_settings()
        self.previous_controls = {
            key: self.controls.get(key) for key in self.profile["renderer_controls"]
        }
        self.previous_capture_on_play = self.controls.get("/omni/replicator/captureOnPlay")
        rep.orchestrator.set_capture_on_play(False)
        for key, value in self.profile["renderer_controls"].items():
            self.controls.set(key, value)
        self.frames, self.steps = 0, 0
        self.minimum_depth, self.comparisons = {}, []
        self.history_ready = False
        self.cameras = sim.env.cameras[0]
        self.original_near = sim.env.config.wrist_near_clip_mm
        self.original_ranges = {
            role: self.cameras[role].get_clipping_range() for role in ("left_wrist", "back_wrist")
        }
        for role in self.original_ranges:
            self.cameras[role].set_clipping_range(
                self.profile["capture"]["wrist_near_clip_mm"], self.original_ranges[role][1]
            )
        sim.env.config.wrist_near_clip_mm = self.profile["capture"]["wrist_near_clip_mm"]
        self.original_pause = {role: camera.is_paused() for role, camera in self.cameras.items()}
        self.products, self.annotators = {}, {}
        self.reference_paths, self.reference_xforms = {}, {}
        for camera in self.cameras.values():
            camera.pause()
            camera._render_product.hydra_texture.set_updates_enabled(False)
        reference_root = "/__OrdinaryFourReferenceCameras"
        UsdGeom.Xform.Define(sim.env.stage, reference_root)
        flattened = sim.env.stage.Flatten()
        for name, role in self.names.items():
            camera = self.cameras[role]
            camera.pause()
            camera._render_product.hydra_texture.set_updates_enabled(False)
            target = reference_root + "/" + role
            assert Sdf.CopySpec(flattened, camera.prim_path, sim.env.stage.GetRootLayer(), target)
            prim = sim.env.stage.GetPrimAtPath(target)
            xform = UsdGeom.Xformable(prim).MakeMatrixXform()
            xform.Set(
                UsdGeom.XformCache().GetLocalToWorldTransform(
                    sim.env.stage.GetPrimAtPath(camera.prim_path)
                )
            )
            self.reference_paths[role], self.reference_xforms[role] = target, xform
            product = rep.create.render_product(
                target, self.resolution, name="ordinary_reference_" + role
            )
            product.hydra_texture.set_updates_enabled(False)
            annotator = rep.AnnotatorRegistry.get_annotator(
                "rgb", device=rgb_device, do_array_copy=True
            )
            annotator.attach(product.path)
            self.products[role], self.annotators[name] = product, annotator
        source_path = resources.path("scene")
        source = Usd.Stage.Open(str(source_path))
        dome_path = "/__DualReplayCalibrationRender/DefaultPreviewDome"
        saved_dome, runtime_dome = (
            source.GetPrimAtPath(dome_path),
            sim.env.stage.GetPrimAtPath(dome_path),
        )
        assert saved_dome and runtime_dome
        for attr in saved_dome.GetAttributes():
            if attr.GetName().startswith("inputs:") and attr.HasAuthoredValueOpinion():
                runtime_dome.CreateAttribute(
                    attr.GetName(), attr.GetTypeName(), custom=attr.IsCustom()
                ).Set(attr.Get())
        self.masks = {}
        for role, root in zip(("left_wrist", "back_wrist"), sim.env.robot_paths[0].values()):
            self.masks[role] = []
            mount = sim.env.stage.GetPrimAtPath(root + "/gripper_base/WristCameraMount")
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
        loading_key = "/exts/omni.replicator.core/maxAssetLoadingTime"
        previous_loading_timeout = self.controls.get_as_float(loading_key)
        self.controls.set(loading_key, max(previous_loading_timeout, 120.0))
        try:
            for _ in range(2):
                self._step(2)
            self._wait_for_exposure()
        finally:
            self.controls.set(loading_key, previous_loading_timeout)
        # SDK render-product initialization can overwrite global defaults.
        # Reassert the identical frozen profile once, before any policy frame.
        before = {key: self.controls.get(key) for key in self.profile["renderer_controls"]}
        for key, value in self.profile["renderer_controls"].items():
            current = self.controls.get(key)
            same = (
                np.allclose(current, value, rtol=1e-6, atol=1e-8)
                if isinstance(value, (int, float, list)) and not isinstance(value, bool)
                else current == value
            )
            if not same:
                self.controls.set(key, value)
        after = {key: self.controls.get(key) for key in self.profile["renderer_controls"]}
        (Path(output) / "startup-renderer-controls.json").write_text(
            json.dumps(
                dict(
                    before=before,
                    expected=self.profile["renderer_controls"],
                    after=after,
                    scope="Startup only; original strict response and pixel checks remain",
                ),
                default=str,
                indent=2,
            )
        )
        self.expected_attributes = []
        for name, role in self.names.items():
            response = self.profile["cameras"][name]
            prim = sim.env.stage.GetPrimAtPath(self.reference_paths[role])
            prim.AddAppliedSchema("OmniRtxCameraExposureAPI_1")
            for suffix, value in (
                ("time", response["exposure_time_s"]),
                ("iso", self.profile["renderer_controls"]["/rtx/post/tonemap/filmIso"]),
                ("fStop", self.profile["renderer_controls"]["/rtx/post/tonemap/fNumber"]),
                ("responsivity", self.controls.get("/rtx/post/tonemap/cm2Factor")),
            ):
                assert value is not None
                attr = prim.CreateAttribute("exposure:" + suffix, Sdf.ValueTypeNames.Float)
                attr.Set(value)
                self.expected_attributes.append((attr, value))
            product_prim = sim.env.stage.GetPrimAtPath(self.products[role].path)
            attr = product_prim.CreateAttribute(
                "omni:rtx:post:tonemap:whitepoint", Sdf.ValueTypeNames.Color3f
            )
            attr.Set(Gf.Vec3f(*response["whitepoint"]))
            self.expected_attributes.append((attr, response["whitepoint"]))
        self._verify_response()

    def _wait_for_exposure(self, *, timeout_s=240.0, max_steps=32):
        """Drive cold renderer initialization without advancing simulation time."""
        key = "/rtx/post/tonemap/cm2Factor"
        started = time.monotonic()
        steps = 0
        pending = self.controls.get(key) is None
        try:
            if pending:
                # Disabled products cannot demand the cold exposure kernels.
                for product in self.products.values():
                    product.hydra_texture.set_updates_enabled(True)
            while self.controls.get(key) is None:
                if steps >= max_steps or time.monotonic() - started >= timeout_s:
                    raise TimeoutError(f"Renderer exposure initialization timed out: {key}")
                self._step(2)
                steps += 1
        finally:
            if pending:
                for product in self.products.values():
                    product.hydra_texture.set_updates_enabled(False)
            (Path(self.output) / "renderer-readiness.json").write_text(
                json.dumps(
                    dict(
                        ready=self.controls.get(key) is not None,
                        key=key,
                        value=self.controls.get(key),
                        render_steps=steps,
                        elapsed_s=time.monotonic() - started,
                        timeout_s=timeout_s,
                        max_steps=max_steps,
                    ),
                    indent=2,
                )
                + "\n"
            )

    def __call__(self):
        from pxr import UsdGeom  # pyright: ignore[reportMissingImports]

        timings = {}
        stamp = time.perf_counter()
        before = self._before_capture()
        self._verify_response()
        self._pose()
        cache = UsdGeom.XformCache()
        for role in self.names.values():
            camera = self.cameras[role]
            self.reference_xforms[role].Set(
                cache.GetLocalToWorldTransform(self.sim.env.stage.GetPrimAtPath(camera.prim_path))
            )
        timings["state_checks_and_camera_pose_s"] = time.perf_counter() - stamp
        images = {}
        initial = not self.history_ready
        for roles, repeats, subframes in self.schedule:
            if not set(roles).intersection(self.products):
                continue
            self.active_render_roles = roles
            stamp = time.perf_counter()
            self._activate_roles(roles)
            self._apply_masks(roles)
            for _ in range(32 if initial else repeats):
                self._step(subframes)
            label = "+".join(roles)
            timings[label + "_render_s"] = time.perf_counter() - stamp
            stamp = time.perf_counter()
            for name, role in self.names.items():
                if role in roles:
                    images[name] = self._read_rgb(name)
            timings[label + "_readback_s"] = time.perf_counter() - stamp
        stamp = time.perf_counter()
        self._apply_masks(())
        self._activate_roles(())
        after = self._after_capture()
        assert before.time == after.time
        for field in ("qpos", "qvel", "blocks", "block_velocities", "eef_world"):
            assert np.array_equal(getattr(before, field), getattr(after, field)), field
        for name, angle in self.sim.settings.get("camera_rotation_corrections", {}).items():
            if name in images and angle == 180:
                images[name] = self._rotate_rgb(images[name])
        self.frames += 1
        self.history_ready = True
        timings["after_checks_and_rotation_s"] = time.perf_counter() - stamp
        timings["initial_warmup"] = initial
        if self.profile_timings:
            self.frame_timings.append(timings)
        return images

    def _before_capture(self):
        return self.sim.read()

    def _after_capture(self):
        return self.sim.read()

    def _read_rgb(self, name):
        rgba = np.asarray(self.annotators[name].get_data()).copy()
        assert rgba.shape[:2] == self.resolution[::-1], (name, rgba.shape)
        rgb = rgba[:, :, :3].astype(np.uint8)
        assert rgb.mean() > 5 and rgb.std() > 5, name
        return rgb

    def _activate_roles(self, roles):
        for role, product in self.products.items():
            product.hydra_texture.set_updates_enabled(role in roles)

    def _apply_masks(self, roles):
        for role, attrs in self.masks.items():
            for attr in attrs:
                attr.Set(role in roles)

    def _rotate_rgb(self, rgb):
        return np.ascontiguousarray(np.rot90(rgb, 2))

    def report(self):
        return {
            "mode": "ordinary_rtx_cpu_rgb",
            "frames": self.frames,
            "render_steps": self.steps,
            "persistent_products": len(self.products),
            "capture_on_play": False,
            "minimum_block_depth_mm": self.minimum_depth,
            "independent_checks": self.comparisons,
            "preset": self.preset,
            "schedule": self.schedule,
            "renderer_controls": self.profile["renderer_controls"],
            "resolution": self.resolution,
            "frame_timings": self.frame_timings,
            "response_and_pose_checks": "All frames; saved V3 poses/exposure, independent first-frame comparison against validated tiled render.",
        }

    def close(self):
        import omni.replicator.core as rep  # pyright: ignore[reportMissingImports]

        for name, role in self.names.items():
            self.annotators[name].detach(self.products[role].path)
            self.products[role].destroy()
        for role, limits in self.original_ranges.items():
            self.cameras[role].set_clipping_range(*limits)
        self.sim.env.config.wrist_near_clip_mm = self.original_near
        for role, camera in self.cameras.items():
            if not self.original_pause[role]:
                camera.resume()
        for key, value in self.previous_controls.items():
            if value is not None:
                self.controls.set(key, value)
        rep.orchestrator.set_capture_on_play(self.previous_capture_on_play)
