"""Independent ordinary render products posed by the original serial source.

This is a matched-state camera check, never an independent policy trajectory.
Keep products attached until simulator shutdown to preserve SDK tensor handles.
"""

from __future__ import annotations

import hashlib
import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import numpy as np

from .camera_guard import TerminalGuard
from .observation_validity import compare_pixels as compare_pixels


def physical_snapshot(states):
    return [
        {
            field: np.asarray(getattr(state, field)).tolist()
            for field in ("time", "qpos", "qvel", "blocks", "block_velocities", "eef_world")
        }
        for state in states
    ]


class SerialCameraReference:
    """Render the selected environment through the source serial SDK camera path."""

    def __init__(self, camera, environment_index=0, *, guard: TerminalGuard):
        self.guard = guard
        self.camera = camera
        if type(environment_index) is not int or not 0 <= environment_index < camera.sim.env.n:
            raise ValueError("Invalid source renderer environment index")
        self.environment_index = environment_index
        self.sim = camera.sim
        self.products, self.annotators, self.xforms = {}, {}, {}
        self.own_attributes = []
        self.source = None
        self.created = False

    def _pose(self):
        from pxr import UsdGeom  # pyright: ignore[reportMissingImports]

        from .four_capture import Native61Capture

        env = self.sim.env
        index = self.environment_index
        # Source function indexes row0; provide a strictly indexed view of the
        # actual corresponding tensor state, never a recalculated FK estimate.
        indexed_env = SimpleNamespace(
            physics_output=env.physics_output,
            report=env.report,
            link_poses=[values[index : index + 1] for values in env.link_poses],
            robots=env.robots,
            camera_report=[env.camera_report[index]],
            _latest_block_poses=env._latest_block_poses[index : index + 1],
            offsets=env.offsets[index : index + 1],
        )
        proxy = SimpleNamespace(
            sim=SimpleNamespace(env=indexed_env),
            cameras=env.cameras[index],
            minimum_depth={},
            near_guard=self.guard.check,
        )
        # Exact original generator pose implementation, independent of the
        # tiled subclass's legacy pose path and copied reference transforms.
        with self.guard.reference_environment(index):
            Native61Capture._pose(cast(Native61Capture, proxy))
        source = inspect.getsourcefile(Native61Capture._pose)
        if source is None:
            raise RuntimeError("Serial pose source unavailable")
        self.source = Path(source).resolve()
        cache = UsdGeom.XformCache()
        transforms = {}
        for name, role in self.camera.names.items():
            source_camera = self.sim.env.cameras[self.environment_index][role]
            matrix = cache.GetLocalToWorldTransform(
                self.sim.env.stage.GetPrimAtPath(source_camera.prim_path)
            )
            transforms[name] = np.asarray(matrix).tolist()
            if self.created:
                self.xforms[role].Set(matrix)
        return transforms

    def _create(self):
        import omni.replicator.core as rep  # pyright: ignore[reportMissingImports]
        from pxr import Gf, Sdf, UsdGeom  # pyright: ignore[reportMissingImports]

        env = self.sim.env
        root = f"/__V3IndependentSerialSourceCameras_{self.environment_index}"
        if env.stage.GetPrimAtPath(root):
            raise ValueError("Independent source camera namespace already exists")
        UsdGeom.Xform.Define(env.stage, root)
        flattened = env.stage.Flatten()
        for name, role in self.camera.names.items():
            target = root + "/" + role
            if not Sdf.CopySpec(
                flattened,
                env.cameras[self.environment_index][role].prim_path,
                env.stage.GetRootLayer(),
                target,
            ):
                raise ValueError("Cannot copy original source camera")
            prim = env.stage.GetPrimAtPath(target)
            self.xforms[role] = UsdGeom.Xformable(prim).MakeMatrixXform()
            product = rep.create.render_product(
                target, (320, 240), name=f"v3_serial_source_{self.environment_index}_" + role
            )
            annotator = rep.AnnotatorRegistry.get_annotator("rgb", device="cpu", do_array_copy=True)
            annotator.attach(product.path)
            product.hydra_texture.set_updates_enabled(False)
            self.products[role], self.annotators[name] = product, annotator
        self.created = True
        self._pose()
        rep.orchestrator.step(
            rt_subframes=2, delta_time=0.0, pause_timeline=True, wait_for_render=True
        )
        for name, role in self.camera.names.items():
            response = self.camera.profile["cameras"][name]
            prim = env.stage.GetPrimAtPath(root + "/" + role)
            prim.AddAppliedSchema("OmniRtxCameraExposureAPI_1")
            for suffix, value in (
                ("time", response["exposure_time_s"]),
                ("iso", self.camera.profile["renderer_controls"]["/rtx/post/tonemap/filmIso"]),
                ("fStop", self.camera.profile["renderer_controls"]["/rtx/post/tonemap/fNumber"]),
                ("responsivity", self.camera.controls.get("/rtx/post/tonemap/cm2Factor")),
            ):
                if value is None:
                    raise ValueError("Missing source renderer exposure")
                attr = prim.CreateAttribute("exposure:" + suffix, Sdf.ValueTypeNames.Float)
                self.own_attributes.append((attr, value))
            attr = env.stage.GetPrimAtPath(self.products[role].path).CreateAttribute(
                "omni:rtx:post:tonemap:whitepoint", Sdf.ValueTypeNames.Color3f
            )
            self.own_attributes.append((attr, response["whitepoint"]))
        for attr, value in [*self.camera.expected_attributes, *self.own_attributes]:
            attr.Set(
                Gf.Vec3f(*value) if attr.GetTypeName() == Sdf.ValueTypeNames.Color3f else value
            )
        self.camera._verify_response()

    def observe(self):
        import omni.replicator.core as rep  # pyright: ignore[reportMissingImports]
        from pxr import Gf, Sdf  # pyright: ignore[reportMissingImports]

        from .render_freshness import verified_step

        self._pose()
        for product in self.camera.products.values():
            product.hydra_texture.set_updates_enabled(False)
        try:
            if not self.created:
                self._create()
            transforms = self._pose()
            for attr, value in [*self.camera.expected_attributes, *self.own_attributes]:
                attr.Set(
                    Gf.Vec3f(*value) if attr.GetTypeName() == Sdf.ValueTypeNames.Color3f else value
                )
            self.camera._verify_response()
            self.camera._apply_masks(tuple(self.camera.names.values()))
            for product in self.products.values():
                product.hydra_texture.set_updates_enabled(True)
            deliveries = []
            for _ in range(32):
                deliveries.append(
                    verified_step(
                        lambda: rep.orchestrator.step(
                            rt_subframes=6,
                            delta_time=0.0,
                            pause_timeline=True,
                            wait_for_render=True,
                        )
                    )
                )
            images = []
            for name in self.camera.names:
                rgba = np.asarray(self.annotators[name].get_data(device="cpu", do_array_copy=True))
                if rgba.shape != (240, 320, 4) or rgba.dtype != np.uint8:
                    raise ValueError("Invalid independent source renderer RGB format")
                rgb = rgba[..., :3].copy()
                if self.sim.settings.get("camera_rotation_corrections", {}).get(name) == 180:
                    rgb = np.ascontiguousarray(np.rot90(rgb, 2))
                images.append(rgb)
            assert self.source is not None
            receipt = dict(
                representative_environment=self.environment_index,
                source_pose_function="Native61Capture._pose",
                source_path=str(self.source),
                source_sha256=hashlib.sha256(self.source.read_bytes()).hexdigest(),
                source_camera_world_transforms=transforms,
                renderer_controls=self.camera.profile["renderer_controls"],
                resolution=[320, 240],
                completion="32 synchronous Replicator steps, six subframes, wait_for_render",
                render_deliveries=deliveries,
                independent_ordinary_render_products=True,
                same_physics_state_required=True,
            )
            return np.stack(images), receipt
        finally:
            for product in self.products.values():
                product.hydra_texture.set_updates_enabled(False)
            for product in self.camera.products.values():
                product.hydra_texture.set_updates_enabled(True)
