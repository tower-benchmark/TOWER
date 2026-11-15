# SPDX-License-Identifier: Apache-2.0
"""Render-product scheduling for existing calibrated camera prims."""

import math

import numpy as np


class IndividualCameraPasses:
    """Keep each camera's temporal history in its own visibility pass."""

    def __init__(self, camera_batches):
        import omni.replicator.core as rep

        self.products = []
        for cameras in camera_batches:
            for name, camera in cameras.items():
                product = rep.create.render_product(
                    camera.prim_path, resolution=(640, 480), force_new=False
                )
                if product.path != camera.get_render_product_path():
                    raise RuntimeError(
                        f"Could not reuse the existing camera render product: {name}"
                    )
                self.products.append((name, product))
        self.prepare(())

    def prepare(self, names):
        selected = set(names)
        for name, product in self.products:
            product.hydra_texture.set_updates_enabled(name in selected)


def untile_rgba(image: np.ndarray, count: int, width: int = 640, height: int = 480) -> np.ndarray:
    if image.ndim != 3 or image.shape[2] != 4 or image.shape[0] % height or image.shape[1] % width:
        raise ValueError(f"Unexpected tiled RGBA dimensions: {image.shape}")
    rows, columns = image.shape[0] // height, image.shape[1] // width
    if not 0 < count <= rows * columns:
        raise ValueError("Tiled RGBA buffer has insufficient camera slots")
    return (
        image.reshape(rows, height, columns, width, 4)
        .transpose(0, 2, 1, 3, 4)
        .reshape(rows * columns, height, width, 4)[:count]
    )


class TiledRgbaCapture:
    def __init__(self, camera_batches, names=None, suffix="all"):
        import carb
        import omni.replicator.core as rep
        import omni.usd
        from isaacsim.core.utils.render_product import get_resolution

        from tower_sim.replay.dual_core import CAMERA_NAMES

        self.names = CAMERA_NAMES if names is None else names
        self.count = len(camera_batches)
        paths = [cameras[name].prim_path for cameras in camera_batches for name in self.names]
        render_paths = list(paths)
        padding_count = math.ceil(math.sqrt(len(paths))) ** 2 - len(paths)
        stage = omni.usd.get_context().get_stage()
        # Calibrated lens projection textures in Kit 107 distort rectangular
        # tile grids. Distinct, discarded camera views make the grid square.
        for index in range(padding_count):
            path = f"/__JengaTiledPadding_{suffix}/Camera_{index}"
            prim = stage.DefinePrim(path, "Camera")
            prim.GetReferences().AddInternalReference(paths[0])
            render_paths.append(path)
        settings = carb.settings.get_settings()
        settings.set_int(
            "/rtx/viewTile/limit",
            max(len(render_paths), settings.get_as_int("/rtx/viewTile/limit")),
        )
        self.product = rep.create.render_product_tiled(
            cameras=render_paths,
            tile_resolution=(640, 480),
            name=f"jenga_calibrated_camera_tiles_{suffix}",
        )
        self.annotator = rep.AnnotatorRegistry.get_annotator(
            "rgb", device="cpu", do_array_copy=True
        )
        self.annotator.attach(self.product.path)
        self.resolution = tuple(get_resolution(self.product.path))
        self.report = {
            "capture_api": "Replicator tiled RGB annotator",
            "camera_paths": paths,
            "padding_camera_paths": render_paths[len(paths) :],
            "square_projection_grid": True,
            "tile_resolution": [640, 480],
            "render_product_resolution": list(self.resolution),
            "render_product": self.product.path,
            "camera_order": "environment-major, reference camera order",
        }

    def read(self):
        data = self.annotator.get_data()
        image = np.asarray(data["data"] if isinstance(data, dict) else data)
        expected = (self.resolution[1], self.resolution[0], 4)
        if image.shape != expected:
            raise RuntimeError(f"Tiled render returned {image.shape}, expected {expected}")
        tiles = untile_rgba(image, self.count * len(self.names))
        return [
            {name: tiles[env * len(self.names) + i] for i, name in enumerate(self.names)}
            for env in range(self.count)
        ]


class SplitTiledRgbaCapture:
    """Render only the calibrated views needed by each visibility pass."""

    def __init__(self, camera_batches):
        self.count = len(camera_batches)
        self.groups = [
            TiledRgbaCapture(camera_batches, ("low", "high"), "fixed"),
            TiledRgbaCapture(camera_batches, ("left_wrist", "back_wrist"), "wrist"),
        ]
        self.active = []
        self.report = {
            "capture_api": "Replicator split tiled RGB annotators",
            "render_product": [group.product.path for group in self.groups],
            "groups": [group.report for group in self.groups],
        }
        self.prepare(("low", "high"))

    def prepare(self, names):
        self.active = []
        for group in self.groups:
            enabled = bool(set(group.names).intersection(names))
            group.product.hydra_texture.set_updates_enabled(enabled)
            if enabled:
                self.active.append(group)

    def read(self):
        frames = [{} for _ in range(self.count)]
        for group in self.active:
            for frame, captured in zip(frames, group.read()):
                frame.update(captured)
        return frames
