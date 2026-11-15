"""Shared methods actually inherited by the ordinary near10 capture.

The legacy tiled-only constructor is not on the benchmark path. There is one
static inheritance chain; no class substitution or captured asset globals.
"""

from collections.abc import Callable
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


class Native61Capture:
    # SDK objects are lazy, dynamically supplied boundaries; the capture itself
    # has an explicit type and shared state contract.
    sim: Any
    cameras: dict[str, Any]
    controls: Any
    profile: dict[str, Any]
    expected_attributes: list[tuple[Any, Any]]
    near_guard: Callable[..., float]
    minimum_depth: dict[str, float]
    steps: int
    history_ready: bool

    names = {
        "right_front_camera": "high",
        "left_back_camera": "low",
        "left_wrist_camera": "left_wrist",
        "back_wrist_camera": "back_wrist",
    }

    def _step(self, subframes):
        import omni.replicator.core as rep  # pyright: ignore[reportMissingImports]

        rep.orchestrator.step(
            rt_subframes=subframes, delta_time=0.0, pause_timeline=True, wait_for_render=True
        )
        self.steps += 1

    def _verify_response(self):
        if self.controls.get("/omni/replicator/captureOnPlay") is not False:
            raise ValueError("Automatic capture-on-play is enabled")
        for key, expected in self.profile["renderer_controls"].items():
            actual = self.controls.get(key)
            same = (
                np.allclose(actual, expected, rtol=1e-6, atol=1e-8)
                if isinstance(expected, (int, float, list)) and not isinstance(expected, bool)
                else actual == expected
            )
            if not same:
                raise ValueError("Native rendering control drift: " + key)
        for attr, expected in self.expected_attributes:
            if not np.allclose(attr.Get(), expected, rtol=1e-6, atol=1e-8):
                raise ValueError("Native camera response drift: " + str(attr.GetPath()))

    def _pose(self):
        env = self.sim.env
        env.physics_output.verify_capture()
        for arm, role in enumerate(("left_wrist", "back_wrist")):
            relative = np.asarray(
                env.report["wrist_camera_proxies"][role]["camera_to_gripper_matrix"]
            )
            tool = env.link_poses[arm][0, env.robots[arm].body_names.index("gripper_base")]
            rotation = Rotation.from_quat(tool[3:])
            xyz = tool[:3] + rotation.apply(relative[3, :3] * 1000)
            quaternion = (rotation * Rotation.from_matrix((relative[:3, :3] * 1000).T)).as_quat()
            camera = self.cameras[role]
            K = np.asarray(env.camera_report[0][role]["intrinsic_matrix_k"])
            depth = self.near_guard(
                env._latest_block_poses[0],
                env.offsets[0],
                xyz,
                quaternion,
                K,
                camera.get_clipping_range()[0],
                "native61_v2_2:" + role,
            )
            self.minimum_depth[role] = min(depth, self.minimum_depth.get(role, float("inf")))
            camera.set_world_pose(xyz, np.roll(quaternion, 1), camera_axes="usd")

    def invalidate_history(self):
        self.history_ready = False
