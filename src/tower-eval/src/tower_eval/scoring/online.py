# SPDX-License-Identifier: Apache-2.0
"""Pose-based online stack scoring; thresholds come from self.protocol."""

from __future__ import annotations

import numpy as np

from .constants import BLOCK_SIZE_M, GRIPPER_CHANNELS
from .geometry import angle, area, geometry, inside, intersection, polygon


class StackScorer:
    """Require a released, stable two-block-per-layer destination stack.

    Overhang/yaw are allowed. COM must project inside the base and the convex
    hull of contact patches, with >=20% footprint support and <=8 mm vertical
    separation. This is a geometric diagnostic, not a measured contact-force test.
    """

    protocol = {
        "name": "supported_two_layer_stack_v1",
        "dwell_s": 3.0,
        "support_height_tolerance_m": 0.008,
        "minimum_support_fraction": 0.2,
        "support_com_margin_m": 0.002,
        "linear_velocity_m_s": 0.005,
        "angular_velocity_rad_s": 0.05,
        "anchor_translation_m": 0.002,
        "anchor_rotation_deg": 2.0,
        "open_gripper_width_m": 0.075,
        "strict_position_m": 0.01,
        "strict_rotation_deg": 5.0,
        "required_blocks_per_layer": [2, 2],
    }

    def __init__(self, spec):
        self.spec = spec
        self.movable = [t.block_index for t in spec.transfers]
        self.foundation = [
            i for i, b in enumerate(spec.blocks) if b.tower == "destination" and b.fixed_on_table
        ]
        destination_layers = [t.destination_layer for t in spec.transfers]
        self.layer_count = max(destination_layers, default=0)
        self.required_blocks_per_layer = [
            destination_layers.count(level) for level in range(1, self.layer_count + 1)
        ]
        if (
            not self.foundation
            or not self.movable
            or sorted(set(destination_layers)) != list(range(1, self.layer_count + 1))
            or any(
                count != (1 if getattr(spec, "layout_metadata", {}).get("catalog_task") == 2 else 2)
                for count in self.required_blocks_per_layer
            )
        ):
            raise ValueError(
                "Stack scoring requires the task-specific block count per layer and a foundation"
            )
        self.protocol = {
            **type(self).protocol,
            "name": f"supported_{self.layer_count}_layer_stack_v1",
            "required_blocks_per_layer": self.required_blocks_per_layer,
        }
        self.dimensions = [b.dimensions_m or BLOCK_SIZE_M for b in spec.blocks]
        self.anchor = None
        self.last_time = None
        self.dwell = self.strict_dwell = 0.0
        self.max_supported = self.max_strict = 0
        self.result = {
            "success": False,
            "strict_success": False,
            "supported_blocks": 0,
            "strict_blocks": 0,
            "layers": {},
            "stable_seconds": 0.0,
        }

    def update(self, state):
        protocol = self.protocol
        poses = np.asarray(state.blocks, dtype=float)
        velocity = np.asarray(state.block_velocities, dtype=float)
        if (
            poses.shape != (len(self.spec.blocks), 7)
            or not np.isfinite(poses).all()
            or not np.isfinite(velocity).all()
            or velocity.shape != (len(poses), 6)
        ):
            raise ValueError("Invalid block state for scoring")
        if not np.isfinite(state.time) or (
            self.last_time is not None and state.time < self.last_time
        ):
            raise ValueError("Invalid scoring clock")
        dt = 0.0 if self.last_time is None else state.time - self.last_time
        self.last_time = state.time
        geo = [geometry(p, d) for p, d in zip(poses, self.dimensions)]
        base = polygon(np.vstack([geo[i][0] for i in self.foundation]))
        region_blocks = [
            i for i in self.movable if inside(poses[i, :2], base, protocol["support_com_margin_m"])
        ]
        stationary_region_blocks = [
            i
            for i in region_blocks
            if np.linalg.norm(velocity[i, :3]) <= protocol["linear_velocity_m_s"]
            and np.linalg.norm(velocity[i, 3:]) <= protocol["angular_velocity_rad_s"]
        ]
        levels, support_details = {}, {}
        supporters = list(self.foundation)
        for level in range(1, self.layer_count + 1):
            placed = []
            for i in self.movable:
                if i in levels or not inside(poses[i, :2], base, protocol["support_com_margin_m"]):
                    continue
                foot, bottom, _ = geo[i]
                patches = [
                    intersection(foot, geo[j][0])
                    for j in supporters
                    if abs(bottom - geo[j][2]) <= protocol["support_height_tolerance_m"]
                ]
                patches = [p for p in patches if area(p) > 1e-9]
                if not patches:
                    continue
                fraction = min(1.0, sum(area(p) for p in patches) / area(foot))
                com_supported = inside(
                    poses[i, :2], polygon(np.vstack(patches)), protocol["support_com_margin_m"]
                )
                if fraction >= protocol["minimum_support_fraction"] and com_supported:
                    levels[i] = level
                    support_details[i] = {"level": level, "support_fraction": fraction}
                    placed.append(i)
            supporters = placed
        strict_count = sum(
            np.linalg.norm(poses[t.block_index, :3] - t.goal[:3]) <= protocol["strict_position_m"]
            and angle(poses[t.block_index, 3:], t.goal[3:])
            <= np.deg2rad(protocol["strict_rotation_deg"])
            for t in self.spec.transfers
        )
        moving = self.movable
        stationary = (
            np.linalg.norm(velocity[moving, :3], axis=1).max() <= protocol["linear_velocity_m_s"]
            and np.linalg.norm(velocity[moving, 3:], axis=1).max()
            <= protocol["angular_velocity_rad_s"]
        )
        if self.anchor is None or any(
            np.linalg.norm(poses[i, :3] - self.anchor[i, :3]) > protocol["anchor_translation_m"]
            or angle(poses[i, 3:], self.anchor[i, 3:]) > np.deg2rad(protocol["anchor_rotation_deg"])
            for i in moving
        ):
            self.anchor = poses.copy()
            stationary = False
        grippers = np.asarray(state.qpos)[GRIPPER_CHANNELS]
        released = bool(
            np.isfinite(grippers).all() and np.all(grippers >= protocol["open_gripper_width_m"])
        )
        layer_counts = [
            sum(x == level for x in levels.values()) for level in range(1, self.layer_count + 1)
        ]
        stack_valid = (
            layer_counts == protocol["required_blocks_per_layer"] and released and stationary
        )
        self.dwell = self.dwell + dt if stack_valid else 0.0
        self.strict_dwell = (
            self.strict_dwell + dt
            if strict_count == len(self.movable) and released and stationary
            else 0.0
        )
        self.max_supported = max(self.max_supported, len(levels))
        self.max_strict = max(self.max_strict, int(strict_count))
        self.result = {
            "success": self.dwell + 1e-9 >= protocol["dwell_s"],
            "strict_success": self.strict_dwell + 1e-9 >= protocol["dwell_s"],
            "destination_region_blocks": len(region_blocks),
            "destination_region_block_indices": region_blocks,
            "stationary_destination_region_blocks": len(stationary_region_blocks),
            "supported_blocks": len(levels),
            "strict_blocks": int(strict_count),
            "max_supported_blocks": self.max_supported,
            "max_strict_blocks": self.max_strict,
            "layers": levels,
            "support": support_details,
            "layer_counts": layer_counts,
            "released": released,
            "stationary": bool(stationary),
            "stable_seconds": self.dwell,
            "strict_stable_seconds": self.strict_dwell,
        }
        return self.result
