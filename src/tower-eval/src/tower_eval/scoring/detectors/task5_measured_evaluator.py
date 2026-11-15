"""Measured Task5 adaptation of historical InsertionEvaluator.
Same Task5 tolerances and dwell/latched failure semantics; receiving tower source
naming and measured side-wall extents replace historical destination/50mm boxes.
Original TransferEvaluator is imported unchanged; no simulation/controller work.
"""

from dataclasses import replace

import numpy as np

from ..geometry import Rotation


class InsertionEvaluator:
    def __init__(self, config, spec):
        from .task5_dependencies.tasks import TransferEvaluator

        self.config, self.spec = config, spec
        self.transfer = TransferEvaluator(config, spec)
        self.structural = [
            i
            for i, b in enumerate(spec.blocks)
            if b.tower == "source" and i not in {t.block_index for t in spec.transfers}
        ]
        self.anchors = np.asarray([b.pose for b in spec.blocks], dtype=float)
        self.last_time = None
        self.structure_dwell = 0.0
        self.measurements = {}
        self.structural_bad_dwell = 0.0
        self.structural_failure = False

    def update(
        self,
        state,
        active_block,
        released,
        controller_failure="",
        *,
        controller_done=True,
    ):
        result = self.transfer.update(
            state,
            active_block,
            released,
            controller_failure,
            controller_done=controller_done,
        )
        dt = 0 if self.last_time is None else max(0.0, state.time - self.last_time)
        self.last_time = state.time
        expected = np.asarray([b.pose for b in self.spec.blocks])
        positions = np.linalg.norm(
            state.blocks[self.structural, :3] - expected[self.structural, :3], axis=1
        )
        angles = (
            Rotation.from_quat(expected[self.structural, 3:]).inv()
            * Rotation.from_quat(state.blocks[self.structural, 3:])
        ).magnitude()
        anchor_move = np.linalg.norm(
            state.blocks[self.structural, :3] - self.anchors[self.structural, :3],
            axis=1,
        )
        anchor_angle = (
            Rotation.from_quat(self.anchors[self.structural, 3:]).inv()
            * Rotation.from_quat(state.blocks[self.structural, 3:])
        ).magnitude()
        stable = bool(
            np.all(positions <= self.config.placement_m)
            and np.all(angles <= np.deg2rad(self.config.placement_degrees))
            and np.all(anchor_move <= 0.0005)
            and np.all(anchor_angle <= np.deg2rad(0.5))
        )
        velocities = np.asarray(state.block_velocities)[self.structural]
        stable &= bool(
            np.all(np.linalg.norm(velocities[:, :3], axis=1) <= 0.01)
            and np.all(np.linalg.norm(velocities[:, 3:], axis=1) <= 0.1)
        )
        if not stable:
            self.anchors[self.structural] = state.blocks[self.structural]
        self.structure_dwell = self.structure_dwell + dt if stable else 0.0
        manipulated = active_block is not None or bool(released)
        disturbed = bool(
            np.any(positions > self.config.placement_m)
            or np.any(angles > np.deg2rad(self.config.placement_degrees))
        )
        self.structural_bad_dwell = (
            self.structural_bad_dwell + dt if manipulated and disturbed else 0.0
        )
        self.structural_failure |= self.structural_bad_dwell + 1e-9 >= self.config.collapse_seconds
        # Seated insert must also lie inside the actual side-support gap; a
        # near nominal pose alone cannot pass if supports were displaced.
        contained = True
        for t in self.spec.transfers:
            if t.block_index not in released or t.block_index == active_block:
                contained = False
                continue
            block = state.blocks[t.block_index]
            goal_rot = Rotation.from_quat(t.goal[3:])
            local = goal_rot.inv().apply(block[:3] - np.asarray(t.goal[:3]))
            contained &= abs(local[0]) <= self.config.placement_m
            from itertools import product

            dims = np.asarray(self.spec.blocks[t.block_index].dimensions_m)
            corners = np.array(list(product([-1.0, 1.0], repeat=3))) * dims / 2
            actual_corners = Rotation.from_quat(block[3:]).apply(corners) + block[:3]
            insert_y = goal_rot.inv().apply(actual_corners - np.asarray(t.goal[:3]))[:, 1]
            sides = [i for i in self.structural if self.spec.blocks[i].layer == t.destination_layer]
            wall = []
            for i in sides:
                sd = np.asarray(self.spec.blocks[i].dimensions_m)
                sc = np.array(list(product([-1.0, 1.0], repeat=3))) * sd / 2
                sy = goal_rot.inv().apply(
                    Rotation.from_quat(state.blocks[i, 3:]).apply(sc)
                    + state.blocks[i, :3]
                    - np.asarray(t.goal[:3])
                )[:, 1]
                wall.append((float(sy.mean()), float(sy.min()), float(sy.max())))
            wall.sort()
            contained &= (
                len(wall) == 2
                and wall[0][2] <= insert_y.min() + 0.0005
                and insert_y.max() <= wall[1][1] + 0.0005
            )
            from .task5_dependencies.contact_planning import box_penetration

            # Exact oriented boxes reject a tilted side or a low/tilted roof
            # crossing the insert, even when nominal center projections fit.
            for i in self.structural:
                if self.spec.blocks[i].layer < t.destination_layer:
                    continue
                penetration = box_penetration(
                    block,
                    self.spec.blocks[t.block_index].dimensions_m,
                    state.blocks[i],
                    self.spec.blocks[i].dimensions_m,
                )
                contained &= penetration <= 0.0005
        self.measurements = {
            "maximum_support_position_error_m": float(max(positions, default=0)),
            "maximum_support_orientation_error_rad": float(max(angles, default=0)),
            "structural_settle_seconds": self.structure_dwell,
            "insert_containment": bool(contained),
            "structural_bad_dwell_seconds": self.structural_bad_dwell,
            "structural_failure_latched": self.structural_failure,
        }
        failure = result.failure_reason
        if self.structural_failure:
            failure = failure or "insertion_support_disturbance"
        if not failure and controller_done and state.time >= self.config.max_seconds:
            failure = "insertion_structure_or_containment_timeout"
        success = bool(
            result.success
            and not failure
            and contained
            and self.structure_dwell + 1e-9 >= self.config.final_settle_seconds
        )
        return replace(result, success=success, failure_reason=failure)
