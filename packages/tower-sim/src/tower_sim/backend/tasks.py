"""Shared evaluation geometry and outcomes."""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation
from tower_eval.simulator.spec import EpisodeSpec, Evaluation, State, EvaluatorConfig as Config


def orientation_error(q1: np.ndarray, q2: np.ndarray) -> float:
    delta = Rotation.from_quat(q1).inv() * Rotation.from_quat(q2)
    # A wooden rectangular block is identical after a half-turn around its Z axis.
    flipped = delta * Rotation.from_euler("z", 180, degrees=True)
    return float(min(delta.magnitude(), flipped.magnitude()))


class TransferEvaluator:
    def __init__(self, config: Config, spec: EpisodeSpec):
        self.config, self.spec = config, spec
        self.dwell = np.zeros(len(spec.transfers))
        self.bad_since = np.zeros(len(spec.blocks))
        self.complete_dwell = 0.0
        self.last_time: float | None = None
        self.collapse = False
        self.transferred_once: set[int] = set()
        self.last_block_poses = None
        self.stability_anchors = np.asarray([b.pose for b in spec.blocks])

    def update(self, state: State, active_block: int | None, released: set[int], controller_failure: str = "", *, controller_done: bool = True) -> Evaluation:
        dt = 0.0 if self.last_time is None else max(0.0, state.time - self.last_time)
        self.last_time = state.time
        config = self.config
        expected = np.asarray([b.pose for b in self.spec.blocks])
        self.transferred_once.update(released)
        completed = []
        for i, transfer in enumerate(self.spec.transfers):
            index = transfer.block_index
            # A placement recovery removes the block from `released` while the
            # arm approaches it again.  It still belongs at the destination;
            # comparing it to its old source height would report a false fall.
            if index in self.transferred_once:
                expected[index] = transfer.goal
            actual = state.blocks[index]
            close = np.linalg.norm(actual[:3] - np.asarray(transfer.goal[:3])) <= config.placement_m
            aligned = orientation_error(actual[3:], np.asarray(transfer.goal[3:])) <= np.deg2rad(config.placement_degrees)
            anchor = self.stability_anchors[index]
            stationary = self.last_block_poses is not None and np.linalg.norm(actual[:3] - anchor[:3]) <= 0.0005 and orientation_error(actual[3:], anchor[3:]) <= np.deg2rad(0.5)
            valid = index in released and index != active_block and close and aligned and stationary
            if not valid:
                self.stability_anchors[index] = actual
            self.dwell[i] = self.dwell[i] + dt if valid else 0.0
            if self.dwell[i] + 1e-9 >= config.settle_seconds:
                completed.append(transfer.block_id)
        for index, block in enumerate(state.blocks):
            if index == active_block:
                self.bad_since[index] = 0.0
                continue
            upright = Rotation.from_quat(block[3:]).apply([0, 0, 1])[2]
            bad = expected[index, 2] - block[2] > config.collapse_drop_m or upright < np.cos(np.deg2rad(config.collapse_degrees))
            self.bad_since[index] = self.bad_since[index] + dt if bad else 0.0
        self.collapse |= bool(np.any(self.bad_since + 1e-9 >= config.collapse_seconds))
        all_done = len(completed) == len(self.spec.transfers)
        self.complete_dwell = self.complete_dwell + dt if all_done and active_block is None else 0.0
        finished = self.complete_dwell + 1e-9 >= config.final_settle_seconds and controller_done
        failure = controller_failure
        if self.collapse:
            failure = "collapse"
        elif state.time >= config.max_seconds and not finished:
            failure = failure or "timeout"
        layer_ids = {t.destination_layer for t in self.spec.transfers}
        done_layers = sum(all(t.block_id in completed for t in self.spec.transfers if t.destination_layer == layer) for layer in layer_ids)
        self.last_block_poses = state.blocks.copy()
        return Evaluation(
            success=finished and not failure,
            failure_reason=failure,
            collapse=self.collapse,
            block_progress=len(completed) / len(self.spec.transfers),
            layer_progress=done_layers / len(layer_ids),
            completed=completed,
        )
