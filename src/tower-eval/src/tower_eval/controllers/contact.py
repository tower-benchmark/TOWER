# SPDX-License-Identifier: Apache-2.0
"""Monotonic, contact-limited finger closure; no physical state writes."""

from collections import deque
from dataclasses import dataclass, field
from typing import cast

import numpy as np

from ..constants import CONTROLLER_DT


@dataclass
class ContactClosure:
    width: float
    floor: float
    force_n: float = 1.5
    speed: float = 0.01
    free_speed: float = 0.09
    contact_width: float = 0.05
    acceleration: float = 0.2
    max_preload: float = 0.006
    velocity: float = 0.0
    contact_seconds: float = 0.0
    latched: bool = False
    position_window: deque = field(default_factory=lambda: deque(maxlen=4))

    def update(
        self,
        measured_width,
        finger_forces,
        finger_velocities,
        *,
        finger_positions,
        dt=CONTROLLER_DT,
    ):
        forces = np.abs(np.asarray(finger_forces, dtype=float))
        velocities = np.asarray(finger_velocities, dtype=float)
        positions = np.asarray(finger_positions, dtype=float)
        if (
            forces.shape != (2,)
            or velocities.shape != (2,)
            or positions.shape != (2,)
            or not np.isfinite([measured_width, *forces, *velocities, *positions]).all()
        ):
            raise ValueError("closure needs both measured finger positions, forces and velocities")
        if dt != CONTROLLER_DT:
            raise ValueError("closure observation cadence must be 30 Hz")
        if self.latched:
            return self.width
        # The first loaded finger selects gentle speed below, but cannot stop
        # the opposite jaw before it touches. Brake on the weaker finger's
        # load, then hold once both contacts are stable. The fixed preload
        # bound still limits the initial one-sided squeeze. Never reopen.
        load = min(1.0, float(forces.min()) / self.force_n)
        # Brake before the expected block span, not after hitting it. Both the
        # commanded and observed width must indicate clear space for fast travel.
        distance = max(0.0, min(self.width, measured_width) - self.contact_width - 0.01)
        travel_speed = min(
            self.free_speed, np.sqrt(self.speed**2 + 2 * self.acceleration * distance)
        )
        if forces.max() > 0.1 * self.force_n:
            travel_speed = self.speed
        target_velocity = -travel_speed * max(0.0, 1.0 - load)
        self.velocity += np.clip(
            target_velocity - self.velocity, -self.acceleration * dt, self.acceleration * dt
        )
        self.velocity = min(0.0, self.velocity)
        proposed = max(
            self.floor, measured_width - self.max_preload, self.width + self.velocity * dt
        )
        proposed = min(self.width, proposed)
        self.velocity = (proposed - self.width) / dt
        self.width = proposed
        # PhysX joint velocity readbacks fluctuate under a static preload.
        # Require BOTH actual jaws to stay within 0.25 mm over 0.1 s instead.
        # This checks raw positions; it never smooths observations or commands.
        self.position_window.append(positions.copy())
        jaws_stable = (
            len(self.position_window) == 4
            and np.max(np.ptp(np.asarray(self.position_window), axis=0)) <= 0.00025
        )
        contact = (
            forces.min() >= 0.5 * self.force_n
            and forces.max() >= 0.9 * self.force_n
            and jaws_stable
            and measured_width > self.floor + 0.002
            and measured_width - self.width > 0.0005
        )
        self.contact_seconds = self.contact_seconds + dt if contact else 0.0
        if self.contact_seconds >= 0.1 and abs(self.velocity) < 0.001:
            self.latched = True
            self.velocity = 0.0
        return self.width


class ClosureMotion:
    """Constant-memory raw TCP/block displacement during one close attempt."""

    def __init__(self):
        self.minimum = self.maximum = self.previous = None
        self.step = np.zeros(2)
        self.samples = 0

    def update(self, tcp, block):
        current = np.asarray([tcp, block], dtype=float)
        if self.previous is None:
            self.minimum = current.copy()
            self.maximum = current.copy()
        else:
            self.step = np.maximum(self.step, np.linalg.norm(current - self.previous, axis=1))
            self.minimum = np.minimum(cast(np.ndarray, self.minimum), current)
            self.maximum = np.maximum(cast(np.ndarray, self.maximum), current)
        self.previous = current.copy()
        self.samples += 1

    def report(self):
        drift = (
            np.zeros(2)
            if not self.samples
            else np.linalg.norm(
                cast(np.ndarray, self.maximum) - cast(np.ndarray, self.minimum), axis=1
            )
            * 1000
        )
        return {
            "samples": self.samples,
            "tcp_range_mm": float(drift[0]),
            "block_range_mm": float(drift[1]),
            "tcp_maximum_step_mm": float(self.step[0] * 1000),
            "block_maximum_step_mm": float(self.step[1] * 1000),
            "observations_filtered": False,
        }
