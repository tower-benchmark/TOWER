"""Shared simulation DTOs and evaluator protocols, without generation imports."""

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

ARM_ORDER = ("left", "back")
JOINT_NAMES = tuple(f"joint{i}" for i in range(1, 8)) + ("gripper",)
CAMERAS = ("left_back_camera", "right_front_camera", "left_wrist_camera", "back_wrist_camera")
DT = 1.0 / 30.0

TASK_ALIASES = {
    alias: (f"tower_transfer_{layers}", layers)
    for layers in (2, 3)
    for alias in (f"task1_{layers}", f"task1-{layers}", f"tower_transfer_{layers}")
}


def resolve_task(name: str, layers: int | None = None) -> tuple[str, int]:
    if name not in TASK_ALIASES:
        raise ValueError(f"unknown Task 1 variant {name!r}; choose {sorted(TASK_ALIASES)}")
    canonical, expected = TASK_ALIASES[name]
    if layers is not None and layers != expected:
        raise ValueError(f"{canonical} requires layers-above-base={expected}")
    return canonical, expected


def digest(value: dict) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@dataclass(frozen=True)
class Block:
    name: str
    tower: str
    layer: int
    slot: int
    pose: tuple[float, ...]
    asset_reference: str | None = None
    dimensions_m: tuple[float, ...] | None = None
    asset_scale: tuple[float, ...] | None = None
    mass_kg: float | None = None
    diagonal_inertia_kg_m2: tuple[float, ...] | None = None
    texture_variant: int | None = None
    # Foundation blocks remain colliders, but are attached to the tabletop
    # rather than being dynamic bodies that can be pushed during a grasp.
    fixed_on_table: bool = False


@dataclass(frozen=True)
class Transfer:
    block_index: int
    block_id: str
    source_layer: int
    destination_layer: int
    arm: int
    goal: tuple[float, ...]


@dataclass
class EpisodeSpec:
    logical_id: int
    seed: int
    blocks: list[Block]
    transfers: list[Transfer]
    layout_metadata: dict = field(default_factory=dict)


@dataclass
class State:
    time: float
    qpos: np.ndarray
    qvel: np.ndarray
    effort: np.ndarray
    eef: np.ndarray
    eef_world: np.ndarray
    gripper: np.ndarray
    blocks: np.ndarray
    block_velocities: np.ndarray


@dataclass
class Evaluation:
    success: bool = False
    failure_reason: str = ""
    collapse: bool = False
    block_progress: float = 0.0
    layer_progress: float = 0.0
    completed: list[str] = field(default_factory=list)


class Environment(Protocol):
    def reset(self, env_ids: list[int], episode_specs: list[EpisodeSpec]) -> None: ...
    def step(self, actions: np.ndarray) -> list[State]: ...


class Recorder(Protocol):
    def append(self, frame: dict) -> None: ...
    def finalize(self, outcome: Evaluation) -> Path: ...


class Evaluator(Protocol):
    def update(
        self,
        state: State,
        active_block: int | None,
        released: set[int],
        controller_failure: str = "",
        *,
        controller_done: bool = True,
    ) -> Evaluation: ...


class EvaluatorConfig(Protocol):
    placement_m: float
    placement_degrees: float
    settle_seconds: float
    final_settle_seconds: float
    collapse_drop_m: float
    collapse_degrees: float
    collapse_seconds: float
    max_seconds: float
