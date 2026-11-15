"""Pure Task5 scorer contracts; defaults retain their measured semantics."""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Config:
    task: str = "tower_transfer_3"
    max_seconds: float = 300.0
    placement_m: float = 0.005
    placement_degrees: float = 5.0
    settle_seconds: float = 0.5
    final_settle_seconds: float = 2.0
    collapse_drop_m: float = 0.015
    collapse_degrees: float = 30.0
    collapse_seconds: float = 0.25


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
