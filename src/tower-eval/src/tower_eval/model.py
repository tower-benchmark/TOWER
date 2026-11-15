# SPDX-License-Identifier: Apache-2.0
"""Small public data model used by scene verification and result tooling."""

from __future__ import annotations

import math
from dataclasses import dataclass, field


def _positive_finite_triplet(value: tuple[float, ...] | None, field_name: str) -> None:
    if value is None:
        return
    if len(value) != 3 or any(not math.isfinite(float(item)) or float(item) <= 0 for item in value):
        raise ValueError(f"{field_name} must contain three finite positive values")


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
    fixed_on_table: bool = False

    def __post_init__(self) -> None:
        if len(self.pose) != 7 or any(not math.isfinite(float(item)) for item in self.pose):
            raise ValueError("pose must contain seven finite values (xyz + quaternion)")
        _positive_finite_triplet(self.dimensions_m, "dimensions_m")
        _positive_finite_triplet(self.asset_scale, "asset_scale")
        _positive_finite_triplet(self.diagonal_inertia_kg_m2, "diagonal_inertia_kg_m2")
        if self.mass_kg is not None and (
            not math.isfinite(float(self.mass_kg)) or self.mass_kg <= 0
        ):
            raise ValueError("mass_kg must be finite and positive")
        if self.texture_variant is not None and (
            isinstance(self.texture_variant, bool)
            or not isinstance(self.texture_variant, int)
            or self.texture_variant < 0
        ):
            raise ValueError("texture_variant must be a non-negative integer")


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
