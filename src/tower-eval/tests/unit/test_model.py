# SPDX-License-Identifier: Apache-2.0
import pytest
from tower_eval.model import Block


def test_blocks_can_have_task_specific_dimensions_and_physics():
    standard = Block(
        "standard",
        "source",
        0,
        0,
        (0, 0, 0, 0, 0, 0, 1),
        dimensions_m=(0.075, 0.025, 0.015),
        mass_kg=0.012,
        diagonal_inertia_kg_m2=(1e-6, 2e-6, 3e-6),
    )
    resized = Block(
        "resized",
        "destination",
        1,
        0,
        (0, 0, 0, 0, 0, 0, 1),
        asset_reference="blocks/custom/v1/geometry.usd",
        asset_scale=(1.1, 1.0, 0.9),
        dimensions_m=(0.0825, 0.025, 0.0135),
        mass_kg=0.013,
        diagonal_inertia_kg_m2=(1.1e-6, 2.1e-6, 3.1e-6),
    )

    assert standard.dimensions_m != resized.dimensions_m
    assert resized.asset_reference.endswith("geometry.usd")


@pytest.mark.parametrize("field", ["dimensions_m", "asset_scale", "diagonal_inertia_kg_m2"])
def test_block_rejects_invalid_geometry_metadata(field):
    kwargs = {field: (0.1, float("nan"), 0.1)}
    with pytest.raises(ValueError, match=field):
        Block("bad", "source", 0, 0, (0, 0, 0, 0, 0, 0, 1), **kwargs)
