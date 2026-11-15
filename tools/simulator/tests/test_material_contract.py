# SPDX-License-Identifier: Apache-2.0
"""Shared backend material binding, independent of the retired paper runner."""

from pathlib import Path

import pytest

TOOLS = Path(__file__).resolve().parents[1]


def test_block_material_binding_overrides_descendant_and_preserves_source(tmp_path):
    pytest.importorskip("pxr")
    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade
    from tower_sim.backend import material_contract as module

    source = tmp_path / "source.usda"
    authored = Usd.Stage.CreateNew(str(source))
    root = UsdGeom.Xform.Define(authored, "/Block").GetPrim()
    authored.SetDefaultPrim(root)
    collider = UsdGeom.Cube.Define(authored, "/Block/Collider").GetPrim()
    UsdPhysics.CollisionAPI.Apply(collider)
    old = UsdShade.Material.Define(authored, "/Block/Wood")
    mat = UsdPhysics.MaterialAPI.Apply(old.GetPrim())
    mat.CreateStaticFrictionAttr(0.4)
    mat.CreateDynamicFrictionAttr(0.23)
    mat.CreateRestitutionAttr(0.02)
    UsdShade.MaterialBindingAPI.Apply(collider).Bind(
        old, UsdShade.Tokens.strongerThanDescendants, "physics"
    )
    authored.GetRootLayer().Save()
    before = source.read_bytes()
    stage = Usd.Stage.CreateInMemory()
    block = stage.DefinePrim("/Blocks/One", "Xform")
    block.GetReferences().AddReference(str(source))
    block.SetInstanceable(True)
    result = module.bind_block_material(
        stage, block, {"static_friction": 0.551, "dynamic_friction": 0.263, "restitution": 0.02}
    )
    assert len(result) == 1
    assert result[0]["static_friction"] == pytest.approx(0.551)
    assert result[0]["dynamic_friction"] == pytest.approx(0.263)
    assert result[0]["restitution"] == pytest.approx(0.02)
    assert source.read_bytes() == before
