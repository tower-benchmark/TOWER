# SPDX-License-Identifier: Apache-2.0
"""Explicit runtime material binding without modifying referenced source assets."""

import math


def bind_block_material(stage, block, values):
    from pxr import Usd, UsdPhysics, UsdShade

    expected_keys = {"static_friction", "dynamic_friction", "restitution"}
    if (
        set(values) != expected_keys
        or any(
            isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) or v < 0
            for v in values.values()
        )
        or values["restitution"] > 1
    ):
        raise ValueError("Expected finite nonnegative block friction and restitution in [0, 1]")
    material = UsdShade.Material.Define(stage, "/__TowerBlockPhysicsMaterial")
    api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    api.CreateStaticFrictionAttr(float(values["static_friction"]))
    api.CreateDynamicFrictionAttr(float(values["dynamic_friction"]))
    api.CreateRestitutionAttr(float(values["restitution"]))
    # Strong parent binding also overrides materials inside instance proxies.
    UsdShade.MaterialBindingAPI.Apply(block).Bind(
        material, UsdShade.Tokens.strongerThanDescendants, "physics"
    )
    report = []
    for prim in Usd.PrimRange(block, Usd.TraverseInstanceProxies()):
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        bound, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial("physics")
        if not bound:
            raise ValueError(f"Missing block physics material: {prim.GetPath()}")
        actual = UsdPhysics.MaterialAPI(bound.GetPrim())
        readback = {
            "static_friction": actual.GetStaticFrictionAttr().Get(),
            "dynamic_friction": actual.GetDynamicFrictionAttr().Get(),
            "restitution": actual.GetRestitutionAttr().Get(),
        }
        if any(
            v is None or not math.isclose(v, values[k], rel_tol=1e-6, abs_tol=1e-8)
            for k, v in readback.items()
        ):
            raise ValueError(f"Block physics material readback mismatch: {prim.GetPath()}")
        report.append(
            {"collider": str(prim.GetPath()), "material": str(bound.GetPath()), **readback}
        )
    if not report:
        raise ValueError(f"No block colliders: {block.GetPath()}")
    return report
