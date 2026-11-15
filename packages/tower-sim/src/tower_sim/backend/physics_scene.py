# SPDX-License-Identifier: Apache-2.0
"""Drop visual-only source prims before headless cloning, preserving physics."""

from tower_eval.simulator.spec import digest


def prepare_physics_scene(stage, root_path, *, render, mode="auto"):
    if mode not in ("auto", "full"):
        raise ValueError("unknown physics scene mode")
    report = {
        "schema": "jenga_physics_scene_v1",
        "mode": mode,
        "render": render,
        "visual_pruning_enabled": not render and mode == "auto",
        "deactivated_paths": [],
        "copy_from_source": True,
    }
    if not report["visual_pruning_enabled"]:
        return report
    from pxr import Usd, UsdGeom, UsdLux, UsdPhysics

    root = stage.GetPrimAtPath(root_path)

    def physical_prims():
        return {
            str(p.GetPath()): [p.GetTypeName(), list(p.GetAppliedSchemas())]
            for p in Usd.PrimRange(root, Usd.TraverseInstanceProxies())
            if p.HasAPI(UsdPhysics.CollisionAPI)
            or p.HasAPI(UsdPhysics.RigidBodyAPI)
            or p.HasAPI(UsdPhysics.MassAPI)
            or p.IsA(UsdPhysics.Joint)
        }

    before = physical_prims()
    candidates = []
    for prim in Usd.PrimRange(root):
        if prim.IsInstanceProxy():
            continue
        if not (prim.IsA(UsdGeom.Mesh) or prim.IsA(UsdGeom.Camera) or prim.HasAPI(UsdLux.LightAPI)):
            continue
        # Robot colliders commonly put CollisionAPI on a parent Xform. A mesh
        # without its own API can still be a real collider and must be kept.
        ancestor = prim
        physical = False
        while ancestor and ancestor.GetPath().HasPrefix(root.GetPath()):
            if ancestor.HasAPI(UsdPhysics.CollisionAPI):
                physical = True
                break
            ancestor = ancestor.GetParent()
        if not physical and str(prim.GetPath()) not in before:
            candidates.append(prim.GetPath())
    for path in candidates:
        stage.GetPrimAtPath(path).SetActive(False)
    if before != physical_prims():
        raise RuntimeError("visual pruning changed physical prim topology")
    report.update(
        deactivated_paths=[str(p).removeprefix(root_path) for p in candidates],
        physical_prim_count=len(before),
        physical_topology_unchanged=True,
        physical_topology_sha256=digest(before),
    )
    return report
