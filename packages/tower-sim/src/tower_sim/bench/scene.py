"""Heterogeneous scene hooks; execute before tensor initialization, in clone order."""

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from tower_eval.tasks import benchmark_task


def task_layers(task):
    return benchmark_task(task).layers


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def plans_for_candidates(candidates, validate_spec):
    """Validate topology and bind each environment slot to its own identity."""
    if not 1 <= len(candidates) <= 16:
        raise ValueError("Expected 1..16 candidate scenes")
    plans, topology, hashes = [], None, {}
    for env_id, source in enumerate(candidates):
        candidate = source.to_dict() if hasattr(source, "to_dict") else source
        spec, physics = candidate["episode_spec"], candidate["physics"]
        if not candidate["provenance"]["asset_closure_verified"]:
            raise ValueError("Unverified candidate assets")
        current = [(b["name"], b["fixed_on_table"]) for b in spec["blocks"]]
        if topology is None:
            topology = current
        task = validate_spec(spec)
        if spec["layout_metadata"].get("movable_layers") != task_layers(task):
            raise ValueError("Task movable_layers mismatch")
        if current != topology or len(current) != benchmark_task(task).block_count:
            raise ValueError(
                "Heterogeneous scenes require the same task-specific named block slots and fixed mask"
            )
        blocks = []
        for i, block in enumerate(spec["blocks"]):
            for key, expected in [
                ("asset_reference", "asset_sha256"),
                ("texture_atlas", "texture_sha256"),
            ]:
                if block[key] is None:
                    if block["tower"] != "tools" or block[expected] is not None:
                        raise ValueError("Missing measured wood asset")
                    continue
                path = str(Path(block[key]).resolve())
                if path not in hashes:
                    hashes[path] = _file_hash(path)
                if hashes[path] != block[expected]:
                    raise ValueError(f"Asset changed before instantiation: {path}")
            material = physics[block["name"]]
            values = [
                *block["pose"],
                *block["dimensions_m"],
                *block["asset_scale"],
                block["mass_kg"],
                *(block["diagonal_inertia_kg_m2"] or [1.0, 1.0, 1.0]),
                material["static_friction"],
                material["dynamic_friction"],
                material["restitution"],
            ]
            if not np.isfinite(values).all() or min(values[7:17]) <= 0:
                raise ValueError("Nonfinite/nonpositive block properties")
            blocks.append(
                dict(
                    block=block,
                    material=material,
                    root=f"/World/envs/env_{env_id}/Blocks/block_{i:02}",
                    material_path=f"/World/envs/env_{env_id}/Materials/block_{i:02}",
                )
            )
        plans.append(
            dict(
                env_id=env_id,
                env_path=f"/World/envs/env_{env_id}",
                seed=int(candidate["seed"]),
                candidate_sha256=_digest(candidate),
                blocks=blocks,
                task=task,
                task7_scene=candidate.get("task7_scene"),
            )
        )
    return plans


def _author_block(stage, row, stage_units_per_meter):
    from pxr import (  # pyright: ignore[reportMissingImports]
        Gf,
        Sdf,
        Usd,
        UsdGeom,
        UsdPhysics,
        UsdShade,
    )

    block, material = row["block"], row["material"]
    root = stage.GetPrimAtPath(row["root"])
    if not root:
        raise ValueError(f"Missing cloned block root: {row['root']}")
    # Clear inherited-instance sharing before replacing references/materials.
    root.SetInstanceable(False)
    root.GetReferences().ClearReferences()
    root.GetReferences().AddReference(block["asset_reference"])
    root.CreateAttribute("tower:measuredAssetId", Sdf.ValueTypeNames.Int).Set(
        -1 if block["measured_asset_id"] is None else block["measured_asset_id"]
    )
    transform = UsdGeom.Xformable(root)
    transform.ClearXformOpOrder()
    transform.AddTranslateOp().Set(
        Gf.Vec3d(*(np.asarray(block["pose"][:3]) * stage_units_per_meter))
    )
    transform.AddOrientOp().Set(Gf.Quatf(block["pose"][6], Gf.Vec3f(*block["pose"][3:6])))
    transform.AddScaleOp().Set(Gf.Vec3f(*block["asset_scale"]))
    body = UsdPhysics.RigidBodyAPI.Apply(root)
    body.CreateKinematicEnabledAttr(bool(block["fixed_on_table"]))
    mass = UsdPhysics.MassAPI.Apply(root)
    mass.CreateMassAttr(float(block["mass_kg"]))
    if block["diagonal_inertia_kg_m2"] is not None:
        mass.CreateDiagonalInertiaAttr(
            Gf.Vec3f(*(np.asarray(block["diagonal_inertia_kg_m2"]) * stage_units_per_meter**2))
        )
        mass.CreateCenterOfMassAttr(Gf.Vec3f(0, 0, 0))
    elif block["tower"] != "tools":
        raise ValueError("Missing measured wood inertia")
    contact = UsdShade.Material.Define(stage, row["material_path"])
    api = UsdPhysics.MaterialAPI.Apply(contact.GetPrim())
    api.CreateStaticFrictionAttr(float(material["static_friction"]))
    api.CreateDynamicFrictionAttr(float(material["dynamic_friction"]))
    api.CreateRestitutionAttr(float(material["restitution"]))
    contact.GetPrim().CreateAttribute(
        "physxMaterial:frictionCombineMode", Sdf.ValueTypeNames.Token
    ).Set("average")
    # Bind directly to each collider too: an inherited stronger descendant binding
    # must never retain env_0's material or a reference asset's default material.
    colliders = [
        p
        for p in Usd.PrimRange(root, Usd.TraverseInstanceProxies())
        if p.HasAPI(UsdPhysics.CollisionAPI)
    ]
    if not colliders:
        raise ValueError("Referenced measured block has no collision geometry")
    for prim in [root, *colliders]:
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(
            contact, UsdShade.Tokens.strongerThanDescendants, "physics"
        )


def author_heterogeneous_blocks(stage, plans, bond_path):
    """Call after cloning and before any initialize_physics/tensor views."""
    from pxr import UsdGeom  # pyright: ignore[reportMissingImports]

    units = 1.0 / UsdGeom.GetStageMetersPerUnit(stage)
    if not np.isclose(units, 1000.0):
        raise ValueError("Measured v3 capsule requires the millimetre stage")
    for plan in plans:
        for row in plan["blocks"]:
            _author_block(stage, row, units)
    from .task7 import author_bonds

    bond = author_bonds(stage, plans, bond_path)
    result = verify_usd_readback(stage, plans)
    result["task7_bonds"] = bond
    return result


def verify_usd_readback(stage, plans) -> dict[str, Any]:
    """Check actual composed USD: references, textures, collision dims and bindings."""
    from pxr import Gf, Usd, UsdGeom, UsdPhysics, UsdShade  # pyright: ignore[reportMissingImports]

    meters = UsdGeom.GetStageMetersPerUnit(stage)
    cache = UsdGeom.XformCache()
    records, material_paths = [], set()
    for plan in plans:
        for row in plan["blocks"]:
            block, wanted = row["block"], row["material"]
            if block["tower"] == "tools":
                continue
            root = stage.GetPrimAtPath(row["root"])
            if not root or root.IsInstance():
                raise ValueError("Block remains instance-shared")
            refs = root.GetMetadata("references").GetAddedOrExplicitItems()
            if [r.assetPath for r in refs] != [block["asset_reference"]]:
                raise ValueError("Wrong environment asset reference")
            if root.GetAttribute("tower:measuredAssetId").Get() != block["measured_asset_id"]:
                raise ValueError("Measured asset ID mismatch")
            mass_api = UsdPhysics.MassAPI(root)
            mass = mass_api.GetMassAttr().Get()
            inertia = np.asarray(mass_api.GetDiagonalInertiaAttr().Get()) * meters**2
            if not np.isclose(mass, block["mass_kg"], atol=2e-7) or not np.allclose(
                inertia, block["diagonal_inertia_kg_m2"], rtol=5e-5, atol=1e-10
            ):
                raise ValueError("Mass/inertia readback mismatch")
            fixed = UsdPhysics.RigidBodyAPI(root).GetKinematicEnabledAttr().Get()
            if fixed != block["fixed_on_table"]:
                raise ValueError("Fixed body readback mismatch")
            colliders = [
                p
                for p in Usd.PrimRange(root, Usd.TraverseInstanceProxies())
                if p.HasAPI(UsdPhysics.CollisionAPI)
            ]
            if len(colliders) != 1 or not colliders[0].IsA(UsdGeom.Cube):
                raise ValueError("Expected the measured asset Cube collider")
            collider = colliders[0]
            transform = cache.GetLocalToWorldTransform(collider)
            size = UsdGeom.Cube(collider).GetSizeAttr().Get()
            dimensions = (
                np.array(
                    [np.linalg.norm(transform.TransformDir(Gf.Vec3d(*axis))) for axis in np.eye(3)]
                )
                * size
                * meters
            )
            if not np.allclose(dimensions, block["dimensions_m"], atol=2e-7):
                raise ValueError("Composed collision dimensions differ from measured asset")
            actual_material, _ = UsdShade.MaterialBindingAPI(collider).ComputeBoundMaterial(
                "physics"
            )
            path = str(actual_material.GetPath())
            if path != row["material_path"] or path in material_paths:
                raise ValueError("Physics material binding leaked across block/environment")
            material_paths.add(path)
            api = UsdPhysics.MaterialAPI(actual_material.GetPrim())
            material_values = [
                api.GetStaticFrictionAttr().Get(),
                api.GetDynamicFrictionAttr().Get(),
                api.GetRestitutionAttr().Get(),
            ]
            expected = [wanted[k] for k in ("static_friction", "dynamic_friction", "restitution")]
            if not np.allclose(material_values, expected, atol=2e-6):
                raise ValueError("Physics material readback mismatch")
            textures = set()
            for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
                if not prim.IsA(UsdGeom.Mesh):
                    continue
                surface, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
                if not surface:
                    continue
                for shader in Usd.PrimRange(surface.GetPrim()):
                    attr = shader.GetAttribute("inputs:file")
                    if attr and attr.Get():
                        asset = attr.Get()
                        path = asset.resolvedPath or asset.path
                        if Path(path).is_file():
                            textures.add(_file_hash(path))
            if block["texture_sha256"] not in textures:
                raise ValueError("Measured visual texture is not bound")
            records.append(
                dict(
                    env_id=plan["env_id"],
                    candidate_sha256=plan["candidate_sha256"],
                    seed=plan["seed"],
                    block=block["name"],
                    measured_asset_id=block["measured_asset_id"],
                    mass_kg=float(mass),
                    diagonal_inertia_kg_m2=inertia.tolist(),
                    dimensions_m=dimensions.tolist(),
                    material_path=row["material_path"],
                    material_properties=material_values,
                    texture_sha256=block["texture_sha256"],
                    fixed_on_table=bool(fixed),
                )
            )
    from .task7 import verify_tools

    records.extend(verify_tools(stage, plans))
    return dict(
        schema="tower-v3-heterogeneous-usd-readback-v1",
        passed=True,
        environment_count=len(plans),
        block_count=len(records),
        records=records,
        physics_tensor_readback_complete=False,
    )


def verify_physics_readback(env, candidates, validate_spec):
    """Read-only post-init PhysX validation for every logical environment slot."""
    plans = plans_for_candidates(candidates, validate_spec)
    if list(env.env_paths) != [p["env_path"] for p in plans]:
        raise ValueError("Environment order differs from candidate manifest")
    indices = np.asarray(env.block_indices)
    if (
        indices.shape != (len(plans), len(plans[0]["blocks"]))
        or len(np.unique(indices)) != indices.size
    ):
        raise ValueError("Block tensor indices alias or have incorrect shape")
    masses = np.asarray(env.block_view.get_masses()).reshape(-1)
    inertias = np.asarray(env.block_view.get_inertias()).reshape(-1, 3, 3) / 1e6
    materials = np.asarray(env.block_view._physics_view.get_material_properties())
    records = []
    for plan in plans:
        for slot, row in enumerate(plan["blocks"]):
            index = indices[plan["env_id"], slot]
            expected, material = row["block"], row["material"]
            if env.block_paths[index] != row["root"]:
                raise ValueError("Tensor block path differs from logical environment/slot")
            if not np.isclose(masses[index], expected["mass_kg"], atol=2e-7):
                raise ValueError("PhysX mass mismatch")
            if expected["diagonal_inertia_kg_m2"] is not None and not np.allclose(
                inertias[index], np.diag(expected["diagonal_inertia_kg_m2"]), rtol=5e-5, atol=1e-10
            ):
                raise ValueError("PhysX inertia mismatch")
            wanted = [material[k] for k in ("static_friction", "dynamic_friction", "restitution")]
            actual_material = np.asarray(materials[index]).reshape(-1, 3)
            collider_count = 3 if expected["tower"] == "tools" else 1
            if not np.allclose(actual_material[:collider_count], wanted, atol=2e-6):
                raise ValueError("PhysX material mismatch")
            records.append(
                dict(
                    env_id=plan["env_id"],
                    seed=plan["seed"],
                    candidate_sha256=plan["candidate_sha256"],
                    block=expected["name"],
                    tensor_index=int(index),
                    passed=True,
                )
            )
    return dict(
        schema="tower-v3-heterogeneous-physics-readback-v1",
        passed=True,
        records=records,
        usd_readback=verify_usd_readback(env.stage, plans),
        reset_isolation_qualified=False,
        camera_isolation_qualified=False,
    )


class CloneHook:
    """A per-construction cloner wrapper; never replaces the SDK class method."""

    def __init__(self, plans, bond_path, *, author=author_heterogeneous_blocks):
        self.plans, self.bond_path, self.author = plans, bond_path, author
        self.report: dict[str, Any] = dict(
            schema="tower-v3-heterogeneous-construction-v1",
            clone_calls=0,
            physics_replication=False,
            runtime_qualified=False,
        )

    def factory(self, cloner_class, stage):
        hook = self

        class HeterogeneousCloner(cloner_class):
            def clone(self, source_prim_path, prim_paths, *args, **kwargs):
                if (
                    args
                    or source_prim_path != "/World/envs/env_0"
                    or list(prim_paths) != [p["env_path"] for p in hook.plans]
                ):
                    raise ValueError("Unexpected backend clone signature/topology")
                if hook.report["clone_calls"]:
                    raise ValueError("Only one clone operation is allowed per construction")
                kwargs.update(replicate_physics=False, copy_from_source=True)
                result = super().clone(source_prim_path, prim_paths, **kwargs)
                hook.report["clone_calls"] += 1
                hook.report["usd_readback"] = hook.author(stage, hook.plans, hook.bond_path)
                return result

        return HeterogeneousCloner(stage)

    def verify(self):
        if self.report["clone_calls"] != 1:
            raise RuntimeError("Backend did not invoke the expected pre-physics clone hook")


def prepare_physics_scene(prepare, stage, root, **kwargs):
    from pxr import PhysxSchema, Usd, UsdPhysics  # pyright: ignore[reportMissingImports]

    report = prepare(stage, root, **kwargs)
    for prim in Usd.PrimRange(stage.GetPrimAtPath(root + "/Station")):
        if "InstalledRobots" in str(prim.GetPath()) and prim.HasAPI(UsdPhysics.RigidBodyAPI):
            PhysxSchema.PhysxRigidBodyAPI.Apply(prim).CreateDisableGravityAttr(True)
    return report
