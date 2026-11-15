"""Retained-center bond and compound-tool geometry; no task generation."""

import json
from pathlib import Path

import numpy as np

from .resources import sha

PAIR = [7, 10]


def author_bonds(stage, plans, bond_path):
    tasks = {p.get("task") for p in plans}
    if tasks == {"task7-3"}:
        scenes = [p["task7_scene"] for p in plans]
        return install_bond(stage, scenes, bond_path)
    assert "task7-3" not in tasks
    return dict(passed=True, rows=[], bond_required=False)


def verify_tools(stage, plans):
    """Keep the three authored T-tool colliders; never replace them with a box."""
    from pxr import Usd, UsdGeom, UsdPhysics, UsdShade  # pyright: ignore[reportMissingImports]

    records = []
    for plan in plans:
        for row in plan["blocks"]:
            b = row["block"]
            if b["tower"] != "tools":
                continue
            root = stage.GetPrimAtPath(row["root"])
            assert root and not root.IsInstance()
            assert [
                r.assetPath for r in root.GetMetadata("references").GetAddedOrExplicitItems()
            ] == [b["asset_reference"]]
            assert np.allclose(
                root.GetAttribute("xformOp:scale").Get(), b["asset_scale"], rtol=0, atol=0
            )
            assert not UsdPhysics.RigidBodyAPI(root).GetKinematicEnabledAttr().Get()
            mass = UsdPhysics.MassAPI(root).GetMassAttr().Get()
            assert np.isclose(mass, b["mass_kg"], atol=2e-7, rtol=0)
            colliders = [
                p
                for p in Usd.PrimRange(root, Usd.TraverseInstanceProxies())
                if p.HasAPI(UsdPhysics.CollisionAPI)
            ]
            assert {p.GetName() for p in colliders} == {"Head", "Shaft", "HandleEnd"}, [
                str(p.GetPath()) for p in colliders
            ]
            materials = []
            for collider in colliders:
                assert collider.IsA(UsdGeom.Mesh)
                material, _ = UsdShade.MaterialBindingAPI(collider).ComputeBoundMaterial("physics")
                assert str(material.GetPath()) == row["material_path"]
                api = UsdPhysics.MaterialAPI(material.GetPrim())
                values = [
                    api.GetStaticFrictionAttr().Get(),
                    api.GetDynamicFrictionAttr().Get(),
                    api.GetRestitutionAttr().Get(),
                ]
                expected = [
                    row["material"][k]
                    for k in ("static_friction", "dynamic_friction", "restitution")
                ]
                assert np.allclose(values, expected, atol=2e-6, rtol=0)
                materials.append(values)
            records.append(
                dict(
                    env_id=plan["env_id"],
                    seed=plan["seed"],
                    candidate_sha256=plan["candidate_sha256"],
                    block=b["name"],
                    mass_kg=float(mass),
                    material_path=row["material_path"],
                    material_properties=materials[0],
                    collision_parts=[p.GetName() for p in colliders],
                    authored_tool_geometry_preserved=True,
                    fixed_on_table=False,
                    asset_sha256=b["asset_sha256"],
                )
            )
    return records


def install_bond(stage, scenes, spec_path):
    from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics  # pyright: ignore[reportMissingImports]

    spec = json.loads(Path(spec_path).read_text())
    assert spec["bonded_block_indices"] == PAIR and spec["task"] == "task73"
    rows = []
    for ei, scene in enumerate(scenes):
        assert scene["task"] == "task73"
        config = scene["source_center_bond"]
        assert config["block_indices"] == PAIR and config["amendment_sha256"] == sha(spec_path)
        assert not set(PAIR) & {x["block_index"] for x in scene["transfers"]}
        bodies = []
        matrices = []
        quaternions = []
        for bi, layer in zip(PAIR, [1, 2]):
            b = scene["blocks"][bi]
            assert b["tower"] == "source" and b["layer"] == layer and b["slot"] == 0
            path = Sdf.Path(f"/World/envs/env_{ei}/Blocks/block_{bi:02d}")
            p = stage.GetPrimAtPath(path)
            assert p and p.HasAPI(UsdPhysics.RigidBodyAPI)
            api = UsdPhysics.RigidBodyAPI(p)
            assert api.GetRigidBodyEnabledAttr().Get() and not api.GetKinematicEnabledAttr().Get()
            matrix = UsdGeom.Xformable(p).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            bodies.append(path)
            matrices.append(matrix)
            quaternions.append(Gf.Transform(matrix).GetRotation().GetQuat())
        # Use body0's origin/orientation as the common joint frame. USD local
        # position is in the scaled prim space, so invert its full world matrix.
        anchor = matrices[0].Transform(Gf.Vec3d(0.0))
        local0 = Gf.Vec3f(0.0)
        local1 = Gf.Vec3f(matrices[1].GetInverse().Transform(anchor))
        rot0 = Gf.Quatf(1.0)
        rot1 = Gf.Quatf(quaternions[1].GetInverse() * quaternions[0])
        joint_path = f"/World/envs/env_{ei}/SourceCenterBond583"
        assert not stage.GetPrimAtPath(joint_path)
        joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
        joint.CreateBody0Rel().SetTargets([bodies[0]])
        joint.CreateBody1Rel().SetTargets([bodies[1]])
        joint.CreateLocalPos0Attr(local0)
        joint.CreateLocalPos1Attr(local1)
        joint.CreateLocalRot0Attr(rot0)
        joint.CreateLocalRot1Attr(rot1)
        joint.CreateJointEnabledAttr(True)
        joint.CreateCollisionEnabledAttr(True)
        joint.CreateExcludeFromArticulationAttr(True)
        # Nonbreaking fixed constraint: no invented glue strength or changes to
        # mass/friction/contacts. Both centers remain dynamic, with no world bond.
        assert np.isinf(joint.GetBreakForceAttr().Get()) and np.isinf(
            joint.GetBreakTorqueAttr().Get()
        )
        assert joint.GetBody0Rel().GetTargets() == [
            bodies[0]
        ] and joint.GetBody1Rel().GetTargets() == [bodies[1]]
        assert (
            joint.GetJointEnabledAttr().Get()
            and joint.GetCollisionEnabledAttr().Get()
            and joint.GetExcludeFromArticulationAttr().Get()
        )
        world0 = matrices[0].Transform(Gf.Vec3d(joint.GetLocalPos0Attr().Get()))
        world1 = matrices[1].Transform(Gf.Vec3d(joint.GetLocalPos1Attr().Get()))
        error = float(np.linalg.norm(np.asarray(world0) - np.asarray(world1))) * 0.001
        assert error < 2e-6, error
        rows.append(
            dict(
                environment=ei,
                joint=joint_path,
                bodies=list(map(str, bodies)),
                block_indices=PAIR,
                local_positions_stage=[list(local0), list(local1)],
                local_rotations_wxyz=[
                    [float(q.GetReal()), *map(float, q.GetImaginary())] for q in [rot0, rot1]
                ],
                initial_anchor_error_m=error,
                enabled=True,
                collision_enabled=True,
                excluded_from_articulation=True,
                break_force="infinite",
                break_torque="infinite",
                world_anchor=False,
                bodies_remain_dynamic=True,
            )
        )
    return dict(
        passed=True,
        revision=583,
        rows=rows,
        amendment_sha256=sha(spec_path),
        code_sha256=sha(__file__),
        meaning="Ideal nonbreaking glued fixture between retained layer2/layer3 centers; not measured adhesive material.",
        target_blocks_bonded=False,
        robot_or_tool_bonded=False,
        physics_before_first_step=True,
    )


def assess(scene, poses):
    from scipy.spatial.transform import Rotation

    a = np.asarray(poses, dtype=float)
    assert a.ndim == 3 and a.shape[1:] == (len(scene["blocks"]), 7) and np.isfinite(a).all()
    b0, b1 = PAIR

    def relative(x):
        r0 = Rotation.from_quat(x[:, b0, 3:])
        r1 = Rotation.from_quat(x[:, b1, 3:])
        return r0.inv().apply(x[:, b1, :3] - x[:, b0, :3]), r0.inv() * r1

    expected = np.asarray([[b["pose"] for b in scene["blocks"]]])
    p0, q0 = relative(expected)
    p, q = relative(a)
    position = np.linalg.norm(p - p0[0], axis=1)
    angle = (q0[0].inv() * q).magnitude()
    return dict(
        passed=bool(position.max() <= 0.001 and angle.max() <= np.radians(1.0)),
        block_indices=PAIR,
        frames=len(a),
        maximum_relative_translation_error_m=float(position.max()),
        maximum_relative_rotation_error_deg=float(np.degrees(angle.max())),
        translation_tolerance_m=0.001,
        rotation_tolerance_deg=1.0,
        scope="Additional bond integrity check; original collapse/contact/placement gates remain required.",
    )
