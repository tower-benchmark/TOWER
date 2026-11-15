# SPDX-License-Identifier: Apache-2.0
"""Block-centered pinch targets from the installed finger collision geometry."""

import numpy as np
from scipy.spatial.transform import Rotation


def collision_meshes(root):
    """CollisionAPI may be on a containing Xform, not on the mesh itself."""
    from pxr import Usd, UsdGeom, UsdPhysics

    for mesh in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if not mesh.IsA(UsdGeom.Mesh):
            continue
        ancestor = mesh
        while ancestor and ancestor.GetPath().HasPrefix(root.GetPath()):
            if ancestor.HasAPI(UsdPhysics.CollisionAPI):
                if UsdPhysics.CollisionAPI(ancestor).GetCollisionEnabledAttr().Get() is not False:
                    yield mesh
                break
            ancestor = ancestor.GetParent()


def inspect_fingers(stage, robot_path):
    from pxr import Gf, UsdGeom

    cache = UsdGeom.XformCache()
    matrix = cache.GetLocalToWorldTransform(stage.GetPrimAtPath(robot_path + "/gripper_base"))
    # The link has a 1000x ancestor scale; a full inverse followed by another
    # mm->m conversion would incorrectly shrink the geometry by 1000x.
    transform = Gf.Transform(matrix)
    q = transform.GetRotation().GetQuat()
    rotation = Rotation.from_quat([*q.GetImaginary(), q.GetReal()])
    origin = np.asarray(transform.GetTranslation()) * 0.001
    fingers = []
    for name in ("gripper_link1", "gripper_link2"):
        vertices, sources = [], []
        for mesh in collision_meshes(stage.GetPrimAtPath(robot_path + "/" + name)):
            points = UsdGeom.Mesh(mesh).GetPointsAttr().Get()
            world = cache.GetLocalToWorldTransform(mesh)
            xyz = np.asarray([world.Transform(Gf.Vec3d(*p)) for p in points]) * 0.001
            vertices.append(rotation.inv().apply(xyz - origin))
            sources.append(str(mesh.GetPath()))
        if not vertices:
            raise ValueError(f"no collision mesh for {name}")
        points = np.concatenate(vertices)
        # The distal 20 mm contains the actual narrow gripping region. Use its
        # measured bounds, including rounded tip vertices, rather than 177 mm.
        distal = points[points[:, 2] >= points[:, 2].max() - 0.020]
        low, high = distal.min(0), distal.max(0)
        if not 0.01 < high[2] - low[2] <= 0.020001 or not 0.1 < high[2] < 0.3:
            raise ValueError("finger collision geometry has unexpected units or depth")
        link_transform = Gf.Transform(
            cache.GetLocalToWorldTransform(stage.GetPrimAtPath(robot_path + "/" + name))
        )
        link_q = link_transform.GetRotation().GetQuat()
        link_rotation = Rotation.from_quat([*link_q.GetImaginary(), link_q.GetReal()])
        link_origin = np.asarray(link_transform.GetTranslation()) * 0.001
        link_points = link_rotation.inv().apply(rotation.apply(distal) + origin - link_origin)
        corners = (
            np.array(np.meshgrid(*zip(link_points.min(0), link_points.max(0)))).reshape(3, -1).T
        )
        fingers.append(
            {
                "link": name,
                "sources": sources,
                "distal_min_m": low.tolist(),
                "distal_max_m": high.tolist(),
                "distal_link_corners_m": corners.tolist(),
            }
        )
    center = np.mean(
        [(np.array(f["distal_min_m"]) + f["distal_max_m"]) / 2 for f in fingers], axis=0
    )
    return {
        "schema": "jenga_finger_geometry_v1",
        "fingers": fingers,
        "pinch_center_in_flange_m": center.tolist(),
        "tip_depth_m": max(f["distal_max_m"][2] for f in fingers),
        "minimum_support_clearance_m": 0.004,
    }


def block_top(pose, dimensions):
    rotation = Rotation.from_quat(pose[3:]).as_matrix()
    return float(pose[2] + np.abs(rotation[2]) @ (np.asarray(dimensions) / 2))


def support_height(state, spec, target):
    """Conservative support top over the target and both open-finger footprints."""
    block = state.blocks[target]
    tops = [0.0]
    for i, item in enumerate(spec.blocks):
        pose = state.blocks[i]
        if i == target or pose[2] >= block[2] - 0.010:
            continue
        dimensions = item.dimensions_m or (0.15, 0.05, 0.03)
        extent = np.abs(Rotation.from_quat(pose[3:]).as_matrix()[:2]) @ (np.asarray(dimensions) / 2)
        # Sweep the distal pads over the full 100 mm opening. The same-row
        # mate is outside this footprint; it must not become a fictitious
        # supporting surface as the held block starts rising.
        footprint = np.abs(Rotation.from_quat(block[3:]).as_matrix()[:2]) @ np.array(
            [0.012, 0.060, 0.0]
        )
        if np.all(np.abs(pose[:2] - block[:2]) <= extent + footprint):
            tops.append(block_top(pose, dimensions))
    return max(tops)


def centered_grasp(
    block, previous_flange_rotation, geometry, support_top, tcp_offset, tcp_rotation
):
    closing = Rotation.from_quat(block[3:]).as_matrix()[:, 1].copy()
    closing[2] = 0.0
    norm = np.linalg.norm(closing)
    if norm < 0.5:
        raise ValueError("block is too tilted for a vertical centered grasp")
    closing /= norm
    if np.dot(closing, previous_flange_rotation.as_matrix()[:, 1]) < 0:
        closing *= -1
    down = previous_flange_rotation.as_matrix()[:, 2].copy()
    down -= closing * np.dot(closing, down)
    down /= np.linalg.norm(down)
    if down[2] > -0.8:
        raise ValueError("wrist tilt is too large for a centered top grasp")
    rotation = Rotation.from_matrix(np.column_stack([np.cross(closing, down), closing, down]))
    center = np.asarray(geometry["pinch_center_in_flange_m"])
    flange = np.asarray(block[:3]) - rotation.apply(center)
    corners = np.concatenate(
        [
            np.array(np.meshgrid(*zip(f["distal_min_m"], f["distal_max_m"]))).reshape(3, -1).T
            for f in geometry["fingers"]
        ]
    )
    lowest = float(rotation.apply(corners)[:, 2].min())
    clearance = float(flange[2] + lowest - support_top)
    raise_by = max(0.0, geometry["minimum_support_clearance_m"] - clearance)
    if raise_by > 0.004:
        raise ValueError(f"centered grasp has insufficient support clearance: {clearance:.6f} m")
    # Preserve the horizontal center; raise only enough to avoid the supporting
    # collision surface. At most 4 mm keeps the block center within the measured
    # ~18 mm gripping region. Record the adjustment instead of hiding an offset.
    flange[2] += raise_by
    clearance += raise_by
    tcp = np.r_[flange + rotation.apply(tcp_offset), (rotation * tcp_rotation).as_quat()]
    return tcp, {
        "pinch_center_world_m": (flange + rotation.apply(center)).tolist(),
        "block_center_world_m": np.asarray(block[:3]).tolist(),
        "vertical_clearance_adjustment_m": raise_by,
        "wrist_tilt_degrees": float(np.degrees(np.arccos(-down[2]))),
        "support_top_m": support_top,
        "planned_tip_clearance_m": clearance,
        "jaw_axis_world": closing.tolist(),
        "finger_geometry": geometry,
        "geometry_source": "live_block_and_installed_finger_colliders",
    }
