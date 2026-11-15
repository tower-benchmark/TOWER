# SPDX-License-Identifier: Apache-2.0
"""Isaac Sim worker for the dual-arm launcher.

This module is intentionally started in a fresh subprocess.  It imports
SimulationApp before importing PXR or Omni modules.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
import traceback
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from tower_sim.bench.asset_identity import station_identity

from .dual_core import (
    ARM_ORDER,
    CAMERA_NAMES,
    COORDINATE_SYSTEM,
    DATA_ORIGIN_WORLD_MM,
    LIFTER_LEVELS_MM,
    LIFTER_REPLAY_MEASURED,
    LinearLifterTrajectory,
    angular_quantity_numeric_scale,
    expected_block_paths,
    json_dump,
    lifter_zero_reference_mm,
    load_alignment_parameters,
    load_config,
    load_dual_episode,
    load_profile,
    load_robot_alignment3_dual_parameters,
    load_single_arm_replay_profile,
    offset_block_poses_mm,
    profile_arm_parameters,
    profile_delays,
    profile_gains,
    sha256_file,
    split_arm_vectors,
    task_block_poses_data_mm,
)

CAMERA_RESOLUTION = (640, 480)
CALIBRATION_RENDER_WARMUP_FRAMES = 24
CALIBRATION_DOME_PATH = "/__DualReplayCalibrationRender/DefaultPreviewDome"
CAMERA_PATHS = {
    "low": "/PNP_DUALD1200_STATION/CameraAlignment/low_camera",
    "high": "/PNP_DUALD1200_STATION/CameraAlignment/high_camera",
    "left_wrist": (
        "/PNP_DUALD1200_STATION/StationModel/InstalledRobots/StandA_Robot/"
        "NeroRobotWithWristCamera/gripper_base/WristCameraMount/camera_stand_link/"
        "d435_camera_link/camera_link/camera_color_frame/camera_color_optical_frame/"
        "wrist_camera"
    ),
    "back_wrist": (
        "/PNP_DUALD1200_STATION/StationModel/InstalledRobots/StandB_Robot/"
        "NeroRobotWithWristCamera/gripper_base/WristCameraMount/camera_stand_link/"
        "d435_camera_link/camera_link/camera_color_frame/camera_color_optical_frame/"
        "wrist_camera"
    ),
}
ROBOT_PATHS = {
    "left": (
        "/PNP_DUALD1200_STATION/StationModel/InstalledRobots/StandA_Robot/NeroRobotWithWristCamera"
    ),
    "back": (
        "/PNP_DUALD1200_STATION/StationModel/InstalledRobots/StandB_Robot/NeroRobotWithWristCamera"
    ),
}
STAND_ROOT_PATHS = {arm: path.rsplit("/", 1)[0] for arm, path in ROBOT_PATHS.items()}
LIFT_COMPONENT_PATHS = {
    "left": (
        "/PNP_DUALD1200_STATION/StationModel/tn__b4b08Ou03GRNY0xCb0/LiftColumnA_TopPlate",
        "/PNP_DUALD1200_STATION/StationModel/tn__b4b08Ou03GRNY0xCb0/LiftColumnA_InnerColumn",
    ),
    "back": (
        "/PNP_DUALD1200_STATION/StationModel/tn__b4b08Ou03GRNY0xCb0/LiftColumnB_TopPlate",
        "/PNP_DUALD1200_STATION/StationModel/tn__b4b08Ou03GRNY0xCb0/LiftColumnB_InnerColumn",
    ),
}
LIFTER_CONTROL_ROOT = "/PNP_DUALD1200_STATION/LifterControls"
STAND_ASSEMBLY_PATHS = {
    arm: tuple(
        f"/PNP_DUALD1200_STATION/StationModel/tn__b4b08Ou03GRNY0xCb0/LiftColumn{letter}_{component}"
        for component in (
            "BasePlate",
            "OuterHousing",
            "MiddleSleeve",
            "InnerColumn",
            "TopPlate",
        )
    )
    for arm, letter in (("left", "A"), ("back", "B"))
}
ARM_JOINTS = tuple(f"joint{index}" for index in range(1, 8))
ISAAC_URDF_DEFAULT_KP_AUTHORED_PER_DEGREE = 1000.0
ISAAC_URDF_DEFAULT_KV_AUTHORED_PER_DEGREE = 100.0
USD_ANGULAR_GAIN_TO_RUNTIME = 180.0 / math.pi
CALIBRATED_GAIN_MODES = {
    "single_arm_v0_0_1",
    "dual_arm_seed0",
    "mixed_left_single_back_seed0",
    "robot_alignment3_dual_200hz",
}
GRIPPER_PHYSICS_MATERIAL_PATH = "/__DualReplayGripperPhysicsMaterial"
WRIST_CAMERA_SELF_MASK_LINKS = (
    "gripper_flange",
    "gripper_base",
    "gripper_link1",
    "gripper_link2",
)
WRIST_CAMERA_MOUNT_PATH = "gripper_base/WristCameraMount"


def _world_translation(stage: Any, prim: Any) -> tuple[float, float, float]:
    from pxr import UsdGeom

    value = UsdGeom.XformCache().GetLocalToWorldTransform(prim).ExtractTranslation()
    return tuple(float(value[index]) for index in range(3))


def _shift_prim_world_z(stage: Any, path: str, delta_mm: float) -> dict[str, Any]:
    """Shift one calibrated prim along table-world Z in the session layer."""

    from pxr import Gf, UsdGeom

    prim = stage.GetPrimAtPath(path)
    if not prim:
        raise RuntimeError(f"Missing stand-height component: {path}")
    xformable = UsdGeom.Xformable(prim)
    ops = xformable.GetOrderedXformOps()
    before = _world_translation(stage, prim)
    parent_world = UsdGeom.XformCache().GetLocalToWorldTransform(prim.GetParent())
    local_delta = parent_world.GetInverse().TransformDir(Gf.Vec3d(0.0, 0.0, float(delta_mm)))

    transform_ops = [op for op in ops if op.GetOpName() == "xformOp:transform"]
    translate_ops = [op for op in ops if op.GetOpName() == "xformOp:translate"]
    if len(transform_ops) == 1:
        op = transform_ops[0]
        matrix = Gf.Matrix4d(op.Get())
        matrix.SetTranslateOnly(matrix.ExtractTranslation() + local_delta)
        if not op.Set(matrix):
            raise RuntimeError(f"Could not shift stand-height component: {path}")
        edited_op = op.GetOpName()
    elif len(translate_ops) == 1:
        op = translate_ops[0]
        original = Gf.Vec3d(op.Get())
        active = original + local_delta
        if not op.Set(active):
            raise RuntimeError(f"Could not shift stand-height component: {path}")
        edited_op = op.GetOpName()
    else:
        raise RuntimeError(
            f"Unsupported stand-height xform stack on {path}: {[op.GetOpName() for op in ops]}"
        )

    after = _world_translation(stage, prim)
    actual_delta = after[2] - before[2]
    if not math.isclose(actual_delta, float(delta_mm), abs_tol=1e-6):
        raise RuntimeError(
            f"Stand-height shift verification failed for {path}: "
            f"requested={delta_mm}, actual={actual_delta}"
        )
    return {
        "path": path,
        "edited_op": edited_op,
        "world_translation_before_mm": list(before),
        "world_translation_after_mm": list(after),
        "world_z_delta_mm": float(actual_delta),
    }


def _apply_stand_height_offsets(stage: Any, offsets_mm: dict[str, float]) -> dict[str, Any]:
    """Raise moving lift geometry and its robot; leave every stand base fixed."""

    result: dict[str, Any] = {}
    for arm in ARM_ORDER:
        delta_mm = float(offsets_mm[arm])
        paths = (*LIFT_COMPONENT_PATHS[arm], STAND_ROOT_PATHS[arm])
        result[arm] = {
            "requested_offset_mm": delta_mm,
            "measured_base_bottom_to_disc_top_mm": 440.0 + delta_mm,
            "components": [_shift_prim_world_z(stage, path, delta_mm) for path in paths],
            "fixed_components_not_moved": [
                "base_plate",
                "outer_housing",
                "middle_sleeve",
            ],
        }
    return {
        "mode": "static_measured_installation_height_v1",
        "units": "millimetres",
        "arms": result,
        "lifter_trajectory_replayed": False,
        "session_layer_only": True,
        "source_usd_saved": False,
    }


def _capture_absolute_z_binding(stage: Any, path: str, factor: float) -> dict[str, Any]:
    """Capture one authored xform so runtime lift updates never accumulate."""

    from pxr import Gf, UsdGeom

    prim = stage.GetPrimAtPath(path)
    if not prim:
        raise RuntimeError(f"Missing lifter motion target: {path}")
    ops = UsdGeom.Xformable(prim).GetOrderedXformOps()
    transform_ops = [op for op in ops if op.GetOpName() == "xformOp:transform"]
    translate_ops = [op for op in ops if op.GetOpName() == "xformOp:translate"]
    if len(transform_ops) == 1:
        op = transform_ops[0]
        baseline = Gf.Matrix4d(op.Get())
        mode = "transform"
    elif len(translate_ops) == 1:
        op = translate_ops[0]
        baseline = Gf.Vec3d(op.Get())
        mode = "translate"
    else:
        raise RuntimeError(
            f"Unsupported lifter xform stack on {path}: {[op.GetOpName() for op in ops]}"
        )
    parent_world = UsdGeom.XformCache().GetLocalToWorldTransform(prim.GetParent())
    local_world_z = parent_world.GetInverse().TransformDir(Gf.Vec3d(0.0, 0.0, 1.0))
    return {
        "path": path,
        "factor": float(factor),
        "prim": prim,
        "op": op,
        "mode": mode,
        "baseline": baseline,
        "local_world_z": local_world_z,
        "baseline_world_translation_mm": list(_world_translation(stage, prim)),
    }


def _set_absolute_z_binding(
    stage: Any, binding: dict[str, Any], displacement_stage_units: float
) -> list[float]:
    """Set one moving lift component from its immutable captured baseline."""

    from pxr import Gf

    delta = binding["local_world_z"] * (float(displacement_stage_units) * float(binding["factor"]))
    if binding["mode"] == "transform":
        value = Gf.Matrix4d(binding["baseline"])
        value.SetTranslateOnly(value.ExtractTranslation() + delta)
    else:
        value = Gf.Vec3d(binding["baseline"]) + delta
    if not binding["op"].Set(value):
        raise RuntimeError(f"Could not update lifter motion target: {binding['path']}")
    return list(_world_translation(stage, binding["prim"]))


def _load_lifter_control_bindings(stage: Any, configured_middle_ratio: float) -> dict[str, Any]:
    """Resolve and validate the reusable joint8 relationships authored in USD."""

    result: dict[str, Any] = {}
    for arm in ARM_ORDER:
        control_path = f"{LIFTER_CONTROL_ROOT}/{arm}_joint8"
        prim = stage.GetPrimAtPath(control_path)
        if not prim:
            raise RuntimeError(f"Missing USD lifter control binding: {control_path}")
        joint_name = prim.GetAttribute("lifter:jointName").Get()
        lower = float(prim.GetAttribute("lifter:lowerLimitMm").Get())
        upper = float(prim.GetAttribute("lifter:upperLimitMm").Get())
        levels = tuple(float(value) for value in prim.GetAttribute("lifter:allowedLevelsMm").Get())
        ratio = float(prim.GetAttribute("lifter:middleSleeveRatio").Get())
        full_targets = tuple(
            str(path) for path in prim.GetRelationship("lifter:fullStrokeTargets").GetTargets()
        )
        half_targets = tuple(
            str(path) for path in prim.GetRelationship("lifter:halfStrokeTargets").GetTargets()
        )
        fixed_targets = tuple(
            str(path) for path in prim.GetRelationship("lifter:fixedTargets").GetTargets()
        )
        if str(joint_name) != "joint8":
            raise RuntimeError(f"Unexpected lifter joint name on {control_path}")
        if not math.isclose(lower, 0.0, abs_tol=1e-9) or not math.isclose(
            upper, 500.0, abs_tol=1e-9
        ):
            raise RuntimeError(f"Unexpected lifter limits on {control_path}")
        if levels != LIFTER_LEVELS_MM:
            raise RuntimeError(f"Unexpected lifter levels on {control_path}: {levels}")
        if not math.isclose(ratio, configured_middle_ratio, abs_tol=1e-12):
            raise RuntimeError(
                f"Configured middle-sleeve ratio {configured_middle_ratio} "
                f"does not match USD {ratio} for {arm}"
            )
        robot_targets = [path for path in full_targets if path == STAND_ROOT_PATHS[arm]]
        geometry_targets = [path for path in full_targets if path != STAND_ROOT_PATHS[arm]]
        if robot_targets != [STAND_ROOT_PATHS[arm]] or len(geometry_targets) != 2:
            raise RuntimeError(f"Invalid full-stroke bindings for {arm}: {full_targets}")
        if len(half_targets) != 1 or len(fixed_targets) != 2:
            raise RuntimeError(f"Invalid telescoping/fixed bindings for {arm}")
        moving = [
            *(_capture_absolute_z_binding(stage, path, 1.0) for path in geometry_targets),
            *(_capture_absolute_z_binding(stage, path, ratio) for path in half_targets),
        ]
        result[arm] = {
            "control_path": control_path,
            "robot_root_path": robot_targets[0],
            "moving": moving,
            "fixed_paths": fixed_targets,
            "fixed_world_translation_mm": {
                path: list(_world_translation(stage, stage.GetPrimAtPath(path)))
                for path in fixed_targets
            },
            "middle_sleeve_ratio": ratio,
        }
    return result


def _apply_lifter_pose(
    stage: Any,
    bindings: dict[str, Any],
    robots: dict[str, Any],
    robot_baselines: dict[str, tuple[np.ndarray, np.ndarray]],
    displacement_mm: np.ndarray,
    stage_units_per_mm: float,
) -> dict[str, Any]:
    """Apply a continuous two-arm lift sample to visuals and PhysX roots."""

    result: dict[str, Any] = {}
    values = np.asarray(displacement_mm, dtype=np.float64)
    if values.shape != (len(ARM_ORDER),) or not np.isfinite(values).all():
        raise ValueError("lifter displacement must contain two finite values")
    for index, arm in enumerate(ARM_ORDER):
        displacement_stage_units = float(values[index]) * float(stage_units_per_mm)
        component_positions = {
            binding["path"]: _set_absolute_z_binding(stage, binding, displacement_stage_units)
            for binding in bindings[arm]["moving"]
        }
        baseline_position, baseline_orientation = robot_baselines[arm]
        position = baseline_position.copy()
        position[2] += displacement_stage_units
        robots[arm].set_world_pose(
            position=position,
            orientation=baseline_orientation,
        )
        active_position, _ = robots[arm].get_world_pose()
        active_position = np.asarray(active_position, dtype=np.float64)
        if not math.isclose(
            float(active_position[2] - baseline_position[2]),
            displacement_stage_units,
            abs_tol=1e-5,
        ):
            raise RuntimeError(f"{arm} lifter articulation-root update did not apply")
        result[arm] = {
            "displacement_mm": float(values[index]),
            "robot_root_world_position_stage_units": active_position.tolist(),
            "component_world_translation_mm": component_positions,
        }
    return result


def _apply_robot_z_offsets(stage: Any, offsets_mm: dict[str, float]) -> dict[str, Any]:
    """Apply an experimental robot-only world-Z offset in the session layer."""

    arms = {}
    for arm in ARM_ORDER:
        delta_mm = float(offsets_mm[arm])
        arms[arm] = {
            "requested_offset_mm": delta_mm,
            "component": _shift_prim_world_z(stage, STAND_ROOT_PATHS[arm], delta_mm),
        }
    return {
        "mode": "experimental_robot_root_world_z_v1",
        "units": "millimetres",
        "arms": arms,
        "stand_geometry_moved": False,
        "session_layer_only": True,
        "source_usd_saved": False,
    }


def _apply_robot_base_link_world_xyz(
    stage: Any, targets_mm: dict[str, dict[str, float]] | None
) -> dict[str, Any]:
    """Move each complete stand assembly to an absolute base_link world XYZ target."""

    from pxr import Gf, UsdGeom

    if targets_mm is None:
        return {
            "mode": "not_applied",
            "arms": {},
            "session_layer_only": True,
            "source_usd_saved": False,
        }

    arms: dict[str, Any] = {}
    for arm in ARM_ORDER:
        root_path = STAND_ROOT_PATHS[arm]
        base_link_path = ROBOT_PATHS[arm] + "/base_link"
        base_link = stage.GetPrimAtPath(base_link_path)
        if not base_link:
            raise RuntimeError(f"Missing robot base_link for {arm}: {base_link_path}")

        before = _world_translation(stage, base_link)
        target = targets_mm[arm]
        world_delta = Gf.Vec3d(
            float(target["x"]) - before[0],
            float(target["y"]) - before[1],
            float(target["z"]) - before[2],
        )
        components = []
        for path in (*STAND_ASSEMBLY_PATHS[arm], root_path):
            prim = stage.GetPrimAtPath(path)
            if not prim:
                raise RuntimeError(f"Missing stand assembly component: {path}")
            component_before = _world_translation(stage, prim)
            parent_world = UsdGeom.XformCache().GetLocalToWorldTransform(prim.GetParent())
            local_delta = parent_world.GetInverse().TransformDir(world_delta)
            xformable = UsdGeom.Xformable(prim)
            ops = xformable.GetOrderedXformOps()
            transform_ops = [op for op in ops if op.GetOpName() == "xformOp:transform"]
            translate_ops = [op for op in ops if op.GetOpName() == "xformOp:translate"]
            if len(transform_ops) == 1:
                op = transform_ops[0]
                matrix = Gf.Matrix4d(op.Get())
                matrix.SetTranslateOnly(matrix.ExtractTranslation() + local_delta)
                if not op.Set(matrix):
                    raise RuntimeError(f"Could not move stand component: {path}")
            elif len(translate_ops) == 1:
                op = translate_ops[0]
                if not op.Set(Gf.Vec3d(op.Get()) + local_delta):
                    raise RuntimeError(f"Could not move stand component: {path}")
            else:
                raise RuntimeError(
                    f"Unsupported stand-component xform stack on {path}: "
                    f"{[op.GetOpName() for op in ops]}"
                )
            component_after = _world_translation(stage, prim)
            expected_component = tuple(
                component_before[index] + float(world_delta[index]) for index in range(3)
            )
            if any(
                not math.isclose(component_after[index], expected_component[index], abs_tol=1e-6)
                for index in range(3)
            ):
                raise RuntimeError(
                    f"Stand-component XYZ verification failed for {path}: "
                    f"expected={expected_component}, actual={component_after}"
                )
            components.append(
                {
                    "path": path,
                    "edited_op": op.GetOpName(),
                    "world_translation_before_mm": list(component_before),
                    "world_translation_after_mm": list(component_after),
                }
            )

        after = _world_translation(stage, base_link)
        expected = tuple(float(target[axis]) for axis in ("x", "y", "z"))
        if any(not math.isclose(after[index], expected[index], abs_tol=1e-6) for index in range(3)):
            raise RuntimeError(
                f"base_link XYZ verification failed for {arm}: expected={expected}, actual={after}"
            )
        arms[arm] = {
            "robot_root_path": root_path,
            "base_link_path": base_link_path,
            "requested_world_xyz_mm": {axis: float(target[axis]) for axis in ("x", "y", "z")},
            "world_translation_before_mm": list(before),
            "world_translation_after_mm": list(after),
            "applied_world_delta_mm": list(world_delta),
            "moved_components": components,
        }
    return {
        "mode": "complete_stand_absolute_base_link_world_xyz_v1",
        "arms": arms,
        "fixed_cameras_moved": False,
        "session_layer_only": True,
        "source_usd_saved": False,
    }


def _initialize_robot(
    arm: str, root: str | None = None, name_prefix: str = ""
) -> tuple[str, Any, list[str]]:
    from isaacsim.core.prims import SingleArticulation

    errors = {}
    root = root or ROBOT_PATHS[arm]
    for path in (root, root + "/root_joint"):
        robot = SingleArticulation(path, name=f"{name_prefix}nero_{arm}_dual_replay")
        try:
            robot.initialize()
            names = list(robot.dof_names or [])
            if names:
                return path, robot, names
            errors[path] = "no DOFs"
        except Exception as exc:  # noqa: BLE001 - Preserve the existing failure/reporting boundary and exception semantics.
            errors[path] = str(exc)
    raise RuntimeError(f"Could not initialize {arm} Nero articulation: {errors}")


def _create_viewports() -> tuple[dict[str, Any], list[Any]]:
    import omni.kit.viewport.utility
    from pxr import Sdf

    active = omni.kit.viewport.utility.get_active_viewport()
    if active is None:
        raise RuntimeError("Isaac Sim did not create an active viewport")
    active.camera_path = Sdf.Path(CAMERA_PATHS["low"])
    active.set_texture_resolution(CAMERA_RESOLUTION)
    viewports = {"low": active}
    windows = []
    for offset, name in enumerate(CAMERA_NAMES[1:], start=1):
        window = omni.kit.viewport.utility.create_viewport_window(
            name=f"Dual Replay {name.replace('_', ' ').title()}",
            width=CAMERA_RESOLUTION[0],
            height=CAMERA_RESOLUTION[1],
            position_x=offset * CAMERA_RESOLUTION[0],
            position_y=0,
            camera_path=Sdf.Path(CAMERA_PATHS[name]),
        )
        if window is None:
            raise RuntimeError(f"Could not create {name} viewport")
        viewport = window.viewport_api
        viewport.camera_path = Sdf.Path(CAMERA_PATHS[name])
        viewport.set_texture_resolution(CAMERA_RESOLUTION)
        viewports[name] = viewport
        windows.append(window)
    return viewports, windows


def _montage(frames: dict[str, np.ndarray]) -> np.ndarray:
    ordered = [frames[name] for name in CAMERA_NAMES]
    if any(frame.shape != (480, 640, 3) for frame in ordered):
        raise ValueError(f"Unexpected camera frame shapes: {[frame.shape for frame in ordered]}")
    return np.concatenate(ordered, axis=1)


def _create_wrist_camera_proxies(
    stage: Any,
    camera_paths: dict[str, str] | None = None,
    robot_paths: dict[str, str] | None = None,
    proxy_root: str = "/__DualReplayWristCameraProxies",
) -> tuple[dict[str, str], dict[str, str], dict[str, Any]]:
    """Create root-level wrist cameras that follow each live gripper base.

    Isaac/Fabric evaluates Camera prims nested below the metre-scaled robot
    articulation with an extra inherited transform. A root-level camera avoids
    that transform ambiguity. Its pose is updated from the live gripper-base
    world transform and the source camera's authored mount transform.
    """

    from pxr import UsdGeom

    camera_paths = camera_paths or CAMERA_PATHS
    robot_paths = robot_paths or ROBOT_PATHS
    capture_paths = dict(camera_paths)
    intrinsic_sources = dict(camera_paths)
    mounts: dict[str, Any] = {}
    cache = UsdGeom.XformCache()
    for name, arm in (("left_wrist", "left"), ("back_wrist", "back")):
        source_path = camera_paths[name]
        source = UsdGeom.Camera(stage.GetPrimAtPath(source_path))
        gripper_path = f"{robot_paths[arm]}/gripper_base"
        gripper = stage.GetPrimAtPath(gripper_path)
        if not source or not gripper:
            raise RuntimeError(f"Could not create {name} proxy camera")
        relative, reset = cache.ComputeRelativeTransform(source.GetPrim(), gripper)
        if reset:
            raise RuntimeError(f"Unexpected reset transform for {name} camera mount")
        proxy_path = f"{proxy_root}/{name}"
        proxy = UsdGeom.Camera.Define(stage, proxy_path)
        for source_attr, target_attr in (
            (source.GetProjectionAttr(), proxy.GetProjectionAttr()),
            (source.GetFocalLengthAttr(), proxy.GetFocalLengthAttr()),
            (source.GetHorizontalApertureAttr(), proxy.GetHorizontalApertureAttr()),
            (source.GetVerticalApertureAttr(), proxy.GetVerticalApertureAttr()),
            (source.GetHorizontalApertureOffsetAttr(), proxy.GetHorizontalApertureOffsetAttr()),
            (source.GetVerticalApertureOffsetAttr(), proxy.GetVerticalApertureOffsetAttr()),
            (source.GetFStopAttr(), proxy.GetFStopAttr()),
            (source.GetFocusDistanceAttr(), proxy.GetFocusDistanceAttr()),
        ):
            value = source_attr.Get()
            if value is not None:
                target_attr.Set(value)
        proxy.GetClippingRangeAttr().Set((10.0, 10_000_000.0))
        pose_op = UsdGeom.Xformable(proxy).AddTransformOp(
            UsdGeom.XformOp.PrecisionDouble,
            "runtimePose",
        )
        capture_paths[name] = proxy_path
        mounts[name] = {
            "arm": arm,
            "source_path": source_path,
            "proxy_path": proxy_path,
            "gripper_path": gripper_path,
            "finger_paths": [
                f"{robot_paths[arm]}/gripper_link1",
                f"{robot_paths[arm]}/gripper_link2",
            ],
            "camera_to_gripper": relative,
            "pose_op": pose_op,
        }
    return capture_paths, intrinsic_sources, mounts


def _update_wrist_camera_proxies(
    stage: Any,
    mounts: dict[str, Any],
    aim_mode: str = "source_mount",
    target_world: Sequence[float] | None = None,
    position_mode: str = "source_mount",
    finger_behind_mm: float = 70.0,
    finger_height_mm: float = 80.0,
    tracking_mode: str = "per_frame",
    robots: dict[str, Any] | None = None,
    cameras: dict[str, Any] | None = None,
) -> None:
    """Update root-level wrist-camera poses from live articulation transforms."""

    from pxr import Gf, UsdGeom

    cache = UsdGeom.XformCache()
    for name, mount in mounts.items():
        gripper = stage.GetPrimAtPath(mount["gripper_path"])
        gripper_to_world = cache.GetLocalToWorldTransform(gripper)
        pose = mount["camera_to_gripper"] * gripper_to_world
        if (
            tracking_mode == "lock_to_gripper_at_first_live_pose"
            and "locked_camera_to_gripper" in mount
        ):
            pose = mount["locked_camera_to_gripper"] * gripper_to_world
            mount["pose_op"].Set(pose)
            if cameras is not None:
                position = pose.ExtractTranslation()
                orientation = pose.ExtractRotationQuat()
                cameras[name].set_world_pose(
                    position=[float(value) for value in position],
                    orientation=[
                        float(orientation.GetReal()),
                        *[float(value) for value in orientation.GetImaginary()],
                    ],
                    camera_axes="usd",
                )
            continue
        if tracking_mode not in {
            "per_frame",
            "lock_to_gripper_at_first_live_pose",
        }:
            raise ValueError(f"Unsupported wrist camera tracking mode: {tracking_mode}")
        if position_mode == "finger_midpoint":
            if target_world is None:
                raise ValueError("A target is required for finger-relative wrist cameras")
            if robots is None:
                # Camera render products are created before articulations are
                # initialized. Keep the source mount during that warm-up; the
                # live finger-relative pose is applied before replay capture.
                mount["pose_op"].Set(pose)
                continue
            finger_positions = [
                cache.GetLocalToWorldTransform(stage.GetPrimAtPath(path)).ExtractTranslation()
                for path in mount["finger_paths"]
            ]
            midpoint = (finger_positions[0] + finger_positions[1]) * 0.5
            toward_target_xy = Gf.Vec3d(
                float(target_world[0]) - midpoint[0],
                float(target_world[1]) - midpoint[1],
                0.0,
            )
            if toward_target_xy.GetLength() <= 1e-6:
                raise RuntimeError("Finger midpoint has no horizontal direction to target")
            toward_target_xy.Normalize()
            position = (
                midpoint
                - toward_target_xy * float(finger_behind_mm)
                + Gf.Vec3d(0.0, 0.0, float(finger_height_mm))
            )
            pose.SetTranslateOnly(position)
        elif position_mode != "source_mount":
            raise ValueError(f"Unsupported wrist camera position mode: {position_mode}")
        if aim_mode == "look_at_task_center":
            if target_world is None:
                raise ValueError("A wrist-camera look-at target is required")
            position = pose.ExtractTranslation()
            forward = Gf.Vec3d(
                float(target_world[0]) - position[0],
                float(target_world[1]) - position[1],
                float(target_world[2]) - position[2],
            )
            if forward.GetLength() <= 1e-6:
                raise RuntimeError("Wrist camera coincides with its look-at target")
            forward.Normalize()
            world_up = Gf.Vec3d(0.0, 0.0, 1.0)
            right = Gf.Cross(forward, world_up)
            if right.GetLength() <= 1e-6:
                world_up = Gf.Vec3d(0.0, 1.0, 0.0)
                right = Gf.Cross(forward, world_up)
            right.Normalize()
            up = Gf.Cross(right, forward).GetNormalized()
            pose = Gf.Matrix4d(
                right[0],
                right[1],
                right[2],
                0.0,
                up[0],
                up[1],
                up[2],
                0.0,
                -forward[0],
                -forward[1],
                -forward[2],
                0.0,
                position[0],
                position[1],
                position[2],
                1.0,
            )
        elif aim_mode != "source_mount":
            raise ValueError(f"Unsupported wrist camera aim mode: {aim_mode}")
        if tracking_mode == "lock_to_gripper_at_first_live_pose" and robots is not None:
            locked = pose * gripper_to_world.GetInverse()
            mount["locked_camera_to_gripper"] = locked
            mount["camera_to_gripper"] = locked
        mount["pose_op"].Set(pose)
        if cameras is not None:
            position = pose.ExtractTranslation()
            orientation = pose.ExtractRotationQuat()
            cameras[name].set_world_pose(
                position=[float(value) for value in position],
                orientation=[
                    float(orientation.GetReal()),
                    *[float(value) for value in orientation.GetImaginary()],
                ],
                camera_axes="usd",
            )


def _create_calibrated_cameras(
    stage: Any,
    camera_paths: dict[str, str] | None = None,
    intrinsic_source_paths: dict[str, str] | None = None,
    name_prefix: str = "",
    *,
    initialize_render_products: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Create 640x480 Isaac Camera render products using each authored K."""

    from isaacsim.sensors.camera import Camera

    camera_paths = camera_paths or dict(CAMERA_PATHS)
    intrinsic_source_paths = intrinsic_source_paths or dict(CAMERA_PATHS)
    cameras = {}
    report = {}
    for name in CAMERA_NAMES:
        path = camera_paths[name]
        intrinsic_source_path = intrinsic_source_paths[name]
        _prim = stage.GetPrimAtPath(path)
        intrinsic_prim = stage.GetPrimAtPath(intrinsic_source_path)
        raw_k = intrinsic_prim.GetAttribute("calibration:intrinsicMatrixK").Get()
        if raw_k is None:
            raise ValueError(f"Camera has no calibration:intrinsicMatrixK: {path}")
        intrinsic = np.asarray(raw_k, dtype=np.float64).reshape(3, 3)
        if not np.isfinite(intrinsic).all():
            raise ValueError(f"Camera intrinsic matrix contains non-finite values: {path}")
        camera = Camera(
            prim_path=path,
            name=f"{name_prefix}dual_replay_{name}",
            frequency=30,
            resolution=CAMERA_RESOLUTION,
        )
        if initialize_render_products:
            camera.initialize(attach_rgb_annotator=True)
            camera.set_resolution(CAMERA_RESOLUTION)
        camera.set_opencv_pinhole_properties(
            cx=float(intrinsic[0, 2]),
            cy=float(intrinsic[1, 2]),
            fx=float(intrinsic[0, 0]),
            fy=float(intrinsic[1, 1]),
            pinhole=[0.0] * 12,
        )
        cameras[name] = camera
        report[name] = {
            "path": path,
            "intrinsic_source_path": intrinsic_source_path,
            "resolution": list(CAMERA_RESOLUTION),
            "intrinsic_matrix_k": intrinsic.tolist(),
            "capture_api": "isaacsim.sensors.camera.Camera RGB annotator",
            "opencv_pinhole_applied_explicitly": True,
        }
    return cameras, report


def _capture_calibrated_cameras(
    cameras: dict[str, Any],
    rep: Any,
    simulation_app: Any,
    rt_subframes: int,
    fixed_camera_display_rotation_degrees: int = 0,
    stage: Any | None = None,
    wrist_self_mask_paths: Sequence[str] = (),
) -> dict[str, np.ndarray]:
    return _capture_calibrated_camera_batches(
        [cameras],
        rep,
        simulation_app,
        rt_subframes,
        fixed_camera_display_rotation_degrees,
        stage,
        wrist_self_mask_paths,
    )[0]


def _configure_calibration_renderer(
    stage: Any,
    dome_path: str = CALIBRATION_DOME_PATH,
    *,
    anti_aliasing: int = 3,
    dlss_mode: int | None = None,
) -> None:
    """Reuse calibration lighting, with optional capture-quality overrides."""
    import carb
    from pxr import UsdLux

    settings = carb.settings.get_settings()
    settings.set("/rtx/post/aa/op", anti_aliasing)
    if dlss_mode is not None:
        settings.set("/rtx/post/dlss/execMode", dlss_mode)
    settings.set("/rtx/post/motionblur/enabled", False)
    settings.set("/omni/replicator/captureMotionBlur", False)
    dome = UsdLux.DomeLight.Define(stage, dome_path)
    dome.CreateIntensityAttr(650.0)
    dome.CreateExposureAttr(0.0)


def _capture_calibrated_camera_batches(
    camera_batches: Sequence[dict[str, Any]],
    rep: Any,
    simulation_app: Any,
    rt_subframes: int,
    fixed_camera_display_rotation_degrees: int = 0,
    stage: Any | None = None,
    wrist_self_mask_paths: Sequence[str] = (),
    rgba_batch_reader: Any | None = None,
    rgba_batch_prepare: Any | None = None,
    replicator_sync_only: bool = False,
) -> list[dict[str, np.ndarray]]:
    """Capture all cloned environments in the replay's two render passes."""
    rgba_batches = None

    def render() -> None:
        nonlocal rgba_batches
        rep.orchestrator.step(
            rt_subframes=rt_subframes,
            delta_time=0.0,
            pause_timeline=True,
        )
        if not replicator_sync_only:
            simulation_app.update()
        if rgba_batch_reader is not None:
            rgba_batches = rgba_batch_reader()
            if len(rgba_batches) != len(camera_batches):
                raise RuntimeError("Tiled camera batch count does not match environments")

    def read(cameras: dict[str, Any], names: Sequence[str], index: int) -> dict[str, np.ndarray]:
        result = {}
        for name in names:
            rgba = cameras[name].get_rgba() if rgba_batches is None else rgba_batches[index][name]
            if rgba is None:
                raise RuntimeError(f"Calibrated {name} camera returned no RGB frame")
            rgba = np.asarray(rgba)
            if rgba.shape != (480, 640, 4):
                raise RuntimeError(f"Calibrated {name} camera returned shape {rgba.shape}")
            if np.issubdtype(rgba.dtype, np.floating):
                maximum = float(np.nanmax(rgba))
                if maximum <= 1.0:
                    rgba = np.clip(rgba * 255.0, 0.0, 255.0)
            frame = np.asarray(rgba[..., :3], dtype=np.uint8).copy()
            if name in ("low", "high") and fixed_camera_display_rotation_degrees == 180:
                frame = np.ascontiguousarray(np.rot90(frame, 2))
            result[name] = frame
        return result

    if rgba_batch_prepare is not None:
        rgba_batch_prepare(("low", "high") if wrist_self_mask_paths else CAMERA_NAMES)
    render()
    if not wrist_self_mask_paths:
        return [read(cameras, CAMERA_NAMES, index) for index, cameras in enumerate(camera_batches)]
    if stage is None:
        raise ValueError("A stage is required for wrist-camera self masking")

    # Preserve complete robots in the fixed-camera render. Then temporarily
    # hide both rendered end effectors for the wrist render. The root-level
    # proxy cameras and all articulation/physics state remain untouched.
    frames = [read(cameras, ("low", "high"), index) for index, cameras in enumerate(camera_batches)]
    from pxr import UsdGeom

    visibility_state = []
    try:
        if rgba_batch_prepare is not None:
            rgba_batch_prepare(("left_wrist", "back_wrist"))
        for path in wrist_self_mask_paths:
            prim = stage.GetPrimAtPath(path)
            if not prim:
                raise RuntimeError(f"Missing wrist-camera self-mask prim: {path}")
            imageable = UsdGeom.Imageable(prim)
            if not imageable:
                raise RuntimeError(f"Wrist-camera self-mask prim is not imageable: {path}")
            visibility_attr = imageable.GetVisibilityAttr()
            visibility_state.append((visibility_attr, visibility_attr.Get()))
            imageable.MakeInvisible()
        if not replicator_sync_only:
            simulation_app.update()
        render()
        for index, (cameras, batch) in enumerate(zip(camera_batches, frames)):
            batch.update(read(cameras, ("left_wrist", "back_wrist"), index))
    finally:
        if rgba_batch_prepare is not None:
            rgba_batch_prepare(())
        for visibility_attr, original in reversed(visibility_state):
            visibility_attr.Set(original or UsdGeom.Tokens.inherited)
        if not replicator_sync_only:
            simulation_app.update()
    return frames


def _wrist_camera_self_mask_paths(mode: str) -> tuple[str, ...]:
    if mode == "none":
        return ()
    if mode == "hide_camera_mount":
        return tuple(f"{ROBOT_PATHS[arm]}/{WRIST_CAMERA_MOUNT_PATH}" for arm in ARM_ORDER)
    if mode != "hide_gripper_and_camera_mount":
        raise ValueError(f"Unsupported wrist-camera self-mask mode: {mode}")
    return tuple(
        f"{ROBOT_PATHS[arm]}/{link}" for arm in ARM_ORDER for link in WRIST_CAMERA_SELF_MASK_LINKS
    )


def _measured_episode_vector(
    robot: Any,
    dof_names: Sequence[str],
    meters_per_unit: float,
    joint_positions: np.ndarray | None = None,
) -> np.ndarray:
    positions = np.asarray(
        robot.get_joint_positions() if joint_positions is None else joint_positions,
        dtype=np.float64,
    )
    result = np.asarray([positions[dof_names.index(name)] for name in ARM_JOINTS] + [0.0])
    # The imported parent ``gripper`` DOF has zero gains.  The physical jaw
    # width is represented by the two driven finger joints, so prefer their
    # separation whenever both are present (the original single-arm reporter
    # read the parent first and could falsely report a stationary gripper).
    if "gripper_joint1" in dof_names and "gripper_joint2" in dof_names:
        result[7] = meters_per_unit * (
            positions[dof_names.index("gripper_joint1")]
            - positions[dof_names.index("gripper_joint2")]
        )
    elif "gripper" in dof_names:
        result[7] = meters_per_unit * positions[dof_names.index("gripper")]
    else:
        raise RuntimeError(f"Robot has no supported gripper DOFs: {dof_names}")
    return result


def _finger_link_distance(
    stage: Any, arm: str, meters_per_unit: float
) -> tuple[list[float], list[float], float]:
    """Return finger-link origins and separation in SI metres."""

    from pxr import UsdGeom

    cache = UsdGeom.XformCache()
    positions: list[list[float]] = []
    for link in ("gripper_link1", "gripper_link2"):
        prim = stage.GetPrimAtPath(f"{ROBOT_PATHS[arm]}/{link}")
        if not prim:
            raise RuntimeError(f"Missing rendered {arm} finger link: {link}")
        translation = cache.GetLocalToWorldTransform(prim).ExtractTranslation()
        positions.append([float(translation[i]) * meters_per_unit for i in range(3)])
    distance = float(np.linalg.norm(np.asarray(positions[0]) - np.asarray(positions[1])))
    return positions[0], positions[1], distance


def _apply_gripper_mount(
    stage: Any,
    arm: str,
    degrees: float,
    tool_axis_offset_mm: float,
) -> dict[str, Any]:
    """Correct the installed gripper pose at its fixed flange joint."""

    from pxr import Gf, UsdPhysics

    path = f"{ROBOT_PATHS[arm]}/joints/gripper_base_joint"
    joint = UsdPhysics.FixedJoint(stage.GetPrimAtPath(path))
    if not joint:
        raise RuntimeError(f"Missing {arm} gripper base fixed joint: {path}")
    attr = joint.GetLocalRot0Attr()
    original = attr.Get()
    position_attr = joint.GetLocalPos0Attr()
    original_position = position_attr.Get()
    active_position = Gf.Vec3f(
        float(original_position[0]),
        float(original_position[1]),
        float(original_position[2]) + float(tool_axis_offset_mm) / 1000.0,
    )
    if not position_attr.Set(active_position):
        raise RuntimeError(f"Could not apply {arm} gripper mount tool-axis offset")
    radians = math.radians(float(degrees))
    rotation = Gf.Quatf(math.cos(radians * 0.5), Gf.Vec3f(0.0, 0.0, math.sin(radians * 0.5)))
    active = rotation * original
    if not attr.Set(active):
        raise RuntimeError(f"Could not apply {arm} gripper mount yaw")
    return {
        "joint_path": path,
        "yaw_degrees": float(degrees),
        "tool_axis_offset_mm": float(tool_axis_offset_mm),
        "original_local_pos0": [float(value) for value in original_position],
        "active_local_pos0": [float(value) for value in active_position],
        "original_local_rot0": [
            float(original.GetReal()),
            *[float(value) for value in original.GetImaginary()],
        ],
        "active_local_rot0": [
            float(active.GetReal()),
            *[float(value) for value in active.GetImaginary()],
        ],
        "session_layer_only": True,
    }


def _apply_gripper_physics_material(
    stage: Any, material_config: dict[str, float] | None
) -> dict[str, Any] | None:
    """Bind a session-only physics material to all four finger links."""

    if material_config is None:
        return None
    from pxr import UsdPhysics, UsdShade

    material = UsdShade.Material.Define(stage, GRIPPER_PHYSICS_MATERIAL_PATH)
    api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    api.CreateStaticFrictionAttr().Set(material_config["static_friction"])
    api.CreateDynamicFrictionAttr().Set(material_config["dynamic_friction"])
    api.CreateRestitutionAttr().Set(material_config["restitution"])
    bound_paths: list[str] = []
    for arm in ARM_ORDER:
        for link_name in ("gripper_link1", "gripper_link2"):
            path = f"{ROBOT_PATHS[arm]}/{link_name}"
            prim = stage.GetPrimAtPath(path)
            if not prim:
                raise RuntimeError(f"Missing {arm} finger link: {path}")
            UsdShade.MaterialBindingAPI.Apply(prim).Bind(
                material,
                UsdShade.Tokens.strongerThanDescendants,
                "physics",
            )
            collision = stage.GetPrimAtPath(path + "/collisions")
            effective, _ = UsdShade.MaterialBindingAPI(collision).ComputeBoundMaterial("physics")
            if not effective or str(effective.GetPath()) != GRIPPER_PHYSICS_MATERIAL_PATH:
                raise RuntimeError(f"Physics material did not bind to {path}")
            bound_paths.append(path)
    return {
        **material_config,
        "material_path": GRIPPER_PHYSICS_MATERIAL_PATH,
        "finger_link_paths": bound_paths,
        "session_layer_only": True,
        "source_usd_saved": False,
    }


def _apply_jenga_physics_material(
    stage: Any, material_config: dict[str, float] | None, block_paths: Sequence[str] | None = None
) -> dict[str, Any] | None:
    """Override each block's existing wood physics material at runtime.

    Replay blocks reference an instanceable asset whose collision mesh already
    binds to ``PhysicsMaterials/Wood``. Authoring a new material binding at the
    instance root makes the geometry disappear in this packaged Hydra scene.
    Instead, disable composition instancing in the anonymous session layer and
    edit only the two friction attributes on the existing per-block material.
    Visual bindings, restitution, and PhysX combine modes remain untouched.
    """

    if material_config is None:
        return None
    from pxr import UsdPhysics, UsdShade

    material_paths: list[str] = []
    collision_paths: list[str] = []
    original_values: dict[str, float] | None = None
    for path in expected_block_paths() if block_paths is None else block_paths:
        prim = stage.GetPrimAtPath(path)
        if not prim or not prim.IsActive():
            raise RuntimeError(f"Missing replay Jenga block: {path}")
        if prim.IsInstance() and not prim.SetInstanceable(False):
            raise RuntimeError(f"Could not disable runtime instancing for {path}")
        material_path = path + "/PhysicsMaterials/Wood"
        collision_path = path + "/Geometry"
        material_prim = stage.GetPrimAtPath(material_path)
        collision_prim = stage.GetPrimAtPath(collision_path)
        if not material_prim or not collision_prim:
            raise RuntimeError(f"Missing composed Jenga material/collision under {path}")
        api = UsdPhysics.MaterialAPI(material_prim)
        if not api:
            raise RuntimeError(f"Missing physics material API: {material_path}")
        if original_values is None:
            original_values = {
                "static_friction": float(api.GetStaticFrictionAttr().Get()),
                "dynamic_friction": float(api.GetDynamicFrictionAttr().Get()),
            }
        api.GetStaticFrictionAttr().Set(material_config["static_friction"])
        api.GetDynamicFrictionAttr().Set(material_config["dynamic_friction"])
        effective, _ = UsdShade.MaterialBindingAPI(collision_prim).ComputeBoundMaterial("physics")
        if not effective or str(effective.GetPath()) != material_path:
            raise RuntimeError(
                f"Existing Jenga physics material did not resolve on {collision_path}"
            )
        active_api = UsdPhysics.MaterialAPI(effective.GetPrim())
        active_values = (
            float(active_api.GetStaticFrictionAttr().Get()),
            float(active_api.GetDynamicFrictionAttr().Get()),
        )
        requested_values = (
            material_config["static_friction"],
            material_config["dynamic_friction"],
        )
        if any(
            not math.isclose(actual, requested, abs_tol=1e-7)
            for actual, requested in zip(active_values, requested_values, strict=True)
        ):
            raise RuntimeError(
                f"Jenga friction readback mismatch on {material_path}: "
                f"active={active_values}, requested={requested_values}"
            )
        material_paths.append(material_path)
        collision_paths.append(collision_path)
    return {
        **material_config,
        "measurement_pair": "jenga_on_jenga",
        "original_composed_friction": original_values,
        "material_paths": material_paths,
        "collision_paths": collision_paths,
        "block_count": len(material_paths),
        "existing_physics_bindings_preserved": True,
        "visual_material_bindings_unchanged": True,
        "runtime_composition_instancing_disabled": True,
        "restitution_authored": False,
        "friction_combine_mode_authored": False,
        "session_layer_only": True,
        "source_usd_saved": False,
    }


def _apply_robot_visual_material_mode(stage: Any, mode: str) -> dict[str, Any]:
    """Select source-root override or restore authored per-part materials."""

    from pxr import Usd, UsdGeom, UsdShade

    report: dict[str, Any] = {
        "mode": mode,
        "session_layer_only": mode != "source",
        "source_usd_saved": False,
        "robots": {},
    }
    for arm, path in ROBOT_PATHS.items():
        robot = stage.GetPrimAtPath(path)
        if not robot:
            raise RuntimeError(f"Missing {arm} robot visual root: {path}")
        binding_api = UsdShade.MaterialBindingAPI(robot)
        original = binding_api.ComputeBoundMaterial()[0]
        original_path = str(original.GetPath()) if original else None
        if mode == "restore_descendant_bindings":
            binding_api.UnbindDirectBinding()
        elif mode != "source":
            raise ValueError(f"Unsupported robot visual material mode: {mode}")

        active_root = binding_api.ComputeBoundMaterial()[0]
        render_mesh_count = 0
        bound_render_mesh_count = 0
        active_material_paths: set[str] = set()
        predicate = Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate)
        for prim in Usd.PrimRange(robot, predicate):
            if not prim.IsA(UsdGeom.Mesh):
                continue
            purpose = UsdGeom.Imageable(prim).ComputePurpose()
            if purpose not in (UsdGeom.Tokens.default_, UsdGeom.Tokens.render):
                continue
            render_mesh_count += 1
            material = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
            if material:
                bound_render_mesh_count += 1
                active_material_paths.add(str(material.GetPath()))
        active_root_path = str(active_root.GetPath()) if active_root else None
        if mode == "restore_descendant_bindings":
            if active_root_path is not None:
                raise RuntimeError(f"Could not clear {arm} robot root visual binding")
            if not active_material_paths:
                raise RuntimeError(f"No authored descendant materials found for {arm}")
            if original_path in active_material_paths:
                raise RuntimeError(
                    f"{arm} root material still overrides descendant robot materials"
                )
        report["robots"][arm] = {
            "root_path": path,
            "original_root_material": original_path,
            "active_root_material": active_root_path,
            "render_mesh_count": render_mesh_count,
            "bound_render_mesh_count": bound_render_mesh_count,
            "active_material_count": len(active_material_paths),
            "active_material_examples": sorted(active_material_paths)[:12],
        }
    return report


def _apply_gripper_visual_material_mode(
    stage: Any, mode: str, robot_paths: dict[str, str] | None = None
) -> dict[str, Any]:
    """Restore the authored dark gripper-finger material after USD packaging.

    The URDF visual references bind the two finger meshes to
    ``/nero/Looks/material_gripper_dark_gray``. That absolute relationship is
    outside each link-level reference and is consequently discarded when the
    standalone USDZ is composed. Rebinding to the equivalent per-robot Looks
    material in the session layer preserves the intended appearance without
    changing collision geometry or physics materials.
    """

    from pxr import Usd, UsdGeom, UsdShade

    report: dict[str, Any] = {
        "mode": mode,
        "session_layer_only": mode != "source",
        "source_usd_saved": False,
        "robots": {},
    }
    if mode == "source":
        return report
    if mode != "restore_authored_dark_gray_fingers":
        raise ValueError(f"Unsupported gripper visual material mode: {mode}")
    predicate = Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate)
    for arm, robot_path in (robot_paths or ROBOT_PATHS).items():
        material_path = f"{robot_path}/Looks/material_gripper_dark_gray"
        material_prim = stage.GetPrimAtPath(material_path)
        if not material_prim:
            raise RuntimeError(f"Missing authored gripper material: {material_path}")
        material = UsdShade.Material(material_prim)
        bindings: list[dict[str, Any]] = []
        for link_name in ("gripper_link1", "gripper_link2"):
            visual_root = stage.GetPrimAtPath(f"{robot_path}/{link_name}/UrdfVisual")
            if not visual_root:
                raise RuntimeError(f"Missing packaged URDF visual for {arm}/{link_name}")
            meshes = [
                prim
                for prim in Usd.PrimRange(visual_root, predicate)
                if prim.IsA(UsdGeom.Mesh)
                and UsdGeom.Imageable(prim).ComputePurpose()
                in (UsdGeom.Tokens.default_, UsdGeom.Tokens.render)
            ]
            if not meshes:
                raise RuntimeError(f"No render mesh for {arm}/{link_name}")
            # The meshes are instance proxies inside the packaged reference
            # and cannot receive session opinions. Bind at the locally
            # authored UrdfVisual reference root with stronger-than-descendant
            # strength so it overrides the broken scoped material target.
            binding_api = UsdShade.MaterialBindingAPI.Apply(visual_root)
            original = binding_api.ComputeBoundMaterial()[0]
            binding_api.Bind(
                material,
                bindingStrength=UsdShade.Tokens.strongerThanDescendants,
            )
            for mesh in meshes:
                active = UsdShade.MaterialBindingAPI(mesh).ComputeBoundMaterial()[0]
                if not active or str(active.GetPath()) != material_path:
                    raise RuntimeError(
                        f"Could not restore dark gripper material on {mesh.GetPath()}"
                    )
                bindings.append(
                    {
                        "mesh_path": str(mesh.GetPath()),
                        "original_material": (str(original.GetPath()) if original else None),
                        "active_material": str(active.GetPath()),
                    }
                )
        report["robots"][arm] = {
            "material_path": material_path,
            "bindings": bindings,
        }
    return report


def _apply_robot_visual_link_rebase(stage: Any, enabled: bool) -> dict[str, Any]:
    """Move robot-root-authored visual meshes into their link-local frames."""

    from pxr import Usd, UsdGeom

    report: dict[str, Any] = {
        "enabled": bool(enabled),
        "method": "inverse_default_link_to_robot_transform",
        "session_layer_only": bool(enabled),
        "source_usd_saved": False,
        "robots": {},
    }
    if not enabled:
        return report
    links = (
        "base_link",
        *(f"link{index}" for index in range(1, 8)),
        "gripper_flange",
        "gripper_base",
        "gripper_link1",
        "gripper_link2",
    )
    cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    for arm, robot_path in ROBOT_PATHS.items():
        robot = stage.GetPrimAtPath(robot_path)
        if not robot:
            raise RuntimeError(f"Missing {arm} robot for visual rebase")
        authored: list[str] = []
        for link_name in links:
            link = stage.GetPrimAtPath(f"{robot_path}/{link_name}")
            visuals = stage.GetPrimAtPath(f"{robot_path}/{link_name}/visuals")
            if not link or not visuals:
                raise RuntimeError(f"Missing {arm} visual subtree for {link_name}")
            xformable = UsdGeom.Xformable(visuals)
            existing = xformable.GetOrderedXformOps()
            if existing:
                raise RuntimeError(f"{arm} {link_name} visual subtree already has xform ops")
            link_to_robot, reset = cache.ComputeRelativeTransform(link, robot)
            if reset:
                raise RuntimeError(f"Unexpected reset stack below {arm} robot at {link_name}")
            xformable.AddTransformOp(
                UsdGeom.XformOp.PrecisionDouble,
                "robotVisualRebase",
            ).Set(link_to_robot.GetInverse())
            readback = [op.GetOpName() for op in xformable.GetOrderedXformOps()]
            if readback != ["xformOp:transform:robotVisualRebase"]:
                raise RuntimeError(f"Could not apply {arm} {link_name} visual rebase")
            authored.append(str(visuals.GetPath()))
        report["robots"][arm] = {
            "robot_path": robot_path,
            "rebased_visual_subtrees": authored,
        }
    return report


def _apply_urdf_robot_visuals(stage: Any, asset_path: str | None) -> dict[str, Any] | None:
    """Replace flattened link visuals with link-local visuals from the URDF asset."""

    if asset_path is None:
        return None
    from pxr import Usd, UsdGeom, UsdShade

    source_path = Path(asset_path).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)
    source_stage = Usd.Stage.Open(str(source_path))
    if source_stage is None:
        raise RuntimeError(f"Could not open robot URDF USD asset: {source_path}")
    source_root = source_stage.GetDefaultPrim()
    if not source_root or str(source_root.GetPath()) != "/nero":
        raise RuntimeError("Robot URDF USD asset must have default prim /nero")
    source_mpu = float(UsdGeom.GetStageMetersPerUnit(source_stage))
    target_mpu = float(UsdGeom.GetStageMetersPerUnit(stage))
    unit_scale = source_mpu / target_mpu
    links = (
        "base_link",
        *(f"link{index}" for index in range(1, 8)),
        "gripper_flange",
        "gripper_base",
        "gripper_link1",
        "gripper_link2",
    )
    report: dict[str, Any] = {
        "asset_path": str(source_path),
        "asset_sha256": sha256_file(source_path),
        "source_meters_per_unit": source_mpu,
        "target_meters_per_unit": target_mpu,
        "unit_scale": unit_scale,
        "session_layer_only": True,
        "source_usd_saved": False,
        "robots": {},
    }
    predicate = Usd.TraverseInstanceProxies(Usd.PrimDefaultPredicate)
    for arm, robot_path in ROBOT_PATHS.items():
        robot = stage.GetPrimAtPath(robot_path)
        robot_to_world = UsdGeom.XformCache().GetLocalToWorldTransform(robot)
        inherited_scales = tuple(
            robot_to_world.TransformDir(axis).GetLength()
            for axis in (
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (0.0, 0.0, 1.0),
            )
        )
        if max(inherited_scales) - min(inherited_scales) > 1e-6:
            raise RuntimeError(
                f"URDF visual host scale is anisotropic for {arm}: {inherited_scales}"
            )
        inherited_scale = sum(inherited_scales) / 3.0
        local_scale = unit_scale / inherited_scale
        replacements: list[dict[str, Any]] = []
        for link_name in links:
            source_visuals = source_stage.GetPrimAtPath(f"/nero/{link_name}/visuals")
            if not source_visuals:
                raise RuntimeError(f"URDF asset is missing {link_name}/visuals")
            existing_path = f"{robot_path}/{link_name}/visuals"
            existing = stage.GetPrimAtPath(existing_path)
            if not existing:
                raise RuntimeError(f"Scene is missing {existing_path}")
            UsdGeom.Imageable(existing).MakeInvisible()
            replacement_path = f"{robot_path}/{link_name}/UrdfVisual"
            replacement = UsdGeom.Xform.Define(stage, replacement_path).GetPrim()
            replacement.GetReferences().AddReference(str(source_path), f"/nero/{link_name}/visuals")
            UsdGeom.Xformable(replacement).AddScaleOp().Set((local_scale, local_scale, local_scale))
            mesh_count = 0
            bound_mesh_count = 0
            for prim in Usd.PrimRange(replacement, predicate):
                if not prim.IsA(UsdGeom.Mesh):
                    continue
                mesh_count += 1
                material = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()[0]
                subset_materials = [
                    UsdShade.MaterialBindingAPI(subset.GetPrim()).ComputeBoundMaterial()[0]
                    for subset in UsdGeom.Subset.GetAllGeomSubsets(UsdGeom.Imageable(prim))
                ]
                if material or (subset_materials and all(subset_materials)):
                    bound_mesh_count += 1
            if mesh_count == 0 or bound_mesh_count != mesh_count:
                raise RuntimeError(
                    f"URDF visuals/materials did not resolve for {arm} {link_name}: "
                    f"meshes={mesh_count}, bound={bound_mesh_count}"
                )
            replacements.append(
                {
                    "link": link_name,
                    "hidden_flattened_visuals": existing_path,
                    "urdf_visuals": replacement_path,
                    "mesh_count": mesh_count,
                    "bound_mesh_count": bound_mesh_count,
                }
            )
        report["robots"][arm] = {
            "robot_path": robot_path,
            "inherited_linear_scale": inherited_scale,
            "local_linear_scale": local_scale,
            "replacements": replacements,
        }
    return report


def _gripper_limits_in_stage_units(name, limits, units_per_meter):
    """Normalize the two supported Nero asset conventions, without double scaling.

    Older imports retain SI limits; the calibrated standalone station already
    authors millimetres. Reject unknown geometry instead of guessing its units.
    Both screening and rendered replay use this same normalization.
    """
    expected_si = {
        "gripper": (0.0, 0.1),
        "gripper_joint1": (-0.01, 0.06),
        "gripper_joint2": (-0.06, 0.01),
    }[name]
    if not np.isfinite(units_per_meter) or units_per_meter <= 0:
        raise ValueError("gripper stage units must be finite and positive")
    limits = np.asarray(limits, dtype=float)
    active = np.asarray(expected_si) * units_per_meter
    if limits.shape != (2,) or not np.all(np.isfinite(limits)):
        raise ValueError("invalid gripper joint limits")
    if np.allclose(limits, active, rtol=1e-6, atol=1e-8):
        return active, "already_in_stage_units"
    if np.allclose(limits, expected_si, rtol=1e-6, atol=1e-8):
        return active, "si_to_stage_units"
    raise ValueError(f"Unsupported {name} limits {limits.tolist()}; verify asset units")


def _apply_gripper_linear_unit_scale(
    stage: Any, arm: str, units_per_meter: float, robot_path: str | None = None
) -> dict[str, Any]:
    """Normalize supported gripper limits into this stage's units once."""

    from pxr import UsdPhysics

    joints: dict[str, Any] = {}
    for name in ("gripper", "gripper_joint1", "gripper_joint2"):
        path = f"{robot_path or ROBOT_PATHS[arm]}/joints/{name}"
        joint = UsdPhysics.PrismaticJoint(stage.GetPrimAtPath(path))
        if not joint:
            raise RuntimeError(f"Missing {arm} gripper joint: {path}")
        original_lower = float(joint.GetLowerLimitAttr().Get())
        original_upper = float(joint.GetUpperLimitAttr().Get())
        (active_lower, active_upper), conversion = _gripper_limits_in_stage_units(
            name, [original_lower, original_upper], units_per_meter
        )
        joint.GetLowerLimitAttr().Set(active_lower)
        joint.GetUpperLimitAttr().Set(active_upper)
        joints[name] = {
            "joint_path": path,
            "original_limits": [original_lower, original_upper],
            "conversion": conversion,
            "active_limits": [
                float(joint.GetLowerLimitAttr().Get()),
                float(joint.GetUpperLimitAttr().Get()),
            ],
        }
    return {
        "units_per_meter": float(units_per_meter),
        "joints": joints,
        "session_layer_only": True,
        "source_usd_saved": False,
    }


def _scale_gripper_command_for_stage(
    indices: np.ndarray,
    positions: np.ndarray,
    velocities: np.ndarray,
    dof_names: Sequence[str],
    units_per_meter: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert gripper positions and velocities from SI into stage units."""

    scaled_positions = np.asarray(positions, dtype=np.float64).copy()
    scaled_velocities = np.asarray(velocities, dtype=np.float64).copy()
    gripper_names = {"gripper", "gripper_joint1", "gripper_joint2"}
    for command_index, dof_index in enumerate(np.asarray(indices, dtype=np.int64)):
        if dof_names[int(dof_index)] in gripper_names:
            scaled_positions[command_index] *= units_per_meter
            scaled_velocities[command_index] *= units_per_meter
    return scaled_positions, scaled_velocities


def _stage_profile(stage: Any, arm: str) -> dict[str, Any]:
    prim = stage.GetPrimAtPath(ROBOT_PATHS[arm])
    if not prim:
        raise ValueError(f"Stage is missing {arm} robot at {ROBOT_PATHS[arm]}")
    keys = {
        "profile_schema": "robotAlignment:profileSchema",
        "arm": "robotAlignment:arm",
        "stand": "robotAlignment:stand",
        "physics_hz": "robotAlignment:physicsHz",
        "command_interpolation": "robotAlignment:commandInterpolation",
        "kp": "robotAlignment:kpRuntimePerRad",
        "kd": "robotAlignment:kdRuntimePerRad",
        "armature": "robotAlignment:armatureKgM2",
        "coulomb": "robotAlignment:coulombFriction",
        "viscous": "robotAlignment:viscousFrictionRuntime",
        "delays": "robotAlignment:commandDelaySeconds",
    }
    result = {}
    for key, attribute in keys.items():
        value = prim.GetAttribute(attribute).Get()
        if value is None:
            raise ValueError(f"Missing embedded profile attribute {attribute} on {arm}")
        if key in {"kp", "kd", "armature", "coulomb", "viscous", "delays"}:
            value = [float(item) for item in value]
        elif key == "physics_hz":
            value = float(value)
        else:
            value = str(value)
        result[key] = value
    return result


def _apply_and_verify_profile(
    robot: Any,
    dof_names: list[str],
    profile: dict[str, Any],
    arm: str,
    joint_gain_mode: str,
    single_arm_stage1: dict[str, Any] | None = None,
    single_arm_stage2: dict[str, Any] | None = None,
    single_arm_replay_profile: dict[str, Any] | None = None,
    angular_unit_scale: float = 1.0,
    linear_units_per_meter: float = 1.0,
    reproduction_seed: int | None = None,
) -> dict[str, Any]:
    from .replay_actions import (
        _apply_arm_dynamics_parameters,
        _apply_gripper_finger_limits,
        _articulation_dynamics_arrays,
        _controller_gains,
    )

    missing = [name for name in ARM_JOINTS if name not in dof_names]
    if missing:
        raise RuntimeError(f"{arm} robot missing joints: {missing}")
    indices = np.asarray([dof_names.index(name) for name in ARM_JOINTS], dtype=np.int32)
    controller = robot.get_articulation_controller()
    active_kp, active_kd = controller.get_gains()
    default_kp = np.asarray(active_kp, dtype=np.float64).copy()
    default_kd = np.asarray(active_kd, dtype=np.float64).copy()
    if joint_gain_mode == "profile":
        desired_kp, desired_kd = profile_gains(profile, arm)
        requested_kp = default_kp.copy()
        requested_kd = default_kd.copy()
        requested_kp[indices] = desired_kp
        requested_kd[indices] = desired_kd
        controller.set_gains(kps=requested_kp, kds=requested_kd, save_to_usd=False)
    elif joint_gain_mode == "stage_default":
        # Preserve the gains composed from the selected robot USD.  In
        # particular, do not call set_gains: this keeps both arm and gripper
        # defaults exactly as authored by the scene.
        desired_kp = default_kp[indices].copy()
        desired_kd = default_kd[indices].copy()
    elif joint_gain_mode == "isaac_urdf_importer_default":
        # Isaac Sim 5.1's URDF importer UI resets its position-drive defaults
        # to stiffness=1e3 and damping=1e2. Angular DriveAPI gains are authored
        # per degree, while ArticulationController exposes gains per radian.
        desired_kp = np.full(
            len(indices),
            ISAAC_URDF_DEFAULT_KP_AUTHORED_PER_DEGREE * USD_ANGULAR_GAIN_TO_RUNTIME,
            dtype=np.float64,
        )
        desired_kd = np.full(
            len(indices),
            ISAAC_URDF_DEFAULT_KV_AUTHORED_PER_DEGREE * USD_ANGULAR_GAIN_TO_RUNTIME,
            dtype=np.float64,
        )
        requested_kp = default_kp.copy()
        requested_kd = default_kd.copy()
        requested_kp[indices] = desired_kp
        requested_kd[indices] = desired_kd
        controller.set_gains(kps=requested_kp, kds=requested_kd, save_to_usd=False)
    elif joint_gain_mode in CALIBRATED_GAIN_MODES:
        if (
            single_arm_stage1 is None
            or single_arm_stage2 is None
            or single_arm_replay_profile is None
        ):
            raise ValueError("calibrated Stage 1, Stage 2, and replay profile are required")
        desired_kp = (
            np.asarray(
                [single_arm_stage2["joints"][name]["kp_runtime_per_rad"] for name in ARM_JOINTS],
                dtype=np.float64,
            )
            * angular_unit_scale
        )
        desired_kd = (
            np.asarray(
                [single_arm_stage2["joints"][name]["kv_runtime_per_rad"] for name in ARM_JOINTS],
                dtype=np.float64,
            )
            * angular_unit_scale
        )
        requested_kp = default_kp.copy()
        requested_kd = default_kd.copy()
        requested_kp[indices] = desired_kp
        requested_kd[indices] = desired_kd
        controller.set_gains(kps=requested_kp, kds=requested_kd, save_to_usd=False)
    else:
        raise ValueError(f"Unsupported joint_gain_mode: {joint_gain_mode!r}")
    if joint_gain_mode in CALIBRATED_GAIN_MODES:
        dynamics_parameters = {
            "path": single_arm_stage1["path"],
            "sha256": single_arm_stage1["sha256"],
            "joints": {
                name: {
                    key: float(single_arm_stage1["joints"][name][key]) * angular_unit_scale
                    for key in (
                        "armature_kg_m2",
                        "coulomb_friction",
                        "viscous_friction",
                    )
                }
                for name in ARM_JOINTS
            },
        }
        expected = dynamics_parameters["joints"]
        command_delays = np.asarray(
            [single_arm_stage2["joints"][name]["command_delay_seconds"] for name in ARM_JOINTS],
            dtype=np.float64,
        )
        # These torque-like runtime values scale with squared stage length;
        # this preserves the accepted single-arm physical profile in the
        # millimetre-authored dual stage.
        source_effort_limit = 100.0
        controller.set_max_efforts(
            np.full(
                len(indices),
                source_effort_limit * angular_unit_scale,
                dtype=np.float32,
            ),
            joint_indices=indices,
        )
        gripper = single_arm_replay_profile["gripper"]
        finger_names = ("gripper_joint1", "gripper_joint2")
        missing_fingers = [name for name in finger_names if name not in dof_names]
        if missing_fingers:
            raise RuntimeError(f"{arm} robot missing gripper DOFs: {missing_fingers}")
        finger_indices = np.asarray(
            [dof_names.index(name) for name in finger_names], dtype=np.int32
        )
        # Isaac's articulation controller accepts linear-drive gains and
        # effort limits in its runtime SI interface.  Use the accepted
        # single-arm numbers literally even when the containing USD stage is
        # authored in millimetres.
        finger_kp = np.asarray(gripper["kp_runtime_per_meter"], dtype=np.float64)[1:]
        finger_kd = np.asarray(gripper["kv_runtime_per_meter"], dtype=np.float64)[1:]
        active_all_kp, active_all_kd = controller.get_gains()
        requested_all_kp = np.asarray(active_all_kp, dtype=np.float64).copy()
        requested_all_kd = np.asarray(active_all_kd, dtype=np.float64).copy()
        requested_all_kp[finger_indices] = finger_kp
        requested_all_kd[finger_indices] = finger_kd
        controller.set_gains(kps=requested_all_kp, kds=requested_all_kd, save_to_usd=False)
        gripper_limits = _apply_gripper_finger_limits(
            robot,
            dof_names,
            float(gripper["finger_effort_limit"]) * linear_units_per_meter,
            float(gripper["finger_velocity_limit_m_s"]) * linear_units_per_meter,
        )
        active_all_kp, active_all_kd = controller.get_gains()
        if not np.allclose(
            np.asarray(active_all_kp)[finger_indices], finger_kp, rtol=1e-6, atol=1e-5
        ) or not np.allclose(
            np.asarray(active_all_kd)[finger_indices], finger_kd, rtol=1e-6, atol=1e-5
        ):
            raise RuntimeError(f"{arm} gripper gain readback mismatch")
    else:
        dynamics_parameters = profile_arm_parameters(profile, arm)
        expected = profile["arms"][arm]["parameters"]
        command_delays = profile_delays(profile, arm)
    dynamics_application = _apply_arm_dynamics_parameters(robot, dof_names, dynamics_parameters)
    read_kp, read_kd, gain_report = _controller_gains(robot, dof_names)
    _, _, dynamics_report = _articulation_dynamics_arrays(robot, dof_names)
    if not np.allclose(read_kp[indices], desired_kp, rtol=1e-6, atol=1e-6):
        raise RuntimeError(f"{arm} Kp {joint_gain_mode} readback mismatch")
    if not np.allclose(read_kd[indices], desired_kd, rtol=1e-6, atol=1e-6):
        raise RuntimeError(f"{arm} Kd {joint_gain_mode} readback mismatch")
    for name in ARM_JOINTS:
        actual = dynamics_report[name]
        for actual_key, expected_key in (
            ("armature_kg_m2", "armature_kg_m2"),
            ("static_friction", "coulomb_friction"),
            ("dynamic_friction", "coulomb_friction"),
            ("viscous_friction", "viscous_friction"),
        ):
            if not math.isclose(
                actual[actual_key], float(expected[name][expected_key]), rel_tol=1e-5, abs_tol=1e-6
            ):
                raise RuntimeError(f"{arm}/{name}/{actual_key} profile readback mismatch")
    arrays = [read_kp[indices], read_kd[indices]]
    arrays.extend(
        np.asarray([dynamics_report[name][key] for name in ARM_JOINTS])
        for key in ("armature_kg_m2", "static_friction", "dynamic_friction", "viscous_friction")
    )
    if not all(np.isfinite(array).all() for array in arrays):
        raise RuntimeError(f"{arm} runtime profile readback contains non-finite values")
    return {
        "joint_gain_mode": joint_gain_mode,
        "gain_source": {
            "profile": "alignment_profile",
            "stage_default": "composed_stage_usd",
            "isaac_urdf_importer_default": "isaac_sim_5_1_urdf_importer_default",
            "single_arm_v0_0_1": "accepted_single_arm_replay_v0_0_1",
            "dual_arm_seed0": "arm_specific_seed0_alignment",
            "mixed_left_single_back_seed0": ("left_single_arm_v0_0_1_back_seed0_alignment"),
            "robot_alignment3_dual_200hz": ("robot_alignment3_arm_specific_seed0_dedicated_200hz"),
        }[joint_gain_mode],
        "gain_setter_called": joint_gain_mode != "stage_default",
        "stage_default_gains_preserved": joint_gain_mode == "stage_default",
        "isaac_urdf_importer_defaults": (
            {
                "authored_kp_per_degree": ISAAC_URDF_DEFAULT_KP_AUTHORED_PER_DEGREE,
                "authored_kv_per_degree": ISAAC_URDF_DEFAULT_KV_AUTHORED_PER_DEGREE,
                "runtime_kp_per_radian": float(desired_kp[0]),
                "runtime_kv_per_radian": float(desired_kd[0]),
            }
            if joint_gain_mode == "isaac_urdf_importer_default"
            else None
        ),
        "single_arm_gain_profile": (
            {
                "stage1_physical_parameters": single_arm_stage1["path"],
                "stage1_sha256": single_arm_stage1["sha256"],
                "stage2_gain_parameters": single_arm_stage2["path"],
                "stage2_sha256": single_arm_stage2["sha256"],
                "source_verification": single_arm_stage2.get(
                    "source_verification", "original_calibration_stage"
                ),
                "source_stage_meters_per_unit": single_arm_stage2["source_stage_meters_per_unit"],
                "calibration_source_stage_meters_per_unit": single_arm_stage2.get(
                    "calibration_source_stage_meters_per_unit"
                ),
                "target_stage_meters_per_unit": single_arm_stage2["target_stage_meters_per_unit"],
                "angular_quantity_numeric_scale": angular_unit_scale,
                "source_effort_limit_nm": 100.0,
                "active_effort_limit_stage_units": 100.0 * angular_unit_scale,
                "calibration_control_hz": single_arm_stage2.get("calibration_control_hz"),
                "calibration_physics_hz": single_arm_stage2.get("calibration_physics_hz"),
                "calibration_interpretation": single_arm_stage2.get("calibration_interpretation"),
            }
            if joint_gain_mode in CALIBRATED_GAIN_MODES
            else None
        ),
        "gripper_profile": (
            {
                "path": single_arm_replay_profile["path"],
                "sha256": single_arm_replay_profile["sha256"],
                "runtime_units": "isaac_si",
                "finger_kp_runtime_per_meter": finger_kp.tolist(),
                "finger_kv_runtime_per_meter": finger_kd.tolist(),
                "finger_effort_limit_n": float(
                    single_arm_replay_profile["gripper"]["finger_effort_limit"]
                ),
                "finger_velocity_limit_authored_m_s": float(
                    single_arm_replay_profile["gripper"]["finger_velocity_limit_m_s"]
                ),
                "limits": gripper_limits,
                "verified": True,
            }
            if joint_gain_mode in CALIBRATED_GAIN_MODES
            else None
        ),
        "reproduction_seed": reproduction_seed,
        "gains": gain_report,
        "dynamics": dynamics_report,
        "dynamics_application": dynamics_application,
        "command_delays_seconds": command_delays.tolist(),
        "finite": True,
        "verified": True,
    }


def _collect_block_physics_poses(view: Any) -> list[dict[str, Any]]:
    if not view.is_physics_handle_valid():
        raise RuntimeError("Block pose logging requires a live PhysX view")
    positions, orientations = view.get_world_poses()
    positions = np.asarray(positions, dtype=np.float64)
    orientations = np.asarray(orientations, dtype=np.float64)
    if not np.isfinite(positions).all() or not np.isfinite(orientations).all():
        raise RuntimeError("Non-finite block physics poses")
    if not np.allclose(np.linalg.norm(orientations, axis=1), 1.0, atol=1e-5):
        raise RuntimeError("Block physics orientations are not normalized")
    return [
        {"block_path": path, "position": position.tolist(), "quaternion_wxyz": orientation.tolist()}
        for path, position, orientation in zip(
            view.prim_paths, positions, orientations, strict=True
        )
    ]


def run(config_path: Path, overlay: Path, max_frames: int) -> dict[str, Any]:
    from isaacsim import SimulationApp

    config = load_config(config_path)
    stage_source = Path(config["stage"])
    stage_hash_before = sha256_file(stage_source)
    stage_identity_before = station_identity(stage_hash_before)
    reference_path = config["runtime_profile_reference_report"]
    reference_report = json.loads(Path(reference_path).read_text()) if reference_path else None
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    episode = load_dual_episode(config["fetch"]["local_path"], max_frames=max_frames)
    lifter_trajectory = None
    lifter_zero_mm = None
    if config["lifter_replay"] == LIFTER_REPLAY_MEASURED:
        lifter_zero_mm = lifter_zero_reference_mm(episode, config.get("lifter_zero_reference_mm"))
        lifter_trajectory = LinearLifterTrajectory(
            episode.lifter_position_mm,
            episode.fps,
            lifter_zero_mm,
        )
    profile = load_profile(config["profile_manifest"], bool(config["allow_provisional_profile"]))
    stage1_by_arm: dict[str, Any] = {}
    stage2_by_arm: dict[str, Any] = {}
    single_arm_replay_profile = None
    if config["joint_gain_mode"] == "single_arm_v0_0_1":
        stage1 = load_alignment_parameters(
            config["single_arm_stage1_parameters"],
            ("armature_kg_m2", "coulomb_friction", "viscous_friction"),
        )
        stage2 = load_alignment_parameters(
            config["single_arm_stage2_parameters"],
            ("kp_runtime_per_rad", "kv_runtime_per_rad", "command_delay_seconds"),
        )
        stage1_by_arm = {arm: stage1 for arm in ARM_ORDER}
        stage2_by_arm = {arm: stage2 for arm in ARM_ORDER}
        single_arm_replay_profile = load_single_arm_replay_profile(
            config["single_arm_replay_profile"]
        )
    elif config["joint_gain_mode"] == "dual_arm_seed0":
        stage1_by_arm = {
            arm: load_alignment_parameters(
                config["dual_arm_stage1_parameters"][arm],
                ("armature_kg_m2", "coulomb_friction", "viscous_friction"),
            )
            for arm in ARM_ORDER
        }
        stage2_by_arm = {
            arm: load_alignment_parameters(
                config["dual_arm_stage2_parameters"][arm],
                ("kp_runtime_per_rad", "kv_runtime_per_rad", "command_delay_seconds"),
            )
            for arm in ARM_ORDER
        }
        single_arm_replay_profile = load_single_arm_replay_profile(
            config["single_arm_replay_profile"]
        )
    elif config["joint_gain_mode"] == "mixed_left_single_back_seed0":
        stage1_by_arm = {
            "left": load_alignment_parameters(
                config["single_arm_stage1_parameters"],
                ("armature_kg_m2", "coulomb_friction", "viscous_friction"),
            ),
            "back": load_alignment_parameters(
                config["dual_arm_stage1_parameters"]["back"],
                ("armature_kg_m2", "coulomb_friction", "viscous_friction"),
            ),
        }
        stage2_by_arm = {
            "left": load_alignment_parameters(
                config["single_arm_stage2_parameters"],
                ("kp_runtime_per_rad", "kv_runtime_per_rad", "command_delay_seconds"),
            ),
            "back": load_alignment_parameters(
                config["dual_arm_stage2_parameters"]["back"],
                ("kp_runtime_per_rad", "kv_runtime_per_rad", "command_delay_seconds"),
            ),
        }
        single_arm_replay_profile = load_single_arm_replay_profile(
            config["single_arm_replay_profile"]
        )
    elif config["joint_gain_mode"] == "robot_alignment3_dual_200hz":
        dual_parameters = load_robot_alignment3_dual_parameters(
            config["robot_alignment3_dual_parameters"],
            replay_report=reference_report,
            runtime_stage_sha256=stage_identity_before if reference_report is not None else None,
            allow_stage_mismatch=bool(
                config.get("allow_profile_stage_mismatch", False)
                and str(config.get("profile_stage_mismatch_acknowledgement", "")).strip()
            ),
        )
        for arm in ARM_ORDER:
            dual_parameters["arms"][arm]["parameter_meters_per_unit"] = float(
                config["robot_alignment3_parameter_meters_per_unit"]
            )
            if reference_report is not None and not math.isclose(
                dual_parameters["arms"][arm]["parameter_meters_per_unit"],
                dual_parameters["arms"][arm]["reference_source_stage_meters_per_unit"],
                abs_tol=1e-12,
            ):
                raise ValueError("Parameter units do not match the verified runtime report")
        stage1_by_arm = dict(dual_parameters["arms"])
        stage2_by_arm = dict(dual_parameters["arms"])
        single_arm_replay_profile = load_single_arm_replay_profile(
            config["single_arm_replay_profile"]
        )
    episode_hash_before = sha256_file(episode.path)
    started = time.monotonic()
    started_at = datetime.now(UTC).isoformat()
    simulation_app = SimulationApp(
        {
            "headless": True,
            "renderer": "RaytracedLighting",
            "width": CAMERA_RESOLUTION[0],
            "height": CAMERA_RESOLUTION[1],
            # Match camera_calibration_dual/render_low_icp_review.py and the
            # other calibration review renderers exactly.
            "anti_aliasing": 3,
            **(
                {"active_gpu": config["gpu"], "physics_gpu": config["gpu"], "multi_gpu": False}
                if config["gpu"] is not None
                else {}
            ),
        }
    )
    writers: dict[str, Any] = {}
    windows: list[Any] = []
    try:
        import carb  # noqa: F401 - Initialize Isaac settings after SimulationApp.
        import cv2
        import omni.physx
        import omni.replicator.core as rep
        import omni.timeline
        import omni.usd
        from isaacsim.core.prims import RigidPrim
        from isaacsim.core.simulation_manager import SimulationManager
        from isaacsim.core.utils.stage import is_stage_loading, open_stage
        from isaacsim.core.utils.types import ArticulationAction
        from pxr import Gf, Usd, UsdGeom, UsdLux  # noqa: F401 - Keep USD schema initialization.

        from .replay_actions import (
            CommandTrajectory,
            H264VideoWriter,
            _video_metadata,
            command_for_state,
        )

        if not open_stage(str(overlay)):
            raise RuntimeError(f"Could not open replay overlay: {overlay}")
        while is_stage_loading():
            simulation_app.update()
        for _ in range(30):
            simulation_app.update()
        context = omni.usd.get_context()
        stage = context.get_stage()
        stage.SetEditTarget(stage.GetSessionLayer())
        robot_visual_materials = _apply_robot_visual_material_mode(
            stage, config["robot_visual_material_mode"]
        )
        robot_visual_link_rebase = _apply_robot_visual_link_rebase(
            stage, config["robot_visual_link_rebase"]
        )
        urdf_robot_visuals = _apply_urdf_robot_visuals(stage, config["robot_visual_urdf_asset"])
        gripper_visual_materials = _apply_gripper_visual_material_mode(
            stage, config["gripper_visual_material_mode"]
        )
        stand_height_overrides = _apply_stand_height_offsets(
            stage, config["stand_height_offsets_mm"]
        )
        lifter_bindings = (
            _load_lifter_control_bindings(stage, config["lifter_middle_sleeve_ratio"])
            if lifter_trajectory is not None
            else {}
        )
        robot_base_link_overrides = _apply_robot_base_link_world_xyz(
            stage, config["robot_base_link_world_xyz_mm"]
        )
        robot_z_overrides = _apply_robot_z_offsets(stage, config["robot_z_offsets_mm"])
        gripper_mount_overrides = {
            arm: _apply_gripper_mount(
                stage,
                arm,
                float(config["gripper_base_mount_yaw_degrees"][arm]),
                float(config["gripper_base_mount_tool_axis_offsets_mm"][arm]),
            )
            for arm in ARM_ORDER
        }
        gripper_physics_material = _apply_gripper_physics_material(
            stage, config["gripper_physics_material"]
        )
        jenga_physics_material = _apply_jenga_physics_material(
            stage,
            config["jenga_physics_material"],
            expected_block_paths(config["scene_layout_mode"]),
        )
        target_meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
        if not math.isfinite(target_meters_per_unit) or target_meters_per_unit <= 0:
            raise ValueError(f"Invalid target stage metersPerUnit: {target_meters_per_unit}")
        linear_units_per_meter = 1.0 / target_meters_per_unit
        stage_units_per_mm = 0.001 / target_meters_per_unit
        gripper_linear_unit_overrides = {
            arm: _apply_gripper_linear_unit_scale(stage, arm, linear_units_per_meter)
            for arm in ARM_ORDER
        }
        single_arm_angular_unit_scale = 1.0
        if config["joint_gain_mode"] in CALIBRATED_GAIN_MODES:
            source_units = []
            for arm in ARM_ORDER:
                stage2 = stage2_by_arm[arm]
                if str(stage2.get("source_verification", "")).startswith(
                    "reference_replay_on_verified_runtime_stage"
                ):
                    source_meters_per_unit = stage2[
                        "reference_calibration_source_stage_meters_per_unit"
                    ]
                else:
                    source_stage = Usd.Stage.Open(str(stage2["source_stage"]))
                    if source_stage is None:
                        raise ValueError(
                            f"Could not open Stage 2 source stage: {stage2['source_stage']}"
                        )
                    source_meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(source_stage))
                parameter_meters_per_unit = float(
                    stage2.get("parameter_meters_per_unit", source_meters_per_unit)
                )
                source_units.append(parameter_meters_per_unit)
                stage2["source_stage_meters_per_unit"] = parameter_meters_per_unit
                stage2["calibration_source_stage_meters_per_unit"] = source_meters_per_unit
                stage2["target_stage_meters_per_unit"] = target_meters_per_unit
            if not np.allclose(source_units, source_units[0], atol=1e-12):
                raise ValueError("Arm-specific Stage 2 source unit systems differ")
            single_arm_angular_unit_scale = angular_quantity_numeric_scale(
                source_units[0], target_meters_per_unit
            )
        # Follow camera_calibration_dual's review path: author a neutral
        # session-only preview dome instead of using a viewport light preset.
        _configure_calibration_renderer(stage)
        wrist_camera_overrides = {}
        for camera_name, camera_path in CAMERA_PATHS.items():
            prim = stage.GetPrimAtPath(camera_path)
            if not prim:
                raise ValueError(f"Stage is missing {camera_name} camera: {camera_path}")
            if camera_name.endswith("_wrist"):
                camera = UsdGeom.Camera(prim)
                original_clip = camera.GetClippingRangeAttr().Get()
                xformable = UsdGeom.Xformable(prim)
                orient_op = next(
                    (
                        op
                        for op in xformable.GetOrderedXformOps()
                        if op.GetOpName() == "xformOp:orient"
                    ),
                    None,
                )
                if orient_op is None:
                    raise RuntimeError(f"Wrist camera has no local orientation op: {camera_path}")
                original_orientation = orient_op.Get()
                if config["wrist_camera_local_orientation_mode"] == "optical_frame_identity":
                    # The parent prim is already camera_color_optical_frame.
                    # The authored 180-degree X rotation therefore points the
                    # USD camera upward. Identity makes local -Z follow the
                    # optical frame's downward viewing axis.
                    orient_op.Set(Gf.Quatd(1.0, Gf.Vec3d(0.0, 0.0, 0.0)))
                # The wrist cameras sit below a robot root scaled from metres
                # into this millimetre stage. Their inherited (0.01, 100)
                # range otherwise clips the complete workcell to black.
                camera.GetClippingRangeAttr().Set(Gf.Vec2f(10.0, 10_000_000.0))
                active_orientation = orient_op.Get()
                world = UsdGeom.XformCache().GetLocalToWorldTransform(prim)
                forward = world.TransformDir(Gf.Vec3d(0.0, 0.0, -1.0)).GetNormalized()
                wrist_camera_overrides[camera_name] = {
                    "path": camera_path,
                    "local_orientation_mode": config["wrist_camera_local_orientation_mode"],
                    "original_local_orientation_wxyz": [
                        float(original_orientation.GetReal()),
                        *[float(item) for item in original_orientation.GetImaginary()],
                    ],
                    "active_local_orientation_wxyz": [
                        float(active_orientation.GetReal()),
                        *[float(item) for item in active_orientation.GetImaginary()],
                    ],
                    "default_time_world_forward": [float(item) for item in forward],
                    "original_clipping_range": [float(item) for item in original_clip],
                    "active_clipping_range": [
                        float(item) for item in camera.GetClippingRangeAttr().Get()
                    ],
                    "layer": "anonymous_session_layer",
                    "source_stage_saved": False,
                }
        embedded_profiles = {}
        for arm in ARM_ORDER:
            try:
                embedded = _stage_profile(stage, arm)
            except ValueError:
                embedded_profiles[arm] = {
                    "present": False,
                    "source": "external_manifest",
                    "manifest_path": profile["path"],
                    "manifest_sha256": profile["sha256"],
                }
            else:
                embedded["present"] = True
                embedded_profiles[arm] = embedded
                if embedded["profile_schema"] != profile["schema_version"]:
                    raise ValueError(f"{arm} embedded profile schema mismatch")
                if embedded["arm"] != arm or embedded["stand"] != profile["arm_mapping"][arm]:
                    raise ValueError(f"{arm} embedded profile arm/stand mismatch")
                if embedded["command_interpolation"] != "zoh_30hz" or embedded["physics_hz"] != 120:
                    raise ValueError(f"{arm} embedded profile runtime mismatch")
                expected = profile["arms"][arm]["parameters"]
                embedded_fields = [
                    ("armature", "armature_kg_m2"),
                    ("coulomb", "coulomb_friction"),
                    ("viscous", "viscous_friction"),
                    ("delays", "command_delay_seconds"),
                ]
                if config["joint_gain_mode"] == "profile":
                    embedded_fields[0:0] = [
                        ("kp", "kp_runtime_per_rad"),
                        ("kd", "kv_runtime_per_rad"),
                    ]
                for embedded_key, profile_key in embedded_fields:
                    wanted = [float(expected[name][profile_key]) for name in ARM_JOINTS]
                    if not np.allclose(embedded[embedded_key], wanted, rtol=1e-7, atol=1e-8):
                        raise ValueError(f"{arm} embedded {embedded_key} differs from manifest")

        provenance_hash = str(profile.get("provenance", {}).get("source_stage_sha256", ""))
        profile_source_stage_verified = provenance_hash == stage_identity_before
        profile_stage_mismatch_acknowledged = bool(
            config.get("allow_profile_stage_mismatch", False)
            and str(config.get("profile_stage_mismatch_acknowledgement", "")).strip()
        )
        if not profile_source_stage_verified and not profile_stage_mismatch_acknowledged:
            raise ValueError(
                "External alignment profile provenance does not match the selected camera stage: "
                f"profile={provenance_hash}, stage={stage_hash_before}"
            )

        camera_capture_paths, camera_intrinsic_sources, wrist_camera_mounts = (
            _create_wrist_camera_proxies(stage)
        )
        wrist_camera_aim_mode = config["wrist_camera_proxy_aim_mode"]
        wrist_camera_target_world = config["wrist_camera_proxy_target_world_mm"]
        wrist_camera_position_mode = config["wrist_camera_proxy_position_mode"]
        wrist_camera_tracking_mode = config["wrist_camera_proxy_tracking_mode"]
        wrist_camera_finger_behind_mm = config["wrist_camera_proxy_finger_behind_mm"]
        wrist_camera_finger_height_mm = config["wrist_camera_proxy_finger_height_mm"]
        wrist_camera_self_mask_mode = config["wrist_camera_self_mask_mode"]
        wrist_camera_self_mask_paths = _wrist_camera_self_mask_paths(wrist_camera_self_mask_mode)
        _update_wrist_camera_proxies(
            stage,
            wrist_camera_mounts,
            wrist_camera_aim_mode,
            wrist_camera_target_world,
            wrist_camera_position_mode,
            wrist_camera_finger_behind_mm,
            wrist_camera_finger_height_mm,
            wrist_camera_tracking_mode,
        )
        wrist_camera_proxy_report = {
            name: {
                "arm": mount["arm"],
                "source_path": mount["source_path"],
                "proxy_path": mount["proxy_path"],
                "gripper_path": mount["gripper_path"],
                "camera_to_gripper_matrix": [
                    [float(mount["camera_to_gripper"][row][column]) for column in range(4)]
                    for row in range(4)
                ],
                "runtime_follow": True,
                "aim_mode": wrist_camera_aim_mode,
                "target_world_mm": wrist_camera_target_world,
                "position_mode": wrist_camera_position_mode,
                "tracking_mode": wrist_camera_tracking_mode,
                "finger_behind_mm": wrist_camera_finger_behind_mm,
                "finger_height_mm": wrist_camera_finger_height_mm,
                "self_mask_mode": wrist_camera_self_mask_mode,
                "own_self_mask_paths": [
                    path
                    for path in wrist_camera_self_mask_paths
                    if path.startswith(f"{ROBOT_PATHS[mount['arm']]}/")
                ],
                "session_layer_only": True,
            }
            for name, mount in wrist_camera_mounts.items()
        }
        cameras, camera_calibration_report = _create_calibrated_cameras(
            stage,
            camera_capture_paths,
            camera_intrinsic_sources,
        )
        for _ in range(CALIBRATION_RENDER_WARMUP_FRAMES):
            _update_wrist_camera_proxies(
                stage,
                wrist_camera_mounts,
                wrist_camera_aim_mode,
                wrist_camera_target_world,
                wrist_camera_position_mode,
                wrist_camera_finger_behind_mm,
                wrist_camera_finger_height_mm,
                wrist_camera_tracking_mode,
            )
            preview = _capture_calibrated_cameras(
                cameras,
                rep,
                simulation_app,
                rt_subframes=3,
                fixed_camera_display_rotation_degrees=config[
                    "fixed_camera_display_rotation_degrees"
                ],
                stage=stage,
                wrist_self_mask_paths=wrist_camera_self_mask_paths,
            )
        for name, frame in preview.items():
            cv2.imwrite(
                str(output_dir / f"pre_replay_{name}_camera.png"),
                cv2.cvtColor(frame, cv2.COLOR_RGB2BGR),
            )

        timeline = omni.timeline.get_timeline_interface()
        timeline.play()
        timeline.commit()
        for _ in range(2):
            simulation_app.update()
        timeline.pause()
        timeline.commit()
        simulation_app.update()
        physics = omni.physx.get_physx_simulation_interface()
        simulation_time = float(SimulationManager.get_simulation_time())
        physics_dt = 1.0 / 120.0
        substeps = int(round(120.0 / episode.fps))
        if substeps != 4 or not math.isclose(episode.fps, 30.0, abs_tol=1e-9):
            raise ValueError(f"Expected 30 Hz episode and four substeps, got fps={episode.fps}")

        robots: dict[str, Any] = {}
        dof_names: dict[str, list[str]] = {}
        control_paths: dict[str, str] = {}
        runtime_profiles: dict[str, Any] = {}
        for arm in ARM_ORDER:
            path, robot, names = _initialize_robot(arm)
            robots[arm] = robot
            dof_names[arm] = names
            control_paths[arm] = path
            runtime_profiles[arm] = _apply_and_verify_profile(
                robot,
                names,
                profile,
                arm,
                config["joint_gain_mode"],
                stage1_by_arm.get(arm),
                stage2_by_arm.get(arm),
                single_arm_replay_profile,
                single_arm_angular_unit_scale,
                linear_units_per_meter,
                config.get("reproduction_seed"),
            )

        for _ in range(int(round(float(config["warmup_seconds"]) * 120.0))):
            physics.simulate(physics_dt, simulation_time)
            physics.fetch_results()
            simulation_time += physics_dt

        lifter_robot_baselines: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for arm in ARM_ORDER:
            position, orientation = robots[arm].get_world_pose()
            lifter_robot_baselines[arm] = (
                np.asarray(position, dtype=np.float64).copy(),
                np.asarray(orientation, dtype=np.float64).copy(),
            )
        last_lifter_displacement = np.zeros(len(ARM_ORDER), dtype=np.float64)
        max_lifter_substep_distance = np.zeros(len(ARM_ORDER), dtype=np.float64)
        initial_lifter_pose: dict[str, Any] = {}
        if lifter_trajectory is not None:
            initial_lifter_sample = lifter_trajectory.sample(0.0)
            initial_lifter_pose = _apply_lifter_pose(
                stage,
                lifter_bindings,
                robots,
                lifter_robot_baselines,
                initial_lifter_sample.displacement_mm,
                stage_units_per_mm,
            )
            last_lifter_displacement = initial_lifter_sample.displacement_mm.copy()

        action_by_arm = split_arm_vectors(episode.actions)
        observation_by_arm = split_arm_vectors(episode.observations)
        joint7_offsets = {arm: float(config["joint7_offsets_rad"][arm]) for arm in ARM_ORDER}
        for arm in ARM_ORDER:
            action_by_arm[arm] = action_by_arm[arm].copy()
            observation_by_arm[arm] = observation_by_arm[arm].copy()
            action_by_arm[arm][:, 6] += joint7_offsets[arm]
            observation_by_arm[arm][:, 6] += joint7_offsets[arm]
        trajectories = {
            arm: CommandTrajectory(
                action_by_arm[arm],
                episode.fps,
                "zoh",
                arm_command_delays_seconds=np.asarray(
                    runtime_profiles[arm]["command_delays_seconds"], dtype=np.float64
                ),
            )
            for arm in ARM_ORDER
        }
        for arm in ARM_ORDER:
            indices, positions, velocities, _ = command_for_state(
                observation_by_arm[arm][0], np.zeros(8), dof_names[arm]
            )
            positions, velocities = _scale_gripper_command_for_stage(
                indices,
                positions,
                velocities,
                dof_names[arm],
                linear_units_per_meter,
            )
            robots[arm].set_joint_positions(positions, joint_indices=indices)
            robots[arm].set_joint_velocities(velocities, joint_indices=indices)

        # Flush articulation transforms before recording the camera readback.
        # This also makes the diagnostic distinguish a bad optical transform
        # from a stale pre-initialization render-product transform.
        simulation_app.update()
        rep.orchestrator.step(
            rt_subframes=1,
            delta_time=0.0,
            pause_timeline=True,
        )
        simulation_app.update()
        _update_wrist_camera_proxies(
            stage,
            wrist_camera_mounts,
            wrist_camera_aim_mode,
            wrist_camera_target_world,
            wrist_camera_position_mode,
            wrist_camera_finger_behind_mm,
            wrist_camera_finger_height_mm,
            wrist_camera_tracking_mode,
            robots,
            cameras,
        )
        for name, mount in wrist_camera_mounts.items():
            wrist_camera_proxy_report[name]["camera_to_gripper_matrix"] = [
                [float(mount["camera_to_gripper"][row][column]) for column in range(4)]
                for row in range(4)
            ]
            wrist_camera_proxy_report[name]["locked_at_first_live_pose"] = (
                "locked_camera_to_gripper" in mount
            )
        simulation_app.update()
        initial_camera_world_poses = {}
        for name, camera in cameras.items():
            position, orientation = camera.get_world_pose(camera_axes="usd")
            initial_camera_world_poses[name] = {
                "position_stage_units": [float(item) for item in position],
                "orientation_wxyz": [float(item) for item in orientation],
            }

        block_paths = expected_block_paths(config["scene_layout_mode"])
        block_view = RigidPrim(
            list(block_paths),
            name="replay_block_state",
            reset_xform_properties=False,
            track_contact_forces=False,
            prepare_contact_sensors=False,
        )
        block_view.initialize()
        if set(block_view.prim_paths) != set(block_paths):
            raise RuntimeError("Block physics view does not match the authored task blocks")
        initial_block_poses = _collect_block_physics_poses(block_view)
        video_paths = {
            name: output_dir / f"{episode.path.stem}_{name}_camera_sim.mp4" for name in CAMERA_NAMES
        }
        video_paths["montage"] = output_dir / f"{episode.path.stem}_four_camera_sim.mp4"
        for name, path in video_paths.items():
            size = (2560, 480) if name == "montage" else CAMERA_RESOLUTION
            writers[name] = H264VideoWriter(path, episode.fps, size)

        joint_csv = output_dir / "dual_arm_tracking.csv"
        block_csv = output_dir / "block_poses.csv"
        lifter_csv = output_dir / "lifter_tracking.csv"
        fields = ["frame", "time_seconds", "arm", "source_index"]
        for name in ARM_JOINTS + ("gripper",):
            fields.extend(
                (f"command_{name}", f"measured_{name}", f"error_{name}", f"observation_{name}")
            )
        fields.extend(
            [
                "finger1_world_x",
                "finger1_world_y",
                "finger1_world_z",
                "finger2_world_x",
                "finger2_world_y",
                "finger2_world_z",
                "rendered_finger_link_distance",
            ]
        )
        block_fields = [
            "frame",
            "time_seconds",
            "block_path",
            "x",
            "y",
            "z",
            "qw",
            "qx",
            "qy",
            "qz",
        ]
        lifter_fields = [
            "frame",
            "time_seconds",
            "arm",
            "source_index",
            "action_target_mm",
            "measured_source_mm",
            "interpolated_position_mm",
            "zero_reference_mm",
            "displacement_mm",
            "velocity_mm_s",
            "full_stroke_displacement_mm",
            "middle_sleeve_displacement_mm",
            "robot_root_z_stage_units",
            "max_substep_distance_mm",
        ]
        tracking_errors: dict[str, list[np.ndarray]] = {arm: [] for arm in ARM_ORDER}
        with (
            joint_csv.open("w", newline="", encoding="utf-8") as joint_file,
            block_csv.open("w", newline="", encoding="utf-8") as block_file,
            lifter_csv.open("w", newline="", encoding="utf-8") as lifter_file,
        ):
            joint_writer = csv.DictWriter(joint_file, fieldnames=fields)
            block_writer = csv.DictWriter(block_file, fieldnames=block_fields)
            lifter_writer = csv.DictWriter(lifter_file, fieldnames=lifter_fields)
            joint_writer.writeheader()
            block_writer.writeheader()
            lifter_writer.writeheader()
            for frame_index in range(episode.selected_frames):
                capture_time = frame_index / episode.fps
                lifter_sample_at_capture = None
                lifter_pose_at_capture: dict[str, Any] = {}
                if lifter_trajectory is not None:
                    lifter_sample_at_capture = lifter_trajectory.sample(capture_time)
                    capture_distance = np.abs(
                        lifter_sample_at_capture.displacement_mm - last_lifter_displacement
                    )
                    max_lifter_substep_distance = np.maximum(
                        max_lifter_substep_distance, capture_distance
                    )
                    lifter_pose_at_capture = _apply_lifter_pose(
                        stage,
                        lifter_bindings,
                        robots,
                        lifter_robot_baselines,
                        lifter_sample_at_capture.displacement_mm,
                        stage_units_per_mm,
                    )
                    last_lifter_displacement = lifter_sample_at_capture.displacement_mm.copy()
                samples = {}
                for arm in ARM_ORDER:
                    sample = trajectories[arm].sample(capture_time)
                    samples[arm] = sample
                    indices, positions, velocities, _ = command_for_state(
                        sample.qpos, sample.qvel, dof_names[arm]
                    )
                    positions, velocities = _scale_gripper_command_for_stage(
                        indices,
                        positions,
                        velocities,
                        dof_names[arm],
                        linear_units_per_meter,
                    )
                    robots[arm].apply_action(
                        ArticulationAction(
                            joint_positions=positions,
                            joint_velocities=velocities,
                            joint_indices=indices,
                        )
                    )
                # Synchronize the rebased rendered link transforms without
                # advancing simulation time. Wrist proxy placement must track
                # the visible URDF fingers rather than their unre-based PhysX
                # body frames.
                rep.orchestrator.step(
                    rt_subframes=1,
                    delta_time=0.0,
                    pause_timeline=True,
                )
                simulation_app.update()
                _update_wrist_camera_proxies(
                    stage,
                    wrist_camera_mounts,
                    wrist_camera_aim_mode,
                    wrist_camera_target_world,
                    wrist_camera_position_mode,
                    wrist_camera_finger_behind_mm,
                    wrist_camera_finger_height_mm,
                    wrist_camera_tracking_mode,
                    robots,
                    cameras,
                )
                frames = _capture_calibrated_cameras(
                    cameras,
                    rep,
                    simulation_app,
                    rt_subframes=3 if frame_index == 0 else 1,
                    fixed_camera_display_rotation_degrees=config[
                        "fixed_camera_display_rotation_degrees"
                    ],
                    stage=stage,
                    wrist_self_mask_paths=wrist_camera_self_mask_paths,
                )
                for name in CAMERA_NAMES:
                    writers[name].write(cv2.cvtColor(frames[name], cv2.COLOR_RGB2BGR))
                writers["montage"].write(cv2.cvtColor(_montage(frames), cv2.COLOR_RGB2BGR))
                if frame_index == 0:
                    for name in CAMERA_NAMES:
                        cv2.imwrite(
                            str(output_dir / f"frame_000000_{name}_camera_lossless.png"),
                            cv2.cvtColor(frames[name], cv2.COLOR_RGB2BGR),
                        )
                for arm in ARM_ORDER:
                    measured = _measured_episode_vector(
                        robots[arm], dof_names[arm], target_meters_per_unit
                    )
                    if not np.isfinite(measured).all():
                        raise RuntimeError(
                            f"{arm} measured trajectory contains non-finite values at frame {frame_index}"
                        )
                    sample = samples[arm]
                    tracking_errors[arm].append(measured - sample.qpos)
                    row: dict[str, Any] = {
                        "frame": frame_index,
                        "time_seconds": capture_time,
                        "arm": arm,
                        "source_index": sample.source_index,
                    }
                    for index, name in enumerate(ARM_JOINTS + ("gripper",)):
                        row[f"command_{name}"] = float(sample.qpos[index])
                        row[f"measured_{name}"] = float(measured[index])
                        row[f"error_{name}"] = float(measured[index] - sample.qpos[index])
                        row[f"observation_{name}"] = float(
                            observation_by_arm[arm][frame_index, index]
                        )
                    finger1, finger2, finger_distance = _finger_link_distance(
                        stage, arm, target_meters_per_unit
                    )
                    for axis, value in zip(("x", "y", "z"), finger1, strict=True):
                        row[f"finger1_world_{axis}"] = value
                    for axis, value in zip(("x", "y", "z"), finger2, strict=True):
                        row[f"finger2_world_{axis}"] = value
                    row["rendered_finger_link_distance"] = finger_distance
                    joint_writer.writerow(row)
                if lifter_sample_at_capture is not None:
                    assert episode.lifter_action_mm is not None
                    assert episode.lifter_position_mm is not None
                    assert lifter_zero_mm is not None
                    for index, arm in enumerate(ARM_ORDER):
                        lifter_writer.writerow(
                            {
                                "frame": frame_index,
                                "time_seconds": capture_time,
                                "arm": arm,
                                "source_index": lifter_sample_at_capture.source_index,
                                "action_target_mm": float(
                                    episode.lifter_action_mm[frame_index, index]
                                ),
                                "measured_source_mm": float(
                                    episode.lifter_position_mm[frame_index, index]
                                ),
                                "interpolated_position_mm": float(
                                    lifter_sample_at_capture.position_mm[index]
                                ),
                                "zero_reference_mm": float(lifter_zero_mm[index]),
                                "displacement_mm": float(
                                    lifter_sample_at_capture.displacement_mm[index]
                                ),
                                "velocity_mm_s": float(
                                    lifter_sample_at_capture.velocity_mm_s[index]
                                ),
                                "full_stroke_displacement_mm": float(
                                    lifter_sample_at_capture.displacement_mm[index]
                                ),
                                "middle_sleeve_displacement_mm": float(
                                    lifter_sample_at_capture.displacement_mm[index]
                                    * config["lifter_middle_sleeve_ratio"]
                                ),
                                "robot_root_z_stage_units": float(
                                    lifter_pose_at_capture[arm][
                                        "robot_root_world_position_stage_units"
                                    ][2]
                                ),
                                "max_substep_distance_mm": float(
                                    max_lifter_substep_distance[index]
                                ),
                            }
                        )
                for block in _collect_block_physics_poses(block_view):
                    position = block["position"]
                    quat = block["quaternion_wxyz"]
                    block_writer.writerow(
                        {
                            "frame": frame_index,
                            "time_seconds": capture_time,
                            "block_path": block["block_path"],
                            "x": position[0],
                            "y": position[1],
                            "z": position[2],
                            "qw": quat[0],
                            "qx": quat[1],
                            "qy": quat[2],
                            "qz": quat[3],
                        }
                    )
                if frame_index == 0 or (frame_index + 1) % 100 == 0:
                    print(  # noqa: T201 - CLI progress.
                        f"dual replay frame {frame_index + 1}/{episode.selected_frames}", flush=True
                    )
                if frame_index == episode.selected_frames - 1:
                    continue
                for substep in range(substeps):
                    command_time = capture_time + substep * physics_dt
                    if substep > 0:
                        for arm in ARM_ORDER:
                            sample = trajectories[arm].sample(command_time)
                            indices, positions, velocities, _ = command_for_state(
                                sample.qpos, sample.qvel, dof_names[arm]
                            )
                            positions, velocities = _scale_gripper_command_for_stage(
                                indices,
                                positions,
                                velocities,
                                dof_names[arm],
                                linear_units_per_meter,
                            )
                            robots[arm].apply_action(
                                ArticulationAction(
                                    joint_positions=positions,
                                    joint_velocities=velocities,
                                    joint_indices=indices,
                                )
                            )
                    if lifter_trajectory is not None:
                        lift_step_time = min(
                            capture_time + (substep + 1) * physics_dt,
                            (frame_index + 1) / episode.fps,
                        )
                        lifter_step_sample = lifter_trajectory.sample(lift_step_time)
                        step_distance = np.abs(
                            lifter_step_sample.displacement_mm - last_lifter_displacement
                        )
                        max_lifter_substep_distance = np.maximum(
                            max_lifter_substep_distance, step_distance
                        )
                        _apply_lifter_pose(
                            stage,
                            lifter_bindings,
                            robots,
                            lifter_robot_baselines,
                            lifter_step_sample.displacement_mm,
                            stage_units_per_mm,
                        )
                        last_lifter_displacement = lifter_step_sample.displacement_mm.copy()
                    physics.simulate(physics_dt, simulation_time)
                    physics.fetch_results()
                    simulation_time += physics_dt

        timeline.pause()
        for writer in writers.values():
            writer.release()
        writers = {}
        metadata = _video_metadata(video_paths)
        errors = []
        for name, item in metadata.items():
            expected_size = (2560, 480) if name == "montage" else CAMERA_RESOLUTION
            if item["frame_count"] != episode.selected_frames:
                errors.append(f"{name}: frames={item['frame_count']}")
            if (item["width"], item["height"]) != expected_size:
                errors.append(f"{name}: size={item['width']}x{item['height']}")
            if not math.isclose(item["fps"], episode.fps, abs_tol=1e-6):
                errors.append(f"{name}: fps={item['fps']}")
        if errors:
            raise RuntimeError(f"Sim video verification failed: {errors}")
        error_report = {}
        for arm in ARM_ORDER:
            values = np.asarray(tracking_errors[arm])
            if not np.isfinite(values).all():
                raise RuntimeError(f"{arm} tracking errors contain non-finite values")
            error_report[arm] = {
                "rms": float(np.sqrt(np.mean(np.square(values)))),
                "max_abs": float(np.max(np.abs(values))),
                "finite": True,
            }
        final_block_poses = _collect_block_physics_poses(block_view)
        if lifter_trajectory is None:
            lifter_report = {
                "mode": config["lifter_replay"],
                "datasets": list(episode.lifter_datasets),
                "trajectory_replayed": False,
            }
        else:
            assert episode.lifter_position_mm is not None
            assert episode.lifter_action_mm is not None
            assert lifter_zero_mm is not None
            reconstruction_errors = []
            for frame_index in range(episode.selected_frames):
                sample = lifter_trajectory.sample(frame_index / episode.fps)
                reconstruction_errors.append(
                    sample.position_mm - episode.lifter_position_mm[frame_index]
                )
            reconstruction_error = np.asarray(reconstruction_errors)
            max_frame_delta = (
                np.max(np.abs(np.diff(episode.lifter_position_mm, axis=0)), axis=0)
                if episode.selected_frames > 1
                else np.zeros(len(ARM_ORDER))
            )
            fixed_components: dict[str, Any] = {}
            binding_report: dict[str, Any] = {}
            for arm in ARM_ORDER:
                fixed_components[arm] = {}
                for path, before in lifter_bindings[arm]["fixed_world_translation_mm"].items():
                    after = list(_world_translation(stage, stage.GetPrimAtPath(path)))
                    if not np.allclose(before, after, rtol=0.0, atol=1e-6):
                        raise RuntimeError(f"Fixed lifter component moved during replay: {path}")
                    fixed_components[arm][path] = {
                        "before_world_translation_mm": before,
                        "after_world_translation_mm": after,
                        "unchanged": True,
                    }
                binding_report[arm] = {
                    "control_path": lifter_bindings[arm]["control_path"],
                    "robot_root_path": lifter_bindings[arm]["robot_root_path"],
                    "full_stroke_paths": [
                        item["path"]
                        for item in lifter_bindings[arm]["moving"]
                        if math.isclose(item["factor"], 1.0, abs_tol=1e-12)
                    ],
                    "half_stroke_paths": [
                        item["path"]
                        for item in lifter_bindings[arm]["moving"]
                        if not math.isclose(item["factor"], 1.0, abs_tol=1e-12)
                    ],
                    "fixed_paths": list(lifter_bindings[arm]["fixed_paths"]),
                }
            lifter_report = {
                "mode": config["lifter_replay"],
                "joint_name": "joint8",
                "units": "millimetres",
                "allowed_levels_mm": list(LIFTER_LEVELS_MM),
                "command_source": "lifter/action_mm",
                "motion_source": "observations/lifter/position_mm",
                "interpolation": "piecewise_linear_120hz",
                "trajectory_replayed": True,
                "zero_reference_mm": {
                    arm: float(lifter_zero_mm[index]) for index, arm in enumerate(ARM_ORDER)
                },
                "middle_sleeve_ratio": float(config["lifter_middle_sleeve_ratio"]),
                "max_capture_reconstruction_error_mm": float(np.max(np.abs(reconstruction_error))),
                "max_recorded_frame_delta_mm": {
                    arm: float(max_frame_delta[index]) for index, arm in enumerate(ARM_ORDER)
                },
                "max_runtime_substep_distance_mm": {
                    arm: float(max_lifter_substep_distance[index])
                    for index, arm in enumerate(ARM_ORDER)
                },
                "initial_pose": initial_lifter_pose,
                "bindings": binding_report,
                "fixed_components": fixed_components,
                "session_layer_geometry_updates": True,
                "physics_view_robot_root_updates": True,
                "source_usd_saved": False,
            }
        stage_hash_after = sha256_file(stage_source)
        episode_hash_after = sha256_file(episode.path)
        if stage_hash_after != stage_hash_before or episode_hash_after != episode_hash_before:
            raise RuntimeError("Source HDF5 or USDZ changed during replay")
        report = {
            "status": "complete",
            "schema_version": "jenga_dual_replay_report_v1",
            "started_at": started_at,
            "finished_at": datetime.now(UTC).isoformat(),
            "elapsed_seconds": time.monotonic() - started,
            "config": str(config_path),
            "episode": {
                "path": str(episode.path),
                "sha256_before": episode_hash_before,
                "sha256_after": episode_hash_after,
                "source_frames": episode.source_frames,
                "replayed_frames": episode.selected_frames,
                "fps": episode.fps,
                "schema_version": episode.schema_version,
                "arm_order": list(episode.arm_order),
            },
            "stage": {
                "source_path": str(stage_source),
                "source_sha256_before": stage_hash_before,
                "source_sha256_after": stage_hash_after,
                "overlay_path": str(overlay),
                "overlay_sha256": sha256_file(overlay),
                "scene_layout_mode": config["scene_layout_mode"],
            },
            "coordinate_system": COORDINATE_SYSTEM,
            "data_origin_world_mm": list(DATA_ORIGIN_WORLD_MM),
            "task_scene_offset_mm": dict(config["task_scene_offset_mm"]),
            "lighting": {
                "mode": "camera_calibration_dual_default_preview_dome",
                "dome_path": CALIBRATION_DOME_PATH,
                "intensity": 650.0,
                "exposure": 0.0,
                "layer": "anonymous_session_layer",
                "source_stage_saved": False,
            },
            "rendering": {
                "renderer": "RaytracedLighting",
                "camera_resolution": list(CAMERA_RESOLUTION),
                "native_render_resolution": list(CAMERA_RESOLUTION),
                "anti_aliasing_mode": 3,
                "motion_blur": False,
                "capture_method": "calibrated_isaac_camera_rgb_annotators",
                "fixed_camera_display_rotation_degrees": config[
                    "fixed_camera_display_rotation_degrees"
                ],
                "calibration_render_reference": (
                    "camera_calibration_dual/render_low_icp_review.py"
                ),
                "warmup_frames": CALIBRATION_RENDER_WARMUP_FRAMES,
                "warmup_rt_subframes": 3,
                "cameras": camera_calibration_report,
            },
            "robot_visual_materials": robot_visual_materials,
            "gripper_visual_materials": gripper_visual_materials,
            "robot_visual_link_rebase": robot_visual_link_rebase,
            "urdf_robot_visuals": urdf_robot_visuals,
            "wrist_camera_runtime_overrides": wrist_camera_overrides,
            "wrist_camera_proxies": wrist_camera_proxy_report,
            "wrist_camera_self_mask": {
                "mode": wrist_camera_self_mask_mode,
                "paths_hidden_during_wrist_capture": list(wrist_camera_self_mask_paths),
                "fixed_camera_geometry_preserved": True,
                "render_only": True,
                "session_layer_only": True,
                "source_stage_saved": False,
            },
            "initial_camera_world_poses": initial_camera_world_poses,
            "task_blocks": {
                "count": len(block_paths),
                "pose_source": "physx_rigid_body_view",
                "position_units": "stage_units_mm",
                "poses_data_mm": [
                    list(item)
                    for item in task_block_poses_data_mm(
                        (
                            config["task_layout_top_right_data_mm"]["central"]["x"],
                            config["task_layout_top_right_data_mm"]["central"]["y"],
                        ),
                        (
                            config["task_layout_top_right_data_mm"]["rotated"]["x"],
                            config["task_layout_top_right_data_mm"]["rotated"]["y"],
                        ),
                    )
                ]
                if config["scene_layout_mode"] == "rebuild_task"
                else None,
                "poses_world_mm": [
                    list(item)
                    for item in offset_block_poses_mm(
                        config["task_scene_offset_mm"],
                        (
                            config["task_layout_top_right_data_mm"]["central"]["x"],
                            config["task_layout_top_right_data_mm"]["central"]["y"],
                        ),
                        (
                            config["task_layout_top_right_data_mm"]["rotated"]["x"],
                            config["task_layout_top_right_data_mm"]["rotated"]["y"],
                        ),
                    )
                ]
                if config["scene_layout_mode"] == "rebuild_task"
                else None,
                "initial": initial_block_poses,
                "final": final_block_poses,
            },
            "lifter_replay": config["lifter_replay"],
            "lifter_control": lifter_report,
            "stand_height_overrides": stand_height_overrides,
            "robot_base_link_overrides": robot_base_link_overrides,
            "robot_z_overrides": robot_z_overrides,
            "robot_root_motion": (
                "measured_lifter_joint8_physics_view_v1"
                if lifter_trajectory is not None
                else robot_base_link_overrides["mode"]
            ),
            "joint7_offsets_rad": joint7_offsets,
            "gripper_mount_overrides": gripper_mount_overrides,
            "gripper_physics_material": gripper_physics_material,
            "jenga_physics_material": jenga_physics_material,
            "gripper_linear_unit_overrides": gripper_linear_unit_overrides,
            "control_paths": control_paths,
            "profile": {
                "joint_gain_mode": config["joint_gain_mode"],
                "path": profile["path"],
                "sha256": profile["sha256"],
                "schema_version": profile["schema_version"],
                "provisional": bool(profile["provisional"]),
                "validation_passed": bool(profile["validation_passed"]),
                "publication_authorized": bool(profile["publication_authorized"]),
                "provisional_acknowledged": bool(profile["provisional_acknowledged"]),
                "embedded": embedded_profiles,
                "external_manifest_source_stage_verified": profile_source_stage_verified,
                "source_stage_mismatch_acknowledged": profile_stage_mismatch_acknowledged,
                "source_stage_mismatch_acknowledgement": config.get(
                    "profile_stage_mismatch_acknowledgement", ""
                ),
                "runtime_readback": runtime_profiles,
                "runtime_reference_report": (
                    {"path": reference_path, "sha256": sha256_file(reference_path)}
                    if reference_path
                    else None
                ),
            },
            "timing": {
                "command_hz": episode.fps,
                "physics_hz": 120.0,
                "substeps": 4,
                "interpolation": "zoh",
            },
            "gripper_control": {
                "mode": "single_arm_drive_zoh_30hz",
                "command_source": "leader/qpos total width",
                "single_arm_profile_retained": True,
                "arm_joint_control_unchanged": True,
                "direct_state_writes": False,
            },
            "tracking": error_report,
            "outputs": {
                "dual_arm_tracking_csv": str(joint_csv),
                "lifter_tracking_csv": str(lifter_csv),
                "block_poses_csv": str(block_csv),
                "sim_videos": metadata,
            },
            "source_files_unchanged": True,
        }
        json_dump(output_dir / "dual_replay_report.json", report)
        return report
    except Exception as exc:
        # Kit shutdown can terminate the process before Python prints an exception.
        failure = {
            "status": "error",
            "started_at": started_at,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        json_dump(output_dir / "dual_replay_error.json", failure)
        # Kit can exit before stderr flushes; keep the traceback on stdout as before.
        print(failure["traceback"], flush=True)  # noqa: T201 - Failure report.
        raise
    finally:
        for writer in writers.values():
            writer.release()
        windows.clear()
        simulation_app.close()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--overlay", type=Path, required=True)
    parser.add_argument("--max-frames", type=int, default=0)
    args = parser.parse_args(argv)
    report = run(args.config.resolve(), args.overlay.resolve(), args.max_frames)
    print(  # noqa: T201 - CLI report.
        json.dumps(
            {
                "status": report["status"],
                "report": str(
                    Path(load_config(args.config)["output_dir"]) / "dual_replay_report.json"
                ),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
