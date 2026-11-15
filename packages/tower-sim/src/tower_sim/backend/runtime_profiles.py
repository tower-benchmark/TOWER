# SPDX-License-Identifier: Apache-2.0
"""Apply the supplied replay calibration, independently of motion style profiles."""

from __future__ import annotations

from tower_sim.bench.asset_identity import station_identity

import json
from pathlib import Path

import numpy as np

from tower_sim.replay.dual_core import (
    angular_quantity_numeric_scale,
    load_profile,
    load_robot_alignment3_dual_parameters,
    load_single_arm_replay_profile,
)

from .assets import REPORT, checksum
from tower_eval.simulator.spec import ARM_ORDER, DT, JOINT_NAMES, digest
from tower_sim.config import DEFAULT_RUNTIME_PROFILES

MEMBERS = (
    "DUAL_ROBOT_ALIGNMENT_PROFILE.json",
    "dual_parameters.json",
    "REPLAY_PROFILE_V0_0_1.json",
)
PARAMETERS = (
    "kp_runtime_per_rad",
    "kv_runtime_per_rad",
    "armature_kg_m2",
    "coulomb_friction",
    "viscous_friction",
    "command_delay_seconds",
)


def runtime_profile_files(directory: str | Path = DEFAULT_RUNTIME_PROFILES) -> list[Path]:
    return [Path(directory) / name for name in (*MEMBERS, "bundle.json")]


def load_runtime_profiles(
    directory: str | Path,
    stage_sha256: str,
    *,
    target_meters_per_unit: float = 0.001,
    calibration_report: Path = REPORT,
) -> dict:
    directory = Path(directory).resolve()
    bundle = json.loads((directory / "bundle.json").read_text())
    if bundle.get("schema_version") != "jenga_runtime_profile_bundle_v1" or set(
        bundle.get("files", {})
    ) != set(MEMBERS):
        raise ValueError("Invalid runtime profile bundle manifest")
    hashes = {name: checksum(directory / name) for name in MEMBERS}
    if hashes != bundle["files"]:
        raise ValueError("Runtime profile bundle checksum mismatch")
    report = json.loads(calibration_report.read_text())
    # This user-supplied reference baseline is explicitly provisional. Preserve
    # that status instead of treating successful simulator readbacks as validation.
    alignment = load_profile(directory / MEMBERS[0], allow_provisional=True)
    if alignment["sha256"] != report["profile"]["sha256"]:
        raise ValueError("Alignment manifest does not match the reference replay")
    if alignment["arm_mapping"] != {"left": "StandA", "back": "StandB"}:
        raise ValueError("Runtime profile arm/stand mapping mismatch")
    stage_sha256 = station_identity(stage_sha256)
    compatibility = bundle.get("compatible_runtime_stages", {}).get(stage_sha256)
    if compatibility is not None:
        reference_hash = report["stage"]["source_sha256_before"]
        if (
            compatibility.get("reference_stage_sha256") != reference_hash
            or report["stage"]["source_sha256_after"] != reference_hash
            or not compatibility.get("reason", "").strip()
        ):
            raise ValueError("Invalid runtime stage compatibility evidence")
    dual = load_robot_alignment3_dual_parameters(
        directory / MEMBERS[1],
        replay_report=report,
        runtime_stage_sha256=stage_sha256,
        allow_stage_mismatch=compatibility is not None,
    )
    replay = load_single_arm_replay_profile(directory / MEMBERS[2])
    if (
        replay["physics_hz"] != 120
        or replay["command_source"] != "leader/qpos"
        or replay["command_interpolation"] != "zoh"
    ):
        raise ValueError("Replay profile must use 30 Hz absolute joint commands and 120 Hz physics")
    velocity_limits = np.asarray(
        [replay["arm_velocity_target_limits_rad_s"][joint] for joint in JOINT_NAMES[:7]],
        dtype=float,
    )
    if (
        velocity_limits.shape != (7,)
        or not np.isfinite(velocity_limits).all()
        or np.any(velocity_limits <= 0)
    ):
        raise ValueError("Invalid replay arm velocity-target limits")
    if replay["gripper"]["command_interpolation"] != "30 Hz ZOH":
        raise ValueError("Gripper profile must use 30 Hz zero-order hold")

    arms, comparisons = {}, {}
    for arm in ARM_ORDER:
        parameters = dual["arms"][arm]["joints"]
        for name in JOINT_NAMES[:7]:
            for key in PARAMETERS:
                if parameters[name][key] != alignment["arms"][arm]["parameters"][name][key]:
                    raise ValueError(
                        f"Dual parameters and alignment manifest disagree: {arm}/{name}/{key}"
                    )
        archived = report["profile"]["runtime_readback"][arm]
        units = archived["single_arm_gain_profile"]
        # The fit parameters are SI even though its calibration USD used mm.
        scale = angular_quantity_numeric_scale(
            units["source_stage_meters_per_unit"], target_meters_per_unit
        )
        archive_scale = angular_quantity_numeric_scale(
            units["source_stage_meters_per_unit"], units["target_stage_meters_per_unit"]
        )
        if not np.isclose(archive_scale, units["angular_quantity_numeric_scale"]):
            raise ValueError("Reference replay angular unit conversion is inconsistent")
        arrays = {
            key: np.array([parameters[joint][key] for joint in JOINT_NAMES[:7]])
            for key in PARAMETERS
        }
        delays = arrays.pop("command_delay_seconds")
        if np.any(delays >= DT):
            raise ValueError(
                "Streaming replay currently supports arm delays shorter than one 30 Hz command interval"
            )
        for key in arrays:
            arrays[key] *= scale
        arrays["command_delays_seconds"] = delays
        arrays["max_effort"] = units["source_effort_limit_nm"] * scale
        arms[arm] = arrays
        relative_errors = []
        float32_equal = True
        for key in PARAMETERS[:-1]:
            group = "gains" if key in PARAMETERS[:2] else "dynamics"
            readback_key = "static_friction" if key == "coulomb_friction" else key
            expected = arrays[key] * archive_scale / scale
            actual = np.array([archived[group][joint][readback_key] for joint in JOINT_NAMES[:7]])
            if not np.allclose(expected, actual, rtol=1e-6, atol=1e-6):
                raise ValueError(f"Supplied profile differs from reference readbacks: {arm}/{key}")
            relative_errors.extend(
                (np.abs(expected - actual) / np.maximum(1, np.abs(expected))).tolist()
            )
            float32_equal = float32_equal and bool(
                np.array_equal(expected.astype(np.float32), actual.astype(np.float32))
            )
        if not np.array_equal(delays, archived["command_delays_seconds"]):
            raise ValueError(f"Supplied command delays differ from the reference replay: {arm}")
        if replay["sha256"] != archived["gripper_profile"]["sha256"]:
            raise ValueError("Gripper replay profile does not match the reference replay")
        comparisons[arm] = {
            "matches_reference_readback": True,
            "float32_equal": float32_equal,
            "max_relative_difference": max(relative_errors),
        }

    # The replay uses this historical single-arm profile for gripper settings,
    # not its old scene placement offsets. V5 already has the installed robots.
    for key in ("robot_z_overrides", "stand_height_overrides"):
        if any(value["requested_offset_mm"] != 0 for value in report[key]["arms"].values()):
            raise ValueError("Reference replay requires an unsupported installation offset")
    if report["robot_base_link_overrides"]["mode"] != "not_applied":
        raise ValueError("Reference replay requires a robot base-link placement override")
    provenance = {
        "schema_version": "jenga_runtime_calibration_v1",
        "source": "supplied_runtime_profiles",
        "directory": str(directory),
        "bundle_sha256": digest(hashes),
        "files": hashes,
        "source_archive": bundle["source_archive"],
        "source_archive_sha256": bundle["source_archive_sha256"],
        "reference_report": str(calibration_report),
        "reference_report_sha256": checksum(calibration_report),
        "runtime_stage_sha256": stage_sha256,
        "source_verification": dual["source_verification"],
        "original_calibration_stage_verified": False,
        "original_calibration_stage": dual["source_stage"],
        "original_calibration_stage_sha256": dual["source_stage_sha256"],
        "joint_gain_mode": "robot_alignment3_dual_200hz",
        "arm_mapping": alignment["arm_mapping"],
        "parameter_meters_per_unit": units["source_stage_meters_per_unit"],
        "target_meters_per_unit": target_meters_per_unit,
        "angular_quantity_numeric_scale": scale,
        "physics_hz": 120,
        "action_hz": 30,
        "command_interpolation": "zero_order_hold",
        "command_delays_seconds": {
            arm: arms[arm]["command_delays_seconds"].tolist() for arm in ARM_ORDER
        },
        "gripper_delay_seconds": 0.0,
        "arm_velocity_target_limits_rad_s": velocity_limits.tolist(),
        "provisional": alignment["provisional"],
        "provisional_acknowledged": True,
        "validation_passed": alignment["validation_passed"],
        "publication_authorized": alignment["publication_authorized"],
        "interpretation": "Trajectory-domain calibration, not independently measured hardware dynamics. Readback verification is not real-robot alignment validation.",
        "reference_comparison": comparisons,
        "stage_compatibility": compatibility,
        "placement": {
            "mode": "authored_v5_installation",
            "robot_root_offsets_m": {arm: 0.0 for arm in ARM_ORDER},
            "legacy_single_arm_placement_applied": False,
            "legacy_single_arm_placement": replay["placement"],
            "robot_root_animation": False,
            "lifters_stationary": True,
        },
    }
    return {"arms": arms, "gripper": replay["gripper"], "provenance": provenance}


def apply_runtime_profile(
    robot, arm: str, profile: dict, finger_effort_limit_n: float | None = None
) -> dict:
    """Apply and read back the reference values on every row of an Isaac view."""
    names = robot.dof_names
    missing = set(JOINT_NAMES[:7] + ("gripper", "gripper_joint1", "gripper_joint2")) - set(names)
    if missing:
        raise RuntimeError(f"{arm} robot is missing calibrated DOFs: {sorted(missing)}")
    kp, kd = (np.asarray(array).copy() for array in robot.get_gains())
    effort = robot.get_max_efforts().copy()
    velocity = robot.get_joint_max_velocities().copy()
    armatures = robot.get_armatures().copy()
    friction = robot._physics_view.get_dof_friction_properties().copy()
    values, gripper = profile["arms"][arm], profile["gripper"]
    applied_finger_effort_n = float(
        gripper["finger_effort_limit"] if finger_effort_limit_n is None else finger_effort_limit_n
    )
    linear_scale = 1 / profile["provenance"]["target_meters_per_unit"]
    for i, name in enumerate(JOINT_NAMES[:7]):
        j = names.index(name)
        kp[:, j], kd[:, j] = values["kp_runtime_per_rad"][i], values["kv_runtime_per_rad"][i]
        armatures[:, j] = values["armature_kg_m2"][i]
        friction[:, j] = [
            values["coulomb_friction"][i],
            values["coulomb_friction"][i],
            values["viscous_friction"][i],
        ]
        effort[:, j] = values["max_effort"]
    for i, name in enumerate(gripper["dof_order"]):
        j = names.index(name)
        # Preserve the replay's literal linear-drive gains; scale only limits.
        kp[:, j], kd[:, j] = gripper["kp_runtime_per_meter"][i], gripper["kv_runtime_per_meter"][i]
        if i:
            effort[:, j] = applied_finger_effort_n * linear_scale
            velocity[:, j] = gripper["finger_velocity_limit_m_s"] * linear_scale
    indices = np.arange(len(kp), dtype=np.int32)
    robot.set_gains(kp, kd, save_to_usd=False)
    robot.set_armatures(armatures)
    robot.set_max_efforts(effort)
    robot._physics_view.set_dof_max_velocities(velocity.astype(np.float32), indices)
    robot._physics_view.set_dof_friction_properties(friction.astype(np.float32), indices)
    actual_kp, actual_kd = robot.get_gains()
    requested = {
        "kp": kp,
        "kd": kd,
        "armatures": armatures,
        "friction": friction,
        "max_efforts": effort,
        "max_velocities": velocity,
    }
    actual = {
        "kp": actual_kp,
        "kd": actual_kd,
        "armatures": robot.get_armatures(),
        "friction": robot._physics_view.get_dof_friction_properties(),
        "max_efforts": robot.get_max_efforts(),
        "max_velocities": robot.get_joint_max_velocities(),
    }
    for key, desired in requested.items():
        readback = np.asarray(actual[key])
        if (
            readback.shape != desired.shape
            or not np.isfinite(readback).all()
            or not np.allclose(readback, desired, rtol=1e-6, atol=1e-6)
        ):
            raise RuntimeError(f"{arm} runtime calibration {key} readback mismatch")
    return {
        "verified": True,
        "arm": arm,
        "environments": len(kp),
        "dof_names": list(names),
        "source_finger_effort_limit_n": float(gripper["finger_effort_limit"]),
        "applied_finger_effort_limit_n": applied_finger_effort_n,
        "readback": {key: np.asarray(value).tolist() for key, value in actual.items()},
        "save_to_usd": False,
    }


def delayed_actions(
    previous: np.ndarray, current: np.ndarray, delays: np.ndarray, substep: int
) -> np.ndarray:
    """Bounded streaming equivalent of CommandTrajectory; grippers have no delay."""
    result = current.copy()
    for arm in range(2):
        columns = slice(arm * 8, arm * 8 + 7)
        result[:, columns] = np.where(
            substep / 120 + 1e-12 < delays[arm], previous[:, columns], current[:, columns]
        )
    return result
