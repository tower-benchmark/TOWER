# SPDX-License-Identifier: Apache-2.0
"""Pure-Python configuration, data, scene, and video helpers for dual replay."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

DUAL_SCHEMA = "jenga_nero_hdf5_v5_bimanual_lifter"
DUAL_SCHEMA_V6 = "jenga_nero_hdf5_v6_keyboard_lifter"
DUAL_SCHEMA_V7 = "jenga_nero_hdf5_v7_camera_roles"
DUAL_SCHEMA_V8 = "jenga_nero_hdf5_v8_camera_orientation"
DUAL_SCHEMA_V9 = "jenga_nero_hdf5_v9_discrete_lifter_levels"
DUAL_SCHEMAS = (
    DUAL_SCHEMA,
    DUAL_SCHEMA_V6,
    DUAL_SCHEMA_V7,
    DUAL_SCHEMA_V8,
    DUAL_SCHEMA_V9,
)
PROFILE_SCHEMA = "jenga_dual_robot_alignment_profile_v1"
ALIGNMENT_PARAMETER_SCHEMA = "robot_alignment_parameters_v1"
ROBOT_ALIGNMENT3_DUAL_PARAMETER_SCHEMA = "robot_alignment3_dual_parameters_v1"
LIFTER_REPLAY_IGNORED = "ignored_v1"
LIFTER_REPLAY_MEASURED = "measured_kinematic_v1"
LIFTER_REPLAY_MODES = (LIFTER_REPLAY_IGNORED, LIFTER_REPLAY_MEASURED)
LIFTER_REPLAY = LIFTER_REPLAY_IGNORED
LIFTER_LEVELS_MM = tuple(float(value) for value in range(0, 501, 50))
LIFTER_DATASETS = {
    "action": "lifter/action_mm",
    "position": "observations/lifter/position_mm",
    "valid": "observations/lifter/valid",
    "is_moving": "observations/lifter/is_moving",
}
ARM_ORDER = ("left", "back")
JOINT_NAMES = tuple(f"joint{index}" for index in range(1, 8)) + ("gripper",)
CAMERA_NAMES = ("low", "high", "left_wrist", "back_wrist")
CAMERA_DATASETS = {
    "low": "observations/images/low_camera",
    "high": "observations/images/high_camera",
    "left_wrist": "observations/images/left_wrist_camera",
    "back_wrist": "observations/images/back_wrist_camera",
}
CAMERA_DATASETS_V8 = {
    "low": "observations/images/left_back_camera",
    "high": "observations/images/right_front_camera",
    "left_wrist": "observations/images/left_wrist_camera",
    "back_wrist": "observations/images/back_wrist_camera",
}
BLOCK_ROOT = "/PNP_DUALD1200_STATION/ReplayTaskBlocks"
CALIBRATION_BLOCK_ROOT = "/PNP_DUALD1200_STATION/JengaBlocks"
TABLETOP_COLLISION_PATH = "/PNP_DUALD1200_STATION/ReplayTabletopCollision"
BLOCK_MASS_KG = 0.1056
FIXED_BASE_BLOCK_NAMES = frozenset(
    {
        "Foundation01",
        "Foundation02",
        "Foundation03",
        "RotatedBase01",
        "RotatedBase02",
        "RotatedBase03",
    }
)
TOWER_TRANSFER_3_BLOCK_NAMES = FIXED_BASE_BLOCK_NAMES | frozenset(
    f"TowerL{level}_{side:02d}" for level in (2, 3, 4) for side in (1, 2)
)
TABLE_DEPTH_MM = 660.0
DATA_ORIGIN_WORLD_MM = (0.0, -TABLE_DEPTH_MM)
COORDINATE_SYSTEM = (
    "data origin at tabletop back-right/top-right; +X left; +Y front; Z up; millimetres"
)
# The measured X/Y values locate each complete 150 x 150 mm assembly's
# top-right (back-right) footprint corner, not its center.  Block poses below
# are centers derived from those two measured corners.
CENTRAL_ASSEMBLY_TOP_RIGHT_DATA_MM = (285.0, 280.0)
ROTATED_ASSEMBLY_TOP_RIGHT_DATA_MM = (100.0, 60.0)


def task_block_poses_data_mm(
    central_top_right_data_mm: tuple[float, float] = CENTRAL_ASSEMBLY_TOP_RIGHT_DATA_MM,
    rotated_top_right_data_mm: tuple[float, float] = ROTATED_ASSEMBLY_TOP_RIGHT_DATA_MM,
) -> tuple[tuple[str, float, float, float, float], ...]:
    """Build the real test_20 layout from measured assembly footprint corners.

    The center is a complete six-level tower (three blocks per alternating
    level).  The corner assembly is the photographed 3-1-1 cross: an
    unchanged three-block foundation, one centered block aligned with the
    foundation on the second level, and one centered perpendicular block on
    the third level.  Both upper blocks use the requested +90-degree rotation
    from the earlier cross arrangement.
    """

    cx, cy = (float(value) for value in central_top_right_data_mm)
    rx, ry = (float(value) for value in rotated_top_right_data_mm)
    return (
        ("Foundation01", cx + 25.0, cy + 75.0, 15.0, -90.0),
        ("Foundation02", cx + 75.0, cy + 75.0, 15.0, -90.0),
        ("Foundation03", cx + 125.0, cy + 75.0, 15.0, -90.0),
        ("TowerL2_01", cx + 75.0, cy + 25.0, 45.0, 0.0),
        ("TowerL2_02", cx + 75.0, cy + 125.0, 45.0, 0.0),
        ("TowerL2_03", cx + 75.0, cy + 75.0, 45.0, 0.0),
        ("TowerL3_01", cx + 25.0, cy + 75.0, 75.0, -90.0),
        ("TowerL3_02", cx + 125.0, cy + 75.0, 75.0, -90.0),
        ("TowerL3_03", cx + 75.0, cy + 75.0, 75.0, -90.0),
        ("TowerL4_01", cx + 75.0, cy + 25.0, 105.0, 0.0),
        ("TowerL4_02", cx + 75.0, cy + 125.0, 105.0, 0.0),
        ("TowerL4_03", cx + 75.0, cy + 75.0, 105.0, 0.0),
        ("TowerL5_01", cx + 25.0, cy + 75.0, 135.0, -90.0),
        ("TowerL5_02", cx + 125.0, cy + 75.0, 135.0, -90.0),
        ("TowerL5_03", cx + 75.0, cy + 75.0, 135.0, -90.0),
        ("TowerL6_01", cx + 75.0, cy + 25.0, 165.0, 0.0),
        ("TowerL6_02", cx + 75.0, cy + 125.0, 165.0, 0.0),
        ("TowerL6_03", cx + 75.0, cy + 75.0, 165.0, 0.0),
        ("RotatedBase01", rx + 25.0, ry + 75.0, 15.0, -90.0),
        ("RotatedBase02", rx + 75.0, ry + 75.0, 15.0, -90.0),
        ("RotatedBase03", rx + 125.0, ry + 75.0, 15.0, -90.0),
        ("RotatedL2_03", rx + 75.0, ry + 75.0, 45.0, 90.0),
        ("RotatedL3_01", rx + 75.0, ry + 75.0, 75.0, 0.0),
    )


BLOCK_POSES_DATA_MM = task_block_poses_data_mm()

_DOCUMENTED_BLOCK_POSES_DATA_MM = (
    ("Foundation01", 310.0, 355.0, 15.0, -90.0),
    ("Foundation02", 360.0, 355.0, 15.0, -90.0),
    ("Foundation03", 410.0, 355.0, 15.0, -90.0),
    ("TowerL2_01", 360.0, 305.0, 45.0, 0.0),
    ("TowerL2_02", 360.0, 405.0, 45.0, 0.0),
    ("TowerL2_03", 360.0, 355.0, 45.0, 0.0),
    ("TowerL3_01", 310.0, 355.0, 75.0, -90.0),
    ("TowerL3_02", 410.0, 355.0, 75.0, -90.0),
    ("TowerL3_03", 360.0, 355.0, 75.0, -90.0),
    ("TowerL4_01", 360.0, 305.0, 105.0, 0.0),
    ("TowerL4_02", 360.0, 405.0, 105.0, 0.0),
    ("TowerL4_03", 360.0, 355.0, 105.0, 0.0),
    ("TowerL5_01", 310.0, 355.0, 135.0, -90.0),
    ("TowerL5_02", 410.0, 355.0, 135.0, -90.0),
    ("TowerL5_03", 360.0, 355.0, 135.0, -90.0),
    ("TowerL6_01", 360.0, 305.0, 165.0, 0.0),
    ("TowerL6_02", 360.0, 405.0, 165.0, 0.0),
    ("TowerL6_03", 360.0, 355.0, 165.0, 0.0),
    ("RotatedBase01", 125.0, 135.0, 15.0, -90.0),
    ("RotatedBase02", 175.0, 135.0, 15.0, -90.0),
    ("RotatedBase03", 225.0, 135.0, 15.0, -90.0),
    ("RotatedL2_03", 175.0, 135.0, 45.0, 90.0),
    ("RotatedL3_01", 175.0, 135.0, 75.0, 0.0),
)
assert BLOCK_POSES_DATA_MM == _DOCUMENTED_BLOCK_POSES_DATA_MM
BLOCK_POSES_MM = tuple(
    (name, x, DATA_ORIGIN_WORLD_MM[1] + y, z, yaw) for name, x, y, z, yaw in BLOCK_POSES_DATA_MM
)


def offset_block_poses_mm(
    offset_mm: Mapping[str, float] | None = None,
    central_top_right_data_mm: tuple[float, float] = CENTRAL_ASSEMBLY_TOP_RIGHT_DATA_MM,
    rotated_top_right_data_mm: tuple[float, float] = ROTATED_ASSEMBLY_TOP_RIGHT_DATA_MM,
) -> tuple[tuple[str, float, float, float, float], ...]:
    """Return task-block world poses with a session-only XYZ translation."""

    offset = offset_mm or {"x": 0.0, "y": 0.0, "z": 0.0}
    dx, dy, dz = (float(offset[axis]) for axis in ("x", "y", "z"))
    data_poses = task_block_poses_data_mm(central_top_right_data_mm, rotated_top_right_data_mm)
    return tuple(
        (name, x + dx, DATA_ORIGIN_WORLD_MM[1] + y + dy, z + dz, yaw)
        for name, x, y, z, yaw in data_poses
    )


@dataclass(frozen=True)
class DualEpisode:
    path: Path
    actions: np.ndarray
    observations: np.ndarray
    fps: float
    source_frames: int
    selected_frames: int
    schema_version: str
    arm_order: tuple[str, str]
    joint_names: tuple[str, ...]
    lifter_datasets: tuple[str, ...]
    lifter_action_mm: np.ndarray | None
    lifter_position_mm: np.ndarray | None
    lifter_valid: np.ndarray | None
    lifter_is_moving: np.ndarray | None
    camera_datasets: dict[str, str]


@dataclass(frozen=True)
class LifterSample:
    position_mm: np.ndarray
    displacement_mm: np.ndarray
    velocity_mm_s: np.ndarray
    source_index: int


class LinearLifterTrajectory:
    """Piecewise-linear measured lift motion sampled on a continuous clock."""

    def __init__(
        self,
        position_mm: np.ndarray,
        fps: float,
        zero_reference_mm: np.ndarray,
    ) -> None:
        values = np.asarray(position_mm, dtype=np.float64)
        zero = np.asarray(zero_reference_mm, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != len(ARM_ORDER):
            raise ValueError(f"lifter position must be N×2, got {values.shape}")
        if values.shape[0] < 1 or not np.isfinite(values).all():
            raise ValueError("lifter position must contain finite samples")
        if zero.shape != (len(ARM_ORDER),) or not np.isfinite(zero).all():
            raise ValueError("lifter zero reference must contain two finite values")
        if not math.isfinite(float(fps)) or float(fps) <= 0.0:
            raise ValueError("lifter fps must be finite and positive")
        self.position_mm = values
        self.zero_reference_mm = zero
        self.fps = float(fps)

    def sample(self, time_seconds: float) -> LifterSample:
        time_value = float(time_seconds)
        if not math.isfinite(time_value):
            raise ValueError("lifter sample time must be finite")
        coordinate = min(
            max(time_value * self.fps, 0.0),
            float(self.position_mm.shape[0] - 1),
        )
        left = int(math.floor(coordinate))
        right = min(left + 1, self.position_mm.shape[0] - 1)
        fraction = coordinate - left
        position = (
            self.position_mm[left] + (self.position_mm[right] - self.position_mm[left]) * fraction
        )
        if right == left:
            velocity = np.zeros(len(ARM_ORDER), dtype=np.float64)
        else:
            velocity = (self.position_mm[right] - self.position_mm[left]) * self.fps
        return LifterSample(
            position_mm=position,
            displacement_mm=position - self.zero_reference_mm,
            velocity_mm_s=velocity,
            source_index=left,
        )


def sha256_file(path: str | Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _json_attr(value: Any, name: str) -> Any:
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    if isinstance(value, str):
        try:
            return json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError(f"HDF5 attribute {name!r} is not valid JSON") from exc
    if isinstance(value, np.ndarray):
        return [item.decode() if isinstance(item, bytes) else str(item) for item in value]
    return value


def camera_datasets_for_schema(schema: str) -> dict[str, str]:
    return dict(
        CAMERA_DATASETS_V8
        if schema in (DUAL_SCHEMA_V7, DUAL_SCHEMA_V8, DUAL_SCHEMA_V9)
        else CAMERA_DATASETS
    )


def load_dual_episode(path: str | Path, max_frames: int = 0) -> DualEpisode:
    """Load dual-arm trajectories and optional v6+ two-axis lifter channels."""

    import h5py

    path = Path(path).expanduser().resolve()
    if max_frames < 0:
        raise ValueError("max_frames cannot be negative")
    if not path.is_file():
        raise FileNotFoundError(path)
    with h5py.File(path, "r") as handle:
        for dataset in ("leader/qpos", "observations/qpos"):
            if dataset not in handle:
                raise ValueError(f"Missing HDF5 dataset: {dataset}")
        actions_ds = handle["leader/qpos"]
        observations_ds = handle["observations/qpos"]
        action_shape = tuple(actions_ds.shape)
        source_frames = int(handle.attrs.get("valid_frames", actions_ds.shape[0]))
        selected = source_frames if max_frames == 0 else min(source_frames, max_frames)
        actions = np.asarray(actions_ds[:selected], dtype=np.float64)
        observations = np.asarray(observations_ds[:selected], dtype=np.float64)
        attrs = dict(handle.attrs)
        raw_schema = attrs.get("schema_version", "")
        if isinstance(raw_schema, bytes):
            raw_schema = raw_schema.decode()
        expected_camera_datasets = camera_datasets_for_schema(str(raw_schema))
        camera_datasets = {
            name: dataset for name, dataset in expected_camera_datasets.items() if dataset in handle
        }
        all_datasets: list[str] = []
        handle.visititems(
            lambda name, obj: all_datasets.append(name) if isinstance(obj, h5py.Dataset) else None
        )
        lifter_arrays = {
            name: np.asarray(handle[dataset][:selected])
            for name, dataset in LIFTER_DATASETS.items()
            if dataset in handle
        }

    if actions.shape != (selected, 16):
        raise ValueError(f"Expected leader/qpos shape (N, 16), got {actions.shape}")
    if observations.shape != actions.shape:
        raise ValueError(
            f"observations/qpos must match leader/qpos; got {observations.shape} and {actions.shape}"
        )
    if selected < 1:
        raise ValueError("Episode selection contains no frames")
    if source_frames != int(attrs.get("valid_frames", source_frames)):
        raise ValueError("Invalid valid_frames attribute")
    if not np.isfinite(actions).all() or not np.isfinite(observations).all():
        raise ValueError("Episode joint data contains non-finite values")
    schema = attrs.get("schema_version", "")
    if isinstance(schema, bytes):
        schema = schema.decode()
    if str(schema) not in DUAL_SCHEMAS:
        raise ValueError(f"Unexpected schema_version={schema!r}; expected one of {DUAL_SCHEMAS!r}")
    arm_order = tuple(_json_attr(attrs.get("arm_order", "[]"), "arm_order"))
    joint_names = tuple(_json_attr(attrs.get("joint_names", "[]"), "joint_names"))
    if arm_order != ARM_ORDER:
        raise ValueError(f"Unexpected arm_order={arm_order}; expected={ARM_ORDER}")
    if joint_names != JOINT_NAMES:
        raise ValueError(f"Unexpected joint_names={joint_names}; expected={JOINT_NAMES}")
    fps = float(attrs.get("fps", 0.0))
    if not math.isfinite(fps) or fps <= 0:
        raise ValueError(f"Episode fps must be positive, got {fps}")
    if source_frames != action_shape[0]:
        raise ValueError(
            f"valid_frames={source_frames} does not match dataset length={action_shape[0]}"
        )
    missing_cameras = sorted(set(CAMERA_NAMES) - set(camera_datasets))
    if missing_cameras:
        raise ValueError(f"Missing real camera datasets: {missing_cameras}")
    lifter_datasets = tuple(
        sorted(
            name
            for name in all_datasets
            if name.startswith("lifter/") or name.startswith("observations/lifter/")
        )
    )
    lifter_action_mm = lifter_arrays.get("action")
    lifter_position_mm = lifter_arrays.get("position")
    lifter_valid = lifter_arrays.get("valid")
    lifter_is_moving = lifter_arrays.get("is_moving")
    for name, values in (
        ("lifter/action_mm", lifter_action_mm),
        ("observations/lifter/position_mm", lifter_position_mm),
        ("observations/lifter/valid", lifter_valid),
        ("observations/lifter/is_moving", lifter_is_moving),
    ):
        if values is not None and values.shape != (selected, len(ARM_ORDER)):
            raise ValueError(f"Expected {name} shape ({selected}, 2), got {values.shape}")
    return DualEpisode(
        path=path,
        actions=actions,
        observations=observations,
        fps=fps,
        source_frames=source_frames,
        selected_frames=selected,
        schema_version=str(schema),
        arm_order=(str(arm_order[0]), str(arm_order[1])),
        joint_names=joint_names,
        lifter_datasets=lifter_datasets,
        lifter_action_mm=(
            None if lifter_action_mm is None else np.asarray(lifter_action_mm, dtype=np.float64)
        ),
        lifter_position_mm=(
            None if lifter_position_mm is None else np.asarray(lifter_position_mm, dtype=np.float64)
        ),
        lifter_valid=(None if lifter_valid is None else np.asarray(lifter_valid, dtype=bool)),
        lifter_is_moving=(
            None if lifter_is_moving is None else np.asarray(lifter_is_moving, dtype=bool)
        ),
        camera_datasets=camera_datasets,
    )


def lifter_zero_reference_mm(
    episode: DualEpisode,
    overrides_mm: Mapping[str, float] | None = None,
    nominal_tolerance_mm: float = 5.0,
) -> np.ndarray:
    """Resolve each arm's observed sensor value at the nominal 0 mm level."""

    arrays = {
        "action": episode.lifter_action_mm,
        "position": episode.lifter_position_mm,
        "valid": episode.lifter_valid,
        "is_moving": episode.lifter_is_moving,
    }
    missing = [LIFTER_DATASETS[key] for key, value in arrays.items() if value is None]
    if missing:
        raise ValueError("Measured lifter replay requires datasets: " + ", ".join(missing))
    action = np.asarray(episode.lifter_action_mm, dtype=np.float64)
    position = np.asarray(episode.lifter_position_mm, dtype=np.float64)
    valid = np.asarray(episode.lifter_valid, dtype=bool)
    moving = np.asarray(episode.lifter_is_moving, dtype=bool)
    if not np.isfinite(action).all() or not np.isfinite(position).all():
        raise ValueError("Measured lifter replay contains non-finite values")
    if not valid.all():
        invalid = int(np.count_nonzero(~valid))
        raise ValueError(f"Measured lifter replay contains {invalid} invalid samples")
    overrides = overrides_mm or {}
    if any(arm not in ARM_ORDER for arm in overrides):
        raise ValueError("lifter zero-reference overrides may contain only left/back")
    tolerance = float(nominal_tolerance_mm)
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("nominal lifter zero tolerance must be finite and non-negative")
    result = np.zeros(len(ARM_ORDER), dtype=np.float64)
    for index, arm in enumerate(ARM_ORDER):
        if arm in overrides:
            value = float(overrides[arm])
        else:
            mask = np.isclose(action[:, index], 0.0, rtol=0.0, atol=1e-6) & ~moving[:, index]
            if not np.any(mask):
                raise ValueError(f"Cannot estimate {arm} lifter zero: no stationary 0 mm samples")
            value = float(np.median(position[mask, index]))
        if not math.isfinite(value) or abs(value) > tolerance:
            raise ValueError(f"{arm} lifter zero reference {value!r} is outside ±{tolerance:g} mm")
        result[index] = value
    return result


def split_arm_vectors(values: np.ndarray) -> dict[str, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim < 1 or values.shape[-1] != 16:
        raise ValueError(f"Expected final dimension 16, got {values.shape}")
    return {"left": values[..., :8], "back": values[..., 8:]}


def load_profile(path: str | Path, allow_provisional: bool) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != PROFILE_SCHEMA:
        raise ValueError(f"Unexpected alignment profile schema: {data.get('schema_version')!r}")
    if data.get("runtime_physics_hz") != 120:
        raise ValueError("Dual alignment profile must target 120 Hz physics")
    if data.get("runtime_command_interpolation") != "zoh_30hz":
        raise ValueError("Dual alignment profile must target zoh_30hz")
    provisional = bool(data.get("provisional"))
    if provisional and not allow_provisional:
        raise PermissionError(
            "Alignment profile is provisional; set allow_provisional_profile=true explicitly"
        )
    if set(data.get("arms", {})) != set(ARM_ORDER):
        raise ValueError("Alignment profile must contain exactly left and back arms")
    for arm in ARM_ORDER:
        parameters = data["arms"][arm].get("parameters", {})
        if tuple(parameters) != JOINT_NAMES[:7]:
            raise ValueError(f"Profile {arm} joint order is invalid: {tuple(parameters)}")
        for joint in JOINT_NAMES[:7]:
            values = parameters[joint]
            for key in (
                "kp_runtime_per_rad",
                "kv_runtime_per_rad",
                "armature_kg_m2",
                "coulomb_friction",
                "viscous_friction",
                "command_delay_seconds",
            ):
                value = float(values[key])
                if not math.isfinite(value) or value < 0:
                    raise ValueError(f"Invalid {arm}/{joint}/{key}: {value}")
    data["path"] = str(path)
    data["sha256"] = sha256_file(path)
    data["provisional_acknowledged"] = bool(provisional and allow_provisional)
    return data


def profile_arm_parameters(profile: Mapping[str, Any], arm: str) -> dict[str, Any]:
    joints = {}
    for name in JOINT_NAMES[:7]:
        source = profile["arms"][arm]["parameters"][name]
        joints[name] = {
            "armature_kg_m2": float(source["armature_kg_m2"]),
            "coulomb_friction": float(source["coulomb_friction"]),
            "viscous_friction": float(source["viscous_friction"]),
        }
    return {"path": profile["path"], "sha256": profile["sha256"], "joints": joints}


def profile_delays(profile: Mapping[str, Any], arm: str) -> np.ndarray:
    return np.asarray(
        [
            profile["arms"][arm]["parameters"][name]["command_delay_seconds"]
            for name in JOINT_NAMES[:7]
        ],
        dtype=np.float64,
    )


def profile_gains(profile: Mapping[str, Any], arm: str) -> tuple[np.ndarray, np.ndarray]:
    parameters = profile["arms"][arm]["parameters"]
    return (
        np.asarray(
            [parameters[name]["kp_runtime_per_rad"] for name in JOINT_NAMES[:7]], dtype=np.float64
        ),
        np.asarray(
            [parameters[name]["kv_runtime_per_rad"] for name in JOINT_NAMES[:7]], dtype=np.float64
        ),
    )


def load_alignment_parameters(path: str | Path, required_fields: tuple[str, ...]) -> dict[str, Any]:
    """Load and validate a single-arm robot-alignment parameter file."""

    path = Path(path).expanduser().resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != ALIGNMENT_PARAMETER_SCHEMA:
        raise ValueError(f"Unexpected alignment parameter schema in {path}")
    joints = data.get("joints", {})
    if tuple(joints) != JOINT_NAMES[:7]:
        raise ValueError(f"Alignment parameter joint order is invalid in {path}")
    for joint in JOINT_NAMES[:7]:
        for field in required_fields:
            value = float(joints[joint][field])
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"Invalid {joint}/{field} in {path}: {value}")
    data["path"] = str(path)
    data["sha256"] = sha256_file(path)
    return data


def load_robot_alignment3_dual_parameters(
    path: str | Path,
    *,
    replay_report: Mapping[str, Any] | None = None,
    runtime_stage_sha256: str | None = None,
    allow_stage_mismatch: bool = False,
) -> dict[str, Any]:
    """Load the combined, arm-specific robot_alignment3 seed result.

    The file stores Isaac runtime values for both arms in one document. The
    returned arm dictionaries match ``load_alignment_parameters`` so the
    replay worker can use the same application and readback path.

    Portable bundles may use a replay report that binds the exact parameter
    bytes to a verified runtime stage. This does not verify the unavailable
    original calibration stage; the default still requires that stage.
    """

    path = Path(path).expanduser().resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != ROBOT_ALIGNMENT3_DUAL_PARAMETER_SCHEMA:
        raise ValueError(f"Unexpected robot_alignment3 dual schema in {path}")
    if data.get("gain_units") != "isaac_runtime_per_radian":
        raise ValueError(f"Unexpected robot_alignment3 gain units in {path}")
    if data.get("command_interpolation") != "zero_order_hold":
        raise ValueError(f"robot_alignment3 parameters must use zero_order_hold in {path}")
    if float(data.get("control_hz", 0.0)) != 200.0:
        raise ValueError(f"robot_alignment3 parameters must be a 200 Hz fit in {path}")
    if float(data.get("physics_hz", 0.0)) != 400.0:
        raise ValueError(f"robot_alignment3 parameters must be a 400 Hz fit in {path}")
    if tuple(data.get("seeds", ())) != (0,):
        raise ValueError(f"robot_alignment3 parameters must contain only seed 0 in {path}")
    source_stage = Path(str(data.get("source_stage", ""))).expanduser()
    if not source_stage.is_absolute():
        source_stage = path.parent / source_stage
    source_stage = source_stage.resolve()
    expected_source_hash = str(data.get("source_stage_sha256", ""))
    if len(expected_source_hash) != 64:
        raise ValueError(f"robot_alignment3 source stage hash is invalid in {path}")
    digest = sha256_file(path)
    source_verification = "original_calibration_stage"
    if replay_report is not None or runtime_stage_sha256 is not None:
        if replay_report is None or runtime_stage_sha256 is None:
            raise ValueError(
                "Portable parameters require both a replay report and runtime stage hash"
            )
        stage_evidence = replay_report["stage"]
        stage_mismatch = len(runtime_stage_sha256) != 64 or any(
            stage_evidence[key] != runtime_stage_sha256
            for key in ("source_sha256_before", "source_sha256_after")
        )
        if stage_mismatch and not allow_stage_mismatch:
            raise ValueError("Runtime stage does not match the reference replay report")
        if replay_report["profile"]["joint_gain_mode"] != "robot_alignment3_dual_200hz":
            raise ValueError("Reference replay did not use robot_alignment3_dual_200hz")
        for arm in ARM_ORDER:
            evidence = replay_report["profile"]["runtime_readback"][arm]["single_arm_gain_profile"]
            if any(evidence[key] != digest for key in ("stage1_sha256", "stage2_sha256")):
                raise ValueError(f"{arm} parameters do not match the reference replay report")
        source_verification = (
            "reference_replay_on_verified_runtime_stage_stage_mismatch_acknowledged"
            if stage_mismatch
            else "reference_replay_on_verified_runtime_stage"
        )
    else:
        if not source_stage.is_file():
            raise FileNotFoundError(f"robot_alignment3 source stage is missing: {source_stage}")
        actual_source_hash = sha256_file(source_stage)
        if actual_source_hash != expected_source_hash:
            raise ValueError(
                "robot_alignment3 source stage hash mismatch: "
                f"expected {expected_source_hash}, got {actual_source_hash}"
            )

    required_fields = (
        "armature_kg_m2",
        "coulomb_friction",
        "viscous_friction",
        "kp_runtime_per_rad",
        "kv_runtime_per_rad",
        "command_delay_seconds",
    )
    arms = data.get("arms", {})
    if set(arms) != set(ARM_ORDER):
        raise ValueError(f"robot_alignment3 parameters must contain left and back in {path}")
    normalized_arms: dict[str, Any] = {}
    for arm in ARM_ORDER:
        arm_data = arms[arm]
        if int(arm_data.get("winning_seed", -1)) != 0:
            raise ValueError(f"robot_alignment3 {arm} winning seed is not 0 in {path}")
        joints = arm_data.get("parameters", {})
        if tuple(joints) != JOINT_NAMES[:7]:
            raise ValueError(f"robot_alignment3 {arm} joint order is invalid in {path}")
        for joint in JOINT_NAMES[:7]:
            for field in required_fields:
                value = float(joints[joint][field])
                if not math.isfinite(value) or value < 0:
                    raise ValueError(
                        f"Invalid robot_alignment3 {arm}/{joint}/{field} in {path}: {value}"
                    )
        normalized_arms[arm] = {
            "schema_version": ROBOT_ALIGNMENT3_DUAL_PARAMETER_SCHEMA,
            "arm": arm,
            "reproduction_seed": 0,
            "joints": joints,
            "source_stage": str(source_stage),
            "source_stage_sha256": expected_source_hash,
            "path": str(path),
            "sha256": digest,
            "calibration_control_hz": 200.0,
            "calibration_physics_hz": 400.0,
            "calibration_command_interpolation": "zero_order_hold",
            "calibration_interpretation": str(data.get("calibration_interpretation", "")),
        }
        normalized_arms[arm]["source_verification"] = source_verification
        if replay_report is not None:
            evidence = replay_report["profile"]["runtime_readback"][arm]["single_arm_gain_profile"]
            for key in ("source_stage_meters_per_unit", "calibration_source_stage_meters_per_unit"):
                value = float(evidence[key])
                if not math.isfinite(value) or value <= 0:
                    raise ValueError(f"Invalid reference replay unit metadata: {arm}/{key}")
                normalized_arms[arm]["reference_" + key] = value
    return {
        "schema_version": ROBOT_ALIGNMENT3_DUAL_PARAMETER_SCHEMA,
        "path": str(path),
        "sha256": digest,
        "source_stage": str(source_stage),
        "source_stage_sha256": expected_source_hash,
        "source_verification": source_verification,
        "control_hz": 200.0,
        "physics_hz": 400.0,
        "arms": normalized_arms,
    }


def angular_quantity_numeric_scale(
    source_meters_per_unit: float, target_meters_per_unit: float
) -> float:
    """Convert torque-like numeric values between USD stage unit systems."""

    source = float(source_meters_per_unit)
    target = float(target_meters_per_unit)
    if not math.isfinite(source) or source <= 0:
        raise ValueError("source_meters_per_unit must be finite and positive")
    if not math.isfinite(target) or target <= 0:
        raise ValueError("target_meters_per_unit must be finite and positive")
    return (source / target) ** 2


def load_single_arm_replay_profile(path: str | Path) -> dict[str, Any]:
    """Load the accepted v0.0.1 replay profile used for gripper settings."""

    path = Path(path).expanduser().resolve()
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema") != "jenga_replay_profile_v1":
        raise ValueError(f"Unexpected single-arm replay profile schema in {path}")
    gripper = data.get("gripper", {})
    if tuple(gripper.get("dof_order", ())) != (
        "gripper",
        "gripper_joint1",
        "gripper_joint2",
    ):
        raise ValueError(f"Unexpected gripper DOF order in {path}")
    for key, length in (
        ("kp_runtime_per_meter", 3),
        ("kv_runtime_per_meter", 3),
    ):
        values = np.asarray(gripper.get(key, ()), dtype=np.float64)
        if values.shape != (length,) or not np.isfinite(values).all() or np.any(values < 0):
            raise ValueError(f"Invalid gripper {key} in {path}")
    for key in ("finger_effort_limit", "finger_velocity_limit_m_s"):
        value = float(gripper[key])
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"Invalid gripper {key} in {path}")
    data["path"] = str(path)
    data["sha256"] = sha256_file(path)
    return data


def expected_block_paths(scene_layout_mode: str = "rebuild_task") -> tuple[str, ...]:
    return tuple(
        f"{BLOCK_ROOT}/{item[0]}"
        for item in BLOCK_POSES_MM
        if scene_layout_mode != "tower_transfer_3" or item[0] in TOWER_TRANSFER_3_BLOCK_NAMES
    )


def write_tower_transfer_3_overlay(source_stage: str | Path, output_path: str | Path) -> Path:
    """Reuse the V5 station and its existing 12 transfer-task blocks unchanged."""
    from pxr import Sdf, Usd, UsdGeom

    source_stage = Path(source_stage).resolve()
    output_path = Path(output_path).resolve()
    if source_stage == output_path:
        raise ValueError("Task overlay must not overwrite the source scene")
    source = Usd.Stage.Open(str(source_stage))
    if source is None or not source.GetDefaultPrim():
        raise ValueError(f"Could not open the supplied scene: {source_stage}")
    layer = Sdf.Layer.CreateAnonymous("tower_transfer_3.usda")
    layer.subLayerPaths = [str(source_stage)]
    stage = Usd.Stage.Open(layer)
    if stage is None:
        raise ValueError(f"Could not compose the supplied scene: {source_stage}")
    stage.SetDefaultPrim(stage.GetPrimAtPath(source.GetDefaultPrim().GetPath()))
    UsdGeom.SetStageMetersPerUnit(stage, UsdGeom.GetStageMetersPerUnit(source))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.GetStageUpAxis(source))
    stage.SetTimeCodesPerSecond(source.GetTimeCodesPerSecond())
    for path in expected_block_paths():
        if path.rsplit("/", 1)[-1] not in TOWER_TRANSFER_3_BLOCK_NAMES:
            stage.OverridePrim(path).SetActive(False)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not layer.Export(str(output_path)):
        raise RuntimeError(f"Could not write task selection overlay: {output_path}")
    return output_path


def validate_authored_scene_with_pxr(
    path: str | Path, *, scene_layout_mode: str = "authored"
) -> dict[str, Any]:
    """Inspect the supplied replay scene without authoring an overlay or poses."""
    from pxr import Usd, UsdGeom, UsdPhysics

    path = Path(path).resolve()
    stage = Usd.Stage.Open(str(path))
    if stage is None:
        raise ValueError(f"Could not open authored scene: {path}")
    if UsdGeom.GetStageUpAxis(stage) != "Z" or not math.isclose(
        UsdGeom.GetStageMetersPerUnit(stage), 0.001, abs_tol=1e-12
    ):
        raise ValueError("Authored replay scene must be Z-up and millimetre-scaled")
    rows = []
    cache = UsdGeom.XformCache()
    paths = expected_block_paths(scene_layout_mode)
    root = stage.GetPrimAtPath(BLOCK_ROOT)
    if not root or {str(prim.GetPath()) for prim in root.GetChildren()} != set(paths):
        raise ValueError("Authored scene active block set does not match the selected task")
    for block_path in paths:
        prim = stage.GetPrimAtPath(block_path)
        if not prim or not prim.IsActive() or not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            raise ValueError(f"Missing authored rigid block: {block_path}")
        geometry = prim.GetPrototype() if prim.IsInstance() else prim
        if not geometry or not geometry.GetChildren():
            raise ValueError(f"Unresolved authored block geometry: {block_path}")
        body = UsdPhysics.RigidBodyAPI(prim)
        if not body.GetRigidBodyEnabledAttr().Get():
            raise ValueError(f"Disabled authored rigid block: {block_path}")
        rows.append(
            {
                "path": block_path,
                "position_world_mm": list(
                    cache.GetLocalToWorldTransform(prim).ExtractTranslation()
                ),
                "kinematic": bool(body.GetKinematicEnabledAttr().Get()),
                "instanceable": prim.IsInstanceable(),
            }
        )
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "scene_layout_mode": scene_layout_mode,
        "meters_per_unit": 0.001,
        "up_axis": "Z",
        "source_layout_preserved": scene_layout_mode == "authored",
        "retained_block_transforms_preserved": True,
        "block_paths": [row["path"] for row in rows],
        "authored_blocks": rows,
        "fixed_task_block_count": sum(row["kinematic"] for row in rows),
        "dynamic_task_block_count": sum(not row["kinematic"] for row in rows),
        "camera_paths": [
            str(prim.GetPath()) for prim in stage.Traverse() if prim.IsA(UsdGeom.Camera)
        ],
    }


def load_config(path: str | Path) -> dict[str, Any]:
    path = Path(path).expanduser().resolve()
    config = json.loads(path.read_text(encoding="utf-8"))
    if config.get("schema_version") != "jenga_dual_replay_config_v1":
        raise ValueError("Unexpected dual replay config schema")
    for key in ("stage", "profile_manifest", "output_dir"):
        value = Path(config[key]).expanduser()
        if not value.is_absolute():
            value = path.parent / value
        config[key] = str(value.resolve())
    config.setdefault("robot_visual_urdf_asset", None)
    if config["robot_visual_urdf_asset"] is not None:
        visual_asset = Path(config["robot_visual_urdf_asset"]).expanduser()
        if not visual_asset.is_absolute():
            visual_asset = path.parent / visual_asset
        config["robot_visual_urdf_asset"] = str(visual_asset.resolve())
    local = Path(config["fetch"]["local_path"]).expanduser()
    if not local.is_absolute():
        local = path.parent / local
    config["fetch"]["local_path"] = str(local.resolve())
    config["config_path"] = str(path)
    config.setdefault("scene_layout_mode", "rebuild_task")
    if config["scene_layout_mode"] not in {"rebuild_task", "authored", "tower_transfer_3"}:
        raise ValueError("scene_layout_mode must be rebuild_task, authored, or tower_transfer_3")
    config.setdefault("gpu", None)
    if config["gpu"] is not None and (type(config["gpu"]) is not int or config["gpu"] < 0):
        raise ValueError("gpu must be a non-negative integer or null")
    config.setdefault("runtime_profile_reference_report", None)
    if config["runtime_profile_reference_report"] is not None:
        reference = Path(config["runtime_profile_reference_report"]).expanduser()
        if not reference.is_absolute():
            reference = path.parent / reference
        config["runtime_profile_reference_report"] = str(reference.resolve())
    config.setdefault("physics_hz", 120.0)
    config.setdefault("warmup_seconds", 0.5)
    config.setdefault("fixed_camera_display_rotation_degrees", 0)
    display_rotation = int(config["fixed_camera_display_rotation_degrees"])
    if display_rotation not in (0, 180):
        raise ValueError("fixed_camera_display_rotation_degrees must be 0 or 180")
    config["fixed_camera_display_rotation_degrees"] = display_rotation
    config.setdefault("allow_provisional_profile", False)
    config.setdefault("allow_profile_stage_mismatch", False)
    config.setdefault("joint_gain_mode", "profile")
    config.setdefault("fixed_tower_bases", False)
    if not isinstance(config["fixed_tower_bases"], bool):
        raise ValueError("fixed_tower_bases must be a boolean")
    config.setdefault("lifter_replay", LIFTER_REPLAY_IGNORED)
    if config["lifter_replay"] not in LIFTER_REPLAY_MODES:
        raise ValueError(f"lifter_replay must be one of {LIFTER_REPLAY_MODES!r}")
    config.setdefault("lifter_zero_reference_mm", None)
    zero_reference = config["lifter_zero_reference_mm"]
    if zero_reference is not None:
        if not isinstance(zero_reference, dict) or any(
            arm not in ARM_ORDER for arm in zero_reference
        ):
            raise ValueError("lifter_zero_reference_mm must be null or contain only left/back")
        config["lifter_zero_reference_mm"] = {
            arm: float(value) for arm, value in zero_reference.items()
        }
        if not all(math.isfinite(value) for value in config["lifter_zero_reference_mm"].values()):
            raise ValueError("lifter_zero_reference_mm values must be finite")
    config.setdefault("lifter_middle_sleeve_ratio", 0.5)
    config["lifter_middle_sleeve_ratio"] = float(config["lifter_middle_sleeve_ratio"])
    if not (
        math.isfinite(config["lifter_middle_sleeve_ratio"])
        and 0.0 <= config["lifter_middle_sleeve_ratio"] <= 1.0
    ):
        raise ValueError("lifter_middle_sleeve_ratio must be in [0, 1]")
    config.setdefault("robot_visual_material_mode", "source")
    if config["robot_visual_material_mode"] not in {
        "source",
        "restore_descendant_bindings",
    }:
        raise ValueError(
            "robot_visual_material_mode must be 'source' or 'restore_descendant_bindings'"
        )
    config.setdefault("gripper_visual_material_mode", "source")
    if config["gripper_visual_material_mode"] not in {
        "source",
        "restore_authored_dark_gray_fingers",
    }:
        raise ValueError(
            "gripper_visual_material_mode must be 'source' or 'restore_authored_dark_gray_fingers'"
        )
    config.setdefault("wrist_camera_local_orientation_mode", "authored")
    if config["wrist_camera_local_orientation_mode"] not in {
        "authored",
        "optical_frame_identity",
    }:
        raise ValueError(
            "wrist_camera_local_orientation_mode must be 'authored' or 'optical_frame_identity'"
        )
    config.setdefault("wrist_camera_proxy_aim_mode", "source_mount")
    if config["wrist_camera_proxy_aim_mode"] not in {
        "source_mount",
        "look_at_task_center",
    }:
        raise ValueError(
            "wrist_camera_proxy_aim_mode must be 'source_mount' or 'look_at_task_center'"
        )
    target = config.setdefault("wrist_camera_proxy_target_world_mm", [0.0, 0.0, 0.0])
    if not isinstance(target, list) or len(target) != 3:
        raise ValueError("wrist_camera_proxy_target_world_mm must contain x, y, z")
    config["wrist_camera_proxy_target_world_mm"] = [float(value) for value in target]
    config.setdefault("wrist_camera_proxy_position_mode", "source_mount")
    if config["wrist_camera_proxy_position_mode"] not in {
        "source_mount",
        "finger_midpoint",
    }:
        raise ValueError(
            "wrist_camera_proxy_position_mode must be 'source_mount' or 'finger_midpoint'"
        )
    config.setdefault("wrist_camera_proxy_tracking_mode", "per_frame")
    if config["wrist_camera_proxy_tracking_mode"] not in {
        "per_frame",
        "lock_to_gripper_at_first_live_pose",
    }:
        raise ValueError(
            "wrist_camera_proxy_tracking_mode must be 'per_frame' or "
            "'lock_to_gripper_at_first_live_pose'"
        )
    config.setdefault("wrist_camera_self_mask_mode", "none")
    if config["wrist_camera_self_mask_mode"] not in {
        "none",
        "hide_camera_mount",
        "hide_gripper_and_camera_mount",
    }:
        raise ValueError(
            "wrist_camera_self_mask_mode must be 'none', "
            "'hide_camera_mount', or 'hide_gripper_and_camera_mount'"
        )
    for key, default in (
        ("wrist_camera_proxy_finger_behind_mm", 70.0),
        ("wrist_camera_proxy_finger_height_mm", 80.0),
    ):
        config.setdefault(key, default)
        config[key] = float(config[key])
        if not math.isfinite(config[key]) or config[key] < 0.0:
            raise ValueError(f"{key} must be finite and non-negative")
    config.setdefault("robot_visual_link_rebase", False)
    if not isinstance(config["robot_visual_link_rebase"], bool):
        raise ValueError("robot_visual_link_rebase must be a boolean")
    config.setdefault("joint7_offsets_rad", {arm: 0.0 for arm in ARM_ORDER})
    config.setdefault("task_scene_offset_mm", {axis: 0.0 for axis in ("x", "y", "z")})
    task_scene_offset = config["task_scene_offset_mm"]
    if set(task_scene_offset) != {"x", "y", "z"}:
        raise ValueError("task_scene_offset_mm must contain x, y, and z")
    config["task_scene_offset_mm"] = {
        axis: float(task_scene_offset[axis]) for axis in ("x", "y", "z")
    }
    if not all(math.isfinite(value) for value in config["task_scene_offset_mm"].values()):
        raise ValueError("task_scene_offset_mm values must be finite")
    config.setdefault(
        "task_layout_top_right_data_mm",
        {
            "central": {
                "x": CENTRAL_ASSEMBLY_TOP_RIGHT_DATA_MM[0],
                "y": CENTRAL_ASSEMBLY_TOP_RIGHT_DATA_MM[1],
            },
            "rotated": {
                "x": ROTATED_ASSEMBLY_TOP_RIGHT_DATA_MM[0],
                "y": ROTATED_ASSEMBLY_TOP_RIGHT_DATA_MM[1],
            },
        },
    )
    layout = config["task_layout_top_right_data_mm"]
    if set(layout) != {"central", "rotated"} or any(
        set(layout[name]) != {"x", "y"} for name in ("central", "rotated")
    ):
        raise ValueError("task_layout_top_right_data_mm must contain central/rotated x and y")
    config["task_layout_top_right_data_mm"] = {
        name: {axis: float(layout[name][axis]) for axis in ("x", "y")}
        for name in ("central", "rotated")
    }
    if not all(
        math.isfinite(value)
        for point in config["task_layout_top_right_data_mm"].values()
        for value in point.values()
    ):
        raise ValueError("task_layout_top_right_data_mm values must be finite")
    config.setdefault("stand_height_offsets_mm", {arm: 0.0 for arm in ARM_ORDER})
    stand_height_offsets = config["stand_height_offsets_mm"]
    if set(stand_height_offsets) != set(ARM_ORDER):
        raise ValueError("stand_height_offsets_mm must contain left and back")
    config["stand_height_offsets_mm"] = {arm: float(stand_height_offsets[arm]) for arm in ARM_ORDER}
    if not all(math.isfinite(value) for value in config["stand_height_offsets_mm"].values()):
        raise ValueError("stand_height_offsets_mm values must be finite")
    config.setdefault("robot_z_offsets_mm", {arm: 0.0 for arm in ARM_ORDER})
    robot_z_offsets = config["robot_z_offsets_mm"]
    if set(robot_z_offsets) != set(ARM_ORDER):
        raise ValueError("robot_z_offsets_mm must contain left and back")
    config["robot_z_offsets_mm"] = {arm: float(robot_z_offsets[arm]) for arm in ARM_ORDER}
    if not all(math.isfinite(value) for value in config["robot_z_offsets_mm"].values()):
        raise ValueError("robot_z_offsets_mm values must be finite")
    config.setdefault("robot_base_link_world_xyz_mm", None)
    base_link_targets = config["robot_base_link_world_xyz_mm"]
    if base_link_targets is not None:
        if set(base_link_targets) != set(ARM_ORDER) or any(
            set(base_link_targets[arm]) != {"x", "y", "z"} for arm in ARM_ORDER
        ):
            raise ValueError("robot_base_link_world_xyz_mm must contain left/back x, y, and z")
        config["robot_base_link_world_xyz_mm"] = {
            arm: {axis: float(base_link_targets[arm][axis]) for axis in ("x", "y", "z")}
            for arm in ARM_ORDER
        }
        if not all(
            math.isfinite(value)
            for target in config["robot_base_link_world_xyz_mm"].values()
            for value in target.values()
        ):
            raise ValueError("robot_base_link_world_xyz_mm values must be finite")
    offsets = config["joint7_offsets_rad"]
    if set(offsets) != set(ARM_ORDER):
        raise ValueError("joint7_offsets_rad must contain left and back")
    config["joint7_offsets_rad"] = {arm: float(offsets[arm]) for arm in ARM_ORDER}
    if not all(math.isfinite(value) for value in config["joint7_offsets_rad"].values()):
        raise ValueError("joint7_offsets_rad values must be finite")
    config.setdefault("gripper_base_mount_yaw_degrees", {arm: 0.0 for arm in ARM_ORDER})
    mount_yaws = config["gripper_base_mount_yaw_degrees"]
    if set(mount_yaws) != set(ARM_ORDER):
        raise ValueError("gripper_base_mount_yaw_degrees must contain left and back")
    config["gripper_base_mount_yaw_degrees"] = {arm: float(mount_yaws[arm]) for arm in ARM_ORDER}
    if not all(math.isfinite(value) for value in config["gripper_base_mount_yaw_degrees"].values()):
        raise ValueError("gripper_base_mount_yaw_degrees values must be finite")
    config.setdefault(
        "gripper_base_mount_tool_axis_offsets_mm",
        {arm: 0.0 for arm in ARM_ORDER},
    )
    mount_offsets = config["gripper_base_mount_tool_axis_offsets_mm"]
    if set(mount_offsets) != set(ARM_ORDER):
        raise ValueError("gripper_base_mount_tool_axis_offsets_mm must contain left and back")
    config["gripper_base_mount_tool_axis_offsets_mm"] = {
        arm: float(mount_offsets[arm]) for arm in ARM_ORDER
    }
    if not all(
        math.isfinite(value) for value in config["gripper_base_mount_tool_axis_offsets_mm"].values()
    ):
        raise ValueError("gripper_base_mount_tool_axis_offsets_mm values must be finite")
    config.setdefault("gripper_physics_material", None)
    material = config["gripper_physics_material"]
    if material is not None:
        if set(material) != {"static_friction", "dynamic_friction", "restitution"}:
            raise ValueError(
                "gripper_physics_material must contain static_friction, "
                "dynamic_friction, and restitution"
            )
        material = {key: float(value) for key, value in material.items()}
        if (
            not all(math.isfinite(value) for value in material.values())
            or material["static_friction"] < 0.0
            or material["dynamic_friction"] < 0.0
            or not 0.0 <= material["restitution"] <= 1.0
        ):
            raise ValueError(
                "gripper_physics_material friction must be non-negative and "
                "restitution must be in [0, 1]"
            )
        config["gripper_physics_material"] = material
    config.setdefault("jenga_physics_material", None)
    material = config["jenga_physics_material"]
    if material is not None:
        if set(material) != {"static_friction", "dynamic_friction"}:
            raise ValueError(
                "jenga_physics_material must contain static_friction and dynamic_friction"
            )
        material = {key: float(value) for key, value in material.items()}
        if (
            not all(math.isfinite(value) for value in material.values())
            or material["static_friction"] < 0.0
            or material["dynamic_friction"] < 0.0
            or material["dynamic_friction"] > material["static_friction"]
        ):
            raise ValueError(
                "jenga_physics_material friction must be finite, non-negative, "
                "and dynamic friction must not exceed static friction"
            )
        config["jenga_physics_material"] = material
    if config["joint_gain_mode"] not in {
        "profile",
        "stage_default",
        "isaac_urdf_importer_default",
        "single_arm_v0_0_1",
        "dual_arm_seed0",
        "mixed_left_single_back_seed0",
        "robot_alignment3_dual_200hz",
    }:
        raise ValueError(
            "joint_gain_mode must be 'profile', 'stage_default', or "
            "'isaac_urdf_importer_default', 'single_arm_v0_0_1', or "
            "'dual_arm_seed0', 'mixed_left_single_back_seed0', or "
            "'robot_alignment3_dual_200hz'"
        )
    if (
        config["allow_provisional_profile"]
        and not str(config.get("provisional_profile_acknowledgement", "")).strip()
    ):
        raise ValueError(
            "allow_provisional_profile=true requires a non-empty "
            "provisional_profile_acknowledgement"
        )
    if (
        config["allow_profile_stage_mismatch"]
        and not str(config.get("profile_stage_mismatch_acknowledgement", "")).strip()
    ):
        raise ValueError(
            "allow_profile_stage_mismatch=true requires a non-empty "
            "profile_stage_mismatch_acknowledgement"
        )
    if config["joint_gain_mode"] in {
        "single_arm_v0_0_1",
        "mixed_left_single_back_seed0",
    }:
        for key in (
            "single_arm_stage1_parameters",
            "single_arm_stage2_parameters",
            "single_arm_replay_profile",
        ):
            if key not in config:
                raise ValueError(f"{key} is required for single_arm_v0_0_1")
            value = Path(config[key]).expanduser()
            if not value.is_absolute():
                value = path.parent / value
            config[key] = str(value.resolve())
    if config["joint_gain_mode"] in {
        "dual_arm_seed0",
        "mixed_left_single_back_seed0",
    }:
        if int(config.get("reproduction_seed", -1)) != 0:
            raise ValueError("dual_arm_seed0 requires reproduction_seed=0")
        config["reproduction_seed"] = 0
        for key in (
            "dual_arm_stage1_parameters",
            "dual_arm_stage2_parameters",
        ):
            values = config.get(key, {})
            if set(values) != set(ARM_ORDER):
                raise ValueError(f"{key} must contain left and back")
            resolved = {}
            for arm in ARM_ORDER:
                value = Path(values[arm]).expanduser()
                if not value.is_absolute():
                    value = path.parent / value
                resolved[arm] = str(value.resolve())
            config[key] = resolved
        key = "single_arm_replay_profile"
        if key not in config:
            raise ValueError(f"{key} is required for dual_arm_seed0")
        value = Path(config[key]).expanduser()
        if not value.is_absolute():
            value = path.parent / value
        config[key] = str(value.resolve())
    if config["joint_gain_mode"] == "robot_alignment3_dual_200hz":
        if int(config.get("reproduction_seed", -1)) != 0:
            raise ValueError("robot_alignment3_dual_200hz requires reproduction_seed=0")
        config["reproduction_seed"] = 0
        key = "robot_alignment3_dual_parameters"
        if key not in config:
            raise ValueError(f"{key} is required for robot_alignment3_dual_200hz")
        value = Path(config[key]).expanduser()
        if not value.is_absolute():
            value = path.parent / value
        config[key] = str(value.resolve())
        key = "robot_alignment3_parameter_meters_per_unit"
        if key not in config:
            raise ValueError(f"{key} is required for robot_alignment3_dual_200hz")
        value = float(config[key])
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{key} must be finite and positive")
        config[key] = value
        key = "single_arm_replay_profile"
        if key not in config:
            raise ValueError(f"{key} is required for robot_alignment3_dual_200hz")
        value = Path(config[key]).expanduser()
        if not value.is_absolute():
            value = path.parent / value
        config[key] = str(value.resolve())
    if float(config["physics_hz"]) != 120.0:
        raise ValueError("This launcher requires physics_hz=120")
    if config["scene_layout_mode"] in {"authored", "tower_transfer_3"} and any(
        config["task_scene_offset_mm"].values()
    ):
        raise ValueError("Authored scene mode cannot apply task_scene_offset_mm")
    return config


def json_dump(path: str | Path, value: Any) -> None:
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
