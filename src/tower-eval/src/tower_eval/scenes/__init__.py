# SPDX-License-Identifier: Apache-2.0
"""Calibrated task scenes, optionally bound to a legacy training checkpoint."""

import json
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from ..artifacts import sha256

# Scene, rendering and actuator settings only. Episode scheduling and policy
# command generation remain owned by the runner and the remote policy.
SCENE_FIELDS = {
    "task6_neighbor_height_extra_m",
    "task6_protrusion_m",
    "stage",
    "task",
    "layers_above_base",
    "randomization_manifest",
    "motion_profile",
    "tower_jitter_m",
    "center_block_jitter_m",
    "grasp_point_offsets_m",
    "profile_path",
    "runtime_profile_dir",
    "reset_settle_seconds",
    "gripper_finger_effort_limit_n",
    "renderer",
    "anti_aliasing",
    "dlss_mode",
    "render_subframes",
    "lighting_samples",
    "wrist_near_clip_mm",
    "camera_batch_mode",
    "replicator_sync_only",
    "jpeg_quality",
}


def compare(expected, actual, label):
    """Check recorded fields, allowing only numerical serialization roundoff."""
    if isinstance(expected, dict):
        if not isinstance(actual, dict):
            raise ValueError(f"Scene mismatch: {label}")
        for key, value in expected.items():
            if key not in actual:
                raise ValueError(f"Scene mismatch: {label}.{key} missing")
            compare(value, actual[key], f"{label}.{key}")
    elif isinstance(expected, (list, tuple)):
        if not isinstance(actual, (list, tuple)) or len(expected) != len(actual):
            raise ValueError(f"Scene mismatch: {label} length")
        for i, (left, right) in enumerate(zip(expected, actual)):
            compare(left, right, f"{label}[{i}]")
    elif isinstance(expected, (int, float)) and not isinstance(expected, bool):
        if not isinstance(actual, (int, float)) or not np.isclose(
            expected, actual, rtol=1e-9, atol=1e-12
        ):
            raise ValueError(f"Scene mismatch: {label}: {actual!r} != {expected!r}")
    elif expected != actual:
        raise ValueError(f"Scene mismatch: {label}: {actual!r} != {expected!r}")


def load_scene(path, config, server_config=None, usd_override=None):
    path = Path(path).resolve()
    preset = json.loads(path.read_text())
    if preset.get("schema") not in ("tower-scene-v1", "tower-training-scene-v1"):
        raise ValueError("Unsupported scene preset schema")
    if server_config is not None and preset["schema"] == "tower-training-scene-v1":
        if preset["checkpoint_id"] != server_config["checkpoint_id"]:
            raise ValueError("Scene preset checkpoint mismatch")
        for key in ("model_sha256", "norm_stats_sha256"):
            if preset["artifact_identity"][key] != server_config["artifact_identity"][key]:
                raise ValueError(f"Scene preset {key} mismatch")
    settings = preset["config"]
    unknown = set(settings) - SCENE_FIELDS
    if unknown:
        raise ValueError(f"Unsupported scene settings: {sorted(unknown)}")
    manifest = settings.get("randomization_manifest")
    if manifest is not None and not isinstance(manifest, str):
        raise ValueError("Training scene randomization manifest must be a path or null")
    if usd_override and Path(usd_override).resolve() != Path(settings["stage"]).resolve():
        raise ValueError("--usd conflicts with the checkpoint-bound training scene")
    for key, value in settings.items():
        setattr(config, key, value)
    verify_scene_files(preset)
    return preset, {
        "preset_path": str(path),
        "preset_sha256": sha256(path),
        "source": preset["source"],
        "verified_files": preset["files_sha256"],
    }


def verify_scene_files(preset):
    for path, expected in preset["files_sha256"].items():
        if sha256(path) != expected:
            raise ValueError(f"Scene asset checksum mismatch: {path}")


def verify_layout(preset, spec):
    compare(preset["expected_layout"], asdict(spec), "layout")


def frozen_layout(preset):
    """Load an explicitly selected task layout, without generating new poses.

    This is a scene asset, not a controller or a success definition. The
    source layout and physics/rendering references remain independently pinned.
    """
    from tower_eval.simulator.spec import Block, EpisodeSpec, Transfer

    layout = preset["expected_layout"]
    blocks = [Block(**block) for block in layout["blocks"]]
    transfers = [Transfer(**transfer) for transfer in layout["transfers"]]
    names = [block.name for block in blocks]
    if not blocks or len(set(names)) != len(names):
        raise ValueError("Frozen layout requires unique, nonempty block names")
    for block in blocks:
        pose = np.asarray(block.pose, dtype=float)
        if (
            not block.name
            or pose.shape != (7,)
            or not np.isfinite(pose).all()
            or not np.isclose(np.linalg.norm(pose[3:]), 1.0, rtol=0, atol=1e-5)
        ):
            raise ValueError("Invalid frozen block pose")
        for value in (block.dimensions_m, block.asset_scale, block.diagonal_inertia_kg_m2):
            if value is not None:
                array = np.asarray(value, dtype=float)
                if array.shape != (3,) or not np.isfinite(array).all() or np.any(array <= 0):
                    raise ValueError("Invalid frozen block geometry or inertia")
        if block.mass_kg is not None and (not np.isfinite(block.mass_kg) or block.mass_kg <= 0):
            raise ValueError("Invalid frozen block mass")
    for transfer in transfers:
        goal = np.asarray(transfer.goal, dtype=float)
        if (
            not 0 <= transfer.block_index < len(blocks)
            or transfer.block_id != blocks[transfer.block_index].name
            or transfer.arm not in (0, 1)
            or goal.shape != (7,)
            or not np.isfinite(goal).all()
            or not np.isclose(np.linalg.norm(goal[3:]), 1.0, rtol=0, atol=1e-5)
        ):
            raise ValueError("Invalid frozen task target")
    return EpisodeSpec(
        layout["logical_id"], layout["seed"], blocks, transfers, layout.get("layout_metadata", {})
    )


def initial_block_overrides(preset, spec):
    """Explicit diagnostic starting layout; never move blocks during rollout."""
    overrides = preset.get("initial_block_pose_overrides", {})
    if set(overrides) - {b.name for b in spec.blocks}:
        raise ValueError("Unknown initial block override")
    blocks = []
    for block in spec.blocks:
        if block.name in overrides:
            pose = np.asarray(overrides[block.name], dtype=float)
            if (
                block.fixed_on_table
                or pose.shape != (7,)
                or not np.isfinite(pose).all()
                or not np.isclose(np.linalg.norm(pose[3:]), 1.0)
            ):
                raise ValueError("Invalid movable-block initial pose override")
            block = replace(block, pose=tuple(pose))
        blocks.append(block)
    return replace(spec, blocks=blocks) if overrides else spec


def verify_runtime(preset, env, *, frozen=False):
    compare(preset["expected_visuals"], env.visual_reports, "block_visuals")
    compare(preset["expected_cameras"], env.camera_report, "cameras")
    # Inspect the authored physics properties as well as the generated spec.
    for i, expected in enumerate(preset["expected_visuals"]):
        prim = env.stage.GetPrimAtPath(f"{env.env_paths[0]}/Blocks/block_{i:02}")
        mass = prim.GetAttribute("physics:mass").Get()
        if not np.isclose(mass, expected["mass_kg"], rtol=1e-6, atol=1e-9):
            raise ValueError(f"Scene physics mass mismatch: block {i}")
        expected_inertia = expected.get("diagonal_inertia_kg_m2")
        # Canonical station blocks inherit their authored inertia. Randomized
        # blocks carry an explicit measured inertia that remains checksum- and
        # readback-verified here.
        if expected_inertia is not None:
            inertia = prim.GetAttribute("physics:diagonalInertia").Get()
            if not np.allclose(np.asarray(inertia) * 1e-6, expected_inertia, rtol=1e-6, atol=1e-10):
                raise ValueError(f"Scene physics inertia mismatch: block {i}")
    if frozen:
        return {
            "layout_matches_reference": True,
            "visuals_match_reference": True,
            "cameras_match_reference": True,
            "physics_properties_match_reference": True,
            "reference_kind": "frozen-task-profile",
        }
    return {
        "layout_matches_training": True,
        "visuals_match_training": True,
        "cameras_match_training": True,
        "physics_properties_match_training": True,
    }


def compare_initial_state(preset, state):
    reference = preset["initial_observation"]
    joint_indices = [*range(7), *range(8, 15)]
    joint_error = float(
        np.max(np.abs(state.qpos[joint_indices] - np.asarray(reference["qpos"])[joint_indices]))
    )
    block_error = float(
        np.max(np.linalg.norm(state.blocks[:, :3] - np.asarray(reference["blocks"])[:, :3], axis=1))
    )
    # Settling can differ slightly across source episodes. Record measured
    # differences and fail if the configured scene settles materially away.
    if joint_error > 0.001 or block_error > 0.001:
        raise ValueError(
            f"Training initial-state mismatch: joints={joint_error:.6g} rad, blocks={block_error:.6g} m"
        )
    return {
        "max_joint_error_rad": joint_error,
        "max_block_position_error_m": block_error,
        "joint_tolerance_rad": 0.001,
        "block_position_tolerance_m": 0.001,
    }
