"""Lossless v1 config reader/validation; no generation implementation dependencies."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .backend.resources import default_path

DEFAULT_STAGE = str(default_path("stage"))
DEFAULT_PROFILE = str(default_path("motion_profile"))
DEFAULT_RUNTIME_PROFILES = str(default_path("runtime_profiles"))


def defaults():
    return {
        "task": "tower_transfer_3",
        "stage": DEFAULT_STAGE,
        "randomization_manifest": "",
        "real_root": "review-data/real",
        "output": "simulator-runs/generated",
        "spool": "spool",
        "profile_path": DEFAULT_PROFILE,
        "runtime_profile_dir": DEFAULT_RUNTIME_PROFILES,
        "layers_above_base": 3,
        "targets": (list)(),
        "seed": 0,
        "rollouts": 100,
        "episode_start": 0,
        "episode_ids": None,
        "gpus": (lambda: list(range(8)))(),
        "envs_per_gpu": 4,
        "gpu_total_aggregate_pairs_capacity": 1048576,
        "gpu_found_lost_aggregate_pairs_capacity": 1048576,
        "motion_profile": "human_like",
        "free_space_planner": "lula",
        "curobo_config": "",
        "curobo_sequence": "compact",
        "curobo_travel_speed_scale": 1.0,
        "curobo_placement_time_scale": 2.0,
        "curobo_transport_clearance_m": 0.1,
        "grasp_hold_force_n": 1.5,
        "max_seconds": 300.0,
        "placement_m": 0.005,
        "placement_degrees": 5.0,
        "placement_clearance_m": 0.001,
        "placement_release_compensation_m": 0.0,
        "placement_command_offsets_m": (dict)(),
        "settle_seconds": 0.5,
        "final_settle_seconds": 2.0,
        "collapse_drop_m": 0.015,
        "collapse_degrees": 30.0,
        "collapse_seconds": 0.25,
        "reset_settle_seconds": 1.0,
        "tower_jitter_m": 0.003,
        "center_block_jitter_m": 0.002,
        "max_recoveries": 0,
        "jpeg_quality": 95,
        "worker_timeout_seconds": 1800.0,
        "stress_grasp_offset_m": 0.0,
        "grasp_vertical_offset_m": 0.0,
        "grasp_point_offsets_m": (dict)(),
        "randomized_grasp_clearance_m": 0.006,
        "grasp_geometry_source_id": "",
        "post_verify_lift_m": 0.04,
        "gripper_open_width_m": 0.1,
        "gripper_closed_width_m": 0.03,
        "joint_command_max_speed_rad_s": 1.0,
        "joint_command_max_acceleration_rad_s2": 3.0,
        "joint_command_max_jerk_rad_s3": 30.0,
        "joint1_transport_max_speed_rad_s": 0.72,
        "joint_command_tracking_lead": 0.0,
        "gripper_close_speed_m_s": 0.09,
        "gripper_contact_width_m": 0.045,
        "gripper_contact_close_speed_m_s": 0.01,
        "gripper_open_speed_m_s": 0.16,
        "gripper_finger_effort_limit_n": 22.0,
        "render": True,
        "physics_usd_writeback": "auto",
        "physics_scene_mode": "auto",
        "anti_aliasing": 4,
        "dlss_mode": 2,
        "render_subframes": 4,
        "lighting_samples": 16,
        "wrist_near_clip_mm": 10.0,
        "renderer": "RaytracedLighting",
        "camera_batch_mode": "individual",
        "replicator_sync_only": False,
        "resume": False,
        "keep_local_episodes": False,
    }


def validate(self) -> None:
    if not Path(self.stage).is_file():
        raise ValueError(
            f"stage file does not exist: {self.stage}; provide --stage with an existing USD stage"
        )
    if (
        self.free_space_planner == "curobo"
        and self.curobo_sequence == "compact"
        and (
            self.grasp_point_offsets_m or self.grasp_vertical_offset_m or self.stress_grasp_offset_m
        )
    ):
        raise ValueError(
            "compact cuRobo grasps are block-centered; grasp-offset overrides are incompatible"
        )
    for name in (
        "gpu_total_aggregate_pairs_capacity",
        "gpu_found_lost_aggregate_pairs_capacity",
    ):
        capacity = getattr(self, name)
        if not isinstance(capacity, int) or not 1024 <= capacity <= 2**30:
            raise ValueError(f"{name.replace('_', '-')} must be an integer in [1024, 2^30]")
    expected_layers = {"tower_transfer_2": 2, "tower_transfer_3": 3}
    if self.task in expected_layers and self.layers_above_base != expected_layers[self.task]:
        raise ValueError(f"{self.task} requires layers-above-base={expected_layers[self.task]}")
    if self.layers_above_base < 1 or self.layers_above_base > 9:
        raise ValueError("layers-above-base must be in [1, 9]")
    if self.rollouts < 1 or self.envs_per_gpu < 1 or self.episode_start < 0:
        raise ValueError("rollouts and envs-per-gpu must be positive")
    if self.episode_ids is not None and (
        len(self.episode_ids) != self.rollouts
        or len(set(self.episode_ids)) != self.rollouts
        or any(i < 0 for i in self.episode_ids)
    ):
        raise ValueError("episode-ids must contain one unique nonnegative logical ID per rollout")
    if not self.gpus or len(set(self.gpus)) != len(self.gpus) or min(self.gpus) < 0:
        raise ValueError("gpus must contain unique nonnegative device indices")
    if self.motion_profile not in ("human_like", "procedural", "baseline", "smooth"):
        raise ValueError("unknown motion profile")
    if (
        not np.isfinite(self.grasp_hold_force_n)
        or not 0.25 <= self.grasp_hold_force_n <= self.gripper_finger_effort_limit_n
    ):
        raise ValueError(
            "grasp hold force must be positive and below the calibrated finger effort limit"
        )
    if (
        not 1.0 <= self.curobo_travel_speed_scale <= 3.0
        or not 1.0 <= self.curobo_placement_time_scale <= 3.0
    ):
        raise ValueError("cuRobo travel speed and placement time scales must be in [1, 3]")
    if not 0.03 <= self.curobo_transport_clearance_m <= 0.15:
        raise ValueError("cuRobo transport clearance must be in [0.03, 0.15] m")
    if self.curobo_sequence not in ("compact", "staged"):
        raise ValueError("unknown cuRobo sequence")
    if self.free_space_planner not in ("lula", "curobo"):
        raise ValueError("unknown free-space planner")
    if self.free_space_planner == "curobo" and not Path(self.curobo_config).is_file():
        raise ValueError(
            "cuRobo requires --curobo-config with calibrated robot and collision geometry"
        )
    if (
        not np.isfinite(self.joint_command_max_jerk_rad_s3)
        or self.joint_command_max_jerk_rad_s3 <= 0
    ):
        raise ValueError("joint jerk limit must be finite and positive")
    if self.physics_scene_mode not in ("auto", "full"):
        raise ValueError("physics-scene-mode must be auto or full")
    if self.physics_usd_writeback not in ("auto", "on"):
        raise ValueError("physics-usd-writeback must be auto or on")
    if not 0.0 <= self.center_block_jitter_m <= 0.01:
        raise ValueError("center-block-jitter-m must be in [0, 0.01]")
    if not 0.0 <= self.tower_jitter_m <= 0.01:
        raise ValueError("tower-jitter-m must be in [0, 0.01]")
    if self.anti_aliasing not in (3, 4) or self.dlss_mode not in (0, 1, 2, 3):
        raise ValueError(
            "anti-aliasing must be 3 (DLSS) or 4 (DLAA), and dlss-mode must be in [0, 3]"
        )
    if self.render_subframes < 1:
        raise ValueError("render-subframes must be positive")
    if self.lighting_samples is None or not 1 <= self.lighting_samples <= 64:
        raise ValueError("lighting-samples must be in [1, 64]")
    if not 5 <= self.wrist_near_clip_mm <= 60:
        raise ValueError("wrist-near-clip-mm must be in [5, 60]")
    if self.camera_batch_mode not in ("individual", "tiled", "tiled_split"):
        raise ValueError("camera-batch-mode must be individual, tiled, or tiled_split")
    if self.render and self.camera_batch_mode != "individual":
        raise ValueError(
            "Dataset publication requires individual cameras; "
            "tiled capture is restricted to unpublished diagnostics after the render-quality regression"
        )
    if not -0.012 <= self.grasp_vertical_offset_m <= 0.012:
        raise ValueError("grasp-vertical-offset-m must be in [-0.012, 0.012]")
    if not isinstance(self.grasp_point_offsets_m, dict):
        raise ValueError("grasp-point-offsets-m must map block IDs to local xyz offsets")
    if not isinstance(self.grasp_geometry_source_id, str):
        raise ValueError("grasp-geometry-source-id must be a string")
    if not 0.0 <= self.randomized_grasp_clearance_m <= 0.015:
        raise ValueError("randomized-grasp-clearance-m must be in [0, 0.015]")
    for name, value in self.grasp_point_offsets_m.items():
        try:
            offset = np.asarray(value, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError("grasp-point offsets must be finite xyz vectors") from exc
        if (
            not isinstance(name, str)
            or not name
            or offset.shape != (3,)
            or not np.isfinite(offset).all()
            or np.linalg.norm(offset) > 0.025
        ):
            raise ValueError(
                "grasp-point offsets require a block ID and finite xyz vector within 25 mm"
            )
    if not 0.01 <= self.post_verify_lift_m <= 0.04 or not 0.08 <= self.gripper_open_width_m <= 0.1:
        raise ValueError("invalid lift clearance or gripper opening")
    if not 0.0 <= self.gripper_closed_width_m <= 0.06:
        raise ValueError("gripper-closed-width-m must be in [0, 0.06]")
    if self.gripper_closed_width_m >= self.gripper_open_width_m:
        raise ValueError("gripper-closed-width-m must be smaller than gripper-open-width-m")
    if (
        not 0.2 <= self.joint_command_max_speed_rad_s <= 2.0
        or not 0.5 <= self.joint_command_max_acceleration_rad_s2 <= 10.0
        or not 0.3 <= self.joint1_transport_max_speed_rad_s <= self.joint_command_max_speed_rad_s
    ):
        raise ValueError("invalid joint command speed or acceleration envelope")
    if not 0.0 <= self.joint_command_tracking_lead <= 0.5:
        raise ValueError("joint-command-tracking-lead must be in [0, 0.5]")
    if (
        not 0.03 <= self.gripper_close_speed_m_s <= 0.2
        or not 0.03 <= self.gripper_open_speed_m_s <= 0.2
    ):
        raise ValueError("invalid gripper command speed envelope")
    if not self.gripper_closed_width_m < self.gripper_contact_width_m < self.gripper_open_width_m:
        raise ValueError("gripper-contact-width-m must lie between the closed and open commands")
    if not 0.005 <= self.gripper_contact_close_speed_m_s <= min(0.05, self.gripper_close_speed_m_s):
        raise ValueError("invalid gripper contact-zone close speed")
    if not 0.5 <= self.gripper_finger_effort_limit_n <= 22.0:
        raise ValueError("gripper-finger-effort-limit-n must be in [0.5, 22]")
    if self.max_recoveries != 0:
        raise ValueError("retries are disabled: the first failed attempt ends the episode")
    if not 1 <= self.jpeg_quality <= 100 or self.max_seconds <= 0:
        raise ValueError("invalid JPEG quality or episode duration")
    if not 0.003 <= self.placement_m <= 0.010 or not 3.0 <= self.placement_degrees <= 10.0:
        raise ValueError("placement tolerance must be 3-10 mm and 3-10 degrees")
    if not 0.001 <= self.placement_clearance_m <= 0.010:
        raise ValueError("placement-clearance-m must be in [0.001, 0.010]")
    if not 0.0 <= self.placement_release_compensation_m <= 0.025:
        raise ValueError("placement-release-compensation-m must be in [0, 0.025]")
    if not isinstance(self.placement_command_offsets_m, dict):
        raise ValueError("placement command offsets must map block IDs to xyz vectors")
    for block, offset in self.placement_command_offsets_m.items():
        allowed = {
            f"source_l{layer}_{slot}"
            for layer in range(1, self.layers_above_base + 1)
            for slot in ("a", "b")
        }
        vector = np.asarray(offset, dtype=float)
        if (
            block not in allowed
            or vector.shape != (3,)
            or not np.all(np.isfinite(vector))
            or np.linalg.norm(vector) > 0.010
        ):
            raise ValueError(
                "placement command offsets require known block IDs and finite xyz vectors of at most 10 mm"
            )
    if Path(self.output).resolve() == Path(self.real_root).resolve():
        raise ValueError("simulation output cannot be the real-data root")
    if self.randomization_manifest and not Path(self.randomization_manifest).is_file():
        raise ValueError(f"randomization manifest does not exist: {self.randomization_manifest}")


def validate_values(values):
    unknown = values.keys() - defaults().keys()
    if unknown:
        raise ValueError(f"Unsupported v1 config fields: {sorted(unknown)}")
    validate(SimpleNamespace(**(defaults() | values)))
