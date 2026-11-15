"""Evaluation configuration; historical generation keys are read only as v1 data."""

from dataclasses import asdict, dataclass, field

from .backend.resources import default_path

DEFAULT_STAGE = str(default_path("stage"))
DEFAULT_PROFILE = str(default_path("motion_profile"))
DEFAULT_RUNTIME_PROFILES = str(default_path("runtime_profiles"))


@dataclass
class SimConfig:
    task: str = "tower_transfer_3"
    stage: str = DEFAULT_STAGE
    output: str = "simulator-runs/generated"
    profile_path: str = DEFAULT_PROFILE
    runtime_profile_dir: str = DEFAULT_RUNTIME_PROFILES
    layers_above_base: int = 3
    seed: int = 0
    gpus: list[int] = field(default_factory=lambda: list(range(8)))
    envs_per_gpu: int = 4
    gpu_total_aggregate_pairs_capacity: int = 1048576
    gpu_found_lost_aggregate_pairs_capacity: int = 1048576
    max_seconds: float = 300.0
    placement_m: float = 0.005
    placement_degrees: float = 5.0
    settle_seconds: float = 0.5
    final_settle_seconds: float = 2.0
    collapse_drop_m: float = 0.015
    collapse_degrees: float = 30.0
    collapse_seconds: float = 0.25
    reset_settle_seconds: float = 1.0
    jpeg_quality: int = 95
    gripper_finger_effort_limit_n: float = 22.0
    render: bool = True
    physics_usd_writeback: str = "auto"
    physics_scene_mode: str = "auto"
    anti_aliasing: int = 4
    dlss_mode: int = 2
    render_subframes: int = 4
    lighting_samples: int | None = 16
    wrist_near_clip_mm: float = 10.0
    renderer: str = "RaytracedLighting"
    camera_batch_mode: str = "individual"
    replicator_sync_only: bool = False

    def __post_init__(self):
        # Reader metadata is deliberately outside dataclass fields/to_dict().
        self._v1_generation_values: dict = {}

    @classmethod
    def from_v1(cls, values):
        from dataclasses import fields

        from ._config_v1 import defaults

        unknown = values.keys() - defaults().keys()
        if unknown:
            raise ValueError(f"Unsupported v1 config fields: {sorted(unknown)}")
        supported = {field.name for field in fields(cls)}
        config = cls(**{key: value for key, value in values.items() if key in supported})
        config._v1_generation_values = {
            key: value for key, value in values.items() if key not in supported
        }
        return config

    def validate(self) -> None:
        from ._config_v1 import validate_values

        validate_values(getattr(self, "_v1_generation_values", {}) | self.to_dict())

    def to_dict(self) -> dict:
        return asdict(self)
