# SPDX-License-Identifier: Apache-2.0
"""Capture-setting guards and visibility checks for reference materials."""

import numpy as np

SAMPLING_KEYS = (
    "/rtx/directLighting/sampledLighting/samplesPerPixel",
    "/rtx/indirectDiffuse/fetchSampleCount",
    "/rtx/reflections/sampledLighting/samplesPerPixel",
    "/rtx/directLighting/domeLight/sampleCount",
    "/rtx/shadows/sampleCount",
)


def verify_capture_settings(settings, config) -> None:
    expected = {
        "/rtx/rendermode": config.renderer,
        "/rtx/post/aa/op": config.anti_aliasing,
        "/rtx/post/dlss/execMode": config.dlss_mode,
        "/rtx/post/motionblur/enabled": False,
        "/omni/replicator/captureMotionBlur": False,
        "/rtx-transient/dlssg/enabled": False,
    }
    if config.lighting_samples is not None:
        expected.update({key: config.lighting_samples for key in SAMPLING_KEYS})
    differences = {
        key: {"requested": value, "actual": settings.get(key)}
        for key, value in expected.items()
        if settings.get(key) != value
    }
    if differences:
        raise RuntimeError(
            f"Renderer settings disagree with the recording configuration: {differences}"
        )


def check_wrist_near_plane(
    blocks, offset_m, camera_position_mm, camera_xyzw, intrinsic, near_mm, label
):
    """Conservatively reject block bounds intersecting a visible near plane."""
    from itertools import product

    from scipy.spatial.transform import Rotation

    corners = np.array(list(product((-1, 1), repeat=3))) * [75, 25, 15]
    points = np.einsum("bij,kj->bki", Rotation.from_quat(blocks[:, 3:]).as_matrix(), corners)
    points += (blocks[:, :3] + offset_m)[:, None, :] * 1000
    optical = np.einsum(
        "bki,ij->bkj", points - camera_position_mm, Rotation.from_quat(camera_xyzw).as_matrix()
    ) * [1, -1, -1]
    x, y, z = optical.transpose(2, 0, 1)
    fx, fy, cx, cy = intrinsic[0, 0], intrinsic[1, 1], intrinsic[0, 2], intrinsic[1, 2]
    planes = np.stack(
        (x + cx / fx * z, (640 - cx) / fx * z - x, y + cy / fy * z, (480 - cy) / fy * z - y, z)
    )
    visible = ~np.any(np.all(planes <= 0, axis=2), axis=0)
    minimum = float(np.min(z[visible], initial=np.inf))
    if minimum < near_mm:
        raise RuntimeError(
            f"Wrist near plane may clip a task block: {label} ({minimum:.3f} < {near_mm} mm)"
        )
    return minimum


def check_wood_region(image: np.ndarray, pixels: np.ndarray, label: str) -> dict:
    if image.shape != (480, 640, 3) or image.dtype != np.uint8 or not np.isfinite(pixels).all():
        raise RuntimeError(f"invalid camera buffer or projection: {label}")
    lower = np.maximum(np.floor(pixels.min(axis=0)).astype(int) - 3, [0, 0])
    upper = np.minimum(np.ceil(pixels.max(axis=0)).astype(int) + 4, [640, 480])
    region = image[lower[1] : upper[1], lower[0] : upper[0]].astype(np.int16)
    if min(region.shape[:2], default=0) < 8:
        raise RuntimeError(f"tower is outside the calibrated camera: {label}")
    wood = (region[..., 0] - region[..., 1] > 10) & (region[..., 1] - region[..., 2] > 10)
    fraction = float(wood.mean())
    if fraction < 0.08 or wood.sum() < 100:
        raise RuntimeError(
            f"reference wood is missing from projected tower region: {label} ({fraction:.3f})"
        )
    return {
        "label": label,
        "bounds_xyxy": [*lower.tolist(), *upper.tolist()],
        "wood_fraction": fraction,
    }


def verify_render_profile(rendering, require_clear=False):
    if (
        rendering["capture_profile"]
        not in ("replay_individual_v1", "individual_v2", "single_pass_v3")
        or rendering["camera_batch_mode"] != "individual"
        or rendering["replicator_sync_only"]
    ):
        raise RuntimeError("Unexpected render capture profile")
    if require_clear:
        expected = {
            "/rtx/post/aa/op": 4,
            "/rtx/post/dlss/execMode": 2,
            "/rtx/rendermode": "RaytracedLighting",
            "/rtx/post/motionblur/enabled": False,
        }
        expected.update({key: 16 for key in SAMPLING_KEYS})
        if (
            rendering.get("schema_version") != "jenga_render_capture_v3"
            or rendering["capture_profile"] != "single_pass_v3"
            or rendering.get("visibility_passes") != 1
            or rendering.get("wrist_self_mask") is not False
            or rendering.get("wrist_near_clip_mm") != 10
            or rendering.get("near_plane_block_guard") is not True
            or rendering.get("render_subframes") != 4
            or any(
                rendering.get("settings", {}).get(key) != value for key, value in expected.items()
            )
        ):
            raise RuntimeError("Episode does not contain the verified clear capture settings")
