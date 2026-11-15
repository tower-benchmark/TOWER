# SPDX-License-Identifier: Apache-2.0
"""Versioned TCP fit identities, independent of module layout and release bytes.

See docs/reference/tcp-calibration.md for the algorithm and trust boundary.
No raw demonstration data or native simulator imports are needed for cache use.
"""

import copy
import json
import re
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from ..artifacts import sha256

SCHEMA = "tower-recorded-tcp-v2"
FIT_ALGORITHM = "tower-tcp-median-rotation-mean-v1"
COORDINATES = "nero-base-link-gripper-base-to-eef-m-xyzw-rad-v1"
FIT_SELECTION = "first-source-with-split-fit-v1"
PROFILE_CHANNEL_SCOPE = "leader_qpos_observation_eef_gripper_timestamps"
FIT_CHANNEL_SCOPE = "tower-tcp-observation-channels-v1"


def _digest(value, name):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"Invalid TCP cache identity: {name} must be a SHA-256 digest")
    return value


def _source(profile):
    try:
        source = next(s for s in profile["sources"] if s["split"] == "fit")
        episode = source["episode_id"]
        channels = source["channel_sha256"]
        scope = profile["source_hash_scope"]
    except (KeyError, TypeError, StopIteration) as exc:
        raise ValueError(
            "Invalid TCP fit profile: first fit source and channel scope required"
        ) from exc
    if not isinstance(episode, str) or not episode:
        raise ValueError("Invalid TCP fit episode identity")
    _digest(channels, "source_channel_sha256")
    if scope != PROFILE_CHANNEL_SCOPE:
        raise ValueError("Unsupported TCP profile channel scope")
    return {"episode_id": episode, "profile_channel_sha256": channels}


def tcp_calibration_key(
    *,
    profile,
    stage_sha256,
    kinematics,
    fit_channel_sha256,
    kinematic_geometry_sha256=None,
):
    """Build the expected identity from independently verified runtime inputs.

    ``stage_sha256`` identifies the verified station; ``kinematics`` contains
    the actual nero.urdf/nero.yaml. The fit digest comes from an independently
    verified fit-input receipt (or a fresh channel hash), not the cache being
    validated. Geometry defaults to the serial chain in nero.urdf; supply a
    separate digest when the runtime uses a separate geometry artifact.
    """
    models = {name: sha256(Path(kinematics) / name) for name in ("nero.urdf", "nero.yaml")}
    return {
        "fit_algorithm": FIT_ALGORITHM,
        "coordinate_convention": COORDINATES,
        "stage_sha256": _digest(stage_sha256, "stage_sha256"),
        "kinematics_sha256": models,
        "kinematic_geometry_sha256": _digest(
            models["nero.urdf"] if kinematic_geometry_sha256 is None else kinematic_geometry_sha256,
            "kinematic_geometry_sha256",
        ),
        "fit_source": {
            **_source(profile),
            "fit_channel_sha256": _digest(fit_channel_sha256, "fit_channel_sha256"),
        },
        "profile_selection": {
            "rule": FIT_SELECTION,
            "profile_channel_scope": PROFILE_CHANNEL_SCOPE,
            "fit_channel_scope": FIT_CHANNEL_SCOPE,
        },
    }


def _validate_key(key, owner="TCP v2 requires an independently constructed semantic key"):
    if not isinstance(key, dict):
        raise ValueError(owner)
    try:
        if set(key) != {
            "fit_algorithm",
            "coordinate_convention",
            "stage_sha256",
            "kinematics_sha256",
            "kinematic_geometry_sha256",
            "fit_source",
            "profile_selection",
        }:
            raise ValueError("Invalid TCP v2 key fields")
        if key["fit_algorithm"] != FIT_ALGORITHM:
            raise ValueError("Unsupported TCP fit_algorithm")
        if key["coordinate_convention"] != COORDINATES:
            raise ValueError("Unsupported TCP coordinate_convention")
        if key["profile_selection"] != {
            "rule": FIT_SELECTION,
            "profile_channel_scope": PROFILE_CHANNEL_SCOPE,
            "fit_channel_scope": FIT_CHANNEL_SCOPE,
        }:
            raise ValueError("Unsupported TCP profile_selection")
        for name in ("stage_sha256", "kinematic_geometry_sha256"):
            _digest(key[name], name)
        if set(key["kinematics_sha256"]) != {"nero.urdf", "nero.yaml"}:
            raise ValueError("Invalid TCP kinematics_sha256 fields")
        for name, value in key["kinematics_sha256"].items():
            _digest(value, name)
        source = key["fit_source"]
        if set(source) != {"episode_id", "profile_channel_sha256", "fit_channel_sha256"}:
            raise ValueError("Invalid TCP fit_source fields")
        if not isinstance(source["episode_id"], str) or not source["episode_id"]:
            raise ValueError("Invalid TCP fit episode identity")
        for name in ("profile_channel_sha256", "fit_channel_sha256"):
            _digest(source[name], name)
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError("Malformed TCP v2 semantic key") from exc


def _report(cache, key):
    try:
        report = cache["tcp"]
        arrays = [
            np.asarray(report[name], dtype=float)
            for name in (
                "translation_m",
                "rotation_xyzw",
                "position_residual_p50_p95_m",
                "rotation_residual_p50_p95_deg",
            )
        ]
        if any(a.shape != shape for a, shape in zip(arrays, ((3,), (4,), (2,), (2,)))):
            raise ValueError("Invalid recorded TCP calibration fit")
        offset, quaternion, position, rotation = arrays
        if (
            not all(np.isfinite(a).all() for a in arrays)
            or not np.isclose(np.linalg.norm(quaternion), 1.0, atol=1e-6, rtol=0)
            or not 0 <= position[0] <= position[1] <= 0.005
            or not 0 <= rotation[0] <= rotation[1] <= 3.0
        ):
            raise ValueError("Invalid recorded TCP calibration fit")
        if (
            report["fit_episode_id"] != key["fit_source"]["episode_id"]
            or report["source_channel_sha256"] != key["fit_source"]["profile_channel_sha256"]
        ):
            raise ValueError("TCP v2 report training source mismatch")
    except (KeyError, TypeError) as exc:
        raise ValueError("Malformed recorded TCP calibration fit") from exc
    return copy.deepcopy(report), offset, quaternion


@contextmanager
def cached_tcp_calibration_v2(model_class, cache_path, *, key, kinematics=None):
    """Patch the supplied class object, including its aliases; restore on exit.

    Pass ``kinematics``, the directory the model actually loads, when it is
    written during construction: as in v1, its nero.urdf/nero.yaml are then
    re-hashed against the key when the fit is restored. Like v1 this is a
    process-wide constructor context, not a thread-local patch. Serial/nested
    contexts are supported; overlapping threads are not.
    """
    cache = json.loads(Path(cache_path).read_text())
    if cache.get("schema") != SCHEMA:
        raise ValueError(f"Expected {SCHEMA}, got {cache.get('schema')!r}")
    _validate_key(key)
    _validate_key(cache.get("key"), "TCP v2 cache has no semantic key")
    expected = copy.deepcopy(key)
    for name, actual in expected.items():
        if cache["key"][name] != actual:
            raise ValueError(f"TCP v2 key {name} mismatch")
    report, offset, quaternion = _report(cache, expected)
    original = model_class.calibrate_tcp

    def restore(model, profile):
        actual_source = _source(profile)
        if any(expected["fit_source"][name] != value for name, value in actual_source.items()):
            raise ValueError("TCP cache training source mismatch")
        if kinematics is not None:
            for name, digest in expected["kinematics_sha256"].items():
                if sha256(Path(kinematics) / name) != digest:
                    raise ValueError(f"TCP cache kinematics mismatch: {name}")
        model.tcp_offset = offset.copy()
        model.tcp_rotation = Rotation.from_quat(quaternion)
        model.tcp_report = copy.deepcopy(report)

    model_class.calibrate_tcp = restore
    try:
        yield
    finally:
        model_class.calibrate_tcp = original
