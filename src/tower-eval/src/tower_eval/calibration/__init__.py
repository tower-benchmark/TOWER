# SPDX-License-Identifier: Apache-2.0
"""Explicit reuse of a recorded TCP fit when its original source is offline."""

import copy
import json
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from ..artifacts import sha256
from .semantic import cached_tcp_calibration_v2, tcp_calibration_key

__all__ = ["cached_tcp_calibration", "cached_tcp_calibration_v2", "tcp_calibration_key"]


@contextmanager
def cached_tcp_calibration(
    model_class,
    cache_path,
    stage_hash=None,
    profile_path=None,
    kinematics=None,
    backend_path=None,
    *,
    key=None,
):
    """Dispatch by schema; v2 callers supply a semantic key, never backend bytes."""
    if cache_path is None:
        yield
        return
    cache = json.loads(Path(cache_path).read_text())
    schema = cache.get("schema")
    if schema == "tower-recorded-tcp-v2":
        if stage_hash is not None or profile_path is not None or backend_path is not None:
            raise ValueError("TCP v2 is checked by its semantic key; pass key= only")
        with cached_tcp_calibration_v2(model_class, cache_path, key=key, kinematics=kinematics):
            yield
    elif schema == "tower-recorded-tcp-v1":
        if key is not None:
            raise ValueError("TCP v1 caches do not take a semantic key")
        with _cached_tcp_calibration_v1(
            model_class, cache_path, stage_hash, profile_path, kinematics, backend_path
        ):
            yield
    else:
        raise ValueError(f"Unsupported TCP calibration cache schema: {schema!r}")


@contextmanager
def _cached_tcp_calibration_v1(
    model_class, cache_path, stage_hash, profile_path, kinematics, backend_path
):
    """Restore a verified fit for this constructor only; never silently fall back."""
    if cache_path is None:
        yield
        return
    cache = json.loads(Path(cache_path).read_text())
    if cache.get("schema") != "tower-recorded-tcp-v1":
        raise ValueError("Unsupported TCP calibration cache")
    for key, actual in [
        ("usd_sha256", stage_hash),
        ("profile_sha256", sha256(profile_path)),
        ("backend_sha256", sha256(backend_path)),
    ]:
        if cache.get(key) != actual:
            raise ValueError(f"TCP cache {key} mismatch")
    report = cache["tcp"]
    offset = np.asarray(report["translation_m"], dtype=float)
    quaternion = np.asarray(report["rotation_xyzw"], dtype=float)
    position_residual = np.asarray(report["position_residual_p50_p95_m"], dtype=float)
    rotation_residual = np.asarray(report["rotation_residual_p50_p95_deg"], dtype=float)
    if (
        offset.shape != (3,)
        or quaternion.shape != (4,)
        or position_residual.shape != (2,)
        or rotation_residual.shape != (2,)
        or not all(
            np.isfinite(x).all() for x in [offset, quaternion, position_residual, rotation_residual]
        )
        or not np.isclose(np.linalg.norm(quaternion), 1.0, atol=1e-6, rtol=0)
        or np.any(position_residual < 0)
        or np.any(rotation_residual < 0)
        or position_residual[1] > 0.005
        or rotation_residual[1] > 3.0
    ):
        raise ValueError("Invalid recorded TCP calibration fit")
    original = model_class.calibrate_tcp

    def restore(model, profile):
        source = next(s for s in profile["sources"] if s["split"] == "fit")
        if (
            source["episode_id"] != report["fit_episode_id"]
            or source["channel_sha256"] != report["source_channel_sha256"]
        ):
            raise ValueError("TCP cache training source mismatch")
        for name in ["nero.urdf", "nero.yaml"]:
            if sha256(Path(kinematics) / name) != cache["kinematics_sha256"][name]:
                raise ValueError(f"TCP cache kinematics mismatch: {name}")
        model.tcp_offset = offset.copy()
        model.tcp_rotation = Rotation.from_quat(quaternion)
        model.tcp_report = copy.deepcopy(report)

    model_class.calibrate_tcp = restore
    try:
        yield
    finally:
        model_class.calibrate_tcp = original
