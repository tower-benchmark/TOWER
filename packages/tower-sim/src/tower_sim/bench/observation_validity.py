"""Separate synchronized RGB delivery from physical scene visibility."""

from __future__ import annotations

import hashlib
import math
from typing import Any

import numpy as np


class DeliveryGuard:
    def __init__(self):
        self.previous_step = None
        self.previous_time = None

    def validate(
        self,
        pixels,
        stamp,
        quality,
        *,
        render_step,
        synchronized,
        paused,
        delta_time_s,
        independent_reference=None,
    ):
        pixels = np.asarray(pixels)
        result: dict[str, Any] = dict(
            schema="tower-v3-observation-delivery-v1",
            valid=False,
            time_s=stamp,
            render_step=render_step,
            synchronized=synchronized,
            paused=paused,
            delta_time_s=delta_time_s,
            photometric_visibility=len(quality) == 4
            and all(q.get("valid") is True for q in quality.values()),
        )
        if pixels.shape != (4, 240, 320, 3) or pixels.dtype != np.uint8:
            return dict(result, reason="Invalid four-camera RGB format")
        result["pixels_sha256"] = hashlib.sha256(pixels.tobytes()).hexdigest()
        if (
            type(render_step) is not int
            or render_step <= 0
            or (self.previous_step is not None and render_step <= self.previous_step)
            or synchronized is not True
            or paused is not True
            or delta_time_s != 0
        ):
            return dict(result, reason="Rendering completion/freshness is not established")
        if (
            type(stamp) not in (int, float)
            or not math.isfinite(stamp)
            or stamp < 0
            or (self.previous_time is not None and stamp <= self.previous_time)
        ):
            return dict(result, reason="Observation time is not finite and strictly increasing")
        self.previous_step, self.previous_time = render_step, stamp
        if not result["photometric_visibility"]:
            if independent_reference is None:
                return dict(
                    result, reason="Low visibility requires an independent same-state source render"
                )
            try:
                reference, evidence = independent_reference()
                comparison = compare_pixels(pixels, reference)
                result["independent_reference"] = evidence
                result["comparison"] = comparison
                result["reference_pixels_sha256"] = hashlib.sha256(reference.tobytes()).hexdigest()
                if evidence.get("physical_state_preserved") is not True or not comparison["passed"]:
                    return dict(
                        result,
                        reason="Independent source render does not establish physical low visibility",
                    )
            except Exception as exc:  # noqa: BLE001 -- callback failure is a delivery rejection
                return dict(
                    result, reason=f"Independent render failed: {type(exc).__name__}: {exc}"
                )
        return dict(result, valid=True, reason="Delivered frame; visibility recorded separately")


def compare_pixels(actual, reference):
    actual, reference = np.asarray(actual), np.asarray(reference)
    if (
        actual.shape != (4, 240, 320, 3)
        or reference.shape != actual.shape
        or actual.dtype != np.uint8
        or reference.dtype != np.uint8
    ):
        raise ValueError("Source renderer requires four uint8 RGB camera images")
    difference = actual.astype(float) - reference
    mae = np.abs(difference).mean(axis=(1, 2, 3))
    bias = np.abs(difference.mean(axis=(1, 2)))
    return dict(
        passed=bool(np.max(mae) <= 8 and np.max(bias) <= 6),
        per_camera_mae=mae.tolist(),
        absolute_channel_bias=bias.tolist(),
        maximum_mae=8.0,
        maximum_channel_bias=6.0,
        threshold_source="Native61 independent-reference image fidelity thresholds",
    )
