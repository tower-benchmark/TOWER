"""Exact uint8 camera-quality fast path with instance-owned audit counters."""

import json

import numpy as np


def camera_quality(rgb):
    """Reject blank capture buffers before they become policy observations."""
    value = np.asarray(rgb)
    if value.shape != (240, 320, 3) or value.dtype != np.uint8:
        return dict(
            valid=False,
            shape=list(value.shape),
            dtype=str(value.dtype),
            reason="invalid RGB format",
        )
    spatial_std = value.astype(float).std(axis=(0, 1))
    dynamic_range = int(value.max()) - int(value.min())
    non_dark_fraction = float(np.mean(value.max(axis=2) > 8))
    valid = dynamic_range >= 16 and float(spatial_std.max()) >= 2.0 and non_dark_fraction >= 0.01
    return dict(
        valid=bool(valid),
        minimum=int(value.min()),
        maximum=int(value.max()),
        dynamic_range=dynamic_range,
        channel_spatial_std=spatial_std.tolist(),
        non_dark_fraction=non_dark_fraction,
        rule="range>=16, spatial_std>=2 in at least one channel, >=1% pixels above8",
    )


def _fast(value):
    levels = np.arange(256, dtype=np.float64)
    n = 240 * 320
    flat = value.reshape(n, 3)
    columns = np.ascontiguousarray(flat.T)
    std = np.empty(3)
    for c in range(3):
        channel = columns[c]
        counts = np.bincount(channel, minlength=256)
        mean = np.float64(int(counts @ np.arange(256, dtype=np.int64))) / np.float64(n)
        deviation = levels - mean
        square = deviation * deviation
        total = np.add.accumulate(square[channel])[-1]
        std[c] = np.sqrt(total / np.float64(n))
    bright = int(np.count_nonzero(np.maximum(np.maximum(columns[0], columns[1]), columns[2]) > 8))
    maximum, minimum = int(value.max()), int(value.min())
    non_dark_fraction = float(np.float64(bright) / n)
    valid = maximum - minimum >= 16 and float(std.max()) >= 2.0 and non_dark_fraction >= 0.01
    return dict(
        valid=bool(valid),
        minimum=minimum,
        maximum=maximum,
        dynamic_range=maximum - minimum,
        channel_spatial_std=std.tolist(),
        non_dark_fraction=non_dark_fraction,
        rule="range>=16, spatial_std>=2 in at least one channel, >=1% pixels above8",
    )


class CameraQuality:
    def __init__(self, *, audit_every=256):
        if type(audit_every) is not int or audit_every < 1:
            raise ValueError("audit_every must be a positive integer")
        self.audit_every = audit_every
        self.stats = dict(calls=0, fast=0, fallback=0, audited=0)

    def __call__(self, rgb):
        self.stats["calls"] += 1
        value = np.asarray(rgb)
        if value.shape != (240, 320, 3) or value.dtype != np.uint8 or not value.flags.c_contiguous:
            self.stats["fallback"] += 1
            return camera_quality(rgb)
        result = _fast(value)
        self.stats["fast"] += 1
        if self.stats["fast"] % self.audit_every == 0:
            self.stats["audited"] += 1
            if json.dumps(camera_quality(rgb)) != json.dumps(result):
                raise RuntimeError("Fast camera_quality differs from the original")
        return result
