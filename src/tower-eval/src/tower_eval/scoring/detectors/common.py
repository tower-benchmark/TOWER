# SPDX-License-Identifier: Apache-2.0
"""Shared temporal mask primitives."""

from __future__ import annotations

import numpy as np


def spans(mask):
    edges = np.diff(np.r_[False, mask, False].astype(int))
    return [
        (int(a), int(b)) for a, b in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))
    ]


def sustained(mask, times, seconds):
    result = np.zeros(len(times), dtype=bool)
    for a, b in spans(mask):
        result[a:b] = times[a:b] - times[a] >= seconds - 1e-8
    return result


def first(mask):
    found = np.flatnonzero(mask)
    return int(found[0]) if len(found) else None
