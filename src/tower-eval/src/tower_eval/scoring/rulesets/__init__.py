# SPDX-License-Identifier: Apache-2.0
"""Explicit version selection without a second scorer tree."""

from dataclasses import asdict

from . import bench_2026_11, paper_v1

RULESETS = {"paper-v1": paper_v1.Rules, "bench-2026.11": bench_2026_11.Rules}


def select(name, overrides=None):
    if name not in RULESETS:
        raise ValueError(f"Unknown scoring ruleset: {name!r}")
    return RULESETS[name](**({} if overrides is None else asdict(overrides)))
