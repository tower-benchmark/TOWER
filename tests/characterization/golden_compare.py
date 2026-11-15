"""Portable geometric floats; exact JSON types, decisions and discrete leaves.

The 1e-12 tolerances absorb CPU/SIMD accumulation drift, not scorer decisions.
Frozen scoring/offline/trajectory.py::Rules sets position_m=.01,
stable_translation_m=support_com_margin_m=.002, slot_clearance_m=.0005,
and disturbance_motion_m=.00025. The last is the smallest positive distance
threshold in Rules: 2.5e8 times the absolute tolerance. Numerical
implementation epsilons and zero-boundary predicates are not task tolerances;
the resulting bool/int decisions must still agree exactly, even at a boundary.
"""

from __future__ import annotations

import json
import math
import re
import struct
from dataclasses import dataclass
from typing import Any

FLOAT_REL_TOL = 1e-12
FLOAT_ABS_TOL = 1e-12
# Audited serialized hull output: task4_final_state.table_supported only.
HULL_POINTER_PATTERNS = (r"/final_state/targets/[0-9]+/table_support/contact_polygon_m",)


@dataclass
class ComparisonStats:
    differences: int = 0
    max_abs_delta: float = 0.0
    max_rel_delta: float = 0.0


STATS = ComparisonStats()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def _compare(expected: Any, actual: Any, portable: bool, pointer: str, events: list) -> None:
    _require(type(expected) is type(actual), f"JSON type differs at {pointer}")
    if isinstance(expected, float):
        # Bit equality also handles matching infinities/NaN payloads. No
        # non-finite value may enter the tolerance branch.
        if struct.pack("!d", expected) == struct.pack("!d", actual):
            return
        _require(math.isfinite(expected) and math.isfinite(actual), pointer)
        if expected == actual == 0.0:
            raise AssertionError(f"Zero sign differs at {pointer}")
        if portable and math.isclose(
            expected, actual, rel_tol=FLOAT_REL_TOL, abs_tol=FLOAT_ABS_TOL
        ):
            absolute = abs(expected - actual)
            relative = absolute / max(abs(expected), abs(actual))
            events.append((pointer, expected, actual, absolute, relative))
            return
        raise AssertionError(f"JSON float differs at {pointer}: {expected!r} != {actual!r}")
    if isinstance(expected, dict):
        _require(all(type(key) is str for key in (*expected, *actual)), pointer)
        _require(expected.keys() == actual.keys(), f"JSON keys differ at {pointer}")
        for key in expected:
            escaped = key.replace("~", "~0").replace("/", "~1")
            _compare(expected[key], actual[key], portable, f"{pointer}/{escaped}", events)
    elif isinstance(expected, list):
        _require(len(expected) == len(actual), f"JSON length differs at {pointer}")
        if portable and any(re.fullmatch(pattern, pointer) for pattern in HULL_POINTER_PATTERNS):
            _require(len(expected) >= 3, pointer)
            for shift in range(len(expected)):
                candidate: list = []
                try:
                    for index, right in enumerate(actual):
                        _compare(
                            expected[(index + shift) % len(expected)],
                            right,
                            portable,
                            f"{pointer}/{index}",
                            candidate,
                        )
                except AssertionError:
                    continue
                if shift:
                    events.append((pointer, shift))
                events.extend(candidate)
                return
            raise AssertionError(
                f"Hull is not a cyclic rotation within float tolerances: {pointer}"
            )
        for index, (left, right) in enumerate(zip(expected, actual, strict=True)):
            _compare(left, right, portable, f"{pointer}/{index}", events)
    else:
        _require(expected == actual, f"JSON differs at {pointer}: {expected!r} != {actual!r}")


def assert_equivalent(
    expected: Any, actual: Any, *, portable: bool = True, pointer: str = ""
) -> None:
    """Compare recursively; publish only differences from a successful comparison.

    Relative delta is abs(a-b)/max(abs(a),abs(b)), matching math.isclose's
    symmetric scaling. Signed zeros remain distinct when both values are zero.
    Exact mode retains authoritative canonical-JSON byte equality.
    """
    if not portable:
        if canonical(expected) == canonical(actual):
            return
        _compare(expected, actual, False, pointer, [])
        raise AssertionError(f"Canonical JSON differs at {pointer}")
    events: list = []
    _compare(expected, actual, True, pointer, events)
    for event in events:
        if len(event) == 2:
            path, shift = event
            print(f"INFO: accepted hull rotation at {path}: shift={shift}")
        else:
            path, left, right, absolute, relative = event
            STATS.differences += 1
            STATS.max_abs_delta = max(STATS.max_abs_delta, absolute)
            STATS.max_rel_delta = max(STATS.max_rel_delta, relative)
            print(
                f"INFO: accepted float difference at {path}: expected={left!r} actual={right!r} "
                f"abs_delta={absolute:.17g} rel_delta={relative:.17g}"
            )


def summary() -> str:
    return (
        f"Golden float differences: count={STATS.differences} "
        f"max_abs_delta={STATS.max_abs_delta:.17g} max_rel_delta={STATS.max_rel_delta:.17g}"
    )
