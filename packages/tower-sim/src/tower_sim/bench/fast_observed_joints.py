"""Exact-decision fast path for control.validate_observed_joints; every 400 Hz tick is still checked.

The original check converts measured physical joints to encoder coordinates with six fixed-point
iterations of the J2 gravity compliance, then tests the encoder joints against the kill line. Only J2
changes in that conversion, and the change is |k2 * tau2| where tau2 is a central difference of the
gravity potential. By the mean value theorem that difference equals the true derivative at some
nearby q, so |tau2| <= sum_n m_n g R_n, with R_n an upper bound on the distance from link n's COM to
the J2 axis (sum of URDF origin/travel norms from link2 plus |com_n|).

The fast path returns "pass" only when every original check is guaranteed to pass:
finite arms, physical URDF bounds, finite bounded gripper widths, non-J2 kill line (identical
comparisons on identical floats), and physical J2 inside the kill line by more than the J2 bound.
Any other input goes to the original function, so it raises the same error at the same tick.
The original return value (encoder joints) is unused by batch_policy_worker; the fast path returns None.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from . import control
from .options import DEFAULT_CONTROL_OPTIONS, ControlOptions

G = 9.81
WIDTH_RANGE = (-0.01, 0.12)  # widths outside this (or non-finite) always use the original
ABSOLUTE_MARGIN_RAD = 1e-6  # floating-point cushion, >>1e-11 rad estimated rounding error
RELATIVE_INFLATION = 1.01
ARM_IDS = np.r_[np.arange(7), np.arange(8, 15)]
ARMS = ("left", "back")


@dataclass
class ObservationValidator:
    """Callable plus per-instance audit counters/certificate for the runner's receipt."""

    validate: Callable
    stats: dict[str, int]
    certificate: dict[str, Any]

    def __call__(self, physical, compensation, limits):
        return self.validate(physical, compensation, limits)


def tau2_bound(compensation, width_range=WIDTH_RANGE):
    fk = compensation.urdf_fk
    joints = fk.JOINTS
    j2 = joints["joint2"]
    if j2["kind"] != "revolute" or j2["mimic"] is not None:
        raise ValueError("Expected revolute J2")
    children = {}
    for name, j in joints.items():
        children.setdefault(j["parent"], []).append(name)
    reach = {j2["child"]: 0.0}
    stack = [j2["child"]]
    while stack:
        link = stack.pop()
        for name in children.get(link, []):
            j = joints[name]
            travel = 0.0
            if j["kind"] == "prismatic":
                axis = float(np.linalg.norm(j["axis"]))
                if j["mimic"]:
                    src, mult, off = j["mimic"]
                    if src != "gripper":
                        raise ValueError("Expected gripper mimic")
                    travel = axis * max(abs(mult * w + off) for w in width_range)
                elif name == "gripper":
                    travel = axis * max(abs(w) for w in width_range)
            elif j["kind"] not in ("revolute", "fixed"):
                raise ValueError("Unsupported joint kind " + j["kind"])
            if j["child"] in reach:
                raise ValueError("URDF is not a tree")
            reach[j["child"]] = reach[link] + float(np.linalg.norm(j["xyz"])) + travel
            stack.append(j["child"])
    distal = {fk.SIM_NAME.get(k, k): v for k, v in reach.items()}
    rows = {
        n: v["mass_kg"] * G * (distal[n] + float(np.linalg.norm(v["com_m"])))
        for n, v in compensation.MASSES.items()
        if n in distal
    }
    proximal = sorted(n for n in compensation.MASSES if n not in distal)
    return sum(rows.values()), rows, proximal


def observed_joint_validator(compensation, *, options: ControlOptions = DEFAULT_CONTROL_OPTIONS):
    """mode='fast': skip the conversion when certified, but re-run the original on every audit_every-th
    certified call and stop if it disagrees. mode='shadow': run both on every call, return the original."""
    mode, audit_every = options.observed_joints, options.observation_audit_every
    if mode == "off":
        return control.validate_observed_joints
    original = control.validate_observed_joints
    tol = control.OBSERVED_TRACKING_TOLERANCE_RAD
    kill_lo = control.KILL_LOWER - tol
    kill_hi = control.KILL_UPPER + tol
    bound, rows, proximal = tau2_bound(compensation)
    delta = {
        arm: abs(float(compensation.K2[arm])) * bound * RELATIVE_INFLATION + ABSOLUTE_MARGIN_RAD
        for arm in ARMS
    }
    j2_window = {
        arm: (float(kill_lo[1]) + delta[arm], float(kill_hi[1]) - delta[arm]) for arm in ARMS
    }
    if not all(lo < hi for lo, hi in j2_window.values()):
        raise ValueError("Empty certified J2 window")
    non_j2 = np.array([0, 2, 3, 4, 5, 6])
    stats = dict(calls=0, certified=0, fallback=0, audited=0)

    def certified(physical, limits):
        x = np.asarray(physical, float)
        if x.shape != (16,):
            return False
        arms = x[ARM_IDS]
        if not np.isfinite(arms).all():
            return False
        bounds = np.asarray(limits, float)
        if np.any(arms < bounds[ARM_IDS, 0] - tol) or np.any(arms > bounds[ARM_IDS, 1] + tol):
            return False
        for a, arm in enumerate(ARMS):
            w = x[a * 8 + 7]
            if not (np.isfinite(w) and WIDTH_RANGE[0] <= w <= WIDTH_RANGE[1]):
                return False
            q = x[a * 8 : a * 8 + 7]
            if np.any(q[non_j2] < kill_lo[non_j2]) or np.any(q[non_j2] > kill_hi[non_j2]):
                return False
            lo, hi = j2_window[arm]
            if not lo <= q[1] <= hi:
                return False
        return True

    def fast(physical, comp, limits):
        stats["calls"] += 1
        if comp is compensation and certified(physical, limits):
            stats["certified"] += 1
            if stats["certified"] % audit_every == 0:
                stats["audited"] += 1
                try:
                    original(physical, comp, limits)
                except control.PolicyActionError as exc:
                    raise RuntimeError("Fast path certified a state the original rejected") from exc
            return None
        stats["fallback"] += 1
        return original(physical, comp, limits)

    def shadow(physical, comp, limits):
        stats["calls"] += 1
        claim = comp is compensation and certified(physical, limits)
        try:
            result = original(physical, comp, limits)
        except control.PolicyActionError as exc:
            if claim:
                raise RuntimeError("Fast path certified a state the original rejected") from exc
            stats["fallback"] += 1
            raise
        stats["certified" if claim else "fallback"] += 1
        return result

    certificate = dict(
        mode=mode,
        tau2_bound_Nm=bound,
        per_link_bound_Nm=rows,
        proximal_links_zero_derivative=proximal,
        j2_delta_bound_rad=delta,
        certified_physical_j2_window_rad=j2_window,
        width_range_m=WIDTH_RANGE,
        absolute_margin_rad=ABSOLUTE_MARGIN_RAD,
        relative_inflation=RELATIVE_INFLATION,
        check_frequency_unchanged=True,
        fallback_is_original_function=True,
        audit_every=audit_every if mode == "fast" else 1,
    )
    return ObservationValidator(fast if mode == "fast" else shadow, stats, certificate)
