# SPDX-License-Identifier: Apache-2.0
"""Repeated geometry may reuse a decision, never eligibility or changed poses."""

from types import SimpleNamespace

import numpy as np
import pytest
from tower_eval.scoring.detectors import place
from tower_eval.scoring.rulesets.bench_2026_11 import DEFAULT_RULES

DIMENSIONS = [[0.1, 0.1, 0.02]] * 3


def recording(dtype=np.float64):
    poses = np.zeros((6, 3, 7), dtype=dtype)
    poses[..., -1] = 1
    poses[:, 0, 2] = 0.03
    poses[:, 1:, 2] = 0.01
    poses[:, 2, 0] = 0.5
    return SimpleNamespace(
        times=np.arange(6) * 0.05,
        poses=poses,
        velocities=np.zeros((6, 3, 6), dtype=dtype),
    )


def support(r, candidate=None, *, supporters=(1,), dimensions=DIMENSIONS, rules=DEFAULT_RULES):
    if candidate is None:
        candidate = np.ones(len(r.times), dtype=bool)
    return place.support_mask(r, 0, supporters, dimensions, candidate, rules)


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("moving_block", [0, 1])
@pytest.mark.parametrize("movement", ["translation", "rotation"])
@pytest.mark.parametrize("layout", ["contiguous", "strided", "reversed"])
def test_changed_target_or_supporter_matches_uncached_samples(
    dtype, moving_block, movement, layout
):
    r = recording(dtype)
    # Supported, unsupported, supported again: neither false nor true decisions
    # may leak into a different configuration, even with an unchanged target.
    if movement == "translation":
        r.poses[2:4, moving_block, 0] += 0.3
    else:
        r.poses[2:4, moving_block, 3:] = [0, np.sqrt(0.5), 0, np.sqrt(0.5)]
    if layout == "strided":
        # Keep the same logical values in a view with non-contiguous storage.
        r.poses = np.repeat(r.poses, 2, axis=0)[::2]
        r.velocities = np.repeat(r.velocities, 2, axis=0)[::2]
    elif layout == "reversed":
        r.poses = r.poses[::-1]
        r.velocities = r.velocities[::-1]
    expected = []
    for k in range(len(r.times)):
        single = SimpleNamespace(
            times=r.times[k : k + 1],
            poses=r.poses[k : k + 1],
            velocities=r.velocities[k : k + 1],
        )
        # A one-sample call cannot reuse a prior decision.
        expected.append(support(single)[0])
    np.testing.assert_array_equal(expected, [True, True, False, False, True, True])
    np.testing.assert_array_equal(support(r), expected)


def test_every_supporter_participates_in_the_reuse_key():
    r = recording()
    r.poses[:, 1, 0] = 0.5
    r.poses[:, 2, 0] = 0.0
    r.poses[2:4, 2, 0] = 0.5
    np.testing.assert_array_equal(
        support(r, supporters=(1, 2)), [True, True, False, False, True, True]
    )


def test_repeated_poses_still_check_candidate_and_both_velocity_limits(monkeypatch):
    r = recording()
    r.velocities[1, 0, 0] = DEFAULT_RULES.linear_velocity_m_s * 2
    r.velocities[3, 0, 3] = DEFAULT_RULES.angular_velocity_rad_s * 2
    candidate = np.array([True, True, False, True, True, True])
    calls = []
    original = place.block_geometry

    def counted(*args):
        calls.append(1)
        return original(*args)

    monkeypatch.setattr(place, "block_geometry", counted)
    np.testing.assert_array_equal(support(r, candidate), [True, False, False, False, True, True])
    assert len(calls) == 2  # One target and supporter computation, all identical poses.


def test_reuse_is_exact_and_does_not_round_poses(monkeypatch):
    r = recording()
    r.poses[2:4, 1, 0] = np.nextafter(0.0, 1.0)
    r.poses[4:, 1, 0] = -0.0
    calls = []
    original = place.block_geometry

    def counted(*args):
        calls.append(1)
        return original(*args)

    monkeypatch.setattr(place, "block_geometry", counted)
    assert support(r).all()
    assert len(calls) == 6  # Recompute for a one-bit change and for signed zero.


def test_supporter_iterator_is_snapshotted_for_every_sample():
    # Unlike the old exhausted-iterator behavior, an iterator names a fixed set
    # of supporters for the entire call, exactly like the production list input.
    np.testing.assert_array_equal(support(recording(), supporters=iter([1])), [True] * 6)


def test_empty_and_ineligible_samples_do_not_evaluate_geometry(monkeypatch):
    def unexpected(*args):
        pytest.fail("Ineligible samples must not evaluate support geometry")

    monkeypatch.setattr(place, "block_geometry", unexpected)
    r = recording()
    assert not support(r, np.zeros(len(r.times), dtype=bool)).any()
    r.times, r.poses, r.velocities = r.times[:0], r.poses[:0], r.velocities[:0]
    result = support(r)
    assert result.dtype == np.bool_ and result.shape == (0,)


def test_rules_dimensions_and_supporters_are_not_reused_across_calls():
    from dataclasses import replace

    r = recording()
    assert support(r).all()
    assert not support(r, supporters=(2,)).any()
    assert not support(r, supporters=()).any()
    assert not support(r, dimensions=[[0.1, 0.1, 0.1], *DIMENSIONS[1:]]).any()
    assert not support(r, rules=replace(DEFAULT_RULES, minimum_support_fraction=1.01)).any()
    assert support(r).all()
