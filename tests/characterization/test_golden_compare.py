import math

import pytest

from tests.characterization import golden_compare as comparison
from tests.characterization.golden_compare import assert_equivalent

POINTER = "/final_state/targets/2/table_support/contact_polygon_m"
POLYGON = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]


@pytest.fixture(autouse=True)
def isolate_synthetic_deltas(monkeypatch: pytest.MonkeyPatch) -> None:
    # Session statistics describe real characterizations, not boundary probes.
    monkeypatch.setattr(comparison, "STATS", comparison.ComparisonStats())


def test_rotation_is_logged_and_exact_tier_rejects_it(capsys: pytest.CaptureFixture[str]) -> None:
    rotated = POLYGON[1:] + POLYGON[:1]
    assert_equivalent(POLYGON, rotated, pointer=POINTER)
    assert f"INFO: accepted hull rotation at {POINTER}" in capsys.readouterr().out
    with pytest.raises(AssertionError):
        assert_equivalent(POLYGON, rotated, portable=False, pointer=POINTER)


@pytest.mark.parametrize(
    "expected,actual",
    [
        ({"s": 1}, {"s": 2}),
        ({"passed": True}, {"passed": False}),
        (1.0, 1.0 + 1e-9),
        ([1, 2], [1]),
        (1, 1.0),
        (True, 1),
        ("success", "failure"),
        (None, False),
        ({"s": 1}, {"g": 1}),
        (0.0, -0.0),
        (float("inf"), float("-inf")),
        (float("nan"), 1.0),
    ],
)
def test_portable_contract_rejects_changes(expected: object, actual: object) -> None:
    with pytest.raises(AssertionError):
        assert_equivalent(expected, actual)


@pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan"), -0.0])
def test_identical_nonfinite_values_and_zero_sign(value: float) -> None:
    assert_equivalent(value, value)


def test_float_tolerance_logs_pointer_values_and_deltas(capsys: pytest.CaptureFixture[str]) -> None:
    actual = math.nextafter(0.25, 1.0)
    assert_equivalent({"distance": 0.25}, {"distance": actual})
    output = capsys.readouterr().out
    assert "/distance" in output and "expected=0.25" in output and f"actual={actual!r}" in output
    assert "abs_delta=" in output and "rel_delta=" in output
    assert comparison.STATS.differences == 1
    assert comparison.STATS.max_abs_delta == actual - 0.25
    assert "max_abs_delta=" in comparison.summary() and "max_rel_delta=" in comparison.summary()
    with pytest.raises(AssertionError):
        assert_equivalent(0.25, actual, portable=False)


def test_hull_rotation_uses_the_same_float_rule(capsys: pytest.CaptureFixture[str]) -> None:
    rotated = [[1.0 + 1e-13, 0.0], *POLYGON[2:], POLYGON[0]]
    assert_equivalent(POLYGON, rotated, pointer=POINTER)
    output = capsys.readouterr().out
    assert "accepted hull rotation" in output and "accepted float difference" in output
    assert comparison.STATS.differences == 1  # Rejected speculative rotations do not log.


@pytest.mark.parametrize(
    "changed",
    [
        POLYGON[::-1],
        [POLYGON[0], POLYGON[2], POLYGON[1], POLYGON[3]],
        [[-0.0, 0.0], *POLYGON[1:]],
        [[1e-9, 0.0], *POLYGON[1:]],
        POLYGON + [POLYGON[0]],
        [[0, 0], *POLYGON[1:]],
    ],
)
def test_hull_rule_preserves_types_orientation_and_length(changed: list[list[float]]) -> None:
    with pytest.raises(AssertionError):
        assert_equivalent(POLYGON, changed, pointer=POINTER)


@pytest.mark.parametrize(
    "pointer",
    [
        "/scores/s",
        "/final_state/targets/2/table_support/com_margin_m",
        "/unlisted/contact_polygon_m",
    ],
)
def test_exception_is_not_a_suffix_wildcard(pointer: str) -> None:
    with pytest.raises(AssertionError):
        assert_equivalent(POLYGON, POLYGON[1:] + POLYGON[:1], pointer=pointer)
